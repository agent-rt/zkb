//! Memory and facts: the parts where being wrong is silent.
//!
//! A wrong current-value is the worst failure mode in the whole system — it does
//! not look like an error, it looks like an answer.

const std = @import("std");
const zkb = @import("zkb");

const testing = std.testing;
const gpa = testing.allocator;

// ---------------------------------------------------------------------------
// facts: current value and history
// ---------------------------------------------------------------------------

const salary_csv =
    \\key,value,at,recorded_at,src,note
    \\birth_date,1990-03-15,2020-01-01,2026-08-17,user,
    \\salary,450000,2025-04-01,2025-04-02,user,月薪
    \\salary,480000,2026-04-01,2026-08-17,user,调薪后
    \\
;

/// `parse` is the layer `currentAll` sits on; testing it directly keeps the
/// file-reading wrapper thin enough not to need its own fixture.
fn currentOf(source: []const u8, key: []const u8) !struct { value: []u8, at: []u8 } {
    var parsed = try zkb.facts.parse(gpa, source);
    defer parsed.deinit(gpa);

    var best: ?zkb.facts.Fact = null;
    for (parsed.facts) |f| {
        if (!std.mem.eql(u8, f.key, key)) continue;
        if (best == null or std.mem.order(u8, f.at, best.?.at) != .lt) best = f;
    }
    const b = best orelse return error.NotFound;
    return .{ .value = try gpa.dupe(u8, b.value_txt), .at = try gpa.dupe(u8, b.at) };
}

test "the latest effective date wins, not the last line written" {
    // The append order and the effective order are different axes: a back-dated
    // correction appended today must not become the current value.
    const back_dated = salary_csv ++ "salary,400000,2024-01-01,2026-08-17,user,忘了记的旧数据\n";
    const c = try currentOf(back_dated, "salary");
    defer gpa.free(c.value);
    defer gpa.free(c.at);
    try testing.expectEqualStrings("480000", c.value);
    try testing.expectEqualStrings("2026-04-01", c.at);
}

test "a numeric value is parsed as a number and kept as text" {
    var parsed = try zkb.facts.parse(gpa, salary_csv);
    defer parsed.deinit(gpa);
    for (parsed.facts) |f| {
        if (!std.mem.eql(u8, f.key, "salary")) continue;
        try testing.expect(f.value_num != null);
        return;
    }
    return error.NotFound;
}

test "a non-numeric value survives intact rather than becoming null" {
    // A postcode with a leading zero, a version string, an ID: parseFloat fails
    // and the text is all there is. Losing it would be silent.
    var parsed = try zkb.facts.parse(gpa, "key,value,at\nzip,0123-4567,2026-01-01\n");
    defer parsed.deinit(gpa);
    try testing.expectEqualStrings("0123-4567", parsed.facts[0].value_txt);
    try testing.expectEqual(@as(?f64, null), parsed.facts[0].value_num);
}

test "the embedded text excludes the value" {
    // 450000 and 480000 are neighbours in a 1024-dim space, so embedding the
    // value adds noise and no signal — the value is always read from the row.
    const f: zkb.facts.Fact = .{
        .key = "salary",
        .value_num = 480000,
        .value_txt = "480000",
        .at = "2026-04-01",
        .recorded_at = "2026-08-17",
        .src = "user",
        .note = "调薪后",
    };
    const text = try zkb.facts.renderForEmbedding(gpa, f);
    defer gpa.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "480000") == null);
    try testing.expect(std.mem.indexOf(u8, text, "salary") != null);
    try testing.expect(std.mem.indexOf(u8, text, "调薪后") != null);
}

test "a missing required column is an error, not an empty result" {
    try testing.expectError(
        error.MissingColumn,
        zkb.facts.parse(gpa, "key,note\nsalary,no value column\n"),
    );
}

// ---------------------------------------------------------------------------
// memory frontmatter
// ---------------------------------------------------------------------------

test "frontmatter round-trips through render and parse" {
    const meta: zkb.memory.Meta = .{
        .type = .decision,
        .status = .active,
        .created = "2026-08-15",
        .source = "claude-code",
        .subjects = "retrieval,rrf",
        .refs = "doc:SPEC.md",
    };
    const rendered = try zkb.memory.render(gpa, meta, "  RRF fuses ranks, not scores.\n\n");
    defer gpa.free(rendered);

    // The body is trimmed but otherwise untouched.
    try testing.expect(std.mem.endsWith(u8, rendered, "RRF fuses ranks, not scores.\n"));

    const fm_end = std.mem.indexOfPos(u8, rendered, 4, "---").?;
    var back = try zkb.memory.parseMeta(gpa, rendered[4..fm_end]);
    defer back.deinit(gpa);

    try testing.expectEqual(zkb.memory.Type.decision, back.type);
    try testing.expectEqual(zkb.memory.Status.active, back.status);
    try testing.expectEqualStrings("2026-08-15", back.created);
    try testing.expectEqualStrings("retrieval,rrf", back.subjects);
    try testing.expectEqualStrings("doc:SPEC.md", back.refs);
}

test "a hand-written memory missing fields still parses" {
    // People will edit these files. Rejecting one for a missing `source` would
    // make the system feel like it owns the directory, which it does not.
    var m = try zkb.memory.parseMeta(gpa, "type: user\ncreated: 2026-08-15\n");
    defer m.deinit(gpa);
    try testing.expectEqual(zkb.memory.Type.user, m.type);
    try testing.expectEqual(zkb.memory.Status.active, m.status);
    try testing.expectEqualStrings("", m.source);
}

test "an unknown type falls back rather than failing the file" {
    var m = try zkb.memory.parseMeta(gpa, "type: nonsense\nstatus: weird\n");
    defer m.deinit(gpa);
    try testing.expectEqual(zkb.memory.Type.feedback, m.type);
    try testing.expectEqual(zkb.memory.Status.active, m.status);
}

test "subjects normalize from both list syntaxes" {
    var a = try zkb.memory.parseMeta(gpa, "subjects: [vcs, tooling]\n");
    defer a.deinit(gpa);
    var b = try zkb.memory.parseMeta(gpa, "subjects: vcs, tooling\n");
    defer b.deinit(gpa);
    try testing.expectEqualStrings("vcs,tooling", a.subjects);
    try testing.expectEqualStrings(a.subjects, b.subjects);
}

// ---------------------------------------------------------------------------
// slug
// ---------------------------------------------------------------------------

test "slug 取 ASCII 词与 CJK 字，两者混排时都保留" {
    // 这个测试原本断言「纯中文取不到可用内容，用日期当名字」——那是把 bug 写成了规格。
    // 一个自带 CJK 分词器的工具不该在取名这里把 CJK 当空白：一天记两条纯中文记忆，
    // 文件名就变成 2026-08-18 和 2026-08-18-2，谁都认不出哪条是哪条。
    const s1 = try zkb.memory.slug(gpa, "用户偏好使用 jj 管理 git 仓库", "2026-08-15");
    defer gpa.free(s1);
    try testing.expectEqualStrings("用户偏好使用-jj-管理-git-仓库", s1);

    // 纯中文要能取出名字，而不是退回日期。中文标点处收尾：名字不该断在半句话上。
    const s2 = try zkb.memory.slug(gpa, "禁止打补丁：不查根因就改症状", "2026-08-15");
    defer gpa.free(s2);
    try testing.expectEqualStrings("禁止打补丁", s2);

    // 日文同理。
    const s3 = try zkb.memory.slug(gpa, "障害報告書のテンプレート", "2026-08-15");
    defer gpa.free(s3);
    try testing.expectEqualStrings("障害報告書のテンプレート", s3);

    // 真的没有可用字符时才退回。
    const s4 = try zkb.memory.slug(gpa, "!!! ??? ...", "2026-08-15");
    defer gpa.free(s4);
    try testing.expectEqualStrings("2026-08-15", s4);
}

test "slug stays short on a long body" {
    const s = try zkb.memory.slug(gpa, "one two three four five six seven eight nine", "x");
    defer gpa.free(s);
    try testing.expectEqualStrings("one-two-three-four", s);
}

test "slug 不会在多字节字符中间截断" {
    // 上限按字符数算，不是字节数：从中间切开会写出非法 UTF-8 的文件名。
    const long = "根因分析与最小可证伪实验是排查问题的基本手段而不是可选项目再多写一些";
    const s = try zkb.memory.slug(gpa, long, "x");
    defer gpa.free(s);
    try testing.expect(std.unicode.utf8ValidateSlice(s));
    try testing.expect(s.len > 3);
}

// ---------------------------------------------------------------------------
// recall's ranking inputs
// ---------------------------------------------------------------------------

test "recency and relevance fuse without either dominating" {
    // The property that matters: a memory found by search but old still places,
    // and a memory that is merely new still places. Neither list wins outright.
    const relevance = [_]i64{ 10, 20, 30 };
    const recency = [_]i64{ 40, 30, 50 };

    const fused = try zkb.rrf.fuse(gpa, &relevance, &recency, .{ .fts_min_hits = 0 });
    defer gpa.free(fused);

    // 30 is in both lists, so it must come first even though it is last in one
    // and second in the other — cross-path agreement is the strongest signal.
    try testing.expectEqual(@as(i64, 30), fused[0].chunk_id);

    // Every id from both lists survives; a recency-only memory is not dropped.
    var seen: [6]bool = @splat(false);
    for (fused) |f| {
        for ([_]i64{ 10, 20, 30, 40, 50 }, 0..) |id, i| if (f.chunk_id == id) {
            seen[i] = true;
        };
    }
    for (seen[0..5]) |s| try testing.expect(s);
}

test "a single-entry recency list is not discarded as a sparse path" {
    // The fts_min_hits cutoff exists to protect against noisy BM25 on a small
    // vocabulary. Applied to recency it would mean "you have one memory, so
    // ignore it", which is exactly backwards.
    const fused = try zkb.rrf.fuse(gpa, &.{}, &.{99}, .{ .fts_min_hits = 0 });
    defer gpa.free(fused);
    try testing.expectEqual(@as(usize, 1), fused.len);
    try testing.expectEqual(@as(i64, 99), fused[0].chunk_id);
}

test "the two time axes are independent and both survive a round trip" {
    // `at` is when the salary changed; `recorded_at` is when it was written
    // down. Neither can be derived from the other, which is the whole reason
    // the second one is a column rather than a commit timestamp.
    var parsed = try zkb.facts.parse(gpa, salary_csv);
    defer parsed.deinit(gpa);

    for (parsed.facts) |f| {
        if (!std.mem.eql(u8, f.key, "salary")) continue;
        if (!std.mem.eql(u8, f.value_txt, "480000")) continue;
        try testing.expectEqualStrings("2026-04-01", f.at);
        try testing.expectEqualStrings("2026-08-17", f.recorded_at);
        return;
    }
    return error.NotFound;
}

test "a file predating the column still parses, with the axis simply absent" {
    // Absent, not invented: a made-up recording date would be worse than an
    // empty one, because nothing downstream could tell it was a guess.
    var parsed = try zkb.facts.parse(gpa, "key,value,at\nsalary,480000,2026-04-01\n");
    defer parsed.deinit(gpa);
    try testing.expectEqualStrings("2026-04-01", parsed.facts[0].at);
    try testing.expectEqualStrings("", parsed.facts[0].recorded_at);
}
