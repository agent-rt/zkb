//! Memory and facts: the parts where being wrong is silent.
//!
//! A wrong current-value is the worst failure mode in the whole system — it does
//! not look like an error, it looks like an answer.

const std = @import("std");
const zkb = @import("zkb");

const testing = std.testing;
const gpa = testing.allocator;
// 这些测试要写 rec_memory，需要一个真库和一个向量。store_test 里有同名 helper，
// 但那是另一个文件的私有函数——复制两行胜过为此导出。
const store = zkb.store;
const dim: usize = @intCast(zkb.schema.embedding_dim);
fn dummyVector(seed: u8) [dim]f32 {
    var v: [dim]f32 = @splat(0);
    v[@as(usize, seed) % dim] = 1.0;
    return v;
}


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
        .scope = "",
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

test "an unscoped fact is universal, a scoped one needs to be asked for" {
    const uni: zkb.facts.Current = .{
        .key = "wife.birthday", .value = "1993-07-23", .at = "1993-07-23",
        .recorded_at = "2026-08-17", .note = "", .scope = "",
    };
    const work: zkb.facts.Current = .{
        .key = "vpn.endpoint", .value = "x", .at = "2026-08-01",
        .recorded_at = "2026-08-01", .note = "", .scope = "work",
    };

    // Empty scope is injected everywhere, including into a recall that names none.
    try testing.expect(uni.inScope(null));
    try testing.expect(uni.inScope("work"));
    try testing.expect(uni.inScope("personal"));

    // A label is the opposite: only when named. This is the direction that
    // matters — `recall` attaches facts unconditionally, so before this a work
    // fact reached every session including ones whose output is public.
    try testing.expect(!work.inScope(null));
    try testing.expect(work.inScope("work"));
    try testing.expect(!work.inScope("personal"));

    // No prefix or hierarchy matching: a scope is an opaque string, and deciding
    // that `work` contains `work/acme` would be zkb inventing a meaning it has no
    // business having.
    try testing.expect(!work.inScope("work/acme"));
    try testing.expect(!work.inScope("wor"));
}

test "labelling a memory only ever narrows where it appears" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollectionKind("memory", "/tmp/m", .memory, 1000);
    const did = try s.upsertDocContent(cid, "m.md", "sha-m", 10, 1000);
    var vec = dummyVector(1);
    const chunk = try s.insertChunk(cid, did, .{
        .idx = 0, .heading_path = "", .byte_start = 0, .byte_end = 10,
        .n_tokens = 5, .text = "记忆正文",
    }, &vec);

    // Unscoped: visible in every scope.
    try zkb.memory.replaceMeta(&db, did, chunk, .{
        .type = .feedback, .status = .active, .created = "2026-08-18",
        .source = "test", .subjects = "", .refs = "", .scope = "",
    });
    try testing.expect(try zkb.memory.chunkInScope(&db, chunk, null));
    try testing.expect(try zkb.memory.chunkInScope(&db, chunk, "work"));

    // Re-projected with a scope: gone from the unscoped recall, still there for
    // the one that names it. Monotonic — labelling cannot widen exposure.
    try zkb.memory.replaceMeta(&db, did, chunk, .{
        .type = .feedback, .status = .active, .created = "2026-08-18",
        .source = "test", .subjects = "", .refs = "", .scope = "work",
    });
    try testing.expect(!try zkb.memory.chunkInScope(&db, chunk, null));
    try testing.expect(try zkb.memory.chunkInScope(&db, chunk, "work"));
    try testing.expect(!try zkb.memory.chunkInScope(&db, chunk, "personal"));

    // A chunk that is no memory at all has no scope to violate: recall fuses
    // results across collections, and a document must not be dropped for lacking
    // a field it never had.
    try testing.expect(try zkb.memory.chunkInScope(&db, 99999, null));
}

test "recency ranking respects the scope" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollectionKind("memory", "/tmp/m", .memory, 1000);
    // Two documents, not two chunks of one: `replaceMeta` deletes by doc_id, so
    // one memory per file is the shape it is written for — which is also how
    // `remember` writes them.
    const d_uni = try s.upsertDocContent(cid, "uni.md", "sha-u", 10, 1000);
    const d_work = try s.upsertDocContent(cid, "work.md", "sha-w", 10, 1000);
    var v1 = dummyVector(1);
    var v2 = dummyVector(2);
    const c_uni = try s.insertChunk(cid, d_uni, .{ .idx = 0, .heading_path = "", .byte_start = 0, .byte_end = 5, .n_tokens = 5, .text = "通用" }, &v1);
    const c_work = try s.insertChunk(cid, d_work, .{ .idx = 0, .heading_path = "", .byte_start = 0, .byte_end = 5, .n_tokens = 5, .text = "工作" }, &v2);

    try zkb.memory.replaceMeta(&db, d_uni, c_uni, .{ .type = .feedback, .status = .active, .created = "2026-08-17", .source = "t", .subjects = "", .refs = "", .scope = "" });
    try zkb.memory.replaceMeta(&db, d_work, c_work, .{ .type = .feedback, .status = .active, .created = "2026-08-18", .source = "t", .subjects = "", .refs = "", .scope = "work" });

    // Filtered in SQL, not after fusing: a dropped candidate would still have
    // consumed a slot in the budget and starved one that belongs.
    const none = try zkb.memory.recencyRanked(gpa, &db, 10, null);
    defer gpa.free(none);
    try testing.expectEqual(@as(usize, 1), none.len);
    try testing.expectEqual(c_uni, none[0]);

    const work = try zkb.memory.recencyRanked(gpa, &db, 10, "work");
    defer gpa.free(work);
    try testing.expectEqual(@as(usize, 2), work.len);
}

test "a scoped row is freed like any other, whichever wrapper parsed it" {
    // `scope` was added to `Fact` and three places had to grow a line: the two
    // hand-written `deinit` field lists and `Current.deinit`. Only the last one
    // did, so every `zkb facts` leaked one string per row — invisible, because a
    // leak in a process about to exit costs nothing anyone would notice.
    //
    // `testing.allocator` is the whole assertion: it fails on a leak, and it
    // panics on the double free that lived one layer up in `recall`.
    {
        var parsed = try zkb.facts.parse(gpa, scoped_csv);
        defer parsed.deinit(gpa);
        try testing.expectEqual(@as(usize, 3), parsed.facts.len);
        try testing.expectEqualStrings("work", parsed.facts[0].scope);
    }

    // The dedup path replaces a winner's strings in place, `scope` among them,
    // and that is a second owner of the same field.
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = try TmpCsv.init(io, "facts-scope", scoped_csv);
    defer tmp.deinit(io);

    const rows = try zkb.facts.currentAll(gpa, io, tmp.path);
    defer {
        for (rows) |r| r.deinit(gpa);
        gpa.free(rows);
    }
    try testing.expectEqual(@as(usize, 2), rows.len);
    for (rows) |r| {
        if (!std.mem.eql(u8, r.key, "salary")) continue;
        try testing.expectEqualStrings("480000", r.value);
        try testing.expectEqualStrings("work", r.scope);
    }
}

test "a labelled fact is injected only for its own scope" {
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = try TmpCsv.init(io, "facts-inscope",
        \\key,value,at,recorded_at,src,note,scope
        \\height,178,2026-01-01,2026-01-02,user,,
        \\salary,450000,2026-01-01,2026-01-02,user,,work
        \\
    );
    defer tmp.deinit(io);

    const rows = try zkb.facts.currentAll(gpa, io, tmp.path);
    defer {
        for (rows) |r| r.deinit(gpa);
        gpa.free(rows);
    }

    for (rows) |r| {
        const universal = std.mem.eql(u8, r.key, "height");
        // Unscoped: everywhere. Scoped: only where it was named — and `null`,
        // which is what a caller that passed no scope looks like, is not a match.
        try testing.expectEqual(universal, r.inScope(null));
        try testing.expectEqual(universal, r.inScope("personal"));
        try testing.expect(r.inScope("work"));
    }
}

test "currentInScope applies that rule, so both recall transports get one answer" {
    // `inScope` was right all along; what was missing was a caller that applied
    // it on both paths. The CLI filtered; `handleRecall` read `currentAll` and
    // rendered everything it found — so a scoped recall hid another label's
    // memories while printing that label's facts, on the transport that normally
    // serves. One function now, so there is one behaviour to be wrong about.
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = try TmpCsv.init(io, "facts-currentinscope",
        \\key,value,at,recorded_at,src,note,scope
        \\height,178,2026-01-01,2026-01-02,user,,
        \\salary,450000,2026-01-01,2026-01-02,user,,work
        \\
    );
    defer tmp.deinit(io);

    for ([_]struct { scope: ?[]const u8, want: usize }{
        // No scope named: universal only. This is what an unset `--scope` means,
        // and what an absent field on the wire has to keep meaning.
        .{ .scope = null, .want = 1 },
        .{ .scope = "work", .want = 2 },
        // The leak, stated as a test: another label must not see it.
        .{ .scope = "personal", .want = 1 },
    }) |c| {
        const rows = try zkb.facts.currentInScope(gpa, io, tmp.path, c.scope);
        defer {
            for (rows) |r| r.deinit(gpa);
            gpa.free(rows);
        }
        try testing.expectEqual(c.want, rows.len);
        for (rows) |r| try testing.expect(r.inScope(c.scope));
    }
}

const scoped_csv =
    \\key,value,at,recorded_at,src,note,scope
    \\salary,450000,2026-01-01,2026-01-02,user,月額,work
    \\height,178,2026-01-01,2026-01-02,user,,
    \\salary,480000,2026-06-01,2026-06-02,user,昇給後,work
    \\
;

/// A facts.csv under /tmp, because `currentAll` reads a path, not a string —
/// which is the half where the ownership bug lived.
const TmpCsv = struct {
    dir: []u8,
    path: []u8,

    fn init(io: std.Io, name: []const u8, body: []const u8) !TmpCsv {
        const dir = try std.fmt.allocPrint(gpa, "/tmp/zkb-{s}-{x}", .{ name, @intFromPtr(body.ptr) });
        errdefer gpa.free(dir);
        var d = try std.Io.Dir.cwd().createDirPathOpen(io, dir, .{});
        defer d.close(io);
        var f = try d.createFile(io, "facts.csv", .{});
        defer f.close(io);
        var buf: [1024]u8 = undefined;
        var wr = f.writer(io, &buf);
        try wr.interface.writeAll(body);
        try wr.interface.flush();
        return .{ .dir = dir, .path = try std.fmt.allocPrint(gpa, "{s}/facts.csv", .{dir}) };
    }

    fn deinit(self: *TmpCsv, io: std.Io) void {
        std.Io.Dir.cwd().deleteTree(io, self.dir) catch {};
        gpa.free(self.path);
        gpa.free(self.dir);
    }
};

test "appending to a file written before a column widens its header first" {
    // The parser rejects a row whose field count differs from the header, in
    // both directions. So a writer emitting today's seven fields into a file
    // whose header still says six does not add a fact — it adds a bad row, and
    // `remember-fact` prints "appended" over something nothing can read back.
    // That was live for *every* `remember-fact` on a pre-`scope` file, not only
    // the ones passing `--scope`.
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = try TmpCsv.init(io, "facts-migrate",
        \\key,value,at,recorded_at,src,note
        \\height,178,2026-01-01,2026-01-02,user,原有的
        \\salary,450000,2026-02-01,2026-02-02,user,"带,逗号的备注"
        \\
    );
    defer tmp.deinit(io);

    try zkb.facts.append(gpa, io, tmp.path, "weight", "70", "2026-08-24", "2026-08-24", "user", "新的", "");

    const rows = try zkb.facts.currentAll(gpa, io, tmp.path);
    defer {
        for (rows) |r| r.deinit(gpa);
        gpa.free(rows);
    }
    // Three: the new row is readable, and neither old row was lost to the
    // widening — the failure mode of migrating the header on its own.
    try testing.expectEqual(@as(usize, 3), rows.len);
    for (rows) |r| {
        if (!std.mem.eql(u8, r.key, "salary")) continue;
        // Quoting survives the rewrite; a comma inside a field is why the file
        // is rewritten with the csv writer rather than by appending a column.
        try testing.expectEqualStrings("带,逗号的备注", r.note);
    }
    const still_bad = try zkb.facts.badRowLines(gpa, io, tmp.path);
    defer gpa.free(still_bad);
    try testing.expectEqual(@as(usize, 0), still_bad.len);
}

test "a file that already has bad rows is refused, not rewritten" {
    // Those rows are the data at risk — most likely ones a newer zkb appended
    // and nothing has read since. `Table` keeps only their line numbers, so
    // rewriting from it would quietly finish the job the bug started.
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const before =
        \\key,value,at,recorded_at,src,note
        \\height,178,2026-01-01,2026-01-02,user,好行
        \\lost,1,2026-02-01,2026-02-02,user,已经丢过的行,work
        \\
    ;
    var tmp = try TmpCsv.init(io, "facts-badrows", before);
    defer tmp.deinit(io);

    try testing.expectError(
        error.StaleHeaderWithBadRows,
        zkb.facts.append(gpa, io, tmp.path, "weight", "70", "2026-08-24", "2026-08-24", "user", "", ""),
    );

    // Byte-for-byte untouched: refusing has to mean refusing, not "refused
    // after writing most of it".
    const after = try readWholeFile(io, tmp.path);
    defer gpa.free(after);
    try testing.expectEqualStrings(before, after);

    const lines = try zkb.facts.badRowLines(gpa, io, tmp.path);
    defer gpa.free(lines);
    try testing.expectEqualSlices(usize, &.{3}, lines);
}

fn readWholeFile(io: std.Io, path: []const u8) ![]u8 {
    var f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    const size = (try f.stat(io)).size;
    const buf = try gpa.alloc(u8, @intCast(size));
    errdefer gpa.free(buf);
    var rbuf: [4096]u8 = undefined;
    var r = f.reader(io, &rbuf);
    try r.interface.readSliceAll(buf);
    return buf;
}

test "activeScopes is the registry, and archived labels drop out of it" {
    // There is no list of allowed scopes and there should not be one: a second
    // truth about what is permitted would have to be kept in step with the files,
    // and the files are the truth. So "is this scope new?" is answered by what
    // the store already carries — which is also why an archived memory must stop
    // counting, or a typo made once would be accepted forever.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollectionKind("memory", "/tmp/m", .memory, 1000);

    const rows = [_]struct { path: []const u8, scope: []const u8, status: zkb.memory.Status }{
        .{ .path = "a.md", .scope = "work", .status = .active },
        .{ .path = "b.md", .scope = "work", .status = .active },
        .{ .path = "c.md", .scope = "", .status = .active },
        .{ .path = "d.md", .scope = "retired", .status = .archived },
    };
    for (rows, 0..) |r, i| {
        var sha_buf: [32]u8 = undefined;
        const did = try s.upsertDocContent(
            cid,
            r.path,
            try std.fmt.bufPrint(&sha_buf, "sha-{d}", .{i}),
            10,
            1000,
        );
        var vec = dummyVector(@intCast(i));
        const chunk = try s.insertChunk(cid, did, .{
            .idx = 0, .heading_path = "", .byte_start = 0, .byte_end = 10,
            .n_tokens = 5, .text = "记忆正文",
        }, &vec);
        try zkb.memory.replaceMeta(&db, did, chunk, .{
            .type = .project,
            .status = r.status,
            .created = "2026-08-28",
            .source = "t",
            .subjects = "",
            .refs = "",
            .scope = r.scope,
        });
    }

    const scopes = try zkb.memory.activeScopes(gpa, &db);
    defer {
        for (scopes) |x| gpa.free(x);
        gpa.free(scopes);
    }
    // `work` once despite two memories; the empty one is universal, not a label;
    // `retired` is gone with its only memory.
    try testing.expectEqual(@as(usize, 1), scopes.len);
    try testing.expectEqualStrings("work", scopes[0]);
}
