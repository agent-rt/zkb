//! Link extraction, link resolution, and the structural checks.
//!
//! The dangerous failures here are false positives: a maintenance report that
//! cries wolf gets ignored after the second read, and then the real findings go
//! unseen too. So most of these tests are about what must *not* be reported.

const std = @import("std");
const zkb = @import("zkb");
const sqlite = zkb.sqlite;
const store = zkb.store;
const schema = zkb.schema;
const markdown = zkb.markdown;
const maintain = zkb.maintain;

const testing = std.testing;
const gpa = testing.allocator;

const dim: usize = @intCast(schema.embedding_dim);

fn addDoc(s: *store.Store, cid: i64, path: []const u8, source: []const u8) !i64 {
    var sha_buf: [80]u8 = undefined;
    const sha = try std.fmt.bufPrint(&sha_buf, "sha-{s}", .{path});
    const did = try s.upsertDocContent(cid, path, sha, @intCast(source.len), 1000);
    // Mirrors the real ingest path: chunks are replaced, not appended to.
    try s.deleteChunks(did);

    var doc = try markdown.scan(gpa, source);
    defer doc.deinit(gpa);
    try s.setDocMeta(did, doc.title, doc.frontmatter);

    const links = try markdown.extractLinks(gpa, source, &doc);
    defer gpa.free(links);
    try maintain.replaceLinks(s.db, did, links);

    var v: [dim]f32 = @splat(0);
    v[0] = 1;
    _ = try s.insertChunk(cid, did, .{
        .idx = 0,
        .heading_path = "",
        .byte_start = 0,
        .byte_end = @intCast(source.len),
        .n_tokens = 500,
        .text = source,
    }, &v);
    try s.markIndexed(did, 1, 1000);
    return did;
}

// ---------------------------------------------------------------------------
// extraction
// ---------------------------------------------------------------------------

test "links are extracted from every syntax the corpus actually uses" {
    const src =
        \\---
        \\depends_on:
        \\  - projects/alpha/REQ.md
        \\related_to: [skills/alpha/SKILL.md]
        \\---
        \\
        \\# Doc
        \\
        \\See [the spec](../other/SPEC.md) and [[wikilink]].
        \\Also zkb://projects/x/index.md and [ext](https://example.com/page).
        \\
    ;
    var doc = try markdown.scan(gpa, src);
    defer doc.deinit(gpa);
    const links = try markdown.extractLinks(gpa, src, &doc);
    defer gpa.free(links);

    var kinds: std.EnumMap(markdown.LinkKind, usize) = .{};
    for (links) |l| kinds.put(l.kind, (kinds.get(l.kind) orelse 0) + 1);

    try testing.expectEqual(@as(usize, 2), kinds.get(.frontmatter).?);
    try testing.expectEqual(@as(usize, 1), kinds.get(.md).?);
    try testing.expectEqual(@as(usize, 1), kinds.get(.wiki).?);
    try testing.expectEqual(@as(usize, 1), kinds.get(.collection_uri).?);
    // http links are recorded but never resolved: zkb does not go online, so
    // reporting them as broken would be reporting on something it cannot know.
    try testing.expectEqual(@as(usize, 1), kinds.get(.external).?);
}

test "opaque uri schemes are external, not relative paths" {
    // A scheme is not always followed by `//`. Looking for `://` misses
    // data: and mailto:, which then fall through to the relative-path branch
    // and get reported as broken links — nine inline SVGs in one downloaded
    // article did exactly that.
    const src =
        \\# Doc
        \\
        \\![svg](data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%3F%3E)
        \\Write to [me](mailto:someone@example.com) or call [here](tel:+81-3-1234-5678).
        \\Spec at zkb://projects/x/SPEC.md, notes in [file](../notes/today.md).
        \\A [dated file](2026-08-17:log.md) is a path, not a scheme.
        \\详见 zkb://projects/x/REQ.md。另见 zkb://projects/x/PLAN.md、以上。
        \\
    ;
    var doc = try markdown.scan(gpa, src);
    defer doc.deinit(gpa);
    const links = try markdown.extractLinks(gpa, src, &doc);
    defer gpa.free(links);

    var kinds: std.EnumMap(markdown.LinkKind, usize) = .{};
    for (links) |l| kinds.put(l.kind, (kinds.get(l.kind) orelse 0) + 1);

    // data:, mailto: and tel: — none of them a path to resolve.
    try testing.expectEqual(@as(usize, 3), kinds.get(.external).?);
    // Three bare collection uris, each followed by punctuation: an ascii comma,
    // a full-width period and a full-width enumeration comma. Swallowing any of
    // them turns `.md` into `.md,` and files a document reference as an asset.
    try testing.expectEqual(@as(usize, 3), kinds.get(.collection_uri).?);
    try testing.expect(kinds.get(.asset) == null);
    // The relative link, plus the one that only looks like it has a scheme:
    // a leading digit cannot start one, so it stays a path.
    try testing.expectEqual(@as(usize, 2), kinds.get(.md).?);
}

test "paths inside code fences are not links" {
    // A path in a shell snippet or sample config is not a reference. Counting it
    // produces broken-link noise for something nobody meant as a link.
    const src =
        \\# Doc
        \\
        \\```bash
        \\cat [not a link](./nope.md)
        \\see zkb://projects/fake/thing.md
        \\```
        \\
        \\Real [link](./real.md).
        \\
    ;
    var doc = try markdown.scan(gpa, src);
    defer doc.deinit(gpa);
    const links = try markdown.extractLinks(gpa, src, &doc);
    defer gpa.free(links);

    try testing.expectEqual(@as(usize, 1), links.len);
    try testing.expectEqualStrings("./real.md", links[0].raw);
}

test "link syntax written as inline code is not a link" {
    // Documentation that explains linking is full of `[text](path)` examples.
    // Counting them yields phantom broken links to "path" forever.
    const src =
        \\# Doc
        \\
        \\Supported forms: `[text](path)`, `[[wikilink]]`, and `zkb://x/y.md`.
        \\
        \\A real one: [spec](./SPEC.md).
        \\
    ;
    var doc = try markdown.scan(gpa, src);
    defer doc.deinit(gpa);
    const links = try markdown.extractLinks(gpa, src, &doc);
    defer gpa.free(links);

    try testing.expectEqual(@as(usize, 1), links.len);
    try testing.expectEqualStrings("./SPEC.md", links[0].raw);
}

test "assets and file URLs are distinguished from broken links" {
    const src =
        \\# Doc
        \\
        \\[data](./data/table.json) and [script](file:///Users/x/build.sh)
        \\and [img](../img/pic.png).
        \\
    ;
    var doc = try markdown.scan(gpa, src);
    defer doc.deinit(gpa);
    const links = try markdown.extractLinks(gpa, src, &doc);
    defer gpa.free(links);

    // A .json that exists on disk is not a document zkb indexes; calling it a
    // broken link would be reporting something untrue. file:// is an absolute
    // machine path, never collection-relative.
    var assets: usize = 0;
    var external: usize = 0;
    for (links) |l| switch (l.kind) {
        .asset => assets += 1,
        .external => external += 1,
        else => {},
    };
    try testing.expectEqual(@as(usize, 2), assets);
    try testing.expectEqual(@as(usize, 1), external);
}

// ---------------------------------------------------------------------------
// resolution
// ---------------------------------------------------------------------------

test "resolution happens after the whole scan, not during it" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);

    // a.md links to b.md, which is indexed *after* it. Resolving inline would
    // call this broken, and whether it did would depend on iteration order.
    _ = try addDoc(&s, cid, "a.md", "# A\n\nSee [b](./b.md).\n");
    const broken_before = (try db.queryI64(
        "SELECT count(*) FROM links WHERE target_doc_id IS NULL AND kind != 'external'",
    )).?;
    try testing.expectEqual(@as(i64, 1), broken_before);

    _ = try addDoc(&s, cid, "b.md", "# B\n");
    _ = try maintain.resolveLinks(gpa, &db);

    try testing.expectEqual(@as(?i64, 0), try db.queryI64(
        "SELECT count(*) FROM links WHERE target_doc_id IS NULL AND kind != 'external'",
    ));
}

test "relative, absolute, collection-uri and wiki targets all resolve" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);

    _ = try addDoc(&s, cid, "projects/x/REQ.md", "# REQ\n");
    _ = try addDoc(&s, cid, "projects/x/SPEC.md", "# SPEC\n");
    _ = try addDoc(&s, cid, "skills/y/SKILL.md", "# SKILL\n");
    _ = try addDoc(&s, cid, "projects/x/index.md",
        \\# Index
        \\
        \\- [req](./REQ.md)
        \\- [spec](SPEC.md)
        \\- zkb://docs/skills/y/SKILL.md
        \\- [[SKILL]]
        \\
    );

    _ = try maintain.resolveLinks(gpa, &db);
    try testing.expectEqual(@as(?i64, 0), try db.queryI64(
        "SELECT count(*) FROM links WHERE target_doc_id IS NULL AND kind != 'external'",
    ));
}

test "anchors and external URLs are never counted as broken" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    _ = try addDoc(&s, cid, "a.md",
        \\# A
        \\
        \\[section](#somewhere) and [site](https://example.com).
        \\
    );
    _ = try maintain.resolveLinks(gpa, &db);

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.broken_link} });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), report.count(.broken_link));
}

// ---------------------------------------------------------------------------
// checks
// ---------------------------------------------------------------------------

test "a genuinely missing target is reported once, with the raw text" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    _ = try addDoc(&s, cid, "a.md", "# A\n\n[gone](./missing.md)\n");
    _ = try maintain.resolveLinks(gpa, &db);

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.broken_link} });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), report.count(.broken_link));
    try testing.expectEqualStrings("a.md", report.findings[0].path);
    try testing.expect(std.mem.indexOf(u8, report.findings[0].detail, "missing.md") != null);
}

test "index and readme files are not reported as orphans" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);

    // Nothing links to any of these; only the content file is a real finding.
    _ = try addDoc(&s, cid, "index.md", "# Root\n");
    _ = try addDoc(&s, cid, "projects/x/README.md", "# Readme\n");
    _ = try addDoc(&s, cid, "projects/x/orphan.md", "# Orphan\n");
    _ = try maintain.resolveLinks(gpa, &db);

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.orphan} });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), report.count(.orphan));
    try testing.expectEqualStrings("projects/x/orphan.md", report.findings[0].path);
}

test "not_in_index fires only for a directory that has an index" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);

    _ = try addDoc(&s, cid, "projects/x/index.md", "# X\n\n- [req](./REQ.md)\n");
    _ = try addDoc(&s, cid, "projects/x/REQ.md", "# REQ\n"); // listed
    _ = try addDoc(&s, cid, "projects/x/SPEC.md", "# SPEC\n"); // not listed
    _ = try addDoc(&s, cid, "projects/y/LONE.md", "# Lone\n"); // no index.md here
    _ = try maintain.resolveLinks(gpa, &db);

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.not_in_index} });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), report.count(.not_in_index));
    try testing.expectEqualStrings("projects/x/SPEC.md", report.findings[0].path);
}

test "fragment, oversized and missing frontmatter" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);

    const with_fm = "---\nnode_type: wiki\n---\n\n# Has frontmatter\n";
    _ = try addDoc(&s, cid, "fm.md", with_fm);
    _ = try addDoc(&s, cid, "plain.md", "# No frontmatter\n");

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.no_frontmatter} });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), report.count(.no_frontmatter));
    try testing.expectEqualStrings("plain.md", report.findings[0].path);

    // fragment uses the stored token count, so it needs a small chunk.
    try db.exec("UPDATE chunks SET n_tokens = 20 WHERE doc_id = (SELECT id FROM docs WHERE rel_path='plain.md');");
    var frag = try maintain.run(gpa, &db, .{ .checks = &.{.fragment} });
    defer frag.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), frag.count(.fragment));

    try db.exec("UPDATE docs SET chunk_count = 80 WHERE rel_path='fm.md';");
    var big = try maintain.run(gpa, &db, .{ .checks = &.{.oversized} });
    defer big.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), big.count(.oversized));
}

test "link checks are skipped, not passed, when the graph is empty" {
    // An empty graph would make every document look unlinked. Reporting 195
    // orphans because the graph has not been built yet is worse than useless.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    _ = try s.upsertDocContent(cid, "a.md", "sha", 10, 1000);

    var report = try maintain.run(gpa, &db, .{});
    defer report.deinit(gpa);
    try testing.expect(report.link_graph_empty);
    try testing.expectEqual(@as(usize, 0), report.count(.orphan));
}

// ---------------------------------------------------------------------------
// history
// ---------------------------------------------------------------------------

test "the diff separates new from resolved and survives re-indexing" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    _ = try addDoc(&s, cid, "a.md", "# A\n\n[gone](./missing.md)\n");
    _ = try maintain.resolveLinks(gpa, &db);

    var first = try maintain.run(gpa, &db, .{ .checks = &.{.broken_link} });
    defer first.deinit(gpa);
    {
        var d = try maintain.diffAgainstLast(gpa, &db, &first);
        defer d.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), d.new_keys.len);
    }
    try maintain.record(gpa, &db, &first, maintain.Check.default(), 1000);

    // Re-index the same document: chunk and link row ids change, but the finding
    // is the same one. Keys are content-derived precisely so this is "unchanged"
    // rather than "everything is new again".
    _ = try addDoc(&s, cid, "a.md", "# A\n\n[gone](./missing.md)\n");
    _ = try maintain.resolveLinks(gpa, &db);
    var second = try maintain.run(gpa, &db, .{ .checks = &.{.broken_link} });
    defer second.deinit(gpa);
    {
        var d = try maintain.diffAgainstLast(gpa, &db, &second);
        defer d.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), d.new_keys.len);
        try testing.expectEqual(@as(usize, 1), d.unchanged);
    }
    try maintain.record(gpa, &db, &second, maintain.Check.default(), 2000);

    // Fix it: the finding must show up as resolved.
    _ = try addDoc(&s, cid, "missing.md", "# Now exists\n");
    _ = try maintain.resolveLinks(gpa, &db);
    var third = try maintain.run(gpa, &db, .{ .checks = &.{.broken_link} });
    defer third.deinit(gpa);
    var d = try maintain.diffAgainstLast(gpa, &db, &third);
    defer d.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), d.new_keys.len);
    try testing.expectEqual(@as(usize, 1), d.resolved_keys.len);
}

// ---------------------------------------------------------------------------
// vector checks: the classification rules, which E7 measured as mattering more
// than the threshold does
// ---------------------------------------------------------------------------

test "a table of contents is recognised as navigation" {
    // The single largest false-positive class E7 found: two project index.md
    // files whose "related knowledge" sections resemble each other because both
    // are lists of links. The highest-scoring pair in the whole pool (0.973)
    // was one of these.
    const toc =
        \\## 关联知识
        \\- [projects/a/index.md](zkb://projects/a/index.md) — A
        \\- [projects/b/index.md](zkb://projects/b/index.md) — B
    ;
    try testing.expect(zkb.maintain_vec.isNavigationForTest(toc));

    const prose =
        \\本 API 所有对外可见的资源 ID 采用 GID 规范：资源对象的 JSON 字段
        \\统一使用 id 表示 GID，跨服务引用时必须携带 kind 前缀。
    ;
    try testing.expect(!zkb.maintain_vec.isNavigationForTest(prose));
}

test "a prose paragraph with one link is not navigation" {
    const mostly_prose =
        \\这一节说明为什么选择 RRF 而不是加权求和，详见 [SPEC](zkb://SPEC.md)。
        \\加权求和需要一个系数，而没有实验支撑的系数正是当初要避开的那笔债。
    ;
    try testing.expect(!zkb.maintain_vec.isNavigationForTest(mostly_prose));
}

test "two files of one project are the same project, nested or not" {
    const T = zkb.maintain_vec;
    try testing.expect(T.sameProjectForTest("projects/x/REQ.md", "projects/x/SPEC.md"));
    // A doc under a subdirectory still belongs to its project.
    try testing.expect(T.sameProjectForTest("projects/x/docs/design.md", "projects/x/README.md"));
    try testing.expect(!T.sameProjectForTest("projects/x/REQ.md", "projects/y/REQ.md"));
    try testing.expect(!T.sameProjectForTest("projects/x/REQ.md", "skills/z/case.md"));
}

test "a section title drops its number so two numberings compare equal" {
    // `REQ.md > 11. 验收标准` and `SPEC.md > 14. 验收标准` are the same section
    // restated, which is a finding; two *different* sections overlapping is the
    // document matrix working as intended, which is not.
    const T = zkb.maintain_vec;
    try testing.expectEqualStrings("验收标准", T.sectionTitleForTest("toolname — 需求规格说明书 > 11. 验收标准"));
    try testing.expectEqualStrings("验收标准", T.sectionTitleForTest("toolname — 技术规格说明书 > 14. 验收标准"));
    try testing.expectEqualStrings("Resource IDs", T.sectionTitleForTest("Resource IDs"));
    try testing.expectEqualStrings("安全机制", T.sectionTitleForTest("a > b > 4. 安全机制"));
}

test "太短的 chunk 不参与相似度比较" {
    // 真实语料上这一类占了 near_duplicate 报告的 8/18：五份事故报告共用日式商务信函的
    // 固定结尾，`以上` 是 2 个 token，任意两个这样的 chunk 余弦都是 1.000。那说明的是
    // 「短」，不是「重复」——而它把真正需要看的几对埋掉了。
    //
    // 两组 chunk 用同一个向量，余弦都是 1.000；唯一的变量是 n_tokens。
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var vs = store.Store.init(&db);

    var vec: [1024]f32 = @splat(0);
    vec[7] = 1.0;
    const v = vec[0..@intCast(zkb.schema.embedding_dim)];

    const cid = try vs.ensureCollection("docs", "/tmp/docs", 1000);
    const doc_a = try vs.upsertDocContent(cid, "a.md", "sha-a", 10, 1000);
    const doc_b = try vs.upsertDocContent(cid, "b.md", "sha-b", 10, 1000);

    for ([_]i64{ doc_a, doc_b }) |doc| {
        _ = try vs.insertChunk(cid, doc, .{
            .idx = 0,
            .heading_path = "报告",
            .byte_start = 0,
            .byte_end = 6,
            .n_tokens = 2,
            .text = "以上",
        }, v);
    }

    var r1 = try zkb.maintain_vec.run(gpa, &db, .{});
    defer r1.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), r1.pairs.len);

    // 同样两篇各加一个足够长、向量也相同的 chunk —— 这一对该报出来。
    for ([_]i64{ doc_a, doc_b }) |doc| {
        _ = try vs.insertChunk(cid, doc, .{
            .idx = 1,
            .heading_path = "报告 > 原因",
            .byte_start = 10,
            .byte_end = 400,
            .n_tokens = 300,
            .text = "本事象は複数の要因が重なって発生し、自動復旧が期待どおりに働かなかった",
        }, v);
    }

    var r2 = try zkb.maintain_vec.run(gpa, &db, .{});
    defer r2.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), r2.pairs.len);
}

test "orphan is a documents-collection finding, not a memory one" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);

    const docs = try s.ensureCollectionKind("docs", "/tmp/docs", .documents, 1000);
    const mem = try s.ensureCollectionKind("memory", "/tmp/mem", .memory, 1000);
    const kb = try s.ensureCollectionKind("numbers", "/tmp/numbers", .records, 1000);

    // Nothing links to any of these.
    _ = try s.upsertDocContent(docs, "lonely.md", "sha1", 10, 1000);
    _ = try s.upsertDocContent(mem, "a-memory.md", "sha2", 10, 1000);
    _ = try s.upsertDocContent(kb, "facts.csv", "sha3", 10, 1000);

    // The graph has to look built, or every link check is skipped as meaningless.
    try schema.setMeta(&db, "links_extracted", "1");

    var report = try zkb.maintain.run(gpa, &db, .{ .checks = &.{.orphan} });
    defer report.deinit(gpa);

    // A memory with no inbound link is not a finding: `remember` writes one file
    // per memory and no index ever points at them, so every memory would be one.
    try testing.expectEqual(@as(usize, 1), report.count(.orphan));
    try testing.expectEqualStrings("lonely.md", report.findings[0].path);
}

/// A short chunk in a memory collection.
///
/// Deliberately not `addDoc`: that one writes 500 tokens, and the whole point
/// here is the shape `remember` actually produces — one file, one fact, one
/// small chunk.
fn addTinyDoc(s: *store.Store, cid: i64, path: []const u8, text: []const u8) !i64 {
    var sha_buf: [80]u8 = undefined;
    const sha = try std.fmt.bufPrint(&sha_buf, "sha-{s}", .{path});
    const did = try s.upsertDocContent(cid, path, sha, @intCast(text.len), 1000);
    var v: [dim]f32 = @splat(0);
    v[0] = 1;
    _ = try s.insertChunk(cid, did, .{
        .idx = 0,
        .heading_path = "",
        .byte_start = 0,
        .byte_end = @intCast(text.len),
        .n_tokens = 20,
        .text = text,
    }, &v);
    try s.markIndexed(did, 1, 1000);
    return did;
}

test "fragment is a documents-collection finding, not a memory one" {
    // The bug this encodes: measured on the real index, `fragment` reported 28 of
    // the 35 memories zkb had written itself. One memory is one fact, so a single
    // short chunk is the correct shape — `maintain` was reporting `remember`'s
    // normal output back as a defect.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);

    const docs = try s.ensureCollectionKind("docs", "/tmp/docs", .documents, 1000);
    const mem = try s.ensureCollectionKind("memory", "/tmp/mem", .memory, 1000);

    _ = try addTinyDoc(&s, docs, "stub.md", "barely anything");
    _ = try addTinyDoc(&s, mem, "prefers-jj.md", "uses jj over git");

    var report = try zkb.maintain.run(gpa, &db, .{ .checks = &.{.fragment} });
    defer report.deinit(gpa);

    try testing.expectEqual(@as(usize, 1), report.count(.fragment));
    try testing.expectEqualStrings("stub.md", report.findings[0].path);
}

test "a collection declines a check its kind would otherwise allow" {
    // Two `documents` collections, one of which keeps no index.md. The kind is
    // identical, so nothing derived from the kind can tell them apart — which is
    // the whole reason `checks_off` exists rather than a fourth kind.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);

    const docs = try s.ensureCollectionKind("docs", "/tmp/docs", .documents, 1000);
    const loose = try s.ensureCollectionKind("synap", "/tmp/synap", .documents, 1000);

    _ = try s.upsertDocContent(docs, "lonely.md", "sha1", 10, 1000);
    _ = try s.upsertDocContent(loose, "quickstart.md", "sha2", 10, 1000);
    try schema.setMeta(&db, "links_extracted", "1");

    {
        var before = try zkb.maintain.run(gpa, &db, .{ .checks = &.{.orphan} });
        defer before.deinit(gpa);
        try testing.expectEqual(@as(usize, 2), before.count(.orphan));
    }

    try s.setChecksOff(loose, "orphan");

    var after = try zkb.maintain.run(gpa, &db, .{ .checks = &.{.orphan} });
    defer after.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), after.count(.orphan));
    try testing.expectEqualStrings("lonely.md", after.findings[0].path);

    // And back: an opt-out entered by mistake has to be undoable without
    // rebuilding anything.
    try s.setChecksOff(loose, "");
    var undone = try zkb.maintain.run(gpa, &db, .{ .checks = &.{.orphan} });
    defer undone.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), undone.count(.orphan));
}

test "an opt-out names one check and leaves the collection's others alone" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);

    const docs = try s.ensureCollectionKind("docs", "/tmp/docs", .documents, 1000);
    _ = try addTinyDoc(&s, docs, "stub.md", "barely anything");
    try schema.setMeta(&db, "links_extracted", "1");

    // A substring of another check's name, and a name with surrounding spaces:
    // both are ways a sloppy match would switch off the wrong thing.
    try s.setChecksOff(docs, " orphan , fragment ");

    var report = try zkb.maintain.run(gpa, &db, .{ .checks = &.{ .orphan, .fragment, .oversized } });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), report.count(.orphan));
    try testing.expectEqual(@as(usize, 0), report.count(.fragment));

    try s.setChecksOff(docs, "orphan_chunk");
    var partial = try zkb.maintain.run(gpa, &db, .{ .checks = &.{ .orphan, .fragment } });
    defer partial.deinit(gpa);
    // `orphan_chunk` shares a prefix with `orphan`; switching one off must not
    // take the other with it.
    try testing.expectEqual(@as(usize, 1), partial.count(.orphan));
    try testing.expectEqual(@as(usize, 1), partial.count(.fragment));
}

test "--collection narrows the report without narrowing the conventions" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);

    const docs = try s.ensureCollectionKind("docs", "/tmp/docs", .documents, 1000);
    const other = try s.ensureCollectionKind("synap", "/tmp/synap", .documents, 1000);
    _ = try s.upsertDocContent(docs, "lonely.md", "sha1", 10, 1000);
    _ = try s.upsertDocContent(other, "also-lonely.md", "sha2", 10, 1000);
    try schema.setMeta(&db, "links_extracted", "1");
    try s.setChecksOff(other, "orphan");

    var report = try zkb.maintain.run(gpa, &db, .{
        .checks = &.{.orphan},
        .only_collection = other,
    });
    defer report.deinit(gpa);
    // Asked for `synap` alone, and `synap` declines this check: the answer is
    // nothing, not "the other collection's finding because you asked for one".
    try testing.expectEqual(@as(usize, 0), report.count(.orphan));
}

test "a namespace convention file is content, and has to be linked like content" {
    // `zkb.md` is the per-directory convention file — what depends on what, and
    // in which order edits propagate. It looks like CLAUDE.md, but it is not: the
    // corpus registers it in the namespace's index.md, the same as every other
    // document, and it carries `part_of:` pointing back there. Exempting it from
    // the orphan check would buy a convention file silence at the price of the
    // one signal that says nobody wrote it down.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);

    // A namespace that did it right: index.md lists its convention file.
    _ = try addDoc(&s, cid, "projects/a/index.md",
        \\# A
        \\
        \\| [zkb.md](zkb://docs/projects/a/zkb.md) | 本命名空间约定 |
        \\
    );
    _ = try addDoc(&s, cid, "projects/a/zkb.md", "# projects/a 命名空间约定\n");

    // A namespace that forgot. Its index.md links nothing.
    _ = try addDoc(&s, cid, "projects/b/index.md", "# B\n");
    _ = try addDoc(&s, cid, "projects/b/zkb.md", "# projects/b 命名空间约定\n");

    _ = try maintain.resolveLinks(gpa, &db);

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.orphan} });
    defer report.deinit(gpa);

    // Only the unregistered one. Both index.md files stay exempt — nothing links
    // to an index by construction, which is the distinction zkb.md does not share.
    try testing.expectEqual(@as(usize, 1), report.count(.orphan));
    try testing.expectEqualStrings("projects/b/zkb.md", report.findings[0].path);
}

test "an indexed memory with no projection row is reported" {
    // The failure this exists for: `recall` ranks memories out of `rec_memory`
    // alone, so a missing row makes a memory unrankable while `status` still
    // counts the document and `search` still finds its chunks. Every surface that
    // could have contradicted the empty recall was reading a different table, and
    // 40 of 40 memories stayed missing for two days behind "No memories yet".
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.upsertCollection("memory", "/tmp/memory", .memory, null, null, 1000);
    _ = try addDoc(&s, cid, "one.md", "# One\n\nbody\n");

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.unprojected_memory} });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), report.count(.unprojected_memory));
    try testing.expectEqualStrings("memory", report.findings[0].path);
    try testing.expect(std.mem.indexOf(u8, report.findings[0].detail, "1 indexed memories") != null);
}

test "a projected memory, and any document collection, are left alone" {
    // The other half: a check that fires on a healthy corpus is worse than none,
    // and this one would otherwise report every document in the knowledge base —
    // no `documents` collection has a `rec_memory` row, and none should.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);

    const mem_cid = try s.upsertCollection("memory", "/tmp/memory", .memory, null, null, 1000);
    const did = try addDoc(&s, mem_cid, "one.md", "# One\n\nbody\n");
    const chunk_id = (try db.queryI64("SELECT id FROM chunks WHERE doc_id = 1")).?;
    try zkb.memory.replaceMeta(&db, did, chunk_id, .{
        .type = .feedback,
        .status = .active,
        .created = "2026-08-27",
        .source = "t",
        .subjects = "",
        .refs = "",
        .scope = "",
    });

    const doc_cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    _ = try addDoc(&s, doc_cid, "a.md", "# A\n\nbody\n");

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.unprojected_memory} });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), report.count(.unprojected_memory));
}

test "a project memory with no scope is reported, other types are not" {
    // Measured over a real store of 39: every scoped memory was `project` bar two
    // tool-name labels, and the two `project` memories left universal were both
    // about zkb — opening every unrelated session with it. So an unscoped
    // `project` memory is a forgotten label, and the cost lands elsewhere.
    //
    // The other types stay out on purpose. A lesson learned *in* a project is
    // usually still a lesson; several in that store name a project as their
    // evidence and their conclusion is general.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.upsertCollection("memory", "/tmp/memory", .memory, null, null, 1000);

    const cases = [_]struct { path: []const u8, t: zkb.memory.Type, scope: []const u8 }{
        .{ .path = "unscoped-project.md", .t = .project, .scope = "" },
        .{ .path = "scoped-project.md", .t = .project, .scope = "zkb" },
        .{ .path = "unscoped-feedback.md", .t = .feedback, .scope = "" },
    };
    for (cases) |c| {
        const did = try addDoc(&s, cid, c.path, "# X\n\nbody\n");
        var q = try db.prepare("SELECT id FROM chunks WHERE doc_id = ?1");
        defer q.finalize();
        try q.bindI64(1, did);
        try testing.expect(try q.step());
        try zkb.memory.replaceMeta(&db, did, q.columnI64(0), .{
            .type = c.t,
            .status = .active,
            .created = "2026-08-28",
            .source = "t",
            .subjects = "",
            .refs = "",
            .scope = c.scope,
        });
    }

    var report = try maintain.run(gpa, &db, .{ .checks = &.{.unscoped_project_memory} });
    defer report.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), report.count(.unscoped_project_memory));
    try testing.expectEqualStrings("unscoped-project.md", report.findings[0].path);
}

test "a link may name another collection, and resolves there" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const docs = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const aglet = try s.ensureCollection("aglet", "/tmp/aglet", 1000);

    _ = try addDoc(&s, aglet, "README.md", "# aglet\n");
    // The same rel_path in both, which is what made the old whole-corpus
    // lookup ambiguous — `index.md` existed in three collections on the real
    // one. Naming the collection is what removes the ambiguity.
    _ = try addDoc(&s, docs, "README.md", "# docs\n");
    _ = try addDoc(&s, docs, "notes.md",
        \\# Notes
        \\
        \\- [other project](zkb://aglet/README.md)
        \\
    );

    _ = try maintain.resolveLinks(gpa, &db);

    var st = try db.prepare(
        \\SELECT c.name FROM links l
        \\JOIN docs d ON d.id = l.target_doc_id
        \\JOIN collections c ON c.id = d.collection_id
        \\JOIN docs src ON src.id = l.doc_id
        \\WHERE src.rel_path = 'notes.md'
    );
    defer st.finalize();
    try testing.expect(try st.step());
    // Before this it resolved inside the *linking* document's collection, so
    // this landed on docs/README.md — the wrong file, silently — or on nothing.
    try testing.expectEqualStrings("aglet", st.columnText(0));
}

test "a relative link still means the collection it was written in" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const docs = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const aglet = try s.ensureCollection("aglet", "/tmp/aglet", 1000);

    _ = try addDoc(&s, aglet, "README.md", "# aglet\n");
    _ = try addDoc(&s, docs, "README.md", "# docs\n");
    _ = try addDoc(&s, docs, "notes.md", "# Notes\n\n- [here](README.md)\n");

    _ = try maintain.resolveLinks(gpa, &db);

    var st = try db.prepare(
        \\SELECT c.name FROM links l
        \\JOIN docs d ON d.id = l.target_doc_id
        \\JOIN collections c ON c.id = d.collection_id
        \\JOIN docs src ON src.id = l.doc_id
        \\WHERE src.rel_path = 'notes.md'
    );
    defer st.finalize();
    try testing.expect(try st.step());
    try testing.expectEqualStrings("docs", st.columnText(0));
}

test "relink throws away resolutions the old rule produced" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);

    _ = try addDoc(&s, cid, "a.md", "# A\n");
    const from = try addDoc(&s, cid, "b.md", "# B\n\n- [x](zkb://nosuch/a.md)\n");

    // Stand in for a resolution computed under a rule that no longer applies:
    // the link names a collection that does not exist, so nothing should point
    // at anything — but a stored target keeps `resolveLinks` from ever looking,
    // because it only considers links with none.
    {
        var st = try db.prepare(
            "UPDATE links SET target_doc_id = (SELECT id FROM docs WHERE rel_path = 'a.md') WHERE doc_id = ?1",
        );
        defer st.finalize();
        try st.bindI64(1, from);
        _ = try st.step();
    }
    _ = try maintain.resolveLinks(gpa, &db);
    try testing.expectEqual(@as(?i64, 0), try db.queryI64(
        "SELECT count(*) FROM links WHERE target_doc_id IS NULL",
    ));

    _ = try maintain.relinkAll(gpa, &db);
    // Now it is what it always was: a link to a collection that is not there.
    try testing.expectEqual(@as(?i64, 1), try db.queryI64(
        "SELECT count(*) FROM links WHERE target_doc_id IS NULL",
    ));
}

test "relink keeps the resolutions that are still right" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    var s = store.Store.init(&db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);

    _ = try addDoc(&s, cid, "a.md", "# A\n");
    _ = try addDoc(&s, cid, "b.md", "# B\n\n- [x](zkb://docs/a.md)\n");
    _ = try maintain.resolveLinks(gpa, &db);

    const n = try maintain.relinkAll(gpa, &db);
    try testing.expectEqual(@as(usize, 1), n);
    try testing.expectEqual(@as(?i64, 0), try db.queryI64(
        "SELECT count(*) FROM links WHERE target_doc_id IS NULL AND kind != 'external'",
    ));
}
