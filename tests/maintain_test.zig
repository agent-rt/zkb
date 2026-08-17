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
        \\- zkb://skills/y/SKILL.md
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
