//! Context-pack assembly.
//!
//! The failure modes here are quiet ones: a budget that silently drops the best
//! document, ranges that leave a hole in the middle of continuous prose, or an
//! `omitted` list that lies about what was cut.

const std = @import("std");
const zkb = @import("zkb");
const sqlite = zkb.sqlite;
const store = zkb.store;
const schema = zkb.schema;
const hybrid = zkb.hybrid;
const pack = zkb.pack;

const testing = std.testing;
const gpa = testing.allocator;

const dim: usize = @intCast(schema.embedding_dim);

/// `pack.assemble` reads `data/contexts.csv` to attach what each subtree is.
/// These tests are about grouping and budget, and have no such file: the layout
/// points at a directory that does not exist, `contexts.load` returns empty, and
/// the pack is exactly what it was before descriptions existed. Contexts have
/// their own tests in `contexts_test.zig`.
fn assembleNoContexts(
    db: *sqlite.Db,
    query: []const u8,
    results: *const hybrid.Results,
    cfg: pack.Config,
) !pack.Pack {
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var env: std.process.Environ.Map = .init(gpa);
    defer env.deinit();
    try env.put("ZKB_HOME", "/tmp/zkb-pack-test-no-such-home");
    var layout = try zkb.paths.resolve(gpa, &env);
    defer layout.deinit(gpa);
    return pack.assemble(gpa, threaded.io(), &layout, db, query, results, cfg);
}

fn vec(n: u8) [dim]f32 {
    var v: [dim]f32 = @splat(0);
    v[@as(usize, n) % dim] = 1.0;
    return v;
}

/// A database with one collection and `docs` documents of `chunks_per` chunks,
/// each chunk worth `tokens` tokens.
fn seed(db: *sqlite.Db, docs: usize, chunks_per: usize, tokens: i64) !void {
    var s = store.Store.init(db);
    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    for (0..docs) |d| {
        var path_buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "doc{d}.md", .{d});
        var sha_buf: [64]u8 = undefined;
        const sha = try std.fmt.bufPrint(&sha_buf, "sha{d}", .{d});
        const did = try s.upsertDocContent(cid, path, sha, 10, 1000);
        try s.setDocMeta(did, "Title", null);
        for (0..chunks_per) |i| {
            var text_buf: [128]u8 = undefined;
            const text = try std.fmt.bufPrint(&text_buf, "doc {d} chunk {d} body text", .{ d, i });
            var head_buf: [64]u8 = undefined;
            const head = try std.fmt.bufPrint(&head_buf, "Title > Section {d}", .{i});
            var v = vec(@intCast(i));
            _ = try s.insertChunk(cid, did, .{
                .idx = @intCast(i),
                .heading_path = head,
                .byte_start = @intCast(i * 100),
                .byte_end = @intCast((i + 1) * 100),
                .n_tokens = tokens,
                .text = text,
            }, &v);
        }
        try s.markIndexed(did, @intCast(chunks_per), 1000);
    }
}

/// Build a Results by hand: the pack is what is under test, not retrieval.
fn hitsFor(db: *sqlite.Db, chunk_ids: []const i64, scores: []const f64) !hybrid.Results {
    var hits = try gpa.alloc(hybrid.Hit, chunk_ids.len);
    errdefer gpa.free(hits);
    for (chunk_ids, 0..) |cid, i| {
        var st = try db.prepare(
            \\SELECT c.doc_id, c.idx, COALESCE(c.heading_path,''), c.text, c.n_tokens,
            \\       d.rel_path, COALESCE(d.title,''), col.name
            \\FROM chunks c JOIN docs d ON d.id = c.doc_id
            \\JOIN collections col ON col.id = d.collection_id WHERE c.id = ?1
        );
        defer st.finalize();
        try st.bindI64(1, cid);
        try testing.expect(try st.step());
        hits[i] = .{
            .chunk_id = cid,
            .score = scores[i],
            .vec_rank = @intCast(i + 1),
            .fts_rank = null,
            .doc_id = st.columnI64(0),
            .idx = st.columnI64(1),
            .heading_path = try gpa.dupe(u8, st.columnText(2)),
            .text = try gpa.dupe(u8, st.columnText(3)),
            .n_tokens = st.columnI64(4),
            .rel_path = try gpa.dupe(u8, st.columnText(5)),
            .title = try gpa.dupe(u8, st.columnText(6)),
            .collection = try gpa.dupe(u8, st.columnText(7)),
        };
    }
    return .{
        .mode = .hybrid,
        .hits = hits,
        .dropped_terms = &.{},
        .fts_skipped = false,
        .vec_candidates = chunk_ids.len,
        .fts_candidates = 0,
    };
}

test "hits from one document collapse into a single group" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 1, 10, 100);

    // Three separate chunks of the same file.
    var results = try hitsFor(&db, &.{ 1, 4, 7 }, &.{ 0.9, 0.5, 0.3 });
    defer results.deinit(gpa);

    var p = try assembleNoContexts(&db, "q", &results, .{ .neighbors = 0 });
    defer p.deinit(gpa);

    try testing.expectEqual(@as(usize, 1), p.groups.len);
    try testing.expectEqualStrings("doc0.md", p.groups[0].rel_path);
    // The group takes the best of its hits' scores.
    try testing.expectEqual(@as(f64, 0.9), p.groups[0].score);
    try testing.expectEqual(@as(usize, 3), p.groups[0].spans.len);
}

test "neighbour expansion merges adjacent hits into one continuous span" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 1, 10, 100);

    // Chunks 3 and 5: with one neighbour each the ranges become 2-4 and 4-6,
    // which overlap and must come back as a single 2-6 span, not two fragments
    // with chunk 4 duplicated.
    var results = try hitsFor(&db, &.{ 4, 6 }, &.{ 0.9, 0.8 });
    defer results.deinit(gpa);

    var p = try assembleNoContexts(&db, "q", &results, .{ .neighbors = 1 });
    defer p.deinit(gpa);

    try testing.expectEqual(@as(usize, 1), p.groups.len);
    try testing.expectEqual(@as(usize, 1), p.groups[0].spans.len);
    const span = p.groups[0].spans[0];
    try testing.expectEqual(@as(i64, 2), span.first_idx);
    try testing.expectEqual(@as(i64, 6), span.last_idx);
    // Chunk 4 appears exactly once despite being in both ranges.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, span.text, "chunk 4 body"));
}

test "distant hits stay separate spans rather than swallowing the gap" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 1, 20, 100);

    var results = try hitsFor(&db, &.{ 1, 15 }, &.{ 0.9, 0.8 });
    defer results.deinit(gpa);

    var p = try assembleNoContexts(&db, "q", &results, .{ .neighbors = 1 });
    defer p.deinit(gpa);

    try testing.expectEqual(@as(usize, 2), p.groups[0].spans.len);
    // Nothing from the middle of the document leaked in.
    try testing.expect(std.mem.indexOf(u8, p.groups[0].spans[0].text, "chunk 8") == null);
}

test "documents are ordered by their best score" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 3, 3, 100);

    // doc2's chunk scores highest, even though doc0's hit comes first.
    var results = try hitsFor(&db, &.{ 1, 7 }, &.{ 0.4, 0.95 });
    defer results.deinit(gpa);

    var p = try assembleNoContexts(&db, "q", &results, .{ .neighbors = 0 });
    defer p.deinit(gpa);

    try testing.expectEqual(@as(usize, 2), p.groups.len);
    try testing.expectEqualStrings("doc2.md", p.groups[0].rel_path);
    try testing.expectEqualStrings("doc0.md", p.groups[1].rel_path);
}

test "the budget keeps the best documents and lists what it dropped" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 4, 1, 500);

    var results = try hitsFor(&db, &.{ 1, 2, 3, 4 }, &.{ 0.9, 0.8, 0.7, 0.6 });
    defer results.deinit(gpa);

    // Room for two documents (500 tokens each plus overhead), not four.
    var p = try assembleNoContexts(&db, "q", &results, .{
        .neighbors = 0,
        .budget_tokens = 1100,
        .per_doc_overhead_tokens = 20,
    });
    defer p.deinit(gpa);

    try testing.expectEqual(@as(usize, 2), p.groups.len);
    try testing.expectEqualStrings("doc0.md", p.groups[0].rel_path);
    try testing.expect(p.total_tokens <= 1100);

    // The rest are named, not silently gone.
    try testing.expectEqual(@as(usize, 2), p.omitted.len);
    try testing.expectEqualStrings("doc2.md", p.omitted[0].rel_path);
    try testing.expectEqualStrings("doc3.md", p.omitted[1].rel_path);
}

test "a single document larger than the budget keeps its strongest span" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 1, 10, 300);

    var results = try hitsFor(&db, &.{ 1, 5, 9 }, &.{ 0.9, 0.8, 0.7 });
    defer results.deinit(gpa);

    var p = try assembleNoContexts(&db, "q", &results, .{
        .neighbors = 0,
        .budget_tokens = 700,
        .per_doc_overhead_tokens = 20,
    });
    defer p.deinit(gpa);

    // Trailing spans are dropped rather than the whole document: a long document
    // whose first span answers the question should still contribute.
    try testing.expectEqual(@as(usize, 1), p.groups.len);
    try testing.expect(p.groups[0].spans.len < 3);
    try testing.expect(p.total_tokens <= 700);
}

test "an empty result set renders without claiming to have found anything" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 1, 1, 10);

    var results = try hitsFor(&db, &.{}, &.{});
    defer results.deinit(gpa);

    var p = try assembleNoContexts(&db, "q", &results, .{});
    defer p.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), p.groups.len);

    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try pack.renderMarkdown(&w, &p);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "No relevant documents") != null);
}

test "markdown output states the token count as approximate" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 2, 2, 50);

    var results = try hitsFor(&db, &.{ 1, 3 }, &.{ 0.9, 0.5 });
    defer results.deinit(gpa);
    var p = try assembleNoContexts(&db, "how does it work", &results, .{ .neighbors = 0 });
    defer p.deinit(gpa);

    var buf: [8192]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try pack.renderMarkdown(&w, &p);
    const out = w.buffered();

    try testing.expect(std.mem.indexOf(u8, out, "# Context for: how does it work") != null);
    try testing.expect(std.mem.indexOf(u8, out, "## docs/doc0.md") != null);
    // The budget is counted with the retrieval model's tokenizer, not the one
    // that will read this. Saying so is the honest part.
    try testing.expect(std.mem.indexOf(u8, out, "approx") != null);
}

test "json output is parseable and carries the same documents" {
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 2, 2, 50);

    var results = try hitsFor(&db, &.{ 1, 3 }, &.{ 0.9, 0.5 });
    defer results.deinit(gpa);
    var p = try assembleNoContexts(&db, "q", &results, .{ .neighbors = 0 });
    defer p.deinit(gpa);

    var buf: [16384]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try pack.renderJson(&w, &p);

    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, w.buffered(), .{});
    defer parsed.deinit();
    const docs = parsed.value.object.get("documents").?.array;
    try testing.expectEqual(@as(usize, 2), docs.items.len);
    try testing.expect(parsed.value.object.get("total_tokens").?.integer > 0);
}

test "an empty pack says whether the corpus was silent or the budget was" {
    // These are different facts and the markdown gave them the same sentence.
    // Measured on the real corpus: five qlit documents matched, every one was
    // dropped whole for exceeding a 500-token budget, and the output read "No
    // relevant documents found." The json had said `omitted: [all five]` the
    // whole time. Paired with `--path`, that sentence reads as "this project has
    // nothing about it" — a conclusion someone acts on.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 2, 2, 400);

    var results = try hitsFor(&db, &.{ 1, 3 }, &.{ 0.9, 0.5 });
    defer results.deinit(gpa);

    // Small enough that the per-document ceiling admits nothing.
    var p = try assembleNoContexts(&db, "q", &results, .{ .budget_tokens = 20, .neighbors = 0 });
    defer p.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), p.groups.len);
    try testing.expect(p.omitted.len != 0);

    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try pack.renderMarkdown(&w, &p);
    const out = w.buffered();

    try testing.expect(std.mem.indexOf(u8, out, "none fitted the 20-token budget") != null);
    // `omitted` carries the rel_path alone, where a rendered document heading
    // carries `collection/rel_path`. Asserted as it is rather than as it reads.
    try testing.expect(std.mem.indexOf(u8, out, "doc0.md") != null);
    try testing.expect(std.mem.indexOf(u8, out, "Raise --budget") != null);
    // The other sentence must not appear: it is the claim being corrected.
    try testing.expect(std.mem.indexOf(u8, out, "No relevant documents") == null);
}

/// A pack rendered directly, and the same pack rendered after a round trip
/// through the json the daemon sends. Byte-identical or the transports have
/// drifted.
fn assertRoundTripRenders(p: *const pack.Pack) !void {
    var direct_buf: [16384]u8 = undefined;
    var direct = std.Io.Writer.fixed(&direct_buf);
    try pack.renderMarkdown(&direct, p);

    var wire_buf: [16384]u8 = undefined;
    var wire = std.Io.Writer.fixed(&wire_buf);
    try pack.renderJson(&wire, p);

    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, wire.buffered(), .{});
    defer parsed.deinit();

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var rebuilt = try pack.fromJson(arena_state.allocator(), parsed.value);

    var round_buf: [16384]u8 = undefined;
    var round = std.Io.Writer.fixed(&round_buf);
    try pack.renderMarkdown(&round, &rebuilt);

    try testing.expectEqualStrings(direct.buffered(), round.buffered());
}

test "the daemon's json renders to the same markdown as the pack it came from" {
    // The regression this exists for: `recall` and `query` each hand-wrote a
    // markdown renderer for the daemon's json, and all three copies drifted from
    // `renderMarkdown` — `### x.md` for `## memory/x.md`, a missing
    // `omitted`/`tokens` footer, an `omitted` line without its scores. Every one
    // of them was on the daemon side, which is the side that normally runs, so
    // the drift was invisible from the path anyone tested.
    //
    // Asserting the bytes rather than the fields is the point: a renderer added
    // later can only diverge by making this fail.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 2, 2, 50);

    var results = try hitsFor(&db, &.{ 1, 3 }, &.{ 0.9, 0.5 });
    defer results.deinit(gpa);
    var p = try assembleNoContexts(&db, "q", &results, .{ .neighbors = 0 });
    defer p.deinit(gpa);

    try testing.expect(p.groups.len != 0);
    try assertRoundTripRenders(&p);
}

test "a pack emptied by the budget survives the round trip with its omitted list" {
    // The footer was the half the daemon path dropped, so the empty-but-omitted
    // shape is the one worth pinning: it renders entirely out of the fields that
    // used not to be read back.
    var db = try store.open(":memory:", .read_write);
    defer db.close();
    try seed(&db, 2, 2, 400);

    var results = try hitsFor(&db, &.{ 1, 3 }, &.{ 0.9, 0.5 });
    defer results.deinit(gpa);
    var p = try assembleNoContexts(&db, "q", &results, .{ .budget_tokens = 20, .neighbors = 0 });
    defer p.deinit(gpa);

    try testing.expectEqual(@as(usize, 0), p.groups.len);
    try testing.expect(p.omitted.len != 0);
    try assertRoundTripRenders(&p);
}
