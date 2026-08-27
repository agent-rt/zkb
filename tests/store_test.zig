//! Storage-layer invariants.
//!
//! The one that matters: chunks, fts_chunks and vec_chunks must never drift
//! apart. Virtual tables are not reached by ON DELETE CASCADE, so a missed
//! delete produces search hits for content that no longer exists — a wrong
//! answer with no error attached. These tests are the guard.

const std = @import("std");
const zkb = @import("zkb");
const sqlite = zkb.sqlite;
const schema = zkb.schema;
const store = zkb.store;

const testing = std.testing;

fn openMem() !sqlite.Db {
    return store.open(":memory:", .read_write);
}

const dim: usize = @intCast(schema.embedding_dim);

fn dummyVector(seed: u8) [dim]f32 {
    var v: [dim]f32 = @splat(0);
    v[@as(usize, seed) % dim] = 1.0;
    return v;
}

fn addChunks(s: *store.Store, collection_id: i64, doc_id: i64, n: usize) !void {
    for (0..n) |i| {
        var vec = dummyVector(@intCast(i));
        _ = try s.insertChunk(collection_id, doc_id, .{
            .idx = @intCast(i),
            .heading_path = "doc > section",
            .byte_start = @intCast(i * 100),
            .byte_end = @intCast((i + 1) * 100),
            .n_tokens = 42,
            .text = "reciprocal rank fusion keeps the weights out of it",
        }, &vec);
    }
}

test "migrate is idempotent and records its versions" {
    var db = try openMem();
    defer db.close();

    try schema.migrate(&db); // second run must be a no-op, not an error

    var buf: [32]u8 = undefined;
    var expect: [8]u8 = undefined;
    const want = try std.fmt.bufPrint(&expect, "{d}", .{schema.schema_version});
    try testing.expectEqualStrings(want, (try schema.getMeta(&db, "schema_version", &buf)).?);
    try testing.expectEqualStrings("1024", (try schema.getMeta(&db, "embedding_dim", &buf)).?);
    // chunker_version must be recorded: a chunker change invalidates vectors
    // just as a model change does, and nothing else would notice.
    try testing.expect((try schema.getMeta(&db, "chunker_version", &buf)) != null);
}

test "insertChunk writes all three tables, deleteChunks clears all three" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const did = try s.upsertDocContent(cid, "a.md", "sha-a", 10, 1000);
    try addChunks(&s, cid, did, 5);

    var c = try s.counts();
    try testing.expectEqual(@as(i64, 5), c.chunks);
    try testing.expectEqual(@as(i64, 5), c.fts_rows);
    try testing.expectEqual(@as(i64, 5), c.vec_rows);

    try s.deleteChunks(did);
    c = try s.counts();
    try testing.expectEqual(@as(i64, 0), c.chunks);
    try testing.expectEqual(@as(i64, 0), c.fts_rows);
    try testing.expectEqual(@as(i64, 0), c.vec_rows);
}

test "deleteDoc leaves no searchable residue" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const keep = try s.upsertDocContent(cid, "keep.md", "sha-keep", 10, 1000);
    const drop = try s.upsertDocContent(cid, "drop.md", "sha-drop", 10, 1000);
    try addChunks(&s, cid, keep, 3);
    try addChunks(&s, cid, drop, 3);

    try s.deleteDoc(drop);

    const c = try s.counts();
    try testing.expectEqual(@as(i64, 1), c.docs);
    try testing.expectEqual(@as(i64, 3), c.chunks);
    try testing.expectEqual(@as(i64, 3), c.fts_rows);
    try testing.expectEqual(@as(i64, 3), c.vec_rows);

    // The deleted doc must not be reachable through either index.
    const fts_hits = (try db.queryI64(
        \\SELECT count(*) FROM fts_chunks
        \\WHERE fts_chunks MATCH '"reciprocal"'
        \\  AND rowid NOT IN (SELECT id FROM chunks)
    )) orelse -1;
    try testing.expectEqual(@as(i64, 0), fts_hits);

    const orphan_vecs = (try db.queryI64(
        "SELECT count(*) FROM vec_chunks WHERE chunk_id NOT IN (SELECT id FROM chunks)",
    )) orelse -1;
    try testing.expectEqual(@as(i64, 0), orphan_vecs);
}

test "deleteDoc clears the links keyed on it, and the links pointing at it" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const keep = try s.upsertDocContent(cid, "keep.md", "sha-keep", 10, 1000);
    const drop = try s.upsertDocContent(cid, "drop.md", "sha-drop", 10, 1000);
    try addChunks(&s, cid, keep, 1);
    try addChunks(&s, cid, drop, 1);

    // The doc being deleted emits a link, and is itself the target of one. Both
    // sides matter and only the first has a foreign key to catch it.
    try db.exec("INSERT INTO links(doc_id, chunk_id, kind, raw, target_doc_id) VALUES (2, NULL, 'wiki', 'zkb://x', NULL)");
    try db.exec("INSERT INTO links(doc_id, chunk_id, kind, raw, target_doc_id) VALUES (1, NULL, 'wiki', 'zkb://drop', 2)");

    // Without the links delete this fails on the foreign key, after the chunks
    // are already gone.
    try s.deleteDoc(drop);

    const c = try s.counts();
    try testing.expectEqual(@as(i64, 1), c.docs);

    const outbound = (try db.queryI64("SELECT count(*) FROM links WHERE doc_id = 2")) orelse -1;
    try testing.expectEqual(@as(i64, 0), outbound);

    // The inbound link survives — the document that wrote it is still there —
    // but it must no longer claim to resolve, or it would alias the next doc to
    // reuse id 2.
    const inbound = (try db.queryI64("SELECT count(*) FROM links WHERE doc_id = 1")) orelse -1;
    try testing.expectEqual(@as(i64, 1), inbound);
    const dangling = (try db.queryI64("SELECT count(*) FROM links WHERE target_doc_id = 2")) orelse -1;
    try testing.expectEqual(@as(i64, 0), dangling);
}

test "re-index of a changed doc replaces chunks rather than accumulating" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const did = try s.upsertDocContent(cid, "a.md", "sha-v1", 10, 1000);
    try addChunks(&s, cid, did, 4);
    try s.markIndexed(did, 4, 1000);

    // Content changed: same path, new sha. upsert must reset indexed_at.
    const again = try s.upsertDocContent(cid, "a.md", "sha-v2", 20, 2000);
    try testing.expectEqual(did, again);
    const row = (try s.findDoc(cid, "a.md")).?;
    try testing.expectEqual(@as(?i64, null), row.indexed_at);

    try s.deleteChunks(did);
    try addChunks(&s, cid, did, 2);

    const c = try s.counts();
    try testing.expectEqual(@as(i64, 2), c.chunks);
    try testing.expectEqual(@as(i64, 2), c.fts_rows);
    try testing.expectEqual(@as(i64, 2), c.vec_rows);
}

test "rename keeps the doc id and its vectors" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const did = try s.upsertDocContent(cid, "old.md", "sha-x", 10, 1000);
    try addChunks(&s, cid, did, 3);
    try s.markIndexed(did, 3, 1000);

    // Same content at a different path *may* be a rename; whether it is depends on
    // the old path still existing, which only the scanner can answer. This layer
    // hands back the path so it can (see tests/scan_test.zig).
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const found = (try s.findDocByShaExcludingPath(arena.allocator(), cid, "sha-x", "new.md")).?;
    try testing.expectEqual(did, found.id);
    try testing.expectEqualStrings("old.md", found.rel_path);

    try s.moveDoc(did, "new.md", 2000);

    try testing.expect((try s.findDoc(cid, "old.md")) == null);
    const moved = (try s.findDoc(cid, "new.md")).?;
    try testing.expectEqual(did, moved.id);
    // Still indexed: re-embedding identical bytes would be pure waste.
    try testing.expect(moved.indexed_at != null);

    const c = try s.counts();
    try testing.expectEqual(@as(i64, 3), c.chunks);
    try testing.expectEqual(@as(i64, 3), c.vec_rows);
}

test "markFailed zeroes the chunk count so a partial index cannot look complete" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const did = try s.upsertDocContent(cid, "a.md", "sha-a", 10, 1000);
    try addChunks(&s, cid, did, 3);

    // Simulating a mid-document embed failure: drop the partial work, then mark.
    try s.deleteChunks(did);
    try s.markFailed(did, "embed failed on chunk 3");

    const c = try s.counts();
    try testing.expectEqual(@as(i64, 0), c.chunks);
    try testing.expectEqual(@as(i64, 1), c.failed);
    try testing.expectEqual(@as(i64, 0), c.pending);
}

// ---------------------------------------------------------------------------
// v1 -> v2 migration: the FTS tokenizer swap.
//
// The property worth guarding is that this is *cheap*: the keyword index is
// rebuilt from chunks.text while the vectors stay untouched. If a future change
// makes the migration drop vectors, a full re-index (minutes) silently replaces
// a rebuild (seconds), and nothing else would notice.
// ---------------------------------------------------------------------------

/// Build a database in the v1 shape: same tables, but the old trigram tokenizer.
fn openV1() !sqlite.Db {
    var db = try sqlite.Db.open(":memory:", .read_write);
    errdefer db.close();
    try db.exec(
        \\CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
        \\CREATE TABLE collections (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
        \\  root TEXT NOT NULL, created_at INTEGER NOT NULL) STRICT;
        \\CREATE TABLE docs (id INTEGER PRIMARY KEY, collection_id INTEGER NOT NULL
        \\  REFERENCES collections(id), rel_path TEXT NOT NULL, title TEXT,
        \\  frontmatter TEXT, content_sha TEXT NOT NULL, size INTEGER NOT NULL,
        \\  mtime_ms INTEGER NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0,
        \\  indexed_at INTEGER, index_error TEXT, UNIQUE(collection_id, rel_path)) STRICT;
        \\CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL
        \\  REFERENCES docs(id), idx INTEGER NOT NULL, heading_path TEXT,
        \\  byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL,
        \\  n_tokens INTEGER NOT NULL, text TEXT NOT NULL, UNIQUE(doc_id, idx)) STRICT;
        \\CREATE VIRTUAL TABLE fts_chunks USING fts5(text, heading_path,
        \\  content='', contentless_delete=1, tokenize='trigram case_sensitive 0');
        \\CREATE VIRTUAL TABLE vec_chunks USING vec0(collection_id INTEGER partition key,
        \\  chunk_id INTEGER PRIMARY KEY, embedding FLOAT[1024] distance_metric=cosine);
        \\INSERT INTO meta(key,value) VALUES ('schema_version','1'),
        \\  ('chunker_version','1'), ('embedding_dim','1024');
    );
    return db;
}

test "v1 to v2 rebuilds the FTS index from chunks.text and keeps every vector" {
    var db = try openV1();
    defer db.close();
    var s = store.Store.init(&db);

    // Seeded with raw v1 SQL, not `ensureCollection`: the Store API writes the
    // *current* schema (v4 added `collections.kind`), so a migration test that
    // builds its "old" database through today's API stops testing a migration.
    try db.exec(
        "INSERT INTO collections(id, name, root, created_at) VALUES (1, 'docs', '/tmp/docs', 1000);",
    );
    const cid: i64 = 1;
    const did = try s.upsertDocContent(cid, "a.md", "sha-a", 10, 1000);
    {
        var vec = dummyVector(3);
        _ = try s.insertChunk(cid, did, .{
            .idx = 0,
            .heading_path = "doc > 检索",
            .byte_start = 0,
            .byte_end = 100,
            .n_tokens = 42,
            .text = "检索融合使用 RRF 而不是加权和",
        }, &vec);
    }

    // Under trigram, this two-character word matches nothing.
    try testing.expectEqual(@as(?i64, 0), try db.queryI64(
        \\SELECT count(*) FROM fts_chunks WHERE fts_chunks MATCH '"融合"'
    ));
    const vec_before = (try db.queryI64("SELECT count(*) FROM vec_chunks")).?;

    try schema.migrate(&db);

    var buf: [32]u8 = undefined;
    var expect: [8]u8 = undefined;
    const want = try std.fmt.bufPrint(&expect, "{d}", .{schema.schema_version});
    try testing.expectEqualStrings(want, (try schema.getMeta(&db, "schema_version", &buf)).?);

    // Same content, now matchable.
    try testing.expectEqual(@as(?i64, 1), try db.queryI64(
        \\SELECT count(*) FROM fts_chunks WHERE fts_chunks MATCH '"融合"'
    ));
    // Vectors were never touched — that is what makes the migration seconds.
    try testing.expectEqual(@as(?i64, vec_before), try db.queryI64("SELECT count(*) FROM vec_chunks"));

    const c = try s.counts();
    try testing.expectEqual(c.chunks, c.fts_rows);
    try testing.expectEqual(c.chunks, c.vec_rows);
}

test "migrating twice is a no-op" {
    var db = try openV1();
    defer db.close();
    try schema.migrate(&db);
    try schema.migrate(&db);
}

test "upsertCollection updates the root instead of ignoring it" {
    var db = try openMem();
    defer db.close();
    try schema.migrate(&db);
    var s = store.Store.init(&db);

    const first = try s.upsertCollection("notes", "/a", .documents, null, null, 1);
    // Re-registering the same name at a new root used to return the existing row
    // untouched, so `zkb index --root NEW --collection notes` reported success and
    // kept scanning the old place.
    const second = try s.upsertCollection("notes", "/b", .documents, null, null, 2);
    try testing.expectEqual(first, second);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const rows = try s.listCollections(arena.allocator());
    try testing.expectEqual(@as(usize, 1), rows.len);
    try testing.expectEqualStrings("/b", rows[0].root);
}

test "a null filter keeps what is stored, so a root can be moved alone" {
    var db = try openMem();
    defer db.close();
    try schema.migrate(&db);
    var s = store.Store.init(&db);

    _ = try s.upsertCollection("m", "/a", .documents, ".md", "*/memory/*.md", 1);
    _ = try s.upsertCollection("m", "/b", .documents, null, null, 2);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const rows = try s.listCollections(arena.allocator());
    try testing.expectEqualStrings("/b", rows[0].root);
    try testing.expectEqualStrings(".md", rows[0].extensions.?);
    try testing.expectEqualStrings("*/memory/*.md", rows[0].include.?);
}

test "re-registering keeps the kind, which is what supplies the built-in filters" {
    var db = try openMem();
    defer db.close();
    try schema.migrate(&db);
    var s = store.Store.init(&db);

    // A memory collection skips `archive/` because of its kind, not because of
    // anything stored on the row. `zkb index` always says `.documents`, so this
    // update used to turn a memory collection into a documents one — after
    // which `recall` answered from memories that had been deliberately retired,
    // and nothing said so. agent-rt/zkb#1.
    _ = try s.upsertCollection("mem", "/a", .memory, null, null, 1);
    _ = try s.upsertCollection("mem", "/b", .documents, null, null, 2);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const row = (try s.collectionByName(arena.allocator(), "mem")).?;
    try testing.expectEqual(store.Store.Kind.memory, row.kind);
    // The root it *was* asked to change still changes.
    try testing.expectEqualStrings("/b", row.root);
}

test "collectionByName answers about one collection, or about none" {
    var db = try openMem();
    defer db.close();
    try schema.migrate(&db);
    var s = store.Store.init(&db);

    _ = try s.upsertCollection("one", "/a", .documents, ".md", "keep/**", 1);
    _ = try s.upsertCollection("two", "/b", .records, null, null, 1);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = (try s.collectionByName(arena.allocator(), "one")).?;
    try testing.expectEqualStrings("/a", a.root);
    try testing.expectEqualStrings("keep/**", a.include.?);
    try testing.expectEqual(store.Store.Kind.records, (try s.collectionByName(arena.allocator(), "two")).?.kind);
    try testing.expect(try s.collectionByName(arena.allocator(), "nope") == null);
}

test "ensureCollectionKind corrects a kind that is already wrong" {
    var db = try openMem();
    defer db.close();
    try schema.migrate(&db);
    var s = store.Store.init(&db);

    // The state agent-rt/zkb#1 left behind: a memory collection recorded as a
    // documents one, so its built-in `archive/` exclusion is gone. Nothing
    // repaired it — every path that knew the right kind stopped at "the row
    // exists", which made the function's name a claim it did not keep.
    _ = try s.upsertCollection("memory", "/m", .documents, null, null, 1);
    const id = try s.ensureCollectionKind("memory", "/m", .memory, 2);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const row = (try s.collectionByName(arena.allocator(), "memory")).?;
    try testing.expectEqual(id, row.id);
    try testing.expectEqual(store.Store.Kind.memory, row.kind);
    // It repairs the kind and nothing else: the root is not its business.
    try testing.expectEqualStrings("/m", row.root);
}

test "correcting a kind requeues the documents the wrong kind mis-indexed" {
    var db = try openMem();
    defer db.close();
    try schema.migrate(&db);
    var s = store.Store.init(&db);

    // Repairing the row is not repairing the damage. `indexer.indexOne` writes
    // the per-chunk projections only when the collection's kind says to, so every
    // document indexed under a wrong kind lost its `rec_memory` row and would
    // never get it back on its own: the files did not change, so `scan` calls
    // them unchanged forever. Measured once for real — all 40 memories invisible
    // to `recall` for two days while `status` kept counting all 40.
    const cid = try s.upsertCollection("memory", "/m", .documents, null, null, 1);
    const did = try s.upsertDocContent(cid, "one.md", "sha", 10, 1);
    try s.markIndexed(did, 1, 1);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const before = try s.listPending(arena.allocator(), cid, 10);
    try testing.expectEqual(@as(usize, 0), before.items.len);

    _ = try s.ensureCollectionKind("memory", "/m", .memory, 2);

    const after = try s.listPending(arena.allocator(), cid, 10);
    try testing.expectEqual(@as(usize, 1), after.items.len);
    try testing.expectEqualStrings("one.md", after.items[0].rel_path);

    // And only when something was actually wrong. This runs on every `remember`
    // and every index pass, so a kind that already agrees must cost nothing —
    // otherwise the repair becomes a full re-embed on a normal command.
    try s.markIndexed(did, 1, 3);
    _ = try s.ensureCollectionKind("memory", "/m", .memory, 4);
    const idempotent = try s.listPending(arena.allocator(), cid, 10);
    try testing.expectEqual(@as(usize, 0), idempotent.items.len);
}

test "collections created before v7 have no filters, not empty ones" {
    var db = try openMem();
    defer db.close();
    try schema.migrate(&db);
    var s = store.Store.init(&db);

    // `ensureCollectionKind` is the pre-v7 path and writes neither column. The
    // difference matters: null resolves to the kind's defaults, while '' would be
    // a collection that matches no file at all.
    _ = try s.ensureCollectionKind("old", "/a", .documents, 1);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const rows = try s.listCollections(arena.allocator());
    try testing.expect(rows[0].extensions == null);
    try testing.expect(rows[0].include == null);
}

test "listCollections keeps every kind and its id" {
    var db = try openMem();
    defer db.close();
    try schema.migrate(&db);
    var s = store.Store.init(&db);

    const d = try s.upsertCollection("docs", "/d", .documents, null, null, 1);
    const m = try s.upsertCollection("memory", "/m", .memory, null, null, 1);
    const k = try s.upsertCollection("numbers", "/n", .records, null, null, 1);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const rows = try s.listCollections(arena.allocator());
    try testing.expectEqual(@as(usize, 3), rows.len);
    // Ordered by id, which is what makes the daemon's scan order stable.
    try testing.expectEqual(d, rows[0].id);
    try testing.expectEqual(m, rows[1].id);
    try testing.expectEqual(k, rows[2].id);
    try testing.expectEqual(store.Store.Kind.documents, rows[0].kind);
    try testing.expectEqual(store.Store.Kind.memory, rows[1].kind);
    try testing.expectEqual(store.Store.Kind.records, rows[2].kind);
}

test "deleteOrphanChunks clears all three tables and leaves live chunks alone" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const keep = try s.upsertDocContent(cid, "keep.md", "sha-keep", 10, 1000);
    const gone = try s.upsertDocContent(cid, "gone.md", "sha-gone", 10, 1000);
    try addChunks(&s, cid, keep, 3);
    try addChunks(&s, cid, gone, 4);

    // The corruption this exists for: the document row disappears without its
    // chunks going with it.
    //
    // Fabricating it needs foreign keys off, which is itself the finding — the
    // `chunks.doc_id REFERENCES docs(id)` constraint makes this unreachable on
    // any connection that has them on, and every writer in this codebase does
    // (`schema.migrate` sets the pragma before anything else, and has since the
    // first ingest commit). A real index still accumulated 1198 such chunks; the
    // path that produced them was never identified, which is exactly why the
    // sweep is unconditional rather than guarded by a guess about the cause.
    try db.exec("PRAGMA foreign_keys = OFF;");
    try db.exec("DELETE FROM docs WHERE rel_path = 'gone.md';");
    try db.exec("PRAGMA foreign_keys = ON;");

    var c = try s.counts();
    try testing.expectEqual(@as(i64, 7), c.chunks);
    try testing.expectEqual(@as(i64, 7), c.fts_rows);

    const removed = try s.deleteOrphanChunks(testing.allocator);
    try testing.expectEqual(@as(usize, 4), removed);

    // All three tables, or a search still matches text that is no longer there.
    c = try s.counts();
    try testing.expectEqual(@as(i64, 3), c.chunks);
    try testing.expectEqual(@as(i64, 3), c.fts_rows);
    try testing.expectEqual(@as(i64, 3), c.vec_rows);

    // The surviving document keeps every chunk it had.
    try testing.expectEqual(@as(?i64, 3), try db.queryI64("SELECT count(*) FROM chunks"));
    try testing.expectEqual(
        @as(?i64, keep),
        try db.queryI64("SELECT DISTINCT doc_id FROM chunks"),
    );
}

test "deleteOrphanChunks is a no-op on a consistent index" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const did = try s.upsertDocContent(cid, "a.md", "sha-a", 10, 1000);
    try addChunks(&s, cid, did, 3);

    try testing.expectEqual(@as(usize, 0), try s.deleteOrphanChunks(testing.allocator));
    try testing.expectEqual(@as(i64, 3), (try s.counts()).chunks);
}

test "deleting a document leaves no orphan behind" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const did = try s.upsertDocContent(cid, "a.md", "sha-a", 10, 1000);
    try addChunks(&s, cid, did, 3);
    try s.deleteDoc(did);

    // The forward guarantee: the sweep is for historical residue, not for
    // covering a delete path that leaks.
    try testing.expectEqual(@as(usize, 0), try s.deleteOrphanChunks(testing.allocator));
    try testing.expectEqual(@as(i64, 0), (try s.counts()).chunks);
}

test "a stale index entry costs its own result, not the whole search" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const keep = try s.upsertDocContent(cid, "keep.md", "sha-keep", 10, 1000);
    const gone = try s.upsertDocContent(cid, "gone.md", "sha-gone", 10, 1000);
    try addChunks(&s, cid, keep, 3);
    try addChunks(&s, cid, gone, 3);

    try db.exec("PRAGMA foreign_keys = OFF;");
    try db.exec("DELETE FROM docs WHERE rel_path = 'gone.md';");
    try db.exec("PRAGMA foreign_keys = ON;");

    // Both paths rank the orphaned chunks: they are still in fts_chunks and
    // vec_chunks, which is exactly why this used to be fatal. `hydrate` mapped
    // "no row" to `error.SqliteStep`, so one unresolvable candidate turned a
    // working query into `internal: search failed` — measured on a real index
    // where `zkb search "交接"` worked and `zkb search "i18n locales"` did not,
    // purely on which chunks landed in the top-k.
    var query_vec = dummyVector(0);
    var res = try zkb.hybrid.search(
        testing.allocator,
        &db,
        .hybrid,
        "fusion",
        &query_vec,
        cid,
        // No ceiling: these assert exactly what the index holds, and a breadth
        // cap would put a second reason between the rows and the count.
        .{ .top_k = 10, .max_per_doc = null },
    );
    defer res.deinit(testing.allocator);

    // Three live chunks come back; the three stale ones are simply absent.
    try testing.expectEqual(@as(usize, 3), res.hits.len);
    for (res.hits) |h| {
        try testing.expectEqual(keep, h.doc_id);
        try testing.expectEqualStrings("keep.md", h.rel_path);
    }
}

test "a search over an index with nothing but stale entries returns empty, not an error" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const gone = try s.upsertDocContent(cid, "gone.md", "sha-gone", 10, 1000);
    try addChunks(&s, cid, gone, 3);

    try db.exec("PRAGMA foreign_keys = OFF;");
    try db.exec("DELETE FROM docs WHERE rel_path = 'gone.md';");
    try db.exec("PRAGMA foreign_keys = ON;");

    // The degenerate case the shrink has to handle: every hit is dropped, so the
    // slice is realloc'd to zero. Getting this wrong frees the wrong allocation.
    var query_vec = dummyVector(0);
    var res = try zkb.hybrid.search(
        testing.allocator,
        &db,
        .hybrid,
        "fusion",
        &query_vec,
        cid,
        // No ceiling: these assert exactly what the index holds, and a breadth
        // cap would put a second reason between the rows and the count.
        .{ .top_k = 10, .max_per_doc = null },
    );
    defer res.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), res.hits.len);
}

test "deleteCollection leaves no residue in any of the three tables" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const keep = try s.ensureCollection("keep", "/tmp/keep", 1000);
    const drop = try s.ensureCollection("drop", "/tmp/drop", 1000);
    const k1 = try s.upsertDocContent(keep, "a.md", "sha-a", 10, 1000);
    const d1 = try s.upsertDocContent(drop, "b.md", "sha-b", 10, 1000);
    const d2 = try s.upsertDocContent(drop, "c.md", "sha-c", 10, 1000);
    try addChunks(&s, keep, k1, 2);
    try addChunks(&s, drop, d1, 3);
    try addChunks(&s, drop, d2, 4);

    try testing.expectEqual(@as(i64, 9), (try s.counts()).chunks);

    const n = try s.deleteCollection(testing.allocator, drop);
    try testing.expectEqual(@as(usize, 2), n);

    // The collection row is gone, and so is every chunk it owned — in all three
    // tables, or search would still match text whose document no longer exists.
    try testing.expectEqual(@as(?i64, null), try s.findCollection("drop"));
    const c = try s.counts();
    try testing.expectEqual(@as(i64, 2), c.chunks);
    try testing.expectEqual(@as(i64, 2), c.fts_rows);
    try testing.expectEqual(@as(i64, 2), c.vec_rows);

    // And no orphaned chunks, which is the residue this whole class of bug leaves.
    try testing.expectEqual(@as(usize, 0), try s.deleteOrphanChunks(testing.allocator));

    // The other collection is untouched.
    try testing.expectEqual(@as(?i64, keep), try s.findCollection("keep"));
    try testing.expectEqual(@as(?i64, 2), try db.queryI64("SELECT count(*) FROM chunks"));
}

test "a path filter scopes retrieval without splitting the corpus" {
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const inside = try s.upsertDocContent(cid, "agents/handoffs/h1.md", "sha-h", 10, 1000);
    const outside = try s.upsertDocContent(cid, "projects/spec.md", "sha-s", 10, 1000);
    try addChunks(&s, cid, inside, 2);
    try addChunks(&s, cid, outside, 2);

    var query_vec = dummyVector(0);

    // Unscoped: both documents are reachable.
    {
        var res = try zkb.hybrid.search(testing.allocator, &db, .hybrid, "fusion", &query_vec, cid, .{ .top_k = 10, .max_per_doc = null });
        defer res.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 4), res.hits.len);
    }

    // Scoped: only the subtree, and exactly its chunks rather than whatever
    // survived a post-filter on a global top-k.
    {
        var res = try zkb.hybrid.search(testing.allocator, &db, .hybrid, "fusion", &query_vec, cid, .{
            .top_k = 10,
            .path = "agents/handoffs/**",
            .max_per_doc = null,
        });
        defer res.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 2), res.hits.len);
        for (res.hits) |h| try testing.expectEqual(inside, h.doc_id);
    }

    // A pattern matching nothing returns nothing rather than falling back to
    // everything — the empty-list inversion that cost a release once already.
    {
        var res = try zkb.hybrid.search(testing.allocator, &db, .hybrid, "fusion", &query_vec, cid, .{
            .top_k = 10,
            .path = "nowhere/**",
            .max_per_doc = null,
        });
        defer res.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 0), res.hits.len);
    }
}

/// Every chunk gets the *same* vector, so a whole document can be made to rank
/// ahead of everything else.
///
/// `addChunks` varies the vector by chunk index, which makes chunk 0 of every
/// document the best match and spreads the top-k across documents by
/// construction — the opposite of the shape under test here, and a test written
/// on it would pass without the ceiling ever running.
fn addChunksSharingVector(
    s: *store.Store,
    collection_id: i64,
    doc_id: i64,
    n: usize,
    seed: u8,
) !void {
    for (0..n) |i| {
        var vec = dummyVector(seed);
        _ = try s.insertChunk(collection_id, doc_id, .{
            .idx = @intCast(i),
            .heading_path = "doc > section",
            .byte_start = @intCast(i * 100),
            .byte_end = @intCast((i + 1) * 100),
            .n_tokens = 42,
            .text = "reciprocal rank fusion keeps the weights out of it",
        }, &vec);
    }
}

test "one long document cannot take every slot in the top-k" {
    // Measured on ~/docs before the ceiling existed: `zkb search "emqx" -k 10`
    // returned two files, nine of the ten hits from one 20-chunk guide. `-k N`
    // reads as "N results" and was answering "N chunks".
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const long = try s.upsertDocContent(cid, "long.md", "sha-long", 10, 1000);
    try addChunksSharingVector(&s, cid, long, 20, 0);
    for (0..6) |i| {
        var path_buf: [32]u8 = undefined;
        var sha_buf: [32]u8 = undefined;
        const did = try s.upsertDocContent(
            cid,
            try std.fmt.bufPrint(&path_buf, "short{d}.md", .{i}),
            try std.fmt.bufPrint(&sha_buf, "sha-short{d}", .{i}),
            10,
            1000,
        );
        try addChunksSharingVector(&s, cid, did, 1, 5);
    }

    var query_vec = dummyVector(0);

    // The control, and the reason this test is not vacuous: with no ceiling the
    // long document really does take the whole top-k.
    {
        var res = try zkb.hybrid.search(testing.allocator, &db, .hybrid, "fusion", &query_vec, cid, .{
            .top_k = 10,
            .max_per_doc = null,
        });
        defer res.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 10), res.hits.len);
        try testing.expectEqual(@as(usize, 1), distinctDocs(res.hits));
    }

    {
        var res = try zkb.hybrid.search(testing.allocator, &db, .hybrid, "fusion", &query_vec, cid, .{
            .top_k = 10,
            .max_per_doc = zkb.hybrid.search_max_per_doc,
        });
        defer res.deinit(testing.allocator);
        // Still ten results — the ceiling decides which, never how many.
        try testing.expectEqual(@as(usize, 10), res.hits.len);
        // Six short documents plus the long one: every document that matched.
        try testing.expectEqual(@as(usize, 7), distinctDocs(res.hits));
    }
}

test "the ceiling never costs a result when the corpus is smaller than -k" {
    // The failure the second pass exists to prevent. Two documents, five chunks
    // each, a ceiling of three: refusing the surplus would answer `-k 10` with
    // six hits and call the other four unavailable.
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    for (0..2) |i| {
        var path_buf: [32]u8 = undefined;
        var sha_buf: [32]u8 = undefined;
        const did = try s.upsertDocContent(
            cid,
            try std.fmt.bufPrint(&path_buf, "doc{d}.md", .{i}),
            try std.fmt.bufPrint(&sha_buf, "sha-doc{d}", .{i}),
            10,
            1000,
        );
        try addChunksSharingVector(&s, cid, did, 5, 0);
    }

    var query_vec = dummyVector(0);
    var res = try zkb.hybrid.search(testing.allocator, &db, .hybrid, "fusion", &query_vec, cid, .{
        .top_k = 10,
        .max_per_doc = zkb.hybrid.search_max_per_doc,
    });
    defer res.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 10), res.hits.len);
    try testing.expectEqual(@as(usize, 2), distinctDocs(res.hits));
}

fn distinctDocs(hits: []const zkb.hybrid.Hit) usize {
    var seen: [64]i64 = undefined;
    var n: usize = 0;
    outer: for (hits) |h| {
        for (seen[0..n]) |d| {
            if (d == h.doc_id) continue :outer;
        }
        seen[n] = h.doc_id;
        n += 1;
    }
    return n;
}

test "one exit code per error, whichever path produced it" {
    const EC = zkb.proto.ErrorCode;

    // The contract the CI smoke test asserts, pinned here so a new error code has
    // to choose deliberately rather than inherit whatever the last one used.
    try testing.expectEqual(@as(u8, 2), EC.bad_request.exitCode());
    try testing.expectEqual(@as(u8, 3), EC.not_found.exitCode());
    try testing.expectEqual(@as(u8, 3), EC.indexing.exitCode());
    try testing.expectEqual(@as(u8, 4), EC.model_mismatch.exitCode());
    try testing.expectEqual(@as(u8, 4), EC.model_unavailable.exitCode());
    try testing.expectEqual(@as(u8, 3), EC.internal.exitCode());

    // Clients see the code as text off the wire.
    try testing.expectEqual(@as(u8, 2), EC.exitCodeOf("bad_request"));
    try testing.expectEqual(@as(u8, 4), EC.exitCodeOf("model_mismatch"));

    // A name from a newer daemon is not something this client can act on, so it
    // gets "nothing to act on" rather than "your request was wrong".
    try testing.expectEqual(@as(u8, 3), EC.exitCodeOf("something_new"));

    // Every code has to be mapped: a new variant left out of the switch would not
    // compile, and this loop keeps the enum and the contract in one test.
    inline for (std.meta.fields(EC)) |f| {
        const code: EC = @enumFromInt(f.value);
        const n = code.exitCode();
        try testing.expect(n >= 1 and n <= 4);
    }
}

test "a wildcard-free --path names a place, and still names a file" {
    // Measured against the forms the argument actually gets typed in:
    // `projects/qlit/**`, `projects/qlit/*` and even a stray `/projects/qlit/**`
    // all worked, while `projects/qlit` — the plainest one — returned nothing.
    // Not an error, nothing: identical output to "that project has nothing about
    // this", which is the answer a reader will believe.
    var db = try openMem();
    defer db.close();
    var s = store.Store.init(&db);

    const cid = try s.ensureCollection("docs", "/tmp/docs", 1000);
    const inside = try s.upsertDocContent(cid, "projects/qlit/PLAN.md", "sha-p", 10, 1000);
    const deeper = try s.upsertDocContent(cid, "projects/qlit/notes/a.md", "sha-a", 10, 1000);
    const sibling = try s.upsertDocContent(cid, "projects/qlit-old/PLAN.md", "sha-o", 10, 1000);
    const root_index = try s.upsertDocContent(cid, "index.md", "sha-i", 10, 1000);
    for ([_]i64{ inside, deeper, sibling, root_index }) |d| try addChunks(&s, cid, d, 1);

    var query_vec = dummyVector(0);
    const Case = struct { pattern: []const u8, want: []const i64 };
    for ([_]Case{
        // Every spelling of "that directory" selects the subtree, including the
        // shapes that arrive pasted out of a shell.
        .{ .pattern = "projects/qlit/**", .want = &.{ inside, deeper } },
        .{ .pattern = "projects/qlit", .want = &.{ inside, deeper } },
        .{ .pattern = "projects/qlit/", .want = &.{ inside, deeper } },
        .{ .pattern = "./projects/qlit", .want = &.{ inside, deeper } },
        .{ .pattern = "/projects/qlit", .want = &.{ inside, deeper } },
        // The subtree reading is an addition, not a rewrite into `<pat>/**`:
        // a bare filename still means that file.
        .{ .pattern = "index.md", .want = &.{root_index} },
        // The prefix has to end on a component boundary, or `qlit` would drag in
        // `qlit-old` and the filter would quietly be wrong rather than empty.
        .{ .pattern = "projects/ql", .want = &.{} },
        // A typo stays empty. Guessing at one would be the same failure as the
        // silent no-match, only louder about the wrong answer.
        .{ .pattern = "projcts/qlit", .want = &.{} },
    }) |c| {
        var res = try zkb.hybrid.search(testing.allocator, &db, .hybrid, "fusion", &query_vec, cid, .{
            .top_k = 10,
            .path = c.pattern,
            .max_per_doc = null,
        });
        defer res.deinit(testing.allocator);
        testing.expectEqual(c.want.len, res.hits.len) catch |e| {
            std.debug.print("pattern {s}: got {d} hit(s)\n", .{ c.pattern, res.hits.len });
            return e;
        };
        for (res.hits) |h| {
            var ok = false;
            for (c.want) |d| if (d == h.doc_id) {
                ok = true;
            };
            testing.expect(ok) catch |e| {
                std.debug.print("pattern {s}: unexpected doc {d}\n", .{ c.pattern, h.doc_id });
                return e;
            };
        }
    }
}
