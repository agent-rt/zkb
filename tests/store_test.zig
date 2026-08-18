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

    // Same content found at a different path is a rename, not new content.
    const found = try s.findDocByShaExcludingPath(cid, "sha-x", "new.md");
    try testing.expectEqual(@as(?i64, did), found);

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
    const k = try s.upsertCollection("kb", "/k", .records, null, null, 1);

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
