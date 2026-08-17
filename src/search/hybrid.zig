//! Hybrid retrieval: vector KNN + FTS5 BM25, fused with RRF.
//!
//! The two paths answer different questions and neither subsumes the other:
//! vectors find "the same idea in different words" (including across Chinese and
//! English); BM25 finds the exact identifier, error code or proper noun that a
//! 1024-dim embedding smears away.

const std = @import("std");
const sqlite = @import("../db/sqlite.zig");
const fts_query = @import("fts_query.zig");
const rrf = @import("rrf.zig");

pub const Mode = enum { hybrid, vector, keyword };

pub const Config = struct {
    /// Per-path candidate depth before fusion.
    candidates: usize = 50,
    top_k: usize = 10,
    rrf: rrf.Config = .{},
};

pub const Hit = struct {
    chunk_id: i64,
    score: f64,
    vec_rank: ?u32,
    fts_rank: ?u32,
    collection: []const u8,
    rel_path: []const u8,
    title: []const u8,
    heading_path: []const u8,
    text: []const u8,
    n_tokens: i64,
    doc_id: i64,
    idx: i64,
};

pub const Results = struct {
    mode: Mode,
    hits: []Hit,
    /// Query terms the tokenizer cannot match; surfaced so the caller knows the
    /// search did not cover them (SPEC §5.1).
    dropped_terms: [][]const u8,
    /// True when the keyword path was skipped or produced too few hits to trust.
    fts_skipped: bool,
    vec_candidates: usize,
    fts_candidates: usize,

    pub fn deinit(self: *Results, gpa: std.mem.Allocator) void {
        for (self.hits) |h| {
            gpa.free(h.collection);
            gpa.free(h.rel_path);
            gpa.free(h.title);
            gpa.free(h.heading_path);
            gpa.free(h.text);
        }
        gpa.free(self.hits);
        for (self.dropped_terms) |d| gpa.free(d);
        gpa.free(self.dropped_terms);
        self.* = undefined;
    }
};

/// `query_vec` may be null for keyword-only search (or when the model is
/// unavailable and the caller has degraded gracefully).
pub fn search(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    mode: Mode,
    query_text: []const u8,
    query_vec: ?[]const f32,
    collection_id: ?i64,
    cfg: Config,
) !Results {
    var vec_ids: std.ArrayList(i64) = .empty;
    defer vec_ids.deinit(gpa);
    var fts_ids: std.ArrayList(i64) = .empty;
    defer fts_ids.deinit(gpa);

    var dropped: [][]const u8 = &.{};
    var fts_skipped = true;

    // ---- vector path
    if (mode != .keyword) {
        if (query_vec) |vec| {
            try knn(gpa, db, vec, collection_id, cfg.candidates, &vec_ids);
        }
    }

    // ---- keyword path
    if (mode != .vector) {
        var q = try fts_query.build(gpa, query_text);
        // Ownership of `dropped` moves to Results; the expr is freed here.
        defer if (q.expr) |e| gpa.free(e);
        dropped = q.dropped;
        q.dropped = &.{};

        if (q.expr) |expr| {
            try bm25(gpa, db, expr, collection_id, cfg.candidates, &fts_ids);
            fts_skipped = fts_ids.items.len < cfg.rrf.fts_min_hits;
        }
    }

    const fused = switch (mode) {
        .vector => try rrf.fuse(gpa, vec_ids.items, &.{}, cfg.rrf),
        .keyword => try rrf.fuse(gpa, &.{}, fts_ids.items, .{
            .k = cfg.rrf.k,
            .top_boost_fraction = cfg.rrf.top_boost_fraction,
            // Keyword-only mode must not discard its own single path just because
            // the corpus is sparse — there is nothing to fall back to.
            .fts_min_hits = 0,
        }),
        .hybrid => try rrf.fuse(gpa, vec_ids.items, fts_ids.items, cfg.rrf),
    };
    defer gpa.free(fused);

    const take = @min(cfg.top_k, fused.len);
    var hits = try gpa.alloc(Hit, take);
    var filled: usize = 0;
    errdefer {
        for (hits[0..filled]) |h| {
            gpa.free(h.collection);
            gpa.free(h.rel_path);
            gpa.free(h.title);
            gpa.free(h.heading_path);
            gpa.free(h.text);
        }
        gpa.free(hits);
    }
    for (fused[0..take]) |f| {
        hits[filled] = try hydrate(gpa, db, f);
        filled += 1;
    }

    return .{
        .mode = mode,
        .hits = hits,
        .dropped_terms = dropped,
        .fts_skipped = fts_skipped and mode != .vector,
        .vec_candidates = vec_ids.items.len,
        .fts_candidates = fts_ids.items.len,
    };
}

fn knn(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    vec: []const f32,
    collection_id: ?i64,
    k: usize,
    out: *std.ArrayList(i64),
) !void {
    if (collection_id) |cid| {
        // Partition-scoped KNN: the filter is applied inside the search, so
        // asking for k really returns k from this collection (verified by E1).
        var st = try db.prepare(
            \\SELECT chunk_id FROM vec_chunks
            \\WHERE collection_id = ?1 AND embedding MATCH ?2 AND k = ?3
        );
        defer st.finalize();
        try st.bindI64(1, cid);
        try st.bindVector(2, vec);
        try st.bindI64(3, @intCast(k));
        while (try st.step()) try out.append(gpa, st.columnI64(0));
        return;
    }

    // No collection filter: vec0 constrains a partition key to a single value,
    // so query each partition and merge. Distances are comparable across
    // partitions (same metric, same normalized space), so a sort is sufficient.
    var ids: std.ArrayList(i64) = .empty;
    defer ids.deinit(gpa);
    {
        var st = try db.prepare("SELECT id FROM collections ORDER BY id");
        defer st.finalize();
        while (try st.step()) try ids.append(gpa, st.columnI64(0));
    }

    const Scored = struct { id: i64, dist: f64 };
    var scored: std.ArrayList(Scored) = .empty;
    defer scored.deinit(gpa);

    for (ids.items) |cid| {
        var st = try db.prepare(
            \\SELECT chunk_id, distance FROM vec_chunks
            \\WHERE collection_id = ?1 AND embedding MATCH ?2 AND k = ?3
        );
        defer st.finalize();
        try st.bindI64(1, cid);
        try st.bindVector(2, vec);
        try st.bindI64(3, @intCast(k));
        while (try st.step()) {
            try scored.append(gpa, .{ .id = st.columnI64(0), .dist = st.columnF64(1) });
        }
    }

    const S = struct {
        fn less(_: void, a: Scored, b: Scored) bool {
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.id < b.id;
        }
    };
    std.mem.sort(Scored, scored.items, {}, S.less);
    for (scored.items[0..@min(k, scored.items.len)]) |s| try out.append(gpa, s.id);
}

fn bm25(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    expr: []const u8,
    collection_id: ?i64,
    limit: usize,
    out: *std.ArrayList(i64),
) !void {
    // heading_path is weighted below the body: a title match is a weak signal,
    // a body match is a strong one. bm25() returns negative values, smaller is
    // more relevant, hence plain ASC ordering.
    if (collection_id) |cid| {
        var st = try db.prepare(
            \\SELECT f.rowid FROM fts_chunks f
            \\JOIN chunks c ON c.id = f.rowid
            \\JOIN docs d ON d.id = c.doc_id
            \\WHERE f.fts_chunks MATCH ?1 AND d.collection_id = ?2
            \\ORDER BY bm25(fts_chunks, 1.0, 0.5) LIMIT ?3
        );
        defer st.finalize();
        try st.bindText(1, expr);
        try st.bindI64(2, cid);
        try st.bindI64(3, @intCast(limit));
        while (try st.step()) try out.append(gpa, st.columnI64(0));
    } else {
        var st = try db.prepare(
            \\SELECT rowid FROM fts_chunks
            \\WHERE fts_chunks MATCH ?1
            \\ORDER BY bm25(fts_chunks, 1.0, 0.5) LIMIT ?2
        );
        defer st.finalize();
        try st.bindText(1, expr);
        try st.bindI64(2, @intCast(limit));
        while (try st.step()) try out.append(gpa, st.columnI64(0));
    }
}

/// Public so `recall` can fuse in a third ranking path (recency) and still
/// produce Hits identical to the ones this module returns — a second, slightly
/// different hydration would drift from this one.
pub fn hydrate(gpa: std.mem.Allocator, db: *sqlite.Db, f: rrf.Fused) !Hit {
    var st = try db.prepare(
        \\SELECT c.doc_id, c.idx, c.heading_path, c.text, c.n_tokens,
        \\       d.rel_path, COALESCE(d.title, ''), col.name
        \\FROM chunks c
        \\JOIN docs d ON d.id = c.doc_id
        \\JOIN collections col ON col.id = d.collection_id
        \\WHERE c.id = ?1
    );
    defer st.finalize();
    try st.bindI64(1, f.chunk_id);
    if (!try st.step()) return error.SqliteStep;

    return .{
        .chunk_id = f.chunk_id,
        .score = f.score,
        .vec_rank = f.vec_rank,
        .fts_rank = f.fts_rank,
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
