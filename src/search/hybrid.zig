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
const glob = @import("../ingest/glob.zig");

pub const Mode = enum { hybrid, vector, keyword };

pub const Config = struct {
    /// Per-path candidate depth before fusion.
    candidates: usize = 50,
    top_k: usize = 10,
    rrf: rrf.Config = .{},
    /// Restrict to documents whose rel_path matches this glob.
    ///
    /// Scoping retrieval to part of a corpus without splitting it. A separate
    /// collection would do it too — vec0 applies a partition filter inside the
    /// KNN (E1) — but a collection is an identity, and carving a subdirectory out
    /// of `~/docs` severed every `zkb://agents/handoffs/...` link in both
    /// directions. Filtering is a property of one query; a collection is a
    /// property of the corpus.
    path: ?[]const u8 = null,
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

    // Resolved once, up front: both paths need the same document set, and a glob
    // cannot be expressed in SQL.
    var scope: ?[]const i64 = null;
    defer if (scope) |ids| gpa.free(ids);
    if (cfg.path) |pat| scope = try scopeDocIds(gpa, db, collection_id, pat);

    // ---- vector path
    if (mode != .keyword) {
        if (query_vec) |vec| {
            if (scope) |ids| {
                try knnWithin(gpa, db, vec, ids, cfg.candidates, &vec_ids);
            } else {
                try knn(gpa, db, vec, collection_id, cfg.candidates, &vec_ids);
            }
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
            try bm25(gpa, db, expr, collection_id, scope, cfg.candidates, &fts_ids);
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
        hits[filled] = try hydrate(gpa, db, f) orelse continue;
        filled += 1;
    }
    // Shrunk rather than returned as a subslice: `Results.deinit` frees
    // `self.hits`, and freeing a subslice of a larger allocation is not the same
    // allocation the allocator handed out.
    if (filled != hits.len) hits = try gpa.realloc(hits, filled);

    return .{
        .mode = mode,
        .hits = hits,
        .dropped_terms = dropped,
        .fts_skipped = fts_skipped and mode != .vector,
        .vec_candidates = vec_ids.items.len,
        .fts_candidates = fts_ids.items.len,
    };
}

/// Documents in scope, by glob over `rel_path`.
///
/// A scan of `docs` rather than a SQL predicate, because the pattern is a glob and
/// SQL's LIKE cannot express `**` or a character class. At corpus scale this is a
/// few hundred short strings; when it stops being cheap the fix is an index on the
/// prefix, not a weaker pattern language.
fn scopeDocIds(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    collection_id: ?i64,
    pattern: []const u8,
) ![]i64 {
    var out: std.ArrayList(i64) = .empty;
    errdefer out.deinit(gpa);

    var st = if (collection_id) |_|
        try db.prepare("SELECT id, rel_path FROM docs WHERE collection_id = ?1")
    else
        try db.prepare("SELECT id, rel_path FROM docs");
    defer st.finalize();
    if (collection_id) |cid| try st.bindI64(1, cid);

    while (try st.step()) {
        if (inPathScope(pattern, st.columnText(1))) try out.append(gpa, st.columnI64(0));
    }
    return out.toOwnedSlice(gpa);
}

/// Does `rel_path` fall inside what a `--path` argument names?
///
/// A pattern with no wildcard reads as a place, not a filename. `--path
/// projects/qlit` is how the argument gets typed, and matching it only as an
/// exact `rel_path` finds nothing — output that is indistinguishable from "that
/// project has nothing about this". Measured across the forms someone actually
/// reaches for, `projects/qlit/**`, `projects/qlit/*` and even the stray
/// `/projects/qlit/**` all worked, while the plainest one silently did not.
///
/// The two readings do not compete: a document either *is* that path or lives
/// under it, so both count. That keeps `--path index.md` meaning the file —
/// rewriting a bare pattern into `<pat>/**` instead would have broken it.
///
/// Only wildcard-free patterns get the subtree reading. Once a pattern contains
/// `*` or `?` the person is writing a glob and is entitled to have it obeyed.
/// `glob` itself is left alone: it is also the scan's `--include` matcher, and
/// widening a filter that decides which files get indexed is a different and much
/// less reversible decision.
fn inPathScope(pattern: []const u8, rel_path: []const u8) bool {
    if (glob.match(pattern, rel_path)) return true;

    for (pattern) |c| if (c == '*' or c == '?') return false;
    // `./` and a leading `/` are how a path arrives when it was pasted out of a
    // shell or a file listing. A rel_path carries neither, so stripping them can
    // only turn a non-match into a match — and `glob` already tolerates the
    // leading slash on wildcard patterns, so not doing it here would leave
    // `/projects/qlit/**` working while `/projects/qlit` did not.
    var dir = pattern;
    if (std.mem.startsWith(u8, dir, "./")) dir = dir[2..];
    dir = std.mem.trimStart(u8, dir, "/");
    dir = std.mem.trimEnd(u8, dir, "/");
    if (dir.len == 0) return false;
    return rel_path.len > dir.len + 1 and
        std.mem.startsWith(u8, rel_path, dir) and
        rel_path[dir.len] == '/';
}

/// Exact nearest neighbours within a document subset.
///
/// vec0 constrains a partition key, not an arbitrary id set, so a `chunk_id IN`
/// clause on a KNN query would at best post-filter its top-k — and a selective
/// path would then return almost nothing, silently. Reading the subset's vectors
/// and scoring them directly is exact instead of approximate, and the subset is
/// small by construction: the case this exists for is 89 chunks out of 3928, and
/// 1024 floats times a few hundred rows is not work worth approximating.
fn knnWithin(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    vec: []const f32,
    doc_ids: []const i64,
    k: usize,
    out: *std.ArrayList(i64),
) !void {
    const Scored = struct { id: i64, dist: f64 };
    var scored: std.ArrayList(Scored) = .empty;
    defer scored.deinit(gpa);

    var st = try db.prepare(
        \\SELECT v.chunk_id, v.embedding FROM vec_chunks v
        \\JOIN chunks c ON c.id = v.chunk_id
        \\WHERE c.doc_id = ?1
    );
    defer st.finalize();

    for (doc_ids) |doc_id| {
        st.reset();
        try st.bindI64(1, doc_id);
        while (try st.step()) {
            const raw = st.columnBlob(1);
            // Stored as little-endian f32; a short or odd-sized row means the
            // vector table disagrees with the schema and skipping beats a
            // misaligned read.
            if (raw.len != vec.len * @sizeOf(f32)) continue;
            const emb = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(raw)));
            try scored.append(gpa, .{ .id = st.columnI64(0), .dist = cosineDistance(vec, emb) });
        }
    }

    const S = struct {
        fn less(_: void, a: Scored, b: Scored) bool {
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.id < b.id;
        }
    };
    std.mem.sort(Scored, scored.items, {}, S.less);
    for (scored.items[0..@min(k, scored.items.len)]) |x| try out.append(gpa, x.id);
}

/// Same metric vec0 is configured with (`distance_metric=cosine`), so ranks from
/// this path and from the vtab are interchangeable.
fn cosineDistance(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    if (na == 0 or nb == 0) return 1.0;
    return 1.0 - dot / (@sqrt(na) * @sqrt(nb));
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
    scope: ?[]const i64,
    limit: usize,
    out: *std.ArrayList(i64),
) !void {
    // With a scope, restrict inside the query so the top-k is the top-k *of the
    // subset*. Post-filtering a global top-k would quietly return a handful of
    // rows for a selective path, which is the failure mode this whole feature is
    // meant to remove.
    if (scope) |ids| {
        if (ids.len == 0) return;
        var sql: std.ArrayList(u8) = .empty;
        defer sql.deinit(gpa);
        try sql.appendSlice(gpa,
            \\SELECT f.rowid FROM fts_chunks f
            \\JOIN chunks c ON c.id = f.rowid
            \\WHERE f.fts_chunks MATCH ?1 AND c.doc_id IN (
        );
        // Inlined rather than bound: SQLite has no list parameter, and these are
        // integers this process just read out of the same database.
        for (ids, 0..) |id, i| {
            if (i != 0) try sql.append(gpa, ',');
            try sql.print(gpa, "{d}", .{id});
        }
        try sql.appendSlice(gpa, ") ORDER BY bm25(fts_chunks, 1.0, 0.5) LIMIT ?2");

        var st = try db.prepare(sql.items);
        defer st.finalize();
        try st.bindText(1, expr);
        try st.bindI64(2, @intCast(limit));
        while (try st.step()) try out.append(gpa, st.columnI64(0));
        return;
    }
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
/// Null when the candidate cannot be hydrated: the chunk is in the search
/// indexes but its document is gone.
///
/// This used to return `error.SqliteStep` on a missing row, which is two mistakes
/// at once. `step()` returning false is "no row", not a sqlite failure — and
/// mapping it to an error meant one stale index entry aborted the whole query.
/// Measured on this machine: 1198 of 5213 chunks had no document row (residue
/// from an older binary during a bulk move), and `zkb search "i18n locales"`
/// answered `internal: search failed` while `zkb search "交接"` worked, purely
/// on whether a stale chunk landed in the top-k.
///
/// A derived index that has drifted must cost the results it cannot resolve, not
/// every result. The drift itself is reported by `maintain --check orphan_chunk`,
/// which is where a caller can act on it.
pub fn hydrate(gpa: std.mem.Allocator, db: *sqlite.Db, f: rrf.Fused) !?Hit {
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
    if (!try st.step()) return null;

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
