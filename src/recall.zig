//! `recall` — what an Agent should know before it starts.
//!
//! Shared by the CLI and the daemon so there is one ranking, not two that drift.
//! Two things make this different from `query`:
//!
//!   1. **Facts are injected, never retrieved.** The current value of every fact
//!      is attached unconditionally, read from `facts.csv`. Retrieval would just
//!      as happily return a narrative mentioning last year's salary, with nothing
//!      to mark which is current — and 450000 and 480000 are neighbours in a
//!      1024-dimensional space anyway (SPEC §15.5, §16.5).
//!
//!   2. **Recency is a ranking path, not a decay coefficient.** A decay needs a
//!      constant, and a constant with no experiment behind it is exactly the debt
//!      RRF was chosen to avoid. Two rank lists need no calibration and share no
//!      units (SPEC §15.6).
//!
//! Recency applies here and nowhere else. In `search` the user asked for
//! something specific, and letting the newest memory outrank the relevant one
//! would be plainly wrong.

const std = @import("std");
const sqlite = @import("db/sqlite.zig");
const store = @import("db/store.zig");
const hybrid = @import("search/hybrid.zig");
const rrf = @import("search/rrf.zig");
const packmod = @import("search/pack.zig");
const memory = @import("memory.zig");
const facts = @import("facts.zig");

pub const Config = struct {
    /// Smaller than `query`'s 8000 by design. Recall is injected into every
    /// session; a large one crowds out the actual task.
    budget_tokens: usize = 1500,
    candidates: usize = 20,
    recency_depth: usize = 20,
};

pub const Result = struct {
    /// Owns the hits the pack points at, so it must outlive the pack.
    ranked: hybrid.Results,
    pack: packmod.Pack,

    pub fn deinit(self: *Result, gpa: std.mem.Allocator) void {
        self.pack.deinit(gpa);
        self.ranked.deinit(gpa);
        self.* = undefined;
    }
};

/// `query` may be empty — that is session start, where nothing has been asked
/// and ranking is recency alone. `query_vec` may be null when the model is
/// unavailable; the keyword path still works.
pub fn assemble(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    query: []const u8,
    query_vec: ?[]const f32,
    cfg: Config,
) !Result {
    var s = store.Store.init(db);
    const memory_cid = try s.findCollection("memory");

    var hits: []hybrid.Hit = &.{};
    var filled: usize = 0;
    errdefer {
        for (hits[0..filled]) |h| freeHit(gpa, h);
        gpa.free(hits);
    }

    if (memory_cid) |cid| {
        var search_ids: std.ArrayList(i64) = .empty;
        defer search_ids.deinit(gpa);

        if (query.len != 0) {
            const mode: hybrid.Mode = if (query_vec == null) .keyword else .hybrid;
            var results = try hybrid.search(gpa, db, mode, query, query_vec, cid, .{
                .top_k = cfg.candidates,
                .candidates = @max(50, cfg.candidates),
            });
            defer results.deinit(gpa);
            for (results.hits) |h| try search_ids.append(gpa, h.chunk_id);
        }

        const recency = try memory.recencyRanked(gpa, db, cfg.recency_depth);
        defer gpa.free(recency);

        // Fusing an already-fused list with a third is legitimate: RRF consumes
        // ranks, and the relevance list is a valid ranking whatever produced it.
        const fused = try rrf.fuse(gpa, search_ids.items, recency, .{
            // Both inputs are trusted here. A short recency list means few
            // memories exist, not a weak signal, so the sparse-path cutoff that
            // protects against noisy BM25 would be actively wrong.
            .fts_min_hits = 0,
        });
        defer gpa.free(fused);

        hits = try gpa.alloc(hybrid.Hit, @min(cfg.candidates, fused.len));
        for (fused[0..hits.len]) |f| {
            hits[filled] = try hybrid.hydrate(gpa, db, f);
            filled += 1;
        }
    }

    var ranked: hybrid.Results = .{
        .mode = if (query.len == 0) .vector else .hybrid,
        .hits = hits,
        .dropped_terms = &.{},
        .fts_skipped = false,
        .vec_candidates = filled,
        .fts_candidates = 0,
    };
    errdefer ranked.deinit(gpa);

    const p = try packmod.assemble(gpa, db, query, &ranked, .{
        .budget_tokens = cfg.budget_tokens,
        // Memories are short and self-contained; expanding to neighbours would
        // mostly pull in *other* memories, which the ranking already declined.
        .neighbors = 0,
        .candidates = cfg.candidates,
    });

    return .{ .ranked = ranked, .pack = p };
}

fn freeHit(gpa: std.mem.Allocator, h: hybrid.Hit) void {
    gpa.free(h.collection);
    gpa.free(h.rel_path);
    gpa.free(h.title);
    gpa.free(h.heading_path);
    gpa.free(h.text);
}

// ---------------------------------------------------------------------------
// facts snapshot
// ---------------------------------------------------------------------------

pub fn renderFactsMarkdown(w: *std.Io.Writer, current: []const facts.Current) !void {
    if (current.len == 0) return;
    try w.writeAll("## Facts (current values)\n\n");
    for (current) |f| {
        try w.print("- {s}: {s}  (as of {s})", .{ f.key, f.value, f.at });
        if (f.note.len != 0) try w.print(" — {s}", .{f.note});
        try w.writeAll("\n");
    }
    try w.writeAll("\n");
}

pub fn renderFactsJson(w: *std.Io.Writer, current: []const facts.Current) !void {
    try w.writeAll("[");
    for (current, 0..) |f, i| {
        if (i != 0) try w.writeAll(",");
        try w.writeAll("{\"key\":");
        try std.json.Stringify.value(f.key, .{}, w);
        try w.writeAll(",\"value\":");
        try std.json.Stringify.value(f.value, .{}, w);
        try w.writeAll(",\"at\":");
        try std.json.Stringify.value(f.at, .{}, w);
        try w.writeAll(",\"note\":");
        try std.json.Stringify.value(f.note, .{}, w);
        try w.writeAll("}");
    }
    try w.writeAll("]");
}
