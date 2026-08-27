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
    /// Which context this recall is for. Null means universal memories only.
    ///
    /// zkb does not infer this from the working directory: guessing would bake one
    /// person's layout into the tool, and guessing wrong would leak exactly what
    /// the scope exists to contain. The caller knows where it is.
    scope: ?[]const u8 = null,
};

pub const Result = struct {
    /// Owns the hits the pack points at, so it must outlive the pack.
    ranked: hybrid.Results,
    pack: packmod.Pack,
    /// How many memories the index holds, counted from `docs` and not from
    /// `rec_memory`.
    ///
    /// Which table it comes from is the entire point. Ranking reads `rec_memory`
    /// (see `memory.recencyRanked`), so when that projection is the thing that
    /// broke, counting it reports zero and corroborates the empty result instead
    /// of contradicting it. `docs` is written by a different path — `scan` fills
    /// it, `indexer` only stamps it — so it can still say "there are forty of
    /// them", which is the only way an empty recall ever becomes reportable as a
    /// fault rather than as an answer.
    memory_docs: usize,

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
    const memory_docs: usize = if (memory_cid) |cid| try indexedDocs(db, cid) else 0;

    var hits: []hybrid.Hit = &.{};
    var filled: usize = 0;
    errdefer {
        for (hits[0..filled]) |h| hybrid.freeHit(gpa, h);
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
                // A candidate pool for `pack`, as in `query` — and one memory is
                // one document, so a per-document ceiling would cap nothing here
                // anyway.
                .max_per_doc = null,
            });
            defer results.deinit(gpa);
            // Filtered here rather than inside `hybrid.search`: a scope is a
            // property of a memory, not of retrieval, and threading it into search
            // would put the same filter on a second code path — the shape that has
            // gone wrong repeatedly in this codebase.
            for (results.hits) |h| {
                if (try memory.chunkInScope(db, h.chunk_id, cfg.scope)) {
                    try search_ids.append(gpa, h.chunk_id);
                }
            }
        }

        const recency = try memory.recencyRanked(gpa, db, cfg.recency_depth, cfg.scope);
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
            hits[filled] = try hybrid.hydrate(gpa, db, f) orelse continue;
            filled += 1;
        }
        // A memory whose file is gone must cost itself, not the whole recall.
        if (filled != hits.len) hits = try gpa.realloc(hits, filled);
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

    return .{ .ranked = ranked, .pack = p, .memory_docs = memory_docs };
}

/// Indexed documents in a collection. `indexed_at IS NOT NULL` rather than a bare
/// count: a document queued but not yet embedded is not something recall could
/// have returned, so counting it would turn a mid-index moment into a fault
/// report.
fn indexedDocs(db: *sqlite.Db, collection_id: i64) !usize {
    var st = try db.prepare(
        "SELECT count(*) FROM docs WHERE collection_id = ?1 AND indexed_at IS NOT NULL",
    );
    defer st.finalize();
    try st.bindI64(1, collection_id);
    if (!try st.step()) return 0;
    return @intCast(@max(0, st.columnI64(0)));
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

/// The memories half of `recall`'s markdown, for both transports.
///
/// `memory_docs` is what separates "you have not recorded anything yet" from
/// "you have, and not one of them came back". Those had a single message between
/// them, and it asserted the first: when `rec_memory` silently emptied, every
/// session for two days opened with `No memories yet. Record one with: zkb
/// remember`, while `zkb status` counted forty of them one command away. A
/// message that names the wrong cause is worse than no message, because it ends
/// the investigation instead of starting one.
///
/// This is `pack.renderEmpty`'s distinction — matched but dropped, against
/// nothing matched — drawn once more for the store itself.
pub fn renderMemoriesMarkdown(
    w: *std.Io.Writer,
    pack: *const packmod.Pack,
    memory_docs: usize,
) !void {
    // `omitted` non-empty means memories were found and the budget dropped them,
    // which `renderMarkdown` already explains better than this can.
    if (pack.groups.len == 0 and pack.omitted.len == 0) {
        if (memory_docs == 0) {
            try w.writeAll("No memories yet. Record one with: zkb remember \"...\"\n");
        } else {
            try w.print(
                "{d} memories are indexed, but none of them ranked into this recall.\n" ++
                    "That is not an ordinary empty result — check the index: zkb maintain\n",
                .{memory_docs},
            );
        }
        return;
    }
    try packmod.renderMarkdown(w, pack);
}

// ---------------------------------------------------------------------------
// facts snapshot
// ---------------------------------------------------------------------------

/// A fact narrowed to the fields that reach the markdown.
///
/// The daemon sends these four and not `recorded_at` or `scope`, so the path that
/// rebuilds facts from json has nothing to put in a `facts.Current`. Filling two
/// fields with empty strings to satisfy the type would move the lie from the
/// renderer into the struct, where the next reader cannot see it.
pub const FactLine = struct {
    key: []const u8,
    value: []const u8,
    at: []const u8,
    note: []const u8,
};

/// Called by both transports: one holds `facts.Current`, the other holds
/// `FactLine`, and they share exactly the four fields printed here. `anytype`
/// rather than converting one into the other, because the conversion would have
/// to invent the two fields the wire does not carry — and what must not diverge
/// is the wording, which this way exists once.
pub fn renderFactsMarkdown(w: *std.Io.Writer, current: anytype) !void {
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
