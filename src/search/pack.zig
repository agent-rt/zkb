//! Assemble retrieval hits into a context pack for an LLM.
//!
//! `search` answers "which chunks match"; `query` answers "what should I put in
//! the prompt". The difference is not intelligence, it is that the output is
//! shaped for a context window instead of for human eyes (SPEC §5.4):
//!
//!   - neighbouring chunks are pulled in, because a hit mid-document usually
//!     starts with a pronoun referring to the chunk before it
//!   - chunks are regrouped per document and contiguous runs are merged, so the
//!     model sees continuous prose rather than overlapping fragments
//!   - a token budget is enforced, and what was dropped is *listed* rather than
//!     silently discarded
//!
//! The budget uses `chunks.n_tokens`, measured at index time with the embedding
//! model's own tokenizer. No tokenizer call is needed here — and the count is
//! honest about what it is: the retrieval model's tokenizer, not the tokenizer of
//! whatever model finally reads this.

const std = @import("std");
const sqlite = @import("../db/sqlite.zig");
const hybrid = @import("hybrid.zig");

pub const Config = struct {
    /// Chunks to retrieve before grouping. Larger than the final document count
    /// because several chunks usually collapse into one document.
    candidates: usize = 30,
    /// Chunks on each side of a hit to pull in for context.
    neighbors: i64 = 1,
    /// Token ceiling for the assembled pack.
    budget_tokens: usize = 8000,
    /// Rough per-document formatting overhead (path line, heading line, blank
    /// lines). Counted against the budget so the estimate errs on the safe side.
    per_doc_overhead_tokens: usize = 24,
    /// No single document may take more than `budget / max_doc_divisor`.
    ///
    /// Measured need: one long README scored highest, its neighbour-expanded span
    /// came to 1247 tokens, and it consumed 83% of a 1500-token budget — pushing
    /// out two documents that answered the question better. For a pack meant to
    /// inform an answer, breadth across documents beats depth in one.
    ///
    /// 3 is not a tuned constant: it is the guarantee that at least three
    /// documents can fit, which is what "breadth" has to mean at minimum.
    max_doc_divisor: usize = 3,
};

/// A contiguous run of chunks from one document, already merged.
pub const Span = struct {
    first_idx: i64,
    last_idx: i64,
    heading_path: []const u8,
    text: []const u8,
    n_tokens: usize,
};

pub const DocGroup = struct {
    doc_id: i64,
    collection: []const u8,
    rel_path: []const u8,
    title: []const u8,
    /// Best fused score among this document's hits; the sort key.
    score: f64,
    spans: []Span,
    n_tokens: usize,
};

pub const Omitted = struct {
    rel_path: []const u8,
    score: f64,
};

pub const Pack = struct {
    query: []const u8,
    mode: hybrid.Mode,
    groups: []DocGroup,
    /// Documents that matched but did not fit the budget. Listed so the caller
    /// knows more exists rather than believing it saw everything.
    omitted: []Omitted,
    total_tokens: usize,
    budget_tokens: usize,
    dropped_terms: [][]const u8,
    fts_skipped: bool,

    pub fn deinit(self: *Pack, gpa: std.mem.Allocator) void {
        for (self.groups) |g| {
            gpa.free(g.collection);
            gpa.free(g.rel_path);
            gpa.free(g.title);
            for (g.spans) |s| {
                gpa.free(s.heading_path);
                gpa.free(s.text);
            }
            gpa.free(g.spans);
        }
        gpa.free(self.groups);
        for (self.omitted) |o| gpa.free(o.rel_path);
        gpa.free(self.omitted);
        for (self.dropped_terms) |d| gpa.free(d);
        gpa.free(self.dropped_terms);
        self.* = undefined;
    }
};

/// Takes ownership of nothing; `results` stays the caller's to free.
pub fn assemble(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    query: []const u8,
    results: *const hybrid.Results,
    cfg: Config,
) !Pack {
    // ---- 1. group hits by document, keeping the best score per document
    const Acc = struct {
        doc_id: i64,
        score: f64,
        collection: []const u8,
        rel_path: []const u8,
        title: []const u8,
        idxs: std.ArrayList(i64),
    };
    var accs: std.ArrayList(Acc) = .empty;
    defer {
        for (accs.items) |*a| a.idxs.deinit(gpa);
        accs.deinit(gpa);
    }

    for (results.hits) |h| {
        var found: ?*Acc = null;
        for (accs.items) |*a| {
            if (a.doc_id == h.doc_id) {
                found = a;
                break;
            }
        }
        if (found) |a| {
            a.score = @max(a.score, h.score);
            try a.idxs.append(gpa, h.idx);
        } else {
            var idxs: std.ArrayList(i64) = .empty;
            try idxs.append(gpa, h.idx);
            try accs.append(gpa, .{
                .doc_id = h.doc_id,
                .score = h.score,
                .collection = h.collection,
                .rel_path = h.rel_path,
                .title = h.title,
                .idxs = idxs,
            });
        }
    }

    // Best document first: the caller reads top-down and may be truncated.
    std.mem.sort(Acc, accs.items, {}, struct {
        fn less(_: void, a: Acc, b: Acc) bool {
            if (a.score != b.score) return a.score > b.score;
            return a.doc_id < b.doc_id;
        }
    }.less);

    // ---- 2. expand with neighbours, merge into contiguous ranges, fit budget
    var groups: std.ArrayList(DocGroup) = .empty;
    errdefer {
        for (groups.items) |g| freeGroup(gpa, g);
        groups.deinit(gpa);
    }
    var omitted: std.ArrayList(Omitted) = .empty;
    errdefer {
        for (omitted.items) |o| gpa.free(o.rel_path);
        omitted.deinit(gpa);
    }

    var used: usize = 0;

    for (accs.items) |*a| {
        // Per-document ceiling, but never below one span: a document whose single
        // span exceeds the cap still contributes rather than vanishing.
        const doc_cap = @max(
            cfg.budget_tokens / @max(1, cfg.max_doc_divisor),
            cfg.per_doc_overhead_tokens + 1,
        );

        // Neighbours are a nicety; the hit itself is the necessity. If expanding
        // blows the per-document cap, retry with none — measured case: one long
        // README's neighbour-expanded span came to 1247 tokens and ate 83% of a
        // 1500-token budget, pushing out two better documents. Trimming trailing
        // spans does not help when the offender *is* the first span.
        var group = collectGroup(gpa, db, a.doc_id, a.idxs.items, cfg.neighbors, cfg, used, doc_cap) catch |err| return err;
        if (group.tokens > doc_cap and cfg.neighbors > 0) {
            freeSpans(gpa, group.spans.items);
            group.spans.deinit(gpa);
            group = try collectGroup(gpa, db, a.doc_id, a.idxs.items, 0, cfg, used, doc_cap);
        }
        var spans = group.spans;
        errdefer {
            freeSpans(gpa, spans.items);
            spans.deinit(gpa);
        }

        if (spans.items.len == 0 or used + group.tokens > cfg.budget_tokens) {
            freeSpans(gpa, spans.items);
            spans.deinit(gpa);
            try omitted.append(gpa, .{
                .rel_path = try gpa.dupe(u8, a.rel_path),
                .score = a.score,
            });
            continue;
        }

        used += group.tokens;
        try groups.append(gpa, .{
            .doc_id = a.doc_id,
            .collection = try gpa.dupe(u8, a.collection),
            .rel_path = try gpa.dupe(u8, a.rel_path),
            .title = try gpa.dupe(u8, a.title),
            .score = a.score,
            .spans = try spans.toOwnedSlice(gpa),
            .n_tokens = group.tokens,
        });
    }

    // dropped_terms is copied so the pack outlives the Results it came from.
    var dropped = try gpa.alloc([]const u8, results.dropped_terms.len);
    var filled: usize = 0;
    errdefer {
        for (dropped[0..filled]) |d| gpa.free(d);
        gpa.free(dropped);
    }
    for (results.dropped_terms) |d| {
        dropped[filled] = try gpa.dupe(u8, d);
        filled += 1;
    }

    return .{
        .query = query,
        .mode = results.mode,
        .groups = try groups.toOwnedSlice(gpa),
        .omitted = try omitted.toOwnedSlice(gpa),
        .total_tokens = used,
        .budget_tokens = cfg.budget_tokens,
        .dropped_terms = dropped,
        .fts_skipped = results.fts_skipped,
    };
}

fn freeSpans(gpa: std.mem.Allocator, spans: []const Span) void {
    for (spans) |s| {
        gpa.free(s.heading_path);
        gpa.free(s.text);
    }
}

const Collected = struct { spans: std.ArrayList(Span), tokens: usize };

/// Load one document's spans at a given neighbour width, stopping when either the
/// global budget or the per-document cap would be exceeded.
fn collectGroup(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    doc_id: i64,
    idxs: []const i64,
    neighbors: i64,
    cfg: Config,
    used: usize,
    doc_cap: usize,
) !Collected {
    var ranges = try mergeRanges(gpa, idxs, neighbors);
    defer ranges.deinit(gpa);

    var spans: std.ArrayList(Span) = .empty;
    errdefer {
        freeSpans(gpa, spans.items);
        spans.deinit(gpa);
    }

    var tokens: usize = cfg.per_doc_overhead_tokens;
    for (ranges.items) |r| {
        const span = try loadSpan(gpa, db, doc_id, r.first, r.last);
        const over_budget = used + tokens + span.n_tokens > cfg.budget_tokens;
        const over_cap = tokens + span.n_tokens > doc_cap;
        // Trailing spans go first: the strongest part of a long document should
        // survive truncation.
        if ((over_budget or over_cap) and spans.items.len > 0) {
            gpa.free(span.heading_path);
            gpa.free(span.text);
            break;
        }
        tokens += span.n_tokens;
        try spans.append(gpa, span);
    }
    return .{ .spans = spans, .tokens = tokens };
}

fn freeGroup(gpa: std.mem.Allocator, g: DocGroup) void {
    gpa.free(g.collection);
    gpa.free(g.rel_path);
    gpa.free(g.title);
    for (g.spans) |s| {
        gpa.free(s.heading_path);
        gpa.free(s.text);
    }
    gpa.free(g.spans);
}

const Range = struct { first: i64, last: i64 };

/// Expand each hit by `neighbors` on both sides and merge overlaps, so two hits
/// three chunks apart come back as one continuous span rather than two fragments
/// with a hole.
fn mergeRanges(
    gpa: std.mem.Allocator,
    idxs: []const i64,
    neighbors: i64,
) !std.ArrayList(Range) {
    const sorted = try gpa.dupe(i64, idxs);
    defer gpa.free(sorted);
    std.mem.sort(i64, sorted, {}, std.sort.asc(i64));

    var out: std.ArrayList(Range) = .empty;
    errdefer out.deinit(gpa);

    for (sorted) |i| {
        const lo = @max(0, i - neighbors);
        const hi = i + neighbors;
        if (out.items.len > 0) {
            const last = &out.items[out.items.len - 1];
            // Touching counts as overlapping: adjacent ranges should merge, not
            // produce two spans with nothing between them.
            if (lo <= last.last + 1) {
                last.last = @max(last.last, hi);
                continue;
            }
        }
        try out.append(gpa, .{ .first = lo, .last = hi });
    }
    return out;
}

/// Concatenate chunks `[first, last]` of a document. Missing indices (the range
/// may extend past the end) are simply absent.
fn loadSpan(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    doc_id: i64,
    first: i64,
    last: i64,
) !Span {
    var st = try db.prepare(
        \\SELECT idx, COALESCE(heading_path, ''), text, n_tokens
        \\FROM chunks WHERE doc_id = ?1 AND idx BETWEEN ?2 AND ?3 ORDER BY idx
    );
    defer st.finalize();
    try st.bindI64(1, doc_id);
    try st.bindI64(2, first);
    try st.bindI64(3, last);

    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(gpa);
    var heading: []u8 = try gpa.dupe(u8, "");
    errdefer gpa.free(heading);

    var tokens: usize = 0;
    var actual_first: i64 = first;
    var actual_last: i64 = first;
    var seen = false;

    while (try st.step()) {
        const idx = st.columnI64(0);
        if (!seen) {
            // The first chunk's heading path names the span.
            gpa.free(heading);
            heading = try gpa.dupe(u8, st.columnText(1));
            actual_first = idx;
            seen = true;
        }
        actual_last = idx;
        if (text.items.len != 0) try text.appendSlice(gpa, "\n\n");
        try text.appendSlice(gpa, st.columnText(2));
        tokens += @intCast(st.columnI64(3));
    }

    return .{
        .first_idx = actual_first,
        .last_idx = actual_last,
        .heading_path = heading,
        .text = try text.toOwnedSlice(gpa),
        .n_tokens = tokens,
    };
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

/// Markdown, meant to be pasted straight into a prompt.
pub fn renderMarkdown(w: *std.Io.Writer, pack: *const Pack) !void {
    try w.print("# Context for: {s}\n", .{pack.query});

    if (pack.dropped_terms.len != 0) {
        try w.writeAll("\n> not searched (unmatchable by the tokenizer):");
        for (pack.dropped_terms) |d| try w.print(" {s}", .{d});
        try w.writeAll("\n");
    }
    if (pack.groups.len == 0) {
        try w.writeAll("\nNo relevant documents found.\n");
        return;
    }

    for (pack.groups) |g| {
        try w.print("\n## {s}/{s}\n", .{ g.collection, g.rel_path });
        for (g.spans) |s| {
            if (s.heading_path.len != 0) try w.print("> {s}\n", .{s.heading_path});
            try w.print("\n{s}\n", .{s.text});
        }
    }

    try w.writeAll("\n---\n");
    if (pack.omitted.len != 0) {
        try w.writeAll("omitted (over budget):");
        for (pack.omitted, 0..) |o, i| {
            if (i != 0) try w.writeAll(",");
            try w.print(" {s} ({d:.3})", .{ o.rel_path, o.score });
        }
        try w.writeAll("\n");
    }
    // Stated as approximate on purpose: this is the embedding model's tokenizer,
    // and whatever model consumes the pack will count differently.
    try w.print(
        "tokens: {d} / {d} (approx, counted with the embedding model's tokenizer)\n",
        .{ pack.total_tokens, pack.budget_tokens },
    );
}

pub fn renderJson(w: *std.Io.Writer, pack: *const Pack) !void {
    try w.writeAll("{\"query\":");
    try std.json.Stringify.value(pack.query, .{}, w);
    try w.print(",\"mode\":\"{t}\",\"total_tokens\":{d},\"budget_tokens\":{d},\"fts_skipped\":{},", .{
        pack.mode, pack.total_tokens, pack.budget_tokens, pack.fts_skipped,
    });
    try w.writeAll("\"dropped_terms\":[");
    for (pack.dropped_terms, 0..) |d, i| {
        if (i != 0) try w.writeAll(",");
        try std.json.Stringify.value(d, .{}, w);
    }
    try w.writeAll("],\"documents\":[");
    for (pack.groups, 0..) |g, i| {
        if (i != 0) try w.writeAll(",");
        try w.writeAll("{\"collection\":");
        try std.json.Stringify.value(g.collection, .{}, w);
        try w.writeAll(",\"path\":");
        try std.json.Stringify.value(g.rel_path, .{}, w);
        try w.writeAll(",\"title\":");
        try std.json.Stringify.value(g.title, .{}, w);
        try w.print(",\"score\":{d:.6},\"n_tokens\":{d},\"spans\":[", .{ g.score, g.n_tokens });
        for (g.spans, 0..) |s, j| {
            if (j != 0) try w.writeAll(",");
            try w.print("{{\"first_idx\":{d},\"last_idx\":{d},\"heading_path\":", .{
                s.first_idx, s.last_idx,
            });
            try std.json.Stringify.value(s.heading_path, .{}, w);
            try w.writeAll(",\"text\":");
            try std.json.Stringify.value(s.text, .{}, w);
            try w.print(",\"n_tokens\":{d}}}", .{s.n_tokens});
        }
        try w.writeAll("]}");
    }
    try w.writeAll("],\"omitted\":[");
    for (pack.omitted, 0..) |o, i| {
        if (i != 0) try w.writeAll(",");
        try w.writeAll("{\"path\":");
        try std.json.Stringify.value(o.rel_path, .{}, w);
        try w.print(",\"score\":{d:.6}}}", .{o.score});
    }
    try w.writeAll("]}");
}
