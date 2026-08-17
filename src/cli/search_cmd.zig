//! `zkb search` — hybrid retrieval against the index.
//!
//! Prints `vec_rank` / `fts_rank` alongside the fused score. Without those two
//! numbers, tuning chunking or the tokenizer is guesswork.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub const Options = struct {
    query: []const u8,
    mode: zkb.hybrid.Mode = .hybrid,
    top_k: usize = 10,
    collection: ?[]const u8 = null,
    model: ?[]const u8 = null,
    json: bool = false,
    /// Print the full chunk text rather than a leading excerpt.
    full: bool = false,
};

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    opts: Options,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    std.Io.Dir.accessAbsolute(io, layout.db, .{}) catch {
        try w.print("no index at {s}\nrun: zkb index\n", .{layout.db});
        return 3;
    };

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch |err| switch (err) {
        // A read-only connection cannot migrate. Say what to run rather than
        // failing inside a DDL statement.
        error.SchemaStale => {
            try w.writeAll("index schema is out of date\nrun: zkb index\n");
            return 3;
        },
        error.SchemaFromFuture => {
            try w.writeAll("index was written by a newer zkb\nupgrade zkb, or: rm ~/.zkb/zkb.db && zkb index\n");
            return 3;
        },
        else => return err,
    };
    defer db.close();

    var collection_id: ?i64 = null;
    if (opts.collection) |name| {
        var st = try db.prepare("SELECT id FROM collections WHERE name = ?1");
        defer st.finalize();
        try st.bindText(1, name);
        if (!try st.step()) {
            try w.print("unknown collection: {s}\n", .{name});
            return 2;
        }
        collection_id = st.columnI64(0);
    }

    // Vector path needs the model; keyword-only does not. Degrading to keyword
    // beats failing outright — a usable answer is worth more than a correct error.
    var query_vec: ?[]f32 = null;
    defer if (query_vec) |v| gpa.free(v);
    var mode = opts.mode;

    if (mode != .keyword) {
        const model_path = opts.model orelse
            try std.fmt.allocPrint(gpa, "{s}/Qwen3-Embedding-0.6B-Q8_0.gguf", .{layout.models});
        defer if (opts.model == null) gpa.free(model_path);

        if (std.Io.Dir.accessAbsolute(io, model_path, .{})) |_| {
            var embedder = try zkb.embed.Embedder.init(gpa, model_path, .{});
            defer embedder.deinit();

            // Query side takes the instruct prefix; documents do not (SPEC §3.3).
            const vec = try gpa.alloc(f32, embedder.n_embd);
            _ = try embedder.embedQuery(zkb.embed.default_query_task, opts.query, vec);
            query_vec = vec;
        } else |_| {
            try w.print("note: model unavailable, falling back to keyword search\n", .{});
            mode = .keyword;
        }
    }

    var results = try zkb.hybrid.search(gpa, &db, mode, opts.query, query_vec, collection_id, .{
        .top_k = opts.top_k,
    });
    defer results.deinit(gpa);

    // Staleness is reported, never hidden: a stopped daemon or a failed document
    // means the answer is incomplete and the user cannot tell otherwise.
    var s = zkb.store.Store.init(&db);
    const counts = try s.counts();

    if (opts.json) {
        try printJson(w, &results, counts);
        return 0;
    }

    if (counts.pending != 0 or counts.failed != 0) {
        try w.print("index incomplete: {d} pending, {d} failed\n\n", .{ counts.pending, counts.failed });
    }
    if (results.dropped_terms.len != 0) {
        try w.writeAll("dropped terms (too short for the trigram index):");
        for (results.dropped_terms) |d| try w.print(" {s}", .{d});
        try w.writeAll("\n");
    }
    if (results.fts_skipped and mode == .hybrid) {
        try w.print("keyword path skipped ({d} hit(s), below threshold)\n", .{results.fts_candidates});
    }
    if (results.hits.len == 0) {
        try w.writeAll("no matches\n");
        return 0;
    }

    try w.print("{d} hit(s), mode {t}\n\n", .{ results.hits.len, mode });
    for (results.hits, 0..) |h, i| {
        try w.print("{d}. {s}/{s}", .{ i + 1, h.collection, h.rel_path });
        try w.print("  [score {d:.5}", .{h.score});
        if (h.vec_rank) |r| try w.print(" vec#{d}", .{r});
        if (h.fts_rank) |r| try w.print(" fts#{d}", .{r});
        try w.print(" chunk {d}]\n", .{h.idx});
        if (h.heading_path.len != 0) try w.print("   {s}\n", .{h.heading_path});
        try w.writeAll("   ");
        try writeExcerpt(w, h.text, if (opts.full) h.text.len else 220);
        try w.writeAll("\n\n");
    }
    return 0;
}

/// Collapse whitespace so a multi-line chunk stays one readable line, and cut on
/// a UTF-8 boundary so the terminal never sees a broken sequence.
fn writeExcerpt(w: *Writer, text: []const u8, max_bytes: usize) !void {
    var end = @min(text.len, max_bytes);
    if (end < text.len) {
        while (end > 0 and (text[end] & 0xC0) == 0x80) end -= 1;
    }
    var last_space = false;
    for (text[0..end]) |c| {
        const is_space = c == '\n' or c == '\r' or c == '\t' or c == ' ';
        if (is_space) {
            if (!last_space) try w.writeByte(' ');
            last_space = true;
        } else {
            try w.writeByte(c);
            last_space = false;
        }
    }
    if (end < text.len) try w.writeAll(" ...");
}

fn printJson(w: *Writer, results: *const zkb.hybrid.Results, counts: zkb.store.Store.Counts) !void {
    try w.print("{{\"mode\":\"{t}\",\"fts_skipped\":{},", .{ results.mode, results.fts_skipped });
    try w.print("\"index\":{{\"docs\":{d},\"chunks\":{d},\"pending\":{d},\"failed\":{d},\"complete\":{}}},", .{
        counts.docs, counts.chunks, counts.pending, counts.failed,
        counts.pending == 0 and counts.failed == 0,
    });
    try w.writeAll("\"dropped_terms\":[");
    for (results.dropped_terms, 0..) |d, i| {
        if (i != 0) try w.writeAll(",");
        try std.json.Stringify.value(d, .{}, w);
    }
    try w.writeAll("],\"hits\":[");
    for (results.hits, 0..) |h, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{{\"chunk_id\":{d},\"score\":{d:.6},", .{ h.chunk_id, h.score });
        if (h.vec_rank) |r| try w.print("\"vec_rank\":{d},", .{r}) else try w.writeAll("\"vec_rank\":null,");
        if (h.fts_rank) |r| try w.print("\"fts_rank\":{d},", .{r}) else try w.writeAll("\"fts_rank\":null,");
        try w.writeAll("\"collection\":");
        try std.json.Stringify.value(h.collection, .{}, w);
        try w.writeAll(",\"path\":");
        try std.json.Stringify.value(h.rel_path, .{}, w);
        try w.writeAll(",\"title\":");
        try std.json.Stringify.value(h.title, .{}, w);
        try w.writeAll(",\"heading_path\":");
        try std.json.Stringify.value(h.heading_path, .{}, w);
        try w.writeAll(",\"text\":");
        try std.json.Stringify.value(h.text, .{}, w);
        try w.print(",\"chunk_idx\":{d},\"n_tokens\":{d}}}", .{ h.idx, h.n_tokens });
    }
    try w.writeAll("]}\n");
}
