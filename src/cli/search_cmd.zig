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
    /// Restrict to documents whose path matches this glob, without splitting the
    /// corpus into another collection.
    path: ?[]const u8 = null,
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

    // Prefer the daemon: its model is already resident, which is the entire
    // reason the daemon exists. Falling back to in-process is not auto-starting
    // anything — it just pays the 1-2s model load here, and says so.
    if (try viaDaemon(gpa, io, layout.sock, w, opts)) |code| return code;

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
        const found = zkb.model_registry.resolve(gpa, io, env, &layout, opts.model, .q8_0) catch null;
        defer if (found) |f| f.deinit(gpa);

        if (found) |f| {
            var embedder = try zkb.embed.Embedder.init(gpa, f.path, .{});
            defer embedder.deinit();

            // Query side takes the instruct prefix; documents do not (SPEC §3.3).
            const vec = try gpa.alloc(f32, embedder.n_embd);
            _ = try embedder.embedQuery(zkb.embed.default_query_task, opts.query, vec);
            query_vec = vec;
        } else {
            try w.print("note: model unavailable, falling back to keyword search\n", .{});
            mode = .keyword;
        }
    }

    const trace_t0: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).nanoseconds, std.time.ns_per_ms));
    var results = try zkb.hybrid.search(gpa, &db, mode, opts.query, query_vec, collection_id, .{
        .path = opts.path,
        .top_k = opts.top_k,
    });
    defer results.deinit(gpa);
    {
        // Same trace as the daemon writes, so a corpus of queries is not split
        // by which path happened to serve them.
        const tw = zkb.trace.Writer.init(io, env, layout.trace);
        const now: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).nanoseconds, std.time.ns_per_ms));
        tw.record(gpa, opts.query, &results, now - trace_t0);
    }

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
    const end = zkb.utf8.truncate(text, max_bytes);
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
        counts.docs,                                counts.chunks, counts.pending, counts.failed,
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

/// Returns null when the daemon is unavailable, so the caller falls through to
/// the in-process path.
/// The exact params `search` puts on the wire. See `query_cmd.requestParams`.
pub fn requestParams(w: *Writer, opts: Options) !void {
    try w.writeAll("{\"query\":");
    try std.json.Stringify.value(opts.query, .{}, w);
    try w.print(",\"k\":{d},\"mode\":\"{t}\"", .{ opts.top_k, opts.mode });
    // 不带上这两个过滤器的话，`--collection x` 在有 daemon 时被静默忽略：命令照常
    // 返回结果，只是过滤没生效——比报错更难发现，因为输出看起来完全正常。
    try zkb.proto.writeFilters(w, opts.collection, opts.path);
    try w.writeAll("}");
}

fn viaDaemon(
    gpa: std.mem.Allocator,
    io: std.Io,
    sock: []const u8,
    w: *Writer,
    opts: Options,
) !?u8 {
    var c = zkb.ipc_client.Client.connect(io, sock) catch return null;
    defer c.close();

    // Params are small (a query string plus two numbers), so a fixed buffer is
    // simpler than growing an ArrayList and cannot fail mid-write.
    var pbuf: [8192]u8 = undefined;
    var pw = std.Io.Writer.fixed(&pbuf);
    try requestParams(&pw, opts);

    var resp = c.call(gpa, .search, pw.buffered()) catch return null;
    defer resp.deinit(gpa);

    if (!resp.ok) {
        try w.print("{s}: {s}\n", .{ resp.code, resp.message });
        if (resp.hint) |h| try w.print("hint: {s}\n", .{h});
        return zkb.proto.ErrorCode.exitCodeOf(resp.code);
    }
    if (opts.json) {
        // Hand the daemon's own JSON through unchanged: re-serializing would risk
        // the two shapes drifting apart.
        try w.print("{s}\n", .{resp.line});
        return 0;
    }
    try renderDaemonResult(w, resp.result.?, opts);
    return 0;
}

fn renderDaemonResult(w: *Writer, result: std.json.Value, opts: Options) !void {
    const obj = result.object;
    if (obj.get("index")) |idx| if (idx == .object) {
        const o = idx.object;
        const pending = jsonInt(o, "pending");
        const failed = jsonInt(o, "failed");
        if (pending != 0 or failed != 0) {
            try w.print("index incomplete: {d} pending, {d} failed\n\n", .{ pending, failed });
        }
    };
    if (obj.get("degraded")) |d| if (d == .string) {
        try w.print("DEGRADED: {s}\n", .{d.string});
    };
    if (obj.get("dropped_terms")) |dt| if (dt == .array and dt.array.items.len != 0) {
        try w.writeAll("dropped terms (unmatchable by the tokenizer):");
        for (dt.array.items) |t| if (t == .string) try w.print(" {s}", .{t.string});
        try w.writeAll("\n");
    };

    const hits = if (obj.get("hits")) |h| (if (h == .array) h.array.items else &.{}) else &.{};
    if (hits.len == 0) {
        try w.writeAll("no matches\n");
        return;
    }
    const mode = if (obj.get("mode")) |m| (if (m == .string) m.string else "?") else "?";
    try w.print("{d} hit(s), mode {s}\n\n", .{ hits.len, mode });

    for (hits, 1..) |h, i| {
        const o = h.object;
        try w.print("{d}. {s}/{s}", .{ i, jsonStr(o, "collection"), jsonStr(o, "path") });
        try w.print("  [score {d:.5}", .{jsonFloat(o, "score")});
        if (o.get("vec_rank")) |v| if (v == .integer) try w.print(" vec#{d}", .{v.integer});
        if (o.get("fts_rank")) |v| if (v == .integer) try w.print(" fts#{d}", .{v.integer});
        try w.print(" chunk {d}]\n", .{jsonInt(o, "chunk_idx")});
        const heading = jsonStr(o, "heading_path");
        if (heading.len != 0) try w.print("   {s}\n", .{heading});
        try w.writeAll("   ");
        try writeExcerpt(w, jsonStr(o, "text"), if (opts.full) std.math.maxInt(usize) else 220);
        try w.writeAll("\n\n");
    }
}

fn jsonStr(o: std.json.ObjectMap, key: []const u8) []const u8 {
    const v = o.get(key) orelse return "";
    return switch (v) {
        .string => |s| s,
        else => "",
    };
}

fn jsonInt(o: std.json.ObjectMap, key: []const u8) i64 {
    const v = o.get(key) orelse return 0;
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => 0,
    };
}

fn jsonFloat(o: std.json.ObjectMap, key: []const u8) f64 {
    const v = o.get(key) orelse return 0;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
}
