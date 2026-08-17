//! `zkb query` — retrieval plus assembly, output shaped for a prompt.
//!
//! zkb does not answer the question. The caller is an LLM already; assembling
//! the right context and handing it over is the part zkb can do better, and
//! generating the answer is the part it would only duplicate (REQ D1).

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub const Format = enum { markdown, json };

pub const Options = struct {
    query: []const u8,
    budget: usize = 8000,
    neighbors: i64 = 1,
    candidates: usize = 30,
    format: Format = .markdown,
    model: ?[]const u8 = null,
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

    if (try viaDaemon(gpa, io, layout.sock, w, opts)) |code| return code;
    return inProcess(gpa, io, &layout, w, opts);
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

    var pbuf: [8192]u8 = undefined;
    var pw = std.Io.Writer.fixed(&pbuf);
    try pw.writeAll("{\"query\":");
    try std.json.Stringify.value(opts.query, .{}, &pw);
    try pw.print(",\"budget\":{d},\"neighbors\":{d},\"candidates\":{d}}}", .{
        opts.budget, opts.neighbors, opts.candidates,
    });

    var resp = c.call(gpa, .query, pw.buffered()) catch return null;
    defer resp.deinit(gpa);

    if (!resp.ok) {
        try w.print("{s}: {s}\n", .{ resp.code, resp.message });
        if (resp.hint) |h| try w.print("hint: {s}\n", .{h});
        return 4;
    }
    if (opts.format == .json) {
        // The daemon's JSON is the canonical shape; passing it through unchanged
        // keeps the two renderings from drifting.
        try w.print("{s}\n", .{resp.line});
        return 0;
    }
    try renderFromJson(w, resp.result.?, opts.query);
    return 0;
}

/// Same output as the daemon path, rebuilt from its JSON. The markdown lives here
/// rather than in the daemon so the wire format stays one thing.
fn renderFromJson(w: *Writer, result: std.json.Value, query: []const u8) !void {
    const obj = result.object;
    try w.print("# Context for: {s}\n", .{query});

    if (obj.get("dropped_terms")) |dt| if (dt == .array and dt.array.items.len != 0) {
        try w.writeAll("\n> not searched (unmatchable by the tokenizer):");
        for (dt.array.items) |t| if (t == .string) try w.print(" {s}", .{t.string});
        try w.writeAll("\n");
    };

    const docs = if (obj.get("documents")) |d| (if (d == .array) d.array.items else &.{}) else &.{};
    if (docs.len == 0) {
        try w.writeAll("\nNo relevant documents found.\n");
        return;
    }

    for (docs) |d| {
        const o = d.object;
        try w.print("\n## {s}/{s}\n", .{ str(o, "collection"), str(o, "path") });
        const spans = if (o.get("spans")) |sp| (if (sp == .array) sp.array.items else &.{}) else &.{};
        for (spans) |sv| {
            const so = sv.object;
            const heading = str(so, "heading_path");
            if (heading.len != 0) try w.print("> {s}\n", .{heading});
            try w.print("\n{s}\n", .{str(so, "text")});
        }
    }

    try w.writeAll("\n---\n");
    if (obj.get("omitted")) |om| if (om == .array and om.array.items.len != 0) {
        try w.writeAll("omitted (over budget):");
        for (om.array.items, 0..) |ov, i| {
            if (i != 0) try w.writeAll(",");
            try w.print(" {s}", .{str(ov.object, "path")});
        }
        try w.writeAll("\n");
    };
    try w.print(
        "tokens: {d} / {d} (approx, counted with the embedding model's tokenizer)\n",
        .{ int(obj, "total_tokens"), int(obj, "budget_tokens") },
    );
}

/// Without a daemon the model has to be loaded here, which costs 1-2s. Reported
/// rather than hidden, because it is the difference the daemon exists to remove.
fn inProcess(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const zkb.paths.Layout,
    w: *Writer,
    opts: Options,
) !u8 {
    std.Io.Dir.accessAbsolute(io, layout.db, .{}) catch {
        try w.print("no index at {s}\nrun: zkb index\n", .{layout.db});
        return 3;
    };

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch |err| switch (err) {
        error.SchemaStale => {
            try w.writeAll("index schema is out of date\nrun: zkb index\n");
            return 3;
        },
        else => return err,
    };
    defer db.close();

    var mode: zkb.hybrid.Mode = .hybrid;
    var vec: ?[]f32 = null;
    defer if (vec) |v| gpa.free(v);

    const model_path = opts.model orelse
        try std.fmt.allocPrint(gpa, "{s}/Qwen3-Embedding-0.6B-Q8_0.gguf", .{layout.models});
    defer if (opts.model == null) gpa.free(model_path);

    if (std.Io.Dir.accessAbsolute(io, model_path, .{})) |_| {
        var embedder = try zkb.embed.Embedder.init(gpa, model_path, .{});
        defer embedder.deinit();
        const buf = try gpa.alloc(f32, embedder.n_embd);
        _ = try embedder.embedQuery(zkb.embed.default_query_task, opts.query, buf);
        vec = buf;
    } else |_| {
        try w.writeAll("note: model unavailable, keyword-only context\n");
        mode = .keyword;
    }

    var results = try zkb.hybrid.search(gpa, &db, mode, opts.query, vec, null, .{
        .top_k = opts.candidates,
        .candidates = @max(50, opts.candidates),
    });
    defer results.deinit(gpa);

    var p = try zkb.pack.assemble(gpa, &db, opts.query, &results, .{
        .budget_tokens = opts.budget,
        .neighbors = opts.neighbors,
        .candidates = opts.candidates,
    });
    defer p.deinit(gpa);

    switch (opts.format) {
        .markdown => try zkb.pack.renderMarkdown(w, &p),
        .json => {
            try zkb.pack.renderJson(w, &p);
            try w.writeAll("\n");
        },
    }
    return 0;
}

fn str(o: std.json.ObjectMap, key: []const u8) []const u8 {
    const v = o.get(key) orelse return "";
    return switch (v) {
        .string => |s| s,
        else => "",
    };
}

fn int(o: std.json.ObjectMap, key: []const u8) i64 {
    const v = o.get(key) orelse return 0;
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => 0,
    };
}
