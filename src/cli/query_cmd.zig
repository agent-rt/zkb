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
    /// Restrict to one collection, by name.
    collection: ?[]const u8 = null,
    /// Restrict to documents whose rel_path matches this glob.
    ///
    /// `search` has had both of these since the filters existed; `query` did not,
    /// and the two answer the same question about the same corpus. Asking about
    /// one project meant either getting a context pack drawn from everything, or
    /// dropping to `search` and giving up the assembly that is the point of
    /// `query`. A question about one project came back full of methodology docs
    /// that merely shared the word 阶段.
    path: ?[]const u8 = null,
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
    const tw = zkb.trace.Writer.init(io, env, layout.trace);
    return inProcess(gpa, io, env, &layout, w, opts, tw);
}

/// The exact params `query` puts on the wire.
///
/// Split out of `viaDaemon` so a test can read them. A filter reaching only the
/// in-process path is this codebase's most repeated bug, and the reason it keeps
/// surviving review is that the daemon branch is a few lines buried inside a
/// function that also opens a socket — nothing a test can call.
pub fn requestParams(w: *Writer, opts: Options) !void {
    try w.writeAll("{\"query\":");
    try std.json.Stringify.value(opts.query, .{}, w);
    try w.print(",\"budget\":{d},\"neighbors\":{d},\"candidates\":{d}", .{
        opts.budget, opts.neighbors, opts.candidates,
    });
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

    var pbuf: [8192]u8 = undefined;
    var pw = std.Io.Writer.fixed(&pbuf);
    try requestParams(&pw, opts);

    var resp = c.call(gpa, .query, pw.buffered()) catch return null;
    defer resp.deinit(gpa);

    if (!resp.ok) {
        try w.print("{s}: {s}\n", .{ resp.code, resp.message });
        if (resp.hint) |h| try w.print("hint: {s}\n", .{h});
        return zkb.proto.ErrorCode.exitCodeOf(resp.code);
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
    env: *const std.process.Environ.Map,
    layout: *const zkb.paths.Layout,
    w: *Writer,
    opts: Options,
    tw: zkb.trace.Writer,
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

    // Resolved before the model is loaded: an unknown name is a typo, and paying
    // 600 MB of model load to then reject the argument helps nobody.
    var collection_id: ?i64 = null;
    if (opts.collection) |name| {
        var s = zkb.store.Store.init(&db);
        collection_id = try s.findCollection(name) orelse {
            try w.print("unknown collection: {s}\nzkb status lists them\n", .{name});
            return 2;
        };
    }

    var mode: zkb.hybrid.Mode = .hybrid;
    var vec: ?[]f32 = null;
    defer if (vec) |v| gpa.free(v);

    const found = zkb.model_registry.resolve(gpa, io, env, layout, opts.model, .q8_0) catch null;
    defer if (found) |f| f.deinit(gpa);

    if (found) |f| {
        var embedder = try zkb.embed.Embedder.init(gpa, f.path, .{});
        defer embedder.deinit();
        const buf = try gpa.alloc(f32, embedder.n_embd);
        _ = try embedder.embedQuery(zkb.embed.default_query_task, opts.query, buf);
        vec = buf;
    } else {
        try w.writeAll("note: model unavailable, keyword-only context\n");
        mode = .keyword;
    }

    const trace_t0: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).nanoseconds, std.time.ns_per_ms));
    var results = try zkb.hybrid.search(gpa, &db, mode, opts.query, vec, collection_id, .{
        .top_k = opts.candidates,
        .candidates = @max(50, opts.candidates),
        .path = opts.path,
    });
    defer results.deinit(gpa);
    {
        // Same trace as the daemon writes, so a corpus of queries is not split
        // by which path happened to serve them.
        const now: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).nanoseconds, std.time.ns_per_ms));
        tw.record(gpa, opts.query, &results, now - trace_t0);
    }

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

const testing = std.testing;

test "every retrieval command puts its filters on the wire" {
    // The bug this exists for: `--collection` / `--path` parsed by the CLI, applied
    // in process, and dropped on the way to the daemon. Nothing fails — the command
    // prints results, and they are simply drawn from the whole corpus. It shipped
    // once with `--path`, and `query` spent its whole life this way because the
    // flags were never wired at all.
    //
    // Each command builds its params here, so a new one that forgets
    // `proto.writeFilters` fails on the row someone adds for it, and a new filter
    // added to `writeFilters` reaches all of them at once.
    const search_cmd = @import("search_cmd.zig");

    var buf: [1024]u8 = undefined;
    inline for (.{
        .{ "query", requestParams, Options{ .query = "q", .collection = "docs", .path = "projects/qlit/**" } },
        .{ "search", search_cmd.requestParams, search_cmd.Options{ .query = "q", .collection = "docs", .path = "projects/qlit/**" } },
    }) |row| {
        var w = std.Io.Writer.fixed(&buf);
        const build = row[1];
        try build(&w, row[2]);

        const line = try std.fmt.allocPrint(
            testing.allocator,
            "{{\"id\":1,\"method\":\"search\",\"params\":{s}}}",
            .{w.buffered()},
        );
        defer testing.allocator.free(line);
        var req = try zkb.proto.parseRequest(testing.allocator, line);
        defer req.deinit();

        try testing.expectEqualStrings("docs", req.str("collection") orelse
            return testing.expect(false) catch @panic(row[0] ++ ": collection never reached the wire"));
        try testing.expectEqualStrings("projects/qlit/**", req.str("path") orelse
            return testing.expect(false) catch @panic(row[0] ++ ": path never reached the wire"));
    }
}

test "an unset filter stays absent, rather than becoming an empty string" {
    // `""` is a glob that matches nothing, so writing the field unconditionally
    // would turn "no filter" into "filter everything out" — the empty-list
    // inversion, one layer up from where it was caught before.
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try requestParams(&w, .{ .query = "q" });
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "collection") == null);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "path") == null);
}

test "the filters hold on the path that has no daemon" {
    // The other half of `daemon_test`'s socket test, and the half sabotage found
    // uncovered: dropping `collection_id` on the way into `hybrid.search` here
    // broke nothing that ran. Both paths answer the same question, so both are
    // asserted — a filter honoured by one of them is the bug, not the feature.
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var root_buf: [64]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/zkb-query-test-{x}", .{@intFromPtr(&threaded)});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};

    {
        const index_dir = try std.fmt.allocPrint(testing.allocator, "{s}/index", .{root});
        defer testing.allocator.free(index_dir);
        var d = try std.Io.Dir.cwd().createDirPathOpen(io, index_dir, .{});
        d.close(io);
        const db_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/zkb.db", .{index_dir}, 0);
        defer testing.allocator.free(db_path);
        var db = try zkb.store.open(db_path, .read_write);
        defer db.close();
        var s = zkb.store.Store.init(&db);
        const cid = try s.ensureCollection("corpus", "/tmp/corpus", 1000);
        // `decoy` holds the same rel_path as the wanted document, so `--path a/**`
        // alone cannot tell them apart. Without it the collection filter is a
        // no-op on this fixture and dropping it changes no assertion — which is
        // precisely how the in-process half went uncovered in the first place.
        const decoy = try s.ensureCollection("decoy", "/tmp/decoy", 1000);
        for ([_][2][]const u8{
            .{ "corpus", "a/one.md" },
            .{ "corpus", "b/two.md" },
            .{ "decoy", "a/one.md" },
        }) |row| {
            const in_corpus = std.mem.eql(u8, row[0], "corpus");
            const cur = if (in_corpus) cid else decoy;
            const rel = row[1];
            const body = "RRF fusion merges the two rankers.";
            const did = try s.upsertDocContent(cur, rel, row[0], body.len, 1000);
            var v: [@intCast(zkb.schema.embedding_dim)]f32 = @splat(0);
            v[0] = 1;
            _ = try s.insertChunk(cur, did, .{
                .idx = 0,
                .heading_path = "",
                .byte_start = 0,
                .byte_end = @intCast(body.len),
                .n_tokens = 8,
                .text = body,
            }, &v);
            try s.markIndexed(did, 1, 1000);
        }
    }

    // Only ZKB_HOME: no HOME and no HF_* means no model is reachable, so this
    // stays on the keyword path and never loads 600 MB to assert a glob.
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    try env.put("ZKB_HOME", root);

    var out: [16 * 1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&out);
    const code = try run(testing.allocator, io, &env, &w, .{
        .query = "fusion",
        .budget = 2000,
        .collection = "corpus",
        .path = "a/**",
    });
    try testing.expectEqual(@as(u8, 0), code);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "corpus/a/one.md") != null);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "b/two.md") == null);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "decoy/") == null);
}

test "an unknown collection is refused in process too, not quietly dropped" {
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var root_buf: [64]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/zkb-query-unk-{x}", .{@intFromPtr(&threaded)});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    {
        const index_dir = try std.fmt.allocPrint(testing.allocator, "{s}/index", .{root});
        defer testing.allocator.free(index_dir);
        var d = try std.Io.Dir.cwd().createDirPathOpen(io, index_dir, .{});
        d.close(io);
        const db_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/zkb.db", .{index_dir}, 0);
        defer testing.allocator.free(db_path);
        var db = try zkb.store.open(db_path, .read_write);
        db.close();
    }

    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    try env.put("ZKB_HOME", root);

    var out: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&out);
    const code = try run(testing.allocator, io, &env, &w, .{ .query = "q", .collection = "nope" });
    try testing.expectEqual(@as(u8, 2), code);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "unknown collection") != null);
}
