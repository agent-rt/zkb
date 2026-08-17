//! `zkb mcp` — MCP stdio server, forwarding to the daemon over the unix socket.
//!
//! JSON-RPC 2.0, newline-delimited, on stdin/stdout. **Nothing may be written to
//! stdout except protocol frames** — a stray log line corrupts the stream and the
//! client sees a parse error rather than a diagnostic. Diagnostics go to stderr.
//!
//! Four tools. Tool definitions are re-sent to the model on every turn, so each
//! one costs context permanently; `search`, `query`, `recall` and `records` are
//! what a resource URI cannot express (SPEC §8). Anything readable is a resource
//! instead, which costs no tool slot.
//!
//! `records` deliberately covers introspection *and* querying in one slot:
//! calling it with no type lists the types and their inferred schemas. Two tools
//! would cost twice the context to say the same thing.
//!
//! No write tools. Over documents that is because zkb only reads them — an agent
//! that wants to change a file uses its own editing tools and the next scan picks
//! it up. Over memories it is because the daemon keeps a single writer (the
//! ingest thread), which is what keeps SQLITE_BUSY off the table; `zkb remember`
//! is a shell command instead, which an agent can run just as easily.

const std = @import("std");
const zkb = @import("zkb");
const client = zkb.ipc_client;
const records_cmd = @import("../cli/records_cmd.zig");

const Writer = std.Io.Writer;

/// Echoed back to the client when it asks for one we understand.
const supported_protocol = "2025-06-18";
const fallback_protocol = "2024-11-05";

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    var in_buf: [1 << 20]u8 = undefined;
    var out_buf: [1 << 20]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(io, &in_buf);
    var stdout = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const w = &stdout.interface;

    while (true) {
        const raw = stdin.interface.takeDelimiterInclusive('\n') catch |err| switch (err) {
            error.EndOfStream => return 0,
            else => return 0,
        };
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        if (line.len == 0) continue;

        handle(gpa, io, env, layout.sock, w, line) catch |err| {
            std.debug.print("zkb mcp: {t}\n", .{err});
        };
        w.flush() catch return 0;
    }
}

fn handle(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    sock: []const u8,
    w: *Writer,
    line: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, line, .{}) catch {
        return writeError(w, .{ .null = {} }, -32700, "parse error");
    };
    defer parsed.deinit();
    if (parsed.value != .object) return writeError(w, .{ .null = {} }, -32600, "invalid request");
    const obj = parsed.value.object;

    const method = switch (obj.get("method") orelse .null) {
        .string => |s| s,
        else => return writeError(w, obj.get("id") orelse .null, -32600, "missing method"),
    };
    // Notifications carry no id and must produce no response at all — replying to
    // one is a protocol violation the client reports as an unmatched response.
    const id = obj.get("id") orelse {
        return;
    };
    const params = obj.get("params") orelse std.json.Value{ .object = .empty };

    if (std.mem.eql(u8, method, "initialize")) return handleInitialize(w, id, params);
    if (std.mem.eql(u8, method, "ping")) {
        try beginResult(w, id);
        try w.writeAll("{}");
        return endResult(w);
    }
    if (std.mem.eql(u8, method, "tools/list")) return handleToolsList(w, id);
    if (std.mem.eql(u8, method, "resources/list")) return handleResourcesList(w, id);
    if (std.mem.eql(u8, method, "tools/call")) return handleToolsCall(gpa, io, env, sock, w, id, params);
    if (std.mem.eql(u8, method, "resources/read")) return handleResourcesRead(gpa, io, sock, w, id, params);

    return writeError(w, id, -32601, "method not found");
}

fn handleInitialize(w: *Writer, id: std.json.Value, params: std.json.Value) !void {
    // Echo the client's version when it is one we speak; otherwise offer ours and
    // let the client decide whether to continue.
    var version: []const u8 = fallback_protocol;
    if (params == .object) {
        if (params.object.get("protocolVersion")) |v| if (v == .string) {
            if (std.mem.eql(u8, v.string, supported_protocol) or
                std.mem.eql(u8, v.string, fallback_protocol))
            {
                version = v.string;
            } else version = supported_protocol;
        };
    }

    try beginResult(w, id);
    try w.print(
        "{{\"protocolVersion\":\"{s}\",\"capabilities\":{{\"tools\":{{}},\"resources\":{{}}}}," ++
            "\"serverInfo\":{{\"name\":\"zkb\",\"version\":\"{s}\"}}}}",
        .{ version, zkb.version },
    );
    try endResult(w);
}

fn handleToolsList(w: *Writer, id: std.json.Value) !void {
    try beginResult(w, id);
    try writeCompact(w,
        \\{"tools":[
        \\{"name":"zkb_search",
        \\ "description":"Search the personal knowledge base. Returns matching chunks with their file path and heading path. Use when you need to locate where something is written; use zkb_query when you need context to answer with.",
        \\ "inputSchema":{"type":"object","properties":{
        \\   "query":{"type":"string","description":"Natural language or keywords. Chinese and English both work."},
        \\   "k":{"type":"integer","description":"Number of chunks to return (default 10)."},
        \\   "mode":{"type":"string","enum":["hybrid","vector","keyword"],"description":"Default hybrid. keyword works even when the embedding model is unavailable."}},
        \\  "required":["query"]}},
        \\{"name":"zkb_query",
        \\ "description":"Retrieve and assemble context for a question: neighbouring chunks pulled in, grouped per document, trimmed to a token budget. Returns markdown ready to reason over. Does not answer the question.",
        \\ "inputSchema":{"type":"object","properties":{
        \\   "query":{"type":"string","description":"The question to gather context for."},
        \\   "budget":{"type":"integer","description":"Token budget for the assembled context (default 8000)."}},
        \\  "required":["query"]}},
        \\{"name":"zkb_recall",
        \\ "description":"What you should know about this user before answering: their recorded preferences, decisions and corrections, plus the current value of every stored fact. Call this at the start of a session, and again when the topic shifts. Facts are exact values, not retrieved text — trust them over anything a document says. To record something new, run the shell command: zkb remember \"...\"",
        \\ "inputSchema":{"type":"object","properties":{
        \\   "query":{"type":"string","description":"Optional. Omit at session start to get the most recent memories; pass a topic to bias towards memories about it. Facts are always included either way."},
        \\   "budget":{"type":"integer","description":"Token budget for the memories (default 1500). Facts are not counted against it."}}}},
        \\{"name":"zkb_records",
        \\ "description":"Query the user's structured data (expenses, weight logs, any csv under records/). Exact filtering and aggregation over real columns — use this for numbers, not zkb_search. Call with no arguments to list the available types and their columns. Output is TSV.",
        \\ "inputSchema":{"type":"object","properties":{
        \\   "type":{"type":"string","description":"Record type, e.g. 'expenses'. Omit to list all types and their inferred schemas."},
        \\   "where":{"type":"string","description":"Filter, e.g. \"amount > 1000 AND category = 'food'\". Operators: = != < <= > >= LIKE IN, IS NULL; AND/OR and parentheses. Field names must match the schema."},
        \\   "agg":{"type":"string","description":"Aggregate, e.g. \"sum(amount) by category\". Functions: sum avg min max count."},
        \\   "window":{"type":"string","description":"Moving aggregate, e.g. \"avg(kg) over 7 by date\" for a 7-row moving average. Optional trailing 'partition <field>'."},
        \\   "limit":{"type":"integer","description":"Max rows for a plain listing (default 50)."},
        \\   "schema":{"type":"boolean","description":"Show the inferred column kinds (date/number/enum/id/string) instead of rows."}}}}
        \\]}
    );
    try endResult(w);
}

fn handleResourcesList(w: *Writer, id: std.json.Value) !void {
    try beginResult(w, id);
    try writeCompact(w,
        \\{"resources":[
        \\{"uri":"zkb://stats","name":"Index status","description":"Document and chunk counts, pending and failed work, embedding model identity.","mimeType":"application/json"},
        \\{"uri":"zkb://health","name":"Daemon health","description":"Uptime, whether the model is resident, embedding queue depth, degraded reason if any.","mimeType":"application/json"}
        \\]}
    );
    try endResult(w);
}

fn handleToolsCall(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    sock: []const u8,
    w: *Writer,
    id: std.json.Value,
    params: std.json.Value,
) !void {
    if (params != .object) return writeError(w, id, -32602, "invalid params");
    const name = switch (params.object.get("name") orelse .null) {
        .string => |s| s,
        else => return writeError(w, id, -32602, "missing tool name"),
    };
    const args = params.object.get("arguments") orelse std.json.Value{ .object = .empty };
    if (args != .object) return writeError(w, id, -32602, "arguments must be an object");

    // Structured queries touch no model and need no daemon: they are SQL over
    // an index that is already on disk. Handled before the socket connect below,
    // so `zkb_records` keeps working when the daemon is down.
    if (std.mem.eql(u8, name, "zkb_records")) {
        return recordsTool(gpa, io, env, w, id, args.object);
    }

    // Optional here, required per tool: `zkb_recall` with no query is the
    // session-start case, where nothing has been asked yet.
    const query = switch (args.object.get("query") orelse .null) {
        .string => |s| s,
        else => "",
    };
    if (query.len == 0 and !std.mem.eql(u8, name, "zkb_recall")) {
        return toolError(w, id, "query is required");
    }

    var c = client.Client.connect(io, sock) catch {
        // Surfaced as a tool error rather than a protocol error: the model can act
        // on "start the daemon", but a JSON-RPC error just aborts the call.
        return toolError(w, id, "zkb daemon is not running. Run: zkb daemon start");
    };
    defer c.close();

    var pbuf: [16384]u8 = undefined;
    var pw = std.Io.Writer.fixed(&pbuf);

    if (std.mem.eql(u8, name, "zkb_search")) {
        const k = intArg(args.object, "k") orelse 10;
        const mode = strArg(args.object, "mode") orelse "hybrid";
        try pw.writeAll("{\"query\":");
        try std.json.Stringify.value(query, .{}, &pw);
        try pw.print(",\"k\":{d},\"mode\":", .{k});
        try std.json.Stringify.value(mode, .{}, &pw);
        try pw.writeAll("}");

        var resp = c.call(gpa, .search, pw.buffered()) catch
            return toolError(w, id, "daemon did not answer");
        defer resp.deinit(gpa);
        if (!resp.ok) return toolError(w, id, resp.message);

        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        try renderSearchText(gpa, &text, resp.result.?);
        return toolText(w, id, text.items);
    }

    if (std.mem.eql(u8, name, "zkb_query")) {
        const budget = intArg(args.object, "budget") orelse 8000;
        try pw.writeAll("{\"query\":");
        try std.json.Stringify.value(query, .{}, &pw);
        try pw.print(",\"budget\":{d}}}", .{budget});

        var resp = c.call(gpa, .query, pw.buffered()) catch
            return toolError(w, id, "daemon did not answer");
        defer resp.deinit(gpa);
        if (!resp.ok) return toolError(w, id, resp.message);

        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        try renderPackText(gpa, &text, query, resp.result.?);
        return toolText(w, id, text.items);
    }

    if (std.mem.eql(u8, name, "zkb_recall")) {
        const budget = intArg(args.object, "budget") orelse 1500;
        try pw.writeAll("{\"query\":");
        try std.json.Stringify.value(query, .{}, &pw);
        try pw.print(",\"budget\":{d}}}", .{budget});

        var resp = c.call(gpa, .recall, pw.buffered()) catch
            return toolError(w, id, "daemon did not answer");
        defer resp.deinit(gpa);
        if (!resp.ok) return toolError(w, id, resp.message);

        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        try renderRecallText(gpa, &text, resp.result.?);
        return toolText(w, id, text.items);
    }

    return writeError(w, id, -32602, "unknown tool");
}

/// Reuses the CLI path verbatim rather than restating the query builder here.
///
/// **No `search` argument.** Semantic search over records already works through
/// `zkb_search` — a record row is a chunk in the same retrieval space as a
/// document. The one thing that would be new is `--search` combined with
/// `--where` (the prefilter path), and that needs the embedding model: loading
/// 620 MB into a long-lived stdio server to save one tool call is the wrong
/// trade until real use says otherwise.
fn recordsTool(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    id: std.json.Value,
    args: std.json.ObjectMap,
) !void {
    var opts: records_cmd.Options = .{
        .type_name = strArg(args, "type"),
        .where = strArg(args, "where"),
        .agg = strArg(args, "agg"),
        .window = strArg(args, "window"),
    };
    if (intArg(args, "limit")) |n| opts.limit = @intCast(@max(1, n));
    // Explicit rather than inferred from "no other argument": a type with no
    // filter most obviously means "show me the rows", and the TSV header already
    // carries the column names an agent needs to write a filter. `schema` is for
    // when it also needs the inferred kinds.
    opts.show_schema = boolArg(args, "schema");

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();

    const code = records_cmd.run(gpa, io, env, &out.writer, opts) catch |err| {
        return toolError(w, id, @errorName(err));
    };
    if (code != 0) return toolError(w, id, out.written());
    return toolText(w, id, out.written());
}

/// Facts first and labelled as exact values: the failure this guards against is
/// a model reading "salary: 480000" as just another retrieved sentence and then
/// preferring an older number it found in prose.
fn renderRecallText(gpa: std.mem.Allocator, out: *std.ArrayList(u8), result: std.json.Value) !void {
    const obj = result.object;
    if (obj.get("facts")) |fv| if (fv == .array and fv.array.items.len != 0) {
        try out.appendSlice(gpa, "## Facts (exact current values)\n\n");
        for (fv.array.items) |item| {
            const o = item.object;
            try out.print(gpa, "- {s}: {s}  (as of {s})", .{
                jsonStr(o, "key"), jsonStr(o, "value"), jsonStr(o, "at"),
            });
            const note = jsonStr(o, "note");
            if (note.len != 0) try out.print(gpa, " — {s}", .{note});
            try out.appendSlice(gpa, "\n");
        }
        try out.appendSlice(gpa, "\n");
    };

    const mem = obj.get("memories") orelse return;
    const docs = if (mem.object.get("documents")) |d|
        (if (d == .array) d.array.items else &.{})
    else
        &.{};
    if (docs.len == 0) {
        try out.appendSlice(gpa, "No memories recorded yet.\n");
        return;
    }
    try out.appendSlice(gpa, "## Memories\n");
    for (docs) |d| {
        const o = d.object;
        try out.print(gpa, "\n### {s}\n", .{jsonStr(o, "path")});
        const spans = if (o.get("spans")) |sp| (if (sp == .array) sp.array.items else &.{}) else &.{};
        for (spans) |sv| try out.print(gpa, "\n{s}\n", .{jsonStr(sv.object, "text")});
    }
}

fn handleResourcesRead(
    gpa: std.mem.Allocator,
    io: std.Io,
    sock: []const u8,
    w: *Writer,
    id: std.json.Value,
    params: std.json.Value,
) !void {
    if (params != .object) return writeError(w, id, -32602, "invalid params");
    const uri = switch (params.object.get("uri") orelse .null) {
        .string => |s| s,
        else => return writeError(w, id, -32602, "missing uri"),
    };

    const method: zkb.proto.Method = if (std.mem.eql(u8, uri, "zkb://stats"))
        .stats
    else if (std.mem.eql(u8, uri, "zkb://health"))
        .health
    else
        return writeError(w, id, -32602, "unknown resource uri");

    var c = client.Client.connect(io, sock) catch
        return writeError(w, id, -32000, "zkb daemon is not running");
    defer c.close();

    var resp = c.call(gpa, method, "{}") catch
        return writeError(w, id, -32000, "daemon did not answer");
    defer resp.deinit(gpa);

    try beginResult(w, id);
    try w.writeAll("{\"contents\":[{\"uri\":");
    try std.json.Stringify.value(uri, .{}, w);
    try w.writeAll(",\"mimeType\":\"application/json\",\"text\":");
    // The daemon's own JSON, re-encoded as a string payload.
    var buf: [1 << 16]u8 = undefined;
    var bw = std.Io.Writer.fixed(&buf);
    if (resp.result) |r| try std.json.Stringify.value(r, .{}, &bw) else try bw.writeAll("null");
    try std.json.Stringify.value(bw.buffered(), .{}, w);
    try w.writeAll("}]}");
    try endResult(w);
}

// ---------------------------------------------------------------------------
// rendering: tools return text, because that is what a model reads
// ---------------------------------------------------------------------------

fn renderSearchText(gpa: std.mem.Allocator, out: *std.ArrayList(u8), result: std.json.Value) !void {
    const obj = result.object;
    const hits = if (obj.get("hits")) |h| (if (h == .array) h.array.items else &.{}) else &.{};

    if (obj.get("index")) |idx| if (idx == .object) {
        const pending = jsonInt(idx.object, "pending");
        const failed = jsonInt(idx.object, "failed");
        if (pending != 0 or failed != 0) {
            try out.print(gpa, "index incomplete: {d} pending, {d} failed\n\n", .{ pending, failed });
        }
    };
    if (obj.get("degraded")) |d| if (d == .string) {
        try out.print(gpa, "DEGRADED: {s}\n\n", .{d.string});
    };
    if (hits.len == 0) {
        try out.appendSlice(gpa, "No matches.\n");
        return;
    }

    for (hits, 1..) |h, i| {
        const o = h.object;
        try out.print(gpa, "{d}. {s}/{s}\n", .{ i, jsonStr(o, "collection"), jsonStr(o, "path") });
        const heading = jsonStr(o, "heading_path");
        if (heading.len != 0) try out.print(gpa, "   {s}\n", .{heading});
        try out.print(gpa, "   {s}\n\n", .{jsonStr(o, "text")});
    }
}

fn renderPackText(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(u8),
    query: []const u8,
    result: std.json.Value,
) !void {
    const obj = result.object;
    try out.print(gpa, "# Context for: {s}\n", .{query});

    const docs = if (obj.get("documents")) |d| (if (d == .array) d.array.items else &.{}) else &.{};
    if (docs.len == 0) {
        try out.appendSlice(gpa, "\nNo relevant documents found.\n");
        return;
    }
    for (docs) |d| {
        const o = d.object;
        try out.print(gpa, "\n## {s}/{s}\n", .{ jsonStr(o, "collection"), jsonStr(o, "path") });
        const spans = if (o.get("spans")) |sp| (if (sp == .array) sp.array.items else &.{}) else &.{};
        for (spans) |sv| {
            const so = sv.object;
            const heading = jsonStr(so, "heading_path");
            if (heading.len != 0) try out.print(gpa, "> {s}\n", .{heading});
            try out.print(gpa, "\n{s}\n", .{jsonStr(so, "text")});
        }
    }
    try out.appendSlice(gpa, "\n---\n");
    if (obj.get("omitted")) |om| if (om == .array and om.array.items.len != 0) {
        try out.appendSlice(gpa, "omitted (over budget):");
        for (om.array.items) |ov| try out.print(gpa, " {s}", .{jsonStr(ov.object, "path")});
        try out.appendSlice(gpa, "\n");
    };
    try out.print(gpa, "tokens: {d} / {d} (approx)\n", .{
        jsonInt(obj, "total_tokens"), jsonInt(obj, "budget_tokens"),
    });
}

// ---------------------------------------------------------------------------
// JSON-RPC framing
// ---------------------------------------------------------------------------

/// Emit a multi-line source literal as a single line.
///
/// NDJSON framing means one message per line, but a Zig multiline string literal
/// keeps the newlines between its segments — writing one directly produced a
/// response split across two lines, which the client reports as a parse error
/// rather than anything diagnosable. Dropping newlines is safe because JSON
/// ignores whitespace between tokens and a raw newline cannot appear inside a
/// JSON string in the first place.
fn writeCompact(w: *Writer, text: []const u8) !void {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |part| try w.writeAll(part);
}

fn beginResult(w: *Writer, id: std.json.Value) !void {
    try w.writeAll("{\"jsonrpc\":\"2.0\",\"id\":");
    try std.json.Stringify.value(id, .{}, w);
    try w.writeAll(",\"result\":");
}

fn endResult(w: *Writer) !void {
    try w.writeAll("}\n");
}

fn writeError(w: *Writer, id: std.json.Value, code: i32, message: []const u8) !void {
    try w.writeAll("{\"jsonrpc\":\"2.0\",\"id\":");
    try std.json.Stringify.value(id, .{}, w);
    try w.print(",\"error\":{{\"code\":{d},\"message\":", .{code});
    try std.json.Stringify.value(message, .{}, w);
    try w.writeAll("}}\n");
}

/// A failed tool call is a *successful* JSON-RPC response with isError set: that
/// is what lets the model read the reason and try something else, instead of the
/// call simply aborting.
fn toolError(w: *Writer, id: std.json.Value, message: []const u8) !void {
    try beginResult(w, id);
    try w.writeAll("{\"isError\":true,\"content\":[{\"type\":\"text\",\"text\":");
    try std.json.Stringify.value(message, .{}, w);
    try w.writeAll("}]}");
    try endResult(w);
}

fn toolText(w: *Writer, id: std.json.Value, text: []const u8) !void {
    try beginResult(w, id);
    try w.writeAll("{\"content\":[{\"type\":\"text\",\"text\":");
    try std.json.Stringify.value(text, .{}, w);
    try w.writeAll("}]}");
    try endResult(w);
}

fn strArg(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn boolArg(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool;
}

fn intArg(o: std.json.ObjectMap, key: []const u8) ?i64 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => null,
    };
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

// ---------------------------------------------------------------------- tests

/// The tool and resource descriptors are hand-written JSON inside Zig multiline
/// string literals, which do no escape processing: a `\"` there is already the
/// two bytes JSON wants, and writing `\\"` — the form a normal quoted literal
/// would need — emits an escaped backslash followed by a quote that closes the
/// string early. That shipped in 0.0.1 and made `tools/list` unparseable, which
/// takes the whole MCP surface down, since every client calls it during startup.
///
/// Nothing in the type system prevents it, so the guard is to parse what we
/// actually emit.
fn renderToJson(gpa: std.mem.Allocator, comptime handler: anytype) !std.json.Parsed(std.json.Value) {
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try handler(&out.writer, .{ .integer = 1 });
    return std.json.parseFromSlice(std.json.Value, gpa, out.written(), .{});
}

test "tools/list emits parseable json" {
    const gpa = std.testing.allocator;
    var parsed = try renderToJson(gpa, handleToolsList);
    defer parsed.deinit();

    const tools = parsed.value.object.get("result").?.object.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 4), tools.items.len);

    // Every tool needs the three fields a client reads, and the descriptions are
    // where the quoting goes wrong, so assert one round-trips with its quotes.
    for (tools.items) |t| {
        try std.testing.expect(t.object.get("name").?.string.len > 0);
        try std.testing.expect(t.object.get("description").?.string.len > 0);
        _ = t.object.get("inputSchema").?.object;
    }
    const records = tools.items[3].object.get("inputSchema").?.object
        .get("properties").?.object.get("where").?.object.get("description").?.string;
    try std.testing.expect(std.mem.indexOf(u8, records, "\"amount > 1000 AND category = 'food'\"") != null);
}

test "resources/list emits parseable json" {
    const gpa = std.testing.allocator;
    var parsed = try renderToJson(gpa, handleResourcesList);
    defer parsed.deinit();

    const resources = parsed.value.object.get("result").?.object.get("resources").?.array;
    try std.testing.expectEqual(@as(usize, 2), resources.items.len);
    for (resources.items) |r| {
        try std.testing.expect(std.mem.startsWith(u8, r.object.get("uri").?.string, "zkb://"));
    }
}
