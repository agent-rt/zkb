//! `zkb mcp` — MCP stdio server, forwarding to the daemon over the unix socket.
//!
//! JSON-RPC 2.0, newline-delimited, on stdin/stdout. **Nothing may be written to
//! stdout except protocol frames** — a stray log line corrupts the stream and the
//! client sees a parse error rather than a diagnostic. Diagnostics go to stderr.
//!
//! Two tools only. Tool definitions are re-sent to the model on every turn, so
//! each one costs context permanently; `search` and `query` are the two things a
//! resource URI cannot express (SPEC §8). Anything readable is a resource
//! instead, which costs no tool slot.
//!
//! No write tools: zkb is read-only over documents. An agent that wants to change
//! a file uses its own editing tools, and the next scan picks the change up.

const std = @import("std");
const zkb = @import("zkb");
const client = zkb.ipc_client;

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
    var stdin = std.Io.File.stdin().reader(io, &in_buf);
    var stdout = std.Io.File.stdout().writer(io, &out_buf);
    const w = &stdout.interface;

    while (true) {
        const raw = stdin.interface.takeDelimiterInclusive('\n') catch |err| switch (err) {
            error.EndOfStream => return 0,
            else => return 0,
        };
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        if (line.len == 0) continue;

        handle(gpa, io, layout.sock, w, line) catch |err| {
            std.debug.print("zkb mcp: {t}\n", .{err});
        };
        w.flush() catch return 0;
    }
}

fn handle(
    gpa: std.mem.Allocator,
    io: std.Io,
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
    if (std.mem.eql(u8, method, "tools/call")) return handleToolsCall(gpa, io, sock, w, id, params);
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
        \\  "required":["query"]}}
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

    const query = switch (args.object.get("query") orelse .null) {
        .string => |s| s,
        else => return toolError(w, id, "query is required"),
    };

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

    return writeError(w, id, -32602, "unknown tool");
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
