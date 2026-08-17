//! Newline-delimited JSON over a unix socket.
//!
//! One line per request, one per response, request/response paired by `id`.
//! No HTTP, no msgpack: the request rate is single digits per second, so parsing
//! cost is irrelevant, and `std.json` is in the standard library (SPEC §6.4).
//!
//! Errors are part of the protocol rather than transport failures, because the
//! interesting ones are all things the user can act on — a stale schema, a
//! missing model, an index still building.

const std = @import("std");

pub const protocol_version: u32 = 1;

pub const Method = enum {
    health,
    stats,
    search,
    query,
    index,
    maintain,
    shutdown,

    pub fn parse(s: []const u8) ?Method {
        return std.meta.stringToEnum(Method, s);
    }
};

pub const ErrorCode = enum {
    bad_request,
    not_found,
    model_mismatch,
    model_unavailable,
    indexing,
    internal,
    version_mismatch,

    pub fn text(self: ErrorCode) []const u8 {
        return @tagName(self);
    }
};

/// A parsed request. `params` stays as a JSON value so each handler pulls out
/// what it needs without a per-method struct.
pub const Request = struct {
    id: i64,
    method: Method,
    params: std.json.Value,
    /// Owns the parsed document; freed by `deinit`.
    parsed: std.json.Parsed(std.json.Value),

    pub fn deinit(self: *Request) void {
        self.parsed.deinit();
        self.* = undefined;
    }

    pub fn str(self: *const Request, name: []const u8) ?[]const u8 {
        const v = self.params.object.get(name) orelse return null;
        return switch (v) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn int(self: *const Request, name: []const u8) ?i64 {
        const v = self.params.object.get(name) orelse return null;
        return switch (v) {
            .integer => |i| i,
            .float => |f| @intFromFloat(f),
            else => null,
        };
    }

    pub fn boolean(self: *const Request, name: []const u8) ?bool {
        const v = self.params.object.get(name) orelse return null;
        return switch (v) {
            .bool => |b| b,
            else => null,
        };
    }
};

pub const ParseError = error{
    InvalidJson,
    MissingId,
    MissingMethod,
    UnknownMethod,
    OutOfMemory,
};

/// Parse one NDJSON line. The returned Request owns its backing memory.
pub fn parseRequest(gpa: std.mem.Allocator, line: []const u8) ParseError!Request {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, line, .{}) catch
        return error.InvalidJson;
    errdefer parsed.deinit();

    if (parsed.value != .object) return error.InvalidJson;
    const obj = parsed.value.object;

    const id: i64 = switch (obj.get("id") orelse return error.MissingId) {
        .integer => |i| i,
        else => return error.MissingId,
    };
    const method_name = switch (obj.get("method") orelse return error.MissingMethod) {
        .string => |s| s,
        else => return error.MissingMethod,
    };
    const method = Method.parse(method_name) orelse return error.UnknownMethod;

    // Absent params is the same as empty params; requiring the key would make
    // every no-argument call carry `"params":{}`. An empty ObjectMap needs no
    // allocation, so `.empty` avoids a failure path for the common case.
    const params = obj.get("params") orelse std.json.Value{ .object = .empty };
    if (params != .object) return error.InvalidJson;

    return .{ .id = id, .method = method, .params = params, .parsed = parsed };
}

/// `{"id":N,"ok":true,"result":` — the caller writes the result body and then
/// calls `finishOk`.
pub fn beginOk(w: *std.Io.Writer, id: i64) !void {
    try w.print("{{\"id\":{d},\"ok\":true,\"result\":", .{id});
}

pub fn finishOk(w: *std.Io.Writer) !void {
    try w.writeAll("}\n");
}

pub fn writeError(
    w: *std.Io.Writer,
    id: i64,
    code: ErrorCode,
    message: []const u8,
    hint: ?[]const u8,
) !void {
    try w.print("{{\"id\":{d},\"ok\":false,\"error\":{{\"code\":\"{s}\",\"message\":", .{
        id, code.text(),
    });
    try std.json.Stringify.value(message, .{}, w);
    if (hint) |h| {
        try w.writeAll(",\"hint\":");
        try std.json.Stringify.value(h, .{}, w);
    }
    try w.writeAll("}}\n");
}

/// Maximum accepted request line. A search query is short; anything this long is
/// a bug or an attack, and an unbounded line would let one client exhaust memory.
pub const max_line_bytes: usize = 64 * 1024;
