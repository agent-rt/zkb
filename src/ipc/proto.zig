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
    /// Memories plus the current-value snapshot of every fact. Read-only, so it
    /// belongs on the connection thread like search and query; `remember` does
    /// not, because writing would break the single-writer invariant that keeps
    /// SQLITE_BUSY off the table (see `handleIndex`).
    recall,
    index,
    /// Remove a collection and everything indexed under it. A write, so it is
    /// handed to the ingest thread rather than done here — same reason as `index`.
    collection_rm,
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

    /// Process exit code for this error.
    ///
    /// One mapping, because there were four. Every `viaDaemon` picked its own
    /// constant — `collection rm` returned 3, search/query/memory returned 4 — so
    /// the same refusal exited differently depending on whether a daemon happened
    /// to be running, and differently again from the in-process path that produced
    /// the same message. Refusing to delete a built-in collection was 2 standalone
    /// and 3 over IPC.
    ///
    /// The contract these follow, already asserted by the CI smoke test:
    ///
    ///   2  the request was wrong — usage, bad flags, a refusal
    ///   3  nothing to act on — no index, no such thing, daemon unavailable
    ///   4  the embedding model is missing or does not match the index
    ///   1  the index is inconsistent with itself
    pub fn exitCode(self: ErrorCode) u8 {
        return switch (self) {
            .bad_request => 2,
            .not_found, .indexing => 3,
            .model_mismatch, .model_unavailable => 4,
            // A daemon-side failure the caller cannot act on. Not 2: the request
            // was fine.
            .internal, .version_mismatch => 3,
        };
    }

    /// Exit code for a code that arrived as text, which is how a client sees it.
    /// An unrecognised name means a newer daemon, and 3 is the honest answer:
    /// nothing this client can do about it.
    pub fn exitCodeOf(name: []const u8) u8 {
        const c = std.meta.stringToEnum(ErrorCode, name) orelse return 3;
        return c.exitCode();
    }

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
