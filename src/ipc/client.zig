//! Thin client: connect, send one request, read one response, exit.
//!
//! Deliberately does **not** start the daemon when the socket is missing.
//! Implicitly spawning a background process because a command was typed is a
//! surprising side effect; printing what to run is not (SPEC §7).

const std = @import("std");
const proto = @import("proto.zig");

pub const Error = error{
    DaemonNotRunning,
    ConnectFailed,
    ResponseTooLong,
    ConnectionClosed,
};

pub const Response = struct {
    /// Raw response line, without the trailing newline. Owned.
    line: []u8,
    ok: bool,
    /// Present when `ok` is false.
    code: []const u8 = "",
    message: []const u8 = "",
    hint: ?[]const u8 = null,
    /// The `result` value when ok; borrowed from `parsed`.
    result: ?std.json.Value = null,
    parsed: ?std.json.Parsed(std.json.Value) = null,

    pub fn deinit(self: *Response, gpa: std.mem.Allocator) void {
        if (self.parsed) |p| p.deinit();
        gpa.free(self.line);
        self.* = undefined;
    }
};

pub const Client = struct {
    stream: std.Io.net.Stream,
    io: std.Io,
    next_id: i64 = 1,

    pub fn connect(io: std.Io, sock_path: []const u8) Error!Client {
        const addr = std.Io.net.UnixAddress.init(sock_path) catch return error.ConnectFailed;
        const stream = addr.connect(io) catch |err| switch (err) {
            // A missing socket file and a socket nobody is listening on are the
            // same thing to a user: the daemon is not running. (ConnectionRefused
            // is not in this error set — a unix socket with no listener surfaces
            // as FileNotFound or is reported through the generic path.)
            error.FileNotFound => return error.DaemonNotRunning,
            else => return error.ConnectFailed,
        };
        return .{ .stream = stream, .io = io };
    }

    pub fn close(self: *Client) void {
        self.stream.close(self.io);
    }

    /// Send `{"id":N,"method":M,"params":<params_json>}` and read the reply.
    pub fn call(
        self: *Client,
        gpa: std.mem.Allocator,
        method: proto.Method,
        params_json: []const u8,
    ) !Response {
        const id = self.next_id;
        self.next_id += 1;

        {
            var wbuf: [proto.max_line_bytes]u8 = undefined;
            var writer = self.stream.writer(self.io, &wbuf);
            const w = &writer.interface;
            w.print("{{\"id\":{d},\"method\":\"{s}\",\"params\":{s}}}\n", .{
                id, @tagName(method), params_json,
            }) catch return error.ConnectionClosed;
            // A daemon at its connection limit accepts and immediately closes, so
            // the first write is where that shows up. It is an expected condition,
            // not a crash.
            w.flush() catch return error.ConnectionClosed;
        }

        // Reader and writer are per-call rather than held on the struct: the
        // struct is returned by value, and a Reader keeps a pointer into its own
        // buffer, so a copy would leave that pointer aimed at the temporary.
        // Safe here because the protocol is strictly one line out, one line back.
        var rbuf: [proto.max_line_bytes]u8 = undefined;
        var reader = self.stream.reader(self.io, &rbuf);
        // Inclusive so the delimiter is consumed; see the note in daemon.zig.
        const raw = reader.interface.takeDelimiterInclusive('\n') catch |err| switch (err) {
            error.EndOfStream => return error.ConnectionClosed,
            error.StreamTooLong => return error.ResponseTooLong,
            else => return err,
        };
        const line = std.mem.trimEnd(u8, raw, "\r\n");

        var resp: Response = .{ .line = try gpa.dupe(u8, line), .ok = false };
        errdefer gpa.free(resp.line);

        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, resp.line, .{});
        resp.parsed = parsed;
        const obj = parsed.value.object;
        resp.ok = switch (obj.get("ok") orelse .null) {
            .bool => |b| b,
            else => false,
        };
        if (resp.ok) {
            resp.result = obj.get("result");
        } else if (obj.get("error")) |e| {
            if (e == .object) {
                if (e.object.get("code")) |c| if (c == .string) {
                    resp.code = c.string;
                };
                if (e.object.get("message")) |m| if (m == .string) {
                    resp.message = m.string;
                };
                if (e.object.get("hint")) |h| if (h == .string) {
                    resp.hint = h.string;
                };
            }
        }
        return resp;
    }
};

/// Print the daemon-not-running message once, in one place, so every command
/// says the same thing.
pub fn reportNotRunning(w: *std.Io.Writer, sock_path: []const u8) !void {
    try w.print("daemon is not running (no listener at {s})\nrun: zkb daemon start\n", .{sock_path});
}
