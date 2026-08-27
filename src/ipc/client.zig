//! Thin client: connect, send one request, read one response, exit.
//!
//! Deliberately does **not** start the daemon when the socket is missing.
//! Implicitly spawning a background process because a command was typed is a
//! surprising side effect; printing what to run is not (SPEC §7).

const std = @import("std");
const proto = @import("proto.zig");
const paths = @import("../util/paths.zig");

pub const Error = error{
    DaemonNotRunning,
    ConnectFailed,
    ResponseTooLong,
    ConnectionClosed,
    /// The daemon is running a different binary than this process.
    DaemonStale,
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
    /// Whether the daemon has been shown to be this build. Null until asked.
    build_checked: ?bool = null,

    /// Refuse to be answered by a daemon running a different binary.
    ///
    /// The check has to live on this side. A stale daemon executes only the code
    /// it was started with, so it cannot notice that it is stale, and no field
    /// added to the request will ever be read by it. What the caller can do is
    /// ask what the daemon is, and treat anything but its own build as a daemon
    /// that is not there.
    ///
    /// Silence counts as a mismatch. A daemon predating the `build` field answers
    /// `health` without it, and that absence is the only evidence such a daemon
    /// can give — which is exactly the case that has to be caught, since it is the
    /// state every upgrade leaves behind.
    ///
    /// One round trip per process, on the first request that needs it. `health`
    /// touches no model and no index.
    /// The error set is written out rather than inferred: this calls `call`, and
    /// `call` calls this, so leaving both to inference is a dependency loop.
    fn requireCurrentBuild(self: *Client, gpa: std.mem.Allocator) error{DaemonStale}!void {
        if (self.build_checked) |ok| return if (ok) {} else error.DaemonStale;

        // Marked before the call so the `health` request below cannot recurse:
        // `health` does not need a current build, but the flag also keeps a
        // failure from being retried once per call.
        self.build_checked = false;

        var resp = self.call(gpa, .health, "{}") catch return error.DaemonStale;
        defer resp.deinit(gpa);

        const mine = paths.selfBuildId(gpa, self.io) catch return error.DaemonStale;
        const theirs: ?u64 = blk: {
            const r = resp.result orelse break :blk null;
            if (r != .object) break :blk null;
            const v = r.object.get("build") orelse break :blk null;
            break :blk switch (v) {
                .integer => |i| @bitCast(i),
                else => null,
            };
        };

        if (theirs == null or theirs.? != mine) {
            // stderr, not the command's writer: the fallback that follows owns
            // stdout, and `--json` callers parse it. A note printed there would
            // corrupt the very output it was trying to explain.
            std.debug.print(
                "note: the running zkb daemon is a different build; using the slower in-process path\n" ++
                    "      to fix: zkb daemon stop && zkb daemon start\n",
                .{},
            );
            return error.DaemonStale;
        }
        self.build_checked = true;
    }

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
        if (method.needsCurrentBuild()) try self.requireCurrentBuild(gpa);

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
