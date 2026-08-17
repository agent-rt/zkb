//! Priority queue behaviour.
//!
//! The property that matters: a background index of ~1500 chunks must not starve
//! interactive queries. A plain mutex would serialize in arrival order and make
//! the daemon unusable for minutes at a time, so strict priority is not a tuning
//! knob — it is load-bearing (SPEC §3.4).

const std = @import("std");
const zkb = @import("zkb");
const queue = zkb.embed_queue;
const proto = zkb.proto;

const testing = std.testing;

/// The queue's synchronisation takes an explicit `Io`; tests get a real threaded
/// one so `spawn`/`wait` behave as they do in the daemon.
fn withIo(comptime f: fn (std.Io) anyerror!void) !void {
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    try f(threaded.io());
}

fn mkJob(text: []const u8, out: []f32) queue.Job {
    return .{ .kind = .query, .text = text, .out = out };
}

test "interactive jobs are served before ingest jobs already queued" {
    try withIo(struct {
        fn run(io: std.Io) anyerror!void {
            var q: queue.Queue = .{};
            var out: [1]f32 = .{0};

            var ingest_jobs: [5]queue.Job = undefined;
            for (&ingest_jobs) |*j| {
                j.* = mkJob("ingest", &out);
                try q.push(io, j, .ingest);
            }
            var interactive = mkJob("interactive", &out);
            try q.push(io, &interactive, .interactive);

            // Despite arriving last, the interactive job comes out first.
            const first = q.pop(io).?;
            try testing.expectEqual(&interactive, first);
            try testing.expectEqualStrings("interactive", first.text);

            // The ingest backlog is intact and drains in order afterwards.
            for (0..5) |_| {
                const j = q.pop(io).?;
                try testing.expectEqualStrings("ingest", j.text);
            }
        }
    }.run);
}

test "preemption depth is recorded, so a test cannot pass without exercising it" {
    try withIo(struct {
        fn run(io: std.Io) anyerror!void {
            var q: queue.Queue = .{};
            var out: [1]f32 = .{0};

            var ingest_jobs: [7]queue.Job = undefined;
            for (&ingest_jobs) |*j| {
                j.* = mkJob("ingest", &out);
                try q.push(io, j, .ingest);
            }
            var interactive = mkJob("q", &out);
            try q.push(io, &interactive, .interactive);

            _ = q.pop(io);
            // Seven ingest jobs were skipped over. If this were 0 the queue would have
            // had nothing to preempt and the previous test would prove nothing.
            try testing.expectEqual(@as(usize, 7), q.max_preempted);
            try testing.expectEqual(@as(usize, 1), q.served_interactive);
        }
    }.run);
}

test "a closed and drained queue returns null, which is the worker's exit signal" {
    try withIo(struct {
        fn run(io: std.Io) anyerror!void {
            var q: queue.Queue = .{};
            var out: [1]f32 = .{0};
            var job = mkJob("x", &out);
            try q.push(io, &job, .ingest);

            q.close(io);
            // Queued work still drains: closing must not abandon jobs half-done.
            try testing.expectEqual(&job, q.pop(io).?);
            try testing.expectEqual(@as(?*queue.Job, null), q.pop(io));
            try testing.expectError(error.QueueClosed, q.push(io, &job, .ingest));
        }
    }.run);
}

test "drainWithError releases every waiter instead of hanging them" {
    try withIo(struct {
        fn run(io: std.Io) anyerror!void {
            // If the embedder cannot load, waiters must fail rather than block forever.
            var q: queue.Queue = .{};
            var out: [1]f32 = .{0};
            var a = mkJob("a", &out);
            var b = mkJob("b", &out);
            try q.push(io, &a, .interactive);
            try q.push(io, &b, .ingest);

            q.drainWithError(io, error.ModelUnavailable);

            try testing.expectError(error.ModelUnavailable, a.wait(io));
            try testing.expectError(error.ModelUnavailable, b.wait(io));
            const d = q.depth(io);
            try testing.expectEqual(@as(usize, 0), d.interactive);
            try testing.expectEqual(@as(usize, 0), d.ingest);
        }
    }.run);
}

test "a real worker thread serves the interactive job first under load" {
    try withIo(struct {
        fn run(io: std.Io) anyerror!void {
            var q: queue.Queue = .{};
            var out: [1]f32 = .{0};

            // Order in which jobs were completed, as observed by the worker.
            const Ctx = struct {
                q: *queue.Queue,
                io: std.Io,
                order: std.ArrayList([]const u8) = .empty,
                gpa: std.mem.Allocator,

                fn work(self: *@This()) void {
                    while (self.q.pop(self.io)) |job| {
                        self.order.append(self.gpa, job.text) catch {};
                        job.done.post(self.io);
                    }
                }
            };

            var ingest_jobs: [20]queue.Job = undefined;
            for (&ingest_jobs) |*j| {
                j.* = mkJob("ingest", &out);
                try q.push(io, j, .ingest);
            }
            var interactive = mkJob("interactive", &out);
            try q.push(io, &interactive, .interactive);
            q.close(io);

            var ctx: Ctx = .{ .q = &q, .io = io, .gpa = testing.allocator };
            defer ctx.order.deinit(testing.allocator);
            const t = try std.Thread.spawn(.{}, Ctx.work, .{&ctx});
            t.join();

            try testing.expectEqual(@as(usize, 21), ctx.order.items.len);
            try testing.expectEqualStrings("interactive", ctx.order.items[0]);
        }
    }.run);
}

// ---------------------------------------------------------------------------
// IPC protocol
// ---------------------------------------------------------------------------

test "a well-formed request round-trips" {
    var req = try proto.parseRequest(testing.allocator,
        \\{"id":7,"method":"search","params":{"query":"融合","k":5,"json":true}}
    );
    defer req.deinit();

    try testing.expectEqual(@as(i64, 7), req.id);
    try testing.expectEqual(proto.Method.search, req.method);
    try testing.expectEqualStrings("融合", req.str("query").?);
    try testing.expectEqual(@as(?i64, 5), req.int("k"));
    try testing.expectEqual(@as(?bool, true), req.boolean("json"));
    try testing.expectEqual(@as(?[]const u8, null), req.str("missing"));
}

test "params may be omitted entirely" {
    var req = try proto.parseRequest(testing.allocator, "{\"id\":1,\"method\":\"health\"}");
    defer req.deinit();
    try testing.expectEqual(proto.Method.health, req.method);
    try testing.expectEqual(@as(?i64, null), req.int("anything"));
}

test "malformed requests are rejected with a specific reason" {
    const cases = [_]struct { line: []const u8, want: anyerror }{
        .{ .line = "not json", .want = error.InvalidJson },
        .{ .line = "[]", .want = error.InvalidJson },
        .{ .line = "{\"method\":\"health\"}", .want = error.MissingId },
        .{ .line = "{\"id\":1}", .want = error.MissingMethod },
        .{ .line = "{\"id\":1,\"method\":\"nope\"}", .want = error.UnknownMethod },
        .{ .line = "{\"id\":\"x\",\"method\":\"health\"}", .want = error.MissingId },
        .{ .line = "{\"id\":1,\"method\":\"health\",\"params\":3}", .want = error.InvalidJson },
    };
    for (cases) |c| {
        try testing.expectError(c.want, proto.parseRequest(testing.allocator, c.line));
    }
}

test "error responses carry an actionable hint" {
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try proto.writeError(&w, 3, .model_mismatch, "embedding model changed", "run: zkb reindex");
    const out = w.buffered();

    try testing.expect(std.mem.indexOf(u8, out, "\"ok\":false") != null);
    try testing.expect(std.mem.indexOf(u8, out, "model_mismatch") != null);
    try testing.expect(std.mem.indexOf(u8, out, "run: zkb reindex") != null);
    // NDJSON framing: exactly one trailing newline, none inside.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "\n"));
    try testing.expect(std.mem.endsWith(u8, out, "\n"));
}

test "messages needing JSON escaping do not break framing" {
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try proto.writeError(&w, 1, .internal, "line1\nline2 \"quoted\"", null);
    const out = w.buffered();
    // The embedded newline must be escaped, not emitted raw.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "\n"));
    try testing.expect(std.mem.indexOf(u8, out, "\\n") != null);
}
