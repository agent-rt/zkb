//! Retrieval trace: one JSON line per query, recording what each path ranked.
//!
//! Every threshold and weight in this system was supposed to be earned by
//! measurement, and each time (E2, E2b, E7) the measurement needed a corpus of
//! real queries with their per-path rankings. Reconstructing that after the fact
//! means replaying queries against an index that has since changed. Writing it
//! down as it happens costs microseconds and makes the next calibration possible
//! from real usage rather than from a hand-written query set.
//!
//! **Off unless `ZKB_TRACE=1`.** A query log is a record of what someone was
//! thinking about, which is not something to collect by default even locally.
//! When on, it is a plain file the user can read and delete.

const std = @import("std");
const hybrid = @import("hybrid.zig");

/// Stop appending past this size. A trace that fills a disk is a worse problem
/// than a trace that stops — and by the time it is this large it has plenty to
/// calibrate from.
pub const max_bytes: u64 = 64 * 1024 * 1024;

pub const Writer = struct {
    io: std.Io,
    path: []const u8,
    enabled: bool,

    /// `env` decides: absent or not "1" means every call below is a no-op.
    pub fn init(io: std.Io, env: *const std.process.Environ.Map, path: []const u8) Writer {
        const on = if (env.get("ZKB_TRACE")) |v| std.mem.eql(u8, v, "1") else false;
        return .{ .io = io, .path = path, .enabled = on };
    }

    /// Record one query. Never fails the query it is tracing: a trace that can
    /// break a search is worse than no trace.
    pub fn record(
        self: *const Writer,
        gpa: std.mem.Allocator,
        query: []const u8,
        results: *const hybrid.Results,
        elapsed_ms: i64,
    ) void {
        if (!self.enabled) return;
        self.tryRecord(gpa, query, results, elapsed_ms) catch {};
    }

    fn tryRecord(
        self: *const Writer,
        gpa: std.mem.Allocator,
        query: []const u8,
        results: *const hybrid.Results,
        elapsed_ms: i64,
    ) !void {
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(gpa);

        try line.appendSlice(gpa, "{\"q\":");
        try appendJson(gpa, &line, query);
        try line.print(gpa, ",\"mode\":\"{t}\",\"ms\":{d},\"vec_cands\":{d},\"fts_cands\":{d}," ++
            "\"fts_skipped\":{},\"hits\":[", .{
            results.mode,
            elapsed_ms,
            results.vec_candidates,
            results.fts_candidates,
            results.fts_skipped,
        });

        for (results.hits, 0..) |h, i| {
            if (i != 0) try line.append(gpa, ',');
            try line.appendSlice(gpa, "{\"p\":");
            try appendJson(gpa, &line, h.rel_path);
            // The two per-path ranks are the whole point: a fused score alone
            // cannot tell you which path found the document, and that is exactly
            // the question every tuning decision turns on.
            try line.print(gpa, ",\"idx\":{d},\"score\":{d:.5},\"vec\":", .{ h.idx, h.score });
            if (h.vec_rank) |r| try line.print(gpa, "{d}", .{r}) else try line.appendSlice(gpa, "null");
            try line.appendSlice(gpa, ",\"fts\":");
            if (h.fts_rank) |r| try line.print(gpa, "{d}", .{r}) else try line.appendSlice(gpa, "null");
            try line.append(gpa, '}');
        }

        try line.appendSlice(gpa, "],\"dropped\":[");
        for (results.dropped_terms, 0..) |d, i| {
            if (i != 0) try line.append(gpa, ',');
            try appendJson(gpa, &line, d);
        }
        try line.appendSlice(gpa, "]}\n");

        var file = try std.Io.Dir.createFileAbsolute(self.io, self.path, .{ .truncate = false });
        defer file.close(self.io);

        const end = (try file.stat(self.io)).size;
        if (end >= max_bytes) return;

        var buf: [8192]u8 = undefined;
        var fw = file.writer(self.io, &buf);
        fw.pos = end;
        try fw.interface.writeAll(line.items);
        try fw.interface.flush();
    }
};

fn appendJson(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    std.json.Stringify.value(s, .{}, &w) catch {
        // Longer than the buffer: record that it existed rather than dropping
        // the whole line.
        try out.appendSlice(gpa, "\"(too long)\"");
        return;
    };
    try out.appendSlice(gpa, w.buffered());
}
