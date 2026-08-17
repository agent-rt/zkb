//! Drain the pending-doc queue: parse, chunk, embed, write.
//!
//! Ordering that matters: **all vectors for a document are computed before the
//! write transaction opens.** Embedding is tens of milliseconds per chunk; holding
//! a write lock across it would block every reader for the length of a document
//! (SPEC §4.3).
//!
//! Failure granularity is one document. A half-indexed document is worse than an
//! unindexed one — the user would believe it was searched — so on any error its
//! chunks are dropped and `index_error` is set.

const std = @import("std");
const store = @import("../db/store.zig");
const schema = @import("../db/schema.zig");
const markdown = @import("markdown.zig");
const chunk = @import("chunk.zig");
const embed = @import("../embed/llama.zig");

pub const Stats = struct {
    docs_indexed: usize = 0,
    docs_failed: usize = 0,
    chunks_written: usize = 0,
    embed_calls: usize = 0,
    embed_ns: u64 = 0,
    total_ns: u64 = 0,

    pub fn avgEmbedMs(self: Stats) f64 {
        if (self.embed_calls == 0) return 0;
        return @as(f64, @floatFromInt(self.embed_ns)) /
            @as(f64, @floatFromInt(self.embed_calls)) / std.time.ns_per_ms;
    }
};

pub const Options = struct {
    chunking: chunk.Config = .{},
    /// Stop after this many documents; 0 means no limit.
    limit: usize = 0,
    progress: ?*const fn (done: usize, total: usize, rel_path: []const u8) void = null,
};

/// Wraps the embedder's tokenizer for the chunker. Generic character heuristics
/// are off by up to 30% on CJK and an over-long chunk is silently truncated by
/// the model, so the count has to come from the model itself.
const EmbedCounter = struct {
    embedder: *embed.Embedder,

    fn counter(self: *EmbedCounter) chunk.TokenCounter {
        return .{ .ctx = self, .countFn = countImpl };
    }

    fn countImpl(ctx: *anyopaque, text: []const u8) anyerror!usize {
        const self: *EmbedCounter = @ptrCast(@alignCast(ctx));
        return self.embedder.countTokens(text);
    }
};

pub fn indexPending(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
    embedder: *embed.Embedder,
    collection_id: i64,
    root: []const u8,
    now_ms: i64,
    opts: Options,
) !Stats {
    var stats: Stats = .{};
    const started = nowNs(io);

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const pending = try s.listPending(
        arena_state.allocator(),
        collection_id,
        // Sentinel must survive the trip through SQLite's i64 bind.
        if (opts.limit == 0) std.math.maxInt(i64) else opts.limit,
    );

    var ec: EmbedCounter = .{ .embedder = embedder };
    const dim: usize = @intCast(schema.embedding_dim);

    for (pending.items, 0..) |doc, n| {
        if (opts.progress) |cb| cb(n + 1, pending.items.len, doc.rel_path);

        indexOne(gpa, io, s, embedder, &ec, dim, collection_id, root, doc, now_ms, opts, &stats) catch |err| {
            // Drop any partial work, then record why. Never leave a document
            // looking indexed when it is not.
            s.rollback();
            s.deleteChunks(doc.id) catch {};
            var buf: [128]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "{t}", .{err}) catch "unknown error";
            s.markFailed(doc.id, msg) catch {};
            stats.docs_failed += 1;
            continue;
        };
        stats.docs_indexed += 1;
    }

    stats.total_ns = @intCast(nowNs(io) - started);
    return stats;
}

fn indexOne(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
    embedder: *embed.Embedder,
    ec: *EmbedCounter,
    dim: usize,
    collection_id: i64,
    root: []const u8,
    doc: store.Store.PendingDoc,
    now_ms: i64,
    opts: Options,
    stats: *Stats,
) !void {
    const abs = try std.fs.path.join(gpa, &.{ root, doc.rel_path });
    defer gpa.free(abs);

    const source = try readFile(gpa, io, abs);
    defer gpa.free(source);

    var parsed = try markdown.scan(gpa, source);
    defer parsed.deinit(gpa);

    var chunks = try chunk.split(gpa, source, &parsed, ec.counter(), opts.chunking);
    defer chunks.deinit(gpa);

    // Vectors first, outside any transaction.
    const vectors = try gpa.alloc(f32, chunks.items.len * dim);
    defer gpa.free(vectors);
    for (chunks.items, 0..) |c, i| {
        const slot = vectors[i * dim ..][0..dim];
        const t0 = nowNs(io);
        _ = try embedder.embedDocument(c.heading_path, c.text, slot);
        stats.embed_ns += @intCast(nowNs(io) - t0);
        stats.embed_calls += 1;
    }

    // Then a short write transaction.
    try s.begin();
    errdefer s.rollback();

    try s.deleteChunks(doc.id);
    try s.setDocMeta(doc.id, parsed.title, parsed.frontmatter);
    for (chunks.items, 0..) |c, i| {
        _ = try s.insertChunk(collection_id, doc.id, .{
            .idx = @intCast(c.idx),
            .heading_path = c.heading_path,
            .byte_start = @intCast(c.byte_start),
            .byte_end = @intCast(c.byte_end),
            .n_tokens = @intCast(c.n_tokens),
            .text = c.text,
        }, vectors[i * dim ..][0..dim]);
    }
    try s.markIndexed(doc.id, @intCast(chunks.items.len), now_ms);
    try s.commit();

    stats.chunks_written += chunks.items.len;
}

/// Monotonic nanoseconds. Zig 0.16 has no ambient Timer: clocks come from Io,
/// and `.awake` is the monotonic one (excludes time the machine was suspended).
fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

fn readFile(gpa: std.mem.Allocator, io: std.Io, abs: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFileAbsolute(io, abs, .{});
    defer file.close(io);
    const st = try file.stat(io);
    const buf = try gpa.alloc(u8, @intCast(st.size));
    errdefer gpa.free(buf);
    var read_buf: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    try reader.interface.readSliceAll(buf);
    return buf;
}
