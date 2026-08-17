//! E5b — does n_ctx dominate embedding throughput?
//!
//! The first full index of ~/docs ran at roughly 410 ms per chunk, an order of
//! magnitude slower than the SPEC's estimate. Hypothesis: the context is sized
//! 8192 while chunks are capped at 1024 tokens, so every decode pays for compute
//! buffers eight times larger than the work being done.
//!
//! One variable: n_ctx. Same model, same texts, same token counts.
//!
//! Run: zig build e5-ctx -- <path-to-gguf>

const std = @import("std");
const zkb = @import("zkb");
const emb = zkb.embed;

const iterations = 20;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var args = init.minimal.args.iterate();
    _ = args.next();
    const model_path = args.next() orelse {
        std.debug.print("usage: e5_ctx <path-to-gguf>\n", .{});
        return error.MissingArgument;
    };

    var out_buf: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    const w = &stdout.interface;

    // A chunk-sized body: representative of what the indexer actually embeds.
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(gpa);
    while (body.items.len < 3200) {
        try body.appendSlice(gpa,
            \\检索融合使用 RRF，而不是加权和：加权系数没有实验依据，
            \\而 rank-based 融合无量纲，新增检索路时不需要重新标定。
            \\Hybrid retrieval combines vector KNN with BM25 over an FTS5 index.
            \\
        );
    }

    try w.print("E5b — embedding throughput vs n_ctx ({d} iterations each)\n", .{iterations});
    try w.print("model: {s}\n\n", .{model_path});
    try w.flush();

    // A short query, to separate fixed per-call overhead from token-count cost.
    const short = "Instruct: retrieve relevant passages\nQuery: 混合检索怎么设计";

    try w.print("{s:<12}{s:>14}{s:>14}\n", .{ "n_ctx", "chunk ms", "query ms" });
    for ([_]u32{ 8192, 4096, 2048, 1024, 512, 256 }) |n_ctx| {
        var e = try emb.Embedder.init(gpa, model_path, .{ .n_ctx = n_ctx });
        defer e.deinit();

        const vec = try gpa.alloc(f32, e.n_embd);
        defer gpa.free(vec);

        // A chunk-sized body only fits when the context is large enough.
        var chunk_ms: f64 = 0;
        if ((try e.countTokens(body.items)) < n_ctx) {
            _ = try e.embed(body.items, vec); // warm-up, so buffer setup is excluded
            const t0 = std.Io.Timestamp.now(init.io, .awake).nanoseconds;
            for (0..iterations) |_| _ = try e.embed(body.items, vec);
            const el: u64 = @intCast(std.Io.Timestamp.now(init.io, .awake).nanoseconds - t0);
            chunk_ms = @as(f64, @floatFromInt(el)) / @as(f64, @floatFromInt(iterations)) / std.time.ns_per_ms;
        }

        _ = try e.embed(short, vec);
        const t1 = std.Io.Timestamp.now(init.io, .awake).nanoseconds;
        for (0..iterations) |_| _ = try e.embed(short, vec);
        const el2: u64 = @intCast(std.Io.Timestamp.now(init.io, .awake).nanoseconds - t1);
        const query_ms = @as(f64, @floatFromInt(el2)) / @as(f64, @floatFromInt(iterations)) / std.time.ns_per_ms;

        try w.print("{d:<12}{d:>14.1}{d:>14.1}\n", .{ n_ctx, chunk_ms, query_ms });
        try w.flush();
    }

    try w.writeAll(
        \\
        \\If query time tracks n_ctx rather than token count, the cost is fixed
        \\per-call buffer work and a small dedicated query context is the lever.
        \\
    );
    try w.flush();
}
