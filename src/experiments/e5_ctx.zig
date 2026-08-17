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
    var stdout = std.Io.File.stdout().writer(init.io, &out_buf);
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

    for ([_]u32{ 8192, 4096, 2048, 1536 }) |n_ctx| {
        var e = try emb.Embedder.init(gpa, model_path, .{ .n_ctx = n_ctx });
        defer e.deinit();

        const vec = try gpa.alloc(f32, e.n_embd);
        defer gpa.free(vec);

        const tokens = try e.countTokens(body.items);

        // One warm-up pass so buffer allocation is not attributed to the measurement.
        _ = try e.embed(body.items, vec);

        const t0 = std.Io.Timestamp.now(init.io, .awake).nanoseconds;
        for (0..iterations) |_| _ = try e.embed(body.items, vec);
        const elapsed: u64 = @intCast(std.Io.Timestamp.now(init.io, .awake).nanoseconds - t0);

        const per_call_ms = @as(f64, @floatFromInt(elapsed)) /
            @as(f64, @floatFromInt(iterations)) / std.time.ns_per_ms;
        try w.print("  n_ctx {d:>5}: {d:>7.1} ms/chunk   ({d} tokens per chunk)\n", .{
            n_ctx, per_call_ms, tokens,
        });
        try w.flush();
    }

    try w.writeAll(
        \\
        \\Chunks are capped at 1024 tokens (SPEC 4.2), so anything above ~1536
        \\buys nothing. If the times differ materially, n_ctx is the knob.
        \\
    );
    try w.flush();
}
