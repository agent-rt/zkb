//! Qwen3-Embedding via llama.cpp. One forward pass per call, pooled hidden
//! state out, L2-normalized.
//!
//! The call sequence is one already validated on real hardware — deliberately
//! not reinvented here. What is
//! added here: L2 normalization, the asymmetric query/document input contract
//! (SPEC §3.3), a token counter for the chunker, and an explicit pooling
//! override so E3 can test both paths.
//!
//! A single llama_context cannot be used concurrently. Callers must serialize;
//! the daemon does so through the priority queue (SPEC §3.4).

const std = @import("std");

pub const c = @cImport({
    @cInclude("llama.h");
});

pub const Error = error{
    LoadFailed,
    TokenizeFailed,
    DecodeFailed,
    NoEmbedding,
    ContextTooSmall,
    OutOfMemory,
};

/// Matches llama.cpp's enum. `unspecified` lets the GGUF header decide, which
/// is what we want in production — but E3 must be able to force `last` to
/// prove the header is actually being honoured.
pub const Pooling = enum {
    unspecified,
    none,
    mean,
    cls,
    last,
    rank,

    fn toC(self: Pooling) c_int {
        return switch (self) {
            .unspecified => c.LLAMA_POOLING_TYPE_UNSPECIFIED,
            .none => c.LLAMA_POOLING_TYPE_NONE,
            .mean => c.LLAMA_POOLING_TYPE_MEAN,
            .cls => c.LLAMA_POOLING_TYPE_CLS,
            .last => c.LLAMA_POOLING_TYPE_LAST,
            .rank => c.LLAMA_POOLING_TYPE_RANK,
        };
    }
};

pub const Options = struct {
    /// Chunks are capped at 1024 tokens (SPEC §4.2), so a large context buys
    /// nothing and costs real time: llama.cpp sizes its compute buffers from
    /// n_batch, and every decode pays for them whether or not the work fills it.
    ///
    /// Measured on M2 Pro with an 841-token chunk (docs/experiments/E5-indexing.md):
    ///   n_ctx 8192 -> 562 ms   4096 -> 425 ms   2048 -> 353 ms   1536 -> 334 ms
    ///
    /// 2048 rather than 1536: only 5% off the floor, and it leaves 2x the chunk
    /// ceiling as headroom so an unusually long query still embeds instead of
    /// failing with ContextTooSmall.
    n_ctx: u32 = 2048,
    pooling: Pooling = .unspecified,
    n_gpu_layers: i32 = 999,
    /// Silence llama.cpp's load-time chatter. Diagnostics go through zkb's
    /// own logging; leaving this on makes CLI output unreadable.
    ///
    /// Callers should expose a way to turn this off: the silent path hides the
    /// one thing E4 needs to see — whether Metal was really used, or whether it
    /// fell back to CPU and still looked fine.
    quiet: bool = true,
};

var backend_inited = false;

fn silentLog(level: c.ggml_log_level, text: [*c]const u8, user: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user;
}

pub const Embedder = struct {
    allocator: std.mem.Allocator,
    model: *c.llama_model,
    ctx: *c.llama_context,
    vocab: *const c.llama_vocab,
    n_embd: u32,
    n_ctx: u32,
    pooling: Pooling,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, opts: Options) Error!Embedder {
        if (opts.quiet) c.llama_log_set(silentLog, null) else c.llama_log_set(null, null);
        if (!backend_inited) {
            c.llama_backend_init();
            backend_inited = true;
        }

        const path_z = try allocator.dupeZ(u8, model_path);
        defer allocator.free(path_z);

        var mparams = c.llama_model_default_params();
        mparams.n_gpu_layers = opts.n_gpu_layers;
        const model = c.llama_model_load_from_file(path_z.ptr, mparams) orelse
            return error.LoadFailed;
        errdefer c.llama_model_free(model);

        const vocab = c.llama_model_get_vocab(model) orelse return error.LoadFailed;
        const n_embd: u32 = @intCast(c.llama_model_n_embd(model));

        var cparams = c.llama_context_default_params();
        cparams.n_ctx = opts.n_ctx;
        // One full-prompt forward per call: batch must hold the whole prompt.
        cparams.n_batch = opts.n_ctx;
        cparams.embeddings = true;
        cparams.pooling_type = opts.pooling.toC();

        const ctx = c.llama_init_from_model(model, cparams) orelse return error.LoadFailed;

        return .{
            .allocator = allocator,
            .model = model,
            .ctx = ctx,
            .vocab = vocab,
            .n_embd = n_embd,
            .n_ctx = opts.n_ctx,
            .pooling = opts.pooling,
        };
    }

    pub fn deinit(self: *Embedder) void {
        c.llama_free(self.ctx);
        c.llama_model_free(self.model);
        self.* = undefined;
    }

    /// Token count using the embedding model's own tokenizer. The chunker must
    /// use this rather than a character heuristic: generic estimates are off by
    /// up to 30% on CJK, and a chunk over the limit is silently truncated by
    /// the model with no error (SPEC §4.2).
    pub fn countTokens(self: *Embedder, text: []const u8) Error!usize {
        const probe = c.llama_tokenize(self.vocab, text.ptr, @intCast(text.len), null, 0, true, true);
        return if (probe >= 0) @intCast(probe) else @intCast(-probe);
    }

    fn tokenize(self: *Embedder, text: []const u8) Error![]c.llama_token {
        const n = try self.countTokens(text);
        const tokens = try self.allocator.alloc(c.llama_token, n);
        errdefer self.allocator.free(tokens);
        const got = c.llama_tokenize(
            self.vocab,
            text.ptr,
            @intCast(text.len),
            tokens.ptr,
            @intCast(tokens.len),
            true,
            true,
        );
        if (got < 0 or @as(usize, @intCast(got)) != n) return error.TokenizeFailed;
        return tokens;
    }

    /// Embed raw text as-is. `out` must be at least `n_embd` long. Returns the
    /// L2-normalized vector as a slice of `out`.
    pub fn embed(self: *Embedder, text: []const u8, out: []f32) Error![]f32 {
        if (out.len < self.n_embd) return error.NoEmbedding;

        const tokens = try self.tokenize(text);
        defer self.allocator.free(tokens);
        if (tokens.len == 0) return error.TokenizeFailed;
        if (tokens.len > self.n_ctx) return error.ContextTooSmall;

        // Each call is independent — no KV carry-over between texts.
        c.llama_memory_clear(c.llama_get_memory(self.ctx), true);

        const batch = c.llama_batch_get_one(@constCast(tokens.ptr), @intCast(tokens.len));
        if (c.llama_decode(self.ctx, batch) != 0) return error.DecodeFailed;

        const dim: usize = self.n_embd;
        const src = c.llama_get_embeddings_seq(self.ctx, 0) orelse blk: {
            // Model didn't pool: fall back to the last token's hidden state,
            // which is what Qwen3-Embedding's own pooling would produce.
            break :blk c.llama_get_embeddings_ith(self.ctx, @intCast(tokens.len - 1)) orelse
                return error.NoEmbedding;
        };

        const vec = out[0..dim];
        @memcpy(vec, src[0..dim]);
        l2Normalize(vec);
        return vec;
    }

    /// Document side: heading path is prepended so an isolated chunk still
    /// lands in the right semantic neighbourhood (SPEC §3.3). No instruction
    /// prefix — Qwen3-Embedding wants that on the query side only.
    pub fn embedDocument(
        self: *Embedder,
        heading_path: []const u8,
        text: []const u8,
        out: []f32,
    ) Error![]f32 {
        if (heading_path.len == 0) return self.embed(text, out);
        const joined = std.fmt.allocPrint(
            self.allocator,
            "{s}\n\n{s}",
            .{ heading_path, text },
        ) catch return error.OutOfMemory;
        defer self.allocator.free(joined);
        return self.embed(joined, out);
    }

    /// Query side: Qwen3-Embedding is instruction-aware and measurably better
    /// with the prefix. `task` is part of the model identity (SPEC §3.3) —
    /// changing it shifts the query distribution away from stored documents.
    pub fn embedQuery(
        self: *Embedder,
        task: []const u8,
        query: []const u8,
        out: []f32,
    ) Error![]f32 {
        const prompt = std.fmt.allocPrint(
            self.allocator,
            "Instruct: {s}\nQuery: {s}",
            .{ task, query },
        ) catch return error.OutOfMemory;
        defer self.allocator.free(prompt);
        return self.embed(prompt, out);
    }
};

pub fn l2Normalize(v: []f32) void {
    var sumsq: f64 = 0;
    for (v) |x| sumsq += @as(f64, x) * @as(f64, x);
    if (sumsq == 0) return;
    const inv: f32 = @floatCast(1.0 / @sqrt(sumsq));
    for (v) |*x| x.* *= inv;
}

/// Cosine similarity. Both inputs are expected to be L2-normalized already,
/// so this is a plain dot product; kept explicit for readability at call sites.
pub fn cosine(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var dot: f64 = 0;
    for (a, b) |x, y| dot += @as(f64, x) * @as(f64, y);
    return @floatCast(dot);
}

pub const default_query_task =
    "Given a search query, retrieve relevant passages from a personal knowledge base";
