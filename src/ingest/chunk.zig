//! Greedy block packing into chunks.
//!
//! Token counts come from the embedding model's own tokenizer, injected as a
//! `TokenCounter`. Generic character heuristics are off by up to 30% on CJK,
//! and a chunk over the model's limit is silently truncated with no error —
//! so the count has to be the real one (SPEC §4.2).
//!
//! Parameters are the v0.1 values inherited from an earlier design. They are configurable
//! precisely because they have no experimental backing yet; E2's baseline is
//! what will let them be tuned against something.

const std = @import("std");
const markdown = @import("markdown.zig");

pub const Config = struct {
    target_tokens: usize = 800,
    max_tokens: usize = 1024,
    overlap_tokens: usize = 80,
};

/// Injected tokenizer. `ctx` is opaque so tests can count bytes and production
/// can call llama_tokenize without this module knowing about either.
pub const TokenCounter = struct {
    ctx: *anyopaque,
    countFn: *const fn (ctx: *anyopaque, text: []const u8) anyerror!usize,

    pub fn count(self: TokenCounter, text: []const u8) anyerror!usize {
        return self.countFn(self.ctx, text);
    }
};

pub const Chunk = struct {
    idx: usize,
    /// Owned by the Chunks collection this belongs to.
    heading_path: []u8,
    byte_start: usize,
    byte_end: usize,
    n_tokens: usize,
    /// Borrowed from the source document.
    text: []const u8,
};

pub const Chunks = struct {
    items: []Chunk,

    pub fn deinit(self: *Chunks, gpa: std.mem.Allocator) void {
        for (self.items) |c| gpa.free(c.heading_path);
        gpa.free(self.items);
        self.* = undefined;
    }
};

pub fn split(
    gpa: std.mem.Allocator,
    source: []const u8,
    doc: *const markdown.Document,
    counter: TokenCounter,
    cfg: Config,
) !Chunks {
    var out: std.ArrayList(Chunk) = .empty;
    errdefer {
        for (out.items) |c| gpa.free(c.heading_path);
        out.deinit(gpa);
    }

    // Token count per block, computed once. Summing per-block counts is not
    // exactly the count of the concatenation (BPE crosses boundaries), but it is
    // within a few tokens and only drives the packing decision — the stored
    // n_tokens is measured on the emitted text.
    const block_tokens = try gpa.alloc(usize, doc.blocks.len);
    defer gpa.free(block_tokens);
    for (doc.blocks, 0..) |b, i| {
        block_tokens[i] = try counter.count(source[b.byte_start..b.byte_end]);
    }

    var i: usize = 0;
    while (i < doc.blocks.len) {
        const start_block = i;

        // A single block over the hard limit cannot be packed; split it by line.
        if (block_tokens[i] > cfg.max_tokens) {
            try emitOversized(gpa, &out, source, doc, counter, cfg, i);
            i += 1;
            continue;
        }

        var tokens: usize = 0;
        var j = i;
        while (j < doc.blocks.len) {
            const bt = block_tokens[j];
            if (j > start_block) {
                if (tokens + bt > cfg.max_tokens) break;
                // A heading is a natural semantic boundary: prefer cutting here
                // over packing to exactly `target`, once the chunk has enough
                // substance to stand alone.
                if (doc.blocks[j].kind == .heading and tokens >= cfg.target_tokens / 2) break;
                // An oversized block starts its own chunk rather than being
                // line-split mid-pack.
                if (bt > cfg.max_tokens) break;
            }
            tokens += bt;
            j += 1;
            if (tokens >= cfg.target_tokens) break;
        }
        if (j == start_block) j = start_block + 1; // always consume something

        try emit(gpa, &out, source, doc, counter, start_block, j - 1);

        // Overlap: walk back over whole blocks until roughly `overlap_tokens`
        // are covered. Backing up by block rather than by token keeps chunks
        // from starting mid-sentence.
        var next = j;
        if (cfg.overlap_tokens > 0 and j - start_block > 1) {
            var acc: usize = 0;
            var k = j;
            while (k > start_block + 1 and acc < cfg.overlap_tokens) {
                k -= 1;
                acc += block_tokens[k];
            }
            next = k;
        }
        // Strict progress: without this a pathological overlap could loop.
        i = @max(next, start_block + 1);
    }

    return .{ .items = try out.toOwnedSlice(gpa) };
}

fn emit(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(Chunk),
    source: []const u8,
    doc: *const markdown.Document,
    counter: TokenCounter,
    first_block: usize,
    last_block: usize,
) !void {
    const byte_start = doc.blocks[first_block].byte_start;
    const byte_end = doc.blocks[last_block].byte_end;
    const text = std.mem.trim(u8, source[byte_start..byte_end], " \t\r\n");
    if (text.len == 0) return;

    const heading_path = try doc.headingPathAt(gpa, source, first_block);
    errdefer gpa.free(heading_path);

    try out.append(gpa, .{
        .idx = out.items.len,
        .heading_path = heading_path,
        .byte_start = byte_start,
        .byte_end = byte_end,
        .n_tokens = try counter.count(text),
        .text = text,
    });
}

/// One block larger than max_tokens — in practice a long code block or table.
/// Split on line boundaries so the pieces stay readable, and mark the
/// continuations in the heading path so a hit is not mistaken for a whole unit.
fn emitOversized(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(Chunk),
    source: []const u8,
    doc: *const markdown.Document,
    counter: TokenCounter,
    cfg: Config,
    block_index: usize,
) !void {
    const b = doc.blocks[block_index];
    const base_path = try doc.headingPathAt(gpa, source, block_index);
    defer gpa.free(base_path);

    var part: usize = 0;
    var pos = b.byte_start;
    while (pos < b.byte_end) {
        var end = pos;
        var tokens: usize = 0;
        while (end < b.byte_end) {
            const nl = std.mem.indexOfScalarPos(u8, source, end, '\n') orelse b.byte_end;
            const line_end = @min(nl + 1, b.byte_end);
            const lt = try counter.count(source[end..line_end]);
            if (tokens + lt > cfg.max_tokens and end > pos) break;
            tokens += lt;
            end = line_end;
            if (end >= b.byte_end) break;
        }
        if (end == pos) end = @min(pos + 1, b.byte_end); // never stall

        const text = std.mem.trim(u8, source[pos..end], " \t\r\n");
        if (text.len != 0) {
            const path = if (part == 0)
                try gpa.dupe(u8, base_path)
            else
                try std.fmt.allocPrint(gpa, "{s} (cont. {d})", .{ base_path, part });
            errdefer gpa.free(path);
            try out.append(gpa, .{
                .idx = out.items.len,
                .heading_path = path,
                .byte_start = pos,
                .byte_end = end,
                .n_tokens = try counter.count(text),
                .text = text,
            });
            part += 1;
        }
        pos = end;
    }
}

// ---------------------------------------------------------------------------
// A byte-based counter for tests and for callers that have no model loaded.
// Not a substitute for the real tokenizer in production.
// ---------------------------------------------------------------------------

pub const ByteCounter = struct {
    bytes_per_token: usize = 4,

    pub fn counter(self: *ByteCounter) TokenCounter {
        return .{ .ctx = self, .countFn = countImpl };
    }

    fn countImpl(ctx: *anyopaque, text: []const u8) anyerror!usize {
        const self: *ByteCounter = @ptrCast(@alignCast(ctx));
        return (text.len + self.bytes_per_token - 1) / self.bytes_per_token;
    }
};
