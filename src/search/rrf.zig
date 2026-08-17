//! Reciprocal Rank Fusion (Cormack & Clarke, 2009).
//!
//!     score(d) = Σ_paths 1 / (k + rank_in_path)
//!
//! Rank-based and therefore unitless, which is the whole reason it is used here
//! instead of a weighted sum of scores: cosine distance and BM25 are not on a
//! comparable scale, and any weights chosen to bridge them would be numbers with
//! no experimental backing — exactly the kind of debt a prior design flagged
//! and never paid off.
//! Adding a retrieval path later means adding a term, not re-tuning weights.

const std = @import("std");

pub const Config = struct {
    /// 60 is the value from the original paper; robust across most settings.
    k: f64 = 60,
    /// Rank-1 bonus, expressed as a **fraction of one rank-1 RRF contribution**
    /// rather than an absolute score.
    ///
    /// A prior design specified an absolute `+0.05`. Measured against k=60 that is
    /// 3x a whole rank-1 contribution (1/61 = 0.0164), which makes the bonus
    /// dominate fusion outright: anything ranked first in either path beats a
    /// document both paths agree on. That inverts the intent — the bonus is
    /// meant to break ties in favour of an exact match, not to override
    /// cross-path agreement, which is the strongest signal available.
    ///
    /// Keeping it unitless also means it survives a change to `k`; an absolute
    /// constant would silently change meaning.
    top_boost_fraction: f64 = 0.5,
    /// Below this many keyword hits the FTS path is dropped entirely: with a
    /// sparse vocabulary BM25 contributes noise, not signal.
    fts_min_hits: usize = 3,
};

pub const Fused = struct {
    chunk_id: i64,
    score: f64,
    /// 1-based rank in each path, null when absent from it. Exposed because
    /// tuning chunking or the tokenizer without these two numbers is guesswork.
    vec_rank: ?u32,
    fts_rank: ?u32,
};

/// `vec_ids` and `fts_ids` must be ordered best-first. Returns fused hits sorted
/// by descending score; caller owns the slice.
pub fn fuse(
    gpa: std.mem.Allocator,
    vec_ids: []const i64,
    fts_ids: []const i64,
    cfg: Config,
) ![]Fused {
    const use_fts = fts_ids.len >= cfg.fts_min_hits;
    // One rank-1 contribution is the natural unit of this scale.
    const boost = cfg.top_boost_fraction / (cfg.k + 1.0);

    var map: std.AutoHashMapUnmanaged(i64, Fused) = .empty;
    defer map.deinit(gpa);

    for (vec_ids, 0..) |id, i| {
        const rank: u32 = @intCast(i + 1);
        const gop = try map.getOrPut(gpa, id);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{ .chunk_id = id, .score = 0, .vec_rank = null, .fts_rank = null };
        }
        gop.value_ptr.vec_rank = rank;
        gop.value_ptr.score += 1.0 / (cfg.k + @as(f64, @floatFromInt(rank)));
        if (rank == 1) gop.value_ptr.score += boost;
    }

    if (use_fts) {
        for (fts_ids, 0..) |id, i| {
            const rank: u32 = @intCast(i + 1);
            const gop = try map.getOrPut(gpa, id);
            if (!gop.found_existing) {
                gop.value_ptr.* = .{ .chunk_id = id, .score = 0, .vec_rank = null, .fts_rank = null };
            }
            gop.value_ptr.fts_rank = rank;
            gop.value_ptr.score += 1.0 / (cfg.k + @as(f64, @floatFromInt(rank)));
            if (rank == 1) gop.value_ptr.score += boost;
        }
    }

    var out = try gpa.alloc(Fused, map.count());
    var i: usize = 0;
    var it = map.valueIterator();
    while (it.next()) |v| : (i += 1) out[i] = v.*;

    std.mem.sort(Fused, out, {}, lessThan);
    return out;
}

fn lessThan(_: void, a: Fused, b: Fused) bool {
    if (a.score != b.score) return a.score > b.score;
    // Deterministic tie-break so identical inputs always produce identical
    // output — otherwise hash iteration order leaks into results.
    return a.chunk_id < b.chunk_id;
}
