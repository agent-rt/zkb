//! Retrieval-layer behaviour: MATCH construction and RRF arithmetic.
//!
//! Both are places where a subtle mistake produces plausible-looking but wrong
//! rankings, with nothing to signal it.

const std = @import("std");
const zkb = @import("zkb");
const fts_query = zkb.fts_query;
const rrf = zkb.rrf;

const testing = std.testing;
const gpa = testing.allocator;

// ---------------------------------------------------------------------------
// MATCH construction
// ---------------------------------------------------------------------------

test "terms are forced to literals so FTS5 syntax in input is inert" {
    var q = try fts_query.build(gpa, "retrieval OR fusion NEAR ranking");
    defer q.deinit(gpa);

    const expr = q.expr.?;
    // Every term quoted; the user's "OR"/"NEAR" become searched words, not
    // operators. Only the separators we insert are bare.
    try testing.expect(std.mem.indexOf(u8, expr, "\"retrieval\"") != null);
    try testing.expect(std.mem.indexOf(u8, expr, "\"NEAR\"") != null);
    try testing.expect(std.mem.indexOf(u8, expr, "\"ranking\"") != null);
}

test "embedded double quotes are doubled, not left to break the expression" {
    var q = try fts_query.build(gpa, "say\"hello");
    defer q.deinit(gpa);
    // FTS5 escapes " inside a quoted string by doubling it.
    try testing.expectEqualStrings("\"say\"\"hello\"", q.expr.?);
}

test "two-character CJK words survive now that the tokenizer bigrams them" {
    // Under trigram these were dropped, which threw away real content words.
    // The bigram tokenizer indexes them, so the drop rule must not fire.
    var q = try fts_query.build(gpa, "排障 融合 解析 规范 存储");
    defer q.deinit(gpa);

    try testing.expect(q.expr != null);
    try testing.expectEqual(@as(usize, 0), q.dropped.len);
    for ([_][]const u8{ "排障", "融合", "解析", "规范", "存储" }) |t| {
        try testing.expect(std.mem.indexOf(u8, q.expr.?, t) != null);
    }
}

test "short Latin and digit terms are matchable as whole-word tokens" {
    var q = try fts_query.build(gpa, "a of 0 q8");
    defer q.deinit(gpa);
    try testing.expect(q.expr != null);
    try testing.expectEqual(@as(usize, 0), q.dropped.len);
}

test "a lone CJK character is still dropped, and reported" {
    // A single character forms no bigram. This is the entire remaining scope of
    // the drop rule.
    var q = try fts_query.build(gpa, "的 检索");
    defer q.deinit(gpa);
    try testing.expect(q.expr != null);
    try testing.expect(std.mem.indexOf(u8, q.expr.?, "检索") != null);
    try testing.expectEqual(@as(usize, 1), q.dropped.len);
    try testing.expectEqualStrings("的", q.dropped[0]);
}

test "term length is measured in codepoints, not bytes" {
    // "排障" is 2 characters but 6 bytes. Under the bigram tokenizer it is
    // matchable; a byte-length rule would have judged it by the wrong number
    // in either direction.
    var q = try fts_query.build(gpa, "排障");
    defer q.deinit(gpa);
    try testing.expect(q.expr != null);
    try testing.expectEqual(@as(usize, 0), q.dropped.len);
}

test "a query of only unusable terms yields no expression" {
    var q = try fts_query.build(gpa, "的 了 ., ;");
    defer q.deinit(gpa);
    try testing.expect(q.expr == null);
    try testing.expectEqual(@as(usize, 2), q.dropped.len);
}

test "punctuation splits terms but multibyte sequences are never split" {
    var q = try fts_query.build(gpa, "sqlite-vec,partition(key)");
    defer q.deinit(gpa);
    const expr = q.expr.?;
    // '-' is not a separator here (it is common inside identifiers), but ',' and
    // parens are.
    try testing.expect(std.mem.indexOf(u8, expr, "\"sqlite-vec\"") != null);
    try testing.expect(std.mem.indexOf(u8, expr, "\"partition\"") != null);
    try testing.expect(std.mem.indexOf(u8, expr, "\"key\"") != null);

    // A space-free CJK run is cut into the same overlapping bigrams the index
    // holds, never kept as one phrase. Keeping it whole demanded that the entire
    // character sequence appear adjacently, which for a natural-language question
    // never happens — measured 0 keyword hits for a real query before this fix.
    var q2 = try fts_query.build(gpa, "向量检索与全文检索互补");
    defer q2.deinit(gpa);
    const e2 = q2.expr.?;
    try testing.expect(std.mem.indexOf(u8, e2, "向量检索与全文检索互补") == null);
    for ([_][]const u8{ "\"向量\"", "\"量检\"", "\"检索\"", "\"互补\"" }) |bg| {
        try testing.expect(std.mem.indexOf(u8, e2, bg) != null);
    }
    // 11 characters -> 10 overlapping bigrams.
    try testing.expectEqual(@as(usize, 9), std.mem.count(u8, e2, " OR "));
}

test "a mixed-script term is segmented per script, like the tokenizer does" {
    var q = try fts_query.build(gpa, "Qwen3中英双语");
    defer q.deinit(gpa);
    const e = q.expr.?;
    try testing.expect(std.mem.indexOf(u8, e, "\"qwen3\"") != null or
        std.mem.indexOf(u8, e, "\"Qwen3\"") != null);
    try testing.expect(std.mem.indexOf(u8, e, "\"中英\"") != null);
    try testing.expect(std.mem.indexOf(u8, e, "\"双语\"") != null);
    // The script boundary must not be fused into a single token.
    try testing.expect(std.mem.indexOf(u8, e, "\"Qwen3中\"") == null);
}

// ---------------------------------------------------------------------------
// RRF
// ---------------------------------------------------------------------------

test "a document ranked in both paths outranks one ranked in only one" {
    // id 2 is rank 2 in both; id 1 is rank 1 in vectors only. Two contributions
    // beat one plus the top boost.
    const vec = [_]i64{ 1, 2, 3 };
    const fts = [_]i64{ 5, 2, 6 };
    const fused = try rrf.fuse(gpa, &vec, &fts, .{});
    defer gpa.free(fused);

    try testing.expectEqual(@as(i64, 2), fused[0].chunk_id);
    try testing.expectEqual(@as(?u32, 2), fused[0].vec_rank);
    try testing.expectEqual(@as(?u32, 2), fused[0].fts_rank);
}

test "the first-place bonus nudges but never overrides cross-path agreement" {
    // The bonus is a fraction of one rank-1 contribution, so a rank-1 hit in a
    // single path must still lose to a document both paths found. An absolute
    // bonus (an absolute +0.05 against k=60) inverted this.
    const vec = [_]i64{ 1, 2, 3 };
    const fts = [_]i64{ 5, 2, 6 };
    const fused = try rrf.fuse(gpa, &vec, &fts, .{});
    defer gpa.free(fused);
    try testing.expectEqual(@as(i64, 2), fused[0].chunk_id);

    // But among documents seen in one path only, rank 1 wins.
    const v2 = [_]i64{ 10, 11, 12 };
    const f2 = [_]i64{ 20, 21, 22 };
    const fused2 = try rrf.fuse(gpa, &v2, &f2, .{});
    defer gpa.free(fused2);
    try testing.expect(fused2[0].chunk_id == 10 or fused2[0].chunk_id == 20);
    try testing.expect(fused2[1].chunk_id == 10 or fused2[1].chunk_id == 20);
    try testing.expect(fused2[1].score > fused2[2].score);
}

test "the keyword path is dropped entirely when the vocabulary is too sparse" {
    // Fewer than fts_min_hits keyword results means BM25 is contributing noise.
    const vec = [_]i64{ 1, 2, 3 };
    const fts = [_]i64{99}; // 1 < default fts_min_hits (3)
    const fused = try rrf.fuse(gpa, &vec, &fts, .{});
    defer gpa.free(fused);

    for (fused) |f| {
        try testing.expect(f.chunk_id != 99);
        try testing.expectEqual(@as(?u32, null), f.fts_rank);
    }
}

test "fusion is deterministic for identical inputs" {
    const vec = [_]i64{ 7, 8, 9, 10 };
    const fts = [_]i64{ 10, 9, 8, 7 };

    const a = try rrf.fuse(gpa, &vec, &fts, .{});
    defer gpa.free(a);
    const b = try rrf.fuse(gpa, &vec, &fts, .{});
    defer gpa.free(b);

    // Hash iteration order must not leak into the ranking.
    try testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| try testing.expectEqual(x.chunk_id, y.chunk_id);
}

test "scores are ordered descending and ranks are 1-based" {
    const vec = [_]i64{ 4, 5, 6 };
    const fused = try rrf.fuse(gpa, &vec, &.{}, .{});
    defer gpa.free(fused);

    try testing.expectEqual(@as(usize, 3), fused.len);
    try testing.expectEqual(@as(?u32, 1), fused[0].vec_rank);
    var i: usize = 1;
    while (i < fused.len) : (i += 1) {
        try testing.expect(fused[i - 1].score >= fused[i].score);
    }
}
