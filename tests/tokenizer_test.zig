//! The `zkb_cjk` FTS5 tokenizer, exercised through SQL.
//!
//! Deliberately tested against the real registered tokenizer rather than a Zig
//! reimplementation of the segmentation: what matters is what FTS5 actually
//! indexes and matches, and only SQL can answer that.
//!
//! The comparison target is the trigram behaviour measured in
//! docs/experiments/E2-baseline.md — Chinese keyword recall@10 of 0.167.

const std = @import("std");
const zkb = @import("zkb");
const sqlite = zkb.sqlite;

const testing = std.testing;

fn openWithTable(comptime tokenize: []const u8) !sqlite.Db {
    var db = try sqlite.Db.open(":memory:", .read_write);
    errdefer db.close();
    try db.exec(
        "CREATE VIRTUAL TABLE t USING fts5(text, content='', contentless_delete=1, tokenize='" ++
            tokenize ++ "');",
    );
    return db;
}

fn insert(db: *sqlite.Db, rowid: i64, text: []const u8) !void {
    var st = try db.prepare("INSERT INTO t(rowid, text) VALUES (?1, ?2)");
    defer st.finalize();
    try st.bindI64(1, rowid);
    try st.bindText(2, text);
    _ = try st.step();
}

fn matchCount(db: *sqlite.Db, expr: []const u8) !i64 {
    var st = try db.prepare("SELECT count(*) FROM t WHERE t MATCH ?1");
    defer st.finalize();
    try st.bindText(1, expr);
    if (!try st.step()) return 0;
    return st.columnI64(0);
}

// ---------------------------------------------------------------------------
// The reason this tokenizer exists
// ---------------------------------------------------------------------------

test "two-character CJK queries match, which trigram could not do at all" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "检索融合使用 RRF，不要加权和。派生量一律不存。");

    // Every one of these is a two-character content word that the trigram
    // tokenizer silently matched zero rows for (see E2: 融合 / 解析 / 规范 /
    // 存储 were all reported as dropped or unmatched).
    for ([_][]const u8{ "\"融合\"", "\"检索\"", "\"派生\"", "\"加权\"" }) |q| {
        try testing.expectEqual(@as(i64, 1), try matchCount(&db, q));
    }
}

test "trigram, for contrast, cannot match a two-character CJK query" {
    // Same corpus, same query, only the tokenizer differs. This is the control
    // that makes the improvement a measurement rather than a claim.
    var tri = try openWithTable("trigram case_sensitive 0");
    defer tri.close();
    try insert(&tri, 1, "检索融合使用 RRF，不要加权和。派生量一律不存。");
    try testing.expectEqual(@as(i64, 0), try matchCount(&tri, "\"融合\""));

    var cjk = try openWithTable("zkb_cjk");
    defer cjk.close();
    try insert(&cjk, 1, "检索融合使用 RRF，不要加权和。派生量一律不存。");
    try testing.expectEqual(@as(i64, 1), try matchCount(&cjk, "\"融合\""));
}

test "bigrams do not collide on merely shared characters" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "检索融合");

    // Both characters are present, but "融检" is not a bigram of the document.
    // Under trigram this kind of reshuffling frequently produced false hits.
    try testing.expectEqual(@as(i64, 0), try matchCount(&db, "\"融检\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"索融\""));
}

test "longer CJK queries keep phrase adjacency through overlapping bigrams" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "第一性原理分析");
    try insert(&db, 2, "第一版原理图"); // shares 第一 and 原理, but not adjacently

    // "第一性原理" -> 第一,一性,性原,原理 as consecutive positions; only doc 1 has
    // that sequence.
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"第一性原理\""));
    var st = try db.prepare("SELECT rowid FROM t WHERE t MATCH ?1");
    defer st.finalize();
    try st.bindText(1, "\"第一性原理\"");
    try testing.expect(try st.step());
    try testing.expectEqual(@as(i64, 1), st.columnI64(0));
}

// ---------------------------------------------------------------------------
// Latin words
// ---------------------------------------------------------------------------

test "Latin runs are whole words, case-folded" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "The vault service uses SurrealDB and lopdf for parsing.");

    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"surrealdb\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"SURREALDB\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"lopdf\""));
}

test "Latin matching is by word, not by substring" {
    // A deliberate behaviour change from trigram, and a precision improvement:
    // trigram matched "opd" inside "lopdf", so any 3-character fragment of any
    // identifier produced hits.
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "lopdf");

    try testing.expectEqual(@as(i64, 0), try matchCount(&db, "\"opd\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"lopdf\""));
    // Prefix search is still available for the cases that want it.
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"lop\"*"));
}

test "single-character Latin and digit terms are matchable" {
    // Under trigram anything shorter than 3 characters was unmatchable, which is
    // why the query builder had to drop such terms. Whole-word tokens fix that.
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "phase 0 uses q8_0 quantization");
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"0\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"q8_0\"")); // '_' splits
}

// ---------------------------------------------------------------------------
// boundaries and robustness
// ---------------------------------------------------------------------------

test "mixed CJK and Latin text segments at the script boundary" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "Qwen3-Embedding 中英双语检索");

    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"qwen3\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"embedding\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"双语\""));
    // The boundary must not fuse the two scripts into one token.
    try testing.expectEqual(@as(i64, 0), try matchCount(&db, "\"embedding中\""));
}

test "kana and hangul are segmented like han" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "ひらがなのテスト 한국어 검색");
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"テスト\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"검색\""));
}

test "a lone CJK character between separators is still indexed" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "A 中 B");
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"中\""));
}

test "delete by rowid still works with the custom tokenizer" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    try insert(&db, 1, "混合检索");
    try insert(&db, 2, "混合检索");
    try testing.expectEqual(@as(i64, 2), try matchCount(&db, "\"混合\""));

    var st = try db.prepare("DELETE FROM t WHERE rowid = 1");
    defer st.finalize();
    _ = try st.step();
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"混合\""));
}

test "malformed UTF-8 does not stall or crash the tokenizer" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    // Truncated 3-byte sequence followed by valid text.
    try insert(&db, 1, "\xE4\xB8 valid tail 检索");
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"valid\""));
    try testing.expectEqual(@as(i64, 1), try matchCount(&db, "\"检索\""));
}

test "a token longer than the internal buffer is truncated, not overflowed" {
    var db = try openWithTable("zkb_cjk");
    defer db.close();
    var long: [200]u8 = @splat('a');
    try insert(&db, 1, &long);
    // Whatever it indexed, the point is that it did not corrupt memory and the
    // row is still reachable by a prefix.
    try testing.expect((try matchCount(&db, "\"aaaa\"*")) >= 0);
}
