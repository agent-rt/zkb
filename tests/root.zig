//! M0 地基验证实验。E1 与 E6 是常驻单测——它们锁住的是升级 sqlite / sqlite-vec
//! 时最容易静默回归的两处语义。见 SPEC §10。

const std = @import("std");
const zkb = @import("zkb");
const sqlite = zkb.sqlite;

const testing = std.testing;

test {
    _ = @import("store_test.zig");
    _ = @import("chunk_test.zig");
    _ = @import("search_test.zig");
    _ = @import("tokenizer_test.zig");
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn openMem() !sqlite.Db {
    return sqlite.Db.open(":memory:", .read_write);
}

/// Deterministic pseudo-random unit vector. Seeded per call so the test is
/// reproducible across runs and machines.
fn unitVector(seed: u64, out: []f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    var sumsq: f32 = 0;
    for (out) |*v| {
        v.* = rand.float(f32) * 2.0 - 1.0;
        sumsq += v.* * v.*;
    }
    const norm = @sqrt(sumsq);
    if (norm > 0) for (out) |*v| {
        v.* /= norm;
    };
}

// ---------------------------------------------------------------------------
// build contract
// ---------------------------------------------------------------------------

test "build contract: FTS5 compiled in and versions pinned" {
    var db = try openMem();
    defer db.close();

    try testing.expect(sqlite.hasCompileOption(&db, "ENABLE_FTS5"));
    try testing.expect(sqlite.hasCompileOption(&db, "THREADSAFE=2"));
    try testing.expectEqualStrings("3.49.1", sqlite.libVersion());
    try testing.expectEqualStrings("v0.1.6", sqlite.vecVersion());
}

// ---------------------------------------------------------------------------
// E1 — vec0 integer partition key coexists with a `k = ?` KNN constraint,
//      and the partition filter is applied *inside* the KNN, not after it.
//
// This is the load-bearing assumption behind multi-collection search: if the
// filter were applied after the top-k, asking for 50 hits in collection 1
// would silently return fewer than 50. SPEC §5.1.
// ---------------------------------------------------------------------------

test "E1: vec0 partition key + k constraint, filter applied inside KNN" {
    const dim = 8;
    var db = try openMem();
    defer db.close();

    db.exec(
        \\CREATE VIRTUAL TABLE vec_chunks USING vec0(
        \\  collection_id INTEGER partition key,
        \\  chunk_id INTEGER PRIMARY KEY,
        \\  embedding FLOAT[8] distance_metric=cosine
        \\);
    ) catch |err| {
        std.debug.print("E1 DDL failed: {s}\n", .{db.lastError()});
        return err;
    };

    // Collection 1 gets chunk_id 1000..1099, collection 2 gets 2000..2099.
    {
        var st = try db.prepare(
            "INSERT INTO vec_chunks(collection_id, chunk_id, embedding) VALUES (?1, ?2, ?3)",
        );
        defer st.finalize();
        var vec: [dim]f32 = undefined;
        for (0..100) |i| {
            inline for (.{ .{ 1, 1000 }, .{ 2, 2000 } }) |pair| {
                unitVector(@intCast(pair[0] * 100000 + i), &vec);
                st.reset();
                try st.bindI64(1, pair[0]);
                try st.bindI64(2, pair[1] + @as(i64, @intCast(i)));
                try st.bindVector(3, &vec);
                try testing.expect(!try st.step());
            }
        }
    }

    try testing.expectEqual(@as(?i64, 200), try db.queryI64("SELECT count(*) FROM vec_chunks"));

    // Ask for exactly 50 neighbours within collection 1.
    var query: [dim]f32 = undefined;
    unitVector(42, &query);

    var st = db.prepare(
        \\SELECT chunk_id, distance FROM vec_chunks
        \\WHERE collection_id = ?1 AND embedding MATCH ?2 AND k = 50
    ) catch |err| {
        std.debug.print("E1 KNN prepare failed: {s}\n", .{db.lastError()});
        return err;
    };
    defer st.finalize();
    try st.bindI64(1, 1);
    try st.bindVector(2, &query);

    var count: usize = 0;
    var last_distance: f64 = -1;
    while (try st.step()) {
        const id = st.columnI64(0);
        const distance = st.columnF64(1);
        // Every hit must belong to collection 1.
        try testing.expect(id >= 1000 and id <= 1099);
        // Results must arrive in nondecreasing distance order.
        try testing.expect(distance >= last_distance);
        last_distance = distance;
        count += 1;
    }

    // The whole point: a post-filter would yield fewer than 50 here.
    try testing.expectEqual(@as(usize, 50), count);
}

test "E1b: production DDL (1024-dim, cosine) is accepted" {
    var db = try openMem();
    defer db.close();
    db.exec(
        \\CREATE VIRTUAL TABLE vec_chunks USING vec0(
        \\  collection_id INTEGER partition key,
        \\  chunk_id INTEGER PRIMARY KEY,
        \\  embedding FLOAT[1024] distance_metric=cosine
        \\);
    ) catch |err| {
        std.debug.print("E1b DDL failed: {s}\n", .{db.lastError()});
        return err;
    };
}

test "E1c: KNN without k or LIMIT is rejected at prepare time" {
    // Documents the constraint rather than rediscovering it later. Note the
    // rejection lands in prepare(), not step(): vec0's xBestIndex runs during
    // statement preparation, so a malformed KNN fails before any row work.
    var db = try openMem();
    defer db.close();
    try db.exec(
        \\CREATE VIRTUAL TABLE v USING vec0(
        \\  chunk_id INTEGER PRIMARY KEY, embedding FLOAT[4] distance_metric=cosine
        \\);
    );
    var vec = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    {
        var ins = try db.prepare("INSERT INTO v(chunk_id, embedding) VALUES (1, ?1)");
        defer ins.finalize();
        try ins.bindVector(1, &vec);
        try testing.expect(!try ins.step());
    }

    try testing.expectError(
        error.SqlitePrepare,
        db.prepare("SELECT chunk_id FROM v WHERE embedding MATCH ?1"),
    );
    // The message is what a user would have to act on, so assert it is useful.
    try testing.expect(std.mem.indexOf(u8, db.lastError(), "k = ?") != null);

    // Same query with k is fine.
    var st = try db.prepare("SELECT chunk_id FROM v WHERE embedding MATCH ?1 AND k = 1");
    defer st.finalize();
    try st.bindVector(1, &vec);
    try testing.expect(try st.step());
    try testing.expectEqual(@as(i64, 1), st.columnI64(0));
}

// ---------------------------------------------------------------------------
// E6 — FTS5 contentless_delete: deleting by rowid alone must actually remove
//      the row from the index, and the rowid must be reusable afterwards.
//
// Without contentless_delete an external-content table needs the original
// column values to delete, which would force the ingest path to re-read text
// it already discarded. If this regressed we would silently keep matching
// deleted chunks. SPEC §2.5.
// ---------------------------------------------------------------------------

test "E6: fts5 contentless_delete removes rows and allows rowid reuse" {
    var db = try openMem();
    defer db.close();

    db.exec(
        \\CREATE VIRTUAL TABLE fts_chunks USING fts5(
        \\  text, heading_path,
        \\  content='', contentless_delete=1,
        \\  tokenize='trigram case_sensitive 0'
        \\);
    ) catch |err| {
        std.debug.print("E6 DDL failed: {s}\n", .{db.lastError()});
        return err;
    };

    {
        var st = try db.prepare("INSERT INTO fts_chunks(rowid, text, heading_path) VALUES (?1, ?2, ?3)");
        defer st.finalize();
        for (1..101) |i| {
            st.reset();
            try st.bindI64(1, @intCast(i));
            try st.bindText(2, "retrieval fusion uses reciprocal rank");
            try st.bindText(3, "spec > retrieval");
            try testing.expect(!try st.step());
        }
    }

    const match_count =
        \\SELECT count(*) FROM fts_chunks WHERE fts_chunks MATCH '"reciprocal"'
    ;
    try testing.expectEqual(@as(?i64, 100), try db.queryI64(match_count));

    // Delete the even rowids by rowid alone — no column values supplied.
    {
        var st = try db.prepare("DELETE FROM fts_chunks WHERE rowid = ?1");
        defer st.finalize();
        var i: i64 = 2;
        while (i <= 100) : (i += 2) {
            st.reset();
            try st.bindI64(1, i);
            try testing.expect(!try st.step());
        }
    }

    try testing.expectEqual(@as(?i64, 50), try db.queryI64(match_count));

    // No deleted rowid may survive in the index.
    {
        var st = try db.prepare(
            \\SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH '"reciprocal"'
        );
        defer st.finalize();
        while (try st.step()) {
            try testing.expect(@rem(st.columnI64(0), 2) == 1);
        }
    }

    // A deleted rowid must be reusable, with fresh content queryable.
    {
        var st = try db.prepare("INSERT INTO fts_chunks(rowid, text, heading_path) VALUES (?1, ?2, ?3)");
        defer st.finalize();
        try st.bindI64(1, 2);
        try st.bindText(2, "materialized columns beat entity attribute value");
        try st.bindText(3, "spec > records");
        try testing.expect(!try st.step());
    }
    try testing.expectEqual(
        @as(?i64, 2),
        try db.queryI64(
            \\SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH '"materialized"'
        ),
    );
}

// ---------------------------------------------------------------------------
// trigram tokenizer behaviour — the basis for the query-construction rule in
// SPEC §5.1 (terms shorter than 3 characters cannot match and must be
// reported as dropped rather than silently ignored).
// ---------------------------------------------------------------------------

test "trigram: CJK matches at 3+ chars, cannot match at 2" {
    var db = try openMem();
    defer db.close();
    try db.exec(
        \\CREATE VIRTUAL TABLE t USING fts5(
        \\  text, content='', contentless_delete=1, tokenize='trigram case_sensitive 0'
        \\);
    );
    {
        var st = try db.prepare("INSERT INTO t(rowid, text) VALUES (1, ?1)");
        defer st.finalize();
        try st.bindText(1, "第一性原理排障：先测量再推理");
        try testing.expect(!try st.step());
    }

    // 3 characters -> a full trigram exists -> matches.
    {
        var st = try db.prepare("SELECT count(*) FROM t WHERE t MATCH ?1");
        defer st.finalize();
        try st.bindText(1, "\"第一性\"");
        try testing.expect(try st.step());
        try testing.expectEqual(@as(i64, 1), st.columnI64(0));
    }

    // 2 characters -> no trigram can be formed, and FTS5 does NOT error: it
    // silently matches nothing. That silence is exactly why the query builder
    // must report such terms as `dropped_terms` instead of letting the caller
    // believe the search covered them. SPEC §5.1.
    {
        var st = try db.prepare("SELECT count(*) FROM t WHERE t MATCH ?1");
        defer st.finalize();
        try st.bindText(1, "\"排障\"");
        try testing.expect(try st.step());
        try testing.expectEqual(@as(i64, 0), st.columnI64(0));
    }

    // Same for a 1-char ASCII term, so the drop rule is length-based and not
    // script-specific.
    {
        var st = try db.prepare("SELECT count(*) FROM t WHERE t MATCH ?1");
        defer st.finalize();
        try st.bindText(1, "\"先\"");
        try testing.expect(try st.step());
        try testing.expectEqual(@as(i64, 0), st.columnI64(0));
    }
}
