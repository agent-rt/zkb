const std = @import("std");
const zkb = @import("zkb");
const bench = zkb.bench;

const testing = std.testing;

// ---------------------------------------------------------------------------
// path matching
// ---------------------------------------------------------------------------

test "pathMatches: equal, and suffix at a component boundary" {
    try testing.expect(bench.pathMatches("docs/DESIGN.md", "docs/DESIGN.md"));
    try testing.expect(bench.pathMatches("docs/DESIGN.md", "DESIGN.md"));
    try testing.expect(bench.pathMatches("notes/docs/DESIGN.md", "docs/DESIGN.md"));
}

test "pathMatches: a suffix that is not on a boundary is not a match" {
    // The regression this whole rule exists for. qmd's bench harness matches
    // with a bare bidirectional endsWith, so `a.md` counts as a hit against a
    // result of `xa.md` — recall comes out inflated and nothing says so.
    try testing.expect(!bench.pathMatches("xa.md", "a.md"));
    try testing.expect(!bench.pathMatches("docs/xDESIGN.md", "DESIGN.md"));
    try testing.expect(!bench.pathMatches("adocs/DESIGN.md", "docs/DESIGN.md"));
}

test "pathMatches: not symmetric — a result may not stand in for a longer expectation" {
    // Also unlike qmd, which tries the suffix both ways. Returning `DESIGN.md`
    // when the fixture asked for `docs/DESIGN.md` is a different document.
    try testing.expect(!bench.pathMatches("DESIGN.md", "docs/DESIGN.md"));
}

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

test "score: expected document at rank 1" {
    const ranked = [_][]const u8{ "docs/DESIGN.md", "docs/recipes.md", "README.md" };
    const expected = [_][]const u8{"DESIGN.md"};
    const m = bench.score(&ranked, &expected, 10);
    try testing.expectEqual(@as(f64, 1), m.recall_at_1);
    try testing.expectEqual(@as(f64, 1), m.recall_at_k);
    try testing.expectEqual(@as(f64, 1), m.mrr);
    try testing.expectEqual(@as(usize, 1), m.found);
}

test "score: rank 3 lands in @3 and @5 but not @1" {
    const ranked = [_][]const u8{ "a.md", "b.md", "docs/DESIGN.md", "c.md" };
    const expected = [_][]const u8{"docs/DESIGN.md"};
    const m = bench.score(&ranked, &expected, 10);
    try testing.expectEqual(@as(f64, 0), m.recall_at_1);
    try testing.expectEqual(@as(f64, 1), m.recall_at_3);
    try testing.expectEqual(@as(f64, 1), m.recall_at_5);
    try testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), m.mrr, 1e-12);
}

test "score: two expected documents, one found" {
    const ranked = [_][]const u8{ "a.md", "docs/DESIGN.md" };
    const expected = [_][]const u8{ "docs/DESIGN.md", "docs/recipes.md" };
    const m = bench.score(&ranked, &expected, 10);
    try testing.expectEqual(@as(f64, 0.5), m.recall_at_k);
    try testing.expectEqual(@as(usize, 1), m.found);
    try testing.expectEqual(@as(usize, 2), m.expected);
    try testing.expectEqual(@as(f64, 0.5), m.mrr);
}

test "score: nothing found is all zeros, not an error" {
    const ranked = [_][]const u8{ "a.md", "b.md" };
    const expected = [_][]const u8{"docs/DESIGN.md"};
    const m = bench.score(&ranked, &expected, 10);
    try testing.expectEqual(@as(f64, 0), m.recall_at_k);
    try testing.expectEqual(@as(f64, 0), m.mrr);
}

test "score: k truncates the ranking" {
    // A hit at rank 4 is not a hit when the caller asked for 3.
    const ranked = [_][]const u8{ "a.md", "b.md", "c.md", "docs/DESIGN.md" };
    const expected = [_][]const u8{"docs/DESIGN.md"};
    const m3 = bench.score(&ranked, &expected, 3);
    try testing.expectEqual(@as(f64, 0), m3.recall_at_k);
    try testing.expectEqual(@as(f64, 0), m3.mrr);
    const m10 = bench.score(&ranked, &expected, 10);
    try testing.expectEqual(@as(f64, 1), m10.recall_at_k);
}

test "score: empty expectation scores zero rather than dividing by zero" {
    const ranked = [_][]const u8{"a.md"};
    const m = bench.score(&ranked, &.{}, 10);
    try testing.expectEqual(@as(usize, 0), m.expected);
    try testing.expectEqual(@as(f64, 0), m.recall_at_k);
}

test "misses names what never came back" {
    const gpa = testing.allocator;
    const ranked = [_][]const u8{ "a.md", "docs/DESIGN.md" };
    const expected = [_][]const u8{ "docs/DESIGN.md", "docs/recipes.md" };
    const m = try bench.misses(gpa, &ranked, &expected, 10);
    defer gpa.free(m);
    try testing.expectEqual(@as(usize, 1), m.len);
    try testing.expectEqualStrings("docs/recipes.md", m[0]);
}

// ---------------------------------------------------------------------------
// fixture parsing
// ---------------------------------------------------------------------------

test "parseFixture: minimal columns" {
    const gpa = testing.allocator;
    var f = try bench.parseFixture(gpa,
        \\query,expected
        \\rrf fusion,docs/DESIGN.md
        \\
    );
    defer f.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), f.cases.len);
    try testing.expectEqualStrings("rrf fusion", f.cases[0].query);
    try testing.expectEqualStrings("docs/DESIGN.md", f.cases[0].expected[0]);
    // id defaults to the row number so a case can still be pointed at.
    try testing.expectEqualStrings("1", f.cases[0].id);
    try testing.expectEqualStrings("", f.cases[0].kind);
}

test "parseFixture: several expected documents in one cell" {
    const gpa = testing.allocator;
    var f = try bench.parseFixture(gpa,
        \\id,kind,query,expected
        \\fusion,semantic,how is fusion decided,docs/DESIGN.md;docs/recipes.md
        \\
    );
    defer f.deinit(gpa);
    try testing.expectEqualStrings("fusion", f.cases[0].id);
    try testing.expectEqualStrings("semantic", f.cases[0].kind);
    try testing.expectEqual(@as(usize, 2), f.cases[0].expected.len);
    try testing.expectEqualStrings("docs/recipes.md", f.cases[0].expected[1]);
}

test "parseFixture: column order is free and unknown columns are ignored" {
    const gpa = testing.allocator;
    var f = try bench.parseFixture(gpa,
        \\expected,note,query
        \\docs/DESIGN.md,why this case exists,rrf
        \\
    );
    defer f.deinit(gpa);
    try testing.expectEqualStrings("rrf", f.cases[0].query);
    try testing.expectEqualStrings("docs/DESIGN.md", f.cases[0].expected[0]);
}

test "parseFixture: a comma inside a quoted query survives" {
    const gpa = testing.allocator;
    var f = try bench.parseFixture(gpa,
        \\query,expected
        \\"vectors, and bm25",docs/DESIGN.md
        \\
    );
    defer f.deinit(gpa);
    try testing.expectEqualStrings("vectors, and bm25", f.cases[0].query);
}

test "parseFixture: a missing required column is an error, not an empty run" {
    const gpa = testing.allocator;
    try testing.expectError(error.MissingColumn, bench.parseFixture(gpa,
        \\id,query
        \\x,rrf
        \\
    ));
}

test "parseFixture: a row with no query is skipped, a malformed row is reported" {
    const gpa = testing.allocator;
    var f = try bench.parseFixture(gpa,
        \\query,expected
        \\rrf,docs/DESIGN.md
        \\,docs/recipes.md
        \\too,many,fields
        \\
    );
    defer f.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), f.cases.len);
    try testing.expectEqual(@as(usize, 1), f.bad_rows.len);
}

// ---------------------------------------------------------------------------
// accumulator
// ---------------------------------------------------------------------------

test "Accumulator averages per case, not per document" {
    var acc: bench.Accumulator = .{};
    // One case names two documents and finds both; the other names one and
    // finds none. Per case that is 0.5; per document it would be 2/3.
    acc.add(.{
        .recall_at_1 = 1,
        .recall_at_3 = 1,
        .recall_at_5 = 1,
        .recall_at_k = 1,
        .mrr = 1,
        .found = 2,
        .expected = 2,
    });
    acc.add(.zero);
    const m = acc.mean();
    try testing.expectEqual(@as(usize, 2), acc.cases);
    try testing.expectEqual(@as(f64, 0.5), m.recall_at_k);
    try testing.expectEqual(@as(f64, 0.5), m.mrr);
}

test "Accumulator with no cases does not divide by zero" {
    var acc: bench.Accumulator = .{};
    try testing.expectEqual(@as(f64, 0), acc.mean().recall_at_k);
}
