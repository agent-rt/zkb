//! Retrieval quality measurement: a fixture of queries with known-relevant
//! documents, scored against what each retrieval path actually returns.
//!
//! Why this exists: every knob in retrieval — the chunker, the tokenizer, the
//! RRF constants, whether a reranker is worth its weights — is a claim about
//! recall, and until now the only way to check one was to run a query and look.
//! The 0.525 → 0.793 keyword recall in the README was measured once, by hand,
//! and cannot be re-run. A number that cannot be reproduced cannot catch a
//! regression.
//!
//! The fixture is csv, parsed by the same reader as facts and records: the
//! header is the schema, and a person can add a case in a spreadsheet without
//! learning a format. See `parseFixture` for the columns.

const std = @import("std");
const csv = @import("ingest/csv.zig");

pub const Case = struct {
    /// Stable name for the case, so a regression can be pointed at.
    id: []const u8,
    /// Free-form label used only to group the report (`exact`, `semantic`,
    /// `cross-domain`, …). It does not change how anything is searched — it
    /// exists so a drop can be attributed to a kind of query rather than to
    /// "the score went down".
    kind: []const u8,
    query: []const u8,
    /// Documents that should come back, as paths. Matched by `pathMatches`.
    expected: [][]const u8,

    fn deinit(self: *const Case, gpa: std.mem.Allocator) void {
        gpa.free(self.id);
        gpa.free(self.kind);
        gpa.free(self.query);
        for (self.expected) |e| gpa.free(e);
        gpa.free(self.expected);
    }
};

pub const Fixture = struct {
    cases: []Case,
    /// Lines whose field count did not match the header. Reported, never
    /// skipped silently — a fixture that quietly lost a case would make the
    /// next run's numbers incomparable with this one's.
    bad_rows: []usize,

    pub fn deinit(self: *Fixture, gpa: std.mem.Allocator) void {
        for (self.cases) |c| c.deinit(gpa);
        gpa.free(self.cases);
        gpa.free(self.bad_rows);
        self.* = undefined;
    }
};

pub const ParseError = error{
    /// Neither `query` nor `expected` is optional: without both there is
    /// nothing to run or nothing to check.
    MissingColumn,
    OutOfMemory,
    UnterminatedQuote,
};

/// Parse a fixture csv.
///
/// Required columns: `query`, `expected`. Optional: `id` (defaults to the row
/// number), `kind` (defaults to empty). Unknown columns are ignored, so a
/// fixture may carry notes for a human alongside the fields used here.
///
/// `expected` holds one or more paths separated by `;`. Semicolon rather than
/// comma because the cell already lives in a csv, and requiring the author to
/// quote every multi-document row is how a fixture ends up with one silently
/// truncated case.
pub fn parseFixture(gpa: std.mem.Allocator, src: []const u8) ParseError!Fixture {
    var table = try csv.parse(gpa, src);
    defer table.deinit(gpa);

    const q_col = table.columnIndex("query") orelse return error.MissingColumn;
    const e_col = table.columnIndex("expected") orelse return error.MissingColumn;
    const id_col = table.columnIndex("id");
    const kind_col = table.columnIndex("kind");

    var cases: std.ArrayList(Case) = .empty;
    errdefer {
        for (cases.items) |c| c.deinit(gpa);
        cases.deinit(gpa);
    }

    for (table.rows, 0..) |row, i| {
        const query = std.mem.trim(u8, row[q_col], " \t");
        // A row with no query is a blank line someone left behind, not a case.
        if (query.len == 0) continue;

        var expected: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (expected.items) |e| gpa.free(e);
            expected.deinit(gpa);
        }
        var it = std.mem.splitScalar(u8, row[e_col], ';');
        while (it.next()) |part| {
            const p = std.mem.trim(u8, part, " \t");
            if (p.len == 0) continue;
            try expected.append(gpa, try gpa.dupe(u8, p));
        }

        const id = if (id_col) |c| blk: {
            const v = std.mem.trim(u8, row[c], " \t");
            if (v.len != 0) break :blk try gpa.dupe(u8, v);
            break :blk try std.fmt.allocPrint(gpa, "{d}", .{i + 1});
        } else try std.fmt.allocPrint(gpa, "{d}", .{i + 1});
        errdefer gpa.free(id);

        const kind = if (kind_col) |c|
            try gpa.dupe(u8, std.mem.trim(u8, row[c], " \t"))
        else
            try gpa.dupe(u8, "");
        errdefer gpa.free(kind);

        try cases.append(gpa, .{
            .id = id,
            .kind = kind,
            .query = try gpa.dupe(u8, query),
            .expected = try expected.toOwnedSlice(gpa),
        });
    }

    return .{
        .cases = try cases.toOwnedSlice(gpa),
        .bad_rows = try gpa.dupe(usize, table.bad_rows),
    };
}

/// Does a returned document satisfy an expected path?
///
/// True when they are equal, or when `expected` is a suffix of `result` **at a
/// path-component boundary** — so a fixture can write `DESIGN.md` against a
/// result of `docs/DESIGN.md`, or spell out `docs/DESIGN.md` against
/// `notes/docs/DESIGN.md`, without either one having to know the collection's
/// name.
///
/// The boundary is load-bearing. A plain `endsWith` (which is what qmd's bench
/// harness uses, in both directions) makes `expected = "a.md"` match a result of
/// `xa.md` — a false positive that inflates recall silently, and worst on
/// exactly the short filenames a fixture is most likely to use.
pub fn pathMatches(result: []const u8, expected: []const u8) bool {
    if (std.mem.eql(u8, result, expected)) return true;
    if (result.len <= expected.len) return false;
    const tail = result[result.len - expected.len ..];
    if (!std.mem.eql(u8, tail, expected)) return false;
    return result[result.len - expected.len - 1] == '/';
}

pub const Metrics = struct {
    /// Fraction of this case's expected documents found within the first N
    /// returned documents.
    recall_at_1: f64,
    recall_at_3: f64,
    recall_at_5: f64,
    /// …within everything that came back.
    recall_at_k: f64,
    /// Reciprocal rank of the first expected document, 0 when none appeared.
    mrr: f64,
    /// Counts behind `recall_at_k`, kept so a set of cases can be averaged by
    /// document rather than by case when that is the question being asked.
    found: usize,
    expected: usize,

    pub const zero: Metrics = .{
        .recall_at_1 = 0,
        .recall_at_3 = 0,
        .recall_at_5 = 0,
        .recall_at_k = 0,
        .mrr = 0,
        .found = 0,
        .expected = 0,
    };
};

/// Precision and F1 are deliberately absent.
///
/// A fixture case names one to three relevant documents out of a corpus of
/// thousands, so precision@10 is pinned near 0.1 by the shape of the fixture
/// and moves only when recall does — it carries no information recall does not
/// already carry, and printing it invites reading a structural ceiling as a
/// quality problem. (qmd reports a `precision_at_k` whose denominator is
/// `min(k, |expected|)`, which makes it a second copy of recall@k under a name
/// that says otherwise.)
///
/// `ranked` must already be reduced to documents, in rank order, deduplicated.
pub fn score(ranked: []const []const u8, expected: []const []const u8, k: usize) Metrics {
    if (expected.len == 0) return .zero;

    const n: f64 = @floatFromInt(expected.len);
    var m: Metrics = .zero;
    m.expected = expected.len;
    m.recall_at_1 = @as(f64, @floatFromInt(hitsWithin(ranked, expected, 1))) / n;
    m.recall_at_3 = @as(f64, @floatFromInt(hitsWithin(ranked, expected, 3))) / n;
    m.recall_at_5 = @as(f64, @floatFromInt(hitsWithin(ranked, expected, 5))) / n;
    m.found = hitsWithin(ranked, expected, k);
    m.recall_at_k = @as(f64, @floatFromInt(m.found)) / n;

    for (ranked, 0..) |r, i| {
        if (i >= k) break;
        for (expected) |e| {
            if (pathMatches(r, e)) {
                m.mrr = 1.0 / @as(f64, @floatFromInt(i + 1));
                return m;
            }
        }
    }
    return m;
}

fn hitsWithin(ranked: []const []const u8, expected: []const []const u8, n: usize) usize {
    const upto = @min(n, ranked.len);
    var hits: usize = 0;
    for (expected) |e| {
        for (ranked[0..upto]) |r| {
            if (pathMatches(r, e)) {
                hits += 1;
                break;
            }
        }
    }
    return hits;
}

/// Which of `expected` never appeared. The report prints these because a score
/// that dropped tells you nothing about what got lost.
pub fn misses(
    gpa: std.mem.Allocator,
    ranked: []const []const u8,
    expected: []const []const u8,
    k: usize,
) ![][]const u8 {
    const upto = @min(k, ranked.len);
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(gpa);
    for (expected) |e| {
        var hit = false;
        for (ranked[0..upto]) |r| {
            if (pathMatches(r, e)) {
                hit = true;
                break;
            }
        }
        if (!hit) try out.append(gpa, e);
    }
    return out.toOwnedSlice(gpa);
}

/// Running mean over cases. Averaged per case, not per document: every query
/// counts once regardless of how many documents it names, which is what makes
/// two fixtures of different shapes comparable.
pub const Accumulator = struct {
    cases: usize = 0,
    recall_at_1: f64 = 0,
    recall_at_3: f64 = 0,
    recall_at_5: f64 = 0,
    recall_at_k: f64 = 0,
    mrr: f64 = 0,

    pub fn add(self: *Accumulator, m: Metrics) void {
        self.cases += 1;
        self.recall_at_1 += m.recall_at_1;
        self.recall_at_3 += m.recall_at_3;
        self.recall_at_5 += m.recall_at_5;
        self.recall_at_k += m.recall_at_k;
        self.mrr += m.mrr;
    }

    pub fn mean(self: *const Accumulator) Metrics {
        if (self.cases == 0) return .zero;
        const n: f64 = @floatFromInt(self.cases);
        return .{
            .recall_at_1 = self.recall_at_1 / n,
            .recall_at_3 = self.recall_at_3 / n,
            .recall_at_5 = self.recall_at_5 / n,
            .recall_at_k = self.recall_at_k / n,
            .mrr = self.mrr / n,
            .found = 0,
            .expected = 0,
        };
    }
};
