//! The vector-based maintenance checks: near-duplicate, island, stale.
//!
//! Separate from the structural checks because they are a different kind of
//! thing. Those are exact — a link either resolves or it does not — and have no
//! thresholds and no false positives. These are judgements over a continuous
//! similarity, and every one of them needs a number that has to be earned by
//! measurement (E7, docs/experiments/E7-maintenance-thresholds.md).
//!
//! They use only vectors that are already stored, so there is no inference and
//! no model load: pure CPU, safe to run in the background.
//!
//! **The suppression rules matter more than the thresholds.** `~/docs` is
//! written as a four-layer document matrix (REQ / SPEC / PLAN / TECH-DESIGN),
//! where one project's REQ and SPEC are *supposed* to share nouns, constraints
//! and whole tables. On that corpus a raw near-duplicate report is mostly
//! restatements of the method, not findings. Raising the threshold until they
//! disappear also removes the real duplicates, so the answer is to classify the
//! structurally-explainable overlap separately rather than to hide it
//! (SPEC §14.5).

const std = @import("std");
const sqlite = @import("db/sqlite.zig");
const utf8 = @import("util/utf8.zig");

pub const Config = struct {
    /// Cosine above which two cross-document chunks are "the same thing said
    /// twice".
    ///
    /// 0.84 is where E7 measured the false-positive rate crossing 20% on ~/docs:
    /// 7 pairs reported, 6 of them real. SPEC §14.5 guessed 0.92, which would
    /// have reported one pair and missed five true duplicates — but the more
    /// important finding is that the threshold is not the lever. See
    /// `classify` and docs/experiments/E7-maintenance-thresholds.md.
    dup_threshold: f64 = 0.84,
    /// A chunk whose best cross-document neighbour is below this is connected to
    /// nothing else in the corpus.
    ///
    /// Measured as **not viable** on ~/docs and off by default — see `Check`
    /// in maintain.zig. Kept because the number is meaningful on a corpus that
    /// is one topic rather than forty projects.
    island_threshold: f64 = 0.50,
    /// Only a document older than this can be a stale candidate.
    stale_days: i64 = 120,
    /// Above this many chunks the all-pairs comparison is skipped rather than
    /// run: it is O(n²) in time and holds every vector in memory (4 KB each), so
    /// somewhere it stops being the right algorithm. A personal knowledge base
    /// is three orders of magnitude below this.
    max_chunks: usize = 50_000,
    /// Ceiling on reported pairs, so one pathological corpus cannot produce a
    /// report nobody can read. Truncation is always reported, never silent.
    max_pairs: usize = 200,
};

pub const PairKind = enum {
    /// Two documents with identical content at two paths. Not a near-duplicate
    /// at all — a different problem with a different fix (SPEC §14.4).
    duplicate_content,
    /// Overlap between files of one project's document matrix. Expected by
    /// construction; listed, but not a finding.
    expected_overlap,
    /// Genuine near-duplication: the same thing written twice in two places
    /// that have no structural reason to agree.
    near_duplicate,
};

pub const Pair = struct {
    kind: PairKind,
    cos: f64,
    a_path: []const u8,
    a_heading: []const u8,
    a_excerpt: []const u8,
    a_mtime_ms: i64,
    b_path: []const u8,
    b_heading: []const u8,
    b_excerpt: []const u8,
    b_mtime_ms: i64,

    pub fn deinit(self: Pair, gpa: std.mem.Allocator) void {
        gpa.free(self.a_path);
        gpa.free(self.a_heading);
        gpa.free(self.a_excerpt);
        gpa.free(self.b_path);
        gpa.free(self.b_heading);
        gpa.free(self.b_excerpt);
    }
};

pub const Island = struct {
    path: []const u8,
    heading: []const u8,
    excerpt: []const u8,
    /// Best cross-document neighbour, and how close it is.
    nearest_path: []const u8,
    cos: f64,

    pub fn deinit(self: Island, gpa: std.mem.Allocator) void {
        gpa.free(self.path);
        gpa.free(self.heading);
        gpa.free(self.excerpt);
        gpa.free(self.nearest_path);
    }
};

pub const Result = struct {
    pairs: []Pair,
    islands: []Island,
    /// Chunks examined, so a report over an empty index is distinguishable from
    /// a clean one.
    chunks_scanned: usize,
    /// True when `max_pairs` cut the list short.
    truncated: bool,

    pub fn deinit(self: *Result, gpa: std.mem.Allocator) void {
        for (self.pairs) |p| p.deinit(gpa);
        gpa.free(self.pairs);
        for (self.islands) |i| i.deinit(gpa);
        gpa.free(self.islands);
        self.* = undefined;
    }
};

/// Fraction of a chunk that is markdown link syntax, above which it is a table
/// of contents rather than prose.
///
/// E7 measured this as the single biggest false-positive class: 41 of 170
/// candidate pairs were two project `index.md` files whose "关联知识" sections
/// resemble each other because they are both lists of links. Two tables of
/// contents looking alike is not a knowledge problem, and no threshold
/// separates them — the highest-scoring pair in the whole pool (0.973) was one.
const nav_link_density: f64 = 0.30;

fn linkDensity(text: []const u8) f64 {
    if (text.len == 0) return 0;
    var link_bytes: usize = 0;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] != '[') continue;
        const close = std.mem.indexOfScalarPos(u8, text, i, ']') orelse break;
        if (close + 1 >= text.len or text[close + 1] != '(') continue;
        const end = std.mem.indexOfScalarPos(u8, text, close, ')') orelse break;
        link_bytes += end - i + 1;
        i = end;
    }
    return @as(f64, @floatFromInt(link_bytes)) / @as(f64, @floatFromInt(text.len));
}

fn isNavigation(text: []const u8) bool {
    return linkDensity(text) >= nav_link_density;
}

// The classification rules are what E7 measured as load-bearing, so they are
// reachable from tests without exposing the internals they run on.
pub const isNavigationForTest = isNavigation;

/// How many leading path segments two files share, ignoring their basenames.
fn sharedDepth(a: []const u8, b: []const u8) usize {
    var ia = std.mem.splitScalar(u8, std.fs.path.dirname(a) orelse "", '/');
    var ib = std.mem.splitScalar(u8, std.fs.path.dirname(b) orelse "", '/');
    var n: usize = 0;
    while (ia.next()) |x| {
        const y = ib.next() orelse break;
        if (!std.mem.eql(u8, x, y)) break;
        n += 1;
    }
    return n;
}

/// Two files of one project. `projects/<name>/...` shares two segments, which
/// also covers `projects/<name>/docs/...` against `projects/<name>/README.md`.
fn sameProject(a: []const u8, b: []const u8) bool {
    return sharedDepth(a, b) >= 2;
}

pub const sameProjectForTest = sameProject;

/// The section title, with its number stripped.
///
/// `REQ.md > 11. 验收标准` and `SPEC.md > 14. 验收标准` are the same section of one
/// project's documentation, numbered differently — which is exactly the shape of
/// a document matrix restating itself.
fn sectionTitle(heading_path: []const u8) []const u8 {
    var tail = heading_path;
    if (std.mem.lastIndexOf(u8, heading_path, ">")) |gt| tail = heading_path[gt + 1 ..];
    tail = std.mem.trim(u8, tail, " \t");
    var i: usize = 0;
    while (i < tail.len and (std.ascii.isDigit(tail[i]) or tail[i] == '.')) i += 1;
    return std.mem.trimStart(u8, tail[i..], " \t");
}

pub const sectionTitleForTest = sectionTitle;

// ---------------------------------------------------------------------------

const ChunkRow = struct {
    id: i64,
    doc_id: i64,
    collection_id: i64,
    /// A table of contents rather than prose. Excluded from the pair check.
    nav: bool,
};

/// Compare every chunk against every other, and classify what comes back.
///
/// **All pairs, not a k-nearest lookup.** SPEC §14.4 specified `KNN k=6` per
/// chunk; measured on ~/docs that took 30 s (vec0 rescans the whole partition
/// per call), and worse, it is not correct for this question. Chunks overlap
/// their neighbours by 80 tokens and share a heading-path prefix in the
/// embedding input, so a chunk's six nearest are routinely its own document's —
/// which pushes the cross-document duplicate, the only kind this check reports,
/// off the end of the list entirely. Raising k would paper over it; comparing
/// everything removes the parameter.
///
/// The vectors are L2-normalized at write time, so cosine is a dot product, and
/// the whole thing is one SIMD reduction per pair.
///
/// One pass produces both checks: the nearest cross-document neighbour is the
/// input to the island test, and the ones above `dup_threshold` are the pairs.
pub fn run(gpa: std.mem.Allocator, db: *sqlite.Db, cfg: Config) !Result {
    var chunks: std.ArrayList(ChunkRow) = .empty;
    defer chunks.deinit(gpa);
    var vectors: std.ArrayList(f32) = .empty;
    defer vectors.deinit(gpa);

    var dim: usize = 0;
    {
        var st = try db.prepare(
            \\SELECT c.id, c.doc_id, d.collection_id, v.embedding, c.text
            \\FROM chunks c
            \\JOIN docs d ON d.id = c.doc_id
            \\JOIN vec_chunks v ON v.chunk_id = c.id
            \\ORDER BY c.id
        );
        defer st.finalize();
        while (try st.step()) {
            const blob = st.columnBlob(3);
            if (dim == 0) dim = blob.len / @sizeOf(f32);
            // A row whose vector is a different width belongs to another model
            // and is not comparable to the rest.
            if (blob.len != dim * @sizeOf(f32)) continue;
            if (chunks.items.len >= cfg.max_chunks) break;

            try chunks.append(gpa, .{
                .id = st.columnI64(0),
                .doc_id = st.columnI64(1),
                .collection_id = st.columnI64(2),
                .nav = isNavigation(st.columnText(4)),
            });
            const at = vectors.items.len;
            try vectors.resize(gpa, at + dim);
            @memcpy(std.mem.sliceAsBytes(vectors.items[at..]), blob);
        }
    }

    var pairs: std.ArrayList(Pair) = .empty;
    errdefer {
        for (pairs.items) |p| p.deinit(gpa);
        pairs.deinit(gpa);
    }
    var islands: std.ArrayList(Island) = .empty;
    errdefer {
        for (islands.items) |i| i.deinit(gpa);
        islands.deinit(gpa);
    }

    const n = chunks.items.len;
    if (n == 0 or dim == 0) {
        return .{
            .pairs = try pairs.toOwnedSlice(gpa),
            .islands = try islands.toOwnedSlice(gpa),
            .chunks_scanned = 0,
            .truncated = false,
        };
    }

    // Best cross-document neighbour per chunk, for the island check.
    const best_cos = try gpa.alloc(f64, n);
    defer gpa.free(best_cos);
    const best_at = try gpa.alloc(usize, n);
    defer gpa.free(best_at);
    @memset(best_cos, -2);
    @memset(best_at, 0);

    var truncated = false;
    const RawPair = struct { a: usize, b: usize, cos: f64 };
    var raw_pairs: std.ArrayList(RawPair) = .empty;
    defer raw_pairs.deinit(gpa);

    for (0..n) |i| {
        const vi = vectors.items[i * dim ..][0..dim];
        for (i + 1..n) |j| {
            // Adjacent chunks of one document are near-identical by
            // construction, so only cross-document pairs mean anything
            // (SPEC §14.4). Collections are separate spaces of their own.
            if (chunks.items[i].doc_id == chunks.items[j].doc_id) continue;
            if (chunks.items[i].collection_id != chunks.items[j].collection_id) continue;

            const cos = dot(vi, vectors.items[j * dim ..][0..dim]);
            if (cos > best_cos[i]) {
                best_cos[i] = cos;
                best_at[i] = j;
            }
            if (cos > best_cos[j]) {
                best_cos[j] = cos;
                best_at[j] = i;
            }
            if (cos < cfg.dup_threshold) continue;
            // A pair of tables of contents is not a duplicate-knowledge finding,
            // and on this corpus it was the largest single class of them.
            if (chunks.items[i].nav or chunks.items[j].nav) continue;
            try raw_pairs.append(gpa, .{ .a = i, .b = j, .cos = cos });
        }
    }

    const S = struct {
        fn byCosDesc(_: void, x: RawPair, y: RawPair) bool {
            return x.cos > y.cos;
        }
        fn byPairCos(_: void, x: Pair, y: Pair) bool {
            return x.cos > y.cos;
        }
        fn byIslandCos(_: void, x: Island, y: Island) bool {
            return x.cos < y.cos;
        }
    };
    std.mem.sort(RawPair, raw_pairs.items, {}, S.byCosDesc);

    // One document pair, one finding. Five chunk pairs between the same two
    // files is one fact about those files, and listing it five times is how a
    // report stops being read.
    var seen_docs: std.AutoHashMapUnmanaged(u128, void) = .empty;
    defer seen_docs.deinit(gpa);

    for (raw_pairs.items) |rp| {
        const da: u128 = @intCast(@min(chunks.items[rp.a].doc_id, chunks.items[rp.b].doc_id));
        const db_id: u128 = @intCast(@max(chunks.items[rp.a].doc_id, chunks.items[rp.b].doc_id));
        if ((try seen_docs.getOrPut(gpa, (da << 64) | db_id)).found_existing) continue;

        if (pairs.items.len >= cfg.max_pairs) {
            truncated = true;
            break;
        }
        try pairs.append(gpa, try describePair(
            gpa,
            db,
            chunks.items[rp.a].id,
            chunks.items[rp.b].id,
            rp.cos,
        ));
    }

    for (0..n) |i| {
        if (best_cos[i] < -1) continue; // nothing to compare against at all
        if (best_cos[i] >= cfg.island_threshold) continue;
        try islands.append(gpa, try describeIsland(
            gpa,
            db,
            chunks.items[i].id,
            chunks.items[best_at[i]].id,
            best_cos[i],
        ));
    }

    std.mem.sort(Pair, pairs.items, {}, S.byPairCos);
    std.mem.sort(Island, islands.items, {}, S.byIslandCos);

    return .{
        .pairs = try pairs.toOwnedSlice(gpa),
        .islands = try islands.toOwnedSlice(gpa),
        .chunks_scanned = n,
        .truncated = truncated,
    };
}

/// Dot product of two unit vectors, which is their cosine.
fn dot(a: []const f32, b: []const f32) f64 {
    const width = 8;
    const V = @Vector(width, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + width <= a.len) : (i += width) {
        const x: V = a[i..][0..width].*;
        const y: V = b[i..][0..width].*;
        acc += x * y;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

const excerpt_bytes = 180;

fn describePair(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    a_id: i64,
    b_id: i64,
    cos: f64,
) !Pair {
    var st = try db.prepare(
        \\SELECT d.rel_path, COALESCE(c.heading_path, ''), c.text, d.mtime_ms, d.content_sha
        \\FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.id = ?1
    );
    defer st.finalize();

    try st.bindI64(1, a_id);
    if (!try st.step()) return error.SqliteStep;
    const a_path = try gpa.dupe(u8, st.columnText(0));
    errdefer gpa.free(a_path);
    const a_heading = try gpa.dupe(u8, st.columnText(1));
    errdefer gpa.free(a_heading);
    const a_excerpt = try dupeExcerpt(gpa, st.columnText(2));
    errdefer gpa.free(a_excerpt);
    const a_mtime = st.columnI64(3);
    var sha_buf: [80]u8 = undefined;
    const a_sha_text = st.columnText(4);
    const a_sha = sha_buf[0..@min(a_sha_text.len, sha_buf.len)];
    @memcpy(a_sha, a_sha_text[0..a_sha.len]);

    st.reset();
    try st.bindI64(1, b_id);
    if (!try st.step()) return error.SqliteStep;
    const b_path = try gpa.dupe(u8, st.columnText(0));
    errdefer gpa.free(b_path);
    const b_heading = try gpa.dupe(u8, st.columnText(1));
    errdefer gpa.free(b_heading);
    const b_excerpt = try dupeExcerpt(gpa, st.columnText(2));
    errdefer gpa.free(b_excerpt);
    const b_mtime = st.columnI64(3);
    const b_sha = st.columnText(4);

    // Order matters: identical content is its own finding regardless of where
    // the two copies live.
    const kind: PairKind = if (std.mem.eql(u8, a_sha, b_sha))
        .duplicate_content
    else if (std.mem.eql(u8, std.fs.path.basename(a_path), std.fs.path.basename(b_path)) and
        !std.mem.eql(u8, std.fs.path.dirname(a_path) orelse "", std.fs.path.dirname(b_path) orelse ""))
        // The same filename in two projects is a template: every project's
        // index.md, README.md and AGENTS.md has the same skeleton by convention.
        .expected_overlap
    else if (sameProject(a_path, b_path) and
        !std.mem.eql(u8, sectionTitle(a_heading), sectionTitle(b_heading)))
        // Two different sections of one project's docs overlapping is the
        // document matrix doing its job — REQ says what, SPEC says how, and they
        // share nouns by construction. Two copies of the *same* section is not,
        // so that falls through to near_duplicate.
        .expected_overlap
    else
        .near_duplicate;

    return .{
        .kind = kind,
        .cos = cos,
        .a_path = a_path,
        .a_heading = a_heading,
        .a_excerpt = a_excerpt,
        .a_mtime_ms = a_mtime,
        .b_path = b_path,
        .b_heading = b_heading,
        .b_excerpt = b_excerpt,
        .b_mtime_ms = b_mtime,
    };
}

fn describeIsland(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    chunk_id: i64,
    nearest_id: i64,
    cos: f64,
) !Island {
    var st = try db.prepare(
        \\SELECT d.rel_path, COALESCE(c.heading_path, ''), c.text
        \\FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.id = ?1
    );
    defer st.finalize();

    try st.bindI64(1, chunk_id);
    if (!try st.step()) return error.SqliteStep;
    const path = try gpa.dupe(u8, st.columnText(0));
    errdefer gpa.free(path);
    const heading = try gpa.dupe(u8, st.columnText(1));
    errdefer gpa.free(heading);
    const excerpt = try dupeExcerpt(gpa, st.columnText(2));
    errdefer gpa.free(excerpt);

    var nearest: []u8 = try gpa.dupe(u8, "");
    if (nearest_id != 0) {
        st.reset();
        try st.bindI64(1, nearest_id);
        if (try st.step()) {
            gpa.free(nearest);
            nearest = try gpa.dupe(u8, st.columnText(0));
        }
    }

    return .{
        .path = path,
        .heading = heading,
        .excerpt = excerpt,
        .nearest_path = nearest,
        .cos = cos,
    };
}

/// A single-line excerpt, cut on a UTF-8 boundary.
fn dupeExcerpt(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    const end = utf8.truncate(text, excerpt_bytes);

    var out = try gpa.alloc(u8, end);
    var n: usize = 0;
    var last_space = false;
    for (text[0..end]) |ch| {
        const sp = ch == '\n' or ch == '\r' or ch == '\t' or ch == ' ';
        if (sp) {
            if (last_space) continue;
            out[n] = ' ';
            last_space = true;
        } else {
            out[n] = ch;
            last_space = false;
        }
        n += 1;
    }
    return gpa.realloc(out, n);
}

// ---------------------------------------------------------------------------
// stale candidates
// ---------------------------------------------------------------------------

pub const Stale = struct {
    old_path: []const u8,
    old_mtime_ms: i64,
    newer_path: []const u8,
    cos: f64,

    pub fn deinit(self: Stale, gpa: std.mem.Allocator) void {
        gpa.free(self.old_path);
        gpa.free(self.newer_path);
    }
};

/// Documents that are old *and* have a newer near-duplicate.
///
/// Age alone says nothing — a note written once and still correct is not stale.
/// What makes it a candidate is that something newer now says the same thing,
/// which is the shape of knowledge that has moved and left a copy behind.
pub fn staleCandidates(
    gpa: std.mem.Allocator,
    pairs: []const Pair,
    now_ms: i64,
    cfg: Config,
) ![]Stale {
    var out: std.ArrayList(Stale) = .empty;
    errdefer {
        for (out.items) |s| s.deinit(gpa);
        out.deinit(gpa);
    }

    const cutoff = now_ms - cfg.stale_days * std.time.ms_per_day;
    for (pairs) |p| {
        // Only genuine near-duplicates: a document matrix pair is not one
        // superseding the other, and identical content is its own finding.
        if (p.kind != .near_duplicate) continue;

        const older_first = p.a_mtime_ms < p.b_mtime_ms;
        const old_path = if (older_first) p.a_path else p.b_path;
        const old_mtime = if (older_first) p.a_mtime_ms else p.b_mtime_ms;
        const new_path = if (older_first) p.b_path else p.a_path;
        if (old_mtime >= cutoff) continue;

        var already = false;
        for (out.items) |s| {
            if (std.mem.eql(u8, s.old_path, old_path)) already = true;
        }
        if (already) continue;

        try out.append(gpa, .{
            .old_path = try gpa.dupe(u8, old_path),
            .old_mtime_ms = old_mtime,
            .newer_path = try gpa.dupe(u8, new_path),
            .cos = p.cos,
        });
    }
    return out.toOwnedSlice(gpa);
}
