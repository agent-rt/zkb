//! Structural health checks over the knowledge base.
//!
//! Two layers of maintenance exist and must not be confused (SPEC §14.0):
//! the derived index is zkb's problem and is automatic; the *content* is a
//! judgement problem and belongs to the person and their agent. This module is
//! strictly the evidence side of the second: **facts only, no semantic verdicts,
//! and nothing is ever modified.**
//!
//! Every finding carries a path so it can be checked against the source. The
//! point is not to be believed — it is to be verifiable.

const std = @import("std");
const sqlite = @import("db/sqlite.zig");
const store = @import("db/store.zig");
const schema = @import("db/schema.zig");
const maintain_vec = @import("maintain_vec.zig");
const markdown = @import("ingest/markdown.zig");

/// Set the first time link extraction runs.
///
/// "Is the link graph built?" cannot be answered by counting rows: a knowledge
/// base with genuinely no links looks exactly like one that has never been
/// parsed. Getting this wrong in the "not built" direction reports every
/// document as an orphan; getting it wrong the other way hides real findings.
const links_extracted_key = "links_extracted";

pub const Check = enum {
    /// Documents whose last index attempt failed.
    index_failed,
    /// Links whose target does not exist.
    broken_link,
    /// Documents nothing links to.
    orphan,
    /// A document in a directory that has an index.md which does not list it.
    not_in_index,
    /// One tiny chunk: probably a stub rather than a document.
    fragment,
    /// Very many chunks: probably wants splitting.
    oversized,
    /// No frontmatter block at all.
    no_frontmatter,
    /// The same thing written twice in two places with no structural reason to
    /// agree. Vector-based, thresholded — see maintain_vec.zig.
    near_duplicate,
    /// Identical content at two paths.
    duplicate_content,
    /// A chunk connected to nothing else in the corpus.
    ///
    /// **Off by default**, because E7 measured it as not viable on a corpus of
    /// unrelated projects: at any threshold that reports a readable number of
    /// chunks, what it reports is ordinary content about a topic only one
    /// project covers. That is not isolated knowledge, it is a corpus of forty
    /// projects. The check is meaningful where documents are expected to be
    /// topically interlinked, so it stays available via `--check island`.
    island,
    /// Old, and superseded by something newer that says the same thing.
    stale,
    /// A chunk still in the search indexes whose document no longer exists.
    ///
    /// This is the invariant store.zig is written to protect, and until now
    /// nothing detected a violation of it. Measured on a real index: 1198 of 5213
    /// chunks (23%) were residue from an older binary during a bulk move, and
    /// they sat there silently for weeks — the only symptom was `zkb search`
    /// failing on whichever queries happened to rank one of them into the top-k.
    ///
    /// Unlike every other check here, there is no judgment involved: a chunk
    /// whose document is gone carries no information and cannot be re-derived.
    /// It is always garbage, which is why this one has a repair.
    orphan_chunk,

    /// Checks that run by default.
    ///
    /// `no_frontmatter` is deliberately excluded: measured against ~/docs it
    /// fires on 135 of 195 documents (69%). A check that flags most of the corpus
    /// is not a finding, it is a description of the corpus — and it would bury
    /// the handful of findings that do need acting on. Frontmatter is a
    /// convention of specific namespaces, not a global rule.
    /// Still available via `--check no_frontmatter`.
    pub fn default() []const Check {
        return &.{
            .index_failed,      .broken_link, .orphan,         .not_in_index,
            .fragment,          .oversized,   .near_duplicate, .duplicate_content,
            .orphan_chunk,
        };
    }

    pub fn all() []const Check {
        return &.{
            .index_failed,      .broken_link, .orphan,          .not_in_index,
            .fragment,          .oversized,   .island,          .near_duplicate,
            .duplicate_content, .stale,       .no_frontmatter,  .orphan_chunk,
        };
    }

    /// True for the checks that need vectors, which are the expensive ones.
    /// Grouped so a run that wants none of them can skip the whole KNN pass.
    pub fn isVector(self: Check) bool {
        return switch (self) {
            .near_duplicate, .duplicate_content, .island, .stale => true,
            else => false,
        };
    }

    pub fn parse(name: []const u8) ?Check {
        return std.meta.stringToEnum(Check, name);
    }
};

pub const Finding = struct {
    check: Check,
    /// Stable identity across runs, so `--since last` can diff. Built from
    /// content rather than row ids: chunk and link ids change on every re-index,
    /// and a finding that changes identity every run makes every run "all new".
    key: []const u8,
    path: []const u8,
    detail: []const u8,

    pub fn deinit(self: Finding, gpa: std.mem.Allocator) void {
        gpa.free(self.key);
        gpa.free(self.path);
        gpa.free(self.detail);
    }
};

pub const Report = struct {
    findings: []Finding,
    /// True when the link graph is empty, which makes link-based checks
    /// meaningless rather than clean.
    link_graph_empty: bool,

    pub fn deinit(self: *Report, gpa: std.mem.Allocator) void {
        for (self.findings) |f| f.deinit(gpa);
        gpa.free(self.findings);
        self.* = undefined;
    }

    pub fn count(self: *const Report, check: Check) usize {
        var n: usize = 0;
        for (self.findings) |f| if (f.check == check) {
            n += 1;
        };
        return n;
    }
};

pub const Options = struct {
    checks: []const Check = Check.default(),
    vec: maintain_vec.Config = .{},
    /// Wall clock, for the stale check. Zero means "skip the age comparison",
    /// which is what a caller with no clock should get rather than 1970.
    now_ms: i64 = 0,
    /// A document with one chunk under this many tokens is a fragment.
    fragment_max_tokens: i64 = 100,
    /// More chunks than this suggests the document should be split.
    oversized_chunks: i64 = 50,
};

/// Filenames that are indexes or conventions rather than content, and so are not
/// expected to be linked *to*.
const not_expected_inbound = [_][]const u8{
    "index.md", "README.md", "readme.md", "CLAUDE.md", "AGENTS.md", "SKILL.md",
};

pub fn run(gpa: std.mem.Allocator, db: *sqlite.Db, opts: Options) !Report {
    var findings: std.ArrayList(Finding) = .empty;
    errdefer {
        for (findings.items) |f| f.deinit(gpa);
        findings.deinit(gpa);
    }

    var mbuf: [16]u8 = undefined;
    // Either signal is sufficient, and together they cover both directions:
    // existing links prove extraction ran (self-healing for an index written by
    // an older build), and the flag covers a corpus that genuinely has none.
    const graph_built = (try schema.getMeta(db, links_extracted_key, &mbuf)) != null or
        ((try db.queryI64("SELECT count(*) FROM links")) orelse 0) > 0;

    for (opts.checks) |c| switch (c) {
        .index_failed => try checkIndexFailed(gpa, db, &findings),
        .broken_link => if (graph_built) try checkBrokenLinks(gpa, db, &findings),
        .orphan => if (graph_built) try checkOrphans(gpa, db, &findings),
        .not_in_index => if (graph_built) try checkNotInIndex(gpa, db, &findings),
        .fragment => try checkFragment(gpa, db, &findings, opts.fragment_max_tokens),
        .oversized => try checkOversized(gpa, db, &findings, opts.oversized_chunks),
        .no_frontmatter => try checkNoFrontmatter(gpa, db, &findings),
        .orphan_chunk => try checkOrphanChunks(gpa, db, &findings),
        // Handled together below: all four read the same KNN pass.
        .near_duplicate, .duplicate_content, .island, .stale => {},
    };

    var wants_vector = false;
    for (opts.checks) |c| {
        if (c.isVector()) wants_vector = true;
    }
    if (wants_vector) try runVectorChecks(gpa, db, &findings, opts);

    return .{
        .findings = try findings.toOwnedSlice(gpa),
        .link_graph_empty = !graph_built,
    };
}

fn add(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(Finding),
    check: Check,
    key: []const u8,
    path: []const u8,
    detail: []const u8,
) !void {
    try out.append(gpa, .{
        .check = check,
        .key = try gpa.dupe(u8, key),
        .path = try gpa.dupe(u8, path),
        .detail = try gpa.dupe(u8, detail),
    });
}

/// One KNN pass feeds all four vector checks, because they are all questions
/// about the same neighbour list.
fn runVectorChecks(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    out: *std.ArrayList(Finding),
    opts: Options,
) !void {
    var wanted: [4]bool = @splat(false);
    for (opts.checks) |c| switch (c) {
        .near_duplicate => wanted[0] = true,
        .duplicate_content => wanted[1] = true,
        .island => wanted[2] = true,
        .stale => wanted[3] = true,
        else => {},
    };

    var result = try maintain_vec.run(gpa, db, opts.vec);
    defer result.deinit(gpa);

    var kbuf: [1024]u8 = undefined;
    var dbuf: [1024]u8 = undefined;

    for (result.pairs) |p| {
        const check: Check = switch (p.kind) {
            .duplicate_content => .duplicate_content,
            .near_duplicate => .near_duplicate,
            // Expected overlap is computed and then deliberately not reported:
            // on a corpus built as a document matrix it is the majority of what
            // the threshold finds, and reporting it is what makes the whole
            // check read as noise (SPEC §14.5).
            .expected_overlap => continue,
        };
        if (check == .near_duplicate and !wanted[0]) continue;
        if (check == .duplicate_content and !wanted[1]) continue;

        // Keyed on the path pair, not on chunk ids: chunk ids change on every
        // re-index, and a finding that changes identity every run makes every
        // run look entirely new.
        const key = try std.fmt.bufPrint(&kbuf, "{t}:{s}|{s}", .{ check, p.a_path, p.b_path });
        const detail = try std.fmt.bufPrint(&dbuf, "cos {d:.3} with {s}{s}{s}", .{
            p.cos,
            p.b_path,
            if (p.b_heading.len != 0) " > " else "",
            p.b_heading,
        });
        try add(gpa, out, check, key, p.a_path, detail);
    }

    if (wanted[2]) {
        for (result.islands) |i| {
            const key = try std.fmt.bufPrint(&kbuf, "island:{s}|{s}", .{ i.path, i.heading });
            const detail = try std.fmt.bufPrint(&dbuf, "nearest is {s} at cos {d:.3}", .{
                if (i.nearest_path.len != 0) i.nearest_path else "(nothing)",
                i.cos,
            });
            try add(gpa, out, .island, key, i.path, detail);
        }
    }

    if (wanted[3] and opts.now_ms != 0) {
        const stale = try maintain_vec.staleCandidates(gpa, result.pairs, opts.now_ms, opts.vec);
        defer {
            for (stale) |s| s.deinit(gpa);
            gpa.free(stale);
        }
        for (stale) |s| {
            const key = try std.fmt.bufPrint(&kbuf, "stale:{s}", .{s.old_path});
            const days = @divTrunc(opts.now_ms - s.old_mtime_ms, std.time.ms_per_day);
            const detail = try std.fmt.bufPrint(&dbuf, "{d} days old; {s} says the same at cos {d:.3}", .{
                days, s.newer_path, s.cos,
            });
            try add(gpa, out, .stale, key, s.old_path, detail);
        }
    }
}

fn checkIndexFailed(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare("SELECT rel_path, index_error FROM docs WHERE index_error IS NOT NULL ORDER BY rel_path");
    defer st.finalize();
    var kbuf: [512]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        const key = try std.fmt.bufPrint(&kbuf, "index_failed:{s}", .{path});
        try add(gpa, out, .index_failed, key, path, st.columnText(1));
    }
}

fn checkBrokenLinks(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path, l.raw, l.kind FROM links l
        \\JOIN docs d ON d.id = l.doc_id
        \\WHERE l.target_doc_id IS NULL AND l.kind NOT IN ('external', 'asset')
        \\ORDER BY d.rel_path, l.raw
    );
    defer st.finalize();
    var kbuf: [1024]u8 = undefined;
    var dbuf: [512]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        const raw = st.columnText(1);
        const key = try std.fmt.bufPrint(&kbuf, "broken_link:{s}:{s}", .{ path, raw });
        const detail = try std.fmt.bufPrint(&dbuf, "{s} -> {s}", .{ st.columnText(2), raw });
        try add(gpa, out, .broken_link, key, path, detail);
    }
}

/// Only for `documents` collections.
///
/// "Nothing links here" is a finding about a corpus that is supposed to be
/// interlinked. It is not a finding about a memory or a records row: `remember`
/// writes one file per memory and no index ever points at them, so every memory
/// zkb has ever written is an orphan by construction. Measured on this index, 35
/// of 69 orphans were memories and 1 was `facts.csv` — more than half the check's
/// output was it disagreeing with how the other half of zkb works.
///
/// The kind is the right discriminator rather than a path prefix: it is what
/// already says who writes a collection and how it is parsed.
fn checkOrphans(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path FROM docs d
        \\JOIN collections c ON c.id = d.collection_id
        \\WHERE c.kind = 'documents'
        \\  AND NOT EXISTS (SELECT 1 FROM links l WHERE l.target_doc_id = d.id)
        \\ORDER BY d.rel_path
    );
    defer st.finalize();
    var kbuf: [512]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        if (isConventionFile(path)) continue;
        const key = try std.fmt.bufPrint(&kbuf, "orphan:{s}", .{path});
        try add(gpa, out, .orphan, key, path, "nothing links here");
    }
}

/// One finding for the whole condition, not one per chunk.
///
/// Per-chunk would be 1198 findings on the index that motivated this, burying
/// every other check — and there would be nothing to act on per row anyway,
/// since the document that would give a row a path is exactly what is missing.
/// The count goes in the key rather than only the detail, so `--since last` treats
/// a growing leak as news instead of as the same finding.
fn checkOrphanChunks(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT count(*), count(DISTINCT c.doc_id) FROM chunks c
        \\LEFT JOIN docs d ON d.id = c.doc_id
        \\WHERE d.id IS NULL
    );
    defer st.finalize();
    if (!try st.step()) return;
    const chunks = st.columnI64(0);
    if (chunks == 0) return;
    const docs = st.columnI64(1);

    var kbuf: [128]u8 = undefined;
    var dbuf: [256]u8 = undefined;
    const key = try std.fmt.bufPrint(&kbuf, "orphan_chunk:{d}", .{chunks});
    const detail = try std.fmt.bufPrint(
        &dbuf,
        "{d} chunks from {d} deleted document(s) are still searchable; run: zkb index",
        .{ chunks, docs },
    );
    try add(gpa, out, .orphan_chunk, key, "(index)", detail);
}

/// A document sitting next to an index.md that does not mention it. This encodes
/// an actual convention of this knowledge base (one index.md per project), not a
/// universal rule — which is why it is a separate check that can be turned off.
fn checkNotInIndex(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path, idx.rel_path FROM docs d
        \\JOIN docs idx
        \\  ON idx.collection_id = d.collection_id
        \\ AND idx.rel_path = CASE
        \\       WHEN instr(d.rel_path, '/') = 0 THEN 'index.md'
        \\       ELSE rtrim(d.rel_path, replace(d.rel_path, '/', '')) || 'index.md'
        \\     END
        \\WHERE d.id != idx.id
        \\  AND NOT EXISTS (
        \\    SELECT 1 FROM links l WHERE l.doc_id = idx.id AND l.target_doc_id = d.id
        \\  )
        \\ORDER BY d.rel_path
    );
    defer st.finalize();
    var kbuf: [512]u8 = undefined;
    var dbuf: [512]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        if (isConventionFile(path)) continue;
        const key = try std.fmt.bufPrint(&kbuf, "not_in_index:{s}", .{path});
        const detail = try std.fmt.bufPrint(&dbuf, "not listed in {s}", .{st.columnText(1)});
        try add(gpa, out, .not_in_index, key, path, detail);
    }
}

fn checkFragment(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    out: *std.ArrayList(Finding),
    max_tokens: i64,
) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path, c.n_tokens FROM docs d
        \\JOIN chunks c ON c.doc_id = d.id
        \\WHERE d.chunk_count = 1 AND c.n_tokens < ?1
        \\ORDER BY d.rel_path
    );
    defer st.finalize();
    try st.bindI64(1, max_tokens);
    var kbuf: [512]u8 = undefined;
    var dbuf: [128]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        const key = try std.fmt.bufPrint(&kbuf, "fragment:{s}", .{path});
        const detail = try std.fmt.bufPrint(&dbuf, "single chunk, {d} tokens", .{st.columnI64(1)});
        try add(gpa, out, .fragment, key, path, detail);
    }
}

fn checkOversized(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    out: *std.ArrayList(Finding),
    max_chunks: i64,
) !void {
    var st = try db.prepare(
        "SELECT rel_path, chunk_count FROM docs WHERE chunk_count > ?1 ORDER BY chunk_count DESC",
    );
    defer st.finalize();
    try st.bindI64(1, max_chunks);
    var kbuf: [512]u8 = undefined;
    var dbuf: [128]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        const key = try std.fmt.bufPrint(&kbuf, "oversized:{s}", .{path});
        const detail = try std.fmt.bufPrint(&dbuf, "{d} chunks", .{st.columnI64(1)});
        try add(gpa, out, .oversized, key, path, detail);
    }
}

fn checkNoFrontmatter(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        "SELECT rel_path FROM docs WHERE frontmatter IS NULL AND indexed_at IS NOT NULL ORDER BY rel_path",
    );
    defer st.finalize();
    var kbuf: [512]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        const key = try std.fmt.bufPrint(&kbuf, "no_frontmatter:{s}", .{path});
        // Only presence, never compliance: which fields belong in frontmatter is
        // a convention of specific namespaces, not something zkb gets to judge.
        try add(gpa, out, .no_frontmatter, key, path, "no frontmatter block");
    }
}

/// Schemes that point outside the knowledge base. Everything else is treated as
/// a collection-absolute path.
fn isExternalScheme(scheme: []const u8) bool {
    inline for (.{ "http", "https", "file", "ftp", "mailto", "data" }) |ext| {
        if (std.ascii.eqlIgnoreCase(scheme, ext)) return true;
    }
    return false;
}

fn isConventionFile(path: []const u8) bool {
    const base = std.fs.path.basename(path);
    for (not_expected_inbound) |n| if (std.mem.eql(u8, base, n)) return true;
    return false;
}

// ---------------------------------------------------------------------------
// history: making a report comparable to the previous one
// ---------------------------------------------------------------------------

pub const Diff = struct {
    new_keys: [][]const u8,
    resolved_keys: [][]const u8,
    unchanged: usize,

    pub fn deinit(self: *Diff, gpa: std.mem.Allocator) void {
        for (self.new_keys) |k| gpa.free(k);
        gpa.free(self.new_keys);
        for (self.resolved_keys) |k| gpa.free(k);
        gpa.free(self.resolved_keys);
        self.* = undefined;
    }
};

/// Compare against the most recent stored run.
///
/// Without this a report is 200 lines every time and stops being read after the
/// second look; what makes it usable is showing only what changed (SPEC §14.6).
pub fn diffAgainstLast(gpa: std.mem.Allocator, db: *sqlite.Db, report: *const Report) !Diff {
    var previous: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = previous.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        previous.deinit(gpa);
    }

    {
        var st = try db.prepare("SELECT report FROM maintenance_runs ORDER BY id DESC LIMIT 1");
        defer st.finalize();
        if (try st.step()) {
            const json = st.columnText(0);
            const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch
                return emptyDiff(gpa, report);
            defer parsed.deinit();
            if (parsed.value == .object) {
                if (parsed.value.object.get("keys")) |ks| if (ks == .array) {
                    for (ks.array.items) |k| if (k == .string) {
                        try previous.put(gpa, try gpa.dupe(u8, k.string), {});
                    };
                };
            }
        }
    }

    var new_keys: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (new_keys.items) |k| gpa.free(k);
        new_keys.deinit(gpa);
    }
    var unchanged: usize = 0;
    var current: std.StringHashMapUnmanaged(void) = .empty;
    defer current.deinit(gpa);

    for (report.findings) |f| {
        try current.put(gpa, f.key, {});
        if (previous.contains(f.key)) unchanged += 1 else try new_keys.append(gpa, try gpa.dupe(u8, f.key));
    }

    var resolved: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (resolved.items) |k| gpa.free(k);
        resolved.deinit(gpa);
    }
    var it = previous.keyIterator();
    while (it.next()) |k| {
        if (!current.contains(k.*)) try resolved.append(gpa, try gpa.dupe(u8, k.*));
    }

    return .{
        .new_keys = try new_keys.toOwnedSlice(gpa),
        .resolved_keys = try resolved.toOwnedSlice(gpa),
        .unchanged = unchanged,
    };
}

fn emptyDiff(gpa: std.mem.Allocator, report: *const Report) !Diff {
    var new_keys = try gpa.alloc([]const u8, report.findings.len);
    for (report.findings, 0..) |f, i| new_keys[i] = try gpa.dupe(u8, f.key);
    return .{ .new_keys = new_keys, .resolved_keys = &.{}, .unchanged = 0 };
}

/// Persist the finding keys so the next run can diff against them.
/// `checks` is recorded, not assumed: `--since last` diffs a run against the
/// previous one, and comparing a single-check run against a full one would
/// report every finding the narrow run did not look for as newly resolved.
pub fn record(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    report: *const Report,
    checks: []const Check,
    now_ms: i64,
) !void {
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.appendSlice(gpa, "{\"keys\":[");
    for (report.findings, 0..) |f, i| {
        if (i != 0) try json.appendSlice(gpa, ",");
        var buf: [1024]u8 = undefined;
        var w = std.Io.Writer.fixed(&buf);
        try std.json.Stringify.value(f.key, .{}, &w);
        try json.appendSlice(gpa, w.buffered());
    }
    try json.appendSlice(gpa, "]}");

    var st = try db.prepare(
        "INSERT INTO maintenance_runs(started_at, checks, report) VALUES (?1, ?2, ?3)",
    );
    defer st.finalize();
    var names: std.ArrayList(u8) = .empty;
    defer names.deinit(gpa);
    for (checks, 0..) |c, i| {
        if (i != 0) try names.append(gpa, ',');
        try names.appendSlice(gpa, @tagName(c));
    }

    try st.bindI64(1, now_ms);
    try st.bindText(2, names.items);
    try st.bindText(3, json.items);
    _ = try st.step();
}

// ---------------------------------------------------------------------------
// link resolution
// ---------------------------------------------------------------------------

/// Resolve every unresolved link to a document id.
///
/// **Runs after the whole scan, never during it.** A document may link to one
/// that has not been indexed yet; resolving inline would mark those as broken and
/// the result would depend on filesystem iteration order — a bug that only shows
/// up when documents are added, and looks random when it does.
pub fn resolveLinks(gpa: std.mem.Allocator, db: *sqlite.Db) !usize {
    var resolved: usize = 0;

    var st = try db.prepare(
        \\SELECT l.id, l.raw, l.kind, d.rel_path, d.collection_id
        \\FROM links l JOIN docs d ON d.id = l.doc_id
        \\WHERE l.target_doc_id IS NULL AND l.kind NOT IN ('external', 'asset')
    );
    defer st.finalize();

    const Pending = struct { id: i64, target: []u8, collection_id: i64 };
    var pending: std.ArrayList(Pending) = .empty;
    defer {
        for (pending.items) |p| gpa.free(p.target);
        pending.deinit(gpa);
    }

    while (try st.step()) {
        const raw = st.columnText(1);
        const kind = st.columnText(2);
        const from = st.columnText(3);
        const target = try normalizeTarget(gpa, raw, from, std.mem.eql(u8, kind, "wiki"));
        if (target) |t| {
            try pending.append(gpa, .{
                .id = st.columnI64(0),
                .target = t,
                .collection_id = st.columnI64(4),
            });
        }
    }

    for (pending.items) |p| {
        var find = try db.prepare(
            \\SELECT id FROM docs
            \\WHERE collection_id = ?1 AND (rel_path = ?2 OR rel_path LIKE '%/' || ?2)
            \\ORDER BY length(rel_path) LIMIT 1
        );
        defer find.finalize();
        try find.bindI64(1, p.collection_id);
        try find.bindText(2, p.target);
        if (try find.step()) {
            const target_id = find.columnI64(0);
            var upd = try db.prepare("UPDATE links SET target_doc_id = ?2 WHERE id = ?1");
            defer upd.finalize();
            try upd.bindI64(1, p.id);
            try upd.bindI64(2, target_id);
            _ = try upd.step();
            resolved += 1;
        }
    }
    return resolved;
}

/// Turn a raw link into a collection-relative path, or null when it cannot be
/// one (anchors, mail, absolute URLs).
fn normalizeTarget(
    gpa: std.mem.Allocator,
    raw: []const u8,
    from: []const u8,
    /// Wikilinks name a document globally by stem, by convention. Resolving one
    /// relative to the linking document's directory finds the wrong file, or
    /// nothing — `[[SKILL]]` in projects/x/index.md means skills/y/SKILL.md, not
    /// projects/x/SKILL.md.
    is_wiki: bool,
) !?[]u8 {
    var t = raw;
    // A custom URI scheme is rooted at the collection, not relative to the
    // document that mentions it: `zkb://projects/x/REQ.md` means exactly that
    // path. Knowledge-base tools commonly define one, and resolving it as a
    // relative path is the single largest source of false "broken link"
    // findings — measured at 288 of 348 on one corpus.
    //
    // Generic on purpose: any scheme that is not a network or filesystem URL is
    // treated this way, so no tool's name is hardcoded here.
    const scheme_end = std.mem.indexOf(u8, t, "://");
    const is_collection_uri = if (scheme_end) |e| !isExternalScheme(t[0..e]) else false;
    if (is_collection_uri) t = t[scheme_end.? + 3 ..];
    if (t.len == 0) return null;
    // A pure anchor points inside the same document; not a link between files.
    if (t[0] == '#') return null;
    // Drop any fragment.
    if (std.mem.indexOfScalar(u8, t, '#')) |h| t = t[0..h];
    if (t.len == 0) return null;
    // Anything still carrying a scheme is external: http(s), file, mailto.
    if (std.mem.indexOf(u8, t, "://") != null) return null;

    // A wikilink without an extension names a document by stem.
    const needs_ext = std.mem.indexOfScalar(u8, t, '.') == null;

    if (is_collection_uri or t[0] == '/') {
        const abs = std.mem.trimStart(u8, t, "/");
        return if (needs_ext)
            try std.fmt.allocPrint(gpa, "{s}.md", .{abs})
        else
            try gpa.dupe(u8, abs);
    }

    // Relative to the linking document's directory — except wikilinks, which are
    // global names and are matched by basename downstream.
    const dir = if (is_wiki) "" else (std.fs.path.dirname(from) orelse "");
    var joined: std.ArrayList(u8) = .empty;
    errdefer joined.deinit(gpa);
    if (dir.len != 0 and !std.mem.startsWith(u8, t, "/")) {
        try joined.appendSlice(gpa, dir);
        try joined.append(gpa, '/');
    }
    try joined.appendSlice(gpa, t);
    if (needs_ext) try joined.appendSlice(gpa, ".md");

    const normalized = try normalizeDots(gpa, joined.items);
    joined.deinit(gpa);
    return normalized;
}

/// Collapse `.` and `..` segments. Done here rather than with a path API because
/// these are collection-relative virtual paths, not filesystem paths.
fn normalizeDots(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    defer parts.deinit(gpa);
    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |seg| {
        if (seg.len == 0 or std.mem.eql(u8, seg, ".")) continue;
        if (std.mem.eql(u8, seg, "..")) {
            if (parts.items.len != 0) _ = parts.pop();
            continue;
        }
        try parts.append(gpa, seg);
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (parts.items, 0..) |seg, i| {
        if (i != 0) try out.append(gpa, '/');
        try out.appendSlice(gpa, seg);
    }
    return out.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------------------
// storage
// ---------------------------------------------------------------------------

/// Replace a document's links. Called inside the same transaction that replaces
/// its chunks, so the graph never disagrees with the content it came from.
pub fn replaceLinks(
    db: *sqlite.Db,
    doc_id: i64,
    links: []const markdown.Link,
) !void {
    {
        var st = try db.prepare("DELETE FROM links WHERE doc_id = ?1");
        defer st.finalize();
        try st.bindI64(1, doc_id);
        _ = try st.step();
    }
    // Records that extraction has run at all, including when a document has no
    // links — which is exactly the case a row count cannot distinguish.
    try schema.setMeta(db, links_extracted_key, "1");

    var st = try db.prepare(
        "INSERT INTO links(doc_id, chunk_id, kind, raw, target_doc_id) VALUES (?1, NULL, ?2, ?3, NULL)",
    );
    defer st.finalize();
    for (links) |l| {
        st.reset();
        try st.bindI64(1, doc_id);
        try st.bindText(2, @tagName(l.kind));
        try st.bindText(3, l.raw);
        _ = try st.step();
    }
}

pub const store_unused = store;
