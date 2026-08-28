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
const paths = @import("util/paths.zig");
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
    /// An indexed memory with no `rec_memory` row.
    ///
    /// The mirror image of `orphan_chunk`, and it went undetected the same way.
    /// `recall` ranks memories by recency out of that projection alone
    /// (`memory.recencyRanked`), so a missing row makes the memory invisible to
    /// recall while `status` still counts the document and `search` still finds
    /// its chunks. Every surface that could have contradicted the empty result
    /// was reading a different table.
    ///
    /// Measured: all 40 rows vanished when a collection was re-indexed under the
    /// wrong `kind`, and nothing noticed for two days — `status` said 40
    /// documents, `doctor` passed 10 of 10, and the only symptom was `recall`
    /// saying the store was empty.
    ///
    /// No judgment in this one either: the projection is derived from frontmatter
    /// and either covers the indexed documents or does not.
    unprojected_memory,
    /// A `type: project` memory with no scope.
    ///
    /// Recorded types split cleanly on this machine: `feedback`, `reference`,
    /// `user` and `decision` are lessons and preferences that hold everywhere,
    /// while `project` states a fact about one project. Measured over 39
    /// memories, every scoped one was `project` except two `feedback` labelled
    /// with a tool name — and the two `project` memories left universal were both
    /// about zkb, opening every unrelated session with it.
    ///
    /// So this is not a style rule: an unscoped `project` memory is a label its
    /// author forgot, and the cost lands on every other project. `zkb rescope`
    /// is the repair.
    ///
    /// Not extended to the other types. A lesson learned *in* one project is
    /// usually still a lesson — several here are phrased with a project as the
    /// evidence and the conclusion general, and scoping those by the name in
    /// their filename would hide exactly what they were written for.
    unscoped_project_memory,

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
            .index_failed,      .broken_link,  .orphan,         .not_in_index,
            .fragment,          .oversized,    .near_duplicate, .duplicate_content,
            .orphan_chunk,      .unprojected_memory, .unscoped_project_memory,
        };
    }

    pub fn all() []const Check {
        return &.{
            .index_failed,      .broken_link, .orphan,          .not_in_index,
            .fragment,          .oversized,   .island,          .near_duplicate,
            .duplicate_content, .stale,       .no_frontmatter,  .orphan_chunk,
            .unprojected_memory, .unscoped_project_memory,
        };
    }

    /// Whether this check means anything for a collection of this kind, before
    /// the collection's own opt-outs.
    ///
    /// Exhaustive on both axes on purpose: a new check cannot be added, and a new
    /// kind cannot be introduced, without deciding what the pair means. The
    /// alternative is a check that quietly applies everywhere, which is how
    /// `fragment` came to report 28 of the 35 memories zkb itself wrote — one
    /// memory is one fact, so "one short chunk" is the shape of a correct memory,
    /// not a defect. `maintain` was reporting `remember`'s normal output.
    ///
    /// This is only the default. It cannot express everything that matters:
    /// `synap` and `docs` are both `documents` and only one of them keeps an
    /// index.md. That is what `checks_off` on the collection is for.
    pub fn defaultFor(self: Check, kind: store.Store.Kind) bool {
        return switch (self) {
            // Failures of zkb's own machinery. Corpus conventions do not enter
            // into it, so these hold everywhere.
            .index_failed, .orphan_chunk => true,

            // Also zkb's own machinery, but only one kind has the projection to
            // be missing. Scoped by kind rather than answered `true` so a
            // documents collection cannot be told to re-index over a row it was
            // never supposed to have.
            .unprojected_memory, .unscoped_project_memory => kind == .memory,

            // Wiki conventions: something links here, an index lists it, links
            // resolve, a topic has neighbours. A memory corpus follows none of
            // them by construction — `remember` writes one file per memory and no
            // index ever points at one — and the writing convention for memories
            // makes a dangling `[[name]]` a deliberate marker for a memory not
            // written yet, not an error. A csv has no links at all.
            .orphan, .not_in_index, .broken_link, .island => kind == .documents,

            // "Too short to be a document" describes a memory rather than
            // faulting it.
            .fragment => kind == .documents,

            // Size and redundancy are properties of text, not of a convention.
            // Rows in a csv are neither, and are not documents to begin with.
            .oversized, .near_duplicate, .duplicate_content, .stale => kind != .records,

            // Off by default everywhere (see `default()`), but when asked for
            // explicitly it is meaningful for anything parsed as a document. A
            // memory without frontmatter does not parse at all.
            .no_frontmatter => kind != .records,
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
    /// Which collection the finding is about, or 0 for one about the index as a
    /// whole. Carried rather than looked up from `path` afterwards: `rel_path` is
    /// collection-relative, so two collections can hold the same path, and a
    /// lookup that guesses wrong would silently move a finding between corpora
    /// with different conventions.
    collection_id: i64,
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
    /// Report only this collection. Zero means all of them.
    ///
    /// Narrowing the report, not the conventions: a collection's `checks_off`
    /// still applies when it is the only one asked for.
    only_collection: i64 = 0,
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
        .unprojected_memory => try checkUnprojectedMemories(gpa, db, &findings),
        .unscoped_project_memory => try checkUnscopedProjectMemories(gpa, db, &findings),
        // Handled together below: all four read the same KNN pass.
        .near_duplicate, .duplicate_content, .island, .stale => {},
    };

    var wants_vector = false;
    for (opts.checks) |c| {
        if (c.isVector()) wants_vector = true;
    }
    if (wants_vector) try runVectorChecks(gpa, db, &findings, opts);

    const policies = try loadPolicies(gpa, db);
    defer {
        for (policies) |p| gpa.free(p.off);
        gpa.free(policies);
    }
    var kept: usize = 0;
    for (findings.items) |f| {
        if (keep(policies, f, opts.only_collection)) {
            findings.items[kept] = f;
            kept += 1;
        } else {
            f.deinit(gpa);
        }
    }
    findings.shrinkRetainingCapacity(kept);

    return .{
        .findings = try findings.toOwnedSlice(gpa),
        .link_graph_empty = !graph_built,
    };
}

/// The one place a finding comes into existence, which is why `collection_id` is
/// a required parameter here rather than something a check may forget to set:
/// a check that does not know which corpus it is talking about cannot be filtered
/// by that corpus's conventions, and the compiler is the only reviewer that never
/// skips a call site.
fn add(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(Finding),
    check: Check,
    collection_id: i64,
    key: []const u8,
    path: []const u8,
    detail: []const u8,
) !void {
    try out.append(gpa, .{
        .check = check,
        .collection_id = collection_id,
        .key = try gpa.dupe(u8, key),
        .path = try gpa.dupe(u8, path),
        .detail = try gpa.dupe(u8, detail),
    });
}

/// What a collection has decided about the checks, resolved once per run.
const Policy = struct {
    id: i64,
    kind: store.Store.Kind,
    /// Comma-separated check names this collection has switched off.
    off: []const u8,

    fn allows(self: Policy, check: Check) bool {
        if (!check.defaultFor(self.kind)) return false;
        var it = std.mem.splitScalar(u8, self.off, ',');
        while (it.next()) |raw| {
            const name = std.mem.trim(u8, raw, " \t");
            if (name.len == 0) continue;
            if (std.mem.eql(u8, name, @tagName(check))) return false;
        }
        return true;
    }
};

fn loadPolicies(gpa: std.mem.Allocator, db: *sqlite.Db) ![]Policy {
    var out: std.ArrayList(Policy) = .empty;
    errdefer {
        for (out.items) |p| gpa.free(p.off);
        out.deinit(gpa);
    }
    var st = try db.prepare("SELECT id, kind, coalesce(checks_off, '') FROM collections");
    defer st.finalize();
    while (try st.step()) {
        try out.append(gpa, .{
            .id = st.columnI64(0),
            .kind = std.meta.stringToEnum(store.Store.Kind, st.columnText(1)) orelse .documents,
            .off = try gpa.dupe(u8, st.columnText(2)),
        });
    }
    return out.toOwnedSlice(gpa);
}

/// A finding about a collection that has declined this check is dropped here, in
/// one place, after every check has run.
///
/// Filtering inside each check's SQL instead would put the same decision in ten
/// queries — and a query that forgot the clause would report noise silently,
/// which is exactly the failure this whole change is repairing.
fn keep(policies: []const Policy, f: Finding, only: i64) bool {
    if (only != 0 and f.collection_id != 0 and f.collection_id != only) return false;
    if (f.collection_id == 0) return true;
    for (policies) |p| {
        if (p.id == f.collection_id) return p.allows(f.check);
    }
    return true;
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
        try add(gpa, out, check, p.collection_id, key, p.a_path, detail);
    }

    if (wanted[2]) {
        for (result.islands) |i| {
            const key = try std.fmt.bufPrint(&kbuf, "island:{s}|{s}", .{ i.path, i.heading });
            const detail = try std.fmt.bufPrint(&dbuf, "nearest is {s} at cos {d:.3}", .{
                if (i.nearest_path.len != 0) i.nearest_path else "(nothing)",
                i.cos,
            });
            try add(gpa, out, .island, i.collection_id, key, i.path, detail);
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
            try add(gpa, out, .stale, s.collection_id, key, s.old_path, detail);
        }
    }
}

fn checkIndexFailed(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT rel_path, index_error, collection_id FROM docs
        \\WHERE index_error IS NOT NULL ORDER BY rel_path
    );
    defer st.finalize();
    var kbuf: [512]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        const key = try std.fmt.bufPrint(&kbuf, "index_failed:{s}", .{path});
        try add(gpa, out, .index_failed, st.columnI64(2), key, path, st.columnText(1));
    }
}

fn checkBrokenLinks(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path, l.raw, l.kind, d.collection_id FROM links l
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
        try add(gpa, out, .broken_link, st.columnI64(3), key, path, detail);
    }
}

/// "Nothing links here" is a finding about a corpus that is supposed to be
/// interlinked, which is why it is one of the checks `defaultFor` withholds from
/// memory and records collections.
///
/// The kind gate used to live here, as `WHERE c.kind = 'documents'` in this one
/// query. That was right about orphans and invisible to every other check, so
/// `fragment` and `broken_link` went on reporting the same corpora. The decision
/// now has exactly one home; putting a copy back here would restore the split it
/// came from.
fn checkOrphans(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path, d.collection_id FROM docs d
        \\WHERE NOT EXISTS (SELECT 1 FROM links l WHERE l.target_doc_id = d.id)
        \\ORDER BY d.rel_path
    );
    defer st.finalize();
    var kbuf: [512]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        if (isConventionFile(path)) continue;
        const key = try std.fmt.bufPrint(&kbuf, "orphan:{s}", .{path});
        try add(gpa, out, .orphan, st.columnI64(1), key, path, "nothing links here");
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
    try add(gpa, out, .orphan_chunk, 0, key, "(index)", detail);
}

/// Indexed memories with no row in `rec_memory`, per collection.
///
/// One finding per collection rather than per document: they always break as a
/// set — the projection is written in the same transaction as the chunks, so
/// whatever skipped it skipped it for every document that pass touched — and
/// forty identical findings would say the same thing forty times while burying
/// everything else in the report.
///
/// The finding carries the real collection id, so `defaultFor` can scope it to
/// memory collections and `--collection` can select it.
fn checkUnprojectedMemories(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.collection_id, c.name, count(*)
        \\FROM docs d
        \\JOIN collections c ON c.id = d.collection_id
        \\LEFT JOIN rec_memory m ON m.doc_id = d.id
        \\WHERE c.kind = 'memory' AND d.indexed_at IS NOT NULL AND m.doc_id IS NULL
        \\GROUP BY d.collection_id, c.name
    );
    defer st.finalize();

    while (try st.step()) {
        const cid = st.columnI64(0);
        const name = st.columnText(1);
        const missing = st.columnI64(2);

        var kbuf: [128]u8 = undefined;
        var dbuf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&kbuf, "unprojected_memory:{d}:{d}", .{ cid, missing });
        const detail = try std.fmt.bufPrint(
            &dbuf,
            "{d} indexed memories have no metadata row; recall cannot rank them; run: zkb index --force",
            .{missing},
        );
        try add(gpa, out, .unprojected_memory, cid, key, name, detail);
    }
}

/// `type: project` memories with no scope, one finding each.
///
/// Per document rather than per collection, unlike `unprojected_memory`: that one
/// always breaks as a set, this one is a per-memory omission by whoever wrote it,
/// and the only useful finding is which file to run `zkb rescope` on.
fn checkUnscopedProjectMemories(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.collection_id, d.rel_path
        \\FROM rec_memory m
        \\JOIN docs d ON d.id = m.doc_id
        \\JOIN collections c ON c.id = d.collection_id
        \\WHERE c.kind = 'memory' AND m.type = 'project'
        \\  AND m.status = 'active' AND m.scope IS NULL
        \\ORDER BY d.rel_path
    );
    defer st.finalize();

    while (try st.step()) {
        const cid = st.columnI64(0);
        const path = st.columnText(1);

        var kbuf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&kbuf, "unscoped_project_memory:{s}", .{path});
        try add(
            gpa,
            out,
            .unscoped_project_memory,
            cid,
            key,
            path,
            "a project memory with no scope opens every other project's session; run: zkb rescope",
        );
    }
}

/// A document sitting next to an index.md that does not mention it. This encodes
/// an actual convention of this knowledge base (one index.md per project), not a
/// universal rule — which is why it is a separate check that can be turned off.
fn checkNotInIndex(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path, idx.rel_path, d.collection_id FROM docs d
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
        try add(gpa, out, .not_in_index, st.columnI64(2), key, path, detail);
    }
}

fn checkFragment(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    out: *std.ArrayList(Finding),
    max_tokens: i64,
) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path, c.n_tokens, d.collection_id FROM docs d
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
        try add(gpa, out, .fragment, st.columnI64(2), key, path, detail);
    }
}

fn checkOversized(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    out: *std.ArrayList(Finding),
    max_chunks: i64,
) !void {
    var st = try db.prepare(
        "SELECT rel_path, chunk_count, collection_id FROM docs WHERE chunk_count > ?1 ORDER BY chunk_count DESC",
    );
    defer st.finalize();
    try st.bindI64(1, max_chunks);
    var kbuf: [512]u8 = undefined;
    var dbuf: [128]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        const key = try std.fmt.bufPrint(&kbuf, "oversized:{s}", .{path});
        const detail = try std.fmt.bufPrint(&dbuf, "{d} chunks", .{st.columnI64(1)});
        try add(gpa, out, .oversized, st.columnI64(2), key, path, detail);
    }
}

fn checkNoFrontmatter(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        "SELECT rel_path, collection_id FROM docs WHERE frontmatter IS NULL AND indexed_at IS NOT NULL ORDER BY rel_path",
    );
    defer st.finalize();
    var kbuf: [512]u8 = undefined;
    while (try st.step()) {
        const path = st.columnText(0);
        const key = try std.fmt.bufPrint(&kbuf, "no_frontmatter:{s}", .{path});
        // Only presence, never compliance: which fields belong in frontmatter is
        // a convention of specific namespaces, not something zkb gets to judge.
        try add(gpa, out, .no_frontmatter, st.columnI64(1), key, path, "no frontmatter block");
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
/// Forget every resolution and work them all out again.
///
/// `resolveLinks` only looks at links with no target yet, which is right for the
/// steady state — a resolved link does not change when a new document arrives.
/// It is wrong exactly once: when the rule for reading a reference changes.
/// Then every stored `target_doc_id` is an answer to a question nobody asks any
/// more, and nothing re-asks it, because re-extraction happens only when a
/// document is re-indexed.
///
/// Found the day `zkb://` gained a collection segment: 928 of 932 scheme links
/// still carried a target computed by the old reading, `maintain` reported them
/// healthy, and each would have broken silently and separately whenever its
/// document was next touched. A latent break spread over weeks is worse than a
/// loud one on the day of the change.
///
/// Cheap on purpose — this rewrites the graph, not the index. Re-indexing the
/// corpus would also do it, at the cost of re-embedding every chunk to fix
/// something no embedding was involved in.
pub fn relinkAll(gpa: std.mem.Allocator, db: *sqlite.Db) !usize {
    try db.exec("UPDATE links SET target_doc_id = NULL");
    return resolveLinks(gpa, db);
}

pub fn resolveLinks(gpa: std.mem.Allocator, db: *sqlite.Db) !usize {
    var resolved: usize = 0;

    var st = try db.prepare(
        \\SELECT l.id, l.raw, l.kind, d.rel_path, d.collection_id
        \\FROM links l JOIN docs d ON d.id = l.doc_id
        \\WHERE l.target_doc_id IS NULL AND l.kind NOT IN ('external', 'asset')
    );
    defer st.finalize();

    const Pending = struct {
        id: i64,
        target: []u8,
        /// The collection the link named, or null to mean "the one the linking
        /// document is in".
        named: ?[]u8,
        collection_id: i64,
    };
    var pending: std.ArrayList(Pending) = .empty;
    defer {
        for (pending.items) |p| {
            gpa.free(p.target);
            if (p.named) |n| gpa.free(n);
        }
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
                .target = t.rel,
                .named = if (t.collection) |c| try gpa.dupe(u8, c) else null,
                .collection_id = st.columnI64(4),
            });
        }
    }

    for (pending.items) |p| {
        // A link that named a collection is resolved in that collection — which
        // is the whole reason the scheme carries one. Before this the query was
        // always scoped to the *linking* document's collection, so a reference
        // across collections could not resolve at all and was reported broken.
        var find = if (p.named) |_|
            try db.prepare(
                \\SELECT d.id FROM docs d JOIN collections c ON c.id = d.collection_id
                \\WHERE c.name = ?1 AND (d.rel_path = ?2 OR d.rel_path LIKE '%/' || ?2)
                \\ORDER BY length(d.rel_path) LIMIT 1
            )
        else
            try db.prepare(
                \\SELECT id FROM docs
                \\WHERE collection_id = ?1 AND (rel_path = ?2 OR rel_path LIKE '%/' || ?2)
                \\ORDER BY length(rel_path) LIMIT 1
            );
        defer find.finalize();
        if (p.named) |n| try find.bindText(1, n) else try find.bindI64(1, p.collection_id);
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

/// Where a link points: which collection, and the path under it.
///
/// `collection` is null for a relative or wiki link, which means "the one I am
/// already in" — the only reading that makes sense for a reference that never
/// named one.
pub const Target = struct {
    collection: ?[]const u8,
    rel: []u8,
};

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
) !?Target {
    var t = raw;
    // A custom URI scheme names a collection and a path under it:
    // `zkb://docs/projects/x/REQ.md` is the collection `docs`, not a path
    // relative to the document that mentions it. Knowledge-base tools commonly
    // define such a scheme, and resolving one as a relative path is the single
    // largest source of false "broken link" findings — measured at 288 of 348
    // on one corpus.
    //
    // Generic on purpose: any scheme that is not a network or filesystem URL is
    // read this way, so no tool's name is hardcoded here.
    const scheme_end = std.mem.indexOf(u8, t, "://");
    const is_collection_uri = if (scheme_end) |e| !isExternalScheme(t[0..e]) else false;
    var named_collection: ?[]const u8 = null;
    if (is_collection_uri) {
        const u = paths.parseUri(t);
        named_collection = u.collection;
        t = u.rel;
        // `zkb://docs` with no path names a collection, not a document.
        if (t.len == 0) return null;
    }
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
        const rel = if (needs_ext)
            try std.fmt.allocPrint(gpa, "{s}.md", .{abs})
        else
            try gpa.dupe(u8, abs);
        return .{ .collection = named_collection, .rel = rel };
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
    return .{ .collection = null, .rel = normalized };
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
