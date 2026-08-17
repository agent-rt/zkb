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

    /// Checks that run by default.
    ///
    /// `no_frontmatter` is deliberately excluded: measured against ~/docs it
    /// fires on 135 of 195 documents (69%). A check that flags most of the corpus
    /// is not a finding, it is a description of the corpus — and it would bury
    /// the handful of findings that do need acting on. Frontmatter is a
    /// convention of specific namespaces (specific namespaces), not a global rule.
    /// Still available via `--check no_frontmatter`.
    pub fn default() []const Check {
        return &.{ .index_failed, .broken_link, .orphan, .not_in_index, .fragment, .oversized };
    }

    pub fn all() []const Check {
        return &.{ .index_failed, .broken_link, .orphan, .not_in_index, .fragment, .oversized, .no_frontmatter };
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
    };

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

fn checkOrphans(gpa: std.mem.Allocator, db: *sqlite.Db, out: *std.ArrayList(Finding)) !void {
    var st = try db.prepare(
        \\SELECT d.rel_path FROM docs d
        \\WHERE NOT EXISTS (SELECT 1 FROM links l WHERE l.target_doc_id = d.id)
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
pub fn record(gpa: std.mem.Allocator, db: *sqlite.Db, report: *const Report, now_ms: i64) !void {
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
    try st.bindI64(1, now_ms);
    try st.bindText(2, "all");
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
    // A kb:// URI is rooted at the collection, not relative to the document
    // that mentions it: kb://projects/alpha/REQ.md means exactly that path.
    const is_lore = std.mem.startsWith(u8, t, "kb://");
    if (is_lore) t = t["kb://".len..];
    if (t.len == 0) return null;
    // A pure anchor points inside the same document; not a link between files.
    if (t[0] == '#') return null;
    // Drop any fragment.
    if (std.mem.indexOfScalar(u8, t, '#')) |h| t = t[0..h];
    if (t.len == 0) return null;
    if (std.mem.indexOf(u8, t, "://") != null) return null;

    // A wikilink without an extension names a document by stem.
    const needs_ext = std.mem.indexOfScalar(u8, t, '.') == null;

    if (is_lore or t[0] == '/') {
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
