//! Memory: one markdown file per remembered thing.
//!
//! A memory is a document. It goes through the same ingest pipeline, occupies the
//! same `chunks` rows, and is retrieved by the same fused search. That is the
//! whole point: split memory and knowledge into two systems and "how did I design
//! the retrieval layer" can no longer return both the design document *and* the
//! preference decided while writing it (SPEC §15.1).
//!
//! Four differences, and only four:
//!   frontmatter   parsed into `rec_memory` instead of stored as opaque text
//!   writer        zkb writes these files; documents it only ever reads
//!   recall        pulled in at session start, not only when asked for
//!   recency       time is a ranking signal here and nowhere else
//!
//! Memories are the one thing in zkb that is *not* reproducible from something
//! else — a session is gone once it ends. They are still files, so jj is their
//! backup, history and undo; zkb does not reimplement any of that.

const std = @import("std");
const sqlite = @import("db/sqlite.zig");
const markdown = @import("ingest/markdown.zig");
const scan = @import("ingest/scan.zig");

/// `archive/` is excluded, which is what makes `forget` mean something: the file
/// stays on disk and in version control, but leaves the index entirely. Relying
/// on `status: archived` alone would not do it — the frontmatter keeps archived
/// memories out of the recency and duplicate paths, but a keyword or vector hit
/// would still surface one, so a forgotten memory would keep coming back.
pub const scan_filters: scan.Filters = .{
    .extensions = &.{".md"},
    .exclude_dirs = &.{ ".git", ".jj", "archive" },
};

pub const Type = enum {
    /// Who the person is: role, preferences, constraints.
    user,
    /// How to work: corrections and confirmed approaches.
    feedback,
    /// A decision and its reasoning.
    decision,
    /// Ongoing work not derivable from the code.
    project,
    /// A pointer to something external.
    reference,

    pub fn parse(s: []const u8) ?Type {
        return std.meta.stringToEnum(Type, s);
    }
};

pub const Status = enum { active, archived };

pub const Meta = struct {
    type: Type = .feedback,
    status: Status = .active,
    /// ISO 8601. Ordering key for the recency path.
    created: []const u8 = "",
    source: []const u8 = "",
    /// Comma-joined; grouping key for contradiction detection.
    subjects: []const u8 = "",
    /// Comma-joined `fact:` / `doc:` references.
    refs: []const u8 = "",

    pub fn deinit(self: Meta, gpa: std.mem.Allocator) void {
        gpa.free(self.created);
        gpa.free(self.source);
        gpa.free(self.subjects);
        gpa.free(self.refs);
    }
};

/// Parse a memory's frontmatter. Deliberately not a YAML parser: the schema is
/// fixed and small, and every field has a defined default so a hand-written file
/// missing one is still usable rather than rejected.
pub fn parseMeta(gpa: std.mem.Allocator, frontmatter: ?[]const u8) !Meta {
    var m: Meta = .{
        .created = try gpa.dupe(u8, ""),
        .source = try gpa.dupe(u8, ""),
        .subjects = try gpa.dupe(u8, ""),
        .refs = try gpa.dupe(u8, ""),
    };
    const fm = frontmatter orelse return m;

    var lines = std.mem.splitScalar(u8, fm, '\n');
    while (lines.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        const colon = std.mem.indexOfScalar(u8, t, ':') orelse continue;
        const key = std.mem.trim(u8, t[0..colon], " \t");
        const val = std.mem.trim(u8, t[colon + 1 ..], " \t\"'");

        if (std.mem.eql(u8, key, "type")) {
            if (Type.parse(val)) |ty| m.type = ty;
        } else if (std.mem.eql(u8, key, "status")) {
            if (std.meta.stringToEnum(Status, val)) |st| m.status = st;
        } else if (std.mem.eql(u8, key, "created")) {
            gpa.free(m.created);
            m.created = try gpa.dupe(u8, val);
        } else if (std.mem.eql(u8, key, "source")) {
            gpa.free(m.source);
            m.source = try gpa.dupe(u8, val);
        } else if (std.mem.eql(u8, key, "subjects")) {
            gpa.free(m.subjects);
            m.subjects = try normalizeList(gpa, val);
        } else if (std.mem.eql(u8, key, "refs")) {
            gpa.free(m.refs);
            m.refs = try normalizeList(gpa, val);
        }
    }
    return m;
}

/// `[a, b]` or `a, b` -> `a,b`.
fn normalizeList(gpa: std.mem.Allocator, val: []const u8) ![]u8 {
    const inner = std.mem.trim(u8, val, "[] \t");
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.splitScalar(u8, inner, ',');
    var first = true;
    while (it.next()) |part| {
        const v = std.mem.trim(u8, part, " \t\"'");
        if (v.len == 0) continue;
        if (!first) try out.append(gpa, ',');
        try out.appendSlice(gpa, v);
        first = false;
    }
    return out.toOwnedSlice(gpa);
}

/// Replace a memory's metadata row, in the same transaction as its chunks.
pub fn replaceMeta(db: *sqlite.Db, doc_id: i64, chunk_id: i64, m: Meta) !void {
    {
        var st = try db.prepare("DELETE FROM rec_memory WHERE doc_id = ?1");
        defer st.finalize();
        try st.bindI64(1, doc_id);
        _ = try st.step();
    }
    var st = try db.prepare(
        \\INSERT INTO rec_memory(chunk_id, doc_id, type, status, created, source, subjects, refs)
        \\VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
    );
    defer st.finalize();
    try st.bindI64(1, chunk_id);
    try st.bindI64(2, doc_id);
    try st.bindText(3, @tagName(m.type));
    try st.bindText(4, @tagName(m.status));
    try st.bindText(5, m.created);
    try st.bindText(6, m.source);
    try st.bindText(7, m.subjects);
    try st.bindText(8, m.refs);
    _ = try st.step();
}

/// Compose a memory file. Kept in one place so every writer produces the same
/// shape — the reason `remember` is an API at all rather than "the agent writes
/// a file" is that frontmatter drift would otherwise be invisible until a query
/// silently stopped matching.
pub fn render(
    gpa: std.mem.Allocator,
    m: Meta,
    body: []const u8,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    const w = struct {
        fn app(o: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
            try o.appendSlice(a, s);
        }
    };
    try w.app(&out, gpa, "---\ntype: ");
    try w.app(&out, gpa, @tagName(m.type));
    try w.app(&out, gpa, "\ncreated: ");
    try w.app(&out, gpa, m.created);
    try w.app(&out, gpa, "\nstatus: ");
    try w.app(&out, gpa, @tagName(m.status));
    if (m.source.len != 0) {
        try w.app(&out, gpa, "\nsource: ");
        try w.app(&out, gpa, m.source);
    }
    if (m.subjects.len != 0) {
        try w.app(&out, gpa, "\nsubjects: [");
        try w.app(&out, gpa, m.subjects);
        try w.app(&out, gpa, "]");
    }
    if (m.refs.len != 0) {
        try w.app(&out, gpa, "\nrefs: [");
        try w.app(&out, gpa, m.refs);
        try w.app(&out, gpa, "]");
    }
    try w.app(&out, gpa, "\n---\n\n");
    try w.app(&out, gpa, std.mem.trim(u8, body, " \t\r\n"));
    try w.app(&out, gpa, "\n");
    return out.toOwnedSlice(gpa);
}

/// Derive a filename from the body: lowercase ASCII words joined by dashes.
///
/// Falls back to the timestamp when the body has no usable ASCII — a Chinese-only
/// memory would otherwise produce an empty name. The slug is cosmetic; identity
/// is the path, and collisions are resolved by the caller appending a counter.
pub fn slug(gpa: std.mem.Allocator, body: []const u8, fallback: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);

    var words: usize = 0;
    var i: usize = 0;
    var in_word = false;
    while (i < body.len and words < 6) : (i += 1) {
        const c = body[i];
        if (std.ascii.isAlphanumeric(c)) {
            if (!in_word and out.items.len != 0) try out.append(gpa, '-');
            try out.append(gpa, std.ascii.toLower(c));
            in_word = true;
        } else if (in_word) {
            in_word = false;
            words += 1;
        }
    }
    if (out.items.len < 3) {
        out.clearRetainingCapacity();
        try out.appendSlice(gpa, fallback);
    }
    return out.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------------------
// recency
// ---------------------------------------------------------------------------

/// Memories ordered newest first, as a ranking path for RRF.
///
/// Time enters ranking as **its own path**, not as a multiplicative decay on the
/// score. A decay needs a coefficient, and a coefficient with no experimental
/// backing is exactly the debt a prior design flagged and that RRF was chosen
/// to avoid. A rank list needs no calibration and no units (SPEC §15.6).
///
/// Only used by `recall`. In `search` the user asked for something specific, and
/// letting the newest memory outrank the relevant one would be wrong.
pub fn recencyRanked(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    limit: usize,
) ![]i64 {
    var out: std.ArrayList(i64) = .empty;
    errdefer out.deinit(gpa);
    var st = try db.prepare(
        \\SELECT chunk_id FROM rec_memory
        \\WHERE status = 'active'
        \\ORDER BY created DESC, chunk_id DESC
        \\LIMIT ?1
    );
    defer st.finalize();
    try st.bindI64(1, @intCast(limit));
    while (try st.step()) try out.append(gpa, st.columnI64(0));
    return out.toOwnedSlice(gpa);
}

/// Chunk ids of active memories, for restricting a KNN to memories only.
pub fn activeChunkIds(gpa: std.mem.Allocator, db: *sqlite.Db) ![]i64 {
    var out: std.ArrayList(i64) = .empty;
    errdefer out.deinit(gpa);
    var st = try db.prepare("SELECT chunk_id FROM rec_memory WHERE status = 'active'");
    defer st.finalize();
    while (try st.step()) try out.append(gpa, st.columnI64(0));
    return out.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------------------
// duplicate detection
// ---------------------------------------------------------------------------

pub const Candidate = struct {
    rel_path: []const u8,
    excerpt: []const u8,
    cos: f64,

    pub fn deinit(self: Candidate, gpa: std.mem.Allocator) void {
        gpa.free(self.rel_path);
        gpa.free(self.excerpt);
    }
};

/// Default similarity above which a new memory is treated as a possible repeat.
///
/// Not tuned yet — it wants the same pooled-judgement treatment E2 got before it
/// can claim a number. Erring high means more duplicates slip through; erring low
/// means refusing to record genuinely new things, which is worse, so it starts
/// permissive.
pub const default_dup_threshold: f64 = 0.90;

/// Memories similar to `vec`, most similar first.
///
/// The check runs **before** writing, not as a later cleanup. Recording the same
/// thing twenty times is the characteristic failure of memory systems: each
/// session's agent has no idea the last one already wrote it, and by the time a
/// maintenance pass notices, the duplicates are already polluting every recall.
pub fn findSimilar(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    collection_id: i64,
    vec: []const f32,
    limit: usize,
) ![]Candidate {
    var out: std.ArrayList(Candidate) = .empty;
    errdefer {
        for (out.items) |c| c.deinit(gpa);
        out.deinit(gpa);
    }

    var st = try db.prepare(
        \\SELECT d.rel_path, c.text, v.distance
        \\FROM vec_chunks v
        \\JOIN chunks c ON c.id = v.chunk_id
        \\JOIN docs d ON d.id = c.doc_id
        \\JOIN rec_memory m ON m.chunk_id = v.chunk_id
        \\WHERE v.collection_id = ?1 AND v.embedding MATCH ?2 AND k = ?3
        \\  AND m.status = 'active'
    );
    defer st.finalize();
    try st.bindI64(1, collection_id);
    try st.bindVector(2, vec);
    try st.bindI64(3, @intCast(limit));

    while (try st.step()) {
        const text = st.columnText(1);
        // vec0 reports cosine *distance*; similarity is 1 - d.
        const cos = 1.0 - st.columnF64(2);
        try out.append(gpa, .{
            .rel_path = try gpa.dupe(u8, st.columnText(0)),
            .excerpt = try gpa.dupe(u8, text[0..@min(text.len, 160)]),
            .cos = cos,
        });
    }
    return out.toOwnedSlice(gpa);
}
