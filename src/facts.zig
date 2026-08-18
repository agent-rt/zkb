//! `facts.csv` — low-frequency single-valued facts, append-only.
//!
//! Three kinds of "number" exist and must not be stored the same way (SPEC §16.0):
//!
//!   derived        age, BMI, account balance   **not stored at all**
//!   archive fact   birth date, height, salary  here
//!   growing series weight log, expenses        records/*.csv
//!
//! Derived quantities are never stored. Store `birth_date`, compute the age.
//! Storing both creates two truths and they drift — a birthday passes and the
//! age does not. That one rule removes a whole class of problems.
//!
//! **Two time axes, both in the file** (SPEC §16.2.1):
//!
//!   `at`            when the fact took effect
//!   `recorded_at`   when it was written down
//!
//! "Salary became 480k in April" written down in August is `at = 2026-04-01`,
//! `recorded_at = 2026-08-17`. The two are genuinely different questions and
//! neither can be derived from the other.
//!
//! An earlier design left the second axis to version control — "the commit time
//! already tracks it, do not store two truths". That was wrong for a reason
//! worth remembering: **nothing ever read it.** An axis you cannot query is not
//! an axis, and reading it would have meant forking `jj` and parsing its output.
//! One column makes it real and makes `where recorded_at > '2026-08'` work.
//!
//! Append-only means changing a value never rewrites a row, so a correction is
//! just another row with a later `recorded_at`. Version control is still a good
//! idea here, but it is no longer load-bearing.

const std = @import("std");
const sqlite = @import("db/sqlite.zig");
const csvmod = @import("ingest/csv.zig");
const scan = @import("ingest/scan.zig");

pub const columns = [_][]const u8{ "key", "value", "at", "recorded_at", "src", "note", "scope" };

/// The kb root holds `facts.csv` (and later `records/*.csv`) beside `memory/`.
/// Restricting this collection to `.csv` is what keeps the memory markdown from
/// being indexed twice — the memory root is a subdirectory of this one.
pub const scan_filters: scan.Filters = .{ .extensions = &.{".csv"} };

pub const Error = error{
    MissingColumn,
    OutOfMemory,
};

pub const Fact = struct {
    key: []const u8,
    /// Numeric when the value parses as one. Numbers go to SQL, never to the
    /// vector index: 450000 and 480000 are neighbours in embedding space and the
    /// question "what is my salary" is a comparison, not a similarity.
    value_num: ?f64,
    value_txt: []const u8,
    at: []const u8,
    /// When this row was written. Empty for a hand-written file that omits the
    /// column — absent, not zero, because a made-up date would be worse.
    recorded_at: []const u8,
    src: []const u8,
    note: []const u8,
    /// Which context this fact belongs to, or empty for "any".
    ///
    /// Same rule as a memory's scope: empty is universal and always injected, a
    /// label is injected only when the caller names it. `recall` attaches every
    /// current fact unconditionally, so before this column a work fact reached
    /// every session in every project — including one whose output is public.
    scope: []const u8,
};

/// Parse `facts.csv`. The six columns are built in — no `_schema.json` needed.
pub fn parse(gpa: std.mem.Allocator, source: []const u8) !struct {
    facts: []Fact,
    bad_rows: []usize,

    pub fn deinit(self: *@This(), a: std.mem.Allocator) void {
        for (self.facts) |f| {
            a.free(f.key);
            a.free(f.value_txt);
            a.free(f.at);
            a.free(f.recorded_at);
            a.free(f.src);
            a.free(f.note);
        }
        a.free(self.facts);
        a.free(self.bad_rows);
    }
} {
    var table = try csvmod.parse(gpa, source);
    defer table.deinit(gpa);

    const i_key = table.columnIndex("key") orelse return error.MissingColumn;
    const i_value = table.columnIndex("value") orelse return error.MissingColumn;
    const i_at = table.columnIndex("at") orelse return error.MissingColumn;
    // Optional: a hand-written file predating the column still parses, it just
    // cannot answer "when did I learn this".
    const i_recorded = table.columnIndex("recorded_at");
    const i_src = table.columnIndex("src");
    const i_note = table.columnIndex("note");
    const i_scope = table.columnIndex("scope");

    var out: std.ArrayList(Fact) = .empty;
    errdefer {
        for (out.items) |f| {
            gpa.free(f.key);
            gpa.free(f.value_txt);
            gpa.free(f.at);
            gpa.free(f.recorded_at);
            gpa.free(f.src);
            gpa.free(f.note);
            gpa.free(f.scope);
        }
        out.deinit(gpa);
    }

    for (table.rows) |row| {
        const raw_value = row[i_value];
        try out.append(gpa, .{
            .key = try gpa.dupe(u8, row[i_key]),
            // Try a number; keep the text either way so nothing is lost. A
            // leading-zero string like a postcode stays intact in value_txt.
            .value_num = std.fmt.parseFloat(f64, raw_value) catch null,
            .value_txt = try gpa.dupe(u8, raw_value),
            .at = try gpa.dupe(u8, row[i_at]),
            .recorded_at = try gpa.dupe(u8, if (i_recorded) |i| row[i] else ""),
            .src = try gpa.dupe(u8, if (i_src) |i| row[i] else ""),
            .note = try gpa.dupe(u8, if (i_note) |i| row[i] else ""),
            .scope = try gpa.dupe(u8, if (i_scope) |i| row[i] else ""),
        });
    }

    return .{
        .facts = try out.toOwnedSlice(gpa),
        .bad_rows = try gpa.dupe(usize, table.bad_rows),
    };
}

/// Replace this document's facts. Called in the same transaction that replaces
/// its chunks, like every other materialized view of a file.
pub fn replaceFor(
    db: *sqlite.Db,
    doc_id: i64,
    chunk_ids: []const i64,
    facts: []const Fact,
) !void {
    {
        var st = try db.prepare("DELETE FROM facts WHERE doc_id = ?1");
        defer st.finalize();
        try st.bindI64(1, doc_id);
        _ = try st.step();
    }
    var st = try db.prepare(
        \\INSERT INTO facts(chunk_id, doc_id, line_no, key, value_num, value_txt,
        \\                  at, recorded_at, src, note)
        \\VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
    );
    defer st.finalize();
    for (facts, 0..) |f, i| {
        if (i >= chunk_ids.len) break;
        st.reset();
        try st.bindI64(1, chunk_ids[i]);
        try st.bindI64(2, doc_id);
        try st.bindI64(3, @intCast(i + 2)); // header is line 1
        try st.bindText(4, f.key);
        if (f.value_num) |n| try st.bindF64(5, n) else try st.bindNull(5);
        try st.bindText(6, f.value_txt);
        try st.bindText(7, f.at);
        try st.bindText(8, f.recorded_at);
        try st.bindText(9, f.src);
        try st.bindText(10, f.note);
        _ = try st.step();
    }
}

pub const Current = struct {
    key: []const u8,
    value: []const u8,
    at: []const u8,
    recorded_at: []const u8,
    note: []const u8,
    scope: []const u8,

    pub fn deinit(self: Current, gpa: std.mem.Allocator) void {
        gpa.free(self.key);
        gpa.free(self.value);
        gpa.free(self.at);
        gpa.free(self.recorded_at);
        gpa.free(self.note);
        gpa.free(self.scope);
    }

    /// Does this fact belong in a recall for `want`?
    ///
    /// Empty scope is universal. A labelled fact needs the caller to name that
    /// exact label — no prefix or hierarchy matching, because a scope is an opaque
    /// string and inventing a hierarchy would mean zkb deciding that `work` and
    /// `work/acme` are related.
    pub fn inScope(self: Current, want: ?[]const u8) bool {
        if (self.scope.len == 0) return true;
        const w = want orelse return false;
        return std.mem.eql(u8, self.scope, w);
    }
};

/// Current values and history are read **from the file, not from the index**.
///
/// The index is derived data that can be stale or absent; facts.csv is the
/// truth, it is a few dozen rows, and parsing it costs microseconds. Reading it
/// directly also means `zkb facts` and the recall snapshot never need a model
/// loaded or an index built — asking what your salary is should not depend on
/// whether the embedder is warm.
///
/// The `facts` table still exists, but for one job only: giving a semantically
/// retrieved fact row its value back (§16.5). Display and injection go here.
fn parseFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !?struct {
    source: []u8,
    facts: []Fact,

    pub fn deinit(self: *@This(), a: std.mem.Allocator) void {
        for (self.facts) |f| {
            a.free(f.key);
            a.free(f.value_txt);
            a.free(f.at);
            a.free(f.recorded_at);
            a.free(f.src);
            a.free(f.note);
        }
        a.free(self.facts);
        a.free(self.source);
    }
} {
    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    const size = (try file.stat(io)).size;
    const source = try gpa.alloc(u8, @intCast(size));
    errdefer gpa.free(source);
    var rbuf: [4096]u8 = undefined;
    var reader = file.reader(io, &rbuf);
    try reader.interface.readSliceAll(source);

    const parsed = try parse(gpa, source);
    // `bad_rows` is reported by `zkb maintain`, not here: a broken row must not
    // stop the other facts from being answered.
    defer gpa.free(parsed.bad_rows);
    return .{ .source = source, .facts = parsed.facts };
}

/// The current value of every key: the row with the largest `at` per key.
///
/// This is what `recall` injects. Numeric facts must not be reached by search —
/// a stale narrative mentioning last year's salary would be retrieved just as
/// happily as the fact itself (SPEC §15.5).
pub fn currentAll(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]Current {
    var parsed = (try parseFile(gpa, io, path)) orelse return &.{};
    defer parsed.deinit(gpa);

    var out: std.ArrayList(Current) = .empty;
    errdefer {
        for (out.items) |c| c.deinit(gpa);
        out.deinit(gpa);
    }

    // Last row wins on an equal `at`: appending is how a value is corrected, so
    // within one effective date the later line is the more recent statement.
    for (parsed.facts) |f| {
        for (out.items) |*existing| {
            if (!std.mem.eql(u8, existing.key, f.key)) continue;
            if (std.mem.order(u8, f.at, existing.at) == .lt) break;
            gpa.free(existing.value);
            gpa.free(existing.at);
            gpa.free(existing.recorded_at);
            gpa.free(existing.note);
            existing.value = try gpa.dupe(u8, f.value_txt);
            existing.at = try gpa.dupe(u8, f.at);
            existing.recorded_at = try gpa.dupe(u8, f.recorded_at);
            existing.note = try gpa.dupe(u8, f.note);
            gpa.free(existing.scope);
            // The winning row's scope, not a merge: a fact corrected into a
            // narrower scope must not keep the old wider one.
            existing.scope = try gpa.dupe(u8, f.scope);
            break;
        } else {
            try out.append(gpa, .{
                .key = try gpa.dupe(u8, f.key),
                .value = try gpa.dupe(u8, f.value_txt),
                .at = try gpa.dupe(u8, f.at),
                .recorded_at = try gpa.dupe(u8, f.recorded_at),
                .note = try gpa.dupe(u8, f.note),
                .scope = try gpa.dupe(u8, f.scope),
            });
        }
    }

    const S = struct {
        fn less(_: void, a: Current, b: Current) bool {
            return std.mem.order(u8, a.key, b.key) == .lt;
        }
    };
    std.mem.sort(Current, out.items, {}, S.less);
    return out.toOwnedSlice(gpa);
}

/// Every recorded value for one key, oldest first. Append-only storage means the
/// history is simply all its rows — a weight log has the same shape as a salary
/// history, with no separate mechanism.
pub fn history(
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    key: []const u8,
) ![]Current {
    var parsed = (try parseFile(gpa, io, path)) orelse return &.{};
    defer parsed.deinit(gpa);

    var out: std.ArrayList(Current) = .empty;
    errdefer {
        for (out.items) |c| c.deinit(gpa);
        out.deinit(gpa);
    }
    for (parsed.facts) |f| {
        if (!std.mem.eql(u8, f.key, key)) continue;
        try out.append(gpa, .{
            .key = try gpa.dupe(u8, f.key),
            .value = try gpa.dupe(u8, f.value_txt),
            .at = try gpa.dupe(u8, f.at),
            .recorded_at = try gpa.dupe(u8, f.recorded_at),
            .note = try gpa.dupe(u8, f.note),
            .scope = try gpa.dupe(u8, f.scope),
        });
    }
    // File order is already chronological in practice; sorting makes it so even
    // when a back-dated row was appended later.
    const S = struct {
        fn less(_: void, a: Current, b: Current) bool {
            return std.mem.order(u8, a.at, b.at) == .lt;
        }
    };
    std.mem.sort(Current, out.items, {}, S.less);
    return out.toOwnedSlice(gpa);
}

/// Append one fact to `facts.csv`, creating the file with a header if needed.
///
/// Appending rather than updating is the whole design: a changed value is a new
/// row with a new `at`, the old row stays, and "how do I undo this" is answered
/// by version control instead of by an undo feature.
pub fn append(
    io: std.Io,
    path: []const u8,
    key: []const u8,
    value: []const u8,
    at: []const u8,
    recorded_at: []const u8,
    src: []const u8,
    note: []const u8,
    scope: []const u8,
) !void {
    const exists = blk: {
        std.Io.Dir.accessAbsolute(io, path, .{}) catch break :blk false;
        break :blk true;
    };

    var file = try std.Io.Dir.createFileAbsolute(io, path, .{ .truncate = false });
    defer file.close(io);

    const end = (try file.stat(io)).size;
    var buf: [4096]u8 = undefined;
    var writer = file.writer(io, &buf);
    writer.pos = end;
    const w = &writer.interface;

    if (!exists or end == 0) try csvmod.writeRow(w, &columns);
    try csvmod.writeRow(w, &.{ key, value, at, recorded_at, src, note, scope });
    try w.flush();
}

/// Render each fact as one line for embedding.
///
/// Only the key, the note and the effective date go into the text; the value
/// does not. A number in a 1024-dimensional semantic space is noise — 450000 and
/// 480000 land in the same place — and the value is always read from SQL anyway.
pub fn renderForEmbedding(gpa: std.mem.Allocator, f: Fact) ![]u8 {
    if (f.note.len != 0) {
        return std.fmt.allocPrint(gpa, "fact: {s}\nnote: {s}\nsince: {s}", .{ f.key, f.note, f.at });
    }
    return std.fmt.allocPrint(gpa, "fact: {s}\nsince: {s}", .{ f.key, f.at });
}
