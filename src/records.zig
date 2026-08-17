//! `records/*.csv` — growing series: expenses, a weight log, a merchant list.
//!
//! The dividing line from `facts.csv` is whether it grows without bound
//! (SPEC §16.2). A salary is a handful of rows over a lifetime; a spend log is a
//! row a day forever. Same mechanism, different file, because a table that grows
//! wants per-column indexes and aggregation, and an archive fact does not.
//!
//! **The header is the schema.** Types come from the header plus the first
//! `sample_rows` rows, deterministically — no `_schema.json` required and no LLM
//! involved. `_schema.json` exists only to override a column the inference got
//! wrong, and it only has to name that column.
//!
//! **Materialized columns, not EAV.** An EAV store earns its complexity when the
//! truth lives in the database and an agent adding a field must not force a
//! migration. zkb's
//! truth is the file and the index is disposable: change the header, re-index,
//! done. All of EAV's complexity buys the avoidance of migrations, and zkb has
//! no migrations to avoid (SPEC §16.3). What it buys instead is `amount > 1000`
//! going through a B-tree rather than three self-joins.

const std = @import("std");
const sqlite = @import("db/sqlite.zig");
const csvmod = @import("ingest/csv.zig");
const hash = @import("util/hash.zig");
const utf8 = @import("util/utf8.zig");

/// How many rows the inference looks at. Enough to be confident, few enough that
/// a large file costs nothing extra.
pub const sample_rows: usize = 200;

pub const Kind = enum {
    /// ISO 8601 date or datetime.
    date,
    number,
    /// Low-cardinality text: a category, a yes/no column.
    @"enum",
    /// An opaque identifier. Indexed like an enum, but never embedded — a SKU or
    /// a row id has nothing for a semantic neighbourhood to be built out of.
    ///
    /// Not in SPEC §16.3, which has only the five below. Added because the very
    /// first real file exposed the gap: `店铺ID` values like `A_0001` are all
    /// distinct, so they fall through to `string` and get embedded, which is
    /// precisely the noise that embedding a whole record row produces.
    id,
    /// Free text — the only kind that goes into the embedding.
    string,
    /// A path that exists relative to the collection root.
    file,

    /// Only free text carries meaning a vector can represent. Numbers, dates and
    /// categories are answered by comparison and grouping, and putting them in a
    /// 1024-dimensional space adds noise without adding an answer (SPEC §16.4).
    pub fn vectorized(self: Kind) bool {
        return self == .string;
    }

    /// Anything worth filtering or sorting on gets a B-tree. Free text does not:
    /// that is what the FTS index is for.
    pub fn indexed(self: Kind) bool {
        return switch (self) {
            .date, .number, .@"enum", .id => true,
            .string, .file => false,
        };
    }

    /// STRICT tables need a declared type. Everything but `number` stays TEXT so
    /// a value that stops parsing later is stored rather than rejected.
    pub fn sqlType(self: Kind) []const u8 {
        return if (self == .number) "REAL" else "TEXT";
    }

    pub fn parse(s: []const u8) ?Kind {
        return std.meta.stringToEnum(Kind, s);
    }
};

pub const Field = struct {
    /// The header as written, used for display and for `--where`.
    name: []const u8,
    kind: Kind,
    /// Which csv column this field reads from.
    ///
    /// Kept explicitly because `_schema.json`'s `display` reorders `fields`, and
    /// after that the position in this slice is no longer the position in the
    /// row. Indexing the row by the field's position put every value one column
    /// off — silently, and only for types that had a display override.
    col: usize = 0,
    /// Fraction of sampled values that are distinct. Only meaningful for a
    /// freshly inferred schema — `rowLabel` uses it to pick which field names a
    /// row, and that runs at index time where inference just happened.
    distinct_ratio: f64 = 0,
    /// True when `_schema.json` set the kind rather than inference.
    overridden: bool = false,

    pub fn deinit(self: Field, gpa: std.mem.Allocator) void {
        gpa.free(self.name);
    }
};

pub const Schema = struct {
    type_name: []const u8,
    fields: []Field,

    pub fn deinit(self: *Schema, gpa: std.mem.Allocator) void {
        for (self.fields) |f| f.deinit(gpa);
        gpa.free(self.fields);
        gpa.free(self.type_name);
        self.* = undefined;
    }

    pub fn find(self: *const Schema, name: []const u8) ?Field {
        for (self.fields) |f| {
            if (std.mem.eql(u8, f.name, name)) return f;
        }
        return null;
    }
};

/// `records/expenses/2026-08.csv` -> `expenses`; `records/weight.csv` -> `weight`.
///
/// A directory groups files of one type so a growing series can be split by
/// month: one file does not grow without bound, jj diffs stay small, and two
/// agents appending in different months never touch the same file (SPEC §16.3).
pub fn typeOf(rel_path: []const u8) ?[]const u8 {
    const prefix = "records/";
    if (!std.mem.startsWith(u8, rel_path, prefix)) return null;
    const rest = rel_path[prefix.len..];
    if (rest.len == 0) return null;

    if (std.mem.indexOfScalar(u8, rest, '/')) |slash| {
        return if (slash == 0) null else rest[0..slash];
    }
    const base = std.fs.path.basename(rest);
    if (!std.mem.endsWith(u8, base, ".csv")) return null;
    const stem = base[0 .. base.len - 4];
    return if (stem.len == 0) null else stem;
}

// ---------------------------------------------------------------------------
// inference
// ---------------------------------------------------------------------------

/// The files that make up one type, newest name first.
///
/// Sorted descending because a type split by month is named by month: the most
/// recent file is the most representative of the shape the data has now, and it
/// is the one a truncated sample should be drawn from.
pub fn typeFiles(
    gpa: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    type_name: []const u8,
) ![][]u8 {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |p| gpa.free(p);
        out.deinit(gpa);
    }

    const dir_path = try std.fmt.allocPrint(gpa, "{s}/records/{s}", .{ root, type_name });
    defer gpa.free(dir_path);

    if (std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true })) |opened| {
        var dir = opened;
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".csv")) continue;
            try out.append(gpa, try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir_path, entry.name }));
        }
    } else |_| {
        // A type can also be a single file: `records/weight.csv`.
        const single = try std.fmt.allocPrint(gpa, "{s}/records/{s}.csv", .{ root, type_name });
        errdefer gpa.free(single);
        std.Io.Dir.accessAbsolute(io, single, .{}) catch {
            gpa.free(single);
            return out.toOwnedSlice(gpa);
        };
        try out.append(gpa, single);
    }

    const S = struct {
        fn desc(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .gt;
        }
    };
    std.mem.sort([]u8, out.items, {}, S.desc);
    return out.toOwnedSlice(gpa);
}

pub const Sample = struct {
    table: csvmod.Table,
    /// Set when some file's header disagrees with the first one's.
    mismatch: ?[]u8 = null,

    pub fn deinit(self: *Sample, gpa: std.mem.Allocator) void {
        self.table.deinit(gpa);
        if (self.mismatch) |m| gpa.free(m);
        self.* = undefined;
    }
};

/// Sample rows from **every** file of a type, so the inferred schema is a
/// property of the type rather than of whichever file happened to be indexed
/// last.
///
/// Per-file inference looks fine until a type spans two files, which SPEC §16.3
/// actively recommends (one file per month). Then each file infers its own
/// shape, the shapes disagree on a small month, and `ensureTable` rebuilds the
/// table on every index — each rebuild discarding the other file's rows. That is
/// not an edge case, it is the normal path.
pub fn collectSample(
    gpa: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    type_name: []const u8,
) !Sample {
    const files = try typeFiles(gpa, io, root, type_name);
    defer {
        for (files) |f| gpa.free(f);
        gpa.free(files);
    }

    // Rebuilt as one csv so the existing parser does the work; at `sample_rows`
    // rows this is kilobytes.
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();

    // Owned, not borrowed: each file's parsed table is freed at the end of its
    // iteration, so keeping a slice into one to compare against the next reads
    // freed memory — and it fails in the most misleading way possible, by
    // reporting that two identical headers differ.
    var header: ?[][]u8 = null;
    defer if (header) |h| {
        for (h) |name| gpa.free(name);
        gpa.free(h);
    };
    var mismatch: ?[]u8 = null;
    errdefer if (mismatch) |m| gpa.free(m);
    var rows: usize = 0;

    for (files) |path| {
        if (rows >= sample_rows and header != null) break;

        const source = readFile(gpa, io, path) catch continue;
        defer gpa.free(source);
        var t = csvmod.parse(gpa, source) catch continue;
        defer t.deinit(gpa);
        if (t.header.len == 0) continue;

        if (header) |h| {
            if (!sameHeader(h, t.header)) {
                if (mismatch == null) {
                    mismatch = try gpa.dupe(u8, std.fs.path.basename(path));
                }
                continue;
            }
        } else {
            const owned = try gpa.alloc([]u8, t.header.len);
            var n: usize = 0;
            errdefer {
                for (owned[0..n]) |x| gpa.free(x);
                gpa.free(owned);
            }
            for (t.header) |name| {
                owned[n] = try gpa.dupe(u8, name);
                n += 1;
            }
            header = owned;
            try csvmod.writeRow(&aw.writer, t.header);
        }

        for (t.rows) |row| {
            if (rows >= sample_rows) break;
            try csvmod.writeRow(&aw.writer, row);
            rows += 1;
        }
    }

    return .{
        .table = try csvmod.parse(gpa, aw.written()),
        .mismatch = mismatch,
    };
}

fn sameHeader(a: []const []const u8, b: []const []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!std.mem.eql(u8, x, y)) return false;
    }
    return true;
}

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    const buf = try gpa.alloc(u8, @intCast(size));
    errdefer gpa.free(buf);
    var rbuf: [4096]u8 = undefined;
    var reader = file.reader(io, &rbuf);
    try reader.interface.readSliceAll(buf);
    return buf;
}

/// Infer a schema from a parsed table. Deterministic and cheap: the whole point
/// is that adding a column to a csv needs no configuration step.
pub fn infer(
    gpa: std.mem.Allocator,
    type_name: []const u8,
    table: *const csvmod.Table,
) !Schema {
    var fields = try gpa.alloc(Field, table.header.len);
    var filled: usize = 0;
    errdefer {
        for (fields[0..filled]) |f| f.deinit(gpa);
        gpa.free(fields);
    }

    for (table.header, 0..) |name, col| {
        const inferred = try inferColumn(gpa, table, col);
        fields[filled] = .{
            .name = try gpa.dupe(u8, name),
            .kind = inferred.kind,
            .distinct_ratio = inferred.distinct_ratio,
            .col = col,
        };
        filled += 1;
    }

    return .{ .type_name = try gpa.dupe(u8, type_name), .fields = fields };
}

const Inferred = struct { kind: Kind, distinct_ratio: f64 };

fn inferColumn(gpa: std.mem.Allocator, table: *const csvmod.Table, col: usize) !Inferred {
    const limit = @min(table.rows.len, sample_rows);

    var total: usize = 0;
    var all_date = true;
    var all_number = true;
    var all_identifier = true;

    var distinct: std.StringHashMapUnmanaged(void) = .empty;
    defer distinct.deinit(gpa);

    for (table.rows[0..limit]) |row| {
        if (col >= row.len) continue;
        const v = std.mem.trim(u8, row[col], " \t");
        // An empty field is NULL and carries no evidence either way; a column of
        // dates with one blank is still a column of dates.
        if (v.len == 0) continue;
        total += 1;

        if (!isIsoDate(v)) all_date = false;
        if (!isNumber(v)) all_number = false;
        if (!isIdentifier(v)) all_identifier = false;
        try distinct.put(gpa, v, {});
    }

    // An all-empty column has nothing to go on. `string` is the safe default: it
    // is the only kind that neither indexes nor coerces.
    if (total == 0) return .{ .kind = .string, .distinct_ratio = 0 };

    const n_distinct = distinct.count();
    const ratio = @as(f64, @floatFromInt(n_distinct)) / @as(f64, @floatFromInt(total));

    if (all_date) return .{ .kind = .date, .distinct_ratio = ratio };
    if (all_number) return .{ .kind = .number, .distinct_ratio = ratio };

    // Below the evidence floor nothing is claimed at all: with three rows every
    // column is either all-distinct or all-same, and both ratios are as
    // meaningless as they are extreme.
    if (total < min_evidence_rows) return .{ .kind = .string, .distinct_ratio = ratio };

    // Nearly-unique short tokens with a digit in them are identifiers, not prose.
    // Requiring a digit is what keeps a column of distinct Japanese shop names
    // (which are prose, and should be embedded) from being swallowed here.
    if (all_identifier and ratio > 0.95) return .{ .kind = .id, .distinct_ratio = ratio };

    if (n_distinct <= max_enum_values and ratio <= max_enum_ratio) {
        return .{ .kind = .@"enum", .distinct_ratio = ratio };
    }
    return .{ .kind = .string, .distinct_ratio = ratio };
}

/// A category set a person can hold in their head. Beyond this the column is not
/// something anyone groups by.
const max_enum_values: usize = 50;

/// Below this many values there is not enough evidence to classify a text column
/// at all: three rows sharing a value proves nothing, and three distinct rows
/// prove just as little. Applies to `enum` and `id` alike — a distinctness ratio
/// computed over two samples is always either 1.0 or 0.5.
const min_evidence_rows: usize = 8;

/// At most this fraction distinct — i.e. each value repeats three times on
/// average.
///
/// SPEC §16.3 specified `< 0.1`, which the first real file falsified: `类别` has
/// 13 categories across 90 rows (ratio 0.144) and is unmistakably an enum, but
/// fell through to `string`. The flaw is that a pure ratio scales wrong in both
/// directions — under 0.1, a 20-row file can have no enum at all (it would need
/// ≤2 distinct), while a 100k-row file would accept 9,000 distinct values as a
/// category. Two independent bounds, an absolute count and a repetition rate,
/// say what was actually meant.
///
/// Erring is not symmetric, which is why this sits at 1/3 rather than higher:
/// calling free text an enum removes it from the embedding and so from semantic
/// search, while calling a category a string only costs an index and adds a
/// little noise. The rule should fall towards `string`.
const max_enum_ratio: f64 = 1.0 / 3.0;

/// `2026-08-14`, optionally with a time part.
fn isIsoDate(v: []const u8) bool {
    if (v.len < 10) return false;
    for (v[0..10], 0..) |c, i| {
        const want_dash = i == 4 or i == 7;
        if (want_dash) {
            if (c != '-') return false;
        } else if (!std.ascii.isDigit(c)) return false;
    }
    if (v.len == 10) return true;
    // Accept `T` or a space before the time, and anything time-shaped after.
    if (v[10] != 'T' and v[10] != ' ') return false;
    for (v[11..]) |c| {
        if (!std.ascii.isDigit(c) and c != ':' and c != '.' and
            c != '+' and c != '-' and c != 'Z') return false;
    }
    return true;
}

/// Parses as a float — with one carve-out.
///
/// A leading zero is meaningful in an identifier (a postcode, a phone number, an
/// account) and meaningless in a quantity, so `0912345678` is text. SPEC §16.3
/// lists this as the trap `_schema.json` exists for; it is cheap enough to just
/// not fall into.
fn isNumber(v: []const u8) bool {
    const body = if (v.len != 0 and (v[0] == '+' or v[0] == '-')) v[1..] else v;
    if (body.len > 1 and body[0] == '0' and body[1] != '.') return false;
    _ = std.fmt.parseFloat(f64, v) catch return false;
    return true;
}

/// Short, no whitespace, and contains a digit.
fn isIdentifier(v: []const u8) bool {
    if (v.len == 0 or v.len > 32) return false;
    var has_digit = false;
    for (v) |c| {
        if (c == ' ' or c == '\t' or c == '\n') return false;
        if (std.ascii.isDigit(c)) has_digit = true;
    }
    return has_digit;
}

// ---------------------------------------------------------------------------
// _schema.json overrides
// ---------------------------------------------------------------------------

/// Apply `records/<type>/_schema.json` on top of an inferred schema.
///
/// Only the columns it names change, and an unknown column name is ignored
/// rather than fatal — the file is an override, not a definition, so it must not
/// be able to make a working csv unreadable.
pub fn applyOverrides(
    gpa: std.mem.Allocator,
    schema: *Schema,
    json_source: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json_source, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .object) return;

    if (parsed.value.object.get("fields")) |fv| if (fv == .object) {
        var it = fv.object.iterator();
        while (it.next()) |entry| {
            const spec = entry.value_ptr.*;
            if (spec != .object) continue;
            const tv = spec.object.get("type") orelse continue;
            if (tv != .string) continue;
            const kind = Kind.parse(tv.string) orelse continue;
            for (schema.fields) |*f| {
                if (!std.mem.eql(u8, f.name, entry.key_ptr.*)) continue;
                f.kind = kind;
                f.overridden = true;
            }
        }
    };

    // `display` reorders the columns for output. Reordering the fields slice is
    // enough: the materialized table is addressed by name, never by position.
    if (parsed.value.object.get("display")) |dv| if (dv == .array) {
        var ordered: std.ArrayList(Field) = .empty;
        defer ordered.deinit(gpa);
        for (dv.array.items) |item| {
            if (item != .string) continue;
            for (schema.fields) |f| {
                if (std.mem.eql(u8, f.name, item.string)) try ordered.append(gpa, f);
            }
        }
        // Columns the display list omits still exist and still get stored; they
        // just sort after the named ones.
        for (schema.fields) |f| {
            var listed = false;
            for (ordered.items) |o| {
                if (std.mem.eql(u8, o.name, f.name)) listed = true;
            }
            if (!listed) try ordered.append(gpa, f);
        }
        if (ordered.items.len == schema.fields.len) {
            @memcpy(schema.fields, ordered.items);
        }
    };
}

// ---------------------------------------------------------------------------
// materialized table
// ---------------------------------------------------------------------------

/// `rec_<type>`, quoted. Identifiers are always quoted rather than sanitized so a
/// Chinese or Japanese header stays itself in `zkb sql` output.
pub fn tableName(gpa: std.mem.Allocator, type_name: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa, "rec_{s}", .{type_name});
}

/// Double-quote an identifier, doubling any embedded quote.
///
/// The only way a header reaches SQL. Column names come from a file a person
/// edits, so they are untrusted input in exactly the way a value is.
pub fn quoteIdent(gpa: std.mem.Allocator, name: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.append(gpa, '"');
    for (name) |c| {
        if (c == '"') try out.append(gpa, '"');
        try out.append(gpa, c);
    }
    try out.append(gpa, '"');
    return out.toOwnedSlice(gpa);
}

/// Create the materialized table if its shape matches, or rebuild it if the
/// header changed.
///
/// Rebuilding is free here in a way it never is for a normal application: the
/// rows are a projection of files that are still on disk, so dropping the table
/// loses nothing that re-indexing does not restore.
pub fn ensureTable(gpa: std.mem.Allocator, db: *sqlite.Db, schema: *const Schema) !void {
    const stored = try loadSchema(gpa, db, schema.type_name);
    defer if (stored) |*st| {
        var m = st.*;
        m.deinit(gpa);
    };

    const table = try tableName(gpa, schema.type_name);
    defer gpa.free(table);
    const qtable = try quoteIdent(gpa, table);
    defer gpa.free(qtable);

    if (stored) |st| {
        if (sameShape(&st, schema)) return;
        // A changed header means the old rows describe a different table.
        const drop = try std.fmt.allocPrintSentinel(gpa, "DROP TABLE IF EXISTS {s};", .{qtable}, 0);
        defer gpa.free(drop);
        try db.exec(drop);
        // Dropping the table discards the rows of every file of this type, not
        // just the one being indexed — so every one of them has to be read
        // again. Without this the other months simply vanish, and nothing
        // downstream can tell that they used to be there.
        try requeueType(gpa, db, schema.type_name);
    }

    var ddl: std.ArrayList(u8) = .empty;
    defer ddl.deinit(gpa);
    try ddl.print(gpa,
        \\CREATE TABLE {s} (
        \\  chunk_id INTEGER PRIMARY KEY,
        \\  doc_id   INTEGER NOT NULL,
        \\  line_no  INTEGER NOT NULL
    , .{qtable});
    for (schema.fields) |f| {
        const q = try quoteIdent(gpa, f.name);
        defer gpa.free(q);
        try ddl.print(gpa, ",\n  {s} {s}", .{ q, f.kind.sqlType() });
    }
    try ddl.appendSlice(gpa, "\n) STRICT;\n");

    for (schema.fields, 0..) |f, i| {
        if (!f.kind.indexed()) continue;
        const q = try quoteIdent(gpa, f.name);
        defer gpa.free(q);
        // Index names are generated rather than derived from the column, because
        // two types may share a column name.
        const iname = try std.fmt.allocPrint(gpa, "{s}_f{d}", .{ table, i });
        defer gpa.free(iname);
        const qi = try quoteIdent(gpa, iname);
        defer gpa.free(qi);
        try ddl.print(gpa, "CREATE INDEX {s} ON {s}({s});\n", .{ qi, qtable, q });
    }

    const ddl_sql = try ddl.toOwnedSliceSentinel(gpa, 0);
    defer gpa.free(ddl_sql);
    try db.exec(ddl_sql);
    try storeSchema(gpa, db, schema);
}

/// Mark every file of a type as needing indexing again.
///
/// Matched by path rather than by joining the dropped table, for the obvious
/// reason that it no longer exists.
pub fn requeueType(gpa: std.mem.Allocator, db: *sqlite.Db, type_name: []const u8) !void {
    const dir_prefix = try std.fmt.allocPrint(gpa, "records/{s}/%", .{type_name});
    defer gpa.free(dir_prefix);
    const single = try std.fmt.allocPrint(gpa, "records/{s}.csv", .{type_name});
    defer gpa.free(single);

    var st = try db.prepare(
        "UPDATE docs SET indexed_at = NULL WHERE rel_path LIKE ?1 OR rel_path = ?2",
    );
    defer st.finalize();
    try st.bindText(1, dir_prefix);
    try st.bindText(2, single);
    _ = try st.step();
}

fn sameShape(a: *const Schema, b: *const Schema) bool {
    if (a.fields.len != b.fields.len) return false;
    for (a.fields, b.fields) |x, y| {
        if (!std.mem.eql(u8, x.name, y.name)) return false;
        if (x.kind != y.kind) return false;
        if (x.col != y.col) return false;
    }
    return true;
}

/// The inferred schema is written down so `zkb records --schema` can show it.
/// Inference that cannot be inspected is inference that cannot be debugged when
/// it guesses wrong (SPEC §16.3).
fn storeSchema(gpa: std.mem.Allocator, db: *sqlite.Db, schema: *const Schema) !void {
    _ = gpa;
    {
        var st = try db.prepare("DELETE FROM rec_meta WHERE type = ?1");
        defer st.finalize();
        try st.bindText(1, schema.type_name);
        _ = try st.step();
    }
    var st = try db.prepare(
        \\INSERT INTO rec_meta(type, field, kind, indexed, vectorized, ord, overridden, src_col)
        \\VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
    );
    defer st.finalize();
    for (schema.fields, 0..) |f, i| {
        st.reset();
        try st.bindText(1, schema.type_name);
        try st.bindText(2, f.name);
        try st.bindText(3, @tagName(f.kind));
        try st.bindI64(4, if (f.kind.indexed()) 1 else 0);
        try st.bindI64(5, if (f.kind.vectorized()) 1 else 0);
        try st.bindI64(6, @intCast(i));
        try st.bindI64(7, if (f.overridden) 1 else 0);
        try st.bindI64(8, @intCast(f.col));
        _ = try st.step();
    }
}

/// The schema currently materialized for a type, or null if there is none.
pub fn loadSchema(gpa: std.mem.Allocator, db: *sqlite.Db, type_name: []const u8) !?Schema {
    var fields: std.ArrayList(Field) = .empty;
    errdefer {
        for (fields.items) |f| f.deinit(gpa);
        fields.deinit(gpa);
    }
    var st = try db.prepare(
        "SELECT field, kind, overridden, src_col FROM rec_meta WHERE type = ?1 ORDER BY ord",
    );
    defer st.finalize();
    try st.bindText(1, type_name);
    while (try st.step()) {
        try fields.append(gpa, .{
            .name = try gpa.dupe(u8, st.columnText(0)),
            .kind = Kind.parse(st.columnText(1)) orelse .string,
            .overridden = st.columnI64(2) != 0,
            .col = @intCast(st.columnI64(3)),
        });
    }
    if (fields.items.len == 0) {
        fields.deinit(gpa);
        return null;
    }
    return .{
        .type_name = try gpa.dupe(u8, type_name),
        .fields = try fields.toOwnedSlice(gpa),
    };
}

/// Every type that has a materialized table.
pub fn listTypes(gpa: std.mem.Allocator, db: *sqlite.Db) ![][]u8 {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |t| gpa.free(t);
        out.deinit(gpa);
    }
    var st = try db.prepare("SELECT DISTINCT type FROM rec_meta ORDER BY type");
    defer st.finalize();
    while (try st.step()) try out.append(gpa, try gpa.dupe(u8, st.columnText(0)));
    return out.toOwnedSlice(gpa);
}

/// Replace one document's rows in the materialized table.
pub fn replaceFor(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    schema: *const Schema,
    doc_id: i64,
    chunk_ids: []const i64,
    table: *const csvmod.Table,
) !void {
    const name = try tableName(gpa, schema.type_name);
    defer gpa.free(name);
    const qtable = try quoteIdent(gpa, name);
    defer gpa.free(qtable);

    {
        const sql = try std.fmt.allocPrintSentinel(
            gpa,
            "DELETE FROM {s} WHERE doc_id = ?1",
            .{qtable},
            0,
        );
        defer gpa.free(sql);
        var st = try db.prepare(sql);
        defer st.finalize();
        try st.bindI64(1, doc_id);
        _ = try st.step();
    }

    var sql: std.ArrayList(u8) = .empty;
    defer sql.deinit(gpa);
    try sql.print(gpa, "INSERT INTO {s}(chunk_id, doc_id, line_no", .{qtable});
    for (schema.fields) |f| {
        const q = try quoteIdent(gpa, f.name);
        defer gpa.free(q);
        try sql.print(gpa, ", {s}", .{q});
    }
    try sql.appendSlice(gpa, ") VALUES (?1, ?2, ?3");
    for (schema.fields, 0..) |_, i| try sql.print(gpa, ", ?{d}", .{i + 4});
    try sql.appendSlice(gpa, ")");

    const insert_sql = try sql.toOwnedSliceSentinel(gpa, 0);
    defer gpa.free(insert_sql);
    var st = try db.prepare(insert_sql);
    defer st.finalize();

    for (table.rows, 0..) |row, i| {
        if (i >= chunk_ids.len) break;
        st.reset();
        try st.bindI64(1, chunk_ids[i]);
        try st.bindI64(2, doc_id);
        try st.bindI64(3, @intCast(i + 2)); // the header is line 1
        for (schema.fields, 0..) |f, slot| {
            const idx: i32 = @intCast(slot + 4);
            const raw = if (f.col < row.len) std.mem.trim(u8, row[f.col], " \t") else "";
            // Empty is NULL, not an empty string: "no value recorded" and "the
            // value is blank" are the same thing in a spreadsheet, and only NULL
            // is excluded from aggregates.
            if (raw.len == 0) {
                try st.bindNull(idx);
            } else if (f.kind == .number) {
                if (std.fmt.parseFloat(f64, raw)) |n| {
                    try st.bindF64(idx, n);
                } else |_| try st.bindNull(idx);
            } else {
                try st.bindText(idx, raw);
            }
        }
        _ = try st.step();
    }
}

/// Requeue any type whose `_schema.json` changed since it was last indexed.
///
/// The override file is part of a type's shape, but it is not a document: the
/// scanner only sees csv, so editing `_schema.json` changes nothing and the
/// column it was written to fix stays wrong. Watching it here is what makes the
/// override usable at all — otherwise the only way to apply one is to also touch
/// a data file, which nobody would guess.
///
/// Compared by content hash rather than mtime: a file restored from version
/// control has a new mtime and the same meaning.
pub fn reconcileOverrides(
    gpa: std.mem.Allocator,
    io: std.Io,
    db: *sqlite.Db,
    root: []const u8,
) !usize {
    const types = try listTypes(gpa, db);
    defer {
        for (types) |t| gpa.free(t);
        gpa.free(types);
    }

    var requeued: usize = 0;
    for (types) |type_name| {
        const path = try std.fmt.allocPrint(gpa, "{s}/records/{s}/_schema.json", .{ root, type_name });
        defer gpa.free(path);

        // The absent case has to hash to something stable too, so that deleting
        // an override is noticed just like editing one.
        var digest: [64]u8 = undefined;
        if (readFile(gpa, io, path)) |content| {
            defer gpa.free(content);
            digest = hash.bytesSha256(content);
        } else |_| {
            digest = hash.bytesSha256("");
        }

        const key = try std.fmt.allocPrint(gpa, "rec_override:{s}", .{type_name});
        defer gpa.free(key);

        const previous = try getMeta(gpa, db, key);
        defer if (previous) |p| gpa.free(p);

        if (previous) |p| {
            if (std.mem.eql(u8, p, &digest)) continue;
        }
        try requeueType(gpa, db, type_name);
        try setMeta(db, key, &digest);
        requeued += 1;
    }
    return requeued;
}

fn getMeta(gpa: std.mem.Allocator, db: *sqlite.Db, key: []const u8) !?[]u8 {
    var st = try db.prepare("SELECT value FROM meta WHERE key = ?1");
    defer st.finalize();
    try st.bindText(1, key);
    if (!try st.step()) return null;
    return try gpa.dupe(u8, st.columnText(0));
}

fn setMeta(db: *sqlite.Db, key: []const u8, value: []const u8) !void {
    var st = try db.prepare(
        "INSERT INTO meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = ?2",
    );
    defer st.finalize();
    try st.bindText(1, key);
    try st.bindText(2, value);
    _ = try st.step();
}

// ---------------------------------------------------------------------------
// vector reuse
// ---------------------------------------------------------------------------

/// Rendered row text -> the vector already stored for it.
///
/// A csv file is the unit of indexing, so appending one row re-embeds the whole
/// file: 2.0s for a ten-row file, and linear in the file from there. But a row's
/// embedding is a pure function of its rendered text, and appending changes no
/// existing row — so almost every vector about to be recomputed is already on
/// disk and identical.
///
/// Keyed by text rather than by row number so that inserting or deleting a row
/// in the middle still reuses the rest, which sorting a file by date does.
pub const VectorCache = struct {
    map: std.StringHashMapUnmanaged([]f32) = .empty,

    pub fn deinit(self: *VectorCache, gpa: std.mem.Allocator) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            gpa.free(e.key_ptr.*);
            gpa.free(e.value_ptr.*);
        }
        self.map.deinit(gpa);
        self.* = undefined;
    }

    pub fn get(self: *const VectorCache, text: []const u8) ?[]const f32 {
        return self.map.get(text);
    }
};

/// Cap on cached rows. A very large file is re-embedded rather than held in
/// memory: 5000 rows of 1024 floats is 20 MB, and past that the tradeoff stops
/// being obviously good.
pub const max_cached_rows: usize = 5000;

pub fn loadVectors(
    gpa: std.mem.Allocator,
    db: *sqlite.Db,
    doc_id: i64,
    dim: usize,
) !VectorCache {
    var cache: VectorCache = .{};
    errdefer cache.deinit(gpa);

    var st = try db.prepare(
        \\SELECT c.text, v.embedding
        \\FROM chunks c JOIN vec_chunks v ON v.chunk_id = c.id
        \\WHERE c.doc_id = ?1
    );
    defer st.finalize();
    try st.bindI64(1, doc_id);

    while (try st.step()) {
        if (cache.map.count() >= max_cached_rows) break;
        const blob = st.columnBlob(1);
        if (blob.len != dim * @sizeOf(f32)) continue;

        const text = try gpa.dupe(u8, st.columnText(0));
        errdefer gpa.free(text);
        if (cache.map.contains(text)) {
            gpa.free(text);
            continue;
        }
        const vec = try gpa.alloc(f32, dim);
        errdefer gpa.free(vec);
        @memcpy(std.mem.sliceAsBytes(vec), blob);
        try cache.map.put(gpa, text, vec);
    }
    return cache;
}

// ---------------------------------------------------------------------------
// embedding text
// ---------------------------------------------------------------------------

/// One record rendered for the embedding: the type, its date if it has one, and
/// the free-text fields. Nothing else.
///
/// "How much did I spend on coffee at the convenience store last time" should
/// reach this row semantically; the exact amount then comes from the
/// materialized column, where a comparison actually means something.
pub fn renderForEmbedding(
    gpa: std.mem.Allocator,
    schema: *const Schema,
    row: []const []const u8,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);

    try out.appendSlice(gpa, schema.type_name);
    for (schema.fields) |f| {
        if (f.kind != .date or f.col >= row.len) continue;
        const v = std.mem.trim(u8, row[f.col], " \t");
        if (v.len == 0) continue;
        try out.print(gpa, " · {s}", .{v});
        break;
    }

    for (schema.fields) |f| {
        if (!f.kind.vectorized() or f.col >= row.len) continue;
        const v = std.mem.trim(u8, row[f.col], " \t");
        if (v.len == 0) continue;
        try out.print(gpa, "\n{s}: {s}", .{ f.name, v });
    }
    return out.toOwnedSlice(gpa);
}

/// A short label for the chunk, used as its heading path in search output.
///
/// The **most distinctive** free-text field, not the first one. A row is best
/// identified by whatever varies between rows — a merchant name, a title — and
/// the leftmost column is often the least distinctive thing there is. Measured
/// on a four-row file: the label came out `JPY`, because `currency` happened to
/// sit before `merchant` and, at that sample size, was still classed as text.
pub fn rowLabel(gpa: std.mem.Allocator, schema: *const Schema, row: []const []const u8) ![]u8 {
    var best: ?Field = null;
    for (schema.fields) |f| {
        if (!f.kind.vectorized() or f.col >= row.len) continue;
        if (std.mem.trim(u8, row[f.col], " \t").len == 0) continue;
        if (best == null or f.distinct_ratio > best.?.distinct_ratio) best = f;
    }
    const f = best orelse return gpa.dupe(u8, schema.type_name);
    const v = std.mem.trim(u8, row[f.col], " \t");
    // Cut on a character boundary: this string is stored as
    // `chunks.heading_path` and indexed by FTS, so a split sequence is
    // persisted corruption, not just a display glitch.
    return gpa.dupe(u8, utf8.cut(v, 64));
}
