//! RFC 4180 CSV reader.
//!
//! Zig has no CSV in the standard library, so this is the one-time cost of the
//! md + csv decision (REQ D8). What it buys: the header *is* the schema, a person
//! can open a ledger in Numbers and fix a row, and a bank export can be dropped
//! into the directory as-is.
//!
//! The three things that make a hand-rolled CSV parser wrong are all here:
//! a comma inside quotes, a **newline** inside quotes, and `""` as an escaped
//! quote. Splitting on commas and newlines handles none of them and fails
//! silently — the row count just comes out wrong.
//!
//! Rows are yielded one at a time with fields borrowed from an internal buffer,
//! so a caller that needs a field past `next()` must copy it.

const std = @import("std");

pub const Error = error{
    UnterminatedQuote,
    OutOfMemory,
};

pub const Row = struct {
    /// Fields of the current row; valid until the next `next()`.
    fields: []const []const u8,
    /// 1-based line where the row started, for error messages that a person can
    /// act on ("row 41 has 3 fields, header has 5").
    line: usize,
};

pub const Reader = struct {
    src: []const u8,
    pos: usize = 0,
    line: usize = 1,
    gpa: std.mem.Allocator,
    /// Field boundaries within `scratch`, not slices of it: appending a later
    /// field can reallocate `scratch` and would dangle every slice taken before.
    spans: std.ArrayList(Span) = .empty,
    fields: std.ArrayList([]const u8) = .empty,
    /// Unescaped field bytes. A field containing `""` cannot borrow from the
    /// source, so decoded content accumulates here.
    scratch: std.ArrayList(u8) = .empty,

    pub fn init(gpa: std.mem.Allocator, src: []const u8) Reader {
        // A BOM in front of the header would make the first column name unequal
        // to itself, which reads as "missing column" much later.
        const s = if (std.mem.startsWith(u8, src, "\xEF\xBB\xBF")) src[3..] else src;
        return .{ .src = s, .gpa = gpa };
    }

    const Span = struct { start: usize, end: usize };

    pub fn deinit(self: *Reader) void {
        self.spans.deinit(self.gpa);
        self.fields.deinit(self.gpa);
        self.scratch.deinit(self.gpa);
        self.* = undefined;
    }

    /// Next row, or null at end of input. Blank lines are skipped.
    pub fn next(self: *Reader) Error!?Row {
        while (self.pos < self.src.len and (self.src[self.pos] == '\n' or self.src[self.pos] == '\r')) {
            if (self.src[self.pos] == '\n') self.line += 1;
            self.pos += 1;
        }
        if (self.pos >= self.src.len) return null;

        const row_line = self.line;
        self.spans.clearRetainingCapacity();
        self.fields.clearRetainingCapacity();
        self.scratch.clearRetainingCapacity();

        while (true) {
            const start = self.scratch.items.len;
            try self.readField();
            try self.spans.append(self.gpa, .{ .start = start, .end = self.scratch.items.len });

            if (self.pos >= self.src.len) break;
            if (self.src[self.pos] == ',') {
                self.pos += 1;
                continue;
            }
            if (self.src[self.pos] == '\r') self.pos += 1;
            if (self.pos < self.src.len and self.src[self.pos] == '\n') {
                self.pos += 1;
                self.line += 1;
            }
            break;
        }

        // Materialize only now that `scratch` has stopped growing for this row.
        for (self.spans.items) |sp| {
            try self.fields.append(self.gpa, self.scratch.items[sp.start..sp.end]);
        }
        return .{ .fields = self.fields.items, .line = row_line };
    }

    fn readField(self: *Reader) Error!void {
        if (self.pos < self.src.len and self.src[self.pos] == '"') {
            self.pos += 1;
            while (true) {
                if (self.pos >= self.src.len) return error.UnterminatedQuote;
                const c = self.src[self.pos];
                if (c == '"') {
                    // `""` is one literal quote; a single `"` closes the field.
                    if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '"') {
                        try self.scratch.append(self.gpa, '"');
                        self.pos += 2;
                        continue;
                    }
                    self.pos += 1;
                    return;
                }
                // A newline inside quotes is field content, not a row break.
                if (c == '\n') self.line += 1;
                try self.scratch.append(self.gpa, c);
                self.pos += 1;
            }
        }

        while (self.pos < self.src.len) {
            const c = self.src[self.pos];
            if (c == ',' or c == '\n' or c == '\r') break;
            try self.scratch.append(self.gpa, c);
            self.pos += 1;
        }
    }
};

/// Read the whole file into owned rows. Simpler for callers that need random
/// access and fine for the sizes involved — facts is a few hundred lines, a
/// monthly ledger a few thousand.
pub const Table = struct {
    header: [][]const u8,
    rows: [][][]const u8,
    /// Rows whose field count did not match the header, with their line numbers.
    /// Reported rather than skipped: a person editing a csv in a spreadsheet can
    /// break one, and silently dropping it loses data without telling anyone.
    bad_rows: []usize,

    pub fn deinit(self: *Table, gpa: std.mem.Allocator) void {
        for (self.header) |h| gpa.free(h);
        gpa.free(self.header);
        for (self.rows) |r| {
            for (r) |f| gpa.free(f);
            gpa.free(r);
        }
        gpa.free(self.rows);
        gpa.free(self.bad_rows);
        self.* = undefined;
    }

    pub fn columnIndex(self: *const Table, name: []const u8) ?usize {
        for (self.header, 0..) |h, i| if (std.mem.eql(u8, h, name)) return i;
        return null;
    }
};

pub fn parse(gpa: std.mem.Allocator, src: []const u8) !Table {
    var r = Reader.init(gpa, src);
    defer r.deinit();

    var header: [][]const u8 = &.{};
    errdefer {
        for (header) |h| gpa.free(h);
        gpa.free(header);
    }
    var rows: std.ArrayList([][]const u8) = .empty;
    errdefer {
        for (rows.items) |row| {
            for (row) |f| gpa.free(f);
            gpa.free(row);
        }
        rows.deinit(gpa);
    }
    var bad: std.ArrayList(usize) = .empty;
    errdefer bad.deinit(gpa);

    if (try r.next()) |first| {
        header = try dupeRow(gpa, first.fields);
        // A trailing whitespace in a header cell would make every lookup miss.
        for (header) |*h| {
            const t = std.mem.trim(u8, h.*, " \t");
            if (t.len != h.len) {
                const fixed = try gpa.dupe(u8, t);
                gpa.free(h.*);
                h.* = fixed;
            }
        }
    } else {
        return .{ .header = &.{}, .rows = &.{}, .bad_rows = &.{} };
    }

    while (try r.next()) |row| {
        if (row.fields.len != header.len) {
            try bad.append(gpa, row.line);
            continue;
        }
        try rows.append(gpa, try dupeRow(gpa, row.fields));
    }

    return .{
        .header = header,
        .rows = try rows.toOwnedSlice(gpa),
        .bad_rows = try bad.toOwnedSlice(gpa),
    };
}

fn dupeRow(gpa: std.mem.Allocator, fields: []const []const u8) ![][]const u8 {
    var out = try gpa.alloc([]const u8, fields.len);
    var filled: usize = 0;
    errdefer {
        for (out[0..filled]) |f| gpa.free(f);
        gpa.free(out);
    }
    for (fields) |f| {
        out[filled] = try gpa.dupe(u8, f);
        filled += 1;
    }
    return out;
}

/// Append one row, quoting only the fields that need it.
///
/// zkb is the only writer of the files it manages, so correctness here is what
/// keeps a hand edit in Numbers from being the only way a file gets malformed.
pub fn writeRow(w: *std.Io.Writer, fields: []const []const u8) !void {
    for (fields, 0..) |f, i| {
        if (i != 0) try w.writeByte(',');
        if (needsQuoting(f)) {
            try w.writeByte('"');
            for (f) |c| {
                if (c == '"') try w.writeByte('"');
                try w.writeByte(c);
            }
            try w.writeByte('"');
        } else try w.writeAll(f);
    }
    try w.writeByte('\n');
}

fn needsQuoting(f: []const u8) bool {
    if (f.len == 0) return false;
    // Leading/trailing spaces survive only inside quotes.
    if (f[0] == ' ' or f[f.len - 1] == ' ') return true;
    for (f) |c| {
        if (c == ',' or c == '"' or c == '\n' or c == '\r') return true;
    }
    return false;
}
