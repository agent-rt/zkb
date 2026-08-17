//! `--where` and `--agg`: a restricted expression language compiled to
//! **parameterized** SQL.
//!
//! Two rules, and everything here follows from them:
//!
//!   1. **A value never reaches the SQL text.** Values become `?n` bindings, so
//!      there is no quoting to get right and no escaping to forget. This is not
//!      defence against a hostile user — it is their own knowledge base — it is
//!      defence against an apostrophe in a merchant name silently breaking the
//!      query, or a value like `1 OR 1=1` doing something surprising.
//!
//!   2. **A field name is checked against the schema before it is written.**
//!      Identifiers cannot be parameterized in SQL, so the only safe form is a
//!      whitelist: the name must equal a column the schema declares, or the
//!      query is refused. Nothing is sanitized or escaped into existence.
//!
//! The grammar is deliberately small. `zkb sql` is the escape hatch for anything
//! it cannot say (SPEC §16.6), which is what lets this stay small instead of
//! growing into a second SQL dialect.

const std = @import("std");
const records = @import("../records.zig");

pub const Error = error{
    UnknownField,
    UnexpectedToken,
    UnexpectedEnd,
    UnknownOperator,
    EmptyExpression,
    TooComplex,
    OutOfMemory,
};

pub const max_bindings = 64;

pub const Value = union(enum) {
    text: []const u8,
    number: f64,
};

/// A compiled WHERE clause: SQL with `?n` placeholders, plus the values to bind.
pub const Compiled = struct {
    sql: []u8,
    values: []Value,
    /// Points into the caller's expression string; no copies are made.
    pub fn deinit(self: *Compiled, gpa: std.mem.Allocator) void {
        gpa.free(self.sql);
        gpa.free(self.values);
        self.* = undefined;
    }
};

/// Compile `amount > 1000 AND category = 'food'` against a schema.
pub fn compileWhere(
    gpa: std.mem.Allocator,
    schema: *const records.Schema,
    source: []const u8,
) Error!Compiled {
    var p: Parser = .{ .gpa = gpa, .schema = schema, .src = source };
    defer p.sql.deinit(gpa);
    defer p.values.deinit(gpa);

    p.skipSpace();
    if (p.pos >= p.src.len) return error.EmptyExpression;

    try p.parseOr();
    p.skipSpace();
    if (p.pos < p.src.len) return error.UnexpectedToken;

    return .{
        .sql = try p.sql.toOwnedSlice(gpa),
        .values = try p.values.toOwnedSlice(gpa),
    };
}

const Parser = struct {
    gpa: std.mem.Allocator,
    schema: *const records.Schema,
    src: []const u8,
    pos: usize = 0,
    sql: std.ArrayList(u8) = .empty,
    values: std.ArrayList(Value) = .empty,

    fn skipSpace(self: *Parser) void {
        while (self.pos < self.src.len and
            (self.src[self.pos] == ' ' or self.src[self.pos] == '\t')) self.pos += 1;
    }

    /// OR binds loosest, so it sits outermost. Explicit precedence rather than a
    /// flat left-to-right chain: `a = 1 OR b = 2 AND c = 3` surprising anyone is
    /// worse than the parser being ten lines longer.
    fn parseOr(self: *Parser) Error!void {
        try self.parseAnd();
        while (true) {
            self.skipSpace();
            if (!self.eatKeyword("OR")) return;
            try self.sql.appendSlice(self.gpa, " OR ");
            try self.parseAnd();
        }
    }

    fn parseAnd(self: *Parser) Error!void {
        try self.parseTerm();
        while (true) {
            self.skipSpace();
            if (!self.eatKeyword("AND")) return;
            try self.sql.appendSlice(self.gpa, " AND ");
            try self.parseTerm();
        }
    }

    fn parseTerm(self: *Parser) Error!void {
        self.skipSpace();
        if (self.pos >= self.src.len) return error.UnexpectedEnd;

        if (self.src[self.pos] == '(') {
            self.pos += 1;
            try self.sql.append(self.gpa, '(');
            try self.parseOr();
            self.skipSpace();
            if (self.pos >= self.src.len or self.src[self.pos] != ')') return error.UnexpectedToken;
            self.pos += 1;
            try self.sql.append(self.gpa, ')');
            return;
        }

        const field_name = try self.readFieldName();
        const field = self.schema.find(field_name) orelse return error.UnknownField;

        const quoted = try records.quoteIdent(self.gpa, field.name);
        defer self.gpa.free(quoted);
        try self.sql.appendSlice(self.gpa, quoted);

        self.skipSpace();
        const op = try self.readOperator();
        try self.sql.append(self.gpa, ' ');
        try self.sql.appendSlice(self.gpa, op);
        try self.sql.append(self.gpa, ' ');

        if (std.mem.eql(u8, op, "IN")) return self.parseInList(field);
        if (std.mem.eql(u8, op, "IS NULL") or std.mem.eql(u8, op, "IS NOT NULL")) {
            // No operand: `IS NULL` is the only way to ask about an empty cell,
            // since `= ''` cannot match a NULL.
            _ = self.sql.pop();
            return;
        }
        return self.parseValue(field);
    }

    fn parseInList(self: *Parser, field: records.Field) Error!void {
        self.skipSpace();
        if (self.pos >= self.src.len or self.src[self.pos] != '(') return error.UnexpectedToken;
        self.pos += 1;
        try self.sql.append(self.gpa, '(');
        var first = true;
        while (true) {
            self.skipSpace();
            if (self.pos >= self.src.len) return error.UnexpectedEnd;
            if (self.src[self.pos] == ')') {
                self.pos += 1;
                break;
            }
            if (!first) {
                if (self.src[self.pos] != ',') return error.UnexpectedToken;
                self.pos += 1;
                try self.sql.appendSlice(self.gpa, ", ");
            }
            first = false;
            try self.parseValue(field);
        }
        try self.sql.append(self.gpa, ')');
    }

    fn parseValue(self: *Parser, field: records.Field) Error!void {
        self.skipSpace();
        if (self.pos >= self.src.len) return error.UnexpectedEnd;
        if (self.values.items.len >= max_bindings) return error.TooComplex;

        const raw = try self.readValueToken();

        // The column's declared type decides how the value is bound, not how the
        // value happens to look: binding "1000" as text against a REAL column
        // compares as text and `amount > 900` would then miss 1000.
        const v: Value = if (field.kind == .number) blk: {
            const n = std.fmt.parseFloat(f64, raw) catch return error.UnexpectedToken;
            break :blk .{ .number = n };
        } else .{ .text = raw };

        try self.values.append(self.gpa, v);
        try self.sql.print(self.gpa, "?{d}", .{self.values.items.len});
    }

    /// A bare word, or anything inside single or double quotes. Quoting is the
    /// only way to write a value containing a space or an operator character.
    fn readValueToken(self: *Parser) Error![]const u8 {
        const c = self.src[self.pos];
        if (c == '\'' or c == '"') {
            self.pos += 1;
            const start = self.pos;
            while (self.pos < self.src.len and self.src[self.pos] != c) self.pos += 1;
            if (self.pos >= self.src.len) return error.UnexpectedEnd;
            const out = self.src[start..self.pos];
            self.pos += 1;
            return out;
        }
        const start = self.pos;
        while (self.pos < self.src.len) : (self.pos += 1) {
            const ch = self.src[self.pos];
            if (ch == ' ' or ch == '\t' or ch == ',' or ch == ')' or ch == '(') break;
        }
        if (self.pos == start) return error.UnexpectedToken;
        return self.src[start..self.pos];
    }

    /// Field names may be quoted too, which is how a header containing a space
    /// is addressed.
    fn readFieldName(self: *Parser) Error![]const u8 {
        self.skipSpace();
        if (self.pos >= self.src.len) return error.UnexpectedEnd;
        const c = self.src[self.pos];
        if (c == '\'' or c == '"') return self.readValueToken();

        const start = self.pos;
        while (self.pos < self.src.len) : (self.pos += 1) {
            const ch = self.src[self.pos];
            if (ch == ' ' or ch == '\t' or ch == '=' or ch == '<' or ch == '>' or
                ch == '!' or ch == '(' or ch == ')' or ch == ',') break;
        }
        if (self.pos == start) return error.UnexpectedToken;
        return self.src[start..self.pos];
    }

    /// Only these operators exist. Anything else is refused rather than passed
    /// through — an operator written into the SQL text is an operator that was
    /// never checked.
    fn readOperator(self: *Parser) Error![]const u8 {
        self.skipSpace();
        if (self.pos >= self.src.len) return error.UnexpectedEnd;

        const two = if (self.pos + 2 <= self.src.len) self.src[self.pos..][0..2] else "";
        inline for (.{ "!=", "<=", ">=" }) |op| {
            if (std.mem.eql(u8, two, op)) {
                self.pos += 2;
                return if (std.mem.eql(u8, op, "!=")) "!=" else op;
            }
        }
        switch (self.src[self.pos]) {
            '=' => {
                self.pos += 1;
                return "=";
            },
            '<' => {
                self.pos += 1;
                return "<";
            },
            '>' => {
                self.pos += 1;
                return ">";
            },
            else => {},
        }
        if (self.eatKeyword("LIKE")) return "LIKE";
        if (self.eatKeyword("IN")) return "IN";
        if (self.eatKeyword("IS")) {
            self.skipSpace();
            if (self.eatKeyword("NOT")) {
                if (!self.eatKeyword("NULL")) return error.UnknownOperator;
                return "IS NOT NULL";
            }
            if (!self.eatKeyword("NULL")) return error.UnknownOperator;
            return "IS NULL";
        }
        return error.UnknownOperator;
    }

    /// Case-insensitive, and only when followed by a boundary — otherwise a
    /// column named `income` would have its `IN` eaten.
    fn eatKeyword(self: *Parser, comptime kw: []const u8) bool {
        self.skipSpace();
        if (self.pos + kw.len > self.src.len) return false;
        if (!std.ascii.eqlIgnoreCase(self.src[self.pos..][0..kw.len], kw)) return false;
        const after = self.pos + kw.len;
        if (after < self.src.len) {
            const c = self.src[after];
            if (std.ascii.isAlphanumeric(c) or c == '_') return false;
        }
        self.pos = after;
        return true;
    }
};

// ---------------------------------------------------------------------------
// --agg
// ---------------------------------------------------------------------------

pub const AggFn = enum { sum, avg, min, max, count };

pub const Agg = struct {
    func: AggFn,
    /// Empty for `count(*)`.
    field: []const u8,
    group_by: ?[]const u8,
};

/// `sum(amount) by category`, `count(*)`, `avg(amount)`.
pub fn parseAgg(schema: *const records.Schema, source: []const u8) Error!Agg {
    var s = std.mem.trim(u8, source, " \t");

    const open = std.mem.indexOfScalar(u8, s, '(') orelse return error.UnexpectedToken;
    const func = std.meta.stringToEnum(AggFn, lowerBuf(s[0..open])) orelse
        return error.UnknownOperator;

    const close = std.mem.indexOfScalarPos(u8, s, open, ')') orelse return error.UnexpectedEnd;
    const arg = std.mem.trim(u8, s[open + 1 .. close], " \t");

    var field: []const u8 = "";
    if (!std.mem.eql(u8, arg, "*")) {
        if (schema.find(arg) == null) return error.UnknownField;
        field = arg;
    } else if (func != .count) {
        // `sum(*)` has no meaning; only count works without a column.
        return error.UnexpectedToken;
    }

    s = std.mem.trim(u8, s[close + 1 ..], " \t");
    var group: ?[]const u8 = null;
    if (s.len != 0) {
        if (!std.ascii.startsWithIgnoreCase(s, "by")) return error.UnexpectedToken;
        const g = std.mem.trim(u8, s[2..], " \t");
        if (g.len == 0) return error.UnexpectedEnd;
        if (schema.find(g) == null) return error.UnknownField;
        group = g;
    }
    return .{ .func = func, .field = field, .group_by = group };
}

/// Lowercase into a fixed buffer; aggregate names are at most 5 characters.
fn lowerBuf(s: []const u8) []const u8 {
    const S = struct {
        threadlocal var buf: [8]u8 = undefined;
    };
    const trimmed = std.mem.trim(u8, s, " \t");
    if (trimmed.len > S.buf.len) return trimmed;
    for (trimmed, 0..) |c, i| S.buf[i] = std.ascii.toLower(c);
    return S.buf[0..trimmed.len];
}
