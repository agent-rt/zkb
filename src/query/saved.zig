//! Named SQL queries, stored as files.
//!
//! `zkb sql` answers questions the restricted `--where` / `--agg` grammar cannot
//! say. In practice the same handful of statements get rewritten from memory
//! every time, which is how a query that took an afternoon to get right becomes
//! a query nobody runs.
//!
//! A saved query is a `.sql` file under `data/queries/`. Three consequences of
//! it being a file rather than a row:
//!
//! - It lives in the precious half of the layout. Putting them in the index —
//!   or making them real sqlite views — would mean `zkb index` deletes them.
//! - It is editable with the tools everything else here is edited with, so
//!   there is no `--save` command to learn, and no writer to keep consistent.
//! - It gets indexed, so `zkb search` finds the query itself. Asking "how did I
//!   count stalled projects" is the same kind of question as any other.
//!
//! ## Format
//!
//! ```sql
//! -- Stalled projects: active, with no next action.
//! -- param: days = 7
//!
//! SELECT name FROM rec_projects
//! WHERE date(review_at) <= date('now', '+' || :days || ' days')
//! ```
//!
//! Leading `--` comments are the description. `-- param: name = default` declares
//! a parameter; without a default it is required. Everything after the comment
//! block is the statement.
//!
//! Parameters are bound through sqlite, never pasted in. String interpolation is
//! what makes a saved query a way to run arbitrary sql with someone else's input
//! in it.

const std = @import("std");
const sqlite = @import("../db/sqlite.zig");

pub const max_bytes = 64 * 1024;

pub const Param = struct {
    name: []const u8,
    /// null = required
    default: ?[]const u8,
};

pub const Query = struct {
    name: []const u8,
    description: []const u8,
    sql: []const u8,
    params: []Param,
    /// Backing buffer for every slice above.
    source: []u8,

    pub fn deinit(self: *Query, gpa: std.mem.Allocator) void {
        gpa.free(self.params);
        gpa.free(self.name);
        gpa.free(self.source);
        self.* = undefined;
    }
};

pub const Error = error{
    NotFound,
    Empty,
    NotSelect,
    MultipleStatements,
} || std.mem.Allocator.Error;

/// Parse the file body. `name` and `source` are duped; the rest points into
/// `source`, so it stays valid until deinit.
pub fn parse(gpa: std.mem.Allocator, name: []const u8, body: []const u8) Error!Query {
    const source = try gpa.dupe(u8, body);
    errdefer gpa.free(source);

    var params: std.ArrayList(Param) = .empty;
    errdefer params.deinit(gpa);

    var desc_end: usize = 0;
    var sql_start: usize = 0;
    var desc_lines: usize = 0;

    var it = std.mem.splitScalar(u8, source, '\n');
    var offset: usize = 0;
    while (it.next()) |line| {
        const line_start = offset;
        offset += line.len + 1;
        const t = std.mem.trim(u8, line, " \t\r");
        if (t.len == 0) {
            // A blank line inside the header is fine; the statement starts at
            // the first line that is neither blank nor a comment.
            continue;
        }
        if (!std.mem.startsWith(u8, t, "--")) {
            sql_start = line_start;
            break;
        }
        const rest = std.mem.trim(u8, t[2..], " \t");
        if (std.mem.startsWith(u8, rest, "param:")) {
            const decl = std.mem.trim(u8, rest[6..], " \t");
            const eq = std.mem.indexOfScalar(u8, decl, '=');
            const pname = std.mem.trim(u8, if (eq) |e| decl[0..e] else decl, " \t");
            if (pname.len == 0) continue;
            try params.append(gpa, .{
                .name = pname,
                .default = if (eq) |e| std.mem.trim(u8, decl[e + 1 ..], " \t") else null,
            });
        } else {
            // Description is the comment lines that are not declarations.
            if (desc_lines == 0) desc_end = line_start;
            desc_end = line_start + line.len;
            desc_lines += 1;
        }
    } else {
        // Ran out of lines without finding a statement.
        sql_start = source.len;
    }

    const sql = std.mem.trim(u8, source[sql_start..], " \t\r\n;");
    if (sql.len == 0) return error.Empty;
    if (!std.ascii.startsWithIgnoreCase(sql, "select") and
        !std.ascii.startsWithIgnoreCase(sql, "with") and
        !std.ascii.startsWithIgnoreCase(sql, "explain")) return error.NotSelect;
    if (std.mem.indexOfScalar(u8, sql, ';') != null) return error.MultipleStatements;

    // Strip the leading `--` from each description line, joined by spaces.
    var desc: []const u8 = "";
    if (desc_lines != 0) {
        const raw = source[0..desc_end];
        var out: usize = 0;
        var dit = std.mem.splitScalar(u8, raw, '\n');
        while (dit.next()) |line| {
            const t = std.mem.trim(u8, line, " \t\r");
            if (!std.mem.startsWith(u8, t, "--")) continue;
            const c = std.mem.trim(u8, t[2..], " \t");
            if (std.mem.startsWith(u8, c, "param:")) continue;
            if (c.len == 0) continue;
            if (out != 0 and out < source.len) {
                source[out] = ' ';
                out += 1;
            }
            // Rewriting in place inside `source` is safe: the description always
            // sits before the statement, and each rewritten line is shorter than
            // the comment it came from.
            std.mem.copyForwards(u8, source[out..][0..c.len], c);
            out += c.len;
        }
        desc = source[0..out];
    }

    return .{
        .name = try gpa.dupe(u8, name),
        .description = desc,
        .sql = sql,
        .params = try params.toOwnedSlice(gpa),
        .source = source,
    };
}

/// Read `<dir>/<name>.sql`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !Query {
    if (!isValidName(name)) return error.NotFound;
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.sql", .{ dir, name });
    defer gpa.free(path);

    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.NotFound;
    defer file.close(io);
    const size = (try file.stat(io)).size;
    if (size > max_bytes) return error.Empty;

    const buf = try gpa.alloc(u8, @intCast(size));
    defer gpa.free(buf);
    var rbuf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &rbuf);
    try r.interface.readSliceAll(buf);

    return parse(gpa, name, buf);
}

/// Names of every saved query, sorted. Missing directory means none.
pub fn list(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![][]u8 {
    var names: std.ArrayList([]u8) = .empty;
    errdefer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }

    var d = std.Io.Dir.openDirAbsolute(io, dir, .{ .iterate = true }) catch return names.toOwnedSlice(gpa);
    defer d.close(io);

    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".sql")) continue;
        const stem = entry.name[0 .. entry.name.len - 4];
        if (!isValidName(stem)) continue;
        try names.append(gpa, try gpa.dupe(u8, stem));
    }

    const out = try names.toOwnedSlice(gpa);
    std.mem.sort([]u8, out, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);
    return out;
}

/// Keeps a name from reaching outside the queries directory, and keeps `zkb sql
/// @…` unambiguous.
pub fn isValidName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64) return false;
    for (name) |ch| {
        if (std.ascii.isAlphanumeric(ch) or ch == '-' or ch == '_' or ch == '.') continue;
        return false;
    }
    // `.` is allowed inside a name but a leading dot or `..` is not a name.
    if (name[0] == '.') return false;
    return true;
}

/// Bind `k=v` arguments and declared defaults onto a prepared statement.
///
/// Values are bound as integers when they parse as one, so `:days` works in
/// arithmetic and in `LIMIT`, where a text value would compare as text and
/// silently order 10 before 9.
pub fn bindArgs(
    st: *sqlite.Stmt,
    q: *const Query,
    args: []const [2][]const u8,
    missing: *?[]const u8,
) !void {
    for (q.params) |p| {
        var value: ?[]const u8 = p.default;
        for (args) |kv| {
            if (std.mem.eql(u8, kv[0], p.name)) value = kv[1];
        }
        const v = value orelse {
            missing.* = p.name;
            return error.MissingParameter;
        };
        const idx = st.parameterIndex(p.name) orelse continue;
        if (std.fmt.parseInt(i64, v, 10)) |n| {
            try st.bindI64(idx, n);
        } else |_| {
            try st.bindText(idx, v);
        }
    }

    // A parameter in the statement that no comment declares would otherwise
    // bind to null and quietly return nothing.
    const n = st.parameterCount();
    var i: c_int = 1;
    while (i <= n) : (i += 1) {
        const raw = st.parameterName(i) orelse continue;
        const pname = if (raw.len > 0 and (raw[0] == ':' or raw[0] == '@' or raw[0] == '$'))
            raw[1..]
        else
            raw;
        var declared = false;
        for (q.params) |p| {
            if (std.mem.eql(u8, p.name, pname)) declared = true;
        }
        if (!declared) {
            missing.* = pname;
            return error.UndeclaredParameter;
        }
    }
}
