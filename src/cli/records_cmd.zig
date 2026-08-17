//! `zkb records` and `zkb sql`.
//!
//! Output is TSV by default. Several rows of data cost roughly half the tokens
//! the same data would as JSON, and no agent has trouble reading it — a judgement
//! that several agent-facing tools have converged on.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub const Options = struct {
    type_name: ?[]const u8 = null,
    where: ?[]const u8 = null,
    search: ?[]const u8 = null,
    agg: ?[]const u8 = null,
    window: ?[]const u8 = null,
    limit: usize = 50,
    show_schema: bool = false,
    json: bool = false,
    model: ?[]const u8 = null,
};

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    opts: Options,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch |err| switch (err) {
        error.SchemaStale => {
            try w.writeAll("index schema is out of date\nrun: zkb index\n");
            return 3;
        },
        else => {
            try w.print("no index at {s}\nrun: zkb index\n", .{layout.db});
            return 3;
        },
    };
    defer db.close();

    const type_name = opts.type_name orelse return listTypes(gpa, &db, w);

    var schema = (try zkb.records.loadSchema(gpa, &db, type_name)) orelse {
        try w.print("no records type named {s}\n", .{type_name});
        _ = try listTypes(gpa, &db, w);
        return 3;
    };
    defer schema.deinit(gpa);

    if (opts.show_schema) return printSchema(w, &schema);

    // ---- WHERE, compiled once and reused by both paths below
    var where: ?zkb.expr.Compiled = null;
    defer if (where) |*c| c.deinit(gpa);
    if (opts.where) |src| {
        where = zkb.expr.compileWhere(gpa, &schema, src) catch |err| {
            try w.print("cannot parse --where: {t}\n", .{err});
            if (err == error.UnknownField) try printFieldNames(w, &schema);
            return 2;
        };
    }

    if (opts.agg) |src| return runAgg(gpa, &db, w, &schema, src, if (where) |c| &c else null);
    if (opts.window) |src| return runWindow(gpa, &db, w, &schema, src, if (where) |c| &c else null);

    // ---- semantic search, restricted to the rows the filter allows
    if (opts.search) |query| {
            return runSearch(gpa, io, env, &db, &layout, w, &schema, query, if (where) |c| &c else null, opts);
    }

    return runSelect(gpa, &db, w, &schema, if (where) |c| &c else null, opts);
}

fn listTypes(gpa: std.mem.Allocator, db: *zkb.sqlite.Db, w: *Writer) !u8 {
    const types = try zkb.records.listTypes(gpa, db);
    defer {
        for (types) |t| gpa.free(t);
        gpa.free(types);
    }
    if (types.len == 0) {
        try w.writeAll("no records types yet\n");
        try w.writeAll("create one: ~/.zkb/data/records/<type>/<file>.csv, then: zkb index\n");
        return 0;
    }
    try w.writeAll("type\trows\n");
    for (types) |t| {
        const name = try zkb.records.tableName(gpa, t);
        defer gpa.free(name);
        const q = try zkb.records.quoteIdent(gpa, name);
        defer gpa.free(q);
        const sql = try std.fmt.allocPrintSentinel(gpa, "SELECT count(*) FROM {s}", .{q}, 0);
        defer gpa.free(sql);
        const n = db.queryI64(sql) catch @as(?i64, null);
        try w.print("{s}\t{d}\n", .{ t, n orelse 0 });
    }
    return 0;
}

/// Inference has to be inspectable, or a wrong guess has no diagnosis path.
fn printSchema(w: *Writer, schema: *const zkb.records.Schema) !u8 {
    try w.print("{s}\n\n", .{schema.type_name});
    try w.writeAll("field\tkind\tindexed\tembedded\tsource\n");
    for (schema.fields) |f| {
        try w.print("{s}\t{s}\t{s}\t{s}\t{s}\n", .{
            f.name,
            @tagName(f.kind),
            if (f.kind.indexed()) "yes" else "-",
            if (f.kind.vectorized()) "yes" else "-",
            if (f.overridden) "_schema.json" else "inferred",
        });
    }
    return 0;
}

fn printFieldNames(w: *Writer, schema: *const zkb.records.Schema) !void {
    try w.writeAll("fields:");
    for (schema.fields) |f| try w.print(" {s}", .{f.name});
    try w.writeAll("\n");
}

fn bindCompiled(st: *zkb.sqlite.Stmt, c: *const zkb.expr.Compiled, first: i32) !void {
    for (c.values, 0..) |v, i| {
        const idx = first + @as(i32, @intCast(i));
        switch (v) {
            .text => |t| try st.bindText(idx, t),
            .number => |n| try st.bindF64(idx, n),
        }
    }
}

fn runSelect(
    gpa: std.mem.Allocator,
    db: *zkb.sqlite.Db,
    w: *Writer,
    schema: *const zkb.records.Schema,
    where: ?*const zkb.expr.Compiled,
    opts: Options,
) !u8 {
    var sql: std.ArrayList(u8) = .empty;
    defer sql.deinit(gpa);

    try sql.appendSlice(gpa, "SELECT ");
    for (schema.fields, 0..) |f, i| {
        if (i != 0) try sql.appendSlice(gpa, ", ");
        const q = try zkb.records.quoteIdent(gpa, f.name);
        defer gpa.free(q);
        try sql.appendSlice(gpa, q);
    }
    const table = try tableIdent(gpa, schema);
    defer gpa.free(table);
    try sql.print(gpa, " FROM {s}", .{table});
    if (where) |c| try sql.print(gpa, " WHERE {s}", .{c.sql});
    try sql.print(gpa, " LIMIT {d}", .{opts.limit});

    const text = try sql.toOwnedSliceSentinel(gpa, 0);
    defer gpa.free(text);
    var st = try db.prepare(text);
    defer st.finalize();
    if (where) |c| try bindCompiled(&st, c, 1);

    return emitRows(gpa, &st, w, schema, opts.json);
}

fn emitRows(
    gpa: std.mem.Allocator,
    st: *zkb.sqlite.Stmt,
    w: *Writer,
    schema: *const zkb.records.Schema,
    json: bool,
) !u8 {
    _ = gpa;
    if (json) try w.writeAll("[");

    var n: usize = 0;
    if (!json) {
        for (schema.fields, 0..) |f, i| {
            if (i != 0) try w.writeAll("\t");
            try w.writeAll(f.name);
        }
        try w.writeAll("\n");
    }

    while (try st.step()) {
        if (json) {
            if (n != 0) try w.writeAll(",");
            try w.writeAll("{");
            for (schema.fields, 0..) |f, i| {
                if (i != 0) try w.writeAll(",");
                try std.json.Stringify.value(f.name, .{}, w);
                try w.writeAll(":");
                try std.json.Stringify.value(st.columnText(@intCast(i)), .{}, w);
            }
            try w.writeAll("}");
        } else {
            for (schema.fields, 0..) |_, i| {
                if (i != 0) try w.writeAll("\t");
                // A tab inside a value would break the column alignment the
                // format depends on; a space keeps the row readable and parseable.
                try writeTsvCell(w, st.columnText(@intCast(i)));
            }
            try w.writeAll("\n");
        }
        n += 1;
    }

    if (json) try w.writeAll("]\n") else if (n == 0) try w.writeAll("(no rows)\n");
    return 0;
}

fn writeTsvCell(w: *Writer, text: []const u8) !void {
    for (text) |c| {
        try w.writeByte(if (c == '\t' or c == '\n' or c == '\r') ' ' else c);
    }
}

fn runAgg(
    gpa: std.mem.Allocator,
    db: *zkb.sqlite.Db,
    w: *Writer,
    schema: *const zkb.records.Schema,
    agg_src: []const u8,
    where: ?*const zkb.expr.Compiled,
) !u8 {
    const agg = zkb.expr.parseAgg(schema, agg_src) catch |err| {
        try w.print("cannot parse --agg: {t}\n", .{err});
        try w.writeAll("form: sum|avg|min|max|count(field) [by field]\n");
        if (err == error.UnknownField) try printFieldNames(w, schema);
        return 2;
    };

    const table = try tableIdent(gpa, schema);
    defer gpa.free(table);

    var sql: std.ArrayList(u8) = .empty;
    defer sql.deinit(gpa);

    try sql.appendSlice(gpa, "SELECT ");
    if (agg.group_by) |g| {
        const q = try zkb.records.quoteIdent(gpa, g);
        defer gpa.free(q);
        try sql.print(gpa, "{s}, ", .{q});
    }
    if (agg.field.len == 0) {
        try sql.appendSlice(gpa, "count(*)");
    } else {
        const q = try zkb.records.quoteIdent(gpa, agg.field);
        defer gpa.free(q);
        try sql.print(gpa, "{t}({s})", .{ agg.func, q });
    }
    try sql.print(gpa, " FROM {s}", .{table});
    if (where) |c| try sql.print(gpa, " WHERE {s}", .{c.sql});
    if (agg.group_by) |g| {
        const q = try zkb.records.quoteIdent(gpa, g);
        defer gpa.free(q);
        // Ordering by the aggregate puts the answer to "where does it all go"
        // at the top, which is the question a grouped aggregate usually is.
        try sql.print(gpa, " GROUP BY {s} ORDER BY 2 DESC", .{q});
    }

    const text = try sql.toOwnedSliceSentinel(gpa, 0);
    defer gpa.free(text);
    var st = try db.prepare(text);
    defer st.finalize();
    if (where) |c| try bindCompiled(&st, c, 1);

    if (agg.group_by) |g| try w.print("{s}\t", .{g});
    try w.print("{t}({s})\n", .{ agg.func, if (agg.field.len == 0) "*" else agg.field });
    while (try st.step()) {
        if (agg.group_by != null) {
            try writeTsvCell(w, st.columnText(0));
            try w.writeAll("\t");
            try w.print("{s}\n", .{st.columnText(1)});
        } else {
            try w.print("{s}\n", .{st.columnText(0)});
        }
    }
    return 0;
}

/// `avg(amount) over 7 by date` — a moving aggregate, one output row per input
/// row.
///
/// The window frame is `N PRECEDING` through `CURRENT ROW`, i.e. the N rows
/// ending here, which is what "7 day moving average" means to a person. SQLite
/// has done this since 3.25; the only new thing is letting a field name through
/// the same whitelist `--where` uses.
fn runWindow(
    gpa: std.mem.Allocator,
    db: *zkb.sqlite.Db,
    w: *Writer,
    schema: *const zkb.records.Schema,
    src: []const u8,
    where: ?*const zkb.expr.Compiled,
) !u8 {
    const win = zkb.expr.parseWindow(schema, src) catch |err| {
        try w.print("cannot parse --window: {t}\n", .{err});
        try w.writeAll("form: sum|avg|min|max|count(field) over N by field [partition field]\n");
        if (err == error.UnknownField) try printFieldNames(w, schema);
        return 2;
    };

    const table = try tableIdent(gpa, schema);
    defer gpa.free(table);
    const qfield = try zkb.records.quoteIdent(gpa, win.field);
    defer gpa.free(qfield);
    const qorder = try zkb.records.quoteIdent(gpa, win.order_by);
    defer gpa.free(qorder);

    var sql: std.ArrayList(u8) = .empty;
    defer sql.deinit(gpa);

    try sql.print(gpa, "SELECT {s}, {s}, {t}({s}) OVER (", .{ qorder, qfield, win.func, qfield });
    if (win.partition_by) |p| {
        const qp = try zkb.records.quoteIdent(gpa, p);
        defer gpa.free(qp);
        try sql.print(gpa, "PARTITION BY {s} ", .{qp});
    }
    try sql.print(gpa, "ORDER BY {s} ROWS BETWEEN {d} PRECEDING AND CURRENT ROW)", .{
        qorder, win.n - 1,
    });
    try sql.print(gpa, " FROM {s}", .{table});
    if (where) |c| try sql.print(gpa, " WHERE {s}", .{c.sql});
    try sql.print(gpa, " ORDER BY {s}", .{qorder});

    const text = try sql.toOwnedSliceSentinel(gpa, 0);
    defer gpa.free(text);
    var st = try db.prepare(text);
    defer st.finalize();
    if (where) |c| try bindCompiled(&st, c, 1);

    try w.print("{s}\t{s}\t{t}({s}) over {d}\n", .{
        win.order_by, win.field, win.func, win.field, win.n,
    });
    while (try st.step()) {
        try writeTsvCell(w, st.columnText(0));
        try w.writeAll("\t");
        try writeTsvCell(w, st.columnText(1));
        try w.writeAll("\t");
        try w.print("{s}\n", .{st.columnText(2)});
    }
    return 0;
}

/// `--search` combined with `--where`: **filter first, then KNN**.
///
/// The other order — take a large k and then drop rows that fail the filter — is
/// the classic bug in this kind of system. It loses results, and it loses them
/// unpredictably: if the 20 nearest rows are all outside the filter, the answer
/// is empty even though matching rows exist a little further out. sqlite-vec
/// 0.1.6 supports exactly one `rowid IN (...)` constraint on a KNN, which is
/// what makes the correct order expressible at all (SPEC §16.5).
fn runSearch(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    db: *zkb.sqlite.Db,
    layout: *const zkb.paths.Layout,
    w: *Writer,
    schema: *const zkb.records.Schema,
    query: []const u8,
    where: ?*const zkb.expr.Compiled,
    opts: Options,
) !u8 {
    const found = zkb.model_registry.resolve(gpa, io, env, layout, opts.model, .q8_0) catch {
        try w.writeAll("model not found\nrun: zkb model pull\n");
        return 4;
    };
    defer found.deinit(gpa);
    const model_path = found.path;

    var embedder = try zkb.embed.Embedder.init(gpa, model_path, .{});
    defer embedder.deinit();
    const vec = try gpa.alloc(f32, embedder.n_embd);
    defer gpa.free(vec);
    _ = try embedder.embedQuery(zkb.embed.default_query_task, query, vec);

    const table = try tableIdent(gpa, schema);
    defer gpa.free(table);

    var sql: std.ArrayList(u8) = .empty;
    defer sql.deinit(gpa);

    // The KNN needs its own bind slots after the filter's, so the filter's `?n`
    // numbering is kept and the vector and k follow it.
    const n_where: i32 = if (where) |c| @intCast(c.values.len) else 0;
    try sql.print(gpa,
        \\SELECT v.chunk_id, v.distance FROM vec_chunks v
        \\WHERE v.chunk_id IN (SELECT chunk_id FROM {s}
    , .{table});
    if (where) |c| try sql.print(gpa, " WHERE {s}", .{c.sql});
    try sql.print(gpa,
        \\)
        \\  AND v.embedding MATCH ?{d} AND k = ?{d}
    , .{ n_where + 1, n_where + 2 });

    const knn_sql = try sql.toOwnedSliceSentinel(gpa, 0);
    defer gpa.free(knn_sql);
    var st = db.prepare(knn_sql) catch {
        try w.print("query rejected: {s}\n", .{db.lastError()});
        return 1;
    };
    defer st.finalize();
    if (where) |c| try bindCompiled(&st, c, 1);
    try st.bindVector(n_where + 1, vec);
    try st.bindI64(n_where + 2, @intCast(opts.limit));

    var ids: std.ArrayList(i64) = .empty;
    defer ids.deinit(gpa);
    while (try st.step()) try ids.append(gpa, st.columnI64(0));

    if (ids.items.len == 0) {
        try w.writeAll("(no rows)\n");
        return 0;
    }

    // Hydrate in KNN order. One statement per row rather than an IN-list, so the
    // ranking survives — SQL has no ordering by list position.
    var sel: std.ArrayList(u8) = .empty;
    defer sel.deinit(gpa);
    try sel.appendSlice(gpa, "SELECT ");
    for (schema.fields, 0..) |f, i| {
        if (i != 0) try sel.appendSlice(gpa, ", ");
        const q = try zkb.records.quoteIdent(gpa, f.name);
        defer gpa.free(q);
        try sel.appendSlice(gpa, q);
    }
    try sel.print(gpa, " FROM {s} WHERE chunk_id = ?1", .{table});
    const sel_sql = try sel.toOwnedSliceSentinel(gpa, 0);
    defer gpa.free(sel_sql);

    for (schema.fields, 0..) |f, i| {
        if (i != 0) try w.writeAll("\t");
        try w.writeAll(f.name);
    }
    try w.writeAll("\n");

    for (ids.items) |id| {
        var row = try db.prepare(sel_sql);
        defer row.finalize();
        try row.bindI64(1, id);
        if (!try row.step()) continue;
        for (schema.fields, 0..) |_, i| {
            if (i != 0) try w.writeAll("\t");
            try writeTsvCell(w, row.columnText(@intCast(i)));
        }
        try w.writeAll("\n");
    }
    return 0;
}

fn tableIdent(gpa: std.mem.Allocator, schema: *const zkb.records.Schema) ![]u8 {
    const name = try zkb.records.tableName(gpa, schema.type_name);
    defer gpa.free(name);
    return zkb.records.quoteIdent(gpa, name);
}

// ---------------------------------------------------------------------------
// zkb sql
// ---------------------------------------------------------------------------

/// The escape hatch. A restricted expression language will always be unable to
/// say something — a join across two record types, a window function — and the
/// cost of having no escape hatch is an agent that simply gets stuck.
///
/// Safety comes from the connection being read-only, plus two gates: the
/// statement must read like a query, and SQLite itself must agree it writes
/// nothing. The residual risk is an agent misreading its own results, which is
/// not a risk this program can do anything about (SPEC §16.6).
pub const SqlOptions = struct {
    json: bool = false,
    /// `k=v` pairs for a saved query's parameters.
    args: []const [2][]const u8 = &.{},
};

pub fn sqlCmd(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    query: []const u8,
    opts: SqlOptions,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    // `@name` runs a saved query. Everything else is a statement typed on the
    // spot, which is what gets appended to the history.
    var loaded: ?zkb.saved_sql.Query = null;
    defer if (loaded) |*q| q.deinit(gpa);
    var statement = query;
    if (std.mem.startsWith(u8, query, "@")) {
        const name = query[1..];
        loaded = zkb.saved_sql.load(gpa, io, layout.queries, name) catch |err| {
            switch (err) {
                error.NotFound => {
                    try w.print("no saved query named {s}\n", .{name});
                    try w.print("looked in: {s}\n", .{layout.queries});
                    try w.writeAll("list them: zkb sql --list\n");
                },
                error.Empty => try w.print("{s}.sql has no statement\n", .{name}),
                error.NotSelect => try w.print("{s}.sql is not a SELECT / WITH / EXPLAIN\n", .{name}),
                error.MultipleStatements => try w.print("{s}.sql has more than one statement\n", .{name}),
                else => return err,
            }
            return 2;
        };
        statement = loaded.?.sql;
    }

    const trimmed = std.mem.trim(u8, statement, " \t\r\n;");
    if (!std.ascii.startsWithIgnoreCase(trimmed, "select") and
        !std.ascii.startsWithIgnoreCase(trimmed, "with") and
        !std.ascii.startsWithIgnoreCase(trimmed, "explain"))
    {
        try w.writeAll("only SELECT / WITH / EXPLAIN are accepted\n");
        return 2;
    }
    // One statement per call: a trailing `;` is fine, but anything after it is
    // a second statement that would never be shown.
    if (std.mem.indexOfScalar(u8, trimmed, ';') != null) {
        try w.writeAll("one statement per call\n");
        return 2;
    }

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);

    var db = zkb.store.open(db_path, .read_only) catch {
        try w.writeAll("no index yet\nrun: zkb index\n");
        return 3;
    };
    defer db.close();

    const sql = try gpa.dupeZ(u8, trimmed);
    defer gpa.free(sql);

    var st = db.prepare(sql) catch {
        try w.print("{s}\n", .{db.lastError()});
        return 2;
    };
    defer st.finalize();

    // Belt and braces: the connection is already read-only, but this catches a
    // statement that reads harmlessly and writes as a side effect.
    if (!st.isReadOnly()) {
        try w.writeAll("statement is not read-only\n");
        return 2;
    }

    if (loaded) |*q| {
        var missing: ?[]const u8 = null;
        zkb.saved_sql.bindArgs(&st, q, opts.args, &missing) catch |err| switch (err) {
            error.MissingParameter => {
                try w.print("{s} needs a value for {s}\n", .{ q.name, missing.? });
                try w.print("run: zkb sql @{s} {s}=<value>\n", .{ q.name, missing.? });
                return 2;
            },
            error.UndeclaredParameter => {
                // Binding nothing would leave it null and return no rows, which
                // reads as "no results" rather than as a mistake in the file.
                try w.print("{s}.sql uses :{s} but never declares it\n", .{ q.name, missing.? });
                try w.print("add to the file: -- param: {s} = <default>\n", .{missing.?});
                return 2;
            },
            else => return err,
        };
    } else {
        appendHistory(gpa, io, layout.sql_history, trimmed);
    }

    const n_cols = st.columnCount();
    if (opts.json) try w.writeAll("[");
    var rows: usize = 0;
    while (try st.step()) {
        if (opts.json) {
            if (rows != 0) try w.writeAll(",");
            try w.writeAll("{");
            var i: i32 = 0;
            while (i < n_cols) : (i += 1) {
                if (i != 0) try w.writeAll(",");
                try std.json.Stringify.value(st.columnName(i), .{}, w);
                try w.writeAll(":");
                try std.json.Stringify.value(st.columnText(i), .{}, w);
            }
            try w.writeAll("}");
        } else {
            if (rows == 0) {
                var i: i32 = 0;
                while (i < n_cols) : (i += 1) {
                    if (i != 0) try w.writeAll("\t");
                    try w.writeAll(st.columnName(i));
                }
                try w.writeAll("\n");
            }
            var i: i32 = 0;
            while (i < n_cols) : (i += 1) {
                if (i != 0) try w.writeAll("\t");
                try writeTsvCell(w, st.columnText(i));
            }
            try w.writeAll("\n");
        }
        rows += 1;
    }
    if (opts.json) try w.writeAll("]\n") else if (rows == 0) try w.writeAll("(no rows)\n");
    return 0;
}

/// Cap on the history file. It is disposable, but a file that grows without
/// bound is still a file someone has to notice.
const history_max_bytes: u64 = 4 * 1024 * 1024;

/// Append one ad-hoc statement. Never fails the query it is recording: a
/// history that can break `zkb sql` is worse than no history.
fn appendHistory(gpa: std.mem.Allocator, io: std.Io, path: []const u8, sql: []const u8) void {
    tryAppendHistory(gpa, io, path, sql) catch {};
}

fn tryAppendHistory(gpa: std.mem.Allocator, io: std.Io, path: []const u8, sql: []const u8) !void {
    var line: std.ArrayList(u8) = .empty;
    defer line.deinit(gpa);
    var ts_buf: [32]u8 = undefined;
    try line.appendSlice(gpa, "{\"at\":");
    try appendJsonString(gpa, &line, try isoStamp(&ts_buf, io));
    try line.appendSlice(gpa, ",\"sql\":");
    try appendJsonString(gpa, &line, sql);
    try line.appendSlice(gpa, "}\n");

    var file = try std.Io.Dir.createFileAbsolute(io, path, .{ .truncate = false });
    defer file.close(io);
    const end = (try file.stat(io)).size;
    if (end >= history_max_bytes) return;
    var buf: [4096]u8 = undefined;
    var fw = file.writer(io, &buf);
    fw.pos = end;
    try fw.interface.writeAll(line.items);
    try fw.interface.flush();
}

/// Same shape as the trace writer: a fixed buffer so a long statement cannot
/// allocate unboundedly, and a marker instead of dropping the line.
fn appendJsonString(gpa: std.mem.Allocator, out: *std.ArrayList(u8), v: []const u8) !void {
    var buf: [8192]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    std.json.Stringify.value(v, .{}, &w) catch {
        try out.appendSlice(gpa, "\"(too long)\"");
        return;
    };
    try out.appendSlice(gpa, w.buffered());
}

fn isoStamp(buf: []u8, io: std.Io) ![]const u8 {
    const secs: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .real).nanoseconds, std.time.ns_per_s));
    const epoch: std.time.epoch.EpochSeconds = .{ .secs = @intCast(secs) };
    const yd = epoch.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = epoch.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        yd.year,               md.month.numeric(),      md.day_index + 1,
        ds.getHoursIntoDay(),  ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(),
    });
}

/// `zkb sql --list`
pub fn sqlList(gpa: std.mem.Allocator, io: std.Io, env: *const std.process.Environ.Map, w: *Writer) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    const names = try zkb.saved_sql.list(gpa, io, layout.queries);
    defer {
        for (names) |n| gpa.free(n);
        gpa.free(names);
    }

    if (names.len == 0) {
        try w.print("no saved queries in {s}\n", .{layout.queries});
        try w.writeAll(
            \\
            \\a saved query is a .sql file there:
            \\
            \\  -- what it answers
            \\  -- param: days = 7
            \\  SELECT ... WHERE ... :days ...
            \\
            \\then: zkb sql @<name> [key=value ...]
            \\
        );
        return 0;
    }

    for (names) |n| {
        var q = zkb.saved_sql.load(gpa, io, layout.queries, n) catch {
            try w.print("{s}\t(unreadable)\n", .{n});
            continue;
        };
        defer q.deinit(gpa);
        try w.print("{s}\t{s}", .{ q.name, q.description });
        if (q.params.len != 0) {
            try w.writeAll("\t[");
            for (q.params, 0..) |p, i| {
                if (i != 0) try w.writeAll(" ");
                if (p.default) |d| try w.print("{s}={s}", .{ p.name, d }) else try w.print("{s}!", .{p.name});
            }
            try w.writeAll("]");
        }
        try w.writeAll("\n");
    }
    return 0;
}

/// `zkb sql --history [-n N]`
pub fn sqlHistory(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    limit: usize,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    const file = std.Io.Dir.openFileAbsolute(io, layout.sql_history, .{}) catch {
        try w.writeAll("no history yet\n");
        return 0;
    };
    defer file.close(io);
    const size = (try file.stat(io)).size;
    const body = try gpa.alloc(u8, @intCast(size));
    defer gpa.free(body);
    var rbuf: [8192]u8 = undefined;
    var r = file.readerStreaming(io, &rbuf);
    try r.interface.readSliceAll(body);

    // Newest last in the file, so collect the tail and print it oldest-first —
    // the useful order for spotting a statement you have now typed three times.
    var lines: std.ArrayList([]const u8) = .empty;
    defer lines.deinit(gpa);
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |l| if (std.mem.trim(u8, l, " \t\r").len != 0) try lines.append(gpa, l);

    const start = if (lines.items.len > limit) lines.items.len - limit else 0;
    for (lines.items[start..]) |l| {
        var parsed = std.json.parseFromSlice(std.json.Value, gpa, l, .{}) catch continue;
        defer parsed.deinit();
        const at = jsonStr(parsed.value, "at");
        const sql = jsonStr(parsed.value, "sql");
        try w.print("{s}\t{s}\n", .{ at, sql });
    }
    if (lines.items.len == 0) try w.writeAll("no history yet\n");
    return 0;
}

fn jsonStr(v: std.json.Value, key: []const u8) []const u8 {
    const o = switch (v) {
        .object => |o| o,
        else => return "",
    };
    return switch (o.get(key) orelse return "") {
        .string => |s| s,
        else => "",
    };
}
