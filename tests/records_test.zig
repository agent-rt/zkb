//! Records: type inference and the `--where` compiler.
//!
//! Inference decides what gets indexed and what gets embedded, so a wrong guess
//! is silent — the column simply stops being findable one way or the other. The
//! compiler decides what reaches SQL, where a wrong guess is worse than silent.

const std = @import("std");
const zkb = @import("zkb");
const records = zkb.records;
const expr = zkb.expr;

const testing = std.testing;
const gpa = testing.allocator;

fn inferFrom(source: []const u8) !records.Schema {
    var table = try zkb.csv.parse(gpa, source);
    defer table.deinit(gpa);
    return records.infer(gpa, "t", &table);
}

fn kindOf(source: []const u8, field: []const u8) !records.Kind {
    var schema = try inferFrom(source);
    defer schema.deinit(gpa);
    return (schema.find(field) orelse return error.NotFound).kind;
}

// ---------------------------------------------------------------------------
// type inference
// ---------------------------------------------------------------------------

test "an ISO date column is a date, and a partial one is not" {
    try testing.expectEqual(records.Kind.date, try kindOf(
        "d\n2026-08-01\n2026-08-02\n2026-08-03\n",
        "d",
    ));
    // Year-month is not a date here: half a date sorts and compares wrong.
    try testing.expectEqual(records.Kind.string, try kindOf("d\n2026-08\n2026-09\n", "d"));
}

test "a blank cell is no evidence, so one blank does not break a date column" {
    try testing.expectEqual(records.Kind.date, try kindOf(
        "d\n2026-08-01\n\n2026-08-03\n",
        "d",
    ));
}

test "a leading zero means an identifier, not a number" {
    // The trap SPEC §16.3 lists as the reason `_schema.json` exists. Parsing
    // 0120345 as a number silently drops the zero and the value is unrecoverable.
    // Whether it lands in `string` or `id` is a display choice; `number` is the
    // one answer that loses data.
    try testing.expect(records.Kind.number != try kindOf(
        "zip\n0120345\n0150001\n0160002\n",
        "zip",
    ));
    const many = try kindOf(
        "zip\n0120345\n0150001\n0160002\n0170003\n0180004\n" ++
            "0190005\n0200006\n0210007\n0220008\n",
        "zip",
    );
    try testing.expectEqual(records.Kind.id, many);
    try testing.expect(!many.vectorized());
    try testing.expectEqual(records.Kind.number, try kindOf("n\n580\n12800\n320\n", "n"));
    // A decimal below one is a real number, not an identifier.
    try testing.expectEqual(records.Kind.number, try kindOf("n\n0.5\n0.25\n1.5\n", "n"));
    try testing.expectEqual(records.Kind.number, try kindOf("n\n-4\n12\n-0.5\n", "n"));
}

test "a repeated small vocabulary is an enum" {
    // 13 categories over 90 rows — ratio 0.144, which the original `< 0.1` rule
    // rejected. Measured against the real file that motivated the change.
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "c\n");
    for (0..90) |i| try src.print(gpa, "cat{d}\n", .{i % 13});

    try testing.expectEqual(records.Kind.@"enum", try kindOf(src.items, "c"));
}

test "free text is not an enum even when it repeats a little" {
    // Notes repeat by accident; categories repeat by design. The line between
    // them is how often, and erring towards `string` keeps text searchable.
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "note\n");
    for (0..20) |i| try src.print(gpa, "note number {d}\n", .{i % 15});

    try testing.expectEqual(records.Kind.string, try kindOf(src.items, "note"));
}

test "too few rows is not enough evidence for an enum" {
    // Three rows sharing a value proves nothing; the column stays string until
    // there is something to generalize from.
    try testing.expectEqual(records.Kind.string, try kindOf("c\nJPY\nJPY\nJPY\n", "c"));
}

test "a distinct short token containing a digit is an identifier, not prose" {
    // Embedding a SKU adds noise to the semantic neighbourhood and answers no
    // question — the same noise that embedding a whole record row produces.
    try testing.expectEqual(records.Kind.id, try kindOf(
        "id\nA_0001\nA_0002\nA_0003\nB_0001\nB_0002\nB_0003\nC_0001\nC_0002\nC_0003\n",
        "id",
    ));
}

test "distinct prose is still prose, and gets embedded" {
    // The guard that keeps shop names out of the identifier bucket: no digits.
    const kind = try kindOf(
        "name\nイオン\nマックスバリュ\n無印良品\nヨドバシカメラ\nローソン\n" ++
            "セブンイレブン\nユニクロ\nビックカメラ\nドトール\n",
        "name",
    );
    try testing.expectEqual(records.Kind.string, kind);
    try testing.expect(kind.vectorized());
}

test "an all-empty column defaults to string rather than to a guess" {
    try testing.expectEqual(records.Kind.string, try kindOf("a,b\n1,\n2,\n3,\n", "b"));
}

test "only free text is embedded; everything else is indexed" {
    var schema = try inferFrom(
        "date,amount,cat,note\n" ++
            "2026-08-01,100,food,coffee at the station\n" ++
            "2026-08-02,200,food,lunch with a colleague\n" ++
            "2026-08-03,300,home,a box for the shelf\n" ++
            "2026-08-04,400,food,dinner alone\n" ++
            "2026-08-05,500,home,another box\n" ++
            "2026-08-06,600,food,breakfast\n" ++
            "2026-08-07,700,home,lamp\n" ++
            "2026-08-08,800,food,tea\n",
    );
    defer schema.deinit(gpa);

    try testing.expect(!schema.find("date").?.kind.vectorized());
    try testing.expect(!schema.find("amount").?.kind.vectorized());
    try testing.expect(!schema.find("cat").?.kind.vectorized());
    try testing.expect(schema.find("note").?.kind.vectorized());

    try testing.expect(schema.find("date").?.kind.indexed());
    try testing.expect(schema.find("amount").?.kind.indexed());
    try testing.expect(!schema.find("note").?.kind.indexed());
}

test "the embedded text omits numbers, dates and categories" {
    var schema = try inferFrom(
        "date,amount,cat,note\n" ++
            "2026-08-01,580,food,便利店咖啡\n" ++
            "2026-08-02,320,food,饭团\n" ++
            "2026-08-03,300,home,收纳盒\n" ++
            "2026-08-04,400,food,晚饭\n" ++
            "2026-08-05,500,home,灯\n" ++
            "2026-08-06,600,food,早饭\n" ++
            "2026-08-07,700,home,椅子\n" ++
            "2026-08-08,800,food,茶\n",
    );
    defer schema.deinit(gpa);

    const row = [_][]const u8{ "2026-08-01", "580", "food", "便利店咖啡" };
    const text = try records.renderForEmbedding(gpa, &schema, &row);
    defer gpa.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "便利店咖啡") != null);
    try testing.expect(std.mem.indexOf(u8, text, "580") == null);
    // The date is a locator, not a quantity, so it stays as a prefix.
    try testing.expect(std.mem.indexOf(u8, text, "2026-08-01") != null);
}

// ---------------------------------------------------------------------------
// type name from path
// ---------------------------------------------------------------------------

test "the type comes from the directory, or from the file stem" {
    try testing.expectEqualStrings("expenses", records.typeOf("records/expenses/2026-08.csv").?);
    try testing.expectEqualStrings("weight", records.typeOf("records/weight.csv").?);
    try testing.expectEqual(@as(?[]const u8, null), records.typeOf("facts.csv"));
    try testing.expectEqual(@as(?[]const u8, null), records.typeOf("memory/a.md"));
    try testing.expectEqual(@as(?[]const u8, null), records.typeOf("records/"));
}

// ---------------------------------------------------------------------------
// identifier quoting
// ---------------------------------------------------------------------------

test "an identifier is quoted, and an embedded quote is doubled" {
    // Column names come from a file a person edits, so they are untrusted input
    // in exactly the way a value is — but a name cannot be a bound parameter.
    const q = try records.quoteIdent(gpa, "he\"re");
    defer gpa.free(q);
    try testing.expectEqualStrings("\"he\"\"re\"", q);

    const cjk = try records.quoteIdent(gpa, "类别");
    defer gpa.free(cjk);
    try testing.expectEqualStrings("\"类别\"", cjk);
}

// ---------------------------------------------------------------------------
// --where
// ---------------------------------------------------------------------------

fn testSchema() !records.Schema {
    return inferFrom(
        "date,amount,cat,note\n" ++
            "2026-08-01,580,food,coffee\n" ++
            "2026-08-02,320,food,rice\n" ++
            "2026-08-03,300,home,box\n" ++
            "2026-08-04,400,food,dinner\n" ++
            "2026-08-05,500,home,lamp\n" ++
            "2026-08-06,600,food,breakfast\n" ++
            "2026-08-07,700,home,chair\n" ++
            "2026-08-08,800,food,tea\n",
    );
}

test "a value never appears in the generated SQL" {
    // The property the whole design rests on. If a value can reach the SQL text
    // then quoting has to be right, and quoting is never right for long.
    var schema = try testSchema();
    defer schema.deinit(gpa);

    var c = try expr.compileWhere(gpa, &schema, "cat = 'foo' OR note LIKE '%1=1%'");
    defer c.deinit(gpa);

    try testing.expect(std.mem.indexOf(u8, c.sql, "foo") == null);
    try testing.expect(std.mem.indexOf(u8, c.sql, "1=1") == null);
    try testing.expectEqual(@as(usize, 2), c.values.len);
    try testing.expectEqualStrings("foo", c.values[0].text);
}

test "an unknown field is refused rather than passed through" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    try testing.expectError(
        error.UnknownField,
        expr.compileWhere(gpa, &schema, "secret = 1"),
    );
    // Including when it is spelled to look like SQL.
    try testing.expectError(
        error.UnknownField,
        expr.compileWhere(gpa, &schema, "1=1 OR cat = 'x'"),
    );
}

test "AND binds tighter than OR" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    var c = try expr.compileWhere(gpa, &schema, "cat = a OR cat = b AND amount > 5");
    defer c.deinit(gpa);
    // Rendered flat, but grouped by construction: the AND pair is parsed as one
    // operand of the OR, so SQLite's own precedence agrees with the parse.
    try testing.expect(std.mem.indexOf(u8, c.sql, "OR") != null);
    try testing.expectEqual(@as(usize, 3), c.values.len);
}

test "parentheses group explicitly" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    var c = try expr.compileWhere(gpa, &schema, "(cat = a OR cat = b) AND amount > 5");
    defer c.deinit(gpa);
    try testing.expect(std.mem.startsWith(u8, c.sql, "("));
}

test "a number column binds a number, not the text of one" {
    // Binding "1000" as text against a REAL column compares as text, and
    // `amount > 900` would then silently miss 1000.
    var schema = try testSchema();
    defer schema.deinit(gpa);
    var c = try expr.compileWhere(gpa, &schema, "amount > 1000");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(f64, 1000), c.values[0].number);
}

test "a non-numeric value against a number column is an error" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    try testing.expectError(
        error.UnexpectedToken,
        expr.compileWhere(gpa, &schema, "amount > lots"),
    );
}

test "IN takes a list, and each element is a binding" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    var c = try expr.compileWhere(gpa, &schema, "cat IN (food, home)");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), c.values.len);
    try testing.expect(std.mem.indexOf(u8, c.sql, "?1") != null);
    try testing.expect(std.mem.indexOf(u8, c.sql, "?2") != null);
}

test "IS NULL takes no operand" {
    // The only way to ask about an empty cell: `= ''` cannot match a NULL, and
    // empty cells are stored as NULL.
    var schema = try testSchema();
    defer schema.deinit(gpa);
    var c = try expr.compileWhere(gpa, &schema, "note IS NULL");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.values.len);
    try testing.expect(std.mem.endsWith(u8, c.sql, "IS NULL"));
}

test "an unknown operator is refused" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    try testing.expectError(
        error.UnknownOperator,
        expr.compileWhere(gpa, &schema, "cat GLOB x"),
    );
}

test "a keyword is only a keyword at a boundary" {
    // A column called `income` must not have its `IN` eaten.
    var table = try zkb.csv.parse(gpa, "income\n1\n2\n");
    defer table.deinit(gpa);
    var schema = try records.infer(gpa, "t", &table);
    defer schema.deinit(gpa);

    var c = try expr.compileWhere(gpa, &schema, "income > 1");
    defer c.deinit(gpa);
    try testing.expect(std.mem.startsWith(u8, c.sql, "\"income\""));
}

test "a quoted value may contain spaces and operators" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    var c = try expr.compileWhere(gpa, &schema, "note = 'a > b and c'");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.values.len);
    try testing.expectEqualStrings("a > b and c", c.values[0].text);
}

test "an empty expression is an error, not a match-everything" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    try testing.expectError(error.EmptyExpression, expr.compileWhere(gpa, &schema, "   "));
}

test "trailing junk is refused rather than ignored" {
    // Silently ignoring the tail would mean a typo quietly widens the result.
    var schema = try testSchema();
    defer schema.deinit(gpa);
    try testing.expectError(
        error.UnexpectedToken,
        expr.compileWhere(gpa, &schema, "cat = food garbage"),
    );
}

// ---------------------------------------------------------------------------
// --agg
// ---------------------------------------------------------------------------

test "agg parses a function, a field and an optional grouping" {
    var schema = try testSchema();
    defer schema.deinit(gpa);

    const a = try expr.parseAgg(&schema, "sum(amount) by cat");
    try testing.expectEqual(expr.AggFn.sum, a.func);
    try testing.expectEqualStrings("amount", a.field);
    try testing.expectEqualStrings("cat", a.group_by.?);

    const b = try expr.parseAgg(&schema, "count(*)");
    try testing.expectEqual(expr.AggFn.count, b.func);
    try testing.expectEqualStrings("", b.field);
    try testing.expectEqual(@as(?[]const u8, null), b.group_by);
}

test "agg refuses an unknown field, in either position" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    try testing.expectError(error.UnknownField, expr.parseAgg(&schema, "sum(nope)"));
    try testing.expectError(error.UnknownField, expr.parseAgg(&schema, "sum(amount) by nope"));
}

test "sum(*) is meaningless and is refused" {
    var schema = try testSchema();
    defer schema.deinit(gpa);
    try testing.expectError(error.UnexpectedToken, expr.parseAgg(&schema, "sum(*)"));
}
