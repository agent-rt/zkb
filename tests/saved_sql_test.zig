//! 命名查询的解析与参数绑定。
//!
//! 这里锁住的两件事都是「不报错但答错」的类型：把没声明的参数绑成 null 会让查询返回空
//! 结果（读起来像「没有匹配」而不是「文件写错了」），而把数字绑成文本会让 LIMIT 和比较
//! 按字典序走——10 排在 9 前面。

const std = @import("std");
const zkb = @import("zkb");
const saved = zkb.saved_sql;
const sqlite = zkb.sqlite;

const testing = std.testing;
const gpa = std.testing.allocator;

test "parse：说明、参数、语句三段分开" {
    const body =
        \\-- 停滞项目：active 但没有下一步行动。
        \\-- 第二行说明也算。
        \\-- param: days = 7
        \\-- param: owner
        \\
        \\SELECT name FROM rec_projects
        \\WHERE date(review_at) <= date('now', '+' || :days || ' days')
        \\
    ;
    var q = try saved.parse(gpa, "stalled", body);
    defer q.deinit(gpa);

    try testing.expectEqualStrings("stalled", q.name);
    // 多行说明合成一行，`--` 去掉，param 声明不算说明。
    try testing.expectEqualStrings(
        "停滞项目：active 但没有下一步行动。 第二行说明也算。",
        q.description,
    );
    try testing.expectEqual(@as(usize, 2), q.params.len);
    try testing.expectEqualStrings("days", q.params[0].name);
    try testing.expectEqualStrings("7", q.params[0].default.?);
    // 没有 `=` 就是必填。
    try testing.expectEqualStrings("owner", q.params[1].name);
    try testing.expect(q.params[1].default == null);
    try testing.expect(std.mem.startsWith(u8, q.sql, "SELECT name FROM rec_projects"));
    // 结尾的空白和分号被裁掉，语句里不含说明。
    try testing.expect(std.mem.indexOf(u8, q.sql, "--") == null);
}

test "parse：只接受读语句" {
    try testing.expectError(error.NotSelect, saved.parse(gpa, "x", "DELETE FROM docs"));
    try testing.expectError(error.NotSelect, saved.parse(gpa, "x", "-- 说明\nUPDATE docs SET title = ''"));
    try testing.expectError(error.Empty, saved.parse(gpa, "x", "-- 只有说明，没有语句\n"));
    // 两条语句里第二条永远不会被显示，静默丢弃比报错糟。
    try testing.expectError(
        error.MultipleStatements,
        saved.parse(gpa, "x", "SELECT 1; SELECT 2"),
    );
    // WITH 和 EXPLAIN 是允许的。
    var w = try saved.parse(gpa, "x", "WITH t AS (SELECT 1 AS a) SELECT a FROM t");
    defer w.deinit(gpa);
}

test "isValidName：名字不能指向 queries 目录之外" {
    try testing.expect(saved.isValidName("stalled-projects"));
    try testing.expect(saved.isValidName("kb_health.v2"));
    try testing.expect(!saved.isValidName("../../etc/passwd"));
    try testing.expect(!saved.isValidName("/etc/passwd"));
    try testing.expect(!saved.isValidName(".hidden"));
    try testing.expect(!saved.isValidName(""));
    try testing.expect(!saved.isValidName("has space"));
}

fn openMem() !sqlite.Db {
    return sqlite.Db.open(":memory:", .read_write);
}

test "bindArgs：默认值、覆盖、必填缺失" {
    var db = try openMem();
    defer db.close();

    const body =
        \\-- 说明
        \\-- param: n = 2
        \\-- param: name
        \\
        \\SELECT :n AS n, :name AS name
        \\
    ;
    var q = try saved.parse(gpa, "probe", body);
    defer q.deinit(gpa);

    // 必填参数没给 → 报出是哪一个，而不是绑成 null 返回空结果。
    {
        var st = try db.prepare("SELECT :n AS n, :name AS name");
        defer st.finalize();
        var missing: ?[]const u8 = null;
        try testing.expectError(
            error.MissingParameter,
            saved.bindArgs(&st, &q, &.{}, &missing),
        );
        try testing.expectEqualStrings("name", missing.?);
    }

    // 给了必填、用默认的 n。
    {
        var st = try db.prepare("SELECT :n AS n, :name AS name");
        defer st.finalize();
        var missing: ?[]const u8 = null;
        try saved.bindArgs(&st, &q, &.{.{ "name", "alice" }}, &missing);
        try testing.expect(try st.step());
        try testing.expectEqual(@as(i64, 2), st.columnI64(0));
        try testing.expectEqualStrings("alice", st.columnText(1));
    }

    // 覆盖默认值。
    {
        var st = try db.prepare("SELECT :n AS n, :name AS name");
        defer st.finalize();
        var missing: ?[]const u8 = null;
        try saved.bindArgs(&st, &q, &.{ .{ "n", "9" }, .{ "name", "bob" } }, &missing);
        try testing.expect(try st.step());
        try testing.expectEqual(@as(i64, 9), st.columnI64(0));
    }
}

test "bindArgs：数字按整数绑定，不是文本" {
    var db = try openMem();
    defer db.close();

    var q = try saved.parse(gpa, "probe", "-- 说明\n-- param: n = 10\n\nSELECT :n AS v");
    defer q.deinit(gpa);

    var st = try db.prepare("SELECT typeof(:n) AS t, :n > 9 AS gt");
    defer st.finalize();
    var missing: ?[]const u8 = null;
    try saved.bindArgs(&st, &q, &.{}, &missing);
    try testing.expect(try st.step());
    // 绑成文本的话 typeof 是 'text'，且 '10' > 9 会按字典序比，结果不是 1。
    try testing.expectEqualStrings("integer", st.columnText(0));
    try testing.expectEqual(@as(i64, 1), st.columnI64(1));
}

test "bindArgs：语句用了未声明的参数要报错" {
    var db = try openMem();
    defer db.close();

    // 文件只声明了 n，语句里却还有 :other。绑不上就是 null，
    // 结果是「查不到」而不是「文件写错了」。
    var q = try saved.parse(gpa, "probe", "-- 说明\n-- param: n = 1\n\nSELECT :n, :other");
    defer q.deinit(gpa);

    var st = try db.prepare("SELECT :n AS a, :other AS b");
    defer st.finalize();
    var missing: ?[]const u8 = null;
    try testing.expectError(
        error.UndeclaredParameter,
        saved.bindArgs(&st, &q, &.{}, &missing),
    );
    try testing.expectEqualStrings("other", missing.?);
}
