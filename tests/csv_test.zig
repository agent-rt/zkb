//! CSV reading and writing.
//!
//! Every case here is one that a split-on-comma parser gets wrong *silently* —
//! the row count simply comes out different and nothing reports it.

const std = @import("std");
const zkb = @import("zkb");
const csv = zkb.csv;

const testing = std.testing;
const gpa = testing.allocator;

test "a comma inside quotes is content, not a separator" {
    var t = try csv.parse(gpa, "a,b\n\"one,two\",three\n");
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), t.rows.len);
    try testing.expectEqualStrings("one,two", t.rows[0][0]);
    try testing.expectEqualStrings("three", t.rows[0][1]);
}

test "a newline inside quotes is content, not a row break" {
    // The case a line-based parser cannot see at all: it counts two rows.
    var t = try csv.parse(gpa, "note,x\n\"line one\nline two\",2\n");
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), t.rows.len);
    try testing.expectEqualStrings("line one\nline two", t.rows[0][0]);
    try testing.expectEqualStrings("2", t.rows[0][1]);
}

test "a doubled quote is one literal quote" {
    var t = try csv.parse(gpa, "a\n\"he said \"\"hi\"\"\"\n");
    defer t.deinit(gpa);
    try testing.expectEqualStrings("he said \"hi\"", t.rows[0][0]);
}

test "fields stay valid after later fields grow the buffer" {
    // Fields are materialized from offsets, not taken as slices while the
    // backing buffer is still growing — otherwise every field before the last
    // reallocation dangles, and the corruption is data-dependent.
    var long: [4096]u8 = @splat('x');
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "a,b,c\nfirst,");
    try src.appendSlice(gpa, &long);
    try src.appendSlice(gpa, ",last\n");

    var t = try csv.parse(gpa, src.items);
    defer t.deinit(gpa);
    try testing.expectEqualStrings("first", t.rows[0][0]);
    try testing.expectEqualStrings("last", t.rows[0][2]);
    try testing.expectEqual(@as(usize, 4096), t.rows[0][1].len);
}

test "CRLF, trailing newline and blank lines" {
    var t = try csv.parse(gpa, "a,b\r\n1,2\r\n\r\n3,4\r\n");
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), t.rows.len);
    try testing.expectEqualStrings("2", t.rows[0][1]);
    try testing.expectEqualStrings("3", t.rows[1][0]);
}

test "a UTF-8 BOM does not become part of the first column name" {
    // Otherwise the first column never matches by name and reads as missing.
    var t = try csv.parse(gpa, "\xEF\xBB\xBFkey,value\nk,v\n");
    defer t.deinit(gpa);
    try testing.expectEqualStrings("key", t.header[0]);
    try testing.expectEqual(@as(?usize, 0), t.columnIndex("key"));
}

test "empty fields are preserved, not collapsed" {
    var t = try csv.parse(gpa, "a,b,c\n1,,3\n");
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), t.rows[0].len);
    try testing.expectEqualStrings("", t.rows[0][1]);
}

test "a row with the wrong field count is reported, never silently dropped" {
    // A person editing in a spreadsheet can break one row. Skipping it quietly
    // loses data and tells nobody.
    var t = try csv.parse(gpa, "a,b,c\n1,2,3\n4,5\n6,7,8\n");
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), t.rows.len);
    try testing.expectEqual(@as(usize, 1), t.bad_rows.len);
    try testing.expectEqual(@as(usize, 3), t.bad_rows[0]); // 1-based line
}

test "an unterminated quote is an error, not a truncated file" {
    try testing.expectError(error.UnterminatedQuote, csv.parse(gpa, "a\n\"never closed\n"));
}

test "an empty file yields nothing rather than failing" {
    var t = try csv.parse(gpa, "");
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), t.header.len);
    try testing.expectEqual(@as(usize, 0), t.rows.len);
}

test "CJK content round-trips byte for byte" {
    var t = try csv.parse(gpa, "key,note\nweight,\"体重 72.5kg，早上量的\"\n");
    defer t.deinit(gpa);
    try testing.expectEqualStrings("体重 72.5kg，早上量的", t.rows[0][1]);
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

test "writeRow quotes only what needs it, and survives a round trip" {
    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try csv.writeRow(&w, &.{ "plain", "has,comma", "has\"quote", "has\nnewline", "" });
    const line = w.buffered();

    try testing.expect(std.mem.startsWith(u8, line, "plain,"));
    try testing.expect(std.mem.indexOf(u8, line, "\"has,comma\"") != null);
    try testing.expect(std.mem.indexOf(u8, line, "\"has\"\"quote\"") != null);

    var full: std.ArrayList(u8) = .empty;
    defer full.deinit(gpa);
    try full.appendSlice(gpa, "a,b,c,d,e\n");
    try full.appendSlice(gpa, line);

    var t = try csv.parse(gpa, full.items);
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), t.rows.len);
    try testing.expectEqualStrings("has,comma", t.rows[0][1]);
    try testing.expectEqualStrings("has\"quote", t.rows[0][2]);
    try testing.expectEqualStrings("has\nnewline", t.rows[0][3]);
    try testing.expectEqualStrings("", t.rows[0][4]);
}

test "leading and trailing spaces survive only if quoted" {
    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try csv.writeRow(&w, &.{" padded "});
    try testing.expectEqualStrings("\" padded \"\n", w.buffered());
}
