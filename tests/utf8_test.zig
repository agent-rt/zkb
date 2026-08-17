//! Cutting CJK text by byte count.
//!
//! Every one of these was reachable in production: excerpts, labels and previews
//! are all bounded in bytes because their destination is — a column, a line, a
//! report row. Two of the four call sites had this wrong, and the symptom was
//! invalid UTF-8 written into `chunks.heading_path`, which FTS then indexes.

const std = @import("std");
const zkb = @import("zkb");
const utf8 = zkb.utf8;

const testing = std.testing;

test "a cut landing inside a character backs up to the boundary" {
    // 混 合 检 索 — three bytes each, so a 5-byte limit lands mid-character.
    const s = "混合检索";
    try testing.expectEqual(@as(usize, 12), s.len);

    try testing.expectEqual(@as(usize, 3), utf8.truncate(s, 5));
    try testing.expectEqual(@as(usize, 3), utf8.truncate(s, 4));
    try testing.expectEqual(@as(usize, 3), utf8.truncate(s, 3));
    try testing.expectEqual(@as(usize, 0), utf8.truncate(s, 2));
    try testing.expectEqual(@as(usize, 6), utf8.truncate(s, 7));
}

test "no cut is made when the text already fits" {
    // Returning `text.len` rather than the limit is what lets a caller tell
    // "truncated" from "fits" and decide whether to add an ellipsis.
    const s = "混合检索";
    try testing.expectEqual(s.len, utf8.truncate(s, 12));
    try testing.expectEqual(s.len, utf8.truncate(s, 999));
    try testing.expectEqual(@as(usize, 0), utf8.truncate("", 10));
}

test "every prefix of a CJK string cuts to valid UTF-8" {
    // The property that matters, stated directly: whatever limit is chosen,
    // what comes out can be read back as text.
    const s = "株式会社日本橋高島屋三越伊勢丹百貨店銀座本館第二別館";
    var limit: usize = 0;
    while (limit <= s.len + 4) : (limit += 1) {
        try testing.expect(utf8.isValid(utf8.cut(s, limit)));
    }
}

test "mixed scripts, including four-byte sequences" {
    // Emoji are four bytes; a naive back-up of one byte would still be wrong.
    const s = "zkb 检索 🔍 done";
    var limit: usize = 0;
    while (limit <= s.len) : (limit += 1) {
        try testing.expect(utf8.isValid(utf8.cut(s, limit)));
    }
}

test "ASCII is unaffected" {
    try testing.expectEqualStrings("hello", utf8.cut("hello world", 5));
    try testing.expectEqualStrings("hello world", utf8.cut("hello world", 99));
}

test "the record label a 64-byte cut produces stays valid" {
    // The exact shape of the bug: a 25-character Japanese shop name is 75 bytes,
    // and `rowLabel` cuts at 64 — landing inside the 22nd character.
    const name = "株式会社日本橋高島屋三越伊勢丹百貨店銀座本館第二別館";
    try testing.expect(name.len > 64);
    const label = utf8.cut(name, 64);
    try testing.expect(utf8.isValid(label));
    try testing.expect(label.len <= 64);
    // And it really did have to back up — 64 is not a multiple of 3.
    try testing.expectEqual(@as(usize, 63), label.len);
}
