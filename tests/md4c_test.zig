//! md4c 绑定：标题结构与字节偏移。
//!
//! 这些是行扫描器全部漏掉的构造。断言偏移落在正确的行首，因为 chunk 边界要靠它。

const std = @import("std");
const zkb = @import("zkb");
const md4c = zkb.md4c;

const testing = std.testing;
const gpa = std.testing.allocator;

fn lineAt(src: []const u8, off: usize) []const u8 {
    const end = std.mem.indexOfScalarPos(u8, src, off, '\n') orelse src.len;
    return src[off..end];
}

test "md4c 认出行扫描器漏掉的四种标题" {
    const src =
        "# ATX 一级\n\n正文。\n\n" ++
        "Setext 一级\n===========\n\n正文。\n\n" ++
        "Setext 二级\n-----------\n\n" ++
        "  ### 缩进三空格的 ATX\n\n" ++
        "<div>\n# HTML 块里的井号\n</div>\n\n" ++
        "> # 引用块里的标题\n\n" ++
        "## 日本語と中文の見出し\n\n最後の段落。\n";

    const hs = try md4c.headings(gpa, src);
    defer gpa.free(hs);

    // HTML 块里那个不是标题——行扫描器会把它当标题，还会重置标题路径。
    try testing.expectEqual(@as(usize, 6), hs.len);

    try testing.expectEqual(@as(u8, 1), hs[0].level);
    try testing.expectEqualStrings("# ATX 一级", lineAt(src, hs[0].byte_offset));

    // setext：偏移落在正文行，不是下划线行。
    try testing.expectEqual(@as(u8, 1), hs[1].level);
    try testing.expectEqualStrings("Setext 一级", lineAt(src, hs[1].byte_offset));
    try testing.expectEqual(@as(u8, 2), hs[2].level);
    try testing.expectEqualStrings("Setext 二级", lineAt(src, hs[2].byte_offset));

    // 缩进 1-3 空格的 ATX 是合法标题，偏移含缩进。
    try testing.expectEqual(@as(u8, 3), hs[3].level);
    try testing.expectEqualStrings("  ### 缩进三空格的 ATX", lineAt(src, hs[3].byte_offset));

    // 引用块内的标题带深度，调用方自己决定要不要计入标题栈。
    try testing.expectEqual(@as(u8, 1), hs[4].quote_depth);
    try testing.expectEqualStrings("> # 引用块里的标题", lineAt(src, hs[4].byte_offset));

    // CJK：偏移是字节，不是字符。
    try testing.expectEqual(@as(u8, 2), hs[5].level);
    try testing.expectEqual(@as(u8, 0), hs[5].quote_depth);
    try testing.expectEqualStrings("## 日本語と中文の見出し", lineAt(src, hs[5].byte_offset));
}

test "围栏内的井号不是标题" {
    const src = "# 真标题\n\n```\n# 围栏里的\n```\n\n~~~sh\n## 波浪号围栏里的\n~~~\n";
    const hs = try md4c.headings(gpa, src);
    defer gpa.free(hs);
    try testing.expectEqual(@as(usize, 1), hs.len);
    try testing.expectEqualStrings("# 真标题", lineAt(src, hs[0].byte_offset));
}

test "空输入与无标题文档" {
    const empty = try md4c.headings(gpa, "");
    defer gpa.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const plain = try md4c.headings(gpa, "只是一段文字。\n\n还有一段。\n");
    defer gpa.free(plain);
    try testing.expectEqual(@as(usize, 0), plain.len);
}

test "标题以实体开头时仍能定位到行首" {
    // 第一段可用文本是实体之后的部分，指针不在输入缓冲里；回扫仍应落到正确行首。
    const src = "# &amp; 与号开头\n\n正文。\n";
    const hs = try md4c.headings(gpa, src);
    defer gpa.free(hs);
    try testing.expectEqual(@as(usize, 1), hs.len);
    try testing.expectEqualStrings("# &amp; 与号开头", lineAt(src, hs[0].byte_offset));
}
