//! Markdown scanning and chunking boundaries.
//!
//! These are the cases where getting it wrong quietly degrades retrieval rather
//! than throwing: a `#` inside a code fence read as a heading, a fence split
//! across two chunks, a heading path that loses its parent.

const std = @import("std");
const zkb = @import("zkb");
const markdown = zkb.markdown;
const chunk = zkb.chunk;

const testing = std.testing;
const gpa = testing.allocator;

fn scan(src: []const u8) !markdown.Document {
    return markdown.scan(gpa, src);
}

test "frontmatter is separated from the body" {
    const src =
        \\---
        \\node_type: wiki
        \\tags: [a, b]
        \\---
        \\
        \\# Title
        \\
        \\Body text.
        \\
    ;
    var doc = try scan(src);
    defer doc.deinit(gpa);

    try testing.expect(doc.frontmatter != null);
    try testing.expect(std.mem.indexOf(u8, doc.frontmatter.?, "node_type: wiki") != null);
    // The closing fence must not leak into the frontmatter.
    try testing.expect(std.mem.indexOf(u8, doc.frontmatter.?, "Title") == null);
    try testing.expectEqualStrings("Title", doc.title.?);
}

test "unterminated frontmatter is treated as body, not swallowed" {
    const src =
        \\---
        \\this never closes
        \\
        \\# Heading
        \\
    ;
    var doc = try scan(src);
    defer doc.deinit(gpa);
    try testing.expect(doc.frontmatter == null);
    try testing.expect(doc.blocks.len > 0);
}

test "hashes inside a code fence are not headings" {
    const src =
        \\# Real Heading
        \\
        \\```bash
        \\# this is a shell comment, not a heading
        \\## neither is this
        \\```
        \\
        \\Tail paragraph.
        \\
    ;
    var doc = try scan(src);
    defer doc.deinit(gpa);

    try testing.expectEqual(@as(usize, 1), doc.headings.len);
    try testing.expectEqualStrings("Real Heading", doc.title.?);

    var saw_code = false;
    for (doc.blocks) |b| if (b.kind == .code) {
        saw_code = true;
    };
    try testing.expect(saw_code);
}

test "atx heading requires a space: #hashtag is not a heading" {
    const src =
        \\#hashtag
        \\
        \\# Actual
        \\
    ;
    var doc = try scan(src);
    defer doc.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), doc.headings.len);
    try testing.expectEqualStrings("Actual", doc.title.?);
}

test "tilde fences and long fences close correctly" {
    const src =
        \\~~~
        \\```not a close
        \\~~~
        \\
        \\After.
        \\
    ;
    var doc = try scan(src);
    defer doc.deinit(gpa);

    var code_blocks: usize = 0;
    for (doc.blocks) |b| if (b.kind == .code) {
        code_blocks += 1;
    };
    try testing.expectEqual(@as(usize, 1), code_blocks);
}

test "table is one block including its delimiter row" {
    const src =
        \\| a | b |
        \\|---|---|
        \\| 1 | 2 |
        \\| 3 | 4 |
        \\
        \\After.
        \\
    ;
    var doc = try scan(src);
    defer doc.deinit(gpa);

    var tables: usize = 0;
    var table_text: []const u8 = &.{};
    for (doc.blocks) |b| if (b.kind == .table) {
        tables += 1;
        table_text = src[b.byte_start..b.byte_end];
    };
    try testing.expectEqual(@as(usize, 1), tables);
    try testing.expect(std.mem.indexOf(u8, table_text, "| 3 | 4 |") != null);
}

test "heading path carries the full ancestor chain" {
    const src =
        \\# Doc
        \\
        \\## Section
        \\
        \\### Sub
        \\
        \\Deep paragraph.
        \\
    ;
    var doc = try scan(src);
    defer doc.deinit(gpa);

    const last = doc.blocks.len - 1;
    const path = try doc.headingPathAt(gpa, src, last);
    defer gpa.free(path);
    try testing.expectEqualStrings("Doc > Section > Sub", path);
}

test "heading path drops deeper levels when a shallower heading appears" {
    const src =
        \\# Doc
        \\
        \\## First
        \\
        \\### Deep
        \\
        \\## Second
        \\
        \\Paragraph under Second.
        \\
    ;
    var doc = try scan(src);
    defer doc.deinit(gpa);

    const last = doc.blocks.len - 1;
    const path = try doc.headingPathAt(gpa, src, last);
    defer gpa.free(path);
    // "Deep" belonged to "First" and must not linger.
    try testing.expectEqualStrings("Doc > Second", path);
}

// ---------------------------------------------------------------------------
// chunking
// ---------------------------------------------------------------------------

test "chunks respect the hard token limit and stay ordered" {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "# Doc\n\n");
    for (0..60) |i| {
        try buf.print(gpa, "## Section {d}\n\nSome prose for section {d}. " ++
            "It is long enough to matter when packing chunks together.\n\n", .{ i, i });
    }

    var doc = try scan(buf.items);
    defer doc.deinit(gpa);

    var bc: chunk.ByteCounter = .{};
    var chunks = try chunk.split(gpa, buf.items, &doc, bc.counter(), .{});
    defer chunks.deinit(gpa);

    try testing.expect(chunks.items.len > 1);
    for (chunks.items, 0..) |c, n| {
        try testing.expect(c.n_tokens <= 1024);
        try testing.expectEqual(n, c.idx);
        try testing.expect(c.byte_end > c.byte_start);
        // Heading path must always name the document.
        try testing.expect(std.mem.startsWith(u8, c.heading_path, "Doc"));
    }
}

test "a code fence is never split across chunks" {
    // A fence that fits under max_tokens must survive packing intact.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "# Doc\n\n");
    for (0..30) |i| try buf.print(gpa, "Filler paragraph number {d} to push the packer along.\n\n", .{i});
    try buf.appendSlice(gpa, "```zig\nconst a = 1;\nconst b = 2;\nconst c = 3;\n```\n\n");
    for (0..30) |i| try buf.print(gpa, "Trailing paragraph {d}.\n\n", .{i});

    var doc = try scan(buf.items);
    defer doc.deinit(gpa);

    var bc: chunk.ByteCounter = .{};
    var chunks = try chunk.split(gpa, buf.items, &doc, bc.counter(), .{});
    defer chunks.deinit(gpa);

    // Every chunk containing the fence body must contain both delimiters.
    var holding: usize = 0;
    for (chunks.items) |c| {
        if (std.mem.indexOf(u8, c.text, "const b = 2;") != null) {
            holding += 1;
            try testing.expect(std.mem.indexOf(u8, c.text, "```zig") != null);
            try testing.expect(std.mem.count(u8, c.text, "```") >= 2);
        }
    }
    try testing.expect(holding >= 1);
}

test "an oversized code block is line-split and marked as continued" {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "# Doc\n\n## Big\n\n```text\n");
    for (0..2000) |i| try buf.print(gpa, "line {d} of a very long generated block\n", .{i});
    try buf.appendSlice(gpa, "```\n");

    var doc = try scan(buf.items);
    defer doc.deinit(gpa);

    var bc: chunk.ByteCounter = .{};
    var chunks = try chunk.split(gpa, buf.items, &doc, bc.counter(), .{});
    defer chunks.deinit(gpa);

    try testing.expect(chunks.items.len > 2);
    for (chunks.items) |c| try testing.expect(c.n_tokens <= 1024);

    var continued: usize = 0;
    for (chunks.items) |c| if (std.mem.indexOf(u8, c.heading_path, "(cont.") != null) {
        continued += 1;
    };
    try testing.expect(continued > 0);
}

test "chunking terminates on degenerate input" {
    // Overlap larger than target could stall the walk; the strict progress rule
    // in split() is what makes this terminate.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    for (0..40) |i| try buf.print(gpa, "Paragraph {d}.\n\n", .{i});

    var doc = try scan(buf.items);
    defer doc.deinit(gpa);

    var bc: chunk.ByteCounter = .{};
    var chunks = try chunk.split(gpa, buf.items, &doc, bc.counter(), .{
        .target_tokens = 10,
        .max_tokens = 20,
        .overlap_tokens = 100,
    });
    defer chunks.deinit(gpa);

    try testing.expect(chunks.items.len > 0);
    try testing.expect(chunks.items.len < 500); // no runaway
}

test "empty and whitespace-only documents produce no chunks" {
    for ([_][]const u8{ "", "\n\n\n", "   \n\t\n" }) |src| {
        var doc = try scan(src);
        defer doc.deinit(gpa);
        var bc: chunk.ByteCounter = .{};
        var chunks = try chunk.split(gpa, src, &doc, bc.counter(), .{});
        defer chunks.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), chunks.items.len);
    }
}

test "一行超过上限时按字符边界切开，不整行发出" {
    // 一段没有任何换行的长文本。按行切到这里就没得切了，旧实现整行发出，
    // 结果是 chunk 超过 max_tokens——真实语料上出现过 2 个，最大 1318。
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "# Doc\n\n");
    // 中日文，一个字三字节：确认切点落在字符边界上，而不是把字劈开。
    for (0..2000) |_| try buf.appendSlice(gpa, "这是一段很长的中文内容");
    try buf.appendSlice(gpa, "\n");

    var doc = try scan(buf.items);
    defer doc.deinit(gpa);

    var bc: chunk.ByteCounter = .{};
    var chunks = try chunk.split(gpa, buf.items, &doc, bc.counter(), .{});
    defer chunks.deinit(gpa);

    try testing.expect(chunks.items.len > 1);
    for (chunks.items) |c| {
        try testing.expect(c.n_tokens <= 1024);
        // 切点必须是合法 UTF-8，否则写进 FTS 列就是另一个故障。
        try testing.expect(std.unicode.utf8ValidateSlice(c.text));
        try testing.expect(c.byte_end > c.byte_start);
    }

    // 覆盖率：分段应当基本铺满源文本，只差首尾修剪掉的空白。
    var covered: usize = 0;
    for (chunks.items) |c| covered += c.byte_end - c.byte_start;
    try testing.expect(covered + 64 >= buf.items.len);
}

test "被切开的表格，续块的标题路径带上列名" {
    // 表头只在第一段里。续段是一堆没有列名的行——对读者和 embedding 都近乎无意义。
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "# Doc\n\n## 现行\n\n| 路径 | 内容 | 何时该读 |\n|---|---|---|\n");
    for (0..200) |i| {
        try buf.print(gpa, "| path/to/file-{d}.md | 这一行的说明足够长，用来把表格推过分块上限 | 需要的时候 |\n", .{i});
    }

    var doc = try scan(buf.items);
    defer doc.deinit(gpa);

    var bc: chunk.ByteCounter = .{};
    var chunks = try chunk.split(gpa, buf.items, &doc, bc.counter(), .{});
    defer chunks.deinit(gpa);

    var conts: usize = 0;
    for (chunks.items) |c| {
        if (std.mem.indexOf(u8, c.heading_path, "(cont.") == null) continue;
        conts += 1;
        // 列名必须在标题路径里——它在嵌入时会被前置到文本前面。
        try testing.expect(std.mem.indexOf(u8, c.heading_path, "路径 | 内容 | 何时该读") != null);
        // 分隔行不带信息，不该出现。
        try testing.expect(std.mem.indexOf(u8, c.heading_path, "---") == null);
        // 仍然要能看出它属于哪个小节。
        try testing.expect(std.mem.indexOf(u8, c.heading_path, "现行") != null);
    }
    try testing.expect(conts > 0);

    // 第一段自己就含表头，不该重复。
    for (chunks.items) |c| {
        if (std.mem.indexOf(u8, c.heading_path, "(cont.") != null) continue;
        if (std.mem.indexOf(u8, c.text, "| 路径 |") == null) continue;
        try testing.expect(std.mem.indexOf(u8, c.heading_path, "路径 | 内容") == null);
    }
}
