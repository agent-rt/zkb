//! md4c 绑定，只取一件事：标题结构。
//!
//! 行扫描器（markdown.zig）在 CommonMark 官方 655 个例子上，标题栈只对 37%——它不认
//! setext 标题、不认缩进 1-3 空格的 ATX 标题，也不知道 HTML 块里的 `#` 不是标题（后者
//! 更糟：它会重置标题路径，让后面的段落挂到一个不存在的标题下）。
//!
//! 而 `heading_path` 会被前置进 embedding、写进 FTS 列、显示在搜索结果里，所以这不是
//! 排版细节。
//!
//! ## 为什么偏移能拿到
//!
//! md4c 是 SAX 式的，`enter_block` **不给字节偏移**——这一点让它看起来不满足 zkb 的溯源
//! 要求。但 `text` 回调给的是指向输入缓冲的指针（对 MD_TEXT_NORMAL 而言），所以标题的
//! 位置可以反推：取标题里第一个落在缓冲内的 text 指针，回扫到行首，就是含 `##` 标记的
//! 起始字节。实测过 ATX / setext / 缩进 ATX / 引用块内 / CJK 五种，全部拿到。
//!
//! 实体和部分规范化后的文本指向静态字符串而不是输入，所以要判断指针是否落在缓冲内——
//! 一个 `# &amp; foo` 这样的标题，第一段可用文本是 `foo`，回扫仍然落到正确的行首。
//!
//! 块的打包继续交给行扫描器：它对段落、围栏、表格的判断是好的，而且刚修好表头与超长行。

const std = @import("std");

pub const c = @cImport({
    @cInclude("md4c.h");
});

pub const Heading = struct {
    /// 1-6
    level: u8,
    /// 标题所在行的起始字节（含 `##` 标记或 setext 的正文行）
    byte_offset: usize,
    /// 嵌套在几层引用块里。0 = 文档正文。
    quote_depth: u8,
};

const Collector = struct {
    gpa: std.mem.Allocator,
    source: []const u8,
    out: std.ArrayList(Heading),
    err: ?anyerror = null,

    in_heading: bool = false,
    level: u8 = 0,
    quote_depth: u8 = 0,
    offset: ?usize = null,
};

fn enterBlock(kind: c.MD_BLOCKTYPE, detail: ?*anyopaque, userdata: ?*anyopaque) callconv(.c) c_int {
    const self: *Collector = @ptrCast(@alignCast(userdata.?));
    switch (kind) {
        c.MD_BLOCK_QUOTE => self.quote_depth +|= 1,
        c.MD_BLOCK_H => {
            const d: *c.MD_BLOCK_H_DETAIL = @ptrCast(@alignCast(detail.?));
            self.in_heading = true;
            self.level = @intCast(d.level);
            self.offset = null;
        },
        else => {},
    }
    return 0;
}

fn leaveBlock(kind: c.MD_BLOCKTYPE, _: ?*anyopaque, userdata: ?*anyopaque) callconv(.c) c_int {
    const self: *Collector = @ptrCast(@alignCast(userdata.?));
    switch (kind) {
        c.MD_BLOCK_QUOTE => self.quote_depth -|= 1,
        c.MD_BLOCK_H => {
            self.in_heading = false;
            const text_at = self.offset orelse return 0;
            // 回扫到行首，得到含标记的起始字节。setext 标题的第一个 text 指针落在正文行
            // 上，所以拿到的是正文行首而不是下划线行 —— 正是想要的。
            var line = text_at;
            while (line > 0 and self.source[line - 1] != '\n') line -= 1;
            self.out.append(self.gpa, .{
                .level = self.level,
                .byte_offset = line,
                .quote_depth = self.quote_depth,
            }) catch |e| {
                self.err = e;
            };
        },
        else => {},
    }
    return 0;
}

fn onText(
    kind: c.MD_TEXTTYPE,
    text: [*c]const c.MD_CHAR,
    size: c.MD_SIZE,
    userdata: ?*anyopaque,
) callconv(.c) c_int {
    const self: *Collector = @ptrCast(@alignCast(userdata.?));
    if (!self.in_heading or self.offset != null) return 0;
    if (kind != c.MD_TEXT_NORMAL and kind != c.MD_TEXT_CODE) return 0;
    _ = size;

    // 实体与部分规范化文本指向静态字符串，不在输入缓冲里；那种指针换算出来是垃圾偏移。
    const ptr = @intFromPtr(text);
    const base = @intFromPtr(self.source.ptr);
    if (ptr < base or ptr >= base + self.source.len) return 0;
    self.offset = ptr - base;
    return 0;
}

/// md4c 只把 `debug_log` 标为可空；span 回调没有这个标注，也不做判空，留 null 会在遇到
/// 第一个行内元素（链接、强调、行内代码）时直接段错误。标题结构不需要 span，但回调必须
/// 存在。
fn noopSpan(_: c.MD_SPANTYPE, _: ?*anyopaque, _: ?*anyopaque) callconv(.c) c_int {
    return 0;
}

pub const Error = error{ParseFailed} || std.mem.Allocator.Error;

/// 文档里的所有标题，按出现顺序。
pub fn headings(gpa: std.mem.Allocator, source: []const u8) Error![]Heading {
    var collector: Collector = .{ .gpa = gpa, .source = source, .out = .empty };
    errdefer collector.out.deinit(gpa);

    var parser: c.MD_PARSER = std.mem.zeroes(c.MD_PARSER);
    parser.abi_version = 0;
    // GFM：表格与删除线。zkb 的语料里表格很常见，而 md4c 只在这个方言下把它们当块。
    parser.flags = c.MD_DIALECT_GITHUB;
    parser.enter_block = enterBlock;
    parser.leave_block = leaveBlock;
    parser.text = onText;
    parser.enter_span = noopSpan;
    parser.leave_span = noopSpan;

    if (source.len == 0) return collector.out.toOwnedSlice(gpa);

    const rc = c.md_parse(source.ptr, @intCast(source.len), &parser, &collector);
    if (collector.err) |e| return @errorCast(e);
    if (rc != 0) return error.ParseFailed;
    return collector.out.toOwnedSlice(gpa);
}
