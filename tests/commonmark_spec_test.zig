//! CommonMark 官方测试集上的标题栈符合率。
//!
//! zkb 是公开发布的通用工具，所以不能拿一个人的语料当总体：真实世界里会遇到的是
//! CommonMark 全集。这里量的是 `heading_path` 依赖的那一件事——标题栈，因为它会被
//! 前置进 embedding、写进 FTS 列、显示在搜索结果里。
//!
//! 基准取 spec 期望 HTML 里的 `<h1>..<h6>`。比对做规范化：spec 的 HTML 把 `*bar*`
//! 渲染成 `bar`，而 zkb 在标题路径里保留原文（嵌入时原文更有用），所以两侧都去掉行内
//! 标记再比。剩下的差异就是真正的结构性错误。
//!
//! 这个测试断言的是**下限**而不是 100%：它的作用是把当前水平钉住，让换解析器这件事有
//! 可验收的标准，也让退步立刻可见。

const std = @import("std");
const zkb = @import("zkb");
const markdown = zkb.markdown;

const testing = std.testing;
const gpa = std.testing.allocator;

const spec = @embedFile("fixtures/commonmark-spec.txt");

/// spec 用 32 个反引号包住每个例子，中间用 `.` 单独一行分开输入与期望输出。
const fence = "`" ** 32;

const Example = struct {
    markdown: []const u8,
    html: []const u8,
};

fn nextExample(src: []const u8, cursor: *usize) ?Example {
    const open = std.mem.indexOfPos(u8, src, cursor.*, fence ++ " example\n") orelse return null;
    const md_start = open + (fence ++ " example\n").len;
    const sep = std.mem.indexOfPos(u8, src, md_start, "\n.\n") orelse return null;
    const html_start = sep + 3;
    const close = std.mem.indexOfPos(u8, src, html_start, "\n" ++ fence ++ "\n") orelse return null;
    cursor.* = close + 1;
    return .{
        .markdown = src[md_start .. sep + 1],
        .html = src[html_start .. close + 1],
    };
}

/// 去掉行内标记与转义，折叠空白，转小写。两侧都过一遍。
fn normalize(out: []u8, text: []const u8) []const u8 {
    var n: usize = 0;
    var prev_space = true;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const ch = text[i];
        switch (ch) {
            '*', '_', '`', '\\', '#' => continue,
            ' ', '\t', '\n', '\r' => {
                if (prev_space) continue;
                if (n >= out.len) break;
                out[n] = ' ';
                n += 1;
                prev_space = true;
            },
            else => {
                if (n >= out.len) break;
                out[n] = std.ascii.toLower(ch);
                n += 1;
                prev_space = false;
            },
        }
    }
    return std.mem.trim(u8, out[0..n], " ");
}

const Heading = struct { level: u8, text: []const u8 };

/// spec 期望 HTML 里的标题。标签去掉，只还原 spec 实际用到的五个实体。
fn expectedHeadings(list: *std.ArrayList(Heading), buf: *std.ArrayList(u8), html: []const u8) !void {
    var i: usize = 0;
    while (i < html.len) {
        const lt = std.mem.indexOfPos(u8, html, i, "<h") orelse break;
        if (lt + 3 >= html.len or !std.ascii.isDigit(html[lt + 2]) or html[lt + 3] != '>') {
            i = lt + 2;
            continue;
        }
        const level = html[lt + 2] - '0';
        const body_start = lt + 4;
        var close_buf: [8]u8 = undefined;
        const close = try std.fmt.bufPrint(&close_buf, "</h{c}>", .{html[lt + 2]});
        const body_end = std.mem.indexOfPos(u8, html, body_start, close) orelse break;

        // 去标签
        const start = buf.items.len;
        var in_tag = false;
        var j = body_start;
        while (j < body_end) : (j += 1) {
            const ch = html[j];
            if (ch == '<') {
                in_tag = true;
            } else if (ch == '>') {
                in_tag = false;
            } else if (!in_tag) {
                if (ch == '&') {
                    const ents = [_]struct { k: []const u8, v: u8 }{
                        .{ .k = "&amp;", .v = '&' },   .{ .k = "&lt;", .v = '<' },
                        .{ .k = "&gt;", .v = '>' },    .{ .k = "&quot;", .v = '"' },
                        .{ .k = "&#39;", .v = '\'' },
                    };
                    var matched = false;
                    for (ents) |e| {
                        if (std.mem.startsWith(u8, html[j..body_end], e.k)) {
                            try buf.append(gpa, e.v);
                            j += e.k.len - 1;
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) try buf.append(gpa, ch);
                } else {
                    try buf.append(gpa, ch);
                }
            }
        }
        try list.append(gpa, .{ .level = level, .text = buf.items[start..] });
        i = body_end + close.len;
    }
}

test "CommonMark 官方测试集：标题栈符合率" {
    var cursor: usize = 0;
    var total: usize = 0;
    var with_headings: usize = 0;
    var ok: usize = 0;
    var by_design: usize = 0;

    var norm_a: [512]u8 = undefined;
    var norm_b: [512]u8 = undefined;

    while (nextExample(spec, &cursor)) |ex| {
        total += 1;

        var exp: std.ArrayList(Heading) = .empty;
        defer exp.deinit(gpa);
        var exp_buf: std.ArrayList(u8) = .empty;
        defer exp_buf.deinit(gpa);
        // 先攒满再取切片：append 会搬动底层缓冲，中途持有的切片会失效。
        try expectedHeadings(&exp, &exp_buf, ex.html);

        if (exp.items.len == 0) continue;
        with_headings += 1;

        // spec 用 → 表示制表符。
        const md = try replaceArrows(gpa, ex.markdown);
        defer gpa.free(md);

        // 例子里的标题是否都被 `>` 包着。
        var quoted = true;
        {
            var lines = std.mem.splitScalar(u8, md, '\n');
            var saw = false;
            while (lines.next()) |l| {
                const t = std.mem.trim(u8, l, " \t\r");
                if (t.len == 0) continue;
                if (std.mem.indexOfScalar(u8, t, '#') == null) continue;
                saw = true;
                if (t[0] != '>') quoted = false;
            }
            if (!saw) quoted = false;
        }

        var doc = try markdown.scan(gpa, md);
        defer doc.deinit(gpa);

        // 比的是 doc.headings，不是 .heading 块：setext 标题不会成为块（扫描器把正文
        // 看成段落、下划线看成分隔线），但它必须出现在标题栈里，而标题栈才是
        // heading_path 的来源。
        var matched = doc.headings.len == exp.items.len;
        if (matched) {
            for (doc.headings, exp.items) |h, e| {
                const a = normalize(&norm_a, std.mem.trim(u8, md[h.text_start..h.text_end], " \t\r\n"));
                const b_text = normalize(&norm_b, e.text);
                if (h.level != e.level or !std.mem.eql(u8, a, b_text)) {
                    matched = false;
                    break;
                }
            }
        }
        if (matched) {
            ok += 1;
        } else if (quoted) {
            // 有意排除：引用块里的标题是被引用的内容，不是这篇文档的结构。让它改变后续
            // 段落的标题路径是错的，所以 rebuildHeadings 按 quote_depth 跳过它们。
            by_design += 1;
        }
    }

    const pct = ok * 100 / with_headings;
    const intended = with_headings - by_design;
    const pct_intended = ok * 100 / intended;
    std.debug.print(
        "\nCommonMark {d} 例，{d} 例期望有标题\n" ++
            "  符合 {d}（{d}%）\n" ++
            "  其中 {d} 例是有意排除的引用块标题；按 zkb 的意图算 {d}/{d}（{d}%）\n",
        .{ total, with_headings, ok, pct, by_design, ok, intended, pct_intended },
    );

    // spec 0.31.2 有 652 个例子；数字变了说明夹具没解析对，而不是实现变好了。
    try testing.expect(total >= 640);
    try testing.expect(with_headings >= 35);

    // 下限。接 md4c 之前是 37%（不认 setext、不认缩进 ATX、把 HTML 块里的 `#` 当标题）。
    // 现在 80%，剩下的差额里一半是有意排除的引用块标题。
    // 这个下限的作用是让退步立刻可见，所以订在刚好低于当前水平。
    try testing.expect(pct >= 75);
    try testing.expect(pct_intended >= 85);
}

fn replaceArrows(a: std.mem.Allocator, src: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < src.len) {
        if (std.mem.startsWith(u8, src[i..], "→")) {
            try out.append(a, '\t');
            i += "→".len;
        } else {
            try out.append(a, src[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}
