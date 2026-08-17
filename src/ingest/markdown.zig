//! Line-driven Markdown block scanner.
//!
//! Not a Markdown parser and not trying to be: chunking only needs block
//! boundaries and the heading stack. Zig has no mature Markdown AST library and
//! building one here would be a much larger surface than the job requires.
//!
//! What it must get right, because getting it wrong damages retrieval:
//!   - fenced code blocks are opaque (a `#` inside a fence is not a heading,
//!     and a fence must never be split across chunks)
//!   - the heading stack, since chunk text is prefixed with the heading path
//!     before embedding (SPEC §3.3)
//!   - byte offsets, so a hit can be traced back to the source file

const std = @import("std");
const md4c = @import("md4c.zig");

pub const BlockKind = enum {
    heading,
    paragraph,
    code,
    table,
    list,
    rule,
    blank,
};

pub const Block = struct {
    kind: BlockKind,
    byte_start: usize,
    byte_end: usize,
    /// Heading level 1-6, only meaningful for `.heading`.
    level: u8 = 0,
    /// Depth of the heading stack in effect for this block.
    heading_depth: u8 = 0,
};

pub const Document = struct {
    /// Raw frontmatter text without the `---` fences, or null when absent.
    frontmatter: ?[]const u8,
    /// First H1 if there is one, else null. Callers fall back to the filename.
    title: ?[]const u8,
    blocks: []Block,
    /// Heading text per (depth, block) is reconstructed on demand by
    /// `headingPathAt`; storing a string per block would duplicate the source.
    headings: []Heading,

    pub const Heading = struct {
        level: u8,
        text_start: usize,
        text_end: usize,
        /// Index into `blocks` where this heading appeared.
        block_index: usize,
    };

    pub fn deinit(self: *Document, gpa: std.mem.Allocator) void {
        gpa.free(self.blocks);
        gpa.free(self.headings);
        self.* = undefined;
    }

    /// "Title > Section > Subsection" for the heading stack in effect at
    /// `block_index`. Result is owned by the caller.
    pub fn headingPathAt(
        self: *const Document,
        gpa: std.mem.Allocator,
        source: []const u8,
        block_index: usize,
    ) std.mem.Allocator.Error![]u8 {
        // Walk headings that precede this block, keeping the last one seen at
        // each level and dropping deeper levels when a shallower one appears.
        var stack: [7]?Document.Heading = @splat(null);
        for (self.headings) |h| {
            if (h.block_index > block_index) break;
            stack[h.level] = h;
            var deeper: u8 = h.level + 1;
            while (deeper <= 6) : (deeper += 1) stack[deeper] = null;
        }

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        var level: u8 = 1;
        while (level <= 6) : (level += 1) {
            const h = stack[level] orelse continue;
            if (out.items.len != 0) try out.appendSlice(gpa, " > ");
            try out.appendSlice(gpa, std.mem.trim(u8, source[h.text_start..h.text_end], " \t"));
        }
        return out.toOwnedSlice(gpa);
    }
};

/// Scan `source` into blocks. Borrows `source`; all offsets index into it.
pub fn scan(gpa: std.mem.Allocator, source: []const u8) std.mem.Allocator.Error!Document {
    var blocks: std.ArrayList(Block) = .empty;
    errdefer blocks.deinit(gpa);
    var headings: std.ArrayList(Document.Heading) = .empty;
    errdefer headings.deinit(gpa);

    var pos: usize = 0;
    var frontmatter: ?[]const u8 = null;
    var title: ?[]const u8 = null;
    var heading_depth: u8 = 0;

    // Frontmatter only counts at the very start of the file.
    if (std.mem.startsWith(u8, source, "---\n") or std.mem.startsWith(u8, source, "---\r\n")) {
        const body_start = std.mem.indexOfScalar(u8, source, '\n').? + 1;
        if (findClosingFrontmatter(source, body_start)) |close| {
            frontmatter = source[body_start..close.content_end];
            pos = close.next_line;
        }
    }

    const body_offset = pos;

    while (pos < source.len) {
        const line_end = lineEnd(source, pos);
        const line = trimEol(source[pos..line_end]);
        const next = if (line_end < source.len) line_end + 1 else source.len;

        if (isBlank(line)) {
            pos = next;
            continue;
        }

        // Fenced code: consume through the closing fence. Everything inside is
        // opaque — a `#` in a shell snippet is not a heading.
        if (fenceMarker(line)) |fence| {
            const start = pos;
            var p = next;
            while (p < source.len) {
                const le = lineEnd(source, p);
                const l = trimEol(source[p..le]);
                p = if (le < source.len) le + 1 else source.len;
                if (closesFence(l, fence)) break;
            }
            try blocks.append(gpa, .{
                .kind = .code,
                .byte_start = start,
                .byte_end = p,
                .heading_depth = heading_depth,
            });
            pos = p;
            continue;
        }

        if (headingLevel(line)) |level| {
            const text_off = level + 1; // "### " -> skip hashes and one space
            const h: Document.Heading = .{
                .level = level,
                .text_start = pos + @min(text_off, line.len),
                .text_end = pos + line.len,
                .block_index = blocks.items.len,
            };
            try headings.append(gpa, h);
            if (level == 1 and title == null) title = source[h.text_start..h.text_end];
            heading_depth = level;
            try blocks.append(gpa, .{
                .kind = .heading,
                .byte_start = pos,
                .byte_end = line_end,
                .level = level,
                .heading_depth = heading_depth,
            });
            pos = next;
            continue;
        }

        if (isRule(line)) {
            try blocks.append(gpa, .{
                .kind = .rule,
                .byte_start = pos,
                .byte_end = line_end,
                .heading_depth = heading_depth,
            });
            pos = next;
            continue;
        }

        // Table: a `|` row followed by a delimiter row. Consume the whole table
        // so it is never split down the middle.
        if (isTableRow(line) and isTableDelimiter(peekLine(source, next))) {
            const start = pos;
            var p = next;
            while (p < source.len) {
                const le = lineEnd(source, p);
                const l = trimEol(source[p..le]);
                if (!isTableRow(l)) break;
                p = if (le < source.len) le + 1 else source.len;
            }
            try blocks.append(gpa, .{
                .kind = .table,
                .byte_start = start,
                .byte_end = p,
                .heading_depth = heading_depth,
            });
            pos = p;
            continue;
        }

        // Otherwise a paragraph or list run: to the next blank line, heading,
        // fence, or table start.
        {
            const start = pos;
            const kind: BlockKind = if (isListItem(line)) .list else .paragraph;
            var p = next;
            while (p < source.len) {
                const le = lineEnd(source, p);
                const l = trimEol(source[p..le]);
                if (isBlank(l) or headingLevel(l) != null or fenceMarker(l) != null or isRule(l)) break;
                p = if (le < source.len) le + 1 else source.len;
            }
            try blocks.append(gpa, .{
                .kind = kind,
                .byte_start = start,
                .byte_end = p,
                .heading_depth = heading_depth,
            });
            pos = p;
        }
    }

    const out_blocks = try blocks.toOwnedSlice(gpa);
    errdefer gpa.free(out_blocks);
    var out_headings = try headings.toOwnedSlice(gpa);
    errdefer gpa.free(out_headings);

    // 标题栈改由 md4c 给出。行扫描器只认顶格的 ATX，在 CommonMark 官方测试集上标题
    // 只对 37%：setext、缩进 1-3 空格的 ATX 全部漏掉，而 HTML 块里的 `#` 被当成标题还
    // 会重置标题路径，让后面的段落挂到一个不存在的标题下。
    //
    // 块仍由扫描器切分——它对段落、围栏、表格的判断是好的，装箱也只需要边界。md4c 在
    // 这里只负责回答「此处生效的标题栈是什么」，那是 heading_path 的唯一来源。
    //
    // 喂进去的是 frontmatter 之后的正文：`node_type: wiki` 紧跟 `---` 按 CommonMark 是
    // 一个 setext H2，整库 84 个 frontmatter 会各自变出一个假标题。
    if (rebuildHeadings(gpa, source, body_offset, out_blocks)) |rebuilt| {
        gpa.free(out_headings);
        out_headings = rebuilt;
        title = null;
        for (out_headings) |h| {
            if (h.level == 1) {
                title = std.mem.trim(u8, source[h.text_start..h.text_end], " \t");
                break;
            }
        }
    } else |_| {
        // md4c 只在分配失败时会失败。退回扫描器自己的标题：标题栈差一点，比整篇文档
        // 索引不了好。
    }

    return .{
        .frontmatter = frontmatter,
        .title = title,
        .blocks = out_blocks,
        .headings = out_headings,
    };
}

/// md4c 的标题映射成 Document.Heading。
///
/// 引用块里的标题不计入：`> # 注意` 是被引用的内容，不是这篇文档的结构，让它改变后续
/// 段落的标题路径是错的。md4c 给了 quote_depth，所以这是一个明写的决定，而不是碰巧。
fn rebuildHeadings(
    gpa: std.mem.Allocator,
    source: []const u8,
    body_offset: usize,
    blocks: []const Block,
) ![]Document.Heading {
    const hs = try md4c.headings(gpa, source[body_offset..]);
    defer gpa.free(hs);

    var out: std.ArrayList(Document.Heading) = .empty;
    errdefer out.deinit(gpa);

    for (hs) |h| {
        if (h.quote_depth != 0) continue;
        const at = body_offset + h.byte_offset;
        const text = headingSpan(source, at);
        try out.append(gpa, .{
            .level = h.level,
            .text_start = text.start,
            .text_end = text.end,
            .block_index = blockIndexAt(blocks, at),
        });
    }
    return out.toOwnedSlice(gpa);
}

/// 标题正文的绝对区间。`at` 是标题起始行的行首。
///
/// ATX 去掉前导 `#` 与尾部的收尾 `#`。setext 的正文**可以跨多行**——正文一直延续到全为
/// `=` 或 `-` 的下划线行为止，只取第一行会把 `Foo\nbar` 这样的标题截成一半。
fn headingSpan(source: []const u8, at: usize) struct { start: usize, end: usize } {
    const first_end = lineEnd(source, at);
    const line = trimEol(source[at..first_end]);

    var i: usize = 0;
    while (i < line.len and i < 3 and line[i] == ' ') i += 1;
    var hashes: usize = 0;
    while (i + hashes < line.len and line[i + hashes] == '#') hashes += 1;

    if (hashes != 0 and hashes <= 6) {
        var start = i + hashes;
        while (start < line.len and (line[start] == ' ' or line[start] == '\t')) start += 1;
        var end = line.len;
        while (end > start and (line[end - 1] == ' ' or line[end - 1] == '\t')) end -= 1;
        // 收尾序列：`## foo ##` 的正文是 foo，但 `# foo#` 的正文是 foo# —— 只有前面隔着
        // 空白才算收尾。
        var trailing = end;
        while (trailing > start and line[trailing - 1] == '#') trailing -= 1;
        if (trailing != end and trailing > start and
            (line[trailing - 1] == ' ' or line[trailing - 1] == '\t'))
        {
            end = trailing;
            while (end > start and (line[end - 1] == ' ' or line[end - 1] == '\t')) end -= 1;
        } else if (trailing == start) {
            end = start; // 整行都是 `#`
        }
        return .{ .start = at + start, .end = at + end };
    }

    // setext：往下走到下划线行，正文是它之前的全部。
    var text_end = at + line.len;
    var p = if (first_end < source.len) first_end + 1 else source.len;
    while (p < source.len) {
        const le = lineEnd(source, p);
        const l = std.mem.trim(u8, trimEol(source[p..le]), " \t");
        if (l.len != 0 and (allOf(l, '=') or allOf(l, '-'))) break;
        if (l.len == 0) break;
        text_end = p + trimEol(source[p..le]).len;
        p = if (le < source.len) le + 1 else source.len;
    }
    const start = at + (std.mem.indexOfNone(u8, source[at..text_end], " \t") orelse 0);
    return .{ .start = start, .end = text_end };
}

fn allOf(s: []const u8, ch: u8) bool {
    for (s) |c| if (c != ch) return false;
    return s.len != 0;
}

/// 含 `at` 的块下标，没有就取最后一个起点不晚于它的块。
fn blockIndexAt(blocks: []const Block, at: usize) usize {
    var best: usize = 0;
    for (blocks, 0..) |b, i| {
        if (b.byte_start > at) break;
        best = i;
    }
    return best;
}

// ---------------------------------------------------------------------------
// line classification
// ---------------------------------------------------------------------------

fn lineEnd(source: []const u8, from: usize) usize {
    return std.mem.indexOfScalarPos(u8, source, from, '\n') orelse source.len;
}

fn trimEol(line: []const u8) []const u8 {
    return std.mem.trimEnd(u8, line, "\r");
}

fn peekLine(source: []const u8, from: usize) []const u8 {
    if (from >= source.len) return &.{};
    return trimEol(source[from..lineEnd(source, from)]);
}

fn isBlank(line: []const u8) bool {
    return std.mem.trim(u8, line, " \t").len == 0;
}

fn headingLevel(line: []const u8) ?u8 {
    var n: u8 = 0;
    while (n < line.len and line[n] == '#') n += 1;
    if (n == 0 or n > 6) return null;
    // ATX headings require a space after the hashes; "#hashtag" is not one.
    if (n < line.len and line[n] != ' ' and line[n] != '\t') return null;
    return n;
}

const Fence = struct { char: u8, len: usize };

fn fenceMarker(line: []const u8) ?Fence {
    const t = std.mem.trimStart(u8, line, " ");
    if (t.len < 3) return null;
    const ch = t[0];
    if (ch != '`' and ch != '~') return null;
    var n: usize = 0;
    while (n < t.len and t[n] == ch) n += 1;
    if (n < 3) return null;
    return .{ .char = ch, .len = n };
}

fn closesFence(line: []const u8, open: Fence) bool {
    const f = fenceMarker(line) orelse return false;
    // A closing fence must use the same character and be at least as long.
    return f.char == open.char and f.len >= open.len;
}

fn isRule(line: []const u8) bool {
    const t = std.mem.trim(u8, line, " \t");
    if (t.len < 3) return false;
    const ch = t[0];
    if (ch != '-' and ch != '*' and ch != '_') return false;
    for (t) |c| if (c != ch) return false;
    return true;
}

fn isTableRow(line: []const u8) bool {
    const t = std.mem.trim(u8, line, " \t");
    return t.len > 0 and t[0] == '|';
}

fn isTableDelimiter(line: []const u8) bool {
    const t = std.mem.trim(u8, line, " \t");
    if (t.len == 0 or t[0] != '|') return false;
    var seen_dash = false;
    for (t) |c| switch (c) {
        '-' => seen_dash = true,
        '|', ':', ' ', '\t' => {},
        else => return false,
    };
    return seen_dash;
}

fn isListItem(line: []const u8) bool {
    const t = std.mem.trimStart(u8, line, " \t");
    if (t.len < 2) return false;
    if ((t[0] == '-' or t[0] == '*' or t[0] == '+') and (t[1] == ' ' or t[1] == '\t')) return true;
    // ordered: digits then '.' or ')'
    var i: usize = 0;
    while (i < t.len and std.ascii.isDigit(t[i])) i += 1;
    return i > 0 and i + 1 < t.len and (t[i] == '.' or t[i] == ')') and (t[i + 1] == ' ');
}

const FrontmatterClose = struct { content_end: usize, next_line: usize };

fn findClosingFrontmatter(source: []const u8, from: usize) ?FrontmatterClose {
    var pos = from;
    while (pos < source.len) {
        const le = lineEnd(source, pos);
        const line = trimEol(source[pos..le]);
        if (std.mem.eql(u8, std.mem.trim(u8, line, " \t"), "---")) {
            return .{
                .content_end = pos,
                .next_line = if (le < source.len) le + 1 else source.len,
            };
        }
        pos = if (le < source.len) le + 1 else source.len;
    }
    // Unterminated frontmatter: treat the whole thing as body rather than
    // swallowing the document.
    return null;
}

// ---------------------------------------------------------------------------
// link extraction
// ---------------------------------------------------------------------------

pub const LinkKind = enum {
    /// [text](target)
    md,
    /// [[wikilink]]
    wiki,
    /// A `scheme://` rooted at the collection rather than the web —
    /// `zkb://projects/x/REQ.md` names exactly that path.
    ///
    /// The scheme itself is **not** matched: anything that is not a network or
    /// filesystem URL is treated this way. `zkb://` is the documented spelling,
    /// but a corpus that already uses another tool's scheme keeps resolving
    /// without a migration, which matters when the links number in the hundreds.
    collection_uri,
    /// depends_on / related_to / part_of in frontmatter
    frontmatter,
    /// Any scheme:// — recorded but never resolved. zkb does not go online, and
    /// a file:// URL is an absolute machine path, not a collection-relative one.
    external,
    /// A target that exists as a file but is not an indexed document (.json,
    /// images, PDFs). Distinct from broken: the file is there, it is simply not
    /// something zkb has an entry for, and reporting it as missing would be wrong.
    asset,
};

pub const Link = struct {
    kind: LinkKind,
    /// Raw target as written. Resolution happens later, against the full
    /// document set.
    raw: []const u8,
    byte_offset: usize,
};

/// Extract links from `source`, using `doc.blocks` to skip fenced code.
///
/// Code blocks are excluded deliberately: a path inside a shell snippet or a
/// sample config is not a reference, and counting it produces broken-link noise
/// for something nobody intended as a link.
pub fn extractLinks(
    gpa: std.mem.Allocator,
    source: []const u8,
    doc: *const Document,
) std.mem.Allocator.Error![]Link {
    var out: std.ArrayList(Link) = .empty;
    errdefer out.deinit(gpa);

    if (doc.frontmatter) |fm| try extractFrontmatterLinks(gpa, &out, fm, source);

    for (doc.blocks) |b| {
        if (b.kind == .code) continue;
        try extractInline(gpa, &out, source, b.byte_start, b.byte_end);
    }
    return out.toOwnedSlice(gpa);
}

/// YAML list items under depends_on / related_to / part_of. Deliberately not a
/// YAML parser: only these three keys matter, and only their `- item` entries.
fn extractFrontmatterLinks(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(Link),
    fm: []const u8,
    source: []const u8,
) !void {
    const keys = [_][]const u8{ "depends_on:", "related_to:", "part_of:" };
    var lines = std.mem.splitScalar(u8, fm, '\n');
    var in_list = false;
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        var is_key = false;
        for (keys) |k| {
            if (std.mem.startsWith(u8, trimmed, k)) {
                is_key = true;
                // `depends_on: [a, b]` inline form.
                const rest = std.mem.trim(u8, trimmed[k.len..], " \t[]");
                if (rest.len != 0) {
                    var items = std.mem.splitScalar(u8, rest, ',');
                    while (items.next()) |it| {
                        const v = std.mem.trim(u8, it, " \t\"'");
                        if (v.len != 0) try out.append(gpa, .{
                            .kind = .frontmatter,
                            .raw = v,
                            .byte_offset = offsetOf(source, v),
                        });
                    }
                    in_list = false;
                } else in_list = true;
                break;
            }
        }
        if (is_key) continue;
        if (in_list) {
            if (std.mem.startsWith(u8, trimmed, "- ")) {
                const v = std.mem.trim(u8, trimmed[2..], " \t\"'");
                if (v.len != 0) try out.append(gpa, .{
                    .kind = .frontmatter,
                    .raw = v,
                    .byte_offset = offsetOf(source, v),
                });
            } else if (trimmed.len != 0 and !std.mem.startsWith(u8, trimmed, "#")) {
                // A new key ends the list.
                in_list = false;
            }
        }
    }
}

fn offsetOf(source: []const u8, needle: []const u8) usize {
    if (needle.len == 0) return 0;
    // `needle` points into `source` when it came from a slice of it.
    const base = @intFromPtr(source.ptr);
    const at = @intFromPtr(needle.ptr);
    return if (at >= base and at < base + source.len) at - base else 0;
}

fn extractInline(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(Link),
    source: []const u8,
    start: usize,
    end: usize,
) !void {
    var i = start;
    while (i < end) {
        // Inline code is skipped for the same reason fenced code is: documentation
        // that explains link syntax writes `[text](path)` and `[[wikilink]]` as
        // illustrations. Measured on ~/docs, treating those as real links produced
        // a steady stream of phantom broken links pointing at "path" and "url".
        if (source[i] == '`') {
            var ticks: usize = 0;
            while (i + ticks < end and source[i + ticks] == '`') ticks += 1;
            const fence = source[i .. i + ticks];
            if (std.mem.indexOfPos(u8, source[0..end], i + ticks, fence)) |close| {
                i = close + ticks;
                continue;
            }
            // Unterminated: treat the backtick as ordinary text.
            i += ticks;
            continue;
        }
        // [[wikilink]]
        if (i + 1 < end and source[i] == '[' and source[i + 1] == '[') {
            if (std.mem.indexOfPos(u8, source[0..end], i + 2, "]]")) |close| {
                const raw = std.mem.trim(u8, source[i + 2 .. close], " \t");
                if (raw.len != 0) try out.append(gpa, .{
                    .kind = .wiki,
                    .raw = raw,
                    .byte_offset = i,
                });
                i = close + 2;
                continue;
            }
        }
        // [text](target)
        if (source[i] == '[') {
            if (std.mem.indexOfScalarPos(u8, source[0..end], i, ']')) |rb| {
                if (rb + 1 < end and source[rb + 1] == '(') {
                    if (std.mem.indexOfScalarPos(u8, source[0..end], rb + 2, ')')) |rp| {
                        var raw = std.mem.trim(u8, source[rb + 2 .. rp], " \t");
                        // Strip an optional "title" after the target.
                        if (std.mem.indexOfScalar(u8, raw, ' ')) |sp| raw = raw[0..sp];
                        // A pure anchor is navigation inside the same document,
                        // not a reference to another one. Recording it would make
                        // it permanently unresolvable and therefore permanently
                        // "broken" in every report.
                        if (raw.len != 0 and raw[0] != '#') {
                            try out.append(gpa, .{
                                .kind = classify(raw),
                                .raw = raw,
                                .byte_offset = rb + 2,
                            });
                        }
                        i = rp + 1;
                        continue;
                    }
                }
            }
        }
        // A bare `scheme://path` outside any markdown link syntax. Scanned
        // generically so no tool's URI scheme has to be known here.
        if (bareUriAt(source, i, end)) |raw| {
            // A scheme with nothing after it is prose naming the scheme, not a
            // reference. Recording it guarantees a permanently broken link.
            const sep = std.mem.indexOf(u8, raw, "://").? + 3;
            if (raw.len > sep) {
                try out.append(gpa, .{ .kind = classify(raw), .raw = raw, .byte_offset = i });
            }
            i += raw.len;
            continue;
        }
        i += 1;
    }
}

/// Schemes that leave the knowledge base.
fn isExternalScheme(scheme: []const u8) bool {
    inline for (.{ "http", "https", "file", "ftp", "data", "mailto" }) |ext| {
        if (std.ascii.eqlIgnoreCase(scheme, ext)) return true;
    }
    return false;
}

/// The scheme at the start of `raw`, if there is one: a letter, then letters,
/// digits, `+`, `-` or `.`, up to a colon (RFC 3986 §3.1).
///
/// Needed because not every scheme is followed by `//`. `data:image/svg+xml,…`
/// and `mailto:a@b` are opaque, so looking for `://` misses them and they fall
/// through to the relative-path branch — which is how nine inline SVGs in one
/// downloaded article turned into nine reported broken links.
fn schemeOf(raw: []const u8) ?[]const u8 {
    if (raw.len == 0 or !std.ascii.isAlphabetic(raw[0])) return null;
    var i: usize = 1;
    while (i < raw.len) : (i += 1) {
        if (raw[i] == ':') return raw[0..i];
        const c = raw[i];
        // A slash or fragment before any colon means this is a path.
        if (!std.ascii.isAlphanumeric(c) and c != '+' and c != '-' and c != '.') return null;
    }
    return null;
}

/// A bare `scheme://rest` starting at `i`, if there is one. The scheme must be
/// letters, which is what keeps `://` inside prose from matching.
fn bareUriAt(source: []const u8, i: usize, end: usize) ?[]const u8 {
    if (!std.ascii.isAlphabetic(source[i])) return null;
    if (i > 0 and (std.ascii.isAlphanumeric(source[i - 1]) or source[i - 1] == '/')) return null;
    var j = i;
    while (j < end and std.ascii.isAlphanumeric(source[j])) j += 1;
    if (j + 3 > end or !std.mem.eql(u8, source[j..][0..3], "://")) return null;
    j += 3;
    while (j < end) {
        const c = source[j];
        if (std.ascii.isWhitespace(c) or c == ')' or c == ']' or c == '`') break;
        // Full-width punctuation ends the uri. It has to be a hard boundary
        // rather than something trimmed afterwards: CJK prose has no spaces, so
        // `zkb://a/b.md。另见 …` would otherwise run to the next ascii space and
        // swallow the rest of the clause into the link.
        if (cjkPunctLen(source[j..end]) != 0) break;
        j += 1;
    }
    return source[i .. i + trimUriTail(source[i..j])];
}

/// Byte length of the full-width punctuation mark at the start of `rest`, or 0.
fn cjkPunctLen(rest: []const u8) usize {
    const marks = [_][]const u8{
        "，", "。", "、", "；", "：", "！", "？", "（", "）", "「", "」",
        "『", "』", "《", "》", "【", "】", "〜", "…", "　",
    };
    for (marks) |m| {
        if (rest.len >= m.len and std.mem.eql(u8, rest[0..m.len], m)) return m.len;
    }
    return 0;
}

/// Length of `raw` with sentence punctuation trimmed off the end.
///
/// A bare URI in prose is almost always followed by punctuation — `zkb://a/b.md,`
/// or `…参见 zkb://a/b.md。` — and swallowing it changes what the link means: the
/// extension becomes `.md,`, which is not a document extension, so the reference
/// is filed as an asset and silently stops counting as a link between documents.
/// That corrupts the orphan and inbound-link checks rather than announcing itself.
///
/// Ascii punctuation only. Unlike the full-width marks, these do occur inside
/// real urls, so they are trimmed from the end rather than treated as boundaries.
fn trimUriTail(raw: []const u8) usize {
    const tail = ",.;:!?'\"<>";
    var n = raw.len;
    while (n > 0 and std.mem.indexOfScalar(u8, tail, raw[n - 1]) != null) n -= 1;
    return n;
}

/// Extensions zkb indexes. A link to anything else is an asset reference.
const document_exts = [_][]const u8{ ".md", ".txt", ".mdx" };

fn classify(raw: []const u8) LinkKind {
    if (schemeOf(raw)) |scheme| {
        // http/file/data point outside the knowledge base — file:// in
        // particular is an absolute machine path that must never be joined onto
        // a document's directory.
        if (isExternalScheme(scheme)) return .external;
        // Any other scheme with an authority is a collection-rooted URI, which
        // is what keeps zkb:// working without naming it here.
        if (std.mem.startsWith(u8, raw[scheme.len..], "://"))
            return if (isAsset(raw)) .asset else .collection_uri;
        // An opaque scheme we do not know (tel:, urn:, javascript:). Calling it
        // external is wrong only in naming; calling it a path would make it a
        // broken link on every document that uses one.
        return .external;
    }
    if (isAsset(raw)) return .asset;
    return .md;
}

fn isAsset(raw: []const u8) bool {
    var t = raw;
    if (std.mem.indexOfScalar(u8, t, '#')) |h| t = t[0..h];
    const base = std.fs.path.basename(t);
    const dot = std.mem.lastIndexOfScalar(u8, base, '.') orelse return false;
    const ext = base[dot..];
    for (document_exts) |e| if (std.ascii.eqlIgnoreCase(ext, e)) return false;
    // No extension at all is a wikilink-style stem, not an asset.
    return ext.len > 1;
}
