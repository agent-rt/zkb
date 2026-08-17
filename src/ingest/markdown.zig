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

    return .{
        .frontmatter = frontmatter,
        .title = title,
        .blocks = try blocks.toOwnedSlice(gpa),
        .headings = try headings.toOwnedSlice(gpa),
    };
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
