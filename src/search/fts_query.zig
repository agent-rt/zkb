//! Build an FTS5 MATCH expression from arbitrary user text.
//!
//! Two hard rules:
//!
//! 1. **User input is never spliced into MATCH syntax.** FTS5 has its own
//!    grammar (AND / OR / NOT / NEAR / `*` / quotes / `-`); passing raw input
//!    through means a stray quote is a syntax error and a stray `OR` silently
//!    changes what was searched. Every term is wrapped in double quotes with
//!    internal quotes doubled, forcing it to be a literal.
//!
//! 2. **Terms the tokenizer cannot match are dropped and reported.** FTS5 does
//!    not error on such a term — it silently matches nothing, and silence is the
//!    dangerous case because the caller would believe the word was searched.
//!
//!    What counts as unmatchable follows the tokenizer, and the tokenizer
//!    changed. Under `trigram` it was anything under 3 characters, which threw
//!    away real Chinese content words (融合, 解析, 规范, 存储). Under `zkb_cjk`
//!    (SPEC §2.5) CJK is indexed as bigrams and Latin as whole words, so:
//!
//!      - a CJK term needs **2** characters, not 3
//!      - a Latin/digit term needs **1** — whole-word tokens have no floor
//!
//!    Only a lone CJK character is still weak, since a bigram cannot be formed
//!    from it. That is the entire remaining scope of this rule; it exists now for
//!    one narrow case instead of papering over a tokenizer mismatch.
//!
//! Length is counted in **codepoints, not bytes**. A two-character Chinese word
//! is six UTF-8 bytes; a byte-length check would misjudge it entirely.

const std = @import("std");

/// Minimum characters for a CJK term: one bigram.
pub const min_cjk_chars: usize = 2;
/// Latin and digit runs become whole-word tokens, so any length is matchable.
pub const min_latin_chars: usize = 1;

pub const Query = struct {
    /// MATCH expression, or null when nothing usable survived.
    expr: ?[]const u8,
    /// Terms discarded for being too short for the tokenizer.
    dropped: [][]const u8,

    pub fn deinit(self: *Query, gpa: std.mem.Allocator) void {
        if (self.expr) |e| gpa.free(e);
        for (self.dropped) |d| gpa.free(d);
        gpa.free(self.dropped);
        self.* = undefined;
    }
};

pub fn build(gpa: std.mem.Allocator, text: []const u8) !Query {
    var terms: std.ArrayList([]const u8) = .empty;
    defer terms.deinit(gpa);
    var dropped: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (dropped.items) |d| gpa.free(d);
        dropped.deinit(gpa);
    }

    var it = tokenize(text);
    while (it.next()) |term| {
        // A term may still mix scripts ("Qwen3中英"), and a CJK run must be cut
        // the same way the tokenizer cuts it. Whitespace splitting alone is not
        // enough: Chinese is written without spaces, so an entire question
        // arrives as one "term".
        var runs = scriptRuns(term);
        while (runs.next()) |run| {
            if (run.is_cjk) {
                const chars = codepointLen(run.text);
                if (chars < min_cjk_chars) {
                    try dropped.append(gpa, try gpa.dupe(u8, run.text));
                    continue;
                }
                // Emit the run's overlapping bigrams as separate terms — exactly
                // the tokens the index holds.
                //
                // Keeping the run as one quoted phrase (what this code used to
                // do) demands the whole character sequence appear adjacently,
                // which for a natural-language question is essentially never:
                // measured 0 keyword hits for "知识库的混合检索怎么设计".
                // OR-ing the bigrams lets BM25 rank by how many matched.
                var i: usize = 0;
                var prev: ?usize = null;
                while (i < run.text.len) {
                    const len = std.unicode.utf8ByteSequenceLength(run.text[i]) catch 1;
                    if (prev) |p| try terms.append(gpa, run.text[p .. i + len]);
                    prev = i;
                    i += len;
                }
            } else {
                if (codepointLen(run.text) < min_latin_chars) {
                    try dropped.append(gpa, try gpa.dupe(u8, run.text));
                    continue;
                }
                try terms.append(gpa, run.text);
            }
        }
    }

    if (terms.items.len == 0) {
        return .{ .expr = null, .dropped = try dropped.toOwnedSlice(gpa) };
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (terms.items, 0..) |term, i| {
        // OR rather than AND. With CJK expanded to bigrams a query becomes many
        // terms, and AND would require every single bigram present — one unusual
        // character would zero the result set. OR lets BM25 rank by how many
        // matched, which is the standard dictionary-free CJK approach.
        if (i != 0) try out.appendSlice(gpa, " OR ");
        try out.append(gpa, '"');
        for (term) |ch| {
            if (ch == '"') try out.append(gpa, '"'); // FTS5 escapes " by doubling
            try out.append(gpa, ch);
        }
        try out.append(gpa, '"');
    }

    return .{
        .expr = try out.toOwnedSlice(gpa),
        .dropped = try dropped.toOwnedSlice(gpa),
    };
}

/// Split on whitespace and ASCII punctuation. CJK runs survive this stage intact
/// (they contain no separators) and are segmented afterwards by `scriptRuns`.
const Tokenizer = struct {
    text: []const u8,
    pos: usize = 0,

    fn next(self: *Tokenizer) ?[]const u8 {
        while (self.pos < self.text.len and isSeparator(self.text[self.pos])) self.pos += 1;
        if (self.pos >= self.text.len) return null;
        const start = self.pos;
        while (self.pos < self.text.len and !isSeparator(self.text[self.pos])) self.pos += 1;
        return self.text[start..self.pos];
    }
};

fn tokenize(text: []const u8) Tokenizer {
    return .{ .text = text };
}

fn isSeparator(c: u8) bool {
    // Only ASCII separators: multi-byte UTF-8 continuation bytes are all >= 0x80
    // and must never be treated as boundaries.
    if (c >= 0x80) return false;
    return switch (c) {
        ' ', '\t', '\n', '\r' => true,
        '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}' => true,
        '<', '>', '/', '\\', '|', '=', '+', '*', '&', '^', '%', '$', '#', '@' => true,
        '\'', '`', '~' => true,
        else => false,
    };
}

fn codepointLen(s: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        const len = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
        i += len;
        n += 1;
    }
    return n;
}

/// Mirrors the CJK ranges in src/db/fts5_cjk.c. Kept deliberately narrow: this
/// only has to decide "would the tokenizer bigram this or word-tokenize it".
fn isCjk(cp: u21) bool {
    return (cp >= 0x3040 and cp <= 0x30FF) or // Hiragana + Katakana
        (cp >= 0x3400 and cp <= 0x4DBF) or // CJK Ext A
        (cp >= 0x4E00 and cp <= 0x9FFF) or // CJK Unified
        (cp >= 0xF900 and cp <= 0xFAFF) or // CJK Compatibility
        (cp >= 0xAC00 and cp <= 0xD7AF) or // Hangul Syllables
        (cp >= 0x20000 and cp <= 0x2FA1F); // CJK Ext B+
}

fn decodeAt(s: []const u8, i: usize) struct { cp: u21, len: usize } {
    const len = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
    if (len == 1) return .{ .cp = s[i], .len = 1 };
    if (i + len > s.len) return .{ .cp = 0xFFFD, .len = 1 };
    const cp = std.unicode.utf8Decode(s[i .. i + len]) catch 0xFFFD;
    return .{ .cp = cp, .len = len };
}

/// Splits a term into maximal same-script runs, mirroring how the tokenizer
/// decides between bigram and whole-word segmentation.
const ScriptRuns = struct {
    text: []const u8,
    pos: usize = 0,

    const Run = struct { text: []const u8, is_cjk: bool };

    fn next(self: *ScriptRuns) ?Run {
        if (self.pos >= self.text.len) return null;
        const first = decodeAt(self.text, self.pos);
        const want = isCjk(first.cp);
        const start = self.pos;
        while (self.pos < self.text.len) {
            const d = decodeAt(self.text, self.pos);
            if (isCjk(d.cp) != want) break;
            self.pos += d.len;
        }
        return .{ .text = self.text[start..self.pos], .is_cjk = want };
    }
};

fn scriptRuns(term: []const u8) ScriptRuns {
    return .{ .text = term };
}
