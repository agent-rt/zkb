//! Path glob matching for scan include patterns.
//!
//! Deliberately not a general fnmatch: patterns are matched against the
//! slash-separated relative path of an entry under a collection root, so the
//! component boundary is the thing that matters. `*` and `?` never cross a `/`;
//! `**` matches any number of whole components, including zero.
//!
//! The reason `matchPrefix` exists alongside `match` is pruning. The scanner
//! walks selectively, and a pattern like `*/memory/*.md` should not cause it to
//! descend into every project's `tool-results` directory just to reject the
//! files afterwards. `matchPrefix` answers the weaker question the walker needs
//! at a directory: "could anything under here still match?"

const std = @import("std");

/// Does `path` (relative, `/`-separated, no leading slash) match `pattern`?
pub fn match(pattern: []const u8, path: []const u8) bool {
    return matchSeq(pattern, path, .exact);
}

/// Could some path *under* directory `dir` match `pattern`?
///
/// Conservative in the safe direction: a false positive costs one wasted
/// directory descent, a false negative would silently drop files. Never returns
/// false for a directory that has a matching descendant.
pub fn matchPrefix(pattern: []const u8, dir: []const u8) bool {
    return matchSeq(pattern, dir, .prefix);
}

/// True when any pattern matches. An empty list means "no include filter", so
/// everything is in — that is the default, not a rejection of everything.
pub fn matchAny(patterns: []const []const u8, path: []const u8) bool {
    if (patterns.len == 0) return true;
    for (patterns) |p| if (match(p, path)) return true;
    return false;
}

/// True when any pattern could still match below `dir`. Same empty-list rule.
pub fn matchAnyPrefix(patterns: []const []const u8, dir: []const u8) bool {
    if (patterns.len == 0) return true;
    for (patterns) |p| if (matchPrefix(p, dir)) return true;
    return false;
}

const Mode = enum { exact, prefix };

/// Component-wise match with `**` backtracking.
///
/// The standard two-pointer wildcard walk, one level up: components instead of
/// bytes. Iterative rather than recursive so a pathological pattern cannot blow
/// the stack. `star_pat`/`star_path` remember the most recent `**` so a wrong
/// guess costs a retry rather than a combinatorial search.
fn matchSeq(pattern: []const u8, path: []const u8, mode: Mode) bool {
    var pat = Components{ .s = pattern };
    var p = Components{ .s = path };

    // Where to resume if the current `**` guess consumed too little.
    var star_pat: ?Components = null;
    var star_path: ?Components = null;

    while (p.peek() != null) {
        const at_pat = pat;
        if (pat.next()) |pc| {
            if (std.mem.eql(u8, pc, "**")) {
                // Try zero components first; widen only if the rest fails.
                star_pat = pat;
                star_path = p;
                continue;
            }
            if (matchComponent(pc, p.peek().?)) {
                _ = p.next();
                continue;
            }
            pat = at_pat;
        }

        // Either the pattern ran out with path left, or this component did not
        // match. Both are recoverable only by letting a `**` swallow one more.
        if (star_pat) |sp| {
            var wider = star_path.?;
            _ = wider.next();
            pat = sp;
            p = wider;
            star_path = wider;
            continue;
        }
        return false;
    }

    // Path exhausted. For a directory that is the whole point: whatever is left
    // of the pattern can still be satisfied by entries further down, so the
    // directory is worth entering.
    if (mode == .prefix) return true;

    // For a file, only trailing `**`s may remain — they match zero components.
    while (pat.next()) |pc| if (!std.mem.eql(u8, pc, "**")) return false;
    return true;
}

/// Yields non-empty path components, so `a//b` and a trailing `/` behave like
/// the paths they obviously mean rather than producing empty components that
/// only `*` would match.
const Components = struct {
    s: []const u8,
    i: usize = 0,

    fn next(self: *Components) ?[]const u8 {
        while (self.i < self.s.len and self.s[self.i] == '/') self.i += 1;
        if (self.i >= self.s.len) return null;
        const start = self.i;
        while (self.i < self.s.len and self.s[self.i] != '/') self.i += 1;
        return self.s[start..self.i];
    }

    fn peek(self: Components) ?[]const u8 {
        var copy = self;
        return copy.next();
    }
};

/// `*`, `?`, `[...]` and backslash escapes, within one path component.
///
/// Character classes are here because gitignore has them, and `.zkbignore` claims
/// gitignore semantics — a class that silently failed to match would be a claim
/// the implementation does not honour.
fn matchComponent(pattern: []const u8, name: []const u8) bool {
    var pi: usize = 0;
    var ni: usize = 0;
    // Resume point for the most recent `*`, which is what keeps this linear.
    var star_pi: ?usize = null;
    var star_ni: usize = 0;

    while (ni < name.len) {
        if (pi < pattern.len) switch (pattern[pi]) {
            '*' => {
                star_pi = pi;
                pi += 1;
                star_ni = ni;
                continue;
            },
            '?' => {
                pi += 1;
                ni += 1;
                continue;
            },
            '[' => {
                if (classMatch(pattern[pi..], name[ni])) |used| {
                    pi += used;
                    ni += 1;
                    continue;
                }
                // An unterminated `[` is a literal bracket, which is what git
                // does rather than rejecting the pattern.
            },
            '\\' => {
                // Escapes the next byte, so `\*` is a literal asterisk.
                if (pi + 1 < pattern.len and pattern[pi + 1] == name[ni]) {
                    pi += 2;
                    ni += 1;
                    continue;
                }
            },
            else => {},
        };
        if (pi < pattern.len and pattern[pi] == name[ni]) {
            pi += 1;
            ni += 1;
            continue;
        }
        if (star_pi) |sp| {
            pi = sp + 1;
            star_ni += 1;
            ni = star_ni;
            continue;
        }
        return false;
    }
    while (pi < pattern.len and pattern[pi] == '*') pi += 1;
    return pi == pattern.len;
}

/// Match `c` against a `[...]` class at the start of `pattern`.
///
/// Returns how many pattern bytes the class occupies, or null when it does not
/// match (or is unterminated, which makes the `[` a literal).
fn classMatch(pattern: []const u8, c: u8) ?usize {
    std.debug.assert(pattern[0] == '[');
    var i: usize = 1;
    // Both spellings of negation, as git accepts either.
    const negated = i < pattern.len and (pattern[i] == '!' or pattern[i] == '^');
    if (negated) i += 1;

    var hit = false;
    var first = true;
    while (i < pattern.len) {
        // A `]` in the first position is a literal, not the terminator.
        if (pattern[i] == ']' and !first) {
            const matched = hit != negated;
            return if (matched) i + 1 else null;
        }
        first = false;

        var lo = pattern[i];
        if (lo == '\\' and i + 1 < pattern.len) {
            i += 1;
            lo = pattern[i];
        }
        // `a-z`, but a trailing `-` before `]` is a literal dash.
        if (i + 2 < pattern.len and pattern[i + 1] == '-' and pattern[i + 2] != ']') {
            const hi = pattern[i + 2];
            if (c >= lo and c <= hi) hit = true;
            i += 3;
            continue;
        }
        if (c == lo) hit = true;
        i += 1;
    }
    // Unterminated: caller falls back to treating `[` literally.
    return null;
}
