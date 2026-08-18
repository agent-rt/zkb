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

/// Like `matchAny`, but an empty list matches *nothing*.
///
/// A separate function rather than a flag because the empty case means the
/// opposite thing in the two uses, and sharing one made an exclude list of zero
/// patterns reject every path: a scan with no `--exclude` saw 0 of 5 files. The
/// unit tests missed it because they only ever passed non-empty exclude lists —
/// the end-to-end run is what caught it.
///
/// Read the names as what they defend: `matchAny` answers "is this allowed",
/// where no rules means yes; `matchAnyStrict` answers "is this forbidden", where
/// no rules means no.
pub fn matchAnyStrict(patterns: []const []const u8, path: []const u8) bool {
    for (patterns) |p| if (match(p, path)) return true;
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

/// `*` (any run, not crossing `/` — components never contain one) and `?`.
fn matchComponent(pattern: []const u8, name: []const u8) bool {
    var pi: usize = 0;
    var ni: usize = 0;
    var star_pi: ?usize = null;
    var star_ni: usize = 0;

    while (ni < name.len) {
        if (pi < pattern.len and pattern[pi] == '*') {
            star_pi = pi;
            pi += 1;
            star_ni = ni;
            continue;
        }
        if (pi < pattern.len and (pattern[pi] == '?' or pattern[pi] == name[ni])) {
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
