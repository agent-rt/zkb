//! `.zkbignore` — gitignore semantics, not a lookalike.
//!
//! The rules live in the corpus rather than in the database, which is the same
//! reason the index is derived: the filesystem is the truth. A `.zkbignore` is
//! versioned with the documents, travels with a clone, and is unset by deleting a
//! line. The `collections.exclude` column it replaces could be set and not
//! cleared — `coalesce(?, exclude)` keeps whatever is stored, so the only way
//! back was rebuilding the index.
//!
//! **Claiming gitignore semantics is a commitment.** The intricate parts are all
//! load-bearing in real files, and a subset that merely resembles them is worse
//! than an obviously different mechanism, because it invites gitignore intuition
//! and then fails somewhere specific:
//!
//!   * last matching pattern wins, so `!` can re-include
//!   * a pattern with no `/` matches a basename at any depth; one with a `/`
//!     is anchored to the file's own directory
//!   * a trailing `/` matches directories only
//!   * a pruned directory cannot be re-included from inside (git's own rule)
//!   * nested files apply to their subtree and outrank their parents
//!
//! Verified against `git check-ignore` as an oracle rather than against my
//! reading of the manual — see tests/fixtures/gitignore-cases.txt.

const std = @import("std");
const glob = @import("glob.zig");

pub const Pattern = struct {
    /// Already normalized into the dialect glob.zig speaks.
    glob: []const u8,
    /// `!` prefix: a match re-includes instead of excluding.
    negated: bool,
    /// Trailing `/`: matches directories only.
    dir_only: bool,
    /// Pattern was written `foo/**`, which in gitignore means the contents of
    /// `foo` and *not* `foo` itself.
    ///
    /// glob.zig cannot express this: a trailing `**` there matches zero
    /// components, deliberately, so an `--include docs/**` does not exclude the
    /// root it is scoping to. The two uses want opposite answers for the same
    /// spelling, so the distinction is carried here rather than by bending the
    /// matcher under one of its callers.
    contents_only: bool = false,
    /// Directory the owning file sits in, `""` for the root. Patterns only apply
    /// to paths under it.
    base: []const u8,
};

pub const Decision = enum { none, ignore, keep };

pub const Matcher = struct {
    /// In evaluation order: outermost file first, then deeper ones. Within a
    /// file, source order. Last match wins, so deeper and later rules override.
    patterns: []const Pattern = &.{},

    /// Where the collection root sits inside the coordinate system the patterns
    /// are expressed in, or `""` when they coincide.
    ///
    /// A repo's `.gitignore` usually lives above the collection root — a
    /// collection rooted at `<repo>/docs` has its rules one level up — and its
    /// patterns are relative to the repo root. Rewriting each
    /// pattern to the collection's coordinates means re-anchoring globs with
    /// `**` in the head, which is where a translation quietly stops matching git.
    /// Restoring the repo-relative path before matching is what git itself does,
    /// and needs no translation at all.
    prefix: []const u8 = "",

    /// Longest path this can build. Paths are already bounded by the filesystem.
    const max_path = std.fs.max_path_bytes;

    /// gitignore's own limitation, reproduced deliberately: once a directory is
    /// excluded, git never descends into it, so no `!` inside can bring a file
    /// back. A walker that pruned the directory could not honour such a rule
    /// anyway, and pretending otherwise would make `.zkbignore` diverge from the
    /// tool it claims to copy.
    pub fn isIgnored(self: Matcher, path: []const u8, is_dir: bool) bool {
        return self.decide(path, is_dir) == .ignore;
    }

    /// An excluded ancestor is decisive and is checked first.
    ///
    /// git evaluates directory by directory and never descends into an excluded
    /// one, so `!build/keep.md` under a `build/` rule does not bring the file
    /// back. Encoding that as "an ignored ancestor short-circuits" reproduces it
    /// exactly, and it also fixes the other half: a rule for the directory must
    /// not make the directory's own *path* ignored when asked about it as a file.
    pub fn decide(self: Matcher, path: []const u8, is_dir: bool) Decision {
        if (self.prefix.len == 0) return self.decideFull(path, is_dir);

        var buf: [max_path]u8 = undefined;
        const full = std.fmt.bufPrint(&buf, "{s}/{s}", .{ self.prefix, path }) catch
            return self.decideFull(path, is_dir);
        // Ancestors inside the prefix are walked too: a repo rule naming the
        // collection's own directory ignores the whole collection, and skipping
        // those components would miss it.
        return self.decideFull(full, is_dir);
    }

    fn decideFull(self: Matcher, path: []const u8, is_dir: bool) Decision {
        var i: usize = 0;
        while (std.mem.indexOfScalarPos(u8, path, i, '/')) |slash| {
            if (self.decideOwn(path[0..slash], true) == .ignore) return .ignore;
            i = slash + 1;
        }
        return self.decideOwn(path, is_dir);
    }

    /// Last matching pattern wins, for this exact path only.
    fn decideOwn(self: Matcher, path: []const u8, is_dir: bool) Decision {
        var out: Decision = .none;
        for (self.patterns) |p| {
            if (p.dir_only and !is_dir) continue;
            if (!underBase(p.base, path)) continue;
            const rel = relativeTo(p.base, path);
            if (p.contents_only and !hasComponentBeyond(p.glob, rel)) continue;
            if (glob.match(p.glob, rel)) out = if (p.negated) .keep else .ignore;
        }
        return out;
    }

    /// Whether a directory could still contain something not ignored.
    ///
    /// Only false when the directory itself is ignored and no negation could
    /// bring anything back — which, per the rule above, is any ignored directory.
    /// Kept as a named method so the walker reads as what it is doing.
    pub fn prunes(self: Matcher, dir: []const u8) bool {
        return self.isIgnored(dir, true);
    }
};

/// True when `rel` reaches past the literal prefix of a `foo/**` pattern.
///
/// Only the fixed head is compared: `a/**/b/**` still needs the glob itself to
/// decide, and this just rules out the path that *is* the prefix.
fn hasComponentBeyond(pattern: []const u8, rel: []const u8) bool {
    const head = pattern[0 .. pattern.len - "/**".len];
    if (std.mem.indexOfScalar(u8, head, '*') != null) return true;
    if (!std.mem.startsWith(u8, rel, head)) return true;
    return rel.len > head.len and rel[head.len] == '/';
}

fn underBase(base: []const u8, path: []const u8) bool {
    if (base.len == 0) return true;
    if (!std.mem.startsWith(u8, path, base)) return false;
    return path.len > base.len and path[base.len] == '/';
}

fn relativeTo(base: []const u8, path: []const u8) []const u8 {
    if (base.len == 0) return path;
    return path[base.len + 1 ..];
}

/// Parse one `.zkbignore` into patterns based at `base` (`""` for the root).
///
/// Appends rather than returns, so a caller can build one ordered list across a
/// root file and the nested files found on the way down.
pub fn parseInto(
    arena: std.mem.Allocator,
    out: *std.ArrayList(Pattern),
    base: []const u8,
    text: []const u8,
) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw_line| {
        var line = std.mem.trimEnd(u8, raw_line, "\r");
        // Trailing spaces are not part of a pattern unless escaped. Leading ones
        // are significant, so only the right side is trimmed.
        line = trimUnescapedTrailingSpace(line);
        if (line.len == 0) continue;
        // `#` only comments at the start; `\#` is a literal hash.
        if (line[0] == '#') continue;

        var negated = false;
        if (line[0] == '!') {
            negated = true;
            line = line[1..];
            if (line.len == 0) continue;
        } else if (line.len >= 2 and line[0] == '\\' and (line[1] == '#' or line[1] == '!')) {
            line = line[1..];
        }

        var dir_only = false;
        if (line.len > 1 and line[line.len - 1] == '/') {
            dir_only = true;
            line = line[0 .. line.len - 1];
        }
        if (line.len == 0) continue;

        // A `/` anywhere but the end anchors the pattern to `base`. Without one,
        // the pattern matches a basename at any depth — which is `**/` in the
        // dialect glob.zig speaks.
        const anchored = std.mem.indexOfScalar(u8, line, '/') != null;
        const body = if (line[0] == '/') line[1..] else line;
        if (body.len == 0) continue;

        const g = if (anchored)
            try arena.dupe(u8, body)
        else
            try std.fmt.allocPrint(arena, "**/{s}", .{body});

        try out.append(arena, .{
            .glob = g,
            .negated = negated,
            .dir_only = dir_only,
            .contents_only = std.mem.endsWith(u8, g, "/**"),
            .base = base,
        });

        // No auxiliary `foo/**` pattern: `decide` reaches the contents by asking
        // about each ancestor instead. Synthesising one made `foo` itself match
        // when queried as a file, because a trailing `**` matches zero components.
    }
}

/// Strip trailing spaces, keeping any that a backslash protects.
fn trimUnescapedTrailingSpace(line: []const u8) []const u8 {
    var end = line.len;
    while (end > 0 and line[end - 1] == ' ') {
        // Count the backslashes before this space; an odd number escapes it.
        var bs: usize = 0;
        var i = end - 1;
        while (i > 0 and line[i - 1] == '\\') : (i -= 1) bs += 1;
        if (bs % 2 == 1) break;
        end -= 1;
    }
    return line[0..end];
}

/// zkb's own file, and the repo's.
///
/// Both are parsed by the same code because `.zkbignore` claims gitignore
/// semantics; having two parsers would be two chances to diverge. `.gitignore` is
/// loaded first in each directory so `.zkbignore` can override it under
/// last-match-wins — `!draft.md` brings back a file git ignores, and `!drafts/`
/// brings back a whole directory.
///
/// Respecting `.gitignore` at all is a judgement worth stating: a file not worth
/// committing is not knowledge worth indexing, and the hardcoded `exclude_dirs`
/// list in scan.zig (`node_modules`, `target`, `dist`, ...) is really a guess at
/// what the repo has already declared.
pub const file_names = [_][]const u8{ ".gitignore", ".zkbignore" };
pub const file_name = ".zkbignore";
