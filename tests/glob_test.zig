//! Path glob matching, including the pruning question the scanner asks.

const std = @import("std");
const testing = std.testing;
const glob = @import("zkb").glob;

test "literal path" {
    try testing.expect(glob.match("docs/a.md", "docs/a.md"));
    try testing.expect(!glob.match("docs/a.md", "docs/b.md"));
    try testing.expect(!glob.match("docs/a.md", "docs/sub/a.md"));
}

test "star does not cross a slash" {
    try testing.expect(glob.match("*.md", "a.md"));
    try testing.expect(!glob.match("*.md", "sub/a.md"));
    try testing.expect(glob.match("*/*.md", "sub/a.md"));
    try testing.expect(!glob.match("*/*.md", "a.md"));
}

test "double star spans components, including zero" {
    try testing.expect(glob.match("**/a.md", "a.md"));
    try testing.expect(glob.match("**/a.md", "x/a.md"));
    try testing.expect(glob.match("**/a.md", "x/y/z/a.md"));
    try testing.expect(glob.match("docs/**", "docs/a.md"));
    try testing.expect(glob.match("docs/**", "docs/x/y/a.md"));
    // Zero components: `docs/**` should still accept `docs` itself, so a root
    // whose pattern is "everything under here" does not exclude the root.
    try testing.expect(glob.match("docs/**", "docs"));
    try testing.expect(!glob.match("docs/**", "other/a.md"));
}

test "question mark is exactly one byte" {
    try testing.expect(glob.match("a?.md", "ab.md"));
    try testing.expect(!glob.match("a?.md", "a.md"));
    try testing.expect(!glob.match("a?.md", "abc.md"));
}

test "backtracking within a component" {
    try testing.expect(glob.match("*-*.md", "a-b.md"));
    try testing.expect(glob.match("a*b*c", "axxbyyc"));
    try testing.expect(!glob.match("a*b*c", "axxbyy"));
    try testing.expect(glob.match("*", "anything"));
}

test "duplicate and trailing slashes mean what they look like" {
    try testing.expect(glob.match("a/b.md", "a//b.md"));
    try testing.expect(glob.match("a/*", "a/b/"));
}

// The pattern the whole feature exists for: one memory directory per project,
// under a root that also holds directories far larger than the ones wanted.
const claude_pattern = "*/memory/*.md";

test "the memory pattern selects only memory files" {
    try testing.expect(glob.match(claude_pattern, "proj-a/memory/note.md"));
    try testing.expect(!glob.match(claude_pattern, "proj-a/tool-results/dump.txt"));
    try testing.expect(!glob.match(claude_pattern, "proj-a/tool-results/dump.md"));
    try testing.expect(!glob.match(claude_pattern, "proj-a/notes.md"));
    // One level deeper than the pattern describes is a non-match, not a bonus.
    try testing.expect(!glob.match(claude_pattern, "proj-a/memory/sub/note.md"));
}

test "prefix pruning enters only what can still match" {
    // The root itself, and each project, must be entered to reach memory/.
    try testing.expect(glob.matchPrefix(claude_pattern, ""));
    try testing.expect(glob.matchPrefix(claude_pattern, "proj-a"));
    try testing.expect(glob.matchPrefix(claude_pattern, "proj-a/memory"));
    // These cannot contain a match, and skipping them is the point: on a real
    // ~/.claude they hold far more files than the memory dirs do.
    try testing.expect(!glob.matchPrefix(claude_pattern, "proj-a/tool-results"));
    try testing.expect(!glob.matchPrefix(claude_pattern, "proj-a/memory/sub"));
}

test "prefix pruning never rejects a directory with a matching descendant" {
    // A `**` pattern can match at any depth, so nothing may be pruned.
    try testing.expect(glob.matchPrefix("**/a.md", "x"));
    try testing.expect(glob.matchPrefix("**/a.md", "x/y/z"));
    try testing.expect(glob.matchPrefix("docs/**", "docs/x/y"));
    // But a mismatch in a fixed leading component is still prunable.
    try testing.expect(!glob.matchPrefix("docs/**/a.md", "other"));
}

test "an empty pattern list is no filter, not an empty allowlist" {
    const none: []const []const u8 = &.{};
    try testing.expect(glob.matchAny(none, "anything/at/all.md"));
    try testing.expect(glob.matchAnyPrefix(none, "any/dir"));
}

test "matchAny is a union" {
    const pats: []const []const u8 = &.{ "*/memory/*.md", "handoffs/*.md" };
    try testing.expect(glob.matchAny(pats, "p/memory/a.md"));
    try testing.expect(glob.matchAny(pats, "handoffs/b.md"));
    try testing.expect(!glob.matchAny(pats, "other/c.md"));
    // Pruning must consider every pattern, or the second one's directory is
    // skipped because the first one rejected it.
    try testing.expect(glob.matchAnyPrefix(pats, "handoffs"));
    try testing.expect(glob.matchAnyPrefix(pats, "p"));
    try testing.expect(!glob.matchAnyPrefix(pats, "p/other"));
}

test "an empty exclude list forbids nothing, unlike an empty include list" {
    const none: []const []const u8 = &.{};
    // The bug this pins: sharing `matchAny` for both made a scan with no
    // --exclude see 0 of 5 files, because an empty list means "no rules" and
    // "no rules" reads as allow-all for include and forbid-all for exclude.
    try testing.expect(glob.matchAny(none, "anything.md"));
    try testing.expect(!glob.matchAnyStrict(none, "anything.md"));

    const pats: []const []const u8 = &.{"agents/handoffs/**"};
    try testing.expect(glob.matchAnyStrict(pats, "agents/handoffs/a.md"));
    try testing.expect(glob.matchAnyStrict(pats, "agents/handoffs"));
    try testing.expect(!glob.matchAnyStrict(pats, "agents/notes/a.md"));
}
