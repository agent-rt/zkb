//! Folding several given roots into the single root a collection has, and
//! resolving a stored row's scan configuration.

const std = @import("std");
const testing = std.testing;
const zkb = @import("zkb");
const roots = zkb.roots;

fn arena(a: *std.heap.ArenaAllocator) std.mem.Allocator {
    return a.allocator();
}

test "one root folds to itself with no patterns" {
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    const f = try roots.fold(arena(&a), &.{"/home/u/docs"});
    try testing.expectEqualStrings("/home/u/docs", f.root);
    try testing.expectEqual(@as(usize, 0), f.include.len);
}

test "siblings fold to their parent, one pattern each" {
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    const f = try roots.fold(arena(&a), &.{
        "/home/u/.claude/projects/alpha/memory",
        "/home/u/.claude/projects/beta/memory",
    });
    try testing.expectEqualStrings("/home/u/.claude/projects", f.root);
    try testing.expectEqual(@as(usize, 2), f.include.len);
    try testing.expectEqualStrings("alpha/memory/**", f.include[0]);
    try testing.expectEqualStrings("beta/memory/**", f.include[1]);
}

test "the ancestor is a component boundary, not a byte prefix" {
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    // `/p/q/abc` and `/p/q/abd` share the bytes `/p/q/ab`, which is not a directory.
    const f = try roots.fold(arena(&a), &.{ "/p/q/abc/m", "/p/q/abd/m" });
    try testing.expectEqualStrings("/p/q", f.root);
    try testing.expectEqualStrings("abc/m/**", f.include[0]);
    try testing.expectEqualStrings("abd/m/**", f.include[1]);
}

test "a root that contains the others swallows the patterns" {
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    // Asking for a directory and something inside it is asking for the directory.
    const f = try roots.fold(arena(&a), &.{ "/home/u/docs", "/home/u/docs/sub" });
    try testing.expectEqualStrings("/home/u/docs", f.root);
    try testing.expectEqual(@as(usize, 0), f.include.len);
}

test "disjoint roots are refused rather than scanned from the top" {
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    // Ancestor would be `/`, and walking from there looks like a hang.
    try testing.expectError(error.RootsTooDisjoint, roots.fold(arena(&a), &.{ "/opt/a", "/var/b" }));
    // Ancestor `/home` is one component: still the whole machine's worth of files.
    try testing.expectError(error.RootsTooDisjoint, roots.fold(arena(&a), &.{ "/home/alice/d", "/home/bob/d" }));
    try testing.expectError(error.NoRoots, roots.fold(arena(&a), &.{}));
}

test "trailing slashes do not change the fold" {
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    const f = try roots.fold(arena(&a), &.{ "/p/q/a/memory/", "/p/q/b/memory/" });
    try testing.expectEqualStrings("/p/q", f.root);
    try testing.expectEqualStrings("a/memory/**", f.include[0]);
}

test "a stored list round-trips, and an empty one stays null" {
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    const al = arena(&a);

    const joined = (try roots.joinList(al, &.{ "*/memory/**", "handoffs/*.md" })).?;
    const back = try roots.splitList(al, joined);
    try testing.expectEqual(@as(usize, 2), back.len);
    try testing.expectEqualStrings("*/memory/**", back[0]);
    try testing.expectEqualStrings("handoffs/*.md", back[1]);

    // Null, not "", so a row can still mean "the kind's default".
    try testing.expect((try roots.joinList(al, &.{})) == null);
    // A trailing separator is not a pattern that matches nothing.
    try testing.expectEqual(@as(usize, 1), (try roots.splitList(al, "*.md\n")).len);
}

test "kind decides the defaults a row overrides" {
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    const al = arena(&a);

    // A documents row with no overrides scans exactly as it did before v7.
    const plain = try roots.filtersFor(al, .{
        .id = 1,
        .name = "docs",
        .root = "/d",
        .kind = .documents,
        .extensions = null,
        .include = null,
    });
    try testing.expectEqual(@as(usize, 3), plain.extensions.len);
    try testing.expectEqual(@as(usize, 0), plain.include.len);

    const narrowed = try roots.filtersFor(al, .{
        .id = 2,
        .name = "agent-memory",
        .root = "/c",
        .kind = .documents,
        .extensions = ".md",
        .include = "*/memory/**",
    });
    try testing.expectEqual(@as(usize, 1), narrowed.extensions.len);
    try testing.expectEqualStrings(".md", narrowed.extensions[0]);
    try testing.expectEqualStrings("*/memory/**", narrowed.include[0]);

    // memory keeps its exclude_dirs: `archive/` holds forgotten memories, and a
    // row must not be able to bring them back by overriding the file selection.
    const mem = try roots.filtersFor(al, .{
        .id = 3,
        .name = "memory",
        .root = "/m",
        .kind = .memory,
        .extensions = ".md\n.txt",
        .include = null,
    });
    var has_archive = false;
    for (mem.skip_dirs) |d| if (std.mem.eql(u8, d, "archive")) {
        has_archive = true;
    };
    try testing.expect(has_archive);

    // A stored empty string degrades to the default rather than selecting no
    // files: a collection that silently indexes nothing is the worse failure.
    const empty = try roots.filtersFor(al, .{
        .id = 4,
        .name = "x",
        .root = "/x",
        .kind = .documents,
        .extensions = "",
        .include = null,
    });
    try testing.expectEqual(@as(usize, 3), empty.extensions.len);
}

test "an ignored directory is prunable, not just its files" {
    // 仍然是 .zkbignore 依赖的 glob 性质：目录层剪枝要能在目录自身上判定。
    // The prune check asks matchAny on the directory path itself. A trailing `**`
    // matches zero components, so `agents/handoffs/**` stops the walk at the
    // directory rather than listing it and rejecting each file.
    try testing.expect(zkb.glob.match("agents/handoffs/**", "agents/handoffs"));
    try testing.expect(zkb.glob.match("agents/handoffs/**", "agents/handoffs/a.md"));
    // A pattern that can only match deeper must not prune an ancestor.
    try testing.expect(!zkb.glob.match("**/draft.md", "agents/handoffs"));
}
