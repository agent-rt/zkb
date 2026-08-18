//! The list of roots to scan, resolved from the database.
//!
//! This used to be a literal array — one in the daemon, another in `zkb index`.
//! Two consequences, both found by measurement rather than reading:
//!
//!   * `zkb index --root X --collection Y` with a daemon running printed
//!     "daemon rescanning ..." and then success, having created no collection and
//!     indexed nothing. The request body was the constant `"{}"`, so the daemon
//!     only ever woke the ingest thread, which walked its own three roots.
//!   * A collection registered while no daemon was running (the only way that
//!     worked) was scanned exactly once. Nothing rescanned it, because the
//!     rescanner's list was a literal that did not mention it.
//!
//! So the roots live in `collections` and every scanner reads them from there.
//! Adding a root is a write, not an edit.

const std = @import("std");
const scan = @import("scan.zig");
const store = @import("../db/store.zig");
const memory = @import("../memory.zig");
const facts = @import("../facts.zig");
const paths = @import("../util/paths.zig");

pub const Root = struct {
    id: i64,
    name: []const u8,
    path: []const u8,
    kind: store.Store.Kind,
    filters: scan.Filters,
};

/// How a collection is parsed follows from its kind, so the kind decides the
/// defaults. A row may override the file selection on top of these; it may not
/// override `exclude_dirs`, because for memory that list is load-bearing —
/// `archive/` holding forgotten memories must stay unscanned (see memory.zig).
pub fn baseFilters(kind: store.Store.Kind) scan.Filters {
    return switch (kind) {
        .documents => .{},
        .memory => memory.scan_filters,
        .records => facts.scan_filters,
    };
}

/// The separator for a stored list. Newline rather than comma because these hold
/// glob patterns, and a comma is a plausible character inside one.
pub const list_sep = "\n";

/// Split a stored list, skipping empty entries so a trailing newline is not a
/// pattern that matches nothing.
pub fn splitList(arena: std.mem.Allocator, s: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitSequence(u8, s, list_sep);
    while (it.next()) |part| {
        const t = std.mem.trim(u8, part, " \t\r");
        if (t.len == 0) continue;
        try out.append(arena, try arena.dupe(u8, t));
    }
    return out.items;
}

/// Join a list for storage. Returns null for an empty list, so "the caller did
/// not say" and "the caller said nothing matches" stay distinguishable — a NULL
/// column means the kind's default, an empty string would mean match nothing.
pub fn joinList(arena: std.mem.Allocator, items: []const []const u8) !?[]const u8 {
    if (items.len == 0) return null;
    return try std.mem.join(arena, list_sep, items);
}

/// Apply a row's stored overrides on top of its kind's defaults.
pub fn filtersFor(arena: std.mem.Allocator, row: store.Store.CollectionRow) !scan.Filters {
    var f = baseFilters(row.kind);
    if (row.extensions) |raw| {
        const exts = try splitList(arena, raw);
        // An explicitly stored empty list would select no files at all. Treating
        // it as "no override" instead means a bad write degrades to the default
        // rather than to a collection that silently indexes nothing.
        if (exts.len != 0) f.extensions = exts;
    }
    if (row.include) |raw| f.include = try splitList(arena, raw);
    return f;
}

pub const FoldError = error{
    /// The given paths share no useful ancestor, so the collection root would be
    /// `/` or a single top-level directory and the scan would walk the machine.
    RootsTooDisjoint,
    NoRoots,
    OutOfMemory,
};

/// The shape a collection is stored in: exactly one root, plus include patterns.
pub const Folded = struct {
    root: []const u8,
    include: []const []const u8,
};

/// Fold several given roots into the one root a collection has.
///
/// `docs.rel_path` is relative to a single root, and keeping it that way is worth
/// more than literally storing a list: the deepest shared ancestor plus one
/// include pattern per path selects exactly the same files, and every path in the
/// index stays resolvable by one join.
///
/// The practical entry point is a shell glob — `--root ~/.claude/projects/*/memory`
/// expands before zkb sees it, so what arrives is a list of absolute paths.
/// Note what that means: the resulting patterns name the directories that existed
/// at registration time. `--include '*/memory/**'` under the shared root is the
/// form that also picks up projects created later.
pub fn fold(arena: std.mem.Allocator, given: []const []const u8) FoldError!Folded {
    if (given.len == 0) return error.NoRoots;
    if (given.len == 1) return .{ .root = given[0], .include = &.{} };

    var root = given[0];
    for (given[1..]) |p| root = commonAncestor(root, p);

    // Refuse rather than walk: with roots in unrelated trees the ancestor is `/`
    // or `/Users`, and scanning from there would look like a hang, not an error.
    if (componentCount(root) < 2) return error.RootsTooDisjoint;

    var include: std.ArrayList([]const u8) = .empty;
    for (given) |p| {
        const rel = std.mem.trim(u8, p[root.len..], "/");
        // A given path equal to the root means "everything under it", which no
        // pattern needs to say — and an empty pattern would match nothing.
        if (rel.len == 0) return .{ .root = root, .include = &.{} };
        try include.append(arena, try std.fmt.allocPrint(arena, "{s}/**", .{rel}));
    }
    return .{ .root = root, .include = include.items };
}

/// Deepest directory that is a prefix of both, on component boundaries — so
/// `/a/bcd` and `/a/bce` share `/a`, not `/a/bc`.
fn commonAncestor(a: []const u8, b: []const u8) []const u8 {
    var last_slash: usize = 0;
    var i: usize = 0;
    while (i < a.len and i < b.len and a[i] == b[i]) : (i += 1) {
        if (a[i] == '/') last_slash = i;
    }
    // A full match of the shorter path at a boundary makes it the ancestor.
    if (i == a.len and (i == b.len or b[i] == '/')) return a;
    if (i == b.len and (i == a.len or a[i] == '/')) return b;
    if (last_slash == 0) return a[0..1];
    return a[0..last_slash];
}

fn componentCount(path: []const u8) usize {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, path, '/');
    while (it.next()) |_| n += 1;
    return n;
}

/// The two collections zkb writes itself. Rows like any other, so the scan loop
/// needs no special case for them; this only guarantees they exist.
///
/// Their roots are fixed by the layout, so there is nothing for a caller to
/// override and nothing to update on a later run.
pub fn ensureOwn(s: *store.Store, layout: *const paths.Layout, now_ms: i64) !void {
    _ = try s.ensureCollectionKind("memory", layout.memory, .memory, now_ms);
    _ = try s.ensureCollectionKind("kb", layout.data, .records, now_ms);
}

/// Guarantee the documents collection exists.
///
/// `explicit` is whether the caller was actually told a root. When it was not —
/// the daemon starting with no `--root` and falling back to `~/docs` — a stored
/// root must win, or every daemon restart would quietly undo a
/// `zkb index --root elsewhere --collection docs`. When it was, the caller's root
/// is the answer, because ignoring a root the user just typed is the failure this
/// whole change is about.
pub fn ensureDocs(
    s: *store.Store,
    name: []const u8,
    root: []const u8,
    explicit: bool,
    now_ms: i64,
) !i64 {
    if (explicit) return s.upsertCollection(name, root, .documents, null, null, now_ms);
    return s.ensureCollectionKind(name, root, .documents, now_ms);
}

/// Every root to scan, in id order.
///
/// Ordered by id, which puts the built-ins first: documents is the bulk and
/// wants to start first, and memory and records are small enough that their
/// position barely matters. A user's own collections follow in the order they
/// were registered, which is at least stable and explainable.
///
/// A root whose directory is missing is dropped rather than reported: an
/// unmounted volume or a deleted project should not stall the loop, and the docs
/// rows survive so the collection recovers when the path comes back.
pub fn list(
    arena: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
) ![]Root {
    const rows = try s.listCollections(arena);
    var out: std.ArrayList(Root) = .empty;
    for (rows) |row| {
        std.Io.Dir.accessAbsolute(io, row.root, .{}) catch continue;
        try out.append(arena, .{
            .id = row.id,
            .name = row.name,
            .path = row.root,
            .kind = row.kind,
            .filters = try filtersFor(arena, row),
        });
    }
    return out.items;
}
