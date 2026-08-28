//! What a subtree of the corpus *is*, in one sentence, attached to every result
//! that comes out of it.
//!
//! A search result is a path and a chunk. The caller — a language model that has
//! never seen this corpus — gets `agent-memory/…/feedback_verify_the_instrument.md`
//! and a paragraph, and has to guess whether that is a note, a decision, a draft
//! or somebody else's document quoted in passing. A one-line description of the
//! tree it came from is the cheapest thing that answers the question, and it is
//! knowledge only the corpus owner has.
//!
//! Borrowed from `tobi/qmd`, whose author calls it that project's key feature.
//! Two things are deliberately different:
//!
//!   * **Every matching prefix is returned, general to specific** — not only the
//!     longest. qmd's README says longest-match-wins and its code accumulates;
//!     accumulating is the better of the two, because "these are my notes" and
//!     "this subtree is work-related" are both true and neither implies the
//!     other.
//!   * **It never reaches ranking.** In qmd the context is attached at result
//!     assembly and the reranker only ever sees chunk text; zkb has no reranker,
//!     and this must stay out of RRF regardless. A description is a property of
//!     a whole subtree, so scoring with it would push every document in a
//!     well-described tree above every document in an undescribed one — a
//!     collection-level bias wearing the clothes of relevance.
//!
//! Descriptions live in `~/.zkb/data/contexts.csv`, for the same reason
//! collection registrations do (see `ingest/roots.zig`): they are authored, not
//! derived, so an index rebuild must not lose them.

const std = @import("std");
const csvmod = @import("ingest/csv.zig");
const paths = @import("util/paths.zig");

pub const registry_file = "contexts.csv";

const columns = [_][]const u8{ "collection", "prefix", "text" };

pub const Entry = struct {
    collection: []const u8,
    /// Path prefix relative to the collection root, matched on component
    /// boundaries. Empty means the whole collection.
    prefix: []const u8,
    text: []const u8,
};

pub const Map = struct {
    entries: []Entry,

    pub const empty: Map = .{ .entries = &.{} };

    /// Every description that applies to `rel_path`, general first.
    ///
    /// Ordered by prefix length so a caller reading top to bottom goes from the
    /// widest claim to the narrowest, which is the order the sentences make
    /// sense in. Ties keep file order, so two descriptions at the same depth
    /// read as the author arranged them.
    pub fn forPath(
        self: Map,
        arena: std.mem.Allocator,
        collection: []const u8,
        rel_path: []const u8,
    ) ![]const []const u8 {
        var hits: std.ArrayList(Entry) = .empty;
        for (self.entries) |e| {
            if (!std.mem.eql(u8, e.collection, collection)) continue;
            if (!prefixMatches(e.prefix, rel_path)) continue;
            try hits.append(arena, e);
        }
        std.mem.sort(Entry, hits.items, {}, struct {
            fn less(_: void, a: Entry, b: Entry) bool {
                return a.prefix.len < b.prefix.len;
            }
        }.less);

        var out: std.ArrayList([]const u8) = .empty;
        for (hits.items) |e| try out.append(arena, e.text);
        return out.items;
    }

    /// The same thing as one string, or null when nothing matched. What the
    /// renderers want; `forPath` is for callers that need the pieces.
    pub fn joinedForPath(
        self: Map,
        arena: std.mem.Allocator,
        collection: []const u8,
        rel_path: []const u8,
    ) !?[]const u8 {
        const parts = try self.forPath(arena, collection, rel_path);
        if (parts.len == 0) return null;
        return try std.mem.join(arena, " · ", parts);
    }
};

/// Does `prefix` cover `rel_path`?
///
/// An empty prefix covers everything in the collection. Otherwise the match is
/// on a component boundary, so `research` covers `research/qmd.md` and never
/// `research-notes/qmd.md` — the same rule as `bench.pathMatches` and
/// `paths.isInside`, because a prefix that matched half a directory name would
/// attach the wrong sentence to a real result and nothing would look wrong.
pub fn prefixMatches(prefix: []const u8, rel_path: []const u8) bool {
    const p = std.mem.trim(u8, prefix, "/");
    if (p.len == 0) return true;
    if (!std.mem.startsWith(u8, rel_path, p)) return false;
    if (rel_path.len == p.len) return true;
    return rel_path[p.len] == '/';
}

pub fn registryPath(arena: std.mem.Allocator, layout: *const paths.Layout) ![]u8 {
    return std.fmt.allocPrint(arena, "{s}/{s}", .{ layout.data, registry_file });
}

/// Load every description. A missing file is not an error — most corpora have
/// none, and a knowledge base with no descriptions must work exactly as it did
/// before this existed.
pub fn load(
    arena: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
) !Map {
    const path = try registryPath(arena, layout);
    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return .empty;
    defer file.close(io);

    const size = (try file.stat(io)).size;
    const src = try arena.alloc(u8, @intCast(size));
    var rbuf: [4096]u8 = undefined;
    var reader = file.reader(io, &rbuf);
    try reader.interface.readSliceAll(src);

    var table = try csvmod.parse(arena, src);
    const c_col = table.columnIndex("collection") orelse return .empty;
    const p_col = table.columnIndex("prefix") orelse return .empty;
    const t_col = table.columnIndex("text") orelse return .empty;

    var out: std.ArrayList(Entry) = .empty;
    for (table.rows) |row| {
        const collection = std.mem.trim(u8, row[c_col], " \t");
        const text = std.mem.trim(u8, row[t_col], " \t");
        // A row with no text is a description that says nothing; a row with no
        // collection cannot be attached to anything. Both are half-finished
        // edits, and skipping beats attaching an empty line to every result.
        if (collection.len == 0 or text.len == 0) continue;
        try out.append(arena, .{
            .collection = try arena.dupe(u8, collection),
            .prefix = try arena.dupe(u8, std.mem.trim(u8, row[p_col], " \t/")),
            .text = try arena.dupe(u8, text),
        });
    }
    return .{ .entries = out.items };
}

/// Add or replace the description at one (collection, prefix).
pub fn record(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
    entry: Entry,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const map = try load(arena, io, layout);
    var rows: std.ArrayList(Entry) = .empty;
    var replaced = false;
    const want_prefix = std.mem.trim(u8, entry.prefix, "/");
    for (map.entries) |e| {
        if (std.mem.eql(u8, e.collection, entry.collection) and
            std.mem.eql(u8, e.prefix, want_prefix))
        {
            try rows.append(arena, .{ .collection = entry.collection, .prefix = want_prefix, .text = entry.text });
            replaced = true;
        } else {
            try rows.append(arena, e);
        }
    }
    if (!replaced) {
        try rows.append(arena, .{ .collection = entry.collection, .prefix = want_prefix, .text = entry.text });
    }
    try write(arena, io, layout, rows.items);
}

/// Remove one description. Returns whether anything was there.
pub fn forget(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
    collection: []const u8,
    prefix: []const u8,
) !bool {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const map = try load(arena, io, layout);
    const want_prefix = std.mem.trim(u8, prefix, "/");
    var keep: std.ArrayList(Entry) = .empty;
    for (map.entries) |e| {
        if (std.mem.eql(u8, e.collection, collection) and std.mem.eql(u8, e.prefix, want_prefix)) continue;
        try keep.append(arena, e);
    }
    if (keep.items.len == map.entries.len) return false;
    try write(arena, io, layout, keep.items);
    return true;
}

fn write(
    arena: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
    rows: []const Entry,
) !void {
    const path = try registryPath(arena, layout);
    const tmp = try std.fmt.allocPrint(arena, "{s}.writing", .{path});
    {
        var out = try std.Io.Dir.createFileAbsolute(io, tmp, .{ .truncate = true });
        defer out.close(io);
        var buf: [4096]u8 = undefined;
        var writer = out.writer(io, &buf);
        const w = &writer.interface;
        try csvmod.writeRow(w, &columns);
        for (rows) |e| try csvmod.writeRow(w, &.{ e.collection, e.prefix, e.text });
        try w.flush();
    }
    try std.Io.Dir.renameAbsolute(tmp, path, io);
}

/// Split a `zkb://collection/prefix` reference — or a bare `collection/prefix` —
/// into its two halves.
///
/// The scheme is accepted because that is the form these paths take everywhere
/// else in zkb, and a caller holding one should not have to take it apart.
pub fn splitRef(raw: []const u8) struct { collection: []const u8, prefix: []const u8 } {
    const rel = paths.relFromUri(raw);
    const i = std.mem.indexOfScalar(u8, rel, '/') orelse
        return .{ .collection = rel, .prefix = "" };
    return .{ .collection = rel[0..i], .prefix = std.mem.trim(u8, rel[i + 1 ..], "/") };
}
