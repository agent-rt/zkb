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
const csvmod = @import("csv.zig");

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

/// Everything that must exist in `collections` before anything reads it: the two
/// zkb writes itself, then every registration recorded in `data/`.
///
/// The built-ins are rows like any other, so the scan loop needs no special case
/// for them; their roots are fixed by the layout, so there is nothing for a
/// caller to override and nothing to update on a later run.
///
/// The replay is folded in here rather than given its own function on purpose.
/// All three callers — `zkb index`, the daemon's ingest thread, the daemon's
/// startup — already call this before listing roots, so folding it in costs no
/// new call site and leaves no fourth place that could forget. A registration
/// that is only replayed on some paths is worse than one that is never
/// replayed: it comes back or not depending on which command ran first.
///
/// A registration whose row already exists is upserted, not skipped, so a hand
/// edit to `collections.csv` takes effect on the next scan — the file is the
/// record, and the table is the projection.
pub fn ensureOwn(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
    layout: *const paths.Layout,
    now_ms: i64,
) !void {
    _ = try s.ensureCollectionKind("memory", layout.memory, .memory, now_ms);
    _ = try s.ensureCollectionKind("numbers", layout.data, .records, now_ms);

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // A malformed registration file must not stop zkb from running: the index is
    // still usable, and refusing to start is a worse answer than starting with
    // whatever the table already holds.
    const rows = loadRegistry(arena, io, layout) catch return;
    for (rows) |r| {
        _ = s.upsertCollection(r.collection, r.root, .documents, r.extensions, r.include, now_ms) catch continue;
    }
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

/// One collection's registration: the root and filters a caller named, in the
/// form they are stored in.
///
/// There used to be two of these — one in `cli/index_cmd.zig` for the direct
/// path and one in `daemon.zig` for the IPC path — and a parameter reached the
/// index only if it was threaded through five places by hand: both structs, the
/// serialiser, the parser, and both `upsertCollection` calls. Missing one is
/// silent in the worst possible way, because the command still succeeds: that is
/// exactly how `zkb index --root X` came to register nothing whenever a daemon
/// happened to be running.
///
/// So the wire format is not written out field by field anywhere. `writeJson`
/// and `fromLookup` both walk `std.meta.fields(Registration)`, which makes them
/// the same list by construction — an optional field added here crosses the
/// socket with no other edit, and a non-optional one fails to compile at every
/// literal until it is filled in.
pub const Registration = struct {
    collection: []const u8,
    root: []const u8,
    /// Newline-separated, or null for "the caller did not say" — which must stay
    /// distinct from `""`, meaning "match nothing" (see `joinList`).
    extensions: ?[]const u8 = null,
    include: ?[]const u8 = null,

    /// The wire name for `collection`, kept as it is because an older daemon on
    /// the other end of the socket during an upgrade still reads that key.
    const default_collection = "docs";

    /// Pairs with `fromLookup` only. A registration built by the CLI holds arena
    /// memory and must not be freed with a gpa.
    ///
    /// Walks the field list for the same reason the wire pair does. This one was
    /// written out by hand at first, and adding a fifth field to try the claim
    /// that a new field costs nothing leaked exactly one allocation — the same
    /// shape as the bug this whole change is about, surviving in the last place
    /// still enumerating fields by hand.
    pub fn deinit(self: Registration, gpa: std.mem.Allocator) void {
        inline for (std.meta.fields(Registration)) |f| {
            const v: ?[]const u8 = @field(self, f.name);
            if (v) |owned| gpa.free(owned);
        }
    }

    pub fn writeJson(self: Registration, w: *std.Io.Writer) !void {
        try w.writeAll("{");
        var first = true;
        inline for (std.meta.fields(Registration)) |f| {
            const v: ?[]const u8 = @field(self, f.name);
            if (v) |val| {
                if (!first) try w.writeAll(",");
                first = false;
                try std.json.Stringify.value(f.name, .{}, w);
                try w.writeAll(":");
                try std.json.Stringify.value(val, .{}, w);
            }
        }
        try w.writeAll("}");
    }

    /// Read from anything with `str(name) ?[]const u8` — a `proto.Request` in the
    /// daemon, a parsed object in a test. Taken as `anytype` so this module never
    /// imports the wire protocol: the field list is the contract, not the socket.
    ///
    /// Null when no root was named, which is the request asking for a rescan of
    /// what is already registered rather than for a new registration.
    pub fn fromLookup(gpa: std.mem.Allocator, src: anytype) !?Registration {
        if (src.str("root") == null) return null;
        var r: Registration = undefined;
        inline for (std.meta.fields(Registration)) |f| {
            const raw = src.str(f.name);
            @field(r, f.name) = switch (@typeInfo(f.type)) {
                .optional => if (raw) |v| try gpa.dupe(u8, v) else null,
                else => try gpa.dupe(u8, raw orelse default_collection),
            };
        }
        return r;
    }

    /// Write the collection row, record the registration in `data/`, and return
    /// the id. The one place either path turns a registration into stored state.
    ///
    /// The file write lives here for the reason the whole struct exists: two
    /// call sites, and a second step next to each of them is a step one of them
    /// eventually does not take. `gpa`, `io` and `layout` are not optional for
    /// the same reason — a caller with nowhere to record is a caller whose
    /// registration disappears the next time the index is rebuilt, and that was
    /// the bug.
    ///
    /// The row is written even if recording fails: a registration the index can
    /// act on now, that a later reset loses, still beats one that never
    /// happened. The error is returned so the caller can say so.
    pub fn apply(
        self: Registration,
        gpa: std.mem.Allocator,
        io: std.Io,
        layout: *const paths.Layout,
        s: *store.Store,
        kind: store.Store.Kind,
        now_ms: i64,
    ) !i64 {
        const id = try s.upsertCollection(self.collection, self.root, kind, self.extensions, self.include, now_ms);
        try recordRegistration(gpa, io, layout, self);
        return id;
    }

    /// The scan filters this registration implies, on top of its kind's defaults.
    pub fn filters(
        self: Registration,
        arena: std.mem.Allocator,
        cid: i64,
        kind: store.Store.Kind,
    ) !scan.Filters {
        return filtersFor(arena, .{
            .id = cid,
            .name = self.collection,
            .root = self.root,
            .kind = kind,
            .extensions = self.extensions,
            .include = self.include,
        });
    }
};

// ---------------------------------------------------------------------------
// The registration file
// ---------------------------------------------------------------------------

/// `~/.zkb/data/collections.csv` — what a collection registration *is*.
///
/// Until this existed, a registration lived only in the index, and DESIGN.md's
/// first sentence ("everything in SQLite is a projection of files that are
/// still on disk") was false for the one row that decides which files get
/// projected at all. Measured on a scratch home: register `notes` and `proj`,
/// run the reset the README puts in its opening lines —
///
///     rm -rf ~/.zkb/index && zkb index
///
/// — and afterwards only `docs`, `memory` and `numbers` exist. Both user
/// collections are gone, nothing says so, and the exit code is 0. On a real
/// machine that is five of eight collections, after which `recall --scope` and
/// `search --collection agent-memory` answer emptily and plausibly.
///
/// Only collections a caller registered are written here. `memory` and
/// `numbers` are not: their roots come from the layout, so a file recording
/// them could only ever disagree with it.
const registry_file = "collections.csv";

/// `kind` is deliberately absent. Every registration that reaches this file is
/// a documents collection, and a hand-edited `kind` would be a way to turn one
/// into a `memory` collection — precisely the transition `upsertCollection`
/// refuses to make, for reasons written there.
const registry_columns = [_][]const u8{ "name", "root", "extensions", "include" };

pub fn registryPath(arena: std.mem.Allocator, layout: *const paths.Layout) ![]u8 {
    return std.fmt.allocPrint(arena, "{s}/{s}", .{ layout.data, registry_file });
}

/// Everything the file says, in the order it says it. Missing file is not an
/// error: a machine that has only ever used the default collection has nothing
/// to record yet.
pub fn loadRegistry(
    arena: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
) ![]Registration {
    const path = try registryPath(arena, layout);
    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return &.{};
    defer file.close(io);

    const size = (try file.stat(io)).size;
    const src = try arena.alloc(u8, @intCast(size));
    var rbuf: [4096]u8 = undefined;
    var reader = file.reader(io, &rbuf);
    try reader.interface.readSliceAll(src);

    var table = try csvmod.parse(arena, src);
    const name_col = table.columnIndex("name") orelse return &.{};
    const root_col = table.columnIndex("root") orelse return &.{};
    const ext_col = table.columnIndex("extensions");
    const inc_col = table.columnIndex("include");

    var out: std.ArrayList(Registration) = .empty;
    for (table.rows) |row| {
        const name = std.mem.trim(u8, row[name_col], " \t");
        const root = std.mem.trim(u8, row[root_col], " \t");
        if (name.len == 0 or root.len == 0) continue;
        out.append(arena, .{
            .collection = try arena.dupe(u8, name),
            .root = try arena.dupe(u8, root),
            // An empty cell is "the caller did not say", which is the NULL the
            // column means — not "match nothing". `joinList` draws the same
            // line, and the two have to agree or a reset would narrow a
            // collection to zero extensions without anyone asking.
            .extensions = try optionalCell(arena, ext_col, row),
            .include = try optionalCell(arena, inc_col, row),
        }) catch return error.OutOfMemory;
    }
    return out.items;
}

fn optionalCell(
    arena: std.mem.Allocator,
    col: ?usize,
    row: []const []const u8,
) !?[]const u8 {
    const i = col orelse return null;
    const v = std.mem.trim(u8, row[i], " \t");
    if (v.len == 0) return null;
    return try arena.dupe(u8, v);
}

/// Record one registration, replacing any row with the same name.
///
/// The whole file is rewritten through a temporary and renamed, the way
/// `facts` migrates its own: a dozen rows is not worth an in-place edit, and a
/// half-written registration file would take the collections with it.
pub fn recordRegistration(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
    reg: Registration,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var rows = std.ArrayList(Registration).fromOwnedSlice(try loadRegistry(arena, io, layout));
    var replaced = false;
    for (rows.items) |*r| {
        if (std.mem.eql(u8, r.collection, reg.collection)) {
            r.* = reg;
            replaced = true;
        }
    }
    if (!replaced) try rows.append(arena, reg);
    try writeRegistry(arena, io, layout, rows.items);
}

/// Drop a registration. Silent when it was not there: `collection rm` on
/// something only the index knew about must still leave the file consistent.
pub fn forgetRegistration(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
    name: []const u8,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const rows = try loadRegistry(arena, io, layout);
    var keep: std.ArrayList(Registration) = .empty;
    for (rows) |r| {
        if (std.mem.eql(u8, r.collection, name)) continue;
        try keep.append(arena, r);
    }
    if (keep.items.len == rows.len) return;
    try writeRegistry(arena, io, layout, keep.items);
}

fn writeRegistry(
    arena: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
    rows: []const Registration,
) !void {
    const path = try registryPath(arena, layout);
    const tmp = try std.fmt.allocPrint(arena, "{s}.writing", .{path});
    {
        var out = try std.Io.Dir.createFileAbsolute(io, tmp, .{ .truncate = true });
        defer out.close(io);
        var buf: [4096]u8 = undefined;
        var writer = out.writer(io, &buf);
        const w = &writer.interface;
        try csvmod.writeRow(w, &registry_columns);
        for (rows) |r| {
            try csvmod.writeRow(w, &.{
                r.collection,
                r.root,
                r.extensions orelse "",
                r.include orelse "",
            });
        }
        try w.flush();
    }
    try std.Io.Dir.renameAbsolute(tmp, path, io);
}

/// Drop a collection from both places it is recorded.
///
/// Registration has one funnel (`Registration.apply`); removal had two — the
/// CLI's direct path and the daemon's queued one — and a second step beside
/// each is a step one of them eventually does not take. Here the shape is the
/// worse direction of the bug this file exists to fix: a collection dropped
/// through the daemon but left in `collections.csv` would come back at the next
/// `ensureOwn`, which reads as the removal silently failing.
///
/// Returns the number of documents that were unindexed with it.
pub fn dropCollection(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const paths.Layout,
    s: *store.Store,
    name: []const u8,
    id: i64,
) !usize {
    const n = try s.deleteCollection(gpa, id);
    try forgetRegistration(gpa, io, layout, name);
    return n;
}
