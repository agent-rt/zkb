//! `zkb context` — say what a subtree of the corpus is.
//!
//! No daemon path. These write `~/.zkb/data/contexts.csv` and nothing else, and
//! the daemon reads that file on every search rather than holding it — so there
//! is no cached state to invalidate and nothing for an IPC round trip to buy.
//! `collection rm` needs the daemon because it deletes index rows; this does not
//! touch the index at all.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub fn add(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    ref: []const u8,
    text: []const u8,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);
    try layout.ensureDirs(io);

    const split = zkb.contexts.splitRef(ref);
    if (split.collection.len == 0) {
        try w.writeAll("usage: zkb context add <zkb://collection/prefix> <text>\n");
        return 2;
    }
    if (text.len == 0) {
        try w.writeAll("a context with no text describes nothing\n");
        return 2;
    }

    // Checked against the index rather than accepted blindly: a typo in the
    // collection name writes a row that matches no result, and nothing later
    // would ever say so — the description simply never appears.
    if (try knownCollection(gpa, io, &layout, split.collection)) |known| {
        if (!known) {
            try w.print("unknown collection: {s}\n", .{split.collection});
            try w.writeAll("zkb status lists them\n");
            return 2;
        }
    }

    try zkb.contexts.record(gpa, io, &layout, .{
        .collection = split.collection,
        .prefix = split.prefix,
        .text = text,
    });
    if (split.prefix.len == 0) {
        try w.print("context set for the whole of {s}\n", .{split.collection});
    } else {
        try w.print("context set for {s}/{s}\n", .{ split.collection, split.prefix });
    }
    return 0;
}

pub fn list(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const map = try zkb.contexts.load(arena, io, &layout);
    if (map.entries.len == 0) {
        const path = try zkb.contexts.registryPath(arena, &layout);
        try w.print("no contexts. add one:\n", .{});
        try w.writeAll("  zkb context add zkb://docs/research \"技术调研与外部资料摘录\"\n");
        try w.print("they live in {s}\n", .{path});
        return 0;
    }
    for (map.entries) |e| {
        if (e.prefix.len == 0) {
            try w.print("zkb://{s}\n", .{e.collection});
        } else {
            try w.print("zkb://{s}/{s}\n", .{ e.collection, e.prefix });
        }
        try w.print("    {s}\n", .{e.text});
    }
    return 0;
}

pub fn rm(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    ref: []const u8,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    const split = zkb.contexts.splitRef(ref);
    if (split.collection.len == 0) {
        try w.writeAll("usage: zkb context rm <zkb://collection/prefix>\n");
        return 2;
    }
    const gone = try zkb.contexts.forget(gpa, io, &layout, split.collection, split.prefix);
    if (!gone) {
        try w.print("no context at {s}\n", .{ref});
        try w.writeAll("zkb context list shows them\n");
        return 3;
    }
    try w.print("removed context for {s}\n", .{ref});
    return 0;
}

/// Whether the collection exists, or null when the index cannot be consulted.
///
/// Null rather than false: with no index yet there is nothing to check against,
/// and refusing to record a description because the corpus has not been indexed
/// would make the order of two unrelated commands matter.
fn knownCollection(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const zkb.paths.Layout,
    name: []const u8,
) !?bool {
    std.Io.Dir.accessAbsolute(io, layout.db, .{}) catch return null;
    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch return null;
    defer db.close();
    var s = zkb.store.Store.init(&db);
    return (try s.findCollection(name)) != null;
}
