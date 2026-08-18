//! `zkb collection rm NAME` — remove a collection and everything indexed under it.
//!
//! Nothing could do this before. `zkb index --collection X --root Y` created a
//! collection and there was no way back, so undoing one registered by mistake
//! meant `rm -rf ~/.zkb/index && zkb index` and re-embedding the whole corpus.
//! That is what a wrongly-registered subdirectory of `~/docs` actually cost:
//! nine and a half minutes of embedding to undo one command.
//!
//! The documents are not touched. A collection is a view onto files that keep
//! existing; dropping it removes the index's knowledge of them, nothing else.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub const Options = struct {
    name: []const u8,
};

pub fn rm(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    opts: Options,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    // zkb's own write areas would be recreated by the next ingest pass, so
    // accepting the request would report a success that does not survive.
    if (std.mem.eql(u8, opts.name, "memory") or std.mem.eql(u8, opts.name, "kb")) {
        try w.print("{s} is one of zkb's own collections\n", .{opts.name});
        try w.writeAll("it would be recreated on the next scan\n");
        return 2;
    }

    if (try viaDaemon(gpa, io, &layout, w, opts.name)) |code| return code;

    std.Io.Dir.accessAbsolute(io, layout.db, .{}) catch {
        try w.print("no index at {s}\n", .{layout.db});
        return 3;
    };
    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = try zkb.store.open(db_path, .read_write);
    defer db.close();
    var s = zkb.store.Store.init(&db);

    const id = (try s.findCollection(opts.name)) orelse {
        try w.print("no such collection: {s}\n", .{opts.name});
        try w.writeAll("zkb status lists them\n");
        return 3;
    };

    const n = try s.deleteCollection(gpa, id);
    try w.print("removed collection {s} ({d} document(s) unindexed)\n", .{ opts.name, n });
    // Said plainly, because "removed" reads as destructive and this is not.
    try w.writeAll("the files themselves are untouched\n");
    return 0;
}

/// Hand the request to the daemon when one is listening.
///
/// The ingest thread owns the only write connection, so this process must not
/// delete rows behind its back — the same invariant `zkb index` respects.
fn viaDaemon(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const zkb.paths.Layout,
    w: *Writer,
    name: []const u8,
) !?u8 {
    var c = zkb.ipc_client.Client.connect(io, layout.sock) catch return null;
    defer c.close();

    var buf: [4096]u8 = undefined;
    var pw = std.Io.Writer.fixed(&buf);
    try pw.writeAll("{\"collection\":");
    try std.json.Stringify.value(name, .{}, &pw);
    try pw.writeAll("}");

    var resp = c.call(gpa, .collection_rm, pw.buffered()) catch return null;
    defer resp.deinit(gpa);

    if (!resp.ok) {
        try w.print("{s}: {s}\n", .{ resp.code, resp.message });
        if (resp.hint) |h| try w.print("hint: {s}\n", .{h});
        return 3;
    }
    try w.print("queued removal of {s}; the daemon applies it within a second\n", .{name});
    try w.writeAll("the files themselves are untouched\n");
    return 0;
}
