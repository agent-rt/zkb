//! `scan.reconcile` against a real directory tree.
//!
//! This function had no test coverage at all — nothing in the suite touched the
//! filesystem — which is why a rename-detection bug reached a real corpus and was
//! only found by counting files on disk against rows in the index.

const std = @import("std");
const testing = std.testing;
const zkb = @import("zkb");
const store = zkb.store;
const scan = zkb.scan;

const gpa = testing.allocator;

/// A throwaway directory tree, outside this repository.
///
/// Deliberately not `std.testing.tmpDir`: that builds under `.zig-cache/tmp`, and
/// this project's own `.gitignore` excludes `.zig-cache/`. Since `reconcile` now
/// honours a `.gitignore` above its root, a tree there is correctly ignored in
/// full and every assertion sees an empty scan. The feature working as designed
/// is not a place to put a fixture.
const Tree = struct {
    io: std.Io,
    root: []u8,
    handle: std.Io.Dir,

    fn init(io: std.Io) !Tree {
        // Unique per call so tests can run in any order without colliding; the
        // address of a stack local is enough entropy for a temp directory name.
        var seed: usize = @intFromPtr(&io);
        seed ^= @as(usize, @bitCast(@as(isize, @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds))));
        const root = try std.fmt.allocPrint(gpa, "/tmp/zkb-scan-test-{x}", .{seed});
        const h = try std.Io.Dir.cwd().createDirPathOpen(io, root, .{});
        return .{ .io = io, .root = root, .handle = h };
    }

    fn deinit(self: *Tree) void {
        self.handle.close(self.io);
        std.Io.Dir.cwd().deleteTree(self.io, self.root) catch {};
        gpa.free(self.root);
    }

    fn write(self: *Tree, rel: []const u8, body: []const u8) !void {
        if (std.fs.path.dirname(rel)) |sub| {
            var d = try self.handle.createDirPathOpen(self.io, sub, .{});
            d.close(self.io);
        }
        var f = try self.handle.createFile(self.io, rel, .{});
        defer f.close(self.io);
        var buf: [512]u8 = undefined;
        var wr = f.writer(self.io, &buf);
        try wr.interface.writeAll(body);
        try wr.interface.flush();
    }

    fn remove(self: *Tree, rel: []const u8) !void {
        try self.handle.deleteFile(self.io, rel);
    }

    /// `target` is written into the link verbatim, so a test can choose between
    /// an absolute escape and a relative `../` one — they take different paths
    /// through the resolver.
    fn symlink(self: *Tree, target: []const u8, rel: []const u8) !void {
        if (std.fs.path.dirname(rel)) |sub| {
            var d = try self.handle.createDirPathOpen(self.io, sub, .{});
            d.close(self.io);
        }
        try self.handle.symLink(self.io, target, rel, .{});
    }

    fn join(self: *Tree, arena: std.mem.Allocator, rel: []const u8) ![]u8 {
        return std.fs.path.join(arena, &.{ self.root, rel });
    }
};

fn openMem() !zkb.sqlite.Db {
    return store.open(":memory:", .read_write);
}

fn pathsIn(db: *zkb.sqlite.Db, arena: std.mem.Allocator, cid: i64) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var st = try db.prepare("SELECT rel_path FROM docs WHERE collection_id = ?1 ORDER BY rel_path");
    defer st.finalize();
    try st.bindI64(1, cid);
    while (try st.step()) try out.append(arena, try arena.dupe(u8, st.columnText(0)));
    return out.items;
}

fn withIo(comptime f: fn (std.Io) anyerror!void) !void {
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    try f(threaded.io());
}

test "two byte-identical files are two documents, not a rename" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();

            // The shape found on a real corpus: the same memory written in two
            // projects. Treating the second as a move of the first left one real
            // file with no row at all — 315 files on disk, 314 in the index.
            const same = "---\nnode_type: wiki\n---\n\n# 同一份内容\n";
            try tree.write("a/memory/note.md", same);
            try tree.write("b/memory/note.md", same);

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 2), r.seen);
            try testing.expectEqual(@as(usize, 2), r.queued);
            try testing.expectEqual(@as(usize, 0), r.renamed);

            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const paths = try pathsIn(&db, arena_state.allocator(), cid);
            try testing.expectEqual(@as(usize, 2), paths.len);
            try testing.expectEqualStrings("a/memory/note.md", paths[0]);
            try testing.expectEqualStrings("b/memory/note.md", paths[1]);
        }
    }.run);
}

test "a real rename keeps the document, and its vectors, at the new path" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();

            const body = "---\nnode_type: wiki\n---\n\n# 会被改名\n";
            try tree.write("old.md", body);

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            _ = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            const first = (try s.findDoc(cid, "old.md")).?;
            // Only an indexed document can be renamed rather than re-queued; an
            // unindexed one has nothing worth preserving.
            try s.markIndexed(first.id, 1, 1000);

            try tree.remove("old.md");
            try tree.write("new.md", body);

            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 2000);
            try testing.expectEqual(@as(usize, 1), r.renamed);
            try testing.expectEqual(@as(usize, 0), r.queued);

            // Same row, new path: the point of detecting a rename is not
            // re-embedding bytes that did not change.
            const moved = (try s.findDoc(cid, "new.md")).?;
            try testing.expectEqual(first.id, moved.id);
            try testing.expect((try s.findDoc(cid, "old.md")) == null);
        }
    }.run);
}

test "a vanished file loses its row" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();
            try tree.write("keep.md", "# 甲\n");
            try tree.write("gone.md", "# 乙\n");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);
            _ = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);

            try tree.remove("gone.md");
            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 2000);
            try testing.expectEqual(@as(usize, 1), r.deleted);
            try testing.expect((try s.findDoc(cid, "gone.md")) == null);
            try testing.expect((try s.findDoc(cid, "keep.md")) != null);
        }
    }.run);
}

test "default patterns keep the usual junk out, and a corpus can override them" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();

            try tree.write("notes.md", "# 甲\n");
            // 806 markdown files live under node_modules in one real repo, so the
            // default has to hold even where no .gitignore exists.
            try tree.write("node_modules/pkg/README.md", "# 依赖的说明\n");
            try tree.write("dist/out.md", "# 构建产物\n");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            var r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 1), r.seen);
            try testing.expect(r.ignored >= 2);

            // The whole reason these moved out of a hardcoded list: a corpus that
            // means it can say so. `exclude_dirs` was checked outside the ignore
            // matcher, so nothing could override it.
            try tree.write(".zkbignore", "!node_modules/\n");
            r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 2000);
            try testing.expectEqual(@as(usize, 2), r.seen);
        }
    }.run);
}

test "a version-control directory is never walked, whatever the rules say" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();

            try tree.write("notes.md", "# 甲\n");
            try tree.write(".git/COMMIT_EDITMSG.md", "# 不该被索引\n");
            // `.git` appears in no ignore file anywhere — git excludes it
            // structurally — so a negation must not bring it back.
            try tree.write(".zkbignore", "!.git/\n");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 1), r.seen);
            try testing.expect((try s.findDoc(cid, ".git/COMMIT_EDITMSG.md")) == null);
        }
    }.run);
}

test "forget stays irreversible: archive cannot be re-included" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();

            try tree.write("live.md", "# 活着的记忆\n");
            try tree.write("archive/forgotten.md", "# 已经忘掉的\n");
            // A `.zkbignore` able to switch this back on would let a forgotten
            // memory return through a keyword or vector hit, which is the one
            // thing forgetting has to prevent.
            try tree.write(".zkbignore", "!archive/\n");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("m", tree.root, 1000);

            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, zkb.memory.scan_filters, 1000);
            try testing.expectEqual(@as(usize, 1), r.seen);
            try testing.expect((try s.findDoc(cid, "archive/forgotten.md")) == null);
        }
    }.run);
}

// ---------------------------------------------------------------------------
// symbolic links
//
// A link is indexed as the file it points at, so before this it could carry
// content from anywhere on the disk into the index — and from there into what
// `search` and `recall` hand an agent. Registering a cloned repository as a
// collection makes that somebody else's choice.
// ---------------------------------------------------------------------------

test "a link inside the root is followed" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();

            try tree.write("real.md", "# 真文件\n\n内容在这里。\n");
            try tree.symlink("real.md", "alias.md");
            try tree.symlink("../real.md", "sub/deep.md");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            // The root is under /tmp, which on macOS is itself a link to
            // /private/tmp. If the root were not resolved the same way as the
            // target, every one of these would look like an escape.
            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 0), r.escaped);
            try testing.expectEqual(@as(usize, 3), r.seen);
        }
    }.run);
}

test "a link out of the root is skipped and counted" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var outside = try Tree.init(io);
            defer outside.deinit();
            try outside.write("secret.md", "# 不该被索引的东西\n");

            var tree = try Tree.init(io);
            defer tree.deinit();
            try tree.write("real.md", "# 真文件\n");

            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const target = try outside.join(arena_state.allocator(), "secret.md");
            try tree.symlink(target, "leak.md");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 1), r.escaped);
            try testing.expectEqual(@as(usize, 1), r.seen);

            const paths = try pathsIn(&db, arena_state.allocator(), cid);
            try testing.expectEqual(@as(usize, 1), paths.len);
            try testing.expectEqualStrings("real.md", paths[0]);
        }
    }.run);
}

test "a relative ../ escape is skipped" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();
            // Climbs out of /tmp/zkb-scan-test-*/ entirely.
            try tree.symlink("../../etc/hosts", "hosts.md");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 1), r.escaped);
            try testing.expectEqual(@as(usize, 0), r.seen);
        }
    }.run);
}

test "a two-link chain out of the root is skipped" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var outside = try Tree.init(io);
            defer outside.deinit();
            try outside.write("secret.md", "# 不该被索引的东西\n");

            var tree = try Tree.init(io);
            defer tree.deinit();

            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const target = try outside.join(arena_state.allocator(), "secret.md");

            // The case a lexical resolver gets wrong: `a.md` points at `b.md`,
            // which is inside the root, so `std.fs.path.resolve` says the link
            // stays home. Only following the chain per component sees that it
            // does not. Two `ln -s` in a repository is no harder than one.
            try tree.symlink(target, "b.md");
            try tree.symlink("b.md", "a.md");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 2), r.escaped);
            try testing.expectEqual(@as(usize, 0), r.seen);
        }
    }.run);
}

test "a link through a symlinked directory is skipped" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var outside = try Tree.init(io);
            defer outside.deinit();
            try outside.write("secret.md", "# 不该被索引的东西\n");

            var tree = try Tree.init(io);
            defer tree.deinit();

            // `sub` is a link to a directory elsewhere. The walker never
            // descends into it — a linked directory is not `.directory` — but a
            // link *through* it would still resolve, if resolution stopped at
            // the last component.
            try tree.symlink(outside.root, "sub");
            try tree.symlink("sub/secret.md", "via.md");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 1), r.escaped);
            try testing.expectEqual(@as(usize, 0), r.seen);
        }
    }.run);
}

test "a dangling link is skipped, not counted as readable" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var tree = try Tree.init(io);
            defer tree.deinit();
            try tree.symlink("nowhere.md", "broken.md");

            var db = try openMem();
            defer db.close();
            var s = store.Store.init(&db);
            const cid = try s.ensureCollection("t", tree.root, 1000);

            // The target does not exist, so it resolves to where it *would* be —
            // inside the root — and is admitted here, then fails to open. The
            // point is that it neither crashes nor lands in the index.
            const r = try scan.reconcile(gpa, io, &s, cid, tree.root, .{}, 1000);
            try testing.expectEqual(@as(usize, 0), r.queued);
        }
    }.run);
}
