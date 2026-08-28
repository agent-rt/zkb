//! Descriptions attached to a subtree, and the prefix rule that decides which
//! ones a result gets.

const std = @import("std");
const testing = std.testing;
const zkb = @import("zkb");
const contexts = zkb.contexts;

const gpa = testing.allocator;

// ---------------------------------------------------------------------------
// prefix matching
// ---------------------------------------------------------------------------

test "an empty prefix covers the whole collection" {
    try testing.expect(contexts.prefixMatches("", "anything/at/all.md"));
    try testing.expect(contexts.prefixMatches("/", "anything/at/all.md"));
}

test "a prefix covers itself and what is under it" {
    try testing.expect(contexts.prefixMatches("research", "research/qmd.md"));
    try testing.expect(contexts.prefixMatches("research", "research"));
    try testing.expect(contexts.prefixMatches("agents/handoffs", "agents/handoffs/2026-08.md"));
}

test "a prefix does not cover a directory that merely starts with it" {
    // The failure this rule exists for: `research` attaching its sentence to
    // `research-notes/`, where it is not true and nothing looks wrong.
    try testing.expect(!contexts.prefixMatches("research", "research-notes/qmd.md"));
    try testing.expect(!contexts.prefixMatches("agents", "agentsx/y.md"));
}

test "a deeper prefix does not cover a shallower path" {
    try testing.expect(!contexts.prefixMatches("agents/handoffs", "agents/index.md"));
}

// ---------------------------------------------------------------------------
// lookup
// ---------------------------------------------------------------------------

fn mapOf(entries: []const contexts.Entry) contexts.Map {
    return .{ .entries = @constCast(entries) };
}

test "every matching prefix applies, general before specific" {
    var a = std.heap.ArenaAllocator.init(gpa);
    defer a.deinit();
    // Deliberately out of order in the file: the sort is by prefix length, not
    // by the order somebody happened to add them.
    const m = mapOf(&.{
        .{ .collection = "docs", .prefix = "agents/handoffs", .text = "handoffs" },
        .{ .collection = "docs", .prefix = "", .text = "the whole thing" },
        .{ .collection = "docs", .prefix = "agents", .text = "agent material" },
    });
    const parts = try m.forPath(a.allocator(), "docs", "agents/handoffs/2026-08.md");
    try testing.expectEqual(@as(usize, 3), parts.len);
    try testing.expectEqualStrings("the whole thing", parts[0]);
    try testing.expectEqualStrings("agent material", parts[1]);
    try testing.expectEqualStrings("handoffs", parts[2]);
}

test "a description belongs to its collection only" {
    var a = std.heap.ArenaAllocator.init(gpa);
    defer a.deinit();
    const m = mapOf(&.{
        .{ .collection = "docs", .prefix = "research", .text = "docs research" },
        .{ .collection = "notes", .prefix = "research", .text = "notes research" },
    });
    const parts = try m.forPath(a.allocator(), "notes", "research/x.md");
    try testing.expectEqual(@as(usize, 1), parts.len);
    try testing.expectEqualStrings("notes research", parts[0]);
}

test "nothing matching joins to null rather than an empty line" {
    var a = std.heap.ArenaAllocator.init(gpa);
    defer a.deinit();
    const m = mapOf(&.{.{ .collection = "docs", .prefix = "research", .text = "x" }});
    try testing.expect((try m.joinedForPath(a.allocator(), "docs", "other/y.md")) == null);
    try testing.expect((try contexts.Map.empty.joinedForPath(a.allocator(), "docs", "a.md")) == null);
}

test "joined output reads general to specific" {
    var a = std.heap.ArenaAllocator.init(gpa);
    defer a.deinit();
    const m = mapOf(&.{
        .{ .collection = "docs", .prefix = "research", .text = "调研" },
        .{ .collection = "docs", .prefix = "", .text = "知识库" },
    });
    const joined = (try m.joinedForPath(a.allocator(), "docs", "research/qmd.md")).?;
    try testing.expectEqualStrings("知识库 · 调研", joined);
}

// ---------------------------------------------------------------------------
// references
// ---------------------------------------------------------------------------

test "a reference splits into collection and prefix, with or without the scheme" {
    const a = contexts.splitRef("zkb://docs/agents/handoffs");
    try testing.expectEqualStrings("docs", a.collection);
    try testing.expectEqualStrings("agents/handoffs", a.prefix);

    const b = contexts.splitRef("docs/research");
    try testing.expectEqualStrings("docs", b.collection);
    try testing.expectEqualStrings("research", b.prefix);

    // A collection on its own describes all of it.
    const c = contexts.splitRef("zkb://docs");
    try testing.expectEqualStrings("docs", c.collection);
    try testing.expectEqualStrings("", c.prefix);

    // Trailing slashes are the same reference; a caller assembling the string
    // from parts produces both.
    const d = contexts.splitRef("zkb://docs/research/");
    try testing.expectEqualStrings("research", d.prefix);
}

// ---------------------------------------------------------------------------
// the file
// ---------------------------------------------------------------------------

const Home = struct {
    io: std.Io,
    root: []u8,
    layout: zkb.paths.Layout,
    env: std.process.Environ.Map,

    fn init(io: std.Io) !Home {
        var seed: usize = @intFromPtr(&io);
        seed ^= @as(usize, @bitCast(@as(isize, @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds))));
        const root = try std.fmt.allocPrint(gpa, "/tmp/zkb-contexts-test-{x}", .{seed});
        var env: std.process.Environ.Map = .init(gpa);
        try env.put("ZKB_HOME", root);
        var layout = try zkb.paths.resolve(gpa, &env);
        try layout.ensureDirs(io);
        return .{ .io = io, .root = root, .layout = layout, .env = env };
    }

    fn deinit(self: *Home) void {
        self.layout.deinit(gpa);
        self.env.deinit();
        std.Io.Dir.cwd().deleteTree(self.io, self.root) catch {};
        gpa.free(self.root);
    }
};

fn withIo(comptime f: fn (std.Io) anyerror!void) !void {
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    try f(threaded.io());
}

test "no file reads as no contexts" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var a = std.heap.ArenaAllocator.init(gpa);
            defer a.deinit();
            const m = try contexts.load(a.allocator(), io, &home.layout);
            try testing.expectEqual(@as(usize, 0), m.entries.len);
        }
    }.run);
}

test "a description round-trips, and a prefix is stored without its slashes" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var a = std.heap.ArenaAllocator.init(gpa);
            defer a.deinit();

            try contexts.record(gpa, io, &home.layout, .{
                .collection = "docs",
                .prefix = "/research/",
                .text = "技术调研，多为一次性实测",
            });
            const m = try contexts.load(a.allocator(), io, &home.layout);
            try testing.expectEqual(@as(usize, 1), m.entries.len);
            try testing.expectEqualStrings("research", m.entries[0].prefix);
            try testing.expectEqualStrings("技术调研，多为一次性实测", m.entries[0].text);
        }
    }.run);
}

test "recording the same reference twice replaces it" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var a = std.heap.ArenaAllocator.init(gpa);
            defer a.deinit();

            try contexts.record(gpa, io, &home.layout, .{ .collection = "docs", .prefix = "r", .text = "first" });
            try contexts.record(gpa, io, &home.layout, .{ .collection = "docs", .prefix = "r", .text = "second" });
            const m = try contexts.load(a.allocator(), io, &home.layout);
            try testing.expectEqual(@as(usize, 1), m.entries.len);
            try testing.expectEqualStrings("second", m.entries[0].text);
        }
    }.run);
}

test "the same prefix in two collections is two descriptions" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var a = std.heap.ArenaAllocator.init(gpa);
            defer a.deinit();

            try contexts.record(gpa, io, &home.layout, .{ .collection = "docs", .prefix = "r", .text = "docs" });
            try contexts.record(gpa, io, &home.layout, .{ .collection = "notes", .prefix = "r", .text = "notes" });
            const m = try contexts.load(a.allocator(), io, &home.layout);
            try testing.expectEqual(@as(usize, 2), m.entries.len);
        }
    }.run);
}

test "forget removes one and reports whether it was there" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var a = std.heap.ArenaAllocator.init(gpa);
            defer a.deinit();

            try contexts.record(gpa, io, &home.layout, .{ .collection = "docs", .prefix = "a", .text = "A" });
            try contexts.record(gpa, io, &home.layout, .{ .collection = "docs", .prefix = "b", .text = "B" });

            try testing.expect(try contexts.forget(gpa, io, &home.layout, "docs", "a"));
            try testing.expect(!try contexts.forget(gpa, io, &home.layout, "docs", "never"));

            const m = try contexts.load(a.allocator(), io, &home.layout);
            try testing.expectEqual(@as(usize, 1), m.entries.len);
            try testing.expectEqualStrings("b", m.entries[0].prefix);
        }
    }.run);
}

test "a row with no text is skipped rather than attached to every result" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var a = std.heap.ArenaAllocator.init(gpa);
            defer a.deinit();

            // Written by hand, the way a half-finished edit leaves it.
            const path = try contexts.registryPath(a.allocator(), &home.layout);
            var f = try std.Io.Dir.createFileAbsolute(io, path, .{ .truncate = true });
            var buf: [256]u8 = undefined;
            var wr = f.writer(io, &buf);
            try wr.interface.writeAll("collection,prefix,text\ndocs,research,\ndocs,agents,ok\n");
            try wr.interface.flush();
            f.close(io);

            const m = try contexts.load(a.allocator(), io, &home.layout);
            try testing.expectEqual(@as(usize, 1), m.entries.len);
            try testing.expectEqualStrings("agents", m.entries[0].prefix);
        }
    }.run);
}

test "a comma and a newline inside a description survive the file" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var a = std.heap.ArenaAllocator.init(gpa);
            defer a.deinit();

            const text = "调研笔记，含实测；\n第二行也算";
            try contexts.record(gpa, io, &home.layout, .{ .collection = "docs", .prefix = "r", .text = text });
            const m = try contexts.load(a.allocator(), io, &home.layout);
            try testing.expectEqualStrings(text, m.entries[0].text);
        }
    }.run);
}
