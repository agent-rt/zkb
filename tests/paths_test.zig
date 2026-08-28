//! The on-disk layout is a contract, not an implementation detail: `zkb doctor`
//! tells people which directories are safe to delete, and a rename that quietly
//! moves data under a "disposable" path would turn that advice destructive.

const std = @import("std");
const zkb = @import("zkb");

const testing = std.testing;
const gpa = testing.allocator;

fn envWith(pairs: []const [2][]const u8) !std.process.Environ.Map {
    var map: std.process.Environ.Map = .init(gpa);
    for (pairs) |kv| try map.put(kv[0], kv[1]);
    return map;
}

test "everything lives under one root" {
    var env = try envWith(&.{.{ "ZKB_HOME", "/tmp/zkbtest" }});
    defer env.deinit();
    var l = try zkb.paths.resolve(gpa, &env);
    defer l.deinit(gpa);

    inline for (.{ "data", "memory", "facts", "index_dir", "db", "models", "run_dir", "sock", "pid", "log", "trace" }) |f| {
        try testing.expect(std.mem.startsWith(u8, @field(l, f), "/tmp/zkbtest/"));
    }
}

test "the irreplaceable data never sits under a disposable directory" {
    // The property `zkb doctor` promises: `rm -rf <index|models|run>` is always
    // safe. If data ever moved beneath one of them that advice would delete a
    // memory, and nothing would catch it.
    var env = try envWith(&.{.{ "ZKB_HOME", "/tmp/zkbtest" }});
    defer env.deinit();
    var l = try zkb.paths.resolve(gpa, &env);
    defer l.deinit(gpa);

    inline for (.{ "memory", "facts" }) |f| {
        const p = @field(l, f);
        try testing.expect(std.mem.startsWith(u8, p, l.data));
        try testing.expect(!std.mem.startsWith(u8, p, l.index_dir));
        try testing.expect(!std.mem.startsWith(u8, p, l.models));
        try testing.expect(!std.mem.startsWith(u8, p, l.run_dir));
    }
    // And the converse: nothing disposable hides inside data.
    inline for (.{ "db", "sock", "pid", "log", "trace" }) |f| {
        try testing.expect(!std.mem.startsWith(u8, @field(l, f), l.data));
    }
}

test "$ZKB_DATA relocates the data without dragging the model along" {
    // The point of the override: put the megabyte that matters in a synced
    // folder or its own repo, and leave 600 MB of gguf where it is.
    var env = try envWith(&.{
        .{ "ZKB_HOME", "/tmp/zkbtest" },
        .{ "ZKB_DATA", "/elsewhere/kb" },
    });
    defer env.deinit();
    var l = try zkb.paths.resolve(gpa, &env);
    defer l.deinit(gpa);

    try testing.expectEqualStrings("/elsewhere/kb", l.data);
    try testing.expectEqualStrings("/elsewhere/kb/memory", l.memory);
    try testing.expectEqualStrings("/elsewhere/kb/facts.csv", l.facts);
    try testing.expectEqualStrings("/tmp/zkbtest/models", l.models);
    try testing.expectEqualStrings("/tmp/zkbtest/index/zkb.db", l.db);
}

test "the socket path stays well inside the sun_path limit" {
    // 104 bytes is a kernel constant; the nesting added by run/ costs four of
    // them, and a long $HOME plus a deep layout is how that ceiling gets hit.
    var env = try envWith(&.{.{ "HOME", "/Users/somebody-with-a-fairly-long-name" }});
    defer env.deinit();
    var l = try zkb.paths.resolve(gpa, &env);
    defer l.deinit(gpa);
    try testing.expect(l.sock.len < 104);
}

test "parseUri takes the first segment as the collection" {
    // `maintain` reads a link this way, so this must too — a second rule would
    // make the resolver and the broken-link check disagree about one link.
    const f = zkb.paths.parseUri;

    const a = f("zkb://docs/projects/x/REQ.md");
    try testing.expectEqualStrings("docs", a.collection.?);
    try testing.expectEqualStrings("projects/x/REQ.md", a.rel);

    // Extra slashes after the scheme name the same thing; a caller assembling
    // the string from parts produces both.
    const b = f("zkb:///docs/projects/x/REQ.md");
    try testing.expectEqualStrings("docs", b.collection.?);
    try testing.expectEqualStrings("projects/x/REQ.md", b.rel);

    // Any non-network scheme, so a corpus written against another tool's scheme
    // keeps working — and reads its collection correctly.
    const c = f("qmd://notes/x.md");
    try testing.expectEqualStrings("notes", c.collection.?);
    try testing.expectEqualStrings("x.md", c.rel);
}

test "a scheme with no path names a collection, not a document" {
    const u = zkb.paths.parseUri("zkb://docs");
    try testing.expectEqualStrings("docs", u.collection.?);
    try testing.expectEqualStrings("", u.rel);
}

test "a reference with no scheme names no collection" {
    const f = zkb.paths.parseUri;
    // A relative or wiki link means "in whichever collection I am already in",
    // which is the only reading available to a reference that never said one.
    const a = f("projects/x/REQ.md");
    try testing.expect(a.collection == null);
    try testing.expectEqualStrings("projects/x/REQ.md", a.rel);

    const b = f("/projects/x/REQ.md");
    try testing.expect(b.collection == null);
    try testing.expectEqualStrings("projects/x/REQ.md", b.rel);
}

test "nothing after the scheme is an empty request" {
    const u = zkb.paths.parseUri("zkb://");
    try testing.expect(u.collection == null);
    try testing.expectEqualStrings("", u.rel);
}

test "the old reading is gone, deliberately" {
    // `zkb://projects/x/REQ.md` used to mean the rel_path `projects/x/REQ.md`
    // in whichever collection had it. It now means the collection `projects`.
    // Pinned as a test because the corpus was migrated on this promise: a
    // fallback that tried both readings would flip meaning the day somebody
    // registers a collection called `projects`.
    const u = zkb.paths.parseUri("zkb://projects/x/REQ.md");
    try testing.expectEqualStrings("projects", u.collection.?);
    try testing.expectEqualStrings("x/REQ.md", u.rel);
}
