//! The daemon over its real socket.
//!
//! Nothing in the suite reached this code. `daemon run` refused to start without
//! an embedding model, CI deliberately keeps none, and so the socket, the request
//! framing, the dispatch table and every handler could only ever be exercised by
//! hand. The cost was measured once: `zkb search --path` was wired into the wrong
//! handler and shipped, because the unit tests call the library directly and the
//! only path that would have caught it was the one nobody could run.
//!
//! Now that a missing model degrades instead of refusing, the absence of a model
//! is exactly what makes this testable: no GGUF is downloaded, no vector is
//! computed, and the whole request/response surface is still real.

const std = @import("std");
const testing = std.testing;
const zkb = @import("zkb");

const gpa = testing.allocator;

/// A ZKB_HOME under /tmp, not under `.zig-cache`.
///
/// Two independent reasons, and either alone is enough: this project's own
/// `.gitignore` excludes `.zig-cache/`, which a scan correctly honours; and a
/// unix socket path has a hard length limit that a nested cache path exceeds —
/// `daemon run` reports `NameTooLong` and nothing listens.
const Home = struct {
    io: std.Io,
    root: []u8,
    sock: []u8,
    corpus: []u8,

    fn init(io: std.Io) !Home {
        var seed: usize = @intFromPtr(&io);
        seed ^= @as(usize, @bitCast(@as(isize, @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds))));
        const root = try std.fmt.allocPrint(gpa, "/tmp/zkb-daemon-test-{x}", .{seed});
        var h = try std.Io.Dir.cwd().createDirPathOpen(io, root, .{});
        h.close(io);
        const corpus = try std.fmt.allocPrint(gpa, "{s}/corpus", .{root});
        var c = try std.Io.Dir.cwd().createDirPathOpen(io, corpus, .{});
        c.close(io);
        return .{
            .io = io,
            .root = root,
            .sock = try std.fmt.allocPrint(gpa, "{s}/run/zkb.sock", .{root}),
            .corpus = corpus,
        };
    }

    fn deinit(self: *Home) void {
        std.Io.Dir.cwd().deleteTree(self.io, self.root) catch {};
        gpa.free(self.corpus);
        gpa.free(self.sock);
        gpa.free(self.root);
    }
};

/// Only ZKB_HOME. No HOME and no HF_* means `registry.resolve` has nowhere to
/// look, so the degraded path is reached on a developer's machine — which does
/// have a model in its Hugging Face cache — and not only in CI.
fn envFor(home: *const Home) !std.process.Environ.Map {
    var env: std.process.Environ.Map = .init(gpa);
    try env.put("ZKB_HOME", home.root);
    return env;
}

const Server = struct {
    thread: std.Thread,

    fn start(io: std.Io, env: *const std.process.Environ.Map, home: *const Home) !Server {
        const T = struct {
            fn go(i: std.Io, e: *const std.process.Environ.Map, root: []const u8) void {
                zkb.daemon.run(gpa, i, e, .{
                    .root = root,
                    .collection = "notes",
                    // Long enough that no rescan lands inside the test.
                    .scan_interval_s = 3600,
                }) catch {};
            }
        };
        const t = try std.Thread.spawn(.{}, T.go, .{ io, env, home.corpus });
        return .{ .thread = t };
    }

    /// Bounded, because a test that hangs is worse than one that fails.
    fn waitReady(io: std.Io, sock: []const u8) !zkb.ipc_client.Client {
        var waited_ms: usize = 0;
        while (waited_ms < 10_000) : (waited_ms += 20) {
            if (zkb.ipc_client.Client.connect(io, sock)) |c| return c else |_| {}
            std.Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .awake) catch {};
        }
        return error.DaemonNeverListened;
    }

    /// Shut down, then knock.
    ///
    /// `shutdown` only sets a flag, and the accept loop is blocked inside
    /// `accept` until some connection arrives to wake it — so the flag alone is
    /// not enough to end the process. `zkb daemon stop` never notices because it
    /// polls liveness afterwards and each probe is the knock. A test that only
    /// sends the request hangs in `join` forever, which is how this was found.
    fn stop(self: Server, io: std.Io, sock: []const u8) void {
        if (zkb.ipc_client.Client.connect(io, sock)) |c| {
            var client = c;
            if (client.call(gpa, .shutdown, "{}")) |r| {
                var resp = r;
                resp.deinit(gpa);
            } else |_| {}
            client.close();
        } else |_| {}

        var knocks: usize = 0;
        while (knocks < 50) : (knocks += 1) {
            if (zkb.ipc_client.Client.connect(io, sock)) |c| {
                var probe = c;
                probe.close();
            } else |_| break;
            std.Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .awake) catch {};
        }
        self.thread.join();
    }
};

fn writeFile(io: std.Io, dir: []const u8, name: []const u8, body: []const u8) !void {
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, dir, .{});
    defer d.close(io);
    var f = try d.createFile(io, name, .{});
    defer f.close(io);
    var buf: [512]u8 = undefined;
    var wr = f.writer(io, &buf);
    try wr.interface.writeAll(body);
    try wr.interface.flush();
}

fn withIo(comptime f: fn (std.Io) anyerror!void) !void {
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    try f(threaded.io());
}

test "the daemon comes up without a model and says so" {
    try withIo(struct {
        fn f(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var env = try envFor(&home);
            defer env.deinit();

            const server = try Server.start(io, &env, &home);
            var client = try Server.waitReady(io, home.sock);
            defer {
                client.close();
                server.stop(io, home.sock);
            }

            var resp = try client.call(gpa, .health, "{}");
            defer resp.deinit(gpa);
            try testing.expect(resp.ok);

            // Serving, and honest about what it cannot do. Refusing to start was
            // the old behaviour and it withheld scanning, keyword search and
            // maintenance over the one capability that was missing.
            const obj = resp.result.?.object;
            const reason = obj.get("degraded").?;
            try testing.expect(reason == .string);
            try testing.expect(std.mem.indexOf(u8, reason.string, "model") != null);
        }
    }.f);
}

test "a registration crossing the real socket lands with every field intact" {
    // The daemon half of what `roots_test.zig` checks in process. That one drives
    // the serialiser and the parser directly; this one puts them at opposite ends
    // of the socket, through the request framing, the dispatch table and the
    // ingest thread's handover — the stretch that had no coverage at all, and
    // where a parameter last went missing while the command reported success.
    try withIo(struct {
        fn f(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var env = try envFor(&home);
            defer env.deinit();

            try writeFile(io, home.corpus, "a.md", "# A\n\nbody\n");

            const server = try Server.start(io, &env, &home);
            var client = try Server.waitReady(io, home.sock);
            defer {
                client.close();
                server.stop(io, home.sock);
            }

            const reg: zkb.roots.Registration = .{
                .collection = "notes",
                .root = home.corpus,
                .extensions = ".md",
                .include = "*.md",
            };
            var body: std.Io.Writer.Allocating = .init(gpa);
            defer body.deinit();
            try reg.writeJson(&body.writer);

            var resp = try client.call(gpa, .index, body.written());
            defer resp.deinit(gpa);
            try testing.expect(resp.ok);

            // Applied by the ingest thread, so the reply is "queued" and the row
            // appears a moment later. Polled rather than slept on: a fixed sleep
            // is either flaky or slow, and usually manages both.
            const db_path = try std.fmt.allocPrintSentinel(gpa, "{s}/index/zkb.db", .{home.root}, 0);
            defer gpa.free(db_path);

            const want = try std.fmt.allocPrint(gpa, "{s}|.md|*.md", .{home.corpus});
            defer gpa.free(want);

            // Waits for the value, not for the row. The row exists from the
            // moment the daemon starts — `ensureDocs` creates it with no filters
            // — so "a row is there" is true before the request is applied and
            // reading then compares against startup state. This test asserted
            // exactly that and reported the filters lost.
            var waited_ms: usize = 0;
            var last: ?[]u8 = null;
            defer if (last) |v| gpa.free(v);
            while (waited_ms < 10_000) : (waited_ms += 20) {
                if (try registeredRoot(db_path)) |row| {
                    if (last) |v| gpa.free(v);
                    last = row;
                    if (std.mem.eql(u8, row, want)) return;
                }
                std.Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .awake) catch {};
            }
            std.debug.print("last seen: {s}\nwanted:    {s}\n", .{ last orelse "(no row)", want });
            return error.RegistrationNeverApplied;
        }
    }.f);
}

/// `root|extensions|include` for the `notes` collection, or null while the ingest
/// thread has not written it yet.
fn registeredRoot(db_path: [:0]const u8) !?[]u8 {
    var db = zkb.store.open(db_path, .read_only) catch return null;
    defer db.close();
    var st = db.prepare(
        "SELECT root || '|' || coalesce(extensions,'-') || '|' || coalesce(include,'-') " ++
            "FROM collections WHERE name = 'notes'",
    ) catch return null;
    defer st.finalize();
    if (!(st.step() catch return null)) return null;
    return try gpa.dupe(u8, st.columnText(0));
}

/// Seed an index at ZKB_HOME before the daemon opens it.
///
/// Without a model the ingest thread cannot embed, so a corpus on disk stays
/// pending and never becomes a chunk — which is exactly the state that makes the
/// socket testable, and also the state in which a retrieval test would have
/// nothing to retrieve. Writing the chunks directly sidesteps the embedder while
/// leaving the request path, the handler and the pack entirely real.
fn seedIndex(io: std.Io, home: *const Home) !void {
    const index_dir = try std.fmt.allocPrint(gpa, "{s}/index", .{home.root});
    defer gpa.free(index_dir);
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, index_dir, .{});
    d.close(io);

    const db_path = try std.fmt.allocPrintSentinel(gpa, "{s}/zkb.db", .{index_dir}, 0);
    defer gpa.free(db_path);
    var db = try zkb.store.open(db_path, .read_write);
    defer db.close();
    var s = zkb.store.Store.init(&db);

    const cid = try s.ensureCollection("corpus", "/tmp/corpus", 1000);
    for ([_][]const u8{ "a/one.md", "b/two.md" }) |rel| {
        const body = "RRF fusion merges the two rankers.";
        const did = try s.upsertDocContent(cid, rel, rel, body.len, 1000);
        var v: [@intCast(zkb.schema.embedding_dim)]f32 = @splat(0);
        v[0] = 1;
        _ = try s.insertChunk(cid, did, .{
            .idx = 0,
            .heading_path = "",
            .byte_start = 0,
            .byte_end = @intCast(body.len),
            .n_tokens = 8,
            .text = body,
        }, &v);
        try s.markIndexed(did, 1, 1000);
    }
}

test "query's filters cross the socket, and are read on the other side" {
    // `query` grew `--collection` and `--path` long after `search` had them, and
    // a filter that exists only on the in-process path is the failure this
    // codebase keeps repeating: the command reports success, returns results, and
    // the filter silently did nothing. The two halves are wired once each now —
    // `proto.writeFilters` writes, `searchConfig`/`requestedCollection` read — so
    // this drives the request through the same serialiser the CLI uses, over the
    // real socket, into the real handler.
    try withIo(struct {
        fn f(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var env = try envFor(&home);
            defer env.deinit();
            try seedIndex(io, &home);

            const server = try Server.start(io, &env, &home);
            var client = try Server.waitReady(io, home.sock);
            defer {
                client.close();
                server.stop(io, home.sock);
            }

            // Unfiltered first, so the filtered result below is a narrowing and
            // not an empty index dressed up as one.
            {
                var resp = try client.call(gpa, .query, "{\"query\":\"fusion\",\"budget\":2000}");
                defer resp.deinit(gpa);
                try testing.expect(resp.ok);
                try testing.expectEqual(@as(usize, 2), resp.result.?.object.get("documents").?.array.items.len);
            }

            var pbuf: [512]u8 = undefined;
            var pw = std.Io.Writer.fixed(&pbuf);
            try pw.writeAll("{\"query\":\"fusion\",\"budget\":2000");
            try zkb.proto.writeFilters(&pw, "corpus", "a/**");
            try pw.writeAll("}");

            var resp = try client.call(gpa, .query, pw.buffered());
            defer resp.deinit(gpa);
            try testing.expect(resp.ok);

            const docs = resp.result.?.object.get("documents").?.array;
            try testing.expectEqual(@as(usize, 1), docs.items.len);
            try testing.expectEqualStrings("a/one.md", docs.items[0].object.get("path").?.string);
        }
    }.f);
}

test "query rejects a collection that does not exist, rather than ignoring it" {
    // The quiet failure mode is worse than the loud one: an unknown name that is
    // dropped returns the whole corpus, which reads as "no matches in this
    // project" only after someone notices the results come from elsewhere.
    try withIo(struct {
        fn f(io: std.Io) !void {
            var home = try Home.init(io);
            defer home.deinit();
            var env = try envFor(&home);
            defer env.deinit();
            try seedIndex(io, &home);

            const server = try Server.start(io, &env, &home);
            var client = try Server.waitReady(io, home.sock);
            defer {
                client.close();
                server.stop(io, home.sock);
            }

            var resp = try client.call(gpa, .query, "{\"query\":\"fusion\",\"collection\":\"nope\"}");
            defer resp.deinit(gpa);
            try testing.expect(!resp.ok);
            try testing.expectEqualStrings("bad_request", resp.code);
        }
    }.f);
}
