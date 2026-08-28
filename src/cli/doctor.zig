//! `zkb doctor` — prove the geology holds on *this* machine.
//!
//! Every line is a measured fact, not a build-time assumption: FTS5 presence
//! comes from pragma_compile_options, the vec0 KNN path is exercised for real,
//! the embedding dimension is read off a loaded model. The whole point is that
//! "it compiled" and "it works here" are different claims (SPEC §7).

const std = @import("std");
// Via the module, not ../root.zig: importing the file directly would compile a
// second instance of the module (and its @cImport) into this binary.
const zkb = @import("zkb");
const sqlite = zkb.sqlite;

const Writer = std.Io.Writer;

pub const Result = struct { checks: usize = 0, failures: usize = 0 };

fn ok(w: *Writer, r: *Result, comptime fmt: []const u8, args: anytype) !void {
    r.checks += 1;
    try w.print("  ok    ", .{});
    try w.print(fmt, args);
    try w.writeAll("\n");
}

fn fail(w: *Writer, r: *Result, comptime fmt: []const u8, args: anytype) !void {
    r.checks += 1;
    r.failures += 1;
    try w.print("  FAIL  ", .{});
    try w.print(fmt, args);
    try w.writeAll("\n");
}

fn info(w: *Writer, comptime fmt: []const u8, args: anytype) !void {
    try w.print("        ", .{});
    try w.print(fmt, args);
    try w.writeAll("\n");
}

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    model_path_override: ?[]const u8,
) !Result {
    var r: Result = .{};

    try w.print("zkb doctor — zkb {s}, zig {s}\n\n", .{ zkb.version, @import("builtin").zig_version_string });

    // ---------------------------------------------------------------- layout
    try w.print("layout\n", .{});
    var layout = zkb.paths.resolve(gpa, env) catch |err| {
        try fail(w, &r, "cannot resolve ~/.zkb: {t}", .{err});
        return r;
    };
    defer layout.deinit(gpa);
    try ok(w, &r, "root {s}", .{layout.root});

    // Which subdirectory is safe to delete is the single most useful thing this
    // command can say, and it must be said in a form somebody can copy. Half the
    // fixes for a broken index are "delete it and re-run", and that sentence has
    // to be unambiguous about what "it" is.
    try info(w, "data      {s}   ← the only irreplaceable directory", .{layout.data});
    try info(w, "disposable {s}", .{layout.index_dir});
    try info(w, "           {s}", .{layout.models});
    try info(w, "           {s}", .{layout.run_dir});

    // Memories are the only thing here that cannot be rebuilt from something
    // else: a session is gone once it ends. Everything else under ~/.zkb is
    // derived, so this is the one directory whose loss is permanent.
    //
    // Reported, not failed. Nothing in zkb calls `jj` or `git` — the two time
    // axes both live in the file now (facts.csv gained `recorded_at`), so
    // version control adds review and undo rather than being load-bearing.
    // A check that fails on a working setup teaches people to ignore checks.
    if (std.Io.Dir.accessAbsolute(io, layout.data, .{})) |_| {
        var versioned = false;
        var buf: [512]u8 = undefined;
        inline for (.{ ".jj", ".git" }) |marker| {
            const p = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ layout.data, marker });
            if (std.Io.Dir.accessAbsolute(io, p, .{})) |_| versioned = true else |_| {}
        }
        if (versioned) {
            try ok(w, &r, "data is under version control", .{});
        } else {
            try info(w, "data is not under version control (optional)", .{});
        }
    } else |_| {
        try info(w, "no data yet (created by: zkb remember)", .{});
    }

    // These belong beside the layout that just described `data/`, and they must
    // come before anything that can `return r` early. Placed after the model
    // section at first, they never ran on a machine without a model — a check
    // that silently does not run is the failure this whole file exists to
    // prevent, reproduced inside it.
    try checkAuthoredFiles(gpa, io, w, &r, &layout);

    // ---------------------------------------------------------------- sqlite
    try w.print("\nsqlite\n", .{});
    var db = sqlite.Db.open(":memory:", .read_write) catch |err| {
        try fail(w, &r, "cannot open in-memory database: {t}", .{err});
        return r;
    };
    defer db.close();

    try ok(w, &r, "sqlite {s}, sqlite-vec {s}", .{ sqlite.libVersion(), sqlite.vecVersion() });

    if (sqlite.hasCompileOption(&db, "ENABLE_FTS5"))
        try ok(w, &r, "FTS5 compiled in", .{})
    else
        try fail(w, &r, "FTS5 missing — keyword retrieval would be unavailable", .{});

    if (sqlite.hasCompileOption(&db, "THREADSAFE=2"))
        try ok(w, &r, "THREADSAFE=2 (one connection per thread)", .{})
    else
        try fail(w, &r, "THREADSAFE=2 missing — daemon threading model is unsound", .{});

    // trigram tokenizer: creating the table is the only honest test.
    if (db.exec(
        \\CREATE VIRTUAL TABLE _probe_fts USING fts5(
        \\  text, content='', contentless_delete=1, tokenize='trigram case_sensitive 0'
        \\);
    )) |_| {
        try ok(w, &r, "fts5 trigram tokenizer + contentless_delete", .{});
    } else |_| {
        try fail(w, &r, "fts5 trigram/contentless_delete unavailable: {s}", .{db.lastError()});
    }

    // vec0: full round trip, not just table creation.
    vec0: {
        db.exec(
            \\CREATE VIRTUAL TABLE _probe_vec USING vec0(
            \\  collection_id INTEGER partition key,
            \\  chunk_id INTEGER PRIMARY KEY,
            \\  embedding FLOAT[4] distance_metric=cosine
            \\);
        ) catch {
            try fail(w, &r, "vec0 table creation failed: {s}", .{db.lastError()});
            break :vec0;
        };

        var vec = [_]f32{ 1, 0, 0, 0 };
        var ins = db.prepare("INSERT INTO _probe_vec(collection_id, chunk_id, embedding) VALUES (1, 7, ?1)") catch {
            try fail(w, &r, "vec0 insert prepare failed: {s}", .{db.lastError()});
            break :vec0;
        };
        defer ins.finalize();
        try ins.bindVector(1, &vec);
        _ = ins.step() catch {
            try fail(w, &r, "vec0 insert failed: {s}", .{db.lastError()});
            break :vec0;
        };

        var st = db.prepare(
            \\SELECT chunk_id FROM _probe_vec
            \\WHERE collection_id = 1 AND embedding MATCH ?1 AND k = 1
        ) catch {
            try fail(w, &r, "vec0 KNN prepare failed: {s}", .{db.lastError()});
            break :vec0;
        };
        defer st.finalize();
        try st.bindVector(1, &vec);
        if ((st.step() catch false) and st.columnI64(0) == 7)
            try ok(w, &r, "vec0 KNN round trip (partition key + k)", .{})
        else
            try fail(w, &r, "vec0 KNN returned unexpected result", .{});
    }

    // -------------------------------------------------------------- database
    try w.print("\ndatabase\n", .{});
    if (std.Io.Dir.accessAbsolute(io, layout.db, .{})) |_| {
        const db_path_z = try gpa.dupeZ(u8, layout.db);
        defer gpa.free(db_path_z);
        var real = sqlite.Db.open(db_path_z, .read_only) catch |err| {
            try fail(w, &r, "cannot open {s}: {t}", .{ layout.db, err });
            return r;
        };
        defer real.close();
        var buf: [64]u8 = undefined;
        const verdict = (try real.queryText("PRAGMA integrity_check", &buf)) orelse "";
        if (std.mem.eql(u8, verdict, "ok"))
            try ok(w, &r, "integrity_check ok", .{})
        else
            try fail(w, &r, "integrity_check: {s}", .{verdict});
    } else |_| {
        try info(w, "no database yet at {s} (nothing indexed)", .{layout.db});
    }

    // ----------------------------------------------------------------- model
    try w.print("\nembedding model\n", .{});
    if (!zkb.build_options.llama) {
        try info(w, "built with -Dllama=false — embedding unavailable", .{});
        try printSummary(w, r);
        return r;
    }

    const found = zkb.model_registry.resolve(gpa, io, env, &layout, model_path_override, .q8_0) catch {
        try fail(w, &r, "model not found", .{});
        try info(w, "run: zkb model pull", .{});
        try printSummary(w, r);
        return r;
    };
    defer found.deinit(gpa);
    const model_path = found.path;

    const digest = zkb.hash.fileSha256(io, model_path) catch |err| {
        try fail(w, &r, "cannot hash model: {t}", .{err});
        try printSummary(w, r);
        return r;
    };
    // Where it came from matters: a Hugging Face cache hit means zkb never
    // downloaded a second copy, and saying so is how anyone finds that out.
    try ok(w, &r, "model {s} ({t})", .{ std.fs.path.basename(model_path), found.source });
    if (found.source == .hf_cache) try info(w, "{s}", .{model_path});
    try info(w, "sha256 {s}", .{digest[0..16]});

    var embedder = zkb.embed.Embedder.init(gpa, model_path, .{}) catch |err| {
        try fail(w, &r, "model load failed: {t}", .{err});
        try printSummary(w, r);
        return r;
    };
    defer embedder.deinit();

    if (embedder.n_embd == 1024)
        try ok(w, &r, "embedding dim 1024 (measured)", .{})
    else
        try fail(w, &r, "embedding dim {d}, schema expects 1024", .{embedder.n_embd});

    // A real forward pass: a loadable model that cannot embed is not usable.
    {
        const vec = try gpa.alloc(f32, embedder.n_embd);
        defer gpa.free(vec);
        if (embedder.embed("zkb doctor probe", vec)) |v| {
            var sumsq: f64 = 0;
            for (v) |x| sumsq += @as(f64, x) * @as(f64, x);
            if (@abs(sumsq - 1.0) < 1e-4)
                try ok(w, &r, "forward pass ok, vector is unit-norm", .{})
            else
                try fail(w, &r, "vector norm {d:.6}, expected 1.0", .{sumsq});
        } else |err| {
            try fail(w, &r, "forward pass failed: {t}", .{err});
        }
    }

    try printSummary(w, r);
    return r;
}

/// The two files in `data/` that a person may edit by hand.
///
/// `collections.csv` and `contexts.csv` are the record; the tables are their
/// projection, rebuilt on the next scan. The README says so and invites hand
/// edits — which means a typo in a `root` re-points a collection at the wrong
/// directory, and nothing anywhere says a word: the scan finds no files, the
/// collection quietly empties, and `search --collection x` answers nothing at
/// all, plausibly. That is the failure this section exists to catch, and the
/// only place in zkb that can catch it, because every other reader treats the
/// file as authoritative by construction.
///
/// Absent files are not a finding: a machine that has only ever used the
/// default collection has neither.
fn checkAuthoredFiles(
    gpa: std.mem.Allocator,
    io: std.Io,
    w: *Writer,
    r: *Result,
    layout: *const zkb.paths.Layout,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const reg_path = try zkb.roots.registryPath(arena, layout);
    const ctx_path = try zkb.contexts.registryPath(arena, layout);
    const have_reg = if (std.Io.Dir.accessAbsolute(io, reg_path, .{})) |_| true else |_| false;
    const have_ctx = if (std.Io.Dir.accessAbsolute(io, ctx_path, .{})) |_| true else |_| false;
    if (!have_reg and !have_ctx) return;

    try w.print("\nhand-edited files in data/\n", .{});

    var names: std.ArrayList([]const u8) = .empty;

    if (have_reg) {
        const rows = zkb.roots.loadRegistry(arena, io, layout) catch |err| {
            try fail(w, r, "collections.csv unreadable: {t}", .{err});
            return;
        };
        var bad = false;
        for (rows) |row| {
            // Checked in this order because each answer makes the next one
            // meaningful: a relative root is wrong whether or not it exists,
            // and "does not exist" is worth saying before "is not a directory".
            if (!std.fs.path.isAbsolute(row.root)) {
                try fail(w, r, "collection \"{s}\": root is not absolute: {s}", .{ row.collection, row.root });
                bad = true;
                continue;
            }
            std.Io.Dir.accessAbsolute(io, row.root, .{}) catch {
                try fail(w, r, "collection \"{s}\": root does not exist: {s}", .{ row.collection, row.root });
                try info(w, "the collection will index nothing, and say nothing about why", .{});
                bad = true;
                continue;
            };
            // zkb's own two collections take their roots from the layout. A row
            // naming one of them is a row that loses every scan, silently.
            if (std.mem.eql(u8, row.collection, "memory") or std.mem.eql(u8, row.collection, "numbers")) {
                try fail(w, r, "collection \"{s}\" is one of zkb's own; its root comes from the layout", .{row.collection});
                bad = true;
                continue;
            }
            for (names.items) |seen| {
                if (std.mem.eql(u8, seen, row.collection)) {
                    try fail(w, r, "collection \"{s}\" is registered twice; the last row wins", .{row.collection});
                    bad = true;
                }
            }
            try names.append(arena, row.collection);
        }
        if (!bad) try ok(w, r, "collections.csv: {d} registration(s), every root present", .{rows.len});
    }

    if (have_ctx) {
        const map = zkb.contexts.load(arena, io, layout) catch |err| {
            try fail(w, r, "contexts.csv unreadable: {t}", .{err});
            return;
        };
        var orphans: usize = 0;
        for (map.entries) |e| {
            var known = std.mem.eql(u8, e.collection, "memory") or std.mem.eql(u8, e.collection, "numbers");
            for (names.items) |n| {
                if (std.mem.eql(u8, n, e.collection)) known = true;
            }
            // Only when there is a registry to check against: with none, every
            // collection was registered some other way and "unknown" would be
            // noise on a working setup.
            if (have_reg and !known) {
                try fail(w, r, "context for \"{s}\" names no registered collection; it will never appear", .{e.collection});
                orphans += 1;
            }
        }
        if (orphans == 0) try ok(w, r, "contexts.csv: {d} description(s)", .{map.entries.len});
    }
}

fn printSummary(w: *Writer, r: Result) !void {
    try w.print("\n{d}/{d} checks passed", .{ r.checks - r.failures, r.checks });
    if (r.failures != 0) try w.print(" — {d} FAILED", .{r.failures});
    try w.writeAll("\n");
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// A `data/` holding whichever of the two hand-edited files a test needs.
const TestHome = struct {
    io: std.Io,
    root: []u8,
    layout: zkb.paths.Layout,
    env: std.process.Environ.Map,

    fn init(io: std.Io) !TestHome {
        var seed: usize = @intFromPtr(&io);
        seed ^= @as(usize, @bitCast(@as(isize, @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds))));
        const root = try std.fmt.allocPrint(testing.allocator, "/tmp/zkb-doctor-test-{x}", .{seed});
        var env: std.process.Environ.Map = .init(testing.allocator);
        try env.put("ZKB_HOME", root);
        var layout = try zkb.paths.resolve(testing.allocator, &env);
        try layout.ensureDirs(io);
        return .{ .io = io, .root = root, .layout = layout, .env = env };
    }

    fn deinit(self: *TestHome) void {
        self.layout.deinit(testing.allocator);
        self.env.deinit();
        std.Io.Dir.cwd().deleteTree(self.io, self.root) catch {};
        testing.allocator.free(self.root);
    }

    fn writeData(self: *TestHome, name: []const u8, body: []const u8) !void {
        var d = try std.Io.Dir.openDirAbsolute(self.io, self.layout.data, .{});
        defer d.close(self.io);
        var f = try d.createFile(self.io, name, .{});
        defer f.close(self.io);
        var buf: [1024]u8 = undefined;
        var wr = f.writer(self.io, &buf);
        try wr.interface.writeAll(body);
        try wr.interface.flush();
    }

    /// Run only the section under test and report what it found.
    fn check(self: *TestHome, out: []u8) !Result {
        var w = std.Io.Writer.fixed(out);
        var r: Result = .{};
        try checkAuthoredFiles(testing.allocator, self.io, &w, &r, &self.layout);
        return r;
    }
};

fn withIo(comptime f: fn (std.Io) anyerror!void) !void {
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    try f(threaded.io());
}

test "neither file present says nothing at all" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try TestHome.init(io);
            defer home.deinit();
            var out: [4096]u8 = undefined;
            const r = try home.check(&out);
            // Not "ok": a machine that has only ever used the default collection
            // has neither file, and a section that reports on nothing trains the
            // reader to skim past this part of the output.
            try testing.expectEqual(@as(usize, 0), r.checks);
        }
    }.run);
}

test "a root that does not exist is a failure, not a shrug" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try TestHome.init(io);
            defer home.deinit();
            try home.writeData("collections.csv",
                \\name,root,extensions,include
                \\notes,/tmp/zkb-doctor-no-such-root,.md,
                \\
            );
            var out: [4096]u8 = undefined;
            const r = try home.check(&out);
            try testing.expectEqual(@as(usize, 1), r.failures);
        }
    }.run);
}

test "a relative root is a failure even before it is looked for" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try TestHome.init(io);
            defer home.deinit();
            // Relative is wrong whether or not it resolves: the root is read
            // later by a daemon whose working directory is unrelated.
            try home.writeData("collections.csv",
                \\name,root,extensions,include
                \\notes,docs/notes,.md,
                \\
            );
            var out: [4096]u8 = undefined;
            const r = try home.check(&out);
            try testing.expectEqual(@as(usize, 1), r.failures);
        }
    }.run);
}

test "a row naming one of zkb's own collections is a failure" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try TestHome.init(io);
            defer home.deinit();
            const body = try std.fmt.allocPrint(testing.allocator,
                "name,root,extensions,include\nmemory,{s},.md,\n", .{home.layout.data});
            defer testing.allocator.free(body);
            try home.writeData("collections.csv", body);
            var out: [4096]u8 = undefined;
            const r = try home.check(&out);
            try testing.expectEqual(@as(usize, 1), r.failures);
        }
    }.run);
}

test "the same collection registered twice is a failure" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try TestHome.init(io);
            defer home.deinit();
            const body = try std.fmt.allocPrint(testing.allocator,
                "name,root,extensions,include\nn,{s},.md,\nn,{s},.md,\n",
                .{ home.layout.data, home.layout.data });
            defer testing.allocator.free(body);
            try home.writeData("collections.csv", body);
            var out: [4096]u8 = undefined;
            const r = try home.check(&out);
            try testing.expectEqual(@as(usize, 1), r.failures);
        }
    }.run);
}

test "a context naming no registered collection is a failure" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try TestHome.init(io);
            defer home.deinit();
            const body = try std.fmt.allocPrint(testing.allocator,
                "name,root,extensions,include\nnotes,{s},.md,\n", .{home.layout.data});
            defer testing.allocator.free(body);
            try home.writeData("collections.csv", body);
            try home.writeData("contexts.csv",
                \\collection,prefix,text
                \\notes,,fine
                \\ghost,,names nothing
                \\
            );
            var out: [4096]u8 = undefined;
            const r = try home.check(&out);
            try testing.expectEqual(@as(usize, 1), r.failures);
        }
    }.run);
}

test "a context is not judged when there is no registry to judge it against" {
    try withIo(struct {
        fn run(io: std.Io) !void {
            var home = try TestHome.init(io);
            defer home.deinit();
            // Collections registered before this file existed live only in the
            // index. Calling their descriptions orphans would be noise on a
            // working setup, which is how a check stops being read.
            try home.writeData("contexts.csv",
                \\collection,prefix,text
                \\registered-elsewhere,,fine
                \\
            );
            var out: [4096]u8 = undefined;
            const r = try home.check(&out);
            try testing.expectEqual(@as(usize, 0), r.failures);
        }
    }.run);
}
