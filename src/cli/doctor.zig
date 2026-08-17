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

fn printSummary(w: *Writer, r: Result) !void {
    try w.print("\n{d}/{d} checks passed", .{ r.checks - r.failures, r.checks });
    if (r.failures != 0) try w.print(" — {d} FAILED", .{r.failures});
    try w.writeAll("\n");
}
