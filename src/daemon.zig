//! The daemon: one process, several threads, one job each.
//!
//!   accept loop     main thread; hands each connection to a worker
//!   conn thread xN  one read-only SQLite connection each (SQLITE_THREADSAFE=2
//!                   means per-thread connections, never shared)
//!   Embedder x1     owns the single llama_context, drains the priority queue
//!   IngestWorker x1 the only writer
//!
//! Two decisions are load-bearing rather than incidental:
//!
//! **The model loads lazily.** Loading 640MB of GGUF plus Metal init takes 1-2s;
//! doing it on the startup path would make `zkb daemon start` visibly stall. The
//! target is under 200ms to a serving socket, so the first embedding request pays
//! instead, and `health` reports `model_loaded` honestly (SPEC §6.3).
//!
//! **A model mismatch degrades rather than fails.** Vectors are only comparable
//! to query vectors from the same model, so semantic search must stop — but the
//! keyword path is model-independent and keeps working. A usable answer beats a
//! correct error (SPEC §3.5).

const std = @import("std");
const sqlite = @import("db/sqlite.zig");
const store = @import("db/store.zig");
const schema = @import("db/schema.zig");
const scan = @import("ingest/scan.zig");
const indexer = @import("ingest/indexer.zig");
const hybrid = @import("search/hybrid.zig");
const embed = @import("embed/llama.zig");
const equeue = @import("embed/queue.zig");
const proto = @import("ipc/proto.zig");
const paths = @import("util/paths.zig");

pub const max_conns: usize = 8;

pub const Options = struct {
    /// Load the model at startup instead of on first use. Costs ~1-2s of startup
    /// for a warm first query.
    preload_model: bool = false,
    /// Seconds between filesystem rescans. 195 files cost 0.09s to stat, so this
    /// can be frequent without mattering.
    scan_interval_s: u64 = 30,
    collection: []const u8 = "docs",
    root: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
};

pub const State = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: paths.Layout,
    opts: Options,

    db_path_z: [:0]const u8,
    root: []const u8,
    model_path: []const u8,
    collection_id: i64 = 0,

    queue: equeue.Queue = .{},

    /// Guards the fields below.
    mutex: std.Io.Mutex = .init,
    embedder: ?embed.Embedder = null,
    model_load_failed: bool = false,
    /// Semantic search unavailable: model missing, or its id does not match the
    /// vectors on disk.
    degraded_reason: ?[]const u8 = null,
    n_embd: u32 = 0,

    /// When an interactive job was last served. The ingest loop backs off for a
    /// short window afterwards.
    ///
    /// Strict priority alone is not enough: ingest submits one chunk and waits, so
    /// the ingest queue is never deeper than 1 and there is nothing to jump ahead
    /// of. What a query actually waits for is the chunk already inside
    /// llama_decode, which cannot be interrupted — measured at ~300ms.
    ///
    /// This does not help the first query of a burst. It does keep the rest of a
    /// burst fast, which is the shape real use has: a person or an agent runs
    /// several searches in a row. Index freshness can wait a second; the person
    /// cannot.
    last_interactive_ms: std.atomic.Value(i64) = .init(0),

    running: std.atomic.Value(bool) = .init(true),
    started_ms: i64 = 0,
    conns: std.atomic.Value(usize) = .init(0),

    pub fn shuttingDown(self: *State) bool {
        return !self.running.load(.acquire);
    }

    /// Load the model if it is not loaded yet. Called from the embedder thread
    /// only, so the llama_context is never touched by two threads.
    fn ensureEmbedder(self: *State) !*embed.Embedder {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.embedder) |*e| return e;
        if (self.model_load_failed) return error.ModelUnavailable;

        self.embedder = embed.Embedder.init(self.gpa, self.model_path, .{}) catch |err| {
            self.model_load_failed = true;
            self.degraded_reason = "embedding model failed to load";
            return err;
        };
        self.n_embd = self.embedder.?.n_embd;
        return &self.embedder.?;
    }

    pub fn modelLoaded(self: *State) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.embedder != null;
    }

    pub fn degraded(self: *State) ?[]const u8 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.degraded_reason;
    }

    fn setDegraded(self: *State, reason: []const u8) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.degraded_reason = reason;
    }

    /// Count tokens with the embedding model's own tokenizer, via the same queue.
    pub fn countTokens(self: *State, prio: equeue.Priority, text: []const u8) !usize {
        if (self.degraded()) |_| return error.ModelUnavailable;
        var job: equeue.Job = .{ .kind = .count_tokens, .text = text };
        try self.queue.push(self.io, &job, prio);
        try job.wait(self.io);
        return job.count;
    }

    /// Block until no interactive request has been seen for the backoff window.
    /// Called by the ingest path before every chunk.
    pub fn waitWhileInteractive(self: *State) void {
        while (self.interactiveRecently() and !self.shuttingDown()) {
            std.Io.sleep(self.io, .{ .nanoseconds = 50 * std.time.ns_per_ms }, .awake) catch return;
        }
    }

    /// True while a recent interactive request should keep ingest quiet.
    pub fn interactiveRecently(self: *State) bool {
        const last = self.last_interactive_ms.load(.acquire);
        if (last == 0) return false;
        return nowMs(self.io) - last < ingest_backoff_ms;
    }

    /// Submit an embedding job and block until the embedder thread finishes it.
    /// `.interactive` jumps the entire ingest backlog.
    pub fn embedText(
        self: *State,
        prio: equeue.Priority,
        kind: equeue.Kind,
        heading_path: []const u8,
        text: []const u8,
        out: []f32,
    ) !void {
        if (self.degraded()) |_| return error.ModelUnavailable;
        if (prio == .interactive) self.last_interactive_ms.store(nowMs(self.io), .release);
        var job: equeue.Job = .{
            .kind = kind,
            .text = text,
            .heading_path = heading_path,
            .out = out,
        };
        try self.queue.push(self.io, &job, prio);
        return job.wait(self.io);
    }
};

// ---------------------------------------------------------------------------
// threads
// ---------------------------------------------------------------------------

/// Owns the single llama_context for the process lifetime.
fn embedderThread(st: *State) void {
    while (st.queue.pop(st.io)) |job| {
        const e = st.ensureEmbedder() catch |err| {
            st.queue.finish(st.io, job, err);
            // Every other waiter would block forever otherwise.
            st.queue.drainWithError(st.io, err);
            continue;
        };
        switch (job.kind) {
            .document => if (e.embedDocument(job.heading_path, job.text, job.out)) |_| {
                st.queue.finish(st.io, job, null);
            } else |err| st.queue.finish(st.io, job, err),
            .query => if (e.embedQuery(embed.default_query_task, job.text, job.out)) |_| {
                st.queue.finish(st.io, job, null);
            } else |err| st.queue.finish(st.io, job, err),
            .count_tokens => if (e.countTokens(job.text)) |n| {
                job.count = n;
                st.queue.finish(st.io, job, null);
            } else |err| st.queue.finish(st.io, job, err),
        }
    }
}

/// How long the ingest loop stays out of the way after an interactive request.
pub const ingest_backoff_ms: i64 = 1_500;

/// The only writer. Rescans the filesystem, then drains the pending queue.
fn ingestThread(st: *State) void {
    var db = store.open(st.db_path_z, .read_write) catch return;
    defer db.close();
    var s = store.Store.init(&db);

    const now_ms = nowMs(st.io);
    st.collection_id = s.ensureCollection(st.opts.collection, st.root, now_ms) catch return;

    // Bind the model identity on first run; on later runs compare it. Mismatch
    // means the stored vectors were produced by a different model and are not
    // comparable to anything this process would compute.
    checkModelIdentity(st, &db) catch {};

    while (!st.shuttingDown()) {
        _ = scan.reconcile(st.gpa, st.io, &s, st.collection_id, st.root, .{}, nowMs(st.io)) catch {};

        if (st.degraded() == null) {
            drainPending(st, &s) catch {};
        }

        // Sleep in short slices so shutdown does not wait a whole interval.
        var slept: u64 = 0;
        while (slept < st.opts.scan_interval_s and !st.shuttingDown()) : (slept += 1) {
            std.Io.sleep(st.io, .{ .nanoseconds = std.time.ns_per_s }, .awake) catch {};
        }
    }
}

/// Embed and write pending documents one at a time, re-checking shutdown between
/// documents so a stop request does not wait for the whole backlog.
fn drainPending(st: *State, s: *store.Store) !void {
    while (!st.shuttingDown()) {
        st.waitWhileInteractive();
        if (st.shuttingDown()) return;

        var arena_state = std.heap.ArenaAllocator.init(st.gpa);
        defer arena_state.deinit();
        const pending = try s.listPending(arena_state.allocator(), st.collection_id, 1);
        if (pending.items.len == 0) return;

        const doc = pending.items[0];
        indexer.indexOneQueued(st, s, doc, nowMs(st.io)) catch |err| {
            s.rollback();
            s.deleteChunks(doc.id) catch {};
            var buf: [128]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "{t}", .{err}) catch "unknown error";
            s.markFailed(doc.id, msg) catch {};
        };
    }
}

fn checkModelIdentity(st: *State, db: *sqlite.Db) !void {
    const hash = @import("util/hash.zig");
    std.Io.Dir.accessAbsolute(st.io, st.model_path, .{}) catch {
        st.setDegraded("embedding model file not found");
        return;
    };
    const digest = try hash.fileSha256(st.io, st.model_path);
    const task_digest = hash.bytesSha256(embed.default_query_task);
    const id = try std.fmt.allocPrint(st.gpa, "{s}@{s}+task:{s}", .{
        std.fs.path.basename(st.model_path), digest[0..16], task_digest[0..8],
    });
    defer st.gpa.free(id);

    var buf: [256]u8 = undefined;
    if (try schema.getMeta(db, "embedding_model_id", &buf)) |stored| {
        if (!std.mem.eql(u8, stored, id)) {
            // Not repaired automatically: rebuilding every vector costs minutes
            // and burns battery, so it stays the user's explicit decision.
            st.setDegraded("embedding model changed; semantic search disabled, run: zkb reindex");
            return;
        }
    } else {
        try schema.setMeta(db, "embedding_model_id", id);
    }
}

/// One connection, one thread, one read-only database handle.
fn connThread(st: *State, stream: std.Io.net.Stream) void {
    defer {
        stream.close(st.io);
        _ = st.conns.fetchSub(1, .release);
    }

    var db = store.open(st.db_path_z, .read_only) catch return;
    defer db.close();

    var rbuf: [proto.max_line_bytes]u8 = undefined;
    var wbuf: [256 * 1024]u8 = undefined;
    var reader = stream.reader(st.io, &rbuf);
    var writer = stream.writer(st.io, &wbuf);
    const w = &writer.interface;

    while (!st.shuttingDown()) {
        // Inclusive, not Exclusive: `takeDelimiterExclusive` returns the content
        // without the delimiter but *leaves the delimiter in the stream*. Reading
        // that way and looping spins forever — the next call immediately returns
        // an empty slice at the same position and never advances. Measured as
        // 100% CPU in mem.findScalarPos with the second request never read.
        const raw = reader.interface.takeDelimiterInclusive('\n') catch return;
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        if (line.len == 0) continue;

        handleLine(st, &db, w, line) catch return;
        w.flush() catch return;
    }
}

fn handleLine(st: *State, db: *sqlite.Db, w: *std.Io.Writer, line: []const u8) !void {
    var req = proto.parseRequest(st.gpa, line) catch |err| {
        const msg = switch (err) {
            error.InvalidJson => "request is not a JSON object",
            error.MissingId => "missing or non-integer id",
            error.MissingMethod => "missing method",
            error.UnknownMethod => "unknown method",
            error.OutOfMemory => "out of memory",
        };
        // id 0: the request had no usable id to echo back.
        return proto.writeError(w, 0, .bad_request, msg, null);
    };
    defer req.deinit();

    switch (req.method) {
        .health => try handleHealth(st, w, req.id),
        .stats => try handleStats(st, db, w, req.id),
        .search => try handleSearch(st, db, w, &req),
        .index => try handleIndex(st, w, req.id),
        .shutdown => {
            try proto.beginOk(w, req.id);
            try w.writeAll("{\"stopping\":true}");
            try proto.finishOk(w);
            st.running.store(false, .release);
        },
    }
}

fn handleHealth(st: *State, w: *std.Io.Writer, id: i64) !void {
    const d = st.queue.depth(st.io);
    try proto.beginOk(w, id);
    try w.print(
        "{{\"version\":\"{s}\",\"protocol\":{d},\"uptime_ms\":{d},\"model_loaded\":{}," ++
            "\"queue\":{{\"interactive\":{d},\"ingest\":{d}}},\"degraded\":",
        .{
            @import("root.zig").version,  proto.protocol_version,
            nowMs(st.io) - st.started_ms, st.modelLoaded(),
            d.interactive,                d.ingest,
        },
    );
    if (st.degraded()) |reason| try std.json.Stringify.value(reason, .{}, w) else try w.writeAll("null");
    try w.writeAll("}");
    try proto.finishOk(w);
}

fn handleStats(st: *State, db: *sqlite.Db, w: *std.Io.Writer, id: i64) !void {
    var s = store.Store.init(db);
    const c = try s.counts();
    try proto.beginOk(w, id);
    try w.print(
        "{{\"docs\":{d},\"chunks\":{d},\"pending\":{d},\"failed\":{d}," ++
            "\"fts_rows\":{d},\"vec_rows\":{d},\"drift\":{}," ++
            "\"served_interactive\":{d},\"served_ingest\":{d},\"max_preempted\":{d}}}",
        .{
            c.docs,                                           c.chunks,
            c.pending,                                        c.failed,
            c.fts_rows,                                       c.vec_rows,
            c.chunks != c.fts_rows or c.chunks != c.vec_rows, st.queue.served_interactive,
            st.queue.served_ingest,                           st.queue.max_preempted,
        },
    );
    try proto.finishOk(w);
}

fn handleIndex(_: *State, w: *std.Io.Writer, id: i64) !void {
    // The ingest thread owns the only write connection and rescans on its own
    // schedule. A second writer here would break the single-writer invariant that
    // makes SQLITE_BUSY a non-issue, so this only acknowledges.
    try proto.beginOk(w, id);
    try w.writeAll("{\"queued\":true,\"note\":\"the ingest thread rescans on its own interval\"}");
    try proto.finishOk(w);
}

fn handleSearch(st: *State, db: *sqlite.Db, w: *std.Io.Writer, req: *const proto.Request) !void {
    const query = req.str("query") orelse
        return proto.writeError(w, req.id, .bad_request, "search requires a query", null);
    const k: usize = @intCast(@max(1, req.int("k") orelse 10));
    var mode: hybrid.Mode = if (req.str("mode")) |m|
        std.meta.stringToEnum(hybrid.Mode, m) orelse
            return proto.writeError(w, req.id, .bad_request, "mode must be hybrid, vector or keyword", null)
    else
        .hybrid;

    var vec: ?[]f32 = null;
    defer if (vec) |v| st.gpa.free(v);

    if (mode != .keyword) {
        if (st.degraded()) |reason| {
            if (mode == .vector) {
                return proto.writeError(w, req.id, .model_mismatch, reason, "zkb reindex");
            }
            // Hybrid degrades to keyword rather than failing: the keyword path
            // does not depend on the model at all.
            mode = .keyword;
        } else {
            const dim: usize = @intCast(schema.embedding_dim);
            const buf = try st.gpa.alloc(f32, dim);
            st.embedText(.interactive, .query, "", query, buf) catch {
                st.gpa.free(buf);
                if (mode == .vector) {
                    return proto.writeError(w, req.id, .model_unavailable, "embedding failed", null);
                }
                // Hybrid keeps working without the vector path.
                return searchAndWrite(st, db, w, req.id, .keyword, query, null, k);
            };
            vec = buf;
        }
    }
    return searchAndWrite(st, db, w, req.id, mode, query, vec, k);
}

fn searchAndWrite(
    st: *State,
    db: *sqlite.Db,
    w: *std.Io.Writer,
    id: i64,
    mode: hybrid.Mode,
    query: []const u8,
    vec: ?[]const f32,
    k: usize,
) !void {
    var results = hybrid.search(st.gpa, db, mode, query, vec, null, .{ .top_k = k }) catch
        return proto.writeError(w, id, .internal, "search failed", null);
    defer results.deinit(st.gpa);

    var s = store.Store.init(db);
    const c = try s.counts();

    try proto.beginOk(w, id);
    try w.print("{{\"mode\":\"{t}\",\"fts_skipped\":{},", .{ results.mode, results.fts_skipped });
    try w.print(
        "\"index\":{{\"docs\":{d},\"chunks\":{d},\"pending\":{d},\"failed\":{d},\"complete\":{}}},",
        .{ c.docs, c.chunks, c.pending, c.failed, c.pending == 0 and c.failed == 0 },
    );
    if (st.degraded()) |reason| {
        try w.writeAll("\"degraded\":");
        try std.json.Stringify.value(reason, .{}, w);
        try w.writeAll(",");
    }
    try w.writeAll("\"dropped_terms\":[");
    for (results.dropped_terms, 0..) |d, i| {
        if (i != 0) try w.writeAll(",");
        try std.json.Stringify.value(d, .{}, w);
    }
    try w.writeAll("],\"hits\":[");
    for (results.hits, 0..) |h, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{{\"chunk_id\":{d},\"score\":{d:.6},", .{ h.chunk_id, h.score });
        if (h.vec_rank) |r| try w.print("\"vec_rank\":{d},", .{r}) else try w.writeAll("\"vec_rank\":null,");
        if (h.fts_rank) |r| try w.print("\"fts_rank\":{d},", .{r}) else try w.writeAll("\"fts_rank\":null,");
        try w.writeAll("\"collection\":");
        try std.json.Stringify.value(h.collection, .{}, w);
        try w.writeAll(",\"path\":");
        try std.json.Stringify.value(h.rel_path, .{}, w);
        try w.writeAll(",\"title\":");
        try std.json.Stringify.value(h.title, .{}, w);
        try w.writeAll(",\"heading_path\":");
        try std.json.Stringify.value(h.heading_path, .{}, w);
        try w.writeAll(",\"text\":");
        try std.json.Stringify.value(h.text, .{}, w);
        try w.print(",\"chunk_idx\":{d},\"n_tokens\":{d}}}", .{ h.idx, h.n_tokens });
    }
    try w.writeAll("]}");
    try proto.finishOk(w);
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

pub fn nowMs(io: std.Io) i64 {
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .real).nanoseconds, std.time.ns_per_ms));
}

/// Run in the foreground until told to stop. `on_ready` fires once the socket is
/// accepting, which is what `daemon start` waits for.
pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    opts: Options,
) !void {
    var layout = try paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    std.Io.Dir.createDirPath(.cwd(), io, layout.root) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const root = opts.root orelse blk: {
        const home = env.get("HOME") orelse return error.NoHomeDirectory;
        break :blk try std.fmt.allocPrint(gpa, "{s}/docs", .{home});
    };
    defer if (opts.root == null) gpa.free(root);

    const model_path = opts.model_path orelse
        try std.fmt.allocPrint(gpa, "{s}/Qwen3-Embedding-0.6B-Q8_0.gguf", .{layout.models});
    defer if (opts.model_path == null) gpa.free(model_path);

    const db_path_z = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path_z);

    // Migrate before any reader opens: read-only connections cannot do DDL.
    {
        var db = try store.open(db_path_z, .read_write);
        db.close();
    }

    var st: State = .{
        .gpa = gpa,
        .io = io,
        .layout = layout,
        .opts = opts,
        .db_path_z = db_path_z,
        .root = root,
        .model_path = model_path,
        .started_ms = nowMs(io),
    };

    // A stale socket from a killed daemon would block bind; nothing is listening
    // on it by definition, since we checked the pid file first.
    std.Io.Dir.deleteFileAbsolute(io, layout.sock) catch {};

    const addr = try std.Io.net.UnixAddress.init(layout.sock);
    var server = try addr.listen(io, .{});
    defer server.socket.close(io);
    defer std.Io.Dir.deleteFileAbsolute(io, layout.sock) catch {};

    // 0600: the socket is the whole authorization model, so it must not be
    // reachable by other users on the machine.
    try chmod600(layout.sock);

    try writePidFile(io, layout.pid);
    defer std.Io.Dir.deleteFileAbsolute(io, layout.pid) catch {};

    const embedder_thread = try std.Thread.spawn(.{}, embedderThread, .{&st});
    const ingest = try std.Thread.spawn(.{}, ingestThread, .{&st});

    if (opts.preload_model) {
        var probe: [1]f32 = .{0};
        // Deliberately ignored: a preload failure is reported through `degraded`,
        // and must not stop the daemon from serving keyword search.
        st.embedText(.interactive, .query, "", "preload", &probe) catch {};
    }

    while (!st.shuttingDown()) {
        const stream = server.accept(io) catch break;
        if (st.conns.load(.acquire) >= max_conns) {
            stream.close(io);
            continue;
        }
        _ = st.conns.fetchAdd(1, .acquire);
        const t = std.Thread.spawn(.{}, connThread, .{ &st, stream }) catch {
            _ = st.conns.fetchSub(1, .release);
            stream.close(io);
            continue;
        };
        t.detach();
    }

    // Shutdown: stop the workers, then checkpoint. Ingest is asked to stop
    // between documents so no transaction is abandoned mid-write.
    st.running.store(false, .release);
    st.queue.close(io);
    embedder_thread.join();
    ingest.join();

    if (st.embedder) |*e| e.deinit();

    var db = store.open(db_path_z, .read_write) catch return;
    defer db.close();
    db.exec("PRAGMA wal_checkpoint(TRUNCATE);") catch {};
}

fn writePidFile(io: std.Io, path: []const u8) !void {
    var file = try std.Io.Dir.createFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [32]u8 = undefined;
    var writer = file.writer(io, &buf);
    try writer.interface.print("{d}\n", .{std.c.getpid()});
    try writer.interface.flush();
}

fn chmod600(path: []const u8) !void {
    const c = @cImport({
        @cInclude("sys/stat.h");
    });
    var buf: [512]u8 = undefined;
    const z = std.fmt.bufPrintZ(&buf, "{s}", .{path}) catch return error.NameTooLong;
    if (c.chmod(z.ptr, 0o600) != 0) return error.AccessDenied;
}
