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
const rootsmod = @import("ingest/roots.zig");
const indexer = @import("ingest/indexer.zig");
const hybrid = @import("search/hybrid.zig");
const packmod = @import("search/pack.zig");
const embed = @import("embed/llama.zig");
const equeue = @import("embed/queue.zig");
const proto = @import("ipc/proto.zig");
const paths = @import("util/paths.zig");
const registry = @import("embed/registry.zig");
const fsevents = @import("util/fsevents.zig");
const maintain = @import("maintain.zig");
const recallmod = @import("recall.zig");
const tracemod = @import("search/trace.zig");
const records = @import("records.zig");
const memorymod = @import("memory.zig");
const factsmod = @import("facts.zig");

pub const max_conns: usize = 8;

pub const Options = struct {
    /// Load the model at startup instead of on first use. Costs ~1-2s of startup
    /// for a warm first query.
    preload_model: bool = false,
    /// Seconds between filesystem rescans. 195 files cost 0.09s to stat, so this
    /// can be frequent without mattering.
    scan_interval_s: u64 = 30,
    /// Hour of day (local, 0-23) for the daily maintenance pass. Defaults to
    /// 04:00 — the point is that it lands when nobody is waiting on a query
    /// (SPEC §14.9).
    maintain_hour: u8 = 4,
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
    /// Retrieval trace. Disabled unless $ZKB_TRACE=1, in which case every search
    /// and query appends one line.
    trace: tracemod.Writer = undefined,

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

    /// True from the moment a rescan is requested until the pass that follows
    /// it has finished. Without it a client cannot tell "nothing pending because
    /// everything is indexed" from "nothing pending because the scan that would
    /// have queued the work has not run yet" — and it always polls fast enough
    /// to hit the second case.
    scanning: std.atomic.Value(bool) = .init(true),

    /// Set by an `index` request. The ingest loop wakes early instead of
    /// waiting out the interval, which is the difference between "appended a row
    /// and it is queryable" taking under a second and taking up to 30.
    rescan: std.atomic.Value(bool) = .init(false),

    /// Collections an `index` request asked to register, waiting for the ingest
    /// thread to write them.
    ///
    /// A connection thread cannot write: the ingest thread owns the only write
    /// connection, and that invariant is what keeps SQLITE_BUSY off the table.
    /// So the request hands over the intent and the owner performs it. Before
    /// this existed, `--root` and `--collection` were simply not in the request
    /// body, and `zkb index --root X --collection Y` against a running daemon
    /// registered nothing while reporting success.
    register_mutex: std.Io.Mutex = .init,
    pending_registers: std.ArrayList(Registration) = .empty,

    /// How many times the anti-starvation deadline fired. Surfaced in `stats`
    /// because if this is nonzero and growing, the machine is under enough query
    /// load that ingestion is competing, which is worth being able to see.
    starvation_overrides: usize = 0,

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

    /// Block until no interactive request has been seen for the backoff window —
    /// but never for longer than `max_ingest_starvation_ms`.
    ///
    /// The deadline is not a nicety. Every interactive request refreshes
    /// `last_interactive_ms`, so under a steady stream of them the backoff window
    /// never expires and ingestion stops **forever**. Measured: a search every
    /// 100 ms kept a newly written document unindexed for the entire 20 s run,
    /// and it indexed within 3 s of the load stopping. An agent searching in a
    /// loop is not a pathological case here, it is the design target — and the
    /// failure is silent, since the index simply stays stale.
    ///
    /// So: yield to interactive work, but age out of it. Falling behind on a
    /// query by ~300 ms once every few seconds is a far better trade than an
    /// index that never updates.
    pub fn waitWhileInteractive(self: *State) void {
        const started = nowMs(self.io);
        while (self.interactiveRecently() and !self.shuttingDown()) {
            if (nowMs(self.io) - started >= max_ingest_starvation_ms) {
                self.starvation_overrides += 1;
                return;
            }
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

/// Granularity of the ingest loop's sleep, and so the upper bound on how long a
/// rescan request waits before anything happens.
pub const rescan_tick_ms: u64 = 100;

/// How long the ingest loop stays out of the way after an interactive request.
pub const ingest_backoff_ms: i64 = 1_500;

/// Upper bound on how long a single chunk may be deferred by that backoff.
///
/// Without it the backoff is unbounded under sustained load — see
/// `waitWhileInteractive`. Five seconds is long enough that a burst of queries
/// is fully absorbed, short enough that nobody notices the index lagging.
pub const max_ingest_starvation_ms: i64 = 5_000;

/// The only writer. Rescans the filesystem, then drains the pending queue.
fn ingestThread(st: *State) void {
    var db = store.open(st.db_path_z, .read_write) catch return;
    defer db.close();
    var s = store.Store.init(&db);

    const now_ms = nowMs(st.io);
    rootsmod.ensureOwn(&s, &st.layout, now_ms) catch return;
    _ = rootsmod.ensureDocs(&s, st.opts.collection, st.root, st.opts.root != null, now_ms) catch return;
    st.collection_id = s.findCollection(st.opts.collection) catch return orelse return;

    // Bind the model identity on first run; on later runs compare it. Mismatch
    // means the stored vectors were produced by a different model and are not
    // comparable to anything this process would compute.
    checkModelIdentity(st, &db) catch {};

    while (!st.shuttingDown()) {
        st.rescan.store(false, .release);
        st.scanning.store(true, .release);

        // Every root zkb knows about, not just the documents one, and read from
        // the database on every pass rather than from a literal: a collection
        // registered mid-session is picked up by the next pass, without a
        // restart. The kb roots were previously scanned only by `zkb index`,
        // which meant a records csv an agent appended to was never picked up by
        // the running daemon.
        applyRegistrations(st, &s);

        var pass_arena = std.heap.ArenaAllocator.init(st.gpa);
        defer pass_arena.deinit();
        const pass_roots = rootsmod.list(pass_arena.allocator(), st.io, &s) catch &.{};

        for (pass_roots) |r| {
            const cid = r.id;
            _ = scan.reconcile(st.gpa, st.io, &s, cid, r.path, r.filters, nowMs(st.io)) catch {};
            if (r.kind == .records) {
                _ = records.reconcileOverrides(st.gpa, st.io, &db, r.path) catch 0;
            }
            if (st.degraded() == null) {
                // Repeated because a records rebuild requeues the type's other
                // files, and those are not in the pass's own pending list.
                var pass: usize = 0;
                while (pass < 3) : (pass += 1) {
                    const did = drainPending(st, &s, cid, r.path) catch break;
                    if (did == 0) break;
                }
            }
        }

        // After the pass, never during it: a document can link to one not yet
        // indexed, and resolving inline would call those broken depending on
        // filesystem iteration order.
        _ = maintain.resolveLinks(st.gpa, &db) catch 0;
        st.scanning.store(false, .release);

        maybeRunDailyMaintenance(st, &db) catch {};

        // Sleep in short slices so neither shutdown nor an explicit rescan
        // request waits a whole interval.
        //
        // The slice is what bounds how fast an appended row becomes queryable:
        // at one second it dominated the whole append-to-query time, which was
        // otherwise down to the write. Ten atomic loads a second cost nothing.
        const ticks = st.opts.scan_interval_s * (std.time.ms_per_s / rescan_tick_ms);
        var slept: u64 = 0;
        while (slept < ticks and !st.shuttingDown() and
            !st.rescan.load(.acquire)) : (slept += 1)
        {
            std.Io.sleep(st.io, .{ .nanoseconds = rescan_tick_ms * std.time.ns_per_ms }, .awake) catch {};
        }
    }
}

/// Run the full check set once a day, and record it so `zkb maintain --since
/// last` can show what changed.
///
/// It lives on the ingest thread rather than in its own launchd job because it
/// wants the same single write connection — `maintenance_runs` is a write — and
/// because a second process would have to load nothing but still coordinate.
/// The vector checks read stored vectors only, so this competes for CPU but
/// never for the embedder (SPEC §14.9).
fn maybeRunDailyMaintenance(st: *State, db: *sqlite.Db) !void {
    const now_ms = nowMs(st.io);
    const day_ms = std.time.ms_per_day;

    // "Once since the last one, and only after the configured hour" — expressed
    // as a floor on elapsed time plus an hour check, so a machine that was
    // asleep at 04:00 still gets a run when it wakes rather than skipping a day.
    const last = (try db.queryI64("SELECT COALESCE(max(started_at), 0) FROM maintenance_runs")) orelse 0;
    if (now_ms - last < day_ms) return;

    const secs = @divTrunc(now_ms, std.time.ms_per_s);
    const day_secs = @mod(secs, std.time.s_per_day);
    const hour: u8 = @intCast(@divTrunc(day_secs, std.time.s_per_hour));
    // First run of a fresh index has no history to compare against, so it is
    // allowed at any hour; after that the schedule applies.
    if (last != 0 and hour != st.opts.maintain_hour) return;

    const checks = maintain.Check.default();
    var report = maintain.run(st.gpa, db, .{ .checks = checks, .now_ms = now_ms }) catch return;
    defer report.deinit(st.gpa);
    try maintain.record(st.gpa, db, &report, checks, now_ms);
}

/// The roots the ingest thread reconciles now come from `collections` — see
/// ingest/roots.zig for why they stopped being a literal here.
const IngestRoot = rootsmod.Root;

/// A collection to register, handed from a connection thread to the writer.
/// All strings are gpa-owned and freed once applied.
pub const Registration = struct {
    name: []const u8,
    root: []const u8,
    /// Newline-separated, or null to keep whatever is stored.
    extensions: ?[]const u8,
    include: ?[]const u8,

    fn deinit(self: Registration, gpa: std.mem.Allocator) void {
        gpa.free(self.name);
        gpa.free(self.root);
        if (self.extensions) |v| gpa.free(v);
        if (self.include) |v| gpa.free(v);
    }
};

/// Write any handed-over registrations. Called by the ingest thread only.
///
/// Applied before the pass reads the root list, so a collection registered by
/// the request that triggered this wake-up is scanned by that same pass rather
/// than the next one — otherwise `zkb index --root X` would return "nothing
/// pending" while X had never been looked at.
fn applyRegistrations(st: *State, s: *store.Store) void {
    st.register_mutex.lockUncancelable(st.io);
    const taken = st.pending_registers.toOwnedSlice(st.gpa) catch &.{};
    st.register_mutex.unlock(st.io);

    defer st.gpa.free(taken);
    for (taken) |r| {
        defer r.deinit(st.gpa);
        _ = s.upsertCollection(r.name, r.root, .documents, r.extensions, r.include, nowMs(st.io)) catch {};
    }
}

/// Embed and write pending documents one at a time, re-checking shutdown between
/// documents so a stop request does not wait for the whole backlog.
fn drainPending(st: *State, s: *store.Store, collection_id: i64, root: []const u8) !usize {
    var done: usize = 0;
    while (!st.shuttingDown()) {
        st.waitWhileInteractive();
        if (st.shuttingDown()) return done;

        // Between documents, not only between passes. A registration arriving
        // while a large backlog is draining would otherwise wait for the whole
        // pass: measured on a first run over 606 documents, `zkb index --root X`
        // timed out after 120s with X still absent from `collections`. The row is
        // what makes the collection exist for `zkb status` and for the next pass,
        // and one document's embed time is a much better bound than one pass's.
        applyRegistrations(st, s);

        var arena_state = std.heap.ArenaAllocator.init(st.gpa);
        defer arena_state.deinit();
        const pending = try s.listPending(arena_state.allocator(), collection_id, 1);
        if (pending.items.len == 0) return done;

        const doc = pending.items[0];
        indexer.indexOneQueued(st, s, collection_id, root, doc, nowMs(st.io)) catch |err| {
            s.rollback();
            s.deleteChunks(doc.id) catch {};
            var buf: [128]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "{t}", .{err}) catch "unknown error";
            s.markFailed(doc.id, msg) catch {};
        };
        done += 1;
    }
    return done;
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
        .query => try handleQuery(st, db, w, &req),
        .recall => try handleRecall(st, db, w, &req),
        .index => try handleIndex(st, w, req.id, &req),
        .maintain => try handleMaintain(st, db, w, &req),
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
            "\"served_interactive\":{d},\"served_ingest\":{d},\"max_preempted\":{d}," ++
            "\"scanning\":{},\"starvation_overrides\":{d}}}",
        .{
            c.docs,                                           c.chunks,
            c.pending,                                        c.failed,
            c.fts_rows,                                       c.vec_rows,
            c.chunks != c.fts_rows or c.chunks != c.vec_rows, st.queue.served_interactive,
            st.queue.served_ingest,                           st.queue.max_preempted,
            st.scanning.load(.acquire),          st.starvation_overrides,
        },
    );
    try proto.finishOk(w);
}

fn handleQuery(st: *State, db: *sqlite.Db, w: *std.Io.Writer, req: *const proto.Request) !void {
    const query = req.str("query") orelse
        return proto.writeError(w, req.id, .bad_request, "query requires a query", null);
    var cfg: packmod.Config = .{};
    if (req.int("budget")) |b| cfg.budget_tokens = @intCast(@max(0, b));
    if (req.int("neighbors")) |n| cfg.neighbors = @max(0, n);
    if (req.int("candidates")) |c| cfg.candidates = @intCast(@max(1, c));

    var mode: hybrid.Mode = .hybrid;
    var vec: ?[]f32 = null;
    defer if (vec) |v| st.gpa.free(v);

    if (st.degraded()) |_| {
        mode = .keyword;
    } else {
        const dim: usize = @intCast(schema.embedding_dim);
        const buf = try st.gpa.alloc(f32, dim);
        if (st.embedText(.interactive, .query, "", query, buf)) |_| {
            vec = buf;
        } else |_| {
            st.gpa.free(buf);
            mode = .keyword;
        }
    }

    const t0 = nowMs(st.io);
    const coll = requestedCollection(db, req) catch |e| switch (e) {
        error.UnknownCollection => return proto.writeError(
            w, req.id, .bad_request, "unknown collection", "zkb status lists them"),
        else => return proto.writeError(w, req.id, .internal, "search failed", null),
    };
    var results = hybrid.search(st.gpa, db, mode, query, vec, coll orelse null, .{
        .top_k = cfg.candidates,
        .candidates = @max(50, cfg.candidates),
    }) catch return proto.writeError(w, req.id, .internal, "search failed", null);
    defer results.deinit(st.gpa);
    st.trace.record(st.gpa, query, &results, nowMs(st.io) - t0);

    var p = packmod.assemble(st.gpa, db, query, &results, cfg) catch
        return proto.writeError(w, req.id, .internal, "pack assembly failed", null);
    defer p.deinit(st.gpa);

    try proto.beginOk(w, req.id);
    try packmod.renderJson(w, &p);
    try proto.finishOk(w);
}

/// Memories plus the facts snapshot. Read-only, so it runs on the connection
/// thread; the write side (`remember`) stays in the CLI, because a second writer
/// would break the single-writer invariant this daemon is built on.
fn handleRecall(st: *State, db: *sqlite.Db, w: *std.Io.Writer, req: *const proto.Request) !void {
    const query = req.str("query") orelse "";
    var cfg: recallmod.Config = .{};
    if (req.int("budget")) |b| cfg.budget_tokens = @intCast(@max(0, b));
    if (req.int("candidates")) |c| cfg.candidates = @intCast(@max(1, c));

    var vec: ?[]f32 = null;
    defer if (vec) |v| st.gpa.free(v);
    if (query.len != 0 and st.degraded() == null) {
        const buf = try st.gpa.alloc(f32, @intCast(schema.embedding_dim));
        if (st.embedText(.interactive, .query, "", query, buf)) |_| {
            vec = buf;
        } else |_| st.gpa.free(buf);
    }

    // Facts come from `facts.csv`, not from the index: the index can be stale or
    // mid-rebuild, and a stale salary reads exactly like a current one.
    const current = factsmod.currentAll(st.gpa, st.io, st.layout.facts) catch &.{};
    defer {
        for (current) |f| f.deinit(st.gpa);
        st.gpa.free(current);
    }

    var r = recallmod.assemble(st.gpa, db, query, vec, cfg) catch
        return proto.writeError(w, req.id, .internal, "recall failed", null);
    defer r.deinit(st.gpa);

    try proto.beginOk(w, req.id);
    try w.writeAll("{\"facts\":");
    try recallmod.renderFactsJson(w, current);
    try w.writeAll(",\"memories\":");
    try packmod.renderJson(w, &r.pack);
    try w.writeAll("}");
    try proto.finishOk(w);
}

fn handleMaintain(st: *State, db: *sqlite.Db, w: *std.Io.Writer, req: *const proto.Request) !void {
    var report = maintain.run(st.gpa, db, .{}) catch
        return proto.writeError(w, req.id, .internal, "maintenance failed", null);
    defer report.deinit(st.gpa);

    try proto.beginOk(w, req.id);
    try w.print("{{\"link_graph_empty\":{},\"findings\":[", .{report.link_graph_empty});
    for (report.findings, 0..) |f, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{{\"check\":\"{t}\",\"key\":", .{f.check});
        try std.json.Stringify.value(f.key, .{}, w);
        try w.writeAll(",\"path\":");
        try std.json.Stringify.value(f.path, .{}, w);
        try w.writeAll(",\"detail\":");
        try std.json.Stringify.value(f.detail, .{}, w);
        try w.writeAll("}");
    }
    try w.writeAll("]}");
    try proto.finishOk(w);
}

fn handleIndex(st: *State, w: *std.Io.Writer, id: i64, req: *const proto.Request) !void {
    // A request naming a root is asking for that root to be part of the index
    // from now on, not just scanned once. Queue it for the writer; the reply is
    // still immediate, because the client polls `stats` for completion.
    if (req.str("root")) |root| {
        const name = req.str("collection") orelse "docs";
        const reg: Registration = .{
            .name = try st.gpa.dupe(u8, name),
            .root = try st.gpa.dupe(u8, root),
            .extensions = if (req.str("extensions")) |v| try st.gpa.dupe(u8, v) else null,
            .include = if (req.str("include")) |v| try st.gpa.dupe(u8, v) else null,
        };
        st.register_mutex.lockUncancelable(st.io);
        defer st.register_mutex.unlock(st.io);
        st.pending_registers.append(st.gpa, reg) catch reg.deinit(st.gpa);
    }

    // Still no writing from this thread — the ingest thread owns the only write
    // connection, and a second writer would break the single-writer invariant
    // that keeps SQLITE_BUSY off the table. What this does is wake it, which
    // turns "queryable within the scan interval" into "queryable in about a
    // second" without touching that invariant.
    // Both set before replying, so a client that polls immediately cannot
    // observe the gap between "asked for a scan" and "scan started".
    st.scanning.store(true, .release);
    st.rescan.store(true, .release);
    try proto.beginOk(w, id);
    try w.writeAll("{\"queued\":true,\"note\":\"ingest thread woken\"}");
    try proto.finishOk(w);
}

/// 请求里的 collection 名解析成 id。名字不存在返回 null 之外的 error，让调用方能报错而
/// 不是静默返回全库结果——`--collection` 被无视时命令看起来完全正常，只是过滤没生效。
fn requestedCollection(db: *sqlite.Db, req: *const proto.Request) !??i64 {
    const name = req.str("collection") orelse return @as(??i64, null);
    var st = try db.prepare("SELECT id FROM collections WHERE name = ?1");
    defer st.finalize();
    try st.bindText(1, name);
    if (!try st.step()) return error.UnknownCollection;
    return @as(??i64, st.columnI64(0));
}

fn handleSearch(st: *State, db: *sqlite.Db, w: *std.Io.Writer, req: *const proto.Request) !void {
    const query = req.str("query") orelse
        return proto.writeError(w, req.id, .bad_request, "search requires a query", null);
    const k: usize = @intCast(@max(1, req.int("k") orelse 10));
    const coll = requestedCollection(db, req) catch |e| switch (e) {
        error.UnknownCollection => return proto.writeError(
            w, req.id, .bad_request, "unknown collection", "zkb status lists them"),
        else => return proto.writeError(w, req.id, .internal, "search failed", null),
    } orelse null;
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
                return searchAndWrite(st, db, w, req.id, .keyword, query, null, k, coll);
            };
            vec = buf;
        }
    }
    return searchAndWrite(st, db, w, req.id, mode, query, vec, k, coll);
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
    collection_id: ?i64,
) !void {
    const t0 = nowMs(st.io);
    var results = hybrid.search(st.gpa, db, mode, query, vec, collection_id, .{ .top_k = k }) catch
        return proto.writeError(w, id, .internal, "search failed", null);
    defer results.deinit(st.gpa);
    st.trace.record(st.gpa, query, &results, nowMs(st.io) - t0);

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

    try layout.ensureDirs(io);

    const root = opts.root orelse blk: {
        const home = env.get("HOME") orelse return error.NoHomeDirectory;
        break :blk try std.fmt.allocPrint(gpa, "{s}/docs", .{home});
    };
    defer if (opts.root == null) gpa.free(root);

    // Resolved once at startup: the daemon holds the path for its whole life,
    // and a Hugging Face cache hit should not be re-discovered per request.
    const found = try registry.resolve(gpa, io, env, &layout, opts.model_path, .q8_0);
    const model_path = found.path;
    defer if (opts.model_path == null) gpa.free(model_path);

    const db_path_z = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path_z);

    // Migrate before any reader opens: read-only connections cannot do DDL.
    // Register the built-in collections in the same window — still before any
    // thread exists, so the single-writer invariant is not in play yet — because
    // the watcher below reads the root list back out, and on a first-ever start
    // it would otherwise find an empty table and watch nothing.
    {
        var db = try store.open(db_path_z, .read_write);
        defer db.close();
        var s = store.Store.init(&db);
        rootsmod.ensureOwn(&s, &layout, nowMs(io)) catch {};
        _ = rootsmod.ensureDocs(&s, opts.collection, root, opts.root != null, nowMs(io)) catch {};
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
        .trace = tracemod.Writer.init(io, env, layout.trace),
    };

    // A stale socket from a killed daemon would block bind; nothing is listening
    // on it by definition, since we checked the pid file first.
    std.Io.Dir.deleteFileAbsolute(io, layout.sock) catch {};

    const addr = std.Io.net.UnixAddress.init(layout.sock) catch |err| {
        // The sun_path limit is a kernel constant, not something we can work
        // around, so say which limit was hit and by how much — `NameTooLong`
        // raised from inside the std networking stack points nowhere useful.
        std.debug.print(
            "socket path is {d} bytes, over the unix sun_path limit:\n  {s}\n" ++
                "set $ZKB_HOME to a shorter path\n",
            .{ layout.sock.len, layout.sock },
        );
        return err;
    };
    var server = try addr.listen(io, .{});
    defer server.socket.close(io);
    defer std.Io.Dir.deleteFileAbsolute(io, layout.sock) catch {};

    // 0600: the socket is the whole authorization model, so it must not be
    // reachable by other users on the machine.
    try chmod600(layout.sock);

    try writePidFile(io, layout.pid);
    defer std.Io.Dir.deleteFileAbsolute(io, layout.pid) catch {};

    // Watches the same roots the ingest loop scans. Purely an accelerator: if it
    // fails to start, the interval scan still finds everything, just later.
    var watch_arena = std.heap.ArenaAllocator.init(gpa);
    defer watch_arena.deinit();
    var watch_roots: std.ArrayList([]const u8) = .empty;
    {
        // Read-only: the ingest thread has not started, but it will own the only
        // write connection once it does, and this needs nothing more than a list.
        var db = store.open(db_path_z, .read_only) catch null;
        if (db) |*d| {
            defer d.close();
            var s = store.Store.init(d);
            const rs = rootsmod.list(watch_arena.allocator(), io, &s) catch &.{};
            for (rs) |r| watch_roots.append(watch_arena.allocator(), r.path) catch {};
        }
    }
    var watcher = fsevents.Watcher.start(gpa, watch_roots.items, &st.rescan) catch
        fsevents.Watcher{ .flag = &st.rescan };
    _ = &watcher;

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
