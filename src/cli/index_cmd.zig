//! `zkb index` — reconcile a root with the database, then embed what changed.
//!
//! Runs in the foreground for now; M2 moves the same two stages into the daemon.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub const Options = struct {
    root: ?[]const u8 = null,
    collection: []const u8 = "docs",
    model: ?[]const u8 = null,
    /// Re-embed everything, ignoring content hashes.
    force: bool = false,
};

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    opts: Options,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    // With a daemon running, the ingest thread already owns the only write
    // connection. Opening a second writer here would work — SQLite would
    // serialize it — but it would abandon the invariant the daemon is designed
    // around, and for no gain: the daemon has the model resident and this
    // process would have to load its own.
    if (!opts.force) {
        if (try viaDaemon(gpa, io, layout.sock, w)) |code| return code;
    }

    const root = opts.root orelse blk: {
        const home = env.get("HOME") orelse return error.NoHomeDirectory;
        break :blk try std.fmt.allocPrint(gpa, "{s}/docs", .{home});
    };
    defer if (opts.root == null) gpa.free(root);

    try layout.ensureDirs(io);

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = try zkb.store.open(db_path, .read_write);
    defer db.close();
    var s = zkb.store.Store.init(&db);

    const now_ms: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .real).nanoseconds, std.time.ns_per_ms));
    const cid = try s.ensureCollection(opts.collection, root, now_ms);

    if (opts.force) {
        // Force means "distrust the hashes": clear the stamps so every doc is
        // queued again. Chunks are replaced per-document during indexing.
        try db.exec("UPDATE docs SET indexed_at = NULL, index_error = NULL;");
    }

    try w.print("scanning {s}\n", .{root});
    try w.flush();

    const report = try zkb.scan.reconcile(gpa, io, &s, cid, root, .{}, now_ms);
    try w.print(
        "  seen {d}, unchanged {d}, queued {d}, renamed {d}, touched {d}, deleted {d}",
        .{ report.seen, report.unchanged, report.queued, report.renamed, report.touched, report.deleted },
    );
    if (report.unreadable != 0) try w.print(", unreadable {d}", .{report.unreadable});
    try w.writeAll("\n");
    try w.flush();

    // Resolution is about the link graph, not about ingestion: it must run even
    // when nothing needed indexing. The graph can be left unresolved by an
    // interrupted run or by an index written before a resolution fix, and
    // "nothing to index" is exactly when that goes unnoticed.
    try resolveAndReport(gpa, &db, w);

    // Before the pending check, not after: the docs root being up to date says
    // nothing about the memory root, and deciding to skip the model load on the
    // docs count alone would leave new memories unindexed.
    var kb_roots = kbRoots(&layout);
    const kb_pending = try reconcileKb(gpa, io, &s, &kb_roots, now_ms);
    if (kb_pending != 0) try w.print("  kb: {d} queued\n", .{kb_pending});

    // `counts()` is global, so this covers the docs root and both kb roots.
    const pending_before = (try s.counts()).pending;
    if (pending_before == 0) {
        try w.writeAll("nothing to index\n");
        return 0;
    }

    // Model identity is bound into the index: vectors are only comparable to
    // query vectors from the same model and the same query task (SPEC §3.3).
    const found = zkb.model_registry.resolve(gpa, io, env, &layout, opts.model, .q8_0) catch {
        try w.writeAll("model not found\nrun: zkb model pull\n");
        return 4;
    };
    defer found.deinit(gpa);
    const model_path = found.path;

    try w.print("loading model ...\n", .{});
    try w.flush();
    var embedder = try zkb.embed.Embedder.init(gpa, model_path, .{});
    defer embedder.deinit();

    const model_id = try modelId(gpa, io, model_path);
    defer gpa.free(model_id);
    try zkb.schema.setMeta(&db, "embedding_model_id", model_id);

    try w.print("embedding {d} document(s)\n", .{pending_before});
    try w.flush();

    const stats = try zkb.indexer.indexPending(gpa, io, &s, &embedder, cid, root, now_ms, .{});
    try indexKb(gpa, io, &s, &embedder, &kb_roots, now_ms, w);
    try resolveAndReport(gpa, &db, w);

    const c = try s.counts();
    try w.print("\nindexed {d} doc(s), {d} chunk(s)", .{ stats.docs_indexed, stats.chunks_written });
    if (stats.docs_failed != 0) try w.print(", {d} FAILED", .{stats.docs_failed});
    try w.writeAll("\n");
    try w.print("  wall {d:.1}s, {d} embed calls, {d:.1}ms avg\n", .{
        @as(f64, @floatFromInt(stats.total_ns)) / std.time.ns_per_s,
        stats.embed_calls,
        stats.avgEmbedMs(),
    });
    try w.print("  totals: {d} docs, {d} chunks, {d} pending, {d} failed\n", .{
        c.docs, c.chunks, c.pending, c.failed,
    });

    // Three-table drift would show up as searching content that is not there.
    if (c.chunks != c.fts_rows or c.chunks != c.vec_rows) {
        try w.print("  WARNING index drift: chunks {d}, fts {d}, vec {d}\n", .{
            c.chunks, c.fts_rows, c.vec_rows,
        });
        return 1;
    }

    if (stats.docs_failed != 0) {
        try w.writeAll("\nfailed documents:\n");
        var st = try db.prepare(
            "SELECT rel_path, index_error FROM docs WHERE index_error IS NOT NULL LIMIT 20",
        );
        defer st.finalize();
        while (try st.step()) {
            try w.print("  {s}: {s}\n", .{ st.columnText(0), st.columnText(1) });
        }
        return 1;
    }
    return 0;
}

/// Ask the running daemon to rescan, then wait until it has nothing pending.
///
/// Returns null when no daemon is listening, so the caller falls back to doing
/// the work in this process. `--force` also takes the local path: re-embedding
/// everything is a deliberate, long operation whose progress the user wants to
/// watch, not something to hand to a background thread.
fn viaDaemon(gpa: std.mem.Allocator, io: std.Io, sock: []const u8, w: *Writer) !?u8 {
    var c = zkb.ipc_client.Client.connect(io, sock) catch return null;
    defer c.close();

    {
        var resp = c.call(gpa, .index, "{}") catch return null;
        defer resp.deinit(gpa);
        if (!resp.ok) return null;
    }
    try w.writeAll("daemon rescanning ...\n");
    try w.flush();

    // Poll rather than wait for a push: the protocol is request/response, and a
    // rescan of a large corpus can take minutes, which no reasonable request
    // timeout would cover.
    var waited_ms: u64 = 0;
    while (waited_ms < 120_000) : (waited_ms += 200) {
        std.Io.sleep(io, .{ .nanoseconds = 200 * std.time.ns_per_ms }, .awake) catch {};

        var resp = c.call(gpa, .stats, "{}") catch break;
        defer resp.deinit(gpa);
        if (!resp.ok) break;
        const obj = (resp.result orelse break).object;
        // Both conditions: pending alone is zero in the window between asking
        // for a scan and the scan queueing anything.
        if (intOf(obj, "scanning") != 0) continue;
        if (intOf(obj, "pending") != 0) continue;

        try w.print("  {d} docs, {d} chunks, {d} failed\n", .{
            intOf(obj, "docs"), intOf(obj, "chunks"), intOf(obj, "failed"),
        });
        if (intOf(obj, "drift") != 0) {
            try w.writeAll("  WARNING index drift — run: zkb daemon stop && zkb index\n");
            return 1;
        }
        return 0;
    }
    try w.writeAll("still indexing; check: zkb status\n");
    return 0;
}

fn intOf(o: std.json.ObjectMap, key: []const u8) i64 {
    const v = o.get(key) orelse return 0;
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        .bool => |b| @intFromBool(b),
        else => 0,
    };
}

/// What zkb itself writes: `~/.zkb/data/memory/*.md` and `~/.zkb/data/*.csv`.
///
/// Two collections over nested roots, told apart only by extension — the memory
/// root is a subdirectory of the data root, and restricting the data collection
/// to `.csv` is what stops every memory being indexed twice.
///
/// Facts are indexed for *retrieval* (so "what do I know about my salary" can
/// find the fact row); the current value itself is always read straight from the
/// csv, never from the index (§16.5).
const KbRoot = struct {
    path: []const u8,
    name: []const u8,
    kind: zkb.store.Store.Kind,
    filters: zkb.scan.Filters,
    cid: i64 = 0,
    pending: usize = 0,
};

fn kbRoots(layout: *const zkb.paths.Layout) [2]KbRoot {
    return .{
        .{
            .path = layout.memory,
            .name = "memory",
            .kind = .memory,
            .filters = zkb.memory.scan_filters,
        },
        .{
            .path = layout.data,
            .name = "kb",
            .kind = .records,
            .filters = zkb.facts.scan_filters,
        },
    };
}

/// Reconcile happens before the model is loaded, so its result can decide
/// whether loading is needed at all — the docs root having nothing to do says
/// nothing about the memory root.
fn reconcileKb(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *zkb.store.Store,
    roots: []KbRoot,
    now_ms: i64,
) !usize {
    var total: usize = 0;
    for (roots) |*r| {
        std.Io.Dir.accessAbsolute(io, r.path, .{}) catch continue;
        r.cid = try s.ensureCollectionKind(r.name, r.path, r.kind, now_ms);
        const report = try zkb.scan.reconcile(gpa, io, s, r.cid, r.path, r.filters, now_ms);
        r.pending = report.queued;
        total += report.queued;

        // A `_schema.json` edit changes a type's shape without touching any csv,
        // so the scanner cannot see it. Checked here, where the result can still
        // affect this run's pending count.
        if (r.kind == .records) {
            const requeued = zkb.records.reconcileOverrides(gpa, io, s.db, r.path) catch 0;
            if (requeued != 0) {
                const again = try zkb.scan.reconcile(gpa, io, s, r.cid, r.path, r.filters, now_ms);
                _ = again;
                r.pending += 1;
                total += 1;
            }
        }
    }
    return total;
}

/// Indexing a records type can requeue its sibling files: a changed header means
/// the materialized table is rebuilt, which discards every file's rows, so they
/// all have to be read again (`records.ensureTable`). The pending list for a pass
/// is taken before that happens, so one pass is not enough — hence the loop.
///
/// It terminates because the rebuild only fires when the stored shape differs
/// from the inferred one, and after the first pass they agree.
const max_kb_passes: usize = 3;

fn indexKb(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *zkb.store.Store,
    embedder: *zkb.embed.Embedder,
    roots: []const KbRoot,
    now_ms: i64,
    w: *Writer,
) !void {
    for (roots) |r| {
        if (r.cid == 0 or r.pending == 0) continue;

        var docs: usize = 0;
        var chunks: usize = 0;
        var failed: usize = 0;
        var pass: usize = 0;
        while (pass < max_kb_passes) : (pass += 1) {
            const st = try zkb.indexer.indexPending(gpa, io, s, embedder, r.cid, r.path, now_ms, .{});
            docs += st.docs_indexed;
            chunks += st.chunks_written;
            failed += st.docs_failed;
            if (st.docs_indexed == 0) break;
        }

        try w.print("  {s}: {d} doc(s), {d} chunk(s)", .{ r.name, docs, chunks });
        if (pass > 1) try w.print(" in {d} passes", .{pass});
        if (failed != 0) try w.print(", {d} FAILED", .{failed});
        try w.writeAll("\n");
        try w.flush();
    }
}

/// Resolve pending links after a pass, never during one: a document can link to
/// one that has not been indexed yet, and resolving inline would call those
/// broken depending on filesystem iteration order.
fn resolveAndReport(gpa: std.mem.Allocator, db: *zkb.sqlite.Db, w: *Writer) !void {
    const resolved = zkb.maintain.resolveLinks(gpa, db) catch return;
    if (resolved != 0) {
        try w.print("  resolved {d} link(s)\n", .{resolved});
        try w.flush();
    }
}

/// "<file>@<sha16>+task:<sha8>" — see SPEC §3.3. The query task is part of the
/// identity because changing it shifts the query distribution away from the
/// stored document vectors.
fn modelId(gpa: std.mem.Allocator, io: std.Io, model_path: []const u8) ![]u8 {
    const digest = try zkb.hash.fileSha256(io, model_path);
    const task_digest = zkb.hash.bytesSha256(zkb.embed.default_query_task);
    return std.fmt.allocPrint(gpa, "{s}@{s}+task:{s}", .{
        std.fs.path.basename(model_path),
        digest[0..16],
        task_digest[0..8],
    });
}
