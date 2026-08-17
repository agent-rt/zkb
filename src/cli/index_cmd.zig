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

    const root = opts.root orelse blk: {
        const home = env.get("HOME") orelse return error.NoHomeDirectory;
        break :blk try std.fmt.allocPrint(gpa, "{s}/docs", .{home});
    };
    defer if (opts.root == null) gpa.free(root);

    std.Io.Dir.createDirPath(.cwd(), io, layout.root) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

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
    const model_path = opts.model orelse
        try std.fmt.allocPrint(gpa, "{s}/Qwen3-Embedding-0.6B-Q8_0.gguf", .{layout.models});
    defer if (opts.model == null) gpa.free(model_path);

    std.Io.Dir.accessAbsolute(io, model_path, .{}) catch {
        try w.print("model not found: {s}\nrun: zkb model pull\n", .{model_path});
        return 4;
    };

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

/// What zkb itself writes: `~/kb/memory/*.md` and `~/kb/*.csv`.
///
/// Two collections over nested roots, told apart only by extension — the memory
/// root is a subdirectory of the kb root, and restricting the kb collection to
/// `.csv` is what stops every memory being indexed twice.
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
            .path = layout.kb,
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
    }
    return total;
}

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
        const st = try zkb.indexer.indexPending(gpa, io, s, embedder, r.cid, r.path, now_ms, .{});
        try w.print("  {s}: {d} doc(s), {d} chunk(s)", .{ r.name, st.docs_indexed, st.chunks_written });
        if (st.docs_failed != 0) try w.print(", {d} FAILED", .{st.docs_failed});
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
