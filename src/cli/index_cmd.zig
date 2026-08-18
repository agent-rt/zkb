//! `zkb index` — reconcile a root with the database, then embed what changed.
//!
//! Runs in the foreground for now; M2 moves the same two stages into the daemon.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub const Options = struct {
    /// Repeatable `--root`. Several are folded into one collection root plus
    /// include patterns (see ingest/roots.zig), because a doc's rel_path is
    /// relative to a single root and keeping it that way is worth more than
    /// storing a list.
    roots: []const []const u8 = &.{},
    collection: []const u8 = "docs",
    /// Repeatable `--ext`, with or without the dot. Empty means the default set.
    extensions: []const []const u8 = &.{},
    /// Repeatable `--include`, glob against the path relative to the root.
    include: []const []const u8 = &.{},
    model: ?[]const u8 = null,
    /// Re-embed everything, ignoring content hashes.
    force: bool = false,
};

/// What to register, derived from the flags: exactly one root, and the two
/// filter lists in the newline-separated form they are stored in.
const Registration = struct {
    root: []const u8,
    extensions: ?[]const u8,
    include: ?[]const u8,
};

/// Fold the flags into a single registration, or explain why they cannot be.
///
/// `--include` and several `--root`s are refused together on purpose: they are
/// two ways of saying the same thing, and combining them means inventing a rule
/// about whether they intersect or union. Narrowing by file type is `--ext`,
/// which composes with either.
fn registrationFrom(
    arena: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    opts: Options,
    w: *Writer,
) !?Registration {
    if (opts.roots.len > 1 and opts.include.len > 0) {
        try w.writeAll(
            \\--include cannot be combined with several --root: both select paths.
            \\  several roots  -> the directories that exist now
            \\  one --include  -> a pattern, so directories added later match too
            \\
        );
        return error.ConflictingOptions;
    }

    const default_root = if (opts.roots.len == 0) blk: {
        const home = env.get("HOME") orelse return error.NoHomeDirectory;
        break :blk try std.fmt.allocPrint(arena, "{s}/docs", .{home});
    } else null;

    const given: []const []const u8 = if (default_root) |d| &.{d} else opts.roots;

    // Absolute, because the root is stored and later read by the daemon, whose
    // working directory has nothing to do with the one this command ran in.
    var abs: std.ArrayList([]const u8) = .empty;
    for (given) |p| try abs.append(arena, try absolutize(arena, io, p));

    const folded = zkb.roots.fold(arena, abs.items) catch |e| switch (e) {
        error.RootsTooDisjoint => {
            try w.writeAll(
                \\those roots share no common directory worth scanning.
                \\register them as separate collections, one --root each.
                \\
            );
            return error.ConflictingOptions;
        },
        else => return e,
    };

    var include: std.ArrayList([]const u8) = .empty;
    for (folded.include) |p| try include.append(arena, p);
    for (opts.include) |p| try include.append(arena, p);

    var exts: std.ArrayList([]const u8) = .empty;
    for (opts.extensions) |e| {
        // `--ext md` and `--ext .md` mean the same thing; the stored form has the
        // dot because that is what the matcher compares against.
        const dotted = if (std.mem.startsWith(u8, e, ".")) e else try std.fmt.allocPrint(arena, ".{s}", .{e});
        try exts.append(arena, dotted);
    }

    return .{
        .root = folded.root,
        .extensions = try zkb.roots.joinList(arena, exts.items),
        .include = try zkb.roots.joinList(arena, include.items),
    };
}

/// `/a/b/` and `/a/b` are the same directory, and the stored root is compared and
/// sliced against doc paths, so only one of the two forms may reach the database.
/// `/` itself keeps its slash — there is nothing left to strip.
fn stripTrailingSlash(p: []const u8) []const u8 {
    var end = p.len;
    while (end > 1 and p[end - 1] == '/') end -= 1;
    return p[0..end];
}

/// A root is stored, and later read by a daemon whose working directory has
/// nothing to do with the one this command ran in, so a relative path has to be
/// resolved now or it means something different later.
///
/// An already-absolute path is stored as given rather than canonicalized: it
/// would otherwise change under a symlinked `~/docs`, and the roots already in
/// the database are the uncanonicalized form.
fn absolutize(arena: std.mem.Allocator, io: std.Io, p: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(p)) return stripTrailingSlash(p);
    const abs = try std.Io.Dir.cwd().realPathFileAlloc(io, stripTrailingSlash(p), arena);
    return abs;
}

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    opts: Options,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Resolved before the daemon is contacted, so a bad flag combination is
    // rejected here rather than half-applied by whichever path runs.
    const reg = (registrationFrom(arena, io, env, opts, w) catch |e| switch (e) {
        error.ConflictingOptions => return 2,
        else => return e,
    }) orelse return 2;

    // With a daemon running, the ingest thread already owns the only write
    // connection. Opening a second writer here would work — SQLite would
    // serialize it — but it would abandon the invariant the daemon is designed
    // around, and for no gain: the daemon has the model resident and this
    // process would have to load its own.
    if (!opts.force) {
        if (try viaDaemon(gpa, io, &layout, w, opts.collection, reg)) |code| return code;
    }

    const root = reg.root;

    try layout.ensureDirs(io);

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = try zkb.store.open(db_path, .read_write);
    defer db.close();
    var s = zkb.store.Store.init(&db);

    const now_ms: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .real).nanoseconds, std.time.ns_per_ms));
    const cid = try s.upsertCollection(opts.collection, root, .documents, reg.extensions, reg.include, now_ms);
    const filters = try zkb.roots.filtersFor(arena, .{
        .id = cid,
        .name = opts.collection,
        .root = root,
        .kind = .documents,
        .extensions = reg.extensions,
        .include = reg.include,
    });

    if (opts.force) {
        // Force means "distrust the hashes": clear the stamps so every doc is
        // queued again. Chunks are replaced per-document during indexing.
        try db.exec("UPDATE docs SET indexed_at = NULL, index_error = NULL;");
    }

    // Before the scan, so a corrupted index cannot make this run's numbers look
    // consistent while stale rows are still answering queries. Nothing here
    // produces these; a real index accumulated them from an older binary.
    const swept = try s.deleteOrphanChunks(gpa);
    if (swept != 0) try w.print("dropped {d} chunk(s) whose document is gone\n", .{swept});

    try w.print("scanning {s}\n", .{root});
    try w.flush();

    const report = try zkb.scan.reconcile(gpa, io, &s, cid, root, filters, now_ms);
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
    const kb_roots = try otherRoots(arena, io, &s, &layout, cid, now_ms);
    const kb_pending = try reconcileKb(gpa, io, &s, kb_roots, now_ms);
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
    try indexKb(gpa, io, &s, &embedder, kb_roots, now_ms, w);
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

    // `stats` 只覆盖 docs 集合；kb 集合（记忆与 records）的失败由 indexKb 产生，只体现在
    // `c.failed` 里。按 stats 判断会漏掉整整一类——一个列不一致的 csv 就是这样只留下
    // 一个孤零零的计数，而原因明明已经写进 docs.index_error 了。
    if (c.failed != 0) {
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
fn viaDaemon(
    gpa: std.mem.Allocator,
    io: std.Io,
    layout: *const zkb.paths.Layout,
    w: *Writer,
    collection: []const u8,
    reg: Registration,
) !?u8 {
    var c = zkb.ipc_client.Client.connect(io, layout.sock) catch return null;
    defer c.close();

    {
        // The request body used to be the constant `"{}"`, so a root the user
        // named on the command line never crossed the socket: the daemon woke its
        // ingest thread, which walked the roots it already knew, and the command
        // reported success having registered nothing.
        // Allocating rather than a fixed buffer: a shell glob over a directory of
        // projects turns into one include pattern per match, and a legitimate
        // sixty-project root must not hit a buffer limit.
        var bw = std.Io.Writer.Allocating.init(gpa);
        defer bw.deinit();
        const pw = &bw.writer;

        try pw.writeAll("{\"collection\":");
        try std.json.Stringify.value(collection, .{}, pw);
        try pw.writeAll(",\"root\":");
        try std.json.Stringify.value(reg.root, .{}, pw);
        if (reg.extensions) |v| {
            try pw.writeAll(",\"extensions\":");
            try std.json.Stringify.value(v, .{}, pw);
        }
        if (reg.include) |v| {
            try pw.writeAll(",\"include\":");
            try std.json.Stringify.value(v, .{}, pw);
        }
        try pw.writeAll("}");

        var resp = c.call(gpa, .index, bw.written()) catch return null;
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

        const failed = intOf(obj, "failed");
        try w.print("  {d} docs, {d} chunks, {d} failed\n", .{
            intOf(obj, "docs"), intOf(obj, "chunks"), failed,
        });
        // 单机路径会把失败文档和原因一起打印；走 daemon 时只回了计数，于是同一条命令
        // 有没有 daemon 决定你看不看得到解释。原因一直写在 docs.index_error 里，这里
        // 只是把它接回来——一个孤零零的 `3 failed` 会让人以为是自己没建对。
        if (failed != 0) try printFailedDocs(gpa, layout, w);
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

/// Every root except the documents one just handled above.
///
/// Includes what zkb itself writes — `~/.zkb/data/memory/*.md` and
/// `~/.zkb/data/*.csv`, two collections over nested roots told apart only by
/// extension, which is what stops every memory being indexed twice — and also
/// any collection the user registered. Those used to be missing here, so
/// `zkb index` scanned three roots while `zkb status` listed five.
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

fn otherRoots(
    arena: std.mem.Allocator,
    io: std.Io,
    s: *zkb.store.Store,
    layout: *const zkb.paths.Layout,
    docs_cid: i64,
    now_ms: i64,
) ![]KbRoot {
    try zkb.roots.ensureOwn(s, layout, now_ms);
    const rs = try zkb.roots.list(arena, io, s);
    var out: std.ArrayList(KbRoot) = .empty;
    for (rs) |r| {
        if (r.id == docs_cid) continue;
        try out.append(arena, .{
            .path = r.path,
            .name = r.name,
            .kind = r.kind,
            .filters = r.filters,
            .cid = r.id,
        });
    }
    return out.items;
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


/// 失败文档及原因，读自索引。daemon 路径用它补上单机路径本来就有的输出。
fn printFailedDocs(gpa: std.mem.Allocator, layout: *const zkb.paths.Layout, w: *Writer) !void {
    const db_path = gpa.dupeZ(u8, layout.db) catch return;
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch return;
    defer db.close();

    var st = db.prepare(
        "SELECT rel_path, index_error FROM docs WHERE index_error IS NOT NULL LIMIT 20",
    ) catch return;
    defer st.finalize();

    try w.writeAll("\nfailed documents:\n");
    while (st.step() catch false) {
        try w.print("  {s}: {s}\n", .{ st.columnText(0), st.columnText(1) });
    }
}
