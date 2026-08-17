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
