//! Drain the pending-doc queue: parse, chunk, embed, write.
//!
//! Ordering that matters: **all vectors for a document are computed before the
//! write transaction opens.** Embedding is tens of milliseconds per chunk; holding
//! a write lock across it would block every reader for the length of a document
//! (SPEC §4.3).
//!
//! Failure granularity is one document. A half-indexed document is worse than an
//! unindexed one — the user would believe it was searched — so on any error its
//! chunks are dropped and `index_error` is set.

const std = @import("std");
const store = @import("../db/store.zig");
const schema = @import("../db/schema.zig");
const markdown = @import("markdown.zig");
const chunk = @import("chunk.zig");
const embed = @import("../embed/llama.zig");
const maintain = @import("../maintain.zig");
const memory = @import("../memory.zig");
const factsmod = @import("../facts.zig");
const recordsmod = @import("../records.zig");
const csvmod = @import("csv.zig");

pub const Stats = struct {
    docs_indexed: usize = 0,
    docs_failed: usize = 0,
    chunks_written: usize = 0,
    embed_calls: usize = 0,
    embed_ns: u64 = 0,
    total_ns: u64 = 0,

    pub fn avgEmbedMs(self: Stats) f64 {
        if (self.embed_calls == 0) return 0;
        return @as(f64, @floatFromInt(self.embed_ns)) /
            @as(f64, @floatFromInt(self.embed_calls)) / std.time.ns_per_ms;
    }
};

pub const Options = struct {
    chunking: chunk.Config = .{},
    /// Stop after this many documents; 0 means no limit.
    limit: usize = 0,
    progress: ?*const fn (done: usize, total: usize, rel_path: []const u8) void = null,
};

/// Whatever can turn text into vectors, so the same pipeline serves both the
/// foreground `zkb index` (embedder called directly) and the daemon (embedder
/// reached through the priority queue). Without this the two paths would be two
/// copies of the parse/chunk/write logic, and they would drift.
pub const Backend = struct {
    ctx: *anyopaque,
    embedDocFn: *const fn (ctx: *anyopaque, heading: []const u8, text: []const u8, out: []f32) anyerror!void,
    countFn: *const fn (ctx: *anyopaque, text: []const u8) anyerror!usize,

    fn embedDoc(self: Backend, heading: []const u8, text: []const u8, out: []f32) anyerror!void {
        return self.embedDocFn(self.ctx, heading, text, out);
    }

    fn counter(self: *const Backend) chunk.TokenCounter {
        return .{ .ctx = @constCast(self), .countFn = countAdapter };
    }

    fn countAdapter(ctx: *anyopaque, text: []const u8) anyerror!usize {
        const self: *Backend = @ptrCast(@alignCast(ctx));
        return self.countFn(self.ctx, text);
    }
};

/// Calls a loaded Embedder directly. Used by the foreground CLI, which owns the
/// model outright and has no other thread to coordinate with.
pub const DirectBackend = struct {
    embedder: *embed.Embedder,

    pub fn backend(self: *DirectBackend) Backend {
        return .{ .ctx = self, .embedDocFn = embedDoc, .countFn = count };
    }

    fn embedDoc(ctx: *anyopaque, heading: []const u8, text: []const u8, out: []f32) anyerror!void {
        const self: *DirectBackend = @ptrCast(@alignCast(ctx));
        _ = try self.embedder.embedDocument(heading, text, out);
    }

    fn count(ctx: *anyopaque, text: []const u8) anyerror!usize {
        const self: *DirectBackend = @ptrCast(@alignCast(ctx));
        return self.embedder.countTokens(text);
    }
};

pub fn indexPending(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
    embedder: *embed.Embedder,
    collection_id: i64,
    root: []const u8,
    now_ms: i64,
    opts: Options,
) !Stats {
    var direct: DirectBackend = .{ .embedder = embedder };
    const backend = direct.backend();
    var stats: Stats = .{};
    const started = nowNs(io);

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const pending = try s.listPending(
        arena_state.allocator(),
        collection_id,
        // Sentinel must survive the trip through SQLite's i64 bind.
        if (opts.limit == 0) std.math.maxInt(i64) else opts.limit,
    );

    const dim: usize = @intCast(schema.embedding_dim);

    for (pending.items, 0..) |doc, n| {
        if (opts.progress) |cb| cb(n + 1, pending.items.len, doc.rel_path);

        indexOne(gpa, io, s, backend, dim, collection_id, root, doc, now_ms, opts, &stats) catch |err| {
            // Drop any partial work, then record why. Never leave a document
            // looking indexed when it is not.
            s.rollback();
            s.deleteChunks(doc.id) catch {};
            var buf: [128]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "{t}", .{err}) catch "unknown error";
            s.markFailed(doc.id, msg) catch {};
            stats.docs_failed += 1;
            continue;
        };
        stats.docs_indexed += 1;
    }

    stats.total_ns = @intCast(nowNs(io) - started);
    return stats;
}

fn indexOne(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
    backend: Backend,
    dim: usize,
    collection_id: i64,
    root: []const u8,
    doc: store.Store.PendingDoc,
    now_ms: i64,
    opts: Options,
    stats: *Stats,
) !void {
    const abs = try std.fs.path.join(gpa, &.{ root, doc.rel_path });
    defer gpa.free(abs);

    const source = try readFile(gpa, io, abs);
    defer gpa.free(source);

    // A .csv in any collection is facts/records, not prose: one row per chunk so
    // structured filtering has something to join against.
    if (std.mem.endsWith(u8, doc.rel_path, ".csv")) {
        return indexCsv(gpa, io, s, backend, dim, collection_id, root, doc, source, now_ms, stats);
    }

    var parsed = try markdown.scan(gpa, source);
    defer parsed.deinit(gpa);

    var chunks = try chunk.split(gpa, source, &parsed, backend.counter(), opts.chunking);
    defer chunks.deinit(gpa);

    // Vectors first, outside any transaction.
    const vectors = try gpa.alloc(f32, chunks.items.len * dim);
    defer gpa.free(vectors);
    for (chunks.items, 0..) |c, i| {
        const slot = vectors[i * dim ..][0..dim];
        const t0 = nowNs(io);
        try backend.embedDoc(c.heading_path, c.text, slot);
        stats.embed_ns += @intCast(nowNs(io) - t0);
        stats.embed_calls += 1;
    }

    // Then a short write transaction.
    try s.begin();
    errdefer s.rollback();

    try s.deleteChunks(doc.id);
    try s.setDocMeta(doc.id, parsed.title, parsed.frontmatter);

    // Links go in the same transaction as the chunks they came from, so the
    // graph can never describe a version of the document that is not stored.
    const links = try markdown.extractLinks(gpa, source, &parsed);
    defer gpa.free(links);
    try maintain.replaceLinks(s.db, doc.id, links);

    var chunk_ids: std.ArrayList(i64) = .empty;
    defer chunk_ids.deinit(gpa);
    for (chunks.items, 0..) |c, i| {
        const cid = try s.insertChunk(collection_id, doc.id, .{
            .idx = @intCast(c.idx),
            .heading_path = c.heading_path,
            .byte_start = @intCast(c.byte_start),
            .byte_end = @intCast(c.byte_end),
            .n_tokens = @intCast(c.n_tokens),
            .text = c.text,
        }, vectors[i * dim ..][0..dim]);
        try chunk_ids.append(gpa, cid);
    }
    // Memory metadata is materialized from frontmatter, in the same transaction
    // as the chunks it describes.
    if ((try s.collectionKind(collection_id)) == .memory and chunk_ids.items.len != 0) {
        var meta = try memory.parseMeta(gpa, parsed.frontmatter);
        defer meta.deinit(gpa);
        try memory.replaceMeta(s.db, doc.id, chunk_ids.items[0], meta);
    }

    try s.markIndexed(doc.id, @intCast(chunks.items.len), now_ms);
    try s.commit();

    stats.chunks_written += chunks.items.len;
}

/// One CSV row becomes one chunk, so a row is retrievable and joinable.
///
/// Only the free-text columns go into the embedding. Numbers and dates are
/// answered by SQL comparison, and putting them in a 1024-dimensional semantic
/// space adds noise without adding an answer (SPEC §16.4).
fn indexCsv(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
    backend: Backend,
    dim: usize,
    collection_id: i64,
    root: []const u8,
    doc: store.Store.PendingDoc,
    source: []const u8,
    now_ms: i64,
    stats: *Stats,
) !void {
    if (recordsmod.typeOf(doc.rel_path)) |type_name| {
        return indexRecords(gpa, io, s, backend, dim, collection_id, root, type_name, doc, source, now_ms, stats);
    }
    if (!std.mem.endsWith(u8, doc.rel_path, "facts.csv")) {
        // A csv that is neither facts nor under records/ has no declared shape.
        // Recorded as an error rather than indexed as prose: guessing would put
        // rows in the semantic space that nothing can ever usefully retrieve.
        try s.begin();
        errdefer s.rollback();
        try s.deleteChunks(doc.id);
        try s.markFailed(doc.id, "csv outside records/ and not facts.csv");
        try s.commit();
        return;
    }

    var parsed = try factsmod.parse(gpa, source);
    defer parsed.deinit(gpa);

    const vectors = try gpa.alloc(f32, parsed.facts.len * dim);
    defer gpa.free(vectors);
    var texts: std.ArrayList([]u8) = .empty;
    defer {
        for (texts.items) |t| gpa.free(t);
        texts.deinit(gpa);
    }
    for (parsed.facts, 0..) |f, i| {
        const text = try factsmod.renderForEmbedding(gpa, f);
        try texts.append(gpa, text);
        try backend.embedDoc("", text, vectors[i * dim ..][0..dim]);
        stats.embed_calls += 1;
    }

    try s.begin();
    errdefer s.rollback();
    try s.deleteChunks(doc.id);

    var chunk_ids: std.ArrayList(i64) = .empty;
    defer chunk_ids.deinit(gpa);
    for (parsed.facts, 0..) |f, i| {
        const id = try s.insertChunk(collection_id, doc.id, .{
            .idx = @intCast(i),
            .heading_path = f.key,
            .byte_start = @intCast(i),
            .byte_end = @intCast(i + 1),
            .n_tokens = 16,
            .text = texts.items[i],
        }, vectors[i * dim ..][0..dim]);
        try chunk_ids.append(gpa, id);
    }
    try factsmod.replaceFor(s.db, doc.id, chunk_ids.items, parsed.facts);

    // A row a person broke while editing in a spreadsheet is recorded as an
    // error rather than skipped: silently dropping it loses data quietly.
    if (parsed.bad_rows.len != 0) {
        var buf: [128]u8 = undefined;
        const msg = try std.fmt.bufPrint(&buf, "{d} malformed row(s), first at line {d}", .{
            parsed.bad_rows.len, parsed.bad_rows[0],
        });
        try s.markFailed(doc.id, msg);
    } else {
        try s.markIndexed(doc.id, @intCast(parsed.facts.len), now_ms);
    }
    try s.commit();
    stats.chunks_written += parsed.facts.len;
}

/// A `records/` csv: infer the schema, rebuild the materialized table if the
/// header changed, then write one chunk plus one materialized row per csv row.
///
/// The schema is inferred per *file* but stored per *type*. Two files of one
/// type that disagree on their header is a real error worth reporting — the
/// alternative is one of them silently determining the table shape and the other
/// losing columns, which nothing downstream could detect.
fn indexRecords(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
    backend: Backend,
    dim: usize,
    collection_id: i64,
    root: []const u8,
    type_name: []const u8,
    doc: store.Store.PendingDoc,
    source: []const u8,
    now_ms: i64,
    stats: *Stats,
) !void {
    var table = try csvmod.parse(gpa, source);
    defer table.deinit(gpa);

    if (table.header.len == 0) {
        try s.begin();
        errdefer s.rollback();
        try s.deleteChunks(doc.id);
        try s.markFailed(doc.id, "no header row");
        try s.commit();
        return;
    }

    // Inference reads every file of the type, not just this one: the schema is a
    // property of the type, and making it depend on which file was indexed last
    // would make the result depend on scan order.
    var sample = try recordsmod.collectSample(gpa, io, root, type_name);
    defer sample.deinit(gpa);

    if (sample.mismatch) |other| {
        try s.begin();
        errdefer s.rollback();
        try s.deleteChunks(doc.id);
        var buf: [256]u8 = undefined;
        const msg = try std.fmt.bufPrint(
            &buf,
            "header differs from other files of type '{s}' (e.g. {s})",
            .{ type_name, other },
        );
        try s.markFailed(doc.id, msg);
        try s.commit();
        return;
    }

    var rec_schema = try recordsmod.infer(gpa, type_name, &sample.table);
    defer rec_schema.deinit(gpa);

    // `_schema.json` sits beside the data, so it travels with the files it
    // describes and is versioned with them.
    if (try readSchemaOverride(gpa, io, root, type_name)) |override| {
        defer gpa.free(override);
        try recordsmod.applyOverrides(gpa, &rec_schema, override);
    }

    const vectors = try gpa.alloc(f32, table.rows.len * dim);
    defer gpa.free(vectors);
    var texts: std.ArrayList([]u8) = .empty;
    defer {
        for (texts.items) |t| gpa.free(t);
        texts.deinit(gpa);
    }
    var labels: std.ArrayList([]u8) = .empty;
    defer {
        for (labels.items) |t| gpa.free(t);
        labels.deinit(gpa);
    }

    // Rows whose rendered text is unchanged keep the vector already on disk.
    // Appending one row to a csv would otherwise re-embed every row in it.
    var cache = recordsmod.loadVectors(gpa, s.db, doc.id, dim) catch recordsmod.VectorCache{};
    defer cache.deinit(gpa);

    for (table.rows, 0..) |row, i| {
        const text = try recordsmod.renderForEmbedding(gpa, &rec_schema, row);
        try texts.append(gpa, text);
        try labels.append(gpa, try recordsmod.rowLabel(gpa, &rec_schema, row));

        const slot = vectors[i * dim ..][0..dim];
        if (cache.get(text)) |cached| {
            @memcpy(slot, cached);
            continue;
        }
        const t0 = nowNs(io);
        try backend.embedDoc("", text, slot);
        stats.embed_ns += @intCast(nowNs(io) - t0);
        stats.embed_calls += 1;
    }

    try s.begin();
    errdefer s.rollback();

    try recordsmod.ensureTable(gpa, s.db, &rec_schema);
    try s.deleteChunks(doc.id);

    var chunk_ids: std.ArrayList(i64) = .empty;
    defer chunk_ids.deinit(gpa);
    for (table.rows, 0..) |_, i| {
        const id = try s.insertChunk(collection_id, doc.id, .{
            .idx = @intCast(i),
            .heading_path = labels.items[i],
            .byte_start = @intCast(i),
            .byte_end = @intCast(i + 1),
            .n_tokens = 24,
            .text = texts.items[i],
        }, vectors[i * dim ..][0..dim]);
        try chunk_ids.append(gpa, id);
    }
    try recordsmod.replaceFor(gpa, s.db, &rec_schema, doc.id, chunk_ids.items, &table);

    if (table.bad_rows.len != 0) {
        var buf: [128]u8 = undefined;
        const msg = try std.fmt.bufPrint(&buf, "{d} malformed row(s), first at line {d}", .{
            table.bad_rows.len, table.bad_rows[0],
        });
        try s.markFailed(doc.id, msg);
    } else {
        try s.markIndexed(doc.id, @intCast(table.rows.len), now_ms);
    }
    try s.commit();
    stats.chunks_written += table.rows.len;
}

/// `records/<type>/_schema.json`, or null when there is none — which is the
/// normal case. The override file exists for the columns inference gets wrong,
/// not as a required declaration.
fn readSchemaOverride(
    gpa: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    type_name: []const u8,
) !?[]u8 {
    const path = try std.fmt.allocPrint(gpa, "{s}/records/{s}/_schema.json", .{ root, type_name });
    defer gpa.free(path);
    return readFile(gpa, io, path) catch null;
}

/// Monotonic nanoseconds. Zig 0.16 has no ambient Timer: clocks come from Io,
/// and `.awake` is the monotonic one (excludes time the machine was suspended).
fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

fn readFile(gpa: std.mem.Allocator, io: std.Io, abs: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFileAbsolute(io, abs, .{});
    defer file.close(io);
    const st = try file.stat(io);
    const buf = try gpa.alloc(u8, @intCast(st.size));
    errdefer gpa.free(buf);
    var read_buf: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    try reader.interface.readSliceAll(buf);
    return buf;
}

// ---------------------------------------------------------------------------
// daemon path
// ---------------------------------------------------------------------------

/// Index one document from inside the daemon, reaching the embedder through the
/// priority queue so an interactive query can preempt between chunks.
///
/// `daemon_state` is `*daemon.State`, passed as anyopaque to avoid an import
/// cycle (daemon imports indexer for exactly this function).
pub fn indexOneQueued(
    daemon_state: anytype,
    s: *store.Store,
    collection_id: i64,
    root: []const u8,
    doc: store.Store.PendingDoc,
    now_ms: i64,
) !void {
    const St = @TypeOf(daemon_state);
    const QueuedBackend = struct {
        st: St,

        fn backend(self: *@This()) Backend {
            return .{ .ctx = self, .embedDocFn = embedDoc, .countFn = count };
        }

        fn embedDoc(ctx: *anyopaque, heading: []const u8, text: []const u8, out: []f32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            // Per-chunk, not per-document. Checking only before each document
            // meant a 60-chunk file held the embedder for ~18s regardless of who
            // was waiting — the backoff has to be at the granularity of the thing
            // that actually blocks.
            self.st.waitWhileInteractive();
            return self.st.embedText(.ingest, .document, heading, text, out);
        }

        fn count(ctx: *anyopaque, text: []const u8) anyerror!usize {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return self.st.countTokens(.ingest, text);
        }
    };

    var qb: QueuedBackend = .{ .st = daemon_state };
    const dim: usize = @intCast(schema.embedding_dim);
    var stats: Stats = .{};
    return indexOne(
        daemon_state.gpa,
        daemon_state.io,
        s,
        qb.backend(),
        dim,
        collection_id,
        root,
        doc,
        now_ms,
        .{},
        &stats,
    );
}
