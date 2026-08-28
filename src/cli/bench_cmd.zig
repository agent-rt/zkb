//! `zkb bench` — run a fixture of queries through every retrieval path and
//! report what each one recalls.
//!
//! Deliberately in-process, never through the daemon. Everywhere else the CLI
//! prefers the daemon because its model is already resident; here that would
//! measure a daemon of unknown vintage rather than this checkout, and a
//! benchmark whose subject depends on whether a background process happens to
//! be running is not a benchmark. The model is loaded once and every query is
//! embedded once, so the usual reason to want the daemon does not apply.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub const Options = struct {
    /// Path to the fixture csv. See `zkb.bench.parseFixture` for the columns.
    fixture: []const u8,
    collection: ?[]const u8 = null,
    path: ?[]const u8 = null,
    /// Result depth **in chunks** — the same `-k` `zkb search` takes, so the
    /// report describes what that command would have shown.
    ///
    /// Not documents: one document may supply up to `search_max_per_doc`
    /// chunks, so `-k 10` is a median of six documents on a real corpus and
    /// reaches ten in well under one per cent of queries. The `docs/q` column
    /// exists so `R@10` is not read as "within ten documents".
    top_k: usize = 10,
    model: ?[]const u8 = null,
    json: bool = false,
};

/// Every path a query can take, in the order the report prints them. Fixed
/// rather than selectable: the value of the report is the comparison between
/// them, and a run that quietly omits one is not comparable with a run that
/// did not.
const modes = [_]zkb.hybrid.Mode{ .keyword, .vector, .hybrid };

const CaseResult = struct {
    metrics: zkb.bench.Metrics,
    /// Expected documents that did not come back, borrowed from the fixture.
    missed: [][]const u8,
    docs_returned: usize,
};

const ModeRun = struct {
    mode: zkb.hybrid.Mode,
    results: []CaseResult,
    overall: zkb.bench.Metrics,
    ms_total: i64,
    /// Documents the caller actually saw, summed over cases.
    ///
    /// `-k` counts chunks, and `search_max_per_doc` lets one document supply
    /// three of them, so ten chunks are a median of six documents on this
    /// corpus and reach ten in 0.6% of queries. Without this column `R@10`
    /// reads as "within ten documents", which is not what was measured.
    docs_total: usize,
    /// Set when the mode could not run as asked — no model for the vector
    /// paths. Reported instead of quietly scoring zeros, which would read as a
    /// retrieval failure rather than a missing prerequisite.
    unavailable: bool = false,

    fn deinit(self: *ModeRun, gpa: std.mem.Allocator) void {
        for (self.results) |r| gpa.free(r.missed);
        gpa.free(self.results);
    }
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

    const src = readFixture(gpa, io, opts.fixture) catch |err| switch (err) {
        error.FileNotFound => {
            try w.print("no such fixture: {s}\n", .{opts.fixture});
            return 3;
        },
        else => return err,
    };
    defer gpa.free(src);

    var fixture = zkb.bench.parseFixture(gpa, src) catch |err| switch (err) {
        error.MissingColumn => {
            try w.writeAll("fixture needs a header with at least: query,expected\n");
            try w.writeAll("optional columns: id, kind\n");
            return 2;
        },
        error.UnterminatedQuote => {
            try w.print("{s}: a quoted field is never closed\n", .{opts.fixture});
            return 2;
        },
        else => |e| return e,
    };
    defer fixture.deinit(gpa);

    if (fixture.cases.len == 0) {
        try w.print("{s}: no cases\n", .{opts.fixture});
        return 3;
    }

    std.Io.Dir.accessAbsolute(io, layout.db, .{}) catch {
        try w.print("no index at {s}\nrun: zkb index\n", .{layout.db});
        return 3;
    };
    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch |err| switch (err) {
        error.SchemaStale => {
            try w.writeAll("index schema is out of date\nrun: zkb index\n");
            return 3;
        },
        error.SchemaFromFuture => {
            try w.writeAll("index was written by a newer zkb\nupgrade zkb, or: rm ~/.zkb/index/zkb.db && zkb index\n");
            return 3;
        },
        else => return err,
    };
    defer db.close();

    var collection_id: ?i64 = null;
    if (opts.collection) |name| {
        var st = try db.prepare("SELECT id FROM collections WHERE name = ?1");
        defer st.finalize();
        try st.bindText(1, name);
        if (!try st.step()) {
            try w.print("unknown collection: {s}\n", .{name});
            return 2;
        }
        collection_id = st.columnI64(0);
    }

    // ---- embed every query once, up front.
    //
    // Both vector modes need the same vectors, and the model costs 1-2s to
    // load. Doing it here also keeps the per-query timings below about
    // retrieval rather than about who paid for the model.
    var vectors: []?[]f32 = try gpa.alloc(?[]f32, fixture.cases.len);
    defer {
        for (vectors) |v| if (v) |vec| gpa.free(vec);
        gpa.free(vectors);
    }
    @memset(vectors, null);

    var have_model = false;
    {
        const found = zkb.model_registry.resolve(gpa, io, env, &layout, opts.model, .q8_0) catch null;
        defer if (found) |f| f.deinit(gpa);
        if (found) |f| {
            var embedder = try zkb.embed.Embedder.init(gpa, f.path, .{});
            defer embedder.deinit();
            for (fixture.cases, 0..) |c, i| {
                const vec = try gpa.alloc(f32, embedder.n_embd);
                errdefer gpa.free(vec);
                _ = try embedder.embedQuery(zkb.embed.default_query_task, c.query, vec);
                vectors[i] = vec;
            }
            have_model = true;
        }
    }

    // ---- run every mode over every case.
    var runs: [modes.len]ModeRun = undefined;
    var built: usize = 0;
    defer for (runs[0..built]) |*r| r.deinit(gpa);

    for (modes, 0..) |mode, mi| {
        const needs_vec = mode != .keyword;
        var results = try gpa.alloc(CaseResult, fixture.cases.len);
        errdefer gpa.free(results);
        var acc: zkb.bench.Accumulator = .{};
        var ms_total: i64 = 0;

        if (needs_vec and !have_model) {
            for (results) |*r| r.* = .{ .metrics = .zero, .missed = &.{}, .docs_returned = 0 };
            runs[mi] = .{
                .mode = mode,
                .results = results,
                .overall = .zero,
                .ms_total = 0,
                .docs_total = 0,
                .unavailable = true,
            };
            built += 1;
            continue;
        }

        var filled: usize = 0;
        errdefer for (results[0..filled]) |r| gpa.free(r.missed);

        for (fixture.cases, 0..) |c, i| {
            const t0 = nowMs(io);
            var res = try zkb.hybrid.search(gpa, &db, mode, c.query, vectors[i], collection_id, .{
                .path = opts.path,
                .top_k = opts.top_k,
                // The same ceiling `zkb search` applies, so the numbers describe
                // that command rather than a configuration nobody runs.
                .max_per_doc = zkb.hybrid.search_max_per_doc,
            });
            defer res.deinit(gpa);
            ms_total += nowMs(io) - t0;

            // Hits are chunks; a fixture names documents. Collapse to the
            // document ranking a reader would see, first occurrence winning.
            var docs: std.ArrayList([]const u8) = .empty;
            defer {
                for (docs.items) |d| gpa.free(d);
                docs.deinit(gpa);
            }
            for (res.hits) |h| {
                const display = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ h.collection, h.rel_path });
                errdefer gpa.free(display);
                var seen = false;
                for (docs.items) |d| {
                    if (std.mem.eql(u8, d, display)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) {
                    gpa.free(display);
                } else {
                    try docs.append(gpa, display);
                }
            }

            const m = zkb.bench.score(docs.items, c.expected, opts.top_k);
            acc.add(m);
            results[i] = .{
                .metrics = m,
                .missed = try zkb.bench.misses(gpa, docs.items, c.expected, opts.top_k),
                .docs_returned = docs.items.len,
            };
            filled += 1;
        }

        var docs_total: usize = 0;
        for (results) |r| docs_total += r.docs_returned;
        runs[mi] = .{
            .mode = mode,
            .results = results,
            .overall = acc.mean(),
            .ms_total = ms_total,
            .docs_total = docs_total,
        };
        built += 1;
    }

    var s = zkb.store.Store.init(&db);
    const counts = try s.counts();

    if (opts.json) {
        try printJson(w, &fixture, runs[0..built], counts, opts);
        return 0;
    }
    try printReport(gpa, w, &fixture, runs[0..built], counts, opts, have_model);
    return 0;
}

fn nowMs(io: std.Io) i64 {
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).nanoseconds, std.time.ns_per_ms));
}

fn readFixture(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = if (std.fs.path.isAbsolute(path))
        try std.Io.Dir.openFileAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    const buf = try gpa.alloc(u8, @intCast(size));
    errdefer gpa.free(buf);
    var rbuf: [4096]u8 = undefined;
    var reader = file.reader(io, &rbuf);
    try reader.interface.readSliceAll(buf);
    return buf;
}

fn printReport(
    gpa: std.mem.Allocator,
    w: *Writer,
    fixture: *const zkb.bench.Fixture,
    runs: []const ModeRun,
    counts: zkb.store.Store.Counts,
    opts: Options,
    have_model: bool,
) !void {
    try w.print("fixture: {s}  ({d} case(s)", .{ opts.fixture, fixture.cases.len });
    if (opts.collection) |c| try w.print(", collection {s}", .{c});
    if (opts.path) |p| try w.print(", path {s}", .{p});
    try w.print(")\nindex:   {d} docs / {d} chunks", .{ counts.docs, counts.chunks });
    if (counts.pending != 0 or counts.failed != 0) {
        // Scores measured against a half-built index are not comparable with
        // anything, so this is a warning about the numbers, not about the index.
        try w.print("  — INCOMPLETE: {d} pending, {d} failed", .{ counts.pending, counts.failed });
    }
    try w.writeAll("\n");
    if (fixture.bad_rows.len != 0) {
        try w.print("fixture has {d} malformed row(s), not run:", .{fixture.bad_rows.len});
        for (fixture.bad_rows) |line| try w.print(" line {d}", .{line});
        try w.writeAll("\n");
    }
    if (!have_model) {
        try w.writeAll("model unavailable — the vector and hybrid rows could not run (zkb model pull)\n");
    }
    try w.writeAll("\n");

    var bucket_buf: [4]usize = undefined;
    const buckets = recallBuckets(&bucket_buf, opts.top_k);

    try w.writeAll("mode    ");
    for (buckets) |b| {
        var lbl_buf: [8]u8 = undefined;
        const lbl = try std.fmt.bufPrint(&lbl_buf, "R@{d}", .{b});
        try w.print(" {s:>6}", .{lbl});
    }
    try w.writeAll("    MRR  docs/q   ms/q\n");

    for (runs) |r| {
        try w.print("{t:<8}", .{r.mode});
        if (r.unavailable) {
            for (buckets) |_| try w.writeAll("      —");
            try w.writeAll("      —      —      —\n");
            continue;
        }
        const m = r.overall;
        for (buckets) |b| try w.print(" {d:>6.3}", .{recallAt(m, b, opts.top_k)});
        const n: f64 = @floatFromInt(@max(1, fixture.cases.len));
        const per_q: u64 = @intCast(@max(0, @divTrunc(r.ms_total, @as(i64, @intCast(@max(1, fixture.cases.len))))));
        const docs_per_q = @as(f64, @floatFromInt(r.docs_total)) / n;
        try w.print(" {d:>6.3} {d:>6.1} {d:>6}\n", .{ m.mrr, docs_per_q, per_q });
    }

    // Per-kind breakdown for the strongest mode that ran. A total that moved
    // says nothing about which kind of query moved it.
    try printByKind(gpa, w, fixture, runs, opts);
    // The miss list stays single-mode: it is a list of documents, and three
    // interleaved copies of it would not read. `reportRun` says which.
    if (reportRun(runs)) |r| try printMisses(w, fixture, r, opts);
}

/// The recall depths worth printing for a given `k`: the fixed 1/3/5 ladder,
/// keeping only the rungs below `k`, then `k` itself. Without the filter a
/// `-k 3` run prints an `R@3` column twice with identical numbers, which reads
/// as a bug in the tool rather than as an artifact of the ladder.
fn recallBuckets(buf: *[4]usize, k: usize) []const usize {
    var n: usize = 0;
    for ([_]usize{ 1, 3, 5 }) |b| {
        if (b < k) {
            buf[n] = b;
            n += 1;
        }
    }
    buf[n] = k;
    return buf[0 .. n + 1];
}

fn recallAt(m: zkb.bench.Metrics, bucket: usize, k: usize) f64 {
    if (bucket == k) return m.recall_at_k;
    return switch (bucket) {
        1 => m.recall_at_1,
        3 => m.recall_at_3,
        else => m.recall_at_5,
    };
}

/// Which mode the per-kind breakdown and the miss list describe.
///
/// Always `hybrid` when it ran, because that is what `zkb search` does by
/// default and the point of these two blocks is to explain the command people
/// actually run. Picking the highest-scoring mode instead looks reasonable and
/// is not: modes tie often on a small fixture, so the block would silently
/// change which path it is describing between runs — and a miss list that
/// belongs to a different mode than the reader assumes is worse than none.
fn reportRun(runs: []const ModeRun) ?*const ModeRun {
    var fallback: ?*const ModeRun = null;
    for (runs) |*r| {
        if (r.unavailable) continue;
        if (r.mode == .hybrid) return r;
        fallback = r;
    }
    return fallback;
}

/// Every mode, broken down by kind.
///
/// One mode is not enough. The first question anyone asks of the summary table
/// is *which* kind of query a path is losing — "is keyword already enough when
/// I know the exact wording?" — and answering that from a single mode's
/// breakdown is impossible: it needs the same kind across paths. Printing only
/// the production mode sent the first real user of this report to `--json` and
/// a python one-liner.
fn printByKind(
    gpa: std.mem.Allocator,
    w: *Writer,
    fixture: *const zkb.bench.Fixture,
    runs: []const ModeRun,
    opts: Options,
) !void {
    var kinds: std.ArrayList([]const u8) = .empty;
    defer kinds.deinit(gpa);
    for (fixture.cases) |c| {
        if (c.kind.len == 0) continue;
        var seen = false;
        for (kinds.items) |k| {
            if (std.mem.eql(u8, k, c.kind)) {
                seen = true;
                break;
            }
        }
        if (!seen) try kinds.append(gpa, c.kind);
    }
    if (kinds.items.len == 0) return;

    var bucket_buf: [4]usize = undefined;
    const buckets = recallBuckets(&bucket_buf, opts.top_k);

    try w.writeAll("\nby kind\n");
    try w.writeAll("                    ");
    for (buckets) |b| {
        var lbl_buf: [8]u8 = undefined;
        const lbl = try std.fmt.bufPrint(&lbl_buf, "R@{d}", .{b});
        try w.print(" {s:>6}", .{lbl});
    }
    try w.writeAll("    MRR\n");

    for (kinds.items) |k| {
        var label_buf: [64]u8 = undefined;
        var n: usize = 0;
        for (fixture.cases) |c| {
            if (std.mem.eql(u8, c.kind, k)) n += 1;
        }
        const label = std.fmt.bufPrint(&label_buf, "{s} ({d})", .{ k, n }) catch k;
        try w.print("  {s}\n", .{label});

        for (runs) |r| {
            if (r.unavailable) continue;
            var acc: zkb.bench.Accumulator = .{};
            for (fixture.cases, r.results) |c, cr| {
                if (std.mem.eql(u8, c.kind, k)) acc.add(cr.metrics);
            }
            const m = acc.mean();
            try w.print("    {t:<16}", .{r.mode});
            for (buckets) |b| try w.print(" {d:>6.3}", .{recallAt(m, b, opts.top_k)});
            try w.print(" {d:>6.3}\n", .{m.mrr});
        }
    }
}

fn printMisses(
    w: *Writer,
    fixture: *const zkb.bench.Fixture,
    run_: *const ModeRun,
    opts: Options,
) !void {
    var any = false;
    for (fixture.cases, run_.results) |c, r| {
        if (r.missed.len == 0) continue;
        if (!any) {
            try w.print("\nmisses ({t}, within {d})\n", .{ run_.mode, opts.top_k });
            any = true;
        }
        try w.print("  {s}  \"{s}\"\n", .{ c.id, c.query });
        for (r.missed) |m| try w.print("      {s}\n", .{m});
    }
}

fn printJson(
    w: *Writer,
    fixture: *const zkb.bench.Fixture,
    runs: []const ModeRun,
    counts: zkb.store.Store.Counts,
    opts: Options,
) !void {
    try w.writeAll("{\"fixture\":");
    try std.json.Stringify.value(opts.fixture, .{}, w);
    try w.print(",\"cases\":{d},\"top_k\":{d},", .{ fixture.cases.len, opts.top_k });
    try w.print("\"index\":{{\"docs\":{d},\"chunks\":{d},\"pending\":{d},\"failed\":{d},\"complete\":{}}},", .{
        counts.docs,                                counts.chunks, counts.pending, counts.failed,
        counts.pending == 0 and counts.failed == 0,
    });
    try w.writeAll("\"modes\":[");
    for (runs, 0..) |r, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{{\"mode\":\"{t}\",\"available\":{},", .{ r.mode, !r.unavailable });
        const m = r.overall;
        try w.print("\"recall_at_1\":{d:.6},\"recall_at_3\":{d:.6},\"recall_at_5\":{d:.6},", .{
            m.recall_at_1, m.recall_at_3, m.recall_at_5,
        });
        try w.print("\"recall_at_k\":{d:.6},\"mrr\":{d:.6},\"ms_total\":{d},\"docs_total\":{d},", .{
            m.recall_at_k, m.mrr, r.ms_total, r.docs_total,
        });
        try w.writeAll("\"cases\":[");
        for (fixture.cases, r.results, 0..) |c, cr, j| {
            if (j != 0) try w.writeAll(",");
            try w.writeAll("{\"id\":");
            try std.json.Stringify.value(c.id, .{}, w);
            try w.writeAll(",\"kind\":");
            try std.json.Stringify.value(c.kind, .{}, w);
            try w.writeAll(",\"query\":");
            try std.json.Stringify.value(c.query, .{}, w);
            try w.print(",\"docs_returned\":{d},\"found\":{d},\"expected\":{d},", .{
                cr.docs_returned, cr.metrics.found, cr.metrics.expected,
            });
            try w.print("\"recall_at_k\":{d:.6},\"mrr\":{d:.6},\"missed\":[", .{
                cr.metrics.recall_at_k, cr.metrics.mrr,
            });
            for (cr.missed, 0..) |miss, n| {
                if (n != 0) try w.writeAll(",");
                try std.json.Stringify.value(miss, .{}, w);
            }
            try w.writeAll("]}");
        }
        try w.writeAll("]}");
    }
    try w.writeAll("]}\n");
}
