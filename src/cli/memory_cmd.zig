//! `zkb remember | forget | recall | facts | remember-fact`
//!
//! zkb is the only writer of the memory root. That is a deliberately narrowed
//! exception to "the filesystem is the user's": it writes human-readable
//! markdown and csv into a directory under version control, never a private
//! format, and never into the documents collection (SPEC §15.4).
//!
//! Why an API instead of letting the agent write the file itself: naming
//! collisions and frontmatter drift, but mostly that **the duplicate check has
//! to be on the write path**. Expecting each session's agent to search before
//! writing is not a mechanism, it is a hope.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

// ---------------------------------------------------------------------------
// remember
// ---------------------------------------------------------------------------

pub const RememberOptions = struct {
    body: []const u8,
    type: zkb.memory.Type = .feedback,
    subjects: []const u8 = "",
    refs: []const u8 = "",
    source: []const u8 = "claude-code",
    /// Record anyway. The agent's explicit override *after* seeing candidates —
    /// which is why the duplicate report exits non-zero rather than prompting.
    force: bool = false,
    /// Empty means universal. See memory.Meta.scope.
    scope: []const u8 = "",
    model: ?[]const u8 = null,
};

pub fn remember(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    opts: RememberOptions,
) !u8 {
    if (std.mem.trim(u8, opts.body, " \t\r\n").len == 0) {
        try w.writeAll("nothing to remember (empty body)\n");
        return 2;
    }

    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);
    try layout.ensureDirs(io);
    try ensureDir(io, layout.memory);

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = try zkb.store.open(db_path, .read_write);
    defer db.close();
    var s = zkb.store.Store.init(&db);

    const now_ms = nowMs(io);
    const cid = try s.ensureCollectionKind("memory", layout.memory, .memory, now_ms);

    const found = zkb.model_registry.resolve(gpa, io, env, &layout, opts.model, .q8_0) catch null;
    defer if (found) |f| f.deinit(gpa);

    var embedder: ?zkb.embed.Embedder = null;
    defer if (embedder) |*e| e.deinit();
    if (found) |f| embedder = zkb.embed.Embedder.init(gpa, f.path, .{}) catch null;

    // ---- duplicate check, before anything is written
    if (embedder) |*e| {
        const vec = try gpa.alloc(f32, e.n_embd);
        defer gpa.free(vec);
        // Document-side embedding, matching how the stored memories were
        // embedded. Using the query prefix here would compare across two
        // different distributions and the cosines would not mean what they say.
        _ = try e.embedDocument("", opts.body, vec);

        const similar = try zkb.memory.findSimilar(gpa, &db, cid, vec, 5);
        defer {
            for (similar) |c| c.deinit(gpa);
            gpa.free(similar);
        }
        if (!opts.force and similar.len != 0 and similar[0].cos >= zkb.memory.default_dup_threshold) {
            try w.writeAll("possible duplicate; not recorded\n\n");
            for (similar) |c| {
                if (c.cos < zkb.memory.default_dup_threshold) continue;
                try w.print("  {s}  (cos {d:.3})\n    ", .{ c.rel_path, c.cos });
                try writeOneLine(w, c.excerpt);
                try w.writeAll("\n");
            }
            // The decision belongs to the agent: update that memory, or assert
            // this really is a new fact. zkb supplies the evidence and stops.
            try w.writeAll("\nedit the existing file, or re-run with --force\n");
            return 5;
        }
    } else {
        try w.writeAll("note: model unavailable, duplicate check skipped\n");
    }

    // ---- write the file
    var ts_buf: [32]u8 = undefined;
    const created = try isoDate(&ts_buf, io);

    const meta: zkb.memory.Meta = .{
        .type = opts.type,
        .status = .active,
        .created = created,
        .source = opts.source,
        .subjects = opts.subjects,
        .refs = opts.refs,
        .scope = opts.scope,
    };
    const content = try zkb.memory.render(gpa, meta, opts.body);
    defer gpa.free(content);

    const stem = try zkb.memory.slug(gpa, opts.body, created);
    defer gpa.free(stem);

    const path = try uniquePath(gpa, io, layout.memory, stem);
    defer gpa.free(path);
    try writeFile(io, path, content);

    try w.print("remembered: {s}\n", .{path});

    // Indexed now rather than at the next scan. An agent commonly writes a
    // memory and recalls within the same session, and "I just saved that and it
    // is not there" is how a tool stops being trusted.
    if (embedder) |*e| {
        _ = try zkb.scan.reconcile(gpa, io, &s, cid, layout.memory, zkb.memory.scan_filters, now_ms);
        const stats = try zkb.indexer.indexPending(gpa, io, &s, e, cid, layout.memory, now_ms, .{});
        if (stats.docs_failed != 0) {
            try w.writeAll("WARNING: written, but indexing failed — run: zkb index\n");
            return 1;
        }
    } else {
        try w.writeAll("(will be indexed on the next daemon scan)\n");
    }

    try warnIfUnversioned(io, w, layout.data);
    return 0;
}

// ---------------------------------------------------------------------------
// forget
// ---------------------------------------------------------------------------

pub fn forget(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    rel_path: []const u8,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    const src = try std.fs.path.join(gpa, &.{ layout.memory, rel_path });
    defer gpa.free(src);
    std.Io.Dir.accessAbsolute(io, src, .{}) catch {
        try w.print("no such memory: {s}\n", .{src});
        return 3;
    };

    const archive_dir = try std.fmt.allocPrint(gpa, "{s}/archive", .{layout.memory});
    defer gpa.free(archive_dir);
    try ensureDir(io, archive_dir);

    const dst = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ archive_dir, std.fs.path.basename(src) });
    defer gpa.free(dst);

    // Status flipped in the file, then moved — never deleted. An automatic,
    // unrecoverable delete of a memory would cost the trust of the whole system;
    // a real deletion is a person deciding to, with jj.
    const content = try readAll(gpa, io, src);
    defer gpa.free(content);
    const updated = try setStatusArchived(gpa, content);
    defer gpa.free(updated);

    try writeFile(io, dst, updated);
    try std.Io.Dir.deleteFileAbsolute(io, src);

    // The index still holds the old chunks. `archive/` is excluded from the
    // memory scan, so this reconcile sees the file as deleted and cascades its
    // chunks, FTS rows and vectors out — which is what makes forgetting stick.
    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    if (zkb.store.open(db_path, .read_write)) |opened| {
        var db = opened;
        defer db.close();
        var s = zkb.store.Store.init(&db);
        if (try s.findCollection("memory")) |cid| {
            _ = try zkb.scan.reconcile(gpa, io, &s, cid, layout.memory, zkb.memory.scan_filters, nowMs(io));
        }
    } else |_| {}

    try w.print("archived: {s}\n", .{dst});
    try w.writeAll("(kept on disk — use jj if you really want it gone)\n");
    try warnIfUnversioned(io, w, layout.data);
    return 0;
}

fn setStatusArchived(gpa: std.mem.Allocator, content: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var lines = std.mem.splitScalar(u8, content, '\n');
    var replaced = false;
    var first = true;
    while (lines.next()) |line| {
        if (!first) try out.append(gpa, '\n');
        first = false;
        if (!replaced and std.mem.startsWith(u8, std.mem.trimStart(u8, line, " \t"), "status:")) {
            try out.appendSlice(gpa, "status: archived");
            replaced = true;
        } else try out.appendSlice(gpa, line);
    }
    return out.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------------------
// recall
// ---------------------------------------------------------------------------

pub const Format = enum { markdown, json };

pub const RecallOptions = struct {
    /// Empty means "session start": no question yet, so ranking is recency only.
    query: []const u8 = "",
    /// Smaller than `query`'s 8000 on purpose. Recall is injected into every
    /// session; a large one crowds out the actual task (SPEC §15.5).
    budget: usize = 1500,
    candidates: usize = 20,
    recency_depth: usize = 20,
    format: Format = .markdown,
    /// Which scope this session is in. Null means universal only.
    ///
    /// Passed in rather than inferred from the working directory: guessing would
    /// bake one person's layout into the tool, and guessing wrong would leak
    /// exactly what the scope exists to contain.
    scope: ?[]const u8 = null,
    model: ?[]const u8 = null,
};

pub fn recall(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    opts: RecallOptions,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    if (try recallViaDaemon(gpa, io, layout.sock, w, opts)) |code| return code;

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch |err| switch (err) {
        error.SchemaStale => {
            try w.writeAll("index schema is out of date\nrun: zkb index\n");
            return 3;
        },
        else => {
            try w.print("no index at {s}\nrun: zkb index\n", .{layout.db});
            return 3;
        },
    };
    defer db.close();

    // ---- 1. facts snapshot, always
    //
    // The most important line in the memory system: numeric facts are
    // *injected*, never retrieved (SPEC §15.5, §16.5). Read from the csv, so it
    // works with no model loaded and no index built.
    const all_current = zkb.facts.currentAll(gpa, io, layout.facts) catch &.{};
    // Filtered before rendering, not inside the renderer: `zkb facts` shows
    // everything on purpose (you asked for it), while recall is injected without
    // anyone asking — which is the whole reason a scope exists.
    const current = blk: {
        var keep: std.ArrayList(zkb.facts.Current) = .empty;
        for (all_current) |c| {
            if (c.inScope(opts.scope)) try keep.append(gpa, c);
        }
        break :blk try keep.toOwnedSlice(gpa);
    };
    defer gpa.free(current);
    defer {
        for (current) |f| f.deinit(gpa);
        gpa.free(current);
    }

    // ---- 2. ranked memories
    //
    // Without a daemon the model has to be loaded here, which costs 1-2s; the
    // daemon serves the same ranking with the model already resident.
    var vec: ?[]f32 = null;
    defer if (vec) |v| gpa.free(v);
    if (opts.query.len != 0) {
        const found = zkb.model_registry.resolve(gpa, io, env, &layout, opts.model, .q8_0) catch null;
        defer if (found) |f| f.deinit(gpa);

        if (found) |f| {
            var e = try zkb.embed.Embedder.init(gpa, f.path, .{});
            defer e.deinit();
            const buf = try gpa.alloc(f32, e.n_embd);
            _ = try e.embedQuery(zkb.embed.default_query_task, opts.query, buf);
            vec = buf;
        }
    }

    var r = try zkb.recall.assemble(gpa, &db, opts.query, vec, .{
        .budget_tokens = opts.budget,
        .candidates = opts.candidates,
        .recency_depth = opts.recency_depth,
        .scope = opts.scope,
    });
    defer r.deinit(gpa);

    switch (opts.format) {
        .markdown => {
            try zkb.recall.renderFactsMarkdown(w, current);
            if (r.ranked.hits.len == 0) {
                try w.writeAll("No memories yet. Record one with: zkb remember \"...\"\n");
            } else {
                try zkb.pack.renderMarkdown(w, &r.pack);
            }
        },
        .json => {
            try w.writeAll("{\"facts\":");
            try zkb.recall.renderFactsJson(w, current);
            try w.writeAll(",\"memories\":");
            try zkb.pack.renderJson(w, &r.pack);
            try w.writeAll("}\n");
        },
    }
    return 0;
}

/// Returns null when no daemon is listening, so the caller falls back to doing
/// it in-process. Worth trying first: the daemon has the model resident, which
/// is the whole difference between 60ms and two seconds.
fn recallViaDaemon(
    gpa: std.mem.Allocator,
    io: std.Io,
    sock: []const u8,
    w: *Writer,
    opts: RecallOptions,
) !?u8 {
    var c = zkb.ipc_client.Client.connect(io, sock) catch return null;
    defer c.close();

    var pbuf: [8192]u8 = undefined;
    var pw = std.Io.Writer.fixed(&pbuf);
    try pw.writeAll("{\"query\":");
    try std.json.Stringify.value(opts.query, .{}, &pw);
    try pw.print(",\"budget\":{d},\"candidates\":{d}}}", .{ opts.budget, opts.candidates });

    var resp = c.call(gpa, .recall, pw.buffered()) catch return null;
    defer resp.deinit(gpa);

    if (!resp.ok) {
        try w.print("{s}: {s}\n", .{ resp.code, resp.message });
        return zkb.proto.ErrorCode.exitCodeOf(resp.code);
    }
    if (opts.format == .json) {
        try w.print("{s}\n", .{resp.line});
        return 0;
    }
    try renderRecallJson(w, resp.result.?);
    return 0;
}

/// The same markdown as the in-process path, rebuilt from the daemon's JSON.
/// Kept here rather than in the daemon so the wire format stays one thing.
fn renderRecallJson(w: *Writer, result: std.json.Value) !void {
    const obj = result.object;

    if (obj.get("facts")) |fv| if (fv == .array and fv.array.items.len != 0) {
        try w.writeAll("## Facts (current values)\n\n");
        for (fv.array.items) |item| {
            const o = item.object;
            try w.print("- {s}: {s}  (as of {s})", .{
                jsonStr(o, "key"), jsonStr(o, "value"), jsonStr(o, "at"),
            });
            const note = jsonStr(o, "note");
            if (note.len != 0) try w.print(" — {s}", .{note});
            try w.writeAll("\n");
        }
        try w.writeAll("\n");
    };

    const mem = obj.get("memories") orelse return;
    const docs = if (mem.object.get("documents")) |d|
        (if (d == .array) d.array.items else &.{})
    else
        &.{};
    if (docs.len == 0) {
        try w.writeAll("No memories yet. Record one with: zkb remember \"...\"\n");
        return;
    }
    try w.writeAll("## Memories\n");
    for (docs) |d| {
        const o = d.object;
        try w.print("\n### {s}\n", .{jsonStr(o, "path")});
        const spans = if (o.get("spans")) |sp| (if (sp == .array) sp.array.items else &.{}) else &.{};
        for (spans) |sv| try w.print("\n{s}\n", .{jsonStr(sv.object, "text")});
    }
}

fn jsonStr(o: std.json.ObjectMap, key: []const u8) []const u8 {
    const v = o.get(key) orelse return "";
    return switch (v) {
        .string => |s| s,
        else => "",
    };
}

// ---------------------------------------------------------------------------
// facts
// ---------------------------------------------------------------------------

/// Reads facts.csv directly — no index and no model. Asking what your salary is
/// should not depend on whether the embedder is warm (§16.5).
pub fn factsCmd(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    key: ?[]const u8,
    want_history: bool,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    if (want_history) {
        const k = key orelse {
            try w.writeAll("--history needs a key\n");
            return 2;
        };
        const rows = try zkb.facts.history(gpa, io, layout.facts, k);
        defer {
            for (rows) |r| r.deinit(gpa);
            gpa.free(rows);
        }
        if (rows.len == 0) {
            try w.print("no fact recorded for {s}\n", .{k});
            return 3;
        }
        try w.writeAll("at\tvalue\trecorded_at\tnote\n");
        for (rows) |r| try w.print("{s}\t{s}\t{s}\t{s}\n", .{ r.at, r.value, r.recorded_at, r.note });
        return 0;
    }

    const rows = try zkb.facts.currentAll(gpa, io, layout.facts);
    defer {
        for (rows) |r| r.deinit(gpa);
        gpa.free(rows);
    }

    if (key) |k| {
        for (rows) |r| {
            if (!std.mem.eql(u8, r.key, k)) continue;
            // Bare value: this is the form something substitutes into a command.
            try w.print("{s}\n", .{r.value});
            return 0;
        }
        try w.print("no fact recorded for {s}\n", .{k});
        return 3;
    }

    if (rows.len == 0) {
        try w.writeAll("no facts recorded\n");
        return 0;
    }
    // TSV: a handful of rows costs roughly half the tokens the same data would
    // as JSON, and no agent has trouble reading it.
    try w.writeAll("key\tvalue\tat\tnote\n");
    for (rows) |r| try w.print("{s}\t{s}\t{s}\t{s}\n", .{ r.key, r.value, r.at, r.note });
    return 0;
}

pub fn rememberFact(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    key: []const u8,
    value: []const u8,
    at: ?[]const u8,
    note: []const u8,
    scope: []const u8,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);
    try layout.ensureDirs(io);

    var ts_buf: [32]u8 = undefined;
    // Defaults to today, but `--at` is the entire point of the column: a raise
    // effective in April and written down in August is `at = 2026-04-01`. When
    // it was *learned* is jj's commit time — two axes, each carried by whatever
    // already tracks it (SPEC §16.2.1).
    const effective = at orelse try isoDate(&ts_buf, io);

    // The other axis: `at` is when it took effect, `recorded_at` is now. Both
    // go in the file, because an axis that lives only in version control cannot
    // be queried and so is not really there.
    var today_buf: [32]u8 = undefined;
    const recorded = try isoDate(&today_buf, io);

    try zkb.facts.append(io, layout.facts, key, value, effective, recorded, "user", note, scope);
    try w.print("{s} = {s}  (effective {s}, recorded {s})\n", .{ key, value, effective, recorded });
    try w.print("appended to {s}\n", .{layout.facts});
    try warnIfUnversioned(io, w, layout.data);
    return 0;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn ensureDir(io: std.Io, path: []const u8) !void {
    std.Io.Dir.createDirPath(.cwd(), io, path) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

fn writeFile(io: std.Io, path: []const u8, content: []const u8) !void {
    var file = try std.Io.Dir.createFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [8192]u8 = undefined;
    var fw = file.writer(io, &buf);
    try fw.interface.writeAll(content);
    try fw.interface.flush();
}

fn nowMs(io: std.Io) i64 {
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .real).nanoseconds, std.time.ns_per_ms));
}

fn isoDate(buf: []u8, io: std.Io) ![]const u8 {
    const secs: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .real).nanoseconds, std.time.ns_per_s));
    const epoch: std.time.epoch.EpochSeconds = .{ .secs = @intCast(secs) };
    const year_day = epoch.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}", .{
        year_day.year,
        month_day.month.numeric(),
        month_day.day_index + 1,
    });
}

fn uniquePath(
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    stem: []const u8,
) ![]u8 {
    var n: usize = 0;
    while (n < 100) : (n += 1) {
        const candidate = if (n == 0)
            try std.fmt.allocPrint(gpa, "{s}/{s}.md", .{ dir, stem })
        else
            try std.fmt.allocPrint(gpa, "{s}/{s}-{d}.md", .{ dir, stem, n + 1 });
        if (std.Io.Dir.accessAbsolute(io, candidate, .{})) |_| {
            gpa.free(candidate);
            continue;
        } else |_| return candidate;
    }
    return error.TooManyCollisions;
}

fn readAll(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    const buf = try gpa.alloc(u8, @intCast(size));
    errdefer gpa.free(buf);
    var rbuf: [4096]u8 = undefined;
    var reader = file.reader(io, &rbuf);
    try reader.interface.readSliceAll(buf);
    return buf;
}

fn writeOneLine(w: *Writer, text: []const u8) !void {
    var last_space = false;
    for (text) |ch| {
        const sp = ch == '\n' or ch == '\r' or ch == '\t' or ch == ' ';
        if (sp) {
            if (!last_space) try w.writeByte(' ');
            last_space = true;
        } else {
            try w.writeByte(ch);
            last_space = false;
        }
    }
}

/// Mention version control once, the first time a memory root has none.
///
/// Not a warning any more: nothing in zkb calls `jj` or `git`, and since
/// `facts.csv` carries `recorded_at` there is no feature that depends on a repo.
/// It is still the only data here that cannot be rebuilt, so it is worth saying
/// — but a line printed on every single write is a line people stop reading.
fn warnIfUnversioned(io: std.Io, w: *Writer, kb: []const u8) !void {
    var buf: [512]u8 = undefined;
    inline for (.{ ".jj", ".git" }) |marker| {
        const p = std.fmt.bufPrint(&buf, "{s}/{s}", .{ kb, marker }) catch return;
        if (std.Io.Dir.accessAbsolute(io, p, .{})) |_| return else |_| {}
    }
    try w.print("\nnote: {s} is not under version control (optional)\n", .{kb});
}
