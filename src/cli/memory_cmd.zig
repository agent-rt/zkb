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

    // Before the model is resolved: a mistyped label should not cost a 1-2s load
    // to be told it is mistyped.
    if (try refuseUnknownScope(gpa, &db, w, opts.scope, opts.force)) return 5;

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

/// Refuse a scope no active memory carries. Returns true when it refused.
///
/// The same shape as the duplicate check above, and written for the same reason:
/// a scope is free text, so `--scope zbk` is accepted, stored, and then never
/// matches anything again — the memory is invisible and nothing reports it. The
/// skill calls this the one mistake here that does not announce itself, and a
/// mistake that cannot announce itself has to be refused at the moment it is
/// made, not detected later.
///
/// Creating a scope stays possible, it just stops being accidental: `--force` is
/// the difference between naming a new project and mistyping an old one. That is
/// the same bargain `remember` already strikes with near-duplicates.
///
/// Reading is never refused. `recall --scope X` for an unknown X returns the
/// universal memories, because a session must still start; refusing there would
/// turn a typo into a broken session instead of a thinner one.
fn refuseUnknownScope(
    gpa: std.mem.Allocator,
    db: *zkb.sqlite.Db,
    w: *Writer,
    scope: []const u8,
    force: bool,
) !bool {
    if (scope.len == 0 or force) return false;

    const known = try zkb.memory.activeScopes(gpa, db);
    defer {
        for (known) |s| gpa.free(s);
        gpa.free(known);
    }
    for (known) |s| {
        if (std.mem.eql(u8, s, scope)) return false;
    }

    try w.print("unknown scope \"{s}\"; not recorded\n\n", .{scope});
    if (known.len == 0) {
        try w.writeAll("  no memory carries a scope yet\n");
    } else {
        try w.writeAll("  in use:");
        for (known, 0..) |s, i| try w.print("{s} {s}", .{ if (i == 0) "" else ",", s });
        try w.writeAll("\n");
    }
    try w.writeAll("\na scope starts existing by being used — re-run with --force if that is what you mean\n");
    return true;
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
    return setFrontmatterField(gpa, content, "status", "archived");
}

// ---------------------------------------------------------------------------
// rescope
// ---------------------------------------------------------------------------

/// Move an existing memory to another scope, or to universal with an empty one.
///
/// Until this existed there was no way to change a memory's scope at all. The
/// three things left were: `forget` and re-`remember`, which resets `created` and
/// so rewrites the recency ranking recall is built on; editing the file by hand,
/// which the skill and this module both forbid because `remember` is meant to be
/// the only writer of the data directory; or living with it. A label that cannot
/// be corrected is a label nobody should be told to apply.
///
/// The projection is rebuilt by re-indexing, not by an UPDATE on `rec_memory`.
/// Writing the row here would be a second path that produces it, and the first
/// one is `indexer.indexOne` — two writers of the same derived table is the shape
/// that emptied it once already.
pub fn rescope(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    rel_path: []const u8,
    scope: []const u8,
    force: bool,
    model: ?[]const u8,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    const path = try std.fs.path.join(gpa, &.{ layout.memory, rel_path });
    defer gpa.free(path);
    std.Io.Dir.accessAbsolute(io, path, .{}) catch {
        try w.print("no such memory: {s}\n", .{path});
        return 3;
    };

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = try zkb.store.open(db_path, .read_write);
    defer db.close();

    // Same gate as `remember`: the label is the thing that goes wrong silently,
    // and moving a memory to a mistyped scope hides it exactly as writing one
    // there would.
    if (try refuseUnknownScope(gpa, &db, w, scope, force)) return 5;

    const content = try readAll(gpa, io, path);
    defer gpa.free(content);
    const updated = setFrontmatterField(gpa, content, "scope", scope) catch |err| switch (err) {
        error.NoFrontmatter => {
            try w.print("not a memory (no frontmatter): {s}\n", .{path});
            return 2;
        },
        else => return err,
    };
    defer gpa.free(updated);

    if (std.mem.eql(u8, content, updated)) {
        try w.print("unchanged: already {s}\n", .{
            if (scope.len == 0) "universal" else scope,
        });
        return 0;
    }
    try writeFile(io, path, updated);

    // Re-indexed here rather than left for the daemon: the whole point is that
    // the next `recall` sees it, and "I moved that memory and it is still in the
    // wrong scope" is how a tool stops being trusted.
    var s = zkb.store.Store.init(&db);
    const now_ms = nowMs(io);
    const cid = try s.ensureCollectionKind("memory", layout.memory, .memory, now_ms);

    const found = zkb.model_registry.resolve(gpa, io, env, &layout, model, .q8_0) catch null;
    defer if (found) |f| f.deinit(gpa);
    if (found) |f| {
        var e = zkb.embed.Embedder.init(gpa, f.path, .{}) catch null;
        if (e) |*emb| {
            defer emb.deinit();
            _ = try zkb.scan.reconcile(gpa, io, &s, cid, layout.memory, zkb.memory.scan_filters, now_ms);
            _ = try zkb.indexer.indexPending(gpa, io, &s, emb, cid, layout.memory, now_ms, .{});
        }
    } else {
        try w.writeAll("note: model unavailable; the index updates on the next daemon scan\n");
    }

    if (scope.len == 0) {
        try w.print("{s} is now universal\n", .{rel_path});
    } else {
        try w.print("{s} is now scoped to {s}\n", .{ rel_path, scope });
    }
    return 0;
}

/// Set one frontmatter field, adding it when absent and dropping it when `value`
/// is empty. The body is returned untouched.
///
/// Bounded to the block between the first two `---` lines, which the previous
/// version was not: it rewrote the first line starting with the key anywhere in
/// the file, so a memory whose *text* began a line with `status:` would have had
/// its content edited instead of its metadata. No memory did, but that is a
/// property of the corpus rather than of the function.
pub fn setFrontmatterField(
    gpa: std.mem.Allocator,
    content: []const u8,
    key: []const u8,
    value: []const u8,
) ![]u8 {
    if (!std.mem.startsWith(u8, content, "---\n")) return error.NoFrontmatter;
    // The newline that ends the last field line, immediately before the closing
    // fence. Searching from 4 skips the opening one.
    const end = std.mem.indexOfPos(u8, content, 4, "\n---") orelse return error.NoFrontmatter;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, "---\n");

    var replaced = false;
    var it = std.mem.splitScalar(u8, content[4 .. end + 1], '\n');
    while (it.next()) |line| {
        // The split leaves an empty final piece after the trailing newline.
        if (line.len == 0) continue;
        const t = std.mem.trimStart(u8, line, " \t");
        const is_key = std.mem.startsWith(u8, t, key) and t.len > key.len and t[key.len] == ':';
        if (is_key and !replaced) {
            replaced = true;
            if (value.len == 0) continue;
            try out.print(gpa, "{s}: {s}\n", .{ key, value });
        } else {
            try out.appendSlice(gpa, line);
            try out.append(gpa, '\n');
        }
    }
    if (!replaced and value.len != 0) try out.print(gpa, "{s}: {s}\n", .{ key, value });

    try out.appendSlice(gpa, content[end + 1 ..]);
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
    //
    // Filtered before rendering, not inside the renderer: `zkb facts` shows
    // everything on purpose (you asked for it), while recall is injected without
    // anyone asking — which is the whole reason a scope exists. The filter lives
    // in `facts` so the daemon applies the same one.
    const current = zkb.facts.currentInScope(gpa, io, layout.facts, opts.scope) catch &.{};
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
            try zkb.recall.renderMemoriesMarkdown(w, &r.pack, r.memory_docs);
        },
        .json => {
            try w.writeAll("{\"facts\":");
            try zkb.recall.renderFactsJson(w, current);
            // On the wire for the same reason it is in the markdown, and in both
            // json shapes because a field that exists on one transport only is
            // how the two drifted in the first place.
            try w.print(",\"memory_docs\":{d},\"memories\":", .{r.memory_docs});
            try zkb.pack.renderJson(w, &r.pack);
            try w.writeAll("}\n");
        },
    }
    return 0;
}

/// Returns null when no daemon is listening, so the caller falls back to doing
/// it in-process. Worth trying first: the daemon has the model resident, which
/// is the whole difference between 60ms and two seconds.
/// Every `RecallOptions` field that changes the answer, written in one place.
///
/// This used to be three lines inside `recallViaDaemon`, carrying three of the
/// five fields. The two it left out were `scope` and `recency_depth`, and nothing
/// failed: with a daemon running, `zkb recall --scope work` answered with the
/// universal memories and silently dropped the scoped one — measured, six
/// documents without a daemon and five with.
///
/// That is precisely the bug `query_cmd`'s "every retrieval command puts its
/// filters on the wire" was written to catch, and its guard could not reach here,
/// because `recall` had never adopted this shape: there was no function for the
/// table to call. Adding a row would not have helped; the missing thing was this.
pub fn requestParams(w: *Writer, opts: RecallOptions) !void {
    try w.writeAll("{\"query\":");
    try std.json.Stringify.value(opts.query, .{}, w);
    try w.print(",\"budget\":{d},\"candidates\":{d},\"recency_depth\":{d}", .{
        opts.budget, opts.candidates, opts.recency_depth,
    });
    // Absent when unset, never `""`. An empty scope is not "no scope" — it is how
    // a *universal* memory is stored (`memory.Meta.scope`) — so a field written
    // unconditionally would turn "the caller named no scope" into "the caller
    // asked for the scope whose name is empty" on the other side.
    if (opts.scope) |s| {
        try w.writeAll(",\"scope\":");
        try std.json.Stringify.value(s, .{}, w);
    }
    try w.writeAll("}");
}

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
    try requestParams(&pw, opts);

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
    try renderRecallJson(gpa, w, resp.result.?);
    return 0;
}

/// The daemon's json, handed to the same two functions the in-process path uses.
/// Rebuilt here rather than rendered in the daemon so the wire format stays one
/// thing — but rebuilt into the *types* those functions take, not into a second
/// renderer. This function used to be that second renderer, and it had drifted:
/// `### x.md` for `## memory/x.md`, and no `omitted`/`tokens` footer at all.
fn renderRecallJson(gpa: std.mem.Allocator, w: *Writer, result: std.json.Value) !void {
    const obj = result.object;

    // Everything below borrows from `result`, which outlives this call.
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const fact_items = if (obj.get("facts")) |fv|
        (if (fv == .array) fv.array.items else &.{})
    else
        &.{};
    const lines = try arena.alloc(zkb.recall.FactLine, fact_items.len);
    for (fact_items, 0..) |item, i| {
        const o = item.object;
        lines[i] = .{
            .key = jsonStr(o, "key"),
            .value = jsonStr(o, "value"),
            .at = jsonStr(o, "at"),
            .note = jsonStr(o, "note"),
        };
    }
    try zkb.recall.renderFactsMarkdown(w, lines);

    const mem = obj.get("memories") orelse return;
    var pack = try zkb.pack.fromJson(arena, mem);
    try zkb.recall.renderMemoriesMarkdown(w, &pack, jsonUsize(obj, "memory_docs"));
}

fn jsonUsize(o: std.json.ObjectMap, key: []const u8) usize {
    const v = o.get(key) orelse return 0;
    return switch (v) {
        .integer => |i| @intCast(@max(0, i)),
        else => 0,
    };
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

    zkb.facts.append(gpa, io, layout.facts, key, value, effective, recorded, "user", note, scope) catch |err| switch (err) {
        // The header predates a column *and* rows are already malformed. Those
        // rows are the data at risk — most likely ones a newer zkb appended and
        // nothing has been able to read since — so the file is left exactly as
        // it is. A stack trace would be the wrong answer to a file the person
        // can open and fix.
        error.StaleHeaderWithBadRows => {
            const lines = zkb.facts.badRowLines(gpa, io, layout.facts) catch &.{};
            defer gpa.free(lines);
            try w.print("{s}: {d} row(s) do not match the header", .{ layout.facts, lines.len });
            for (lines, 0..) |n, i| try w.print("{s}{d}", .{ if (i == 0) " — line " else ", ", n });
            try w.writeAll("\nnot written: the header is out of date, and widening it would drop those rows\n");
            try w.writeAll("fix them (each row needs one field per header column), then retry\n");
            return 2;
        },
        else => return err,
    };
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

const testing = std.testing;

test "recall hands the facts back once, and only the ones in scope" {
    // This crashed. `zkb recall` with no daemon and at least one fact freed the
    // filtered slice twice — a `defer gpa.free(current)` left behind when the
    // scope filter arrived and brought its own freeing defer. Nobody saw it
    // because the machine that would notice always has a daemon running, and the
    // daemon path never reaches this code. A fresh install has no daemon, and
    // `zkb recall` is the first command the skill tells an agent to run.
    //
    // `testing.allocator` panics on the double free and fails on the leak of the
    // facts the filter dropped, so both halves of the fix are asserted by simply
    // getting here.
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var root_buf: [64]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/zkb-recall-test-{x}", .{@intFromPtr(&threaded)});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};

    {
        const data = try std.fmt.allocPrint(testing.allocator, "{s}/data", .{root});
        defer testing.allocator.free(data);
        var d = try std.Io.Dir.cwd().createDirPathOpen(io, data, .{});
        defer d.close(io);
        var f = try d.createFile(io, "facts.csv", .{});
        defer f.close(io);
        var fbuf: [512]u8 = undefined;
        var wr = f.writer(io, &fbuf);
        try wr.interface.writeAll(
            \\key,value,at,recorded_at,src,note,scope
            \\height,178,2026-01-01,2026-01-02,user,,
            \\salary,450000,2026-01-01,2026-01-02,user,,work
            \\
        );
        try wr.interface.flush();

        // An index has to exist or recall returns 3 before reaching any of this.
        const index_dir = try std.fmt.allocPrint(testing.allocator, "{s}/index", .{root});
        defer testing.allocator.free(index_dir);
        var i = try std.Io.Dir.cwd().createDirPathOpen(io, index_dir, .{});
        i.close(io);
        const db_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/zkb.db", .{index_dir}, 0);
        defer testing.allocator.free(db_path);
        var db = try zkb.store.open(db_path, .read_write);
        db.close();
    }

    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    try env.put("ZKB_HOME", root);

    var out: [8192]u8 = undefined;
    {
        var w = std.Io.Writer.fixed(&out);
        const code = try recall(testing.allocator, io, &env, &w, .{ .budget = 500 });
        try testing.expectEqual(@as(u8, 0), code);
        // Unscoped caller: the universal fact, and not the labelled one.
        try testing.expect(std.mem.indexOf(u8, w.buffered(), "height") != null);
        try testing.expect(std.mem.indexOf(u8, w.buffered(), "salary") == null);
    }
    {
        var w = std.Io.Writer.fixed(&out);
        _ = try recall(testing.allocator, io, &env, &w, .{ .budget = 500, .scope = "work" });
        try testing.expect(std.mem.indexOf(u8, w.buffered(), "height") != null);
        try testing.expect(std.mem.indexOf(u8, w.buffered(), "salary") != null);
    }
}

test "setFrontmatterField edits metadata and never the body" {
    // The previous rewriter matched the first line starting with the key
    // *anywhere*, so a memory whose text began a line with `scope:` would have
    // had its content edited. This body is written to trip exactly that.
    const doc =
        \\---
        \\type: project
        \\created: 2026-08-18
        \\status: active
        \\---
        \\
        \\scope: 这行是正文，不是元数据。
        \\
    ;

    const set = try setFrontmatterField(testing.allocator, doc, "scope", "zkb");
    defer testing.allocator.free(set);
    // Added, because the file had no such field.
    try testing.expect(std.mem.indexOf(u8, set, "\nscope: zkb\n") != null);
    // And the body line survived untouched.
    try testing.expect(std.mem.indexOf(u8, set, "scope: 这行是正文，不是元数据。") != null);
    try testing.expect(std.mem.indexOf(u8, set, "type: project") != null);

    // Replacing, not appending a second one.
    const again = try setFrontmatterField(testing.allocator, set, "scope", "aglet");
    defer testing.allocator.free(again);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, again, "\nscope: aglet\n"));
    try testing.expect(std.mem.indexOf(u8, again, "scope: zkb") == null);

    // An empty value clears it, which is how a memory goes back to universal.
    const cleared = try setFrontmatterField(testing.allocator, again, "scope", "");
    defer testing.allocator.free(cleared);
    try testing.expect(std.mem.indexOf(u8, cleared, "\nscope: aglet\n") == null);
    try testing.expect(std.mem.indexOf(u8, cleared, "scope: 这行是正文，不是元数据。") != null);
    try testing.expect(std.mem.indexOf(u8, cleared, "status: active") != null);
}

test "setFrontmatterField refuses a file that has no frontmatter" {
    // Rather than inventing a block: a file without one is not a memory, and
    // writing metadata into arbitrary text is worse than declining.
    try testing.expectError(
        error.NoFrontmatter,
        setFrontmatterField(testing.allocator, "just prose\n", "scope", "zkb"),
    );
}
