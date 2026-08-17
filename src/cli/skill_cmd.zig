//! `zkb skill` — emit zkb's own SKILL.md, for whatever agent will read it.
//!
//! **Content only; installing it is somebody else's job.** Skill managers own
//! distribution — global install, per-project selection, and whatever stable
//! block they inject into `AGENTS.md` or `CLAUDE.md`. Those files are written as
//! rarely as possible, because rewriting them costs prompt-cache hits, so a
//! second tool editing the same file would break the property they protect.
//! This writes to stdout and stops.
//!
//! **A command rather than a checked-in file, because half of it is local.**
//! Which record types exist and what columns they have is not something an agent
//! can guess, and it differs per machine. A static SKILL.md can describe the
//! commands but not the data; the introspected half is where the agent stops
//! having to try things to find out.

const std = @import("std");
const zkb = @import("zkb");

const Writer = std.Io.Writer;

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    try writeFrontmatter(w);
    try writeOverview(w);
    try writeDecisionTable(w);
    try writeLocalState(gpa, io, w, &layout);
    try writeGotchas(w);
    try writeMcp(w);
    return 0;
}

/// The description is the only part loaded on every turn, so it carries the
/// activation triggers. Everything else is read on demand.
fn writeFrontmatter(w: *Writer) !void {
    try w.writeAll(
        \\---
        \\name: zkb
        \\description: "Personal knowledge base and agent memory over the user's own markdown and csv. Use when: the user asks what they wrote, decided or preferred before; a question needs their documents rather than general knowledge; you need their recorded facts (salary, height, birth date) or structured records (expenses, weight logs); or you learn a durable preference worth remembering. Provides hybrid semantic+keyword retrieval, a memory system with write-time deduplication, and SQL-grade filtering over csv."
        \\---
        \\
        \\# zkb
        \\
        \\
    );
}

fn writeOverview(w: *Writer) !void {
    try w.writeAll(
        \\Retrieval and memory over files the user already owns: markdown for prose,
        \\csv for numbers. zkb never generates answers — it returns evidence and you
        \\reason over it.
        \\
        \\**Truth is the filesystem.** The index is derived and can be rebuilt at any
        \\time, so nothing here is authoritative except the files.
        \\
        \\
    );
}

/// The table an agent gets wrong without help: four commands that all look like
/// "search" but answer different questions.
fn writeDecisionTable(w: *Writer) !void {
    try w.writeAll(
        \\## Which command
        \\
        \\| The question | Use | Why not the others |
        \\|---|---|---|
        \\| Where is this written? | `zkb search <q>` | Returns chunks with paths; cheapest |
        \\| I need context to answer with | `zkb query <q>` | Pulls in neighbours, groups per document, fits a token budget |
        \\| What do I know about this user? | `zkb recall [topic]` | Facts snapshot + memories, ranked by relevance *and* recency |
        \\| What is the exact number? | `zkb records <type> --where/--agg` or `zkb facts <key>` | **Numbers are not in the vector index** — search cannot answer them |
        \\
        \\Start a session with `zkb recall`. It is cheap (1500 tokens) and it is how
        \\you find out what was already decided.
        \\
        \\
    );
}

fn writeLocalState(
    gpa: std.mem.Allocator,
    io: std.Io,
    w: *Writer,
    layout: *const zkb.paths.Layout,
) !void {
    try w.writeAll("## On this machine\n\n");

    // Daemon first: everything else is faster when it is up, and "start it" is
    // the single most useful instruction if it is not.
    if (std.Io.Dir.accessAbsolute(io, layout.sock, .{})) |_| {
        try w.writeAll("The daemon is running (searches are ~60ms).\n\n");
    } else |_| {
        try w.writeAll(
            \\The daemon is **not** running: every command will load the model itself
            \\and take 1-2s. Start it once with `zkb daemon start --preload`.
            \\
            \\
        );
    }

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch {
        try w.writeAll("No index yet. Run `zkb index` before anything else.\n\n");
        return;
    };
    defer db.close();

    var s = zkb.store.Store.init(&db);
    if (s.counts()) |c| {
        try w.print("Indexed: {d} documents, {d} chunks", .{ c.docs, c.chunks });
        if (c.pending != 0) try w.print(", {d} pending", .{c.pending});
        if (c.failed != 0) try w.print(", **{d} failed**", .{c.failed});
        try w.writeAll(".\n\n");
    } else |_| {}

    try writeCollections(gpa, &db, w);
    try writeRecordTypes(gpa, &db, w);
    try writeFactKeys(gpa, io, w, layout);
}

fn writeCollections(gpa: std.mem.Allocator, db: *zkb.sqlite.Db, w: *Writer) !void {
    var st = db.prepare(
        \\SELECT c.name, c.root, count(d.id)
        \\FROM collections c LEFT JOIN docs d ON d.collection_id = c.id
        \\GROUP BY c.id ORDER BY c.name
    ) catch return;
    defer st.finalize();

    var any = false;
    while (st.step() catch false) {
        if (!any) {
            try w.writeAll("| collection | root | docs |\n|---|---|---|\n");
            any = true;
        }
        try w.print("| {s} | `{s}` | {d} |\n", .{ st.columnText(0), st.columnText(1), st.columnI64(2) });
    }
    if (any) try w.writeAll("\n");
    _ = gpa;
}

/// The part an agent cannot guess and would otherwise discover by trial: which
/// record types exist and what their columns are called.
fn writeRecordTypes(gpa: std.mem.Allocator, db: *zkb.sqlite.Db, w: *Writer) !void {
    const types = zkb.records.listTypes(gpa, db) catch return;
    defer {
        for (types) |t| gpa.free(t);
        gpa.free(types);
    }
    if (types.len == 0) {
        try w.writeAll(
            \\No record types yet. One appears as soon as a csv lands under
            \\`~/.zkb/data/records/<type>/` and `zkb index` runs.
            \\
            \\
        );
        return;
    }

    try w.writeAll("### Record types\n\n");
    for (types) |t| {
        var schema = (zkb.records.loadSchema(gpa, db, t) catch null) orelse continue;
        defer schema.deinit(gpa);

        try w.print("**{s}** — ", .{t});
        for (schema.fields, 0..) |f, i| {
            if (i != 0) try w.writeAll(", ");
            try w.print("`{s}` {t}", .{ f.name, f.kind });
        }
        try w.writeAll("\n\n");
    }
    try w.writeAll(
        \\Only `string` columns are embedded; `number` / `date` / `enum` / `id` are
        \\filterable but invisible to semantic search. Ask for them with `--where`.
        \\
        \\
    );
}

fn writeFactKeys(
    gpa: std.mem.Allocator,
    io: std.Io,
    w: *Writer,
    layout: *const zkb.paths.Layout,
) !void {
    const rows = zkb.facts.currentAll(gpa, io, layout.facts) catch return;
    defer {
        for (rows) |r| r.deinit(gpa);
        gpa.free(rows);
    }
    if (rows.len == 0) return;

    try w.writeAll("### Known facts\n\n");
    for (rows) |r| try w.print("- `{s}`\n", .{r.key});
    try w.writeAll(
        \\
        \\Read a value with `zkb facts <key>`. `zkb recall` already injects all of
        \\them, so you rarely need to ask twice.
        \\
        \\
    );
}

/// Each of these is a mistake an agent makes on its own, and each one is silent.
fn writeGotchas(w: *Writer) !void {
    try w.writeAll(
        \\## Things that are easy to get wrong
        \\
        \\**Numbers are not retrievable.** `450000` and `480000` are neighbours in a
        \\1024-dimensional space, so searching for a salary returns prose that
        \\mentions salaries, not the value. Every number lives in a csv column: use
        \\`zkb facts` or `zkb records --where`. `zkb recall` injects the current
        \\value of every fact for exactly this reason — trust that block over
        \\anything a document says.
        \\
        \\**`zkb remember` exiting 5 is not a failure.** It means a near-duplicate
        \\memory already exists, and it prints the candidates with their cosine.
        \\Read them, then either edit the existing file or re-run with `--force` if
        \\it really is new. Recording the same thing twenty times is the way memory
        \\systems die.
        \\
        \\**Two time axes on a fact.** `at` is when it took effect; `recorded_at` is
        \\when it was written down. "Salary rose in April, noted in August" is
        \\`--at 2026-04-01`. Passing today's date as `--at` loses the distinction.
        \\
        \\**Never write into the data directory by hand.** `zkb remember` /
        \\`remember-fact` are the only writers of `~/.zkb/data`; they also index
        \\immediately, which hand-editing does not. Documents are the opposite: edit
        \\them with your normal tools, zkb picks the change up within seconds.
        \\
        \\**Dropped query terms are reported.** If search says a term was not
        \\searched, the tokenizer could not match it — that result is narrower than
        \\the question asked.
        \\
        \\
    );
}

fn writeMcp(w: *Writer) !void {
    try w.writeAll(
        \\## Over MCP
        \\
        \\`zkb mcp` exposes `zkb_search`, `zkb_query`, `zkb_recall` and
        \\`zkb_records`, plus `zkb://stats` and `zkb://health` as resources.
        \\
        \\`zkb_records` with no arguments lists the types; with `schema: true` it
        \\lists a type's columns. There is no write tool — `zkb remember` is a shell
        \\command, and the daemon keeps a single writer on purpose.
        \\
        \\## Escape hatch
        \\
        \\`zkb sql "select ..."` runs read-only SQL over the index when the
        \\restricted `--where` / `--agg` grammar cannot say it: date functions,
        \\joins across record types, window functions. See `docs/recipes.md` in the
        \\zkb repository for worked examples.
        \\
        \\**Check `zkb sql --list` before writing one.** A query worth keeping is a
        \\`.sql` file under `data/queries/`, run as `zkb sql @<name> [key=value ...]`.
        \\Its parameters are bound by sqlite, so prefer a saved query over building
        \\a statement out of user input — and never paste a value into sql text.
        \\
        \\`zkb sql --history` lists statements typed before this one. If you find
        \\yourself writing the same thing twice, save it as a file instead.
        \\
    );
}
