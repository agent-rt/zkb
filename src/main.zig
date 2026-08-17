const std = @import("std");
const zkb = @import("zkb");
const doctor = @import("cli/doctor.zig");
const model = @import("cli/model.zig");
const index_cmd = @import("cli/index_cmd.zig");
const search_cmd = @import("cli/search_cmd.zig");
const daemon_cmd = @import("cli/daemon_cmd.zig");

const usage =
    \\zkb — Agent memory + personal knowledge base
    \\
    \\usage: zkb <command> [args]
    \\
    \\  daemon start [--preload] [--root DIR] [--model PATH]
    \\  daemon stop | status | run | install | uninstall
    \\  index [--root DIR] [--collection NAME] [--force] [--model PATH]
    \\  search <query> [-k N] [--mode hybrid|vector|keyword] [--collection NAME]
    \\                 [--json] [--full] [--model PATH]
    \\  status
    \\  doctor [--model PATH]
    \\  model pull [--quant q8_0|f16]
    \\  version
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;

    var out_buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &out_buf);
    const w = &stdout.interface;
    defer w.flush() catch {};

    var args = init.minimal.args.iterate();
    _ = args.next();
    const cmd = args.next() orelse {
        try w.writeAll(usage);
        return 2;
    };

    if (std.mem.eql(u8, cmd, "daemon")) {
        const sub = args.next() orelse {
            try w.writeAll("usage: zkb daemon start|stop|status|run|install|uninstall\n");
            return 2;
        };
        var opts: daemon_cmd.Options = .{};
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--preload")) {
                opts.preload_model = true;
            } else if (std.mem.eql(u8, a, "--root")) {
                opts.root = args.next();
            } else if (std.mem.eql(u8, a, "--model")) {
                opts.model = args.next();
            } else if (std.mem.eql(u8, a, "--collection")) {
                opts.collection = args.next() orelse "docs";
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }

        if (std.mem.eql(u8, sub, "start"))
            return daemon_cmd.start(gpa, init.io, init.environ_map, w, opts);
        if (std.mem.eql(u8, sub, "stop"))
            return daemon_cmd.stop(gpa, init.io, init.environ_map, w);
        if (std.mem.eql(u8, sub, "status"))
            return daemon_cmd.status(gpa, init.io, init.environ_map, w);
        if (std.mem.eql(u8, sub, "install"))
            return daemon_cmd.install(gpa, init.io, init.environ_map, w);
        if (std.mem.eql(u8, sub, "uninstall"))
            return daemon_cmd.uninstall(gpa, init.io, init.environ_map, w);
        if (std.mem.eql(u8, sub, "run")) {
            // Foreground: this is what launchd supervises and what `start` spawns.
            try w.flush();
            try zkb.daemon.run(gpa, init.io, init.environ_map, .{
                .preload_model = opts.preload_model,
                .scan_interval_s = opts.scan_interval_s,
                .collection = opts.collection,
                .root = opts.root,
                .model_path = opts.model,
            });
            return 0;
        }
        try w.print("unknown daemon subcommand: {s}\n", .{sub});
        return 2;
    }

    if (std.mem.eql(u8, cmd, "index")) {
        var opts: index_cmd.Options = .{};
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--root")) {
                opts.root = args.next();
            } else if (std.mem.eql(u8, a, "--collection")) {
                opts.collection = args.next() orelse "docs";
            } else if (std.mem.eql(u8, a, "--model")) {
                opts.model = args.next();
            } else if (std.mem.eql(u8, a, "--force")) {
                opts.force = true;
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        return index_cmd.run(gpa, init.io, init.environ_map, w, opts);
    }

    if (std.mem.eql(u8, cmd, "search")) {
        // Options and the query are parsed in one pass: requiring the query first
        // made `zkb search --mode keyword ...` report "unknown option: keyword",
        // which is both wrong and confusing.
        var query: ?[]const u8 = null;
        var opts: search_cmd.Options = .{ .query = "" };
        while (args.next()) |a| {
            if (!std.mem.startsWith(u8, a, "-")) {
                if (query == null) query = a else {
                    try w.print("unexpected extra argument: {s}\n", .{a});
                    try w.writeAll("(quote a multi-word query)\n");
                    return 2;
                }
            } else if (std.mem.eql(u8, a, "-k")) {
                const v = args.next() orelse break;
                opts.top_k = std.fmt.parseInt(usize, v, 10) catch 10;
            } else if (std.mem.eql(u8, a, "--mode")) {
                const v = args.next() orelse break;
                opts.mode = std.meta.stringToEnum(zkb.hybrid.Mode, v) orelse {
                    try w.print("unknown mode: {s} (hybrid|vector|keyword)\n", .{v});
                    return 2;
                };
            } else if (std.mem.eql(u8, a, "--collection")) {
                opts.collection = args.next();
            } else if (std.mem.eql(u8, a, "--model")) {
                opts.model = args.next();
            } else if (std.mem.eql(u8, a, "--json")) {
                opts.json = true;
            } else if (std.mem.eql(u8, a, "--full")) {
                opts.full = true;
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        opts.query = query orelse {
            try w.writeAll("usage: zkb search <query> [-k N] [--mode hybrid|vector|keyword]\n");
            return 2;
        };
        return search_cmd.run(gpa, init.io, init.environ_map, w, opts);
    }

    if (std.mem.eql(u8, cmd, "status")) {
        return status(gpa, init.io, init.environ_map, w);
    }

    if (std.mem.eql(u8, cmd, "doctor")) {
        var model_path: ?[]const u8 = null;
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--model")) model_path = args.next();
        }
        const r = try doctor.run(gpa, init.io, init.environ_map, w, model_path);
        return if (r.failures == 0) 0 else 1;
    }

    if (std.mem.eql(u8, cmd, "model")) {
        const sub = args.next() orelse {
            try w.writeAll("usage: zkb model pull [--quant q8_0|f16]\n");
            return 2;
        };
        if (!std.mem.eql(u8, sub, "pull")) {
            try w.print("unknown model subcommand: {s}\n", .{sub});
            return 2;
        }
        var quant: model.Quant = .q8_0;
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--quant")) {
                const v = args.next() orelse break;
                quant = std.meta.stringToEnum(model.Quant, v) orelse {
                    try w.print("unknown quant: {s} (expected q8_0 or f16)\n", .{v});
                    return 2;
                };
            }
        }
        model.pull(gpa, init.io, init.environ_map, w, quant) catch |err| {
            try w.print("model pull failed: {t}\n", .{err});
            return 1;
        };
        return 0;
    }

    if (std.mem.eql(u8, cmd, "version")) {
        try w.print("zkb {s}\n", .{zkb.version});
        try w.print("sqlite {s} / sqlite-vec {s}\n", .{
            zkb.sqlite.libVersion(),
            zkb.sqlite.vecVersion(),
        });
        try w.print("llama backend: {s}\n", .{if (zkb.build_options.llama) "on" else "off"});
        return 0;
    }

    try w.print("unknown command: {s}\n\n", .{cmd});
    try w.writeAll(usage);
    return 2;
}

fn status(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *std.Io.Writer,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    std.Io.Dir.accessAbsolute(io, layout.db, .{}) catch {
        try w.print("no index at {s}\nrun: zkb index\n", .{layout.db});
        return 3;
    };

    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_only) catch |err| switch (err) {
        // A read-only connection cannot migrate. Say what to run rather than
        // failing inside a DDL statement.
        error.SchemaStale => {
            try w.writeAll("index schema is out of date\nrun: zkb index\n");
            return 3;
        },
        error.SchemaFromFuture => {
            try w.writeAll("index was written by a newer zkb\nupgrade zkb, or: rm ~/.zkb/zkb.db && zkb index\n");
            return 3;
        },
        else => return err,
    };
    defer db.close();
    var s = zkb.store.Store.init(&db);

    var buf: [256]u8 = undefined;
    if (try zkb.schema.getMeta(&db, "embedding_model_id", &buf)) |id| {
        try w.print("model  {s}\n", .{id});
    } else {
        try w.writeAll("model  (none recorded)\n");
    }

    try w.writeAll("\ncollections\n");
    {
        var st = try db.prepare(
            \\SELECT col.name, col.root, count(d.id),
            \\       COALESCE(sum(d.chunk_count), 0)
            \\FROM collections col LEFT JOIN docs d ON d.collection_id = col.id
            \\GROUP BY col.id ORDER BY col.id
        );
        defer st.finalize();
        while (try st.step()) {
            try w.print("  {s}  {d} docs, {d} chunks  ({s})\n", .{
                st.columnText(0), st.columnI64(2), st.columnI64(3), st.columnText(1),
            });
        }
    }

    const c = try s.counts();
    try w.print("\ntotals  {d} docs, {d} chunks\n", .{ c.docs, c.chunks });
    if (c.pending != 0) try w.print("pending {d}\n", .{c.pending});
    if (c.failed != 0) try w.print("failed  {d}\n", .{c.failed});

    // Drift between the three tables is the silent failure this project guards
    // against hardest; surface it wherever counts are shown.
    if (c.chunks != c.fts_rows or c.chunks != c.vec_rows) {
        try w.print("\nWARNING index drift: chunks {d}, fts {d}, vec {d}\n", .{
            c.chunks, c.fts_rows, c.vec_rows,
        });
        return 1;
    }
    return 0;
}
