const std = @import("std");
const zkb = @import("zkb");
const doctor = @import("cli/doctor.zig");
const model = @import("cli/model.zig");
const index_cmd = @import("cli/index_cmd.zig");
const search_cmd = @import("cli/search_cmd.zig");
const daemon_cmd = @import("cli/daemon_cmd.zig");
const query_cmd = @import("cli/query_cmd.zig");
const memory_cmd = @import("cli/memory_cmd.zig");
const records_cmd = @import("cli/records_cmd.zig");
const skill_cmd = @import("cli/skill_cmd.zig");
const collection_cmd = @import("cli/collection_cmd.zig");

const usage =
    \\zkb — Agent memory + personal knowledge base
    \\
    \\usage: zkb <command> [args]
    \\
    \\  daemon start [--preload] [--root DIR] [--model PATH]
    \\  daemon stop | status | run | install | uninstall
    \\  index [--root DIR ...] [--collection NAME] [--ext md ...]
    \\        [--include GLOB ...] [--force] [--model PATH]
    \\  search <query> [-k N] [--mode hybrid|vector|keyword] [--collection NAME]
    \\                 [--path GLOB] [--json] [--full] [--model PATH]
    \\  query <question> [--budget N] [--neighbors N] [--format markdown|json]
    \\                   [--collection NAME] [--path GLOB]
    \\
    \\  remember <text> [--type user|feedback|decision|project|reference]
    \\                  [--subjects a,b] [--refs a,b] [--scope NAME] [--force]
    \\  recall [query] [--budget N] [--scope NAME] [--json]
    \\  forget <memory-file>
    \\  rescope <memory-file> <scope>   move it to another scope; empty means universal
    \\  facts [key] [--history]
    \\  remember-fact <key> <value> [--at YYYY-MM-DD] [--note TEXT]
    \\  records [type] [--where EXPR] [--search TEXT] [--agg EXPR]
    \\                 [--window "avg(f) over N by f"] [--schema] [-n N] [--json]
    \\  sql <select ...> [--json]
    \\
    \\  status
    \\  collection rm NAME           drop a collection; the files are untouched
    \\  collection checks NAME [--off a,b]   checks this corpus declines
    \\  maintain [--since last] [--check NAME] [--collection NAME] [--all] [--json]
    \\  skill                        emit zkb's SKILL.md (pipe it where your agent reads skills)
    \\  doctor [--model PATH]
    \\  model pull [--quant q8_0|f16]
    \\  version
    \\
    \\A collection remembers its root and filters, so the daemon keeps it fresh:
    \\  zkb index --collection notes --root ~/notes --ext md
    \\  zkb index --collection agent-memory --root ~/.claude/projects \
    \\            --include '*/memory/*.md' --ext md
    \\Several --root are folded into their shared parent. Prefer --include when
    \\directories will be added later: several roots name only what exists now.
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;

    var out_buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
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
            zkb.daemon.run(gpa, init.io, init.environ_map, .{
                .preload_model = opts.preload_model,
                .scan_interval_s = opts.scan_interval_s,
                .collection = opts.collection,
                .root = opts.root,
                .model_path = opts.model,
            }) catch |err| switch (err) {
                // A second daemon is a configuration mistake, not a crash. Under
                // launchd it is also the common case during an upgrade, and a
                // stack trace in a log file is a worse answer than one line.
                error.AlreadyRunning => {
                    try w.writeAll("a daemon is already running for this ZKB_HOME\n");
                    try w.writeAll("stop it first: zkb daemon stop\n");
                    return 3;
                },
                else => return err,
            };
            return 0;
        }
        try w.print("unknown daemon subcommand: {s}\n", .{sub});
        return 2;
    }

    if (std.mem.eql(u8, cmd, "index")) {
        var opts: index_cmd.Options = .{};
        // Repeatable flags accumulate. A shell glob is the expected way to give
        // several roots — `--root ~/.claude/projects/*/memory` arrives already
        // expanded — so the list has to grow rather than the last one winning.
        var roots_list: std.ArrayList([]const u8) = .empty;
        defer roots_list.deinit(gpa);
        var ext_list: std.ArrayList([]const u8) = .empty;
        defer ext_list.deinit(gpa);
        var include_list: std.ArrayList([]const u8) = .empty;
        defer include_list.deinit(gpa);
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--root")) {
                const v = args.next() orelse {
                    try w.writeAll("--root needs a path\n");
                    return 2;
                };
                try roots_list.append(gpa, v);
            } else if (std.mem.eql(u8, a, "--collection")) {
                opts.collection = args.next() orelse "docs";
            } else if (std.mem.eql(u8, a, "--ext")) {
                const v = args.next() orelse {
                    try w.writeAll("--ext needs an extension, e.g. --ext md\n");
                    return 2;
                };
                try ext_list.append(gpa, v);
            } else if (std.mem.eql(u8, a, "--include")) {
                const v = args.next() orelse {
                    try w.writeAll("--include needs a glob, e.g. --include '*/memory/*.md'\n");
                    return 2;
                };
                try include_list.append(gpa, v);
            } else if (std.mem.eql(u8, a, "--model")) {
                opts.model = args.next();
            } else if (std.mem.eql(u8, a, "--force")) {
                opts.force = true;
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        opts.roots = roots_list.items;
        opts.extensions = ext_list.items;
        opts.include = include_list.items;
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
            } else if (std.mem.eql(u8, a, "--path")) {
                const v = args.next() orelse {
                    try w.writeAll("--path needs a glob, e.g. --path 'agents/handoffs/**'\n");
                    return 2;
                };
                opts.path = v;
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

    if (std.mem.eql(u8, cmd, "query")) {
        var question: ?[]const u8 = null;
        var opts: query_cmd.Options = .{ .query = "" };
        while (args.next()) |a| {
            if (!std.mem.startsWith(u8, a, "-")) {
                if (question == null) question = a else {
                    try w.print("unexpected extra argument: {s}\n(quote a multi-word question)\n", .{a});
                    return 2;
                }
            } else if (std.mem.eql(u8, a, "--budget")) {
                opts.budget = std.fmt.parseInt(usize, args.next() orelse "8000", 10) catch 8000;
            } else if (std.mem.eql(u8, a, "--neighbors")) {
                opts.neighbors = std.fmt.parseInt(i64, args.next() orelse "1", 10) catch 1;
            } else if (std.mem.eql(u8, a, "--candidates")) {
                opts.candidates = std.fmt.parseInt(usize, args.next() orelse "30", 10) catch 30;
            } else if (std.mem.eql(u8, a, "--format")) {
                const v = args.next() orelse "markdown";
                opts.format = std.meta.stringToEnum(query_cmd.Format, v) orelse {
                    try w.print("unknown format: {s} (markdown|json)\n", .{v});
                    return 2;
                };
            } else if (std.mem.eql(u8, a, "--collection")) {
                opts.collection = args.next();
            } else if (std.mem.eql(u8, a, "--path")) {
                opts.path = args.next() orelse {
                    try w.writeAll("--path needs a glob, e.g. --path 'projects/qlit/**'\n");
                    return 2;
                };
            } else if (std.mem.eql(u8, a, "--model")) {
                opts.model = args.next();
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        opts.query = question orelse {
            try w.writeAll("usage: zkb query <question> [--budget N] [--collection NAME] [--path GLOB]\n");
            return 2;
        };
        return query_cmd.run(gpa, init.io, init.environ_map, w, opts);
    }

    if (std.mem.eql(u8, cmd, "remember")) {
        var opts: memory_cmd.RememberOptions = .{ .body = "" };
        var body: ?[]const u8 = null;
        while (args.next()) |a| {
            if (!std.mem.startsWith(u8, a, "--")) {
                if (body == null) body = a else {
                    try w.writeAll("one memory per call; quote the whole text\n");
                    return 2;
                }
            } else if (std.mem.eql(u8, a, "--type")) {
                const v = args.next() orelse "";
                opts.type = zkb.memory.Type.parse(v) orelse {
                    try w.print("unknown type: {s} (user|feedback|decision|project|reference)\n", .{v});
                    return 2;
                };
            } else if (std.mem.eql(u8, a, "--scope")) {
                opts.scope = args.next() orelse "";
            } else if (std.mem.eql(u8, a, "--subjects")) {
                opts.subjects = args.next() orelse "";
            } else if (std.mem.eql(u8, a, "--refs")) {
                opts.refs = args.next() orelse "";
            } else if (std.mem.eql(u8, a, "--source")) {
                opts.source = args.next() orelse "claude-code";
            } else if (std.mem.eql(u8, a, "--force")) {
                opts.force = true;
            } else if (std.mem.eql(u8, a, "--model")) {
                opts.model = args.next();
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        opts.body = body orelse {
            try w.writeAll("usage: zkb remember <text> [--type T] [--subjects a,b] [--force]\n");
            return 2;
        };
        return memory_cmd.remember(gpa, init.io, init.environ_map, w, opts);
    }

    if (std.mem.eql(u8, cmd, "forget")) {
        const target = args.next() orelse {
            try w.writeAll("usage: zkb forget <memory-file>\n");
            return 2;
        };
        return memory_cmd.forget(gpa, init.io, init.environ_map, w, target);
    }

    if (std.mem.eql(u8, cmd, "rescope")) {
        const target = args.next() orelse {
            try w.writeAll("usage: zkb rescope <memory-file> <scope>   (empty scope makes it universal)\n");
            return 2;
        };
        const scope = args.next() orelse {
            try w.writeAll("usage: zkb rescope <memory-file> <scope>   (empty scope makes it universal)\n");
            return 2;
        };
        return memory_cmd.rescope(gpa, init.io, init.environ_map, w, target, scope, null);
    }

    if (std.mem.eql(u8, cmd, "recall")) {
        var opts: memory_cmd.RecallOptions = .{};
        var query: ?[]const u8 = null;
        while (args.next()) |a| {
            if (!std.mem.startsWith(u8, a, "--")) {
                if (query == null) query = a;
            } else if (std.mem.eql(u8, a, "--budget")) {
                opts.budget = std.fmt.parseInt(usize, args.next() orelse "1500", 10) catch 1500;
            } else if (std.mem.eql(u8, a, "--scope")) {
                opts.scope = args.next();
            } else if (std.mem.eql(u8, a, "--json")) {
                opts.format = .json;
            } else if (std.mem.eql(u8, a, "--model")) {
                opts.model = args.next();
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        opts.query = query orelse "";
        return memory_cmd.recall(gpa, init.io, init.environ_map, w, opts);
    }

    if (std.mem.eql(u8, cmd, "facts")) {
        var key: ?[]const u8 = null;
        var want_history = false;
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--history")) {
                want_history = true;
            } else if (!std.mem.startsWith(u8, a, "--")) {
                if (key == null) key = a;
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        return memory_cmd.factsCmd(gpa, init.io, init.environ_map, w, key, want_history);
    }

    if (std.mem.eql(u8, cmd, "remember-fact")) {
        var positional: [2]?[]const u8 = .{ null, null };
        var n: usize = 0;
        var at: ?[]const u8 = null;
        var note: []const u8 = "";
        var scope: []const u8 = "";
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--at")) {
                at = args.next();
            } else if (std.mem.eql(u8, a, "--note")) {
                note = args.next() orelse "";
            } else if (std.mem.eql(u8, a, "--scope")) {
                scope = args.next() orelse "";
            } else if (!std.mem.startsWith(u8, a, "--")) {
                if (n < positional.len) {
                    positional[n] = a;
                    n += 1;
                }
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        if (n != 2) {
            try w.writeAll("usage: zkb remember-fact <key> <value> [--at YYYY-MM-DD] [--note TEXT] [--scope NAME]\n");
            return 2;
        }
        return memory_cmd.rememberFact(
            gpa,
            init.io,
            init.environ_map,
            w,
            positional[0].?,
            positional[1].?,
            at,
            note,
            scope,
        );
    }

    if (std.mem.eql(u8, cmd, "records")) {
        var opts: records_cmd.Options = .{};
        while (args.next()) |a| {
            if (!std.mem.startsWith(u8, a, "-")) {
                if (opts.type_name == null) opts.type_name = a else {
                    try w.print("unexpected extra argument: {s}\n", .{a});
                    return 2;
                }
            } else if (std.mem.eql(u8, a, "--where")) {
                opts.where = args.next();
            } else if (std.mem.eql(u8, a, "--search")) {
                opts.search = args.next();
            } else if (std.mem.eql(u8, a, "--agg")) {
                opts.agg = args.next();
            } else if (std.mem.eql(u8, a, "--window")) {
                opts.window = args.next();
            } else if (std.mem.eql(u8, a, "--schema")) {
                opts.show_schema = true;
            } else if (std.mem.eql(u8, a, "--json")) {
                opts.json = true;
            } else if (std.mem.eql(u8, a, "--model")) {
                opts.model = args.next();
            } else if (std.mem.eql(u8, a, "-n")) {
                opts.limit = std.fmt.parseInt(usize, args.next() orelse "50", 10) catch 50;
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        return records_cmd.run(gpa, init.io, init.environ_map, w, opts);
    }

    if (std.mem.eql(u8, cmd, "sql")) {
        var query: ?[]const u8 = null;
        var opts: records_cmd.SqlOptions = .{};
        var want_list = false;
        var want_history = false;
        var limit: usize = 20;
        // `k=v` pairs for a saved query. Collected rather than substituted: they
        // are bound through sqlite, so a value containing a quote is a value,
        // not a second statement.
        var kvs: std.ArrayList([2][]const u8) = .empty;
        defer kvs.deinit(gpa);

        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--json")) {
                opts.json = true;
            } else if (std.mem.eql(u8, a, "--list")) {
                want_list = true;
            } else if (std.mem.eql(u8, a, "--history")) {
                want_history = true;
            } else if (std.mem.eql(u8, a, "-n")) {
                limit = std.fmt.parseInt(usize, args.next() orelse "20", 10) catch 20;
            } else if (query != null and std.mem.indexOfScalar(u8, a, '=') != null and
                !std.mem.startsWith(u8, a, "-"))
            {
                const eq = std.mem.indexOfScalar(u8, a, '=').?;
                try kvs.append(gpa, .{ a[0..eq], a[eq + 1 ..] });
            } else if (query == null) {
                query = a;
            } else {
                try w.writeAll("quote the whole statement as one argument\n");
                return 2;
            }
        }

        if (want_list) return records_cmd.sqlList(gpa, init.io, init.environ_map, w);
        if (want_history) return records_cmd.sqlHistory(gpa, init.io, init.environ_map, w, limit);

        const q = query orelse {
            try w.writeAll(
                \\usage: zkb sql "select ..." [--json]
                \\       zkb sql @<name> [key=value ...] [--json]   run a saved query
                \\       zkb sql --list                             saved queries
                \\       zkb sql --history [-n N]                   statements typed before
                \\
            );
            return 2;
        };
        opts.args = kvs.items;
        return records_cmd.sqlCmd(gpa, init.io, init.environ_map, w, q, opts);
    }

    if (std.mem.eql(u8, cmd, "skill")) {
        return skill_cmd.run(gpa, init.io, init.environ_map, w);
    }

    if (std.mem.eql(u8, cmd, "collection")) {
        const coll_usage = "usage: zkb collection rm NAME\n" ++
            "       zkb collection checks NAME --off a,b\n";
        const sub = args.next() orelse {
            try w.writeAll(coll_usage);
            return 2;
        };
        const is_rm = std.mem.eql(u8, sub, "rm");
        if (!is_rm and !std.mem.eql(u8, sub, "checks")) {
            try w.print("unknown collection subcommand: {s}\n", .{sub});
            return 2;
        }
        const name = args.next() orelse {
            try w.writeAll(coll_usage);
            return 2;
        };
        if (is_rm) return collection_cmd.rm(gpa, init.io, init.environ_map, w, .{ .name = name });

        // No `--off` at all clears the list rather than erroring: "this corpus
        // answers to everything its kind allows" has to be sayable, and it is the
        // only way back from an opt-out entered by mistake.
        var off: []const u8 = "";
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--off")) {
                off = args.next() orelse "";
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        return collection_cmd.checks(gpa, init.io, init.environ_map, w, name, off);
    }

    if (std.mem.eql(u8, cmd, "status")) {
        return status(gpa, init.io, init.environ_map, w);
    }

    if (std.mem.eql(u8, cmd, "maintain")) {
        var since_last = false;
        var as_json = false;
        var selected: [8]zkb.maintain.Check = undefined;
        var n_selected: usize = 0;
        var use_all = false;
        var only: []const u8 = "";
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--since")) {
                const v = args.next() orelse "last";
                since_last = std.mem.eql(u8, v, "last");
            } else if (std.mem.eql(u8, a, "--json")) {
                as_json = true;
            } else if (std.mem.eql(u8, a, "--all")) {
                use_all = true;
            } else if (std.mem.eql(u8, a, "--collection")) {
                only = args.next() orelse "";
            } else if (std.mem.eql(u8, a, "--check")) {
                const v = args.next() orelse "";
                const c = zkb.maintain.Check.parse(v) orelse {
                    try w.print("unknown check: {s}\n", .{v});
                    return 2;
                };
                if (n_selected < selected.len) {
                    selected[n_selected] = c;
                    n_selected += 1;
                }
            } else {
                try w.print("unknown option: {s}\n", .{a});
                return 2;
            }
        }
        const checks: []const zkb.maintain.Check = if (n_selected != 0)
            selected[0..n_selected]
        else if (use_all)
            zkb.maintain.Check.all()
        else
            zkb.maintain.Check.default();
        return maintainCmd(gpa, init.io, init.environ_map, w, since_last, as_json, checks, only);
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

/// Print a stored newline-separated list on one line.
///
/// A shell glob over forty projects stores forty patterns, each as long as a
/// directory name. Listing three of them and a count is worse than useless — the
/// three are arbitrary, and the line still buries the collection counts this
/// command exists to show. Past a handful, the count alone is the information.
fn printList(w: *std.Io.Writer, label: []const u8, stored: []const u8, noun: []const u8) !void {
    const max_shown = 3;
    var total: usize = 0;
    var it = std.mem.splitScalar(u8, stored, '\n');
    while (it.next()) |part| if (part.len != 0) {
        total += 1;
    };

    try w.writeAll(label);
    if (total > max_shown) {
        try w.print("{d} {s}s", .{ total, noun });
        return;
    }
    var first = true;
    it = std.mem.splitScalar(u8, stored, '\n');
    while (it.next()) |part| {
        if (part.len == 0) continue;
        if (!first) try w.writeAll(", ");
        try w.writeAll(part);
        first = false;
    }
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
            \\       COALESCE(sum(d.chunk_count), 0),
            \\       COALESCE(col.extensions, ''), COALESCE(col.include, '')
            \\FROM collections col LEFT JOIN docs d ON d.collection_id = col.id
            \\GROUP BY col.id ORDER BY col.id
        );
        defer st.finalize();
        while (try st.step()) {
            try w.print("  {s}  {d} docs, {d} chunks  ({s})\n", .{
                st.columnText(0), st.columnI64(2), st.columnI64(3), st.columnText(1),
            });
            // Without this, "why does this collection only have three documents"
            // has no answer anywhere in the CLI — the filters that produced the
            // count would be invisible while the count itself is right there.
            const exts = st.columnText(4);
            const include = st.columnText(5);
            if (exts.len != 0 or include.len != 0) {
                try w.writeAll("      only");
                if (exts.len != 0) try printList(w, " ", exts, "extension");
                if (include.len != 0) try printList(w, " matching ", include, "pattern");
                try w.writeAll("\n");
            }
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

/// `zkb maintain` runs against the database directly rather than through the
/// daemon: it is a read-only sweep, and recording the run needs a write the
/// daemon's single-writer invariant would otherwise have to arbitrate.
fn maintainCmd(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *std.Io.Writer,
    since_last: bool,
    as_json: bool,
    checks: []const zkb.maintain.Check,
    only: []const u8,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    std.Io.Dir.accessAbsolute(io, layout.db, .{}) catch {
        try w.print("no index at {s}\nrun: zkb index\n", .{layout.db});
        return 3;
    };
    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = zkb.store.open(db_path, .read_write) catch |err| switch (err) {
        error.SchemaStale => {
            try w.writeAll("index schema is out of date\nrun: zkb index\n");
            return 3;
        },
        else => return err,
    };
    defer db.close();

    var only_id: i64 = 0;
    if (only.len != 0) {
        var s = zkb.store.Store.init(&db);
        only_id = (try s.findCollection(only)) orelse {
            try w.print("no such collection: {s}\n", .{only});
            try w.writeAll("zkb status lists them\n");
            return 3;
        };
    }

    const now_ms: i64 = @intCast(@divTrunc(std.Io.Timestamp.now(io, .real).nanoseconds, std.time.ns_per_ms));
    var report = try zkb.maintain.run(gpa, &db, .{
        .checks = checks,
        .now_ms = now_ms,
        .only_collection = only_id,
    });
    defer report.deinit(gpa);

    if (as_json) {
        try w.writeAll("{\"findings\":[");
        for (report.findings, 0..) |f, i| {
            if (i != 0) try w.writeAll(",");
            try w.print("{{\"check\":\"{t}\",\"path\":", .{f.check});
            try std.json.Stringify.value(f.path, .{}, w);
            try w.writeAll(",\"detail\":");
            try std.json.Stringify.value(f.detail, .{}, w);
            try w.writeAll("}");
        }
        try w.writeAll("]}\n");
        return 0;
    }

    if (report.link_graph_empty) {
        // Otherwise every document looks unlinked and the report is a lie.
        try w.writeAll(
            "note: the link graph is empty, so link checks were skipped.\n" ++
                "      run `zkb index --force` once to populate it.\n\n",
        );
    }

    if (since_last) {
        var diff = try zkb.maintain.diffAgainstLast(gpa, &db, &report);
        defer diff.deinit(gpa);
        try w.print("new ({d})\n", .{diff.new_keys.len});
        for (report.findings) |f| {
            for (diff.new_keys) |k| if (std.mem.eql(u8, k, f.key)) {
                try w.print("  {t:<16} {s}  {s}\n", .{ f.check, f.path, f.detail });
            };
        }
        if (diff.resolved_keys.len != 0) {
            try w.print("\nresolved ({d})\n", .{diff.resolved_keys.len});
            for (diff.resolved_keys) |k| try w.print("  {s}\n", .{k});
        }
        try w.print("\nunchanged: {d}\n", .{diff.unchanged});
    } else {
        for (checks) |c| {
            const n = report.count(c);
            if (n == 0) continue;
            try w.print("\n{t} ({d})\n", .{ c, n });
            var shown: usize = 0;
            for (report.findings) |f| {
                if (f.check != c) continue;
                if (shown == 15) {
                    try w.print("  ... {d} more\n", .{n - shown});
                    break;
                }
                try w.print("  {s}  {s}\n", .{ f.path, f.detail });
                shown += 1;
            }
        }
        if (report.findings.len == 0) try w.writeAll("no findings\n");
    }

    try zkb.maintain.record(gpa, &db, &report, checks, now_ms);
    return 0;
}

// Pull every cli-side file into the test build.
//
// Zig only collects tests from the root source file; other files need an
// explicit reference like this one, which is why tests/root.zig has the same
// block. Without it a `test` in any file here compiles to nothing and passes by
// not existing.
test {
    _ = @import("cli/doctor.zig");
    _ = @import("cli/model.zig");
    _ = @import("cli/index_cmd.zig");
    _ = @import("cli/search_cmd.zig");
    _ = @import("cli/daemon_cmd.zig");
    _ = @import("cli/query_cmd.zig");
    _ = @import("cli/memory_cmd.zig");
    _ = @import("cli/records_cmd.zig");
    _ = @import("cli/skill_cmd.zig");
}
