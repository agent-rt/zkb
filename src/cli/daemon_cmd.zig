//! `zkb daemon start | stop | status | run | install | uninstall`
//!
//! `start` spawns `zkb daemon run` and waits for the socket to accept, then
//! returns. Waiting for the socket rather than sleeping a fixed time is what
//! makes "started" mean "ready to serve" — and it is why the model must load
//! lazily, since the wait would otherwise include 1-2s of GGUF loading.

const std = @import("std");
const zkb = @import("zkb");
const proto = zkb.proto;
const client = zkb.ipc_client;

const Writer = std.Io.Writer;

pub const Options = struct {
    preload_model: bool = false,
    root: ?[]const u8 = null,
    collection: []const u8 = "docs",
    model: ?[]const u8 = null,
    scan_interval_s: u64 = 30,
};

/// How long `start` waits for the socket. The target is under 200ms; this is the
/// give-up point, not the expectation.
const ready_timeout_ms: u64 = 5_000;

pub fn start(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    opts: Options,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);
    // The log below is opened before the daemon is spawned, so run/ has to
    // exist first. Every other writing command already does this; without it a
    // first `zkb daemon start` on a machine that has never indexed anything
    // died with a bare `error: FileNotFound` and created nothing.
    try layout.ensureDirs(io);

    if (try isRunning(gpa, io, layout.sock)) {
        try w.writeAll("daemon is already running\n");
        return 0;
    }

    // Resolved here as well as in the daemon, so the most likely first-run
    // problem is reported by the command the caller ran rather than discovered
    // as a five second readiness timeout and a log file holding one word. The
    // decision itself is not duplicated — `resolveForDaemon` owns it, and this
    // side only prints what it returns.
    switch (try zkb.model_registry.resolveForDaemon(gpa, io, env, &layout, opts.model, .q8_0)) {
        .ready => |found| {
            var f = found;
            f.deinit(gpa);
        },
        // A warning, not a refusal: keyword search, scanning and maintenance need
        // no model, and refusing here would withhold all of them over the one
        // thing that is missing.
        .degraded => |reason| try w.print("warning: {s}\n", .{reason}),
    }

    const exe = try zkb.paths.selfExe(gpa, io);
    defer gpa.free(exe);

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.appendSlice(gpa, &.{ exe, "daemon", "run" });
    if (opts.preload_model) try argv.append(gpa, "--preload");
    if (opts.root) |r| try argv.appendSlice(gpa, &.{ "--root", r });
    if (opts.model) |m| try argv.appendSlice(gpa, &.{ "--model", m });

    // stdio goes to the log: the daemon outlives this shell, so its output has to
    // land somewhere inspectable. Opened without truncating so a crash loop
    // leaves a history rather than only its last attempt.
    const log = try std.Io.Dir.createFileAbsolute(io, layout.log, .{ .truncate = false });
    defer log.close(io);
    var child = try std.process.spawn(io, .{
        .argv = argv.items,
        .stdin = .ignore,
        .stdout = .{ .file = log },
        .stderr = .{ .file = log },
    });
    // Not waited on: the child is the daemon and is expected to outlive us. The
    // readiness probe below is what tells us it came up.
    _ = &child;

    const started = nowMs(io);
    while (nowMs(io) - started < ready_timeout_ms) {
        if (try isRunning(gpa, io, layout.sock)) {
            try w.print("daemon started in {d}ms\n", .{nowMs(io) - started});
            return 0;
        }
        std.Io.sleep(io, .{ .nanoseconds = 5 * std.time.ns_per_ms }, .awake) catch {};
    }

    try w.print("daemon did not become ready within {d}ms\nsee {s}\n", .{
        ready_timeout_ms, layout.log,
    });
    return 1;
}

pub fn stop(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    var c = client.Client.connect(io, layout.sock) catch |err| switch (err) {
        // See the note in status: a stale socket file means the daemon died
        // without cleaning up, which is still "not running".
        error.DaemonNotRunning, error.ConnectFailed => {
            try w.writeAll("daemon is not running\n");
            return 0;
        },
        else => return err,
    };
    defer c.close();

    var resp = c.call(gpa, .shutdown, "{}") catch |err| {
        try w.print("could not reach the daemon: {t}\n", .{err});
        try w.print("if it is wedged: pkill -f 'zkb daemon run'\n", .{});
        return 1;
    };
    defer resp.deinit(gpa);
    if (!resp.ok) {
        try w.print("shutdown failed: {s}\n", .{resp.message});
        return 1;
    }

    // Wait for the socket to actually go away: the daemon still has to join its
    // threads and checkpoint the WAL, and reporting "stopped" before that would
    // make an immediate restart race the old process.
    const started = nowMs(io);
    while (nowMs(io) - started < ready_timeout_ms) {
        if (!try isRunning(gpa, io, layout.sock)) {
            try w.print("daemon stopped in {d}ms\n", .{nowMs(io) - started});
            return 0;
        }
        std.Io.sleep(io, .{ .nanoseconds = 5 * std.time.ns_per_ms }, .awake) catch {};
    }
    try w.writeAll("daemon did not exit within the timeout\n");
    return 1;
}

pub fn status(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    var c = client.Client.connect(io, layout.sock) catch |err| switch (err) {
        // Two ways to not be running, and only one of them removes the socket
        // file. A daemon that was killed rather than asked to stop — SIGKILL,
        // `launchctl bootout`, a crash — leaves the file behind, so connecting
        // fails instead of finding nothing. isRunning has always handled both;
        // here only the first was covered, which turned the command you reach
        // for to diagnose a dead daemon into `error: ConnectFailed`.
        error.DaemonNotRunning, error.ConnectFailed => {
            try client.reportNotRunning(w, layout.sock);
            return 3;
        },
        else => return err,
    };
    defer c.close();

    var health = c.call(gpa, .health, "{}") catch |err| {
        try w.print("daemon is not answering: {t}\n", .{err});
        return 1;
    };
    defer health.deinit(gpa);
    var stats = c.call(gpa, .stats, "{}") catch |err| {
        try w.print("daemon answered health but not stats: {t}\n", .{err});
        return 1;
    };
    defer stats.deinit(gpa);

    if (health.ok) {
        const r = health.result.?.object;
        try w.print("daemon   running, up {d}ms\n", .{intOf(r, "uptime_ms")});
        try w.print("model    {s}\n", .{if (boolOf(r, "model_loaded")) "loaded" else "not loaded (lazy)"});
        if (r.get("degraded")) |d| if (d == .string) {
            try w.print("DEGRADED {s}\n", .{d.string});
        };
        if (r.get("queue")) |q| if (q == .object) {
            try w.print("queue    {d} interactive, {d} ingest\n", .{
                intOf(q.object, "interactive"), intOf(q.object, "ingest"),
            });
        };
    }
    if (stats.ok) {
        const r = stats.result.?.object;
        try w.print("index    {d} docs, {d} chunks", .{ intOf(r, "docs"), intOf(r, "chunks") });
        const pending = intOf(r, "pending");
        const failed = intOf(r, "failed");
        if (pending != 0) try w.print(", {d} pending", .{pending});
        if (failed != 0) try w.print(", {d} FAILED", .{failed});
        try w.writeAll("\n");
        // Evidence that the priority queue actually preempted something, rather
        // than merely being present.
        try w.print("embed    {d} interactive, {d} ingest served; max preempted {d}\n", .{
            intOf(r, "served_interactive"), intOf(r, "served_ingest"), intOf(r, "max_preempted"),
        });
        if (boolOf(r, "drift")) {
            try w.writeAll("\nWARNING index drift between chunks / fts / vec\n");
            return 1;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// launchd
// ---------------------------------------------------------------------------

/// Reverse DNS of a namespace that actually exists and is actually ours.
///
/// Was `io.zkb.daemon`, which claims `zkb.io` — a domain nobody here owns. The
/// GitHub form is the one with a real owner: `agent-rt.github.io` is the Pages
/// namespace that comes with the org, so its reverse is `io.github.agent-rt`.
/// Note it is *not* `com.github.agent-rt`: that would reverse to
/// `agent-rt.github.com`, which does not exist.
///
/// `.daemon` stays on the end because a project can grow a second agent, and a
/// label that names the role is cheaper than renaming again later.
const plist_label = "io.github.agent-rt.zkb.daemon";

/// The label used up to 0.0.15.
///
/// Kept so `install` can unload it and `uninstall` can clean it up. Without that,
/// upgrading writes a second plist and leaves the first loaded — two launchd
/// agents both running `zkb daemon run` against one index. That used to mean two
/// writers; `daemon run` now refuses, so it means the new agent fails to start
/// and launchd retries it forever. Either way the fix belongs in `install`.
const legacy_plist_label = "io.zkb.daemon";

pub fn install(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
) !u8 {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    const home = env.get("HOME") orelse return error.NoHomeDirectory;
    const dir = try std.fmt.allocPrint(gpa, "{s}/Library/LaunchAgents", .{home});
    defer gpa.free(dir);
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.plist", .{ dir, plist_label });
    defer gpa.free(path);

    std.Io.Dir.createDirPath(.cwd(), io, dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const exe = try zkb.paths.selfExe(gpa, io);
    defer gpa.free(exe);

    // `daemon run` rather than `daemon start`: launchd tracks the process it
    // spawns, so forking would make KeepAlive watch a parent that exits at once.
    const plist = try std.fmt.allocPrint(gpa,
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        \\<plist version="1.0">
        \\<dict>
        \\  <key>Label</key><string>{s}</string>
        \\  <key>ProgramArguments</key>
        \\  <array>
        \\    <string>{s}</string>
        \\    <string>daemon</string>
        \\    <string>run</string>
        \\  </array>
        \\  <key>RunAtLoad</key><true/>
        \\  <key>KeepAlive</key>
        \\  <dict><key>SuccessfulExit</key><false/></dict>
        \\  <key>StandardOutPath</key><string>{s}</string>
        \\  <key>StandardErrorPath</key><string>{s}</string>
        \\  <key>ProcessType</key><string>Background</string>
        \\</dict>
        \\</plist>
        \\
    , .{ plist_label, exe, layout.log, layout.log });
    defer gpa.free(plist);

    var file = try std.Io.Dir.createFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var writer = file.writer(io, &buf);
    try writer.interface.writeAll(plist);
    try writer.interface.flush();

    try w.print("wrote {s}\n", .{path});

    // Done after the new plist is on disk, so a failure here leaves the machine
    // with a working agent rather than none.
    if (try removeLegacyPlist(gpa, io, home)) {
        try w.print("removed the old {s}.plist\n", .{legacy_plist_label});
        try w.print("unload it too: launchctl bootout gui/$(id -u)/{s}\n", .{legacy_plist_label});
    }
    try w.print("load with: launchctl bootstrap gui/$(id -u) {s}\n", .{path});
    return 0;
}

pub fn uninstall(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
) !u8 {
    const home = env.get("HOME") orelse return error.NoHomeDirectory;
    const path = try std.fmt.allocPrint(gpa, "{s}/Library/LaunchAgents/{s}.plist", .{ home, plist_label });
    defer gpa.free(path);

    var removed = false;
    if (std.Io.Dir.deleteFileAbsolute(io, path)) |_| {
        try w.print("removed {s}\n", .{path});
        try w.print("unload with: launchctl bootout gui/$(id -u)/{s}\n", .{plist_label});
        removed = true;
    } else |_| {}

    // A machine that upgraded but never re-ran `install` still has only the old
    // one, so uninstall has to know about it or it silently leaves it behind.
    if (try removeLegacyPlist(gpa, io, home)) {
        try w.print("removed the old {s}.plist\n", .{legacy_plist_label});
        try w.print("unload with: launchctl bootout gui/$(id -u)/{s}\n", .{legacy_plist_label});
        removed = true;
    }

    if (!removed) try w.writeAll("no LaunchAgent plist to remove\n");
    return 0;
}

/// Delete the pre-0.0.16 plist if present. Returns whether it was there.
///
/// Only the file is removed; unloading needs `launchctl`, which is the caller's
/// to print rather than ours to run — spawning launchctl from a CLI that may be
/// running under launchd itself is a good way to fight the supervisor.
fn removeLegacyPlist(gpa: std.mem.Allocator, io: std.Io, home: []const u8) !bool {
    const legacy = try std.fmt.allocPrint(gpa, "{s}/Library/LaunchAgents/{s}.plist", .{ home, legacy_plist_label });
    defer gpa.free(legacy);
    std.Io.Dir.deleteFileAbsolute(io, legacy) catch return false;
    return true;
}

// ---------------------------------------------------------------------------

/// A connect attempt is the only honest liveness check: a pid file can be stale
/// and a socket file can outlive the process that made it.
fn isRunning(gpa: std.mem.Allocator, io: std.Io, sock: []const u8) !bool {
    var c = client.Client.connect(io, sock) catch |err| switch (err) {
        error.DaemonNotRunning, error.ConnectFailed => return false,
        else => return err,
    };
    defer c.close();
    var resp = c.call(gpa, .health, "{}") catch return false;
    defer resp.deinit(gpa);
    return resp.ok;
}

fn nowMs(io: std.Io) i64 {
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .real).nanoseconds, std.time.ns_per_ms));
}

fn intOf(obj: std.json.ObjectMap, key: []const u8) i64 {
    const v = obj.get(key) orelse return 0;
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => 0,
    };
}

fn boolOf(obj: std.json.ObjectMap, key: []const u8) bool {
    const v = obj.get(key) orelse return false;
    return switch (v) {
        .bool => |b| b,
        else => false,
    };
}
