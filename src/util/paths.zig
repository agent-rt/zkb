//! zkb's on-disk layout. All state lives under one directory so it can be
//! deleted wholesale — the index is derived data (SPEC §14.1).

const std = @import("std");

pub const Layout = struct {
    /// ~/.zkb
    root: []const u8,
    db: []const u8,
    sock: []const u8,
    pid: []const u8,
    log: []const u8,
    models: []const u8,

    pub fn deinit(self: *Layout, gpa: std.mem.Allocator) void {
        gpa.free(self.root);
        gpa.free(self.db);
        gpa.free(self.sock);
        gpa.free(self.pid);
        gpa.free(self.log);
        gpa.free(self.models);
        self.* = undefined;
    }
};

pub const Error = error{ NoHomeDirectory, OutOfMemory };

/// Resolve the layout from $ZKB_HOME, else $HOME/.zkb. The environment is
/// passed in rather than read globally — Zig 0.16 has no ambient getenv, and
/// injecting it keeps this testable.
pub fn resolve(gpa: std.mem.Allocator, env: *const std.process.Environ.Map) Error!Layout {
    const root = if (env.get("ZKB_HOME")) |z|
        try gpa.dupe(u8, z)
    else blk: {
        const home = env.get("HOME") orelse return error.NoHomeDirectory;
        break :blk try std.fmt.allocPrint(gpa, "{s}/.zkb", .{home});
    };
    errdefer gpa.free(root);

    return .{
        .root = root,
        .db = try std.fmt.allocPrint(gpa, "{s}/zkb.db", .{root}),
        .sock = try std.fmt.allocPrint(gpa, "{s}/zkb.sock", .{root}),
        .pid = try std.fmt.allocPrint(gpa, "{s}/zkb.pid", .{root}),
        .log = try std.fmt.allocPrint(gpa, "{s}/zkb.log", .{root}),
        .models = try std.fmt.allocPrint(gpa, "{s}/models", .{root}),
    };
}
