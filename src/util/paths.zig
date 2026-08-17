//! zkb's on-disk layout — one directory, with the disposable/precious boundary
//! drawn as a directory boundary rather than a convention.
//!
//! ```
//! ~/.zkb/
//! ├── data/      memory/ facts.csv records/   ← the only irreplaceable thing
//! ├── index/     zkb.db                       ← rebuild: zkb index
//! ├── models/    *.gguf                       ← rebuild: zkb model pull
//! └── run/       zkb.sock zkb.pid zkb.log trace.jsonl
//! ```
//!
//! **Everything outside `data/` can be deleted at any time.** That is the whole
//! reason for the split: "delete the index and re-run" is the standard fix for
//! half the things that can go wrong, and it must not be a sentence that can
//! destroy a memory. A convention inside a flat directory would be violated the
//! first time anyone typed `rm -rf ~/.zkb`; a subdirectory cannot be.
//!
//! It also keeps backup granularity free. `data/` is about a megabyte a year and
//! irreplaceable; the rest is hundreds of megabytes and re-downloadable. Any
//! naive backup tool gets this right without an exclusion rule.

const std = @import("std");

pub const Layout = struct {
    /// ~/.zkb
    root: []const u8,

    /// The one directory whose loss is permanent: what zkb *writes*.
    ///
    /// Kept out of `~/docs` on purpose — memories are written several times per
    /// session, and mixing that commit rate into the documents repo would bury
    /// its history.
    data: []const u8,
    memory: []const u8,
    facts: []const u8,

    // ---- everything below is derived and disposable
    index_dir: []const u8,
    db: []const u8,
    models: []const u8,
    run_dir: []const u8,
    sock: []const u8,
    pid: []const u8,
    log: []const u8,
    /// Retrieval trace, written only when $ZKB_TRACE=1.
    trace: []const u8,

    pub fn deinit(self: *Layout, gpa: std.mem.Allocator) void {
        gpa.free(self.root);
        gpa.free(self.data);
        gpa.free(self.memory);
        gpa.free(self.facts);
        gpa.free(self.index_dir);
        gpa.free(self.db);
        gpa.free(self.models);
        gpa.free(self.run_dir);
        gpa.free(self.sock);
        gpa.free(self.pid);
        gpa.free(self.log);
        gpa.free(self.trace);
        self.* = undefined;
    }

    /// The directories zkb creates before writing. Called on every path that
    /// writes, because a missing directory is not an error worth surfacing to
    /// someone who just ran `zkb remember`.
    pub fn ensureDirs(self: *const Layout, io: std.Io) !void {
        for ([_][]const u8{ self.root, self.data, self.index_dir, self.models, self.run_dir }) |d| {
            std.Io.Dir.createDirPath(.cwd(), io, d) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };
        }
    }
};

pub const Error = error{ NoHomeDirectory, OutOfMemory };

/// Resolve the layout from $ZKB_HOME, else $HOME/.zkb. The environment is
/// passed in rather than read globally — Zig 0.16 has no ambient getenv, and
/// injecting it keeps this testable.
///
/// `$ZKB_DATA` moves just the data directory, for putting it in a synced folder
/// or its own repository without relocating 600 MB of model.
pub fn resolve(gpa: std.mem.Allocator, env: *const std.process.Environ.Map) Error!Layout {
    const root = if (env.get("ZKB_HOME")) |z|
        try gpa.dupe(u8, z)
    else blk: {
        const home = env.get("HOME") orelse return error.NoHomeDirectory;
        break :blk try std.fmt.allocPrint(gpa, "{s}/.zkb", .{home});
    };
    errdefer gpa.free(root);

    const data = if (env.get("ZKB_DATA")) |z|
        try gpa.dupe(u8, z)
    else
        try std.fmt.allocPrint(gpa, "{s}/data", .{root});
    errdefer gpa.free(data);

    const index_dir = try std.fmt.allocPrint(gpa, "{s}/index", .{root});
    errdefer gpa.free(index_dir);
    const run_dir = try std.fmt.allocPrint(gpa, "{s}/run", .{root});
    errdefer gpa.free(run_dir);

    return .{
        .root = root,
        // `data`, `index_dir` and `run_dir` are already owned here; duping them
        // again would leak the originals.
        .data = data,
        .memory = try std.fmt.allocPrint(gpa, "{s}/memory", .{data}),
        .facts = try std.fmt.allocPrint(gpa, "{s}/facts.csv", .{data}),
        .index_dir = index_dir,
        .db = try std.fmt.allocPrint(gpa, "{s}/zkb.db", .{index_dir}),
        .models = try std.fmt.allocPrint(gpa, "{s}/models", .{root}),
        .run_dir = run_dir,
        .sock = try std.fmt.allocPrint(gpa, "{s}/zkb.sock", .{run_dir}),
        .pid = try std.fmt.allocPrint(gpa, "{s}/zkb.pid", .{run_dir}),
        .log = try std.fmt.allocPrint(gpa, "{s}/zkb.log", .{run_dir}),
        .trace = try std.fmt.allocPrint(gpa, "{s}/trace.jsonl", .{run_dir}),
    };
}

extern "c" fn _NSGetExecutablePath(buf: [*]u8, bufsize: *u32) c_int;

/// Absolute path of the running executable.
///
/// Zig 0.16 dropped `std.fs.selfExePathAlloc`, and argv[0] is not a substitute:
/// it can be a bare name resolved through PATH, or a symlink. The daemon has to
/// re-spawn *itself*, so exactness matters more than portability here.
pub fn selfExe(gpa: std.mem.Allocator, io: std.Io) ![]u8 {
    switch (@import("builtin").os.tag) {
        .macos, .ios => {
            var buf: [4096]u8 = undefined;
            var size: u32 = buf.len;
            // Declared directly instead of @cInclude("mach-o/dyld.h"): that
            // header drags in mach message types whose translate-c static
            // assertions fail. One symbol needs no header.
            if (_NSGetExecutablePath(&buf, &size) != 0) return error.NameTooLong;
            const len = std.mem.indexOfScalar(u8, &buf, 0) orelse buf.len;
            return gpa.dupe(u8, buf[0..len]);
        },
        else => {
            var buf: [4096]u8 = undefined;
            const n = try std.Io.Dir.readLinkAbsolute(io, "/proc/self/exe", &buf);
            return gpa.dupe(u8, n);
        },
    }
}
