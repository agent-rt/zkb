//! macOS FSEvents: wake the ingest loop when a watched directory changes.
//!
//! **This accelerates the poll, it does not replace it.** The scan loop keeps
//! running on its interval regardless. FSEvents can miss things — a stream can
//! fail to start, a network mount may not report, the process may be denied
//! access to a directory — and every one of those failures is silent. A watcher
//! that is only ever an optimisation degrades to "up to 30 s late"; a watcher
//! that is load-bearing degrades to "your edit is never indexed", and the user
//! has no way to tell which.
//!
//! The callback does one thing: set a flag. No allocation, no database, no
//! locking beyond an atomic store — it runs on a CoreFoundation run loop thread
//! that must not block.
//!
//! Declared by hand rather than through `@cImport("CoreServices/...")`: that
//! header pulls in the whole of Carbon, and translate-c fails on it. Six symbols
//! need no header.

const std = @import("std");
const builtin = @import("builtin");

pub const supported = builtin.os.tag == .macos;

const CFAllocatorRef = ?*anyopaque;
const CFStringRef = ?*const anyopaque;
const CFArrayRef = ?*const anyopaque;
const CFRunLoopRef = ?*anyopaque;
const FSEventStreamRef = ?*anyopaque;
const CFIndex = c_long;
const CFTimeInterval = f64;
const FSEventStreamEventId = u64;

const kCFStringEncodingUTF8: u32 = 0x08000100;
/// Start from now; there is no persisted event id to resume from, and the scan
/// loop covers whatever happened while the daemon was down.
const kFSEventStreamEventIdSinceNow: FSEventStreamEventId = 0xFFFFFFFFFFFFFFFF;
/// Deliver the first event of a burst immediately rather than after `latency`.
const kFSEventStreamCreateFlagNoDefer: u32 = 0x00000002;
const kFSEventStreamCreateFlagWatchRoot: u32 = 0x00000004;

const FSEventStreamContext = extern struct {
    version: CFIndex = 0,
    info: ?*anyopaque = null,
    retain: ?*const anyopaque = null,
    release: ?*const anyopaque = null,
    copyDescription: ?*const anyopaque = null,
};

const Callback = *const fn (
    stream: FSEventStreamRef,
    info: ?*anyopaque,
    num_events: usize,
    paths: ?*anyopaque,
    flags: [*]const u32,
    ids: [*]const FSEventStreamEventId,
) callconv(.c) void;

extern "c" fn CFStringCreateWithCString(CFAllocatorRef, [*:0]const u8, u32) CFStringRef;
extern "c" fn CFArrayCreate(CFAllocatorRef, [*]const ?*const anyopaque, CFIndex, ?*const anyopaque) CFArrayRef;
extern "c" fn CFRelease(?*const anyopaque) void;
extern "c" fn CFRunLoopGetCurrent() CFRunLoopRef;
extern "c" fn CFRunLoopRun() void;
extern "c" fn FSEventStreamCreate(
    allocator: CFAllocatorRef,
    callback: Callback,
    context: ?*FSEventStreamContext,
    paths: CFArrayRef,
    since: FSEventStreamEventId,
    latency: CFTimeInterval,
    flags: u32,
) FSEventStreamRef;
extern "c" fn FSEventStreamScheduleWithRunLoop(FSEventStreamRef, CFRunLoopRef, CFStringRef) void;
extern "c" fn FSEventStreamStart(FSEventStreamRef) bool;
extern "c" fn FSEventStreamStop(FSEventStreamRef) void;
extern "c" fn FSEventStreamInvalidate(FSEventStreamRef) void;
extern "c" fn FSEventStreamRelease(FSEventStreamRef) void;

extern "c" const kCFTypeArrayCallBacks: anyopaque;
extern "c" const kCFRunLoopDefaultMode: CFStringRef;

/// Coalescing window. Editors write a file several times in a burst (temp file,
/// rename, attribute set); half a second turns that into one wake-up instead of
/// four, and it is far below any latency a person notices.
pub const latency_s: CFTimeInterval = 0.5;

pub const Watcher = struct {
    flag: *std.atomic.Value(bool),
    thread: ?std.Thread = null,

    /// Start watching `roots`. Returns without an error if the platform has no
    /// FSEvents or the stream cannot start — the caller's polling loop is the
    /// correctness guarantee, so a failed watcher is a missing optimisation and
    /// nothing more.
    pub fn start(
        gpa: std.mem.Allocator,
        roots: []const []const u8,
        flag: *std.atomic.Value(bool),
    ) !Watcher {
        var w: Watcher = .{ .flag = flag };
        if (!supported or roots.len == 0) return w;

        // The thread owns its copy: the caller's slices may be freed while the
        // run loop is still using them.
        const owned = try gpa.alloc([]u8, roots.len);
        var n: usize = 0;
        errdefer {
            for (owned[0..n]) |p| gpa.free(p);
            gpa.free(owned);
        }
        for (roots) |r| {
            owned[n] = try gpa.dupe(u8, r);
            n += 1;
        }

        const ctx = try gpa.create(ThreadCtx);
        errdefer gpa.destroy(ctx);
        ctx.* = .{ .gpa = gpa, .roots = owned, .flag = flag };

        w.thread = std.Thread.spawn(.{}, threadMain, .{ctx}) catch {
            for (owned) |p| gpa.free(p);
            gpa.free(owned);
            gpa.destroy(ctx);
            return w;
        };
        return w;
    }
};

const ThreadCtx = struct {
    gpa: std.mem.Allocator,
    roots: [][]u8,
    flag: *std.atomic.Value(bool),
};

fn threadMain(ctx: *ThreadCtx) void {
    // Never freed: the run loop below does not return, and the daemon exits by
    // process teardown. Freeing on a path that cannot be reached would only be
    // there to look tidy.
    var refs = ctx.gpa.alloc(?*const anyopaque, ctx.roots.len) catch return;
    var made: usize = 0;
    for (ctx.roots) |r| {
        const z = ctx.gpa.dupeZ(u8, r) catch break;
        defer ctx.gpa.free(z);
        const s = CFStringCreateWithCString(null, z.ptr, kCFStringEncodingUTF8) orelse break;
        refs[made] = s;
        made += 1;
    }
    if (made == 0) return;

    const array = CFArrayCreate(null, refs.ptr, @intCast(made), &kCFTypeArrayCallBacks) orelse return;
    var context: FSEventStreamContext = .{ .info = ctx.flag };

    const stream = FSEventStreamCreate(
        null,
        onEvents,
        &context,
        array,
        kFSEventStreamEventIdSinceNow,
        latency_s,
        kFSEventStreamCreateFlagNoDefer | kFSEventStreamCreateFlagWatchRoot,
    ) orelse return;

    FSEventStreamScheduleWithRunLoop(stream, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode);
    if (!FSEventStreamStart(stream)) return;

    CFRunLoopRun();
}

/// Runs on the run-loop thread. One atomic store, nothing else: the ingest
/// thread decides what a change means, and doing any of that here would block a
/// thread that must stay responsive.
fn onEvents(
    _: FSEventStreamRef,
    info: ?*anyopaque,
    _: usize,
    _: ?*anyopaque,
    _: [*]const u32,
    _: [*]const FSEventStreamEventId,
) callconv(.c) void {
    const flag: *std.atomic.Value(bool) = @ptrCast(@alignCast(info orelse return));
    flag.store(true, .release);
}
