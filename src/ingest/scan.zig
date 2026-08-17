//! Reconcile a collection root with the docs table.
//!
//! Two-level change detection: (size, mtime) filters first, SHA-256 confirms
//! (SPEC §4.1). At 193 files the stat pass is microseconds and hashing 7 MB is
//! milliseconds, so neither level is load-bearing *yet* — but getting the order
//! right now means not revisiting it when the corpus is a hundred times larger.
//!
//! This stage does not parse Markdown and does not embed. It only makes the docs
//! table reflect the filesystem, leaving `indexed_at = NULL` on anything that
//! needs work. The table is therefore the work queue, which is what makes an
//! interrupted run resumable without separate bookkeeping.

const std = @import("std");
const store = @import("../db/store.zig");
const hash = @import("../util/hash.zig");

pub const Report = struct {
    seen: usize = 0,
    unchanged: usize = 0,
    /// Content changed (or first seen) -> queued for indexing.
    queued: usize = 0,
    /// Same content at a new path: path moved, vectors kept.
    renamed: usize = 0,
    /// mtime moved but content identical: metadata touch only.
    touched: usize = 0,
    deleted: usize = 0,
    /// Files that could not be read; skipped rather than aborting the run.
    unreadable: usize = 0,
};

pub const Filters = struct {
    /// Matched against the file extension including the dot.
    extensions: []const []const u8 = &.{ ".md", ".txt", ".mdx" },
    /// Directory basenames never descended into.
    exclude_dirs: []const []const u8 = &.{
        ".git", ".jj", "node_modules", ".zig-cache", "zig-out", "target",
        ".venv", "__pycache__", ".next", "dist", "build",
    },
    /// Hard ceiling per file. Larger files are skipped: a multi-megabyte single
    /// document is not prose, and chunking it would flood the index.
    max_file_bytes: u64 = 4 * 1024 * 1024,
};

pub fn reconcile(
    gpa: std.mem.Allocator,
    io: std.Io,
    s: *store.Store,
    collection_id: i64,
    root: []const u8,
    filters: Filters,
    now_ms: i64,
) !Report {
    var report: Report = .{};

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Paths present on disk this pass, so vanished files can be spotted after.
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);

    var dir = try std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walkSelectively(gpa);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        switch (entry.kind) {
            .directory => {
                // Selective walking: only descend where we want to look. Entering
                // .git on a real repo would dwarf the actual work.
                if (!isExcluded(entry.basename, filters.exclude_dirs)) {
                    try walker.enter(io, entry);
                }
                continue;
            },
            .file, .sym_link => {},
            else => continue,
        }

        if (!hasExtension(entry.basename, filters.extensions)) continue;
        // macOS resource forks and editor droppings are not documents.
        if (std.mem.startsWith(u8, entry.basename, "._")) continue;

        report.seen += 1;
        const rel_path = try arena.dupe(u8, entry.path);
        try seen.put(gpa, rel_path, {});

        const st = entry.dir.statFile(io, entry.basename, .{}) catch {
            report.unreadable += 1;
            continue;
        };
        if (st.size > filters.max_file_bytes) {
            report.unreadable += 1;
            continue;
        }
        const mtime_ms: i64 = @intCast(@divTrunc(st.mtime.nanoseconds, std.time.ns_per_ms));
        const size: i64 = @intCast(st.size);

        const existing = try s.findDoc(collection_id, rel_path);
        if (existing) |row| {
            // Cheap path: identity unchanged and already indexed.
            if (row.size == size and row.mtime_ms == mtime_ms and row.indexed_at != null) {
                report.unchanged += 1;
                continue;
            }
        }

        const digest = hashFile(io, entry.dir, entry.basename) catch {
            report.unreadable += 1;
            continue;
        };

        if (existing) |row| {
            if (std.mem.eql(u8, &row.content_sha, &digest) and row.indexed_at != null) {
                // Touched but not changed: keep the vectors, update the stamp so
                // the next pass takes the cheap path again.
                try s.touchDoc(row.id, mtime_ms);
                report.touched += 1;
                continue;
            }
        } else if (try s.findDocByShaExcludingPath(collection_id, &digest, rel_path)) |moved_id| {
            // Same bytes, different path, and nothing recorded here: a rename.
            // Re-embedding identical content would be pure waste.
            try s.moveDoc(moved_id, rel_path, mtime_ms);
            report.renamed += 1;
            continue;
        }

        _ = try s.upsertDocContent(collection_id, rel_path, &digest, size, mtime_ms);
        report.queued += 1;
    }

    // Anything recorded but no longer on disk goes away, chunks and all.
    {
        const recorded = try s.listAllPaths(arena, collection_id);
        for (recorded.items) |p| {
            if (seen.contains(p)) continue;
            if (try s.findDoc(collection_id, p)) |row| {
                try s.deleteDoc(row.id);
                report.deleted += 1;
            }
        }
    }

    _ = now_ms;
    return report;
}

fn hashFile(io: std.Io, dir: std.Io.Dir, basename: []const u8) !hash.Sha256Hex {
    var file = try dir.openFile(io, basename, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buf);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk_buf: [64 * 1024]u8 = undefined;
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk_buf);
        if (n == 0) break;
        hasher.update(chunk_buf[0..n]);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    var out: hash.Sha256Hex = undefined;
    _ = std.fmt.bufPrint(&out, "{x}", .{&digest}) catch unreachable;
    return out;
}

fn isExcluded(basename: []const u8, list: []const []const u8) bool {
    for (list) |d| if (std.mem.eql(u8, basename, d)) return true;
    return false;
}

fn hasExtension(basename: []const u8, exts: []const []const u8) bool {
    for (exts) |e| if (std.mem.endsWith(u8, basename, e)) return true;
    return false;
}
