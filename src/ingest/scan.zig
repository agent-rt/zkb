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
const glob = @import("glob.zig");
const ignoremod = @import("ignore.zig");

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
    /// Entries an ignore rule removed, files and directories both.
    ///
    /// Counted so "seen 0" can be told apart from "seen 0 because a `.gitignore`
    /// above this root excludes it". Both look identical in the output otherwise,
    /// and the second is a configuration surprise rather than an empty directory:
    /// a root inside a repo's ignored subtree indexes nothing, correctly and
    /// silently. Found by putting a test tree under `.zig-cache`, which this
    /// project's own `.gitignore` excludes.
    ignored: usize = 0,
};

pub const Filters = struct {
    /// Matched against the file extension including the dot.
    extensions: []const []const u8 = &.{ ".md", ".txt", ".mdx" },
    /// Glob patterns against the path relative to the root. Empty means no
    /// include filter — the common case, and the reason the empty list cannot
    /// mean "allow nothing": a collection with no patterns must scan everything,
    /// or every existing caller would silently index zero files.
    include: []const []const u8 = &.{},
    /// Directory basenames never descended into.
    exclude_dirs: []const []const u8 = &.{
        ".git",  ".jj",         "node_modules", ".zig-cache", "zig-out", "target",
        ".venv", "__pycache__", ".next",        "dist",       "build",
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

    // Rules accumulate outermost-first, which is what makes last-match-wins mean
    // "deeper and later overrides": any `.gitignore` between the git repo root
    // and this root, then this root's own files, then each subdirectory's as it
    // is entered.
    var ignore_patterns: std.ArrayList(ignoremod.Pattern) = .empty;
    const ignore_prefix = try loadAncestorIgnores(arena, io, root, &ignore_patterns);
    try loadIgnoreFiles(arena, io, dir, ignore_prefix, &ignore_patterns);

    var walker = try dir.walkSelectively(gpa);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        const ignorer: ignoremod.Matcher = .{ .patterns = ignore_patterns.items, .prefix = ignore_prefix };
        switch (entry.kind) {
            .directory => {
                // Selective walking: only descend where we want to look. Entering
                // .git on a real repo would dwarf the actual work.
                //
                // Include patterns prune here too, not only at the file check.
                // With `*/memory/*.md` over ~/.claude/projects, filtering only
                // files would still walk every project's tool-results directory —
                // orders of magnitude more entries than the ones being kept.
                // An ignored subtree is not entered at all, which is both the
                // cheap thing to do and what makes a negation inside it unable to
                // bring anything back — git's own rule.
                if (ignorer.isIgnored(entry.path, true)) {
                    report.ignored += 1;
                    continue;
                }
                if (!isExcluded(entry.basename, filters.exclude_dirs) and
                    glob.matchAnyPrefix(filters.include, entry.path))
                {
                    // Its own rules load only after it survives the parent's:
                    // an ignored directory is never opened, which is also why a
                    // `!` inside one cannot bring anything back (git's rule).
                    var sub = entry.dir.openDir(io, entry.basename, .{}) catch {
                        try walker.enter(io, entry);
                        continue;
                    };
                    defer sub.close(io);
                    const sub_base = if (ignore_prefix.len == 0)
                        try arena.dupe(u8, entry.path)
                    else
                        try std.fmt.allocPrint(arena, "{s}/{s}", .{ ignore_prefix, entry.path });
                    try loadIgnoreFiles(arena, io, sub, sub_base, &ignore_patterns);
                    try walker.enter(io, entry);
                }
                continue;
            },
            .file, .sym_link => {},
            else => continue,
        }

        if (!hasExtension(entry.basename, filters.extensions)) continue;
        if (!glob.matchAny(filters.include, entry.path)) continue;
        if (ignorer.isIgnored(entry.path, false)) {
            report.ignored += 1;
            continue;
        }
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
        } else if (try s.findDocByShaExcludingPath(arena, collection_id, &digest, rel_path)) |match| {
            // Same bytes at another path is only a rename if that path is gone.
            //
            // Without the check, two byte-identical files in one collection make
            // the second look like a move of the first, and `moveDoc` repoints the
            // single record — leaving one real file with no row at all. Measured on
            // ~/.claude/projects: 315 files on disk, 314 indexed, and the missing
            // one was a memory written identically in two projects.
            //
            // The filesystem is asked rather than the `seen` set, because the old
            // path may simply not have been walked yet this pass.
            const old_gone = if (std.mem.eql(u8, match.rel_path, rel_path))
                false
            else blk: {
                dir.access(io, match.rel_path, .{}) catch break :blk true;
                break :blk false;
            };
            if (old_gone) {
                // Re-embedding identical content would be pure waste.
                try s.moveDoc(match.id, rel_path, mtime_ms);
                report.renamed += 1;
                continue;
            }
            // Falls through to insert: a genuine second copy. If several documents
            // share the sha, the query returns an arbitrary one, so a real rename
            // can be missed here and pay for a re-embed. That is the safe
            // direction — correct content at a cost, rather than a missing file.
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

/// Load every `.gitignore` between the enclosing git repo root and `root`,
/// returning `root`'s path relative to the outermost one that was read.
///
/// Needed because a repo's rules usually sit above a collection root: a
/// collection rooted at `<repo>/docs` has its `.gitignore` at `<repo>`. Reading only
/// at or below the root would respect some of `.gitignore` and silently not the
/// rest, which is the lookalike failure this whole feature is trying to avoid.
///
/// Stops at the directory holding `.git`, or at the filesystem root if there is
/// none. Returns `""` when nothing above `root` had rules, which keeps the
/// no-repo case allocation-free and the matcher on its fast path.
fn loadAncestorIgnores(
    arena: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    out: *std.ArrayList(ignoremod.Pattern),
) ![]const u8 {
    // Walk up collecting candidates, then load them outermost-first.
    var dirs: std.ArrayList([]const u8) = .empty;
    var cur = std.mem.trimEnd(u8, root, "/");
    var found_repo = false;
    while (std.fs.path.dirname(cur)) |parent| {
        if (parent.len == 0) break;
        try dirs.append(arena, parent);
        var probe = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch break;
        defer probe.close(io);
        if (probe.access(io, ".git", .{})) |_| {
            found_repo = true;
            break;
        } else |_| {}
        cur = parent;
    }
    // Without a repo boundary, walking to `/` would pick up unrelated rules from
    // a home directory or a volume root.
    if (!found_repo) return "";

    const outermost = dirs.items[dirs.items.len - 1];
    var i = dirs.items.len;
    while (i > 0) {
        i -= 1;
        const d = dirs.items[i];
        var dh = std.Io.Dir.openDirAbsolute(io, d, .{}) catch continue;
        defer dh.close(io);
        const base = if (d.len > outermost.len) d[outermost.len + 1 ..] else "";
        try loadIgnoreFiles(arena, io, dh, base, out);
    }
    return root[outermost.len + 1 ..];
}

/// Read `<dir>/.gitignore` then `<dir>/.zkbignore` and append their patterns.
///
/// A missing file is the common case and not an error. An unreadable one is
/// skipped rather than failing the scan: losing a filter is better than losing
/// the whole pass, and the file being absent from the pattern list is visible in
/// what gets indexed.
fn loadIgnoreFiles(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    base: []const u8,
    out: *std.ArrayList(ignoremod.Pattern),
) !void {
    // Order matters: `.gitignore` first so `.zkbignore` can override it.
    for (ignoremod.file_names) |name| try loadOneIgnoreFile(arena, io, dir, base, name, out);
}

fn loadOneIgnoreFile(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    base: []const u8,
    name: []const u8,
    out: *std.ArrayList(ignoremod.Pattern),
) !void {
    var file = dir.openFile(io, name, .{}) catch return;
    defer file.close(io);
    const st = file.stat(io) catch return;
    if (st.size > 256 * 1024) return;
    const text = arena.alloc(u8, @intCast(st.size)) catch return;
    var reader = file.reader(io, text);
    reader.interface.readSliceAll(text) catch return;
    try ignoremod.parseInto(arena, out, base, text);
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
