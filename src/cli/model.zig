//! `zkb model pull` — fetch the embedding GGUF from Hugging Face with resume.
//!
//! Resume is not a nicety: the file is 609 MiB and a half-finished download is
//! the normal state of the world (this machine already had a 264 MB `.part`
//! from an earlier interrupted attempt). Verified against HF: the CDN answers
//! `accept-ranges: bytes` and range requests survive the cross-domain redirect.
//!
//! Integrity is checked against the LFS oid published by the HF API, which is
//! the file's SHA-256. A truncated or mismatched download is never promoted to
//! the final name.

const std = @import("std");
const zkb = @import("zkb");
const registry = zkb.model_registry;

const Writer = std.Io.Writer;

pub const Quant = registry.Quant;
pub const Spec = registry.Spec;
pub const spec = registry.spec;
pub const hfBlob = registry.hfBlob;
pub const resolve = registry.resolve;
pub const Resolved = registry.Resolved;
pub const Source = registry.Source;

pub const Error = error{
    HashMismatch,
    ShortDownload,
    RangeNotHonoured,
    HttpFailed,
};

pub fn pull(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    w: *Writer,
    q: Quant,
) !void {
    var layout = try zkb.paths.resolve(gpa, env);
    defer layout.deinit(gpa);

    const s = spec(q);
    const final = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ layout.models, s.file });
    defer gpa.free(final);
    const part = try std.fmt.allocPrint(gpa, "{s}.part", .{final});
    defer gpa.free(part);

    try layout.ensureDirs(io);

    // Already in a Hugging Face cache? Nothing to do — zkb reads it in place
    // rather than making a second copy of the same 609 MiB.
    if (try hfBlob(gpa, io, env, s)) |p| {
        defer gpa.free(p);
        try w.print("already present in the Hugging Face cache:\n  {s}\n", .{p});
        try w.writeAll("zkb will read it there; nothing downloaded\n");
        return;
    }

    // Already there and intact? Say so and stop — re-hashing 609 MiB is much
    // cheaper than re-downloading it, and silently redownloading would be rude.
    if (std.Io.Dir.accessAbsolute(io, final, .{})) |_| {
        try w.print("verifying existing {s} ...\n", .{s.file});
        try w.flush();
        const digest = try zkb.hash.fileSha256(io, final);
        if (std.mem.eql(u8, &digest, s.sha256)) {
            try w.print("already present and verified: {s}\n", .{final});
            return;
        }
        try w.print("sha256 mismatch, re-downloading\n", .{});
    } else |_| {}

    // How much of the .part do we already have?
    var have: u64 = 0;
    if (std.Io.Dir.openFileAbsolute(io, part, .{})) |f| {
        defer f.close(io);
        const st = try f.stat(io);
        have = st.size;
    } else |_| {}

    if (have > s.size) {
        // Longer than the real file: it is not a prefix of anything useful.
        try w.print("partial file is larger than expected, starting over\n", .{});
        try std.Io.Dir.deleteFileAbsolute(io, part);
        have = 0;
    }

    if (have == s.size) {
        try w.print("partial download is complete, verifying\n", .{});
    } else {
        try w.print("downloading {s}\n", .{s.file});
        if (have > 0) {
            try w.print("  resuming at {d:.1} MiB of {d:.1} MiB\n", .{
                mib(have), mib(s.size),
            });
        } else {
            try w.print("  {d:.1} MiB\n", .{mib(s.size)});
        }
        try w.flush();

        var file = try std.Io.Dir.createFileAbsolute(io, part, .{ .truncate = false });
        defer file.close(io);

        var fbuf: [256 * 1024]u8 = undefined;
        var fw = file.writer(io, &fbuf);
        // Positional writer: appending is just starting at the byte we reached.
        fw.pos = have;

        var client: std.http.Client = .{ .allocator = gpa, .io = io };
        defer client.deinit();

        const range = try std.fmt.allocPrint(gpa, "bytes={d}-", .{have});
        defer gpa.free(range);

        // extra_headers survive a cross-domain redirect, which matters because
        // HF redirects to its CDN — a stripped Range header would silently
        // restart the transfer at byte 0 and corrupt the resumed file.
        const headers: []const std.http.Header = if (have > 0)
            &.{.{ .name = "range", .value = range }}
        else
            &.{};

        const res = try client.fetch(.{
            .location = .{ .url = s.url },
            .method = .GET,
            .response_writer = &fw.interface,
            .extra_headers = headers,
        });
        try fw.interface.flush();

        if (have > 0 and res.status != .partial_content) {
            // 200 here means the server ignored Range and sent the whole file
            // from byte 0, which we just appended after `have` bytes of prefix.
            try w.print("server ignored Range (status {d}); discarding and retrying from scratch\n", .{
                @intFromEnum(res.status),
            });
            try std.Io.Dir.deleteFileAbsolute(io, part);
            return error.RangeNotHonoured;
        }
        if (have == 0 and res.status != .ok) {
            try w.print("http status {d}\n", .{@intFromEnum(res.status)});
            return error.HttpFailed;
        }
    }

    // Size then hash: a size check gives a clearer message for the common
    // truncation case than a bare hash mismatch would.
    {
        var f = try std.Io.Dir.openFileAbsolute(io, part, .{});
        const st = try f.stat(io);
        f.close(io);
        if (st.size != s.size) {
            try w.print("short download: {d} of {d} bytes\n", .{ st.size, s.size });
            return error.ShortDownload;
        }
    }

    try w.print("verifying sha256 ...\n", .{});
    try w.flush();
    const digest = try zkb.hash.fileSha256(io, part);
    if (!std.mem.eql(u8, &digest, s.sha256)) {
        try w.print("sha256 mismatch\n  expected {s}\n  got      {s}\n", .{ s.sha256, digest });
        // The .part is only worth keeping as a *correct* resumable prefix.
        // Once the hash proves the bytes are wrong, keeping it guarantees every
        // retry fails identically at full size — a permanent dead end the user
        // cannot escape without knowing about a file they never see. Drop it so
        // the next attempt starts clean.
        std.Io.Dir.deleteFileAbsolute(io, part) catch {};
        try w.print("discarded the partial file; run again to download from scratch\n", .{});
        return error.HashMismatch;
    }

    try std.Io.Dir.renameAbsolute(part, final, io);
    try w.print("ok: {s}\n", .{final});
}

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}
