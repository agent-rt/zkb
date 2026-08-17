const std = @import("std");

pub const Sha256Hex = [64]u8;

/// SHA-256 of a file, streamed. Used for content fingerprints (docs) and the
/// model identity in `embedding_model_id` (SPEC §3.3).
pub fn fileSha256(io: std.Io, path: []const u8) !Sha256Hex {
    var file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var buf: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buf);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});

    // readSliceShort signals end-of-file with a short read, not an error.
    var chunk: [64 * 1024]u8 = undefined;
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        if (n == 0) break;
        hasher.update(chunk[0..n]);
    }

    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return toHex(&digest);
}

pub fn bytesSha256(data: []const u8) Sha256Hex {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return toHex(&digest);
}

fn toHex(digest: *const [32]u8) Sha256Hex {
    var out: Sha256Hex = undefined;
    _ = std.fmt.bufPrint(&out, "{x}", .{digest}) catch unreachable;
    return out;
}
