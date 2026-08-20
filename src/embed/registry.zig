//! Which embedding models exist, and where to find one on this machine.
//!
//! Separate from `cli/model.zig` (which downloads them) because the daemon lives
//! inside the `zkb` module and the CLI does not: a file inside the module cannot
//! import one outside it that imports the module back. Knowing *where the model
//! is* is needed by both, so it belongs here; knowing *how to fetch it* is only
//! needed by the CLI.

const std = @import("std");
const paths = @import("../util/paths.zig");

pub const Quant = enum { q8_0, f16 };


pub const Spec = struct {
    file: []const u8,
    url: []const u8,
    size: u64,
    /// SHA-256 == the LFS oid from https://huggingface.co/api/models/<repo>/tree/main
    ///
    /// This doubles as the filename inside a Hugging Face cache: `huggingface_hub`
    /// names blobs by their LFS oid, which is this exact string. That coincidence
    /// is what makes `hfBlob` a path join rather than a cache-format parser.
    sha256: []const u8,
    /// `<org>/<name>`, for locating the same file in a Hugging Face cache.
    repo: []const u8,
};

const repo_id = "Qwen/Qwen3-Embedding-0.6B-GGUF";
const repo_base = "https://huggingface.co/" ++ repo_id ++ "/resolve/main/";

pub fn spec(q: Quant) Spec {
    return switch (q) {
        // q8_0 is the default: E3 measured cos 0.9997 against the fp32
        // reference, so the quality gap to f16 is inside quantization noise
        // while halving resident memory. See docs/experiments/E3-pooling.md.
        .q8_0 => .{
            .file = "Qwen3-Embedding-0.6B-Q8_0.gguf",
            .url = repo_base ++ "Qwen3-Embedding-0.6B-Q8_0.gguf",
            .size = 639_150_592,
            .sha256 = "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
            .repo = repo_id,
        },
        .f16 => .{
            .file = "Qwen3-Embedding-0.6B-f16.gguf",
            .url = repo_base ++ "Qwen3-Embedding-0.6B-f16.gguf",
            .size = 1_197_629_632,
            .sha256 = "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340",
            .repo = repo_id,
        },
    };
}

// ---------------------------------------------------------------------------
// finding the model
// ---------------------------------------------------------------------------

pub const Source = enum {
    /// `--model` on the command line.
    override,
    /// `~/.zkb/models`, downloaded by `zkb model pull`.
    local,
    /// An existing Hugging Face cache — the same file, already on disk.
    hf_cache,
};

pub const Resolved = struct {
    path: []u8,
    source: Source,

    pub fn deinit(self: Resolved, gpa: std.mem.Allocator) void {
        gpa.free(self.path);
    }
};

/// Where the embedding model actually is.
///
/// Checks the Hugging Face cache before downloading anything, because a machine
/// that has ever pulled this repo already holds the exact bytes. Measured on
/// this one: 620 MB in `~/.zkb/models` alongside a 264 MB abandoned `.part` of
/// the same blob in `~/.cache/huggingface` — paid for twice, used once.
///
/// **Read-only.** zkb never writes into the HF cache: that would mean
/// maintaining `refs/`, `snapshots/` and their symlinks, which is
/// `huggingface_hub`'s job and easy to corrupt from outside.
pub fn resolve(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    layout: *const paths.Layout,
    override: ?[]const u8,
    q: Quant,
) !Resolved {
    // Checked like every other branch. This was the one path handed back without
    // verifying it exists, so `--model /typo` got all the way to llama_model_load
    // and surfaced as an unhandled `error.LoadFailed` with a stack trace, while
    // having no model at all produced a clean message. Same command, two failure
    // shapes, and the ugly one was the case where the user had typed something.
    if (override) |o| {
        std.Io.Dir.accessAbsolute(io, o, .{}) catch return error.ModelNotFound;
        return .{ .path = try gpa.dupe(u8, o), .source = .override };
    }

    const s = spec(q);
    const local = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ layout.models, s.file });
    if (std.Io.Dir.accessAbsolute(io, local, .{})) |_| {
        return .{ .path = local, .source = .local };
    } else |_| gpa.free(local);

    if (try hfBlob(gpa, io, env, s)) |p| return .{ .path = p, .source = .hf_cache };

    return error.ModelNotFound;
}

/// The complete blob for `s` inside a Hugging Face cache, or null.
///
/// A `.part` is deliberately not accepted: an interrupted download is exactly
/// what tends to be sitting there, and half a model loads as a corrupt one.
pub fn hfBlob(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    s: Spec,
) !?[]u8 {
    const hub = try hfHub(gpa, env) orelse return null;
    defer gpa.free(hub);

    // `Qwen/Qwen3-Embedding-0.6B-GGUF` -> `models--Qwen--Qwen3-Embedding-0.6B-GGUF`
    var owner: std.ArrayList(u8) = .empty;
    defer owner.deinit(gpa);
    try owner.appendSlice(gpa, "models--");
    for (s.repo) |c| {
        if (c == '/') try owner.appendSlice(gpa, "--") else try owner.append(gpa, c);
    }

    const path = try std.fmt.allocPrint(gpa, "{s}/{s}/blobs/{s}", .{ hub, owner.items, s.sha256 });
    errdefer gpa.free(path);

    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch {
        gpa.free(path);
        return null;
    };
    defer file.close(io);

    // Size is the cheap gate. The name already asserts the hash — that is how HF
    // stores it — and re-hashing 620 MB on every command would not be.
    const st = file.stat(io) catch {
        gpa.free(path);
        return null;
    };
    if (st.size != s.size) {
        gpa.free(path);
        return null;
    }
    return path;
}

fn hfHub(gpa: std.mem.Allocator, env: *const std.process.Environ.Map) !?[]u8 {
    if (env.get("HF_HUB_CACHE")) |v| return try gpa.dupe(u8, v);
    if (env.get("HF_HOME")) |v| return try std.fmt.allocPrint(gpa, "{s}/hub", .{v});
    if (env.get("XDG_CACHE_HOME")) |v| {
        return try std.fmt.allocPrint(gpa, "{s}/huggingface/hub", .{v});
    }
    const home = env.get("HOME") orelse return null;
    return try std.fmt.allocPrint(gpa, "{s}/.cache/huggingface/hub", .{home});
}


/// What a daemon should do about the model it found, or did not.
pub const Startup = union(enum) {
    ready: Resolved,
    /// Start anyway, saying this. Keyword search, scanning and maintenance all
    /// work without a model; only embedding does not.
    degraded: []const u8,
};

/// `resolve` for a daemon, which unlike a one-shot command has something useful
/// to do without a model.
///
/// The rule lives here because two callers need the same answer: `daemon start`
/// decides whether to spawn at all, and `daemon run` decides whether to come up.
/// They used to hold it separately, and the first one still refused after the
/// second learned to degrade — so the daemon that could now start was never
/// launched, and the two disagreed about the same question in the same release.
///
/// A named `--model` that is not there stays an error either way. That is a
/// typo, and answering a typo with a degraded daemon hides it.
pub fn resolveForDaemon(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    layout: *const paths.Layout,
    override: ?[]const u8,
    q: Quant,
) !Startup {
    const found = resolve(gpa, io, env, layout, override, q) catch |err| switch (err) {
        error.ModelNotFound => {
            if (override != null) return err;
            return .{ .degraded = "no embedding model; run: zkb model pull" };
        },
        else => return err,
    };
    return .{ .ready = found };
}
