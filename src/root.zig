//! zkb — Agent 的记忆系统 + 个人知识库。
//!
//! Truth lives in the filesystem (md + csv); everything in SQLite is a
//! derived index that can be deleted and rebuilt. See docs REQ/SPEC.

pub const sqlite = @import("db/sqlite.zig");
pub const build_options = @import("build_options");

/// Only present when built with -Dllama (the default). Guarded so the storage
/// layer and its tests can be worked on without a cmake build.
pub const embed = if (build_options.llama) @import("embed/llama.zig") else struct {};

pub const version = "0.1.0-dev";

test {
    @import("std").testing.refAllDecls(@This());
}
