//! zkb — Agent 的记忆系统 + 个人知识库。
//!
//! Truth lives in the filesystem (md + csv); everything in SQLite is a
//! derived index that can be deleted and rebuilt. See docs REQ/SPEC.

pub const sqlite = @import("db/sqlite.zig");
pub const schema = @import("db/schema.zig");
pub const store = @import("db/store.zig");
pub const markdown = @import("ingest/markdown.zig");
pub const chunk = @import("ingest/chunk.zig");
pub const scan = @import("ingest/scan.zig");
pub const csv = @import("ingest/csv.zig");
pub const indexer = @import("ingest/indexer.zig");
pub const fts_query = @import("search/fts_query.zig");
pub const rrf = @import("search/rrf.zig");
pub const hybrid = @import("search/hybrid.zig");
pub const pack = @import("search/pack.zig");
pub const trace = @import("search/trace.zig");
pub const maintain = @import("maintain.zig");
pub const maintain_vec = @import("maintain_vec.zig");
pub const facts = @import("facts.zig");
pub const memory = @import("memory.zig");
pub const recall = @import("recall.zig");
pub const records = @import("records.zig");
pub const expr = @import("query/expr.zig");
pub const embed_queue = @import("embed/queue.zig");
pub const proto = @import("ipc/proto.zig");
pub const ipc_client = @import("ipc/client.zig");
pub const daemon = if (build_options.llama) @import("daemon.zig") else struct {};
pub const hash = @import("util/hash.zig");
pub const paths = @import("util/paths.zig");
pub const utf8 = @import("util/utf8.zig");
pub const build_options = @import("build_options");

/// Only present when built with -Dllama (the default). Guarded so the storage
/// layer and its tests can be worked on without a cmake build.
pub const model_registry = @import("embed/registry.zig");
pub const embed = if (build_options.llama) @import("embed/llama.zig") else struct {};

pub const version = "0.0.3";

test {
    @import("std").testing.refAllDecls(@This());
}
