//! zkb — Agent 的记忆系统 + 个人知识库。
//!
//! Truth lives in the filesystem (md + csv); everything in SQLite is a
//! derived index that can be deleted and rebuilt. See docs REQ/SPEC.

pub const sqlite = @import("db/sqlite.zig");
pub const schema = @import("db/schema.zig");
pub const store = @import("db/store.zig");
pub const markdown = @import("ingest/markdown.zig");
pub const md4c = @import("ingest/md4c.zig");
pub const chunk = @import("ingest/chunk.zig");
pub const scan = @import("ingest/scan.zig");
pub const glob = @import("ingest/glob.zig");
pub const ignore = @import("ingest/ignore.zig");
pub const roots = @import("ingest/roots.zig");
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
pub const bench = @import("bench.zig");
pub const contexts = @import("contexts.zig");
pub const expr = @import("query/expr.zig");
pub const saved_sql = @import("query/saved.zig");
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

/// From `build.zig.zon`, which is also what the homebrew formula is generated
/// from. Written here as a literal until 0.0.26, where releasing meant editing
/// two files and remembering both: the version that ships in the tarball name and
/// the version the binary reports had no mechanism holding them together, and a
/// release with `zkb version` disagreeing with `brew info` came within one step of
/// going out. Nothing downstream would have caught it — the formula, the release
/// assets and the tag would all have been right.
pub const version = build_options.version;

test {
    @import("std").testing.refAllDecls(@This());
}
