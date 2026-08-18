//! Schema DDL and migrations.
//!
//! The database is derived data: `rm zkb.db && zkb index` rebuilds it from the
//! filesystem. That is why migrations here only ever need to reach a valid
//! *shape* — they never have to preserve rows. When a change would be awkward
//! to migrate, bumping `chunker_version` / re-indexing is the sanctioned answer
//! (SPEC §14.1).

const std = @import("std");
const sqlite = @import("sqlite.zig");

pub const schema_version: i64 = 7;

/// Bumped when chunk boundaries change. Vectors are only valid for the chunk
/// text that produced them, so a chunker change invalidates the index just as
/// surely as a model change does.
pub const chunker_version: i64 = 1;

pub const embedding_dim: i64 = 1024;

pub const Error = sqlite.Error || error{
    SchemaFromFuture,
    SchemaStale,
    DimensionMismatch,
};

const ddl_v1 =
    \\CREATE TABLE meta (
    \\  key   TEXT PRIMARY KEY,
    \\  value TEXT NOT NULL
    \\) STRICT;
    \\
    \\CREATE TABLE collections (
    \\  id         INTEGER PRIMARY KEY,
    \\  name       TEXT NOT NULL UNIQUE,
    \\  root       TEXT NOT NULL,
    \\  created_at INTEGER NOT NULL
    \\) STRICT;
    \\
    \\CREATE TABLE docs (
    \\  id            INTEGER PRIMARY KEY,
    \\  collection_id INTEGER NOT NULL REFERENCES collections(id),
    \\  rel_path      TEXT NOT NULL,
    \\  title         TEXT,
    \\  frontmatter   TEXT,
    \\  content_sha   TEXT NOT NULL,
    \\  size          INTEGER NOT NULL,
    \\  mtime_ms      INTEGER NOT NULL,
    \\  chunk_count   INTEGER NOT NULL DEFAULT 0,
    \\  indexed_at    INTEGER,
    \\  index_error   TEXT,
    \\  UNIQUE(collection_id, rel_path)
    \\) STRICT;
    \\
    \\CREATE INDEX docs_sha ON docs(content_sha);
    \\CREATE INDEX docs_pending ON docs(collection_id) WHERE indexed_at IS NULL;
    \\
    \\CREATE TABLE chunks (
    \\  id           INTEGER PRIMARY KEY,
    \\  doc_id       INTEGER NOT NULL REFERENCES docs(id),
    \\  idx          INTEGER NOT NULL,
    \\  heading_path TEXT,
    \\  byte_start   INTEGER NOT NULL,
    \\  byte_end     INTEGER NOT NULL,
    \\  n_tokens     INTEGER NOT NULL,
    \\  text         TEXT NOT NULL,
    \\  UNIQUE(doc_id, idx)
    \\) STRICT;
    \\
    \\CREATE INDEX chunks_doc ON chunks(doc_id, idx);
;

// Contentless so the body is not stored twice; contentless_delete so a chunk
// can be removed by rowid alone, without re-reading text the ingest path
// already discarded.
//
// tokenize='zkb_cjk' (src/db/fts5_cjk.c): CJK bigrams + Latin whole words.
// unicode61 does not segment CJK at all; trigram does, but measured only 0.167
// recall@10 on Chinese queries (docs/experiments/E2-baseline.md), which is what
// motivated writing a tokenizer instead of living with it.
const ddl_fts =
    \\CREATE VIRTUAL TABLE fts_chunks USING fts5(
    \\  text,
    \\  heading_path,
    \\  content='', contentless_delete=1,
    \\  tokenize='zkb_cjk'
    \\);
;

// collection_id is a partition key so multi-collection KNN filters *inside*
// the search rather than after top-k (verified by E1).
const ddl_vec =
    \\CREATE VIRTUAL TABLE vec_chunks USING vec0(
    \\  collection_id INTEGER partition key,
    \\  chunk_id      INTEGER PRIMARY KEY,
    \\  embedding     FLOAT[1024] distance_metric=cosine
    \\);
;

/// Verify the schema without writing. For read-only connections, which cannot
/// migrate: returning a specific error lets the caller print something the user
/// can act on instead of failing on a write attempt deep inside a DDL statement.
pub fn verify(db: *sqlite.Db) Error!void {
    const have = try currentVersion(db);
    if (have > schema_version) return error.SchemaFromFuture;
    if (have < schema_version) return error.SchemaStale;
    try checkDimension(db);
}

/// v3 adds the link graph and the maintenance history. Both are pure derived
/// data — links are re-extracted on every index, and the history is a log — so
/// this migration only has to create the shape.
const ddl_v3 =
    \\CREATE TABLE links (
    \\  id            INTEGER PRIMARY KEY,
    \\  doc_id        INTEGER NOT NULL REFERENCES docs(id),
    \\  chunk_id      INTEGER,
    \\  kind          TEXT NOT NULL,
    \\  raw           TEXT NOT NULL,
    \\  target_doc_id INTEGER
    \\) STRICT;
    \\
    \\CREATE INDEX links_doc    ON links(doc_id);
    \\CREATE INDEX links_target ON links(target_doc_id);
    \\
    \\CREATE TABLE maintenance_runs (
    \\  id         INTEGER PRIMARY KEY,
    \\  started_at INTEGER NOT NULL,
    \\  checks     TEXT NOT NULL,
    \\  report     TEXT NOT NULL
    \\) STRICT;
;

/// v4 adds the memory collection and facts. Both are materialized from files
/// (md and csv), so like everything else here they are derived and rebuildable.
const ddl_v4 =
    \\ALTER TABLE collections ADD COLUMN kind TEXT NOT NULL DEFAULT 'documents';
    \\
    \\CREATE TABLE facts (
    \\  chunk_id  INTEGER PRIMARY KEY,
    \\  doc_id    INTEGER NOT NULL REFERENCES docs(id),
    \\  line_no   INTEGER NOT NULL,
    \\  key       TEXT NOT NULL,
    \\  value_num REAL,
    \\  value_txt TEXT,
    \\  at        TEXT NOT NULL,
    \\  src       TEXT,
    \\  note      TEXT
    \\) STRICT;
    \\
    \\CREATE INDEX facts_key_at ON facts(key, at DESC);
    \\
    \\CREATE TABLE rec_memory (
    \\  chunk_id INTEGER PRIMARY KEY,
    \\  doc_id   INTEGER NOT NULL REFERENCES docs(id),
    \\  type     TEXT NOT NULL,
    \\  status   TEXT NOT NULL,
    \\  created  TEXT NOT NULL,
    \\  source   TEXT,
    \\  subjects TEXT,
    \\  refs     TEXT
    \\) STRICT;
    \\
    \\CREATE INDEX rec_memory_doc     ON rec_memory(doc_id);
    \\CREATE INDEX rec_memory_status  ON rec_memory(status, created DESC);
;

/// v5: the inferred schema of each records type.
///
/// The `rec_<type>` tables themselves are **not** created here — their columns
/// come from a csv header, so they are built at index time and rebuilt whenever
/// the header changes (SPEC §16.3). This table is what makes that inference
/// inspectable: `zkb records <type> --schema` reads it, and inference you cannot
/// look at is inference you cannot debug when it guesses wrong.
const ddl_v5 =
    \\CREATE TABLE rec_meta (
    \\  type       TEXT NOT NULL,
    \\  field      TEXT NOT NULL,
    \\  kind       TEXT NOT NULL,
    \\  indexed    INTEGER NOT NULL,
    \\  vectorized INTEGER NOT NULL,
    \\  ord        INTEGER NOT NULL,
    \\  overridden INTEGER NOT NULL DEFAULT 0,
    \\  src_col    INTEGER NOT NULL DEFAULT 0,
    \\  PRIMARY KEY (type, field)
    \\) STRICT;
;

/// v6: `facts.recorded_at` — when the row was written, as opposed to when the
/// fact took effect.
///
/// v5 and earlier left this to version control ("the commit time already tracks
/// it"). Nothing ever read it, which is what makes an axis imaginary rather than
/// real. A column can be queried; a commit timestamp would have meant forking
/// `jj` and parsing its output.
const ddl_v6 =
    \\ALTER TABLE facts ADD COLUMN recorded_at TEXT NOT NULL DEFAULT '';
;

/// v7 moves a collection's scan configuration out of the code and into the row.
///
/// It used to live in a fixed three-element array inside the daemon, which meant
/// a collection registered by `zkb index --root X --collection Y` was scanned
/// exactly once and then forgotten — the daemon had no way to learn it existed.
/// A root is only usefully a root if whatever rescans knows about it.
///
/// Both columns are NULL for "use the built-in default", which is not the same as
/// the empty string: `extensions = ''` would be a collection that matches no
/// file, and that has to stay expressible for a caller who really means it.
const ddl_v7 =
    \\ALTER TABLE collections ADD COLUMN extensions TEXT;
    \\ALTER TABLE collections ADD COLUMN include TEXT;
;

/// Bring `db` to `schema_version`, creating it if empty. Idempotent.
pub fn migrate(db: *sqlite.Db) Error!void {
    try db.exec("PRAGMA foreign_keys = ON;");

    const have = try currentVersion(db);
    if (have == schema_version) {
        try checkDimension(db);
        return;
    }
    if (have > schema_version) return error.SchemaFromFuture;

    if (have == 0) {
        try db.exec("BEGIN IMMEDIATE;");
        errdefer db.exec("ROLLBACK;") catch {};
        try db.exec(ddl_v1);
        try db.exec(ddl_v3);
        try db.exec(ddl_v4);
        try db.exec(ddl_v5);
        try db.exec(ddl_v6);
        try db.exec(ddl_v7);
        try db.exec(ddl_fts);
        try db.exec(ddl_vec);
        try setMetaInt(db, "schema_version", schema_version);
        try setMetaInt(db, "chunker_version", chunker_version);
        try setMetaInt(db, "embedding_dim", embedding_dim);
        try db.exec("COMMIT;");
        return;
    }

    if (have == 1) try migrateV1ToV2(db);
    if (have < 3) try migrateToV3(db);
    if (have < 4) try migrateToV4(db);
    if (have < 5) try migrateToV5(db);
    if (have < 6) try migrateToV6(db);
    if (have < 7) try migrateToV7(db);
}

/// v6 -> v7: scan configuration becomes per-collection data.
///
/// Existing rows keep NULL, which resolves to the same defaults the code applied
/// before this column existed, so an upgraded database scans exactly as it did.
fn migrateToV7(db: *sqlite.Db) Error!void {
    try db.exec("BEGIN IMMEDIATE;");
    errdefer db.exec("ROLLBACK;") catch {};
    try db.exec(ddl_v7);
    try setMetaInt(db, "schema_version", 7);
    try db.exec("COMMIT;");
}

/// v5 -> v6: the second time axis becomes a column.
///
/// Existing rows get `''` rather than a guessed date: "not recorded" is the
/// truth for a row written before the column existed, and inventing a timestamp
/// would make it un-auditable.
fn migrateToV6(db: *sqlite.Db) Error!void {
    try db.exec("BEGIN IMMEDIATE;");
    errdefer db.exec("ROLLBACK;") catch {};
    try db.exec(ddl_v6);
    try setMetaInt(db, "schema_version", 6);
    try db.exec("COMMIT;");
}

/// v4 -> v5: the records schema registry.
///
/// Existing `rec_<type>` tables, if any, are left alone: they are keyed by a
/// header that has not changed, and index time rebuilds them anyway when it has.
fn migrateToV5(db: *sqlite.Db) Error!void {
    try db.exec("BEGIN IMMEDIATE;");
    errdefer db.exec("ROLLBACK;") catch {};
    try db.exec(ddl_v5);
    try setMetaInt(db, "schema_version", 5);
    try db.exec("COMMIT;");
}

/// v3 -> v4: collection kinds, facts, memory metadata.
fn migrateToV4(db: *sqlite.Db) Error!void {
    try db.exec("BEGIN IMMEDIATE;");
    errdefer db.exec("ROLLBACK;") catch {};
    try db.exec(ddl_v4);
    try setMetaInt(db, "schema_version", 4);
    try db.exec("COMMIT;");
}

/// v2 -> v3: add the link graph and the maintenance history.
///
/// The link graph starts empty and fills in as documents are re-parsed. It does
/// not backfill: extracting links needs the source text, and the indexer only
/// re-reads a document when its content hash changes. So on an existing index the
/// graph stays empty until documents change, or until `zkb index --force`.
/// Stated rather than hidden — a maintenance report over an empty graph would
/// otherwise claim every document is unlinked.
fn migrateToV3(db: *sqlite.Db) Error!void {
    try db.exec("BEGIN IMMEDIATE;");
    errdefer db.exec("ROLLBACK;") catch {};
    try db.exec(ddl_v3);
    try setMetaInt(db, "schema_version", 3);
    try db.exec("COMMIT;");
}

/// v1 -> v2: swap the FTS tokenizer from `trigram` to `zkb_cjk`.
///
/// The keyword index is rebuilt from `chunks.text`, which is still on disk —
/// **no re-embedding**. That is a direct dividend of the three tables sharing
/// `chunks.id` (SPEC §2.4): the expensive artefact (vectors) is independent of
/// the cheap one (the inverted index), so changing the tokenizer costs seconds
/// instead of the ~5 minutes a full re-index would take.
fn migrateV1ToV2(db: *sqlite.Db) Error!void {
    try db.exec("BEGIN IMMEDIATE;");
    errdefer db.exec("ROLLBACK;") catch {};

    try db.exec("DROP TABLE IF EXISTS fts_chunks;");
    try db.exec(ddl_fts);
    try db.exec(
        \\INSERT INTO fts_chunks(rowid, text, heading_path)
        \\SELECT id, text, COALESCE(heading_path, '') FROM chunks;
    );
    // Only its own target: setting the latest version here would mark the
    // database as v3 without v3's tables if the next step failed.
    try setMetaInt(db, "schema_version", 2);

    try db.exec("COMMIT;");
}

fn currentVersion(db: *sqlite.Db) Error!i64 {
    // meta may not exist yet; absence means version 0.
    const has_meta = (try db.queryI64(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='meta'",
    )) != null;
    if (!has_meta) return 0;
    var buf: [32]u8 = undefined;
    const v = (try db.queryText("SELECT value FROM meta WHERE key='schema_version'", &buf)) orelse
        return 0;
    return std.fmt.parseInt(i64, v, 10) catch 0;
}

/// The vec0 DDL hardcodes FLOAT[1024]; meta records what the rest of the code
/// believes. If they ever disagree, every vector written is silently the wrong
/// shape — so refuse to proceed rather than corrupt the index.
fn checkDimension(db: *sqlite.Db) Error!void {
    var buf: [32]u8 = undefined;
    const v = (try db.queryText("SELECT value FROM meta WHERE key='embedding_dim'", &buf)) orelse
        return error.DimensionMismatch;
    const dim = std.fmt.parseInt(i64, v, 10) catch return error.DimensionMismatch;
    if (dim != embedding_dim) return error.DimensionMismatch;
}

pub fn setMeta(db: *sqlite.Db, key: []const u8, value: []const u8) sqlite.Error!void {
    var st = try db.prepare(
        "INSERT INTO meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = ?2",
    );
    defer st.finalize();
    try st.bindText(1, key);
    try st.bindText(2, value);
    _ = try st.step();
}

pub fn setMetaInt(db: *sqlite.Db, key: []const u8, value: i64) sqlite.Error!void {
    var buf: [24]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{d}", .{value}) catch unreachable;
    return setMeta(db, key, s);
}

/// Returns a copy in `buf`, or null if unset.
pub fn getMeta(db: *sqlite.Db, key: []const u8, buf: []u8) sqlite.Error!?[]const u8 {
    var st = try db.prepare("SELECT value FROM meta WHERE key = ?1");
    defer st.finalize();
    try st.bindText(1, key);
    if (!try st.step()) return null;
    const v = st.columnText(0);
    const n = @min(v.len, buf.len);
    @memcpy(buf[0..n], v[0..n]);
    return buf[0..n];
}
