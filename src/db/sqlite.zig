//! Minimal SQLite wrapper. Narrow on purpose: open / exec / prepare / bind /
//! step / column, plus the vec0 registration every connection needs.
//!
//! Build-time contract (see build.zig):
//!   SQLITE_ENABLE_FTS5=1   keyword retrieval path
//!   SQLITE_THREADSAFE=2    multi-thread: one connection per thread, never shared
//!   SQLITE_CORE=1          sqlite-vec links against sqlite3 directly
//!
//! A `Db` is owned by exactly one thread. The daemon gives each connection
//! thread its own; nothing here is guarded by a mutex.
//!
//! Errors collapse to a small set; the SQLite message is snapshotted into
//! `last_msg` at failure time because sqlite3_errmsg() gets clobbered by the
//! next call on the connection.

const std = @import("std");

pub const c = @cImport({
    @cInclude("sqlite3.h");
    @cInclude("sqlite-vec.h");
});

/// Registers the `zkb_cjk` FTS5 tokenizer (src/db/fts5_cjk.c).
extern fn zkb_register_cjk_tokenizer(db: *c.sqlite3) c_int;

extern fn zkb_bind_text_transient(stmt: *c.sqlite3_stmt, idx: c_int, value: [*]const u8, n_bytes: c_int) c_int;
extern fn zkb_bind_blob_transient(stmt: *c.sqlite3_stmt, idx: c_int, value: *const anyopaque, n_bytes: c_int) c_int;

pub const Error = error{
    Sqlite,
    SqliteOpen,
    SqlitePrepare,
    SqliteStep,
    SqliteBind,
    OutOfMemory,
};

pub const OpenMode = enum { read_write, read_only };

pub const Db = struct {
    handle: *c.sqlite3,
    last_msg: [512]u8 = @splat(0),
    last_msg_len: usize = 0,

    /// Open (or create) a database. Pass ":memory:" for a transient one.
    /// Registers the vec0 virtual table module on this connection — vec0 is
    /// per-connection, so every new Db must do this or `USING vec0(...)`
    /// fails with "no such module".
    pub fn open(path: [:0]const u8, mode: OpenMode) Error!Db {
        var raw: ?*c.sqlite3 = null;
        const flags: c_int = switch (mode) {
            // NOMUTEX is already the default under THREADSAFE=2; passing it
            // documents that connections are never shared across threads.
            .read_write => c.SQLITE_OPEN_READWRITE | c.SQLITE_OPEN_CREATE | c.SQLITE_OPEN_NOMUTEX,
            .read_only => c.SQLITE_OPEN_READONLY | c.SQLITE_OPEN_NOMUTEX,
        };
        const rc = c.sqlite3_open_v2(path.ptr, &raw, flags, null);
        const h = raw orelse return error.SqliteOpen;
        if (rc != c.SQLITE_OK) {
            _ = c.sqlite3_close_v2(h);
            return error.SqliteOpen;
        }

        var verr: [*c]u8 = null;
        // Third arg (api routines) is ignored: built with SQLITE_CORE.
        if (c.sqlite3_vec_init(h, &verr, null) != c.SQLITE_OK) {
            if (verr) |m| c.sqlite3_free(m);
            _ = c.sqlite3_close_v2(h);
            return error.SqliteOpen;
        }

        // FTS5 tokenizers are per-connection too: without this, opening a
        // table declared with tokenize='zkb_cjk' fails outright.
        if (zkb_register_cjk_tokenizer(h) != c.SQLITE_OK) {
            _ = c.sqlite3_close_v2(h);
            return error.SqliteOpen;
        }

        var db: Db = .{ .handle = h };
        // busy_timeout covers WAL checkpoint contention; the writer is single
        // so this is the only BUSY source we expect.
        db.exec("PRAGMA busy_timeout = 5000;") catch {};
        return db;
    }

    pub fn close(self: *Db) void {
        _ = c.sqlite3_close_v2(self.handle);
    }

    /// Enable WAL. Separate from open() because ":memory:" rejects it and
    /// read-only connections must not attempt it.
    pub fn enableWal(self: *Db) Error!void {
        // journal_mode returns a row; sqlite3_exec tolerates that.
        try self.exec("PRAGMA journal_mode = WAL;");
    }

    pub fn errmsg(self: *const Db) []const u8 {
        const p = c.sqlite3_errmsg(self.handle);
        if (p == null) return "";
        return std.mem.span(p);
    }

    /// Message captured at the moment of the last failure in this wrapper.
    pub fn lastError(self: *const Db) []const u8 {
        return self.last_msg[0..self.last_msg_len];
    }

    fn capture(self: *Db) void {
        const msg = self.errmsg();
        const n = @min(msg.len, self.last_msg.len);
        @memcpy(self.last_msg[0..n], msg[0..n]);
        self.last_msg_len = n;
    }

    /// Run SQL returning no rows of interest. Multiple `;`-separated
    /// statements are allowed.
    pub fn exec(self: *Db, sql: [:0]const u8) Error!void {
        var err: [*c]u8 = null;
        const rc = c.sqlite3_exec(self.handle, sql.ptr, null, null, &err);
        if (err) |m| c.sqlite3_free(m);
        if (rc != c.SQLITE_OK) {
            self.capture();
            return error.Sqlite;
        }
    }

    pub fn prepare(self: *Db, sql: []const u8) Error!Stmt {
        var raw: ?*c.sqlite3_stmt = null;
        const rc = c.sqlite3_prepare_v2(self.handle, sql.ptr, @intCast(sql.len), &raw, null);
        if (rc != c.SQLITE_OK) {
            self.capture();
            return error.SqlitePrepare;
        }
        return .{ .stmt = raw orelse return error.SqlitePrepare };
    }

    /// Single-row single-column integer query. Returns null if no row.
    pub fn queryI64(self: *Db, sql: []const u8) Error!?i64 {
        var st = try self.prepare(sql);
        defer st.finalize();
        if (!try st.step()) return null;
        return st.columnI64(0);
    }

    /// Single-row single-column text query, copied into `buf`. Returns the
    /// slice of `buf` that was filled, or null if no row.
    pub fn queryText(self: *Db, sql: []const u8, buf: []u8) Error!?[]const u8 {
        var st = try self.prepare(sql);
        defer st.finalize();
        if (!try st.step()) return null;
        const txt = st.columnText(0);
        const n = @min(txt.len, buf.len);
        @memcpy(buf[0..n], txt[0..n]);
        return buf[0..n];
    }

    pub fn lastInsertRowId(self: *const Db) i64 {
        return c.sqlite3_last_insert_rowid(self.handle);
    }

    pub fn changes(self: *const Db) i64 {
        return c.sqlite3_changes64(self.handle);
    }
};

/// Owns a prepared statement. Bind indices are 1-based, column indices 0-based.
pub const Stmt = struct {
    stmt: *c.sqlite3_stmt,

    pub fn finalize(self: *Stmt) void {
        _ = c.sqlite3_finalize(self.stmt);
    }

    pub fn reset(self: *Stmt) void {
        _ = c.sqlite3_reset(self.stmt);
        _ = c.sqlite3_clear_bindings(self.stmt);
    }

    pub fn bindI64(self: *Stmt, idx: c_int, v: i64) Error!void {
        if (c.sqlite3_bind_int64(self.stmt, idx, v) != c.SQLITE_OK) return error.SqliteBind;
    }

    pub fn bindF64(self: *Stmt, idx: c_int, v: f64) Error!void {
        if (c.sqlite3_bind_double(self.stmt, idx, v) != c.SQLITE_OK) return error.SqliteBind;
    }

    pub fn bindText(self: *Stmt, idx: c_int, v: []const u8) Error!void {
        if (zkb_bind_text_transient(self.stmt, idx, v.ptr, @intCast(v.len)) != c.SQLITE_OK)
            return error.SqliteBind;
    }

    pub fn bindBlob(self: *Stmt, idx: c_int, v: []const u8) Error!void {
        if (zkb_bind_blob_transient(self.stmt, idx, v.ptr, @intCast(v.len)) != c.SQLITE_OK)
            return error.SqliteBind;
    }

    /// Bind a vector for vec0. sqlite-vec reads FLOAT[N] columns as a raw
    /// little-endian f32 blob, so this is a straight reinterpret — no copy.
    pub fn bindVector(self: *Stmt, idx: c_int, v: []const f32) Error!void {
        return self.bindBlob(idx, std.mem.sliceAsBytes(v));
    }

    pub fn bindNull(self: *Stmt, idx: c_int) Error!void {
        if (c.sqlite3_bind_null(self.stmt, idx) != c.SQLITE_OK) return error.SqliteBind;
    }

    /// 1-based index of a named parameter (`:days`), or null if the statement
    /// has no such parameter. Needed to bind by name rather than by position:
    /// a saved query's parameters are written by a human in whatever order,
    /// and matching them positionally would silently pair the wrong values.
    pub fn parameterIndex(self: *Stmt, name: []const u8) ?c_int {
        var buf: [128]u8 = undefined;
        if (name.len + 2 > buf.len) return null;
        buf[0] = ':';
        @memcpy(buf[1 .. name.len + 1], name);
        buf[name.len + 1] = 0;
        const idx = c.sqlite3_bind_parameter_index(self.stmt, &buf);
        return if (idx == 0) null else idx;
    }

    /// Number of parameters the statement declares.
    pub fn parameterCount(self: *Stmt) c_int {
        return c.sqlite3_bind_parameter_count(self.stmt);
    }

    /// Name of the i-th parameter (1-based), including its `:` prefix.
    pub fn parameterName(self: *Stmt, i: c_int) ?[]const u8 {
        const p = c.sqlite3_bind_parameter_name(self.stmt, i) orelse return null;
        return std.mem.span(p);
    }

    /// True = a row is available, false = done.
    pub fn step(self: *Stmt) Error!bool {
        return switch (c.sqlite3_step(self.stmt)) {
            c.SQLITE_ROW => true,
            c.SQLITE_DONE => false,
            else => error.SqliteStep,
        };
    }

    pub fn columnI64(self: *Stmt, idx: c_int) i64 {
        return c.sqlite3_column_int64(self.stmt, idx);
    }

    pub fn columnF64(self: *Stmt, idx: c_int) f64 {
        return c.sqlite3_column_double(self.stmt, idx);
    }

    pub fn columnIsNull(self: *Stmt, idx: c_int) bool {
        return c.sqlite3_column_type(self.stmt, idx) == c.SQLITE_NULL;
    }

    /// Valid until the next step()/reset()/finalize() on this statement.
    pub fn columnText(self: *Stmt, idx: c_int) []const u8 {
        const p = c.sqlite3_column_text(self.stmt, idx);
        if (p == null) return &.{};
        const n: usize = @intCast(c.sqlite3_column_bytes(self.stmt, idx));
        return p[0..n];
    }

    pub fn columnCount(self: *Stmt) c_int {
        return c.sqlite3_column_count(self.stmt);
    }

    /// The name SQLite gives the result column — the alias if there is one, the
    /// expression text otherwise. Only meaningful for a header row.
    pub fn columnName(self: *Stmt, idx: c_int) []const u8 {
        const p = c.sqlite3_column_name(self.stmt, idx);
        if (p == null) return &.{};
        return std.mem.span(p);
    }

    /// SQLite's own verdict on whether this statement writes anything.
    ///
    /// Used by `zkb sql` as a second gate behind the read-only connection: a
    /// keyword check on the source text can be fooled, and this cannot — it is
    /// derived from the compiled program, not from how the SQL was spelled.
    pub fn isReadOnly(self: *Stmt) bool {
        return c.sqlite3_stmt_readonly(self.stmt) != 0;
    }

    /// Valid until the next step()/reset()/finalize() on this statement.
    pub fn columnBlob(self: *Stmt, idx: c_int) []const u8 {
        const p = c.sqlite3_column_blob(self.stmt, idx);
        if (p == null) return &.{};
        const n: usize = @intCast(c.sqlite3_column_bytes(self.stmt, idx));
        const bytes: [*]const u8 = @ptrCast(p);
        return bytes[0..n];
    }
};

pub fn libVersion() []const u8 {
    return std.mem.span(c.sqlite3_libversion());
}

pub fn vecVersion() []const u8 {
    return std.mem.span(@as([*:0]const u8, c.SQLITE_VEC_VERSION));
}

/// True if the amalgamation was compiled with the given option, e.g.
/// "ENABLE_FTS5". Uses pragma_compile_options rather than trusting build.zig.
pub fn hasCompileOption(db: *Db, needle: []const u8) bool {
    var st = db.prepare("SELECT 1 FROM pragma_compile_options() WHERE compile_options = ?1") catch return false;
    defer st.finalize();
    st.bindText(1, needle) catch return false;
    return st.step() catch false;
}
