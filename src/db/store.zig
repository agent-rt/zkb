//! Storage operations over the three-table chunk representation.
//!
//! `chunks.id` == `fts_chunks.rowid` == `vec_chunks.chunk_id`. One integer ties
//! all three together, so fusion needs no mapping table (SPEC §2.4).
//!
//! **The invariant this module exists to protect:** virtual tables do not
//! participate in `ON DELETE CASCADE`. Deleting a doc's row does not touch
//! fts_chunks or vec_chunks, and a missed delete shows up only as "search
//! returns content that was deleted" — a silent wrong answer, never an error.
//! So chunk writes and deletes each go through exactly one function here, and
//! nothing outside this file may INSERT or DELETE on those tables.

const std = @import("std");
const sqlite = @import("sqlite.zig");
const schema = @import("schema.zig");

pub const Error = sqlite.Error;

pub const DocRow = struct {
    id: i64,
    collection_id: i64,
    content_sha: [64]u8,
    size: i64,
    mtime_ms: i64,
    indexed_at: ?i64,
};

pub const ChunkInput = struct {
    idx: i64,
    heading_path: []const u8,
    byte_start: i64,
    byte_end: i64,
    n_tokens: i64,
    text: []const u8,
};

pub const Store = struct {
    db: *sqlite.Db,

    pub fn init(db: *sqlite.Db) Store {
        return .{ .db = db };
    }

    // ------------------------------------------------------------ transactions

    pub fn begin(self: *Store) Error!void {
        return self.db.exec("BEGIN IMMEDIATE;");
    }
    pub fn commit(self: *Store) Error!void {
        return self.db.exec("COMMIT;");
    }
    pub fn rollback(self: *Store) void {
        self.db.exec("ROLLBACK;") catch {};
    }

    // ------------------------------------------------------------ collections

    /// How a collection's files are parsed and who may write them.
    pub const Kind = enum { documents, memory, records };

    pub fn ensureCollection(self: *Store, name: []const u8, root: []const u8, now_ms: i64) Error!i64 {
        return self.ensureCollectionKind(name, root, .documents, now_ms);
    }

    /// The collection by this name, with this kind — creating it if it is not
    /// there and **correcting it if it is there with the wrong one**.
    ///
    /// It used to return the existing row untouched, which made the name a lie
    /// and left no way back from agent-rt/zkb#1: `zkb index` had turned the
    /// `memory` collection into a `documents` one, and since the kind is what
    /// supplies the built-in filters, `recall` began answering from the 25
    /// retired memories under `archive/`. Nothing repaired it — not `zkb
    /// memory`, not a re-index — because every path that knew the right kind
    /// stopped at "the row exists".
    ///
    /// The callers are the ones that own their kind: `memory`, `records`, and
    /// the known roots in `index_cmd`. A caller that does not know had no
    /// business calling this.
    pub fn ensureCollectionKind(
        self: *Store,
        name: []const u8,
        root: []const u8,
        kind: Kind,
        now_ms: i64,
    ) Error!i64 {
        // The lookup gets a scope of its own so its `defer` runs before the
        // repair statement is prepared: `finalize` does not clear the handle,
        // so finalizing the same statement twice hands SQLite a pointer it has
        // already freed.
        const found: ?struct { id: i64, kind: Kind } = blk: {
            var st = try self.db.prepare("SELECT id, kind FROM collections WHERE name = ?1");
            defer st.finalize();
            try st.bindText(1, name);
            if (!try st.step()) break :blk null;
            break :blk .{
                .id = st.columnI64(0),
                .kind = std.meta.stringToEnum(Kind, st.columnText(1)) orelse .documents,
            };
        };
        if (found) |row| {
            if (row.kind != kind) {
                {
                    var fix = try self.db.prepare("UPDATE collections SET kind = ?2 WHERE id = ?1");
                    defer fix.finalize();
                    try fix.bindI64(1, row.id);
                    try fix.bindText(2, @tagName(kind));
                    _ = try fix.step();
                }
                // The kind is not a label on the row, it decides whether the
                // per-chunk projections get written at all (`indexer.indexOne`).
                // So a wrong kind does not merely misname a collection — every
                // document re-indexed under it had its `rec_memory` row deleted
                // and not rewritten, and correcting the row here cannot bring one
                // of them back. Only re-indexing can, and nothing else would ever
                // mark them dirty: the files did not change, so `scan` sees them
                // as unchanged forever.
                //
                // Measured: a memory collection re-indexed under the wrong kind
                // lost all 40 projection rows. `zkb status` kept reporting 40
                // documents, `doctor` passed every check, and `recall` returned
                // nothing for two days. Clearing the stamps is what makes the fix
                // reach the data the wrong kind destroyed.
                var requeue = try self.db.prepare(
                    "UPDATE docs SET indexed_at = NULL, index_error = NULL WHERE collection_id = ?1",
                );
                defer requeue.finalize();
                try requeue.bindI64(1, row.id);
                _ = try requeue.step();
            }
            return row.id;
        }
        var st = try self.db.prepare(
            "INSERT INTO collections(name, root, kind, created_at) VALUES (?1, ?2, ?3, ?4)",
        );
        defer st.finalize();
        try st.bindText(1, name);
        try st.bindText(2, root);
        try st.bindText(3, @tagName(kind));
        try st.bindI64(4, now_ms);
        _ = try st.step();
        return self.db.lastInsertRowId();
    }

    /// Register a collection, or point an existing one at a new root and scan
    /// configuration.
    ///
    /// Unlike `ensureCollectionKind`, this updates instead of returning early.
    /// Re-running `zkb index --root NEW --collection existing` used to keep the
    /// old root and report success, which is the same silent-ignore shape as a
    /// dropped filter: the command looks like it worked and nothing moved.
    ///
    /// The root is not compared to the old one and docs are not deleted. A moved
    /// root makes every rel_path vanish from disk, and the next reconcile deletes
    /// them for the ordinary reason — no special case needed here.
    ///
    /// `extensions` and `include` are newline-separated. NULL means "the built-in
    /// default"; passing them through unchanged is what lets a caller update the
    /// root without having to restate the filters.
    pub fn upsertCollection(
        self: *Store,
        name: []const u8,
        root: []const u8,
        kind: Kind,
        extensions: ?[]const u8,
        include: ?[]const u8,
        now_ms: i64,
    ) Error!i64 {
        if (try self.findCollection(name)) |id| {
            // `kind` is deliberately not in the SET list. It is what the row is,
            // not what a caller is asking for: the CLI always passes
            // `.documents`, so re-indexing a `memory` or `records` collection
            // used to turn it into a documents collection — and the kind is
            // what supplies the built-in filters, so a `memory` collection
            // silently stopped skipping `archive/` and `recall` began answering
            // from memories that had been retired. See agent-rt/zkb#1.
            //
            // `extensions` and `include` coalesce for the same reason: a value
            // the caller never mentioned is not a value they asked to clear.
            var st = try self.db.prepare(
                \\UPDATE collections
                \\   SET root = ?2,
                \\       extensions = coalesce(?4, extensions),
                \\       include    = coalesce(?5, include)
                \\ WHERE id = ?1
            );
            defer st.finalize();
            try st.bindI64(1, id);
            try st.bindText(2, root);
            if (extensions) |v| try st.bindText(4, v) else try st.bindNull(4);
            if (include) |v| try st.bindText(5, v) else try st.bindNull(5);
            _ = try st.step();
            return id;
        }

        var st = try self.db.prepare(
            \\INSERT INTO collections(name, root, kind, created_at, extensions, include)
            \\VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        );
        defer st.finalize();
        try st.bindText(1, name);
        try st.bindText(2, root);
        try st.bindText(3, @tagName(kind));
        try st.bindI64(4, now_ms);
        if (extensions) |v| try st.bindText(5, v) else try st.bindNull(5);
        if (include) |v| try st.bindText(6, v) else try st.bindNull(6);
        _ = try st.step();
        return self.db.lastInsertRowId();
    }

    /// One row of what to scan. Strings are owned by the caller's arena.
    pub const CollectionRow = struct {
        id: i64,
        name: []const u8,
        root: []const u8,
        kind: Kind,
        /// Newline-separated, or null for the built-in default.
        extensions: ?[]const u8,
        include: ?[]const u8,
    };

    /// Every collection, ordered by id so the built-ins keep coming first and the
    /// scan order stays stable across runs.
    ///
    /// This is the list the ingest loop walks. It being a query rather than a
    /// literal is the whole point of v7: a collection registered by any means is
    /// picked up by whatever rescans, without that code having to be edited.
    pub fn listCollections(
        self: *Store,
        arena: std.mem.Allocator,
    ) Error![]CollectionRow {
        var out: std.ArrayList(CollectionRow) = .empty;
        var st = try self.db.prepare(
            "SELECT id, name, root, kind, extensions, include FROM collections ORDER BY id",
        );
        defer st.finalize();
        while (try st.step()) {
            try out.append(arena, .{
                .id = st.columnI64(0),
                .name = try arena.dupe(u8, st.columnText(1)),
                .root = try arena.dupe(u8, st.columnText(2)),
                .kind = std.meta.stringToEnum(Kind, st.columnText(3)) orelse .documents,
                .extensions = if (st.columnIsNull(4)) null else try arena.dupe(u8, st.columnText(4)),
                .include = if (st.columnIsNull(5)) null else try arena.dupe(u8, st.columnText(5)),
            });
        }
        return out.items;
    }

    /// One collection by name, as it is stored.
    ///
    /// Exists so a command can reuse what a collection already says about
    /// itself instead of inventing a default for anything the caller left out.
    pub fn collectionByName(
        self: *Store,
        arena: std.mem.Allocator,
        name: []const u8,
    ) Error!?CollectionRow {
        var st = try self.db.prepare(
            "SELECT id, name, root, kind, extensions, include FROM collections WHERE name = ?1",
        );
        defer st.finalize();
        try st.bindText(1, name);
        if (!try st.step()) return null;
        return .{
            .id = st.columnI64(0),
            .name = try arena.dupe(u8, st.columnText(1)),
            .root = try arena.dupe(u8, st.columnText(2)),
            .kind = std.meta.stringToEnum(Kind, st.columnText(3)) orelse .documents,
            .extensions = if (st.columnIsNull(4)) null else try arena.dupe(u8, st.columnText(4)),
            .include = if (st.columnIsNull(5)) null else try arena.dupe(u8, st.columnText(5)),
        };
    }

    pub fn collectionKind(self: *Store, id: i64) Error!Kind {
        var st = try self.db.prepare("SELECT kind FROM collections WHERE id = ?1");
        defer st.finalize();
        try st.bindI64(1, id);
        if (!try st.step()) return .documents;
        return std.meta.stringToEnum(Kind, st.columnText(0)) orelse .documents;
    }

    /// Replace a collection's list of switched-off health checks.
    ///
    /// The whole list, not one flag at a time: the caller states the end state it
    /// wants, so reading the current value and reasoning about the difference is
    /// never part of setting it.
    pub fn setChecksOff(self: *Store, id: i64, csv: []const u8) Error!void {
        var st = try self.db.prepare("UPDATE collections SET checks_off = ?2 WHERE id = ?1");
        defer st.finalize();
        try st.bindI64(1, id);
        if (csv.len == 0) try st.bindNull(2) else try st.bindText(2, csv);
        _ = try st.step();
    }

    pub fn findCollection(self: *Store, name: []const u8) Error!?i64 {
        var st = try self.db.prepare("SELECT id FROM collections WHERE name = ?1");
        defer st.finalize();
        try st.bindText(1, name);
        if (!try st.step()) return null;
        return st.columnI64(0);
    }

    // ------------------------------------------------------------------- docs

    pub fn findDoc(self: *Store, collection_id: i64, rel_path: []const u8) Error!?DocRow {
        var st = try self.db.prepare(
            \\SELECT id, collection_id, content_sha, size, mtime_ms, indexed_at
            \\FROM docs WHERE collection_id = ?1 AND rel_path = ?2
        );
        defer st.finalize();
        try st.bindI64(1, collection_id);
        try st.bindText(2, rel_path);
        if (!try st.step()) return null;

        var row: DocRow = .{
            .id = st.columnI64(0),
            .collection_id = st.columnI64(1),
            .content_sha = @splat(0),
            .size = st.columnI64(3),
            .mtime_ms = st.columnI64(4),
            .indexed_at = if (st.columnIsNull(5)) null else st.columnI64(5),
        };
        const sha = st.columnText(2);
        @memcpy(row.content_sha[0..@min(sha.len, 64)], sha[0..@min(sha.len, 64)]);
        return row;
    }

    /// Look for a doc with this content anywhere in the collection whose file no
    /// longer exists at its recorded path — i.e. a rename. Returns its id so the
    /// caller can move the path instead of re-embedding identical content.
    /// A document elsewhere in this collection with the same bytes.
    ///
    /// The path comes back with the id because same content at another path is not
    /// proof of a rename — the caller has to check whether that path still exists.
    /// Two byte-identical files in one collection is an ordinary thing (the same
    /// memory written in two projects), and treating the second as a move of the
    /// first left one of them out of the index entirely.
    pub const ShaMatch = struct { id: i64, rel_path: []const u8 };

    pub fn findDocByShaExcludingPath(
        self: *Store,
        arena: std.mem.Allocator,
        collection_id: i64,
        sha: []const u8,
        exclude_rel_path: []const u8,
    ) Error!?ShaMatch {
        var st = try self.db.prepare(
            \\SELECT id, rel_path FROM docs
            \\WHERE collection_id = ?1 AND content_sha = ?2 AND rel_path != ?3
            \\LIMIT 1
        );
        defer st.finalize();
        try st.bindI64(1, collection_id);
        try st.bindText(2, sha);
        try st.bindText(3, exclude_rel_path);
        if (!try st.step()) return null;
        return .{ .id = st.columnI64(0), .rel_path = try arena.dupe(u8, st.columnText(1)) };
    }

    /// Record a file's identity (path + content fingerprint) and mark it as
    /// needing (re)indexing. Deliberately does not touch title/frontmatter:
    /// those come from parsing, which the scanner does not do. `indexed_at =
    /// NULL` makes the docs table itself the work queue, so an interrupted run
    /// resumes without extra bookkeeping.
    pub fn upsertDocContent(
        self: *Store,
        collection_id: i64,
        rel_path: []const u8,
        sha: []const u8,
        size: i64,
        mtime_ms: i64,
    ) Error!i64 {
        var st = try self.db.prepare(
            \\INSERT INTO docs(collection_id, rel_path, content_sha, size, mtime_ms,
            \\                 chunk_count, indexed_at, index_error)
            \\VALUES (?1, ?2, ?3, ?4, ?5, 0, NULL, NULL)
            \\ON CONFLICT(collection_id, rel_path) DO UPDATE SET
            \\  content_sha = ?3, size = ?4, mtime_ms = ?5,
            \\  indexed_at = NULL, index_error = NULL
            \\RETURNING id
        );
        defer st.finalize();
        try st.bindI64(1, collection_id);
        try st.bindText(2, rel_path);
        try st.bindText(3, sha);
        try st.bindI64(4, size);
        try st.bindI64(5, mtime_ms);
        if (!try st.step()) return error.SqliteStep;
        return st.columnI64(0);
    }

    /// Set the parsed metadata. Called by the indexer, which is the only stage
    /// that actually reads the Markdown.
    pub fn setDocMeta(
        self: *Store,
        doc_id: i64,
        title: ?[]const u8,
        frontmatter: ?[]const u8,
    ) Error!void {
        var st = try self.db.prepare("UPDATE docs SET title = ?2, frontmatter = ?3 WHERE id = ?1");
        defer st.finalize();
        try st.bindI64(1, doc_id);
        if (title) |t| try st.bindText(2, t) else try st.bindNull(2);
        if (frontmatter) |f| try st.bindText(3, f) else try st.bindNull(3);
        _ = try st.step();
    }

    pub const PendingDoc = struct { id: i64, rel_path: []const u8 };

    /// Docs awaiting indexing, oldest first. `rel_path` is copied into `arena`.
    pub fn listPending(
        self: *Store,
        arena: std.mem.Allocator,
        collection_id: i64,
        limit: usize,
    ) !std.ArrayList(PendingDoc) {
        var out: std.ArrayList(PendingDoc) = .empty;
        var st = try self.db.prepare(
            \\SELECT id, rel_path FROM docs
            \\WHERE collection_id = ?1 AND indexed_at IS NULL AND index_error IS NULL
            \\ORDER BY id LIMIT ?2
        );
        defer st.finalize();
        try st.bindI64(1, collection_id);
        try st.bindI64(2, @intCast(limit));
        while (try st.step()) {
            try out.append(arena, .{
                .id = st.columnI64(0),
                .rel_path = try arena.dupe(u8, st.columnText(1)),
            });
        }
        return out;
    }

    /// Every path recorded for a collection, so the scanner can spot files that
    /// disappeared. Copied into `arena`.
    pub fn listAllPaths(
        self: *Store,
        arena: std.mem.Allocator,
        collection_id: i64,
    ) !std.ArrayList([]const u8) {
        var out: std.ArrayList([]const u8) = .empty;
        var st = try self.db.prepare("SELECT rel_path FROM docs WHERE collection_id = ?1");
        defer st.finalize();
        try st.bindI64(1, collection_id);
        while (try st.step()) try out.append(arena, try arena.dupe(u8, st.columnText(0)));
        return out;
    }

    /// Rename: path moves, content and therefore vectors stay. Deliberately does
    /// not clear indexed_at — re-embedding identical bytes would be pure waste
    /// (SPEC §4.1).
    pub fn moveDoc(self: *Store, doc_id: i64, new_rel_path: []const u8, mtime_ms: i64) Error!void {
        var st = try self.db.prepare("UPDATE docs SET rel_path = ?2, mtime_ms = ?3 WHERE id = ?1");
        defer st.finalize();
        try st.bindI64(1, doc_id);
        try st.bindText(2, new_rel_path);
        try st.bindI64(3, mtime_ms);
        _ = try st.step();
    }

    pub fn touchDoc(self: *Store, doc_id: i64, mtime_ms: i64) Error!void {
        var st = try self.db.prepare("UPDATE docs SET mtime_ms = ?2 WHERE id = ?1");
        defer st.finalize();
        try st.bindI64(1, doc_id);
        try st.bindI64(2, mtime_ms);
        _ = try st.step();
    }

    pub fn markIndexed(self: *Store, doc_id: i64, chunk_count: i64, now_ms: i64) Error!void {
        var st = try self.db.prepare(
            "UPDATE docs SET chunk_count = ?2, indexed_at = ?3, index_error = NULL WHERE id = ?1",
        );
        defer st.finalize();
        try st.bindI64(1, doc_id);
        try st.bindI64(2, chunk_count);
        try st.bindI64(3, now_ms);
        _ = try st.step();
    }

    /// A half-indexed doc is worse than an unindexed one: the user would believe
    /// it was searched. Callers must have dropped its chunks before calling this.
    pub fn markFailed(self: *Store, doc_id: i64, reason: []const u8) Error!void {
        var st = try self.db.prepare(
            "UPDATE docs SET chunk_count = 0, indexed_at = NULL, index_error = ?2 WHERE id = ?1",
        );
        defer st.finalize();
        try st.bindI64(1, doc_id);
        try st.bindText(2, reason);
        _ = try st.step();
    }

    /// THE single delete path for a document.
    ///
    /// `links` is keyed on the doc rather than the chunk, so it does not belong
    /// in deleteChunks and was missed there: a doc that had emitted any link
    /// could not be deleted at all, because `links.doc_id` held a foreign key
    /// on it. The delete failed *after* the chunks were gone, so the index was
    /// left with a doc row that had no content, and `zkb forget` reported
    /// success on the file it had already moved.
    pub fn deleteDoc(self: *Store, doc_id: i64) Error!void {
        try self.deleteChunks(doc_id);
        {
            var st = try self.db.prepare("DELETE FROM links WHERE doc_id = ?1");
            defer st.finalize();
            try st.bindI64(1, doc_id);
            _ = try st.step();
        }
        // Inbound references have no foreign key, so nothing would have caught
        // them — and doc ids are reused, since they are INTEGER PRIMARY KEY.
        // Left alone, a link into this document would silently start resolving
        // to whichever unrelated document is created next. Nulling the target
        // hands it back to the link resolver, which either re-resolves it or
        // reports it as broken.
        {
            var st = try self.db.prepare("UPDATE links SET target_doc_id = NULL WHERE target_doc_id = ?1");
            defer st.finalize();
            try st.bindI64(1, doc_id);
            _ = try st.step();
        }
        var st = try self.db.prepare("DELETE FROM docs WHERE id = ?1");
        defer st.finalize();
        try st.bindI64(1, doc_id);
        _ = try st.step();
    }

    // ----------------------------------------------------------------- chunks

    /// THE single delete path for chunks. Removes the rows from every table
    /// keyed on a chunk. Virtual tables are not reached by foreign keys, so this
    /// must never be bypassed — see the module comment.
    ///
    /// Schema v4 added `facts` and `rec_memory`, two more per-chunk projections,
    /// and forgetting them here made `DELETE FROM docs` fail on a foreign key —
    /// after the chunks were already gone, leaving a doc row with no content.
    /// The FK was doing its job; the invariant is that *this* function knows
    /// every table a chunk touches, so anything added later belongs here too.
    /// Drop chunks whose document no longer exists, returning how many went.
    ///
    /// This is the invariant at the top of this file, swept rather than asserted.
    /// Nothing in the current code produces such a row — `deleteDoc` removes
    /// chunks before the document, and `upsertDocContent` keeps the id through
    /// ON CONFLICT — but a real index accumulated 1198 of them (23% of all
    /// chunks) as residue from an older binary during a bulk move, and they were
    /// invisible until a query happened to rank one into its top-k.
    ///
    /// Reuses `deleteChunks` per document rather than writing a parallel bulk
    /// delete: the five-table order in there is the whole point of this module,
    /// and a second copy of it would be a second thing to get wrong.
    ///
    /// Cleanup, not diagnosis: the index is derived data and keeping it
    /// self-consistent is zkb's own job (SPEC §14.0). `maintain` reports the
    /// condition and modifies nothing; this is what actually fixes it.
    pub fn deleteOrphanChunks(self: *Store, gpa: std.mem.Allocator) Error!usize {
        var ids: std.ArrayList(i64) = .empty;
        defer ids.deinit(gpa);
        {
            var st = try self.db.prepare(
                \\SELECT DISTINCT c.doc_id FROM chunks c
                \\LEFT JOIN docs d ON d.id = c.doc_id
                \\WHERE d.id IS NULL
            );
            defer st.finalize();
            // Collected before deleting: mutating the chunks table while stepping
            // over a statement that reads it invites a half-done sweep.
            while (try st.step()) try ids.append(gpa, st.columnI64(0));
        }

        var removed: usize = 0;
        for (ids.items) |doc_id| {
            var st = try self.db.prepare("SELECT count(*) FROM chunks WHERE doc_id = ?1");
            try st.bindI64(1, doc_id);
            const n: i64 = if (try st.step()) st.columnI64(0) else 0;
            st.finalize();

            try self.deleteChunks(doc_id);
            removed += @intCast(n);
        }
        return removed;
    }

    /// Remove a collection and every document in it.
    ///
    /// Per document rather than by bulk DELETE, because `deleteDoc` is what knows
    /// the five-table order and the link fixups; a second copy of that is a second
    /// thing to get wrong. Returns how many documents went.
    ///
    /// Nothing could remove a collection before this. `zkb index --collection X`
    /// created one and there was no way back, so undoing a collection registered
    /// by mistake meant deleting the index and re-embedding everything — which is
    /// exactly what a wrongly-registered subdirectory of ~/docs cost.
    pub fn deleteCollection(self: *Store, gpa: std.mem.Allocator, id: i64) Error!usize {
        var ids: std.ArrayList(i64) = .empty;
        defer ids.deinit(gpa);
        {
            var st = try self.db.prepare("SELECT id FROM docs WHERE collection_id = ?1");
            defer st.finalize();
            try st.bindI64(1, id);
            // Collected first: deleting while stepping over the same table is a
            // half-done sweep waiting to happen.
            while (try st.step()) try ids.append(gpa, st.columnI64(0));
        }
        for (ids.items) |doc_id| try self.deleteDoc(doc_id);

        var st = try self.db.prepare("DELETE FROM collections WHERE id = ?1");
        defer st.finalize();
        try st.bindI64(1, id);
        _ = try st.step();
        return ids.items.len;
    }

    pub fn deleteChunks(self: *Store, doc_id: i64) Error!void {
        // Projections first: they reference chunks, so they must go before the
        // rows they point at.
        {
            var st = try self.db.prepare("DELETE FROM facts WHERE doc_id = ?1");
            defer st.finalize();
            try st.bindI64(1, doc_id);
            _ = try st.step();
        }
        {
            var st = try self.db.prepare("DELETE FROM rec_memory WHERE doc_id = ?1");
            defer st.finalize();
            try st.bindI64(1, doc_id);
            _ = try st.step();
        }
        // Virtual tables first: if this fails we still have chunks rows to
        // retry from. Doing it the other way round would lose the id list.
        {
            var st = try self.db.prepare(
                "DELETE FROM fts_chunks WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?1)",
            );
            defer st.finalize();
            try st.bindI64(1, doc_id);
            _ = try st.step();
        }
        {
            var st = try self.db.prepare(
                "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?1)",
            );
            defer st.finalize();
            try st.bindI64(1, doc_id);
            _ = try st.step();
        }
        {
            var st = try self.db.prepare("DELETE FROM chunks WHERE doc_id = ?1");
            defer st.finalize();
            try st.bindI64(1, doc_id);
            _ = try st.step();
        }
    }

    /// THE single insert path for chunks: all three tables in one call so there
    /// is no way to write a chunk that is missing from the FTS or vector index.
    pub fn insertChunk(
        self: *Store,
        collection_id: i64,
        doc_id: i64,
        in: ChunkInput,
        embedding: []const f32,
    ) Error!i64 {
        const chunk_id = blk: {
            var st = try self.db.prepare(
                \\INSERT INTO chunks(doc_id, idx, heading_path, byte_start, byte_end, n_tokens, text)
                \\VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) RETURNING id
            );
            defer st.finalize();
            try st.bindI64(1, doc_id);
            try st.bindI64(2, in.idx);
            try st.bindText(3, in.heading_path);
            try st.bindI64(4, in.byte_start);
            try st.bindI64(5, in.byte_end);
            try st.bindI64(6, in.n_tokens);
            try st.bindText(7, in.text);
            if (!try st.step()) return error.SqliteStep;
            break :blk st.columnI64(0);
        };
        {
            var st = try self.db.prepare(
                "INSERT INTO fts_chunks(rowid, text, heading_path) VALUES (?1, ?2, ?3)",
            );
            defer st.finalize();
            try st.bindI64(1, chunk_id);
            try st.bindText(2, in.text);
            try st.bindText(3, in.heading_path);
            _ = try st.step();
        }
        {
            var st = try self.db.prepare(
                "INSERT INTO vec_chunks(collection_id, chunk_id, embedding) VALUES (?1, ?2, ?3)",
            );
            defer st.finalize();
            try st.bindI64(1, collection_id);
            try st.bindI64(2, chunk_id);
            try st.bindVector(3, embedding);
            _ = try st.step();
        }
        return chunk_id;
    }

    // ------------------------------------------------------------------ stats

    pub const Counts = struct {
        docs: i64 = 0,
        chunks: i64 = 0,
        pending: i64 = 0,
        failed: i64 = 0,
        fts_rows: i64 = 0,
        vec_rows: i64 = 0,
    };

    pub fn counts(self: *Store) Error!Counts {
        return .{
            .docs = (try self.db.queryI64("SELECT count(*) FROM docs")) orelse 0,
            .chunks = (try self.db.queryI64("SELECT count(*) FROM chunks")) orelse 0,
            .pending = (try self.db.queryI64("SELECT count(*) FROM docs WHERE indexed_at IS NULL AND index_error IS NULL")) orelse 0,
            .failed = (try self.db.queryI64("SELECT count(*) FROM docs WHERE index_error IS NOT NULL")) orelse 0,
            .fts_rows = (try self.db.queryI64("SELECT count(*) FROM fts_chunks")) orelse 0,
            .vec_rows = (try self.db.queryI64("SELECT count(*) FROM vec_chunks")) orelse 0,
        };
    }
};

/// Open a database ready for use.
///
/// Read-write connections migrate; read-only ones only verify. A read-only
/// connection physically cannot run DDL, so attempting to migrate there fails
/// somewhere inside a CREATE statement with an opaque message — `verify` returns
/// `error.SchemaStale` instead, which the CLI can turn into "run zkb index".
pub fn open(path: [:0]const u8, mode: sqlite.OpenMode) (schema.Error)!sqlite.Db {
    var db = try sqlite.Db.open(path, mode);
    errdefer db.close();
    switch (mode) {
        .read_write => {
            if (!std.mem.eql(u8, path, ":memory:")) try db.enableWal();
            try schema.migrate(&db);
        },
        .read_only => try schema.verify(&db),
    }
    return db;
}
