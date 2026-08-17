//! Priority queue in front of the embedder.
//!
//! A single `llama_context` cannot be used concurrently, so every embedding
//! request has to serialize somewhere. A plain mutex would serialize them in
//! arrival order, which means a background index of ~1500 chunks starves
//! interactive queries for minutes. That is not an optimization to add later —
//! it is the difference between a usable daemon and an unusable one (SPEC §3.4).
//!
//! Two levels, strict priority: an interactive job never waits behind an ingest
//! job. Enqueue granularity is **one chunk**, not one document, so the longest a
//! query can wait is a single embed call (~200ms measured) rather than a whole
//! document.
//!
//! No allocation: jobs are intrusively linked and normally live on the caller's
//! stack until completion.
//!
//! `io` is threaded through every operation because Zig 0.16's synchronisation
//! primitives live under `std.Io` and take it explicitly. The uncancelable
//! variants are used deliberately: a half-released mutex or a waiter abandoned
//! mid-embed would be worse than blocking.

const std = @import("std");

pub const Priority = enum { interactive, ingest };

pub const Kind = enum {
    /// Heading path prefixed, no instruction prefix.
    document,
    /// Instruction prefix applied (SPEC §3.3).
    query,
    /// Token count only, no forward pass. Routed through the queue rather than
    /// called directly so that **every** llama.cpp access happens on the embedder
    /// thread by construction. `llama_tokenize` only reads the immutable vocab
    /// and would probably be safe to call concurrently, but "probably safe"
    /// is not a property worth relying on for a data race.
    count_tokens,
};

pub const Job = struct {
    kind: Kind,
    /// Borrowed; must outlive the job.
    text: []const u8,
    /// Only used for `.document`.
    heading_path: []const u8 = "",
    /// Caller-owned destination, at least `n_embd` long. Unused for
    /// `.count_tokens`.
    out: []f32 = &.{},
    /// Result of a `.count_tokens` job.
    count: usize = 0,

    /// Set by the worker before signalling completion.
    err: ?anyerror = null,
    /// Completion signal. A semaphore starting at 0: the worker posts once, the
    /// waiter waits once.
    done: std.Io.Semaphore = .{ .permits = 0 },

    next: ?*Job = null,

    /// Block until the worker has finished this job.
    pub fn wait(self: *Job, io: std.Io) !void {
        self.done.waitUncancelable(io);
        if (self.err) |e| return e;
    }

    fn complete(self: *Job, io: std.Io, err: ?anyerror) void {
        self.err = err;
        self.done.post(io);
    }
};

const List = struct {
    head: ?*Job = null,
    tail: ?*Job = null,
    len: usize = 0,

    fn push(self: *List, job: *Job) void {
        job.next = null;
        if (self.tail) |t| t.next = job else self.head = job;
        self.tail = job;
        self.len += 1;
    }

    fn pop(self: *List) ?*Job {
        const job = self.head orelse return null;
        self.head = job.next;
        if (self.head == null) self.tail = null;
        job.next = null;
        self.len -= 1;
        return job;
    }
};

pub const Queue = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    interactive: List = .{},
    ingest: List = .{},
    closed: bool = false,

    /// Statistics, for `zkb status` and for proving the priority actually works.
    served_interactive: usize = 0,
    served_ingest: usize = 0,
    /// Largest number of ingest jobs ever skipped over to serve an interactive
    /// one. A zero here across a full index would mean the queue never actually
    /// had to preempt anything, i.e. the test did not exercise it.
    max_preempted: usize = 0,

    pub fn push(self: *Queue, io: std.Io, job: *Job, prio: Priority) error{QueueClosed}!void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.closed) return error.QueueClosed;
        switch (prio) {
            .interactive => self.interactive.push(job),
            .ingest => self.ingest.push(job),
        }
        self.cond.signal(io);
    }

    /// Blocks until a job is available. Returns null once the queue is closed and
    /// drained, which is the worker thread's exit signal.
    pub fn pop(self: *Queue, io: std.Io) ?*Job {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (true) {
            if (self.interactive.pop()) |job| {
                self.served_interactive += 1;
                // How deep the ingest backlog was when we jumped it.
                self.max_preempted = @max(self.max_preempted, self.ingest.len);
                return job;
            }
            if (self.ingest.pop()) |job| {
                self.served_ingest += 1;
                return job;
            }
            if (self.closed) return null;
            self.cond.waitUncancelable(io, &self.mutex);
        }
    }

    /// Stop accepting work and wake the worker. Jobs already queued are still
    /// drained, so nothing in flight is abandoned half-done.
    pub fn close(self: *Queue, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed = true;
        self.cond.broadcast(io);
    }

    /// Fail and release every queued job without running it. Used when the
    /// embedder itself is unavailable — waiters must not block forever.
    pub fn drainWithError(self: *Queue, io: std.Io, err: anyerror) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        for ([_]*List{ &self.interactive, &self.ingest }) |list| {
            while (list.pop()) |job| job.complete(io, err);
        }
    }

    /// Hand a finished job back to its waiter.
    pub fn finish(_: *Queue, io: std.Io, job: *Job, err: ?anyerror) void {
        job.complete(io, err);
    }

    pub const Depth = struct { interactive: usize, ingest: usize };

    pub fn depth(self: *Queue, io: std.Io) Depth {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return .{ .interactive = self.interactive.len, .ingest = self.ingest.len };
    }
};
