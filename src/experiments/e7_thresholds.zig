//! E7 — calibrate `dup_threshold` and `island_threshold` by judgement.
//!
//! Same method as E2: produce a pool of candidates well below any plausible
//! threshold, judge each one, then pick the threshold from the judgements rather
//! than from taste. A maintenance threshold picked by taste is the thing that
//! makes a report either noise or empty, and neither failure announces itself.
//!
//! Emits JSON on stdout: the pairs sorted by cosine, and the least-connected
//! chunks, each with enough text to judge without opening the files.
//!
//! Uses no model — the vectors are already in the index.

const std = @import("std");
const zkb = @import("zkb");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;

    var out_buf: [1 << 20]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    const w = &stdout.interface;
    defer w.flush() catch {};

    var pool_threshold: f64 = 0.80;
    var island_pool: f64 = 0.75;
    var args = init.minimal.args.iterate();
    _ = args.next();
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--pool")) {
            pool_threshold = std.fmt.parseFloat(f64, args.next() orelse "0.80") catch 0.80;
        } else if (std.mem.eql(u8, a, "--island-pool")) {
            island_pool = std.fmt.parseFloat(f64, args.next() orelse "0.75") catch 0.75;
        }
    }

    var layout = try zkb.paths.resolve(gpa, init.environ_map);
    defer layout.deinit(gpa);
    const db_path = try gpa.dupeZ(u8, layout.db);
    defer gpa.free(db_path);
    var db = try zkb.store.open(db_path, .read_only);
    defer db.close();

    const started = std.Io.Timestamp.now(init.io, .awake).nanoseconds;
    // The pool is deliberately far below any threshold that would ship: the
    // point is to see what a lower threshold *would* have admitted, which is
    // exactly what a run at the shipping threshold cannot show.
    var result = try zkb.maintain_vec.run(gpa, &db, .{
        .dup_threshold = pool_threshold,
        .island_threshold = island_pool,
        .max_pairs = 200,
    });
    defer result.deinit(gpa);
    const elapsed_ms = @divTrunc(
        std.Io.Timestamp.now(init.io, .awake).nanoseconds - started,
        std.time.ns_per_ms,
    );

    try w.print(
        "{{\"chunks\":{d},\"elapsed_ms\":{d},\"pool_threshold\":{d:.2}," ++
            "\"island_pool\":{d:.2},\"truncated\":{},\"pairs\":[",
        .{ result.chunks_scanned, elapsed_ms, pool_threshold, island_pool, result.truncated },
    );
    for (result.pairs, 0..) |p, i| {
        if (i != 0) try w.writeAll(",\n");
        try w.print("{{\"cos\":{d:.4},\"kind\":\"{t}\",\"a\":", .{ p.cos, p.kind });
        try std.json.Stringify.value(p.a_path, .{}, w);
        try w.writeAll(",\"a_heading\":");
        try std.json.Stringify.value(p.a_heading, .{}, w);
        try w.writeAll(",\"a_text\":");
        try std.json.Stringify.value(p.a_excerpt, .{}, w);
        try w.writeAll(",\"b\":");
        try std.json.Stringify.value(p.b_path, .{}, w);
        try w.writeAll(",\"b_heading\":");
        try std.json.Stringify.value(p.b_heading, .{}, w);
        try w.writeAll(",\"b_text\":");
        try std.json.Stringify.value(p.b_excerpt, .{}, w);
        try w.writeAll("}");
    }
    try w.writeAll("],\n\"islands\":[");
    for (result.islands, 0..) |is, i| {
        if (i != 0) try w.writeAll(",\n");
        try w.print("{{\"cos\":{d:.4},\"path\":", .{is.cos});
        try std.json.Stringify.value(is.path, .{}, w);
        try w.writeAll(",\"heading\":");
        try std.json.Stringify.value(is.heading, .{}, w);
        try w.writeAll(",\"text\":");
        try std.json.Stringify.value(is.excerpt, .{}, w);
        try w.writeAll(",\"nearest\":");
        try std.json.Stringify.value(is.nearest_path, .{}, w);
        try w.writeAll("}");
    }
    try w.writeAll("]}\n");
    return 0;
}
