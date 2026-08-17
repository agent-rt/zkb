//! E3 — Is Qwen3-Embedding's pooling actually correct under llama.cpp?
//!
//! This is the project's most dangerous single point (SPEC §10, §12): if the
//! GGUF header's pooling type is not honoured and the fallback path is also
//! wrong, retrieval quality degrades toward random **and nothing reports an
//! error**. So it gets measured, not reasoned about.
//!
//! Four checks:
//!   A. self-similarity      same text twice -> cos == 1.0
//!   B. header honoured      pooling=unspecified vs pooling=last, same text
//!                           -> cos == 1.0 proves the header resolves to LAST
//!   C. triplet ordering     cos(anchor, near) > cos(anchor, far), incl. a
//!                           cross-lingual pair
//!   D. reference dump       vectors for 3 fixed texts, for the HF
//!                           transformers cross-check (E3 item 3)
//!
//! Run: zig build e3 -- <path-to-gguf>

const std = @import("std");
const zkb = @import("zkb");
const emb = zkb.embed;

const Triplet = struct {
    name: []const u8,
    anchor: []const u8,
    near: []const u8,
    far: []const u8,
};

const triplets = [_]Triplet{
    .{
        .name = "rrf",
        .anchor = "检索融合用 RRF，不要加权和",
        .near = "用 reciprocal rank fusion 做多路融合",
        .far = "Metal shader 静态链接后在异位 cwd 找不到",
    },
    .{
        .name = "wal",
        .anchor = "SQLite WAL 模式下读不阻塞写",
        .near = "WAL 允许多读者并发与单一写者",
        .far = "记账流水按月分文件存放",
    },
    .{
        // Cross-lingual: an English anchor must retrieve its Chinese paraphrase.
        .name = "cross-lingual",
        .anchor = "embedding model version binding",
        .near = "向量必须与生成它的 embedding 模型版本绑定",
        .far = "launchd 每日跑一次维护任务",
    },
    .{
        .name = "trigram",
        .anchor = "trigram tokenizer 无法匹配少于 3 个字符的词",
        .near = "少于三个字的查询词在 trigram 下匹配不到任何结果",
        .far = "Qwen3-Embedding 输出 1024 维向量",
    },
    .{
        .name = "derived",
        .anchor = "年龄不该存，应该存出生日期",
        .near = "派生量一律不存，只存能推导出它的原始事实",
        .far = "Unix socket 权限设为 0600",
    },
};

const reference_texts = [_][]const u8{
    "第一性原理：先用最小实验拿到 ground truth，再推理根因。",
    "Retrieval fusion uses reciprocal rank fusion with k = 60.",
    "sqlite-vec 的 vec0 虚拟表支持 partition key 与 metadata 列过滤。",
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var args = init.minimal.args.iterate();
    _ = args.next();
    var model_path_opt: ?[]const u8 = null;
    // --verbose lets llama.cpp's own load log through. E4 needs it to confirm
    // the Metal backend is genuinely active rather than a silent CPU fallback.
    var verbose = false;
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--verbose")) verbose = true else model_path_opt = a;
    }
    const model_path = model_path_opt orelse {
        std.debug.print("usage: e3_pooling [--verbose] <path-to-gguf>\n", .{});
        return error.MissingArgument;
    };

    var out_buf: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &out_buf);
    const w = &stdout.interface;

    try w.print("E3 — Qwen3-Embedding pooling correctness\n", .{});
    try w.print("model: {s}\n\n", .{model_path});
    try w.flush();

    var failures: usize = 0;

    // ---- production configuration: let the GGUF header choose ------------
    var prod = try emb.Embedder.init(gpa, model_path, .{ .quiet = !verbose });
    defer prod.deinit();

    try w.print("n_embd = {d}\n", .{prod.n_embd});
    if (prod.n_embd != 1024) {
        try w.print("  FAIL: expected 1024 dims\n", .{});
        failures += 1;
    }

    const dim = prod.n_embd;
    const va = try gpa.alloc(f32, dim);
    defer gpa.free(va);
    const vb = try gpa.alloc(f32, dim);
    defer gpa.free(vb);
    const vc = try gpa.alloc(f32, dim);
    defer gpa.free(vc);

    // ---- A. self-similarity ----------------------------------------------
    try w.print("\n[A] self-similarity (same text twice)\n", .{});
    {
        const text = reference_texts[0];
        _ = try prod.embed(text, va);
        _ = try prod.embed(text, vb);
        const cos = emb.cosine(va, vb);
        const ok = cos > 0.99999;
        try w.print("  cos = {d:.6}  {s}\n", .{ cos, if (ok) "PASS" else "FAIL" });
        if (!ok) failures += 1;
    }

    // ---- B. is the header's pooling type actually honoured? ---------------
    // A direct test of the risky assumption: if UNSPECIFIED resolves to LAST
    // (what Qwen3-Embedding declares), the two vectors must be identical.
    try w.print("\n[B] pooling=unspecified vs pooling=last\n", .{});
    {
        const text = reference_texts[0];
        _ = try prod.embed(text, va);

        var forced = try emb.Embedder.init(gpa, model_path, .{ .pooling = .last });
        defer forced.deinit();
        _ = try forced.embed(text, vb);
        const cos_last = emb.cosine(va, vb);

        var mean = try emb.Embedder.init(gpa, model_path, .{ .pooling = .mean });
        defer mean.deinit();
        _ = try mean.embed(text, vc);
        const cos_mean = emb.cosine(va, vc);

        try w.print("  cos(unspecified, last) = {d:.6}\n", .{cos_last});
        try w.print("  cos(unspecified, mean) = {d:.6}\n", .{cos_mean});
        if (cos_last > 0.99999) {
            try w.print("  PASS: header resolves to LAST pooling\n", .{});
        } else if (cos_mean > 0.99999) {
            try w.print("  FAIL: header resolves to MEAN — wrong for Qwen3-Embedding.\n", .{});
            try w.print("        Set pooling = .last explicitly in production.\n", .{});
            failures += 1;
        } else {
            try w.print("  FAIL: unspecified matches neither last nor mean.\n", .{});
            failures += 1;
        }
    }

    // ---- C. triplet ordering ---------------------------------------------
    try w.print("\n[C] triplet ordering: cos(anchor,near) > cos(anchor,far)\n", .{});
    for (triplets) |t| {
        _ = try prod.embed(t.anchor, va);
        _ = try prod.embed(t.near, vb);
        _ = try prod.embed(t.far, vc);
        const near = emb.cosine(va, vb);
        const far = emb.cosine(va, vc);
        const ok = near > far;
        if (!ok) failures += 1;
        try w.print(
            "  {s:<14} near={d:.4} far={d:.4} margin={d:.4}  {s}\n",
            .{ t.name, near, far, near - far, if (ok) "PASS" else "FAIL" },
        );
    }

    // ---- C2. asymmetric query/document contract ---------------------------
    // The instruct prefix must not break the ordering; if it does, §3.3 is wrong.
    try w.print("\n[C2] query-side instruct prefix preserves ordering\n", .{});
    for (triplets) |t| {
        _ = try prod.embedQuery(emb.default_query_task, t.anchor, va);
        _ = try prod.embedDocument("", t.near, vb);
        _ = try prod.embedDocument("", t.far, vc);
        const near = emb.cosine(va, vb);
        const far = emb.cosine(va, vc);
        const ok = near > far;
        if (!ok) failures += 1;
        try w.print(
            "  {s:<14} near={d:.4} far={d:.4} margin={d:.4}  {s}\n",
            .{ t.name, near, far, near - far, if (ok) "PASS" else "FAIL" },
        );
    }

    // ---- D. reference vectors for the HF cross-check ---------------------
    try w.print("\n[D] reference vectors -> /tmp/zkb-e3-vectors.json\n", .{});
    {
        var file = try std.Io.Dir.createFileAbsolute(init.io, "/tmp/zkb-e3-vectors.json", .{});
        defer file.close(init.io);
        var fbuf: [1 << 16]u8 = undefined;
        var fw = file.writer(init.io, &fbuf);
        const jw = &fw.interface;

        try jw.writeAll("{\n  \"model\": \"Qwen3-Embedding-0.6B-Q8_0\",\n  \"vectors\": [\n");
        for (reference_texts, 0..) |text, i| {
            _ = try prod.embed(text, va);
            try jw.print("    {{\"text\": ", .{});
            try std.json.Stringify.value(text, .{}, jw);
            try jw.writeAll(", \"vec\": [");
            for (va, 0..) |x, j| {
                if (j != 0) try jw.writeAll(",");
                try jw.print("{d:.7}", .{x});
            }
            try jw.writeAll("]}");
            if (i + 1 != reference_texts.len) try jw.writeAll(",");
            try jw.writeAll("\n");
        }
        try jw.writeAll("  ]\n}\n");
        try jw.flush();
        try w.print("  written ({d} texts x {d} dims)\n", .{ reference_texts.len, dim });
    }

    try w.print("\n{s}: {d} failure(s)\n", .{ if (failures == 0) "E3 PASS" else "E3 FAIL", failures });
    try w.flush();
    if (failures != 0) std.process.exit(1);
}
