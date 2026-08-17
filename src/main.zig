const std = @import("std");
const zkb = @import("zkb");
const doctor = @import("cli/doctor.zig");
const model = @import("cli/model.zig");

const usage =
    \\zkb — Agent memory + personal knowledge base
    \\
    \\usage: zkb <command> [args]
    \\
    \\commands:
    \\  doctor [--model PATH]        verify this machine can run zkb
    \\  model pull [--quant Q]       download the embedding model (q8_0 | f16)
    \\  version                      print version info
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;

    var out_buf: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &out_buf);
    const w = &stdout.interface;
    defer w.flush() catch {};

    var args = init.minimal.args.iterate();
    _ = args.next();
    const cmd = args.next() orelse {
        try w.writeAll(usage);
        return 2;
    };

    if (std.mem.eql(u8, cmd, "doctor")) {
        var model_path: ?[]const u8 = null;
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--model")) model_path = args.next();
        }
        const r = try doctor.run(gpa, init.io, init.environ_map, w, model_path);
        return if (r.failures == 0) 0 else 1;
    }

    if (std.mem.eql(u8, cmd, "model")) {
        const sub = args.next() orelse {
            try w.writeAll("usage: zkb model pull [--quant q8_0|f16]\n");
            return 2;
        };
        if (!std.mem.eql(u8, sub, "pull")) {
            try w.print("unknown model subcommand: {s}\n", .{sub});
            return 2;
        }
        var quant: model.Quant = .q8_0;
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--quant")) {
                const v = args.next() orelse break;
                quant = std.meta.stringToEnum(model.Quant, v) orelse {
                    try w.print("unknown quant: {s} (expected q8_0 or f16)\n", .{v});
                    return 2;
                };
            }
        }
        model.pull(gpa, init.io, init.environ_map, w, quant) catch |err| {
            try w.print("model pull failed: {t}\n", .{err});
            return 1;
        };
        return 0;
    }

    if (std.mem.eql(u8, cmd, "version")) {
        try w.print("zkb {s}\n", .{zkb.version});
        try w.print("sqlite {s} / sqlite-vec {s}\n", .{
            zkb.sqlite.libVersion(),
            zkb.sqlite.vecVersion(),
        });
        try w.print("llama backend: {s}\n", .{if (zkb.build_options.llama) "on" else "off"});
        return 0;
    }

    try w.print("unknown command: {s}\n\n", .{cmd});
    try w.writeAll(usage);
    return 2;
}
