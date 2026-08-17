const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const t = target.result;
    const is_apple = t.os.tag == .macos or t.os.tag == .ios;

    // llama.cpp is only needed for embeddings. Turn it off to work on the
    // storage layer (or to run the SQLite-only tests) without waiting on a
    // cmake build.
    const want_llama = b.option(bool, "llama", "Link llama.cpp embedding backend") orelse true;

    const opts = b.addOptions();
    opts.addOption(bool, "llama", want_llama);

    // ---------------------------------------------------------------- SQLite
    // Two of these flags are load-bearing:
    //   FTS5 on          — the keyword retrieval path depends on it
    //   THREADSAFE=2     — daemon holds one connection per thread
    // Everything else trims features we never reach for.
    const sqlite_flags: []const []const u8 = &.{
        "-std=c11",
        "-DSQLITE_ENABLE_FTS5=1",
        "-DSQLITE_THREADSAFE=2",
        "-DSQLITE_DQS=0",
        "-DSQLITE_OMIT_LOAD_EXTENSION=1",
        "-DSQLITE_OMIT_DEPRECATED",
        "-DSQLITE_OMIT_SHARED_CACHE",
        "-DSQLITE_OMIT_PROGRESS_CALLBACK",
        "-DSQLITE_DEFAULT_MEMSTATUS=0",
        "-DSQLITE_DEFAULT_WAL_SYNCHRONOUS=1",
        "-DSQLITE_LIKE_DOESNT_MATCH_BLOBS",
        "-DSQLITE_USE_ALLOCA",
        "-DSQLITE_ENABLE_MATH_FUNCTIONS",
        "-DSQLITE_MAX_EXPR_DEPTH=0",
        "-Wno-everything",
        // See the note on sqlite-vec.c below: this applies to sqlite3.c for
        // the same reason, and for many more call sites.
        "-fno-sanitize=function",
    };

    const sqlite_lib = b.addLibrary(.{
        .name = "zkb_sqlite",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    sqlite_lib.root_module.addCSourceFile(.{
        .file = b.path("third_party/sqlite/sqlite3.c"),
        .flags = sqlite_flags,
    });
    // SQLITE_CORE makes sqlite-vec link against sqlite3 directly instead of
    // going through the loadable-extension api routines.
    //
    // `-fno-sanitize=function` is load-bearing in ReleaseSafe. Zig turns on the
    // C undefined-behaviour sanitizer in trap mode there, and its `function`
    // check requires a call through a function pointer to match the callee's
    // prototype exactly. C libraries do not work that way: sqlite-vec stores
    // `fvec_cleanup_noop`, a `void(f32*)`, in a `void(void*)` slot and calls it
    // through the latter. That is ABI-identical on every target zkb builds for,
    // but it is a prototype mismatch, so the check fires — silently, as `brk`,
    // with no panic message.
    //
    // It is not one site. A ReleaseSafe build carries 412 of these traps, all
    // in vendored C, any of which could fire at runtime. Disabling one check on
    // code we do not own is the honest scope; every other UB check stays on
    // here, and our own C below keeps the full set.
    sqlite_lib.root_module.addCSourceFile(.{
        .file = b.path("third_party/sqlite-vec/sqlite-vec.c"),
        .flags = &.{
            "-std=c11",
            "-DSQLITE_CORE=1",
            "-DSQLITE_VEC_STATIC=1",
            "-Wno-everything",
            "-fno-sanitize=function",
        },
    });
    sqlite_lib.root_module.addCSourceFile(.{
        .file = b.path("src/db/sqlite_helpers.c"),
        .flags = &.{ "-std=c11", "-Wno-everything" },
    });
    // Custom FTS5 tokenizer. Lives with the sqlite lib because it needs the
    // fts5 api declarations, not because it is part of sqlite.
    sqlite_lib.root_module.addCSourceFile(.{
        .file = b.path("src/db/fts5_cjk.c"),
        .flags = &.{ "-std=c11", "-Wno-everything" },
    });
    sqlite_lib.root_module.addIncludePath(b.path("third_party/sqlite"));
    sqlite_lib.root_module.addIncludePath(b.path("third_party/sqlite-vec"));

    // ------------------------------------------------------------ zkb module
    const zkb_mod = b.addModule("zkb", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    zkb_mod.addOptions("build_options", opts);
    zkb_mod.addIncludePath(b.path("third_party/sqlite"));
    zkb_mod.addIncludePath(b.path("third_party/sqlite-vec"));
    // These must match how sqlite-vec.c is compiled above. Without them
    // sqlite-vec.h takes its `#include "sqlite3ext.h"` branch, picks up the
    // macOS SDK's sqlite3.h, and collides with the vendored one.
    zkb_mod.addCMacro("SQLITE_CORE", "1");
    zkb_mod.addCMacro("SQLITE_VEC_STATIC", "1");
    zkb_mod.linkLibrary(sqlite_lib);
    if (want_llama) addLlamaBindings(b, zkb_mod);

    // ----------------------------------------------------------- executable
    const exe = b.addExecutable(.{
        .name = "zkb",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zkb", .module = zkb_mod }},
        }),
    });
    if (want_llama) addLlamaLinks(b, exe.root_module, is_apple);
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    const run_step = b.step("run", "Run zkb");
    run_step.dependOn(&run.step);

    // ---------------------------------------------------------------- tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zkb", .module = zkb_mod }},
        }),
    });
    if (want_llama) addLlamaLinks(b, tests.root_module, is_apple);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_tests.step);

    // The suite above is rooted at tests/root.zig, which can only reach what the
    // `zkb` module exports. Everything on the cli side — main.zig, cli/*, mcp/* —
    // imports `zkb` rather than being part of it, so no test there was ever
    // compiled, let alone run. A `test` block in src/mcp/server.zig passed by
    // never executing, which is how a malformed tools/list payload shipped in
    // 0.0.1.
    //
    // Rooting a second artifact at main.zig covers the cli side, since every
    // file there is reachable from it.
    const cli_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zkb", .module = zkb_mod }},
        }),
    });
    if (want_llama) addLlamaLinks(b, cli_tests.root_module, is_apple);
    test_step.dependOn(&b.addRunArtifact(cli_tests).step);

    // ---------------------------------------------------------- experiments
    // M0's blocking geology checks (SPEC §10). Separate binaries rather than
    // unit tests: they need a 639MB model on disk, which no test should.
    if (want_llama) {
        const e3 = b.addExecutable(.{
            .name = "e3_pooling",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/experiments/e3_pooling.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .imports = &.{.{ .name = "zkb", .module = zkb_mod }},
            }),
        });
        addLlamaLinks(b, e3.root_module, is_apple);
        const run_e3 = b.addRunArtifact(e3);
        if (b.args) |args| run_e3.addArgs(args);
        const e3_step = b.step("e3", "E3: Qwen3-Embedding pooling correctness (blocking)");
        e3_step.dependOn(&run_e3.step);

        const e5 = b.addExecutable(.{
            .name = "e5_ctx",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/experiments/e5_ctx.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .imports = &.{.{ .name = "zkb", .module = zkb_mod }},
            }),
        });
        addLlamaLinks(b, e5.root_module, is_apple);
        const run_e5 = b.addRunArtifact(e5);
        if (b.args) |args| run_e5.addArgs(args);
        const e5_step = b.step("e5-ctx", "E5b: embedding throughput vs n_ctx");
        e5_step.dependOn(&run_e5.step);

    }

    // E7 needs no model either: it reads vectors that are already indexed.
    {
        const e7 = b.addExecutable(.{
            .name = "e7_thresholds",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/experiments/e7_thresholds.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .imports = &.{.{ .name = "zkb", .module = zkb_mod }},
            }),
        });
        if (want_llama) addLlamaLinks(b, e7.root_module, is_apple);
        const run_e7 = b.addRunArtifact(e7);
        if (b.args) |args| run_e7.addArgs(args);
        const e7_step = b.step("e7", "E7: pool near-duplicate and island candidates for judgement");
        e7_step.dependOn(&run_e7.step);
    }

    // ------------------------------------------------- llama.cpp cmake step
    // Artifacts land under build/llama.cpp/dev/ and are linked by path.
    // Only libllama + libggml* are needed: no mtmd, no vision, no audio.
    {
        const src = b.dependency("llama_cpp", .{}).path("").getPath(b);
        const cfg = b.addSystemCommand(&.{ "cmake", "-B", "build/llama.cpp/dev", "-S", src });
        cfg.addArgs(&.{
            "-G",                          "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",  "-DBUILD_SHARED_LIBS=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",     "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",    "-DLLAMA_BUILD_TOOLS=OFF",
            "-DLLAMA_CURL=OFF",            "-DGGML_OPENMP=OFF",
            "-DGGML_LLAMAFILE=OFF",
        });
        if (@import("builtin").os.tag == .macos) {
            cfg.addArgs(&.{ "-DGGML_METAL=ON", "-DGGML_BLAS=ON" });
        }
        const bld = b.addSystemCommand(&.{ "cmake", "--build", "build/llama.cpp/dev", "-j" });
        bld.step.dependOn(&cfg.step);
        const s = b.step("llama-cpp", "Build vendored llama.cpp (static, embeddings only)");
        s.dependOn(&bld.step);
    }
}

fn addLlamaBindings(b: *std.Build, mod: *std.Build.Module) void {
    const dep = b.dependency("llama_cpp", .{});
    mod.addIncludePath(dep.path("include"));
    mod.addIncludePath(dep.path("ggml/include"));
}

fn addLlamaLinks(b: *std.Build, mod: *std.Build.Module, is_apple: bool) void {
    const libs = [_][]const u8{
        "build/llama.cpp/dev/src/libllama.a",
        "build/llama.cpp/dev/ggml/src/libggml.a",
        "build/llama.cpp/dev/ggml/src/libggml-base.a",
        "build/llama.cpp/dev/ggml/src/libggml-cpu.a",
    };
    for (libs) |lib| mod.addObjectFile(b.path(lib));
    if (is_apple) {
        mod.addObjectFile(b.path("build/llama.cpp/dev/ggml/src/ggml-blas/libggml-blas.a"));
        mod.addObjectFile(b.path("build/llama.cpp/dev/ggml/src/ggml-metal/libggml-metal.a"));
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Metal", .{});
        mod.linkFramework("MetalKit", .{});
        mod.linkFramework("Accelerate", .{});
        // FSEvents + CoreFoundation, for the daemon's filesystem watcher
        // (src/util/fsevents.zig).
        mod.linkFramework("CoreServices", .{});
    }
    mod.link_libcpp = true;
}
