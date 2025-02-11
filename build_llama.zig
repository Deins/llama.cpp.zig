const std = @import("std");
const Builder = std.Build;
const Target = std.Build.ResolvedTarget;
const Mode = std.builtin.Mode;
const CompileStep = std.Build.Step.Compile;
const LazyPath = std.Build.LazyPath;
const Module = std.Build.Module;

pub const Backends = struct {
    cpu: bool = true,
    metal: bool = false,
    // untested possibly unsupported at the moment:
    cuda: bool = false,
    sycl: bool = false,
    vulkan: bool = false,
    opencl: bool = false,
    cann: bool = false,
    blas: bool = false,
    rpc: bool = false,
    kompute: bool = false,

    pub fn addDefines(self: @This(), comp: *CompileStep) void {
        if (self.cuda) comp.defineCMacro("GGML_USE_CUDA", null);
        if (self.metal) comp.defineCMacro("GGML_USE_METAL", null);
        if (self.sycl) comp.defineCMacro("GGML_USE_SYCL", null);
        if (self.vulkan) comp.defineCMacro("GGML_USE_VULKAN", null);
        if (self.opencl) comp.defineCMacro("GGML_USE_OPENCL", null);
        if (self.cann) comp.defineCMacro("GGML_USE_CANN", null);
        if (self.blas) comp.defineCMacro("GGML_USE_BLAS", null);
        if (self.rpc) comp.defineCMacro("GGML_USE_RPC", null);
        if (self.kompute) comp.defineCMacro("GGML_USE_KOMPUTE", null);
        if (self.cpu) comp.defineCMacro("GGML_USE_CPU", null);
    }
};

pub const Options = struct {
    target: Target,
    optimize: Mode,
    backends: Backends = .{},
    shared: bool, // static or shared lib
    build_number: usize = 0, // number that will be writen in build info
    metal_ndebug: bool = false,
    metal_use_bf16: bool = false,
};

// Build context
pub const Context = struct {
    b: *Builder,
    options: Options,
    build_info: *CompileStep,
    path_prefix: []const u8 = "",
    lib: ?*CompileStep = null,

    pub fn init(b: *Builder, op: Options) Context {
        const path_prefix = b.pathJoin(&.{ thisPath(), "/llama.cpp" });
        const zig_version = @import("builtin").zig_version_string;
        const commit_hash = std.process.Child.run(
            .{ .allocator = b.allocator, .argv = &.{ "git", "rev-parse", "HEAD" } },
        ) catch |err| {
            std.log.err("Cant get git comiit hash! err: {}", .{err});
            unreachable;
        };

        const build_info_zig = true; // use cpp or zig file for build-info
        const build_info_path = b.pathJoin(&.{ "common", "build-info." ++ if (build_info_zig) "zig" else "cpp" });
        const build_info = b.fmt(if (build_info_zig)
            \\pub export var LLAMA_BUILD_NUMBER : c_int = {};
            \\pub export var LLAMA_COMMIT = "{s}";
            \\pub export var LLAMA_COMPILER = "Zig {s}";
            \\pub export var LLAMA_BUILD_TARGET = "{s}_{s}";
            \\
        else
            \\int LLAMA_BUILD_NUMBER = {};
            \\char const *LLAMA_COMMIT = "{s}";
            \\char const *LLAMA_COMPILER = "Zig {s}";
            \\char const *LLAMA_BUILD_TARGET = "{s}_{s}";
            \\
        , .{ op.build_number, commit_hash.stdout[0 .. commit_hash.stdout.len - 1], zig_version, op.target.result.zigTriple(b.allocator) catch unreachable, @tagName(op.optimize) });

        return .{
            .b = b,
            .options = op,
            .path_prefix = path_prefix,
            .build_info = b.addObject(.{ .name = "llama-build-info", .target = op.target, .optimize = op.optimize, .root_source_file = b.addWriteFiles().add(build_info_path, build_info) }),
        };
    }

    /// just builds everything needed and links it to your target
    pub fn link(ctx: *Context, comp: *CompileStep) void {
        const lib = ctx.library();
        comp.linkLibrary(lib);
        if (ctx.options.shared) ctx.b.installArtifact(lib);
    }

    /// build single library containing everything
    pub fn library(ctx: *Context) *CompileStep {
        if (ctx.lib) |l| return l;
        const lib_opt = .{ .name = "llama.cpp", .target = ctx.options.target, .optimize = ctx.options.optimize };
        const lib = if (ctx.options.shared) blk: {
            const lib = ctx.b.addSharedLibrary(lib_opt);
            lib.defineCMacro("LLAMA_SHARED", null);
            lib.defineCMacro("LLAMA_BUILD", null);
            if (ctx.options.target.result.os.tag == .windows) {
                std.log.warn("For shared linking to work, requires header llama.h modification:\n\'#    if defined(_WIN32) && (!defined(__MINGW32__) || defined(ZIG))'", .{});
                lib.defineCMacro("ZIG", null);
            }
            break :blk lib;
        } else ctx.b.addStaticLibrary(lib_opt);
        ctx.options.backends.addDefines(lib);
        ctx.addAll(lib);
        if (ctx.options.target.result.abi != .msvc)
            lib.defineCMacro("_GNU_SOURCE", null);
        ctx.lib = lib;
        return lib;
    }

    /// link everything directly to target
    pub fn addAll(ctx: *Context, compile: *CompileStep) void {
        ctx.addBuildInfo(compile);
        ctx.addGgml(compile);
        ctx.addLLama(compile);
    }

    /// zig module with translated headers
    pub fn moduleLlama(ctx: *Context) *Module {
        const tc = ctx.b.addTranslateC(.{
            .root_source_file = ctx.includePath("llama.h"),
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });
        if (ctx.options.shared) tcDefineCMacro(tc, "LLAMA_SHARED", null);
        tc.addIncludeDir(ctx.path(&.{ "ggml", "include" }).getPath(ctx.b));
        tcDefineCMacro(tc, "NDEBUG", null); // otherwise zig is unhappy about c ASSERT macro
        return tc.addModule("llama.h");
    }

    /// zig module with translated headers
    pub fn moduleGgml(ctx: *Context) *Module {
        const tc = ctx.b.addTranslateC(.{
            .root_source_file = ctx.path(&.{ "ggml", "include", "ggml.h" }),
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });

        tcDefineCMacro(tc, "LLAMA_SHARED", null);
        tcDefineCMacro(tc, "NDEBUG", null);

        return tc.addModule("ggml.h");
    }

    pub fn addBuildInfo(ctx: *Context, compile: *CompileStep) void {
        compile.addObject(ctx.build_info);
    }

    pub fn addGgml(ctx: *Context, compile: *CompileStep) void {
        ctx.common(compile);
        compile.addIncludePath(ctx.path(&.{ "ggml", "include" }));
        compile.addIncludePath(ctx.path(&.{ "ggml", "src" }));

        if (ctx.options.target.result.os.tag == .windows) {
            compile.defineCMacro("GGML_ATTRIBUTE_FORMAT(...)", "");
        }

        var sources = std.ArrayList(LazyPath).init(ctx.b.allocator);
        sources.appendSlice(&.{
            ctx.path(&.{ "ggml", "src", "ggml-alloc.c" }),
            ctx.path(&.{ "ggml", "src", "ggml-backend-reg.cpp" }),
            ctx.path(&.{ "ggml", "src", "ggml-backend.cpp" }),
            ctx.path(&.{ "ggml", "src", "ggml-opt.cpp" }),
            ctx.path(&.{ "ggml", "src", "ggml-quants.c" }),
            ctx.path(&.{ "ggml", "src", "ggml-threading.cpp" }),
            ctx.path(&.{ "ggml", "src", "ggml.c" }),
            ctx.path(&.{ "ggml", "src", "gguf.cpp" }),
        }) catch unreachable;

        if (ctx.options.backends.cpu) {
            compile.addIncludePath(ctx.path(&.{ "ggml", "src", "ggml-cpu" }));
            compile.linkLibCpp();
            sources.appendSlice(&.{
                ctx.path(&.{ "ggml", "src", "ggml-cpu", "ggml-cpu.c" }),
                ctx.path(&.{ "ggml", "src", "ggml-cpu", "ggml-cpu.cpp" }),
                ctx.path(&.{ "ggml", "src", "ggml-cpu", "ggml-cpu-aarch64.cpp" }),
                ctx.path(&.{ "ggml", "src", "ggml-cpu", "ggml-cpu-hbm.cpp" }),
                ctx.path(&.{ "ggml", "src", "ggml-cpu", "ggml-cpu-quants.c" }),
                ctx.path(&.{ "ggml", "src", "ggml-cpu", "ggml-cpu-traits.cpp" }),
                ctx.path(&.{ "ggml", "src", "ggml-cpu", "amx/amx.cpp" }),
                ctx.path(&.{ "ggml", "src", "ggml-cpu", "amx/mmq.cpp" }),
            }) catch unreachable;
        }

        if (ctx.options.backends.metal) {
            compile.addIncludePath(ctx.path(&.{ "ggml", "src", "ggml-metal" }));
            compile.linkLibCpp();
            compile.defineCMacro("GGML_METAL", null);
            if (ctx.options.metal_ndebug) {
                compile.defineCMacro("GGML_METAL_NDEBUG", null);
            }
            if (ctx.options.metal_use_bf16) {
                compile.defineCMacro("GGML_METAL_USE_BF16", null);
            }
            // Create a separate Metal library
            const metal_lib = ctx.b.addStaticLibrary(.{
                .name = "ggml-metal",
                .target = ctx.options.target,
                .optimize = ctx.options.optimize,
            });
            metal_lib.addIncludePath(ctx.path(&.{ "ggml", "include" }));
            metal_lib.addIncludePath(ctx.path(&.{ "ggml", "src" }));
            metal_lib.addIncludePath(ctx.path(&.{ "ggml", "src", "ggml-metal" }));
            metal_lib.linkFramework("Foundation");
            metal_lib.linkFramework("AppKit");
            metal_lib.linkFramework("Metal");
            metal_lib.linkFramework("MetalKit");
            metal_lib.addCSourceFile(.{ .file = ctx.path(&.{ "ggml", "src", "ggml-metal", "ggml-metal.m" }), .flags = ctx.flags() });
            const metal_files = [_][]const u8{
                "ggml-metal.metal",
                "ggml-metal-impl.h",
            };
            // Compile the metal shader [requires xcode installed]
            const metal_compile = ctx.b.addSystemCommand(&.{
                "xcrun", "-sdk", "macosx", "metal", "-fno-fast-math", "-g", "-c", ctx.b.pathJoin(&.{ ctx.b.install_path, "metal", "ggml-metal.metal" }), "-o", ctx.b.pathJoin(&.{ ctx.b.install_path, "metal", "ggml-metal.air" }),
            });
            const common_src = ctx.path(&.{ "ggml", "src", "ggml-common.h" });
            const common_dst = "ggml-common.h";
            const common_install_step = ctx.b.addInstallFile(common_src, common_dst);
            metal_compile.step.dependOn(&common_install_step.step);
            for (metal_files) |file| {
                const src = ctx.path(&.{ "ggml", "src", "ggml-metal", file });
                const dst = ctx.b.pathJoin(&.{ "metal", file });
                const install_step = ctx.b.addInstallFile(src, dst);
                metal_compile.step.dependOn(&install_step.step);
            }
            const metallib_compile = ctx.b.addSystemCommand(&.{
                "xcrun", "-sdk", "macosx", "metallib", ctx.b.pathJoin(&.{ ctx.b.install_path, "metal", "ggml-metal.air" }), "-o", ctx.b.pathJoin(&.{ ctx.b.install_path, "metal", "default.metallib" }),
            });
            metallib_compile.step.dependOn(&metal_compile.step);
            // Install the metal shader source file to bin directory
            const metal_shader_install = ctx.b.addInstallBinFile(ctx.path(&.{ "ggml", "src", "ggml-metal", "ggml-metal.metal" }), "ggml-metal.metal");
            const default_lib_install = ctx.b.addInstallBinFile(.{ .cwd_relative = ctx.b.pathJoin(&.{ ctx.b.install_path, "metal", "default.metallib" }) }, "default.metallib");
            metal_shader_install.step.dependOn(&metallib_compile.step);
            default_lib_install.step.dependOn(&metal_shader_install.step);
            // Link the metal library with the main compilation
            compile.linkLibrary(metal_lib);
            compile.step.dependOn(&metal_lib.step);
            compile.step.dependOn(&default_lib_install.step);
        }

        for (sources.items) |src| compile.addCSourceFile(.{ .file = src, .flags = ctx.flags() });
    }

    pub fn addLLama(ctx: *Context, compile: *CompileStep) void {
        ctx.common(compile);
        compile.addIncludePath(ctx.path(&.{"include"}));
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-adapter.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-arch.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-batch.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-chat.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-context.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-grammar.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-hparams.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-impl.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-kv-cache.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-mmap.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-model-loader.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-model.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-sampling.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-vocab.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-vocab.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("unicode-data.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("unicode.cpp"), .flags = ctx.flags() });

        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "common.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "sampling.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "console.cpp" }), .flags = ctx.flags() });

        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "json-schema-to-grammar.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "speculative.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "ngram-cache.cpp" }), .flags = ctx.flags() });

        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "log.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "arg.cpp" }), .flags = ctx.flags() });
    }

    pub fn samples(ctx: *Context, install: bool) !void {
        const b = ctx.b;
        const examples = [_][]const u8{
            "main",
            "simple",
            // "perplexity",
            // "embedding",
            // "finetune",
            // "train-text-from-scratch",
            // "lookahead",
            // "speculative",
            // "parallel",
        };

        for (examples) |ex| {
            const exe = b.addExecutable(.{ .name = ex, .target = ctx.options.target, .optimize = ctx.options.optimize });
            exe.addIncludePath(ctx.path(&.{"include"}));
            exe.addIncludePath(ctx.path(&.{"common"}));
            exe.addIncludePath(ctx.path(&.{ "ggml", "include" }));
            exe.addIncludePath(ctx.path(&.{ "ggml", "src" }));

            exe.want_lto = false; // TODO: review, causes: error: lld-link: undefined symbol: __declspec(dllimport) _create_locale
            if (install) b.installArtifact(exe);
            { // add all c/cpp files from example dir
                const rpath = b.pathJoin(&.{ ctx.path_prefix, "examples", ex });
                exe.addIncludePath(.{ .cwd_relative = rpath });
                var dir = if (@hasDecl(std.fs, "openIterableDirAbsolute")) try std.fs.openIterableDirAbsolute(b.pathFromRoot(rpath), .{}) else try std.fs.openDirAbsolute(b.pathFromRoot(rpath), .{ .iterate = true }); // zig 11 vs nightly compatibility
                defer dir.close();
                var dir_it = dir.iterate();
                while (try dir_it.next()) |f| switch (f.kind) {
                    .file => if (std.ascii.endsWithIgnoreCase(f.name, ".c") or std.ascii.endsWithIgnoreCase(f.name, ".cpp")) {
                        const src = b.pathJoin(&.{ ctx.path_prefix, "examples", ex, f.name });
                        exe.addCSourceFile(.{ .file = .{ .cwd_relative = src }, .flags = ctx.flags() });
                    },
                    else => {},
                };
            }
            ctx.common(exe);
            ctx.link(exe);

            const run_exe = b.addRunArtifact(exe);
            if (b.args) |args| run_exe.addArgs(args); // passes on args like: zig build run -- my fancy args
            run_exe.step.dependOn(b.default_step); // allways copy output, to avoid confusion
            b.step(b.fmt("run-cpp-{s}", .{ex}), b.fmt("Run llama.cpp example: {s}", .{ex})).dependOn(&run_exe.step);
        }
    }

    fn flags(ctx: Context) []const []const u8 {
        _ = ctx;
        return &.{"-fno-sanitize=undefined"};
    }

    fn common(ctx: Context, lib: *CompileStep) void {
        lib.linkLibCpp();
        lib.addIncludePath(ctx.path(&.{"common"}));
        if (ctx.options.optimize != .Debug) lib.defineCMacro("NDEBUG", null);
    }

    pub fn path(self: Context, subpath: []const []const u8) LazyPath {
        const sp = self.b.pathJoin(subpath);
        return .{ .cwd_relative = self.b.pathJoin(&.{ self.path_prefix, sp }) };
    }

    pub fn srcPath(self: Context, p: []const u8) LazyPath {
        return .{ .cwd_relative = self.b.pathJoin(&.{ self.path_prefix, "src", p }) };
    }

    pub fn includePath(self: Context, p: []const u8) LazyPath {
        return .{ .cwd_relative = self.b.pathJoin(&.{ self.path_prefix, "include", p }) };
    }
};

fn thisPath() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}

// TODO: idk, defineCMacro returns: TranslateC.zig:110:28: error: root struct of file 'Build' has no member named 'constructranslate_cMacro'
// use raw macro for now
fn tcDefineCMacro(tc: *std.Build.Step.TranslateC, comptime name: []const u8, comptime value: ?[]const u8) void {
    tc.defineCMacroRaw(name ++ "=" ++ (value orelse "1"));
}
