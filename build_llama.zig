const std = @import("std");
const Builder = std.Build;
const CrossTarget = std.zig.CrossTarget;
const Mode = std.builtin.Mode;
const CompileStep = std.Build.CompileStep;
const LazyPath = std.Build.LazyPath;
const Module = std.Build.Module;
pub const clblast = @import("clblast");

pub const Options = struct {
    target: CrossTarget,
    optimize: Mode,
    shared: bool, // static or shared lib
    lto: bool = false,
    opencl: ?clblast.OpenCL = null,
    clblast: bool = false,
    build_number: usize = 0, // number that will be writen in build info
};

// Build context
pub const Context = struct {
    b: *Builder,
    options: Options,
    build_info: *CompileStep,
    path_prefix: []const u8 = "llama.cpp",

    pub fn init(b: *Builder, op: Options, path_prefix: []const u8) Context {
        const zig_version = @import("builtin").zig_version_string;
        const commit_hash = std.ChildProcess.exec(
            .{ .allocator = b.allocator, .argv = &.{ "git", "rev-parse", "HEAD" } },
        ) catch |err| {
            std.log.err("Cant get git comiit hash! err: {}", .{err});
            unreachable;
        };

        const build_info_path = b.pathJoin(&.{ path_prefix, "common", "build-info.cpp" });
        const build_info = b.fmt(
            \\int LLAMA_BUILD_NUMBER = {};
            \\char const *LLAMA_COMMIT = "{s}";
            \\char const *LLAMA_COMPILER = "Zig {s}";
            \\char const *LLAMA_BUILD_TARGET = "{s}_{s}";
            \\
        , .{ op.build_number, commit_hash.stdout[0 .. commit_hash.stdout.len - 1], zig_version, op.target.allocDescription(b.allocator) catch @panic("OOM"), @tagName(op.optimize) });

        return .{
            .b = b,
            .options = op,
            .path_prefix = path_prefix,
            .build_info = b.addObject(.{ .name = "llama-build-info", .target = op.target, .optimize = op.optimize, .root_source_file = b.addWriteFiles().add(build_info_path, build_info) }),
        };
    }

    /// build single library containing everything
    pub fn libAll(ctx: Context) *CompileStep {
        const lib_opt = .{ .name = "llama.cpp", .target = ctx.options.target, .optimize = ctx.options.optimize };
        const lib = if (ctx.options.shared) ctx.b.addSharedLibrary(lib_opt) else ctx.b.addStaticLibrary(lib_opt);
        ctx.addAll(lib);
        if (ctx.options.shared) {
            lib.defineCMacro("LLAMA_SHARED", null);
            lib.defineCMacro("LLAMA_BUILD", null);
            if (ctx.options.target.getOsTag() == .windows) {
                std.log.warn("For shared linking to work, requires header llama.h modification:\n\'#    if defined(_WIN32) && (!defined(__MINGW32__) || defined(ZIG))'", .{});
                lib.defineCMacro("ZIG", null);
            }
        }
        return lib;
    }

    /// link everything directly to target
    pub fn addAll(ctx: Context, compile: *CompileStep) void {
        ctx.addBuildInfo(compile);
        ctx.addGmaa(compile);
        ctx.addLLama(compile);
    }

    // zig module with translated headers
    pub fn addModule(ctx: Context) *Module {
        const tc = ctx.b.addTranslateC(.{
            .source_file = ctx.path("llama.h"),
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });
        if (ctx.options.shared) tc.defineCMacro("LLAMA_SHARED", null);
        return tc.addModule("llama.h");
    }

    pub fn addBuildInfo(ctx: Context, compile: *CompileStep) void {
        compile.addObject(ctx.build_info);
    }

    pub fn addGmaa(ctx: Context, compile: *CompileStep) void {
        ctx.common(compile);

        const sources = [_]LazyPath{
            ctx.path("ggml-alloc.c"),
            ctx.path("ggml-backend.c"),
            //ctx.path("ggml-mpi.c"),
            ctx.path("ggml-quants.c"),
            ctx.path("ggml.c"),
        };
        if (ctx.options.clblast) {
            compile.addCSourceFile(.{ .file = ctx.path("ggml-opencl.cpp"), .flags = ctx.flags() });
            compile.defineCMacro("GGML_USE_CLBLAST", null);

            const blast = clblast.ClBlast.build(ctx.b, .{
                .target = ctx.options.target,
                .optimize = ctx.options.optimize,
                .backend = .{ .opencl = ctx.options.opencl.? },
            });
            blast.link(compile);
            ctx.options.opencl.?.link(compile);
        }
        for (sources) |src| compile.addCSourceFile(.{ .file = src, .flags = ctx.flags() });
        //if (ctx.cuda) compile.ctx.path("ggml-cuda.cu");

    }

    pub fn addLLama(ctx: Context, compile: *CompileStep) void {
        ctx.common(compile);
        compile.addCSourceFile(.{ .file = ctx.path("llama.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path("common/common.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path("common/sampling.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path("common/grammar-parser.cpp"), .flags = ctx.flags() });
    }

    fn flags(ctx: Context) [][]const u8 {
        _ = ctx;
        return &.{};
    }

    fn common(ctx: Context, lib: *CompileStep) void {
        // https://github.com/ziglang/zig/issues/15448
        if (lib.target.getAbi() == .msvc) {
            lib.defineCMacro("_GNU_SOURCE", null);
            lib.linkLibC(); // need winsdk + crt
        } else lib.linkLibCpp(); // linkLibCpp already add (libc++ + libunwind + libc)

        lib.addIncludePath(ctx.path("")); // root
    }

    pub fn path(self: Context, p: []const u8) LazyPath {
        return .{ .path = self.b.pathJoin(&.{ self.path_prefix, p }) };
    }
};
