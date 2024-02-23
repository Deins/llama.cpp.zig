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
    lib: ?*CompileStep = null,

    pub fn init(b: *Builder, op: Options) Context {
        const path_prefix = b.pathJoin(&.{ thisPath(), "/llama.cpp" });
        const zig_version = @import("builtin").zig_version_string;
        const exec = if (@hasDecl(std.ChildProcess, "exec")) std.ChildProcess.exec else std.ChildProcess.run; // zig 11 vs nightly compatibility
        const commit_hash = exec(
            .{ .allocator = b.allocator, .argv = &.{ "git", "rev-parse", "HEAD" } },
        ) catch |err| {
            std.log.err("Cant get git comiit hash! err: {}", .{err});
            unreachable;
        };

        const build_info_path = b.pathJoin(&.{ "common", "build-info.cpp" });
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

    /// just builds everything needed and links it to your target
    pub fn link(ctx: *Context, comp: *CompileStep) void {
        comp.linkLibrary(ctx.library());
        if (ctx.options.opencl) |ocl| ocl.link(comp);
    }

    /// build single library containing everything
    pub fn library(ctx: *Context) *CompileStep {
        if (ctx.lib) |l| return l;
        const lib_opt = .{ .name = "llama.cpp", .target = ctx.options.target, .optimize = ctx.options.optimize };
        const lib = if (ctx.options.shared) ctx.b.addSharedLibrary(lib_opt) else ctx.b.addStaticLibrary(lib_opt);
        ctx.addAll(lib);
        if (ctx.options.target.getAbi() != .msvc)
            lib.defineCMacro("_GNU_SOURCE", null);
        if (ctx.options.shared) {
            lib.defineCMacro("LLAMA_SHARED", null);
            lib.defineCMacro("LLAMA_BUILD", null);
            if (ctx.options.target.getOsTag() == .windows) {
                std.log.warn("For shared linking to work, requires header llama.h modification:\n\'#    if defined(_WIN32) && (!defined(__MINGW32__) || defined(ZIG))'", .{});
                lib.defineCMacro("ZIG", null);
            }
        }
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
            .source_file = ctx.path("llama.h"),
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });
        if (ctx.options.shared) tc.defineCMacro("LLAMA_SHARED", null);
        tc.defineCMacro("NDEBUG", null); // otherwise zig is unhappy about c ASSERT macro
        return tc.addModule("llama.h");
    }

    /// zig module with translated headers
    pub fn moduleGgml(ctx: *Context) *Module {
        const tc = ctx.b.addTranslateC(.{
            .source_file = ctx.path("ggml.h"),
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });
        if (ctx.options.shared) tc.defineCMacro("LLAMA_SHARED", null);
        tc.defineCMacro("NDEBUG", null); // otherwise zig is unhappy about c ASSERT macro
        return tc.addModule("ggml.h");
    }
    
    /// zig module with translated headers
    pub fn moduleLlamaCppBindings(ctx: *Context) *Module {
        const tc = ctx.b.addTranslateC(.{
            .source_file = .{ .path = "cpp_bindings/bindings.h", },
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });
        if (ctx.options.shared) tc.defineCMacro("LLAMA_SHARED", null);
        tc.defineCMacro("NDEBUG", null); // otherwise zig is unhappy about c ASSERT macro
        tc.addIncludeDir("llama.cpp");
        return tc.addModule("lamma_cpp_bindings.h");
    }

    pub fn addBuildInfo(ctx: *Context, compile: *CompileStep) void {
        compile.addObject(ctx.build_info);
    }

    pub fn addGgml(ctx: *Context, compile: *CompileStep) void {
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

    pub fn addLLama(ctx: *Context, compile: *CompileStep) void {
        ctx.common(compile);
        compile.addCSourceFile(.{ .file = ctx.path("llama.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path("common/common.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path("common/sampling.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path("common/grammar-parser.cpp"), .flags = ctx.flags() });
        // kind of optional depending on context, see if worth spliting in another lib
        compile.addCSourceFile(.{ .file = ctx.path("common/train.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path("common/console.cpp"), .flags = ctx.flags() });
        // c++ bindings
        compile.addIncludePath(ctx.path("common"));
        compile.addIncludePath(.{ .path = "cpp_bindings", });
        compile.addCSourceFile(.{ .file = .{ .path = ctx.b.pathJoin(&.{"cpp_bindings", "grammar.cpp"}) }, .flags = ctx.flags() });
    }

    pub fn samples(ctx: *Context, install: bool) !void {
        const b = ctx.b;
        const examples = [_][]const u8{
            "main",
            "quantize",
            "perplexity",
            "embedding",
            "finetune",
            "train-text-from-scratch",
            "lookahead",
            "speculative",
            "parallel",
        };

        for (examples) |ex| {
            const exe = b.addExecutable(.{ .name = ex, .target = ctx.options.target, .optimize = ctx.options.optimize });
            exe.want_lto = false; // TODO: review, causes: error: lld-link: undefined symbol: __declspec(dllimport) _create_locale
            if (install) b.installArtifact(exe);
            { // add all c/cpp files from example dir
                const rpath = b.pathJoin(&.{ ctx.path_prefix, "examples", ex });
                exe.addIncludePath(.{ .path = rpath });
                var dir = if (@hasDecl(std.fs, "openIterableDirAbsolute")) try std.fs.openIterableDirAbsolute(b.pathFromRoot(rpath), .{}) else try std.fs.openDirAbsolute(b.pathFromRoot(rpath), .{ .iterate = true }); // zig 11 vs nightly compatibility
                defer dir.close();
                var dir_it = dir.iterate();
                while (try dir_it.next()) |f| switch (f.kind) {
                    .file => if (std.ascii.endsWithIgnoreCase(f.name, ".c") or std.ascii.endsWithIgnoreCase(f.name, ".cpp")) {
                        const src = b.pathJoin(&.{ ctx.path_prefix, "examples", ex, f.name });
                        exe.addCSourceFile(.{ .file = .{ .path = src }, .flags = &.{} });
                    },
                    else => {},
                };
            }
            exe.addIncludePath(ctx.path("common"));
            ctx.common(exe);
            ctx.link(exe);

            const run_exe = b.addRunArtifact(exe);
            if (b.args) |args| run_exe.addArgs(args); // passes on args like: zig build run -- my fancy args
            run_exe.step.dependOn(b.default_step); // allways copy output, to avoid confusion
            b.step(b.fmt("run-cpp-{s}", .{ex}), b.fmt("Run llama.cpp example: {s}", .{ex})).dependOn(&run_exe.step);
        }
    }

    fn flags(ctx: Context) [][]const u8 {
        _ = ctx;
        return &.{};
    }

    fn common(ctx: Context, lib: *CompileStep) void {
        lib.linkLibCpp();
        lib.addIncludePath(ctx.path("")); // root
        if (ctx.options.optimize != .Debug) lib.defineCMacro("NDEBUG", null);
    }

    pub fn path(self: Context, p: []const u8) LazyPath {
        return .{ .path = self.b.pathJoin(&.{ self.path_prefix, p }) };
    }
};

fn thisPath() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}
