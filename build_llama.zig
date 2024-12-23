const std = @import("std");
const Builder = std.Build;
const Target = std.Build.ResolvedTarget;
const Mode = std.builtin.Mode;
const CompileStep = std.Build.Step.Compile;
const LazyPath = std.Build.LazyPath;
const Module = std.Build.Module;

pub const Options = struct {
    target: Target,
    optimize: Mode,
    shared: bool, // static or shared lib
    build_number: usize = 0, // number that will be writen in build info
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
        comp.linkLibrary(ctx.library());
    }

    /// build single library containing everything
    pub fn library(ctx: *Context) *CompileStep {
        if (ctx.lib) |l| return l;
        const lib_opt = .{ .name = "llama.cpp", .target = ctx.options.target, .optimize = ctx.options.optimize };
        const lib = if (ctx.options.shared) ctx.b.addSharedLibrary(lib_opt) else ctx.b.addStaticLibrary(lib_opt);
        ctx.addAll(lib);
        if (ctx.options.target.result.abi != .msvc)
            lib.defineCMacro("_GNU_SOURCE", null);
        if (ctx.options.shared) {
            lib.defineCMacro("LLAMA_SHARED", null);
            lib.defineCMacro("LLAMA_BUILD", null);
            if (ctx.options.target.result.os.tag == .windows) {
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

        const sources = [_]LazyPath{
            ctx.path(&.{ "ggml", "src", "ggml-alloc.c" }),
            ctx.path(&.{ "ggml", "src", "ggml-backend-reg.cpp" }),
            ctx.path(&.{ "ggml", "src", "ggml-backend.cpp" }),
            ctx.path(&.{ "ggml", "src", "ggml-opt.cpp" }),
            ctx.path(&.{ "ggml", "src", "ggml-quants.c" }),
            ctx.path(&.{ "ggml", "src", "ggml-threading.cpp" }),
            ctx.path(&.{ "ggml", "src", "ggml.c" }),
        };
        for (sources) |src| compile.addCSourceFile(.{ .file = src, .flags = ctx.flags() });
        //if (ctx.cuda) compile.ctx.path("ggml-cuda.cu");

    }

    pub fn addLLama(ctx: *Context, compile: *CompileStep) void {
        ctx.common(compile);
        compile.addIncludePath(ctx.path(&.{"include"}));
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-vocab.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-grammar.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-sampling.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-vocab.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("unicode-data.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("unicode.cpp"), .flags = ctx.flags() });

        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "common.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "sampling.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "console.cpp" }), .flags = ctx.flags() });

        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "json-schema-to-grammar.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "speculative.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "ngram-cache.cpp" }), .flags = ctx.flags() });
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
            exe.addCSourceFile(.{ .file = ctx.path(&.{ "common", "log.cpp" }), .flags = ctx.flags() });
            exe.addCSourceFile(.{ .file = ctx.path(&.{ "common", "arg.cpp" }), .flags = ctx.flags() });

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
                        exe.addCSourceFile(.{ .file = .{ .cwd_relative = src }, .flags = &.{} });
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

    fn flags(ctx: Context) [][]const u8 {
        _ = ctx;
        return &.{};
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
