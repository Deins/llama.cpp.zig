// Compatible with Zig Version 0.11.0
// NOTE: this file borrows stuff from llama.cpp build zig, but as is it doesn't expose anything so could not be used
// TODO: implement/push needed changes upstream
const std = @import("std");
const CrossTarget = std.zig.CrossTarget;
const ArrayList = std.ArrayList;
const CompileStep = std.Build.CompileStep;
const ConfigHeader = std.Build.Step.ConfigHeader;
const Mode = std.builtin.Mode;
const TranslateCStep = std.Build.TranslateCStep;
const Module = std.Build.Module;

const llama_cpp_path_prefix = "llama.cpp/"; // point to where llama.cpp root is

pub const LlamaCpp = struct {
    maker: Maker,
    ggml: *CompileStep,
    ggml_alloc: *CompileStep,
    ggml_backend: *CompileStep,
    ggml_quants: *CompileStep,
    llama: *CompileStep,
    buildinfo: *CompileStep,
    common: *CompileStep,
    console: *CompileStep,
    sampling: *CompileStep,
    grammar_parser: *CompileStep,
    train: *CompileStep,
    clip: *CompileStep,
    translate_c: *TranslateCStep,
    module: *Module,

    pub fn init(b: *std.Build, target: CrossTarget, optimize: Mode, lto: bool, comptime path_prefix: []const u8) !LlamaCpp {
        var make = try Maker.init(b, target, optimize, path_prefix);
        make.enable_lto = lto;
        const tc = b.addTranslateC(.{ .source_file = .{ .path = path_prefix ++ "llama.h" }, .target = target, .optimize = optimize });
        tc.defineCMacro("NDEBUG", null); // assert macros transpile but cause compile error: cannot compare strings with != at GGML_ASSERT(!("statement should not be reached" != 0))
        var res = LlamaCpp{
            .maker = make,
            .ggml = make.obj("ggml", path_prefix ++ "ggml.c"),
            .ggml_alloc = make.obj("ggml-alloc", path_prefix ++ "ggml-alloc.c"),
            .ggml_backend = make.obj("ggml-backend", path_prefix ++ "ggml-backend.c"),
            .ggml_quants = make.obj("ggml-quants", path_prefix ++ "ggml-quants.c"),
            .llama = make.obj("llama", path_prefix ++ "llama.cpp"),
            .buildinfo = make.obj("common", path_prefix ++ "common/build-info.cpp"),
            .common = make.obj("common", path_prefix ++ "common/common.cpp"),
            .console = make.obj("console", path_prefix ++ "common/console.cpp"),
            .sampling = make.obj("sampling", path_prefix ++ "common/sampling.cpp"),
            .grammar_parser = make.obj("grammar-parser", path_prefix ++ "common/grammar-parser.cpp"),
            .train = make.obj("train", path_prefix ++ "common/train.cpp"),
            .clip = make.obj("clip", path_prefix ++ "examples/llava/clip.cpp"),

            .translate_c = tc,
            .module = tc.createModule(),
        };
        for (res.maker.objs.items) |obj| obj.addIncludePath(.{ .path = path_prefix });
        return res;
    }

    pub fn link(self: LlamaCpp, lib_or_exe: *CompileStep, deps: []const *CompileStep) void {
        const m = self.maker;
        for (deps) |d| lib_or_exe.addObject(d);
        for (m.objs.items) |o| lib_or_exe.addObject(o);
        for (m.include_dirs.items) |i| lib_or_exe.addIncludePath(.{ .path = i });

        // https://github.com/ziglang/zig/issues/15448
        if (lib_or_exe.target.getAbi() == .msvc) {
            lib_or_exe.linkLibC(); // need winsdk + crt
        } else {
            // linkLibCpp already add (libc++ + libunwind + libc)
            lib_or_exe.linkLibCpp();
        }
        lib_or_exe.want_lto = m.enable_lto;
        lib_or_exe.addModule("llama.h", self.module);
    }

    /// link all lama.cpp components to your lib/exe target
    pub fn linkAll(self: LlamaCpp, lib_or_exe: *CompileStep) void {
        self.link(lib_or_exe, &.{ self.ggml, self.ggml_alloc, self.ggml_backend, self.ggml_quants, self.llama, self.common, self.buildinfo, self.sampling, self.grammar_parser });
    }
};

pub fn linkLlamaZig(lcpp: LlamaCpp, lib_or_exe: *CompileStep) void {
    lcpp.link(lib_or_exe, &.{ lcpp.ggml, lcpp.ggml_alloc, lcpp.ggml_backend, lcpp.ggml_quants, lcpp.llama, lcpp.common, lcpp.buildinfo, lcpp.sampling, lcpp.grammar_parser });
}

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lto = b.option(bool, "lto", "Enable LTO optimization, (default: false)") orelse false;

    var lcpp = try LlamaCpp.init(b, target, optimize, lto, "./llama.cpp/");
    const llama_zig = b.createModule(.{ .source_file = .{ .path = "llama.cpp.zig/llama.zig" }, .dependencies = &.{.{
        .name = "llama.h",
        .module = lcpp.module,
    }} });

    { // simple exampel
        var exe = b.addExecutable(.{ .name = "simple", .root_source_file = .{ .path = "examples/simple.zig" } });
        linkLlamaZig(lcpp, exe);
        exe.addModule("llama", llama_zig);
        b.installArtifact(exe); // location when the user invokes the "install" step (the default step when running `zig build`).

        const run_exe = b.addRunArtifact(exe);
        if (b.args) |args| run_exe.addArgs(args); // passes on args like: zig build run -- my fancy args
        run_exe.step.dependOn(b.default_step); // allways copy output, to avoid confusion
        b.step("run-simple", "Run simple example").dependOn(&run_exe.step);
    }

    { // tests
        const main_tests = b.addTest(.{
            .root_source_file = .{ .path = "src/llama.zig" },
            .target = target,
            .optimize = optimize,
        });
        linkLlamaZig(lcpp, main_tests);
        const run_main_tests = b.addRunArtifact(main_tests);

        const test_step = b.step("test", "Run library tests");
        test_step.dependOn(&run_main_tests.step);
    }
}

//
// Stuff from original llama.cpp zig build
//  TODO: make needed chages and push them to upstream
//

const Maker = struct {
    builder: *std.build.Builder,
    target: CrossTarget,
    optimize: Mode,
    enable_lto: bool,

    include_dirs: ArrayList([]const u8),
    cflags: ArrayList([]const u8),
    cxxflags: ArrayList([]const u8),
    objs: ArrayList(*CompileStep),

    fn addInclude(m: *Maker, dir: []const u8) !void {
        try m.include_dirs.append(dir);
    }
    fn addProjectInclude(m: *Maker, path: []const []const u8) !void {
        try m.addInclude(try m.builder.build_root.join(m.builder.allocator, path));
    }
    fn addCFlag(m: *Maker, flag: []const u8) !void {
        try m.cflags.append(flag);
    }
    fn addCxxFlag(m: *Maker, flag: []const u8) !void {
        try m.cxxflags.append(flag);
    }
    fn addFlag(m: *Maker, flag: []const u8) !void {
        try m.addCFlag(flag);
        try m.addCxxFlag(flag);
    }

    fn init(builder: *std.build.Builder, target: CrossTarget, optimize: std.builtin.Mode, comptime path_prefix: []const u8) !Maker {
        const zig_version = @import("builtin").zig_version_string;
        const commit_hash = try std.ChildProcess.exec(
            .{ .allocator = builder.allocator, .argv = &.{ "git", "rev-parse", "HEAD" } },
        );
        try std.fs.cwd().writeFile(path_prefix ++ "common/build-info.cpp", builder.fmt(
            \\int LLAMA_BUILD_NUMBER = {};
            \\char const *LLAMA_COMMIT = "{s}";
            \\char const *LLAMA_COMPILER = "Zig {s}";
            \\char const *LLAMA_BUILD_TARGET = "{s}";
            \\
        , .{ 0, commit_hash.stdout[0 .. commit_hash.stdout.len - 1], zig_version, try target.allocDescription(builder.allocator) }));
        var m = Maker{
            .builder = builder,
            .target = target,
            .optimize = optimize,
            .enable_lto = false,
            .include_dirs = ArrayList([]const u8).init(builder.allocator),
            .cflags = ArrayList([]const u8).init(builder.allocator),
            .cxxflags = ArrayList([]const u8).init(builder.allocator),
            .objs = ArrayList(*CompileStep).init(builder.allocator),
        };

        try m.addCFlag("-std=c11");
        try m.addCxxFlag("-std=c++11");
        try m.addProjectInclude(&.{path_prefix});
        try m.addProjectInclude(&.{path_prefix ++ "common"});
        return m;
    }

    fn obj(m: *const Maker, name: []const u8, src: []const u8) *CompileStep {
        const o = m.builder.addObject(.{ .name = name, .target = m.target, .optimize = m.optimize });
        if (o.target.getAbi() != .msvc)
            o.defineCMacro("_GNU_SOURCE", null);

        if (std.mem.endsWith(u8, src, ".c")) {
            o.addCSourceFiles(&.{src}, m.cflags.items);
            o.linkLibC();
        } else {
            o.addCSourceFiles(&.{src}, m.cxxflags.items);
            if (o.target.getAbi() == .msvc) {
                o.linkLibC(); // need winsdk + crt
            } else {
                // linkLibCpp already add (libc++ + libunwind + libc)
                o.linkLibCpp();
            }
        }
        for (m.include_dirs.items) |i| o.addIncludePath(.{ .path = i });
        o.want_lto = m.enable_lto;
        return o;
    }

    fn exe(m: *const Maker, name: []const u8, src: []const u8, deps: []const *CompileStep) *CompileStep {
        const e = m.builder.addExecutable(.{ .name = name, .target = m.target, .optimize = m.optimize });
        e.addCSourceFiles(&.{src}, m.cxxflags.items);
        for (deps) |d| e.addObject(d);
        for (m.objs.items) |o| e.addObject(o);
        for (m.include_dirs.items) |i| e.addIncludePath(.{ .path = i });

        // https://github.com/ziglang/zig/issues/15448
        if (e.target.getAbi() == .msvc) {
            e.linkLibC(); // need winsdk + crt
        } else {
            // linkLibCpp already add (libc++ + libunwind + libc)
            e.linkLibCpp();
        }
        m.builder.installArtifact(e);
        e.want_lto = m.enable_lto;
        return e;
    }

    /// build sources with dependencies as static library for easier linking
    fn staticLib(m: *const Maker, name: []const u8, src: []const u8, deps: []const *CompileStep) *CompileStep {
        const e = m.builder.addStaticLibrary(.{ .name = name, .target = m.target, .optimize = m.optimize });
        e.addCSourceFiles(&.{src}, m.cxxflags.items);
        for (deps) |d| e.addObject(d);
        for (m.objs.items) |o| e.addObject(o);
        for (m.include_dirs.items) |i| e.addIncludePath(.{ .path = i });

        // https://github.com/ziglang/zig/issues/15448
        if (e.target.getAbi() == .msvc) {
            e.linkLibC(); // need winsdk + crt
        } else {
            // linkLibCpp already add (libc++ + libunwind + libc)
            e.linkLibCpp();
        }
        m.builder.installArtifact(e);
        e.want_lto = m.enable_lto;
        return e;
    }
};
