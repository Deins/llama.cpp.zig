// Compatible with Zig Version 0.11.0
// NOTE: this file borrows stuff from llama.cpp build zig, but as is it doesn't expose anything so could not be used
// TODO: implement/push needed changes upstream
const std = @import("std");
const llama = @import("build_llama.zig");
const CrossTarget = std.zig.CrossTarget;
const ArrayList = std.ArrayList;
const CompileStep = std.Build.CompileStep;
const ConfigHeader = std.Build.Step.ConfigHeader;
const Mode = std.builtin.Mode;
const TranslateCStep = std.Build.TranslateCStep;
const Module = std.Build.Module;

const clblast = @import("clblast");

const llama_cpp_path_prefix = "llama.cpp/"; // point to where llama.cpp root is

pub fn build(b: *std.Build) !void {
    const use_clblast = b.option(bool, "clblast", "Use clblast acceleration") orelse true;

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lto = b.option(bool, "lto", "Enable LTO optimization, (default: false)") orelse false;

    const opencl_maybe = llama.clblast.OpenCL.fromOCL(b, target);
    if (use_clblast and opencl_maybe == null) @panic("OpenCL not found. Please specify include or libs manually if its installed!");
    var llama_build_ctx = llama.Context.init(b, .{
        .target = target,
        .optimize = optimize,
        .shared = false,
        .lto = lto,
        .opencl = opencl_maybe,
        .clblast = use_clblast,
    }, "llama.cpp");
    var llama_lib = llama_build_ctx.libAll();
    b.installArtifact(llama_lib);

    // var lcpp = try LlamaCpp.init(b, target, optimize, lto, "./llama.cpp/");
    const llama_zig = b.createModule(.{ .source_file = .{ .path = "llama.cpp.zig/llama.zig" }, .dependencies = &.{.{
        .name = "llama.h",
        .module = llama_build_ctx.addModule(),
    }} });

    { // simple exampel
        var exe = b.addExecutable(.{ .name = "simple", .root_source_file = .{ .path = "examples/simple.zig" } });
        exe.linkLibrary(llama_lib);
        exe.addModule("llama", llama_zig);
        if (opencl_maybe) |ocl| ocl.link(exe);
        b.installArtifact(exe); // location when the user invokes the "install" step (the default step when running `zig build`).

        const run_exe = b.addRunArtifact(exe);
        if (b.args) |args| run_exe.addArgs(args); // passes on args like: zig build run -- my fancy args
        run_exe.step.dependOn(b.default_step); // allways copy output, to avoid confusion
        b.step("run-simple", "Run simple example").dependOn(&run_exe.step);
    }

    { // tests
        const main_tests = b.addTest(.{
            .root_source_file = .{ .path = "llama.cpp.zig/llama.zig" },
            .target = target,
            .optimize = optimize,
        });
        main_tests.linkLibrary(llama_lib);
        const run_main_tests = b.addRunArtifact(main_tests);

        const test_step = b.step("test", "Run library tests");
        test_step.dependOn(&run_main_tests.step);
    }
}
