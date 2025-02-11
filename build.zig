const std = @import("std");
const llama = @import("build_llama.zig");
const Target = std.Build.ResolvedTarget;
const ArrayList = std.ArrayList;
const CompileStep = std.Build.Step.Compile;
const ConfigHeader = std.Build.Step.ConfigHeader;
const Mode = std.builtin.Mode;
const TranslateCStep = std.Build.TranslateCStep;
const Module = std.Build.Module;

pub const clblast = @import("clblast");

pub const llama_cpp_path_prefix = "llama.cpp/"; // point to where llama.cpp root is

pub const Options = struct {
    target: Target,
    optimize: Mode,
    clblast: bool = false,
    source_path: []const u8 = "",
    backends: llama.Backends = .{},
};

/// Build context
pub const Context = struct {
    const Self = @This();
    b: *std.Build,
    options: Options,
    /// llama.cpp build context
    llama: llama.Context,
    /// zig module
    module: *Module,
    /// llama.h translated header file module, mostly for internal use
    llama_h_module: *Module,
    /// ggml.h  translated header file module, mostly for internal use
    ggml_h_module: *Module,

    pub fn init(b: *std.Build, options: Options) Self {
        var llama_cpp = llama.Context.init(b, .{
            .target = options.target,
            .optimize = options.optimize,
            .shared = false,
            .backends = options.backends,
        });

        const llama_h_module = llama_cpp.moduleLlama();
        const ggml_h_module = llama_cpp.moduleGgml();
        const imports: []const std.Build.Module.Import = &.{
            .{
                .name = "llama.h",
                .module = llama_h_module,
            },
            .{
                .name = "ggml.h",
                .module = ggml_h_module,
            },
        };
        const mod = b.createModule(.{
            .root_source_file = b.path(b.pathJoin(&.{ options.source_path, "llama.cpp.zig/llama.zig" })),
            .imports = imports,
        });

        return .{
            .b = b,
            .options = options,
            .llama = llama_cpp,
            .module = mod,
            .llama_h_module = llama_h_module,
            .ggml_h_module = ggml_h_module,
        };
    }

    pub fn link(self: *Self, comp: *CompileStep) void {
        self.llama.link(comp);
    }

    pub fn sample(self: *Self, path: []const u8, name: []const u8) void {
        const b = self.b;
        var exe = b.addExecutable(.{
            .name = name,
            .target = self.options.target,
            .optimize = self.options.optimize,
            .root_source_file = b.path(b.pathJoin(&.{ path, std.mem.join(b.allocator, "", &.{ name, ".zig" }) catch @panic("OOM") })),
        });
        exe.stack_size = 32 * 1024 * 1024;
        exe.root_module.addImport("llama", self.module);
        self.link(exe);
        b.installArtifact(exe); // location when the user invokes the "install" step (the default step when running `zig build`).

        const run_exe = b.addRunArtifact(exe);
        if (b.args) |args| run_exe.addArgs(args); // passes on args like: zig build run -- my fancy args
        run_exe.step.dependOn(b.default_step); // allways copy output, to avoid confusion
        b.step(b.fmt("run-{s}", .{name}), b.fmt("Run {s} example", .{name})).dependOn(&run_exe.step);
    }
};

pub fn build(b: *std.Build) !void {
    const install_cpp_samples = b.option(bool, "cpp_samples", "Install llama.cpp samples") orelse false;

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    var llama_zig = Context.init(b, .{
        .target = target,
        .optimize = optimize,
    });

    llama_zig.llama.samples(install_cpp_samples) catch |err| std.log.err("Can't build CPP samples, error: {}", .{err});

    llama_zig.sample("examples", "simple");
    // llama_zig.sample("examples", "interactive");

    { // tests
        const main_tests = b.addTest(.{
            .root_source_file = b.path("llama.cpp.zig/llama.zig"),
            .target = target,
            .optimize = optimize,
        });
        llama_zig.link(main_tests);
        main_tests.root_module.addImport("llama.h", llama_zig.llama_h_module);
        const run_main_tests = b.addRunArtifact(main_tests);

        const test_step = b.step("test", "Run library tests");
        test_step.dependOn(&run_main_tests.step);
    }
}
