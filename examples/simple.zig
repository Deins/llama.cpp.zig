const std = @import("std");
const llama = @import("llama");
const slog = std.log.scoped(.main);
const arg_utils = @import("utils/args.zig");

const Model = llama.Model;
const Context = llama.Context;
const Token = llama.Token;
const TokenData = llama.TokenData;
const TokenDataArray = llama.TokenDataArray;

pub const Args = struct {
    model_path: [:0]const u8 = "models/dolphin-2.2.1-mistral-7b.Q3_K_M.gguf",
    prompt: ?[]const u8 = null,
    max_len: usize = 1024, // generate until eos, or this many characters are in prompt
    seed: ?u32 = null,
    threads: ?usize = null,
    threads_batch: ?usize = null,
    gpu_layers: i32 = 0,
};

pub fn run(alloc: std.mem.Allocator, args: Args) !void {
    llama.Backend.init(false);
    defer llama.Backend.deinit();
    slog.info("llama_system_info: {s}", .{llama.printSystemInfo()});
    llama.logSet(llama.utils.scopedLog, null);

    var mparams = Model.defaultParams();
    mparams.n_gpu_layers = args.gpu_layers;
    const model = try Model.initFromFile(args.model_path.ptr, mparams);
    defer model.deinit();

    var cparams = Context.defaultParams();
    cparams.seed = args.seed orelse 1234;
    const n_ctx_train = model.nCtxTrain();
    const n_ctx = n_ctx_train;
    cparams.n_ctx = @intCast(n_ctx_train);
    if (n_ctx > n_ctx_train) slog.warn("model was trained on only {} context tokens ({} specified)\n", .{ n_ctx_train, n_ctx });

    const cpu_threads = try std.Thread.getCpuCount(); // logical cpu cores
    cparams.n_threads = @intCast(args.threads orelse @min(cpu_threads, 4)); // for me: non batched doesn't scale above 3-4 cores
    cparams.n_threads_batch = @intCast(args.threads_batch orelse cpu_threads / 2); // for me without 2x hyperthreads per core works faster

    const ctx = try llama.Context.initWithModel(model, cparams);
    defer ctx.deinit();

    var prompt = try llama.Prompt.init(alloc, .{
        .model = model,
        .ctx = ctx,
        .batch_size = 512,
    });
    defer prompt.deinit();
    try prompt.appendText(args.prompt orelse @panic("--prompt argument is required"), true);
    const initial_prompt_len = prompt.tokens.items.len;

    var detokenizer = llama.Detokenizer.init(alloc);
    defer detokenizer.deinit();
    for (prompt.tokens.items) |tok| _ = try detokenizer.detokenize(model, tok);
    std.debug.print("PROMPT:\n{s}", .{detokenizer.getText()});
    detokenizer.clearRetainingCapacity();

    // generate response
    const token_eos = model.tokenEos();
    for (0..args.max_len) |_| {
        const token = try prompt.generateAppendOne();
        if (token == token_eos) break;
        std.debug.print("{s}", .{try detokenizer.detokenize(model, token)});
        detokenizer.clearRetainingCapacity();
    }

    // Generate alternate response
    prompt.shrink(initial_prompt_len);
    std.debug.print("\n\nAlternative generated reponse:\n", .{});
    for (0..args.max_len) |_| {
        const token = try prompt.generateAppendOne();
        if (token == token_eos) break;
        std.debug.print("{s}", .{try detokenizer.detokenize(model, token)});
        detokenizer.clearRetainingCapacity();
    }
    std.debug.print("\n", .{});

    ctx.printTimings();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer if (gpa.deinit() != .ok) @panic("memory leak detected when exiting");
    const alloc = gpa.allocator();

    slog.info("=" ** 80, .{});
    const args_raw = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args_raw);
    const maybe_args = arg_utils.parseArgs(Args, args_raw[1..]) catch |err| {
        slog.err("Could not parse comand line arguments! {}", .{err});
        arg_utils.printHelp(Args);
        return err;
    };
    const args = if (maybe_args) |args| args else {
        arg_utils.printHelp(Args);
        return;
    };

    try run(alloc, args);
}

pub const std_options = struct {
    pub const log_level = std.log.Level.debug;
    pub const log_scope_levels: []const std.log.ScopeLevel = &.{.{ .scope = .llama_cpp, .level = .info }};
};
