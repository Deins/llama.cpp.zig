const std = @import("std");
const llama = @import("llama");
const slog = std.log.scoped(.main);
const arg_utils = @import("utils/args.zig");

const Model = llama.Model;
const Context = llama.Context;
const Token = llama.Token;
const TokenData = llama.TokenData;
const TokenDataArray = llama.TokenDataArray;
const trimFront = llama.utils.trimFront;
const trimBack = llama.utils.trimBack;
const trim = llama.utils.trim;

pub const Args = struct {
    model_path: [:0]const u8 = "models/dolphin-2.2.1-mistral-7b.Q3_K_M.gguf",
    max_len: usize = 1024, // generate until eos, or this many characters are in prompt
    seed: ?u32 = null,
    threads: ?usize = null,
    threads_batch: ?usize = null,
    gpu_layers: i32 = 0,
    template: ?[]const u8 = null,
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
    const n_ctx = n_ctx_train * 2;
    cparams.n_ctx = @intCast(n_ctx_train);
    if (n_ctx > n_ctx_train) slog.warn("model was trained on only {} context tokens ({} specified)\n", .{ n_ctx_train, n_ctx });

    const cpu_threads = try std.Thread.getCpuCount(); // logical cpu cores
    cparams.n_threads = @intCast(args.threads orelse @min(cpu_threads, 4)); // for me: non batched doesn't scale above 3-4 cores
    cparams.n_threads_batch = @intCast(args.threads_batch orelse cpu_threads / 2); // for me with 2x hyperthreads per core works faster

    const ctx = try llama.Context.initWithModel(model, cparams);
    defer ctx.deinit();

    var prompt_gen: ?llama.utils.TemplatedPrompt = blk: {
        if (args.template == null) break :blk null;
        var prompt_gen_parms = llama.utils.TemplatedPrompt.templateFromName(args.template.?) orelse return error.InvalidTemplate;
        prompt_gen_parms.additional_replacements = &[_][2][]const u8{
            [2][]const u8{ "{{char}}", "Alice" },
            [2][]const u8{ "{{user}}", "Bob" },
        };
        //prompt_parms.setTokensFromModel(model); // behaviour is inconsistent between models, better adapt/specify template directly if needed
        break :blk llama.utils.TemplatedPrompt.init(alloc, prompt_gen_parms);
    };
    defer if (prompt_gen != null) prompt_gen.?.deinit();

    var prompt = try llama.Prompt.init(alloc, .{
        .model = model,
        .ctx = ctx,
        .batch_size = 512,
    });
    defer prompt.deinit();

    var detokenizer = llama.Detokenizer.init(alloc);
    defer detokenizer.deinit();

    const stdin = std.io.getStdIn();
    var reader = stdin.reader();

    std.debug.print("Template: {s}\n", .{if (prompt_gen == null) "off" else "on"});
    std.debug.print("{s}", .{if (prompt_gen != null)
        \\TEMPLATED PROMPT (start typing)\n
        \\input \g to generate response
        \\input \u {{user}} to switch user the prompt is for. 
        \\          Most prompt formats expect either: system, user & assistant
        \\--------------------------------------------------------------------------------
        \\
    else
        \\RAW PROMPT (start typing)\n
        \\If you use instructed model, you have to follow the template yourself.
        \\input \g to generate
        \\--------------------------------------------------------------------------------
        \\
    });
    var user: []const u8 = try alloc.dupe(u8, "user");
    defer alloc.free(user);
    while (true) {
        var input_txt = std.ArrayList(u8).init(alloc);
        defer input_txt.deinit();

        if (prompt_gen) |*gen| {
            gen.clearRetainingCapacity();
            while (true) {
                const input = try reader.readUntilDelimiterAlloc(alloc, '\n', std.math.maxInt(usize));
                defer alloc.free(input);
                var ti = trimFront(input);
                if (std.ascii.startsWithIgnoreCase(ti, "\\u")) {
                    if (input_txt.items.len > 0) {
                        try gen.add(user, input_txt.items);
                        input_txt.clearRetainingCapacity();
                    }
                    alloc.free(user);
                    user = try alloc.dupe(u8, llama.utils.trim(ti[2..]));
                    slog.info("continuing as user: '{s}'", .{user});
                    continue;
                }
                if (std.ascii.indexOfIgnoreCase(ti, "\\g")) |pos| {
                    try input_txt.appendSlice(trimBack(ti[0..pos]));
                    break;
                }
                if (std.ascii.startsWithIgnoreCase(ti, "\\p")) {
                    for (prompt.tokens.items) |tok| _ = try detokenizer.detokenizeWithSpecial(model, tok);
                    std.debug.print("PROMPT:\n" ++ "-" ** 80 ++ "\n{s}\n" ++ "-" ** 80 ++ "\n", .{detokenizer.getText()});
                    detokenizer.clearRetainingCapacity();
                    continue;
                }
                try input_txt.appendSlice(trimBack(ti));
                try input_txt.append('\n');
            }
            if (input_txt.items.len > 0) {
                try gen.add(user, input_txt.items);
                input_txt.clearRetainingCapacity();
            }
            if (!std.ascii.eqlIgnoreCase("assistant", user)) try gen.add("assistant", "{{generate}}"); // auto switch user
            try prompt.appendText(gen.text.items, false);
        } else {
            while (true) {
                const input = try reader.readUntilDelimiterAlloc(alloc, '\n', std.math.maxInt(usize));
                defer alloc.free(input);
                const end = std.mem.indexOf(u8, input, "\\g");
                if (end) |eidx| try prompt.appendText(trim(input[0..eidx]), true) else try prompt.appendText(trim(input), true);
                if (end != null) break;
            }
        }

        // generate response
        const eos = model.tokenEos();
        for (0..args.max_len) |_| {
            const tok = try prompt.generateAppendOne();
            if (tok == eos) break;
            std.debug.print("{s}", .{try detokenizer.detokenize(model, tok)});
            detokenizer.clearRetainingCapacity();
        }
        std.debug.print("\n", .{});
    }

    ctx.printTimings();
}

pub fn main() !void {
    //arg_utils.configure_console();

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
