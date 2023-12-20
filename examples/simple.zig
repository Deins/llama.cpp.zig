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
    json_prompt_path: ?[:0]const u8 = null, // exclusive, set either prompt or json_prompt_path
    max_len: usize = 1024, // generate until eos, or this many characters are in prompt
    seed: ?u32 = null,
    threads: ?usize = null,
    threads_batch: ?usize = null,
};

pub fn run(alloc: std.mem.Allocator, args: Args) !void {
    llama.Backend.init(false);
    defer llama.Backend.deinit();
    llama.logSet(llama.utils.scopedLog, null);

    var mparams = llama.modelDefaultParams();
    const model = try Model.initFromFile(args.model_path.ptr, mparams);
    defer model.deinit();

    var cparams = llama.contextDefaultParams();
    const n_ctx_train = model.nCtxTrain();
    const n_ctx = n_ctx_train;
    cparams.seed = args.seed orelse 1234;
    cparams.n_ctx = @intCast(n_ctx_train);
    if (n_ctx > n_ctx_train) slog.warn("model was trained on only {} context tokens ({} specified)\n", .{ n_ctx_train, n_ctx });

    const cpu_threads = try std.Thread.getCpuCount(); // logical cpu cores
    cparams.n_threads = @intCast(args.threads orelse @min(cpu_threads, 4)); // for me: non batched doesn't scale above 3-4 cores
    cparams.n_threads_batch = @intCast(args.threads_batch orelse cpu_threads); // for me with 2x hyperthreads per core: n_cores/2 wors tiny bit faster, but lets keep defaults simple

    const ctx = try llama.Context.initWithModel(model, cparams);
    defer ctx.deinit();

    var prompt_parms = llama.utils.TemplatedPrompt.Params{};
    //prompt_parms.setTokensFromModel(model); // usually doesn't work, many models have bos token '<|im_end|>' and other weirdnesses that doesn't match model description
    var prompt_gen = llama.utils.TemplatedPrompt.init(alloc, prompt_parms);
    defer prompt_gen.deinit();
    if (args.json_prompt_path) |path| try prompt_gen.addFromJsonFile(path);

    const prompt = args.prompt orelse (if (args.json_prompt_path != null) prompt_gen.text.items else "User forgot to pass in promt! What should we do? ");

    var tokenizer = llama.Tokenizer.init(alloc);
    defer tokenizer.deinit();
    try tokenizer.tokenize(model, prompt, false, true);

    var detokenizer = llama.Detokenizer.init(alloc);
    defer detokenizer.deinit();
    for (tokenizer.getTokens()) |tok| try detokenizer.detokenizeWithSpecial(model, tok);
    std.debug.print("PROMPT:\n{s}", .{detokenizer.getText()});
    detokenizer.clearRetainingCapacity();

    // Sampling is responsible of picking next token after decoding computes probability for each token.
    var sampling_ctx = try llama.sampling.SamplingContext.init(alloc, .{});
    defer sampling_ctx.deinit();

    const n_vocab = model.nVocab();
    var batch = llama.Batch.init(512, 0, 1);
    for (tokenizer.getTokens(), 0..) |tok, i| {
        sampling_ctx.accept(ctx, tok, false); // sampling_ctx need to know history of prompt therefore it has to be kept in sync
        batch.add(tok, @intCast(i), &.{0}, false);
    }
    // make llama_decode to output logits only for the last token of the prompt
    batch.logits[@as(usize, @intCast(batch.n_tokens)) - 1] = true;

    var n_cur = batch.n_tokens;
    var n_decoded: usize = 0;
    var candidates: []TokenData = try alloc.alloc(TokenData, @intCast(n_vocab));
    defer alloc.free(candidates);

    while (n_cur <= args.max_len) {
        // evaluate the current batch with the transformer model
        batch.decode(ctx) catch |err| switch (err) {
            error.NoKvSlotWarning => slog.warn("could not find a KV slot for the batch (try reducing the size of the batch or increase the context ", .{}),
            error.UnknownWarning => slog.warn("decoding warning: {}", .{err}),
            else => return err,
        };

        // sample the next token
        {
            const new_token = blk: {
                const greedy_sampling = false; // shows how to to decode using lower level functions without sampling_ctx
                if (greedy_sampling) {
                    const logits = ctx.getLogitsIth(batch.n_tokens - 1);
                    for (0..@intCast(n_vocab)) |token_id| candidates[token_id] = .{ .id = @intCast(token_id), .logit = logits[token_id], .p = 0.0 };
                    var candidates_p: TokenDataArray = .{ .data = candidates.ptr, .size = candidates.len, .sorted = false };
                    break :blk llama.c.llama_sample_token_greedy(ctx.cPtr(), &candidates_p);
                } else break :blk try sampling_ctx.sample(ctx, null, batch.n_tokens - 1);
            };

            try detokenizer.detokenizeWithSpecial(model, new_token);
            defer detokenizer.clearRetainingCapacity();
            std.debug.print("{s}", .{detokenizer.getText()});

            // is it an end of stream?
            if (new_token == model.tokenEos()) {
                std.debug.print("\nGeneration stopped: EOS token reached\n", .{});
                break;
            }

            if (n_cur >= args.max_len) {
                std.debug.print("\nGeneeration stopped: max_len reached\n", .{});
                break;
            }

            // prepare the next batch
            batch.clear();

            // push this new token for next evaluation
            sampling_ctx.accept(ctx, new_token, true);
            batch.add(new_token, n_cur, &.{0}, true);

            n_decoded += 1;
        }

        n_cur += 1;
    }

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
        //arg_utils.printHelp(Args);
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
