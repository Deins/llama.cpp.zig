const std = @import("std");
const llama = @import("llama.zig");
const Sampling = llama.Sampling;

const Tokenizer = llama.utils.Tokenizer;

const Token = llama.Token;
const TokenType = llama.TokenType;
const LogLevel = llama.LogLevel;
const SeqId = llama.SeqId;

const Prompt = @This();

/// what extenson method use to extend context when its full
pub const ContextExtensionStrategy = union(enum) {
    /// return error when context is full
    none: void,
    /// similar to shifting, but doesn't keep first n characters. Importantly keeps tokens in sync with kv cache.
    sliding_window: void,
    /// infinite text generation via context shifting
    /// - take the n_keep first tokens from the original prompt
    /// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
    shifting: struct {
        /// how many tokens to keep from beggining of kv cache, can be 0, to keep only last messages, or lenght of system prompt that you want to keep for example
        /// NOTE: must include (+1) bos token if one is used
        keep_first_n: usize = 0,
    },
    /// Self extend method: https://github.com/ggerganov/llama.cpp/discussions/4785
    self_extend: struct {},
};

/// utility to manage promt and keep with sync with kv cache & sampler for most basic use cases
pub const Options = struct {
    model: *llama.Model,
    ctx: *llama.Context,
    batch_size: usize = 1024,
    sampling_params: Sampling.Params = .{},
    n_seq_max: usize = 1,
    seq_id: SeqId = 0,
    ctx_extension: ContextExtensionStrategy = .{ .shifting = .{} },
};

model: *llama.Model,
ctx: *llama.Context,
batch: llama.Batch,
batch_size: usize,
sampling: Sampling,
seq_id: SeqId = 0,
ctx_extension: ContextExtensionStrategy,
/// any time tokens or indexes are read/modified, this mutex has to be locked
tokens_mutex: std.Thread.Mutex = .{},
/// prompt tokens
tokens: std.ArrayList(llama.Token),
/// index pointing one past last embedded character
embed_idx: usize = 0,
/// count, how many chaacters are embedded in kv cache
/// from my understanding this matches n_past in cpp samples
ctx_used: usize = 0,
/// one past sampling_ctx accepted token
accepted_idx: usize = 0,

pub fn init(alloc: std.mem.Allocator, opt: Options) !Prompt {
    var s = try Sampling.init(alloc, opt.sampling_params);
    errdefer s.deinit();
    var t = try std.ArrayList(Token).initCapacity(alloc, opt.batch_size);
    errdefer t.deinit();
    return .{
        .model = opt.model,
        .ctx = opt.ctx,
        .batch = llama.Batch.init(@intCast(opt.batch_size), 0, @intCast(opt.n_seq_max)),
        .batch_size = opt.batch_size,
        .sampling = s,
        .seq_id = opt.seq_id,
        .ctx_extension = opt.ctx_extension,
        .tokens = t,
    };
}

pub fn deinit(self: *@This()) void {
    self.batch.deinit();
    self.sampling.deinit();
    self.tokens.deinit();
}

pub fn clearRetainingCapacity(self: *@This()) void {
    self.invalidate();
    self.tokens_mutex.lock();
    defer self.tokens_mutex.unlock();
    self.tokens.clearRetainingCapacity();
}

pub fn appendText(self: *@This(), text: []const u8, special: bool) !void {
    self.tokens_mutex.lock();
    defer self.tokens_mutex.unlock();

    var t: Tokenizer = .{ .data = self.tokens }; // tokenize directly in our buffer
    try t.tokenize(self.model, text, false, special);
    self.tokens = t.data; // in case it grew, we must reasign it back
}

pub fn addOneToken(self: *@This(), token: Token) !void {
    self.tokens_mutex.lock();
    defer self.tokens_mutex.unlock();
    (try self.tokens.addOne()).* = token;
}

/// invalidate prompt forcing reprocess everything from beginning
pub fn invalidate(self: *@This()) void {
    self.ctx.kvCacheSeqRm(self.seq_id, 0, -1);
    self.accepted_idx = 0;
    self.embed_idx = 0;
    self.ctx_used = 0;
    self.sampling.reset();
}

/// Shrink prompt by removing last characters from back
pub fn shrink(self: *@This(), new_len: usize) void {
    self.tokens_mutex.lock();
    defer self.tokens_mutex.unlock();

    if (new_len == self.tokens.items.len) return;
    std.debug.assert(new_len < self.tokens.items.len);
    self.ctx.kvCacheSeqRm(self.seq_id, @intCast(new_len), @intCast(self.tokens.items.len));
    self.ctx_used -= self.tokens.items.len - new_len;
    self.embed_idx = @min(self.embed_idx, new_len);
    self.embed_idx -|= 1; // recompute last char
    const accepted_diff = self.accepted_idx - @min(self.accepted_idx, new_len);
    self.accepted_idx -= accepted_diff;
    self.sampling.prev.len -|= accepted_diff;
    self.tokens.shrinkRetainingCapacity(new_len);
}

/// Shrink by removing first characters from front
pub fn shrinkFront(self: *@This(), new_len: usize) void {
    self.tokens_mutex.lock();
    defer self.tokens_mutex.unlock();

    if (new_len == self.tokens.items.len) return;
    std.debug.assert(new_len < self.tokens.items.len);
    const tokens = self.tokens.items;
    const diff = tokens.len - new_len;
    self.ctx.kvCacheSeqRm(self.seq_id, @intCast(0), @intCast(new_len));
    self.ctx.kvCacheSeqShift(self.seq_id, @intCast(new_len), @intCast(tokens.len), -@as(i32, @intCast(diff)));
    std.mem.copyForwards(Token, tokens[0..new_len], tokens[diff..]);
    self.embed_idx -|= diff;
    self.ctx_used -|= diff;
    self.embed_idx -|= 1; // recompute last char
    self.ctx_used -|= 1; // recompute last char
    self.accepted_idx -|= diff;
    if (self.sampling.prev.len > new_len) self.sampling.prev.shrinkFront(new_len) else self.sampling.prev.clear();
    self.tokens.shrinkRetainingCapacity(new_len);
}

/// attempt to ensure context free context space
/// performs context extension if not enough context space is free
/// returns context space available
/// NOTE: can return less than requested if not enough space can be freed, or more if there is more space
pub fn ensureContextUnusedCapacty(self: *@This(), addtonal_count: usize) usize {
    const space_left = self.contextAvailable();
    const overflow = addtonal_count -| space_left;
    if (overflow <= 0) return space_left;
    switch (self.ctx_extension) {
        .none => return space_left,
        .sliding_window => {
            const shift = self.ctx_used / 2;
            self.shrinkFront(self.tokens.items.len - shift);
        },
        .shifting => |cfg| {
            const keep = cfg.keep_first_n;
            const left_n = self.ctx_used - keep;
            const discard_n = left_n / 2;
            if (discard_n <= 0) return self.contextAvailable();
            std.log.scoped(.prompt).info("context shift: keep {}; left: {}; discard: {};", .{ keep, left_n, discard_n });
            self.ctx.kvCacheSeqRm(self.seq_id, @intCast(keep), @intCast(keep + discard_n));
            self.ctx.kvCacheSeqShift(self.seq_id, @intCast(keep + discard_n), @intCast(self.ctx_used), -@as(i32, @intCast(discard_n)));
            self.ctx_used -= discard_n;
            // regenerate logits for last token
            self.embed_idx -|= 1;
        },
        .self_extend => @panic("TODO: implement"),
    }
    return self.contextAvailable();
}

pub fn generateAppendOne(self: *@This()) !Token {
    const tokens_len = blk: {
        self.tokens_mutex.lock();
        const tokens_len = self.tokens.items.len;
        self.tokens_mutex.unlock();
        if (self.embed_idx >= self.tokens.items.len) {
            const bos = self.model.tokenBos();
            try self.addOneToken(bos);
            return bos;
        }
        break :blk tokens_len;
    };

    {
        while (self.embed_idx < tokens_len) {
            const add_n = self.tokens.items.len - self.embed_idx;
            const ctx_space = self.ensureContextUnusedCapacty(add_n);
            const batch_len = @min(@min(ctx_space, add_n), self.batch_size);
            self.batch.clear();

            self.tokens_mutex.lock();
            defer self.tokens_mutex.unlock();
            for (0..batch_len) |p| {
                const i = self.embed_idx + p;
                self.batch.add(self.tokens.items[i], @intCast(self.ctx_used + p), &.{self.seq_id}, false);
            }
            self.embed_idx += batch_len;
            self.ctx_used += batch_len;
        }
    }

    // enable llama_decode to output logits only for the last token of the prompt
    self.batch.logits[@as(usize, @intCast(self.batch.n_tokens)) - 1] = true;

    // evaluate the current batch with the transformer model
    _ = self.ensureContextUnusedCapacty(1);
    try self.batch.decode(self.ctx);
    //  catch |err| switch (err) {
    //     error.NoKvSlotWarning => std.log.scoped(.prompt).warn("could not find a KV slot for the batch (try reducing the size of the batch or increase the context ", .{}),
    //     error.UnknownWarning => std.log.scoped(.prompt).warn("decoding warning: {}", .{err}),
    //     else => return err,
    // };

    self.tokens_mutex.lock();
    // update sampler with tokens that might have been added if any
    for (self.accepted_idx..self.tokens.items.len) |ai| self.sampling.accept(self.ctx, self.tokens.items[ai], false);
    self.accepted_idx = self.tokens.items.len;

    // sample for next token
    const new_token = try self.sampling.sample(self.ctx, null, self.batch.n_tokens - 1);
    self.tokens_mutex.unlock();
    try self.addOneToken(new_token);
    return new_token;
}

pub fn contextAvailable(self: *@This()) usize {
    return @as(usize, self.ctx.nCtx()) - self.ctx_used;
}
