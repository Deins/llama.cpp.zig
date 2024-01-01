const std = @import("std");
const llama = @import("llama.zig");
const Sampling = llama.Sampling;

const Tokenizer = llama.utils.Tokenizer;

const Token = llama.Token;
const TokenType = llama.TokenType;
const LogLevel = llama.LogLevel;
const SeqId = llama.SeqId;

const Prompt = @This();

/// utility to manage promt and keep with sync with kv cache & sampler for most basic use cases
pub const Options = struct {
    model: *llama.Model,
    ctx: *llama.Context,
    batch_size: usize = 1024,
    sampling_params: Sampling.Params = .{},
    n_seq_max: usize = 1,
    seq_id: SeqId = 0,
};

model: *llama.Model,
ctx: *llama.Context,
batch: llama.Batch,
batch_size: usize,
sampling: Sampling,
seq_id: SeqId = 0,
/// any time tokens or indexes are read/modified, this mutex has to be locked
tokens_mutex: std.Thread.Mutex = .{},
/// prompt tokens
tokens: std.ArrayList(llama.Token),
/// index pointing one past last embeded character
embed_idx: usize = 0,
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
    self.tokens_mutex.lock();
    defer self.tokens_mutex.unlock();

    self.ctx.kvCacheSeqRm(self.seq_id, 0, -1);
    self.accepted_idx = 0;
    self.embed_idx = 0;
    self.sampling.reset();
}

/// Shrink prompt by removing last characters from back
pub fn shrink(self: *@This(), new_len: usize) void {
    self.tokens_mutex.lock();
    defer self.tokens_mutex.unlock();

    if (new_len == self.tokens.items.len) return;
    std.debug.assert(new_len < self.tokens.items.len);
    self.ctx.kvCacheSeqRm(self.seq_id, @intCast(new_len), @intCast(self.tokens.items.len));
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
    self.ctx.kvCacheSeqRm(self.seq_id, @intCast(0), new_len);
    self.ctx.kvCacheSeqShift(self.seq_id, new_len, tokens.len, -diff);
    std.mem.copyForwards(Token, tokens[0..new_len], tokens[diff..]);
    self.embed_idx -|= diff;
    self.accepted_idx -| diff;
    self.sampling.prev.eraseFront();
    self.tokens.items.shrinkRetainingCapacity(new_len);
}

pub fn generateAppendOne(self: *@This()) !Token {
    if (self.embed_idx >= self.tokens.items.len) {
        const bos = self.model.tokenBos();
        try self.addOneToken(bos);
        return bos;
    }
    defer self.batch.clear();

    {
        self.tokens_mutex.lock();
        defer self.tokens_mutex.unlock();
        while (self.embed_idx < self.tokens.items.len) {
            const batch_len = @min(self.tokens.items.len - self.embed_idx, self.batch_size);
            self.batch.clear();
            for (self.embed_idx..self.embed_idx + batch_len) |i| self.batch.add(self.tokens.items[i], @intCast(i), &.{self.seq_id}, false);
            self.embed_idx += batch_len;
        }
    }

    // enable llama_decode to output logits only for the last token of the prompt
    self.batch.logits[@as(usize, @intCast(self.batch.n_tokens)) - 1] = true;

    // evaluate the current batch with the transformer model
    self.batch.decode(self.ctx) catch |err| switch (err) {
        error.NoKvSlotWarning => std.log.scoped(.prompt).warn("could not find a KV slot for the batch (try reducing the size of the batch or increase the context ", .{}),
        error.UnknownWarning => std.log.scoped(.prompt).warn("decoding warning: {}", .{err}),
        else => return err,
    };

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
