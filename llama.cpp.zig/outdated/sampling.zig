// this is a rewrite of original sampling.cpp as it had no c api
// NOTE: changes from cpp version:
// grammar is not passed as string to be parsed, instead parsed llama_grammar* params, as at time of writing grammar_parser has no C api.
// implements prev token as ring buffer
// reimplements llama_sample_repetition_penalties to support ring buffer

const std = @import("std");
const llama = @import("llama.zig");

const Token = llama.Token;
const TokenData = llama.TokenData;
const TokenDataArray = llama.TokenDataArray;
const Context = llama.Context;

// sampling parameters
pub const Params = struct {
    n_prev: i32 = 64, // number of previous tokens to remember
    n_probs: i32 = 0, // if greater than 0, output the probabilities of top n_probs tokens.
    top_k: i32 = 40, // <= 0 to use vocab size
    top_p: f32 = 0.95, // 1.0 = disabled
    min_p: f32 = 0.05, // 0.0 = disabled
    tfs_z: f32 = 1.00, // 1.0 = disabled
    typical_p: f32 = 1.00, // 1.0 = disabled
    temp: f32 = 0.80, // 1.0 = disabled
    penalty_last_n: i32 = 64, // last n tokens to penalize (0 = disable penalty, -1 = context size)
    penalty_repeat: f32 = 1.10, // 1.0 = disabled
    penalty_freq: f32 = 0.00, // 0.0 = disabled
    penalty_present: f32 = 0.00, // 0.0 = disabled
    mirostat: i32 = 0, // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    mirostat_tau: f32 = 5.00, // target entropy
    mirostat_eta: f32 = 0.10, // learning rate
    penalize_nl: bool = true, // consider newlines as a repeatable token
    samplers_sequence: []const SamplerType = &.{ .top_k, .tail_free, .typical_p, .top_p, .min_p, .temp }, // note: if allocated, user is responsible to free its contents

    grammar: ?*llama.Grammar = null,

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    cfg_negative_prompt: ?[]const u8 = null, // string to help guidance
    cfg_scale: f32 = 1.0, // how strong is guidance

    logit_bias: ?std.AutoArrayHashMap(Token, f32) = null, // logit bias for specific tokens
};

pub const SamplerType = enum {
    top_k,
    tail_free,
    typical_p,
    top_p,
    min_p,
    temp,

    pub fn fromChar(c: u8) ?SamplerType {
        return switch (c) {
            'k' => .top_k,
            'f' => .tail_free,
            'y' => .typical_p,
            'p' => .top_p,
            'm' => .min_p,
            't' => .temp,
            else => null,
        };
    }

    pub fn toChar(s: @This()) u8 {
        return switch (s) {
            .top_k => 'k',
            .tail_free => 'f',
            .typical_p => 'y',
            .top_p => 'p',
            .min_p => 'm',
            .temp => 't',
        };
    }

    pub fn fromString(str: []const u8, out_buff: []SamplerType, ignore_invalid: bool) ![]SamplerType {
        if (out_buff.len < str.len) return error.BufferTooSmall;
        var idx: usize = 0;
        for (str) |c| {
            out_buff[idx] = fromChar(c) orelse (if (ignore_invalid) continue else return error.InvalidChar);
            idx += 1;
        }
        return out_buff[0..idx];
    }

    pub fn toString(seq: []const u8, out_buff: []u8) ![]u8 {
        if (out_buff.len < out_buff.len) return error.BufferTooSmall;
        for (seq, 0..) |s, i| out_buff[i] = toChar(s);
        return out_buff;
    }
};

alloc: std.mem.Allocator,
// parameters that will be used for sampling
params: Params,

// mirostat SamplerType state
mirostat_mu: f32 = 0,

prev: llama.utils.TokenRingBuffer,
cur: std.ArrayListUnmanaged(TokenData),

pub fn init(alloc: std.mem.Allocator, params: Params) !@This() {
    return .{
        .alloc = alloc,
        .params = params,
        .prev = .{ .data = try alloc.alloc(Token, @intCast(params.n_prev)) },
        .cur = .{},
    };
}

pub fn deinit(self: *@This()) void {
    self.alloc.free(self.prev.data);
    self.cur.deinit(self.alloc);
}

pub fn reset(self: *@This()) void {
    self.cur.clearRetainingCapacity();
    self.prev.clear();
}

// this is a common sampling function used across the examples for convenience
// it can serve as a starting point for implementing your own sampling function
// Note: When using multiple sequences, it is the caller's responsibility to call
//       llama_sampling_reset when a sequence ends
//
// required:
//  - ctx_main:     context to use for sampling
//  - self: sampling-specific context
//
// optional:
//  - ctx_cfg:      context to use for classifier-free guidance
//  - idx:          sample from llama_get_logits_ith(ctx, idx) or 0
//
// returns:
//  - token:      sampled token
//  - candidates: vector of candidate tokens
//
pub fn sample(self: *@This(), ctx_main: *llama.Context, ctx_cfg: ?*llama.Context, idx: i32) !Token {
    const params = self.params;

    const model = ctx_main.getModel();
    const n_vocab = model.nVocab();

    const temp = params.temp;
    const penalty_last_n = if (params.penalty_last_n < 0) params.n_prev else params.penalty_last_n;
    const penalty_repeat = params.penalty_repeat;
    const penalty_freq = params.penalty_freq;
    const penalty_present = params.penalty_present;
    const mirostat = params.mirostat;
    const mirostat_tau = params.mirostat_tau;
    const mirostat_eta = params.mirostat_eta;
    const penalize_nl = params.penalize_nl;

    var id: Token = 0;

    const logits: [*]f32 = ctx_main.getLogitsIth(idx);

    // apply params.logit_bias map
    if (params.logit_bias) |bias| {
        var it = bias.iterator();
        while (it.next()) |e| logits[@intCast(e.key_ptr.*)] += e.value_ptr.*;
    }

    self.cur.clearRetainingCapacity();

    for (0..@intCast(n_vocab)) |token_id| try self.cur.append(self.alloc, .{ .id = @intCast(token_id), .logit = logits[token_id], .p = 0 });

    var cur_p: TokenDataArray = .{ .data = self.cur.items.ptr, .size = self.cur.items.len, .sorted = false };

    if (ctx_cfg) |c| ctx_main.sampleClassifierFreeGuidance(&cur_p, c, params.cfg_scale);

    // apply penalties
    if (self.prev.len > 0) {
        const nl_logit = logits[@intCast(model.tokenNl())];

        const pparams: RepetitionPenaltyParams = .{
            .penalty_last_n = @intCast(penalty_last_n),
            .penalty_repeat = penalty_repeat,
            .penalty_freq = penalty_freq,
            .penalty_present = penalty_present,
        };
        if (!pparams.isDisabled()) {
            var p = try RepetitionPenalizer.init(self.alloc, pparams);
            defer p.deinit();
            const n = @min(self.prev.len, pparams.penalty_last_n);
            const begin = @as(isize, @bitCast(self.prev.len)) - n;
            const end = begin + n;
            const slices = self.prev.slices(@intCast(begin), @intCast(end));
            for (slices[0]) |tok| p.addLastToken(tok);
            for (slices[1]) |tok| p.addLastToken(tok);
            p.sampleTokenDataArray(&cur_p);
        }

        if (!penalize_nl) {
            for (0..cur_p.size) |i| {
                if (cur_p.data[i].id == model.tokenNl()) {
                    cur_p.data[i].logit = nl_logit;
                    break;
                }
            }
        }
    }

    if (self.params.grammar) |g| ctx_main.sampleGrammar(&cur_p, g);

    if (temp < 0.0) {
        // greedy sampling, with probs
        ctx_main.sampleSoftmax(&cur_p);
        id = cur_p.data[0].id;
    } else if (temp == 0.0) {
        // greedy sampling, no probs
        id = ctx_main.sampleTokenGreedy(&cur_p);
    } else {
        if (mirostat == 1) {
            const mirostat_m = 100;
            ctx_main.sampleTemp(&cur_p, temp);
            id = ctx_main.sampleTokenMirostat(&cur_p, mirostat_tau, mirostat_eta, mirostat_m, &self.mirostat_mu);
        } else if (mirostat == 2) {
            ctx_main.sampleTemp(&cur_p, temp);
            id = ctx_main.sampleTokenMirostatV2(&cur_p, mirostat_tau, mirostat_eta, &self.mirostat_mu);
        } else {
            // temperature sampling
            const min_keep: usize = @max(params.n_probs, 1);
            samplerQueue(ctx_main, params, &cur_p, min_keep);
            id = ctx_main.sampleToken(&cur_p);
            // LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx_main, id).c_str());
        }
    }

    return id;
}

pub fn accept(self: *@This(), ctx_main: *llama.Context, id: Token, apply_grammar: bool) void {
    self.prev.appendEraseFifo(id);

    if (apply_grammar and self.params.grammar != null)
        ctx_main.grammarAcceptToken(@ptrCast(self.params.grammar.?), id);
}

// no reasons to expose this function in header
fn samplerQueue(ctx_main: *llama.Context, params: Params, cur_p: *TokenDataArray, min_keep: usize) void {
    const n_vocab = ctx_main.getModel().nVocab();

    const temp = params.temp;
    const top_k = if (params.top_k <= 0) n_vocab else params.top_k;
    const top_p = params.top_p;
    const min_p = params.min_p;
    const tfs_z = params.tfs_z;
    const typical_p = params.typical_p;

    for (params.samplers_sequence) |s| switch (s) {
        .top_k => ctx_main.sampleTopK(cur_p, top_k, min_keep),
        .tail_free => ctx_main.sampleTailFree(cur_p, tfs_z, min_keep),
        .typical_p => ctx_main.sampleTypical(cur_p, typical_p, min_keep),
        .top_p => ctx_main.sampleTopP(cur_p, top_p, min_keep),
        .min_p => ctx_main.sampleMinP(cur_p, min_p, min_keep),
        .temp => ctx_main.sampleTemp(cur_p, temp),
    };
}

pub const RepetitionPenaltyParams = struct {
    penalty_last_n: u32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,

    /// returns true if repetition penalties are disabled
    pub inline fn isDisabled(self: RepetitionPenaltyParams) bool {
        return self.penalty_last_n == 0 or (self.penalty_repeat == 1 and self.penalty_freq == 0 and self.penalty_present == 0);
    }
};

/// implementation of llama_sample_repetition_penalties that allows directly insert previous tokens from any source
/// usage:
///     init()
///     for (penalty_last_n) tokens addLastToken()
///     sample()
pub const RepetitionPenalizer = struct {
    token_count: std.AutoHashMap(Token, u32), // ensures total capacity on init
    params: RepetitionPenaltyParams,

    pub fn init(alloc: std.mem.Allocator, params: RepetitionPenaltyParams) !RepetitionPenalizer {
        var hm = std.AutoHashMap(Token, u32).init(alloc);
        if (!params.isDisabled()) try hm.ensureTotalCapacity(params.penalty_last_n);
        return .{
            .token_count = hm,
            .params = params,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.token_count.deinit();
    }

    /// NOTE: asserts that no more than penalty_last_n tokens are inserted
    pub fn addLastToken(self: *@This(), token: Token) void {
        std.debug.assert(self.token_count.count() < self.params.penalty_last_n);
        const res = self.token_count.getOrPutAssumeCapacity(token);
        res.value_ptr.* *= @intFromBool(res.found_existing); // initialize to 0 value if its newly created value
        res.value_ptr.* += 1;
    }

    pub fn sampleTokenDataArray(self: *@This(), candidates: *TokenDataArray) void {
        self.sample(candidates.data[0..candidates.size]);
        candidates.sorted = false;
    }

    /// WARNING: modifies logit values so array might not be sorted after this. If passing from TokenDataArray use samepleTokenDataArray to invalidate sorted flag
    pub fn sample(self: *@This(), candidates: []TokenData) void {
        const penalty_repeat = self.params.penalty_repeat;
        const penalty_freq = self.params.penalty_freq;
        const penalty_present = self.params.penalty_present;
        for (0..candidates.len) |i| {
            // Apply frequency and presence penalties to the candidates
            if (self.token_count.get(candidates[i].id)) |count| {
                // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
                // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
                if (candidates[i].logit <= 0) candidates[i].logit *= penalty_repeat else candidates[i].logit /= penalty_repeat;
                const count_not_empty_f = @as(f32, @floatFromInt(@intFromBool(count > 0)));
                candidates[i].logit -= @as(f32, @floatFromInt(count)) * penalty_freq + count_not_empty_f * penalty_present;
            }
        }
        // if (ctx)  ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
};
