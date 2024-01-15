pub const c = @import("llama.h");
pub const std = @import("std"); // mem.span and other utils
// utilities
pub const options = @import("llama_options");
pub const Prompt = @import("prompt.zig");
pub const Sampling = @import("sampling.zig");
pub const utils = @import("utils.zig");
pub const Tokenizer = utils.Tokenizer;
pub const Detokenizer = utils.Detokenizer;

pub const opencl_utils = if (options.opencl) @import("opencl_utils.zig");

//
//  Actual llama.h bindings
//

//pub const Model = c.llama_model;
//pub const Context = c.llama_context;

pub const Pos = c.llama_pos;
pub const Token = c.llama_token;
pub const SeqId = c.llama_seq_id;

pub const VocabType = c.llama_vocab_type;
//pub const TokenType = c.llama_token_type;
pub const FType = c.llama_ftype;
pub const RopeScalingType = c.llama_rope_scaling_type;

pub const TokenData = c.llama_token_data;
pub const TokenDataArray = c.llama_token_data_array;
pub const LlamaContext = c.llama_context;

pub const progress_callback = c.llama_progress_callback;

//pub const Batch = c.llama_batch;
pub const ModelKvOverrideType = c.llama_model_kv_override_type;
pub const ModelKvOverride = c.llama_model_kv_override;
pub const ModelQuantizeParams = c.llama_model_quantize_params;
//pub const Grammar = c.llama_grammar;
pub const Gretpye = c.llama_gretype;
pub const GrammarElement = c.llama_grammar_element;
pub const Timings = c.llama_timings;

// zigified opaque, structs, enums

pub const TokenType = enum(c_int) {
    LLAMA_TOKEN_TYPE_UNDEFINED = 0,
    LLAMA_TOKEN_TYPE_NORMAL = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN = 2,
    LLAMA_TOKEN_TYPE_CONTROL = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED = 5,
    LLAMA_TOKEN_TYPE_BYTE = 6,
};

pub const Backend = opaque {
    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    pub fn init(numa: bool) void {
        c.llama_backend_init(numa);
    }

    pub fn deinit() void {
        c.llama_backend_free();
    }
};

pub const Model = extern opaque {
    pub const Params = c.llama_model_params;
    pub const defaultParams = c.llama_model_default_params;

    pub inline fn initFromFile(path_to_file: CStr, params: Params) !*Model {
        const ptr = c.llama_load_model_from_file(path_to_file, params);
        if (ptr == null) return error.FailedToLoadModel;
        return @ptrCast(ptr);
    }

    pub inline fn deinit(self: *@This()) void {
        c.llama_free_model(@ptrCast(self));
    }

    pub inline fn vocabType(self: *const @This()) VocabType {
        return (c.llama_vocab_type(self));
    }
    pub inline fn nVocab(self: *const @This()) i32 {
        return (c.llama_n_vocab(self.cCPtr()));
    }
    pub inline fn nCtxTrain(self: *const @This()) i32 {
        return (c.llama_n_ctx_train(self.cCPtr()));
    }
    pub inline fn nEmbd(self: *const @This()) i32 {
        return (c.llama_n_embd(self.cCPtr()));
    }
    // Get the model's RoPE frequency scaling factor
    pub inline fn ropeFreqScaleTrain(self: *const @This()) f32 {
        return c.llama_rope_freq_scale_train(self.cCPtr());
    }

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure ==> ZIG bindings return slice or null
    // - The output string is always null-terminated and cleared on failure
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    pub inline fn metaValStr(self: *const @This(), key: CStr, out_buff: []u8) ?[]u8 {
        const idx = c.llama_model_meta_val_str(self.cCPtr(), key, out_buff.ptr, out_buff.len);
        return if (idx >= 0) out_buff[0..idx] else null;
    }

    // Get the number of metadata key/value pairs
    pub inline fn metaCount(self: *const @This()) usize {
        return @intCast(c.llama_model_meta_count(self.cCPtr()));
    }

    // Get metadata key name by index
    pub inline fn metaKeyByIndex(self: *const @This(), i: usize, out_buff: []u8) ?[]u8 {
        const idx = c.llama_model_meta_key_by_index(self.cCPtr(), @intCast(i), out_buff.ptr, out_buff.len);
        return if (idx >= 0) out_buff[0..@intCast(idx)] else null;
    }

    // Get metadata value as a string by index
    pub inline fn metaValStrByIndex(self: *const @This(), i: usize, out_buff: []u8) ?[]u8 {
        const idx = c.llama_model_meta_val_str_by_index(self.cCPtr(), @intCast(i), out_buff.ptr, out_buff.len);
        return if (idx >= 0) out_buff[0..@intCast(idx)] else null;
    }

    // Get a string describing the model type
    pub inline fn desc(self: *const @This(), out_buff: []u8) []u8 {
        const idx: usize = @intCast(c.llama_model_desc(self.cCPtr(), out_buff.ptr, out_buff.len));
        return if (idx >= 0) out_buff[0..idx] else null;
    }

    // Returns the total size of all the tensors in the model in bytes
    pub inline fn modelSize(self: *const @This()) usize {
        return c.llama_model_size(self.cCPtr());
    }

    // Returns the total number of parameters in the model
    pub inline fn mParams(self: *const @This()) usize {
        return c.llama_model_n_params(self.cCPtr());
    }

    // Get a llama model tensor
    pub inline fn getTensor(self: *const @This(), name: CStr) ?*c.ggml_tensor {
        return c.llama_get_model_tensor(self.cCPtr(), name);
    }

    pub inline fn modelQuantize(self: *const @This(), fname_input: CStr, fname_output: CStr, params: ModelQuantizeParams) !void {
        return if (c.llama_model_quantize(self.cCPtr(), fname_input, fname_output, params) == 0) void else error.ERROR;
    }

    // Apply a LoRA adapter to a loaded model
    // path_base_model is the path to a higher quality model to use as a base for
    // the layers modified by the adapter. Can be NULL to use the current loaded model.
    // The model needs to be reloaded before applying a new adapter, otherwise the adapter
    // will be applied on top of the previous one
    pub inline fn applyLoraFromFile(self: *const @This(), path_lora: CStr, scale: f32, path_base_model: ?CStr, n_threads: usize) !void {
        if (c.llama_model_apply_lora_from_file(self.cCPtr(), path_lora, scale, path_base_model, @intCast(n_threads)) != 0) return error.LORA_ERROR;
    }

    //
    // Vocab
    //

    pub inline fn tokenGetText(self: *const @This(), token: Token) CStr {
        return c.llama_token_get_text(self.cCPtr(), token);
    }
    pub inline fn tokenGetTextSlice(self: *const @This(), token: Token) [:0]const u8 {
        return std.mem.span(tokenGetText(self, token));
    }
    pub inline fn tokenGetScore(self: *const @This(), token: Token) f32 {
        return c.llama_token_get_score(self.cCPtr(), token);
    }
    pub inline fn tokenGetType(self: *const @This(), token: Token) TokenType {
        return @enumFromInt(c.llama_token_get_type(self.cCPtr(), token));
    }

    // Special tokens
    pub inline fn tokenBos(self: *const @This()) Token {
        return c.llama_token_bos(self.cCPtr());
    }
    pub inline fn tokenEos(self: *const @This()) Token {
        return c.llama_token_eos(self.cCPtr());
    }
    pub inline fn tokenNl(self: *const @This()) Token {
        return c.llama_token_nl(self.cCPtr());
    }

    /// gets state => Returns null for unknown, or true/false.
    pub inline fn addBosToken(self: *const @This()) ?bool {
        const ret = c.llama_add_bos_token(self.cCPtr());
        if (ret < 0) return null;
        return ret > 0;
    }

    /// gets state => Returns null for unknown, or true/false.
    pub inline fn llama_add_eos_token(self: *const @This()) ?bool {
        const ret = llama_add_eos_token(self.cCPtr());
        if (ret < 0) return null;
        return ret > 0;
    }

    // codellama infill tokens
    pub inline fn tokenPrefix(self: *const @This()) Token {
        return c.llama_token_prefix(self.cCPtr());
    } // Beginning of infill prefix
    pub inline fn tokenMiddle(self: *const @This()) Token {
        return c.llama_token_middle(self.cCPtr());
    } // Beginning of infill middle
    pub inline fn tokenSuffix(self: *const @This()) Token {
        return c.llama_token_suffix(self.cCPtr());
    } // Beginning of infill suffix
    pub inline fn tokenEot(self: *const @This()) Token {
        return c.llama_token_eot(self.cCPtr());
    } // End of infill middle

    //
    // Tokenization
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_max_tokens
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @param special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.
    ///                Does not insert a leading space.
    pub inline fn tokenize(model: *const @This(), text: []const u8, out_tokens: []Token, add_bos: bool, special: bool) i32 {
        return c.llama_tokenize(model.cCPtr(), text.ptr, @intCast(text.len), out_tokens.ptr, @intCast(out_tokens.len), add_bos, special);
    }

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.
    pub inline fn tokenToPiece(self: *const @This(), token: Token, out_text: []u8) i32 {
        return c.llama_token_to_piece(self.cCPtr(), token, out_text.ptr, @intCast(out_text.len));
    }

    pub inline fn cCPtr(self: *const Model) *const c.llama_model {
        return @ptrCast(self);
    }
};

pub const Context = opaque {
    pub const Params = c.llama_context_params;
    pub const defaultParams = c.llama_context_default_params;

    pub inline fn initWithModel(model: *Model, params: Params) !*Context {
        const ptr = c.llama_new_context_with_model(@ptrCast(model), params) orelse return error.ContextCreationFailed;
        return @ptrCast(ptr);
    }

    pub inline fn deinit(self: *@This()) void {
        c.llama_free(self.cPtr());
    }

    pub inline fn getModel(self: *const @This()) *const Model {
        return @ptrCast(c.llama_get_model(self.cCPtr()));
    }

    pub inline fn nCtx(self: *const @This()) u32 {
        return c.llama_n_ctx(self.cCPtr());
    }

    //
    // KV cache - see also KvCacheView
    //

    /// Returns the number of tokens in the KV cache (slow, use only for debug)
    /// If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    pub inline fn getKvCacheTokenCount(self: *const @This()) i32 {
        return c.llama_get_kv_cache_token_count(self.cCPtr());
    }

    /// Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    pub inline fn getKvCacheUsedCells(self: *const @This()) i32 {
        return c.llama_get_kv_cache_used_cells(self.cCPtr());
    }

    /// Clear the KV cache
    pub inline fn kvCacheClear(self: *@This()) void {
        return c.llama_kv_cache_clear(self.cPtr());
    }

    /// Integer division of the positions by factor of `d > 1`
    /// If the KV cache is RoPEd, the KV data is updated accordingly
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    pub inline fn kvCacheSeqDiv(self: *@This(), seq_id: SeqId, p0: Pos, p1: Pos, d: c_int) void {
        c.llama_kv_cache_seq_div(self.cPtr(), seq_id, p0, p1, d);
    }

    /// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    /// seq_id < 0 : match any sequence
    /// p0 < 0     : [0,  p1]
    /// p1 < 0     : [p0, inf)
    pub inline fn kvCacheSeqRm(self: *@This(), seq_id: SeqId, p0: Pos, p1: Pos) void {
        return c.llama_kv_cache_seq_rm(self.cPtr(), seq_id, p0, p1);
    }

    /// Copy all tokens that belong to the specified sequence to another sequence
    /// Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    pub inline fn kvCacheSeqCp(self: *@This(), seq_id_src: SeqId, seq_id_dst: SeqId, p0: Pos, p1: Pos) void {
        return c.llama_kv_cache_seq_cp(self.cPtr(), seq_id_src, seq_id_dst, p0, p1);
    }

    /// Removes all tokens that do not belong to the specified sequence
    pub inline fn kvCacheSeqKeep(self: *@This(), seq_id: SeqId) void {
        return c.llama_kv_cache_seq_keep(self.cPtr(), seq_id);
    }

    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    /// If the KV cache is RoPEd, the KV data is updated accordingly
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    pub inline fn kvCacheSeqShift(self: *@This(), seq_id: SeqId, p0: Pos, p1: Pos, delta: Pos) void {
        return c.llama_kv_cache_seq_shift(self.cPtr(), seq_id, p0, p1, delta);
    }

    //
    // State / sessions
    //

    /// Returns the maximum size in bytes of the state (rng, logits, embedding
    /// and kv_cache) - will often be smaller after compacting tokens
    pub inline fn getStateSize(self: *@This()) usize {
        return c.llama_get_state_size(self.cCPtr());
    }

    /// Copies the state to the specified destination address.
    /// Destination needs to have allocated enough memory.
    /// Returns slice of dest, of actual bytes that was written
    pub inline fn copyStateData(self: *const @This(), dst: [*]u8) []u8 {
        return dst[0..c.llama_copy_state_data(self.cCPtr(), dst)];
    }

    /// Set the state reading from the specified address
    /// Returns the number of bytes read
    pub inline fn setStateData(self: *const @This(), src: [*]u8) usize {
        _ = src;
        return c.llama_set_state_data(self.cCPtr());
    }

    /// Save/load session file
    pub inline fn loadSessionFile(self: *const @This(), path_session: CStr, tokens_out: []Token) error.CANT_LOAD_SESSION![]Token {
        var read_len: usize = undefined;
        if (!c.llama_load_session_file(self.cCPtr(), path_session, tokens_out.ptr, tokens_out.len, &read_len)) return error.CANT_LOAD_SESSION;
        return tokens_out[0..read_len];
    }

    pub inline fn saveSessionFile(self: *const @This(), path_session: CStr, tokens: []const Token) error.CANT_SAVE_SESSION!void {
        if (!c.llama_save_session_file(self.cCPtr(), path_session, tokens.ptr, tokens.len)) return error.CANT_SAVE_SESSION;
    }

    //
    // Decoding
    //

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    pub inline fn setNThreads(self: *const @This(), n_threads: u32, n_threads_batch: u32) void {
        c.llama_set_n_threads(self.cCPtr(), (n_threads), (n_threads_batch));
    }

    // Token logits obtained from the last call to llama_eval()
    // The logits for the last token are stored in the last row
    // Logits for which llama_batch.logits[i] == 0 are undefined
    // Rows: n_tokens provided with llama_batch
    // Cols: n_vocab
    pub inline fn getLogits(self: *@This()) [*]f32 {
        return c.llama_get_logits(self.cPtr());
    }

    // Logits for the ith token. Equivalent to:
    // llama_get_logits(ctx) + i*n_vocab
    pub inline fn getLogitsIth(self: *@This(), i: i32) [*]f32 {
        return c.llama_get_logits_ith(self.cPtr(), i);
    }

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    pub inline fn llama_get_embeddings(self: *@This()) [*]f32 {
        return c.llama_get_embeddings(self.cPtr());
    }

    // Performance information
    pub inline fn getTimings(self: *const Context) Timings {
        return c.llama_get_timings(self.cCPtr());
    }

    pub inline fn printTimings(self: *Context) void {
        c.llama_print_timings(self.cPtr());
    }
    pub inline fn resetTimings(self: *Context) void {
        c.llama_reset_timings(self.cPtr());
    }

    //
    // Sampling functions
    //

    // Sets the current rng seed.
    pub inline fn setRngSeed(self: *@This(), seed: u32) void {
        c.llama_set_rng_seed(self.cPtr(), seed);
    }

    /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    pub inline fn sampleRepetitionPenalties(self: *@This(), candidates: *TokenDataArray, last_tokens: [*]const Token, penalty_last_n: usize, penalty_repeat: f32, penalty_freq: f32, penalty_present: f32) void {
        c.llama_sample_repetition_penalties(self.cPtr(), candidates, last_tokens, penalty_last_n, penalty_repeat, penalty_freq, penalty_present);
    }

    /// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
    /// @params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
    /// @params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
    pub inline fn sampleClassifierFreeGuidance(self: *@This(), candidates: *TokenDataArray, guidance_ctx: *Context, scale: f32) void {
        c.llama_sample_classifier_free_guidance(self.cPtr(), candidates, guidance_ctx.cPtr(), scale);
    }

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    pub inline fn sampleSoftmax(self: *@This(), candidates: *TokenDataArray) void {
        c.llama_sample_softmax(self.cPtr(), candidates);
    }

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    pub inline fn sampleTopK(self: *@This(), candidates: *TokenDataArray, k: i32, min_keep: usize) void {
        c.llama_sample_top_k(self.cPtr(), candidates, k, min_keep);
    }

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    pub inline fn sampleTopP(self: *@This(), candidates: *TokenDataArray, p: f32, min_keep: usize) void {
        c.llama_sample_top_p(self.cPtr(), candidates, p, min_keep);
    }

    /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
    pub inline fn sampleMinP(self: *@This(), candidates: *TokenDataArray, p: f32, min_keep: usize) void {
        c.llama_sample_min_p(self.cPtr(), candidates, p, min_keep);
    }

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    pub inline fn sampleTailFree(self: *@This(), candidates: *TokenDataArray, z: f32, min_keep: usize) void {
        c.llama_sample_tail_free(self.cPtr(), candidates, z, min_keep);
    }

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    pub inline fn sampleTypical(self: *@This(), candidates: *TokenDataArray, p: f32, min_keep: usize) void {
        c.llama_sample_typical(self.cPtr(), candidates, p, min_keep);
    }

    pub inline fn sampleTemp(self: *@This(), candidates: *TokenDataArray, temp: f32) void {
        c.llama_sample_temp(self.cPtr(), candidates, temp);
    }

    /// @details Apply constraints from grammar
    pub inline fn sampleGrammar(self: *@This(), candidates: *TokenDataArray, grammar: *const Grammar) void {
        c.llama_sample_grammar(self.cPtr(), candidates, @ptrCast(grammar));
    }

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    pub inline fn sampleTokenMirostat(self: *@This(), candidates: *TokenDataArray, tau: f32, eta: f32, m: i32, mu: *f32) Token {
        return c.llama_sample_token_mirostat(self.cPtr(), candidates, tau, eta, m, mu);
    }

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    pub inline fn sampleTokenMirostatV2(self: *@This(), candidates: *TokenDataArray, tau: f32, eta: f32, mu: *f32) Token {
        return c.llama_sample_token_mirostat_v2(self.cPtr(), candidates, tau, eta, mu);
    }

    /// @details Selects the token with the highest probability.
    ///          Does not compute the token probabilities. Use llama_sample_softmax() instead.
    pub inline fn sampleTokenGreedy(self: *@This(), candidates: *TokenDataArray) Token {
        return c.llama_sample_token_greedy(self.cPtr(), candidates);
    }

    /// @details Randomly selects a token from the candidates based on their probabilities.
    pub inline fn sampleToken(self: *@This(), candidates: *TokenDataArray) Token {
        return c.llama_sample_token(self.cPtr(), candidates);
    }

    /// @details Accepts the sampled token into the grammar
    pub inline fn grammarAcceptToken(self: *@This(), grammar: *Grammar, tok: Token) void {
        c.llama_grammar_accept_token(self.cPtr(), @ptrCast(grammar), tok);
    }

    //
    // Utils for binding betwenn translate-c pointers and opaque
    pub inline fn cCPtr(self: *const Context) *const c.llama_context {
        return @ptrCast(self);
    }

    pub inline fn cPtr(self: *Context) *c.llama_context {
        return @ptrCast(self);
    }
};

// Information associated with an individual cell in the KV cache view.
pub const KvCacheViewCell = extern struct {
    // The position for this cell. Takes KV cache shifts into account.
    // May be negative if the cell is not populated.
    pos: Pos,
};
comptime {
    if (@sizeOf(KvCacheViewCell) != @sizeOf(c.llama_kv_cache_view_cell)) unreachable;
}

// An updateable view of the KV cache.
pub const KvCacheView = extern struct {
    // Number of KV cache cells. This will be the same as the context size.
    n_cells: i32,

    // Maximum number of sequences that can exist in a cell. It's not an error
    // if there are more sequences in a cell than this value, however they will
    // not be visible in the view cells_sequences.
    n_max_seq: i32,

    // Number of tokens in the cache. For example, if there are two populated
    // cells, the first with 1 sequence id in it and the second with 2 sequence
    // ids then you'll have 3 tokens.
    token_count: i32,

    // Number of populated cache cells.
    used_cells: i32,

    // Maximum contiguous empty slots in the cache.
    max_contiguous: i32,

    // Index to the start of the max_contiguous slot range. Can be negative
    // when cache is full.
    max_contiguous_idx: i32,

    // Information for an individual cell.
    cells: ?[*]KvCacheViewCell,

    // The sequences for each cell. There will be n_max_seq items per cell.
    cells_sequences: ?[*]SeqId,

    // Create an empty KV cache view. (use only for debugging purposes)
    pub fn init(ctx: *Context, n_max_seq: i32) KvCacheView {
        return c.llama_kv_cache_view_init(ctx, n_max_seq);
    }

    // Free a KV cache view. (use only for debugging purposes)
    pub fn deinit(self: *const @This()) void {
        c.llama_kv_cache_view_free(self.cCPtr());
    }

    // Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
    pub fn viewUpdate(self: @This(), ctx: *Context) void {
        _ = ctx;
        c.llama_kv_cache_view_update(self.cCPtr());
    }
};
comptime {
    if (@sizeOf(KvCacheView) != @sizeOf(c.llama_kv_cache_view)) unreachable;
}

// functions

// Helpers for getting default parameters
pub const timeUs = c.llama_time_us;

pub const maxDevices = c.llama_max_devices;
pub const mmapSupported = c.llama_mmap_supported;
pub const mlockSupported = c.llama_mlock_supported;

//
// Decoding
//

// Run the llama inference to obtain the logits and probabilities for the next token(s).
// tokens + n_tokens is the provided batch of new tokens to process
// n_past is the number of tokens to use from previous eval calls
// Returns 0 on success
// DEPRECATED: use llama_decode() instead
// LLAMA_API DEPRECATED(int llama_eval(
//         struct llama_context * ctx,
//                  llama_token * tokens,
//                      int32_t   n_tokens,
//                          int   n_past),
//         "use llama_decode() instead");

// // Same as llama_eval, but use float matrix input directly.
// // DEPRECATED: use llama_decode() instead
// LLAMA_API DEPRECATED(int llama_eval_embd(
//         struct llama_context * ctx,
//                        float * embd,
//                      int32_t   n_tokens,
//                          int   n_past),
//         "use llama_decode() instead");

/// Decoding batch
/// TODO: review which pointers are optional and one vs many
pub const Batch = extern struct {
    n_tokens: i32,

    token: [*]Token,
    embd: ?[*]f32,
    pos: [*]Pos,
    n_seq_id: [*]i32,
    seq_id: [*][*]SeqId,
    logits: [*]bool,

    // NOTE: helpers for smooth API transition - can be deprecated in the future
    //       for future-proof code, use the above fields instead and ignore everything below
    //
    // pos[i] = all_pos_0 + i*all_pos_1
    //
    all_pos_0: Pos, // used if pos == NULL
    llama_pos: Pos, // used if pos == NULL
    all_seq_id: SeqId, // used if seq_id == NULL

    // Return batch for single sequence of tokens starting at pos_0
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    pub fn initOne(tokens: []Token, pos_0: Pos, seq_id: SeqId) Batch {
        return @bitCast(c.llama_batch_get_one(tokens.ptr, @intCast(tokens.len), pos_0, seq_id));
    }

    /// Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    /// Each token can be assigned up to n_seq_max sequence ids
    /// The batch has to be freed with llama_batch_free()
    /// If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    /// Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    /// The rest of the llama_batch members are allocated with size n_tokens
    /// All members are left uninitialized
    pub fn init(n_tokens: i32, embed: i32, n_seq_max: i32) Batch {
        return @bitCast(c.llama_batch_init(n_tokens, embed, n_seq_max));
    }

    pub fn deinit(self: @This()) void {
        c.llama_batch_free(@bitCast(self));
    }

    /// Positive return values does not mean a fatal error, but rather a warning.
    ///   error.NoKvSlot - just a warning, can could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    pub inline fn decode(self: @This(), ctx: *Context) error{ DecodeError, NoKvSlotWarning, UnknownWarning }!void {
        const status = c.llama_decode(ctx.cPtr(), @bitCast(self));
        if (status > 0) {
            // Positive return values does not mean a fatal error, but rather a warning.
            if (status == 1) return error.NoKvSlotWarning;
            return error.UnknownWarning;
        }
        if (status < 0) return error.DecodeError;
    }

    //  Functions from common.h
    //
    pub fn add(self: *Batch, token: Token, pos: Pos, seq_ids: []const SeqId, logits: bool) void {
        const tail_idx: usize = @intCast(self.n_tokens);
        self.token[tail_idx] = @intCast(token);
        self.pos[tail_idx] = pos;
        self.n_seq_id[tail_idx] = @intCast(seq_ids.len);
        for (seq_ids, 0..) |sid, i| self.seq_id[tail_idx][i] = sid;
        self.logits[tail_idx] = logits;
        self.n_tokens += 1;
    }

    pub fn clear(self: *Batch) void {
        self.n_tokens = 0;
    }
};
comptime {
    if (@sizeOf(Batch) != @sizeOf(c.llama_batch)) unreachable;
}

//
// Grammar
//
pub const Grammar = opaque {
    pub fn init(rules: [][*]const GrammarElement, start_rule_index: usize) *Grammar {
        return c.llama_grammar_init(rules.ptr, rules.len, start_rule_index);
    }

    pub fn deinit(self: @This()) void {
        c.llama_grammar_free(self.cCPtr());
    }

    pub fn copy(self: @This()) *Grammar {
        return c.llama_grammar_free(self.cCPtr());
    }
};

//
// Beam search
//

pub const BeamView = c.llama_beam_view;

// Passed to beam_search_callback function.
// Whenever 0 < common_prefix_length, this number of tokens should be copied from any of the beams
// (e.g. beams[0]) as they will be removed (shifted) from all beams in all subsequent callbacks.
// These pointers are valid only during the synchronous callback, so should not be saved.
pub const BeamsState = c.llama_beams_state;

// Type of pointer to the beam_search_callback function.
// void* callback_data is any custom data passed to llama_beam_search, that is subsequently
// passed back to beam_search_callback. This avoids having to use global variables in the callback.
pub const BeamSearchCallback = *const fn (callback_data: ?*anyopaque, state: BeamsState) callconv(.C) void;

/// @details Deterministically returns entire sentence constructed by a beam search.
/// @param ctx Pointer to the llama_context.
/// @param callback Invoked for each iteration of the beam_search loop, passing in beams_state.
/// @param callback_data A pointer that is simply passed back to callback.
/// @param n_beams Number of beams to use.
/// @param n_past Number of tokens already evaluated.
/// @param n_predict Maximum number of tokens to predict. EOS may occur earlier.
pub fn beamSearch(ctx: *Context, callback: BeamSearchCallback, callback_data: ?*anyopaque, n_beams: usize, n_past: c_int, n_predict: c_int) void {
    return c.llama_beam_search(ctx.cPtr(), callback, callback_data, n_beams, n_past, n_predict);
}

// Print system information
pub const printSystemInfo = c.llama_print_system_info;

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
pub const LogLevel = enum(c_int) {
    Error = 2, // GGML_LOG_LEVEL_ERROR = 2,
    Warn = 3, // GGML_LOG_LEVEL_WARN = 3,
    Info = 4, // GGML_LOG_LEVEL_INFO = 4
};
pub const LogCallback = *const fn (level: LogLevel, text: CStr, user_data: ?*anyopaque) callconv(.C) void;
pub fn logSet(cb: ?LogCallback, user_data: ?*anyopaque) void {
    c.llama_log_set(@ptrCast(cb), user_data);
}

pub const dumpTimingInfoToYaml = c.llama_dump_timing_info_yaml;

//
// Unrelated utils
//

pub const CStr = [*:0]const u8;

test {
    // include all tests that can be publicly referenced from here
    @import("std").testing.refAllDecls(@This());
}
