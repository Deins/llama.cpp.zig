const std = @import("std");
const llama = @import("llama.zig");

const Token = llama.Token;
const TokenType = llama.TokenType;
const LogLevel = llama.LogLevel;
const SeqId = llama.SeqId;

pub fn scopedLog(level: LogLevel, text_: [*:0]const u8, user_data: ?*anyopaque) callconv(.C) void {
    _ = user_data;
    const sl = std.log.scoped(.llama_cpp);
    var text: []const u8 = std.mem.span(text_);
    while (text.len > 0 and text[text.len - 1] == '\n') text = text[0 .. text.len - 1]; // trim newlines
    if (text.len == 1 and text[0] == '.') return; // clean up formatting strings
    switch (level) {
        .Error => sl.err("{s}", .{text}),
        .Warn => sl.warn("{s}", .{text}),
        .Info => sl.info("{s}", .{text}),
    }
}

pub const Tokenizer = struct {
    data: std.ArrayList(Token),

    pub fn init(alloc: std.mem.Allocator) Tokenizer {
        return .{
            .data = std.ArrayList(llama.Token).init(alloc),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.data.deinit();
    }

    pub fn clearRetainingCapacity(self: *@This()) void {
        self.data.clearRetainingCapacity();
    }

    /// tokenize more text
    pub fn tokenize(self: *@This(), model: *llama.Model, text: []const u8, add_bos: bool, special: bool) !void {
        try self.data.ensureUnusedCapacity(text.len / 3 + 8); // assume that token on average is ~3 chars long
        var size = model.tokenize(text, self.data.unusedCapacitySlice(), add_bos, special);
        if (size < 0) {
            try self.data.ensureUnusedCapacity(@intCast(-size));
            size = model.tokenize(text, self.data.unusedCapacitySlice(), add_bos, special);
            if (size < 0) @panic("unexpected tokenization error"); // TODO: switch to unreachable once sure it works
        }
        self.data.items = self.data.allocatedSlice()[0 .. self.data.items.len + @as(usize, @intCast(size))];
    }

    pub fn getTokens(self: *@This()) []llama.Token {
        return self.data.items;
    }
};

/// direct token lookup from vocab
pub fn tokenGetText(model: *llama.Model, token: llama.Token) []const u8 {
    return std.mem.span(model.tokenGetText(token));
}

pub const Detokenizer = struct {
    data: std.ArrayList(u8),

    pub fn init(alloc: std.mem.Allocator) Detokenizer {
        return .{
            .data = std.ArrayList(u8).init(alloc),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.data.deinit();
    }

    pub fn clearRetainingCapacity(self: *@This()) void {
        self.data.clearRetainingCapacity();
    }

    /// de-tokenize another token. Doesn't display special tokens.
    pub fn detokenize(self: *@This(), model: *llama.Model, token: llama.Token) ![]const u8 {
        try self.data.ensureUnusedCapacity(16);
        var size = model.tokenToPiece(token, self.data.unusedCapacitySlice());
        if (size < 0) {
            try self.data.ensureUnusedCapacity(@intCast(-size));
            size = model.tokenToPiece(token, self.data.unusedCapacitySlice());
            if (size < 0) @panic("unexpected tokenization error"); // TODO: switch to unreachable once sure it works
        }
        const len = self.data.items.len;
        self.data.items = self.data.items.ptr[0 .. len + @as(usize, @intCast(size))];
        return self.data.items[len..];
    }

    /// detokenize, but also display special tokens in their text form. (useful for debugging raw prompt)
    pub fn detokenizeWithSpecial(self: *@This(), model: *llama.Model, token: llama.Token) ![]const u8 {
        switch (model.tokenGetType(token)) {
            .LLAMA_TOKEN_TYPE_NORMAL, .LLAMA_TOKEN_TYPE_BYTE => return try self.detokenize(model, token),
            .LLAMA_TOKEN_TYPE_UNUSED,
            .LLAMA_TOKEN_TYPE_UNDEFINED,
            .LLAMA_TOKEN_TYPE_USER_DEFINED,
            .LLAMA_TOKEN_TYPE_CONTROL,
            .LLAMA_TOKEN_TYPE_UNKNOWN,
            => {
                const len = self.data.items.len;
                try self.data.appendSlice(tokenGetText(model, token));
                return self.data.items[len..];
            },
        }
    }

    // ascii, trims whitespaces and special chars
    pub fn trimBack(self: *@This()) void {
        while (self.data.getLastOrNull()) |e| {
            if (e <= ' ') _ = self.data.pop() else break;
        }
    }

    pub fn getText(self: *@This()) []u8 {
        return self.data.items;
    }
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

/// simple prompt template mechanism, to allow building prompts matching their native training format
pub const TemplatedPrompt = struct {
    pub const Self = @This();

    pub const Params = struct {
        /// prompt template
        template: []const u8 =
            \\<|im_start|>{{role}}
            \\{{content}}{{eos}}
            \\
        ,
        /// for prompts that need it, role specific template
        role_specific_template: ?[]const Message = null,
        /// trim whitespaces for input role
        trim_role: bool = true,
        /// trim whitespaces for input content
        trim_content: bool = true,
        // optional user defined replacements in format: {[2][]const u8 {"needle_to_replace", "replacement"}, ...}
        additional_replacements: []const [2][]const u8 = &.{},

        // output values
        /// begining of sentence special token
        bos: []const u8 = "<|im_start|>",
        /// end of sentence special token
        eos: []const u8 = "<|im_end|>",
        /// newline special token
        nl: []const u8 = "\n",

        // WARNING: borrows values from model, will be valid only as long as model is loaded
        // NOTE: many models report wrong values, so make sure they are right
        pub fn setTokensFromModel(self: *Params, model: *llama.Model) void {
            self.nl = std.mem.span(model.tokenGetText(model.tokenEos()));
            self.bos = std.mem.span(model.tokenGetText(model.tokenBos()));
            self.eos = std.mem.span(model.tokenGetText(model.tokenEos()));
        }
        pub fn tokensFromModel(model: *llama.Model) Params {
            var p: Params = .{};
            p.setTokensFromModel(model);
            return p;
        }
    };

    pub const template_chatml: Params = .{};
    pub const template_basic_chat: Params = .{ .template = "{{role}}: {{content}}{{eos}}\n" };
    pub const template_alpaca: Params = .{
        .template = "### {{role}}:\n{{content}}{{eos}}\n",
        .role_specific_template = &[_]Message{
            .{
                .role = "system",
                .content = "### Instruction:\n{{content}}\n",
            },
            .{
                .role = "user",
                .content = "### Input:\n{{content}}\n",
            },
            .{
                .role = "assistant",
                .content = "### Response:\n{{content}}\n",
            },
        },
    };

    // pub const State = enum {
    //     Eos, // last sentence finished, can start new one
    //     Role, // new role has started
    //     Text, // writing more text, can either continue with more txt, or eos
    // };
    // state: State = .Eos,

    params: Params,
    text: std.ArrayList(u8), // assemble promt in text form here
    // two additional buffers for temp storage during processing. one kept for input second for writing, then swapped
    buffers: [2]std.ArrayListUnmanaged(u8),

    pub fn init(allocator: std.mem.Allocator, params: Params) Self {
        return .{
            .text = std.ArrayList(u8).init(allocator),
            .params = params,
            .buffers = [2]std.ArrayListUnmanaged(u8){ .{}, .{} },
        };
    }

    pub fn deinit(self: *Self) void {
        self.buffers[0].deinit(self.alloc());
        self.buffers[1].deinit(self.alloc());
        self.text.deinit();
    }

    pub fn alloc(self: *Self) std.mem.Allocator {
        return self.text.allocator;
    }

    /// Replace needle with replacement as many times as possible. Result is writen to one of the temp buffers
    /// TODO: might not be too optimal, investigate for better solution
    fn replace(self: *Self, haystack: []const u8, needle: []const u8, replacement: []const u8) ![]u8 {
        std.mem.swap(@TypeOf(self.buffers[0]), &self.buffers[0], &self.buffers[1]);
        const b = &self.buffers[0];
        b.clearRetainingCapacity();
        const shrink = replacement.len <= needle.len;
        try b.ensureTotalCapacity(self.alloc(), if (shrink) haystack.len else std.mem.replacementSize(u8, haystack, needle, replacement));
        const rcount = std.mem.replace(u8, haystack, needle, replacement, b.unusedCapacitySlice());
        if (shrink) return b.items.ptr[0 .. haystack.len - rcount * (needle.len - replacement.len)];
        return b.items.ptr[0 .. haystack.len + rcount * (replacement.len - needle.len)];
    }

    pub fn add(self: *Self, role_: []const u8, content_: []const u8) !void {
        const role = if (self.params.trim_role) trim(role_) else role_;
        var content = if (self.params.trim_content) trim(content_) else content_;
        const ends_with_generation = std.ascii.endsWithIgnoreCase(content, "{{generate}}");
        if (ends_with_generation) content = content[0 .. content.len - "{{generate}}".len];

        var text = self.params.template;
        if (self.params.role_specific_template) |rsts| for (rsts) |rt| if (std.ascii.eqlIgnoreCase(rt.role, role)) {
            text = rt.content;
        };

        const replacements = [_][2][]const u8{
            [2][]const u8{ "{{bos}}", self.params.bos },
            [2][]const u8{ "{{eos}}", if (ends_with_generation) "" else self.params.eos },
            [2][]const u8{ "{{role}}", role },
            [2][]const u8{ "{{content}}", content },
            [2][]const u8{ "\n", self.params.nl },
        };
        for (replacements) |r| {
            if (std.mem.eql(u8, r[0], r[1])) continue;
            text = try self.replace(text, r[0], r[1]);
        }
        for (self.params.additional_replacements) |r| {
            if (std.mem.eql(u8, r[0], r[1])) continue;
            text = try self.replace(text, r[0], r[1]);
        }
        if (ends_with_generation) text = trimBack(text);
        try self.text.appendSlice(text);
    }

    pub fn addMany(self: *Self, items: []const Message) !void {
        for (items) |item| try self.add(item.role, item.content);
    }

    pub fn addFromJson(self: *Self, json: []const u8) !void {
        var arena = std.heap.ArenaAllocator.init(self.alloc());
        const aaloc = arena.allocator();
        const parsed = try std.json.parseFromSlice([]Message, aaloc, json, .{});
        try self.addMany(parsed.value);
        arena.deinit();
    }

    pub fn addFromJsonFile(self: *Self, path: []const u8) !void {
        var pbuf: [std.fs.MAX_PATH_BYTES]u8 = undefined;
        const f = try std.fs.openFileAbsolute(try std.fs.realpath(path, &pbuf), .{});
        defer f.close();
        const content = try f.readToEndAlloc(self.alloc(), std.math.maxInt(usize));
        defer self.alloc().free(content);
        try self.addFromJson(content);
    }

    pub fn templateFromName(name_: []const u8) ?Params {
        const name = trim(name_);
        if (std.ascii.eqlIgnoreCase(name, "chatml")) return template_chatml;
        if (std.ascii.eqlIgnoreCase(name, "basic_chat")) return template_basic_chat;
        if (std.ascii.eqlIgnoreCase(name, "alpaca")) return template_alpaca;
        return null;
    }

    pub fn clearRetainingCapacity(self: *@This()) void {
        self.text.clearRetainingCapacity();
    }
};

// ascii TODO: unicode?
pub fn trimFront(text: []const u8) []const u8 {
    var i: usize = 0;
    while (i < text.len and text[i] <= ' ') i += 1; // ascii
    return text[i..];
}

// ascii TODO: unicode?
pub fn trimBack(text: []const u8) []const u8 {
    if (text.len == 0) return text;
    var i: usize = text.len - 1;
    while (text[i] <= ' ') {
        if (i > 0) i -= 1 else break;
    }
    return text[0 .. i + 1];
}

pub fn trim(text: []const u8) []const u8 {
    return trimBack(trimFront(text));
}

/// fixed size ring buffer to remember last data.len tokens
pub const TokenRingBuffer = struct {
    data: []Token,
    /// index to first element
    idx: usize = 0,
    /// lenght of elements used
    len: usize = 0,

    pub inline fn dataIdx(self: @This(), i: usize) usize {
        return (self.idx + i) % self.len;
    }

    pub inline fn at(self: @This(), i: usize) Token {
        return self.data[self.dataIdx(i)];
    }

    pub fn clear(self: *@This()) void {
        self.idx = 0;
        self.len = 0;
    }

    pub fn shrinkFront(self: *@This(), new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.idx = (self.idx + self.len - new_len) % self.data.len;
        self.len = new_len;
    }

    /// appends element at end, overwrites first element if full
    pub fn appendEraseFifo(self: *@This(), t: Token) void {
        std.debug.assert(self.data.len > 0);
        self.data[(self.idx + self.len) % self.data.len] = t;
        self.len += 1;
        if (self.len > self.data.len) {
            self.idx = (self.idx + self.len - self.data.len) % self.data.len;
            self.len = self.data.len;
        }
    }

    /// for fast iteration without many modulos
    /// returns two slices (second one can be empty if data is actually contigous)
    /// @param start_idx first element index (0 => first inserted element, self.len-1 => last element)
    pub fn slices(self: *@This(), start_idx: usize, len: usize) [2][]Token {
        std.debug.assert(self.data.len > 0);
        const a = self.idx + start_idx;
        const b = a + len;
        if (b <= self.data.len) return [2][]Token{ self.data[a..b], &.{} };
        return [2][]Token{ self.data[a..self.data.len], self.data[0 .. b % self.data.len] };
    }
};

test "TokenRingBuffer" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var buf: [5]Token = undefined;
    var ring: TokenRingBuffer = .{ .data = &buf };
    ring.appendEraseFifo(0);
    ring.appendEraseFifo(1);
    ring.appendEraseFifo(2);
    ring.appendEraseFifo(3);
    // 0 1 2 3 X
    try testing.expectEqualSlices(Token, &[_]Token{ 0, 1 }, ring.slices(0, 2)[0]);
    try testing.expectEqualSlices(Token, &[_]Token{}, ring.slices(0, 2)[1]);
    ring.appendEraseFifo(4);
    ring.appendEraseFifo(5);
    // 5 1 2 3 4
    try testing.expectEqualSlices(Token, &[_]Token{ 2, 3 }, ring.slices(1, 2)[0]);
    try testing.expectEqualSlices(Token, &[_]Token{}, ring.slices(1, 2)[1]);
    try testing.expectEqualSlices(Token, &[_]Token{4}, ring.slices(3, 2)[0]);
    try testing.expectEqualSlices(Token, &[_]Token{5}, ring.slices(3, 2)[1]);
    for (0..33) |_| ring.appendEraseFifo(9);
    try testing.expectEqualSlices(Token, &[_]Token{ 9, 9, 9, 9, 9 }, try std.mem.concat(alloc, Token, &ring.slices(0, 5)));
    ring.appendEraseFifo(8);
    try testing.expectEqual(@as(Token, 8), ring.at(9));
    ring.appendEraseFifo(7);
    ring.appendEraseFifo(6);
    try testing.expectEqualSlices(Token, &[_]Token{ 9, 9, 8, 7, 6 }, try std.mem.concat(alloc, Token, &ring.slices(0, 5)));
}
