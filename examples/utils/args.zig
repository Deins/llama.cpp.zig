const std = @import("std");
const slog = std.log.scoped(.ArgumentParser);
const builtin = @import("builtin");

pub fn parseArgs(comptime ArgsStruct: type, args: [][:0]u8) !?ArgsStruct {
    var self = ArgsStruct{};
    var arg_i: usize = 0;
    var total_parsed_args: usize = 0;
    while (arg_i < args.len) : (arg_i += 1) {
        var curr = args[arg_i];
        const next: ?[:0]const u8 = if (arg_i + 1 < args.len) args[arg_i + 1] else null;
        if (curr.len <= 2 or !std.ascii.startsWithIgnoreCase(curr, "--")) {
            slog.err("Unknown argument: '{s}'", .{curr});
            return error.InvalidArguments;
        }
        var args_used: usize = 0;
        // use struct field names as arguments automagically using meta-programming
        inline for (@typeInfo(ArgsStruct).Struct.fields) |field| {
            if (std.ascii.eqlIgnoreCase(field.name, curr[2..])) {
                const field_type = if (@typeInfo(field.type) == .Optional) @typeInfo(field.type).Optional.child else field.type;
                switch (@typeInfo(field_type)) {
                    .Pointer => @field(self, field.name) = try arg(next),
                    .Int => @field(self, field.name) = try std.fmt.parseInt(field_type, try arg(next), 10),
                    .Float => @field(self, field.name) = try std.fmt.parseFloat(field_type, try arg(next)),
                    else => @panic("Unsuported argument type!"),
                }
                args_used += 2;
                //slog.info("arg: {s} = " ++ if (@typeInfo(field_type) == .Pointer) "{?s}" else "{?}", .{ field.name, @field(self, field.name) });
            }
        }
        if (args_used <= 0) slog.warn("Unknown arg: {s}", .{curr}) else {
            total_parsed_args += args_used;
            arg_i += args_used - 1;
        }
    }
    return if (total_parsed_args > 0) self else null;
}

pub fn printHelp(comptime ArgsStruct: type) void {
    const help = comptime blk: {
        var help: [:0]const u8 = "Help: Arguments:\n" ++ "\tArgument\t\tType\t\tDefault value\n";
        for (@typeInfo(ArgsStruct).Struct.fields) |field| {
            help = help ++ "\t--" ++ field.name ++ "\t\t" ++ @typeName(field.type) ++ "\n"; // todo: default value
        }
        break :blk help;
    };
    std.debug.print(help ++ "\n", .{});
}

fn arg(a: ?[:0]const u8) ![:0]const u8 {
    if (a) |aa| return aa;
    slog.err("Expected another command line argument, but none provided", .{});
    return error.InvalidArguments;
}

pub fn configure_console() void {
    if (builtin.os.tag == .windows) {
        // configure windows console - use utf8 and ascii VT100 escape sequences
        const win_con = struct {
            const CP_UTF8: u32 = 65001;
            const ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;
            const ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200;
            const BOOL = std.os.windows.BOOL;
            const HANDLE = std.os.windows.HANDLE;
            const DWORD = std.os.windows.DWORD;
            const GetStdHandle = std.os.windows.GetStdHandle;
            const STD_OUTPUT_HANDLE = std.os.windows.STD_OUTPUT_HANDLE;
            const STD_ERROR_HANDLE = std.os.windows.STD_ERROR_HANDLE;
            const kernel32 = std.os.windows.kernel32;
            //const STD_INPUT_HANDLE: (DWORD) = -10;
            //const STD_OUTPUT_HANDLE: (DWORD) = -11;
            //const STD_ERROR_HANDLE: (DWORD) = -12;
            pub extern "kernel32" fn SetConsoleOutputCP(wCodePageID: std.os.windows.UINT) BOOL;
            pub extern "kernel32" fn SetConsoleMode(hConsoleHandle: HANDLE, dwMode: DWORD) BOOL;
            //pub const GetStdHandle = kernel32.GetStdHandle;
            pub fn configure() void {
                if (SetConsoleOutputCP(CP_UTF8) == 0) {
                    std.log.scoped(.console).err("Can't configure windows console to UTF8!", .{});
                }
                const stdout_handle: HANDLE = GetStdHandle(STD_OUTPUT_HANDLE) catch |err| {
                    std.log.scoped(.console).err("Windows: Can't get stdout handle! err: {}", .{err});
                    return;
                };
                // var stdin_handle: HANDLE = GetStdHandle(STD_INPUT_HANDLE) catch |err| {
                //     std.log.scoped(.console).err("Windows: Can't get stdin handle! err: {}", .{err});
                //     return;
                // };
                const stderr_handle: HANDLE = GetStdHandle(STD_ERROR_HANDLE) catch |err| {
                    std.log.scoped(.console).err("Windows: Can't get stderr handle! err: {}", .{err});
                    return;
                };
                // Get console mode
                var stdout_mode: DWORD = 0;
                //var stdin_mode: DWORD = 0;
                var stderr_mode: DWORD = 0;
                if (kernel32.GetConsoleMode(stdout_handle, &stdout_mode) == 0) {
                    std.log.scoped(.console).err("Windows can't get stdout console mode! {}", .{kernel32.GetLastError()});
                }
                // if (kernel32.GetConsoleMode(stdin_handle, &stdout_mode) == 0) {
                //     std.log.scoped(.console).err("Windows can't get stdin_mode console mode! {}", .{kernel32.GetLastError()});
                // }
                if (kernel32.GetConsoleMode(stderr_handle, &stderr_mode) == 0) {
                    std.log.scoped(.console).err("Windows can't get stderr console mode! {}", .{kernel32.GetLastError()});
                }
                // set ENABLE_VIRTUAL_TERMINAL_PROCESSING
                if (SetConsoleMode(stdout_handle, stdout_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0) {
                    std.log.scoped(.console).err("Windows can't set stdout console mode! {}", .{kernel32.GetLastError()});
                }
                // if (SetConsoleMode(stdin_handle, stdin_mode | ENABLE_VIRTUAL_TERMINAL_INPUT) == 0) {
                //     std.log.scoped(.console).err("Windows can't set stdin_mode console mode! {}", .{kernel32.GetLastError()});
                // }
                if (SetConsoleMode(stderr_handle, stderr_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0) {
                    std.log.scoped(.console).err("Windows can't set stderr console mode! {}", .{kernel32.GetLastError()});
                }
            }
        };
        win_con.configure();
    }
}
