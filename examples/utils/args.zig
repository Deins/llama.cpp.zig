const std = @import("std");
const slog = std.log.scoped(.ArgumentParser);

pub fn parseArgs(comptime ArgsStruct: type, args: [][:0]u8) !?ArgsStruct {
    var self = ArgsStruct{};
    var arg_i: usize = 0;
    var total_parsed_args: usize = 0;
    while (arg_i < args.len) : (arg_i += 1) {
        var curr = args[arg_i];
        var next: ?[:0]const u8 = if (arg_i + 1 < args.len) args[arg_i + 1] else null;
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
        inline for (@typeInfo(ArgsStruct).Struct.fields) |field| {
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
