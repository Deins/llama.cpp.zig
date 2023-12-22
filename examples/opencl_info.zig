const std = @import("std");
const llama = @import("llama");
const opencl = llama.opencl_utils;
const log = std.debug.print;

pub fn checkErr(status: opencl.cl_int) !void {
    if (status != 0) {
        log("ERROR: {}", .{status});
        return error.OpenCL;
    }
}

pub fn castFromBuf(comptime T: type, buf: []const u8) T {
    var val: T = undefined;
    std.mem.copy(u8, @as([*]u8, @ptrCast(&val))[0..@sizeOf(@TypeOf(val))], buf);
    return val;
}

pub fn main() !void {
    const max_num = 256;
    var platforms_buffer: [max_num]*opencl.ClPlatformId = undefined;
    var platforms_len: opencl.cl_uint = 0;
    try checkErr(opencl.clGetPlatformIDs(max_num, &platforms_buffer, &platforms_len));
    const platforms = platforms_buffer[0..platforms_len];
    for (platforms, 0..) |p, pidx| {
        log("\nPlatform[{}]:\n", .{pidx});
        var param_buff: [512]u8 = undefined;
        var param_len: usize = 0;
        try checkErr(opencl.clGetPlatformInfo(p, .name, param_buff.len, &param_buff, &param_len));
        log("\tname:\t\t{s}\n", .{param_buff[0..param_len]});
        try checkErr(opencl.clGetPlatformInfo(p, .vendor, param_buff.len, &param_buff, &param_len));
        log("\tvendor:\t\t{s}\n", .{param_buff[0..param_len]});
        try checkErr(opencl.clGetPlatformInfo(p, .profile, param_buff.len, &param_buff, &param_len));
        log("\tprofile:\t\t{s}\n", .{param_buff[0..param_len]});
        try checkErr(opencl.clGetPlatformInfo(p, .version, param_buff.len, &param_buff, &param_len));
        log("\tversion:\t\t{s}\n", .{param_buff[0..param_len]});

        log("\tdevices:\n", .{});
        var device_buff: [512]*opencl.ClDeviceId = undefined;
        var device_len: opencl.cl_uint = 0;
        // pub extern fn clGetDeviceIDs(platform: *ClPlatformId, device_type: ClDeviceType, num_entries: cl_uint, devices: [*]*ClDeviceId, num_devices: *cl_uint) callconv(.C) cl_int;
        try checkErr(opencl.clGetDeviceIDs(p, .all, device_buff.len, &device_buff, &device_len));
        for (device_buff[0..device_len], 0..) |dev, dev_idx| {
            log("\t\tdevice[{}]:\n", .{dev_idx});

            try checkErr(opencl.clGetDeviceInfo(dev, .name, param_buff.len, &param_buff, &param_len));
            log("\t\t\tname:\t\t{s}\n", .{(param_buff[0..param_len])});

            try checkErr(opencl.clGetDeviceInfo(dev, .type, param_buff.len, &param_buff, &param_len));
            log("\t\t\ttype:\t\t{}\n", .{castFromBuf(opencl.ClDeviceType, param_buff[0..param_len])});
        }
    }
}
