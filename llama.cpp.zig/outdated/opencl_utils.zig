//
// OpenCL
//

pub const ClPlatformId = opaque {};
pub const ClDeviceId = opaque {};

pub const ClDeviceType = enum(cl_bitfield) {
    default = (1 << 0), // CL_DEVICE_TYPE_DEFAULT
    cpu = (1 << 1), // CL_DEVICE_TYPE_CPU
    gpu = (1 << 2), // CL_DEVICE_TYPE_GPU
    accelerator = (1 << 3), // CL_DEVICE_TYPE_ACCELERATOR
    custom = (1 << 4), // CL_DEVICE_TYPE_CUSTOM
    all = 0xFFFFFFFF, // CL_DEVICE_TYPE_ALL
};

pub const ClPlatformInfo = enum(cl_uint) {
    profile = 0x0900, // CL_PLATFORM_PROFILE
    version = 0x0901, // CL_PLATFORM_VERSION
    name = 0x0902, // CL_PLATFORM_NAME
    vendor = 0x0903, // CL_PLATFORM_VENDOR
    extensions = 0x0904, // CL_PLATFORM_EXTENSIONS
};

pub extern fn clGetPlatformIDs(num_entries: cl_uint, platforms: [*]*ClPlatformId, n_platforms: *cl_uint) callconv(.C) cl_int;
pub extern fn clGetDeviceIDs(platform: *ClPlatformId, device_type: ClDeviceType, num_entries: cl_uint, devices: [*]*ClDeviceId, num_devices: *cl_uint) callconv(.C) cl_int;
pub extern fn clGetPlatformInfo(platform: *ClPlatformId, param_name: ClPlatformInfo, param_value_size: usize, param_value: [*]u8, param_value_size_ret: *usize) cl_int;
pub extern fn clGetDeviceInfo(device: *ClDeviceId, param: ClDeviceInfo, val_size: usize, val: [*]u8, val_size_ret: *usize) callconv(.C) cl_int;

pub const cl_char = i8;
pub const cl_uchar = u8;
pub const cl_short = i16;
pub const cl_ushort = u16;
pub const cl_int = i32;
pub const cl_uint = u32;
pub const cl_long = i64;
pub const cl_ulong = u64;

pub const cl_half = i16;
pub const cl_float = f32;
pub const cl_double = f64;

pub const cl_bitfield = cl_ulong;

pub const ClDeviceInfo = enum(cl_uint) {
    type = 0x1000, // CL_DEVICE_TYPE
    vendor_id = 0x1001, // CL_DEVICE_VENDOR_ID
    max_compute_units = 0x1002, // CL_DEVICE_MAX_COMPUTE_UNITS
    max_work_item_dimensions = 0x1003, // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
    max_work_group_size = 0x1004, // CL_DEVICE_MAX_WORK_GROUP_SIZE
    max_work_item_sizes = 0x1005, // CL_DEVICE_MAX_WORK_ITEM_SIZES
    preferred_vector_width_char = 0x1006, // CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
    preferred_vector_width_short = 0x1007, // CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
    preferred_vector_width_int = 0x1008, // CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT
    preferred_vector_width_long = 0x1009, // CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG
    preferred_vector_width_float = 0x100A, // CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
    preferred_vector_width_double = 0x100B, // CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
    max_clock_frequency = 0x100C, // CL_DEVICE_MAX_CLOCK_FREQUENCY
    address_bits = 0x100D, // CL_DEVICE_ADDRESS_BITS
    max_read_image_args = 0x100E, // CL_DEVICE_MAX_READ_IMAGE_ARGS
    max_write_image_args = 0x100F, // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
    max_mem_alloc_size = 0x1010, // CL_DEVICE_MAX_MEM_ALLOC_SIZE
    image2d_max_width = 0x1011, // CL_DEVICE_IMAGE2D_MAX_WIDTH
    image2d_max_height = 0x1012, // CL_DEVICE_IMAGE2D_MAX_HEIGHT
    image3d_max_width = 0x1013, // CL_DEVICE_IMAGE3D_MAX_WIDTH
    image3d_max_height = 0x1014, // CL_DEVICE_IMAGE3D_MAX_HEIGHT
    image3d_max_depth = 0x1015, // CL_DEVICE_IMAGE3D_MAX_DEPTH
    image_support = 0x1016, // CL_DEVICE_IMAGE_SUPPORT
    max_parameter_size = 0x1017, // CL_DEVICE_MAX_PARAMETER_SIZE
    max_samplers = 0x1018, // CL_DEVICE_MAX_SAMPLERS
    mem_base_addr_align = 0x1019, // CL_DEVICE_MEM_BASE_ADDR_ALIGN
    min_data_type_align_size = 0x101A, // CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE
    single_fp_config = 0x101B, // CL_DEVICE_SINGLE_FP_CONFIG
    global_mem_cache_type = 0x101C, // CL_DEVICE_GLOBAL_MEM_CACHE_TYPE
    global_mem_cacheline_size = 0x101D, // CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
    global_mem_cache_size = 0x101E, // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
    global_mem_size = 0x101F, // CL_DEVICE_GLOBAL_MEM_SIZE
    max_constant_buffer_size = 0x1020, // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    max_constant_args = 0x1021, // CL_DEVICE_MAX_CONSTANT_ARGS
    local_mem_type = 0x1022, // CL_DEVICE_LOCAL_MEM_TYPE
    local_mem_size = 0x1023, // CL_DEVICE_LOCAL_MEM_SIZE
    error_correction_support = 0x1024, // CL_DEVICE_ERROR_CORRECTION_SUPPORT
    profiling_timer_resolution = 0x1025, // CL_DEVICE_PROFILING_TIMER_RESOLUTION
    endian_little = 0x1026, // CL_DEVICE_ENDIAN_LITTLE
    available = 0x1027, // CL_DEVICE_AVAILABLE
    compiler_available = 0x1028, // CL_DEVICE_COMPILER_AVAILABLE
    execution_capabilities = 0x1029, // CL_DEVICE_EXECUTION_CAPABILITIES
    //queue_properties = 0x102A, //  /* deprecated */ // CL_DEVICE_QUEUE_PROPERTIES
    queue_on_host_properties = 0x102A, // CL_DEVICE_QUEUE_ON_HOST_PROPERTIES
    name = 0x102B, // CL_DEVICE_NAME
    vendor = 0x102C, // CL_DEVICE_VENDOR
    driver_version = 0x102D, // CL_DRIVER_VERSION
    profile = 0x102E, // CL_DEVICE_PROFILE
    device_version = 0x102F, // CL_DEVICE_VERSION
    extensions = 0x1030, // CL_DEVICE_EXTENSIONS
    platform = 0x1031, // CL_DEVICE_PLATFORM
    double_fp_config = 0x1032, // CL_DEVICE_DOUBLE_FP_CONFIG
    //  0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG
    preferred_vector_width_half = 0x1034, // CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
    host_unified_memory = 0x1035, // CL_DEVICE_HOST_UNIFIED_MEMORY   /* deprecated */
    native_vector_width_char = 0x1036, // CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
    native_vector_width_short = 0x1037, // CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
    native_vector_width_int = 0x1038, // CL_DEVICE_NATIVE_VECTOR_WIDTH_INT
    native_vector_width_long = 0x1039, // CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
    native_vector_width_float = 0x103A, // CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
    native_vector_width_double = 0x103B, // CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
    native_vector_width_half = 0x103C, // CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
    opencl_c_version = 0x103D, // CL_DEVICE_OPENCL_C_VERSION
    linker_available = 0x103E, // CL_DEVICE_LINKER_AVAILABLE
    built_in_kernels = 0x103F, // CL_DEVICE_BUILT_IN_KERNELS
    image_max_buffer_size = 0x1040, // CL_DEVICE_IMAGE_MAX_BUFFER_SIZE
    image_max_array_size = 0x1041, // CL_DEVICE_IMAGE_MAX_ARRAY_SIZE
    parent_device = 0x1042, // CL_DEVICE_PARENT_DEVICE
    partition_max_sub_devices = 0x1043, // CL_DEVICE_PARTITION_MAX_SUB_DEVICES
    partition_properties = 0x1044, // CL_DEVICE_PARTITION_PROPERTIES
    partition_affinity_domain = 0x1045, // CL_DEVICE_PARTITION_AFFINITY_DOMAIN
    partition_type = 0x1046, // CL_DEVICE_PARTITION_TYPE
    reference_count = 0x1047, // CL_DEVICE_REFERENCE_COUNT
    preferred_interop_user_sync = 0x1048, // CL_DEVICE_PREFERRED_INTEROP_USER_SYNC
    printf_buffer_size = 0x1049, // CL_DEVICE_PRINTF_BUFFER_SIZE
    image_pitch_alignment = 0x104A, // CL_DEVICE_IMAGE_PITCH_ALIGNMENT
    image_base_address_alignment = 0x104B, // CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT
    max_read_write_image_args = 0x104C, // CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS
    max_global_variable_size = 0x104D, // CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE
    queue_on_device_properties = 0x104E, // CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES
    queue_on_device_preferred_size = 0x104F, // CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE
    queue_on_device_max_size = 0x1050, // CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE
    max_on_device_queues = 0x1051, // CL_DEVICE_MAX_ON_DEVICE_QUEUES
    max_on_device_events = 0x1052, // CL_DEVICE_MAX_ON_DEVICE_EVENTS
    svm_capabilities = 0x1053, // CL_DEVICE_SVM_CAPABILITIES
    global_variable_preferred_total_size = 0x1054, // CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE
    max_pipe_args = 0x1055, // CL_DEVICE_MAX_PIPE_ARGS
    pipe_max_active_reservations = 0x1056, // CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS
    pipe_max_packet_size = 0x1057, // CL_DEVICE_PIPE_MAX_PACKET_SIZE
    preferred_platform_atomic_alignment = 0x1058, // CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT
    preferred_global_atomic_alignment = 0x1059, // CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT
    preferred_local_atomic_alignment = 0x105A, // CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT

};
