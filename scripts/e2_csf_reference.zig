// Thin C ABI for the unmodified external TensorFold cuda_csf.zig module.
// Build: zig build-lib -dynamic -O ReleaseSafe --dep codec -Mroot=this.zig -Mcodec=.../cuda_csf.zig -lc
const codec = @import("codec");

export fn csf_decode(fixed: [*]const u8, fixed_len: usize, exceptions: [*]const u8, exceptions_len: usize,
    n: usize, k: usize, out: [*]u8, out_len: usize) c_int {
    codec.decode(fixed[0..fixed_len], exceptions[0..exceptions_len], n, k, out[0..out_len]) catch return -1;
    return 0;
}
