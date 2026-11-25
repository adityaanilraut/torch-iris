#pragma once

#include <array>

namespace iris {

inline constexpr std::array ops_phase3_kernels{
// Chunk 0: Float atomic helper, Multinomial, Searchsorted, MaskedScatter, Isin, IndexAdd, RepeatInterleave, Roll
R"CLC(
inline void atomic_add_f32(__global float* addr, float val) {
    if (val == 0.0f) return;
    __global volatile uint* uaddr = (__global volatile uint*)addr;
    uint old_u = *uaddr;
    uint assumed;
    do {
        assumed = old_u;
        float sum = as_float(assumed) + val;
        old_u = atomic_cmpxchg(uaddr, assumed, as_uint(sum));
    } while (old_u != assumed);
}

inline float philox_u01_open(ulong seed, ulong counter, uint lane) {
    uint4 r = philox(seed, counter);
    uint bits = (lane == 0) ? r.s0 : ((lane == 1) ? r.s1 : ((lane == 2) ? r.s2 : r.s3));
    // Map to (0, 1) strictly open interval so log(u) is always finite
    return ((float)(bits >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

__kernel void multinomial_build_cdf(
    __global float* cdf, __global const float* probs, volatile __global int* invalid,
    int rows, int K, int fp16
) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    const ulong base = (ulong)row * (ulong)K;
    float sum = 0.0f;
    for (int j = 0; j < K; ++j) {
        const float v = read_float(probs, base + (ulong)j, fp16 != 0);
        if (isnan(v) || isinf(v) || v < 0.0f) {
            atomic_or(invalid, 1);
            return;
        }
        sum += v;
        cdf[base + (ulong)j] = sum;
    }
    if (sum <= 0.0f || isnan(sum) || isinf(sum)) {
        atomic_or(invalid, 2);
        return;
    }
    const float inv = 1.0f / sum;
    for (int j = 0; j < K - 1; ++j) {
        cdf[base + (ulong)j] *= inv;
    }
    cdf[base + (ulong)(K - 1)] = 1.0f;
}

__kernel void multinomial_sample_repl(
    __global long* out, __global const float* cdf,
    int rows, int K, int num_samples, ulong seed, ulong start_counter
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)rows * (ulong)num_samples;
    if (gid >= total) return;
    const int row = (int)(gid / (ulong)num_samples);
    const ulong base = (ulong)row * (ulong)K;
    const float u = philox_u01_open(seed, start_counter + (gid >> 2), (uint)(gid & 3));
    int lo = 0, hi = K - 1;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (cdf[base + (ulong)mid] < u) lo = mid + 1;
        else hi = mid;
    }
    out[gid] = (long)lo;
}

__kernel void multinomial_exp_keys(
    __global float* keys, __global const float* probs, volatile __global int* invalid,
    int rows, int K, int num_samples, ulong seed, ulong start_counter, int fp16
) {
    const int row = get_group_id(0);
    const int tid = get_local_id(0);
    if (row >= rows) return;
    __local int nz_local[128];
    const ulong base = (ulong)row * (ulong)K;
    int nz = 0;
    for (int j = tid; j < K; j += 128) {
        const float v = read_float(probs, base + (ulong)j, fp16 != 0);
        if (isnan(v) || isinf(v) || v < 0.0f) {
            atomic_or(invalid, 1);
            keys[base + (ulong)j] = -INFINITY;
        } else if (v > 0.0f) {
            ++nz;
            const ulong elem = base + (ulong)j;
            const float u = philox_u01_open(seed, start_counter + (elem >> 2), (uint)(elem & 3));
            // Efraimidis-Spirakis: maximizing log(u) / v gives exact weighted sampling without replacement
            keys[elem] = log(u) / v;
        } else {
            keys[base + (ulong)j] = -INFINITY;
        }
    }
    nz_local[tid] = nz;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) nz_local[tid] += nz_local[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        if (nz_local[0] == 0) atomic_or(invalid, 2);
        else if (nz_local[0] < num_samples) atomic_or(invalid, 4);
    }
}

__kernel void searchsorted_kernel(
    __global uchar* out, __global const uchar* seq, __global const uchar* vals,
    __global const long* sorter,
    ulong out_len, ulong seq_rows, ulong seq_len, ulong vals_per_row,
    uint seq_type, uint val_type, int has_sorter, int right, int out_int32
) {
    const ulong gid = get_global_id(0);
    if (gid >= out_len) return;
    const ulong row = (seq_rows == 1) ? 0 : (gid / vals_per_row);
    const ulong seq_base = row * seq_len;
    const bool both_int = (seq_type != VALUE_FLOAT && seq_type != VALUE_HALF &&
                           val_type != VALUE_FLOAT && val_type != VALUE_HALF);
    ulong lo = 0, hi = seq_len;
    if (both_int) {
        const long target = read_integer(vals, gid, val_type);
        while (lo < hi) {
            const ulong mid = lo + ((hi - lo) >> 1);
            const ulong sidx = has_sorter ? (ulong)sorter[seq_base + mid] : mid;
            const long v = read_integer(seq, seq_base + sidx, seq_type);
            if (right ? (v <= target) : (v < target)) lo = mid + 1;
            else hi = mid;
        }
    } else {
        const float target = (val_type == VALUE_FLOAT || val_type == VALUE_HALF)
            ? read_float((__global const float*)vals, gid, val_type == VALUE_HALF)
            : convert_float(read_integer(vals, gid, val_type));
        if (isnan(target)) {
            lo = seq_len;
        } else {
            while (lo < hi) {
                const ulong mid = lo + ((hi - lo) >> 1);
                const ulong sidx = has_sorter ? (ulong)sorter[seq_base + mid] : mid;
                const float v = (seq_type == VALUE_FLOAT || seq_type == VALUE_HALF)
                    ? read_float((__global const float*)seq, seq_base + sidx, seq_type == VALUE_HALF)
                    : convert_float(read_integer(seq, seq_base + sidx, seq_type));
                if (isnan(v)) {
                    hi = mid;
                } else if (right ? (v <= target) : (v < target)) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
        }
    }
    if (out_int32) ((__global int*)out)[gid] = (int)lo;
    else ((__global long*)out)[gid] = (long)lo;
}

__kernel void masked_scatter_count(
    __global ulong* block_counts, __global const uchar* mask, ulong length
) {
    __local ulong sc[256];
    const uint tid = get_local_id(0);
    const ulong gid = get_global_id(0);
    sc[tid] = (gid < length && mask[gid] != 0) ? 1UL : 0UL;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sc[tid] += sc[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) block_counts[get_group_id(0)] = sc[0];
}

__kernel void masked_scatter_prefix(
    __global ulong* block_offsets, __global ulong* total_out,
    __global const ulong* block_counts, ulong num_blocks
) {
    ulong acc = 0;
    for (ulong i = 0; i < num_blocks; ++i) {
        block_offsets[i] = acc;
        acc += block_counts[i];
    }
    total_out[0] = acc;
}

__kernel void masked_scatter_apply(
    __global uchar* dst, __global const uchar* mask, __global const uchar* src,
    __global const ulong* block_offsets, ulong length, ulong src_len, uint type
) {
    __local ulong sc[256];
    const uint tid = get_local_id(0);
    const ulong gid = get_global_id(0);
    const uchar m = (gid < length) ? (mask[gid] != 0) : 0;
    sc[tid] = (ulong)m;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Hillis-Steele inclusive scan in SLM
    for (uint off = 1; off < 256; off <<= 1) {
        ulong v = (tid >= off) ? sc[tid - off] : 0UL;
        barrier(CLK_LOCAL_MEM_FENCE);
        sc[tid] += v;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (m) {
        const ulong src_idx = block_offsets[get_group_id(0)] + sc[tid] - 1UL;
        if (src_idx < src_len) {
            const uint bytes = value_size(type);
            for (uint b = 0; b < bytes; ++b) {
                dst[gid * bytes + b] = src[src_idx * bytes + b];
            }
        }
    }
}

__kernel void isin_kernel(
    __global uchar* out, __global const uchar* elems, __global const uchar* tests,
    ulong elem_len, ulong test_len, uint elem_type, uint test_type, int is_sorted, int invert
) {
    const ulong gid = get_global_id(0);
    if (gid >= elem_len) return;
    const bool use_fp = (elem_type == VALUE_FLOAT || elem_type == VALUE_HALF ||
                         test_type == VALUE_FLOAT || test_type == VALUE_HALF);
    bool found = false;
    if (!use_fp) {
        const long ev = read_integer(elems, gid, elem_type);
        if (is_sorted && test_len > 16) {
            ulong lo = 0, hi = test_len;
            while (lo < hi) {
                const ulong mid = lo + ((hi - lo) >> 1);
                const long tv = read_integer(tests, mid, test_type);
                if (tv < ev) lo = mid + 1;
                else if (tv > ev) hi = mid;
                else { found = true; break; }
            }
        } else {
            for (ulong j = 0; j < test_len; ++j) {
                if (read_integer(tests, j, test_type) == ev) { found = true; break; }
            }
        }
    } else {
        const float ev = (elem_type == VALUE_FLOAT || elem_type == VALUE_HALF)
            ? read_float((__global const float*)elems, gid, elem_type == VALUE_HALF)
            : convert_float(read_integer(elems, gid, elem_type));
        if (!isnan(ev)) {
            if (is_sorted && test_len > 16) {
                ulong lo = 0, hi = test_len;
                while (lo < hi) {
                    const ulong mid = lo + ((hi - lo) >> 1);
                    const float tv = (test_type == VALUE_FLOAT || test_type == VALUE_HALF)
                        ? read_float((__global const float*)tests, mid, test_type == VALUE_HALF)
                        : convert_float(read_integer(tests, mid, test_type));
                    if (isnan(tv)) { hi = mid; }
                    else if (tv < ev) lo = mid + 1;
                    else if (tv > ev) hi = mid;
                    else { found = true; break; }
                }
            } else {
                for (ulong j = 0; j < test_len; ++j) {
                    const float tv = (test_type == VALUE_FLOAT || test_type == VALUE_HALF)
                        ? read_float((__global const float*)tests, j, test_type == VALUE_HALF)
                        : convert_float(read_integer(tests, j, test_type));
                    if (tv == ev) { found = true; break; }
                }
            }
        }
    }
    out[gid] = (uchar)(invert ? (!found) : found);
}

__kernel void index_add_kernel(
    __global uchar* self_buf, __global const uchar* src_buf, __global const uchar* idx_buf,
    volatile __global int* invalid,
    ulong outer_size, ulong self_dim, ulong idx_len, ulong inner_size,
    float alpha_f, long alpha_l, uint data_type, uint idx_type
) {
    const ulong slice = get_global_id(0);
    if (slice >= outer_size * inner_size) return;
    const ulong outer = slice / inner_size;
    const ulong inner = slice - outer * inner_size;
    const ulong self_base = outer * self_dim * inner_size + inner;
    const ulong src_base = outer * idx_len * inner_size + inner;
    const bool is_fp = (data_type == VALUE_FLOAT || data_type == VALUE_HALF);
    for (ulong i = 0; i < idx_len; ++i) {
        long target = read_integer(idx_buf, i, idx_type);
        if (target < 0) target += (long)self_dim;
        if (target < 0 || (ulong)target >= self_dim) {
            atomic_or(invalid, 1);
            continue;
        }
        const ulong dst_pos = self_base + (ulong)target * inner_size;
        const ulong src_pos = src_base + i * inner_size;
        if (is_fp) {
            const float cur = read_float((__global const float*)self_buf, dst_pos, data_type == VALUE_HALF);
            const float addv = read_float((__global const float*)src_buf, src_pos, data_type == VALUE_HALF);
            write_float((__global float*)self_buf, dst_pos, fma(alpha_f, addv, cur), data_type == VALUE_HALF);
        } else {
            const long cur = read_integer(self_buf, dst_pos, data_type);
            const long addv = read_integer(src_buf, src_pos, data_type);
            write_integer(self_buf, dst_pos, data_type, cur + alpha_l * addv);
        }
    }
}

__kernel void repeat_interleave_fill(
    __global long* out, __global const long* ends, ulong total_out, ulong num_repeats
) {
    const ulong pos = get_global_id(0);
    if (pos >= total_out) return;
    ulong lo = 0, hi = num_repeats - 1;
    while (lo < hi) {
        const ulong mid = lo + ((hi - lo) >> 1);
        if ((ulong)ends[mid] <= pos) lo = mid + 1;
        else hi = mid;
    }
    out[pos] = (long)lo;
}

__kernel void roll_kernel(
    __global uchar* out, __global const uchar* inp,
    __global const ulong* sizes, __global const long* shifts,
    ulong length, ulong rank, uint type, int flat_mode, long flat_shift
) {
    const ulong gid = get_global_id(0);
    if (gid >= length) return;
    ulong src_idx = 0;
    if (flat_mode) {
        long s = ((long)gid - flat_shift) % (long)length;
        if (s < 0) s += (long)length;
        src_idx = (ulong)s;
    } else {
        ulong rem = gid;
        ulong stride = 1;
        for (ulong d = rank; d > 0; --d) {
            const ulong sz = sizes[d - 1];
            const ulong coord = rem % sz;
            rem /= sz;
            long sc = ((long)coord - shifts[d - 1]) % (long)sz;
            if (sc < 0) sc += (long)sz;
            src_idx += (ulong)sc * stride;
            stride *= sz;
        }
    }
    const uint bytes = value_size(type);
    for (uint b = 0; b < bytes; ++b) {
        out[gid * bytes + b] = inp[src_idx * bytes + b];
    }
}
)CLC",

// Chunk 1: ConstantPadND, Reflection/Replication Pad2D (+ backward), Activations (+ backward), PReLU (+ backward)
R"CLC(
__kernel void constant_pad_nd_kernel(
    __global uchar* out, __global const uchar* inp,
    __global const ulong* out_sizes, __global const ulong* in_sizes,
    __global const long* pad_before,
    ulong length, ulong rank, uint type, float fill_f, long fill_l
) {
    const ulong gid = get_global_id(0);
    if (gid >= length) return;
    ulong rem = gid;
    ulong in_idx = 0;
    ulong in_stride = 1;
    bool in_bounds = true;
    for (ulong d = rank; d > 0; --d) {
        const ulong osz = out_sizes[d - 1];
        const ulong isz = in_sizes[d - 1];
        const ulong oc = rem % osz;
        rem /= osz;
        const long ic = (long)oc - pad_before[d - 1];
        if (ic < 0 || ic >= (long)isz) {
            in_bounds = false;
            break;
        }
        in_idx += (ulong)ic * in_stride;
        in_stride *= isz;
    }
    if (in_bounds) {
        const uint bytes = value_size(type);
        for (uint b = 0; b < bytes; ++b) {
            out[gid * bytes + b] = inp[in_idx * bytes + b];
        }
    } else if (type == VALUE_FLOAT || type == VALUE_HALF) {
        write_float((__global float*)out, gid, fill_f, type == VALUE_HALF);
    } else {
        write_integer(out, gid, type, fill_l);
    }
}

inline int map_pad_coord(int oc, int pad_l, int sz, int mode) {
    int ic = oc - pad_l;
    if (mode == 0) { // reflect
        if (ic < 0) ic = -ic;
        if (ic >= sz) ic = 2 * sz - 2 - ic;
    } else { // replicate
        if (ic < 0) ic = 0;
        if (ic >= sz) ic = sz - 1;
    }
    return ic;
}

__kernel void pad2d_mode_fwd(
    __global float* out, __global const float* inp,
    int NC, int H, int W, int OH, int OW,
    int pad_l, int pad_t, int mode, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)NC * (ulong)OH * (ulong)OW;
    if (gid >= total) return;
    const int ow = (int)(gid % (ulong)OW);
    const ulong tmp = gid / (ulong)OW;
    const int oh = (int)(tmp % (ulong)OH);
    const int nc = (int)(tmp / (ulong)OH);
    const int ih = map_pad_coord(oh, pad_t, H, mode);
    const int iw = map_pad_coord(ow, pad_l, W, mode);
    const ulong src = ((ulong)nc * (ulong)H + (ulong)ih) * (ulong)W + (ulong)iw;
    write_float(out, gid, read_float(inp, src, fp16 != 0), fp16 != 0);
}

__kernel void pad2d_mode_bwd(
    __global float* grad_in, __global const float* grad_out,
    int NC, int H, int W, int OH, int OW,
    int pad_l, int pad_r, int pad_t, int pad_b, int mode, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)NC * (ulong)H * (ulong)W;
    if (gid >= total) return;
    const int iw = (int)(gid % (ulong)W);
    const ulong tmp = gid / (ulong)W;
    const int ih = (int)(tmp % (ulong)H);
    const int nc = (int)(tmp / (ulong)H);
    const ulong go_base = (ulong)nc * (ulong)OH * (ulong)OW;

    // Collect all output y coordinates that map to ih, and all output x coordinates that map to iw
    // For reflect (mode 0): at most 3 y's (ih + pad_t, pad_t - ih if ih>0, 2*H-2-ih+pad_t if ih<H-1)
    // For replicate (mode 1): if ih == 0 -> [0..pad_t], if ih == H-1 -> [pad_t+H-1..OH-1], else [ih+pad_t]
    float acc = 0.0f;
    if (mode == 0) {
        int ys[3]; int ny = 0;
        int y0 = ih + pad_t; if (y0 >= 0 && y0 < OH) ys[ny++] = y0;
        if (ih > 0) { int y1 = pad_t - ih; if (y1 >= 0 && y1 < pad_t && y1 < OH) ys[ny++] = y1; }
        if (ih < H - 1) { int y2 = 2 * H - 2 - ih + pad_t; if (y2 >= pad_t + H && y2 < OH) ys[ny++] = y2; }

        int xs[3]; int nx = 0;
        int x0 = iw + pad_l; if (x0 >= 0 && x0 < OW) xs[nx++] = x0;
        if (iw > 0) { int x1 = pad_l - iw; if (x1 >= 0 && x1 < pad_l && x1 < OW) xs[nx++] = x1; }
        if (iw < W - 1) { int x2 = 2 * W - 2 - iw + pad_l; if (x2 >= pad_l + W && x2 < OW) xs[nx++] = x2; }

        for (int a = 0; a < ny; ++a) {
            for (int b = 0; b < nx; ++b) {
                acc += read_float(grad_out, go_base + (ulong)ys[a] * (ulong)OW + (ulong)xs[b], fp16 != 0);
            }
        }
    } else {
        const int y_lo = (ih == 0) ? 0 : (ih + pad_t);
        const int y_hi = (ih == H - 1) ? (OH - 1) : (ih + pad_t);
        const int x_lo = (iw == 0) ? 0 : (iw + pad_l);
        const int x_hi = (iw == W - 1) ? (OW - 1) : (iw + pad_l);
        for (int oy = max(0, y_lo); oy <= min(OH - 1, y_hi); ++oy) {
            for (int ox = max(0, x_lo); ox <= min(OW - 1, x_hi); ++ox) {
                acc += read_float(grad_out, go_base + (ulong)oy * (ulong)OW + (ulong)ox, fp16 != 0);
            }
        }
    }
    write_float(grad_in, gid, acc, fp16 != 0);
}

enum ActP3 {
    ACT_LEAKY_RELU = 0, ACT_ELU = 1, ACT_HARDSWISH = 2, ACT_HARDSIGMOID = 3,
    ACT_MISH = 4, ACT_HARDTANH = 5
};

__kernel void act_p3_fwd(
    __global float* out, __global const float* inp, ulong length,
    int act, float p0, float p1, float p2, int fp16
) {
    const ulong i = get_global_id(0);
    if (i >= length) return;
    const float x = read_float(inp, i, fp16 != 0);
    float y = x;
    if (act == ACT_LEAKY_RELU) {
        y = (x > 0.0f) ? x : (x * p0);
    } else if (act == ACT_ELU) {
        // alpha=p0, scale=p1, input_scale=p2
        y = (x > 0.0f) ? (p1 * x) : (p1 * p0 * expm1(x * p2));
    } else if (act == ACT_HARDSWISH) {
        y = (x <= -3.0f) ? 0.0f : ((x >= 3.0f) ? x : (x * (x + 3.0f) * (1.0f / 6.0f)));
    } else if (act == ACT_HARDSIGMOID) {
        y = (x <= -3.0f) ? 0.0f : ((x >= 3.0f) ? 1.0f : ((x + 3.0f) * (1.0f / 6.0f)));
    } else if (act == ACT_MISH) {
        const float sp = (x > 20.0f) ? x : log1p(exp(x));
        y = x * tanh(sp);
    } else if (act == ACT_HARDTANH) {
        y = (x < p0) ? p0 : ((x > p1) ? p1 : x);
    }
    write_float(out, i, y, fp16 != 0);
}

__kernel void act_p3_bwd(
    __global float* grad_in, __global const float* grad_out, __global const float* self_or_res,
    ulong length, int act, float p0, float p1, float p2, int is_result, int fp16
) {
    const ulong i = get_global_id(0);
    if (i >= length) return;
    const float g = read_float(grad_out, i, fp16 != 0);
    const float x = read_float(self_or_res, i, fp16 != 0);
    float dx = g;
    if (act == ACT_LEAKY_RELU) {
        dx = (x > 0.0f) ? g : (g * p0);
    } else if (act == ACT_ELU) {
        if (is_result) {
            dx = (x <= 0.0f) ? (g * p2 * (x + p0 * p1)) : (g * p1);
        } else {
            dx = (x <= 0.0f) ? (g * p1 * p0 * p2 * exp(x * p2)) : (g * p1);
        }
    } else if (act == ACT_HARDSWISH) {
        dx = (x <= -3.0f) ? 0.0f : ((x >= 3.0f) ? g : (g * (2.0f * x + 3.0f) * (1.0f / 6.0f)));
    } else if (act == ACT_HARDSIGMOID) {
        dx = (x <= -3.0f || x >= 3.0f) ? 0.0f : (g * (1.0f / 6.0f));
    } else if (act == ACT_MISH) {
        const float sp = (x > 20.0f) ? x : log1p(exp(x));
        const float tsp = tanh(sp);
        const float sig = 1.0f / (1.0f + exp(-x));
        const float sech2 = 1.0f - tsp * tsp;
        dx = g * (tsp + x * sig * sech2);
    } else if (act == ACT_HARDTANH) {
        dx = (x <= p0 || x >= p1) ? 0.0f : g;
    }
    write_float(grad_in, i, dx, fp16 != 0);
}

__kernel void prelu_fwd_kernel(
    __global float* out, __global const float* inp, __global const float* weight,
    ulong length, ulong channels, ulong inner_size, int single_weight, int fp16
) {
    const ulong i = get_global_id(0);
    if (i >= length) return;
    const float x = read_float(inp, i, fp16 != 0);
    const ulong c = single_weight ? 0UL : ((i / inner_size) % channels);
    const float w = read_float(weight, c, fp16 != 0);
    write_float(out, i, (x > 0.0f) ? x : (x * w), fp16 != 0);
}

__kernel void prelu_bwd_dx_kernel(
    __global float* dx, __global const float* grad_out, __global const float* inp,
    __global const float* weight, ulong length, ulong channels, ulong inner_size,
    int single_weight, int fp16
) {
    const ulong i = get_global_id(0);
    if (i >= length) return;
    const float x = read_float(inp, i, fp16 != 0);
    const float g = read_float(grad_out, i, fp16 != 0);
    const ulong c = single_weight ? 0UL : ((i / inner_size) % channels);
    const float w = read_float(weight, c, fp16 != 0);
    write_float(dx, i, (x > 0.0f) ? g : (g * w), fp16 != 0);
}

__kernel void prelu_bwd_dw_kernel(
    __global float* dw, __global const float* grad_out, __global const float* inp,
    ulong outer_size, ulong channels, ulong inner_size, int single_weight, int fp16
) {
    __local float sc[128];
    const int w_idx = get_group_id(0);
    const int tid = get_local_id(0);
    float acc = 0.0f;
    if (single_weight) {
        const ulong total = outer_size * channels * inner_size;
        for (ulong i = (ulong)tid; i < total; i += 128) {
            const float x = read_float(inp, i, fp16 != 0);
            if (x <= 0.0f) acc = fma(read_float(grad_out, i, fp16 != 0), x, acc);
        }
    } else {
        const ulong per_c = outer_size * inner_size;
        for (ulong k = (ulong)tid; k < per_c; k += 128) {
            const ulong o = k / inner_size;
            const ulong inr = k - o * inner_size;
            const ulong pos = (o * channels + (ulong)w_idx) * inner_size + inr;
            const float x = read_float(inp, pos, fp16 != 0);
            if (x <= 0.0f) acc = fma(read_float(grad_out, pos, fp16 != 0), x, acc);
        }
    }
    sc[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sc[tid] += sc[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) write_float(dw, (ulong)w_idx, sc[0], fp16 != 0);
}
)CLC",

// Chunk 2: GridSampler 2D (+ backward)
R"CLC(
inline float gs_unnormalize(float coord, int size, int align_corners) {
    if (align_corners) return ((coord + 1.0f) * 0.5f) * (float)(size - 1);
    return ((coord + 1.0f) * (float)size - 1.0f) * 0.5f;
}

inline float gs_unnormalize_grad(float coord, int size, int align_corners, float* g_mult) {
    if (align_corners) {
        *g_mult = 0.5f * (float)(size - 1);
        return ((coord + 1.0f) * 0.5f) * (float)(size - 1);
    }
    *g_mult = 0.5f * (float)size;
    return ((coord + 1.0f) * (float)size - 1.0f) * 0.5f;
}

inline float gs_reflect(float in_x, float min_v, float max_v) {
    const float span = max_v - min_v;
    if (span <= 0.0f) return 0.0f;
    float x = fabs(in_x - min_v);
    float extra = fmod(x, span);
    int flips = (int)floor(x / span);
    return (flips & 1) ? (max_v - extra) : (min_v + extra);
}

inline float gs_compute_coord(float coord, int size, int padding_mode, int align_corners) {
    float c = gs_unnormalize(coord, size, align_corners);
    if (padding_mode == 1) { // border
        c = fmin(fmax(c, 0.0f), (float)(size - 1));
    } else if (padding_mode == 2) { // reflection
        c = align_corners ? gs_reflect(c, 0.0f, (float)(size - 1))
                          : gs_reflect(c, -0.5f, (float)size - 0.5f);
        c = fmin(fmax(c, 0.0f), (float)(size - 1));
    }
    return c;
}

inline float gs_compute_coord_grad(float coord, int size, int padding_mode, int align_corners, float* grad_in_mult) {
    float mult = 1.0f;
    float c = gs_unnormalize_grad(coord, size, align_corners, &mult);
    if (padding_mode == 1) { // border
        if (c < 0.0f) { c = 0.0f; mult = 0.0f; }
        else if (c > (float)(size - 1)) { c = (float)(size - 1); mult = 0.0f; }
    } else if (padding_mode == 2) { // reflection
        const float min_v = align_corners ? 0.0f : -0.5f;
        const float max_v = align_corners ? (float)(size - 1) : ((float)size - 0.5f);
        const float span = max_v - min_v;
        if (span <= 0.0f) {
            c = 0.0f; mult = 0.0f;
        } else {
            int flip = 0;
            float dx = c - min_v;
            if (dx < 0.0f) { dx = -dx; flip ^= 1; }
            int nflips = (int)floor(dx / span);
            float extra = fmod(dx, span);
            if (nflips & 1) { c = max_v - extra; flip ^= 1; }
            else { c = min_v + extra; }
            if (flip) mult = -mult;
            if (c < 0.0f) { c = 0.0f; mult = 0.0f; }
            else if (c > (float)(size - 1)) { c = (float)(size - 1); mult = 0.0f; }
        }
    }
    *grad_in_mult = mult;
    return c;
}

inline float gs_cubic_w(float t, int idx) {
    const float A = -0.75f;
    if (idx == 0) return ((A * (t + 1.0f) - 5.0f * A) * (t + 1.0f) + 8.0f * A) * (t + 1.0f) - 4.0f * A;
    if (idx == 1) return ((A + 2.0f) * t - (A + 3.0f)) * t * t + 1.0f;
    if (idx == 2) return ((A + 2.0f) * (1.0f - t) - (A + 3.0f)) * (1.0f - t) * (1.0f - t) + 1.0f;
    return ((A * (2.0f - t) - 5.0f * A) * (2.0f - t) + 8.0f * A) * (2.0f - t) - 4.0f * A;
}

inline float gs_cubic_dw(float t, int idx) {
    const float A = -0.75f;
    if (idx == 0) return (3.0f * A * (t + 1.0f) - 10.0f * A) * (t + 1.0f) + 8.0f * A;
    if (idx == 1) return (3.0f * (A + 2.0f) * t - 2.0f * (A + 3.0f)) * t;
    if (idx == 2) return -((3.0f * (A + 2.0f) * (1.0f - t) - 2.0f * (A + 3.0f)) * (1.0f - t));
    return -((3.0f * A * (2.0f - t) - 10.0f * A) * (2.0f - t) + 8.0f * A);
}

inline int gs_bound_idx(int idx, int size, int padding_mode, int align_corners, bool* valid) {
    if (idx >= 0 && idx < size) { *valid = true; return idx; }
    if (padding_mode == 0) { *valid = false; return 0; }
    if (padding_mode == 1) { *valid = true; return clamp(idx, 0, size - 1); }
    float c = align_corners ? gs_reflect((float)idx, 0.0f, (float)(size - 1))
                            : gs_reflect((float)idx, -0.5f, (float)size - 0.5f);
    *valid = true;
    return clamp((int)round(c), 0, size - 1);
}

__kernel void grid_sampler_2d_fwd(
    __global float* out, __global const float* inp, __global const float* grid,
    int N, int C, int IH, int IW, int OH, int OW,
    int interp_mode, int padding_mode, int align_corners, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)N * (ulong)OH * (ulong)OW;
    if (gid >= total) return;
    const int ow = (int)(gid % (ulong)OW);
    const ulong tmp = gid / (ulong)OW;
    const int oh = (int)(tmp % (ulong)OH);
    const int n = (int)(tmp / (ulong)OH);

    const ulong gpos = (((ulong)n * (ulong)OH + (ulong)oh) * (ulong)OW + (ulong)ow) * 2UL;
    const float gx = read_float(grid, gpos, fp16 != 0);
    const float gy = read_float(grid, gpos + 1UL, fp16 != 0);

    float ix = (interp_mode == 2) ? gs_unnormalize(gx, IW, align_corners)
                                  : gs_compute_coord(gx, IW, padding_mode, align_corners);
    float iy = (interp_mode == 2) ? gs_unnormalize(gy, IH, align_corners)
                                  : gs_compute_coord(gy, IH, padding_mode, align_corners);

    const ulong in_n = (ulong)n * (ulong)C * (ulong)IH * (ulong)IW;
    const ulong out_n = (ulong)n * (ulong)C * (ulong)OH * (ulong)OW + (ulong)oh * (ulong)OW + (ulong)ow;
    const ulong ihw = (ulong)IH * (ulong)IW;
    const ulong ohw = (ulong)OH * (ulong)OW;

    if (interp_mode == 0) { // bilinear
        const int ix_nw = (int)floor(ix);
        const int iy_nw = (int)floor(iy);
        const int ix_ne = ix_nw + 1, iy_ne = iy_nw;
        const int ix_sw = ix_nw, iy_sw = iy_nw + 1;
        const int ix_se = ix_nw + 1, iy_se = iy_nw + 1;
        const float nw = (ix_se - ix) * (iy_se - iy);
        const float ne = (ix - ix_sw) * (iy_sw - iy);
        const float sw = (ix_ne - ix) * (iy - iy_ne);
        const float se = (ix - ix_nw) * (iy - iy_nw);
        for (int c = 0; c < C; ++c) {
            const ulong cb = in_n + (ulong)c * ihw;
            float v = 0.0f;
            if (iy_nw >= 0 && iy_nw < IH && ix_nw >= 0 && ix_nw < IW) v += read_float(inp, cb + (ulong)iy_nw * IW + ix_nw, fp16 != 0) * nw;
            if (iy_ne >= 0 && iy_ne < IH && ix_ne >= 0 && ix_ne < IW) v += read_float(inp, cb + (ulong)iy_ne * IW + ix_ne, fp16 != 0) * ne;
            if (iy_sw >= 0 && iy_sw < IH && ix_sw >= 0 && ix_sw < IW) v += read_float(inp, cb + (ulong)iy_sw * IW + ix_sw, fp16 != 0) * sw;
            if (iy_se >= 0 && iy_se < IH && ix_se >= 0 && ix_se < IW) v += read_float(inp, cb + (ulong)iy_se * IW + ix_se, fp16 != 0) * se;
            write_float(out, out_n + (ulong)c * ohw, v, fp16 != 0);
        }
    } else if (interp_mode == 1) { // nearest
        const int ix_n = (int)rint(ix);
        const int iy_n = (int)rint(iy);
        const bool ok = (ix_n >= 0 && ix_n < IW && iy_n >= 0 && iy_n < IH);
        for (int c = 0; c < C; ++c) {
            float v = ok ? read_float(inp, in_n + (ulong)c * ihw + (ulong)iy_n * IW + ix_n, fp16 != 0) : 0.0f;
            write_float(out, out_n + (ulong)c * ohw, v, fp16 != 0);
        }
    } else { // bicubic
        if (padding_mode == 2) {
            ix = align_corners ? gs_reflect(ix, 0.0f, (float)(IW - 1)) : gs_reflect(ix, -0.5f, (float)IW - 0.5f);
            iy = align_corners ? gs_reflect(iy, 0.0f, (float)(IH - 1)) : gs_reflect(iy, -0.5f, (float)IH - 0.5f);
        }
        const int ix0 = (int)floor(ix);
        const int iy0 = (int)floor(iy);
        const float tx = ix - (float)ix0;
        const float ty = iy - (float)iy0;
        float wx[4], wy[4];
        for (int k = 0; k < 4; ++k) { wx[k] = gs_cubic_w(tx, k); wy[k] = gs_cubic_w(ty, k); }
        int xs[4], ys[4]; bool vx[4], vy[4];
        for (int k = 0; k < 4; ++k) {
            xs[k] = gs_bound_idx(ix0 - 1 + k, IW, padding_mode, align_corners, &vx[k]);
            ys[k] = gs_bound_idx(iy0 - 1 + k, IH, padding_mode, align_corners, &vy[k]);
        }
        for (int c = 0; c < C; ++c) {
            const ulong cb = in_n + (ulong)c * ihw;
            float v = 0.0f;
            for (int ky = 0; ky < 4; ++ky) {
                if (!vy[ky]) continue;
                float row_v = 0.0f;
                for (int kx = 0; kx < 4; ++kx) {
                    if (vx[kx]) row_v += wx[kx] * read_float(inp, cb + (ulong)ys[ky] * IW + xs[kx], fp16 != 0);
                }
                v += wy[ky] * row_v;
            }
            write_float(out, out_n + (ulong)c * ohw, v, fp16 != 0);
        }
    }
}

__kernel void grid_sampler_2d_bwd(
    __global float* grad_in_f32, __global float* grad_grid,
    __global const float* grad_out, __global const float* inp, __global const float* grid,
    int N, int C, int IH, int IW, int OH, int OW,
    int interp_mode, int padding_mode, int align_corners,
    int need_gi, int need_gg, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)N * (ulong)OH * (ulong)OW;
    if (gid >= total) return;
    const int ow = (int)(gid % (ulong)OW);
    const ulong tmp = gid / (ulong)OW;
    const int oh = (int)(tmp % (ulong)OH);
    const int n = (int)(tmp / (ulong)OH);

    const ulong gpos = (((ulong)n * (ulong)OH + (ulong)oh) * (ulong)OW + (ulong)ow) * 2UL;
    const float gx = read_float(grid, gpos, fp16 != 0);
    const float gy = read_float(grid, gpos + 1UL, fp16 != 0);
    float gix_mult = 0.0f, giy_mult = 0.0f;
    float ix = 0.0f, iy = 0.0f;
    if (interp_mode == 2) {
        if (padding_mode == 2) {
            ix = gs_compute_coord_grad(gx, IW, 2, align_corners, &gix_mult);
            iy = gs_compute_coord_grad(gy, IH, 2, align_corners, &giy_mult);
            // Undo border clamp inside gs_compute_coord_grad? No, reflect stays within [min_v, max_v], handle directly:
            float m_x = 1.0f, m_y = 1.0f;
            float ux = gs_unnormalize_grad(gx, IW, align_corners, &m_x);
            float uy = gs_unnormalize_grad(gy, IH, align_corners, &m_y);
            const float min_x = align_corners ? 0.0f : -0.5f, max_x = align_corners ? (float)(IW - 1) : ((float)IW - 0.5f);
            const float min_y = align_corners ? 0.0f : -0.5f, max_y = align_corners ? (float)(IH - 1) : ((float)IH - 0.5f);
            const float sp_x = max_x - min_x, sp_y = max_y - min_y;
            if (sp_x > 0.0f) {
                int fl = 0; float dx = ux - min_x; if (dx < 0.0f) { dx = -dx; fl ^= 1; }
                int nf = (int)floor(dx / sp_x); float ex = fmod(dx, sp_x);
                if (nf & 1) { ux = max_x - ex; fl ^= 1; } else { ux = min_x + ex; }
                if (fl) m_x = -m_x;
            } else { ux = 0.0f; m_x = 0.0f; }
            if (sp_y > 0.0f) {
                int fl = 0; float dy = uy - min_y; if (dy < 0.0f) { dy = -dy; fl ^= 1; }
                int nf = (int)floor(dy / sp_y); float ex = fmod(dy, sp_y);
                if (nf & 1) { uy = max_y - ex; fl ^= 1; } else { uy = min_y + ex; }
                if (fl) m_y = -m_y;
            } else { uy = 0.0f; m_y = 0.0f; }
            ix = ux; iy = uy; gix_mult = m_x; giy_mult = m_y;
        } else {
            ix = gs_unnormalize_grad(gx, IW, align_corners, &gix_mult);
            iy = gs_unnormalize_grad(gy, IH, align_corners, &giy_mult);
        }
    } else {
        ix = gs_compute_coord_grad(gx, IW, padding_mode, align_corners, &gix_mult);
        iy = gs_compute_coord_grad(gy, IH, padding_mode, align_corners, &giy_mult);
    }

    const ulong in_n = (ulong)n * (ulong)C * (ulong)IH * (ulong)IW;
    const ulong out_n = (ulong)n * (ulong)C * (ulong)OH * (ulong)OW + (ulong)oh * (ulong)OW + (ulong)ow;
    const ulong ihw = (ulong)IH * (ulong)IW;
    const ulong ohw = (ulong)OH * (ulong)OW;

    float ggx = 0.0f, ggy = 0.0f;
    if (interp_mode == 0) { // bilinear
        const int ix_nw = (int)floor(ix), iy_nw = (int)floor(iy);
        const int ix_ne = ix_nw + 1, iy_ne = iy_nw;
        const int ix_sw = ix_nw, iy_sw = iy_nw + 1;
        const int ix_se = ix_nw + 1, iy_se = iy_nw + 1;
        const float nw = (ix_se - ix) * (iy_se - iy);
        const float ne = (ix - ix_sw) * (iy_sw - iy);
        const float sw = (ix_ne - ix) * (iy - iy_ne);
        const float se = (ix - ix_nw) * (iy - iy_nw);
        for (int c = 0; c < C; ++c) {
            const float go = read_float(grad_out, out_n + (ulong)c * ohw, fp16 != 0);
            const ulong cb = in_n + (ulong)c * ihw;
            if (iy_nw >= 0 && iy_nw < IH && ix_nw >= 0 && ix_nw < IW) {
                const ulong pos = cb + (ulong)iy_nw * IW + ix_nw;
                if (need_gi) atomic_add_f32(grad_in_f32 + pos, go * nw);
                if (need_gg) {
                    const float iv = read_float(inp, pos, fp16 != 0);
                    ggx -= iv * (iy_se - iy) * go;
                    ggy -= iv * (ix_se - ix) * go;
                }
            }
            if (iy_ne >= 0 && iy_ne < IH && ix_ne >= 0 && ix_ne < IW) {
                const ulong pos = cb + (ulong)iy_ne * IW + ix_ne;
                if (need_gi) atomic_add_f32(grad_in_f32 + pos, go * ne);
                if (need_gg) {
                    const float iv = read_float(inp, pos, fp16 != 0);
                    ggx += iv * (iy_sw - iy) * go;
                    ggy -= iv * (ix - ix_sw) * go;
                }
            }
            if (iy_sw >= 0 && iy_sw < IH && ix_sw >= 0 && ix_sw < IW) {
                const ulong pos = cb + (ulong)iy_sw * IW + ix_sw;
                if (need_gi) atomic_add_f32(grad_in_f32 + pos, go * sw);
                if (need_gg) {
                    const float iv = read_float(inp, pos, fp16 != 0);
                    ggx -= iv * (iy - iy_ne) * go;
                    ggy += iv * (ix_ne - ix) * go;
                }
            }
            if (iy_se >= 0 && iy_se < IH && ix_se >= 0 && ix_se < IW) {
                const ulong pos = cb + (ulong)iy_se * IW + ix_se;
                if (need_gi) atomic_add_f32(grad_in_f32 + pos, go * se);
                if (need_gg) {
                    const float iv = read_float(inp, pos, fp16 != 0);
                    ggx += iv * (iy - iy_nw) * go;
                    ggy += iv * (ix - ix_nw) * go;
                }
            }
        }
    } else if (interp_mode == 1) { // nearest
        const int ix_n = (int)rint(ix), iy_n = (int)rint(iy);
        if (need_gi && ix_n >= 0 && ix_n < IW && iy_n >= 0 && iy_n < IH) {
            for (int c = 0; c < C; ++c) {
                const float go = read_float(grad_out, out_n + (ulong)c * ohw, fp16 != 0);
                atomic_add_f32(grad_in_f32 + in_n + (ulong)c * ihw + (ulong)iy_n * IW + ix_n, go);
            }
        }
    } else { // bicubic
        const int ix0 = (int)floor(ix), iy0 = (int)floor(iy);
        const float tx = ix - (float)ix0, ty = iy - (float)iy0;
        float wx[4], wy[4], dwx[4], dwy[4];
        for (int k = 0; k < 4; ++k) {
            wx[k] = gs_cubic_w(tx, k); wy[k] = gs_cubic_w(ty, k);
            dwx[k] = gs_cubic_dw(tx, k); dwy[k] = gs_cubic_dw(ty, k);
        }
        int xs[4], ys[4]; bool vx[4], vy[4];
        for (int k = 0; k < 4; ++k) {
            xs[k] = gs_bound_idx(ix0 - 1 + k, IW, padding_mode, align_corners, &vx[k]);
            ys[k] = gs_bound_idx(iy0 - 1 + k, IH, padding_mode, align_corners, &vy[k]);
        }
        for (int c = 0; c < C; ++c) {
            const float go = read_float(grad_out, out_n + (ulong)c * ohw, fp16 != 0);
            const ulong cb = in_n + (ulong)c * ihw;
            for (int ky = 0; ky < 4; ++ky) {
                if (!vy[ky]) continue;
                for (int kx = 0; kx < 4; ++kx) {
                    if (!vx[kx]) continue;
                    const ulong pos = cb + (ulong)ys[ky] * IW + xs[kx];
                    if (need_gi) atomic_add_f32(grad_in_f32 + pos, go * wy[ky] * wx[kx]);
                    if (need_gg) {
                        const float iv = read_float(inp, pos, fp16 != 0);
                        ggx += iv * wy[ky] * dwx[kx] * go;
                        ggy += iv * dwy[ky] * wx[kx] * go;
                    }
                }
            }
        }
    }
    if (need_gg) {
        write_float(grad_grid, gpos, ggx * gix_mult, fp16 != 0);
        write_float(grad_grid, gpos + 1UL, ggy * giy_mult, fp16 != 0);
    }
}
)CLC",

// Chunk 3: 3D Vision Ops (Conv3d, ConvTranspose3d, MaxPool3d, AvgPool3d + backward)
R"CLC(
__kernel void conv3d_fwd_kernel(
    __global float* out, __global const float* inp, __global const float* w, __global const float* bias,
    int N, int C_in, int ID, int IH, int IW,
    int C_out, int OD, int OH, int OW,
    int KD, int KH, int KW, int sd, int sh, int sw,
    int pd, int ph, int pw, int dd, int dh, int dw, int groups, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)N * C_out * OD * OH * OW;
    if (gid >= total) return;
    const int ow = (int)(gid % OW); ulong t = gid / OW;
    const int oh = (int)(t % OH); t /= OH;
    const int od = (int)(t % OD); t /= OD;
    const int oc = (int)(t % C_out);
    const int n = (int)(t / C_out);
    const int oc_g = C_out / groups;
    const int ic_g = C_in / groups;
    const int g = oc / oc_g;
    float acc = bias ? read_float(bias, (ulong)oc, fp16 != 0) : 0.0f;
    const ulong idhw = (ulong)ID * IH * IW;
    const ulong kdhw = (ulong)KD * KH * KW;
    const ulong in_n = ((ulong)n * C_in + (ulong)(g * ic_g)) * idhw;
    const ulong w_oc = (ulong)oc * ic_g * kdhw;
    for (int ic = 0; ic < ic_g; ++ic) {
        const ulong in_c = in_n + (ulong)ic * idhw;
        const ulong w_c = w_oc + (ulong)ic * kdhw;
        for (int kd = 0; kd < KD; ++kd) {
            const int id = od * sd - pd + kd * dd;
            if (id < 0 || id >= ID) continue;
            for (int kh = 0; kh < KH; ++kh) {
                const int ih = oh * sh - ph + kh * dh;
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    const int iw = ow * sw - pw + kw * dw;
                    if (iw < 0 || iw >= IW) continue;
                    const float iv = read_float(inp, in_c + ((ulong)id * IH + ih) * IW + iw, fp16 != 0);
                    const float wv = read_float(w, w_c + ((ulong)kd * KH + kh) * KW + kw, fp16 != 0);
                    acc = fma(iv, wv, acc);
                }
            }
        }
    }
    write_float(out, gid, acc, fp16 != 0);
}

__kernel void conv3d_grad_input_kernel(
    __global float* grad_in, __global const float* grad_out, __global const float* w,
    int N, int C_in, int ID, int IH, int IW,
    int C_out, int OD, int OH, int OW,
    int KD, int KH, int KW, int sd, int sh, int sw,
    int pd, int ph, int pw, int dd, int dh, int dw, int groups, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)N * C_in * ID * IH * IW;
    if (gid >= total) return;
    const int iw = (int)(gid % IW); ulong t = gid / IW;
    const int ih = (int)(t % IH); t /= IH;
    const int id = (int)(t % ID); t /= ID;
    const int ic = (int)(t % C_in);
    const int n = (int)(t / C_in);
    const int oc_g = C_out / groups;
    const int ic_g = C_in / groups;
    const int g = ic / ic_g;
    const int ic_l = ic - g * ic_g;
    float acc = 0.0f;
    const ulong odhw = (ulong)OD * OH * OW;
    const ulong kdhw = (ulong)KD * KH * KW;
    const ulong go_n = ((ulong)n * C_out + (ulong)(g * oc_g)) * odhw;
    for (int ocl = 0; ocl < oc_g; ++ocl) {
        const ulong go_c = go_n + (ulong)ocl * odhw;
        const ulong w_c = ((ulong)(g * oc_g + ocl) * ic_g + ic_l) * kdhw;
        for (int kd = 0; kd < KD; ++kd) {
            const int td = id + pd - kd * dd;
            if (td < 0 || (td % sd) != 0) continue;
            const int od = td / sd;
            if (od >= OD) continue;
            for (int kh = 0; kh < KH; ++kh) {
                const int th = ih + ph - kh * dh;
                if (th < 0 || (th % sh) != 0) continue;
                const int oh = th / sh;
                if (oh >= OH) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    const int tw = iw + pw - kw * dw;
                    if (tw < 0 || (tw % sw) != 0) continue;
                    const int ow = tw / sw;
                    if (ow >= OW) continue;
                    const float gv = read_float(grad_out, go_c + ((ulong)od * OH + oh) * OW + ow, fp16 != 0);
                    const float wv = read_float(w, w_c + ((ulong)kd * KH + kh) * KW + kw, fp16 != 0);
                    acc = fma(gv, wv, acc);
                }
            }
        }
    }
    write_float(grad_in, gid, acc, fp16 != 0);
}

__kernel void conv3d_grad_weight_kernel(
    __global float* grad_w, __global const float* grad_out, __global const float* inp,
    int N, int C_in, int ID, int IH, int IW,
    int C_out, int OD, int OH, int OW,
    int KD, int KH, int KW, int sd, int sh, int sw,
    int pd, int ph, int pw, int dd, int dh, int dw, int groups, int fp16
) {
    const int ic_g = C_in / groups;
    const int oc_g = C_out / groups;
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)C_out * ic_g * KD * KH * KW;
    if (gid >= total) return;
    const int kw = (int)(gid % KW); ulong t = gid / KW;
    const int kh = (int)(t % KH); t /= KH;
    const int kd = (int)(t % KD); t /= KD;
    const int ic_l = (int)(t % ic_g);
    const int oc = (int)(t / ic_g);
    const int g = oc / oc_g;
    const int ic = g * ic_g + ic_l;
    const ulong idhw = (ulong)ID * IH * IW;
    const ulong odhw = (ulong)OD * OH * OW;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        const ulong go_c = ((ulong)n * C_out + oc) * odhw;
        const ulong in_c = ((ulong)n * C_in + ic) * idhw;
        for (int od = 0; od < OD; ++od) {
            const int id = od * sd - pd + kd * dd;
            if (id < 0 || id >= ID) continue;
            for (int oh = 0; oh < OH; ++oh) {
                const int ih = oh * sh - ph + kh * dh;
                if (ih < 0 || ih >= IH) continue;
                for (int ow = 0; ow < OW; ++ow) {
                    const int iw = ow * sw - pw + kw * dw;
                    if (iw < 0 || iw >= IW) continue;
                    const float gv = read_float(grad_out, go_c + ((ulong)od * OH + oh) * OW + ow, fp16 != 0);
                    const float iv = read_float(inp, in_c + ((ulong)id * IH + ih) * IW + iw, fp16 != 0);
                    acc = fma(gv, iv, acc);
                }
            }
        }
    }
    write_float(grad_w, gid, acc, fp16 != 0);
}

__kernel void max_pool3d_fwd_kernel(
    __global float* out, __global long* indices, __global const float* inp,
    int NC, int ID, int IH, int IW, int OD, int OH, int OW,
    int KD, int KH, int KW, int sd, int sh, int sw,
    int pd, int ph, int pw, int dd, int dh, int dw, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)NC * OD * OH * OW;
    if (gid >= total) return;
    const int ow = (int)(gid % OW); ulong t = gid / OW;
    const int oh = (int)(t % OH); t /= OH;
    const int od = (int)(t % OD);
    const int nc = (int)(t / OD);
    const ulong in_base = (ulong)nc * ID * IH * IW;
    float best = -INFINITY;
    long best_idx = 0;
    bool found = false;
    for (int kd = 0; kd < KD; ++kd) {
        const int id = od * sd - pd + kd * dd;
        if (id < 0 || id >= ID) continue;
        for (int kh = 0; kh < KH; ++kh) {
            const int ih = oh * sh - ph + kh * dh;
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; ++kw) {
                const int iw = ow * sw - pw + kw * dw;
                if (iw < 0 || iw >= IW) continue;
                const long idx = ((long)id * IH + ih) * IW + iw;
                const float v = read_float(inp, in_base + (ulong)idx, fp16 != 0);
                if (!found || v > best || isnan(v)) {
                    best = v; best_idx = idx; found = true;
                    if (isnan(v)) break;
                }
            }
            if (found && isnan(best)) break;
        }
        if (found && isnan(best)) break;
    }
    write_float(out, gid, best, fp16 != 0);
    indices[gid] = best_idx;
}

__kernel void max_pool3d_bwd_kernel(
    __global float* grad_in_f32, __global const float* grad_out, __global const long* indices,
    int NC, int ID, int IH, int IW, int OD, int OH, int OW, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)NC * OD * OH * OW;
    if (gid >= total) return;
    const ulong odhw = (ulong)OD * OH * OW;
    const ulong idhw = (ulong)ID * IH * IW;
    const ulong nc = gid / odhw;
    const long idx = indices[gid];
    const float g = read_float(grad_out, gid, fp16 != 0);
    atomic_add_f32(grad_in_f32 + nc * idhw + (ulong)idx, g);
}

__kernel void avg_pool3d_fwd_kernel(
    __global float* out, __global const float* inp,
    int NC, int ID, int IH, int IW, int OD, int OH, int OW,
    int KD, int KH, int KW, int sd, int sh, int sw,
    int pd, int ph, int pw, int include_pad, int divisor_override, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)NC * OD * OH * OW;
    if (gid >= total) return;
    const int ow = (int)(gid % OW); ulong t = gid / OW;
    const int oh = (int)(t % OH); t /= OH;
    const int od = (int)(t % OD);
    const int nc = (int)(t / OD);
    const int dstart = od * sd - pd, hstart = oh * sh - ph, wstart = ow * sw - pw;
    const int dend = min(dstart + KD, ID + pd), hend = min(hstart + KH, IH + ph), wend = min(wstart + KW, IW + pw);
    const int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    const int d0 = max(dstart, 0), h0 = max(hstart, 0), w0 = max(wstart, 0);
    const int d1 = min(dend, ID), h1 = min(hend, IH), w1 = min(wend, IW);
    const ulong in_base = (ulong)nc * ID * IH * IW;
    float sum = 0.0f;
    for (int id = d0; id < d1; ++id) {
        for (int ih = h0; ih < h1; ++ih) {
            for (int iw = w0; iw < w1; ++iw) {
                sum += read_float(inp, in_base + ((ulong)id * IH + ih) * IW + iw, fp16 != 0);
            }
        }
    }
    const int div = (divisor_override > 0) ? divisor_override
                  : (include_pad ? pool_size : max(1, (d1 - d0) * (h1 - h0) * (w1 - w0)));
    write_float(out, gid, sum / (float)div, fp16 != 0);
}

__kernel void avg_pool3d_bwd_kernel(
    __global float* grad_in, __global const float* grad_out,
    int NC, int ID, int IH, int IW, int OD, int OH, int OW,
    int KD, int KH, int KW, int sd, int sh, int sw,
    int pd, int ph, int pw, int include_pad, int divisor_override, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)NC * ID * IH * IW;
    if (gid >= total) return;
    const int iw = (int)(gid % IW); ulong t = gid / IW;
    const int ih = (int)(t % IH); t /= IH;
    const int id = (int)(t % ID);
    const int nc = (int)(t / ID);
    const ulong go_base = (ulong)nc * OD * OH * OW;
    const int od0 = (id + pd < KD) ? 0 : ((id + pd - KD) / sd + 1);
    const int od1 = min((id + pd) / sd + 1, OD);
    const int oh0 = (ih + ph < KH) ? 0 : ((ih + ph - KH) / sh + 1);
    const int oh1 = min((ih + ph) / sh + 1, OH);
    const int ow0 = (iw + pw < KW) ? 0 : ((iw + pw - KW) / sw + 1);
    const int ow1 = min((iw + pw) / sw + 1, OW);
    float acc = 0.0f;
    for (int od = od0; od < od1; ++od) {
        const int dstart = od * sd - pd;
        const int dend = min(dstart + KD, ID + pd);
        const int d0 = max(dstart, 0), d1 = min(dend, ID);
        if (id < d0 || id >= d1) continue;
        for (int oh = oh0; oh < oh1; ++oh) {
            const int hstart = oh * sh - ph;
            const int hend = min(hstart + KH, IH + ph);
            const int h0 = max(hstart, 0), h1 = min(hend, IH);
            if (ih < h0 || ih >= h1) continue;
            for (int ow = ow0; ow < ow1; ++ow) {
                const int wstart = ow * sw - pw;
                const int wend = min(wstart + KW, IW + pw);
                const int w0 = max(wstart, 0), w1 = min(wend, IW);
                if (iw < w0 || iw >= w1) continue;
                const int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                const int div = (divisor_override > 0) ? divisor_override
                              : (include_pad ? pool_size : max(1, (d1 - d0) * (h1 - h0) * (w1 - w0)));
                acc += read_float(grad_out, go_base + ((ulong)od * OH + oh) * OW + ow, fp16 != 0) / (float)div;
            }
        }
    }
    write_float(grad_in, gid, acc, fp16 != 0);
}
)CLC",

// Chunk 4: Var/Std, Prod, VectorNorm, Ternary (lerp/addcmul/addcdiv), Global Integer Bitonic Sort
R"CLC(
__kernel void var_std_reduce_kernel(
    __global float* var_out, __global float* mean_out, __global const float* inp,
    ulong outer_size, ulong reduce_len, float correction, int take_sqrt, int write_mean, int fp16
) {
    __local float sc[128];
    const ulong row = get_group_id(0);
    const int tid = get_local_id(0);
    if (row >= outer_size) return;
    const ulong base = row * reduce_len;
    float s = 0.0f;
    for (ulong i = (ulong)tid; i < reduce_len; i += 128) {
        s += read_float(inp, base + i, fp16 != 0);
    }
    sc[tid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 64; step > 0; step >>= 1) {
        if (tid < step) sc[tid] += sc[tid + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = (reduce_len > 0) ? (sc[0] / (float)reduce_len) : NAN;
    barrier(CLK_LOCAL_MEM_FENCE);
    float sq = 0.0f;
    for (ulong i = (ulong)tid; i < reduce_len; i += 128) {
        const float d = read_float(inp, base + i, fp16 != 0) - mean;
        sq = fma(d, d, sq);
    }
    sc[tid] = sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 64; step > 0; step >>= 1) {
        if (tid < step) sc[tid] += sc[tid + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        const float denom = (float)reduce_len - correction;
        float v = (denom <= 0.0f) ? ((reduce_len == 0 || sc[0] == 0.0f) ? NAN : INFINITY) : (sc[0] / denom);
        if (take_sqrt) v = sqrt(v);
        write_float(var_out, row, v, fp16 != 0);
        if (write_mean) write_float(mean_out, row, mean, fp16 != 0);
    }
}

__kernel void prod_reduce_kernel(
    __global uchar* out, __global const uchar* inp,
    ulong outer_size, ulong reduce_len, uint in_type, uint out_type
) {
    __local float sc_f[128];
    __local long sc_i[128];
    const ulong row = get_group_id(0);
    const int tid = get_local_id(0);
    if (row >= outer_size) return;
    const ulong base = row * reduce_len;
    const bool out_fp = (out_type == VALUE_FLOAT || out_type == VALUE_HALF);
    const bool in_fp = (in_type == VALUE_FLOAT || in_type == VALUE_HALF);
    if (out_fp) {
        float acc = 1.0f;
        for (ulong i = (ulong)tid; i < reduce_len; i += 128) {
            const float v = in_fp ? read_float((__global const float*)inp, base + i, in_type == VALUE_HALF)
                                  : convert_float(read_integer(inp, base + i, in_type));
            acc *= v;
        }
        sc_f[tid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int step = 64; step > 0; step >>= 1) {
            if (tid < step) sc_f[tid] *= sc_f[tid + step];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) write_float((__global float*)out, row, sc_f[0], out_type == VALUE_HALF);
    } else {
        long acc = 1;
        for (ulong i = (ulong)tid; i < reduce_len; i += 128) {
            acc *= read_integer(inp, base + i, in_type);
        }
        sc_i[tid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int step = 64; step > 0; step >>= 1) {
            if (tid < step) sc_i[tid] *= sc_i[tid + step];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) write_integer(out, row, out_type, sc_i[0]);
    }
}

__kernel void vector_norm_reduce_kernel(
    __global float* out, __global const float* inp,
    ulong outer_size, ulong reduce_len, float ord, int ord_mode, int in_fp16, int out_fp16
) {
    // ord_mode: 0=L2, 1=L1, 2=L0, 3=Linf, 4=L-inf, 5=general p
    __local float sc[128];
    const ulong row = get_group_id(0);
    const int tid = get_local_id(0);
    if (row >= outer_size) return;
    const ulong base = row * reduce_len;
    float acc = (ord_mode == 3) ? 0.0f : ((ord_mode == 4) ? INFINITY : 0.0f);
    for (ulong i = (ulong)tid; i < reduce_len; i += 128) {
        const float a = fabs(read_float(inp, base + i, in_fp16 != 0));
        if (ord_mode == 0) acc = fma(a, a, acc);
        else if (ord_mode == 1) acc += a;
        else if (ord_mode == 2) acc += (a != 0.0f) ? 1.0f : 0.0f;
        else if (ord_mode == 3) acc = (isnan(a) || isnan(acc)) ? NAN : fmax(acc, a);
        else if (ord_mode == 4) acc = (isnan(a) || isnan(acc)) ? NAN : fmin(acc, a);
        else acc += pow(a, ord);
    }
    sc[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 64; step > 0; step >>= 1) {
        if (tid < step) {
            if (ord_mode == 3) sc[tid] = (isnan(sc[tid]) || isnan(sc[tid + step])) ? NAN : fmax(sc[tid], sc[tid + step]);
            else if (ord_mode == 4) sc[tid] = (isnan(sc[tid]) || isnan(sc[tid + step])) ? NAN : fmin(sc[tid], sc[tid + step]);
            else sc[tid] += sc[tid + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        float res = sc[0];
        if (reduce_len == 0) {
            res = (ord_mode == 4 || (ord_mode == 5 && ord < 0.0f)) ? INFINITY : 0.0f;
        } else if (ord_mode == 0) {
            res = sqrt(res);
        } else if (ord_mode == 5) {
            res = pow(res, 1.0f / ord);
        }
        write_float(out, row, res, out_fp16 != 0);
    }
}

__kernel void ternary_contiguous_kernel(
    __global float* out, __global const float* a, __global const float* b, __global const float* c,
    ulong length, int op, float alpha, int fp16
) {
    // op: 0=lerp(a, b, c), 1=addcmul(a, b, c, alpha), 2=addcdiv(a, b, c, alpha)
    const ulong i = get_global_id(0);
    if (i >= length) return;
    const float va = read_float(a, i, fp16 != 0);
    const float vb = read_float(b, i, fp16 != 0);
    const float vc = read_float(c, i, fp16 != 0);
    float res = 0.0f;
    if (op == 0) {
        res = (fabs(vc) < 0.5f) ? fma(vc, vb - va, va) : (vb - (vb - va) * (1.0f - vc));
    } else if (op == 1) {
        res = fma(alpha * vb, vc, va);
    } else {
        res = fma(alpha, vb / vc, va);
    }
    write_float(out, i, res, fp16 != 0);
}

__kernel void bitonic_sort_init_i64(
    __global long* buf_v, __global int* buf_i, __global const uchar* input,
    int rows, int width, int P, uint type
) {
    const ulong gid = get_global_id(0);
    if (gid >= (ulong)rows * (ulong)P) return;
    const ulong row = gid / (ulong)P;
    const int col = (int)(gid - row * (ulong)P);
    buf_v[gid] = (col < width) ? read_integer(input, row * (ulong)width + (ulong)col, type) : 0L;
    buf_i[gid] = col;
}

__kernel void bitonic_sort_step_i64(
    __global long* buf_v, __global int* buf_i,
    int rows, int width, int P, int k, int j, int descending
) {
    const ulong half_p = (ulong)(P >> 1);
    const ulong gid = get_global_id(0);
    if (gid >= (ulong)rows * half_p) return;
    const ulong row = gid / half_p;
    const int t = (int)(gid - row * half_p);
    const int block = t / j;
    const int offset = t - block * j;
    const int i = block * (j << 1) + offset;
    const int ixj = i + j;
    const ulong base = row * (ulong)P;
    const long va = buf_v[base + i], vb = buf_v[base + ixj];
    const int ia = buf_i[base + i], ib = buf_i[base + ixj];
    const bool asc = ((i & k) == 0);
    const bool swap = asc
        ? sort_after_i64(va, ia, vb, ib, width, descending)
        : sort_after_i64(vb, ib, va, ia, width, descending);
    if (swap) {
        buf_v[base + i] = vb; buf_v[base + ixj] = va;
        buf_i[base + i] = ib; buf_i[base + ixj] = ia;
    }
}

__kernel void bitonic_sort_extract_i64(
    __global uchar* out_v, __global long* out_i,
    __global const long* buf_v, __global const int* buf_i,
    int rows, int out_k, int P, uint type
) {
    const ulong gid = get_global_id(0);
    if (gid >= (ulong)rows * (ulong)out_k) return;
    const ulong row = gid / (ulong)out_k;
    const int col = (int)(gid - row * (ulong)out_k);
    const ulong src = row * (ulong)P + (ulong)col;
    write_integer(out_v, gid, type, buf_v[src]);
    out_i[gid] = (long)buf_i[src];
}
)CLC"
};

}  // namespace iris
