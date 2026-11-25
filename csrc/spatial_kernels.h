#pragma once

#include <array>

namespace iris {

inline constexpr char spatial_kernels[] = R"CLC(
enum SpatialDimension {
    BATCH, IN_CHANNELS, IN_HEIGHT, IN_WIDTH, OUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH,
    KERNEL_H, KERNEL_W, STRIDE_H, STRIDE_W, PAD_H, PAD_W, DILATION_H, DILATION_W, GROUPS
};

inline long image_index(long image, long channel, long y, long x, long channels, long height, long width) {
    return ((image * channels + channel) * height + y) * width + x;
}

inline long4 image_coordinates(long index, long channels, long height, long width) {
    return (long4)(index / (width * height * channels), index / (width * height) % channels,
                   index / width % height, index % width);
}

__kernel void conv_forward(__global const long* cfg, __global float* output, __global const float* input,
                           __global const float* weights, __global const float* bias,
                           __global const uchar* unused1, __global const uchar* unused2,
                           ulong out_offset, ulong in_offset, ulong w_offset, ulong bias_offset,
                           ulong unused3, ulong unused4) {
    const long pos = get_global_id(0);
    const long4 point = image_coordinates(pos, cfg[OUT_CHANNELS], cfg[OUT_HEIGHT], cfg[OUT_WIDTH]);
    const long group_width = cfg[IN_CHANNELS] / cfg[GROUPS];
    const long start_channel = point.y / (cfg[OUT_CHANNELS] / cfg[GROUPS]) * group_width;
    float sum = bias ? bias[bias_offset + point.y] : 0.0f;
    for (long channel = 0; channel < group_width; ++channel) {
        for (long ky = 0; ky < cfg[KERNEL_H]; ++ky) {
            const long y = point.z * cfg[STRIDE_H] - cfg[PAD_H] + ky * cfg[DILATION_H];
            for (long kx = 0; kx < cfg[KERNEL_W]; ++kx) {
                const long x = point.w * cfg[STRIDE_W] - cfg[PAD_W] + kx * cfg[DILATION_W];
                const long w = image_index(point.y, channel, ky, kx, group_width, cfg[KERNEL_H], cfg[KERNEL_W]);
                const bool inside = y >= 0 && y < cfg[IN_HEIGHT] && x >= 0 && x < cfg[IN_WIDTH];
                const float sample = inside ? input[in_offset + image_index(
                    point.x, start_channel + channel, y, x, cfg[IN_CHANNELS], cfg[IN_HEIGHT], cfg[IN_WIDTH])] : 0.0f;
                sum = fma(sample, weights[w_offset + w], sum);
            }
        }
    }
    output[out_offset + pos] = sum;
}

__kernel void conv_grad_input(__global const long* cfg, __global float* output, __global const float* grad,
                              __global const float* weights, __global const uchar* unused0,
                              __global const uchar* unused1, __global const uchar* unused2,
                              ulong out_offset, ulong grad_offset, ulong w_offset, ulong unused3,
                              ulong unused4, ulong unused5) {
    const long pos = get_global_id(0);
    const long4 point = image_coordinates(pos, cfg[IN_CHANNELS], cfg[IN_HEIGHT], cfg[IN_WIDTH]);
    const long in_group = cfg[IN_CHANNELS] / cfg[GROUPS];
    const long out_group = cfg[OUT_CHANNELS] / cfg[GROUPS];
    const long start_filter = point.y / in_group * out_group;
    float sum = 0.0f;
    for (long filter = start_filter; filter < start_filter + out_group; ++filter) {
        for (long ky = 0; ky < cfg[KERNEL_H]; ++ky) {
            const long y = point.z + cfg[PAD_H] - ky * cfg[DILATION_H];
            if (y < 0 || y % cfg[STRIDE_H] || y / cfg[STRIDE_H] >= cfg[OUT_HEIGHT]) continue;
            for (long kx = 0; kx < cfg[KERNEL_W]; ++kx) {
                const long x = point.w + cfg[PAD_W] - kx * cfg[DILATION_W];
                if (x < 0 || x % cfg[STRIDE_W] || x / cfg[STRIDE_W] >= cfg[OUT_WIDTH]) continue;
                const long w = image_index(filter, point.y % in_group, ky, kx, in_group, cfg[KERNEL_H], cfg[KERNEL_W]);
                const long g = image_index(point.x, filter, y / cfg[STRIDE_H], x / cfg[STRIDE_W],
                                           cfg[OUT_CHANNELS], cfg[OUT_HEIGHT], cfg[OUT_WIDTH]);
                sum = fma(grad[grad_offset + g], weights[w_offset + w], sum);
            }
        }
    }
    output[out_offset + pos] = sum;
}

__kernel void conv_grad_weight(__global const long* cfg, __global float* output, __global const float* grad,
                               __global const float* input, __global const uchar* unused0,
                               __global const uchar* unused1, __global const uchar* unused2,
                               ulong out_offset, ulong grad_offset, ulong in_offset, ulong unused3,
                               ulong unused4, ulong unused5) {
    const long pos = get_global_id(0);
    const long in_group = cfg[IN_CHANNELS] / cfg[GROUPS];
    const long4 point = image_coordinates(pos, in_group, cfg[KERNEL_H], cfg[KERNEL_W]);
    const long channel = point.x / (cfg[OUT_CHANNELS] / cfg[GROUPS]) * in_group + point.y;
    float sum = 0.0f;
    for (long image = 0; image < cfg[BATCH]; ++image) {
        for (long oy = 0; oy < cfg[OUT_HEIGHT]; ++oy) {
            const long y = oy * cfg[STRIDE_H] - cfg[PAD_H] + point.z * cfg[DILATION_H];
            for (long ox = 0; ox < cfg[OUT_WIDTH]; ++ox) {
                const long x = ox * cfg[STRIDE_W] - cfg[PAD_W] + point.w * cfg[DILATION_W];
                const bool inside = y >= 0 && y < cfg[IN_HEIGHT] && x >= 0 && x < cfg[IN_WIDTH];
                const float sample = inside ? input[in_offset + image_index(
                    image, channel, y, x, cfg[IN_CHANNELS], cfg[IN_HEIGHT], cfg[IN_WIDTH])] : 0.0f;
                const long g = image_index(image, point.x, oy, ox, cfg[OUT_CHANNELS], cfg[OUT_HEIGHT], cfg[OUT_WIDTH]);
                sum = fma(sample, grad[grad_offset + g], sum);
            }
        }
    }
    output[out_offset + pos] = sum;
}

enum PoolOption { AVERAGE = GROUPS, INCLUDE_PAD, DIVISOR };
inline long pool_divisor(__global const long* c, long y, long x) {
    if (c[DIVISOR]) return c[DIVISOR];
    const long pad_h = c[INCLUDE_PAD] ? c[PAD_H] : 0;
    const long pad_w = c[INCLUDE_PAD] ? c[PAD_W] : 0;
    const long height = min(y + c[KERNEL_H], c[IN_HEIGHT] + pad_h) - max(y, -pad_h);
    const long width = min(x + c[KERNEL_W], c[IN_WIDTH] + pad_w) - max(x, -pad_w);
    return height * width;
}

__kernel void pool_forward(__global const long* cfg, __global float* output, __global const float* input,
                           __global long* indices, __global const uchar* unused0,
                           __global const uchar* unused1, __global const uchar* unused2,
                           ulong out_offset, ulong in_offset, ulong index_offset, ulong unused3,
                           ulong unused4, ulong unused5) {
    const long pos = get_global_id(0);
    const long ox = pos % cfg[OUT_WIDTH];
    const long oy = pos / cfg[OUT_WIDTH] % cfg[OUT_HEIGHT];
    const long plane = pos / (cfg[OUT_WIDTH] * cfg[OUT_HEIGHT]);
    const long y0 = oy * cfg[STRIDE_H] - cfg[PAD_H];
    const long x0 = ox * cfg[STRIDE_W] - cfg[PAD_W];
    float value = cfg[AVERAGE] ? 0.0f : -INFINITY;
    long chosen = -1;
    for (long ky = 0; ky < cfg[KERNEL_H]; ++ky) {
        const long y = y0 + ky * cfg[DILATION_H];
        if (y < 0 || y >= cfg[IN_HEIGHT]) continue;
        for (long kx = 0; kx < cfg[KERNEL_W]; ++kx) {
            const long x = x0 + kx * cfg[DILATION_W];
            if (x < 0 || x >= cfg[IN_WIDTH]) continue;
            const long pixel = y * cfg[IN_WIDTH] + x;
            const float sample = input[in_offset + plane * cfg[IN_HEIGHT] * cfg[IN_WIDTH] + pixel];
            if (cfg[AVERAGE]) {
                value += sample;
            } else if (chosen < 0 || sample > value || isnan(sample)) {
                value = sample;
                chosen = pixel;
            }
        }
    }
    if (cfg[AVERAGE]) {
        value /= (float)pool_divisor(cfg, y0, x0);
    } else {
        indices[index_offset + pos] = chosen;
    }
    output[out_offset + pos] = value;
}

__kernel void pool_backward(__global const long* cfg, __global float* output, __global const float* grad,
                            __global const long* indices, __global const uchar* unused0,
                            __global const uchar* unused1, __global const uchar* unused2,
                            ulong out_offset, ulong grad_offset, ulong index_offset, ulong unused3,
                            ulong unused4, ulong unused5) {
    const long pos = get_global_id(0);
    const long area = cfg[IN_HEIGHT] * cfg[IN_WIDTH];
    const long pixel = pos % area;
    const long plane = pos / area;
    const long y = pixel / cfg[IN_WIDTH] + cfg[PAD_H];
    const long x = pixel % cfg[IN_WIDTH] + cfg[PAD_W];
    const long first_y = max(0L, (y - (cfg[KERNEL_H] - 1) * cfg[DILATION_H] + cfg[STRIDE_H] - 1) / cfg[STRIDE_H]);
    const long first_x = max(0L, (x - (cfg[KERNEL_W] - 1) * cfg[DILATION_W] + cfg[STRIDE_W] - 1) / cfg[STRIDE_W]);
    const long last_y = min(cfg[OUT_HEIGHT] - 1, y / cfg[STRIDE_H]);
    const long last_x = min(cfg[OUT_WIDTH] - 1, x / cfg[STRIDE_W]);
    float sum = 0.0f;
    for (long oy = first_y; oy <= last_y; ++oy) {
        for (long ox = first_x; ox <= last_x; ++ox) {
            const long i = (plane * cfg[OUT_HEIGHT] + oy) * cfg[OUT_WIDTH] + ox;
            const float value = grad[grad_offset + i];
            if (cfg[AVERAGE]) {
                sum += value / (float)pool_divisor(cfg, oy * cfg[STRIDE_H] - cfg[PAD_H],
                                                  ox * cfg[STRIDE_W] - cfg[PAD_W]);
            } else if (indices[index_offset + i] == pixel) {
                sum += value;
            }
        }
    }
    output[out_offset + pos] = sum;
}

inline float local_sum(__local float* values, uint lane, float value) {
    values[lane] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint half_width = get_local_size(0) / 2; half_width > 0; half_width /= 2) {
        if (lane < half_width) values[lane] += values[lane + half_width];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float total = values[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
}

__kernel void layer_norm_forward(__global const ulong* cfg, __global float* output,
                                  __global const float* input, __global const float* affine,
                                  __global const float* bias, __global float* mean, __global float* rstd,
                                  ulong out_offset, ulong in_offset, ulong w_offset, ulong b_offset,
                                  ulong mean_offset, ulong rstd_offset) {
    const ulong width = cfg[1];
    const ulong row = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const ulong start = row * width;
    __local float values[64];
    float sum = 0.0f;
    for (ulong j = lane; j < width; j += group) sum += input[in_offset + start + j];
    const float mu = local_sum(values, lane, sum) / convert_float(width);
    float squares = 0.0f;
    for (ulong j = lane; j < width; j += group) {
        const float diff = input[in_offset + start + j] - mu;
        squares = fma(diff, diff, squares);
    }
    const float inverse = rsqrt(local_sum(values, lane, squares) / convert_float(width) + as_float((uint)cfg[2]));
    if (lane == 0) {
        mean[mean_offset + row] = mu;
        rstd[rstd_offset + row] = inverse;
    }
    for (ulong j = lane; j < width; j += group) {
        const float value = (input[in_offset + start + j] - mu) * inverse;
        output[out_offset + start + j] = value * (affine ? affine[w_offset + j] : 1.0f) +
            (bias ? bias[b_offset + j] : 0.0f);
    }
}

__kernel void layer_norm_grad_input(__global const ulong* cfg, __global float* output,
                                     __global const float* grad, __global const float* input,
                                     __global const float* affine, __global const float* mean,
                                     __global const float* rstd, ulong out_offset, ulong grad_offset,
                                     ulong in_offset, ulong w_offset, ulong mean_offset, ulong rstd_offset) {
    const ulong width = cfg[1];
    const ulong row = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const ulong start = row * width;
    const float mu = mean[mean_offset + row];
    const float inverse = rstd[rstd_offset + row];
    __local float values[64];
    float sum = 0.0f;
    float dot = 0.0f;
    for (ulong j = lane; j < width; j += group) {
        const float normalized = (input[in_offset + start + j] - mu) * inverse;
        const float scaled = grad[grad_offset + start + j] * (affine ? affine[w_offset + j] : 1.0f);
        sum += scaled;
        dot = fma(scaled, normalized, dot);
    }
    const float avg = local_sum(values, lane, sum) / convert_float(width);
    const float avg_dot = local_sum(values, lane, dot) / convert_float(width);
    for (ulong j = lane; j < width; j += group) {
        const float normalized = (input[in_offset + start + j] - mu) * inverse;
        const float scaled = grad[grad_offset + start + j] * (affine ? affine[w_offset + j] : 1.0f);
        output[out_offset + start + j] = (scaled - avg - normalized * avg_dot) * inverse;
    }
}

__kernel void layer_norm_grad_affine(__global const ulong* cfg, __global float* grad_weight,
                                      __global float* grad_bias, __global const float* grad,
                                      __global const float* input, __global const float* mean,
                                      __global const float* rstd, ulong gw_offset, ulong gb_offset,
                                      ulong grad_offset, ulong in_offset, ulong mean_offset, ulong rstd_offset) {
    const ulong column = get_global_id(0);
    const ulong width = cfg[1];
    float bias = 0.0f;
    float weight = 0.0f;
    for (ulong row = 0; row < cfg[0]; ++row) {
        const ulong i = row * width + column;
        const float g = grad[grad_offset + i];
        bias += g;
        if (grad_weight) {
            const float normalized = (input[in_offset + i] - mean[mean_offset + row]) * rstd[rstd_offset + row];
            weight = fma(g, normalized, weight);
        }
    }
    if (grad_weight) grad_weight[gw_offset + column] = weight;
    if (grad_bias) grad_bias[gb_offset + column] = bias;
}
)CLC";

inline constexpr std::array attention_kernels{
R"CLC(
#if defined(IRIS_HAS_SUBGROUPS) && defined(IRIS_HAS_REQD_SG8)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

inline float sg16_reduce_max_nan(float x) {
    x = max_nan(x, intel_sub_group_shuffle_xor(x, 8));
    x = max_nan(x, intel_sub_group_shuffle_xor(x, 4));
    x = max_nan(x, intel_sub_group_shuffle_xor(x, 2));
    x = max_nan(x, intel_sub_group_shuffle_xor(x, 1));
    return x;
}

#define SDPA_LOAD_F32(ptr, idx) ((ptr)[(idx)])
#define SDPA_STORE_F32(ptr, idx, val) ((ptr)[(idx)] = (val))
#define SDPA_LOAD_F16(ptr, idx) vload_half((idx), (__global const half*)(ptr))
#define SDPA_STORE_F16(ptr, idx, val) vstore_half_rte((val), (idx), (__global half*)(ptr))

#define DEFINE_SDPA_FLASH(NAME, D_MAX, D_CHUNKS, BK, LOAD_FN, STORE_FN) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME( \
    __global float* output, __global float* lse_out, __global const float* query, \
    __global const float* key, __global const float* value, \
    ulong o_off, ulong lse_off, ulong q_off, ulong k_off, ulong v_off, \
    int B, int Hq, int Hkv, int Lq, int Lk, int Dq, int Dv, \
    ulong q_sb, ulong q_sh, ulong q_sl, ulong q_sd, \
    ulong k_sb, ulong k_sh, ulong k_sl, ulong k_sd, \
    ulong v_sb, ulong v_sh, ulong v_sl, ulong v_sd, \
    ulong o_sb, ulong o_sh, ulong o_sl, ulong o_sd, \
    float scale, int causal \
) { \
    const int lx = get_local_id(0); \
    const int ly = get_local_id(1); \
    const int tid = ly * 16 + lx; \
    const int q_base = get_group_id(0) * 8; \
    const int bh = get_group_id(1); \
    const int batch = bh / Hq; \
    const int head = bh - batch * Hq; \
    const int kv_head = head / (Hq / Hkv); \
    const int q_row = q_base + ly; \
    const bool q_valid = q_row < Lq; \
    const ulong q_bh = q_off + (ulong)batch * q_sb + (ulong)head * q_sh; \
    const ulong k_bh = k_off + (ulong)batch * k_sb + (ulong)kv_head * k_sh; \
    const ulong v_bh = v_off + (ulong)batch * v_sb + (ulong)kv_head * v_sh; \
    const ulong o_bh = o_off + (ulong)batch * o_sb + (ulong)head * o_sh; \
    float q_reg[D_CHUNKS]; \
    float o_reg[D_CHUNKS]; \
    const ulong q_row_off = q_bh + (ulong)q_row * q_sl; \
    _Pragma("unroll") \
    for (int r = 0; r < D_CHUNKS; ++r) { \
        const int d = r * 16 + lx; \
        q_reg[r] = (q_valid && d < Dq) ? LOAD_FN(query, q_row_off + (ulong)d * q_sd) : 0.0f; \
        o_reg[r] = 0.0f; \
    } \
    __local float k_tile[D_MAX][BK + 1]; \
    __local float v_tile[BK][D_MAX + 1]; \
    const int wg_k_end = causal ? min(Lk, q_base + 8) : Lk; \
    const int my_visible = causal ? min(Lk, q_row + 1) : Lk; \
    float m_acc = -INFINITY; \
    float l_acc = 0.0f; \
    for (int k0 = 0; k0 < wg_k_end; k0 += BK) { \
        for (int idx = tid; idx < BK * D_MAX; idx += 128) { \
            const int j = idx / D_MAX; \
            const int d = idx - j * D_MAX; \
            const int k_abs = k0 + j; \
            float kv = 0.0f; \
            float vv = 0.0f; \
            if (k_abs < Lk) { \
                if (d < Dq) kv = LOAD_FN(key, k_bh + (ulong)k_abs * k_sl + (ulong)d * k_sd); \
                if (d < Dv) vv = LOAD_FN(value, v_bh + (ulong)k_abs * v_sl + (ulong)d * v_sd); \
            } \
            k_tile[d][j] = kv; \
            v_tile[j][d] = vv; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (q_valid) { \
            float s0 = 0.0f; \
            float s1 = 0.0f; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) { \
                const int d_base = r * 16; \
                if (d_base < Dq) { \
                    _Pragma("unroll") \
                    for (int c = 0; c < 16; ++c) { \
                        const float qv = sub_group_broadcast(q_reg[r], c); \
                        s0 = fma(qv, k_tile[d_base + c][lx], s0); \
                        if (BK == 32) { \
                            s1 = fma(qv, k_tile[d_base + c][lx + 16], s1); \
                        } \
                    } \
                } \
            } \
            const bool v0 = (k0 + lx) < my_visible; \
            s0 = v0 ? (s0 * scale) : -INFINITY; \
            const bool v1 = (BK == 32) && ((k0 + 16 + lx) < my_visible); \
            s1 = v1 ? (s1 * scale) : -INFINITY; \
            const float local_max = (BK == 32) ? max_nan(s0, s1) : s0; \
            const float tile_max = sg16_reduce_max_nan(local_max); \
            const float m_new = max_nan(m_acc, tile_max); \
            const float rescale = (m_acc == m_new) ? 1.0f : ((m_acc == -INFINITY) ? 0.0f : exp(m_acc - m_new)); \
            const float p0 = v0 ? (m_new == -INFINITY ? 0.0f : exp(s0 - m_new)) : 0.0f; \
            const float p1 = v1 ? (m_new == -INFINITY ? 0.0f : exp(s1 - m_new)) : 0.0f; \
            const float p_sum = sub_group_reduce_add((BK == 32) ? (p0 + p1) : p0); \
            l_acc = l_acc * rescale + p_sum; \
            m_acc = m_new; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) { \
                o_reg[r] *= rescale; \
            } \
            _Pragma("unroll") \
            for (int j = 0; j < 16; ++j) { \
                const float pv = sub_group_broadcast(p0, j); \
                _Pragma("unroll") \
                for (int r = 0; r < D_CHUNKS; ++r) { \
                    o_reg[r] = fma(pv, v_tile[j][r * 16 + lx], o_reg[r]); \
                } \
            } \
            if (BK == 32) { \
                _Pragma("unroll") \
                for (int j = 0; j < 16; ++j) { \
                    const float pv = sub_group_broadcast(p1, j); \
                    _Pragma("unroll") \
                    for (int r = 0; r < D_CHUNKS; ++r) { \
                        o_reg[r] = fma(pv, v_tile[16 + j][r * 16 + lx], o_reg[r]); \
                    } \
                } \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (q_valid) { \
        const float inv_l = (l_acc == 0.0f) ? 0.0f : (1.0f / l_acc); \
        const ulong o_row_off = o_bh + (ulong)q_row * o_sl; \
        _Pragma("unroll") \
        for (int r = 0; r < D_CHUNKS; ++r) { \
            const int d = r * 16 + lx; \
            if (d < Dv) { \
                STORE_FN(output, o_row_off + (ulong)d * o_sd, o_reg[r] * inv_l); \
            } \
        } \
        if (lse_out && lx == 0) { \
            lse_out[lse_off + (ulong)bh * Lq + q_row] = (l_acc == 0.0f) ? -INFINITY : (m_acc + log(l_acc)); \
        } \
    } \
}

DEFINE_SDPA_FLASH(sdpa_flash_d32, 32, 2, 32, SDPA_LOAD_F32, SDPA_STORE_F32)
DEFINE_SDPA_FLASH(sdpa_flash_d64, 64, 4, 16, SDPA_LOAD_F32, SDPA_STORE_F32)
DEFINE_SDPA_FLASH(sdpa_flash_d128, 128, 8, 16, SDPA_LOAD_F32, SDPA_STORE_F32)
#ifdef IRIS_HAS_FP16
DEFINE_SDPA_FLASH(sdpa_flash_d32_fp16, 32, 2, 32, SDPA_LOAD_F16, SDPA_STORE_F16)
DEFINE_SDPA_FLASH(sdpa_flash_d64_fp16, 64, 4, 16, SDPA_LOAD_F16, SDPA_STORE_F16)
DEFINE_SDPA_FLASH(sdpa_flash_d128_fp16, 128, 8, 16, SDPA_LOAD_F16, SDPA_STORE_F16)
#endif
)CLC",
R"CLC(
// High-throughput Q32 FlashAttention (4 query rows/subgroup, 32 query rows/WG, BK=16, reused kv_tile SLM)
#define DEFINE_SDPA_FLASH_Q32(NAME, D_MAX, D_CHUNKS, LOAD_FN, STORE_FN, IS_FP16) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME( \
    __global float* output, __global float* lse_out, __global const float* query, \
    __global const float* key, __global const float* value, \
    ulong o_off, ulong lse_off, ulong q_off, ulong k_off, ulong v_off, \
    int B, int Hq, int Hkv, int Lq, int Lk, int Dq, int Dv, \
    ulong q_sb, ulong q_sh, ulong q_sl, ulong q_sd, \
    ulong k_sb, ulong k_sh, ulong k_sl, ulong k_sd, \
    ulong v_sb, ulong v_sh, ulong v_sl, ulong v_sd, \
    ulong o_sb, ulong o_sh, ulong o_sl, ulong o_sd, \
    float scale, int causal \
) { \
    const int lx = get_local_id(0); \
    const int ly = get_local_id(1); \
    const int tid = ly * 16 + lx; \
    const int q_base = get_group_id(0) * 32; \
    const int bh = get_group_id(1); \
    const int batch = bh / Hq; \
    const int head = bh - batch * Hq; \
    const int kv_head = head / (Hq / Hkv); \
    const ulong q_bh = q_off + (ulong)batch * q_sb + (ulong)head * q_sh; \
    const ulong k_bh = k_off + (ulong)batch * k_sb + (ulong)kv_head * k_sh; \
    const ulong v_bh = v_off + (ulong)batch * v_sb + (ulong)kv_head * v_sh; \
    const ulong o_bh = o_off + (ulong)batch * o_sb + (ulong)head * o_sh; \
    __local float q_tile[32][D_MAX + 1]; \
    __local float kv_tile[16][D_MAX + 1]; \
    const bool q_vec = (q_sd == 1) && ((Dq & 3) == 0); \
    const bool k_vec = (k_sd == 1) && ((Dq & 3) == 0); \
    const bool v_vec = (v_sd == 1) && ((Dv & 3) == 0); \
    for (int idx = tid; idx < 32 * (D_MAX / 4); idx += 128) { \
        const int qr = idx / (D_MAX / 4); \
        const int d4 = (idx - qr * (D_MAX / 4)) * 4; \
        const int q_abs = q_base + qr; \
        float4 qv = (float4)(0.0f); \
        if (q_abs < Lq) { \
            const ulong row_p = q_bh + (ulong)q_abs * q_sl; \
            if (q_vec && d4 + 3 < Dq) { \
                qv = IS_FP16 ? vload_half4(0, ((__global const half*)query) + row_p + d4) \
                             : vload4(0, query + row_p + d4); \
            } else { \
                if (d4 + 0 < Dq) qv.s0 = LOAD_FN(query, row_p + (ulong)(d4 + 0) * q_sd); \
                if (d4 + 1 < Dq) qv.s1 = LOAD_FN(query, row_p + (ulong)(d4 + 1) * q_sd); \
                if (d4 + 2 < Dq) qv.s2 = LOAD_FN(query, row_p + (ulong)(d4 + 2) * q_sd); \
                if (d4 + 3 < Dq) qv.s3 = LOAD_FN(query, row_p + (ulong)(d4 + 3) * q_sd); \
            } \
        } \
        q_tile[qr][d4 + 0] = qv.s0; q_tile[qr][d4 + 1] = qv.s1; \
        q_tile[qr][d4 + 2] = qv.s2; q_tile[qr][d4 + 3] = qv.s3; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float o_reg[4][D_CHUNKS]; \
    float m_acc[4]; \
    float l_acc[4]; \
    _Pragma("unroll") \
    for (int qi = 0; qi < 4; ++qi) { \
        m_acc[qi] = -INFINITY; \
        l_acc[qi] = 0.0f; \
        _Pragma("unroll") \
        for (int r = 0; r < D_CHUNKS; ++r) o_reg[qi][r] = 0.0f; \
    } \
    const int qr0 = ly * 4; \
    const int wg_k_end = causal ? min(Lk, q_base + 32) : Lk; \
    const int sg_k_end = causal ? min(Lk, q_base + qr0 + 4) : Lk; \
    for (int k0 = 0; k0 < wg_k_end; k0 += 16) { \
        for (int idx = tid; idx < 16 * (D_MAX / 4); idx += 128) { \
            const int j = idx / (D_MAX / 4); \
            const int d4 = (idx - j * (D_MAX / 4)) * 4; \
            const int k_abs = k0 + j; \
            float4 kv = (float4)(0.0f); \
            if (k_abs < Lk) { \
                const ulong row_p = k_bh + (ulong)k_abs * k_sl; \
                if (k_vec && d4 + 3 < Dq) { \
                    kv = IS_FP16 ? vload_half4(0, ((__global const half*)key) + row_p + d4) \
                                 : vload4(0, key + row_p + d4); \
                } else { \
                    if (d4 + 0 < Dq) kv.s0 = LOAD_FN(key, row_p + (ulong)(d4 + 0) * k_sd); \
                    if (d4 + 1 < Dq) kv.s1 = LOAD_FN(key, row_p + (ulong)(d4 + 1) * k_sd); \
                    if (d4 + 2 < Dq) kv.s2 = LOAD_FN(key, row_p + (ulong)(d4 + 2) * k_sd); \
                    if (d4 + 3 < Dq) kv.s3 = LOAD_FN(key, row_p + (ulong)(d4 + 3) * k_sd); \
                } \
            } \
            kv_tile[j][d4 + 0] = kv.s0; kv_tile[j][d4 + 1] = kv.s1; \
            kv_tile[j][d4 + 2] = kv.s2; kv_tile[j][d4 + 3] = kv.s3; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f; \
        const bool sg_active = (q_base + qr0 < Lq) && (k0 < sg_k_end); \
        if (sg_active) { \
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f; \
            int d = 0; \
            for (; d + 3 < Dq; d += 4) { \
                const float k_0 = kv_tile[lx][d + 0]; \
                const float k_1 = kv_tile[lx][d + 1]; \
                const float k_2 = kv_tile[lx][d + 2]; \
                const float k_3 = kv_tile[lx][d + 3]; \
                s0 = fma(q_tile[qr0 + 0][d + 0], k_0, fma(q_tile[qr0 + 0][d + 1], k_1, fma(q_tile[qr0 + 0][d + 2], k_2, fma(q_tile[qr0 + 0][d + 3], k_3, s0)))); \
                s1 = fma(q_tile[qr0 + 1][d + 0], k_0, fma(q_tile[qr0 + 1][d + 1], k_1, fma(q_tile[qr0 + 1][d + 2], k_2, fma(q_tile[qr0 + 1][d + 3], k_3, s1)))); \
                s2 = fma(q_tile[qr0 + 2][d + 0], k_0, fma(q_tile[qr0 + 2][d + 1], k_1, fma(q_tile[qr0 + 2][d + 2], k_2, fma(q_tile[qr0 + 2][d + 3], k_3, s2)))); \
                s3 = fma(q_tile[qr0 + 3][d + 0], k_0, fma(q_tile[qr0 + 3][d + 1], k_1, fma(q_tile[qr0 + 3][d + 2], k_2, fma(q_tile[qr0 + 3][d + 3], k_3, s3)))); \
            } \
            for (; d < Dq; ++d) { \
                const float k_v = kv_tile[lx][d]; \
                s0 = fma(q_tile[qr0 + 0][d], k_v, s0); \
                s1 = fma(q_tile[qr0 + 1][d], k_v, s1); \
                s2 = fma(q_tile[qr0 + 2][d], k_v, s2); \
                s3 = fma(q_tile[qr0 + 3][d], k_v, s3); \
            } \
            float sv[4] = {s0, s1, s2, s3}; \
            float pv[4]; \
            _Pragma("unroll") \
            for (int qi = 0; qi < 4; ++qi) { \
                const int q_abs = q_base + qr0 + qi; \
                const int vis = causal ? min(Lk, q_abs + 1) : Lk; \
                const bool ok = (q_abs < Lq) && ((k0 + lx) < vis); \
                const float sc = ok ? (sv[qi] * scale) : -INFINITY; \
                const float tmax = sg16_reduce_max_nan(sc); \
                const float m_new = max_nan(m_acc[qi], tmax); \
                const float resc = (m_acc[qi] == m_new) ? 1.0f : ((m_acc[qi] == -INFINITY) ? 0.0f : exp(m_acc[qi] - m_new)); \
                const float p = ok ? (m_new == -INFINITY ? 0.0f : exp(sc - m_new)) : 0.0f; \
                l_acc[qi] = l_acc[qi] * resc + sub_group_reduce_add(p); \
                m_acc[qi] = m_new; \
                pv[qi] = p; \
                _Pragma("unroll") \
                for (int r = 0; r < D_CHUNKS; ++r) o_reg[qi][r] *= resc; \
            } \
            p0 = pv[0]; p1 = pv[1]; p2 = pv[2]; p3 = pv[3]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        for (int idx = tid; idx < 16 * (D_MAX / 4); idx += 128) { \
            const int j = idx / (D_MAX / 4); \
            const int d4 = (idx - j * (D_MAX / 4)) * 4; \
            const int k_abs = k0 + j; \
            float4 vv = (float4)(0.0f); \
            if (k_abs < Lk) { \
                const ulong row_p = v_bh + (ulong)k_abs * v_sl; \
                if (v_vec && d4 + 3 < Dv) { \
                    vv = IS_FP16 ? vload_half4(0, ((__global const half*)value) + row_p + d4) \
                                 : vload4(0, value + row_p + d4); \
                } else { \
                    if (d4 + 0 < Dv) vv.s0 = LOAD_FN(value, row_p + (ulong)(d4 + 0) * v_sd); \
                    if (d4 + 1 < Dv) vv.s1 = LOAD_FN(value, row_p + (ulong)(d4 + 1) * v_sd); \
                    if (d4 + 2 < Dv) vv.s2 = LOAD_FN(value, row_p + (ulong)(d4 + 2) * v_sd); \
                    if (d4 + 3 < Dv) vv.s3 = LOAD_FN(value, row_p + (ulong)(d4 + 3) * v_sd); \
                } \
            } \
            kv_tile[j][d4 + 0] = vv.s0; kv_tile[j][d4 + 1] = vv.s1; \
            kv_tile[j][d4 + 2] = vv.s2; kv_tile[j][d4 + 3] = vv.s3; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (sg_active) { \
            _Pragma("unroll") \
            for (int j = 0; j < 16; ++j) { \
                const float w0 = sub_group_broadcast(p0, j); \
                const float w1 = sub_group_broadcast(p1, j); \
                const float w2 = sub_group_broadcast(p2, j); \
                const float w3 = sub_group_broadcast(p3, j); \
                _Pragma("unroll") \
                for (int r = 0; r < D_CHUNKS; ++r) { \
                    const float vv = kv_tile[j][r * 16 + lx]; \
                    o_reg[0][r] = fma(w0, vv, o_reg[0][r]); \
                    o_reg[1][r] = fma(w1, vv, o_reg[1][r]); \
                    o_reg[2][r] = fma(w2, vv, o_reg[2][r]); \
                    o_reg[3][r] = fma(w3, vv, o_reg[3][r]); \
                } \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    _Pragma("unroll") \
    for (int qi = 0; qi < 4; ++qi) { \
        const int q_abs = q_base + qr0 + qi; \
        if (q_abs < Lq) { \
            const float inv_l = (l_acc[qi] == 0.0f) ? 0.0f : (1.0f / l_acc[qi]); \
            const ulong o_row_off = o_bh + (ulong)q_abs * o_sl; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) { \
                const int d = r * 16 + lx; \
                if (d < Dv) STORE_FN(output, o_row_off + (ulong)d * o_sd, o_reg[qi][r] * inv_l); \
            } \
            if (lse_out && lx == 0) { \
                lse_out[lse_off + (ulong)bh * Lq + q_abs] = (l_acc[qi] == 0.0f) ? -INFINITY : (m_acc[qi] + log(l_acc[qi])); \
            } \
        } \
    } \
}

DEFINE_SDPA_FLASH_Q32(sdpa_flash_d64_q32, 64, 4, SDPA_LOAD_F32, SDPA_STORE_F32, 0)
DEFINE_SDPA_FLASH_Q32(sdpa_flash_d128_q32, 128, 8, SDPA_LOAD_F32, SDPA_STORE_F32, 0)
#ifdef IRIS_HAS_FP16
DEFINE_SDPA_FLASH_Q32(sdpa_flash_d64_q32_fp16, 64, 4, SDPA_LOAD_F16, SDPA_STORE_F16, 1)
DEFINE_SDPA_FLASH_Q32(sdpa_flash_d128_q32_fp16, 128, 8, SDPA_LOAD_F16, SDPA_STORE_F16, 1)
#endif
)CLC",
R"CLC(
// SIMD8 high-throughput FlashAttention for D <= 128 (WG=(8,16)=128 threads, BQ=64, BK=32, 8KB SLM, vload4 SLM, zero softmax barriers)
inline float sg8_reduce_max_nan(float x) {
    x = max_nan(x, intel_sub_group_shuffle_xor(x, 4));
    x = max_nan(x, intel_sub_group_shuffle_xor(x, 2));
    x = max_nan(x, intel_sub_group_shuffle_xor(x, 1));
    return x;
}

#define SG8_SOFTMAX_ROW(ROW_IDX, SVEC, PVEC, MVAR, LVAR, O0, O1, O2, O3) do { \
    const int qa = q_base + qr0 + (ROW_IDX); \
    const int vis = causal ? min(Lk, qa + 1) : Lk; \
    const bool qok = qa < Lq; \
    (SVEC).s0 = (qok && k_lane + 0 < vis) ? ((SVEC).s0 * scale) : -INFINITY; \
    (SVEC).s1 = (qok && k_lane + 1 < vis) ? ((SVEC).s1 * scale) : -INFINITY; \
    (SVEC).s2 = (qok && k_lane + 2 < vis) ? ((SVEC).s2 * scale) : -INFINITY; \
    (SVEC).s3 = (qok && k_lane + 3 < vis) ? ((SVEC).s3 * scale) : -INFINITY; \
    const float tmax = sg8_reduce_max_nan(max_nan(max_nan((SVEC).s0, (SVEC).s1), max_nan((SVEC).s2, (SVEC).s3))); \
    const float mn = max_nan(MVAR, tmax); \
    const float rc = (MVAR == mn) ? 1.0f : ((MVAR == -INFINITY) ? 0.0f : exp(MVAR - mn)); \
    (PVEC).s0 = (qok && k_lane + 0 < vis && mn != -INFINITY) ? exp((SVEC).s0 - mn) : 0.0f; \
    (PVEC).s1 = (qok && k_lane + 1 < vis && mn != -INFINITY) ? exp((SVEC).s1 - mn) : 0.0f; \
    (PVEC).s2 = (qok && k_lane + 2 < vis && mn != -INFINITY) ? exp((SVEC).s2 - mn) : 0.0f; \
    (PVEC).s3 = (qok && k_lane + 3 < vis && mn != -INFINITY) ? exp((SVEC).s3 - mn) : 0.0f; \
    LVAR = LVAR * rc + sub_group_reduce_add((PVEC).s0 + (PVEC).s1 + (PVEC).s2 + (PVEC).s3); \
    MVAR = mn; \
    if (rc != 1.0f) { O0 *= rc; O1 *= rc; O2 *= rc; O3 *= rc; } \
} while (0)

#define SG8_PV_STEP(VB, W0C, W1C, W2C, W3C) do { \
    const float4 v0 = vload4(0, &smem[(VB) + 0]); const float4 v1 = vload4(0, &smem[(VB) + 32]); \
    const float4 v2 = vload4(0, &smem[(VB) + 64]); const float4 v3 = vload4(0, &smem[(VB) + 96]); \
    o00 = fma(W0C, v0, o00); o01 = fma(W0C, v1, o01); o02 = fma(W0C, v2, o02); o03 = fma(W0C, v3, o03); \
    o10 = fma(W1C, v0, o10); o11 = fma(W1C, v1, o11); o12 = fma(W1C, v2, o12); o13 = fma(W1C, v3, o13); \
    o20 = fma(W2C, v0, o20); o21 = fma(W2C, v1, o21); o22 = fma(W2C, v2, o22); o23 = fma(W2C, v3, o23); \
    o30 = fma(W3C, v0, o30); o31 = fma(W3C, v1, o31); o32 = fma(W3C, v2, o32); o33 = fma(W3C, v3, o33); \
} while (0)

#define DEFINE_SDPA_FLASH_D128_SG8(NAME, LOAD_FN, STORE_FN, IS_FP16) \
__attribute__((intel_reqd_sub_group_size(8))) \
__kernel void NAME( \
    __global float* output, __global float* lse_out, __global const float* query, \
    __global const float* key, __global const float* value, \
    ulong o_off, ulong lse_off, ulong q_off, ulong k_off, ulong v_off, \
    int B, int Hq, int Hkv, int Lq, int Lk, int Dq, int Dv, \
    ulong q_sb, ulong q_sh, ulong q_sl, ulong q_sd, \
    ulong k_sb, ulong k_sh, ulong k_sl, ulong k_sd, \
    ulong v_sb, ulong v_sh, ulong v_sl, ulong v_sd, \
    ulong o_sb, ulong o_sh, ulong o_sl, ulong o_sd, \
    float scale, int causal \
) { \
    const int lx = get_local_id(0); \
    const int ly = get_local_id(1); \
    const int tid = ly * 8 + lx; \
    const int q_base = get_group_id(0) * 64; \
    const int bh = get_group_id(1); \
    const int batch = bh / Hq; \
    const int head = bh - batch * Hq; \
    const int kv_head = head / (Hq / Hkv); \
    const ulong q_bh = q_off + (ulong)batch * q_sb + (ulong)head * q_sh; \
    const ulong k_bh = k_off + (ulong)batch * k_sb + (ulong)kv_head * k_sh; \
    const ulong v_bh = v_off + (ulong)batch * v_sb + (ulong)kv_head * v_sh; \
    const ulong o_bh = o_off + (ulong)batch * o_sb + (ulong)head * o_sh; \
    __local float smem[2048]; \
    const bool q_vec = (q_sd == 1) && ((Dq & 3) == 0); \
    const bool k_vec = (k_sd == 1) && ((Dq & 3) == 0); \
    const bool v_vec = (v_sd == 1) && ((Dv & 3) == 0); \
    float4 o00 = (float4)(0.0f), o01 = (float4)(0.0f), o02 = (float4)(0.0f), o03 = (float4)(0.0f); \
    float4 o10 = (float4)(0.0f), o11 = (float4)(0.0f), o12 = (float4)(0.0f), o13 = (float4)(0.0f); \
    float4 o20 = (float4)(0.0f), o21 = (float4)(0.0f), o22 = (float4)(0.0f), o23 = (float4)(0.0f); \
    float4 o30 = (float4)(0.0f), o31 = (float4)(0.0f), o32 = (float4)(0.0f), o33 = (float4)(0.0f); \
    float m0 = -INFINITY, m1 = -INFINITY, m2 = -INFINITY, m3 = -INFINITY; \
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f; \
    const int qr0 = ly * 4; \
    const int wg_k_end = causal ? min(Lk, q_base + 64) : Lk; \
    const int sg_k_end = causal ? min(Lk, q_base + qr0 + 4) : Lk; \
    const int tid_row = tid >> 2; \
    const int tid_d4 = (tid & 3) << 2; \
    const int v_jj0 = tid >> 5; \
    const int v_d4 = (tid & 31) << 2; \
    for (int k0 = 0; k0 < wg_k_end; k0 += 32) { \
        const bool sg_active = (q_base + qr0 < Lq) && (k0 < sg_k_end); \
        float4 s0 = (float4)(0.0f), s1 = (float4)(0.0f), s2 = (float4)(0.0f), s3 = (float4)(0.0f); \
        for (int d0 = 0; d0 < Dq; d0 += 16) { \
            const int d_abs = d0 + tid_d4; \
            for (int p = 0; p < 2; ++p) { \
                const int qr = tid_row + p * 32; \
                const int q_abs = q_base + qr; \
                float4 qv = (float4)(0.0f); \
                if (q_abs < Lq && d_abs < Dq) { \
                    const ulong rp = q_bh + (ulong)q_abs * q_sl; \
                    if (q_vec && d_abs + 3 < Dq) { \
                        qv = IS_FP16 ? vload_half4(0, ((__global const half*)query) + rp + d_abs) : vload4(0, query + rp + d_abs); \
                    } else { \
                        if (d_abs + 0 < Dq) qv.s0 = LOAD_FN(query, rp + (ulong)(d_abs + 0) * q_sd); \
                        if (d_abs + 1 < Dq) qv.s1 = LOAD_FN(query, rp + (ulong)(d_abs + 1) * q_sd); \
                        if (d_abs + 2 < Dq) qv.s2 = LOAD_FN(query, rp + (ulong)(d_abs + 2) * q_sd); \
                        if (d_abs + 3 < Dq) qv.s3 = LOAD_FN(query, rp + (ulong)(d_abs + 3) * q_sd); \
                    } \
                } \
                smem[(tid_d4 + 0) * 65 + qr] = qv.s0; \
                smem[(tid_d4 + 1) * 65 + qr] = qv.s1; \
                smem[(tid_d4 + 2) * 65 + qr] = qv.s2; \
                smem[(tid_d4 + 3) * 65 + qr] = qv.s3; \
            } \
            { \
                const int j = tid_row; \
                const int k_abs = k0 + j; \
                float4 kv = (float4)(0.0f); \
                if (k_abs < Lk && d_abs < Dq) { \
                    const ulong rp = k_bh + (ulong)k_abs * k_sl; \
                    if (k_vec && d_abs + 3 < Dq) { \
                        kv = IS_FP16 ? vload_half4(0, ((__global const half*)key) + rp + d_abs) : vload4(0, key + rp + d_abs); \
                    } else { \
                        if (d_abs + 0 < Dq) kv.s0 = LOAD_FN(key, rp + (ulong)(d_abs + 0) * k_sd); \
                        if (d_abs + 1 < Dq) kv.s1 = LOAD_FN(key, rp + (ulong)(d_abs + 1) * k_sd); \
                        if (d_abs + 2 < Dq) kv.s2 = LOAD_FN(key, rp + (ulong)(d_abs + 2) * k_sd); \
                        if (d_abs + 3 < Dq) kv.s3 = LOAD_FN(key, rp + (ulong)(d_abs + 3) * k_sd); \
                    } \
                } \
                smem[1040 + (tid_d4 + 0) * 33 + j] = kv.s0; \
                smem[1040 + (tid_d4 + 1) * 33 + j] = kv.s1; \
                smem[1040 + (tid_d4 + 2) * 33 + j] = kv.s2; \
                smem[1040 + (tid_d4 + 3) * 33 + j] = kv.s3; \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
            if (sg_active) { \
                _Pragma("unroll") \
                for (int d = 0; d < 16; ++d) { \
                    const float4 kv = vload4(0, &smem[1040 + d * 33 + lx * 4]); \
                    const int qb = d * 65 + qr0; \
                    s0 = fma(smem[qb + 0], kv, s0); \
                    s1 = fma(smem[qb + 1], kv, s1); \
                    s2 = fma(smem[qb + 2], kv, s2); \
                    s3 = fma(smem[qb + 3], kv, s3); \
                } \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
        float4 p0 = (float4)(0.0f), p1 = (float4)(0.0f), p2 = (float4)(0.0f), p3 = (float4)(0.0f); \
        if (sg_active) { \
            const int k_lane = k0 + lx * 4; \
            SG8_SOFTMAX_ROW(0, s0, p0, m0, l0, o00, o01, o02, o03); \
            SG8_SOFTMAX_ROW(1, s1, p1, m1, l1, o10, o11, o12, o13); \
            SG8_SOFTMAX_ROW(2, s2, p2, m2, l2, o20, o21, o22, o23); \
            SG8_SOFTMAX_ROW(3, s3, p3, m3, l3, o30, o31, o32, o33); \
        } \
        for (int half_idx = 0; half_idx < 2; ++half_idx) { \
            const int j0 = half_idx * 16; \
            for (int p = 0; p < 4; ++p) { \
                const int jj = v_jj0 + p * 4; \
                const int k_abs = k0 + j0 + jj; \
                float4 vv = (float4)(0.0f); \
                if (k_abs < Lk && v_d4 < Dv) { \
                    const ulong rp = v_bh + (ulong)k_abs * v_sl; \
                    if (v_vec && v_d4 + 3 < Dv) { \
                        vv = IS_FP16 ? vload_half4(0, ((__global const half*)value) + rp + v_d4) : vload4(0, value + rp + v_d4); \
                    } else { \
                        if (v_d4 + 0 < Dv) vv.s0 = LOAD_FN(value, rp + (ulong)(v_d4 + 0) * v_sd); \
                        if (v_d4 + 1 < Dv) vv.s1 = LOAD_FN(value, rp + (ulong)(v_d4 + 1) * v_sd); \
                        if (v_d4 + 2 < Dv) vv.s2 = LOAD_FN(value, rp + (ulong)(v_d4 + 2) * v_sd); \
                        if (v_d4 + 3 < Dv) vv.s3 = LOAD_FN(value, rp + (ulong)(v_d4 + 3) * v_sd); \
                    } \
                } \
                vstore4(vv, 0, &smem[jj * 128 + v_d4]); \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
            if (sg_active) { \
                const int lane0 = half_idx * 4; \
                for (int dl = 0; dl < 4; ++dl) { \
                    const int sl = lane0 + dl; \
                    const float4 w0 = (float4)(sub_group_broadcast(p0.s0, sl), sub_group_broadcast(p0.s1, sl), sub_group_broadcast(p0.s2, sl), sub_group_broadcast(p0.s3, sl)); \
                    const float4 w1 = (float4)(sub_group_broadcast(p1.s0, sl), sub_group_broadcast(p1.s1, sl), sub_group_broadcast(p1.s2, sl), sub_group_broadcast(p1.s3, sl)); \
                    const float4 w2 = (float4)(sub_group_broadcast(p2.s0, sl), sub_group_broadcast(p2.s1, sl), sub_group_broadcast(p2.s2, sl), sub_group_broadcast(p2.s3, sl)); \
                    const float4 w3 = (float4)(sub_group_broadcast(p3.s0, sl), sub_group_broadcast(p3.s1, sl), sub_group_broadcast(p3.s2, sl), sub_group_broadcast(p3.s3, sl)); \
                    SG8_PV_STEP((dl * 4 + 0) * 128 + lx * 4, w0.s0, w1.s0, w2.s0, w3.s0); \
                    SG8_PV_STEP((dl * 4 + 1) * 128 + lx * 4, w0.s1, w1.s1, w2.s1, w3.s1); \
                    SG8_PV_STEP((dl * 4 + 2) * 128 + lx * 4, w0.s2, w1.s2, w2.s2, w3.s2); \
                    SG8_PV_STEP((dl * 4 + 3) * 128 + lx * 4, w0.s3, w1.s3, w2.s3, w3.s3); \
                } \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    } \
    const float inv_l[4] = { \
        (l0 == 0.0f) ? 0.0f : (1.0f / l0), (l1 == 0.0f) ? 0.0f : (1.0f / l1), \
        (l2 == 0.0f) ? 0.0f : (1.0f / l2), (l3 == 0.0f) ? 0.0f : (1.0f / l3) \
    }; \
    const float m_arr[4] = {m0, m1, m2, m3}; \
    const float l_arr[4] = {l0, l1, l2, l3}; \
    float4 o_rows[4][4] = { \
        {o00, o01, o02, o03}, {o10, o11, o12, o13}, \
        {o20, o21, o22, o23}, {o30, o31, o32, o33} \
    }; \
    _Pragma("unroll") \
    for (int qi = 0; qi < 4; ++qi) { \
        const int qa = q_base + qr0 + qi; \
        if (qa < Lq) { \
            const ulong o_row_off = o_bh + (ulong)qa * o_sl; \
            const float sc = inv_l[qi]; \
            _Pragma("unroll") \
            for (int blk = 0; blk < 4; ++blk) { \
                const int d0 = blk * 32 + lx * 4; \
                const float4 ov = o_rows[qi][blk] * sc; \
                if (o_sd == 1 && d0 + 3 < Dv) { \
                    if (IS_FP16) vstore_half4_rte(ov, 0, ((__global half*)output) + o_row_off + d0); \
                    else vstore4(ov, 0, output + o_row_off + d0); \
                } else { \
                    if (d0 + 0 < Dv) STORE_FN(output, o_row_off + (ulong)(d0 + 0) * o_sd, ov.s0); \
                    if (d0 + 1 < Dv) STORE_FN(output, o_row_off + (ulong)(d0 + 1) * o_sd, ov.s1); \
                    if (d0 + 2 < Dv) STORE_FN(output, o_row_off + (ulong)(d0 + 2) * o_sd, ov.s2); \
                    if (d0 + 3 < Dv) STORE_FN(output, o_row_off + (ulong)(d0 + 3) * o_sd, ov.s3); \
                } \
            } \
            if (lse_out && lx == 0) { \
                lse_out[lse_off + (ulong)bh * Lq + qa] = (l_arr[qi] == 0.0f) ? -INFINITY : (m_arr[qi] + log(l_arr[qi])); \
            } \
        } \
    } \
}

DEFINE_SDPA_FLASH_D128_SG8(sdpa_flash_d128_sg8, SDPA_LOAD_F32, SDPA_STORE_F32, 0)
#ifdef IRIS_HAS_FP16
DEFINE_SDPA_FLASH_D128_SG8(sdpa_flash_d128_sg8_fp16, SDPA_LOAD_F16, SDPA_STORE_F16, 1)
#endif
)CLC",
R"CLC(
// Split-KV FlashDecoding for Lq <= 4 and long Lk KV-cache (all 8 subgroups cooperate per split)
#define DEFINE_SDPA_DECODE(NAME, D_MAX, D_CHUNKS, LOAD_FN, STORE_FN) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME( \
    __global float* output, __global float* partial_o, __global float* partial_lse, \
    __global const float* query, __global const float* key, __global const float* value, \
    ulong o_off, ulong q_off, ulong k_off, ulong v_off, \
    int B, int Hq, int Hkv, int Lq, int Lk, int Dq, int Dv, \
    ulong q_sb, ulong q_sh, ulong q_sl, ulong q_sd, \
    ulong k_sb, ulong k_sh, ulong k_sl, ulong k_sd, \
    ulong v_sb, ulong v_sh, ulong v_sl, ulong v_sd, \
    ulong o_sb, ulong o_sh, ulong o_sl, ulong o_sd, \
    float scale, int causal, int num_splits \
) { \
    const int lx = get_local_id(0); \
    const int ly = get_local_id(1); \
    const int split_id = get_group_id(0); \
    const int q_row = get_group_id(1); \
    const int bh = get_group_id(2); \
    const int batch = bh / Hq; \
    const int head = bh - batch * Hq; \
    const int kv_head = head / (Hq / Hkv); \
    const ulong q_row_off = q_off + (ulong)batch * q_sb + (ulong)head * q_sh + (ulong)q_row * q_sl; \
    const ulong k_bh = k_off + (ulong)batch * k_sb + (ulong)kv_head * k_sh; \
    const ulong v_bh = v_off + (ulong)batch * v_sb + (ulong)kv_head * v_sh; \
    float q_reg[D_CHUNKS]; \
    float o_reg[D_CHUNKS]; \
    _Pragma("unroll") \
    for (int r = 0; r < D_CHUNKS; ++r) { \
        const int d = r * 16 + lx; \
        q_reg[r] = (d < Dq) ? LOAD_FN(query, q_row_off + (ulong)d * q_sd) : 0.0f; \
        o_reg[r] = 0.0f; \
    } \
    const int my_visible = causal ? min(Lk, q_row + 1) : Lk; \
    const int chunk = (my_visible + num_splits - 1) / num_splits; \
    const int k_start = split_id * chunk; \
    const int k_end = min(my_visible, k_start + chunk); \
    float m_sg = -INFINITY; \
    float l_sg = 0.0f; \
    for (int k = k_start + ly; k < k_end; k += 8) { \
        const ulong k_row_off = k_bh + (ulong)k * k_sl; \
        const ulong v_row_off = v_bh + (ulong)k * v_sl; \
        float dot = 0.0f; \
        _Pragma("unroll") \
        for (int r = 0; r < D_CHUNKS; ++r) { \
            const int d = r * 16 + lx; \
            if (d < Dq) dot = fma(q_reg[r], LOAD_FN(key, k_row_off + (ulong)d * k_sd), dot); \
        } \
        const float s = sub_group_reduce_add(dot) * scale; \
        const float m_new = max_nan(m_sg, s); \
        const float rescale = (m_sg == m_new) ? 1.0f : ((m_sg == -INFINITY) ? 0.0f : exp(m_sg - m_new)); \
        const float p = (m_new == -INFINITY) ? 0.0f : exp(s - m_new); \
        l_sg = l_sg * rescale + p; \
        m_sg = m_new; \
        _Pragma("unroll") \
        for (int r = 0; r < D_CHUNKS; ++r) { \
            const int d = r * 16 + lx; \
            const float vv = (d < Dv) ? LOAD_FN(value, v_row_off + (ulong)d * v_sd) : 0.0f; \
            o_reg[r] = fma(p, vv, o_reg[r] * rescale); \
        } \
    } \
    __local float sm_m[8]; \
    __local float sm_l[8]; \
    __local float sm_o[8][D_MAX]; \
    if (lx == 0) { sm_m[ly] = m_sg; sm_l[ly] = l_sg; } \
    _Pragma("unroll") \
    for (int r = 0; r < D_CHUNKS; ++r) sm_o[ly][r * 16 + lx] = o_reg[r]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (ly == 0) { \
        float m_all = sm_m[0]; \
        for (int s = 1; s < 8; ++s) m_all = max_nan(m_all, sm_m[s]); \
        float l_all = 0.0f; \
        float o_all[D_CHUNKS]; \
        _Pragma("unroll") \
        for (int r = 0; r < D_CHUNKS; ++r) o_all[r] = 0.0f; \
        for (int s = 0; s < 8; ++s) { \
            const float w = (sm_m[s] == -INFINITY || m_all == -INFINITY) ? 0.0f : exp(sm_m[s] - m_all); \
            l_all += sm_l[s] * w; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) o_all[r] = fma(w, sm_o[s][r * 16 + lx], o_all[r]); \
        } \
        const float inv_l = (l_all == 0.0f) ? 0.0f : (1.0f / l_all); \
        if (num_splits == 1) { \
            const ulong o_row_off = o_off + (ulong)batch * o_sb + (ulong)head * o_sh + (ulong)q_row * o_sl; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) { \
                const int d = r * 16 + lx; \
                if (d < Dv) STORE_FN(output, o_row_off + (ulong)d * o_sd, o_all[r] * inv_l); \
            } \
        } else { \
            const ulong slot = ((ulong)bh * Lq + q_row) * num_splits + split_id; \
            if (lx == 0) partial_lse[slot] = (l_all == 0.0f) ? -INFINITY : (m_all + log(l_all)); \
            const ulong po_base = slot * (ulong)Dv; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) { \
                const int d = r * 16 + lx; \
                if (d < Dv) partial_o[po_base + d] = o_all[r] * inv_l; \
            } \
        } \
    } \
}

DEFINE_SDPA_DECODE(sdpa_decode_d32, 32, 2, SDPA_LOAD_F32, SDPA_STORE_F32)
DEFINE_SDPA_DECODE(sdpa_decode_d64, 64, 4, SDPA_LOAD_F32, SDPA_STORE_F32)
DEFINE_SDPA_DECODE(sdpa_decode_d128, 128, 8, SDPA_LOAD_F32, SDPA_STORE_F32)
#ifdef IRIS_HAS_FP16
DEFINE_SDPA_DECODE(sdpa_decode_d32_fp16, 32, 2, SDPA_LOAD_F16, SDPA_STORE_F16)
DEFINE_SDPA_DECODE(sdpa_decode_d64_fp16, 64, 4, SDPA_LOAD_F16, SDPA_STORE_F16)
DEFINE_SDPA_DECODE(sdpa_decode_d128_fp16, 128, 8, SDPA_LOAD_F16, SDPA_STORE_F16)
#endif

__kernel void sdpa_decode_reduce(
    __global float* output, __global const float* partial_o, __global const float* partial_lse,
    ulong o_off, int B, int Hq, int Lq, int Dv,
    ulong o_sb, ulong o_sh, ulong o_sl, ulong o_sd,
    int num_splits, int fp16
) {
    const int d = get_global_id(0);
    const int q_row = get_global_id(1);
    const int bh = get_global_id(2);
    if (d >= Dv || q_row >= Lq || bh >= B * Hq) return;
    const int batch = bh / Hq;
    const int head = bh - batch * Hq;
    const ulong base_slot = ((ulong)bh * Lq + q_row) * num_splits;
    float m_all = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        m_all = max_nan(m_all, partial_lse[base_slot + s]);
    }
    float l_all = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        const float lse = partial_lse[base_slot + s];
        const float w = (lse == -INFINITY || m_all == -INFINITY) ? 0.0f : exp(lse - m_all);
        l_all += w;
        acc = fma(w, partial_o[(base_slot + s) * (ulong)Dv + d], acc);
    }
    const float out_val = (l_all == 0.0f) ? 0.0f : (acc / l_all);
    const ulong dst = o_off + (ulong)batch * o_sb + (ulong)head * o_sh + (ulong)q_row * o_sl + (ulong)d * o_sd;
    write_float(output, dst, out_val, fp16 != 0);
}
)CLC",
R"CLC(
// FlashAttention Backward: Preprocess D = sum_d(dO * O), dQ kernel, and dK/dV kernel (causal/non-causal, GQA, FP32/FP16)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void sdpa_bwd_preprocess(
    __global float* d_out, __global const float* dout, __global const float* out,
    ulong do_off, ulong o_off,
    int B, int Hq, int Lq, int Dv,
    ulong do_sb, ulong do_sh, ulong do_sl, ulong do_sd,
    ulong o_sb, ulong o_sh, ulong o_sl, ulong o_sd,
    int fp16
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int q_row = get_group_id(0) * 8 + ly;
    const int bh = get_group_id(1);
    if (q_row >= Lq || bh >= B * Hq) return;
    const int batch = bh / Hq;
    const int head = bh - batch * Hq;
    const ulong do_row = do_off + (ulong)batch * do_sb + (ulong)head * do_sh + (ulong)q_row * do_sl;
    const ulong o_row = o_off + (ulong)batch * o_sb + (ulong)head * o_sh + (ulong)q_row * o_sl;
    float acc = 0.0f;
    for (int d = lx; d < Dv; d += 16) {
        const float g = read_float(dout, do_row + (ulong)d * do_sd, fp16 != 0);
        const float y = read_float(out, o_row + (ulong)d * o_sd, fp16 != 0);
        acc = fma(g, y, acc);
    }
    const float total = sub_group_reduce_add(acc);
    if (lx == 0) {
        d_out[(ulong)bh * Lq + q_row] = total;
    }
}

#define DEFINE_SDPA_BWD_DQ(NAME, D_MAX, D_CHUNKS, LOAD_FN, STORE_FN) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME( \
    __global float* dq_out, __global const float* query, __global const float* key, \
    __global const float* value, __global const float* dout, \
    __global const float* lse_buf, __global const float* d_buf, \
    ulong dq_off, ulong q_off, ulong k_off, ulong v_off, ulong do_off, \
    int B, int Hq, int Hkv, int Lq, int Lk, int Dq, int Dv, \
    ulong q_sb, ulong q_sh, ulong q_sl, ulong q_sd, \
    ulong k_sb, ulong k_sh, ulong k_sl, ulong k_sd, \
    ulong v_sb, ulong v_sh, ulong v_sl, ulong v_sd, \
    ulong do_sb, ulong do_sh, ulong do_sl, ulong do_sd, \
    ulong dq_sb, ulong dq_sh, ulong dq_sl, ulong dq_sd, \
    float scale, int causal \
) { \
    const int lx = get_local_id(0); \
    const int ly = get_local_id(1); \
    const int tid = ly * 16 + lx; \
    const int q_base = get_group_id(0) * 8; \
    const int bh = get_group_id(1); \
    const int batch = bh / Hq; \
    const int head = bh - batch * Hq; \
    const int kv_head = head / (Hq / Hkv); \
    const int q_row = q_base + ly; \
    const bool q_valid = q_row < Lq; \
    const ulong q_bh = q_off + (ulong)batch * q_sb + (ulong)head * q_sh; \
    const ulong do_bh = do_off + (ulong)batch * do_sb + (ulong)head * do_sh; \
    const ulong dq_bh = dq_off + (ulong)batch * dq_sb + (ulong)head * dq_sh; \
    const ulong k_bh = k_off + (ulong)batch * k_sb + (ulong)kv_head * k_sh; \
    const ulong v_bh = v_off + (ulong)batch * v_sb + (ulong)kv_head * v_sh; \
    float q_reg[D_CHUNKS]; \
    float do_reg[D_CHUNKS]; \
    float dq_reg[D_CHUNKS]; \
    const ulong q_row_off = q_bh + (ulong)q_row * q_sl; \
    const ulong do_row_off = do_bh + (ulong)q_row * do_sl; \
    _Pragma("unroll") \
    for (int r = 0; r < D_CHUNKS; ++r) { \
        const int d = r * 16 + lx; \
        q_reg[r] = (q_valid && d < Dq) ? LOAD_FN(query, q_row_off + (ulong)d * q_sd) : 0.0f; \
        do_reg[r] = (q_valid && d < Dv) ? LOAD_FN(dout, do_row_off + (ulong)d * do_sd) : 0.0f; \
        dq_reg[r] = 0.0f; \
    } \
    const float lse_i = q_valid ? lse_buf[(ulong)bh * Lq + q_row] : -INFINITY; \
    const float d_i = q_valid ? d_buf[(ulong)bh * Lq + q_row] : 0.0f; \
    __local float k_tile[16][D_MAX + 1]; \
    __local float v_tile[16][D_MAX + 1]; \
    const int wg_k_end = causal ? min(Lk, q_base + 8) : Lk; \
    const int my_visible = causal ? min(Lk, q_row + 1) : Lk; \
    for (int k0 = 0; k0 < wg_k_end; k0 += 16) { \
        for (int idx = tid; idx < 16 * D_MAX; idx += 128) { \
            const int j = idx / D_MAX; \
            const int d = idx - j * D_MAX; \
            const int k_abs = k0 + j; \
            float kv = 0.0f, vv = 0.0f; \
            if (k_abs < Lk) { \
                if (d < Dq) kv = LOAD_FN(key, k_bh + (ulong)k_abs * k_sl + (ulong)d * k_sd); \
                if (d < Dv) vv = LOAD_FN(value, v_bh + (ulong)k_abs * v_sl + (ulong)d * v_sd); \
            } \
            k_tile[j][d] = kv; \
            v_tile[j][d] = vv; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (q_valid) { \
            float s = 0.0f; \
            float dp = 0.0f; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) { \
                const int d_base = r * 16; \
                if (d_base < Dq) { \
                    _Pragma("unroll") \
                    for (int c = 0; c < 16; ++c) { \
                        s = fma(sub_group_broadcast(q_reg[r], c), k_tile[lx][d_base + c], s); \
                    } \
                } \
                if (d_base < Dv) { \
                    _Pragma("unroll") \
                    for (int c = 0; c < 16; ++c) { \
                        dp = fma(sub_group_broadcast(do_reg[r], c), v_tile[lx][d_base + c], dp); \
                    } \
                } \
            } \
            const bool vis = (k0 + lx) < my_visible; \
            const float p = (vis && lse_i > -INFINITY) ? exp(s * scale - lse_i) : 0.0f; \
            const float ds = vis ? (p * (dp - d_i) * scale) : 0.0f; \
            _Pragma("unroll") \
            for (int j = 0; j < 16; ++j) { \
                const float dsv = sub_group_broadcast(ds, j); \
                _Pragma("unroll") \
                for (int r = 0; r < D_CHUNKS; ++r) { \
                    dq_reg[r] = fma(dsv, k_tile[j][r * 16 + lx], dq_reg[r]); \
                } \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (q_valid) { \
        const ulong dq_row_off = dq_bh + (ulong)q_row * dq_sl; \
        _Pragma("unroll") \
        for (int r = 0; r < D_CHUNKS; ++r) { \
            const int d = r * 16 + lx; \
            if (d < Dq) STORE_FN(dq_out, dq_row_off + (ulong)d * dq_sd, dq_reg[r]); \
        } \
    } \
}

DEFINE_SDPA_BWD_DQ(sdpa_bwd_dq_d32, 32, 2, SDPA_LOAD_F32, SDPA_STORE_F32)
DEFINE_SDPA_BWD_DQ(sdpa_bwd_dq_d64, 64, 4, SDPA_LOAD_F32, SDPA_STORE_F32)
DEFINE_SDPA_BWD_DQ(sdpa_bwd_dq_d128, 128, 8, SDPA_LOAD_F32, SDPA_STORE_F32)
#ifdef IRIS_HAS_FP16
DEFINE_SDPA_BWD_DQ(sdpa_bwd_dq_d32_fp16, 32, 2, SDPA_LOAD_F16, SDPA_STORE_F16)
DEFINE_SDPA_BWD_DQ(sdpa_bwd_dq_d64_fp16, 64, 4, SDPA_LOAD_F16, SDPA_STORE_F16)
DEFINE_SDPA_BWD_DQ(sdpa_bwd_dq_d128_fp16, 128, 8, SDPA_LOAD_F16, SDPA_STORE_F16)
#endif
)CLC",
R"CLC(
#define DEFINE_SDPA_BWD_DKV(NAME, D_MAX, D_CHUNKS, LOAD_FN, STORE_FN) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME( \
    __global float* dk_out, __global float* dv_out, \
    __global const float* query, __global const float* key, \
    __global const float* value, __global const float* dout, \
    __global const float* lse_buf, __global const float* d_buf, \
    ulong dk_off, ulong dv_off, ulong q_off, ulong k_off, ulong v_off, ulong do_off, \
    int B, int Hq, int Hkv, int Lq, int Lk, int Dq, int Dv, \
    ulong q_sb, ulong q_sh, ulong q_sl, ulong q_sd, \
    ulong k_sb, ulong k_sh, ulong k_sl, ulong k_sd, \
    ulong v_sb, ulong v_sh, ulong v_sl, ulong v_sd, \
    ulong do_sb, ulong do_sh, ulong do_sl, ulong do_sd, \
    ulong dk_sb, ulong dk_sh, ulong dk_sl, ulong dk_sd, \
    ulong dv_sb, ulong dv_sh, ulong dv_sl, ulong dv_sd, \
    float scale, int causal, int need_dk, int need_dv \
) { \
    const int lx = get_local_id(0); \
    const int ly = get_local_id(1); \
    const int tid = ly * 16 + lx; \
    const int k_base = get_group_id(0) * 8; \
    const int bh_kv = get_group_id(1); \
    const int batch = bh_kv / Hkv; \
    const int kv_head = bh_kv - batch * Hkv; \
    const int gqa_ratio = Hq / Hkv; \
    const int k_row = k_base + ly; \
    const bool k_valid = k_row < Lk; \
    const ulong k_bh = k_off + (ulong)batch * k_sb + (ulong)kv_head * k_sh; \
    const ulong v_bh = v_off + (ulong)batch * v_sb + (ulong)kv_head * v_sh; \
    const ulong dk_bh = dk_off + (ulong)batch * dk_sb + (ulong)kv_head * dk_sh; \
    const ulong dv_bh = dv_off + (ulong)batch * dv_sb + (ulong)kv_head * dv_sh; \
    float k_reg[D_CHUNKS]; \
    float v_reg[D_CHUNKS]; \
    float dk_reg[D_CHUNKS]; \
    float dv_reg[D_CHUNKS]; \
    const ulong k_row_off = k_bh + (ulong)k_row * k_sl; \
    const ulong v_row_off = v_bh + (ulong)k_row * v_sl; \
    _Pragma("unroll") \
    for (int r = 0; r < D_CHUNKS; ++r) { \
        const int d = r * 16 + lx; \
        k_reg[r] = (k_valid && d < Dq) ? LOAD_FN(key, k_row_off + (ulong)d * k_sd) : 0.0f; \
        v_reg[r] = (k_valid && d < Dv) ? LOAD_FN(value, v_row_off + (ulong)d * v_sd) : 0.0f; \
        dk_reg[r] = 0.0f; \
        dv_reg[r] = 0.0f; \
    } \
    __local float q_tile[16][D_MAX + 1]; \
    __local float do_tile[16][D_MAX + 1]; \
    __local float lse_tile[16]; \
    __local float d_tile[16]; \
    const int q_start = causal ? (k_base & ~15) : 0; \
    for (int gq = 0; gq < gqa_ratio; ++gq) { \
        const int head = kv_head * gqa_ratio + gq; \
        const int bh_q = batch * Hq + head; \
        const ulong q_bh = q_off + (ulong)batch * q_sb + (ulong)head * q_sh; \
        const ulong do_bh = do_off + (ulong)batch * do_sb + (ulong)head * do_sh; \
        for (int q0 = q_start; q0 < Lq; q0 += 16) { \
            for (int idx = tid; idx < 16 * D_MAX; idx += 128) { \
                const int qi = idx / D_MAX; \
                const int d = idx - qi * D_MAX; \
                const int q_abs = q0 + qi; \
                float qv = 0.0f, dov = 0.0f; \
                if (q_abs < Lq) { \
                    if (d < Dq) qv = LOAD_FN(query, q_bh + (ulong)q_abs * q_sl + (ulong)d * q_sd); \
                    if (d < Dv) dov = LOAD_FN(dout, do_bh + (ulong)q_abs * do_sl + (ulong)d * do_sd); \
                } \
                q_tile[qi][d] = qv; \
                do_tile[qi][d] = dov; \
            } \
            if (tid < 16) { \
                const int q_abs = q0 + tid; \
                lse_tile[tid] = (q_abs < Lq) ? lse_buf[(ulong)bh_q * Lq + q_abs] : -INFINITY; \
                d_tile[tid] = (q_abs < Lq) ? d_buf[(ulong)bh_q * Lq + q_abs] : 0.0f; \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
            if (k_valid) { \
                float s = 0.0f; \
                float dp = 0.0f; \
                _Pragma("unroll") \
                for (int r = 0; r < D_CHUNKS; ++r) { \
                    const int d_base = r * 16; \
                    if (d_base < Dq) { \
                        _Pragma("unroll") \
                        for (int c = 0; c < 16; ++c) { \
                            s = fma(sub_group_broadcast(k_reg[r], c), q_tile[lx][d_base + c], s); \
                        } \
                    } \
                    if (d_base < Dv) { \
                        _Pragma("unroll") \
                        for (int c = 0; c < 16; ++c) { \
                            dp = fma(sub_group_broadcast(v_reg[r], c), do_tile[lx][d_base + c], dp); \
                        } \
                    } \
                } \
                const int q_abs = q0 + lx; \
                const bool vis = (q_abs < Lq) && (!causal || k_row <= q_abs); \
                const float lse_q = lse_tile[lx]; \
                const float p = (vis && lse_q > -INFINITY) ? exp(s * scale - lse_q) : 0.0f; \
                const float ds = vis ? (p * (dp - d_tile[lx]) * scale) : 0.0f; \
                _Pragma("unroll") \
                for (int qi = 0; qi < 16; ++qi) { \
                    const float pv = sub_group_broadcast(p, qi); \
                    const float dsv = sub_group_broadcast(ds, qi); \
                    _Pragma("unroll") \
                    for (int r = 0; r < D_CHUNKS; ++r) { \
                        if (need_dv) dv_reg[r] = fma(pv, do_tile[qi][r * 16 + lx], dv_reg[r]); \
                        if (need_dk) dk_reg[r] = fma(dsv, q_tile[qi][r * 16 + lx], dk_reg[r]); \
                    } \
                } \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    } \
    if (k_valid) { \
        if (need_dk) { \
            const ulong dk_row_off = dk_bh + (ulong)k_row * dk_sl; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) { \
                const int d = r * 16 + lx; \
                if (d < Dq) STORE_FN(dk_out, dk_row_off + (ulong)d * dk_sd, dk_reg[r]); \
            } \
        } \
        if (need_dv) { \
            const ulong dv_row_off = dv_bh + (ulong)k_row * dv_sl; \
            _Pragma("unroll") \
            for (int r = 0; r < D_CHUNKS; ++r) { \
                const int d = r * 16 + lx; \
                if (d < Dv) STORE_FN(dv_out, dv_row_off + (ulong)d * dv_sd, dv_reg[r]); \
            } \
        } \
    } \
}

DEFINE_SDPA_BWD_DKV(sdpa_bwd_dkv_d32, 32, 2, SDPA_LOAD_F32, SDPA_STORE_F32)
DEFINE_SDPA_BWD_DKV(sdpa_bwd_dkv_d64, 64, 4, SDPA_LOAD_F32, SDPA_STORE_F32)
DEFINE_SDPA_BWD_DKV(sdpa_bwd_dkv_d128, 128, 8, SDPA_LOAD_F32, SDPA_STORE_F32)
#ifdef IRIS_HAS_FP16
DEFINE_SDPA_BWD_DKV(sdpa_bwd_dkv_d32_fp16, 32, 2, SDPA_LOAD_F16, SDPA_STORE_F16)
DEFINE_SDPA_BWD_DKV(sdpa_bwd_dkv_d64_fp16, 64, 4, SDPA_LOAD_F16, SDPA_STORE_F16)
DEFINE_SDPA_BWD_DKV(sdpa_bwd_dkv_d128_fp16, 128, 8, SDPA_LOAD_F16, SDPA_STORE_F16)
#endif
#endif

__kernel void fused_attention(__global const ulong* cfg, __global float* output,
                               __global const float* query, __global const float* key,
                               __global const float* value, __global const uchar* unused0,
                               __global const uchar* unused1, ulong output_offset, ulong query_offset,
                               ulong key_offset, ulong value_offset, ulong unused2, ulong unused3) {
    const ulong row = get_group_id(0);
    const ulong q_count = cfg[0], k_count = cfg[1], width = cfg[2], value_width = cfg[3], heads = cfg[4];
    const ulong position = row % q_count;
    const ulong head = row / q_count % heads;
    const ulong batch = row / (q_count * heads);
    const ulong q = query_offset + batch * cfg[5] + head * cfg[6] + position * cfg[7];
    const ulong k = key_offset + batch * cfg[9] + head * cfg[10];
    const ulong v = value_offset + batch * cfg[13] + head * cfg[14];
    const ulong visible = cfg[17] ? min(k_count, position + 1) : k_count;
    const float scale = as_float((uint)cfg[18]);
    const uint lane = get_local_id(0), group = get_local_size(0);
    __local float weights[256];
    __local float workspace[64];
    float maximum = -INFINITY;
    for (ulong j = lane; j < visible; j += group) {
        float dot = 0.0f;
        for (ulong d = 0; d < width; ++d) {
            dot = fma(query[q + d * cfg[8]], key[k + j * cfg[11] + d * cfg[12]], dot);
        }
        const float score = dot * scale;
        weights[j] = score;
        maximum = max_nan(maximum, score);
    }
    workspace[lane] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = group / 2; step > 0; step /= 2) {
        if (lane < step) workspace[lane] = max_nan(workspace[lane], workspace[lane + step]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = workspace[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float denominator = 0.0f;
    for (ulong j = lane; j < visible; j += group) {
        const float weight = maximum == -INFINITY ? 0.0f : exp(weights[j] - maximum);
        weights[j] = weight;
        denominator += weight;
    }
    denominator = local_sum(workspace, lane, denominator);
    for (ulong d = lane; d < value_width; d += group) {
        float sum = 0.0f;
        for (ulong j = 0; j < visible; ++j) {
            sum = fma(weights[j], value[v + j * cfg[15] + d * cfg[16]], sum);
        }
        const ulong dst = output_offset + batch * cfg[19] + head * cfg[20] + position * cfg[21] + d * cfg[22];
        output[dst] = denominator == 0.0f ? 0.0f : sum / denominator;
    }
}
)CLC"
};

inline constexpr std::array conv_fast_kernels{
R"CLC(
__kernel void conv2d_depthwise_3x3(
    __global float* output, __global const float* input, __global const float* weight,
    __global const float* bias, ulong out_off, ulong in_off, ulong w_off, ulong b_off,
    int channels, int h, int w, int oh, int ow, int sy, int sx, int py, int px, int fp16
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const ulong plane = get_global_id(2);
    if (x >= ow || y >= oh) return;
    const int channel = plane % channels;
    const ulong in_base = in_off + plane * h * w;
    const ulong w_base = w_off + (ulong)channel * 9;
    const int in_x = x * sx - px;
    const int in_y = y * sy - py;
    float sum = 0.0f;
    #pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        const int iy = in_y + ky;
        #pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            const int ix = in_x + kx;
            const float value = (uint)iy < (uint)h && (uint)ix < (uint)w
                ? read_float(input, in_base + (ulong)iy * w + ix, fp16) : 0.0f;
            sum = fma(value, read_float(weight, w_base + ky * 3 + kx, fp16), sum);
        }
    }
    if (bias) sum += read_float(bias, b_off + channel, fp16);
    write_float(output, out_off + (plane * oh + y) * ow + x, sum, fp16);
}

// Spatial-halo tiled direct 3x3 stride=1 dilation=1 Conv2d (OC_BLK=16, TH=8, TW=16, WG=(16,8))
__kernel void conv2d_direct_3x3_s1_oc16(
    __global float* output, __global const float* input, __global const float* weights, __global const float* bias,
    ulong out_offset, ulong in_offset, ulong w_offset, ulong bias_offset,
    int N, int C_in, int H, int W, int C_out, int OH, int OW, int py, int px, int groups, int fp16
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int tid = ly * 16 + lx;
    const bool h = fp16 != 0;

    const int tiles_w = (OW + 15) / 16;
    const int tile_idx = get_group_id(0);
    const int tile_x = (tile_idx % tiles_w) * 16;
    const int tile_y = (tile_idx / tiles_w) * 8;

    const int oc_per_g = C_out / groups;
    const int ic_per_g = C_in / groups;
    const int oc_blocks_per_g = (oc_per_g + 15) / 16;
    const int ng_oc = get_group_id(1);
    const int oc_blk_in_g = ng_oc % oc_blocks_per_g;
    const int g = (ng_oc / oc_blocks_per_g) % groups;
    const int n = get_group_id(2);

    const int oc_g_start = oc_blk_in_g * 16;
    const int oc_base = g * oc_per_g + oc_g_start;
    const int ic_base = g * ic_per_g;

    __local float in_tile[4][10][18];
    __local float w_tile[16][4][9];

    const int row_sub = ly & 3;
    const int oc_half = (ly >> 2) * 8;

    float acc0[8];
    float acc1[8];
    #pragma unroll
    for (int o = 0; o < 8; ++o) {
        const int oc_local = oc_g_start + oc_half + o;
        const float bval = (bias && oc_local < oc_per_g) ? read_float(bias, bias_offset + oc_base + oc_half + o, h) : 0.0f;
        acc0[o] = bval;
        acc1[o] = bval;
    }

    const int hw = H * W;
    const int ohw = OH * OW;
    const ulong in_n_base = in_offset + (ulong)n * C_in * hw;

    for (int ic = 0; ic < ic_per_g; ic += 4) {
        for (int idx = tid; idx < 4 * 10 * 18; idx += 128) {
            const int ix_l = idx % 18;
            const int t = idx / 18;
            const int iy_l = t % 10;
            const int c_l = t / 10;
            const int c_abs = ic + c_l;
            const int iy = tile_y + iy_l - py;
            const int ix = tile_x + ix_l - px;
            float v = 0.0f;
            if (c_abs < ic_per_g && iy >= 0 && iy < H && ix >= 0 && ix < W) {
                v = read_float(input, in_n_base + (ulong)(ic_base + c_abs) * hw + iy * W + ix, h);
            }
            in_tile[c_l][iy_l][ix_l] = v;
        }
        for (int idx = tid; idx < 16 * 4 * 9; idx += 128) {
            const int k = idx % 9;
            const int t = idx / 9;
            const int c_l = t % 4;
            const int o_l = t / 4;
            const int oc_local = oc_g_start + o_l;
            const int c_abs = ic + c_l;
            float wv = 0.0f;
            if (oc_local < oc_per_g && c_abs < ic_per_g) {
                wv = read_float(weights, w_offset + ((ulong)(oc_base + o_l) * ic_per_g + c_abs) * 9 + k, h);
            }
            w_tile[o_l][c_l][k] = wv;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int c_l = 0; c_l < 4; ++c_l) {
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const float x0 = in_tile[c_l][row_sub + ky][lx + kx];
                    const float x1 = in_tile[c_l][row_sub + 4 + ky][lx + kx];
                    const int k = ky * 3 + kx;
                    #pragma unroll
                    for (int o = 0; o < 8; ++o) {
                        const float wv = w_tile[oc_half + o][c_l][k];
                        acc0[o] = fma(x0, wv, acc0[o]);
                        acc1[o] = fma(x1, wv, acc1[o]);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int ox = tile_x + lx;
    if (ox < OW) {
        const int oy0 = tile_y + row_sub;
        const int oy1 = oy0 + 4;
        const ulong out_n_base = out_offset + (ulong)n * C_out * ohw;
        #pragma unroll
        for (int o = 0; o < 8; ++o) {
            const int oc_local = oc_g_start + oc_half + o;
            if (oc_local < oc_per_g) {
                const ulong c_off = out_n_base + (ulong)(oc_base + oc_half + o) * ohw;
                if (oy0 < OH) write_float(output, c_off + oy0 * OW + ox, acc0[o], h);
                if (oy1 < OH) write_float(output, c_off + oy1 * OW + ox, acc1[o], h);
            }
        }
    }
}

// Spatial-halo tiled direct 3x3 stride=1 dilation=1 Conv2d (OC_BLK=32, TH=8, TW=16, WG=(16,8))
__kernel void conv2d_direct_3x3_s1_oc32(
    __global float* output, __global const float* input, __global const float* weights, __global const float* bias,
    ulong out_offset, ulong in_offset, ulong w_offset, ulong bias_offset,
    int N, int C_in, int H, int W, int C_out, int OH, int OW, int py, int px, int groups, int fp16
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int tid = ly * 16 + lx;
    const bool h = fp16 != 0;

    const int tiles_w = (OW + 15) / 16;
    const int tile_idx = get_group_id(0);
    const int tile_x = (tile_idx % tiles_w) * 16;
    const int tile_y = (tile_idx / tiles_w) * 8;

    const int oc_per_g = C_out / groups;
    const int ic_per_g = C_in / groups;
    const int oc_blocks_per_g = (oc_per_g + 31) / 32;
    const int ng_oc = get_group_id(1);
    const int oc_blk_in_g = ng_oc % oc_blocks_per_g;
    const int g = (ng_oc / oc_blocks_per_g) % groups;
    const int n = get_group_id(2);

    const int oc_g_start = oc_blk_in_g * 32;
    const int oc_base = g * oc_per_g + oc_g_start;
    const int ic_base = g * ic_per_g;

    __local float in_tile[4][10][18];
    __local float w_tile[32][4][9];

    const int y_base = (ly & 1) * 4;
    const int oc_sub = (ly >> 1) * 8;

    float acc[4][8];
    #pragma unroll
    for (int o = 0; o < 8; ++o) {
        const int oc_local = oc_g_start + oc_sub + o;
        const float bval = (bias && oc_local < oc_per_g) ? read_float(bias, bias_offset + oc_base + oc_sub + o, h) : 0.0f;
        acc[0][o] = bval; acc[1][o] = bval; acc[2][o] = bval; acc[3][o] = bval;
    }

    const int hw = H * W;
    const int ohw = OH * OW;
    const ulong in_n_base = in_offset + (ulong)n * C_in * hw;

    for (int ic = 0; ic < ic_per_g; ic += 4) {
        for (int idx = tid; idx < 4 * 10 * 18; idx += 128) {
            const int ix_l = idx % 18;
            const int t = idx / 18;
            const int iy_l = t % 10;
            const int c_l = t / 10;
            const int c_abs = ic + c_l;
            const int iy = tile_y + iy_l - py;
            const int ix = tile_x + ix_l - px;
            float v = 0.0f;
            if (c_abs < ic_per_g && iy >= 0 && iy < H && ix >= 0 && ix < W) {
                v = read_float(input, in_n_base + (ulong)(ic_base + c_abs) * hw + iy * W + ix, h);
            }
            in_tile[c_l][iy_l][ix_l] = v;
        }
        for (int idx = tid; idx < 32 * 4 * 9; idx += 128) {
            const int k = idx % 9;
            const int t = idx / 9;
            const int c_l = t % 4;
            const int o_l = t / 4;
            const int oc_local = oc_g_start + o_l;
            const int c_abs = ic + c_l;
            float wv = 0.0f;
            if (oc_local < oc_per_g && c_abs < ic_per_g) {
                wv = read_float(weights, w_offset + ((ulong)(oc_base + o_l) * ic_per_g + c_abs) * 9 + k, h);
            }
            w_tile[o_l][c_l][k] = wv;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int c_l = 0; c_l < 4; ++c_l) {
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const float x0 = in_tile[c_l][y_base + 0 + ky][lx + kx];
                    const float x1 = in_tile[c_l][y_base + 1 + ky][lx + kx];
                    const float x2 = in_tile[c_l][y_base + 2 + ky][lx + kx];
                    const float x3 = in_tile[c_l][y_base + 3 + ky][lx + kx];
                    const int k = ky * 3 + kx;
                    #pragma unroll
                    for (int o = 0; o < 8; ++o) {
                        const float wv = w_tile[oc_sub + o][c_l][k];
                        acc[0][o] = fma(x0, wv, acc[0][o]);
                        acc[1][o] = fma(x1, wv, acc[1][o]);
                        acc[2][o] = fma(x2, wv, acc[2][o]);
                        acc[3][o] = fma(x3, wv, acc[3][o]);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int ox = tile_x + lx;
    if (ox < OW) {
        const ulong out_n_base = out_offset + (ulong)n * C_out * ohw;
        #pragma unroll
        for (int o = 0; o < 8; ++o) {
            const int oc_local = oc_g_start + oc_sub + o;
            if (oc_local < oc_per_g) {
                const ulong c_off = out_n_base + (ulong)(oc_base + oc_sub + o) * ohw;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int oy = tile_y + y_base + r;
                    if (oy < OH) write_float(output, c_off + oy * OW + ox, acc[r][o], h);
                }
            }
        }
    }
}

// General Implicit-GEMM Conv2d (BM=32, BN=64, BK=16, WG=(16,8), 0 B temp memory)
)CLC",
R"CLC(
__kernel void conv2d_implicit_gemm(
    __global float* output, __global const float* input, __global const float* weights, __global const float* bias,
    ulong out_offset, ulong in_offset, ulong w_offset, ulong bias_offset,
    int N, int C_in, int H, int W, int C_out, int OH, int OW,
    int KH, int KW, int sy, int sx, int py, int px, int dy, int dx, int groups, int fp16
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int tid = ly * 16 + lx;
    const bool h = fp16 != 0;

    const int oc_per_g = C_out / groups;
    const int ic_per_g = C_in / groups;
    const int P = OH * OW;
    const int K = ic_per_g * KH * KW;

    const int col_base = get_group_id(0) * 64;
    const int oc_blocks_per_g = (oc_per_g + 31) / 32;
    const int ng_oc = get_group_id(1);
    const int oc_blk_in_g = ng_oc % oc_blocks_per_g;
    const int g = ng_oc / oc_blocks_per_g;
    const int n = get_group_id(2);

    const int row_base = oc_blk_in_g * 32;
    const int oc_abs_base = g * oc_per_g + row_base;
    const int ic_abs_base = g * ic_per_g;

    __local float a_tile[16][33];
    __local float b_tile[16][65];

    const int p_col_l = tid & 63;
    const int p_abs = col_base + p_col_l;
    const int p_valid = p_abs < P;
    const int p_oy = p_valid ? (p_abs / OW) : 0;
    const int p_ox = p_valid ? (p_abs - p_oy * OW) : 0;
    const int p_iy_base = p_oy * sy - py;
    const int p_ix_base = p_ox * sx - px;

    const int a_m_l = tid >> 2;
    const int a_k_sub = (tid & 3) << 2;
    const int a_oc_valid = (row_base + a_m_l) < oc_per_g;
    const ulong w_row_ptr = w_offset + (ulong)(oc_abs_base + a_m_l) * K;

    const int hw = H * W;
    const ulong in_n_base = in_offset + (ulong)n * C_in * hw;
    const int khw = KH * KW;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int oc_l = row_base + ly + i * 8;
        const float bval = (bias && oc_l < oc_per_g) ? read_float(bias, bias_offset + oc_abs_base + ly + i * 8, h) : 0.0f;
        acc[i][0] = bval; acc[i][1] = bval; acc[i][2] = bval; acc[i][3] = bval;
    }

    for (int k0 = 0; k0 < K; k0 += 16) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k_l = a_k_sub + j;
            const int k_abs = k0 + k_l;
            a_tile[k_l][a_m_l] = (a_oc_valid && k_abs < K) ? read_float(weights, w_row_ptr + k_abs, h) : 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int k_l = (tid >> 6) + j * 2;
            const int k_abs = k0 + k_l;
            float val = 0.0f;
            if (p_valid && k_abs < K) {
                const int c_l = k_abs / khw;
                const int rem = k_abs - c_l * khw;
                const int kh = rem / KW;
                const int kw = rem - kh * KW;
                const int iy = p_iy_base + kh * dy;
                const int ix = p_ix_base + kw * dx;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    val = read_float(input, in_n_base + (ulong)(ic_abs_base + c_l) * hw + iy * W + ix, h);
                }
            }
            b_tile[k_l][p_col_l] = val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k_l = 0; k_l < 16; ++k_l) {
            const float b0 = b_tile[k_l][lx * 4 + 0];
            const float b1 = b_tile[k_l][lx * 4 + 1];
            const float b2 = b_tile[k_l][lx * 4 + 2];
            const float b3 = b_tile[k_l][lx * 4 + 3];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float a = a_tile[k_l][ly + i * 8];
                acc[i][0] = fma(a, b0, acc[i][0]);
                acc[i][1] = fma(a, b1, acc[i][1]);
                acc[i][2] = fma(a, b2, acc[i][2]);
                acc[i][3] = fma(a, b3, acc[i][3]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong out_n_base = out_offset + (ulong)n * C_out * P;
    const int p0 = col_base + lx * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int oc_l = row_base + ly + i * 8;
        if (oc_l < oc_per_g) {
            const ulong dst = out_n_base + (ulong)(oc_abs_base + ly + i * 8) * P + p0;
            if (p0 + 3 < P) {
                const float4 out4 = (float4)(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                if (h) vstore_half4_rte(out4, 0, ((__global half*)output) + dst);
                else vstore4(out4, 0, output + dst);
            } else {
                for (int j = 0; j < 4 && p0 + j < P; ++j) {
                    write_float(output, dst + j, acc[i][j], h);
                }
            }
        }
    }
}

// Spatial-halo tiled direct 3x3 stride=1 dilation=1 Conv2d grad_input (0 B temp memory)
__kernel void conv2d_grad_input_3x3_s1(
    __global float* grad_in, __global const float* grad_out, __global const float* weights,
    ulong out_offset, ulong grad_offset, ulong w_offset,
    int N, int C_in, int H, int W, int C_out, int OH, int OW, int py, int px, int groups, int fp16
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int tid = ly * 16 + lx;
    const bool h = fp16 != 0;

    const int tiles_w = (W + 15) / 16;
    const int tile_idx = get_group_id(0);
    const int tile_x = (tile_idx % tiles_w) * 16;
    const int tile_y = (tile_idx / tiles_w) * 8;

    const int ic_per_g = C_in / groups;
    const int oc_per_g = C_out / groups;
    const int ic_blocks_per_g = (ic_per_g + 15) / 16;
    const int ng_ic = get_group_id(1);
    const int ic_blk_in_g = ng_ic % ic_blocks_per_g;
    const int g = (ng_ic / ic_blocks_per_g) % groups;
    const int n = get_group_id(2);

    const int ic_g_start = ic_blk_in_g * 16;
    const int ic_base = g * ic_per_g + ic_g_start;
    const int oc_base = g * oc_per_g;

    __local float g_tile[4][10][18];
    __local float w_tile[16][4][9];

    const int row_sub = ly & 3;
    const int ic_half = (ly >> 2) * 8;
    float acc0[8] = {0}, acc1[8] = {0};

    const int hw = H * W;
    const int ohw = OH * OW;
    const ulong grad_n_base = grad_offset + (ulong)n * C_out * ohw;

    for (int oc = 0; oc < oc_per_g; oc += 4) {
        for (int idx = tid; idx < 4 * 10 * 18; idx += 128) {
            const int ix_l = idx % 18;
            const int t = idx / 18;
            const int iy_l = t % 10;
            const int c_l = t / 10;
            const int oc_abs = oc + c_l;
            const int oy = tile_y + iy_l + py - 2;
            const int ox = tile_x + ix_l + px - 2;
            float v = 0.0f;
            if (oc_abs < oc_per_g && oy >= 0 && oy < OH && ox >= 0 && ox < OW) {
                v = read_float(grad_out, grad_n_base + (ulong)(oc_base + oc_abs) * ohw + oy * OW + ox, h);
            }
            g_tile[c_l][iy_l][ix_l] = v;
        }
        for (int idx = tid; idx < 16 * 4 * 9; idx += 128) {
            const int k = idx % 9;
            const int t = idx / 9;
            const int oc_l = t % 4;
            const int ic_l = t / 4;
            const int ic_local = ic_g_start + ic_l;
            const int oc_abs = oc + oc_l;
            float wv = 0.0f;
            if (ic_local < ic_per_g && oc_abs < oc_per_g) {
                wv = read_float(weights, w_offset + ((ulong)(oc_base + oc_abs) * ic_per_g + ic_local) * 9 + (8 - k), h);
            }
            w_tile[ic_l][oc_l][k] = wv;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int oc_l = 0; oc_l < 4; ++oc_l) {
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const float x0 = g_tile[oc_l][row_sub + ky][lx + kx];
                    const float x1 = g_tile[oc_l][row_sub + 4 + ky][lx + kx];
                    const int k = ky * 3 + kx;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float wv = w_tile[ic_half + i][oc_l][k];
                        acc0[i] = fma(x0, wv, acc0[i]);
                        acc1[i] = fma(x1, wv, acc1[i]);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int ix = tile_x + lx;
    if (ix < W) {
        const int iy0 = tile_y + row_sub;
        const int iy1 = iy0 + 4;
        const ulong out_n_base = out_offset + (ulong)n * C_in * hw;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int ic_local = ic_g_start + ic_half + i;
            if (ic_local < ic_per_g) {
                const ulong c_off = out_n_base + (ulong)(ic_base + ic_half + i) * hw;
                if (iy0 < H) write_float(grad_in, c_off + iy0 * W + ix, acc0[i], h);
                if (iy1 < H) write_float(grad_in, c_off + iy1 * W + ix, acc1[i], h);
            }
        }
    }
}

// General 32-bit scalar-arg Conv2d grad_input (arbitrary KH, KW, stride, padding, dilation, groups)
)CLC",
R"CLC(
__kernel void conv2d_grad_input_general(
    __global float* output, __global const float* grad, __global const float* weights,
    ulong out_offset, ulong grad_offset, ulong w_offset,
    int N, int C_in, int H, int W, int C_out, int OH, int OW,
    int KH, int KW, int sy, int sx, int py, int px, int dy, int dx, int groups, int fp16
) {
    const ulong pos = get_global_id(0);
    const ulong total = (ulong)N * C_in * H * W;
    if (pos >= total) return;
    const bool h = fp16 != 0;
    const int hw = H * W;
    const int ohw = OH * OW;
    const int x = (int)(pos % (ulong)W);
    const ulong t0 = pos / (ulong)W;
    const int y = (int)(t0 % (ulong)H);
    const ulong t1 = t0 / (ulong)H;
    const int c = (int)(t1 % (ulong)C_in);
    const int n = (int)(t1 / (ulong)C_in);

    const int in_group = C_in / groups;
    const int out_group = C_out / groups;
    const int g = c / in_group;
    const int ic_in_g = c - g * in_group;
    const int start_filter = g * out_group;
    const int khw = KH * KW;
    const ulong grad_n_base = grad_offset + (ulong)n * C_out * ohw;

    float sum = 0.0f;
    for (int ky = 0; ky < KH; ++ky) {
        const int y_num = y + py - ky * dy;
        if (y_num < 0) continue;
        const int oy = y_num / sy;
        if (oy * sy != y_num || oy >= OH) continue;
        for (int kx = 0; kx < KW; ++kx) {
            const int x_num = x + px - kx * dx;
            if (x_num < 0) continue;
            const int ox = x_num / sx;
            if (ox * sx != x_num || ox >= OW) continue;
            const int g_pix = oy * OW + ox;
            const int k_off = ky * KW + kx;
            for (int f = start_filter; f < start_filter + out_group; ++f) {
                const ulong wi = w_offset + ((ulong)f * in_group + ic_in_g) * khw + k_off;
                const ulong gi = grad_n_base + (ulong)f * ohw + g_pix;
                sum = fma(read_float(grad, gi, h), read_float(weights, wi, h), sum);
            }
        }
    }
    write_float(output, out_offset + pos, sum, h);
}

// Split-K tiled implicit-GEMM Conv2d grad_weight with Kahan outer summation (32x32 tile, WG=(8,8))
__kernel void conv2d_grad_weight_splitk_32x32(
    __global float* partial, __global const float* grad_out, __global const float* input,
    ulong out_offset, ulong grad_offset, ulong in_offset,
    int N, int C_in, int H, int W, int C_out, int OH, int OW,
    int KH, int KW, int sy, int sx, int py, int px, int dy, int dx, int groups, int split_k,
    int in_fp16, int out_fp16
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int tid = ly * 8 + lx;
    const bool ih = in_fp16 != 0;
    const bool oh = out_fp16 != 0;

    const int ic_per_g = C_in / groups;
    const int oc_per_g = C_out / groups;
    const int K_col = ic_per_g * KH * KW;
    const int P = OH * OW;
    const int NP = N * P;
    const int hw = H * W;
    const int khw = KH * KW;

    const int k_blk = get_group_id(0) * 32;
    const int oc_blocks_per_g = (oc_per_g + 31) / 32;
    const int g_oc = get_group_id(1);
    const int oc_blk_in_g = g_oc % oc_blocks_per_g;
    const int g = g_oc / oc_blocks_per_g;
    const int s = get_group_id(2);

    const int oc_g_start = oc_blk_in_g * 32;
    const int oc_abs_base = g * oc_per_g + oc_g_start;
    const int ic_abs_base = g * ic_per_g;

    const int chunk = (NP + split_k - 1) / split_k;
    const int p_start = s * chunk;
    const int p_end = min(NP, p_start + chunk);

    __local float g_tile[32][33];
    __local float x_tile[32][33];

    const int p_sub = tid & 31;
    const int r_half = tid >> 5;

    float acc[4][4];
    float comp[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) { acc[i][j] = 0.0f; comp[i][j] = 0.0f; }

    for (int p0 = p_start; p0 < p_end; p0 += 32) {
        const int np = p0 + p_sub;
        const bool p_valid = np < p_end;
        const int n = p_valid ? (np / P) : 0;
        const int p_rem = p_valid ? (np - n * P) : 0;
        const int oy = p_rem / OW;
        const int ox = p_rem - oy * OW;
        const int iy_base = oy * sy - py;
        const int ix_base = ox * sx - px;

        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int r = r_half * 16 + j;
            const int oc_l = oc_g_start + r;
            float gv = 0.0f;
            if (p_valid && oc_l < oc_per_g) {
                gv = read_float(grad_out, grad_offset + ((ulong)n * C_out + (oc_abs_base + r)) * P + p_rem, ih);
            }
            g_tile[r][p_sub] = gv;
        }
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int r = r_half * 16 + j;
            const int k_abs = k_blk + r;
            float xv = 0.0f;
            if (p_valid && k_abs < K_col) {
                const int c_l = k_abs / khw;
                const int rem = k_abs - c_l * khw;
                const int kh = rem / KW;
                const int kw = rem - kh * KW;
                const int iy = iy_base + kh * dy;
                const int ix = ix_base + kw * dx;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    xv = read_float(input, in_offset + ((ulong)n * C_in + (ic_abs_base + c_l)) * hw + iy * W + ix, ih);
                }
            }
            x_tile[r][p_sub] = xv;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float part[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) part[i][j] = 0.0f;

        #pragma unroll
        for (int t = 0; t < 32; ++t) {
            float gv[4], xv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                gv[i] = g_tile[ly + i * 8][t];
                xv[i] = x_tile[lx + i * 8][t];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    part[i][j] = fma(gv[i], xv[j], part[i][j]);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float y = part[i][j] - comp[i][j];
                const float t_sum = acc[i][j] + y;
                comp[i][j] = (t_sum - acc[i][j]) - y;
                acc[i][j] = t_sum;
            }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong w_total = (ulong)C_out * K_col;
    const ulong s_off = out_offset + (ulong)s * w_total;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int oc_l = oc_g_start + ly + i * 8;
        if (oc_l < oc_per_g) {
            const ulong row_off = s_off + (ulong)(oc_abs_base + ly + i * 8) * K_col;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kc = k_blk + lx + j * 8;
                if (kc < K_col) write_float(partial, row_off + kc, acc[i][j], oh);
            }
        }
    }
}

__kernel void reduce_splitk(__global float* output, __global const float* partial,
                            ulong out_offset, ulong count, int split_k, int fp16) {
    const ulong i = get_global_id(0);
    if (i >= count) return;
    float sum = 0.0f;
    float c = 0.0f;
    for (int s = 0; s < split_k; ++s) {
        const float y = partial[(ulong)s * count + i] - c;
        const float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    write_float(output, out_offset + i, sum, fp16 != 0);
}
)CLC",
R"CLC(
// Fast 32-bit scalar-arg MaxPool2d and AvgPool2d forward/backward kernels (FP32 & FP16)
__kernel void max_pool2d_forward_2x2_s2(
    __global float* output, __global long* indices, __global const float* input,
    ulong out_offset, ulong idx_offset, ulong in_offset,
    int NC, int H, int W, int OH, int OW, int fp16
) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int nc = get_global_id(2);
    if (ox >= OW || oy >= OH || nc >= NC) return;
    const bool h = fp16 != 0;
    const int y0 = oy << 1;
    const int x0 = ox << 1;
    const ulong in_plane = in_offset + (ulong)nc * (H * W);
    const int p00 = y0 * W + x0;
    float best = read_float(input, in_plane + p00, h);
    int chosen = p00;
    if (x0 + 1 < W) {
        const int p01 = p00 + 1;
        const float s = read_float(input, in_plane + p01, h);
        if (s > best || isnan(s)) { best = s; chosen = p01; }
    }
    if (y0 + 1 < H) {
        const int p10 = p00 + W;
        const float s = read_float(input, in_plane + p10, h);
        if (s > best || isnan(s)) { best = s; chosen = p10; }
        if (x0 + 1 < W) {
            const int p11 = p10 + 1;
            const float s2 = read_float(input, in_plane + p11, h);
            if (s2 > best || isnan(s2)) { best = s2; chosen = p11; }
        }
    }
    const ulong out_pos = ((ulong)nc * OH + oy) * OW + ox;
    write_float(output, out_offset + out_pos, best, h);
    indices[idx_offset + out_pos] = (long)chosen;
}

__kernel void max_pool2d_forward_general(
    __global float* output, __global long* indices, __global const float* input,
    ulong out_offset, ulong idx_offset, ulong in_offset,
    int NC, int H, int W, int OH, int OW,
    int kh, int kw, int sy, int sx, int py, int px, int dy, int dx, int fp16
) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int nc = get_global_id(2);
    if (ox >= OW || oy >= OH || nc >= NC) return;
    const bool h = fp16 != 0;
    const int y0 = oy * sy - py;
    const int x0 = ox * sx - px;
    const ulong in_plane = in_offset + (ulong)nc * (H * W);
    float best = -INFINITY;
    int chosen = -1;
    for (int ky = 0; ky < kh; ++ky) {
        const int y = y0 + ky * dy;
        if (y < 0 || y >= H) continue;
        const int row_base = y * W;
        for (int kx = 0; kx < kw; ++kx) {
            const int x = x0 + kx * dx;
            if (x < 0 || x >= W) continue;
            const int pixel = row_base + x;
            const float s = read_float(input, in_plane + pixel, h);
            if (chosen < 0 || s > best || isnan(s)) {
                best = s;
                chosen = pixel;
            }
        }
    }
    const ulong out_pos = ((ulong)nc * OH + oy) * OW + ox;
    write_float(output, out_offset + out_pos, best, h);
    indices[idx_offset + out_pos] = (long)chosen;
}

__kernel void max_pool2d_backward_2x2_s2(
    __global float* grad_in, __global const float* grad_out, __global const long* indices,
    ulong out_offset, ulong grad_offset, ulong idx_offset,
    int NC, int H, int W, int OH, int OW, int fp16
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int nc = get_global_id(2);
    if (x >= W || y >= H || nc >= NC) return;
    const bool h = fp16 != 0;
    const int oy = y >> 1;
    const int ox = x >> 1;
    const int pixel = y * W + x;
    float sum = 0.0f;
    if (oy < OH && ox < OW) {
        const ulong oi = ((ulong)nc * OH + oy) * OW + ox;
        if (indices[idx_offset + oi] == (long)pixel) {
            sum = read_float(grad_out, grad_offset + oi, h);
        }
    }
    write_float(grad_in, out_offset + ((ulong)nc * H + y) * W + x, sum, h);
}

__kernel void max_pool2d_backward_general(
    __global float* grad_in, __global const float* grad_out, __global const long* indices,
    ulong out_offset, ulong grad_offset, ulong idx_offset,
    int NC, int H, int W, int OH, int OW,
    int kh, int kw, int sy, int sx, int py, int px, int dy, int dx, int fp16
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int nc = get_global_id(2);
    if (x >= W || y >= H || nc >= NC) return;
    const bool h = fp16 != 0;
    const int pixel = y * W + x;
    const int yp = y + py;
    const int xp = x + px;
    const int first_y = max(0, (yp - (kh - 1) * dy + sy - 1) / sy);
    const int first_x = max(0, (xp - (kw - 1) * dx + sx - 1) / sx);
    const int last_y = min(OH - 1, yp / sy);
    const int last_x = min(OW - 1, xp / sx);
    const ulong o_plane = (ulong)nc * OH * OW;
    float sum = 0.0f;
    for (int oy = first_y; oy <= last_y; ++oy) {
        const ulong row_base = o_plane + (ulong)oy * OW;
        for (int ox = first_x; ox <= last_x; ++ox) {
            const ulong oi = row_base + ox;
            if (indices[idx_offset + oi] == (long)pixel) {
                sum += read_float(grad_out, grad_offset + oi, h);
            }
        }
    }
    write_float(grad_in, out_offset + ((ulong)nc * H + y) * W + x, sum, h);
}

inline int pool_divisor_i32(int y0, int x0, int H, int W, int kh, int kw, int py, int px,
                            int include_pad, int divisor) {
    if (divisor) return divisor;
    const int ph = include_pad ? py : 0;
    const int pw = include_pad ? px : 0;
    const int h = min(y0 + kh, H + ph) - max(y0, -ph);
    const int w = min(x0 + kw, W + pw) - max(x0, -pw);
    return h * w;
}

__kernel void avg_pool2d_forward_general(
    __global float* output, __global const float* input,
    ulong out_offset, ulong in_offset,
    int NC, int H, int W, int OH, int OW,
    int kh, int kw, int sy, int sx, int py, int px, int include_pad, int divisor, int fp16
) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int nc = get_global_id(2);
    if (ox >= OW || oy >= OH || nc >= NC) return;
    const bool h = fp16 != 0;
    const int y0 = oy * sy - py;
    const int x0 = ox * sx - px;
    const int y_start = max(0, y0);
    const int y_end = min(H, y0 + kh);
    const int x_start = max(0, x0);
    const int x_end = min(W, x0 + kw);
    const ulong in_plane = in_offset + (ulong)nc * (H * W);
    float sum = 0.0f;
    for (int y = y_start; y < y_end; ++y) {
        const ulong row_base = in_plane + (ulong)y * W;
        for (int x = x_start; x < x_end; ++x) {
            sum += read_float(input, row_base + x, h);
        }
    }
    const int div = pool_divisor_i32(y0, x0, H, W, kh, kw, py, px, include_pad, divisor);
    write_float(output, out_offset + ((ulong)nc * OH + oy) * OW + ox, sum / (float)div, h);
}

__kernel void avg_pool2d_backward_general(
    __global float* grad_in, __global const float* grad_out,
    ulong out_offset, ulong grad_offset,
    int NC, int H, int W, int OH, int OW,
    int kh, int kw, int sy, int sx, int py, int px, int include_pad, int divisor, int fp16
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int nc = get_global_id(2);
    if (x >= W || y >= H || nc >= NC) return;
    const bool h = fp16 != 0;
    const int yp = y + py;
    const int xp = x + px;
    const int first_y = max(0, (yp - kh + sy) / sy);
    const int first_x = max(0, (xp - kw + sx) / sx);
    const int last_y = min(OH - 1, yp / sy);
    const int last_x = min(OW - 1, xp / sx);
    const ulong o_plane = grad_offset + (ulong)nc * OH * OW;
    float sum = 0.0f;
    for (int oy = first_y; oy <= last_y; ++oy) {
        const int y0 = oy * sy - py;
        const ulong row_base = o_plane + (ulong)oy * OW;
        for (int ox = first_x; ox <= last_x; ++ox) {
            const int x0 = ox * sx - px;
            const int div = pool_divisor_i32(y0, x0, H, W, kh, kw, py, px, include_pad, divisor);
            sum += read_float(grad_out, row_base + ox, h) / (float)div;
        }
    }
    write_float(grad_in, out_offset + ((ulong)nc * H + y) * W + x, sum, h);
}
)CLC"
};

}  // namespace iris
