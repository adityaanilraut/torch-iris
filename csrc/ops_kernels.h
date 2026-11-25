#pragma once

#include <array>

namespace iris {

inline constexpr std::array ops_extra_kernels{
// Chunk 1: BatchNorm & GroupNorm
R"CLC(
inline float wg_sum_128(__local float* smem, uint lane, float val) {
#if defined(IRIS_HAS_SUBGROUPS)
    val = sub_group_reduce_add(val);
    const uint sg_id = get_sub_group_id();
    const uint sg_lane = get_sub_group_local_id();
    const uint num_sg = get_num_sub_groups();
    if (sg_lane == 0) smem[sg_id] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    float total = 0.0f;
    if (lane == 0) {
        for (uint i = 0; i < num_sg; ++i) total += smem[i];
        smem[0] = total;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return smem[0];
#else
    smem[lane] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = get_local_size(0) >> 1; s > 0; s >>= 1) {
        if (lane < s) smem[lane] += smem[lane + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float total = smem[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
#endif
}

__kernel void bn_stats_fp32(
    __global const float* input,
    __global float* save_mean, __global float* save_invstd,
    __global float* running_mean, __global float* running_var,
    int N, int C, int S, float momentum, float eps, int update_running,
    int in_fp16, int stat_fp16
) {
    __local float smem[128];
    const int c = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const int M = N * S;
    const bool ih = in_fp16 != 0;
    const bool sh = stat_fp16 != 0;
    float sum = 0.0f;
    for (int idx = (int)lane; idx < M; idx += (int)group) {
        const int n = idx / S;
        const int s = idx - n * S;
        sum += read_float(input, ((ulong)n * C + c) * S + s, ih);
    }
    const float mu = wg_sum_128(smem, lane, sum) / (float)M;
    float sq = 0.0f;
    for (int idx = (int)lane; idx < M; idx += (int)group) {
        const int n = idx / S;
        const int s = idx - n * S;
        const float d = read_float(input, ((ulong)n * C + c) * S + s, ih) - mu;
        sq = fma(d, d, sq);
    }
    const float sum_sq = wg_sum_128(smem, lane, sq);
    if (lane == 0) {
        const float var = sum_sq / (float)M;
        write_float(save_mean, c, mu, sh);
        write_float(save_invstd, c, rsqrt(var + eps), sh);
        if (update_running && running_mean && running_var) {
            const float var_unbiased = M > 1 ? (sum_sq / (float)(M - 1)) : 0.0f;
            write_float(running_mean, c, (1.0f - momentum) * read_float(running_mean, c, sh) + momentum * mu, sh);
            write_float(running_var, c, (1.0f - momentum) * read_float(running_var, c, sh) + momentum * var_unbiased, sh);
        }
    }
}

__kernel void bn_transform_fp32(
    __global float* output, __global const float* input,
    __global const float* weight, __global const float* bias,
    __global const float* mean_buf, __global const float* var_or_invstd,
    int N, int C, int S, float eps, int is_invstd,
    int in_fp16, int param_fp16, int stat_fp16
) {
    const ulong idx = get_global_id(0);
    const ulong total = (ulong)N * C * S;
    if (idx >= total) return;
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const bool sh = stat_fp16 != 0;
    const int c = (int)((idx / (ulong)S) % (ulong)C);
    const float mu = read_float(mean_buf, c, sh);
    const float v_or_i = read_float(var_or_invstd, c, sh);
    const float inv = is_invstd ? v_or_i : rsqrt(v_or_i + eps);
    const float w = weight ? read_float(weight, c, ph) : 1.0f;
    const float b = bias ? read_float(bias, c, ph) : 0.0f;
    write_float(output, idx, fma((read_float(input, idx, ih) - mu) * inv, w, b), ih);
}

__kernel void bn_backward_reduce_fp32(
    __global const float* grad_out, __global const float* input,
    __global const float* mean_buf, __global const float* var_or_invstd,
    __global float* sum_dy_out, __global float* sum_dy_xmu_out,
    __global float* grad_weight, __global float* grad_bias,
    int N, int C, int S, float eps, int is_invstd,
    int in_fp16, int param_fp16, int stat_fp16
) {
    __local float smem[128];
    const int c = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const int M = N * S;
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const bool sh = stat_fp16 != 0;
    const float mu = read_float(mean_buf, c, sh);
    const float v_or_i = read_float(var_or_invstd, c, sh);
    const float inv = is_invstd ? v_or_i : rsqrt(v_or_i + eps);
    float s_dy = 0.0f;
    float s_dy_xmu = 0.0f;
    for (int idx = (int)lane; idx < M; idx += (int)group) {
        const int n = idx / S;
        const int s = idx - n * S;
        const ulong off = ((ulong)n * C + c) * S + s;
        const float dy = read_float(grad_out, off, ih);
        s_dy += dy;
        s_dy_xmu = fma(dy, read_float(input, off, ih) - mu, s_dy_xmu);
    }
    const float total_dy = wg_sum_128(smem, lane, s_dy);
    const float total_dy_xmu = wg_sum_128(smem, lane, s_dy_xmu);
    if (lane == 0) {
        if (sum_dy_out) sum_dy_out[c] = total_dy;
        if (sum_dy_xmu_out) sum_dy_xmu_out[c] = total_dy_xmu;
        if (grad_weight) write_float(grad_weight, c, total_dy_xmu * inv, ph);
        if (grad_bias) write_float(grad_bias, c, total_dy, ph);
    }
}

__kernel void bn_backward_input_fp32(
    __global float* grad_in, __global const float* grad_out, __global const float* input,
    __global const float* weight, __global const float* mean_buf, __global const float* var_or_invstd,
    __global const float* sum_dy, __global const float* sum_dy_xmu,
    int N, int C, int S, float eps, int train,
    int in_fp16, int param_fp16, int stat_fp16
) {
    const ulong idx = get_global_id(0);
    const ulong total = (ulong)N * C * S;
    if (idx >= total) return;
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const bool sh = stat_fp16 != 0;
    const int c = (int)((idx / (ulong)S) % (ulong)C);
    const float w = weight ? read_float(weight, c, ph) : 1.0f;
    const float dy = read_float(grad_out, idx, ih);
    if (train) {
        const float mu = read_float(mean_buf, c, sh);
        const float inv = read_float(var_or_invstd, c, sh);
        const float invM = 1.0f / (float)(N * S);
        const float k = sum_dy_xmu[c] * inv * inv * invM;
        const float m_dy = sum_dy[c] * invM;
        write_float(grad_in, idx, (dy - m_dy - (read_float(input, idx, ih) - mu) * k) * inv * w, ih);
    } else {
        const float inv = rsqrt(read_float(var_or_invstd, c, sh) + eps);
        write_float(grad_in, idx, dy * inv * w, ih);
    }
}

__kernel void gn_forward_fp32(
    __global float* output, __global float* mean_out, __global float* rstd_out,
    __global const float* input, __global const float* weight, __global const float* bias,
    int N, int C, int HxW, int G, float eps, int in_fp16, int param_fp16
) {
    __local float smem[128];
    const int ng = get_group_id(0);
    const int n = ng / G;
    const int g = ng - n * G;
    const int D = C / G;
    const int group_size = D * HxW;
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const ulong base = ((ulong)n * C + (ulong)g * D) * HxW;
    float sum = 0.0f;
    for (int i = (int)lane; i < group_size; i += (int)wg) {
        sum += read_float(input, base + i, ih);
    }
    const float mu = wg_sum_128(smem, lane, sum) / (float)group_size;
    float sq = 0.0f;
    for (int i = (int)lane; i < group_size; i += (int)wg) {
        const float diff = read_float(input, base + i, ih) - mu;
        sq = fma(diff, diff, sq);
    }
    const float rstd = rsqrt(wg_sum_128(smem, lane, sq) / (float)group_size + eps);
    if (lane == 0) {
        write_float(mean_out, ng, mu, ih);
        write_float(rstd_out, ng, rstd, ih);
    }
    for (int i = (int)lane; i < group_size; i += (int)wg) {
        const int c = g * D + i / HxW;
        const float w = weight ? read_float(weight, c, ph) : 1.0f;
        const float b = bias ? read_float(bias, c, ph) : 0.0f;
        write_float(output, base + i, fma((read_float(input, base + i, ih) - mu) * rstd, w, b), ih);
    }
}

__kernel void gn_backward_input_fp32(
    __global float* grad_in, __global const float* grad_out, __global const float* input,
    __global const float* mean_buf, __global const float* rstd_buf, __global const float* weight,
    int N, int C, int HxW, int G, int in_fp16, int param_fp16
) {
    __local float smem[128];
    const int ng = get_group_id(0);
    const int n = ng / G;
    const int g = ng - n * G;
    const int D = C / G;
    const int group_size = D * HxW;
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const ulong base = ((ulong)n * C + (ulong)g * D) * HxW;
    const float mu = read_float(mean_buf, ng, ih);
    const float rstd = read_float(rstd_buf, ng, ih);
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int i = (int)lane; i < group_size; i += (int)wg) {
        const int c = g * D + i / HxW;
        const float w = weight ? read_float(weight, c, ph) : 1.0f;
        const float dy_w = read_float(grad_out, base + i, ih) * w;
        const float xhat = (read_float(input, base + i, ih) - mu) * rstd;
        s1 += dy_w;
        s2 = fma(dy_w, xhat, s2);
    }
    const float c1 = wg_sum_128(smem, lane, s1) / (float)group_size;
    const float c2 = wg_sum_128(smem, lane, s2) / (float)group_size;
    for (int i = (int)lane; i < group_size; i += (int)wg) {
        const int c = g * D + i / HxW;
        const float w = weight ? read_float(weight, c, ph) : 1.0f;
        const float dy_w = read_float(grad_out, base + i, ih) * w;
        const float xhat = (read_float(input, base + i, ih) - mu) * rstd;
        write_float(grad_in, base + i, (dy_w - c1 - xhat * c2) * rstd, ih);
    }
}

__kernel void gn_backward_affine_fp32(
    __global float* grad_weight, __global float* grad_bias,
    __global const float* grad_out, __global const float* input,
    __global const float* mean_buf, __global const float* rstd_buf,
    int N, int C, int HxW, int G, int in_fp16, int param_fp16
) {
    __local float smem[128];
    const int c = get_group_id(0);
    const int D = C / G;
    const int g = c / D;
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const int M = N * HxW;
    float sw = 0.0f;
    float sb = 0.0f;
    for (int idx = (int)lane; idx < M; idx += (int)wg) {
        const int n = idx / HxW;
        const int s = idx - n * HxW;
        const ulong off = ((ulong)n * C + c) * HxW + s;
        const float dy = read_float(grad_out, off, ih);
        sb += dy;
        if (grad_weight) {
            const int ng = n * G + g;
            const float xhat = (read_float(input, off, ih) - read_float(mean_buf, ng, ih)) * read_float(rstd_buf, ng, ih);
            sw = fma(dy, xhat, sw);
        }
    }
    const float total_b = wg_sum_128(smem, lane, sb);
    const float total_w = grad_weight ? wg_sum_128(smem, lane, sw) : 0.0f;
    if (lane == 0) {
        if (grad_weight) write_float(grad_weight, c, total_w, ph);
        if (grad_bias) write_float(grad_bias, c, total_b, ph);
    }
}
)CLC",
// Chunk 2: Fast Subgroup LayerNorm, RMSNorm, AdaptiveAvgPool2d (FP32 & FP16)
R"CLC(
__kernel void ln_forward_fast(
    __global float* output, __global const float* input,
    __global const float* weight, __global const float* bias,
    __global float* mean_out, __global float* rstd_out,
    ulong out_off, ulong in_off, ulong w_off, ulong b_off, ulong m_off, ulong r_off,
    int width, float eps, int in_fp16, int param_fp16
) {
    __local float smem[128];
    const ulong row = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const ulong base = row * (ulong)width;
    float sum = 0.0f;
    for (int j = (int)lane; j < width; j += (int)wg) sum += read_float(input, in_off + base + j, ih);
    const float mu = wg_sum_128(smem, lane, sum) / (float)width;
    float sq = 0.0f;
    for (int j = (int)lane; j < width; j += (int)wg) {
        const float d = read_float(input, in_off + base + j, ih) - mu;
        sq = fma(d, d, sq);
    }
    const float rstd = rsqrt(wg_sum_128(smem, lane, sq) / (float)width + eps);
    if (lane == 0) {
        write_float(mean_out, m_off + row, mu, ih);
        write_float(rstd_out, r_off + row, rstd, ih);
    }
    for (int j = (int)lane; j < width; j += (int)wg) {
        const float xhat = (read_float(input, in_off + base + j, ih) - mu) * rstd;
        const float w = weight ? read_float(weight, w_off + j, ph) : 1.0f;
        const float b = bias ? read_float(bias, b_off + j, ph) : 0.0f;
        write_float(output, out_off + base + j, fma(xhat, w, b), ih);
    }
}

__kernel void ln_grad_input_fast(
    __global float* dx, __global const float* grad, __global const float* input,
    __global const float* weight, __global const float* mean_buf, __global const float* rstd_buf,
    ulong dx_off, ulong g_off, ulong in_off, ulong w_off, ulong m_off, ulong r_off,
    int width, int in_fp16, int param_fp16
) {
    __local float smem[128];
    const ulong row = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const ulong base = row * (ulong)width;
    const float mu = read_float(mean_buf, m_off + row, ih);
    const float rstd = read_float(rstd_buf, r_off + row, ih);
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int j = (int)lane; j < width; j += (int)wg) {
        const float xhat = (read_float(input, in_off + base + j, ih) - mu) * rstd;
        const float dy_w = read_float(grad, g_off + base + j, ih) * (weight ? read_float(weight, w_off + j, ph) : 1.0f);
        s1 += dy_w;
        s2 = fma(dy_w, xhat, s2);
    }
    const float c1 = wg_sum_128(smem, lane, s1) / (float)width;
    const float c2 = wg_sum_128(smem, lane, s2) / (float)width;
    for (int j = (int)lane; j < width; j += (int)wg) {
        const float xhat = (read_float(input, in_off + base + j, ih) - mu) * rstd;
        const float dy_w = read_float(grad, g_off + base + j, ih) * (weight ? read_float(weight, w_off + j, ph) : 1.0f);
        write_float(dx, dx_off + base + j, (dy_w - c1 - xhat * c2) * rstd, ih);
    }
}

__kernel void ln_grad_affine_fast(
    __global float* dw, __global float* db,
    __global const float* grad, __global const float* input,
    __global const float* mean_buf, __global const float* rstd_buf,
    ulong dw_off, ulong db_off, ulong g_off, ulong in_off, ulong m_off, ulong r_off,
    int rows, int width, int in_fp16, int param_fp16
) {
    const int col = get_global_id(0);
    if (col >= width) return;
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    float sb = 0.0f;
    float sw = 0.0f;
    for (int r = 0; r < rows; ++r) {
        const ulong i = (ulong)r * width + col;
        const float g = read_float(grad, g_off + i, ih);
        sb += g;
        if (dw) {
            const float xhat = (read_float(input, in_off + i, ih) - read_float(mean_buf, m_off + r, ih)) * read_float(rstd_buf, r_off + r, ih);
            sw = fma(g, xhat, sw);
        }
    }
    if (dw) write_float(dw, dw_off + col, sw, ph);
    if (db) write_float(db, db_off + col, sb, ph);
}

__kernel void rms_norm_forward_fp32(
    __global float* output, __global float* rstd_out,
    __global const float* input, __global const float* weight,
    ulong out_off, ulong in_off, ulong w_off,
    int width, float eps, int in_fp16, int param_fp16
) {
    __local float smem[128];
    const ulong row = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const ulong base = row * (ulong)width;
    float sq = 0.0f;
    for (int j = (int)lane; j < width; j += (int)wg) {
        const float x = read_float(input, in_off + base + j, ih);
        sq = fma(x, x, sq);
    }
    const float rstd = rsqrt(wg_sum_128(smem, lane, sq) / (float)width + eps);
    if (lane == 0 && rstd_out) {
        rstd_out[row] = rstd;
    }
    for (int j = (int)lane; j < width; j += (int)wg) {
        const float w = weight ? read_float(weight, w_off + j, ph) : 1.0f;
        write_float(output, out_off + base + j, read_float(input, in_off + base + j, ih) * rstd * w, ih);
    }
}

__kernel void rms_norm_grad_input_fp32(
    __global float* dx, __global const float* grad, __global const float* input,
    __global const float* weight, __global const float* rstd_buf,
    ulong dx_off, ulong g_off, ulong in_off, ulong w_off,
    int width, int in_fp16, int param_fp16
) {
    __local float smem[128];
    const ulong row = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    const ulong base = row * (ulong)width;
    const float rstd = rstd_buf[row];
    float dot = 0.0f;
    for (int j = (int)lane; j < width; j += (int)wg) {
        const float dy_w = read_float(grad, g_off + base + j, ih) * (weight ? read_float(weight, w_off + j, ph) : 1.0f);
        dot = fma(dy_w, read_float(input, in_off + base + j, ih), dot);
    }
    const float c = wg_sum_128(smem, lane, dot) * rstd * rstd / (float)width;
    for (int j = (int)lane; j < width; j += (int)wg) {
        const float dy_w = read_float(grad, g_off + base + j, ih) * (weight ? read_float(weight, w_off + j, ph) : 1.0f);
        write_float(dx, dx_off + base + j, (dy_w - read_float(input, in_off + base + j, ih) * c) * rstd, ih);
    }
}

__kernel void rms_norm_grad_weight_fp32(
    __global float* dw, __global const float* grad, __global const float* input,
    __global const float* rstd_buf,
    ulong dw_off, ulong g_off, ulong in_off,
    int rows, int width, int in_fp16, int param_fp16
) {
    const int col = get_global_id(0);
    if (col >= width) return;
    const bool ih = in_fp16 != 0;
    const bool ph = param_fp16 != 0;
    float sw = 0.0f;
    for (int r = 0; r < rows; ++r) {
        const ulong i = (ulong)r * width + col;
        sw = fma(read_float(grad, g_off + i, ih), read_float(input, in_off + i, ih) * rstd_buf[r], sw);
    }
    write_float(dw, dw_off + col, sw, ph);
}

__kernel void adaptive_avg_pool2d_1x1(
    __global float* output, __global const float* input, int NC, int HW, int fp16
) {
    __local float smem[128];
    const int nc = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool h = fp16 != 0;
    const ulong base = (ulong)nc * HW;
    float sum = 0.0f;
    for (int i = (int)lane; i < HW; i += (int)wg) sum += read_float(input, base + i, h);
    const float total = wg_sum_128(smem, lane, sum);
    if (lane == 0) write_float(output, nc, total / (float)HW, h);
}

__kernel void adaptive_avg_pool2d_fwd(
    __global float* output, __global const float* input,
    int NC, int H, int W, int OH, int OW, int fp16
) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int nc = get_global_id(2);
    if (ox >= OW || oy >= OH || nc >= NC) return;
    const bool h = fp16 != 0;
    const int h0 = (oy * H) / OH;
    const int h1 = ((oy + 1) * H + OH - 1) / OH;
    const int w0 = (ox * W) / OW;
    const int w1 = ((ox + 1) * W + OW - 1) / OW;
    const ulong plane = (ulong)nc * H * W;
    float sum = 0.0f;
    for (int y = h0; y < h1; ++y) {
        const ulong row = plane + (ulong)y * W;
        for (int x = w0; x < w1; ++x) sum += read_float(input, row + x, h);
    }
    write_float(output, ((ulong)nc * OH + oy) * OW + ox, sum / (float)((h1 - h0) * (w1 - w0)), h);
}

__kernel void adaptive_avg_pool2d_bwd(
    __global float* grad_in, __global const float* grad_out,
    int NC, int H, int W, int OH, int OW, int fp16
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int nc = get_global_id(2);
    if (x >= W || y >= H || nc >= NC) return;
    const bool h = fp16 != 0;
    const int oh0 = (y * OH) / H;
    const int oh1 = min(OH - 1, ((y + 1) * OH - 1) / H);
    const int ow0 = (x * OW) / W;
    const int ow1 = min(OW - 1, ((x + 1) * OW - 1) / W);
    const ulong oplane = (ulong)nc * OH * OW;
    float sum = 0.0f;
    for (int oy = oh0; oy <= oh1; ++oy) {
        const int kh = ((oy + 1) * H + OH - 1) / OH - (oy * H) / OH;
        const ulong row = oplane + (ulong)oy * OW;
        for (int ox = ow0; ox <= ow1; ++ox) {
            const int kw = ((ox + 1) * W + OW - 1) / OW - (ox * W) / OW;
            sum += read_float(grad_out, row + ox, h) / (float)(kh * kw);
        }
    }
    write_float(grad_in, ((ulong)nc * H + y) * W + x, sum, h);
}
)CLC",
// Chunk 3: Upsample Nearest2d, Bilinear2d, and ConvTranspose2d forward (FP32 & FP16)
R"CLC(
__kernel void upsample_nearest2d_fwd(
    __global float* output, __global const float* input,
    int NC, int H, int W, int OH, int OW, float scale_h, float scale_w, int fp16
) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int nc = get_global_id(2);
    if (ox >= OW || oy >= OH || nc >= NC) return;
    const bool h = fp16 != 0;
    const int iy = min((int)floor((float)oy * scale_h), H - 1);
    const int ix = min((int)floor((float)ox * scale_w), W - 1);
    write_float(output, ((ulong)nc * OH + oy) * OW + ox, read_float(input, ((ulong)nc * H + iy) * W + ix, h), h);
}

__kernel void upsample_nearest2d_bwd(
    __global float* grad_in, __global const float* grad_out,
    int NC, int H, int W, int OH, int OW, float scale_h, float scale_w, int fp16
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int nc = get_global_id(2);
    if (ix >= W || iy >= H || nc >= NC) return;
    const bool h = fp16 != 0;
    const int oy_start = max(0, (int)floor((float)iy / scale_h) - 1);
    const int oy_end = (iy == H - 1) ? OH : min(OH, (int)ceil((float)(iy + 1) / scale_h) + 1);
    const int ox_start = max(0, (int)floor((float)ix / scale_w) - 1);
    const int ox_end = (ix == W - 1) ? OW : min(OW, (int)ceil((float)(ix + 1) / scale_w) + 1);
    const ulong oplane = (ulong)nc * OH * OW;
    float sum = 0.0f;
    for (int oy = oy_start; oy < oy_end; ++oy) {
        if (min((int)floor((float)oy * scale_h), H - 1) != iy) continue;
        const ulong row = oplane + (ulong)oy * OW;
        for (int ox = ox_start; ox < ox_end; ++ox) {
            if (min((int)floor((float)ox * scale_w), W - 1) == ix) {
                sum += read_float(grad_out, row + ox, h);
            }
        }
    }
    write_float(grad_in, ((ulong)nc * H + iy) * W + ix, sum, h);
}

__kernel void upsample_bilinear2d_fwd(
    __global float* output, __global const float* input,
    int NC, int H, int W, int OH, int OW, float rheight, float rwidth, int align_corners, int fp16
) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int nc = get_global_id(2);
    if (ox >= OW || oy >= OH || nc >= NC) return;
    const bool h = fp16 != 0;
    float h1r = align_corners ? (rheight * (float)oy) : (rheight * ((float)oy + 0.5f) - 0.5f);
    h1r = fmax(0.0f, h1r);
    const int h1 = min((int)h1r, H - 1);
    const int h1p = (h1 < H - 1) ? 1 : 0;
    const float h1lambda = h1r - (float)h1;
    const float h0lambda = 1.0f - h1lambda;

    float w1r = align_corners ? (rwidth * (float)ox) : (rwidth * ((float)ox + 0.5f) - 0.5f);
    w1r = fmax(0.0f, w1r);
    const int w1 = min((int)w1r, W - 1);
    const int w1p = (w1 < W - 1) ? 1 : 0;
    const float w1lambda = w1r - (float)w1;
    const float w0lambda = 1.0f - w1lambda;

    const ulong plane = (ulong)nc * H * W;
    const float v00 = read_float(input, plane + (ulong)h1 * W + w1, h);
    const float v01 = read_float(input, plane + (ulong)h1 * W + w1 + w1p, h);
    const float v10 = read_float(input, plane + (ulong)(h1 + h1p) * W + w1, h);
    const float v11 = read_float(input, plane + (ulong)(h1 + h1p) * W + w1 + w1p, h);
    write_float(output, ((ulong)nc * OH + oy) * OW + ox,
        h0lambda * (w0lambda * v00 + w1lambda * v01) +
        h1lambda * (w0lambda * v10 + w1lambda * v11), h);
}

__kernel void upsample_bilinear2d_bwd(
    __global float* grad_in, __global const float* grad_out,
    int NC, int H, int W, int OH, int OW, float rheight, float rwidth, int align_corners, int fp16
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int nc = get_global_id(2);
    if (ix >= W || iy >= H || nc >= NC) return;
    const bool h = fp16 != 0;
    int oy0 = 0, oy1 = OH;
    int ox0 = 0, ox1 = OW;
    if (rheight > 0.0f) {
        const float lo = align_corners ? ((float)(iy - 1) / rheight) : (((float)(iy - 1) + 0.5f) / rheight - 0.5f);
        const float hi = align_corners ? ((float)(iy + 1) / rheight) : (((float)(iy + 1) + 0.5f) / rheight - 0.5f);
        oy0 = max(0, (int)floor(lo) - 1);
        oy1 = min(OH, (int)ceil(hi) + 2);
        if (iy == 0) oy0 = 0;
        if (iy == H - 1) oy1 = OH;
    }
    if (rwidth > 0.0f) {
        const float lo = align_corners ? ((float)(ix - 1) / rwidth) : (((float)(ix - 1) + 0.5f) / rwidth - 0.5f);
        const float hi = align_corners ? ((float)(ix + 1) / rwidth) : (((float)(ix + 1) + 0.5f) / rwidth - 0.5f);
        ox0 = max(0, (int)floor(lo) - 1);
        ox1 = min(OW, (int)ceil(hi) + 2);
        if (ix == 0) ox0 = 0;
        if (ix == W - 1) ox1 = OW;
    }
    const ulong oplane = (ulong)nc * OH * OW;
    float sum = 0.0f;
    for (int oy = oy0; oy < oy1; ++oy) {
        float h1r = align_corners ? (rheight * (float)oy) : (rheight * ((float)oy + 0.5f) - 0.5f);
        h1r = fmax(0.0f, h1r);
        const int h1 = min((int)h1r, H - 1);
        const int h1p = (h1 < H - 1) ? 1 : 0;
        const float h1lambda = h1r - (float)h1;
        const float h0lambda = 1.0f - h1lambda;
        float hy = 0.0f;
        if (h1 == iy) hy += h0lambda;
        if (h1 + h1p == iy) hy += h1lambda;
        if (hy == 0.0f) continue;
        const ulong row = oplane + (ulong)oy * OW;
        for (int ox = ox0; ox < ox1; ++ox) {
            float w1r = align_corners ? (rwidth * (float)ox) : (rwidth * ((float)ox + 0.5f) - 0.5f);
            w1r = fmax(0.0f, w1r);
            const int w1 = min((int)w1r, W - 1);
            const int w1p = (w1 < W - 1) ? 1 : 0;
            const float w1lambda = w1r - (float)w1;
            const float w0lambda = 1.0f - w1lambda;
            float wx = 0.0f;
            if (w1 == ix) wx += w0lambda;
            if (w1 + w1p == ix) wx += w1lambda;
            if (wx != 0.0f) {
                sum = fma(read_float(grad_out, row + ox, h), hy * wx, sum);
            }
        }
    }
    write_float(grad_in, ((ulong)nc * H + iy) * W + ix, sum, h);
}

// General ConvTranspose2d forward (output [N, C_out, OH, OW], input [N, C_in, H, W], weight [C_in, C_out/g, KH, KW])
__kernel void conv_transpose2d_fwd_general(
    __global float* output, __global const float* input, __global const float* weights,
    __global const float* bias, ulong out_offset, ulong in_offset, ulong w_offset, ulong bias_offset,
    int N, int C_in, int H, int W, int C_out, int OH, int OW,
    int KH, int KW, int sy, int sx, int py, int px, int dy, int dx, int groups, int fp16
) {
    const ulong pos = get_global_id(0);
    const ulong total = (ulong)N * C_out * OH * OW;
    if (pos >= total) return;
    const bool h = fp16 != 0;
    const int ox = (int)(pos % (ulong)OW);
    const ulong t0 = pos / (ulong)OW;
    const int oy = (int)(t0 % (ulong)OH);
    const ulong t1 = t0 / (ulong)OH;
    const int oc = (int)(t1 % (ulong)C_out);
    const int n = (int)(t1 / (ulong)C_out);

    const int in_per_g = C_in / groups;
    const int out_per_g = C_out / groups;
    const int g = oc / out_per_g;
    const int oc_in_g = oc - g * out_per_g;
    const int ic_start = g * in_per_g;
    const int hw = H * W;
    const int khw = KH * KW;
    const ulong in_n_base = in_offset + (ulong)n * C_in * hw;

    float sum = bias ? read_float(bias, bias_offset + oc, h) : 0.0f;
    for (int ky = 0; ky < KH; ++ky) {
        const int y_num = oy + py - ky * dy;
        if (y_num < 0) continue;
        const int iy = y_num / sy;
        if (iy * sy != y_num || iy >= H) continue;
        for (int kx = 0; kx < KW; ++kx) {
            const int x_num = ox + px - kx * dx;
            if (x_num < 0) continue;
            const int ix = x_num / sx;
            if (ix * sx != x_num || ix >= W) continue;
            const int i_pix = iy * W + ix;
            const int k_off = ky * KW + kx;
            for (int ic = ic_start; ic < ic_start + in_per_g; ++ic) {
                const ulong wi = w_offset + ((ulong)ic * out_per_g + oc_in_g) * khw + k_off;
                const ulong ii = in_n_base + (ulong)ic * hw + i_pix;
                sum = fma(read_float(input, ii, h), read_float(weights, wi, h), sum);
            }
        }
    }
    write_float(output, out_offset + pos, sum, h);
}
)CLC",
// Chunk 4: Argmax/Argmin, Max/Min dim, Bitonic Sort, TopK, Cumsum
R"CLC(
inline bool better_arg_f32(float cand_v, long cand_i, float best_v, long best_i, uint is_min) {
    if (best_i < 0) return true;
    const bool c_nan = isnan(cand_v);
    const bool b_nan = isnan(best_v);
    if (c_nan && !b_nan) return true;
    if (!c_nan && b_nan) return false;
    if (c_nan && b_nan) return cand_i < best_i;
    if (cand_v == best_v) return cand_i < best_i;
    return is_min ? (cand_v < best_v) : (cand_v > best_v);
}

inline bool better_arg_i64(long cand_v, long cand_i, long best_v, long best_i, uint is_min) {
    if (best_i < 0) return true;
    if (cand_v == best_v) return cand_i < best_i;
    return is_min ? (cand_v < best_v) : (cand_v > best_v);
}

__kernel void argmax_argmin_kernel(
    __global long* out_indices, __global uchar* out_values,
    __global const uchar* input,
    ulong outer_size, ulong dim_size, ulong inner_size,
    uint type, uint is_min, uint write_values
) {
    __local float sf[128];
    __local long sl[128];
    __local long si[128];
    const ulong line = get_group_id(0);
    if (line >= outer_size * inner_size) return;
    const ulong outer = line / inner_size;
    const ulong inner = line - outer * inner_size;
    const ulong base = outer * dim_size * inner_size + inner;
    const uint lane = get_local_id(0);
    const uint wg = get_local_size(0);
    const bool is_fp = (type == VALUE_FLOAT || type == VALUE_HALF);

    if (is_fp) {
        float best_v = 0.0f;
        long best_i = -1;
        for (ulong j = lane; j < dim_size; j += wg) {
            const float v = read_float((__global const float*)input, base + j * inner_size, type == VALUE_HALF);
            if (better_arg_f32(v, (long)j, best_v, best_i, is_min)) {
                best_v = v;
                best_i = (long)j;
            }
        }
        sf[lane] = best_v;
        si[lane] = best_i;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = wg >> 1; s > 0; s >>= 1) {
            if (lane < s && si[lane + s] >= 0) {
                if (better_arg_f32(sf[lane + s], si[lane + s], sf[lane], si[lane], is_min)) {
                    sf[lane] = sf[lane + s];
                    si[lane] = si[lane + s];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane == 0) {
            out_indices[line] = si[0];
            if (write_values) write_float((__global float*)out_values, line, sf[0], type == VALUE_HALF);
        }
    } else {
        long best_v = 0;
        long best_i = -1;
        for (ulong j = lane; j < dim_size; j += wg) {
            const long v = read_integer(input, base + j * inner_size, type);
            if (better_arg_i64(v, (long)j, best_v, best_i, is_min)) {
                best_v = v;
                best_i = (long)j;
            }
        }
        sl[lane] = best_v;
        si[lane] = best_i;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = wg >> 1; s > 0; s >>= 1) {
            if (lane < s && si[lane + s] >= 0) {
                if (better_arg_i64(sl[lane + s], si[lane + s], sl[lane], si[lane], is_min)) {
                    sl[lane] = sl[lane + s];
                    si[lane] = si[lane + s];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane == 0) {
            out_indices[line] = si[0];
            if (write_values) write_integer(out_values, line, type, sl[0]);
        }
    }
}

inline bool sort_after_f32(float va, int ia, float vb, int ib, int width, int descending) {
    const bool pa = ia >= width;
    const bool pb = ib >= width;
    if (pa != pb) return pa;
    if (pa && pb) return ia > ib;
    const bool na = isnan(va);
    const bool nb = isnan(vb);
    if (na != nb) return descending ? nb : na;
    if (na && nb) return ia > ib;
    if (va == vb) return ia > ib;
    return descending ? (va < vb) : (va > vb);
}

inline bool sort_after_i64(long va, int ia, long vb, int ib, int width, int descending) {
    const bool pa = ia >= width;
    const bool pb = ib >= width;
    if (pa != pb) return pa;
    if (pa && pb) return ia > ib;
    if (va == vb) return ia > ib;
    return descending ? (va < vb) : (va > vb);
}

// Local SLM bitonic sort for rows of width <= 1024 (P = next pow2 >= width, WG size = P/2 or min(256, P/2))
__kernel void bitonic_sort_local_f32(
    __global float* out_vals, __global long* out_idxs, __global const float* input,
    int width, int out_k, int P, int descending, int fp16
) {
    __local float sv[1024];
    __local int si[1024];
    const ulong row = get_group_id(0);
    const int tid = get_local_id(0);
    const int wg = get_local_size(0);
    const ulong in_base = row * (ulong)width;
    for (int i = tid; i < P; i += wg) {
        sv[i] = (i < width) ? read_float(input, in_base + i, fp16) : 0.0f;
        si[i] = i;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 2; k <= P; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int t = tid; t < (P >> 1); t += wg) {
                const int block = t / j;
                const int offset = t - block * j;
                const int i = block * (j << 1) + offset;
                const int ixj = i + j;
                const bool asc = ((i & k) == 0);
                const float va = sv[i], vb = sv[ixj];
                const int ia = si[i], ib = si[ixj];
                const bool swap = asc
                    ? sort_after_f32(va, ia, vb, ib, width, descending)
                    : sort_after_f32(vb, ib, va, ia, width, descending);
                if (swap) {
                    sv[i] = vb; sv[ixj] = va;
                    si[i] = ib; si[ixj] = ia;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    const ulong out_base = row * (ulong)out_k;
    for (int i = tid; i < out_k; i += wg) {
        write_float(out_vals, out_base + i, sv[i], fp16);
        out_idxs[out_base + i] = (long)si[i];
    }
}

__kernel void bitonic_sort_local_i64(
    __global uchar* out_vals, __global long* out_idxs, __global const uchar* input,
    int width, int out_k, int P, int descending, uint type
) {
    __local long sv[1024];
    __local int si[1024];
    const ulong row = get_group_id(0);
    const int tid = get_local_id(0);
    const int wg = get_local_size(0);
    const ulong in_base = row * (ulong)width;
    for (int i = tid; i < P; i += wg) {
        sv[i] = (i < width) ? read_integer(input, in_base + i, type) : 0;
        si[i] = i;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 2; k <= P; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int t = tid; t < (P >> 1); t += wg) {
                const int block = t / j;
                const int offset = t - block * j;
                const int i = block * (j << 1) + offset;
                const int ixj = i + j;
                const bool asc = ((i & k) == 0);
                const long va = sv[i], vb = sv[ixj];
                const int ia = si[i], ib = si[ixj];
                const bool swap = asc
                    ? sort_after_i64(va, ia, vb, ib, width, descending)
                    : sort_after_i64(vb, ib, va, ia, width, descending);
                if (swap) {
                    sv[i] = vb; sv[ixj] = va;
                    si[i] = ib; si[ixj] = ia;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    const ulong out_base = row * (ulong)out_k;
    for (int i = tid; i < out_k; i += wg) {
        write_integer(out_vals, out_base + i, type, sv[i]);
        out_idxs[out_base + i] = (long)si[i];
    }
}

// Fast single-launch GPU TopK for large vocabularies (width > 1024, out_k <= 256)
__kernel void topk_large_vocab_f32(
    __global float* out_vals, __global long* out_idxs, __global const float* input,
    int width, int out_k, int largest, int fp16
) {
    __local float sv[512];
    __local int si[512];
    const ulong row = get_group_id(0);
    const int tid = get_local_id(0); // 0..255
    const ulong in_base = row * (ulong)width;

    // Initialize top-256 slots with padding sentinels
    sv[tid] = 0.0f;
    si[tid] = width + tid;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int base = 0; base < width; base += 256) {
        const int pos = base + tid;
        sv[256 + tid] = (pos < width) ? read_float(input, in_base + pos, fp16) : 0.0f;
        si[256 + tid] = (pos < width) ? pos : (width + 256 + tid);
        barrier(CLK_LOCAL_MEM_FENCE);

        // Full 512-element bitonic sort in SLM (256 threads)
        for (int k = 2; k <= 512; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int block = tid / j;
                const int offset = tid - block * j;
                const int i = block * (j << 1) + offset;
                const int ixj = i + j;
                const bool asc = ((i & k) == 0);
                const float va = sv[i], vb = sv[ixj];
                const int ia = si[i], ib = si[ixj];
                const bool swap = asc
                    ? sort_after_f32(va, ia, vb, ib, width, largest)
                    : sort_after_f32(vb, ib, va, ia, width, largest);
                if (swap) {
                    sv[i] = vb; sv[ixj] = va;
                    si[i] = ib; si[ixj] = ia;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        // Keep only slots [0..out_k-1], reset [out_k..255] to padding sentinels
        if (tid >= out_k) {
            sv[tid] = 0.0f;
            si[tid] = width + tid;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong out_base = row * (ulong)out_k;
    if (tid < out_k) {
        write_float(out_vals, out_base + tid, sv[tid], fp16);
        out_idxs[out_base + tid] = (long)si[tid];
    }
}

// Global bitonic sort step for arbitrary width > 1024
__kernel void bitonic_sort_init_f32(
    __global float* buf_v, __global int* buf_i, __global const float* input,
    int rows, int width, int P, int fp16
) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong)rows * P;
    if (gid >= total) return;
    const ulong row = gid / (ulong)P;
    const int col = (int)(gid - row * (ulong)P);
    buf_v[gid] = (col < width) ? read_float(input, row * (ulong)width + col, fp16) : 0.0f;
    buf_i[gid] = col;
}

__kernel void bitonic_sort_step_f32(
    __global float* buf_v, __global int* buf_i,
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
    const float va = buf_v[base + i], vb = buf_v[base + ixj];
    const int ia = buf_i[base + i], ib = buf_i[base + ixj];
    const bool asc = ((i & k) == 0);
    const bool swap = asc
        ? sort_after_f32(va, ia, vb, ib, width, descending)
        : sort_after_f32(vb, ib, va, ia, width, descending);
    if (swap) {
        buf_v[base + i] = vb; buf_v[base + ixj] = va;
        buf_i[base + i] = ib; buf_i[base + ixj] = ia;
    }
}

__kernel void bitonic_sort_extract_f32(
    __global float* out_v, __global long* out_i,
    __global const float* buf_v, __global const int* buf_i,
    int rows, int out_k, int P, int fp16
) {
    const ulong gid = get_global_id(0);
    if (gid >= (ulong)rows * (ulong)out_k) return;
    const ulong row = gid / (ulong)out_k;
    const int col = (int)(gid - row * (ulong)out_k);
    const ulong src = row * (ulong)P + col;
    write_float(out_v, gid, buf_v[src], fp16);
    out_i[gid] = (long)buf_i[src];
}

__kernel void cumsum_kernel(
    __global uchar* output, __global const uchar* input,
    ulong outer_size, ulong dim_size, ulong inner_size,
    uint in_type, uint out_type
) {
    const ulong line = get_global_id(0);
    if (line >= outer_size * inner_size) return;
    const ulong outer = line / inner_size;
    const ulong inner = line - outer * inner_size;
    const ulong base = outer * dim_size * inner_size + inner;
    const bool out_fp = (out_type == VALUE_FLOAT || out_type == VALUE_HALF);
    const bool in_fp = (in_type == VALUE_FLOAT || in_type == VALUE_HALF);
    if (out_fp) {
        float acc = 0.0f;
        for (ulong j = 0; j < dim_size; ++j) {
            const ulong pos = base + j * inner_size;
            const float val = in_fp
                ? read_float((__global const float*)input, pos, in_type == VALUE_HALF)
                : convert_float(read_integer(input, pos, in_type));
            acc += val;
            write_float((__global float*)output, pos, acc, out_type == VALUE_HALF);
        }
    } else {
        long acc = 0;
        for (ulong j = 0; j < dim_size; ++j) {
            const ulong pos = base + j * inner_size;
            const long val = in_fp
                ? convert_long_rtz(read_float((__global const float*)input, pos, in_type == VALUE_HALF))
                : read_integer(input, pos, in_type);
            acc = as_long((ulong)acc + (ulong)val);
            write_integer(output, pos, out_type, acc);
        }
    }
}
)CLC",
// Chunk 5: Arange, Linspace, Random Int, Flip, Fused SGD/Adam/AdamW
R"CLC(
__kernel void arange_kernel(
    __global uchar* output, ulong length, float start_f, float step_f,
    long start_l, long step_l, uint type
) {
    const ulong i = get_global_id(0);
    if (i >= length) return;
    if (type == VALUE_FLOAT || type == VALUE_HALF) {
        const float val = fma((float)i, step_f, start_f);
        write_float((__global float*)output, i, val, type == VALUE_HALF);
    } else {
        const long val = as_long((ulong)start_l + i * (ulong)step_l);
        write_integer(output, i, type, val);
    }
}

__kernel void linspace_kernel(
    __global uchar* output, ulong steps, float start_f, float end_f, uint type
) {
    const ulong i = get_global_id(0);
    if (i >= steps) return;
    float val_f;
    if (steps == 1) {
        val_f = start_f;
    } else if (i == steps - 1) {
        val_f = end_f;
    } else {
        const float step = (end_f - start_f) / (float)(steps - 1);
        val_f = fma((float)i, step, start_f);
    }
    if (type == VALUE_FLOAT || type == VALUE_HALF) {
        write_float((__global float*)output, i, val_f, type == VALUE_HALF);
    } else {
        write_integer(output, i, type, (long)val_f);
    }
}

__kernel void random_int_kernel(
    __global const ulong* geometry, __global uchar* output,
    ulong length, ulong offset, ulong rank, ulong seed, ulong start,
    long low, ulong range, uint type
) {
    const ulong block = get_global_id(0);
    const uint4 bits = philox(seed, start + block);
    for (uint lane = 0; lane < 4 && block * 4 + lane < length; ++lane) {
        const ulong idx = block * 4 + lane;
        const ulong dst = offset + reduction_offset(idx, geometry, 0, rank);
        const uint r32 = lane == 0 ? bits.x : lane == 1 ? bits.y : lane == 2 ? bits.z : bits.w;
        ulong sample;
        if (range <= 0x100000000UL) {
            sample = (ulong)r32 % range;
        } else {
            const uint r32_hi = lane == 0 ? bits.y : lane == 1 ? bits.z : lane == 2 ? bits.w : bits.x;
            sample = (((ulong)r32_hi << 32) | (ulong)r32) % range;
        }
        const long val = as_long((ulong)low + sample);
        if (type == VALUE_FLOAT || type == VALUE_HALF) {
            write_float((__global float*)output, dst, (float)val, type == VALUE_HALF);
        } else {
            write_integer(output, dst, type, val);
        }
    }
}

__kernel void flip_kernel(
    __global const ulong* geometry, __global uchar* output, __global const uchar* input,
    ulong length, ulong in_offset, ulong rank, uint flip_mask, uint type
) {
    const ulong i = get_global_id(0);
    if (i >= length) return;
    ulong logical = i;
    ulong src = in_offset;
    for (ulong d = rank; d > 0; --d) {
        const ulong size = geometry[(d - 1) * 2];
        const ulong stride = geometry[(d - 1) * 2 + 1];
        ulong coord = logical % size;
        logical /= size;
        if ((flip_mask >> (uint)(d - 1)) & 1u) {
            coord = size - 1 - coord;
        }
        src += coord * stride;
    }
    copy_index_value(output, i, input, src, value_size(type));
}

__kernel void fused_sgd_fp32(
    __global float* param, __global const float* grad, __global float* mom_buf,
    __global const float* lr_ptr, __global const float* grad_scale, __global const float* found_inf,
    ulong p_off, ulong g_off, ulong m_off, ulong length,
    float weight_decay, float momentum, float lr_scalar, float dampening,
    uint nesterov, uint maximize, uint is_first_step, uint has_mom, uint has_lr_ptr
) {
    if (found_inf && found_inf[0] != 0.0f) return;
    const ulong i = get_global_id(0);
    if (i >= length) return;
    const float lr = has_lr_ptr ? lr_ptr[0] : lr_scalar;
    float g = grad[g_off + i];
    if (grad_scale) g *= grad_scale[0];
    if (maximize) g = -g;
    float p = param[p_off + i];
    if (weight_decay != 0.0f) g = fma(weight_decay, p, g);
    if (has_mom && momentum != 0.0f) {
        float buf;
        if (is_first_step) {
            buf = g;
        } else {
            buf = fma(momentum, mom_buf[m_off + i], (1.0f - dampening) * g);
        }
        mom_buf[m_off + i] = buf;
        g = nesterov ? fma(momentum, buf, g) : buf;
    }
    param[p_off + i] = fma(-lr, g, p);
}

__kernel void fused_adam_fp32(
    __global float* param, __global const float* grad,
    __global float* exp_avg, __global float* exp_avg_sq, __global float* max_exp_avg_sq,
    __global const float* state_step, __global const float* lr_ptr,
    __global const float* grad_scale, __global const float* found_inf,
    ulong p_off, ulong g_off, ulong ea_off, ulong eas_off, ulong meas_off, ulong step_off,
    ulong length,
    float lr_scalar, float beta1, float beta2, float weight_decay, float eps,
    uint amsgrad, uint maximize, uint decoupled_wd, uint has_lr_ptr
) {
    if (found_inf && found_inf[0] != 0.0f) return;
    const ulong i = get_global_id(0);
    if (i >= length) return;
    const float lr = has_lr_ptr ? lr_ptr[0] : lr_scalar;
    const float step = state_step[step_off];
    const float bc1 = 1.0f - pow(beta1, step);
    const float bc2 = 1.0f - pow(beta2, step);
    const float step_size = lr / bc1;
    const float bc2_sqrt = sqrt(bc2);

    float p = param[p_off + i];
    float g = grad[g_off + i];
    if (grad_scale) g *= grad_scale[0];
    if (maximize) g = -g;

    if (weight_decay != 0.0f) {
        if (decoupled_wd) {
            p *= (1.0f - lr * weight_decay);
        } else {
            g = fma(weight_decay, p, g);
        }
    }

    const float m = fma(beta1, exp_avg[ea_off + i], (1.0f - beta1) * g);
    const float v = fma(beta2, exp_avg_sq[eas_off + i], (1.0f - beta2) * g * g);
    exp_avg[ea_off + i] = m;
    exp_avg_sq[eas_off + i] = v;

    float denom;
    if (amsgrad) {
        const float max_v = fmax(max_exp_avg_sq[meas_off + i], v);
        max_exp_avg_sq[meas_off + i] = max_v;
        denom = sqrt(max_v) / bc2_sqrt + eps;
    } else {
        denom = sqrt(v) / bc2_sqrt + eps;
    }
    param[p_off + i] = p - step_size * (m / denom);
}
)CLC",
R"CLC(
inline float recurrent_gate(__global const float* gate, __global const float* bias,
                            ulong gate_offset, ulong bias_offset, ulong row, ulong col, uint fp16) {
    return read_float(gate, gate_offset + row + col, fp16) +
           (bias ? read_float(bias, bias_offset + col, fp16) : 0.0f);
}

__kernel void recurrent_forward(
    __global float* output, __global float* new_cell, __global float* workspace,
    __global const float* input, __global const float* hidden, __global const float* state,
    __global const float* ib, __global const float* hb,
    ulong input_offset, ulong hidden_offset, ulong state_offset, ulong ib_offset, ulong hb_offset,
    ulong width, uint lstm, uint fp16) {
    const ulong i = get_global_id(0);
    const ulong row = i / width;
    const ulong col = i % width;
    const ulong gate_row = row * width * (lstm ? 4 : 3);
    const ulong work = row * width * (lstm ? 4 : 5) + col;
    const float x0 = recurrent_gate(input, ib, input_offset, ib_offset, gate_row, col, fp16);
    const float x1 = recurrent_gate(input, ib, input_offset, ib_offset, gate_row, col + width, fp16);
    const float x2 = recurrent_gate(input, ib, input_offset, ib_offset, gate_row, col + 2 * width, fp16);
    const float h0 = recurrent_gate(hidden, hb, hidden_offset, hb_offset, gate_row, col, fp16);
    const float h1 = recurrent_gate(hidden, hb, hidden_offset, hb_offset, gate_row, col + width, fp16);
    const float h2 = recurrent_gate(hidden, hb, hidden_offset, hb_offset, gate_row, col + 2 * width, fp16);
    const float a = logistic(x0 + h0);
    const float b = logistic(x1 + h1);
    const float prev = read_float(state, state_offset + i, fp16);
    write_float(workspace, work, a, fp16);
    write_float(workspace, work + width, b, fp16);
    if (lstm) {
        const float x3 = recurrent_gate(input, ib, input_offset, ib_offset, gate_row, col + 3 * width, fp16);
        const float h3 = recurrent_gate(hidden, hb, hidden_offset, hb_offset, gate_row, col + 3 * width, fp16);
        const float candidate = tanh(x2 + h2);
        const float out = logistic(x3 + h3);
        const float cell = prev * b + a * candidate;
        write_float(workspace, work + 2 * width, candidate, fp16);
        write_float(workspace, work + 3 * width, out, fp16);
        write_float(new_cell, i, cell, fp16);
        write_float(output, i, out * tanh(cell), fp16);
    } else {
        const float candidate = tanh(x2 + a * h2);
        write_float(workspace, work + 2 * width, candidate, fp16);
        write_float(workspace, work + 3 * width, prev, fp16);
        write_float(workspace, work + 4 * width, h2, fp16);
        write_float(output, i, candidate + (prev - candidate) * b, fp16);
    }
}

__kernel void recurrent_backward(
    __global float* input_gates, __global float* hidden_gates, __global float* grad_state,
    __global const float* grad_output, __global const float* grad_cell, __global const float* state,
    __global const float* new_cell, __global const float* workspace,
    ulong go_offset, ulong gc_offset, ulong state_offset, ulong cell_offset, ulong work_offset,
    ulong width, uint lstm, uint fp16) {
    const ulong i = get_global_id(0);
    const ulong row = i / width;
    const ulong col = i % width;
    const ulong gate = row * width * (lstm ? 4 : 3) + col;
    const ulong work = work_offset + row * width * (lstm ? 4 : 5) + col;
    const float a = read_float(workspace, work, fp16);
    const float b = read_float(workspace, work + width, fp16);
    const float c = read_float(workspace, work + 2 * width, fp16);
    const float d = read_float(workspace, work + 3 * width, fp16);
    const float gh = grad_output ? read_float(grad_output, go_offset + i, fp16) : 0.0f;
    if (lstm) {
        const float prev = read_float(state, state_offset + i, fp16);
        const float cell = tanh(read_float(new_cell, cell_offset + i, fp16));
        const float gc = grad_cell ? read_float(grad_cell, gc_offset + i, fp16) : 0.0f;
        const float combined = gc + gh * d * (1.0f - cell * cell);
        write_float(input_gates, gate, combined * c * a * (1.0f - a), fp16);
        write_float(input_gates, gate + width, combined * prev * b * (1.0f - b), fp16);
        write_float(input_gates, gate + 2 * width, combined * a * (1.0f - c * c), fp16);
        write_float(input_gates, gate + 3 * width, gh * cell * d * (1.0f - d), fp16);
        write_float(grad_state, i, combined * b, fp16);
    } else {
        const float hn = read_float(workspace, work + 4 * width, fp16);
        const float dn = gh * (1.0f - b) * (1.0f - c * c);
        const float dz = gh * (d - c) * b * (1.0f - b);
        const float dr = dn * hn * a * (1.0f - a);
        write_float(input_gates, gate, dr, fp16);
        write_float(input_gates, gate + width, dz, fp16);
        write_float(input_gates, gate + 2 * width, dn, fp16);
        write_float(hidden_gates, gate, dr, fp16);
        write_float(hidden_gates, gate + width, dz, fp16);
        write_float(hidden_gates, gate + 2 * width, dn * a, fp16);
        write_float(grad_state, i, gh * b, fp16);
    }
}
)CLC"
};

}  // namespace iris
