#pragma once

#include <array>

namespace iris {

inline constexpr std::array kernel_sources{
R"CLC(
__kernel void add_tensor(
    __global float* output, __global const float* left,
    __global const float* right, ulong length, float alpha
) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        const float4 a = vload4(0, left + base);
        const float4 b = vload4(0, right + base);
        vstore4(fma((float4)(alpha), b, a), 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            output[i] = left[i] + alpha * right[i];
        }
    }
}

__kernel void multiply_tensor(
    __global float* output, __global const float* left,
    __global const float* right, ulong length
) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        vstore4(vload4(0, left + base) * vload4(0, right + base), 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            output[i] = left[i] * right[i];
        }
    }
}

__kernel void relu_tensor(__global float* output, __global const float* input, ulong length) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        const float4 x = vload4(0, input + base);
        vstore4(select(x, (float4)(0.0f), x <= (float4)(0.0f)), 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            const float value = input[i];
            output[i] = value <= 0.0f ? 0.0f : value;
        }
    }
}

__kernel void scale_tensor(__global float* output, __global const float* input, ulong length, float alpha) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        vstore4(alpha * vload4(0, input + base), 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            output[i] = alpha * input[i];
        }
    }
}

__kernel void add_scalar_tensor(__global float* output, __global const float* input, ulong length, float bias) {
    const ulong base = get_global_id(0) * 4;
    if (base + 3 < length) {
        vstore4(vload4(0, input + base) + bias, 0, output + base);
    } else {
        for (ulong i = base; i < length; ++i) output[i] = input[i] + bias;
    }
}

// Preserve the rounding of two eager FP32 operations.
#pragma OPENCL FP_CONTRACT OFF
__kernel void scale_add_tensor(
    __global float* output, __global const float* input, ulong length, float scale, float bias
) {
    const ulong base = get_global_id(0) * 4;
    if (base + 3 < length) {
        vstore4(vload4(0, input + base) * scale + bias, 0, output + base);
    } else {
        for (ulong i = base; i < length; ++i) output[i] = input[i] * scale + bias;
    }
}
#pragma OPENCL FP_CONTRACT ON

__kernel void threshold_backward_tensor(
    __global float* output, __global const float* grad, __global const float* input,
    ulong length, float threshold
) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        const float4 g = vload4(0, grad + base);
        const float4 x = vload4(0, input + base);
        vstore4(select(g, (float4)(0.0f), x <= (float4)(threshold)), 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            output[i] = input[i] <= threshold ? 0.0f : grad[i];
        }
    }
}

__kernel void squared_difference_tensor(
    __global float* output, __global const float* left, __global const float* right, ulong length
) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        const float4 d = vload4(0, left + base) - vload4(0, right + base);
        vstore4(d * d, 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            const float d = left[i] - right[i];
            output[i] = d * d;
        }
    }
}

__kernel void scaled_difference_tensor(
    __global float* output, __global const float* left, __global const float* right,
    ulong length, float alpha
) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        vstore4(alpha * (vload4(0, left + base) - vload4(0, right + base)), 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            output[i] = alpha * (left[i] - right[i]);
        }
    }
}
)CLC",
R"CLC(
// Preserve ATen's complex division and signed-zero behavior for sgn.
#pragma OPENCL FP_CONTRACT OFF
inline float2 divide_complex(float2 a, float2 b) {
    if (fabs(b.x) >= fabs(b.y)) {
        if (b.x == 0.0f && b.y == 0.0f) return a / fabs(b);
        const float ratio = b.y / b.x;
        const float scale = 1.0f / (b.x + b.y * ratio);
        return (float2)(a.x + a.y * ratio, a.y - a.x * ratio) * scale;
    }
    const float ratio = b.x / b.y;
    const float scale = 1.0f / (b.y + b.x * ratio);
    return (float2)(a.x * ratio + a.y, a.y * ratio - a.x) * scale;
}

__kernel void complex_unary(__global uchar* output, __global const float2* input,
                            ulong input_offset, uint operation) {
    enum { ABS, SGN, NEG, IS_NAN, IS_FINITE, IS_INF };
    const ulong i = get_global_id(0);
    const float2 z = input[input_offset + i];
    switch (operation) {
        case ABS:
            ((__global float*)output)[i] = hypot(z.x, z.y);
            break;
        case SGN: {
            float2 result = (float2)(0.0f);
            if (z.x != 0.0f || z.y != 0.0f) {
                result = divide_complex(z, (float2)(hypot(z.x, z.y), 0.0f));
            }
            ((__global float2*)output)[i] = result;
            break;
        }
        case NEG:
            ((__global float2*)output)[i] = -z;
            break;
        case IS_NAN:
            output[i] = isnan(z.x) || isnan(z.y);
            break;
        case IS_FINITE:
            output[i] = isfinite(z.x) && isfinite(z.y);
            break;
        case IS_INF:
            output[i] = isinf(z.x) || isinf(z.y);
            break;
    }
}

__kernel void complex_division(
    __global const ulong* geometry, __global float2* output,
    __global const float2* left, __global const float2* right,
    ulong left_offset, ulong right_offset, ulong rank
) {
    const ulong i = get_global_id(0);
    ulong logical = i;
    ulong a = left_offset, b = right_offset;
    for (ulong d = rank; d > 0; --d) {
        const __global ulong* axis = geometry + (d - 1) * 4;
        const ulong index = logical % axis[0];
        logical /= axis[0];
        a += index * axis[2];
        b += index * axis[3];
    }
    output[i] = divide_complex(left[a], right[b]);
}
#pragma OPENCL FP_CONTRACT ON

inline float read_float(__global const float* source, ulong i, bool fp16) {
    return fp16 ? vload_half(i, (__global const half*)source) : source[i];
}

inline void write_float(__global float* output, ulong i, float value, bool fp16) {
    if (fp16) vstore_half_rte(value, i, (__global half*)output);
    else output[i] = value;
}

inline const __global float* float_offset(__global const float* source, ulong i, bool fp16) {
    return (__global const float*)((__global const uchar*)source + i * (fp16 ? 2 : 4));
}

#define GEMM_ARGUMENTS \
    __global float* output, __global const float* left, __global const float* right, \
    __global const float* bias, ulong rows, ulong columns, ulong inner, \
    ulong left_offset, ulong left_batch, ulong left_row, ulong left_col, \
    ulong right_offset, ulong right_batch, ulong right_row, ulong right_col, \
    ulong bias_offset, ulong bias_row, ulong bias_col, float alpha, float beta

#define GEMM_PARAMETERS output, left, right, bias, rows, columns, inner, \
    left_offset, left_batch, left_row, left_col, right_offset, right_batch, right_row, right_col, \
    bias_offset, bias_row, bias_col, alpha, beta

inline float gemm_result(float value, __global const float* bias, ulong offset,
                         float alpha, float beta, bool fp16) {
    const float product = alpha * value;
    return beta == 0.0f ? product : beta * read_float(bias, offset, fp16) + product;
}

inline float gemm_load(__global const float* source, ulong row, ulong col,
                       ulong height, ulong width, ulong row_stride, ulong col_stride,
                       float padding, bool fp16) {
    return row < height && col < width ? read_float(source, row * row_stride + col * col_stride, fp16) : padding;
}

inline float4 gemm_load4(__global const float* source, ulong row, ulong col,
                         ulong height, ulong width, ulong row_stride, ulong col_stride,
                         float padding, bool fp16) {
    if (row >= height || col >= width) {
        return (float4)(padding);
    }
    const __global float* start = float_offset(source, row * row_stride + col * col_stride, fp16);
    if (col_stride == 1 && width - col >= 4) {
        return fp16 ? vload_half4(0, (__global const half*)start) : vload4(0, start);
    }
    return (float4)(read_float(start, 0, fp16), width - col > 1 ? read_float(start, col_stride, fp16) : padding,
                    width - col > 2 ? read_float(start, 2 * col_stride, fp16) : padding,
                    width - col > 3 ? read_float(start, 3 * col_stride, fp16) : padding);
}

inline void gemm_scalar(GEMM_ARGUMENTS, bool fp16) {
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1);
    const size_t batch = get_global_id(2);
    const ulong a = left_offset + batch * left_batch + row * left_row;
    const ulong b = right_offset + batch * right_batch + column * right_col;
    float value = 0.0f;
    for (ulong k = 0; k < inner; ++k) {
        value = fma(read_float(left, a + k * left_col, fp16), read_float(right, b + k * right_row, fp16), value);
    }
    write_float(output, (batch * rows + row) * columns + column,
                gemm_result(value, bias, bias_offset + row * bias_row + column * bias_col, alpha, beta, fp16),
                fp16);
}

__kernel void matmul(GEMM_ARGUMENTS) {
    gemm_scalar(GEMM_PARAMETERS, false);
}

__kernel void matmul_half(GEMM_ARGUMENTS) {
    gemm_scalar(GEMM_PARAMETERS, true);
}

inline void gemm_tiled(GEMM_ARGUMENTS, bool fp16,
                       __local float (*left_tile)[MATMUL_TILE_SIZE + 1],
                       __local float (*right_tile)[MATMUL_TILE_SIZE + 1]) {
    const uint x = get_local_id(0);
    const uint y = get_local_id(1);
    const size_t first_row = get_group_id(1) * MATMUL_TILE_SIZE;
    const size_t first_column = get_group_id(0) * MATMUL_TILE_SIZE;
    const size_t row = first_row + y;
    const size_t column = first_column + x;
    const size_t batch = get_group_id(2);
    left = float_offset(left, left_offset + batch * left_batch, fp16);
    right = float_offset(right, right_offset + batch * right_batch, fp16);
    float value = 0.0f;

    for (ulong base = 0; base < inner; base += MATMUL_TILE_SIZE) {
        // Map consecutive lanes to consecutive addresses in either physical layout.
        // Padded tile pitches avoid bank conflicts when storing transposed loads.
        if (left_row == 1 && left_col != 1) {
            left_tile[x][y] = gemm_load(left, first_row + x, base + y, rows, inner,
                                       left_row, left_col, -0.0f, fp16);
        } else {
            left_tile[y][x] = gemm_load(left, row, base + x, rows, inner, left_row, left_col, -0.0f, fp16);
        }
        if (right_row == 1 && right_col != 1) {
            right_tile[x][y] = gemm_load(right, base + x, first_column + y, inner, columns,
                                         right_row, right_col, 0.0f, fp16);
        } else {
            right_tile[y][x] = gemm_load(right, base + y, column, inner, columns,
                                         right_row, right_col, 0.0f, fp16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float partial = 0.0f;
        #pragma unroll
        for (uint k = 0; k < MATMUL_TILE_SIZE; ++k) {
            partial = fma(left_tile[y][k], right_tile[k][x], partial);
        }
        value += partial;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < rows && column < columns) {
        write_float(output, (batch * rows + row) * columns + column,
                    gemm_result(value, bias, bias_offset + row * bias_row + column * bias_col, alpha, beta, fp16),
                    fp16);
    }
}

__kernel void matmul_tiled(GEMM_ARGUMENTS) {
    __local float a[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE + 1];
    __local float b[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE + 1];
    gemm_tiled(GEMM_PARAMETERS, false, a, b);
}

__kernel void matmul_tiled_half(GEMM_ARGUMENTS) {
    __local float a[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE + 1];
    __local float b[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE + 1];
    gemm_tiled(GEMM_PARAMETERS, true, a, b);
}

// The large kernel keeps multiple outputs per work-item in registers.
#define MATMUL_OUTPUT_ROWS (MATMUL_BLOCK_SIZE / MATMUL_BLOCK_GROUP_Y)
#define MATMUL_OUTPUT_COLS (MATMUL_BLOCK_SIZE / MATMUL_BLOCK_GROUP_X)
#define MATMUL_BLOCK_THREADS (MATMUL_BLOCK_GROUP_X * MATMUL_BLOCK_GROUP_Y)
#define MATMUL_VECTORS_PER_THREAD (MATMUL_BLOCK_SIZE * MATMUL_TILE_SIZE / (4 * MATMUL_BLOCK_THREADS))

inline float4 gemm_block_load(__global const float* source, ulong row, ulong col,
                              ulong height, ulong width, ulong row_stride, ulong col_stride,
                              float padding, bool full_tiles, bool fp16) {
    if (!full_tiles) {
        return gemm_load4(source, row, col, height, width, row_stride, col_stride, padding, fp16);
    }
    const ulong index = row * row_stride + col;
    return fp16 ? vload_half4(0, (__global const half*)source + index) : vload4(0, source + index);
}

inline void gemm_blocked(GEMM_ARGUMENTS, bool full_tiles, bool fp16,
                         __local float (*left_tile)[MATMUL_BLOCK_SIZE + 1],
                         __local float (*right_tile)[MATMUL_BLOCK_SIZE + 1]) {
    const uint x = get_local_id(0);
    const uint y = get_local_id(1);
    const uint lane = y * MATMUL_BLOCK_GROUP_X + x;
    const size_t first_row = get_group_id(1) * MATMUL_BLOCK_SIZE;
    const size_t first_column = get_group_id(0) * MATMUL_BLOCK_SIZE;
    const size_t batch = get_group_id(2);
    left = float_offset(left, left_offset + batch * left_batch, fp16);
    right = float_offset(right, right_offset + batch * right_batch, fp16);
    float values[MATMUL_OUTPUT_ROWS][MATMUL_OUTPUT_COLS] = {0.0f};

    for (ulong base = 0; base < inner; base += MATMUL_TILE_SIZE) {
        #pragma unroll
        for (uint i = 0; i < MATMUL_VECTORS_PER_THREAD; ++i) {
            const uint offset = 4 * (lane + i * MATMUL_BLOCK_THREADS);
            const uint wide = offset % MATMUL_BLOCK_SIZE;
            const uint narrow = offset / MATMUL_BLOCK_SIZE;
            const uint r = offset / MATMUL_TILE_SIZE;
            const uint k = offset % MATMUL_TILE_SIZE;

            if (left_row == 1 && left_col != 1) {
                const float4 a = gemm_block_load(left, base + narrow, first_row + wide,
                                           inner, rows, left_col, left_row, -0.0f, full_tiles, fp16);
                vstore4(a, 0, &left_tile[narrow][wide]);
            } else {
                const float4 a = gemm_block_load(left, first_row + r, base + k,
                                           rows, inner, left_row, left_col, -0.0f, full_tiles, fp16);
                left_tile[k][r] = a.s0;
                left_tile[k + 1][r] = a.s1;
                left_tile[k + 2][r] = a.s2;
                left_tile[k + 3][r] = a.s3;
            }
            if (right_row == 1 && right_col != 1) {
                const float4 b = gemm_block_load(right, first_column + r, base + k,
                                           columns, inner, right_col, right_row, 0.0f, full_tiles, fp16);
                right_tile[k][r] = b.s0;
                right_tile[k + 1][r] = b.s1;
                right_tile[k + 2][r] = b.s2;
                right_tile[k + 3][r] = b.s3;
            } else {
                const float4 b = gemm_block_load(right, base + narrow, first_column + wide,
                                           inner, columns, right_row, right_col, 0.0f, full_tiles, fp16);
                vstore4(b, 0, &right_tile[narrow][wide]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate short K blocks separately to limit roundoff on long dot products.
        float partial[MATMUL_OUTPUT_ROWS][MATMUL_OUTPUT_COLS] = {0.0f};
        #pragma unroll
        for (uint k = 0; k < MATMUL_TILE_SIZE; ++k) {
            float right_values[MATMUL_OUTPUT_COLS];
            #pragma unroll
            for (uint j = 0; j < MATMUL_OUTPUT_COLS; ++j) {
                right_values[j] = right_tile[k][x * 4 + (j / 4) * (4 * MATMUL_BLOCK_GROUP_X) + j % 4];
            }
            #pragma unroll
            for (uint i = 0; i < MATMUL_OUTPUT_ROWS; ++i) {
                const float a = left_tile[k][y + i * MATMUL_BLOCK_GROUP_Y];
                #pragma unroll
                for (uint j = 0; j < MATMUL_OUTPUT_COLS; ++j) {
                    partial[i][j] = fma(a, right_values[j], partial[i][j]);
                }
            }
        }
        #pragma unroll
        for (uint i = 0; i < MATMUL_OUTPUT_ROWS; ++i) {
            #pragma unroll
            for (uint j = 0; j < MATMUL_OUTPUT_COLS; ++j) {
                values[i][j] += partial[i][j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (uint i = 0; i < MATMUL_OUTPUT_ROWS; ++i) {
        const size_t row = first_row + y + i * MATMUL_BLOCK_GROUP_Y;
        if (!full_tiles && row >= rows) {
            continue;
        }
        #pragma unroll
        for (uint j = 0; j < MATMUL_OUTPUT_COLS; j += 4) {
            const size_t column = first_column + x * 4 + j * MATMUL_BLOCK_GROUP_X;
            const size_t dst = (batch * rows + row) * columns + column;
            const ulong bias_index = bias_offset + row * bias_row + column * bias_col;
            if (full_tiles || column + 3 < columns) {
                const float4 product = alpha * (float4)(
                    values[i][j], values[i][j + 1], values[i][j + 2], values[i][j + 3]);
                float4 result = product;
                if (beta != 0.0f) {
                    const float4 bias_values = gemm_load4(float_offset(bias, bias_index, fp16), 0, 0, 1, 4, 0, bias_col, 0.0f, fp16);
                    result = beta * bias_values + product;
                }
                if (fp16) vstore_half4_rte(result, 0, (__global half*)output + dst);
                else vstore4(result, 0, output + dst);
            } else {
                for (uint c = 0; c < 4 && column + c < columns; ++c) {
                    write_float(output, dst + c, gemm_result(values[i][j + c], bias, bias_index + c * bias_col,
                                                             alpha, beta, fp16), fp16);
                }
            }
        }
    }
}

__kernel void matmul_blocked(GEMM_ARGUMENTS) {
    __local float left_tile[MATMUL_TILE_SIZE][MATMUL_BLOCK_SIZE + 1];
    __local float right_tile[MATMUL_TILE_SIZE][MATMUL_BLOCK_SIZE + 1];
    gemm_blocked(GEMM_PARAMETERS, false, false, left_tile, right_tile);
}

__kernel void matmul_blocked_aligned(GEMM_ARGUMENTS) {
    __local float left_tile[MATMUL_TILE_SIZE][MATMUL_BLOCK_SIZE + 1];
    __local float right_tile[MATMUL_TILE_SIZE][MATMUL_BLOCK_SIZE + 1];
    gemm_blocked(GEMM_PARAMETERS, true, false, left_tile, right_tile);
}

__kernel void matmul_blocked_half(GEMM_ARGUMENTS) {
    __local float left_tile[MATMUL_TILE_SIZE][MATMUL_BLOCK_SIZE + 1];
    __local float right_tile[MATMUL_TILE_SIZE][MATMUL_BLOCK_SIZE + 1];
    gemm_blocked(GEMM_PARAMETERS, false, true, left_tile, right_tile);
}

__kernel void matmul_blocked_aligned_half(GEMM_ARGUMENTS) {
    __local float left_tile[MATMUL_TILE_SIZE][MATMUL_BLOCK_SIZE + 1];
    __local float right_tile[MATMUL_TILE_SIZE][MATMUL_BLOCK_SIZE + 1];
    gemm_blocked(GEMM_PARAMETERS, true, true, left_tile, right_tile);
}

)CLC",
R"CLC(
inline float logistic(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float gelu_tanh_inner(float x) {
    return 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
}

inline float max_nan(float x, float y) {
    return isnan(x) ? x : isnan(y) ? y : fmax(x, y);
}

// Match ATen's floor-division rule, including values near integer quotients.
inline float rounded_float_divide(float x, float y) {
    const float quotient = x / y;
    if (y == 0.0f) return quotient;
    const float mod = fmod(x, y);
    float div = (x - mod) / y;
    if (mod != 0.0f && ((y < 0.0f) != (mod < 0.0f))) div -= 1.0f;
    if (div == 0.0f) return copysign(0.0f, quotient);
    const float base = floor(div);
    return base + (div - base > 0.5f);
}

inline float min_nan(float x, float y) {
    return isnan(x) ? x : isnan(y) ? y : fmin(x, y);
}

// Geometry lists the logical size followed by the participating strides for each axis.
__kernel void strided_elementwise(
    __global const ulong* geometry, __global float* output,
    __global const float* left, __global const float* right,
    ulong length, ulong output_offset, ulong left_offset, ulong right_offset,
    ulong rank, float alpha, float beta, uint op, uint fp16
) {
    enum {
        ADD, MULTIPLY, RELU, COPY, FILL, SCALE, THRESHOLD_BACKWARD,
        SQUARED_DIFFERENCE, SCALED_DIFFERENCE, DIVIDE, CEIL, FLOOR, TRUNC, ABS, SQRT,
        RSQRT, EXP, LOG, TANH, SIGMOID, SILU, GELU, GELU_TANH, ADD_SCALAR, SUBTRACT,
        DIVIDE_TENSOR, POW_SCALAR, POW_TENSOR, CLAMP, RECIPROCAL, TANH_BACKWARD,
        SIGMOID_BACKWARD, SILU_BACKWARD, GELU_BACKWARD, GELU_TANH_BACKWARD,
        IS_FINITE, IS_NAN, IS_INF, EQ, NE, LT, LE, GT, GE, BOOL_AND, MAXIMUM, MINIMUM,
        SIN, COS, NEG, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, LOGICAL_XOR, SUB_SCALAR, RSUB_SCALAR,
        ATAN2, ERF, LOG1P, EXPM1, FMOD_TENSOR, FMOD_SCALAR, REM_TENSOR, REM_SCALAR, REM_SCALAR_TENSOR,
        EXP2, LOG2, LOG10, TAN, ASIN, ACOS, ATAN, SINH, COSH, ASINH, ACOSH, ATANH, ERFC, FRAC, ROUND, SIGN, ROUND_DECIMALS, SOFTPLUS, LOG_SIGMOID, LOG_SIGMOID_BUFFER, SOFTPLUS_BACKWARD, LOG_SIGMOID_BACKWARD, DIVIDE_TRUNC, DIVIDE_FLOOR
    };
    for (ulong i = get_global_id(0); i < length; i += get_global_size(0)) {
        ulong logical = i;
        ulong dst = output_offset;
        ulong a = left_offset;
        ulong b = right_offset;
        for (ulong d = rank; d > 0; --d) {
            const __global ulong* axis = geometry + (d - 1) * 4;
            const ulong index = logical % axis[0];
            logical /= axis[0];
            dst += index * axis[1];
            a += index * axis[2];
            b += index * axis[3];
        }
        switch (op) {
            case ADD: write_float(output, dst, read_float(left, a, fp16) + alpha * read_float(right, b, fp16), fp16); break;
            case MULTIPLY: write_float(output, dst, read_float(left, a, fp16) * read_float(right, b, fp16), fp16); break;
            case RELU: {
                const float value = read_float(left, a, fp16);
                write_float(output, dst, value <= 0.0f ? 0.0f : value, fp16);
                break;
            }
            case COPY: write_float(output, dst, read_float(left, a, fp16), fp16); break;
            case FILL: write_float(output, dst, alpha, fp16); break;
            case SCALE: write_float(output, dst, alpha * read_float(left, a, fp16), fp16); break;
            case THRESHOLD_BACKWARD: write_float(output, dst, read_float(right, b, fp16) <= alpha ? 0.0f : read_float(left, a, fp16), fp16); break;
            case SQUARED_DIFFERENCE: {
                const float difference = read_float(left, a, fp16) - read_float(right, b, fp16);
                write_float(output, dst, difference * difference, fp16);
                break;
            }
            case SCALED_DIFFERENCE: write_float(output, dst, alpha * (read_float(left, a, fp16) - read_float(right, b, fp16)), fp16); break;
            case DIVIDE: write_float(output, dst, read_float(left, a, fp16) / alpha, fp16); break;
            case CEIL: write_float(output, dst, ceil(read_float(left, a, fp16)), fp16); break;
            case FLOOR: write_float(output, dst, floor(read_float(left, a, fp16)), fp16); break;
            case TRUNC: write_float(output, dst, trunc(read_float(left, a, fp16)), fp16); break;
            case ABS: write_float(output, dst, fabs(read_float(left, a, fp16)), fp16); break;
            case SQRT: write_float(output, dst, sqrt(read_float(left, a, fp16)), fp16); break;
            case RSQRT: write_float(output, dst, rsqrt(read_float(left, a, fp16)), fp16); break;
            case EXP: write_float(output, dst, exp(read_float(left, a, fp16)), fp16); break;
            case LOG: write_float(output, dst, log(read_float(left, a, fp16)), fp16); break;
            case TANH: write_float(output, dst, tanh(read_float(left, a, fp16)), fp16); break;
            case SIGMOID: write_float(output, dst, logistic(read_float(left, a, fp16)), fp16); break;
            case SILU: write_float(output, dst, read_float(left, a, fp16) * logistic(read_float(left, a, fp16)), fp16); break;
            case GELU: write_float(output, dst, 0.5f * read_float(left, a, fp16) * (1.0f + erf(read_float(left, a, fp16) * 0.7071067811865475f)), fp16); break;
            case GELU_TANH: write_float(output, dst, 0.5f * read_float(left, a, fp16) * (1.0f + tanh(gelu_tanh_inner(read_float(left, a, fp16)))), fp16); break;
            case ADD_SCALAR: write_float(output, dst, read_float(left, a, fp16) + alpha, fp16); break;
            case SUBTRACT: write_float(output, dst, read_float(left, a, fp16) - alpha * read_float(right, b, fp16), fp16); break;
            case DIVIDE_TENSOR: write_float(output, dst, read_float(left, a, fp16) / read_float(right, b, fp16), fp16); break;
            case DIVIDE_TRUNC: write_float(output, dst, trunc(read_float(left, a, fp16) / read_float(right, b, fp16)), fp16); break;
            case DIVIDE_FLOOR: write_float(output, dst, rounded_float_divide(read_float(left, a, fp16), read_float(right, b, fp16)), fp16); break;
            case POW_SCALAR: write_float(output, dst, pow(read_float(left, a, fp16), alpha), fp16); break;
            case POW_TENSOR: write_float(output, dst, pow(read_float(left, a, fp16), read_float(right, b, fp16)), fp16); break;
            case CLAMP: write_float(output, dst, min_nan(max_nan(read_float(left, a, fp16), alpha), beta), fp16); break;
            case RECIPROCAL: write_float(output, dst, 1.0f / read_float(left, a, fp16), fp16); break;
            case TANH_BACKWARD: write_float(output, dst, read_float(left, a, fp16) * (1.0f - read_float(right, b, fp16) * read_float(right, b, fp16)), fp16); break;
            case SIGMOID_BACKWARD: write_float(output, dst, read_float(left, a, fp16) * read_float(right, b, fp16) * (1.0f - read_float(right, b, fp16)), fp16); break;
            case SILU_BACKWARD: {
                const float sigmoid = logistic(read_float(right, b, fp16));
                write_float(output, dst, read_float(left, a, fp16) * sigmoid * (1.0f + read_float(right, b, fp16) * (1.0f - sigmoid)), fp16);
                break;
            }
            case GELU_BACKWARD: {
                const float x = read_float(right, b, fp16);
                write_float(output, dst, read_float(left, a, fp16) * (0.5f * (1.0f + erf(x * 0.7071067811865475f)) +
                    x * 0.3989422804014327f * exp(-0.5f * x * x)), fp16);
                break;
            }
            case GELU_TANH_BACKWARD: {
                const float x = read_float(right, b, fp16);
                const float t = tanh(gelu_tanh_inner(x));
                write_float(output, dst, read_float(left, a, fp16) * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) *
                    0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x)), fp16);
                break;
            }
            case IS_FINITE: ((__global uchar*)output)[dst] = isfinite(read_float(left, a, fp16)) != 0; break;
            case IS_NAN: ((__global uchar*)output)[dst] = isnan(read_float(left, a, fp16)) != 0; break;
            case IS_INF: ((__global uchar*)output)[dst] = isinf(read_float(left, a, fp16)) != 0; break;
            case EQ: ((__global uchar*)output)[dst] = read_float(left, a, fp16) == (beta != 0.0f ? alpha : read_float(right, b, fp16)); break;
            case NE: ((__global uchar*)output)[dst] = read_float(left, a, fp16) != (beta != 0.0f ? alpha : read_float(right, b, fp16)); break;
            case LT: ((__global uchar*)output)[dst] = read_float(left, a, fp16) < (beta != 0.0f ? alpha : read_float(right, b, fp16)); break;
            case LE: ((__global uchar*)output)[dst] = read_float(left, a, fp16) <= (beta != 0.0f ? alpha : read_float(right, b, fp16)); break;
            case GT: ((__global uchar*)output)[dst] = read_float(left, a, fp16) > (beta != 0.0f ? alpha : read_float(right, b, fp16)); break;
            case GE: ((__global uchar*)output)[dst] = read_float(left, a, fp16) >= (beta != 0.0f ? alpha : read_float(right, b, fp16)); break;
            case BOOL_AND:
                ((__global uchar*)output)[dst] = ((__global const uchar*)left)[a] &
                                                ((__global const uchar*)right)[b];
                break;
            case MAXIMUM: write_float(output, dst, max_nan(read_float(left, a, fp16), read_float(right, b, fp16)), fp16); break;
            case MINIMUM: write_float(output, dst, min_nan(read_float(left, a, fp16), read_float(right, b, fp16)), fp16); break;
            case SIN: write_float(output, dst, sin(read_float(left, a, fp16)), fp16); break;
            case COS: write_float(output, dst, cos(read_float(left, a, fp16)), fp16); break;
            case NEG: write_float(output, dst, -read_float(left, a, fp16), fp16); break;
            case LOGICAL_AND:
                ((__global uchar*)output)[dst] = (read_float(left, a, fp16) != 0.0f) && (read_float(right, b, fp16) != 0.0f);
                break;
            case LOGICAL_OR:
                ((__global uchar*)output)[dst] = (read_float(left, a, fp16) != 0.0f) || (read_float(right, b, fp16) != 0.0f);
                break;
            case LOGICAL_NOT:
                ((__global uchar*)output)[dst] = read_float(left, a, fp16) == 0.0f;
                break;
            case LOGICAL_XOR:
                ((__global uchar*)output)[dst] = (read_float(left, a, fp16) != 0.0f) != (read_float(right, b, fp16) != 0.0f);
                break;
            case SUB_SCALAR: write_float(output, dst, read_float(left, a, fp16) - alpha, fp16); break;
            case RSUB_SCALAR: write_float(output, dst, alpha - beta * read_float(left, a, fp16), fp16); break;
            case ATAN2: write_float(output, dst, atan2(read_float(left, a, fp16), read_float(right, b, fp16)), fp16); break;
            case ERF: write_float(output, dst, erf(read_float(left, a, fp16)), fp16); break;
            case LOG1P: write_float(output, dst, log1p(read_float(left, a, fp16)), fp16); break;
            case EXPM1: write_float(output, dst, expm1(read_float(left, a, fp16)), fp16); break;
            case FMOD_TENSOR: write_float(output, dst, fmod(read_float(left, a, fp16), read_float(right, b, fp16)), fp16); break;
            case FMOD_SCALAR: write_float(output, dst, fmod(read_float(left, a, fp16), alpha), fp16); break;
            case REM_TENSOR: {
                const float x = read_float(left, a, fp16), y = read_float(right, b, fp16);
                float r = fmod(x, y);
                if (r != 0.0f && ((r < 0.0f) != (y < 0.0f))) r += y;
                write_float(output, dst, r, fp16);
                break;
            }
            case REM_SCALAR: {
                const float x = read_float(left, a, fp16);
                float r = fmod(x, alpha);
                if (r != 0.0f && ((r < 0.0f) != (alpha < 0.0f))) r += alpha;
                write_float(output, dst, r, fp16);
                break;
            }
            case EXP2: { const float x = read_float(left, a, fp16); write_float(output, dst, exp2(x), fp16); break; }
            case LOG2: { const float x = read_float(left, a, fp16); write_float(output, dst, log2(x), fp16); break; }
            case LOG10: { const float x = read_float(left, a, fp16); write_float(output, dst, log10(x), fp16); break; }
            case TAN: { const float x = read_float(left, a, fp16); write_float(output, dst, tan(x), fp16); break; }
            case ASIN: { const float x = read_float(left, a, fp16); write_float(output, dst, asin(x), fp16); break; }
            case ACOS: { const float x = read_float(left, a, fp16); write_float(output, dst, acos(x), fp16); break; }
            case ATAN: { const float x = read_float(left, a, fp16); write_float(output, dst, atan(x), fp16); break; }
            case SINH: { const float x = read_float(left, a, fp16); write_float(output, dst, sinh(x), fp16); break; }
            case COSH: { const float x = read_float(left, a, fp16); write_float(output, dst, cosh(x), fp16); break; }
            case ASINH: { const float x = read_float(left, a, fp16); write_float(output, dst, asinh(x), fp16); break; }
            case ACOSH: { const float x = read_float(left, a, fp16); write_float(output, dst, acosh(x), fp16); break; }
            case ATANH: { const float x = read_float(left, a, fp16); write_float(output, dst, atanh(x), fp16); break; }
            case ERFC: { const float x = read_float(left, a, fp16); write_float(output, dst, erfc(x), fp16); break; }
            case FRAC: { const float x = read_float(left, a, fp16); write_float(output, dst, x - trunc(x), fp16); break; }
            case ROUND: { const float x = read_float(left, a, fp16); write_float(output, dst, rint(x), fp16); break; }
            case SIGN: { const float x = read_float(left, a, fp16); write_float(output, dst, (float)((x > 0.0f) - (x < 0.0f)), fp16); break; }
            case ROUND_DECIMALS: { const float x = read_float(left, a, fp16); write_float(output, dst, beta > 0.0f ? rint(x * alpha) / alpha : rint(x / alpha) * alpha, fp16); break; }
            case SOFTPLUS: { const float x = read_float(left, a, fp16); write_float(output, dst, x * alpha > beta ? x : (fmax(x * alpha, 0.0f) + log1p(exp(-fabs(x * alpha)))) / alpha, fp16); break; }
            case LOG_SIGMOID: { const float x = read_float(left, a, fp16); write_float(output, dst, fmin(x, 0.0f) - log1p(exp(-fabs(x))), fp16); break; }
            case LOG_SIGMOID_BUFFER: { const float x = read_float(left, a, fp16); write_float(output, dst, exp(-fabs(x)), fp16); break; }
            case SOFTPLUS_BACKWARD: { const float z = alpha * read_float(right, b, fp16); write_float(output, dst, read_float(left, a, fp16) * (z > beta ? 1.0f : logistic(z)), fp16); break; }
            case LOG_SIGMOID_BACKWARD: write_float(output, dst, read_float(left, a, fp16) * logistic(-read_float(right, b, fp16)), fp16); break;
            case REM_SCALAR_TENSOR: {
                const float y = read_float(left, a, fp16);
                float r = fmod(alpha, y);
                if (r != 0.0f && ((r < 0.0f) != (y < 0.0f))) r += y;
                write_float(output, dst, r, fp16);
                break;
            }
        }
    }
}

)CLC",
R"CLC(
enum ValueType { VALUE_FLOAT, VALUE_LONG, VALUE_BOOL, VALUE_BYTE, VALUE_INT, VALUE_HALF,
                 VALUE_CHAR, VALUE_SHORT, VALUE_BFLOAT, VALUE_COMPLEX };

__kernel void fft_radix2_local(__global float2* output, __global const float* real,
                               __global const float* imaginary, ulong real_offset, ulong imag_offset,
                               uint length, float sign) {
    __local float2 values[1024];
    const uint lane = get_local_id(0);
    const uint lanes = get_local_size(0);
    const ulong start = get_group_id(0) * (ulong)length;
    for (uint index = lane; index < length; index += lanes) {
        uint reverse = 0;
        uint bits = index;
        for (uint remaining = length; remaining > 1; remaining >>= 1) {
            reverse = (reverse << 1) | (bits & 1);
            bits >>= 1;
        }
        values[reverse] = (float2)(real[real_offset + start + index], imaginary[imag_offset + start + index]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint span = 1; span < length; span <<= 1) {
        for (uint pair = lane; pair < length / 2; pair += lanes) {
            const uint angle_index = pair & (span - 1);
            const uint a = 2 * pair - angle_index;
            const uint b = a + span;
            float cosine;
            const float sine = sincos(sign * 3.14159265358979323846f * angle_index / span, &cosine);
            const float2 even = values[a];
            const float2 odd = values[b];
            const float2 rotated = (float2)(odd.x * cosine - odd.y * sine, odd.x * sine + odd.y * cosine);
            values[a] = even + rotated;
            values[b] = even - rotated;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint index = lane; index < length; index += lanes) {
        output[start + index] = values[index];
    }
}

inline long rounded_integer_divide(long x, long y, bool floor_mode) {
    if (y == 0) return 0; // The dispatcher checks denominators before submitting this kernel.
    if (x == LONG_MIN && y == -1) return LONG_MIN;
    const long q = x / y;
    return q - (floor_mode && x % y != 0 && ((x < 0) != (y < 0)));
}

inline long read_integer(__global const uchar* input, ulong index, uint type) {
    if (type == VALUE_LONG) return ((__global const long*)input)[index];
    if (type == VALUE_INT) return ((__global const int*)input)[index];
    if (type == VALUE_CHAR) return ((__global const char*)input)[index];
    if (type == VALUE_SHORT) return ((__global const short*)input)[index];
    return input[index];
}

inline void write_integer(__global uchar* output, ulong index, uint type, long value) {
    if (type == VALUE_LONG || type == VALUE_COMPLEX) {
        ((__global long*)output)[index] = value;
    } else if (type == VALUE_INT) {
        ((__global int*)output)[index] = (int)value;
    } else if (type == VALUE_SHORT || type == VALUE_BFLOAT) {
        ((__global ushort*)output)[index] = (ushort)value;
    } else {
        output[index] = type == VALUE_BOOL ? value != 0 : (uchar)value;
    }
}

inline uint value_size(uint type) {
    if (type == VALUE_LONG || type == VALUE_COMPLEX) return 8;
    if (type == VALUE_BOOL || type == VALUE_BYTE || type == VALUE_CHAR) return 1;
    return type == VALUE_HALF || type == VALUE_SHORT || type == VALUE_BFLOAT ? 2 : 4;
}

inline void copy_math_bits(__global uchar* output, ulong dst, __global const uchar* input,
                           ulong src, uint type, uint conjugate, uint negate) {
    if (type == VALUE_COMPLEX) {
        const uint2 bits = ((__global const uint2*)input)[src];
        ((__global uint2*)output)[dst] = bits ^ (uint2)(negate << 31, (conjugate ^ negate) << 31);
    } else if (type == VALUE_FLOAT) {
        ((__global uint*)output)[dst] = ((__global const uint*)input)[src] ^ (negate << 31);
    } else if (type == VALUE_HALF || type == VALUE_BFLOAT) {
        ((__global ushort*)output)[dst] = ((__global const ushort*)input)[src] ^ (negate << 15);
    } else {
        write_integer(output, dst, type, as_long(0UL - (ulong)read_integer(input, src, type)));
    }
}

inline bool is_float_storage(uint type) {
    return type == VALUE_FLOAT || type == VALUE_HALF || type == VALUE_BFLOAT;
}

inline float load_float_storage(__global const uchar* input, ulong index, uint type) {
    if (type == VALUE_BFLOAT) return as_float((uint)((__global const ushort*)input)[index] << 16);
    return read_float((__global const float*)input, index, type == VALUE_HALF);
}

inline void store_float_storage(__global uchar* output, ulong index, uint type, float value) {
    if (type == VALUE_BFLOAT) {
        const uint bits = as_uint(value);
        ((__global ushort*)output)[index] = isnan(value) ? (ushort)0x7fc0
            : (ushort)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
    } else {
        write_float((__global float*)output, index, value, type == VALUE_HALF);
    }
}

inline bool storage_nonzero(__global const uchar* data, ulong index, uint type) {
    if (is_float_storage(type)) return load_float_storage(data, index, type) != 0.0f;
    if (type == VALUE_COMPLEX) return any(((__global const float2*)data)[index] != (float2)(0.0f));
    return read_integer(data, index, type) != 0;
}

inline void storage_add(__global uchar* output, ulong dst, __global const uchar* input, ulong src, uint type) {
    if (is_float_storage(type)) {
        store_float_storage(output, dst, type, load_float_storage(output, dst, type) + load_float_storage(input, src, type));
    } else if (type == VALUE_COMPLEX) {
        ((__global float2*)output)[dst] += ((__global const float2*)input)[src];
    } else {
        const ulong left = (ulong)read_integer(output, dst, type);
        const ulong right = (ulong)read_integer(input, src, type);
        write_integer(output, dst, type, as_long(left + right));
    }
}

__kernel void integral_matmul(
    __global uchar* out, __global const uchar* left, __global const uchar* right,
    ulong rows, ulong cols, ulong inner, ulong a, ulong ab, ulong ar, ulong ac,
    ulong b, ulong bb, ulong br, ulong bc, uint type
) {
    const ulong index = get_global_id(0);
    const ulong batch = index / (rows * cols);
    a += batch * ab + (index / cols % rows) * ar;
    b += batch * bb + (index % cols) * bc;
    ulong result = 0;
    for (ulong k = 0; k < inner; ++k)
        result += (ulong)read_integer(left, a + k * ac, type) * (ulong)read_integer(right, b + k * br, type);
    write_integer(out, index, type, as_long(result));
}

__kernel void int8_matmul_strided(
    __global int* out, __global const char* left, __global const char* right,
    ulong output_offset, ulong columns, ulong inner, ulong a, ulong ar, ulong ac, ulong b, ulong br, ulong bc
) {
    const ulong row = get_global_id(1);
    const ulong col = get_global_id(0);
    a += row * ar;
    b += col * bc;
    uint sum = 0;
    for (ulong k = 0; k < inner; ++k) {
        sum += (uint)((int)left[a + k * ac] * (int)right[b + k * br]);
    }
    out[output_offset + row * columns + col] = as_int(sum);
}

__kernel void integer_binary(__global const ulong* geometry, __global uchar* output,
                             __global const uchar* left, __global const uchar* right,
                             ulong dst, ulong a, ulong b, ulong rank, uint input_type,
                             uint output_type, uint op, long alpha) {
    ulong logical = get_global_id(0);
    for (ulong d = rank; d > 0; --d) {
        const __global ulong* entry = geometry + (d - 1) * 4;
        const ulong coord = logical % entry[0];
        logical /= entry[0];
        dst += coord * entry[1];
        a += coord * entry[2];
        b += coord * entry[3];
    }
    const long x = read_integer(left, a, input_type);
    const long y = read_integer(right, b, input_type);
    long value = 0;
    switch (op) {
        case 0: value = output_type == VALUE_BOOL ? (x || (alpha && y))
            : as_long((ulong)x + (ulong)alpha * (ulong)y); break;
        case 1: value = as_long((ulong)x - (ulong)alpha * (ulong)y); break;
        case 2: value = as_long((ulong)x * (ulong)y); break;
        case 3: value = x == y; break;
        case 4: value = x != y; break;
        case 5: value = x < y; break;
        case 6: value = x <= y; break;
        case 7: value = x > y; break;
        case 8: value = x >= y; break;
        case 9: value = output_type == VALUE_BOOL ? (x && y) : (x & y); break;
        case 10: value = output_type == VALUE_BOOL ? (x || y) : (x | y); break;
        case 11: value = output_type == VALUE_BOOL ? ((x != 0) != (y != 0)) : (x ^ y); break;
        case 12: value = output_type == VALUE_BOOL ? (x == 0) : (~x); break;
        case 13: value = as_long(0UL - (ulong)x); break;
        case 14: value = (y != 0) ? (x % y) : 0; break;
        case 15: {
            long r = (y != 0) ? (x % y) : 0;
            if (r != 0 && ((r < 0) != (y < 0))) r += y;
            value = r;
            break;
        }
        case 16: value = (x > 0) - (x < 0); break;
        case 17: value = x < 0 ? as_long(0UL - (ulong)x) : x; break;
        case 18: value = max(x, y); break;
        case 19: value = min(x, y); break;
        case 20: case 21: value = rounded_integer_divide(x, y, op == 21); break;
    }
    write_integer(output, dst, output_type, value);
}

__kernel void typed_copy(__global const ulong* geometry, __global uchar* output,
                         __global const uchar* input, ulong length, ulong output_offset,
                         ulong input_offset, ulong rank, uint output_type, uint input_type,
                         ulong integer_fill, uint fill, uint conjugate, uint negate) {
    for (ulong i = get_global_id(0); i < length; i += get_global_size(0)) {
        ulong logical = i;
        ulong dst = output_offset;
        ulong src = input_offset;
        for (ulong d = rank; d > 0; --d) {
            const __global ulong* axis = geometry + (d - 1) * 4;
            const ulong index = logical % axis[0];
            logical /= axis[0];
            dst += index * axis[1];
            src += index * axis[2];
        }
        if (fill) {
            write_integer(output, dst, output_type, as_long(integer_fill));
        } else if (conjugate || negate) {
            copy_math_bits(output, dst, input, src, input_type, conjugate, negate);
        } else if (output_type == input_type) {
            const uint bytes = value_size(input_type);
            for (uint j = 0; j < bytes; ++j) {
                output[dst * bytes + j] = input[src * bytes + j];
            }
        } else if (output_type == VALUE_COMPLEX) {
            const float value = is_float_storage(input_type)
                ? load_float_storage(input, src, input_type)
                : convert_float(read_integer(input, src, input_type));
            ((__global float2*)output)[dst] = (float2)(value, 0.0f);
        } else if (is_float_storage(output_type) || is_float_storage(input_type) || input_type == VALUE_COMPLEX) {
            const float2 complex_value = input_type == VALUE_COMPLEX
                ? ((__global const float2*)input)[src] : (float2)(0.0f);
            const float value = input_type == VALUE_COMPLEX ? complex_value.x
                : is_float_storage(input_type) ? load_float_storage(input, src, input_type)
                : convert_float(read_integer(input, src, input_type));
            if (is_float_storage(output_type)) {
                store_float_storage(output, dst, output_type, value);
                continue;
            }
            if (output_type == VALUE_BOOL) {
                output[dst] = value != 0.0f || complex_value.y != 0.0f;
            } else if (output_type == VALUE_INT) {
                ((__global int*)output)[dst] = value >= -0x1.0p31f && value < 0x1.0p31f
                    ? convert_int_rtz(value) : (-2147483647 - 1);
            } else {
                const long integer = value >= -0x1.0p63f && value < 0x1.0p63f
                    ? convert_long_rtz(value) : (-9223372036854775807L - 1);
                write_integer(output, dst, output_type, integer);
            }
        } else {
            write_integer(output, dst, output_type, read_integer(input, src, input_type));
        }
    }
}

inline ulong reduction_offset(
    ulong index, __global const ulong* geometry, ulong begin, ulong end
) {
    ulong offset = 0;
    for (ulong d = end; d > begin; --d) {
        const ulong size = geometry[(d - 1) * 2];
        offset += (index % size) * geometry[(d - 1) * 2 + 1];
        index /= size;
    }
    return offset;
}

__kernel void masked_copy(__global const ulong* geometry, __global uchar* output,
                          __global long* count, __global const uchar* input, __global const uchar* mask,
                          ulong length, ulong input_offset, ulong mask_offset, ulong rank, uint type) {
    // One work-group per tile would require an exclusive scan to preserve stable ordering.
    ulong written = 0;
    const uint bytes = value_size(type);
    for (ulong i = 0; i < length; ++i) {
        ulong logical = i;
        ulong src = input_offset;
        ulong condition = mask_offset;
        for (ulong d = rank; d > 0; --d) {
            const __global ulong* axis = geometry + (d - 1) * 4;
            const ulong index = logical % axis[0];
            logical /= axis[0];
            src += index * axis[2];
            condition += index * axis[3];
        }
        if (mask[condition]) {
            for (uint j = 0; j < bytes; ++j) {
                output[written * bytes + j] = input[src * bytes + j];
            }
            ++written;
        }
    }
    count[0] = (long)written;
}

)CLC",
R"CLC(
__kernel void strided_extrema(__global const ulong* geometry, __global uchar* output,
                              __global const uchar* input, ulong length, ulong offset, ulong rank,
                              uint type, uint op) {
    __local float partial[64];
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    __local long integers[64];
    if (op <= 1 && !is_float_storage(type)) {
        long value = op == 0 ? LONG_MAX : LONG_MIN;
        for (ulong i = lane; i < length; i += group) {
            const long x = read_integer(input, offset + reduction_offset(i, geometry, 0, rank), type);
            value = op == 0 ? min(value, x) : max(value, x);
        }
        integers[lane] = value;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint width = group / 2; width; width /= 2) {
            if (lane < width) {
                integers[lane] = op == 0 ? min(integers[lane], integers[lane + width])
                                         : max(integers[lane], integers[lane + width]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane == 0) write_integer(output, 0, type, integers[0]);
        return;
    }
    float accumulator = op == 0 ? INFINITY : op == 1 ? -INFINITY : (float)(op == 3);
    for (ulong i = lane; i < length; i += group) {
        const ulong index = offset + reduction_offset(i, geometry, 0, rank);
        if (op <= 1) {
            const float value = load_float_storage(input, index, type);
            accumulator = op == 0 ? min_nan(accumulator, value) : max_nan(accumulator, value);
        } else {
            const bool value = storage_nonzero(input, index, type);
            accumulator = op == 3 ? (accumulator && value) : (accumulator || value);
        }
    }
    partial[lane] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint width = group / 2; width > 0; width /= 2) {
        if (lane < width) {
            partial[lane] = op == 0 || op == 3
                ? min_nan(partial[lane], partial[lane + width])
                : max_nan(partial[lane], partial[lane + width]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        if (op <= 1) store_float_storage(output, 0, type, partial[0]);
        else output[0] = partial[0] != 0.0f;
    }
}


inline uint4 philox(ulong seed, ulong counter) {
    uint4 value = (uint4)((uint)counter, (uint)(counter >> 32), 0, 0);
    uint2 key = (uint2)((uint)seed, (uint)(seed >> 32));
    #pragma unroll
    for (uint round = 0; round < 10; ++round) {
        const uint lo0 = value.x * 0xD2511F53U;
        const uint hi0 = mul_hi(value.x, 0xD2511F53U);
        const uint lo1 = value.z * 0xCD9E8D57U;
        const uint hi1 = mul_hi(value.z, 0xCD9E8D57U);
        value = (uint4)(hi1 ^ value.y ^ key.x, lo1, hi0 ^ value.w ^ key.y, lo0);
        key += (uint2)(0x9E3779B9U, 0xBB67AE85U);
    }
    return value;
}

__kernel void random_fill(__global const ulong* geometry, __global float* output,
                          ulong length, ulong offset, ulong rank, ulong seed,
                          ulong start, float first, float second, uint kind, uint fp16) {
    const ulong block = get_global_id(0);
    const uint4 bits = philox(seed, start + block);
    const float4 unit = convert_float4(bits >> 8) * 0x1.0p-24f;
    float4 samples;
    if (kind == 0) {
        samples = first + unit * (second - first);
        // Rounding the affine transform can reach the exclusive upper bound.
        if (first != second) {
            samples = select(samples, (float4)(nextafter(second, first)), samples >= second);
        }
    } else if (kind == 1) {
        const float2 radius = sqrt(-2.0f * log(1.0f - unit.xz));
        const float2 angle = 6.283185307179586f * unit.yw;
        const float2 cosine = radius * cos(angle);
        const float2 sine = radius * sin(angle);
        samples = first + second * (float4)(cosine.x, sine.x, cosine.y, sine.y);
    } else {
        samples = convert_float4(-(unit < first));
    }
    for (uint lane = 0; lane < 4 && block * 4 + lane < length; ++lane) {
        const ulong index = block * 4 + lane;
        write_float(output, offset + reduction_offset(index, geometry, 0, rank), samples[lane], fp16);
    }
}
__kernel void strided_sum(
    __global const ulong* geometry, __global float* output, __global const float* input,
    ulong reduce_length, ulong offset, ulong outer_rank, ulong rank,
    uint input_half, uint output_half, uint average
) {
    __local float partial[64];
    const uint lane = get_local_id(0);
    const uint group_size = get_local_size(0);
    const size_t out = get_group_id(0);
    const ulong start = offset + reduction_offset(out, geometry, 0, outer_rank);
    float value = 0.0f;
    for (ulong i = lane; i < reduce_length; i += group_size) {
        value += read_float(input, start + reduction_offset(i, geometry, outer_rank, rank), input_half);
    }
    partial[lane] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint width = group_size / 2; width > 0; width /= 2) {
        if (lane < width) {
            partial[lane] += partial[lane + width];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        const float result = average ? partial[0] / convert_float(reduce_length) : partial[0];
        write_float(output, out, result, output_half);
    }
}

__kernel void strided_softmax(
    __global const ulong* geometry, __global float* output, __global const float* input,
    ulong width, ulong inner, ulong offset, ulong dim_stride, ulong outer_rank, uint logarithmic, uint fp16
) {
    __local float partial[64];
    const uint lane = get_local_id(0);
    const uint group_size = get_local_size(0);
    const size_t line = get_group_id(0);
    const ulong source = offset + reduction_offset(line, geometry, 0, outer_rank);
    const ulong target = line / inner * width * inner + line % inner;
    float maximum = -INFINITY;
    for (ulong i = lane; i < width; i += group_size) {
        maximum = fmax(maximum, read_float(input, source + i * dim_stride, fp16));
    }
    partial[lane] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial[lane] = fmax(partial[lane], partial[lane + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = partial[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float denominator = 0.0f;
    for (ulong i = lane; i < width; i += group_size) {
        denominator += exp(read_float(input, source + i * dim_stride, fp16) - maximum);
    }
    partial[lane] = denominator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (ulong i = lane; i < width; i += group_size) {
        const float shifted = read_float(input, source + i * dim_stride, fp16) - maximum;
        write_float(output, target + i * inner,
                    logarithmic ? shifted - log(partial[0]) : exp(shifted) / partial[0], fp16);
    }
}
)CLC",
R"CLC(

inline void copy_index_value(__global uchar* output, ulong dst, __global const uchar* input,
                             ulong src, uint bytes) {
    for (uint i = 0; i < bytes; ++i) {
        output[dst * bytes + i] = input[src * bytes + i];
    }
}

__kernel void tensor_where(__global const ulong* geometry, __global uchar* output,
                           __global const uchar* condition, __global const uchar* left,
                           __global const uchar* right, ulong dst, ulong mask, ulong a, ulong b,
                           ulong rank, uint type) {
    ulong logical = get_global_id(0);
    for (ulong d = rank; d > 0; --d) {
        const __global ulong* entry = geometry + (d - 1) * 5;
        const ulong coord = logical % entry[0];
        logical /= entry[0];
        dst += coord * entry[1];
        mask += coord * entry[2];
        a += coord * entry[3];
        b += coord * entry[4];
    }
    const bool choose_left = condition[mask] != 0;
    copy_index_value(output, dst, choose_left ? left : right, choose_left ? a : b, value_size(type));
}

// Geometry gives each dimension's logical size and output, input, and index strides.
__kernel void indexed_copy(__global const ulong* geometry, __global uchar* output,
                           __global const uchar* input, __global const uchar* indices,
                           volatile __global int* invalid, ulong dst_offset, ulong src_offset,
                           ulong index_offset, ulong rank, ulong axis, ulong axis_size,
                           uint data_type, uint index_type, uint operation, uint include_self) {
    enum { SELECT, GATHER, SCATTER, SCATTER_ADD, SCATTER_SUM, SCATTER_PROD,
           SCATTER_MEAN, SCATTER_MAX, SCATTER_MIN };
    const bool scatter = operation >= SCATTER;
    ulong logical = get_global_id(0);
    ulong dst = dst_offset;
    ulong src = src_offset;
    ulong index = index_offset;
    const uint bytes = value_size(data_type);
    for (ulong d = rank; d > 0; --d) {
        const __global ulong* entry = geometry + (d - 1) * 4;
        if (scatter && d - 1 == axis) continue;
        const ulong coordinate = logical % entry[0];
        logical /= entry[0];
        if (d - 1 != axis || scatter) src += coordinate * entry[2];
        dst += coordinate * entry[1];
        index += coordinate * entry[3];
    }
    const __global ulong* indexed_axis = geometry + axis * 4;
    if (operation >= SCATTER_SUM) {
        const bool fp = is_float_storage(data_type);
        for (ulong j = 0; j < indexed_axis[0]; ++j) {
            const long target = read_integer(indices, index + j * indexed_axis[3], index_type);
            if (target < 0 || (ulong)target >= axis_size) {
                atomic_or(invalid, 1);
                return;
            }
        }
        for (ulong target = 0; target < axis_size; ++target) {
            const ulong at = dst + target * indexed_axis[1];
            ulong count = include_self;
            float fl = fp && include_self ? load_float_storage(output, at, data_type) : 0.0f;
            long integer = !fp && include_self ? read_integer(output, at, data_type) : 0;
            for (ulong j = 0; j < indexed_axis[0]; ++j) {
                if ((ulong)read_integer(indices, index + j * indexed_axis[3], index_type) != target) continue;
                const ulong from = src + j * indexed_axis[2];
                if (fp) {
                    const float value = load_float_storage(input, from, data_type);
                    if (!count) fl = value;
                    else if (operation == SCATTER_PROD) fl *= value;
                    else if (operation == SCATTER_MAX) fl = max_nan(fl, value);
                    else if (operation == SCATTER_MIN) fl = min_nan(fl, value);
                    else fl += value;
                } else {
                    const long value = read_integer(input, from, data_type);
                    if (!count) integer = value;
                    else if (operation == SCATTER_PROD) integer = as_long((ulong)integer * (ulong)value);
                    else if (operation == SCATTER_MAX) integer = max(integer, value);
                    else if (operation == SCATTER_MIN) integer = min(integer, value);
                    else integer = as_long((ulong)integer + (ulong)value);
                }
                ++count;
            }
            if (count == include_self) continue;
            if (fp) {
                if (operation == SCATTER_MEAN) fl /= convert_float(count);
                store_float_storage(output, at, data_type, fl);
            } else {
                if (operation == SCATTER_MEAN) {
                    if (data_type == VALUE_INT) integer = (int)integer;
                    if (data_type == VALUE_BYTE) integer = (uchar)integer;
                    if (data_type == VALUE_CHAR) integer = (char)integer;
                    if (data_type == VALUE_SHORT) integer = (short)integer;
                    integer = rounded_integer_divide(integer, (long)count, true);
                }
                write_integer(output, at, data_type, integer);
            }
        }
        return;
    }
    if (scatter) {
        for (ulong j = 0; j < indexed_axis[0]; ++j) {
            const long target = read_integer(indices, index + j * indexed_axis[3], index_type);
            if (target < 0 || (ulong)target >= axis_size) {
                atomic_or(invalid, 1);
                continue;
            }
            const ulong at = dst + (ulong)target * indexed_axis[1];
            const ulong from = src + j * indexed_axis[2];
            if (operation == SCATTER_ADD) {
                storage_add(output, at, input, from, data_type);
            } else {
                copy_index_value(output, at, input, from, bytes);
            }
        }
    } else {
        const long target = read_integer(indices, index, index_type);
        if (target < 0 || (ulong)target >= axis_size) {
            atomic_or(invalid, 1);
            return;
        }
        copy_index_value(output, dst, input, src + (ulong)target * indexed_axis[2], bytes);
    }
}

__kernel void embedding_gradient(__global float* output, __global const float* grad,
                                  __global const uchar* indices, ulong index_offset, ulong index_stride,
                                  ulong grad_offset, ulong grad_row, ulong grad_column, ulong count,
                                  ulong width, long padding, uint index_type, uint scale) {
    const size_t row = get_global_id(0) / width;
    const size_t column = get_global_id(0) % width;
    float value = 0.0f;
    ulong frequency = 0;
    if ((long)row != padding) {
        for (ulong i = 0; i < count; ++i) {
            const long index = read_integer(indices, index_offset + i * index_stride, index_type);
            if (index == (long)row) {
                value += grad[grad_offset + i * grad_row + column * grad_column];
                ++frequency;
            }
        }
    }
    output[row * width + column] = scale && frequency ? value / convert_float(frequency) : value;
}

__kernel void nll_loss(__global float* output, __global float* weights, volatile __global int* invalid,
                       __global const float* input, __global const long* target,
                       __global const float* weight, __global const float* grad,
                       __global const float* total_weight, ulong input_offset, ulong input_row,
                       ulong input_column, ulong target_offset, ulong target_stride, ulong weight_offset,
                       ulong weight_stride, ulong grad_offset, ulong grad_stride, ulong total_offset, ulong classes,
                       long ignore_index, uint weighted, uint backward, uint mean) {
    const ulong row = get_global_id(0);
    const long label = target[target_offset + row * target_stride];
    if (label == ignore_index) {
        if (!backward) {
            output[row] = 0.0f;
            weights[row] = 0.0f;
        }
        return;
    }
    if (label < 0 || (ulong)label >= classes) {
        atomic_or(invalid, 1);
        return;
    }
    const float scale = weighted ? weight[weight_offset + (ulong)label * weight_stride] : 1.0f;
    if (backward) {
        const float value = -grad[grad_offset + row * grad_stride];
        output[row * classes + (ulong)label] = scale * (mean ? value / total_weight[total_offset] : value);
    } else {
        output[row] = -scale * input[input_offset + row * input_row + (ulong)label * input_column];
        weights[row] = scale;
    }
}

__kernel void nonzero_coordinates(__global const ulong* geometry, __global long* coordinates,
                                   __global long* count, __global const uchar* input,
                                   ulong length, ulong offset, ulong rank, uint type) {
    ulong selected = 0;
    for (ulong i = 0; i < length; ++i) {
        const ulong src = offset + reduction_offset(i, geometry, 0, rank);
        const bool valid = storage_nonzero(input, src, type);
        if (!valid) continue;
        ulong logical = i;
        for (ulong d = rank; d > 0; --d) {
            const ulong size = geometry[(d - 1) * 2];
            coordinates[selected * rank + d - 1] = (long)(logical % size);
            logical /= size;
        }
        ++selected;
    }
    count[0] = (long)selected;
}

__kernel void advanced_offsets(__global const ulong* geometry, __global ulong* offsets,
                                __global const uchar* index, volatile __global int* invalid,
                                ulong index_offset, ulong rank, ulong axis_size, ulong axis_stride, uint type) {
    const ulong i = get_global_id(0);
    const ulong index_pos = index_offset + reduction_offset(i, geometry, 0, rank);
    long selected = read_integer(index, index_pos, type);
    if (selected < 0) selected += (long)axis_size;
    if (selected < 0 || (ulong)selected >= axis_size) {
        atomic_or(invalid, 1);
        return;
    }
    offsets[i] += (ulong)selected * axis_stride;
}

__kernel void advanced_copy(__global const ulong* geometry, __global uchar* base,
                             __global uchar* linear, __global const ulong* offsets, ulong length,
                             ulong base_offset, ulong rank, ulong trailing, ulong index_length,
                             uint type, uint put, uint accumulate) {
    const uint bytes = value_size(type);
    for (ulong i = get_global_id(0); i < length; i += get_global_size(0)) {
        const ulong offset = base_offset + reduction_offset(i, geometry, 0, rank) +
            offsets[i / trailing % index_length];
        if (!put) {
            copy_index_value(linear, i, base, offset, bytes);
        } else if (!accumulate) {
            copy_index_value(base, offset, linear, i, bytes);
        } else {
            storage_add(base, offset, linear, i, type);
        }
    }
}

__kernel void integer_sum(__global const ulong* geometry, __global uchar* output,
                          __global const uchar* input, ulong length, ulong offset,
                          ulong outer_rank, ulong rank, uint input_type, uint output_type,
                          uint contiguous, ulong chunks) {
    __local ulong partial[64];
    const uint lane = get_local_id(0);
    const uint width = get_local_size(0);
    const ulong group = get_group_id(0);
    const ulong row = group / chunks;
    const ulong start = offset + (contiguous ? row * length : reduction_offset(row, geometry, 0, outer_rank));
    ulong sum = 0;
    for (ulong j = group % chunks * width + lane; j < length; j += chunks * width) {
        const ulong at = start + (contiguous ? j : reduction_offset(j, geometry, outer_rank, rank));
        sum += (ulong)read_integer(input, at, input_type);
    }
    partial[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = width / 2; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) write_integer(output, group, output_type, as_long(partial[0]));
}

__kernel void logical_reduce(__global const ulong* geometry, __global uchar* output, __global const uchar* input,
                             ulong length, ulong offset, ulong outer_rank, ulong rank, uint type, uint all) {
    const ulong i = get_global_id(0);
    const ulong start = offset + reduction_offset(i, geometry, 0, outer_rank);
    for (ulong j = 0; j < length; ++j) {
        const ulong at = start + reduction_offset(j, geometry, outer_rank, rank);
        const bool value = storage_nonzero(input, at, type);
        if (value != (all != 0)) {
            output[i] = !all;
            return;
        }
    }
    output[i] = all != 0;
}

// Number of coordinates in the first n rows of the growing section.
inline ulong tril_prefix(ulong n, ulong first_width) {
    return n ? n * first_width + n * (n - 1) / 2 : 0;
}

__kernel void tril_indices(__global uchar* output, ulong offset, ulong row_stride,
                           ulong column_stride, ulong first_row, ulong first_width,
                           ulong growing_rows, ulong growing_count, ulong columns, int int32) {
    const ulong index = get_global_id(0);
    ulong row, column;
    if (index < growing_count) {
        // Find the last row whose prefix does not exceed the output index.
        ulong lo = 0, hi = growing_rows;
        while (hi - lo > 1) {
            const ulong mid = lo + (hi - lo) / 2;
            if (tril_prefix(mid, first_width) <= index) lo = mid;
            else hi = mid;
        }
        row = first_row + lo;
        column = index - tril_prefix(lo, first_width);
    } else {
        const ulong remaining = index - growing_count;
        row = first_row + growing_rows + remaining / columns;
        column = remaining % columns;
    }
    const ulong at = offset + index * column_stride;
    if (int32) {
        ((__global int*)output)[at] = as_int((uint)row);
        ((__global int*)output)[at + row_stride] = as_int((uint)column);
    } else {
        ((__global long*)output)[at] = (long)row;
        ((__global long*)output)[at + row_stride] = (long)column;
    }
}

__kernel void triangular(__global const ulong* geometry, __global uchar* output, __global const uchar* input,
                          ulong offset, ulong rows, ulong columns, long diagonal, ulong rank, uint type, uint upper) {
    const ulong i = get_global_id(0);
    const long delta = (long)(i % columns) - (long)(i / columns % rows);
    const uint bytes = value_size(type);
    if (upper ? delta >= diagonal : delta <= diagonal) {
        copy_index_value(output, i, input, offset + reduction_offset(i, geometry, 0, rank), bytes);
    } else {
        for (uint j = 0; j < bytes; ++j) output[i * bytes + j] = 0;
    }
}
)CLC",
R"CLC(
inline void swizzle_gemm_groups(uint gx, uint gy, uint num_gx, uint num_gy, uint panel_w,
                                uint* out_gx, uint* out_gy) {
    const uint linear = gy * num_gx + gx;
    const uint full_panels = num_gx / panel_w;
    const uint full_panel_tiles = full_panels * panel_w * num_gy;
    if (linear < full_panel_tiles) {
        const uint panel_id = linear / (panel_w * num_gy);
        const uint in_panel = linear % (panel_w * num_gy);
        *out_gy = in_panel / panel_w;
        *out_gx = panel_id * panel_w + (in_panel % panel_w);
    } else {
        const uint rem = linear - full_panel_tiles;
        const uint rem_w = num_gx - full_panels * panel_w;
        *out_gy = rem / rem_w;
        *out_gx = full_panels * panel_w + (rem % rem_w);
    }
}

#ifdef IRIS_HAS_REQD_SG8
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_blocked_sg8_aligned(GEMM_ARGUMENTS) {
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    uint gx, gy;
    swizzle_gemm_groups(get_group_id(0), get_group_id(1), get_num_groups(0), get_num_groups(1), 4, &gx, &gy);
    const size_t first_row = (size_t)gy * 128;
    const size_t first_col = (size_t)gx * 64;
    const size_t batch = get_group_id(2);
    left += left_offset + batch * left_batch;
    right += right_offset + batch * right_batch;

    __local float a_tile[16][129];
    __local float b_tile[16][65];

    const bool lt = (left_row == 1 && left_col != 1);
    const bool rt = (right_row == 1 && right_col != 1);
    const uint tid = ly * 8 + lx;

    for (ulong k_chunk = 0; k_chunk < inner; k_chunk += 512) {
        const ulong k_end = min(inner, k_chunk + 512);
        float c[8][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 8; ++j) c[i][j] = 0.0f;

        for (ulong k0 = k_chunk; k0 < k_end; k0 += 16) {
            if (lt) {
                const uint ac4 = (tid & 31) * 4;
                const uint ar0 = tid >> 5;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint kr = ar0 + p * 4;
                    vstore4(vload4(0, left + (k0 + kr) * left_col + first_row + ac4), 0, &a_tile[kr][ac4]);
                }
            } else {
                const uint ak4 = (tid & 3) * 4;
                const uint am0 = tid >> 2;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint mr = am0 + p * 32;
                    const float4 v = vload4(0, left + (first_row + mr) * left_row + k0 + ak4);
                    a_tile[ak4][mr] = v.s0;
                    a_tile[ak4 + 1][mr] = v.s1;
                    a_tile[ak4 + 2][mr] = v.s2;
                    a_tile[ak4 + 3][mr] = v.s3;
                }
            }
            if (rt) {
                const uint bk4 = (tid & 3) * 4;
                const uint bn0 = tid >> 2;
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const uint nr = bn0 + p * 32;
                    const float4 v = vload4(0, right + (first_col + nr) * right_col + k0 + bk4);
                    b_tile[bk4][nr] = v.s0;
                    b_tile[bk4 + 1][nr] = v.s1;
                    b_tile[bk4 + 2][nr] = v.s2;
                    b_tile[bk4 + 3][nr] = v.s3;
                }
            } else {
                const uint bc4 = (tid & 15) * 4;
                const uint br0 = tid >> 4;
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const uint kr = br0 + p * 8;
                    vstore4(vload4(0, right + (k0 + kr) * right_row + first_col + bc4), 0, &b_tile[kr][bc4]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                const float4 bv0 = vload4(0, &b_tile[k][lx * 4]);
                const float4 bv1 = vload4(0, &b_tile[k][lx * 4 + 32]);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float av = a_tile[k][ly * 8 + i];
                    c[i][0] = fma(av, bv0.s0, c[i][0]);
                    c[i][1] = fma(av, bv0.s1, c[i][1]);
                    c[i][2] = fma(av, bv0.s2, c[i][2]);
                    c[i][3] = fma(av, bv0.s3, c[i][3]);
                    c[i][4] = fma(av, bv1.s0, c[i][4]);
                    c[i][5] = fma(av, bv1.s1, c[i][5]);
                    c[i][6] = fma(av, bv1.s2, c[i][6]);
                    c[i][7] = fma(av, bv1.s3, c[i][7]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const bool first = (k_chunk == 0);
        const bool last = (k_end == inner);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const size_t row = first_row + ly * 8 + i;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const size_t col = first_col + lx * 4 + h * 32;
                const size_t dst = (batch * rows + row) * columns + col;
                float4 val = (float4)(c[i][h * 4], c[i][h * 4 + 1], c[i][h * 4 + 2], c[i][h * 4 + 3]);
                if (!first) val += vload4(0, output + dst);
                if (last) {
                    float4 prod = alpha * val;
                    if (beta != 0.0f) {
                        const ulong bi = bias_offset + row * bias_row + col * bias_col;
                        const float4 bv = gemm_load4(bias + bi, 0, 0, 1, 4, 0, bias_col, 0.0f, false);
                        prod = beta * bv + prod;
                    }
                    vstore4(prod, 0, output + dst);
                } else {
                    vstore4(val, 0, output + dst);
                }
            }
        }
    }
}
#endif

__kernel void sum_all_stage1(__global float* partial, __global const float* input,
                             ulong length, ulong offset) {
    __local float scratch[128];
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const ulong gid = get_global_id(0);
    const ulong stride = get_global_size(0);
    float acc = 0.0f;
    for (ulong i = gid; i < length; i += stride) {
        acc += input[offset + i];
    }
    scratch[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint w = group / 2; w > 0; w /= 2) {
        if (lane < w) scratch[lane] += scratch[lane + w];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        partial[get_group_id(0)] = scratch[0];
    }
}

__kernel void sum_all_stage2(__global float* output, __global const float* partial,
                             ulong num_groups, ulong total_length, uint average) {
    __local float scratch[128];
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    float acc = 0.0f;
    for (ulong i = lane; i < num_groups; i += group) {
        acc += partial[i];
    }
    scratch[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint w = group / 2; w > 0; w /= 2) {
        if (lane < w) scratch[lane] += scratch[lane + w];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        output[0] = average ? scratch[0] / convert_float(total_length) : scratch[0];
    }
}

__kernel void sum_2d_dim0(__global float* output, __global const float* input,
                          ulong offset, int rows, int cols, uint average) {
    const int col = get_global_id(0);
    if (col >= cols) return;
    float acc = 0.0f;
    ulong idx = offset + (ulong)col;
    for (int r = 0; r < rows; ++r) {
        acc += input[idx];
        idx += (ulong)cols;
    }
    output[col] = average ? acc / convert_float(rows) : acc;
}

__kernel void sum_4d_nchw_c(__global float* output, __global const float* input,
                            ulong offset, int n_batch, int channels, int hw, uint average) {
    __local float scratch[256];
    const int c = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const int total = n_batch * hw;
    float acc = 0.0f;
    for (int idx = (int)lane; idx < total; idx += (int)group) {
        const int n = idx / hw;
        const int p = idx - n * hw;
        acc += input[offset + ((ulong)n * channels + c) * hw + p];
    }
    scratch[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint w = group / 2; w > 0; w /= 2) {
        if (lane < w) scratch[lane] += scratch[lane + w];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        output[c] = average ? scratch[0] / convert_float(total) : scratch[0];
    }
}
)CLC",
R"CLC(
__kernel void cast_f32_to_f16(__global half* output, __global const float* input, ulong length) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        vstore_half4_rte(vload4(0, input + base), 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            vstore_half_rte(input[i], i, output);
        }
    }
}

__kernel void cast_f16_to_f32(__global float* output, __global const half* input, ulong length) {
    const ulong v = get_global_id(0);
    const ulong base = v * 4;
    if (base + 3 < length) {
        vstore4(vload_half4(0, input + base), 0, output + base);
    } else if (base < length) {
        for (ulong i = base; i < length; ++i) {
            output[i] = vload_half(i, input);
        }
    }
}

#if defined(IRIS_HAS_FP16) && defined(IRIS_HAS_REQD_SG8)
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_blocked_sg8_aligned_half(GEMM_ARGUMENTS) {
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    uint gx, gy;
    swizzle_gemm_groups(get_group_id(0), get_group_id(1), get_num_groups(0), get_num_groups(1), 4, &gx, &gy);
    const size_t first_row = (size_t)gy * 128;
    const size_t first_col = (size_t)gx * 64;
    const size_t batch = get_group_id(2);
    __global const half* left_h = ((__global const half*)left) + left_offset + batch * left_batch;
    __global const half* right_h = ((__global const half*)right) + right_offset + batch * right_batch;
    __global half* out_h = (__global half*)output;

    __local float a_tile[16][129];
    __local float b_tile[16][65];

    const bool lt = (left_row == 1 && left_col != 1);
    const bool rt = (right_row == 1 && right_col != 1);
    const uint tid = ly * 8 + lx;

    float c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) c[i][j] = 0.0f;

    for (ulong k0 = 0; k0 < inner; k0 += 16) {
        if (lt) {
            const uint ac4 = (tid & 31) * 4;
            const uint ar0 = tid >> 5;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint kr = ar0 + p * 4;
                vstore4(vload_half4(0, left_h + (k0 + kr) * left_col + first_row + ac4), 0, &a_tile[kr][ac4]);
            }
        } else {
            const uint ak4 = (tid & 3) * 4;
            const uint am0 = tid >> 2;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint mr = am0 + p * 32;
                const float4 v = vload_half4(0, left_h + (first_row + mr) * left_row + k0 + ak4);
                a_tile[ak4][mr] = v.s0;
                a_tile[ak4 + 1][mr] = v.s1;
                a_tile[ak4 + 2][mr] = v.s2;
                a_tile[ak4 + 3][mr] = v.s3;
            }
        }
        if (rt) {
            const uint bk4 = (tid & 3) * 4;
            const uint bn0 = tid >> 2;
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const uint nr = bn0 + p * 32;
                const float4 v = vload_half4(0, right_h + (first_col + nr) * right_col + k0 + bk4);
                b_tile[bk4][nr] = v.s0;
                b_tile[bk4 + 1][nr] = v.s1;
                b_tile[bk4 + 2][nr] = v.s2;
                b_tile[bk4 + 3][nr] = v.s3;
            }
        } else {
            const uint bc4 = (tid & 15) * 4;
            const uint br0 = tid >> 4;
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const uint kr = br0 + p * 8;
                vstore4(vload_half4(0, right_h + (k0 + kr) * right_row + first_col + bc4), 0, &b_tile[kr][bc4]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float4 bv0 = vload4(0, &b_tile[k][lx * 4]);
            const float4 bv1 = vload4(0, &b_tile[k][lx * 4 + 32]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float av = a_tile[k][ly * 8 + i];
                c[i][0] = fma(av, bv0.s0, c[i][0]);
                c[i][1] = fma(av, bv0.s1, c[i][1]);
                c[i][2] = fma(av, bv0.s2, c[i][2]);
                c[i][3] = fma(av, bv0.s3, c[i][3]);
                c[i][4] = fma(av, bv1.s0, c[i][4]);
                c[i][5] = fma(av, bv1.s1, c[i][5]);
                c[i][6] = fma(av, bv1.s2, c[i][6]);
                c[i][7] = fma(av, bv1.s3, c[i][7]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const size_t row = first_row + ly * 8 + i;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const size_t col = first_col + lx * 4 + h * 32;
            const size_t dst = (batch * rows + row) * columns + col;
            float4 prod = alpha * (float4)(c[i][h * 4], c[i][h * 4 + 1], c[i][h * 4 + 2], c[i][h * 4 + 3]);
            if (beta != 0.0f) {
                const ulong bi = bias_offset + row * bias_row + col * bias_col;
                const float4 bv = gemm_load4(float_offset(bias, bi, true), 0, 0, 1, 4, 0, bias_col, 0.0f, true);
                prod = beta * bv + prod;
            }
            vstore_half4_rte(prod, 0, out_h + dst);
        }
    }
}
#endif

// Small-batch / skinny-M GEMM: 1 WG of 128 threads reduces K per output element (when M*N <= 256)
// or 1 subgroup of 16 threads per output element (when M*N > 256).
__kernel void matmul_skinny_wg(GEMM_ARGUMENTS, uint fp16) {
    __local float scratch[128];
    const uint lane = get_local_id(0);
    const size_t col = get_group_id(0);
    const size_t row = get_group_id(1);
    const size_t batch = get_group_id(2);
    const ulong a_base = left_offset + batch * left_batch + row * left_row;
    const ulong b_base = right_offset + batch * right_batch + col * right_col;
    float acc = 0.0f;
    if (!fp16 && left_col == 1 && right_row == 1) {
        const ulong vec_k = inner >> 2;
        for (ulong kv = lane; kv < vec_k; kv += 128) {
            const float4 av = vload4(0, left + a_base + (kv << 2));
            const float4 bv = vload4(0, right + b_base + (kv << 2));
            acc = fma(av.s0, bv.s0, acc);
            acc = fma(av.s1, bv.s1, acc);
            acc = fma(av.s2, bv.s2, acc);
            acc = fma(av.s3, bv.s3, acc);
        }
        if (lane == 0) {
            for (ulong k = vec_k << 2; k < inner; ++k) {
                acc = fma(left[a_base + k], right[b_base + k], acc);
            }
        }
    } else if (fp16 && left_col == 1 && right_row == 1) {
        __global const half* lh = (__global const half*)left + a_base;
        __global const half* rh = (__global const half*)right + b_base;
        const ulong vec_k = inner >> 2;
        for (ulong kv = lane; kv < vec_k; kv += 128) {
            const float4 av = vload_half4(0, lh + (kv << 2));
            const float4 bv = vload_half4(0, rh + (kv << 2));
            acc = fma(av.s0, bv.s0, acc);
            acc = fma(av.s1, bv.s1, acc);
            acc = fma(av.s2, bv.s2, acc);
            acc = fma(av.s3, bv.s3, acc);
        }
        if (lane == 0) {
            for (ulong k = vec_k << 2; k < inner; ++k) {
                acc = fma(vload_half(k, lh), vload_half(k, rh), acc);
            }
        }
    } else {
        for (ulong k = lane; k < inner; k += 128) {
            acc = fma(read_float(left, a_base + k * left_col, fp16),
                      read_float(right, b_base + k * right_row, fp16), acc);
        }
    }
    scratch[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint w = 64; w > 0; w >>= 1) {
        if (lane < w) scratch[lane] += scratch[lane + w];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        write_float(output, (batch * rows + row) * columns + col,
                    gemm_result(scratch[0], bias, bias_offset + row * bias_row + col * bias_col,
                                alpha, beta, fp16), fp16);
    }
}

#if defined(IRIS_HAS_SUBGROUPS) && defined(IRIS_HAS_REQD_SG8)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul_skinny_sg16(GEMM_ARGUMENTS, uint fp16) {
    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const size_t col = get_group_id(0) * 8 + sg;
    const size_t row = get_group_id(1);
    const size_t batch = get_group_id(2);
    if (col >= columns) return;
    const ulong a_base = left_offset + batch * left_batch + row * left_row;
    const ulong b_base = right_offset + batch * right_batch + col * right_col;
    float acc = 0.0f;
    if (!fp16 && left_col == 1 && right_row == 1) {
        const ulong vec_k = inner >> 2;
        for (ulong kv = lane; kv < vec_k; kv += 16) {
            const float4 av = vload4(0, left + a_base + (kv << 2));
            const float4 bv = vload4(0, right + b_base + (kv << 2));
            acc = fma(av.s0, bv.s0, acc);
            acc = fma(av.s1, bv.s1, acc);
            acc = fma(av.s2, bv.s2, acc);
            acc = fma(av.s3, bv.s3, acc);
        }
        if (lane == 0) {
            for (ulong k = vec_k << 2; k < inner; ++k) {
                acc = fma(left[a_base + k], right[b_base + k], acc);
            }
        }
    } else if (fp16 && left_col == 1 && right_row == 1) {
        __global const half* lh = (__global const half*)left + a_base;
        __global const half* rh = (__global const half*)right + b_base;
        const ulong vec_k = inner >> 2;
        for (ulong kv = lane; kv < vec_k; kv += 16) {
            const float4 av = vload_half4(0, lh + (kv << 2));
            const float4 bv = vload_half4(0, rh + (kv << 2));
            acc = fma(av.s0, bv.s0, acc);
            acc = fma(av.s1, bv.s1, acc);
            acc = fma(av.s2, bv.s2, acc);
            acc = fma(av.s3, bv.s3, acc);
        }
        if (lane == 0) {
            for (ulong k = vec_k << 2; k < inner; ++k) {
                acc = fma(vload_half(k, lh), vload_half(k, rh), acc);
            }
        }
    } else {
        for (ulong k = lane; k < inner; k += 16) {
            acc = fma(read_float(left, a_base + k * left_col, fp16),
                      read_float(right, b_base + k * right_row, fp16), acc);
        }
    }
    const float sum = sub_group_reduce_add(acc);
    if (lane == 0) {
        write_float(output, (batch * rows + row) * columns + col,
                    gemm_result(sum, bias, bias_offset + row * bias_row + col * bias_col,
                                alpha, beta, fp16), fp16);
    }
}
#endif
)CLC",
R"CLC(
__kernel void softmax_lastdim_fp32(__global float* output, __global const float* input,
                                   ulong input_offset, ulong width, uint logarithmic, uint fp16) {
    __local float scratch[64];
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const size_t row = get_group_id(0);
    const ulong base = input_offset + (ulong)row * width;
    const ulong out_base = (ulong)row * width;
    float mx = -INFINITY;
    for (ulong i = lane; i < width; i += group) {
        mx = fmax(mx, read_float(input, base + i, fp16 != 0));
    }
    scratch[lane] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = group >> 1; s > 0; s >>= 1) {
        if (lane < s) scratch[lane] = fmax(scratch[lane], scratch[lane + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    mx = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float denom = 0.0f;
    for (ulong i = lane; i < width; i += group) {
        denom += exp(read_float(input, base + i, fp16 != 0) - mx);
    }
    scratch[lane] = denom;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = group >> 1; s > 0; s >>= 1) {
        if (lane < s) scratch[lane] += scratch[lane + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    denom = scratch[0];
    const float log_denom = logarithmic ? log(denom) : 0.0f;
    for (ulong i = lane; i < width; i += group) {
        const float shifted = read_float(input, base + i, fp16 != 0) - mx;
        write_float(output, out_base + i,
                    logarithmic ? (shifted - log_denom) : (exp(shifted) / denom), fp16 != 0);
    }
}

__kernel void softmax_lastdim_vec4(__global float* output, __global const float* input,
                                   ulong input_offset, ulong width, uint logarithmic) {
    __local float scratch[64];
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const ulong row = get_group_id(0) * width;
    const __global float* source = input + input_offset + row;
    __global float* target = output + row;
    const ulong vectors = width / 4;
    float4 maximum = (float4)(-INFINITY);
    for (ulong i = lane; i < vectors; i += group) maximum = fmax(maximum, vload4(i, source));
    scratch[lane] = fmax(fmax(maximum.s0, maximum.s1), fmax(maximum.s2, maximum.s3));
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = group / 2; step; step /= 2) {
        if (lane < step) scratch[lane] = fmax(scratch[lane], scratch[lane + step]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float4 mx = (float4)(scratch[0]);
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 sums = (float4)(0.0f);
    for (ulong i = lane; i < vectors; i += group) sums += exp(vload4(i, source) - mx);
    scratch[lane] = (sums.s0 + sums.s1) + (sums.s2 + sums.s3);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = group / 2; step; step /= 2) {
        if (lane < step) scratch[lane] += scratch[lane + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float4 normalizer = (float4)(logarithmic ? log(scratch[0]) : 1.0f / scratch[0]);
    for (ulong i = lane; i < vectors; i += group) {
        const float4 shifted = vload4(i, source) - mx;
        vstore4(logarithmic ? shifted - normalizer : exp(shifted) * normalizer, i, target);
    }
}

__kernel void softmax_backward_lastdim(__global float* grad_in, __global const float* grad_out,
                                       __global const float* output,
                                       ulong gi_off, ulong go_off, ulong o_off,
                                       ulong width, uint logarithmic, uint fp16) {
    __local float scratch[64];
    const uint lane = get_local_id(0);
    const uint group = get_local_size(0);
    const size_t row = get_group_id(0);
    const ulong r_off = (ulong)row * width;
    float acc = 0.0f;
    for (ulong i = lane; i < width; i += group) {
        const float g = read_float(grad_out, go_off + r_off + i, fp16 != 0);
        if (logarithmic) {
            acc += g;
        } else {
            acc = fma(g, read_float(output, o_off + r_off + i, fp16 != 0), acc);
        }
    }
    scratch[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = group >> 1; s > 0; s >>= 1) {
        if (lane < s) scratch[lane] += scratch[lane + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float total = scratch[0];
    for (ulong i = lane; i < width; i += group) {
        const float g = read_float(grad_out, go_off + r_off + i, fp16 != 0);
        const float y = read_float(output, o_off + r_off + i, fp16 != 0);
        const float res = logarithmic ? (g - exp(y) * total) : ((g - total) * y);
        write_float(grad_in, gi_off + r_off + i, res, fp16 != 0);
    }
}

__kernel void nll_loss_forward_fused(__global float* loss_out, __global float* total_weight_out,
                                     volatile __global int* invalid,
                                     __global const float* input, __global const long* target,
                                     __global const float* weight,
                                     ulong input_offset, ulong input_row, ulong input_column,
                                     ulong target_offset, ulong target_stride,
                                     ulong weight_offset, ulong weight_stride,
                                     ulong batch, ulong classes, long ignore_index,
                                     uint weighted, uint mean) {
    __local float s_loss[64];
    __local float s_weight[64];
    const uint lane = get_local_id(0);
    float acc_loss = 0.0f;
    float acc_weight = 0.0f;
    for (ulong row = lane; row < batch; row += 64) {
        const long label = target[target_offset + row * target_stride];
        if (label == ignore_index) continue;
        if (label < 0 || (ulong)label >= classes) {
            atomic_or(invalid, 1);
            continue;
        }
        const float scale = weighted ? weight[weight_offset + (ulong)label * weight_stride] : 1.0f;
        acc_loss += -scale * input[input_offset + row * input_row + (ulong)label * input_column];
        acc_weight += scale;
    }
    s_loss[lane] = acc_loss;
    s_weight[lane] = acc_weight;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint w = 32; w > 0; w >>= 1) {
        if (lane < w) {
            s_loss[lane] += s_loss[lane + w];
            s_weight[lane] += s_weight[lane + w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        total_weight_out[0] = s_weight[0];
        loss_out[0] = mean ? (s_loss[0] / s_weight[0]) : s_loss[0];
    }
}

__kernel void nll_loss_backward_fused(__global float* output, volatile __global int* invalid,
                                      __global const long* target, __global const float* weight,
                                      __global const float* grad, __global const float* total_weight,
                                      ulong target_offset, ulong target_stride,
                                      ulong weight_offset, ulong weight_stride,
                                      ulong grad_offset, ulong grad_stride, ulong total_offset,
                                      ulong batch, ulong classes, long ignore_index,
                                      uint weighted, uint mean) {
    const ulong idx = get_global_id(0);
    const ulong total = batch * classes;
    if (idx >= total) return;
    const ulong row = idx / classes;
    const ulong col = idx - row * classes;
    const long label = target[target_offset + row * target_stride];
    if (label == ignore_index) {
        output[idx] = 0.0f;
        return;
    }
    if (label < 0 || (ulong)label >= classes) {
        if (col == 0) atomic_or(invalid, 1);
        output[idx] = 0.0f;
        return;
    }
    if (col == (ulong)label) {
        const float scale = weighted ? weight[weight_offset + (ulong)label * weight_stride] : 1.0f;
        const float value = -grad[grad_offset + row * grad_stride];
        output[idx] = scale * (mean ? value / total_weight[total_offset] : value);
    } else {
        output[idx] = 0.0f;
    }
}
)CLC",
R"CLC(
// Zero-allocation inline geometry (up to 6D = 24 ulongs) & vectorized contiguous elementwise + multi-row GEMV
typedef struct {
    ulong v[24];
} Geom24;

__kernel void add_scalar_batch(Geom24 offsets, float scalar,
    __global float* a, __global float* b, __global float* c, __global float* d,
    __global float* e, __global float* f, __global float* g, __global float* h) {
    const uint i = get_global_id(0);
    __global float* target;
    switch (i) {
        case 0: target = a; break;
        case 1: target = b; break;
        case 2: target = c; break;
        case 3: target = d; break;
        case 4: target = e; break;
        case 5: target = f; break;
        case 6: target = g; break;
        default: target = h; break;
    }
    target[offsets.v[i]] += scalar;
}

__kernel void amp_unscale(__global float* grad, __global uint* found_inf,
                          __global const float* inv_scale, ulong grad_offset, ulong found_offset,
                          ulong inv_offset, ulong length, uint fp16) {
    const ulong first = get_global_id(0) * 4;
    const float scale = inv_scale[inv_offset];
    bool finite = true;
    for (ulong i = first; i < min(length, first + 4); ++i) {
        const float value = read_float(grad, grad_offset + i, fp16);
        finite = finite && isfinite(value);
        if (scale != 1.0f) write_float(grad, grad_offset + i, value * scale, fp16);
    }
    if (!finite) atomic_xchg(found_inf + found_offset, as_uint(1.0f));
}

__kernel void amp_update_scale(__global float* scale, __global int* growth, __global const float* found_inf,
                               ulong scale_offset, ulong growth_offset, ulong found_offset,
                               float growth_factor, float backoff_factor, int interval) {
    if (found_inf[found_offset] != 0.0f) {
        scale[scale_offset] *= backoff_factor;
        growth[growth_offset] = 0;
    } else {
        const int next = growth[growth_offset] + 1;
        if (next == interval) {
            const float grown = scale[scale_offset] * growth_factor;
            if (isfinite(grown)) scale[scale_offset] = grown;
            growth[growth_offset] = 0;
        } else {
            growth[growth_offset] = next;
        }
    }
}

)CLC",
R"CLC(
inline void eval_ew_item(__global float* output, __global const float* left, __global const float* right,
                         ulong dst, ulong a, ulong b, float alpha, float beta, uint op, uint fp16) {
    enum {
        ADD, MULTIPLY, RELU, COPY, FILL, SCALE, THRESHOLD_BACKWARD,
        SQUARED_DIFFERENCE, SCALED_DIFFERENCE, DIVIDE, CEIL, FLOOR, TRUNC, ABS, SQRT,
        RSQRT, EXP, LOG, TANH, SIGMOID, SILU, GELU, GELU_TANH, ADD_SCALAR, SUBTRACT,
        DIVIDE_TENSOR, POW_SCALAR, POW_TENSOR, CLAMP, RECIPROCAL, TANH_BACKWARD,
        SIGMOID_BACKWARD, SILU_BACKWARD, GELU_BACKWARD, GELU_TANH_BACKWARD,
        IS_FINITE, IS_NAN, IS_INF, EQ, NE, LT, LE, GT, GE, BOOL_AND, MAXIMUM, MINIMUM,
        SIN, COS, NEG, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, LOGICAL_XOR, SUB_SCALAR, RSUB_SCALAR,
        ATAN2, ERF, LOG1P, EXPM1, FMOD_TENSOR, FMOD_SCALAR, REM_TENSOR, REM_SCALAR, REM_SCALAR_TENSOR,
        EXP2, LOG2, LOG10, TAN, ASIN, ACOS, ATAN, SINH, COSH, ASINH, ACOSH, ATANH, ERFC, FRAC, ROUND, SIGN, ROUND_DECIMALS, SOFTPLUS, LOG_SIGMOID, LOG_SIGMOID_BUFFER, SOFTPLUS_BACKWARD, LOG_SIGMOID_BACKWARD, DIVIDE_TRUNC, DIVIDE_FLOOR
    };
    const bool h = fp16 != 0;
    switch (op) {
        case ADD: write_float(output, dst, read_float(left, a, h) + alpha * read_float(right, b, h), h); break;
        case MULTIPLY: write_float(output, dst, read_float(left, a, h) * read_float(right, b, h), h); break;
        case RELU: { const float v = read_float(left, a, h); write_float(output, dst, v <= 0.0f ? 0.0f : v, h); break; }
        case COPY: write_float(output, dst, read_float(left, a, h), h); break;
        case FILL: write_float(output, dst, alpha, h); break;
        case SCALE: write_float(output, dst, alpha * read_float(left, a, h), h); break;
        case THRESHOLD_BACKWARD: write_float(output, dst, read_float(right, b, h) <= alpha ? 0.0f : read_float(left, a, h), h); break;
        case SQUARED_DIFFERENCE: { const float d = read_float(left, a, h) - read_float(right, b, h); write_float(output, dst, d * d, h); break; }
        case SCALED_DIFFERENCE: write_float(output, dst, alpha * (read_float(left, a, h) - read_float(right, b, h)), h); break;
        case DIVIDE: write_float(output, dst, read_float(left, a, h) / alpha, h); break;
        case CEIL: write_float(output, dst, ceil(read_float(left, a, h)), h); break;
        case FLOOR: write_float(output, dst, floor(read_float(left, a, h)), h); break;
        case TRUNC: write_float(output, dst, trunc(read_float(left, a, h)), h); break;
        case ABS: write_float(output, dst, fabs(read_float(left, a, h)), h); break;
        case SQRT: write_float(output, dst, sqrt(read_float(left, a, h)), h); break;
        case RSQRT: write_float(output, dst, rsqrt(read_float(left, a, h)), h); break;
        case EXP: write_float(output, dst, exp(read_float(left, a, h)), h); break;
        case LOG: write_float(output, dst, log(read_float(left, a, h)), h); break;
        case TANH: write_float(output, dst, tanh(read_float(left, a, h)), h); break;
        case SIGMOID: write_float(output, dst, logistic(read_float(left, a, h)), h); break;
        case SILU: { const float x = read_float(left, a, h); write_float(output, dst, x * logistic(x), h); break; }
        case GELU: { const float x = read_float(left, a, h); write_float(output, dst, 0.5f * x * (1.0f + erf(x * 0.7071067811865475f)), h); break; }
        case GELU_TANH: { const float x = read_float(left, a, h); write_float(output, dst, 0.5f * x * (1.0f + tanh(gelu_tanh_inner(x))), h); break; }
        case ADD_SCALAR: write_float(output, dst, read_float(left, a, h) + alpha, h); break;
        case SUBTRACT: write_float(output, dst, read_float(left, a, h) - alpha * read_float(right, b, h), h); break;
        case DIVIDE_TENSOR: write_float(output, dst, read_float(left, a, h) / read_float(right, b, h), h); break;
        case DIVIDE_TRUNC: write_float(output, dst, trunc(read_float(left, a, h) / read_float(right, b, h)), h); break;
        case DIVIDE_FLOOR: write_float(output, dst, rounded_float_divide(read_float(left, a, h), read_float(right, b, h)), h); break;
        case POW_SCALAR: write_float(output, dst, pow(read_float(left, a, h), alpha), h); break;
        case POW_TENSOR: write_float(output, dst, pow(read_float(left, a, h), read_float(right, b, h)), h); break;
        case CLAMP: write_float(output, dst, min_nan(max_nan(read_float(left, a, h), alpha), beta), h); break;
        case RECIPROCAL: write_float(output, dst, 1.0f / read_float(left, a, h), h); break;
        case TANH_BACKWARD: { const float y = read_float(right, b, h); write_float(output, dst, read_float(left, a, h) * (1.0f - y * y), h); break; }
        case SIGMOID_BACKWARD: { const float y = read_float(right, b, h); write_float(output, dst, read_float(left, a, h) * y * (1.0f - y), h); break; }
        case SILU_BACKWARD: { const float x = read_float(right, b, h); const float s = logistic(x); write_float(output, dst, read_float(left, a, h) * s * (1.0f + x * (1.0f - s)), h); break; }
        case GELU_BACKWARD: { const float x = read_float(right, b, h); write_float(output, dst, read_float(left, a, h) * (0.5f * (1.0f + erf(x * 0.7071067811865475f)) + x * 0.3989422804014327f * exp(-0.5f * x * x)), h); break; }
        case GELU_TANH_BACKWARD: { const float x = read_float(right, b, h); const float t = tanh(gelu_tanh_inner(x)); write_float(output, dst, read_float(left, a, h) * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x)), h); break; }
        case IS_FINITE: ((__global uchar*)output)[dst] = isfinite(read_float(left, a, h)) != 0; break;
        case IS_NAN: ((__global uchar*)output)[dst] = isnan(read_float(left, a, h)) != 0; break;
        case IS_INF: ((__global uchar*)output)[dst] = isinf(read_float(left, a, h)) != 0; break;
        case EQ: ((__global uchar*)output)[dst] = read_float(left, a, h) == (beta != 0.0f ? alpha : read_float(right, b, h)); break;
        case NE: ((__global uchar*)output)[dst] = read_float(left, a, h) != (beta != 0.0f ? alpha : read_float(right, b, h)); break;
        case LT: ((__global uchar*)output)[dst] = read_float(left, a, h) < (beta != 0.0f ? alpha : read_float(right, b, h)); break;
        case LE: ((__global uchar*)output)[dst] = read_float(left, a, h) <= (beta != 0.0f ? alpha : read_float(right, b, h)); break;
        case GT: ((__global uchar*)output)[dst] = read_float(left, a, h) > (beta != 0.0f ? alpha : read_float(right, b, h)); break;
        case GE: ((__global uchar*)output)[dst] = read_float(left, a, h) >= (beta != 0.0f ? alpha : read_float(right, b, h)); break;
        case BOOL_AND: ((__global uchar*)output)[dst] = ((__global const uchar*)left)[a] & ((__global const uchar*)right)[b]; break;
        case MAXIMUM: write_float(output, dst, max_nan(read_float(left, a, h), read_float(right, b, h)), h); break;
        case MINIMUM: write_float(output, dst, min_nan(read_float(left, a, h), read_float(right, b, h)), h); break;
        case SIN: write_float(output, dst, sin(read_float(left, a, h)), h); break;
        case COS: write_float(output, dst, cos(read_float(left, a, h)), h); break;
        case NEG: write_float(output, dst, -read_float(left, a, h), h); break;
        case LOGICAL_AND: ((__global uchar*)output)[dst] = (read_float(left, a, h) != 0.0f) && (read_float(right, b, h) != 0.0f); break;
        case LOGICAL_OR: ((__global uchar*)output)[dst] = (read_float(left, a, h) != 0.0f) || (read_float(right, b, h) != 0.0f); break;
        case LOGICAL_NOT: ((__global uchar*)output)[dst] = read_float(left, a, h) == 0.0f; break;
        case LOGICAL_XOR: ((__global uchar*)output)[dst] = (read_float(left, a, h) != 0.0f) != (read_float(right, b, h) != 0.0f); break;
        case SUB_SCALAR: write_float(output, dst, read_float(left, a, h) - alpha, h); break;
        case RSUB_SCALAR: write_float(output, dst, alpha - beta * read_float(left, a, h), h); break;
        case ATAN2: write_float(output, dst, atan2(read_float(left, a, h), read_float(right, b, h)), h); break;
        case ERF: write_float(output, dst, erf(read_float(left, a, h)), h); break;
        case LOG1P: write_float(output, dst, log1p(read_float(left, a, h)), h); break;
        case EXPM1: write_float(output, dst, expm1(read_float(left, a, h)), h); break;
        case FMOD_TENSOR: write_float(output, dst, fmod(read_float(left, a, h), read_float(right, b, h)), h); break;
        case FMOD_SCALAR: write_float(output, dst, fmod(read_float(left, a, h), alpha), h); break;
        case REM_TENSOR: {
            const float x = read_float(left, a, h), y = read_float(right, b, h);
            float r = fmod(x, y);
            if (r != 0.0f && ((r < 0.0f) != (y < 0.0f))) r += y;
            write_float(output, dst, r, h);
            break;
        }
        case REM_SCALAR: {
            const float x = read_float(left, a, h);
            float r = fmod(x, alpha);
            if (r != 0.0f && ((r < 0.0f) != (alpha < 0.0f))) r += alpha;
            write_float(output, dst, r, h);
            break;
        }
        case EXP2: { const float x = read_float(left, a, h); write_float(output, dst, exp2(x), h); break; }
        case LOG2: { const float x = read_float(left, a, h); write_float(output, dst, log2(x), h); break; }
        case LOG10: { const float x = read_float(left, a, h); write_float(output, dst, log10(x), h); break; }
        case TAN: { const float x = read_float(left, a, h); write_float(output, dst, tan(x), h); break; }
        case ASIN: { const float x = read_float(left, a, h); write_float(output, dst, asin(x), h); break; }
        case ACOS: { const float x = read_float(left, a, h); write_float(output, dst, acos(x), h); break; }
        case ATAN: { const float x = read_float(left, a, h); write_float(output, dst, atan(x), h); break; }
        case SINH: { const float x = read_float(left, a, h); write_float(output, dst, sinh(x), h); break; }
        case COSH: { const float x = read_float(left, a, h); write_float(output, dst, cosh(x), h); break; }
        case ASINH: { const float x = read_float(left, a, h); write_float(output, dst, asinh(x), h); break; }
        case ACOSH: { const float x = read_float(left, a, h); write_float(output, dst, acosh(x), h); break; }
        case ATANH: { const float x = read_float(left, a, h); write_float(output, dst, atanh(x), h); break; }
        case ERFC: { const float x = read_float(left, a, h); write_float(output, dst, erfc(x), h); break; }
        case FRAC: { const float x = read_float(left, a, h); write_float(output, dst, x - trunc(x), h); break; }
        case ROUND: { const float x = read_float(left, a, h); write_float(output, dst, rint(x), h); break; }
        case SIGN: { const float x = read_float(left, a, h); write_float(output, dst, (float)((x > 0.0f) - (x < 0.0f)), h); break; }
        case ROUND_DECIMALS: { const float x = read_float(left, a, h); write_float(output, dst, beta > 0.0f ? rint(x * alpha) / alpha : rint(x / alpha) * alpha, h); break; }
        case SOFTPLUS: { const float x = read_float(left, a, h); write_float(output, dst, x * alpha > beta ? x : (fmax(x * alpha, 0.0f) + log1p(exp(-fabs(x * alpha)))) / alpha, h); break; }
        case LOG_SIGMOID: { const float x = read_float(left, a, h); write_float(output, dst, fmin(x, 0.0f) - log1p(exp(-fabs(x))), h); break; }
        case LOG_SIGMOID_BUFFER: { const float x = read_float(left, a, h); write_float(output, dst, exp(-fabs(x)), h); break; }
        case SOFTPLUS_BACKWARD: { const float z = alpha * read_float(right, b, h); write_float(output, dst, read_float(left, a, h) * (z > beta ? 1.0f : logistic(z)), h); break; }
        case LOG_SIGMOID_BACKWARD: write_float(output, dst, read_float(left, a, h) * logistic(-read_float(right, b, h)), h); break;
        case REM_SCALAR_TENSOR: {
            const float y = read_float(left, a, h);
            float r = fmod(alpha, y);
            if (r != 0.0f && ((r < 0.0f) != (y < 0.0f))) r += y;
            write_float(output, dst, r, h);
            break;
        }
    }
}

__kernel void contiguous_elementwise_kernel(
    __global float* output, __global const float* left, __global const float* right,
    ulong length, ulong out_off, ulong left_off, ulong right_off,
    float alpha, float beta, uint op, uint fp16
) {
    const ulong base = (ulong)get_global_id(0) * 4UL;
    if (base >= length) return;
    const ulong end = min(base + 4UL, length);
    for (ulong i = base; i < end; ++i) {
        eval_ew_item(output, left, right, out_off + i, left_off + i, right_off + i, alpha, beta, op, fp16);
    }
}

__kernel void strided_elementwise_inline(
    Geom24 geom, __global float* output, __global const float* left, __global const float* right,
    ulong length, ulong output_offset, ulong left_offset, ulong right_offset,
    ulong rank, float alpha, float beta, uint op, uint fp16
) {
    for (ulong i = get_global_id(0); i < length; i += get_global_size(0)) {
        ulong logical = i;
        ulong dst = output_offset;
        ulong a = left_offset;
        ulong b = right_offset;
        for (ulong d = rank; d > 0; --d) {
            const ulong base_idx = (d - 1) * 4;
            const ulong sz = geom.v[base_idx];
            const ulong index = logical % sz;
            logical /= sz;
            dst += index * geom.v[base_idx + 1];
            a += index * geom.v[base_idx + 2];
            b += index * geom.v[base_idx + 3];
        }
        eval_ew_item(output, left, right, dst, a, b, alpha, beta, op, fp16);
    }
}

)CLC",
R"CLC(
__kernel void typed_copy_inline(
    Geom24 geom, __global uchar* output, __global const uchar* input,
    ulong length, ulong output_offset, ulong input_offset, ulong rank,
    uint output_type, uint input_type, ulong integer_fill, uint fill, uint conjugate, uint negate
) {
    for (ulong i = get_global_id(0); i < length; i += get_global_size(0)) {
        ulong logical = i;
        ulong dst = output_offset;
        ulong src = input_offset;
        for (ulong d = rank; d > 0; --d) {
            const ulong bi = (d - 1) * 4;
            const ulong sz = geom.v[bi];
            const ulong index = logical % sz;
            logical /= sz;
            dst += index * geom.v[bi + 1];
            src += index * geom.v[bi + 2];
        }
        if (fill) {
            write_integer(output, dst, output_type, as_long(integer_fill));
        } else if (conjugate || negate) {
            copy_math_bits(output, dst, input, src, input_type, conjugate, negate);
        } else if (output_type == input_type) {
            const uint bytes = value_size(input_type);
            for (uint j = 0; j < bytes; ++j) output[dst * bytes + j] = input[src * bytes + j];
        } else if (output_type == VALUE_COMPLEX) {
            const float value = is_float_storage(input_type)
                ? load_float_storage(input, src, input_type)
                : convert_float(read_integer(input, src, input_type));
            ((__global float2*)output)[dst] = (float2)(value, 0.0f);
        } else if (is_float_storage(output_type) || is_float_storage(input_type) || input_type == VALUE_COMPLEX) {
            const float2 complex_value = input_type == VALUE_COMPLEX
                ? ((__global const float2*)input)[src] : (float2)(0.0f);
            const float value = input_type == VALUE_COMPLEX ? complex_value.x
                : is_float_storage(input_type) ? load_float_storage(input, src, input_type)
                : convert_float(read_integer(input, src, input_type));
            if (is_float_storage(output_type)) {
                store_float_storage(output, dst, output_type, value);
                continue;
            }
            if (output_type == VALUE_BOOL) {
                output[dst] = value != 0.0f || complex_value.y != 0.0f;
            } else if (output_type == VALUE_INT) {
                ((__global int*)output)[dst] = value >= -0x1.0p31f && value < 0x1.0p31f
                    ? convert_int_rtz(value) : (-2147483647 - 1);
            } else {
                const long integer = value >= -0x1.0p63f && value < 0x1.0p63f
                    ? convert_long_rtz(value) : (-9223372036854775807L - 1);
                write_integer(output, dst, output_type, integer);
            }
        } else {
            write_integer(output, dst, output_type, read_integer(input, src, input_type));
        }
    }
}

__kernel void integer_binary_inline(
    Geom24 geom, __global uchar* output, __global const uchar* left, __global const uchar* right,
    ulong dst, ulong a, ulong b, ulong rank, uint input_type, uint output_type, uint op, long alpha
) {
    ulong logical = get_global_id(0);
    for (ulong d = rank; d > 0; --d) {
        const ulong bi = (d - 1) * 4;
        const ulong sz = geom.v[bi];
        const ulong coord = logical % sz;
        logical /= sz;
        dst += coord * geom.v[bi + 1];
        a += coord * geom.v[bi + 2];
        b += coord * geom.v[bi + 3];
    }
    const long x = read_integer(left, a, input_type);
    const long y = read_integer(right, b, input_type);
    long value = 0;
    switch (op) {
        case 0: value = output_type == VALUE_BOOL ? (x || (alpha && y)) : as_long((ulong)x + (ulong)alpha * (ulong)y); break;
        case 1: value = as_long((ulong)x - (ulong)alpha * (ulong)y); break;
        case 2: value = as_long((ulong)x * (ulong)y); break;
        case 3: value = x == y; break;
        case 4: value = x != y; break;
        case 5: value = x < y; break;
        case 6: value = x <= y; break;
        case 7: value = x > y; break;
        case 8: value = x >= y; break;
        case 9: value = output_type == VALUE_BOOL ? (x && y) : (x & y); break;
        case 10: value = output_type == VALUE_BOOL ? (x || y) : (x | y); break;
        case 11: value = output_type == VALUE_BOOL ? ((x != 0) != (y != 0)) : (x ^ y); break;
        case 12: value = output_type == VALUE_BOOL ? (x == 0) : (~x); break;
        case 13: value = as_long(0UL - (ulong)x); break;
        case 14: value = (y != 0) ? (x % y) : 0; break;
        case 15: {
            long r = (y != 0) ? (x % y) : 0;
            if (r != 0 && ((r < 0) != (y < 0))) r += y;
            value = r;
            break;
        }
        case 16: value = (x > 0) - (x < 0); break;
        case 17: value = x < 0 ? as_long(0UL - (ulong)x) : x; break;
        case 18: value = max(x, y); break;
        case 19: value = min(x, y); break;
        case 20: case 21: value = rounded_integer_divide(x, y, op == 21); break;
    }
    write_integer(output, dst, output_type, value);
}
)CLC",
R"CLC(
#if defined(IRIS_HAS_SUBGROUPS) && defined(IRIS_HAS_REQD_SG8)
// Multi-row M=1..8 subgroup GEMV for transposed RHS (right_row == 1, i.e., F.linear W^T)
// For M=1, processes 4 output columns per subgroup to maximize LPDDR memory-level parallelism!
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul_gemv_sg16_nt_m1(GEMM_ARGUMENTS, uint fp16) {
    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const size_t col0 = (get_group_id(0) * 8 + sg) * 4;
    if (col0 >= columns) return;
    const uint n_valid = (uint)min((ulong)4, columns - col0);
    const ulong b0 = right_offset + (col0 + 0) * right_col;
    const ulong b1 = right_offset + (col0 + (n_valid > 1 ? 1 : 0)) * right_col;
    const ulong b2 = right_offset + (col0 + (n_valid > 2 ? 2 : 0)) * right_col;
    const ulong b3 = right_offset + (col0 + (n_valid > 3 ? 3 : 0)) * right_col;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const ulong vec_k = inner >> 2;
    if (!fp16) {
        const __global float* lp = left + left_offset;
        const __global float* r0 = right + b0;
        const __global float* r1 = right + b1;
        const __global float* r2 = right + b2;
        const __global float* r3 = right + b3;
        for (ulong kv = lane; kv < vec_k; kv += 16) {
            const ulong ko = kv << 2;
            const float4 av = vload4(0, lp + ko);
            const float4 bv0 = vload4(0, r0 + ko);
            const float4 bv1 = vload4(0, r1 + ko);
            const float4 bv2 = vload4(0, r2 + ko);
            const float4 bv3 = vload4(0, r3 + ko);
            s0 = fma(av.s0, bv0.s0, fma(av.s1, bv0.s1, fma(av.s2, bv0.s2, fma(av.s3, bv0.s3, s0))));
            s1 = fma(av.s0, bv1.s0, fma(av.s1, bv1.s1, fma(av.s2, bv1.s2, fma(av.s3, bv1.s3, s1))));
            s2 = fma(av.s0, bv2.s0, fma(av.s1, bv2.s1, fma(av.s2, bv2.s2, fma(av.s3, bv2.s3, s2))));
            s3 = fma(av.s0, bv3.s0, fma(av.s1, bv3.s1, fma(av.s2, bv3.s2, fma(av.s3, bv3.s3, s3))));
        }
        if (lane == 0) {
            for (ulong k = vec_k << 2; k < inner; ++k) {
                const float av = lp[k];
                s0 = fma(av, r0[k], s0); s1 = fma(av, r1[k], s1);
                s2 = fma(av, r2[k], s2); s3 = fma(av, r3[k], s3);
            }
        }
    } else {
        const __global half* lp = ((__global const half*)left) + left_offset;
        const __global half* r0 = ((__global const half*)right) + b0;
        const __global half* r1 = ((__global const half*)right) + b1;
        const __global half* r2 = ((__global const half*)right) + b2;
        const __global half* r3 = ((__global const half*)right) + b3;
        for (ulong kv = lane; kv < vec_k; kv += 16) {
            const ulong ko = kv << 2;
            const float4 av = vload_half4(0, lp + ko);
            const float4 bv0 = vload_half4(0, r0 + ko);
            const float4 bv1 = vload_half4(0, r1 + ko);
            const float4 bv2 = vload_half4(0, r2 + ko);
            const float4 bv3 = vload_half4(0, r3 + ko);
            s0 = fma(av.s0, bv0.s0, fma(av.s1, bv0.s1, fma(av.s2, bv0.s2, fma(av.s3, bv0.s3, s0))));
            s1 = fma(av.s0, bv1.s0, fma(av.s1, bv1.s1, fma(av.s2, bv1.s2, fma(av.s3, bv1.s3, s1))));
            s2 = fma(av.s0, bv2.s0, fma(av.s1, bv2.s1, fma(av.s2, bv2.s2, fma(av.s3, bv2.s3, s2))));
            s3 = fma(av.s0, bv3.s0, fma(av.s1, bv3.s1, fma(av.s2, bv3.s2, fma(av.s3, bv3.s3, s3))));
        }
        if (lane == 0) {
            for (ulong k = vec_k << 2; k < inner; ++k) {
                const float av = vload_half(k, lp);
                s0 = fma(av, vload_half(k, r0), s0); s1 = fma(av, vload_half(k, r1), s1);
                s2 = fma(av, vload_half(k, r2), s2); s3 = fma(av, vload_half(k, r3), s3);
            }
        }
    }
    const float rsum0 = sub_group_reduce_add(s0);
    const float rsum1 = sub_group_reduce_add(s1);
    const float rsum2 = sub_group_reduce_add(s2);
    const float rsum3 = sub_group_reduce_add(s3);
    if (lane == 0) {
        const float sums[4] = {rsum0, rsum1, rsum2, rsum3};
        for (uint c = 0; c < n_valid; ++c) {
            write_float(output, col0 + c,
                        gemm_result(sums[c], bias, bias_offset + (col0 + c) * bias_col,
                                    alpha, beta, fp16 != 0), fp16 != 0);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul_gemv_sg16_nt(GEMM_ARGUMENTS, uint fp16) {
    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const size_t col = get_group_id(0) * 8 + sg;
    if (col >= columns) return;
    const ulong b_base = right_offset + col * right_col;
    const uint M = (uint)rows;
    float acc[8];
    #pragma unroll
    for (uint r = 0; r < 8; ++r) acc[r] = 0.0f;

    if (!fp16) {
        const ulong vec_k = inner >> 2;
        if (M == 1) {
            const ulong a0 = left_offset;
            float s0 = 0.0f;
            for (ulong kv = lane; kv < vec_k; kv += 16) {
                const float4 bv = vload4(0, right + b_base + (kv << 2));
                const float4 av = vload4(0, left + a0 + (kv << 2));
                s0 = fma(av.s0, bv.s0, fma(av.s1, bv.s1, fma(av.s2, bv.s2, fma(av.s3, bv.s3, s0))));
            }
            if (lane == 0) {
                for (ulong k = vec_k << 2; k < inner; ++k) s0 = fma(left[a0 + k], right[b_base + k], s0);
            }
            acc[0] = s0;
        } else {
            for (ulong kv = lane; kv < vec_k; kv += 16) {
                const float4 bv = vload4(0, right + b_base + (kv << 2));
                for (uint r = 0; r < M; ++r) {
                    const float4 av = vload4(0, left + left_offset + (ulong)r * left_row + (kv << 2));
                    acc[r] = fma(av.s0, bv.s0, fma(av.s1, bv.s1, fma(av.s2, bv.s2, fma(av.s3, bv.s3, acc[r]))));
                }
            }
            if (lane == 0) {
                for (ulong k = vec_k << 2; k < inner; ++k) {
                    const float bval = right[b_base + k];
                    for (uint r = 0; r < M; ++r) {
                        acc[r] = fma(left[left_offset + (ulong)r * left_row + k], bval, acc[r]);
                    }
                }
            }
        }
    } else {
        __global const half* lh = (__global const half*)left;
        __global const half* rh = (__global const half*)right + b_base;
        const ulong vec_k = inner >> 2;
        if (M == 1) {
            __global const half* l0 = lh + left_offset;
            float s0 = 0.0f;
            for (ulong kv = lane; kv < vec_k; kv += 16) {
                const float4 bv = vload_half4(0, rh + (kv << 2));
                const float4 av = vload_half4(0, l0 + (kv << 2));
                s0 = fma(av.s0, bv.s0, fma(av.s1, bv.s1, fma(av.s2, bv.s2, fma(av.s3, bv.s3, s0))));
            }
            if (lane == 0) {
                for (ulong k = vec_k << 2; k < inner; ++k) s0 = fma(vload_half(k, l0), vload_half(k, rh), s0);
            }
            acc[0] = s0;
        } else {
            for (ulong kv = lane; kv < vec_k; kv += 16) {
                const float4 bv = vload_half4(0, rh + (kv << 2));
                for (uint r = 0; r < M; ++r) {
                    const float4 av = vload_half4(0, lh + left_offset + (ulong)r * left_row + (kv << 2));
                    acc[r] = fma(av.s0, bv.s0, fma(av.s1, bv.s1, fma(av.s2, bv.s2, fma(av.s3, bv.s3, acc[r]))));
                }
            }
            if (lane == 0) {
                for (ulong k = vec_k << 2; k < inner; ++k) {
                    const float bval = vload_half(k, rh);
                    for (uint r = 0; r < M; ++r) {
                        acc[r] = fma(vload_half(left_offset + (ulong)r * left_row + k, lh), bval, acc[r]);
                    }
                }
            }
        }
    }
    for (uint r = 0; r < M; ++r) {
        const float sum = sub_group_reduce_add(acc[r]);
        if (lane == 0) {
            write_float(output, (ulong)r * columns + col,
                        gemm_result(sum, bias, bias_offset + (ulong)r * bias_row + col * bias_col,
                                    alpha, beta, fp16 != 0), fp16 != 0);
        }
    }
}

// Multi-row M=1..8 subgroup GEMV for row-major RHS (right_col == 1, coalesced along N)
__kernel void matmul_gemv_sg16_nn(GEMM_ARGUMENTS, uint fp16) {
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    const size_t col = get_group_id(0) * 16 + lx;
    const uint M = (uint)rows;
    float acc[8];
    #pragma unroll
    for (uint r = 0; r < 8; ++r) acc[r] = 0.0f;
    if (col < columns) {
        for (ulong k = ly; k < inner; k += 8) {
            const float bv = read_float(right, right_offset + k * right_row + col, fp16 != 0);
            for (uint r = 0; r < M; ++r) {
                const float av = read_float(left, left_offset + (ulong)r * left_row + k * left_col, fp16 != 0);
                acc[r] = fma(av, bv, acc[r]);
            }
        }
    }
    __local float smem[8][8][16];
    for (uint r = 0; r < M; ++r) smem[r][ly][lx] = acc[r];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && col < columns) {
        for (uint r = 0; r < M; ++r) {
            float sum = smem[r][0][lx] + smem[r][1][lx] + smem[r][2][lx] + smem[r][3][lx] +
                        smem[r][4][lx] + smem[r][5][lx] + smem[r][6][lx] + smem[r][7][lx];
            write_float(output, (ulong)r * columns + col,
                        gemm_result(sum, bias, bias_offset + (ulong)r * bias_row + col * bias_col,
                                    alpha, beta, fp16 != 0), fp16 != 0);
        }
    }
}
#endif
)CLC",
R"CLC(
__kernel void linear_quant_portable(
    __global float* output, __global const float* x, __global const uchar* weight,
    __global const float* scales, __global const float* zeros, __global const float* bias,
    ulong x_offset, ulong w_offset, ulong scale_offset, ulong zero_offset, ulong bias_offset,
    int n, int k, int group_size, int num_groups, int x_fp16, int scales_fp16,
    int has_zeros, int has_bias, int int4
) {
    const ulong col = get_global_id(0);
    const ulong row = get_global_id(1);
    const ulong qrow = w_offset + col * (int4 ? k / 2 : k);
    float value = 0.0f;
    for (int group = 0; group < num_groups; ++group) {
        const ulong group_index = col * num_groups + group;
        const float scale = read_float(scales, scale_offset + group_index, scales_fp16);
        const float zero = has_zeros ? read_float(zeros, zero_offset + group_index, scales_fp16)
                                     : (int4 ? 8.0f : 128.0f);
        const int end = (group + 1) * group_size;
        for (int i = group * group_size; i < end; ++i) {
            const uchar q = int4 ? ((weight[qrow + i / 2] >> (4 * (i % 2))) & 15)
                                 : weight[qrow + i];
            const float input = read_float(x, x_offset + row * k + i, x_fp16);
            value = fma(input, (q - zero) * scale, value);
        }
    }
    if (has_bias) value += read_float(bias, bias_offset + col, x_fp16);
    write_float(output, row * n + col, value, x_fp16);
}

#if defined(IRIS_HAS_SUBGROUPS) && defined(IRIS_HAS_REQD_SG8)
// Weight-only INT8 group-wise dequant-in-registers subgroup GEMV/GEMM (qweight: [N, K] uint8)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void linear_int8_gemv_sg16(
    __global float* output, __global const float* x, __global const uchar* qweight,
    __global const float* scales, __global const float* zeros, __global const float* bias,
    ulong x_off, ulong qw_off, ulong sc_off, ulong zp_off, ulong b_off,
    int M, int N, int K, int group_size, int num_groups,
    int x_fp16, int sc_fp16, int has_zeros, int has_bias
) {
    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const int n = get_group_id(0) * 8 + (int)sg;
    const int m0 = get_group_id(1) * 4;
    if (n >= N || m0 >= M) return;
    const int m_cnt = min(4, M - m0);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const __global uchar* qw_row = qweight + qw_off + (ulong)n * (ulong)K;
    const ulong sc_row = sc_off + (ulong)n * (ulong)num_groups;
    const ulong zp_row = zp_off + (ulong)n * (ulong)num_groups;
    const int vec_k = K >> 2;
    for (int kv = (int)lane; kv < vec_k; kv += 16) {
        const int k = kv << 2;
        const int g = k / group_size;
        const float sc = read_float(scales, sc_row + (ulong)g, sc_fp16 != 0);
        const float zp = has_zeros ? read_float(zeros, zp_row + (ulong)g, sc_fp16 != 0) : 128.0f;
        const uchar4 q = vload4(0, qw_row + k);
        const float4 w = (convert_float4(q) - (float4)(zp)) * sc;
        for (int r = 0; r < m_cnt; ++r) {
            const ulong xr = x_off + (ulong)(m0 + r) * (ulong)K + (ulong)k;
            const float4 xv = x_fp16 ? vload_half4(0, ((__global const half*)x) + xr) : vload4(0, x + xr);
            acc[r] = fma(xv.s0, w.s0, fma(xv.s1, w.s1, fma(xv.s2, w.s2, fma(xv.s3, w.s3, acc[r]))));
        }
    }
    if (lane == 0) {
        for (int k = vec_k << 2; k < K; ++k) {
            const int g = k / group_size;
            const float sc = read_float(scales, sc_row + (ulong)g, sc_fp16 != 0);
            const float zp = has_zeros ? read_float(zeros, zp_row + (ulong)g, sc_fp16 != 0) : 128.0f;
            const float w = ((float)qw_row[k] - zp) * sc;
            for (int r = 0; r < m_cnt; ++r) {
                const float xv = read_float(x, x_off + (ulong)(m0 + r) * (ulong)K + (ulong)k, x_fp16 != 0);
                acc[r] = fma(xv, w, acc[r]);
            }
        }
    }
    const float bval = has_bias ? read_float(bias, b_off + (ulong)n, x_fp16 != 0) : 0.0f;
    for (int r = 0; r < m_cnt; ++r) {
        const float sum = sub_group_reduce_add(acc[r]) + bval;
        if (lane == 0) {
            write_float(output, (ulong)(m0 + r) * (ulong)N + (ulong)n, sum, x_fp16 != 0);
        }
    }
}

// Weight-only INT4 (2 weights/byte, low nibble=even k, high nibble=odd k) group-wise dequant-in-registers subgroup GEMV/GEMM
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void linear_int4_gemv_sg16(
    __global float* output, __global const float* x, __global const uchar* qweight,
    __global const float* scales, __global const float* zeros, __global const float* bias,
    ulong x_off, ulong qw_off, ulong sc_off, ulong zp_off, ulong b_off,
    int M, int N, int K, int group_size, int num_groups,
    int x_fp16, int sc_fp16, int has_zeros, int has_bias
) {
    const uint lane = get_local_id(0);
    const uint sg = get_local_id(1);
    const int n = get_group_id(0) * 8 + (int)sg;
    const int m0 = get_group_id(1) * 4;
    if (n >= N || m0 >= M) return;
    const int m_cnt = min(4, M - m0);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int k_half = K >> 1;
    const __global uchar* qw_row = qweight + qw_off + (ulong)n * (ulong)k_half;
    const ulong sc_row = sc_off + (ulong)n * (ulong)num_groups;
    const ulong zp_row = zp_off + (ulong)n * (ulong)num_groups;
    const int vec8 = K >> 3; // 8 weights = 4 packed bytes per iteration
    for (int kv = (int)lane; kv < vec8; kv += 16) {
        const int k = kv << 3;
        const int g = k / group_size;
        const float sc = read_float(scales, sc_row + (ulong)g, sc_fp16 != 0);
        const float zp = has_zeros ? read_float(zeros, zp_row + (ulong)g, sc_fp16 != 0) : 8.0f;
        const uchar4 pk = vload4(0, qw_row + (kv << 2));
        const float4 w0 = ((float4)((float)(pk.s0 & 15u), (float)(pk.s0 >> 4),
                                    (float)(pk.s1 & 15u), (float)(pk.s1 >> 4)) - (float4)(zp)) * sc;
        const float4 w1 = ((float4)((float)(pk.s2 & 15u), (float)(pk.s2 >> 4),
                                    (float)(pk.s3 & 15u), (float)(pk.s3 >> 4)) - (float4)(zp)) * sc;
        for (int r = 0; r < m_cnt; ++r) {
            const ulong xr = x_off + (ulong)(m0 + r) * (ulong)K + (ulong)k;
            float4 x0, x1;
            if (x_fp16) {
                const __global half* xh = ((__global const half*)x) + xr;
                x0 = vload_half4(0, xh);
                x1 = vload_half4(0, xh + 4);
            } else {
                x0 = vload4(0, x + xr);
                x1 = vload4(0, x + xr + 4);
            }
            acc[r] = fma(x0.s0, w0.s0, fma(x0.s1, w0.s1, fma(x0.s2, w0.s2, fma(x0.s3, w0.s3, acc[r]))));
            acc[r] = fma(x1.s0, w1.s0, fma(x1.s1, w1.s1, fma(x1.s2, w1.s2, fma(x1.s3, w1.s3, acc[r]))));
        }
    }
    if (lane == 0) {
        for (int k = vec8 << 3; k < K; k += 2) {
            const int g = k / group_size;
            const float sc = read_float(scales, sc_row + (ulong)g, sc_fp16 != 0);
            const float zp = has_zeros ? read_float(zeros, zp_row + (ulong)g, sc_fp16 != 0) : 8.0f;
            const uchar pk = qw_row[k >> 1];
            const float w0 = ((float)(pk & 15u) - zp) * sc;
            const float w1 = ((float)(pk >> 4) - zp) * sc;
            for (int r = 0; r < m_cnt; ++r) {
                const ulong xr = x_off + (ulong)(m0 + r) * (ulong)K + (ulong)k;
                acc[r] = fma(read_float(x, xr, x_fp16 != 0), w0, acc[r]);
                acc[r] = fma(read_float(x, xr + 1, x_fp16 != 0), w1, acc[r]);
            }
        }
    }
    const float bval = has_bias ? read_float(bias, b_off + (ulong)n, x_fp16 != 0) : 0.0f;
    for (int r = 0; r < m_cnt; ++r) {
        const float sum = sub_group_reduce_add(acc[r]) + bval;
        if (lane == 0) {
            write_float(output, (ulong)(m0 + r) * (ulong)N + (ulong)n, sum, x_fp16 != 0);
        }
    }
}
#endif
)CLC"
};

// Built separately with OpenCL C 3.0 on drivers exposing accelerated signed packed dot products.
inline constexpr const char* int8_packed_kernels = R"CLC(
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void int8_matmul_packed(__global int* out, __global const uint* left, __global const uint* right_transposed,
                        ulong output_offset, ulong columns, ulong packs, ulong left_offset, ulong right_offset) {
    const uint x = get_local_id(0);
    const uint y = get_local_id(1);
    const uint lane = y * 8 + x;
    const ulong col = get_group_id(0) * 32;
    const ulong row = get_group_id(1) * 32;
    __local uint a_tile[16][33];
    __local uint b_tile[16][33];
    uint sums[4][4] = {{0}};

    for (ulong k = 0; k < packs; k += 16) {
        for (uint i = lane; i < 32 * 16; i += 64) {
            const uint output = i / 16;
            const uint pack = i % 16;
            a_tile[pack][output] = left[left_offset + (row + output) * packs + k + pack];
            b_tile[pack][output] = right_transposed[right_offset + (col + output) * packs + k + pack];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint pack = 0; pack < 16; ++pack) {
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const uint a = a_tile[pack][y + i * 8];
                #pragma unroll
                for (uint j = 0; j < 4; ++j) {
                    const uint b = b_tile[pack][x + j * 8];
                    sums[i][j] += (uint)dot_4x8packed_ss_int(a, b);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            out[output_offset + (row + y + i * 8) * columns + col + x + j * 8] = as_int(sums[i][j]);
        }
    }
}
)CLC";

}  // namespace iris
