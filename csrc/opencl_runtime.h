#pragma once

#include <CL/cl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace iris {

enum class Elementwise : cl_uint {
    Add, Multiply, Relu, Copy, Fill, Scale, ThresholdBackward, SquaredDifference,
    ScaledDifference, Divide, Ceil, Floor, Trunc, Abs, Sqrt, Rsqrt, Exp, Log, Tanh,
    Sigmoid, Silu, Gelu, GeluTanh, AddScalar, Subtract, DivideTensor, PowScalar, PowTensor,
    Clamp, Reciprocal, TanhBackward, SigmoidBackward, SiluBackward, GeluBackward,
    GeluTanhBackward, IsFinite, IsNan, IsInf, Eq, Ne, Lt, Le, Gt, Ge, BoolAnd, Maximum, Minimum,
    Sin, Cos, Neg, LogicalAnd, LogicalOr, LogicalNot, LogicalXor, SubScalar, RsubScalar,
    Atan2, Erf, Log1p, Expm1, FmodTensor, FmodScalar, RemainderTensor, RemainderScalar, RemainderScalarTensor,
    Exp2, Log2, Log10, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh, Erfc, Frac, Round, Sign, RoundDecimals, Softplus, LogSigmoid, LogSigmoidBuffer, SoftplusBackward, LogSigmoidBackward,
    DivideTrunc, DivideFloor
};

enum class RandomDistribution : cl_uint { Uniform, Normal, Bernoulli };
enum class ValueType : cl_uint { Float, Long, Bool, Byte, Int, Half, Char, Short, BFloat, Complex };
enum class Extrema : cl_uint { Min, Max, Any, All };
enum class IndexOperation : cl_uint { Select, Gather, Scatter, ScatterAdd, ReduceSum, ReduceProduct,
                                      ReduceMean, ReduceMax, ReduceMin };
enum class IntegerOperation : cl_uint {
    Add, Subtract, Multiply, Eq, Ne, Lt, Le, Gt, Ge,
    BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, Neg, Fmod, Remainder, Sign, Abs, Maximum, Minimum,
    DivideTrunc, DivideFloor
};

bool gpu_available() noexcept;
bool has_fp16() noexcept;
bool context_initialized() noexcept;
const std::string& gpu_name();
struct DeviceProperties {
    cl_ulong total_memory;
    cl_ulong max_allocation;
    cl_device_fp_config double_fp_config;
    cl_bool host_unified_memory;
    cl_uint compute_units;
    std::string opencl_version;
    std::string driver_version;
};
DeviceProperties device_properties();
void initialize();
cl_mem allocate(std::size_t bytes);
void release(void* buffer) noexcept;
std::size_t external_buffer_size(cl_mem buffer);
void upload(cl_mem destination, const void* source, std::size_t bytes, std::size_t offset = 0,
            bool non_blocking = false);
void download(void* destination, cl_mem source, std::size_t bytes, std::size_t offset = 0,
              bool non_blocking = false);
void copy(cl_mem destination, cl_mem source, std::size_t bytes,
          std::size_t destination_offset = 0, std::size_t source_offset = 0);
void fill(cl_mem destination, std::size_t length, std::size_t offset, float value);
void fill_bytes(cl_mem destination, std::size_t length, std::size_t offset,
                const void* value, std::size_t item_size);
void elementwise(
    Elementwise op, cl_mem output, cl_mem left, cl_mem right,
    std::size_t length, float alpha);
void scale_add(cl_mem output, cl_mem input, cl_ulong length, float scale, float bias);
void contiguous_elementwise(
    Elementwise op, cl_mem output, cl_mem left, cl_mem right,
    cl_ulong length, cl_ulong output_offset, cl_ulong left_offset, cl_ulong right_offset,
    float alpha, float beta, bool fp16);
void strided_elementwise(
    Elementwise op, cl_mem output, cl_mem left, cl_mem right, cl_ulong length,
    cl_ulong output_offset, cl_ulong left_offset, cl_ulong right_offset,
    std::span<const cl_ulong> geometry, float alpha, float beta, bool serial, bool fp16);
void typed_copy(cl_mem output, cl_mem input, cl_ulong length, cl_ulong output_offset,
                cl_ulong input_offset, std::span<const cl_ulong> geometry, ValueType output_type,
                ValueType input_type, cl_ulong integer_fill, bool fill,
                bool conjugate = false, bool negate = false);
void integer_binary(cl_mem output, cl_mem left, cl_mem right, cl_ulong length,
                    cl_ulong output_offset, cl_ulong left_offset, cl_ulong right_offset,
                    std::span<const cl_ulong> geometry, ValueType input_type, ValueType output_type,
                    IntegerOperation operation, cl_long alpha);
void masked_copy(cl_mem output, cl_mem count, cl_mem input, cl_mem mask, cl_ulong length,
                 cl_ulong input_offset, cl_ulong mask_offset, std::span<const cl_ulong> geometry,
                 ValueType type);
void extrema(cl_mem output, cl_mem input, cl_ulong length, cl_ulong offset,
             std::span<const cl_ulong> geometry, ValueType type, Extrema op);
void integer_sum(cl_mem output, cl_mem input, cl_ulong elements, cl_ulong reduce_length,
                 cl_ulong offset, cl_ulong outer_rank, std::span<const cl_ulong> geometry,
                 ValueType input_type, ValueType output_type, bool contiguous, cl_ulong chunks = 1);
void logical_reduce(cl_mem output, cl_mem input, cl_ulong length, cl_ulong reduce_length,
                    cl_ulong offset, cl_ulong outer_rank, std::span<const cl_ulong> geometry,
                    ValueType type, bool all);
void triangular(cl_mem output, cl_mem input, cl_ulong length, cl_ulong offset, cl_ulong rows,
                cl_ulong columns, cl_long diagonal, std::span<const cl_ulong> geometry,
                ValueType type, bool upper);
void tril_indices(cl_mem output, cl_ulong count, cl_ulong offset, cl_ulong row_stride,
                  cl_ulong column_stride, cl_ulong first_row, cl_ulong first_width,
                  cl_ulong growing_rows, cl_ulong growing_count, cl_ulong columns, bool int32);
void indexed_copy(cl_mem output, cl_mem input, cl_mem index, cl_mem invalid, cl_ulong work_items,
                  cl_ulong output_offset, cl_ulong input_offset, cl_ulong index_offset,
                  std::span<const cl_ulong> geometry, cl_ulong axis, cl_ulong axis_size,
                  ValueType data_type, ValueType index_type, IndexOperation operation, bool include_self);
void embedding_gradient(cl_mem output, cl_mem grad, cl_mem index, cl_ulong length,
                        cl_ulong index_offset, cl_ulong index_stride, cl_ulong grad_offset,
                        cl_ulong grad_row, cl_ulong grad_column, cl_ulong count, cl_ulong width,
                        cl_long padding, ValueType index_type, bool scale);
void tensor_where(cl_mem output, cl_mem condition, cl_mem left, cl_mem right, cl_ulong length,
                  cl_ulong output_offset, cl_ulong condition_offset, cl_ulong left_offset,
                  cl_ulong right_offset, std::span<const cl_ulong> geometry, ValueType type);
void nonzero_coordinates(cl_mem coordinates, cl_mem count, cl_mem input, cl_ulong length,
                         cl_ulong offset, std::span<const cl_ulong> geometry, ValueType type);
void advanced_offsets(cl_mem offsets, cl_mem index, cl_mem invalid, cl_ulong length,
                      cl_ulong index_offset, std::span<const cl_ulong> geometry, cl_ulong axis_size,
                      cl_ulong axis_stride, ValueType type);
void advanced_copy(cl_mem base, cl_mem linear, cl_mem offsets, cl_ulong length, cl_ulong base_offset,
                   cl_ulong trailing, cl_ulong index_length, std::span<const cl_ulong> geometry,
                   ValueType type, bool put, bool accumulate);
struct MatrixView {
    cl_mem buffer = nullptr;
    cl_ulong offset = 0;
    cl_ulong batch_stride = 0;
    cl_ulong row_stride = 0;
    cl_ulong column_stride = 0;
};
void integer_matmul(cl_mem output, MatrixView left, MatrixView right, cl_ulong batches,
                    cl_ulong rows, cl_ulong columns, cl_ulong inner, ValueType type);
bool supports_packed_int8_matmul();
void packed_int8_matmul(cl_mem output, cl_ulong output_offset, MatrixView left, MatrixView right_transposed,
                        cl_ulong rows, cl_ulong columns, cl_ulong inner);
void strided_int8_matmul(cl_mem output, cl_ulong output_offset, MatrixView left, MatrixView right,
                         cl_ulong rows, cl_ulong columns, cl_ulong inner);

struct BufferSlice {
    cl_mem buffer = nullptr;
    cl_ulong offset = 0;
};

void fft_radix2(BufferSlice output, BufferSlice real, BufferSlice imaginary,
                cl_ulong batches, cl_uint length, bool forward);

void recurrent_forward(BufferSlice output, BufferSlice new_cell, BufferSlice workspace,
                       BufferSlice input, BufferSlice hidden, BufferSlice state,
                       BufferSlice input_bias, BufferSlice hidden_bias,
                       cl_ulong length, cl_ulong width, bool lstm, bool fp16);
void recurrent_backward(BufferSlice input_gates, BufferSlice hidden_gates, BufferSlice grad_state,
                        BufferSlice grad_output, BufferSlice grad_cell, BufferSlice state,
                        BufferSlice new_cell, BufferSlice workspace,
                        cl_ulong length, cl_ulong width, bool lstm, bool fp16);

void add_scalar_batch(const std::array<BufferSlice, 8>& batch, std::size_t count, float scalar);
void amp_unscale(BufferSlice grad, BufferSlice found_inf, BufferSlice inv_scale, cl_ulong length, bool fp16);
void amp_update_scale(BufferSlice scale, BufferSlice growth, BufferSlice found_inf,
                      float growth_factor, float backoff_factor, cl_int growth_interval);

void run_spatial(const char* name, std::size_t work_items, std::size_t local_size,
                 std::span<const cl_ulong> geometry, const std::array<BufferSlice, 6>& tensors);

struct Conv2dParams {
    cl_int n, c_in, h, w, c_out, oh, ow;
    cl_int kh, kw, sy, sx, py, px, dy, dx, groups;
};

void conv2d_forward(BufferSlice output, BufferSlice input, BufferSlice weight, BufferSlice bias,
                    const Conv2dParams& p, bool fp16 = false);
void conv2d_grad_input(BufferSlice grad_input, BufferSlice grad_output, BufferSlice weight,
                       const Conv2dParams& p, bool fp16 = false);
void conv2d_grad_weight(BufferSlice grad_weight, BufferSlice partial, BufferSlice grad_output,
                        BufferSlice input, const Conv2dParams& p, cl_int split_k, bool fp16 = false);

struct Pool2dParams {
    cl_int n, c, h, w, oh, ow;
    cl_int kh, kw, sy, sx, py, px, dy, dx;
    cl_int include_pad, divisor;
};

void max_pool2d_forward(BufferSlice output, BufferSlice indices, BufferSlice input, const Pool2dParams& p, bool fp16 = false);
void max_pool2d_backward(BufferSlice grad_input, BufferSlice grad_output, BufferSlice indices, const Pool2dParams& p, bool fp16 = false);
void avg_pool2d_forward(BufferSlice output, BufferSlice input, const Pool2dParams& p, bool fp16 = false);
void avg_pool2d_backward(BufferSlice grad_input, BufferSlice grad_output, const Pool2dParams& p, bool fp16 = false);

struct SdpaParams {
    cl_int b, hq, hkv, lq, lk, dq, dv;
    cl_ulong q_sb, q_sh, q_sl, q_sd;
    cl_ulong k_sb, k_sh, k_sl, k_sd;
    cl_ulong v_sb, v_sh, v_sl, v_sd;
    cl_ulong o_sb, o_sh, o_sl, o_sd;
    float scale;
    bool causal;
    bool fp16 = false;
};

struct SdpaBwdParams {
    SdpaParams fwd;
    cl_ulong do_sb, do_sh, do_sl, do_sd;
    cl_ulong dq_sb, dq_sh, dq_sl, dq_sd;
    cl_ulong dk_sb, dk_sh, dk_sl, dk_sd;
    cl_ulong dv_sb, dv_sh, dv_sl, dv_sd;
};

bool has_subgroups() noexcept;
void sdpa_flash_forward(BufferSlice output, BufferSlice lse, BufferSlice query, BufferSlice key, BufferSlice value,
                        const SdpaParams& p);
void sdpa_flash_decode(BufferSlice output, BufferSlice partial_o, BufferSlice partial_lse,
                       BufferSlice query, BufferSlice key, BufferSlice value,
                       const SdpaParams& p, cl_int num_splits);
void sdpa_flash_backward(BufferSlice dq, BufferSlice dk, BufferSlice dv, BufferSlice d_buf,
                         BufferSlice dout, BufferSlice out, BufferSlice query, BufferSlice key, BufferSlice value,
                         BufferSlice lse, const SdpaBwdParams& p, bool need_dq, bool need_dk, bool need_dv);
void cast_fp32_fp16_contiguous(cl_mem output, cl_mem input, std::size_t length, bool to_fp16);
void softmax_lastdim_contiguous(cl_mem output, cl_mem input, cl_ulong input_offset,
                                cl_ulong rows, cl_ulong width, bool logarithmic, bool fp16 = false);
void softmax_backward_lastdim(cl_mem grad_in, cl_mem grad_out, cl_mem output,
                              cl_ulong grad_in_off, cl_ulong grad_out_off, cl_ulong out_off,
                              cl_ulong rows, cl_ulong width, bool logarithmic, bool fp16 = false);
void nll_loss_forward_fused(cl_mem loss_out, cl_mem total_weight_out,
                            MatrixView input, MatrixView target, MatrixView weight,
                            cl_ulong batch, cl_ulong classes, cl_long ignore_index, bool mean);
void nll_loss_backward_fused(cl_mem grad_input, MatrixView target, MatrixView weight,
                             MatrixView grad, MatrixView total_weight,
                             cl_ulong batch, cl_ulong classes, cl_long ignore_index, bool mean);

void sum_all_contiguous(cl_mem output, cl_mem partial, cl_mem input, cl_ulong length,
                        cl_ulong input_offset, cl_ulong num_groups, bool average);
void sum_2d_dim0(cl_mem output, cl_mem input, cl_ulong input_offset, cl_int rows, cl_int cols,
                 bool average);
void sum_4d_nchw_c(cl_mem output, cl_mem input, cl_ulong input_offset, cl_int n, cl_int c,
                   cl_int hw, bool average);

struct NllLoss {
    cl_mem output;
    cl_mem weights;
    cl_mem invalid;
    MatrixView input;
    MatrixView target;
    MatrixView weight;
    MatrixView grad;
    MatrixView total_weight;
    cl_ulong batch;
    cl_ulong classes;
    cl_long ignore_index;
    bool mean;
};

void nll_loss(const NllLoss& arguments);
void nll_loss_async(const NllLoss& arguments);
void check_async_errors();

// Phase 2 Vision & LLM runtime entry points
void conv_transpose2d_forward(BufferSlice output, BufferSlice input, BufferSlice weight, BufferSlice bias,
                              const Conv2dParams& p, bool fp16 = false);
void batch_norm_forward(cl_mem output, cl_mem save_mean, cl_mem save_invstd,
                        cl_mem running_mean, cl_mem running_var,
                        cl_mem input, cl_mem weight, cl_mem bias,
                        cl_int n, cl_int c, cl_int s, bool training, float momentum, float eps,
                        bool in_fp16 = false, bool param_fp16 = false, bool stat_fp16 = false);
void batch_norm_backward(cl_mem grad_in, cl_mem grad_weight, cl_mem grad_bias,
                         cl_mem sum_dy, cl_mem sum_dy_xmu,
                         cl_mem grad_out, cl_mem input, cl_mem weight,
                         cl_mem mean_buf, cl_mem var_or_invstd,
                         cl_int n, cl_int c, cl_int s, float eps, bool train,
                         bool in_fp16 = false, bool param_fp16 = false, bool stat_fp16 = false);
void group_norm_forward(cl_mem output, cl_mem mean_out, cl_mem rstd_out,
                        cl_mem input, cl_mem weight, cl_mem bias,
                        cl_int n, cl_int c, cl_int hxw, cl_int groups, float eps,
                        bool in_fp16 = false, bool param_fp16 = false);
void group_norm_backward(cl_mem grad_in, cl_mem grad_weight, cl_mem grad_bias,
                         cl_mem grad_out, cl_mem input, cl_mem mean_buf, cl_mem rstd_buf, cl_mem weight,
                         cl_int n, cl_int c, cl_int hxw, cl_int groups,
                         bool in_fp16 = false, bool param_fp16 = false);
void layer_norm_forward_fast(BufferSlice output, BufferSlice input, BufferSlice weight, BufferSlice bias,
                             BufferSlice mean_out, BufferSlice rstd_out, cl_ulong rows, cl_int width, float eps,
                             bool in_fp16 = false, bool param_fp16 = false);
void layer_norm_grad_input_fast(BufferSlice dx, BufferSlice grad, BufferSlice input, BufferSlice weight,
                                BufferSlice mean_buf, BufferSlice rstd_buf, cl_ulong rows, cl_int width,
                                bool in_fp16 = false, bool param_fp16 = false);
void layer_norm_grad_affine_fast(BufferSlice dw, BufferSlice db, BufferSlice grad, BufferSlice input,
                                 BufferSlice mean_buf, BufferSlice rstd_buf, cl_int rows, cl_int width,
                                 bool in_fp16 = false, bool param_fp16 = false);
void rms_norm_forward(BufferSlice output, cl_mem rstd_out, BufferSlice input, BufferSlice weight,
                      cl_ulong rows, cl_int width, float eps, bool in_fp16 = false, bool param_fp16 = false);
void rms_norm_backward(BufferSlice dx, BufferSlice dw, BufferSlice grad, BufferSlice input, BufferSlice weight,
                       cl_mem rstd_buf, cl_int rows, cl_int width, bool in_fp16 = false, bool param_fp16 = false);
void adaptive_avg_pool2d_forward(cl_mem output, cl_mem input, cl_int nc, cl_int h, cl_int w, cl_int oh, cl_int ow, bool fp16 = false);
void adaptive_avg_pool2d_backward(cl_mem grad_in, cl_mem grad_out, cl_int nc, cl_int h, cl_int w, cl_int oh, cl_int ow, bool fp16 = false);
void upsample_nearest2d_forward(cl_mem output, cl_mem input, cl_int nc, cl_int h, cl_int w,
                                cl_int oh, cl_int ow, float scale_h, float scale_w, bool fp16 = false);
void upsample_nearest2d_backward(cl_mem grad_in, cl_mem grad_out, cl_int nc, cl_int h, cl_int w,
                                 cl_int oh, cl_int ow, float scale_h, float scale_w, bool fp16 = false);
void upsample_bilinear2d_forward(cl_mem output, cl_mem input, cl_int nc, cl_int h, cl_int w,
                                 cl_int oh, cl_int ow, float rheight, float rwidth, bool align_corners, bool fp16 = false);
void upsample_bilinear2d_backward(cl_mem grad_in, cl_mem grad_out, cl_int nc, cl_int h, cl_int w,
                                  cl_int oh, cl_int ow, float rheight, float rwidth, bool align_corners, bool fp16 = false);
void linear_quant_forward(cl_mem output, BufferSlice x, BufferSlice qweight, BufferSlice scales,
                          BufferSlice zeros, BufferSlice bias,
                          cl_int m, cl_int n, cl_int k, cl_int group_size, cl_int num_groups,
                          bool x_fp16, bool sc_fp16, bool is_int4);
void argmax_argmin(cl_mem out_indices, cl_mem out_values, cl_mem input,
                   cl_ulong outer_size, cl_ulong dim_size, cl_ulong inner_size,
                   ValueType type, bool is_min, bool write_values);
void sort_rows_f32(cl_mem out_vals, cl_mem out_idxs, cl_mem input,
                   cl_int rows, cl_int width, cl_int out_k, bool descending, bool fp16,
                   cl_mem tmp_vals = nullptr, cl_mem tmp_idxs = nullptr);
void sort_rows_int(cl_mem out_vals, cl_mem out_idxs, cl_mem input,
                   cl_int rows, cl_int width, cl_int out_k, bool descending, ValueType type,
                   cl_mem tmp_vals = nullptr, cl_mem tmp_idxs = nullptr);

// Phase 3 runtime entry points
void multinomial_build_cdf_launch(cl_mem cdf, cl_mem probs, cl_mem invalid, cl_int rows, cl_int k, bool fp16);
void multinomial_sample_repl_launch(cl_mem out, cl_mem cdf, cl_int rows, cl_int k, cl_int num_samples,
                                    cl_ulong seed, cl_ulong start_counter);
void multinomial_exp_keys_launch(cl_mem keys, cl_mem probs, cl_mem invalid, cl_int rows, cl_int k,
                                 cl_int num_samples, cl_ulong seed, cl_ulong start_counter, bool fp16);
void searchsorted_launch(cl_mem out, cl_mem seq, cl_mem vals, cl_mem sorter,
                         cl_ulong out_len, cl_ulong seq_rows, cl_ulong seq_len, cl_ulong vals_per_row,
                         ValueType seq_type, ValueType val_type, bool has_sorter, bool right, bool out_int32);
cl_ulong masked_scatter_launch(cl_mem dst, cl_mem mask, cl_mem src, cl_mem block_counts, cl_mem block_offsets,
                               cl_mem total_out, cl_ulong length, cl_ulong src_len, ValueType type);
void isin_launch(cl_mem out, cl_mem elems, cl_mem tests, cl_ulong elem_len, cl_ulong test_len,
                 ValueType elem_type, ValueType test_type, bool is_sorted, bool invert);
void index_add_launch(cl_mem self_buf, cl_mem src_buf, cl_mem idx_buf, cl_mem invalid,
                      cl_ulong outer_size, cl_ulong self_dim, cl_ulong idx_len, cl_ulong inner_size,
                      float alpha_f, cl_long alpha_l, ValueType data_type, ValueType idx_type);
void repeat_interleave_fill_launch(cl_mem out, cl_mem ends, cl_ulong total_out, cl_ulong num_repeats);
void roll_launch(cl_mem out, cl_mem inp, std::span<const cl_ulong> sizes, std::span<const cl_long> shifts,
                 cl_ulong length, ValueType type, bool flat_mode, cl_long flat_shift);
void constant_pad_nd_launch(cl_mem out, cl_mem inp, std::span<const cl_ulong> out_sizes,
                            std::span<const cl_ulong> in_sizes, std::span<const cl_long> pad_before,
                            cl_ulong length, ValueType type, float fill_f, cl_long fill_l);
void pad2d_mode_fwd_launch(cl_mem out, cl_mem inp, cl_int nc, cl_int h, cl_int w, cl_int oh, cl_int ow,
                           cl_int pad_l, cl_int pad_t, cl_int mode, bool fp16);
void pad2d_mode_bwd_launch(cl_mem grad_in, cl_mem grad_out, cl_int nc, cl_int h, cl_int w, cl_int oh, cl_int ow,
                           cl_int pad_l, cl_int pad_r, cl_int pad_t, cl_int pad_b, cl_int mode, bool fp16);
void act_p3_fwd_launch(cl_mem out, cl_mem inp, cl_ulong length, cl_int act,
                       float p0, float p1, float p2, bool fp16);
void act_p3_bwd_launch(cl_mem grad_in, cl_mem grad_out, cl_mem self_or_res, cl_ulong length, cl_int act,
                       float p0, float p1, float p2, bool is_result, bool fp16);
void prelu_fwd_launch(cl_mem out, cl_mem inp, cl_mem weight, cl_ulong length,
                      cl_ulong channels, cl_ulong inner_size, bool single_weight, bool fp16);
void prelu_bwd_launch(cl_mem dx, cl_mem dw, cl_mem grad_out, cl_mem inp, cl_mem weight,
                      cl_ulong outer_size, cl_ulong channels, cl_ulong inner_size,
                      bool single_weight, bool fp16);
void grid_sampler_2d_fwd_launch(cl_mem out, cl_mem inp, cl_mem grid,
                                cl_int n, cl_int c, cl_int ih, cl_int iw, cl_int oh, cl_int ow,
                                cl_int interp_mode, cl_int padding_mode, bool align_corners, bool fp16);
void grid_sampler_2d_bwd_launch(cl_mem grad_in_f32, cl_mem grad_grid, cl_mem grad_out, cl_mem inp, cl_mem grid,
                                cl_int n, cl_int c, cl_int ih, cl_int iw, cl_int oh, cl_int ow,
                                cl_int interp_mode, cl_int padding_mode, bool align_corners,
                                bool need_gi, bool need_gg, bool fp16);

struct Conv3dParams {
    cl_int n, c_in, id, ih, iw, c_out, od, oh, ow;
    cl_int kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups;
};

void conv3d_fwd_launch(cl_mem out, cl_mem inp, cl_mem w, cl_mem bias, const Conv3dParams& p, bool fp16);
void conv3d_grad_input_launch(cl_mem grad_in, cl_mem grad_out, cl_mem w, const Conv3dParams& p, bool fp16);
void conv3d_grad_weight_launch(cl_mem grad_w, cl_mem grad_out, cl_mem inp, const Conv3dParams& p, bool fp16);

struct Pool3dParams {
    cl_int nc, id, ih, iw, od, oh, ow;
    cl_int kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw;
    cl_int include_pad, divisor;
};

void max_pool3d_fwd_launch(cl_mem out, cl_mem indices, cl_mem inp, const Pool3dParams& p, bool fp16);
void max_pool3d_bwd_launch(cl_mem grad_in_f32, cl_mem grad_out, cl_mem indices, const Pool3dParams& p, bool fp16);
void avg_pool3d_fwd_launch(cl_mem out, cl_mem inp, const Pool3dParams& p, bool fp16);
void avg_pool3d_bwd_launch(cl_mem grad_in, cl_mem grad_out, const Pool3dParams& p, bool fp16);
void var_std_reduce_launch(cl_mem var_out, cl_mem mean_out, cl_mem inp,
                           cl_ulong outer_size, cl_ulong reduce_len, float correction,
                           bool take_sqrt, bool write_mean, bool fp16);
void prod_reduce_launch(cl_mem out, cl_mem inp, cl_ulong outer_size, cl_ulong reduce_len,
                        ValueType in_type, ValueType out_type);
void vector_norm_reduce_launch(cl_mem out, cl_mem inp, cl_ulong outer_size, cl_ulong reduce_len,
                               float ord, cl_int ord_mode, bool in_fp16, bool out_fp16);
void ternary_contiguous_launch(cl_mem out, cl_mem a, cl_mem b, cl_mem c,
                               cl_ulong length, cl_int op, float alpha, bool fp16);
void cumsum_launch(cl_mem output, cl_mem input, cl_ulong outer_size, cl_ulong dim_size, cl_ulong inner_size,
                   ValueType in_type, ValueType out_type);
void arange_launch(cl_mem output, cl_ulong length, float start_f, float step_f,
                   cl_long start_l, cl_long step_l, ValueType type);
void linspace_launch(cl_mem output, cl_ulong steps, float start_f, float end_f, ValueType type);
void random_int_fill(cl_mem output, cl_ulong length, cl_ulong offset, std::span<const cl_ulong> geometry,
                     cl_ulong seed, cl_ulong start, cl_long low, cl_ulong range, ValueType type);
void flip_launch(cl_mem output, cl_mem input, cl_ulong length, cl_ulong in_offset,
                 std::span<const cl_ulong> geometry, cl_uint flip_mask, ValueType type);
void fused_sgd_step(cl_mem param, cl_mem grad, cl_mem mom_buf, cl_mem lr_ptr,
                    cl_mem grad_scale, cl_mem found_inf,
                    cl_ulong p_off, cl_ulong g_off, cl_ulong m_off, cl_ulong length,
                    float weight_decay, float momentum, float lr_scalar, float dampening,
                    bool nesterov, bool maximize, bool is_first_step);
void fused_adam_step(cl_mem param, cl_mem grad, cl_mem exp_avg, cl_mem exp_avg_sq, cl_mem max_exp_avg_sq,
                     cl_mem state_step, cl_mem lr_ptr, cl_mem grad_scale, cl_mem found_inf,
                     cl_ulong p_off, cl_ulong g_off, cl_ulong ea_off, cl_ulong eas_off, cl_ulong meas_off,
                     cl_ulong step_off, cl_ulong length,
                     float lr_scalar, float beta1, float beta2, float weight_decay, float eps,
                     bool amsgrad, bool maximize, bool decoupled_wd);
std::pair<std::uint64_t, std::uint64_t> reserve_rng_blocks(std::uint64_t blocks);

void matrix_multiply(
    cl_mem output, MatrixView left, MatrixView right,
    cl_ulong batches, cl_ulong rows, cl_ulong columns, cl_ulong inner,
    bool fp16, MatrixView bias = {}, float alpha = 1.0f, float beta = 0.0f);
void sum(
    cl_mem output, cl_mem input, cl_ulong length, cl_ulong reduce_length, cl_ulong offset,
    cl_ulong outer_rank, std::span<const cl_ulong> geometry, bool input_half, bool output_half,
    bool average);
void softmax(
    cl_mem output, cl_mem input, cl_ulong length, cl_ulong width, cl_ulong inner,
    cl_ulong offset, cl_ulong dim_stride, std::span<const cl_ulong> geometry, bool logarithmic, bool fp16);
void random_fill(cl_mem output, cl_ulong length, cl_ulong offset, std::span<const cl_ulong> geometry,
                 cl_ulong seed, cl_ulong start, float first, float second, RandomDistribution distribution,
                 bool fp16);
void synchronize();
std::int64_t new_stream();
std::int64_t current_stream() noexcept;
std::int64_t exchange_stream(std::int64_t stream);
void synchronize_stream(std::int64_t stream);
bool query_stream(std::int64_t stream);
cl_event record_event(std::int64_t stream);
void wait_event(std::int64_t stream, cl_event event);
bool query_event(cl_event event);
void synchronize_event(cl_event event);
void release_event(cl_event event) noexcept;
double elapsed_time(cl_event first, cl_event last);
void record_buffer(cl_mem buffer, std::int64_t stream);

struct MemoryStats {
    std::size_t allocated_bytes = 0;
    std::size_t cached_bytes = 0;
    std::uint64_t allocations = 0;
    std::uint64_t reuses = 0;
    std::size_t peak_allocated_bytes = 0;
    std::size_t peak_reserved_bytes = 0;
    std::uint64_t cache_reclaims = 0;
};

void empty_cache();
MemoryStats memory_stats();
void reset_peak_memory_stats();
void* allocate_pinned(std::size_t bytes);
bool is_pinned(const void* pointer);
void release_pinned(void* pointer) noexcept;
std::uint64_t kernel_submissions() noexcept;
std::uint64_t tiled_matmul_submissions() noexcept;
std::uint64_t blocked_matmul_submissions() noexcept;

}  // namespace iris
