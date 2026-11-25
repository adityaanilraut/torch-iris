#include "opencl_runtime.h"
#include "rng_internal.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace iris {
namespace {

void check_iris_fp32(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1 &&
                t.layout() == c10::kStrided && t.scalar_type() == at::kFloat,
                "Iris ", name, " requires strided float32 tensors on iris:0");
}

void check_iris_fp(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1 &&
                t.layout() == c10::kStrided &&
                (t.scalar_type() == at::kFloat || (has_fp16() && t.scalar_type() == at::kHalf)),
                "Iris ", name, " requires strided float32 or float16 tensors on iris:0");
}

void check_iris_dev(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1 &&
                t.layout() == c10::kStrided,
                "Iris ", name, " requires strided tensors on iris:0");
}

cl_mem buf(const at::Tensor& t) {
    if (!t.defined()) return nullptr;
    return static_cast<cl_mem>(t.storage().data_ptr().get());
}

BufferSlice bslice(const at::Tensor& t) {
    if (!t.defined()) return {};
    return {static_cast<cl_mem>(t.storage().data_ptr().get()),
            static_cast<cl_ulong>(t.storage_offset())};
}

ValueType to_vtype(at::ScalarType t) {
    switch (t) {
        case at::kFloat: return ValueType::Float;
        case at::kHalf:  return ValueType::Half;
        case at::kLong:  return ValueType::Long;
        case at::kInt:   return ValueType::Int;
        case at::kByte:  return ValueType::Byte;
        case at::kBool:  return ValueType::Bool;
        default: TORCH_CHECK(false, "Unsupported Iris scalar type ", t);
    }
}

at::Tensor packed0(const at::Tensor& t) {
    if (!t.defined()) return t;
    if (t.is_contiguous() && t.storage_offset() == 0) return t;
    return t.clone(at::MemoryFormat::Contiguous);
}

at::Tensor opt_contig_fp(const std::optional<at::Tensor>& opt, const char* name) {
    if (!opt || !opt->defined()) return {};
    check_iris_fp(*opt, name);
    return packed0(*opt);
}

// ---------- BatchNorm ----------

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_impl(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {
    check_iris_fp(input, "batch_norm");
    TORCH_CHECK(input.dim() >= 2, "batch_norm expects at least 2D input");
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t S = input.numel() / std::max<int64_t>(1, N * C);
    const auto x = packed0(input);
    const auto w = opt_contig_fp(weight, "batch_norm");
    const auto b = opt_contig_fp(bias, "batch_norm");
    const bool in_fp16 = input.scalar_type() == at::kHalf;
    const bool param_fp16 = w.defined() ? (w.scalar_type() == at::kHalf)
                          : (b.defined() ? (b.scalar_type() == at::kHalf) : in_fp16);
    TORCH_CHECK(!w.defined() || !b.defined() || w.scalar_type() == b.scalar_type(),
                "Iris batch_norm weight and bias dtypes must match");
    auto output = at::empty(input.sizes(), input.options());

    if (training) {
        const bool has_running = running_mean && running_mean->defined() && running_var && running_var->defined();
        bool stat_fp16 = false;
        at::Tensor rm_c, rv_c;
        bool rm_copyback = false, rv_copyback = false;
        if (has_running) {
            check_iris_fp(*running_mean, "batch_norm");
            check_iris_fp(*running_var, "batch_norm");
            TORCH_CHECK(running_mean->scalar_type() == running_var->scalar_type(),
                        "Iris batch_norm running_mean and running_var dtypes must match");
            stat_fp16 = running_mean->scalar_type() == at::kHalf;
            rm_c = running_mean->is_contiguous() && running_mean->storage_offset() == 0
                ? *running_mean : (rm_copyback = true, running_mean->clone(at::MemoryFormat::Contiguous));
            rv_c = running_var->is_contiguous() && running_var->storage_offset() == 0
                ? *running_var : (rv_copyback = true, running_var->clone(at::MemoryFormat::Contiguous));
        }
        const auto stat_dtype = stat_fp16 ? at::kHalf : at::kFloat;
        auto save_mean = at::empty({C}, input.options().dtype(stat_dtype));
        auto save_invstd = at::empty({C}, input.options().dtype(stat_dtype));
        if (input.numel()) {
            TORCH_CHECK(N * S > 1, "Expected more than 1 value per channel when training");
            batch_norm_forward(buf(output), buf(save_mean), buf(save_invstd),
                               buf(rm_c), buf(rv_c), buf(x), buf(w), buf(b),
                               static_cast<cl_int>(N), static_cast<cl_int>(C), static_cast<cl_int>(S),
                               true, static_cast<float>(momentum), static_cast<float>(eps),
                               in_fp16, param_fp16, stat_fp16);
            if (rm_copyback) const_cast<at::Tensor&>(*running_mean).copy_(rm_c);
            if (rv_copyback) const_cast<at::Tensor&>(*running_var).copy_(rv_c);
        }
        return {output, save_mean, save_invstd};
    } else {
        TORCH_CHECK(running_mean && running_mean->defined() && running_var && running_var->defined(),
                    "batch_norm eval requires running_mean and running_var");
        check_iris_fp(*running_mean, "batch_norm");
        check_iris_fp(*running_var, "batch_norm");
        TORCH_CHECK(running_mean->scalar_type() == running_var->scalar_type(),
                    "Iris batch_norm running_mean and running_var dtypes must match");
        const bool stat_fp16 = running_mean->scalar_type() == at::kHalf;
        const auto rm = packed0(*running_mean);
        const auto rv = packed0(*running_var);
        const auto stat_dtype = stat_fp16 ? at::kHalf : at::kFloat;
        auto save_mean = at::empty({0}, input.options().dtype(stat_dtype));
        auto save_invstd = at::empty({0}, input.options().dtype(stat_dtype));
        if (input.numel()) {
            batch_norm_forward(buf(output), nullptr, nullptr, buf(rm), buf(rv),
                               buf(x), buf(w), buf(b),
                               static_cast<cl_int>(N), static_cast<cl_int>(C), static_cast<cl_int>(S),
                               false, static_cast<float>(momentum), static_cast<float>(eps),
                               in_fp16, param_fp16, stat_fp16);
        }
        return {output, save_mean, save_invstd};
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_legit(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, at::Tensor& running_mean,
    at::Tensor& running_var, bool training, double momentum, double eps) {
    return native_batch_norm_impl(input, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_legit_no_stats(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, bool training, double momentum, double eps) {
    return native_batch_norm_impl(input, weight, bias, std::nullopt, std::nullopt, training, momentum, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward_impl(
    const at::Tensor& grad_out, const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& running_mean, const std::optional<at::Tensor>& running_var,
    const std::optional<at::Tensor>& save_mean, const std::optional<at::Tensor>& save_invstd,
    bool train, double eps, std::array<bool, 3> output_mask) {
    check_iris_fp(grad_out, "batch_norm_backward");
    check_iris_fp(input, "batch_norm_backward");
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t S = input.numel() / std::max<int64_t>(1, N * C);
    const auto g = packed0(grad_out);
    const auto x = packed0(input);
    const auto w = opt_contig_fp(weight, "batch_norm_backward");
    const bool in_fp16 = input.scalar_type() == at::kHalf;
    const bool param_fp16 = w.defined() ? (w.scalar_type() == at::kHalf) : in_fp16;
    const auto param_dtype = param_fp16 ? at::kHalf : at::kFloat;
    at::Tensor mean_t, var_or_invstd_t;
    if (train) {
        TORCH_CHECK(save_mean && save_mean->defined() && save_invstd && save_invstd->defined(),
                    "batch_norm_backward train requires save_mean and save_invstd");
        mean_t = packed0(*save_mean);
        var_or_invstd_t = packed0(*save_invstd);
    } else {
        TORCH_CHECK(running_mean && running_mean->defined() && running_var && running_var->defined(),
                    "batch_norm_backward eval requires running_mean and running_var");
        mean_t = packed0(*running_mean);
        var_or_invstd_t = packed0(*running_var);
    }
    const bool stat_fp16 = mean_t.scalar_type() == at::kHalf;
    at::Tensor dx, dw, db;
    if (output_mask[0]) dx = at::empty(input.sizes(), input.options());
    if (output_mask[1]) dw = at::empty({C}, input.options().dtype(param_dtype));
    if (output_mask[2]) db = at::empty({C}, input.options().dtype(param_dtype));
    if (input.numel()) {
        at::Tensor sum_dy, sum_dy_xmu;
        if (output_mask[0] && train) {
            sum_dy = at::empty({C}, input.options().dtype(at::kFloat));
            sum_dy_xmu = at::empty({C}, input.options().dtype(at::kFloat));
        }
        batch_norm_backward(buf(dx), buf(dw), buf(db), buf(sum_dy), buf(sum_dy_xmu),
                            buf(g), buf(x), buf(w), buf(mean_t), buf(var_or_invstd_t),
                            static_cast<cl_int>(N), static_cast<cl_int>(C), static_cast<cl_int>(S),
                            static_cast<float>(eps), train, in_fp16, param_fp16, stat_fp16);
    } else {
        if (dw.defined()) dw.zero_();
        if (db.defined()) db.zero_();
    }
    return {dx, dw, db};
}

// ---------- GroupNorm ----------

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_impl(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, int64_t N, int64_t C, int64_t HxW,
    int64_t group, double eps) {
    check_iris_fp(input, "group_norm");
    TORCH_CHECK(group > 0 && C % group == 0, "group_norm channels must be divisible by groups");
    const auto x = packed0(input);
    const auto w = opt_contig_fp(weight, "group_norm");
    const auto b = opt_contig_fp(bias, "group_norm");
    const bool in_fp16 = input.scalar_type() == at::kHalf;
    const bool param_fp16 = w.defined() ? (w.scalar_type() == at::kHalf)
                          : (b.defined() ? (b.scalar_type() == at::kHalf) : in_fp16);
    auto output = at::empty(input.sizes(), input.options());
    auto mean = at::empty({N, group}, input.options());
    auto rstd = at::empty({N, group}, input.options());
    if (input.numel()) {
        group_norm_forward(buf(output), buf(mean), buf(rstd), buf(x), buf(w), buf(b),
                           static_cast<cl_int>(N), static_cast<cl_int>(C),
                           static_cast<cl_int>(HxW), static_cast<cl_int>(group),
                           static_cast<float>(eps), in_fp16, param_fp16);
    }
    return {output, mean, rstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward_impl(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight,
    int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool, 3> output_mask) {
    check_iris_fp(grad_out, "group_norm_backward");
    check_iris_fp(input, "group_norm_backward");
    const auto g = packed0(grad_out);
    const auto x = packed0(input);
    const auto m = packed0(mean);
    const auto r = packed0(rstd);
    const auto w = opt_contig_fp(weight, "group_norm_backward");
    const bool in_fp16 = input.scalar_type() == at::kHalf;
    const bool param_fp16 = w.defined() ? (w.scalar_type() == at::kHalf) : in_fp16;
    const auto param_dtype = param_fp16 ? at::kHalf : at::kFloat;
    at::Tensor dx, dw, db;
    if (output_mask[0]) dx = at::empty(input.sizes(), input.options());
    if (output_mask[1]) dw = at::empty({C}, input.options().dtype(param_dtype));
    if (output_mask[2]) db = at::empty({C}, input.options().dtype(param_dtype));
    if (input.numel()) {
        group_norm_backward(buf(dx), buf(dw), buf(db), buf(g), buf(x), buf(m), buf(r), buf(w),
                            static_cast<cl_int>(N), static_cast<cl_int>(C),
                            static_cast<cl_int>(HxW), static_cast<cl_int>(group),
                            in_fp16, param_fp16);
    } else {
        if (dw.defined()) dw.zero_();
        if (db.defined()) db.zero_();
    }
    return {dx, dw, db};
}

// ---------- RMSNorm (native FP32 & FP16 fused forward + backward via autograd::Function) ----------

std::tuple<at::Tensor, at::Tensor> fused_rms_norm_impl(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight, std::optional<double> eps) {
    check_iris_fp(input, "rms_norm");
    TORCH_CHECK(!normalized_shape.empty() && normalized_shape.size() <= static_cast<size_t>(input.dim()) &&
                input.sizes().slice(input.dim() - normalized_shape.size()) == normalized_shape,
                "Iris RMSNorm normalized_shape mismatch");
    const auto w = (weight && weight->defined()) ? (check_iris_fp(*weight, "rms_norm"), weight->contiguous()) : at::Tensor{};
    TORCH_CHECK(!w.defined() || w.sizes() == normalized_shape, "Iris RMSNorm weight shape mismatch");
    const auto prefix = input.dim() - static_cast<int64_t>(normalized_shape.size());
    const int64_t rows = c10::multiply_integers(input.sizes().slice(0, prefix));
    const int64_t width = c10::multiply_integers(normalized_shape);
    at::DimVector rstd_shape(input.sizes().begin(), input.sizes().begin() + prefix);
    rstd_shape.insert(rstd_shape.end(), normalized_shape.size(), 1);
    const float eps_f = eps.has_value() ? static_cast<float>(*eps) : std::numeric_limits<float>::epsilon();
    const auto x = input.contiguous();
    const bool in_fp16 = input.scalar_type() == at::kHalf;
    const bool param_fp16 = w.defined() ? (w.scalar_type() == at::kHalf) : in_fp16;
    auto output = at::empty(input.sizes(), input.options());
    auto rstd = at::empty(rstd_shape, input.options().dtype(at::kFloat));
    if (output.numel()) {
        rms_norm_forward(bslice(output), buf(rstd), bslice(x), bslice(w),
                         static_cast<cl_ulong>(rows), static_cast<cl_int>(width), eps_f,
                         in_fp16, param_fp16);
    }
    return {output, rstd};
}

struct IrisRmsNormFunction : public torch::autograd::Function<IrisRmsNormFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
        const std::optional<at::Tensor>& weight_opt,
        int64_t normalized_ndim,
        double eps_val) {
        const auto prefix = input.dim() - normalized_ndim;
        const int64_t rows = c10::multiply_integers(input.sizes().slice(0, prefix));
        const int64_t width = c10::multiply_integers(input.sizes().slice(prefix));
        const auto x = input.contiguous();
        const bool has_w = weight_opt.has_value() && weight_opt->defined();
        const auto w = has_w ? weight_opt->contiguous() : at::Tensor{};
        const bool in_fp16 = x.scalar_type() == at::kHalf;
        const bool param_fp16 = has_w ? (w.scalar_type() == at::kHalf) : in_fp16;
        auto output = at::empty(input.sizes(), input.options());
        auto rstd = at::empty({rows}, input.options().dtype(at::kFloat));
        if (output.numel()) {
            rms_norm_forward(bslice(output), buf(rstd), bslice(x), bslice(w),
                             static_cast<cl_ulong>(rows), static_cast<cl_int>(width),
                             static_cast<float>(eps_val), in_fp16, param_fp16);
        }
        ctx->save_for_backward({x, w, rstd});
        ctx->saved_data["rows"] = rows;
        ctx->saved_data["width"] = width;
        ctx->saved_data["has_weight"] = has_w;
        ctx->saved_data["weight_shape"] = has_w ? weight_opt->sizes().vec() : std::vector<int64_t>{};
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        const auto& x = saved[0];
        const auto& w = saved[1];
        const auto& rstd = saved[2];
        const int64_t rows = ctx->saved_data["rows"].toInt();
        const int64_t width = ctx->saved_data["width"].toInt();
        const bool has_weight = ctx->saved_data["has_weight"].toBool();
        const auto grad = grad_outputs[0].contiguous();
        const bool in_fp16 = x.scalar_type() == at::kHalf;
        const bool param_fp16 = has_weight ? (w.scalar_type() == at::kHalf) : in_fp16;
        at::Tensor dx, dw;
        if (ctx->needs_input_grad(0)) {
            dx = at::empty(x.sizes(), x.options());
        }
        if (has_weight && ctx->needs_input_grad(1)) {
            const auto wshape = ctx->saved_data["weight_shape"].toIntVector();
            dw = at::empty(wshape, w.options());
        }
        if (x.numel()) {
            rms_norm_backward(bslice(dx), bslice(dw), bslice(grad), bslice(x), bslice(w), buf(rstd),
                              static_cast<cl_int>(rows), static_cast<cl_int>(width), in_fp16, param_fp16);
        } else if (dw.defined()) {
            dw.zero_();
        }
        return {dx, dw, at::Tensor{}, at::Tensor{}};
    }
};

at::Tensor rms_norm_autograd(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight, std::optional<double> eps) {
    check_iris_fp(input, "rms_norm");
    TORCH_CHECK(!normalized_shape.empty() && normalized_shape.size() <= static_cast<size_t>(input.dim()) &&
                input.sizes().slice(input.dim() - normalized_shape.size()) == normalized_shape,
                "Iris RMSNorm normalized_shape mismatch");
    const bool has_w = weight.has_value() && weight->defined();
    if (has_w) {
        check_iris_fp(*weight, "rms_norm");
        TORCH_CHECK(weight->sizes() == normalized_shape, "Iris RMSNorm weight shape mismatch");
    }
    const double eps_val = eps.value_or(static_cast<double>(std::numeric_limits<float>::epsilon()));
    const bool needs_grad = at::GradMode::is_enabled() &&
        (input.requires_grad() || (has_w && weight->requires_grad()));
    if (!needs_grad) {
        const auto prefix = input.dim() - static_cast<int64_t>(normalized_shape.size());
        const int64_t rows = c10::multiply_integers(input.sizes().slice(0, prefix));
        const int64_t width = c10::multiply_integers(normalized_shape);
        const auto x = input.contiguous();
        const auto wc = has_w ? weight->contiguous() : at::Tensor{};
        const bool in_fp16 = x.scalar_type() == at::kHalf;
        const bool param_fp16 = has_w ? (wc.scalar_type() == at::kHalf) : in_fp16;
        auto output = at::empty(input.sizes(), input.options());
        if (output.numel()) {
            rms_norm_forward(bslice(output), nullptr, bslice(x), bslice(wc),
                             static_cast<cl_ulong>(rows), static_cast<cl_int>(width),
                             static_cast<float>(eps_val), in_fp16, param_fp16);
        }
        return output;
    }
    return IrisRmsNormFunction::apply(input, has_w ? weight : std::nullopt,
                                      static_cast<int64_t>(normalized_shape.size()), eps_val);
}

// ---------- AdaptiveAvgPool2d ----------

at::Tensor adaptive_avg_pool2d_impl(const at::Tensor& self, at::IntArrayRef output_size) {
    check_iris_fp(self, "adaptive_avg_pool2d");
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "adaptive_avg_pool2d expects 3D or 4D input");
    TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d expects 2D output_size");
    const int64_t oh = output_size[0], ow = output_size[1];
    TORCH_CHECK(oh >= 0 && ow >= 0, "adaptive_avg_pool2d output_size must be non-negative");
    at::DimVector out_shape(self.sizes());
    out_shape[out_shape.size() - 2] = oh;
    out_shape.back() = ow;
    auto output = at::empty(out_shape, self.options());
    if (output.numel()) {
        const auto x = packed0(self);
        const int64_t nc = (self.dim() == 4 ? self.size(0) : 1) * self.size(-3);
        adaptive_avg_pool2d_forward(buf(output), buf(x),
                                    static_cast<cl_int>(nc), static_cast<cl_int>(self.size(-2)),
                                    static_cast<cl_int>(self.size(-1)),
                                    static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                                    self.scalar_type() == at::kHalf);
    }
    return output;
}

at::Tensor adaptive_avg_pool2d_backward_impl(const at::Tensor& grad_output, const at::Tensor& self) {
    check_iris_fp(grad_output, "adaptive_avg_pool2d_backward");
    check_iris_fp(self, "adaptive_avg_pool2d_backward");
    auto grad_in = at::empty(self.sizes(), self.options());
    if (grad_in.numel()) {
        const auto g = packed0(grad_output);
        const int64_t nc = (self.dim() == 4 ? self.size(0) : 1) * self.size(-3);
        adaptive_avg_pool2d_backward(buf(grad_in), buf(g),
                                     static_cast<cl_int>(nc), static_cast<cl_int>(self.size(-2)),
                                     static_cast<cl_int>(self.size(-1)),
                                     static_cast<cl_int>(grad_output.size(-2)),
                                     static_cast<cl_int>(grad_output.size(-1)),
                                     self.scalar_type() == at::kHalf);
    }
    return grad_in;
}

// ---------- Upsample Nearest2d & Bilinear2d ----------

float compute_scale_nearest(int64_t in_size, int64_t out_size, std::optional<double> scale) {
    if (scale.has_value() && *scale > 0.0) {
        return static_cast<float>(1.0 / *scale);
    }
    return static_cast<float>(in_size) / static_cast<float>(out_size);
}

float compute_scale_bilinear(int64_t in_size, int64_t out_size, bool align_corners, std::optional<double> scale) {
    if (align_corners) {
        return out_size > 1 ? static_cast<float>(in_size - 1) / static_cast<float>(out_size - 1) : 0.0f;
    }
    if (scale.has_value() && *scale > 0.0) {
        return static_cast<float>(1.0 / *scale);
    }
    return static_cast<float>(in_size) / static_cast<float>(out_size);
}

at::Tensor upsample_nearest2d_impl(const at::Tensor& self, at::IntArrayRef output_size,
                                   std::optional<double> scales_h, std::optional<double> scales_w) {
    check_iris_fp(self, "upsample_nearest2d");
    TORCH_CHECK(self.dim() == 4 && output_size.size() == 2, "upsample_nearest2d expects 4D input");
    const int64_t n = self.size(0), c = self.size(1), h = self.size(2), w = self.size(3);
    const int64_t oh = output_size[0], ow = output_size[1];
    auto output = at::empty({n, c, oh, ow}, self.options());
    if (output.numel()) {
        const auto x = packed0(self);
        upsample_nearest2d_forward(buf(output), buf(x), static_cast<cl_int>(n * c),
                                   static_cast<cl_int>(h), static_cast<cl_int>(w),
                                   static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                                   compute_scale_nearest(h, oh, scales_h),
                                   compute_scale_nearest(w, ow, scales_w),
                                   self.scalar_type() == at::kHalf);
    }
    return output;
}

at::Tensor upsample_nearest2d_backward_impl(const at::Tensor& grad_output, at::IntArrayRef output_size,
                                            at::IntArrayRef input_size,
                                            std::optional<double> scales_h, std::optional<double> scales_w) {
    check_iris_fp(grad_output, "upsample_nearest2d_backward");
    const int64_t n = input_size[0], c = input_size[1], h = input_size[2], w = input_size[3];
    const int64_t oh = output_size[0], ow = output_size[1];
    auto grad_in = at::empty(input_size, grad_output.options());
    if (grad_in.numel()) {
        const auto g = packed0(grad_output);
        upsample_nearest2d_backward(buf(grad_in), buf(g), static_cast<cl_int>(n * c),
                                    static_cast<cl_int>(h), static_cast<cl_int>(w),
                                    static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                                    compute_scale_nearest(h, oh, scales_h),
                                    compute_scale_nearest(w, ow, scales_w),
                                    grad_output.scalar_type() == at::kHalf);
    }
    return grad_in;
}

at::Tensor upsample_bilinear2d_impl(const at::Tensor& self, at::IntArrayRef output_size,
                                    bool align_corners,
                                    std::optional<double> scales_h, std::optional<double> scales_w) {
    check_iris_fp(self, "upsample_bilinear2d");
    TORCH_CHECK(self.dim() == 4 && output_size.size() == 2, "upsample_bilinear2d expects 4D input");
    const int64_t n = self.size(0), c = self.size(1), h = self.size(2), w = self.size(3);
    const int64_t oh = output_size[0], ow = output_size[1];
    auto output = at::empty({n, c, oh, ow}, self.options());
    if (output.numel()) {
        const auto x = packed0(self);
        upsample_bilinear2d_forward(buf(output), buf(x), static_cast<cl_int>(n * c),
                                    static_cast<cl_int>(h), static_cast<cl_int>(w),
                                    static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                                    compute_scale_bilinear(h, oh, align_corners, scales_h),
                                    compute_scale_bilinear(w, ow, align_corners, scales_w),
                                    align_corners, self.scalar_type() == at::kHalf);
    }
    return output;
}

at::Tensor upsample_bilinear2d_backward_impl(const at::Tensor& grad_output, at::IntArrayRef output_size,
                                             at::IntArrayRef input_size, bool align_corners,
                                             std::optional<double> scales_h, std::optional<double> scales_w) {
    check_iris_fp(grad_output, "upsample_bilinear2d_backward");
    const int64_t n = input_size[0], c = input_size[1], h = input_size[2], w = input_size[3];
    const int64_t oh = output_size[0], ow = output_size[1];
    auto grad_in = at::empty(input_size, grad_output.options());
    if (grad_in.numel()) {
        const auto g = packed0(grad_output);
        upsample_bilinear2d_backward(buf(grad_in), buf(g), static_cast<cl_int>(n * c),
                                     static_cast<cl_int>(h), static_cast<cl_int>(w),
                                     static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                                     compute_scale_bilinear(h, oh, align_corners, scales_h),
                                     compute_scale_bilinear(w, ow, align_corners, scales_w),
                                     align_corners, grad_output.scalar_type() == at::kHalf);
    }
    return grad_in;
}

// ---------- Argmax / Argmin / Max.dim / Min.dim ----------

template <bool is_min>
at::Tensor arg_extrema_impl(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    check_iris_dev(self, is_min ? "argmin" : "argmax");
    const auto type = to_vtype(self.scalar_type());
    if (!dim.has_value()) {
        TORCH_CHECK(self.numel() > 0, is_min ? "argmin" : "argmax", "(): Expected reduction dim for empty tensor");
        at::DimVector out_shape;
        if (keepdim) out_shape.assign(self.dim(), 1);
        auto out = at::empty(out_shape, self.options().dtype(at::kLong));
        const auto x = packed0(self);
        argmax_argmin(buf(out), nullptr, buf(x), 1, static_cast<cl_ulong>(x.numel()), 1, type, is_min, false);
        return out;
    }
    const int64_t d = at::maybe_wrap_dim(*dim, self.dim());
    if (self.dim() == 0) {
        auto out = at::zeros({}, self.options().dtype(at::kLong));
        return out;
    }
    const int64_t dim_size = self.size(d);
    TORCH_CHECK(dim_size > 0, is_min ? "argmin" : "argmax", "(): Expected non-empty reduction dimension");
    at::DimVector out_shape;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (i < d) { outer *= self.size(i); out_shape.push_back(self.size(i)); }
        else if (i > d) { inner *= self.size(i); out_shape.push_back(self.size(i)); }
        else if (keepdim) { out_shape.push_back(1); }
    }
    auto out = at::empty(out_shape, self.options().dtype(at::kLong));
    if (out.numel()) {
        const auto x = packed0(self);
        argmax_argmin(buf(out), nullptr, buf(x),
                      static_cast<cl_ulong>(outer), static_cast<cl_ulong>(dim_size),
                      static_cast<cl_ulong>(inner), type, is_min, false);
    }
    return out;
}

template <bool is_min>
at::Tensor& arg_extrema_out(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim, at::Tensor& out) {
    auto res = arg_extrema_impl<is_min>(self, dim, keepdim);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

template <bool is_min>
std::tuple<at::Tensor, at::Tensor> extrema_dim_impl(const at::Tensor& self, int64_t dim, bool keepdim) {
    check_iris_dev(self, is_min ? "min.dim" : "max.dim");
    const auto type = to_vtype(self.scalar_type());
    const int64_t d = at::maybe_wrap_dim(dim, self.dim());
    if (self.dim() == 0) {
        return {self.clone(), at::zeros({}, self.options().dtype(at::kLong))};
    }
    const int64_t dim_size = self.size(d);
    TORCH_CHECK(dim_size > 0, "Expected non-empty reduction dimension");
    at::DimVector out_shape;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (i < d) { outer *= self.size(i); out_shape.push_back(self.size(i)); }
        else if (i > d) { inner *= self.size(i); out_shape.push_back(self.size(i)); }
        else if (keepdim) { out_shape.push_back(1); }
    }
    auto values = at::empty(out_shape, self.options());
    auto indices = at::empty(out_shape, self.options().dtype(at::kLong));
    if (values.numel()) {
        const auto x = packed0(self);
        argmax_argmin(buf(indices), buf(values), buf(x),
                      static_cast<cl_ulong>(outer), static_cast<cl_ulong>(dim_size),
                      static_cast<cl_ulong>(inner), type, is_min, true);
    }
    return {values, indices};
}

// ---------- Sort & TopK ----------

std::tuple<at::Tensor, at::Tensor> sort_topk_core(
    const at::Tensor& self, int64_t k, int64_t dim, bool descending) {
    check_iris_dev(self, "sort/topk");
    const auto type = to_vtype(self.scalar_type());
    if (self.dim() == 0) {
        TORCH_CHECK(k == 1 || k == 0, "selected index k out of range");
        if (k == 0) {
            return {at::empty({0}, self.options()), at::empty({0}, self.options().dtype(at::kLong))};
        }
        return {self.clone(), at::zeros({}, self.options().dtype(at::kLong))};
    }
    const int64_t d = at::maybe_wrap_dim(dim, self.dim());
    const int64_t width = self.size(d);
    TORCH_CHECK(k >= 0 && k <= width, "selected index k out of range");
    const bool moved = (d != self.dim() - 1);
    const auto x_last = packed0(moved ? self.movedim(d, -1) : self);
    at::DimVector out_last_shape(x_last.sizes());
    out_last_shape.back() = k;
    auto vals_last = at::empty(out_last_shape, self.options());
    auto idxs_last = at::empty(out_last_shape, self.options().dtype(at::kLong));
    const int64_t rows = x_last.numel() / std::max<int64_t>(1, width);
    if (rows > 0 && k > 0) {
        if (width == 1) {
            vals_last.copy_(x_last);
            idxs_last.zero_();
        } else if (self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf) {
            const bool is_fp16 = (self.scalar_type() == at::kHalf);
            if (width <= 1024 || k <= 256) {
                sort_rows_f32(buf(vals_last), buf(idxs_last), buf(x_last),
                              static_cast<cl_int>(rows), static_cast<cl_int>(width),
                              static_cast<cl_int>(k), descending, is_fp16);
            } else {
                int64_t p = 2;
                while (p < width) p <<= 1;
                auto tmp_v = at::empty({rows, p}, self.options().dtype(at::kFloat));
                auto tmp_i = at::empty({rows, p}, self.options().dtype(at::kInt));
                sort_rows_f32(buf(vals_last), buf(idxs_last), buf(x_last),
                              static_cast<cl_int>(rows), static_cast<cl_int>(width),
                              static_cast<cl_int>(k), descending, is_fp16,
                              buf(tmp_v), buf(tmp_i));
            }
        } else {
            if (width <= 1024) {
                sort_rows_int(buf(vals_last), buf(idxs_last), buf(x_last),
                              static_cast<cl_int>(rows), static_cast<cl_int>(width),
                              static_cast<cl_int>(k), descending, type);
            } else {
                int64_t p = 2;
                while (p < width) p <<= 1;
                auto tmp_v = at::empty({rows, p}, self.options().dtype(at::kLong));
                auto tmp_i = at::empty({rows, p}, self.options().dtype(at::kInt));
                sort_rows_int(buf(vals_last), buf(idxs_last), buf(x_last),
                              static_cast<cl_int>(rows), static_cast<cl_int>(width),
                              static_cast<cl_int>(k), descending, type,
                              buf(tmp_v), buf(tmp_i));
            }
        }
    }
    if (moved) {
        return {vals_last.movedim(-1, d).contiguous(), idxs_last.movedim(-1, d).contiguous()};
    }
    return {vals_last, idxs_last};
}

std::tuple<at::Tensor, at::Tensor> topk_impl(
    const at::Tensor& self, int64_t k, int64_t dim, bool largest, bool /*sorted*/) {
    return sort_topk_core(self, k, dim, largest);
}

std::tuple<at::Tensor&, at::Tensor&> topk_values(
    const at::Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted,
    at::Tensor& values, at::Tensor& indices) {
    auto [v, i] = topk_impl(self, k, dim, largest, sorted);
    at::native::resize_output(values, v.sizes());
    at::native::resize_output(indices, i.sizes());
    values.copy_(v);
    indices.copy_(i);
    return {values, indices};
}

std::tuple<at::Tensor, at::Tensor> sort_impl(const at::Tensor& self, int64_t dim, bool descending) {
    const int64_t k = self.dim() == 0 ? 1 : self.size(at::maybe_wrap_dim(dim, self.dim()));
    return sort_topk_core(self, k, dim, descending);
}

std::tuple<at::Tensor, at::Tensor> sort_stable_impl(
    const at::Tensor& self, std::optional<bool> /*stable*/, int64_t dim, bool descending) {
    return sort_impl(self, dim, descending);
}

std::tuple<at::Tensor&, at::Tensor&> sort_values_impl(
    const at::Tensor& self, int64_t dim, bool descending, at::Tensor& values, at::Tensor& indices) {
    auto [v, i] = sort_impl(self, dim, descending);
    at::native::resize_output(values, v.sizes());
    at::native::resize_output(indices, i.sizes());
    values.copy_(v);
    indices.copy_(i);
    return {values, indices};
}

std::tuple<at::Tensor&, at::Tensor&> sort_values_stable_impl(
    const at::Tensor& self, std::optional<bool> /*stable*/, int64_t dim, bool descending,
    at::Tensor& values, at::Tensor& indices) {
    return sort_values_impl(self, dim, descending, values, indices);
}

// ---------- Cumsum ----------

at::Tensor cumsum_impl(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype) {
    check_iris_dev(self, "cumsum");
    const auto in_type = to_vtype(self.scalar_type());
    auto target_dtype = dtype.value_or(
        (self.scalar_type() == at::kBool || self.scalar_type() == at::kByte || self.scalar_type() == at::kInt)
            ? at::kLong : self.scalar_type());
    const auto out_type = to_vtype(target_dtype);
    const int64_t d = at::maybe_wrap_dim(dim, self.dim());
    if (self.dim() == 0) {
        return self.to(target_dtype).clone();
    }
    auto output = at::empty(self.sizes(), self.options().dtype(target_dtype));
    if (output.numel()) {
        const auto x = packed0(self);
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < d; ++i) outer *= self.size(i);
        for (int64_t i = d + 1; i < self.dim(); ++i) inner *= self.size(i);
        cumsum_launch(buf(output), buf(x),
                      static_cast<cl_ulong>(outer), static_cast<cl_ulong>(self.size(d)),
                      static_cast<cl_ulong>(inner), in_type, out_type);
    }
    return output;
}

at::Tensor& cumsum_out(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype, at::Tensor& out) {
    auto res = cumsum_impl(self, dim, dtype.value_or(out.scalar_type()));
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

// ---------- Arange / Linspace / Random_ ----------

at::Tensor& arange_start_out(const at::Scalar& start, const at::Scalar& end,
                             const at::Scalar& step, at::Tensor& out) {
    check_iris_dev(out, "arange");
    const auto type = to_vtype(out.scalar_type());
    const bool is_fp = (out.scalar_type() == at::kFloat || out.scalar_type() == at::kHalf);
    int64_t size = 0;
    if (is_fp) {
        const double s = start.toDouble(), e = end.toDouble(), st = step.toDouble();
        TORCH_CHECK(st != 0.0, "step must be nonzero");
        TORCH_CHECK((st > 0 && e >= s) || (st < 0 && e <= s), "upper bound and larger bound inconsistent with step sign");
        size = static_cast<int64_t>(std::ceil((e - s) / st));
    } else {
        const int64_t s = start.toLong(), e = end.toLong(), st = step.toLong();
        TORCH_CHECK(st != 0, "step must be nonzero");
        TORCH_CHECK((st > 0 && e >= s) || (st < 0 && e <= s), "upper bound and larger bound inconsistent with step sign");
        const int64_t diff = e - s;
        size = (diff + st + (st > 0 ? -1 : 1)) / st;
    }
    at::native::resize_output(out, {size});
    if (size > 0) {
        if (out.is_contiguous() && out.storage_offset() == 0) {
            arange_launch(buf(out), static_cast<cl_ulong>(size),
                          static_cast<float>(start.toDouble()), static_cast<float>(step.toDouble()),
                          static_cast<cl_long>(start.toLong()), static_cast<cl_long>(step.toLong()), type);
        } else {
            auto tmp = at::empty({size}, out.options());
            arange_launch(buf(tmp), static_cast<cl_ulong>(size),
                          static_cast<float>(start.toDouble()), static_cast<float>(step.toDouble()),
                          static_cast<cl_long>(start.toLong()), static_cast<cl_long>(step.toLong()), type);
            out.copy_(tmp);
        }
    }
    return out;
}

at::Tensor& arange_out(const at::Scalar& end, at::Tensor& out) {
    return arange_start_out(0, end, 1, out);
}

at::Tensor& linspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps, at::Tensor& out) {
    check_iris_dev(out, "linspace");
    TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
    const auto type = to_vtype(out.scalar_type());
    at::native::resize_output(out, {steps});
    if (steps > 0) {
        if (out.is_contiguous() && out.storage_offset() == 0) {
            linspace_launch(buf(out), static_cast<cl_ulong>(steps),
                            static_cast<float>(start.toDouble()), static_cast<float>(end.toDouble()), type);
        } else {
            auto tmp = at::empty({steps}, out.options());
            linspace_launch(buf(tmp), static_cast<cl_ulong>(steps),
                            static_cast<float>(start.toDouble()), static_cast<float>(end.toDouble()), type);
            out.copy_(tmp);
        }
    }
    return out;
}

at::Tensor& random_from_impl(at::Tensor& self, int64_t from, std::optional<int64_t> to,
                             std::optional<at::Generator> generator) {
    check_iris_dev(self, "random_");
    at::assert_no_internal_overlap(self);
    const auto type = to_vtype(self.scalar_type());
    int64_t upper = 0;
    if (to.has_value()) {
        upper = *to;
    } else {
        switch (self.scalar_type()) {
            case at::kBool:  upper = 2; break;
            case at::kByte:  upper = 256; break;
            case at::kInt:   upper = std::numeric_limits<int32_t>::max(); break;
            case at::kLong:  upper = std::numeric_limits<int64_t>::max(); break;
            case at::kFloat: upper = (1LL << 24) + 1; break;
            case at::kHalf:  upper = (1LL << 11) + 1; break;
            default: TORCH_CHECK(false, "Unsupported dtype for Iris random_");
        }
    }
    TORCH_CHECK(from < upper, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", upper);
    if (!self.numel()) return self;
    c10::SmallVector<cl_ulong, 16> geometry;
    for (int64_t d = 0; d < self.dim(); ++d) {
        geometry.append({static_cast<cl_ulong>(self.size(d)), static_cast<cl_ulong>(self.stride(d))});
    }
    const auto blocks = (static_cast<std::uint64_t>(self.numel()) - 1) / 4 + 1;
    const auto [seed, start] = reserve_rng_blocks_opt(blocks, generator);
    const uint64_t range = static_cast<uint64_t>(upper) - static_cast<uint64_t>(from);
    random_int_fill(buf(self), static_cast<cl_ulong>(self.numel()),
                    static_cast<cl_ulong>(self.storage_offset()), geometry,
                    seed, start, static_cast<cl_long>(from), static_cast<cl_ulong>(range), type);
    return self;
}

at::Tensor& random_to_impl(at::Tensor& self, int64_t to, std::optional<at::Generator> generator) {
    return random_from_impl(self, 0, to, generator);
}

at::Tensor& random_default_impl(at::Tensor& self, std::optional<at::Generator> generator) {
    return random_from_impl(self, 0, std::nullopt, generator);
}

// ---------- Flip ----------

at::Tensor flip_impl(const at::Tensor& self, at::IntArrayRef dims) {
    check_iris_dev(self, "flip");
    const auto type = to_vtype(self.scalar_type());
    TORCH_CHECK(self.dim() <= 32, "Iris flip supports up to 32 dimensions");
    cl_uint mask = 0;
    for (auto d : dims) {
        d = at::maybe_wrap_dim(d, self.dim());
        TORCH_CHECK(!((mask >> d) & 1u), "dim ", d, " appears multiple times in flip");
        mask |= (1u << d);
    }
    auto output = at::empty(self.sizes(), self.options());
    if (output.numel()) {
        c10::SmallVector<cl_ulong, 16> geometry;
        for (int64_t d = 0; d < self.dim(); ++d) {
            geometry.append({static_cast<cl_ulong>(self.size(d)), static_cast<cl_ulong>(self.stride(d))});
        }
        flip_launch(buf(output), buf(self), static_cast<cl_ulong>(output.numel()),
                    static_cast<cl_ulong>(self.storage_offset()), geometry, mask, type);
    }
    return output;
}

// ---------- Recurrent cell gates and gradients ----------

at::Tensor recurrent_bias(const std::optional<at::Tensor>& bias, const at::Tensor& gates) {
    if (!bias || !bias->defined()) return {};
    check_iris_fp(*bias, "recurrent cell bias");
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == gates.size(1) &&
                bias->scalar_type() == gates.scalar_type(), "Invalid Iris recurrent cell bias");
    return bias->contiguous();
}

void check_cell(const at::Tensor& input, const at::Tensor& hidden, const at::Tensor& state,
                int64_t gate_count, const char* op) {
    check_iris_fp(input, op);
    check_iris_fp(hidden, op);
    check_iris_fp(state, op);
    TORCH_CHECK(input.scalar_type() == state.scalar_type() && hidden.scalar_type() == state.scalar_type() &&
                state.dim() == 2 && input.dim() == 2 && hidden.sizes() == input.sizes() &&
                input.size(0) == state.size(0) && input.size(1) == state.size(1) * gate_count,
                "Invalid Iris ", op, " gate or state shapes");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> recurrent_cell(
    const at::Tensor& input, const at::Tensor& hidden, const at::Tensor& state,
    const std::optional<at::Tensor>& input_bias, const std::optional<at::Tensor>& hidden_bias, bool lstm) {
    check_cell(input, hidden, state, lstm ? 4 : 3, lstm ? "LSTM" : "GRU");
    const auto ib = recurrent_bias(input_bias, input);
    const auto hb = recurrent_bias(hidden_bias, hidden);
    auto output = at::empty(state.sizes(), state.options());
    auto new_cell = lstm ? at::empty_like(output) : at::Tensor{};
    const auto width = state.size(1);
    auto workspace = at::empty({state.size(0), width * (lstm ? 4 : 5)}, state.options());
    if (output.numel()) {
        recurrent_forward(bslice(output), bslice(new_cell), bslice(workspace),
                          bslice(input.contiguous()), bslice(hidden.contiguous()), bslice(state.contiguous()),
                          bslice(ib), bslice(hb), output.numel(), width, lstm, state.scalar_type() == at::kHalf);
    }
    return {output, new_cell, workspace};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_cell(
    const at::Tensor& input, const at::Tensor& hidden, const at::Tensor& cx,
    const std::optional<at::Tensor>& input_bias, const std::optional<at::Tensor>& hidden_bias) {
    return recurrent_cell(input, hidden, cx, input_bias, hidden_bias, true);
}

void check_recurrent_grad(const at::Tensor& grad, const at::Tensor& state) {
    if (!grad.defined()) return;
    check_iris_fp(grad, "recurrent cell backward");
    TORCH_CHECK(grad.sizes() == state.sizes() && grad.scalar_type() == state.scalar_type(),
                "Invalid Iris recurrent cell gradient shape or dtype");
}

void check_recurrent_workspace(const at::Tensor& workspace, const at::Tensor& state, int64_t gates) {
    check_iris_fp(state, "recurrent cell backward");
    check_iris_fp(workspace, "recurrent cell workspace");
    TORCH_CHECK(state.dim() == 2 && workspace.dim() == 2 && workspace.size(0) == state.size(0) &&
                workspace.size(1) == gates * state.size(1) && workspace.scalar_type() == state.scalar_type(),
                "Invalid Iris recurrent cell workspace shape or dtype");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_cell_backward(
    const std::optional<at::Tensor>& grad_hy, const std::optional<at::Tensor>& grad_cy,
    const at::Tensor& cx, const at::Tensor& cy, const at::Tensor& workspace, bool has_bias) {
    const auto gh = grad_hy && grad_hy->defined() ? grad_hy->contiguous() : at::Tensor{};
    const auto gc = grad_cy && grad_cy->defined() ? grad_cy->contiguous() : at::Tensor{};
    if (!gh.defined() && !gc.defined()) return {};
    check_recurrent_workspace(workspace, cx, 4);
    TORCH_CHECK(cy.defined(), "Iris LSTM backward requires a cell state");
    check_recurrent_grad(cy, cx);
    check_recurrent_grad(gh, cx);
    check_recurrent_grad(gc, cx);
    const auto width = cx.size(1);
    auto gates = at::empty({cx.size(0), width * 4}, cx.options());
    auto grad_state = at::empty(cx.sizes(), cx.options());
    if (cx.numel()) {
        recurrent_backward(bslice(gates), {}, bslice(grad_state), bslice(gh), bslice(gc),
                           bslice(cx.contiguous()), bslice(cy.contiguous()), bslice(workspace.contiguous()),
                           cx.numel(), width, true, cx.scalar_type() == at::kHalf);
    }
    return {gates, grad_state, has_bias ? gates.sum({0}) : at::Tensor{}};
}

std::tuple<at::Tensor, at::Tensor> gru_cell(
    const at::Tensor& input, const at::Tensor& hidden, const at::Tensor& hx,
    const std::optional<at::Tensor>& input_bias, const std::optional<at::Tensor>& hidden_bias) {
    auto [output, unused, workspace] = recurrent_cell(input, hidden, hx, input_bias, hidden_bias, false);
    return {output, workspace};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> gru_cell_backward(
    const at::Tensor& grad, const at::Tensor& workspace, bool has_bias) {
    check_recurrent_workspace(workspace, grad, 5);
    const auto width = grad.size(1);
    auto dx = at::empty({grad.size(0), width * 3}, grad.options());
    auto dh = at::empty_like(dx);
    auto grad_state = at::empty(grad.sizes(), grad.options());
    if (grad.numel()) {
        recurrent_backward(bslice(dx), bslice(dh), bslice(grad_state), bslice(grad.contiguous()), {}, {}, {},
                           bslice(workspace.contiguous()), grad.numel(), width, false, grad.scalar_type() == at::kHalf);
    }
    return {dx, dh, grad_state, has_bias ? dx.sum({0}) : at::Tensor{}, has_bias ? dh.sum({0}) : at::Tensor{}};
}

// ---------- Fused Optimizers (_fused_sgd_, _fused_adam_, _fused_adamw_) ----------

void fused_sgd_common(
    at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list,
    double weight_decay, double momentum, double lr_scalar, const at::Tensor* lr_tensor,
    double dampening, bool nesterov, bool maximize, bool is_first_step,
    const std::optional<at::Tensor>& grad_scale, const std::optional<at::Tensor>& found_inf) {
    TORCH_CHECK(params.size() == grads.size(), "_fused_sgd_ params and grads size mismatch");
    const bool has_mom = !momentum_buffer_list.empty();
    TORCH_CHECK(!has_mom || momentum_buffer_list.size() == params.size(),
                "_fused_sgd_ momentum_buffer_list size mismatch");
    at::Tensor gs_t = (grad_scale && grad_scale->defined()) ? packed0(*grad_scale) : at::Tensor{};
    at::Tensor fi_t = (found_inf && found_inf->defined()) ? packed0(*found_inf) : at::Tensor{};
    at::Tensor lr_t;
    cl_mem lr_buf = nullptr;
    float lr_val = static_cast<float>(lr_scalar);
    if (lr_tensor && lr_tensor->defined()) {
        if (lr_tensor->device().is_cpu()) {
            lr_val = lr_tensor->item<float>();
        } else {
            lr_t = packed0(*lr_tensor);
            lr_buf = buf(lr_t);
        }
    }
    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        const auto& g = grads[i];
        check_iris_fp32(p, "_fused_sgd_");
        check_iris_fp32(g, "_fused_sgd_");
        if (!p.numel()) continue;
        const bool p_contig = p.is_contiguous();
        auto p_work = p_contig ? p : p.contiguous();
        const auto g_work = g.contiguous();
        at::Tensor m_work;
        bool m_copyback = false;
        if (has_mom) {
            auto m = momentum_buffer_list[i];
            check_iris_fp32(m, "_fused_sgd_");
            if (m.is_contiguous()) {
                m_work = m;
            } else {
                m_work = m.contiguous();
                m_copyback = true;
            }
        }
        fused_sgd_step(buf(p_work), buf(g_work), buf(m_work), lr_buf, buf(gs_t), buf(fi_t),
                       static_cast<cl_ulong>(p_work.storage_offset()),
                       static_cast<cl_ulong>(g_work.storage_offset()),
                       m_work.defined() ? static_cast<cl_ulong>(m_work.storage_offset()) : 0,
                       static_cast<cl_ulong>(p_work.numel()),
                       static_cast<float>(weight_decay), static_cast<float>(momentum),
                       lr_val, static_cast<float>(dampening), nesterov, maximize, is_first_step);
        if (!p_contig) p.copy_(p_work);
        if (m_copyback) momentum_buffer_list[i].copy_(m_work);
    }
}

void fused_sgd_scalar_lr(
    at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list,
    double weight_decay, double momentum, double lr, double dampening,
    bool nesterov, bool maximize, bool is_first_step,
    const std::optional<at::Tensor>& grad_scale, const std::optional<at::Tensor>& found_inf) {
    fused_sgd_common(params, grads, momentum_buffer_list, weight_decay, momentum, lr, nullptr,
                     dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
}

void fused_sgd_tensor_lr(
    at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list,
    double weight_decay, double momentum, const at::Tensor& lr, double dampening,
    bool nesterov, bool maximize, bool is_first_step,
    const std::optional<at::Tensor>& grad_scale, const std::optional<at::Tensor>& found_inf) {
    fused_sgd_common(params, grads, momentum_buffer_list, weight_decay, momentum, 0.0, &lr,
                     dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
}

void fused_adam_common(
    at::TensorList params, at::TensorList grads, at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
    double lr_scalar, const at::Tensor* lr_tensor,
    double beta1, double beta2, double weight_decay, double eps,
    bool amsgrad, bool maximize, bool decoupled_wd,
    const std::optional<at::Tensor>& grad_scale, const std::optional<at::Tensor>& found_inf) {
    const size_t n = params.size();
    TORCH_CHECK(grads.size() == n && exp_avgs.size() == n && exp_avg_sqs.size() == n && state_steps.size() == n,
                "_fused_adam_ list sizes mismatch");
    TORCH_CHECK(!amsgrad || max_exp_avg_sqs.size() == n, "_fused_adam_ max_exp_avg_sqs size mismatch");
    at::Tensor gs_t = (grad_scale && grad_scale->defined()) ? packed0(*grad_scale) : at::Tensor{};
    at::Tensor fi_t = (found_inf && found_inf->defined()) ? packed0(*found_inf) : at::Tensor{};
    at::Tensor lr_t;
    cl_mem lr_buf = nullptr;
    float lr_val = static_cast<float>(lr_scalar);
    if (lr_tensor && lr_tensor->defined()) {
        if (lr_tensor->device().is_cpu()) {
            lr_val = lr_tensor->item<float>();
        } else {
            lr_t = packed0(*lr_tensor);
            lr_buf = buf(lr_t);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        auto p = params[i];
        const auto& g = grads[i];
        auto ea = exp_avgs[i];
        auto eas = exp_avg_sqs[i];
        check_iris_fp32(p, "_fused_adam_");
        check_iris_fp32(g, "_fused_adam_");
        check_iris_fp32(ea, "_fused_adam_");
        check_iris_fp32(eas, "_fused_adam_");
        if (!p.numel()) continue;

        at::Tensor step_t = state_steps[i];
        if (step_t.device().is_cpu()) {
            step_t = step_t.to(p.device(), at::kFloat);
        } else if (step_t.scalar_type() != at::kFloat) {
            step_t = step_t.to(at::kFloat);
        }
        step_t = step_t.contiguous();

        const bool p_c = p.is_contiguous();
        const bool ea_c = ea.is_contiguous();
        const bool eas_c = eas.is_contiguous();
        auto p_w = p_c ? p : p.contiguous();
        const auto g_w = g.contiguous();
        auto ea_w = ea_c ? ea : ea.contiguous();
        auto eas_w = eas_c ? eas : eas.contiguous();
        at::Tensor meas_w;
        bool meas_copyback = false;
        if (amsgrad) {
            auto meas = max_exp_avg_sqs[i];
            check_iris_fp32(meas, "_fused_adam_");
            if (meas.is_contiguous()) meas_w = meas;
            else { meas_w = meas.contiguous(); meas_copyback = true; }
        }
        fused_adam_step(buf(p_w), buf(g_w), buf(ea_w), buf(eas_w), buf(meas_w),
                        buf(step_t), lr_buf, buf(gs_t), buf(fi_t),
                        static_cast<cl_ulong>(p_w.storage_offset()),
                        static_cast<cl_ulong>(g_w.storage_offset()),
                        static_cast<cl_ulong>(ea_w.storage_offset()),
                        static_cast<cl_ulong>(eas_w.storage_offset()),
                        meas_w.defined() ? static_cast<cl_ulong>(meas_w.storage_offset()) : 0,
                        static_cast<cl_ulong>(step_t.storage_offset()),
                        static_cast<cl_ulong>(p_w.numel()),
                        lr_val, static_cast<float>(beta1), static_cast<float>(beta2),
                        static_cast<float>(weight_decay), static_cast<float>(eps),
                        amsgrad, maximize, decoupled_wd);
        if (!p_c) p.copy_(p_w);
        if (!ea_c) ea.copy_(ea_w);
        if (!eas_c) eas.copy_(eas_w);
        if (meas_copyback) max_exp_avg_sqs[i].copy_(meas_w);
    }
}

template <bool decoupled_wd>
void fused_adam_scalar_lr(
    at::TensorList params, at::TensorList grads, at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
    double lr, double beta1, double beta2, double weight_decay, double eps,
    bool amsgrad, bool maximize,
    const std::optional<at::Tensor>& grad_scale, const std::optional<at::Tensor>& found_inf) {
    fused_adam_common(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
                      lr, nullptr, beta1, beta2, weight_decay, eps, amsgrad, maximize, decoupled_wd,
                      grad_scale, found_inf);
}

template <bool decoupled_wd>
void fused_adam_tensor_lr(
    at::TensorList params, at::TensorList grads, at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
    const at::Tensor& lr, double beta1, double beta2, double weight_decay, double eps,
    bool amsgrad, bool maximize,
    const std::optional<at::Tensor>& grad_scale, const std::optional<at::Tensor>& found_inf) {
    fused_adam_common(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
                      0.0, &lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, decoupled_wd,
                      grad_scale, found_inf);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_thnn_fused_lstm_cell", &lstm_cell);
    m.impl("_thnn_fused_lstm_cell_backward_impl", &lstm_cell_backward);
    m.impl("_thnn_fused_gru_cell", &gru_cell);
    m.impl("_thnn_fused_gru_cell_backward", &gru_cell_backward);
    m.impl("native_batch_norm", &native_batch_norm_impl);
    m.impl("_native_batch_norm_legit", &native_batch_norm_legit);
    m.impl("_native_batch_norm_legit.no_stats", &native_batch_norm_legit_no_stats);
    m.impl("native_batch_norm_backward", &native_batch_norm_backward_impl);
    m.impl("native_group_norm", &native_group_norm_impl);
    m.impl("native_group_norm_backward", &native_group_norm_backward_impl);
    m.impl("_fused_rms_norm", &fused_rms_norm_impl);
    m.impl("_adaptive_avg_pool2d", &adaptive_avg_pool2d_impl);
    m.impl("_adaptive_avg_pool2d_backward", &adaptive_avg_pool2d_backward_impl);
    m.impl("upsample_nearest2d", &upsample_nearest2d_impl);
    m.impl("upsample_nearest2d_backward", &upsample_nearest2d_backward_impl);
    m.impl("upsample_bilinear2d", &upsample_bilinear2d_impl);
    m.impl("upsample_bilinear2d_backward", &upsample_bilinear2d_backward_impl);
    m.impl("argmax", &arg_extrema_impl<false>);
    m.impl("argmax.out", &arg_extrema_out<false>);
    m.impl("argmin", &arg_extrema_impl<true>);
    m.impl("argmin.out", &arg_extrema_out<true>);
    m.impl("max.dim", &extrema_dim_impl<false>);
    m.impl("min.dim", &extrema_dim_impl<true>);
    m.impl("topk", &topk_impl);
    m.impl("topk.values", &topk_values);
    m.impl("sort", &sort_impl);
    m.impl("sort.stable", &sort_stable_impl);
    m.impl("sort.values", &sort_values_impl);
    m.impl("sort.values_stable", &sort_values_stable_impl);
    m.impl("cumsum", &cumsum_impl);
    m.impl("cumsum.out", &cumsum_out);
    m.impl("arange.out", &arange_out);
    m.impl("arange.start_out", &arange_start_out);
    m.impl("linspace.out", &linspace_out);
    m.impl("random_.from", &random_from_impl);
    m.impl("random_.to", &random_to_impl);
    m.impl("random_", &random_default_impl);
    m.impl("flip", &flip_impl);
    m.impl("_fused_sgd_", &fused_sgd_scalar_lr);
    m.impl("_fused_sgd_.tensor_lr", &fused_sgd_tensor_lr);
    m.impl("_fused_adam_", &fused_adam_scalar_lr<false>);
    m.impl("_fused_adam_.tensor_lr", &fused_adam_tensor_lr<false>);
    m.impl("_fused_adamw_", &fused_adam_scalar_lr<true>);
    m.impl("_fused_adamw_.tensor_lr", &fused_adam_tensor_lr<true>);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("rms_norm", &rms_norm_autograd);
}

}  // namespace
}  // namespace iris
