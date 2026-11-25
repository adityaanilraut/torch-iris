#include "opencl_runtime.h"

#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/Pool.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sum.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <bit>
#include <cmath>

namespace iris {
namespace {

void check_fp(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                tensor.layout() == c10::kStrided &&
                (tensor.scalar_type() == at::kFloat || (has_fp16() && tensor.scalar_type() == at::kHalf)),
                "Iris convolution, pooling and layer norm require strided float32 or float16 tensors on iris:0");
}

BufferSlice buf_slice(const at::Tensor& tensor) {
    if (!tensor.defined()) {
        return {};
    }
    return {static_cast<cl_mem>(tensor.storage().data_ptr().get()),
            static_cast<cl_ulong>(tensor.storage_offset())};
}

at::Tensor packed_optional(const std::optional<at::Tensor>& tensor) {
    if (!tensor || !tensor->defined()) {
        return {};
    }
    check_fp(*tensor);
    return tensor->contiguous();
}

std::array<int64_t, 2> dimensions(at::IntArrayRef values, const char* name) {
    TORCH_CHECK(values.size() == 1 || values.size() == 2, name, " must have one or two values");
    return {values[0], values.size() == 1 ? values[0] : values[1]};
}

struct Convolution {
    at::DimVector output;
    std::array<cl_ulong, 16> args;
    Conv2dParams params;
};

Convolution convolution_shape(const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride,
                              at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                              at::IntArrayRef output_padding, int64_t groups) {
    check_fp(input);
    check_fp(weight);
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "Iris convolution input and weight dtypes must match");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4,
                "Iris convolution supports 1D (NCL) and 2D (NCHW) tensors");
    const auto [sy, sx] = dimensions(stride, "stride");
    const auto [py, px] = dimensions(padding, "padding");
    const auto [dy, dx] = dimensions(dilation, "dilation");
    const auto [opy, opx] = output_padding.empty() ? std::array<int64_t, 2>{0, 0}
                                                   : dimensions(output_padding, "output_padding");
    const auto n = input.size(0), channels = input.size(1), h = input.size(2), w = input.size(3);
    const auto kh = weight.size(2), kw = weight.size(3);
    TORCH_CHECK(h > 0 && w > 0 && kh > 0 && kw > 0 && sy > 0 && sx > 0 &&
                dy > 0 && dx > 0 && py >= 0 && px >= 0 && opy >= 0 && opx >= 0,
                "Invalid Iris Conv2d geometry");
    int64_t filters = 0;
    int64_t oh = 0, ow = 0;
    if (!transposed) {
        TORCH_CHECK(opy == 0 && opx == 0, "Iris Conv2d requires zero output_padding");
        filters = weight.size(0);
        TORCH_CHECK(groups > 0 && channels > 0 && filters > 0 && filters % groups == 0 &&
                    channels % groups == 0 && weight.size(1) == channels / groups,
                    "Iris Conv2d input and weight channels do not match groups");
        TORCH_CHECK(h + 2 * py >= dy * (kh - 1) + 1 && w + 2 * px >= dx * (kw - 1) + 1,
                    "Kernel size cannot be greater than padded input size");
        oh = (h + 2 * py - dy * (kh - 1) - 1) / sy + 1;
        ow = (w + 2 * px - dx * (kw - 1) - 1) / sx + 1;
    } else {
        filters = weight.size(1) * groups;
        TORCH_CHECK(groups > 0 && channels > 0 && filters > 0 && channels % groups == 0 &&
                    weight.size(0) == channels,
                    "Iris ConvTranspose2d input and weight channels do not match groups");
        TORCH_CHECK(opy < std::max(sy, dy) && opx < std::max(sx, dx),
                    "output padding must be smaller than either stride or dilation");
        oh = (h - 1) * sy - 2 * py + dy * (kh - 1) + opy + 1;
        ow = (w - 1) * sx - 2 * px + dx * (kw - 1) + opx + 1;
        TORCH_CHECK(oh > 0 && ow > 0, "Invalid Iris ConvTranspose2d output size");
    }
    Convolution result{{n, filters, oh, ow}, {}, {
        static_cast<cl_int>(n), static_cast<cl_int>(channels), static_cast<cl_int>(h), static_cast<cl_int>(w),
        static_cast<cl_int>(filters), static_cast<cl_int>(oh), static_cast<cl_int>(ow),
        static_cast<cl_int>(kh), static_cast<cl_int>(kw), static_cast<cl_int>(sy), static_cast<cl_int>(sx),
        static_cast<cl_int>(py), static_cast<cl_int>(px), static_cast<cl_int>(dy), static_cast<cl_int>(dx),
        static_cast<cl_int>(groups)
    }};
    const std::array geom{n, channels, h, w, filters, oh, ow, kh, kw, sy, sx, py, px, dy, dx, groups};
    std::copy(geom.begin(), geom.end(), result.args.begin());
    return result;
}

std::array<int64_t, 3> dimensions3(at::IntArrayRef values, const char* name) {
    TORCH_CHECK(values.size() == 1 || values.size() == 3, name, " must have one or three values");
    if (values.size() == 1) return {values[0], values[0], values[0]};
    return {values[0], values[1], values[2]};
}

at::Tensor conv3d_dispatch(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias,
                           at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                           at::IntArrayRef output_padding, int64_t groups) {
    check_fp(input);
    check_fp(weight);
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "Iris Conv3d input and weight dtypes must match");
    const auto [sd, sh, sw] = dimensions3(stride, "stride");
    const auto [pd, ph, pw] = dimensions3(padding, "padding");
    const auto [dd, dh, dw] = dimensions3(dilation, "dilation");
    const auto [opd, oph, opw] = output_padding.empty() ? std::array<int64_t, 3>{0, 0, 0}
                                                        : dimensions3(output_padding, "output_padding");
    const int64_t n = input.size(0), c_in = input.size(1), id = input.size(2), ih = input.size(3), iw = input.size(4);
    const int64_t kd = weight.size(2), kh = weight.size(3), kw = weight.size(4);
    const bool fp16 = input.scalar_type() == at::kHalf;
    const auto data = input.contiguous();
    const auto kern = weight.contiguous();
    const auto offset = packed_optional(bias);
    if (!transposed) {
        const int64_t c_out = weight.size(0);
        const int64_t od = (id + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
        const int64_t oh = (ih + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        const int64_t ow = (iw + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        TORCH_CHECK(od > 0 && oh > 0 && ow > 0, "Invalid Iris Conv3d output shape");
        auto out = at::empty({n, c_out, od, oh, ow}, input.options());
        if (out.numel()) {
            Conv3dParams p{static_cast<cl_int>(n), static_cast<cl_int>(c_in), static_cast<cl_int>(id),
                           static_cast<cl_int>(ih), static_cast<cl_int>(iw), static_cast<cl_int>(c_out),
                           static_cast<cl_int>(od), static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                           static_cast<cl_int>(kd), static_cast<cl_int>(kh), static_cast<cl_int>(kw),
                           static_cast<cl_int>(sd), static_cast<cl_int>(sh), static_cast<cl_int>(sw),
                           static_cast<cl_int>(pd), static_cast<cl_int>(ph), static_cast<cl_int>(pw),
                           static_cast<cl_int>(dd), static_cast<cl_int>(dh), static_cast<cl_int>(dw),
                           static_cast<cl_int>(groups)};
            conv3d_fwd_launch(buf_slice(out).buffer, buf_slice(data).buffer, buf_slice(kern).buffer,
                              buf_slice(offset).buffer, p, fp16);
        }
        return out;
    } else {
        const int64_t c_out = weight.size(1) * groups;
        const int64_t od = (id - 1) * sd - 2 * pd + dd * (kd - 1) + opd + 1;
        const int64_t oh = (ih - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1;
        const int64_t ow = (iw - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1;
        TORCH_CHECK(od > 0 && oh > 0 && ow > 0, "Invalid Iris ConvTranspose3d output shape");
        auto out = at::empty({n, c_out, od, oh, ow}, input.options());
        if (out.numel()) {
            // Dual Conv3d grad_input: grad_in := out [N, c_out, od, oh, ow], grad_out := data [N, c_in, id, ih, iw]
            Conv3dParams dual{static_cast<cl_int>(n), static_cast<cl_int>(c_out), static_cast<cl_int>(od),
                              static_cast<cl_int>(oh), static_cast<cl_int>(ow), static_cast<cl_int>(c_in),
                              static_cast<cl_int>(id), static_cast<cl_int>(ih), static_cast<cl_int>(iw),
                              static_cast<cl_int>(kd), static_cast<cl_int>(kh), static_cast<cl_int>(kw),
                              static_cast<cl_int>(sd), static_cast<cl_int>(sh), static_cast<cl_int>(sw),
                              static_cast<cl_int>(pd), static_cast<cl_int>(ph), static_cast<cl_int>(pw),
                              static_cast<cl_int>(dd), static_cast<cl_int>(dh), static_cast<cl_int>(dw),
                              static_cast<cl_int>(groups)};
            conv3d_grad_input_launch(buf_slice(out).buffer, buf_slice(data).buffer, buf_slice(kern).buffer, dual, fp16);
            if (offset.defined()) {
                out.add_(offset.view({1, c_out, 1, 1, 1}));
            }
        }
        return out;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv3d_backward_dispatch(
    const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef /*output_padding*/, int64_t groups, std::array<bool, 3> mask) {
    check_fp(grad);
    check_fp(input);
    check_fp(weight);
    const auto [sd, sh, sw] = dimensions3(stride, "stride");
    const auto [pd, ph, pw] = dimensions3(padding, "padding");
    const auto [dd, dh, dw] = dimensions3(dilation, "dilation");
    const bool fp16 = input.scalar_type() == at::kHalf;
    const auto pg = grad.contiguous();
    const auto px = input.contiguous();
    const auto pw_t = weight.contiguous();
    at::Tensor gi, gw, gb;
    if (!transposed) {
        Conv3dParams p{static_cast<cl_int>(input.size(0)), static_cast<cl_int>(input.size(1)),
                       static_cast<cl_int>(input.size(2)), static_cast<cl_int>(input.size(3)), static_cast<cl_int>(input.size(4)),
                       static_cast<cl_int>(grad.size(1)), static_cast<cl_int>(grad.size(2)),
                       static_cast<cl_int>(grad.size(3)), static_cast<cl_int>(grad.size(4)),
                       static_cast<cl_int>(weight.size(2)), static_cast<cl_int>(weight.size(3)), static_cast<cl_int>(weight.size(4)),
                       static_cast<cl_int>(sd), static_cast<cl_int>(sh), static_cast<cl_int>(sw),
                       static_cast<cl_int>(pd), static_cast<cl_int>(ph), static_cast<cl_int>(pw),
                       static_cast<cl_int>(dd), static_cast<cl_int>(dh), static_cast<cl_int>(dw),
                       static_cast<cl_int>(groups)};
        if (mask[0]) {
            gi = at::empty(input.sizes(), input.options());
            if (gi.numel()) conv3d_grad_input_launch(buf_slice(gi).buffer, buf_slice(pg).buffer, buf_slice(pw_t).buffer, p, fp16);
        }
        if (mask[1]) {
            gw = at::empty(weight.sizes(), weight.options());
            if (gw.numel()) conv3d_grad_weight_launch(buf_slice(gw).buffer, buf_slice(pg).buffer, buf_slice(px).buffer, p, fp16);
        }
    } else {
        Conv3dParams dual{static_cast<cl_int>(grad.size(0)), static_cast<cl_int>(grad.size(1)),
                          static_cast<cl_int>(grad.size(2)), static_cast<cl_int>(grad.size(3)), static_cast<cl_int>(grad.size(4)),
                          static_cast<cl_int>(input.size(1)), static_cast<cl_int>(input.size(2)),
                          static_cast<cl_int>(input.size(3)), static_cast<cl_int>(input.size(4)),
                          static_cast<cl_int>(weight.size(2)), static_cast<cl_int>(weight.size(3)), static_cast<cl_int>(weight.size(4)),
                          static_cast<cl_int>(sd), static_cast<cl_int>(sh), static_cast<cl_int>(sw),
                          static_cast<cl_int>(pd), static_cast<cl_int>(ph), static_cast<cl_int>(pw),
                          static_cast<cl_int>(dd), static_cast<cl_int>(dh), static_cast<cl_int>(dw),
                          static_cast<cl_int>(groups)};
        if (mask[0]) {
            gi = at::empty(input.sizes(), input.options());
            if (gi.numel()) conv3d_fwd_launch(buf_slice(gi).buffer, buf_slice(pg).buffer, buf_slice(pw_t).buffer, nullptr, dual, fp16);
        }
        if (mask[1]) {
            gw = at::empty(weight.sizes(), weight.options());
            if (gw.numel()) conv3d_grad_weight_launch(buf_slice(gw).buffer, buf_slice(px).buffer, buf_slice(pg).buffer, dual, fp16);
        }
    }
    if (mask[2]) {
        gb = at::sum(grad, {0, 2, 3, 4});
    }
    return {gi, gw, gb};
}

at::Tensor conv(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias,
                at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                at::IntArrayRef output_padding, int64_t groups) {
    if (input.dim() == 5 && weight.dim() == 5) {
        return conv3d_dispatch(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    }
    if (input.dim() == 3 && weight.dim() == 3) {
        const auto s1 = stride.empty() ? int64_t{1} : stride[0];
        const auto p1 = padding.empty() ? int64_t{0} : padding[0];
        const auto d1 = dilation.empty() ? int64_t{1} : dilation[0];
        const auto op1 = output_padding.empty() ? int64_t{0} : output_padding[0];
        return conv(input.unsqueeze(2), weight.unsqueeze(2), bias,
                    {1, s1}, {0, p1}, {1, d1}, transposed, {0, op1}, groups).squeeze(2);
    }
    const auto config = convolution_shape(input, weight, stride, padding, dilation, transposed, output_padding, groups);
    const auto data = input.contiguous();
    const auto kernel = weight.contiguous();
    const auto offset = packed_optional(bias);
    TORCH_CHECK(!offset.defined() || (offset.dim() == 1 && offset.size(0) == config.output[1]),
                "Iris Conv2d bias must have one entry per output channel");
    auto output = at::empty(config.output, input.options());
    if (!output.numel()) {
        return output;
    }
    const auto& p = config.params;
    const bool fp16 = input.scalar_type() == at::kHalf;
    TORCH_CHECK(!offset.defined() || offset.scalar_type() == input.scalar_type(),
                "Iris Conv2d bias dtype must match input dtype");
    if (transposed) {
        conv_transpose2d_forward(buf_slice(output), buf_slice(data), buf_slice(kernel), buf_slice(offset), p, fp16);
        return output;
    }
    if (p.kh == 1 && p.kw == 1 && p.sy == 1 && p.sx == 1 && p.py == 0 && p.px == 0 && p.groups == 1) {
        const auto hw = static_cast<cl_ulong>(p.h) * static_cast<cl_ulong>(p.w);
        const auto in_slice = buf_slice(data);
        const auto w_slice = buf_slice(kernel);
        const auto b_slice = buf_slice(offset);
        const MatrixView left{w_slice.buffer, w_slice.offset, 0, static_cast<cl_ulong>(p.c_in), 1};
        const MatrixView right{in_slice.buffer, in_slice.offset, static_cast<cl_ulong>(p.c_in) * hw, hw, 1};
        const MatrixView bias_view = b_slice.buffer ? MatrixView{b_slice.buffer, b_slice.offset, 0, 1, 0} : MatrixView{};
        matrix_multiply(buf_slice(output).buffer, left, right, p.n, p.c_out, hw, p.c_in, fp16,
                        bias_view, 1.0f, b_slice.buffer ? 1.0f : 0.0f);
        return output;
    }
    conv2d_forward(buf_slice(output), buf_slice(data), buf_slice(kernel), buf_slice(offset), p, fp16);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_backward(
    const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, std::array<bool, 3> mask) {
    if (input.dim() == 5 && weight.dim() == 5) {
        return conv3d_backward_dispatch(grad, input, weight, stride, padding, dilation, transposed, output_padding, groups, mask);
    }
    if (input.dim() == 3 && weight.dim() == 3) {
        const auto s1 = stride.empty() ? int64_t{1} : stride[0];
        const auto p1 = padding.empty() ? int64_t{0} : padding[0];
        const auto d1 = dilation.empty() ? int64_t{1} : dilation[0];
        const auto op1 = output_padding.empty() ? int64_t{0} : output_padding[0];
        auto [gi, gw, gb] = conv_backward(grad.unsqueeze(2), input.unsqueeze(2), weight.unsqueeze(2),
                                          {1, s1}, {0, p1}, {1, d1}, transposed, {0, op1}, groups, mask);
        return {gi.defined() ? gi.squeeze(2) : gi, gw.defined() ? gw.squeeze(2) : gw, gb};
    }
    const auto config = convolution_shape(input, weight, stride, padding, dilation, transposed, output_padding, groups);
    check_fp(grad);
    TORCH_CHECK(grad.scalar_type() == input.scalar_type(), "Iris Conv2d grad dtype must match input dtype");
    TORCH_CHECK(grad.sizes() == config.output, "Invalid Iris Conv2d gradient shape");
    const auto packed_grad = grad.contiguous();
    const auto& p = config.params;
    const bool fp16 = input.scalar_type() == at::kHalf;
    at::Tensor grad_input, grad_weight, grad_bias;
    if (transposed) {
        // Dual Conv2d geometry: input := packed_grad [N, C_out, OH, OW], output := grad_input [N, C_in, H, W],
        // weight := weight [C_in, C_out/g, KH, KW]
        const Conv2dParams dual{
            p.n, p.c_out, p.oh, p.ow, p.c_in, p.h, p.w,
            p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.dy, p.dx, p.groups
        };
        if (mask[0]) {
            const auto kernel = weight.contiguous();
            grad_input = at::empty(input.sizes(), input.options());
            if (grad_input.numel()) {
                conv2d_forward(buf_slice(grad_input), buf_slice(packed_grad), buf_slice(kernel), {}, dual, fp16);
            }
        }
        if (mask[1]) {
            const auto data = input.contiguous();
            grad_weight = at::empty(weight.sizes(), weight.options());
            if (grad_weight.numel()) {
                if (!packed_grad.numel() || !data.numel()) {
                    grad_weight.zero_();
                } else {
                    const int64_t np = static_cast<int64_t>(dual.n) * dual.oh * dual.ow;
                    const int64_t k_col = static_cast<int64_t>(dual.c_in / dual.groups) * dual.kh * dual.kw;
                    const int64_t w_tiles = ((dual.c_out / dual.groups + 31) / 32) * dual.groups * ((k_col + 31) / 32);
                    const int64_t max_mem_split = std::max<int64_t>(1, (8LL * 1024 * 1024) / std::max<int64_t>(1, grad_weight.numel() * 4));
                    int64_t split_k = np >= 512 ? std::min<int64_t>({16, max_mem_split, np / 256, std::max<int64_t>(1, 160 / std::max<int64_t>(1, w_tiles))}) : 1;
                    if (split_k < 1) split_k = 1;
                    at::Tensor partial;
                    if (split_k > 1) partial = at::empty({split_k, grad_weight.numel()}, weight.options().dtype(at::kFloat));
                    conv2d_grad_weight(buf_slice(grad_weight), buf_slice(partial), buf_slice(data), buf_slice(packed_grad), dual,
                                       static_cast<cl_int>(split_k), fp16);
                }
            }
        }
        if (mask[2]) {
            grad_bias = at::sum(grad, {0, 2, 3});
        }
        return {grad_input, grad_weight, grad_bias};
    }
    if (mask[0]) {
        const auto kernel = weight.contiguous();
        grad_input = at::empty(input.sizes(), input.options());
        if (grad_input.numel()) {
            if (p.kh == 1 && p.kw == 1 && p.sy == 1 && p.sx == 1 && p.py == 0 && p.px == 0 && p.groups == 1) {
                const auto hw = static_cast<cl_ulong>(p.h) * static_cast<cl_ulong>(p.w);
                const auto g_slice = buf_slice(packed_grad);
                const auto w_slice = buf_slice(kernel);
                const MatrixView left{w_slice.buffer, w_slice.offset, 0, 1, static_cast<cl_ulong>(p.c_in)};
                const MatrixView right{g_slice.buffer, g_slice.offset, static_cast<cl_ulong>(p.c_out) * hw, hw, 1};
                matrix_multiply(buf_slice(grad_input).buffer, left, right, p.n, p.c_in, hw, p.c_out, fp16);
            } else if (!fp16 && p.groups == 1 && p.kh == 3 && p.kw == 3 && p.sy == 1 && p.sx == 1 &&
                       p.py == 1 && p.px == 1 && p.dy == 1 && p.dx == 1 && p.n >= 4 &&
                       p.h >= 32 && p.w >= 32 && p.c_in >= 64 && p.c_out >= 64) {
                // Reuse the wider output-channel tiling of the forward kernel for dense gradients.
                const auto flipped = kernel.transpose(0, 1).flip({2, 3});
                const Conv2dParams backward{p.n, p.c_out, p.oh, p.ow, p.c_in, p.h, p.w,
                                           3, 3, 1, 1, 1, 1, 1, 1, 1};
                conv2d_forward(buf_slice(grad_input), buf_slice(packed_grad), buf_slice(flipped), {}, backward, false);
            } else {
                conv2d_grad_input(buf_slice(grad_input), buf_slice(packed_grad), buf_slice(kernel), p, fp16);
            }
        }
    }
    if (mask[1]) {
        const auto data = input.contiguous();
        grad_weight = at::empty(weight.sizes(), weight.options());
        if (grad_weight.numel()) {
            if (!packed_grad.numel() || !data.numel()) {
                grad_weight.zero_();
            } else {
                const int64_t np = static_cast<int64_t>(p.n) * p.oh * p.ow;
                const int64_t k_col = static_cast<int64_t>(p.c_in / p.groups) * p.kh * p.kw;
                const int64_t w_tiles = ((p.c_out / p.groups + 31) / 32) * p.groups * ((k_col + 31) / 32);
                const int64_t max_mem_split = std::max<int64_t>(1, (8LL * 1024 * 1024) / std::max<int64_t>(1, grad_weight.numel() * 4));
                int64_t split_k = np >= 512 ? std::min<int64_t>({16, max_mem_split, np / 256, std::max<int64_t>(1, 160 / std::max<int64_t>(1, w_tiles))}) : 1;
                if (split_k < 1) {
                    split_k = 1;
                }
                at::Tensor partial;
                if (split_k > 1) {
                    partial = at::empty({split_k, grad_weight.numel()}, weight.options().dtype(at::kFloat));
                }
                conv2d_grad_weight(buf_slice(grad_weight), buf_slice(partial), buf_slice(packed_grad), buf_slice(data), p,
                                   static_cast<cl_int>(split_k), fp16);
            }
        }
    }
    if (mask[2]) {
        grad_bias = at::sum(grad, {0, 2, 3});
    }
    return {grad_input, grad_weight, grad_bias};
}

struct Pooling {
    at::DimVector output;
    Pool2dParams params;
};

Pooling pool_shape(const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
                   at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, bool average,
                   bool include_pad = false, std::optional<int64_t> divisor = std::nullopt) {
    check_fp(input);
    TORCH_CHECK(input.dim() == 3 || input.dim() == 4, "Iris pooling expects CHW or NCHW input");
    const auto [kh, kw] = dimensions(kernel, "kernel_size");
    const auto [sy, sx] = stride.empty() ? std::array{kh, kw} : dimensions(stride, "stride");
    const auto [py, px] = dimensions(padding, "padding");
    const auto [dy, dx] = dimensions(dilation, "dilation");
    const auto channels = input.size(-3), h = input.size(-2), w = input.size(-1);
    TORCH_CHECK(kh > 0 && kw > 0 && sy > 0 && sx > 0 && dy > 0 && dx > 0, "Invalid Iris pooling geometry");
    TORCH_CHECK(!divisor || *divisor != 0, "divisor must be not zero");
    const auto oh = at::native::pooling_output_shape(h, kh, py, sy, dy, ceil_mode);
    const auto ow = at::native::pooling_output_shape(w, kw, px, sx, dx, ceil_mode);
    at::native::pool2d_shape_check(input, kh, kw, sy, sx, py, px, dy, dx, channels, h, w, oh, ow,
                                   c10::MemoryFormat::Contiguous);
    const auto n = input.dim() == 4 ? input.size(0) : 1;
    Pooling result{at::DimVector(input.sizes()), {
        static_cast<cl_int>(n), static_cast<cl_int>(channels), static_cast<cl_int>(h), static_cast<cl_int>(w),
        static_cast<cl_int>(oh), static_cast<cl_int>(ow),
        static_cast<cl_int>(kh), static_cast<cl_int>(kw), static_cast<cl_int>(sy), static_cast<cl_int>(sx),
        static_cast<cl_int>(py), static_cast<cl_int>(px), static_cast<cl_int>(dy), static_cast<cl_int>(dx),
        static_cast<cl_int>(include_pad), static_cast<cl_int>(divisor.value_or(0))
    }};
    result.output[result.output.size() - 2] = oh;
    result.output.back() = ow;
    return result;
}

std::tuple<at::Tensor, at::Tensor> pool_max(
    const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool ceil_mode) {
    const auto config = pool_shape(input, kernel, stride, padding, dilation, ceil_mode, false);
    auto output = at::empty(config.output, input.options());
    auto indices = at::empty(config.output, input.options().dtype(at::kLong));
    if (output.numel()) {
        const auto packed = input.contiguous();
        max_pool2d_forward(buf_slice(output), buf_slice(indices), buf_slice(packed), config.params,
                           input.scalar_type() == at::kHalf);
    }
    return {output, indices};
}

at::Tensor pool_max_backward(
    const at::Tensor& grad, const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor& indices) {
    const auto config = pool_shape(input, kernel, stride, padding, dilation, ceil_mode, false);
    check_fp(grad);
    TORCH_CHECK(grad.scalar_type() == input.scalar_type(), "Iris pooling grad dtype must match input");
    TORCH_CHECK(grad.sizes() == config.output, "Invalid Iris pooling gradient shape");
    TORCH_CHECK(indices.device() == input.device() && indices.scalar_type() == at::kLong &&
                indices.sizes() == config.output, "Invalid Iris max pooling indices");
    auto output = at::empty(input.sizes(), input.options());
    if (output.numel()) {
        max_pool2d_backward(buf_slice(output), buf_slice(grad.contiguous()), buf_slice(indices.contiguous()),
                            config.params, input.scalar_type() == at::kHalf);
    }
    return output;
}

at::Tensor pool_avg(
    const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride, at::IntArrayRef padding,
    bool ceil_mode, bool include_pad, std::optional<int64_t> divisor) {
    const auto config = pool_shape(input, kernel, stride, padding, {1, 1}, ceil_mode, true, include_pad, divisor);
    auto output = at::empty(config.output, input.options());
    if (output.numel()) {
        avg_pool2d_forward(buf_slice(output), buf_slice(input.contiguous()), config.params,
                           input.scalar_type() == at::kHalf);
    }
    return output;
}

at::Tensor pool_avg_backward(
    const at::Tensor& grad, const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool include_pad, std::optional<int64_t> divisor) {
    const auto config = pool_shape(input, kernel, stride, padding, {1, 1}, ceil_mode, true, include_pad, divisor);
    check_fp(grad);
    TORCH_CHECK(grad.scalar_type() == input.scalar_type(), "Iris pooling grad dtype must match input");
    TORCH_CHECK(grad.sizes() == config.output, "Invalid Iris pooling gradient shape");
    auto output = at::empty(input.sizes(), input.options());
    if (output.numel()) {
        avg_pool2d_backward(buf_slice(output), buf_slice(grad.contiguous()), config.params,
                            input.scalar_type() == at::kHalf);
    }
    return output;
}

struct Normalization {
    at::DimVector stats_shape;
    cl_ulong rows;
    cl_ulong width;
};

Normalization layer_norm_shape(const at::Tensor& input, at::IntArrayRef shape,
                               const at::Tensor& weight, const at::Tensor& bias) {
    check_fp(input);
    TORCH_CHECK(!shape.empty() && shape.size() <= input.dim() &&
                input.sizes().slice(input.dim() - shape.size()) == shape,
                "Iris LayerNorm normalized shape must match the trailing input dimensions");
    TORCH_CHECK(!weight.defined() || weight.sizes() == shape, "Iris LayerNorm weight shape mismatch");
    TORCH_CHECK(!bias.defined() || bias.sizes() == shape, "Iris LayerNorm bias shape mismatch");
    const auto prefix = input.dim() - shape.size();
    Normalization result{at::DimVector(input.sizes()),
        static_cast<cl_ulong>(c10::multiply_integers(input.sizes().slice(0, prefix))),
        static_cast<cl_ulong>(c10::multiply_integers(shape))};
    std::fill(result.stats_shape.begin() + prefix, result.stats_shape.end(), 1);
    return result;
}

constexpr std::size_t normalization_group_size = 64;

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm(
    const at::Tensor& input, at::IntArrayRef normalized_shape, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, double epsilon) {
    const auto affine = packed_optional(weight);
    const auto offset = packed_optional(bias);
    const auto config = layer_norm_shape(input, normalized_shape, affine, offset);
    const bool in_fp16 = input.scalar_type() == at::kHalf;
    const bool param_fp16 = affine.defined() ? (affine.scalar_type() == at::kHalf)
                          : (offset.defined() ? (offset.scalar_type() == at::kHalf) : in_fp16);
    TORCH_CHECK(!affine.defined() || !offset.defined() || affine.scalar_type() == offset.scalar_type(),
                "Iris LayerNorm weight and bias dtypes must match");
    auto output = at::empty(input.sizes(), input.options());
    auto mean = at::empty(config.stats_shape, input.options());
    auto rstd = at::empty(config.stats_shape, input.options());
    if (output.numel()) {
        const auto packed = input.contiguous();
        layer_norm_forward_fast(buf_slice(output), buf_slice(packed), buf_slice(affine), buf_slice(offset),
                                buf_slice(mean), buf_slice(rstd), config.rows,
                                static_cast<cl_int>(config.width), static_cast<float>(epsilon),
                                in_fp16, param_fp16);
    }
    return {output, mean, rstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward(
    const at::Tensor& grad, const at::Tensor& input, at::IntArrayRef normalized_shape,
    const at::Tensor& mean, const at::Tensor& rstd, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, std::array<bool, 3> mask) {
    check_fp(grad);
    check_fp(mean);
    check_fp(rstd);
    const auto affine = packed_optional(weight);
    const auto offset = packed_optional(bias);
    const auto config = layer_norm_shape(input, normalized_shape, affine, offset);
    const bool in_fp16 = input.scalar_type() == at::kHalf;
    const bool param_fp16 = affine.defined() ? (affine.scalar_type() == at::kHalf)
                          : (offset.defined() ? (offset.scalar_type() == at::kHalf) : in_fp16);
    const auto param_dtype = param_fp16 ? at::kHalf : at::kFloat;
    TORCH_CHECK(grad.sizes() == input.sizes() && mean.numel() == config.rows && rstd.numel() == config.rows,
                "Invalid Iris LayerNorm gradient shape");
    const auto packed = input.contiguous();
    const auto packed_grad = grad.contiguous();
    const auto packed_mean = mean.contiguous();
    const auto packed_rstd = rstd.contiguous();
    at::Tensor dx, dw, db;
    if (mask[0]) {
        dx = at::empty(input.sizes(), input.options());
        if (dx.numel()) {
            layer_norm_grad_input_fast(buf_slice(dx), buf_slice(packed_grad), buf_slice(packed), buf_slice(affine),
                                       buf_slice(packed_mean), buf_slice(packed_rstd), config.rows,
                                       static_cast<cl_int>(config.width), in_fp16, param_fp16);
        }
    }
    if (mask[1]) {
        dw = at::empty(normalized_shape, input.options().dtype(param_dtype));
    }
    if (mask[2]) {
        db = at::empty(normalized_shape, input.options().dtype(param_dtype));
    }
    if (mask[1] || mask[2]) {
        if (config.rows == 0 || config.width == 0) {
            if (dw.defined()) dw.zero_();
            if (db.defined()) db.zero_();
        } else {
            layer_norm_grad_affine_fast(buf_slice(dw), buf_slice(db), buf_slice(packed_grad), buf_slice(packed),
                                        buf_slice(packed_mean), buf_slice(packed_rstd),
                                        static_cast<cl_int>(config.rows), static_cast<cl_int>(config.width),
                                        in_fp16, param_fp16);
        }
    }
    return {dx, dw, db};
}

SdpaParams make_sdpa_params(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                            const at::Tensor& output, float multiplier, bool causal) {
    return SdpaParams{
        static_cast<cl_int>(query.size(0)),
        static_cast<cl_int>(query.size(1)),
        static_cast<cl_int>(key.size(1)),
        static_cast<cl_int>(query.size(2)),
        static_cast<cl_int>(key.size(2)),
        static_cast<cl_int>(query.size(3)),
        static_cast<cl_int>(value.size(3)),
        static_cast<cl_ulong>(query.stride(0)), static_cast<cl_ulong>(query.stride(1)),
        static_cast<cl_ulong>(query.stride(2)), static_cast<cl_ulong>(query.stride(3)),
        static_cast<cl_ulong>(key.stride(0)), static_cast<cl_ulong>(key.stride(1)),
        static_cast<cl_ulong>(key.stride(2)), static_cast<cl_ulong>(key.stride(3)),
        static_cast<cl_ulong>(value.stride(0)), static_cast<cl_ulong>(value.stride(1)),
        static_cast<cl_ulong>(value.stride(2)), static_cast<cl_ulong>(value.stride(3)),
        static_cast<cl_ulong>(output.stride(0)), static_cast<cl_ulong>(output.stride(1)),
        static_cast<cl_ulong>(output.stride(2)), static_cast<cl_ulong>(output.stride(3)),
        multiplier, causal, query.scalar_type() == at::kHalf,
    };
}

struct IrisFlashSdpaFunction : public torch::autograd::Function<IrisFlashSdpaFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        bool causal,
        double scale_val) {
        at::DimVector shape(query.sizes());
        shape.back() = value.size(-1);
        const bool preserve_layout = query.size(-1) == value.size(-1) &&
            query.stride(-1) == 1 && key.stride(-1) == 1 && value.stride(-1) == 1;
        auto output = preserve_layout ? at::empty_like(query) : at::empty(shape, query.options());
        auto lse = at::empty({query.size(0), query.size(1), query.size(2)},
                             query.options().dtype(at::kFloat));
        if (output.numel()) {
            const auto params = make_sdpa_params(query, key, value, output, static_cast<float>(scale_val), causal);
            sdpa_flash_forward(buf_slice(output), buf_slice(lse), buf_slice(query), buf_slice(key), buf_slice(value), params);
        }
        ctx->save_for_backward({query, key, value, output, lse});
        ctx->saved_data["causal"] = causal;
        ctx->saved_data["scale"] = scale_val;
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        const auto& query = saved[0];
        const auto& key = saved[1];
        const auto& value = saved[2];
        const auto& output = saved[3];
        const auto& lse = saved[4];
        const bool causal = ctx->saved_data["causal"].toBool();
        const float scale_val = static_cast<float>(ctx->saved_data["scale"].toDouble());
        const auto& dout = grad_outputs[0];

        const bool need_dq = ctx->needs_input_grad(0);
        const bool need_dk = ctx->needs_input_grad(1);
        const bool need_dv = ctx->needs_input_grad(2);

        at::Tensor dq, dk, dv;
        if (need_dq) dq = at::empty(query.sizes(), query.options());
        if (need_dk) dk = at::empty(key.sizes(), key.options());
        if (need_dv) dv = at::empty(value.sizes(), value.options());

        if (output.numel() && (need_dq || need_dk || need_dv)) {
            auto d_buf = at::empty({query.size(0), query.size(1), query.size(2)},
                                   query.options().dtype(at::kFloat));
            SdpaBwdParams bp{};
            bp.fwd = make_sdpa_params(query, key, value, output, scale_val, causal);
            bp.do_sb = static_cast<cl_ulong>(dout.stride(0));
            bp.do_sh = static_cast<cl_ulong>(dout.stride(1));
            bp.do_sl = static_cast<cl_ulong>(dout.stride(2));
            bp.do_sd = static_cast<cl_ulong>(dout.stride(3));
            if (need_dq) {
                bp.dq_sb = static_cast<cl_ulong>(dq.stride(0));
                bp.dq_sh = static_cast<cl_ulong>(dq.stride(1));
                bp.dq_sl = static_cast<cl_ulong>(dq.stride(2));
                bp.dq_sd = static_cast<cl_ulong>(dq.stride(3));
            }
            if (need_dk) {
                bp.dk_sb = static_cast<cl_ulong>(dk.stride(0));
                bp.dk_sh = static_cast<cl_ulong>(dk.stride(1));
                bp.dk_sl = static_cast<cl_ulong>(dk.stride(2));
                bp.dk_sd = static_cast<cl_ulong>(dk.stride(3));
            }
            if (need_dv) {
                bp.dv_sb = static_cast<cl_ulong>(dv.stride(0));
                bp.dv_sh = static_cast<cl_ulong>(dv.stride(1));
                bp.dv_sl = static_cast<cl_ulong>(dv.stride(2));
                bp.dv_sd = static_cast<cl_ulong>(dv.stride(3));
            }
            sdpa_flash_backward(buf_slice(dq), buf_slice(dk), buf_slice(dv), buf_slice(d_buf),
                                buf_slice(dout), buf_slice(output), buf_slice(query), buf_slice(key), buf_slice(value),
                                buf_slice(lse), bp, need_dq, need_dk, need_dv);
        }
        return {dq, dk, dv, at::Tensor{}, at::Tensor{}};
    }
};

at::Tensor attention(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                     const std::optional<at::Tensor>& mask, double dropout, bool causal,
                     std::optional<double> scale, bool gqa) {
    const bool needs_grad = at::GradMode::is_enabled() &&
        (query.requires_grad() || key.requires_grad() || value.requires_grad() ||
         (mask && mask->defined() && mask->requires_grad()));
    const bool heads_ok = query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
        query.size(0) == key.size(0) && query.size(0) == value.size(0) &&
        key.size(1) == value.size(1) && key.size(1) > 0 &&
        (gqa ? (query.size(1) % key.size(1) == 0) : (query.size(1) == key.size(1)));
    const bool use_flash = has_subgroups();
    const bool dtype_ok = query.scalar_type() == key.scalar_type() &&
        query.scalar_type() == value.scalar_type() &&
        (query.scalar_type() == at::kFloat || (use_flash && has_fp16() && query.scalar_type() == at::kHalf));
    const int64_t max_head_dim = use_flash ? 128 : 32;
    const bool seq_ok = use_flash ? (key.size(-2) > 0) : (key.size(-2) > 0 && key.size(-2) <= 256);
    const bool compatible = (!mask || !mask->defined()) && dropout == 0.0 &&
        heads_ok && (!gqa || use_flash) &&
        query.device() == key.device() && query.device() == value.device() &&
        dtype_ok && (!needs_grad || use_flash) &&
        query.size(-1) == key.size(-1) && query.size(-1) > 0 && query.size(-1) <= max_head_dim &&
        value.size(-1) > 0 && value.size(-1) <= max_head_dim &&
        key.size(-2) == value.size(-2) && seq_ok;
    // Iris Xe GEMM outpaces the current D128 fused backward for these causal training shapes.
    // Bound the materialized attention matrix to four million FP32 elements.
    constexpr int64_t max_scores = 4LL * 1024 * 1024;
    const bool use_math = compatible && needs_grad && causal && !gqa && query.scalar_type() == at::kFloat &&
        query.size(-1) == 128 && value.size(-1) == 128 && query.size(-2) == key.size(-2) &&
        query.size(-2) >= 256 && query.size(-2) <= 1024 &&
        query.size(0) * query.size(1) <= max_scores / (query.size(-2) * query.size(-2)) &&
        query.stride(-1) == 1 && key.stride(-1) == 1 && value.stride(-1) == 1;
    if (!compatible || use_math) {
        return at::native::scaled_dot_product_attention(query, key, value, mask, dropout, causal, scale, gqa);
    }

    const double scale_val = scale.value_or(1.0 / std::sqrt(static_cast<double>(query.size(-1))));
    if (needs_grad) {
        return IrisFlashSdpaFunction::apply(query, key, value, causal, scale_val);
    }

    at::DimVector shape(query.sizes());
    shape.back() = value.size(-1);
    const bool preserve_layout = query.size(-1) == value.size(-1) &&
        query.stride(-1) == 1 && key.stride(-1) == 1 && value.stride(-1) == 1;
    auto output = preserve_layout ? at::empty_like(query) : at::empty(shape, query.options());
    if (output.numel()) {
        const float multiplier = static_cast<float>(scale_val);
        if (use_flash) {
            const auto params = make_sdpa_params(query, key, value, output, multiplier, causal);
            if (params.lq <= 4 && params.lk >= 64) {
                const int64_t bhq = static_cast<int64_t>(params.b) * params.hq * params.lq;
                cl_int num_splits = 1;
                if (params.lk >= 128 && bhq < 128) {
                    const cl_int desired = static_cast<cl_int>((160 + bhq - 1) / bhq);
                    const cl_int max_by_lk = std::max<cl_int>(1, params.lk / 64);
                    num_splits = std::min<cl_int>({32, desired, max_by_lk});
                }
                at::Tensor partial_o, partial_lse;
                if (num_splits > 1) {
                    partial_o = at::empty({params.b * params.hq, params.lq, num_splits, params.dv},
                                          query.options().dtype(at::kFloat));
                    partial_lse = at::empty({params.b * params.hq, params.lq, num_splits},
                                            query.options().dtype(at::kFloat));
                }
                sdpa_flash_decode(buf_slice(output), buf_slice(partial_o), buf_slice(partial_lse),
                                  buf_slice(query), buf_slice(key), buf_slice(value), params, num_splits);
            } else {
                sdpa_flash_forward(buf_slice(output), {}, buf_slice(query), buf_slice(key), buf_slice(value), params);
            }
        } else {
            std::array<cl_ulong, 23> args{
                static_cast<cl_ulong>(query.size(-2)), static_cast<cl_ulong>(key.size(-2)),
                static_cast<cl_ulong>(query.size(-1)), static_cast<cl_ulong>(value.size(-1)),
                static_cast<cl_ulong>(query.size(1)),
            };
            std::copy(query.strides().begin(), query.strides().end(), args.begin() + 5);
            std::copy(key.strides().begin(), key.strides().end(), args.begin() + 9);
            std::copy(value.strides().begin(), value.strides().end(), args.begin() + 13);
            args[17] = causal;
            args[18] = std::bit_cast<cl_uint>(multiplier);
            std::copy(output.strides().begin(), output.strides().end(), args.begin() + 19);
            run_spatial("fused_attention", output.numel() / value.size(-1) * normalization_group_size,
                        normalization_group_size, args, {buf_slice(output), buf_slice(query), buf_slice(key), buf_slice(value)});
        }
    }
    return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, library) {
    library.impl("convolution_overrideable", &conv);
    library.impl("convolution_backward_overrideable", &conv_backward);
    library.impl("max_pool2d_with_indices", &pool_max);
    library.impl("max_pool2d_with_indices_backward", &pool_max_backward);
    library.impl("avg_pool2d", &pool_avg);
    library.impl("avg_pool2d_backward", &pool_avg_backward);
    library.impl("native_layer_norm", &layer_norm);
    library.impl("native_layer_norm_backward", &layer_norm_backward);
    library.impl("scaled_dot_product_attention", &attention);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, library) {
    library.impl("scaled_dot_product_attention", &attention);
}

}  // namespace
}  // namespace iris
