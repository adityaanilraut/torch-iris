#include "opencl_runtime.h"
#include "rng_internal.h"

#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace iris {
namespace {

constexpr auto kDev = c10::DeviceType::PrivateUse1;

void check_iris(const at::Tensor& t, const char* op) {
    TORCH_CHECK(t.device().type() == kDev, "Iris ", op, " expects an Iris tensor");
}

void check_iris_fp(const at::Tensor& t, const char* op) {
    check_iris(t, op);
    TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf,
                "Iris ", op, " supports float32 and float16");
}

ValueType to_vtype(at::ScalarType st) {
    switch (st) {
        case at::kFloat: return ValueType::Float;
        case at::kHalf:  return ValueType::Half;
        case at::kLong:  return ValueType::Long;
        case at::kInt:   return ValueType::Int;
        case at::kByte:  return ValueType::Byte;
        case at::kBool:  return ValueType::Bool;
        default: TORCH_CHECK(false, "Iris unsupported scalar type: ", st);
    }
}

cl_mem buf(const at::Tensor& t) {
    if (!t.defined() || t.numel() == 0) return nullptr;
    return static_cast<cl_mem>(t.storage().data_ptr().get());
}

at::Tensor packed0(const at::Tensor& t) {
    if (t.is_contiguous() && t.storage_offset() == 0) return t;
    auto out = at::empty(t.sizes(), t.options());
    if (out.numel()) out.copy_(t);
    return out;
}

template <typename T>
T read_scalar_val(const at::Tensor& t) {
    T v{};
    download(&v, buf(t), sizeof(T), static_cast<std::size_t>(t.storage_offset()) * sizeof(T));
    return v;
}

// =========================================================================
// 1. LLM Sampling, Masking & Indexing
// =========================================================================

at::Tensor multinomial_impl(
    const at::Tensor& self, int64_t num_samples, bool replacement,
    std::optional<at::Generator> generator) {
    check_iris_fp(self, "multinomial");
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "prob_dist must be 1 or 2 dim");
    TORCH_CHECK(num_samples > 0, "cannot sample num_samples <= 0 samples");
    const bool is_1d = (self.dim() == 1);
    const auto probs2d = packed0(is_1d ? self.unsqueeze(0) : self);
    const int64_t rows = probs2d.size(0);
    const int64_t k = probs2d.size(1);
    TORCH_CHECK(k > 0, "prob_dist must have non-zero last dimension");
    TORCH_CHECK(replacement || num_samples <= k,
                "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
    const bool is_fp16 = (self.scalar_type() == at::kHalf);

    auto invalid = at::zeros({1}, self.options().dtype(at::kInt));
    at::Tensor out2d;
    if (replacement) {
        auto cdf = at::empty({rows, k}, self.options().dtype(at::kFloat));
        multinomial_build_cdf_launch(buf(cdf), buf(probs2d), buf(invalid),
                                     static_cast<cl_int>(rows), static_cast<cl_int>(k), is_fp16);
        const int32_t err = read_scalar_val<int32_t>(invalid);
        TORCH_CHECK((err & 1) == 0, "probability tensor contains either `inf`, `nan` or element < 0");
        TORCH_CHECK((err & 2) == 0, "invalid multinomial distribution (sum of probabilities <= 0)");

        out2d = at::empty({rows, num_samples}, self.options().dtype(at::kLong));
        const uint64_t total = static_cast<uint64_t>(rows) * static_cast<uint64_t>(num_samples);
        const uint64_t blocks = (total + 3) / 4;
        const auto [seed, start] = reserve_rng_blocks_opt(blocks, generator);
        multinomial_sample_repl_launch(buf(out2d), buf(cdf),
                                       static_cast<cl_int>(rows), static_cast<cl_int>(k),
                                       static_cast<cl_int>(num_samples), seed, start);
    } else {
        auto keys = at::empty({rows, k}, self.options().dtype(at::kFloat));
        const uint64_t total = static_cast<uint64_t>(rows) * static_cast<uint64_t>(k);
        const uint64_t blocks = (total + 3) / 4;
        const auto [seed, start] = reserve_rng_blocks_opt(blocks, generator);
        multinomial_exp_keys_launch(buf(keys), buf(probs2d), buf(invalid),
                                    static_cast<cl_int>(rows), static_cast<cl_int>(k),
                                    static_cast<cl_int>(num_samples), seed, start, is_fp16);
        const int32_t err = read_scalar_val<int32_t>(invalid);
        TORCH_CHECK((err & 1) == 0, "probability tensor contains either `inf`, `nan` or element < 0");
        TORCH_CHECK((err & 2) == 0, "invalid multinomial distribution (sum of probabilities <= 0)");
        TORCH_CHECK((err & 4) == 0, "invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)");

        auto [vals, idxs] = at::topk(keys, num_samples, -1, /*largest=*/true, /*sorted=*/false);
        out2d = idxs;
    }
    return is_1d ? out2d.squeeze(0) : out2d;
}

at::Tensor& multinomial_out(
    const at::Tensor& self, int64_t num_samples, bool replacement,
    std::optional<at::Generator> generator, at::Tensor& out) {
    auto res = multinomial_impl(self, num_samples, replacement, generator);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

bool parse_searchsorted_right(bool right, std::optional<c10::string_view> side) {
    if (side.has_value()) {
        TORCH_CHECK(*side == "left" || *side == "right", "side can only be 'left' or 'right'");
        TORCH_CHECK(!(*side == "left" && right), "side and right can't be set to opposites");
        return *side == "right";
    }
    return right;
}

at::Tensor searchsorted_tensor_impl(
    const at::Tensor& sorted_sequence, const at::Tensor& self,
    bool out_int32, bool right, std::optional<c10::string_view> side,
    const std::optional<at::Tensor>& sorter) {
    check_iris(sorted_sequence, "searchsorted");
    check_iris(self, "searchsorted");
    const bool use_right = parse_searchsorted_right(right, side);
    TORCH_CHECK(sorted_sequence.dim() >= 1, "sorted_sequence must be at least 1D");
    const int64_t seq_len = sorted_sequence.size(-1);
    const int64_t seq_rows = sorted_sequence.dim() == 1 ? 1 : (sorted_sequence.numel() / std::max<int64_t>(1, seq_len));
    if (sorted_sequence.dim() > 1) {
        TORCH_CHECK(self.dim() == sorted_sequence.dim(),
                    "searchsorted: dims of sorted_sequence and values must match when sorted_sequence is >1D");
        for (int64_t d = 0; d < self.dim() - 1; ++d) {
            TORCH_CHECK(self.size(d) == sorted_sequence.size(d),
                        "searchsorted: leading dimensions must match");
        }
    }
    at::Tensor sorter_c;
    if (sorter && sorter->defined()) {
        check_iris(*sorter, "searchsorted");
        TORCH_CHECK(sorter->sizes() == sorted_sequence.sizes(), "sorter must have same shape as sorted_sequence");
        TORCH_CHECK(sorter->scalar_type() == at::kLong, "sorter must be int64");
        sorter_c = packed0(*sorter);
    }
    auto out = at::empty(self.sizes(), self.options().dtype(out_int32 ? at::kInt : at::kLong));
    if (out.numel()) {
        const auto seq_c = packed0(sorted_sequence);
        const auto val_c = packed0(self);
        const int64_t vals_per_row = sorted_sequence.dim() == 1 ? val_c.numel() : self.size(-1);
        searchsorted_launch(buf(out), buf(seq_c), buf(val_c), buf(sorter_c),
                            static_cast<cl_ulong>(out.numel()),
                            static_cast<cl_ulong>(seq_rows),
                            static_cast<cl_ulong>(seq_len),
                            static_cast<cl_ulong>(vals_per_row),
                            to_vtype(seq_c.scalar_type()),
                            to_vtype(val_c.scalar_type()),
                            sorter_c.defined(), use_right, out_int32);
    }
    return out;
}

at::Tensor& searchsorted_tensor_out(
    const at::Tensor& sorted_sequence, const at::Tensor& self,
    bool out_int32, bool right, std::optional<c10::string_view> side,
    const std::optional<at::Tensor>& sorter, at::Tensor& out) {
    auto res = searchsorted_tensor_impl(sorted_sequence, self, out_int32, right, side, sorter);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor searchsorted_scalar_impl(
    const at::Tensor& sorted_sequence, const at::Scalar& self,
    bool out_int32, bool right, std::optional<c10::string_view> side,
    const std::optional<at::Tensor>& sorter) {
    check_iris(sorted_sequence, "searchsorted");
    TORCH_CHECK(sorted_sequence.dim() == 1, "sorted_sequence must be 1D when values is a scalar");
    auto val_dtype = at::result_type(sorted_sequence, self);
    auto val_t = at::empty({}, sorted_sequence.options().dtype(val_dtype));
    val_t.fill_(self);
    return searchsorted_tensor_impl(sorted_sequence, val_t, out_int32, right, side, sorter);
}

at::Tensor& searchsorted_scalar_out(
    const at::Tensor& sorted_sequence, const at::Scalar& self,
    bool out_int32, bool right, std::optional<c10::string_view> side,
    const std::optional<at::Tensor>& sorter, at::Tensor& out) {
    auto res = searchsorted_scalar_impl(sorted_sequence, self, out_int32, right, side, sorter);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor& masked_scatter_impl(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source) {
    check_iris(self, "masked_scatter_");
    check_iris(mask, "masked_scatter_");
    check_iris(source, "masked_scatter_");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_scatter_ expects a bool mask");
    TORCH_CHECK(self.scalar_type() == source.scalar_type(), "masked_scatter_ expects matching self and source dtypes");
    at::assert_no_internal_overlap(self);
    if (!self.numel()) return self;
    const auto bmask = packed0(mask.expand(self.sizes()));
    const auto src_c = packed0(source);
    const bool direct = self.is_contiguous() && self.storage_offset() == 0;
    auto work = direct ? self : packed0(self);
    const int64_t len = work.numel();
    const int64_t num_blocks = (len + 255) / 256;
    auto b_counts = at::empty({num_blocks}, self.options().dtype(at::kLong));
    auto b_offsets = at::empty({num_blocks}, self.options().dtype(at::kLong));
    auto total_t = at::empty({1}, self.options().dtype(at::kLong));
    const cl_ulong total = masked_scatter_launch(
        buf(work), buf(bmask), buf(src_c),
        buf(b_counts), buf(b_offsets), buf(total_t),
        static_cast<cl_ulong>(len), static_cast<cl_ulong>(src_c.numel()),
        to_vtype(self.scalar_type()));
    TORCH_CHECK(static_cast<int64_t>(total) <= src_c.numel(),
                "masked_scatter_: number of elements of source (", src_c.numel(),
                ") must be >= number of True elements in mask (", total, ")");
    if (!direct) self.copy_(work);
    return self;
}

at::Tensor isin_tt_impl(const at::Tensor& elements, const at::Tensor& test_elements,
                        bool /*assume_unique*/, bool invert) {
    check_iris(elements, "isin");
    check_iris(test_elements, "isin");
    auto out = at::empty(elements.sizes(), elements.options().dtype(at::kBool));
    if (!out.numel()) return out;
    if (!test_elements.numel()) {
        out.fill_(invert);
        return out;
    }
    const auto e_c = packed0(elements);
    auto t_flat = packed0(test_elements).reshape({-1});
    bool is_sorted = false;
    if (t_flat.numel() > 16) {
        t_flat = std::get<0>(at::sort(t_flat, 0, false));
        is_sorted = true;
    }
    isin_launch(buf(out), buf(e_c), buf(t_flat),
                static_cast<cl_ulong>(e_c.numel()), static_cast<cl_ulong>(t_flat.numel()),
                to_vtype(e_c.scalar_type()), to_vtype(t_flat.scalar_type()),
                is_sorted, invert);
    return out;
}

at::Tensor& isin_tt_out(const at::Tensor& elements, const at::Tensor& test_elements,
                        bool assume_unique, bool invert, at::Tensor& out) {
    auto res = isin_tt_impl(elements, test_elements, assume_unique, invert);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor isin_ts_impl(const at::Tensor& elements, const at::Scalar& test_element,
                        bool assume_unique, bool invert) {
    check_iris(elements, "isin");
    auto dt = at::result_type(elements, test_element);
    auto t = at::empty({1}, elements.options().dtype(dt));
    t.fill_(test_element);
    return isin_tt_impl(elements, t, assume_unique, invert);
}

at::Tensor& isin_ts_out(const at::Tensor& elements, const at::Scalar& test_element,
                        bool assume_unique, bool invert, at::Tensor& out) {
    auto res = isin_ts_impl(elements, test_element, assume_unique, invert);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor isin_st_impl(const at::Scalar& element, const at::Tensor& test_elements,
                        bool assume_unique, bool invert) {
    check_iris(test_elements, "isin");
    auto dt = at::result_type(test_elements, element);
    auto e = at::empty({}, test_elements.options().dtype(dt));
    e.fill_(element);
    return isin_tt_impl(e, test_elements, assume_unique, invert);
}

at::Tensor& isin_st_out(const at::Scalar& element, const at::Tensor& test_elements,
                        bool assume_unique, bool invert, at::Tensor& out) {
    auto res = isin_st_impl(element, test_elements, assume_unique, invert);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor& index_add_inplace(at::Tensor& self, int64_t dim, const at::Tensor& index,
                              const at::Tensor& source, const at::Scalar& alpha) {
    check_iris(self, "index_add_");
    check_iris(index, "index_add_");
    check_iris(source, "index_add_");
    TORCH_CHECK(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt,
                "index_add_ expects int64 or int32 index");
    TORCH_CHECK(index.dim() <= 1, "index_add_ expects 1D index");
    TORCH_CHECK(self.scalar_type() == source.scalar_type(),
                "index_add_ expects matching self and source dtypes");
    at::assert_no_internal_overlap(self);
    const int64_t ndim = self.dim();
    const int64_t d = ndim == 0 ? 0 : at::maybe_wrap_dim(dim, ndim);
    const int64_t idx_len = index.numel();
    if (!idx_len || !source.numel()) return self;
    if (ndim == 0) {
        auto idx_c = packed0(index.reshape({-1}));
        auto src_c = packed0(source.reshape({1}));
        auto work = packed0(self.reshape({1}));
        auto invalid = at::zeros({1}, self.options().dtype(at::kInt));
        index_add_launch(buf(work), buf(src_c), buf(idx_c), buf(invalid),
                         1, 1, static_cast<cl_ulong>(idx_len), 1,
                         alpha.toFloat(), alpha.toLong(),
                         to_vtype(self.scalar_type()), to_vtype(idx_c.scalar_type()));
        TORCH_CHECK_INDEX(!read_scalar_val<int32_t>(invalid), "index_add_ index out of bounds");
        self.copy_(work.squeeze(0));
        return self;
    }
    TORCH_CHECK(source.dim() == ndim, "index_add_: source and self must have same rank");
    TORCH_CHECK(source.size(d) == idx_len, "index_add_: source.size(dim) must equal index.numel()");
    for (int64_t i = 0; i < ndim; ++i) {
        TORCH_CHECK(i == d || source.size(i) == self.size(i), "index_add_: shape mismatch");
    }
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < d; ++i) outer *= self.size(i);
    for (int64_t i = d + 1; i < ndim; ++i) inner *= self.size(i);
    const bool direct = self.is_contiguous() && self.storage_offset() == 0;
    auto work = direct ? self : packed0(self);
    const auto src_c = packed0(source);
    const auto idx_c = packed0(index.reshape({-1}));
    auto invalid = at::zeros({1}, self.options().dtype(at::kInt));
    index_add_launch(buf(work), buf(src_c), buf(idx_c), buf(invalid),
                     static_cast<cl_ulong>(outer), static_cast<cl_ulong>(self.size(d)),
                     static_cast<cl_ulong>(idx_len), static_cast<cl_ulong>(inner),
                     alpha.toFloat(), alpha.toLong(),
                     to_vtype(self.scalar_type()), to_vtype(idx_c.scalar_type()));
    TORCH_CHECK_INDEX(!read_scalar_val<int32_t>(invalid), "index_add_ index out of bounds");
    if (!direct) self.copy_(work);
    return self;
}

at::Tensor index_add_impl(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                          const at::Tensor& source, const at::Scalar& alpha) {
    auto out = self.clone();
    return index_add_inplace(out, dim, index, source, alpha);
}

at::Tensor& index_add_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                          const at::Tensor& source, const at::Scalar& alpha, at::Tensor& out) {
    at::native::resize_output(out, self.sizes());
    out.copy_(self);
    return index_add_inplace(out, dim, index, source, alpha);
}

at::Tensor repeat_interleave_tensor_impl(const at::Tensor& repeats, std::optional<int64_t> output_size) {
    check_iris(repeats, "repeat_interleave");
    TORCH_CHECK(repeats.dim() == 1, "repeat_interleave only accepts 1D vector of repeats");
    TORCH_CHECK(repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
                "repeats must have int64 or int32 dtype");
    const int64_t n = repeats.size(0);
    if (n == 0) {
        return at::empty({0}, repeats.options().dtype(at::kLong));
    }
    const auto r_long = packed0(repeats.to(at::kLong));
    TORCH_CHECK(!r_long.lt(0).any().item<bool>(), "repeats can not be negative");
    auto ends = r_long.cumsum(0);
    const int64_t total = output_size.has_value() ? *output_size : ends[-1].item<int64_t>();
    auto out = at::empty({total}, repeats.options().dtype(at::kLong));
    if (total > 0) {
        repeat_interleave_fill_launch(buf(out), buf(ends),
                                      static_cast<cl_ulong>(total), static_cast<cl_ulong>(n));
    }
    return out;
}

at::Tensor& repeat_interleave_tensor_out(const at::Tensor& repeats, std::optional<int64_t> output_size,
                                         at::Tensor& out) {
    auto res = repeat_interleave_tensor_impl(repeats, output_size);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor roll_impl(const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims) {
    check_iris(self, "roll");
    TORCH_CHECK(!shifts.empty(), "`shifts` required");
    auto out = at::empty(self.sizes(), self.options());
    if (!out.numel()) return out;
    const auto x = packed0(self);
    const auto type = to_vtype(self.scalar_type());
    if (dims.empty()) {
        TORCH_CHECK(shifts.size() == 1, "roll: if dims is empty, shifts must have length 1");
        roll_launch(buf(out), buf(x), {}, {}, static_cast<cl_ulong>(out.numel()), type, true, shifts[0]);
        return out;
    }
    TORCH_CHECK(shifts.size() == dims.size(), "shifts and dimensions must align");
    std::vector<cl_ulong> sizes(self.dim());
    std::vector<cl_long> dim_shifts(self.dim(), 0);
    for (int64_t d = 0; d < self.dim(); ++d) sizes[d] = static_cast<cl_ulong>(self.size(d));
    for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t d = at::maybe_wrap_dim(dims[i], self.dim());
        dim_shifts[d] += shifts[i];
    }
    roll_launch(buf(out), buf(x), sizes, dim_shifts, static_cast<cl_ulong>(out.numel()), type, false, 0);
    return out;
}

at::Tensor& roll_out(const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims, at::Tensor& out) {
    auto res = roll_impl(self, shifts, dims);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

// =========================================================================
// 2. Padding & Activations (+ Backward)
// =========================================================================

at::Tensor constant_pad_nd_impl(const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value) {
    check_iris(self, "constant_pad_nd");
    TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even");
    const int64_t l_pad = static_cast<int64_t>(pad.size()) / 2;
    const int64_t ndim = self.dim();
    TORCH_CHECK(l_pad <= ndim, "Length of pad should be no more than twice the number of dimensions");
    at::DimVector out_shape(self.sizes());
    std::vector<cl_ulong> out_u(ndim), in_u(ndim);
    std::vector<cl_long> pad_before(ndim, 0);
    for (int64_t d = 0; d < ndim; ++d) {
        in_u[d] = static_cast<cl_ulong>(self.size(d));
        out_u[d] = in_u[d];
    }
    for (int64_t i = 0; i < l_pad; ++i) {
        const int64_t d = ndim - 1 - i;
        const int64_t pl = pad[2 * i];
        const int64_t pr = pad[2 * i + 1];
        const int64_t ns = self.size(d) + pl + pr;
        TORCH_CHECK(ns >= 0, "constant_pad_nd: resulting dimension size must be non-negative");
        out_shape[d] = ns;
        out_u[d] = static_cast<cl_ulong>(ns);
        pad_before[d] = pl;
    }
    auto out = at::empty(out_shape, self.options());
    if (out.numel()) {
        const auto x = packed0(self);
        const auto type = to_vtype(self.scalar_type());
        const bool is_fp = (self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf);
        constant_pad_nd_launch(buf(out), buf(x), out_u, in_u, pad_before,
                               static_cast<cl_ulong>(out.numel()), type,
                               is_fp ? value.toFloat() : 0.0f,
                               is_fp ? 0 : value.toLong());
    }
    return out;
}

at::Tensor& constant_pad_nd_out(const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value, at::Tensor& out) {
    auto res = constant_pad_nd_impl(self, pad, value);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

template <int mode>
at::Tensor pad2d_mode_impl(const at::Tensor& self, at::IntArrayRef padding) {
    check_iris_fp(self, mode == 0 ? "reflection_pad2d" : "replication_pad2d");
    TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "Expected 3D or 4D input");
    const int64_t pl = padding[0], pr = padding[1], pt = padding[2], pb = padding[3];
    const bool is_3d = (self.dim() == 3);
    const int64_t n = is_3d ? 1 : self.size(0);
    const int64_t c = is_3d ? self.size(0) : self.size(1);
    const int64_t h = self.size(-2), w = self.size(-1);
    const int64_t oh = h + pt + pb, ow = w + pl + pr;
    TORCH_CHECK(oh > 0 && ow > 0, "padded output dimensions must be positive");
    if (mode == 0) {
        TORCH_CHECK(pl < w && pr < w && pt < h && pb < h,
                    "Padding size should be less than the corresponding input dimension");
    }
    at::DimVector out_shape = is_3d ? at::DimVector{c, oh, ow} : at::DimVector{n, c, oh, ow};
    auto out = at::empty(out_shape, self.options());
    if (out.numel()) {
        const auto x = packed0(self);
        pad2d_mode_fwd_launch(buf(out), buf(x), static_cast<cl_int>(n * c),
                              static_cast<cl_int>(h), static_cast<cl_int>(w),
                              static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                              static_cast<cl_int>(pl), static_cast<cl_int>(pt),
                              mode, self.scalar_type() == at::kHalf);
    }
    return out;
}

template <int mode>
at::Tensor& pad2d_mode_out(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out) {
    auto res = pad2d_mode_impl<mode>(self, padding);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

template <int mode>
at::Tensor pad2d_mode_bwd_impl(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding) {
    check_iris_fp(grad_output, "pad2d_backward");
    check_iris_fp(self, "pad2d_backward");
    const int64_t pl = padding[0], pr = padding[1], pt = padding[2], pb = padding[3];
    const bool is_3d = (self.dim() == 3);
    const int64_t nc = is_3d ? self.size(0) : (self.size(0) * self.size(1));
    const int64_t h = self.size(-2), w = self.size(-1);
    const int64_t oh = grad_output.size(-2), ow = grad_output.size(-1);
    auto gi = at::empty(self.sizes(), self.options());
    if (gi.numel()) {
        const auto go = packed0(grad_output);
        pad2d_mode_bwd_launch(buf(gi), buf(go), static_cast<cl_int>(nc),
                              static_cast<cl_int>(h), static_cast<cl_int>(w),
                              static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                              static_cast<cl_int>(pl), static_cast<cl_int>(pr),
                              static_cast<cl_int>(pt), static_cast<cl_int>(pb),
                              mode, self.scalar_type() == at::kHalf);
    }
    return gi;
}

template <int mode>
at::Tensor& pad2d_mode_bwd_out(const at::Tensor& grad_output, const at::Tensor& self,
                               at::IntArrayRef padding, at::Tensor& grad_input) {
    auto res = pad2d_mode_bwd_impl<mode>(grad_output, self, padding);
    at::native::resize_output(grad_input, res.sizes());
    grad_input.copy_(res);
    return grad_input;
}

template <int mode>
at::Tensor pad1d_mode_impl(const at::Tensor& self, at::IntArrayRef padding) {
    TORCH_CHECK(padding.size() == 2, "padding size is expected to be 2");
    TORCH_CHECK(self.dim() == 2 || self.dim() == 3, "Expected 2D or 3D input");
    auto res = pad2d_mode_impl<mode>(self.unsqueeze(-2), {padding[0], padding[1], 0, 0});
    return res.squeeze(-2);
}

template <int mode>
at::Tensor& pad1d_mode_out(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out) {
    auto res = pad1d_mode_impl<mode>(self, padding);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

template <int mode>
at::Tensor pad1d_mode_bwd_impl(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding) {
    auto res = pad2d_mode_bwd_impl<mode>(grad_output.unsqueeze(-2), self.unsqueeze(-2), {padding[0], padding[1], 0, 0});
    return res.squeeze(-2);
}

template <int mode>
at::Tensor& pad1d_mode_bwd_out(const at::Tensor& grad_output, const at::Tensor& self,
                               at::IntArrayRef padding, at::Tensor& grad_input) {
    auto res = pad1d_mode_bwd_impl<mode>(grad_output, self, padding);
    at::native::resize_output(grad_input, res.sizes());
    grad_input.copy_(res);
    return grad_input;
}

// Activations
at::Tensor run_act_fwd(const at::Tensor& self, cl_int act, float p0, float p1, float p2) {
    check_iris_fp(self, "activation");
    auto out = at::empty(self.sizes(), self.options());
    if (out.numel()) {
        const auto x = packed0(self);
        act_p3_fwd_launch(buf(out), buf(x), static_cast<cl_ulong>(out.numel()), act,
                          p0, p1, p2, self.scalar_type() == at::kHalf);
    }
    return out;
}

at::Tensor& run_act_inplace(at::Tensor& self, cl_int act, float p0, float p1, float p2) {
    check_iris_fp(self, "activation_");
    at::assert_no_internal_overlap(self);
    if (!self.numel()) return self;
    if (self.is_contiguous() && self.storage_offset() == 0) {
        act_p3_fwd_launch(buf(self), buf(self), static_cast<cl_ulong>(self.numel()), act,
                          p0, p1, p2, self.scalar_type() == at::kHalf);
    } else {
        auto res = run_act_fwd(self, act, p0, p1, p2);
        self.copy_(res);
    }
    return self;
}

at::Tensor run_act_bwd(const at::Tensor& grad_output, const at::Tensor& self_or_res,
                       cl_int act, float p0, float p1, float p2, bool is_result) {
    check_iris_fp(grad_output, "activation_backward");
    check_iris_fp(self_or_res, "activation_backward");
    auto go = packed0(grad_output.expand(self_or_res.sizes()));
    auto x = packed0(self_or_res);
    auto gi = at::empty(self_or_res.sizes(), self_or_res.options());
    if (gi.numel()) {
        act_p3_bwd_launch(buf(gi), buf(go), buf(x), static_cast<cl_ulong>(gi.numel()), act,
                          p0, p1, p2, is_result, self_or_res.scalar_type() == at::kHalf);
    }
    return gi;
}

at::Tensor leaky_relu_impl(const at::Tensor& self, const at::Scalar& neg_slope) {
    return run_act_fwd(self, 0, neg_slope.toFloat(), 0.0f, 0.0f);
}
at::Tensor& leaky_relu_inplace(at::Tensor& self, const at::Scalar& neg_slope) {
    return run_act_inplace(self, 0, neg_slope.toFloat(), 0.0f, 0.0f);
}
at::Tensor& leaky_relu_out(const at::Tensor& self, const at::Scalar& neg_slope, at::Tensor& out) {
    auto res = leaky_relu_impl(self, neg_slope);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}
at::Tensor leaky_relu_bwd_impl(const at::Tensor& grad_output, const at::Tensor& self,
                               const at::Scalar& neg_slope, bool self_is_result) {
    return run_act_bwd(grad_output, self, 0, neg_slope.toFloat(), 0.0f, 0.0f, self_is_result);
}
at::Tensor& leaky_relu_bwd_out(const at::Tensor& grad_output, const at::Tensor& self,
                               const at::Scalar& neg_slope, bool self_is_result, at::Tensor& gi) {
    auto res = leaky_relu_bwd_impl(grad_output, self, neg_slope, self_is_result);
    at::native::resize_output(gi, res.sizes());
    gi.copy_(res);
    return gi;
}

at::Tensor elu_impl(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {
    return run_act_fwd(self, 1, alpha.toFloat(), scale.toFloat(), input_scale.toFloat());
}
at::Tensor& elu_inplace(at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {
    return run_act_inplace(self, 1, alpha.toFloat(), scale.toFloat(), input_scale.toFloat());
}
at::Tensor& elu_out(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale, at::Tensor& out) {
    auto res = elu_impl(self, alpha, scale, input_scale);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}
at::Tensor elu_bwd_impl(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale,
                        const at::Scalar& input_scale, bool is_result, const at::Tensor& self_or_result) {
    return run_act_bwd(grad_output, self_or_result, 1, alpha.toFloat(), scale.toFloat(), input_scale.toFloat(), is_result);
}
at::Tensor& elu_bwd_out(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale,
                        const at::Scalar& input_scale, bool is_result, const at::Tensor& self_or_result, at::Tensor& gi) {
    auto res = elu_bwd_impl(grad_output, alpha, scale, input_scale, is_result, self_or_result);
    at::native::resize_output(gi, res.sizes());
    gi.copy_(res);
    return gi;
}

at::Tensor hardswish_impl(const at::Tensor& self) { return run_act_fwd(self, 2, 0, 0, 0); }
at::Tensor& hardswish_inplace(at::Tensor& self) { return run_act_inplace(self, 2, 0, 0, 0); }
at::Tensor& hardswish_out(const at::Tensor& self, at::Tensor& out) {
    auto res = hardswish_impl(self); at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}
at::Tensor hardswish_bwd_impl(const at::Tensor& grad_output, const at::Tensor& self) {
    return run_act_bwd(grad_output, self, 2, 0, 0, 0, false);
}
at::Tensor& hardswish_bwd_out(const at::Tensor& grad_output, const at::Tensor& self, at::Tensor& out) {
    auto res = hardswish_bwd_impl(grad_output, self); at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}

at::Tensor hardsigmoid_impl(const at::Tensor& self) { return run_act_fwd(self, 3, 0, 0, 0); }
at::Tensor& hardsigmoid_inplace(at::Tensor& self) { return run_act_inplace(self, 3, 0, 0, 0); }
at::Tensor& hardsigmoid_out(const at::Tensor& self, at::Tensor& out) {
    auto res = hardsigmoid_impl(self); at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}
at::Tensor hardsigmoid_bwd_impl(const at::Tensor& grad_output, const at::Tensor& self) {
    return run_act_bwd(grad_output, self, 3, 0, 0, 0, false);
}
at::Tensor& hardsigmoid_bwd_out(const at::Tensor& grad_output, const at::Tensor& self, at::Tensor& gi) {
    auto res = hardsigmoid_bwd_impl(grad_output, self); at::native::resize_output(gi, res.sizes()); gi.copy_(res); return gi;
}

at::Tensor mish_impl(const at::Tensor& self) { return run_act_fwd(self, 4, 0, 0, 0); }
at::Tensor& mish_inplace(at::Tensor& self) { return run_act_inplace(self, 4, 0, 0, 0); }
at::Tensor& mish_out(const at::Tensor& self, at::Tensor& out) {
    auto res = mish_impl(self); at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}
at::Tensor mish_bwd_impl(const at::Tensor& grad_output, const at::Tensor& self) {
    return run_act_bwd(grad_output, self, 4, 0, 0, 0, false);
}

at::Tensor hardtanh_impl(const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val) {
    return run_act_fwd(self, 5, min_val.toFloat(), max_val.toFloat(), 0.0f);
}
at::Tensor& hardtanh_inplace(at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val) {
    return run_act_inplace(self, 5, min_val.toFloat(), max_val.toFloat(), 0.0f);
}
at::Tensor& hardtanh_out(const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val, at::Tensor& out) {
    auto res = hardtanh_impl(self, min_val, max_val); at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}
at::Tensor hardtanh_bwd_impl(const at::Tensor& grad_output, const at::Tensor& self,
                             const at::Scalar& min_val, const at::Scalar& max_val) {
    return run_act_bwd(grad_output, self, 5, min_val.toFloat(), max_val.toFloat(), 0.0f, false);
}
at::Tensor& hardtanh_bwd_out(const at::Tensor& grad_output, const at::Tensor& self,
                             const at::Scalar& min_val, const at::Scalar& max_val, at::Tensor& gi) {
    auto res = hardtanh_bwd_impl(grad_output, self, min_val, max_val);
    at::native::resize_output(gi, res.sizes());
    gi.copy_(res);
    return gi;
}

std::tuple<int64_t, int64_t, int64_t, bool> prelu_dims(const at::Tensor& self, const at::Tensor& weight) {
    const bool single = (weight.numel() == 1);
    if (single || self.dim() == 0) {
        return {1, 1, self.numel(), true};
    }
    const int64_t c = (self.dim() >= 2) ? self.size(1) : self.size(0);
    const int64_t outer = (self.dim() >= 2) ? self.size(0) : 1;
    int64_t inner = 1;
    for (int64_t d = 2; d < self.dim(); ++d) inner *= self.size(d);
    return {outer, c, inner, false};
}

at::Tensor prelu_kernel_impl(const at::Tensor& self, const at::Tensor& weight) {
    check_iris_fp(self, "_prelu_kernel");
    check_iris_fp(weight, "_prelu_kernel");
    auto out = at::empty(self.sizes(), self.options());
    if (out.numel()) {
        const auto x = packed0(self);
        const auto w = packed0(weight.to(self.scalar_type()));
        auto [outer, c, inner, single] = prelu_dims(self, weight);
        prelu_fwd_launch(buf(out), buf(x), buf(w), static_cast<cl_ulong>(out.numel()),
                         static_cast<cl_ulong>(c), static_cast<cl_ulong>(inner),
                         single, self.scalar_type() == at::kHalf);
    }
    return out;
}

std::tuple<at::Tensor, at::Tensor> prelu_kernel_bwd_impl(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight) {
    check_iris_fp(grad_output, "_prelu_kernel_backward");
    check_iris_fp(self, "_prelu_kernel_backward");
    check_iris_fp(weight, "_prelu_kernel_backward");
    auto dx = at::empty(self.sizes(), self.options());
    auto dw = at::zeros(weight.sizes(), weight.options().dtype(self.scalar_type()));
    if (dx.numel()) {
        const auto go = packed0(grad_output);
        const auto x = packed0(self);
        const auto w = packed0(weight.to(self.scalar_type()));
        auto [outer, c, inner, single] = prelu_dims(self, weight);
        prelu_bwd_launch(buf(dx), buf(dw), buf(go), buf(x), buf(w),
                         static_cast<cl_ulong>(outer), static_cast<cl_ulong>(c),
                         static_cast<cl_ulong>(inner), single, self.scalar_type() == at::kHalf);
    }
    return {dx, dw.scalar_type() == weight.scalar_type() ? dw : dw.to(weight.scalar_type())};
}

// =========================================================================
// 3. GridSampler 2D (+ Backward)
// =========================================================================

at::Tensor grid_sampler_2d_impl(const at::Tensor& input, const at::Tensor& grid,
                                int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    check_iris_fp(input, "grid_sampler_2d");
    check_iris_fp(grid, "grid_sampler_2d");
    TORCH_CHECK(input.dim() == 4 && grid.dim() == 4 && grid.size(3) == 2,
                "grid_sampler_2d expects 4D input and [N, OH, OW, 2] grid");
    TORCH_CHECK(input.size(0) == grid.size(0), "grid_sampler_2d batch mismatch");
    const int64_t n = input.size(0), c = input.size(1), ih = input.size(2), iw = input.size(3);
    const int64_t oh = grid.size(1), ow = grid.size(2);
    auto out = at::empty({n, c, oh, ow}, input.options());
    if (out.numel()) {
        const auto x = packed0(input);
        const auto g = packed0(grid.to(input.scalar_type()));
        grid_sampler_2d_fwd_launch(buf(out), buf(x), buf(g),
                                   static_cast<cl_int>(n), static_cast<cl_int>(c),
                                   static_cast<cl_int>(ih), static_cast<cl_int>(iw),
                                   static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                                   static_cast<cl_int>(interpolation_mode),
                                   static_cast<cl_int>(padding_mode),
                                   align_corners, input.scalar_type() == at::kHalf);
    }
    return out;
}

at::Tensor& grid_sampler_2d_out(const at::Tensor& input, const at::Tensor& grid,
                                int64_t interpolation_mode, int64_t padding_mode,
                                bool align_corners, at::Tensor& out) {
    auto res = grid_sampler_2d_impl(input, grid, interpolation_mode, padding_mode, align_corners);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

std::tuple<at::Tensor, at::Tensor> grid_sampler_2d_bwd_impl(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
    std::array<bool, 2> output_mask) {
    check_iris_fp(grad_output, "grid_sampler_2d_backward");
    check_iris_fp(input, "grid_sampler_2d_backward");
    check_iris_fp(grid, "grid_sampler_2d_backward");
    const bool need_gi = output_mask[0];
    const bool need_gg = output_mask[1];
    const int64_t n = input.size(0), c = input.size(1), ih = input.size(2), iw = input.size(3);
    const int64_t oh = grid.size(1), ow = grid.size(2);
    const bool is_fp16 = (input.scalar_type() == at::kHalf);

    at::Tensor gi_f32 = need_gi ? at::zeros(input.sizes(), input.options().dtype(at::kFloat)) : at::Tensor{};
    at::Tensor gg = need_gg ? at::zeros(grid.sizes(), grid.options().dtype(input.scalar_type())) : at::Tensor{};
    if ((need_gi || need_gg) && grad_output.numel() && input.numel()) {
        const auto go = packed0(grad_output.to(input.scalar_type()));
        const auto x = packed0(input);
        const auto g = packed0(grid.to(input.scalar_type()));
        grid_sampler_2d_bwd_launch(buf(gi_f32), buf(gg), buf(go), buf(x), buf(g),
                                   static_cast<cl_int>(n), static_cast<cl_int>(c),
                                   static_cast<cl_int>(ih), static_cast<cl_int>(iw),
                                   static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                                   static_cast<cl_int>(interpolation_mode),
                                   static_cast<cl_int>(padding_mode),
                                   align_corners, need_gi, need_gg, is_fp16);
    }
    at::Tensor gi = need_gi ? (is_fp16 ? gi_f32.to(at::kHalf) : gi_f32) : at::Tensor{};
    if (need_gg && gg.scalar_type() != grid.scalar_type()) gg = gg.to(grid.scalar_type());
    return {gi, gg};
}

std::tuple<at::Tensor&, at::Tensor&> grid_sampler_2d_bwd_out(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
    std::array<bool, 2> output_mask, at::Tensor& out0, at::Tensor& out1) {
    auto [gi, gg] = grid_sampler_2d_bwd_impl(grad_output, input, grid, interpolation_mode,
                                             padding_mode, align_corners, output_mask);
    if (output_mask[0]) { at::native::resize_output(out0, gi.sizes()); out0.copy_(gi); }
    if (output_mask[1]) { at::native::resize_output(out1, gg.sizes()); out1.copy_(gg); }
    return {out0, out1};
}

// =========================================================================
// 4. 3D Pooling (MaxPool3d & AvgPool3d + Backward)
// =========================================================================

int64_t pool_out_dim(int64_t in_sz, int64_t k, int64_t s, int64_t p, int64_t d, bool ceil_mode) {
    int64_t out = 0;
    if (ceil_mode) {
        out = (in_sz + 2 * p - d * (k - 1) - 1 + s - 1) / s + 1;
        if ((out - 1) * s >= in_sz + p) --out;
    } else {
        out = (in_sz + 2 * p - d * (k - 1) - 1) / s + 1;
    }
    TORCH_CHECK(out > 0, "pool3d output size must be positive");
    return out;
}

std::tuple<at::Tensor, at::Tensor> max_pool3d_impl(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    check_iris_fp(self, "max_pool3d");
    TORCH_CHECK(self.dim() == 4 || self.dim() == 5, "max_pool3d expects 4D or 5D input");
    const bool is_4d = (self.dim() == 4);
    const int64_t kd = kernel_size[0], kh = kernel_size.size() > 1 ? kernel_size[1] : kd, kw = kernel_size.size() > 2 ? kernel_size[2] : kh;
    const int64_t sd = stride.empty() ? kd : stride[0], sh = stride.empty() ? kh : (stride.size() > 1 ? stride[1] : sd), sw = stride.empty() ? kw : (stride.size() > 2 ? stride[2] : sh);
    const int64_t pd = padding[0], ph = padding.size() > 1 ? padding[1] : pd, pw = padding.size() > 2 ? padding[2] : ph;
    const int64_t dd = dilation[0], dh = dilation.size() > 1 ? dilation[1] : dd, dw = dilation.size() > 2 ? dilation[2] : dh;
    const int64_t n = is_4d ? 1 : self.size(0);
    const int64_t c = is_4d ? self.size(0) : self.size(1);
    const int64_t id = self.size(-3), ih = self.size(-2), iw = self.size(-1);
    const int64_t od = pool_out_dim(id, kd, sd, pd, dd, ceil_mode);
    const int64_t oh = pool_out_dim(ih, kh, sh, ph, dh, ceil_mode);
    const int64_t ow = pool_out_dim(iw, kw, sw, pw, dw, ceil_mode);
    at::DimVector out_shape = is_4d ? at::DimVector{c, od, oh, ow} : at::DimVector{n, c, od, oh, ow};
    auto out = at::empty(out_shape, self.options());
    auto idxs = at::empty(out_shape, self.options().dtype(at::kLong));
    if (out.numel()) {
        const auto x = packed0(self);
        Pool3dParams p{static_cast<cl_int>(n * c), static_cast<cl_int>(id), static_cast<cl_int>(ih), static_cast<cl_int>(iw),
                       static_cast<cl_int>(od), static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                       static_cast<cl_int>(kd), static_cast<cl_int>(kh), static_cast<cl_int>(kw),
                       static_cast<cl_int>(sd), static_cast<cl_int>(sh), static_cast<cl_int>(sw),
                       static_cast<cl_int>(pd), static_cast<cl_int>(ph), static_cast<cl_int>(pw),
                       static_cast<cl_int>(dd), static_cast<cl_int>(dh), static_cast<cl_int>(dw), 0, 0};
        max_pool3d_fwd_launch(buf(out), buf(idxs), buf(x), p, self.scalar_type() == at::kHalf);
    }
    return {out, idxs};
}

std::tuple<at::Tensor&, at::Tensor&> max_pool3d_out(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices) {
    auto [o, i] = max_pool3d_impl(self, kernel_size, stride, padding, dilation, ceil_mode);
    at::native::resize_output(out, o.sizes()); out.copy_(o);
    at::native::resize_output(indices, i.sizes()); indices.copy_(i);
    return {out, indices};
}

at::Tensor max_pool3d_bwd_impl(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef /*kernel_size*/, at::IntArrayRef /*stride*/,
    at::IntArrayRef /*padding*/, at::IntArrayRef /*dilation*/,
    bool /*ceil_mode*/, const at::Tensor& indices) {
    check_iris_fp(grad_output, "max_pool3d_backward");
    check_iris_fp(self, "max_pool3d_backward");
    const bool is_4d = (self.dim() == 4);
    const int64_t nc = is_4d ? self.size(0) : (self.size(0) * self.size(1));
    const bool is_fp16 = (self.scalar_type() == at::kHalf);
    auto gi_f32 = at::zeros(self.sizes(), self.options().dtype(at::kFloat));
    if (grad_output.numel()) {
        const auto go = packed0(grad_output);
        const auto idx = packed0(indices);
        Pool3dParams p{static_cast<cl_int>(nc), static_cast<cl_int>(self.size(-3)), static_cast<cl_int>(self.size(-2)), static_cast<cl_int>(self.size(-1)),
                       static_cast<cl_int>(grad_output.size(-3)), static_cast<cl_int>(grad_output.size(-2)), static_cast<cl_int>(grad_output.size(-1)),
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        max_pool3d_bwd_launch(buf(gi_f32), buf(go), buf(idx), p, is_fp16);
    }
    return is_fp16 ? gi_f32.to(at::kHalf) : gi_f32;
}

at::Tensor& max_pool3d_bwd_out(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode, const at::Tensor& indices, at::Tensor& grad_input) {
    auto res = max_pool3d_bwd_impl(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
    at::native::resize_output(grad_input, res.sizes());
    grad_input.copy_(res);
    return grad_input;
}

at::Tensor avg_pool3d_impl(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    check_iris_fp(self, "avg_pool3d");
    TORCH_CHECK(self.dim() == 4 || self.dim() == 5, "avg_pool3d expects 4D or 5D input");
    const bool is_4d = (self.dim() == 4);
    const int64_t kd = kernel_size[0], kh = kernel_size.size() > 1 ? kernel_size[1] : kd, kw = kernel_size.size() > 2 ? kernel_size[2] : kh;
    const int64_t sd = stride.empty() ? kd : stride[0], sh = stride.empty() ? kh : (stride.size() > 1 ? stride[1] : sd), sw = stride.empty() ? kw : (stride.size() > 2 ? stride[2] : sh);
    const int64_t pd = padding[0], ph = padding.size() > 1 ? padding[1] : pd, pw = padding.size() > 2 ? padding[2] : ph;
    const int64_t n = is_4d ? 1 : self.size(0);
    const int64_t c = is_4d ? self.size(0) : self.size(1);
    const int64_t id = self.size(-3), ih = self.size(-2), iw = self.size(-1);
    const int64_t od = pool_out_dim(id, kd, sd, pd, 1, ceil_mode);
    const int64_t oh = pool_out_dim(ih, kh, sh, ph, 1, ceil_mode);
    const int64_t ow = pool_out_dim(iw, kw, sw, pw, 1, ceil_mode);
    at::DimVector out_shape = is_4d ? at::DimVector{c, od, oh, ow} : at::DimVector{n, c, od, oh, ow};
    auto out = at::empty(out_shape, self.options());
    if (out.numel()) {
        const auto x = packed0(self);
        Pool3dParams p{static_cast<cl_int>(n * c), static_cast<cl_int>(id), static_cast<cl_int>(ih), static_cast<cl_int>(iw),
                       static_cast<cl_int>(od), static_cast<cl_int>(oh), static_cast<cl_int>(ow),
                       static_cast<cl_int>(kd), static_cast<cl_int>(kh), static_cast<cl_int>(kw),
                       static_cast<cl_int>(sd), static_cast<cl_int>(sh), static_cast<cl_int>(sw),
                       static_cast<cl_int>(pd), static_cast<cl_int>(ph), static_cast<cl_int>(pw),
                       1, 1, 1, count_include_pad ? 1 : 0, static_cast<cl_int>(divisor_override.value_or(0))};
        avg_pool3d_fwd_launch(buf(out), buf(x), p, self.scalar_type() == at::kHalf);
    }
    return out;
}

at::Tensor& avg_pool3d_out(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& out) {
    auto res = avg_pool3d_impl(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor avg_pool3d_bwd_impl(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool /*ceil_mode*/, bool count_include_pad, std::optional<int64_t> divisor_override) {
    check_iris_fp(grad_output, "avg_pool3d_backward");
    check_iris_fp(self, "avg_pool3d_backward");
    const bool is_4d = (self.dim() == 4);
    const int64_t kd = kernel_size[0], kh = kernel_size.size() > 1 ? kernel_size[1] : kd, kw = kernel_size.size() > 2 ? kernel_size[2] : kh;
    const int64_t sd = stride.empty() ? kd : stride[0], sh = stride.empty() ? kh : (stride.size() > 1 ? stride[1] : sd), sw = stride.empty() ? kw : (stride.size() > 2 ? stride[2] : sh);
    const int64_t pd = padding[0], ph = padding.size() > 1 ? padding[1] : pd, pw = padding.size() > 2 ? padding[2] : ph;
    const int64_t nc = is_4d ? self.size(0) : (self.size(0) * self.size(1));
    auto gi = at::empty(self.sizes(), self.options());
    if (gi.numel()) {
        const auto go = packed0(grad_output);
        Pool3dParams p{static_cast<cl_int>(nc), static_cast<cl_int>(self.size(-3)), static_cast<cl_int>(self.size(-2)), static_cast<cl_int>(self.size(-1)),
                       static_cast<cl_int>(grad_output.size(-3)), static_cast<cl_int>(grad_output.size(-2)), static_cast<cl_int>(grad_output.size(-1)),
                       static_cast<cl_int>(kd), static_cast<cl_int>(kh), static_cast<cl_int>(kw),
                       static_cast<cl_int>(sd), static_cast<cl_int>(sh), static_cast<cl_int>(sw),
                       static_cast<cl_int>(pd), static_cast<cl_int>(ph), static_cast<cl_int>(pw),
                       1, 1, 1, count_include_pad ? 1 : 0, static_cast<cl_int>(divisor_override.value_or(0))};
        avg_pool3d_bwd_launch(buf(gi), buf(go), p, self.scalar_type() == at::kHalf);
    }
    return gi;
}

at::Tensor& avg_pool3d_bwd_out(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& grad_input) {
    auto res = avg_pool3d_bwd_impl(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    at::native::resize_output(grad_input, res.sizes());
    grad_input.copy_(res);
    return grad_input;
}

// =========================================================================
// 5. Reductions (var, std, var_mean, std_mean, prod, linalg_vector_norm) & Ternary Ops
// =========================================================================

struct ReducePrep {
    at::Tensor contiguous_in;
    at::DimVector out_shape;
    int64_t outer = 1;
    int64_t reduce_len = 1;
};

ReducePrep prepare_reduction(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim) {
    const int64_t ndim = self.dim();
    std::vector<bool> reduced(ndim, false);
    if (!dim.has_value() || dim->empty()) {
        std::fill(reduced.begin(), reduced.end(), true);
    } else {
        for (int64_t d : *dim) {
            d = at::maybe_wrap_dim(d, ndim);
            TORCH_CHECK(!reduced[d], "dim ", d, " appears multiple times in reduction");
            reduced[d] = true;
        }
    }
    ReducePrep rp;
    std::vector<int64_t> perm;
    for (int64_t d = 0; d < ndim; ++d) {
        if (!reduced[d]) {
            perm.push_back(d);
            rp.outer *= self.size(d);
            rp.out_shape.push_back(self.size(d));
        } else if (keepdim) {
            rp.out_shape.push_back(1);
        }
    }
    for (int64_t d = 0; d < ndim; ++d) {
        if (reduced[d]) {
            perm.push_back(d);
            rp.reduce_len *= self.size(d);
        }
    }
    bool identity = true;
    for (int64_t i = 0; i < ndim; ++i) {
        if (perm[i] != i) { identity = false; break; }
    }
    rp.contiguous_in = packed0(identity ? self : self.permute(perm));
    return rp;
}

std::tuple<at::Tensor, at::Tensor> var_std_core(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<at::Scalar>& correction, bool keepdim,
    bool take_sqrt, bool need_mean) {
    check_iris_fp(self, take_sqrt ? "std" : "var");
    const float corr = correction.has_value() ? correction->toFloat() : 1.0f;
    auto rp = prepare_reduction(self, dim, keepdim);
    auto v_out = at::empty(rp.out_shape, self.options());
    auto m_out = need_mean ? at::empty(rp.out_shape, self.options()) : at::Tensor{};
    if (v_out.numel()) {
        var_std_reduce_launch(buf(v_out), buf(m_out), buf(rp.contiguous_in),
                              static_cast<cl_ulong>(rp.outer), static_cast<cl_ulong>(rp.reduce_len),
                              corr, take_sqrt, need_mean, self.scalar_type() == at::kHalf);
    }
    return {v_out, m_out};
}

at::Tensor var_correction(const at::Tensor& self, at::OptionalIntArrayRef dim,
                          const std::optional<at::Scalar>& correction, bool keepdim) {
    return std::get<0>(var_std_core(self, dim, correction, keepdim, false, false));
}
at::Tensor& var_correction_out(const at::Tensor& self, at::OptionalIntArrayRef dim,
                               const std::optional<at::Scalar>& correction, bool keepdim, at::Tensor& out) {
    auto res = var_correction(self, dim, correction, keepdim);
    at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}

at::Tensor std_correction(const at::Tensor& self, at::OptionalIntArrayRef dim,
                          const std::optional<at::Scalar>& correction, bool keepdim) {
    return std::get<0>(var_std_core(self, dim, correction, keepdim, true, false));
}
at::Tensor& std_correction_out(const at::Tensor& self, at::OptionalIntArrayRef dim,
                               const std::optional<at::Scalar>& correction, bool keepdim, at::Tensor& out) {
    auto res = std_correction(self, dim, correction, keepdim);
    at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}

std::tuple<at::Tensor, at::Tensor> var_mean_correction(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<at::Scalar>& correction, bool keepdim) {
    return var_std_core(self, dim, correction, keepdim, false, true);
}

std::tuple<at::Tensor, at::Tensor> std_mean_correction(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<at::Scalar>& correction, bool keepdim) {
    return var_std_core(self, dim, correction, keepdim, true, true);
}

at::Tensor prod_core(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                     std::optional<at::ScalarType> dtype) {
    check_iris(self, "prod");
    auto out_dt = dtype.value_or(
        (self.scalar_type() == at::kBool || self.scalar_type() == at::kByte || self.scalar_type() == at::kInt)
            ? at::kLong : self.scalar_type());
    auto rp = prepare_reduction(self, dim, keepdim);
    auto out = at::empty(rp.out_shape, self.options().dtype(out_dt));
    if (out.numel()) {
        prod_reduce_launch(buf(out), buf(rp.contiguous_in),
                           static_cast<cl_ulong>(rp.outer), static_cast<cl_ulong>(rp.reduce_len),
                           to_vtype(self.scalar_type()), to_vtype(out_dt));
    }
    return out;
}

at::Tensor prod_all(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
    return prod_core(self, std::nullopt, false, dtype);
}
at::Tensor& prod_all_out(const at::Tensor& self, std::optional<at::ScalarType> dtype, at::Tensor& out) {
    auto res = prod_all(self, dtype.value_or(out.scalar_type()));
    at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}
at::Tensor prod_dim_int(const at::Tensor& self, int64_t dim, bool keepdim, std::optional<at::ScalarType> dtype) {
    return prod_core(self, at::IntArrayRef(dim), keepdim, dtype);
}
at::Tensor& prod_int_out(const at::Tensor& self, int64_t dim, bool keepdim,
                         std::optional<at::ScalarType> dtype, at::Tensor& out) {
    auto res = prod_dim_int(self, dim, keepdim, dtype.value_or(out.scalar_type()));
    at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}

at::Tensor linalg_vector_norm_impl(
    const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype) {
    check_iris_fp(self, "linalg_vector_norm");
    const auto out_dt = dtype.value_or(self.scalar_type());
    TORCH_CHECK(out_dt == at::kFloat || out_dt == at::kHalf, "linalg_vector_norm supports float32/float16");
    const double ord_d = ord.toDouble();
    cl_int ord_mode = 5;
    if (ord_d == 2.0) ord_mode = 0;
    else if (ord_d == 1.0) ord_mode = 1;
    else if (ord_d == 0.0) ord_mode = 2;
    else if (std::isinf(ord_d) && ord_d > 0.0) ord_mode = 3;
    else if (std::isinf(ord_d) && ord_d < 0.0) ord_mode = 4;

    auto rp = prepare_reduction(self, dim, keepdim);
    auto out = at::empty(rp.out_shape, self.options().dtype(out_dt));
    if (out.numel()) {
        vector_norm_reduce_launch(buf(out), buf(rp.contiguous_in),
                                  static_cast<cl_ulong>(rp.outer), static_cast<cl_ulong>(rp.reduce_len),
                                  static_cast<float>(ord_d), ord_mode,
                                  self.scalar_type() == at::kHalf, out_dt == at::kHalf);
    }
    return out;
}

at::Tensor& linalg_vector_norm_out(
    const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype, at::Tensor& out) {
    auto res = linalg_vector_norm_impl(self, ord, dim, keepdim, dtype.value_or(out.scalar_type()));
    at::native::resize_output(out, res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor ternary_common(const at::Tensor& self, const at::Tensor& t1, const at::Tensor& t2,
                          cl_int op, float alpha) {
    check_iris_fp(self, "ternary");
    check_iris_fp(t1, "ternary");
    check_iris_fp(t2, "ternary");
    const auto dt = c10::promoteTypes(at::result_type(self, t1), t2.scalar_type());
    auto shape = at::infer_size(at::infer_size(self.sizes(), t1.sizes()), t2.sizes());
    auto out = at::empty(shape, self.options().dtype(dt));
    if (out.numel()) {
        const auto a = packed0(self.to(dt).expand(shape));
        const auto b = packed0(t1.to(dt).expand(shape));
        const auto c = packed0(t2.to(dt).expand(shape));
        ternary_contiguous_launch(buf(out), buf(a), buf(b), buf(c),
                                  static_cast<cl_ulong>(out.numel()), op, alpha, dt == at::kHalf);
    }
    return out;
}

at::Tensor lerp_tensor_impl(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
    return ternary_common(self, end, weight, 0, 1.0f);
}
at::Tensor& lerp_tensor_out(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight, at::Tensor& out) {
    auto res = lerp_tensor_impl(self, end, weight);
    at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}
at::Tensor& lerp_tensor_inplace(at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
    at::assert_no_internal_overlap(self);
    auto res = lerp_tensor_impl(self, end, weight);
    TORCH_CHECK(res.sizes() == self.sizes(), "lerp_ cannot change shape");
    self.copy_(res);
    return self;
}
at::Tensor lerp_scalar_impl(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
    auto w = at::empty({}, self.options());
    w.fill_(weight);
    return ternary_common(self, end, w, 0, 1.0f);
}
at::Tensor& lerp_scalar_out(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight, at::Tensor& out) {
    auto res = lerp_scalar_impl(self, end, weight);
    at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}

at::Tensor addcmul_impl(const at::Tensor& self, const at::Tensor& t1, const at::Tensor& t2, const at::Scalar& value) {
    return ternary_common(self, t1, t2, 1, value.toFloat());
}
at::Tensor& addcmul_out(const at::Tensor& self, const at::Tensor& t1, const at::Tensor& t2, const at::Scalar& value, at::Tensor& out) {
    auto res = addcmul_impl(self, t1, t2, value);
    at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}

at::Tensor addcdiv_impl(const at::Tensor& self, const at::Tensor& t1, const at::Tensor& t2, const at::Scalar& value) {
    return ternary_common(self, t1, t2, 2, value.toFloat());
}
at::Tensor& addcdiv_out(const at::Tensor& self, const at::Tensor& t1, const at::Tensor& t2, const at::Scalar& value, at::Tensor& out) {
    auto res = addcdiv_impl(self, t1, t2, value);
    at::native::resize_output(out, res.sizes()); out.copy_(res); return out;
}

bool equal_impl(const at::Tensor& self, const at::Tensor& other) {
    check_iris(self, "equal");
    check_iris(other, "equal");
    if (self.sizes() != other.sizes()) return false;
    if (self.scalar_type() != other.scalar_type()) return false;
    if (self.numel() == 0) return true;
    return self.eq(other).all().item<bool>();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("equal", &equal_impl);
    m.impl("multinomial", &multinomial_impl);
    m.impl("multinomial.out", &multinomial_out);
    m.impl("searchsorted.Tensor", &searchsorted_tensor_impl);
    m.impl("searchsorted.Tensor_out", &searchsorted_tensor_out);
    m.impl("searchsorted.Scalar", &searchsorted_scalar_impl);
    m.impl("searchsorted.Scalar_out", &searchsorted_scalar_out);
    m.impl("masked_scatter_", &masked_scatter_impl);
    m.impl("isin.Tensor_Tensor", &isin_tt_impl);
    m.impl("isin.Tensor_Tensor_out", &isin_tt_out);
    m.impl("isin.Tensor_Scalar", &isin_ts_impl);
    m.impl("isin.Tensor_Scalar_out", &isin_ts_out);
    m.impl("isin.Scalar_Tensor", &isin_st_impl);
    m.impl("isin.Scalar_Tensor_out", &isin_st_out);
    m.impl("index_add_", &index_add_inplace);
    m.impl("index_add", &index_add_impl);
    m.impl("index_add.out", &index_add_out);
    m.impl("repeat_interleave.Tensor", &repeat_interleave_tensor_impl);
    m.impl("repeat_interleave.Tensor_out", &repeat_interleave_tensor_out);
    m.impl("roll", &roll_impl);
    m.impl("roll.out", &roll_out);

    m.impl("constant_pad_nd", &constant_pad_nd_impl);
    m.impl("constant_pad_nd.out", &constant_pad_nd_out);
    m.impl("reflection_pad2d", &pad2d_mode_impl<0>);
    m.impl("reflection_pad2d.out", &pad2d_mode_out<0>);
    m.impl("reflection_pad2d_backward", &pad2d_mode_bwd_impl<0>);
    m.impl("reflection_pad2d_backward.grad_input", &pad2d_mode_bwd_out<0>);
    m.impl("replication_pad2d", &pad2d_mode_impl<1>);
    m.impl("replication_pad2d.out", &pad2d_mode_out<1>);
    m.impl("replication_pad2d_backward", &pad2d_mode_bwd_impl<1>);
    m.impl("replication_pad2d_backward.grad_input", &pad2d_mode_bwd_out<1>);
    m.impl("reflection_pad1d", &pad1d_mode_impl<0>);
    m.impl("reflection_pad1d.out", &pad1d_mode_out<0>);
    m.impl("reflection_pad1d_backward", &pad1d_mode_bwd_impl<0>);
    m.impl("reflection_pad1d_backward.grad_input", &pad1d_mode_bwd_out<0>);
    m.impl("replication_pad1d", &pad1d_mode_impl<1>);
    m.impl("replication_pad1d.out", &pad1d_mode_out<1>);
    m.impl("replication_pad1d_backward", &pad1d_mode_bwd_impl<1>);
    m.impl("replication_pad1d_backward.grad_input", &pad1d_mode_bwd_out<1>);

    m.impl("leaky_relu", &leaky_relu_impl);
    m.impl("leaky_relu_", &leaky_relu_inplace);
    m.impl("leaky_relu.out", &leaky_relu_out);
    m.impl("leaky_relu_backward", &leaky_relu_bwd_impl);
    m.impl("leaky_relu_backward.grad_input", &leaky_relu_bwd_out);
    m.impl("elu", &elu_impl);
    m.impl("elu_", &elu_inplace);
    m.impl("elu.out", &elu_out);
    m.impl("elu_backward", &elu_bwd_impl);
    m.impl("elu_backward.grad_input", &elu_bwd_out);
    m.impl("hardswish", &hardswish_impl);
    m.impl("hardswish_", &hardswish_inplace);
    m.impl("hardswish.out", &hardswish_out);
    m.impl("hardswish_backward", &hardswish_bwd_impl);
    m.impl("hardswish_backward.out", &hardswish_bwd_out);
    m.impl("hardsigmoid", &hardsigmoid_impl);
    m.impl("hardsigmoid_", &hardsigmoid_inplace);
    m.impl("hardsigmoid.out", &hardsigmoid_out);
    m.impl("hardsigmoid_backward", &hardsigmoid_bwd_impl);
    m.impl("hardsigmoid_backward.grad_input", &hardsigmoid_bwd_out);
    m.impl("mish", &mish_impl);
    m.impl("mish_", &mish_inplace);
    m.impl("mish.out", &mish_out);
    m.impl("mish_backward", &mish_bwd_impl);
    m.impl("hardtanh", &hardtanh_impl);
    m.impl("hardtanh_", &hardtanh_inplace);
    m.impl("hardtanh.out", &hardtanh_out);
    m.impl("hardtanh_backward", &hardtanh_bwd_impl);
    m.impl("hardtanh_backward.grad_input", &hardtanh_bwd_out);
    m.impl("_prelu_kernel", &prelu_kernel_impl);
    m.impl("_prelu_kernel_backward", &prelu_kernel_bwd_impl);

    m.impl("grid_sampler_2d", &grid_sampler_2d_impl);
    m.impl("grid_sampler_2d.out", &grid_sampler_2d_out);
    m.impl("grid_sampler_2d_backward", &grid_sampler_2d_bwd_impl);
    m.impl("grid_sampler_2d_backward.out", &grid_sampler_2d_bwd_out);

    m.impl("max_pool3d_with_indices", &max_pool3d_impl);
    m.impl("max_pool3d_with_indices.out", &max_pool3d_out);
    m.impl("max_pool3d_with_indices_backward", &max_pool3d_bwd_impl);
    m.impl("max_pool3d_with_indices_backward.grad_input", &max_pool3d_bwd_out);
    m.impl("avg_pool3d", &avg_pool3d_impl);
    m.impl("avg_pool3d.out", &avg_pool3d_out);
    m.impl("avg_pool3d_backward", &avg_pool3d_bwd_impl);
    m.impl("avg_pool3d_backward.grad_input", &avg_pool3d_bwd_out);

    m.impl("var.correction", &var_correction);
    m.impl("var.correction_out", &var_correction_out);
    m.impl("std.correction", &std_correction);
    m.impl("std.correction_out", &std_correction_out);
    m.impl("var_mean.correction", &var_mean_correction);
    m.impl("std_mean.correction", &std_mean_correction);

    m.impl("prod", &prod_all);
    m.impl("prod.out", &prod_all_out);
    m.impl("prod.dim_int", &prod_dim_int);
    m.impl("prod.int_out", &prod_int_out);
    m.impl("linalg_vector_norm", &linalg_vector_norm_impl);
    m.impl("linalg_vector_norm.out", &linalg_vector_norm_out);

    m.impl("lerp.Scalar", &lerp_scalar_impl);
    m.impl("lerp.Tensor", &lerp_tensor_impl);
    m.impl("lerp_.Tensor", &lerp_tensor_inplace);
    m.impl("lerp.Scalar_out", &lerp_scalar_out);
    m.impl("lerp.Tensor_out", &lerp_tensor_out);
    m.impl("addcmul", &addcmul_impl);
    m.impl("addcmul.out", &addcmul_out);
    m.impl("addcdiv", &addcdiv_impl);
    m.impl("addcdiv.out", &addcdiv_out);
}

}  // namespace
}  // namespace iris
