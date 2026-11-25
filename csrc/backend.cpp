#include "opencl_runtime.h"
#include "rng_internal.h"

#include <ATen/core/Generator.h>
#include <ATen/DLConvertor.h>
#include <c10/core/GeneratorImpl.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/Reduction.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/any.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/isnan.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/max.h>
#include <ATen/ops/min.h>
#include <ATen/ops/where.h>
#include <ATen/ops/index.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_values_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/copy_sparse_to_sparse_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/indices_native.h>
#include <ATen/ops/is_coalesced_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/values_native.h>
#include <ATen/core/IListRef.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/view_native.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/safe_numerics.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_set>

namespace iris {
namespace {

constexpr auto device_type = c10::DeviceType::PrivateUse1;
constexpr c10::DispatchKeySet dispatch_keys(c10::DispatchKey::PrivateUse1);

c10::Device iris_device() {
    return {device_type, 0};
}

void check_index(c10::DeviceIndex index) {
    TORCH_CHECK(index == 0 || index == -1, "Iris supports only device iris:0");
}

void check_device(c10::Device device) {
    TORCH_CHECK(device.type() == device_type, "Expected an Iris tensor or device");
    check_index(device.index());
}

void resize_storage(const c10::Storage& storage, size_t bytes) {
    TORCH_CHECK(storage.resizable() && storage.allocator(), "Iris storage is not resizable");
    if (bytes == storage.nbytes()) {
        return;
    }
    auto replacement = storage.allocator()->allocate(bytes);
    iris::copy(static_cast<cl_mem>(replacement.get()),
               static_cast<cl_mem>(storage.data_ptr().get()),
               (std::min)(bytes, storage.nbytes()));
    storage.set_data_ptr_noswap(std::move(replacement));
    storage.set_nbytes(bytes);
}

class PinnedAllocator final : public c10::Allocator {
public:
    c10::DataPtr allocate(std::size_t bytes) override {
        auto* pointer = allocate_pinned(bytes);
        return {pointer, pointer, &release_pinned, c10::Device(c10::kCPU)};
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &release_pinned;
    }

    void copy_data(void* destination, const void* source, std::size_t bytes) const override {
        if (bytes) {
            std::memcpy(destination, source, bytes);
        }
    }
};

PinnedAllocator pinned_allocator;

class IrisGuard final : public c10::impl::DeviceGuardImplInterface {
public:
    c10::DeviceType type() const override {
        return device_type;
    }

    c10::Device exchangeDevice(c10::Device device) const override {
        setDevice(device);
        return iris_device();
    }

    c10::Device getDevice() const override {
        return iris_device();
    }

    void setDevice(c10::Device device) const override {
        check_device(device);
    }

    void uncheckedSetDevice(c10::Device) const noexcept override {}

    c10::Stream getStream(c10::Device device) const override {
        check_device(device);
        return c10::Stream(c10::Stream::UNSAFE, iris_device(), current_stream());
    }

    c10::Stream getDefaultStream(c10::Device device) const override {
        check_device(device);
        return c10::Stream(c10::Stream::DEFAULT, iris_device());
    }

    c10::Stream getNewStream(c10::Device device, int = 0) const override {
        check_device(device);
        return c10::Stream(c10::Stream::UNSAFE, iris_device(), new_stream());
    }

    c10::Stream getStreamFromGlobalPool(c10::Device device, bool = false) const override {
        return getNewStream(device);
    }

    c10::Stream exchangeStream(c10::Stream stream) const override {
        check_device(stream.device());
        return c10::Stream(c10::Stream::UNSAFE, iris_device(), exchange_stream(stream.id()));
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        return gpu_available() ? 1 : 0;
    }

    void synchronizeDevice(c10::DeviceIndex index) const override {
        check_index(index);
        synchronize();
    }

    void synchronizeStream(const c10::Stream& stream) const override {
        check_device(stream.device());
        synchronize_stream(stream.id());
    }

    bool queryStream(const c10::Stream& stream) const override {
        check_device(stream.device());
        return query_stream(stream.id());
    }

    void record(void** event, const c10::Stream& stream, c10::DeviceIndex index,
                c10::EventFlag) const override {
        check_device(stream.device());
        check_index(index);
        auto replacement = record_event(stream.id());
        release_event(static_cast<cl_event>(*event));
        *event = replacement;
    }

    void destroyEvent(void* event, c10::DeviceIndex) const noexcept override {
        release_event(static_cast<cl_event>(event));
    }

    void block(void* event, const c10::Stream& stream) const override {
        check_device(stream.device());
        wait_event(stream.id(), static_cast<cl_event>(event));
    }

    bool queryEvent(void* event) const override {
        return query_event(static_cast<cl_event>(event));
    }

    void synchronizeEvent(void* event) const override {
        synchronize_event(static_cast<cl_event>(event));
    }

    double elapsedTime(void* first, void* last, c10::DeviceIndex index) const override {
        check_index(index);
        return elapsed_time(static_cast<cl_event>(first), static_cast<cl_event>(last));
    }

    void recordDataPtrOnStream(const c10::DataPtr& ptr, const c10::Stream& stream) const override {
        check_device(stream.device());
        record_buffer(static_cast<cl_mem>(ptr.get()), stream.id());
    }
};

class IrisHooks final : public at::PrivateUse1HooksInterface {
public:
    bool isBuilt() const override {
        return true;
    }

    bool isAvailable() const override {
        return gpu_available();
    }

    bool hasPrimaryContext(c10::DeviceIndex index) const override {
        check_index(index);
        return context_initialized();
    }

    void init() const override {
        initialize();
    }

    c10::DeviceIndex deviceCount() const override {
        return gpu_available() ? 1 : 0;
    }

    void setCurrentDevice(c10::DeviceIndex index) const override {
        check_index(index);
    }

    c10::DeviceIndex getCurrentDevice() const override {
        return 0;
    }

    c10::DeviceIndex exchangeDevice(c10::DeviceIndex index) const override {
        check_index(index);
        return 0;
    }

    c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex index) const override {
        return exchangeDevice(index);
    }

    at::Device getDeviceFromPtr(void*) const override {
        return iris_device();
    }

    void resizePrivateUse1Bytes(const c10::Storage& storage, size_t bytes) const override {
        resize_storage(storage, bytes);
    }

    bool isPinnedPtr(const void* pointer) const override {
        return is_pinned(pointer);
    }

    at::Allocator* getPinnedMemoryAllocator() const override {
        return &pinned_allocator;
    }

    const at::Generator& getDefaultGenerator(c10::DeviceIndex index) const override;
    at::Generator getNewGenerator(c10::DeviceIndex index = -1) const override;
};

class IrisAllocator final : public c10::Allocator {
public:
    c10::DataPtr allocate(std::size_t bytes) override {
        const auto buffer = iris::allocate(bytes);
        // Tensor storage holds an opaque cl_mem. Only CPU tensors are dereferenced.
        return {buffer, buffer, &release, iris_device()};
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &release;
    }

    void copy_data(void* destination, const void* source, std::size_t bytes) const override {
        iris::copy(static_cast<cl_mem>(destination),
                   static_cast<cl_mem>(const_cast<void*>(source)), bytes);
    }
};

IrisAllocator allocator;

ValueType value_type(at::ScalarType type) {
    switch (type) {
        case at::kFloat: return ValueType::Float;
        case at::kLong: return ValueType::Long;
        case at::kBool: return ValueType::Bool;
        case at::kByte: return ValueType::Byte;
        case at::kChar: return ValueType::Char;
        case at::kShort: return ValueType::Short;
        case at::kBFloat16: return ValueType::BFloat;
        case at::kComplexFloat: return ValueType::Complex;
        case at::kInt: return ValueType::Int;
        case at::kHalf:
            TORCH_CHECK(has_fp16(), "The selected Iris GPU does not support OpenCL FP16");
            return ValueType::Half;
        default: TORCH_CHECK(false, "Iris supports float32, float16, bfloat16, complex64, int64, int32, int16, int8, uint8 and bool storage");
    }
}

bool supported_float(at::ScalarType type) {
    return type == at::kFloat || type == at::kHalf;
}

c10::ScalarType check_options(
    std::optional<c10::ScalarType> dtype, std::optional<c10::Layout> layout,
    std::optional<c10::Device> device, std::optional<bool> pin_memory) {
    check_device(device.value_or(iris_device()));
    const auto type = c10::dtype_or_default(dtype);
    (void)value_type(type);
    TORCH_CHECK(c10::layout_or_default(layout) == c10::kStrided,
                "Iris supports only strided tensors");
    TORCH_CHECK(!pin_memory.value_or(false), "Iris allocations cannot use pinned memory");
    return type;
}

at::Tensor empty(
    c10::SymIntArrayRef sizes, std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout, std::optional<c10::Device> device,
    std::optional<bool> pin_memory, std::optional<c10::MemoryFormat> memory_format) {
    const auto type = check_options(dtype, layout, device, pin_memory);
    TORCH_CHECK(memory_format.value_or(c10::MemoryFormat::Contiguous) ==
                    c10::MemoryFormat::Contiguous,
                "Iris supports only contiguous memory format");
    at::detail::check_size_nonnegative(sizes);
    initialize();
    return at::Tensor(at::detail::empty_generic_symint(
        sizes, &allocator, dispatch_keys, type, memory_format));
}

at::Tensor empty_strided(
    c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides,
    std::optional<c10::ScalarType> dtype, std::optional<c10::Layout> layout,
    std::optional<c10::Device> device, std::optional<bool> pin_memory) {
    const auto type = check_options(dtype, layout, device, pin_memory);
    TORCH_CHECK(sizes.size() == strides.size(), "Iris requires one stride per dimension");
    at::detail::check_size_nonnegative(sizes);
    at::detail::check_size_nonnegative(strides);
    initialize();
    return at::Tensor(at::detail::empty_strided_symint_generic(
        sizes, strides, &allocator, dispatch_keys, type));
}

bool is_iris(const at::Tensor& tensor) {
    return tensor.device().type() == device_type;
}

void check_tensor(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.layout() == c10::kStrided, "Iris operations require strided tensors");
    TORCH_CHECK(supported_float(tensor.scalar_type()), "Iris operation requires float32 or float16");
}

at::Tensor& set_storage(at::Tensor& input, at::Storage storage, int64_t offset,
                        at::IntArrayRef sizes, at::IntArrayRef strides) {
    check_device(storage.device());
    at::detail::check_size_nonnegative(sizes);
    if (!strides.empty()) {
        at::native::checkAsStridedArgs(sizes, strides, offset);
    }
    TORCH_CHECK(offset >= 0, "Iris storage offset must be nonnegative");
    const auto bytes = strides.empty()
        ? at::detail::computeStorageNbytesContiguous(sizes, input.element_size(), offset)
        : at::detail::computeStorageNbytes(sizes, strides, input.element_size(), offset);
    if (bytes > storage.nbytes()) {
        resize_storage(storage, bytes);
    }
    auto* impl = input.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(std::move(storage));
    if (strides.empty()) {
        impl->set_sizes_contiguous(sizes);
        impl->set_storage_offset(offset);
    } else {
        impl->set_sizes_and_strides(sizes, strides, offset);
    }
    return input;
}

at::Tensor& set_storage_full(at::Tensor& input, at::Storage storage) {
    const auto length = static_cast<int64_t>(storage.nbytes() / input.element_size());
    return set_storage(input, std::move(storage), 0, {length}, {});
}

const at::Tensor& resize_tensor(const at::Tensor& input, at::IntArrayRef sizes,
                                std::optional<c10::MemoryFormat> memory_format) {
    TORCH_CHECK(memory_format.value_or(c10::MemoryFormat::Contiguous) == c10::MemoryFormat::Contiguous,
                "Iris resize_ supports contiguous memory format");
    if (!memory_format.has_value() && input.sizes() == sizes) {
        return input;
    }
    at::detail::check_size_nonnegative(sizes);
    const auto bytes = at::detail::computeStorageNbytesContiguous(
        sizes, input.element_size(), input.storage_offset());
    if (bytes > input.storage().nbytes()) {
        resize_storage(input.storage(), bytes);
    }
    input.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
    return input;
}

cl_mem buffer(const at::Tensor& tensor) {
    return static_cast<cl_mem>(tensor.storage().data_ptr().get());
}

at::Tensor make_output(at::IntArrayRef sizes, at::ScalarType type = at::kFloat) {
    return at::Tensor(at::detail::empty_generic(
        sizes, &allocator, dispatch_keys, type, c10::MemoryFormat::Contiguous));
}

void release_dlpack(void* context) {
    auto* managed = static_cast<DLManagedTensor*>(context);
    if (managed->deleter) managed->deleter(managed);
}

pybind11::capsule export_dlpack(const at::Tensor& tensor) {
    check_device(tensor.device());
    TORCH_CHECK(tensor.layout() == at::kStrided && !tensor.requires_grad() && !tensor.is_conj() && !tensor.is_neg(),
                "Iris DLPack export needs a detached strided tensor with resolved conjugate and negative bits");
    synchronize();
    auto packed = std::unique_ptr<DLManagedTensor, decltype(&release_dlpack)>(at::toDLPack(tensor), release_dlpack);
    auto& metadata = packed->dl_tensor;
    metadata.device = {kDLOpenCL, 0};
    metadata.data = tensor.numel() ? buffer(tensor) : nullptr;
    metadata.byte_offset = tensor.numel() ? tensor.storage_offset() * tensor.element_size() : 0;
    auto result = pybind11::capsule(packed.get(), "dltensor", [](PyObject* capsule) {
        if (PyCapsule_IsValid(capsule, "dltensor")) release_dlpack(PyCapsule_GetPointer(capsule, "dltensor"));
    });
    packed.release();
    return result;
}

at::Tensor import_dlpack(const pybind11::capsule& capsule) {
    TORCH_CHECK(PyCapsule_IsValid(capsule.ptr(), "dltensor"), "expected an unconsumed legacy DLPack capsule");
    auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    const auto& t = managed->dl_tensor;
    TORCH_CHECK(t.device.device_type == kDLOpenCL && t.device.device_id == 0,
                "Iris import requires a device 0 kDLOpenCL capsule");
    const auto dtype = at::toScalarType(t.dtype);
    (void)value_type(dtype);
    TORCH_CHECK(t.ndim >= 0 && (!t.ndim || t.shape), "invalid DLPack tensor dimensions");
    const auto item_size = c10::elementSize(dtype);
    TORCH_CHECK(t.byte_offset % item_size == 0 && t.byte_offset / item_size <= std::numeric_limits<int64_t>::max(),
                "invalid DLPack tensor byte offset");
    const auto offset = static_cast<int64_t>(t.byte_offset / item_size);
    const at::IntArrayRef sizes(t.shape, t.ndim);
    at::detail::check_size_nonnegative(sizes);
    const auto strides = t.strides ? at::IntArrayRef(t.strides, t.ndim) : at::IntArrayRef{};
    TORCH_CHECK(std::all_of(strides.begin(), strides.end(), [](int64_t s) { return s >= 0; }),
                "Iris DLPack import requires non-negative strides");
    const auto needed = strides.empty() ? at::detail::computeStorageNbytesContiguous(sizes, item_size, offset)
                                        : at::detail::computeStorageNbytes(sizes, strides, item_size, offset);
    const auto bytes = t.data ? external_buffer_size(static_cast<cl_mem>(t.data)) : 0;
    TORCH_CHECK(bytes >= needed, "DLPack view exceeds the OpenCL buffer");
    auto output = make_output({0}, dtype);
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0) throw pybind11::error_already_set();
    c10::DataPtr data(t.data, managed, &release_dlpack, iris_device());
    auto storage = c10::Storage(c10::Storage::use_byte_size_t(), bytes, std::move(data));
    return set_storage(output, std::move(storage), offset, sizes, strides);
}

c10::SmallVector<cl_ulong, 24> elementwise_geometry(
    const at::Tensor& output, const at::Tensor& left, const at::Tensor& right) {
    const auto sizes = output.sizes();
    const auto a = at::inferExpandGeometry_dimvector(left.sizes(), left.strides(), sizes);
    const auto b = at::inferExpandGeometry_dimvector(right.sizes(), right.strides(), sizes);
    c10::SmallVector<cl_ulong, 24> geometry;
    for (const auto dim : c10::irange(output.dim())) {
        geometry.append({static_cast<cl_ulong>(sizes[dim]),
                         static_cast<cl_ulong>(output.stride(dim)),
                         static_cast<cl_ulong>(a.strides[dim]),
                         static_cast<cl_ulong>(b.strides[dim])});
    }
    return geometry;
}

void run_strided(
    Elementwise op, const at::Tensor& output, const at::Tensor& left, const at::Tensor& right,
    float alpha = 1.0f, float beta = 0.0f, bool serial = false) {
    if (!output.numel()) {
        return;
    }
    if (!serial && output.is_contiguous() && left.is_contiguous() && right.is_contiguous() &&
        output.sizes() == left.sizes() && output.sizes() == right.sizes()) {
        contiguous_elementwise(op, buffer(output), buffer(left), buffer(right), output.numel(),
                               output.storage_offset(), left.storage_offset(), right.storage_offset(),
                               alpha, beta, left.scalar_type() == at::kHalf);
        return;
    }
    const auto geometry = elementwise_geometry(output, left, right);
    strided_elementwise(op, buffer(output), buffer(left), buffer(right), output.numel(),
                        output.storage_offset(), left.storage_offset(), right.storage_offset(),
                        std::span<const cl_ulong>(geometry.data(), geometry.size()), alpha, beta, serial,
                        left.scalar_type() == at::kHalf);
}

void copy_tensor(const at::Tensor& destination, const at::Tensor& source, bool non_blocking = false) {
    TORCH_CHECK(destination.layout() == c10::kStrided && source.layout() == c10::kStrided,
                "Iris copy_ requires strided tensors");
    const bool destination_iris = is_iris(destination);
    const bool source_iris = is_iris(source);
    TORCH_CHECK((destination_iris || destination.device().is_cpu()) &&
                    (source_iris || source.device().is_cpu()) &&
                    (destination_iris || source_iris),
                "Iris supports copies between Iris and CPU tensors");
    const bool same_shape = destination.sizes() == source.sizes();
    if (!destination.numel() || destination.is_same(source)) {
        if (!same_shape) {
            (void)at::expand_inplace(destination, source, "copy_");
        }
        return;
    }
    at::assert_no_internal_overlap(destination);
    at::assert_no_partial_overlap(destination, source);
    const bool same_type = source.scalar_type() == destination.scalar_type();
    const bool conjugate = source.is_conj() != destination.is_conj();
    const bool negate = source.is_neg() != destination.is_neg();
    const bool transform = conjugate || negate;
    const bool same_layout = same_shape && destination.strides() == source.strides() &&
        destination.is_non_overlapping_and_dense() && source.is_non_overlapping_and_dense();
    if (!transform && same_type &&
        (same_layout || (same_shape && destination.is_contiguous() && source.is_contiguous()))) {
        const auto bytes = destination.nbytes();
        const auto dst_offset = destination.storage_offset() * destination.element_size();
        const auto src_offset = source.storage_offset() * source.element_size();
        if (destination_iris && source_iris) {
            iris::copy(buffer(destination), buffer(source), bytes, dst_offset, src_offset);
        } else if (destination_iris) {
            upload(buffer(destination), source.const_data_ptr(), bytes, dst_offset, non_blocking);
        } else {
            download(destination.mutable_data_ptr(), buffer(source), bytes, src_offset, non_blocking);
        }
        return;
    }
    // PyTorch's Conjugate and Negative dispatch keys fall through for copy_.
    // Apply the relative view flags in the device copy, or resolve them on the
    // owning device before staging a transfer or dtype conversion.
    if (!(destination_iris && source_iris && same_type) &&
        (source.is_conj() || source.is_neg() || destination.is_conj() || destination.is_neg())) {
        const auto resolved = source.resolve_conj().resolve_neg();
        if (!destination.is_conj() && !destination.is_neg()) {
            copy_tensor(destination, resolved, non_blocking);
        } else if (destination_iris) {
            auto staging = make_output(destination.sizes(), destination.scalar_type());
            copy_tensor(staging, resolved, non_blocking);
            copy_tensor(destination, staging);
        } else {
            auto staging = at::empty(resolved.sizes(), resolved.options().device(c10::kCPU));
            copy_tensor(staging, resolved);
            destination.copy_(staging);
        }
        return;
    }
    const auto expanded = at::expand_inplace(destination, source, "copy_");
    if (destination_iris && source_iris) {
        if (same_shape && destination.is_contiguous() && source.is_contiguous() &&
            destination.storage_offset() == 0 && source.storage_offset() == 0 &&
            ((source.scalar_type() == at::kFloat && destination.scalar_type() == at::kHalf) ||
             (source.scalar_type() == at::kHalf && destination.scalar_type() == at::kFloat))) {
            cast_fp32_fp16_contiguous(buffer(destination), buffer(source), destination.numel(),
                                       destination.scalar_type() == at::kHalf);
        } else if (!transform && same_type && source.scalar_type() == at::kFloat) {
            run_strided(Elementwise::Copy, destination, *expanded, *expanded);
        } else {
            TORCH_CHECK(!negate || source.scalar_type() != at::kBool, "Iris does not support negation of bool tensors");
            const auto geometry = elementwise_geometry(destination, *expanded, *expanded);
            typed_copy(buffer(destination), buffer(*expanded), destination.numel(),
                       destination.storage_offset(), expanded->storage_offset(), geometry,
                       value_type(destination.scalar_type()), value_type(source.scalar_type()), 0, false,
                       conjugate, negate);
        }
    } else if (destination_iris) {
        const auto packed = expanded->to(destination.scalar_type()).contiguous();
        if (destination.is_contiguous()) {
            copy_tensor(destination, packed);
        } else {
            auto staging = make_output(destination.sizes(), destination.scalar_type());
            copy_tensor(staging, packed);
            copy_tensor(destination, staging);
        }
    } else {
        auto staging = expanded->contiguous();
        if (destination.is_contiguous() && same_type) {
            copy_tensor(destination, staging);
        } else {
            auto packed = at::empty(destination.sizes(), source.options().device(c10::kCPU));
            copy_tensor(packed, staging);
            destination.copy_(packed);
        }
    }
}

at::Tensor& copy_(at::Tensor& destination, const at::Tensor& source, bool non_blocking) {
    copy_tensor(destination, source, non_blocking);
    return destination;
}

at::Tensor copy_from(const at::Tensor& source, const at::Tensor& destination, bool non_blocking) {
    copy_tensor(destination, source, non_blocking);
    return destination;
}

at::Tensor copy_sparse_from(const at::Tensor& source, const at::Tensor& destination, bool non_blocking) {
    auto output = destination;
    return at::native::copy_sparse_wrapper_(output, source, non_blocking);
}

at::Tensor to_copy(
    const at::Tensor& source, std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout, std::optional<c10::Device> device,
    std::optional<bool> pin_memory, bool non_blocking, std::optional<c10::MemoryFormat> memory_format) {
    return at::native::_to_copy(source, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

void record_tensor_stream(at::Tensor& tensor, c10::Stream stream) {
    check_device(stream.device());
    record_buffer(buffer(tensor), stream.id());
}

void check_operand(const at::Tensor& tensor) {
    check_device(tensor.device());
    check_tensor(tensor);
}

bool packed(const at::Tensor& tensor) {
    return tensor.is_contiguous() && tensor.storage_offset() == 0;
}

at::Tensor unary_strided(Elementwise op, const at::Tensor& input, float scalar);

at::Tensor binary_strided(
    Elementwise op, const at::Tensor& left, const at::Tensor& right, float alpha) {
    at::Tensor undefined;
    const auto iter = at::TensorIteratorConfig()
        .add_output(undefined).add_const_input(left).add_const_input(right).build();
    auto output = iter.output();
    run_strided(op, output, left, right, alpha);
    return output;
}

at::Tensor binary(
    Elementwise op, const at::Tensor& left, const at::Tensor& right, float alpha) {
    check_operand(left);
    check_operand(right);
    if (left.scalar_type() == at::kFloat && right.scalar_type() == at::kFloat &&
        left.sizes() == right.sizes() && packed(left) && packed(right)) {
        auto output = make_output(left.sizes());
        elementwise(op, buffer(output), buffer(left), buffer(right), output.numel(), alpha);
        return output;
    }
    return binary_strided(op, left, right, alpha);
}

at::Tensor integral_binary(const at::Tensor& left, const at::Tensor& right, at::ScalarType dtype,
                           at::ScalarType output_type, IntegerOperation op, const at::Scalar& alpha) {
    TORCH_CHECK(c10::isIntegralType(dtype, true), "Iris integral operation requires integral input");
    const auto input_type = value_type(dtype);
    auto output_storage_type = value_type(output_type);
    TORCH_CHECK(alpha.isIntegral(true), "For integral input tensors, alpha must be integral");
    const auto device = is_iris(left) ? left.device() : right.device();
    const auto a = left.to(device, dtype);
    const auto b = right.to(device, dtype);
    at::Tensor undefined;
    auto output = at::TensorIteratorConfig().add_output(undefined).add_const_input(a).add_const_input(b)
        .check_all_same_dtype(false).declare_static_dtype(output_type).build().output();
    if (output.numel()) {
        const auto geometry = elementwise_geometry(output, a, b);
        integer_binary(buffer(output), buffer(a), buffer(b), output.numel(), output.storage_offset(),
                       a.storage_offset(), b.storage_offset(), geometry, input_type, output_storage_type,
                       op, alpha.toLong());
    }
    return output;
}

std::pair<at::Tensor, at::Tensor> complex_parts(const at::Tensor& input) {
    const auto pair = at::native::view_as_real(input.resolve_conj());
    return {pair.select(-1, 0), pair.select(-1, 1)};
}

at::Tensor complex_join(const at::Tensor& real, const at::Tensor& imaginary) {
    return at::native::view_as_complex(at::stack({real, imaginary}, -1));
}

at::Tensor conjugate_physical(const at::Tensor& input) {
    if (!input.is_complex()) return input.clone();
    const auto [real, imaginary] = complex_parts(input);
    return complex_join(real, imaginary.neg());
}

at::Tensor& conjugate_physical_inplace(at::Tensor& input) {
    if (input.is_complex()) {
        at::assert_no_internal_overlap(input);
        copy_tensor(input, input.conj());
    }
    return input;
}

at::Tensor promoted_arithmetic(const at::Tensor& left, const at::Tensor& right,
                               Elementwise op, IntegerOperation integer_op, const at::Scalar& alpha) {
    TORCH_CHECK(is_iris(left) || (left.is_cpu() && left.dim() == 0), "Expected an Iris tensor or CPU scalar");
    TORCH_CHECK(is_iris(right) || (right.is_cpu() && right.dim() == 0), "Expected an Iris tensor or CPU scalar");
    const auto dtype = at::result_type(left, right);
    if (dtype == at::kComplexFloat) {
        const auto device = is_iris(left) ? left.device() : right.device();
        const auto [a, b] = complex_parts(left.to(device, dtype));
        const auto [c, d] = complex_parts(right.to(device, dtype));
        if (op == Elementwise::Multiply) return complex_join(a.mul(c).sub(b.mul(d)), a.mul(d).add(b.mul(c)));
        auto s = alpha.toComplexFloat();
        if (op == Elementwise::Subtract) s = -s;
        if (s.imag() == 0) return complex_join(a.add(c, s.real()), b.add(d, s.real()));
        return complex_join(a.add(c, s.real()).sub(d, s.imag()), b.add(d, s.real()).add(c, s.imag()));
    }
    if (dtype == at::kBFloat16) {
        const auto device = is_iris(left) ? left.device() : right.device();
        return binary_strided(op, left.to(device, at::kFloat), right.to(device, at::kFloat),
                              alpha.toFloat()).to(dtype);
    }
    if (!supported_float(dtype)) {
        TORCH_CHECK(dtype != at::kBool || integer_op != IntegerOperation::Subtract,
                    "Subtraction with two bool tensors is not supported");
        return integral_binary(left, right, dtype, dtype, integer_op, alpha);
    }
    const auto device = is_iris(left) ? left.device() : right.device();
    return binary_strided(op, left.to(device, dtype), right.to(device, dtype), alpha.toFloat());
}

at::Tensor add_tensor(const at::Tensor& left, const at::Tensor& right, const at::Scalar& alpha) {
    TORCH_CHECK(!alpha.isBoolean(), "Iris add requires a numeric alpha");
    if (supported_float(left.scalar_type()) && !right.is_complex() && right.device().is_cpu() && right.dim() == 0) {
        check_operand(left);
        const float value = right.item().toFloat() * alpha.toFloat();
        if (left.scalar_type() == at::kFloat && packed(left)) {
            auto output = make_output(left.sizes());
            elementwise(Elementwise::AddScalar, buffer(output), buffer(left), nullptr, output.numel(), value);
            return output;
        }
        return unary_strided(Elementwise::AddScalar, left, value);
    }
    if (left.scalar_type() == at::kFloat && right.scalar_type() == at::kFloat && is_iris(left) && is_iris(right)) {
        return binary(Elementwise::Add, left, right, alpha.toFloat());
    }
    return promoted_arithmetic(left, right, Elementwise::Add, IntegerOperation::Add, alpha);
}

at::Tensor multiply_tensor(const at::Tensor& left, const at::Tensor& right) {
    if (supported_float(left.scalar_type()) && !right.is_complex() && right.device().is_cpu() && right.dim() == 0) {
        check_operand(left);
        const float value = right.item().toFloat();
        if (left.scalar_type() == at::kFloat && packed(left)) {
            auto output = make_output(left.sizes());
            elementwise(Elementwise::Scale, buffer(output), buffer(left), nullptr, output.numel(), value);
            return output;
        }
        return unary_strided(Elementwise::Scale, left, value);
    }
    if (supported_float(right.scalar_type()) && !left.is_complex() && left.device().is_cpu() && left.dim() == 0) {
        check_operand(right);
        return multiply_tensor(right, left);
    }
    if (left.scalar_type() == at::kFloat && right.scalar_type() == at::kFloat && is_iris(left) && is_iris(right)) {
        return binary(Elementwise::Multiply, left, right, 1.0f);
    }
    return promoted_arithmetic(left, right, Elementwise::Multiply, IntegerOperation::Multiply, 1);
}

at::Tensor& multiply_scalar_inplace(at::Tensor& input, const at::Scalar& scalar) {
    check_device(input.device());
    at::assert_no_internal_overlap(input);
    if (!supported_float(input.scalar_type())) {
        const auto dtype = at::result_type(input, scalar);
        TORCH_CHECK(c10::canCast(dtype, input.scalar_type()), "mul_ result dtype cannot be cast to the input dtype");
        const auto value = at::scalar_tensor(scalar, at::TensorOptions().device(at::kCPU).dtype(dtype));
        copy_tensor(input, promoted_arithmetic(input.to(dtype), value, Elementwise::Multiply, IntegerOperation::Multiply, 1));
        return input;
    }
    TORCH_CHECK(!scalar.isComplex(), "mul_ result dtype cannot be cast to the input dtype");
    if (input.scalar_type() == at::kFloat && packed(input)) {
        elementwise(Elementwise::Scale, buffer(input), buffer(input), nullptr, input.numel(), scalar.toFloat());
    } else {
        run_strided(Elementwise::Scale, input, input, input, scalar.toFloat());
    }
    return input;
}

at::Tensor& multiply_inplace(at::Tensor& input, const at::Tensor& other) {
    if (other.device().is_cpu() && other.dim() == 0) {
        return multiply_scalar_inplace(input, other.item());
    }
    check_device(input.device());
    check_device(other.device());
    at::assert_no_internal_overlap(input);
    at::assert_no_partial_overlap(input, other);
    const auto expanded = at::expand_inplace(input, other, "mul_");
    if (!supported_float(input.scalar_type()) || input.scalar_type() != other.scalar_type()) {
        const auto result = multiply_tensor(input, *expanded);
        TORCH_CHECK(c10::canCast(result.scalar_type(), input.scalar_type()), "mul_ result dtype cannot be cast to the input dtype");
        copy_tensor(input, result);
        return input;
    }
    if (input.scalar_type() == at::kFloat && input.sizes() == other.sizes() && packed(input) && packed(other)) {
        elementwise(Elementwise::Multiply, buffer(input), buffer(input), buffer(other),
                    input.numel(), 1.0f);
    } else {
        run_strided(Elementwise::Multiply, input, input, *expanded);
    }
    return input;
}

at::Tensor threshold_backward(
    const at::Tensor& grad, const at::Tensor& input, const at::Scalar& threshold) {
    check_operand(grad);
    check_operand(input);
    if (grad.scalar_type() == at::kFloat && input.scalar_type() == at::kFloat &&
        grad.sizes() == input.sizes() && packed(grad) && packed(input)) {
        auto output = make_output(grad.sizes());
        elementwise(Elementwise::ThresholdBackward, buffer(output), buffer(grad), buffer(input),
                    output.numel(), threshold.toFloat());
        return output;
    }
    return binary_strided(Elementwise::ThresholdBackward, grad, input, threshold.toFloat());
}

at::Tensor unary_strided(Elementwise op, const at::Tensor& input, float scalar = 1.0f) {
    at::Tensor undefined;
    const auto iter = at::TensorIteratorConfig()
        .add_output(undefined).add_const_input(input).build();
    auto output = iter.output();
    run_strided(op, output, input, input, scalar);
    return output;
}

at::Tensor relu(const at::Tensor& input) {
    check_operand(input);
    if (input.scalar_type() == at::kFloat && packed(input)) {
        auto output = make_output(input.sizes());
        elementwise(Elementwise::Relu, buffer(output), buffer(input), nullptr, output.numel(), 1.0f);
        return output;
    }
    return unary_strided(Elementwise::Relu, input);
}

at::Tensor divide_scalar(const at::Tensor& input, const at::Scalar& scalar) {
    check_operand(input);
    TORCH_CHECK(!scalar.isComplex(), "Iris div requires a real scalar");
    return unary_strided(Elementwise::Divide, input, scalar.toFloat());
}

at::Tensor& add_inplace(at::Tensor& left, const at::Tensor& right, const at::Scalar& alpha) {
    check_device(left.device());
    TORCH_CHECK(!alpha.isBoolean() && !alpha.isComplex(), "Iris add requires a real numeric alpha");
    at::assert_no_internal_overlap(left);
    if (!supported_float(left.scalar_type())) {
        copy_tensor(left, add_tensor(left, right, alpha));
        return left;
    }
    check_operand(left);
    if (right.device().is_cpu() && right.dim() == 0) {
        run_strided(Elementwise::AddScalar, left, left, left, right.item().toFloat() * alpha.toFloat());
        return left;
    }
    check_operand(right);
    at::assert_no_partial_overlap(left, right);
    if (left.scalar_type() != right.scalar_type()) {
        copy_tensor(left, add_tensor(left, right, alpha));
    } else if (left.scalar_type() == at::kFloat && left.sizes() == right.sizes() && packed(left) && packed(right)) {
        elementwise(Elementwise::Add, buffer(left), buffer(left), buffer(right),
                    left.numel(), alpha.toFloat());
    } else {
        const auto expanded = at::expand_inplace(left, right, "add_");
        run_strided(Elementwise::Add, left, left, *expanded, alpha.toFloat());
    }
    return left;
}

template <typename T>
void fill_integer(const at::Tensor& input, ValueType type, T value) {
    if (input.is_non_overlapping_and_dense()) {
        fill_bytes(buffer(input), input.numel(), input.storage_offset(), &value, sizeof(T));
    } else if (input.numel()) {
        const auto geometry = elementwise_geometry(input, input, input);
        typed_copy(buffer(input), nullptr, input.numel(), input.storage_offset(), 0, geometry,
                   type, type, static_cast<cl_ulong>(value), true);
    }
}

at::Tensor& fill_scalar(at::Tensor& input, const at::Scalar& value) {
    check_device(input.device());
    const auto type = value_type(input.scalar_type());
    switch (type) {
        case ValueType::Bool: fill_integer(input, type, static_cast<cl_uchar>(value.toBool())); return input;
        case ValueType::Byte: fill_integer(input, type, value.toByte()); return input;
        case ValueType::Char: fill_integer(input, type, value.toChar()); return input;
        case ValueType::Short: fill_integer(input, type, value.toShort()); return input;
        case ValueType::BFloat: fill_integer(input, type, value.toBFloat16().x); return input;
        case ValueType::Complex: {
            const auto scalar = value.toComplexFloat();
            cl_ulong bits;
            static_assert(sizeof(bits) == sizeof(scalar));
            std::memcpy(&bits, &scalar, sizeof(bits));
            fill_integer(input, type, bits);
            return input;
        }
        case ValueType::Int: fill_integer(input, type, value.toInt()); return input;
        case ValueType::Long: fill_integer(input, type, value.toLong()); return input;
        case ValueType::Half: {
            const auto half = value.toHalf();
            if (input.is_non_overlapping_and_dense()) {
                fill_bytes(buffer(input), input.numel(), input.storage_offset(), &half, sizeof(half));
            } else {
                run_strided(Elementwise::Fill, input, input, input, value.toFloat(), 0.0f, true);
            }
            return input;
        }
        case ValueType::Float: break;
    }
    if (input.is_non_overlapping_and_dense()) {
        fill(buffer(input), input.numel(), input.storage_offset(), value.toFloat());
    } else {
        // Fill can write an overlapping view. A single lane avoids concurrent writes.
        run_strided(Elementwise::Fill, input, input, input, value.toFloat(), 0.0f, true);
    }
    return input;
}

at::Tensor& zero(at::Tensor& input) {
    return fill_scalar(input, 0.0f);
}

template <typename T>
T read_scalar(const at::Tensor& input) {
    T result;
    download(&result, buffer(input), sizeof(T), input.storage_offset() * sizeof(T));
    return result;
}

at::Scalar local_scalar(const at::Tensor& input) {
    TORCH_CHECK(input.numel() == 1, "Iris item() requires a single element");
    switch (value_type(input.scalar_type())) {
        case ValueType::Float: return read_scalar<float>(input);
        case ValueType::Long: return read_scalar<int64_t>(input);
        case ValueType::Bool: return read_scalar<uint8_t>(input) != 0;
        case ValueType::Byte: return read_scalar<uint8_t>(input);
        case ValueType::Char: return read_scalar<int8_t>(input);
        case ValueType::Short: return read_scalar<int16_t>(input);
        case ValueType::BFloat: return static_cast<float>(read_scalar<c10::BFloat16>(input));
        case ValueType::Complex: return read_scalar<c10::complex<float>>(input);
        case ValueType::Int: return read_scalar<int32_t>(input);
        case ValueType::Half: return static_cast<float>(read_scalar<c10::Half>(input));
    }
    TORCH_INTERNAL_ASSERT(false);
}

class IrisGeneratorImpl final : public c10::GeneratorImpl {
public:
    explicit IrisGeneratorImpl(std::uint64_t seed_val = c10::default_rng_seed_val)
        : c10::GeneratorImpl(iris_device(), dispatch_keys), seed_(seed_val), offset_(0) {}

    void set_current_seed(std::uint64_t seed) override {
        seed_ = seed;
        offset_ = 0;
    }

    void set_offset(std::uint64_t offset) override {
        offset_ = offset;
    }

    std::uint64_t get_offset() const override {
        return offset_;
    }

    std::uint64_t current_seed() const override {
        return seed_;
    }

    std::uint64_t seed() override {
        const auto random = c10::detail::getNonDeterministicRandom();
        set_current_seed(random);
        return random;
    }

    void set_state(const c10::TensorImpl& new_state) override {
        TORCH_CHECK(new_state.device().is_cpu() && new_state.is_contiguous() &&
                    new_state.dtype() == c10::ScalarType::Byte && new_state.numel() == 16,
                    "Iris generator state must be a contiguous 16-byte CPU uint8 tensor");
        const auto* data = static_cast<const std::uint8_t*>(new_state.data());
        std::memcpy(&seed_, data, 8);
        std::memcpy(&offset_, data + 8, 8);
    }

    c10::intrusive_ptr<c10::TensorImpl> get_state() const override {
        auto state = at::empty({16}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        auto* data = state.data_ptr<std::uint8_t>();
        std::memcpy(data, &seed_, 8);
        std::memcpy(data + 8, &offset_, 8);
        return state.getIntrusivePtr();
    }

    std::pair<std::uint64_t, std::uint64_t> reserve(std::uint64_t blocks) {
        const std::lock_guard lock(mutex_);
        TORCH_CHECK(offset_ <= std::numeric_limits<std::uint64_t>::max() - blocks,
                    "Iris RNG counter exhausted");
        const auto start = offset_;
        offset_ += blocks;
        return {seed_, start};
    }

protected:
    c10::GeneratorImpl* clone_impl() const override {
        auto* copy = new IrisGeneratorImpl(seed_);
        copy->offset_ = offset_;
        return copy;
    }

private:
    std::uint64_t seed_;
    std::uint64_t offset_;
};

const at::Generator& default_iris_generator() {
    static at::Generator gen = [] {
        auto impl = c10::make_intrusive<IrisGeneratorImpl>(c10::detail::getNonDeterministicRandom());
        return at::Generator(std::move(impl));
    }();
    return gen;
}

const at::Generator& IrisHooks::getDefaultGenerator(c10::DeviceIndex index) const {
    check_index(index);
    return default_iris_generator();
}

at::Generator IrisHooks::getNewGenerator(c10::DeviceIndex index) const {
    check_index(index);
    return at::Generator(c10::make_intrusive<IrisGeneratorImpl>(c10::default_rng_seed_val));
}

void manual_seed(std::uint64_t seed) {
    auto* impl = static_cast<IrisGeneratorImpl*>(default_iris_generator().unsafeGetGeneratorImpl());
    const std::lock_guard lock(impl->mutex_);
    impl->set_current_seed(seed);
}

std::pair<std::uint64_t, std::uint64_t> reserve_rng_blocks_impl(
    std::uint64_t blocks, const std::optional<at::Generator>& generator = std::nullopt) {
    at::Generator gen = generator.has_value() ? *generator : default_iris_generator();
    check_device(gen.device());
    auto* impl = dynamic_cast<IrisGeneratorImpl*>(gen.unsafeGetGeneratorImpl());
    TORCH_CHECK(impl != nullptr, "Expected an Iris torch.Generator");
    return impl->reserve(blocks);
}

at::Tensor& random_fill(at::Tensor& input, RandomDistribution distribution, float first, float second,
                        const std::optional<at::Generator>& generator) {
    check_operand(input);
    at::assert_no_internal_overlap(input);
    if (!input.numel()) {
        return input;
    }
    c10::SmallVector<cl_ulong, 16> geometry;
    for (const auto dim : c10::irange(input.dim())) {
        geometry.append({static_cast<cl_ulong>(input.size(dim)),
                         static_cast<cl_ulong>(input.stride(dim))});
    }
    const auto blocks = (static_cast<std::uint64_t>(input.numel()) - 1) / 4 + 1;
    const auto [seed, start] = reserve_rng_blocks_impl(blocks, generator);
    iris::random_fill(buffer(input), input.numel(), input.storage_offset(), geometry,
                      seed, start, first, second, distribution, input.scalar_type() == at::kHalf);
    return input;
}

at::Tensor& uniform(at::Tensor& input, double from, double to, std::optional<at::Generator> generator) {
    const auto limit = static_cast<double>(std::numeric_limits<float>::max());
    TORCH_CHECK(from >= -limit && to <= limit && from <= to && to - from <= limit,
                "Iris uniform_ requires an ordered interval within the float32 range");
    return random_fill(input, RandomDistribution::Uniform, static_cast<float>(from),
                       static_cast<float>(to), generator);
}

at::Tensor& normal(at::Tensor& input, double mean, double stddev, std::optional<at::Generator> generator) {
    TORCH_CHECK(stddev >= 0, "normal expects std >= 0.0, but found std ", stddev);
    return random_fill(input, RandomDistribution::Normal, static_cast<float>(mean),
                       static_cast<float>(stddev), generator);
}

at::Tensor& bernoulli(at::Tensor& input, double probability, std::optional<at::Generator> generator) {
    TORCH_CHECK(probability >= 0 && probability <= 1, "bernoulli_ expects a probability in [0, 1]");
    return random_fill(input, RandomDistribution::Bernoulli, static_cast<float>(probability),
                       0.0f, generator);
}

template <Elementwise op>
at::Tensor pointwise_unary(const at::Tensor& input) {
    check_operand(input);
    return unary_strided(op, input);
}

template <Elementwise op>
at::Tensor& pointwise_unary_inplace(at::Tensor& input) {
    check_operand(input);
    at::assert_no_internal_overlap(input);
    run_strided(op, input, input, input);
    return input;
}

template <Elementwise op>
at::Tensor& pointwise_unary_out(const at::Tensor& input, at::Tensor& output) {
    check_operand(input);
    check_operand(output);
    TORCH_CHECK(input.scalar_type() == output.scalar_type(), "Iris unary out dtype must match input");
    (void)at::TensorIteratorConfig().add_output(output).add_const_input(input).build();
    run_strided(op, output, input, input);
    return output;
}

template <Elementwise op>
at::Tensor pointwise_binary(const at::Tensor& left, const at::Tensor& right) {
    check_operand(left);
    check_operand(right);
    return binary_strided(op, left, right, 1.0f);
}

template <Elementwise op, IntegerOperation int_op>
at::Tensor typed_unary(const at::Tensor& input) {
    check_device(input.device());
    const auto type = input.scalar_type();
    if (supported_float(type)) return unary_strided(op, input);
    return integral_binary(input, input, type, type, int_op, 1);
}

template <at::Tensor (*fn)(const at::Tensor&), Elementwise op>
at::Tensor& typed_unary_out(const at::Tensor& input, at::Tensor& output) {
    if (supported_float(input.scalar_type())) return pointwise_unary_out<op>(input, output);
    check_device(output.device());
    TORCH_CHECK(input.scalar_type() == output.scalar_type(), "Iris unary out dtype must match input");
    const auto result = fn(input);
    (void)at::TensorIteratorConfig().add_output(output).add_const_input(input).build();
    copy_tensor(output, result);
    return output;
}

template <at::Tensor (*fn)(const at::Tensor&), Elementwise op>
at::Tensor& typed_unary_inplace(at::Tensor& input) {
    if (supported_float(input.scalar_type())) return pointwise_unary_inplace<op>(input);
    at::assert_no_internal_overlap(input);
    copy_tensor(input, fn(input));
    return input;
}

at::Tensor round_decimals(const at::Tensor& input, int64_t decimals) {
    check_device(input.device());
    if (!supported_float(input.scalar_type())) return input.clone();
    auto output = make_output(input.sizes(), input.scalar_type());
    const auto scale = std::pow(10.0f, std::abs(static_cast<float>(decimals)));
    run_strided(Elementwise::RoundDecimals, output, input, input, scale, decimals >= 0 ? 1.0f : 0.0f);
    return output;
}

at::Tensor round_tensor(const at::Tensor& input) {
    check_device(input.device());
    return supported_float(input.scalar_type()) ? unary_strided(Elementwise::Round, input) : input.clone();
}

at::Tensor& round_decimals_out(const at::Tensor& input, int64_t decimals, at::Tensor& output) {
    check_device(output.device());
    TORCH_CHECK(input.scalar_type() == output.scalar_type(), "round requires matching output dtype");
    const auto result = round_decimals(input, decimals);
    at::native::resize_output(output, result.sizes());
    copy_tensor(output, result);
    return output;
}

at::Tensor& round_decimals_inplace(at::Tensor& input, int64_t decimals) {
    return round_decimals_out(input, decimals, input);
}

at::Tensor softplus(const at::Tensor& input, const at::Scalar& beta, const at::Scalar& threshold) {
    check_operand(input);
    auto output = make_output(input.sizes(), input.scalar_type());
    run_strided(Elementwise::Softplus, output, input, input, beta.toFloat(), threshold.toFloat());
    return output;
}

at::Tensor softplus_backward(const at::Tensor& grad, const at::Tensor& input,
                             const at::Scalar& beta, const at::Scalar& threshold) {
    check_operand(grad);
    check_operand(input);
    TORCH_CHECK(grad.scalar_type() == input.scalar_type(), "softplus requires matching gradient dtype");
    auto output = make_output(input.sizes(), input.scalar_type());
    run_strided(Elementwise::SoftplusBackward, output, grad, input, beta.toFloat(), threshold.toFloat());
    return output;
}

std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward(const at::Tensor& input) {
    check_operand(input);
    return {unary_strided(Elementwise::LogSigmoid, input), unary_strided(Elementwise::LogSigmoidBuffer, input)};
}

at::Tensor log_sigmoid_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor&) {
    return pointwise_binary<Elementwise::LogSigmoidBackward>(grad, input);
}

at::Tensor unfold(const at::Tensor& input, int64_t dim, int64_t size, int64_t step) {
    dim = at::maybe_wrap_dim(dim, input.dim());
    const int64_t length = input.dim() ? input.size(dim) : 1;
    TORCH_CHECK(size >= 0 && size <= length, "unfold size must be between zero and the dimension length");
    TORCH_CHECK(step > 0, "unfold step must be positive");
    auto sizes = input.sizes().vec();
    auto strides = input.strides().vec();
    const int64_t stride = input.dim() ? input.stride(dim) : 1;
    sizes.push_back(size);
    strides.push_back(stride);
    if (input.dim()) {
        sizes[dim] = (length - size) / step + 1;
        strides[dim] *= step;
    }
    return input.as_strided(sizes, strides);
}

at::Tensor unfold_backward(const at::Tensor& grad, c10::SymIntArrayRef input_sizes,
                           int64_t dim, int64_t size, int64_t step) {
    const auto sizes = C10_AS_INTARRAYREF_SLOW(input_sizes);
    if (sizes.empty()) return grad.sum();
    dim = at::maybe_wrap_dim(dim, sizes.size());
    auto output = at::zeros(sizes, grad.options());
    if (grad.numel() == 0) return output;
    const auto options = grad.options().dtype(at::kLong);
    const auto starts = at::arange(0, (grad.size(dim) - 1) * step + 1, step, options);
    const auto indices = starts.unsqueeze(1).add(at::arange(size, options).unsqueeze(0)).reshape({-1});
    return output.index_add(dim, indices, grad.movedim(-1, dim + 1).flatten(dim, dim + 1));
}

at::Tensor& eye_out(c10::SymInt rows, c10::SymInt columns, at::Tensor& output) {
    check_device(output.device());
    const auto height = rows.expect_int();
    const auto width = columns.expect_int();
    TORCH_CHECK(height >= 0 && width >= 0, "eye dimensions must be nonnegative");
    at::native::resize_output(output, {height, width});
    const auto options = output.options().dtype(at::kLong);
    const auto diagonal = at::arange(height, options).unsqueeze(1).eq(at::arange(width, options));
    copy_tensor(output, diagonal);
    return output;
}

void foreach_add_scalar(at::TensorList tensors, const at::Scalar& scalar) {
    constexpr std::size_t batch_size = 8;
    std::unordered_set<cl_mem> seen;
    for (const auto& tensor : tensors) {
        if (!is_iris(tensor) || tensor.scalar_type() != at::kFloat || tensor.numel() != 1 ||
            !seen.insert(buffer(tensor)).second) {
            for (auto fallback : tensors) fallback.add_(scalar);
            return;
        }
    }
    for (std::size_t i = 0; i < tensors.size(); i += batch_size) {
        std::array<BufferSlice, batch_size> batch{};
        const auto count = std::min(batch_size, tensors.size() - i);
        for (std::size_t j = 0; j < count; ++j) {
            const auto& tensor = tensors[i + j];
            batch[j] = {buffer(tensor), static_cast<cl_ulong>(tensor.storage_offset())};
        }
        iris::add_scalar_batch(batch, count, scalar.toFloat());
    }
}

void check_amp_scalar(const at::Tensor& scalar, at::ScalarType dtype) {
    check_device(scalar.device());
    TORCH_CHECK(scalar.numel() == 1 && scalar.scalar_type() == dtype, "Invalid Iris gradient scaling state");
}

void amp_unscale(at::TensorList grads, at::Tensor& found_inf, const at::Tensor& inv_scale) {
    check_amp_scalar(found_inf, at::kFloat);
    check_amp_scalar(inv_scale, at::kFloat);
    for (const auto& grad : grads) {
        check_operand(grad);
        at::assert_no_internal_overlap(grad);
    }
    for (auto grad : grads) {
        if (!grad.numel()) continue;
        auto work = grad.contiguous();
        iris::amp_unscale({buffer(work), static_cast<cl_ulong>(work.storage_offset())},
                         {buffer(found_inf), static_cast<cl_ulong>(found_inf.storage_offset())},
                         {buffer(inv_scale), static_cast<cl_ulong>(inv_scale.storage_offset())},
                         work.numel(), work.scalar_type() == at::kHalf);
        if (!work.is_same(grad)) copy_tensor(grad, work);
    }
}

at::Tensor& amp_update_scale(at::Tensor& scale, at::Tensor& growth_tracker, const at::Tensor& found_inf,
                             double growth_factor, double backoff_factor, int64_t growth_interval) {
    check_amp_scalar(scale, at::kFloat);
    check_amp_scalar(growth_tracker, at::kInt);
    check_amp_scalar(found_inf, at::kFloat);
    TORCH_CHECK(growth_interval > 0 && growth_interval <= std::numeric_limits<cl_int>::max(),
                "Iris GradScaler growth interval is out of range");
    iris::amp_update_scale({buffer(scale), static_cast<cl_ulong>(scale.storage_offset())},
                           {buffer(growth_tracker), static_cast<cl_ulong>(growth_tracker.storage_offset())},
                           {buffer(found_inf), static_cast<cl_ulong>(found_inf.storage_offset())},
                           static_cast<float>(growth_factor), static_cast<float>(backoff_factor),
                           static_cast<cl_int>(growth_interval));
    return scale;
}

at::Tensor make_mask(const at::Tensor& left, const at::Tensor& right) {
    at::Tensor undefined;
    return at::TensorIteratorConfig().add_output(undefined).add_const_input(left).add_const_input(right)
        .check_all_same_dtype(false).declare_static_dtype(at::kBool).build().output();
}

template <Elementwise op>
at::Tensor predicate(const at::Tensor& input) {
    check_operand(input);
    auto output = make_mask(input, input);
    run_strided(op, output, input, input);
    return output;
}

template <Elementwise op>
at::Tensor compare_scalar(const at::Tensor& left, const at::Scalar& right) {
    check_device(left.device());
    if (!supported_float(left.scalar_type())) {
        const auto type = at::result_type(left, right);
        if (supported_float(type)) {
            return compare_scalar<op>(left.to(type), right);
        }
        auto scalar = make_output({}, type);
        fill_scalar(scalar, right);
        constexpr auto operation = static_cast<IntegerOperation>(
            static_cast<cl_uint>(op) - static_cast<cl_uint>(Elementwise::Eq) +
            static_cast<cl_uint>(IntegerOperation::Eq));
        return integral_binary(left, scalar, type, at::kBool, operation, 1);
    }
    auto output = make_mask(left, left);
    run_strided(op, output, left, left, right.toFloat(), 1.0f);
    return output;
}

template <Elementwise op>
at::Tensor compare_tensor(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    if (right.device().is_cpu() && right.dim() == 0) {
        return compare_scalar<op>(left, right.item());
    }
    check_device(right.device());
    const auto dtype = at::result_type(left, right);
    if (!supported_float(dtype)) {
        constexpr auto operation = static_cast<IntegerOperation>(
            static_cast<cl_uint>(op) - static_cast<cl_uint>(Elementwise::Eq) +
            static_cast<cl_uint>(IntegerOperation::Eq));
        return integral_binary(left, right, dtype, at::kBool, operation, 1);
    }
    const auto a = left.to(dtype);
    const auto b = right.to(dtype);
    auto output = make_mask(a, b);
    run_strided(op, output, a, b);
    return output;
}

at::Tensor bool_and(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    check_device(right.device());
    const auto dtype = at::result_type(left, right);
    TORCH_CHECK(!supported_float(dtype), "bitwise_and is not supported for floating point tensors");
    return integral_binary(left, right, dtype, dtype, IntegerOperation::BitwiseAnd, 1);
}

template <IntegerOperation op>
at::Tensor bitwise_tensor(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    if (right.device().is_cpu() && right.dim() == 0) {
        const auto dtype = at::result_type(left, right);
        TORCH_CHECK(!supported_float(dtype), "bitwise op is not supported for floating point tensors");
        return integral_binary(left, right, dtype, dtype, op, 1);
    }
    check_device(right.device());
    const auto dtype = at::result_type(left, right);
    TORCH_CHECK(!supported_float(dtype), "bitwise op is not supported for floating point tensors");
    return integral_binary(left, right, dtype, dtype, op, 1);
}

template <IntegerOperation op>
at::Tensor bitwise_scalar(const at::Tensor& left, const at::Scalar& right) {
    check_device(left.device());
    const auto dtype = at::result_type(left, right);
    TORCH_CHECK(!supported_float(dtype), "bitwise op is not supported for floating point tensors");
    auto scalar = make_output({}, dtype);
    fill_scalar(scalar, right);
    return integral_binary(left, scalar, dtype, dtype, op, 1);
}

template <IntegerOperation op>
at::Tensor& bitwise_scalar_out(const at::Tensor& left, const at::Scalar& right, at::Tensor& out) {
    check_device(out.device());
    auto res = bitwise_scalar<op>(left, right);
    at::native::resize_output(out, res.sizes());
    copy_tensor(out, res);
    return out;
}

at::Tensor bitwise_not_tensor(const at::Tensor& input) {
    check_device(input.device());
    TORCH_CHECK(!supported_float(input.scalar_type()), "bitwise_not is not supported for floating point tensors");
    return integral_binary(input, input, input.scalar_type(), input.scalar_type(), IntegerOperation::BitwiseNot, 1);
}

at::Tensor& bitwise_not_out(const at::Tensor& input, at::Tensor& out) {
    check_device(out.device());
    auto res = bitwise_not_tensor(input);
    at::native::resize_output(out, res.sizes());
    copy_tensor(out, res);
    return out;
}

template <Elementwise op>
at::Tensor logical_binary(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    check_device(right.device());
    if (supported_float(left.scalar_type()) && left.scalar_type() == right.scalar_type()) {
        auto output = make_mask(left, right);
        run_strided(op, output, left, right);
        return output;
    }
    const auto a = left.to(at::kBool);
    const auto b = right.to(at::kBool);
    constexpr auto iop = op == Elementwise::LogicalAnd ? IntegerOperation::BitwiseAnd
                       : op == Elementwise::LogicalOr  ? IntegerOperation::BitwiseOr
                                                       : IntegerOperation::BitwiseXor;
    return integral_binary(a, b, at::kBool, at::kBool, iop, 1);
}

at::Tensor logical_not_tensor(const at::Tensor& input) {
    check_device(input.device());
    if (supported_float(input.scalar_type())) {
        auto output = make_mask(input, input);
        run_strided(Elementwise::LogicalNot, output, input, input);
        return output;
    }
    return integral_binary(input.to(at::kBool), input.to(at::kBool), at::kBool, at::kBool,
                           IntegerOperation::BitwiseNot, 1);
}

at::Tensor& logical_not_out(const at::Tensor& input, at::Tensor& out) {
    check_device(out.device());
    auto res = logical_not_tensor(input);
    at::native::resize_output(out, res.sizes());
    copy_tensor(out, res);
    return out;
}

at::Tensor neg_tensor(const at::Tensor& input) {
    check_device(input.device());
    if (supported_float(input.scalar_type())) {
        return unary_strided(Elementwise::Neg, input);
    }
    TORCH_CHECK(input.scalar_type() != at::kBool, "Negation, the `-` operator, on a bool tensor is not supported.");
    return integral_binary(input, input, input.scalar_type(), input.scalar_type(), IntegerOperation::Neg, 1);
}

at::Tensor& neg_inplace(at::Tensor& input) {
    at::assert_no_internal_overlap(input);
    if (supported_float(input.scalar_type())) {
        run_strided(Elementwise::Neg, input, input, input);
        return input;
    }
    copy_tensor(input, neg_tensor(input));
    return input;
}

at::Tensor& neg_out(const at::Tensor& input, at::Tensor& out) {
    check_device(out.device());
    auto res = neg_tensor(input);
    at::native::resize_output(out, res.sizes());
    copy_tensor(out, res);
    return out;
}

at::Tensor sub_scalar(const at::Tensor& input, const at::Scalar& other, const at::Scalar& alpha) {
    check_device(input.device());
    if (supported_float(input.scalar_type())) {
        return unary_strided(Elementwise::SubScalar, input, other.toFloat() * alpha.toFloat());
    }
    const auto dtype = at::result_type(input, other);
    auto scalar = make_output({}, dtype);
    fill_scalar(scalar, other);
    if (supported_float(dtype)) {
        return binary_strided(Elementwise::Subtract, input.to(dtype), scalar, alpha.toFloat());
    }
    return integral_binary(input, scalar, dtype, dtype, IntegerOperation::Subtract, alpha);
}

at::Tensor rsub_scalar(const at::Tensor& input, const at::Scalar& other, const at::Scalar& alpha) {
    check_device(input.device());
    if (supported_float(input.scalar_type())) {
        at::Tensor undefined;
        auto output = at::TensorIteratorConfig().add_output(undefined).add_const_input(input).build().output();
        run_strided(Elementwise::RsubScalar, output, input, input, other.toFloat(), alpha.toFloat());
        return output;
    }
    const auto dtype = at::result_type(input, other);
    auto scalar = make_output({}, dtype);
    fill_scalar(scalar, other);
    if (supported_float(dtype)) {
        return binary_strided(Elementwise::Subtract, scalar, input.to(dtype), alpha.toFloat());
    }
    return integral_binary(scalar, input, dtype, dtype, IntegerOperation::Subtract, alpha);
}

template <Elementwise op>
at::Tensor pointwise_scalar(const at::Tensor& input, const at::Scalar& scalar) {
    if constexpr (op == Elementwise::Scale) {
        if (!supported_float(input.scalar_type()) || scalar.isComplex()) {
            const auto dtype = at::result_type(input, scalar);
            auto value = make_output({}, dtype);
            fill_scalar(value, scalar);
            return promoted_arithmetic(input, value, Elementwise::Multiply, IntegerOperation::Multiply, 1);
        }
    }
    check_operand(input);
    if constexpr (op == Elementwise::Scale || op == Elementwise::AddScalar) {
        if (input.scalar_type() == at::kFloat && packed(input)) {
            auto output = make_output(input.sizes());
            elementwise(op, buffer(output), buffer(input), nullptr, output.numel(), scalar.toFloat());
            return output;
        }
    }
    return unary_strided(op, input, scalar.toFloat());
}

at::Tensor scale_add_op(const at::Tensor& input, double scale, double bias) {
    check_operand(input);
    if (input.scalar_type() != at::kFloat || !packed(input)) {
        return input.mul(scale).add(bias);
    }
    auto output = make_output(input.sizes());
    iris::scale_add(buffer(output), buffer(input), output.numel(), static_cast<float>(scale), static_cast<float>(bias));
    return output;
}

at::Tensor fft_radix2_op(const at::Tensor& real, const at::Tensor& imaginary, bool forward) {
    check_operand(real);
    check_operand(imaginary);
    TORCH_CHECK(real.scalar_type() == at::kFloat && imaginary.scalar_type() == at::kFloat &&
                real.sizes() == imaginary.sizes() && real.dim() > 0,
                "Iris radix-2 FFT requires matching float32 components");
    const auto length = real.size(-1);
    TORCH_CHECK(length >= 2 && length <= 1024 && (length & (length - 1)) == 0,
                "Iris fused FFT requires a power-of-two length from 2 to 1024");
    auto shape = real.sizes().vec();
    shape.push_back(2);
    auto output = make_output(shape);
    if (real.numel()) {
        const auto r = real.resolve_neg().contiguous();
        const auto i = imaginary.resolve_neg().contiguous();
        fft_radix2({buffer(output), 0}, {buffer(r), static_cast<cl_ulong>(r.storage_offset())},
                   {buffer(i), static_cast<cl_ulong>(i.storage_offset())}, real.numel() / length,
                   static_cast<cl_uint>(length), forward);
    }
    return output;
}

template <Elementwise op>
at::Tensor& pointwise_scalar_inplace(at::Tensor& input, const at::Scalar& scalar) {
    check_operand(input);
    at::assert_no_internal_overlap(input);
    run_strided(op, input, input, input, scalar.toFloat());
    return input;
}

at::Tensor divide_tensor(const at::Tensor& left, const at::Tensor& right) {
    TORCH_CHECK(is_iris(left) || (left.device().is_cpu() && left.dim() == 0), "Expected an Iris tensor or CPU scalar");
    TORCH_CHECK(is_iris(right) || (right.device().is_cpu() && right.dim() == 0), "Expected an Iris tensor or CPU scalar");
    const auto device = is_iris(left) ? left.device() : right.device();
    const auto dtype = at::result_type(left, right);
    const auto promoted = c10::isIntegralType(dtype, true) ? c10::get_default_dtype_as_scalartype() : dtype;
    TORCH_CHECK(supported_float(promoted), "Iris division requires a float32 or float16 result");
    if (right.device().is_cpu() && right.dim() == 0) {
        return divide_scalar(left.to(device, promoted), right.item());
    }
    return pointwise_binary<Elementwise::DivideTensor>(left.to(device, promoted), right.to(device, promoted));
}

at::Tensor divide_mode(const at::Tensor& left, const at::Tensor& right,
                       std::optional<c10::string_view> mode) {
    if (!mode) return divide_tensor(left, right);
    TORCH_CHECK(*mode == "trunc" || *mode == "floor", "div expected rounding_mode to be 'trunc' or 'floor'");
    const bool floor_mode = *mode == "floor";
    const auto dtype = at::result_type(left, right);
    TORCH_CHECK(is_iris(left) || (left.is_cpu() && left.dim() == 0), "Expected an Iris tensor or CPU scalar");
    TORCH_CHECK(is_iris(right) || (right.is_cpu() && right.dim() == 0), "Expected an Iris tensor or CPU scalar");
    const auto device = is_iris(left) ? left.device() : right.device();
    if (c10::isIntegralType(dtype, true)) {
        TORCH_CHECK(dtype != at::kBool, "rounded division is not implemented for Bool");
        const auto shape = at::infer_size_dimvector(left.sizes(), right.sizes());
        const auto divisor = right.to(device, dtype);
        TORCH_CHECK(c10::multiply_integers(shape) == 0 || !at::any(at::eq(divisor, 0)).item<bool>(),
                    "ZeroDivisionError");
        return integral_binary(left, divisor, dtype, dtype,
                               floor_mode ? IntegerOperation::DivideFloor : IntegerOperation::DivideTrunc, 1);
    }
    TORCH_CHECK(supported_float(dtype), "Iris division requires a supported real dtype");
    return binary_strided(floor_mode ? Elementwise::DivideFloor : Elementwise::DivideTrunc,
                          left.to(device, dtype), right.to(device, dtype), 1.0f);
}

at::Tensor floor_divide_tensor(const at::Tensor& left, const at::Tensor& right) {
    return divide_mode(left, right, c10::string_view("floor"));
}

at::Tensor pow_scalar_base(const at::Scalar& base, const at::Tensor& exponent) {
    check_device(exponent.device());
    const auto dtype = at::result_type(base, exponent);
    const auto promoted = c10::isIntegralType(dtype, true) ? c10::get_default_dtype_as_scalartype() : dtype;
    TORCH_CHECK(supported_float(promoted), "Iris pow.Scalar requires float32 or float16");
    auto base_t = make_output({}, promoted);
    fill_scalar(base_t, base);
    return pointwise_binary<Elementwise::PowTensor>(base_t, exponent.to(promoted));
}

at::Tensor add_scalar(const at::Tensor& input, const at::Scalar& value, const at::Scalar& alpha) {
    check_device(input.device());
    if (!supported_float(input.scalar_type()) || value.isComplex()) {
        const auto dtype = at::result_type(input, value);
        auto scalar = make_output({}, dtype);
        fill_scalar(scalar, value);
        return promoted_arithmetic(input.to(dtype), scalar, Elementwise::Add, IntegerOperation::Add, alpha);
    }
    return pointwise_scalar<Elementwise::AddScalar>(input, value.toFloat() * alpha.toFloat());
}

at::Tensor& add_scalar_inplace(at::Tensor& input, const at::Scalar& value, const at::Scalar& alpha) {
    check_device(input.device());
    at::assert_no_internal_overlap(input);
    if (!supported_float(input.scalar_type())) {
        const auto result = add_scalar(input, value, alpha);
        TORCH_CHECK(c10::canCast(result.scalar_type(), input.scalar_type()), "result type cannot be cast for add_");
        copy_tensor(input, result);
        return input;
    }
    return pointwise_scalar_inplace<Elementwise::AddScalar>(input, value.toFloat() * alpha.toFloat());
}

at::Tensor subtract_tensor(const at::Tensor& left, const at::Tensor& right, const at::Scalar& alpha) {
    if (supported_float(left.scalar_type()) && !right.is_complex() && right.device().is_cpu() && right.dim() == 0) {
        return add_scalar(left, right.item(), -alpha.toFloat());
    }
    return promoted_arithmetic(left, right, Elementwise::Subtract, IntegerOperation::Subtract, alpha);
}

at::Tensor clamp_tensor(const at::Tensor& input, const std::optional<at::Scalar>& lower,
                         const std::optional<at::Scalar>& upper) {
    check_device(input.device());
    TORCH_CHECK(lower || upper, "torch.clamp: At least one of 'min' or 'max' must not be None");
    TORCH_CHECK((!lower || !lower->isComplex()) && (!upper || !upper->isComplex()),
                "clamp is not supported for complex types");
    const auto representative = lower && (lower->isFloatingPoint() || !upper) ? *lower : *upper;
    const auto dtype = at::result_type(input, representative);
    auto source = input.to(dtype);
    if (!supported_float(dtype)) {
        TORCH_CHECK(dtype != at::kBool, "clamp is not implemented for Bool");
        auto bound = make_output({}, dtype);
        if (lower) {
            fill_scalar(bound, *lower);
            source = integral_binary(source, bound, dtype, dtype, IntegerOperation::Maximum, 1);
        }
        if (upper) {
            fill_scalar(bound, *upper);
            source = integral_binary(source, bound, dtype, dtype, IntegerOperation::Minimum, 1);
        }
        return source;
    }
    const auto inf = std::numeric_limits<float>::infinity();
    auto output = at::empty_like(source);
    run_strided(Elementwise::Clamp, output, source, source,
                lower ? lower->toFloat() : -inf, upper ? upper->toFloat() : inf);
    return output;
}

Elementwise gelu_op(c10::string_view approximation, bool backward) {
    TORCH_CHECK(approximation == "none" || approximation == "tanh",
                "gelu approximate must be 'none' or 'tanh'");
    if (approximation == "tanh") {
        return backward ? Elementwise::GeluTanhBackward : Elementwise::GeluTanh;
    }
    return backward ? Elementwise::GeluBackward : Elementwise::Gelu;
}

at::Tensor gelu(const at::Tensor& input, c10::string_view approximation) {
    check_operand(input);
    return unary_strided(gelu_op(approximation, false), input);
}

at::Tensor gelu_backward(const at::Tensor& grad, const at::Tensor& input, c10::string_view approximation) {
    check_operand(grad);
    check_operand(input);
    return binary_strided(gelu_op(approximation, true), grad, input, 1.0f);
}

at::Tensor& lerp_scalar_inplace(at::Tensor& input, const at::Tensor& end, const at::Scalar& weight) {
    // Reuse the same staged delta for broadcast inputs and safe whole-tensor aliases.
    const auto difference = subtract_tensor(end, input, 1.0);
    return add_inplace(input, difference, weight);
}

at::Tensor& addcmul_inplace(
    at::Tensor& input, const at::Tensor& a, const at::Tensor& b, const at::Scalar& scale) {
    return add_inplace(input, multiply_tensor(a, b), scale);
}

at::Tensor& addcdiv_inplace(
    at::Tensor& input, const at::Tensor& a, const at::Tensor& b, const at::Scalar& scale) {
    return add_inplace(input, divide_tensor(a, b), scale);
}

template <Elementwise op, IntegerOperation int_op>
at::Tensor typed_extrema(const at::Tensor& left, const at::Tensor& right) {
    const auto dtype = at::result_type(left, right);
    if (supported_float(dtype)) return pointwise_binary<op>(left.to(dtype), right.to(dtype));
    return integral_binary(left, right, dtype, dtype, int_op, 1);
}

at::Tensor& maximum_out(const at::Tensor& left, const at::Tensor& right, at::Tensor& output) {
    auto result = typed_extrema<Elementwise::Maximum, IntegerOperation::Maximum>(left, right);
    check_device(output.device());
    at::native::resize_output(output, result.sizes());
    copy_tensor(output, result);
    return output;
}

template <Extrema op>
at::Tensor extrema_all(const at::Tensor& input) {
    check_device(input.device());
    const auto type = value_type(input.scalar_type());
    constexpr bool numerical = op == Extrema::Min || op == Extrema::Max;
    if constexpr (numerical) {
        TORCH_CHECK(!input.is_complex(), "min/max is not supported for complex inputs");
        TORCH_CHECK(input.numel(), "Iris min/max requires a reduction dimension for empty tensors");
    }
    const auto output_type = numerical ? input.scalar_type() :
        input.scalar_type() == at::kByte ? at::kByte : at::kBool;
    auto output = make_output({}, output_type);
    c10::SmallVector<cl_ulong, 16> geometry;
    for (const auto dim : c10::irange(input.dim())) {
        geometry.append({static_cast<cl_ulong>(input.size(dim)),
                         static_cast<cl_ulong>(input.stride(dim))});
    }
    extrema(buffer(output), buffer(input), input.numel(), input.storage_offset(), geometry, type, op);
    return output;
}

template <Extrema op>
at::Tensor extrema_dims(const at::Tensor& input, at::IntArrayRef dims, bool keepdim) {
    check_device(input.device());
    TORCH_CHECK(!input.is_complex(), "amin/amax is not supported for complex inputs");
    if (input.scalar_type() == at::kBFloat16) {
        return extrema_dims<op>(input.to(at::kFloat), dims, keepdim).to(at::kBFloat16);
    }
    if (input.dim() == 0) {
        TORCH_CHECK(dims.size() <= 1, "reduction dimension appears multiple times");
        if (!dims.empty()) at::maybe_wrap_dim(dims[0], int64_t{0});
        return input.clone();
    }
    if (dims.empty() && input.numel()) {
        auto result = extrema_all<op>(input);
        return keepdim ? result.reshape(at::DimVector(input.dim(), 1)) : result;
    }
    std::vector<bool> reduced(input.dim(), dims.empty());
    for (const auto dim : dims) {
        const auto d = at::maybe_wrap_dim(dim, input.dim());
        TORCH_CHECK(!reduced[d], "reduction dimension ", dim, " appears multiple times");
        reduced[d] = true;
    }
    at::DimVector permutation, shape, kept;
    int64_t width = 1;
    for (const auto d : c10::irange(input.dim())) {
        if (reduced[d]) {
            TORCH_CHECK_INDEX(input.size(d) != 0, "Expected non-empty reduction dimension ", d);
            width *= input.size(d);
            if (keepdim) shape.push_back(1);
        } else {
            permutation.push_back(d);
            kept.push_back(input.size(d));
            shape.push_back(input.size(d));
        }
    }
    for (const auto d : c10::irange(input.dim())) {
        if (reduced[d]) permutation.push_back(d);
    }
    kept.push_back(width);
    const auto rows = input.permute(permutation).reshape(kept);
    auto result = op == Extrema::Min ? std::get<0>(at::min(rows, -1)) : std::get<0>(at::max(rows, -1));
    return result.reshape(shape);
}

std::tuple<at::Tensor, at::Tensor> aminmax_impl(const at::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
    const auto dims = dim ? at::IntArrayRef(*dim) : at::IntArrayRef{};
    return {extrema_dims<Extrema::Min>(input, dims, keepdim), extrema_dims<Extrema::Max>(input, dims, keepdim)};
}

at::Tensor masked_select(const at::Tensor& input, const at::Tensor& mask) {
    check_device(input.device());
    check_device(mask.device());
    TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_select expects a BoolTensor for mask");
    const auto sizes = at::infer_size_dimvector(input.sizes(), mask.sizes());
    auto expanded_input = input.expand(sizes);
    auto expanded_mask = mask.expand(sizes);
    auto output = make_output({expanded_input.numel()}, input.scalar_type());
    if (!output.numel()) {
        return output;
    }
    auto count = make_output({}, at::kLong);
    const auto geometry = elementwise_geometry(expanded_input, expanded_input, expanded_mask);
    masked_copy(buffer(output), buffer(count), buffer(input), buffer(mask), output.numel(),
                input.storage_offset(), mask.storage_offset(), geometry, value_type(input.scalar_type()));
    return output.narrow(0, 0, read_scalar<int64_t>(count));
}

at::Tensor concatenate(const at::ITensorListRef& tensors, int64_t dim) {
    TORCH_CHECK(!tensors.empty(), "torch.cat(): expected a non-empty list of Tensors");
    at::Tensor first;
    auto dtype = (*tensors.begin()).scalar_type();
    for (const auto& tensor : tensors) {
        check_device(tensor.device());
        dtype = c10::promoteTypes(dtype, tensor.scalar_type());
        if (!first.defined() && !(tensor.dim() == 1 && !tensor.numel())) {
            first = tensor;
        }
    }
    (void)value_type(dtype);
    if (!first.defined()) {
        at::maybe_wrap_dim(dim, 1);
        return make_output({0}, dtype);
    }
    TORCH_CHECK(first.dim() > 0, "zero-dimensional tensor cannot be concatenated");
    dim = at::maybe_wrap_dim(dim, first.dim());
    at::DimVector sizes(first.sizes());
    sizes[dim] = 0;
    for (const auto& tensor : tensors) {
        if (tensor.dim() == 1 && !tensor.numel()) {
            continue;
        }
        TORCH_CHECK(tensor.dim() == first.dim(), "torch.cat(): tensors must have the same rank");
        for (const auto axis : c10::irange(first.dim())) {
            TORCH_CHECK(axis == dim || tensor.size(axis) == first.size(axis),
                        "Sizes of tensors must match except in the concatenation dimension");
        }
        sizes[dim] += tensor.size(dim);
    }
    auto output = make_output(sizes, dtype);
    int64_t offset = 0;
    for (const auto& tensor : tensors) {
        if (!tensor.numel()) {
            continue;
        }
        copy_tensor(output.narrow(dim, offset, tensor.size(dim)), tensor);
        offset += tensor.size(dim);
    }
    return output;
}

void check_indices(const at::Tensor& index, bool allow_int = false) {
    check_device(index.device());
    TORCH_CHECK(index.scalar_type() == at::kLong || (allow_int && index.scalar_type() == at::kInt),
                "Iris indices must be ", allow_int ? "int64 or int32" : "int64");
}

void run_indexed_copy(const at::Tensor& output, const at::Tensor& input, const at::Tensor& index,
                      int64_t axis, at::IntArrayRef shape, IndexOperation operation, bool include_self = true) {
    const bool scatter = operation >= IndexOperation::Scatter;
    const auto length = c10::multiply_integers(shape);
    if (!length) {
        return;
    }
    auto invalid = make_output({}, at::kInt);
    zero(invalid);
    c10::SmallVector<cl_ulong, 24> geometry;
    for (const auto d : c10::irange(shape.size())) {
        const auto index_stride = operation == IndexOperation::Select ? (d == axis ? index.stride(0) : 0)
                                                                    : index.stride(d);
        geometry.append({static_cast<cl_ulong>(shape[d]), static_cast<cl_ulong>(output.stride(d)),
                         static_cast<cl_ulong>(input.stride(d)), static_cast<cl_ulong>(index_stride)});
    }
    indexed_copy(buffer(output), buffer(input), buffer(index), buffer(invalid),
                 scatter ? length / shape[axis] : length, output.storage_offset(),
                 input.storage_offset(), index.storage_offset(), geometry, axis,
                 scatter ? output.size(axis) : input.size(axis), value_type(input.scalar_type()),
                 value_type(index.scalar_type()), operation, include_self);
    TORCH_CHECK_INDEX(!read_scalar<int32_t>(invalid), "Iris index is out of bounds for dimension ", axis);
}

at::Tensor index_select(const at::Tensor& input, int64_t axis, const at::Tensor& indices) {
    check_device(input.device());
    TORCH_CHECK(indices.dim() <= 1, "index_select(): Index is supposed to be a vector");
    const auto index = indices.to(input.device()).reshape({-1});
    check_indices(index, true);
    axis = at::maybe_wrap_dim(axis, input.dim());
    const auto source = input.dim() ? input : input.unsqueeze(0);
    at::DimVector sizes(source.sizes());
    sizes[axis] = index.numel();
    auto output = make_output(sizes, input.scalar_type());
    run_indexed_copy(output, source, index, axis, sizes, IndexOperation::Select);
    if (!input.dim()) {
        TORCH_CHECK(index.numel() == 1, "index_select(): Index to scalar can have only 1 value");
        return output.squeeze(0);
    }
    return output;
}

at::Tensor gather(const at::Tensor& input, int64_t axis, const at::Tensor& indices, bool) {
    check_device(input.device());
    check_indices(indices);
    axis = at::maybe_wrap_dim(axis, input.dim());
    auto output = make_output(indices.sizes(), input.scalar_type());
    if (!indices.numel()) {
        return output;
    }
    const auto source = input.dim() ? input : input.unsqueeze(0);
    const auto index = indices.dim() ? indices : indices.unsqueeze(0);
    TORCH_CHECK(index.dim() == source.dim(), "Index tensor must have the same number of dimensions as input tensor");
    for (const auto d : c10::irange(source.dim())) {
        TORCH_CHECK(d == axis || index.size(d) <= source.size(d),
                    "Size does not match at dimension ", d, " for gather");
    }
    auto view = output.dim() ? output : output.unsqueeze(0);
    run_indexed_copy(view, source, index, axis, index.sizes(), IndexOperation::Gather);
    return output;
}

at::Tensor scatter_impl(const at::Tensor& input, int64_t axis, const at::Tensor& indices,
                        const at::Tensor& source, IndexOperation operation, bool include_self = true) {
    check_device(input.device());
    check_device(source.device());
    check_indices(indices);
    TORCH_CHECK(input.scalar_type() == source.scalar_type(), "scatter requires matching input and source dtypes");
    TORCH_CHECK(operation != IndexOperation::ReduceMean || input.scalar_type() != at::kBool,
                "scatter_reduce mean is not implemented for Bool");
    TORCH_CHECK(!input.is_complex() || operation <= IndexOperation::ScatterAdd,
                "Iris scatter_reduce requires real inputs");
    axis = at::maybe_wrap_dim(axis, input.dim());
    auto output = input.clone();
    if (!indices.numel()) {
        return output;
    }
    auto view = output.dim() ? output : output.unsqueeze(0);
    const auto src = source.dim() ? source : source.unsqueeze(0);
    const auto index = indices.dim() ? indices : indices.unsqueeze(0);
    TORCH_CHECK(index.dim() == view.dim() && index.dim() == src.dim(),
                "Index tensor must have the same number of dimensions as the input and source tensors");
    for (const auto d : c10::irange(view.dim())) {
        TORCH_CHECK((d == axis || index.size(d) <= view.size(d)) && index.size(d) <= src.size(d),
                    "Size does not match at dimension ", d, " for scatter");
    }
    run_indexed_copy(view, src, index, axis, index.sizes(), operation, include_self);
    return output;
}

template <IndexOperation operation>
at::Tensor scatter(const at::Tensor& input, int64_t axis, const at::Tensor& indices, const at::Tensor& source) {
    return scatter_impl(input, axis, indices, source, operation);
}

at::Tensor scatter_reduce_impl(const at::Tensor& input, int64_t axis, const at::Tensor& indices,
                          const at::Tensor& source, c10::string_view reduce, bool include_self) {
    auto operation = IndexOperation::ReduceSum;
    if (reduce == "prod") operation = IndexOperation::ReduceProduct;
    else if (reduce == "mean") operation = IndexOperation::ReduceMean;
    else if (reduce == "amax") operation = IndexOperation::ReduceMax;
    else if (reduce == "amin") operation = IndexOperation::ReduceMin;
    else TORCH_CHECK(reduce == "sum", "reduce must be sum, prod, mean, amax or amin");
    if (include_self && operation == IndexOperation::ReduceSum) operation = IndexOperation::ScatterAdd;
    return scatter_impl(input, axis, indices, source, operation, include_self);
}

at::Tensor index_copy_impl(const at::Tensor& input, int64_t axis, const at::Tensor& indices, const at::Tensor& source) {
    check_indices(indices);
    TORCH_CHECK(indices.dim() <= 1, "index_copy expects a scalar or vector index");
    axis = at::maybe_wrap_dim(axis, input.dim());
    const auto dst = input.dim() ? input : input.unsqueeze(0);
    const auto src = source.dim() ? source : source.unsqueeze(0);
    TORCH_CHECK(dst.dim() == src.dim() && src.size(axis) == indices.numel(), "index_copy shape mismatch");
    at::DimVector shape(dst.dim(), 1);
    shape[axis] = indices.numel();
    for (const auto d : c10::irange(dst.dim())) {
        TORCH_CHECK(d == axis || dst.size(d) == src.size(d), "index_copy slice shape mismatch");
    }
    auto result = scatter<IndexOperation::Scatter>(dst, axis, indices.reshape(shape).expand(src.sizes()), src);
    return input.dim() ? result : result.squeeze(0);
}

at::Tensor index_fill_tensor(const at::Tensor& input, int64_t axis, const at::Tensor& indices,
                            const at::Tensor& value) {
    TORCH_CHECK(value.dim() == 0, "index_fill expects a scalar value");
    check_indices(indices);
    TORCH_CHECK(indices.dim() <= 1, "index_fill expects a scalar or vector index");
    axis = at::maybe_wrap_dim(axis, input.dim());
    const auto dst = input.dim() ? input : input.unsqueeze(0);
    const auto normalized = at::where(at::lt(indices, 0), indices.add(dst.size(axis)), indices);
    auto shape = dst.sizes().vec();
    shape[axis] = indices.numel();
    const auto source = value.to(input.device(), input.scalar_type()).expand(shape);
    return index_copy_impl(input, axis, normalized, source);
}

at::Tensor index_fill_scalar(const at::Tensor& input, int64_t axis, const at::Tensor& indices,
                            const at::Scalar& value) {
    auto source = make_output({}, input.scalar_type());
    fill_scalar(source, value);
    return index_fill_tensor(input, axis, indices, source);
}

template <IndexOperation operation>
at::Tensor& scatter_inplace(at::Tensor& input, int64_t axis, const at::Tensor& indices, const at::Tensor& source) {
    at::assert_no_internal_overlap(input);
    at::assert_no_partial_overlap(input, source);
    copy_tensor(input, scatter<operation>(input, axis, indices, source));
    return input;
}

at::Tensor scatter_value(const at::Tensor& input, int64_t axis, const at::Tensor& indices, const at::Scalar& value) {
    auto source = make_output(indices.sizes(), input.scalar_type());
    fill_scalar(source, value);
    return scatter<IndexOperation::Scatter>(input, axis, indices, source);
}

at::Tensor& scatter_value_out(const at::Tensor& input, int64_t axis, const at::Tensor& indices,
                              const at::Scalar& value, at::Tensor& output) {
    check_device(output.device());
    TORCH_CHECK(output.scalar_type() == input.scalar_type(), "scatter requires matching output dtype");
    const auto result = scatter_value(input, axis, indices, value);
    at::native::resize_output(output, result.sizes());
    copy_tensor(output, result);
    return output;
}

at::Tensor embedding_backward(const at::Tensor& grad, const at::Tensor& indices, int64_t weights,
                              int64_t padding, bool scale) {
    check_operand(grad);
    TORCH_CHECK(grad.scalar_type() == at::kFloat, "Iris embedding backward requires float32");
    check_indices(indices, true);
    TORCH_CHECK(weights >= 0 && grad.dim() > 0, "Iris embedding gradient requires a weight matrix");
    const auto width = grad.size(-1);
    auto output = make_output({weights, width});
    if (!output.numel()) {
        return output;
    }
    if (!indices.numel()) {
        return zero(output);
    }
    const auto index = indices.reshape({-1});
    const auto source = grad.reshape({index.numel(), width});
    embedding_gradient(buffer(output), buffer(source), buffer(index), output.numel(),
                       index.storage_offset(), index.stride(0), source.storage_offset(),
                       source.stride(0), source.stride(1), index.numel(), width, padding,
                       value_type(index.scalar_type()), scale);
    return output;
}

at::Tensor run_where(const at::Tensor& condition, const at::Tensor& left, const at::Tensor& right,
                     const at::Tensor& destination) {
    check_device(condition.device());
    TORCH_CHECK(condition.scalar_type() == at::kBool, "where expects a boolean condition");
    const auto dtype = at::result_type(left, right);
    const auto type = value_type(dtype);
    TORCH_CHECK(left.device() == condition.device() || (left.device().is_cpu() && !left.dim()),
                "Iris where expects inputs on the same device");
    TORCH_CHECK(right.device() == condition.device() || (right.device().is_cpu() && !right.dim()),
                "Iris where expects inputs on the same device");
    const auto a = left.to(condition.device(), dtype);
    const auto b = right.to(condition.device(), dtype);
    TORCH_CHECK(!destination.defined() || (destination.device() == condition.device() &&
                destination.scalar_type() == dtype), "Iris where out options must match the result");
    auto output = at::TensorIteratorConfig().add_output(destination).add_const_input(condition)
        .add_const_input(a).add_const_input(b).check_all_same_dtype(false).declare_static_dtype(dtype).build().output();
    if (output.numel()) {
        const auto mask = condition.expand(output.sizes());
        const auto expanded_a = a.expand(output.sizes());
        const auto expanded_b = b.expand(output.sizes());
        c10::SmallVector<cl_ulong, 24> geometry;
        for (const auto d : c10::irange(output.dim())) {
            geometry.append({static_cast<cl_ulong>(output.size(d)), static_cast<cl_ulong>(output.stride(d)),
                             static_cast<cl_ulong>(mask.stride(d)), static_cast<cl_ulong>(expanded_a.stride(d)),
                             static_cast<cl_ulong>(expanded_b.stride(d))});
        }
        tensor_where(buffer(output), buffer(condition), buffer(a), buffer(b), output.numel(), output.storage_offset(),
                     condition.storage_offset(), a.storage_offset(), b.storage_offset(), geometry, type);
    }
    return output;
}

at::Tensor where_tensor(const at::Tensor& condition, const at::Tensor& left, const at::Tensor& right) {
    return run_where(condition, left, right, {});
}

at::Tensor& where_out(const at::Tensor& condition, const at::Tensor& left, const at::Tensor& right,
                      at::Tensor& output) {
    run_where(condition, left, right, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> dropout(const at::Tensor& input, double probability,
                                         std::optional<bool> train) {
    check_operand(input);
    TORCH_CHECK(probability >= 0.0 && probability <= 1.0, "Dropout probability must be between 0 and 1");
    if (!train.value_or(true)) {
        auto mask = at::empty_like(input, at::kBool);
        fill_scalar(mask, true);
        return {input, mask};
    }
    auto mask = at::empty_like(input);
    bernoulli(mask, 1.0 - probability, std::nullopt);
    auto output = multiply_tensor(input, mask);
    multiply_scalar_inplace(output, probability == 1.0 ? 0.0 : 1.0 / (1.0 - probability));
    return {output, mask.to(at::kBool)};
}

at::Tensor dropout_backward(const at::Tensor& grad, const at::Tensor& mask, double scale) {
    auto output = multiply_tensor(grad, mask.to(grad.scalar_type()));
    return multiply_scalar_inplace(output, scale);
}

at::Tensor nonzero_indices(const at::Tensor& input) {
    check_device(input.device());
    const auto rank = input.dim();
    if (!input.numel()) {
        return make_output({rank, 0}, at::kLong).transpose(0, 1);
    }
    auto candidates = make_output({input.numel(), rank}, at::kLong);
    auto count = make_output({}, at::kLong);
    c10::SmallVector<cl_ulong, 16> geometry;
    for (const auto dim : c10::irange(rank)) {
        geometry.append({static_cast<cl_ulong>(input.size(dim)),
                         static_cast<cl_ulong>(input.stride(dim))});
    }
    nonzero_coordinates(buffer(candidates), buffer(count), buffer(input), input.numel(),
                        input.storage_offset(), geometry, value_type(input.scalar_type()));
    const auto selected = read_scalar<int64_t>(count);
    auto result = make_output({rank, selected}, at::kLong).transpose(0, 1);
    copy_tensor(result, candidates.narrow(0, 0, selected));
    return result;
}

struct AdvancedIndex {
    at::DimVector output_shape;
    c10::SmallVector<cl_ulong, 16> geometry;
    at::Tensor offsets;
    cl_ulong trailing = 1;
};

AdvancedIndex make_advanced_index(const at::Tensor& input,
                                 const c10::List<std::optional<at::Tensor>>& requested) {
    std::vector<at::Tensor> indices;
    for (const auto i : c10::irange(requested.size())) {
        const auto item = requested.get(i);
        if (!item || !item->defined()) {
            indices.emplace_back();
        } else if (item->scalar_type() == at::kBool || item->scalar_type() == at::kByte) {
            TORCH_CHECK_INDEX(item->dim(), "Iris advanced indexing requires a non-scalar mask");
            for (const auto d : c10::irange(item->dim())) {
                const auto axis = indices.size() + d;
                TORCH_CHECK_INDEX(axis < input.dim() && item->size(d) == input.size(axis),
                                  "Boolean mask shape does not match the indexed tensor");
            }
            const auto positions = item->device().is_cpu() ? item->nonzero().to(input.device()) : nonzero_indices(*item);
            for (const auto d : c10::irange(item->dim())) {
                indices.emplace_back(positions.select(1, d));
            }
        } else {
            auto index = item->device().is_cpu() ? item->to(input.device()) : *item;
            check_indices(index, true);
            indices.emplace_back(std::move(index));
        }
    }
    TORCH_CHECK_INDEX(indices.size() <= input.dim(), "Too many indices for Iris tensor");
    indices.resize(input.dim());
    const auto axis_is_indexed = [&](size_t axis) { return indices[axis].defined(); };

    at::DimVector index_shape;
    c10::SmallVector<int64_t, 8> axes;
    for (const auto axis : c10::irange(input.dim())) {
        if (axis_is_indexed(axis)) {
            axes.push_back(axis);
            index_shape = at::infer_size_dimvector(index_shape, indices[axis].sizes());
        }
    }
    const bool nonadjacent = !axes.empty() && axes.back() - axes.front() + 1 != axes.size();
    const auto first = axes.empty() || nonadjacent ? 0 : axes.front();
    AdvancedIndex result;
    const auto append_axis = [&](int64_t axis) {
        result.output_shape.push_back(input.size(axis));
        result.geometry.append({static_cast<cl_ulong>(input.size(axis)),
                                static_cast<cl_ulong>(input.stride(axis))});
    };
    for (const auto d : c10::irange(first)) {
        append_axis(d);
    }
    for (const auto size : index_shape) {
        result.output_shape.push_back(size);
        result.geometry.append({static_cast<cl_ulong>(size), 0});
    }
    for (const auto d : c10::irange(first, input.dim())) {
        if (!axis_is_indexed(d)) {
            append_axis(d);
            result.trailing *= input.size(d);
        }
    }
    result.offsets = make_output(index_shape, at::kLong);
    if (result.offsets.numel()) {
        zero(result.offsets);
        if (!axes.empty()) {
            auto invalid = make_output({}, at::kInt);
            zero(invalid);
            for (const auto axis : axes) {
                const auto index = indices[axis].expand(index_shape);
                c10::SmallVector<cl_ulong, 16> geometry;
                for (const auto d : c10::irange(index.dim())) {
                    geometry.append({static_cast<cl_ulong>(index.size(d)),
                                    static_cast<cl_ulong>(index.stride(d))});
                }
                advanced_offsets(buffer(result.offsets), buffer(index), buffer(invalid),
                                 index.numel(), index.storage_offset(), geometry, input.size(axis),
                                 input.stride(axis), value_type(index.scalar_type()));
            }
            TORCH_CHECK_INDEX(!read_scalar<int32_t>(invalid), "Iris advanced index is out of bounds");
        }
    }
    return result;
}

at::Tensor advanced_read(const at::Tensor& input, const c10::List<std::optional<at::Tensor>>& indices) {
    check_device(input.device());
    const auto index = make_advanced_index(input, indices);
    auto output = make_output(index.output_shape, input.scalar_type());
    if (output.numel()) {
        advanced_copy(buffer(input), buffer(output), buffer(index.offsets), output.numel(),
                      input.storage_offset(), index.trailing, index.offsets.numel(), index.geometry,
                      value_type(input.scalar_type()), false, false);
    }
    return output;
}

at::Tensor& advanced_put(at::Tensor& input, const c10::List<std::optional<at::Tensor>>& indices,
                         const at::Tensor& values, bool accumulate, bool) {
    check_device(input.device());
    TORCH_CHECK(values.scalar_type() == input.scalar_type(),
                "Iris index_put_ requires matching source and destination dtype");
    at::assert_no_internal_overlap(input);
    const auto index = make_advanced_index(input, indices);
    const auto expanded = values.to(input.device()).expand(index.output_shape);
    if (expanded.numel()) {
        auto linear = make_output({expanded.numel()}, expanded.scalar_type());
        // Copy through a contiguous view to isolate values which alias the destination.
        copy_tensor(linear.view(index.output_shape), expanded);
        advanced_copy(buffer(input), buffer(linear), buffer(index.offsets), expanded.numel(),
                      input.storage_offset(), index.trailing, index.offsets.numel(), index.geometry,
                      value_type(input.scalar_type()), true, accumulate);
    }
    return input;
}

at::Tensor take_impl(const at::Tensor& input, const at::Tensor& index) {
    check_indices(index);
    return advanced_read(input.reshape({-1}), {std::optional<at::Tensor>(index)});
}

at::Tensor put_impl(const at::Tensor& input, const at::Tensor& index, const at::Tensor& source, bool accumulate) {
    check_indices(index);
    TORCH_CHECK(index.numel() == source.numel() && source.device() == input.device(),
                "put expects matching index and source element counts and devices");
    auto linear = input.reshape({-1}).clone();
    advanced_put(linear, {std::optional<at::Tensor>(index.reshape({-1}))},
                 source.reshape({-1}), accumulate, false);
    auto result = at::empty_like(input);
    copy_tensor(result, linear.reshape(input.sizes()));
    return result;
}

template <bool all>
at::Tensor logical_dims(const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim) {
    check_device(input.device());
    const auto type = value_type(input.scalar_type());
    c10::SmallVector<bool, 8> reduced(input.dim(), !dims.has_value());
    if (dims) {
        bool scalar_dim = false;
        for (auto dim : *dims) {
            dim = at::maybe_wrap_dim(dim, input.dim());
            TORCH_CHECK(!(input.dim() ? reduced[dim] : scalar_dim), "dim ", dim, " appears multiple times");
            if (input.dim()) {
                reduced[dim] = true;
            } else {
                scalar_dim = true;
            }
        }
    }
    at::DimVector sizes;
    c10::SmallVector<cl_ulong, 16> geometry;
    for (const auto d : c10::irange(input.dim())) {
        if (!reduced[d]) {
            sizes.push_back(input.size(d));
            geometry.append({static_cast<cl_ulong>(input.size(d)), static_cast<cl_ulong>(input.stride(d))});
        } else if (keepdim) {
            sizes.push_back(1);
        }
    }
    const auto outer_rank = geometry.size() / 2;
    cl_ulong length = 1;
    for (const auto d : c10::irange(input.dim())) {
        if (reduced[d]) {
            length *= input.size(d);
            geometry.append({static_cast<cl_ulong>(input.size(d)), static_cast<cl_ulong>(input.stride(d))});
        }
    }
    auto output = make_output(sizes, type == ValueType::Byte ? at::kByte : at::kBool);
    if (output.numel()) {
        logical_reduce(buffer(output), buffer(input), output.numel(), length, input.storage_offset(),
                       outer_rank, geometry, type, all);
    }
    return output;
}

template <bool all>
at::Tensor logical_dim(const at::Tensor& input, int64_t dim, bool keepdim) {
    return logical_dims<all>(input, at::IntArrayRef(dim), keepdim);
}

template <bool negative>
at::Tensor infinity_predicate(const at::Tensor& input) {
    const auto infinity = std::numeric_limits<float>::infinity();
    return compare_scalar<Elementwise::Eq>(input, negative ? -infinity : infinity);
}

template <bool negative>
at::Tensor& infinity_predicate_out(const at::Tensor& input, at::Tensor& output) {
    check_device(output.device());
    TORCH_CHECK(output.scalar_type() == at::kBool, "Infinity predicate out must have bool dtype");
    auto mask = infinity_predicate<negative>(input);
    at::native::resize_output(output, mask.sizes());
    copy_tensor(output, mask);
    return output;
}

template <bool upper>
at::Tensor triangular_tensor(const at::Tensor& input, int64_t diagonal) {
    check_device(input.device());
    TORCH_CHECK(input.dim() >= 2, "Iris triangular operation requires at least 2 dimensions");
    auto output = make_output(input.sizes(), input.scalar_type());
    if (output.numel()) {
        c10::SmallVector<cl_ulong, 16> geometry;
        for (const auto d : c10::irange(input.dim())) {
            geometry.append({static_cast<cl_ulong>(input.size(d)), static_cast<cl_ulong>(input.stride(d))});
        }
        triangular(buffer(output), buffer(input), output.numel(), input.storage_offset(), input.size(-2),
                   input.size(-1), diagonal, geometry, value_type(input.scalar_type()), upper);
    }
    return output;
}

template <bool upper>
at::Tensor& triangular_out(const at::Tensor& input, int64_t diagonal, at::Tensor& output) {
    check_device(output.device());
    TORCH_CHECK(output.scalar_type() == input.scalar_type(), "Iris triangular out dtype must match input");
    auto result = triangular_tensor<upper>(input, diagonal);
    at::native::resize_output(output, result.sizes());
    copy_tensor(output, result);
    return output;
}

struct TrilIndicesShape {
    int64_t count = 0;
    uint64_t first_row = 0;
    uint64_t first_width = 0;
    uint64_t growing_rows = 0;
    uint64_t growing_count = 0;
};

TrilIndicesShape tril_indices_shape(int64_t rows, int64_t columns, int64_t offset, at::ScalarType dtype) {
    TORCH_CHECK(rows >= 0 && columns >= 0, "row and col must be non-negative");
    TORCH_CHECK(dtype == at::kInt || dtype == at::kLong, "Iris tril_indices requires int32 or int64");
    if (!rows || !columns || offset <= -rows) return {};

    TrilIndicesShape shape;
    const auto width = static_cast<uint64_t>(columns);
    shape.first_row = offset < 0 ? -offset : 0;
    shape.first_width = offset < 0 ? 1 : std::min(offset, columns - 1) + 1;
    const auto active_rows = static_cast<uint64_t>(rows) - shape.first_row;
    shape.growing_rows = std::min(active_rows, width - shape.first_width);
    const auto n = shape.growing_rows;
    uint64_t initial, growth, full, total;
    // The growing section has widths first_width, first_width + 1, and so on.
    TORCH_CHECK(!c10::mul_overflows(n, shape.first_width, &initial) &&
                !c10::mul_overflows(n, n ? n - 1 : 0, &growth) &&
                !c10::add_overflows(initial, growth / 2, &shape.growing_count) &&
                !c10::mul_overflows(active_rows - n, width, &full) &&
                !c10::add_overflows(shape.growing_count, full, &total) &&
                total <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / 2),
                "Iris tril_indices output size overflow");
    shape.count = static_cast<int64_t>(total);
    return shape;
}

void fill_tril_indices(at::Tensor& output, const TrilIndicesShape& shape, int64_t columns) {
    if (shape.count) {
        at::assert_no_internal_overlap(output);
        const auto row_stride = output.stride(0);
        const auto column_stride = output.stride(1);
        TORCH_CHECK(!column_stride || row_stride % column_stride || row_stride / column_stride >= shape.count,
                    "Iris tril_indices requires a non-overlapping out tensor");
        iris::tril_indices(buffer(output), shape.count, output.storage_offset(), row_stride,
                           column_stride, shape.first_row, shape.first_width, shape.growing_rows,
                           shape.growing_count, columns, output.scalar_type() == at::kInt);
    }
}

at::Tensor tril_indices_factory(int64_t rows, int64_t columns, int64_t offset,
                               std::optional<at::ScalarType> dtype, std::optional<c10::Layout> layout,
                               std::optional<c10::Device> device, std::optional<bool> pin_memory) {
    const auto type = check_options(dtype.value_or(at::kLong), layout, device, pin_memory);
    const auto shape = tril_indices_shape(rows, columns, offset, type);
    initialize();
    auto output = make_output({2, shape.count}, type);
    fill_tril_indices(output, shape, columns);
    return output;
}

at::Tensor& tril_indices_out(int64_t rows, int64_t columns, int64_t offset, at::Tensor& output) {
    check_device(output.device());
    const auto shape = tril_indices_shape(rows, columns, offset, output.scalar_type());
    at::native::resize_output(output, {2, shape.count});
    if (output.is_neg()) {
        auto result = make_output({2, shape.count}, output.scalar_type());
        fill_tril_indices(result, shape, columns);
        copy_tensor(output, result);
    } else {
        fill_tril_indices(output, shape, columns);
    }
    return output;
}

at::Tensor masked_fill_scalar(const at::Tensor& input, const at::Tensor& mask, const at::Scalar& value) {
    auto scalar = make_output({}, input.scalar_type());
    fill_scalar(scalar, value);
    return where_tensor(mask, scalar, input);
}

at::Tensor& masked_fill_scalar_inplace(at::Tensor& input, const at::Tensor& mask, const at::Scalar& value) {
    at::assert_no_internal_overlap(input);
    auto result = masked_fill_scalar(input, mask, value);
    TORCH_CHECK(result.sizes() == input.sizes(), "Iris masked_fill_ cannot change the destination shape");
    copy_tensor(input, result);
    return input;
}

at::Tensor& write_result(at::Tensor& output, const at::Tensor& result) {
    check_device(output.device());
    TORCH_CHECK(c10::canCast(result.scalar_type(), output.scalar_type()),
                "Iris out dtype cannot represent the operation's result type");
    at::assert_no_internal_overlap(output);
    if (at::native::resize_output(output, result.sizes()) && output.strides() != result.strides()) {
        set_storage(output, output.storage(), output.storage_offset(), result.sizes(), result.strides());
    }
    copy_tensor(output, result);
    return output;
}

at::Tensor& write_inplace(at::Tensor& output, const at::Tensor& result) {
    TORCH_CHECK(output.sizes() == result.sizes(), "Iris in-place operations cannot change the destination shape");
    return write_result(output, result);
}

template <auto function>
at::Tensor& binary_out(const at::Tensor& left, const at::Tensor& right, at::Tensor& out) {
    return write_result(out, function(left, right));
}

template <auto function>
at::Tensor& binary_inplace(at::Tensor& left, const at::Tensor& right) {
    at::assert_no_partial_overlap(left, right);
    return write_inplace(left, function(left, right));
}

at::Tensor& divide_mode_out(const at::Tensor& left, const at::Tensor& right,
                            std::optional<c10::string_view> mode, at::Tensor& out) {
    return write_result(out, divide_mode(left, right, mode));
}

at::Tensor& divide_mode_inplace(at::Tensor& left, const at::Tensor& right,
                                std::optional<c10::string_view> mode) {
    at::assert_no_partial_overlap(left, right);
    return write_inplace(left, divide_mode(left, right, mode));
}

template <auto function, typename Value>
at::Tensor& indexed_out(const at::Tensor& input, int64_t dim, const at::Tensor& index, const Value& value,
                        at::Tensor& out) {
    TORCH_CHECK(out.scalar_type() == input.scalar_type(), "indexing output must match input dtype");
    return write_result(out, function(input, dim, index, value));
}

template <auto function, typename Value>
at::Tensor& indexed_inplace(at::Tensor& input, int64_t dim, const at::Tensor& index, const Value& value) {
    return write_inplace(input, function(input, dim, index, value));
}

at::Tensor& scatter_reduce_out(const at::Tensor& input, int64_t dim, const at::Tensor& index,
                               const at::Tensor& source, c10::string_view reduce, bool include_self,
                               at::Tensor& out) {
    TORCH_CHECK(out.scalar_type() == input.scalar_type(), "scatter_reduce output must match input dtype");
    return write_result(out, scatter_reduce_impl(input, dim, index, source, reduce, include_self));
}

at::Tensor& scatter_reduce_inplace(at::Tensor& input, int64_t dim, const at::Tensor& index,
                                   const at::Tensor& source, c10::string_view reduce, bool include_self) {
    at::assert_no_partial_overlap(input, source);
    return write_inplace(input, scatter_reduce_impl(input, dim, index, source, reduce, include_self));
}

at::Tensor& put_out(const at::Tensor& input, const at::Tensor& index, const at::Tensor& source,
                    bool accumulate, at::Tensor& out) {
    TORCH_CHECK(out.scalar_type() == input.scalar_type(), "put output must match input dtype");
    return write_result(out, put_impl(input, index, source, accumulate));
}

at::Tensor& put_inplace(at::Tensor& input, const at::Tensor& index, const at::Tensor& source, bool accumulate) {
    at::assert_no_partial_overlap(input, source);
    return write_inplace(input, put_impl(input, index, source, accumulate));
}

template <Extrema op>
at::Tensor& extrema_dims_out(const at::Tensor& input, at::IntArrayRef dims, bool keepdim, at::Tensor& out) {
    TORCH_CHECK(out.scalar_type() == input.scalar_type(), "reduction output must match input dtype");
    return write_result(out, extrema_dims<op>(input, dims, keepdim));
}

std::tuple<at::Tensor&, at::Tensor&> aminmax_out(const at::Tensor& input, std::optional<int64_t> dim,
                                              bool keepdim, at::Tensor& min, at::Tensor& max) {
    TORCH_CHECK(min.scalar_type() == input.scalar_type() && max.scalar_type() == input.scalar_type(),
                "aminmax outputs must match input dtype");
    at::assert_no_overlap(min, max);
    auto [minimum, maximum] = aminmax_impl(input, dim, keepdim);
    return {write_result(min, minimum), write_result(max, maximum)};
}

template <auto function>
at::Tensor& binary_alpha_out(
    const at::Tensor& left, const at::Tensor& right, const at::Scalar& alpha, at::Tensor& out) {
    return write_result(out, function(left, right, alpha));
}

at::Tensor& subtract_inplace(at::Tensor& left, const at::Tensor& right, const at::Scalar& alpha) {
    at::assert_no_partial_overlap(left, right);
    return write_inplace(left, subtract_tensor(left, right, alpha));
}

at::Tensor& pow_scalar_out(const at::Tensor& input, const at::Scalar& exponent, at::Tensor& out) {
    return write_result(out, pointwise_scalar<Elementwise::PowScalar>(input, exponent));
}

at::Tensor& pow_scalar_base_out(const at::Scalar& base, const at::Tensor& exponent, at::Tensor& out) {
    return write_result(out, pow_scalar_base(base, exponent));
}

at::Tensor& clamp_out(const at::Tensor& input, const std::optional<at::Scalar>& lower,
                     const std::optional<at::Scalar>& upper, at::Tensor& out) {
    return write_result(out, clamp_tensor(input, lower, upper));
}

at::Tensor& clamp_inplace(at::Tensor& input, const std::optional<at::Scalar>& lower,
                         const std::optional<at::Scalar>& upper) {
    return write_inplace(input, clamp_tensor(input, lower, upper));
}

at::Tensor rsub_tensor(const at::Tensor& left, const at::Tensor& right, const at::Scalar& alpha) {
    return subtract_tensor(right, left, alpha);
}

at::Tensor clamp_min_scalar(const at::Tensor& input, const at::Scalar& lower) {
    return clamp_tensor(input, lower, std::nullopt);
}

at::Tensor clamp_max_scalar(const at::Tensor& input, const at::Scalar& upper) {
    return clamp_tensor(input, std::nullopt, upper);
}

at::Tensor& clamp_min_scalar_out(const at::Tensor& input, const at::Scalar& lower, at::Tensor& out) {
    return clamp_out(input, lower, std::nullopt, out);
}

at::Tensor& clamp_max_scalar_out(const at::Tensor& input, const at::Scalar& upper, at::Tensor& out) {
    return clamp_out(input, std::nullopt, upper, out);
}

at::Tensor& clamp_min_scalar_inplace(at::Tensor& input, const at::Scalar& lower) {
    return clamp_inplace(input, lower, std::nullopt);
}

at::Tensor& clamp_max_scalar_inplace(at::Tensor& input, const at::Scalar& upper) {
    return clamp_inplace(input, std::nullopt, upper);
}

at::Tensor masked_fill_tensor(const at::Tensor& input, const at::Tensor& mask, const at::Tensor& value) {
    TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor");
    return masked_fill_scalar(input, mask, value.item());
}

at::Tensor& masked_fill_tensor_inplace(at::Tensor& input, const at::Tensor& mask, const at::Tensor& value) {
    TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor");
    return masked_fill_scalar_inplace(input, mask, value.item());
}

at::Tensor& gelu_out(const at::Tensor& input, c10::string_view approximation, at::Tensor& out) {
    return write_result(out, iris::gelu(input, approximation));
}

at::Tensor& gelu_inplace(at::Tensor& input, c10::string_view approximation) {
    check_operand(input);
    at::assert_no_internal_overlap(input);
    run_strided(gelu_op(approximation, false), input, input, input);
    return input;
}

at::Tensor& gather_out(const at::Tensor& input, int64_t axis, const at::Tensor& indices,
                       bool sparse_grad, at::Tensor& out) {
    at::assert_no_overlap(out, input);
    at::assert_no_overlap(out, indices);
    return write_result(out, iris::gather(input, axis, indices, sparse_grad));
}

at::Tensor& index_select_out(const at::Tensor& input, int64_t axis, const at::Tensor& indices,
                             at::Tensor& out) {
    at::assert_no_overlap(out, input);
    at::assert_no_overlap(out, indices);
    return write_result(out, iris::index_select(input, axis, indices));
}

void check_mm(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    check_device(right.device());
    TORCH_CHECK(left.scalar_type() == right.scalar_type(), "Iris matmul input dtypes must match");
    TORCH_CHECK(left.dim() == 2 && right.dim() == 2, "Iris matmul requires two 2-D tensors");
    TORCH_CHECK(left.size(1) == right.size(0), "Iris matmul requires matching inner dimensions");
}

MatrixView matrix_view(const at::Tensor& tensor);

void run_typed_mm(const at::Tensor& out, const at::Tensor& left, const at::Tensor& right,
                  int64_t batches, int64_t rows, int64_t columns, int64_t inner) {
    const auto dtype = left.scalar_type();
    if (dtype == at::kComplexFloat) {
        const auto [ar, ai] = complex_parts(left);
        const auto [br, bi] = complex_parts(right);
        auto mat = [&](const at::Tensor& a, const at::Tensor& b) {
            auto result = make_output(out.sizes());
            matrix_multiply(buffer(result), matrix_view(a), matrix_view(b), batches, rows, columns, inner, false);
            return result;
        };
        auto real = mat(ar, br).sub(mat(ai, bi));
        auto imaginary = mat(ar, bi).add(mat(ai, br));
        copy_tensor(out, complex_join(real, imaginary));
    } else if (dtype == at::kBFloat16) {
        auto a = left.to(at::kFloat);
        auto b = right.to(at::kFloat);
        auto fp32 = make_output(out.sizes());
        matrix_multiply(buffer(fp32), matrix_view(a), matrix_view(b), batches, rows, columns, inner, false);
        copy_tensor(out, fp32);
    } else if (c10::isIntegralType(dtype, false)) {
        integer_matmul(buffer(out), matrix_view(left), matrix_view(right),
                       batches, rows, columns, inner, value_type(dtype));
    } else {
        check_tensor(left);
        matrix_multiply(buffer(out), matrix_view(left), matrix_view(right),
                        batches, rows, columns, inner, dtype == at::kHalf);
    }
}

MatrixView matrix_view(const at::Tensor& tensor) {
    const auto rank = tensor.dim();
    return {buffer(tensor), static_cast<cl_ulong>(tensor.storage_offset()),
            rank == 3 ? static_cast<cl_ulong>(tensor.stride(0)) : 0,
            static_cast<cl_ulong>(tensor.stride(-2)), static_cast<cl_ulong>(tensor.stride(-1))};
}

at::Tensor matmul_2d(const at::Tensor& left, const at::Tensor& right) {
    check_mm(left, right);
    const auto rows = left.size(0);
    const auto inner = left.size(1);
    const auto columns = right.size(1);
    auto output = make_output({rows, columns}, left.scalar_type());
    run_typed_mm(output, left, right, 1, rows, columns, inner);
    return output;
}

at::Tensor batched_mm(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    check_device(right.device());
    TORCH_CHECK(left.scalar_type() == right.scalar_type(), "Iris bmm input dtypes must match");
    TORCH_CHECK(left.dim() == 3 && right.dim() == 3, "Iris bmm requires two 3-D tensors");
    const auto batches = left.size(0);
    const auto rows = left.size(1);
    const auto inner = left.size(2);
    const auto columns = right.size(2);
    TORCH_CHECK(batches == right.size(0) && inner == right.size(1),
                "Iris bmm requires matching batch and inner dimensions");
    auto output = make_output({batches, rows, columns}, left.scalar_type());
    run_typed_mm(output, left, right, batches, rows, columns, inner);
    return output;
}

at::Tensor iris_dot(const at::Tensor& left, const at::Tensor& right) {
    TORCH_CHECK(left.dim() == 1 && right.dim() == 1, "Iris dot expects two vectors");
    return matmul_2d(left.unsqueeze(0), right.unsqueeze(1)).view(at::IntArrayRef{});
}

at::Tensor iris_mv(const at::Tensor& matrix, const at::Tensor& vector) {
    TORCH_CHECK(matrix.dim() == 2 && vector.dim() == 1, "Iris mv expects a matrix and a vector");
    return matmul_2d(matrix, vector.unsqueeze(1)).squeeze(1);
}

at::Tensor matmul(const at::Tensor& left, const at::Tensor& right) {
    if (left.dim() == 2 && right.dim() == 2) return matmul_2d(left, right);
    return at::native::matmul(left, right);
}

std::tuple<at::Tensor, at::Tensor> matmul_backward(
    const at::Tensor& grad, const at::Tensor& left, const at::Tensor& right,
    std::array<bool, 2> mask) {
    const bool left_vector = left.dim() == 1;
    const bool right_vector = right.dim() == 1;
    const auto left_matrix = left_vector ? left.unsqueeze(0) : left;
    const auto right_matrix = right_vector ? right.unsqueeze(-1) : right;
    auto grad_matrix = right_vector ? grad.unsqueeze(-1) : grad;
    if (left_vector) grad_matrix = grad_matrix.unsqueeze(-2);
    at::Tensor grad_left;
    at::Tensor grad_right;
    if (mask[0]) {
        grad_left = at::sum_to(at::native::matmul(grad_matrix, right_matrix.transpose(-2, -1).conj()), left_matrix.sizes());
        if (left_vector) grad_left = grad_left.squeeze(0);
    }
    if (mask[1]) {
        grad_right = at::sum_to(at::native::matmul(left_matrix.transpose(-2, -1).conj(), grad_matrix), right_matrix.sizes());
        if (right_vector) grad_right = grad_right.squeeze(-1);
    }
    return {grad_left, grad_right};
}

at::Tensor addmm(
    const at::Tensor& bias, const at::Tensor& left, const at::Tensor& right,
    const at::Scalar& beta, const at::Scalar& alpha) {
    check_operand(bias);
    check_mm(left, right);
    TORCH_CHECK(bias.scalar_type() == left.scalar_type(), "Iris addmm input dtypes must match");
    const at::DimVector sizes{left.size(0), right.size(1)};
    TORCH_CHECK(bias.dim() <= 2, "Iris addmm bias must broadcast into a 2-D result");
    const auto expanded = at::inferExpandGeometry_dimvector(bias.sizes(), bias.strides(), sizes);
    const auto bias_scale = beta.toFloat();
    const auto product_scale = alpha.toFloat();
    auto output = make_output(sizes, left.scalar_type());
    if (!left.size(1) || product_scale == 0.0f) {
        if (bias_scale == 0.0f) {
            zero(output);
        } else {
            run_strided(Elementwise::Scale, output, bias, bias, bias_scale);
        }
        return output;
    }
    const MatrixView bias_view{buffer(bias), static_cast<cl_ulong>(bias.storage_offset()), 0,
                               static_cast<cl_ulong>(expanded.strides[0]),
                               static_cast<cl_ulong>(expanded.strides[1])};
    matrix_multiply(buffer(output), matrix_view(left), matrix_view(right),
                    1, sizes[0], sizes[1], left.size(1), left.scalar_type() == at::kHalf,
                    bias_view, product_scale, bias_scale);
    return output;
}

at::Tensor iris_addmv(const at::Tensor& input, const at::Tensor& matrix, const at::Tensor& vector,
                 const at::Scalar& beta, const at::Scalar& alpha) {
    TORCH_CHECK(input.dim() <= 1 && matrix.dim() == 2 && vector.dim() == 1,
                "Iris addmv expects a scalar or vector bias, matrix and vector");
    const auto bias = input.dim() ? input.unsqueeze(-1) : input;
    return bias.addmm(matrix, vector.unsqueeze(-1), beta, alpha).squeeze(-1);
}

at::Tensor reduce_dims(
    const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
    std::optional<at::ScalarType> dtype, bool average) {
    check_device(input.device());
    const auto input_type = input.scalar_type();
    if (input_type == at::kComplexFloat || dtype == at::kComplexFloat) {
        const auto requested = dtype.value_or(at::kComplexFloat);
        if (requested != at::kComplexFloat) return reduce_dims(input.to(requested), dims, keepdim, requested, average);
        const auto [real, imaginary] = complex_parts(input.to(at::kComplexFloat));
        return complex_join(reduce_dims(real, dims, keepdim, at::kFloat, average),
                            reduce_dims(imaginary, dims, keepdim, at::kFloat, average));
    }
    if (input_type == at::kBFloat16 || dtype == at::kBFloat16) {
        const auto requested = dtype.value_or(input_type);
        TORCH_CHECK(!average || c10::isFloatingType(requested), "mean requires a floating-point output dtype");
        const auto accumulator = requested == at::kBFloat16 ? at::kFloat : requested;
        return reduce_dims(input.to(accumulator), dims, keepdim, accumulator, average).to(requested);
    }
    const bool input_float = supported_float(input_type);
    const auto output_type = dtype.value_or(average || input_float ? input_type : at::kLong);
    const bool output_float = supported_float(output_type);
    const auto output_storage_type = value_type(output_type);
    TORCH_CHECK(!average || output_float, "mean requires a floating-point output dtype");
    if (input_float != output_float || (output_type == at::kBool && input_type != at::kBool)) {
        return reduce_dims(input.to(output_type), dims, keepdim, output_type, average);
    }
    const auto rank = input.dim();
    c10::SmallVector<bool, 8> reduce(rank, !dims.has_value() || dims->empty());
    if (dims.has_value()) {
        bool scalar_dim_seen = false;
        for (auto dim : *dims) {
            dim = at::maybe_wrap_dim(dim, rank);
            TORCH_CHECK(!(rank ? reduce[dim] : scalar_dim_seen),
                        "dim ", dim, " appears multiple times in the list of dims");
            if (rank) {
                reduce[dim] = true;
            }
            scalar_dim_seen = true;
        }
    }
    at::DimVector sizes;
    c10::SmallVector<cl_ulong, 16> geometry;
    for (const auto dim : c10::irange(rank)) {
        if (!reduce[dim]) {
            sizes.push_back(input.size(dim));
            geometry.push_back(input.size(dim));
            geometry.push_back(input.stride(dim));
        } else if (keepdim) {
            sizes.push_back(1);
        }
    }
    const auto outer_rank = geometry.size() / 2;
    cl_ulong reduce_length = 1;
    for (const auto dim : c10::irange(rank)) {
        if (reduce[dim]) {
            reduce_length *= input.size(dim);
            geometry.push_back(input.size(dim));
            geometry.push_back(input.stride(dim));
        }
    }
    auto output = make_output(sizes, output_type);
    if (!output.numel()) {
        return output;
    }
    if (!reduce_length) {
        fill_scalar(output, average ? std::numeric_limits<float>::quiet_NaN() : 0.0f);
    } else if (!output_float) {
        const auto input_storage_type = value_type(input_type);
        if (outer_rank == 0 && reduce_length >= 4096) {
            const auto chunks = std::min<cl_ulong>(128, (reduce_length + 511) / 512);
            auto partial = make_output({static_cast<int64_t>(chunks)}, at::kLong);
            iris::integer_sum(buffer(partial), buffer(input), 1, reduce_length, input.storage_offset(),
                              0, geometry, input_storage_type, ValueType::Long, input.is_contiguous(), chunks);
            iris::integer_sum(buffer(output), buffer(partial), 1, chunks, 0, 0, {},
                              ValueType::Long, output_storage_type, true);
        } else {
            iris::integer_sum(buffer(output), buffer(input), output.numel(), reduce_length, input.storage_offset(),
                              outer_rank, geometry, input_storage_type, output_storage_type,
                              outer_rank == 0 && input.is_contiguous());
        }
    } else if (input.scalar_type() == at::kFloat && output_type == at::kFloat && input.is_contiguous()) {
        if (outer_rank == 0 && reduce_length >= 1024) {
            const cl_ulong num_groups = std::min<cl_ulong>(128, (reduce_length + 511) / 512);
            auto partial = make_output({static_cast<int64_t>(num_groups)});
            iris::sum_all_contiguous(buffer(output), buffer(partial), buffer(input), reduce_length,
                                     input.storage_offset(), num_groups, average);
        } else if (rank == 2 && reduce[0] && !reduce[1] && input.size(1) >= 64 && input.size(0) >= 16) {
            iris::sum_2d_dim0(buffer(output), buffer(input), input.storage_offset(),
                              static_cast<cl_int>(input.size(0)), static_cast<cl_int>(input.size(1)), average);
        } else if (rank == 4 && reduce[0] && !reduce[1] && reduce[2] && reduce[3] &&
                   input.size(0) * input.size(2) * input.size(3) >= 64) {
            iris::sum_4d_nchw_c(buffer(output), buffer(input), input.storage_offset(),
                                static_cast<cl_int>(input.size(0)), static_cast<cl_int>(input.size(1)),
                                static_cast<cl_int>(input.size(2) * input.size(3)), average);
        } else {
            iris::sum(buffer(output), buffer(input), output.numel(), reduce_length, input.storage_offset(),
                      outer_rank, geometry, false, false, average);
        }
    } else {
        iris::sum(buffer(output), buffer(input), output.numel(), reduce_length, input.storage_offset(),
                  outer_rank, geometry, input.scalar_type() == at::kHalf, output_type == at::kHalf, average);
    }
    return output;
}

at::Tensor sum_dims(
    const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
    std::optional<at::ScalarType> dtype) {
    return reduce_dims(input, dims, keepdim, dtype, false);
}

at::Tensor sum_all(const at::Tensor& input, std::optional<at::ScalarType> dtype) {
    return sum_dims(input, std::nullopt, false, dtype);
}

at::Tensor nansum_impl(const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
                  std::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!input.is_floating_point() || !dtype || c10::isFloatingType(*dtype),
                "nansum requires a floating-point result for floating-point inputs");
    if (input.scalar_type() == at::kBFloat16) {
        const auto result_type = dtype.value_or(at::kBFloat16);
        return nansum_impl(input.to(at::kFloat), dims, keepdim,
                           result_type == at::kBFloat16 ? at::kFloat : result_type).to(result_type);
    }
    const auto clean = input.is_floating_point()
        ? at::where(at::isnan(input), at::Scalar(0), input) : input;
    return sum_dims(clean, dims, keepdim, dtype);
}

at::Tensor& nansum_out(const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
                      std::optional<at::ScalarType> dtype, at::Tensor& output) {
    TORCH_CHECK(!dtype || *dtype == output.scalar_type(), "nansum output dtype must match dtype argument");
    return write_result(output, nansum_impl(input, dims, keepdim, output.scalar_type()));
}

at::Tensor mean_dims(
    const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
    std::optional<at::ScalarType> dtype) {
    return reduce_dims(input, dims, keepdim, dtype, true);
}

at::Tensor mean_all(const at::Tensor& input, std::optional<at::ScalarType> dtype) {
    return mean_dims(input, std::nullopt, false, dtype);
}

template <bool logarithmic>
at::Tensor softmax(const at::Tensor& input, int64_t dim, bool half_to_float) {
    check_operand(input);
    TORCH_CHECK(!half_to_float, "Iris softmax requires matching input and output dtypes");
    dim = at::maybe_wrap_dim(dim, input.dim());
    c10::SmallVector<cl_ulong, 16> geometry;
    cl_ulong inner = 1;
    for (const auto axis : c10::irange(input.dim())) {
        if (axis != dim) {
            geometry.push_back(input.size(axis));
            geometry.push_back(input.stride(axis));
            if (axis > dim) {
                inner *= input.size(axis);
            }
        }
    }
    auto output = make_output(input.sizes(), input.scalar_type());
    const auto width = input.dim() ? input.size(dim) : 1;
    const auto stride = input.dim() ? input.stride(dim) : 0;
    if (output.numel() && supported_float(input.scalar_type()) && input.is_contiguous() &&
        input.dim() > 0 && dim == input.dim() - 1) {
        iris::softmax_lastdim_contiguous(buffer(output), buffer(input), input.storage_offset(),
                                         output.numel() / width, width, logarithmic,
                                         input.scalar_type() == at::kHalf);
    } else {
        iris::softmax(buffer(output), buffer(input), output.numel(), width, inner,
                      input.storage_offset(), stride,
                      geometry, logarithmic, input.scalar_type() == at::kHalf);
    }
    return output;
}

at::Tensor softmax_backward(
    const at::Tensor& grad, const at::Tensor& output, int64_t dim, at::ScalarType input_dtype) {
    TORCH_CHECK(supported_float(input_dtype), "Iris softmax supports float32 and float16");
    TORCH_CHECK(grad.sizes() == output.sizes(), "Iris softmax backward requires matching shapes");
    dim = at::maybe_wrap_dim(dim, output.dim());
    if (output.numel() && output.dim() > 0 && dim == output.dim() - 1 &&
        grad.is_contiguous() && output.is_contiguous() && grad.scalar_type() == output.scalar_type()) {
        auto grad_in = make_output(output.sizes(), output.scalar_type());
        const auto width = output.size(dim);
        iris::softmax_backward_lastdim(buffer(grad_in), buffer(grad), buffer(output),
                                       grad_in.storage_offset(), grad.storage_offset(), output.storage_offset(),
                                       output.numel() / width, width, false, output.scalar_type() == at::kHalf);
        return grad_in;
    }
    const auto product = multiply_tensor(grad, output);
    const auto total = sum_dims(product, at::IntArrayRef(dim), true, std::nullopt);
    const auto difference = binary(Elementwise::Add, grad, total, -1.0f);
    return multiply_tensor(difference, output);
}


at::Tensor log_softmax_backward(
    const at::Tensor& grad, const at::Tensor& output, int64_t dim, at::ScalarType input_dtype) {
    TORCH_CHECK(supported_float(input_dtype), "Iris log_softmax supports float32 and float16");
    TORCH_CHECK(grad.sizes() == output.sizes(), "Iris log_softmax backward requires matching shapes");
    dim = at::maybe_wrap_dim(dim, output.dim());
    if (output.numel() && output.dim() > 0 && dim == output.dim() - 1 &&
        grad.is_contiguous() && output.is_contiguous() && grad.scalar_type() == output.scalar_type()) {
        auto grad_in = make_output(output.sizes(), output.scalar_type());
        const auto width = output.size(dim);
        iris::softmax_backward_lastdim(buffer(grad_in), buffer(grad), buffer(output),
                                       grad_in.storage_offset(), grad.storage_offset(), output.storage_offset(),
                                       output.numel() / width, width, true, output.scalar_type() == at::kHalf);
        return grad_in;
    }
    const auto total = sum_dims(grad, at::IntArrayRef(dim), true, std::nullopt);
    const auto scaled = multiply_tensor(pointwise_unary<Elementwise::Exp>(output), total);
    return subtract_tensor(grad, scaled, 1.0);
}

void check_reduction(int64_t reduction) {
    TORCH_CHECK(reduction == at::Reduction::None || reduction == at::Reduction::Sum ||
                    reduction == at::Reduction::Mean,
                "Iris MSE loss requires reduction none, sum or mean");
}

MatrixView vector_view(const at::Tensor& input) {
    return {buffer(input), static_cast<cl_ulong>(input.storage_offset()), 0,
            input.dim() ? static_cast<cl_ulong>(input.stride(0)) : 0, 0};
}

NllLoss nll_arguments(const at::Tensor& input, const at::Tensor& target,
                      const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index) {
    check_operand(input);
    check_indices(target);
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Iris nll_loss requires float32");
    check_reduction(reduction);
    TORCH_CHECK(input.dim() == 1 || input.dim() == 2, "Iris nll_loss requires a 1-D or 2-D input");
    const cl_ulong batch = input.dim() == 1 ? 1 : input.size(0);
    const cl_ulong classes = input.size(-1);
    TORCH_CHECK((input.dim() == 1 && target.dim() == 0) ||
                (target.dim() == 1 && target.numel() == batch), "Iris nll_loss target shape mismatch");
    MatrixView weight_view{};
    if (weight && weight->defined()) {
        check_operand(*weight);
        TORCH_CHECK(weight->scalar_type() == at::kFloat, "Iris nll_loss requires float32 weights");
        TORCH_CHECK(weight->dim() == 1 && weight->numel() == classes,
                    "Iris nll_loss weight must have one entry per class");
        weight_view = vector_view(*weight);
    }
    const auto input_view = matrix_view(input.dim() == 1 ? input.unsqueeze(0) : input);
    return NllLoss{.output = nullptr, .weights = nullptr, .invalid = nullptr, .input = input_view,
        .target = vector_view(target), .weight = weight_view, .grad = {}, .total_weight = {},
        .batch = batch, .classes = classes, .ignore_index = ignore_index, .mean = reduction == at::Reduction::Mean};
}

std::tuple<at::Tensor, at::Tensor> nll_forward(
    const at::Tensor& input, const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index) {
    auto arguments = nll_arguments(input, target, weight, reduction, ignore_index);
    if (reduction != at::Reduction::None) {
        auto loss = make_output({});
        auto total = make_output({});
        if (arguments.batch) {
            iris::nll_loss_forward_fused(buffer(loss), buffer(total),
                                         arguments.input, arguments.target, arguments.weight,
                                         arguments.batch, arguments.classes, arguments.ignore_index,
                                         arguments.mean);
        } else {
            zero(total);
            fill_scalar(loss, arguments.mean ? std::numeric_limits<float>::quiet_NaN() : 0.0f);
        }
        return {loss, total};
    }
    auto losses = make_output({static_cast<int64_t>(arguments.batch)});
    auto weights = make_output(losses.sizes());
    if (arguments.batch) {
        arguments.output = buffer(losses);
        arguments.weights = buffer(weights);
        iris::nll_loss_async(arguments);
    }
    if (input.dim() == 2) {
        auto total = make_output({});
        zero(total);
        return {losses, total};
    }
    auto total = sum_all(weights, std::nullopt);
    return {losses.squeeze(0), total};
}

at::Tensor nll_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& target,
                        const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index,
                        const at::Tensor& total_weight) {
    auto arguments = nll_arguments(input, target, weight, reduction, ignore_index);
    check_operand(grad);
    check_operand(total_weight);
    TORCH_CHECK(grad.scalar_type() == at::kFloat && total_weight.scalar_type() == at::kFloat,
                "Iris nll_loss backward requires float32");
    const bool per_sample = input.dim() == 2 && reduction == at::Reduction::None;
    TORCH_CHECK(per_sample ? grad.dim() == 1 && grad.numel() == arguments.batch : grad.numel() == 1,
                "Iris nll_loss grad_output shape mismatch");
    TORCH_CHECK(total_weight.numel() == 1, "Iris nll_loss requires scalar total weight");
    auto output = make_output(input.sizes());
    if (arguments.batch) {
        arguments.output = buffer(output);
        arguments.grad = vector_view(grad);
        if (!per_sample) {
            arguments.grad.row_stride = 0;
        }
        arguments.total_weight = vector_view(total_weight);
        if (arguments.batch * arguments.classes <= 262144) {
            iris::nll_loss_backward_fused(buffer(output), arguments.target, arguments.weight,
                                          arguments.grad, arguments.total_weight,
                                          arguments.batch, arguments.classes, arguments.ignore_index,
                                          arguments.mean);
        } else {
            zero(output);
            iris::nll_loss_async(arguments);
        }
    }
    return output;
}

void check_spatial_nll(const at::Tensor& input, const at::Tensor& target) {
    TORCH_CHECK(input.dim() == 4 && target.dim() == 3 && input.size(0) == target.size(0) &&
                input.size(2) == target.size(1) && input.size(3) == target.size(2),
                "Iris nll_loss2d expects NCHW input and NHW target with matching spatial dimensions");
}

std::tuple<at::Tensor, at::Tensor> spatial_nll_forward(
    const at::Tensor& input, const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, c10::SymInt ignore_index) {
    check_spatial_nll(input, target);
    const auto rows = input.permute({0, 2, 3, 1}).reshape({target.numel(), input.size(1)});
    auto [output, total] = nll_forward(rows, target.reshape({-1}), weight, reduction, ignore_index.expect_int());
    if (reduction == at::Reduction::None) output = output.view(target.sizes());
    return {output, total};
}

at::Tensor spatial_nll_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& target,
                                const std::optional<at::Tensor>& weight, int64_t reduction,
                                c10::SymInt ignore_index, const at::Tensor& total_weight) {
    check_spatial_nll(input, target);
    const auto rows = input.permute({0, 2, 3, 1}).reshape({target.numel(), input.size(1)});
    const auto flat_grad = reduction == at::Reduction::None ? grad.reshape({-1}) : grad;
    const auto result = nll_backward(flat_grad, rows, target.reshape({-1}), weight, reduction,
                                     ignore_index.expect_int(), total_weight);
    return result.view({input.size(0), input.size(2), input.size(3), input.size(1)}).permute({0, 3, 1, 2});
}

at::Tensor mse_loss(const at::Tensor& input, const at::Tensor& target, int64_t reduction) {
    check_operand(input);
    check_operand(target);
    check_reduction(reduction);
    at::Tensor squared;
    if (input.scalar_type() == at::kFloat && target.scalar_type() == at::kFloat &&
        input.sizes() == target.sizes() && packed(input) && packed(target)) {
        squared = make_output(input.sizes());
        elementwise(Elementwise::SquaredDifference, buffer(squared), buffer(input), buffer(target),
                    squared.numel(), 1.0f);
    } else {
        squared = binary_strided(Elementwise::SquaredDifference, input, target, 1.0f);
    }
    if (reduction == at::Reduction::None) {
        return squared;
    }
    return reduce_dims(squared, std::nullopt, false, std::nullopt, reduction == at::Reduction::Mean);
}

at::Tensor mse_loss_backward(
    const at::Tensor& grad, const at::Tensor& input, const at::Tensor& target,
    int64_t reduction) {
    check_operand(grad);
    check_operand(input);
    check_operand(target);
    check_reduction(reduction);
    const auto scale = reduction == at::Reduction::Mean ? 2.0f / input.numel() : 2.0f;
    at::Tensor difference;
    if (input.scalar_type() == at::kFloat && target.scalar_type() == at::kFloat &&
        input.sizes() == target.sizes() && packed(input) && packed(target)) {
        difference = make_output(input.sizes());
        elementwise(Elementwise::ScaledDifference, buffer(difference), buffer(input), buffer(target),
                    difference.numel(), scale);
    } else {
        difference = binary_strided(Elementwise::ScaledDifference, input, target, scale);
    }
    return multiply_tensor(difference, grad);
}

at::Tensor fmod_tensor(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    if (right.device().is_cpu() && right.dim() == 0) {
        const auto dt = at::result_type(left, right);
        if (supported_float(dt)) {
            return unary_strided(Elementwise::FmodScalar, left.to(dt), right.item<float>());
        }
        auto s = make_output({}, dt);
        fill_scalar(s, right.item());
        return integral_binary(left, s, dt, dt, IntegerOperation::Fmod, 1);
    }
    check_device(right.device());
    const auto dt = at::result_type(left, right);
    if (!supported_float(dt)) {
        return integral_binary(left, right, dt, dt, IntegerOperation::Fmod, 1);
    }
    return binary_strided(Elementwise::FmodTensor, left.to(dt), right.to(dt), 1.0f);
}

at::Tensor fmod_scalar(const at::Tensor& left, const at::Scalar& right) {
    check_device(left.device());
    const auto dt = at::result_type(left, right);
    if (!supported_float(dt)) {
        auto s = make_output({}, dt);
        fill_scalar(s, right);
        return integral_binary(left, s, dt, dt, IntegerOperation::Fmod, 1);
    }
    return unary_strided(Elementwise::FmodScalar, left.to(dt), right.toFloat());
}

at::Tensor& fmod_scalar_out(const at::Tensor& left, const at::Scalar& right, at::Tensor& out) {
    return write_result(out, fmod_scalar(left, right));
}

at::Tensor& fmod_scalar_inplace(at::Tensor& left, const at::Scalar& right) {
    return write_inplace(left, fmod_scalar(left, right));
}

at::Tensor remainder_tensor(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    if (right.device().is_cpu() && right.dim() == 0) {
        const auto dt = at::result_type(left, right);
        if (supported_float(dt)) {
            return unary_strided(Elementwise::RemainderScalar, left.to(dt), right.item<float>());
        }
        auto s = make_output({}, dt);
        fill_scalar(s, right.item());
        return integral_binary(left, s, dt, dt, IntegerOperation::Remainder, 1);
    }
    check_device(right.device());
    const auto dt = at::result_type(left, right);
    if (!supported_float(dt)) {
        return integral_binary(left, right, dt, dt, IntegerOperation::Remainder, 1);
    }
    return binary_strided(Elementwise::RemainderTensor, left.to(dt), right.to(dt), 1.0f);
}

at::Tensor remainder_scalar(const at::Tensor& left, const at::Scalar& right) {
    check_device(left.device());
    const auto dt = at::result_type(left, right);
    if (!supported_float(dt)) {
        auto s = make_output({}, dt);
        fill_scalar(s, right);
        return integral_binary(left, s, dt, dt, IntegerOperation::Remainder, 1);
    }
    return unary_strided(Elementwise::RemainderScalar, left.to(dt), right.toFloat());
}

at::Tensor remainder_scalar_tensor(const at::Scalar& left, const at::Tensor& right) {
    check_device(right.device());
    const auto dt = at::result_type(left, right);
    if (!supported_float(dt)) {
        auto s = make_output({}, dt);
        fill_scalar(s, left);
        return integral_binary(s, right, dt, dt, IntegerOperation::Remainder, 1);
    }
    return unary_strided(Elementwise::RemainderScalarTensor, right.to(dt), left.toFloat());
}

at::Tensor& remainder_scalar_out(const at::Tensor& left, const at::Scalar& right, at::Tensor& out) {
    return write_result(out, remainder_scalar(left, right));
}

at::Tensor& remainder_scalar_inplace(at::Tensor& left, const at::Scalar& right) {
    return write_inplace(left, remainder_scalar(left, right));
}

at::Tensor& remainder_scalar_tensor_out(const at::Scalar& left, const at::Tensor& right, at::Tensor& out) {
    return write_result(out, remainder_scalar_tensor(left, right));
}

at::Tensor atan2_tensor(const at::Tensor& left, const at::Tensor& right) {
    check_device(left.device());
    check_device(right.device());
    auto dt = at::result_type(left, right);
    if (!supported_float(dt)) dt = at::kFloat;
    return binary_strided(Elementwise::Atan2, left.to(dt), right.to(dt), 1.0f);
}

C10_REGISTER_GUARD_IMPL(PrivateUse1, IrisGuard);
REGISTER_ALLOCATOR(device_type, &allocator);

TORCH_LIBRARY(torch_iris, library) {
    library.def("scale_add(Tensor input, float scale, float bias) -> Tensor");
    library.def("fft_radix2(Tensor real, Tensor imaginary, bool forward) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_iris, PrivateUse1, library) {
    library.impl("scale_add", &scale_add_op);
    library.impl("fft_radix2", &fft_radix2_op);
}

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, library) {
    library.impl("_sparse_coo_tensor_with_dims", &at::native::new_with_dims_sparse);
    library.impl("_sparse_coo_tensor_with_dims_and_tensors", &at::native::new_with_dims_and_tensor_sparse_symint);
    library.impl("_indices", &at::native::_indices_sparse);
    library.impl("_values", &at::native::_values_sparse);
    library.impl("indices", &at::native::indices_sparse);
    library.impl("values", &at::native::values_sparse);
    library.impl("_nnz", &at::native::_nnz_sparse);
    library.impl("sparse_dim", &at::native::sparse_dim_sparse);
    library.impl("dense_dim", &at::native::dense_dim_sparse);
    library.impl("is_coalesced", &at::native::is_coalesced_sparse);
    library.impl("_coalesced_", &at::native::_coalesced_sparse_);
    library.impl("copy_", &at::native::copy_sparse_wrapper_);
    library.impl("copy_sparse_to_sparse_", &at::native::copy_sparse_);
    library.impl("_copy_from", &copy_sparse_from);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, library) {
    library.impl("view_as_real", &at::native::view_as_real);
    library.impl("view_as_complex", &at::native::view_as_complex);
    library.impl("view.dtype", &at::native::view_dtype);
    library.impl("_conj_physical", &conjugate_physical);
    library.impl("conj_physical_", &conjugate_physical_inplace);
    library.impl("empty.memory_format", &empty);
    library.impl("empty_strided", &empty_strided);
    library.impl("resize_", &resize_tensor);
    library.impl("set_.source_Storage", &set_storage_full);
    library.impl("set_.source_Storage_storage_offset", &set_storage);
    library.impl("_local_scalar_dense", &local_scalar);
    library.impl("copy_", &copy_);
    library.impl("_copy_from", &copy_from);
    library.impl("_to_copy", &to_copy);
    library.impl("record_stream", &record_tensor_stream);
    library.impl("as_strided", &at::native::as_strided_tensorimpl);
    library.impl("view", &at::native::view);
    library.impl("_unsafe_view", &at::native::_unsafe_view);
    library.impl("_reshape_alias", &at::native::_reshape_alias);
    library.impl("add.Tensor", &add_tensor);
    library.impl("add_.Tensor", &add_inplace);
    library.impl("mul.Tensor", &multiply_tensor);
    library.impl("mul_.Scalar", &multiply_scalar_inplace);
    library.impl("mul_.Tensor", &multiply_inplace);
    library.impl("div.Scalar", &divide_scalar);
    library.impl("relu", &relu);
    library.impl("relu_", &pointwise_unary_inplace<Elementwise::Relu>);
    library.impl("relu.out", &pointwise_unary_out<Elementwise::Relu>);
    library.impl("threshold_backward", &threshold_backward);
    library.impl("fill_.Scalar", &fill_scalar);
    library.impl("zero_", &zero);
    library.impl("uniform_", &uniform);
    library.impl("normal_", &normal);
    library.impl("bernoulli_.float", &bernoulli);
#define IRIS_REGISTER_UNARY(name, op) \
    library.impl(name, &pointwise_unary<Elementwise::op>); \
    library.impl(name "_", &pointwise_unary_inplace<Elementwise::op>); \
    library.impl(name ".out", &pointwise_unary_out<Elementwise::op>)
    IRIS_REGISTER_UNARY("ceil", Ceil);
    IRIS_REGISTER_UNARY("floor", Floor);
    IRIS_REGISTER_UNARY("trunc", Trunc);
    library.impl("abs", &typed_unary<Elementwise::Abs, IntegerOperation::Abs>);
    library.impl("abs_", &typed_unary_inplace<typed_unary<Elementwise::Abs, IntegerOperation::Abs>, Elementwise::Abs>);
    library.impl("abs.out", &typed_unary_out<typed_unary<Elementwise::Abs, IntegerOperation::Abs>, Elementwise::Abs>);
    IRIS_REGISTER_UNARY("sqrt", Sqrt);
    IRIS_REGISTER_UNARY("rsqrt", Rsqrt);
    IRIS_REGISTER_UNARY("exp", Exp);
    IRIS_REGISTER_UNARY("log", Log);
    IRIS_REGISTER_UNARY("tanh", Tanh);
    IRIS_REGISTER_UNARY("sigmoid", Sigmoid);
    IRIS_REGISTER_UNARY("silu", Silu);
    IRIS_REGISTER_UNARY("reciprocal", Reciprocal);
    IRIS_REGISTER_UNARY("sin", Sin);
    IRIS_REGISTER_UNARY("cos", Cos);
    IRIS_REGISTER_UNARY("erf", Erf);
    IRIS_REGISTER_UNARY("log1p", Log1p);
    IRIS_REGISTER_UNARY("expm1", Expm1);
    IRIS_REGISTER_UNARY("exp2", Exp2);
    IRIS_REGISTER_UNARY("log2", Log2);
    IRIS_REGISTER_UNARY("log10", Log10);
    IRIS_REGISTER_UNARY("tan", Tan);
    IRIS_REGISTER_UNARY("asin", Asin);
    IRIS_REGISTER_UNARY("acos", Acos);
    IRIS_REGISTER_UNARY("atan", Atan);
    IRIS_REGISTER_UNARY("sinh", Sinh);
    IRIS_REGISTER_UNARY("cosh", Cosh);
    IRIS_REGISTER_UNARY("asinh", Asinh);
    IRIS_REGISTER_UNARY("acosh", Acosh);
    IRIS_REGISTER_UNARY("atanh", Atanh);
    IRIS_REGISTER_UNARY("erfc", Erfc);
    IRIS_REGISTER_UNARY("frac", Frac);
#undef IRIS_REGISTER_UNARY
    library.impl("sign", &typed_unary<Elementwise::Sign, IntegerOperation::Sign>);
    library.impl("sign_", &typed_unary_inplace<typed_unary<Elementwise::Sign, IntegerOperation::Sign>, Elementwise::Sign>);
    library.impl("sign.out", &typed_unary_out<typed_unary<Elementwise::Sign, IntegerOperation::Sign>, Elementwise::Sign>);
    library.impl("sgn", &typed_unary<Elementwise::Sign, IntegerOperation::Sign>);
    library.impl("sgn_", &typed_unary_inplace<typed_unary<Elementwise::Sign, IntegerOperation::Sign>, Elementwise::Sign>);
    library.impl("sgn.out", &typed_unary_out<typed_unary<Elementwise::Sign, IntegerOperation::Sign>, Elementwise::Sign>);
    library.impl("round", &round_tensor);
    library.impl("round_", &typed_unary_inplace<round_tensor, Elementwise::Round>);
    library.impl("round.out", &typed_unary_out<round_tensor, Elementwise::Round>);
    library.impl("round.decimals", &round_decimals);
    library.impl("round_.decimals", &round_decimals_inplace);
    library.impl("round.decimals_out", &round_decimals_out);
    library.impl("softplus", &softplus);
    library.impl("softplus_backward", &softplus_backward);
    library.impl("log_sigmoid_forward", &log_sigmoid_forward);
    library.impl("log_sigmoid_backward", &log_sigmoid_backward);
    library.impl("unfold", &unfold);
    library.impl("unfold_backward", &unfold_backward);
    library.impl("eye.m_out", &eye_out);
    library.impl("_foreach_add_.Scalar", &foreach_add_scalar);
    library.impl("_amp_foreach_non_finite_check_and_unscale_", &amp_unscale);
    library.impl("_amp_update_scale_", &amp_update_scale);
    library.impl("atan2", &atan2_tensor);
    library.impl("atan2.out", &binary_out<atan2_tensor>);
    library.impl("atan2_", &binary_inplace<atan2_tensor>);
    library.impl("fmod.Tensor", &fmod_tensor);
    library.impl("fmod.Scalar", &fmod_scalar);
    library.impl("fmod.Tensor_out", &binary_out<fmod_tensor>);
    library.impl("fmod.Scalar_out", &fmod_scalar_out);
    library.impl("fmod_.Tensor", &binary_inplace<fmod_tensor>);
    library.impl("fmod_.Scalar", &fmod_scalar_inplace);
    library.impl("remainder.Tensor", &remainder_tensor);
    library.impl("remainder.Scalar", &remainder_scalar);
    library.impl("remainder.Scalar_Tensor", &remainder_scalar_tensor);
    library.impl("remainder.Tensor_out", &binary_out<remainder_tensor>);
    library.impl("remainder.Scalar_out", &remainder_scalar_out);
    library.impl("remainder.Scalar_Tensor_out", &remainder_scalar_tensor_out);
    library.impl("remainder_.Tensor", &binary_inplace<remainder_tensor>);
    library.impl("remainder_.Scalar", &remainder_scalar_inplace);
    library.impl("neg", &neg_tensor);
    library.impl("neg_", &neg_inplace);
    library.impl("neg.out", &neg_out);
#define IRIS_REGISTER_COMPARE(name, op) \
    library.impl(name ".Tensor", &compare_tensor<Elementwise::op>); \
    library.impl(name ".Scalar", &compare_scalar<Elementwise::op>)
    IRIS_REGISTER_COMPARE("eq", Eq);
    IRIS_REGISTER_COMPARE("ne", Ne);
    IRIS_REGISTER_COMPARE("lt", Lt);
    IRIS_REGISTER_COMPARE("le", Le);
    IRIS_REGISTER_COMPARE("gt", Gt);
    IRIS_REGISTER_COMPARE("ge", Ge);
#undef IRIS_REGISTER_COMPARE
    library.impl("isfinite", &predicate<Elementwise::IsFinite>);
    library.impl("isnan", &predicate<Elementwise::IsNan>);
    library.impl("isinf", &predicate<Elementwise::IsInf>);
    library.impl("bitwise_and.Tensor", &bool_and);
    library.impl("bitwise_and.Scalar", &bitwise_scalar<IntegerOperation::BitwiseAnd>);
    library.impl("bitwise_and.Tensor_out", &binary_out<bitwise_tensor<IntegerOperation::BitwiseAnd>>);
    library.impl("bitwise_and.Scalar_out", &bitwise_scalar_out<IntegerOperation::BitwiseAnd>);
    library.impl("bitwise_or.Tensor", &bitwise_tensor<IntegerOperation::BitwiseOr>);
    library.impl("bitwise_or.Scalar", &bitwise_scalar<IntegerOperation::BitwiseOr>);
    library.impl("bitwise_or.Tensor_out", &binary_out<bitwise_tensor<IntegerOperation::BitwiseOr>>);
    library.impl("bitwise_or.Scalar_out", &bitwise_scalar_out<IntegerOperation::BitwiseOr>);
    library.impl("bitwise_xor.Tensor", &bitwise_tensor<IntegerOperation::BitwiseXor>);
    library.impl("bitwise_xor.Scalar", &bitwise_scalar<IntegerOperation::BitwiseXor>);
    library.impl("bitwise_xor.Tensor_out", &binary_out<bitwise_tensor<IntegerOperation::BitwiseXor>>);
    library.impl("bitwise_xor.Scalar_out", &bitwise_scalar_out<IntegerOperation::BitwiseXor>);
    library.impl("bitwise_not", &bitwise_not_tensor);
    library.impl("bitwise_not.out", &bitwise_not_out);
    library.impl("logical_and", &logical_binary<Elementwise::LogicalAnd>);
    library.impl("logical_and.out", &binary_out<logical_binary<Elementwise::LogicalAnd>>);
    library.impl("logical_or", &logical_binary<Elementwise::LogicalOr>);
    library.impl("logical_or.out", &binary_out<logical_binary<Elementwise::LogicalOr>>);
    library.impl("logical_xor", &logical_binary<Elementwise::LogicalXor>);
    library.impl("logical_xor.out", &binary_out<logical_binary<Elementwise::LogicalXor>>);
    library.impl("logical_not", &logical_not_tensor);
    library.impl("logical_not.out", &logical_not_out);
    library.impl("add.Scalar", &add_scalar);
    library.impl("add_.Scalar", &add_scalar_inplace);
    library.impl("sub.Tensor", &subtract_tensor);
    library.impl("sub.Scalar", &sub_scalar);
    library.impl("rsub.Scalar", &rsub_scalar);
    library.impl("rsub.Tensor", &rsub_tensor);
    library.impl("mul.Scalar", &pointwise_scalar<Elementwise::Scale>);
    library.impl("div.Tensor", &divide_tensor);
    library.impl("div.Tensor_mode", &divide_mode);
    library.impl("div.out_mode", &divide_mode_out);
    library.impl("div_.Tensor_mode", &divide_mode_inplace);
    library.impl("floor_divide", &floor_divide_tensor);
    library.impl("floor_divide.out", &binary_out<floor_divide_tensor>);
    library.impl("floor_divide_.Tensor", &binary_inplace<floor_divide_tensor>);
    library.impl("div_.Scalar", &pointwise_scalar_inplace<Elementwise::Divide>);
    library.impl("pow.Tensor_Scalar", &pointwise_scalar<Elementwise::PowScalar>);
    library.impl("pow.Tensor_Tensor", &pointwise_binary<Elementwise::PowTensor>);
    library.impl("pow.Scalar", &pow_scalar_base);
    library.impl("pow.Scalar_out", &pow_scalar_base_out);
    library.impl("clamp", &clamp_tensor);
    library.impl("gelu", &gelu);
    library.impl("gelu_backward", &gelu_backward);
    library.impl("silu_backward", &pointwise_binary<Elementwise::SiluBackward>);
    library.impl("tanh_backward", &pointwise_binary<Elementwise::TanhBackward>);
    library.impl("sigmoid_backward", &pointwise_binary<Elementwise::SigmoidBackward>);
    library.impl("lerp_.Scalar", &lerp_scalar_inplace);
    library.impl("addcmul_", &addcmul_inplace);
    library.impl("addcdiv_", &addcdiv_inplace);
    library.impl("maximum", &typed_extrema<Elementwise::Maximum, IntegerOperation::Maximum>);
    library.impl("maximum.out", &maximum_out);
    library.impl("minimum", &typed_extrema<Elementwise::Minimum, IntegerOperation::Minimum>);
    library.impl("min", &extrema_all<Extrema::Min>);
    library.impl("max", &extrema_all<Extrema::Max>);
    library.impl("amin", &extrema_dims<Extrema::Min>);
    library.impl("amin.out", &extrema_dims_out<Extrema::Min>);
    library.impl("amax", &extrema_dims<Extrema::Max>);
    library.impl("amax.out", &extrema_dims_out<Extrema::Max>);
    library.impl("aminmax", &aminmax_impl);
    library.impl("aminmax.out", &aminmax_out);
    library.impl("any", &extrema_all<Extrema::Any>);
    library.impl("all", &extrema_all<Extrema::All>);
    library.impl("masked_select", &masked_select);
    library.impl("cat", &concatenate);
    library.impl("nonzero", &nonzero_indices);
    library.impl("index.Tensor", &advanced_read);
    library.impl("_index_put_impl_", &advanced_put);
    library.impl("index_select", &index_select);
    library.impl("gather", &gather);
    library.impl("scatter.src", &scatter<IndexOperation::Scatter>);
    library.impl("scatter.src_out", &indexed_out<scatter<IndexOperation::Scatter>, at::Tensor>);
    library.impl("scatter.value", &scatter_value);
    library.impl("scatter.value_out", &scatter_value_out);
    library.impl("scatter_.src", &scatter_inplace<IndexOperation::Scatter>);
    library.impl("scatter_.value", &indexed_inplace<scatter_value, at::Scalar>);
    library.impl("scatter_add", &scatter<IndexOperation::ScatterAdd>);
    library.impl("scatter_add_", &scatter_inplace<IndexOperation::ScatterAdd>);
    library.impl("scatter_add.out", &indexed_out<scatter<IndexOperation::ScatterAdd>, at::Tensor>);
    library.impl("scatter_reduce.two", &scatter_reduce_impl);
    library.impl("scatter_reduce.two_out", &scatter_reduce_out);
    library.impl("scatter_reduce_.two", &scatter_reduce_inplace);
    library.impl("index_copy", &index_copy_impl);
    library.impl("index_copy.out", &indexed_out<index_copy_impl, at::Tensor>);
    library.impl("index_copy_", &indexed_inplace<index_copy_impl, at::Tensor>);
    library.impl("index_fill.int_Scalar", &index_fill_scalar);
    library.impl("index_fill.int_Scalar_out", &indexed_out<index_fill_scalar, at::Scalar>);
    library.impl("index_fill_.int_Scalar", &indexed_inplace<index_fill_scalar, at::Scalar>);
    library.impl("index_fill.int_Tensor", &index_fill_tensor);
    library.impl("index_fill.int_Tensor_out", &indexed_out<index_fill_tensor, at::Tensor>);
    library.impl("index_fill_.int_Tensor", &indexed_inplace<index_fill_tensor, at::Tensor>);
    library.impl("take", &take_impl);
    library.impl("take.out", &binary_out<take_impl>);
    library.impl("put", &put_impl);
    library.impl("put.out", &put_out);
    library.impl("put_", &put_inplace);
    library.impl("embedding_dense_backward", &embedding_backward);

    library.impl("any.dim", &logical_dim<false>);
    library.impl("any.dims", &logical_dims<false>);
    library.impl("all.dim", &logical_dim<true>);
    library.impl("all.dims", &logical_dims<true>);
    library.impl("isneginf", &infinity_predicate<true>);
    library.impl("isneginf.out", &infinity_predicate_out<true>);
    library.impl("isposinf", &infinity_predicate<false>);
    library.impl("isposinf.out", &infinity_predicate_out<false>);
    library.impl("tril", &triangular_tensor<false>);
    library.impl("tril.out", &triangular_out<false>);
    library.impl("tril_indices", &tril_indices_factory);
    library.impl("tril_indices.out", &tril_indices_out);
    library.impl("triu", &triangular_tensor<true>);
    library.impl("triu.out", &triangular_out<true>);
    library.impl("masked_fill.Scalar", &masked_fill_scalar);
    library.impl("masked_fill_.Scalar", &masked_fill_scalar_inplace);
    library.impl("masked_fill.Tensor", &masked_fill_tensor);
    library.impl("masked_fill_.Tensor", &masked_fill_tensor_inplace);

    library.impl("add.out", &binary_alpha_out<add_tensor>);
    library.impl("sub.out", &binary_alpha_out<subtract_tensor>);
    library.impl("sub_.Tensor", &subtract_inplace);
    library.impl("mul.out", &binary_out<multiply_tensor>);
    library.impl("div.out", &binary_out<divide_tensor>);
    library.impl("div_.Tensor", &binary_inplace<divide_tensor>);
    library.impl("pow.Tensor_Scalar_out", &pow_scalar_out);
    library.impl("pow.Tensor_Tensor_out", &binary_out<pointwise_binary<Elementwise::PowTensor>>);
    library.impl("pow_.Scalar", &pointwise_scalar_inplace<Elementwise::PowScalar>);
    library.impl("pow_.Tensor", &binary_inplace<pointwise_binary<Elementwise::PowTensor>>);
    library.impl("clamp.out", &clamp_out);
    library.impl("clamp_", &clamp_inplace);
    library.impl("clamp_min", &clamp_min_scalar);
    library.impl("clamp_min.out", &clamp_min_scalar_out);
    library.impl("clamp_min_", &clamp_min_scalar_inplace);
    library.impl("clamp_max", &clamp_max_scalar);
    library.impl("clamp_max.out", &clamp_max_scalar_out);
    library.impl("clamp_max_", &clamp_max_scalar_inplace);
    library.impl("clamp_min.Tensor", &typed_extrema<Elementwise::Maximum, IntegerOperation::Maximum>);
    library.impl("clamp_min.Tensor_out", &binary_out<typed_extrema<Elementwise::Maximum, IntegerOperation::Maximum>>);
    library.impl("clamp_min_.Tensor", &binary_inplace<typed_extrema<Elementwise::Maximum, IntegerOperation::Maximum>>);
    library.impl("clamp_max.Tensor", &typed_extrema<Elementwise::Minimum, IntegerOperation::Minimum>);
    library.impl("clamp_max.Tensor_out", &binary_out<typed_extrema<Elementwise::Minimum, IntegerOperation::Minimum>>);
    library.impl("clamp_max_.Tensor", &binary_inplace<typed_extrema<Elementwise::Minimum, IntegerOperation::Minimum>>);
    library.impl("gelu.out", &gelu_out);
    library.impl("gelu_", &gelu_inplace);
    library.impl("gather.out", &gather_out);
    library.impl("index_select.out", &index_select_out);
    library.impl("dot", &iris_dot);
    library.impl("mv", &iris_mv);
    library.impl("addmv", &iris_addmv);
    library.impl("addmv.out", +[](const at::Tensor& input, const at::Tensor& matrix, const at::Tensor& vector,
                                   const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& output) -> at::Tensor& {
        return write_result(output, iris_addmv(input, matrix, vector, beta, alpha));
    });
    library.impl("mm", &matmul_2d);
    library.impl("matmul", &matmul);
    library.impl("matmul_backward", &matmul_backward);
    library.impl("bmm", &batched_mm);
    library.impl("addmm", &addmm);
    library.impl("sum", &sum_all);
    library.impl("sum.dim_IntList", &sum_dims);
    library.impl("nansum", &nansum_impl);
    library.impl("nansum.out", &nansum_out);
    library.impl("mean", &mean_all);
    library.impl("mean.dim", &mean_dims);
    library.impl("_softmax", &softmax<false>);
    library.impl("_log_softmax", &softmax<true>);
    library.impl("_log_softmax_backward_data", &log_softmax_backward);
    library.impl("where.self", &where_tensor);
    library.impl("where.self_out", &where_out);
    library.impl("native_dropout", &dropout);
    library.impl("native_dropout_backward", &dropout_backward);
    library.impl("nll_loss_forward", &nll_forward);
    library.impl("nll_loss_backward", &nll_backward);
    library.impl("nll_loss2d_forward", &spatial_nll_forward);
    library.impl("nll_loss2d_backward", &spatial_nll_backward);
    library.impl("_softmax_backward_data", &softmax_backward);
    library.impl("mse_loss", &mse_loss);
    library.impl("mse_loss_backward", &mse_loss_backward);
}

}  // namespace

at::Tensor linear_quant_op(
    const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
    const std::optional<at::Tensor>& zeros, const std::optional<at::Tensor>& bias,
    int64_t group_size, bool is_int4) {
    check_operand(x);
    check_device(qweight.device());
    check_operand(scales);
    TORCH_CHECK(qweight.scalar_type() == at::kByte && qweight.dim() == 2,
                "Iris linear_int4/int8 expects 2D uint8 qweight [N, K] or [N, K/2]");
    TORCH_CHECK(x.dim() >= 1, "Iris quantized linear expects at least 1D input");
    const int64_t n = qweight.size(0);
    const int64_t k = is_int4 ? (qweight.size(1) * 2) : qweight.size(1);
    TORCH_CHECK(x.size(-1) == k, "Input last dim (", x.size(-1), ") does not match weight K (", k, ")");
    TORCH_CHECK(group_size > 0 && k % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(!is_int4 || group_size % 8 == 0, "INT4 group_size must be a multiple of 8");
    const int64_t num_groups = k / group_size;
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == n && scales.size(1) == num_groups,
                "scales shape must be [N, K / group_size]");

    at::Tensor z_c;
    if (zeros && zeros->defined()) {
        check_operand(*zeros);
        TORCH_CHECK(zeros->scalar_type() == scales.scalar_type(), "zeros dtype must match scales dtype");
        TORCH_CHECK(zeros->dim() == 2 && zeros->size(0) == n && zeros->size(1) == num_groups,
                    "zeros shape must be [N, K / group_size]");
        z_c = zeros->contiguous();
    }
    at::Tensor b_c;
    if (bias && bias->defined()) {
        check_operand(*bias);
        TORCH_CHECK(bias->scalar_type() == x.scalar_type(), "bias dtype must match input dtype");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n, "bias shape must be [N]");
        b_c = bias->contiguous();
    }

    const auto x_c = x.contiguous();
    const auto qw_c = qweight.contiguous();
    const auto sc_c = scales.contiguous();
    at::DimVector out_shape(x.sizes());
    out_shape.back() = n;
    const int64_t m = x_c.numel() / k;
    auto output = make_output(out_shape, x.scalar_type());
    if (output.numel()) {
        const auto to_slice = [](const at::Tensor& t) -> BufferSlice {
            if (!t.defined()) return {};
            return {buffer(t), static_cast<cl_ulong>(t.storage_offset())};
        };
        linear_quant_forward(buffer(output), to_slice(x_c), to_slice(qw_c), to_slice(sc_c),
                             to_slice(z_c), to_slice(b_c),
                             static_cast<cl_int>(m), static_cast<cl_int>(n), static_cast<cl_int>(k),
                             static_cast<cl_int>(group_size), static_cast<cl_int>(num_groups),
                             x.scalar_type() == at::kHalf, scales.scalar_type() == at::kHalf, is_int4);
    }
    return output;
}

std::pair<std::uint64_t, std::uint64_t> reserve_rng_blocks(std::uint64_t blocks) {
    return reserve_rng_blocks_impl(blocks, std::nullopt);
}

std::pair<std::uint64_t, std::uint64_t> reserve_rng_blocks_opt(
    std::uint64_t blocks, const std::optional<at::Generator>& generator) {
    return reserve_rng_blocks_impl(blocks, generator);
}

}  // namespace iris

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    static iris::IrisHooks hooks;
    at::RegisterPrivateUse1HooksInterface(&hooks);
    module.def("is_available", &iris::gpu_available);
    module.def("device_name", &iris::gpu_name);
    module.def("_has_fp16", &iris::has_fp16);
    module.def("synchronize", &iris::synchronize, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("empty_cache", &iris::empty_cache);
    module.def("memory_stats", [] {
        const auto stats = iris::memory_stats();
        pybind11::dict output;
        output["allocated_bytes"] = stats.allocated_bytes;
        output["cached_bytes"] = stats.cached_bytes;
        output["reserved_bytes"] = stats.allocated_bytes + stats.cached_bytes;
        output["allocations"] = stats.allocations;
        output["reuses"] = stats.reuses;
        output["peak_allocated_bytes"] = stats.peak_allocated_bytes;
        output["peak_reserved_bytes"] = stats.peak_reserved_bytes;
        return output;
    });
    module.def("_reset_peak_memory_stats", &iris::reset_peak_memory_stats);
    module.def("_is_initialized", &iris::context_initialized);
    module.def("_init", &iris::initialize);
    module.def("_device_properties", [] {
        const auto properties = iris::device_properties();
        pybind11::dict output;
        output["name"] = iris::gpu_name();
        output["total_memory"] = properties.total_memory;
        output["max_allocation"] = properties.max_allocation;
        output["double_fp_config"] = properties.double_fp_config;
        output["host_unified_memory"] = properties.host_unified_memory == CL_TRUE;
        output["compute_units"] = properties.compute_units;
        output["opencl_version"] = properties.opencl_version;
        output["driver_version"] = properties.driver_version;
        return output;
    });
    module.def("_manual_seed", &iris::manual_seed);
    module.def("_default_generator", [] { return iris::default_iris_generator(); });
    module.def("to_dlpack", &iris::export_dlpack);
    module.def("from_dlpack", &iris::import_dlpack);
    module.def("_kernel_submissions", &iris::kernel_submissions);
    module.def("_matmul_tile_submissions", [] {
        return pybind11::make_tuple(iris::tiled_matmul_submissions(), iris::blocked_matmul_submissions());
    });
    module.def("_linear_int8", [](const at::Tensor& x, const at::Tensor& qw, const at::Tensor& sc,
                                  const std::optional<at::Tensor>& zp, const std::optional<at::Tensor>& b,
                                  int64_t gs) {
        return iris::linear_quant_op(x, qw, sc, zp, b, gs, false);
    });
    module.def("_linear_int4", [](const at::Tensor& x, const at::Tensor& qw, const at::Tensor& sc,
                                  const std::optional<at::Tensor>& zp, const std::optional<at::Tensor>& b,
                                  int64_t gs) {
        return iris::linear_quant_op(x, qw, sc, zp, b, gs, true);
    });
}
