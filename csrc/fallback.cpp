#include "fallback.h"

#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace iris {
namespace {

bool read_env_flag(const char* name, bool default_value) {
    char* raw = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&raw, &len, name) != 0 || raw == nullptr) {
        return default_value;
    }
    std::string value(raw);
    std::free(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    return default_value;
}

std::atomic<bool> cpu_fallback_enabled{read_env_flag("TORCH_IRIS_CPU_FALLBACK", true)};
std::atomic<bool> fallback_warn_enabled{read_env_flag("TORCH_IRIS_FALLBACK_WARN", true)};

std::mutex fallback_mutex;
std::unordered_map<std::string, std::uint64_t> fallback_counts;
std::unordered_set<std::string> warned_ops;

constexpr auto kIrisDeviceType = c10::DeviceType::PrivateUse1;

bool is_fp64_dtype(at::ScalarType dtype) noexcept {
    return dtype == at::kDouble || dtype == at::kComplexDouble;
}

bool is_scalar_type_arg(const c10::TypePtr& type) {
    if (!type) return false;
    if (type->kind() == c10::TypeKind::OptionalType) {
        const auto inner = type->castRaw<c10::OptionalType>()->getElementType();
        return inner && inner->kind() == c10::TypeKind::IntType;
    }
    return type->kind() == c10::TypeKind::IntType;
}

bool contains_iris_tensor(const c10::IValue& ivalue) {
    if (ivalue.isTensor()) {
        const auto& t = ivalue.toTensor();
        return t.defined() && t.device().type() == kIrisDeviceType;
    }
    if (ivalue.isTensorList()) {
        for (const auto& t : ivalue.toTensorVector()) {
            if (t.defined() && t.device().type() == kIrisDeviceType) {
                return true;
            }
        }
    } else if (ivalue.isList()) {
        for (const c10::IValue& elem : ivalue.toListRef()) {
            if (elem.isTensor()) {
                const auto& t = elem.toTensor();
                if (t.defined() && t.device().type() == kIrisDeviceType) {
                    return true;
                }
            }
        }
    }
    return false;
}

void move_ivalue_to_iris(c10::IValue& ivalue, c10::Device target_device) {
    if (ivalue.isTensor()) {
        const auto& t = ivalue.toTensor();
        if (t.defined() && t.device().is_cpu()) {
            ivalue = t.to(target_device);
        }
        return;
    }
    if (ivalue.isTensorList()) {
        auto tensors = ivalue.toTensorVector();
        for (auto& t : tensors) {
            if (t.defined() && t.device().is_cpu()) {
                t = t.to(target_device);
            }
        }
        ivalue = c10::List<at::Tensor>(tensors);
        return;
    }
    if (ivalue.isTuple()) {
        auto elements = ivalue.toTupleRef().elements().vec();
        for (auto& elem : elements) {
            move_ivalue_to_iris(elem, target_device);
        }
        ivalue = c10::ivalue::Tuple::create(std::move(elements));
    }
}

void record_fallback_and_warn(const std::string& op_name) {
    bool should_warn = false;
    {
        std::lock_guard<std::mutex> lock(fallback_mutex);
        ++fallback_counts[op_name];
        if (fallback_warn_enabled.load(std::memory_order_relaxed)) {
            should_warn = warned_ops.insert(op_name).second;
        }
    }
    if (should_warn) {
        TORCH_WARN(
            "torch-iris: operator '", op_name,
            "' is not natively implemented for Intel Iris Xe and fell back to CPU "
            "(warning shown once per operator; set TORCH_IRIS_FALLBACK_WARN=0 to silence "
            "or TORCH_IRIS_CPU_FALLBACK=0 / torch_iris.set_cpu_fallback_enabled(False) for strict GPU-only mode).");
    }
}

struct ReentrancyGuard {
    static thread_local std::vector<const void*> active;
    const void* key;

    explicit ReentrancyGuard(const void* op_key, const std::string& op_name) : key(op_key) {
        TORCH_CHECK_NOT_IMPLEMENTED(
            std::find(active.begin(), active.end(), key) == active.end(),
            "Recursive CPU fallback detected for operator '", op_name, "' on the 'iris' backend.");
        active.push_back(key);
    }

    ~ReentrancyGuard() {
        if (!active.empty() && active.back() == key) {
            active.pop_back();
        }
    }
};

thread_local std::vector<const void*> ReentrancyGuard::active;

void iris_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Preserve PyTorch's native CompositeExplicitAutograd decompositions on Iris
    // so composite ops execute via native OpenCL kernels rather than falling back to CPU.
    if (op.hasKernelForDispatchKey(c10::DispatchKey::CompositeExplicitAutogradNonFunctional)) {
        op.callBoxedForDispatchKey(c10::DispatchKey::CompositeExplicitAutogradNonFunctional, *stack);
        return;
    }
    if (op.hasKernelForDispatchKey(c10::DispatchKey::CompositeExplicitAutograd)) {
        op.callBoxedForDispatchKey(c10::DispatchKey::CompositeExplicitAutograd, *stack);
        return;
    }

    const std::string op_name = c10::toString(op.operator_name());

    TORCH_CHECK_NOT_IMPLEMENTED(
        cpu_fallback_enabled.load(std::memory_order_relaxed),
        "Operator '", op_name,
        "' is not natively implemented for the 'iris' backend and CPU fallback is disabled. "
        "Enable fallback via torch_iris.set_cpu_fallback_enabled(True) or TORCH_IRIS_CPU_FALLBACK=1.");

    TORCH_CHECK_NOT_IMPLEMENTED(
        op.hasComputedKernelForDispatchKey(c10::DispatchKey::CPU),
        "Operator '", op_name,
        "' is not implemented for the 'iris' backend and has no CPU fallback kernel.");

    ReentrancyGuard guard(&op.schema(), op_name);

    const auto& schema = op.schema();
    const auto& args = schema.arguments();
    const std::size_t num_args = args.size();
    const std::size_t stack_start = stack->size() - num_args;

    bool has_iris_tensor = false;
    bool has_iris_device_arg = false;
    c10::Device target_device(kIrisDeviceType, 0);

    for (std::size_t i = 0; i < num_args; ++i) {
        const auto& val = (*stack)[stack_start + i];
        // Explicitly reject FP64/Complex128 dtype requests on Iris rather than
        // silently downcasting or running wasted CPU work before failing copy-back.
        if (args[i].name() == "dtype" && val.isInt() && is_scalar_type_arg(args[i].type())) {
            const auto requested = static_cast<at::ScalarType>(val.toInt());
            TORCH_CHECK(
                !is_fp64_dtype(requested),
                "Iris supports float32, float16, bfloat16, complex64, int64, int32, int16, int8, uint8 and bool storage "
                "(operator '", op_name, "' requested unsupported dtype ", requested, ").");
        }
        if (val.isDevice() && val.toDevice().type() == kIrisDeviceType) {
            const auto dev = val.toDevice();
            TORCH_CHECK(dev.index() == 0 || dev.index() == -1, "Iris supports only device iris:0");
            has_iris_device_arg = true;
            target_device = c10::Device(kIrisDeviceType, 0);
        }
        if (contains_iris_tensor(val)) {
            has_iris_tensor = true;
        }
    }

    record_fallback_and_warn(op_name);

    if (has_iris_device_arg) {
        for (std::size_t i = 0; i < num_args; ++i) {
            auto& val = (*stack)[stack_start + i];
            if (val.isDevice() && val.toDevice().type() == kIrisDeviceType) {
                val = c10::Device(c10::kCPU);
            }
        }
    }

    if (!has_iris_tensor && has_iris_device_arg) {
        // Pure factory op with device='iris' and no Iris tensor arguments:
        // execute on CPU and copy returned tensors back to iris:0.
        op.callBoxedForDispatchKey(c10::DispatchKey::CPU, *stack);
        for (std::size_t i = stack_start; i < stack->size(); ++i) {
            move_ivalue_to_iris((*stack)[i], target_device);
        }
        return;
    }

    at::native::cpu_fallback(op, stack, /*error_on_views=*/false);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&iris_cpu_fallback>());
}

}  // namespace

bool is_cpu_fallback_enabled() noexcept {
    return cpu_fallback_enabled.load(std::memory_order_relaxed);
}

void set_cpu_fallback_enabled(bool enabled) noexcept {
    cpu_fallback_enabled.store(enabled, std::memory_order_relaxed);
}

bool is_fallback_warning_enabled() noexcept {
    return fallback_warn_enabled.load(std::memory_order_relaxed);
}

void set_fallback_warning_enabled(bool enabled) noexcept {
    fallback_warn_enabled.store(enabled, std::memory_order_relaxed);
}

std::unordered_map<std::string, std::uint64_t> fallback_stats() {
    std::lock_guard<std::mutex> lock(fallback_mutex);
    return fallback_counts;
}

void reset_fallback_stats() {
    std::lock_guard<std::mutex> lock(fallback_mutex);
    fallback_counts.clear();
    warned_ops.clear();
}

}  // namespace iris
