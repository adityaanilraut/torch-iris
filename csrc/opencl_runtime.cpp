#include "opencl_runtime.h"

#include "kernels.h"
#include "ops_kernels.h"
#include "ops_phase3_kernels.h"
#include "spatial_kernels.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace iris {
namespace {

#define IRIS_OPENCL_FUNCTIONS(X) \
    X(clGetPlatformIDs) X(clGetDeviceIDs) X(clGetDeviceInfo) \
    X(clCreateContext) X(clReleaseContext) \
    X(clCreateCommandQueue) X(clReleaseCommandQueue) X(clFinish) X(clFlush) \
    X(clEnqueueMarkerWithWaitList) X(clEnqueueBarrierWithWaitList) X(clWaitForEvents) \
    X(clGetEventInfo) X(clGetEventProfilingInfo) X(clReleaseEvent) \
    X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) \
    X(clCreateBuffer) X(clReleaseMemObject) X(clGetMemObjectInfo) \
    X(clEnqueueWriteBuffer) X(clEnqueueReadBuffer) X(clEnqueueCopyBuffer) \
    X(clEnqueueFillBuffer) X(clEnqueueNDRangeKernel) \
    X(clCreateProgramWithSource) X(clBuildProgram) \
    X(clGetProgramBuildInfo) X(clReleaseProgram) \
    X(clCreateKernel) X(clSetKernelArg) X(clGetKernelWorkGroupInfo) X(clReleaseKernel)

struct OpenCL {
#define DECLARE(name) decltype(&::name) name = nullptr;
    IRIS_OPENCL_FUNCTIONS(DECLARE)
#undef DECLARE

    OpenCL() {
        std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(&FreeLibrary)>
            dll(LoadLibraryExW(L"OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32),
                &FreeLibrary);
        TORCH_CHECK(dll, "Iris could not load system OpenCL.dll (Windows error ",
                    GetLastError(), ")");
#define LOAD(name) \
        name = reinterpret_cast<decltype(name)>(GetProcAddress(dll.get(), #name)); \
        TORCH_CHECK(name, "Iris OpenCL.dll is missing ", #name);
        IRIS_OPENCL_FUNCTIONS(LOAD)
#undef LOAD
        // Keep the loader alive for buffer deleters during process teardown.
        dll.release();
    }
};

#undef IRIS_OPENCL_FUNCTIONS

const OpenCL& cl() {
    static const OpenCL api;
    return api;
}

const char* cl_error_name(cl_int status) noexcept {
    switch (status) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        default: return "CL_UNKNOWN_ERROR";
    }
}

bool is_oom_status(cl_int status) noexcept {
    return status == CL_MEM_OBJECT_ALLOCATION_FAILURE ||
           status == CL_OUT_OF_RESOURCES ||
           status == CL_OUT_OF_HOST_MEMORY;
}

std::string format_bytes(std::size_t bytes) {
    char buf[64];
    if (bytes >= (1ULL << 20)) {
        std::snprintf(buf, sizeof(buf), "%.2f MiB (%zu bytes)",
                      static_cast<double>(bytes) / (1024.0 * 1024.0), bytes);
    } else if (bytes >= 1024ULL) {
        std::snprintf(buf, sizeof(buf), "%.2f KiB (%zu bytes)",
                      static_cast<double>(bytes) / 1024.0, bytes);
    } else {
        std::snprintf(buf, sizeof(buf), "%zu bytes", bytes);
    }
    return buf;
}

void check(cl_int status, std::string_view action) {
    if (status == CL_SUCCESS) {
        return;
    }
    if (is_oom_status(status)) {
        TORCH_CHECK_WITH(
            OutOfMemoryError, false,
            "Iris OpenCL ", action, " failed with out-of-memory (", cl_error_name(status), ", ", status,
            "). Try calling torch_iris.empty_cache() or reducing tensor/batch size.");
    }
    TORCH_CHECK(false, "Iris OpenCL ", action, " failed (", cl_error_name(status), ", ", status, ")");
}

template <typename Handle, auto Release>
struct ClDeleter {
    void operator()(Handle handle) const noexcept {
        (cl().*Release)(handle);
    }
};

template <typename Handle, auto Release>
using ClHandle = std::unique_ptr<std::remove_pointer_t<Handle>, ClDeleter<Handle, Release>>;

using Context = ClHandle<cl_context, &OpenCL::clReleaseContext>;
using Queue = ClHandle<cl_command_queue, &OpenCL::clReleaseCommandQueue>;
using Program = ClHandle<cl_program, &OpenCL::clReleaseProgram>;
using Kernel = ClHandle<cl_kernel, &OpenCL::clReleaseKernel>;
using Buffer = ClHandle<cl_mem, &OpenCL::clReleaseMemObject>;

using Event = ClHandle<cl_event, &OpenCL::clReleaseEvent>;

thread_local std::int64_t current_queue_id = 0;
constexpr std::size_t cache_limit = 128 * 1024 * 1024;

struct CacheKey {
    std::size_t bytes;
    std::int64_t queue;
    bool operator==(const CacheKey&) const = default;
};

struct CacheHash {
    std::size_t operator()(const CacheKey& key) const noexcept {
        return std::hash<std::size_t>{}(key.bytes) ^ std::hash<std::int64_t>{}(key.queue);
    }
};

struct ActiveBuffer {
    CacheKey key;
    bool cross_queue;
};

struct QueueState {
    Queue handle;
    Event query_marker;
    std::uint64_t generation = 0;
    std::uint64_t query_generation = 0;
    std::uint64_t completed_generation = 0;
};

struct PinnedBuffer {
    Buffer buffer;
    cl_command_queue queue = nullptr;
    void* address = nullptr;
    std::size_t bytes = 0;

    ~PinnedBuffer() {
        if (address) {
            cl().clEnqueueUnmapMemObject(queue, buffer.get(), address, 0, nullptr, nullptr);
            cl().clFlush(queue);
        }
    }
};

struct PendingTransfer {
    Event event;
    std::shared_ptr<PinnedBuffer> owner;
};

bool event_complete(cl_event event) {
    if (!event) {
        return true;
    }
    cl_int status = CL_QUEUED;
    check(cl().clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, nullptr),
          "query command status");
    TORCH_CHECK(status >= CL_COMPLETE, "Iris OpenCL command failed (", status, ")");
    return status == CL_COMPLETE;
}

struct Device {
    cl_device_id id;
    std::string name;
    bool fp16;
    bool reqd_sg8;
    bool subgroups;
    bool packed_signed_dot;
    cl_ulong max_allocation;
    cl_ulong total_memory;
    std::string driver_version;
};

Device find_device() {
    cl_uint platform_count = 0;
    const auto plat_status = cl().clGetPlatformIDs(0, nullptr, &platform_count);
    if (plat_status == -1001 || (plat_status == CL_SUCCESS && platform_count == 0)) {
        TORCH_CHECK(false,
                    "No Intel Iris Xe OpenCL GPU was found (no OpenCL platforms available). "
                    "Verify that an Intel Iris Xe GPU and an Intel graphics driver with OpenCL support are installed.");
    }
    check(plat_status, "enumerate platforms");
    std::vector<cl_platform_id> platforms(platform_count);
    check(cl().clGetPlatformIDs(platform_count, platforms.data(), nullptr), "list platforms");
    std::vector<std::string> discovered_gpus;
    for (auto platform : platforms) {
        cl_uint count = 0;
        const auto status = cl().clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
        if (status == CL_DEVICE_NOT_FOUND || (status == CL_SUCCESS && count == 0)) {
            continue;
        }
        check(status, "enumerate GPUs");
        std::vector<cl_device_id> devices(count);
        check(cl().clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr),
              "list GPUs");
        for (auto device : devices) {
            std::size_t size = 0;
            check(cl().clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size), "size GPU name");
            std::string name(size, '\0');
            check(cl().clGetDeviceInfo(device, CL_DEVICE_NAME, size, name.data(), nullptr),
                  "read GPU name");
            if (!name.empty() && name.back() == '\0') {
                name.pop_back();
            }
            discovered_gpus.push_back(name);
            auto lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower.find("intel") != std::string::npos &&
                lower.find("iris") != std::string::npos &&
                lower.find("xe") != std::string::npos) {
                cl_bool available = CL_FALSE;
                cl_bool compiler = CL_FALSE;
                check(cl().clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available),
                                           &available, nullptr), "check GPU availability");
                check(cl().clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler),
                                           &compiler, nullptr), "check GPU compiler");
                if (available && compiler) {
                    std::size_t extension_size = 0;
                    check(cl().clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &extension_size),
                          "query GPU extensions");
                    std::string extensions(extension_size, '\0');
                    check(cl().clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensions.size(),
                                               extensions.data(), nullptr), "read GPU extensions");
                    // CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR
                    // is absent from the bundled OpenCL 1.2 headers. The first field is signed_accelerated.
                    constexpr cl_device_info packed_dot_properties = 0x1075;
                    cl_bool dot_acceleration[6]{};
                    const bool packed_signed_dot = extensions.find("cl_khr_integer_dot_product") != std::string::npos &&
                        cl().clGetDeviceInfo(device, packed_dot_properties, sizeof(dot_acceleration),
                                             dot_acceleration, nullptr) == CL_SUCCESS && dot_acceleration[0];
                    cl_ulong max_alloc = 0;
                    cl_ulong total_mem = 0;
                    check(cl().clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                                               &max_alloc, nullptr), "read max_mem_alloc_size");
                    check(cl().clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(total_mem),
                                               &total_mem, nullptr), "read global_mem_size");
                    std::size_t drv_size = 0;
                    std::string drv_ver;
                    if (cl().clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &drv_size) == CL_SUCCESS &&
                        drv_size > 0) {
                        drv_ver.resize(drv_size, '\0');
                        if (cl().clGetDeviceInfo(device, CL_DRIVER_VERSION, drv_size, drv_ver.data(), nullptr) == CL_SUCCESS &&
                            !drv_ver.empty() && drv_ver.back() == '\0') {
                            drv_ver.pop_back();
                        }
                    }
                    return {device, std::move(name), extensions.find("cl_khr_fp16") != std::string::npos,
                            extensions.find("cl_intel_required_subgroup_size") != std::string::npos,
                            extensions.find("cl_intel_subgroups") != std::string::npos,
                            packed_signed_dot, max_alloc, total_mem, std::move(drv_ver)};
                }
            }
        }
    }
    std::string found_list;
    for (std::size_t i = 0; i < discovered_gpus.size(); ++i) {
        if (i > 0) found_list += ", ";
        found_list += discovered_gpus[i];
    }
    if (found_list.empty()) {
        found_list = "none";
    }
    TORCH_CHECK(false,
                "No available Intel Iris Xe OpenCL GPU with compiler support was found (discovered OpenCL GPUs: ",
                found_list, "). Install or update the Intel graphics driver with OpenCL support.");
}

const Device& device() {
    static const Device selected = find_device();
    return selected;
}

std::atomic<bool> initialized{false};
std::atomic<std::uint64_t> submissions{0};
std::atomic<std::uint64_t> tiled_matmul_submissions_count{0};
std::atomic<std::uint64_t> blocked_matmul_submissions_count{0};
constexpr std::size_t matmul_tile_size = 16;
constexpr std::size_t matmul_block_group_width = 8;
constexpr std::size_t matmul_block_group_height = 16;
constexpr std::size_t matmul_block_size = 64;
constexpr char packed_int8_kernel_name[] = "int8_matmul_packed";
constexpr std::size_t reduction_group_size = 64;

struct MatmulTile {
    const char* name;
    std::size_t group_width;
    std::size_t group_height;
    std::size_t output_width;
    std::size_t output_height;
    std::atomic<std::uint64_t>& submission_count;
    std::optional<bool> supported = std::nullopt;
};

class Runtime {
public:
    std::size_t external_buffer_size(cl_mem buffer) const {
        cl_context context = nullptr;
        cl_mem_object_type type = 0;
        cl_mem_flags flags = 0;
        std::size_t bytes = 0;
        check(cl().clGetMemObjectInfo(buffer, CL_MEM_CONTEXT, sizeof(context), &context, nullptr), "read DLPack buffer context");
        TORCH_CHECK(context == context_.get(), "DLPack OpenCL buffer must belong to the Iris context");
        check(cl().clGetMemObjectInfo(buffer, CL_MEM_TYPE, sizeof(type), &type, nullptr), "read DLPack buffer type");
        check(cl().clGetMemObjectInfo(buffer, CL_MEM_FLAGS, sizeof(flags), &flags, nullptr), "read DLPack buffer flags");
        TORCH_CHECK(type == CL_MEM_OBJECT_BUFFER && !(flags & (CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY)),
                    "Iris DLPack requires a read-write OpenCL buffer");
        check(cl().clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(bytes), &bytes, nullptr), "read DLPack buffer size");
        return bytes;
    }
    Runtime() {
        const auto gpu = device().id;
        cl_int status = CL_SUCCESS;
        context_.reset(cl().clCreateContext(nullptr, 1, &gpu, nullptr, nullptr, &status));
        check(status, "create context");
        queues_.push_back({make_queue()});
        const cl_int zero_flag = 0;
        error_flag_.reset(allocate(sizeof(cl_int)));
        check(cl().clEnqueueFillBuffer(queues_.front().handle.get(), error_flag_.get(),
                                       &zero_flag, sizeof(zero_flag), 0, sizeof(zero_flag),
                                       0, nullptr, nullptr), "init error flag");
        initialized.store(true, std::memory_order_release);
    }

    cl_mem error_flag_buffer() noexcept {
        pending_nll_error_check_ = true;
        return error_flag_.get();
    }

    void check_async_errors() {
        const std::lock_guard lock(mutex_);
        if (!pending_nll_error_check_) {
            return;
        }
        check_async_errors_locked(queue_at(current_queue_id).handle.get());
    }

    void check_allocation_limit(std::size_t bytes) const {
        const auto max_alloc = static_cast<std::size_t>(device().max_allocation);
        if (max_alloc > 0 && bytes > max_alloc) {
            TORCH_CHECK_WITH(
                OutOfMemoryError, false,
                "Iris allocation of ", format_bytes(bytes),
                " exceeds the OpenCL device's max_mem_alloc_size limit of ", format_bytes(max_alloc),
                " on ", device().name,
                ". Reduce tensor/batch size or split the workload.");
        }
    }

    void reclaim_cache_locked() noexcept {
        clear_cache_locked();
        for (auto& queue : queues_) {
            (void)cl().clFlush(queue.handle.get());
        }
        for (auto& queue : queues_) {
            (void)cl().clFinish(queue.handle.get());
            queue.completed_generation = queue.generation;
        }
        reap_transfers();
        ++stats_.cache_reclaims;
    }

    cl_mem allocate(std::size_t bytes, void* data = nullptr) const {
        check_allocation_limit(bytes);
        cl_int status = CL_SUCCESS;
        const cl_mem_flags flags = data ? CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
                                        : CL_MEM_READ_WRITE;
        const auto buffer = cl().clCreateBuffer(context_.get(), flags, bytes, data, &status);
        check(status, "allocate buffer");
        return buffer;
    }


    cl_mem allocate_tensor(std::size_t bytes) {
        check_allocation_limit(bytes);
        const std::lock_guard lock(mutex_);
        const CacheKey key{bytes, current_queue_id};
        Buffer buffer;
        if (const auto existing = cached_.find(key); existing != cached_.end()) {
            buffer = std::move(existing->second);
            cached_.erase(existing);
            stats_.cached_bytes -= bytes;
            ++stats_.reuses;
        } else {
            cl_int status = CL_SUCCESS;
            buffer.reset(cl().clCreateBuffer(context_.get(), CL_MEM_READ_WRITE, bytes, nullptr, &status));
            if (status != CL_SUCCESS && (is_oom_status(status) || status == CL_INVALID_BUFFER_SIZE)) {
                if (!cached_.empty() || !pending_.empty()) {
                    reclaim_cache_locked();
                    status = CL_SUCCESS;
                    buffer.reset(cl().clCreateBuffer(context_.get(), CL_MEM_READ_WRITE, bytes, nullptr, &status));
                }
            }
            if (status != CL_SUCCESS) {
                if (is_oom_status(status) || status == CL_INVALID_BUFFER_SIZE) {
                    TORCH_CHECK_WITH(
                        OutOfMemoryError, false,
                        "Iris OpenCL out of memory allocating ", format_bytes(bytes),
                        " on ", device().name, " (", cl_error_name(status), ", ", status,
                        "; allocated=", format_bytes(stats_.allocated_bytes),
                        ", cached=", format_bytes(stats_.cached_bytes),
                        ", total_memory=", format_bytes(static_cast<std::size_t>(device().total_memory)),
                        ", max_mem_alloc_size=", format_bytes(static_cast<std::size_t>(device().max_allocation)),
                        "). Try calling torch_iris.empty_cache() or reducing tensor/batch size.");
                }
                check(status, "allocate buffer");
            }
            ++stats_.allocations;
        }
        active_.emplace(buffer.get(), ActiveBuffer{key, false});
        stats_.allocated_bytes += bytes;
        stats_.peak_allocated_bytes = std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);
        stats_.peak_reserved_bytes = std::max(stats_.peak_reserved_bytes, stats_.allocated_bytes + stats_.cached_bytes);
        return buffer.release();
    }

    void release_tensor(cl_mem mem) noexcept {
        Buffer buffer(mem);
        const std::lock_guard lock(mutex_);
        const auto active = active_.find(mem);
        if (active == active_.end()) {
            return;
        }
        const auto entry = active->second;
        active_.erase(active);
        stats_.allocated_bytes -= entry.key.bytes;
        // Reuse on the allocating queue is ordered after all earlier commands there.
        // Cross-queue buffers are released to OpenCL, which retains command references.
        if (!entry.cross_queue && entry.key.bytes <= cache_limit - stats_.cached_bytes) {
            try {
                cached_.emplace(entry.key, std::move(buffer));
                stats_.cached_bytes += entry.key.bytes;
            } catch (...) {
                // Let the reference go if caching cannot reserve a map entry.
            }
        }
    }

    void record_buffer(cl_mem mem, std::int64_t queue_id) {
        const std::lock_guard lock(mutex_);
        note_buffer_locked(mem, queue_id);
    }

    void clear_cache() {
        const std::lock_guard lock(mutex_);
        clear_cache_locked();
    }

    MemoryStats memory_stats() {
        const std::lock_guard lock(mutex_);
        return stats_;
    }

    void reset_peak_memory_stats() {
        const std::lock_guard lock(mutex_);
        stats_.peak_allocated_bytes = stats_.allocated_bytes;
        stats_.peak_reserved_bytes = stats_.allocated_bytes + stats_.cached_bytes;
    }

    std::int64_t new_queue() {
        const std::lock_guard lock(mutex_);
        queues_.push_back({make_queue()});
        return static_cast<std::int64_t>(queues_.size() - 1);
    }

    void check_queue(std::int64_t id) {
        const std::lock_guard lock(mutex_);
        (void)queue_at(id);
    }

    void synchronize_queue(std::int64_t id) {
        const std::lock_guard lock(mutex_);
        auto& queue = queue_at(id);
        check(cl().clFinish(queue.handle.get()), "finish queue");
        queue.completed_generation = queue.generation;
        reap_transfers();
        if (pending_nll_error_check_) {
            check_async_errors_locked(queue.handle.get());
        }
    }

    void synchronize_all() {
        const std::lock_guard lock(mutex_);
        for (const auto& queue : queues_) {
            check(cl().clFlush(queue.handle.get()), "flush queue");
        }
        for (auto& queue : queues_) {
            check(cl().clFinish(queue.handle.get()), "finish queue");
            queue.completed_generation = queue.generation;
        }
        reap_transfers();
        if (pending_nll_error_check_) {
            check_async_errors_locked(queues_.front().handle.get());
        }
    }

    bool query_queue(std::int64_t id) {
        const std::lock_guard lock(mutex_);
        auto& queue = queue_at(id);
        if (queue.completed_generation == queue.generation) {
            return true;
        }
        if (!queue.query_marker || queue.query_generation != queue.generation) {
            queue.query_marker = enqueue_marker(queue.handle.get());
            queue.query_generation = queue.generation;
        }
        if (!event_complete(queue.query_marker.get())) {
            return false;
        }
        queue.completed_generation = queue.query_generation;
        return true;
    }

    cl_event record(std::int64_t id) {
        const std::lock_guard lock(mutex_);
        auto& queue = queue_at(id);
        auto event = enqueue_marker(queue.handle.get());
        ++queue.generation;
        return event.release();
    }

    void block(std::int64_t id, cl_event event) {
        if (event) {
            const std::lock_guard lock(mutex_);
            auto& queue = queue_at(id);
            check(cl().clEnqueueBarrierWithWaitList(queue.handle.get(), 1, &event, nullptr),
                  "wait for event on queue");
            ++queue.generation;
        }
    }

    void* pin(std::size_t bytes) {
        check_allocation_limit(bytes);
        const std::lock_guard lock(mutex_);
        auto block = std::make_shared<PinnedBuffer>();
        block->bytes = bytes;
        block->queue = queues_.front().handle.get();
        cl_int status = CL_SUCCESS;
        block->buffer.reset(cl().clCreateBuffer(context_.get(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                               bytes, nullptr, &status));
        if (status != CL_SUCCESS && is_oom_status(status) && (!cached_.empty() || !pending_.empty())) {
            reclaim_cache_locked();
            status = CL_SUCCESS;
            block->buffer.reset(cl().clCreateBuffer(context_.get(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                   bytes, nullptr, &status));
        }
        check(status, "allocate pinned host memory");
        block->address = cl().clEnqueueMapBuffer(block->queue, block->buffer.get(), CL_TRUE,
                                                CL_MAP_READ | CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &status);
        check(status, "map pinned host memory");
        auto addr = block->address;
        pinned_.emplace(reinterpret_cast<std::uintptr_t>(addr), std::move(block));
        return addr;
    }

    void unpin(void* address) noexcept {
        const std::lock_guard lock(mutex_);
        pinned_.erase(reinterpret_cast<std::uintptr_t>(address));
    }

    bool is_pinned(const void* address) {
        const std::lock_guard lock(mutex_);
        return static_cast<bool>(find_pinned(address, 1));
    }

    template <typename Transfer>
    void transfer(const void* host, std::size_t bytes, bool non_blocking, cl_mem tensor,
                  std::string_view action, Transfer copy) {
        const std::lock_guard lock(mutex_);
        auto& queue = queue_at(current_queue_id);
        note_buffer_locked(tensor, current_queue_id);
        reap_transfers();
        const auto pinned = non_blocking ? find_pinned(host, bytes) : nullptr;
        if (pinned) {
            // Reserve ownership before submission so a C++ allocation failure cannot
            // leave an outstanding transfer referring to freed host memory.
            pending_.push_back({{}, pinned});
            cl_event event = nullptr;
            const auto status = copy(queue.handle.get(), CL_FALSE, &event);
            if (status != CL_SUCCESS) {
                pending_.pop_back();
                check(status, action);
            }
            pending_.back().event.reset(event);
            check(cl().clFlush(queue.handle.get()), "flush async transfer");
        } else {
            check(copy(queue.handle.get(), CL_TRUE, nullptr), action);
            if (pending_nll_error_check_ && action == "download buffer") {
                check_async_errors_locked(queue.handle.get());
            }
        }
        ++queue.generation;
    }

    template <typename... Args>
    void run_with_geometry(const char* name, std::size_t size, const std::size_t* local,
                           std::span<const cl_ulong> geometry, const Args&... args) {
        const cl_ulong unused = 0;
        if (geometry.empty()) {
            geometry = {&unused, 1};
        }
        Buffer metadata(allocate(geometry.size_bytes(), const_cast<cl_ulong*>(geometry.data())));
        submit(name, [&](cl_command_queue queue) {
            const std::size_t global[] = {size};
            return enqueue(queue, name, global, local, metadata.get(), args...);
        });
    }

    template <typename Command>
    void submit(std::string_view action, Command command, std::initializer_list<cl_mem> buffers = {}) {
        const std::lock_guard lock(mutex_);
        auto& queue = queue_at(current_queue_id);
        for (const auto buffer : buffers) {
            note_buffer_locked(buffer, current_queue_id);
        }
        check(command(queue.handle.get()), action);
        ++queue.generation;
        reap_transfers();
    }

    template <std::size_t Rank, typename... Args>
    void run(const char* name, const std::size_t (&size)[Rank], const Args&... args) {
        submit(name, [&](cl_command_queue queue) {
            return enqueue(queue, name, size, nullptr, args...);
        });
    }

    template <std::size_t Rank, typename... Args>
    void run_local(const char* name, const std::size_t (&size)[Rank], const std::size_t (&local)[Rank],
                   const Args&... args) {
        submit(name, [&](cl_command_queue queue) {
            return enqueue(queue, name, size, local, args...);
        });
    }

    void run_matmul(cl_mem output, MatrixView left, MatrixView right, cl_ulong batches,
                    cl_ulong rows, cl_ulong columns, cl_ulong inner,
                    MatrixView bias, float alpha, float beta, bool fp16) {
        submit("matmul", [&](cl_command_queue queue) {
            if (batches == 1 && rows <= 16 && inner >= 64) {
                const cl_uint is_half = fp16 ? 1u : 0u;
                if (device().subgroups && device().reqd_sg8 && rows == 1 && left.column_stride == 1 &&
                    right.row_stride == 1 && columns >= 256) {
                    const std::size_t col_groups = ((columns + 3) / 4 + 7) / 8;
                    const std::size_t size[] = {col_groups * 16, 8, 1};
                    constexpr std::size_t local[] = {16, 8, 1};
                    const auto status = enqueue(queue, "matmul_gemv_sg16_nt_m1", size, local,
                                                output, left.buffer, right.buffer, bias.buffer,
                                                rows, columns, inner,
                                                left.offset, left.batch_stride, left.row_stride, left.column_stride,
                                                right.offset, right.batch_stride, right.row_stride, right.column_stride,
                                                bias.offset, bias.row_stride, bias.column_stride, alpha, beta, is_half);
                    if (status == CL_SUCCESS) {
                        tiled_matmul_submissions_count.fetch_add(1, std::memory_order_relaxed);
                    }
                    return status;
                }
                if (device().subgroups && device().reqd_sg8 && rows <= 8 && left.column_stride == 1 &&
                    right.row_stride == 1 && (rows * columns > 256 || columns >= 256)) {
                    const std::size_t col_groups = (columns + 7) / 8;
                    const std::size_t size[] = {col_groups * 16, 8, 1};
                    constexpr std::size_t local[] = {16, 8, 1};
                    const auto status = enqueue(queue, "matmul_gemv_sg16_nt", size, local,
                                                output, left.buffer, right.buffer, bias.buffer,
                                                rows, columns, inner,
                                                left.offset, left.batch_stride, left.row_stride, left.column_stride,
                                                right.offset, right.batch_stride, right.row_stride, right.column_stride,
                                                bias.offset, bias.row_stride, bias.column_stride, alpha, beta, is_half);
                    if (status == CL_SUCCESS) {
                        tiled_matmul_submissions_count.fetch_add(1, std::memory_order_relaxed);
                    }
                    return status;
                }
                if (device().subgroups && device().reqd_sg8 && rows <= 8 && left.column_stride == 1 &&
                    right.column_stride == 1 && columns >= 64) {
                    const std::size_t col_groups = (columns + 15) / 16;
                    const std::size_t size[] = {col_groups * 16, 8, 1};
                    constexpr std::size_t local[] = {16, 8, 1};
                    const auto status = enqueue(queue, "matmul_gemv_sg16_nn", size, local,
                                                output, left.buffer, right.buffer, bias.buffer,
                                                rows, columns, inner,
                                                left.offset, left.batch_stride, left.row_stride, left.column_stride,
                                                right.offset, right.batch_stride, right.row_stride, right.column_stride,
                                                bias.offset, bias.row_stride, bias.column_stride, alpha, beta, is_half);
                    if (status == CL_SUCCESS) {
                        tiled_matmul_submissions_count.fetch_add(1, std::memory_order_relaxed);
                    }
                    return status;
                }
                if (inner >= 256 && columns <= 4096) {
                    if (rows * columns <= 256) {
                        const std::size_t size[] = {columns * 128, rows, 1};
                        constexpr std::size_t local[] = {128, 1, 1};
                        const auto status = enqueue(queue, "matmul_skinny_wg", size, local,
                                                    output, left.buffer, right.buffer, bias.buffer,
                                                    rows, columns, inner,
                                                    left.offset, left.batch_stride, left.row_stride, left.column_stride,
                                                    right.offset, right.batch_stride, right.row_stride, right.column_stride,
                                                    bias.offset, bias.row_stride, bias.column_stride, alpha, beta, is_half);
                        if (status == CL_SUCCESS) {
                            tiled_matmul_submissions_count.fetch_add(1, std::memory_order_relaxed);
                        }
                        return status;
                    } else if (device().subgroups && device().reqd_sg8) {
                        const std::size_t col_groups = (columns + 7) / 8;
                        const std::size_t size[] = {col_groups * 16, rows * 8, 1};
                        constexpr std::size_t local[] = {16, 8, 1};
                        const auto status = enqueue(queue, "matmul_skinny_sg16", size, local,
                                                    output, left.buffer, right.buffer, bias.buffer,
                                                    rows, columns, inner,
                                                    left.offset, left.batch_stride, left.row_stride, left.column_stride,
                                                    right.offset, right.batch_stride, right.row_stride, right.column_stride,
                                                    bias.offset, bias.row_stride, bias.column_stride, alpha, beta, is_half);
                        if (status == CL_SUCCESS) {
                            tiled_matmul_submissions_count.fetch_add(1, std::memory_order_relaxed);
                        }
                        return status;
                    }
                }
            }
            const auto* tile = select_matmul_tile(left, right, batches, rows, columns, inner, fp16);
            const auto launch = [&](const char* name, const std::size_t (&size)[3],
                                    const std::size_t* local) {
                return enqueue(queue, name, size, local,
                               output, left.buffer, right.buffer, bias.buffer,
                               rows, columns, inner,
                               left.offset, left.batch_stride, left.row_stride, left.column_stride,
                               right.offset, right.batch_stride, right.row_stride, right.column_stride,
                               bias.offset, bias.row_stride, bias.column_stride, alpha, beta);
            };
            if (tile) {
                const std::size_t local[] = {tile->group_width, tile->group_height, 1};
                const auto global_size = [](std::size_t n, std::size_t tile_span, std::size_t local_size) {
                    return ((n - 1) / tile_span + 1) * local_size;
                };
                const std::size_t size[] = {global_size(columns, tile->output_width, local[0]),
                                            global_size(rows, tile->output_height, local[1]), batches};
                const auto status = launch(tile->name, size, local);
                if (status == CL_SUCCESS) {
                    tile->submission_count.fetch_add(1, std::memory_order_relaxed);
                }
                return status;
            }
            const std::size_t size[] = {columns, rows, batches};
            return launch(fp16 ? "matmul_half" : "matmul", size, nullptr);
        });
    }

    bool supports_packed_int8_matmul() {
        if (!device().packed_signed_dot) return false;
        const std::lock_guard lock(mutex_);
        if (!packed_int8_supported_.has_value()) {
            packed_int8_supported_ = build_packed_int8_kernel();
        }
        return *packed_int8_supported_;
    }

private:

    bool build_packed_int8_kernel() {
        const auto gpu = device().id;
        cl_int status = CL_SUCCESS;
        const char* source = int8_packed_kernels;
        Program program(cl().clCreateProgramWithSource(context_.get(), 1, &source, nullptr, &status));
        check(status, "create packed INT8 program");
        status = cl().clBuildProgram(program.get(), 1, &gpu, "-cl-std=CL3.0", nullptr, nullptr);
        if (is_oom_status(status)) check(status, "build packed INT8 program");
        if (status != CL_SUCCESS) return false;
        Kernel function(cl().clCreateKernel(program.get(), packed_int8_kernel_name, &status));
        check(status, "create packed INT8 kernel");
        std::size_t group_size = 0;
        if (cl().clGetKernelWorkGroupInfo(function.get(), gpu, CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof(group_size), &group_size, nullptr) != CL_SUCCESS || group_size < 64) {
            return false;
        }
        kernels_.emplace(packed_int8_kernel_name, std::move(function));
        return true;
    }

    Queue make_queue() const {
        cl_int status = CL_SUCCESS;
        Queue queue(cl().clCreateCommandQueue(context_.get(), device().id, CL_QUEUE_PROFILING_ENABLE, &status));
        check(status, "create command queue");
        return queue;
    }

    QueueState& queue_at(std::int64_t id) {
        TORCH_CHECK(id >= 0 && static_cast<std::size_t>(id) < queues_.size(), "Unknown Iris stream ", id);
        return queues_[id];
    }

    Event enqueue_marker(cl_command_queue queue) const {
        cl_event event = nullptr;
        check(cl().clEnqueueMarkerWithWaitList(queue, 0, nullptr, &event), "record OpenCL event");
        Event result(event);
        check(cl().clFlush(queue), "flush event");
        return result;
    }

    void clear_cache_locked() {
        cached_.clear();
        stats_.cached_bytes = 0;
    }

    void note_buffer_locked(cl_mem buffer, std::int64_t queue) {
        if (queues_.size() == 1 || !buffer) {
            return;
        }
        if (const auto active = active_.find(buffer); active != active_.end() && active->second.key.queue != queue) {
            active->second.cross_queue = true;
        }
    }

    template <typename T>
    void note_kernel_argument(const T& argument) {
        if constexpr (std::is_same_v<T, cl_mem>) {
            note_buffer_locked(argument, current_queue_id);
        }
    }

    std::shared_ptr<PinnedBuffer> find_pinned(const void* pointer, std::size_t bytes) const {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        auto after = pinned_.upper_bound(address);
        if (after == pinned_.begin()) {
            return {};
        }
        const auto& [base, block] = *std::prev(after);
        const auto offset = address - base;
        return offset < block->bytes && bytes <= block->bytes - offset ? block : nullptr;
    }

    void reap_transfers() {
        std::erase_if(pending_, [](const PendingTransfer& transfer) {
            cl_int status = CL_QUEUED;
            return cl().clGetEventInfo(transfer.event.get(), CL_EVENT_COMMAND_EXECUTION_STATUS,
                                        sizeof(status), &status, nullptr) == CL_SUCCESS && status <= CL_COMPLETE;
        });
    }

    template <std::size_t Rank, typename... Args>
    cl_int enqueue(cl_command_queue queue, const char* name, const std::size_t (&size)[Rank],
                   const std::size_t* local, const Args&... args) {
        const auto function = kernel(name);
        cl_uint index = 0;
        (note_kernel_argument(args), ...);
        (check(cl().clSetKernelArg(function, index++, sizeof(args), &args), name), ...);
        const auto status = cl().clEnqueueNDRangeKernel(
            queue, function, static_cast<cl_uint>(Rank), nullptr, size, local,
            0, nullptr, nullptr);
        if (status == CL_SUCCESS) {
            submissions.fetch_add(1, std::memory_order_relaxed);
            return status;
        }
        std::string dims = "global=[";
        for (std::size_t i = 0; i < Rank; ++i) {
            if (i > 0) dims += ",";
            dims += std::to_string(size[i]);
        }
        dims += "]";
        if (local) {
            dims += " local=[";
            for (std::size_t i = 0; i < Rank; ++i) {
                if (i > 0) dims += ",";
                dims += std::to_string(local[i]);
            }
            dims += "]";
        }
        if (is_oom_status(status)) {
            TORCH_CHECK_WITH(
                OutOfMemoryError, false,
                "Iris OpenCL enqueue kernel '", name, "' failed with out-of-memory (",
                cl_error_name(status), ", ", status, "; ", dims,
                "). Try calling torch_iris.empty_cache() or reducing tensor/batch size.");
        }
        TORCH_CHECK(false, "Iris OpenCL enqueue kernel '", name, "' failed (",
                    cl_error_name(status), ", ", status, "; ", dims, ")");
    }

    const MatmulTile* select_matmul_tile(MatrixView left, MatrixView right, cl_ulong batches,
                                         cl_ulong rows, cl_ulong columns, cl_ulong inner, bool fp16) {
        // Iris Xe event timings favor register blocking for large outputs with enough K
        // reuse, and the smaller 16x16 tile for narrower outputs.
        const cl_ulong minimum_width = batches >= 4 ? 128 : 256;
        auto& aligned = fp16 ? aligned_half_matmul_ : aligned_matmul_;
        auto& blocked = fp16 ? blocked_half_matmul_ : blocked_matmul_;
        auto& tiled = fp16 ? tiled_half_matmul_ : tiled_matmul_;
        auto& sg8 = fp16 ? sg8_half_matmul_ : sg8_matmul_;
        const bool blocked_eligible =
            (rows >= minimum_width && columns >= minimum_width && inner >= 128) ||
            (rows >= 128 && columns >= 128 && inner >= 256 && (rows >= 512 || columns >= 512 || inner >= 512)) ||
            (batches >= 2 && rows >= 64 && columns >= 256 && inner >= 32) ||
            (rows >= 64 && columns >= 512 && inner >= 32);
        if (blocked_eligible) {
            const bool unit_strides =
                (left.row_stride == 1 || left.column_stride == 1) &&
                (right.row_stride == 1 || right.column_stride == 1);
            if ((!fp16 || device().fp16) && device().reqd_sg8 && unit_strides &&
                rows % 128 == 0 && columns % matmul_block_size == 0 && inner % matmul_tile_size == 0 &&
                inner >= 64 && batches * (rows / 128) * (columns / matmul_block_size) >= 16 &&
                supports_matmul_tile(sg8)) {
                return &sg8;
            }
            const bool full_tiles = rows % matmul_block_size == 0 &&
                columns % matmul_block_size == 0 && inner % matmul_tile_size == 0 && unit_strides;
            if (full_tiles && supports_matmul_tile(aligned)) {
                return &aligned;
            }
            if (supports_matmul_tile(blocked)) {
                return &blocked;
            }
        }
        constexpr cl_ulong minimum_tiled_work = 512;
        const bool enough_work = inner >= 8 && rows * columns > (minimum_tiled_work - 1) / inner;
        return enough_work && supports_matmul_tile(tiled) ? &tiled : nullptr;
    }

    bool supports_matmul_tile(MatmulTile& tile) {
        // Called under the queue lock, like the kernel cache.
        if (!tile.supported.has_value()) {
            tile.supported = check_matmul_tile_limits(tile);
        }
        return *tile.supported;
    }

    bool check_matmul_tile_limits(const MatmulTile& tile) {
        const auto gpu = device().id;
        const auto group_size = tile.group_width * tile.group_height;
        std::size_t device_group_size = 0;
        cl_uint dimensions = 0;
        if (cl().clGetDeviceInfo(gpu, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(device_group_size), &device_group_size, nullptr) != CL_SUCCESS ||
            device_group_size < group_size ||
            cl().clGetDeviceInfo(gpu, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                 sizeof(dimensions), &dimensions, nullptr) != CL_SUCCESS ||
            dimensions < 3) {
            return false;
        }
        std::vector<std::size_t> item_sizes(dimensions);
        if (cl().clGetDeviceInfo(gpu, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                 item_sizes.size() * sizeof(item_sizes[0]),
                                 item_sizes.data(), nullptr) != CL_SUCCESS ||
            item_sizes[0] < tile.group_width || item_sizes[1] < tile.group_height) {
            return false;
        }
        const auto function = kernel(tile.name);
        std::size_t kernel_group_size = 0;
        cl_ulong kernel_local_bytes = 0;
        cl_ulong device_local_bytes = 0;
        return cl().clGetKernelWorkGroupInfo(function, gpu, CL_KERNEL_WORK_GROUP_SIZE,
                   sizeof(kernel_group_size), &kernel_group_size, nullptr) == CL_SUCCESS &&
            kernel_group_size >= group_size &&
            cl().clGetKernelWorkGroupInfo(function, gpu, CL_KERNEL_LOCAL_MEM_SIZE,
                   sizeof(kernel_local_bytes), &kernel_local_bytes, nullptr) == CL_SUCCESS &&
            cl().clGetDeviceInfo(gpu, CL_DEVICE_LOCAL_MEM_SIZE,
                   sizeof(device_local_bytes), &device_local_bytes, nullptr) == CL_SUCCESS &&
            kernel_local_bytes <= device_local_bytes;
    }

    cl_kernel kernel(const char* name) {
        if (const auto ptr_hit = kernel_ptr_cache_.find(name); ptr_hit != kernel_ptr_cache_.end()) {
            return ptr_hit->second;
        }
        if (const auto existing = kernels_.find(name); existing != kernels_.end()) {
            kernel_ptr_cache_.emplace(name, existing->second.get());
            return existing->second.get();
        }
        if (!program_) {
            std::vector<const char*> sources(kernel_sources.begin(), kernel_sources.end());
            sources.push_back(spatial_kernels);
            sources.insert(sources.end(), attention_kernels.begin(), attention_kernels.end());
            sources.insert(sources.end(), conv_fast_kernels.begin(), conv_fast_kernels.end());
            sources.insert(sources.end(), ops_extra_kernels.begin(), ops_extra_kernels.end());
            sources.insert(sources.end(), ops_phase3_kernels.begin(), ops_phase3_kernels.end());
            cl_int status = CL_SUCCESS;
            Program program(cl().clCreateProgramWithSource(
                context_.get(), static_cast<cl_uint>(sources.size()), sources.data(), nullptr, &status));
            check(status, "create program");
            const auto gpu = device().id;
            auto options = "-cl-std=CL1.2 -DMATMUL_TILE_SIZE=" + std::to_string(matmul_tile_size) +
                " -DMATMUL_BLOCK_GROUP_X=" + std::to_string(matmul_block_group_width) +
                " -DMATMUL_BLOCK_GROUP_Y=" + std::to_string(matmul_block_group_height) +
                " -DMATMUL_BLOCK_SIZE=" + std::to_string(matmul_block_size);
            if (device().fp16) {
                options += " -DIRIS_HAS_FP16=1";
            }
            if (device().reqd_sg8) {
                options += " -DIRIS_HAS_REQD_SG8=1";
            }
            if (device().subgroups) {
                options += " -DIRIS_HAS_SUBGROUPS=1";
            }
            status = cl().clBuildProgram(program.get(), 1, &gpu, options.c_str(), nullptr, nullptr);
            if (status != CL_SUCCESS) {
                std::size_t size = 0;
                cl().clGetProgramBuildInfo(program.get(), gpu, CL_PROGRAM_BUILD_LOG,
                                          0, nullptr, &size);
                std::string log(size, '\0');
                if (size > 0) {
                    cl().clGetProgramBuildInfo(program.get(), gpu, CL_PROGRAM_BUILD_LOG,
                                              size, log.data(), nullptr);
                    if (log.back() == '\0') {
                        log.pop_back();
                    }
                }
                TORCH_CHECK(false, "Iris OpenCL kernel build failed (", cl_error_name(status), ", ", status,
                            ") on ", device().name, " (driver ", device().driver_version, "): ", log);
            }
            program_ = std::move(program);
        }
        cl_int status = CL_SUCCESS;
        Kernel function(cl().clCreateKernel(program_.get(), name, &status));
        if (status != CL_SUCCESS) {
            TORCH_CHECK(false, "Iris OpenCL create kernel '", name, "' failed (",
                        cl_error_name(status), ", ", status, ")");
        }
        return kernels_.emplace(name, std::move(function)).first->second.get();
    }

    void check_async_errors_locked(cl_command_queue queue) {
        pending_nll_error_check_ = false;
        cl_int flag = 0;
        check(cl().clEnqueueReadBuffer(queue, error_flag_.get(), CL_TRUE, 0, sizeof(flag), &flag,
                                       0, nullptr, nullptr), "read error flag");
        if (flag != 0) {
            const cl_int zero_flag = 0;
            check(cl().clEnqueueFillBuffer(queue, error_flag_.get(), &zero_flag, sizeof(zero_flag),
                                           0, sizeof(zero_flag), 0, nullptr, nullptr), "reset error flag");
            check(cl().clFinish(queue), "finish error flag reset");
            TORCH_CHECK_INDEX(false, "Iris nll_loss target is out of bounds");
        }
    }

    Context context_;
    std::vector<QueueState> queues_;
    Buffer error_flag_;
    bool pending_nll_error_check_ = false;
    std::unordered_map<cl_mem, ActiveBuffer> active_;
    std::unordered_multimap<CacheKey, Buffer, CacheHash> cached_;
    std::map<std::uintptr_t, std::shared_ptr<PinnedBuffer>> pinned_;
    std::vector<PendingTransfer> pending_;
    MemoryStats stats_;
    Program program_;
    std::optional<bool> packed_int8_supported_;
    MatmulTile tiled_matmul_{
        "matmul_tiled", matmul_tile_size, matmul_tile_size, matmul_tile_size, matmul_tile_size,
        tiled_matmul_submissions_count};
    MatmulTile blocked_matmul_{
        "matmul_blocked", matmul_block_group_width, matmul_block_group_height, matmul_block_size, matmul_block_size,
        blocked_matmul_submissions_count};
    MatmulTile aligned_matmul_{
        "matmul_blocked_aligned", matmul_block_group_width, matmul_block_group_height,
        matmul_block_size, matmul_block_size, blocked_matmul_submissions_count};
    MatmulTile sg8_matmul_{
        "matmul_blocked_sg8_aligned", matmul_block_group_width, matmul_block_group_height,
        matmul_block_size, 128, blocked_matmul_submissions_count};
    MatmulTile tiled_half_matmul_{
        "matmul_tiled_half", matmul_tile_size, matmul_tile_size, matmul_tile_size, matmul_tile_size,
        tiled_matmul_submissions_count};
    MatmulTile blocked_half_matmul_{
        "matmul_blocked_half", matmul_block_group_width, matmul_block_group_height, matmul_block_size, matmul_block_size,
        blocked_matmul_submissions_count};
    MatmulTile aligned_half_matmul_{
        "matmul_blocked_aligned_half", matmul_block_group_width, matmul_block_group_height,
        matmul_block_size, matmul_block_size, blocked_matmul_submissions_count};
    MatmulTile sg8_half_matmul_{
        "matmul_blocked_sg8_aligned_half", matmul_block_group_width, matmul_block_group_height,
        matmul_block_size, 128, blocked_matmul_submissions_count};
    std::unordered_map<std::string_view, Kernel> kernels_;
    std::unordered_map<const char*, cl_kernel> kernel_ptr_cache_;
    std::mutex mutex_;
};

Runtime& runtime() {
    // DataPtr deleters can run during static destruction, after the caller's module exits.
    static auto* state = new Runtime();
    return *state;
}

}  // namespace

bool gpu_available() noexcept {
    try {
        return device().id != nullptr;
    } catch (...) {
        return false;
    }
}

bool has_fp16() noexcept {
    try {
        return device().fp16;
    } catch (...) {
        return false;
    }
}

bool has_subgroups() noexcept {
    try {
        return device().subgroups && device().reqd_sg8;
    } catch (...) {
        return false;
    }
}

bool context_initialized() noexcept {
    return initialized.load(std::memory_order_acquire);
}

const std::string& gpu_name() {
    return device().name;
}

void initialize() {
    (void)runtime();
}

cl_mem allocate(std::size_t bytes) {
    return bytes ? runtime().allocate_tensor(bytes) : nullptr;
}

void release(void* buffer) noexcept {
    if (buffer) {
        runtime().release_tensor(static_cast<cl_mem>(buffer));
    }
}

void upload(cl_mem destination, const void* source, std::size_t bytes, std::size_t offset,
            bool non_blocking) {
    runtime().transfer(source, bytes, non_blocking, destination, "upload buffer",
        [&](cl_command_queue queue, cl_bool blocking, cl_event* event) {
            return cl().clEnqueueWriteBuffer(queue, destination, blocking, offset, bytes, source,
                                            0, nullptr, event);
        });
}

void download(void* destination, cl_mem source, std::size_t bytes, std::size_t offset,
              bool non_blocking) {
    runtime().transfer(destination, bytes, non_blocking, source, "download buffer",
        [&](cl_command_queue queue, cl_bool blocking, cl_event* event) {
            return cl().clEnqueueReadBuffer(queue, source, blocking, offset, bytes, destination,
                                           0, nullptr, event);
        });
}

void copy(cl_mem destination, cl_mem source, std::size_t bytes,
          std::size_t destination_offset, std::size_t source_offset) {
    if (!bytes || (destination == source && destination_offset == source_offset)) {
        return;
    }
    runtime().submit("copy buffer", [&](cl_command_queue queue) {
        return cl().clEnqueueCopyBuffer(queue, source, destination, source_offset, destination_offset, bytes,
                                        0, nullptr, nullptr);
    }, {destination, source});
}

void fill(cl_mem destination, std::size_t length, std::size_t offset, float value) {
    fill_bytes(destination, length, offset, &value, sizeof(value));
}

void fill_bytes(cl_mem destination, std::size_t length, std::size_t offset,
                const void* value, std::size_t item_size) {
    if (length) {
        runtime().submit("fill buffer", [&](cl_command_queue queue) {
            return cl().clEnqueueFillBuffer(queue, destination, value, item_size,
                                            offset * item_size, length * item_size,
                                            0, nullptr, nullptr);
        }, {destination});
    }
}

void elementwise(Elementwise op, cl_mem output, cl_mem left, cl_mem right,
                 std::size_t length, float alpha) {
    if (!length) {
        return;
    }
    const std::size_t vec_count = (length + 3) / 4;
    const cl_ulong count = static_cast<cl_ulong>(length);
    const auto dispatch = [&](const char* name, auto&&... args) {
        if (vec_count >= 256) {
            constexpr std::size_t local[] = {256};
            const std::size_t size[] = {((vec_count + 255) / 256) * 256};
            runtime().run_local(name, size, local, args...);
        } else {
            const std::size_t size[] = {vec_count};
            runtime().run(name, size, args...);
        }
    };
    switch (op) {
        case Elementwise::Add:
            dispatch("add_tensor", output, left, right, count, alpha);
            break;
        case Elementwise::Multiply:
            dispatch("multiply_tensor", output, left, right, count);
            break;
        case Elementwise::Relu:
            dispatch("relu_tensor", output, left, count);
            break;
        case Elementwise::Scale:
            dispatch("scale_tensor", output, left, count, alpha);
            break;
        case Elementwise::AddScalar:
            dispatch("add_scalar_tensor", output, left, count, alpha);
            break;
        case Elementwise::ThresholdBackward:
            dispatch("threshold_backward_tensor", output, left, right, count, alpha);
            break;
        case Elementwise::SquaredDifference:
            dispatch("squared_difference_tensor", output, left, right, count);
            break;
        case Elementwise::ScaledDifference:
            dispatch("scaled_difference_tensor", output, left, right, count, alpha);
            break;
        default:
            TORCH_INTERNAL_ASSERT(false, "Invalid Iris contiguous elementwise operation");
    }
}

struct Geom24 {
    cl_ulong v[24];
};

Geom24 pack_geom24(std::span<const cl_ulong> geometry) {
    Geom24 g{};
    const std::size_t n = std::min<std::size_t>(geometry.size(), 24);
    for (std::size_t i = 0; i < n; ++i) g.v[i] = geometry[i];
    return g;
}

void recurrent_forward(BufferSlice output, BufferSlice new_cell, BufferSlice workspace,
                       BufferSlice input, BufferSlice hidden, BufferSlice state,
                       BufferSlice input_bias, BufferSlice hidden_bias,
                       cl_ulong length, cl_ulong width, bool lstm, bool fp16) {
    const std::size_t size[] = {static_cast<std::size_t>(length)};
    runtime().run("recurrent_forward", size, output.buffer, new_cell.buffer, workspace.buffer,
                  input.buffer, hidden.buffer, state.buffer, input_bias.buffer, hidden_bias.buffer,
                  input.offset, hidden.offset, state.offset, input_bias.offset, hidden_bias.offset,
                  width, static_cast<cl_uint>(lstm), static_cast<cl_uint>(fp16));
}

void recurrent_backward(BufferSlice input_gates, BufferSlice hidden_gates, BufferSlice grad_state,
                        BufferSlice grad_output, BufferSlice grad_cell, BufferSlice state,
                        BufferSlice new_cell, BufferSlice workspace,
                        cl_ulong length, cl_ulong width, bool lstm, bool fp16) {
    const std::size_t size[] = {static_cast<std::size_t>(length)};
    runtime().run("recurrent_backward", size, input_gates.buffer, hidden_gates.buffer, grad_state.buffer,
                  grad_output.buffer, grad_cell.buffer, state.buffer, new_cell.buffer, workspace.buffer,
                  grad_output.offset, grad_cell.offset, state.offset, new_cell.offset, workspace.offset,
                  width, static_cast<cl_uint>(lstm), static_cast<cl_uint>(fp16));
}

void add_scalar_batch(const std::array<BufferSlice, 8>& batch, std::size_t count, float scalar) {
    Geom24 offsets{};
    for (std::size_t i = 0; i < count; ++i) offsets.v[i] = batch[i].offset;
    const std::size_t size[] = {count};
    runtime().run("add_scalar_batch", size, offsets, scalar,
                  batch[0].buffer, batch[1].buffer, batch[2].buffer, batch[3].buffer,
                  batch[4].buffer, batch[5].buffer, batch[6].buffer, batch[7].buffer);
}

void amp_unscale(BufferSlice grad, BufferSlice found_inf, BufferSlice inv_scale, cl_ulong length, bool fp16) {
    const std::size_t size[] = {static_cast<std::size_t>((length + 3) / 4)};
    runtime().run("amp_unscale", size, grad.buffer, found_inf.buffer, inv_scale.buffer,
                  grad.offset, found_inf.offset, inv_scale.offset, length, static_cast<cl_uint>(fp16));
}

void amp_update_scale(BufferSlice scale, BufferSlice growth, BufferSlice found_inf,
                      float growth_factor, float backoff_factor, cl_int growth_interval) {
    constexpr std::size_t size[] = {1};
    runtime().run("amp_update_scale", size, scale.buffer, growth.buffer, found_inf.buffer,
                  scale.offset, growth.offset, found_inf.offset, growth_factor, backoff_factor, growth_interval);
}

void scale_add(cl_mem output, cl_mem input, cl_ulong length, float scale, float bias) {
    if (!length) return;
    constexpr std::size_t local[] = {256};
    const auto vectors = static_cast<std::size_t>((length + 3) / 4);
    if (vectors >= local[0]) {
        const std::size_t size[] = {((vectors + local[0] - 1) / local[0]) * local[0]};
        runtime().run_local("scale_add_tensor", size, local, output, input, length, scale, bias);
    } else {
        const std::size_t size[] = {vectors};
        runtime().run("scale_add_tensor", size, output, input, length, scale, bias);
    }
}

void contiguous_elementwise(
    Elementwise op, cl_mem output, cl_mem left, cl_mem right,
    cl_ulong length, cl_ulong output_offset, cl_ulong left_offset, cl_ulong right_offset,
    float alpha, float beta, bool fp16) {
    if (!length) {
        return;
    }
    const std::size_t vec_count = static_cast<std::size_t>((length + 3) / 4);
    if (vec_count >= 256) {
        constexpr std::size_t local[] = {256};
        const std::size_t size[] = {((vec_count + 255) / 256) * 256};
        runtime().run_local("contiguous_elementwise_kernel", size, local,
                            output, left, right, length, output_offset, left_offset, right_offset,
                            alpha, beta, static_cast<cl_uint>(op), static_cast<cl_uint>(fp16 ? 1 : 0));
    } else {
        const std::size_t size[] = {vec_count};
        runtime().run("contiguous_elementwise_kernel", size,
                      output, left, right, length, output_offset, left_offset, right_offset,
                      alpha, beta, static_cast<cl_uint>(op), static_cast<cl_uint>(fp16 ? 1 : 0));
    }
}

void strided_elementwise(
    Elementwise op, cl_mem output, cl_mem left, cl_mem right, cl_ulong length,
    cl_ulong output_offset, cl_ulong left_offset, cl_ulong right_offset,
    std::span<const cl_ulong> geometry, float alpha, float beta, bool serial, bool fp16) {
    if (!length) {
        return;
    }
    if (geometry.size() <= 24) {
        const auto geom = pack_geom24(geometry);
        const std::size_t work = serial ? 1 : static_cast<std::size_t>(length);
        if (!serial && work >= 256) {
            constexpr std::size_t local[] = {256};
            const std::size_t size[] = {((work + 255) / 256) * 256};
            runtime().run_local("strided_elementwise_inline", size, local,
                                geom, output, left, right, length, output_offset, left_offset, right_offset,
                                static_cast<cl_ulong>(geometry.size() / 4), alpha, beta, static_cast<cl_uint>(op),
                                static_cast<cl_uint>(fp16));
        } else {
            const std::size_t size[] = {work};
            runtime().run("strided_elementwise_inline", size,
                          geom, output, left, right, length, output_offset, left_offset, right_offset,
                          static_cast<cl_ulong>(geometry.size() / 4), alpha, beta, static_cast<cl_uint>(op),
                          static_cast<cl_uint>(fp16));
        }
        return;
    }
    runtime().run_with_geometry(
        "strided_elementwise", serial ? 1 : length, nullptr, geometry,
        output, left, right, length, output_offset, left_offset, right_offset,
        static_cast<cl_ulong>(geometry.size() / 4), alpha, beta, static_cast<cl_uint>(op),
        static_cast<cl_uint>(fp16));
}

void typed_copy(cl_mem output, cl_mem input, cl_ulong length, cl_ulong output_offset,
                cl_ulong input_offset, std::span<const cl_ulong> geometry, ValueType output_type,
                ValueType input_type, cl_ulong integer_fill, bool fill, bool conjugate, bool negate) {
    if (geometry.size() <= 24) {
        const auto geom = pack_geom24(geometry);
        const std::size_t size[] = {fill ? 1 : static_cast<std::size_t>(length)};
        runtime().run("typed_copy_inline", size,
                      geom, output, input, length, output_offset, input_offset,
                      static_cast<cl_ulong>(geometry.size() / 4),
                      static_cast<cl_uint>(output_type), static_cast<cl_uint>(input_type),
                      integer_fill, static_cast<cl_uint>(fill),
                      static_cast<cl_uint>(conjugate), static_cast<cl_uint>(negate));
        return;
    }
    runtime().run_with_geometry(
        "typed_copy", fill ? 1 : length, nullptr, geometry,
        output, input, length, output_offset, input_offset, static_cast<cl_ulong>(geometry.size() / 4),
        static_cast<cl_uint>(output_type), static_cast<cl_uint>(input_type), integer_fill,
        static_cast<cl_uint>(fill), static_cast<cl_uint>(conjugate), static_cast<cl_uint>(negate));
}

void integer_binary(cl_mem output, cl_mem left, cl_mem right, cl_ulong length,
                    cl_ulong output_offset, cl_ulong left_offset, cl_ulong right_offset,
                    std::span<const cl_ulong> geometry, ValueType input_type, ValueType output_type,
                    IntegerOperation operation, cl_long alpha) {
    if (geometry.size() <= 24) {
        const auto geom = pack_geom24(geometry);
        const std::size_t size[] = {static_cast<std::size_t>(length)};
        runtime().run("integer_binary_inline", size,
                      geom, output, left, right, output_offset, left_offset, right_offset,
                      static_cast<cl_ulong>(geometry.size() / 4),
                      static_cast<cl_uint>(input_type), static_cast<cl_uint>(output_type),
                      static_cast<cl_uint>(operation), alpha);
        return;
    }
    runtime().run_with_geometry("integer_binary", length, nullptr, geometry, output, left, right, output_offset,
                                left_offset, right_offset, static_cast<cl_ulong>(geometry.size() / 4),
                                static_cast<cl_uint>(input_type), static_cast<cl_uint>(output_type),
                                static_cast<cl_uint>(operation), alpha);
}

void integer_matmul(cl_mem output, MatrixView left, MatrixView right, cl_ulong batches,
                    cl_ulong rows, cl_ulong columns, cl_ulong inner, ValueType type) {
    const auto length = static_cast<std::size_t>(batches * rows * columns);
    if (!length) return;
    const std::size_t size[] = {length};
    runtime().run("integral_matmul", size, output, left.buffer, right.buffer,
                  rows, columns, inner, left.offset, left.batch_stride, left.row_stride, left.column_stride,
                  right.offset, right.batch_stride, right.row_stride, right.column_stride, static_cast<cl_uint>(type));
}

bool supports_packed_int8_matmul() {
    return runtime().supports_packed_int8_matmul();
}

void packed_int8_matmul(cl_mem output, cl_ulong output_offset, MatrixView left, MatrixView right_transposed,
                        cl_ulong rows, cl_ulong columns, cl_ulong inner) {
    const std::size_t size[] = {columns / 4, rows / 4};
    constexpr std::size_t local[] = {8, 8};
    runtime().run_local(packed_int8_kernel_name, size, local, output, left.buffer, right_transposed.buffer,
                        output_offset, columns, inner / 4, left.offset / 4, right_transposed.offset / 4);
}

void strided_int8_matmul(cl_mem output, cl_ulong output_offset, MatrixView left, MatrixView right,
                         cl_ulong rows, cl_ulong columns, cl_ulong inner) {
    const std::size_t size[] = {columns, rows};
    runtime().run("int8_matmul_strided", size, output, left.buffer, right.buffer, output_offset, columns, inner,
                  left.offset, left.row_stride, left.column_stride, right.offset, right.row_stride, right.column_stride);
}

void masked_copy(cl_mem output, cl_mem count, cl_mem input, cl_mem mask, cl_ulong length,
                 cl_ulong input_offset, cl_ulong mask_offset, std::span<const cl_ulong> geometry,
                 ValueType type) {
    runtime().run_with_geometry(
        "masked_copy", 1, nullptr, geometry, output, count, input, mask, length, input_offset, mask_offset,
        static_cast<cl_ulong>(geometry.size() / 4), static_cast<cl_uint>(type));
}

void extrema(cl_mem output, cl_mem input, cl_ulong length, cl_ulong offset,
             std::span<const cl_ulong> geometry, ValueType type, Extrema op) {
    runtime().run_with_geometry(
        "strided_extrema", reduction_group_size, &reduction_group_size, geometry, output, input,
        length, offset, static_cast<cl_ulong>(geometry.size() / 2),
        static_cast<cl_uint>(type), static_cast<cl_uint>(op));
}

void integer_sum(cl_mem output, cl_mem input, cl_ulong elements, cl_ulong reduce_length,
                 cl_ulong offset, cl_ulong outer_rank, std::span<const cl_ulong> geometry,
                 ValueType input_type, ValueType output_type, bool contiguous, cl_ulong chunks) {
    runtime().run_with_geometry("integer_sum", elements * chunks * reduction_group_size,
        &reduction_group_size, geometry, output, input, reduce_length, offset, outer_rank,
        static_cast<cl_ulong>(geometry.size() / 2), static_cast<cl_uint>(input_type),
        static_cast<cl_uint>(output_type), static_cast<cl_uint>(contiguous), chunks);
}

void logical_reduce(cl_mem output, cl_mem input, cl_ulong length, cl_ulong reduce_length,
                    cl_ulong offset, cl_ulong outer_rank, std::span<const cl_ulong> geometry,
                    ValueType type, bool all) {
    runtime().run_with_geometry("logical_reduce", length, nullptr, geometry, output, input, reduce_length,
                                offset, outer_rank, static_cast<cl_ulong>(geometry.size() / 2),
                                static_cast<cl_uint>(type), static_cast<cl_uint>(all));
}

void triangular(cl_mem output, cl_mem input, cl_ulong length, cl_ulong offset, cl_ulong rows,
                cl_ulong columns, cl_long diagonal, std::span<const cl_ulong> geometry,
                ValueType type, bool upper) {
    runtime().run_with_geometry("triangular", length, nullptr, geometry, output, input, offset,
                                rows, columns, diagonal, static_cast<cl_ulong>(geometry.size() / 2),
                                static_cast<cl_uint>(type), static_cast<cl_uint>(upper));
}

void tril_indices(cl_mem output, cl_ulong count, cl_ulong offset, cl_ulong row_stride,
                  cl_ulong column_stride, cl_ulong first_row, cl_ulong first_width,
                  cl_ulong growing_rows, cl_ulong growing_count, cl_ulong columns, bool int32) {
    const std::size_t size[] = {static_cast<std::size_t>(count)};
    runtime().run("tril_indices", size, output, offset, row_stride, column_stride,
                  first_row, first_width, growing_rows, growing_count, columns, static_cast<cl_int>(int32));
}

void indexed_copy(cl_mem output, cl_mem input, cl_mem index, cl_mem invalid, cl_ulong work_items,
                  cl_ulong output_offset, cl_ulong input_offset, cl_ulong index_offset,
                  std::span<const cl_ulong> geometry, cl_ulong axis, cl_ulong axis_size,
                  ValueType data_type, ValueType index_type, IndexOperation operation, bool include_self) {
    runtime().run_with_geometry(
        "indexed_copy", work_items, nullptr, geometry, output, input, index, invalid, output_offset,
        input_offset, index_offset, static_cast<cl_ulong>(geometry.size() / 4), axis, axis_size,
        static_cast<cl_uint>(data_type), static_cast<cl_uint>(index_type), static_cast<cl_uint>(operation),
        static_cast<cl_uint>(include_self));
}

void embedding_gradient(cl_mem output, cl_mem grad, cl_mem index, cl_ulong length,
                        cl_ulong index_offset, cl_ulong index_stride, cl_ulong grad_offset,
                        cl_ulong grad_row, cl_ulong grad_column, cl_ulong count, cl_ulong width,
                        cl_long padding, ValueType index_type, bool scale) {
    const std::size_t size[] = {length};
    runtime().run("embedding_gradient", size, output, grad, index, index_offset, index_stride,
                  grad_offset, grad_row, grad_column, count, width, padding,
                  static_cast<cl_uint>(index_type), static_cast<cl_uint>(scale));
}

void tensor_where(cl_mem output, cl_mem condition, cl_mem left, cl_mem right, cl_ulong length,
                  cl_ulong output_offset, cl_ulong condition_offset, cl_ulong left_offset,
                  cl_ulong right_offset, std::span<const cl_ulong> geometry, ValueType type) {
    runtime().run_with_geometry(
        "tensor_where", length, nullptr, geometry, output, condition, left, right, output_offset,
        condition_offset, left_offset, right_offset, static_cast<cl_ulong>(geometry.size() / 5),
        static_cast<cl_uint>(type));
}

void fft_radix2(BufferSlice output, BufferSlice real, BufferSlice imaginary,
                cl_ulong batches, cl_uint length, bool forward) {
    const std::size_t local[] = {std::min<std::size_t>(128, length / 2)};
    const std::size_t size[] = {batches * local[0]};
    runtime().run_local("fft_radix2_local", size, local, output.buffer, real.buffer, imaginary.buffer,
                        real.offset, imaginary.offset, length, forward ? -1.0f : 1.0f);
}

void run_spatial(const char* name, std::size_t work_items, std::size_t local_size,
                 std::span<const cl_ulong> geometry, const std::array<BufferSlice, 6>& tensors) {
    if (work_items) {
        runtime().run_with_geometry(
            name, work_items, local_size ? &local_size : nullptr, geometry,
            tensors[0].buffer, tensors[1].buffer, tensors[2].buffer,
            tensors[3].buffer, tensors[4].buffer, tensors[5].buffer,
            tensors[0].offset, tensors[1].offset, tensors[2].offset,
            tensors[3].offset, tensors[4].offset, tensors[5].offset);
    }
}

void conv2d_forward(BufferSlice output, BufferSlice input, BufferSlice weight, BufferSlice bias,
                    const Conv2dParams& p, bool fp16) {
    const cl_int h_flag = fp16 ? 1 : 0;
    if (p.groups > 1 && p.c_in == p.groups && p.c_out == p.groups &&
        p.kh == 3 && p.kw == 3 && p.dy == 1 && p.dx == 1 && p.sy <= 2 && p.sx <= 2) {
        constexpr std::size_t local[] = {16, 4, 1};
        const std::size_t size[] = {(static_cast<std::size_t>(p.ow) + 15) / 16 * 16,
                                    (static_cast<std::size_t>(p.oh) + 3) / 4 * 4,
                                    static_cast<std::size_t>(p.n) * p.groups};
        runtime().run_local("conv2d_depthwise_3x3", size, local, output.buffer, input.buffer, weight.buffer,
                            bias.buffer, output.offset, input.offset, weight.offset, bias.offset,
                            p.groups, p.h, p.w, p.oh, p.ow, p.sy, p.sx, p.py, p.px, h_flag);
        return;
    }
    if (p.kh == 3 && p.kw == 3 && p.sy == 1 && p.sx == 1 && p.dy == 1 && p.dx == 1) {
        const int oc_per_g = p.c_out / p.groups;
        const bool use_oc32 = oc_per_g > 16;
        const int oc_blk = use_oc32 ? 32 : 16;
        const std::size_t tiles_w = static_cast<std::size_t>((p.ow + 15) / 16);
        const std::size_t tiles_h = static_cast<std::size_t>((p.oh + 7) / 8);
        const std::size_t oc_bpg = static_cast<std::size_t>((oc_per_g + oc_blk - 1) / oc_blk);
        const std::size_t size[] = {tiles_w * tiles_h * 16, oc_bpg * static_cast<std::size_t>(p.groups) * 8,
                                    static_cast<std::size_t>(p.n)};
        constexpr std::size_t local[] = {16, 8, 1};
        runtime().run_local(use_oc32 ? "conv2d_direct_3x3_s1_oc32" : "conv2d_direct_3x3_s1_oc16", size, local,
                            output.buffer, input.buffer, weight.buffer, bias.buffer,
                            output.offset, input.offset, weight.offset, bias.offset,
                            p.n, p.c_in, p.h, p.w, p.c_out, p.oh, p.ow, p.py, p.px, p.groups, h_flag);
        return;
    }
    const std::size_t P = static_cast<std::size_t>(p.oh) * static_cast<std::size_t>(p.ow);
    const std::size_t oc_per_g = static_cast<std::size_t>(p.c_out / p.groups);
    const std::size_t oc_bpg = (oc_per_g + 31) / 32;
    const std::size_t size[] = {((P + 63) / 64) * 16, oc_bpg * static_cast<std::size_t>(p.groups) * 8,
                                static_cast<std::size_t>(p.n)};
    constexpr std::size_t local[] = {16, 8, 1};
    runtime().run_local("conv2d_implicit_gemm", size, local,
                        output.buffer, input.buffer, weight.buffer, bias.buffer,
                        output.offset, input.offset, weight.offset, bias.offset,
                        p.n, p.c_in, p.h, p.w, p.c_out, p.oh, p.ow,
                        p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.dy, p.dx, p.groups, h_flag);
}

void conv2d_grad_input(BufferSlice grad_input, BufferSlice grad_output, BufferSlice weight,
                       const Conv2dParams& p, bool fp16) {
    const cl_int h_flag = fp16 ? 1 : 0;
    if (p.kh == 3 && p.kw == 3 && p.sy == 1 && p.sx == 1 && p.dy == 1 && p.dx == 1) {
        const std::size_t tiles_w = static_cast<std::size_t>((p.w + 15) / 16);
        const std::size_t tiles_h = static_cast<std::size_t>((p.h + 7) / 8);
        const std::size_t ic_per_g = static_cast<std::size_t>(p.c_in / p.groups);
        const std::size_t ic_bpg = (ic_per_g + 15) / 16;
        const std::size_t size[] = {tiles_w * tiles_h * 16, ic_bpg * static_cast<std::size_t>(p.groups) * 8,
                                    static_cast<std::size_t>(p.n)};
        constexpr std::size_t local[] = {16, 8, 1};
        runtime().run_local("conv2d_grad_input_3x3_s1", size, local,
                            grad_input.buffer, grad_output.buffer, weight.buffer,
                            grad_input.offset, grad_output.offset, weight.offset,
                            p.n, p.c_in, p.h, p.w, p.c_out, p.oh, p.ow, p.py, p.px, p.groups, h_flag);
        return;
    }
    const std::size_t total = static_cast<std::size_t>(p.n) * static_cast<std::size_t>(p.c_in) *
                              static_cast<std::size_t>(p.h) * static_cast<std::size_t>(p.w);
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("conv2d_grad_input_general", size, local,
                        grad_input.buffer, grad_output.buffer, weight.buffer,
                        grad_input.offset, grad_output.offset, weight.offset,
                        p.n, p.c_in, p.h, p.w, p.c_out, p.oh, p.ow,
                        p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.dy, p.dx, p.groups, h_flag);
}

void conv2d_grad_weight(BufferSlice grad_weight, BufferSlice partial, BufferSlice grad_output,
                        BufferSlice input, const Conv2dParams& p, cl_int split_k, bool fp16) {
    const std::size_t ic_per_g = static_cast<std::size_t>(p.c_in / p.groups);
    const std::size_t oc_per_g = static_cast<std::size_t>(p.c_out / p.groups);
    const std::size_t k_col = ic_per_g * static_cast<std::size_t>(p.kh) * static_cast<std::size_t>(p.kw);
    const std::size_t oc_bpg = (oc_per_g + 31) / 32;
    const std::size_t size[] = {((k_col + 31) / 32) * 8, oc_bpg * static_cast<std::size_t>(p.groups) * 8,
                                static_cast<std::size_t>(split_k)};
    constexpr std::size_t local[] = {8, 8, 1};
    const BufferSlice target = split_k > 1 ? partial : grad_weight;
    const cl_int in_h = fp16 ? 1 : 0;
    const cl_int out_h = (split_k > 1) ? 0 : in_h;
    runtime().run_local("conv2d_grad_weight_splitk_32x32", size, local,
                        target.buffer, grad_output.buffer, input.buffer,
                        target.offset, grad_output.offset, input.offset,
                        p.n, p.c_in, p.h, p.w, p.c_out, p.oh, p.ow,
                        p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.dy, p.dx, p.groups, split_k,
                        in_h, out_h);
    if (split_k > 1) {
        const cl_ulong count = static_cast<cl_ulong>(p.c_out) * static_cast<cl_ulong>(k_col);
        constexpr std::size_t rlocal[] = {64};
        const std::size_t rsize[] = {((static_cast<std::size_t>(count) + 63) / 64) * 64};
        runtime().run_local("reduce_splitk", rsize, rlocal,
                            grad_weight.buffer, partial.buffer, grad_weight.offset, count, split_k, in_h);
    }
}

void max_pool2d_forward(BufferSlice output, BufferSlice indices, BufferSlice input, const Pool2dParams& p, bool fp16) {
    const int nc = p.n * p.c;
    const cl_int h_flag = fp16 ? 1 : 0;
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(p.ow) + 15) / 16) * 16,
        ((static_cast<std::size_t>(p.oh) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    if (p.kh == 2 && p.kw == 2 && p.sy == 2 && p.sx == 2 && p.py == 0 && p.px == 0 && p.dy == 1 && p.dx == 1) {
        runtime().run_local("max_pool2d_forward_2x2_s2", size, local,
                            output.buffer, indices.buffer, input.buffer,
                            output.offset, indices.offset, input.offset,
                            nc, p.h, p.w, p.oh, p.ow, h_flag);
    } else {
        runtime().run_local("max_pool2d_forward_general", size, local,
                            output.buffer, indices.buffer, input.buffer,
                            output.offset, indices.offset, input.offset,
                            nc, p.h, p.w, p.oh, p.ow,
                            p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.dy, p.dx, h_flag);
    }
}

void max_pool2d_backward(BufferSlice grad_input, BufferSlice grad_output, BufferSlice indices, const Pool2dParams& p, bool fp16) {
    const int nc = p.n * p.c;
    const cl_int h_flag = fp16 ? 1 : 0;
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(p.w) + 15) / 16) * 16,
        ((static_cast<std::size_t>(p.h) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    if (p.kh == 2 && p.kw == 2 && p.sy == 2 && p.sx == 2 && p.py == 0 && p.px == 0 && p.dy == 1 && p.dx == 1) {
        runtime().run_local("max_pool2d_backward_2x2_s2", size, local,
                            grad_input.buffer, grad_output.buffer, indices.buffer,
                            grad_input.offset, grad_output.offset, indices.offset,
                            nc, p.h, p.w, p.oh, p.ow, h_flag);
    } else {
        runtime().run_local("max_pool2d_backward_general", size, local,
                            grad_input.buffer, grad_output.buffer, indices.buffer,
                            grad_input.offset, grad_output.offset, indices.offset,
                            nc, p.h, p.w, p.oh, p.ow,
                            p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.dy, p.dx, h_flag);
    }
}

void avg_pool2d_forward(BufferSlice output, BufferSlice input, const Pool2dParams& p, bool fp16) {
    const int nc = p.n * p.c;
    const cl_int h_flag = fp16 ? 1 : 0;
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(p.ow) + 15) / 16) * 16,
        ((static_cast<std::size_t>(p.oh) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    runtime().run_local("avg_pool2d_forward_general", size, local,
                        output.buffer, input.buffer, output.offset, input.offset,
                        nc, p.h, p.w, p.oh, p.ow,
                        p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.include_pad, p.divisor, h_flag);
}

void avg_pool2d_backward(BufferSlice grad_input, BufferSlice grad_output, const Pool2dParams& p, bool fp16) {
    const int nc = p.n * p.c;
    const cl_int h_flag = fp16 ? 1 : 0;
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(p.w) + 15) / 16) * 16,
        ((static_cast<std::size_t>(p.h) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    runtime().run_local("avg_pool2d_backward_general", size, local,
                        grad_input.buffer, grad_output.buffer, grad_input.offset, grad_output.offset,
                        nc, p.h, p.w, p.oh, p.ow,
                        p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.include_pad, p.divisor, h_flag);
}

void sdpa_flash_forward(BufferSlice output, BufferSlice lse, BufferSlice query, BufferSlice key, BufferSlice value,
                        const SdpaParams& p) {
    const int d_max = std::max(p.dq, p.dv);
    const std::size_t bh = static_cast<std::size_t>(p.b) * static_cast<std::size_t>(p.hq);
    const cl_int causal_int = p.causal ? 1 : 0;
    if (d_max > 64 && p.lq >= 32) {
        const std::size_t q_blocks = static_cast<std::size_t>((p.lq + 63) / 64);
        const std::size_t size[] = {q_blocks * 8, bh * 16};
        constexpr std::size_t local[] = {8, 16};
        runtime().run_local(p.fp16 ? "sdpa_flash_d128_sg8_fp16" : "sdpa_flash_d128_sg8", size, local,
                            output.buffer, lse.buffer, query.buffer, key.buffer, value.buffer,
                            output.offset, lse.offset, query.offset, key.offset, value.offset,
                            p.b, p.hq, p.hkv, p.lq, p.lk, p.dq, p.dv,
                            p.q_sb, p.q_sh, p.q_sl, p.q_sd,
                            p.k_sb, p.k_sh, p.k_sl, p.k_sd,
                            p.v_sb, p.v_sh, p.v_sl, p.v_sd,
                            p.o_sb, p.o_sh, p.o_sl, p.o_sd,
                            p.scale, causal_int);
        return;
    }
    const std::size_t q32_blocks = static_cast<std::size_t>((p.lq + 31) / 32);
    const bool use_q32 = (d_max > 32 && d_max <= 64 && p.lq >= 64 && bh * q32_blocks >= 48);
    const char* name = nullptr;
    if (use_q32) {
        name = p.fp16 ? "sdpa_flash_d64_q32_fp16" : "sdpa_flash_d64_q32";
    } else {
        if (p.fp16) {
            name = d_max <= 32 ? "sdpa_flash_d32_fp16"
                 : d_max <= 64 ? "sdpa_flash_d64_fp16"
                               : "sdpa_flash_d128_fp16";
        } else {
            name = d_max <= 32 ? "sdpa_flash_d32"
                 : d_max <= 64 ? "sdpa_flash_d64"
                               : "sdpa_flash_d128";
        }
    }
    const std::size_t q_blocks = use_q32 ? q32_blocks : static_cast<std::size_t>((p.lq + 7) / 8);
    const std::size_t size[] = {q_blocks * 16, bh * 8};
    constexpr std::size_t local[] = {16, 8};
    runtime().run_local(name, size, local,
                        output.buffer, lse.buffer, query.buffer, key.buffer, value.buffer,
                        output.offset, lse.offset, query.offset, key.offset, value.offset,
                        p.b, p.hq, p.hkv, p.lq, p.lk, p.dq, p.dv,
                        p.q_sb, p.q_sh, p.q_sl, p.q_sd,
                        p.k_sb, p.k_sh, p.k_sl, p.k_sd,
                        p.v_sb, p.v_sh, p.v_sl, p.v_sd,
                        p.o_sb, p.o_sh, p.o_sl, p.o_sd,
                        p.scale, causal_int);
}

void sdpa_flash_decode(BufferSlice output, BufferSlice partial_o, BufferSlice partial_lse,
                       BufferSlice query, BufferSlice key, BufferSlice value,
                       const SdpaParams& p, cl_int num_splits) {
    const int d_max = std::max(p.dq, p.dv);
    const char* name = nullptr;
    if (p.fp16) {
        name = d_max <= 32 ? "sdpa_decode_d32_fp16"
             : d_max <= 64 ? "sdpa_decode_d64_fp16"
                           : "sdpa_decode_d128_fp16";
    } else {
        name = d_max <= 32 ? "sdpa_decode_d32"
             : d_max <= 64 ? "sdpa_decode_d64"
                           : "sdpa_decode_d128";
    }
    const std::size_t bh = static_cast<std::size_t>(p.b) * static_cast<std::size_t>(p.hq);
    const std::size_t size[] = {static_cast<std::size_t>(num_splits) * 16, static_cast<std::size_t>(p.lq) * 8, bh};
    constexpr std::size_t local[] = {16, 8, 1};
    const cl_int causal_int = p.causal ? 1 : 0;
    runtime().run_local(name, size, local,
                        output.buffer, partial_o.buffer, partial_lse.buffer,
                        query.buffer, key.buffer, value.buffer,
                        output.offset, query.offset, key.offset, value.offset,
                        p.b, p.hq, p.hkv, p.lq, p.lk, p.dq, p.dv,
                        p.q_sb, p.q_sh, p.q_sl, p.q_sd,
                        p.k_sb, p.k_sh, p.k_sl, p.k_sd,
                        p.v_sb, p.v_sh, p.v_sl, p.v_sd,
                        p.o_sb, p.o_sh, p.o_sl, p.o_sd,
                        p.scale, causal_int, num_splits);
    if (num_splits > 1) {
        const std::size_t rsize[] = {
            ((static_cast<std::size_t>(p.dv) + 15) / 16) * 16,
            static_cast<std::size_t>(p.lq),
            bh
        };
        constexpr std::size_t rlocal[] = {16, 1, 1};
        runtime().run_local("sdpa_decode_reduce", rsize, rlocal,
                            output.buffer, partial_o.buffer, partial_lse.buffer,
                            output.offset, p.b, p.hq, p.lq, p.dv,
                            p.o_sb, p.o_sh, p.o_sl, p.o_sd,
                            num_splits, static_cast<cl_int>(p.fp16 ? 1 : 0));
    }
}

void sdpa_flash_backward(BufferSlice dq, BufferSlice dk, BufferSlice dv, BufferSlice d_buf,
                         BufferSlice dout, BufferSlice out, BufferSlice query, BufferSlice key, BufferSlice value,
                         BufferSlice lse, const SdpaBwdParams& bp, bool need_dq, bool need_dk, bool need_dv) {
    const auto& p = bp.fwd;
    const int d_max = std::max(p.dq, p.dv);
    const std::size_t bh = static_cast<std::size_t>(p.b) * static_cast<std::size_t>(p.hq);
    const std::size_t bh_kv = static_cast<std::size_t>(p.b) * static_cast<std::size_t>(p.hkv);
    const std::size_t q_blocks = static_cast<std::size_t>((p.lq + 7) / 8);
    const std::size_t k_blocks = static_cast<std::size_t>((p.lk + 7) / 8);
    constexpr std::size_t local[] = {16, 8};
    const cl_int causal_int = p.causal ? 1 : 0;

    // 1. Preprocess D = sum_d(dO * O)
    {
        const std::size_t size[] = {q_blocks * 16, bh * 8};
        runtime().run_local("sdpa_bwd_preprocess", size, local,
                            d_buf.buffer, dout.buffer, out.buffer,
                            dout.offset, out.offset,
                            p.b, p.hq, p.lq, p.dv,
                            bp.do_sb, bp.do_sh, bp.do_sl, bp.do_sd,
                            p.o_sb, p.o_sh, p.o_sl, p.o_sd,
                            static_cast<cl_int>(p.fp16 ? 1 : 0));
    }

    // 2. dQ kernel
    if (need_dq) {
        const char* dq_name = p.fp16
            ? (d_max <= 32 ? "sdpa_bwd_dq_d32_fp16" : d_max <= 64 ? "sdpa_bwd_dq_d64_fp16" : "sdpa_bwd_dq_d128_fp16")
            : (d_max <= 32 ? "sdpa_bwd_dq_d32"      : d_max <= 64 ? "sdpa_bwd_dq_d64"      : "sdpa_bwd_dq_d128");
        const std::size_t size[] = {q_blocks * 16, bh * 8};
        runtime().run_local(dq_name, size, local,
                            dq.buffer, query.buffer, key.buffer, value.buffer, dout.buffer,
                            lse.buffer, d_buf.buffer,
                            dq.offset, query.offset, key.offset, value.offset, dout.offset,
                            p.b, p.hq, p.hkv, p.lq, p.lk, p.dq, p.dv,
                            p.q_sb, p.q_sh, p.q_sl, p.q_sd,
                            p.k_sb, p.k_sh, p.k_sl, p.k_sd,
                            p.v_sb, p.v_sh, p.v_sl, p.v_sd,
                            bp.do_sb, bp.do_sh, bp.do_sl, bp.do_sd,
                            bp.dq_sb, bp.dq_sh, bp.dq_sl, bp.dq_sd,
                            p.scale, causal_int);
    }

    // 3. dK / dV kernel
    if (need_dk || need_dv) {
        const char* dkv_name = p.fp16
            ? (d_max <= 32 ? "sdpa_bwd_dkv_d32_fp16" : d_max <= 64 ? "sdpa_bwd_dkv_d64_fp16" : "sdpa_bwd_dkv_d128_fp16")
            : (d_max <= 32 ? "sdpa_bwd_dkv_d32"      : d_max <= 64 ? "sdpa_bwd_dkv_d64"      : "sdpa_bwd_dkv_d128");
        const std::size_t size[] = {k_blocks * 16, bh_kv * 8};
        runtime().run_local(dkv_name, size, local,
                            dk.buffer, dv.buffer, query.buffer, key.buffer, value.buffer, dout.buffer,
                            lse.buffer, d_buf.buffer,
                            dk.offset, dv.offset, query.offset, key.offset, value.offset, dout.offset,
                            p.b, p.hq, p.hkv, p.lq, p.lk, p.dq, p.dv,
                            p.q_sb, p.q_sh, p.q_sl, p.q_sd,
                            p.k_sb, p.k_sh, p.k_sl, p.k_sd,
                            p.v_sb, p.v_sh, p.v_sl, p.v_sd,
                            bp.do_sb, bp.do_sh, bp.do_sl, bp.do_sd,
                            bp.dk_sb, bp.dk_sh, bp.dk_sl, bp.dk_sd,
                            bp.dv_sb, bp.dv_sh, bp.dv_sl, bp.dv_sd,
                            p.scale, causal_int,
                            static_cast<cl_int>(need_dk ? 1 : 0),
                            static_cast<cl_int>(need_dv ? 1 : 0));
    }
}

void cast_fp32_fp16_contiguous(cl_mem output, cl_mem input, std::size_t length, bool to_fp16) {
    if (!length) {
        return;
    }
    const std::size_t vec_count = (length + 3) / 4;
    const cl_ulong count = static_cast<cl_ulong>(length);
    const char* name = to_fp16 ? "cast_f32_to_f16" : "cast_f16_to_f32";
    if (vec_count >= 256) {
        constexpr std::size_t local[] = {256};
        const std::size_t size[] = {((vec_count + 255) / 256) * 256};
        runtime().run_local(name, size, local, output, input, count);
    } else {
        const std::size_t size[] = {vec_count};
        runtime().run(name, size, output, input, count);
    }
}

void softmax_lastdim_contiguous(cl_mem output, cl_mem input, cl_ulong input_offset,
                                cl_ulong rows, cl_ulong width, bool logarithmic, bool fp16) {
    if (!rows || !width) {
        return;
    }
    constexpr std::size_t local[] = {64};
    const std::size_t size[] = {static_cast<std::size_t>(rows) * 64};
    if (!fp16 && width >= 512 && width % 4 == 0) {
        runtime().run_local("softmax_lastdim_vec4", size, local,
                            output, input, input_offset, width, static_cast<cl_uint>(logarithmic));
        return;
    }
    runtime().run_local("softmax_lastdim_fp32", size, local,
                        output, input, input_offset, width,
                        static_cast<cl_uint>(logarithmic), static_cast<cl_uint>(fp16 ? 1 : 0));
}

void softmax_backward_lastdim(cl_mem grad_in, cl_mem grad_out, cl_mem output,
                              cl_ulong grad_in_off, cl_ulong grad_out_off, cl_ulong out_off,
                              cl_ulong rows, cl_ulong width, bool logarithmic, bool fp16) {
    if (!rows || !width) {
        return;
    }
    constexpr std::size_t local[] = {64};
    const std::size_t size[] = {static_cast<std::size_t>(rows) * 64};
    runtime().run_local("softmax_backward_lastdim", size, local,
                        grad_in, grad_out, output, grad_in_off, grad_out_off, out_off,
                        width, static_cast<cl_uint>(logarithmic), static_cast<cl_uint>(fp16 ? 1 : 0));
}

void nll_loss_forward_fused(cl_mem loss_out, cl_mem total_weight_out,
                            MatrixView input, MatrixView target, MatrixView weight,
                            cl_ulong batch, cl_ulong classes, cl_long ignore_index, bool mean) {
    const auto err_buf = runtime().error_flag_buffer();
    constexpr std::size_t size[] = {64};
    constexpr std::size_t local[] = {64};
    runtime().run_local("nll_loss_forward_fused", size, local,
                        loss_out, total_weight_out, err_buf,
                        input.buffer, target.buffer, weight.buffer,
                        input.offset, input.row_stride, input.column_stride,
                        target.offset, target.row_stride, weight.offset, weight.row_stride,
                        batch, classes, ignore_index,
                        static_cast<cl_uint>(weight.buffer != nullptr), static_cast<cl_uint>(mean));
}

void nll_loss_backward_fused(cl_mem grad_input, MatrixView target, MatrixView weight,
                             MatrixView grad, MatrixView total_weight,
                             cl_ulong batch, cl_ulong classes, cl_long ignore_index, bool mean) {
    const auto err_buf = runtime().error_flag_buffer();
    const std::size_t total = static_cast<std::size_t>(batch * classes);
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("nll_loss_backward_fused", size, local,
                        grad_input, err_buf, target.buffer, weight.buffer,
                        grad.buffer, total_weight.buffer,
                        target.offset, target.row_stride, weight.offset, weight.row_stride,
                        grad.offset, grad.row_stride, total_weight.offset,
                        batch, classes, ignore_index,
                        static_cast<cl_uint>(weight.buffer != nullptr), static_cast<cl_uint>(mean));
}

void nll_loss_async(const NllLoss& args) {
    NllLoss copy = args;
    copy.invalid = runtime().error_flag_buffer();
    nll_loss(copy);
}

void check_async_errors() {
    runtime().check_async_errors();
}

void conv_transpose2d_forward(BufferSlice output, BufferSlice input, BufferSlice weight, BufferSlice bias,
                              const Conv2dParams& p, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(p.n) * static_cast<std::size_t>(p.c_out) *
                              static_cast<std::size_t>(p.oh) * static_cast<std::size_t>(p.ow);
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("conv_transpose2d_fwd_general", size, local,
                        output.buffer, input.buffer, weight.buffer, bias.buffer,
                        output.offset, input.offset, weight.offset, bias.offset,
                        p.n, p.c_in, p.h, p.w, p.c_out, p.oh, p.ow,
                        p.kh, p.kw, p.sy, p.sx, p.py, p.px, p.dy, p.dx, p.groups,
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void batch_norm_forward(cl_mem output, cl_mem save_mean, cl_mem save_invstd,
                        cl_mem running_mean, cl_mem running_var,
                        cl_mem input, cl_mem weight, cl_mem bias,
                        cl_int n, cl_int c, cl_int s, bool training, float momentum, float eps,
                        bool in_fp16, bool param_fp16, bool stat_fp16) {
    const cl_int ih = in_fp16 ? 1 : 0;
    const cl_int ph = param_fp16 ? 1 : 0;
    const cl_int sh = stat_fp16 ? 1 : 0;
    if (training) {
        constexpr std::size_t local[] = {128};
        const std::size_t stats_size[] = {static_cast<std::size_t>(c) * 128};
        const cl_int update_running = (running_mean && running_var) ? 1 : 0;
        runtime().run_local("bn_stats_fp32", stats_size, local,
                            input, save_mean, save_invstd, running_mean, running_var,
                            n, c, s, momentum, eps, update_running, ih, sh);
        const std::size_t total = static_cast<std::size_t>(n) * c * s;
        const std::size_t xform_size[] = {((total + 127) / 128) * 128};
        runtime().run_local("bn_transform_fp32", xform_size, local,
                            output, input, weight, bias, save_mean, save_invstd,
                            n, c, s, eps, static_cast<cl_int>(1), ih, ph, sh);
    } else {
        constexpr std::size_t local[] = {128};
        const std::size_t total = static_cast<std::size_t>(n) * c * s;
        const std::size_t xform_size[] = {((total + 127) / 128) * 128};
        runtime().run_local("bn_transform_fp32", xform_size, local,
                            output, input, weight, bias, running_mean, running_var,
                            n, c, s, eps, static_cast<cl_int>(0), ih, ph, sh);
    }
}

void batch_norm_backward(cl_mem grad_in, cl_mem grad_weight, cl_mem grad_bias,
                         cl_mem sum_dy, cl_mem sum_dy_xmu,
                         cl_mem grad_out, cl_mem input, cl_mem weight,
                         cl_mem mean_buf, cl_mem var_or_invstd,
                         cl_int n, cl_int c, cl_int s, float eps, bool train,
                         bool in_fp16, bool param_fp16, bool stat_fp16) {
    const cl_int ih = in_fp16 ? 1 : 0;
    const cl_int ph = param_fp16 ? 1 : 0;
    const cl_int sh = stat_fp16 ? 1 : 0;
    constexpr std::size_t local[] = {128};
    if (grad_weight || grad_bias || (grad_in && train)) {
        const std::size_t red_size[] = {static_cast<std::size_t>(c) * 128};
        runtime().run_local("bn_backward_reduce_fp32", red_size, local,
                            grad_out, input, mean_buf, var_or_invstd,
                            sum_dy, sum_dy_xmu, grad_weight, grad_bias,
                            n, c, s, eps, static_cast<cl_int>(train ? 1 : 0), ih, ph, sh);
    }
    if (grad_in) {
        const std::size_t total = static_cast<std::size_t>(n) * c * s;
        const std::size_t in_size[] = {((total + 127) / 128) * 128};
        runtime().run_local("bn_backward_input_fp32", in_size, local,
                            grad_in, grad_out, input, weight, mean_buf, var_or_invstd,
                            sum_dy, sum_dy_xmu, n, c, s, eps, static_cast<cl_int>(train ? 1 : 0), ih, ph, sh);
    }
}

void group_norm_forward(cl_mem output, cl_mem mean_out, cl_mem rstd_out,
                        cl_mem input, cl_mem weight, cl_mem bias,
                        cl_int n, cl_int c, cl_int hxw, cl_int groups, float eps,
                        bool in_fp16, bool param_fp16) {
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {static_cast<std::size_t>(n) * groups * 128};
    runtime().run_local("gn_forward_fp32", size, local,
                        output, mean_out, rstd_out, input, weight, bias, n, c, hxw, groups, eps,
                        static_cast<cl_int>(in_fp16 ? 1 : 0), static_cast<cl_int>(param_fp16 ? 1 : 0));
}

void group_norm_backward(cl_mem grad_in, cl_mem grad_weight, cl_mem grad_bias,
                         cl_mem grad_out, cl_mem input, cl_mem mean_buf, cl_mem rstd_buf, cl_mem weight,
                         cl_int n, cl_int c, cl_int hxw, cl_int groups,
                         bool in_fp16, bool param_fp16) {
    constexpr std::size_t local[] = {128};
    const cl_int ih = in_fp16 ? 1 : 0;
    const cl_int ph = param_fp16 ? 1 : 0;
    if (grad_in) {
        const std::size_t size[] = {static_cast<std::size_t>(n) * groups * 128};
        runtime().run_local("gn_backward_input_fp32", size, local,
                            grad_in, grad_out, input, mean_buf, rstd_buf, weight, n, c, hxw, groups, ih, ph);
    }
    if (grad_weight || grad_bias) {
        const std::size_t size[] = {static_cast<std::size_t>(c) * 128};
        runtime().run_local("gn_backward_affine_fp32", size, local,
                            grad_weight, grad_bias, grad_out, input, mean_buf, rstd_buf, n, c, hxw, groups, ih, ph);
    }
}

void layer_norm_forward_fast(BufferSlice output, BufferSlice input, BufferSlice weight, BufferSlice bias,
                             BufferSlice mean_out, BufferSlice rstd_out, cl_ulong rows, cl_int width, float eps,
                             bool in_fp16, bool param_fp16) {
    if (!rows || !width) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {static_cast<std::size_t>(rows) * 128};
    runtime().run_local("ln_forward_fast", size, local,
                        output.buffer, input.buffer, weight.buffer, bias.buffer,
                        mean_out.buffer, rstd_out.buffer,
                        output.offset, input.offset, weight.offset, bias.offset,
                        mean_out.offset, rstd_out.offset, width, eps,
                        static_cast<cl_int>(in_fp16 ? 1 : 0), static_cast<cl_int>(param_fp16 ? 1 : 0));
}

void layer_norm_grad_input_fast(BufferSlice dx, BufferSlice grad, BufferSlice input, BufferSlice weight,
                                BufferSlice mean_buf, BufferSlice rstd_buf, cl_ulong rows, cl_int width,
                                bool in_fp16, bool param_fp16) {
    if (!rows || !width) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {static_cast<std::size_t>(rows) * 128};
    runtime().run_local("ln_grad_input_fast", size, local,
                        dx.buffer, grad.buffer, input.buffer, weight.buffer,
                        mean_buf.buffer, rstd_buf.buffer,
                        dx.offset, grad.offset, input.offset, weight.offset,
                        mean_buf.offset, rstd_buf.offset, width,
                        static_cast<cl_int>(in_fp16 ? 1 : 0), static_cast<cl_int>(param_fp16 ? 1 : 0));
}

void layer_norm_grad_affine_fast(BufferSlice dw, BufferSlice db, BufferSlice grad, BufferSlice input,
                                 BufferSlice mean_buf, BufferSlice rstd_buf, cl_int rows, cl_int width,
                                 bool in_fp16, bool param_fp16) {
    if (rows <= 0 || width <= 0) return;
    constexpr std::size_t local[] = {64};
    const std::size_t size[] = {((static_cast<std::size_t>(width) + 63) / 64) * 64};
    runtime().run_local("ln_grad_affine_fast", size, local,
                        dw.buffer, db.buffer, grad.buffer, input.buffer,
                        mean_buf.buffer, rstd_buf.buffer,
                        dw.offset, db.offset, grad.offset, input.offset,
                        mean_buf.offset, rstd_buf.offset, rows, width,
                        static_cast<cl_int>(in_fp16 ? 1 : 0), static_cast<cl_int>(param_fp16 ? 1 : 0));
}

void rms_norm_forward(BufferSlice output, cl_mem rstd_out, BufferSlice input, BufferSlice weight,
                      cl_ulong rows, cl_int width, float eps, bool in_fp16, bool param_fp16) {
    if (!rows || !width) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {static_cast<std::size_t>(rows) * 128};
    runtime().run_local("rms_norm_forward_fp32", size, local,
                        output.buffer, rstd_out, input.buffer, weight.buffer,
                        output.offset, input.offset, weight.offset, width, eps,
                        static_cast<cl_int>(in_fp16 ? 1 : 0), static_cast<cl_int>(param_fp16 ? 1 : 0));
}

void rms_norm_backward(BufferSlice dx, BufferSlice dw, BufferSlice grad, BufferSlice input, BufferSlice weight,
                       cl_mem rstd_buf, cl_int rows, cl_int width, bool in_fp16, bool param_fp16) {
    if (!rows || !width) return;
    const cl_int ih = in_fp16 ? 1 : 0;
    const cl_int ph = param_fp16 ? 1 : 0;
    if (dx.buffer) {
        constexpr std::size_t local[] = {128};
        const std::size_t size[] = {static_cast<std::size_t>(rows) * 128};
        runtime().run_local("rms_norm_grad_input_fp32", size, local,
                            dx.buffer, grad.buffer, input.buffer, weight.buffer, rstd_buf,
                            dx.offset, grad.offset, input.offset, weight.offset, width, ih, ph);
    }
    if (dw.buffer) {
        constexpr std::size_t local[] = {64};
        const std::size_t size[] = {((static_cast<std::size_t>(width) + 63) / 64) * 64};
        runtime().run_local("rms_norm_grad_weight_fp32", size, local,
                            dw.buffer, grad.buffer, input.buffer, rstd_buf,
                            dw.offset, grad.offset, input.offset, rows, width, ih, ph);
    }
}

void adaptive_avg_pool2d_forward(cl_mem output, cl_mem input, cl_int nc, cl_int h, cl_int w, cl_int oh, cl_int ow, bool fp16) {
    const cl_int hf = fp16 ? 1 : 0;
    if (oh == 1 && ow == 1) {
        constexpr std::size_t local[] = {128};
        const std::size_t size[] = {static_cast<std::size_t>(nc) * 128};
        runtime().run_local("adaptive_avg_pool2d_1x1", size, local, output, input, nc, h * w, hf);
        return;
    }
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(ow) + 15) / 16) * 16,
        ((static_cast<std::size_t>(oh) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    runtime().run_local("adaptive_avg_pool2d_fwd", size, local, output, input, nc, h, w, oh, ow, hf);
}

void adaptive_avg_pool2d_backward(cl_mem grad_in, cl_mem grad_out, cl_int nc, cl_int h, cl_int w, cl_int oh, cl_int ow, bool fp16) {
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(w) + 15) / 16) * 16,
        ((static_cast<std::size_t>(h) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    runtime().run_local("adaptive_avg_pool2d_bwd", size, local, grad_in, grad_out, nc, h, w, oh, ow,
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void upsample_nearest2d_forward(cl_mem output, cl_mem input, cl_int nc, cl_int h, cl_int w,
                                cl_int oh, cl_int ow, float scale_h, float scale_w, bool fp16) {
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(ow) + 15) / 16) * 16,
        ((static_cast<std::size_t>(oh) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    runtime().run_local("upsample_nearest2d_fwd", size, local, output, input, nc, h, w, oh, ow, scale_h, scale_w,
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void upsample_nearest2d_backward(cl_mem grad_in, cl_mem grad_out, cl_int nc, cl_int h, cl_int w,
                                 cl_int oh, cl_int ow, float scale_h, float scale_w, bool fp16) {
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(w) + 15) / 16) * 16,
        ((static_cast<std::size_t>(h) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    runtime().run_local("upsample_nearest2d_bwd", size, local, grad_in, grad_out, nc, h, w, oh, ow, scale_h, scale_w,
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void upsample_bilinear2d_forward(cl_mem output, cl_mem input, cl_int nc, cl_int h, cl_int w,
                                 cl_int oh, cl_int ow, float rheight, float rwidth, bool align_corners, bool fp16) {
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(ow) + 15) / 16) * 16,
        ((static_cast<std::size_t>(oh) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    runtime().run_local("upsample_bilinear2d_fwd", size, local, output, input, nc, h, w, oh, ow,
                        rheight, rwidth, static_cast<cl_int>(align_corners ? 1 : 0),
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void upsample_bilinear2d_backward(cl_mem grad_in, cl_mem grad_out, cl_int nc, cl_int h, cl_int w,
                                  cl_int oh, cl_int ow, float rheight, float rwidth, bool align_corners, bool fp16) {
    constexpr std::size_t local[] = {16, 8, 1};
    const std::size_t size[] = {
        ((static_cast<std::size_t>(w) + 15) / 16) * 16,
        ((static_cast<std::size_t>(h) + 7) / 8) * 8,
        static_cast<std::size_t>(nc)
    };
    runtime().run_local("upsample_bilinear2d_bwd", size, local, grad_in, grad_out, nc, h, w, oh, ow,
                        rheight, rwidth, static_cast<cl_int>(align_corners ? 1 : 0),
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void linear_quant_forward(cl_mem output, BufferSlice x, BufferSlice qweight, BufferSlice scales,
                          BufferSlice zeros, BufferSlice bias,
                          cl_int m, cl_int n, cl_int k, cl_int group_size, cl_int num_groups,
                          bool x_fp16, bool sc_fp16, bool is_int4) {
    if (!m || !n || !k) return;
    TORCH_CHECK(has_subgroups(), "Iris quantized linear requires OpenCL subgroup support");
    const std::size_t n_groups = (static_cast<std::size_t>(n) + 7) / 8;
    const std::size_t m_groups = (static_cast<std::size_t>(m) + 3) / 4;
    const std::size_t size[] = {n_groups * 16, m_groups * 8};
    constexpr std::size_t local[] = {16, 8};
    runtime().run_local(is_int4 ? "linear_int4_gemv_sg16" : "linear_int8_gemv_sg16", size, local,
                        output, x.buffer, qweight.buffer, scales.buffer, zeros.buffer, bias.buffer,
                        x.offset, qweight.offset, scales.offset, zeros.offset, bias.offset,
                        m, n, k, group_size, num_groups,
                        static_cast<cl_int>(x_fp16 ? 1 : 0),
                        static_cast<cl_int>(sc_fp16 ? 1 : 0),
                        static_cast<cl_int>(zeros.buffer != nullptr ? 1 : 0),
                        static_cast<cl_int>(bias.buffer != nullptr ? 1 : 0));
}

void argmax_argmin(cl_mem out_indices, cl_mem out_values, cl_mem input,
                   cl_ulong outer_size, cl_ulong dim_size, cl_ulong inner_size,
                   ValueType type, bool is_min, bool write_values) {
    const std::size_t lines = static_cast<std::size_t>(outer_size * inner_size);
    if (!lines) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {lines * 128};
    runtime().run_local("argmax_argmin_kernel", size, local,
                        out_indices, out_values, input, outer_size, dim_size, inner_size,
                        static_cast<cl_uint>(type), static_cast<cl_uint>(is_min ? 1 : 0),
                        static_cast<cl_uint>(write_values ? 1 : 0));
}

void sort_rows_f32(cl_mem out_vals, cl_mem out_idxs, cl_mem input,
                   cl_int rows, cl_int width, cl_int out_k, bool descending, bool fp16,
                   cl_mem tmp_vals, cl_mem tmp_idxs) {
    if (!rows || !out_k) return;
    const cl_int desc_i = descending ? 1 : 0;
    const cl_int fp16_i = fp16 ? 1 : 0;
    if (width <= 1024) {
        cl_int p = 2;
        while (p < width) p <<= 1;
        const std::size_t wg = std::min<std::size_t>(256, static_cast<std::size_t>(p / 2));
        const std::size_t local[] = {wg};
        const std::size_t size[] = {static_cast<std::size_t>(rows) * wg};
        runtime().run_local("bitonic_sort_local_f32", size, local,
                            out_vals, out_idxs, input, width, out_k, p, desc_i, fp16_i);
        return;
    }
    if (out_k <= 256 && !tmp_vals) {
        constexpr std::size_t local[] = {256};
        const std::size_t size[] = {static_cast<std::size_t>(rows) * 256};
        runtime().run_local("topk_large_vocab_f32", size, local,
                            out_vals, out_idxs, input, width, out_k, desc_i, fp16_i);
        return;
    }
    cl_int p = 2;
    while (p < width) p <<= 1;
    const std::size_t total = static_cast<std::size_t>(rows) * p;
    constexpr std::size_t local[] = {128};
    const std::size_t init_size[] = {((total + 127) / 128) * 128};
    runtime().run_local("bitonic_sort_init_f32", init_size, local,
                        tmp_vals, tmp_idxs, input, rows, width, p, fp16_i);
    const std::size_t half_total = static_cast<std::size_t>(rows) * (p / 2);
    const std::size_t step_size[] = {((half_total + 127) / 128) * 128};
    for (cl_int k = 2; k <= p; k <<= 1) {
        for (cl_int j = k >> 1; j > 0; j >>= 1) {
            runtime().run_local("bitonic_sort_step_f32", step_size, local,
                                tmp_vals, tmp_idxs, rows, width, p, k, j, desc_i);
        }
    }
    const std::size_t ext_total = static_cast<std::size_t>(rows) * out_k;
    const std::size_t ext_size[] = {((ext_total + 127) / 128) * 128};
    runtime().run_local("bitonic_sort_extract_f32", ext_size, local,
                        out_vals, out_idxs, tmp_vals, tmp_idxs, rows, out_k, p, fp16_i);
}

void sort_rows_int(cl_mem out_vals, cl_mem out_idxs, cl_mem input,
                   cl_int rows, cl_int width, cl_int out_k, bool descending, ValueType type,
                   cl_mem tmp_vals, cl_mem tmp_idxs) {
    if (!rows || !out_k) return;
    const cl_int desc_i = descending ? 1 : 0;
    const cl_uint t_u = static_cast<cl_uint>(type);
    if (width <= 1024) {
        cl_int p = 2;
        while (p < width) p <<= 1;
        const std::size_t wg = std::min<std::size_t>(256, static_cast<std::size_t>(p / 2));
        const std::size_t local[] = {wg};
        const std::size_t size[] = {static_cast<std::size_t>(rows) * wg};
        runtime().run_local("bitonic_sort_local_i64", size, local,
                            out_vals, out_idxs, input, width, out_k, p, desc_i, t_u);
        return;
    }
    cl_int p = 2;
    while (p < width) p <<= 1;
    const std::size_t total = static_cast<std::size_t>(rows) * p;
    constexpr std::size_t local[] = {128};
    const std::size_t init_size[] = {((total + 127) / 128) * 128};
    runtime().run_local("bitonic_sort_init_i64", init_size, local,
                        tmp_vals, tmp_idxs, input, rows, width, p, t_u);
    const std::size_t half_total = static_cast<std::size_t>(rows) * (p / 2);
    const std::size_t step_size[] = {((half_total + 127) / 128) * 128};
    for (cl_int k = 2; k <= p; k <<= 1) {
        for (cl_int j = k >> 1; j > 0; j >>= 1) {
            runtime().run_local("bitonic_sort_step_i64", step_size, local,
                                tmp_vals, tmp_idxs, rows, width, p, k, j, desc_i);
        }
    }
    const std::size_t ext_total = static_cast<std::size_t>(rows) * out_k;
    const std::size_t ext_size[] = {((ext_total + 127) / 128) * 128};
    runtime().run_local("bitonic_sort_extract_i64", ext_size, local,
                        out_vals, out_idxs, tmp_vals, tmp_idxs, rows, out_k, p, t_u);
}

// Phase 3 runtime entry points

void multinomial_build_cdf_launch(cl_mem cdf, cl_mem probs, cl_mem invalid, cl_int rows, cl_int k, bool fp16) {
    if (!rows || !k) return;
    constexpr std::size_t local[] = {64};
    const std::size_t size[] = {((static_cast<std::size_t>(rows) + 63) / 64) * 64};
    runtime().run_local("multinomial_build_cdf", size, local,
                        cdf, probs, invalid, rows, k, static_cast<cl_int>(fp16 ? 1 : 0));
}

void multinomial_sample_repl_launch(cl_mem out, cl_mem cdf, cl_int rows, cl_int k, cl_int num_samples,
                                    cl_ulong seed, cl_ulong start_counter) {
    const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(num_samples);
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("multinomial_sample_repl", size, local,
                        out, cdf, rows, k, num_samples, seed, start_counter);
}

void multinomial_exp_keys_launch(cl_mem keys, cl_mem probs, cl_mem invalid, cl_int rows, cl_int k,
                                 cl_int num_samples, cl_ulong seed, cl_ulong start_counter, bool fp16) {
    if (!rows || !k) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {static_cast<std::size_t>(rows) * 128};
    runtime().run_local("multinomial_exp_keys", size, local,
                        keys, probs, invalid, rows, k, num_samples, seed, start_counter,
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void searchsorted_launch(cl_mem out, cl_mem seq, cl_mem vals, cl_mem sorter,
                         cl_ulong out_len, cl_ulong seq_rows, cl_ulong seq_len, cl_ulong vals_per_row,
                         ValueType seq_type, ValueType val_type, bool has_sorter, bool right, bool out_int32) {
    if (!out_len) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(out_len) + 127) / 128) * 128};
    runtime().run_local("searchsorted_kernel", size, local,
                        out, seq, vals, sorter,
                        out_len, seq_rows, seq_len, vals_per_row,
                        static_cast<cl_uint>(seq_type), static_cast<cl_uint>(val_type),
                        static_cast<cl_int>(has_sorter ? 1 : 0),
                        static_cast<cl_int>(right ? 1 : 0),
                        static_cast<cl_int>(out_int32 ? 1 : 0));
}

cl_ulong masked_scatter_launch(cl_mem dst, cl_mem mask, cl_mem src, cl_mem block_counts, cl_mem block_offsets,
                               cl_mem total_out, cl_ulong length, cl_ulong src_len, ValueType type) {
    if (!length) return 0;
    const std::size_t num_blocks = (static_cast<std::size_t>(length) + 255) / 256;
    constexpr std::size_t local[] = {256};
    const std::size_t size[] = {num_blocks * 256};
    runtime().run_local("masked_scatter_count", size, local, block_counts, mask, length);
    constexpr std::size_t one[] = {1};
    runtime().run_local("masked_scatter_prefix", one, one, block_offsets, total_out, block_counts,
                        static_cast<cl_ulong>(num_blocks));
    cl_ulong total = 0;
    download(&total, total_out, sizeof(total));
    if (total > src_len) {
        return total;
    }
    if (total > 0) {
        runtime().run_local("masked_scatter_apply", size, local,
                            dst, mask, src, block_offsets, length, src_len, static_cast<cl_uint>(type));
    }
    return total;
}

void isin_launch(cl_mem out, cl_mem elems, cl_mem tests, cl_ulong elem_len, cl_ulong test_len,
                 ValueType elem_type, ValueType test_type, bool is_sorted, bool invert) {
    if (!elem_len) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(elem_len) + 127) / 128) * 128};
    runtime().run_local("isin_kernel", size, local,
                        out, elems, tests, elem_len, test_len,
                        static_cast<cl_uint>(elem_type), static_cast<cl_uint>(test_type),
                        static_cast<cl_int>(is_sorted ? 1 : 0), static_cast<cl_int>(invert ? 1 : 0));
}

void index_add_launch(cl_mem self_buf, cl_mem src_buf, cl_mem idx_buf, cl_mem invalid,
                      cl_ulong outer_size, cl_ulong self_dim, cl_ulong idx_len, cl_ulong inner_size,
                      float alpha_f, cl_long alpha_l, ValueType data_type, ValueType idx_type) {
    const std::size_t slices = static_cast<std::size_t>(outer_size * inner_size);
    if (!slices || !idx_len) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((slices + 127) / 128) * 128};
    runtime().run_local("index_add_kernel", size, local,
                        self_buf, src_buf, idx_buf, invalid,
                        outer_size, self_dim, idx_len, inner_size,
                        alpha_f, alpha_l, static_cast<cl_uint>(data_type), static_cast<cl_uint>(idx_type));
}

void repeat_interleave_fill_launch(cl_mem out, cl_mem ends, cl_ulong total_out, cl_ulong num_repeats) {
    if (!total_out || !num_repeats) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(total_out) + 127) / 128) * 128};
    runtime().run_local("repeat_interleave_fill", size, local, out, ends, total_out, num_repeats);
}

void roll_launch(cl_mem out, cl_mem inp, std::span<const cl_ulong> sizes, std::span<const cl_long> shifts,
                 cl_ulong length, ValueType type, bool flat_mode, cl_long flat_shift) {
    if (!length) return;
    cl_mem sz_buf = sizes.empty() ? nullptr : allocate(sizes.size() * sizeof(cl_ulong));
    cl_mem sh_buf = shifts.empty() ? nullptr : allocate(shifts.size() * sizeof(cl_long));
    if (sz_buf) upload(sz_buf, sizes.data(), sizes.size() * sizeof(cl_ulong));
    if (sh_buf) upload(sh_buf, shifts.data(), shifts.size() * sizeof(cl_long));
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("roll_kernel", size, local,
                        out, inp, sz_buf, sh_buf, length, static_cast<cl_ulong>(sizes.size()),
                        static_cast<cl_uint>(type), static_cast<cl_int>(flat_mode ? 1 : 0), flat_shift);
    if (sz_buf) release(sz_buf);
    if (sh_buf) release(sh_buf);
}

void constant_pad_nd_launch(cl_mem out, cl_mem inp, std::span<const cl_ulong> out_sizes,
                            std::span<const cl_ulong> in_sizes, std::span<const cl_long> pad_before,
                            cl_ulong length, ValueType type, float fill_f, cl_long fill_l) {
    if (!length) return;
    const std::size_t rank = out_sizes.size();
    cl_mem os_buf = rank ? allocate(rank * sizeof(cl_ulong)) : nullptr;
    cl_mem is_buf = rank ? allocate(rank * sizeof(cl_ulong)) : nullptr;
    cl_mem pb_buf = rank ? allocate(rank * sizeof(cl_long)) : nullptr;
    if (os_buf) upload(os_buf, out_sizes.data(), rank * sizeof(cl_ulong));
    if (is_buf) upload(is_buf, in_sizes.data(), rank * sizeof(cl_ulong));
    if (pb_buf) upload(pb_buf, pad_before.data(), rank * sizeof(cl_long));
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("constant_pad_nd_kernel", size, local,
                        out, inp, os_buf, is_buf, pb_buf, length, static_cast<cl_ulong>(rank),
                        static_cast<cl_uint>(type), fill_f, fill_l);
    if (os_buf) release(os_buf);
    if (is_buf) release(is_buf);
    if (pb_buf) release(pb_buf);
}

void pad2d_mode_fwd_launch(cl_mem out, cl_mem inp, cl_int nc, cl_int h, cl_int w, cl_int oh, cl_int ow,
                           cl_int pad_l, cl_int pad_t, cl_int mode, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(nc) * oh * ow;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("pad2d_mode_fwd", size, local,
                        out, inp, nc, h, w, oh, ow, pad_l, pad_t, mode, static_cast<cl_int>(fp16 ? 1 : 0));
}

void pad2d_mode_bwd_launch(cl_mem grad_in, cl_mem grad_out, cl_int nc, cl_int h, cl_int w, cl_int oh, cl_int ow,
                           cl_int pad_l, cl_int pad_r, cl_int pad_t, cl_int pad_b, cl_int mode, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(nc) * h * w;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("pad2d_mode_bwd", size, local,
                        grad_in, grad_out, nc, h, w, oh, ow, pad_l, pad_r, pad_t, pad_b, mode,
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void act_p3_fwd_launch(cl_mem out, cl_mem inp, cl_ulong length, cl_int act,
                       float p0, float p1, float p2, bool fp16) {
    if (!length) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("act_p3_fwd", size, local,
                        out, inp, length, act, p0, p1, p2, static_cast<cl_int>(fp16 ? 1 : 0));
}

void act_p3_bwd_launch(cl_mem grad_in, cl_mem grad_out, cl_mem self_or_res, cl_ulong length, cl_int act,
                       float p0, float p1, float p2, bool is_result, bool fp16) {
    if (!length) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("act_p3_bwd", size, local,
                        grad_in, grad_out, self_or_res, length, act, p0, p1, p2,
                        static_cast<cl_int>(is_result ? 1 : 0), static_cast<cl_int>(fp16 ? 1 : 0));
}

void prelu_fwd_launch(cl_mem out, cl_mem inp, cl_mem weight, cl_ulong length,
                      cl_ulong channels, cl_ulong inner_size, bool single_weight, bool fp16) {
    if (!length) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("prelu_fwd_kernel", size, local,
                        out, inp, weight, length, channels, inner_size,
                        static_cast<cl_int>(single_weight ? 1 : 0), static_cast<cl_int>(fp16 ? 1 : 0));
}

void prelu_bwd_launch(cl_mem dx, cl_mem dw, cl_mem grad_out, cl_mem inp, cl_mem weight,
                      cl_ulong outer_size, cl_ulong channels, cl_ulong inner_size,
                      bool single_weight, bool fp16) {
    const cl_ulong length = outer_size * channels * inner_size;
    if (!length) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size_dx[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("prelu_bwd_dx_kernel", size_dx, local,
                        dx, grad_out, inp, weight, length, channels, inner_size,
                        static_cast<cl_int>(single_weight ? 1 : 0), static_cast<cl_int>(fp16 ? 1 : 0));
    const std::size_t w_cnt = single_weight ? 1 : static_cast<std::size_t>(channels);
    const std::size_t size_dw[] = {w_cnt * 128};
    runtime().run_local("prelu_bwd_dw_kernel", size_dw, local,
                        dw, grad_out, inp, outer_size, channels, inner_size,
                        static_cast<cl_int>(single_weight ? 1 : 0), static_cast<cl_int>(fp16 ? 1 : 0));
}

void grid_sampler_2d_fwd_launch(cl_mem out, cl_mem inp, cl_mem grid,
                                cl_int n, cl_int c, cl_int ih, cl_int iw, cl_int oh, cl_int ow,
                                cl_int interp_mode, cl_int padding_mode, bool align_corners, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(n) * oh * ow;
    if (!total || !c) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("grid_sampler_2d_fwd", size, local,
                        out, inp, grid, n, c, ih, iw, oh, ow,
                        interp_mode, padding_mode, static_cast<cl_int>(align_corners ? 1 : 0),
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void grid_sampler_2d_bwd_launch(cl_mem grad_in_f32, cl_mem grad_grid, cl_mem grad_out, cl_mem inp, cl_mem grid,
                                cl_int n, cl_int c, cl_int ih, cl_int iw, cl_int oh, cl_int ow,
                                cl_int interp_mode, cl_int padding_mode, bool align_corners,
                                bool need_gi, bool need_gg, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(n) * oh * ow;
    if (!total || !c) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("grid_sampler_2d_bwd", size, local,
                        grad_in_f32, grad_grid, grad_out, inp, grid,
                        n, c, ih, iw, oh, ow,
                        interp_mode, padding_mode, static_cast<cl_int>(align_corners ? 1 : 0),
                        static_cast<cl_int>(need_gi ? 1 : 0), static_cast<cl_int>(need_gg ? 1 : 0),
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void conv3d_fwd_launch(cl_mem out, cl_mem inp, cl_mem w, cl_mem bias, const Conv3dParams& p, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(p.n) * p.c_out * p.od * p.oh * p.ow;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("conv3d_fwd_kernel", size, local,
                        out, inp, w, bias,
                        p.n, p.c_in, p.id, p.ih, p.iw, p.c_out, p.od, p.oh, p.ow,
                        p.kd, p.kh, p.kw, p.sd, p.sh, p.sw, p.pd, p.ph, p.pw, p.dd, p.dh, p.dw,
                        p.groups, static_cast<cl_int>(fp16 ? 1 : 0));
}

void conv3d_grad_input_launch(cl_mem grad_in, cl_mem grad_out, cl_mem w, const Conv3dParams& p, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(p.n) * p.c_in * p.id * p.ih * p.iw;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("conv3d_grad_input_kernel", size, local,
                        grad_in, grad_out, w,
                        p.n, p.c_in, p.id, p.ih, p.iw, p.c_out, p.od, p.oh, p.ow,
                        p.kd, p.kh, p.kw, p.sd, p.sh, p.sw, p.pd, p.ph, p.pw, p.dd, p.dh, p.dw,
                        p.groups, static_cast<cl_int>(fp16 ? 1 : 0));
}

void conv3d_grad_weight_launch(cl_mem grad_w, cl_mem grad_out, cl_mem inp, const Conv3dParams& p, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(p.c_out) * (p.c_in / p.groups) * p.kd * p.kh * p.kw;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("conv3d_grad_weight_kernel", size, local,
                        grad_w, grad_out, inp,
                        p.n, p.c_in, p.id, p.ih, p.iw, p.c_out, p.od, p.oh, p.ow,
                        p.kd, p.kh, p.kw, p.sd, p.sh, p.sw, p.pd, p.ph, p.pw, p.dd, p.dh, p.dw,
                        p.groups, static_cast<cl_int>(fp16 ? 1 : 0));
}

void max_pool3d_fwd_launch(cl_mem out, cl_mem indices, cl_mem inp, const Pool3dParams& p, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(p.nc) * p.od * p.oh * p.ow;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("max_pool3d_fwd_kernel", size, local,
                        out, indices, inp, p.nc, p.id, p.ih, p.iw, p.od, p.oh, p.ow,
                        p.kd, p.kh, p.kw, p.sd, p.sh, p.sw, p.pd, p.ph, p.pw, p.dd, p.dh, p.dw,
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void max_pool3d_bwd_launch(cl_mem grad_in_f32, cl_mem grad_out, cl_mem indices, const Pool3dParams& p, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(p.nc) * p.od * p.oh * p.ow;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("max_pool3d_bwd_kernel", size, local,
                        grad_in_f32, grad_out, indices, p.nc, p.id, p.ih, p.iw, p.od, p.oh, p.ow,
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void avg_pool3d_fwd_launch(cl_mem out, cl_mem inp, const Pool3dParams& p, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(p.nc) * p.od * p.oh * p.ow;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("avg_pool3d_fwd_kernel", size, local,
                        out, inp, p.nc, p.id, p.ih, p.iw, p.od, p.oh, p.ow,
                        p.kd, p.kh, p.kw, p.sd, p.sh, p.sw, p.pd, p.ph, p.pw,
                        p.include_pad, p.divisor, static_cast<cl_int>(fp16 ? 1 : 0));
}

void avg_pool3d_bwd_launch(cl_mem grad_in, cl_mem grad_out, const Pool3dParams& p, bool fp16) {
    const std::size_t total = static_cast<std::size_t>(p.nc) * p.id * p.ih * p.iw;
    if (!total) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((total + 127) / 128) * 128};
    runtime().run_local("avg_pool3d_bwd_kernel", size, local,
                        grad_in, grad_out, p.nc, p.id, p.ih, p.iw, p.od, p.oh, p.ow,
                        p.kd, p.kh, p.kw, p.sd, p.sh, p.sw, p.pd, p.ph, p.pw,
                        p.include_pad, p.divisor, static_cast<cl_int>(fp16 ? 1 : 0));
}

void var_std_reduce_launch(cl_mem var_out, cl_mem mean_out, cl_mem inp,
                           cl_ulong outer_size, cl_ulong reduce_len, float correction,
                           bool take_sqrt, bool write_mean, bool fp16) {
    if (!outer_size) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {static_cast<std::size_t>(outer_size) * 128};
    runtime().run_local("var_std_reduce_kernel", size, local,
                        var_out, mean_out, inp, outer_size, reduce_len, correction,
                        static_cast<cl_int>(take_sqrt ? 1 : 0),
                        static_cast<cl_int>(write_mean ? 1 : 0),
                        static_cast<cl_int>(fp16 ? 1 : 0));
}

void prod_reduce_launch(cl_mem out, cl_mem inp, cl_ulong outer_size, cl_ulong reduce_len,
                        ValueType in_type, ValueType out_type) {
    if (!outer_size) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {static_cast<std::size_t>(outer_size) * 128};
    runtime().run_local("prod_reduce_kernel", size, local,
                        out, inp, outer_size, reduce_len,
                        static_cast<cl_uint>(in_type), static_cast<cl_uint>(out_type));
}

void vector_norm_reduce_launch(cl_mem out, cl_mem inp, cl_ulong outer_size, cl_ulong reduce_len,
                               float ord, cl_int ord_mode, bool in_fp16, bool out_fp16) {
    if (!outer_size) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {static_cast<std::size_t>(outer_size) * 128};
    runtime().run_local("vector_norm_reduce_kernel", size, local,
                        out, inp, outer_size, reduce_len, ord, ord_mode,
                        static_cast<cl_int>(in_fp16 ? 1 : 0), static_cast<cl_int>(out_fp16 ? 1 : 0));
}

void ternary_contiguous_launch(cl_mem out, cl_mem a, cl_mem b, cl_mem c,
                               cl_ulong length, cl_int op, float alpha, bool fp16) {
    if (!length) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("ternary_contiguous_kernel", size, local,
                        out, a, b, c, length, op, alpha, static_cast<cl_int>(fp16 ? 1 : 0));
}

void cumsum_launch(cl_mem output, cl_mem input, cl_ulong outer_size, cl_ulong dim_size, cl_ulong inner_size,
                   ValueType in_type, ValueType out_type) {
    const std::size_t lines = static_cast<std::size_t>(outer_size * inner_size);
    if (!lines) return;
    constexpr std::size_t local[] = {64};
    const std::size_t size[] = {((lines + 63) / 64) * 64};
    runtime().run_local("cumsum_kernel", size, local,
                        output, input, outer_size, dim_size, inner_size,
                        static_cast<cl_uint>(in_type), static_cast<cl_uint>(out_type));
}

void arange_launch(cl_mem output, cl_ulong length, float start_f, float step_f,
                   cl_long start_l, cl_long step_l, ValueType type) {
    if (!length) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("arange_kernel", size, local,
                        output, length, start_f, step_f, start_l, step_l, static_cast<cl_uint>(type));
}

void linspace_launch(cl_mem output, cl_ulong steps, float start_f, float end_f, ValueType type) {
    if (!steps) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(steps) + 127) / 128) * 128};
    runtime().run_local("linspace_kernel", size, local,
                        output, steps, start_f, end_f, static_cast<cl_uint>(type));
}

void random_int_fill(cl_mem output, cl_ulong length, cl_ulong offset, std::span<const cl_ulong> geometry,
                     cl_ulong seed, cl_ulong start, cl_long low, cl_ulong range, ValueType type) {
    if (!length) return;
    runtime().run_with_geometry(
        "random_int_kernel", (static_cast<std::size_t>(length) - 1) / 4 + 1, nullptr, geometry,
        output, length, offset, static_cast<cl_ulong>(geometry.size() / 2),
        seed, start, low, range, static_cast<cl_uint>(type));
}

void flip_launch(cl_mem output, cl_mem input, cl_ulong length, cl_ulong in_offset,
                 std::span<const cl_ulong> geometry, cl_uint flip_mask, ValueType type) {
    if (!length) return;
    runtime().run_with_geometry(
        "flip_kernel", static_cast<std::size_t>(length), nullptr, geometry,
        output, input, length, in_offset, static_cast<cl_ulong>(geometry.size() / 2),
        flip_mask, static_cast<cl_uint>(type));
}

void fused_sgd_step(cl_mem param, cl_mem grad, cl_mem mom_buf, cl_mem lr_ptr,
                    cl_mem grad_scale, cl_mem found_inf,
                    cl_ulong p_off, cl_ulong g_off, cl_ulong m_off, cl_ulong length,
                    float weight_decay, float momentum, float lr_scalar, float dampening,
                    bool nesterov, bool maximize, bool is_first_step) {
    if (!length) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("fused_sgd_fp32", size, local,
                        param, grad, mom_buf, lr_ptr, grad_scale, found_inf,
                        p_off, g_off, m_off, length,
                        weight_decay, momentum, lr_scalar, dampening,
                        static_cast<cl_uint>(nesterov ? 1 : 0),
                        static_cast<cl_uint>(maximize ? 1 : 0),
                        static_cast<cl_uint>(is_first_step ? 1 : 0),
                        static_cast<cl_uint>(mom_buf != nullptr ? 1 : 0),
                        static_cast<cl_uint>(lr_ptr != nullptr ? 1 : 0));
}

void fused_adam_step(cl_mem param, cl_mem grad, cl_mem exp_avg, cl_mem exp_avg_sq, cl_mem max_exp_avg_sq,
                     cl_mem state_step, cl_mem lr_ptr, cl_mem grad_scale, cl_mem found_inf,
                     cl_ulong p_off, cl_ulong g_off, cl_ulong ea_off, cl_ulong eas_off, cl_ulong meas_off,
                     cl_ulong step_off, cl_ulong length,
                     float lr_scalar, float beta1, float beta2, float weight_decay, float eps,
                     bool amsgrad, bool maximize, bool decoupled_wd) {
    if (!length) return;
    constexpr std::size_t local[] = {128};
    const std::size_t size[] = {((static_cast<std::size_t>(length) + 127) / 128) * 128};
    runtime().run_local("fused_adam_fp32", size, local,
                        param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
                        state_step, lr_ptr, grad_scale, found_inf,
                        p_off, g_off, ea_off, eas_off, meas_off, step_off, length,
                        lr_scalar, beta1, beta2, weight_decay, eps,
                        static_cast<cl_uint>(amsgrad ? 1 : 0),
                        static_cast<cl_uint>(maximize ? 1 : 0),
                        static_cast<cl_uint>(decoupled_wd ? 1 : 0),
                        static_cast<cl_uint>(lr_ptr != nullptr ? 1 : 0));
}

void sum_all_contiguous(cl_mem output, cl_mem partial, cl_mem input, cl_ulong length,
                        cl_ulong input_offset, cl_ulong num_groups, bool average) {
    constexpr std::size_t local[] = {128};
    const std::size_t s1_size[] = {static_cast<std::size_t>(num_groups) * 128};
    runtime().run_local("sum_all_stage1", s1_size, local, partial, input, length, input_offset);
    constexpr std::size_t s2_size[] = {128};
    runtime().run_local("sum_all_stage2", s2_size, local, output, partial, num_groups, length,
                        static_cast<cl_uint>(average));
}

void sum_2d_dim0(cl_mem output, cl_mem input, cl_ulong input_offset, cl_int rows, cl_int cols,
                 bool average) {
    constexpr std::size_t local[] = {64};
    const std::size_t size[] = {((static_cast<std::size_t>(cols) + 63) / 64) * 64};
    runtime().run_local("sum_2d_dim0", size, local, output, input, input_offset, rows, cols,
                        static_cast<cl_uint>(average));
}

void sum_4d_nchw_c(cl_mem output, cl_mem input, cl_ulong input_offset, cl_int n, cl_int c,
                   cl_int hw, bool average) {
    constexpr std::size_t local[] = {256};
    const std::size_t size[] = {static_cast<std::size_t>(c) * 256};
    runtime().run_local("sum_4d_nchw_c", size, local, output, input, input_offset, n, c, hw,
                        static_cast<cl_uint>(average));
}

void nonzero_coordinates(cl_mem coordinates, cl_mem count, cl_mem input, cl_ulong length,
                         cl_ulong offset, std::span<const cl_ulong> geometry, ValueType type) {
    runtime().run_with_geometry("nonzero_coordinates", 1, nullptr, geometry,
                                coordinates, count, input, length, offset,
                                static_cast<cl_ulong>(geometry.size() / 2), static_cast<cl_uint>(type));
}

void advanced_offsets(cl_mem offsets, cl_mem index, cl_mem invalid, cl_ulong length,
                      cl_ulong index_offset, std::span<const cl_ulong> geometry, cl_ulong axis_size,
                      cl_ulong axis_stride, ValueType type) {
    runtime().run_with_geometry("advanced_offsets", length, nullptr, geometry, offsets, index, invalid,
                                index_offset, static_cast<cl_ulong>(geometry.size() / 2), axis_size,
                                axis_stride, static_cast<cl_uint>(type));
}

void advanced_copy(cl_mem base, cl_mem linear, cl_mem offsets, cl_ulong length, cl_ulong base_offset,
                   cl_ulong trailing, cl_ulong index_length, std::span<const cl_ulong> geometry,
                   ValueType type, bool put, bool accumulate) {
    runtime().run_with_geometry("advanced_copy", put ? 1 : length, nullptr, geometry, base, linear,
                                offsets, length, base_offset, static_cast<cl_ulong>(geometry.size() / 2),
                                trailing, index_length, static_cast<cl_uint>(type),
                                static_cast<cl_uint>(put), static_cast<cl_uint>(accumulate));
}

void nll_loss(const NllLoss& args) {
    const std::size_t size[] = {args.batch};
    runtime().run("nll_loss", size, args.output, args.weights, args.invalid, args.input.buffer,
                  args.target.buffer, args.weight.buffer, args.grad.buffer, args.total_weight.buffer,
                  args.input.offset, args.input.row_stride, args.input.column_stride,
                  args.target.offset, args.target.row_stride, args.weight.offset, args.weight.row_stride,
                  args.grad.offset, args.grad.row_stride, args.total_weight.offset, args.classes, args.ignore_index,
                  static_cast<cl_uint>(args.weight.buffer != nullptr),
                  static_cast<cl_uint>(args.grad.buffer != nullptr), static_cast<cl_uint>(args.mean));
}

void matrix_multiply(
    cl_mem output, MatrixView left, MatrixView right,
    cl_ulong batches, cl_ulong rows, cl_ulong columns, cl_ulong inner,
    bool fp16, MatrixView bias, float alpha, float beta) {
    if (!batches || !rows || !columns) {
        return;
    }
    if (!inner) {
        if (fp16) {
            const cl_ushort zero = 0;
            fill_bytes(output, batches * rows * columns, 0, &zero, sizeof(zero));
        } else {
            fill(output, batches * rows * columns, 0, 0.0f);
        }
        return;
    }
    runtime().run_matmul(output, left, right, batches, rows, columns, inner, bias, alpha, beta, fp16);
}

void sum(
    cl_mem output, cl_mem input, cl_ulong length, cl_ulong reduce_length, cl_ulong offset,
    cl_ulong outer_rank, std::span<const cl_ulong> geometry, bool input_half, bool output_half,
    bool average) {
    runtime().run_with_geometry(
        "strided_sum", length * reduction_group_size, &reduction_group_size, geometry,
        output, input, reduce_length, offset, outer_rank,
        static_cast<cl_ulong>(geometry.size() / 2), static_cast<cl_uint>(input_half),
        static_cast<cl_uint>(output_half), static_cast<cl_uint>(average));
}

void softmax(
    cl_mem output, cl_mem input, cl_ulong length, cl_ulong width, cl_ulong inner,
    cl_ulong offset, cl_ulong dim_stride, std::span<const cl_ulong> geometry, bool logarithmic, bool fp16) {
    if (length) {
        runtime().run_with_geometry(
            "strided_softmax", length / width * reduction_group_size, &reduction_group_size,
            geometry, output, input, width, inner, offset, dim_stride,
            static_cast<cl_ulong>(geometry.size() / 2), static_cast<cl_uint>(logarithmic),
            static_cast<cl_uint>(fp16));
    }
}

void random_fill(cl_mem output, cl_ulong length, cl_ulong offset, std::span<const cl_ulong> geometry,
                 cl_ulong seed, cl_ulong start, float first, float second, RandomDistribution distribution,
                 bool fp16) {
    runtime().run_with_geometry(
        "random_fill", (length - 1) / 4 + 1, nullptr, geometry, output, length, offset,
        static_cast<cl_ulong>(geometry.size() / 2), seed, start, first, second,
        static_cast<cl_uint>(distribution), static_cast<cl_uint>(fp16));
}

void synchronize() {
    runtime().synchronize_all();
}

std::int64_t new_stream() {
    return runtime().new_queue();
}

std::int64_t current_stream() noexcept {
    return current_queue_id;
}

std::int64_t exchange_stream(std::int64_t id) {
    if (id != 0) {
        runtime().check_queue(id);
    }
    return std::exchange(current_queue_id, id);
}

void synchronize_stream(std::int64_t id) {
    runtime().synchronize_queue(id);
}

bool query_stream(std::int64_t id) {
    return runtime().query_queue(id);
}

cl_event record_event(std::int64_t id) {
    return runtime().record(id);
}

void wait_event(std::int64_t id, cl_event event) {
    runtime().block(id, event);
}

bool query_event(cl_event event) {
    return event_complete(event);
}

void synchronize_event(cl_event event) {
    if (event) {
        check(cl().clWaitForEvents(1, &event), "synchronize event");
    }
}

void release_event(cl_event event) noexcept {
    if (event) {
        cl().clReleaseEvent(event);
    }
}

double elapsed_time(cl_event first, cl_event last) {
    TORCH_CHECK(first && last, "Iris elapsed_time requires two recorded events");
    cl_ulong start = 0;
    cl_ulong end = 0;
    check(cl().clGetEventProfilingInfo(first, CL_PROFILING_COMMAND_END, sizeof(start), &start, nullptr),
          "read start event timestamp");
    check(cl().clGetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
          "read end event timestamp");
    return (static_cast<double>(end) - static_cast<double>(start)) / 1e6;
}

void record_buffer(cl_mem mem, std::int64_t stream) {
    if (mem) {
        runtime().record_buffer(mem, stream);
    }
}

void empty_cache() {
    runtime().clear_cache();
}

MemoryStats memory_stats() {
    return runtime().memory_stats();
}

std::size_t external_buffer_size(cl_mem buffer) {
    return runtime().external_buffer_size(buffer);
}

void reset_peak_memory_stats() {
    runtime().reset_peak_memory_stats();
}

DeviceProperties device_properties() {
    const auto id = device().id;
    const auto read = [&](cl_device_info parameter, auto& result) {
        check(cl().clGetDeviceInfo(id, parameter, sizeof(result), &result, nullptr), "read GPU property");
    };
    const auto read_string = [&](cl_device_info parameter) {
        std::size_t size = 0;
        check(cl().clGetDeviceInfo(id, parameter, 0, nullptr, &size), "size GPU property");
        std::string result(size, '\0');
        check(cl().clGetDeviceInfo(id, parameter, size, result.data(), nullptr), "read GPU property");
        if (!result.empty() && result.back() == '\0') result.pop_back();
        return result;
    };
    DeviceProperties result{};
    read(CL_DEVICE_GLOBAL_MEM_SIZE, result.total_memory);
    read(CL_DEVICE_MAX_MEM_ALLOC_SIZE, result.max_allocation);
    read(CL_DEVICE_DOUBLE_FP_CONFIG, result.double_fp_config);
    read(CL_DEVICE_HOST_UNIFIED_MEMORY, result.host_unified_memory);
    read(CL_DEVICE_MAX_COMPUTE_UNITS, result.compute_units);
    result.opencl_version = read_string(CL_DEVICE_VERSION);
    result.driver_version = read_string(CL_DRIVER_VERSION);
    return result;
}

void* allocate_pinned(std::size_t bytes) {
    return bytes ? runtime().pin(bytes) : nullptr;
}

bool is_pinned(const void* pointer) {
    return pointer && context_initialized() && runtime().is_pinned(pointer);
}

void release_pinned(void* pointer) noexcept {
    if (pointer) {
        runtime().unpin(pointer);
    }
}

std::uint64_t kernel_submissions() noexcept {
    return submissions.load(std::memory_order_relaxed);
}

std::uint64_t tiled_matmul_submissions() noexcept {
    return tiled_matmul_submissions_count.load(std::memory_order_relaxed);
}

std::uint64_t blocked_matmul_submissions() noexcept {
    return blocked_matmul_submissions_count.load(std::memory_order_relaxed);
}

}  // namespace iris
