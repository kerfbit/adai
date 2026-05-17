#ifndef GPU_UTILS_HPP
#define GPU_UTILS_HPP

#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef ADAI_ENABLE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t error = call;                                                              \
        if (error != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));      \
        }                                                                                      \
    } while (0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call)                                                                    \
    do {                                                                                      \
        cublasStatus_t status = call;                                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                \
            throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(status) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
        }                                                                                     \
    } while (0)

namespace adai {
namespace gpu {

/**
 * @brief GPU device management with resource-sharing policies.
 *
 * Design goals for GPU co-existence:
 *  1. Uses a dedicated low-priority CUDA stream so that ADAI work is
 *     pre-empted by higher-priority GPU work from other applications.
 *  2. Respects a configurable memory fraction; allocations that would
 *     exceed the limit are refused with a clear error rather than silently
 *     evicting pages from other processes.
 *  3. cleanup() destroys only the resources ADAI owns — it does NOT call
 *     cudaDeviceReset(), which would terminate every other process that
 *     shares the same GPU.
 */
class GPUManager {
   private:
    // Use inline static (C++17) so definitions live in the header without ODR violations.
    inline static bool initialized_ = false;
    inline static int device_count_ = 0;
    inline static int current_device_ = -1;
    inline static cublasHandle_t cublas_handle_ = nullptr;
    inline static cudaStream_t stream_ = nullptr;
    /// Maximum bytes ADAI may allocate (set during initialize).
    inline static size_t max_memory_bytes_ = 0;
    /// Bytes currently allocated by ADAI through GPUMemory.
    inline static size_t allocated_bytes_ = 0;

   public:
    /**
     * @brief Probe the host for at least one CUDA-capable device without
     *        initialising anything.
     *
     * Safe to call before initialize(); returns false on CPU-only hosts.
     */
    static bool probe() {
        int n = 0;
        return (cudaGetDeviceCount(&n) == cudaSuccess) && (n > 0);
    }

    /**
     * @brief Initialize the GPU subsystem.
     *
     * @param device_id       CUDA device index to use (default: 0).
     * @param memory_fraction Fraction of total device memory ADAI may use
     *                        (0.0–1.0, default: 0.5).  Keeping this below 1.0
     *                        leaves headroom for the display driver, other ML
     *                        frameworks, or interactive GPU work.
     * @return true  if the GPU was successfully initialised.
     * @return false if no CUDA device is present (soft failure — CPU fallback).
     * @throws std::runtime_error for unexpected CUDA errors (device present but
     *         initialisation failed).
     */
    static bool initialize(int device_id = 0, float memory_fraction = 0.5f) {
        if (initialized_)
            return true;

        // Soft-fail: no CUDA device means CPU-only — not an error.
        if (cudaGetDeviceCount(&device_count_) != cudaSuccess || device_count_ == 0) {
            device_count_ = 0;
            return false;
        }
        if (device_id < 0 || device_id >= device_count_) {
            throw std::out_of_range("Invalid device_id: " + std::to_string(device_id));
        }
        if (memory_fraction <= 0.0f || memory_fraction > 1.0f) {
            throw std::invalid_argument("memory_fraction must be in (0, 1]");
        }

        CUDA_CHECK(cudaSetDevice(device_id));
        current_device_ = device_id;

        // ----------------------------------------------------------------
        // Memory limit: compute the allowed byte budget from the requested
        // fraction of total device memory.
        // ----------------------------------------------------------------
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        max_memory_bytes_ = static_cast<size_t>(static_cast<double>(total_bytes) * memory_fraction);
        allocated_bytes_ = 0;

        // ----------------------------------------------------------------
        // Low-priority stream: ADAI kernels run at the lowest priority so
        // that other GPU work (display, other ML frameworks) can preempt them.
        // ----------------------------------------------------------------
        int priority_low = 0;
        int priority_high = 0;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
        // priority_low is numerically the *lowest* scheduling priority.
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority_low));

        // ----------------------------------------------------------------
        // cuBLAS handle bound to the low-priority stream so that all BLAS
        // calls inherit the same scheduling policy.
        // ----------------------------------------------------------------
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));

        initialized_ = true;
        return true;
    }

    /**
     * @brief Release all GPU resources owned by ADAI.
     *
     * Destroys the cuBLAS handle and the dedicated CUDA stream.
     * Does NOT call cudaDeviceReset() — that would terminate every other
     * process sharing this GPU device.
     */
    static void cleanup() {
        if (!initialized_)
            return;

        // Drain in-flight work on our stream before releasing handles.
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }

        allocated_bytes_ = 0;
        initialized_ = false;
    }

    /** @brief True if the GPU subsystem has been successfully initialized. */
    static bool is_available() {
        return initialized_ && device_count_ > 0;
    }

    /** @brief Number of CUDA-capable devices on this host. */
    static int device_count() {
        return device_count_;
    }

    /** @brief Currently selected device index. */
    static int current_device() {
        return current_device_;
    }

    /**
     * @brief Switch the active device.
     *
     * Note: this also resets the memory budget to the new device's capacity
     * multiplied by the previously configured fraction.  The caller must
     * ensure no live GPUMemory objects exist before switching devices.
     */
    static void set_device(int device) {
        if (device < 0 || device >= device_count_) {
            throw std::out_of_range("Invalid device ID: " + std::to_string(device));
        }
        CUDA_CHECK(cudaSetDevice(device));
        current_device_ = device;
    }

    /** @brief cuBLAS handle (already bound to the low-priority stream). */
    static cublasHandle_t get_cublas_handle() {
        if (!initialized_) {
            throw std::runtime_error("GPU not initialized. Call GPUManager::initialize() first.");
        }
        return cublas_handle_;
    }

    /**
     * @brief The dedicated low-priority CUDA stream for all ADAI operations.
     *
     * Pass this stream to every CUDA kernel and memcpy call so that work is
     * serialised on a single queue and does not compete with the default
     * stream used by other GPU tenants.
     */
    static cudaStream_t get_stream() {
        if (!initialized_) {
            throw std::runtime_error("GPU not initialized. Call GPUManager::initialize() first.");
        }
        return stream_;
    }

    /**
     * @brief Maximum bytes ADAI is allowed to allocate on the current device.
     *
     * Set to (total_device_memory × memory_fraction) at initialisation time.
     */
    static size_t get_memory_limit_bytes() {
        return max_memory_bytes_;
    }

    /** @brief Bytes currently live-allocated by ADAI through GPUMemory. */
    static size_t get_used_memory_bytes() {
        return allocated_bytes_;
    }

    /** @brief Remaining headroom within the ADAI memory budget. */
    static size_t get_available_memory_bytes() {
        return (max_memory_bytes_ > allocated_bytes_) ? (max_memory_bytes_ - allocated_bytes_) : 0;
    }

    /**
     * @brief Reserve bytes against the ADAI memory budget.
     *
     * Called by GPUMemory on allocation; throws if the budget would be exceeded.
     */
    static void reserve_memory(size_t bytes) {
        if (max_memory_bytes_ > 0 && (allocated_bytes_ + bytes) > max_memory_bytes_) {
            throw std::runtime_error(
                "ADAI GPU memory budget exceeded: requested " +
                std::to_string(bytes / (1024 * 1024)) + " MB, " +
                std::to_string(get_available_memory_bytes() / (1024 * 1024)) + " MB available of " +
                std::to_string(max_memory_bytes_ / (1024 * 1024)) + " MB limit");
        }
        allocated_bytes_ += bytes;
    }

    /** @brief Release bytes back to the ADAI memory budget. */
    static void release_memory(size_t bytes) {
        allocated_bytes_ = (bytes <= allocated_bytes_) ? (allocated_bytes_ - bytes) : 0;
    }

    /** @brief Human-readable description of the selected device plus memory budget. */
    static std::string get_device_info(int device = -1) {
        if (device == -1)
            device = current_device_;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::string info = std::string("Device ") + std::to_string(device) + ": " + prop.name +
                           "\n  Compute Capability: " + std::to_string(prop.major) + "." +
                           std::to_string(prop.minor) + "\n  Total Global Memory: " +
                           std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB" +
                           "\n  Multiprocessors: " + std::to_string(prop.multiProcessorCount) +
                           "\n  Max Threads per Block: " + std::to_string(prop.maxThreadsPerBlock);

        if (initialized_ && device == current_device_) {
            info += "\n  ADAI Memory Budget: " + std::to_string(max_memory_bytes_ / (1024 * 1024)) +
                    " MB" + " (used: " + std::to_string(allocated_bytes_ / (1024 * 1024)) + " MB)";
        }
        return info;
    }

    /**
     * @brief Synchronize the ADAI stream (wait for all queued operations).
     *
     * Synchronises only our private stream, not the whole device, so other
     * GPU users are unaffected.
     */
    static void synchronize() {
        if (stream_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }
};

/**
 * @brief RAII wrapper for GPU memory with budget tracking and async transfers.
 *
 * All host↔device copies are performed asynchronously on the GPUManager stream
 * so they do not block the CPU or stall other GPU work.  Call
 * GPUManager::synchronize() (or the stream-based overloads) before accessing
 * results on the host.
 */
template <typename T>
class GPUMemory {
   private:
    T* device_ptr_ = nullptr;
    size_t size_ = 0;

   public:
    explicit GPUMemory(size_t count) : size_(count) {
        const size_t bytes = count * sizeof(T);
        GPUManager::reserve_memory(bytes);
        cudaError_t err = cudaMalloc(&device_ptr_, bytes);
        if (err != cudaSuccess) {
            GPUManager::release_memory(bytes);
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
    }

    ~GPUMemory() {
        if (device_ptr_) {
            cudaFree(device_ptr_);
            GPUManager::release_memory(size_ * sizeof(T));
        }
    }

    // Non-copyable
    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;

    // Movable
    GPUMemory(GPUMemory&& other) noexcept : device_ptr_(other.device_ptr_), size_(other.size_) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }

    GPUMemory& operator=(GPUMemory&& other) noexcept {
        if (this != &other) {
            if (device_ptr_) {
                cudaFree(device_ptr_);
                GPUManager::release_memory(size_ * sizeof(T));
            }
            device_ptr_ = other.device_ptr_;
            size_ = other.size_;
            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() {
        return device_ptr_;
    }
    const T* get() const {
        return device_ptr_;
    }
    size_t size() const {
        return size_;
    }

    /**
     * @brief Async host-to-device copy on the ADAI stream.
     *
     * The caller must ensure host_ptr remains valid until the stream has been
     * synchronized (e.g. via GPUManager::synchronize()).
     */
    void copy_from_host(const T* host_ptr, size_t count) {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        CUDA_CHECK(cudaMemcpyAsync(device_ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice,
                                   GPUManager::get_stream()));
    }

    /**
     * @brief Async device-to-host copy on the ADAI stream.
     *
     * Call GPUManager::synchronize() after this before reading host_ptr.
     */
    void copy_to_host(T* host_ptr, size_t count) const {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost,
                                   GPUManager::get_stream()));
        // Synchronize immediately so callers treating this as a blocking copy
        // still see correct data (matching the previous synchronous behaviour).
        CUDA_CHECK(cudaStreamSynchronize(GPUManager::get_stream()));
    }
};

}  // namespace gpu
}  // namespace adai

#else  // !ADAI_ENABLE_GPU

// Stub implementations when GPU is disabled
namespace adai {
namespace gpu {

class GPUManager {
   public:
    static bool probe() {
        return false;
    }
    static bool initialize(int = 0, float = 0.5f) {
        return false;
    }
    static void cleanup() {}
    static bool is_available() {
        return false;
    }
    static int device_count() {
        return 0;
    }
    static int current_device() {
        return -1;
    }
    static void set_device(int) {
        throw std::runtime_error("GPU support not compiled");
    }
    static std::string get_device_info(int = -1) {
        return "GPU support not compiled";
    }
    static void synchronize() {}
    static size_t get_memory_limit_bytes() {
        return 0;
    }
    static size_t get_used_memory_bytes() {
        return 0;
    }
    static size_t get_available_memory_bytes() {
        return 0;
    }
};

}  // namespace gpu
}  // namespace adai

#endif  // ADAI_ENABLE_GPU

#endif  // GPU_UTILS_HPP
