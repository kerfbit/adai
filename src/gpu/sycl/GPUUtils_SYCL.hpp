#ifndef GPU_UTILS_SYCL_HPP
#define GPU_UTILS_SYCL_HPP

// @adai-status: beta        (TD-041 resolved — dedicated tests added; real-device paths still unverified by an actual run, see below)
// @adai-version: 0.8.0
// @adai-reviewed: 2026-10-20


#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <cstdlib>
#include <sycl/sycl.hpp>

namespace adai {
namespace gpu {

/**
 * TD-041 (resolved September 13, 2026): added dedicated tests (tests/gpuutils_test.cpp, shared
 * with the CUDA backend — GPUManager/GPUMemory expose an identical public interface on both) for
 * probe()/initialize()'s soft-fail contract, set_device()'s bounds check, the memory-budget
 * getters/reserve_memory()/release_memory(), and GPUMemory's allocation/copy/move/budget-tracking
 * behavior. Every environment this codebase has actually been built and run in so far has a
 * compiler toolchain but no physical GPU device, so the real-hardware paths (real device
 * selection, real sycl::malloc_device-backed allocation and host↔device copies) are exercised by
 * these tests only when GPUManager::probe() reports a device present — they skip gracefully
 * rather than asserting behavior nobody has actually run. See TECHNICAL_DEBT_RESOLVED.md.
 */
class GPUManager {
   private:
    inline static bool initialized_ = false;
    inline static int device_count_ = 0;
    inline static int current_device_ = -1;
    inline static sycl::queue* queue_ = nullptr;
    inline static size_t max_memory_bytes_ = 0;
    /// Bytes of live allocations (GPUMemory blocks at their pool size, plus reserve_memory()).
    inline static size_t allocated_bytes_ = 0;
    /// TD-390: blocks a GPUMemory released, kept for reuse instead of freed. On the in-order
    /// queue a reused block is safe at once: commands that still use it were submitted earlier
    /// and run before anything its next owner submits. Keyed by pool size (pool_size()).
    inline static std::unordered_map<size_t, std::vector<void*>> free_blocks_;
    /// Bytes held in free_blocks_. Counts against the budget with allocated_bytes_, and is
    /// returned to USM (trim_pool_locked()) when an allocation would exceed the budget.
    inline static size_t cached_bytes_ = 0;
    /// Bumped whenever the queue (and so the USM context) is replaced, so a block allocated
    /// before cleanup()/set_device() is never put back into the new pool.
    inline static unsigned pool_generation_ = 0;
    /// Guards the budget counters and the pool. Allocation and release run on whatever
    /// application thread owns the GPUMemory (TSan found a race on the counters when release ran
    /// in a host_task, before TD-390 removed those).
    inline static std::mutex memory_mutex_;
    /// Remembered from initialize() so set_device() can recreate the queue
    /// with the same scheduling priority (GPU_STRATEGY) after a device switch.
    inline static bool low_priority_ = true;

    // TODO: See TD-391 in TECHNICAL_DEBT.md - collects GPUs from every platform, so one Arc card appears once per
    // backend (Level Zero and OpenCL); GPU_DEVICE_ID then picks by platform order and the backend isn't logged.
    // TD-394: GPU-only selection also means no test can run these kernels without a GPU (add a CPU-device test mode)
    static std::vector<sycl::device> enumerate_gpu_devices() {
        // TD-394: ADAI_SYCL_ALLOW_CPU_DEVICE=1 also lists CPU devices (oneAPI's OpenCL CPU
        // device), so the real kernels can run in tests on a machine with no GPU. Off by
        // default: production must never fall back to the CPU device silently.
        const char* allow_cpu = std::getenv("ADAI_SYCL_ALLOW_CPU_DEVICE");
        const auto type = (allow_cpu && allow_cpu[0] == '1') ? sycl::info::device_type::all
                                                             : sycl::info::device_type::gpu;
        std::vector<sycl::device> gpus;
        for (auto& platform : sycl::platform::get_platforms()) {
            for (auto& dev : platform.get_devices(type)) {
                gpus.push_back(dev);
            }
        }
        return gpus;
    }

    /**
     * @brief Build the ADAI queue with a scheduling-priority hint, mirroring
     *        the CUDA backend's low/high-priority stream selection.
     *
     * use_low_priority=true  → background mode (GPU_STRATEGY=background,
     *                           default): yields to other GPU work.
     * use_low_priority=false → full mode (GPU_STRATEGY=full): highest
     *                           priority, never preempted.
     *
     * priority_low/priority_high are hints (sycl_ext_oneapi_queue_priority);
     * backends that don't support them fall back to normal scheduling.
     */
    static sycl::queue make_queue(const sycl::device& dev, bool use_low_priority) {
        if (use_low_priority) {
            return sycl::queue(dev, sycl::property_list{
                                        sycl::property::queue::in_order{},
                                        sycl::ext::oneapi::property::queue::priority_low{}});
        }
        return sycl::queue(dev, sycl::property_list{
                                    sycl::property::queue::in_order{},
                                    sycl::ext::oneapi::property::queue::priority_high{}});
    }

    static bool fits_locked(size_t bytes) {
        return max_memory_bytes_ == 0 || (allocated_bytes_ + cached_bytes_ + bytes) <= max_memory_bytes_;
    }

    [[noreturn]] static void throw_budget_exceeded_locked(size_t bytes) {
        throw std::runtime_error(
            "ADAI GPU memory budget exceeded: requested " + std::to_string(bytes / (1024 * 1024)) +
            " MB, " +
            std::to_string(
                (max_memory_bytes_ > allocated_bytes_ ? (max_memory_bytes_ - allocated_bytes_) : 0) /
                (1024 * 1024)) +
            " MB available of " + std::to_string(max_memory_bytes_ / (1024 * 1024)) + " MB limit");
    }

    /// Returns every cached block to USM. sycl::free() isn't ordered against the queue, and a
    /// cached block may still be in use by a command submitted before it was released, so this
    /// waits for the queue first. Called rarely: under budget pressure, cleanup() and
    /// set_device().
    static void trim_pool_locked() {
        if (cached_bytes_ == 0) {
            return;
        }
        queue_->wait();
        for (auto& [size, blocks] : free_blocks_) {
            (void)size;
            for (void* block : blocks) {
                sycl::free(block, *queue_);
            }
        }
        free_blocks_.clear();
        cached_bytes_ = 0;
    }

   public:
    static bool probe() {
        return !enumerate_gpu_devices().empty();
    }

    static std::string probe_diagnostic() {
        auto platforms = sycl::platform::get_platforms();
        std::string diag = "SYCL found 0 GPU devices across " + std::to_string(platforms.size()) +
                           " platform(s):\n";
        for (auto& p : platforms) {
            std::string plat_name = p.get_info<sycl::info::platform::name>();
            auto all_devs = p.get_devices();
            for (auto& d : all_devs) {
                diag += "  " + plat_name + " — " + d.get_info<sycl::info::device::name>() + " [" +
                        (d.is_gpu()   ? "GPU"
                         : d.is_cpu() ? "CPU"
                                      : "other") +
                        "]\n";
            }
            if (all_devs.empty()) {
                diag += "  " + plat_name + " — (no devices)\n";
            }
        }
        diag +=
            "Hint: if Level Zero platform is missing, ensure UR_ADAPTERS_SEARCH_PATH "
            "includes the adapter libs and the user is in the render group "
            "(access to /dev/dri/renderD128).";
        return diag;
    }

    static bool initialize(int device_id = 0, float memory_fraction = 0.5f,
                           bool use_low_priority = true) {
        if (initialized_)
            return true;

        auto gpus = enumerate_gpu_devices();
        device_count_ = static_cast<int>(gpus.size());
        if (device_count_ == 0)
            return false;

        if (device_id < 0 || device_id >= device_count_) {
            throw std::out_of_range("Invalid device_id: " + std::to_string(device_id));
        }
        if (memory_fraction <= 0.0f || memory_fraction > 1.0f) {
            throw std::invalid_argument("memory_fraction must be in (0, 1]");
        }

        current_device_ = device_id;
        low_priority_ = use_low_priority;
        auto& selected = gpus[device_id];

        queue_ = new sycl::queue(make_queue(selected, use_low_priority));

        size_t total_bytes = selected.get_info<sycl::info::device::global_mem_size>();
        {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            max_memory_bytes_ =
                static_cast<size_t>(static_cast<double>(total_bytes) * memory_fraction);
            allocated_bytes_ = 0;
        }

        initialized_ = true;
        return true;
    }

    static void cleanup() {
        if (!initialized_)
            return;

        {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            if (queue_) {
                trim_pool_locked();
            }
            allocated_bytes_ = 0;
            ++pool_generation_;
        }
        if (queue_) {
            queue_->wait();
            delete queue_;
            queue_ = nullptr;
        }
        initialized_ = false;
    }

    static bool is_available() {
        return initialized_ && device_count_ > 0;
    }

    static int device_count() {
        return device_count_;
    }

    static int current_device() {
        return current_device_;
    }

    // TODO: See TD-383 in TECHNICAL_DEBT.md - existing allocations are later freed with the new queue (UB across
    // contexts) and the budget isn't recomputed; no caller
    static void set_device(int device) {
        if (device < 0 || device >= device_count_) {
            throw std::out_of_range("Invalid device ID: " + std::to_string(device));
        }
        // Switching devices requires recreating the queue
        auto gpus = enumerate_gpu_devices();
        {
            // Cached blocks belong to the old queue's context.
            std::lock_guard<std::mutex> lock(memory_mutex_);
            if (queue_) {
                trim_pool_locked();
            }
            ++pool_generation_;
        }
        if (queue_) {
            queue_->wait();
            delete queue_;
        }
        // Recreate with the same priority the queue was originally initialized
        // with (GPU_STRATEGY), matching CUDA's set_device() staying on the
        // stream priority chosen at GPUManager::initialize() time.
        queue_ = new sycl::queue(make_queue(gpus[device], low_priority_));
        current_device_ = device;
    }

    static sycl::queue& get_queue() {
        if (!initialized_) {
            throw std::runtime_error("GPU not initialized. Call GPUManager::initialize() first.");
        }
        return *queue_;
    }

    static size_t get_memory_limit_bytes() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return max_memory_bytes_;
    }

    static size_t get_used_memory_bytes() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return allocated_bytes_;
    }

    static size_t get_available_memory_bytes() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return (max_memory_bytes_ > allocated_bytes_) ? (max_memory_bytes_ - allocated_bytes_) : 0;
    }

    /// Bytes held in the pool for reuse (TD-390), not counted by get_used_memory_bytes().
    static size_t get_cached_memory_bytes() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return cached_bytes_;
    }

    /// Size a pooled allocation of `bytes` is rounded up to: 512-byte steps up to 1 MiB, then an
    /// eighth of the next power of two (at most 12.5% over), so recurring shapes share blocks.
    static size_t pool_size(size_t bytes) {
        if (bytes <= (size_t{1} << 20)) {
            return (bytes + 511) & ~size_t{511};
        }
        size_t pow2 = size_t{1} << 20;
        while (pow2 < bytes) {
            pow2 <<= 1;
        }
        const size_t step = pow2 / 8;
        return (bytes + step - 1) / step * step;
    }

    static unsigned pool_generation() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return pool_generation_;
    }

    /// Hands out a device block of pool_size(bytes) bytes, reusing a released one when there is
    /// one. Counts it in the budget; trims the pool and retries before refusing.
    static void* allocate_block(size_t bytes) {
        const size_t size = pool_size(bytes);
        std::lock_guard<std::mutex> lock(memory_mutex_);
        if (!initialized_) {
            throw std::runtime_error("GPU not initialized. Call GPUManager::initialize() first.");
        }
        auto it = free_blocks_.find(size);
        if (it != free_blocks_.end() && !it->second.empty()) {
            void* block = it->second.back();
            it->second.pop_back();
            cached_bytes_ -= size;
            allocated_bytes_ += size;
            return block;
        }
        if (!fits_locked(size)) {
            trim_pool_locked();
            if (!fits_locked(size)) {
                throw_budget_exceeded_locked(size);
            }
        }
        // TODO: See TD-392 in TECHNICAL_DEBT.md - malloc_device(0) returns nullptr, so a zero-size
        // GPUMemory throws (CUDA allows zero-size allocations)
        void* block = sycl::malloc_device(size, *queue_);
        if (!block) {
            throw std::runtime_error("sycl::malloc_device failed: allocation returned null");
        }
        allocated_bytes_ += size;
        return block;
    }

    /// Takes back a block from allocate_block(). No host_task and no wait: the block goes to the
    /// pool, and its next owner's commands queue behind any still using it. A block from before
    /// the queue was replaced (cleanup()/set_device()) is dropped; its context reclaims it.
    static void release_block(void* block, size_t bytes, unsigned generation) {
        if (!block) {
            return;
        }
        const size_t size = pool_size(bytes);
        std::lock_guard<std::mutex> lock(memory_mutex_);
        if (!initialized_ || generation != pool_generation_) {
            return;
        }
        allocated_bytes_ = (size <= allocated_bytes_) ? (allocated_bytes_ - size) : 0;
        free_blocks_[size].push_back(block);
        cached_bytes_ += size;
    }

    /// Budget-only reservation (no device memory), kept for callers that track memory they
    /// allocate themselves.
    static void reserve_memory(size_t bytes) {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        if (!fits_locked(bytes)) {
            if (queue_) {
                trim_pool_locked();
            }
            if (!fits_locked(bytes)) {
                throw_budget_exceeded_locked(bytes);
            }
        }
        allocated_bytes_ += bytes;
    }

    static void release_memory(size_t bytes) {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        allocated_bytes_ = (bytes <= allocated_bytes_) ? (allocated_bytes_ - bytes) : 0;
    }

    /**
     * @brief Human-readable description of the selected device plus memory budget.
     *
     * @throws std::out_of_range if `device` (or, when left at its default, current_device_) is
     *         not a valid index into the live device list -- including the case this is called
     *         before any successful initialize()/set_device(), when current_device_ is still its
     *         own default of -1. Matches set_device()'s identical validation and exception
     *         type/message for the same "invalid device ID" condition, and the CUDA backend's
     *         identical throw in GPUUtils.hpp -- previously this backend alone returned the
     *         string "Invalid device ID" instead of throwing, silently divergent from every
     *         other invalid-device-ID path in this class (and from CUDA's own get_device_info()).
     */
    static std::string get_device_info(int device = -1) {
        if (device == -1)
            device = current_device_;

        auto gpus = enumerate_gpu_devices();
        if (device < 0 || device >= static_cast<int>(gpus.size()))
            throw std::out_of_range("Invalid device ID: " + std::to_string(device));

        auto& dev = gpus[device];
        std::string name = dev.get_info<sycl::info::device::name>();
        size_t global_mem = dev.get_info<sycl::info::device::global_mem_size>();
        size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
        unsigned int compute_units = dev.get_info<sycl::info::device::max_compute_units>();

        std::string info = "Device " + std::to_string(device) + ": " + name +
                           "\n  Global Memory: " + std::to_string(global_mem / (1024 * 1024)) +
                           " MB" + "\n  Compute Units: " + std::to_string(compute_units) +
                           "\n  Max Work-Group Size: " + std::to_string(max_wg);

        if (initialized_ && device == current_device_) {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            info += "\n  ADAI Memory Budget: " + std::to_string(max_memory_bytes_ / (1024 * 1024)) +
                    " MB" + " (used: " + std::to_string(allocated_bytes_ / (1024 * 1024)) + " MB)";
        }
        return info;
    }

    static void synchronize() {
        if (queue_) {
            queue_->wait();
        }
    }
};

template <typename T>
class GPUMemory {
   private:
    T* device_ptr_ = nullptr;
    size_t size_ = 0;
    unsigned generation_ = 0;

    // TD-390: memory comes from and goes back to GPUManager's pool. Releasing a block used to
    // submit a host_task running sycl::free(), which kept the free behind every command still
    // using the block, but on an in-order queue each such task stalled the device for a host
    // round trip (1.35 ms vs 0.014 ms per op on the oneAPI CPU device). Pooling keeps the same
    // ordering guarantee for free: the block's next owner submits to the same in-order queue.
    void release() {
        if (device_ptr_) {
            GPUManager::release_block(device_ptr_, size_ * sizeof(T), generation_);
        }
        device_ptr_ = nullptr;
        size_ = 0;
    }

   public:
    explicit GPUMemory(size_t count) : size_(count) {
        generation_ = GPUManager::pool_generation();
        device_ptr_ = static_cast<T*>(GPUManager::allocate_block(count * sizeof(T)));
    }

    ~GPUMemory() {
        release();
    }

    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;

    GPUMemory(GPUMemory&& other) noexcept
        : device_ptr_(other.device_ptr_), size_(other.size_), generation_(other.generation_) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }

    GPUMemory& operator=(GPUMemory&& other) noexcept {
        if (this != &other) {
            release();
            device_ptr_ = other.device_ptr_;
            size_ = other.size_;
            generation_ = other.generation_;
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

    // Uploads still wait: callers' host buffers (temporaries) may not outlive an asynchronous
    // copy, and every training sample downloads its loss anyway (TD-390).
    void copy_from_host(const T* host_ptr, size_t count) {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        GPUManager::get_queue().memcpy(device_ptr_, host_ptr, count * sizeof(T)).wait();
    }

    void copy_to_host(T* host_ptr, size_t count) const {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        GPUManager::get_queue().memcpy(host_ptr, device_ptr_, count * sizeof(T)).wait();
    }
};

}  // namespace gpu
}  // namespace adai

#endif  // GPU_UTILS_SYCL_HPP
