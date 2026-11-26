#ifndef GPU_UTILS_SYCL_HPP
#define GPU_UTILS_SYCL_HPP

// @adai-status: beta        (capped by TD-383 — see TECHNICAL_DEBT.md)
// @adai-version: 0.14.0
// @adai-reviewed: 2026-11-26

#include <algorithm>
#include "../TransferCounter.hpp"  // TD-590
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <unordered_map>
#include <vector>
#include "../DeviceSelection.hpp"
#include "../GPUDebugSync.hpp"   // TD-574
#include "../QueuePriority.hpp"  // TD-574

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
    /// and run before anything its next owner submits. Keyed by block size (pool_size()), ordered
    /// so a request can take a slightly larger block (TD-596); each carries its release stamp.
    struct CachedBlock {
        void* ptr;
        std::uint64_t stamp;  ///< release order, for least-recently-used eviction (TD-596)
    };
    inline static std::map<size_t, std::vector<CachedBlock>> free_blocks_;
    /// Bytes held in free_blocks_. Counts against the budget with allocated_bytes_, and is
    /// returned to USM (trim_pool_locked()) when an allocation would exceed the budget.
    inline static size_t cached_bytes_ = 0;
    /// TD-596: the size of every live block, so a block handed out for a smaller request goes back
    /// to the pool at its own size.
    inline static std::unordered_map<void*, size_t> live_block_size_;
    /// TD-596: the most cached_bytes_ may hold (GPU_POOL_CACHE_MB). Over it, the least recently
    /// released blocks are evicted down to three quarters of it. SIZE_MAX = no limit (before TD-596
    /// the pool grew until it filled the budget).
    inline static size_t cache_limit_bytes_ = size_t{1024} << 20;
    /// TD-596: evicted blocks wait here until a barrier queued after them completes (sycl::free()
    /// isn't ordered against the queue), then are freed without waiting.
    struct DeferredFree {
        sycl::event barrier;
        std::vector<void*> blocks;
        size_t bytes;
    };
    inline static std::vector<DeferredFree> deferred_frees_;
    /// Bytes in deferred_frees_: still device memory, so they count against the budget.
    inline static size_t deferred_bytes_ = 0;
    inline static std::uint64_t release_stamp_ = 0;
    /// TD-596: the largest allocated_bytes_ since reset_peak_memory().
    inline static size_t peak_allocated_bytes_ = 0;
    /// TD-596: a request may take a cached block up to this fraction larger (1/4 = 25%).
    static constexpr size_t kReuseSlackDivisor = 4;
    /// Bumped whenever the queue (and so the USM context) is replaced, so a block allocated
    /// before cleanup()/set_device() is never put back into the new pool.
    inline static unsigned pool_generation_ = 0;
    /// Guards the budget counters and the pool. Allocation and release run on whatever
    /// application thread owns the GPUMemory (TSan found a race on the counters when release ran
    /// in a host_task, before TD-390 removed those).
    inline static std::mutex memory_mutex_;
    /// Remembered from initialize() so set_device() can recreate the queue
    /// with the same scheduling priority (GPU_STRATEGY) after a device switch.
    inline static QueuePriority priority_ = QueuePriority::NORMAL;
    /// TD-574: set when a wait timed out (wait_for()). The queue still holds the work that never
    /// finished, so every later wait fails at once, device_usable() is false and cleanup() leaves
    /// the queue alone. A new queue (initialize(), set_device()) starts clear.
    inline static std::atomic<bool> stuck_{false};

    static std::string backend_name(const sycl::device& dev) {
        switch (dev.get_backend()) {
            case sycl::backend::ext_oneapi_level_zero:
                return "level_zero";
            case sycl::backend::opencl:
                return "opencl";
            default:
                return "other";
        }
    }

    // TD-391: one Arc card appears once per backend (Level Zero and OpenCL), so select_devices()
    // keeps one listing of it and numbers Level Zero first, whatever order the platforms come in.
    // ONEAPI_DEVICE_SELECTOR, if set, has already narrowed what the runtime lists here.
    static std::vector<sycl::device> enumerate_gpu_devices() {
        // TD-394: ADAI_SYCL_ALLOW_CPU_DEVICE=1 also lists CPU devices (oneAPI's OpenCL CPU
        // device), so the real kernels can run in tests on a machine with no GPU. Off by
        // default: production must never fall back to the CPU device silently.
        const char* allow_cpu = std::getenv("ADAI_SYCL_ALLOW_CPU_DEVICE");
        const auto type = (allow_cpu && allow_cpu[0] == '1') ? sycl::info::device_type::all
                                                             : sycl::info::device_type::gpu;
        std::vector<sycl::device> listed;
        std::vector<DeviceCandidate> candidates;
        for (auto& platform : sycl::platform::get_platforms()) {
            for (auto& dev : platform.get_devices(type)) {
                listed.push_back(dev);
                candidates.push_back({backend_name(dev), dev.get_info<sycl::info::device::vendor>(),
                                      dev.get_info<sycl::info::device::name>()});
            }
        }
        std::vector<sycl::device> gpus;
        for (std::size_t index : select_devices(candidates)) {
            gpus.push_back(listed[index]);
        }
        return gpus;
    }

    /**
     * @brief Build the ADAI queue with a scheduling-priority hint, mirroring
     *        the CUDA backend's stream priority selection.
     *
     * LOW    → background mode (GPU_STRATEGY=background): priority_low,
     *          yields to other GPU work.
     * NORMAL → GPU_STRATEGY=normal, the default: no priority hint (TD-574: the Arc B60's GuC
     *          resets low-priority compute contexts this one runs without).
     * HIGH   → full mode (GPU_STRATEGY=full): priority_high.
     *
     * priority_low/priority_high are hints (sycl_ext_oneapi_queue_priority);
     * backends that don't support them fall back to normal scheduling.
     */
    static sycl::queue make_queue(const sycl::device& dev, QueuePriority priority) {
        switch (priority) {
            case QueuePriority::LOW:
                return sycl::queue(
                    dev, sycl::property_list{sycl::property::queue::in_order{},
                                             sycl::ext::oneapi::property::queue::priority_low{}});
            case QueuePriority::NORMAL:
                return sycl::queue(dev, sycl::property_list{sycl::property::queue::in_order{}});
            case QueuePriority::HIGH:
                break;
        }
        return sycl::queue(dev, sycl::property_list{
                                    sycl::property::queue::in_order{},
                                    sycl::ext::oneapi::property::queue::priority_high{}});
    }

    static bool fits_locked(size_t bytes) {
        return max_memory_bytes_ == 0 ||
               (allocated_bytes_ + cached_bytes_ + deferred_bytes_ + bytes) <= max_memory_bytes_;
    }

    /// TD-596: frees every deferred batch whose barrier has completed. Never waits.
    static void free_completed_deferred_locked() {
        if (deferred_frees_.empty() || !queue_) {
            return;
        }
        auto done = [](const DeferredFree& d) {
            return d.barrier.get_info<sycl::info::event::command_execution_status>() ==
                   sycl::info::event_command_status::complete;
        };
        std::vector<DeferredFree> pending;
        for (auto& d : deferred_frees_) {
            if (!stuck_ && done(d)) {
                for (void* block : d.blocks) {
                    sycl::free(block, *queue_);
                }
                deferred_bytes_ -= d.bytes;
            } else {
                pending.push_back(std::move(d));
            }
        }
        deferred_frees_ = std::move(pending);
    }

    /// TD-596: evicts the least recently released cached blocks until cached_bytes_ is at most
    /// @p target. They are freed later, once a barrier queued now has completed (nothing waits).
    static void evict_cached_locked(size_t target) {
        if (cached_bytes_ <= target || !queue_) {
            return;
        }
        struct Entry {
            std::uint64_t stamp;
            size_t size;
            void* ptr;
        };
        std::vector<Entry> all;
        for (const auto& [size, blocks] : free_blocks_) {
            for (const auto& b : blocks) {
                all.push_back({b.stamp, size, b.ptr});
            }
        }
        std::sort(all.begin(), all.end(), [](const Entry& a, const Entry& b) { return a.stamp < b.stamp; });
        DeferredFree batch{sycl::event{}, {}, 0};
        for (const auto& e : all) {
            if (cached_bytes_ <= target) {
                break;
            }
            auto& blocks = free_blocks_[e.size];
            blocks.erase(std::find_if(blocks.begin(), blocks.end(),
                                      [&](const CachedBlock& b) { return b.ptr == e.ptr; }));
            if (blocks.empty()) {
                free_blocks_.erase(e.size);
            }
            cached_bytes_ -= e.size;
            batch.blocks.push_back(e.ptr);
            batch.bytes += e.size;
        }
        if (batch.blocks.empty()) {
            return;
        }
        if (stuck_) {  // TD-574: the stuck work may still use them; leave them to process exit
            return;
        }
        batch.barrier = queue_->ext_oneapi_submit_barrier();
        deferred_bytes_ += batch.bytes;
        deferred_frees_.push_back(std::move(batch));
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
        if (cached_bytes_ == 0 && deferred_bytes_ == 0) {
            return;
        }
        if (stuck_) {  // TD-574: blocks the stuck work may use can't be freed; drop the cache
            free_blocks_.clear();
            cached_bytes_ = 0;
            deferred_frees_.clear();
            deferred_bytes_ = 0;
            return;
        }
        wait_for_queue(false);
        for (auto& [size, blocks] : free_blocks_) {
            (void)size;
            for (const auto& block : blocks) {
                sycl::free(block.ptr, *queue_);
            }
        }
        free_blocks_.clear();
        cached_bytes_ = 0;
        free_completed_deferred_locked();  // TD-596: the wait completed every barrier
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
                           QueuePriority priority = QueuePriority::NORMAL) {
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
        priority_ = priority;
        auto& selected = gpus[device_id];

        queue_ = new sycl::queue(make_queue(selected, priority));
        stuck_ = false;

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
            peak_allocated_bytes_ = 0;
            live_block_size_.clear();
            free_blocks_.clear();  // TD-596: a stuck queue's trim leaves nothing to free later
            deferred_frees_.clear();
            cached_bytes_ = 0;
            deferred_bytes_ = 0;
            ++pool_generation_;
        }
        if (queue_) {
            // TD-574: a stuck queue would never finish; leave it (and its memory) to process exit.
            if (!stuck_) {
                queue_->wait();
                delete queue_;
            }
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
            live_block_size_.clear();
            ++pool_generation_;
        }
        if (queue_) {
            queue_->wait();
            delete queue_;
        }
        // Recreate with the same priority the queue was originally initialized
        // with (GPU_STRATEGY), matching CUDA's set_device() staying on the
        // stream priority chosen at GPUManager::initialize() time.
        queue_ = new sycl::queue(make_queue(gpus[device], priority_));
        stuck_ = false;
        current_device_ = device;
    }

    static sycl::queue& get_queue() {
        if (!initialized_) {
            throw std::runtime_error("GPU not initialized. Call GPUManager::initialize() first.");
        }
        return *queue_;
    }

    /// TD-574: called at the start of every kernel entry point. Records @p kernel as the latest GPU
    /// work; with ADAI_GPU_SYNC=kernel (or before every Nth operation with kernel:N) it first
    /// waits for the queue, so an error surfaces here, still attributed to the kernel before
    /// (describe_last_gpu_work()).
    static void trace_kernel(const char* kernel) {
        if (queue_ && wait_before_gpu_operation()) {
            wait_for_queue();
        }
        note_gpu_kernel(kernel);
    }

    /// TD-574: called where a training phase starts queuing work. With ADAI_GPU_SYNC=step or kernel
    /// it first waits for the previous phase's work, so an error is attributed to that phase.
    static void debug_sync_point(const char* phase) {
        if (gpu_debug_settings().sync != DebugSyncMode::Off && queue_) {
            wait_for_queue();
        }
        note_gpu_phase(phase);
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

    /// Bytes held in the pool for reuse (TD-390), not counted by get_used_memory_bytes(). With
    /// TD-596's evicted blocks that aren't freed yet.
    static size_t get_cached_memory_bytes() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return cached_bytes_ + deferred_bytes_;
    }

    /// TD-596: the most the pool keeps cached (GPU_POOL_CACHE_MB); SIZE_MAX = no limit.
    static size_t get_pool_cache_limit_bytes() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return cache_limit_bytes_;
    }
    static void set_pool_cache_limit_bytes(size_t bytes) {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        cache_limit_bytes_ = bytes;
        evict_cached_locked(bytes);
    }

    /// TD-596: the largest get_used_memory_bytes() since the last reset_peak_memory().
    static size_t get_peak_used_memory_bytes() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return peak_allocated_bytes_;
    }
    static void reset_peak_memory() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        peak_allocated_bytes_ = allocated_bytes_;
    }

    /// TD-596: one line for the logs, "X MB in use (peak Y MB), Z MB cached (limit L MB)", then starts a new
    /// peak. The trainers log it once per epoch.
    static std::string memory_summary() {
        const auto mb = [](size_t b) { return std::to_string((b + (size_t{1} << 19)) >> 20); };
        const size_t limit = get_pool_cache_limit_bytes();
        std::string s = mb(get_used_memory_bytes()) + " MB in use (peak " + mb(get_peak_used_memory_bytes()) +
                        " MB), " + mb(get_cached_memory_bytes()) + " MB cached (limit " +
                        (limit == SIZE_MAX ? std::string("none") : mb(limit) + " MB") + ")";
        reset_peak_memory();
        return s;
    }

    /// Size a pooled allocation of `bytes` is rounded up to: 512-byte steps up to 64 KiB, then an
    /// eighth of the next power of two (at most 12.5% over), so recurring shapes share blocks.
    /// TD-596: the fine steps went up to 1 MiB, so every sequence length gave new sizes.
    static size_t pool_size(size_t bytes) {
        if (bytes <= (size_t{1} << 16)) {
            return (bytes + 511) & ~size_t{511};
        }
        size_t pow2 = size_t{1} << 16;
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
    /// one. Counts it in the budget; trims the pool and retries before refusing. Zero bytes give
    /// nullptr, as cudaMalloc(0) does (TD-392); release_block() ignores it.
    static void* allocate_block(size_t bytes) {
        const size_t size = pool_size(bytes);
        std::lock_guard<std::mutex> lock(memory_mutex_);
        if (!initialized_) {
            throw std::runtime_error("GPU not initialized. Call GPUManager::initialize() first.");
        }
        // TD-392: malloc_device(0) returns nullptr, which read as a failed allocation, so an empty
        // GPUMemory threw only on SYCL.
        if (size == 0) {
            return nullptr;
        }
        free_completed_deferred_locked();
        // TD-596: the smallest cached block of at least this size and at most a quarter larger.
        for (auto it = free_blocks_.lower_bound(size);
             it != free_blocks_.end() && it->first <= size + size / kReuseSlackDivisor; ++it) {
            if (it->second.empty()) {
                continue;
            }
            const size_t got = it->first;
            void* block = it->second.back().ptr;
            it->second.pop_back();
            if (it->second.empty()) {
                free_blocks_.erase(it);
            }
            cached_bytes_ -= got;
            note_allocated_locked(block, got);
            return block;
        }
        if (!fits_locked(size)) {
            trim_pool_locked();
            if (!fits_locked(size)) {
                throw_budget_exceeded_locked(size);
            }
        }
        void* block = sycl::malloc_device(size, *queue_);
        if (!block) {
            throw std::runtime_error("sycl::malloc_device failed: allocation returned null");
        }
        note_allocated_locked(block, size);
        return block;
    }

    static void note_allocated_locked(void* block, size_t size) {
        allocated_bytes_ += size;
        peak_allocated_bytes_ = std::max(peak_allocated_bytes_, allocated_bytes_);
        live_block_size_[block] = size;
    }

    /// Takes back a block from allocate_block(). No host_task and no wait: the block goes to the
    /// pool, and its next owner's commands queue behind any still using it. A block from before
    /// the queue was replaced (cleanup()/set_device()) is dropped; its context reclaims it.
    static void release_block(void* block, size_t bytes, unsigned generation) {
        if (!block) {
            return;
        }
        std::lock_guard<std::mutex> lock(memory_mutex_);
        if (!initialized_ || generation != pool_generation_) {
            return;
        }
        size_t size = pool_size(bytes);
        if (auto it = live_block_size_.find(block); it != live_block_size_.end()) {
            size = it->second;  // TD-596: it may be larger than this request's pool size
            live_block_size_.erase(it);
        }
        allocated_bytes_ = (size <= allocated_bytes_) ? (allocated_bytes_ - size) : 0;
        free_blocks_[size].push_back({block, ++release_stamp_});
        cached_bytes_ += size;
        free_completed_deferred_locked();
        if (cached_bytes_ > cache_limit_bytes_) {  // TD-596
            evict_cached_locked(cache_limit_bytes_ - cache_limit_bytes_ / 4);
        }
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
        peak_allocated_bytes_ = std::max(peak_allocated_bytes_, allocated_bytes_);
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
                           "\n  Backend: " + backend_name(dev) + " (" +
                           dev.get_platform().get_info<sycl::info::platform::name>() + ")" +
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
            wait_for_queue(false);
        }
    }

    /// TD-574: waits for @p event like event.wait() (or wait_and_throw() when @p throw_async), but
    /// gives up after gpu_wait_timeout() seconds (0 = no limit): the queue is then marked stuck
    /// and this throws std::runtime_error, which callers handle as a lost device (TD-566). On the
    /// Arc, work that stops without an engine reset otherwise blocks the wait forever.
    static void wait_for(sycl::event event, bool throw_async = true) {
        if (stuck_) {
            throw std::runtime_error(stuck_message());
        }
        const unsigned timeout = gpu_wait_timeout();
        if (timeout != 0) {
            using clock = std::chrono::steady_clock;
            const auto start = clock::now();
            const auto deadline = start + std::chrono::seconds(timeout);
            auto pause = std::chrono::microseconds(10);
            while (event.get_info<sycl::info::event::command_execution_status>() !=
                   sycl::info::event_command_status::complete) {
                const auto now = clock::now();
                if (now >= deadline) {
                    stuck_ = true;
                    note_gpu_work_stuck();
                    throw std::runtime_error(stuck_message());
                }
                // Spin briefly (most waits are short), then back off to at most 1 ms per poll.
                if (now - start < std::chrono::microseconds(200)) {
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(pause);
                    pause = std::min(pause * 2, std::chrono::microseconds(1000));
                }
            }
        }
        if (throw_async) {
            event.wait_and_throw();
        } else {
            event.wait();
        }
    }

    /// TD-574: waits for everything queued so far, with wait_for()'s timeout.
    static void wait_for_queue(bool throw_async = true) {
        if (stuck_) {
            throw std::runtime_error(stuck_message());
        }
        wait_for(queue_->ext_oneapi_submit_barrier(), throw_async);
    }

    /// TD-574: whether a wait timed out on the current queue.
    static bool queue_stuck() {
        return stuck_;
    }

    /// Clears the stuck mark (tests only, after the stuck work was let finish).
    static void clear_stuck_for_testing() {
        stuck_ = false;
        clear_gpu_work_stuck_for_testing();
    }

    static std::string stuck_message() {
        return "GPU work did not finish within " + std::to_string(gpu_wait_timeout()) +
               " s (GPU_WAIT_TIMEOUT_SECONDS, TD-574): the device is treated as lost";
    }

    /// TD-566: whether the device still runs work: one small allocation, a fill, a copy back and a
    /// wait. After a device loss (a driver engine reset) every call fails, so the trainer uses this
    /// after a failed sample to tell a lost device from a bad sample. It allocates outside the pool
    /// and waits for the queue, so call it only after an error. TD-574: false at once when a wait
    /// timed out, and its own waits time out the same way.
    static bool device_usable() noexcept {
        if (!is_available() || !queue_ || stuck_) {
            return false;
        }
        int* probe = nullptr;
        bool ok = false;
        try {
            probe = sycl::malloc_device<int>(1, *queue_);
            if (probe) {
                int host = 1;
                wait_for(queue_->memset(probe, 0, sizeof(int)));
                wait_for(queue_->memcpy(&host, probe, sizeof(int)));
                ok = host == 0;
            }
        } catch (...) {
            ok = false;
        }
        if (probe && !stuck_) {
            try {
                sycl::free(probe, *queue_);
            } catch (...) {
            }
        }
        return ok;
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
        if (count == 0) {
            return;  // TD-392: an empty GPUMemory has no device pointer
        }
        GPUManager::trace_kernel("upload");  // TD-574
        GPUManager::wait_for(
            GPUManager::get_queue().memcpy(device_ptr_, host_ptr, count * sizeof(T)), false);
        note_upload(count * sizeof(T));  // TD-590
    }

    void copy_to_host(T* host_ptr, size_t count) const {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        if (count == 0) {
            return;
        }
        GPUManager::trace_kernel("download");  // TD-574
        GPUManager::wait_for(
            GPUManager::get_queue().memcpy(host_ptr, device_ptr_, count * sizeof(T)), false);
        note_download(count * sizeof(T));  // TD-590
    }
};

}  // namespace gpu
}  // namespace adai

#endif  // GPU_UTILS_SYCL_HPP
