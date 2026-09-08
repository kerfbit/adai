#ifndef GPU_UTILS_SYCL_HPP
#define GPU_UTILS_SYCL_HPP

// @adai-status: beta        (capped by TD-041 — GPUManager/GPUMemory only exercised incidentally, no dedicated test; also unverified (no SYCL toolchain available to build it))
// @adai-version: 0.6.0
// @adai-reviewed: 2026-09-07


#include <cstddef>
#include <stdexcept>
#include <string>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/sycl.hpp>

namespace adai {
namespace gpu {

class GPUManager {
   private:
    inline static bool initialized_ = false;
    inline static int device_count_ = 0;
    inline static int current_device_ = -1;
    inline static sycl::queue* queue_ = nullptr;
    inline static size_t max_memory_bytes_ = 0;
    inline static size_t allocated_bytes_ = 0;
    /// Remembered from initialize() so set_device() can recreate the queue
    /// with the same scheduling priority (GPU_STRATEGY) after a device switch.
    inline static bool low_priority_ = true;

    static std::vector<sycl::device> enumerate_gpu_devices() {
        std::vector<sycl::device> gpus;
        for (auto& platform : sycl::platform::get_platforms()) {
            for (auto& dev : platform.get_devices(sycl::info::device_type::gpu)) {
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
        max_memory_bytes_ = static_cast<size_t>(static_cast<double>(total_bytes) * memory_fraction);
        allocated_bytes_ = 0;

        initialized_ = true;
        return true;
    }

    static void cleanup() {
        if (!initialized_)
            return;

        if (queue_) {
            queue_->wait();
            delete queue_;
            queue_ = nullptr;
        }

        allocated_bytes_ = 0;
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

    static void set_device(int device) {
        if (device < 0 || device >= device_count_) {
            throw std::out_of_range("Invalid device ID: " + std::to_string(device));
        }
        // Switching devices requires recreating the queue
        auto gpus = enumerate_gpu_devices();
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
        return max_memory_bytes_;
    }

    static size_t get_used_memory_bytes() {
        return allocated_bytes_;
    }

    static size_t get_available_memory_bytes() {
        return (max_memory_bytes_ > allocated_bytes_) ? (max_memory_bytes_ - allocated_bytes_) : 0;
    }

    static void reserve_memory(size_t bytes) {
        if (max_memory_bytes_ > 0 && (allocated_bytes_ + bytes) > max_memory_bytes_) {
            // GPUMemory defers its sycl::free()/release_memory() into a queued
            // host_task (see defer_free() below) to avoid racing in-flight
            // kernels, so allocated_bytes_ can lag behind reality under
            // sustained submission pressure. Drain the queue once to let any
            // already-queued frees actually execute before refusing.
            synchronize();
            if ((allocated_bytes_ + bytes) > max_memory_bytes_) {
                throw std::runtime_error(
                    "ADAI GPU memory budget exceeded: requested " +
                    std::to_string(bytes / (1024 * 1024)) + " MB, " +
                    std::to_string(get_available_memory_bytes() / (1024 * 1024)) +
                    " MB available of " + std::to_string(max_memory_bytes_ / (1024 * 1024)) +
                    " MB limit");
            }
        }
        allocated_bytes_ += bytes;
    }

    static void release_memory(size_t bytes) {
        allocated_bytes_ = (bytes <= allocated_bytes_) ? (allocated_bytes_ - bytes) : 0;
    }

    static std::string get_device_info(int device = -1) {
        if (device == -1)
            device = current_device_;

        auto gpus = enumerate_gpu_devices();
        if (device < 0 || device >= static_cast<int>(gpus.size()))
            return "Invalid device ID";

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

    // USM sycl::free() is a host-side call — it is NOT implicitly ordered
    // against kernels previously submitted to the (asynchronous) in-order
    // queue that may still be reading/writing this pointer. Freeing directly
    // from a destructor races those kernels: the C++ object's lifetime ends
    // as soon as its scope exits, which can be before the device has actually
    // executed the last command that touches it. Submitting the free as a
    // host_task on the same in-order queue instead guarantees it only runs
    // once every previously-submitted command on that queue has completed on
    // the device, while keeping the calling thread non-blocking (no .wait()).
    static void defer_free(T* ptr, size_t bytes) {
        if (!ptr) {
            return;
        }
        GPUManager::get_queue().submit([ptr, bytes](sycl::handler& cgh) {
            cgh.host_task([ptr, bytes]() {
                sycl::free(ptr, GPUManager::get_queue());
                GPUManager::release_memory(bytes);
            });
        });
    }

   public:
    explicit GPUMemory(size_t count) : size_(count) {
        const size_t bytes = count * sizeof(T);
        GPUManager::reserve_memory(bytes);
        try {
            device_ptr_ = sycl::malloc_device<T>(count, GPUManager::get_queue());
        } catch (...) {
            GPUManager::release_memory(bytes);
            throw;
        }
        if (!device_ptr_) {
            GPUManager::release_memory(bytes);
            throw std::runtime_error("sycl::malloc_device failed: allocation returned null");
        }
    }

    ~GPUMemory() {
        defer_free(device_ptr_, size_ * sizeof(T));
    }

    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;

    GPUMemory(GPUMemory&& other) noexcept : device_ptr_(other.device_ptr_), size_(other.size_) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }

    GPUMemory& operator=(GPUMemory&& other) noexcept {
        if (this != &other) {
            defer_free(device_ptr_, size_ * sizeof(T));
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
