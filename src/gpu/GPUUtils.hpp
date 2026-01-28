#ifndef GPU_UTILS_HPP
#define GPU_UTILS_HPP

#include <string>
#include <stdexcept>

#ifdef ADAI_ENABLE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                cudaGetErrorString(error) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error: ") + \
                std::to_string(status) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

namespace adai {
namespace gpu {

/**
 * @brief GPU device information and management
 */
class GPUManager {
private:
    static bool initialized_;
    static int device_count_;
    static int current_device_;
    static cublasHandle_t cublas_handle_;
    
public:
    /**
     * @brief Initialize GPU subsystem
     * @throws std::runtime_error if GPU initialization fails
     */
    static void initialize() {
        if (initialized_) return;
        
        CUDA_CHECK(cudaGetDeviceCount(&device_count_));
        if (device_count_ == 0) {
            throw std::runtime_error("No CUDA-capable GPU devices found");
        }
        
        CUDA_CHECK(cudaSetDevice(0));
        current_device_ = 0;
        
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        
        initialized_ = true;
    }
    
    /**
     * @brief Cleanup GPU resources
     */
    static void cleanup() {
        if (!initialized_) return;
        
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
        
        cudaDeviceReset();
        initialized_ = false;
    }
    
    /**
     * @brief Check if GPU is available and initialized
     */
    static bool is_available() {
        return initialized_ && device_count_ > 0;
    }
    
    /**
     * @brief Get number of available GPU devices
     */
    static int device_count() {
        return device_count_;
    }
    
    /**
     * @brief Get current GPU device ID
     */
    static int current_device() {
        return current_device_;
    }
    
    /**
     * @brief Set active GPU device
     * @param device Device ID to set as active
     */
    static void set_device(int device) {
        if (device < 0 || device >= device_count_) {
            throw std::out_of_range("Invalid device ID: " + std::to_string(device));
        }
        CUDA_CHECK(cudaSetDevice(device));
        current_device_ = device;
    }
    
    /**
     * @brief Get cuBLAS handle for operations
     */
    static cublasHandle_t get_cublas_handle() {
        if (!initialized_) {
            throw std::runtime_error("GPU not initialized. Call GPUManager::initialize() first.");
        }
        return cublas_handle_;
    }
    
    /**
     * @brief Get GPU device properties
     */
    static std::string get_device_info(int device = -1) {
        if (device == -1) device = current_device_;
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        return std::string("Device ") + std::to_string(device) + ": " + prop.name +
               "\n  Compute Capability: " + std::to_string(prop.major) + "." + std::to_string(prop.minor) +
               "\n  Total Global Memory: " + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB" +
               "\n  Multiprocessors: " + std::to_string(prop.multiProcessorCount) +
               "\n  Max Threads per Block: " + std::to_string(prop.maxThreadsPerBlock);
    }
    
    /**
     * @brief Synchronize GPU (wait for all operations to complete)
     */
    static void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

// Initialize static members
bool GPUManager::initialized_ = false;
int GPUManager::device_count_ = 0;
int GPUManager::current_device_ = -1;
cublasHandle_t GPUManager::cublas_handle_ = nullptr;

/**
 * @brief RAII wrapper for GPU memory
 */
template<typename T>
class GPUMemory {
private:
    T* device_ptr_;
    size_t size_;
    
public:
    GPUMemory(size_t count) : size_(count) {
        CUDA_CHECK(cudaMalloc(&device_ptr_, count * sizeof(T)));
    }
    
    ~GPUMemory() {
        if (device_ptr_) {
            cudaFree(device_ptr_);
        }
    }
    
    // Disable copy
    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;
    
    // Enable move
    GPUMemory(GPUMemory&& other) noexcept 
        : device_ptr_(other.device_ptr_), size_(other.size_) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    
    GPUMemory& operator=(GPUMemory&& other) noexcept {
        if (this != &other) {
            if (device_ptr_) cudaFree(device_ptr_);
            device_ptr_ = other.device_ptr_;
            size_ = other.size_;
            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() { return device_ptr_; }
    const T* get() const { return device_ptr_; }
    size_t size() const { return size_; }
    
    void copy_from_host(const T* host_ptr, size_t count) {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        CUDA_CHECK(cudaMemcpy(device_ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* host_ptr, size_t count) const {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

} // namespace gpu
} // namespace adai

#else // !ADAI_ENABLE_GPU

// Stub implementations when GPU is disabled
namespace adai {
namespace gpu {

class GPUManager {
public:
    static void initialize() {
        throw std::runtime_error("GPU support not compiled. Rebuild with -DENABLE_GPU=ON");
    }
    static void cleanup() {}
    static bool is_available() { return false; }
    static int device_count() { return 0; }
    static int current_device() { return -1; }
    static void set_device(int) {
        throw std::runtime_error("GPU support not compiled");
    }
    static std::string get_device_info(int = -1) {
        return "GPU support not compiled";
    }
    static void synchronize() {}
};

} // namespace gpu
} // namespace adai

#endif // ADAI_ENABLE_GPU

#endif // GPU_UTILS_HPP
