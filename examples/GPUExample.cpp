#include "Matrix.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <functional>

using namespace std;
using namespace std::chrono;

/**
 * GPU Acceleration Example
 * 
 * Demonstrates the optional GPU acceleration features in the ADAI library.
 * This example compares CPU vs GPU performance for common matrix operations.
 * 
 * To compile with GPU support:
 *   cmake -DENABLE_GPU=ON ..
 *   make
 * 
 * To compile without GPU support (CPU only):
 *   cmake -DENABLE_GPU=OFF ..
 *   make
 */

void print_separator() {
    cout << string(80, '=') << endl;
}

void benchmark_operation(const string& name, 
                        const Matrix& a, 
                        const Matrix& b,
                        function<Matrix(const Matrix&, const Matrix&)> cpu_op,
                        function<Matrix(const Matrix&, const Matrix&)> gpu_op) {
    cout << "\n" << name << ":" << endl;
    
    // CPU timing
    auto cpu_start = high_resolution_clock::now();
    Matrix cpu_result = cpu_op(a, b);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start).count();
    
    cout << "  CPU: " << cpu_duration << " μs" << endl;
    
#ifdef ADAI_ENABLE_GPU
    if (Matrix::gpu_available()) {
        // GPU timing
        auto gpu_start = high_resolution_clock::now();
        Matrix gpu_result = gpu_op(a, b);
        auto gpu_end = high_resolution_clock::now();
        auto gpu_duration = duration_cast<microseconds>(gpu_end - gpu_start).count();
        
        cout << "  GPU: " << gpu_duration << " μs" << endl;
        
        // Calculate speedup
        if (gpu_duration > 0) {
            double speedup = static_cast<double>(cpu_duration) / gpu_duration;
            cout << "  Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
        }
        
        // Verify results match (sample check)
        bool match = true;
        for (int i = 0; i < min(5, a.rows); i++) {
            for (int j = 0; j < min(5, cpu_result.cols); j++) {
                if (abs(cpu_result(i, j) - gpu_result(i, j)) > 1e-3) {
                    match = false;
                    break;
                }
            }
            if (!match) break;
        }
        cout << "  Results match: " << (match ? "YES" : "NO") << endl;
    }
#else
    cout << "  GPU: Not compiled (rebuild with -DENABLE_GPU=ON)" << endl;
#endif
}

int main() {
    print_separator();
    cout << "GPU Acceleration Example - ADAI Library" << endl;
    print_separator();
    
#ifdef ADAI_ENABLE_GPU
    cout << "\nGPU support: COMPILED" << endl;
    
    // Initialize GPU
    try {
        Matrix::gpu_initialize();
        cout << "GPU initialization: SUCCESS" << endl;
        cout << "\nGPU Information:\n" << Matrix::gpu_info() << endl;
    } catch (const exception& e) {
        cout << "GPU initialization: FAILED (" << e.what() << ")" << endl;
        cout << "\nRunning CPU-only tests...\n" << endl;
    }
#else
    cout << "\nGPU support: NOT COMPILED" << endl;
    cout << "To enable GPU support, rebuild with: cmake -DENABLE_GPU=ON .." << endl;
    cout << "\nRunning CPU-only tests...\n" << endl;
#endif
    
    // Create test matrices
    const int size = 512;
    cout << "\nCreating test matrices (" << size << "x" << size << ")..." << endl;
    
    Matrix A(size, size);
    Matrix B(size, size);
    
    A.randomize(1.0f);
    B.randomize(1.0f);
    
    cout << "Matrices created." << endl;
    
    print_separator();
    cout << "PERFORMANCE BENCHMARKS" << endl;
    print_separator();
    
    // 1. Matrix Multiplication
    benchmark_operation(
        "Matrix Multiplication (A * B)",
        A, B,
        [](const Matrix& a, const Matrix& b) { return a * b; },
#ifdef ADAI_ENABLE_GPU
        [](const Matrix& a, const Matrix& b) { return a.multiply_gpu(b); }
#else
        [](const Matrix& a, const Matrix& b) { return a * b; }
#endif
    );
    
    // 2. Matrix Addition
    benchmark_operation(
        "Matrix Addition (A + B)",
        A, B,
        [](const Matrix& a, const Matrix& b) { return a + b; },
#ifdef ADAI_ENABLE_GPU
        [](const Matrix& a, const Matrix& b) { return a.add_gpu(b); }
#else
        [](const Matrix& a, const Matrix& b) { return a + b; }
#endif
    );
    
    // 3. Element-wise Multiplication
    benchmark_operation(
        "Element-wise Multiplication (Hadamard)",
        A, B,
        [](const Matrix& a, const Matrix& b) { return a.hadamard(b); },
#ifdef ADAI_ENABLE_GPU
        [](const Matrix& a, const Matrix& b) { return a.hadamard_gpu(b); }
#else
        [](const Matrix& a, const Matrix& b) { return a.hadamard(b); }
#endif
    );
    
    // 4. Matrix Transpose
    cout << "\nMatrix Transpose:" << endl;
    auto cpu_start = high_resolution_clock::now();
    Matrix cpu_transpose = A.transpose();
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start).count();
    cout << "  CPU: " << cpu_duration << " μs" << endl;
    
#ifdef ADAI_ENABLE_GPU
    if (Matrix::gpu_available()) {
        auto gpu_start = high_resolution_clock::now();
        Matrix gpu_transpose = A.transpose_gpu();
        auto gpu_end = high_resolution_clock::now();
        auto gpu_duration = duration_cast<microseconds>(gpu_end - gpu_start).count();
        cout << "  GPU: " << gpu_duration << " μs" << endl;
        
        if (gpu_duration > 0) {
            double speedup = static_cast<double>(cpu_duration) / gpu_duration;
            cout << "  Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
        }
    }
#else
    cout << "  GPU: Not compiled" << endl;
#endif
    
    // 5. Scalar Multiplication
    cout << "\nScalar Multiplication (A * 2.5):" << endl;
    cpu_start = high_resolution_clock::now();
    Matrix cpu_scaled = A.scale(2.5f);
    cpu_end = high_resolution_clock::now();
    cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start).count();
    cout << "  CPU: " << cpu_duration << " μs" << endl;
    
#ifdef ADAI_ENABLE_GPU
    if (Matrix::gpu_available()) {
        auto gpu_start = high_resolution_clock::now();
        Matrix gpu_scaled = A.scale_gpu(2.5f);
        auto gpu_end = high_resolution_clock::now();
        auto gpu_duration = duration_cast<microseconds>(gpu_end - gpu_start).count();
        cout << "  GPU: " << gpu_duration << " μs" << endl;
        
        if (gpu_duration > 0) {
            double speedup = static_cast<double>(cpu_duration) / gpu_duration;
            cout << "  Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
        }
    }
#else
    cout << "  GPU: Not compiled" << endl;
#endif
    
    print_separator();
    cout << "SUMMARY" << endl;
    print_separator();
    
#ifdef ADAI_ENABLE_GPU
    if (Matrix::gpu_available()) {
        cout << "\nGPU acceleration is available and working!" << endl;
        cout << "For large matrices, GPU operations can be significantly faster." << endl;
        cout << "\nNote: Small matrices may not show speedup due to transfer overhead." << endl;
        cout << "GPU acceleration is most beneficial for:" << endl;
        cout << "  - Large matrix operations (1000x1000 and above)" << endl;
        cout << "  - Batch processing of multiple matrices" << endl;
        cout << "  - Training deep neural networks" << endl;
    } else {
        cout << "\nGPU is compiled but not available on this system." << endl;
        cout << "Check that CUDA drivers are installed and a GPU is present." << endl;
    }
#else
    cout << "\nGPU support not compiled." << endl;
    cout << "To enable GPU acceleration:" << endl;
    cout << "  1. Install CUDA Toolkit (11.0 or later)" << endl;
    cout << "  2. Rebuild with: cmake -DENABLE_GPU=ON .." << endl;
    cout << "  3. Run this example again" << endl;
#endif
    
    print_separator();
    
#ifdef ADAI_ENABLE_GPU
    // Cleanup
    if (Matrix::gpu_available()) {
        Matrix::gpu_cleanup();
        cout << "\nGPU resources cleaned up." << endl;
    }
#endif
    
    return 0;
}
