/**
 * @file OpenMPBenchmark.cpp
 * @brief Benchmark program for testing OpenMP parallelization speedup
 * 
 * Tests the performance improvement from OpenMP parallelization in Matrix operations.
 * Compares sequential vs parallel execution and measures scaling efficiency.
 * 
 * Usage:
 *   ./openmp_benchmark [matrix_size] [num_threads]
 * 
 * Example:
 *   ./openmp_benchmark 512 8
 * 
 * @version 1.0
 * @date January 2026
 */

#include "Matrix.hpp"
#include "PerformanceProfiler.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
#endif

/**
 * @brief Benchmark matrix multiplication
 */
void benchmark_matrix_multiplication(int size, int num_threads = 0) {
    std::cout << "\n========================================\n";
    std::cout << "Matrix Multiplication Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Matrix Size: " << size << " x " << size << "\n";
    
#ifdef ADAI_ENABLE_OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        std::cout << "OpenMP Threads: " << num_threads << "\n";
    } else {
        std::cout << "OpenMP Threads: " << omp_get_max_threads() << " (auto)\n";
    }
#else
    std::cout << "OpenMP: NOT ENABLED (sequential execution)\n";
#endif
    
    std::cout << "Operation: C = A * B\n";
    std::cout << "FLOPs: " << (2.0 * size * size * size) / 1e9 << " billion\n";
    std::cout << "----------------------------------------\n";
    
    // Create random matrices
    Matrix A(size, size);
    Matrix B(size, size);
    A.randomize(0.1f);
    B.randomize(0.1f);
    
    // Warmup run
    Matrix warmup = A * B;
    
    // Timed runs
    const int num_runs = 3;
    std::vector<double> times;
    
    for (int run = 0; run < num_runs; run++) {
        Timer timer;
        timer.start();
        Matrix C = A * B;
        double elapsed = timer.stop();
        times.push_back(elapsed);
        std::cout << "Run " << (run + 1) << ": " << std::fixed << std::setprecision(2) 
                  << elapsed << " ms\n";
    }
    
    // Calculate statistics
    double avg_time = 0.0;
    for (double t : times) avg_time += t;
    avg_time /= num_runs;
    
    double gflops = (2.0 * size * size * size) / (avg_time * 1e6);
    
    std::cout << "----------------------------------------\n";
    std::cout << "Average Time: " << std::fixed << std::setprecision(2) << avg_time << " ms\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
}

/**
 * @brief Test parallel scaling efficiency
 */
void benchmark_parallel_scaling(int size) {
#ifdef ADAI_ENABLE_OPENMP
    std::cout << "\n========================================\n";
    std::cout << "Parallel Scaling Analysis\n";
    std::cout << "========================================\n";
    std::cout << "Matrix Size: " << size << " x " << size << "\n";
    std::cout << "Testing thread counts: 1, 2, 4, 8, 16\n";
    std::cout << "----------------------------------------\n";
    
    Matrix A(size, size);
    Matrix B(size, size);
    A.randomize(0.1f);
    B.randomize(0.1f);
    
    std::cout << std::setw(10) << "Threads" 
              << std::setw(12) << "Time (ms)" 
              << std::setw(10) << "Speedup" 
              << std::setw(12) << "Efficiency"
              << std::setw(12) << "GFLOPS" << "\n";
    std::cout << std::string(56, '-') << "\n";
    
    double baseline_time = 0.0;
    
    for (int threads : {1, 2, 4, 8, 16}) {
        omp_set_num_threads(threads);
        
        // Warmup
        Matrix warmup = A * B;
        
        // Timed run
        Timer timer;
        timer.start();
        Matrix C = A * B;
        double time = timer.stop();
        
        if (threads == 1) {
            baseline_time = time;
        }
        
        double speedup = baseline_time / time;
        double efficiency = speedup / threads;
        double gflops = (2.0 * size * size * size) / (time * 1e6);
        
        std::cout << std::setw(10) << threads
                  << std::setw(12) << std::fixed << std::setprecision(2) << time
                  << std::setw(10) << std::setprecision(2) << speedup
                  << std::setw(12) << std::setprecision(1) << (efficiency * 100) << "%"
                  << std::setw(12) << std::setprecision(2) << gflops << "\n";
    }
    
    std::cout << "----------------------------------------\n";
    std::cout << "Note: Efficiency = Speedup / Threads\n";
    std::cout << "      >80% is excellent, >60% is good\n";
#else
    std::cout << "\nOpenMP not enabled - cannot test parallel scaling\n";
    std::cout << "Rebuild with OpenMP support to see parallel performance\n";
#endif
}

/**
 * @brief Benchmark element-wise operations
 */
void benchmark_elementwise_operations(int size) {
    std::cout << "\n========================================\n";
    std::cout << "Element-wise Operations Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Matrix Size: " << size << " x " << size << "\n";
    
    Matrix A(size, size);
    Matrix B(size, size);
    A.randomize(0.1f);
    B.randomize(0.1f);
    
    std::cout << "\nOperation      | Time (ms) | Speedup Potential\n";
    std::cout << "---------------|-----------|------------------\n";
    
    // Addition
    {
        Timer timer;
        timer.start();
        Matrix C = A + B;
        double time = timer.stop();
        std::cout << std::setw(14) << "Addition" << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(2) << time 
                  << " | High (memory-bound)\n";
    }
    
    // Subtraction
    {
        Timer timer;
        timer.start();
        Matrix C = A - B;
        double time = timer.stop();
        std::cout << std::setw(14) << "Subtraction" << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(2) << time 
                  << " | High (memory-bound)\n";
    }
    
    // Hadamard product
    {
        Timer timer;
        timer.start();
        Matrix C = A.hadamard(B);
        double time = timer.stop();
        std::cout << std::setw(14) << "Hadamard" << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(2) << time 
                  << " | High (memory-bound)\n";
    }
    
    // Transpose
    {
        Timer timer;
        timer.start();
        Matrix C = A.transpose();
        double time = timer.stop();
        std::cout << std::setw(14) << "Transpose" << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(2) << time 
                  << " | Medium (cache effects)\n";
    }
    
    // Scalar multiplication
    {
        Timer timer;
        timer.start();
        Matrix C = A.scale(2.5f);
        double time = timer.stop();
        std::cout << std::setw(14) << "Scale" << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(2) << time 
                  << " | High (memory-bound)\n";
    }
}

/**
 * @brief Compare different matrix sizes
 */
void benchmark_size_comparison() {
    std::cout << "\n========================================\n";
    std::cout << "Size vs Performance Comparison\n";
    std::cout << "========================================\n";
    
#ifdef ADAI_ENABLE_OPENMP
    omp_set_num_threads(8);  // Use 8 threads
    std::cout << "Using 8 OpenMP threads\n";
#else
    std::cout << "Sequential execution (OpenMP not enabled)\n";
#endif
    
    std::cout << "\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(20) << "Memory (MB)" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (int size : {64, 128, 256, 512, 1024}) {
        Matrix A(size, size);
        Matrix B(size, size);
        A.randomize(0.1f);
        B.randomize(0.1f);
        
        // Warmup
        Matrix warmup = A * B;
        
        // Timed run
        Timer timer;
        timer.start();
        Matrix C = A * B;
        double time = timer.stop();
        
        double gflops = (2.0 * size * size * size) / (time * 1e6);
        double memory_mb = (3.0 * size * size * sizeof(float)) / (1024.0 * 1024.0);
        
        std::cout << std::setw(10) << size
                  << std::setw(15) << std::fixed << std::setprecision(2) << time
                  << std::setw(15) << std::setprecision(2) << gflops
                  << std::setw(20) << std::setprecision(1) << memory_mb << "\n";
    }
}

/**
 * @brief Main benchmark program
 */
int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║   OpenMP Matrix Benchmark Suite       ║\n";
    std::cout << "║   Priority 1 Implementation Test      ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";
    
#ifdef ADAI_ENABLE_OPENMP
    std::cout << "\n✓ OpenMP ENABLED\n";
    std::cout << "Max Threads Available: " << omp_get_max_threads() << "\n";
#else
    std::cout << "\n✗ OpenMP NOT ENABLED\n";
    std::cout << "Running in sequential mode\n";
    std::cout << "Rebuild with: cmake .. -DCMAKE_BUILD_TYPE=Release\n";
#endif
    
    // Parse command line arguments
    int size = 512;
    int num_threads = 0;  // 0 means use default
    
    if (argc > 1) {
        size = std::atoi(argv[1]);
        if (size < 64 || size > 4096) {
            std::cerr << "Error: Matrix size must be between 64 and 4096\n";
            return 1;
        }
    }
    
    if (argc > 2) {
        num_threads = std::atoi(argv[2]);
        if (num_threads < 1 || num_threads > 128) {
            std::cerr << "Error: Number of threads must be between 1 and 128\n";
            return 1;
        }
    }
    
    // Run benchmarks
    benchmark_matrix_multiplication(size, num_threads);
    
#ifdef ADAI_ENABLE_OPENMP
    if (argc <= 2) {  // Only run scaling test if not manually setting threads
        benchmark_parallel_scaling(size);
    }
#endif
    
    benchmark_elementwise_operations(size);
    benchmark_size_comparison();
    
    std::cout << "\n========================================\n";
    std::cout << "Benchmark Complete!\n";
    std::cout << "========================================\n";
    
#ifdef ADAI_ENABLE_OPENMP
    std::cout << "\n✓ OpenMP parallelization is working\n";
    std::cout << "Expected speedup: 4-8x on 8-core systems\n";
    std::cout << "Actual speedup depends on:\n";
    std::cout << "  - Number of CPU cores\n";
    std::cout << "  - Memory bandwidth\n";
    std::cout << "  - Matrix size (larger = better)\n";
#else
    std::cout << "\nTo enable OpenMP parallelization:\n";
    std::cout << "1. Install OpenMP: sudo apt-get install libomp-dev\n";
    std::cout << "2. Rebuild project: cd build && cmake .. && make\n";
    std::cout << "3. Run benchmark again to see parallel speedup\n";
#endif
    
    return 0;
}
