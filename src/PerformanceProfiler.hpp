#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

/**
 * Performance Profiling Utilities
 *
 * Provides high-resolution timing and profiling tools for measuring
 * inference latency, identifying bottlenecks, and validating optimizations.
 *
 * Features:
 * - Microsecond-precision timing
 * - Named profiling sections
 * - Nested timing support
 * - Statistical analysis (min, max, mean, median)
 * - Comparison reports
 */

/**
 * Simple timer using high-resolution clock
 */
class Timer {
   private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint start_time;
    bool is_running;

   public:
    Timer() : is_running(false) {}

    /**
     * Start the timer
     */
    void start() {
        start_time = Clock::now();
        is_running = true;
    }

    /**
     * Stop the timer and return elapsed time in milliseconds
     */
    double stop() {
        if (!is_running) {
            return 0.0;
        }
        auto end_time = Clock::now();
        is_running = false;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }

    /**
     * Get elapsed time without stopping (in milliseconds)
     */
    double elapsed() const {
        if (!is_running) {
            return 0.0;
        }
        auto current_time = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);
        return duration.count() / 1000.0;
    }

    /**
     * Reset the timer
     */
    void reset() {
        is_running = false;
    }
};

/**
 * RAII-style scoped timer
 * Automatically starts on construction, stops and reports on destruction
 */
class ScopedTimer {
   private:
    std::string name;
    Timer timer;

   public:
    explicit ScopedTimer(const std::string& name) : name(name) {
        timer.start();
    }

    ~ScopedTimer() {
        double elapsed = timer.stop();
        std::cout << "[" << name << "] " << std::fixed << std::setprecision(2) 
                  << elapsed << " ms" << std::endl;
    }
};

/**
 * Profiling statistics for a named section
 */
struct ProfileStats {
    std::string name;
    std::vector<double> timings;  // All recorded timings (ms)
    int call_count;
    double total_time;
    double min_time;
    double max_time;
    double mean_time;
    double median_time;

    ProfileStats() : call_count(0), total_time(0.0), min_time(0.0), 
                    max_time(0.0), mean_time(0.0), median_time(0.0) {}

    explicit ProfileStats(const std::string& n) : name(n), call_count(0), total_time(0.0),
                                                  min_time(0.0), max_time(0.0), 
                                                  mean_time(0.0), median_time(0.0) {}

    /**
     * Add a timing measurement
     */
    void add_timing(double time_ms) {
        timings.push_back(time_ms);
        call_count++;
        total_time += time_ms;

        if (call_count == 1 || time_ms < min_time) {
            min_time = time_ms;
        }
        if (call_count == 1 || time_ms > max_time) {
            max_time = time_ms;
        }

        mean_time = total_time / call_count;
    }

    /**
     * Compute median (requires sorting timings)
     */
    void compute_median() {
        if (timings.empty()) {
            median_time = 0.0;
            return;
        }

        std::vector<double> sorted_timings = timings;
        std::sort(sorted_timings.begin(), sorted_timings.end());

        size_t n = sorted_timings.size();
        if (n % 2 == 0) {
            median_time = (sorted_timings[n / 2 - 1] + sorted_timings[n / 2]) / 2.0;
        } else {
            median_time = sorted_timings[n / 2];
        }
    }

    /**
     * Get percentile value (0-100)
     */
    double get_percentile(double p) const {
        if (timings.empty()) {
            return 0.0;
        }

        std::vector<double> sorted_timings = timings;
        std::sort(sorted_timings.begin(), sorted_timings.end());

        size_t index = static_cast<size_t>((p / 100.0) * (sorted_timings.size() - 1));
        return sorted_timings[index];
    }

    /**
     * Print statistics
     */
    void print() const {
        std::cout << "Profile: " << name << std::endl;
        std::cout << "  Calls: " << call_count << std::endl;
        std::cout << "  Total: " << std::fixed << std::setprecision(2) 
                  << total_time << " ms" << std::endl;
        std::cout << "  Mean:  " << std::fixed << std::setprecision(2) 
                  << mean_time << " ms" << std::endl;
        std::cout << "  Median: " << std::fixed << std::setprecision(2) 
                  << median_time << " ms" << std::endl;
        std::cout << "  Min:   " << std::fixed << std::setprecision(2) 
                  << min_time << " ms" << std::endl;
        std::cout << "  Max:   " << std::fixed << std::setprecision(2) 
                  << max_time << " ms" << std::endl;
        std::cout << "  P95:   " << std::fixed << std::setprecision(2) 
                  << get_percentile(95.0) << " ms" << std::endl;
        std::cout << "  P99:   " << std::fixed << std::setprecision(2) 
                  << get_percentile(99.0) << " ms" << std::endl;
    }
};

/**
 * Profiler for tracking multiple named sections
 */
class Profiler {
   private:
    std::map<std::string, ProfileStats> profiles;
    std::map<std::string, Timer> active_timers;

   public:
    /**
     * Start timing a named section
     */
    void start(const std::string& name) {
        if (profiles.find(name) == profiles.end()) {
            profiles[name] = ProfileStats(name);
        }
        active_timers[name].start();
    }

    /**
     * Stop timing a named section
     */
    void stop(const std::string& name) {
        if (active_timers.find(name) == active_timers.end()) {
            std::cerr << "Warning: Timer '" << name << "' was never started" << std::endl;
            return;
        }

        double elapsed = active_timers[name].stop();
        profiles[name].add_timing(elapsed);
    }

    /**
     * Get statistics for a specific profile
     */
    ProfileStats get_stats(const std::string& name) {
        if (profiles.find(name) == profiles.end()) {
            return ProfileStats(name);
        }
        
        ProfileStats& stats = profiles[name];
        stats.compute_median();
        return stats;
    }

    /**
     * Print all profiles
     */
    void print_all() {
        std::cout << "\n=== Profiling Results ===" << std::endl;
        for (auto& pair : profiles) {
            pair.second.compute_median();
            std::cout << "\n";
            pair.second.print();
        }
        std::cout << "========================\n" << std::endl;
    }

    /**
     * Reset all profiles
     */
    void reset() {
        profiles.clear();
        active_timers.clear();
    }

    /**
     * Compare two profiling runs
     */
    static void compare(const ProfileStats& baseline, const ProfileStats& optimized) {
        std::cout << "\n=== Performance Comparison ===" << std::endl;
        std::cout << "Profile: " << baseline.name << std::endl;
        std::cout << "\nBaseline:" << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(2) 
                  << baseline.mean_time << " ms" << std::endl;
        std::cout << "  Median: " << std::fixed << std::setprecision(2) 
                  << baseline.median_time << " ms" << std::endl;

        std::cout << "\nOptimized:" << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(2) 
                  << optimized.mean_time << " ms" << std::endl;
        std::cout << "  Median: " << std::fixed << std::setprecision(2) 
                  << optimized.median_time << " ms" << std::endl;

        if (baseline.mean_time > 0) {
            double speedup = baseline.mean_time / optimized.mean_time;
            double improvement = ((baseline.mean_time - optimized.mean_time) / baseline.mean_time) * 100.0;
            
            std::cout << "\nImprovement:" << std::endl;
            std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                      << speedup << "x" << std::endl;
            std::cout << "  Improvement: " << std::fixed << std::setprecision(1) 
                      << improvement << "%" << std::endl;
            std::cout << "  Time saved: " << std::fixed << std::setprecision(2)
                      << (baseline.mean_time - optimized.mean_time) << " ms" << std::endl;
        }
        std::cout << "==============================\n" << std::endl;
    }
};

/**
 * Macro for easy profiling of code blocks
 */
#define PROFILE_SCOPE(profiler, name) \
    profiler.start(name); \
    auto __profiler_guard_##name = [&]() { profiler.stop(name); return 0; }(); \
    (void)__profiler_guard_##name;

/**
 * Benchmark runner for comparing implementations
 */
class Benchmark {
   public:
    /**
     * Run a benchmark function multiple times and return statistics
     */
    template <typename Func>
    static ProfileStats run(const std::string& name, Func func, int iterations = 100,
                           int warmup_iterations = 10) {
        ProfileStats stats(name);

        // Warmup
        for (int i = 0; i < warmup_iterations; ++i) {
            func();
        }

        // Benchmark
        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            timer.start();
            func();
            double elapsed = timer.stop();
            stats.add_timing(elapsed);
        }

        stats.compute_median();
        return stats;
    }

    /**
     * Compare two implementations
     */
    template <typename FuncA, typename FuncB>
    static void compare(const std::string& name_a, FuncA func_a,
                       const std::string& name_b, FuncB func_b,
                       int iterations = 100, int warmup = 10) {
        std::cout << "\n=== Benchmarking ===" << std::endl;
        std::cout << "Running " << iterations << " iterations (after " 
                  << warmup << " warmup)...\n" << std::endl;

        ProfileStats stats_a = run(name_a, func_a, iterations, warmup);
        ProfileStats stats_b = run(name_b, func_b, iterations, warmup);

        Profiler::compare(stats_a, stats_b);
    }
};

