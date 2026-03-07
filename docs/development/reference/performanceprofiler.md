# PerformanceProfiler API Reference

**Module:** `PerformanceProfiler.hpp`
**Purpose:** High-resolution timing and profiling tools for performance analysis
**Use Case:** Measure latency, identify bottlenecks, validate optimizations

---

## Overview

The PerformanceProfiler provides a comprehensive suite of timing and profiling utilities for measuring and analyzing code performance. It offers microsecond-precision timing, statistical analysis, and automated benchmarking capabilities.

### Why Use Performance Profiling?

Before Optimization:

```text
"My code is slow, but I don't know where the bottleneck is..."
```

With PerformanceProfiler:

```text
Profile: forward_pass
  Mean: 45.23 ms

Profile: attention
  Mean: 38.12 ms  ← 84% of total time! Optimize this!

Profile: feedforward
  Mean: 7.11 ms
```

Key Benefits:

- Identify performance bottlenecks
- Validate optimization effectiveness
- Track performance regressions
- Compare different implementations
- Production performance monitoring

---

## Table of Contents

1. [Core Classes](#core-classes)
   - [Timer](#timer)
   - [ScopedTimer](#scopedtimer)
   - [ProfileStats](#profilestats)
   - [Profiler](#profiler)
   - [Benchmark](#benchmark)
2. [Usage Patterns](#usage-patterns)
3. [Statistical Analysis](#statistical-analysis)
4. [Benchmarking](#benchmarking)
5. [Best Practices](#best-practices)
6. [Advanced Topics](#advanced-topics)

---

## Core Classes {#core-classes}

### Timer

High-resolution timer for precise time measurements.

#### Class Definition

```cpp
class Timer {
public:
    void start();
    double stop();
    double elapsed() const;
    void reset();
};
```

#### Constructor

```cpp
Timer()
```

Creates a timer in stopped state.

Example:

```cpp
Timer timer;
```

---

#### Methods

##### `void start()`

Start the timer.

Example:

```cpp
Timer timer;
timer.start();
// ... code to measure ...
double elapsed = timer.stop();
```

---

##### `double stop()`

Stop the timer and return elapsed time.

**Returns:** Elapsed time in milliseconds (double precision)

Example:

```cpp
Timer timer;
timer.start();

process_data();

double elapsed = timer.stop();
std::cout << "Processing took " << elapsed << " ms" << std::endl;
```

---

##### `double elapsed() const`

Get current elapsed time without stopping the timer.

**Returns:** Elapsed time in milliseconds

Example:

```cpp
Timer timer;
timer.start();

for (int i = 0; i < 100; ++i) {
    process_item(i);

    // Check progress without stopping timer
    if (i % 10 == 0) {
        std::cout << "Progress: " << timer.elapsed() << " ms" << std::endl;
    }
}

double total = timer.stop();
```

---

##### `void reset()`

Reset timer to stopped state.

Example:

```cpp
timer.reset();
```

---

### ScopedTimer

RAII-style timer that automatically measures and reports execution time of a scope.

#### Class Definition

```cpp
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();  // Automatically prints timing
};
```

#### Constructor

```cpp
explicit ScopedTimer(const std::string& name)
```

Creates a timer that starts immediately and prints results when destroyed.

Parameters:

- `name` - Name for the timed section (printed in output)

Example:

```cpp
void process_batch() {
    ScopedTimer timer("batch_processing");

    // Your code here...

}  // Timer automatically stops and prints: "[batch_processing] 42.35 ms"
```

**Use Case:** Quick timing of function scope.

---

### ProfileStats

Statistical analysis container for timing data.

#### Structure

```cpp
struct ProfileStats {
    std::string name;
    std::vector<double> timings;
    int call_count;
    double total_time;
    double min_time;
    double max_time;
    double mean_time;
    double median_time;
};
```

#### Fields

##### `name`

Name of the profiled section.

**Type:** `std::string`

##### `timings`

All recorded timing measurements.

**Type:** `std::vector<double>`

##### `call_count`

Number of times section was executed.

**Type:** `int`

##### `total_time`

Sum of all measurements.

**Type:** `double` (milliseconds)

##### `min_time`, `max_time`

Minimum and maximum recorded times.

**Type:** `double` (milliseconds)

##### `mean_time`

Average execution time.

**Type:** `double` (milliseconds)

##### `median_time`

Median execution time (requires calling `compute_median()`).

**Type:** `double` (milliseconds)

---

#### Methods

##### `void add_timing(double time_ms)`

Add a timing measurement.

Parameters:

- `time_ms` - Timing in milliseconds

Example:

```cpp
ProfileStats stats("my_function");
stats.add_timing(12.5);
stats.add_timing(13.2);
stats.add_timing(11.8);
```

---

##### `void compute_median()`

Compute the median time (call before accessing `median_time`).

Example:

```cpp
stats.compute_median();
std::cout << "Median: " << stats.median_time << " ms" << std::endl;
```

---

##### `double get_percentile(double p) const`

Get percentile value (0-100).

Parameters:

- `p` - Percentile (0.0 to 100.0)

**Returns:** Time value at the given percentile

Example:

```cpp
double p50 = stats.get_percentile(50.0);   // Median
double p95 = stats.get_percentile(95.0);   // 95th percentile
double p99 = stats.get_percentile(99.0);   // 99th percentile

std::cout << "95% of calls complete within " << p95 << " ms" << std::endl;
```

---

##### `void print() const`

Print comprehensive statistics.

Output Example:

```text
Profile: inference
  Calls: 100
  Total: 4523.45 ms
  Mean:  45.23 ms
  Median: 44.89 ms
  Min:   42.11 ms
  Max:   58.34 ms
  P95:   48.21 ms
  P99:   52.67 ms
```

Example:

```cpp
ProfileStats stats = profiler.get_stats("inference");
stats.print();
```

---

### Profiler

Multi-section profiling manager.

#### Class Definition

```cpp
class Profiler {
public:
    void start(const std::string& name);
    void stop(const std::string& name);
    ProfileStats get_stats(const std::string& name);
    void print_all();
    void reset();
    static void compare(const ProfileStats& baseline, const ProfileStats& optimized);
};
```

#### Constructor

```cpp
Profiler()
```

Creates an empty profiler.

Example:

```cpp
Profiler profiler;
```

---

#### Methods

##### `void start(const std::string& name)`

Start timing a named section.

Parameters:

- `name` - Name of the section to profile

Example:

```cpp
Profiler profiler;
profiler.start("data_loading");
load_data();
profiler.stop("data_loading");
```

---

##### `void stop(const std::string& name)`

Stop timing a named section.

Parameters:

- `name` - Name of the section (must match `start()` call)

Example:

```cpp
profiler.start("inference");
model.forward(input);
profiler.stop("inference");
```

---

##### `ProfileStats get_stats(const std::string& name)`

Get statistics for a profiled section.

Parameters:

- `name` - Name of the section

**Returns:** `ProfileStats` with timing statistics

Example:

```cpp
// Run multiple times
for (int i = 0; i < 100; ++i) {
    profiler.start("forward");
    model.forward(input);
    profiler.stop("forward");
}

// Get statistics
ProfileStats stats = profiler.get_stats("forward");
std::cout << "Average: " << stats.mean_time << " ms" << std::endl;
```

---

##### `void print_all()`

Print statistics for all profiled sections.

Output Example:

```text
=== Profiling Results ===

Profile: tokenization
  Calls: 100
  Mean:  2.34 ms
  ...

Profile: inference
  Calls: 100
  Mean:  45.23 ms
  ...

Profile: decoding
  Calls: 100
  Mean:  1.12 ms
  ...

========================
```

Example:

```cpp
profiler.print_all();
```

---

##### `void reset()`

Clear all profiling data.

Example:

```cpp
profiler.reset();  // Start fresh for next experiment
```

---

##### `static void compare(const ProfileStats& baseline, const ProfileStats& optimized)`

Compare two profiling runs (e.g., before/after optimization).

Parameters:

- `baseline` - Statistics from baseline implementation
- `optimized` - Statistics from optimized implementation

Output Example:

```text
=== Performance Comparison ===
Profile: inference

Baseline:
  Mean: 45.23 ms
  Median: 44.89 ms

Optimized:
  Mean: 18.42 ms
  Median: 18.12 ms

Improvement:
  Speedup: 2.46x
  Improvement: 59.3%
  Time saved: 26.81 ms
==============================
```

Example:

```cpp
// Baseline
Profiler baseline_profiler;
for (int i = 0; i < 100; ++i) {
    baseline_profiler.start("inference");
    model.forward_without_cache(input);
    baseline_profiler.stop("inference");
}

// Optimized
Profiler optimized_profiler;
for (int i = 0; i < 100; ++i) {
    optimized_profiler.start("inference");
    model.forward_with_cache(input, cache);
    optimized_profiler.stop("inference");
}

// Compare
ProfileStats baseline = baseline_profiler.get_stats("inference");
ProfileStats optimized = optimized_profiler.get_stats("inference");
Profiler::compare(baseline, optimized);
```

---

### Benchmark

Automated benchmarking utility.

#### Class Definition

```cpp
class Benchmark {
public:
    template <typename Func>
    static ProfileStats run(const std::string& name, Func func,
                           int iterations = 100, int warmup_iterations = 10);

    template <typename FuncA, typename FuncB>
    static void compare(const std::string& name_a, FuncA func_a,
                       const std::string& name_b, FuncB func_b,
                       int iterations = 100, int warmup = 10);
};
```

---

#### Methods

##### `static ProfileStats run(name, func, iterations, warmup_iterations)`

Run a benchmark function multiple times and return statistics.

Template Parameters:

- `Func` - Callable type (function, lambda, functor)

Parameters:

- `name` - Name for the benchmark
- `func` - Function to benchmark
- `iterations` - Number of timed iterations (default: 100)
- `warmup_iterations` - Number of warmup runs (default: 10)

**Returns:** `ProfileStats` with timing statistics

Example:

```cpp
// Benchmark a simple function
ProfileStats stats = Benchmark::run("my_function", []() {
    expensive_computation();
}, 100, 10);

stats.print();
```

Full Example:

```cpp
// Benchmark different batch sizes
std::vector<int> batch_sizes = {1, 4, 8, 16, 32};

for (int bs : batch_sizes) {
    ProfileStats stats = Benchmark::run(
        "batch_size_" + std::to_string(bs),
        [&]() {
            auto batches = create_dynamic_batches(sequences, bs, 10, 0);
            process_batches(batches);
        },
        50,   // iterations
        5     // warmup
    );

    std::cout << "Batch size " << bs << ": "
              << stats.mean_time << " ms" << std::endl;
}
```

---

##### `static void compare(name_a, func_a, name_b, func_b, iterations, warmup)`

Compare two implementations automatically.

Template Parameters:

- `FuncA`, `FuncB` - Callable types

Parameters:

- `name_a` - Name for first implementation
- `func_a` - First function to benchmark
- `name_b` - Name for second implementation
- `func_b` - Second function to benchmark
- `iterations` - Number of iterations per function (default: 100)
- `warmup` - Number of warmup iterations (default: 10)

Example:

```cpp
Benchmark::compare(
    "without_cache",
    [&]() { decoder.forward(tokens); },

    "with_cache",
    [&]() { decoder.forward_with_cache(tokens, cache); },

    100,  // iterations
    10    // warmup
);
```

Output:

```text
=== Benchmarking ===
Running 100 iterations (after 10 warmup)...

=== Performance Comparison ===
Profile: without_cache

Baseline:
  Mean: 45.23 ms
  Median: 44.89 ms

Optimized:
  Mean: 18.42 ms
  Median: 18.12 ms

Improvement:
  Speedup: 2.46x
  Improvement: 59.3%
  Time saved: 26.81 ms
==============================
```

---

## Usage Patterns {#usage-patterns}

### Pattern 1: Quick Function Timing

Use `ScopedTimer` for simple measurements.

```cpp
#include "PerformanceProfiler.hpp"

void process_request() {
    ScopedTimer timer("request_processing");

    // Your code here
    tokenize();
    inference();
    decode();

}  // Automatically prints: "[request_processing] 47.23 ms"
```

---

### Pattern 2: Multi-Section Profiling

Profile different parts of your pipeline.

```cpp
#include "PerformanceProfiler.hpp"

Profiler profiler;

void process_batch(const std::vector<std::string>& inputs) {
    profiler.start("tokenization");
    auto tokens = tokenize(inputs);
    profiler.stop("tokenization");

    profiler.start("batching");
    auto batch = create_batch(tokens);
    profiler.stop("batching");

    profiler.start("inference");
    auto outputs = model.forward(batch);
    profiler.stop("inference");

    profiler.start("decoding");
    auto results = decode(outputs);
    profiler.stop("decoding");
}

int main() {
    // Process many batches
    for (int i = 0; i < 100; ++i) {
        process_batch(test_data);
    }

    // Print all statistics
    profiler.print_all();

    // Output shows which section is slowest:
    // tokenization:  2.34 ms
    // batching:      1.12 ms
    // inference:    45.23 ms  ← Bottleneck!
    // decoding:      0.89 ms
}
```

---

### Pattern 3: Before/After Optimization Comparison

Validate your optimizations with hard data.

```cpp
#include "PerformanceProfiler.hpp"

void benchmark_optimization() {
    std::vector<int> tokens = {1, 2, 3, 4, 5};

    // Measure baseline
    Profiler baseline_profiler;
    for (int i = 0; i < 100; ++i) {
        baseline_profiler.start("generation");

        // Baseline: no cache
        for (int j = 0; j < 50; ++j) {
            decoder.forward(tokens);
        }

        baseline_profiler.stop("generation");
    }

    // Measure optimized version
    Profiler optimized_profiler;
    DecoderKVCache cache(num_layers);

    for (int i = 0; i < 100; ++i) {
        cache.clear();
        optimized_profiler.start("generation");

        // Optimized: with cache
        decoder.forward_with_cache(tokens, cache);
        for (int j = 0; j < 49; ++j) {
            std::vector<int> new_token = {j};
            decoder.forward_with_cache(new_token, cache);
        }

        optimized_profiler.stop("generation");
    }

    // Compare
    ProfileStats baseline = baseline_profiler.get_stats("generation");
    ProfileStats optimized = optimized_profiler.get_stats("generation");
    Profiler::compare(baseline, optimized);

    // Output:
    // Speedup: 2.8x
    // Improvement: 64.3%
    // Time saved: 1234.56 ms
}
```

---

### Pattern 4: Automated Benchmarking

Use `Benchmark` class for clean comparisons.

```cpp
#include "PerformanceProfiler.hpp"

void benchmark_batch_sizes() {
    std::vector<std::vector<int>> sequences = load_sequences();

    // Compare different batch sizes
    Benchmark::compare(
        "batch_size_8",
        [&]() {
            auto batches = create_dynamic_batches(sequences, 8, 10, 0);
            process_batches(batches);
        },

        "batch_size_32",
        [&]() {
            auto batches = create_dynamic_batches(sequences, 32, 10, 0);
            process_batches(batches);
        },

        50,   // 50 iterations
        5     // 5 warmup iterations
    );
}
```

---

### Pattern 5: Production Monitoring

Track performance in production.

```cpp
#include "PerformanceProfiler.hpp"

class APIServer {
private:
    Profiler profiler;
    int request_count = 0;

public:
    void handle_request(const Request& req) {
        profiler.start("request");

        // Process request
        auto result = process(req);

        profiler.stop("request");
        request_count++;

        // Print stats every 100 requests
        if (request_count % 100 == 0) {
            ProfileStats stats = profiler.get_stats("request");
            std::cout << "Last 100 requests:" << std::endl;
            std::cout << "  Mean: " << stats.mean_time << " ms" << std::endl;
            std::cout << "  P95: " << stats.get_percentile(95.0) << " ms" << std::endl;
            std::cout << "  P99: " << stats.get_percentile(99.0) << " ms" << std::endl;

            profiler.reset();  // Reset for next batch
        }

        return result;
    }
};
```

---

## Statistical Analysis {#statistical-analysis}

### Understanding the Metrics

Mean (Average):

- Sum of all times / number of runs
- Good for general performance
- Can be skewed by outliers

Median (50th percentile):

- Middle value when sorted
- More robust to outliers
- Better represents "typical" performance

Min/Max:

- Best and worst case
- Useful for identifying variability

Percentiles:

- P95: 95% of runs complete within this time
- P99: 99% of runs complete within this time
- Critical for SLA requirements

### Example: Analyzing Results

```cpp
ProfileStats stats = profiler.get_stats("inference");
stats.compute_median();

// Check consistency
double variance = stats.max_time - stats.min_time;
std::cout << "Variance: " << variance << " ms" << std::endl;

if (variance > stats.mean_time * 0.5) {
    std::cout << "WARNING: High variance detected!" << std::endl;
    std::cout << "Performance is inconsistent." << std::endl;
}

// Check for outliers
double p99 = stats.get_percentile(99.0);
if (p99 > stats.median_time * 1.5) {
    std::cout << "WARNING: Outliers detected!" << std::endl;
    std::cout << "1% of runs are significantly slower." << std::endl;
}

// SLA check
double sla_target = 50.0;  // 50ms SLA
double p95 = stats.get_percentile(95.0);

if (p95 <= sla_target) {
    std::cout << "✓ Meeting SLA (95% < " << sla_target << "ms)" << std::endl;
} else {
    std::cout << "✗ Missing SLA (95% = " << p95 << "ms)" << std::endl;
}
```

---

## Benchmarking {#benchmarking}

### Proper Benchmarking Methodology

Key Principles:

1. **Warmup:** Run code multiple times before measuring to warm caches
2. **Multiple Iterations:** Reduce noise with statistical sampling
3. **Consistent Environment:** Same inputs, minimal background processes
4. **Representative Data:** Use realistic inputs

### Example: Comprehensive Benchmark

```cpp
void comprehensive_benchmark() {
    // 1. Prepare consistent test data
    std::vector<std::vector<int>> test_sequences = generate_test_data(100);

    // 2. Warm up the system
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        process_sequences(test_sequences);
    }

    // 3. Run benchmark with adequate iterations
    std::cout << "Benchmarking..." << std::endl;
    ProfileStats stats = Benchmark::run(
        "sequence_processing",
        [&]() {
            process_sequences(test_sequences);
        },
        100,  // 100 iterations for good statistical power
        10    // 10 warmup iterations
    );

    // 4. Analyze results
    stats.print();

    // 5. Check for issues
    if (stats.get_percentile(99.0) > stats.median_time * 2) {
        std::cout << "WARNING: Some runs are much slower (outliers)" << std::endl;
    }

    // 6. Calculate throughput
    double sequences_per_second = (test_sequences.size() * 1000.0) / stats.mean_time;
    std::cout << "Throughput: " << sequences_per_second << " seq/s" << std::endl;
}
```

### Comparing Multiple Implementations

```cpp
void compare_implementations() {
    std::vector<std::vector<int>> data = load_test_data();

    // Test 3 different approaches
    std::vector<std::pair<std::string, std::function<void()>>> implementations = {
        {"naive", [&]() { process_naive(data); }},
        {"optimized_v1", [&]() { process_optimized_v1(data); }},
        {"optimized_v2", [&]() { process_optimized_v2(data); }}
    };

    std::vector<ProfileStats> results;

    for (const auto& impl : implementations) {
        ProfileStats stats = Benchmark::run(impl.first, impl.second, 50, 5);
        results.push_back(stats);

        std::cout << impl.first << ": " << stats.mean_time << " ms" << std::endl;
    }

    // Find best implementation
    auto best = std::min_element(results.begin(), results.end(),
        [](const ProfileStats& a, const ProfileStats& b) {
            return a.mean_time < b.mean_time;
        });

    std::cout << "\nBest: " << best->name << std::endl;

    // Compare each to baseline
    for (size_t i = 1; i < results.size(); ++i) {
        Profiler::compare(results[0], results[i]);
    }
}
```

---

## Best Practices {#best-practices}

### DO ✅

```cpp
// ✅ Use warmup iterations
ProfileStats stats = Benchmark::run("test", func, 100, 10);

// ✅ Run enough iterations for statistical significance
profiler.start("operation");
for (int i = 0; i < 100; ++i) {
    perform_operation();
}
profiler.stop("operation");

// ✅ Profile in release mode for accurate results
// cmake .. -DCMAKE_BUILD_TYPE=Release

// ✅ Use consistent test data
auto test_data = load_fixed_test_set();
Benchmark::run("test", [&]() { process(test_data); });

// ✅ Check percentiles for production SLAs
double p95 = stats.get_percentile(95.0);
assert(p95 < 50.0);  // 95% of requests under 50ms

// ✅ Profile specific sections to find bottlenecks
profiler.start("tokenization");
// ...
profiler.start("inference");
// ...
profiler.print_all();  // See which is slowest
```

---

### DON'T ❌

```cpp
// ❌ Don't profile in debug mode
// Debug builds are 5-10x slower and not representative

// ❌ Don't use too few iterations
Benchmark::run("test", func, 3);  // Bad! Not statistically significant

// ❌ Don't skip warmup
Benchmark::run("test", func, 100, 0);  // Bad! First runs slower

// ❌ Don't use random data for benchmarks
Benchmark::run("test", [&]() {
    auto data = generate_random_data();  // Bad! Inconsistent
    process(data);
});

// ❌ Don't ignore outliers
// If P99 >> median, investigate why

// ❌ Don't compare mean to median
// Use mean-to-mean or median-to-median
```

---

## Advanced Topics {#advanced-topics}

### Nested Profiling

```cpp
Profiler profiler;

void process_batch() {
    profiler.start("total_batch");

    profiler.start("tokenization");
    tokenize();
    profiler.stop("tokenization");

    profiler.start("inference");
    {
        profiler.start("attention");
        attention();
        profiler.stop("attention");

        profiler.start("feedforward");
        feedforward();
        profiler.stop("feedforward");
    }
    profiler.stop("inference");

    profiler.stop("total_batch");
}

// Results show hierarchy:
// total_batch:   50.00 ms
//   tokenization: 2.00 ms (4% of total)
//   inference:   47.00 ms (94% of total)
//     attention:   38.00 ms (81% of inference)
//     feedforward:  9.00 ms (19% of inference)
```

### Custom Statistics

```cpp
ProfileStats stats = profiler.get_stats("inference");

// Calculate standard deviation
double variance = 0.0;
for (double time : stats.timings) {
    double diff = time - stats.mean_time;
    variance += diff * diff;
}
double std_dev = std::sqrt(variance / stats.call_count);

std::cout << "Standard deviation: " << std_dev << " ms" << std::endl;

// Coefficient of variation (relative variability)
double cv = (std_dev / stats.mean_time) * 100.0;
std::cout << "Coefficient of variation: " << cv << "%" << std::endl;

if (cv > 20.0) {
    std::cout << "High variability detected!" << std::endl;
}
```

### Memory-Efficient Profiling

For long-running systems, limit stored timings:

```cpp
struct LimitedProfileStats : public ProfileStats {
    static const int MAX_TIMINGS = 1000;

    void add_timing(double time_ms) override {
        if (timings.size() >= MAX_TIMINGS) {
            timings.erase(timings.begin());  // Remove oldest
        }
        ProfileStats::add_timing(time_ms);
    }
};
```

### Integration with Logging

```cpp
class LoggingProfiler : public Profiler {
public:
    void log_stats(const std::string& name, std::ostream& log) {
        ProfileStats stats = get_stats(name);

        log << std::time(nullptr) << ","
            << name << ","
            << stats.mean_time << ","
            << stats.median_time << ","
            << stats.get_percentile(95.0) << ","
            << stats.get_percentile(99.0) << std::endl;
    }
};

// Usage
LoggingProfiler profiler;
std::ofstream log("performance.csv");

// ... run profiling ...

profiler.log_stats("inference", log);
// Output: timestamp,inference,45.23,44.89,48.21,52.67
```

---

## Troubleshooting

### Problem: Inconsistent timing results

Possible causes:

1. Background processes
2. Dynamic CPU frequency scaling
3. Insufficient iterations
4. No warmup

Solutions:

```cpp
// Solution 1: More iterations
ProfileStats stats = Benchmark::run("test", func, 500, 50);  // Not 10

// Solution 2: Check variance
stats.print();
if ((stats.max_time - stats.min_time) > stats.mean_time) {
    std::cout << "High variance - system may be busy" << std::endl;
}

// Solution 3: Use median instead of mean
std::cout << "Median: " << stats.median_time << " ms" << std::endl;
```

---

### Problem: Profiling overhead affecting results

**Cause:** Timer calls add small overhead

Solutions:

```cpp
// Solution 1: Profile larger chunks
profiler.start("batch_of_100");
for (int i = 0; i < 100; ++i) {
    quick_operation();  // Don't profile each iteration
}
profiler.stop("batch_of_100");

// Solution 2: Measure overhead
Timer timer;
timer.start();
for (int i = 0; i < 1000; ++i) {
    // Empty loop
}
double overhead = timer.stop() / 1000.0;
std::cout << "Timer overhead: " << overhead << " ms per call" << std::endl;
```

---

### Problem: Debug vs Release performance differences

**Cause:** Debug builds include extra checks and disable optimizations

Solution:

```bash
# Always benchmark in Release mode
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# Run benchmarks
./benchmark
```

---

## Performance Expectations

### Timer Precision

- **Resolution:** Microsecond precision (0.001 ms)
- **Overhead:** ~0.001-0.01 ms per start/stop
- **Accuracy:** ±1-2 microseconds

### Recommended Iteration Counts

|Operation Duration|Iterations|Warmup|
|-------------------|-----------|---------|
|< 1 ms|1000+|100|
|1-10 ms|100-500|10-50|
|10-100 ms|50-100|5-10|
|> 100 ms|10-50|2-5|

---

## See Also

- **[Inference Optimization Guide](../guides/inference-optimization.md)** - Complete optimization guide
- **[KVCache API](kvcache.md)** - KV cache performance measurement examples
- **[BatchProcessor API](batchprocessor.md)** - Batch processing benchmarks
- **[Quick Start](../guides/inference-optimization-quickstart.md)** - 5-minute tutorial

---

**Last Updated:** January 25, 2026
**Version:** 1.0
**Status:** Production-ready
