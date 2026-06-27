# Proposal: Performance Profiler Upgrades for Transformer Model

**Status:** Proposed
**Date:** March 11, 2026
**Author:** GitHub Copilot
**Related Issues:** TD-005

## 1. Summary

This document proposes a set of targeted upgrades to the `PerformanceProfiler` (`src/PerformanceProfiler.hpp`) to better serve the needs of the ADAI transformer model. The current profiler provides solid baseline timing and statistical analysis, but lacks several capabilities that are critical for diagnosing bottlenecks in transformer inference and training workflows. This proposal outlines a plan to:

1. **Add Memory Profiling:** Extend `ProfileStats` to capture peak and current memory usage alongside timing data.
2. **Add Transformer-Specific Metrics:** Introduce throughput measurements (tokens/sec, sequences/sec) and per-layer profiling helpers that align with the transformer's component hierarchy.
3. **Implement Hierarchical Profiling with Visualization:** Enforce and display a parent–child relationship between named profiling sections so that nested component timings are automatically rendered as percentages of their parent.
4. **Add JSON/CSV Export:** Allow profiling results to be serialized to machine-readable formats for offline analysis and regression tracking in CI.
5. **Integrate with KV Cache and Batch Processor:** Add first-class helpers that wrap `DecoderKVCache` and `BatchProcessor` operations and emit profiling events automatically.
6. **Add a Real-Time Profiling Dashboard:** Provide an optional in-place CLI dashboard that updates after every profiled section, giving developers live visibility into hotspots.

These upgrades will make it significantly easier to measure, understand, and improve the performance of the transformer model at every stage of the development lifecycle.

## 2. Motivation

The existing `PerformanceProfiler` (introduced in Phase 3, Part 2) gives developers a foundation for timing named sections of code and computing basic statistics. However, as the ADAI transformer has grown to include multi-layer decoders, KV caches, and batch processing, several gaps have become apparent:

* **No memory visibility.** The current profiler records time only. In transformer workloads, memory pressure is often the binding constraint, not compute time. Without memory measurements, developers cannot determine whether a slowdown is caused by cache eviction, allocation overhead, or actual computation.
* **No throughput metrics.** Transformer performance is conventionally expressed as tokens per second or sequences per second. Converting raw millisecond timings to these units is a manual step that every user must repeat.
* **No structural awareness of transformer components.** The profiler treats all sections as a flat list. The transformer model has a well-defined hierarchy (decoder → decoder block → multi-head attention / feedforward / layer norm). Flat profiles make it hard to understand what fraction of total time is spent in, for example, attention across all layers.
* **No machine-readable output.** Profiling results are printed to stdout and cannot easily be consumed by scripts, CI pipelines, or visualization tools without ad hoc parsing.
* **Manual integration with KV cache and batch processing.** Developers who want to profile `forward_with_cache()` or batched inference must write their own wrapping code. This is error-prone and leads to inconsistent profiling across the codebase.
* **No live feedback during long runs.** Long benchmarks or multi-epoch training runs produce no output until they complete. A live dashboard would allow developers to catch unexpectedly slow runs early.

## 3. Proposed Changes

### 3.1. Memory Profiling

`ProfileStats` will be extended with memory fields populated via platform-portable calls to `sysinfo` (Linux) and `GetProcessMemoryInfo` (Windows), abstracted behind a new `MemorySnapshot` helper struct.

```cpp
// In src/PerformanceProfiler.hpp

/**
 * Snapshot of process memory at a point in time.
 */
struct MemorySnapshot {
    size_t rss_bytes;       // Resident set size (physical RAM)
    size_t virtual_bytes;   // Virtual address space
    size_t heap_bytes;      // Heap allocation (via mallinfo on Linux)
};

/**
 * Capture the current memory footprint of the process.
 */
MemorySnapshot capture_memory_snapshot();

struct ProfileStats {
    // ... existing fields (unchanged) ...

    // New memory fields
    size_t peak_rss_bytes;          // Maximum RSS observed during this section
    int64_t memory_delta_signed;    // RSS at stop minus RSS at start; negative if memory was freed
};
```

The `Profiler::start()` and `Profiler::stop()` methods will each call `capture_memory_snapshot()` and update the memory fields in the corresponding `ProfileStats`.

### 3.2. Transformer-Specific Throughput Metrics

A new `ThroughputProfiler` class will wrap `Profiler` and add token/sequence counting:

```cpp
// In src/PerformanceProfiler.hpp

/**
 * Throughput-aware profiler for transformer workloads.
 * Tracks tokens processed per second in addition to raw timing.
 */
class ThroughputProfiler {
public:
    /**
     * Start timing a section. token_count is the number of tokens that
     * will be processed in this section (used for tokens/sec calculation).
     */
    void start(const std::string& name, int token_count = 0);

    /**
     * Stop timing and record throughput.
     */
    void stop(const std::string& name);

    /**
     * Get combined timing + throughput stats for a section.
     */
    ThroughputStats get_stats(const std::string& name);

    void print_all();
    void reset();

private:
    Profiler timing_profiler;
    std::map<std::string, int> token_counts;
    std::map<std::string, int> sequence_counts;
};

struct ThroughputStats {
    ProfileStats timing;
    double tokens_per_second;    // Average tokens/sec across all calls
    double sequences_per_second; // Average sequences/sec across all calls
    int total_tokens_processed;
    int total_sequences_processed;

    void print() const;
};
```

Example usage:

```cpp
ThroughputProfiler profiler;

// Profile a generation step with token count
profiler.start("decode_step", /* token_count= */ new_tokens.size());
auto output = decoder.forward_with_cache(new_tokens, kv_cache, nullptr, true);
profiler.stop("decode_step");

auto stats = profiler.get_stats("decode_step");
// Output:
//   Tokens/sec: 1250.3
//   Sequences/sec: 62.5
//   Mean latency: 0.80 ms
```

### 3.3. Hierarchical Profiling with Visualization

A new `HierarchicalProfiler` class will maintain a tree of profiling sections with explicit parent–child relationships. This directly models the transformer component hierarchy.

```cpp
// In src/PerformanceProfiler.hpp

/**
 * Node in the profiling tree.
 */
struct ProfileNode {
    std::string name;
    ProfileStats stats;
    std::vector<ProfileNode> children;
    ProfileNode* parent = nullptr;

    /**
     * Return this node's mean time as a percentage of parent's mean time.
     */
    double percent_of_parent() const;

    /**
     * Print this node and all children with indentation.
     */
    void print(int depth = 0) const;
};

/**
 * Tree-structured profiler for hierarchical workloads.
 *
 * Sections are pushed onto a stack so that any section started while
 * another is active automatically becomes a child of the enclosing section.
 */
class HierarchicalProfiler {
public:
    /**
     * Start a new section. If another section is currently active, this
     * section becomes its child.
     */
    void start(const std::string& name);

    /**
     * Stop the most recently started section.
     */
    void stop(const std::string& name);

    /**
     * Print the full tree with indentation and percentage columns.
     */
    void print_tree() const;

    void reset();

private:
    ProfileNode root;
    ProfileNode* current_node = &root;
    std::map<std::string, Timer> active_timers;
};
```

Example output:

```text
=== Hierarchical Profiling Results ===
decoder_forward            45.80 ms  100.0%
  embedding                 1.20 ms    2.6%
  decoder_block[0]         22.10 ms   48.3%
    self_attention          17.50 ms   79.2% of parent
    feedforward              4.40 ms   19.9% of parent
    layer_norm               0.20 ms    0.9% of parent
  decoder_block[1]         21.90 ms   47.8%
    self_attention          17.30 ms   79.0% of parent
    feedforward              4.40 ms   20.1% of parent
    layer_norm               0.20 ms    0.9% of parent
  lm_head                   0.60 ms    1.3%
======================================
```

A convenience macro will be provided for RAII-style hierarchical profiling. It uses `__COUNTER__` (or `__LINE__` as a fallback) to generate a unique guard variable name, ensuring the macro works correctly even when `name` is a string literal containing spaces or special characters:

```cpp
#define PROFILE_SCOPE_HIERARCHICAL(profiler, name) \
    profiler.start(name); \
    auto __hp_guard_##__COUNTER__ = std::shared_ptr<void>(nullptr, [&](void*) { \
        profiler.stop(name); \
    });
```

### 3.4. JSON and CSV Export

`ProfileStats`, `ThroughputStats`, and `ProfileNode` will each gain a `to_json()` method and a static `write_csv()` helper. A top-level `ProfilerExporter` utility class will collect results from any profiler type and serialize them.

```cpp
// In src/PerformanceProfiler.hpp

class ProfilerExporter {
public:
    /**
     * Export all stats from a Profiler to a JSON file.
     *
     * Output format:
     * {
     *   "timestamp": "2026-03-11T01:36:00Z",
     *   "profiles": [
     *     { "name": "attention", "calls": 100, "mean_ms": 38.12, ... },
     *     ...
     *   ]
     * }
     */
    static void to_json(const Profiler& profiler, const std::string& filepath);

    /**
     * Export all stats to a CSV file for spreadsheet analysis.
     *
     * Columns: name,calls,mean_ms,median_ms,min_ms,max_ms,p95_ms,p99_ms,
     *          peak_rss_bytes,memory_delta_bytes
     */
    static void to_csv(const Profiler& profiler, const std::string& filepath);

    /**
     * Append a single ProfileStats row to an existing CSV (creates if absent).
     * Suitable for incremental logging in long training runs.
     */
    static void append_csv_row(const ProfileStats& stats, const std::string& filepath);
};
```

### 3.5. KV Cache and Batch Processor Integration

Two lightweight wrapper helpers will automatically emit profiling events when KV cache or batch operations occur, removing the need for developers to manually instrument these hot paths.

```cpp
// In src/PerformanceProfiler.hpp

/**
 * Wrapper around DecoderKVCache that profiles cache operations via composition.
 * Usage: construct with a DecoderKVCache instance and a profiler, then call
 * the wrapper methods instead of the cache methods directly.
 */
class ProfiledDecoderKVCache {
public:
    ProfiledDecoderKVCache(DecoderKVCache& cache, ThroughputProfiler& profiler);

    /**
     * Delegates to the wrapped cache's append() and records timing + memory delta.
     */
    void append(int layer_idx, const Matrix& new_keys, const Matrix& new_values);

    /**
     * Delegates to the wrapped cache's clear() and logs a cache-clear event.
     */
    void clear();

    /**
     * Access the underlying cache (e.g., to pass to forward_with_cache()).
     */
    DecoderKVCache& cache();

private:
    DecoderKVCache& cache_;
    ThroughputProfiler& profiler_;
};

/**
 * Wrapper around BatchProcessor that profiles batch creation and inference
 * via composition rather than inheritance to avoid object slicing.
 */
class ProfiledBatchProcessor {
public:
    ProfiledBatchProcessor(BatchProcessor& processor, ThroughputProfiler& profiler);

    Batch create_batch(const std::vector<std::vector<int>>& sequences);

    /**
     * Access the underlying processor for any methods not proxied here.
     */
    BatchProcessor& processor();

private:
    BatchProcessor& processor_;
    ThroughputProfiler& profiler_;
};
```

### 3.6. Real-Time Profiling Dashboard

An optional `ProfilerDashboard` class will render a live CLI view of the top hotspots, updating in-place after every `stop()` call via ANSI escape codes. It wraps any existing `Profiler` or `ThroughputProfiler` instance.

```cpp
// In src/PerformanceProfiler.hpp

/**
 * Real-time CLI dashboard for performance profiling.
 *
 * Wraps any Profiler or ThroughputProfiler and re-renders the top N
 * hotspot sections after every stop() call.
 */
class ProfilerDashboard {
public:
    explicit ProfilerDashboard(Profiler& profiler, int top_n = 10);

    /**
     * Enable or disable live rendering.
     * Disable when stdout is not a TTY (e.g., CI environments).
     */
    void set_enabled(bool enabled);

    /**
     * Re-render the dashboard. Called automatically after stop().
     */
    void render();

private:
    Profiler& profiler_;
    int top_n_;
    bool enabled_;
    int last_render_lines_ = 0;  // Used to erase previous output
};
```

Example dashboard output:

```text
╭──────────────────────────────────────────────────────────────────────╮
│ ADAI Performance Profiler - Live View       Sections: 8  Calls: 452  │
├────────────────────────────┬──────────┬──────────┬──────────┬────────┤
│ Section                    │ Mean(ms) │ P95 (ms) │ Calls    │ % Time │
├────────────────────────────┼──────────┼──────────┼──────────┼────────┤
│ self_attention             │   17.50  │   19.20  │     452  │  79.2% │
│ feedforward                │    4.40  │    4.80  │     452  │  19.9% │
│ layer_norm                 │    0.20  │    0.25  │     904  │   0.9% │
│ embedding                  │    1.20  │    1.35  │      76  │   2.6% │
│ lm_head                    │    0.60  │    0.72  │      76  │   1.3% │
╰────────────────────────────┴──────────┴──────────┴──────────┴────────╯
  Peak RSS: 1.24 GB    Tokens/sec: 1,250   Updated: 2026-03-11 01:36:00
```

## 4. Implementation Plan

The work is broken into six incremental tasks, each building on the previous.

### Task 1 — Memory Profiling (TD-005a)

1. Implement `MemorySnapshot capture_memory_snapshot()` with `#ifdef` guards for Linux (`/proc/self/status`) and Windows (`PROCESS_MEMORY_COUNTERS`).
2. Extend `ProfileStats` with `peak_rss_bytes` and `memory_delta_signed`.
3. Update `Profiler::start()` and `Profiler::stop()` to capture snapshots and update stats.
4. Update `ProfileStats::print()` to display memory fields.
5. Add unit tests in `tests/inference_optimization_test.cpp` under a new `MemoryProfilingTest` fixture.

### Task 2 — Throughput Metrics (TD-005b)

1. Add `ThroughputStats` struct.
2. Implement `ThroughputProfiler` class.
3. Add unit tests verifying tokens/sec calculations under known timings (using `std::this_thread::sleep_for` to create deterministic measurements).
4. Update the Inference Optimization Quick Start guide (`docs/development/guides/inference-optimization-quickstart.md`) with a throughput profiling example.

### Task 3 — Hierarchical Profiling (TD-005c)

1. Implement `ProfileNode` and `HierarchicalProfiler`.
2. Implement `print_tree()` with indentation and percentage formatting.
3. Add `PROFILE_SCOPE_HIERARCHICAL` macro.
4. Add unit tests covering: basic parent–child relationships, three-level nesting, percentage accuracy, and reset behavior.

### Task 4 — JSON/CSV Export (TD-005d)

1. Implement `ProfilerExporter::to_json()` using hand-written JSON serialization (no external dependency — the output schema is simple enough).
2. Implement `ProfilerExporter::to_csv()` and `append_csv_row()`.
3. Add unit tests that write to a temp file and verify schema correctness by string matching.
4. Update `ProfileStats` and `ThroughputStats` with `to_json_object()` helpers.

### Task 5 — KV Cache and Batch Processor Integration (TD-005e)

1. Implement `ProfiledDecoderKVCache` using composition, wrapping a `DecoderKVCache` reference.
2. Implement `ProfiledBatchProcessor` using composition, wrapping a `BatchProcessor` reference.
3. Update the decoder integration test in `tests/decoder_integration_test.cpp` to use the profiled wrappers and assert that profiling events are emitted.
4. Update the Inference Optimization Guide (`docs/development/guides/inference-optimization.md`) with a new "Profiled Wrappers" section.

### Task 6 — Real-Time Dashboard (TD-005f)

1. Implement `ProfilerDashboard::render()` using ANSI escape codes (`\033[A` to move up, `\033[2K` to clear lines).
2. Auto-detect TTY with `isatty(STDOUT_FILENO)` and disable rendering when not a TTY so that CI log files are not corrupted with escape codes.
3. Add a `--profile-dashboard` flag to the existing CLI tooling to enable the dashboard at runtime.
4. Add a smoke test that verifies `render()` produces the expected column headers.

## 5. API Backward Compatibility

All existing `Timer`, `ScopedTimer`, `ProfileStats`, `Profiler`, and `Benchmark` APIs are **unchanged**. The new classes and structs are purely additive. No callers need to be updated unless they opt in to the new features.

The `PROFILE_SCOPE` macro is also unchanged.

## 6. Testing Strategy

|Task|Test File|Key Assertions|
|------|-----------|----------------|
|TD-005a|`tests/inference_optimization_test.cpp`|Memory delta is non-negative during allocation; peak RSS is non-decreasing|
|TD-005b|`tests/inference_optimization_test.cpp`|tokens/sec matches expected value within 10% tolerance using sleep-based timing|
|TD-005c|`tests/inference_optimization_test.cpp`|Parent-child relationships correct; percentages sum to ≤ 100% of parent|
|TD-005d|`tests/inference_optimization_test.cpp`|JSON output is valid and contains all required fields; CSV has correct column count|
|TD-005e|`tests/decoder_integration_test.cpp`|Profiled wrappers emit events; stats match expected section names|
|TD-005f|`tests/inference_optimization_test.cpp`|Dashboard render output contains section names and column headers|

## 7. Benefits

* **Faster bottleneck identification.** The hierarchical view and live dashboard make it immediately obvious which transformer component is consuming the most time, without requiring manual calculations.
* **Memory-aware optimization.** Developers can now see whether an optimization reduces time at the cost of increased memory pressure, enabling informed trade-off decisions.
* **Standardized throughput reporting.** All ADAI benchmarks will express performance in tokens/sec, making results directly comparable to published transformer benchmarks.
* **CI regression detection.** JSON/CSV export enables automated comparison of profiling snapshots across commits, catching performance regressions before they ship.
* **Reduced instrumentation boilerplate.** The profiled KV cache and batch processor wrappers eliminate repetitive profiling code, reducing the chance of missing or inconsistent measurements.
* **Zero cost when disabled.** The dashboard and memory profiling paths can be compiled out with `#ifdef ADAI_PROFILING_EXTENDED`, so production builds incur no overhead from the new features.
