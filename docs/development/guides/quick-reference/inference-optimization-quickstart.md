# Inference Optimization Quick Start

Get started with KV cache and batch processing in 5 minutes.

## Installation

No additional dependencies required. The optimization features are built into the core library.

```bash
cd build
cmake .. -DBUILD_EXAMPLES=ON
make
```

## Quick Example: KV Cache

```cpp
#include "Decoder.hpp"
#include "KVCache.hpp"

int main() {
    // Create decoder
    int vocab_size = 1000;
    LLMDecoder decoder(vocab_size, 512, 6, 8, 2048, 1024);

    // Create cache (IMPORTANT: one per sequence)
    DecoderKVCache cache(6);  // 6 = num_layers

    // Initial prompt
    std::vector<int> prompt = {1, 2, 3};
    Matrix output = decoder.forward_with_cache(prompt, cache, nullptr, true);

    // Generate tokens (this is where cache speeds things up!)
    for (int i = 0; i < 50; ++i) {
        // Get next token (from language model head + sampling)
        int next_token = 100 + i;  // Simplified

        // Process only the new token (not the entire sequence!)
        std::vector<int> new_token = {next_token};
        output = decoder.forward_with_cache(new_token, cache, nullptr, true);
    }

    // When done, clear cache for next sequence
    cache.clear();

    return 0;
}
```

**Result**: 2-3x speedup compared to processing entire sequence each time.

## Quick Example: Batch Processing

```cpp
#include "BatchProcessor.hpp"
#include "Decoder.hpp"

int main() {
    // Multiple sequences of varying length
    std::vector<std::vector<int>> sequences = {
        {1, 2, 3, 4, 5},
        {10, 11, 12},
        {20, 21, 22, 23, 24, 25, 26}
    };

    // Create efficient batches
    auto batches = create_dynamic_batches(
        sequences,
        /*max_batch_size=*/4,
        /*length_tolerance=*/5,
        /*pad_token_id=*/0
    );

    // Show batch efficiency
    BatchStats stats = compute_batch_stats(batches);
    stats.print();

    // Process batches
    LLMDecoder decoder(1000, 512, 6, 8, 2048, 1024);
    for (const auto& batch : batches) {
        for (const auto& seq : batch.batch_token_ids) {
            Matrix output = decoder.forward(seq);
            // Process output...
        }
    }

    return 0;
}
```

## Quick Example: Performance Profiling

```cpp
#include "PerformanceProfiler.hpp"
#include "Decoder.hpp"

int main() {
    LLMDecoder decoder(1000, 512, 6, 8, 2048, 1024);
    std::vector<int> tokens = {1, 2, 3, 4, 5};

    Profiler profiler;

    // Measure baseline
    profiler.start("baseline");
    for (int i = 0; i < 100; ++i) {
        decoder.forward(tokens);
    }
    profiler.stop("baseline");

    // Measure with cache
    DecoderKVCache cache(6);
    profiler.start("with_cache");
    for (int i = 0; i < 100; ++i) {
        std::vector<int> new_token = {i};
        decoder.forward_with_cache(new_token, cache, nullptr, true);
    }
    profiler.stop("with_cache");

    // Compare
    auto baseline = profiler.get_stats("baseline");
    auto optimized = profiler.get_stats("with_cache");
    Profiler::compare(baseline, optimized);

    return 0;
}
```

**Output**:

```text
=== Performance Comparison ===
Speedup: 2.8x
Improvement: 64.3%
Time saved: 125.3 ms
```

## Run the Benchmark Suite

```bash
# Build the benchmark
cd build
make inference_optimization_benchmark

# Run it
./inference_optimization_benchmark
```

This will run comprehensive benchmarks and show:

- KV cache speedup
- Batch processing efficiency
- Combined optimization impact
- Latency analysis

## Common Patterns

### Pattern 1: Chatbot Generation

```cpp
// Create cache once per conversation
DecoderKVCache conversation_cache(num_layers);

while (true) {
    std::string user_input = get_user_input();
    auto tokens = tokenizer.encode(user_input);

    // Generate response using cache
    Matrix hidden = decoder.forward_with_cache(
        tokens, conversation_cache, nullptr, true
    );

    // Continue generation...
}

// Clear cache when conversation ends
conversation_cache.clear();
```

### Pattern 2: API Server with Sessions

```cpp
std::map<std::string, DecoderKVCache> session_caches;

void handle_request(std::string session_id, std::string prompt) {
    // Get or create cache for this session
    auto& cache = session_caches[session_id];

    auto tokens = tokenizer.encode(prompt);
    Matrix output = decoder.forward_with_cache(
        tokens, cache, nullptr, true
    );

    // Generate response...
}

void end_session(std::string session_id) {
    session_caches.erase(session_id);
}
```

### Pattern 3: Batch API Requests

```cpp
// Collect multiple requests
std::vector<std::vector<int>> user_prompts;
for (auto& request : pending_requests) {
    user_prompts.push_back(tokenizer.encode(request.text));
}

// Process efficiently in batches
auto batches = create_dynamic_batches(user_prompts, 8, 10, pad_id);

for (auto& batch : batches) {
    // Process batch (in production, use true batched matrix ops)
    for (auto& seq : batch.batch_token_ids) {
        auto output = decoder.forward(seq);
        // ...
    }
}
```

## Key Takeaways

1. **KV Cache**: Create once per sequence, use `forward_with_cache()`
2. **Batch**: Group similar-length sequences with `create_dynamic_batches()`
3. **Profile**: Use `Profiler` to measure improvements
4. **Backward Compatible**: Old code still works, optimizations are opt-in

## Next Steps

- **[KVCache API Reference](../reference/kvcache.md)** - Complete API documentation with all methods and usage patterns
- **[BatchProcessor API Reference](../reference/batchprocessor.md)** - Complete API documentation for batch processing
- **[PerformanceProfiler API Reference](../reference/performanceprofiler.md)** - Complete API documentation for profiling tools
- **[Full Optimization Guide](inference-optimization.md)** - Complete guide with advanced topics
- Run tests: `make inference_optimization_test && ./inference_optimization_test`
- Check examples: `src/InferenceOptimizationBenchmark.cpp`

## Performance Expectations

|Optimization|Expected Speedup|Use Case|
|-------------|------------------|----------|
|KV Cache|2-3x|Autoregressive generation|
|Batching|2-4x throughput|Multiple simultaneous requests|
|Combined|4-12x|Production deployment|

Happy optimizing! 🚀
