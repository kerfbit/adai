# BatchProcessor API Reference

**Module:** `BatchProcessor.hpp`
**Purpose:** Batch processing utilities for efficient multi-sequence inference
**Performance Impact:** 2-4x throughput improvement

---

## Overview

The BatchProcessor provides utilities for processing multiple sequences together in batches, dramatically improving throughput for production deployments. Instead of processing one sequence at a time, batching allows you to process many sequences in a single forward pass.

### Why Use Batching?

**Without Batching:**

```text
Request 1: Process sequence [1, 2, 3, 4, 5] → 10ms
Request 2: Process sequence [6, 7, 8] → 10ms
Request 3: Process sequence [9, 10, 11, 12] → 10ms
Total: 30ms for 3 requests = 100 requests/second
```

**With Batching:**

```text
Batch: Process all 3 sequences together → 15ms
Total: 15ms for 3 requests = 200 requests/second ✅ 2x faster!
```

**Key Benefits:**

- 2-4x higher throughput
- Better hardware utilization (GPU/CPU)
- Lower latency per request in high-traffic scenarios
- Efficient use of matrix operations

---

## Table of Contents

1. [Core Structures](#core-structures)
2. [Batching Functions](#batching-functions)
3. [Utility Functions](#utility-functions)
4. [Usage Patterns](#usage-patterns)
5. [Performance Optimization](#performance-optimization)
6. [Best Practices](#best-practices)
7. [Advanced Topics](#advanced-topics)

---

## Core Structures {#core-structures}

### TokenBatch

Container for a batch of padded sequences.

#### Structure

```cpp
struct TokenBatch {
    std::vector<std::vector<int>> batch_token_ids;  // Padded sequences
    std::vector<int> lengths;                       // Original lengths
    int max_length;                                 // Maximum length in batch
    int pad_token_id;                               // Padding token ID
};
```

#### Fields

##### `batch_token_ids`

Vector of token ID sequences, all padded to `max_length`.

**Type:** `std::vector<std::vector<int>>`
**Example:**

```cpp
batch.batch_token_ids[0] = {1, 2, 3, 0, 0};  // Original: [1, 2, 3], padded with 0s
batch.batch_token_ids[1] = {4, 5, 6, 7, 8};  // Original: [4, 5, 6, 7, 8], no padding
```

##### `lengths`

Original sequence lengths before padding.

**Type:** `std::vector<int>`
**Example:**

```cpp
batch.lengths[0] = 3;  // First sequence has 3 real tokens
batch.lengths[1] = 5;  // Second sequence has 5 real tokens
```

##### `max_length`

Maximum sequence length in the batch (all sequences padded to this length).

**Type:** `int`

##### `pad_token_id`

Token ID used for padding (typically 0).

**Type:** `int`

#### Methods

##### `int batch_size() const`

Get number of sequences in batch.

**Returns:** Number of sequences

**Example:**

```cpp
TokenBatch batch = create_batch(sequences);
std::cout << "Processing " << batch.batch_size() << " sequences" << std::endl;
```

---

##### `bool is_empty() const`

Check if batch is empty.

**Returns:** `true` if batch contains no sequences

**Example:**

```cpp
if (batch.is_empty()) {
    std::cout << "No sequences to process" << std::endl;
}
```

---

### BatchStats

Statistics about batch efficiency.

#### Structure

```cpp
struct BatchStats {
    int total_tokens;       // Total tokens including padding
    int actual_tokens;      // Actual tokens (excluding padding)
    float padding_ratio;    // Ratio of padding to total
    int num_batches;        // Number of batches
    float avg_batch_size;   // Average batch size
};
```

#### Methods

##### `void print() const`

Print batch statistics to stdout.

**Example Output:**

```text
Batch Statistics:
  Total tokens (with padding): 1000
  Actual tokens: 850
  Padding ratio: 15%
  Number of batches: 5
  Average batch size: 8.5
  Efficiency: 85%
```

**Example:**

```cpp
BatchStats stats = compute_batch_stats(batches);
stats.print();
```

---

## Batching Functions {#batching-functions}

### create_batch

Create a padded batch from multiple sequences.

```cpp
TokenBatch create_batch(
    const std::vector<std::vector<int>>& sequences,
    int pad_token_id = 0
)
```

**Parameters:**

- `sequences` - Vector of token ID sequences (variable length)
- `pad_token_id` - Token ID to use for padding (default: 0)

**Returns:** `TokenBatch` with all sequences padded to same length

**Behavior:**

1. Find the longest sequence in the input
2. Pad all shorter sequences to match the longest
3. Store original lengths for later unpadding

**Example:**

```cpp
std::vector<std::vector<int>> sequences = {
    {1, 2, 3},           // Length 3
    {4, 5, 6, 7, 8},     // Length 5 (longest)
    {9, 10}              // Length 2
};

TokenBatch batch = create_batch(sequences, 0);

// Result:
// batch.batch_token_ids[0] = {1, 2, 3, 0, 0}
// batch.batch_token_ids[1] = {4, 5, 6, 7, 8}
// batch.batch_token_ids[2] = {9, 10, 0, 0, 0}
// batch.max_length = 5
// batch.lengths = {3, 5, 2}
```

**Use Case:** Simple batching when all sequences are similar length.

---

### create_dynamic_batches

Create multiple batches by grouping sequences of similar length.

```cpp
std::vector<TokenBatch> create_dynamic_batches(
    const std::vector<std::vector<int>>& sequences,
    int max_batch_size = 32,
    int length_tolerance = 10,
    int pad_token_id = 0
)
```

**Parameters:**

- `sequences` - Vector of token ID sequences
- `max_batch_size` - Maximum number of sequences per batch (default: 32)
- `length_tolerance` - Maximum length difference within a batch (default: 10)
- `pad_token_id` - Token ID to use for padding (default: 0)

**Returns:** Vector of `TokenBatch`, each containing sequences of similar length

**Algorithm:**

1. Sort sequences by length
2. Group similar-length sequences together
3. Create batches respecting `max_batch_size` and `length_tolerance`
4. Minimize padding within each batch

**Example:**

```cpp
std::vector<std::vector<int>> sequences = {
    {1, 2, 3},                    // Length 3
    {4, 5},                       // Length 2
    {6, 7, 8, 9},                 // Length 4
    {10, 11, 12, 13, 14, 15},     // Length 6
    {16, 17, 18}                  // Length 3
};

auto batches = create_dynamic_batches(sequences, 3, 2, 0);

// Result (example):
// Batch 0: sequences of length 2-4 (items 1, 0, 4, 2)
// Batch 1: sequence of length 6 (item 3)
```

**Why Use This?**

Reduces wasted computation from padding:

```cpp
// Without dynamic batching:
// Batch all 5 sequences together → max_length = 6
// Padding: (6-3) + (6-2) + (6-4) + (6-6) + (6-3) = 14 wasted tokens

// With dynamic batching (tolerance=2):
// Batch 1: lengths [2, 3, 3, 4] → max=4 → padding = 5 wasted
// Batch 2: length [6] → max=6 → padding = 0 wasted
// Total: 5 wasted tokens ✅ 64% less padding!
```

**Use Case:** Production systems with variable-length inputs (chatbots, APIs).

---

## Utility Functions {#utility-functions}

### create_padding_mask

Create a mask indicating real vs. padding tokens.

```cpp
Matrix create_padding_mask(const TokenBatch& batch)
```

**Parameters:**

- `batch` - TokenBatch with padding information

**Returns:** Matrix `[batch_size, max_length]` with padding mask

- Value 1.0 for real tokens
- Value 0.0 for padding tokens

**Purpose:** Used in attention mechanisms to prevent attending to padding.

**Example:**

```cpp
std::vector<std::vector<int>> sequences = {
    {1, 2, 3},      // Length 3
    {4, 5, 6, 7}    // Length 4
};

TokenBatch batch = create_batch(sequences, 0);
Matrix mask = create_padding_mask(batch);

// Result: mask is [2, 4]
// Row 0: [1.0, 1.0, 1.0, 0.0]  (3 real tokens, 1 padding)
// Row 1: [1.0, 1.0, 1.0, 1.0]  (4 real tokens, 0 padding)
```

**Integration with Attention:**

```cpp
// In attention computation
Matrix attention_scores = compute_scores(Q, K);  // [batch, seq, seq]

// Apply padding mask to prevent attending to padding
Matrix padding_mask = create_padding_mask(batch);
for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < seq_len; ++j) {
        if (padding_mask(i, j) < 0.5) {
            // This is a padding position, mask it out
            for (int k = 0; k < seq_len; ++k) {
                attention_scores(i, k, j) = -1e9;  // Large negative value
            }
        }
    }
}

Matrix attention = softmax(attention_scores);  // Padding positions get ~0 weight
```

---

### unbatch_outputs

Extract individual sequences from batch output, removing padding.

```cpp
std::vector<Matrix> unbatch_outputs(
    const std::vector<Matrix>& batch_outputs,
    const TokenBatch& batch
)
```

**Parameters:**

- `batch_outputs` - Vector of matrices, one per batch item
- `batch` - Original TokenBatch with length information

**Returns:** Vector of matrices without padding

**Example:**

```cpp
// After processing batch through model
std::vector<Matrix> batch_outputs = model.forward(batch);

// Remove padding from outputs
std::vector<Matrix> individual_outputs = unbatch_outputs(batch_outputs, batch);

// individual_outputs[0] has shape [3, d_model] (original length 3)
// individual_outputs[1] has shape [5, d_model] (original length 5)
```

**Use Case:** Post-processing after batch inference.

---

### compute_batch_stats

Compute efficiency statistics for batches.

```cpp
BatchStats compute_batch_stats(const std::vector<TokenBatch>& batches)
```

**Parameters:**

- `batches` - Vector of TokenBatch

**Returns:** `BatchStats` with efficiency metrics

**Example:**

```cpp
auto batches = create_dynamic_batches(sequences, 32, 10, 0);
BatchStats stats = compute_batch_stats(batches);

std::cout << "Padding overhead: " << (stats.padding_ratio * 100) << "%" << std::endl;
std::cout << "Efficiency: " << ((1.0 - stats.padding_ratio) * 100) << "%" << std::endl;

// If efficiency is low (<80%), consider:
// - Increasing length_tolerance
// - Reducing max_batch_size
// - Using dynamic batching
```

**Use Case:** Monitoring and tuning batch configuration.

---

## Usage Patterns {#usage-patterns}

### Pattern 1: Simple API Server

Process incoming requests in batches.

```cpp
#include "BatchProcessor.hpp"
#include "Decoder.hpp"
#include <queue>

class BatchedInferenceServer {
private:
    LLMDecoder decoder;
    std::queue<std::vector<int>> request_queue;
    int batch_size = 8;

public:
    void add_request(const std::vector<int>& tokens) {
        request_queue.push(tokens);
    }

    std::vector<std::vector<int>> process_batch() {
        // Collect up to batch_size requests
        std::vector<std::vector<int>> sequences;
        while (!request_queue.empty() && sequences.size() < batch_size) {
            sequences.push_back(request_queue.front());
            request_queue.pop();
        }

        if (sequences.empty()) {
            return {};
        }

        // Create batch
        TokenBatch batch = create_batch(sequences, 0);
        Matrix padding_mask = create_padding_mask(batch);

        // Process batch through model
        std::vector<Matrix> outputs;
        for (const auto& seq : batch.batch_token_ids) {
            Matrix output = decoder.forward(seq, nullptr, true);
            outputs.push_back(output);
        }

        // Remove padding
        auto individual_outputs = unbatch_outputs(outputs, batch);

        // Convert back to token IDs (simplified)
        std::vector<std::vector<int>> results;
        for (const auto& output : individual_outputs) {
            results.push_back({/* decode output */});
        }

        return results;
    }
};
```

---

### Pattern 2: Dynamic Batching for Variable-Length Inputs

Optimize for inputs with varying lengths.

```cpp
#include "BatchProcessor.hpp"

void process_documents(const std::vector<std::string>& documents) {
    // Tokenize documents (variable length)
    std::vector<std::vector<int>> token_sequences;
    for (const auto& doc : documents) {
        token_sequences.push_back(tokenizer.encode(doc));
    }

    // Create dynamic batches (minimize padding)
    auto batches = create_dynamic_batches(
        token_sequences,
        32,   // max_batch_size
        20,   // length_tolerance (allow up to 20 token difference)
        0     // pad_token_id
    );

    // Check efficiency
    BatchStats stats = compute_batch_stats(batches);
    std::cout << "Created " << batches.size() << " batches" << std::endl;
    std::cout << "Efficiency: " << ((1.0 - stats.padding_ratio) * 100)
              << "%" << std::endl;

    // Process each batch
    std::vector<Matrix> all_outputs;
    for (const auto& batch : batches) {
        Matrix padding_mask = create_padding_mask(batch);

        // Process batch through model
        std::vector<Matrix> batch_outputs;
        for (const auto& seq : batch.batch_token_ids) {
            batch_outputs.push_back(model.forward(seq));
        }

        // Unbatch and collect
        auto individual = unbatch_outputs(batch_outputs, batch);
        all_outputs.insert(all_outputs.end(), individual.begin(), individual.end());
    }
}
```

---

### Pattern 3: Real-Time Batching with Timeout

Collect requests for a short time, then process.

```cpp
#include "BatchProcessor.hpp"
#include <chrono>
#include <thread>

class RealTimeBatchProcessor {
private:
    std::vector<std::vector<int>> pending_sequences;
    std::chrono::milliseconds batch_timeout{50};  // 50ms timeout
    std::chrono::time_point<std::chrono::steady_clock> last_batch_time;
    int max_batch_size = 16;

public:
    RealTimeBatchProcessor() {
        last_batch_time = std::chrono::steady_clock::now();
    }

    void add_sequence(const std::vector<int>& seq) {
        pending_sequences.push_back(seq);

        // Check if we should process batch
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_batch_time
        );

        bool should_process =
            pending_sequences.size() >= max_batch_size |  |
            elapsed >= batch_timeout;

        if (should_process && !pending_sequences.empty()) {
            process_pending_batch();
        }
    }

private:
    void process_pending_batch() {
        // Create batch
        TokenBatch batch = create_batch(pending_sequences, 0);

        // Process...

        // Reset
        pending_sequences.clear();
        last_batch_time = std::chrono::steady_clock::now();
    }
};
```

---

### Pattern 4: Batch Processing with KV Cache

Combine batching with KV cache for maximum performance.

```cpp
#include "BatchProcessor.hpp"
#include "KVCache.hpp"

void batch_with_cache_example() {
    // Multiple conversation sessions
    std::vector<std::vector<int>> prompts = {
        {1, 2, 3},
        {4, 5, 6, 7},
        {8, 9}
    };

    // One cache per sequence
    std::vector<DecoderKVCache> caches;
    for (int i = 0; i < prompts.size(); ++i) {
        caches.emplace_back(num_layers);
    }

    // Create batch
    TokenBatch batch = create_batch(prompts, 0);

    // Process batch with caches
    std::vector<Matrix> outputs;
    for (int i = 0; i < batch.batch_size(); ++i) {
        Matrix output = decoder.forward_with_cache(
            batch.batch_token_ids[i],
            caches[i],
            nullptr,
            true
        );
        outputs.push_back(output);
    }

    // Generate next tokens for batch
    for (int step = 0; step < 10; ++step) {
        std::vector<int> next_tokens;

        // Sample next token for each sequence
        for (int i = 0; i < batch.batch_size(); ++i) {
            int next_token = sample_from_logits(outputs[i]);
            next_tokens.push_back(next_token);
        }

        // Process new tokens with caches
        for (int i = 0; i < batch.batch_size(); ++i) {
            std::vector<int> new_token = {next_tokens[i]};
            outputs[i] = decoder.forward_with_cache(
                new_token,
                caches[i],
                nullptr,
                true
            );
        }
    }
}
```

---

## Performance Optimization {#performance-optimization}

### Choosing Batch Size

**Trade-offs:**

| Batch Size | Throughput | Latency | Memory |
| ------------ | ----------- | --------- | -------- |
| 1 | Low | Best | Low |
| 8-16 | Medium | Good | Medium |
| 32-64 | High | Acceptable | High |
| 128+ | Very High | Poor | Very High |

**Guidelines:**

```cpp
// Low latency required (interactive chatbot)
int batch_size = 4;  // Small batches, quick responses

// High throughput required (batch processing)
int batch_size = 32;  // Larger batches, better hardware utilization

// Memory constrained
int batch_size = 8;  // Moderate size

// GPU available
int batch_size = 64;  // GPUs excel at larger batches
```

---

### Optimizing Length Tolerance

**Effect of `length_tolerance`:**

```cpp
// Small tolerance (5-10 tokens)
auto batches = create_dynamic_batches(sequences, 32, 5, 0);
// + Less padding (more efficient)
// - More batches (overhead)
// Use for: Mixed short/long sequences

// Medium tolerance (10-20 tokens)
auto batches = create_dynamic_batches(sequences, 32, 15, 0);
// Balanced approach
// Use for: General purpose

// Large tolerance (20-50 tokens)
auto batches = create_dynamic_batches(sequences, 32, 40, 0);
// + Fewer batches (less overhead)
// - More padding (less efficient)
// Use for: Similar-length sequences
```

**Tuning Example:**

```cpp
// Experiment to find optimal tolerance
std::vector<int> tolerances = {5, 10, 15, 20, 30, 50};

for (int tol : tolerances) {
    auto batches = create_dynamic_batches(sequences, 32, tol, 0);
    BatchStats stats = compute_batch_stats(batches);

    std::cout << "Tolerance: " << tol << std::endl;
    std::cout << "  Batches: " << stats.num_batches << std::endl;
    std::cout << "  Padding: " << (stats.padding_ratio * 100) << "%" << std::endl;
    std::cout << "  Efficiency: " << ((1.0 - stats.padding_ratio) * 100)
              << "%" << std::endl;
}

// Choose tolerance with best efficiency and acceptable batch count
```

---

### Monitoring Performance

```cpp
#include "PerformanceProfiler.hpp"
#include "BatchProcessor.hpp"

void benchmark_batching() {
    Profiler profiler;

    // Test different batch sizes
    std::vector<int> batch_sizes = {1, 4, 8, 16, 32};

    for (int bs : batch_sizes) {
        profiler.start("batch_size_" + std::to_string(bs));

        // Create batches
        auto batches = create_dynamic_batches(sequences, bs, 10, 0);

        // Process batches
        for (const auto& batch : batches) {
            // Process through model...
        }

        profiler.stop("batch_size_" + std::to_string(bs));
    }

    // Print results
    std::cout << "\nBatch Size Performance:" << std::endl;
    for (int bs : batch_sizes) {
        auto stats = profiler.get_stats("batch_size_" + std::to_string(bs));
        std::cout << "Batch " << bs << ": " << stats.mean_us / 1000.0
                  << " ms" << std::endl;
    }
}
```

---

## Best Practices {#best-practices}

### DO ✅

```cpp
// ✅ Use dynamic batching for variable-length inputs
auto batches = create_dynamic_batches(sequences, 32, 10, 0);

// ✅ Monitor batch efficiency
BatchStats stats = compute_batch_stats(batches);
if (stats.padding_ratio > 0.3) {  // More than 30% padding
    // Reduce batch size or increase length tolerance
}

// ✅ Use padding masks in attention
Matrix mask = create_padding_mask(batch);
// Pass mask to attention layers

// ✅ Unbatch outputs to remove padding
auto outputs = unbatch_outputs(batch_outputs, batch);

// ✅ Combine with KV cache for maximum speedup
// (See Pattern 4 above)

// ✅ Tune batch size for your hardware
// CPU: 8-16
// GPU: 32-64
```

---

### DON'T ❌

```cpp
// ❌ Don't use fixed batching for variable-length inputs
// This wastes computation on padding
TokenBatch batch = create_batch(highly_variable_sequences);  // Bad!
// Use create_dynamic_batches instead

// ❌ Don't ignore padding in attention
// This causes model to attend to padding tokens
Matrix output = attention(Q, K, V);  // Bad! No mask
// Use: attention(Q, K, V, padding_mask)

// ❌ Don't forget to unbatch outputs
std::vector<Matrix> outputs = process_batch(batch);
// Still have padding! Need to unbatch:
outputs = unbatch_outputs(outputs, batch);  // Good!

// ❌ Don't use batch size 1 in production
// Defeats the purpose of batching
auto batches = create_dynamic_batches(sequences, 1, 10, 0);  // Bad!

// ❌ Don't create too many small batches
auto batches = create_dynamic_batches(sequences, 32, 1, 0);  // Bad!
// Too strict tolerance creates many batches with overhead
```

---

## Advanced Topics {#advanced-topics}

### Custom Padding Strategies

```cpp
// Left padding (for certain model architectures)
TokenBatch create_left_padded_batch(
    const std::vector<std::vector<int>>& sequences,
    int pad_token_id = 0
) {
    TokenBatch batch;
    batch.pad_token_id = pad_token_id;

    // Find max length
    batch.max_length = 0;
    for (const auto& seq : sequences) {
        if (seq.size() > batch.max_length) {
            batch.max_length = seq.size();
        }
    }

    // Left-pad sequences
    for (const auto& seq : sequences) {
        std::vector<int> padded_seq;
        int padding_needed = batch.max_length - seq.size();

        // Add padding at the beginning
        for (int i = 0; i < padding_needed; ++i) {
            padded_seq.push_back(pad_token_id);
        }

        // Add actual sequence
        padded_seq.insert(padded_seq.end(), seq.begin(), seq.end());

        batch.batch_token_ids.push_back(padded_seq);
        batch.lengths.push_back(seq.size());
    }

    return batch;
}
```

---

### Priority-Based Batching

```cpp
struct PriorityRequest {
    std::vector<int> tokens;
    int priority;  // Higher = more important

    bool operator<(const PriorityRequest& other) const {
        return priority < other.priority;  // Max heap
    }
};

std::vector<TokenBatch> create_priority_batches(
    std::vector<PriorityRequest>& requests,
    int max_batch_size
) {
    // Sort by priority (descending)
    std::sort(requests.begin(), requests.end(), std::greater<PriorityRequest>());

    // Extract token sequences
    std::vector<std::vector<int>> sequences;
    for (const auto& req : requests) {
        sequences.push_back(req.tokens);
    }

    // Create batches (high priority requests in early batches)
    std::vector<TokenBatch> batches;
    for (size_t i = 0; i < sequences.size(); i += max_batch_size) {
        size_t end = std::min(i + max_batch_size, sequences.size());
        std::vector<std::vector<int>> batch_seqs(
            sequences.begin() + i,
            sequences.begin() + end
        );
        batches.push_back(create_batch(batch_seqs));
    }

    return batches;
}
```

---

### Adaptive Batching

```cpp
class AdaptiveBatcher {
private:
    int min_batch_size = 4;
    int max_batch_size = 32;
    float target_efficiency = 0.85;  // 85% efficiency target

public:
    std::vector<TokenBatch> create_adaptive_batches(
        const std::vector<std::vector<int>>& sequences
    ) {
        int current_batch_size = (min_batch_size + max_batch_size) / 2;
        int best_batch_size = current_batch_size;
        float best_efficiency = 0.0;

        // Try different batch sizes
        for (int bs = min_batch_size; bs <= max_batch_size; bs += 4) {
            auto batches = create_dynamic_batches(sequences, bs, 10, 0);
            BatchStats stats = compute_batch_stats(batches);

            float efficiency = 1.0 - stats.padding_ratio;

            if (efficiency > best_efficiency) {
                best_efficiency = efficiency;
                best_batch_size = bs;
            }

            // Early exit if we hit target
            if (efficiency >= target_efficiency) {
                break;
            }
        }

        return create_dynamic_batches(sequences, best_batch_size, 10, 0);
    }
};
```

---

## Troubleshooting

### Problem: Low throughput improvement from batching

**Possible causes:**

1. Batch size too small
2. Too much padding overhead
3. Sequential processing instead of parallel

**Solutions:**

```cpp
// Solution 1: Increase batch size
auto batches = create_dynamic_batches(sequences, 32, 10, 0);  // Not 8

// Solution 2: Use dynamic batching
auto batches = create_dynamic_batches(sequences, 32, 10, 0);  // Not create_batch

// Solution 3: Check batch stats
BatchStats stats = compute_batch_stats(batches);
if (stats.padding_ratio > 0.3) {
    // Adjust tolerance or batch size
}
```

---

### Problem: Out of memory when batching

**Possible causes:**

1. Batch size too large for available memory
2. Sequences too long

**Solutions:**

```cpp
// Solution 1: Reduce batch size
auto batches = create_dynamic_batches(sequences, 16, 10, 0);  // Not 64

// Solution 2: Truncate long sequences
for (auto& seq : sequences) {
    if (seq.size() > max_length) {
        seq.resize(max_length);
    }
}

// Solution 3: Process in smaller chunks
std::vector<std::vector<int>> chunk;
for (size_t i = 0; i < sequences.size(); i += 100) {
    chunk.assign(sequences.begin() + i,
                 sequences.begin() + std::min(i + 100, sequences.size()));
    auto batches = create_dynamic_batches(chunk, 16, 10, 0);
    // Process batches...
}
```

---

### Problem: High padding ratio

**Possible causes:**

1. Very variable sequence lengths
2. Length tolerance too large

**Solutions:**

```cpp
// Solution 1: Reduce length tolerance
auto batches = create_dynamic_batches(sequences, 32, 5, 0);  // Not 50

// Solution 2: Pre-filter sequences
std::vector<std::vector<int>> short_seqs, long_seqs;
for (const auto& seq : sequences) {
    if (seq.size() < 50) {
        short_seqs.push_back(seq);
    } else {
        long_seqs.push_back(seq);
    }
}
auto short_batches = create_dynamic_batches(short_seqs, 32, 10, 0);
auto long_batches = create_dynamic_batches(long_seqs, 16, 20, 0);

// Solution 3: Check stats and tune
BatchStats stats = compute_batch_stats(batches);
stats.print();  // Identify the issue
```

---

## Performance Expectations

### Throughput Improvement

| Configuration | Throughput Gain | Use Case |
| -------------- | ---------------- | ---------- |
| Batch size 4 | 1.5-2x | Low latency |
| Batch size 16 | 2-3x | Balanced |
| Batch size 32 | 3-4x | High throughput |
| Batch size 64+ | 3-5x | Maximum throughput |

### Combined with KV Cache

| Optimization | Individual | Combined |
| ------------- | ----------- | ---------- |
| KV Cache only | 2-3x | - |
| Batching only | 2-4x | - |
| Both | - | **4-12x** ✅ |

---

## See Also

- **[Inference Optimization Guide](../guides/inference-optimization.md)** - Complete optimization guide
- **[KVCache API Reference](kvcache.md)** - KV cache for autoregressive generation
- **[PerformanceProfiler API Reference](performanceprofiler.md)** - Profiling and benchmarking tools
- **[Quick Start](../guides/inference-optimization-quickstart.md)** - 5-minute tutorial

---

**Last Updated:** January 25, 2026
**Version:** 1.0
**Status:** Production-ready
