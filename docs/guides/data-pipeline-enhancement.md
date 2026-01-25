# Data Pipeline Enhancement Guide

**Version:** 1.0  
**Date:** January 2026  
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Efficient Batching](#efficient-batching)
3. [Parallel Data Loading](#parallel-data-loading)
4. [Data Augmentation](#data-augmentation)
5. [Complete Examples](#complete-examples)
6. [Performance Optimization](#performance-optimization)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The data pipeline enhancement provides two main components for efficient training:

1. **EfficientBatching**: Advanced batching strategies to minimize padding and maximize GPU/CPU utilization
2. **ParallelDataLoader**: Multi-threaded data loading with background prefetching to hide I/O latency

### Key Features

- ✅ **Dynamic batching by sequence length** - Group similar-length sequences to reduce padding
- ✅ **Bucketing strategy** - Assign sequences to buckets for optimal batching
- ✅ **Multiple padding strategies** - Left, right, or center padding
- ✅ **Data augmentation** - Token dropout, masking, and shuffling
- ✅ **Parallel loading** - Multi-threaded batch preparation
- ✅ **Background prefetching** - Maintain buffer of ready batches
- ✅ **Thread-safe operations** - Safe for multi-threaded training

### Benefits

| Feature | Benefit | Typical Improvement |
|---------|---------|---------------------|
| Dynamic batching | Reduced padding | 20-50% less padding |
| Bucketing | Better memory efficiency | 30-60% less padding |
| Parallel loading | Hide I/O latency | 2-5x faster data loading |
| Prefetching | Maximize GPU utilization | 10-30% faster training |
| Data augmentation | Better generalization | Improved model accuracy |

---

## Efficient Batching

### Basic Usage

```cpp
#include "EfficientBatching.hpp"

// Create test sequences
std::vector<std::vector<int>> sequences = {
    {1, 2, 3},           // length 3
    {4, 5, 6, 7},        // length 4
    {8, 9},              // length 2
    {10, 11, 12, 13, 14} // length 5
};

// Create batches with dynamic batching
auto batches = EfficientBatching::create_dynamic_batches(
    sequences,
    2,                              // batch_size
    0,                              // pad_token_id
    PaddingStrategy::RIGHT,         // padding strategy
    true                            // sort_by_length
);

// Each batch contains:
// - sequences: padded sequences (all same length)
// - masks: attention masks (1=valid, 0=padding)
// - lengths: original sequence lengths
// - max_length: maximum length in batch
```

### Padding Strategies

#### Right Padding (Default)
```cpp
// Original: [1, 2, 3]
// Padded:   [1, 2, 3, 0, 0]
// Mask:     [1, 1, 1, 0, 0]

auto batches = EfficientBatching::create_dynamic_batches(
    sequences, batch_size, 0, PaddingStrategy::RIGHT
);
```

**Best for:** Autoregressive models (GPT-style), left-to-right generation

#### Left Padding
```cpp
// Original: [1, 2, 3]
// Padded:   [0, 0, 1, 2, 3]
// Mask:     [0, 0, 1, 1, 1]

auto batches = EfficientBatching::create_dynamic_batches(
    sequences, batch_size, 0, PaddingStrategy::LEFT
);
```

**Best for:** Right-to-left models, certain encoder architectures

#### Center Padding
```cpp
// Original: [1, 2, 3]
// Padded:   [0, 1, 2, 3, 0]
// Mask:     [0, 1, 1, 1, 0]

auto batches = EfficientBatching::create_dynamic_batches(
    sequences, batch_size, 0, PaddingStrategy::CENTER
);
```

**Best for:** Specialized models, symmetric processing

### Dynamic Batching by Length

Sorting sequences by length before batching significantly reduces padding:

```cpp
// WITHOUT sorting
auto batches_unsorted = EfficientBatching::create_dynamic_batches(
    sequences, 3, 0, PaddingStrategy::RIGHT, false  // sort=false
);

// WITH sorting (recommended)
auto batches_sorted = EfficientBatching::create_dynamic_batches(
    sequences, 3, 0, PaddingStrategy::RIGHT, true  // sort=true
);

// Calculate statistics
auto stats_unsorted = EfficientBatching::calculate_statistics(batches_unsorted);
auto stats_sorted = EfficientBatching::calculate_statistics(batches_sorted);

std::cout << "Unsorted padding: " << stats_unsorted.padding_ratio << "\n";
std::cout << "Sorted padding: " << stats_sorted.padding_ratio << "\n";
// Typical improvement: 20-50% reduction
```

### Bucketing Strategy

For datasets with wide length variation, bucketing is more efficient:

```cpp
// Configure buckets
BucketConfig config;
config.bucket_boundaries = {10, 20, 30, 40};  // Creates 5 buckets
config.max_tokens_per_batch = 500;            // Token budget per batch
config.shuffle_buckets = true;                // Shuffle within buckets

// Create bucketed batches
auto batches = EfficientBatching::create_bucketed_batches(
    sequences, config, 0, PaddingStrategy::RIGHT
);
```

**How it works:**
1. Sequences assigned to buckets by length
2. Batch created within each bucket (similar lengths together)
3. Batch size dynamically adjusted to stay within token budget
4. Results in minimal padding waste

**Bucket boundaries example:**
```
Bucket 0: sequences with length <= 10
Bucket 1: sequences with 10 < length <= 20
Bucket 2: sequences with 20 < length <= 30
Bucket 3: sequences with 30 < length <= 40
Bucket 4: sequences with length > 40
```

### Batch Statistics

Monitor batching efficiency:

```cpp
auto batches = EfficientBatching::create_dynamic_batches(sequences, 8, 0);
auto stats = EfficientBatching::calculate_statistics(batches);

std::cout << "Number of batches: " << stats.num_batches << "\n";
std::cout << "Total sequences: " << stats.total_sequences << "\n";
std::cout << "Total tokens: " << stats.total_tokens << "\n";
std::cout << "Padding tokens: " << stats.total_padding_tokens << "\n";
std::cout << "Avg batch size: " << stats.avg_batch_size << "\n";
std::cout << "Avg sequence length: " << stats.avg_sequence_length << "\n";
std::cout << "Padding ratio: " << stats.padding_ratio << "\n";
std::cout << "Efficiency score: " << stats.efficiency_score << "\n";
```

**Interpreting results:**
- **Padding ratio**: 0.0 = no padding (perfect), 1.0 = all padding (worst)
- **Efficiency score**: 1.0 - padding_ratio (higher is better)
- **Target**: < 0.2 padding ratio (> 0.8 efficiency) for good performance

---

## Parallel Data Loading

### Basic Usage

```cpp
#include "ParallelDataLoader.hpp"
#include "Dataset.hpp"

// Create dataset
Dataset dataset(/* config */);
// ... add samples to dataset ...

// Configure data loader
DataLoaderConfig config;
config.batch_size = 32;
config.num_workers = 4;              // Number of worker threads
config.prefetch_factor = 2;          // Batches to prefetch per worker
config.shuffle = true;               // Shuffle data each epoch
config.use_dynamic_batching = true;  // Enable dynamic batching

// Create loader
ParallelDataLoader loader(dataset, config);

// Start loading
loader.start();

// Get batches
while (auto batch = loader.next_batch()) {
    // Process batch (training step)
    // batch->sequences, batch->masks, etc.
}

// Stop when done
loader.stop();
```

### Iterator Interface

Convenient epoch-based iteration:

```cpp
ParallelDataLoader loader(dataset, config);
DataLoaderIterator iter(loader);

// Iterate through one epoch
while (auto batch = iter.next()) {
    // Process batch
}

// Start new epoch
iter.reset();

// Iterate again
while (auto batch = iter.next()) {
    // Process batch
}
```

### Configuration Options

```cpp
DataLoaderConfig config;

// Batch configuration
config.batch_size = 32;              // Sequences per batch
config.drop_last = false;            // Keep incomplete last batch

// Threading configuration
config.num_workers = 4;              // Worker threads (typically 2-8)
config.prefetch_factor = 2;          // Prefetch buffer multiplier

// Data configuration
config.shuffle = true;               // Shuffle each epoch
config.seed = 42;                    // Random seed for reproducibility

// Padding configuration
config.pad_token_id = 0;             // Padding token ID
config.padding_strategy = PaddingStrategy::RIGHT;

// Batching strategy
config.use_dynamic_batching = true;  // Sort by length
config.use_bucketing = false;        // Use bucketing (alternative)
config.bucket_config = /* ... */;    // Bucket configuration

// Augmentation
config.augmentation_config = /* ... */;  // Data augmentation config
```

### Worker Thread Configuration

**Choosing number of workers:**
```cpp
// CPU-bound (model training on CPU)
config.num_workers = 2;  // Low overhead, avoid CPU contention

// GPU-bound (model training on GPU)
config.num_workers = 4-8;  // Hide I/O latency while GPU computes

// I/O-bound (slow disk, network storage)
config.num_workers = 8-16;  // Compensate for slow I/O
```

**Prefetch factor:**
```cpp
// Low memory
config.prefetch_factor = 1;  // Minimal buffer (1 batch per worker)

// Balanced (recommended)
config.prefetch_factor = 2;  // 2 batches per worker

// High throughput
config.prefetch_factor = 4;  // Large buffer (more memory usage)
```

**Total prefetch buffer size** = `num_workers × prefetch_factor`

### Epoch Management

```cpp
ParallelDataLoader loader(dataset, config);

// Check epoch info
std::cout << "Current epoch: " << loader.current_epoch() << "\n";
std::cout << "Batches per epoch: " << loader.num_batches() << "\n";

// Manually start new epoch (reshuffles if enabled)
loader.new_epoch();

// Monitor progress
std::cout << "Batches loaded: " << loader.batches_loaded() << "\n";
std::cout << "Queue size: " << loader.queue_size() << "\n";
```

---

## Data Augmentation

### Token Dropout

Randomly drop tokens during training to improve robustness:

```cpp
AugmentationConfig config;
config.enable_token_dropout = true;
config.token_dropout_prob = 0.1f;  // Drop 10% of tokens
config.seed = 42;

// Apply to sequences
std::vector<std::vector<int>> sequences = /* ... */;
EfficientBatching::apply_augmentation(sequences, config);

// Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// Augmented: [1, 3, 4, 6, 7, 8, 10]  (tokens 2, 5, 9 dropped)
```

**Use cases:**
- Improve model robustness to missing tokens
- Reduce overfitting
- Simulate noisy input data

### Token Masking

Replace tokens with a mask token (BERT-style):

```cpp
AugmentationConfig config;
config.enable_token_masking = true;
config.token_mask_prob = 0.15f;     // Mask 15% of tokens
config.mask_token_id = 103;         // Mask token ID
config.seed = 42;

EfficientBatching::apply_augmentation(sequences, config);

// Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// Augmented: [1, 103, 3, 4, 103, 6, 7, 103, 9, 10]
```

**Use cases:**
- Masked language modeling
- Denoising autoencoders
- Improve token representations

### Sequence Shuffling

Shuffle adjacent tokens to improve positional robustness:

```cpp
AugmentationConfig config;
config.enable_sequence_shuffle = true;
config.shuffle_prob = 0.05f;  // 5% chance per position
config.seed = 42;

EfficientBatching::apply_augmentation(sequences, config);

// Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// Augmented: [1, 3, 2, 4, 6, 5, 7, 8, 9, 10]  (some adjacent swaps)
```

**Use cases:**
- Reduce positional overfitting
- Improve robustness to word order variations
- Better generalization

### Combined Augmentation

Use multiple strategies together:

```cpp
AugmentationConfig config;
config.enable_token_dropout = true;
config.token_dropout_prob = 0.1f;
config.enable_token_masking = true;
config.token_mask_prob = 0.15f;
config.mask_token_id = 103;
config.enable_sequence_shuffle = true;
config.shuffle_prob = 0.05f;
config.seed = 42;

// Apply all augmentations
EfficientBatching::apply_augmentation(sequences, config);
```

---

## Complete Examples

### Training Loop with Parallel Loading

```cpp
#include "ParallelDataLoader.hpp"
#include "Dataset.hpp"
#include "EncoderDecoderModel.hpp"

// Setup
Dataset dataset(/* config */);
// ... load training data ...

DataLoaderConfig loader_config;
loader_config.batch_size = 32;
loader_config.num_workers = 4;
loader_config.shuffle = true;
loader_config.use_dynamic_batching = true;
loader_config.augmentation_config.enable_token_masking = true;
loader_config.augmentation_config.token_mask_prob = 0.15f;

ParallelDataLoader loader(dataset, loader_config);
EncoderDecoderModel model(/* params */);

// Training loop
int num_epochs = 10;
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << "\n";
    
    DataLoaderIterator iter(loader);
    int step = 0;
    double total_loss = 0.0;
    
    while (auto batch = iter.next()) {
        ++step;
        
        // Forward pass
        auto output = model.forward(batch->sequences);
        
        // Calculate loss
        double loss = calculate_loss(output, batch->sequences);
        total_loss += loss;
        
        // Backward pass
        model.backward(/* gradients */);
        
        // Update weights
        model.update_weights(0.001);  // learning rate
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << "/" << loader.num_batches()
                      << " | Loss: " << loss << "\n";
        }
    }
    
    double avg_loss = total_loss / loader.num_batches();
    std::cout << "Epoch " << (epoch + 1) << " complete | Avg Loss: " 
              << avg_loss << "\n\n";
}
```

### Custom Bucketing Strategy

```cpp
// Define custom buckets for your dataset
BucketConfig custom_config;

// For translation task (English-French)
// Based on analysis: most sentences 10-50 words
custom_config.bucket_boundaries = {15, 25, 35, 50, 70};

// Token budget based on GPU memory
// Assuming 12GB GPU, ~4000 tokens per batch fits comfortably
custom_config.max_tokens_per_batch = 4000;

// Shuffle for randomness
custom_config.shuffle_buckets = true;

// Create batches
auto batches = EfficientBatching::create_bucketed_batches(
    sequences, custom_config, pad_token_id, PaddingStrategy::RIGHT
);

// Analyze efficiency
auto stats = EfficientBatching::calculate_statistics(batches);
std::cout << "Bucketing efficiency: " << (stats.efficiency_score * 100) << "%\n";
```

### Validation Loop

```cpp
// Separate loader for validation (no augmentation, no shuffle)
DataLoaderConfig val_config;
val_config.batch_size = 64;  // Larger batches for inference
val_config.num_workers = 2;
val_config.shuffle = false;  // Sequential for validation
val_config.use_dynamic_batching = true;
// No augmentation for validation

ParallelDataLoader val_loader(val_dataset, val_config);

// Validation
model.set_eval_mode();
DataLoaderIterator val_iter(val_loader);

double total_val_loss = 0.0;
int val_steps = 0;

while (auto batch = val_iter.next()) {
    auto output = model.forward(batch->sequences);
    double loss = calculate_loss(output, batch->sequences);
    total_val_loss += loss;
    ++val_steps;
}

double avg_val_loss = total_val_loss / val_steps;
std::cout << "Validation Loss: " << avg_val_loss << "\n";

model.set_train_mode();
```

---

## Performance Optimization

### Memory Management

**Monitor memory usage:**
```cpp
// Calculate memory requirements
size_t batch_memory = 0;
for (const auto& batch : batches) {
    batch_memory += batch.total_tokens() * sizeof(int);
    batch_memory += batch.total_tokens() * sizeof(int);  // masks
}

std::cout << "Estimated batch memory: " << (batch_memory / 1024.0 / 1024.0) 
          << " MB\n";
```

**Optimize prefetch buffer:**
```cpp
// Balance between throughput and memory
DataLoaderConfig config;

// Low memory system (<8GB RAM)
config.num_workers = 2;
config.prefetch_factor = 1;

// Medium memory system (8-16GB RAM)
config.num_workers = 4;
config.prefetch_factor = 2;

// High memory system (>16GB RAM)
config.num_workers = 8;
config.prefetch_factor = 3;
```

### Throughput Optimization

**Maximize data loading speed:**

1. **Use dynamic batching** to reduce padding overhead
2. **Increase worker threads** if CPU/I/O is bottleneck
3. **Tune prefetch factor** to keep GPU fed
4. **Profile your pipeline** to find bottlenecks

```cpp
// Measure throughput
auto start = std::chrono::high_resolution_clock::now();

DataLoaderIterator iter(loader);
size_t total_sequences = 0;

while (auto batch = iter.next()) {
    total_sequences += batch->sequences.size();
}

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

double throughput = (total_sequences * 1000.0) / duration.count();
std::cout << "Throughput: " << throughput << " sequences/second\n";
```

### Batching Strategy Selection

| Dataset Characteristic | Recommended Strategy | Rationale |
|------------------------|----------------------|-----------|
| Uniform lengths (±20%) | Simple dynamic batching | Minimal padding already |
| Variable lengths (±50%) | Dynamic batching with sorting | Reduces padding significantly |
| Wide variation (±100%+) | Bucketing | Best efficiency for diverse lengths |
| Strict memory budget | Bucketing with token limit | Prevents OOM errors |
| Small dataset (<1000) | Simple batching | Overhead not worth it |
| Large dataset (>10K) | Parallel loading + bucketing | Maximum efficiency |

---

## Best Practices

### Do's ✅

1. **Sort sequences by length** when using dynamic batching
2. **Use bucketing** for datasets with wide length variation
3. **Enable prefetching** for GPU training
4. **Set appropriate worker count** (2-8 typical)
5. **Monitor padding ratio** (target < 20%)
6. **Use data augmentation** for better generalization
7. **Profile your pipeline** to identify bottlenecks
8. **Set random seed** for reproducibility

### Don'ts ❌

1. **Don't use too many workers** (causes thread contention)
2. **Don't prefetch excessively** (wastes memory)
3. **Don't forget to stop loader** (resource leak)
4. **Don't skip validation batching** (can affect metrics)
5. **Don't over-augment** (can hurt performance)
6. **Don't mix train/val loaders** (different configs)

### Configuration Templates

#### Small Dataset (< 1K samples)
```cpp
DataLoaderConfig config;
config.batch_size = 16;
config.num_workers = 1;
config.prefetch_factor = 1;
config.use_dynamic_batching = true;
```

#### Medium Dataset (1K - 100K samples)
```cpp
DataLoaderConfig config;
config.batch_size = 32;
config.num_workers = 4;
config.prefetch_factor = 2;
config.use_dynamic_batching = true;
config.shuffle = true;
```

#### Large Dataset (> 100K samples)
```cpp
DataLoaderConfig config;
config.batch_size = 64;
config.num_workers = 8;
config.prefetch_factor = 3;
config.use_bucketing = true;
config.bucket_config.bucket_boundaries = {10, 20, 40, 80};
config.bucket_config.max_tokens_per_batch = 4096;
config.shuffle = true;
```

#### Inference / Validation
```cpp
DataLoaderConfig config;
config.batch_size = 128;  // Larger batches
config.num_workers = 2;   // Less overhead
config.prefetch_factor = 1;
config.shuffle = false;   // Sequential
// No augmentation
```

---

## Troubleshooting

### Common Issues

#### High Padding Ratio (> 30%)

**Symptoms:** Wasted memory, slow training

**Solutions:**
1. Enable dynamic batching with sorting
2. Switch to bucketing strategy
3. Adjust bucket boundaries
4. Increase batch size (more chances for similar lengths)

```cpp
// Before
auto batches = EfficientBatching::create_dynamic_batches(
    sequences, 16, 0, PaddingStrategy::RIGHT, false
);
// Padding ratio: 35%

// After
auto batches = EfficientBatching::create_dynamic_batches(
    sequences, 32, 0, PaddingStrategy::RIGHT, true
);
// Padding ratio: 18%
```

#### Slow Data Loading

**Symptoms:** GPU idle, low throughput

**Solutions:**
1. Increase number of workers
2. Increase prefetch factor
3. Check for I/O bottleneck (slow disk)
4. Profile worker threads

```cpp
// Diagnose
std::cout << "Queue size: " << loader.queue_size() << "\n";
// If queue_size consistently 0, workers are too slow

// Solution: increase workers
config.num_workers = 8;  // was 2
config.prefetch_factor = 3;  // was 1
```

#### Memory Issues (OOM)

**Symptoms:** Out of memory errors, crashes

**Solutions:**
1. Reduce batch size
2. Reduce prefetch factor
3. Use bucketing with token limit
4. Reduce number of workers

```cpp
// Memory-constrained configuration
DataLoaderConfig config;
config.batch_size = 8;    // smaller batches
config.num_workers = 2;   // fewer workers
config.prefetch_factor = 1;  // minimal prefetch

// Use token budget
BucketConfig bucket_config;
bucket_config.max_tokens_per_batch = 1024;  // strict limit
```

#### Thread Deadlock

**Symptoms:** Loader hangs, no batches returned

**Solutions:**
1. Always call `loader.stop()` when done
2. Check for exceptions in worker threads
3. Ensure dataset is not empty
4. Verify batch_size <= dataset size

```cpp
// Proper cleanup
try {
    ParallelDataLoader loader(dataset, config);
    // ... use loader ...
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
}
// Destructor calls stop() automatically
```

#### Inconsistent Results

**Symptoms:** Different results each run

**Solutions:**
1. Set random seed for reproducibility
2. Disable shuffling if exact order needed
3. Disable augmentation for debugging

```cpp
// Reproducible configuration
config.seed = 42;
config.shuffle = false;  // for debugging
config.augmentation_config.seed = 42;
```

---

## API Reference

### EfficientBatching

#### Static Methods

```cpp
// Create dynamic batches
static std::vector<SequenceBatch> create_dynamic_batches(
    const std::vector<std::vector<int>>& sequences,
    size_t batch_size,
    int pad_token_id = 0,
    PaddingStrategy strategy = PaddingStrategy::RIGHT,
    bool sort_by_length = true
);

// Create bucketed batches
static std::vector<SequenceBatch> create_bucketed_batches(
    const std::vector<std::vector<int>>& sequences,
    const BucketConfig& config,
    int pad_token_id = 0,
    PaddingStrategy strategy = PaddingStrategy::RIGHT
);

// Apply augmentation
static void apply_augmentation(
    std::vector<std::vector<int>>& sequences,
    const AugmentationConfig& config
);

// Calculate statistics
static BatchStatistics calculate_statistics(
    const std::vector<SequenceBatch>& batches
);

// Pad single sequence
static std::vector<int> pad_sequence(
    const std::vector<int>& sequence,
    int target_length,
    int pad_token_id,
    PaddingStrategy strategy
);

// Create attention mask
static std::vector<int> create_attention_mask(
    int original_length,
    int padded_length,
    PaddingStrategy strategy
);
```

### ParallelDataLoader

#### Constructor

```cpp
ParallelDataLoader(const Dataset& dataset, const DataLoaderConfig& config);
```

#### Methods

```cpp
void start();                           // Start background threads
void stop();                            // Stop all threads
void new_epoch();                       // Start new epoch
std::optional<SequenceBatch> next_batch();  // Get next batch

size_t num_batches() const;             // Batches per epoch
size_t current_epoch() const;           // Current epoch number
size_t batches_loaded() const;          // Total batches loaded
size_t queue_size() const;              // Current queue size
bool is_running() const;                // Check if running
```

### DataLoaderIterator

```cpp
DataLoaderIterator(ParallelDataLoader& loader);

std::optional<SequenceBatch> next();    // Get next batch
void reset();                           // Reset to epoch start
size_t batches_returned() const;        // Batches returned
```

---

## Conclusion

The data pipeline enhancements provide production-ready infrastructure for efficient training:

- **20-60% reduction in padding** through dynamic batching and bucketing
- **2-5x faster data loading** with parallel workers and prefetching
- **Better generalization** through data augmentation
- **Flexible configuration** for different use cases

Start with the recommended configurations and tune based on your specific requirements. Monitor padding ratio and throughput to optimize performance.

For more examples, see `src/DataPipelineExample.cpp`.

---

**Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Complete
