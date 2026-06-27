# Dataset Batch Processing - Quick Reference Guide

## Quick API Reference

### Dataset Batch Methods

```cpp
// Get batch with padding
TokenBatch batch = dataset.get_batch_with_padding(
    SplitType::TRAIN,          // Which split
    0,                          // Start index
    32,                         // Batch size
    tokenizer_fn,               // Tokenization function
    0                           // Pad token ID
);

// Get target batch
TokenBatch targets = dataset.get_target_batch_with_padding(
    SplitType::TRAIN, 0, 32, tokenizer_fn, 0);

// Get dynamic batches (optimized by length)
auto batches = dataset.get_dynamic_batches(
    SplitType::TRAIN,          // Which split
    tokenizer_fn,               // Tokenization function
    32,                         // Max batch size
    10,                         // Length tolerance
    0                           // Pad token ID
);

// Get batch statistics
BatchStats stats = dataset.get_batch_statistics(
    SplitType::TRAIN, tokenizer_fn, 32);
```

### TokenBatchLoader

```cpp
// Configure loader
TokenBatchLoaderConfig config;
config.batch_size = 32;
config.num_workers = 4;
config.prefetch_factor = 2;
config.shuffle = true;
config.use_dynamic_batching = true;
config.load_targets = true;

// Create tokenizer function
auto tokenizer_fn = [&tokenizer](const std::string& text) {
    return tokenizer.encode(text);
};

// Create loader
TokenBatchLoader loader(dataset, config, tokenizer_fn, SplitType::TRAIN);
loader.start();

// Get batches
while (auto input_batch = loader.next_batch()) {
    auto target_batch = loader.next_target_batch();
    // Process...
}

loader.stop();
```

### TokenBatchIterator

```cpp
// Use iterator for easier epoch management
TokenBatchIterator iter(loader);

while (auto batch = iter.next()) {
    auto target = iter.next_target();
    // Process batch...
}

// Reset for next epoch
iter.reset();
```

## Common Patterns

### Pattern 1: Simple Inference

```cpp
auto batches = dataset.get_dynamic_batches(
    SplitType::TEST, tokenizer_fn, 32, 10, 0);

for (const auto& batch : batches) {
    Matrix mask = create_padding_mask(batch);
    // Run inference...
}
```

### Pattern 2: Training Loop

```cpp
TokenBatchLoader loader(dataset, config, tokenizer_fn, SplitType::TRAIN);
loader.start();

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    loader.new_epoch();

    while (auto input = loader.next_batch()) {
        auto target = loader.next_target_batch();

        Matrix input_mask = create_padding_mask(*input);
        Matrix target_mask = create_padding_mask(*target);

        // Forward, backward, update...
    }
}

loader.stop();
```

### Pattern 3: Efficiency Analysis

```cpp
// Compare batch sizes
for (size_t bs : {16, 32, 64, 128}) {
    auto stats = dataset.get_batch_statistics(
        SplitType::TRAIN, tokenizer_fn, bs);

    float efficiency = 1.0f - stats.padding_ratio;
    std::cout << "Batch " << bs
              << ": " << (efficiency * 100) << "% efficient\n";
}
```

### Pattern 4: Dynamic vs Fixed Comparison

```cpp
// Fixed batching
auto stats_fixed = dataset.get_batch_statistics(
    SplitType::TRAIN, tokenizer_fn, 32);

// Dynamic batching
auto batches = dataset.get_dynamic_batches(
    SplitType::TRAIN, tokenizer_fn, 32, 10, 0);
BatchStats stats_dynamic = compute_batch_stats(batches);

float token_reduction = (stats_fixed.total_tokens - stats_dynamic.total_tokens)
                      / (float)stats_fixed.total_tokens * 100;
std::cout << "Token reduction: " << token_reduction << "%\n";
```

## Configuration Guidelines

### Batch Size Selection

|Use Case|Recommended Batch Size|
|----------|----------------------|
|Long sequences (>200 tokens)|8-16|
|Medium sequences (50-200)|32-64|
|Short sequences (<50)|64-128|
|Limited GPU memory|16-32|
|Maximum throughput|64-128|

### Parallel Loading Settings

|Hardware|num_workers|prefetch_factor|
|----------|-------------|-----------------|
|4 CPU cores|2|2|
|8 CPU cores|4|2-3|
|16+ CPU cores|6-8|3|
|Fast SSD|+1-2 workers|+1 factor|
|Slow HDD|-1-2 workers|Same|

### Dynamic Batching Parameters

|Data Characteristics|length_tolerance|
|---------------------|------------------|
|Very uniform lengths|5-10 tokens|
|Moderate variation|10-20 tokens|
|High variation|20-30 tokens|
|Extreme variation|Use fixed batching|

## Performance Expectations

### Dynamic vs Fixed Batching

|Sequence Length Variation|Token Reduction|Efficiency Gain|
|--------------------------|-----------------|-----------------|
|Low (std < 10 tokens)|5-15%|+5-10%|
|Medium (std 10-50)|20-30%|+15-25%|
|High (std > 50)|30-50%|+25-40%|

### Parallel Loading Speedup

|num_workers|Expected Speedup|
|-------------|------------------|
|2|1.5-2x|
|4|2-4x|
|8|3-6x|
|16|4-8x|

## Troubleshooting

### Issue: Low Efficiency (<60%)

Solution:

```cpp
// Use dynamic batching
auto batches = dataset.get_dynamic_batches(
    SplitType::TRAIN, tokenizer_fn, 32, 15, 0);

// Or filter extreme lengths
dataset.filter_by_length(10, 500);
```

### Issue: GPU Waiting for Data

Solution:

```cpp
config.num_workers = 8;        // More workers
config.prefetch_factor = 3;    // Larger buffer
```

### Issue: High Memory Usage

Solution:

```cpp
config.num_workers = 2;        // Fewer workers
config.prefetch_factor = 1;    // Smaller buffer
config.batch_size = 16;        // Smaller batches
```

### Issue: OOM Errors

Solution:

```cpp
config.batch_size = 16;        // Reduce batch size
config.drop_last = true;       // Drop irregular batches
config.use_dynamic_batching = true;  // Reduce padding
```

## Key Utilities

### Create Padding Mask

```cpp
TokenBatch batch = dataset.get_batch_with_padding(...);
Matrix mask = create_padding_mask(batch);

// Use in attention
Matrix attention_output = attention.forward(input, mask);
```

### Compute Batch Statistics

```cpp
std::vector<TokenBatch> batches = {...};
BatchStats stats = compute_batch_stats(batches);

std::cout << "Total tokens: " << stats.total_tokens << "\n";
std::cout << "Actual tokens: " << stats.actual_tokens << "\n";
std::cout << "Padding ratio: " << (stats.padding_ratio * 100) << "%\n";
std::cout << "Efficiency: " << ((1.0f - stats.padding_ratio) * 100) << "%\n";
```

## Files Reference

- **Implementation:** `src/Dataset.hpp`, `src/ParallelDataLoader.hpp`
- **Documentation:** `docs/api/data/dataset-batch-processing.md`
- **Examples:** `src/DatasetBatchProcessingExample.cpp`
- **Summary:** `DATASET_BATCH_PROCESSING_SUMMARY.md`
- **Core Utilities:** `src/BatchProcessor.hpp`

## See Also

- [BatchProcessor API Reference](docs/reference/batchprocessor.md)
- [Dataset API](src/Dataset.hpp)
- [Complete Documentation](docs/api/data/dataset-batch-processing.md)
- [Working Examples](src/DatasetBatchProcessingExample.cpp)
