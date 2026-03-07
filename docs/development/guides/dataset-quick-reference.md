# Dataset v2.0 - Quick Reference Card

## Import

```cpp
#include "Dataset.hpp"
```

## Basic Usage

### Load Data

```cpp
Dataset dataset;
dataset.load_from_file("data.txt");  // Auto-detects: conversation, TSV, JSON, CSV
```

### Split Data

```cpp
// Random split
dataset.split(0.8, 0.1, 0.1);

// Stratified split (balanced by length)
dataset.split_stratified(0.8, 0.1, 0.1, 5);  // 5 bins
```

## Iteration

### Direct Iteration (NEW!)

```cpp
for (const auto& sample : dataset) {
    process(sample.input, sample.target);
}
```

### Batch Iteration (NEW!)

```cpp
for (auto batch : dataset.get_batch_iterator(SplitType::TRAIN, 32)) {
    train_on_batch(batch);  // batch = std::vector<DataSample>
}
```

### Traditional

```cpp
auto train_data = dataset.get_split(SplitType::TRAIN);
for (const auto& sample : train_data) {
    process(sample);
}
```

## K-Fold Cross-Validation (NEW!)

```cpp
dataset.setup_k_fold(5);
for (int fold = 0; fold < 5; ++fold) {
    std::vector<DataSample> train, val;
    dataset.get_fold(fold, train, val);
    train_and_validate(train, val);
}
```

## Data Augmentation (NEW!)

```cpp
dataset.set_augmentation([](const DataSample& s) {
    return DataSample(transform(s.input), s.target);
});
dataset.augment_data(2);  // 2x augmented samples
```

## Filtering (NEW!)

```cpp
// By length
dataset.filter_by_length(10, 500);

// By pattern
dataset.filter_by_pattern("spam", false);  // Remove spam
dataset.filter_by_pattern("important", true);  // Keep important
```

## Preprocessing (NEW!)

```cpp
// Custom preprocessing
dataset.set_preprocessing([](const std::string& text) {
    return normalize(lowercase(text));
});
dataset.apply_preprocessing();

// Quick lowercase
dataset.lowercase();
```

## Lazy Loading (NEW!)

```cpp
LazyDataset lazy("huge_dataset.txt");

// Single sample
auto sample = lazy.get_sample(42);

// Range
auto samples = lazy.load_range(0, 1000);

// Process in chunks
for (size_t i = 0; i < lazy.size(); i += 1000) {
    auto chunk = lazy.load_range(i, 1000);
    process(chunk);
}
```

## Information

```cpp
dataset.size()                      // Total samples
dataset.size(SplitType::TRAIN)      // Train samples
dataset.is_split()                  // Has been split?
dataset.get_stats()                 // Statistics
dataset.print_stats()               // Print stats
dataset.get_num_folds()             // Number of CV folds
```

## Shuffling

```cpp
dataset.shuffle()                          // Shuffle all data
dataset.shuffle_split(SplitType::TRAIN)    // Shuffle one split
```

## File Formats

|Format|File|Auto-Detect|
|--------|------|-------------|
|Conversation|`INPUT: ...\nRESPONSE: ...`|✓|
|TSV|`input\ttarget`|✓|
|JSON|`{"input": "...", "target": "..."}`|✓|
|CSV|`input,target`|✓|

## Complete Example

```cpp
#include "Dataset.hpp"

int main() {
    // Load and preprocess
    Dataset dataset;
    dataset.load_from_file("data.json");
    dataset.filter_by_length(10, 500);
    dataset.lowercase();

    // Stratified split
    dataset.split_stratified(0.7, 0.2, 0.1, 5);

    // Training loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        dataset.shuffle_split(SplitType::TRAIN);

        // Mini-batch training
        for (auto batch : dataset.get_batch_iterator(SplitType::TRAIN, 32)) {
            train(batch);
        }

        // Validation
        for (auto batch : dataset.get_batch_iterator(SplitType::VALIDATION, 64)) {
            validate(batch);
        }
    }

    return 0;
}
```

## Key Improvements (v1.0 → v2.0)

|Feature|v1.0|v2.0|
|---------|------|------|
|Iteration|Copy data|✅ Iterator (no copy)|
|Batching|Manual|✅ Batch iterator|
|Formats|2|✅ 4 (+ JSON, CSV)|
|Splitting|Random|✅ + Stratified|
|CV|Manual|✅ K-fold|
|Augmentation|External|✅ Built-in|
|Filtering|External|✅ Built-in|
|Large datasets|Must fit RAM|✅ LazyDataset|

## Performance Tips

1. **Use iterators** - No data copying
2. **Batch iteration** - More efficient than manual slicing
3. **LazyDataset** - For datasets > 10GB
4. **Stratified split** - Better validation sets
5. **Filter before split** - Faster than after

## Documentation

- **Full Guide:** `docs/guides/dataset-enhanced-features.md`
- **Example:** `src/DatasetEnhancedExample.cpp`
- **Pipeline:** `docs/guides/enhanced-training-pipeline.md`

---

**Version:** 2.0
**Date:** January 2026
**Status:** Production Ready ✅
