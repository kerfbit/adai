# Dataset Abstraction - Enhanced Features Documentation

## Overview

The Dataset abstraction has been significantly enhanced with production-ready features for advanced training infrastructure. This document details the improvements and new capabilities.

## Version History

- **v1.0** (January 2026): Initial implementation with basic train/val/test splitting
- **v2.0** (January 2026): Enhanced with iterators, batch iteration, multiple formats, k-fold CV, augmentation, filtering, and lazy loading

---

## New Features (v2.0)

### 1. Iterator Interface

**Purpose:** Enable range-based for loops and memory-efficient iteration

**Usage:**

```cpp
Dataset dataset;
dataset.load_from_file("data.txt");

// Range-based for loop
for (const auto& sample : dataset) {
    std::cout << sample.input << " -> " << sample.target << "\n";
}

// Manual iteration
for (auto it = dataset.begin(); it != dataset.end(); ++it) {
    process(it->input, it->target);
}
```

**Benefits:**

- No data copying - direct access to samples
- Standard C++ idioms
- Compatible with STL algorithms

---

### 2. Batch Iterator

**Purpose:** Iterate over dataset in configurable batches for mini-batch training

**Usage:**

```cpp
Dataset dataset;
dataset.load_from_file("data.txt");
dataset.split(0.8, 0.1, 0.1);

// Iterate training data in batches of 32
for (auto batch : dataset.get_batch_iterator(SplitType::TRAIN, 32)) {
    // batch is std::vector<DataSample>
    train_on_batch(batch);
}

// Iterate entire dataset in batches
for (auto batch : dataset.get_batch_iterator(64)) {
    process_batch(batch);
}
```

**Benefits:**

- Efficient mini-batch training
- No manual batch slicing
- Automatic batch size handling

---

### 3. JSON Format Support

**Purpose:** Load datasets from JSON files

**Supported Formats:**

**Line-delimited JSON (JSONL):**

```json
{"input": "What is AI?", "target": "Artificial Intelligence"}
{"input": "Explain ML", "target": "Machine Learning basics"}
```

**JSON Array:**

```json
[
  {"input": "Question 1", "target": "Answer 1"},
  {"input": "Question 2", "target": "Answer 2"}
]
```

**Usage:**

```cpp
Dataset dataset;
dataset.load_from_file("data.json");  // Auto-detects JSON format
```

---

### 4. CSV Format Support

**Purpose:** Load datasets from CSV files

**Format:**

```csv
input,target
"Hello","Hi there!"
"Goodbye","See you!"
```

**Features:**

- Automatic header detection and skipping
- Quote handling
- Configurable delimiter (default: comma)
- Whitespace trimming

**Usage:**

```cpp
Dataset dataset;
dataset.load_from_file("data.csv");  // Auto-detects CSV format
```

---

### 5. Stratified Splitting

**Purpose:** Create balanced train/val/test splits based on sample characteristics

**How it Works:**

- Bins samples by total length (input + target)
- Splits each bin proportionally
- Ensures balanced distribution across splits

**Usage:**

```cpp
Dataset dataset;
dataset.load_from_file("data.txt");

// Stratified split with 5 length bins
dataset.split_stratified(0.8, 0.1, 0.1, 5);
```

**Benefits:**

- Prevents length bias in splits
- Better generalization
- More reliable validation

---

### 6. K-Fold Cross-Validation

**Purpose:** Enable k-fold cross-validation for robust model evaluation

**Usage:**

```cpp
Dataset dataset;
dataset.load_from_file("data.txt");

// Setup 5-fold CV
dataset.setup_k_fold(5);

// Train on each fold
for (int fold = 0; fold < dataset.get_num_folds(); ++fold) {
    std::vector<DataSample> train_data, val_data;
    dataset.get_fold(fold, train_data, val_data);

    // Train and validate
    train_model(train_data);
    evaluate_model(val_data);
}
```

**Benefits:**

- Better model evaluation
- Reduce overfitting
- Use all data for validation

---

### 7. Data Augmentation

**Purpose:** Expand training data through transformations

**Usage:**

```cpp
Dataset dataset;
dataset.load_from_file("data.txt");

// Define augmentation function
dataset.set_augmentation([](const DataSample& sample) {
    // Example: Add noise, paraphrase, back-translate, etc.
    std::string augmented_input = add_noise(sample.input);
    return DataSample(augmented_input, sample.target);
});

// Generate 2 augmented samples per original
dataset.augment_data(2);
```

**Augmentation Ideas:**

- Synonym replacement
- Paraphrasing
- Back-translation
- Adding noise
- Text perturbation

---

### 8. Data Filtering and Preprocessing

**Purpose:** Clean and preprocess datasets

**Filtering by Length:**

```cpp
Dataset dataset;
dataset.load_from_file("data.txt");

// Keep only samples with total length between 10 and 100 chars
dataset.filter_by_length(10, 100);
```

**Filtering by Pattern:**

```cpp
// Keep samples containing "question"
dataset.filter_by_pattern("question", true);

// Remove samples containing "spam"
dataset.filter_by_pattern("spam", false);
```

**Preprocessing:**

```cpp
// Set preprocessing function
dataset.set_preprocessing([](const std::string& text) {
    std::string processed = text;
    // Lowercase
    std::transform(processed.begin(), processed.end(),
                   processed.begin(), ::tolower);
    // Remove punctuation, normalize, etc.
    return processed;
});

// Apply to all samples
dataset.apply_preprocessing();
```

**Quick Lowercase:**

```cpp
dataset.lowercase();  // Convenience method
```

---

### 9. Lazy Loading (LazyDataset)

**Purpose:** Handle very large datasets that don't fit in memory

**How it Works:**

- Indexes file positions on initialization
- Loads samples on-demand
- Minimal memory footprint

**Usage:**

```cpp
LazyDataset lazy_dataset("huge_dataset.txt");

std::cout << "Dataset size: " << lazy_dataset.size() << "\n";

// Load single sample
auto sample = lazy_dataset.get_sample(42);

// Load range into memory
auto samples = lazy_dataset.load_range(0, 100);  // First 100 samples

// Process in chunks
for (size_t i = 0; i < lazy_dataset.size(); i += 1000) {
    auto chunk = lazy_dataset.load_range(i, 1000);
    process_chunk(chunk);
}
```

**Benefits:**

- Handle datasets larger than RAM
- Fast initialization
- Flexible chunk processing

---

## Complete Example

```cpp
#include "Dataset.hpp"

int main() {
    // Load dataset (auto-detects format)
    Dataset dataset;
    dataset.load_from_file("training_data.json");

    // Filter and preprocess
    dataset.filter_by_length(10, 500);
    dataset.lowercase();

    // Stratified split
    dataset.split_stratified(0.7, 0.2, 0.1, 5);

    // Train with batch iteration
    for (int epoch = 0; epoch < 10; ++epoch) {
        // Shuffle each epoch
        dataset.shuffle_split(SplitType::TRAIN);

        // Mini-batch training
        for (auto batch : dataset.get_batch_iterator(SplitType::TRAIN, 32)) {
            train_on_batch(batch);
        }

        // Validation
        auto val_data = dataset.get_split(SplitType::VALIDATION);
        evaluate(val_data);
    }

    return 0;
}
```

---

## API Reference

### Dataset Class

#### Constructors

```cpp
Dataset(unsigned int seed = 42)
```

#### File Operations

```cpp
bool load_from_file(const std::string& filepath)
bool save_to_file(const std::string& filepath, const std::string& format = "conversation")
```

#### Data Management

```cpp
void add_sample(const std::string& input, const std::string& target)
void add_samples(const std::vector<DataSample>& samples)
void clear()
```

#### Splitting

```cpp
void split(float train_ratio, float val_ratio, float test_ratio)
void split_stratified(float train_ratio, float val_ratio, float test_ratio, int num_bins)
std::vector<DataSample> get_split(SplitType split_type) const
```

#### Shuffling

```cpp
void shuffle()
void shuffle_split(SplitType split_type)
```

#### Iteration

```cpp
iterator begin()
iterator end()
const_iterator begin() const
const_iterator end() const
BatchIterator get_batch_iterator(SplitType split_type, size_t batch_size) const
BatchIterator get_batch_iterator(size_t batch_size) const
```

#### K-Fold Cross-Validation

```cpp
void setup_k_fold(int k)
void get_fold(int fold_idx, std::vector<DataSample>& train_data, std::vector<DataSample>& val_data) const
int get_num_folds() const
```

#### Data Augmentation

```cpp
void set_augmentation(std::function<DataSample(const DataSample&)> fn)
void augment_data(int num_augmented = 1)
```

#### Preprocessing and Filtering

```cpp
void set_preprocessing(std::function<std::string(const std::string&)> fn)
void apply_preprocessing()
void filter_by_length(size_t min_length, size_t max_length)
void filter_by_pattern(const std::string& pattern, bool keep_matching = true)
void lowercase()
```

#### Information

```cpp
size_t size() const
size_t size(SplitType split_type) const
bool empty() const
bool is_split() const
const DatasetStats& get_stats() const
void print_stats() const
```

### LazyDataset Class

```cpp
LazyDataset(const std::string& filepath)
DataSample get_sample(size_t index) const
std::vector<DataSample> load_range(size_t start, size_t count) const
size_t size() const
```

---

## File Format Summary

| Format | Extension | Auto-Detect | Example |
| -------- | ----------- | ------------- | --------- |
| Conversation | .txt | ✓ | `INPUT: ...\nRESPONSE: ...` |
| TSV | .tsv | ✓ | `input\ttarget\n` |
| JSON | .json | ✓ | `{"input": "...", "target": "..."}` |
| CSV | .csv | ✓ | `input,target\n` |

---

## Performance Considerations

### Memory Usage

- **Standard Dataset:** Loads entire dataset into RAM
  - Best for: Datasets < 10GB
  - Fastest iteration

- **LazyDataset:** Loads samples on-demand
  - Best for: Datasets > 10GB
  - Minimal memory usage
  - Slower per-sample access

### Iteration Performance

- **Direct iteration:** `for (auto& sample : dataset)` - Fastest
- **Split iteration:** `dataset.get_split()` - Copies data (slower)
- **Batch iteration:** `get_batch_iterator()` - Copies batches (moderate)

### Recommendations

1. Use iterators when possible (no copying)
2. For large datasets, use LazyDataset
3. Filter/preprocess before splitting
4. Shuffle splits, not entire dataset (when split)

---

## Testing

Comprehensive tests available in `tests/test_dataset.cpp`:

- Iterator functionality
- Batch iteration
- JSON/CSV parsing
- Stratified splitting
- K-fold CV
- Augmentation
- Filtering and preprocessing
- Lazy loading

Run tests:

```bash
cd build
ctest -R Dataset --output-on-failure
```

---

## Future Enhancements

Potential additions:

- Memory-mapped file I/O for even larger datasets
- Parallel data loading (multi-threaded)
- Streaming dataset iterator
- Dynamic batching by sequence length
- Built-in augmentation strategies
- Integration with external data sources (S3, databases)

---

## Migration Guide (v1.0 → v2.0)

All v1.0 features remain compatible. New code benefits:

**Before (v1.0):**

```cpp
auto train_data = dataset.get_split(SplitType::TRAIN);
for (const auto& sample : train_data) {
    train(sample);
}
```

**After (v2.0):**

```cpp
// Option 1: Direct iteration (no copying)
for (const auto& sample : dataset) {
    train(sample);
}

// Option 2: Batch iteration
for (auto batch : dataset.get_batch_iterator(SplitType::TRAIN, 32)) {
    train_batch(batch);
}
```

---

## Summary

The enhanced Dataset abstraction provides:

✅ **Iterator interface** - Standard C++ iteration
✅ **Batch iteration** - Efficient mini-batch training
✅ **Multiple formats** - JSON, CSV, TSV, conversation
✅ **Stratified splitting** - Balanced dataset splits
✅ **K-fold CV** - Robust model evaluation
✅ **Data augmentation** - Expand training data
✅ **Filtering/preprocessing** - Clean datasets
✅ **Lazy loading** - Handle very large datasets
✅ **Production-ready** - Fully tested and documented

---

**For more examples, see:** `src/DatasetEnhancedExample.cpp`

End of Documentation
