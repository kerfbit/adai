# Training Guide

This comprehensive guide covers all aspects of training ADAI models, from basic quickstart to advanced incremental training and performance optimization.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Methods](#training-methods)
3. [Advanced Training Features](#advanced-training-features)
4. [Performance & Optimization](#performance--optimization)
5. [Internals and Testing](#internals-and-testing)

---

## Quick Start

### Basic Training

The simplest way to train a chatbot model:

```bash
./chatbot_trainer --data conversations.txt --vocab vocab.txt --epochs 10 --output chatbot_model.bin
```

**What this does:**

- Loads conversation data from `conversations.txt`
- Uses BPE tokenizer with `vocab.txt`
- Trains for 10 epochs
- Saves trained model to `chatbot_model.bin`

**For detailed instructions**, see:

- **[Training Example](training-example.md)** - Step-by-step training walkthrough
- **[Enhanced Training Pipeline](enhanced-training-pipeline.md)** - Production-ready training infrastructure

---

## Training Methods

### 1. Standard Training

**Best for:** Initial model training, complete retrains

**Documentation:** [Enhanced Training Pipeline](enhanced-training-pipeline.md)

Train from scratch on a complete dataset:

```bash
./chatbot_trainer \
    --data conversations.txt \
    --vocab vocab.txt \
    --epochs 12 \
    --learning-rate 0.0001 \
    --batch-size 32 \
    --output model.bin
```

**Key Features:**

- Full dataset training
- Configurable hyperparameters
- Learning rate scheduling (warmup + cosine decay)
- Automatic checkpointing
- Validation split support

**Expected Performance:** 12 epochs on 7,500 samples ≈ 6 days

---

### 2. Incremental Training

**Best for:** Continuous learning, quick updates with new data

**Documentation:** [Incremental Training Guide](incremental-training-guide.md)

Add new conversation data without retraining from scratch:

```bash
# Initialize
./incremental_trainer init vocab.txt chatbot_model.bin

# Add data incrementally
./incremental_trainer add week1_conversations.txt
./incremental_trainer train 5

# Later: add more data and train only on new data
./incremental_trainer add week2_conversations.txt
./incremental_trainer train 5  # Much faster!
```

**Key Features:**

- Session-based training tracking
- Data versioning and registry
- Resume from interruption
- Auto-save during long runs
- Periodic full retrains

**Performance Advantage:** 87% time reduction for small data additions

- Traditional: 6 days to retrain everything
- Incremental: 12 hours to train on 500 new samples

**When to use:**

- Adding weekly/monthly conversation logs
- Fine-tuning on specific conversation types
- Continuous model improvement

---

### 3. Project Gutenberg Training

**Best for:** Enhancing language understanding with literary data

**Documentation:** [Project Gutenberg Training Guide](gutenberg-training-guide.md)

Train on high-quality literary texts from 70,000+ free books:

```bash
# Download and train on a single book
./incremental_trainer gutenberg 1342 500  # Pride and Prejudice, 500 pairs
./incremental_trainer train 5

# Batch download multiple books
./incremental_trainer gutenberg-batch 1342,11,84,1661 300
./incremental_trainer train 10
```

**Key Features:**

- Automatic book downloading
- Intelligent text processing (removes headers/footers)
- Question-answer pair generation
- Multiple training pair styles

**Recommended Book Combinations:**

- **General conversation:** 1342 (Pride & Prejudice), 11 (Alice), 76 (Huck Finn), 98 (Tale of Two Cities)
- **Formal/professional:** 1661 (Sherlock), 84 (Frankenstein), 1260 (Jane Eyre), 2701 (Moby Dick)
- **Creative/imaginative:** 11 (Alice), 345 (Dracula), 35 (Time Machine), 16328 (Beowulf)

**Best Practice:** Mix Gutenberg books with real conversation data for best results

---

## Advanced Training Features

### Training Metrics and Logging

**Documentation:** [Training Metrics and Logging](chatbot-trainer-metrics-logging.md)

Enhanced tracking system for monitoring training progress:

**Features:**

- **Perplexity tracking** - Model prediction quality metric
- **Learning rate logging** - Track LR schedule over time
- **Gradient statistics** - Monitor gradient norms and clipping
- **Detailed epoch summaries** - Loss breakdown and timing
- **CSV export** - Export metrics for analysis
- **TensorBoard support** - Visual training monitoring

**Enable enhanced logging:**

```cpp
TrainingConfig config;
config.log_metrics = true;
config.metrics_output_file = "training_metrics.csv";
config.log_perplexity = true;
config.log_gradients = true;
```

**Use cases:**

- Debugging training issues
- Hyperparameter tuning
- Performance optimization
- Research and analysis

---

### Training Improvements (2026)

**Documentation:** [Training Improvements Quick Reference](chatbot-trainer-improvements-2026.md)

Latest enhancements added January 2026:

1. **Warmup + Cosine LR Schedule** - Better convergence
2. **Gradient Clipping** - Training stability
3. **Early Stopping** - Prevent overfitting
4. **Validation Split** - Automatic train/val split
5. **Enhanced Logging** - Detailed metrics
6. **Checkpoint System** - Auto-save progress

**Migration from old trainer:**

```cpp
// Old
ChatbotTrainer trainer(vocab_size, d_model, num_heads);
trainer.train(dataset, 10);

// New
TrainingConfig config;
config.num_epochs = 10;
config.lr_schedule = LRSchedule::WARMUP_COSINE;
config.enable_gradient_clipping = true;
config.validation_split = 0.1;
ChatbotTrainer trainer(vocab_size, d_model, num_heads, config);
trainer.train_with_config(dataset);
```

---

## Performance & Optimization

### Data Pipeline Enhancement

**Documentation:** [Data Pipeline Enhancement](data-pipeline-enhancement.md)

Optimize data loading and batching for faster training:

#### 1. Efficient Batching

```cpp
#include "EfficientBatching.hpp"

EfficientBatching batcher;
auto batches = batcher.create_batches_dynamic_bucketing(
    dataset, 
    batch_size, 
    num_buckets
);  // 20-40% efficiency improvement
```

#### 2. Parallel Data Loading

```cpp
#include "ParallelDataLoader.hpp"

ParallelDataLoader loader(4);  // 4 worker threads
loader.start_prefetching(dataset, batch_size);
// 2-6x speedup with background loading
```

---

### Dataset Enhanced Features

**Documentation:**

- [Dataset Enhanced Features](dataset-enhanced-features.md) - Comprehensive guide
- [Dataset Quick Reference](dataset-quick-reference.md) - Quick API reference

Advanced dataset capabilities for improved training:

**Key Features:**

- **Iterator interface** - Range-based for loops
- **Batch iteration** - Built-in batching support
- **Multiple formats** - Conversation, TSV, JSON, CSV
- **K-fold cross-validation** - Advanced validation
- **Data augmentation** - On-the-fly augmentation
- **Lazy loading** - Memory-efficient large datasets
- **Stratified splitting** - Balanced train/val/test splits

**Example:**

```cpp
Dataset dataset;
dataset.load_from_file("data.txt");
dataset.split_stratified(0.8, 0.1, 0.1, 5);  // Stratified split

// Iterate with augmentation
for (const auto& sample : dataset.augmented_view(augmenter)) {
    train_on_sample(sample);
}

// K-fold cross-validation
for (int fold = 0; fold < 5; fold++) {
    dataset.prepare_fold(fold, 5);
    train(dataset.get_train_split());
    validate(dataset.get_val_split());
}
```

---

### Batch Processing

**Documentation:**

- [Batch Processing Integration](BATCH_PROCESSING_INTEGRATION.md)
- [Batch Processing Quick Reference](BATCH_PROCESSING_QUICK_REFERENCE.md)

Process multiple sequences efficiently:

```cpp
#include "BatchProcessor.hpp"

BatchProcessor processor(max_seq_len);
auto batched = processor.prepare_batch(inputs);
// 2-4x throughput improvement
```

---

### Data Augmentation

**Documentation:**

- [Augmentation Implementation](AUGMENTATION_IMPLEMENTATION.md)
- [Augmentation Quick Reference](AUGMENTATION_QUICK_REFERENCE.md)
- [Augmentation Checklist](AUGMENTATION_CHECKLIST.md)

Expand training data with intelligent augmentation:

**Techniques:**

- Synonym substitution
- Back-translation
- Paraphrasing
- Noise injection
- Character-level perturbations

**Implementation:**

```cpp
DataAugmenter augmenter;
augmenter.set_augmentation_probability(0.3);
auto augmented = augmenter.augment(original_text);
```

---

## Training Strategies

### Strategy 1: Initial Training → Incremental Updates

**Best for:** Production systems with regular new data

```bash
# Week 0: Initial comprehensive training
./chatbot_trainer --data initial_dataset.txt --epochs 20 --output model.bin

# Week 1-9: Quick incremental updates
./incremental_trainer init vocab.txt model.bin
for week in {1..9}; do
    ./incremental_trainer add week${week}_data.txt
    ./incremental_trainer train 5
done

# Week 10: Full retrain to consolidate
./incremental_trainer retrain 15
```

---

### Strategy 2: Literature Enhancement

**Best for:** Improving language quality and diversity

```bash
# Step 1: Train on real conversations
./incremental_trainer add real_conversations.txt
./incremental_trainer train 12

# Step 2: Enhance with literary style
./incremental_trainer gutenberg-batch 1342,11,1661,84 400
./incremental_trainer train 8

# Step 3: Fine-tune on conversations again
./incremental_trainer add more_conversations.txt
./incremental_trainer train 5
```

---

### Strategy 3: Specialized Domain Training

**Best for:** Domain-specific chatbots (medical, legal, technical)

```bash
# Base: General conversation ability
./chatbot_trainer --data general_conversations.txt --epochs 10

# Specialized: Domain-specific data
./incremental_trainer init vocab.txt model.bin
./incremental_trainer add medical_dialogues.txt
./incremental_trainer train 15  # More epochs for specialization

# Maintenance: Mix of general + specialized
./incremental_trainer add general_and_medical_mix.txt
./incremental_trainer train 5
```

---

## Troubleshooting Training Issues

**Documentation:** [Training Fix Strategy](troubleshooting/TRAINING_FIX_STRATEGY.md)

### Common Issues

#### 1. Loss Divergence (NaN/Inf)

- Enable gradient clipping
- Reduce learning rate
- Check for corrupted data

#### 2. Slow Convergence

- Increase learning rate
- Use warmup schedule
- Check batch size

#### 3. Overfitting

- Enable early stopping
- Increase validation split
- Add data augmentation
- Use dropout

#### 4. Memory Issues

- Reduce batch size
- Use lazy loading
- Enable gradient accumulation

#### 5. Long Training Times

- Use parallel data loading
- Enable efficient batching
- Use fewer epochs for incremental updates
- Consider smaller model

---

## Internals and Testing

### Training System Internals

**Documentation:** [Training Internals](training-internals.md)

Deep dive into the training system architecture:

- Loss computation
- Backpropagation through transformer
- Optimizer implementation (Adam, AdamW)
- Learning rate scheduling
- Gradient accumulation
- Mixed precision training

---

### Test Suite

**Documentation:** [Chatbot Trainer Tests](../testing/chatbot-trainer-tests.md)

Comprehensive test coverage:

- Unit tests for trainer components
- Integration tests for full training pipeline
- Performance benchmarks
- Regression tests

Run tests:

```bash
./build/src/chatbot_trainer_tests
```

---

## Quick Reference Summary

| Task | Tool | Documentation |
| ------ | ------ | --------------- |
| Initial training | `chatbot_trainer` | [Enhanced Training Pipeline](enhanced-training-pipeline.md) |
| Add new data | `incremental_trainer` | [Incremental Training Guide](incremental-training-guide.md) |
| Train on books | `incremental_trainer gutenberg` | [Gutenberg Training Guide](gutenberg-training-guide.md) |
| Monitor metrics | Enable logging | [Training Metrics and Logging](chatbot-trainer-metrics-logging.md) |
| Optimize data loading | Use DataLoader/Batching | [Data Pipeline Enhancement](data-pipeline-enhancement.md) |
| Advanced datasets | Use Dataset v2.0 | [Dataset Enhanced Features](dataset-enhanced-features.md) |
| Debug issues | Check docs | [Training Fix Strategy](troubleshooting/TRAINING_FIX_STRATEGY.md) |

---

## Related Documentation

### Core Training

- [Training Example](training-example.md) - Complete walkthrough
- [Training Internals](training-internals.md) - System architecture
- [Enhanced Training Pipeline](enhanced-training-pipeline.md) - Production setup
- [Training Improvements (2026)](chatbot-trainer-improvements-2026.md) - Latest features

### Advanced Training

- [Incremental Training Guide](incremental-training-guide.md) - Continuous learning
- [Project Gutenberg Training](gutenberg-training-guide.md) - Literary data
- [Training Metrics and Logging](chatbot-trainer-metrics-logging.md) - Enhanced monitoring

### Data & Optimization

- [Data Pipeline Enhancement](data-pipeline-enhancement.md) - Efficient batching and loading
- [Dataset Enhanced Features](dataset-enhanced-features.md) - Advanced dataset capabilities
- [Dataset Quick Reference](dataset-quick-reference.md) - Quick API lookup
- [Batch Processing Integration](BATCH_PROCESSING_INTEGRATION.md) - Multi-sequence processing

### Augmentation

- [Augmentation Implementation](AUGMENTATION_IMPLEMENTATION.md) - Setup guide
- [Augmentation Quick Reference](AUGMENTATION_QUICK_REFERENCE.md) - Quick lookup
- [Augmentation Checklist](AUGMENTATION_CHECKLIST.md) - Implementation verification

### Testing & Troubleshooting

- [Chatbot Trainer Tests](../testing/chatbot-trainer-tests.md) - Test suite
- [Training Fix Strategy](troubleshooting/TRAINING_FIX_STRATEGY.md) - Issue resolution

---

## Getting Help

- **Quick questions**: Check the [Quick Reference](dataset-quick-reference.md) docs
- **Training issues**: See [Training Fix Strategy](troubleshooting/TRAINING_FIX_STRATEGY.md)
- **Performance**: See [Data Pipeline Enhancement](data-pipeline-enhancement.md)
- **API details**: See [Training Internals](training-internals.md)
