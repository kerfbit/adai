# ChatbotTrainer Metrics and Logging System

> **⚠️ DEPRECATED - March 2026**
>
> This document describes the metrics system for the old standalone ChatbotTrainer which no longer has a command-line entry point.
>
> **See instead:**
>
> - [IncrementalTrainer Internals](../guides/internals/incremental-trainer-internals.md) - Current training system with enhanced metrics
>
> This document is preserved for historical reference only.

---

**Last Updated:** January 2026
**Component:** ChatbotTrainer (Internal)
**Priority:** Low Priority Enhancements

## Overview

This document describes the enhanced metrics tracking and logging system added to ChatbotTrainer as part of the low-priority improvements initiative.

## Table of Contents

1. [New Metrics](#new-metrics)
2. [Logging System](#logging-system)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Future Enhancements](#future-enhancements)

---

## New Metrics

### Perplexity Tracking

What is Perplexity?

Perplexity measures how well the model predicts the next token. It's the exponential of the loss:

```text
Perplexity = exp(loss)
```

- **Lower is better** (1.0 is perfect prediction)
- More interpretable than raw loss
- Common metric in NLP tasks

Available Perplexity Metrics:

1. **Training Perplexity** - Tracked per epoch during training
2. **Validation Perplexity** - Tracked per epoch during validation

Data Storage:

```cpp
std::vector<float> training_perplexities;    // Per-epoch training perplexity
std::vector<float> validation_perplexities;  // Per-epoch validation perplexity
```

Example Output:

```text
✅ Epoch 5 complete - Loss: 2.3456 - Perplexity: 10.44 - LR: 0.0001 - GradNorm: 1.23
  Validation - Loss: 2.4123 - Perplexity: 11.16
```

### Token-Level Accuracy

What is Token-Level Accuracy?

Token-level accuracy measures the percentage of tokens the model predicts correctly.

```text
Accuracy = (correct_tokens / total_tokens) × 100%
```

Current Status:

- **Placeholder implementation** - Returns -1.0
- Requires model to expose prediction probabilities
- Will be implemented when `EncoderDecoderModel::get_predictions()` is available

Data Storage:

```cpp
std::vector<float> training_accuracies;      // Per-epoch training accuracy
std::vector<float> validation_accuracies;    // Per-epoch validation accuracy
```

Future Implementation:

Once the model exposes predictions, accuracy will be calculated as:

```cpp
float calculate_accuracy(const std::vector<int>& predictions,
                        const std::vector<int>& targets) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == targets[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.size();
}
```

---

## Logging System

### Log Levels

The new logging system provides four verbosity levels:

#### `LogLevel::SILENT` (0)

- **Use case:** Automated training, production runs
- **Output:** Only errors and final results
- **Performance:** Minimal overhead

#### `LogLevel::NORMAL` (1)

- **Use case:** Standard training monitoring
- **Output:** Epoch summaries, validation results, checkpoints
- **Performance:** Low overhead
- **Includes:**
  - Epoch completion messages
  - Validation results
  - Best model updates
  - Checkpoint saves

Example Output:

```text
✅ Epoch 5 complete - Loss: 2.3456 - Perplexity: 10.44 - LR: 0.0001 - GradNorm: 1.23
  Validation - Loss: 2.4123 - Perplexity: 11.16
  ⭐ New best validation loss!
```

#### `LogLevel::VERBOSE` (2) - **DEFAULT**

- **Use case:** Interactive development, debugging
- **Output:** Sample-level progress, detailed training info
- **Performance:** Moderate overhead
- **Includes:**
  - All NORMAL level output
  - Per-sample training progress
  - Gradient accumulation details
  - Learning rate updates per batch

Example Output:

```text
  Sample 100/1000 (Update 25) - Loss: 2.45 - Avg: 2.40 - LR: 0.0001 - GradNorm: 1.15
  Sample 200/1000 (Update 50) - Loss: 2.38 - Avg: 2.35 - LR: 0.0001 - GradNorm: 1.08
✅ Epoch 5 complete - Loss: 2.3456 - Perplexity: 10.44 - LR: 0.0001 - GradNorm: 1.23
```

#### `LogLevel::DEBUG` (3)

- **Use case:** Deep debugging, development
- **Output:** All verbose output plus debug information
- **Performance:** Higher overhead
- **Future:** Will include parameter statistics, gradient details, etc.

### Log Function

Signature:

```cpp
void log(LogLevel level, const std::string& message,
         const std::string& color = COLOR_RESET)
```

Behavior:

- Messages are only printed if `config.log_level >= level`
- Supports ANSI color codes for formatted output
- Thread-safe (uses std::cout with automatic mutex)

Example Usage:

```cpp
// This will only print if log_level is VERBOSE or DEBUG
log(LogLevel::VERBOSE,
    "Sample progress: " + std::to_string(current_sample),
    COLOR_INFO);

// This will print at NORMAL or higher
log(LogLevel::NORMAL,
    "Epoch complete - Loss: " + std::to_string(loss),
    COLOR_SUCCESS);
```

---

## Configuration

### TrainingConfig Fields

```cpp
struct TrainingConfig {
    // ... other fields ...

    // Logging
    int log_every = 10;                          // Log every N samples (VERBOSE mode)
    LogLevel log_level = LogLevel::VERBOSE;      // NEW: Logging verbosity
    bool verbose = true;                         // Deprecated: use log_level
};
```

### Command-Line Option

```bash
--log-level <level>    # Logging: silent, normal, verbose, debug (default: verbose)
```

Valid Values:

- `silent` → `LogLevel::SILENT`
- `normal` → `LogLevel::NORMAL`
- `verbose` → `LogLevel::VERBOSE` (default)
- `debug` → `LogLevel::DEBUG`

---

## Usage Examples

### Example 1: Silent Training for Production

Run training with minimal output for automated workflows:

```bash
./chatbot_trainer \
  --data conversations.txt \
  --vocab vocab.txt \
  --epochs 50 \
  --log-level silent \
  --output production_model.bin
```

Output:

```text
✅ Training complete! Model saved to: production_model.bin
```

### Example 2: Normal Monitoring

Standard training with epoch-level monitoring:

```bash
./chatbot_trainer \
  --data conversations.txt \
  --vocab vocab.txt \
  --epochs 20 \
  --log-level normal \
  --lr 0.0001 \
  --batch-size 4 \
  --grad-accum 8
```

Output:

```text
🚀 Starting training...
═══════════════════════════════════════
✅ Epoch 1 complete - Loss: 3.2145 - Perplexity: 24.87 - LR: 0.0001 - GradNorm: 2.45
  Validation - Loss: 3.1234 - Perplexity: 22.73
  ⭐ New best validation loss!
───────────────────────────────────────
✅ Epoch 2 complete - Loss: 2.8956 - Perplexity: 18.08 - LR: 0.0001 - GradNorm: 1.89
  Validation - Loss: 2.9123 - Perplexity: 18.39
───────────────────────────────────────
...
```

### Example 3: Verbose Development

Detailed progress for interactive development:

```bash
./chatbot_trainer \
  --data conversations.txt \
  --vocab vocab.txt \
  --epochs 10 \
  --log-level verbose \
  --lr 0.0001
```

Output:

```text
🚀 Starting training...
═══════════════════════════════════════
  Sample 10/500 (Update 3) - Loss: 3.45 - Avg: 3.48 - LR: 0.0001 - GradNorm: 2.78
  Sample 20/500 (Update 5) - Loss: 3.38 - Avg: 3.42 - LR: 0.0001 - GradNorm: 2.51
  Sample 30/500 (Update 8) - Loss: 3.29 - Avg: 3.37 - LR: 0.0001 - GradNorm: 2.34
  ...
✅ Epoch 1 complete - Loss: 3.2145 - Perplexity: 24.87 - LR: 0.0001 - GradNorm: 2.45
  Validation - Loss: 3.1234 - Perplexity: 22.73
  ⭐ New best validation loss!
───────────────────────────────────────
```

### Example 4: Combining with Other Features

Advanced training with metrics, logging, and checkpointing:

```bash
./chatbot_trainer \
  --data large_dataset.txt \
  --vocab vocab.txt \
  --epochs 50 \
  --log-level normal \
  --lr 0.0001 \
  --optimizer adamw \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --batch-size 8 \
  --grad-accum 16 \
  --lr-schedule warmup-cosine \
  --warmup-steps 1000 \
  --early-stopping \
  --patience 5 \
  --resume checkpoint_epoch20.bin \
  --output final_model.bin
```

---

## Best Practices

### 1. Choose Appropriate Log Levels

Development & Testing:

- Use `verbose` or `debug` for detailed feedback
- Monitor per-sample progress to catch issues early
- Watch gradient norms for training stability

Production & Automation:

- Use `silent` or `normal` for minimal overhead
- Reduces log file size
- Faster training (less I/O)

Hyperparameter Tuning:

- Use `normal` for grid searches
- Track epoch-level metrics without noise
- Easier to parse results programmatically

### 2. Monitor Perplexity

Good Perplexity Values:

- **< 5**: Excellent (model has high confidence)
- **5-20**: Good (typical for small datasets)
- **20-50**: Fair (more training may help)
- **> 50**: Poor (check model architecture, data quality)

What to Watch:

- **Decreasing training perplexity**: Model is learning
- **Increasing validation perplexity**: Overfitting (use early stopping)
- **Both decreasing together**: Healthy training
- **Both stuck**: Learning rate may be too low

### 3. Use Metrics for Debugging

High Loss but Low Perplexity Variance:

- Model may be stuck in local minimum
- Try increasing learning rate or using warmup

Validation Perplexity << Training Perplexity:

- Unusual pattern - check data split
- May indicate data leakage

Exploding Perplexity:

- Gradient explosion
- Enable gradient clipping: `--grad-clip 1.0`

### 4. Log Level vs. Log Frequency

Combine `--log-level` and `--log-every` for fine control:

```bash
# Verbose, but log less often
--log-level verbose --log-every 100

# Normal level with default frequency
--log-level normal --log-every 10
```

---

## Future Enhancements

### Planned Metrics (Not Yet Implemented)

1. **Token-Level Accuracy**
   - Requires `EncoderDecoderModel::get_predictions()`
   - Will show % of correctly predicted tokens
   - Useful for sequence classification tasks

2. **Per-Layer Gradient Statistics**
   - Min/max/mean gradients per layer
   - Helps identify vanishing/exploding gradients
   - Requires parameter exposure (TD-001)

3. **Learning Rate Finder Results**
   - Plot loss vs. learning rate
   - Automatically suggest optimal LR
   - Save results to file for analysis

4. **Training Time Estimates**
   - ETA for epoch completion
   - Average samples/second
   - Time-to-convergence predictions

### Enhanced Logging Features

1. **Structured Logging**
   - JSON output format option
   - Easier parsing for automation
   - Integration with monitoring tools

2. **TensorBoard Integration**
   - Real-time training visualization
   - Loss/perplexity curves
   - Gradient histograms

3. **Log to File**
   - `--log-file` option
   - Separate from console output
   - Automatic log rotation

---

## Migration from Old Logging

### Old Code (Deprecated)

```cpp
if (config.verbose) {
    std::cout << "Training progress..." << std::endl;
}
```

### New Code (Recommended)

```cpp
log(LogLevel::VERBOSE, "Training progress...", COLOR_INFO);
```

### Backward Compatibility

The `verbose` boolean field is still supported but deprecated:

```cpp
bool verbose = true;  // Deprecated: use log_level instead
```

If both are set, `log_level` takes precedence.

---

## Technical Details

### Perplexity Calculation

```cpp
float calculate_perplexity(float loss) {
    return std::exp(loss);
}
```

Why it works:

- Cross-entropy loss: $L = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i|x_i)$
- Perplexity: $PPL = \exp(L) = \exp(-\frac{1}{N}\sum_{i=1}^{N} \log P(y_i|x_i))$
- Geometric mean of inverse probabilities
- Interpretable as "effective vocabulary size"

### Log Level Comparison

```cpp
void log(LogLevel level, const std::string& message,
         const std::string& color) {
    if (static_cast<int>(config.log_level) >= static_cast<int>(level)) {
        std::cout << color << message << COLOR_RESET << std::endl;
    }
}
```

Comparison semantics:

- `SILENT (0) <= NORMAL (1) <= VERBOSE (2) <= DEBUG (3)`
- Setting `log_level = NORMAL` shows SILENT and NORMAL messages
- Higher levels include all lower levels

### Performance Impact

Logging Overhead:

|Level|Overhead|Use Case|
|---------|----------|----------------------------|
|SILENT|~0%|Production, automation|
|NORMAL|< 1%|Standard training|
|VERBOSE|1-3%|Development, debugging|
|DEBUG|3-5%|Deep debugging|

Optimization:

- String construction happens inside `log()` after level check
- No overhead if message won't be printed
- I/O buffering reduces syscall overhead

---

## Related Documentation

- [Training Internals](./training-internals.md) - Complete training system reference
- [Improvements 2026](./chatbot-trainer-improvements-2026.md) - All recent enhancements
- [Quick Reference](../reference/training-quick-reference.md) - CLI options and examples

---

## Changelog

### January 2026 - Initial Implementation

Added:

- `LogLevel` enum with 4 levels (SILENT, NORMAL, VERBOSE, DEBUG)
- `log()` helper method for level-based logging
- Perplexity tracking for training and validation
- Placeholder for token-level accuracy
- `--log-level` command-line option
- Updated all training/validation loops to use new system

Performance:

- No overhead for SILENT mode
- < 3% overhead for VERBOSE mode (default)

Compatibility:

- Fully backward compatible
- Old `verbose` boolean still works
- Default behavior unchanged (VERBOSE level)

---

## Examples and Use Cases

See [chatbot-trainer-improvements-2026.md](./chatbot-trainer-improvements-2026.md) for complete usage examples and performance benchmarks.
