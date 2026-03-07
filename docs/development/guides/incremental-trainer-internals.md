# IncrementalTrainer System - Internal Documentation

**Last Updated:** March 2026
**Component:** IncrementalTrainer
**Status:** Primary Training System

## Overview

The `IncrementalTrainer` is the comprehensive training system for ADAI chatbot models. It provides session-based training with data versioning, automatic checkpointing, and incremental learning capabilities. All model training is performed through this system.

**Key Point:** There is no standalone `chatbot_trainer` command-line tool. All training uses the `incremental_trainer` tool which wraps the `IncrementalTrainer` class.

## Architecture

### Components Hierarchy

```text
┌─────────────────────────────────────┐
│   incremental_trainer (CLI tool)    │  Entry point
│   (IncrementalTrainingTool.cpp)     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      IncrementalTrainer Class       │  Session management
│   (IncrementalTrainer.cpp/.hpp)     │  Data versioning
│                                      │  Checkpointing
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      ChatbotTrainer Class           │  Internal training engine
│   (ChatbotTrainer.cpp/.hpp)         │  (Not directly accessible)
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    EncoderDecoderModel Class        │  Model implementation
│  (EncoderDecoderModel.cpp/.hpp)     │
└─────────────────────────────────────┘
```

### File Locations

- **Header:** `src/IncrementalTrainer.hpp`
- **Implementation:** `src/IncrementalTrainer.cpp`
- **CLI Tool:** `src/IncrementalTrainingTool.cpp`
- **Internal Engine:** `src/ChatbotTrainer.cpp/.hpp`

## Configuration System

### Primary: config.conf File

Training configuration is loaded from `config.conf` following this priority:

1. Explicit path: `--config /path/to/config.conf`
2. Current directory: `./config.conf`
3. System-wide: `/etc/adai/config.conf`
4. Environment variables override file values
5. Default values as fallback

### Configuration Format

```bash
# config.conf

# Required
VOCAB_PATH=/path/to/vocab.txt

# Optional (with defaults)
MODEL_PATH=chatbot_model.bin
D_MODEL=512
NUM_HEADS=8
D_FF=2048
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
MAX_SEQ_LENGTH=1024

# Training hyperparameters
LEARNING_RATE=0.0001
NUM_EPOCHS=10
WEIGHT_DECAY=0.01
GRADIENT_CLIP=1.0
BATCH_SIZE=1
```

### IncrementalConfig Structure

```cpp
struct IncrementalConfig {
    TrainingConfig base_config;  // Base training configuration

    // Session management
    std::string session_dir = "training_sessions";
    int max_sessions_to_keep = 50;

    // Data management
    std::string data_registry_file = "data_registry.txt";
    bool cache_tokenized_data = false;
    std::string tokenized_cache_dir = "tokenized_cache";

    // Auto-save settings
    bool auto_save_enabled = true;
    int auto_save_every_samples = 1000;
    int auto_save_every_minutes = 30;

    // Checkpointing
    bool save_incremental_checkpoints = true;
    std::string checkpoint_dir = "checkpoints";

    // Symlink management
    bool enable_checkpoint_symlinks = true;
    std::string latest_symlink_name = "latest_checkpoint.bin";
    std::string best_symlink_name = "best_checkpoint.bin";
};
```

## Data Structures

### 1. TrainingSession

```cpp
struct TrainingSession {
    int session_id = 0;
    int samples_trained = 0;
    int epochs_completed = 0;
    float final_loss = 0.0f;
    float final_validation_loss = 0.0f;
    std::string checkpoint_path;
    std::chrono::system_clock::time_point timestamp;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;

    // Per-epoch metrics
    std::vector<float>  per_epoch_losses;
    std::vector<float>  per_epoch_validation_losses;
    std::vector<float>  per_epoch_learning_rates;
    std::vector<double> training_time_per_epoch;
    std::vector<float>  per_epoch_perplexities;
    std::vector<float>  per_epoch_validation_perplexities;
};
```

**Purpose:** Track complete information about each training session.

Used by:

- Session history logging
- Dashboard display
- Performance analysis
- Resume functionality

### 2. DataVersion

```cpp
struct DataVersion {
    std::string data_file;
    std::string checksum;
    int num_samples = 0;
    std::chrono::system_clock::time_point added_time;
    bool trained = false;
};
```

**Purpose:** Track which data files have been trained.

**Checksum:** SHA-256 hash to detect file modifications.

Usage:

- Prevents duplicate training
- Incremental training decisions
- Data provenance tracking

## IncrementalTrainer Class

### Public Methods

#### Constructors

```cpp
// Primary constructor - loads from config.conf
explicit IncrementalTrainer(const std::string& config_file_path);

// Explicit paths constructor
IncrementalTrainer(const std::string& vocab_path,
                   const std::string& model_path);

// Full configuration constructor
IncrementalTrainer(const std::string& vocab_path,
                   const std::string& model_path,
                   const IncrementalConfig& cfg);
```

Primary Entry Point:

```cpp
IncrementalTrainer trainer("config.conf");
```

Process:

1. Load configuration from file
2. Build tokenizer from vocab
3. Construct model with specified architecture
4. Load existing model if available
5. Load session history
6. Load data registry
7. Initialize auto-save state

#### Data Management

##### add_new_data()

```cpp
bool add_new_data(const std::string& data_file);
```

**Purpose:** Add a conversation data file to the pending queue.

Process:

1. Check if file already trained (via checksum)
2. If new, add to `pending_data_files` list
3. Save pending list to disk
4. Return true if added, false if already trained

Example:

```cpp
if (trainer.add_new_data("conversations_week1.txt")) {
    // File added successfully
} else {
    // File already trained or error
}
```

##### add_new_data_batch()

```cpp
bool add_new_data_batch(const std::vector<std::string>& data_files);
```

**Purpose:** Add multiple files at once.

**Efficiency:** Single disk write for pending list.

##### add_gutenberg_book()

```cpp
bool add_gutenberg_book(int book_id, int num_pairs = 500);
```

**Purpose:** Download and process a Project Gutenberg book.

Process:

1. Download book text from gutenberg.org
2. Parse into conversation pairs using sliding window
3. Save to `gutenberg_data/gutenberg_<id>.txt`
4. Add to pending queue

Popular Books:

- 1342: Pride and Prejudice
- 11: Alice in Wonderland
- 84: Frankenstein
- 1661: Sherlock Holmes
- 2701: Moby Dick

Example:

```cpp
trainer.add_gutenberg_book(1342, 500);  // 500 pairs from Pride & Prejudice
```

##### add_gutenberg_books()

```cpp
bool add_gutenberg_books(const std::vector<int>& book_ids,
                         int num_pairs_per_book = 300);
```

**Purpose:** Batch download multiple books.

**Parallel:** Uses OpenMP for concurrent downloads if available.

#### Training Methods

##### train_incremental()

```cpp
bool train_incremental(int num_epochs);
```

**Purpose:** Train only on pending (new) data.

Process:

1. Load all pending data files
2. Split into train/validation
3. Tokenize and cache
4. Create `ChatbotTrainer` with current model
5. Train for specified epochs
6. Update data registry (mark as trained)
7. Save checkpoint
8. Clear pending list
9. Update session history

**Performance:** Fast - only processes new data.

**Use Case:** Weekly/monthly incremental updates.

Example:

```cpp
// After adding new data
trainer.add_new_data("week2_conversations.txt");
trainer.train_incremental(5);  // 5 epochs on new data only
```

##### train_on_new_data_only()

```cpp
bool train_on_new_data_only(int num_epochs);
```

**Alias for:** `train_incremental()`

##### train_full_retrain()

```cpp
bool train_full_retrain(int num_epochs);
```

**Purpose:** Retrain from scratch on ALL data (trained + pending).

Process:

1. Reset model weights to random initialization
2. Load ALL data files (from registry + pending)
3. Combine and shuffle
4. Train from scratch
5. Update registry
6. Save checkpoint

**Performance:** Slow - processes all historical data.

**Use Case:** Periodic quality refresh (every 10 sessions).

Example:

```cpp
trainer.train_full_retrain(15);  // Full retrain, 15 epochs
```

#### Session Management

##### resume_last_session()

```cpp
bool resume_last_session();
```

**Purpose:** Resume interrupted training.

Process:

1. Load last checkpoint from session history
2. Restore model state
3. Return success/failure

**Note:** Does not automatically continue training - you must call train methods.

##### get_session_history()

```cpp
std::vector<TrainingSession> get_session_history() const;
```

**Returns:** All completed training sessions.

##### get_current_session()

```cpp
TrainingSession get_current_session() const;
```

**Returns:** Active session or last session if none active.

#### Status and Reporting

##### print_training_summary()

```cpp
void print_training_summary() const;
```

Prints:

- Total sessions completed
- Total samples trained
- Total training time
- Best validation loss
- Trained data files count
- Pending data files count

##### print_session_history()

```cpp
void print_session_history() const;
```

**Prints:** Detailed history with sparklines for loss trends.

##### print_data_registry()

```cpp
void print_data_registry() const;
```

**Prints:** All data files with trained status and checksums.

##### get_total_samples_trained()

```cpp
int get_total_samples_trained() const;
```

**Returns:** Sum of samples from all trained data files.

##### get_total_training_time_hours()

```cpp
float get_total_training_time_hours() const;
```

**Returns:** Total wall-clock training time across all sessions.

#### Configuration

##### set_config()

```cpp
void set_config(const IncrementalConfig& cfg);
```

**Purpose:** Update configuration.

**Note:** Does not rebuild model - call `reset_model_for_config()` after.

##### get_config()

```cpp
IncrementalConfig& get_config();
```

**Returns:** Reference to current configuration.

##### reset_model_for_config()

```cpp
void reset_model_for_config();
```

**Purpose:** Rebuild model with current config architecture.

**Warning:** Discards all learned weights - model returns to random initialization.

**Use Case:** Switching architecture (e.g., 512→768 dimensions).

##### reset_all()

```cpp
bool reset_all(bool keep_data_registry = false);
```

**Purpose:** Complete system reset.

Process:

1. Delete all checkpoints
2. Backup current model to `.bak`
3. Clear session history
4. Clear pending data
5. Optionally preserve/delete data registry
6. Rebuild model from config

**Use Case:** Starting fresh or changing architecture.

#### Checkpointing

##### save_model()

```cpp
bool save_model(const std::string& path);
```

**Purpose:** Save model to specific path.

##### load_model()

```cpp
bool load_model(const std::string& path);
```

**Purpose:** Load model from specific path.

##### get_latest_checkpoint()

```cpp
std::string get_latest_checkpoint() const;
```

**Returns:** Path to most recent checkpoint.

**Symlink:** Also available as `latest_checkpoint.bin` if symlinks enabled.

### Private Methods

#### build_model()

```cpp
void build_model();
```

**Purpose:** Single entry point for model construction.

Process:

1. Create tokenizer from vocab file
2. Construct `EncoderDecoderModel` with config parameters
3. Transfer tokenizer ownership to model

**Note:** This is the ONLY place that instantiates `EncoderDecoderModel`.

#### initialize_session()

```cpp
bool initialize_session();
```

**Purpose:** Start new training session.

**Creates:** Empty `TrainingSession` with next ID.

#### finalize_session()

```cpp
bool finalize_session(int samples_trained, int epochs_completed,
                      float final_loss, float final_val_loss);
```

**Purpose:** Complete current session.

Process:

1. Update session with final metrics
2. Save checkpoint
3. Update best checkpoint if needed
4. Create/update symlinks
5. Save session history to disk

#### Auto-Save State

```cpp
bool should_auto_save();
void perform_auto_save();
```

Triggers:

- Every N samples processed
- Every N minutes elapsed

**Saves to:** `auto_save_session_<id>.bin`

## Command-Line Interface

### Usage Pattern

```bash
incremental_trainer [--config <path>] <command> [options]
```

### Global Options

- `--config <path>` - Path to config.conf (overrides auto-discovery)

### Commands

#### init

```bash
incremental_trainer init [vocab] [model]
```

**Purpose:** Initialize training system.

Args:

- `vocab` - Vocabulary file path (default: from config)
- `model` - Model file path (default: from config)

Creates:

- Session directory
- Empty session history
- Empty data registry

#### add

```bash
incremental_trainer add <data_file>
```

**Purpose:** Add conversation data file to pending queue.

Example:

```bash
incremental_trainer add conversations.txt
```

#### gutenberg

```bash
incremental_trainer gutenberg <book_id> [num_pairs]
```

**Purpose:** Download and add Project Gutenberg book.

Example:

```bash
incremental_trainer gutenberg 1342 500  # Pride & Prejudice, 500 pairs
```

#### gutenberg-batch

```bash
incremental_trainer gutenberg-batch <id1,id2,id3> [num_pairs_each]
```

**Purpose:** Batch download multiple books.

Example:

```bash
incremental_trainer gutenberg-batch 1342,11,84 300
```

#### train

```bash
incremental_trainer train [epochs]
```

**Purpose:** Train on pending data (incremental).

Args:

- `epochs` - Number of epochs (default: from config)

Example:

```bash
incremental_trainer train 10
```

#### retrain

```bash
incremental_trainer retrain [epochs]
```

**Purpose:** Full retrain on all data from scratch.

Example:

```bash
incremental_trainer retrain 15
```

#### reset

```bash
incremental_trainer reset [--yes] [--keep-data]
```

**Purpose:** Reset system and rebuild model.

Options:

- `--yes` - Skip confirmation prompt
- `--keep-data` - Preserve data registry (mark untrained)

Example:

```bash
incremental_trainer reset --yes --keep-data
```

#### resume

```bash
incremental_trainer resume
```

**Purpose:** Resume from last checkpoint.

#### status

```bash
incremental_trainer status
```

**Purpose:** Show current training status.

#### history

```bash
incremental_trainer history
```

**Purpose:** Show session history and data registry.

## Training Workflow Examples

### Initial Training

```bash
# 1. Prepare config.conf
cat > config.conf << EOF
VOCAB_PATH=vocab.txt
D_MODEL=512
NUM_HEADS=8
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
LEARNING_RATE=0.0001
NUM_EPOCHS=10
EOF

# 2. Initialize
incremental_trainer --config config.conf init

# 3. Add initial data
incremental_trainer add initial_conversations.txt

# 4. Train
incremental_trainer train 20  # Long initial training
```

### Weekly Incremental Updates

```bash
# Week 1
incremental_trainer add week1_new_data.txt
incremental_trainer train 5

# Week 2
incremental_trainer add week2_new_data.txt
incremental_trainer train 5

# ... continue weekly
```

### Periodic Full Retrain

```bash
# Every 10 weeks
incremental_trainer retrain 15
```

### Using Gutenberg Data

```bash
# Add classic literature for language understanding
incremental_trainer gutenberg-batch 1342,11,84,1661,2701 400
incremental_trainer train 10
```

### Architecture Change

```bash
# 1. Update config.conf
echo "D_MODEL=768" >> config.conf
echo "NUM_ENCODER_LAYERS=12" >> config.conf

# 2. Reset with data preservation
incremental_trainer reset --yes --keep-data

# 3. Full retrain with new architecture
incremental_trainer retrain 20
```

## Performance Characteristics

### Incremental Training

**Speed:** 10-100x faster than full retrain (for small data additions)

Example:

- Initial 7500 samples, 12 epochs: 6 days
- Add 500 samples, 5 epochs: 12 hours

**Trade-off:** May gradually forget old patterns (catastrophic forgetting)

### Full Retrain

**Speed:** Proportional to total data size

Example:

- 8000 samples, 10 epochs: ~6.5 days

**Benefit:** Maintains quality across all data

### Tokenization Caching

Pre-tokenization provides 10-100x speedup:

- Old approach: Tokenize on every epoch
- New approach: Tokenize once, cache tokens

### Gradient Accumulation

Simulates larger batch sizes:

```cpp
config.base_config.batch_size = 1;
config.base_config.gradient_accumulation_steps = 32;
// Effective batch size: 32
```

## Metrics and Monitoring

### Per-Epoch Metrics

- **Training Loss:** Cross-entropy loss on training set
- **Validation Loss:** Cross-entropy loss on validation set
- **Learning Rate:** Current LR (after schedule adjustments)
- **Training Time:** Wall-clock seconds for epoch
- **Perplexity:** exp(loss) - interpretable quality metric

### Best Checkpoint Tracking

Automatically saves checkpoint when validation loss improves:

```cpp
if (val_loss < best_validation_loss) {
    best_validation_loss = val_loss;
    best_checkpoint_path = checkpoint_path;
    // Update best_checkpoint.bin symlink
}
```

### Dashboard Display

Live dashboard during training shows:

- Session progress
- Current epoch/sample
- Loss trends (sparklines)
- Perplexity
- Time metrics
- ETA

## Callback System

### Per-Epoch Callback

Internal `ChatbotTrainer` supports epoch callbacks:

```cpp
trainer.set_epoch_callback([](int epoch, int total,
                               float loss, float val_loss, float lr) {
    // Custom epoch-end logic
    display_dashboard(...);
    save_checkpoint_if_needed(...);
});
```

### Per-Sample Callback

For detailed monitoring:

```cpp
trainer.set_sample_callback([](int sample, int total,
                                float loss, float grad_norm) {
    // Custom per-sample logic
    update_progress_bar(...);
});
```

## File Structure

```text
project/
├── config.conf                      # Configuration
├── vocab.txt                        # BPE vocabulary
├── chatbot_model.bin               # Main model (or symlink)
├── training_sessions/               # Session tracking
│   ├── session_history.txt         # All sessions
│   ├── data_registry.txt           # Data file tracking
│   ├── pending_data.txt            # Pending queue
│   ├── session_1_checkpoint.bin    # Session checkpoints
│   ├── session_2_checkpoint.bin
│   ├── auto_save_session_2.bin     # Auto-saves
│   ├── latest_checkpoint.bin       # Symlink to latest
│   └── best_checkpoint.bin         # Symlink to best
└── gutenberg_data/                  # Downloaded books
    ├── gutenberg_1342.txt
    └── gutenberg_11.txt
```

## Learning Rate Schedules

Configured via `config.base_config.lr_schedule`:

### Available Schedules

1. **CONSTANT** - Fixed LR
2. **LINEAR_WARMUP** - Gradual increase
3. **COSINE_DECAY** - Smooth cosine decrease
4. **WARMUP_COSINE** - Warmup then cosine (recommended)
5. **STEP_DECAY** - Discrete steps
6. **EXPONENTIAL_DECAY** - Continuous exponential

### Default: WARMUP_COSINE

```cpp
config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
```

Benefits:

- Stable early training (warmup)
- Smooth convergence (cosine)
- Proven for transformers

## Data Format

### Conversation File Format

```text
INPUT: Hello, how are you?
RESPONSE: I'm doing well, thank you for asking!

INPUT: What's the weather like?
RESPONSE: I don't have access to weather data, but you can check online.

INPUT: Tell me a joke
RESPONSE: Why did the programmer quit? They didn't get arrays!
```

Rules:

- Blank line between pairs
- `INPUT:` prefix for user messages
- `RESPONSE:` prefix for bot messages
- UTF-8 encoding
- Lines trimmed of whitespace

## Best Practices

### 1. Configuration Management

✅ **DO:**

- Use version control for config.conf
- Document architecture changes
- Test config changes with small data first

❌ **DON'T:**

- Change architecture mid-training
- Use different configs for same model

### 2. Data Management

✅ **DO:**

- Use descriptive filenames with dates
- Keep original conversation files
- Version your data
- Check pending queue before training

❌ **DON'T:**

- Modify files after adding (checksum will fail)
- Delete data files during training
- Reuse filenames

### 3. Training Strategy

✅ **DO:**

- Start with long initial training (20+ epochs)
- Use incremental for frequent small updates
- Schedule periodic full retrains (every 10 sessions)
- Monitor validation loss for overfitting

❌ **DON'T:**

- Do incremental forever (causes forgetting)
- Skip validation monitoring
- Train too many epochs on small data additions

### 4. Checkpointing

✅ **DO:**

- Enable auto-save for long runs
- Keep multiple checkpoint versions
- Test checkpoints before deleting old ones
- Use best_checkpoint.bin for deployment

❌ **DON'T:**

- Delete checkpoints immediately
- Disable auto-save for multi-day training
- Overwrite only checkpoint

### 5. Architecture Selection

Small Model (Fast, less quality):

```text
D_MODEL=256
NUM_HEADS=4
NUM_ENCODER_LAYERS=4
NUM_DECODER_LAYERS=4
```

Medium Model (Balanced):

```text
D_MODEL=512
NUM_HEADS=8
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
```

Large Model (Best quality, slow):

```text
D_MODEL=768
NUM_HEADS=12
NUM_ENCODER_LAYERS=8
NUM_DECODER_LAYERS=8
```

## Troubleshooting

### "VOCAB_PATH must be set in config.conf"

**Cause:** Missing or empty VOCAB_PATH in config.

Solution:

```bash
echo "VOCAB_PATH=/path/to/vocab.txt" >> config.conf
```

### "No pending data files"

**Cause:** Trying to train without adding data.

Solution:

```bash
incremental_trainer add conversations.txt
```

### "Data file already trained"

**Cause:** File checksum matches registry.

Solutions:

1. Use retrain: `incremental_trainer retrain 10`
2. Add new data instead
3. Clear registry: `incremental_trainer reset --keep-data`

### Training is slower than expected

Causes:

1. Large data addition (proportional slowdown)
2. Inefficient architecture
3. No gradient accumulation

Solutions:

1. Use fewer epochs for incremental (5 instead of 10)
2. Enable gradient accumulation
3. Use smaller model for experimentation

### Model quality degraded

**Cause:** Catastrophic forgetting from too many incremental updates.

Solution:

```bash
incremental_trainer retrain 15
```

### Out of disk space

**Cause:** Too many checkpoints.

Solutions:

1. Reduce `max_sessions_to_keep` in config
2. Delete old sessions manually
3. Use compression for archived checkpoints

## Related Documentation

- [Incremental Training User Guide](../../operations/guides/incremental-training-guide.md) - User-focused guide
- [Building Guide](building.md) - Compilation instructions
- [Configuration Guide](../reference/configuration-reference.md) - config.conf reference
- [ChatbotTrainer Internals](archive/chatbot-trainer-internals.md) - Internal engine (archived)

## Migration from Old ChatbotTrainer

If you have scripts using the old (now removed) `chatbot_trainer` command:

### Old Approach (Removed)

```bash
chatbot_trainer \
    --data conversations.txt \
    --vocab vocab.txt \
    --epochs 10 \
    --lr 0.0001 \
    --output model.bin
```

### New Approach (Current)

```bash
# 1. Create config.conf
cat > config.conf << EOF
VOCAB_PATH=vocab.txt
LEARNING_RATE=0.0001
NUM_EPOCHS=10
EOF

# 2. Use incremental_trainer
incremental_trainer --config config.conf init
incremental_trainer add conversations.txt
incremental_trainer train 10
```

Benefits:

- Session tracking
- Resume capability
- Incremental updates
- Better monitoring

## Performance Benchmarks

### Tokenization (1000 samples)

|Method|Time/Epoch|Speedup|
|-----------------|------------|---------|
|Old (per-epoch)|5.0s|1x|
|New (cached)|0.05s|100x|

### Training Time (7500 samples, 12 epochs)

|Method|Time|Notes|
|-----------------------|----------|---------------|
|Initial training|6 days|Full training|
|Incremental (500 new)|12 hours|20x faster|
|Full retrain (8000)|6.5 days|Proportional|

### Memory Usage

|Component|Memory|
|---------------------------------|---------|
|Model (512-dim)|~200 MB|
|Tokenized cache (1000 samples)|~10 MB|
|Session history|~1 MB|
|Total overhead|~15 MB|

## Future Enhancements

### Planned Features

1. **Distributed Training** - Multi-GPU/multi-node training
2. **Active Learning** - Identify difficult examples
3. **Data Pruning** - Remove redundant samples
4. **Online Learning** - Real-time updates during inference
5. **Curriculum Learning** - Easy→hard progression
6. **Transfer Learning** - Pre-trained model initialization

### Technical Debt

See [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) for current limitations and planned improvements.

## API Reference

### C++ Usage

```cpp
#include "IncrementalTrainer.hpp"
#include "Config.hpp"

int main() {
    // Load from config.conf
    IncrementalTrainer trainer("config.conf");

    // Add data
    trainer.add_new_data("conversations.txt");

    // Train incrementally
    if (trainer.train_incremental(10)) {
        std::cout << "Training successful!\n";
        trainer.print_training_summary();
    }

    // Get metrics
    int total_samples = trainer.get_total_samples_trained();
    float total_hours = trainer.get_total_training_time_hours();

    // Periodic full retrain
    auto history = trainer.get_session_history();
    if (history.size() % 10 == 0) {
        trainer.train_full_retrain(15);
    }

    return 0;
}
```

### Static Helper

```cpp
// Convert ServiceConfig → IncrementalConfig
ServiceConfig svc = ConfigLoader::load("config.conf");
IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc);

// Customize
config.auto_save_every_minutes = 15;
config.max_sessions_to_keep = 100;

// Use
IncrementalTrainer trainer("vocab.txt", "model.bin", config);
```

---

**Version:** 2.0
**Last Updated:** March 2026
**Maintainer:** ADAI Development Team
