# IncrementalTrainer Quick Reference

## Quick Commands and Common Workflows

## Essential Commands

### Setup and Initialization

```bash
# Initialize system (uses config.conf in current directory or /etc/adai/config.conf)
incremental_trainer init

# Initialize with custom config
incremental_trainer --config my_config.conf init

# Initialize with explicit paths
incremental_trainer init /path/to/vocab.txt /path/to/model.bin
```

### Adding Training Data

```bash
# Add single file
incremental_trainer add conversations.txt

# Add Gutenberg book
incremental_trainer gutenberg 1342 500  # Pride & Prejudice, 500 pairs

# Add multiple Gutenberg books
incremental_trainer gutenberg-batch 1342,11,84,1661 300
```

### Training

```bash
# Incremental training (new data only)
incremental_trainer train 10  # 10 epochs

# Full retrain (all data from scratch)
incremental_trainer retrain 15  # 15 epochs

# Resume interrupted training
incremental_trainer resume
```

### Status and Monitoring

```bash
# Current status
incremental_trainer status

# Session history
incremental_trainer history
```

### System Management

```bash
# Reset system (interactive prompt)
incremental_trainer reset

# Reset with auto-confirm
incremental_trainer reset --yes

# Reset but keep data registry
incremental_trainer reset --yes --keep-data
```

## Common Workflows

### Initial Training Workflow

```bash
# 1. Create config
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
incremental_trainer init

# 3. Add data
incremental_trainer add initial_data.txt

# 4. Train (long initial training)
incremental_trainer train 20
```

### Weekly Update Workflow

```bash
# Every week:
incremental_trainer add week_N_data.txt
incremental_trainer train 5
```

### Monthly Full Retrain Workflow

```bash
# Every month:
incremental_trainer retrain 15
```

### Gutenberg Literature Training

```bash
# Add classic books for language understanding
incremental_trainer gutenberg 1342 500      # Pride & Prejudice
incremental_trainer gutenberg 11 500        # Alice in Wonderland
incremental_trainer gutenberg 84 500        # Frankenstein
incremental_trainer train 10
```

### Architecture Change Workflow

```bash
# 1. Update config.conf
echo "D_MODEL=768" >> config.conf
echo "NUM_ENCODER_LAYERS=12" >> config.conf

# 2. Reset (keep data registry)
incremental_trainer reset --yes --keep-data

# 3. Full retrain with new architecture
incremental_trainer retrain 20
```

## Configuration Quick Reference

### Minimal config.conf

```bash
# Required
VOCAB_PATH=vocab.txt
```

### Standard config.conf

```bash
# Paths
VOCAB_PATH=vocab.txt
MODEL_PATH=chatbot_model.bin

# Architecture
D_MODEL=512
NUM_HEADS=8
D_FF=2048
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
MAX_SEQ_LENGTH=1024

# Training
LEARNING_RATE=0.0001
NUM_EPOCHS=10
WEIGHT_DECAY=0.01
GRADIENT_CLIP=1.0
BATCH_SIZE=1
```

### Large Model config.conf

```bash
# High-quality large model
D_MODEL=768
NUM_HEADS=12
D_FF=3072
NUM_ENCODER_LAYERS=8
NUM_DECODER_LAYERS=8
MAX_SEQ_LENGTH=2048
LEARNING_RATE=0.00005
```

### Small/Fast Model config.conf

```bash
# Fast experimentation
D_MODEL=256
NUM_HEADS=4
D_FF=1024
NUM_ENCODER_LAYERS=4
NUM_DECODER_LAYERS=4
MAX_SEQ_LENGTH=512
LEARNING_RATE=0.0003
```

## Data Format

### Conversation File Format

```text
INPUT: User message here
RESPONSE: Bot response here

INPUT: Another user message
RESPONSE: Another bot response

INPUT: Third message
RESPONSE: Third response
```

Rules:

- Blank line separates conversation pairs
- `INPUT:` prefix for user messages
- `RESPONSE:` prefix for bot responses
- UTF-8 encoding

## Popular Gutenberg Books

|ID|Title|Author|
|-----|-------------------|----------------------|
|1342|Pride and Prejudice|Jane Austen|
|11|Alice in Wonderland|Lewis Carroll|
|84|Frankenstein|Mary Shelley|
|1661|Sherlock Holmes|Arthur Conan Doyle|
|2701|Moby Dick|Herman Melville|
|16328|Beowulf|Anonymous|
|1260|Jane Eyre|Charlotte Bronte|
|98|A Tale of Two Cities|Charles Dickens|

## Performance Expectations

### Training Time Estimates (7500 samples baseline)

|Operation|Data Size|Epochs|Time|
|----------------------|---------|------|----------|
|Initial training|7500|12|~6 days|
|Incremental (500 new)|500|5|~12 hours|
|Incremental (1000 new)|1000|5|~1 day|
|Full retrain|8000|10|~6.5 days|

### Speedup from Incremental

|New Data %|Traditional|Incremental|Savings|
|----------|-----------|-----------|-------|
|6% (500)|6 days|12 hours|87%|
|13% (1000)|6 days|1 day|83%|
|25% (2000)|6 days|2 days|67%|

### Strategy 1: Pure Incremental (Fastest)

```bash
# Week 1
incremental_trainer add week1.txt && incremental_trainer train 5
# Week 2
incremental_trainer add week2.txt && incremental_trainer train 5
# Week 3
incremental_trainer add week3.txt && incremental_trainer train 5
```

**Pros:** Very fast
**Cons:** May forget old patterns

### Strategy 2: Periodic Retrain (Balanced)

```bash
# Weeks 1-9: Incremental
for i in {1..9}; do
    incremental_trainer add week${i}.txt
    incremental_trainer train 5
done

# Week 10: Full retrain
incremental_trainer retrain 10
```

**Pros:** Best quality
**Cons:** Slower periodic retrains

### Strategy 3: Hybrid (Recommended)

```bash
# Month 1: Initial
incremental_trainer add month1.txt && incremental_trainer train 20

# Months 2-3: Incremental
incremental_trainer add month2.txt && incremental_trainer train 5
incremental_trainer add month3.txt && incremental_trainer train 5

# Quarter end: Full retrain
incremental_trainer retrain 15
```

## Troubleshooting

### Error: "VOCAB_PATH must be set"

```bash
echo "VOCAB_PATH=/path/to/vocab.txt" >> config.conf
```

### Error: "No pending data files"

```bash
incremental_trainer add conversations.txt
```

### Error: "Data file already trained"

```bash
# Option 1: Use retrain to include it
incremental_trainer retrain 10

# Option 2: Add new data instead
incremental_trainer add new_data.txt

# Option 3: Reset to retrain from scratch
incremental_trainer reset --yes --keep-data
incremental_trainer retrain 10
```

### Training is slow

```bash
# Use fewer epochs for incremental updates
incremental_trainer train 3  # instead of 10

# Or verify you're not adding too much data at once
incremental_trainer status  # Check pending files
```

### Model quality degraded

```bash
# Do a full retrain
incremental_trainer retrain 15
```

### Out of disk space

```bash
# Delete old session files
rm training_sessions/session_1_checkpoint.bin
rm training_sessions/session_2_checkpoint.bin
# Keep latest and best only
```

## File Structure

```text
project/
├── config.conf                         # Configuration
├── vocab.txt                           # Vocabulary
├── chatbot_model.bin                  # Main model
└── training_sessions/
    ├── session_history.txt            # History log
    ├── data_registry.txt              # Data tracking
    ├── pending_data.txt               # Pending queue
    ├── session_N_checkpoint.bin       # Checkpoints
    ├── latest_checkpoint.bin          # → latest session
    └── best_checkpoint.bin            # → best validation
```

## Key Concepts

### Incremental Training

- Trains only on NEW data
- Uses existing model as starting point
- Fast (10-100x for small additions)
- Risk of forgetting old patterns

### Full Retrain

- Trains on ALL data from scratch
- Resets model to random weights
- Slow but maintains quality
- Recommended every 10 sessions

### Session

- One training run
- Tracked with ID, metrics, checkpoint
- History preserved in `session_history.txt`

### Data Registry

- Tracks all data files ever trained
- Uses checksums to detect changes
- Prevents duplicate training
- Persists across sessions

### Checkpointing

- Automatic saves during training
- Best checkpoint tracked separately
- Symlinks for easy access
- Resume capability

## Command Reference

|Command|Purpose|Common Usage|
|------------------|--------------------|-----------------------------|
|`init`|Initialize system|`init`|
|`add`|Add data file|`add data.txt`|
|`gutenberg`|Add Gutenberg book|`gutenberg 1342 500`|
|`gutenberg-batch`|Add multiple books|`gutenberg-batch 1342,11,84`|
|`train`|Incremental training|`train 10`|
|`retrain`|Full retrain|`retrain 15`|
|`resume`|Resume last session|`resume`|
|`status`|Show status|`status`|
|`history`|Show history|`history`|
|`reset`|Reset system|`reset --yes --keep-data`|

Override config.conf values:

```bash
export VOCAB_PATH=/custom/vocab.txt
export MODEL_PATH=/custom/model.bin
export D_MODEL=768
export NUM_ENCODER_LAYERS=12
export LEARNING_RATE=0.00005

incremental_trainer train 10
```

## Best Practices

✅ **DO:**

- Start with long initial training (20+ epochs)
- Use incremental for small updates (< 20% data)
- Schedule periodic full retrains (every 10 sessions)
- Monitor validation loss trends
- Keep multiple checkpoints
- Use version control for config.conf

❌ **DON'T:**

- Use incremental forever (causes forgetting)
- Change architecture mid-training
- Delete all checkpoints at once
- Modify data files after adding
- Skip validation monitoring

## Metrics Reference

### Per-Epoch Metrics

- **Loss:** Cross-entropy loss (lower is better)
- **Validation Loss:** Loss on held-out data
- **Perplexity:** exp(loss) - interpretable quality
- **Learning Rate:** Current LR after schedule
- **Training Time:** Wall-clock seconds per epoch

### Good Perplexity Values

- **< 5:** Excellent
- **5-20:** Good
- **20-50:** Fair
- **> 50:** Poor (investigate)

## C++ API Quick Reference

```cpp
#include "IncrementalTrainer.hpp"

// Initialize
IncrementalTrainer trainer("config.conf");

// Add data
trainer.add_new_data("conversations.txt");
trainer.add_gutenberg_book(1342, 500);

// Train
trainer.train_incremental(10);      // Incremental
trainer.train_full_retrain(15);     // Full retrain

// Status
int samples = trainer.get_total_samples_trained();
float hours = trainer.get_total_training_time_hours();
auto history = trainer.get_session_history();

// Checkpointing
std::string latest = trainer.get_latest_checkpoint();
trainer.save_model("custom_checkpoint.bin");
trainer.load_model("custom_checkpoint.bin");

// Configuration
IncrementalConfig& cfg = trainer.get_config();
cfg.auto_save_every_minutes = 15;
trainer.set_config(cfg);
```

## Related Documentation

- **[IncrementalTrainer Internals](incremental-trainer-internals.md)** - Complete technical reference
- **[Incremental Training Guide](../../operations/guides/incremental-training-guide.md)** - User guide
- **[Building Guide](building.md)** - Compilation instructions

---

**Version:** 2.0
**Last Updated:** March 2026
**For:** Developers and power users
