# Incremental Training System

## Overview

The Incremental Training System allows you to train your chatbot model continuously without starting from scratch each time. This is essential for ongoing learning with new conversation data, especially when initial training takes days to complete.

## Key Features

- **Session-based training**: Each training run is tracked as a session with full history
- **Data versioning**: Automatic tracking of which data files have been trained
- **Resume capability**: Resume from any previous checkpoint
- **Auto-save**: Automatic checkpoint saving during long training runs
- **Incremental updates**: Train only on new data, or periodically retrain on everything
- **Efficient caching**: Tokenized data caching to speed up repeated training

## Problem Solved

**Current issue:** 12 epochs on 7500 samples takes 6 days. If you get new conversation data, you have to retrain from scratch for another 6 days.

**Solution:** Incremental training lets you:

1. Train once on initial data (6 days)
2. Add new data and train only on that new data (much faster)
3. Periodically do full retrains to maintain overall quality

## Quick Start

### 1. Initialize the System

```bash
./incremental_trainer init vocab.txt chatbot_model.bin
```

This creates the session tracking directory and initializes the trainer.

### 2. Add Training Data

```bash
./incremental_trainer add initial_conversations.txt
./incremental_trainer add more_data.txt
```

### 3. Train on New Data

```bash
./incremental_trainer train 10
```

This trains for 10 epochs on all pending (unprocessed) data files.

### 4. Later: Add More Data and Train Incrementally

```bash
./incremental_trainer add new_conversations_week2.txt
./incremental_trainer train 5
```

This trains for 5 epochs ONLY on the new data file, using the previously trained model as a starting point.

### 5. Check Status

```bash
./incremental_trainer status
```

Shows:

- Total sessions completed
- Total data files trained
- Pending data files
- Latest checkpoint location

### 6. View Training History

```bash
./incremental_trainer history
```

Shows detailed history of all training sessions and data files.

## Advanced Usage

### Periodic Full Retrain

Every N incremental updates, do a full retrain on ALL data:

```bash
./incremental_trainer retrain 10
```

This retrains from scratch on all previously trained data plus any pending data.

### Resume from Interruption

If training is interrupted, resume from the last checkpoint:

```bash
./incremental_trainer resume
```

### Auto-Save During Long Training

The system automatically saves checkpoints during training:

- Every 30 minutes (configurable)
- Every 1000 samples (configurable)

This protects against data loss during very long training runs.

## Configuration

Edit the configuration in your code:

```cpp
IncrementalConfig config;

// Base training parameters
config.base_config.num_epochs = 10;
config.base_config.learning_rate = 0.0001f;
config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;

// Incremental-specific settings
config.session_dir = "training_sessions";
config.max_sessions_to_keep = 10;  // Keep last 10 checkpoints

// Auto-save settings
config.auto_save_enabled = true;
config.auto_save_every_minutes = 30;
config.auto_save_every_samples = 1000;

// Training mode
config.accumulate_all_data = false;  // false = train only on new data
config.periodic_full_retrain = true;
config.full_retrain_every_n_sessions = 10;  // Full retrain every 10 sessions
```

## Training Strategies

### Strategy 1: Pure Incremental (Fastest)

Train only on new data each time:

```bash
# Week 1
./incremental_trainer add week1_data.txt
./incremental_trainer train 10

# Week 2
./incremental_trainer add week2_data.txt
./incremental_trainer train 5

# Week 3
./incremental_trainer add week3_data.txt
./incremental_trainer train 5
```

**Pros:** Very fast updates
**Cons:** Model may forget older patterns

### Strategy 2: Periodic Full Retrain (Balanced)

Mix incremental and full retrains:

```bash
# Weeks 1-9: Incremental
for i in 1 2 3 4 5 6 7 8 9; do
    ./incremental_trainer add week${i}_data.txt
    ./incremental_trainer train 5
done

# Week 10: Full retrain on everything
./incremental_trainer retrain 10
```

**Pros:** Best quality, prevents forgetting
**Cons:** Slower periodic retrains

### Strategy 3: Hybrid Approach

```bash
# Initial: Full training
./incremental_trainer add initial_dataset.txt
./incremental_trainer train 20  # Long initial training

# Monthly: Incremental updates (5 epochs each)
./incremental_trainer add new_month_data.txt
./incremental_trainer train 5

# Quarterly: Full retrain
./incremental_trainer retrain 15
```

## File Structure

```text
project/
├── vocab.txt                    # BPE vocabulary
├── chatbot_model.bin           # Main model (symlink to latest)
└── training_sessions/           # Session tracking directory
    ├── session_history.txt     # Log of all training sessions
    ├── data_registry.txt       # Log of all trained data files
    ├── session_1_checkpoint.bin
    ├── session_2_checkpoint.bin
    ├── session_3_checkpoint.bin
    └── auto_save_session_3.bin  # Auto-saved checkpoints
```

## Performance Expectations

### Example Timeline (7500 samples)

**Initial training:**

- 12 epochs: 6 days
- Model reaches loss ~7.0

**Incremental training (500 new samples):**

- 5 epochs: ~12 hours (20x faster!)
- Fine-tunes on new data

**Full retrain (8000 samples total):**

- 10 epochs: ~6.5 days
- Relearns everything with new data integrated

### Efficiency Gains

| Scenario         | Old Method | Incremental Method | Time Saved |
| ---------------- | ---------- | ------------------ | ---------- |
| Add 500 samples  | 6 days     | 12 hours           | 87%        |
| Add 1000 samples | 6 days     | 1 day              | 83%        |
| Add 2000 samples | 6 days     | 2 days             | 67%        |

## Best Practices

1. **Start with good initial training**: Do a thorough initial training (20+ epochs)

2. **Use incremental for frequent updates**: Add new data weekly/monthly with 5-10 epochs

3. **Schedule periodic full retrains**: Every 10 incremental updates, do a full retrain

4. **Monitor validation loss**: If incremental training causes loss to increase, do a full retrain

5. **Keep checkpoints**: Don't delete old checkpoints until you verify new ones work

6. **Use auto-save**: Enable auto-save for training runs longer than 1 hour

7. **Version your data**: Use descriptive filenames (conversations_2026_week1.txt)

## API Usage (C++)

```cpp
#include "IncrementalTrainer.hpp"

// Initialize
IncrementalConfig config;
config.auto_save_enabled = true;
IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin", config);

// Add new data
trainer.add_new_data("new_conversations.txt");

// Train incrementally
if (trainer.train_incremental(5)) {
    std::cout << "Training successful!\n";
    trainer.print_training_summary();
}

// Get statistics
int total_samples = trainer.get_total_samples_trained();
float total_hours = trainer.get_total_training_time_hours();

// Full retrain every 10 sessions
if (trainer.get_session_history().size() % 10 == 0) {
    trainer.train_full_retrain(10);
}
```

## Troubleshooting

### "No pending data files"

- You need to add data with `./incremental_trainer add <file>` before training

### "Data file already trained"

- This file was processed in a previous session
- Use `retrain` to include it again, or add new data

### Training is still slow

- Incremental training is only fast for SMALL amounts of new data
- If you're adding 50% more data, expect 50% more training time
- Consider using fewer epochs for incremental updates (5 instead of 10)

### Model quality degraded

- Incremental training can cause "catastrophic forgetting"
- Do a full retrain: `./incremental_trainer retrain 15`

### Out of disk space

- Old checkpoints accumulate
- Reduce `max_sessions_to_keep` in config
- Manually delete old sessions from `training_sessions/`

## Comparison: Traditional vs Incremental

### Traditional Training

```bash
# Initial
./chatbot_trainer --data all_data.txt --epochs 12 --output model.bin
# 6 days

# Add 500 new samples - must retrain ALL
cat old_data.txt new_data.txt > all_data.txt
./chatbot_trainer --data all_data.txt --epochs 12 --output model.bin
# 6 days again!
```

### Incremental Training

```bash
# Initial
./incremental_trainer add initial_data.txt
./incremental_trainer train 12
# 6 days

# Add 500 new samples - train ONLY on new
./incremental_trainer add new_500.txt
./incremental_trainer train 5
# 12 hours!
```

## Future Enhancements

- **Active learning**: Automatically identify and prioritize training on difficult examples
- **Data pruning**: Remove redundant training samples
- **Distributed training**: Train across multiple machines
- **Curriculum learning**: Start with easy examples, progress to hard
- **Online learning**: Update model in real-time as users chat

## References

- See `include/IncrementalTrainer.hpp` for full API documentation
- See `src/ChatbotTrainer.hpp` for base training configuration
- Example code in `src/IncrementalTrainingTool.cpp`
