# Training Guide

This guide covers all aspects of training ADAI models: queuing data, choosing a training mode, monitoring metrics, tuning hyperparameters, and recovering from common problems.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How Training Works](#how-training-works)
3. [Training Modes](#training-modes)
4. [Dataset Management](#dataset-management)
5. [Configuration](#configuration)
6. [Training Metrics Service](#training-metrics-service)
7. [Training Strategies](#training-strategies)
8. [Monitoring and Inspecting Progress](#monitoring-and-inspecting-progress)
9. [Troubleshooting](#troubleshooting)
10. [Quick Reference](#quick-reference)

---

## Quick Start

New model from scratch in six steps:

```bash
cd /home/rodney/Repos/adai

# 1. Build
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && cd ..

# 2. Create vocabulary
./build/bin/vocab_builder \
    --input sample_training_data.txt \
    --output vocab.txt \
    --vocab-size 5000 \
    --format pairs \
    --stats

# 3. Queue training data
./build/bin/dataset_manager add sample_training_data.txt

# 4. Initialize and train (starts in background)
./build/bin/incremental_trainer init
./build/bin/incremental_trainer train 25

# 5. Monitor progress
tail -f chatbot_server.log

# 6. Check status when done
./build/bin/incremental_trainer status
```

---

## How Training Works

The training system is split across two binaries that operate independently:

| Binary | Responsibility |
| --- | --- |
| `dataset_manager` | Manages the dataset queue (DatasetRegistry) — add files, download from Gutenberg or HuggingFace |
| `incremental_trainer` | Consumes the queue and trains the model |

**All training commands (`train`, `retrain`, `resume`) fork into the background automatically.** The launcher prints the PID and log path, then exits. Output goes to the log file only (default: `chatbot_server.log`; set `LOG_FILE_PATH` in `config.conf`).

**All hyperparameters come from `config.conf`.** There are no per-run CLI flags for learning rate, epochs, or batch size. Set them once in the config file; every training run picks them up at startup.

### Training Data Format

Training data files use alternating `INPUT:` / `RESPONSE:` pairs:

```text
INPUT: Hello
RESPONSE: Hi! How can I help you?
INPUT: What is your name?
RESPONSE: I am the ADAI chatbot assistant.
```

---

## Training Modes

`incremental_trainer` exposes three training modes plus utility commands:

```bash
./build/bin/incremental_trainer [--config <path>] [--gpu-strategy background|full] <command>
```

### `train` — Incremental (pending data only)

```bash
./build/bin/incremental_trainer train [epochs]
```

Trains only on files that have not been trained before (the pending queue). On success, marks those files as trained in the DatasetRegistry. Falls back to `NUM_EPOCHS` in `config.conf` if `[epochs]` is omitted.

**Use when:** Adding new data to an already-trained model.

### `retrain` — Full retrain from scratch

```bash
./build/bin/incremental_trainer retrain [epochs]
```

Resets model weights to the config architecture, then trains on **all** files in the registry (trained + pending). This is the equivalent of a clean restart without losing the data you've accumulated.

**Use when:** Changing model architecture, recovering from a badly diverged model, or doing a scheduled consolidation after many incremental runs.

### `resume` — Continue an interrupted session

```bash
./build/bin/incremental_trainer resume
```

Restarts the most recently incomplete training session without touching the DatasetRegistry. Useful after a crash or deliberate early stop.

**Use when:** Training was interrupted before completing.

### Utility commands

```bash
# Initialize session directory and validate config (run once before first train)
./build/bin/incremental_trainer init

# Reset model to config architecture; optionally preserve registry
./build/bin/incremental_trainer reset [--yes] [--keep-data]

# Print session summary and current pending files
./build/bin/incremental_trainer status

# Print full session history and data registry
./build/bin/incremental_trainer history
```

`--keep-data` preserves all registered files (marking previously trained files as pending again) so a subsequent `retrain` reuses them without re-queuing. `--yes` skips the confirmation prompt.

---

## Dataset Management

`dataset_manager` queues data for `incremental_trainer train` to consume. It never touches the model or the training sessions directly.

```bash
./build/bin/dataset_manager [--config <path>] <command>
```

### Local files

```bash
# Add a local training file
./build/bin/dataset_manager add conversations.txt

# Check what is queued
./build/bin/dataset_manager status
./build/bin/dataset_manager list-pending
./build/bin/dataset_manager list-trained

# Remove all pending files (does not affect already-trained files)
./build/bin/dataset_manager clear-pending
```

### Project Gutenberg

Downloads the book text, strips Gutenberg headers/footers, generates INPUT/RESPONSE pairs, and adds the result to the pending queue.

```bash
# Single book (500 pairs)
./build/bin/dataset_manager gutenberg 1342 500

# Batch download (300 pairs each)
./build/bin/dataset_manager gutenberg-batch 1342,11,84,1661 300
```

Downloaded files are cached in `gutenberg_data/` and can be re-added with `dataset_manager add` if needed.

**Recommended books by training goal:**

| Goal | Command |
| --- | --- |
| General conversation | `gutenberg-batch 1342,11,76,98 400` |
| Formal / professional tone | `gutenberg-batch 1661,84,1260,2701 300` |
| Creative / imaginative | `gutenberg-batch 11,345,35,16328 500` |

**Popular book IDs:**

| ID | Title |
| --- | --- |
| 1342 | Pride and Prejudice (Austen) |
| 11 | Alice in Wonderland (Carroll) |
| 84 | Frankenstein (Shelley) |
| 1661 | Sherlock Holmes (Doyle) |
| 2701 | Moby Dick (Melville) |
| 16328 | Beowulf |
| 1260 | Jane Eyre (Brontë) |
| 98 | A Tale of Two Cities (Dickens) |
| 345 | Dracula (Stoker) |
| 35 | The Time Machine (Wells) |

Find any book ID at: <https://www.gutenberg.org/ebooks/>

**Pair count guidelines:**

| Book length | Recommended pairs |
| --- | --- |
| Short story | 100–200 |
| Novella | 300–500 |
| Novel | 500–1000 |
| Epic / long novel | 1000–2000 |

### HuggingFace datasets

```bash
# Daily conversation pairs (auto-detected format)
./build/bin/dataset_manager huggingface daily_dialog 500

# Instruction-following with explicit field mapping
./build/bin/dataset_manager huggingface tatsu-lab/alpaca 300 train instruction output

# Syntax: huggingface <dataset_id> [num_pairs] [split] [input_field] [output_field]
./build/bin/dataset_manager huggingface databricks/databricks-dolly-15k 500
./build/bin/dataset_manager huggingface Open-Orca/OpenOrca 500 train question response
```

---

## Configuration

All training hyperparameters are set in `config.conf` (or environment variables) and read at startup. Edit the file and re-run; no rebuild required.

### Key training settings

| Key | Default | When to change |
| --- | --- | --- |
| `LEARNING_RATE` | `0.001` | Lower to `0.0001–0.0003` for stable training |
| `NUM_EPOCHS` | `10` | 25 for small datasets; 15–20 for large |
| `BATCH_SIZE` | `32` | Lower if RAM is tight |
| `WEIGHT_DECAY` | `0.01` | L2 regularization; raise if overfitting |
| `GRADIENT_CLIP` | `1.0` | Lower to `0.5` if you see NaN/Inf warnings |
| `SESSION_DIR` | `training_sessions` | Change to separate session directories per model |
| `GPU_STRATEGY` | `background` | Set to `full` on a dedicated training machine |

### Recommended settings by dataset size

| Dataset size | `NUM_EPOCHS` | `LEARNING_RATE` | Notes |
| --- | --- | --- | --- |
| < 1,000 pairs | 25–50 | 0.0003 | Risk of overfitting beyond 30 epochs |
| 1,000–10,000 pairs | 20–30 | 0.0003 | Standard configuration |
| > 10,000 pairs | 15–20 | 0.0005 | Larger `BATCH_SIZE` if RAM allows |

### Learning rate schedule

The trainer always applies `WARMUP_COSINE` scheduling: the learning rate ramps up over the first 10% of steps, then decays via cosine annealing. No config key is needed; it is always active.

### Generation quality scoring during validation

The trainer can compute BLEU/ROUGE scores on a sample of the validation set each epoch. This adds overhead but reveals generation quality beyond loss alone.

```ini
ENABLE_GENERATION_QUALITY_METRICS=true
GENERATION_QUALITY_SAMPLE_SIZE=20    # pairs scored per epoch
GENERATION_QUALITY_MAX_TOKENS=50
GENERATION_QUALITY_ASYNC_THRESHOLD=50  # run scoring in background thread when sample >= this
```

Disable (`false`) during exploratory runs; enable for final evaluation passes.

---

## Training Metrics Service

Metrics are pushed from the trainer to a standalone HTTP daemon. Run the daemon before starting training to enable live monitoring.

```bash
# Start the daemon (port 8081 by default)
./build/bin/metrics_api_server

# Poll metrics while training runs
curl http://localhost:8081/api/metrics/current
curl http://localhost:8081/api/sessions
curl http://localhost:8081/api/metrics/summary
```

The daemon persists metrics to `training_sessions/metrics.jsonl` automatically. Configure push behavior in `config.conf`:

```ini
ENABLE_METRICS_SERVICE=true
METRICS_SERVER_URL=http://localhost:8081
METRICS_PERSIST_EVERY_SAMPLES=100
METRICS_PERSIST_EVERY_SECONDS=30
```

See [OPERATIONS_MANUAL.md §5.3](../OPERATIONS_MANUAL.md#53-training-metrics-service) for the full endpoint and config key reference.

---

## Training Strategies

### Strategy 1: Continuous incremental updates

Best for production systems with a regular stream of new conversation data.

```bash
# One-time initialization
./build/bin/incremental_trainer init
./build/bin/dataset_manager add initial_conversations.txt
./build/bin/incremental_trainer train 25

# Each time new data arrives
./build/bin/dataset_manager add new_week_conversations.txt
./build/bin/incremental_trainer train 10

# Every ~10 incremental sessions, consolidate
./build/bin/incremental_trainer retrain 20
```

**Trade-off:** Fast updates; model may gradually forget older patterns without periodic retrains.

---

### Strategy 2: Literary enhancement

Best for improving language quality and vocabulary diversity when conversation data is limited.

```bash
# Step 1: Train on real conversations first
./build/bin/dataset_manager add real_conversations.txt
./build/bin/incremental_trainer train 20

# Step 2: Add literary style
./build/bin/dataset_manager gutenberg-batch 1342,11,1661,84 400
./build/bin/incremental_trainer train 10

# Step 3: Re-anchor on conversational data
./build/bin/dataset_manager add more_conversations.txt
./build/bin/incremental_trainer train 8
```

**Best practice:** Keep real conversation data as the final training step so the model's conversational register stays grounded.

---

### Strategy 3: Domain specialization

Best for building a chatbot focused on a specific domain (medical, legal, technical).

```bash
# Phase 1: General conversational base
./build/bin/dataset_manager add general_conversations.txt
./build/bin/incremental_trainer train 20

# Phase 2: Domain-specific fine-tuning
./build/bin/dataset_manager add medical_dialogues.txt
./build/bin/incremental_trainer train 15   # more epochs for specialization

# Phase 3: Maintenance — mix general and domain
./build/bin/dataset_manager add general_and_medical_mix.txt
./build/bin/incremental_trainer train 8
```

**Trade-off:** Specialization improves domain accuracy but can reduce general conversational fluency. Maintenance sessions counteract this.

---

## Monitoring and Inspecting Progress

### Follow the training log

```bash
tail -f chatbot_server.log
```

Set `LOG_FILE_PATH` in `config.conf` to change the log location.

### Expected loss progression

| Phase | Expected Loss | Expected Perplexity |
| --- | --- | --- |
| Epochs 1–3 | 4–8 → 4–5 | 1000+ → 50–150 |
| Epochs 4–10 | 2–3 | 10–20 |
| Epochs 11–25 | 1–2 | 3–7 |

### Check session status

```bash
# Current pending files and latest checkpoint
./build/bin/incremental_trainer status

# Full history of all sessions
./build/bin/incremental_trainer history

# Check what data is queued
./build/bin/dataset_manager status
./build/bin/dataset_manager list-pending
```

### Live metrics (requires metrics daemon)

```bash
./build/bin/metrics_api_server &
curl http://localhost:8081/api/metrics/current
curl http://localhost:8081/api/sessions
```

### Verify model files after training

```bash
ls -lh chatbot_model.bin*
# Expected: base file + .config .encoder .decoder .lm_head .vocab symlinks
```

---

## Troubleshooting

### Loss rises or will not decrease past epoch 2

**Cause:** Learning rate too high.

```ini
# config.conf
LEARNING_RATE=0.0003
GRADIENT_CLIP=1.0
```

Then run a full retrain: `./build/bin/incremental_trainer retrain 25`

---

### NaN/Inf warnings in the log

The trainer detects and skips bad gradient updates automatically. If warnings persist:

```ini
LEARNING_RATE=0.0001
GRADIENT_CLIP=0.5
```

Also check training data for extremely long sequences or non-text content.

---

### Perplexity stuck above 100 after 25 epochs

1. Increase `NUM_EPOCHS=50` in `config.conf` and run `train 50`
2. Verify data format — every pair must have `INPUT:` / `RESPONSE:` prefixes
3. Check vocabulary coverage: if many words appear as `<unk>`, rebuild with a larger `--vocab-size`
4. Consider a smaller model (`D_MODEL=256`, `D_FF=1024`) if dataset is tiny (< 500 pairs)

---

### "No pending data. Use DatasetManager to queue training data."

The `train` command found no pending files. Queue data first:

```bash
./build/bin/dataset_manager add my_data.txt
# or
./build/bin/dataset_manager gutenberg 1342 500
```

Then retry `incremental_trainer train`.

---

### Model quality degraded after incremental updates

Incremental training can cause catastrophic forgetting when the new data distribution differs significantly from the old data.

```bash
# Full retrain on all data to recover
./build/bin/incremental_trainer retrain 20
```

---

### Gutenberg download fails

- Verify the book ID at <https://www.gutenberg.org/ebooks/>
- Some books are not available in plain text format
- Manually download and use `dataset_manager add` as a fallback:

```bash
wget "https://www.gutenberg.org/files/1342/1342-0.txt" -O my_book.txt
./build/bin/dataset_manager add my_book.txt
```

---

### Training started but no log output

Training runs in the background; all output goes to the log file, not the terminal.

```bash
tail -f chatbot_server.log
```

If the log file does not exist, check `LOG_FILE_PATH` in `config.conf`. If it is empty, training logs go to `chatbot_server.log` in the working directory.

---

## Quick Reference

| Task | Command |
| --- | --- |
| Add local data file | `./build/bin/dataset_manager add <file>` |
| Download Gutenberg book | `./build/bin/dataset_manager gutenberg <id> <pairs>` |
| Download HuggingFace dataset | `./build/bin/dataset_manager huggingface <id> <pairs>` |
| Check data queue | `./build/bin/dataset_manager status` |
| Initialize trainer | `./build/bin/incremental_trainer init` |
| Train on new (pending) data | `./build/bin/incremental_trainer train [epochs]` |
| Full retrain on all data | `./build/bin/incremental_trainer retrain [epochs]` |
| Resume interrupted session | `./build/bin/incremental_trainer resume` |
| Check session status | `./build/bin/incremental_trainer status` |
| View session history | `./build/bin/incremental_trainer history` |
| Reset to fresh model | `./build/bin/incremental_trainer reset --keep-data --yes` |
| Follow training log | `tail -f chatbot_server.log` |
| Start metrics daemon | `./build/bin/metrics_api_server` |
| Poll current metrics | `curl http://localhost:8081/api/metrics/current` |

### Related documentation

| Document | Status | Notes |
| --- | --- | --- |
| [OPERATIONS_MANUAL.md](../OPERATIONS_MANUAL.md) | Current | Full system reference |
| [COMMANDS.md](COMMANDS.md) | Current | Copy-paste command cheatsheet |
| [troubleshooting/TRAINING_FIX_STRATEGY.md](troubleshooting/TRAINING_FIX_STRATEGY.md) | Current | Detailed issue resolution |
| [incremental-training-guide.md](incremental-training-guide.md) | Outdated | Uses old `incremental_trainer add/gutenberg` syntax; defer to this guide |
| [gutenberg-training-guide.md](gutenberg-training-guide.md) | Outdated | Uses old `incremental_trainer gutenberg` syntax; use `dataset_manager gutenberg` instead |
