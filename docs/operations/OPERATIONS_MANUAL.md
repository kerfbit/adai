# ADAI Chatbot — Operations Manual

**Version:** 1.0  
**Project:** ADAI — Adaptive Dialogue AI (Encoder-Decoder Transformer Chatbot)  
**Build System:** CMake 3.x / C++17  
**Repository:** `/home/rodney/Repos/adai`

---

## Table of Contents

1. [Introduction & Scope](#1-introduction--scope)
2. [System Requirements & Prerequisites](#2-system-requirements--prerequisites)
3. [Building from Source](#3-building-from-source)
4. [Vocabulary Creation](#4-vocabulary-creation)
5. [Model Training](#5-model-training)
   - 5.1 [Incremental Training](#51-incremental-training)
   - 5.2 [Dataset Management](#52-dataset-management)
   - 5.3 [Training Metrics Service](#53-training-metrics-service)
   - 5.4 [Training Hyperparameter Reference](#54-training-hyperparameter-reference)
6. [Running the Chatbot](#6-running-the-chatbot)
   - 6.1 [CLI Chatbot](#61-cli-chatbot)
   - 6.2 [Qt GUI Chatbot](#62-qt-gui-chatbot)
   - 6.3 [API Server](#63-api-server)
   - 6.4 [Model Service Manager](#64-model-service-manager)
7. [Configuration Reference](#7-configuration-reference)
8. [RAG Configuration & Activation](#8-rag-configuration--activation)
9. [Deployment](#9-deployment)
   - 9.1 [Docker Deployment](#91-docker-deployment)
   - 9.2 [systemd Service Deployment](#92-systemd-service-deployment)
10. [Monitoring & Log Management](#10-monitoring--log-management)
11. [Windows Cross-Compilation](#11-windows-cross-compilation)
12. [Troubleshooting](#12-troubleshooting)
    - 12.1 [Build Issues](#121-build-issues)
    - 12.2 [Training Issues](#122-training-issues)
    - 12.3 [Model Loading Issues](#123-model-loading-issues)
    - 12.4 [Generation Quality Issues](#124-generation-quality-issues)
    - 12.5 [GUI Issues](#125-gui-issues)
    - 12.6 [Special Token Issues](#126-special-token-issues)
    - 12.7 [Context Length Issues](#127-context-length-issues)
13. [Quick Reference Card](#13-quick-reference-card)

---

## 1. Introduction & Scope

This manual consolidates all operational knowledge for building, training, running, deploying, and maintaining the ADAI chatbot system. It covers every mode of operation: command-line chatbot, Qt GUI, HTTP API server, Docker containers, and systemd services.

### Architecture Summary

ADAI is an encoder-decoder transformer model built entirely in C++17 with no Python runtime dependency. The system includes:

- **`incremental_trainer`** — trains the encoder-decoder model (incremental, retrain, and resume modes)
- **`dataset_manager`** — manages the training dataset queue (local files, Gutenberg, HuggingFace)
- **`metrics_api_server`** — HTTP daemon for live training metrics with SQLite/PostgreSQL persistence
- **`registry_server`** — HTTP daemon for distributed dataset queue coordination
- **`mns_server`** — Model Name Service for model identity, lifecycle, and artifact registry
- **`mns_cli`** — CLI client for querying and managing the MNS
- **`chatbot`** (CLI) — interactive chatbot on the command line
- **`chatbot_gui`** — Qt5 graphical interface
- **`chatbot_api_server`** — HTTP REST API server
- **`vocab_builder`** — BPE tokenizer vocabulary construction
- **RAG module** — Retrieval-Augmented Generation, wired into the API server

The server-side daemons (`metrics_api_server`, `registry_server`, `mns_server`) are deployed together as the **server bundle** via `install_server_bundle.sh`. See [Server Bundle Deployment](deployment/SERVER_BUNDLE_DEPLOYMENT.md) for full details including SQLite/PostgreSQL database setup, tarball packaging, and systemd service management.

### Key Static Libraries

| Library | Contents |
| --- | --- |
| `adai_core` | Core math, tensors, activation functions, metrics service, SQLite/PostgreSQL backends |
| `adai_transformer` | Transformer layers, attention, positional encoding |
| `adai_models` | EncoderDecoderModel, TextGenerator, KV cache |
| `adai_nlp` | BPETokenizer, vocab utilities |
| `adai_api` | ChatbotAPI, DocumentStore, RAGInference |
| `adai_metrics_api` | TrainingMetricsAPI REST layer (session-scoped routes, DB query endpoints) |
| `adai_mns` | ModelNameService, ModelNameClient (SQLite registry + HTTP client) |

---

## 2. System Requirements & Prerequisites

### Minimum Hardware

| Component | Development | Production |
| --- | --- | --- |
| CPU | 1 core | 2 cores |
| RAM | 2 GB | 4 GB |
| Disk | 5 GB | 10 GB |

### Required Software (Linux)

```bash
# Build essentials
sudo apt-get install build-essential cmake git

# OpenMP (parallel training)
sudo apt-get install libomp-dev

# spdlog (logging)
sudo apt-get install libspdlog-dev

# Qt5 (GUI only)
sudo apt-get install qt5-default libqt5widgets5 libqt5gui5 qtbase5-dev

# Docker (containerized deployment)
sudo apt-get install docker.io docker-compose
```

### Required Software (Windows cross-compile)

```bash
sudo apt-get install mingw-w64
```

### Supported Linux Distributions

Ubuntu 18.04+, Debian 10+, CentOS/RHEL 7+, Fedora 30+, Arch Linux (any systemd-based distro).

---

## 3. Building from Source

### Quick Build

```bash
cd /home/rodney/Repos/adai
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Targets

| Target | Command | Description |
| --- | --- | --- |
| All | `make -j$(nproc)` | Build everything |
| Incremental trainer | `make incremental_trainer -j$(nproc)` | Training executable (all modes) |
| Dataset manager | `make dataset_manager -j$(nproc)` | Dataset queue management |
| Metrics API server | `make metrics_api_server -j$(nproc)` | Training metrics HTTP daemon |
| Registry server | `make registry_server -j$(nproc)` | Distributed dataset coordination |
| CLI chatbot | `make chatbot -j$(nproc)` | CLI chatbot |
| GUI chatbot | `make chatbot_gui -j$(nproc)` | Qt GUI (requires Qt5) |
| API server | `make chatbot_api_server -j$(nproc)` | HTTP API server |
| Vocab builder | `make vocab_builder -j$(nproc)` | BPE vocabulary tool |
| Tests | `make all_tests -j$(nproc)` | Unit tests |

### CMake Options

| Option | Default | Description |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Debug` | `Release` for production (enables `-O3`) |
| `BUILD_GUI` | `ON` | Build Qt5 GUI |
| `BUILD_API_SERVER` | `ON` | Build HTTP API server |
| `BUILD_TESTING` | `ON` | Build unit tests |
| `BUILD_EXAMPLES` | `ON` | Build example programs |

```bash
# Production build, no GUI, no tests
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_GUI=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF
make chatbot_api_server -j$(nproc)
```

### Build Artifacts

```text
build/bin/incremental_trainer
build/bin/dataset_manager
build/bin/metrics_api_server
build/bin/registry_server
build/bin/vocab_builder
build/bin/chatbot
build/src/chatbot_gui
build/src/chatbot_gui_binary
build/src/chatbot_api_server
```

### Rebuild After Code Changes

```bash
cd build
cmake --build . --target incremental_trainer -j$(nproc)
# or for everything:
make -j$(nproc)
```

> **CRITICAL:** Always recompile after modifying C++ source files. Running the old binary after source changes produces undefined behavior.

---

## 4. Vocabulary Creation

The `vocab_builder` creates a BPE (Byte Pair Encoding) tokenizer vocabulary from a text corpus. The vocabulary must be created before training.

### Basic Usage

```bash
./build/bin/vocab_builder \
    --input training_data.txt \
    --output vocab.txt \
    --vocab-size 10000
```

### Full Option Reference

| Option | Default | Description |
| --- | --- | --- |
| `--input FILE` | required | Input text corpus |
| `--output FILE` | `vocab.txt` | Output vocabulary file |
| `--vocab-size N` | `10000` | Target vocabulary size |
| `--min-frequency N` | `2` | Minimum token frequency to include |
| `--num-merges N` | `vocab-size` | Number of BPE merge operations |
| `--special-tokens` | `<pad>,<unk>,<bos>,<eos>` | Special tokens to always include |

### Vocabulary Size Guidelines

| Dataset Size | Recommended Vocab Size |
| --- | --- |
| < 1 MB (small) | 5,000 – 8,000 |
| 1–100 MB (medium) | 8,000 – 16,000 |
| > 100 MB (large) | 16,000 – 32,000 |

### Special Token IDs (Fixed)

| Token | ID | Purpose |
| --- | --- | --- |
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown / out-of-vocabulary |
| `<bos>` | 2 | Beginning of sequence |
| `<eos>` | 3 | End of sequence |

> These IDs are hardcoded in `BPETokenizer.cpp` and must match when training and running inference.

### Multiple Corpora

```bash
cat corpus1.txt corpus2.txt corpus3.txt > combined_corpus.txt
./build/bin/vocab_builder --input combined_corpus.txt --output vocab.txt --vocab-size 10000
```

### Verifying the Vocabulary

```bash
# Count tokens
grep -c . vocab.txt

# Verify special tokens exist
grep "^<.*>" vocab.txt

# Check for duplicates (should be 0)
sort vocab.txt | uniq -d | wc -l
```

---

## 5. Model Training

All training runs through a single binary — `incremental_trainer` — using a subcommand interface. Model architecture, learning rate, and other hyperparameters are read from `config.conf` (or environment variables); there are no per-run CLI flags for these.

Every training command (`train`, `retrain`, `resume`) automatically forks into the background. The launcher prints the PID and log path, then exits. Training output goes to the log file only.

### Preparing Training Data

Training data is a plain text file of alternating `INPUT:` / `RESPONSE:` pairs:

```text
INPUT: Hello
RESPONSE: Hi! How can I help you?
INPUT: What is your name?
RESPONSE: I am the ADAI chatbot assistant.
```

Use `dataset_manager` to queue data files (see [Section 5.2](#52-dataset-management)).

### 5.1 Incremental Training

`incremental_trainer` exposes the following subcommands:

```bash
./build/bin/incremental_trainer [--config <path>] [--gpu-strategy background|full] <command>
```

| Command | What it does |
| --- | --- |
| `init [vocab] [model]` | Initialize trainer — creates session dir, validates config |
| `train [epochs]` | Train on pending queue; marks files trained on success (background) |
| `retrain [epochs]` | Reset model from config, retrain on all data (background) |
| `resume` | Continue last unfinished session without touching the data registry (background) |
| `reset [--yes] [--keep-data]` | Delete checkpoints, rebuild model from config |
| `status` | Print session summary and pending-file list |
| `history` | Print full session history and data registry |

Config is auto-discovered: `--config` flag → `./config.conf` → `/etc/adai/config.conf`.

#### Typical New-Model Workflow

```bash
# 1. Queue training data (see Section 5.2)
./build/bin/dataset_manager add training_data.txt

# 2. Initialize
./build/bin/incremental_trainer init

# 3. Train (runs in background; follow the log)
./build/bin/incremental_trainer train 25
tail -f chatbot_server.log
```

#### Adding More Data Later

```bash
# Queue new data
./build/bin/dataset_manager add additional_data.txt

# Train only on new (pending) files
./build/bin/incremental_trainer train 10
```

#### Full Retrain from Scratch

```bash
# Reset model weights, retrain on everything in the registry
./build/bin/incremental_trainer retrain 25
```

#### Resume an Interrupted Session

```bash
# Continue the last session that did not finish cleanly
./build/bin/incremental_trainer resume
```

#### Reset to a Fresh Model

```bash
# Wipe checkpoints and rebuild model architecture from config
./build/bin/incremental_trainer reset

# Preserve data registry so existing files are re-queued as pending
./build/bin/incremental_trainer reset --keep-data --yes
```

#### After Training

The trainer saves per-epoch checkpoints and promotes the best epoch (lowest validation loss) to the active model symlinks automatically:

```bash
# Verify model files
ls -lh chatbot_model.bin*
# Expected: base file + .config .encoder .decoder .lm_head .vocab symlinks
```

#### Monitoring Training

Training output goes to the log file (default `chatbot_server.log`; set `LOG_FILE_PATH` in `config.conf`):

```bash
tail -f chatbot_server.log
```

Healthy dynamics:

| Phase | Expected Loss | Expected Perplexity |
| --- | --- | --- |
| Epochs 1–3 | 4–8 → 4–5 | 1000+ → 50–150 |
| Epochs 4–10 | 2–3 | 10–20 |
| Epochs 11–25 | 1–2 | 3–7 |

For a live metrics dashboard see [Section 5.3](#53-training-metrics-service).

---

### 5.2 Dataset Management

`dataset_manager` maintains the dataset queue (DatasetRegistry) independently of the trainer. It can add local files, download Gutenberg books, or pull HuggingFace datasets — all queued as pending files that `incremental_trainer train` consumes.

```bash
./build/bin/dataset_manager [--config <path>] <command>
```

| Command | Description |
| --- | --- |
| `add <data_file>` | Add a local training file to the pending queue |
| `gutenberg <id> [pairs]` | Download & queue a Gutenberg book (default: 500 pairs) |
| `gutenberg-batch <id1,id2,...> [pairs]` | Batch-download multiple Gutenberg books |
| `huggingface <id> [pairs] [split] [in_field] [out_field]` | Download a HuggingFace dataset (default: 500 pairs, `train` split) |
| `status` | Show pending/trained file counts and registry |
| `list-pending` | List all pending files |
| `list-trained` | List all trained files |
| `clear-pending` | Remove all files from the pending queue |

#### Local File

```bash
./build/bin/dataset_manager add training_data.txt
```

#### Gutenberg Books

```bash
# Single book
./build/bin/dataset_manager gutenberg 1342 500      # Pride and Prejudice, 500 pairs

# Batch download
./build/bin/dataset_manager gutenberg-batch 1342,11,84,1661 300
```

Popular book IDs:

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

#### HuggingFace Datasets

```bash
# Daily conversation pairs (auto-detects dialog format)
./build/bin/dataset_manager huggingface daily_dialog 500

# Instruction-following with explicit field mapping
./build/bin/dataset_manager huggingface tatsu-lab/alpaca 300 train instruction output

# Other popular datasets
./build/bin/dataset_manager huggingface databricks/databricks-dolly-15k 500
./build/bin/dataset_manager huggingface Open-Orca/OpenOrca 500 train question response
```

#### Complete Gutenberg Workflow Example

```bash
# Queue several books
./build/bin/dataset_manager gutenberg-batch 1342,11,84 500

# Build or verify vocabulary
./build/bin/vocab_builder --input training_sessions/pending/*.txt --output vocab.txt --vocab-size 16000

# Initialize and train
./build/bin/incremental_trainer init
./build/bin/incremental_trainer train 25
```

---

### 5.3 Training Metrics Service

Metrics are pushed from the trainer to a standalone HTTP daemon (`metrics_api_server`) via `MetricsPushClient`. Run the daemon before starting training if you want live monitoring.

#### Starting the Metrics Server

```bash
# Default: port 8081, persists to training_sessions/metrics.jsonl
./build/bin/metrics_api_server

# Custom port and persistence interval
./build/bin/metrics_api_server --port 9090 --persist-samples 50
```

| Option | Default | Description |
| --- | --- | --- |
| `--port PORT` | `8081` | Listening port |
| `--metrics-file FILE` | `training_sessions/metrics.jsonl` | JSONL persistence path |
| `--summary-file FILE` | `training_sessions/metrics_summary.json` | Summary JSON path |
| `--prometheus-file FILE` | `training_sessions/metrics.prom` | Prometheus format output |
| `--persist-samples N` | `100` | Persist every N samples |
| `--persist-seconds N` | `30` | Persist every N seconds |
| `--max-live-sessions N` | `16` | Max concurrent tracked sessions |
| `--completed-ttl-seconds N` | `3600` | Completed session TTL |
| `--no-persistence` | — | Disable disk writes |
| `--enable-prometheus` | — | Enable Prometheus output |
| `--no-control` | — | Disable flush/clear control endpoints |

#### REST Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/metrics/current` | Current training snapshot |
| `GET` | `/api/metrics/summary` | Aggregated metrics summary |
| `GET` | `/api/metrics/history` | Historical records |
| `GET` | `/api/sessions` | List tracked sessions |
| `GET` | `/api/metrics/aggregate` | Aggregate across live sessions |
| `GET` | `/api/sessions/{key}/...` | Session-scoped endpoints |

#### Trainer-Side Config

The trainer pushes metrics automatically when `ENABLE_METRICS_SERVICE=true` (the default). Point it at a non-default server with `METRICS_SERVER_URL`. See the full key list in [Section 7](#7-configuration-reference).

```bash
# Example: check current metrics while training is running
curl http://localhost:8081/api/metrics/current
curl http://localhost:8081/api/sessions
```

#### Distributed Training (Registry Server)

For multi-node setups, `registry_server` coordinates the dataset queue across multiple trainer instances:

```bash
./build/bin/registry_server --port 8082 --data-dir registry_sessions
```

Set `REGISTRY_SERVER_URL=http://<host>:8082` in each node's `config.conf`. Single-node deployments do not need this.

---

### 5.4 Training Hyperparameter Reference

All hyperparameters are set in `config.conf` (or environment variables) and read at trainer startup. There are no per-run CLI flags.

#### Key Training Config Keys

| Key | Default | Description |
| --- | --- | --- |
| `NUM_EPOCHS` | `10` | Number of training epochs |
| `LEARNING_RATE` | `0.001` | Learning rate |
| `BATCH_SIZE` | `32` | Batch size |
| `WEIGHT_DECAY` | `0.01` | L2 weight decay regularization |
| `GRADIENT_CLIP` | `1.0` | Gradient clipping norm (0 = disabled) |
| `SESSION_DIR` | `training_sessions` | Session and checkpoint directory |
| `GPU_STRATEGY` | `background` | `background` (low-priority) or `full` (max throughput) |

#### Recommended Settings by Dataset Size

| Dataset Size | `NUM_EPOCHS` | `LR` | Notes |
| --- | --- | --- | --- |
| < 1,000 pairs | 25–50 | 0.0003 | Risk of overfitting after 30 epochs |
| 1,000–10,000 pairs | 20–30 | 0.0003 | Standard configuration |
| > 10,000 pairs | 15–20 | 0.0005 | Larger batches if RAM allows |

#### Learning Rate Schedule

Default: `WARMUP_COSINE` — ramps up over the first 10% of steps, then decays via cosine annealing. This is applied automatically; no config key required.

#### Gradient Clipping

Default `GRADIENT_CLIP=1.0` prevents gradient explosion. Reduce to `0.5` if you see persistent NaN/Inf warnings in the log.

#### Generation Strategies (Inference)

| Strategy | Config (`STRATEGY=`) | Use Case |
| --- | --- | --- |
| Greedy | `greedy` | Fastest; deterministic |
| Beam Search | `beam` | Best quality; slow |
| Top-K Sampling | `top-k` | Creative; temperature-sensitive |
| Nucleus (Top-P) | `nucleus` | Best balance of quality/variety |

---

## 6. Running the Chatbot

### 6.1 CLI Chatbot

```bash
# Default paths (vocab.txt, chatbot_model.bin)
./build/src/chatbot

# Explicit paths
./build/src/chatbot vocab.txt chatbot_model.bin

# With configuration
./build/src/chatbot vocab.txt chatbot_model.bin \
    --strategy nucleus \
    --temperature 0.8 \
    --max-length 150
```

#### CLI Commands

| Command | Description |
| --- | --- |
| `/help` | Show available commands |
| `/quit` or `/exit` | Exit chatbot |
| `/clear` | Clear conversation history |
| `/stats` | Show context statistics |
| `/save` | Save conversation history |
| `/strategy greedy\|beam\|top-k\|nucleus` | Change generation strategy |
| `/temperature FLOAT` | Set temperature (0.1–2.0) |
| `/max-length N` | Set maximum response length |
| `/beam-width N` | Set beam width (beam search only) |
| `/top-k N` | Set top-k value |
| `/top-p FLOAT` | Set nucleus threshold |

#### Generation Configuration

```bash
./build/src/chatbot vocab.txt chatbot_model.bin \
    --strategy nucleus \
    --temperature 0.8 \
    --top-p 0.9 \
    --max-length 200 \
    --beam-width 5
```

#### Context Management

The conversation context window is capped at **480 tokens** and **20 messages**. Oldest messages are evicted to stay within limits. Use `/clear` to reset conversation history manually.

### 6.2 Qt GUI Chatbot

#### Build

```bash
cd build
cmake .. -DBUILD_GUI=ON
make chatbot_gui -j$(nproc)
```

#### Launch

```bash
# Always use the C++ wrapper (handles library conflicts automatically)
./build/src/chatbot_gui

# With explicit paths
./build/src/chatbot_gui vocab.txt chatbot_model.bin
```

> **Note:** The `chatbot_gui` binary is a C++ wrapper that fixes snap/library conflicts by unsetting problematic environment variables before launching the real GUI (`chatbot_gui_binary`). Run it directly — do not run `chatbot_gui_binary` directly.

#### GUI Features

- **Settings panel** — adjust all generation parameters at runtime
- **Strategy selector** — greedy, beam, top-K, nucleus
- **Temperature slider** — 0.1 to 2.0
- **Max length control** — response token limit
- **Clear button** — reset conversation history
- **Send button** — submit message (also Enter key)

#### System Requirements for GUI

- Qt5 installed (`qt5-default`, `libqt5widgets5`, `libqt5gui5`)
- Graphical environment (X11/Wayland) or X11 forwarding (`ssh -X`)

### 6.3 API Server

The HTTP REST API server provides single-turn and session-based chat endpoints.

#### Start the API Server

```bash
# With config file
./build/src/chatbot_api_server -c config.conf

# With environment variables
export VOCAB_PATH=vocab.txt
export PORT=8080
./build/src/chatbot_api_server

# CLI override
./build/src/chatbot_api_server -c config.conf --port 9000
```

#### Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Single-turn chat |
| `POST` | `/chat/session` | Session-based chat |
| `POST` | `/clear-session` | Clear session history |
| `GET` | `/metrics` | Training metrics |

#### Example API Calls

```bash
# Health check
curl http://localhost:8080/health

# Single-turn chat
curl -X POST http://localhost:8080/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello!"}'

# Session chat (maintains context across requests)
curl -X POST http://localhost:8080/chat/session \
     -H "Content-Type: application/json" \
     -d '{"message": "Tell me about AI", "session_id": "user123"}'

# Clear a session
curl -X POST http://localhost:8080/clear-session \
     -H "Content-Type: application/json" \
     -d '{"session_id": "user123"}'
```

#### Graceful Shutdown

Send `SIGTERM` or `SIGINT` to the process. The server completes in-flight requests and shuts down cleanly.

```bash
kill -TERM $(pgrep chatbot_api)
```

#### Configuration Hot-Reload

Send `SIGHUP` to reload `config.conf` without restarting:

```bash
kill -HUP $(pgrep chatbot_api)
```

### 6.4 Model Service Manager

The `model_service.sh` script provides a convenient service wrapper for the API server.

```bash
# Start the service
./scripts/model_service.sh start

# Stop the service
./scripts/model_service.sh stop

# Restart
./scripts/model_service.sh restart

# Check status
./scripts/model_service.sh status

# View logs
./scripts/model_service.sh logs

# Build (triggers cmake build)
./scripts/model_service.sh build

# Health check
./scripts/model_service.sh health
```

---

## 7. Configuration Reference

All settings can be provided via:

1. **Environment variables** (highest priority)
2. **Configuration file** (`config.conf`)
3. **Built-in defaults** (lowest priority)

### Server Settings

| Key | Default | Description |
| --- | --- | --- |
| `VOCAB_PATH` | `vocab.txt` | Path to vocabulary file |
| `MODEL_PATH` | — | Path to model file (optional; starts untrained if not set) |
| `PORT` | `8080` | HTTP port to listen on |
| `SESSION_TIMEOUT` | `30` | Minutes before session expires |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARN`, `ERROR` |

### Model Architecture

| Key | Default | Description |
| --- | --- | --- |
| `D_MODEL` | `512` | Hidden dimension size |
| `NUM_HEADS` | `8` | Number of attention heads |
| `D_FF` | `2048` | Feed-forward layer dimension |
| `NUM_ENCODER_LAYERS` | `6` | Encoder transformer depth |
| `NUM_DECODER_LAYERS` | `6` | Decoder transformer depth |
| `MAX_SEQ_LENGTH` | `1024` | Maximum sequence length (positional encoding limit: 512) |

### Text Generation

| Key | Default | Description |
| --- | --- | --- |
| `MAX_LENGTH` | `100` | Max tokens in generated response |
| `TEMPERATURE` | `1.0` | Sampling temperature (lower = more deterministic) |
| `TOP_P` | `0.9` | Nucleus sampling threshold |
| `TOP_K` | `50` | Top-K sampling value |
| `BEAM_WIDTH` | `4` | Beam search beam count |
| `STRATEGY` | `nucleus` | `greedy`, `beam`, `top-k`, `nucleus` |

### Training

| Key | Default | Description |
| --- | --- | --- |
| `NUM_EPOCHS` | `10` | Epochs per `train` / `retrain` run |
| `LEARNING_RATE` | `0.001` | Learning rate |
| `BATCH_SIZE` | `32` | Batch size |
| `WEIGHT_DECAY` | `0.01` | L2 weight decay regularization |
| `GRADIENT_CLIP` | `1.0` | Gradient clipping norm (0 = disabled) |
| `SESSION_DIR` | `training_sessions` | Checkpoint and session file directory |
| `GPU_STRATEGY` | `background` | `background` (low-priority) or `full` (max throughput) |
| `ENABLE_GENERATION_QUALITY_METRICS` | `false` | Compute BLEU/ROUGE during validation |
| `GENERATION_QUALITY_SAMPLE_SIZE` | `10` | Validation pairs sampled for scoring per epoch |
| `GENERATION_QUALITY_MAX_TOKENS` | `50` | Max tokens per scoring call |
| `GENERATION_QUALITY_ASYNC_THRESHOLD` | `50` | Sample size at which scoring runs in a background thread |

### Distributed Registry

| Key | Default | Description |
| --- | --- | --- |
| `REGISTRY_SERVER_URL` | — | URL of `registry_server`; empty = local-only |
| `RUN_GROUP` | — | Training pool group name (partitions work across nodes) |
| `RUN_ID` | auto | Node identifier (auto: hostname+PID tail) |
| `REGISTRY_TIMEOUT_MS` | `5000` | HTTP timeout for registry operations (ms) |

### GPU / CUDA

| Key | Default | Description |
| --- | --- | --- |
| `GPU_ENABLED` | `false` | Enable GPU acceleration (requires `-DENABLE_GPU=ON` build) |
| `GPU_DEVICE_ID` | `0` | CUDA device index |
| `GPU_MEMORY_FRACTION` | `0.5` | Fraction of GPU memory ADAI may allocate (0.0–1.0) |
| `GPU_STRATEGY` | `background` | `background` (yields to other GPU work) or `full` (high-priority stream) |

### Training Metrics

| Key | Default | Description |
| --- | --- | --- |
| `ENABLE_METRICS_SERVICE` | `true` | Push metrics from trainer to metrics daemon |
| `METRICS_SERVER_URL` | `http://localhost:8081` | URL of `metrics_api_server`; empty disables push |
| `METRICS_PUSH_TIMEOUT_MS` | `1000` | HTTP timeout for metric pushes (ms) |
| `METRICS_ENABLE_PERSISTENCE` | `true` | Persist metrics to disk |
| `METRICS_FILE` | `training_sessions/metrics.jsonl` | JSONL metrics log |
| `METRICS_SUMMARY_FILE` | `training_sessions/metrics_summary.json` | Summary JSON |
| `METRICS_PERSIST_EVERY_SAMPLES` | `100` | Flush to disk every N samples |
| `METRICS_PERSIST_EVERY_SECONDS` | `30` | Flush to disk every N seconds |
| `METRICS_MAX_RECORDS_IN_MEMORY` | `10000` | Cap on in-memory metric records |
| `METRICS_MAX_RECORDS_ON_DISK` | `100000` | Cap on persisted metric records |
| `METRICS_ENABLE_PROMETHEUS` | `false` | Write Prometheus format output |
| `METRICS_PROMETHEUS_FILE` | `training_sessions/metrics.prom` | Prometheus output path |
| `METRICS_MAX_LIVE_SESSIONS` | `16` | Max concurrent tracked sessions in daemon |
| `METRICS_COMPLETED_TTL_SECONDS` | `3600` | How long completed sessions are retained |
| `METRICS_SWEEP_INTERVAL_SECONDS` | `60` | Session GC sweep interval |
| `METRICS_API_PORT` | `8081` | Port the metrics daemon listens on |

### Log File Rotation

| Key | Default | Description |
| --- | --- | --- |
| `LOG_FILE_PATH` | — | Path to log file; empty = console only |
| `LOG_MAX_SIZE_MB` | `10` | Max size per log file in MB (1–1024) |
| `LOG_MAX_FILES` | `5` | Number of rotated files to keep (1–100) |
| `LOG_COMPRESS` | `false` | Flag for future compression integration |

**Rotated file naming:** `chatbot.log`, `chatbot.log.1`, `chatbot.log.2`, …

### Configuration Priority

```text
CLI flags  >  Environment variables  >  config.conf  >  Built-in defaults
```

### Example `config.conf` (Production)

```ini
# ============================================================================
# Server
# ============================================================================
VOCAB_PATH=/opt/adai/vocab/vocab.txt
MODEL_PATH=/opt/adai/models/chatbot_model.bin
PORT=8080
SESSION_TIMEOUT=30
LOG_LEVEL=INFO

# ============================================================================
# Model Architecture
# Must match the architecture used during training.
# ============================================================================
D_MODEL=512
NUM_HEADS=8
D_FF=2048
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
MAX_SEQ_LENGTH=1024

# ============================================================================
# Training Hyperparameters
# Used by incremental_trainer; ignored by the inference server.
# ============================================================================
LEARNING_RATE=0.0003
NUM_EPOCHS=25
BATCH_SIZE=32
WEIGHT_DECAY=0.01
GRADIENT_CLIP=1.0
SESSION_DIR=training_sessions

# ============================================================================
# Text Generation
# ============================================================================
MAX_GEN_LENGTH=150
TEMPERATURE=0.8
TOP_P=0.9
TOP_K=50
BEAM_WIDTH=4
STRATEGY=nucleus

# ============================================================================
# Generation Quality Metrics
# Opt-in: computes BLEU/ROUGE during validation by calling generate_response()
# on a sample of the validation set. Adds overhead proportional to sample size.
# ============================================================================
ENABLE_GENERATION_QUALITY_METRICS=false
GENERATION_QUALITY_SAMPLE_SIZE=10
GENERATION_QUALITY_MAX_TOKENS=50
# Async threshold: scoring runs in background thread when sample size >= this value.
# Memory overhead: one full model weight copy (~50-150 MB).
GENERATION_QUALITY_ASYNC_THRESHOLD=50

# ============================================================================
# Training Metrics Service
# The trainer pushes metrics to metrics_api_server via MetricsPushClient.
# Start the daemon before training: ./build/bin/metrics_api_server
# ============================================================================
ENABLE_METRICS_SERVICE=true
METRICS_SERVER_URL=http://localhost:8081
METRICS_PUSH_TIMEOUT_MS=1000
METRICS_ENABLE_PERSISTENCE=true
METRICS_FILE=training_sessions/metrics.jsonl
METRICS_SUMMARY_FILE=training_sessions/metrics_summary.json
METRICS_PERSIST_EVERY_SAMPLES=100
METRICS_PERSIST_EVERY_SECONDS=30
METRICS_MAX_RECORDS_IN_MEMORY=10000
METRICS_MAX_RECORDS_ON_DISK=100000
METRICS_ENABLE_PROMETHEUS=false
METRICS_PROMETHEUS_FILE=training_sessions/metrics.prom

# ============================================================================
# Metrics API Daemon (metrics_api_server)
# ============================================================================
METRICS_API_PORT=8081
METRICS_API_ALLOW_CONTROL=true
METRICS_MAX_LIVE_SESSIONS=16
METRICS_COMPLETED_TTL_SECONDS=3600
METRICS_SWEEP_INTERVAL_SECONDS=60
# Seconds without a metrics ingest before a live session is marked stale.
METRICS_STALENESS_THRESHOLD_SECONDS=60

# ============================================================================
# GPU / CUDA
# Requires build with -DENABLE_GPU=ON.
# ============================================================================
GPU_ENABLED=false
GPU_DEVICE_ID=0
GPU_MEMORY_FRACTION=0.5
# GPU_STRATEGY=background   # background (default) or full

# ============================================================================
# Logging
# ============================================================================
LOG_FILE_PATH=/var/log/adai/chatbot.log
LOG_MAX_SIZE_MB=50
LOG_MAX_FILES=10

# ============================================================================
# RAG (Retrieval-Augmented Generation)
# ============================================================================
RAG_ENABLED=false
# RAG_DOCS_PATH=/opt/adai/knowledge
RAG_NUM_DOCS=3
RAG_THRESHOLD=0.0
RAG_MAX_CONTEXT_LENGTH=512

# ============================================================================
# Distributed Registry (multi-node training)
# Leave commented out for standalone single-node operation.
# ============================================================================
# REGISTRY_SERVER_URL=http://<coordinator>:8082
# RUN_GROUP=my-training-group
# RUN_ID=
# REGISTRY_TIMEOUT_MS=5000
```

---

## 8. RAG Configuration & Activation

Retrieval-Augmented Generation (RAG) allows the API server to retrieve relevant documents from a knowledge base and prepend them to the model's context before generating a response. RAG is wired into the `ChatbotAPI` layer and activated via configuration.

### How It Works

1. On startup, the server reads all `*.txt` files from `RAG_DOCS_PATH`
2. Each document is encoded using `LLMEncoder` and stored in `DocumentStore`
3. On each `/chat` request, `RAGInference` retrieves the top-N most similar documents (cosine similarity)
4. Retrieved document text is prepended to the user query before passing to the model
5. If RAG initialization fails, the server logs a warning and continues without RAG

### RAG Configuration Keys

| Key | Default | Description |
| --- | --- | --- |
| `RAG_ENABLED` | `false` | Enable/disable RAG |
| `RAG_DOCS_PATH` | — | Directory containing knowledge base `.txt` files |
| `RAG_NUM_DOCS` | `3` | Number of documents to retrieve per query |
| `RAG_THRESHOLD` | `0.0` | Minimum cosine similarity to include a document (0.0 = all retrieved docs) |
| `RAG_MAX_CONTEXT_LENGTH` | `512` | Max tokens of retrieved context to prepend |

### Activating RAG

**Step 1:** Create a knowledge base directory and populate with `.txt` files:

```bash
mkdir -p /opt/adai/knowledge
# Add knowledge base documents
cp my_faq.txt /opt/adai/knowledge/
cp product_docs.txt /opt/adai/knowledge/
cp policies.txt /opt/adai/knowledge/
```

**Step 2:** Update `config.conf`:

```ini
RAG_ENABLED=true
RAG_DOCS_PATH=/opt/adai/knowledge
RAG_NUM_DOCS=3
RAG_THRESHOLD=0.1
RAG_MAX_CONTEXT_LENGTH=512
```

**Step 3:** Restart the server:

```bash
# systemd
sudo systemctl restart adai

# Docker
docker-compose restart chatbot-api

# Direct
./build/src/chatbot_api_server -c config.conf
```

**Step 4:** Verify in startup logs:

```text
[info] RAG enabled - loading documents from /opt/adai/knowledge
[info] RAG: loaded 3 documents into DocumentStore
[info] RAG engine initialized and attached to ChatbotAPI
```

### RAG Fallback Behavior

If any part of RAG initialization fails (missing directory, empty directory, encoder error), the server logs a warning and starts without RAG. The `/chat` endpoint remains fully operational.

```text
[warn] RAG initialization failed: [reason]. Continuing without RAG.
```

### RAG with Docker

Mount the knowledge base directory as a volume:

```yaml
volumes:
  - /opt/adai/knowledge:/home/adai/knowledge:ro
```

Set environment variable:

```yaml
environment:
  - RAG_ENABLED=true
  - RAG_DOCS_PATH=/home/adai/knowledge
```

---

## 9. Deployment

### 9.1 Docker Deployment

#### Quick Start

```bash
# Build Docker image
./scripts/docker_build.sh

# Deploy with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

#### Configuration via Environment Variables

Environment variables can be passed to the container and take priority over `config.conf`:

```bash
docker run -d \
  --name adai-chatbot-api \
  -p 8080:8080 \
  -e VOCAB_PATH=/home/adai/vocab.txt \
  -e PORT=8080 \
  -e LOG_LEVEL=INFO \
  -e TEMPERATURE=0.8 \
  -e STRATEGY=nucleus \
  -v $(pwd)/vocab.txt:/home/adai/vocab.txt:ro \
  -v $(pwd)/models:/home/adai/models:ro \
  adai-chatbot:latest
```

#### Docker Compose (Production with Nginx)

```yaml
version: '3.8'
services:
  chatbot-api:
    image: adai-chatbot:latest
    environment:
      - VOCAB_PATH=/home/adai/vocab.txt
      - PORT=8080
      - LOG_LEVEL=INFO
      - RAG_ENABLED=false
    volumes:
      - ./vocab.txt:/home/adai/vocab.txt:ro
      - ./models:/home/adai/models:ro
      - ./logs:/var/log/adai
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - chatbot-api
    restart: unless-stopped
```

#### Container Management

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart a service
docker-compose restart chatbot-api

# View logs
docker-compose logs -f chatbot-api

# Execute shell in container
docker-compose exec chatbot-api /bin/bash

# View resource usage
docker stats adai-chatbot-api

# Health check
docker inspect --format='{{.State.Health.Status}}' adai-chatbot-api
curl http://localhost:8080/health
```

#### Volume Mounts

| Host Path | Container Path | Mode | Purpose |
| --- | --- | --- | --- |
| `./vocab.txt` | `/home/adai/vocab.txt` | ro | Vocabulary file |
| `./models/` | `/home/adai/models/` | ro | Model weights |
| `./logs/` | `/var/log/adai/` | rw | Log files |
| `./knowledge/` | `/home/adai/knowledge/` | ro | RAG documents |

#### Security Defaults (Container)

- Runs as non-root user (`adai`)
- Multi-stage build (minimal runtime image)
- Read-only model and vocabulary volumes
- Rate limiting via Nginx

#### Updating the Application

```bash
git pull origin main
./scripts/docker_build.sh -t v1.1.0
docker-compose down
docker-compose up -d
```

### 9.2 systemd Service Deployment

#### Server Bundle (Metrics, MNS, Registry)

For deploying the training infrastructure services, see [Server Bundle Deployment](deployment/SERVER_BUNDLE_DEPLOYMENT.md).

```bash
# Build, package, and install with PostgreSQL
cmake --preset portable && cmake --build --preset portable
./scripts/package_server_bundle.sh
sudo ./scripts/install_server_bundle.sh --build-dir build/portable --setup-postgres --yes

# Verify
systemctl status adai-mns adai-registry adai-metrics
curl http://localhost:8081/health   # metrics
curl http://localhost:8083/health   # MNS
```

#### Chatbot API Server

Automated installation for the chatbot API server:

```bash
# 1. Build
cd /home/rodney/Repos/adai
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_API_SERVER=ON
make chatbot_api_server -j$(nproc)
cd ..

# 2. Install service (creates adai user, /opt/adai/, /etc/adai/, /var/log/adai/)
sudo ./scripts/install_systemd_service.sh

# 3. Verify
systemctl status adai
curl http://localhost:8080/health
```

#### Installed Paths

| Path | Contents |
| --- | --- |
| `/opt/adai/bin/chatbot_api_server` | Executable |
| `/opt/adai/vocab/vocab.txt` | Vocabulary |
| `/opt/adai/models/` | Model files |
| `/etc/adai/config.conf` | Configuration |
| `/var/log/adai/` | Log files |
| `/etc/systemd/system/adai.service` | Service unit |

#### Installation Script Options

```bash
sudo ./scripts/install_systemd_service.sh \
  --install-path /usr/local/adai \
  --port 9000 \
  --user mychatbot \
  --group mychatbot \
  --log-level INFO
```

#### Manual Installation Steps

```bash
# Create system user
sudo useradd -r -s /bin/false -d /opt/adai -c "ADAI Chatbot Service" adai

# Create directories
sudo mkdir -p /opt/adai/{bin,vocab,models} /var/log/adai /etc/adai
sudo chown -R adai:adai /opt/adai /var/log/adai

# Copy files
sudo cp build/src/chatbot_api_server /opt/adai/bin/
sudo cp vocab.txt /opt/adai/vocab/
sudo cp scripts/adai.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable adai.service
sudo systemctl start adai.service
```

#### Service Management

```bash
sudo systemctl start adai        # Start
sudo systemctl stop adai         # Stop (graceful)
sudo systemctl restart adai      # Restart
sudo systemctl reload adai       # Reload config (SIGHUP)
sudo systemctl enable adai       # Start on boot
sudo systemctl disable adai      # Don't start on boot
systemctl status adai            # Status
systemctl is-active adai         # Quick active check
```

#### Viewing systemd Logs

```bash
# Follow live logs
journalctl -u adai -f

# Last 50 lines
journalctl -u adai -n 50

# Since last boot
journalctl -u adai -b

# Errors only
journalctl -u adai -p err

# Time range
journalctl -u adai --since "2026-01-01 10:00:00" --until "2026-01-01 11:00:00"

# Since 1 hour ago
journalctl -u adai --since "1 hour ago"
```

#### Configuration via Environment Overrides

```bash
# Add environment overrides without editing the service file
sudo systemctl edit adai
```

In the drop-in editor:

```ini
[Service]
Environment="LOG_LEVEL=DEBUG"
Environment="TEMPERATURE=0.7"
Environment="RAG_ENABLED=true"
Environment="RAG_DOCS_PATH=/opt/adai/knowledge"
```

After saving:

```bash
sudo systemctl daemon-reload
sudo systemctl restart adai
```

#### Security Hardening (systemd Unit)

The provided unit file includes:

- `ProtectSystem=strict` — system directories read-only
- `ProtectHome=yes` — home directories inaccessible
- `PrivateTmp=yes` — isolated `/tmp`
- `NoNewPrivileges=true` — no privilege escalation
- `RestrictSUIDSGID=yes` — no setuid/setgid
- `SystemCallFilter` — restricted syscall set
- `MemoryMax=4G` / `CPUQuota=50%` — resource limits

#### Resource Limit Adjustments

```bash
sudo systemctl edit adai
```

```ini
[Service]
MemoryMax=8G
CPUQuota=100%
LimitNOFILE=131072
```

---

## 10. Monitoring & Log Management

### Health Check

```bash
curl http://localhost:8080/health
# Returns: {"status": "ok", "uptime": 3600}
```

### Log File Rotation

Log rotation is handled by `spdlog`'s rotating file sink. Configure via `config.conf`:

```ini
LOG_FILE_PATH=/var/log/adai/chatbot.log
LOG_MAX_SIZE_MB=50
LOG_MAX_FILES=10
```

When `chatbot.log` reaches the size limit, it is renamed to `chatbot.log.1`, and older files are incremented. Files beyond `LOG_MAX_FILES` are deleted automatically.

#### Monitoring Log Rotation

```bash
# Watch log directory
watch -n 5 'ls -lh /var/log/adai/'

# Tail current log
tail -f /var/log/adai/chatbot.log

# View rotated files
ls -lh /var/log/adai/chatbot.log*
```

#### Using logrotate for Compression

For production deployments needing compression, use `logrotate` alongside spdlog rotation:

```text
# /etc/logrotate.d/adai
/var/log/adai/chatbot.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    missingok
    copytruncate
}
```

When using `logrotate`, disable spdlog rotation (`LOG_MAX_FILES=1`, very large `LOG_MAX_SIZE_MB`) to avoid conflicts.

### Configuration Hot-Reload

```bash
# 1. Edit config
sudo vim /etc/adai/config.conf

# 2. Reload without restart
kill -HUP $(pgrep chatbot_api)
# or via systemd:
sudo systemctl reload adai

# 3. Confirm reload in logs
tail /var/log/adai/chatbot.log | grep -i reload
```

> **Note:** `LOG_FILE_PATH` changes take effect only on restart. All other settings reload live.

### Resource Monitoring

```bash
# Container resource usage
docker stats adai-chatbot-api

# systemd resource usage
systemctl status adai
systemd-cgtop

# Memory usage
systemctl show adai -p MemoryCurrent
```

---

## 11. Windows Cross-Compilation

Cross-compile a Windows x64 binary from Linux using MinGW-w64.

### Prerequisites

```bash
sudo apt-get install mingw-w64
```

### Build

```bash
mkdir -p build-windows && cd build-windows
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/mingw-w64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_GUI=OFF \
  -DBUILD_TESTING=OFF
make chatbot_api_server -j$(nproc)
```

### Output

```text
build-windows/src/chatbot_api_server.exe
```

### Distribution Package

```bash
# Copy executable and required files
mkdir -p dist-windows/adai-chatbot-windows-x64
cp build-windows/src/chatbot_api_server.exe dist-windows/adai-chatbot-windows-x64/
cp vocab.txt dist-windows/adai-chatbot-windows-x64/
cp config.conf dist-windows/adai-chatbot-windows-x64/
# Package required MinGW DLLs if distributing standalone
```

See `dist-windows/` for a pre-built example distribution package.

---

## 12. Troubleshooting

### 12.1 Build Issues

#### CMake can't find Qt5

```bash
sudo apt-get install qt5-default qtbase5-dev libqt5widgets5
# or disable GUI build:
cmake .. -DBUILD_GUI=OFF
```

#### Missing spdlog

```bash
sudo apt-get install libspdlog-dev
# or build from source (see external/)
```

#### OpenMP errors

```bash
sudo apt-get install libomp-dev
```

#### Linking errors (undefined symbols)

Check that you are building after a clean configure:

```bash
cd build && cmake .. && make -j$(nproc)
```

### 12.2 Training Issues

#### Loss rises after epoch 2 / perplexity stuck above 200

**Cause:** Learning rate too high.

In `config.conf`, lower the learning rate and retrain:

```ini
LEARNING_RATE=0.0003
GRADIENT_CLIP=1.0
NUM_EPOCHS=25
```

```bash
./build/bin/incremental_trainer retrain 25
```

#### NaN/Inf gradient warnings

The trainer detects and skips NaN/Inf updates automatically. If warnings persist, reduce the learning rate in `config.conf`:

```ini
LEARNING_RATE=0.0001
```

Then retrain. If persistent, check training data for extremely long sequences or unusual characters.

#### Perplexity stuck above 100 after 25 epochs

1. Increase `NUM_EPOCHS=50` in `config.conf` and run `incremental_trainer train 50`
2. Verify data quality (diverse inputs, correct `INPUT:`/`RESPONSE:` format)
3. Check learning rate warmup is active (default: automatic 10% warmup)
4. Consider a smaller model (`D_MODEL`, `D_FF`) if dataset is tiny (< 500 pairs)

#### Incorrect loss values

If reported loss looks suspiciously consistent across epochs, verify the trainer was rebuilt after any code changes. The loss averaging bug (dividing by `num_samples` instead of `num_updates`) was fixed — ensure you are running updated code.

### 12.3 Model Loading Issues

#### "No pre-trained model found"

The model requires a specific file structure. Verify:

```bash
ls -la chatbot_model.bin*
# Expected:
# chatbot_model.bin         (0-byte base file)
# chatbot_model.bin.config  (symlink)
# chatbot_model.bin.decoder (symlink)
# chatbot_model.bin.encoder (symlink)
# chatbot_model.bin.lm_head (symlink)
# chatbot_model.bin.vocab   (symlink)
```

If files are missing, create symlinks to the best epoch manually:

```bash
touch chatbot_model.bin
for ext in config decoder lm_head vocab encoder; do
    ln -sf "chatbot_model.bin.epoch8.$ext" "chatbot_model.bin.$ext"
done
```

#### "Vocabulary size mismatch"

The model's embedded vocabulary size must match `vocab.txt`. If you rebuilt the vocabulary after training, retrain the model from scratch.

#### Interrupted training / missing component files

```bash
# Find the best epoch from available files
ls chatbot_model.bin.epoch*.config | sort

# Link to chosen epoch (e.g., epoch 8)
touch chatbot_model.bin
for ext in config decoder lm_head vocab encoder; do
    ln -sf "chatbot_model.bin.epoch8.$ext" "chatbot_model.bin.$ext"
done
```

#### API server: vocab file missing

```bash
# Check config
grep VOCAB_PATH config.conf
# Check file exists
ls -lh vocab.txt

# Systemd: check permissions
ls -ld /opt/adai/vocab/
sudo chown -R adai:adai /opt/adai/vocab/
```

### 12.4 Generation Quality Issues

#### Responses are all `<unk>` tokens

**Root cause:** Vocabulary mismatch — the model was trained with a different vocabulary than the one currently loaded.

**Solution:**

1. Delete old model files: `rm -f chatbot_model.bin*`
2. Delete old checkpoints: `rm -f *.epoch*`
3. Recompile if `BPETokenizer.cpp` was changed: `cd build && make -j$(nproc)`
4. Retrain from scratch with matching vocab and trainer

#### Empty responses (all special tokens stripped)

Same root cause as `<unk>` generation. The model generates only special tokens which are stripped by `decode(tokens, true)`. Retrain from scratch.

#### Responses are repetitive or incoherent

1. Ensure training converged (perplexity < 50 after 25 epochs)
2. Switch to nucleus sampling: `--strategy nucleus --temperature 0.8 --top-p 0.9`
3. Increase training epochs or data volume
4. Verify training data quality (no truncated responses, correct format)

#### Very slow generation

1. Use greedy strategy: `--strategy greedy`
2. Reduce `MAX_LENGTH`
3. Reduce beam width if using beam search: `--beam-width 3`
4. Ensure OpenMP is enabled (parallel attention computation)

### 12.5 GUI Issues

#### `symbol lookup error: __libc_pthread_init`

Run the C++ wrapper (always use this instead of the binary directly):

```bash
./build/src/chatbot_gui  # Wrapper handles environment automatically
```

If the wrapper is unavailable, use the shell script:

```bash
./scripts/chatbot_gui_fixed.sh
# or
./scripts/run_chatbot_gui.sh
```

Manual fix:

```bash
env -u LD_LIBRARY_PATH -u GTK_PATH \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    ./build/src/chatbot_gui_binary
```

#### `QSocketNotifier: Can only be used with threads started with QThread`

Harmless warning. The application works normally. Can be suppressed:

```bash
./build/src/chatbot_gui 2>&1 | grep -v QSocketNotifier
```

#### `Gtk-Message: Failed to load module "canberra-gtk-module"`

Harmless. Fix optionally:

```bash
sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
```

Or suppress:

```bash
export GTK_MODULES=""
./build/src/chatbot_gui
```

#### `qt.qpa.xcb: could not connect to display`

No graphical environment. Solutions:

- SSH with X11: `ssh -X user@host`
- VNC/Remote desktop
- WSL2 with WSLg (Windows 11)

```bash
# Test with virtual display
sudo apt-get install xvfb
xvfb-run ./build/src/chatbot_gui
```

#### GUI freezes during generation

The GUI generates synchronously. Expected for large models or long `max_length` settings. Reduce max length or switch to greedy strategy via the Settings panel.

#### Model fails to load in GUI

```bash
# Run from project root where vocab.txt and model files are located
cd /home/rodney/Repos/adai
./build/src/chatbot_gui
```

Verify files exist:

```bash
ls -lh vocab.txt chatbot_model.bin chatbot_model.bin.config
```

### 12.6 Special Token Issues

If the model produces unexpected behavior related to `<bos>`/`<eos>` tokens or stops prematurely:

**Verify correct token IDs:**

| Token | Correct ID |
| --- | --- |
| `<pad>` | 0 |
| `<unk>` | 1 |
| `<bos>` | 2 |
| `<eos>` | 3 |

**Known historical bug (check if applies to your build):**  
`EncoderDecoderModel::GenerationConfig` was initialized with `bos_token_id=1` and `eos_token_id=2` (swapped). Fixed by setting `bos_token_id=2`, `eos_token_id=3` in the constructor. Verify with:

```bash
grep "bos_token_id" src/EncoderDecoderModel.cpp
# Should show: gen_config.bos_token_id = 2;
```

After any such fix, **retrain the model** to ensure training/inference consistency.

### 12.7 Context Length Issues

#### Warnings: "Input sequence length (N) exceeds max_len (512)"

**Cause:** Conversation context accumulated too many tokens.

**CLI fix:** Use `/clear` command to reset context.

**GUI fix:** Click the **Clear** button.

**Underlying constraint:** Positional encoding is hardcoded at 512 tokens max, even though `MAX_SEQ_LENGTH=1024`. The conversation context is automatically capped at 480 tokens and 20 messages. The Clear button resets to 0.

**If warnings persist after clearing:**

```bash
# Check ConversationContext max_tokens in source
grep "max_tokens" src/ChatbotGUI.cpp src/ChatbotCLI.cpp
# Should be 480, not 2048
```

---

## 13. Quick Reference Card

### Build & Environment Setup

```bash
# Full build
cd /home/rodney/Repos/adai
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Rebuild specific target
cmake --build . --target incremental_trainer -j$(nproc)
```

### Vocabulary

```bash
# Create vocab from corpus
./build/bin/vocab_builder --input data.txt --output vocab.txt --vocab-size 10000
```

### Dataset & Training

```bash
# Queue a local file, then train
./build/bin/dataset_manager add data.txt
./build/bin/incremental_trainer train 25

# Download a Gutenberg book and train
./build/bin/dataset_manager gutenberg 1342 500
./build/bin/incremental_trainer train 25

# Full retrain from scratch on all queued data
./build/bin/incremental_trainer retrain 25

# Resume an interrupted session
./build/bin/incremental_trainer resume

# Check training status / session history
./build/bin/incremental_trainer status
./build/bin/incremental_trainer history

# Follow training log
tail -f chatbot_server.log

# Live metrics (requires metrics_api_server running)
./build/bin/metrics_api_server &
curl http://localhost:8081/api/metrics/current
```

### Running

```bash
# CLI chatbot
./build/src/chatbot vocab.txt chatbot_model.bin

# GUI chatbot
./build/src/chatbot_gui

# API server
./build/src/chatbot_api_server -c config.conf
```

### Docker

```bash
./scripts/docker_build.sh && docker-compose up -d   # Deploy
docker-compose logs -f                              # Logs
docker-compose down                                 # Stop
```

### systemd

```bash
sudo systemctl start adai      # Start
sudo systemctl stop adai       # Stop
sudo systemctl restart adai    # Restart
journalctl -u adai -f          # Live logs
curl localhost:8080/health     # Health check
```

### Troubleshooting One-Liners

```bash
# Check vocab size
grep -c . vocab.txt

# Check model files present
ls -la chatbot_model.bin*

# API health check
curl http://localhost:8080/health

# Check port in use
sudo lsof -i :8080

# Container log tail
docker logs --tail 50 adai-chatbot-api

# Reload config without restart
kill -HUP $(pgrep chatbot_api)

# Fix model file symlinks (after interrupted training)
touch chatbot_model.bin
for ext in config decoder lm_head vocab encoder; do
    ln -sf "chatbot_model.bin.epoch8.$ext" "chatbot_model.bin.$ext"
done
```

### File Locations

| File | Default Location |
| --- | --- |
| Vocabulary | `vocab.txt` / `/opt/adai/vocab/vocab.txt` |
| Model base | `chatbot_model.bin` |
| Config | `config.conf` / `/opt/adai/etc/config.conf` |
| API server executable | `build/bin/chatbot_api_server` |
| CLI chatbot | `build/bin/chatbot` |
| GUI chatbot | `build/bin/chatbot_gui` |
| Incremental trainer | `build/bin/incremental_trainer` |
| Dataset manager | `build/bin/dataset_manager` |
| Metrics API server | `build/bin/metrics_api_server` |
| Registry server | `build/bin/registry_server` |
| MNS server | `build/bin/mns_server` |
| MNS CLI | `build/bin/mns_cli` |
| Logs | `stdout` / `/var/log/adai/` |
| Training sessions | `training_sessions/` |
| Metrics JSONL | `training_sessions/metrics.jsonl` |
| Metrics SQLite DB | `training_sessions/metrics.db` |
| MNS SQLite DB | `<mns-data-dir>/models.db` |
| PostgreSQL schema | `scripts/setup_postgres.sql` |
| Server bundle installer | `scripts/install_server_bundle.sh` |
| Bundle packager | `scripts/package_server_bundle.sh` |
| Dashboard | `dashboard.html` |
| RAG knowledge base | configurable via `RAG_DOCS_PATH` |

---

*Consolidated from all documents in `docs/operations/`. For source-specific deep dives, see the individual files in `docs/operations/guides/` and `docs/operations/deployment/`.*
