# ADAI — Copilot Workspace Context

**Project:** ADAI (Advanced Deep Learning AI)  
**Language:** C++17  
**Build system:** CMake 3.14+  
**Architecture:** Encoder-decoder transformer (seq2seq chatbot)

---

## Canonical Directory Structure

```
adai/
├── .github/                  # GitHub: CI workflows, issue templates, this file
│   ├── copilot-instructions.md
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
│
├── .vscode/                  # VS Code editor settings (not committed by default)
│
├── benchmarks/               # Standalone benchmark programs (*.cpp)
│
├── build/                    # CMake out-of-source build tree (gitignored)
│
├── build-windows/            # MinGW cross-compilation build tree (gitignored)
│
├── cmake/
│   └── toolchains/           # CMake toolchain files (e.g. MinGW cross-compile)
│
├── dist-windows/             # Packaged Windows distribution (gitignored)
│
├── docker/
│   └── nginx/                # Nginx config for containerised reverse proxy
│
├── docs/
│   ├── development/          # Developer-facing documentation
│   │   ├── api/              # Per-component API reference
│   │   ├── architecture/     # Transformer design docs
│   │   ├── archive/          # Superseded docs kept for reference
│   │   ├── guides/           # How-to guides (build, CI, RAG, inference …)
│   │   ├── proposals/        # In-flight feature proposals
│   │   ├── reference/        # Low-level technical references (KV cache, …)
│   │   └── testing/          # Test strategy and coverage notes
│   ├── lessons/              # Accumulated ML training & architecture lessons
│   ├── operations/           # Ops / deployment documentation
│   │   ├── deployment/       # Docker, systemd, platform guides
│   │   └── guides/           # Operator runbooks
│   └── proposals/            # Top-level project proposals
│
├── examples/                 # Compilable usage examples (*.cpp, *.py)
│
├── external/
│   └── cpp-httplib/          # Vendored HTTP library (header-only)
│
├── gtest/                    # Vendored Google Test source
│
├── gutenberg_data/           # Raw Project Gutenberg training text (gitignored)
│
├── huggingface_data/         # Raw HuggingFace dataset files (gitignored)
│
├── legacy/                   # Deprecated code kept for reference only
│
├── scripts/                  # Shell / Python helper scripts
│   # Build helpers: build_and_vocab.sh, build_windows.sh, package_windows.sh
│   # Run helpers:   run_chatbot.sh, run_chatbot_gui.sh, model_service.sh
│   # Deployment:    docker_build.sh, docker_deploy.sh, install_*_service.sh
│   # Data helpers:  download_minipile.py, expand_training_data.py, …
│   # Quality:       format_code.sh, check_tech_debt.sh, scan_todos.sh
│
├── src/                      # All production C++ source & headers
│   ├── gpu/                  # Optional CUDA backend (MatrixGPU.cu/.hpp)
│   │
│   # ── Core math ──────────────────────────────────────────────────────────
│   ├── Matrix.{cpp,hpp}
│   ├── MatrixSIMD.hpp        # SIMD-accelerated matrix ops
│   ├── Activation.{cpp,hpp}
│   ├── LayerNorm.{cpp,hpp}
│   ├── PositionalEncoding.{cpp,hpp}
│   ├── FeedForward.{cpp,hpp}
│   ├── Optimizer.{cpp,hpp}
│   │
│   # ── Transformer blocks ─────────────────────────────────────────────────
│   ├── MultiHeadAttention.{cpp,hpp}
│   ├── CrossAttention.{cpp,hpp}
│   ├── EncoderBlock.{cpp,hpp}
│   ├── DecoderBlock.{cpp,hpp}
│   ├── LLMEncoder.cpp        # LLM-style encoder (decoder-only path)
│   ├── encoder.hpp
│   ├── Decoder.{cpp,hpp}
│   ├── EncoderDecoderModel.{cpp,hpp}
│   ├── TokenEmbedding.{cpp,hpp}
│   ├── LanguageModelHead.{cpp,hpp}
│   │
│   # ── Inference optimisations ────────────────────────────────────────────
│   ├── KVCache.hpp
│   ├── SpeculativeDecoding.hpp
│   ├── Quantization.hpp
│   ├── LoRA.hpp
│   ├── BatchedInferenceEngine.hpp
│   ├── IntegratedInferenceEngine.hpp
│   ├── PipelineInferenceEngine.hpp
│   ├── EfficientBatching.hpp
│   ├── BatchProcessor.hpp
│   ├── ParallelDataLoader.hpp
│   │
│   # ── NLP / Tokenization ─────────────────────────────────────────────────
│   ├── BPETokenizer.{cpp,hpp}
│   ├── SpecialTokens.hpp
│   ├── VocabBuilder.cpp
│   ├── VocabBuilderHelpers.hpp
│   ├── TextGenerator.{cpp,hpp}
│   ├── ConversationContext.{cpp,hpp}
│   │
│   # ── Training ──────────────────────────────────────────────────────────
│   ├── Dataset.hpp
│   ├── ChatbotTrainer.{cpp,hpp}
│   ├── IncrementalTrainer.{cpp,hpp}
│   ├── IncrementalTrainingTool.cpp  # CLI entry point for incremental training
│   ├── CheckpointManager.hpp
│   ├── PPOOptimizer.hpp
│   ├── RewardModel.hpp
│   │
│   # ── RAG / Document retrieval ───────────────────────────────────────────
│   ├── DocumentStore.{cpp,hpp}
│   ├── RAGInference.{cpp,hpp}
│   │
│   # ── Chatbot interfaces ─────────────────────────────────────────────────
│   ├── ChatbotAPI.{cpp,hpp}
│   ├── ChatbotAPIServer.cpp         # REST API server entry point
│   ├── ChatbotCLI.{cpp,hpp}
│   ├── ChatbotCLI_main.cpp          # CLI chatbot entry point
│   ├── ChatbotGUI.{cpp,hpp}
│   ├── ChatbotGUI_main.cpp          # Qt GUI entry point
│   ├── ChatbotGUI_wrapper.cpp
│   │
│   # ── Metrics / Monitoring ───────────────────────────────────────────────
│   ├── TrainingMetricsAPI.{cpp,hpp}
│   ├── TrainingMetricsAPIServer.cpp
│   ├── TrainingMetricsService.{cpp,hpp}
│   ├── MetricsTracker.hpp
│   ├── GenerationQualityMetrics.hpp
│   ├── PerformanceProfiler.hpp
│   │
│   # ── Infrastructure ─────────────────────────────────────────────────────
│   ├── Config.{cpp,hpp}
│   ├── Logger.{cpp,hpp}
│   └── CMakeLists.txt
│
├── tests/                    # Google Test unit & integration tests
│   # One *_test.cpp per src/ component (e.g. matrix_test.cpp).
│   # Shared helpers in test_base.hpp.
│   └── CMakeLists.txt
│
├── tizen-metrics-app/        # Samsung Tizen TV web-app (metrics dashboard)
│   ├── index.html
│   ├── config.xml
│   ├── css/
│   ├── js/
│   ├── certs/
│   └── deploy.sh
│
├── tools/
│   └── abnormal_review/      # Tools for reviewing abnormal training samples
│
├── training_sessions/        # Runtime-generated training artefacts (gitignored)
│   # session_N_checkpoint.bin*  — per-layer weight files for session N
│   # session_N_best.bin*        — best-loss weights snapshot for session N
│   # metrics.jsonl / metrics_summary.json
│   # abnormal_samples.json / abnormal_resolutions.json
│   # trainer.log, tinystories_train.log
│
├── CMakeLists.txt            # Root build definition
├── CMakePresets.json         # Named CMake configure/build presets
├── config.conf               # Runtime server & training configuration
├── docker-compose.yml
├── Dockerfile
├── LICENSE                   # MIT
├── README.md
└── vocab.txt                 # Active BPE vocabulary (committed)
```

---

## What Goes Where — Decision Rules

| Artefact | Canonical location |
|---|---|
| Trained model weights (`*.bin`, `*.bin.*`) | `training_sessions/` (gitignored) |
| Active / published model | `training_sessions/` symlinked or named explicitly in `config.conf` |
| Vocabulary file (committed) | repo root `vocab.txt` |
| Runtime logs | `training_sessions/` or a `logs/` subdirectory (gitignored) |
| Training data | `gutenberg_data/` or `huggingface_data/` (gitignored) |
| New C++ component | `src/Component.{cpp,hpp}` + matching `tests/component_test.cpp` |
| New example | `examples/ComponentExample.cpp` |
| Script (build / deploy / data) | `scripts/` |
| Feature proposal | `docs/development/proposals/` or `docs/proposals/` |
| Completed implementation note | `docs/development/guides/` |
| Architecture decision | `docs/development/architecture/` |
| Deployment guide | `docs/operations/deployment/` |
| ML learning note | `docs/lessons/` |

---

## Build Conventions

```bash
# Standard Linux build
cmake -B build && cmake --build build -j$(nproc)

# Run all tests
cd build && ctest --output-on-failure

# Windows cross-compilation
./scripts/build_windows.sh

# Code formatting (clang-format)
./scripts/format_code.sh
```

CMake targets: `chatbot` (CLI), `chatbot_api_server` (REST), `chatbot_trainer`,
`incremental_trainer`, `chatbot_gui` (Qt), `training_metrics_api_server`.

---

## Configuration Reference (`config.conf`)

Key runtime parameters:

| Key | Description |
|---|---|
| `VOCAB_PATH` | Path to `vocab.txt` |
| `MODEL_PATH` | Path to a trained `*.bin` checkpoint root |
| `PORT` | HTTP API port (default `8080`) |
| `D_MODEL` | Model hidden dimension |
| `NUM_HEADS` | Number of attention heads |
| `D_FF` | Feed-forward dimension |
| `NUM_ENCODER_LAYERS` / `NUM_DECODER_LAYERS` | Depth |
| `MAX_SEQ_LENGTH` | Maximum token sequence length |
| `LEARNING_RATE` / `WEIGHT_DECAY` | Training hyperparameters |

---

## Code Style

- `.clang-format` and `.clang-tidy` are committed; run `./scripts/format_code.sh` before committing.
- Headers use `#pragma once`.
- Logging via `Logger` class (`Logger.hpp`), not `std::cout`.
- No raw owning pointers; prefer `std::unique_ptr` / `std::shared_ptr`.
- New public APIs require a matching unit test and an entry in `docs/development/api/`.

---

## Key Subsystems

- **Matrix / SIMD** — `Matrix`, `MatrixSIMD` (AVX2 fast paths), optional CUDA backend in `src/gpu/`.
- **Transformer** — `EncoderBlock` → `LLMEncoder` / `Decoder` → `EncoderDecoderModel`.
- **Tokeniser** — BPE (`BPETokenizer`) with special-token management (`SpecialTokens.hpp`).
- **Inference** — `TextGenerator` + optional KV cache, speculative decoding, quantisation, LoRA.
- **Training** — `ChatbotTrainer` (full), `IncrementalTrainer` (streaming / online).
- **RAG** — `DocumentStore` (embedding + retrieval) + `RAGInference` (augmented generation).
- **REST API** — `cpp-httplib` based; session management, JSON responses.
- **Metrics** — `TrainingMetricsService` → `TrainingMetricsAPI` → Tizen dashboard app.
