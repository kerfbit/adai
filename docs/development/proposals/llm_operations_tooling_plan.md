# Planning Document: LLM Training and Operations Tooling Suite

## 1. Overview
As the ADAI LLM architecture matures, the ecosystem around model training, evaluation, and deployment needs to evolve. Developing dedicated tools to handle the lifecycle of an LLM—from data ingestion to serving—will standardize debugging, improve data hygiene, and streamline production deployments.

This document outlines the proposal for **four dedicated CLI tools** required to mature the ADAI LLM project.

---

## 2. Proposed Tooling Suite

### 2.1. Data Preparation Toolkit (`adai-data-prep`)
**Purpose:** Ensure high-quality training datasets through automated filtering and curation.
**Features:**
- **Deduplication:** MinHash or exact substring deduplication across large corpus files.
- **Quality Filtering:** Heuristics-based filtering (e.g., length thresholds, symbol-to-word ratios).
- **PII Scrubbing:** regex/dictionary-based masking of personally identifiable information.
- **Tokenization Profiler:** Dry-run tokenization to report token length distributions, sequence trimming percentages, and padding overhead.

### 2.2. Checkpoint & Quantization Manager (`adai-weights-tool`)
**Purpose:** Manage weight files, optimize for inference, and facilitate distributed development.
**Features:**
- **Quantization:** Compress FP32 weights to FP16, INT8, or INT4 (e.g., block-wise quantization) to reduce inference memory footprint.
- **Format Conversion:** Translate raw binary dumps to structured formats like safetensors or GGUF (if supported) for ecosystem compatibility.
- **Checkpoint Merging:** Interpolate/average weights from two continuous checkpoints to stabilize tuning.
- **Inspector:** Print model architecture metadata and layer-wise weight statistics (mean, variance) without loading the full model into VRAM.

### 2.3. Automated Evaluation Engine (`adai-eval`)
**Purpose:** Systematically benchmark newly trained checkpoints against static datasets before promotion.
**Features:**
- **Prompt Templating:** Automatically apply chat/instruction-tuning templates to evaluation inputs.
- **Standardized Benchmarks:** Run test suites containing generic tasks (e.g., MMLU-style Q&A, perplexity on hold-out data).
- **Deterministic Sampling:** Fix temperature to 0.0 (greedy) for consistent logic tests.
- **Diff Reports:** Output JSON/Markdown showing performance degradation/improvements compared to the last "best" checkpoint.

### 2.4. Serving Profiler & Load Tester (`adai-serve-profiler`)
**Purpose:** Evaluate real-world model serving performance and REST API throughput.
**Features:**
- **Concurrency Simulation:** Send simulated concurrent multi-turn user requests to the deployment server.
- **Performance Tracking:** Log Time-to-First-Token (TTFT), Tokens-per-Second (TPS) per user, and end-to-end latency.
- **Memory Profiler:** Dry-run model configuration limits (e.g., `max_seq_length * d_model * batch_size`) to estimate the hard ceiling for required RAM/VRAM.

---

## 3. Implementation Roadmap

### Phase 1: Foundational Checkpoint & Eval Tools
1. Implement `adai-weights-tool` starting with FP16/INT8 weight quantization logic, which directly impacts the current model usability on lower-end hardware.
2. Implement `adai-eval` hooking into our existing testing framework and enabling deterministic testing across fixed `conversation_history` sets.

### Phase 2: Data Pipeline Hardening
1. Build `adai-data-prep` in C++ or Python (if leveraging advanced NLP regex libraries) to preprocess JSONL corpus files.
2. Integrate the tokenization profiler to advise users on optimum `max_seq_length` and `batch_size`.

### Phase 3: Stress Testing
1. Deploy `adai-serve-profiler`.
2. Map throughput limits utilizing our existing BatchedInferenceEngine vs Standard Engine, generating performance curves.

## 4. Maintenance / Debt Resolution
These tools should be isolated as distinct executable targets within `CMakeLists.txt` (e.g., `add_executable(adai-eval tools/eval/main.cpp)`). This prevents binary bloat in the main `ChatbotAPI` while sharing core libraries (like `BPETokenizer`, `EncoderDecoderModel`).
