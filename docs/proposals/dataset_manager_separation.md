# Proposal: Separate Dataset Management from IncrementalTrainer

**Status:** Proposed  
**Date:** June 4, 2026 (revised June 6, 2026)  
**Author:** GitHub Copilot  
**Related code:** `src/IncrementalTrainer.hpp`, `src/IncrementalTrainer.cpp`, `src/IncrementalTrainingTool.cpp`

---

## 1. Summary

`IncrementalTrainer` currently combines three distinct concerns in one class: **training loop management**, **data queue and registry management**, and **external data fetching**. This proposal separates all three into dedicated components:

| Component | Responsibility |
|-----------|---------------|
| `IncrementalTrainer` | Training loop, session history, checkpoints, metrics push, TUI dashboard |
| `DatasetRegistry` | Data queue and registry (local flat-files **or** remote `registry_server`), checksums, training-file parsing; multi-run job assignment in distributed mode |
| `DataFetcher` | Network download + conversion for Project Gutenberg and HuggingFace |

`DataFetcher` is stateless: each method downloads and converts data, then **returns a file path**. The caller enqueues the result via `DatasetRegistry::add_file()`. This means `DataFetcher` has no dependency on `DatasetRegistry`, and `IncrementalTrainer` has no dependency on `DataFetcher` — the dependency graph is a strict DAG with no cycles.

A standalone `dataset_manager` CLI binary wraps `DatasetRegistry` and `DataFetcher` and can run independently of any training process.

In **distributed mode** (Phase 9), `DatasetRegistry` is backed by a lightweight `registry_server` HTTP daemon. This enables a single registry to coordinate multiple concurrent `IncrementalTrainer` instances across different machines: each trainer atomically **acquires** a disjoint subset of pending files, trains independently, and **marks them trained** when done. Local flat-file mode remains the default; distributed mode is additive with no API changes.

---

## 2. Background and Motivation

### 2.1. Current Responsibilities of `IncrementalTrainer`

The class currently owns three unrelated groups of behaviour:

| Group | Members |
|-------|---------|
| **Training** | `train_incremental()`, `train_full_retrain()`, `resume_last_session()`, session history, checkpoint management, metrics push, TUI dashboard |
| **Data queue management** | `add_new_data()`, `add_new_data_batch()`, `clear_pending_data()`, `get_pending_data_files()`, `get_trained_data_files()`, `is_data_trained()`, `compute_data_checksum()`, `load_data_registry()`, `save_data_registry()`, `print_data_registry()`, `load_pending_data_list()`, `save_pending_data_list()` |
| **External data fetching** | `add_gutenberg_book()`, `add_gutenberg_books()`, `download_gutenberg_book()`, `download_gutenberg_books()`, `get_gutenberg_url()`, `download_file()`, `clean_gutenberg_text()`, `extract_sentences()`, `generate_question_from_sentence()`, `create_qa_pairs_from_text()`, `convert_gutenberg_to_training_data()`, `add_huggingface_dataset()`, `download_hf_rows()`, `convert_hf_to_training_data()`, and ~250 lines of hand-written JSON helpers |

The external fetching code alone accounts for roughly **600 lines** inside a class whose primary purpose is training. The HuggingFace JSON helpers, Gutenberg text cleaner, and sentence extractor have no coupling to the optimizer, the loss function, or the session checkpoint system.

### 2.2. Why Not Just One `DatasetManager`?

The straightforward split would be two components: `IncrementalTrainer` and a single `DatasetManager` containing both queue management and external fetching. However, these two data-side concerns have meaningfully different properties that justify a further split:

| Property | Data queue management | External data fetching |
|----------|-----------------------|------------------------|
| Network dependency | None | `curl`, HTTP |
| Dependencies | `<filesystem>`, `<fstream>` | `curl`, `<regex>`, ~250 lines of JSON helpers |
| Change frequency | Stable once written | Evolves as new sources/formats are added |
| Test speed | Milliseconds (temp files) | Requires network or curl mock |
| Security surface | None | `std::system("curl ...")` calls |
| Used by `IncrementalTrainer` | Yes — reads pending queue at session start | No — never needed during training |

`IncrementalTrainer` needs to read the pending queue and mark files trained, but it has no reason to know that Gutenberg or HuggingFace exist. Placing fetching and queue management in the same class would still leave `IncrementalTrainer` coupled to a class that carries network I/O logic.

### 2.3. Problems Caused by the Current Design

1. **Testing friction.** Unit-testing `train_incremental()` requires constructing an `IncrementalTrainer`, which then also loads the data registry and pending-file list. These are orthogonal concerns.

2. **`IncrementalConfig` bloat.** Fields like `data_registry_file`, `cache_tokenized_data`, and `tokenized_cache_dir` live in the same struct as `auto_save_every_samples` and `metrics_server_url`, even though none of the data fields are read during training.

3. **Coupling to network I/O.** A training class that calls `std::system("curl ...")` is harder to sandbox, test in CI, and reason about from a security standpoint.

4. **Cannot add/inspect data without constructing a trainer.** Querying pending files or checking the registry always forces the construction of `BPETokenizer` and `EncoderDecoderModel`, which is wasteful and fragile in scripts.

5. **Cannot run data preparation in parallel with a training run.** Because data state is owned by the trainer object, preparing new data while a training process holds the session directory is risky.

6. **No extension point for new data sources.** Adding a Wikipedia or Common Crawl fetcher means modifying `IncrementalTrainer`. With `DataFetcher` as a dedicated class, new sources are additive changes to one file.

### 2.4. Future Requirements: Remote Registry and Multi-Run Coordination

The single-machine split above is the primary deliverable of this proposal. However, a registry that stores state only in local flat files creates friction for two emerging use-cases:

1. **Multi-machine training pools.** When multiple GPU machines share a common workload, each machine must train on a different subset of the pending data. With a local-only registry there is no way to coordinate which machine trains on which files, leading to duplicate work or manual partitioning.

2. **Centralised data governance.** When a dedicated data-preparation machine downloads and converts data for multiple trainers, those trainers need access to the shared pending queue from remote hosts. A local `pending_files.txt` is not reachable from another machine.

The `RegistryTransport` abstraction (§4.8) makes `DatasetRegistry` **transport-agnostic**. Local flat-file I/O is the default and the only mandatory deliverable of this proposal. Remote transport via `registry_server` is an additive Phase 9 change with no breaking impact on `DatasetRegistry`'s public API.

---

## 3. Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              dataset_manager (CLI binary)                        │
│     fetch commands ──► DataFetcher    queue commands ──► DatasetRegistry │
└──────────────────────────────────────────────────────────────────┘
          │  returns path                      ▲
          │                                    │ add_file()
          └────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│              incremental_trainer (CLI binary)                    │
│     training commands ──► IncrementalTrainer                     │
└──────────────────────────────────────────────────────────────────┘
                                   │ reads pending_files.txt
                                   ▼
                             DatasetRegistry
                             (mark_trained on success)
```

Dependency graph (arrows = "depends on"):

```
DataFetcher          (no dependencies on other adai classes)
DatasetRegistry      (no dependencies on other adai classes)
IncrementalTrainer ──► DatasetRegistry  (reads queue, calls mark_trained)
dataset_manager    ──► DataFetcher + DatasetRegistry
incremental_trainer──► IncrementalTrainer + DatasetRegistry
```

The two components communicate through the **shared manifest file** (`pending_files.txt`) on disk. `DatasetRegistry` writes it; `IncrementalTrainer` reads it once at the start of each session and does not own it.

In **distributed mode** (Phase 9+), the manifest and registry are served by a `registry_server` HTTP daemon and `IncrementalTrainer` calls `DatasetRegistry::acquire_pending(run_id)` to atomically claim a disjoint subset of files. Multiple trainers on different machines run concurrently without double-training:

```
┌─────────────────────────┐     ┌──────────────────────────────┐
│  Machine A (data prep)  │     │  Machines B / C (trainers)   │
│  dataset_manager        │     │  incremental_trainer ×N      │
│  ├─ DataFetcher         │     │  DatasetRegistry             │
│  └─ DatasetRegistry ────┼─────┤  (RemoteTransport)           │
│     (RemoteTransport)   │     └───────────────┬──────────────┘
└─────────────────────────┘                     │
                                                ▼
                               ┌─────────────────────────────┐
                               │  registry_server (HTTP)     │
                               │  pending queue + run locks  │
                               │  data_registry.txt          │
                               └─────────────────────────────┘
```

---

## 4. Proposed Changes

### 4.1. New `DatasetRegistry` Class

**New file:** `src/DatasetRegistry.hpp` / `src/DatasetRegistry.cpp`

`DatasetRegistry` owns all local data state: the registry on disk, the pending queue, checksums, and training-file parsing. It has zero network dependency.

```cpp
// src/DatasetRegistry.hpp

#pragma once
#include <chrono>
#include <set>
#include <string>
#include <vector>
#include "ChatbotTrainer.hpp"  // ConversationPair

struct DataVersion {
    std::string data_file;
    std::string checksum;
    int         num_samples = 0;
    std::chrono::system_clock::time_point added_time;
    bool        trained = false;
};

struct DatasetConfig {
    // Local state
    std::string session_dir          = "training_sessions";
    std::string data_registry_file   = "data_registry.txt";
    bool        cache_tokenized_data = false;
    std::string tokenized_cache_dir  = "tokenized_cache";

    // Remote registry (empty = local mode)
    std::string registry_server_url;          // e.g. "http://registry-host:8081"
    std::string run_group;                    // logical namespace; defaults to session_dir basename
    int         registry_timeout_ms = 5000;   // HTTP timeout in ms

    // Multi-run data partitioning
    std::string run_id;                       // unique per-process ID; auto-generated if empty
    int         max_files_per_run = 0;        // 0 = acquire all available pending files
};

class DatasetRegistry {
public:
    explicit DatasetRegistry(DatasetConfig cfg = {});

    static DatasetConfig make_config(const adai::ServiceConfig& svc);

    // Local file queue
    bool add_file(const std::string& path);
    bool add_files(const std::vector<std::string>& paths);
    void clear_pending();
    std::vector<std::string> pending_files() const;
    std::vector<std::string> trained_files() const;
    bool is_trained(const std::string& path) const;

    // Single-run: mark all trained files at session end.
    void mark_trained(const std::vector<std::string>& paths,
                      const std::vector<int>& sample_counts);

    // Multi-run: atomically acquire up to max_files pending files for run_id.
    // In local mode a file lock serialises concurrent acquisitions on the same host.
    // In remote mode registry_server provides the atomic guarantee.
    // Returns an empty vector if no unassigned files are currently available.
    std::vector<std::string> acquire_pending(const std::string& run_id,
                                              int max_files = 0);

    // Multi-run: mark files trained and release the in-progress reservation.
    void mark_trained(const std::string& run_id,
                      const std::vector<std::string>& paths,
                      const std::vector<int>& sample_counts);

    // Release files back to the pending queue without training (use on failure/crash).
    void release_pending(const std::string& run_id,
                         const std::vector<std::string>& paths);

    // Report which files are currently assigned to which run_id.
    void print_run_assignments() const;

    // Persistence
    bool load_registry();
    bool save_registry();
    bool load_pending_list();
    bool save_pending_list();

    // Reporting
    void print_registry() const;
    int  total_samples_trained() const;

    // Training-file parser (pure I/O, no network, no model dependency)
    static int load_conversation_pairs(const std::string& path,
                                       std::vector<ConversationPair>& out);

    static std::string compute_checksum(const std::string& path);

private:
    DatasetConfig            config_;
    std::vector<DataVersion> registry_;
    std::set<std::string>    trained_set_;
    std::vector<std::string> pending_;
};
```

### 4.2. New `DataFetcher` Class

**New file:** `src/DataFetcher.hpp` / `src/DataFetcher.cpp`

`DataFetcher` is stateless. Every method downloads and converts data, then returns the path to the produced training file (empty string on failure). The caller is responsible for enqueueing the result via `DatasetRegistry::add_file()`. This means `DataFetcher` has **no dependency on `DatasetRegistry`** — it can be used in any context that just needs a converted training file.

```cpp
// src/DataFetcher.hpp

#pragma once
#include <string>
#include <vector>

struct FetcherConfig {
    std::string gutenberg_output_dir  = "gutenberg_data";
    std::string huggingface_output_dir = "huggingface_data";
};

class DataFetcher {
public:
    explicit DataFetcher(FetcherConfig cfg = {});

    // Returns path to produced training file, or "" on failure.
    std::string fetch_gutenberg(int book_id, int num_pairs = 500);

    // Returns paths for each book (empty string for each failed book).
    std::vector<std::string> fetch_gutenberg_batch(const std::vector<int>& ids,
                                                   int num_pairs_each = 500);

    // Returns path to produced training file, or "" on failure.
    std::string fetch_huggingface(const std::string& dataset_id,
                                  int num_pairs = 500,
                                  const std::string& split        = "train",
                                  const std::string& input_field  = "",
                                  const std::string& output_field = "");

private:
    FetcherConfig config_;

    bool        download_file(const std::string& url, const std::string& dest);
    std::string clean_gutenberg_text(const std::string& raw);
    std::vector<std::string> extract_sentences(const std::string& text);
    std::string generate_question_from_sentence(const std::string& sentence);
    bool        convert_gutenberg_to_training_data(const std::string& text_file,
                                                   const std::string& output_file,
                                                   int max_pairs);
    bool        download_hf_rows(const std::string& dataset_id, const std::string& split,
                                 int offset, int length, const std::string& output_path);
    bool        convert_hf_to_training_data(const std::string& rows_dir,
                                            const std::string& output_file,
                                            const std::string& input_field,
                                            const std::string& output_field,
                                            int max_pairs);
    // All JSON helpers (hf_unescape, hf_extract_string, etc.) as private statics
};
```

The CLI wires the two classes together:

```cpp
// dataset_manager CLI — gutenberg command
DataFetcher fetcher(fetcher_cfg);
DatasetRegistry registry(dataset_cfg);
registry.load_registry();
registry.load_pending_list();

std::string path = fetcher.fetch_gutenberg(book_id, num_pairs);
if (!path.empty()) {
    registry.add_file(path);
    std::cout << "✅ Added to pending queue: " << path << "\n";
}
```

### 4.3. Trimmed `IncrementalConfig`

Data-specific fields migrate out of `IncrementalConfig`. It retains only training-relevant settings:

```cpp
struct IncrementalConfig {
    TrainingConfig base_config;

    // Session management
    std::string session_dir          = "training_sessions";
    int         max_sessions_to_keep = 50;

    // Auto-save
    bool auto_save_enabled           = true;
    int  auto_save_every_samples     = 1000;
    int  auto_save_every_minutes     = 30;

    // Checkpointing
    bool        save_incremental_checkpoints = true;
    std::string checkpoint_dir               = "checkpoints";
    bool        enable_checkpoint_symlinks   = true;
    std::string latest_symlink_name          = "latest_checkpoint.bin";
    std::string best_symlink_name            = "best_checkpoint.bin";

    // Metrics push (TD-021)
    std::string metrics_server_url;
    std::string metrics_session_label;
    int         metrics_push_timeout_ms = 1000;
};
```

`IncrementalTrainer::make_incremental_config()` continues to work unchanged. Parallel factories for the new config types:

```cpp
static DatasetConfig  DatasetRegistry::make_config(const adai::ServiceConfig& svc);
static FetcherConfig  DataFetcher::make_config(const adai::ServiceConfig& svc);
```

### 4.4. `IncrementalTrainer` Interface Changes

`IncrementalTrainer` no longer owns any data state. It depends only on `DatasetRegistry` (to read the pending list and mark files trained after a successful session).

**Removed from `IncrementalTrainer`:**
- `add_new_data()`, `add_new_data_batch()`, `clear_pending_data()`
- `get_pending_data_files()`, `get_trained_data_files()`
- `is_data_trained()`, `compute_data_checksum()`
- `load_data_registry()`, `save_data_registry()`
- `print_data_registry()`
- `add_gutenberg_book()`, `add_gutenberg_books()`
- `add_huggingface_dataset()`
- All private Gutenberg/HuggingFace helpers
- `load_conversation_pairs()` (moves to `DatasetRegistry` as a static)

**Added to `IncrementalTrainer`:**

```cpp
// Run an incremental session on a caller-supplied file list.
// The caller must call DatasetRegistry::mark_trained() on success.
bool train_on_files(const std::vector<std::string>& files, int num_epochs);

// Full retrain on a caller-supplied file list.
bool retrain_on_files(const std::vector<std::string>& files, int num_epochs);
```

The existing `train_incremental(int)` and `train_full_retrain(int)` are **deprecated** but preserved as compatibility shims that construct a temporary `DatasetRegistry`, read `pending_files.txt`, and call `train_on_files()`. This keeps existing callers compiling without changes during the transition.

### 4.5. `IncrementalTrainer` Constructor Changes

The constructors no longer call `load_data_registry()` or `load_pending_data_list()`:

```cpp
IncrementalTrainer::IncrementalTrainer(const std::string& config_file_path) {
    // ... load svc, build model, create metrics reporter ...
    ensure_directories_exist();
    load_session_history();        // retained — session state is still trainer-owned
    // load_data_registry();       // REMOVED — DatasetRegistry's responsibility
    // load_pending_data_list();   // REMOVED
    // ... resume best checkpoint logic unchanged ...
}
```

### 4.6. `IncrementalTrainingTool.cpp` CLI Changes

Data commands construct only `DatasetRegistry` (and `DataFetcher` when fetching). No model is loaded and no `BPETokenizer` is initialised for any data command:

```
incremental_trainer --config config.conf add <file>           → DatasetRegistry only
incremental_trainer --config config.conf gutenberg 1342 500   → DataFetcher + DatasetRegistry
incremental_trainer --config config.conf huggingface ...      → DataFetcher + DatasetRegistry
incremental_trainer --config config.conf train [epochs]       → IncrementalTrainer + DatasetRegistry
incremental_trainer --config config.conf retrain [epochs]     → IncrementalTrainer + DatasetRegistry
incremental_trainer --config config.conf status               → DatasetRegistry + IncrementalTrainer (session history)
incremental_trainer --config config.conf history              → IncrementalTrainer
```

### 4.7. New `dataset_manager` Standalone Binary

A new CMake target provides a lightweight binary with zero dependency on the model or tokenizer:

```cmake
add_executable(dataset_manager
    DatasetManagerTool.cpp
    DatasetRegistry.cpp
    DataFetcher.cpp
)
target_link_libraries(dataset_manager PRIVATE adai_common)
# adai_common: Logger, Config — no BPETokenizer, no EncoderDecoderModel
```

`DatasetManagerTool.cpp` exposes the full data workflow independently of any training run:

```
Usage: dataset_manager [--config <path>] [--registry-url <url>] [--run-id <id>] <command> [options]

Commands:
  add <file>                     Add a local data file to the pending queue
  gutenberg <id> [pairs]         Download and convert a Gutenberg book
  gutenberg-batch <id1,id2,...>  Download multiple books
  huggingface <dataset> [pairs] [split] [in] [out]
                                 Download a HuggingFace dataset
  status                         Show registry and pending queue
  clear                          Remove all pending (untrained) entries
  mark-trained <file>            Manually mark a file as trained
  checksum <file>                Print the checksum of a data file
  acquire [--max-files <n>]      Atomically acquire pending files for this run (distributed mode)
  release                        Release all files reserved by this run without training
  runs                           Show active run assignments across all trainers (distributed mode)
```

This binary can run while a training process is active, since it only touches `pending_files.txt` and `data_registry.txt`.

### 4.8. RegistryTransport Abstraction

**New file:** `src/RegistryTransport.hpp`

To keep `DatasetRegistry` transport-agnostic, all persistence I/O is delegated through a `RegistryTransport` interface. `DatasetRegistry` receives a `unique_ptr<RegistryTransport>` at construction via the static factory `DatasetRegistry::make_transport(cfg)`, which returns `LocalTransport` when `cfg.registry_server_url` is empty and `RemoteTransport` otherwise.

```cpp
// src/RegistryTransport.hpp

struct PendingEntry {
    std::string path;
    std::string assigned_to;   // run_id, or "" if unassigned
};

class RegistryTransport {
public:
    virtual ~RegistryTransport() = default;

    virtual bool load_registry(std::vector<DataVersion>& out)      = 0;
    virtual bool save_registry(const std::vector<DataVersion>& in) = 0;

    virtual bool load_pending(std::vector<PendingEntry>& out)      = 0;
    virtual bool save_pending(const std::vector<PendingEntry>& in) = 0;

    // Returns false if path is already reserved for another run_id.
    virtual bool try_acquire(const std::string& run_id, const std::string& path) = 0;
    virtual void release(const std::string& run_id, const std::string& path)     = 0;
};

// LocalTransport: flat-file I/O + flock() (POSIX) / LockFileEx (Win32) for atomicity.
class LocalTransport final : public RegistryTransport { /* ... */ };

// RemoteTransport: HTTP calls to registry_server (§4.9).
// Uses ETag-based optimistic locking to prevent conflicting writes.
class RemoteTransport final : public RegistryTransport { /* ... */ };
```

### 4.9. registry_server (Remote Mode)

**New file:** `src/RegistryServer.cpp` — standalone `registry_server` binary

A lightweight HTTP daemon (same infrastructure pattern as `metrics-api-server`) that provides a central pending-queue and registry service for distributed training pools:

- Stores `data_queue.txt` and `data_registry.txt` on one host reachable by all machines
- Handles atomic acquire/release using an in-memory lock table (flushed to disk on each write)
- Supports multiple independent **run groups** (`DatasetConfig::run_group`) in a single instance

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/registry/<group>/queue` | GET | Return all pending entries with run assignments |
| `/registry/<group>/acquire` | POST `{"run_id":"…","max_files":N}` | Atomically assign up to N files to `run_id` |
| `/registry/<group>/release` | POST `{"run_id":"…","files":[…]}` | Release reserved files back to unassigned pool |
| `/registry/<group>/trained` | POST `{"run_id":"…","files":[…],"samples":[…]}` | Mark files trained |
| `/registry/<group>/registry` | GET | Return full data registry |
| `/registry/<group>/runs` | GET | Return current in-progress run assignments |

The `<group>` segment maps to `DatasetConfig::run_group`, allowing independent projects to share one server. State is stored in `registry_sessions/<group>/` using the same flat-file format as `LocalTransport`, so manual inspection and recovery are possible even when the server is offline.

---

## 5. Data Flow After Refactor

```
# Workflow 1: fetch and train (unchanged user experience)
dataset_manager --config config.conf gutenberg 1342 500
  └── DataFetcher::fetch_gutenberg(1342, 500)
        → download + convert → gutenberg_data/gutenberg_1342_training.txt
  └── DatasetRegistry::add_file(path)
        → append to pending_files.txt

incremental_trainer --config config.conf train 10
  └── DatasetRegistry::load_pending_list()
        → read pending_files.txt → ["gutenberg_data/gutenberg_1342_training.txt"]
  └── IncrementalTrainer::train_on_files(files, 10)
        → DatasetRegistry::load_conversation_pairs() per file
        → train → checkpoint
  └── DatasetRegistry::mark_trained(files, sample_counts)
        → update data_registry.txt, clear pending_files.txt

# Workflow 2: prepare next dataset while training is running
dataset_manager --config config.conf huggingface tatsu-lab/alpaca 500 &
incremental_trainer --config config.conf train 5  # uses previous pending set
# Safe: DataFetcher writes to huggingface_data/; DatasetRegistry appends to
# pending_files.txt only after the current trainer has already read it.

# Workflow 3: multi-machine training pool (distributed mode)
# config.conf on all machines sets:
#   REGISTRY_SERVER_URL=http://registry-host:8081
#   RUN_GROUP=my_project

# Machine A (data preparation only):
dataset_manager gutenberg-batch 1342,11,84,1661 300
  └── DataFetcher::fetch_gutenberg_batch(...)  → 4 training files
  └── DatasetRegistry (RemoteTransport) → POST /registry/my_project/acquire not needed;
        add_file() → POST /registry/my_project/queue  (4 entries, unassigned)

# Machine B (trainer-1, runs concurrently with Machine C):
ADAI_RUN_ID=trainer-b incremental_trainer train 10
  └── DatasetRegistry::acquire_pending("trainer-b", max=2)
        → POST /registry/my_project/acquire → files 1+2 locked for trainer-b
  └── IncrementalTrainer::train_on_files({file1, file2}, 10)
  └── DatasetRegistry::mark_trained("trainer-b", {file1, file2}, {n1, n2})
        → POST /registry/my_project/trained → locked entries marked trained, queue cleared

# Machine C (trainer-2, acquires different files):
ADAI_RUN_ID=trainer-c incremental_trainer train 10
  └── DatasetRegistry::acquire_pending("trainer-c", max=2)
        → POST /registry/my_project/acquire → files 3+4 (B already holds 1+2)
  └── IncrementalTrainer::train_on_files({file3, file4}, 10)
  └── DatasetRegistry::mark_trained("trainer-c", {file3, file4}, {n3, n4})
```

---

## 6. Migration Plan

| Phase | Change | Risk |
|-------|--------|------|
| **1a** | Create `DatasetRegistry.hpp/.cpp` by moving registry/queue/checksum/parser code verbatim from `IncrementalTrainer`. No behaviour change. | Low — pure code move |
| **1b** | Create `DataFetcher.hpp/.cpp` by moving all Gutenberg/HuggingFace code verbatim from `IncrementalTrainer`. JSON helpers become file-local statics. No behaviour change. | Low — pure code move |
| **1c** | Create a thin `DatasetManager` facade that holds both a `DatasetRegistry` and a `DataFetcher` and re-exposes their combined API. `IncrementalTrainer` continues to call `DatasetManager` methods — zero breakage. | Low — additive facade |
| **2** | Add `DatasetConfig` and `FetcherConfig` structs. Update `IncrementalConfig` to remove data fields. Update `make_incremental_config()` to also produce the new config types. | Low — struct refactor |
| **3** | Add `train_on_files()` / `retrain_on_files()` to `IncrementalTrainer`. Implement `train_incremental()` / `train_full_retrain()` as shims. All existing tests pass unchanged. | Low — additive |
| **4** | Remove `load_data_registry()`, `save_data_registry()`, `load_pending_data_list()`, `save_pending_data_list()` from `IncrementalTrainer` constructors. `IncrementalTrainer` takes a `DatasetRegistry&` or reads `pending_files.txt` directly. Update `IncrementalTrainingTool.cpp` and tests in the same commit. | Medium — constructor change |
| **5** | Update `IncrementalTrainingTool.cpp` so data commands use `DatasetRegistry`/`DataFetcher` directly (not via the facade). `DatasetManager` facade can be removed or kept as a convenience type. | Low |
| **6** | Add `dataset_manager` CMake target and `DatasetManagerTool.cpp`. | Low — additive |
| **7** | *(Optional)* Remove deprecated shims `train_incremental()` / `train_full_retrain()`. Remove `DatasetManager` facade if no longer needed. | Deferred |
| **8** | Add `RegistryTransport` interface and `LocalTransport`; wrap all `DatasetRegistry` flat-file I/O in `LocalTransport`. Public API of `DatasetRegistry` unchanged. | Low — internal refactor |
| **9** | Implement `RemoteTransport` + `registry_server` HTTP binary. Add `acquire_pending()`, `release_pending()`, `mark_trained(run_id, ...)` overload to `DatasetRegistry`. Extend `DatasetConfig` with remote/multi-run fields. `ServiceConfig` gains `REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID` keys. | Medium — new network component |
| **10** | Update `IncrementalTrainer` to call `DatasetRegistry::acquire_pending(run_id)` at session start instead of `load_pending_list()`. `run_id` auto-derived from `hostname + PID` when not configured (same derivation as `metrics_session_key` in TD-021). | Low — caller change only |

Phases 1a, 1b, 1c can be merged in a single commit with zero behaviour change. Phases 2–3 are independent additive commits. Phase 4 is the only breaking change and lands with its test update in one commit.

---

## 7. Impact on Tests

`tests/incrementaltrainer_test.cpp` currently constructs `IncrementalTrainer` and calls `add_new_data()` etc. After Phase 4 those calls migrate. The test file is updated in the same commit as Phase 4:

- **`IncrementalTrainerTests`** suite: constructs `DatasetRegistry` for data setup; `IncrementalTrainer` for training. Model construction is unchanged.
- **`DatasetRegistryTests`** suite *(new)*: unit-tests registry load/save, checksum, pending queue, and `load_conversation_pairs()` using temp files only. No model, no tokenizer — runs in milliseconds.
- **`DataFetcherTests`** suite *(new, optional)*: tests field auto-detection, Gutenberg text cleaning, and HuggingFace conversion with pre-recorded JSON fixtures. No network access required in CI.

---

## 8. Files Created / Modified

| File | Action |
|------|--------|
| `src/DatasetRegistry.hpp` | **New** — `DataVersion`, `DatasetConfig`, `DatasetRegistry` class |
| `src/DatasetRegistry.cpp` | **New** — registry I/O, pending list, checksum, `load_conversation_pairs` |
| `src/DataFetcher.hpp` | **New** — `FetcherConfig`, `DataFetcher` class |
| `src/DataFetcher.cpp` | **New** — all Gutenberg + HuggingFace logic; JSON helpers as file-local statics |
| `src/DatasetManager.hpp/.cpp` | **New (Phase 1c, temporary)** — thin facade over `DatasetRegistry` + `DataFetcher` |
| `src/DatasetManagerTool.cpp` | **New** — standalone `dataset_manager` CLI entry point |
| `src/IncrementalTrainer.hpp` | **Modified** — remove data-management public API; add `train_on_files()` / `retrain_on_files()` |
| `src/IncrementalTrainer.cpp` | **Modified** — remove ~900 lines; add shim implementations calling `DatasetRegistry` |
| `src/IncrementalTrainingTool.cpp` | **Modified** — data commands use `DatasetRegistry`/`DataFetcher`; training commands unchanged |
| `src/CMakeLists.txt` | **Modified** — add `dataset_manager` target; split `DatasetRegistry.cpp` / `DataFetcher.cpp` into library |
| `tests/incrementaltrainer_test.cpp` | **Modified** — data-side calls migrate to `DatasetRegistry` |
| `tests/CMakeLists.txt` | **Modified** — add `DatasetRegistryTests`, `DataFetcherTests`, and `RegistryTransportTests` targets |
| `src/RegistryTransport.hpp` | **New (Phase 8)** — `PendingEntry`, `RegistryTransport` interface, `LocalTransport`, `RemoteTransport` |
| `src/RegistryServer.cpp` | **New (Phase 9)** — `registry_server` HTTP daemon |
| `tests/RegistryTransportTests.cpp` | **New (Phase 8/9)** — unit tests for `LocalTransport`; stub-server integration tests for `RemoteTransport` |

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `pending_files.txt` race condition if `dataset_manager` and `incremental_trainer` run simultaneously | `IncrementalTrainer` reads the pending list **once at session start**. `DatasetRegistry::mark_trained()` is called only after training completes. `DataFetcher` writes only to its own output directories and never touches `pending_files.txt` directly — the CLI appends to the queue after fetching. No concurrent writes to the same file. |
| Callers using `add_gutenberg_book()` directly on an `IncrementalTrainer` instance | Compatibility shim in `DatasetManager` facade (Phase 1c) preserved and marked `[[deprecated]]`; removed no earlier than Phase 7. |
| `load_conversation_pairs()` called from within the training loop | Moves to `DatasetRegistry` as a `static` method. `IncrementalTrainer` calls `DatasetRegistry::load_conversation_pairs()` — one clean dependency direction, no circular coupling. |
| `compute_data_checksum()` called from `train_incremental()` when building `DataVersion` entries | Moves to `DatasetRegistry::compute_checksum()`. After the refactor `IncrementalTrainer` calls `DatasetRegistry::mark_trained(files, sample_counts)`, which computes checksums internally. |
| Adding a new data source (e.g. Wikipedia, Common Crawl) | Add a method to `DataFetcher` only. `DatasetRegistry` and `IncrementalTrainer` require no changes. The new method returns a path; the CLI enqueues it. |
| Concurrent queue writes in local multi-run mode (two trainers on same host) | `LocalTransport::try_acquire()` holds an `flock()` (POSIX) / `LockFileEx` (Win32) only for the read + update + write cycle — typically microseconds. Callers block briefly rather than corrupt the file. |
| `registry_server` unavailable during a training run | `RemoteTransport` caches the acquired file list locally at acquire time; training proceeds to completion. `mark_trained` and `release_pending` retry with exponential back-off and flush to the server on reconnect. The trainer never silently double-trains. |
| ETag / optimistic-lock collision on the remote queue | `RemoteTransport::try_acquire()` uses GET + ETag, computes the update, then PUT with `If-Match`. On 412 Precondition Failed the full cycle retries up to 3 times with exponential back-off before returning an error. |
| Two processes generate the same default `run_id` | `run_id` defaults to `sanitize(hostname) + PID`, mirroring the `metrics_session_key` derivation (TD-021). PID is unique per running process on the same host; combine with hostname for cross-machine uniqueness. |
