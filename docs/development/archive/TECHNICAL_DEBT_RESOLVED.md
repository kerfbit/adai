# Technical Debt - Resolved Items

Resolved items extracted from [TECHNICAL_DEBT.md](../guides/TECHNICAL_DEBT.md).

## Resolved Items

### TD-061: GPU LayerNorm Backward (CUDA and SYCL) Computed Wrong Input Gradients

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | GPU / Core Training Math | One-token fix (`r*r*r` → `r*r`) in both backends, verified analytically and numerically |

Summary:
Found while reading `src/gpu/MatrixGPU.cu` end to end. `layer_norm_bwd_dx_kernel`'s gradient
w.r.t. the LayerNorm input had a genuine math bug affecting **every LayerNorm backward call on the
GPU training path** (multiple per encoder/decoder block, every layer, every step) — this is not a
minor rounding issue: verified by finite-difference check to be off by a large fraction of the
gradient's own magnitude, not a small numerical discrepancy.

Root cause: the kernel accumulates `sum(d_xn * xn)` — `xn` being the *already-normalized*
`(x-mean)*rstd` value, cached from the forward pass — then scales it by `-0.5 * rstd^3` to get the
gradient w.r.t. variance (`d_var`). The CPU implementation
(`LayerNorm::backward()` in `src/LayerNorm.cpp`) computes the mathematically equivalent quantity
using `sum(d_xn * (x - mean))` instead — the *pre-normalization* difference — and that version's
`-0.5 * rstd^3` scaling is correct **for that quantity**. Since `xn = (x-mean) * rstd`, reusing the
already-normalized value needs one *fewer* power of `rstd` to reach the same `d_var`; the GPU
kernel copied the CPU version's `rstd^3` scaling without adjusting for the different quantity it
was scaling, leaving an extra, spurious factor of `rstd` in every input gradient's variance
contribution.

Verified three ways:
1. **Analytical derivation** of the correct closed-form LayerNorm backward (`dx_j = rstd·h_j −
   (rstd/N)·xn_j·S − (rstd/N)·H`, where `h = d_xn`, `S = Σh_i·xn_i`, `H = Σh_i`) from first
   principles via the chain rule through `mean` and `rstd` (both functions of every `x_i` in the
   row) — confirms the correct power of `rstd` is 1, not 2, in the `xn_j·S` term the kernel's
   `d_var` feeds into.
2. **Numerical finite-difference check** (standalone Python/NumPy script implementing the exact
   kernel formula structure): the buggy formula (`r^3` in `d_var`) diverges from central-difference
   numerical gradients by up to ~0.3 in a toy example with gradients of magnitude ~0.02–0.6 — a
   large relative error, not float noise. The corrected formula (`r^2` in `d_var`, everything else
   unchanged) matches finite differences to ~1e-10 across 5 random trials of varying size.
3. **SYCL cross-check**: `src/gpu/sycl/MatrixGPU_SYCL.cpp`'s `matrix_layer_norm_bwd_gpu` has the
   byte-for-byte identical `r * r * r` bug — this CUDA kernel was ported "formula-for-formula" from
   the SYCL version (per this file's own section header comment), so the bug predates this session
   and affects both GPU backends identically. The CPU path (`LayerNorm.cpp`) is unaffected — it
   uses `(x - mean)` directly, which happens to need the correct power of `rstd` already.

No test exercises `matrix_layer_norm_bwd_gpu`/`layer_norm_bwd_dx_kernel` on either backend (grep
confirms zero references in `tests/`), consistent with [TD-041](../guides/TECHNICAL_DEBT.md#td-041-gpuutils-has-no-dedicated-test-on-either-backend)'s
broader finding that GPU-only code has minimal dedicated coverage — why this went undetected.

**Checkpoint impact — narrower than TD-059's:** this bug only affects the *backward* pass (gradient
computation during training), not the forward computation a saved checkpoint's weights encode.
Existing checkpoints trained with `-DENABLE_GPU=ON` or `-DENABLE_SYCL=ON` remain valid and usable —
their forward pass was never wrong — but their training likely converged less well than it should
have, since every LayerNorm's contribution to the gradient was systematically distorted throughout
training. No retraining is *required*, though anyone who trained primarily on GPU may want to
consider a fresh run now that the gradient is correct. CPU-only training was never affected.

Changes Made:

- `src/gpu/MatrixGPU.cu`: changed `d_var`'s scaling from `r * r * r` to `r * r` (one token), with a
  comment explaining the derivation and pointing at `LayerNorm.cpp` for the CPU-side reference
  quantity it must match.
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`: identical one-token fix, identical explanatory comment.
- `src/gpu/MatrixGPU.cu`: retagged `stable` → `beta` (0.9.0) — this specific bug is fixed, but the
  incident exposed that most of this file's kernels (not just `layer_norm_bwd`) have no dedicated
  test and, more fundamentally, can only ever execute on real GPU hardware — which was not available
  to verify any of them during this file's original "stable" rollout or since. `stable` was not an
  earned claim. Action item for whoever next touches this file: add dedicated tests for the
  remaining untested kernels (`matrix_add`/`multiply`/`transpose`, `gelu_backward`,
  `cross_entropy_loss`/`grad`, batch ops) alongside `matrixgpu_td003_test.cpp`'s existing coverage,
  and run the full GPU test suite on real hardware before re-promoting to `stable`.

Verification:

- ✅ Analytical re-derivation of the correct formula from the chain rule (see above).
- ✅ Standalone NumPy finite-difference check, both confirming the bug (pre-fix formula) and the fix
  (post-fix formula, ~1e-10 agreement) across multiple random trials.
- ✅ `adai_gpu` (containing `MatrixGPU.cu`) rebuilds clean under the local CUDA toolchain
  (`nvcc` present; no physical GPU in this environment, so the kernel could not be *run* here — the
  fix is verified mathematically and by a clean compile, not by an on-device numerical test).
- ⚠️ The SYCL build could not be compiled or tested in this environment either (no Intel oneAPI
  `icpx` toolchain available — same limitation noted in TD-041); the fix there is verified only by
  the formula being byte-for-byte identical to the now-fixed-and-compiled CUDA version.

Files Changed:

- `src/gpu/MatrixGPU.cu`
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`

---

### TD-060: EncoderDecoderModel::forward() Segfaulted on an Empty target_tokens

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Core Model / Training | Rewrote the loop bound to a form that can't underflow, plus a regression test |

Summary:
Found while reading `src/EncoderDecoderModel.cpp` end to end, in the course of investigating
TD-050's greedy-decode KV-cache workaround (confirmed that workaround still matches the tracker's
description exactly — nothing new there). `forward()` — a public method explicitly documented
"for custom training loops" — built the teacher-forcing decoder input with
`for (size_t i = 0; i < target_tokens.size() - 1; ++i)`. `size_t` is unsigned, so an empty
`target_tokens` makes `size() - 1` wrap to `SIZE_MAX`, turning the loop into an out-of-bounds read
of `target_tokens[i]` for ever-increasing `i`. Reproduced directly with a standalone
ASan/UBSan build: immediate `SEGV` on the first out-of-bounds `vector::push_back` read. Every
built-in caller (`train_step`/`evaluate`, via `tokenizer->encode(text, true)`) always produces a
non-empty `target_tokens` (at minimum bos+eos), which is why this was never hit through the normal
API surface — but `forward()`/`train_step_tokenized()`/`evaluate_tokenized()` all accept
already-tokenized vectors directly and are documented for exactly that use.

Changes Made:

- Changed the loop condition to `i + 1 < target_tokens.size()`, the same underflow-proof idiom used
  for TD-058, instead of `i < target_tokens.size() - 1`.
- Added `EncoderDecoderModelTest.ForwardEmptyTargetTokensDoesNotCrash` to
  `tests/encoderdecoder_test.cpp`.

Verification:

- ✅ Standalone ASan/UBSan repro confirms the exact SEGV crash mode before the fix.
- ✅ `EncoderDecoderTests` rebuilds clean and passes (this suite alone runs ~9 minutes as part of
  the full project's `ctest` — 77/77 suites, 100% pass, 0 failures, including this fix).

Files Changed:

- `src/EncoderDecoderModel.cpp`
- `tests/encoderdecoder_test.cpp`

---

### TD-058: BPETokenizer::get_most_frequent_pair() Could Underflow Its Loop Bound on an Empty Entry

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | NLP / Tokenizer | Rewrote the loop bound to a form that can't underflow, plus a regression test |

Summary:
Found while reading `src/BPETokenizer.cpp` end to end. `get_most_frequent_pair()` — a **public
static** method, not a private helper — counted adjacent-pair frequencies with
`for (size_t i = 0; i < tokens.size() - 1; i++)` for each inner `tokens` vector. `size_t` is
unsigned, so an empty `tokens` entry makes `tokens.size() - 1` wrap to `SIZE_MAX`, turning the loop
into an out-of-bounds read starting at `tokens[0]` on an empty vector. Every internal caller
(`build_bpe_merges()`) happens to only ever push non-empty entries into `word_tokens`, which is why
this was never hit in practice, but as a public API this method has no way to enforce that
invariant on a caller outside this file.

Changes Made:

- Changed the loop condition to `i + 1 < tokens.size()`, an idiom that is correct (and never
  underflows) for every value of `tokens.size()` including 0, instead of relying on callers to
  never pass an empty entry.
- Added `BPETokenizerTest.GetMostFrequentPairIgnoresEmptyWordEntries` to `tests/tokenizer_test.cpp`,
  calling the public static method directly with an empty entry mixed into `word_tokens` and
  asserting it neither throws/crashes nor lets the empty entry affect the correct result.

Verification:

- ✅ `runTests` (the `tokenizer_test.cpp` binary) rebuilds clean; all 53 tests pass.

Files Changed:

- `src/BPETokenizer.cpp`
- `tests/tokenizer_test.cpp`

---

### TD-057: LazyDataset::get_sample() Threw on a Bare "INPUT:"/"RESPONSE:" Legacy-Format Line

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Data / Dataset | Anchored prefix check + correct substr offset, plus a regression test |

Summary:
Found immediately after TD-056 while continuing the same end-to-end read of `src/Dataset.hpp`.
`LazyDataset::get_sample()`'s legacy-format branch used `line.find("INPUT:") != npos` /
`line.find("RESPONSE:") != npos` only as an existence check, then unconditionally sliced with a
fixed offset — `line.substr(7)` / `line.substr(10)` — assuming the key starts at column 0 and is
immediately followed by `": "`. Two independent problems: (1) a bare `"INPUT:"` or `"RESPONSE:"`
line (no trailing space/content — shorter than the assumed prefix) makes the offset exceed the
line's length, and `std::string::substr` throws `std::out_of_range` when `pos > size()` — confirmed
directly (`"INPUT:".substr(7)` throws `basic_string::substr: __pos (which is 7) > this->size()
(which is 6)`); (2) even for a normal line, `find()` matches the key anywhere in the string but the
substr offset is always relative to column 0, so a key not literally at the start would be sliced
from the wrong position. In practice every file this codebase itself writes
(`Dataset::save_to_file(..., "conversation")`) places the key at column 0, which is why this went
unnoticed, but any hand-edited or externally-supplied legacy-format file with a short/malformed line
could crash `LazyDataset::get_sample()` with an uncaught exception.

Changes Made:

- Replaced `find()` + fixed-offset `substr()` with `line.rfind("INPUT:", 0) == 0` /
  `line.rfind("RESPONSE:", 0) == 0` (anchored prefix checks, same idiom already used by
  `IncrementalTrainer::load_conversation_pairs()`'s legacy-format parser) and `substr()` at the
  prefix's actual length (6 / 9, not 7 / 10), followed by trimming leading whitespace — matches
  behavior for well-formed input while making a short/malformed line return an empty field instead
  of throwing.
- Added `DatasetTest.LazyDatasetHandlesBareLegacyKeys` to `tests/dataset_test.cpp`, asserting
  `get_sample()` doesn't throw on a file containing only bare `"INPUT:"`/`"RESPONSE:"` lines.

Verification:

- ✅ Standalone repro confirms the exact `std::out_of_range` thrown by the old code.
- ✅ `datasetTests` rebuilds clean; all 42 tests pass (41 prior + this one), including in isolation
  (`--gtest_filter=*BareLegacyKeys*`).

Files Changed:

- `src/Dataset.hpp`
- `tests/dataset_test.cpp`

---

### TD-056: Dataset::load_from_file() Called front() on an Empty String for Empty Files

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Data / Dataset | One-line guard + regression test |

Summary:
Found while reading `src/Dataset.hpp` end to end. `load_from_file()`'s format-detection chain reads
the first line into `first_line`, then checks `first_line.front() == '{'` to detect JSONL. For an
existing-but-empty (0-byte) file, `std::getline` leaves `first_line` empty and
`std::string::front()` on an empty string is undefined behavior per the C++ standard. Reproduced
directly: compiled a minimal repro with `-D_GLIBCXX_ASSERTIONS -fsanitize=address,undefined`
(libstdc++ 13) and it aborts with `Assertion '!empty()' failed`; in a normal (non-hardened) build it
returns unspecified data instead of crashing, silently falling through the detection chain rather
than reliably defaulting to conversation format. No existing test in `dataset_test.cpp` loaded a
genuinely empty (as opposed to non-existent) file, so this was never caught.

Changes Made:

- Added `!first_line.empty() &&` to the JSONL-detection branch's condition in `load_from_file()`.
- Added `DatasetTest.LoadEmptyFileDoesNotCrash` to `tests/dataset_test.cpp`, creating a real 0-byte
  file and asserting `load_from_file()` returns `false` (dataset stays empty) instead of crashing.

Verification:

- ✅ Standalone repro confirms the crash mode before the fix (`_GLIBCXX_ASSERTIONS` + ASan/UBSan).
- ✅ `datasetTests` rebuilds clean; all 41 tests pass, including the new regression test in
  isolation (`--gtest_filter=*EmptyFile*`).

Files Changed:

- `src/Dataset.hpp`
- `tests/dataset_test.cpp`

---

### TD-055: get_total_training_time_hours() Truncated Every Session to Whole Hours Before Summing

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Training / Reporting | One-line fix: sum fractional hours instead of truncated ones |

Summary:
Found while reading `src/IncrementalTrainer.cpp` end to end. `get_total_training_time_hours()` (used
by `print_training_summary()`'s "Total time" line) computed each session's duration with
`std::chrono::duration_cast<std::chrono::hours>(end_time - start_time)` and summed the results.
`duration_cast` to `hours` truncates toward zero, so any single session under 60 minutes — the
common case for incremental/online training passes — contributed exactly 0 to the total, and a
95-minute session contributed 1 rather than ~1.58. The only existing test
(`GetTotalTrainingTimeHoursZeroInitially`) only covers the trivial no-session case, so this never
surfaced. A deployment training in short, frequent increments would see "Total time: 0.00 h" in the
summary log indefinitely regardless of how much wall-clock time had actually elapsed.

Changes Made:

- Changed the per-session duration computation to `std::chrono::duration<double,
  std::ratio<3600>>` (fractional hours) instead of `duration_cast<hours>`, and accumulate in
  `double` before narrowing to the `float` return type once at the end.
- Also removed a stray, contentless `// Metrics API Server Management` section-header comment
  left at the very end of the file (from the original March 2026 metrics-service commit; the
  actual management code lives in `TrainingMetricsAPI.{cpp,hpp}`, already reviewed and found
  complete) — it read like the file had been truncated mid-edit.

Verification:

- ✅ `incrementaltrainerTests` rebuilds clean and the full suite passes.

Files Changed:

- `src/IncrementalTrainer.cpp`

---

### TD-054: ModelNameService's Legacy JSONL-to-SQLite Migration Silently Dropped Every Record

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | MNS / Persistence | One-line fix: bind the missing 26th column |

Summary:
Found while reading `src/ModelNameService.cpp` end to end. `init_db()` creates the `models` table
with 26 columns (`run_group` added last by a later migration — see the comment above
`add_column_if_missing("models", "run_group", ...)` explaining why it must stay last for
positional-INSERT compatibility). `persist_model()`, the actively-used save path, correctly binds
all 26 columns. `migrate_from_jsonl()` — the one-time import that runs when `models.db` is freshly
created and a legacy `models.jsonl` exists — used an INSERT literal with only 25 `?` placeholders
and never bound `run_group`. Reproduced directly with a standalone SQLite3 script against the
real schema: `sqlite3_prepare_v2` fails with `"table models has 26 columns but 25 values were
supplied"`, and the surrounding `if (... != SQLITE_OK) continue;` silently skips the record with
no logging — so any deployment upgrading from a pre-SQLite (or pre-`run_group`) MNS data directory
would import zero models from its `models.jsonl`, with nothing but a `migrated 0 records` log line
as a symptom (`roles.json` migration is a separate code path and unaffected).

Changes Made:

- Added the missing 26th placeholder to the `INSERT OR REPLACE INTO models VALUES (...)` literal
  in `migrate_from_jsonl()` and bound `r.run_group` (already parsed by `parse_record()`) to it,
  matching `persist_model()`'s statement exactly.
- Added an inline comment at the INSERT explaining the column-count invariant so a future column
  addition doesn't reintroduce the same silent failure in this specific call site.

Verification:

- ✅ Reproduced the failure and the fix against the real 26-column schema with a standalone
  Python/`sqlite3` script (25 placeholders → `OperationalError: table models has 26 columns but
  25 values were supplied`; 26 placeholders → succeeds, `run_group` lands correctly).
- ✅ `adai_mns` (the static library containing `ModelNameService.cpp`) rebuilds clean.

Action item not covered by this fix: `ModelNameService` (the server-side class, as opposed to
`ModelNameClient`) still has no dedicated unit test — only `modelnameclient_test.cpp` (client) and
`mns_manager_gui_test.cpp` (GUI) exercise it, both indirectly over HTTP. A test that actually drives
`migrate_from_jsonl()` against a scratch data dir would have caught this at write time; none exists
today. Filing this gap is out of scope for this fix — flagged for whoever next touches this file,
matching [TD-040](../guides/TECHNICAL_DEBT.md#td-040-ftpdataservers-auth-path-unreviewed-registryserver-untested-in-isolation)'s "no dedicated unit test" framing for a sibling daemon class.

Files Changed:

- `src/ModelNameService.cpp`

---

### TD-029: Fix GCC 13 ICE in raginference_test.cpp

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 7, 2026 (verified, not actively fixed — see note) | Tests / RAGInference | No longer reproduces on GCC 13.3.0 |

Summary:
Filed against an Internal Compiler Error (`cc1plus` SIGSEGV) that reportedly prevented
`raginferenceTests` from compiling on GCC 13. Re-verified during a pass confirming every active
TD item has an appropriate in-code marker: a clean `rm` of the object file followed by a fresh
`cmake --build . --target raginferenceTests` on this machine's GCC 13.3.0 compiled without error,
and all 31 tests in the suite pass. Most likely this was fixed upstream in a GCC 13 point release
between whenever this item was filed (June 7, 2026) and 13.3.0, though the original offending
construct was never identified, so it's possible a different GCC 13.x minor version could still
hit it. No code changes were made — `RAGInference.{cpp,hpp}` were promoted from `beta` to `stable`
since this was their only recorded blocker.

Verification:

- ✅ `raginferenceTests` builds clean on GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) after removing
  the stale object file first.
- ✅ All 31 tests in `raginferenceTests` pass.

If this resurfaces on a different GCC 13.x point release, re-open as a new item rather than
reverting this one — the root cause was never isolated, so there's no fix to "undo."

---

### TD-020: Persistent Metrics Storage via SQL Database

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 7, 2026 (tracker correction — implementation predates this entry) | Training / Metrics / API / Infrastructure | `IMetricsDatabase` abstraction, `SQLiteMetricsDatabase` (WAL mode), optional `PostgresMetricsDatabase`, `MetricsDatabaseFactory`, `MetricsSessionRegistry` DB ownership, four new REST endpoints, config keys, 884-line `MetricsDatabaseTest.cpp` |

Summary:
The proposal (`docs/development/archive/persistent-metrics-sql-storage.md`) was found fully implemented and verified during the per-file production-readiness rollout (see [file-status-standard.md](../guides/file-status-standard.md)) — this entry backfills the tracker, which had continued to list the item as "Active/Planned" after the work was actually done. Nine of the ten action items are complete and verified in code; one narrow sub-item (bundling the SQLite amalgamation specifically for Windows/MinGW cross-compilation, since the current CMake setup only finds a system-installed SQLite3) was never done and has been split off as its own item, [TD-032](../guides/TECHNICAL_DEBT.md#td-032-bundle-sqlite3-amalgamation-for-windows-cross-compilation).

Changes Made:

- ✅ `IMetricsDatabase` interface and `SessionRecord` struct defined in `src/MetricsDatabase.hpp`.
- ✅ `SQLiteMetricsDatabase` implemented with WAL mode (`PRAGMA journal_mode = WAL;`) and prepared statements (`src/SQLiteMetricsDatabase.hpp/.cpp`).
- ✅ Optional `PostgresMetricsDatabase` implemented (`src/PostgresMetricsDatabase.hpp/.cpp`).
- ✅ `MetricsDatabaseFactory::create(...)` implemented — declared in `src/MetricsDatabase.hpp`, defined in `src/SQLiteMetricsDatabase.cpp` (folded into the existing database header rather than a separate `MetricsDatabaseFactory.hpp` file as originally proposed; functionally equivalent).
- ✅ `IMetricsDatabase*` wired into `TrainingMetricsService` via `set_database()`.
- ✅ `MetricsSessionRegistry` owns and initializes the database instance (`src/MetricsSessionRegistry.hpp`), constructing it via `MetricsDatabaseFactory::create()` and injecting it into each session.
- ✅ Four new REST endpoints added to `TrainingMetricsAPI`: `/api/sessions/{key}/metrics/history` (time-range, explicitly commented "TD-020: DB-backed time-range history query"), `/api/metrics/compare` (cross-session), `/api/metrics/aggregate` (status-filtered live session list), `/api/sessions/{key}/metrics/export` (full history export).
- ✅ `METRICS_STORAGE_BACKEND`, `METRICS_DB_PATH`, `METRICS_DB_URL`, `METRICS_DB_POOL_SIZE` all present in `src/Config.hpp/.cpp` and `config.metrics.conf`.
- ✅ `tests/MetricsDatabaseTest.cpp` written (884 lines) covering schema bootstrap, WAL mode, round-trip insert/query, and related paths.
- ⬜ SQLite amalgamation bundling for Windows/MinGW builds — not done; split off as TD-032.

Files Modified:

- `src/MetricsDatabase.hpp` (new)
- `src/SQLiteMetricsDatabase.hpp` / `src/SQLiteMetricsDatabase.cpp` (new)
- `src/PostgresMetricsDatabase.hpp` / `src/PostgresMetricsDatabase.cpp` (new)
- `src/MetricsSessionRegistry.hpp`
- `src/TrainingMetricsService.hpp` / `src/TrainingMetricsService.cpp`
- `src/TrainingMetricsAPI.hpp` / `src/TrainingMetricsAPI.cpp`
- `src/Config.hpp` / `src/Config.cpp`
- `config.metrics.conf`
- `tests/MetricsDatabaseTest.cpp` (new)

Verification:

- ✅ `MetricsDatabaseTest.cpp` present and substantial (884 lines); `src/CMakeLists.txt` links SQLite3 into `adai_core` ("TD-020: Link SQLite3 into adai_core") and registers the SQLite metrics backend
- ✅ Confirmed by direct code inspection during this rollout, not by re-running the historical test suite

---

### TD-031: Fix RegistryServer Logger::init call signature mismatch

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 7, 2026 | RegistryServer / Build | One-line fix: replaced `Logger::init("registry_server")` with `Logger::init(Logger::Level::INFO, "registry_server")` in `src/RegistryServer.cpp:386`; `registry_server` target now builds cleanly |

Summary:
`RegistryServer.cpp` was calling `Logger::init` with a bare string as the first argument, matching a signature that no longer exists after the logger API was updated to require an explicit `Level` as the first parameter. The fix adds `Logger::Level::INFO` (the default used by all other binaries in the project) as the first argument, matching the canonical `Logger::init(Level, const std::string&)` overload.

Changes Made:

- `src/RegistryServer.cpp`: `Logger::init("registry_server")` → `Logger::init(Logger::Level::INFO, "registry_server")` at line 386.

---

### TD-030: GPU Strategy CLI Option for incremental\_trainer

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 7, 2026 | Training / IncrementalTrainer / GPU | Added `--gpu-strategy background\|full` CLI flag and `GPU_STRATEGY` config key; threaded `bool use_low_priority` from `ServiceConfig` through `Matrix::gpu_try_initialize()` to `GPUManager::initialize()`; 4 config-parsing tests added and passing |

Summary:
`incremental_trainer` previously used a hardcoded low-priority CUDA stream regardless of deployment context. This made it polite on shared workstations but underutilised on dedicated training machines. The `background` strategy preserves the original low-priority behaviour; `full` selects the highest-priority CUDA stream so training is never preempted. `GPU_MEMORY_FRACTION` remains an independent knob. The strategy is selectable via `--gpu-strategy` on the command line or `GPU_STRATEGY=` in the config file, with `background` as the default for backward compatibility.

Changes Made:

- `src/Config.hpp`: Added `enum class GPUStrategy : uint8_t { BACKGROUND, FULL }` with `gpu_strategy_from_string()` helper; added `GPUStrategy gpu_strategy = GPUStrategy::BACKGROUND` field to `ServiceConfig`; added `<cstdint>` and `<iostream>` includes.
- `src/Config.cpp`: Added `GPU_STRATEGY` parsing in both `load_from_file()` and `load_from_env()`.
- `src/gpu/GPUUtils.hpp`: Added `bool use_low_priority = true` parameter to `GPUManager::initialize()` (real and CPU-only stub); stream creation now selects `priority_low` or `priority_high` based on the parameter.
- `src/Matrix.hpp` / `src/Matrix.cpp`: Added `bool use_low_priority = true` parameter to `gpu_initialize()` and `gpu_try_initialize()` declarations and definitions; forwarded to `GPUManager::initialize()`.
- `src/IncrementalTrainingTool.cpp`: Added `--gpu-strategy` global flag stripping alongside `--config`; applies CLI override to `svc_config.gpu_strategy`; passes `use_low_priority` to `gpu_try_initialize()`; logs active strategy mode; updated usage text.
- `config.conf` / `config-remote.conf`: Added commented `GPU_STRATEGY` stub after `GPU_MEMORY_FRACTION`.
- `tests/config_test.cpp`: Added `unsetenv("GPU_STRATEGY")` to `clearEnvironmentVariables()`; added 4 tests: `GpuStrategyFileBackground`, `GpuStrategyFileFull`, `GpuStrategyUnknownDefaultsToBackground`, `GpuStrategyEnvVarOverridesFile`; all pass.

---

### TD-028: Separate Dataset Management from IncrementalTrainer

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 7, 2026 | Training / Data Management / IncrementalTrainer | 10-phase refactor introducing `DatasetRegistry`, `DataFetcher`, `RegistryTransport` (Local + Remote), `registry_server` HTTP daemon, and `dataset_manager` binary; 53 tests across 4 suites verified clean |

Summary:
`IncrementalTrainer` previously combined three unrelated concerns: the training loop, data-queue/registry management, and external data fetching (Gutenberg, HuggingFace). The 10-phase refactor separated these into focused components with clean boundaries. `DatasetRegistry` owns the pending queue, trained-file registry, checksums, and the `INPUT:/RESPONSE:` parser; it has zero network dependency and is fully testable with temp files. `DataFetcher` is a stateless class owning all download and conversion logic; each method returns a file path the caller enqueues via `DatasetRegistry::add_file()`. `IncrementalTrainer` was stripped to the training loop, session history, checkpoints, and metrics push, accepting file lists through `train_on_files()` / `retrain_on_files()`. A standalone `dataset_manager` binary links only `DatasetRegistry` and `DataFetcher`, enabling data preparation to run concurrently with a training process. In distributed mode (Phase 9), `DatasetRegistry` is backed by a `registry_server` HTTP daemon that multiple trainer instances query to atomically acquire disjoint subsets of the pending queue, preventing double-training across a pool of machines.

Changes Made:

- ✅ **Phase 1a–1c**: Created `DatasetRegistry`, `DataFetcher`, and a temporary `DatasetManager` backward-compat facade. All Gutenberg/HuggingFace logic and JSON helpers moved to `DataFetcher.cpp`. `#include <regex>` removed from `IncrementalTrainer.cpp`. All three `.cpp` files added to `adai_core`.
- ✅ **Phase 2**: Removed `data_registry_file`, `cache_tokenized_data`, `tokenized_cache_dir` from `IncrementalConfig`; added `DatasetConfig dataset_config_` to `IncrementalTrainer` populated via `DatasetRegistry::make_config(svc)`.
- ✅ **Phase 3**: Added `train_on_files()` and `retrain_on_files()` to `IncrementalTrainer`; deprecated `train_incremental()` and `train_full_retrain()` as delegating shims.
- ✅ **Phase 4**: Removed auto-loading from all three `IncrementalTrainer` constructors; shims now load/save via a local `DatasetRegistry` instance. Six CLI commands updated with explicit load/save calls. All 40 tests pass unchanged.
- ✅ **Phase 5**: `IncrementalTrainingTool` data commands (`add`, `gutenberg`, `gutenberg-batch`, `huggingface`) now construct only `DatasetRegistry`/`DataFetcher` — no model or tokenizer initialised.
- ✅ **Phase 6**: Added `src/DatasetManagerTool.cpp` and `dataset_manager` CMake target. Eight commands (`add`, `gutenberg`, `gutenberg-batch`, `huggingface`, `status`, `list-pending`, `list-trained`, `clear-pending`). No model dependency.
- ✅ **Phase 7**: Removed deprecated shims (`train_incremental()`, `train_on_new_data_only()`, `train_full_retrain()`) and the `DatasetManager` facade entirely. `resume_last_session()` and `IncrementalTrainingTool` `train`/`retrain` commands migrated to `train_on_files()` / `retrain_on_files()` with `DatasetRegistry`.
- ✅ **Phase 8**: Created `src/RegistryTransport.hpp/.cpp` — `DataVersion` (moved from `DatasetRegistry.hpp`), `PendingEntry`, abstract `RegistryTransport` interface, and `LocalTransport` (flat-file I/O). `DatasetRegistry` holds `std::unique_ptr<RegistryTransport> transport_`; transport-injection constructor added for testing. `RegistryTransportTests` (8 tests) added.
- ✅ **Phase 9**: Implemented `RemoteTransport` (cpp-httplib, guarded by `BUILD_METRICS_API_SERVER`) and `registry_server` HTTP daemon (`src/RegistryServer.cpp`, 6 endpoints, port 8082). Added `acquire_pending()`, `release_pending()`, `mark_trained(run_id,…)`, `print_run_assignments()` to `DatasetRegistry`. `LocalTransport` uses advisory `flock()` on a `.lock` sentinel file for per-host atomicity. Pending file format extended to `path\trun_id` (tab-separated; backward compatible). `ServiceConfig`/`DatasetConfig` extended with `REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID`, `REGISTRY_TIMEOUT_MS`. Transport factory auto-selects `LocalTransport` (no URL) or `RemoteTransport`. `RegistryTransportPhase9Tests` (10 tests) added.
- ✅ **Phase 10**: `IncrementalTrainer::resume_last_session()` and `IncrementalTrainingTool` `train`/`retrain` commands now call `acquire_pending(run_id)` instead of `load_pending_list()` + `pending_files()`, and `mark_trained(run_id,…)` instead of the three-call sequence (`mark_trained` + `clear_pending` + `save_pending_list`). `release_pending(run_id, paths)` called on failure to return files to the pool. `run_id` auto-derived from `hostname[:8] + "_" + pid%10000` via `detect_hostname_fragment()` / `detect_pid_mod_10000()` (IncrementalTrainer) or `derive_run_id()` helper (IncrementalTrainingTool).
- ✅ **Tests**: `DatasetRegistryTests` (31 tests across 4 suites: config defaults, `make_config`, `compute_checksum`, `load_conversation_pairs`, full pending-queue and trained-set API, persistence round-trips, and Phase 9 multi-run wrappers) and `DataFetcherTests` (4 offline tests) added to `tests/CMakeLists.txt`. All 53 TD-028 tests pass. Fixed pre-existing bug in `RegistryTransportPhase9Tests` where manually-constructed `DataVersion` objects with empty `checksum` caused the space-delimited registry parser to mis-align fields.

Files Created: `src/DatasetRegistry.hpp`, `src/DatasetRegistry.cpp`, `src/DataFetcher.hpp`, `src/DataFetcher.cpp`, `src/DatasetManagerTool.cpp`, `src/RegistryTransport.hpp`, `src/RegistryTransport.cpp`, `src/RegistryServer.cpp`, `tests/registry_transport_test.cpp`, `tests/RegistryTransportTests.cpp`, `tests/DatasetRegistryTests.cpp`, `tests/DataFetcherTests.cpp`

Files Modified: `src/IncrementalTrainer.hpp`, `src/IncrementalTrainer.cpp`, `src/IncrementalTrainingTool.cpp`, `src/Config.hpp`, `src/Config.cpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`

Files Removed: `src/DatasetManager.hpp`, `src/DatasetManager.cpp` *(Phase 1c backward-compat facade, removed in Phase 7)*

Related Items:

- TD-006: `create_qa_pairs_from_text()` now lives in `DataFetcher.cpp` — natural location for future FIM data generation.
- TD-014: `DataFetcher` is the natural home for future `adai-data-prep` tooling.
- TD-029: Fresh `Config.cpp` compilation in `build-gpu-clang` triggers the same GCC 13 ICE noted during Phase 6; cached object unaffected.

---

### TD-027: Install Script for incremental_trainer Sub-System

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 7, 2026 | Tooling / Deployment / IncrementalTrainer | Created `scripts/install_incremental_trainer.sh` supporting local, remote (SSH+rsync), and coordinator-only install modes; all 12 action items completed |

Summary:
`scripts/install_incremental_trainer.sh` was created to automate deployment of the `incremental_trainer` sub-system (all three binaries: `incremental_trainer`, `dataset_manager`, `registry_server`) to local and remote hosts. The script follows the established pattern from `install_systemd_service.sh` and `install_metrics_service.sh` (color helpers, `set -euo pipefail`, preflight checks, step-by-step logging, confirmation prompt). Local install creates the full directory layout, copies binaries with correct permissions, installs `config.conf` and `vocab.txt`, appends idempotent distributed-registry config stubs, creates the `adai` system user, sets ownership, and verifies both binaries execute correctly. Remote install performs all steps over SSH + rsync. Coordinator mode installs only `registry_server` and writes an `adai-registry.service` systemd unit. `scripts/README.md` was updated with a full usage table. Distributed-registry config stubs (`REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID`, `REGISTRY_TIMEOUT_MS`) were added to both `config.conf` and `config-remote.conf`.

Changes Made:

- ✅ Created `scripts/install_incremental_trainer.sh` with `set -euo pipefail`, color helpers (`info`, `success`, `warn`, `error`), and full argument parsing loop.
- ✅ Implemented all flags: `--install-path`, `--user`, `--group`, `--build-dir`, `--config-src`, `--vocab-src`, `--with-registry-server`, `--coordinator`, `--remote`, `--sync-sessions`, `--ssh-key`, `--help`.
- ✅ Local install: 7-step process covering user creation, directory layout (including `gutenberg_data/`, `huggingface_data/`), binary copy (mode 755), config/vocab copy, registry stub append, ownership, and post-install verification.
- ✅ Remote install: SSH mkdir, rsync binaries + config + vocab, remote chmod, remote registry stub append, optional `--sync-sessions` rsync, and remote post-install verification.
- ✅ Coordinator install: `registry_server` binary only, coordinator `config.conf` stub, `adai-registry.service` systemd unit installed, enabled, and started.
- ✅ Post-install verification runs `incremental_trainer status` (exit-code tolerant) and `dataset_manager --help` both locally and over SSH for remote installs.
- ✅ Updated `scripts/README.md` with full usage examples and options table.
- ✅ Added distributed-registry config stubs to `config.conf` and `config-remote.conf`.

Files Created: `scripts/install_incremental_trainer.sh`

Files Modified: `scripts/README.md`, `config.conf`, `config-remote.conf`

Related Items:

- TD-028 (Resolved): Separated dataset management — defined the three binaries this script deploys.
- TD-008 (Resolved): Daemon Service Implementation — `install_systemd_service.sh` style guide followed.
- TD-025 (Resolved): Background launch capability exposed to operators via the installed trainer.
- TD-018 (Resolved): `install_metrics_service.sh` is a companion script for the metrics API server.

---

### TD-023: Parallel Generation Quality Scoring via Model Snapshot

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 4, 2026 | Training / ChatbotTrainer / Metrics | Added `EncoderDecoderModel::clone()` (save/load via temp files), `std::optional<std::thread> generation_quality_thread_` on `ChatbotTrainer`, `join_generation_quality_thread()` helper, async branch in `compute_generation_quality_metrics()`, join in destructor and `release_model()`, `generation_quality_async_threshold` config key (default 50) wired through `TrainingConfig`, `ServiceConfig`, `Config.cpp`, `IncrementalTrainer::make_incremental_config()`, `config.conf`, and `config-remote.conf`; 10-test suite in `tests/generation_quality_async_test.cpp` |

Summary:
`ChatbotTrainer::compute_generation_quality_metrics()` previously ran synchronously on the training thread, blocking the start of the next epoch for the full `generate_response()` loop. When `generation_quality_sample_size >= generation_quality_async_threshold` (default 50), the function now clones the model weights into a temporary `EncoderDecoderModel` copy via `clone()` (which serialises to a unique temp path under `std::filesystem::temp_directory_path()` and immediately removes the files), launches a `std::thread` that scores against the snapshot, stores the thread in `generation_quality_thread_`, and returns immediately. The training loop can proceed without waiting. Before launching a new thread at the next epoch's validation phase, any previous thread is joined. The destructor and `release_model()` also join the thread, preventing use-after-free. Below the threshold the original synchronous path is unchanged.

Changes Made:

- ✅ Added `generation_quality_async_threshold = 50` to `TrainingConfig` in `src/ChatbotTrainer.hpp`.
- ✅ Added `#include <optional>` and `#include <thread>` to `src/ChatbotTrainer.hpp`; added `std::optional<std::thread> generation_quality_thread_` private member and `join_generation_quality_thread()` private helper declaration.
- ✅ Changed `~ChatbotTrainer() = default` to an explicit destructor in `src/ChatbotTrainer.cpp` that calls `join_generation_quality_thread()`.
- ✅ Implemented `join_generation_quality_thread()` — joins and resets the optional thread.
- ✅ Rewrote `compute_generation_quality_metrics()`: joins prior thread, branches on `sample_size >= config.generation_quality_async_threshold`; async path clones model and launches scoring thread; sync path unchanged.
- ✅ Updated `release_model()` to call `join_generation_quality_thread()` before releasing weights.
- ✅ Implemented `EncoderDecoderModel::clone() const` in `src/EncoderDecoderModel.cpp` using the existing `save_model` / `load_model` pair against a unique temp path; RAII removes all five temp files on success or exception.
- ✅ Added `clone()` declaration to `src/EncoderDecoderModel.hpp` with doc comment.
- ✅ Added `generation_quality_async_threshold = 50` to `ServiceConfig` in `src/Config.hpp`.
- ✅ Added `GENERATION_QUALITY_ASYNC_THRESHOLD` parsing in `src/Config.cpp`.
- ✅ Wired `generation_quality_async_threshold` in `IncrementalTrainer::make_incremental_config()`.
- ✅ Added `GENERATION_QUALITY_ASYNC_THRESHOLD=50` with memory-trade-off comment to `config.conf` and `config-remote.conf`.
- ✅ Created `tests/generation_quality_async_test.cpp` with 10 tests: config defaults, `ServiceConfig` field, `NullMetricsReporter` no-crash, sync path at below-threshold, async path at threshold, 2-epoch thread-join ordering, destructor safety, and sync/async score range validity.
- ✅ Added `generationQualityAsyncTests` target to `tests/CMakeLists.txt`; all 10 tests pass.

Files Modified: `src/ChatbotTrainer.hpp`, `src/ChatbotTrainer.cpp`, `src/EncoderDecoderModel.hpp`, `src/EncoderDecoderModel.cpp`, `src/Config.hpp`, `src/Config.cpp`, `src/IncrementalTrainer.cpp`, `config.conf`, `config-remote.conf`, `tests/CMakeLists.txt`

Files Created: `tests/generation_quality_async_test.cpp`

Related Items: TD-016 (Resolved) — introduced the synchronous `compute_generation_quality_metrics()` path that this item parallelises.

---

### TD-026: Extract GenerationQualityMetrics to Compiled Translation Unit

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 3, 2026 | Metrics / Tooling / Code Quality | Moved all `GenerationQualityEvaluator` method implementations out of the header into a new `src/GenerationQualityMetrics.cpp`; added the `.cpp` to `adai_core`; updated `generationQualityTests` to link `adai_core` |

Summary:
`GenerationQualityMetrics.hpp` was a fully header-only class. All six static method implementations (`evaluate`, `tokenize`, `count_ngrams`, `compute_corpus_bleu`, `compute_corpus_rouge_n`, `compute_corpus_rouge_l`, `lcs_length`) were extracted to `src/GenerationQualityMetrics.cpp`. The header now contains only the `GenerationQualityScore` struct, the `GenerationQualityEvaluator` class declaration with static method signatures, and the three standard-library includes required by the interface (`<map>`, `<string>`, `<vector>`). Implementation-only includes (`<algorithm>`, `<cctype>`, `<cmath>`, `<sstream>`) moved to the `.cpp`. The compiled object is now part of `adai_core`, so any future tooling target (e.g., `adai-eval`) links against it with one line.

Changes Made:

- ✅ Created `src/GenerationQualityMetrics.cpp` with implementations of all seven static methods.
- ✅ Trimmed `src/GenerationQualityMetrics.hpp` to declarations only; removed inline implementations and implementation-only `#include` directives.
- ✅ Added `GenerationQualityMetrics.cpp` to the `adai_core` source list in `src/CMakeLists.txt`.
- ✅ Updated `generationQualityTests` in `tests/CMakeLists.txt` to link `adai_core` (replacing the former header-only setup).
- ✅ `GenerationQualityTests` passes with no regressions.

Files Modified: `src/GenerationQualityMetrics.hpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`

Files Created: `src/GenerationQualityMetrics.cpp`

---

### TD-024: Remove Legacy Standalone ChatbotTrainer Code and Build Target

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 1, 2026 | Training / ChatbotTrainer / Build | Removed nine legacy standalone functions/methods from `ChatbotTrainer` and eliminated no-op `CHATBOT_TRAINER_TEST_BUILD` compile definitions from all CMake targets |

Summary:
`ChatbotTrainer` originally included a full standalone training pipeline (`train(const std::string&)`, `save_checkpoint`, `load_checkpoint`, `finalize_model`, `should_early_stop`, `restore_best_model`, `print_training_summary`, `test_generation`, `print_usage`) and the `TrainingConfig` checkpointing fields that drove them. `IncrementalTrainer` has fully superseded this role; none of the standalone code paths were reachable at runtime. All nine functions, their declarations, the `start_epoch` member variable, four `TrainingConfig` checkpoint fields, and the no-op `CHATBOT_TRAINER_TEST_BUILD` compile definitions were removed, reducing `ChatbotTrainer.cpp` by ~625 lines.

Changes Made:

- ✅ Removed `train(const std::string& output_model_path)` and its body from `src/ChatbotTrainer.cpp`.
- ✅ Removed `print_training_summary(long duration)`, `test_generation()`, and `print_usage()` from `src/ChatbotTrainer.cpp`.
- ✅ Removed `save_checkpoint()`, `load_checkpoint()`, `should_early_stop()`, `restore_best_model()`, `finalize_model()` from `src/ChatbotTrainer.cpp` (~625 lines total removed).
- ✅ Removed `start_epoch` member variable and all references from `src/ChatbotTrainer.cpp`.
- ✅ Removed all corresponding declarations from the public/private sections of `src/ChatbotTrainer.hpp`; removed `save_checkpoints`, `checkpoint_every`, `keep_all_checkpoints`, `resume_from_checkpoint` from `TrainingConfig`.
- ✅ Removed `target_compile_definitions(incremental_trainer PRIVATE CHATBOT_TRAINER_TEST_BUILD)` from `src/CMakeLists.txt`.
- ✅ Removed `CHATBOT_TRAINER_TEST_BUILD` from `chatbottrainerTests`, `incrementaltrainerTests`, and `incrementalTrainerDecouplingTests` in `tests/CMakeLists.txt`.
- ✅ No test cases in `tests/chatbottrainer_test.cpp` referenced the removed code; no test changes needed.
- ✅ `ChatbotTrainerTests` and `IncrementalTrainerDecouplingTests` pass with no regressions.

Files Modified: `src/ChatbotTrainer.cpp`, `src/ChatbotTrainer.hpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`

---

### TD-025: IncrementalTrainer Background Launch with PID Message

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 3, 2026 | Training / IncrementalTrainingTool / CLI | Added `launch_background()` in anonymous namespace; `train`, `retrain`, `resume` branches fork before trainer construction; parent prints structured startup banner and exits 0; child calls `setsid()` and redirects fds; Windows path uses `CreateProcess(DETACHED_PROCESS)` |

Summary:
`incremental_trainer train/retrain/resume` previously blocked the invoking shell for the full duration of training. The fix adds a `launch_background(int argc, char* argv[])` helper (POSIX: `fork()`+`setsid()`; Windows: `CreateProcess(DETACHED_PROCESS)` with `--background-child` sentinel). Each training dispatch branch reads the pending-file count and log path from config before forking, then:

- **Parent** prints a structured startup banner (`[ADAI] Training started in background — PID …`) and returns 0.
- **Child** redirects `stdin`/`stdout`/`stderr` to `/dev/null`, calls `setsid()`, and continues normal training (output flows through `adai::Logger`).

Falls back to foreground execution if `fork()` fails, logging a warning.

Changes Made:

- ✅ Added `launch_background(int argc, char* argv[])` in anonymous namespace at top of `src/IncrementalTrainingTool.cpp` with `#ifndef _WIN32` POSIX path and `#ifdef _WIN32` `CreateProcess` path.
- ✅ Added `--background-child` stripping in the global options parsing loop (Windows sentinel suppression).
- ✅ `train` branch: reads `pending_files.txt` before fork, checks for empty pending list before fork, calls `launch_background()`, prints banner from parent, child continues to `trainer.train_incremental()`.
- ✅ `retrain` branch: counts pending files before fork, calls `launch_background()`, prints banner from parent, child continues to `trainer.train_full_retrain()`.
- ✅ `resume` branch: calls `launch_background()`, prints banner from parent, child continues to `trainer.resume_last_session()`.
- ✅ Created `tests/incremental_trainer_background_test.cpp` covering fork/setsid behavior, PID uniqueness, banner format, log-path inclusion, and Windows `#ifdef` compile-time guard.
- ✅ Added `incrementalTrainerBackgroundTests` target to `tests/CMakeLists.txt`.

Files Modified: `src/IncrementalTrainingTool.cpp`, `tests/CMakeLists.txt`
Files Created: `tests/incremental_trainer_background_test.cpp`

---

### TD-022: Remove Direct Terminal Output from IncrementalTrainer and Dependencies

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 2, 2026 | Training / IncrementalTrainer / BPETokenizer / Transformer / Logging | Converted 120+ `std::cout`/`std::cerr` calls to `adai::Logger` across 4 source files; `display_dashboard()` gated behind `isatty()` |

Summary:
All direct terminal output in the `IncrementalTrainer` dependency graph has been replaced with structured `adai::Logger` calls, eliminating the two-channel output problem and enabling full log control via a single sink in daemon mode.

Changes Made:

- ✅ **`src/IncrementalTrainer.cpp`**: Converted `print_training_summary()`, `print_session_history()`, `print_data_registry()`, and `add_huggingface_dataset()` to `Logger::info()`/`Logger::error()`. Gated `display_dashboard()` behind `isatty(STDOUT_FILENO)` / `_isatty(_fileno(stdout))` — non-TTY path emits a compact `Logger::info()` summary; TTY path retains the full ANSI TUI unchanged. Removed the five `COLOR_*` ANSI macros and "Legacy ANSI codes" comment block.
- ✅ **`src/BPETokenizer.cpp`**: Replaced all `build_vocab()` / `build_bpe_merges()` / `pre_tokenize()` phase-banner and `\r`-overwrite progress `std::cout` calls with `Logger::info()`; removed `std::flush`/`\r` usage. Converted `save_vocab()`/`load_vocab()` confirmation `std::cout` to `Logger::info()`; `std::cerr` file-open error to `Logger::error()`; unknown-token and unknown-special-token `std::cerr` warnings to `Logger::warn()`.
- ✅ **`src/LayerNorm.cpp`**: Replaced `#include <iostream>` with `#include "Logger.hpp"`. Converted `print_config()` `std::cout` calls to `adai::Logger::info()`, `save_weights()`/`load_weights()` confirmations to `adai::Logger::info()`, and `set_gamma()`/`set_beta()` dimension-mismatch `std::cerr` errors to `adai::Logger::error()`.
- ✅ **`src/PositionalEncoding.cpp`**: Replaced `#include <iostream>` with `#include "Logger.hpp"` and added `#include <sstream>`. Converted `print_config()` `std::cout` calls to `adai::Logger::info()`, `forward()` sequence-length `std::cerr` warning to `adai::Logger::warn()`, and `visualize()` table rows built via `std::ostringstream` and emitted through `adai::Logger::info()`.

Files Modified: `src/IncrementalTrainer.cpp`, `src/BPETokenizer.cpp`, `src/LayerNorm.cpp`, `src/PositionalEncoding.cpp`

Verification:

- ✅ `adai_core` builds cleanly (`[100%] Built target adai_core`)
- ✅ `tokenizerErrorHandlingTests` builds and links successfully
- ✅ Only pre-existing `incrementaltrainerTests` failures remain (from TD-021: `metrics_push_enabled`/`metrics_config` member references)

---

### TD-021: IncrementalTrainer × Metrics Service Decoupling

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| May 31, 2026 | Training / Metrics / IncrementalTrainer / ChatbotTrainer | `IMetricsReporter` interface, `MetricsPushClient` background push client, `NullMetricsReporter`, bounded priority queue, session label auto-derivation, multi-session registry with sweep thread |

Description:
`IncrementalTrainer` directly instantiated `TrainingMetricsService` — a server-side class owning a ring buffer, file I/O, and push threads — inside the trainer process. All metrics now flow outward via HTTP only through a lightweight `MetricsPushClient`. `MetricsSessionRegistry` gained the TD-018 §4.8 background sweep thread, and `TrainingMetricsService::to_prometheus()` now emits per-session `{session="key"}` labels eliminating metric collisions across concurrent sessions.

Proposal: `docs/development/archive/incremental-trainer-registry-integration.md`

Changes Made:

- ✅ Create `src/IMetricsReporter.hpp` — abstract reporter interface + `AbnormalSample` struct (moved from `TrainingMetricsService.hpp`) + `NullMetricsReporter` no-op implementation.
- ✅ Create `src/MetricsPushClient.hpp` / `src/MetricsPushClient.cpp` — single background push thread, bounded priority queue (`Sample` events are lossy, `Epoch`/`Session` events are never dropped), 3× retry with back-off, 409-conflict retry loop on `start_session()`. `MetricsPushClient.cpp` added to `adai_core` in `src/CMakeLists.txt`.
- ✅ Replace `void set_metrics_service(TrainingMetricsService*)` with `void set_metrics_reporter(IMetricsReporter*)` in `ChatbotTrainer`; updated all ~15 `metrics_service_->` call sites to `metrics_reporter_->`. Removed `adv_cfg = metrics_service_->get_config()` call; outlier thresholds read directly from `TrainingConfig`.
- ✅ Remove `std::unique_ptr<TrainingMetricsService> metrics_service_` from `IncrementalTrainer`; replaced with `std::unique_ptr<IMetricsReporter> metrics_reporter_` + non-owning `MetricsPushClient* push_client_` alias. All three constructors initialize `NullMetricsReporter`. `MetricsPushClient` is constructed per training run in `train_incremental()` and `train_full_retrain()`. Added `get_metrics_session_key()` accessor. `train_incremental()` and `train_full_retrain()` wrap `MetricsPushClient` creation and `start_session()` in a 3-attempt 409-conflict retry loop (suffix progression: `base_key`, `base_key-2`, `base_key-3`).
- ✅ Replace `IncrementalConfig::enable_metrics_service` + `metrics_push_enabled` + `MetricsServiceConfig metrics_config` with flat fields `metrics_server_url`, `metrics_session_label`, `metrics_push_timeout_ms`; empty URL → `NullMetricsReporter`. Removed `#include "TrainingMetricsService.hpp"` from `IncrementalTrainer.hpp`.
- ✅ Move `loss_outlier_z_threshold`, `grad_norm_outlier_threshold`, `max_abnormal_samples` from `MetricsServiceConfig` into `TrainingConfig` (`src/ChatbotTrainer.hpp`). Added `abnormal_sample_count_` member to `ChatbotTrainer`; `flag_abnormal_sample()` calls are now capped at `config.max_abnormal_samples` per training run. Added all three fields to `ServiceConfig` (`src/Config.hpp`) and wired them through `make_incremental_config()`.
- ✅ Implement session label auto-derivation (`"#{id}: {stem} ({host}, {date})"`): `derive_metrics_session_label()` and `build_config_snapshot()` added to the anonymous namespace in `src/IncrementalTrainer.cpp`. Both `train_incremental()` and `train_full_retrain()` now pass `label` (user value or auto-derived) and `snapshot` JSON to `start_session()`. Auto-derivation uses `fs::path(model_path_).stem()`, `detect_hostname_fragment()`, and `strftime`; snapshot captures `d_model`, `heads`, `d_ff`, `enc_layers`, `dec_layers`, `lr`, `batch`, `grad_accum`.
- ✅ Add `METRICS_SESSION_LABEL` to `config.conf` and `config-remote.conf`; remove `METRICS_PUSH_ENABLED` (empty `METRICS_SERVER_URL` is now sufficient to disable push). Added `metrics_session_label` field to `ServiceConfig` (`src/Config.hpp`) and parsed from config file and env var in `src/Config.cpp`. Removed `ServiceConfig::metrics_push_enabled`; `make_incremental_config()` now maps `svc.metrics_server_url` directly.
- ✅ Add background sweep thread to `MetricsSessionRegistry` (TD-018 §4.8): `sweep_thread_`, `stop_sweep_` atomic, `sweep_cv_` + `sweep_mutex_` — constructor gains 4th param `sweep_interval_seconds` (default: 60); thread started when > 0; destructor sets `stop_sweep_ = true`, notifies, joins.
- ✅ Add `label` and `config_snapshot` fields to `MetricsSessionSummary` and `SessionEntry` (`src/MetricsSessionRegistry.hpp`); update `list_sessions()` to populate them.
- ✅ Add `label` and `config_snapshot` params to `TrainingMetricsService::start_session()`; store as member vars (`label_`, `config_snapshot_`); add to `TrainingMetricsSnapshot`; `list_sessions()` reads them from snapshot. Add optional `session_key` param to `to_prometheus()` and `to_prometheus_internal()`; emits `{session="key"}` labels on each metric line when non-empty.
- ✅ Pass `sweep_interval_seconds` (default 60) as 4th argument to `MetricsSessionRegistry` constructor in `src/TrainingMetricsAPIServer.cpp`; added `--sweep-interval-seconds` CLI flag to `ServerConfig`.
- ✅ Add `GET /api/metrics/prometheus/aggregate` endpoint to `TrainingMetricsAPI` that concatenates per-session labelled Prometheus output; `handle_prometheus_metrics()` now passes `session_key` to `to_prometheus()`; `handle_post_session_start()` parses and forwards `label` and `config` fields from POST body.
- ✅ Write `tests/incremental_trainer_registry_test.cpp` (new, 11 tests): `NullMetricsReporter` all-methods-no-crash, `MetricsPushClient` construction/offline/201/409/retry-suffix, `IncrementalConfig` metrics fields, `IncrementalTrainer::get_metrics_session_key()` initially empty.
- ✅ Write `tests/metrics_push_client_test.cpp` (10 tests): queue overflow policy, retry/back-off, 409 no-retry, shutdown drain, `start_session` body, destructor safety.
- ✅ Add 9 tests to `tests/metrics_session_registry_test.cpp`: sweep-thread eviction, `label`/`config_snapshot` propagation, `get_session()` optional semantics.
- ✅ Update `docs/development/TRAINING_METRICS_API.md`: multi-session API catalog, Prometheus aggregate endpoint, `POST /start` body schema, `--sweep-interval-seconds` CLI option.

Files Created: `src/IMetricsReporter.hpp`, `src/MetricsPushClient.hpp`, `src/MetricsPushClient.cpp`, `tests/incremental_trainer_registry_test.cpp`, `tests/metrics_push_client_test.cpp`

Files Modified: `src/IncrementalTrainer.hpp/.cpp`, `src/ChatbotTrainer.hpp/.cpp`, `src/TrainingMetricsService.hpp/.cpp`, `src/TrainingMetricsAPI.hpp/.cpp`, `src/TrainingMetricsAPIServer.cpp`, `src/MetricsSessionRegistry.hpp`, `src/Config.hpp/.cpp`, `src/CMakeLists.txt`, `config.conf`, `config-remote.conf`, `tests/metrics_session_registry_test.cpp`, `tests/CMakeLists.txt`, `docs/development/TRAINING_METRICS_API.md`

---

### TD-019: Stale Metrics Detection and Liveness Accuracy

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| May 31, 2026 | Training / Metrics / API / Config / Dashboard | Staleness threshold config, snapshot stale fields, `get_current_snapshot()` fix, `to_json()` stale fields, API staleness fields, dashboard stale badge, `staleDetectionTests` |

Description:
Metrics endpoints reported `is_training=true` long after ingest stopped because `get_current_snapshot()` overwrote `last_update_time` with the current clock on every read, masking the true last-ingest time. The dashboard used browser-local time for "Last updated", showing a false freshness indicator during trainer/network failures.

Changes Made:

- ✅ **Config** — Added `metrics_staleness_threshold_seconds` (default: 60) to `ServiceConfig` (`src/Config.hpp`), parsed from config file and env var in `src/Config.cpp`. Wired through `MetricsServiceConfig.staleness_threshold_seconds` in `src/IncrementalTrainer.cpp`. Added key to `config.conf` and `config-remote.conf`.
- ✅ **Snapshot fields** — Added `is_stale`, `seconds_since_last_update`, `effective_is_training` to `TrainingMetricsSnapshot` in `src/TrainingMetricsService.hpp`.
- ✅ **Root cause fix** — Removed `snapshot.last_update_time = std::chrono::system_clock::now()` from `get_current_snapshot()` in `src/TrainingMetricsService.cpp`; staleness fields now computed from preserved ingest timestamp. Same logic added to `to_json()` (which runs independently to avoid lock re-entry).
- ✅ **JSON output** — `to_json()` now appends `is_stale`, `seconds_since_last_update`, `effective_is_training` fields to all snapshot JSON responses.
- ✅ **API endpoints** — `handle_session_status()` adds the three stale fields to its JSON response; `handle_health_check()` uses `effective_is_training` for per-session liveness and exposes `any_stale` in `src/TrainingMetricsAPI.cpp`.
- ✅ **Dashboard** — Added `.stale-badge` CSS, stale badge `<span>` in HTML header, updated `updateDashboard()` to use server-provided `seconds_since_last_update`, show "X seconds ago (stale)" label, and use `effective_is_training` for status colour in `dashboard.html`.
- ✅ **Tests** — Created `tests/stale_detection_test.cpp` with 16 tests across 6 suites (`StaleDetectionIngestTimestamp`, `StaleDetectionSecondsAgo`, `StaleDetectionIsStale`, `StaleDetectionEffectiveIsTraining`, `StaleDetectionJson`, `StaleDetectionConfig`). Registered as `staleDetectionTests` CMake target in `tests/CMakeLists.txt`.

Verification:

- ✅ All 16 `staleDetectionTests` pass in `build-gpu-clang`
- ✅ No regressions in `adai_core` build

---

### TD-018: Multi-Instance Training Metrics Service

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| May 31, 2026 | Training / Metrics / API / Config | `MetricsSessionRegistry`, session-scoped routes, trainer session-key wiring, `GlobalMetricsService` proxy, config keys, API docs |

Description:
The metrics stack assumed a single active training session and used shared file paths and flat API routes. Running multiple trainers against one metrics API server could overwrite metrics, cross-contaminate session state, and make dashboards unreliable. All 10 phases of the proposal (`docs/development/archive/multi-instance-metrics-service.md`) are now complete.

Changes Made:

- ✅ **Phase 1** — `TrainingMetricsService` now prepares configured parent directories for all output files, enabling fully caller-supplied per-instance paths. Regression test added in `tests/training_metrics_service_resume_test.cpp`.
- ✅ **Phase 2** — Created `src/MetricsSessionRegistry.hpp`: owns `unordered_map<string, shared_ptr<TrainingMetricsService>>` under a `shared_mutex`, implements `create_or_get_session()`, `get_session()`, `list_sessions()`, and TTL-based `evict_completed_sessions()`. Enforces `max_live_sessions` cap (default 16). Derives per-session file paths (`{key}_metrics.jsonl`, etc.) while preserving legacy paths for `"0-default"`.
- ✅ **Phase 3** — `TrainingMetricsAPI` rewritten to accept `MetricsSessionRegistry*`. All routes moved to `/api/sessions/{key}/...` prefix. Added `GET /api/sessions` (session index) and `GET /api/metrics/aggregate` (live cross-session view). Legacy flat routes preserved as `"0-default"` aliases emitting `Deprecation: true` / `Link:` headers.
- ✅ **Phase 4** — `TrainingMetricsAPIServer` constructs `MetricsSessionRegistry` from `Config` and injects it into `TrainingMetricsAPI`; removed direct construction of a single `TrainingMetricsService`.
- ✅ **Phase 5** — `src/Config.hpp/.cpp`: added `metrics_session_key`, `metrics_max_live_sessions` (16), `metrics_completed_ttl_seconds` (3600), `metrics_sweep_interval_seconds` (60); parsed from both config file and environment variable overrides.
- ✅ **Phase 6** — `src/IncrementalTrainer.cpp`: added `sanitize_session_key()`, `derive_metrics_session_key()` (auto-derives `{id}-{hostname}{pid%10000}` when key is empty), `build_metrics_session_push_base()`, and updated `make_incremental_config()` to set the session-scoped push URL (`{METRICS_SERVER_URL}/api/sessions/{key}`).
- ✅ **Phase 7** — `GlobalMetricsService` updated to proxy through the `"0-default"` slot of a `MetricsSessionRegistry` singleton, preserving the existing `instance().start_session(id, ...)` call-site API.
- ✅ **Phase 8** — Added multi-session tests in `tests/metrics_session_registry_test.cpp` (`ConcurrentSessionCreationIsThreadSafe`, `ConcurrentReadsAndWritesDoNotDeadlock`, `ConcurrentSessionsHaveIsolatedData`) and `tests/training_metrics_api_routes_test.cpp` (`AggregateEndpointCountsAllLiveSessions`, `SessionStartReturnsConflictForActiveSessionKey`, `SessionStartReturns503WhenRegistryIsFull`).
- ✅ **Phase 9** — `config.conf` and `config-remote.conf` updated with all four new keys and comments.
- ✅ **Phase 10** — `docs/development/TRAINING_METRICS_API.md` updated with the full session-scoped route catalog, `GET /api/sessions` and `GET /api/metrics/aggregate` endpoint documentation, backwards-compatibility alias table, and session key configuration reference.

Files Modified:

- `src/MetricsSessionRegistry.hpp` (new)
- `src/TrainingMetricsAPI.hpp` / `src/TrainingMetricsAPI.cpp`
- `src/TrainingMetricsAPIServer.cpp`
- `src/Config.hpp` / `src/Config.cpp`
- `src/IncrementalTrainer.cpp`
- `src/TrainingMetricsService.hpp` / `src/TrainingMetricsService.cpp` — `GlobalMetricsService`
  ended up as a class inside this file rather than its own `GlobalMetricsService.hpp` as
  originally planned; corrected here September 7, 2026 after finding a stale TODO in that class
  still describing this item as pending
- `tests/metrics_session_registry_test.cpp`
- `tests/training_metrics_api_routes_test.cpp`
- `tests/training_metrics_service_resume_test.cpp`
- `config.conf` / `config-remote.conf`
- `docs/development/TRAINING_METRICS_API.md`

Verification:

- ✅ All `metricsSessionRegistryTests` pass (7 tests including 3 new concurrent-session tests)
- ✅ All `trainingMetricsApiRoutesTests` pass (9 tests including 409/503/aggregate coverage)
- ✅ All `incrementaltrainerTests` pass (40 tests) — no regressions
- ✅ Production build (`build-gpu-clang`) and ASAN build (`build-asan`) both clean

---

### TD-003: GPU Memory Management Optimization

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| May 3, 2026 | GPU / Performance | `GPUMatrix` class with persistent device memory, `Matrix::to_gpu()` / `Matrix::from_gpu()` |

Description:
Implemented persistent GPU-resident matrix storage that eliminates per-operation host↔device transfers when chaining multiple GPU operations.  Previously every `multiply_gpu()` / `add_gpu()` etc. call allocated device memory, transferred data, executed the kernel, copied the result back, and freed device memory — incurring PCIe round-trip cost for every single op.  With `GPUMatrix`, data is uploaded once, all intermediate computations remain on-device, and only the final result is downloaded.

Changes Made:

- ✅ Added `#include "GPUUtils.hpp"` to `src/gpu/MatrixGPU.hpp` so the class can use `GPUMemory<float>`, `GPUManager`, and `CUDA_CHECK`.
- ✅ Created `adai::gpu::GPUMatrix` class in `src/gpu/MatrixGPU.hpp` (move-only, RAII via `GPUMemory<float>`):
  - `GPUMatrix(int rows, int cols)` — allocate on-device
  - `upload(const float*, int)` / `download(float*, int)` — blocking host↔device transfers
  - `copy()` — async device-to-device clone
  - `operator*` — cuBLAS SGEMM matrix multiply (stays on device)
  - `operator+` — element-wise add (stays on device)
  - `operator-` — element-wise subtract (stays on device)
  - `scale(float)` — scalar multiply (stays on device)
  - `hadamard(const GPUMatrix&)` — element-wise multiply (stays on device)
  - `transpose()` — shared-memory transpose kernel (stays on device)
  - `apply_activation_inplace(ActivationType)` — in-place activation (stays on device)
  - `sum()` — parallel reduction, returns scalar to host
- ✅ Added `Matrix::to_gpu() const` to `src/Matrix.hpp` / `src/Matrix.cpp` — uploads CPU matrix to a new `GPUMatrix`.
- ✅ Added `static Matrix Matrix::from_gpu(const adai::gpu::GPUMatrix&)` — downloads a `GPUMatrix` back to a CPU `Matrix`.

Files Modified:

- `src/gpu/MatrixGPU.hpp` — `#include "GPUUtils.hpp"` + `GPUMatrix` class
- `src/Matrix.hpp` — `to_gpu()` and `from_gpu()` declarations
- `src/Matrix.cpp` — `to_gpu()` and `from_gpu()` implementations

Usage Example:

```cpp
// Before (TD-003): 3 upload/download round-trips
Matrix C = A.multiply_gpu(B);   // upload A,B → kernel → download C
Matrix D = C.add_gpu(A);        // upload C,A → kernel → download D
Matrix E = D.transpose_gpu();   // upload D   → kernel → download E

// After (TD-003 resolved): 3 uploads, 1 download, 3 on-device ops
auto A_gpu = A.to_gpu();
auto B_gpu = B.to_gpu();
auto C_gpu = A_gpu * B_gpu;           // on-device matmul
auto D_gpu = C_gpu + A_gpu;           // on-device add
auto E_gpu = D_gpu.transpose();       // on-device transpose
Matrix E = Matrix::from_gpu(E_gpu);   // single download
```

Verification:

- ✅ `adai_core` builds clean with `-DENABLE_GPU=ON`
- ✅ CPU-only builds unaffected (entire class inside `#ifdef ADAI_ENABLE_GPU`)
- ✅ All existing `matrixTests` and `matrixSIMDTests` pass — no regressions

---

### TD-017: Adaptive Gradient Clipping

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| April 19, 2026 | Training / ChatbotTrainer / Config / Metrics / Dashboard | EMA-based adaptive clip threshold with spike suppression and warmup |

Description:
Replaced static `GRADIENT_CLIP` scalar (0.5) that was clipping every gradient step in Session 5 (per-step norms 2–23, threshold 4–46× too small). Implemented EMA-based adaptive threshold: `ema ← α×raw_norm + (1−α)×ema`, `effective_clip = clamp(ema × headroom, min, max)` with spike suppression (skip EMA update when `norm > spike_k × ema`) and warmup period.

Changes Made:

- ✅ Added 7 new `ServiceConfig` fields in `src/Config.hpp`: `adaptive_gradient_clip`, `gradient_clip_min`, `gradient_clip_max`, `gradient_clip_ema_decay`, `gradient_clip_headroom`, `gradient_clip_warmup_steps`, `gradient_clip_spike_k`
- ✅ Implemented full config parsing for 7 new keys in `src/Config.cpp` (file key-value reader + env-var override block)
- ✅ Mirrored 7 fields in `ChatbotTrainerConfig` (`src/ChatbotTrainer.hpp`)
- ✅ Mapped new fields in `IncrementalTrainer::make_incremental_config()` (`src/IncrementalTrainer.cpp`)
- ✅ Replaced fixed-clip call in `ChatbotTrainer::train_epoch()` with EMA + clamp adaptive logic; legacy path preserved when `adaptive_gradient_clip=false`
- ✅ Added `current_adaptive_clip_threshold`, `current_adaptive_clip_spikes`, and `epoch_adaptive_clip_thresholds` to `TrainingMetricsSnapshot`; added `update_adaptive_clip_metrics()` and `update_adaptive_clip_epoch()` to `TrainingMetricsService`
- ✅ Extended `to_json()` in `TrainingMetricsService` to emit both new snapshot fields
- ✅ Added `epoch_adaptive_clip_thresholds` array to `TrainingMetricsAPI::handle_epoch_metrics()`
- ✅ Added **Gradient Clipping** panel to `dashboard.html`: dual-line chart (adaptive threshold + raw gradient norm), two metric cards (threshold + spike count)
- ✅ Written unit tests in `tests/adaptive_clipping_test.cpp` (defaults, update, epoch accumulation, JSON emission)

---

### TD-013: Advanced Training Metrics and Outlier Detection

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 14, 2026 | Training / Service | Welford outlier detection, advanced epoch metrics, and activation/entropy/BLEU/padding sub-items |

Description:
Extended the training pipeline with comprehensive observability beyond basic loss and learning rate: throughput ratios, model optimization health signals, generative validation scoring (BLEU/ROUGE), and automated flagging of training samples with abnormally large loss or gradient variations.

Changes Made:

- ✅ Updated `TrainingMetricsSnapshot` to support additional floats (`gradient_variance`, `compute_time_ratio`, `weight_update_ratio`).
- ✅ Added `AbnormalSample` struct with `epoch`, `sample_id`, `input_text`, `target_text`, `loss`, `grad_norm`, `reason`.
- ✅ Added outlier config to `MetricsServiceConfig`: `loss_outlier_z_threshold`, `grad_norm_outlier_threshold`, `max_abnormal_samples`, `abnormal_samples_file`.
- ✅ Implemented `flag_abnormal_sample()`, `get_abnormal_samples()`, `update_advanced_epoch_metrics()` on `TrainingMetricsService`.
- ✅ Implemented `persist_abnormal_samples()` — writes flagged samples to `abnormal_samples.json` on every flag event.
- ✅ Added `Optimizer::get_weight_norm()` to compute the L2 norm of all weight parameters.
- ✅ Instrumented `ChatbotTrainer::train_epoch()` with:
  - Welford online algorithm for per-step gradient norm variance.
  - Per-step loss z-score outlier guard (triggers after ≥10 samples).
  - Gradient-norm absolute outlier guard (`grad_norm_outlier_threshold`).
  - Compute-time ratio tracking (forward+backward nanoseconds / epoch wall nanoseconds).
  - Weight-update ratio computation: `(lr × ||g||₂) /||w||₂` averaged per epoch.
- ✅ Added REST endpoint `GET /api/metrics/abnormal` to `TrainingMetricsAPI`.
- ✅ Activation saturation tracking (implemented April 11, 2026 — see TD-013b).
- ✅ Attention entropy (implemented April 11, 2026 — see TD-013b).
- ✅ BLEU/ROUGE generation quality scores (implemented April 11, 2026 — see TD-016).
- ✅ Batch padding efficiency (implemented April 11, 2026 — see TD-013b).

---

### TD-013b: Batch Padding Efficiency Tracking

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| April 11, 2026 | Training / Service / Metrics | Per-window efficiency computation in `train_epoch()` + `TrainingMetricsService` integration |

Summary:
Implemented batch padding efficiency — the theoretical fraction of non-padding tokens per gradient-accumulation window — as the final deferred item from TD-013. Each gradient-accumulation window's input/target sequence lengths are collected and compared against their padded extent `(max_input_len + max_target_len) × window_size`. The epoch average is reported to `TrainingMetricsService`, exposed via REST, and displayed on the dashboard. When `gradient_accumulation_steps == 1` efficiency is trivially 1.0; with larger accumulation windows the metric reveals sequence-length mismatch within virtual batches.

Changes Made:

- ✅ Added `current_padding_efficiency` (float, default -1 = not computed) and `epoch_padding_efficiencies` (vector) to `TrainingMetricsSnapshot`.
- ✅ Added `update_padding_efficiency(float)` to `TrainingMetricsService` — mutex-safe snapshot update.
- ✅ `TrainingMetricsService::end_epoch()` now pushes `current_padding_efficiency` into `epoch_padding_efficiencies`.
- ✅ `TrainingMetricsService::to_json()` now emits `"current_padding_efficiency"`.
- ✅ Instrumented `ChatbotTrainer::train_epoch()`:
  - Per-window accumulators: `pad_win_actual`, `pad_win_max_input`, `pad_win_max_target`, `pad_win_count` — reset at each accumulation cycle start and collected per sample.
  - At every `should_update` checkpoint: `eff = pad_win_actual / ((max_in + max_tgt) * window_size)`, accumulated into `pad_eff_sum / pad_eff_count`.
  - At epoch end: `avg_padding_efficiency` reported via `metrics_service_->update_padding_efficiency()`.
- ✅ Added REST endpoint `GET /api/metrics/padding-efficiency` to `TrainingMetricsAPI` — returns `current_padding_efficiency` and `epoch_padding_efficiencies` as JSON.
- ✅ `dashboard.html`: added "Padding Eff." metric card; `metricsHistory.paddingEfficiencies` array; `addToHistory()` and `updateDashboard()` wired up (null entries when score is -1 for clean gap rendering).
- ✅ Created `tests/padding_efficiency_test.cpp` — 16 tests across 2 suites (service-level and pure arithmetic). All 16 pass.
- ✅ Added `paddingEfficiencyTests` target to `tests/CMakeLists.txt`.

Files Modified:

- `src/TrainingMetricsService.hpp` — snapshot fields, method declaration
- `src/TrainingMetricsService.cpp` — `update_padding_efficiency()`, `end_epoch()` history push, `to_json()` output
- `src/TrainingMetricsAPI.hpp` — endpoint comment, `handle_padding_efficiency_metrics()` declaration
- `src/TrainingMetricsAPI.cpp` — route + handler implementation
- `src/ChatbotTrainer.cpp` — per-window accumulators + epoch-end push
- `dashboard.html` — "Padding Eff." metric card and JS data flow
- `tests/CMakeLists.txt` — `paddingEfficiencyTests` target

Files Created:

- `tests/padding_efficiency_test.cpp` — 16 new tests

Verification:

- ✅ All 16 `paddingEfficiencyTests` pass
- ✅ `adai_core` builds clean
- ✅ `chatbottrainerTests` builds clean — no regressions

---

### TD-016: BLEU/ROUGE Generation Quality Scoring

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| April 11, 2026 | Training / Service / Metrics | Header-only `GenerationQualityEvaluator` with corpus BLEU-1/2/4 and macro-averaged ROUGE-1/2/L F1 |

Summary:
Implemented end-to-end BLEU and ROUGE generation quality scoring with no external library dependencies. Scoring is opt-in (`enable_generation_quality_metrics = false` by default) and runs on a configurable sub-sample of the validation set during each `validate()` call, calling `model->generate_response()` in eval mode.

Changes Made:

- ✅ Created `src/GenerationQualityMetrics.hpp` — header-only `GenerationQualityEvaluator` providing:
  - `tokenize()`: whitespace split + lowercase + strip leading/trailing punctuation.
  - Corpus BLEU-1, BLEU-2, BLEU-4 with clipped modified precision (Lin & Och 2004 add-1 smoothing) and corpus-level brevity penalty.
  - Macro-averaged ROUGE-1 and ROUGE-2 F1 (precision × recall / (precision + recall) per sentence).
  - Macro-averaged ROUGE-L F1 via rolling 2-row DP LCS (O(m·n) time, O(n) space).
  - `GenerationQualityScore` struct: `bleu1`, `bleu2`, `bleu4`, `rouge1`, `rouge2`, `rougeL` (all default -1.0 = not computed).
- ✅ Extended `TrainingMetricsSnapshot` with `current_bleu4`, `current_rouge1`, `current_rouge2`, `current_rougeL` and per-epoch history vectors `epoch_bleu4`, `epoch_rouge1`, `epoch_rouge2`, `epoch_rougeL`.
- ✅ Added `MetricsServiceConfig::enable_generation_quality` (default `false`) and `generation_quality_sample_size` (default `10`).
- ✅ Implemented `TrainingMetricsService::update_generation_quality_metrics(bleu4, rouge1, rouge2, rougeL)` — mutex-safe update + optional HTTP push to `/api/metrics/generation-quality`.
- ✅ `TrainingMetricsService::end_epoch()` now pushes all four scores into per-epoch history vectors.
- ✅ `TrainingMetricsService::to_json()` now emits `current_bleu4`, `current_rouge1`, `current_rouge2`, `current_rougeL`.
- ✅ Added `TrainingConfig` fields: `enable_generation_quality_metrics` (default `false`), `generation_quality_sample_size` (default `10`), `generation_quality_max_tokens` (default `50`).
- ✅ `ChatbotTrainer::validate()` now calls `compute_generation_quality_metrics()` after updating validation-loss metrics.
- ✅ Implemented `ChatbotTrainer::compute_generation_quality_metrics()` — samples up to `generation_quality_sample_size` pairs from the front of the validation set, calls `model->generate_response()` in eval mode for each, then delegates scoring to `GenerationQualityEvaluator::evaluate()` and pushes results to `metrics_service_`.
- ✅ Added REST endpoint `GET /api/metrics/generation-quality` to `TrainingMetricsAPI` — returns current and per-epoch BLEU/ROUGE history as JSON.
- ✅ `dashboard.html`: added "BLEU-4" and "ROUGE-L" metric cards; `metricsHistory` extended with `bleu4` / `rougeL` arrays; `addToHistory()` and `updateDashboard()` wired up (null entries used when score is -1 so chart gaps are rendered cleanly).
- ✅ Created `tests/generation_quality_test.cpp` — 23 tests across 5 suites (Tokenizer, BLEU, ROUGE, ROUGE-L, Edge). All 23 pass.
- ✅ Added `generationQualityTests` target to `tests/CMakeLists.txt`.

Files Created:

- `src/GenerationQualityMetrics.hpp` — New header-only BLEU/ROUGE library
- `tests/generation_quality_test.cpp` — 23 new tests

Files Modified:

- `src/TrainingMetricsService.hpp` — snapshot fields, config options, method declaration
- `src/TrainingMetricsService.cpp` — `update_generation_quality_metrics()`, `end_epoch()` history push, `to_json()` output
- `src/TrainingMetricsAPI.hpp` — `handle_generation_quality_metrics()` declaration + endpoint comment
- `src/TrainingMetricsAPI.cpp` — route + handler implementation
- `src/ChatbotTrainer.hpp` — `TrainingConfig` fields, `compute_generation_quality_metrics()` method
- `src/ChatbotTrainer.cpp` — `#include "GenerationQualityMetrics.hpp"`, `compute_generation_quality_metrics()` implementation, `validate()` call-site
- `dashboard.html` — BLEU-4 / ROUGE-L metric cards and JS data flow
- `tests/CMakeLists.txt` — `generationQualityTests` target

Verification:

- ✅ All 23 `generationQualityTests` pass
- ✅ `adai_core` (TrainingMetricsService changes) builds clean
- ✅ `chatbottrainerTests` (ChatbotTrainer changes) builds clean

---

### TD-007: Matrix Operations SIMD Acceleration

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| April 11, 2026 | Performance / Matrix Operations | AVX2/FMA + NEON intrinsics and BLAS SGEMM integration across all Matrix operations |

Description:
Added explicit SIMD intrinsics (AVX2/FMA and ARM NEON) and optional BLAS (SGEMM) integration to all performance-critical `Matrix` operations, delivering 2-5x speedup over the previous scalar/OpenMP-only paths for compute-intensive workloads.

Changes Made:

- ✅ Created `src/MatrixSIMD.hpp` — compile-time SIMD capability macros (`ADAI_SIMD_AVX2`, `ADAI_SIMD_FMA`, `ADAI_SIMD_NEON`), runtime `has_avx2()` / `has_fma()` CPUID detection, and `hsum256()` / `hsum128()` horizontal-reduction helpers.
- ✅ Added `ENABLE_BLAS=ON` and `ENABLE_SIMD=ON` CMake options to top-level `CMakeLists.txt`.
- ✅ Updated `src/CMakeLists.txt`:
  - BLAS detection block: `find_package(BLAS)` + `find_path(CBLAS_INCLUDE_DIR cblas.h)` → sets `ADAI_ENABLE_BLAS` and links `${BLAS_LIBRARIES}` when both are present.
  - SIMD flags block: `check_cxx_compiler_flag(-mavx2/-mfma)` → adds target-specific compile options to `adai_core` for non-Release builds (Release already gets them from `-march=native`).
- ✅ `Matrix::operator*` — added BLAS SGEMM path (pack→`cblas_sgemm`→unpack) for matrices ≥ 256 in all dimensions; added AVX2/FMA `ikj`-order inner loop (processes 8 floats per FMA instruction) and NEON `vfmaq_f32` path; OpenMP and scalar paths preserved as fallbacks.
- ✅ `Matrix::operator+` — AVX2 `_mm256_add_ps` / NEON `vaddq_f32` row-wise loop replacing the collapse-2 OpenMP loop; scalar remainder handles non-multiple-of-8 tails.
- ✅ `Matrix::operator-` — AVX2 `_mm256_sub_ps` / NEON `vsubq_f32` row-wise loop.
- ✅ `Matrix::scale()` — AVX2 `_mm256_mul_ps` / NEON `vmulq_f32` row-wise loop.
- ✅ `Matrix::hadamard()` — AVX2 `_mm256_mul_ps` / NEON `vmulq_f32` element-wise loop.
- ✅ `Matrix::apply_gradients()` — AVX2+FMA `_mm256_fmadd_ps(−lr, g, w)` / NEON `vfmsq_n_f32` fused multiply-subtract; falls back to separate mul+sub on AVX2 without FMA.
- ✅ `Matrix::sum()` — AVX2 horizontal-reduction via `hsum256(_mm256)` accumulator / NEON `vaddvq_f32`; correct for any number of columns.
- ✅ All SIMD paths include an `#ifdef ADAI_ENABLE_OPENMP` outer `#pragma omp parallel for` so thread-level and data-level parallelism compose.
- ✅ Removed all TD-007 TODO comments from `src/Matrix.cpp`, `src/CMakeLists.txt`, and `CMakeLists.txt`.
- ✅ Created `tests/matrix_simd_test.cpp` — 91 tests across 11 test suites: parameterised column widths (1–256 including all non-multiples-of-8), shapes, BLAS large-matrix path, CPU feature detection, edge cases, and numerical stability. All 91 tests pass.
- ✅ Added `matrixSIMDTests` target to `tests/CMakeLists.txt`.

Files Modified:

- `src/MatrixSIMD.hpp` — New file
- `src/Matrix.hpp` — `#include "MatrixSIMD.hpp"`
- `src/Matrix.cpp` — BLAS + AVX2/FMA + NEON code paths for all six operations
- `src/CMakeLists.txt` — BLAS detection + SIMD compiler flags; removed TD-007 TODOs
- `CMakeLists.txt` — `ENABLE_BLAS` and `ENABLE_SIMD` options; removed TD-007 TODOs
- `tests/matrix_simd_test.cpp` — New 91-test SIMD test suite
- `tests/CMakeLists.txt` — `matrixSIMDTests` target

Verification:

- ✅ `cmake .. -DENABLE_SIMD=ON -DENABLE_BLAS=ON` reports "SIMD: AVX2 + FMA intrinsics enabled for adai_core"
- ✅ All 91 `matrixSIMDTests` pass (AVX2+FMA active at runtime)
- ✅ All 58 existing `matrixTests` pass — no regressions

---

### TD-015: Validation Metrics Integration

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 14, 2026 | Training / Service | Extended `TrainingMetricsService` with validation perplexity and accuracy tracking |

Description:
Previously, the training pipeline tracked metrics during training passes but lacked granular, systematized metric tracking for the validation phase. Validation metrics are now fully integrated into `TrainingMetricsService` and the dashboard, providing better insights into model generalization and enabling early detection of over-fitting. Proposal: `docs/development/archive/VALIDATION_METRICS_PROPOSAL.md`.

Changes Made:

- ✅ Extended `TrainingMetricsSnapshot` with `current_validation_perplexity`, `current_validation_accuracy`, `epoch_validation_perplexities`, and `epoch_validation_accuracies`.
- ✅ Updated `update_validation_metrics()` signature to accept `validation_loss`, `validation_accuracy` (default `-1.0`), and `validation_perplexity` (default `0` = auto-derived from loss).
- ✅ `end_epoch()` now copies the current validation perplexity and accuracy into the per-epoch history vectors.
- ✅ `to_json()` now emits `current_validation_perplexity` and `current_validation_accuracy` in the API response.
- ✅ `ChatbotTrainer::validate()` now calls `update_validation_metrics(val_loss, -1.0f, val_perplexity)` — passes computed perplexity.
- ✅ `TrainingMetricsAPI::handle_post_validation_metrics()` parses `validation_accuracy` and `validation_perplexity` from the POST body.
- ✅ `handle_epoch_metrics()` returns `epoch_validation_perplexities` and `epoch_validation_accuracies` arrays.
- ✅ `dashboard.html`: added "Val Perplexity" and "Val Accuracy" metric cards; perplexity chart now shows both training and validation perplexity curves with distinct colors; `metricsHistory` and `addToHistory()` extended to track `validationPerplexities`.

---

### TD-012: Increase Test Coverage

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 1, 2026 | Code Quality / Testing | Comprehensive test suite implementation for all components |

Summary:
Achieved ~100% test coverage of all testable code components (excluding GUI and legacy entry points). Implemented dedicated test suites for all major subsystems including Config, Logger, Trainer, and Inference Engines.

Changes Made:

1. ✅ Created test suites for 13+ major components:
   - Config, Logger, IncrementalTrainer, LLMEncoder, DocumentStore
   - RAGInference, BatchProcessor, ParallelDataLoader
   - IntegratedInferenceEngine, PipelineInferenceEngine
   - BatchedInferenceEngine, SpeculativeDecoding, VocabBuilder
2. ✅ Achieved >95% LoC coverage target
3. ✅ Added edge case testing and error handling verification

Files Modified:

- `tests/*` (New test files for all components)
- `tests/CMakeLists.txt`

Coverage Stats:

- Total Testable Lines: ~25,000 LOC
- Tested Lines: ~24,784 LOC
- Coverage: ~99%

---

### TD-011: File Rotation and Management

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 1, 2026 | Logging and Observability | Integrated spdlog rotating file sink |

Summary:
Implemented production-grade log rotation to prevent infinite log growth. Logs are now automatically rotated based on size and count configuration.

Changes Made:

1. ✅ Added rotating file sink to spdlog (rotating_file_sink_mt)
2. ✅ Configured max file size and rotation count
3. ✅ Support compression flag (requires external tool)
4. ✅ Added `LOG_FILE_PATH`, `LOG_MAX_SIZE_MB`, `LOG_MAX_FILES`, `LOG_COMPRESS` configuration options
5. ✅ Dual-sink pattern (console + rotating file)
6. ✅ Thread-safe multi-threaded sink
7. ✅ Validation for log rotation parameters
8. ✅ Integration with configuration hot-reload

Files Modified:

- `src/Logger.cpp`
- `src/Logger.hpp`
- `src/Config.cpp`
- `src/Config.hpp`
- `src/ChatbotAPIServer.cpp`

**Documentation:** `LOG_FILE_ROTATION_COMPLETE.md`

---

### TD-010: Configuration Hot-Reloading

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 1, 2026 | Configuration and Service Management | Implemented SIGHUP handler and thread-safe config updates |

Summary:
Implemented zero-downtime configuration reloading using SIGHUP signals. This allows changing log levels, timeouts, and other parameters without restarting the service.

Changes Made:

1. ✅ Implemented SIGHUP handler to trigger config reload
2. ✅ Added thread-safe configuration updates with mutex protection
3. ✅ Validate new configuration before applying
4. ✅ Log configuration changes with timestamps

Files Modified:

- `src/Config.cpp`
- `src/ChatbotAPIServer.cpp`
- `src/Logger.cpp`

**Documentation:** `CONFIG_HOT_RELOAD_COMPLETE.md`

---

### TD-009: Incremental Trainer Dashboard and Structured Logging

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 2, 2026 | Training / IncrementalTrainer / Observability | Full implementation of Logger integration, per-epoch metric collection, in-place CLI dashboard, v2 session history serialization, and test coverage |

Summary:
Replaced all unstructured `std::cout`/`std::cerr` output in `IncrementalTrainer` with structured `Logger` calls, implemented a real-time in-place CLI dashboard that redraws after every epoch using ANSI cursor movement, and added complete per-epoch metric collection and persistence. Absorbed the scope of TD-004.

Changes Made:

1. ✅ Added `EpochCallback` typedef and `set_epoch_callback()` to `ChatbotTrainer` — fires once per epoch inside `train(int)` without disrupting LR scheduling or data preprocessing
2. ✅ Extended `TrainingSession` struct with four per-epoch vectors: `per_epoch_losses`, `per_epoch_validation_losses`, `per_epoch_learning_rates`, `training_time_per_epoch`
3. ✅ Added `session_start_time_steady_`, `epoch_start_time_steady_`, and `mutable dashboard_lines_drawn_` members to `IncrementalTrainer`
4. ✅ Replaced all `std::cout`/`std::cerr` calls (20+ sites) with `Logger::info`, `Logger::warn`, `Logger::error`, `Logger::debug` across all methods
5. ✅ Overhauled `train_incremental()` to register the epoch callback, collect per-epoch metrics, and invoke `display_dashboard()` after each epoch
6. ✅ Implemented `display_dashboard()` — 10-line in-place redrawing box (ANSI `\033[NA` cursor-up, UTF-8 box-drawing chars) showing:
   - Epoch progress bar with percentage
   - Elapsed time and ETA (`avg_epoch_time × remaining_epochs`)
   - Current loss / val loss with delta arrows (`v` / `^`)
   - Current LR and epoch duration
   - Session best val loss and average epoch time
7. ✅ Implemented `format_duration()` and `progress_bar()` helpers
8. ✅ Rewrote `save_session_history()` with `# VERSION 2` header and `|losses:...|vallosses:...|lrs:...|times:...` pipe-delimited appendage after `checkpoint_path`
9. ✅ Rewrote `load_session_history()` to parse v2 extended format with full backward compatibility for old single-line format
10. ✅ Overhauled `print_training_summary()` with Unicode sparkline bars (▁▂▃▄▅▆▇█) per session, avg epoch time, best val loss, and total training time
11. ✅ Added `spdlog::spdlog` and `Logger.cpp` to `incremental_trainer` and `incrementaltrainerTests` CMake targets
12. ✅ Added 3 new GTest cases (38 total): `LoadSessionHistoryV2ParsesPerEpochVectors`, `SaveLoadSessionHistoryRoundTripWithPerEpochData`, `DisplayDashboardDoesNotCrash` — all pass

Files Modified:

- `src/ChatbotTrainer.hpp` — `EpochCallback` typedef, `epoch_callback_` member, `set_epoch_callback()` declaration
- `src/ChatbotTrainer.cpp` — `set_epoch_callback()` implementation; callback invocation in epoch loop
- `src/IncrementalTrainer.hpp` — Logger include; extended `TrainingSession`; timing/dashboard members; method declarations
- `src/IncrementalTrainer.cpp` — complete overhaul (Logger calls, timing, dashboard, v2 serialization, sparkline summary)
- `src/CMakeLists.txt` — added `Logger.cpp` and `spdlog::spdlog` to `incremental_trainer` target
- `tests/incrementaltrainer_test.cpp` — 3 new TD-009 tests
- `tests/CMakeLists.txt` — added `../src/Logger.cpp` and `spdlog::spdlog` to `incrementaltrainerTests` target

Verification:

- ✅ Full build succeeds (both `incremental_trainer` executable and `incrementaltrainerTests`)
- ✅ All 38 incremental trainer tests pass including the 3 new TD-009 tests
- ✅ v2 session history round-trip verified: write → parse → verify all four per-epoch vectors
- ✅ Dashboard smoke-tested with empty and populated session history

---

### TD-004: Enhanced Metrics Tracking for Training Sessions

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 2, 2026 | Training / IncrementalTrainer | Absorbed and fully implemented as part of TD-009 |

Summary:
All core implementation tasks from TD-004 were completed as part of the TD-009 implementation. The `TrainingSession` struct now carries four per-epoch vectors; `ChatbotTrainer` exposes per-epoch data via `EpochCallback`; session history serialization (`save_session_history` / `load_session_history`) uses a v2 format that persists and reloads these vectors; and `print_training_summary` renders Unicode sparklines for visual trend analysis. The 55 granular TODO comments that were seeded across the codebase have all been removed.

Tasks Completed:

- ✅ Extended `TrainingSession` struct with `per_epoch_losses`, `per_epoch_validation_losses`, `per_epoch_learning_rates`, `training_time_per_epoch`
- ✅ Modified `ChatbotTrainer` to expose per-epoch data via `EpochCallback` mechanism
- ✅ Updated `save_session_history()` and `load_session_history()` with v2 format (backward-compatible)
- ✅ Updated `print_training_summary()` with sparkline visualization of loss/val-loss trends per session
- ✅ Tests added: `LoadSessionHistoryV2ParsesPerEpochVectors`, `SaveLoadSessionHistoryRoundTripWithPerEpochData`
- ✅ All 55 TD-004 TODO comments removed from source files

**See:** [TD-009](#td-009-incremental-trainer-dashboard-and-structured-logging) for full implementation details.

---

### TD-008: Daemon Service Implementation (Steps 1-5)

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 1, 2026 | Service Management / Production Deployment | Complete 5-step daemon service transformation |

Summary:
Successfully transformed the ADAI Chatbot API Server from a development application into a production-ready daemon service. Implemented external configuration, graceful shutdown, structured logging, Docker deployment, and systemd integration following a comprehensive 5-step plan.

Changes Made:

#### Step 1: External Configuration ✅

1. Created `Config.hpp` and `Config.cpp` for centralized configuration management
2. Implemented multi-source configuration with priority system:
   - Command-line arguments (highest priority)
   - Environment variables
   - Configuration file
   - Default values (lowest priority)
3. Added support for all parameters: server config, model architecture, text generation
4. Created `config.conf` example configuration file
5. Updated `ChatbotAPIServer.cpp` to use Config system
6. Modified Docker files to support environment variables
7. Documentation: `STEP1_COMPLETE.md`

#### Step 2: Signal Handling ✅

1. Implemented async-signal-safe signal handlers for SIGTERM and SIGINT
2. Created graceful shutdown sequence:
   - Stop HTTP server
   - Complete in-flight requests
   - Save model state (if configured)
   - Clean up resources
   - Exit cleanly
3. Used `std::atomic<bool>` for thread-safe shutdown flag
4. Fixed model parameter order bug (vocab_size, d_model, num_heads)
5. Created test scripts: `test_signal_handling.sh`, `test_sigint.sh`
6. Verified 30-second timeout and clean shutdown
7. Documentation: `STEP2_COMPLETE.md`

#### Step 3: Structured Logging ✅

1. Integrated spdlog v1.12.0 via CMake FetchContent
2. Created `Logger.hpp` and `Logger.cpp` wrapper classes
3. Replaced all `std::cout`/`std::cerr` with structured logging:
   - `Logger::debug()` - detailed debugging info
   - `Logger::info()` - normal operations
   - `Logger::warn()` - warnings
   - `Logger::error()` - errors
4. Implemented timestamped log format: `[YYYY-MM-DD HH:MM:SS.mmm] [level] message`
5. Added configurable log levels (DEBUG, INFO, WARN, ERROR)
6. Configured auto-flush for container environments
7. Documentation: `STEP3_COMPLETE.md`

#### Step 4: Docker Configuration ✅

1. Enhanced `Dockerfile` with comprehensive environment variable documentation
2. Documented all 19 configuration options with inline comments
3. Organized into sections: Server, Model Architecture, Text Generation
4. Updated `docker-compose.yml` with detailed configuration examples
5. Created comprehensive deployment guide: `DOCKER_DEPLOYMENT.md` (500+ lines)
6. Documented configuration priority, volume mounts, health checks
7. Added troubleshooting, security, and monitoring sections
8. Multi-stage build already implemented for minimal runtime image
9. Documentation: `STEP4_COMPLETE.md`

#### Step 5: systemd Service File ✅

1. Created production-ready systemd service file: `scripts/adai.service`
2. Implemented extensive security hardening:
   - Filesystem protection (ProtectSystem=strict)
   - Kernel protection (ProtectKernelLogs, ProtectKernelModules, etc.)
   - Privilege restrictions (NoNewPrivileges, CapabilityBoundingSet=)
   - System call filtering (SystemCallFilter)
   - Network isolation (RestrictAddressFamilies=AF_INET AF_INET6)
3. Configured resource limits:
   - Memory: 4GB max, 3GB soft limit
   - CPU: 50% quota
   - Files: 65536 max, 2GB file size limit
4. Implemented automatic restart: on-failure with rate limiting (5 restarts per 10 minutes)
5. Created automated installation script: `scripts/install_systemd_service.sh`
   - One-command installation
   - Configurable paths, user, group, port
   - 8-step installation with verification
   - Preflight checks and interactive confirmation
6. Created comprehensive deployment guide: `SYSTEMD_DEPLOYMENT.md` (900+ lines)
7. Documentation: `STEP5_COMPLETE.md`, `DAEMON_IMPLEMENTATION_COMPLETE.md`

Files Created:

- Configuration: `src/Config.hpp`, `src/Config.cpp`, `config.conf`, `config.conf.example`
- Logging: `src/Logger.hpp`, `src/Logger.cpp`
- systemd: `scripts/adai.service`, `scripts/install_systemd_service.sh` *(the latter was later
  consolidated into `scripts/install_chatbot_API.sh` — commit `d914070`, "chore: consolidate
  systemd deployment onto install_chatbot_API.sh"; corrected here September 7, 2026)*
- Documentation (now under `docs/development/archive/` after a later doc reorg): `STEP1_COMPLETE.md`,
  `STEP2_COMPLETE.md`, `STEP3_COMPLETE.md`, `STEP4_COMPLETE.md`, `STEP5_COMPLETE.md`,
  `DAEMON_IMPLEMENTATION_COMPLETE.md`; the Docker/systemd deployment guides are now
  `docs/operations/deployment/docker.md` and `docs/operations/deployment/SYSTEMD_DEPLOYMENT.md`
- Testing: `scripts/test_signal_handling.sh`, `scripts/test_sigint.sh` *(the latter was later found
  to be a near-duplicate of the former and tagged `experimental` — see TD-046)*

Files Modified:

- `src/ChatbotAPIServer.cpp` - Added Config, Logger, signal handling
- `CMakeLists.txt` - Added spdlog dependency, Logger.cpp
- `src/CMakeLists.txt` - Linked spdlog library
- `Dockerfile` - Comprehensive environment variable documentation
- `docker-compose.yml` - Enhanced configuration with detailed comments

Verification:

- ✅ Configuration: All sources (CLI, env, file) work with correct priority
- ✅ Signal handling: SIGTERM/SIGINT trigger graceful shutdown
- ✅ Logging: Structured logs with timestamps at all levels
- ✅ Docker: Image builds, container starts, health checks pass
- ✅ systemd: Service file syntax valid, installation script functional

Impact:

- **Production Ready:** Service can now run as managed daemon
- **Observable:** Structured logs, configurable verbosity, timestamps
- **Reliable:** Graceful shutdown, automatic restart, resource limits
- **Secure:** Non-root user, extensive hardening, minimal privileges
- **Portable:** Docker and systemd deployment options
- **Documented:** Comprehensive deployment guides (1400+ lines total)

Related Future Enhancements:
See [Future Improvements](#configuration-and-service-management) section for enhancements building on this foundation:

- Configuration hot-reloading (Config Enhancement #1)
- JSON configuration format (Config Enhancement #2)
- Model state persistence (Config Enhancement #4)
- JSON log output (Logging Enhancement #1)
- Metrics endpoint (Container Enhancement #2)
- Socket activation (Container Enhancement #3)

---

### TD-005: Checkpoint Management and Symbolic Links

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| February 18, 2026 | Training / IncrementalTrainer | Complete symlink management implementation with cross-platform support |

Summary:
Implemented automatic checkpoint symlink management to provide easy access to "latest" and "best" model checkpoints. The system creates and maintains symbolic links (or file copies on Windows) that always point to the most recent checkpoint and the checkpoint with the lowest validation loss.

Changes Made:

1. ✅ Added configuration options for symlink management:
   - `enable_checkpoint_symlinks` - toggle feature on/off (default: true)
   - `latest_symlink_name` - configurable name for latest checkpoint link
   - `best_symlink_name` - configurable name for best checkpoint link

2. ✅ Implemented cross-platform symlink support:
   - Unix/Linux: Uses `std::filesystem::create_symlink()` for true symbolic links
   - Windows: Falls back to file copying for compatibility
   - Platform detection with `is_windows_platform()` helper

3. ✅ Created checkpoint tracking infrastructure:
   - Added `best_validation_loss` and `best_checkpoint_path` private members
   - Automatically initializes best checkpoint from session history on startup
   - Tracks and compares validation loss across all sessions

4. ✅ Implemented symlink helper methods:
   - `update_checkpoint_symlinks()` - Updates "latest" link after each session
   - `update_best_checkpoint()` - Updates "best" link when validation loss improves
   - `create_or_update_symlink()` - Creates/updates symlinks with error handling
   - `remove_symlink_if_exists()` - Safely removes existing links
   - `get_best_checkpoint_path()` - Retrieves path to best checkpoint

5. ✅ Integrated with training workflow:
   - `finalize_session()` now creates/updates symlinks after each session
   - `cleanup_old_sessions()` handles symlink cleanup when deleting checkpoints
   - Recalculates best checkpoint if the best model is deleted during cleanup

6. ✅ Enhanced training summary output:
   - `print_training_summary()` displays symlink paths and targets
   - Shows best checkpoint validation loss for quick reference
   - Handles both symlinks and file copies transparently

Files Modified:

- `include/IncrementalTrainer.hpp` - Added configuration and method declarations
- `src/IncrementalTrainer.hpp` - Added configuration and method declarations (duplicate header)
- `src/IncrementalTrainer.cpp` - Implemented all symlink management logic
  - Added `#include <limits>` for `std::numeric_limits`
  - Updated constructor to initialize best checkpoint tracking
  - Implemented 6 new symlink helper methods (~110 lines)
  - Updated `finalize_session()` to create symlinks
  - Updated `cleanup_old_sessions()` to handle symlink cleanup
  - Updated `print_training_summary()` to display symlink information

Usage Example:

```cpp
IncrementalTrainer trainer("vocab.txt", "model.bin");
trainer.config.enable_checkpoint_symlinks = true;  // Enabled by default
trainer.add_new_data("new_data.txt");
trainer.train_incremental(5);

// Symlinks are automatically created:
// - latest_checkpoint.bin -> training_sessions/session_N_checkpoint.bin
// - best_checkpoint.bin -> training_sessions/session_M_checkpoint.bin (lowest val loss)
```

Benefits Realized:

- Deployment scripts can reference `latest_checkpoint.bin` without parsing history
- Easy access to best model for inference and evaluation
- Automatic cleanup when old checkpoints are removed
- Cross-platform compatibility with Windows fallback
- Configurable for different deployment scenarios

Testing:

- Compiled successfully on Linux (GCC)
- All existing tests pass
- No breaking changes to existing functionality

---

### TD-002: Improve Error Handling in BPE Tokenizer

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| January 28, 2026 | NLP / Tokenization | Comprehensive error handling implementation |

Summary:
Implemented robust error handling and validation for the BPE tokenizer, including custom exception types, UTF-8 validation, input validation, and vocabulary file format validation.

Changes Made:

1. ✅ Created custom exception types:
   - `TokenizerInputError` - for empty/invalid input
   - `TokenizerEncodingError` - for UTF-8 encoding issues
   - `VocabularyFileError` - for malformed vocabulary files
   - `TokenIDError` - for out-of-range token IDs

2. ✅ Implemented UTF-8 validation:
   - `is_valid_utf8()` helper method validates character sequences
   - Detects invalid start bytes, incomplete sequences, and malformed continuation bytes
   - Applied to `encode()` and `pre_tokenize()` methods

3. ✅ Added input validation:
   - `validate_input()` helper checks for empty strings and UTF-8 validity
   - `encode()` validates non-empty input with proper UTF-8
   - `decode()` validates non-empty token ID vector and checks for negative IDs

4. ✅ Enhanced vocabulary file validation:
   - Validates filename is not empty
   - Throws descriptive exceptions for file not found
   - Validates special tokens section format and integer parsing
   - Validates vocabulary entries with tab separators and non-negative IDs
   - Validates BPE merges format and non-empty tokens
   - Ensures loaded vocabulary contains required special tokens

5. ✅ Created comprehensive test suite:
   - 27 tests covering all error conditions
   - Tests for exception types, messages, and inheritance
   - Edge case coverage (empty input, invalid UTF-8, malformed files)
   - All tests passing

Files Modified:

- `src/BPETokenizer.hpp` - Added custom exception types and validation methods
- `src/BPETokenizer.cpp` - Implemented validation throughout encode/decode/load_vocab
- `tests/tokenizer_error_handling_test.cpp` - New comprehensive test suite (27 tests)
- `tests/CMakeLists.txt` - Added tokenizerErrorHandlingTests target

Verification:
All 27 tests pass, validating:

- Empty input detection
- UTF-8 validation for various invalid sequences
- Token ID range checking
- Vocabulary file format validation
- Exception type hierarchy
- Descriptive error messages

---

### TD-001: Complete Optimizer Parameter Exposure

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| January 28, 2026 | Optimizer Integration | Complete parameter registration implementation |

Summary:
Successfully completed parameter exposure for all model components. The optimizer now fully manages all model parameters through the centralized `optimizer->step()` mechanism.

Changes Made:

1. ✅ Verified `LLMEncoder::register_parameters_with_optimizer()` properly exposes all parameters (token embedding, encoder blocks, final norm)
2. ✅ Verified `LLMDecoder::register_parameters_with_optimizer()` properly exposes all parameters (token embedding, decoder blocks, final norm)
3. ✅ Verified `LanguageModelHead::set_optimizer()` properly exposes all parameters (W_output, bias)
4. ✅ Updated `ChatbotTrainer` to use `optimizer->step()` instead of `model->update_weights()`
5. ✅ Removed obsolete TODO comments
6. ✅ Tested training with optimizer integration - works correctly

Files Modified:

- `src/ChatbotTrainer.cpp` - Replaced `model->update_weights()` with `optimizer->step()`, removed outdated comments

Verification:
Training runs successfully with AdamW optimizer using centralized parameter management. All parameter groups are properly registered and updated through `optimizer->step()`.

---

---

