# `Dataset` — Source File Reference (Retired)

> **Retired October 11, 2026 (TD-315).** `src/Dataset.hpp`, `tests/dataset_test.cpp` and the
> `dataset_enhanced_example`/`dataset_batch_processing_example` examples were deleted because no
> production code used them. TD-315–TD-320 and TD-219 are closed in
> [TECHNICAL_DEBT_RESOLVED.md](../../archive/TECHNICAL_DEBT_RESOLVED.md). This page is kept as the
> record of what the class did and which bugs to avoid if any of it is rebuilt: the capabilities
> with no other implementation are listed under TD-014 in
> [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md#td-014-llm-operations-and-training-tooling-suite).
> The last version is in git history: `git show 6a5438c:src/Dataset.hpp`. Sections below describe
> the code as it was on that commit.

- **File:** `src/Dataset.hpp` (removed), header-only (~1,580 lines, no `.cpp`)
- **Namespace:** global (no namespace)
- **Built into:** nothing in production; included only by tests and examples (§2)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** `tests/dataset_test.cpp` (removed; 43 tests)
- **Depends on:** [`BatchProcessor.hpp`](BatchProcessor.md) (`TokenBatch`, `create_batch`, …), `TrainingSampleMeta.hpp` (`SampleMeta`, JSONL helpers)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by compiling the header into
> a scratch program (with AddressSanitizer) and running it.

---

## 1. What this file is

An in-memory **dataset container** for `(input, target)` text pairs, plus a lazily-indexed
file reader:

| Type | Purpose |
|---|---|
| `DataSample` | `input`, `target`, `SampleMeta meta` |
| `SplitType` | `TRAIN`, `VALIDATION`, `TEST` |
| `DatasetStats` | Counts plus average and max input/target **character** lengths |
| `Dataset` | Load (5 formats), split (random, stratified, k-fold), shuffle, filter, augment, iterate (per sample or in batches), save, and the `BatchProcessor` wrappers |
| `Dataset::BatchIterator` / `BatchRange` | Range-for over a split in fixed-size batches of copied `DataSample`s |
| `LazyDataset` | Index a JSONL or `INPUT:`/`RESPONSE:` file by line offset and read samples on demand |

### Why it matters (and doesn't)

**No production code uses it.** `grep` finds `Dataset.hpp` included only by
`tests/dataset_test.cpp` and three examples. The real training path is `ChatbotTrainer`'s own
loader plus `DatasetRegistry`, with JSONL parsing via `TrainingSampleMeta.hpp`'s
`parse_jsonl_sample()`, which `Dataset` reuses. It's tagged `stable` and its header comment
advertises "lazy loading for very large datasets" and "memory-efficient iteration", so it reads as
the training data layer. It isn't. The bugs below matter only to anyone who adopts it, which the
`stable` tag invites.

---

## 2. Where it's used

| Caller | Uses |
|---|---|
| `tests/dataset_test.cpp` | All of `Dataset` (43 tests) |
| `examples/DatasetEnhancedExample.cpp` (`dataset_enhanced_example`) | Loading, splitting, k-fold, augmentation, filtering |
| `examples/DatasetBatchProcessingExample.cpp` (`dataset_batch_processing_example`) | The `BatchProcessor` wrappers |
| `examples/EnhancedTrainingExample.cpp` (`enhanced_training_example`) | Loading and splitting for a toy training loop |

`LazyDataset` has no caller at all, not even a test.

---

## 3. `Dataset` API

### Loading

`load_from_file(path)` reads the **first line** and picks a parser. It **appends** to existing
data, and recomputes stats on success:

| First line… | Parser | Notes |
|---|---|---|
| contains `INPUT:` | `parse_conversation_format` | `INPUT:` / `RESPONSE:` lines; a **blank line** ends a pair |
| starts with `{` and contains `"response"` | `parse_jsonl_format` → `parse_jsonl_sample()` | The format `DataFetcher`/`sample_to_jsonl()` write; keeps metadata |
| contains `{` and `"input"`/`"Input"` | `parse_json_format` | One object per line with `"input"`/`"target"`; plain `find`/`substr` |
| contains `,` | `parse_csv_format` | Splits at the **first** comma; skips the first line if it contains `input`/`Input` |
| contains a tab | `parse_tsv_format` | Splits at the first tab |
| anything else | `parse_conversation_format` | Default |

> **Verified parser pitfalls:**
> - **Conversation pairs need blank-line separators.** `INPUT: a / RESPONSE: b / INPUT: c /
>   RESPONSE: d` with no blank line kept only `(c, d)`; the first pair was silently overwritten.
>   A multi-line response keeps only its first line (`line one`, dropping `line two`).
> - **Detection keys off line 1 only.** A JSONL file with a leading blank line loaded **0
>   samples** (`ok=false`), because it fell through to the conversation parser. A conversation file
>   whose first line was `Chat log, exported` was parsed as **CSV**: one bogus pair
>   (`Chat log` → `exported`), and the real `INPUT:`/`RESPONSE:` pair was lost.
> - **CSV ignores quoting.** `"Hello, there",Hi` became input `"Hello`, target `there",Hi`. A data
>   row containing the word "input" on line 1 is treated as a header and dropped.
> - **JSON ignores escapes.** `"say \"hi\" now"` became `say \`. This parser also only knows
>   `"target"`, not `"response"`.

`add_sample(in, tgt)` and `add_samples(vec)` append and **recompute statistics over the whole
dataset each call**. Verified: 20,000 `add_sample()` calls took **1.08 s**, against 0.02 s for one
`add_samples()` of the same 20,000. That's O(n²) for incremental building.

### Splitting

| Method | Behaviour |
|---|---|
| `split(train, val, test)` | Normalizes ratios if they don't sum to 1, shuffles indices with the member `mt19937` (seed 42 by default), and cuts at `floor(n·train)` and `floor(n·val)`; the remainder goes to test |
| `split_stratified(train, val, test, num_bins = 5)` | Bins by total character length (`len · bins / (max+1)`), then splits each bin the same way |
| `setup_k_fold(k)` / `get_fold(i, train&, val&)` | Round-robin shuffled indices into `k` folds; `get_fold` copies the samples out |
| `shuffle()` | Shuffles `data_` itself. Existing split indices then point at different samples (documented) |
| `shuffle_split(type)` | Shuffles one split's indices |

> **Verified edge cases:**
> - `split_stratified(…, num_bins = 0)` **crashes** (SEGV under ASan: `bins[min(…, -1)]`).
> - `split(0, 0, 0)` divides by a zero total. The NaN-to-`size_t` cast is undefined; here it put
>   everything in test.
> - **Stale folds after `clear()`.** `clear()` resets data and splits but **not** `num_folds_` or
>   `fold_indices_`. After `setup_k_fold(5)` on 1,000 samples, `clear()`, and adding one sample,
>   `get_fold(0)` returned **200 samples from a 1-sample dataset**: out-of-range reads of destroyed
>   objects (undefined behaviour).
> - **Samples added after `split()`** (or by `augment_data()`) belong to no split: 11 samples →
>   8 + 2 + 0. Nothing warns.

Small datasets can get an empty train split (`floor(1 · 0.8) = 0`). If you `augment_data()` and
then re-`split()`, augmented copies of validation/test samples can land in train (leakage).

### Transforming

| Method | Behaviour |
|---|---|
| `set_augmentation(fn)` / `augment_data(k)` | Appends `k` augmented copies per original |
| `set_preprocessing(fn)` / `apply_preprocessing()` | Rewrites every input and target in place |
| `filter_by_length(min, max)`, `filter_by_pattern(pat, keep)` | Erase samples and clear splits (correctly) |
| `lowercase()` | `std::transform(…, ::tolower)` on raw `char`s. Passing a negative `char` (any UTF-8 byte ≥ 0x80) to `::tolower` is **undefined behaviour**; use `unsigned char` |

### Access and iteration

- `get_split(type)` returns copies; `get_all()` is a reference.
- `size()`, `size(type)`, `empty()`, `is_split()`, `get_stats()`, `print_stats()` (`std::cout`).
- `begin()`/`end()` iterate over all samples.
- `get_batch_range(type, n)` / `get_batch_range(n)` give range-for over batches of copied
  `std::vector<DataSample>`.

> **Verified:** `get_batch_range(…, 0)` **loops forever**. `operator++` adds a batch size of 0 and
> never reaches the end. There's no check for zero.

`save_to_file(path, "jsonl" | "conversation" | "tsv")` writes all samples and returns `true`
**without checking the write**. TSV and conversation output don't escape tabs or newlines, so
multi-line samples don't round-trip.

### `BatchProcessor` wrappers

These five methods tokenize a split with a caller-supplied `tokenizer_fn` and hand the
sequences to [`BatchProcessor.hpp`](BatchProcessor.md):

| Method | Behaviour |
|---|---|
| `get_batch_with_padding(type, start, n, tok, pad)` | Inputs `[start, start+n)` of the split → `create_batch()` |
| `get_target_batch_with_padding(…)` | The same, for targets (aligned with the input batch) |
| `get_dynamic_batches(type, tok, max_batch, tolerance, pad)` | Tokenizes the whole split → `create_dynamic_batches()` (groups similar lengths) |
| `process_batch<T>(samples, tok, model_fn, pad)` | Tokenize, pad, and call `model_fn` on the batch |
| `get_batch_statistics(type, tok, n)` | `compute_batch_stats()` over the **first** batch only, padded with a hardcoded 0 |

`pad` defaults to `SpecialTokenIDs::PAD`. Already tracked as TD-219: wrong doc examples
(`.slice()`, `model.forward_batch()`, `efficiency_percentage` don't exist), the one-batch stats
helper, and the hardcoded pad. The model takes one sequence at a time, so nothing consumes a
`TokenBatch` (TD-171).

---

## 4. `LazyDataset`

The constructor indexes the file:
- It treats the file as JSONL if the first non-empty line starts with `{`, and records the offset
  of each `{…"input"…}` line.
- Otherwise it records the offset of **every line containing `INPUT:` anywhere**.

`get_sample(i)` reopens the file, seeks, and parses one sample: JSONL via `parse_jsonl_sample()`,
or the `INPUT:`/`RESPONSE:` lines. `load_range(start, n)` calls `get_sample()` per index, so it
opens the file `n` times. `stats_` is declared but never computed.

> **Verified:** a 2-pair conversation file whose second **response** contained the text `INPUT:`
> indexed as **3** samples. The third was `("" → "has INPUT: inside")`. (The anchored parsing fix
> in TD-057 covers `get_sample()`, but the indexer still uses `find()`.)

---

## 5. Tests

`dataset_test.cpp` has 43 `TEST_F(DatasetTest, …)` tests:
- loading in each format;
- add, split and stratified split;
- k-fold;
- augmentation, preprocessing and filtering;
- iteration and batch ranges;
- save round trips;
- the `BatchProcessor` wrappers.

**Not covered:**
- format misdetection;
- conversation files without blank lines;
- CSV quoting and JSON escapes;
- zero or invalid arguments (`batch_size`, `num_bins`, ratios);
- `clear()` after k-fold;
- `LazyDataset` (anything at all).

---

## 6. Known gaps and gotchas (summary)

Each item was closed by the retirement (TD-315) and is archived in
[TECHNICAL_DEBT_RESOLVED.md](../../archive/TECHNICAL_DEBT_RESOLVED.md).

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Unused by production but tagged `stable` and documented as the data layer (§1) | Misleads readers; ~1,580 lines to maintain | By inspection | [TD-315](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-315-datasethpp-is-tagged-stable-but-has-no-production-caller) |
| Parsers lose or mangle data: unseparated conversation pairs, multi-line responses, first-line-only format detection, CSV quoting, JSON escapes (§3) | Silent data loss or corruption for anyone loading real files | **Verified** | [TD-316](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-316-dataset-file-parsers-lose-or-mangle-data) |
| `batch_size 0` loops forever; `num_bins 0` crashes; zero ratio sum is undefined; `clear()` leaves stale k-fold indices (out-of-range reads) (§3) | Hangs, crashes or undefined behaviour on bad arguments or call order | **Verified** | [TD-317](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-317-dataset-hangs-crashes-or-reads-freed-memory-on-bad-arguments-or-call-order) |
| `add_sample()` recomputes stats each call (O(n²)); post-split additions belong to no split; augmentation before re-split leaks val/test (§3) | Slow incremental building; silent split gaps | **Verified** / by inspection | [TD-318](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-318-dataset-split-bookkeeping-and-on-add_sample) |
| `LazyDataset` indexes any line containing `INPUT:`; reopens the file per sample; stats never computed; untested (§4) | Phantom samples; slow ranges | **Verified** | [TD-319](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-319-lazydataset-indexes-phantom-samples) |
| `lowercase()` passes signed `char` to `::tolower`; `save_to_file()` unchecked and TSV/conversation don't escape; `std::cout`/`std::cerr` in a library header (§3) | UB on non-ASCII; silent write failures; lossy round trips | By inspection | [TD-320](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-320-dataset-hygiene-tolower-ub-unchecked-saves-console-output) |
| Batch wrappers' wrong doc examples, one-batch stats, hardcoded pad | Closed with the retirement | — | TD-219 |

---

## 7. Merge notes

This file replaces `docs/development/api/data/dataset-batch-processing.md` (v1.0, January 2026,
about 1,100 lines). That doc was already marked **stale** (September 14, 2026): most of it
described `ParallelDataLoader`, `DataLoaderConfig`, `DataLoaderIterator`, `TokenBatchLoader`,
`TokenBatchIterator` and `ThreadSafeBatchQueue`. All were retired by TD-052 and TD-170, and
`src/ParallelDataLoader.hpp` no longer exists. Only its `Dataset` batch-method sections were still
current, and those are folded into §3 here. Corrections:

- `pad_token_id` defaults to `SpecialTokenIDs::PAD`, not 0 (except `get_batch_statistics()`,
  which hardcodes 0).
- `BatchStats` has no `efficiency_percentage`; use `1 − padding_ratio`. See [BatchProcessor.md](BatchProcessor.md).
- The old doc's "Production Ready" status, prefetching, epoch management and the
  throughput/efficiency tables are dropped: they described the retired loaders, and the model has
  no batch dimension to use them (TD-171).
