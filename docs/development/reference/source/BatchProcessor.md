# `BatchProcessor` — Source File Reference

- **File:** [`src/BatchProcessor.hpp`](../../../../src/BatchProcessor.hpp) (header-only; free functions and two structs, no class named `BatchProcessor`)
- **Library:** none — header-only. Depends only on `Matrix.hpp` and `SpecialTokens.hpp`, so anything linking `adai_core` can use it
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/batchprocessor_test.cpp`](../../../../tests/batchprocessor_test.cpp) → `batchprocessorTests` (29 tests, ctest `BatchProcessorTests`); also 5 `BatchProcessorTest.*` cases in `tests/inference_optimization_test.cpp` → `inferenceOptimizationTests`, and `Dataset` wrapper coverage in `tests/dataset_test.cpp`
- **Example / benchmark:** `examples/DatasetBatchProcessingExample.cpp` → `dataset_batch_processing_example`; `benchmarks/InferenceOptimizationBenchmark.cpp` → `inference_optimization_benchmark` (both built when `BUILD_EXAMPLES=ON`, the default)
- **Last traced against the code:** 2026-10-08

> Code-traced reference: every "where it's used" claim points at a real call site at the time of
> writing. It supersedes the older `docs/development/reference/batchprocessor.md` ("BatchProcessor
> API Reference", now deleted); its accurate content has been merged in here. See
> [§10](#10-merge-notes-what-changed-from-the-old-document) for what was corrected or dropped.

---

## Contents

1. [What this file is — and what it can't do yet](#1-what-this-file-is--and-what-it-cant-do-yet)
2. [Where it's used](#2-where-its-used)
3. [`TokenBatch`](#3-tokenbatch)
4. [`create_batch()`](#4-create_batch)
5. [`create_dynamic_batches()`](#5-create_dynamic_batches)
6. [`create_padding_mask()`](#6-create_padding_mask)
7. [`unbatch_outputs()`](#7-unbatch_outputs)
8. [`BatchStats` and `compute_batch_stats()`](#8-batchstats-and-compute_batch_stats)
9. [Known gaps and gotchas (summary)](#9-known-gaps-and-gotchas-summary)
10. [Merge notes](#10-merge-notes-what-changed-from-the-old-document)

---

## 1. What this file is — and what it can't do yet

`BatchProcessor.hpp` is a small toolkit of **padding and grouping helpers** for token sequences:

| Symbol | Kind | Does |
|---|---|---|
| `TokenBatch` | struct | Padded sequences plus their original lengths |
| `create_batch()` | function | Right-pads a list of sequences to the longest one |
| `create_dynamic_batches()` | function | Sorts sequences by length and groups them into several `TokenBatch`es to reduce padding |
| `create_padding_mask()` | function | Builds a `[batch_size, max_length]` 1/0 matrix marking real vs. padding tokens |
| `unbatch_outputs()` | function | Trims per-sequence output matrices back to their real lengths |
| `BatchStats` | struct | Padding-efficiency figures, with a `print()` helper |
| `compute_batch_stats()` | function | Computes `BatchStats` over a list of batches |

All functions are `inline`, stateless, and allocate their results. They are safe to call from any
thread.

### The important limitation

The file's own header comment lists "Process multiple sequences in one forward pass" as the key
benefit. **Nothing in this codebase can consume a padded batch.** Every layer —
`EncoderDecoderModel::forward()` down to `Matrix` — processes exactly one sequence at a time
(TD-171 in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md)). So:

- No production code feeds a `TokenBatch` to a model.
- `create_padding_mask()` and `unbatch_outputs()` have no production callers at all.
- Feeding a padded sequence through the existing one-sequence-at-a-time model would make it
  process (and attend to) `PAD` tokens with no mask, which is *worse* than not padding.

The one real production job this file does today is **measuring** how much padding a batched
model *would* waste: `ChatbotAPI`'s batch endpoints report `compute_batch_stats()` figures in their
JSON responses (§2). The sibling `TokenBatchLoader` (the training-side consumer of this same
padding model) was retired for exactly this reason (TD-170), and the batch-dimension gap behind it
was split out as TD-171.

### Why it matters

- It's the vocabulary (`TokenBatch`, padding ratio, length tolerance) any future real-batching
  work would start from, so its semantics should stay precise.
- The `stats` block it produces is part of a public HTTP response (`POST /chat/batch`,
  `POST /chat/batch-session`), so its meaning matters to API clients (see §8's caveat).

---

## 2. Where it's used

| Location | Uses | Production? |
|---|---|---|
| `src/ChatbotAPI.hpp` | Includes the header; `BatchResponse::stats` is a `BatchStats` | Yes |
| `src/ChatbotAPI.cpp` — `generate_batch_responses()` (≈ line 944) | Tokenizes all inputs, calls `create_dynamic_batches(seqs, 32, 10, PAD)` **only** to compute `compute_batch_stats()`, then generates each input **one at a time in original order** | Yes |
| `src/ChatbotAPI.cpp` — `generate_batch_session_responses()` (≈ line 1034) | Formats each session's context, calls `generate_batch_responses()`, copies its `stats` | Yes |
| `src/ChatbotAPI.cpp` — `create_batch_json_response()` (≈ line 716) | Serializes `stats` as `{"total_tokens", "actual_tokens", "padding_ratio", "num_batches", "avg_batch_size", "efficiency"}` | Yes |
| `src/ChatbotAPI.cpp` — routes `POST /chat/batch`, `POST /chat/batch-session` (≈ lines 113–128) → `handle_batch_chat()` / `handle_batch_chat_session()` | Entry points that reach the above | Yes |
| `src/Dataset.hpp` — `Dataset::get_batch_with_padding()`, `get_target_batch_with_padding()`, `get_dynamic_batches()`, `process_batch<T>()`, `get_batch_statistics()` (≈ lines 1160–1432) | Thin wrappers: tokenize a split (or sample list) via a caller-supplied function, then call `create_batch()` / `create_dynamic_batches()` / `compute_batch_stats()` | No — tests and the example only |
| `tests/batchprocessor_test.cpp`, `tests/inference_optimization_test.cpp`, `tests/dataset_test.cpp` | Unit tests | — |
| `examples/DatasetBatchProcessingExample.cpp` | Demonstrates `create_padding_mask()` and `compute_batch_stats()` via `Dataset` | — |
| `benchmarks/InferenceOptimizationBenchmark.cpp` | Times `create_dynamic_batches()` | — |

### The ordering bug `ChatbotAPI` already hit

`create_dynamic_batches()` returns batches in **length-sorted** order and a `TokenBatch` carries no
record of each sequence's original index. `generate_batch_responses()` used to build its response
list by iterating those batches, so whenever inputs had different lengths, `/chat/batch` returned
responses in the wrong order. The fix (see the comment above the call) drives generation off the
original `input_token_sequences` and uses the dynamic batches for stats only. Any new caller must
do the same, or keep its own index mapping (§5).

### `Dataset` wrapper notes

- `get_batch_statistics(split, tokenizer_fn, batch_size)` only looks at the **first**
  `batch_size` samples of the split (as one `create_batch()`), not the whole split, and hardcodes
  `pad_token_id = 0` instead of using its usual `SpecialTokenIDs::PAD` default (the two are equal
  today, `PAD == 0`). A real indexing bug in it was fixed earlier; `dataset_test.cpp` (≈ line 370)
  has the regression test.
- Its doc example prints `stats.efficiency_percentage`, which doesn't exist; efficiency is
  `1 - padding_ratio`.
- `process_batch<T>()`'s doc example calls `model.forward_batch(seqs)`, which doesn't exist (TD-171).

---

## 3. `TokenBatch`

```cpp
struct TokenBatch {
    std::vector<std::vector<int>> batch_token_ids;  // padded sequences, all max_length long
    std::vector<int> lengths;                       // original (unpadded) length of each
    int max_length = 0;                             // padded length
    int pad_token_id = 0;                           // token used for padding
    int batch_size() const;                         // batch_token_ids.size()
    bool is_empty() const;                          // batch_token_ids.empty()
};
```

Example, for input `{{1,2,3}, {4,5,6,7,8}}` padded with `0`:

```text
batch_token_ids = {{1,2,3,0,0}, {4,5,6,7,8}}
lengths         = {3, 5}
max_length      = 5
```

Invariants (maintained by `create_batch()`, **not** enforced by the struct):
`lengths.size() == batch_token_ids.size()`, every row has `max_length` entries, and
`lengths[i] <= max_length`. Hand-built `TokenBatch`es that break them will make
`create_padding_mask()`, `unbatch_outputs()` and `compute_batch_stats()` read out of bounds.

Padding is always **right** padding (real tokens first). There's no left-padding option.

Note there is **no original-index field**: once sequences are reordered (by
`create_dynamic_batches()`), there's no way to map a row back to its input.

---

## 4. `create_batch()`

```cpp
TokenBatch create_batch(const std::vector<std::vector<int>>& sequences,
                        int pad_token_id = adai::SpecialTokenIDs::PAD);   // PAD == 0
```

1. `max_length` = the longest input's length (0 for empty input, which returns an empty batch).
2. Copies each sequence and appends `pad_token_id` until it's `max_length` long.
3. Records each original length in `lengths`.

Input order is preserved. Cost is O(batch_size × max_length) copying. Empty sequences are allowed
(length 0, a row of all padding); `EdgeCaseAllEmptySequencesInvalid` documents that.

**Used by:** `create_dynamic_batches()` (once per group), `Dataset::get_batch_with_padding()`,
`get_target_batch_with_padding()`, `process_batch<T>()`, `get_batch_statistics()`, tests,
the example.

**Why it matters:** it's the single place padding is defined; every other function trusts the
invariants it establishes.

---

## 5. `create_dynamic_batches()`

```cpp
std::vector<TokenBatch> create_dynamic_batches(const std::vector<std::vector<int>>& sequences,
                                               int max_batch_size = 32,
                                               int length_tolerance = 10,
                                               int pad_token_id = adai::SpecialTokenIDs::PAD);
```

**Algorithm.**

1. Build `(length, original_index)` pairs and sort them (ascending length; ties by original
   index, so the result is deterministic).
2. Walk the sorted list, appending each sequence to the current group. Start a new group first if
   the current one already has `max_batch_size` items, **or** if this sequence's length exceeds
   the current group's **minimum** (i.e. first) length by more than `length_tolerance`.
3. `create_batch()` each group.

So within any output batch, `max_length − min_length <= length_tolerance`, and batches come out in
ascending-length order.

**Worked example** (verified by running the real function), lengths `[3, 2, 4, 6, 3]`:

| Call | Batches (lengths) | Padding tokens |
|---|---|---|
| `create_batch(all)` (one batch) | `[3 2 4 6 3]` padded to 6 | 12 |
| `create_dynamic_batches(s, 32, 10)` | `[2 3 3 4 6]` | 12 |
| `create_dynamic_batches(s, 32, 2)` | `[2 3 3 4]`, `[6]` | 4 (67% less) |
| `create_dynamic_batches(s, 3, 2)` | `[2 3 3]`, `[4 6]` | 3 |

**Tuning `length_tolerance`.** Smaller tolerance → less padding but more, smaller batches; larger →
fewer batches but more padding. A sweep over the real API is the honest way to pick it:

```cpp
for (int tol : {5, 10, 20, 50}) {
    BatchStats s = compute_batch_stats(create_dynamic_batches(seqs, 32, tol));
    adai::Logger::info("tol={} batches={} padding={:.1f}%", tol, s.num_batches,
                       s.padding_ratio * 100.0f);
}
```

If the padding ratio stays high, a very bimodal length distribution is the usual cause; splitting
short and long sequences into separate calls (each with its own tolerance) helps.

**Edge cases** (verified): `max_batch_size <= 0` or a negative `length_tolerance` don't loop
forever — they produce one batch per sequence. Empty input returns no batches.

**Gotcha — order and identity are lost.** Outputs are length-sorted with no index back to the
input. Keep `(length, index)` pairs yourself if results must be returned in input order (§2).

**Used by:** `ChatbotAPI::generate_batch_responses()` (stats only), `Dataset::get_dynamic_batches()`,
tests, the benchmark.

---

## 6. `create_padding_mask()`

```cpp
Matrix create_padding_mask(const TokenBatch& batch);   // [batch_size, max_length]
```

Row *i*, column *j* is `1.0f` if `j < lengths[i]` (real token), else `0.0f` (padding).

For `{{1,2,3}, {4,5,6,7}}`: row 0 is `[1 1 1 0]`, row 1 is `[1 1 1 1]`.

**How it would relate to this codebase's attention.** `MultiHeadAttention`/`CrossAttention` take an
optional per-sequence mask of shape `[query_len, key_len]` where `0` means "set this score to
`-1e9f`". This function's output is a different shape: one row per *sequence*, marking *key*
positions. Using it would mean expanding row *i* into a `[len, len]` (or `[q, k]`) matrix per
sequence, combining it with any causal mask, and feeding padded sequences through a model that
accepts a batch, which doesn't exist (TD-171).

**Used by:** tests and `DatasetBatchProcessingExample` only. No production caller.

---

## 7. `unbatch_outputs()`

```cpp
std::vector<Matrix> unbatch_outputs(const std::vector<Matrix>& batch_outputs,
                                    const TokenBatch& batch);
```

For each sequence *i*, copies the first `lengths[i]` rows of `batch_outputs[i]` (all columns) into
a new matrix, dropping padding rows.

- Takes one `Matrix` **per sequence**, not a 3-D `[batch, max_length, d_model]` tensor. (The
  header's doc comment says the latter; `Matrix` is 2-D.)
- **No bounds checks:** it assumes `batch_outputs.size() >= batch.batch_size()` and
  `batch_outputs[i].rows >= lengths[i]`.

**Used by:** tests only. No production caller.

---

## 8. `BatchStats` and `compute_batch_stats()`

```cpp
struct BatchStats {
    int total_tokens = 0;         // sum over batches of batch_size × max_length
    int actual_tokens = 0;        // sum of all lengths
    float padding_ratio = 0.0f;   // (total - actual) / total
    int num_batches = 0;
    float avg_batch_size = 0.0f;  // sequences / batches
    void print() const;           // writes a summary to std::cout
};

BatchStats compute_batch_stats(const std::vector<TokenBatch>& batches);
```

`compute_batch_stats()` sums the per-batch figures, then derives `padding_ratio` and
`avg_batch_size`, each guarded against division by zero (0 for empty input). "Efficiency", as
reported by `print()` and by `ChatbotAPI`'s JSON, is `(1 − padding_ratio) × 100`.

**`print()`** writes six lines to `std::cout`. That breaks the project's "never `std::cout` in
library code" rule; use `adai::Logger` with the fields instead. Nothing in `src/` calls it.

**Used by:** `ChatbotAPI::generate_batch_responses()` → the `stats` object in
`POST /chat/batch` / `POST /chat/batch-session` responses; `Dataset::get_batch_statistics()`;
tests; the example.

> **What the `/chat/batch` stats actually mean.** They describe how much padding *would* be wasted
> if the request's inputs were grouped with `max_batch_size = 32`, `length_tolerance = 10`.
> Generation never uses those groups: each input is generated on its own, unpadded. So
> `num_batches`, `avg_batch_size`, `padding_ratio` and `efficiency` are hypothetical, not a
> measurement of work done. API clients shouldn't treat them as performance figures.

---

## 9. Known gaps and gotchas (summary)

> **Planned (October 8, 2026):** real batching is scheduled as a v2 of this file — see
> [real_batching_v2_plan.md](../../../proposals/real_batching_v2_plan.md). Until then the items
> below describe current (v1.x) behaviour.

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Notes | Tracked as |
|---|---|---|---|
| No model can consume a `TokenBatch` (§1) | Header's "one forward pass" benefit is unavailable; padding helpers are mostly unused | Blocked on TD-171 | [TD-171](../../guides/TECHNICAL_DEBT.md#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported), [TD-218](../../guides/TECHNICAL_DEBT.md#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs) |
| `/chat/batch` `stats` describe hypothetical batching (§8) | Clients may read them as real performance data | Generation is per input, unpadded | [TD-217](../../guides/TECHNICAL_DEBT.md#td-217-chatbatch-reports-hypothetical-batching-stats-as-if-they-were-real) |
| `create_dynamic_batches()` loses input order and identity (§5) | Easy to return results in the wrong order | Already caused a `/chat/batch` bug (fixed) | [TD-218](../../guides/TECHNICAL_DEBT.md#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs) |
| `create_padding_mask()`, `unbatch_outputs()` have no production callers (§6–7) | Dead surface | Mask shape doesn't match the attention layers' `[q, k]` convention | [TD-218](../../guides/TECHNICAL_DEBT.md#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs) |
| `unbatch_outputs()` doc says 3-D input; no bounds checks (§7) | Misleading doc; OOB on bad input | | [TD-218](../../guides/TECHNICAL_DEBT.md#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs) |
| `TokenBatch` invariants unenforced (§3) | OOB in downstream helpers for hand-built batches | | [TD-218](../../guides/TECHNICAL_DEBT.md#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs) |
| `BatchStats::print()` uses `std::cout` (§8) | Violates logging convention | No `src/` caller | [TD-218](../../guides/TECHNICAL_DEBT.md#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs) |
| `Dataset::get_batch_statistics()` samples only the first batch, hardcodes pad 0; doc examples use nonexistent `efficiency_percentage` / `forward_batch` (§2) | Misleading docs; partial stats | Wrappers have no production caller | [TD-219](../../guides/TECHNICAL_DEBT.md#td-219-datasets-batch-wrappers-have-wrong-doc-examples-and-a-partial-stats-helper) |

---

## 10. Merge notes: what changed from the old document

This file absorbed `docs/development/reference/batchprocessor.md` (January 2026, "Status:
Production-ready", 1,080 lines). Where they disagreed, this file's code-traced details won.

**Corrected:**
- *"2-4x throughput improvement"*, the batch-size throughput tables, and the "4-12x combined with
  KV cache" table: no batched forward pass exists (TD-171), so batching yields no throughput gain
  here. Dropped.
- *`create_dynamic_batches(sequences, 3, 2, 0)` example* showed a 4-item batch, exceeding
  `max_batch_size = 3`. Real output is `[2 3 3]`, `[4 6]` (§5).
- *Padding arithmetic* (14 → 5 tokens, "64% less") was wrong; the real figures are 12 → 4 (67%
  less), verified by running the function (§5).
- *Default `pad_token_id = 0`*: the real default is `adai::SpecialTokenIDs::PAD`, which currently
  equals 0.
- *"Integration with Attention" example* indexed a 3-D `attention_scores(i, k, j)`; `Matrix` is
  2-D and the attention layers take a `[q, k]` per-sequence mask (§6).
- *"Production-ready"* status and *"Use Case: production systems"* claims: the only production use
  is the stats block (§2).
- *See Also links* pointed at `../guides/inference-optimization.md` and
  `../guides/inference-optimization-quickstart.md`, which live under `guides/features/` and
  `guides/quick-reference/` respectively.

**Dropped (would mislead):** Usage Patterns 1–4 (an API server, document processing, timeout
batching, batching with KV cache). Each fed padded sequences one at a time through a model with no
padding mask, which changes the model's output, and some used signatures that don't match the
current code. Also dropped: the left-padding, priority-batching and adaptive-batching "Advanced
Topics" (user code, not part of this file), the batch-size/latency guidance, and the out-of-memory
troubleshooting (no batched memory use exists to tune).

**Kept and merged:** field-by-field `TokenBatch` description and examples, `create_batch()`
behaviour and example, the dynamic-batching rationale (corrected numbers), the
`create_padding_mask()` example, `unbatch_outputs()` description, the `length_tolerance` tuning
sweep (rewritten to use `adai::Logger`), and the high-padding-ratio troubleshooting advice.
