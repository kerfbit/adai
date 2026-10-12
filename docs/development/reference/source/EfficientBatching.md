# `EfficientBatching` — Source File Reference (Retired)

> **Retired October 11, 2026 (TD-355).** `src/EfficientBatching.hpp`, `tests/datapipeline_test.cpp`,
> `examples/DataPipelineExample.cpp` and `benchmarks/AugmentationBenchmark.cpp` were deleted, and
> `BatchedInferenceConfig`'s dead `padding_strategy` field went with them. TD-352–TD-355 are closed in
> [TECHNICAL_DEBT_RESOLVED.md](../../archive/TECHNICAL_DEBT_RESOLVED.md). The token-budget bucketing
> algorithm is carried into the real-batching v2 plan as a design note
> ([real_batching_v2_plan.md](../../../proposals/real_batching_v2_plan.md) §4.1); the augmentation
> operations are recorded under TD-014. Last version: `git show e8f3580:src/EfficientBatching.hpp`.
> The sections below describe the code as it was.

- **File:** `src/EfficientBatching.hpp` (removed), header-only
- **Namespace:** global (no namespace)
- **Built into:** nothing in production. `BatchedInferenceEngine.hpp` included it only for the `PaddingStrategy` enum of its never-read `padding_strategy` config field (TD-211); real use was limited to tests, `examples/DataPipelineExample.cpp` and `benchmarks/AugmentationBenchmark.cpp`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** `tests/datapipeline_test.cpp` (removed; 15 `EfficientBatchingTest` cases)
- **Related guides:** [AUGMENTATION_IMPLEMENTATION.md](../../guides/features/AUGMENTATION_IMPLEMENTATION.md), [AUGMENTATION_QUICK_REFERENCE.md](../../guides/quick-reference/AUGMENTATION_QUICK_REFERENCE.md) (OpenMP augmentation), [data-pipeline-enhancement.md](../../guides/training/data-pipeline-enhancement.md)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by compiling the header into
> scratch programs: sequential, OpenMP (`-DADAI_ENABLE_OPENMP`) and AddressSanitizer builds.

---

## 1. What this file is

A second, independent token-batching toolkit, alongside [`BatchProcessor.hpp`](BatchProcessor.md).
It has its own `SequenceBatch`/`BatchStatistics` types, parallel to `TokenBatch`/`BatchStats`:

| Piece | Purpose |
|---|---|
| `PaddingStrategy` | `LEFT`, `RIGHT`, `CENTER` |
| `SequenceBatch` | Padded sequences, per-sequence attention masks, original lengths, `max_length`, `pad_token_id`; `total_tokens()`/`padding_tokens()`/`padding_ratio()` |
| `BatchStatistics` | Counts, average batch size and length, padding ratio, `efficiency_score = 1 − padding_ratio` |
| `BucketConfig` | Bucket length boundaries, `max_tokens_per_batch` (4096), `shuffle_buckets` |
| `AugmentationConfig` | Token dropout, token masking (`mask_token_id = 3`), adjacent-token shuffle, `seed = 42` |
| `EfficientBatching` (static methods) | Dynamic and bucketed batch creation, augmentation, statistics, padding, masks |

### Why it matters (and doesn't)

Like `BatchProcessor`'s helpers, nothing in production consumes a padded multi-sequence batch: the
model has no batch dimension (TD-171), and `BatchedInferenceEngine` used it only for the enum of an
unused config field. Its "Priority 2" OpenMP augmentation was documented as a performance feature with
"zero accuracy impact – same results". It's still tagged `stable`. Its correctness issues (§4)
matter to the real-batching v2 plan and to anyone who adopts it from those guides.

---

## 2. API

| Method | Behaviour |
|---|---|
| `create_dynamic_batches(seqs, batch_size, pad = PAD, strategy = RIGHT, sort_by_length = true)` | Throws on empty input. Optionally sorts indices by length (`std::sort`, so equal-length order isn't preserved), then chunks them into groups of `batch_size` and pads each to its own max |
| `create_bucketed_batches(seqs, BucketConfig, pad, strategy)` | Assigns each sequence to the first bucket whose boundary is ≥ its length (overflow goes to the last bucket). Optionally shuffles each bucket with a **`std::random_device`-seeded** RNG. Fills batches until `max_len_in_batch × count` would exceed `max_tokens_per_batch` (TD-074 fixed the running-max check). A single over-limit sequence still gets its own batch |
| `apply_augmentation(seqs&, AugmentationConfig)` | In place, per sequence: token dropout (keeps the original if everything would drop), token masking, adjacent swaps. OpenMP-parallel when `ADAI_ENABLE_OPENMP` is defined, which `adai_core` sets **publicly**, so every consumer gets the parallel version |
| `calculate_statistics(batches)` | Totals and ratios over `SequenceBatch::total_tokens()`/`padding_tokens()` |
| `pad_sequence(seq, target, pad, strategy)` | Returns `seq` unchanged if it's already ≥ `target` |
| `create_attention_mask(orig_len, padded_len, strategy)` | 1 for real positions, 0 for padding |

Batches carry **no source indices**, so after sorting or bucketing a caller can't map results back
to inputs. That's the same order-loss hazard as `BatchProcessor`'s `create_dynamic_batches()`
(TD-218).

---

## 3. Augmentation details (verified)

> **Not reproducible, and not the same as the sequential version.** The OpenMP path seeds one
> `mt19937` per thread with `seed + thread_id` and uses `schedule(dynamic, 16)`, so which thread
> (and so which random stream) handles which sequence depends on thread count **and timing**. On
> 400 sequences with `seed = 42`:
>
> | Build / threads | Output hash | Masked tokens |
> |---|---|---|
> | Sequential | `bb52b539…` | 1153 |
> | OpenMP, 1 thread | `bb52b539…` (same) | 1153 |
> | OpenMP, 2 | `5e7d61d7…` | 1079 |
> | OpenMP, 4 | `c44dcbbf…` | 1102 |
> | OpenMP, 8 | `efee0421…` | 1108 |
> | OpenMP, 4 again | `7d86c4e7…` (differs from the first 4-thread run) | 1102 |
>
> So `seed` doesn't give reproducibility, and the guides' "zero accuracy impact – same results"
> is true only statistically. A fix would derive each sequence's RNG from `(seed, seq_idx)`.

> **"Masking" inserts end-of-sequence tokens (verified).** `mask_token_id` defaults to `3`, which is
> `SpecialTokenIDs::EOS`. The vocabulary has only PAD/UNK/BOS/EOS (TD-225) and **no MASK token**.
> With masking enabled, about 15% of tokens become `EOS`, which teaches a model to see (and emit)
> end-of-sequence mid-text.

---

## 4. Robustness (verified)

- **`create_dynamic_batches(…, batch_size = 0)` loops forever** (`i += 0`).
- **`create_attention_mask()` writes out of bounds** when `original_length > padded_length`.
  AddressSanitizer reported a heap-buffer-overflow WRITE for `(8, 4, RIGHT)`; `LEFT` uses a
  negative start index. `create_single_batch()` never triggers it, but the function is public.

By inspection:
- Bucket boundaries are assumed sorted ascending.
- `std::random_device` makes bucketed batches non-reproducible regardless of any seed.
- Buckets are emitted in boundary order (no cross-bucket shuffle).

---

## 5. Tests

`datapipeline_test.cpp` has 15 `EfficientBatchingTest` cases:
- padding in all three strategies;
- masks for `LEFT`/`RIGHT`;
- dynamic batches, basic and sorted;
- statistics;
- bucketed batches, including the TD-074 token-limit regression;
- dropout and masking;
- empty input, a single sequence, and the padding ratio.

**Not covered:**
- reproducibility across thread counts;
- what token masking actually inserts;
- `batch_size = 0`;
- out-of-range mask arguments.

---

## 6. Known gaps and gotchas (summary)

Each item was closed by the retirement (TD-355) and is archived in
[TECHNICAL_DEBT_RESOLVED.md](../../archive/TECHNICAL_DEBT_RESOLVED.md).

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| OpenMP augmentation isn't reproducible: output depends on thread count and timing, and differs from the sequential build (§3) | `seed` doesn't reproduce runs; guides overstate "same results" | **Verified** | [TD-352](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-352-parallel-data-augmentation-isnt-reproducible) |
| Token masking inserts `EOS` (`mask_token_id = 3`); no MASK token exists (§3) | Masking-augmented data teaches spurious end-of-sequence | **Verified** | [TD-353](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-353-token-masking-augmentation-inserts-eos-tokens) |
| `batch_size = 0` infinite loop; `create_attention_mask()` heap overflow for `original > padded` (§4) | Hang or memory corruption on bad arguments to public functions | **Verified** | [TD-354](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-354-efficientbatching-hangs-or-corrupts-memory-on-bad-arguments) |
| No production use; duplicates `BatchProcessor`'s types; included but unused by `BatchedInferenceEngine`; batches lose source order; `random_device` bucketing (§1–2, §4) | Two parallel batching APIs to maintain; misleading include; order-loss trap | By inspection | [TD-355](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-355-efficientbatching-is-an-unused-duplicate-of-batchprocessor) |
| Batch order loss (shared with `BatchProcessor`) | Already tracked | — | TD-218 |
