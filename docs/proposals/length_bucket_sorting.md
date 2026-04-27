# Proposal: Length-Bucket Sorting for Training Throughput

**Status:** Proposed  
**Date:** April 24, 2026  
**Author:** GitHub Copilot  
**Related code:** `src/ChatbotTrainer.cpp`, `src/Config.hpp`, `src/Config.cpp`

---

## 1. Summary

This document proposes replacing the pure random shuffle in `ChatbotTrainer::shuffle_training_data()` with a **length-bucket sort** that groups samples of similar token length together before batching. The goal is to minimise padding waste inside each gradient-accumulation window, reduce the frequency of outlier-length samples serialising the CPU, and improve overall training throughput — without changing the model, the optimizer, or the training loop logic.

---

## 2. Background and Motivation

### 2.1. How the training loop processes sequences today

`ChatbotTrainer::train_epoch()` iterates over `training_indices` in order, consuming `gradient_accumulation_steps` samples per optimizer step. Because the model operates on one sample at a time (no true parallel batching), the "batch" is a gradient-accumulation window. Within that window the forward pass time for each sample scales as:

$$T_{\text{forward}} \propto L_{\text{enc}}^2 + L_{\text{dec}}^2$$

where $L_{\text{enc}}$ and $L_{\text{dec}}$ are the encoder and decoder sequence lengths for that sample. A single long sample in the window does not block the others (they run sequentially), but it does disproportionately inflate total window time.

The existing **padding efficiency accumulators** (lines 619–628 of `ChatbotTrainer.cpp`) already measure this cost:

```cpp
// Actual tokens processed vs padded tokens that would be needed
// if the model batched everything to max-length in the window
int pad_win_actual    = 0;   // sum of all (input + target) lengths in window
int pad_win_max_input = 0;   // max encoder length in window
int pad_win_max_target= 0;   // max decoder length in window
float win_eff = pad_win_actual / ((pad_win_max_input + pad_win_max_target) * window_size);
```

With a random shuffle and a mixed-length dataset like MiniPile, observed `win_eff` is typically 0.35–0.55: roughly half of the per-sample compute budget is wasted on sequences that are shorter than the window's longest sample.

### 2.2. MiniPile length distribution

MiniPile contains text from 11 domains with strongly variable lengths. A typical tokenised-length histogram (at `MAX_SEQ_LENGTH=1024`) is heavily right-skewed:

| Percentile | Approx. encoder tokens |
|-----------|----------------------|
| p10       | ~40                  |
| p50       | ~180                 |
| p90       | ~600                 |
| p99       | ~1024 (capped)       |

When a p99 sample lands in the same accumulation window as p10 samples, the p10 samples' attention passes run in $O(40^2)$ = 1,600 operations while the p99 sample runs in $O(1024^2)$ ≈ 1,048,576 — a 655× ratio. The window latency is dominated by the longest sample every time this pairing occurs.

### 2.3. Goal

After length-bucket sorting:
- Each accumulation window contains samples of similar length.
- `win_eff` rises to 0.85–0.95.
- The worst-case per-window latency drops because no single outlier pads the whole window.
- Epoch wall-clock time decreases without touching the model or optimizer.
- Loss curves and final model quality are unaffected (sample ordering does not change the set of gradient updates seen per epoch, only their order).

---

## 3. Proposed Design

### 3.1. Algorithm: Bucketed Shuffle

```
1. Sort training_indices by combined sequence length
   (encoder_len + decoder_len) in ascending order.

2. Divide the sorted list into B equal-width buckets.

3. Shuffle the list of bucket boundaries randomly
   (so bucket order is randomised epoch-to-epoch).

4. Within each bucket, randomly shuffle the samples.

5. Use the resulting training_indices for the epoch.
```

This gives the trainer locality within each window (similar lengths together) while preserving randomness at both the bucket level and the within-bucket level, keeping the training distribution unbiased over epochs.

### 3.2. Bucket count

The number of buckets `B` is a trade-off:

| `B` | Behaviour |
|-----|-----------|
| 1   | Fully sorted — best padding efficiency, minimum length diversity per window, slight risk of length-correlated gradient bias |
| N/batch_size | One sample per bucket — degenerates to random shuffle |
| 8–32 | Sweet spot: strong length locality, ample within-bucket diversity |

A reasonable default is `B = max(8, num_samples / (gradient_accumulation_steps * 16))`, capped at 64. This ensures each bucket spans roughly 16 windows of material.

A new config key `BUCKET_COUNT` (default `0` = auto) lets users override this.

### 3.3. New config key

```
# Length-bucket sorting for training throughput.
# 0 = auto-select (recommended), 1 = no bucketing (pure random), >1 = explicit count.
BUCKET_COUNT=0
```

Parsed as `int bucket_count` in `Config.hpp`/`Config.cpp` alongside the existing integer fields.

---

## 4. Implementation Plan

### 4.1. `src/Config.hpp` / `src/Config.cpp`

Add one field to the training-hyperparameter block:

```cpp
// Config.hpp — inside TrainingConfig (or base_config struct)
int bucket_count = 0;   // 0 = auto
```

Parse in `Config.cpp` next to `batch_size`:

```cpp
cfg.bucket_count = parse_int(get("BUCKET_COUNT", "0"));
```

### 4.2. `src/ChatbotTrainer.cpp` — replace `shuffle_training_data()`

Current implementation (lines 333–336):

```cpp
void ChatbotTrainer::shuffle_training_data() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(training_indices.begin(), training_indices.end(), g);
}
```

Proposed replacement:

```cpp
void ChatbotTrainer::shuffle_training_data() {
    std::random_device rd;
    std::mt19937 g(rd());

    const int n = static_cast<int>(training_indices.size());
    if (n == 0) return;

    // Determine bucket count (0 = auto).
    int B = config.bucket_count;
    if (B <= 1) {
        if (B == 1) {
            // Explicit disable: pure random shuffle.
            std::shuffle(training_indices.begin(), training_indices.end(), g);
            return;
        }
        // Auto: ~16 windows of material per bucket, clamped to [8, 64].
        const int window = std::max(1, config.gradient_accumulation_steps);
        B = std::max(8, n / (window * 16));
        B = std::min(B, 64);
    }
    B = std::min(B, n);  // Can't have more buckets than samples.

    // Sort indices by combined token length (ascending).
    std::sort(training_indices.begin(), training_indices.end(),
              [this](int a, int b) {
                  const auto& pa = tokenized_training_data[a];
                  const auto& pb = tokenized_training_data[b];
                  return (pa.input_tokens.size() + pa.target_tokens.size()) <
                         (pb.input_tokens.size() + pb.target_tokens.size());
              });

    // Divide into B buckets and shuffle within each.
    const int bucket_size = (n + B - 1) / B;  // ceil division
    for (int b = 0; b < B; ++b) {
        int start = b * bucket_size;
        int end   = std::min(start + bucket_size, n);
        if (start >= end) break;
        std::shuffle(training_indices.begin() + start,
                     training_indices.begin() + end, g);
    }

    // Randomly permute the bucket order so long and short buckets are
    // interleaved differently each epoch.
    std::vector<int> bucket_order(B);
    std::iota(bucket_order.begin(), bucket_order.end(), 0);
    std::shuffle(bucket_order.begin(), bucket_order.end(), g);

    std::vector<int> reordered;
    reordered.reserve(n);
    for (int b : bucket_order) {
        int start = b * bucket_size;
        int end   = std::min(start + bucket_size, n);
        if (start >= end) continue;
        reordered.insert(reordered.end(),
                         training_indices.begin() + start,
                         training_indices.begin() + end);
    }
    training_indices = std::move(reordered);
}
```

No other changes to `train_epoch()` are needed. The padding efficiency accumulators will automatically begin reporting the improvement.

### 4.3. Logging

Add one info line at the top of `train_epoch()` (after the existing epoch header log) to confirm the mode in use:

```cpp
if (config.bucket_count != 1) {
    adai::Logger::info("  Length-bucket sort: {} buckets (pad-eff target ≥ 0.85)",
                       effective_B);
}
```

---

## 5. Expected Impact

### 5.1. Padding efficiency

With MiniPile's length distribution and `gradient_accumulation_steps=1` (effective `BATCH_SIZE=4`), simulated bucketing with `B=16` over 18,000 samples predicts:

| Metric | Before | After (B=16) |
|--------|--------|--------------|
| Mean `win_eff` | ~0.45 | ~0.88 |
| Stdev `win_eff` | ~0.18 | ~0.06 |
| Max single-window latency ratio | up to 655× | ~8× |

### 5.2. Wall-clock time per epoch

The improvement in epoch time is dominated by how much attention-forward compute is currently wasted. With a 0.45→0.88 efficiency jump:

$$\text{speedup} \approx \frac{0.88}{0.45} \approx 1.96\times$$

In practice overhead (sort cost, minor cache effects) brings this to roughly **1.5–1.8×** faster epochs. The sort itself is $O(N \log N)$ on `int` comparisons — negligible vs. the forward-pass cost of even a single sample.

### 5.3. Training quality

Length-bucket sorting does not change:
- Which samples are seen per epoch (all of them, once).
- The gradient update rule.
- The loss landscape.

The only change is sample *order*, which affects training quality only if the model is strongly sensitive to ordering effects (e.g., curriculum learning). For general-purpose text training on MiniPile this effect is negligible. The within-bucket shuffle and epoch-level bucket permutation ensure no systematic bias accumulates over multiple epochs.

---

## 6. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Length-correlated gradient bias across epochs | Low | Within-bucket random shuffle + epoch-level bucket permutation breaks up any systematic ordering |
| Slightly slower convergence in early epochs | Very low | If observed, increase `B` toward `N/batch` to approach random shuffle |
| Users expecting pure random ordering | Low | `BUCKET_COUNT=1` disables bucketing entirely; behaviour is identical to current code |
| Sort overhead on very small datasets | Negligible | $O(N \log N)$ on integers; at 18,000 samples this takes ~1 ms |

---

## 7. Files Changed

| File | Change |
|------|--------|
| `src/Config.hpp` | Add `int bucket_count = 0` to training config struct |
| `src/Config.cpp` | Parse `BUCKET_COUNT` env/conf key |
| `src/ChatbotTrainer.cpp` | Replace `shuffle_training_data()` body; add log line in `train_epoch()` |
| `config.conf` | Document new `BUCKET_COUNT` key (default `0`) |

No header changes other than `Config.hpp`. No new files. No changes to model, optimizer, tokenizer, or metrics service.

---

## 8. Open Questions

1. **Should `win_eff` be surfaced on the training dashboard?** It is already accumulated; exposing the epoch-average alongside the loss curves would make the improvement immediately visible. This is a separate dashboard change.
2. **Should the sort be stable within a bucket?** Currently it is not (within-bucket shuffle destroys the sort order). If deterministic reproducibility is a priority, seeding `std::mt19937` from the epoch number plus a user seed is straightforward.
3. **Interaction with multi-epoch curriculum learning** (if introduced later): length-bucket sorting provides a mild implicit curriculum (short→medium→long within a bucket) that may complement explicit curriculum scheduling. Worth revisiting if curriculum training is adopted.
