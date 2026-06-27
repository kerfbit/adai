# Proposal: Adaptive Gradient Clipping for the Incremental Trainer

**Status:** Proposed  
**Date:** April 19, 2026  
**Author:** GitHub Copilot  
**Related code:** `src/ChatbotTrainer.cpp`, `src/Config.hpp`, `src/Config.cpp`, `src/Optimizer.hpp`, `src/Optimizer.cpp`

---

## 1. Summary

This document proposes replacing the fixed `GRADIENT_CLIP` scalar in the incremental trainer with an adaptive gradient clipping mechanism that maintains a running estimate of the gradient norm distribution and adjusts the clip threshold dynamically within user-defined bounds.

The motivation is directly observable in session 5: every optimizer step is being clipped because the static threshold (0.5) is far below the true gradient norms (raw norms of 2–23 measured in epoch 4). This wastes gradient information without improving stability. An adaptive scheme would tighten the threshold when norms are low and stable, and relax it when norms are high but consistent, while hard bounds prevent runaway clipping or uncontrolled gradient flow.

The proposed changes touch only the training infrastructure — no model architecture changes are required.

---

## 2. Background and Motivation

### 2.1. Current Behavior

`GRADIENT_CLIP` is a single float read from `config.conf` (or the `GRADIENT_CLIP` environment variable) in `Config.cpp` and copied verbatim into `ChatbotTrainer`'s inner loop:

```cpp
// ChatbotTrainer.cpp ~842
if (config.gradient_clip_norm > 0.0f) {
    optimizer->clip_gradients(config.gradient_clip_norm, grad_norm);
}
```

There is no feedback. The threshold is fixed for the entire run, regardless of how the gradient norm distribution evolves across epochs or training phases.

### 2.2. Observed Problem in Session 5

| Epoch | Epoch-avg grad norm | `GRADIENT_CLIP` | Clipped? |
|-------|--------------|-----------------|----------|
| 1 | 0.5000 | 0.5 | Every step |
| 2 | 0.5001 | 0.5 | Every step |
| 3 | 0.4998 | 0.5 | Every step |
| 4 | 0.500 (avg of 2–23) | 0.5 | Every step |

The epoch-average appearing exactly at the clip ceiling (0.500) is diagnostic — it means _no unclipped step has ever been recorded_. Per-step norms measured via `/api/metrics/history` for epoch 4 ranged from 2.2 to 23.0. The clip threshold is therefore between 4× and 46× smaller than the actual gradient norms at every single optimizer step.

Consequences:
- **Information loss.** The gradient direction is preserved (clipping scales the whole vector) but the step magnitude is artificially constrained. When raw norms are 23 and the threshold is 0.5, the optimizer moves 46× less than the loss surface actually calls for.
- **Masked divergence signals.** Outlier spikes (e.g. the 23-norm at sample 700) may indicate genuinely problematic samples. Clipping them silently makes it impossible to distinguish healthy large-norm samples from real instabilities.
- **Wasted compute.** The Welford-online gradient variance accumulator (TD-013) is tracking variance in already-clipped norms, not raw norms, which means its output underestimates true gradient variability.

### 2.3. Goal

Provide a clip threshold that:
1. Stays within hard user-defined bounds (`GRADIENT_CLIP_MIN`, `GRADIENT_CLIP_MAX`).
2. Adapts toward the recent gradient norm distribution automatically so that roughly a target fraction of steps are actually clipped.
3. Reacts quickly to sudden spikes (loss of convergence) and slowly relaxes when conditions stabilize.
4. Requires zero change to the model or optimizer internals — only the value passed to `clip_gradients()` changes.

---

## 3. Algorithm Design

### 3.1. Exponential Moving Average with Percentile Target

The core idea: maintain an exponential moving average (EMA) of the observed gradient norm and set the clip threshold to a percentile-derived multiple of that EMA. A `target_clip_fraction` parameter controls what fraction of steps are allowed to be clipped.

At each optimizer step:

```
ema_norm ← α × raw_norm + (1 − α) × ema_norm
candidate_threshold ← ema_norm × headroom_factor
effective_threshold ← clamp(candidate_threshold, GRADIENT_CLIP_MIN, GRADIENT_CLIP_MAX)
```

Where:
- `α` (EMA decay, default 0.05): controls how quickly the estimate tracks changes. Smaller values are more stable but lag behind sudden shifts.
- `headroom_factor` (default 2.0): multiplier that determines how much above the mean norm the threshold sits. A value of 2.0 means the threshold is twice the current EMA norm estimate — under a roughly log-normal gradient norm distribution this clips approximately the top 15% of steps.
- `GRADIENT_CLIP_MIN` and `GRADIENT_CLIP_MAX` are hard floor and ceiling values set in `config.conf`.

### 3.2. Warm-Up Period

For the first `warmup_steps` optimizer steps the system collects gradient norms without clipping (or clips at `GRADIENT_CLIP_MAX`). This seeds the EMA with a meaningful initial estimate before the adaptive logic takes over. Suggested default: 100 steps, or 1% of total training steps, whichever is larger.

### 3.3. Spike Protection

If `raw_norm > k × ema_norm` the step is considered an outlier spike. In this case:
- The EMA update is suppressed for that step (the spike does not permanently inflate the estimate).
- The clip threshold is not relaxed for that step.
- An optional counter `adaptive_clip_spike_count` is incremented and exposed via the metrics API.

Suggested default: `k = 5.0`.

### 3.4. Stability Mode

When the gradient norm variance (already tracked by the Welford accumulator in TD-013) falls below a threshold, the system enters _stability mode_: `α` is halved and `headroom_factor` is reduced by 10% toward a tighter bound. This allows the clip threshold to converge toward the minimum headroom once training stabilizes, which is typical behavior in the final epochs of a run.

---

## 4. Configuration Parameters

All new keys are optional and backward-compatible. If none are set, the adaptive system mirrors fixed-clip behavior (`GRADIENT_CLIP_MIN == GRADIENT_CLIP_MAX == GRADIENT_CLIP`).

| `config.conf` key | Type | Default | Description |
|---|---|---|---|
| `GRADIENT_CLIP_MIN` | float | 0.1 | Hard floor — threshold never drops below this |
| `GRADIENT_CLIP_MAX` | float | 5.0 | Hard ceiling — threshold never rises above this |
| `GRADIENT_CLIP_EMA_DECAY` | float | 0.05 | EMA smoothing factor α (0 < α ≤ 1) |
| `GRADIENT_CLIP_HEADROOM` | float | 2.0 | Threshold = ema_norm × headroom |
| `GRADIENT_CLIP_WARMUP_STEPS` | int | 100 | Steps before adaptive logic activates |
| `GRADIENT_CLIP_SPIKE_K` | float | 5.0 | Outlier suppression: norms > k×ema are not fed into EMA |
| `GRADIENT_CLIP_ADAPTIVE` | bool | false | Master switch; false retains legacy fixed-clip behavior |

The existing `GRADIENT_CLIP` key is still parsed. When `GRADIENT_CLIP_ADAPTIVE=false` it operates exactly as before. When `GRADIENT_CLIP_ADAPTIVE=true` it is used as the initial seed value for the EMA (rather than `ema_norm = 0`) and as both `GRADIENT_CLIP_MIN` and `GRADIENT_CLIP_MAX` if those are not set separately.

---

## 5. Proposed Code Changes

### 5.1. `src/Config.hpp` — New fields in `ServiceConfig`

```cpp
// Adaptive gradient clipping (optional; falls back to fixed GRADIENT_CLIP when disabled)
bool   adaptive_gradient_clip         = false;
float  gradient_clip_min              = 0.1f;
float  gradient_clip_max              = 5.0f;
float  gradient_clip_ema_decay        = 0.05f;
float  gradient_clip_headroom         = 2.0f;
int    gradient_clip_warmup_steps     = 100;
float  gradient_clip_spike_k          = 5.0f;
```

And `ChatbotTrainerConfig` (also in `ChatbotTrainer.hpp`) gains a mirrored set to be populated by `IncrementalTrainer` just as `gradient_clip_norm` is today.

### 5.2. `src/Config.cpp` — Key parsing

Extend the `if/else` chain in the config file parser and the `get_env_*` overrides block:

```cpp
} else if (key == "GRADIENT_CLIP_ADAPTIVE") {
    config.adaptive_gradient_clip = (value == "true" || value == "1");
} else if (key == "GRADIENT_CLIP_MIN") {
    config.gradient_clip_min = std::stof(value);
} else if (key == "GRADIENT_CLIP_MAX") {
    config.gradient_clip_max = std::stof(value);
} else if (key == "GRADIENT_CLIP_EMA_DECAY") {
    config.gradient_clip_ema_decay = std::stof(value);
} else if (key == "GRADIENT_CLIP_HEADROOM") {
    config.gradient_clip_headroom = std::stof(value);
} else if (key == "GRADIENT_CLIP_WARMUP_STEPS") {
    config.gradient_clip_warmup_steps = std::stoi(value);
} else if (key == "GRADIENT_CLIP_SPIKE_K") {
    config.gradient_clip_spike_k = std::stof(value);
}
```

### 5.3. `src/ChatbotTrainer.cpp` — Adaptive state and per-step threshold

Alongside the existing Welford accumulators, add adaptive clipping state at the top of `train_epoch()`:

```cpp
// Adaptive gradient clipping state
float agc_ema         = config.gradient_clip_norm;   // seed with fixed value
int   agc_step_count  = 0;
int   agc_spike_count = 0;
bool  agc_active      = config.adaptive_gradient_clip;
```

Replace the clipping call with:

```cpp
float effective_clip = config.gradient_clip_norm;   // fallback: legacy fixed clip

if (agc_active) {
    ++agc_step_count;
    bool in_warmup = (agc_step_count <= config.gradient_clip_warmup_steps);

    if (!in_warmup) {
        bool is_spike = (grad_norm > config.gradient_clip_spike_k * agc_ema);
        if (!is_spike) {
            // Update EMA only on non-spike steps
            agc_ema = config.gradient_clip_ema_decay * grad_norm
                    + (1.0f - config.gradient_clip_ema_decay) * agc_ema;
        } else {
            ++agc_spike_count;
        }
        // Compute candidate threshold
        float candidate = agc_ema * config.gradient_clip_headroom;
        effective_clip  = std::clamp(candidate,
                                     config.gradient_clip_min,
                                     config.gradient_clip_max);
    } else {
        // Warmup: clip at ceiling to protect, but still update EMA
        agc_ema       = config.gradient_clip_ema_decay * grad_norm
                      + (1.0f - config.gradient_clip_ema_decay) * agc_ema;
        effective_clip = config.gradient_clip_max;
    }
}

if (effective_clip > 0.0f) {
    optimizer->clip_gradients(effective_clip, grad_norm);
}
```

After the gradient accumulation window (where `update_count` is incremented), push the current `effective_clip` and `agc_spike_count` into the metrics service so they appear in the dashboard and the JSONL record.

### 5.4. `src/TrainingMetricsService.hpp` / `.cpp` — New fields

Add to the per-step advanced metrics update path:
- `float adaptive_clip_threshold` — the threshold actually used this step.
- `int   adaptive_clip_spike_count` — cumulative spikes since epoch start.

These are low-overhead additions: one float and one int appended to what `update_advanced_epoch_metrics()` already receives.

### 5.5. `src/TrainingMetricsAPI.cpp` — Epoch summary endpoint

Add `epoch_adaptive_clip_thresholds` (vector of per-epoch mean effective threshold) alongside the existing `epoch_gradient_norms` array in `handle_epoch_metrics()`. This allows the dashboard to plot the threshold trajectory across epochs.

### 5.6. `dashboard.html` — Visualization

Add a **Gradient Clipping** panel to the dashboard:
- Line chart: epoch-average adaptive clip threshold over epochs.
- Overlaid: epoch-average raw gradient norm.
- Indicator: spike count per epoch.

This makes it immediately obvious whether the threshold is tracking below, at, or above the norm distribution.

---

## 6. Example Configuration (Session 6)

Based on the observed session 5 gradient norms (epoch-level 2–23, mostly 3–8), the following settings would be a reasonable starting point:

```ini
# config.conf
GRADIENT_CLIP=0.5               # legacy key; used as EMA seed
GRADIENT_CLIP_ADAPTIVE=true
GRADIENT_CLIP_MIN=0.5           # never clip more aggressively than current session
GRADIENT_CLIP_MAX=4.0           # prevent runaway on true divergence spikes
GRADIENT_CLIP_EMA_DECAY=0.05    # ~20-step effective window
GRADIENT_CLIP_HEADROOM=2.0      # threshold = 2× mean norm
GRADIENT_CLIP_WARMUP_STEPS=100
GRADIENT_CLIP_SPIKE_K=5.0
```

With these settings and a seed EMA of 0.5, after 100 warmup steps the EMA will have risen to approximately the true mean norm (~5–7 based on session 5 data) and the effective threshold will settle around 10–14 — well above the actual norms on typical steps, clipping only the genuine outlier spikes.

---

## 7. Implementation Order

1. **Config parsing** — add fields to `Config.hpp` and `Config.cpp`. No behavioral change; safe to land first.
2. **Adaptive state in `ChatbotTrainer`** — implement the EMA + clamp logic behind the `adaptive_gradient_clip` flag. Full backward compatibility: existing runs with `GRADIENT_CLIP_ADAPTIVE=false` (the default) are unchanged.
3. **Metrics plumbing** — pass `effective_clip` and spike count through `TrainingMetricsService` to JSONL and the epoch API.
4. **Dashboard panel** — add the clipping chart to `dashboard.html`.
5. **Validation** — run a short test session (3–5 epochs) with `GRADIENT_CLIP_ADAPTIVE=true` and compare loss curves to a fixed-clip baseline.

---

## 8. Testing Considerations

- **Unit tests**: Add a test in `tests/` that drives the EMA logic directly with a synthetic sequence of gradient norms (including spike cases) and asserts that `effective_clip` stays within `[GRADIENT_CLIP_MIN, GRADIENT_CLIP_MAX]` and that spike norms do not contaminate the EMA.
- **Regression test**: Verify that with `GRADIENT_CLIP_ADAPTIVE=false` the behavior is bit-for-bit identical to the current fixed-clip path.
- **Integration test**: A full 1-epoch training run with both modes; assert that training loss is finite and below the starting loss.

---

## 9. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| EMA seeds too low, threshold overshoots during warmup | Medium | Warmup path clips at `GRADIENT_CLIP_MAX`; EMA seeds from `GRADIENT_CLIP` which is already a sensible human value |
| Spike suppression masks genuine divergence | Low-medium | `agc_spike_count` is exposed via metrics; users can monitor it and reduce `GRADIENT_CLIP_MAX` if needed |
| Adaptive threshold introduces non-determinism across runs | Low | Determinism was already broken by ordered-map iteration in the optimizer; this adds no new source of randomness |
| Stability mode logic interacts poorly with LR schedule | Low | Stability mode only reduces `headroom_factor` by capping it; it never touches the LR path |

---

## 10. Alternatives Considered

- **Raise `GRADIENT_CLIP` to 2.0 (static)**: Simple single-line change. Would immediately help session 6 but still requires manual re-tuning for every new dataset with different norm characteristics. This is a valid short-term fix to do _while_ this proposal is being implemented.
- **Per-layer clipping**: Clip each parameter group independently by its own norm rather than the global norm. More targeted but dramatically increases implementation surface and changes optimizer behavior in ways that are harder to reason about.
- **Loss-spike-triggered clip relaxation**: Only relax the clip threshold in response to a loss spike rather than continuously. Simpler to reason about but introduces discontinuous threshold jumps that can destabilize the optimizer if the spike detector fires on a noisy batch.
- **No clipping at all**: Setting `GRADIENT_CLIP=0` disables clipping entirely. Viable if the model proves stable, but risky with the current large per-step norm variance (up to 46×).
