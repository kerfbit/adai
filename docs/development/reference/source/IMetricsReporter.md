# `IMetricsReporter` — Source File Reference

- **File:** [`src/IMetricsReporter.hpp`](../../../../src/IMetricsReporter.hpp) (header-only, 224 lines)
- **Namespace:** global (`AbnormalSample`, `IMetricsReporter`, `NullMetricsReporter`)
- **Built into:** header-only; used by `ChatbotTrainer`, `IncrementalTrainer`, `IncrementalTrainingTool` (LeJEPA path), `MetricsPushClient` (the real implementation), and the metrics databases (for `AbnormalSample`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.1`, `@adai-reviewed: 2026-09-23`
- **Tests:** no dedicated test file. `tests/generation_quality_async_test.cpp` defines a `RecordingMetricsReporter`; the metrics-database tests use `AbnormalSample`
- **Last traced against the code:** 2026-10-13

> Code-traced reference. The cost of the always-non-null reporter (§3) was measured with a small
> `ChatbotTrainer` run in a scratch program.

---

## 1. What this file is

The **trainer-side metrics interface** (TD-021). `ChatbotTrainer` and the LeJEPA pretraining loop
report epoch, per-sample and diagnostic values through it without depending on the HTTP client or
the server-side `TrainingMetricsService`:

| Type | Purpose |
|---|---|
| `AbnormalSample` | One outlier sample (TD-013): epoch, sample id, loss, gradient norm, reason, **full input and target text**, timestamp. Its canonical definition is here so both the trainer and the server headers can use it |
| `IMetricsReporter` | 16 pure-virtual methods (below) |
| `NullMetricsReporter` | Final no-op implementation |

Session lifecycle (`start_session`/`end_session`/heartbeat) is deliberately **not** part of the
interface. `IncrementalTrainer` calls those on the concrete `MetricsPushClient`.

### Why it matters

Every training metric on the dashboards, the metrics DB and the ops app goes through these 16 calls.
`MetricsPushClient` turns each into a queued HTTP push to `metrics_api_server`.

---

## 2. The interface

| Method | Meaning | `MetricsPushClient` | Called by |
|---|---|---|---|
| `start_epoch(epoch, total_samples)` | Epoch begins | Pushed (`/epoch/start`); resets the buffers | `ChatbotTrainer`, LeJEPA loop |
| `end_epoch(epoch, loss, val_loss, lr, perplexity = 0, grad_norm = 0, secs = 0)` | Epoch aggregates (perplexity 0 = derive from loss) | Pushed (`/epoch/end`) **together with every buffered value below** | Both |
| `update_sample_metrics(sample, loss, grad_norm, lr)` | Per optimizer step | Pushed, lossy under backpressure | Both |
| `update_validation_metrics(loss, accuracy, perplexity)` | Validation results (accuracy −1 = not computed; `ChatbotTrainer` always passes −1) | Pushed | `ChatbotTrainer` |
| `update_best_metrics(loss, epoch)` | Best validation so far | Pushed | `ChatbotTrainer` |
| `update_advanced_epoch_metrics(var, compute_ratio, update_ratio)` | TD-013 diagnostics (running values per step, final at epoch end) | **Buffered** for `end_epoch()` **and** pushed every call (`/metrics/advanced`, lossy) | Both |
| `flag_abnormal_sample(sample)` | Outlier sample | Pushed (`/metrics/abnormal`), capped by `max_abnormal_samples` per epoch; the texts are **not truncated** | `ChatbotTrainer` |
| `update_adaptive_clip_metrics(threshold, spikes)` / `update_adaptive_clip_epoch(avg, total)` | Adaptive clipping (TD-017) | Buffered but **never sent**: `end_epoch()` omits it (see [MetricsPushClient.md](MetricsPushClient.md)) | `ChatbotTrainer` |
| `update_activation_saturation(ratio)`, `update_attention_entropy(h)` | Hook diagnostics (−1 = not computed) | **Buffered** | Both |
| `update_layer_gradient_norms(enc, dec)` | Per-layer gradient norms | Pushed | `ChatbotTrainer` |
| `update_padding_efficiency(eff)` | Padding efficiency | **Buffered** | `ChatbotTrainer` |
| `update_generation_quality_metrics(bleu4, r1, r2, rL)` | BLEU/ROUGE (TD-016). No `bleu1`/`bleu2` parameters (TD-379) | Pushed. **Can be called from `ChatbotTrainer`'s async scoring thread** | `ChatbotTrainer` |
| `update_lejepa_metrics(pred, sigreg)`, `update_lejepa_advanced_metrics(...)` | LeJEPA pretraining (TD-178) | **Buffered** | LeJEPA loop only |

**Thread safety.** `MetricsPushClient` pushes through a mutex-guarded queue, so the async BLEU/ROUGE
thread's call is safe. The buffered setters write plain members and are only called from the
training thread.

**An ordering contract the interface doesn't state.** In `MetricsPushClient`, the "buffered" methods
only store a value; it reaches the server inside the next `end_epoch()` payload, and `start_epoch()`
resets the buffers. So every buffered setter must be called **after `start_epoch()` and before
`end_epoch()`** of the same epoch, or its value lands in the wrong epoch or is lost. Both callers get
this right today. `IncrementalTrainingTool.cpp` documents the rule in a call-site comment ("These
buffered-field setters must all run BEFORE end_epoch()"), but the interface itself says nothing, and
nothing in the signatures shows which methods are immediate and which are buffered. A new
implementation or a reordered caller would break silently.

---

## 3. `NullMetricsReporter` and the always-true guard

`IncrementalTrainer` always hands `ChatbotTrainer` a reporter: a `MetricsPushClient` when a session
starts, otherwise a `NullMetricsReporter` (no server URL, or session start failed). But
`ChatbotTrainer` treats the pointer as an **"are metrics enabled?"** switch: ten
`if (metrics_reporter_)` blocks gate work whose only purpose is reporting:
- installing the activation-saturation and attention-entropy hooks on every layer (on GPU builds,
  the saturation hook synchronizes the queue per layer, TD-370);
- computing per-layer gradient norms;
- BLEU/ROUGE generation (when `enable_generation_quality_metrics` is on);
- assembling the per-epoch diagnostics.

Under `incremental_trainer` these always run, even when the results go to a no-op. On a tiny model the
cost was within noise (two epochs: 5.10 s with no reporter vs. 5.12 s with `NullMetricsReporter`,
5.21 s with BLEU enabled), so this is mainly a semantic trap. It will matter more with a large
model, BLEU enabled, or GPU hook synchronization. A `bool enabled() const` on the interface (false for
`NullMetricsReporter`) would let the trainer skip the work.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Buffered vs. immediately pushed methods, and the "between `start_epoch()` and `end_epoch()`" ordering rule, aren't documented on the interface (§2) | A new implementation or reordered caller silently loses or misattributes metrics | By inspection | [TD-399](../../guides/TECHNICAL_DEBT.md#td-399-imetricsreporter-buffering-and-ordering-contract-is-undocumented) |
| `ChatbotTrainer` uses `metrics_reporter_ != nullptr` as "metrics enabled", but `incremental_trainer` always passes a reporter (`NullMetricsReporter` when there's no server), so hooks, per-layer norms and BLEU generation run with nowhere to report (§3) | Wasted work; small at the tested size, larger on GPU or with BLEU | By inspection; cost **measured** (small) | [TD-400](../../guides/TECHNICAL_DEBT.md#td-400-nullmetricsreporter-keeps-reporting-only-work-switched-on) |
| `AbnormalSample` carries full, untruncated training texts into the push queue and the metrics DB (§2) | Large payloads and training data stored on the metrics server | By inspection | [TD-399](../../guides/TECHNICAL_DEBT.md#td-399-imetricsreporter-buffering-and-ordering-contract-is-undocumented) |
| No `bleu1`/`bleu2` in `update_generation_quality_metrics()` | Already tracked | — | TD-379 |
| No dedicated tests; only a recording test double | — | By inspection | — |
