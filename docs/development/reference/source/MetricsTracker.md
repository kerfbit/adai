# `MetricsTracker` — Source File Reference

- **File:** [`src/MetricsTracker.hpp`](../../../../src/MetricsTracker.hpp) (header-only, 482 lines)
- **Namespace:** global (`MetricsTracker`, `EpochMetrics`)
- **Used by:** `ChatbotTrainer` (owns one, `metrics_tracker_`), `IncrementalTrainer` (CSV export and overfitting warning after each pass); `examples/EnhancedTrainingExample.cpp`
- **Status tag:** `@adai-status: stable` ("TD-169 — now genuinely wired into ChatbotTrainer/IncrementalTrainer"), `@adai-version: 1.0.1`, `@adai-reviewed: 2026-09-13`
- **Tests:** [`tests/metricstracker_test.cpp`](../../../../tests/metricstracker_test.cpp) (19 tests)
- **Guide:** [enhanced-training-pipeline.md](../../guides/training/enhanced-training-pipeline.md) §2 (usage examples)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the
> real header.

---

## 1. What this file is

An in-process, per-pass record of epoch-level training metrics with a few derived analyses. It's
independent of the metrics server: nothing here is pushed anywhere. `ChatbotTrainer::train()` records one
row per epoch, and after a successful pass `IncrementalTrainer` does two things with it:

```text
save_model(checkpoint)
trainer.export_metrics_csv(checkpoint + ".metrics.csv")   → CSV sidecar (deleted with the checkpoint)
if trainer.get_metrics_tracker().is_overfitting():          → "Possible overfitting detected this pass"
    warn via TrainerControlState (admin API log) or Logger
```

`EpochMetrics` holds `epoch`, train/validation loss and perplexity, learning rate, gradient norm and
duration (seconds).

---

## 2. API

| Member | Behaviour | Production caller |
|---|---|---|
| `MetricsTracker(smoothing_window = 3)` | — | `ChatbotTrainer` (default) |
| `record_epoch(epoch, train, val = 0, lr = 0, grad_norm = 0, secs = 0)` | Appends a row; perplexity = `exp(loss)` (0 for negative loss); **`val = 0` means "not computed"**; updates bests and recomputes both smoothed curves | `ChatbotTrainer::train()`, with the **0-based** epoch (the metrics server gets `epoch + 1`) |
| `is_overfitting(gap = 0.5)` | Final epoch only: `val − train > 0.5` nats, i.e. validation perplexity more than e^0.5 ≈ 1.65× training perplexity | `IncrementalTrainer` |
| `export_csv(path)` | Header plus one row per epoch; `std::cerr` and `false` if the file can't be opened | `IncrementalTrainer` (via `ChatbotTrainer::export_metrics_csv()`) |
| `get_history()`, `get_epoch_metrics(e)` (a default row if absent), `size()`, `clear()` | Accessors | — |
| `get_best_*()`, `get_smoothed_*()` | Bests (`FLT_MAX` until set) and moving averages | — |
| `calculate_improvement_rate()` | `(first − last) / first × 100` on training loss | Example only |
| `is_converging(window = 5, threshold = 0.001)` | Standard deviation of the last `window` training losses below `threshold` | Example only |
| `print_summary()`, `print_history()` | Box-drawn tables to `std::cout` | Example only (`ChatbotTrainer` deliberately doesn't call them) |

---

## 3. Gaps (verified unless noted)

None of these affects training; they affect the diagnostics' accuracy and the header's hygiene.

- **The printers change `std::cout` for everyone.** `print_summary()`/`print_history()` set `std::fixed` and
  `setprecision(4)` and never restore them: `0.123456789` printed as `0.123457` before `print_summary()` and
  `0.1235` after. Any later output in the process inherits it.
- **`is_converging()` needs one more epoch than its window.** It returns `false` unless there are `window + 1`
  epochs: five identical epochs with `window = 5` reported "not converging".
- **A zero smoothing window gives `NaN`.** `MetricsTracker(0)` produced a smoothed loss of `-nan` (`0 / 0`);
  a negative window wraps through the `int`/`size_t` comparison. Nothing validates it (production uses 3).
- **The validation curve is misaligned.** Epochs with `val = 0` are skipped when smoothing, so
  `get_smoothed_validation_loss()` has fewer entries than there are epochs (2 for 3) and its indexes no
  longer match epoch numbers.
- **The overfitting check is coarse (by inspection).** It looks only at the final epoch, uses a fixed 1.65×
  perplexity gap, and inherits the validation set's noise and TD-267's low bias (skipped pairs counted as
  loss 0). It's a reasonable heuristic for a warning, not a measurement.
- **Hygiene (by inspection).** `calculate_moving_average()` is never called; the header uses an `#ifndef`
  guard rather than `#pragma once` and writes to `std::cout`/`std::cerr` (CLAUDE.md's conventions); CSV
  epochs are 0-based while the metrics server's are 1-based.

---

## 4. Tests

`metricstracker_test.cpp` (19 tests): construction, recording, perplexity, best tracking, improvement rate,
convergence and overfitting (positive and negative), CSV export, smoothing, `clear()`, a missing epoch, the two
printers, no validation loss, zero loss.

**Not covered:** the `std::cout` state change, the convergence off-by-one, a zero or negative window.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Printers leave `std::cout` in fixed 4-decimal mode; `is_converging()` off by one; window ≤ 0 gives NaN; misaligned validation curve (§3) | Wrong console formatting after a summary; misleading example output | **Verified** | [TD-447](../../guides/TECHNICAL_DEBT.md#td-447-metricstracker-printer-state-leak-convergence-off-by-one-and-smoothing-gaps) |
| Coarse final-epoch overfitting check; dead `calculate_moving_average()`; `#ifndef` guard and console output; 0- vs 1-based epochs (§3) | Hygiene; a noisy warning | By inspection | [TD-447](../../guides/TECHNICAL_DEBT.md#td-447-metricstracker-printer-state-leak-convergence-off-by-one-and-smoothing-gaps) |
| Validation loss biased low by skipped pairs | Already tracked | — | TD-267 |
