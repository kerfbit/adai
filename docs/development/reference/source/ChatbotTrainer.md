# `ChatbotTrainer` — Source File Reference

- **Files:** [`src/ChatbotTrainer.hpp`](../../../../src/ChatbotTrainer.hpp), [`src/ChatbotTrainer.cpp`](../../../../src/ChatbotTrainer.cpp)
- **Built into:** compiled directly into `incremental_trainer`, `training_metrics_example` and `chatbottrainerTests` (listed in each target's sources); it isn't part of a static library
- **Status tag:** `@adai-status: beta` (capped by TD-039, "large, actively evolving core trainer"), `@adai-version: 0.12.0` (hpp) / `0.11.1` (cpp), `@adai-reviewed: 2026-10-19` (both)
- **Tests:** [`tests/chatbottrainer_test.cpp`](../../../../tests/chatbottrainer_test.cpp) → `chatbottrainerTests` (83 tests; `friend class ChatbotTrainerCacheTest` for the tokenized cache); test plan in [chatbot-trainer-tests.md](../../testing/chatbot-trainer-tests.md). `tests/chatbottrainer_test.cpp.old` is an unbuilt January-2026 leftover
- **Related guide:** [enhanced-training-pipeline.md](../../guides/training/enhanced-training-pipeline.md) (multi-component; partly stale per its own banner)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. The empty-pair findings in §6 and §9 were confirmed by running a real
> training pass (a tiny model built from `ChatbotTrainer.cpp` and the debug libraries); the rest is
> from reading the code and its callers.

---

## 1. What this file is

`ChatbotTrainer` runs **one supervised training pass** of an `EncoderDecoderModel` over
(input → response) text pairs. In order, it:

1. holds the raw pairs and optionally splits off validation data;
2. tokenizes everything (in parallel, with an optional on-disk cache);
3. runs epochs of per-sample forward and backward passes with gradient accumulation, an LR
   schedule, fixed or adaptive gradient clipping, and the AdamW/Adam/SGD optimizer;
4. validates, tracks the best model, and optionally stops early;
5. streams per-sample and per-epoch diagnostics to an `IMetricsReporter` and to callbacks.

There's no batch dimension (TD-171): every sample is its own forward/backward pass, and
"batching" means gradient accumulation.

### Why it matters

It's the inner loop of `incremental_trainer`, which is where every chatbot model's weights come
from. What it reports (loss, validation loss, best epoch) drives early stopping, best-model
selection, the metrics dashboard and MNS progress records, so its numbers need to be right as
well as its gradients.

---

## 2. Where it's used

| Caller | How |
|---|---|
| `src/IncrementalTrainer.cpp` (≈ lines 997 and 1064, in the pass functions; `run_training()` ≈ 642) | Per pass: computes `tokenized_cache_key`, constructs a **fresh** `ChatbotTrainer(config.base_config)`, `set_model(std::move(model))`, `add_training_pair()` for every pair (no validation pairs, so `train()` splits), wires the metrics reporter, abort flag and callbacks, calls `train(n)`, then `release_model()` |
| `tests/chatbottrainer_test.cpp` | Unit tests for helpers, config, cache, callbacks |

`IncrementalTrainer` owns model construction itself (its own `build_model()`), so
`ChatbotTrainer::initialize_model()` (and its `validate_and_correct_config()`) only runs when a
trainer is used standalone without `set_model()`.

---

## 3. Types

| Type | Purpose |
|---|---|
| `ConversationPair{input, response, SampleMeta meta}` | Raw pair; `meta` (task type, language, quality, token count…) comes from JSONL |
| `TokenizedPair{input_tokens, target_tokens, input_text, target_text}` | Pre-tokenized pair; texts kept for logging and abnormal-sample reports |
| `LRSchedule` | `CONSTANT`, `LINEAR_WARMUP`, `COSINE_DECAY`, `WARMUP_COSINE` (default), `STEP_DECAY`, `EXPONENTIAL_DECAY` |
| `LogLevel` | `SILENT`, `NORMAL`, `VERBOSE` (default), `DEBUG` |
| `EpochCallback(epoch, total, loss, val_loss, lr)` | Fired after each epoch (TD-009) |
| `SampleCallback(sample, total, running_loss, step_loss, grad_norm, lr)` | Fired after each optimizer step (and on NaN or error) |
| `BestModelCallback(epoch, val_loss)` | Fired when a new best is saved to disk |

### `TrainingConfig` (selected fields; defaults in parentheses)

| Group | Fields |
|---|---|
| Architecture | `d_model` (512), `num_heads` (8), `d_ff` (2048), `num_encoder_layers`/`num_decoder_layers` (6), `max_seq_length` (512) |
| Training | `num_epochs` (10), `learning_rate` (1e-3), `batch_size` (1; logged only), `gradient_accumulation_steps` (1), `validation_split` (10, i.e. 1/10) |
| LR schedule | `lr_schedule`, `warmup_steps` (0 = 10% of total), `min_learning_rate` (1e-6), `lr_decay_factor` (0.1), `lr_decay_steps` (0 = one epoch) |
| Optimizer | `optimizer_type` (AdamW), `adam_beta1/2` (0.9/0.999), `weight_decay` (0.01), `gradient_clip_norm` (1.0) |
| Adaptive clipping (TD-017) | `adaptive_gradient_clip` (off), `gradient_clip_min/max` (0.1/5.0), `gradient_clip_ema_decay` (0.05), `gradient_clip_headroom` (2.0), `gradient_clip_warmup_steps` (100), `gradient_clip_spike_k` (5.0) |
| Early stopping | `enable_early_stopping` (off), `patience` (5), `min_delta` (1e-4), `restore_best_weights` (on) |
| Logging | `log_every` (10), `log_level`, `verbose` (deprecated, unused) |
| BLEU/ROUGE (TD-016/023) | `enable_generation_quality_metrics` (off), `generation_quality_sample_size` (10), `generation_quality_max_tokens` (50), `generation_quality_async_threshold` (50) |
| Quality backfill | `enable_loss_quality_backfill`, `enable_generation_quality_backfill` (both off), `generation_backfill_max_tokens` |
| Outliers (TD-021) | `loss_outlier_z_threshold` (3.0), `grad_norm_outlier_threshold` (10.0), `max_abnormal_samples` (1000) |
| Tokenizer and cache | `tokenizer_mode`, `cache_tokenized_data` (off), `tokenized_cache_dir`, `tokenized_cache_key` |

---

## 4. Setup API

| Method | Behaviour |
|---|---|
| `ChatbotTrainer(cfg)` | Stores config; best validation loss starts at `FLT_MAX`. Copying and moving are deleted |
| `load_tokenizer(vocab_path)` / `build_vocabulary(texts, size, save_path)` | Create a tokenizer in `config.tokenizer_mode`; build also saves it (TD-221 silent-failure caveat applies) |
| `set_tokenizer(unique_ptr)` / `release_tokenizer()` | Ownership transfer |
| `set_model(unique_ptr)` | Takes the model and, if none exists yet, **creates a fresh `Optimizer`** with config settings and registers parameters |
| `release_model()` | Joins the async scoring thread first (TD-023), then hands the model back |
| `load_conversation_data(path)` | Detects format from the first non-blank line: **JSONL** (`{…}` lines via `parse_jsonl_sample()`, keeps `SampleMeta`) or **legacy** `INPUT:`/`RESPONSE:` blocks separated by blank lines. Appends to `training_data`; true if any pairs loaded |
| `add_training_pair(in, resp)` / `add_validation_pair(in, resp)` | Append a pair **without metadata** |
| `set_epoch_callback` / `set_sample_callback` / `set_best_model_callback` / `set_metrics_reporter` / `set_abort_flag` | Hooks; the abort flag is checked only at optimizer-step boundaries |
| `save_to(path)` | `model->save_model(path)` |

---

## 5. `train(num_epochs)`: the pass orchestrator

```text
config.num_epochs = n; was_aborted_ = false
if no model: initialize_model()        (validate config → build model → give it the tokenizer → optimizer → GPU init)
if no validation pairs and validation_split > 0: split_data()   (random 1/split)
preprocess_data()                      (§6)
if no training pairs left: return false   (TD-412: an empty pass fails instead of "training" on nothing)
total_training_steps = n × ceil(train_size / accumulation_steps)
if the run's warmup isn't fixed yet: run_warmup_steps_ = warmup_steps, or 10% of total_training_steps (TD-269)
for each epoch:
    train_epoch(e)                     (§8)
    if aborted: break
    if validation data:
        validate()                     (§9)
        reporter->end_epoch(...)       ← only here (see §12)
        early-stop check
    metrics_tracker_.record_epoch(...) (TD-169)
    epoch_callback_(...)
if early stopping + restore_best_weights and a best was saved: load_model(best_model_path)
if generation backfill: join scoring thread, backfill_generation_quality()
return true   (any exception: logged, return false)
```

`IncrementalTrainer` restores the previous pass's training state with `load_training_state()` before
`train()` and saves it with each checkpoint (TD-269, §7 and the addendum).


---

## 6. Data preparation

### `split_data()`

Shuffles indices with a fresh `std::random_device` seed, moves `size / validation_split` pairs to
`validation_data`, and keeps the rest. It's skipped when `validation_split ≤ 0` or too little data.
**The split is different on every pass**, so validation loss from one incremental pass isn't
measured on the same samples as the next.

### `preprocess_data()`

Uses the model's tokenizer (ownership moved there in `initialize_model()`/by the caller).

1. **Cache hit** (`cache_tokenized_data` and a non-empty key): loads `<dir>/<key>.cache`, restores
   each pair's `meta.token_count` and the shuffle index, and returns.
2. Otherwise tokenizes training and validation pairs **in parallel** (`#pragma omp parallel for`,
   relying on `BPETokenizer`'s concurrent-read safety; see [BPETokenizer.md](BPETokenizer.md) §10):
   - **input:** keep the **tail** at about `5 × max_seq_length` characters
     (`truncate_text_tail()`, UTF-8 safe), encode without specials, keep the last `max_seq_length`
     tokens (`truncate_tokens_tail()`). The tail is kept because pairs are documents split at a
     midpoint, so the end of the input is what the target continues from;
   - **response:** keep the **head** at about `5 × max_seq_length` characters, encode **with**
     `<bos>`/`<eos>`, drop the leading `<bos>` (TD-361, `strip_leading_bos()`; the decoder input
     supplies it), keep the **first** `max_seq_length` tokens. Responses longer than that used to
     lose their `<eos>`, so the model never saw those targets end. Since TD-275's fix (October 17, 2026),
     `truncate_target_keep_eos()` makes the last kept token `<eos>`: the first
     `max_seq_length − 1` content tokens + `<eos>`, as standard seq2seq tokenizers truncate;
   - **empty input or response** (TD-188) and **invalid UTF-8** are counted and logged as
     "Skipped", and their slot is **left as an empty `TokenizedPair`**.
3. Saves the cache if enabled. Since TD-267's fix this comes before compaction, so the cache's counts still match the raw data. Then
   `drop_skipped_pairs()` removes the skipped pairs from `training_data`/`validation_data` and the tokenized vectors together, keeping indices
   aligned. A cache hit runs the same compaction. Finally it builds `training_indices`.

> **Fixed (TD-267, October 17, 2026): "skipped" pairs used to stay as empty entries.** Nothing
> filtered them out, so each one reached `model->forward()` in training. It threw ("Matrix index out
> of bounds"), and the catch block discarded the window's accumulated gradients. In validation each
> one scored loss 0 but still counted in the average, biasing validation loss low: one empty pair
> added to two good ones dropped it from about 3.4 to 2.5.
>
> They are now dropped after tokenizing (step 3), so `get_tokenized_*_size()` counts only real
> pairs. `ChatbotTrainerCacheTest.DropSkippedPairsKeepsRawAndTokenizedAligned` and
> `SkippedAndFailingPairsDontBiasValidationLoss` cover it, and both fail on the old code.

## 7. Model, optimizer and LR schedule

- **`validate_and_correct_config()`** (standalone path only): rounds `d_model` up to a multiple of
  `num_heads`, resets `d_ff` to `4 × d_model` if the ratio is outside [2, 8], resets
  `min_learning_rate` to 1% of the base rate if it's ≥ the base rate, and warns on unusual
  values. Silently changing the architecture would make a checkpoint mismatch any
  MNS-registered architecture, but `IncrementalTrainer` doesn't use this path.
- **`initialize_model()`**: the above, `build_model()`, hands the tokenizer to the model, creates
  and configures the `Optimizer`, registers parameters, and calls `gpu_init_training()` on GPU
  builds.
- **`calculate_learning_rate(step)`**, with `step` = the run-level `global_step`, and warmup =
  `warmup_steps`, else the run's fixed `run_warmup_steps_` (set on the run's first `train()` to 10% of
  that call's steps, then restored with the training state, TD-269). Every denominator is guarded and
  cosine progress is clamped to [0, 1], so a pass shorter than its warmup, or with fewer steps than
  epochs, no longer yields NaN or divides by zero:

  | Schedule | LR at step *s* |
  |---|---|
  | `CONSTANT` | base |
  | `LINEAR_WARMUP` | base·*s*/warmup, then base |
  | `COSINE_DECAY` | min + (base − min)·½(1 + cos(π·*s*/total)) |
  | `WARMUP_COSINE` | linear warmup, then cosine over the remaining steps |
  | `STEP_DECAY` | base·factor^⌊*s*/decay_steps⌋ (decay_steps defaults to one epoch) |
  | `EXPONENTIAL_DECAY` | base·factor^(*s*/decay_steps) |

  `update_learning_rate()` applies it to both the optimizer and the model at each step boundary.

> **Fixed (TD-269, October 19, 2026; owner decisions): optimizer state and the LR schedule used to
> restart every incremental pass.** `IncrementalTrainer` builds a new `ChatbotTrainer` per pass, and
> `set_model()` creates a new `Optimizer`. So AdamW's moments started from zero, `global_step`
> restarted at 0, and the old `WARMUP_COSINE` default re-warmed and decayed within every pass.
>
> Now:
> - `IncrementalTrainer` uses `LINEAR_WARMUP` (warm up once per run, then hold the base LR).
> - Each checkpoint carries `<checkpoint>.optimizer`: the moments, step counts, `global_step`, warmup
>   length and adaptive-clip state.
> - The next pass of the same MNS run restores it, and a new run starts fresh.
>
> `TrainingStateContinuesIntoTheNextPass` checks that pass 2's step count continues from pass 1's.

---

## 8. `train_epoch(epoch)`

Per epoch: notify `start_epoch`, shuffle `training_indices`, reset accumulators, and install the
TD-013 diagnostic hooks if a reporter is set:
- activation saturation (fraction of post-GELU values with |x| < 0.01) and attention entropy;
- CPU builds use per-layer `Matrix` hooks, GPU builds use pre-reduced GPU stats hooks;
- the hooks are cleared at epoch end.

Per sample (in shuffled order):

1. **Abort check**, only when `accumulation_step == 0`.
2. At a window start: update the LR, `zero_grad()` (plus GPU zero), and reset the padding-window
   counters.
3. **Forward, loss, backward:**
   - CPU: `forward`, `compute_loss_for_training`, `compute_loss_gradient_for_training`, scale by
     `1/accumulation_steps`, `backward_pass`.
   - GPU: `gpu_forward` (fused loss), `gpu_backward(scale)`. Until TD-390 a `gpu_synchronize()` followed, to
     drain SYCL deferred frees (the budget-exhaustion spiral). The SYCL memory pool has no deferred frees, so the
     sync, and its copies in the catch blocks, validation and the generation-quality loops, are gone.
   - Optional `exp(-loss)` quality backfill.
4. When the window is full **or it's the last sample**, the `apply_update()` lambda runs:
   - padding efficiency: actual tokens ÷ (max input + max target) × window size;
   - GPU: `gpu_download_grads()` first, so clipping sees real gradients;
   - since TD-272's fix, a window holding fewer than `gradient_accumulation_steps` samples is
     rescaled (`Optimizer::scale_gradients()`) to the mean over its samples, and its step loss is
     divided by that count;
   - gradient norm, and **NaN/Inf guard**: log, fire the callback, zero, skip the update;
   - TD-013 Welford statistics (gradient-norm variance, loss z-score), weight-update ratio every
     10 steps, outlier flagging up to `max_abnormal_samples`, running advanced metrics;
   - **clipping**: fixed `gradient_clip_norm`, or **adaptive** (TD-017). Adaptive clips at the
     ceiling during warmup, then at `clamp(EMA × headroom, min, max)`, keeping spikes
     (`> k × EMA`) out of the EMA;
   - `optimizer->step()` (GPU: `gpu_sync_weights()`), `total_loss += step loss`, `global_step++`,
     periodic log line (TD-062 counting fix), sample callback, `update_sample_metrics`.
5. **Exception:** log, GPU sync, callback. Since TD-272's fix:
   - a sample that failed **before** its backward pass leaves the window's gradients untouched and
     the window continues (`accumulation_step` counts completed samples only);
   - one that failed **during** backward drops the window, since its gradients may be half-written;
   - if the epoch's last sample fails, the pending window still steps;
   - samples that contributed nothing are counted and logged at epoch end.

At epoch end: average loss and gradient norm over updates, perplexity = exp(loss), append to the
history vectors, and report advanced metrics, saturation, entropy, per-layer gradient norms,
padding efficiency and adaptive-clip averages.

> **Fixed (TD-270, October 19, 2026): adaptive clipping used to reset every epoch.** Its EMA and
> warmup counter were local to `train_epoch()`, so each epoch restarted 100 warmup steps at the 5.0
> ceiling. With ≤ 100 optimizer steps per epoch, adaptive clipping never activated. They are now
> members (`agc_ema_`, `agc_step_count_`), initialized once and saved with the training state, so they
> carry across epochs and passes. The per-epoch spike count and clip averages stay per epoch; they're
> reporting only.

> **An error or NaN mid-window dropped the whole window** (fixed October 17, 2026, TD-272). Both
> paths reset `accumulation_step` without stepping, so gradients already accumulated from earlier
> samples in the window were zeroed at the next window start, and those samples contributed
> nothing.
>
> Now only a failure during backward, or a non-finite gradient norm (which can't be traced to one
> sample), drops the window, and the dropped samples are counted. If the epoch's last sample failed,
> the window's earlier samples were lost too; that window now steps.

> **The last, partial window was under-weighted** (fixed October 17, 2026, TD-272). Gradients were
> always scaled by `1/accumulation_steps`, and the step loss divided by the same, even when the
> final window held fewer samples, so its update and reported loss were proportionally too small.
> Short windows are now rescaled to their real sample count.
>
> New tests (both fail on the old code):
> - `ShortLastWindowUsesItsRealSampleCount`;
> - `FailedSampleKeepsTheWindowsEarlierGradients`: 4 good samples plus 1 failing one at
>   accumulation 4 take exactly one 4-sample step per epoch, wherever the failing one lands.

---

## 9. `validate()` and best-model tracking

Eval mode, then per validation pair `evaluate_tokenized()` (GPU: `gpu_evaluate_tokenized()` +
sync, using the already-truncated IDs), optional loss-quality backfill, and back to training mode.
Loss = sum ÷ **number of pairs that evaluated** (TD-267), perplexity = exp(loss). Reports `update_validation_metrics`
(accuracy always −1) and runs BLEU/ROUGE (§10).

If loss < best − `min_delta`: record the best epoch, report `update_best_metrics`, and **if early
stopping and `restore_best_weights` are both on**, `save_model("best_model_temp.bin")` and fire
the best-model callback. Otherwise `epochs_without_improvement++`.

> **Two validation-loss gotchas (both fixed October 17, 2026, TD-267):** samples that throw used to contribute 0 but still be counted, and
> skipped pairs counted as 0 (§6). Either way validation loss read low.
>
> Now skipped pairs are dropped before validation, and the average divides by the samples that actually evaluated. A warning reports how many
> failed. If every sample fails, no loss is recorded, the epoch counts as no improvement, and `validate()` returns the previous validation loss
> rather than 0.

> **`best_model_temp.bin` is a fixed relative path.** It's written to the current working
> directory (as five sidecar files), never cleaned up, and shared by any trainers running in the
> same directory. Restoring at the end of `train()` loads whatever is there.

---

## 10. Generation quality

- **`compute_generation_quality_metrics()`** (only with `enable_generation_quality_metrics` and
  a reporter): generates replies for the first N validation inputs and scores BLEU-4 and
  ROUGE-1/2/L.
  - With N ≥ `generation_quality_async_threshold`, it clones the model (`clone()` round-trips
    through temp files) and scores on a background thread, on CPU. Reporting is safe because
    `MetricsPushClient` enqueues under a mutex.
  - Otherwise it scores synchronously (GPU builds use `gpu_generate_response` with syncs).
- **`backfill_generation_quality()`**: per-pair BLEU-4 into `meta.quality`, for the whole
  dataset, after training.

> **Quality backfill results are unreachable.** Both backfills write `meta.quality` into
> `ChatbotTrainer`'s private `training_data`/`validation_data`, which have no accessor.
> `IncrementalTrainer` passes pairs through `add_training_pair()` (no metadata) and never reads
> them back. Neither option has a config key, so in practice they're off. If enabled in code,
> the generation backfill would spend a full generation per sample for nothing.

---

## 11. Small helpers

`calculate_perplexity(loss)` = exp(loss). `calculate_accuracy(preds, targets)` is token match
rate, −1 on empty or mismatched input. It's tested, but **not used by training**, and the
`training_accuracies`/`validation_accuracies` vectors are never filled. `log(level, msg)` sends
everything to `Logger::info` (the colour argument is ignored, and DEBUG isn't mapped to a debug
level). Getters expose the history vectors, best epoch and loss, early-stop and abort flags,
global step, current LR, config and data sizes. `export_metrics_csv()` delegates to
`MetricsTracker` (TD-169).

---

## 12. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Skipped (empty / invalid-UTF-8) pairs stay as empty entries: training throws per pair and drops the window; validation counts them as loss 0 (§6, §9) | Validation loss biased low (drives best model and early stopping); noisy errors; lost gradients | **Fixed October 17, 2026** (was verified: 3.4 → 2.5 with one empty pair) | [TD-267](../../guides/TECHNICAL_DEBT.md#td-267-chatbottrainer-keeps-skipped-pairs-biasing-validation-loss-low) |
| Reporter `end_epoch()` is only called when validation data exists (§5) | Runs without a validation split never close epochs on the dashboard | By inspection | [TD-268](../../guides/TECHNICAL_DEBT.md#td-268-chatbottrainer-only-ends-metrics-epochs-when-there-is-validation-data) |
| Optimizer state and LR schedule restart every incremental pass (§7) | No coherent schedule across a long run; Adam moments lost | **Fixed October 19, 2026** (training state saved with checkpoints; warmup once per run) | [TD-269](../../guides/TECHNICAL_DEBT.md#td-269-optimizer-state-and-lr-schedule-restart-every-incremental-training-pass) |
| Adaptive clipping state resets every epoch; never activates with ≤ 100 steps/epoch (§8) | TD-017 often had no effect | **Fixed October 19, 2026** (members, saved with the training state) | [TD-270](../../guides/TECHNICAL_DEBT.md#td-270-adaptive-gradient-clipping-resets-every-epoch) |
| Cache hit loaded after a fresh random split misaligns per-index metadata; split differs every pass (§6) | Wrong metadata; validation not comparable across passes | By inspection (cache off by default) | [TD-271](../../guides/TECHNICAL_DEBT.md#td-271-chatbottrainers-tokenized-cache-can-pair-with-a-different-random-split) |
| Error/NaN mid-window drops prior samples' gradients; partial last window under-weighted (§8) | Silent loss of training signal | **Fixed October 17, 2026** | [TD-272](../../guides/TECHNICAL_DEBT.md#td-272-chatbottrainer-silently-drops-gradients-on-mid-window-errors-and-under-weights-the-last-window) |
| Quality backfill writes to unreachable copies; no config keys (§10) | Dead feature; wasted compute if enabled | By inspection | [TD-273](../../guides/TECHNICAL_DEBT.md#td-273-chatbottrainer-quality-backfill-writes-results-nobody-can-read) |
| `best_model_temp.bin` in the working directory, shared and never cleaned (§9) | Collisions; stray files | By inspection | [TD-274](../../guides/TECHNICAL_DEBT.md#td-274-chatbottrainer-saves-the-best-model-to-a-fixed-file-in-the-working-directory) |
| Long responses lose `<eos>` on head truncation (§6) | Model never saw long targets end | **Fixed October 17, 2026** (`truncate_target_keep_eos()`) | [TD-275](../../guides/TECHNICAL_DEBT.md#td-275-chatbottrainer-drops-eos-from-long-training-responses) |
| Hygiene: unused accuracy/`verbose`, `log()` level mapping, `end_epoch` reports last-window gradient norm rather than the epoch average, stale `chatbottrainer_test.cpp.old`, `validate_and_correct_config()` silently changes architecture (standalone path) | Minor | By inspection | [TD-276](../../guides/TECHNICAL_DEBT.md#td-276-chatbottrainer-hygiene-dead-fields-logging-stale-test-file) |
| Whole class capped below `stable` | Already TD-039 | — | TD-039 |

## Addendum (October 19, 2026): training state across passes (TD-269, TD-270)

- **`save_training_state(path)`/`load_training_state(path)`** persist the optimizer's moments and step counts (`Optimizer::save_state()`), the
  run-level `global_step`, the run's warmup length and the adaptive-clip EMA and counter, tagged with the run id set by `set_run_id()`.
  - Loading refuses state from another run, and state whose shapes don't match the model.
  - `save_to()` also writes `<path>.optimizer`, so best snapshots and auto-saves carry the state.
- **`calculate_learning_rate()`** uses `run_warmup_steps_` (fixed on the first `train()` call, then restored) instead of re-deriving 10% of each
  call's steps. `IncrementalTrainer` now uses `LINEAR_WARMUP`, the owner decision: warm up once per run, then hold the base LR.
- **The adaptive-clipping EMA and warmup counter** are members (`agc_ema_`, `agc_step_count_`), so they no longer reset every epoch (TD-270).
- **`calculate_learning_rate()`** guards every denominator and clamps the cosine progress to [0, 1]. `WARMUP_COSINE` used to return NaN when the
  warmup equalled the step count, and `STEP_DECAY`/`EXPONENTIAL_DECAY` divided by zero when a pass had fewer steps than epochs.
- **Atomic save, rechecked October 19, 2026:** `save_training_state()` writes `<path>.tmp` and moves it into place with
  `std::filesystem::rename()`, which replaces the old file on every platform. It used `std::rename()`, which fails on Windows when the
  target exists.

## Addendum (October 19, 2026): GPU training steps on the device (TD-468)

In GPU builds, `train()` calls `bind_gpu_optimizer()`. That uploads the host weights (`gpu_init_training()`) and binds a `GPUOptimizer` to the layers' device mirrors. `train_epoch()` binds its own when called directly.

`apply_update()` then works like this:
- It no longer downloads gradients or uploads weights.
- The gradient norm comes from one float per parameter group.
- The TD-272 window rescale and the clip factor are passed to `GPUOptimizer::step()` as one gradient scale.
- The weight-update ratio and the epoch-end per-layer gradient norms come from device reductions. The per-layer norms now include each block's LayerNorms (TD-356).
- `end_epoch()` reports the last step's norm (`last_grad_norm_`) instead of recomputing it from the host gradients.

While training, the device weights are the live copy. `sync_host_from_device()` copies weights and moments back only:
- before `save_to()`, the best-model save, `save_training_state()` and the quality-metrics `clone()`;
- at the end of `train()`, including after a failure, best effort.

Verified in a SYCL build on the oneAPI CPU device (`ADAI_SYCL_ALLOW_CPU_DEVICE=1`): all 83 tests here pass with every training test on the device path. See [GPUOptimizer.md](GPUOptimizer.md).

---

## Addendum (October 21, 2026): no hippocampal updates while training (TD-343)

`train()` turns off the model's hippocampal updates (`set_hippocampal_updates_enabled(false)`) for its whole run and restores the
previous setting on exit, so its quality and backfill generations neither commit coverage steps nor write episodic memories.

## Addendum (November 4, 2026): every epoch is ended on the reporter (TD-268)

`train()` calls `metrics_reporter_->end_epoch()` after every completed epoch, not only inside the validation branch. Without
validation data (`validation_split = 0`, or a pass with fewer pairs than the split ratio) it sends validation loss 0, which
`IMetricsReporter::end_epoch()` documents as "not computed" and `metrics_api_server` already treats that way. The gradient norm
sent is the epoch's average (`gradient_norms.back()`), not the last window's, and the epoch's wall-clock seconds are sent too
(they were always 0). Early stopping still applies only with validation data, after the epoch has been ended. An aborted epoch
(`was_aborted_`) is still not ended. The `last_grad_norm_` member, used only by the old call, is removed.

Checked with a real `incremental_trainer` pass (6 pairs, so no validation set) pushing to a real `metrics_api_server`: before,
3 epochs trained and the session summary showed `total_epochs_completed: 0` and empty `epoch_losses`; after, 3 epochs with the
trainer's logged losses, validation losses 0, the epoch-average gradient norm and real durations.

## Addendum (November 4, 2026): a deterministic validation split, checked against the cache (TD-271)

`split_data()` no longer shuffles with `std::random_device`. Each pair gets a rank from its own input and response text
(FNV-1a 64), and the validation set is the `size / validation_split` pairs with the lowest ranks (ties by load order); both
sets keep the load order. The same data therefore splits the same way on every pass and whatever order its files were
loaded in, so validation losses on the same data are measured on the same pairs. Different data (a later pass with other
files) still has its own validation set; see TD-405 for what that means for checkpoint selection.

`load_tokenized_cache()` now also checks that every cached pair's stored input and response texts are the raw pair at the
same index (a skipped pair's empty placeholder matches anything) and treats a mismatch as a miss, logging that the cache holds
a different split. `kPreprocessFormatVersion` is 4, so caches written with the old random splits aren't reused, and
`IncrementalTrainer`'s cache key includes `validation_split`.

Checked with real `incremental_trainer` passes (20 pairs, `CACHE_TOKENIZED_DATA=true`): two passes with the cache cleared
between them wrote different cache files with the old binary (different validation pairs) and byte-identical ones with the
new; a third pass hit the cache with both, which for the old binary meant tokenized pairs from another split.

## Addendum (November 4, 2026): quality backfills reported to the metrics server (TD-273)

Owner decision: wire up both backfills and report them to `metrics_api_server`, the loss score on by default and the
generation score opt-in.

- `TrainingConfig::enable_loss_quality_backfill` now defaults to `true` (`LOSS_QUALITY_BACKFILL`); `enable_generation_quality_backfill`
  stays `false` (`GENERATION_QUALITY_BACKFILL`); `quality_report_lowest_samples` (5) sets how many lowest-scoring pairs a report carries.
  `IncrementalTrainer::make_incremental_config()` maps both keys.
- `train_epoch()` and `validate()` reset `meta.quality` to -1 before scoring, so a report counts only that epoch's scores (a pair
  that fails reports none).
- After each completed epoch, `train()` calls `report_sample_quality(epoch, "loss")`; after the generation backfill, once with
  `"generation"`. Each builds one `SampleQualityReport` per split with scored pairs through the public static
  `summarize_sample_quality()` (count, mean, min, p10/median/p90 by linear interpolation, max, the lowest pairs, lowest first, ties in
  data order; negative or non-finite scores left out) and sends it with `IMetricsReporter::report_sample_quality()`.
- `add_training_pair(ConversationPair)`/`add_validation_pair(ConversationPair)` keep a pair's dataset metadata; `IncrementalTrainer`
  now passes its loaded pairs that way. `get_training_pairs()`/`get_validation_pairs()` expose the pairs with their scores.

## Addendum (November 4, 2026): the best-weights copy has its own directory (TD-274)

The note above about `best_model_temp.bin` no longer applies. When `validate()` saves a new best for `restore_best_weights`, it
uses a directory of this trainer's own, `<best_model_dir>/best_model_<16 random hex digits>/best.bin` (`adai::secure_random_hex()`),
created on the first save of a `train()` call. `TrainingConfig::best_model_dir` defaults to empty, meaning the system temp
directory; `IncrementalTrainer::make_incremental_config()` sets it to the session directory, since the copy is as large as a
checkpoint. `remove_best_model_copy()` deletes the whole directory, every sidecar included: at the start of `train()`, after the
restore, on `train()`'s failure path and in the destructor. Two trainers never share a copy, a later `train()` never restores an
earlier call's best, and nothing is written to the working directory any more. A process killed mid-pass still leaves its
directory behind in the session directory; nothing sweeps those.

Checked with real `incremental_trainer` passes (20 pairs, 2 epochs, early stopping on) started from an empty working directory:
the old binary left the 16 `best_model_temp.bin*` files there; the new one left nothing there or in the session directory. Both
restored the best epoch.

## Addendum (November 4, 2026): validation token accuracy and other cleanups (TD-276)

- **Validation accuracy is computed.** `validate()` passes a `TokenAccuracy` to `evaluate_tokenized()`/`gpu_evaluate_tokenized()`
  and sums each sample's `correct`/`total`; the share is pushed to `validation_accuracies` (`get_validation_accuracies()`) and sent
  with `update_validation_metrics()`, where it used to be -1 always. It is -1 only when no target position could be counted, and it
  is logged with the validation loss. Training accuracy isn't computed: `training_accuracies` and `calculate_accuracy()` (used only
  by tests) are removed.
- **`log()`** maps its level to a `Logger` level (SILENT, used for errors, to error; DEBUG to debug; NORMAL/VERBOSE to info) instead
  of sending every message at info. Its unused colour argument and the local `COLOR_*` macros are gone.
- **`TrainingConfig::verbose`**, deprecated and unread, is removed.
- **`validate_and_correct_config()`** never changes the architecture: `d_model` that isn't a positive multiple of `num_heads` throws
  `std::invalid_argument` (it used to be rounded up), and an unusual `d_ff`/`d_model` ratio only warns (it used to be reset to 4x).
  Both silent changes would break compatibility with a checkpoint or MNS registration. `min_learning_rate` is still corrected.
- `tests/chatbottrainer_test.cpp.old` (an unbuilt January 2026 leftover) is deleted.

## Addendum (November 10, 2026): a lost GPU stops the pass (TD-566)

After a training or validation sample fails, a GPU build checks the device with `GPUManager::device_usable()` (a small allocation,
fill and copy back). If the device no longer runs work, as after a driver engine reset (`UR_RESULT_ERROR_DEVICE_LOST`), the pass
stops with `GPUDeviceLostError`, and `train()` returns false without copying the device weights back; a sample that fails on a
working device is still skipped as before (TD-267, TD-272). An epoch in which no training sample produced an update throws
`EpochTrainedNothingError` instead of recording loss 0 and perplexity 1, unless the epoch was aborted. The check is the
`device_usable_` member, empty in a CPU build and replaced by tests.
