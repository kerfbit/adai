# `ChatbotTrainer` — Source File Reference

- **Files:** [`src/ChatbotTrainer.hpp`](../../../../src/ChatbotTrainer.hpp), [`src/ChatbotTrainer.cpp`](../../../../src/ChatbotTrainer.cpp)
- **Built into:** compiled directly into `incremental_trainer`, `training_metrics_example` and `chatbottrainerTests` (listed in each target's sources); it isn't part of a static library
- **Status tag:** `@adai-status: beta` (capped by TD-039, "large, actively evolving core trainer"), `@adai-version: 0.10.0` (hpp) / `0.10.1` (cpp), `@adai-reviewed: 2026-09-13` / `2026-09-17`
- **Tests:** [`tests/chatbottrainer_test.cpp`](../../../../tests/chatbottrainer_test.cpp) → `chatbottrainerTests` (72 tests; `friend class ChatbotTrainerCacheTest` for the tokenized cache); test plan in [chatbot-trainer-tests.md](../../testing/chatbot-trainer-tests.md). `tests/chatbottrainer_test.cpp.old` is an unbuilt January-2026 leftover
- **Related guide:** [enhanced-training-pipeline.md](../../guides/training/enhanced-training-pipeline.md) (multi-component; partly stale per its own banner)
- **Last traced against the code:** 2026-10-09

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
total_training_steps = n × ceil(train_size / accumulation_steps)
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
     `<bos>`/`<eos>`, keep the **first** `max_seq_length` tokens. Responses longer than that lose
     their `<eos>`, so the model never sees those targets end;
   - **empty input or response** (TD-188) and **invalid UTF-8** are counted and logged as
     "Skipped", and their slot is **left as an empty `TokenizedPair`**.
3. Builds `training_indices`, then saves the cache if enabled.

> **"Skipped" pairs are not removed (verified).** The comment says they're "filtered out during
> training", but nothing filters them: `tokenized_training_data` keeps an empty entry per skipped
> pair (`get_tokenized_training_size()` still counts it).
> - **Training:** each one reaches `model->forward()`, which throws ("Matrix index out of
>   bounds" in the test run). The catch block logs an error and resets accumulation, which also
>   **discards the gradients already accumulated** in that window.
> - **Validation:** each one evaluates to loss 0 with no error but still counts in the average, so
>   **validation loss is biased low** by the share of skipped pairs. In the test run, adding one
>   empty pair to two good ones dropped validation loss from about 3.4 to 2.5. Validation loss
>   drives best-model selection, early stopping and the dashboard.

### Tokenized-data cache format

`'TKC1'` magic, `uint64` train and validation counts, then per pair: `u32` length + `int32` input
IDs, the same for target IDs, then `u32` length + bytes for input text and for target text.
Staleness is the caller's job: `IncrementalTrainer`'s key covers file and vocab checksums, tokenizer
mode and `max_seq_length`. Loading also rejects count mismatches and truncated files. Saving is
best-effort, logging rather than failing.

> **Cache hits can pair data from different splits.** `train()` runs `split_data()` (a new random
> split) **before** `preprocess_data()` loads the cache, which holds the tokenized split from the
> run that wrote it. Counts match, so the load succeeds, but `tokenized_training_data[i]` is no
> longer `training_data[i]`. Per-index metadata writes (`meta.token_count`, quality backfill) then
> land on the wrong pairs. Training itself still uses a valid, self-consistent split (the cached
> one). This only applies when `cache_tokenized_data` is on (default off).

---

## 7. Model, optimizer and LR schedule

- **`validate_and_correct_config()`** (standalone path only): rounds `d_model` up to a multiple of
  `num_heads`, resets `d_ff` to `4 × d_model` if the ratio is outside [2, 8], resets
  `min_learning_rate` to 1% of the base rate if it's ≥ the base rate, and warns on unusual
  values. Silently changing the architecture would make a checkpoint mismatch any
  MNS-registered architecture, but `IncrementalTrainer` doesn't use this path.
- **`initialize_model()`**: the above, `build_model()`, hands the tokenizer to the model, creates
  and configures the `Optimizer`, registers parameters, and calls `gpu_init_training()` on GPU
  builds.
- **`calculate_learning_rate(step)`**, with warmup auto-set to 10% of `total_training_steps`:

  | Schedule | LR at step *s* |
  |---|---|
  | `CONSTANT` | base |
  | `LINEAR_WARMUP` | base·*s*/warmup, then base |
  | `COSINE_DECAY` | min + (base − min)·½(1 + cos(π·*s*/total)) |
  | `WARMUP_COSINE` | linear warmup, then cosine over the remaining steps |
  | `STEP_DECAY` | base·factor^⌊*s*/decay_steps⌋ (decay_steps defaults to one epoch) |
  | `EXPONENTIAL_DECAY` | base·factor^(*s*/decay_steps) |

  `update_learning_rate()` applies it to both the optimizer and the model at each step boundary.

> **Optimizer state and LR schedule restart every incremental pass.** `IncrementalTrainer` builds
> a new `ChatbotTrainer` per pass, and `set_model()` creates a **new** `Optimizer`. AdamW's moment
> estimates start from zero again (with bias correction restarting), and `global_step` restarts at
> 0, so the default `WARMUP_COSINE` re-warms from LR 0 and decays to the minimum **within every
> pass**. Long training made of many short passes therefore never runs one coherent schedule. The
> optimizer state isn't saved in checkpoints either.

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
   - GPU: `gpu_forward` (fused loss), `gpu_backward(scale)`, `gpu_synchronize()`. The sync drains
     SYCL deferred frees, the fix for the budget-exhaustion spiral.
   - Optional `exp(-loss)` quality backfill.
4. When the window is full **or it's the last sample**:
   - padding efficiency: actual tokens ÷ (max input + max target) × window size;
   - GPU: `gpu_download_grads()` first, so clipping sees real gradients;
   - gradient norm, and **NaN/Inf guard**: log, fire the callback, zero, skip the update;
   - TD-013 Welford statistics (gradient-norm variance, loss z-score), weight-update ratio every
     10 steps, outlier flagging up to `max_abnormal_samples`, running advanced metrics;
   - **clipping**: fixed `gradient_clip_norm`, or **adaptive** (TD-017). Adaptive clips at the
     ceiling during warmup, then at `clamp(EMA × headroom, min, max)`, keeping spikes
     (`> k × EMA`) out of the EMA;
   - `optimizer->step()` (GPU: `gpu_sync_weights()`), `total_loss += step loss`, `global_step++`,
     periodic log line (TD-062 counting fix), sample callback, `update_sample_metrics`.
5. **Exception:** log, GPU sync, callback, reset accumulation.

At epoch end: average loss and gradient norm over updates, perplexity = exp(loss), append to the
history vectors, and report advanced metrics, saturation, entropy, per-layer gradient norms,
padding efficiency and adaptive-clip averages.

> **Adaptive clipping resets every epoch.** Its EMA, warmup counter and spike count are local to
> `train_epoch()`. Each epoch restarts with 100 warmup steps clipped at the 5.0 ceiling, and with
> ≤ 100 optimizer steps per epoch (common with small passes or large accumulation) **adaptive
> clipping never activates at all**.

> **An error or NaN mid-window drops the whole window.** Both paths reset `accumulation_step`
> without stepping, so gradients already accumulated from earlier samples in the window are
> zeroed at the next window start: those samples contribute nothing.

> **The last, partial window is under-weighted.** Gradients are always scaled by
> `1/accumulation_steps`, and the step loss is divided by the same, even when the final window
> holds fewer samples. Its update and reported loss are proportionally too small.

---

## 9. `validate()` and best-model tracking

Eval mode, then per validation pair `evaluate_tokenized()` (GPU: `gpu_evaluate_tokenized()` +
sync, using the already-truncated IDs), optional loss-quality backfill, and back to training mode.
Loss = sum ÷ **number of pairs**, perplexity = exp(loss). Reports `update_validation_metrics`
(accuracy always −1) and runs BLEU/ROUGE (§10).

If loss < best − `min_delta`: record the best epoch, report `update_best_metrics`, and **if early
stopping and `restore_best_weights` are both on**, `save_model("best_model_temp.bin")` and fire
the best-model callback. Otherwise `epochs_without_improvement++`.

> **Two validation-loss gotchas:** samples that throw contribute 0 but are still counted, and
> skipped pairs count as 0 (§6). Either way validation loss reads low.

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
| Skipped (empty / invalid-UTF-8) pairs stay as empty entries: training throws per pair and drops the window; validation counts them as loss 0 (§6, §9) | Validation loss biased low (drives best model and early stopping); noisy errors; lost gradients | **Verified** (3.4 → 2.5 with one empty pair; "Matrix index out of bounds") | [TD-267](../../guides/TECHNICAL_DEBT.md#td-267-chatbottrainer-keeps-skipped-pairs-biasing-validation-loss-low) |
| Reporter `end_epoch()` is only called when validation data exists (§5) | Runs without a validation split never close epochs on the dashboard | By inspection | [TD-268](../../guides/TECHNICAL_DEBT.md#td-268-chatbottrainer-only-ends-metrics-epochs-when-there-is-validation-data) |
| Optimizer state and LR schedule restart every incremental pass (§7) | No coherent schedule across a long run; Adam moments lost | By inspection (`IncrementalTrainer` + `set_model()`) | [TD-269](../../guides/TECHNICAL_DEBT.md#td-269-optimizer-state-and-lr-schedule-restart-every-incremental-training-pass) |
| Adaptive clipping state resets every epoch; never activates with ≤ 100 steps/epoch (§8) | TD-017 often has no effect | By inspection | [TD-270](../../guides/TECHNICAL_DEBT.md#td-270-adaptive-gradient-clipping-resets-every-epoch) |
| Cache hit loaded after a fresh random split misaligns per-index metadata; split differs every pass (§6) | Wrong metadata; validation not comparable across passes | By inspection (cache off by default) | [TD-271](../../guides/TECHNICAL_DEBT.md#td-271-chatbottrainers-tokenized-cache-can-pair-with-a-different-random-split) |
| Error/NaN mid-window drops prior samples' gradients; partial last window under-weighted (§8) | Silent loss of training signal | By inspection | [TD-272](../../guides/TECHNICAL_DEBT.md#td-272-chatbottrainer-silently-drops-gradients-on-mid-window-errors-and-under-weights-the-last-window) |
| Quality backfill writes to unreachable copies; no config keys (§10) | Dead feature; wasted compute if enabled | By inspection | [TD-273](../../guides/TECHNICAL_DEBT.md#td-273-chatbottrainer-quality-backfill-writes-results-nobody-can-read) |
| `best_model_temp.bin` in the working directory, shared and never cleaned (§9) | Collisions; stray files | By inspection | [TD-274](../../guides/TECHNICAL_DEBT.md#td-274-chatbottrainer-saves-the-best-model-to-a-fixed-file-in-the-working-directory) |
| Long responses lose `<eos>` on head truncation (§6) | Model never sees long targets end | By inspection | [TD-275](../../guides/TECHNICAL_DEBT.md#td-275-chatbottrainer-drops-eos-from-long-training-responses) |
| Hygiene: unused accuracy/`verbose`, `log()` level mapping, `end_epoch` reports last-window gradient norm rather than the epoch average, stale `chatbottrainer_test.cpp.old`, `validate_and_correct_config()` silently changes architecture (standalone path) | Minor | By inspection | [TD-276](../../guides/TECHNICAL_DEBT.md#td-276-chatbottrainer-hygiene-dead-fields-logging-stale-test-file) |
| Whole class capped below `stable` | Already TD-039 | — | TD-039 |
