# `ChatbotTrainer` — Source File Reference

- **Files:** [`src/ChatbotTrainer.hpp`](../../../../src/ChatbotTrainer.hpp) (709 lines), [`src/ChatbotTrainer.cpp`](../../../../src/ChatbotTrainer.cpp) (2,627 lines)
- **Built into:** compiled directly into `incremental_trainer`, `training_metrics_example` and `chatbottrainerTests`; it isn't part of a static library
- **Status tags:** `@adai-status: beta` (capped by TD-039, "large, actively evolving core trainer"); `.hpp` `@adai-version: 0.17.0` (reviewed 2026-11-10), `.cpp` `0.23.0` (2026-11-13)
- **Tests:** [`tests/chatbottrainer_test.cpp`](../../../../tests/chatbottrainer_test.cpp) → `chatbottrainerTests` (99; `friend class ChatbotTrainerCacheTest` for the cache and internals); test plan in [chatbot-trainer-tests.md](../../testing/chatbot-trainer-tests.md)
- **Related:** [IncrementalTrainer.md](IncrementalTrainer.md) (its only production caller), [GPUOptimizer.md](GPUOptimizer.md), [PassHeartbeat.md](PassHeartbeat.md)
- **Last traced against the code:** 2026-11-13

> Code-traced reference. Behaviour marked **verified** was confirmed by a real training pass (a tiny model and the debug libraries, or a real
> `incremental_trainer` pass); the rest is from the code and its callers.

---

## 1. What this file is

`ChatbotTrainer` runs **one supervised training call** of an `EncoderDecoderModel` over (input → response) pairs:

1. holds the pairs, and splits off validation data deterministically;
2. tokenizes them in parallel, with an optional on-disk cache;
3. runs epochs of per-sample forward and backward passes with gradient accumulation, an LR schedule, fixed or adaptive clipping, dropout, and
   SGD/Adam/AdamW (on the device in GPU builds, TD-468);
4. validates (loss and token accuracy), tracks and restores the best weights, and optionally stops early;
5. streams diagnostics to an `IMetricsReporter` and to callbacks, and reports per-pair quality.

There's no batch dimension (TD-171): each sample is its own forward and backward pass, and "batching" is gradient accumulation. It is the inner
loop of `incremental_trainer`: what it reports drives early stopping, best-model selection, the dashboard and MNS progress.

---

## 2. Where it's used

`IncrementalTrainer` (each training pass) builds a **fresh** `ChatbotTrainer(config.base_config)`, `set_model()`s its model, restores the run's
training state (`load_training_state()`), adds the pass's pairs with their metadata (`add_training_pair(ConversationPair)`), wires the reporter,
abort flag and callbacks, calls `train(n)`, saves the training state with each checkpoint, and `release_model()`s. `IncrementalTrainer` builds the
model itself, so `initialize_model()`/`validate_and_correct_config()` only run when the trainer is used standalone.

---

## 3. Types and configuration

| Type | Purpose |
|---|---|
| `ConversationPair{input, response, SampleMeta meta}` | A raw pair; `meta` (task type, language, quality, token count, and the earlier turns of a multi-turn sample, TD-295) from JSONL |
| `TokenizedPair{input_tokens, target_tokens, input_text, target_text}` | A tokenized pair; the texts kept for logging, the cache check and abnormal-sample reports |
| `LRSchedule` | `CONSTANT`, `LINEAR_WARMUP`, `COSINE_DECAY`, `WARMUP_COSINE` (the struct default; `IncrementalTrainer` uses `LINEAR_WARMUP` or `CONSTANT`, TD-269/TD-410), `STEP_DECAY`, `EXPONENTIAL_DECAY` |
| `LogLevel` | `SILENT`, `NORMAL`, `VERBOSE` (default), `DEBUG`; `log()` maps them to logger levels (TD-276) |
| `EpochCallback(epoch, total, loss, val_loss, lr)`, `SampleCallback(sample, total, running_loss, step_loss, grad_norm, lr)`, `BestModelCallback(epoch, val_loss)` | Hooks |
| `GPUDeviceLostError`, `EpochTrainedNothingError` | TD-566: the device stopped running work; an epoch produced no update |

`TrainingConfig` holds the architecture; `num_epochs`, `learning_rate`, `batch_size` (logged only), `gradient_accumulation_steps`,
`validation_split` (10 = one tenth); the LR schedule (`warmup_steps`, 0 = 10% of the first call's steps; `min_learning_rate`; decay factor and
steps); the optimizer (type, betas, `weight_decay` on weight matrices only, TD-469; `gradient_clip_norm`); adaptive clipping (TD-017: off,
min/max 0.1/5.0, EMA decay, headroom, warmup steps, spike factor); `dropout` (0.1, TD-344); early stopping (`enable_early_stopping`, `patience`,
`min_delta`, `restore_best_weights`, `best_model_dir`); BLEU/ROUGE per epoch (off; sample size, max tokens, async threshold); the quality
backfills (loss on, generation off, `quality_report_lowest_samples` 5, TD-273); outlier thresholds (TD-021); the tokenizer mode and the cache
(`cache_tokenized_data`, directory, key).

---

## 4. Setup API

| Method | Behaviour |
|---|---|
| `ChatbotTrainer(cfg)` | Stores the config; copying and moving are deleted; the destructor joins the quality thread and removes the best-weights copy |
| `load_tokenizer()` / `build_vocabulary()` | A tokenizer in `config.tokenizer_mode`; building also saves it, and returns false when the save fails (TD-221) |
| `set_model()` / `release_model()` | Ownership; `set_model()` creates a fresh `Optimizer` and registers parameters when there is none; `release_model()` joins the async quality thread first |
| `load_conversation_data(path)` | JSONL or legacy `INPUT:`/`RESPONSE:`, appended to `training_data` |
| `add_training_pair()` / `add_validation_pair()` | A `ConversationPair` (keeps metadata and history) or two strings |
| `set_*_callback()`, `set_metrics_reporter()`, `set_abort_flag()` | Hooks; the abort flag is checked only at an optimizer-step boundary |
| `save_to(path)` | Copies device weights back (GPU), `model->save_model(path)`, and writes `<path>.optimizer` |
| `save_training_state()` / `load_training_state()` | The optimizer's moments and steps, `global_step`, the run's warmup length and the adaptive-clip state, tagged with `set_run_id()`'s run; a load refuses another run's state or mismatched shapes; saved through a temporary file and `std::filesystem::rename()` (TD-269) |

---

## 5. `train(num_epochs)`

```text
remove any earlier best-weights copy (TD-274); clear the abort and device-lost flags
if no model: initialize_model()
hippocampal updates off, dropout on, both restored on exit (TD-343, TD-344)
split_data() when there's no validation data and validation_split > 0          (§6)
preprocess_data()                                                              (§6)
no training pairs → false (TD-412)
total steps = epochs × ceil(pairs / accumulation); fix the run's warmup on its first call (TD-269)
GPU: bind_gpu_optimizer()  (upload the host weights, bind a GPUOptimizer to the device mirrors, TD-468);
     log the TD-574 diagnostics when any is set
for each epoch:
    train_epoch(e)                                                             (§7)
    aborted → stop
    validate() when there's validation data                                    (§8)
    reporter end_epoch(...) every completed epoch, val loss 0 without validation (TD-268)
    report_sample_quality(epoch, "loss") with the loss backfill on (TD-273)
    metrics_tracker_.record_epoch(...), epoch callback
    early stop (validation data only) → break   (after the callback, TD-571)
GPU: copy device weights and moments back (sync_host_from_device()); with diagnostics set, log how many
     kernel calls ran on the host (TD-574)
restore the best weights when early stopping and restore_best_weights are on; remove the copy
generation backfill: join the quality thread, score every pair, report "generation"
return true
exception → log; GPU: copy the device state back unless the device was lost (TD-566); remove the copy; return false
```

---

## 6. Data preparation

- **`split_data()` (TD-271):** each pair gets a rank from its input and response text (FNV-1a 64); the `size / validation_split` lowest ranks are
  validation (ties by load order), and both sets keep their load order. The same data splits the same way on every pass, whatever order its
  files were loaded in.
- **`preprocess_data()`:** uses the model's tokenizer.
  1. **Cache hit** (`cache_tokenized_data` and a key; `IncrementalTrainer`'s key covers each file's checksum and segment, the vocabulary, mode,
     `max_seq_length`, `kPreprocessFormatVersion` (4) and `validation_split`): loads `<dir>/<key>.cache`, checks every cached pair's texts
     against the raw pair at that index (a mismatch is a miss, TD-271), restores token counts and the index, and drops skipped pairs.
  2. Otherwise tokenizes in parallel (OpenMP; `BPETokenizer` encoding is `const` and thread-safe):
     - **input:** the conversation as ids with turn tokens when the vocabulary has them (`encode_conversation()` of the metadata's earlier
       turns and the message, newest kept), else the text; the tail kept (about 5×`max_seq_length` characters, then the last `max_seq_length` tokens), since
       the end of the input is what the response continues;
     - **response:** the head kept, encoded with `<bos>`/`<eos>`, the leading `<bos>` dropped (the decoder input supplies it, TD-361), cut to
       `max_seq_length` with `<eos>` kept last (TD-275);
     - an empty input or response, or invalid UTF-8, is skipped and counted;
     - each pair notes progress for `trainer_service`'s stall check (TD-569).
  3. Saves the cache when enabled (before compaction, so its counts match the raw data), then `drop_skipped_pairs()` removes the skipped pairs
     from the raw and tokenized vectors together (TD-267), and builds `training_indices`.

---

## 7. `train_epoch(epoch)`

Start the reporter's epoch, shuffle the indices (the init RNG, so `SEED` replays it, TD-368), reset the accumulators, install the TD-013
diagnostic hooks (activation saturation, attention entropy; device-reduced in GPU builds). Per sample, in shuffled order:

1. note progress (TD-569); abort check, at a window start only;
2. at a window start: update the LR, zero the gradients;
3. forward, loss, backward, scaled by `1/accumulation_steps` (GPU: `gpu_forward()` with the fused loss, `gpu_backward(scale)`); optional
   `exp(-loss)` quality score;
4. when the window is full, or at the last sample, `apply_update()`:
   - a short window is rescaled to its real sample count (TD-272);
   - the gradient norm (GPU: one float per parameter group); a non-finite norm is logged, reported, zeroed and skipped;
   - Welford statistics, the weight-update ratio, outlier flagging (`max_abnormal_samples`), running advanced metrics;
   - clipping: fixed, or adaptive (TD-017: the ceiling during warmup, then `clamp(EMA × headroom, min, max)` with spikes kept out of the EMA;
     the EMA and counter are members saved with the training state, TD-270);
   - the optimizer step (GPU: `GPUOptimizer::step()` with the rescale and clip as one gradient scale), `global_step++`, logging, the sample
     callback, `update_sample_metrics()`;
5. **a sample that throws:** in a GPU build, `device_usable()` checks the device first; an unusable one throws `GPUDeviceLostError` (TD-566).
   Otherwise a failure before backward keeps the window's gradients, one during backward drops the window, and the last window still steps
   (TD-272). On a lost device it also logs `Last GPU work: …` (TD-574, [GPUDebugSync.md](GPUDebugSync.md)).

**GPU sync points (TD-574).** `GPUManager::debug_sync_point()` is called where each phase starts queuing work: `forward` before
`gpu_forward()`, `backward` before `gpu_backward()`, `optimizer` at the start of `apply_update()`, and `validation` before each
`gpu_evaluate_tokenized()`. It records the phase, and with `ADAI_GPU_SYNC=step` or `kernel` first waits for the previous phase's work,
so a failure is attributed to it. Off by default, when it only records.

At epoch end: no update at all throws `EpochTrainedNothingError` (unless aborted, TD-566); otherwise the average loss and gradient norm over the
updates, perplexity, the history vectors, and the advanced metrics, saturation, entropy, per-layer gradient norms (LayerNorms included, TD-356),
padding efficiency and adaptive-clip averages.

---

## 8. `validate()` and the best weights

Eval mode (no dropout), then per validation pair `evaluate_tokenized()` (GPU: `gpu_evaluate_tokenized()`), counting correct target tokens
(TD-276), an optional quality score and a progress note; back to training mode. Loss = sum ÷ **pairs that evaluated** (TD-267); a failure is
checked for a lost device as in training. Validation accuracy is the share of correct tokens (−1 when none could be counted). It reports
`update_validation_metrics()` and runs the per-epoch BLEU/ROUGE (§9).

A new best (loss < best − `min_delta`) records the epoch and reports it; with early stopping and `restore_best_weights`, it saves the weights to
this trainer's own `<best_model_dir>/best_model_<16 hex digits>/best.bin` (the session directory under `IncrementalTrainer`, else the system temp
directory) and fires the best-model callback. `remove_best_model_copy()` deletes that directory after the restore, on failure and in the
destructor (TD-274).

---

## 9. Generation quality

- **Per epoch** (`enable_generation_quality_metrics`, with a reporter): BLEU-4 and ROUGE-1/2/L of replies to the first N validation inputs. From
  N ≥ `generation_quality_async_threshold`, a `clone()` of the model is scored on a background CPU thread (`ScopedCpuOnly`, TD-382), which doesn't
  note progress; otherwise synchronously.
- **Backfills** (TD-273): the loss score (`exp(-loss)`) per pair every epoch, and, opt-in, a generated reply's BLEU-4/100 per pair after
  training. `report_sample_quality(epoch, source)` sends one `SampleQualityReport` per split (count, mean, min, p10/median/p90, max, the lowest
  pairs; `summarize_sample_quality()`).

---

## 10. Problems

### Fixed: the epoch early stopping ended on skipped the epoch callback (TD-571, November 11, 2026)

`train()` checked early stopping before `metrics_tracker_.record_epoch()` and the epoch callback, so the epoch that triggered it, trained and
validated in full, never reached `IncrementalTrainer`'s session history, MNS progress or the admin API's epoch. The check now comes after
both. `TheEpochThatTriggersEarlyStoppingReachesTheEpochCallback` (patience 1, an unreachable `min_delta`): the callback sees epochs 0 and 1, and
fails with the check moved back.

---

## 11. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| The early-stopping epoch skipped `record_epoch()` and the epoch callback (§10) | — | **Fixed November 11, 2026** | [TD-571](../../guides/TECHNICAL_DEBT.md#td-571-findings-from-the-november-11-2026-trace-of-the-files-td-569-touched) |
| Skipped pairs biasing validation; `end_epoch()` only with validation; optimizer state and LR schedule per pass; adaptive clipping per epoch; split vs cache; mid-window losses; unread backfills; the best-weights file; `<eos>` lost; hygiene | — | **Fixed** October 17 – November 4, 2026 | TD-267–TD-276 |
| Device-resident optimizer and embeddings | — | **Fixed** October 19, 2026 | TD-468, TD-547 |
| A lost GPU retried every sample; an epoch with no update recorded as loss 0 | — | **Fixed November 10, 2026** | TD-566 |
| TD-574 diagnostics: phase sync points and the last GPU work on a lost device (§7) | Off by default | Added November 13, 2026 | [TD-574](../../guides/TECHNICAL_DEBT.md#td-574-arc-engine-resets-during-device-resident-training-find-the-gpu-work-they-interrupt) |
| The whole class below `stable` | Still changing shape (it changed again today) | — | TD-039 |

## 12. History

- **October 17, 2026:** skipped pairs (TD-267), mid-window and last-window gradients (TD-272), `<eos>` kept (TD-275), double `<bos>` (TD-361).
- **October 19–21, 2026:** training state across passes (TD-269, TD-270); the device-resident optimizer (TD-468) and embeddings (TD-547); no
  hippocampal updates while training (TD-343).
- **November 4, 2026:** every epoch ended (TD-268), the deterministic split (TD-271), quality reports (TD-273), the best-weights directory
  (TD-274), validation accuracy and clean-ups (TD-276).
- **November 8, 2026:** dropout (TD-344), turn tokens and history (TD-295).
- **November 10, 2026:** a lost GPU stops the pass (TD-566).
- **November 13, 2026:** TD-574's engine-reset diagnostics (§7).
- **November 11, 2026:** progress notes (TD-569); the early-stop check after the epoch callback (TD-571). This document was re-traced against the code, replacing its October sections and folding in
  its addenda.
