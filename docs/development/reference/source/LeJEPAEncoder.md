# `LeJEPAEncoder` — Source File Reference

- **Files:** [`src/LeJEPAEncoder.hpp`](../../../../src/LeJEPAEncoder.hpp) (473 lines), [`src/LeJEPAEncoder.cpp`](../../../../src/LeJEPAEncoder.cpp) (890)
- **Namespace:** global
- **Built into:** `adai_lejepa`; used by `incremental_trainer --objective=lejepa` (pretraining, [IncrementalTrainingTool.md](IncrementalTrainingTool.md)) and attached frozen to the chatbot by `IncrementalTrainer::maybe_attach_world_model()` and `chatbot_api_server`
- **Status tag:** `@adai-status: beta` (capped by TD-504; TD-566's device-loss stop doesn't cover LeJEPA passes); `.cpp` `@adai-version: 0.13.0`, `.hpp` `0.12.0`, both `@adai-reviewed: 2026-11-22` (training on the device, TD-588; prepared texts and device diagnostics, TD-589)
- **Tests:** [`tests/lejepaencoder_test.cpp`](../../../../tests/lejepaencoder_test.cpp) (51 tests, plus 5 device tests in GPU builds, passing on the oneAPI CPU device)
- **Design:** [lejepa_world_model_gated_injection_plan.md](../../../proposals/lejepa_world_model_gated_injection_plan.md) (TD-176–TD-189, TD-196)
- **Last traced against the code:** 2026-11-19 (§2, §3's smaller items, §4 and §5; the addenda record earlier changes); device training 2026-11-22 (TD-588, TD-589)

> Code-traced reference. Behaviour marked **verified** was confirmed by training and saving/reloading
> the real class in a scratch program.

---

## 1. What this file is

The **world model**: a standalone transformer encoder (same shape as `LLMEncoder`: embedding, positional
encoding, `EncoderBlock` × N, final `LayerNorm`), pretrained with a LeJEPA-style self-supervised
objective and then attached **frozen** to the chatbot. The chatbot's `DecoderBlock`s cross-attend to its
output through gated layers (TD-180), gates which currently never train (TD-341).

```text
train_step(text):
  tokens = encode(text, BOS/EOS), truncated to max_seq_length
  context view = tokens with a random contiguous 25% span replaced by UNK
  ctx = encoder(context view), tgt = encoder(tokens)
  predicted = Predictor(ctx)                       (FeedForward, d_model → d_ff → d_model)
  predictor_loss = MSE(predicted, tgt) over the masked span only
  sigreg_loss   = SIGReg(tgt)                      (isotropic-Gaussian regularizer, random sketches)
  backward: target view ← −∂MSE + λ·∂SIGReg  → update;  predictor + context view ← ∂MSE → update
```

There's no stop-gradient or EMA teacher on the target view (LeJEPA's premise: SIGReg alone prevents
collapse), so both views train the same encoder.

---

## 2. API

| Member | Behaviour |
|---|---|
| `LeJEPAEncoder(vocab, d_model = 512, layers = 6, heads = 8, d_ff = 2048, max_seq = 512, sketches = 64)` | Builds the encoder, a `Predictor(d_model, d_ff)` and a `SIGReg(d_model, sketches)`; a `BPETokenizer` in Unicode mode (not loaded; TD-416/TD-564, `set_tokenizer_mode()`); logs its configuration in one line at debug level (TD-429; it printed an 8-line banner to `std::cout`) |
| `encode(text)` | Tokenizes with `BOS`/`EOS`, truncates with `resize` (dropping `EOS` when too long, as in TD-364), and runs `encode_tokens()` |
| `train_step(text)` | §1; returns `{predictor_loss, sigreg_loss}` and records diagnostics (masking ratio, cosine similarity, SIGReg variance mean/stddev, gradient norm). Throws `invalid_argument` under 2 tokens. With `requires_grad = false` it only computes the losses |
| `set_sigreg_lambda()`, `set_requires_grad()`, `set_learning_rate()` (not the predictor), `register_parameters_with_optimizer()` (encoder + predictor; SIGReg has no parameters) | Configuration |
| `train_batch(texts)` | The pass's entry point (TD-503, see its addendum). A text that fails to tokenize (empty, or invalid UTF-8: `TokenizerEncodingError`) or has under 2 tokens is skipped; the rest train |
| `load_tokenizer_vocab()`, `build_tokenizer()` | The encoder's own tokenizer; **doesn't resize the embedding** (callers probe the vocab size first). Log at info |
| `save(dir)` / `load(dir)` | `config.bin` (6 ints), `token_embedding.bin`, `encoder_block_<i>.bin` (+ sidecars), `final_norm.bin`, `predictor.bin` (TD-427). The config's writes and reads are checked (`adai::binio`, TD-429): a failed write throws, and a short or over-long `config.bin` throws before anything changes (it used to fail as a misleading architecture mismatch). `load()` throws on a mismatch of five dimensions; a different `max_seq_length` loads with a `Logger` warning, since no weight depends on it (TD-429). The sub-loaders check their own files (TD-423, TD-426). Both log the directory at info |
| Accessors | `get_encoder_block()`, `get_num_layers()`, `get_embedding_dim()`, `get_sigreg_num_sketches()`, `get_last_*()` diagnostics; `print_config()` (one info line, TD-429) |

---

## 3. Problems

### The predictor isn't saved, so every pass restarts it from random (verified)

`save()` writes the encoder only, and `Predictor` has no save or load of its own. `incremental_trainer
--objective=lejepa` resumes the encoder from `<SESSION_DIR>/world_model` each pass (TD-184), but gets a
**freshly initialized predictor** (and a fresh Adam state). Verified: after 60 epochs of training, the
trained instance's predictor loss was **0.0017**; the same weights reloaded through `save()`/`load()`
gave **2.053**. Each pass therefore starts by re-learning the predictor. Meanwhile, the random
predictor's large gradients flow back into the resumed encoder through the context view, pulling the
already-trained representation around until the predictor catches up. The pass-to-pass predictor-loss
curve on the dashboard (stable session key `lejepa-<model>`) restarts from about 2.0 every pass for
this reason, not because of anything in the data.

### The two-update workaround is now unnecessary, and its rationale is out of date

`train_step()` applies the target-view gradient and the context-view gradient as **two separate
`update_weights()` calls**: two optimizer steps per sample, with the context forward pass re-run on
weights the first step just changed. The in-code TD-189 comment explains this as necessary because
`LayerNorm`/`FeedForward`/`MultiHeadAttention::backward()` *overwrite* their gradient members. That's no
longer true: TD-195 made all three accumulate (their status tags and code confirm it). Consequences of
keeping the workaround:
- Adam's step counter advances twice per sample.
- Every parameter, including the predictor during step 1 (its gradient is zero then), is moved by
  momentum twice.
- The objective is optimized as two sequential half-steps rather than one combined gradient.

One `backward()` for each view into accumulating gradients, followed by one `update_weights()`, is now
possible.

**Fixed November 8, 2026 (TD-428):** each text now takes that one update, so it gets one optimizer step instead of two.

### Smaller items (by inspection)

- **An escaping exception type (fixed).** `TokenizerEncodingError` derives from `runtime_error`, so it escaped the pass's
  old `catch (const std::invalid_argument&)` and ended the pass. Since TD-503, `train_batch()` skips a text that fails to tokenize and the
  pass (`run_lejepa_training_pass()` in `LeJEPATrainingPass.cpp`) catches any exception per batch, or, since TD-566, stops when the GPU
  is lost. `TrainBatchSkipsInvalidUtf8` (TD-429) covers invalid UTF-8, the case real books can hit.
- **Hard-coded choices.** The mask ratio is fixed at 25% (`kMaskRatio`). The span RNG follows `SEED` (`adai::next_init_seed()`,
  TD-368); it was seeded from `random_device`.
- **Tokenizer mode:** fixed. It was always ASCII; it follows `TOKENIZER_MODE` since TD-416 and defaults to Unicode since TD-564.
- **Console output and unchecked config reads:** fixed (TD-429, see §2).

---

## 4. Tests

`lejepaencoder_test.cpp` (50 tests) covers:
- shape-compatibility with `LLMEncoder`;
- `encode()` shapes;
- `train_step()` (losses finite, decreasing over steps, `< 2`-token rejection, `requires_grad = false` no-update);
- optimizer registration (single step per update, TD-178);
- diagnostics;
- save/load of the encoder (including architecture mismatch) and of the predictor (TD-427);
- `train_batch()`, including skipped texts and invalid UTF-8 (TD-503, TD-429);
- short and over-long `config.bin`, a different `max_seq_length`, and nothing written to stdout (TD-429);
- checkpoint directory resolution (TD-548) and tokenizer mode (TD-416, TD-564).

---

## 5. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `save()`/`load()` skip the `Predictor`; every resumed pass gets a random predictor (§3) | Predictor re-learned each pass; its random gradients disturb the resumed encoder; misleading loss curve | **Verified** (0.0017 → 2.053) | [TD-427](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-427-lejepa-predictor-is-never-saved-so-each-pass-restarts-it) |
| Two sequential `update_weights()` per sample, justified by a TD-189 overwrite problem that TD-195 fixed (§3) | Two optimizer steps per sample; momentum applied twice; stale rationale | By inspection | [TD-428](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-428-lejepa-two-update-workaround-is-obsolete-after-td-195) |
| `TokenizerEncodingError` escapes the caller's `invalid_argument` catch; `std::cout`; unchecked loads; `max_seq_length` not compared (§3) | — | **Fixed November 19, 2026** | [TD-429](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-429-lejepaencoder-exception-escape-console-output-and-unchecked-loads) |
| SIGReg's frequency grid, NaN inputs and repeated projections | Caps the status at `beta` | By inspection | [TD-504](../../guides/TECHNICAL_DEBT.md#td-504-sigreg-frequency-grid-nan-inputs-and-repeated-projections) |
| `train_batch()` trains on the host even with a GPU in use: only `Matrix::multiply()`'s implicit dispatch reaches the device, with an upload and download per product; the view, loss and pooling loops are host-only; texts are re-tokenized every epoch (filed November 22, 2026) | A LeJEPA pass does its math on the CPU, against the owner rule that GPU training stays on the device | **Fixed November 22, 2026** for GPU builds: `gpu_begin_training()` and `gpu_train_batch()` (addenda below); re-tokenization every epoch fixed by TD-589 | [TD-587](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-587-lejepas-view-loss-and-pooling-arithmetic-is-host-loops), [TD-588](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-588-lejepa-has-no-device-resident-training-step), [TD-589](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-589-the-lejepa-pass-re-tokenizes-every-epoch-and-reads-diagnostics-on-the-host) |

---

## Addendum (October 20, 2026): `gpu_encode()` (TD-342)

`gpu_encode(text)` (GPU builds, 0.5.0) is `encode()` on the device: the same tokenization and truncation, then the device embedding
and positional encoding, encoder blocks and final norm. Weights are uploaded on first use and again after `load()`. It's inference
only, for the frozen world model whose output feeds the decoder's gated path during GPU training and generation. Test:
`gpu_optimizer_test.cpp`'s `WorldModelGpuEncodeMatchesEncode`.

---

## Addendum (October 20, 2026): the predictor is saved (TD-427)

`save()` writes `predictor.bin` (the `Predictor`'s `FeedForward` weights), and `load()` restores it. A checkpoint without one (saved
before this fix) or with an unreadable one logs a warning and keeps a fresh predictor. `get_predictor()` returns it. Three
consecutive real `--objective=lejepa` passes now resume the predictor (average loss 1.07 → 0.33 → 0.22); before, each pass restarted it.
Tests: `SaveAndLoadRestoreThePredictor`, `LoadOfACheckpointWithoutAPredictorStillLoads`.

---

## Addendum (October 21, 2026): `train_batch()` (TD-503)

`train_batch(texts)` applies SIGReg across the batch's mean-pooled target embeddings, one row per text, instead of one text's tokens.
- **Phase 1** pools and takes SIGReg's loss and gradient.
- **Phase 2** trains each text like `train_step()`, adding its pooled row's SIGReg gradient (× λ × batch size, to keep the batch
  objective's balance with per-text steps) spread over its tokens.
- **Result:** `BatchStepResult` gives the trained indices, per-text predictor losses and the batch SIGReg loss. Unusable texts are
  skipped.
- **Shared code:** `train_step()` keeps the per-sequence form; both use the private `train_tokens()`.

The LeJEPA pass trains in batches of `WORLD_MODEL_SIGREG_BATCH_SIZE`. Tests: `TrainBatchSkipsUnusableTextsAndReportsTheTrainedOnes`,
`TrainBatchOfOneHasNoSigregLoss`, `TrainBatchSigregPullsPooledEmbeddingsTowardGaussian`.

---

## Addendum (October 21, 2026): checkpoint directory (TD-548)

The default world-model checkpoint directory is now `<SESSION_DIR>/world_model_checkpoint`. `<SESSION_DIR>/world_model` is also the
standalone `world_model` dataset pool, so the old default collided with it.
- `default_checkpoint_dir()` returns the new default.
- `has_checkpoint()` checks for `config.bin`, not just the directory.
- `resolve_checkpoint_dir(session_dir, artifact_path)` returns the artifact path when set, else the new default, else an old-layout
  checkpoint.

Test: `CheckpointDirectoryResolution`.

## Addendum (October 26, 2026): `clone()` (TD-363)

`clone()` returns an independent copy through a temporary `save()`/`load()`: weights, predictor, tokenizer vocabulary, SIGReg λ and
sketch count, learning rate. No optimizer is attached. `EncoderDecoderModel::clone()` uses it for the copy's world model.

## Addendum (November 2, 2026): `set_tokenizer_mode()` (TD-416)

`set_tokenizer_mode(mode)` starts a fresh tokenizer in that mode; call it before `load_tokenizer_vocab()`.
`get_tokenizer_mode()` reports it. The tokenizer used to be ASCII always; as for the chatbot's tokenizer, a vocabulary
file that records its own `TOKENIZER_MODE` still overrides the mode on load.

## Addendum (November 19, 2026): logging and checked config I/O (TD-429)

The constructor logs one debug line and save/load, the tokenizer methods and `print_config()` log at info, instead of writing to
`std::cout`. `config.bin` is written and read through `adai::binio`, so a short or over-long config throws, and a different
`max_seq_length` loads with a warning. The exception escape was already gone (TD-503); `TrainBatchSkipsInvalidUtf8` now covers it.
Tests: `ShortOrOverLongConfigThrowsAndLeavesTheEncoderUnchanged`, `DifferentMaxSeqLengthLoadsWithAWarning`, `WritesNothingToStdout`,
`TrainBatchSkipsInvalidUtf8`.

## Addendum (November 22, 2026): training on the device (TD-588)

**Turning it on.** `gpu_begin_training()` (GPU builds) uploads every weight (embedding, blocks, final norm, predictor and the positional
table, through `gpu_ensure_weights()`, which `gpu_encode()` shares) and binds a `GPUOptimizer` to the optimizer given to
`register_parameters_with_optimizer()` (`std::logic_error` without one). It returns the number of groups left on the host; all of them
have device copies, so it's 0. From then on `train_batch()` tokenizes as before (`tokenize_batch()`) and hands the IDs to
`gpu_train_batch()`. The device weights and Adam moments are the live copy: `sync_host_from_device()` downloads both, and must come
before `save()` (`run_lejepa_training_pass()` calls it before every checkpoint). `gpu_encode()` uses the device weights while they are
current, so a world model trained this way encodes with its trained weights.

**`gpu_train_batch(ids)`** has `train_batch()`'s semantics with the host math replaced by TD-585 to TD-587's device pieces:
- the batch's IDs go up in one copy (each text 2 to `max_seq_length` IDs, `std::invalid_argument` otherwise);
- phase 1 encodes each target view on the device and mean-pools it into SIGReg's batch; `SIGReg::gpu_compute()` takes the loss, the
  variance stats and the pooled gradient;
- phase 2, per text: `choose_span()` (shared with the host step, so both draw the same spans from `span_rng_`), the context view's IDs
  built on the device, the context and target encoder passes through `TokenEmbedding`'s device-ID lookup, the predictor, the masked
  loss and gradients into per-text device floats, the pooled SIGReg gradient spread over the text (`λ·batch/seq_len`), the backward
  through the target view, the predictor and (after re-encoding) the context view, then one `GPUOptimizer` step and zeroed device
  gradients. The re-encode stays: each layer's device state holds one set of activations, and a second set per layer would save one
  forward pass per text but touch every layer.
- The batch's results (SIGReg loss, variance mean and stddev, each text's predictor loss and cosine) collect in one device buffer and
  come down in one download at the end; the optimizer reads its one float per group per step, as in chatbot training.

**Checked against the host** (`DeviceTrainingMatchesHostTraining`, oneAPI CPU device): same weights, vocabulary, span seed and SIGReg
seed (`seed_sigreg()`, new), two batches: the same texts skipped, per-text losses, SIGReg loss, masking ratio, cosine, gradient norm
and variance mean within float tolerance, the same final encodings, and `gpu_encode()` matching. `DeviceTrainedWeightsSurviveSaveAfterSync`
checks the host copy is stale until synced and a checkpoint after the sync restores the trained weights. Copies are deleted
(`LeJEPAEncoder` was never copyable in practice; it holds `unique_ptr`s), and the destructor moved to the `.cpp`.

## Addendum (November 22, 2026): prepared texts and device diagnostics (TD-589)

`prepare_texts(texts)` tokenizes a pass's texts once (`tokenize_one()`, the same truncation and skips as `train_batch()`) and
`train_prepared(first, last)` trains the range `[first, last)` with them, on the host (`train_ids_host()`) or the device
(`gpu_train_core()`) as `train_batch()` would; its `trained` counts from `first`. On the device every prepared text's IDs go up in one
upload, when training first needs them, and stay for the pass, so an epoch moves no IDs. `texts_tokenized()` counts tokenizations (a
pass now tokenizes each text once, not once per epoch). `train_batch()` and `gpu_train_batch()` still tokenize and upload per call;
both share the cores.

`set_gpu_diagnostics(acc)` points every encoder block's feed-forward saturation accumulator at `acc[0..1]` and attention entropy
accumulator at `acc[2..3]` ([FeedForward.md](FeedForward.md), [MultiHeadAttention.md](MultiHeadAttention.md)); `nullptr` clears them.

Tests: `PreparedTextsTrainLikeTrainBatchAndTokenizeOnce` (host: identical losses over two epochs, each text tokenized once),
`PreparedTextsOnTheDeviceMatchTheHost`, `DeviceDiagnosticsAccumulateOnTheDevice` (one sample per block call, saturation matching the host
hook, `nullptr` stopping it).
