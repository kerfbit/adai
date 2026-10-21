# `LeJEPAEncoder` — Source File Reference

- **Files:** [`src/LeJEPAEncoder.hpp`](../../../../src/LeJEPAEncoder.hpp), [`src/LeJEPAEncoder.cpp`](../../../../src/LeJEPAEncoder.cpp)
- **Namespace:** global
- **Built into:** `adai_lejepa`; used by `incremental_trainer --objective=lejepa` (pretraining, [IncrementalTrainingTool.md](IncrementalTrainingTool.md)) and attached frozen to the chatbot by `IncrementalTrainer::maybe_attach_world_model()` and `chatbot_api_server`
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.4.0`, `@adai-reviewed: 2026-09-17`
- **Tests:** [`tests/lejepaencoder_test.cpp`](../../../../tests/lejepaencoder_test.cpp) (37 tests)
- **Design:** [lejepa_world_model_gated_injection_plan.md](../../../proposals/lejepa_world_model_gated_injection_plan.md) (TD-176–TD-189, TD-196)
- **Last traced against the code:** 2026-10-14

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
| `LeJEPAEncoder(vocab, d_model = 512, layers = 6, heads = 8, d_ff = 2048, max_seq = 512, sketches = 64)` | Builds the encoder, a `Predictor(d_model, d_ff)` and a `SIGReg(d_model, sketches)`; an ASCII `BPETokenizer` (not loaded); prints a banner to `std::cout` |
| `encode(text)` | Tokenizes with `BOS`/`EOS`, truncates with `resize` (dropping `EOS` when too long, as in TD-364), and runs `encode_tokens()` |
| `train_step(text)` | §1; returns `{predictor_loss, sigreg_loss}` and records diagnostics (masking ratio, cosine similarity, SIGReg variance mean/stddev, gradient norm). Throws `invalid_argument` under 2 tokens. With `requires_grad = false` it only computes the losses |
| `set_sigreg_lambda()`, `set_requires_grad()`, `set_learning_rate()` (not the predictor), `register_parameters_with_optimizer()` (encoder + predictor; SIGReg has no parameters) | Configuration |
| `load_tokenizer_vocab()`, `build_tokenizer()` | The encoder's own tokenizer; **doesn't resize the embedding** (callers probe the vocab size first) |
| `save(dir)` / `load(dir)` | `config.bin` (6 ints), `token_embedding.bin`, `encoder_block_<i>.bin` (+ sidecars), `final_norm.bin`. `load()` checks five dimensions, not `max_seq_length`; the sub-loaders' reads are unchecked (TD-357/TD-426 class). **The predictor isn't saved** (§3) |
| Accessors | `get_encoder_block()`, `get_num_layers()`, `get_embedding_dim()`, `get_sigreg_num_sketches()`, `get_last_*()` diagnostics; `print_config()` (stdout) |

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

### Smaller items (by inspection)

- **An escaping exception type.** `TokenizerEncodingError` derives from `runtime_error`, so it escapes
  `run_lejepa_training_pass()`'s `catch (const std::invalid_argument&)` and ends the pass.
- **Hard-coded and unseeded choices.** The span RNG is seeded from `random_device` (TD-368); the mask
  ratio is fixed at 25% (`kMaskRatio`).
- **The tokenizer is always ASCII** (TD-416).
- **Console output:** `std::cout` in the constructor, save/load and the tokenizer methods.
- **Unchecked reads.** `load()` ignores `max_seq_length` and inherits the sub-loaders' unchecked reads.

---

## 4. Tests

`lejepaencoder_test.cpp` (37 tests) covers:
- shape-compatibility with `LLMEncoder`;
- `encode()` shapes;
- `train_step()` (losses finite, decreasing over steps, `< 2`-token rejection, `requires_grad = false` no-update);
- optimizer registration (single step per update, TD-178);
- diagnostics;
- save/load of the encoder (including architecture mismatch).

**Not covered:** the predictor surviving save/load.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `save()`/`load()` skip the `Predictor`; every resumed pass gets a random predictor (§3) | Predictor re-learned each pass; its random gradients disturb the resumed encoder; misleading loss curve | **Verified** (0.0017 → 2.053) | [TD-427](../../guides/TECHNICAL_DEBT.md#td-427-lejepa-predictor-is-never-saved-so-each-pass-restarts-it) |
| Two sequential `update_weights()` per sample, justified by a TD-189 overwrite problem that TD-195 fixed (§3) | Two optimizer steps per sample; momentum applied twice; stale rationale | By inspection | [TD-428](../../guides/TECHNICAL_DEBT.md#td-428-lejepa-two-update-workaround-is-obsolete-after-td-195) |
| `TokenizerEncodingError` escapes the caller's `invalid_argument` catch; `std::cout`; unchecked loads; `max_seq_length` not compared (§3) | One bad sample ends a pass; console noise | By inspection | [TD-429](../../guides/TECHNICAL_DEBT.md#td-429-lejepaencoder-exception-escape-console-output-and-unchecked-loads) |
| Gates never train; unseeded RNG; ASCII-only tokenizer | Already tracked | — | TD-341, TD-368, TD-416 |

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
