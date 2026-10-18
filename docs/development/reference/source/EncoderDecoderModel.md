# `EncoderDecoderModel` — Source File Reference

- **Files:** [`src/EncoderDecoderModel.hpp`](../../../../src/EncoderDecoderModel.hpp), [`src/EncoderDecoderModel.cpp`](../../../../src/EncoderDecoderModel.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_models`; used by `ChatbotTrainer`/`incremental_trainer`, `ChatbotAPI`/`chatbot_api_server`, `chatbot_gui`, `RAGInference`, `RLHFTrainer`, `ModelSerializer`
- **Status tag:** `@adai-status: beta` (capped by TD-050), `@adai-version: 0.18.0`, `@adai-reviewed: 2026-09-18`
- **Tests:** [`tests/encoderdecoder_test.cpp`](../../../../tests/encoderdecoder_test.cpp) → `encoderdecoderTests` (83 tests)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real classes
> (linked against the debug libraries). The GPU-path finding in §4 was reproduced on the CPU with
> the exact decoder input the GPU path uses, because no GPU is available here.

---

## 1. What this file is

The **whole chatbot model**: the composition that turns text into a response and is trained on
`(input, response)` pairs.

```text
input text ─ tokenizer.encode(no specials) → LLMEncoder.encode_with_mask → encoder output ─┐
target ids ─ [BOS] + target[:-1] → LLMDecoder.forward_with_encoder(…, encoder output) → LanguageModelHead → logits
```

It owns:
- the `BPETokenizer`;
- `LLMEncoder`, `LLMDecoder` ([Decoder.md](Decoder.md)) and `LanguageModelHead`;
- the `TextGenerator`;
- optionally a frozen LeJEPA world model and a `HippocampalMemory` (TD-182/185/186).

It exposes forward/backward/loss for training, generation entry points for serving, checkpointing,
LoRA, and a parallel GPU path.

### Why it matters

It's where training semantics (what the model is taught to predict) and serving semantics (how it
generates) must agree. §4 is a case where they don't.

---

## 2. Construction and configuration

`EncoderDecoderModel(vocab, d_model, enc_layers, dec_layers, heads, d_ff, max_seq_length, world_model_inject_every_n_layers = 0)`:
- builds an empty `BPETokenizer` (callers load or replace it with `set_tokenizer()`, which takes
  ownership of a raw pointer);
- builds the encoder, decoder and LM head;
- builds a `TextGenerator` with `max_length = max_seq_length` and fixed seed 42.

Special-token IDs come from `SpecialTokenIDs` until `sync_special_tokens()`/`set_generation_config()`
copies them from the tokenizer.

| Method | Behaviour |
|---|---|
| `set_world_model()`/`set_hippocampal_memory()` + getters | Attach the TD-182/185 components |
| `set_hippocampal_repetition_params(…)` | Knobs passed to every gated forward |
| `set_training(mode)` | Sets the encoder's `requires_grad` and calls `decoder->set_training()` |
| `set_learning_rate(lr)` | **Decoder only**: the encoder call is commented out ("LLMEncoder doesn't have this method"), and the LM head isn't touched. That comment is stale: `LLMEncoder::set_learning_rate()` exists (TD-093 fix); only `LLMEncoder::update_weights()` is missing ([LLMEncoder.md](LLMEncoder.md)) |
| `get_*()` | Architecture fields, special-token IDs, the components, the generator |

---

## 3. Training API

| Method | Behaviour |
|---|---|
| `forward(input_ids, target_ids)` | Takes `model_mutex_`. Encodes the input with an all-ones mask. Builds the decoder input `[BOS] + target[:-1]` (teacher forcing, shifted). If a world model is attached, it **decodes the input ids back to text** and encodes them with the world model. Runs the decoder with world-model/memory, then the LM head, and returns logits `[len(target), vocab]` |
| `compute_loss(logits, target)` / `compute_loss_gradient()` | Per-row softmax cross-entropy against `target[t]` (inline softmax, TD-216). Both divide by `logits.rows`, including rows whose target is out of range |
| `backward(grad)` | Takes the lock; LM head → decoder (collecting the encoder gradient) → encoder |
| `train_step(text, text)` / `train_step_tokenized()` | zero → forward → loss → backward → `update_weights()` |
| `update_weights()` | `decoder->update_weights()` + `lm_head->update_weights()`. **Never updates the encoder** (commented out) |
| `zero_grad()`, `register_parameters(opt)` | All three components |
| `evaluate()`/`evaluate_tokenized()`/`compute_perplexity()` | Forward + loss in eval mode. Note `evaluate()` doesn't truncate; prefer the tokenized form |

> **`train_step()` never trains the encoder (verified).** Ten `train_step()` calls (no optimizer
> registered) moved a decoder self-attention weight (L1 Δ 0.0044) but left the encoder's **exactly
> unchanged** (Δ 0.000000). The LLMDecoder final norm is also never updated on this path (TD-346).
> The shipped `ChatbotTrainer` registers an optimizer and calls `optimizer->step()`, so it isn't
> affected. With an optimizer registered, `update_weights()` over-steps instead (TD-297).

> **Training inserts a double `BOS` (verified).** Targets come from `tokenizer->encode(response, true)`
> = `[BOS, w1…wn, EOS]` (in `train_step()` and in `ChatbotTrainer`'s tokenization), and `forward()`
> prepends another `BOS`. So the decoder input is `[BOS, BOS, w1…wn]`, and position 0 is trained to
> predict `BOS` from `BOS`. Generation follows suit: starting from `[BOS]`, a trained model first
> emits a second `BOS` (seen in the §4 experiment: `2 2 12 23 …`), spending one step and one slot of
> `max_length`. It's harmless to output text (decode skips specials), but wasteful and easy to
> misread.
>
> **Fixed in code October 17, 2026 (TD-361):** training targets are now `[w1…wn, EOS]`.
> `strip_leading_bos()` drops the tokenizer's `BOS` in `train_step()`, `evaluate()`,
> `gpu_evaluate()` and `ChatbotTrainer`'s tokenization, so `make_decoder_input()`'s `BOS` is the
> only one, and the decoder input is `[BOS, w1…wn]` on both CPU and GPU.
>
> New test `TrainedModelsFirstPredictionIsTheFirstWordNotBos` trains a tiny model on three pairs.
> Its first prediction after `[BOS]` is each target's first word on 6/6 runs. With the old targets
> the first prediction was `BOS` (token 2) on 3/3 runs. `ChatbotTrainer::kPreprocessFormatVersion`
> is now 3, so cached token IDs are rebuilt. This changes training alignment, so it belongs to the
> TD-358 retraining like everything else here.

---

## 4. GPU training path (verified by CPU reproduction)

`gpu_init_training()`, `gpu_zero_grads()`, `gpu_forward(input, target)`, `gpu_backward(scale)`,
`gpu_evaluate[_tokenized]()`, `gpu_download_grads()`, `gpu_sync_weights()` and `gpu_synchronize()`
are what `ChatbotTrainer` uses whenever `ADAI_ENABLE_GPU` is set. That means **every
`incremental_trainer` pass on ai-machine** (SYCL).

> **`gpu_forward()` trains the model to copy its input token, not predict the next one.**
> It calls `decoder->gpu_decode(target_tokens, …)` with the **unshifted** targets as the decoder
> input, then `matrix_cross_entropy_loss_gpu(logits, targets)` scores row *t* against `targets[t]`.
> Under the causal mask, position *t* can see `targets[t]` itself, so the loss is minimized by
> copying the current input token. The CPU `forward()` correctly feeds `[BOS] + target[:-1]`.
> `gpu_backward()` uses the same alignment, and `gpu_evaluate()` reports the same (misleadingly
> low) validation loss.
>
> **Reproduced on the CPU** with the same model classes, by training a tiny model on an echo task
> two ways for the same 150 epochs:
>
> | Decoder input during training | Final training loss | Greedy generation exactly right |
> |---|---|---|
> | `[BOS] + target[:-1]` (CPU `forward()`) | 0.0008 | **12 / 12** (`2 2 12 23 18 22 4 3` for `12 23 18 22 4`) |
> | `target` unshifted (GPU `gpu_forward()`) | 0.0006 | **0 / 12** (`2 2 2 17 17 17 17 17 …`) |
>
> Both reach near-zero loss, which is why the bug is invisible in training and validation metrics.
> But the GPU-style model can't generate: at inference it only ever sees tokens it has already
> produced, so a model that learned "output the token I was given" repeats itself.
>
> **Every checkpoint trained on the GPU path is affected** and needs retraining after the fix: shift
> the decoder input in `gpu_forward()` exactly as `forward()` does.
>
> **Fixed in code October 17, 2026 (TD-358):** `forward()` and `gpu_forward()` now build the decoder
> input with one shared helper, `make_decoder_input()` (`[BOS] + target[:-1]`); `gpu_forward()` rejects
> an empty target. Verified on the oneAPI CPU SYCL device with the real classes: before the change,
> `gpu_forward()`'s loss equalled the unshifted-input CPU loss exactly (e.g. 4.879566 vs a shifted
> 4.751482); after it, it equals the CPU `forward()` loss exactly (0.000000 difference on three pairs).
> `EncoderDecoderGPUTest.GpuForwardLossMatchesCpuForwardTeacherForcing` guards it (it also needs the
> TD-384 kernel fix to pass). The on-device run on ai-machine and the retraining remain.

---

## 5. Generation API

| Method | Path | Behaviour |
|---|---|---|
| `generate_response(text, max_length = 100)` | CPU | Lock, encode, optional world-model encode. **Beam** (if the generator's stored `num_beams > 1`): full recompute with world model and memory and invalid-ID masking. **Otherwise:** KV-cached `model_fn` with **no** world model/memory (TD-342) and **no** invalid-vocab masking. Writes the response to hippocampal memory |
| `generate_response_with_strategy(text, max_length, strategy, temperature, top_k, top_p, num_beams)` | CPU | Syncs all generation args into the generator (TD-100). Any `num_beams > 1` forces beam (TD-261). Otherwise uses a cached `model_fn` **with** invalid-vocab masking and dispatches greedy/sampling/topk/nucleus. An unknown strategy goes to `generator->generate()` |
| `gpu_generate_response(text, max_length)` | GPU | Beam → full `gpu_decode()`; else `GPUDecoderKVCache` + `gpu_decode_step()` |
| `gpu_generate_response_with_strategy(…)` | GPU | `ChatbotAPI`'s serving path on GPU builds. Accepts more strategy spellings; **unknown strategy → nucleus** (the CPU version uses `generate()`); `num_beams > 1` with a non-beam strategy is ignored (the CPU version forces beam) |

> **`max_length` is ignored by `generate_response()` and `gpu_generate_response()` (verified).**
> Neither forwards it to the generator, which uses its stored `max_length` (initially
> `max_seq_length`). `generate_response("hello world", 5)` produced 9 tokens with the generator at 64.
> `ChatbotTrainer`'s generation-quality metrics (`generation_quality_max_tokens = 50`) and label
> backfill call exactly this, so their generations can run to `max_seq_length`. The GPU version
> admits it in a comment ("kept for interface parity"). The `_with_strategy` variants do honour
> `max_length`.

By inspection:

- **The CPU and GPU generation paths differ in ways that change output:**
  - invalid-vocab masking happens only in the CPU strategy path and the CPU beam paths;
  - only the CPU paths write hippocampal memory or attach the world model (beam only);
  - unknown-strategy defaults differ;
  - so does how `num_beams` interacts with the strategy.
- **Hippocampal memory is written on every CPU generation**, including the trainer's label-backfill
  and quality-metric calls on the live model (`maybe_write_hippocampal_memory()` after
  `generate_response()`). Training-time evaluation therefore pollutes the episodic memory
  (compounds TD-343).
- **`forward()` re-derives world-model input** by decoding token ids back to text. The tokenizer's
  normalization (TD-224) makes that lossy, so the world model sees different text than the encoder
  did.

---

## 6. Checkpointing, cloning, LoRA

`save_model(path)` writes:
- `.config`: a magic number, format version 2 (Pre-LN), the architecture and special-token IDs;
- `.vocab`: the tokenizer (`save_vocab()` fails silently, TD-221);
- `.encoder`, `.decoder` (plus their per-block sibling files) and `.lm_head`.

`load_model(path)`:
- rejects a missing or old magic number ("pre-Pre-LN checkpoint, retrain") and an older format;
- rejects an architecture mismatch, including `max_seq_length`;
- loads the vocab and weights, and **restores special-token IDs onto the model but not into the
  generator's config** until `sync_special_tokens()` runs.

The config reads aren't checked. The main serving and training checkpoint path is SafeTensors via
`ModelSerializer`; this format is the legacy/fallback one.

`clone()` round-trips `save_model()`/`load_model()` through a temp file, so it inherits all of the
above. **It doesn't copy:**
- the world model or hippocampal memory;
- the generation config or repetition params;
- LoRA adapters (TD-302);
- optimizer state.

It's used by `ChatbotTrainer` for async BLEU/ROUGE snapshots (TD-023), which therefore generate
without world model or memory.

LoRA (`enable_lora()`, `has_lora()`, `register_lora_parameters()`, `merge_lora()`) cascades to every
encoder self-attention and decoder self/cross-attention (TD-038). It has no production caller.

---

## 7. Concurrency

`model_mutex_` serializes the CPU `forward()`/`backward()`/`generate_response*()` and the GPU
generation methods (TD-156, TD-033). `gpu_forward()`/`gpu_backward()` deliberately don't take it:
training is single-threaded. `train_step()` calls `forward()` and `backward()` separately, each
locking on its own, so a concurrent caller could interleave between them. Only tests use
`train_step()`.

---

## 8. Tests

`encoderdecoder_test.cpp` has 83 tests:
- construction;
- forward shapes;
- loss and gradient;
- train steps;
- evaluate and perplexity;
- generation strategies;
- save/load (including the Pre-LN magic);
- clone;
- LoRA;
- the world-model/memory wiring;
- concurrency guards.

**Not covered:**
- GPU training alignment (no GPU in CI) — the §4 bug;
- `generate_response()`'s `max_length`;
- encoder updates through `train_step()`.

---

## 9. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `gpu_forward()`/`gpu_backward()`/`gpu_evaluate()` use unshifted targets as decoder input (§4) | **GPU-trained models learn to copy instead of predict**: near-zero loss, unusable generation; every GPU checkpoint needs retraining | **Verified** (CPU reproduction of the same alignment); **code fix landed October 17, 2026** (on-device run and retraining remain) | [TD-358](../../guides/TECHNICAL_DEBT.md#td-358-gpu-training-feeds-unshifted-targets-teaching-the-model-to-copy-instead-of-predict) |
| `generate_response()`/`gpu_generate_response()` ignore `max_length` (§5) | Trainer quality metrics/backfill generate up to `max_seq_length`; caller limits silently ignored | **Verified** | [TD-359](../../guides/TECHNICAL_DEBT.md#td-359-generate_response-and-gpu_generate_response-ignore-max_length) |
| No-optimizer path never updates (or sets the LR of) the encoder (§3) | `train_step()`/`update_weights()` train only decoder + head | **Verified** | [TD-360](../../guides/TECHNICAL_DEBT.md#td-360-encoderdecodermodel-no-optimizer-path-never-updates-the-encoder) |
| Double `BOS` in training targets (§3) | Wasted step and `max_length` slot in every generation | **Verified**; **fixed in code October 17, 2026** | [TD-361](../../guides/TECHNICAL_DEBT.md#td-361-training-targets-get-a-double-bos) |
| CPU vs GPU generation inconsistencies; invalid-vocab masking missing on some paths; hippocampal writes from training-time generation; lossy world-model input via decode (§5) | Different output by path/device; memory pollution | By inspection | [TD-362](../../guides/TECHNICAL_DEBT.md#td-362-cpu-and-gpu-generation-paths-behave-differently), [TD-343](../../guides/TECHNICAL_DEBT.md#td-343-hippocampal-memory-is-updated-on-every-forward-pass) (amended) |
| `clone()` drops world model/memory/gen config; `load_model()` unchecked reads and generator special tokens not synced (§6) | Async metric snapshots differ from the live model; truncated configs load garbage | By inspection | [TD-363](../../guides/TECHNICAL_DEBT.md#td-363-encoderdecodermodel-clone-load-and-world-model-input-gaps) |
| Beam forced by `num_beams > 1` (CPU) | Already tracked | — | TD-261 |
| Gated paths skipped by cached generation | Already tracked | — | TD-342 |
| Shared-optimizer over-step via `update_weights()` | Already tracked | — | TD-297 |

---

## 10. Merge notes

This file replaces `docs/development/api/transformer/encoder-decoder-model.md` (about 1,100 lines).
Where they disagree, this file wins:

- **`generate_response()`'s model function.** The old doc's version used a non-cached
  `forward_with_encoder()`. Today's plain path is KV-cached, and beam uses a separate non-cached
  function.
- **`update_weights()` and `train_step()`.** The old doc presented them as complete training
  methods without noting that they skip the encoder.
- **Missing features.** There was nothing on the GPU path, the world model/hippocampal wiring,
  LoRA, `clone()`, the checkpoint magic/format version, or `model_mutex_`.

**Kept from the old doc:** the pipeline overview, component roles, the loss formulas and the
generation-strategy list.
