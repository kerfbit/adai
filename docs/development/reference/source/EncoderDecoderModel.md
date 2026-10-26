# `EncoderDecoderModel` — Source File Reference

- **Files:** [`src/EncoderDecoderModel.hpp`](../../../../src/EncoderDecoderModel.hpp), [`src/EncoderDecoderModel.cpp`](../../../../src/EncoderDecoderModel.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_models`; used by `ChatbotTrainer`/`incremental_trainer`, `ChatbotAPI`/`chatbot_api_server`, `chatbot_gui`, `RAGInference`, `RLHFTrainer`, `ModelSerializer`
- **Status tag:** `@adai-status: beta` (capped by TD-050), `EncoderDecoderModel.cpp` `@adai-version: 0.19.1`, `@adai-reviewed: 2026-10-19`; `.hpp` `0.19.0`, `2026-10-18`
- **Tests:** [`tests/encoderdecoder_test.cpp`](../../../../tests/encoderdecoder_test.cpp) → `encoderdecoderTests` (90 tests)
- **Last traced against the code:** 2026-10-19

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

> **Fixed (TD-361, October 17, 2026): training used to insert a double `BOS`.** Targets came from
> `tokenizer->encode(response, true)` = `[BOS, w1…wn, EOS]`, and `forward()` prepends another `BOS`.
> So the decoder input was `[BOS, BOS, w1…wn]`, and generation from `[BOS]` first emitted a second
> `BOS` (`2 2 12 23 …`), spending one step and one slot of `max_length`.
>
> Training targets are now `[w1…wn, EOS]`. `strip_leading_bos()` drops the tokenizer's `BOS` in
> `train_step()`, `evaluate()`, `gpu_evaluate()` and `ChatbotTrainer`'s tokenization. So
> `make_decoder_input()`'s `BOS` is the only one, and the decoder input is `[BOS, w1…wn]` on both CPU
> and GPU. `TrainedModelsFirstPredictionIsTheFirstWordNotBos` checks it: with the new targets, the
> first prediction after `[BOS]` is the target's first word on 6/6 runs; with the old ones it was
> `BOS` on 3/3. `ChatbotTrainer::kPreprocessFormatVersion` went to 3, so cached token IDs are rebuilt.
> Rechecked October 19, 2026: no other training-target path still keeps the `BOS`. The remaining
> `encode(…, true)` calls are encoder inputs (TD-364) and the deferred RLHF code (TD-500).

---

## 4. GPU training path (verified by CPU reproduction)

`gpu_init_training()`, `gpu_zero_grads()`, `gpu_forward(input, target)`, `gpu_backward(scale)`,
`gpu_evaluate[_tokenized]()`, `gpu_download_grads()`, `gpu_sync_weights()` and `gpu_synchronize()`
are what `ChatbotTrainer` uses whenever `ADAI_ENABLE_GPU` is set. That means **every
`incremental_trainer` pass on ai-machine** (SYCL).

> **Fixed (TD-358, October 17, 2026): `gpu_forward()` used to train the model to copy its input
> token.** It called `decoder->gpu_decode(target_tokens, …)` with the **unshifted** targets as the
> decoder input, then scored row *t* against `targets[t]`. Under the causal mask, position *t* could
> see `targets[t]` itself, so copying minimized the loss. `gpu_backward()` used the same alignment,
> and `gpu_evaluate()` reported the same misleadingly low loss. A CPU reproduction on an echo task
> (150 epochs each) showed why this went unnoticed:
>
> | Decoder input during training | Final training loss | Greedy generation exactly right |
> |---|---|---|
> | `[BOS] + target[:-1]` (CPU `forward()`) | 0.0008 | **12 / 12** |
> | `target` unshifted (old `gpu_forward()`) | 0.0006 | **0 / 12** (`2 2 2 17 17 17 …`) |
>
> Now `forward()` and `gpu_forward()` build the decoder input with one shared helper,
> `make_decoder_input()` (`[BOS] + target[:-1]`), and the targets stay the labels. `gpu_forward()`
> rejects an empty target, which would otherwise make the logits and label row counts disagree.
> Verified on the oneAPI CPU SYCL device with the real classes:
> - **Before:** `gpu_forward()`'s loss equalled the unshifted-input CPU loss (4.879566, against 4.751482
>   shifted).
> - **After:** it equals the CPU `forward()` loss exactly.
>
> `EncoderDecoderGPUTest.GpuForwardLossMatchesCpuForwardTeacherForcing` guards it. That test also needs
> the TD-384 kernel fix. Still to do: the on-device run on ai-machine, then retraining every GPU-trained
> checkpoint.

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
- `.vocab`: the tokenizer. `save_vocab()` throws on failure (TD-221), so `save_model()` throws rather than leave a checkpoint without its vocab;
- `.encoder`, `.decoder` and `.lm_head`, plus their sibling component files, all named `<path>_…`:
  - the encoder's `_encoder_token_emb.bin`, `_encoder_block_<i>.bin` and `_encoder_final_norm.bin`;
  - the decoder's `_token_emb.bin`, `_decoder_block_<i>.bin`… and `_final_norm.bin`.

  TD-538 (October 18, 2026): the encoder used to write the decoder's names too, so every checkpoint
  lost the encoder's embedding and norm; it now loads the old names as a fallback. Every file of a
  checkpoint starts with `<path>.` or `<path>_` (`IncrementalTrainer::checkpoint_files()` relies on that).

`load_model(path)`:
- rejects a missing or old magic number ("pre-Pre-LN checkpoint, retrain") and an older format;
- rejects an architecture mismatch, including `max_seq_length`;
- loads the vocab and weights, and **restores special-token IDs onto the model but not into the
  generator's config** until `sync_special_tokens()` runs.

The config reads aren't checked. This is the format every trainer checkpoint and `MODEL_PATH` uses;
the SafeTensors path in `ModelSerializer` has no production caller (TD-465).

`clone()` round-trips `save_model()`/`load_model()` through a temp checkpoint in the system temp
directory, so it inherits all of the above. Since TD-544 (October 19, 2026) it removes every file of
that temp checkpoint afterwards. It used to remove only the five top-level files, so each clone left
the component files (embeddings, every block, the norms) behind: 115 clones' worth, 4.6 MB, had
accumulated in this machine's `/tmp` from test runs alone, and a full-size model leaves its weights
on every epoch with quality metrics on. `CloneLeavesNoTempFiles` covers it. **It doesn't copy:**
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
| `gpu_forward()`/`gpu_backward()`/`gpu_evaluate()` use unshifted targets as decoder input (§4) | **GPU-trained models learn to copy instead of predict**: near-zero loss, unusable generation; every GPU checkpoint needs retraining | **Fixed in code October 17, 2026** (was verified by CPU reproduction); on-device run and retraining remain | [TD-358](../../guides/TECHNICAL_DEBT.md#td-358-gpu-training-feeds-unshifted-targets-teaching-the-model-to-copy-instead-of-predict) |
| `generate_response()`/`gpu_generate_response()` ignore `max_length` (§5) | Trainer quality metrics/backfill generate up to `max_seq_length`; caller limits silently ignored | **Verified** | [TD-359](../../guides/TECHNICAL_DEBT.md#td-359-generate_response-and-gpu_generate_response-ignore-max_length) |
| No-optimizer path never updates (or sets the LR of) the encoder (§3) | `train_step()`/`update_weights()` train only decoder + head | **Verified** | [TD-360](../../guides/TECHNICAL_DEBT.md#td-360-encoderdecodermodel-no-optimizer-path-never-updates-the-encoder) |
| Double `BOS` in training targets (§3) | Wasted step and `max_length` slot in every generation | **Fixed October 17, 2026** | [TD-361](../../guides/TECHNICAL_DEBT.md#td-361-training-targets-get-a-double-bos) |
| CPU vs GPU generation inconsistencies; invalid-vocab masking missing on some paths; hippocampal writes from training-time generation; lossy world-model input via decode (§5) | Different output by path/device; memory pollution | By inspection | [TD-362](../../guides/TECHNICAL_DEBT.md#td-362-cpu-and-gpu-generation-paths-behave-differently), [TD-343](../../guides/TECHNICAL_DEBT.md#td-343-hippocampal-memory-is-updated-on-every-forward-pass) (amended) |
| `clone()` drops world model/memory/gen config; `load_model()` unchecked reads and generator special tokens not synced (§6) | Async metric snapshots differ from the live model; truncated configs load garbage | By inspection | [TD-363](../../guides/TECHNICAL_DEBT.md#td-363-encoderdecodermodel-clone-load-and-world-model-input-gaps) |
| `clone()` left its temp checkpoint's component files behind (§6) | A copy of the weights in the temp directory per clone (one per epoch with quality metrics) | **Fixed October 19, 2026** (was verified: 11 files from a one-layer clone) | [TD-544](../../guides/TECHNICAL_DEBT.md#td-544-encoderdecodermodelclone-leaves-its-temp-checkpoint-files-behind) |
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

---

## Addendum (October 20, 2026): gated paths in cached and GPU paths (TD-342)

- **CPU generation:** the KV-cached `model_fn` in `generate_response()`/`generate_response_with_strategy()` passes the world-model
  encoding and memory, so every strategy, not only beam search, uses them.
- **GPU training:** `gpu_forward()` encodes the decoded input with the world model on the device (`gpu_world_model_encoding()` →
  `LeJEPAEncoder::gpu_encode()`) and passes it with the memory to `gpu_decode()`.
- **GPU generation:** passes them to `gpu_decode()`/`gpu_decode_step()`, and now writes the response to the memory, as the CPU
  paths do (TD-185).
- **Memory writes:** `maybe_write_hippocampal_memory()` encodes on the device when the GPU is active.

Test: `gpu_optimizer_test.cpp`'s `GatedModelTrainsItsGatesOnTheDevice` (0 host groups; every gate opens).

---

## Addendum (October 21, 2026): one memory update per decode step (TD-343)

- **Commits:** every generation path (CPU and GPU, beam and cached) commits the memory's coverage/association update through a
  file-local `HippocampalStepCommitter`. A step is a token length, and beam hypotheses at the same step are averaged. The commit
  happens when the length advances or when generation ends, before the response is written.
- **Training forwards:** `forward()`/`gpu_forward()` never commit.
- **Switch:** `set_hippocampal_updates_enabled(false)` turns off commits and response writes; `ChatbotTrainer` sets it while training.

Tests: `BeamSearchCommitsOncePerDecodeStep`, `CachedDecodingCommitsOncePerToken`,
`TrainingForwardsAndDisabledUpdatesLeaveTheMemoryAlone`.

## Addendum (October 25, 2026): strategies and `num_beams` (TD-261)

`generate_response_with_strategy()` runs beam search only for `"beam"`, or for an unrecognized strategy with `num_beams > 1`, which
keeps that path off the KV cache. Explicit greedy/sampling/topk/nucleus ignore `num_beams`, and the generator's stored config keeps a
beam width only when beam search ran, so a later `generate_response()` isn't switched to beam. Both overloads default `num_beams` to
1. Test: `ExplicitStrategiesIgnoreNumBeams`.

## Addendum (October 26, 2026): `max_length` applied (TD-359)

`generate_response()` and `gpu_generate_response()` apply their `max_length` for the call (`ScopedMaxLength`), restoring the
generator's stored value afterwards; `<= 0` keeps it. Tests: `GenerateResponseHonoursMaxLength`,
`EncoderDecoderGPUTest.GpuGenerateResponseHonoursMaxLength` (oneAPI CPU device).

## Addendum (October 26, 2026): generation capped at `max_seq_length` (TD-347)

All four generation entry points cap the generated length (BOS included) at `max_seq_length` (`capped_generation_length()`). Test:
`GenerationNeverExceedsMaxSeqLength`.

## Addendum (October 26, 2026): `next_token_logits()` for generation callbacks (TD-506)

`next_token_logits(input_tokens, decoder_tokens)` returns the `1 × vocab` logits for the token after `decoder_tokens`. The decoder
input is that sequence as given, with a BOS put in front when it is empty or doesn't start with one. `forward()` instead feeds the
shifted targets (`[BOS] + targets[:-1]`), so its last row predicts the last given token. Use `next_token_logits()` in any
`TextGenerator` callback. Both methods share `run_decoder()`: the same truncated encoder input and the same world-model and hippocampal
reads. Only the last row is projected (TD-424). Test: `ChatbotAPIEchoTest.NextTokenLogitsIsTheLastRowOfTheAlignedForward`.

## Addendum (October 26, 2026): generation cap in new tokens (TD-516)

`TextGenerator`'s `max_length` now counts new tokens. `capped_generation_length()` therefore caps at `max_seq_length − 1` new tokens,
and the generated sequence with its BOS still fits in `max_seq_length` (TD-347).

## Addendum (October 26, 2026): checked checkpoint I/O (TD-423)

`save_model()`'s `.config` and `read_checkpoint_architecture()`; `load_model()` loads its files one after another, so a failure in a later file leaves earlier layers replaced use `adai::binio::Reader`/`Writer` (`src/BinaryIO.{hpp,cpp}`).
- **Loading.** A file that is short, empty or longer than this layer's shapes throws `std::runtime_error`. The whole file is read
  into temporaries first, so a failed load leaves the layer unchanged.
- **Saving.** A failed write or flush (a full disk) throws instead of leaving a short file.

Tests: `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`.

## Addendum (October 26, 2026): one behavior for CPU and GPU generation (TD-362)

All four generation entry points share `normalize_strategy()` (public, static), the beam decision ("beam", or an unrecognized
strategy with `num_beams > 1`), `sync_generation_config()` (stores the beam width only when beam search runs), `mask_unknown_ids()`
(IDs beyond the tokenizer's vocabulary, in every CPU and GPU callback) and `run_single_hypothesis()` (an unrecognized strategy runs
`TextGenerator::generate()`, on GPU too). The same request gives the same output on either path:
`EncoderDecoderGPUTest.GpuAndCpuGenerationAgree` and `PlainCachedPathMasksIdsTheTokenizerLacks`.
