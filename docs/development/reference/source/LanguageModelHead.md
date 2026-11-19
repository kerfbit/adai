# `LanguageModelHead` — Source File Reference

- **Files:** [`src/LanguageModelHead.hpp`](../../../../src/LanguageModelHead.hpp) (213 lines), [`src/LanguageModelHead.cpp`](../../../../src/LanguageModelHead.cpp) (286)
- **Namespace:** global
- **Built into:** `adai_models`; owned by `EncoderDecoderModel`; also driven, until their retirement in October 2026, by `PipelineInferenceEngine` and `IntegratedInferenceEngine`
- **Status tag:** `@adai-status: stable`; both `@adai-version: 2.0.0`; both `@adai-reviewed: 2026-11-19`
- **Tests:** [`tests/languagemodelhead_test.cpp`](../../../../tests/languagemodelhead_test.cpp) (41 tests; 42 in GPU builds)
- **Last traced against the code:** 2026-11-19

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class in a
> scratch program.

---

## 1. What this file is

The model's **output projection**: decoder hidden states `[seq × d_model]` to vocabulary logits
`[seq × vocab_size]` as `input · W_output + bias`. Softmax and sampling happen elsewhere (in
`TextGenerator`, `EncoderDecoderModel`'s loss, or the GPU cross-entropy kernels).

`W_output` is `[d_model × vocab_size]`, a **separate** matrix from the token embedding (no weight
tying), so the vocabulary costs `d_model × vocab_size` parameters twice. Initialization is
Xavier/Glorot, `N(0, √(2/(d_model + vocab_size)))`, through the unseeded `Matrix::randomize()`
(TD-368); the bias starts at zero.

---

## 2. API

| Member | Behaviour |
|---|---|
| `LanguageModelHead(d_model, vocab_size)` | Allocates and initializes; logs its shape and parameter count (64-bit) in one line at debug level (TD-425; it printed four lines to `std::cout`, with an `int` count that overflowed past about 2³¹) |
| `forward(input)` | Throws `std::invalid_argument` unless `input.cols == d_model` (TD-425). Caches `input`, then `input · W_output` plus the bias per row |
| `backward(grad)` | Throws `std::invalid_argument` unless `grad` is `[rows of the last forward() input, vocab_size]` (TD-425). Accumulates `W_output_grad += cached_inputᵀ · grad` and `bias_grad += Σ rows`; returns `grad · W_outputᵀ` |
| `update_weights()` | `optimizer->step()` (TD-297) or SGD; then zeroes the gradients |
| `zero_grad()`, `set_optimizer()`, `register_parameters()` | As in the other layers (double registration, TD-297) |
| `save(path)` / `load(path)` (and the `save_weights`/`load_weights` wrappers) | Two ints, then `W_output` and the bias as raw floats, with checked reads and writes (TD-423, see the addendum): a short, long or mismatched file throws and leaves the head unchanged. A successful `load()` refreshes the GPU mirror, when there is one (TD-425) |
| `get_W_output()`, `get_bias()`, `set_W_output()`, `set_bias()` | SafeTensors accessors (`ModelSerializer` saves and loads both, `lm_head.weight` transposed to HF layout). The setters throw `std::invalid_argument` unless the shape is `[d_model, vocab_size]` / `[1, vocab_size]` (nothing changes), and re-upload the GPU mirror when there is one, as `LayerNorm`'s do since TD-426 (TD-425) |
| GPU (`ADAI_ENABLE_GPU`) | `gpu_upload_weights()`, `gpu_download_grads()` (adds to the host gradients), `gpu_zero_grads()`, `gpu_forward()` (uploads on first use; no shape check), `gpu_backward()` (throws `std::logic_error` before any `gpu_forward()`, where it used to dereference a null `gpu_`, and `std::invalid_argument` on a wrong shape; TD-425) |

---

## 3. Problems (verified)

Both were fixed on October 26, 2026 (see the addenda); the descriptions are kept for the record.

### A truncated `.lm_head` file produced an all-`NaN` model (fixed, TD-423)

`EncoderDecoderModel::save_model()` writes this as the checkpoint's `.lm_head` sidecar, and
`load_model()` reads it with `load_weights()`. A file cut in half **loaded without an error**: 751 of
the 1,600 weights were `NaN`, and **all 100 logits** of a forward pass were `NaN`. A partially written
checkpoint (a full disk, or a crash during save) therefore yields a server that generates garbage
rather than failing to load. That's the same defect as TD-298 (`CrossAttention::load()`), here in a
file every checkpoint has.

### Generation paths projected every position at every step (fixed, TD-424)

The non-cached generation callbacks (`EncoderDecoderModel`'s beam `beam_model_fn`, the uncached
`model_fn`, `/chat/batch`'s `model_->forward()`) run `lm_head->forward(decoder_out)` on the **whole**
decoder output, despite the comment "(last position of output)". `TextGenerator` only uses the last row.
Measured at d_model 512, vocab 8,000, 100 positions: **147 ms** for all positions against **4.9 ms** for
the last one, about 30× the work, per step, per beam hypothesis. The KV-cached path projects only the new
tokens, and the pipeline and integrated engines take the last row first, so they're unaffected.

---

## 4. Tests

`languagemodelhead_test.cpp` (41 tests; 42 in GPU builds) covers:
- construction and initialization ranges;
- forward shapes and bias broadcast;
- backward gradients and accumulation;
- updates with and without an optimizer;
- save/load round trips and dimension mismatch; truncated, over-long and unwritable files (TD-423);
- shape checks in `forward()`, `backward()` and the setters, which leave the head unchanged, and nothing
  written to stdout by the constructor (TD-425);
- in GPU builds, `gpu_backward()` before `gpu_forward()` throwing, and the setters refreshing the device
  copy (`LanguageModelHeadGPUTest`, run on the oneAPI CPU device).

---

## 5. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `load()` pre-fills `NaN` and doesn't check reads; a truncated `.lm_head` loads silently (§3) | A damaged checkpoint serves all-`NaN` logits instead of failing to load | **Verified** | [TD-423](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-423-a-truncated-lm_head-file-loads-as-all-nan-logits) |
| Uncached generation paths project all positions every step (§3) | About 30× the LM-head work per step (and per beam) at 100 tokens | **Verified** (timing) | [TD-424](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-424-uncached-generation-projects-every-position-through-the-lm-head) |
| `std::cout` in the constructor; no input-shape check; setters don't check shapes or refresh the GPU mirror; `gpu_backward()` null dereference; unused `get_probabilities()`; `int` parameter-count overflow (§2) | — | **Fixed November 19, 2026** | [TD-425](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-425-languagemodelhead-hygiene) |
| No weight tying with the token embedding (§1) | Twice the vocabulary parameters; a design choice | By inspection | — (design choice) |
| Shared-optimizer stepping; unseeded init | Already tracked | — | TD-297, TD-368 |

---

## 6. Merge notes

This file replaces `docs/development/api/transformer/language-model-head.md` (964 lines). Kept: the
formulation, the gradients and the persistence layout. Dropped: generic material on cross-entropy,
adaptive softmax and generation-loop pseudocode (`sample_from_distribution`, `sample_with_temperature`,
none of which exist here), and its unsourced complexity and performance tables. Its "weight tying
(optional enhancement)" section is reflected in §1 as a not-implemented design option.

## Addendum (October 26, 2026): beam callbacks project the last row (TD-424)

`EncoderDecoderModel`'s beam callbacks pass only the decoder's last row to `forward()`. The head is row-wise, so this is exact. A
60-token beam run (width 4, vocab 4,000) went from 6.9 s to 4.2 s in the debug build. Since TD-506 (same day) `ChatbotAPI`'s CPU
callbacks use `EncoderDecoderModel::next_token_logits()`, which also projects only the last row.

## Addendum (October 26, 2026): checked checkpoint I/O (TD-423)

`save()`/`load()` use `adai::binio::Reader`/`Writer` (`src/BinaryIO.{hpp,cpp}`).
- **Loading.** A file that is short, empty or longer than this layer's shapes throws `std::runtime_error`. The whole file is read
  into temporaries first, so a failed load leaves the layer unchanged.
- **Saving.** A failed write or flush (a full disk) throws instead of leaving a short file.

Tests: `LanguageModelHeadPersistenceTest.*`, `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad` (all 22 sidecars of a checkpoint, one byte short and one byte long: 43 of 44 cases loaded on the old code).

## Addendum (November 19, 2026): hygiene (TD-425)

The constructor logs through `adai::Logger` (debug) with a 64-bit parameter count; `forward()`, `backward()` and the setters check
shapes; the setters and `load()` refresh the GPU mirror; `gpu_backward()` throws instead of dereferencing a null `gpu_`.
`get_probabilities()` (a static softmax of one logit vector, with no production caller) was removed, with its five tests: generation,
the loss and RLHF all use `Activation::softmax_row()`, whose own tests cover the same properties, and two integration tests here now
call it directly. Hence the 2.0.0 versions. §2 describes the result.
