# `LanguageModelHead` — Source File Reference

- **Files:** [`src/LanguageModelHead.hpp`](../../../../src/LanguageModelHead.hpp), [`src/LanguageModelHead.cpp`](../../../../src/LanguageModelHead.cpp)
- **Namespace:** global
- **Built into:** `adai_models`; owned by `EncoderDecoderModel`; also driven by `PipelineInferenceEngine` and `IntegratedInferenceEngine`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10` (both files)
- **Tests:** [`tests/languagemodelhead_test.cpp`](../../../../tests/languagemodelhead_test.cpp) (42 tests)
- **Last traced against the code:** 2026-10-13

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
| `LanguageModelHead(d_model, vocab_size)` | Allocates and initializes; **prints four lines to `std::cout`**. Its `int` parameter count overflows past about 2³¹ |
| `forward(input)` | Caches `input`, then `input · W_output` plus the bias per row. **No check** that `input.cols == d_model` (the matrix multiply throws on a mismatch) |
| `backward(grad)` | Accumulates `W_output_grad += cached_inputᵀ · grad` and `bias_grad += Σ rows`; returns `grad · W_outputᵀ`. Correct |
| `update_weights()` | `optimizer->step()` (TD-297) or SGD; then zeroes the gradients |
| `zero_grad()`, `set_optimizer()`, `register_parameters()` | As in the other layers (double registration, TD-297) |
| `get_probabilities(logits)` (static) | Softmax of a vector. **No production caller** |
| `save(path)` / `load(path)` (and the `save_weights`/`load_weights` wrappers) | Two ints, then `W_output` and the bias as raw floats. `load()` checks the dimensions, then **pre-fills each value with `NaN` and never checks the stream** |
| `get_W_output()`, `get_bias()`, `set_W_output()`, `set_bias()` | SafeTensors accessors (`ModelSerializer` saves and loads both, `lm_head.weight` transposed to HF layout). The setters don't check shapes or refresh the GPU mirror (as in TD-370) |
| GPU (`ADAI_ENABLE_GPU`) | `gpu_upload_weights()`, `gpu_download_grads()` (adds to the host gradients), `gpu_zero_grads()`, `gpu_forward()` (uploads on first use; no shape check), `gpu_backward()` (**dereferences `gpu_` unchecked**, so calling it before `gpu_forward()` crashes) |

---

## 3. Problems (verified)

### A truncated `.lm_head` file produces an all-`NaN` model

`EncoderDecoderModel::save_model()` writes this as the checkpoint's `.lm_head` sidecar, and
`load_model()` reads it with `load_weights()`. A file cut in half **loaded without an error**: 751 of
the 1,600 weights were `NaN`, and **all 100 logits** of a forward pass were `NaN`. A partially written
checkpoint (a full disk, or a crash during save) therefore yields a server that generates garbage
rather than failing to load. That's the same defect as TD-298 (`CrossAttention::load()`), here in a
file every checkpoint has.

### Generation paths project every position at every step

The non-cached generation callbacks (`EncoderDecoderModel`'s beam `beam_model_fn`, the uncached
`model_fn`, `/chat/batch`'s `model_->forward()`) run `lm_head->forward(decoder_out)` on the **whole**
decoder output, despite the comment "(last position of output)". `TextGenerator` only uses the last row.
Measured at d_model 512, vocab 8,000, 100 positions: **147 ms** for all positions against **4.9 ms** for
the last one, about 30× the work, per step, per beam hypothesis. The KV-cached path projects only the new
tokens, and the pipeline and integrated engines take the last row first, so they're unaffected.

---

## 4. Tests

`languagemodelhead_test.cpp` (42 tests) covers:
- construction and initialization ranges;
- forward shapes and bias broadcast;
- backward gradients and accumulation;
- updates with and without an optimizer;
- `get_probabilities()` (sums to 1, stability);
- save/load round trips and dimension mismatch.

**Not covered:** truncated files.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `load()` pre-fills `NaN` and doesn't check reads; a truncated `.lm_head` loads silently (§3) | A damaged checkpoint serves all-`NaN` logits instead of failing to load | **Verified** | [TD-423](../../guides/TECHNICAL_DEBT.md#td-423-a-truncated-lm_head-file-loads-as-all-nan-logits) |
| Uncached generation paths project all positions every step (§3) | About 30× the LM-head work per step (and per beam) at 100 tokens | **Verified** (timing) | [TD-424](../../guides/TECHNICAL_DEBT.md#td-424-uncached-generation-projects-every-position-through-the-lm-head) |
| `std::cout` in the constructor; no input-shape check; setters don't check shapes or refresh the GPU mirror; `gpu_backward()` null dereference; unused `get_probabilities()`; `int` parameter-count overflow (§2) | Console noise from library code; latent misuse failures | By inspection | [TD-425](../../guides/TECHNICAL_DEBT.md#td-425-languagemodelhead-hygiene) |
| No weight tying with the token embedding (§1) | Twice the vocabulary parameters; a design choice | By inspection | — (design choice) |
| Shared-optimizer stepping; unseeded init | Already tracked | — | TD-297, TD-368 |

---

## 6. Merge notes

This file replaces `docs/development/api/transformer/language-model-head.md` (964 lines). Kept: the
formulation, the gradients and the persistence layout. Dropped: generic material on cross-entropy,
adaptive softmax and generation-loop pseudocode (`sample_from_distribution`, `sample_with_temperature`,
none of which exist here), and its unsourced complexity and performance tables. Its "weight tying
(optional enhancement)" section is reflected in §1 as a not-implemented design option.
