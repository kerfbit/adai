# `CrossAttention` — Source File Reference

- **Files:** [`src/CrossAttention.hpp`](../../../../src/CrossAttention.hpp), [`src/CrossAttention.cpp`](../../../../src/CrossAttention.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_attention` (linked by `adai_transformer` → `adai_models`), so every binary that builds a model
- **Status tag:** `@adai-status: beta` (capped by TD-050), `@adai-version: 0.13.1` (`.hpp`) / `0.13.0` (`.cpp`), `@adai-reviewed: 2026-09-15`
- **Tests:** [`tests/crossattention_test.cpp`](../../../../tests/crossattention_test.cpp) (51 tests), plus indirect coverage in the decoder-block, decoder, encoder-decoder and inference-optimization suites
- **Origin:** original encoder-decoder design; TD-059 (real per-head split), TD-038 (LoRA), TD-050 (GPU incremental cache), TD-174 (`forward_with_scores`), TD-180 (`get_last_attention_weights`)
- **Last traced against the code:** 2026-10-10

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class
> (scratch program linked against `libadai_attention`/`libadai_layers`/`libadai_core`), including a
> finite-difference gradient check of `backward()`.

---

## 1. What this file is

The decoder's **encoder-decoder attention**: queries come from the decoder's hidden states, keys and
values from another sequence. That's normally the encoder output, but `DecoderBlock` also uses two
extra instances to attend to a world model's output and to hippocampal memory slots (TD-180).

```text
query_input [tgt, d]   kv_input [src, d]
      │ × W_q               │ × W_k, × W_v       (+ optional LoRA ΔW on each, TD-038)
      ▼                     ▼
      Q                  K, V
      └──── per head h (d_k = d/num_heads columns each, TD-059) ────┐
            scores_h = Q_h K_hᵀ / √d_k  (+ score_bias, TD-174)      │
            masked where mask == 0 → −1e9                           │
            weights_h = softmax_rows(scores_h)                      │
            out_h = weights_h V_h  → scattered into columns of concat ┘
concat × W_o (+ LoRA_o)  →  output [tgt, d]
```

### Why it matters

Cross-attention is the **only** path by which the encoder's reading of the user's message reaches
the decoder. If it's wrong, the model generates fluent text unrelated to the input. Its two
backward outputs are also how the **encoder** is trained at all: `grad_kv_input` from every decoder
block is summed by `LLMDecoder::backward()` and propagated into `LLMEncoder::backward()` by
`EncoderDecoderModel::backward()`.

---

## 2. Where it's used

| Caller | Instance | Method(s) |
|---|---|---|
| `DecoderBlock` (`src/DecoderBlock.cpp`) | `cross_attention` (always) | `forward()` (training/plain inference), `forward_with_cache()` (KV-cached generation), `backward()`, `update_weights()`, `zero_grad()`, `get_gradient_norm()`, `set_optimizer()`, `save()`/`load()` (`<block>.cross_attn`), and on GPU builds `gpu_forward()`, `gpu_forward_with_cache()`, `gpu_backward()`, `gpu_upload_weights()`/`gpu_download_grads()`/`gpu_zero_grads()` |
| `DecoderBlock` | `world_model_cross_attention` (TD-180, when a world model is attached) | CPU `forward()`/`backward()` only |
| `DecoderBlock` | `hippocampal_cross_attention` (TD-180, when hippocampal memory is enabled) | `forward_with_scores()` with a repetition-penalty bias, then `get_last_attention_weights()` to update slot coverage; `backward()` |
| `ModelSerializer` | the block's `cross_attention` | `get_W*()`/`set_W*()` as SafeTensors `…layer.1.EncDecAttention.{q,k,v,o}.weight` (transposed on load) |
| `EncoderDecoderModel` | all, via `LLMDecoder` | LoRA cascade (`enable_lora`/`register_lora_parameters`/`merge_lora`); GPU weight sync (`gpu_init_training()`, `gpu_sync_weights()`) |

No production caller passes a cross-attention **mask**: `LLMDecoder` always passes `nullptr`, and
the GPU cached path hardcodes `nullptr`. In **decoder-only mode** (no encoder output), `LLMDecoder`
passes a dummy `Matrix(1, d_model)` of zeros. Despite the comment there saying "no
cross-attention", the layer still runs. With `V = 0` its output is exactly zero and its gradients
are zero, so it's wasted compute rather than a correctness problem.

---

## 3. Data model

| Member | Shape | Purpose |
|---|---|---|
| `d_model`, `num_heads`, `d_k` | — | `d_k = d_model / num_heads`, computed in the initializer list |
| `W_q`, `W_k`, `W_v`, `W_o` | `[d, d]` | Projections, initialized `N(0, √(2/d))` (`Matrix::randomize` is Gaussian with that std) |
| `W_*_grad` | `[d, d]` | **Accumulated** gradients; `backward()` adds, `zero_grad()`/`update_weights()` clear |
| `cached_query_input`, `cached_kv_input`, `cached_Q/K/V`, `cached_attention_output` | — | State from the last forward call, consumed by `backward()` |
| `cached_head_weights_` | `num_heads × [tgt, src]` | Real per-head softmax weights; `backward()` differentiates through these |
| `cached_attention_weights` | `[tgt, src]` | Mean over heads, for callers only (`get_last_attention_weights()`) |
| `optimizer` | raw `Optimizer*` | Non-owning; `nullptr` means plain SGD with `learning_rate` |
| `lora_q_/k_/v_/o_` | `unique_ptr<LoRAAdapter>` | Optional adapters (TD-038) |
| `gpu_` (GPU builds) | `unique_ptr<GPUState>` | Device copies of weights/grads plus backward caches, created lazily |

Parameter count is `4·d²` (1,048,576 for `d_model = 512`, about 4 MB of float32, plus the same again
for gradients and, with Adam/AdamW, twice that for moments).

---

## 4. API

### Construction

`CrossAttention(d_model, num_heads)` throws `std::invalid_argument` if `d_model % num_heads != 0`.
The check runs in the constructor **body**, after the initializer list has already computed
`d_model / num_heads`.

> **Verified:** `CrossAttention(8, 0)` kills the process with **SIGFPE** (integer division by zero)
> instead of throwing. A negative `num_heads` that divides `d_model` (e.g. `(9, −3)`) passes the check
> and yields a negative `d_k`. Callers derive `num_heads` from config or MNS. `mns_server` rejects
> 0 at register time and `ConfigLoader::validate()` rejects values outside 1–64, but
> `chatbot_api_server` never runs `validate()` at startup (TD-241). So a local `NUM_HEADS=0` with no
> MNS crashes the process instead of giving a clear error.

### Forward family

All three CPU forwards validate shapes (`query.cols == d_model`, `kv.cols == d_model`, mask and bias
`[tgt, src]`), apply LoRA after each base projection, run the per-head loop, cache everything
`backward()` needs, and return `[tgt, d_model]`.

| Method | Differences |
|---|---|
| `forward(q, kv, mask = nullptr)` | The baseline |
| `forward_with_scores(q, kv, score_bias, mask = nullptr)` | Adds `score_bias[tgt, src]` to every head's scaled scores before masking; masking wins over bias. An all-zero bias reproduces `forward()` exactly. The bias is treated as a constant (no gradient w.r.t. it), which is correct for the hippocampal coverage penalty |
| `forward_with_cache(q, kv, mask, kv_cache, use_cache = true)` | With no cache or `use_cache=false`, it's `forward()`. Otherwise it projects K/V from `kv` **only when `kv_cache->is_empty()`**, appends them, and on every later call reads them back and **ignores `kv` entirely** |

**Masking.** A mask entry of exactly `0.0f` means "don't attend" (the score becomes `−1e9`), and any
other value means attend. This matches the GPU kernel (`matrix_masked_fill_gpu`) and
`BatchProcessor::create_padding_mask()`. A query row whose mask is entirely zero gets a **uniform**
distribution over the masked keys (verified: `0.2` each over 5 keys) rather than zero output, the
standard behaviour of the `−1e9` trick.

**KV-cache contract (verified).** Reusing a non-empty cache with a *different* encoder output
silently returns attention over the **old** encoder's K/V. A 7-row encoder output after a 5-row one
gave a result identical to the first call and 7.9 (L1) away from the uncached answer. A mask sized
for the new encoder is then rejected with a confusing `src_len=5` error. Production is safe:
`EncoderDecoderModel` builds a fresh `DecoderKVCache` per generation. But the method neither checks
that `kv` still matches the cache nor documents that the caller must `clear()` it. Enabling or
changing LoRA after the cache is populated likewise leaves stale K/V.

### Backward

`backward(grad_output, grad_query_input&, grad_kv_input&)` reverses the forward: through `W_o` (and
LoRA_o), then **per head** through `weights_h V_h`, the row-wise softmax Jacobian (an inline copy of
`Activation::softmax_derivative`, TD-216), and the `1/√d_k` scale, then through `W_q`, `W_k`, `W_v` and
the LoRA adapters. It returns two input gradients (`grad_kv_input` sums the K and V paths) and
**adds** into `W_*_grad`.

> **Verified correct.** A central finite-difference check of both input gradients agreed to within
> float32 noise for plain `forward()`, a partial mask, `forward_with_scores()` with a random bias,
> and a layer with trained (non-zero) LoRA adapters.

It sizes from `cached_Q`/`cached_K`, so it's shape-safe after `forward_with_cache()`. Calling
`backward()` with no prior forward indexes an empty `cached_head_weights_` (undefined behaviour), but
no caller does that.

### Optimization

| Method | Behaviour |
|---|---|
| `set_optimizer(opt)` | Stores the pointer and calls `register_parameters()`, which adds the four `(W, W_grad)` groups |
| `update_weights()` | With an optimizer, `optimizer->step()`; otherwise `W −= learning_rate · grad`. Then `zero_grad()` |
| `zero_grad()` | Zeroes the four grads and every active LoRA adapter's grads |
| `get_gradient_norm()` | L2 norm of the four base grads (**excludes LoRA grads**) |

> **Shared-optimizer trap (verified).** `Optimizer::step()` steps **every** group ever registered
> with it, not only this layer's. `DecoderBlock::register_parameters_with_optimizer()` registers
> every sub-layer with one shared optimizer, and `DecoderBlock::update_weights()` then calls each
> sub-layer's `update_weights()` in turn. With two `CrossAttention`s on one SGD optimizer, the first
> layer's `update_weights()` moved the second layer's `W_o` by a full step, and the second's own call
> moved it again: **two steps**. In a full decoder block it's 6 to 10 `step()` calls per update,
> each also advancing Adam's step counter. `LeJEPAEncoder` hit and fixed exactly this
> (TD-178 in TECHNICAL_DEBT_RESOLVED.md) by calling `step()` once at the top level.
>
> **The shipped trainers avoid it.** `ChatbotTrainer` and `RLHFTrainer` call `optimizer->step()`
> directly, once. But `EncoderDecoderModel::train_step()`/`update_weights()` → `LLMDecoder` →
> `DecoderBlock::update_weights()` → this method is a public path that over-steps whenever an
> optimizer is registered. Calling `set_optimizer()` twice with the same optimizer also registers the
> groups twice (verified: 8 groups), so each `step()` applies a double update.

### Persistence

`save(path)` writes `d_model`, `num_heads`, `learning_rate`, then `W_q`, `W_k`, `W_v`, `W_o` as raw
float32. Legacy `DecoderBlock::save()` uses it for `<block>.cross_attn`; the main checkpoint format is
SafeTensors via `ModelSerializer`. `load(path)` reads the header, throws on a dimension mismatch,
then reads the four matrices.

> **Verified:**
> - **Truncated files load silently as NaN.** Each value is pre-set to `NAN` and the stream state is
>   never checked, so a file cut off after `W_k` loaded without error with **all 128** `W_v`/`W_o`
>   entries NaN. An empty file is caught only because its "dimensions" read as 0.
> - **A rejected file still changes the layer.** `learning_rate` is read into the member *before*
>   the dimension check: a mismatched file threw, but left `learning_rate` at the file's 0.5.
>
> `save()` doesn't check writes either. Neither `save()` nor SafeTensors stores LoRA adapters, so
> unmerged adapters are lost on save (see §6).

### Accessors

- `get_d_model()`, `get_num_heads()`.
- `get_last_attention_weights()`: the head-mean `[tgt, src]` weights from the last **CPU** forward;
  GPU forwards don't update it.
- `get_W*()`/`set_W*()`: `set_W*` doesn't check shape and doesn't touch `gpu_`.
- `get_lora_*()`, `has_lora()`.

### LoRA (TD-038)

- `enable_lora(cfg)` creates square adapters on the projections selected by
  `apply_to_query/key/value/output`. B starts at zero, so output is unchanged until training.
  Re-calling it discards trained adapters.
- `register_lora_parameters(opt)` registers only the adapters' A/B matrices, so the base weights
  stay frozen.
- `merge_lora()` folds each ΔW into its base weight and drops the adapter.

---

## 5. GPU path (`ADAI_ENABLE_GPU`)

`gpu_forward()`/`gpu_backward()` mirror the CPU math head by head using `GPUMatrix` primitives, with
slicing done by row-wise device-to-device copies (`d_k`-wide, so `rows × heads` small copies per
projection). `gpu_forward_with_cache()` uses a `GPUKVCache` the same way the CPU version does
(project encoder K/V once, then reuse), and is inference-only.

Differences from the CPU path, by inspection (no GPU in the environment this was traced in):

- **LoRA is ignored.** The header's TD-038 note says so; there's no guard or warning.
- **No shape validation.** There's no check on `query`/`kv` widths or mask dimensions, and
  `masked_fill_inplace` only uses the mask's element count.
- **Weights are snapshots.** `gpu_upload_weights()` copies the CPU weights once (lazily on first
  `gpu_forward`). After any CPU-side change (`update_weights()`, `load()`, `set_W*()`, `merge_lora()`),
  the GPU copy is stale until the owner calls `EncoderDecoderModel::gpu_sync_weights()`.
  `gpu_download_grads()` **adds** device grads into the CPU grads, so the device grads must be
  zeroed between steps (`gpu_zero_grads()`).
- **No CPU-side state is updated.** `get_last_attention_weights()` isn't refreshed, and
  `gpu_forward_with_cache()` doesn't fill `GPUState`'s backward caches.
- `DecoderBlock`'s GPU path runs only the main `cross_attention`; the world-model and hippocampal
  instances are CPU-only (a `DecoderBlock` concern).

---

## 6. Tests

`crossattention_test.cpp` has 51 tests:

| Group | Tests |
|---|---|
| Constructor | 3 (including non-divisible heads) |
| Forward | 5 |
| Masking | 2 |
| Backward | 5 |
| Update | 2 |
| Optimizer | 12 (single-layer optimizer use) |
| SaveLoad | 3 |
| EdgeCase | 4 |
| Architecture | 2 (including one finite-difference check on the query input) |
| Integration | 3 |
| LoRA | 4 |
| ScoreBias | 6 (TD-174) |

The cached path is covered in `inference_optimization_test.cpp` and the encoder-decoder suites. The
older [CROSSATTENTION_TEST_COVERAGE.md](../../testing/coverage/CROSSATTENTION_TEST_COVERAGE.md)
predates TD-059/TD-038/TD-174.

**Not covered:**
- `num_heads ≤ 0`;
- truncated or corrupt `load()` input;
- cache reuse with a different encoder output;
- a shared optimizer across layers;
- a gradient check on `grad_kv_input` or with LoRA/bias;
- anything GPU-side (not built in CI).

---

## 7. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `update_weights()` calls a possibly **shared** optimizer's `step()`, which steps every registered layer; double `set_optimizer()` double-registers (§4) | Public `EncoderDecoderModel::train_step()`/`update_weights()` over-step every parameter when an optimizer is registered; shipped trainers unaffected | **Verified** | [TD-297](../../guides/TECHNICAL_DEBT.md#td-297-layer-update_weights-steps-a-shared-optimizer-once-per-layer) |
| `load()` doesn't check reads; truncated file → NaN weights, no error; `learning_rate` clobbered before the mismatch check; `save()` unchecked (§4) | A damaged legacy `.cross_attn` file silently produces a NaN model | **Verified** | [TD-298](../../guides/TECHNICAL_DEBT.md#td-298-crossattentionload-loads-truncated-files-as-nan-weights) |
| `num_heads == 0` → SIGFPE before the validation runs; negative heads accepted (§4) | A local `NUM_HEADS=0` crashes the process where `validate()` isn't run (TD-241); MNS rejects 0 | **Verified** | [TD-299](../../guides/TECHNICAL_DEBT.md#td-299-crossattention-crashes-with-sigfpe-when-num_heads-is-zero) |
| `forward_with_cache()` silently ignores a new `kv` once the cache is filled; no staleness check (§4) | Wrong attention if a caller reuses a cache across inputs or after enabling LoRA; production is safe today | **Verified** | [TD-300](../../guides/TECHNICAL_DEBT.md#td-300-crossattentionforward_with_cache-silently-reuses-stale-encoder-kv) |
| GPU path ignores LoRA, skips validation, holds weight snapshots, doesn't refresh `get_last_attention_weights()` (§5) | Silent CPU/GPU divergence when LoRA is used or weights change without a sync | By inspection | [TD-301](../../guides/TECHNICAL_DEBT.md#td-301-crossattention-gpu-path-diverges-from-the-cpu-path) |
| LoRA adapters aren't persisted by `save()` or SafeTensors; `get_gradient_norm()` excludes LoRA grads (§4) | Unmerged LoRA training lost on save; gradient clipping ignores adapters. LoRA has no production caller yet | By inspection | [TD-302](../../guides/TECHNICAL_DEBT.md#td-302-lora-adapters-arent-persisted-or-counted-in-gradient-norms) |
| Decoder-only mode still runs cross-attention over a zero vector (§2) | Wasted compute; comment is wrong | By inspection | [TD-303](../../guides/TECHNICAL_DEBT.md#td-303-decoder-only-mode-still-runs-cross-attention-over-a-dummy-input) |
| Inline softmax-backward copy | Already tracked | — | TD-216 |

---

## 8. Merge notes

This file replaces `docs/development/api/attention/cross-attention.md` (v1.1, January 24, 2026, about
1,900 lines). Where they disagree, this file wins. Specifically:

- **Per-head math (TD-059).** The old doc described cross-attention as if it were multi-head while
  the code at the time was single-head over the full `d_model`. Since September 12, 2026 the code
  genuinely splits heads, and checkpoints trained before then need retraining.
- **Missing features.** The old doc had no `forward_with_scores()` (TD-174),
  `get_last_attention_weights()` (TD-180), LoRA (TD-038), GPU path or GPU cache (TD-050), or the
  world-model and hippocampal instances.
- **"Set an optimizer and call `update_weights()`".** The old doc recommended this pattern
  everywhere. It's correct only when the optimizer belongs to that one layer; see the
  shared-optimizer trap in §4.

**Kept from the old doc:**
- the backward derivation, now §4 (applied per head);
- the "two gradients, and `grad_kv_input` must reach the encoder" note (§1);
- parameter count and memory (§3);
- the mask conventions: `[tgt, src]` shape, no causal mask on cross-attention (§4).

**Dropped:**
- estimated CPU benchmarks (never measured);
- "use cases" for translation, captioning and similar;
- future-enhancement wish lists (MQA/GQA/Flash attention);
- a duplicated "Optimizer Usage Patterns"/"Recent Updates" section that appeared twice.
