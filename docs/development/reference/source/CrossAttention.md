# `CrossAttention` — Source File Reference

- **Files:** [`src/CrossAttention.hpp`](../../../../src/CrossAttention.hpp) (473 lines), [`src/CrossAttention.cpp`](../../../../src/CrossAttention.cpp) (1,172 lines)
- **Namespace:** global (no namespace)
- **Built into:** `adai_attention` (linked by `adai_transformer` → `adai_models`), so every binary that builds a model
- **Status tag:** `@adai-status: beta` (capped by TD-050), `@adai-version: 0.17.0` (`.hpp`) / `0.17.0` (`.cpp`), `@adai-reviewed: 2026-11-13`
- **Tests:** [`tests/crossattention_test.cpp`](../../../../tests/crossattention_test.cpp) (59 tests; 64 in a GPU build), plus indirect coverage in the decoder-block, decoder, encoder-decoder and inference-optimization suites
- **Origin:** original encoder-decoder design; TD-059 (real per-head split), TD-038 (LoRA), TD-050 (GPU incremental cache), TD-174 (`forward_with_scores`), TD-180 (`get_last_attention_weights`)
- **Last traced against the code:** 2026-10-10, with the addenda below; construction, the forward family, optimization, accessors (§4) and the GPU path (§5) re-traced 2026-11-13

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

`CrossAttention(d_model, num_heads)` initializes `d_k` from `checked_head_dim(d_model, num_heads)`, a
file-local helper that throws `std::invalid_argument` when either value is below 1 or
`d_model % num_heads != 0`, before dividing.

> **Fixed November 13, 2026 (TD-299).** The divisibility check used to run in the constructor body,
> after the initializer list had computed `d_model / num_heads`, so `CrossAttention(8, 0)` killed the
> process with SIGFPE (verified, and verified again by the new test on the old code) and a negative
> `num_heads` that divides `d_model` (e.g. `(9, −3)`) was accepted with a negative `d_k`.
> `MultiHeadAttention` had the same defect (TD-467) and has the same helper. (`chatbot_api_server` also
> runs `ConfigLoader::validate()` at startup since TD-241.)

### Forward family

All three CPU forwards validate shapes (`query.cols == d_model`, `kv.cols == d_model`, mask and bias
`[tgt, src]`), apply LoRA after each base projection, run the per-head loop, cache everything
`backward()` needs, and return `[tgt, d_model]`.

| Method | Differences |
|---|---|
| `forward(q, kv, mask = nullptr)` | The baseline |
| `forward_with_scores(q, kv, score_bias, mask = nullptr)` | Adds `score_bias[tgt, src]` to every head's scaled scores before masking; masking wins over bias. An all-zero bias reproduces `forward()` exactly. The bias is treated as a constant (no gradient w.r.t. it), which is correct for the hippocampal coverage penalty |
| `forward_with_cache(q, kv, mask, kv_cache, use_cache = true)` | With no cache or `use_cache=false`, it's `forward()`. Otherwise it projects K/V from `kv` **only when `kv_cache->is_empty()`**, appends them and sets the cache's `source_tag`; every later call checks that `kv` is still that encoder output (below) and reads the K/V back |

**Masking.** A mask entry of exactly `0.0f` means "don't attend" (the score becomes `−1e9`), and any
other value means attend. This matches the GPU kernel (`matrix_masked_fill_gpu`) and
`BatchProcessor::create_padding_mask()`. A query row whose mask is entirely zero gets a **uniform**
distribution over the masked keys (verified: `0.2` each over 5 keys) rather than zero output, the
standard behaviour of the `−1e9` trick.

**KV-cache contract (TD-300, fixed November 13, 2026).** A filled cache answers only for the encoder
output, layer and weights it was filled with. When it fills a cache, `forward_with_cache()` stores
`cache_source_tag()` in `KVCache::source_tag`: an FNV-1a hash of a per-instance id, the layer's
weights version, the input's shape and 64 sampled values (the first and last rows included), so the
check costs the same for any input size. Every later call recomputes it and throws
`std::invalid_argument` when the row count differs (naming both shapes) or the tag does, before
anything changes. The weights version (`weights_version_`) changes with every weight change through
the layer: the setters, `load()`, `update_weights()`, `enable_lora()`, `merge_lora()` (all through the
private `weights_changed()`) and `gpu_download_weights()`. So a cache filled before LoRA was enabled,
or by another `CrossAttention`, is refused too. Callers `clear()` the cache for a new encoder output;
`DecoderKVCache::clear_self_attention()` keeps the encoder K/V for the same input. Before the fix,
reusing a filled cache with a different encoder output silently returned attention over the **old**
one (verified: a 7-row encoder output after a 5-row one gave the first call's answer, 7.9 L1 away from
the uncached one, and a mask sized for the new encoder was rejected with a confusing `src_len=5`
error). Limits: a different input that matches at all 64 sampled positions, and a change to the
adapters' own weights (`get_lora_*()->update()`) or an optimizer step taken elsewhere, aren't seen.
Production builds a fresh `DecoderKVCache` per generation and passes the same encoder output every
step.

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
| `update_weights()` | With an optimizer, `optimizer->step()`; otherwise `W −= learning_rate · grad`. Then `zero_grad()`, and the GPU mirror is refreshed (TD-301, §5) |
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
  GPU forwards don't update it (`gpu_last_mean_attention_weights()` gives theirs, §5).
- `get_W*()`/`set_W*()`: since TD-301 the setters are defined in the `.cpp`, need a `d_model × d_model`
  matrix (else `std::invalid_argument` and no change) and refresh the GPU mirror (§5).
- `get_lora_*()`, `has_lora()`.

### LoRA (TD-038)

- `enable_lora(cfg)` creates square adapters on the projections selected by
  `apply_to_query/key/value/output`. B starts at zero, so output is unchanged until training.
  Re-calling it discards trained adapters.
- `register_lora_parameters(opt)` registers only the adapters' A/B matrices, so the base weights
  stay frozen.
- `merge_lora()` folds each ΔW into its base weight, drops the adapter and refreshes the GPU mirror.
- Adapters apply on the CPU path only; the GPU forwards refuse a layer that has any (§5).

---

## 5. GPU path (`ADAI_ENABLE_GPU`)

`gpu_forward()`/`gpu_backward()` mirror the CPU math head by head using `GPUMatrix` primitives, with
slicing done by row-wise device-to-device copies (`d_k`-wide, so `rows × heads` small copies per
projection). `gpu_forward_with_cache()` uses a `GPUKVCache` the same way the CPU version does
(project encoder K/V once, then reuse), and is inference-only.

How it relates to the CPU path (TD-301, fixed November 13, 2026; verified on the oneAPI CPU device):

- **LoRA.** `gpu_forward()`/`gpu_forward_with_cache()` throw `std::logic_error` while any adapter is
  attached; they used to run the base weights silently. `merge_lora()` first.
- **Shapes.** `gpu_forward()` needs `query` and `kv` `d_model` wide and `mask`/`score_bias`
  `[tgt, src]`; `gpu_forward_with_cache()` needs `query` `d_model` wide, `kv` too on the first
  (cache-filling) call only, and the mask `[new, src]`. On later calls `kv` must have the cache's row
  count and the layer's weights must not have changed since the cache was filled
  (`GPUKVCache::source_tag()`, TD-300); the content isn't sampled, as that would copy from the device
  every step. Otherwise `std::invalid_argument`, before
  anything changes; the mask used to be checked only by its element count (`masked_fill_inplace()`).
  `gpu_backward()` throws `std::logic_error` with no `gpu_forward()` before it (it dereferenced a null
  `GPUState`) and `std::invalid_argument` unless `dout` is `[tgt, d_model]`.
- **Weights on the device.** `gpu_upload_weights()` copies the CPU weights on the first GPU call and
  keeps them. The setters, `load()`, `merge_lora()` and `update_weights()` refresh that copy through
  the private `weights_changed()` (which also advances the weights version, TD-300; the refresh is
  as `LayerNorm`'s, TD-426); they used to leave it stale. An
  optimizer step taken elsewhere (the blocks, decoder and model step their optimizer themselves)
  still needs `EncoderDecoderModel::gpu_sync_weights()`. During GPU training the device copy is the
  live one: `GPUOptimizer` steps it (TD-468) and `gpu_download_weights()` brings the host weights up
  to date. `gpu_download_grads()` **adds** device grads into the CPU grads, so the device grads are
  zeroed between steps (`gpu_zero_grads()`).
- **Attention weights.** `get_last_attention_weights()` reflects CPU forwards; the last
  `gpu_forward()`'s are downloaded by `gpu_last_mean_attention_weights()` (TD-342).
  `gpu_forward_with_cache()` doesn't fill `GPUState`'s backward caches.
- `DecoderBlock`'s GPU path runs the world-model and hippocampal instances too, since TD-342.

---

## 6. Tests

`crossattention_test.cpp` has 54 tests (58 in a GPU build), among them:

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
| Construction | 1 (TD-299: zero and negative shapes) |
| Persistence | 1 (TD-298/TD-423: truncated and empty files) |
| CacheSource | 5 (TD-300: the same encoder output keeps the cache; a different length, a same-length different output, a weight or LoRA change, another layer's cache are refused; a cleared cache takes a new output) |
| Setters | 1 (TD-301: shape checked, nothing changed on a refusal) |
| GPU contract (GPU builds) | 5 (TD-301: LoRA refused until merged, shapes, backward without a forward, host changes reach the mirror; TD-300: the cached path checks the encoder length and the weights) |

The cached path is covered in `inference_optimization_test.cpp` and the encoder-decoder suites. The
older [CROSSATTENTION_TEST_COVERAGE.md](../../testing/coverage/CROSSATTENTION_TEST_COVERAGE.md)
predates TD-059/TD-038/TD-174.

**Not covered:**
- a shared optimizer across layers;
- a gradient check on `grad_kv_input` or with LoRA/bias;
- anything GPU-side in CI (the GPU tests run in the SYCL build, on the oneAPI CPU device and on
  ai-machine's Arc through the GPU test package). The TD-301 tests all fail on the old code, the
  backward test with a segfault.

---

## 7. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `update_weights()` calls a possibly **shared** optimizer's `step()`, which steps every registered layer; double `set_optimizer()` double-registers (§4) | Public `EncoderDecoderModel::train_step()`/`update_weights()` over-step every parameter when an optimizer is registered; shipped trainers unaffected | **Verified** | [TD-297](../../guides/TECHNICAL_DEBT.md#td-297-layer-update_weights-steps-a-shared-optimizer-once-per-layer) |
| `load()` doesn't check reads; truncated file → NaN weights, no error; `learning_rate` clobbered before the mismatch check; `save()` unchecked (§4) | A damaged legacy `.cross_attn` file silently produces a NaN model | **Verified** | [TD-298](../../guides/TECHNICAL_DEBT.md#td-298-crossattentionload-loads-truncated-files-as-nan-weights) |
| `num_heads == 0` → SIGFPE before the validation runs; negative heads accepted (§4) | **Fixed in code November 13, 2026**: checked before `d_k` is computed | **Verified** by test | [TD-299](../../guides/TECHNICAL_DEBT.md#td-299-crossattention-crashes-with-sigfpe-when-num_heads-is-zero) |
| `forward_with_cache()` silently ignored a new `kv` once the cache was filled; no staleness check (§4) | **Fixed in code** November 13, 2026: shape, sampled content, layer and weights checked; the content isn't checked on the GPU path, and an input matching at every sampled position isn't caught | **Verified** | [TD-300](../../guides/TECHNICAL_DEBT.md#td-300-crossattentionforward_with_cache-silently-reuses-stale-encoder-kv) |
| GPU path ignored LoRA, skipped validation, dereferenced null in `gpu_backward()` without a forward, kept stale weights after host changes (§5) | **Fixed in code** November 13, 2026 | **Verified** on the oneAPI CPU device | [TD-301](../../guides/TECHNICAL_DEBT.md#td-301-crossattention-gpu-path-diverges-from-the-cpu-path) |
| An optimizer step taken outside `update_weights()` doesn't reach the device copy; `get_last_attention_weights()` is CPU-only (§5) | Callers call `gpu_sync_weights()`, or use `gpu_last_mean_attention_weights()` | By inspection | Documented contract (TD-301) |
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

---

## Addendum (October 20, 2026): GPU score bias (TD-342)

`gpu_forward()` takes an optional `score_bias` ([tgt, src], the same for every head, added after scaling and before the mask, as
`forward_with_scores()`). `gpu_last_mean_attention_weights()` downloads the mean-across-heads weights of the last `gpu_forward()`,
as `get_last_attention_weights()`. Backward needs no change. Test: `gpu_optimizer_test.cpp`'s `ScoreBiasMatchesForwardWithScores`.

## Addendum (October 26, 2026): checked checkpoint I/O (TD-423)

`save()`/`load()` (TD-298; `learning_rate` is assigned only after validation) use `adai::binio::Reader`/`Writer` (`src/BinaryIO.{hpp,cpp}`).
- **Loading.** A file that is short, empty or longer than this layer's shapes throws `std::runtime_error`. The whole file is read
  into temporaries first, so a failed load leaves the layer unchanged.
- **Saving.** A failed write or flush (a full disk) throws instead of leaving a short file.

Tests: `CrossAttentionPersistenceTest.TruncatedOrEmptyFileThrowsAndLeavesTheLayerUnchanged`, `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`.

## Addendum (October 26, 2026): one update per step (TD-297, TD-346, TD-360)

TD-297: the blocks, encoder, decoder and model step a shared optimizer once and don't call the layers' `update_weights()`; `Optimizer::add_parameter_group()` ignores duplicate registrations. Test: `EncoderDecoderModelTest.OneUpdateIsOneStepWithOrWithoutAnOptimizer`.

## Addendum (October 26, 2026): cached attention reads the KV cache in place (TD-388)

`GPUKVCache` exposes its populated prefix as a view (`keys_data()`, `values_data()`, `leading_dimension()`) instead of copying it on
every `get_keys()`/`get_values()`. The new `matrix_gemm_strided_gpu()` is a row-major GEMM with transpose flags, alpha and leading
dimensions (`cublasSgemm` / oneMKL `gemm`). The cached self- and cross-attention paths use it to compute each head straight from the
cache and to write each head's output into its own columns, with no prefix copies, per-row slices or scatter. A 100-token decode on
the oneAPI CPU device went from 6.6 s to 1.4 s. Tests: `MatrixGPUTD388Test.StridedGemmMatchesHostReference`,
`EncoderDecoderGPUTest.CachedGpuDecodeMatchesCpuOverALongGeneration`.

## Addendum (November 13, 2026): shape checked before `d_k` (TD-299)

`checked_head_dim()` validates `d_model`/`num_heads` in the initializer list (§4), so `num_heads == 0` throws
`std::invalid_argument` instead of killing the process. Test: `CrossAttentionConstructionTest.NonPositiveShapesThrowInsteadOfCrashing`.

## Addendum (November 13, 2026): the GPU path's contract (TD-301)

Now reflected in §4–§7: the GPU forwards refuse attached LoRA adapters and check widths, mask and score-bias shapes, `gpu_backward()`
needs a forward of the same shape, and the setters (now shape-checked), `load()`, `merge_lora()` and `update_weights()` refresh the
device copy of the weights. Tests: `CrossAttentionSetterTest`, `CrossAttentionGPUContractTest`.

## Addendum (November 13, 2026): the cache answers only for its own encoder output (TD-300)

Now reflected in §4–§7: `forward_with_cache()` records the layer, weights version and encoder output a cache was filled from and
refuses a later call that differs; `gpu_forward_with_cache()` checks the row count and the weights. Tests: `CrossAttentionCacheSourceTest`
(the three refusal tests fail on the old code) and `CrossAttentionGPUContractTest.ACachedGpuCallChecksTheEncoderLengthAndTheWeights`.
