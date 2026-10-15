# `MultiHeadAttention` — Source File Reference

- **Files:** [`src/MultiHeadAttention.hpp`](../../../../src/MultiHeadAttention.hpp), [`src/MultiHeadAttention.cpp`](../../../../src/MultiHeadAttention.cpp) (1,533 lines together)
- **Namespace:** global
- **Built into:** `adai_attention`; owned by every `EncoderBlock` and `DecoderBlock` (self-attention), hence by `LLMEncoder`, `LLMDecoder`, `LeJEPAEncoder`; saved by `ModelSerializer`
- **Status tag:** `@adai-status: beta` ("capped by TD-050"; TD-038 LoRA, TD-050 GPU incremental-cache forward, TD-195 accumulating gradients), `@adai-version: 0.12.1`, `@adai-reviewed: 2026-09-18` (both files)
- **Tests:** [`tests/multiheadattention_test.cpp`](../../../../tests/multiheadattention_test.cpp) (68 tests)
- **Sibling:** [CrossAttention.md](CrossAttention.md), which shares most of this design and already tracks the shared issues (TD-297–TD-303)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with scratch programs linked against the real class.

---

## 1. What this file is

Multi-head **self**-attention:

```text
Q, K, V = X·W_q, X·W_k, X·W_v                 (each [seq × d_model]; + LoRA deltas if enabled)
for h in 0..num_heads-1:                       (OpenMP over heads)
    Q_h, K_h, V_h = columns [h·d_k, (h+1)·d_k)  (d_k = d_model / num_heads)
    A_h = softmax(Q_h·K_hᵀ / √d_k, masked: mask(i,j) == 0 → −1e9)
    out[:, h·d_k …] = A_h·V_h
Y = out·W_o                                    (+ LoRA)
```

Until TD-059 (fixed September 13, 2026) this was single-head attention over the full width with a mismatched scale; the
per-head split is now real on both the CPU and GPU paths. Masks are multiplicative 0/1 (`1` = may attend), which is what every
caller builds (`LLMDecoder::create_causal_mask()`, the KV-cached decode mask).

---

## 2. API

| Member | Behaviour |
|---|---|
| `MultiHeadAttention(d_model, num_heads)` | Four `d_model × d_model` weights, `N(0, √(2/d_model))` via the unseeded `Matrix::randomize()` (TD-368); throws if `d_model % num_heads != 0`. **`num_heads == 0` crashes with SIGFPE** before that check (`d_k = d_model / num_heads` in the initializer list), verified, as TD-299 for `CrossAttention` |
| `forward(X, mask)` → `forward_parallel(X, mask, true)` | §1. Checks `X.cols == d_model` and the mask shape. Caches `X`, Q/K/V, per-head weights, the head-averaged weights (`get_attention_weights()`) and the concatenated output for `backward()`. Fires the attention hook with the averaged weights |
| `forward_with_cache(X_new, mask, kv_cache, use_cache)` | Incremental decoding: projects only the new rows, appends K/V to the `KVCache` (whose copy-on-append cost is TD-422), attends over the full cache; mask is `[new × total]`. Does **not** fire the attention hook |
| `backward(dY)` | Accumulates `W_o`/`W_q`/`W_k`/`W_v` gradients and returns dX, differentiating each head's own softmax; LoRA branches add their parts. **Verified correct** (also covered by `BackwardPassMatchesNumericalGradient`): against central finite differences, dX's error norm was 2.5–8×10⁻⁵ of the gradient's (with and without a causal mask), and `W_q`'s gradient agreed within 0.6% |
| `set_optimizer()`, `register_parameters()`, `update_weights()`, `zero_grad()` | As the other layers; `update_weights()` steps a shared optimizer (TD-297) |
| `get_gradient_norm()`, `clip_gradients(max)` | Over the four weight gradients only (not LoRA's, TD-302) |
| `enable_lora()`, `register_lora_parameters()`, `merge_lora()` | LoRA on Q/K/V/O (experimental, no production caller; see [LoRA.md](LoRA.md)) |
| `set_attention_hook(fn)` | Called after each `forward()` with the averaged weights; used by `ChatbotTrainer` and the LeJEPA pass for attention-entropy metrics (TD-013) |
| `save_weights()` / `load_weights()` | `d_model`, `num_heads`, then the four matrices. `load_weights()` checks the dimensions but **not its reads** (§3); both print to `std::cout` |
| Getters/setters for `W_q`…`W_o`, `print_config()` | Setters don't check shape or refresh the GPU copy (as TD-370) |
| GPU (`ADAI_ENABLE_GPU`) | `gpu_upload_weights()`, `gpu_download_grads()`, `gpu_zero_grads()`, `gpu_forward()`, `gpu_backward()` over a persistent `GPUState`; kernels in MatrixGPU_cu.md/MatrixGPU_SYCL_cpp.md (softmax race TD-384) |

---

## 3. Problems (verified)

### A truncated weights file loads silently

`load_weights()` reads the four matrices without checking the stream. A file cut in half loaded **without an error**, and
**all 256 entries of `W_o`** (in a 16-wide layer) were still the fresh layer's random initialization. Earlier entries came from
the file, later ones stayed random: a damaged checkpoint sidecar yields a quietly corrupted layer. Same class as TD-298
(`CrossAttention`, which pre-fills NaN), TD-369 (`FeedForward`), TD-423 (`LanguageModelHead`) and TD-426 (`LayerNorm`): the legacy
per-layer loaders all need the same checked-read fix.

### `num_heads == 0` crashes the process

`MultiHeadAttention(8, 0)` died with **SIGFPE** (integer division by zero in the initializer list), before the divisibility check
that would have thrown. Architecture comes from MNS (immutable, and not validated for this, TD-461), so a bad record crashes the
trainer or server rather than producing an error.

---

## 4. Smaller items (by inspection)

- **The hot loops use the bounds-checked accessor.** The per-head score, softmax and value loops index `Q(i, …)`, `K(j, …)`,
  `V(j, …)` through `Matrix::operator()`, which is out-of-line and bounds-checked (2.4× slower than `data[i][j]`, TD-435); the
  value loop also walks `V` column-wise. Parallelism is over heads only, so at most `num_heads` threads.
- **Thread safety**: all intermediate state is cached in members, so concurrent `forward()` calls on one instance race; callers
  serialize with `model_mutex_` (TD-156/TD-227 context).
- **Fully masked rows** become a uniform distribution over the masked positions rather than zeros (all scores −1e9). No current
  caller produces one.
- **Console output** in `save_weights()`/`load_weights()`/`print_config()`.

---

## 5. Tests

`multiheadattention_test.cpp` (68 tests): construction and divisibility, forward shapes, causal and padding masks, attention
weights as distributions, a per-head reference implementation and the single-head case (TD-059), numerical-gradient checks with and
without LoRA, LoRA freezing and merging, gradient accumulation, updates with and without an optimizer, gradient norm and clipping,
save/load and dimension mismatch, training loops, the attention hook (CPU and GPU).

**Not covered:** truncated weight files, `num_heads == 0`. (The KV-cached forward is covered in `inference_optimization_test.cpp`.)

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `load_weights()` doesn't check reads: a half file loads silently, the rest of the weights stay random (§3) | A damaged checkpoint sidecar yields a corrupted layer with no error | **Verified** | [TD-467](../../guides/TECHNICAL_DEBT.md#td-467-multiheadattention-unchecked-loads-num_heads--0-sigfpe-and-hot-loop-overhead) |
| `num_heads == 0` → SIGFPE before validation (§3) | A bad architecture record crashes the process | **Verified** | [TD-467](../../guides/TECHNICAL_DEBT.md#td-467-multiheadattention-unchecked-loads-num_heads--0-sigfpe-and-hot-loop-overhead) |
| Bounds-checked accessor and column-wise `V` walks in hot loops; member-cached state; uniform fully-masked rows; console output (§4) | Slower CPU attention; not concurrency-safe per instance | By inspection | [TD-467](../../guides/TECHNICAL_DEBT.md#td-467-multiheadattention-unchecked-loads-num_heads--0-sigfpe-and-hot-loop-overhead) |
| Shared-optimizer stepping; LoRA persistence/grad norm; unseeded init; GPU softmax race; KV-cache copying; accessor cost | Already tracked | — | TD-297, TD-302, TD-368, TD-384, TD-422, TD-435 |

---

## 7. Merge notes

This file replaces `docs/development/api/attention/multihead-attention.md` (1,645 lines). Where they disagree, this file wins:
- **Single-head math.** Its algorithm sections describe one `Q·Kᵀ/√d_k` over the full width, the pre-TD-059 behaviour; the
  implementation is now genuinely per head.
- **KV-cache speedups.** It claimed "~25× speedup for 50 tokens" and "2–3× faster" without measurements; the cache's
  copy-on-append makes cached decoding O(n²·d) in copying (TD-422), so those figures shouldn't be relied on.

Kept: the formulation, mask semantics, the cached-forward contract, the attention hook and persistence layout. Dropped: generic
tutorial material (complexity tables, training-loop and encoder-layer examples, future-enhancement sketches such as dropout and
Flash Attention).
