# `MultiHeadAttention` — Source File Reference

- **Files:** [`src/MultiHeadAttention.hpp`](../../../../src/MultiHeadAttention.hpp), [`src/MultiHeadAttention.cpp`](../../../../src/MultiHeadAttention.cpp) (1,595 lines together)
- **Namespace:** global
- **Built into:** `adai_attention`; owned by every `EncoderBlock` and `DecoderBlock` (self-attention), hence by `LLMEncoder`, `LLMDecoder`, `LeJEPAEncoder`; saved by `ModelSerializer`
- **Status tag:** `@adai-status: beta` (capped by TD-050), `@adai-version: 0.14.0` (`.hpp`) / `0.15.0` (`.cpp`), `@adai-reviewed: 2026-11-13`
- **Tests:** [`tests/multiheadattention_test.cpp`](../../../../tests/multiheadattention_test.cpp) (65 tests; 73 in a GPU build)
- **Sibling:** [CrossAttention.md](CrossAttention.md), which shares most of this design and already tracks the shared issues (TD-297–TD-303)
- **Last traced against the code:** 2026-11-13

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
| `MultiHeadAttention(d_model, num_heads)` | Four `d_model × d_model` weights, `N(0, √(2/d_model))` via the unseeded `Matrix::randomize()` (TD-368); `d_k` comes from `checked_head_dim()` in the initializer list, which throws `std::invalid_argument` when `d_model` or `num_heads` is below 1 or `d_model % num_heads != 0` (TD-467: `num_heads == 0` used to die with SIGFPE before the old divisibility check ran; `CrossAttention` has the same helper, TD-299) |
| `forward(X, mask)` → `forward_parallel(X, mask, true)` | §1. Checks `X.cols == d_model` and the mask shape. Caches `X`, Q/K/V, per-head weights, the head-averaged weights (`get_attention_weights()`) and the concatenated output for `backward()`. Fires the attention hook with the averaged weights |
| `forward_with_cache(X_new, mask, kv_cache, use_cache)` | Incremental decoding: projects only the new rows, appends K/V to the `KVCache` (whose copy-on-append cost is TD-422), attends over the full cache; mask is `[new × total]`. Does **not** fire the attention hook |
| `backward(dY)` | Accumulates `W_o`/`W_q`/`W_k`/`W_v` gradients and returns dX, differentiating each head's own softmax; LoRA branches add their parts. **Verified correct** (also covered by `BackwardPassMatchesNumericalGradient`): against central finite differences, dX's error norm was 2.5–8×10⁻⁵ of the gradient's (with and without a causal mask), and `W_q`'s gradient agreed within 0.6% |
| `set_optimizer()`, `register_parameters()`, `update_weights()`, `zero_grad()` | As the other layers; `update_weights()` steps a shared optimizer (TD-297), then refreshes the GPU mirror (TD-301) |
| `get_gradient_norm()`, `clip_gradients(max)` | Over the four weight gradients only (not LoRA's, TD-302) |
| `enable_lora()`, `register_lora_parameters()`, `merge_lora()` | LoRA on Q/K/V/O (experimental, no production caller; see [LoRA.md](LoRA.md)); CPU path only, the GPU forwards refuse attached adapters, and `merge_lora()` refreshes the GPU mirror (TD-301) |
| `set_attention_hook(fn)` | Called after each `forward()` with the averaged weights; used by `ChatbotTrainer` and the LeJEPA pass for attention-entropy metrics (TD-013) |
| `save_weights()` / `load_weights()` | `d_model`, `num_heads`, then the four matrices, through `adai::binio` (TD-423, addendum): a short or over-long file throws and changes nothing; a load refreshes the GPU mirror (TD-301). Both log at debug level |
| Getters/setters for `W_q`…`W_o` | The setters (defined in the `.cpp` since TD-301) need a `d_model × d_model` matrix, else `std::invalid_argument` and no change, and refresh the GPU mirror |
| `print_config(name)` | One `adai::Logger::info` line: `d_model`, heads, `d_k`, parameters and megabytes, learning rate. No production caller |
| GPU (`ADAI_ENABLE_GPU`) | `gpu_upload_weights()`, `gpu_collect_parameters()`, `gpu_download_weights()`, `gpu_download_grads()`, `gpu_zero_grads()`, `gpu_forward()`, `gpu_forward_with_cache()`, `gpu_backward()` over a persistent `GPUState`; kernels in MatrixGPU_cu.md/MatrixGPU_SYCL_cpp.md (softmax race TD-384). The contract (TD-301) is in §4 |

---

## 3. Problems found October 14, 2026 (fixed)

### A truncated weights file loaded silently

`load_weights()` used to read the four matrices without checking the stream: a file cut in half loaded without an error, with
all 256 entries of a 16-wide layer's `W_o` still random. **Fixed October 26, 2026** (TD-423, addendum below).

### `num_heads == 0` crashed the process

`MultiHeadAttention(8, 0)` died with SIGFPE (integer division by zero in the initializer list), before the divisibility check
that would have thrown. **Fixed November 13, 2026** (TD-467): `checked_head_dim()` validates both values before dividing (§2).
MNS now rejects such architectures too (TD-461).

---

## 4. Smaller items

- **Per-head loops (TD-467, fixed November 13, 2026).** `forward_parallel()`'s score, mask, softmax and value loops read and
  write through row pointers (`Q.data[i].data() + h·d_k`, …) instead of the out-of-line, bounds-checked `Matrix::operator()`
  (TD-435), handle one query row at a time, and accumulate the value product over `V`'s rows instead of walking it column-wise.
  Every sum adds the same terms in the same order: forward (with and without a causal mask) and backward outputs hashed bit-for-bit
  equal before and after for `d_model`/heads/sequence of 64/2/17, 64/8/33 and 96/3/17. The explicit mask-shape check in
  `forward_parallel()` is what keeps a too-small mask from being read past its end. Parallelism is still over heads only, so at
  most `num_heads` threads; `backward()`'s loops are unchanged.
- **GPU path contract (TD-301, fixed November 13, 2026).**
  - *LoRA:* `gpu_forward()`/`gpu_forward_with_cache()` throw `std::logic_error` while adapters are attached; they used to run the
    base weights silently. `merge_lora()` first.
  - *Shapes:* the input must be `d_model` wide; `gpu_forward()`'s mask `[seq, seq]`, `gpu_forward_with_cache()`'s
    `[new, cached + new]`, checked before the cache grows. Otherwise `std::invalid_argument`; the mask used to be checked only by
    its element count (`masked_fill_inplace()`). `gpu_backward()` throws `std::logic_error` with no `gpu_forward()` before it (it
    dereferenced a null `GPUState`) and `std::invalid_argument` unless `dY` is `[seq, d_model]`.
  - *Device mirror:* the first GPU call uploads the weights and keeps them on the device. The setters, `load_weights()`,
    `merge_lora()` and `update_weights()` refresh it through the private `refresh_gpu_mirror()` (as `LayerNorm`'s, TD-426); they
    used to leave the GPU path on the old weights. An optimizer step taken elsewhere (the blocks, encoder, decoder and model step
    their optimizer themselves) still needs `EncoderDecoderModel::gpu_sync_weights()`. During GPU training the mirror is the live
    copy, and `gpu_download_weights()` brings the host weights up to date.
  - `get_attention_weights()` and the attention hook reflect CPU forwards only; `gpu_forward()` has its own entropy hook.
- **Thread safety**: all intermediate state is cached in members, so concurrent `forward()` calls on one instance race; callers
  serialize with `model_mutex_` (TD-156/TD-227 context).
- **Fully masked rows** become a uniform distribution over the masked positions rather than zeros (all scores −1e9). No current
  caller produces one.
- **Console output** (fixed November 13, 2026, TD-467): `save_weights()`/`load_weights()`/`print_config()` log through
  `adai::Logger`.

---

## 5. Tests

`multiheadattention_test.cpp` (68 tests): construction and divisibility, forward shapes, causal and padding masks, attention
weights as distributions, a per-head reference implementation and the single-head case (TD-059), numerical-gradient checks with and
without LoRA, LoRA freezing and merging, gradient accumulation, updates with and without an optimizer, gradient norm and clipping,
save/load and dimension mismatch, training loops, the attention hook (CPU and GPU). `DimensionNotDivisible` also covers zero and
negative `d_model`/`num_heads` (TD-467; the old code died with SIGFPE), and the `PrintConfig` tests read the log.
`MultiHeadAttentionSetterTest` and, in a GPU build, `MultiHeadAttentionGPUContractTest` (four tests: LoRA refused until merged,
shapes checked before the cache grows, backward needs a forward, setters/load/update reach the mirror) cover TD-301; all five fail
on the old code, the backward test with a segfault. Truncated weight
files: `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`. (The KV-cached forward is covered in
`inference_optimization_test.cpp`.)

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Unchecked loads, `num_heads == 0` SIGFPE, checked accessor and column-wise `V` walks in the forward loops, console output (§3, §4) | **Fixed in code** (loads October 26, the rest November 13, 2026) | **Verified** | [TD-467](../../guides/TECHNICAL_DEBT.md#td-467-multiheadattention-unchecked-loads-num_heads--0-sigfpe-and-hot-loop-overhead) |
| Member-cached state (not concurrency-safe per instance); uniform fully masked rows (§4) | Callers serialize with `model_mutex_`; no caller produces a fully masked row | By inspection | Not tracked |
| GPU path ran without LoRA silently, checked no shapes, dereferenced null in `gpu_backward()` without a forward, kept stale weights after host changes (§4) | **Fixed in code** November 13, 2026 | **Verified** on the oneAPI CPU device | [TD-301](../../guides/TECHNICAL_DEBT.md#td-301-crossattention-gpu-path-diverges-from-the-cpu-path) |
| An optimizer step taken outside `update_weights()` doesn't reach the mirror (§4) | Callers call `gpu_sync_weights()`; GPU training steps on the device | By inspection | Documented contract (TD-301) |
| LoRA persistence/grad norm; accessor cost elsewhere; GPU incremental cache | Open | — | TD-302, TD-435, TD-050 |
| Shared-optimizer stepping; unseeded init; GPU softmax race; KV-cache copying | Fixed in code | — | TD-297, TD-368, TD-384, TD-422 |

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

## Addendum (October 26, 2026): checked checkpoint I/O (TD-423)

`save_weights()`/`load_weights()` (TD-467's read item) use `adai::binio::Reader`/`Writer` (`src/BinaryIO.{hpp,cpp}`).
- **Loading.** A file that is short, empty or longer than this layer's shapes throws `std::runtime_error`. The whole file is read
  into temporaries first, so a failed load leaves the layer unchanged.
- **Saving.** A failed write or flush (a full disk) throws instead of leaving a short file.

Tests: `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`.

## Addendum (October 26, 2026): cached attention reads the KV cache in place (TD-388)

`GPUKVCache` exposes its populated prefix as a view (`keys_data()`, `values_data()`, `leading_dimension()`) instead of copying it on
every `get_keys()`/`get_values()`. The new `matrix_gemm_strided_gpu()` is a row-major GEMM with transpose flags, alpha and leading
dimensions (`cublasSgemm` / oneMKL `gemm`). The cached self- and cross-attention paths use it to compute each head straight from the
cache and to write each head's output into its own columns, with no prefix copies, per-row slices or scatter. A 100-token decode on
the oneAPI CPU device went from 6.6 s to 1.4 s. Tests: `MatrixGPUTD388Test.StridedGemmMatchesHostReference`,
`EncoderDecoderGPUTest.CachedGpuDecodeMatchesCpuOverALongGeneration`.

## Addendum (November 13, 2026): shape check and per-head loops (TD-467)

Now reflected in §2–§6: the constructor validates `d_model`/`num_heads` before computing `d_k` (no more SIGFPE), the forward's
per-head loops use row pointers with bit-for-bit identical results, and logging goes through `adai::Logger`.

## Addendum (November 13, 2026): the GPU path's contract (TD-301)

Now reflected in §2, §4–§6: the GPU forwards refuse attached LoRA adapters and check the input width and mask shape, `gpu_backward()`
needs a forward of the same shape, and the setters (now shape-checked), `load_weights()`, `merge_lora()` and `update_weights()`
refresh the device mirror. Tests: `MultiHeadAttentionSetterTest`, `MultiHeadAttentionGPUContractTest`.
