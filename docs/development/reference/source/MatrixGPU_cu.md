# `gpu/MatrixGPU.cu` — Source File Reference

- **File:** [`src/gpu/MatrixGPU.cu`](../../../../src/gpu/MatrixGPU.cu) (1,063 lines, CUDA)
- **Namespace:** `adai::gpu`
- **Built into:** `adai_gpu` in CUDA builds (`ENABLE_GPU`, the `gpu` preset). The free functions are declared in `src/gpu/MatrixGPU.hpp` (documented separately), which also defines `GPUMatrix` on top of them
- **SYCL counterpart:** `src/gpu/sycl/MatrixGPU_SYCL.cpp`. The training kernels here were ported from it "formula-for-formula", so most findings apply to both (see §4)
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.9.2`, `@adai-reviewed: 2026-10-17` (TD-384 barriers, TD-382 cuBLAS lock)
- **Tests:** none run real kernels in any environment used so far. `matrixgpuTd003Tests` exists in `build/cuda-gputest` and skips without a device
- **Last traced against the code:** 2026-10-19

> Code-traced reference. **No CUDA device is available here**, so no kernel in this file was run;
> everything below is by inspection. The math of the training kernels was checked by hand against
> the CPU implementations.

---

## 1. What this file is

The **CUDA implementations of every GPU matrix primitive** the model uses: element-wise
arithmetic, cuBLAS GEMM, transpose, activations, reductions, and the fused training kernels
(softmax, LayerNorm, GELU backward, cross-entropy, bias, masking), plus two diagnostics reductions and
two transfer helpers. All launches go on `GPUManager::get_stream()`, and GEMMs go through
`GPUManager::get_cublas_handle()` (see [GPUUtils.md](GPUUtils.md)). Errors are checked with
`cudaGetLastError()` after each launch (`CUDA_CHECK`), which catches launch-configuration errors but
not faults during execution; those surface at the next synchronizing call.

### Why it matters

On a CUDA host this file *is* the model's GPU math. `GPUMatrix` (attention, feed-forward, LayerNorm,
LM head) and `EncoderDecoderModel`'s GPU training step call these functions directly, and
`Matrix::operator*` calls `matrix_multiply_gpu()` for every large product. A wrong kernel here gives
silently wrong training, as TD-061 (LayerNorm backward) showed before it was fixed.

---

## 2. Kernels and host wrappers

### Element-wise and BLAS

| Function | Behaviour |
|---|---|
| `matrix_add_gpu`, `matrix_add_scalar_gpu`, `matrix_multiply_elementwise_gpu`, `matrix_multiply_scalar_gpu` | `float4` vectorized kernels over `size / 4` elements, plus a one-block scalar tail for the remainder. **Precondition: every pointer must be 16-byte aligned**; a misaligned pointer is a misaligned-address fault. All current callers pass `GPUMemory` base pointers (aligned by `cudaMalloc`); a future caller passing a sliced pointer (`ptr + k` with `k % 4 != 0`) would fault. `matrix_add_scalar_gpu` has no caller |
| `matrix_transpose_gpu` | 32×32 shared-memory tile with +1 padding (no bank conflicts) |
| `matrix_multiply_gpu(a, b, c, m, k, n)` | `cublasSgemm` computing row-major `C = A·B` as column-major `Cᵀ = Bᵀ·Aᵀ` (operands swapped). Correct. No dimension checks |
| `matrix_apply_activation_gpu` | In-place ReLU, sigmoid, tanh or tanh-approximation GELU (`float4` plus tail). GELU's constants are duplicated from `Activation` (TD-216) |
| `matrix_sum_gpu` | Warp-shuffle block reduction, then **recursion** on the block sums until one block remains; returns 0 for `size <= 0` (TD-088). **Each level allocates a `GPUMemory` buffer** (`cudaMalloc`/`cudaFree`, both synchronizing) and the last level does a blocking copy, so every call stalls the CPU on the GPU. It's called every training step (the loss, below) |
| `matrix_batch_add_gpu`, `matrix_batch_multiply_gpu` | Stage the pointer arrays with `cudaMallocAsync`/`cudaFreeAsync` (CUDA 11.2+), then a fused batched add or `cublasSgemmBatched`. **No caller anywhere**; they're groundwork for batching that doesn't exist yet (TD-171) |

### Training kernels (ported from SYCL, "TD-003")

These use one 256-thread block per row with a shared-memory tree reduction. The comment says
they're "deliberately unoptimized" to limit risk in code that can't be tested here.

| Function | Behaviour |
|---|---|
| `matrix_add_inplace_gpu` | `dst += src` |
| `matrix_softmax_rows_gpu` | Per-row max, then `exp(x − max)` and sum, then divide. Used by `GPUMatrix::softmax_rows_inplace()` (attention) |
| `matrix_softmax_backward_gpu` | `din = s ⊙ (dout − Σ s·dout)`. Correct |
| `matrix_gelu_backward_gpu` | Derivative of the tanh-approximation GELU. Correct |
| `matrix_layer_norm_fwd_gpu` | Mean, variance, then `xn = (x − μ)·rstd` and `y = xn·γ + β`; also writes per-row mean and rstd |
| `matrix_layer_norm_bwd_gpu` | `dx` per row from `Σ d_xn·xn` and `Σ d_xn`, plus `dγ`/`dβ` **accumulated** (`+=`) by a column-parallel kernel. The TD-061 fix (`r²` instead of `r³`) checks out: with `xn = (x − μ)·r`, `Σ d_xn·(x − μ) = Σ d_xn·xn / r`. The unused `mean` argument is kept for API symmetry with SYCL |
| `matrix_add_row_bias_gpu`, `matrix_sum_rows_gpu` | Broadcast bias add; column sums (overwrite, not accumulate; callers accumulate via `add_inplace`) |
| `matrix_masked_fill_gpu` | `data = fill` where `mask == 0` |
| `matrix_cross_entropy_loss_gpu` | Per-row log-sum-exp minus the target logit, then `matrix_sum_gpu(...) / seq_len` (so it synchronizes). **Doesn't check targets**: a target outside `[0, vocab_size)` reads out of bounds |
| `matrix_cross_entropy_grad_gpu` | `(softmax − onehot) / seq_len`. Same target precondition |

### Diagnostics and transfers

| Function | Behaviour |
|---|---|
| `matrix_count_below_threshold_gpu` | Count of `|x| < t` (warp-shuffle reduction); `GPUMatrix::count_below_threshold()` divides by the size for `FeedForward`'s saturation hook |
| `matrix_row_entropy_gpu` | Per-row `−Σ p·log(p + ε)` for attention-entropy hooks |
| `matrix_copy_device_to_device_gpu`, `matrix_download_gpu` | Async device copy; blocking download. Used by layers in place of backend-specific calls |

---

## 3. A data race in the multi-pass row kernels (by inspection, definite)

`softmax_rows_kernel`, `layer_norm_fwd_kernel`, `cross_entropy_loss_kernel` and
`cross_entropy_grad_kernel` each do two reductions in sequence through **the same shared `scratch`
buffer**:

```cpp
for (s = WG/2; s > 0; s >>= 1) { ...; __syncthreads(); }
const float row_max = scratch[0];   // every thread reads scratch[0]
float my_sum = ...;                 // thread 0 may finish this loop early...
scratch[lid] = my_sum;              // ...and overwrite scratch[0] here
__syncthreads();
```

Nothing separates the reads of `scratch[0]` from thread 0's next write to `scratch[0]`. Warps in a
256-thread block aren't scheduled in lockstep, so a warp that's slow to read `row_max` (or `mean`)
can read thread 0's partial sum instead. That's a read/write data race on shared memory, which is
undefined behaviour under the CUDA memory model; `compute-sanitizer --tool racecheck` would flag it.
When it hits, one row of that call is wrong:
- in softmax, `exp(x − wrong_max)` can overflow to `inf`/`NaN` or underflow;
- in LayerNorm, the row is centred on a variance partial instead of the mean;
- in cross-entropy, the loss and gradient for that position are wrong.

It's likely rare, because thread 0 must finish a whole loop pass before a lagging warp executes one
load. But it's intermittent, silent and data-dependent, and it touches attention, every LayerNorm and
the loss in GPU training.

**The SYCL backend has the identical pattern** (`MatrixGPU_SYCL.cpp`'s `matrix_softmax_rows_gpu`
reads `scratch[0]` and then writes `scratch[lid]` without an intervening `item.barrier()`). ai-machine
trains with SYCL on an Intel Arc GPU, where a 256-item work-group runs as several independent
sub-groups, so the hazard applies there too. **Verified October 12, 2026** (correcting an earlier note here):
the SYCL version reproduces deterministically on the oneAPI CPU device, and adding the barrier fixes
all four kernels; see [MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md) §4. The CUDA kernels have the same
shape and haven't been run.

The fix is one barrier after each read of `scratch[0]` that's followed by another reduction (or
reading into a register before a barrier), in both backends.

**Fixed in code October 17, 2026:** each of the four kernels now has a `__syncthreads()` right after
its `scratch[0]` read, and the SYCL kernels have the matching `item.barrier(local_space)`. These are
the only reads followed by another reduction; the other `scratch[0]` reads are each kernel's last
use of the buffer. The CUDA change builds (`gpu` preset) but hasn't been run, since no CUDA device
is available. The SYCL change is verified on the oneAPI CPU device by the new
`MatrixGPUTD003Test.SoftmaxWideRowsMatchHost`/`LayerNormWideRowsMatchHost` and by
`EncoderDecoderGPUTest.GpuForwardLossMatchesCpuForwardTeacherForcing`. All three fail every run
without the barriers, even with TD-393 fixed.

---

## 4. Who calls what

| Caller | Uses |
|---|---|
| `GPUMatrix` (`MatrixGPU.hpp`) | Nearly everything: element-wise ops, GEMM, transpose, activations, softmax and its backward, LayerNorm, GELU backward, bias, row sums, masked fill, in-place add, reductions, transfers |
| `Matrix` (`Matrix.cpp`) | `multiply_gpu()` → `matrix_multiply_gpu`, plus `add`/`hadamard`/`scalar` GPU paths |
| `EncoderDecoderModel` (GPU training) | `matrix_cross_entropy_loss_gpu` / `_grad_gpu` per step |
| `FeedForward`, `MultiHeadAttention`, `CrossAttention`, `LanguageModelHead` | Transfer helpers, bias add, GELU backward, diagnostics |
| *(nobody)* | `matrix_batch_add_gpu`, `matrix_batch_multiply_gpu`, `matrix_add_scalar_gpu` |

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Shared-memory race between consecutive reductions in softmax, LayerNorm forward and cross-entropy loss/grad, in **both** CUDA and SYCL (§3) | Intermittent silently wrong rows (possible `NaN`) in attention, LayerNorm and the loss during GPU training | CUDA by inspection; the identical SYCL kernels **verified** on a CPU SYCL device; **fixed in code October 17, 2026** in both backends | [TD-384](../../guides/TECHNICAL_DEBT.md#td-384-shared-memory-race-in-gpu-softmax-layernorm-and-cross-entropy-kernels-cuda-and-sycl) |
| `matrix_sum_gpu` allocates and frees device memory at every recursion level and synchronizes; the loss calls it every step (§2) | A CPU–GPU stall plus allocator traffic per training step | By inspection | [TD-385](../../guides/TECHNICAL_DEBT.md#td-385-matrix_sum_gpu-allocates-and-synchronizes-on-every-call) |
| Cross-entropy doesn't check target IDs; `float4` kernels require 16-byte-aligned pointers; no dimension checks anywhere; only `cudaGetLastError()` after launches (§1–2) | Out-of-bounds reads or faults on bad input, reported late and far from the cause | By inspection | [TD-386](../../guides/TECHNICAL_DEBT.md#td-386-gpu-kernel-input-checks-and-dead-batch-functions) |
| `matrix_batch_add_gpu`, `matrix_batch_multiply_gpu` and `matrix_add_scalar_gpu` have no callers (§4) | Unexercised code to maintain in two backends | By inspection | [TD-386](../../guides/TECHNICAL_DEBT.md#td-386-gpu-kernel-input-checks-and-dead-batch-functions) |
| No kernel has run on a CUDA device in any environment used so far | All of the above unverified on hardware | — | — (needs a CUDA host) |
| GELU constants duplicated; `matrix_sum_gpu` on empty input | Already tracked / fixed | — | TD-216; TD-088 (fixed) |

## Addendum (October 19, 2026): optimizer kernels (TD-468)

Two kernels were added for the device-resident optimizer ([GPUOptimizer.md](GPUOptimizer.md)), with identical CUDA and SYCL versions:
- `matrix_sum_squares_accumulate_gpu(data, n, out)`: a bounded grid with a grid-stride loop. Each work-group reduces its share and adds one partial to a device float with an atomic, without the temporary buffer or host wait of `matrix_sum_gpu()` (TD-385).
- `optimizer_update_gpu(w, g, m, v, n, update)`: the CPU `Optimizer`'s SGD, momentum, Adam and AdamW update, on gradients times `update.grad_scale`.

The SYCL versions were checked against the CPU `Optimizer` on the oneAPI CPU device (`gpuOptimizerTests`). The CUDA versions are compile-checked only.

### Embedding kernels (TD-547)

- `embedding_gather_add_pe_gpu(table, ids, pe, pe_rows, pos_offset, out, n, d)`: `out[i] = table[ids[i]] + pe[pos_offset + i]`, adding no
  positional row when `pe` is null or the position is past `pe_rows`.
- `embedding_scatter_add_gpu(dout, ids, grad_table, n, d)`: `grad_table[ids[i]] += dout[i]`, with atomic adds for repeated IDs.

Both were checked against `TokenEmbedding`'s host lookup and backward on the oneAPI CPU device (`gpuOptimizerTests`); the CUDA versions are
compile-checked only.

### Gated-residual kernels (TD-342)

- `gated_residual_add_gpu(residual, attn, gate, n)`: `residual += tanh(gate[0]) · attn`, with the gate read from device memory.
- `gated_residual_backward_gpu(d_res, attn, gate, d_attn, gate_grad, n)`: `d_attn = tanh(gate[0]) · d_res` and
  `gate_grad[0] += (1 − tanh²) · Σ d_res · attn`, reduced per work-group and added atomically.

Both were checked against the host formula on the oneAPI CPU device (`gpuOptimizerTests`); the CUDA versions are compile-checked only.

