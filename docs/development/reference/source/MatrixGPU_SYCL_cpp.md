# `gpu/sycl/MatrixGPU_SYCL.cpp` — Source File Reference

- **File:** [`src/gpu/sycl/MatrixGPU_SYCL.cpp`](../../../../src/gpu/sycl/MatrixGPU_SYCL.cpp) (894 lines, SYCL/oneMKL)
- **Namespace:** `adai::gpu`
- **Built into:** `adai_gpu` in SYCL builds (`ENABLE_SYCL`, `icpx`). **This is the code ai-machine's GPU training runs**
- **API:** declared in `sycl/MatrixGPU_SYCL.hpp` (the same functions as the CUDA `MatrixGPU.hpp`; see [MatrixGPU.md](MatrixGPU.md)); queue and memory from [GPUUtils_SYCL.md](GPUUtils_SYCL.md)
- **CUDA counterpart:** [MatrixGPU_cu.md](MatrixGPU_cu.md) (ported from this file)
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.10.1`, `@adai-reviewed: 2026-10-17` (TD-384 barriers, TD-393 reduction)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. **Every kernel in this file was run** in a scratch harness: a copy of this
> `.cpp`, linked against a copy of `GPUUtils_SYCL.hpp` patched to also accept the oneAPI **CPU**
> device (Intel i5-8350U, OpenCL), and compared with plain C++ references (and finite differences
> for LayerNorm backward). The production `GPUManager` only accepts GPUs, so results on a real Intel
> GPU can differ where noted.

---

## 1. What this file is

The SYCL implementations of the GPU matrix API: element-wise ops (`sycl::vec<float,4>` + scalar
tail), tiled transpose, oneMKL GEMM, activations, a sub-group sum reduction, batch helpers, and the
training kernels (softmax and its backward, GELU backward, LayerNorm forward/backward, bias, row
sums, masked fill, cross-entropy loss/grad), plus diagnostics reductions and transfer helpers. Every
command goes on the single in-order queue; per CLAUDE.md, intermediate kernels don't `.wait()`.

---

## 2. Verification results

| Kernel | Result on the CPU device | Notes |
|---|---|---|
| `matrix_add_gpu`, `matrix_multiply_elementwise_gpu`, `matrix_multiply_scalar_gpu` (n = 1003, exercising the `vec4` + tail split) | **Exact** | |
| `matrix_transpose_gpu` (45 × 70, not a multiple of the 32 tile) | **Exact** | Uses a 32 × 32 = 1,024-item work-group; fine on Arc (max 1,024), would fail on a device with a smaller limit |
| `matrix_multiply_gpu` (37 × 50 × 29, oneMKL) | ≤ 1.3×10⁻⁶ | The row-major ↔ column-major swap is correct |
| `matrix_apply_activation_gpu` (GELU) | ≤ 7×10⁻⁸ | |
| `matrix_softmax_backward_gpu`, `matrix_gelu_backward_gpu`, `matrix_add_row_bias_gpu`, `matrix_sum_rows_gpu`, `matrix_row_entropy_gpu` | Correct | |
| **`matrix_sum_gpu`** | **Wrong**: 256 ones → **64**, 1,000 → 256, 100,000 → 8,192, 1,000,000 → 65,536 | §3 |
| **`matrix_count_below_threshold_gpu`** | **Wrong**: 427 instead of 1,667 | Same reduction as `matrix_sum_gpu` |
| **`matrix_softmax_rows_gpu`** | **Wrong** rows (max relative error 2.5×10⁻²) | §4 (TD-384) |
| **`matrix_layer_norm_fwd_gpu`** | **Wrong** (max relative error 0.32) | §4 |
| `matrix_layer_norm_bwd_gpu` | Correct when given correct forward outputs (6.6×10⁻⁵ vs. finite differences); wrong downstream of the corrupted forward | The TD-061 fix is confirmed numerically |
| **`matrix_cross_entropy_loss_gpu` / `_grad_gpu`** | **Wrong** (4×10⁻³ / 2×10⁻³) | §4 |

Results were identical across five runs.

---

## 3. The sub-group sum reduction drops partials when the sub-group size is 8 (verified)

`workgroup_reduce_sum()` reduces within each sub-group (`reduce_over_group`), writes one partial per
sub-group, and then has **only the first sub-group** combine the partials:

```cpp
val = (local_id < num_subgroups) ? subgroup_sums[local_id] : 0.0f;
if (sg_id == 0) val = subgroup_reduce_sum(sg, val);
```

With a 256-item work-group there are `256 / S` partials but only `S` lanes in sub-group 0, so it's
correct only when `S² ≥ 256`, that is `S ≥ 16`. No `reqd_sub_group_size` is specified, so the compiler
chooses `S`. The CPU device chose **`S = 8`**, so 32 partials were produced and 8 were summed, which
is ¼ of the true sum per level; the recursive levels compound it (1,000,000 → 65,536).
`matrix_count_below_threshold_gpu` (FeedForward saturation) has the same reduction, and so does the
final sum in `matrix_cross_entropy_loss_gpu` once `seq_len > 64`.

**Exposure:** ai-machine's Arc B60 (Xe2/Battlemage) supports only sub-group sizes 16 and 32, so this
is probably latent there, but it isn't guaranteed by the code. Intel Arc A-series (Alchemist) and
other Xe-LP/HPG parts support `S = 8`, and the oneAPI CPU device uses it, so the same build would
compute wrong losses and sums on them. The CUDA version (`warp = 32`, 8 warps) is unaffected.

**Fixed in code October 17, 2026:** in sub-group 0, each lane now adds up every
`num_subgroups / S`-th partial before the final `subgroup_reduce_sum`, so all `256 / S` partials
count for any `S`. Verified on the CPU device (`S = 8`):
- the new `MatrixGPUTD003Test.SumIsExactAcrossWorkGroupBoundaries` (1 to 100,000 ones) and
  `CountBelowThresholdIsExactForFullWorkGroup` pass;
- the existing `CountBelowThresholdLargeMatrixMultiBlock` now passes too (it failed before the fix).

---

## 4. TD-384's shared-memory race reproduces (verified)

The four kernels that read `scratch[0]` (the row max or mean) and then reuse `scratch` for a second
reduction without a barrier (TD-384, tagged in this file) produced wrong results on every run. With
one `item.barrier(local_space)` added after each of those reads in the scratch copy, **all of them
passed**:
- softmax: 2.2×10⁻⁸;
- LayerNorm forward: 2×10⁻⁷;
- LayerNorm backward: 6.6×10⁻⁵ vs. finite differences;
- cross-entropy loss and grad: 1.3×10⁻⁸.

On the CPU device the failure is **deterministic**: the runtime runs a work-group's sub-groups one
after another between barriers, so sub-group 0 always overwrites `scratch[0]` before the others
read it. On a GPU, sub-groups run concurrently, so it's timing-dependent, but the race is no longer
theoretical. This corrects the earlier claim (in TD-384 and [MatrixGPU_cu.md](MatrixGPU_cu.md)) that
a CPU SYCL device can't reproduce it.

**Fixed in code October 17, 2026:** the barrier is now in the repository version of all four
kernels, and in the CUDA kernels too. New regression tests (`SoftmaxWideRowsMatchHost`,
`LayerNormWideRowsMatchHost`) fail on every run without the barriers and pass with them.

---

## 5. Other observations (by inspection)

- **Per-call allocation and waits.** `matrix_sum_gpu` and `matrix_count_below_threshold_gpu`
  allocate a `GPUMemory` per level, `.wait()` on the submit, and download (TD-385's SYCL half).
  `count_below_threshold` has no `size <= 0` guard (and `GPUMemory(0)` throws on SYCL, TD-392).
- **Batch helpers** (`matrix_batch_add_gpu`, `matrix_batch_multiply_gpu`, no callers, TD-386) do
  `q.memcpy` from **host** pointer arrays without waiting. Unlike CUDA's pageable-copy staging,
  SYCL doesn't guarantee the copy has read the host buffer before returning, so a caller's temporary
  pointer array could be read after it's freed. They also free their staging arrays with a
  `host_task` (TD-390's pattern).
- **Unchecked inputs.** Cross-entropy target IDs aren't range-checked; `vec4` paths assume 16-byte
  alignment (true for USM base pointers) (TD-386).
- GELU constants are duplicated (TD-216).

---

## 6. Tests

The repository's GPU tests (`gpuutilsTests`, `matrixgpuTd003Tests`) skip in `build/sycl-gputest`
without a GPU, and `GPUManager` won't use the CPU device. So **none of these kernels has ever been
run by the test suite**. The scratch harness above shows that a CPU-device test mode would have
caught both bugs.

---

## 7. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `workgroup_reduce_sum()` assumes sub-group size ≥ 16; at 8 it drops ¾ of the partials per level (§3) | Wrong sums, saturation fractions and (for `seq_len > 64`) losses on Alchemist-class GPUs and CPU devices; probably latent on ai-machine's Xe2 | **Verified** (CPU device, `S = 8`); **fixed in code October 17, 2026** | [TD-393](../../guides/TECHNICAL_DEBT.md#td-393-sycl-sub-group-reduction-is-wrong-when-the-sub-group-size-is-8) |
| TD-384 race reproduces deterministically on the CPU device and disappears with the barrier fix (§4) | Confirms TD-384; its "can't reproduce on CPU" note is wrong | **Verified**; **fixed in code October 17, 2026** | [TD-384](../../guides/TECHNICAL_DEBT.md#td-384-shared-memory-race-in-gpu-softmax-layernorm-and-cross-entropy-kernels-cuda-and-sycl) |
| No way to run the GPU kernel tests without a GPU (§6) | Both bugs above went undetected | **Verified** (harness) | [TD-394](../../guides/TECHNICAL_DEBT.md#td-394-gpu-kernel-tests-cant-run-without-a-gpu) |
| Batch helpers read host pointer arrays asynchronously; per-call allocation/waits in reductions; unchecked inputs (§5) | Latent use-after-free in dead code; stalls; out-of-bounds on bad input | By inspection (overlaps TD-385, TD-386, TD-390) | TD-385, TD-386 (amended), TD-390 |

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

## Addendum (October 26, 2026): cached attention reads the KV cache in place (TD-388)

`GPUKVCache` exposes its populated prefix as a view (`keys_data()`, `values_data()`, `leading_dimension()`) instead of copying it on
every `get_keys()`/`get_values()`. The new `matrix_gemm_strided_gpu()` is a row-major GEMM with transpose flags, alpha and leading
dimensions (`cublasSgemm` / oneMKL `gemm`). The cached self- and cross-attention paths use it to compute each head straight from the
cache and to write each head's output into its own columns, with no prefix copies, per-row slices or scatter. A 100-token decode on
the oneAPI CPU device went from 6.6 s to 1.4 s. Tests: `MatrixGPUTD388Test.StridedGemmMatchesHostReference`,
`EncoderDecoderGPUTest.CachedGpuDecodeMatchesCpuOverALongGeneration`.
