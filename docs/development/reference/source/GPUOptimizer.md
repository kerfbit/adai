# `GPUOptimizer` — Source File Reference

- **Files:** [`src/GPUOptimizer.hpp`](../../../../src/GPUOptimizer.hpp) (87 lines), [`src/GPUOptimizer.cpp`](../../../../src/GPUOptimizer.cpp) (248 lines); kernels in [`src/gpu/MatrixGPU.cu`](../../../../src/gpu/MatrixGPU.cu) and [`src/gpu/sycl/MatrixGPU_SYCL.cpp`](../../../../src/gpu/sycl/MatrixGPU_SYCL.cpp); shared structs in [`src/gpu/GPUOptimizerKernels.hpp`](../../../../src/gpu/GPUOptimizerKernels.hpp), [`src/gpu/GPUParamRef.hpp`](../../../../src/gpu/GPUParamRef.hpp), [`src/gpu/GPUParamSync.hpp`](../../../../src/gpu/GPUParamSync.hpp)
- **Built into:** `adai_core`, only with `ADAI_ENABLE_GPU` (CUDA or SYCL)
- **Used by:** `ChatbotTrainer` (GPU builds), for the whole of `train()`
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-10-19`
- **Tests:** [`tests/gpu_optimizer_test.cpp`](../../../../tests/gpu_optimizer_test.cpp) (6, `gpuOptimizerTests`); skips without a GPU, and runs on the oneAPI CPU device in a SYCL build with `ADAI_SYCL_ALLOW_CPU_DEVICE=1` (TD-394)
- **Last traced against the code:** 2026-10-19

> Code-traced reference, written with the files (TD-468, October 19, 2026).

---

## 1. What this file is

The optimizer step of GPU training, run on the device. It exists because of the owner's rule that GPU training does all the training math on
the GPU and moves no data without a critical reason. Before it, every optimizer step:
- downloaded every gradient (`gpu_download_grads()`);
- ran the norm, NaN check, clip and AdamW on the CPU;
- uploaded every weight back (`gpu_sync_weights()`).

| Member | Behaviour |
|---|---|
| `GPUOptimizer(host, params)` | Binds each of the CPU `Optimizer`'s parameter groups to a device mirror by host-weight identity (`params` from `EncoderDecoderModel::gpu_collect_parameters()`, one `GPUParamRef` per mirrored weight: host `Matrix*`, device weights, device gradients, layer tag). It checks each mirror's size, allocates the device moments the rule needs, and uploads them from the host optimizer, so restored training state (TD-269) continues. Unmatched groups become **host groups** |
| `gradient_sq_norms()` | Zeroes a one-float-per-device-group buffer, runs `matrix_sum_squares_accumulate_gpu()` per group, and downloads that buffer: the only transfer of a step. Host groups are reduced on the host |
| `step(grad_scale)` | `global_step++`. Per group, the zero-gradient skip (`weight_decay == 0` and a squared norm of 0, as `Optimizer::step()` skips all-zero gradients), Adam/AdamW's own `step++` for bias correction, then `optimizer_update_gpu()` with the group's bias corrections and `grad_scale`. Host groups scale their host gradients and use the CPU `step_*` functions |
| `weight_norm()` | The same device reduction over the weights (TD-013's weight-update ratio, every 10 steps) |
| `layer_gradient_norm(layer)` | From the last squared norms, summed over groups tagged `layer` (encoder block *l* = *l*, decoder block *l* = `kDecoderLayerBase` + *l*) |
| `download_state()` | Device moments into the CPU optimizer's groups, before its state is saved |

Kernels (identical in both backends):
- **`matrix_sum_squares_accumulate_gpu(data, n, out)`:** a bounded grid (≤ 1024 work-groups of 256) with a grid-stride loop. Each work-group
  reduces its share and adds one partial into `*out` with a relaxed float atomic. There is no temporary buffer and no host wait, unlike
  `matrix_sum_gpu()` (TD-385).
- **`optimizer_update_gpu(w, g, m, v, n, update)`:** the CPU `Optimizer`'s SGD, SGD-momentum, Adam (coupled decay) and AdamW (decoupled
  decay) formulas, in the same order, on `g × grad_scale`.

### How `ChatbotTrainer` uses it

- **Binding:** `train()` calls `bind_gpu_optimizer()`, which runs `gpu_init_training()` (the host weights are authoritative between passes)
  and constructs the optimizer. `train_epoch()` binds one for itself when called directly, as the tests do.
- **Each step, `apply_update()`:**
  - grad norm = √(Σ squared norms) × the TD-272 window rescale, for the NaN check and adaptive clipping;
  - clip factor = `clip / (norm + 1e-6)` when the norm exceeds the clip, as `Optimizer::clip_gradients(max, norm)` computes it;
  - `step(rescale × clip factor)`.
- **Host copies:** `sync_host_from_device()` (weights via each layer's `gpu_download_weights()`, then `download_state()`) runs before
  `save_to()`, the best-model save, `save_training_state()` and the per-epoch quality-metrics `clone()`, and at the end of `train()`
  (also on failure, best effort).

---

## 2. Gaps

- **Host groups:** the token embeddings (TD-547) and the gated world-model/hippocampal sub-layers (TD-342) have no device mirror, so their
  updates still run on the host. The embedding ones are a full AdamW over `vocab × d_model` twice per step.
- **Zero-gradient skip:** it uses the squared norm, so a gradient whose every entry squares below `float` range counts as zero. The CPU checks
  each value exactly.
- **CUDA:** compile-checked only here (no CUDA device). SYCL was run on the oneAPI CPU device; neither has been timed on ai-machine.

---

## 3. Tests

`gpu_optimizer_test.cpp` (6):
- `DeviceStepMatchesCpuOptimizer` for each rule: three steps with random gradients and a 0.5 gradient scale match the CPU `Optimizer` on
  pre-scaled gradients (≤ 1e-5 relative), and the squared norm matches;
- `StateRoundTripsThroughTheCpuOptimizer`: two steps, `download_state()`, then a new `GPUOptimizer` taking a third step equals three CPU
  steps;
- `ModelStepMatchesTheDownloadAndCpuStepPath`: on a real model (36 device groups, 2 host groups), one device step equals the old
  download-and-CPU-step path, compared through the model's logits (≤ 1e-4).

The trainer suites run the whole path in a SYCL build on the CPU device: `chatbottrainerTests` (83), `encoderdecoderTests` (90) and
`incrementaltrainerTests` (58).
