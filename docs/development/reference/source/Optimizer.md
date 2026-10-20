# `Optimizer` — Source File Reference

- **Files:** [`src/Optimizer.hpp`](../../../../src/Optimizer.hpp), [`src/Optimizer.cpp`](../../../../src/Optimizer.cpp) (838 lines together)
- **Namespace:** global (`Optimizer`, `OptimizerType`, `ParameterGroup`)
- **Built into:** `adai_core`
- **Used by:** `ChatbotTrainer` (the chatbot's AdamW, both CPU and GPU training), `run_lejepa_training_pass()` (Adam for the world model), every layer's `set_optimizer()`/`register_parameters()`/`update_weights()`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.2.0`, `@adai-reviewed: 2026-10-19` (both files)
- **Tests:** [`tests/optimizer_test.cpp`](../../../../tests/optimizer_test.cpp) (51 tests)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linked against the real class
> (`-O3 -march=native`, 8 threads).

---

## 1. What this file is

A first-order optimizer over **parameter groups**: pairs of raw pointers to a layer's weight and gradient matrices, plus per-group
state (`momentum`, `velocity`, `step`).

| Type | Update (per element) |
|---|---|
| `SGD` | `g += wd·w` (if `wd > 0`); `w −= lr·g` |
| `SGD_MOMENTUM` | `g += wd·w`; `m = β·m + g`; `w −= lr·m` |
| `ADAM` | `g += wd·w` (coupled L2); `m = β₁m + (1−β₁)g`; `v = β₂v + (1−β₂)g²`; `w −= lr·m̂/(√v̂ + ε)` with per-group bias correction |
| `ADAMW` | as Adam without coupling, then `w −= lr·wd·w` (decoupled decay) |

`step()` skips a group whose gradient is **exactly all zero** when `wd == 0` (so a group with no signal this step isn't nudged
by stale momentum; added for LeJEPA's predictor, TD-190). Clipping: `clip_gradients()` (configured `max_grad_norm`),
`clip_gradients(max)`, and `clip_gradients(max, precomputed_norm)` scale every group by `max / (norm + 1e-6)` when the global L2 norm
exceeds `max`. `scale_gradients(factor)` (added October 17, 2026 for TD-272) multiplies every gradient by `factor`; `ChatbotTrainer` uses it to turn a short accumulation window's sum into a mean. `get_gradient_norm()`/`get_weight_norm()` (the latter for TD-013's update ratio), `zero_grad()`, `reset_state()`,
`total_parameters()`, `get_optimizer_name()`. `save_state(ostream)`/`load_state(istream)` (TD-269, October 18–19, 2026) persist the
moments and step counts; see the addendum below.

### How production uses it

`ChatbotTrainer` builds one `Optimizer(config.optimizer_type = ADAMW, lr)`, sets `weight_decay` (`WEIGHT_DECAY=0.01` in
`config.trainer.conf`), `max_grad_norm`, and the betas, and the model registers every layer's matrices. Per optimizer step (every
`GRADIENT_ACCUMULATION_STEPS` samples): `get_gradient_norm()`, a NaN/Inf check that **skips the update** (so a non-finite gradient
never reaches Adam's state), clipping (fixed or adaptive, TD-017), then `step()`. The world-model pass uses a separate `ADAM`
optimizer with no weight decay. Since TD-269 (October 19, 2026), `ChatbotTrainer` saves the optimizer's state in each
checkpoint's `<checkpoint>.optimizer` file and the next pass loads it, so AdamW's moments and step counts continue across
incremental passes. They used to restart from zero every pass. The world-model pass still starts its optimizer fresh.

---

## 2. Problems

### Every optimizer step costs ~3× what it should, and GPU training pays it on the CPU (verified, plus an estimate)

The update loops index each element through the bounds-checked, out-of-line `Matrix::operator()` (TD-435): an AdamW step touches
weight, gradient, momentum and velocity, about six accessor calls per parameter. Measured on 5.24M parameters (20 matrices of
512×512), **`Optimizer::step()` took 45.0 ms against 14.2 ms** for the same AdamW update written over raw row pointers (3.2×);
`get_gradient_norm()` adds another 6.4 ms.

This matters most for **GPU training**, because the optimizer is CPU-only: each optimizer step in `ChatbotTrainer`'s GPU path calls
`gpu_download_grads()` (every gradient to the host), runs `Optimizer::step()` on the CPU, then `gpu_sync_weights()` (every weight back
to the device). For the fallback architecture in `config.trainer.conf` (d_model 768, d_ff 3072, 8 encoder + 24 decoder layers ≈
283M parameters before embeddings; MNS's record may differ), that's an estimated **~2.4 s of CPU AdamW** per optimizer step at the
measured rate (~0.8 s with raw pointers), plus about 2.3 GB of host↔device copies, and the SYCL build compiles this code without SIMD
flags (TD-436). That's once per 32 samples with `GRADIENT_ACCUMULATION_STEPS=32`, but it's work the GPU never needed to hand back. A
GPU-side optimizer (keeping weights, gradients and Adam state resident) would remove both the copies and the CPU step.

### Weight decay applies to everything (by inspection)

`weight_decay` is one global value applied to every registered group, including LayerNorm `gamma`/`beta`, all biases and the token
embeddings, which are conventionally excluded from decay. With AdamW, `lr = 1e-4` and `wd = 0.01`, each step shrinks every parameter
by a factor of `1 − 1e-6` before the gradient pulls back, so norms' gains drift toward 0 unless gradients counter it. Its effect is small
per step, but there's no way to exclude groups.

### Smaller items (by inspection)

- **`amsgrad` is declared and never used.**
- **Duplicate registration isn't detected.** `add_parameter_group()` doesn't check whether a matrix is already registered, so
  registering twice (as TD-297's layer paths can) updates it twice per `step()`.
- **Raw pointers.** Groups point into layer members; destroying or moving a layer after registration leaves dangling pointers.
  Production keeps layers in stable `unique_ptr`s.
- **Float accumulation** for the gradient/weight norms over hundreds of millions of elements (as `Matrix::sum()`, TD-435).
- **`get_weight_norm()`**'s debug counter is a function-static without synchronization (harmless).

---

## 3. Tests

`optimizer_test.cpp` (48 tests): constructors and setters, group registration and its error cases, `zero_grad()`, gradient norm,
clipping (below/above threshold, custom norm, disabled), each algorithm's first and repeated steps, momentum accumulation, Adam's
bias correction, coupled vs decoupled decay, the zero-gradient skip (including AdamW still applying decay), `reset_state()`, a training
loop with LR scheduling, and edge cases (tiny gradients, large LR, empty optimizer, large or many groups).

Added October 17–19, 2026: `scale_gradients()` (TD-272), and `save_state()`/`load_state()`. The state tests check a round trip
that continues stepping identically, and that a shape mismatch or a truncated stream leaves the optimizer untouched (TD-269). A type mismatch is rejected by the same code path but not tested separately.

**Not covered:** `get_weight_norm()`, the precomputed-norm clipping overload, duplicate registration, performance.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Bounds-checked accessor in every update loop (3.2× slower); CPU-only optimizer forces GPU training to round-trip all gradients and weights per step (§2) | Seconds of CPU work and GBs of copies per optimizer step at the configured model size | **Verified** (timing); estimate for production size | [TD-468](../../guides/TECHNICAL_DEBT.md#td-468-gpu-training-round-trips-every-gradient-and-weight-through-a-3-slow-cpu-optimizer) |
| One global weight decay, applied to norms, biases and embeddings (§2) | Slow drift of norm gains under AdamW; no per-group control | By inspection | [TD-469](../../guides/TECHNICAL_DEBT.md#td-469-optimizer-global-weight-decay-unused-amsgrad-and-duplicate-registration) |
| Unused `amsgrad`; no duplicate-registration check; raw pointers; float norm accumulation (§2) | Hygiene; double updates if a layer registers twice | By inspection | [TD-469](../../guides/TECHNICAL_DEBT.md#td-469-optimizer-global-weight-decay-unused-amsgrad-and-duplicate-registration) |
| Shared-optimizer stepping; accessor cost | Already tracked | — | TD-297, TD-435 |
| Optimizer state not persisted across passes | Every pass restarted AdamW from zero moments | **Fixed October 19, 2026** (`save_state()`/`load_state()`) | [TD-269](../../guides/TECHNICAL_DEBT.md#td-269-optimizer-state-and-lr-schedule-restart-every-incremental-training-pass) |

---

## 5. Merge notes

This file replaces `docs/development/api/core/optimizer.md` (1,258 lines). Kept: the four algorithms' formulas, clipping and
norm semantics, and the `ChatbotTrainer` integration. Dropped: generic tutorial material (choosing an optimizer, hyperparameter tips,
example training loops, references) and its "Future Enhancements" list, whose items (warmup, schedulers, state persistence) are tracked
elsewhere (TD-269 for state). Added: the zero-gradient skip, the NaN guard in the caller, the GPU round trip and the measured cost.

## Addendum (October 19, 2026): `save_state()`/`load_state()` (TD-269)

`save_state(ostream)` writes the optimizer type, the global step and, for each parameter group in registration order, its shape, step count and
first and second moments.

`load_state(istream)` stages everything first and commits only if the type, the group count and every shape match the groups registered now. A
mismatch or a truncated file returns false and leaves the optimizer untouched.

`ChatbotTrainer` embeds this in its `<checkpoint>.optimizer` training-state file, so AdamW continues across incremental passes instead of
restarting with zero moments. With Adam/AdamW the moments are two more weight-sized tensors, so a checkpoint is about 3× larger; that was accepted
by the owner on October 18, 2026.
