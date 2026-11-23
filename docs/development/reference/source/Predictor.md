# `Predictor` — Source File Reference

- **Files:** [`src/Predictor.hpp`](../../../../src/Predictor.hpp), [`src/Predictor.cpp`](../../../../src/Predictor.cpp) (248 lines together)
- **Namespace:** global
- **Built into:** `adai_lejepa`; owned by `LeJEPAEncoder` only (see [LeJEPAEncoder.md](LeJEPAEncoder.md))
- **Status tag:** `@adai-status: beta`; both `@adai-version: 0.3.0`, `@adai-reviewed: 2026-11-22` (the device path, TD-585)
- **Tests:** [`tests/predictor_test.cpp`](../../../../tests/predictor_test.cpp) (16 tests, plus 4 device tests in GPU builds: they pass on the oneAPI CPU device and skip without a device)
- **Design:** [lejepa_world_model_gated_injection_plan.md](../../../proposals/lejepa_world_model_gated_injection_plan.md), Component 2 (TD-176)
- **Last traced against the code:** 2026-10-17

> Code-traced reference.

---

## 1. What this file is

The LeJEPA world model's embedding-space predictor: during pretraining it maps the context view's encoder output to a prediction of the target
view's, and `LeJEPAEncoder::train_step()` scores it with MSE over the masked span. It's a thin wrapper around one [`FeedForward`](FeedForward.md)
(`d_model → hidden_dim → d_model`, GELU, Xavier initialization): no residual connection, no normalization, no vocabulary projection. It isn't used
at inference; only the frozen encoder's output feeds the chatbot.

| Member | Behaviour |
|---|---|
| `Predictor(d_model, hidden_dim)` | Rejects non-positive dimensions with `invalid_argument` before constructing the `FeedForward` |
| `forward(x)` / `backward(dY)` | Check `cols == d_model`, then delegate. `FeedForward`'s backward was verified against finite differences (FeedForward.md) and is tested here too |
| `update_weights()`, `zero_grad()` | Delegate. With an optimizer registered, `update_weights()` calls the **shared** optimizer's `step()`, which steps every registered parameter, not just the predictor's (TD-297) |
| `register_parameters_with_optimizer(opt)` | `FeedForward::set_optimizer()`, which registers its four matrices; no duplicate check (TD-469) |
| `get_d_model()`, `get_hidden_dim()` | Accessors |

`LeJEPAEncoder` constructs it with `hidden_dim = d_ff`, the full transformer feed-forward width, although the header suggests a "modest" width
suffices.

---

## 2. Gaps

No defects of its own beyond those already tracked:

| Item | Tracked as |
|---|---|
| No save/load: every resumed LeJEPA pass starts with a random predictor (0.0017 → 2.05 predictor loss after reload) | [TD-427](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-427-lejepa-predictor-is-never-saved-so-each-pass-restarts-it) |
| `train_step()`'s two `update_weights()` calls step the shared optimizer twice per sample | [TD-428](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-428-lejepa-two-update-workaround-is-obsolete-after-td-195) |
| Shared-optimizer stepping from layer `update_weights()`; duplicate registration | TD-297, TD-469 |
| Unseeded initialization (via `FeedForward`) | TD-368 |
| No GPU path: a LeJEPA pass ran the predictor on the host even in GPU training (fixed November 22, 2026; addendum below) | [TD-585](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-585-lejepas-predictor-has-no-gpu-path) |

---

## 3. Tests

`predictor_test.cpp` (16 tests; 13 here, the 3 `save()`/`load()` tests in the addendum below, and 4 `PredictorGpuTest` device tests in GPU builds, in the TD-585 addendum): dimension validation, accessors, shape checks, forward shape (batch and single row), determinism, backward shape,
a finite-difference gradient check, updates changing the output, `zero_grad()`, training through a registered optimizer, distinct instances.

---

## Addendum (October 20, 2026): `save()`/`load()` (TD-427)

`save(filename)` writes the wrapped `FeedForward` (`save_weights()` format: two int dims, then W1, W2, b1, b2). `load(filename)` returns
false when the file is missing. It throws, before reading any weights, when the size isn't what this shape writes (a truncated file,
TD-369) or the dims differ, leaving the weights unchanged. `LeJEPAEncoder` saves and loads it with the encoder. Tests:
`SaveLoadRoundTripsTheOutput`, `LoadOfAMissingFileReturnsFalse`,
`LoadRejectsTruncatedAndMismatchedFilesWithoutChangingWeights`.

## Addendum (November 22, 2026): the device path (TD-585)

GPU builds add `gpu_forward()`, `gpu_backward()`, `gpu_upload_weights()`, `gpu_download_weights()`, `gpu_zero_grads()` and
`gpu_collect_parameters()`, each delegating to the wrapped `FeedForward`'s device path (the same one every `EncoderBlock` uses in GPU
training). The device weights are uploaded on first use (`gpu_forward()`, `gpu_zero_grads()` or `gpu_collect_parameters()`, since
`FeedForward` reports no parameters before its device state exists) and again after `load()`, which marks them stale. While training on
the device they are the live copy; `gpu_download_weights()` brings them back to the host before `save()`. `gpu_forward()` checks the
input width; `gpu_backward()` throws `std::logic_error` before any `gpu_forward()` (where `FeedForward::gpu_backward()` would
dereference a null state, TD-370) and `std::invalid_argument` unless the gradient has the last forward's shape. `gpu_collect_parameters()`
reports W1, W2, b1 and b2 keyed by the host matrices `register_parameters_with_optimizer()` registers, so a `GPUOptimizer` steps all
four on the device with no host groups. LeJEPA doesn't call any of this yet; TD-588 does.

Tests (`PredictorGpuTest`, GPU builds): forward and backward match the host path; one Adam step from the device gradients gives the
same weights as one from the host gradients (0 host groups); weights loaded after a first device use reach the device; bad shapes and a
backward before any forward throw.
