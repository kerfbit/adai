# `Predictor` — Source File Reference

- **Files:** [`src/Predictor.hpp`](../../../../src/Predictor.hpp), [`src/Predictor.cpp`](../../../../src/Predictor.cpp) (134 lines together)
- **Namespace:** global
- **Built into:** `adai_lejepa`; owned by `LeJEPAEncoder` only (see [LeJEPAEncoder.md](LeJEPAEncoder.md))
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-09-15` (both files)
- **Tests:** [`tests/predictor_test.cpp`](../../../../tests/predictor_test.cpp) (13 tests)
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
| No save/load: every resumed LeJEPA pass starts with a random predictor (0.0017 → 2.05 predictor loss after reload) | [TD-427](../../guides/TECHNICAL_DEBT.md#td-427-lejepa-predictor-is-never-saved-so-each-pass-restarts-it) |
| `train_step()`'s two `update_weights()` calls step the shared optimizer twice per sample | [TD-428](../../guides/TECHNICAL_DEBT.md#td-428-lejepa-two-update-workaround-is-obsolete-after-td-195) |
| Shared-optimizer stepping from layer `update_weights()`; duplicate registration | TD-297, TD-469 |
| Unseeded initialization (via `FeedForward`) | TD-368 |

---

## 3. Tests

`predictor_test.cpp` (13 tests): dimension validation, accessors, shape checks, forward shape (batch and single row), determinism, backward shape,
a finite-difference gradient check, updates changing the output, `zero_grad()`, training through a registered optimizer, distinct instances.
