# `PPOOptimizer` — Source File Reference

- **File:** [`src/PPOOptimizer.hpp`](../../../../src/PPOOptimizer.hpp) (header-only, 573 lines)
- **Namespace:** global (`Trajectory`, `PPOConfig`, `ValueFunction`, `PPOOptimizer`)
- **Used by:** `RLHFTrainer` only, which has **no production entry point** (no binary, flag or endpoint runs RLHF); tests and `examples/Phase5Examples.cpp`
- **Status tag:** `@adai-status: experimental` ("TD-038 — now driven for real by RLHFTrainer"), `@adai-version: 0.4.0`, `@adai-reviewed: 2026-09-13`
- **Tests:** in [`tests/phase5_test.cpp`](../../../../tests/phase5_test.cpp) (10 tests: `PPOOptimizerTest`, `ValueFunctionTest`); exercised end to end by `rlhftrainer_test.cpp`
- **Related:** [RewardModel](RewardModel.md) (scores rollouts), `RLHFTrainer` (the real policy update)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real header.

---

## 1. What this file is

The pieces of a PPO-style RLHF step, minus the part that changes the language model:

| Type | Behaviour |
|---|---|
| `Trajectory` | Per-token `states`, `actions`, `rewards`, `log_probs`, `values` |
| `PPOConfig` | `clip_epsilon 0.2`, `gamma 0.99`, `gae_lambda 0.95`, `value_coef 0.5`, `entropy_coef 0.01`, `num_epochs 4`, `batch_size 64`, `max_grad_norm 0.5`, `learning_rate 1e-5`, `kl_target 0.01` |
| `ValueFunction(input_dim, {256, 128, 1})` | A small ReLU MLP; `predict()`, `backward()`, `update(states, targets, lr)` (MSE, plain SGD). Its gradients were fixed in TD-034 and train on a toy regression (tested). Initialized with the unseeded C `rand()` |
| `PPOOptimizer::update(traj, current_policy_log_prob)` | GAE advantages and discounted returns; for `num_epochs` (minibatches in order, not shuffled): the clipped surrogate `−min(r·A, clip(r)·A)` with `r = exp(new − old)` from the callback, and a k1 approximate-KL early stop; **trains only the value function**; returns the mean surrogate loss |
| `compute_advantages(traj)`, `estimate_value(state)`, `get_config()`/`set_config()` | — |

### What actually updates the policy

`PPOOptimizer` never touches the language model: the surrogate loss is computed and returned, not differentiated. `RLHFTrainer` does the
real policy update itself, in `apply_policy_gradient()`, weighting each response token's log-probability gradient by the advantages from
`compute_advantages()`. It calls `update()` only to train the value function, with a callback that returns the rollout's **own** old
log-probabilities, so the ratio is exactly 1 for every call. That means no clipping, KL ≡ 0 (the early stop never fires), and a surrogate loss of
`−mean(A) = 0`: **verified**, the reported loss was 0.000000. In practice this is REINFORCE with a learned baseline, not PPO: none of PPO's
step-size control is applied to the policy.

---

## 2. Problems

### Advantages are normalized within each single response (verified)

`compute_gae()` normalizes the advantages of each trajectory (one response) to mean 0 and standard deviation 1. With the reward on the last
token (as `RLHFTrainer` assigns it) and value estimates of 0 (the value function's starting point):

| Rollout | Normalized advantages, tokens 1–5 |
|---|---|
| reward +1 | −1.370, −0.727, −0.043, +0.684, +1.457 |
| reward **+100** | **identical** to +1 |
| reward −1 | +1.370, +0.727, +0.043, −0.684, −1.457 |
| one-token response, +5 | **0.000** |

So the update **ignores the reward's magnitude**, **pushes down the early tokens of a well-rewarded response** (and pushes up those of a badly
rewarded one), and **learns nothing from one-token responses**. Per-batch normalization across many rollouts (or none, relying on the value
baseline) is the standard practice; per-trajectory normalization destroys the signal. As the value function learns, the values reshape this
pattern but the mean-zero, unit-scale constraint per response remains.

### PPO in name only (verified, see §1)

`update()` computes a clipped objective it never applies, the only caller makes the ratio identically 1, and `entropy_coef`, `value_coef` and
`max_grad_norm` (and the private `clip_gradients()`) are never used. The header and the old API doc describe "stable policy updates via the clipped
surrogate objective" and "KL monitoring to prevent policy collapse", none of which reaches the policy.

### Smaller items (by inspection)

- **`reward_model_` is stored but never read** (`RLHFTrainer` scores rollouts itself).
- **Minibatches aren't shuffled**, and `batch_size` only chunks a single trajectory.
- **Unseeded `rand()`** initialization (TD-368's class).
- **Single-step edge**: `compute_gae()` divides by `size()` (fine), but the 1e-8 epsilon makes every one-step advantage exactly 0, as above.

---

## 3. Tests

`phase5_test.cpp` (10 tests): construction, value estimation, a trajectory update, the null-callback check, the ratio reflecting a *live* policy
callback (a test-constructed one that differs from the old log-probs), the KL early stop triggering (likewise), config round trip, and the value
function training on a toy regression and rejecting bad batches.

**Not covered:** advantage scale and sign for realistic reward placements; the ratio-1 situation that `RLHFTrainer` always produces.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Per-trajectory advantage normalization: reward magnitude ignored, early tokens of good responses penalized, one-token responses get 0 (§2) | RLHF's real policy update learns the wrong direction for much of each response | **Verified** | [TD-481](../../guides/TECHNICAL_DEBT.md#td-481-ppo-advantages-are-normalized-within-each-response) |
| No policy update in `update()`; its only caller makes ratio ≡ 1 (no clipping, KL ≡ 0, loss ≡ 0); unused `entropy_coef`/`value_coef`/`max_grad_norm`/`clip_gradients()`/`reward_model_` (§1, §2) | "PPO" provides no step-size control; misleading API and diagnostics | **Verified** (loss 0) | [TD-482](../../guides/TECHNICAL_DEBT.md#td-482-ppooptimizer-never-updates-the-policy-and-its-ratio-is-always-1) |
| Unshuffled minibatches; unseeded `rand()` (§2) | Minor | By inspection | [TD-482](../../guides/TECHNICAL_DEBT.md#td-482-ppooptimizer-never-updates-the-policy-and-its-ratio-is-always-1) |
| No production entry point for RLHF | Context | By inspection | — (RLHF use to be planned later) |

---

## 5. Merge notes

This file replaces `docs/development/api/advanced/ppo-optimizer.md` (471 lines). Where they disagree, this file wins:
- **"Stable policy updates via clipped surrogate objective" / "KL monitoring".** `PPOOptimizer` doesn't update the policy; the actual update
  (`RLHFTrainer::apply_policy_gradient()`) is unclipped and the KL is always 0.
- **Gradient clipping (`max_grad_norm`) and entropy bonus.** Configurable but never applied.

Kept: the data structures, the GAE and surrogate formulas, the configuration fields. Dropped: the generic RLHF tutorial, hyperparameter guide and
best practices.
