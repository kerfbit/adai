# `RewardModel` — Source File Reference

- **File:** [`src/RewardModel.hpp`](../../../../src/RewardModel.hpp) (header-only, 474 lines)
- **Namespace:** global (`PreferencePair`, `RewardModel`)
- **Used by:** `RLHFTrainer` only (`RewardModel(2 × d_model, reward_model_layer_dims = {256, 128, 1})`), which has **no production entry point**; also tests and `examples/Phase5Examples.cpp`. `PPOOptimizer` stores a pointer but never reads it (TD-482)
- **Status tag:** `@adai-status: beta` ("TD-038 — now genuinely driven by RLHFTrainer; still not wired into a shipped binary"), `@adai-version: 0.7.1`, `@adai-reviewed: 2026-09-13`
- **Tests:** 5 in [`tests/phase5_test.cpp`](../../../../tests/phase5_test.cpp) (`RewardModelTest`), plus `RLHFTrainerTest.TrainRewardModelReducesLossOverEpochs` in `rlhftrainer_test.cpp`; all pass
- **Related:** [PPOOptimizer.md](PPOOptimizer.md) (the rest of the RLHF step)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real header.

---

## 1. What this file is

A small MLP that scores a (prompt, response) encoding, trained on human preference pairs with the Bradley–Terry loss
`−log σ(r_chosen − r_rejected)`.

| Member | Behaviour |
|---|---|
| `PreferencePair(prompt, chosen, rejected)` | Three `std::vector<float>` encodings |
| `RewardModel(input_dim, layer_dims)` | Throws unless `layer_dims` is non-empty and ends in 1. Weights uniform in ±√(2/fan_in) from the unseeded C `rand()`, biases 0 |
| `forward(x)` / `predict_reward(x)` | Checks `x.size() == input_dim`; ReLU hidden layers, linear output; caches activations (so it's non-const and not thread-safe) |
| `compute_loss(pair)` | Concatenates prompt + chosen and prompt + rejected (so `input_dim` must equal their combined width) and returns the loss (with a 1e-8 epsilon inside the log) |
| `train_on_batch(pairs, lr)` | For each pair: two forwards, `backward()` for each side (each **recomputes** its forward), gradients accumulated; then one plain-SGD step on the batch mean. Returns the mean loss. Throws on an empty batch |
| `backward(input, output_grad, …)` | Manual backpropagation through the cached activations |
| `save(path)` / `load(path)` | Binary: `input_dim`, layer count, layer dims, then per layer rows, cols, weights, bias size, biases. `load()` adopts whatever the file says |
| `get_input_dim()`, `get_layer_dims()` | Accessors |

**Verified correct:** the backward pass matches central finite differences (median relative error 1×10⁻⁴, maximum 2.3×10⁻³ over 38 weights with
non-negligible gradients). On a toy task (the chosen response has a positive first feature), a `{16, 1}` model went from loss 0.54 to 0.015 and ranked
all 200 pairs correctly.

---

## 2. Gaps

### Loading (verified)

- **`load()` checks no reads.** A file truncated to half its size loaded without error; the next `predict_reward()` threw `Matrix dimensions
  incompatible for multiplication: [1x5] * [6x5]` because the dimensions read from the cut-off part were garbage. Depending on where the cut falls the
  model can also load with silently wrong weights (the same class as TD-467/TD-298).
- **`load()` silently adopts the file's architecture.** A model built as `(6, {5, 4, 1})` that loaded a `(10, {3, 1})` file became a 10-input,
  two-layer model with no warning (as TD-432 for LoRA). Nothing checks that each layer's `rows`/`cols` agree with `layer_dims`.
- **A negative layer count** throws `std::length_error` (`vector::_M_default_append`) rather than a format error.

### Construction (verified, except where noted)

- **Non-positive layer widths are accepted:** `RewardModel(4, {0, 1})` constructed, and every reward is 0 and can never change. `input_dim` isn't checked
  either.
- **Initialization isn't He**, despite the comment: a uniform distribution on ±√(2/fan_in) has standard deviation √(2/fan_in)/√3 (measured 0.026 against
  He's 0.045 at fan-in 1,000). It's also unseeded global `rand()` (TD-368's class; same as `ValueFunction`, TD-482).

### By inspection

- **Twice the forward work:** `backward()` recomputes the forward pass that `train_on_batch()` has just run, so each pair costs four forwards. A
  `(1536, {512, 256, 128, 1})` model took 7.0 s per 100-pair batch here (the replaced doc said "~15 seconds per epoch"); inference was 1.2 ms.
- **No regularization, clipping or optimizer state:** plain SGD only; RLHF's usual reward normalization isn't provided.
- **Not thread-safe:** `forward()`/`predict_reward()` write the activation caches.

---

## 3. Tests

`phase5_test.cpp` (5): constructor validation, forward pass shape, preference loss, loss falling over batches, save/load round trip.
`rlhftrainer_test.cpp`: loss falls when `RLHFTrainer` trains the reward model.

**Not covered:** gradient correctness (verified here, not in a test), truncated or mismatched files, invalid layer widths.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `load()` unchecked: truncated files load, the file's architecture is adopted silently, negative counts give `length_error` (§2) | Garbage or failing reward model after a bad file | **Verified** | [TD-499](../../guides/TECHNICAL_DEBT.md#td-499-rewardmodel-unchecked-loading-invalid-widths-and-non-he-initialization) |
| Zero/negative widths accepted; initialization ⅓-variance of He; unseeded `rand()` (§2) | Dead models; slower start; irreproducible runs | **Verified** | [TD-499](../../guides/TECHNICAL_DEBT.md#td-499-rewardmodel-unchecked-loading-invalid-widths-and-non-he-initialization) |
| Forward recomputed in `backward()`; no regularization/clipping; not thread-safe (§2) | ~2× training cost; minor | By inspection | [TD-499](../../guides/TECHNICAL_DEBT.md#td-499-rewardmodel-unchecked-loading-invalid-widths-and-non-he-initialization) |
| No production entry point for RLHF | Context | — | RLHF use to be planned later, as for TD-481/TD-482 |

---

## 5. Merge notes

This file replaces `docs/development/api/advanced/reward-model.md` (423 lines). Where they disagree, this file wins: initialization is not He (§2);
the timing claim is replaced by the measurement above; the "Integration with PPO" example implies `PPOOptimizer` uses the reward model, which it doesn't
(TD-482). Kept: the loss, architecture, persistence format and the input-width rule (prompt + response). Dropped: generic RLHF pipeline, data
collection and best-practice material.
