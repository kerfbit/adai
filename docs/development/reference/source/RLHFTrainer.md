# `RLHFTrainer` — Source File Reference

- **Files:** [`src/RLHFTrainer.hpp`](../../../../src/RLHFTrainer.hpp) (159 lines), [`src/RLHFTrainer.cpp`](../../../../src/RLHFTrainer.cpp) (283 lines)
- **Namespace:** global (`PreferenceExample`, `RLHFConfig`, `RLHFStepResult`, `RLHFTrainer`)
- **Used by:** tests only. **No binary, flag or endpoint runs RLHF** (the owner will plan its use later; see TD-481/TD-482)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0` (hpp) / `0.1.1` (cpp), `@adai-reviewed: 2026-09-13` / `2026-09-16`
- **Tests:** [`tests/rlhftrainer_test.cpp`](../../../../tests/rlhftrainer_test.cpp) (4 tests, `rlhftrainerTests`; all pass)
- **Related:** [RewardModel.md](RewardModel.md), [PPOOptimizer.md](PPOOptimizer.md), [EncoderDecoderModel.md](EncoderDecoderModel.md)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real classes and the same tiny model the tests
> use (`EncoderDecoderModel(100, 32, 1, 1, 2, 64)`, a 12-sentence vocabulary).

---

## 1. What this file is

The integration that drives [`RewardModel`](RewardModel.md) and [`PPOOptimizer`](PPOOptimizer.md) against a live `EncoderDecoderModel` policy
(TD-038). It doesn't own the model; the constructor registers the model's parameters with its own `Optimizer` (Adam, `policy_learning_rate` 1e-5).

| Config (`RLHFConfig`) | Default |
|---|---|
| `ppo` | `PPOConfig{}` |
| `reward_model_layer_dims`, `reward_model_learning_rate` | `{256, 128, 1}`, 1e-4 (reward model input = 2 × `d_model`) |
| `policy_optimizer_type`, `policy_learning_rate` | Adam, 1e-5 |
| `max_response_length`, `generation_strategy`, `temperature` | 64, `"sampling"`, 1.0 |

| Member | Behaviour |
|---|---|
| `encode_to_vector(tokens)` (private) | Runs the policy's encoder (all-ones mask) and mean-pools to `d_model`; zeros for an empty sequence. Overwrites the encoder's activation caches, so it must not run between `forward()` and `backward_pass()` |
| `score_response(prompt, response)` (private) | `encode_to_vector(encode(prompt, true)) ++ encode_to_vector(encode(response, true))` → `RewardModel::predict_reward()` |
| `train_reward_model_on_batch(examples, epochs)` | Encodes each (prompt, chosen, rejected) once, then `RewardModel::train_on_batch()` per epoch; returns the last epoch's loss |
| `apply_policy_gradient(prompt, response, advantages)` | `set_training(true)`, `zero_grad()`, teacher-forced `forward(prompt, response)`, gradient `A_t · (softmax − one_hot(action_t))` per row, `backward_pass()`, `optimizer.step()`; returns mean `−A_t · log p(action_t)` |
| `run_iteration(prompts)` | Per prompt: (1) generate a response with `generate_response_with_strategy(prompt, max_len, strategy, temperature)`, skipping empty ones; (2) re-encode it with `encode(response, true)`; (3) score it; (4) for each response position, a value-function state = mean-pooled encoding of prompt + response so far; (5) old log-probs from a forward pass; (6) a `Trajectory` with the reward on the last token; (7) `PPOOptimizer::update()` (trains the value function only); (8) GAE advantages → `apply_policy_gradient()`. One optimizer step per prompt |

**Verified correct:** teacher forcing is aligned. `forward()` feeds `[BOS] + response[:-1]`, so logits row *t* predicts `response[t]`, matching the
gradient's `action = response_tokens[t]` (the CPU path; the GPU forward's TD-358 isn't used here).

---

## 2. Problems

As the class has no production entry point, these are LOW, to be fixed before RLHF is used (with TD-481/TD-482).

### Rollouts never explore (verified; TD-261)

`run_iteration()` omits `num_beams`, so the header default (4) forces beam search whatever `generation_strategy` says (TD-261). With the trainer's exact
call (`"sampling"`, temperature 1.0), six generations for the same prompt gave **1 distinct response**; with `num_beams = 1` they gave 6. Every rollout
for a prompt is the same deterministic beam-search output, so the policy gradient never sees alternatives and RLHF can only reinforce or suppress the
current mode. This also makes each rollout use the uncached beam path (TD-424's cost).

### The update is computed on a different sequence than the rollout (verified)

- **Different encoder input.** Generation encodes the prompt without special tokens, as the model was trained; the trainer's `forward()` calls (old
  log-probs and the update) use `encode(prompt, true)`, adding BOS/EOS (the TD-364 mismatch). For `"hello"` that is 3 encoder tokens instead of 1, and the
  response's log-probability changed from −69.2 to −68.2.
- **Actions that were never taken.** The response is decoded to text and re-encoded with special tokens: `response_tokens` started with BOS (id 2) and
  ended with EOS (id 3). Generation never emits BOS as an action, and a response cut off at `max_response_length` never emitted EOS, yet both get
  advantage-weighted gradient. With TD-361's doubled BOS, the first update row trains "predict BOS after BOS". Re-tokenizing decoded text can also differ
  from the sampled token IDs (normalization, TD-224).

The rollout should keep the generated token IDs and the generation-time encoder input, and train on exactly those.

### Missing RLHF safeguards (by inspection)

- **No KL penalty** against a frozen reference (SFT) policy, the standard guard against reward hacking and drift; with TD-482's ratio ≡ 1 there is no
  step-size control at all.
- **One optimizer step per prompt**, not per batch, so the update is as noisy as single-sample REINFORCE; rewards aren't normalized across the batch
  (TD-481 normalizes within each response instead).
- **Cost:** step (4) runs the encoder once per response token over a growing context (64 encoder passes for a 64-token response), on top of beam
  generation, scoring and two forwards.
- `set_training(true)` is left on after the update; the TD-341 gates aren't optimizer parameters, so they don't learn here either.

---

## 3. Tests

`rlhftrainer_test.cpp` (4): reward-model loss falls over epochs; `run_iteration()` changes the policy (retrying fresh random models when a cold model
generates nothing); a positive advantage raises and a negative one lowers a response's log-probability (checked through `apply_policy_gradient()`
directly); an empty prompt list is a no-op. The tests use `"greedy"` generation, so they never notice that `"sampling"` is overridden by beam search.

**Not covered:** rollout diversity, alignment between generated and trained tokens, the encoder-input mismatch.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Rollouts are deterministic beam search because `num_beams` is omitted (§2) | No exploration: 1 distinct response in 6 "sampling" rollouts | **Verified** (cause tracked as TD-261) | [TD-500](../../guides/TECHNICAL_DEBT.md#td-500-rlhf-rollouts-never-explore-and-updates-train-on-a-different-sequence) |
| Update uses BOS/EOS-wrapped prompt and re-encoded response with BOS/EOS actions never sampled (§2) | Gradient on a different sequence than the rollout | **Verified** | [TD-500](../../guides/TECHNICAL_DEBT.md#td-500-rlhf-rollouts-never-explore-and-updates-train-on-a-different-sequence) |
| No KL penalty or batch-level update; per-token encoder passes for value states; training mode left on (§2) | Unsafe, noisy and slow RLHF | By inspection | [TD-501](../../guides/TECHNICAL_DEBT.md#td-501-rlhftrainer-lacks-a-kl-penalty-batch-updates-and-efficient-value-states) |
| Per-response advantage normalization; inert PPO ratio; reward model gaps | Already tracked | — | TD-481, TD-482, TD-499 |

## Addendum (October 25, 2026): rollouts sample again (TD-261)

`run_iteration()` passes `num_beams = 1`. With TD-261's fix, `"sampling"` rollouts are random rather than deterministic beam search;
they were one distinct response in six. TD-500's re-encoding issue remains.
