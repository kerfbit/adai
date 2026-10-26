# `SpeculativeDecoding` — Source File Reference

- **File:** [`src/SpeculativeDecoding.hpp`](../../../../src/SpeculativeDecoding.hpp) (header-only, 471 lines)
- **Namespace:** global (`SpeculativeDecodingConfig`, `TokenProposal`, `SpeculativeDecoder`, `calculate_theoretical_speedup()`, `print_speedup_table()`)
- **Used by:** `ChatbotAPI::generate_response()` when `chatbot_api_server --draft-model <path>` loaded a draft model (`--speculative-candidates`, default 4); tests and examples
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-12`
- **Tests:** [`tests/speculativedecoding_test.cpp`](../../../../tests/speculativedecoding_test.cpp) (53 tests, `speculativedecodingTests`; all pass)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with scratch programs using the real classes: toy `TextGenerator`s with fixed
> draft/target distributions, and a tiny `EncoderDecoderModel` trained on an echo task.

---

## 1. What this file is

Draft-and-verify generation over two `TextGenerator`s: the draft proposes up to K tokens, the target checks them, and on the first rejection a
replacement is sampled.

| Member | Behaviour |
|---|---|
| `SpeculativeDecodingConfig` | `num_candidates` 4, `temperature` 1.0, `max_length` 100, `acceptance_threshold` 0 (**unused**), `use_greedy` false |
| `SpeculativeDecoder(draft, target, config)` | Throws on a null generator; RNG seeded from `std::random_device` |
| `generate_candidates(context)` (private) | Up to K draft calls to `get_next_token_probs()`, sampling each (temperature applied to the sample only), stopping at EOS; records the draft probability **before** temperature |
| `verify_candidates(context, proposals)` (private) | For each proposal **in turn**, one target `get_next_token_probs()` call; accept with probability `min(1, p_target/p_draft)`; on rejection, sample from `max(0, p_target − p_draft)` where `p_draft` is non-zero **only for the rejected token**, then stop |
| `generate(prompt)` / `generate_tokens(prompt_tokens, max_new)` | Loop: candidates → verify → append tokens up to and including the first rejection; stop on EOS or length |
| `get_acceptance_rate()`, `get_speedup()` (`K·α/(K+1)`), `reset_stats()`, `print_stats()` (stdout), `get/set_config()` | Statistics |
| `calculate_theoretical_speedup(K, α)`, `print_speedup_table()` | The same formula; stdout table |

### Production use

`ChatbotAPI` builds draft and target `model_fn`s around `EncoderDecoderModel::forward(input_tokens, decoder_tokens)`, sets `use_greedy` when the strategy
is `"greedy"` and calls `generate_tokens({}, max_length)`. `chatbot_api_server` constructs the draft model with **the same architecture as the target**
(`d_model`, layers, heads and `d_ff` from the same config; only the weights file differs).

---

## 2. Problems

### The CPU serving `model_fn` is off by one, so CPU chat generation is wrong (verified; HIGH, reaches beyond this file)

`EncoderDecoderModel::forward(input, targets)` is the **training** forward: it feeds `[BOS] + targets[:-1]` to the decoder, so its last logits row
predicts `targets.back()`, the last token *already in* the context. `TextGenerator` and this class read the last row as the prediction of the **next**
token. Every `ChatbotAPI` CPU generation path builds its `model_fn` this way: the plain path (`chatbot_api_server`'s default without a GPU), speculative
decoding, `--batched-inference` and `/chat/batch`.

Reproduced with a tiny model trained (CPU `forward()`, loss ≈ 0) to echo 12 five-token sequences:

| `model_fn` | Exact greedy generations | Example (want `15 31 24 30 4`) |
|---|---|---|
| `forward(input, generated)` (as `ChatbotAPI`) | **0 / 12** | `2 2 15 24 31 30 26 3` |
| Aligned (decoder input = `generated`) | **12 / 12** | `2 2 15 31 24 30 4 3` |
| Speculative decoding, `ChatbotAPI`'s `model_fn` | **0 / 12** | same as the first row |
| Speculative decoding, aligned `model_fn` | **12 / 12** | `2 15 31 24 30 4 3` |

So a correctly trained model answers wrongly on a CPU `chatbot_api_server`, while `generate_response()`'s cached path and the GPU path (which feed the
generated tokens as decoder input) are aligned. The fix belongs in `ChatbotAPI`/`EncoderDecoderModel` (a forward that takes decoder input directly, or
`model_fn` padding the context with a dummy final token); the RAG, pipeline and integrated engines don't use this `model_fn`.

### Not speculative decoding: wrong distribution, no fewer target passes (verified)

- **Output distribution isn't the target's.** The rejection residual subtracts the draft probability only for the rejected token instead of the whole
  draft distribution. With target P = (0.5, 0.3, 0.2) and draft Q = (0.2, 0.3, 0.5) over three tokens, 40,000 first tokens came out **(0.40, 0.40,
  0.20)**, not P. The header's "mathematically equivalent to standard sampling" doesn't hold.
- **Greedy mode isn't greedy.** Acceptance is still random (`p_t/p_d`), so the draft's argmax is accepted with that probability even when it isn't the
  target's: **41%** of 2,000 "greedy" first tokens were the draft's argmax instead of the target's.
- **Verification is sequential.** Each candidate gets its own target forward pass ("single forward pass" in the comments is false), and there's no bonus
  token when all K are accepted: 41 tokens took **41 target passes** plus the draft passes. It can only be slower than plain decoding, and
  `generate_tokens(…, 40)` returned 41 tokens (it overshoots by up to K − 1).
- **Same-size draft model.** `chatbot_api_server` builds the draft with the target's architecture, so even a correct implementation couldn't be faster.
- **Temperature mismatch (by inspection).** Candidates are sampled from the temperature-scaled draft distribution but accepted using unscaled draft and
  target probabilities, so with temperature ≠ 1 the acceptance ratio uses the wrong distributions.

### The "speedup" statistic is meaningless (verified)

`get_speedup()` = `K·α/(K+1)` is below 1 for every acceptance rate; it reported **0.6** at 80% acceptance and 0.8 at 100%. It ignores costs and isn't a
speedup formula (the standard expected-tokens-per-target-pass is `(1 − α^(K+1))/(1 − α)`). The tests assert this formula, and the replaced doc promised
"2–3× faster inference".

### Smaller items (by inspection)

`acceptance_threshold` is never read; `print_stats()`/`print_speedup_table()` write to `std::cout`; `generate()` uses `goto`; the RNG is unseeded
(TD-368's class); the class keeps statistics in plain `int`s with no locking (each request builds its own decoder, so this is safe today).

---

## 3. Tests

`speculativedecoding_test.cpp` (53): config and proposal defaults, the speedup helper's formula and monotonicity, construction checks, config get/set,
statistics before and after generation, `TextGenerator::get_next_token_probs()`, and generation not crashing, returning output and respecting
`max_new_tokens` with fixed toy models.

**Not covered:** the output distribution, greedy equivalence with the target, target-pass counts, alignment with a real model's `forward()`.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `ChatbotAPI`'s CPU `model_fn` (plain, speculative, batched, `/chat/batch`) uses the training forward, so the last row predicts the last context token (§2) | Correctly trained models answer wrongly on CPU servers (0/12 vs 12/12) | **Verified** | [TD-506](../../guides/TECHNICAL_DEBT.md#td-506-cpu-chat-generation-is-off-by-one-because-model_fn-uses-the-training-forward) |
| Wrong rejection residual; random acceptance in greedy mode; sequential verification with no bonus token; overshoot; same-size draft; temperature mismatch (§2) | Not target-distributed, not greedy, never faster | **Verified** (except temperature) | [TD-507](../../guides/TECHNICAL_DEBT.md#td-507-speculative-decoding-isnt-target-distributed-greedy-or-faster) |
| `K·α/(K+1)` "speedup" < 1 always; "2–3×" claims (§2) | Misleading statistics | **Verified** | [TD-508](../../guides/TECHNICAL_DEBT.md#td-508-speculative-decoding-speedup-statistic-and-hygiene) |
| Unused `acceptance_threshold`, console output, `goto`, unseeded RNG (§2) | Hygiene | By inspection | [TD-508](../../guides/TECHNICAL_DEBT.md#td-508-speculative-decoding-speedup-statistic-and-hygiene) |

---

## 5. Merge notes

This file replaces `docs/development/api/advanced/speculative-decoding.md` (510 lines). Where they disagree, this file wins: it promised "2–3× faster
inference", "mathematically equivalent" sampling, "target model verifies ALL K in parallel (one forward pass)" and "O(N/K) forward passes", none of which
holds (§2). Kept: the configuration fields and the draft/verify loop structure. Dropped: the benchmarks and tuning guidance, which describe the standard
algorithm rather than this implementation.

## Addendum (October 26, 2026): aligned model callbacks (TD-506)

`ChatbotAPI`'s draft and target callbacks call `EncoderDecoderModel::next_token_logits()`, not the training `forward()`. That method
puts a BOS in front of the empty first context. With the trained model as its own draft, speculative greedy decoding now reproduces the
echo task exactly (`ChatbotAPIEchoTest.SpeculativeDecodingEchoesExactly`). TD-507's algorithm issues are unchanged.

## Addendum (October 26, 2026): standard speculative sampling (TD-507, TD-508)

`SpeculativeDecoder` (2.0.0) now implements the published algorithm, so §2's findings and the "Not covered" list describe the old
version. Each round goes like this:
1. The draft samples up to K tokens from softmax(logits / T).
2. One target pass returns K + 1 rows (`TextGenerator::get_logits()`; `ChatbotAPI` asks `next_token_logits(…, K + 1)`).
3. Each proposal is accepted with probability min(1, p/q).
4. The first rejection is replaced by a sample from norm(max(0, p − q)), using the whole draft distribution.
5. When all K are accepted, a bonus token comes from the next target row.

Greedy mode (or T ≤ 0) accepts only the target's argmax, so its output equals the target's greedy output. A target that returns only its
last row is called once per needed prefix instead.

Other changes:
- Output stops at `max_new_tokens`, and K < 1 throws.
- `get_tokens_per_target_pass()` replaces `get_speedup()`, and `expected_tokens_per_target_pass(K, α)` = (1 − α^(K+1))/(1 − α)
  replaces `calculate_theoretical_speedup()`.
- `acceptance_threshold` and `print_speedup_table()` are removed, `log_stats()` uses `Logger`, and the constructor takes a seed.
- `chatbot_api_server` builds the draft with its checkpoint's own architecture (`read_checkpoint_architecture()`).

Old versus new, with the same mocks:

| Check | Old | New |
|---|---|---|
| First token, target (0.5, 0.3, 0.2) | (0.391, 0.411, 0.197) | within 0.015 of target |
| First token at T = 0.5 (0.658, 0.237, 0.105) | (0.396, 0.341, 0.264) | within 0.015 |
| Greedy equals target greedy | 0 of 10 | 10 of 10 |
| Target passes for 20 tokens, identical models, K = 4 | 20 | 4 |

Tests: `SpeculativeDecodingTD507Test.*`, `SpeculativeDecodingTD508Test.SameSeedSameOutput`, `ExpectedTokensPerTargetPassTest.*`.
