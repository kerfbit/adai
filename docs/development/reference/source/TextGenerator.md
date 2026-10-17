# `TextGenerator` — Source File Reference

- **Files:** [`src/TextGenerator.hpp`](../../../../src/TextGenerator.hpp) (315 lines), [`src/TextGenerator.cpp`](../../../../src/TextGenerator.cpp) (640 lines)
- **Namespace:** global
- **Built into:** `adai_nlp`
- **Used by:** `EncoderDecoderModel` (its own `generator`, driven by `generate_response*()`), `ChatbotAPI`'s CPU paths (plain, `/chat/batch`, speculative drafts/targets), `BatchedInferenceEngine`, [`SpeculativeDecoder`](SpeculativeDecoding.md) (`get_next_token_probs()`), `RAGInference` (config type), `IntegratedInferenceEngine`/`PPOOptimizer` (headers)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10` (both files)
- **Tests:** [`tests/textgenerator_test.cpp`](../../../../tests/textgenerator_test.cpp) (36 tests, `textgeneratorTests`; all pass); see also [TEXTGENERATOR_TEST_COVERAGE.md](../../testing/coverage/TEXTGENERATOR_TEST_COVERAGE.md)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program (under AddressSanitizer) using the real class and toy
> `model_fn`s.

---

## 1. What this file is

Model-agnostic autoregressive decoding. Every strategy repeatedly calls a caller-supplied `ModelForwardFn` (`Matrix(const std::vector<int>&)`) on the
whole sequence so far and reads the **last row** of the returned logits as the next-token distribution.

> **Contract:** the last row must be the prediction of the token *after* the context. `ChatbotAPI`'s CPU `model_fn`s break it by wrapping the training
> `forward()`, whose last row predicts the last context token (TD-506). `EncoderDecoderModel`'s own strategy paths build aligned functions.

### `GenerationConfig`

`max_length` 100, `temperature` 1.0, `top_k` 0 (off), `top_p` 1.0 (off), `repetition_penalty` 1.0, `num_beams` 1, `length_penalty` true with
`length_penalty_alpha` 0.6, `early_stopping` true, `min_length` 0, and the four special-token IDs (defaults from [SpecialTokens.md](SpecialTokens.md)).

### Members

| Member | Behaviour |
|---|---|
| `TextGenerator(config, seed)` / `TextGenerator()` | `seed == 0` (and the default constructor) seed from `std::random_device` |
| `generate_greedy(fn, prompt)` | Starts from `prompt` or `[BOS]`; argmax each step (after the repetition penalty) |
| `generate_sampling(fn, prompt, T)` | Temperature 0 → greedy; else temperature, penalty, softmax, sample |
| `generate_top_k(fn, prompt, k)` / `generate_nucleus(fn, prompt, p)` | Temperature, then top-k (others set to −1e9) or top-p (smallest prefix of the sorted distribution reaching `p`), then penalty, then sample |
| `generate(fn, prompt)` | `num_beams > 1` → beam search; else temperature → top-k → top-p → penalty, then argmax at temperature 0 or sample |
| `generate_beam_search(fn, prompt, num_beams)` | 1 → greedy; else per step, the top `2·num_beams` log-probabilities of each live beam, keep the best `num_beams`; finished beams (stop token or length) are length-normalized with `((5 + len)/6)^α` and set aside; the best finished beam is returned |
| `generate_text(fn, tokenizer, prompt)`, `generate_batch(...)` | Encode the prompt (no special tokens), `generate()`, decode skipping special tokens; batch is a loop |
| `apply_repetition_penalty()` | CTRL-style, once per distinct token in the sequence (the TD-066 fix): positive logits ÷ penalty, negative × penalty |
| `set/get_config()`, `set_seed()`, `set_model_fn()`, `set_tokenizer()`, `get_tokenizer()`, `get_next_token_probs(context)` | Configuration; the stored `model_fn`/tokenizer and `get_next_token_probs()` exist for `SpeculativeDecoder` |
| `is_stop_token()` | EOS or PAD |

---

## 2. Problems

### Beam width 0 or negative crashes the process (verified)

`generate_beam_search()` builds `std::vector<BeamHypothesis> beams(num_beams)` and writes `beams[0]` without checking. Width 0 is a **segfault** (ASan:
SEGV in `vector::capacity()`); a negative width throws `std::length_error` uncaught (`terminate`). `ChatbotAPI` passes `BEAM_WIDTH` straight through for
`STRATEGY=beam`, and `ConfigLoader::validate()` (which rejects widths outside 1–16) only runs on `SIGHUP` reload (TD-241), so `BEAM_WIDTH=0` with the
beam strategy kills `chatbot_api_server` on its first chat request.

### Temperature 0 isn't greedy for top-k and nucleus (verified)

`apply_temperature()` returns the logits unchanged for `temperature <= 0`, which is right only where the caller then takes the argmax. `generate()` and
`generate_sampling()` do; `generate_top_k()` and `generate_nucleus()` sample at an effective temperature of 1. With temperature 0, 50 runs gave **33**
distinct nucleus outputs and **17** top-k outputs (and one from `generate()`). `STRATEGY=nucleus` is the shipped default, so setting `TEMPERATURE=0`
(which `validate()` allows) for deterministic answers doesn't work. Negative temperatures are treated the same way.

### `min_length` and `early_stopping` do nothing (verified / by inspection)

- **`min_length`** is never read: with `min_length = 4` and a model that prefers EOS, greedy and `generate()` both stopped after **one** token (the EOS),
  producing an empty response. The RLHF trainer has a comment about exactly these empty responses.
- **`early_stopping`**: both branches break only when no live beams remain (`if (early_stopping && beams.empty()) break; if (beams.empty()) break;`), so
  the flag has no effect, and beam search never stops early once the best finished beam can't be beaten; it runs until every beam finishes or
  `max_length`.

### NaN logits silently give an empty response (verified)

With all-NaN logits (e.g. a model loaded from a truncated `.lm_head`, TD-423), both sampling and greedy picked token **0 (PAD)**, which is a stop token,
so generation ends at once with no error. `std::discrete_distribution` over NaN weights is also undefined behaviour.

### Smaller items (by inspection, except the length check)

- **`max_length` counts the starting BOS and the prompt** (verified: `max_length = 6` gave 5 new tokens), while callers treat it as "tokens to
  generate".
- **Filter order:** the repetition penalty is applied after top-k/top-p, so a penalized token stays inside the nucleus it was selected for; the usual
  order (HuggingFace) applies the penalty first.
- **`top_p <= 0` disables filtering** instead of meaning "most likely token only".
- **Beam search calls the model once per beam per step**, with the uncached full-sequence `model_fn` (TD-424's cost multiplied by the beam count), and
  candidates from duplicate initial beams carry a −1e9 score.
- **No vocabulary masking** of IDs beyond the tokenizer's vocabulary (done by some callers only; TD-362).
- **Seed 0 means random**, so the seed 0 itself can't be chosen; the softmax is another copy of `Activation::softmax` (TD-216).

---

## 3. Tests

`textgenerator_test.cpp` (36): construction and config, the filters (temperature, top-k, top-p, repetition penalty including the TD-066 fix), each
strategy's output shape and stop handling, beam search basics, seeding/determinism, text helpers and batch generation.

**Not covered:** beam widths ≤ 0, temperature 0 with top-k/nucleus, `min_length`, NaN logits.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Beam width ≤ 0: segfault (0) or uncaught `length_error` (negative) (§2) | `BEAM_WIDTH=0` + `STRATEGY=beam` crashes `chatbot_api_server` | **Verified** | [TD-514](../../guides/TECHNICAL_DEBT.md#td-514-textgenerator-beam-width-of-zero-or-less-crashes-the-process) |
| Temperature 0 ignored by top-k and nucleus (§2) | "Deterministic" config still random (33/50 distinct) | **Verified** | [TD-515](../../guides/TECHNICAL_DEBT.md#td-515-temperature-0-isnt-greedy-for-top-k-and-nucleus-sampling) |
| `min_length` never read; `early_stopping` has no effect (§2) | Empty responses; beam search never stops early | **Verified** (`min_length`); by inspection | [TD-516](../../guides/TECHNICAL_DEBT.md#td-516-textgenerator-ignored-min_lengthearly_stopping-silent-nan-output-and-config-semantics) |
| NaN logits → PAD → silent empty response (§2) | Broken models look like terse ones | **Verified** | [TD-516](../../guides/TECHNICAL_DEBT.md#td-516-textgenerator-ignored-min_lengthearly_stopping-silent-nan-output-and-config-semantics) |
| `max_length` counts BOS/prompt; penalty after filtering; `top_p <= 0`; per-beam uncached calls; no vocab masking; seed 0 (§2) | Minor behaviour and cost issues | **Verified** (length); by inspection | [TD-516](../../guides/TECHNICAL_DEBT.md#td-516-textgenerator-ignored-min_lengthearly_stopping-silent-nan-output-and-config-semantics) |
| CPU `model_fn` contract broken by `ChatbotAPI`; masking/divergence between paths; duplicated softmax | Already tracked | — | TD-506, TD-362, TD-216, TD-424 |

---

## 5. Merge notes

This file replaces `docs/development/api/nlp/text-generator.md` (1,179 lines). Where they disagree, this file wins:
- **"temperature = 0: Greedy (argmax)".** True for `generate()`/`generate_sampling()` only, not top-k or nucleus (§2).
- **`early_stopping` "Stop when beams finish" and `min_length`.** Neither has an effect (§2).

Kept: the strategies, config fields, repetition-penalty and softmax details, and the API list. Dropped: the generic decoding tutorial, parameter-tuning
recipes and examples. The test-coverage report (TEXTGENERATOR_TEST_COVERAGE.md) is unchanged.
