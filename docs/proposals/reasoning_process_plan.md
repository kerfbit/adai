# Plan: Reasoning Process (Thinking Phase) for ADAI

**Status:** Proposed — depends in part on [LeJEPA World Model with Gated Injection](lejepa_world_model_gated_injection_plan.md); see [Interaction with the LeJEPA Plan](#interaction-with-the-lejepa-world-model-plan) below.

## Goal

Give ADAI a generation-time reasoning phase — a bounded span of intermediate deliberation
tokens produced before the final answer, in the same shape as the `<think>...</think>` pattern
discussed for Qwen3.8: same weights, no separate model, effort-configurable, trained rather
than hand-coded. Concretely:

1. A new pair of special tokens delimiting a reasoning span.
2. Generation-loop support for emitting that span with a configurable token budget
   ("reasoning effort"), plus a forced-stop safety valve.
3. A two-stage training standard: SFT cold-start on reasoning-trace data, then optional RL
   fine-tuning that reinforces reasoning trajectories which lead to better final answers.
4. A concrete, testable answer to whether the LeJEPA gated-injection plan should have more
   effect during this phase specifically.

---

## Why this fits ADAI specifically

Same approach as the LeJEPA proposal: grounded in what already exists rather than proposed from
scratch.

| Need | Existing ADAI component |
|---|---|
| Extensible special-token registry | `SpecialTokens.hpp` — `SpecialTokenConfig` already supports custom token IDs beyond the base 4 (see `docs/development/SPECIAL_TOKEN_CONSOLIDATION.md`, "Custom Configurations") |
| Stop-condition machinery in the generation loop | `TextGenerator` + `is_stop_token()` — already the place EOS is checked; a second stop condition (`</think>`) is additive, not a redesign |
| Multi-turn context with role-tagged messages | `ConversationContext::Message` — natural place to add a `reasoning_content` field, mirroring Qwen3.8's `preserve_thinking` |
| Reward scoring for RL | `RewardModel` (`src/RewardModel.hpp`) — Bradley-Terry preference model, already implemented |
| Policy optimization scaffolding | `PPOOptimizer` (`src/PPOOptimizer.hpp`) — **exists but is not production-ready**, see the caveat below |
| Config convention for architecturally-significant, MNS-independent knobs | `config.chatbot.conf` / `config.trainer.conf`'s existing key tables |

### A caveat found while reading the existing RLHF scaffolding

`docs/development/guides/phase5-advanced-features.md` labels `RewardModel` + `PPOOptimizer`
"Production-Ready Implementation." Reading the actual code says otherwise:

- `PPOOptimizer::update()` computes `ratio = exp(new_log_prob - old_log_prob)` but
  `new_log_prob` is literally set equal to `old_log_prob` with a `// Placeholder` comment —
  the ratio is always `1.0`, so the clipped objective never actually clips anything.
- `ValueFunction::update()` accumulates `weight_grads`/`bias_grads` matrices that are never
  written to (no backward pass is implemented) — the value function does not learn.
- The KL-divergence early-stopping check uses `float approx_kl = 0.0f; // Placeholder`.
- Neither class is called anywhere outside `tests/phase5_test.cpp` — there is no live training
  entry point that exercises this path today.

This matters directly for this proposal because Stage 2 (RL fine-tuning) below was the
obvious place to reach for `PPOOptimizer`. **Recommendation: don't.** Use GRPO instead (see
Stage 2) — it needs no value function, which sidesteps the broken half of this scaffolding
entirely, and it's also what Qwen3's own technical report uses for reasoning RL, so it's a
better-precedented choice on its own merits, not just a workaround. The PPO/value-function gap
is real and worth its own technical-debt entry (recommend adding a `TD-029` row to `CLAUDE.md`'s
tag table) independent of whether this proposal proceeds.

---

## Component Specifications

### 1. Special tokens

**File:** `src/SpecialTokens.hpp` (extend in place)

```cpp
namespace adai::SpecialTokenIDs {
    constexpr int PAD = 0;   // unchanged
    constexpr int UNK = 1;   // unchanged
    constexpr int BOS = 2;   // unchanged
    constexpr int EOS = 3;   // unchanged
    constexpr int THINK_START = 4;   // NEW
    constexpr int THINK_END   = 5;   // NEW
}
namespace adai::SpecialTokenStrings {
    // ... unchanged PAD/UNK/BOS/EOS ...
    constexpr const char* THINK_START = "<think>";   // NEW
    constexpr const char* THINK_END   = "</think>";  // NEW
}
```

`SpecialTokenConfig` gains two optional fields, `think_start_id` / `think_end_id`, defaulted to
`-1` ("disabled" — no such token in this vocabulary). This is the same additive pattern the
header already documents for custom IDs; existing vocabularies and checkpoints built before
this proposal see no change at all (`-1` never matches a real token). **Enabling reasoning for a
given model requires rebuilding its vocab** to actually add `<think>`/`</think>` as new entries
(`vocab_builder`) — this doesn't touch the base 4 IDs the consolidation doc protects, but it is
still a vocabulary change, so it's a new model/MNS registration, not an in-place upgrade of an
existing checkpoint.

### 2. Generation loop: reasoning span + effort budget

**Files:** `src/TextGenerator.{hpp,cpp}` (extend), `src/EncoderDecoderModel.{hpp,cpp}` (extend)

```cpp
enum class ReasoningEffort { OFF, LOW, MEDIUM, XHIGH };  // mirrors Qwen3.8's own 3+off scheme

// TextGenerator gains:
std::string generate_with_reasoning(const std::string& prompt,
                                    ReasoningEffort effort,
                                    const std::string& strategy = "nucleus",
                                    int max_answer_length = 100);
```

Mechanically this is one generation loop, not two models or two passes:

1. Emit `<think>`.
2. Generate tokens under the normal stopping rules, **plus** a token-budget cap resolved from
   `effort` (config-driven, see below).
3. Two ways this phase ends: the model emits `</think>` on its own, or the budget is hit — in
   the latter case, **force-emit `</think>`** and continue to the answer rather than letting
   generation run away. This is a direct, deliberate response to the documented Qwen3.8 failure
   mode from the prior conversation (default-`xhigh` runs spending 20+ minutes / 22,000 tokens
   on a trivial prompt, and HF discussion threads titled "This model cannot stop thinking") —
   ADAI's version has a hard backstop by construction, not just a discouraged-but-possible
   outcome.
4. Continue generating the final answer normally; `</think>` is a stop condition for the
   reasoning phase only, not for generation as a whole (EOS remains the real terminator).

**Config** (`config.chatbot.conf`, new block, same key-table convention as existing sections):

```text
REASONING_ENABLED=false
REASONING_EFFORT_DEFAULT=medium
REASONING_TOKEN_BUDGET_LOW=128
REASONING_TOKEN_BUDGET_MEDIUM=512
REASONING_TOKEN_BUDGET_XHIGH=2048
```

`REASONING_ENABLED=false` is the default — `generate_response()` (the existing entry point)
behaves exactly as it does today unless a caller opts in, same "no breaking changes" guarantee
as every prior proposal in this codebase.

### 3. Context: preserving or dropping reasoning across turns

**File:** `src/ConversationContext.hpp` (extend)

```cpp
struct Message {
    std::string role;
    std::string content;             // final-answer content only (unchanged meaning)
    std::string reasoning_content;    // NEW — populated only for assistant messages with
                                       // REASONING_ENABLED; empty otherwise
    int token_count;
};
```

A new constructor flag, `bool preserve_thinking = false` (mirrors Qwen3.8's own naming),
controls whether `reasoning_content` from prior turns is re-serialized into the model's input on
subsequent turns (useful for long-horizon agentic work where earlier deliberation is genuinely
relevant) or dropped (the default — cheaper context, and avoids compounding an already-long
context with reasoning spans that were about a different sub-task).

---

## Training Standard

### Stage 1 — SFT cold start (reuses the existing trainer unchanged)

This is the cheap, low-risk stage, and it requires **no new training-loop code at all**: a
reasoning-trace example is just a longer teacher-forcing target,
`<bos> prompt <think> reasoning trace </think> final answer <eos>`, run through the exact same
`ChatbotTrainer` forward/backward/optimizer-step path every other example already uses. The only
new thing is what's *in* `target_text` — the tokenizer and `EncoderDecoderModel::train_step`
don't need to know reasoning tokens are special beyond what `SpecialTokens.hpp` already declares.

- **Data dependency (out of scope for this proposal):** reasoning-trace training data — either
  human-authored or distilled from a stronger reasoning model's outputs. Sourcing/curating this
  corpus is a prerequisite, not something this architecture change produces on its own.
- **Config:** `REASONING_TRAINING_STAGE=sft` gates whether the dataset loader expects/accepts
  `<think>`-tagged targets, so this can be staged into the existing dataset registry without
  touching unrelated training runs.

### Stage 2 — RL fine-tuning (optional, and where GRPO replaces PPO)

Reinforces reasoning trajectories that lead to *better final answers*, not reasoning-span length
for its own sake — the reward signal is scored on the completed `(reasoning, answer)` pair, via
`RewardModel`, same Bradley-Terry preference-pair training it already supports.

Recommend **GRPO in place of `PPOOptimizer`**, for a new `src/GRPOOptimizer.{hpp,cpp}`:

- Sample a *group* of `K` completions per prompt (varying reasoning trajectories via sampling
  temperature).
- Score each with `RewardModel`.
- Advantage = each completion's reward normalized against the group's own mean/std — **no value
  function required**, which is precisely the half of `PPOOptimizer` that doesn't currently
  work.
- Otherwise the same clipped-surrogate update `PPOOptimizer::compute_policy_loss` already
  implements correctly can be reused as-is; only the advantage source and the (absent) value
  function change.

This mirrors Qwen3's own documented reasoning-RL stage (GRPO against verifiable/scored
outcomes), so it's following established practice for this specific problem, not a
codebase-driven workaround dressed up as a design choice.

### Evaluation standard

- **A/B via config**: `REASONING_ENABLED` toggles the whole feature off at inference, same
  pattern as `WORLD_MODEL_ENABLED` in the LeJEPA plan.
- **Regression gate**: existing `ENABLE_GENERATION_QUALITY_METRICS` BLEU/ROUGE sampling must not
  regress on final-answer quality with reasoning enabled vs. disabled.
- **New metric**: mean/95th-percentile reasoning-span token count per validation batch, pushed
  through `TrainingMetricsService` — this is the direct instrument for catching a Qwen3.8-style
  overthinking regression before it ships, not just after users complain.

---

## Interaction with the LeJEPA World-Model Plan

This is the specific question raised: **should the LeJEPA gated cross-attention injection have
increased effect during the reasoning phase specifically?**

### The case for "yes"

1. **Reasoning tokens are exploratory in a way final-answer tokens aren't.** The reasoning span
   is the model considering, discarding, and revising candidate approaches — it's closer to
   free generation under loose constraints than to precisely-conditioned answer production. A
   geometrically well-formed embedding space (SIGReg's actual job: isotropic, non-collapsed)
   plausibly matters more when the model needs to generate many *diverse* intermediate tokens,
   since a collapsed or degenerate representation space biases exactly that kind of generation
   toward repetition.
2. **This connects directly to the documented Qwen3.8 failure mode from the last conversation.**
   The 22,000-token SVG-circle deliberation — looping through animation strategies, color
   palettes, geometric aesthetics — reads like a representation-diversity problem as much as a
   stopping-criterion problem. SIGReg exists specifically to prevent the kind of collapse that
   produces repetitive, narrow trajectories. That's a plausible mechanism for why the world
   model's geometric prior could matter *more* exactly where ADAI's own reasoning phase is most
   at risk of the same failure.
3. **The existing (mandatory) encoder cross-attention is a fidelity signal** — "attend to what's
   actually in the prompt." That matters most for the final answer's correctness. During
   reasoning, strict fidelity to the prompt matters comparatively less than having a
   well-structured space of plausible next thoughts to draw from — which is the world model's
   contribution, not the existing encoder's.
4. **Reasoning spans are long, and the benefit of a geometric prior compounds with length** —
   small biases toward degenerate generation accumulate over thousands of autoregressive steps.
   The final-answer span, by construction, is short by comparison.

### The case for "no" / the real risk

1. **This is the same evidence read the other way.** If the reasoning phase is *already* the
   most underconstrained, failure-prone part of generation, opening a second injection gate
   wider there is exactly where an uncalibrated side-signal could do the most damage — adding
   degrees of freedom to a phase already prone to derailment is not obviously a fix.
2. **SIGReg has no notion of correctness or task relevance — it is a purely geometric
   constraint.** "Well-shaped embeddings" is not the same claim as "better reasoning." The
   connection above is a plausible mechanism, not a proven one, and it could just as easily turn
   out to be inert.
3. This inherits the parent proposal's own caveat: this exact combination is not documented
   anywhere. Treat the postulate as a hypothesis to test, not a design decision to bake in
   unconditionally.

### The concrete, testable design (recommended)

Don't answer this by assumption — build the ability to *measure* it into the same gate
mechanism the LeJEPA plan already proposes, and let the Phase 1 pilot's own metrics settle it.

Extend `DecoderBlock`'s gated world-model cross-attention (from the LeJEPA plan) with a
**phase-conditioned gate pair** instead of one global scalar:

```cpp
// DecoderBlock, extending the LeJEPA plan's gated cross-attention path:
float gate_reasoning{0.0f};   // active when generating inside <think>...</think>
float gate_answer{0.0f};      // active otherwise
```

At each position, `forward()` selects which gate applies based on whether that position falls
inside the current reasoning span — a signal ADAI already has to track for the `</think>`
stopping logic above, so this costs nothing new to compute. Both still start at exactly `0.0f`
(no-op guarantee preserved), and both are logged separately by the LeJEPA plan's existing
`mean(tanh(gate))` metric — now as two series instead of one.

This turns "should reasoning get more effect" from an architectural assumption into a readout:
after Phase 1 fine-tuning, compare `|tanh(gate_reasoning)|` against `|tanh(gate_answer)|` across
layers. If reasoning-phase gates consistently open further, that's empirical support for the
postulate above. If they don't — or if `gate_reasoning` ends up *more* prone to instability, per
the risk case — that's the answer too, and it's a cheap experiment (two extra floats per gated
layer) either way, run entirely within the LeJEPA plan's existing Phase 1 pilot rather than
requiring a separate one.

**A cheaper first cut**, before committing to two independently-learned gates: a single learned
gate as the LeJEPA plan already specifies, multiplied by a fixed, non-learned
`REASONING_WORLD_MODEL_GATE_SCALE` (default `1.0`) config value applied only while inside the
reasoning span. This answers a narrower but much cheaper question first — does *any* change in
injection strength during reasoning move the needle on repetition/coherence metrics at all —
before spending the extra parameters and pilot time on fully independent learned gates.

### Recommendation

Build the phase-conditioned gate scaffolding (it's a small, backward-compatible extension of
work already planned), but treat "increased effect during reasoning" as the pilot's finding to
report, not a premise to design around. Amend the LeJEPA plan's own pilot (Implementation Phase
5 there) to log the two gate series once this proposal's reasoning phase exists to condition on
— the two proposals share one pilot run rather than needing two.

---

## File Structure

```text
src/
├── SpecialTokens.hpp            # extended: THINK_START/THINK_END
├── ConversationContext.hpp/.cpp # extended: reasoning_content, preserve_thinking
├── TextGenerator.hpp/.cpp       # extended: generate_with_reasoning, effort budgets
├── EncoderDecoderModel.hpp/.cpp # extended: reasoning_effort param on generate_response
├── DecoderBlock.hpp/.cpp        # extended (amends LeJEPA plan): gate_reasoning / gate_answer
├── GRPOOptimizer.hpp/.cpp       # new — value-function-free RL, reuses RewardModel

tests/
├── specialtokens_reasoning_test.cpp   # new
├── textgenerator_reasoning_test.cpp   # new — budget cap + forced-stop cases
├── grpooptimizer_test.cpp             # new
├── decoderblock_test.cpp              # extended: phase-conditioned gate no-op + selection cases
```

## Implementation Phases

Broken into small, independently reviewable chunks. IDs (`RP-*`) are referenced from
[docs/proposals/README.md](README.md)'s cross-proposal recommended order — keep them stable if
this section is edited further.

- **RP-1a — Extend `SpecialTokens.hpp`.** `THINK_START`/`THINK_END` IDs + strings, new
  `SpecialTokenConfig` fields (default `-1`, disabled). No generation-loop changes yet.
- **RP-1b — `vocab_builder` support.** Add the two new token strings to a vocabulary on
  request, without touching the base 4 IDs. Round-trip test: encode/decode a string containing
  `<think>`/`</think>` through `BPETokenizer`.
- **RP-2a — `generate_with_reasoning` core loop.** Phase tracking, token-budget cap, forced
  `</think>` emission; `REASONING_ENABLED=false` by default. Testable with a hand-rolled model
  that emits `<think>` immediately — no trained reasoning behavior needed yet, this chunk is
  purely about the generation-loop mechanics. **This is the chunk LeJEPA's `LJ-3a` wants done
  first** — it's what a phase-conditioned gate would key off of.
- **RP-2b — `ConversationContext` extension.** `reasoning_content` field on `Message`,
  `preserve_thinking` constructor flag. Independent of `RP-2a`; can proceed in parallel.
- **RP-3 — Stage 1 SFT.** Dataset-loader support for `<think>`-tagged targets, then a training
  run once reasoning-trace data exists. No new trainer code, per the Training Standard above —
  this chunk's real prerequisite is the data itself, not engineering.
- **RP-4 — `GRPOOptimizer`.** Fully standalone; testable against `RewardModel` independent of
  every other chunk in either proposal. Also stands on its own as a real fix for the
  `PPOOptimizer`/`ValueFunction` gap noted above, regardless of whether reasoning ships.
- **RP-5 — Phase-conditioned gate extension to `DecoderBlock`.** Contingent on `LJ-3a` (the
  LeJEPA plan's gated `DecoderBlock` extension) already existing. **If `RP-2a` lands before
  `LJ-3a` is built, this chunk is absorbed into `LJ-3a` directly** (build the dual gate the
  first time, per that chunk's sequencing note) rather than done as a separate follow-up edit to
  the same class.
- **RP-6 — Joint pilot.** Small-scale run combining `RP-3` (Stage 1 SFT reasoning) with the
  LeJEPA world model and phase-conditioned gates, specifically to produce the `gate_reasoning`
  vs. `gate_answer` comparison from the Interaction section above. Shared with `LJ-5` rather than
  run separately.

## Testing Strategy

- **Unit:** forced-stop triggers exactly at the configured token budget, never later
  (`textgenerator_reasoning_test.cpp`); `</think>` is a phase-boundary, not a generation
  terminator (EOS still required to end the sequence); phase-conditioned gate selection picks
  the right gate at a span boundary (off-by-one at the `<think>`/`</think>` tokens themselves is
  the case worth being precise about).
- **Regression:** `REASONING_ENABLED=false` reproduces existing `textgenerator_test.cpp`/
  `encoderdecoder_test.cpp` behavior unmodified — same standard as the LeJEPA plan's own
  compatibility guarantee.
- **GRPO:** reward-normalized advantage on a synthetic group of completions with known relative
  quality ordering should rank them correctly; no value-function dependency to test (that's the
  point).

## Compatibility

- **No breaking changes.** `REASONING_ENABLED=false` by default; `think_start_id`/`think_end_id`
  default to `-1` (disabled) in `SpecialTokenConfig`, so existing vocabularies are untouched
  until a model explicitly opts in and rebuilds its vocab.
- **Independent of the LeJEPA plan.** This proposal stands alone — the phase-conditioned gate
  section is an *amendment* to that plan, not a dependency of this one. Reasoning can ship
  without a world model attached at all; `gate_reasoning`/`gate_answer` are only relevant once
  both proposals are in place together.

## Risks / Open Questions

- **Reasoning-trace data sourcing is unscoped here** and is the actual long pole — the
  architecture support above is comparatively cheap next to acquiring or distilling a good
  reasoning-trace corpus.
- **The PPO/value-function gap is real** regardless of whether this proposal proceeds; flagging
  it here so it doesn't get rediscovered later as a surprise when someone tries to actually run
  `PPOOptimizer`.
- **The LeJEPA interaction is a hypothesis, not a finding.** Section above is deliberately
  written to be falsified by the pilot, not to presuppose the answer.
- **Effort-level naming intentionally mirrors Qwen3.8's own `low`/`medium`/`xhigh` scheme** for
  familiarity, but ADAI's budgets are raw token caps (config-defined), not a
  learned/RL-calibrated notion of "effort" the way Qwen3.8's appears to be — worth being clear
  in any user-facing docs that this is a simpler, budget-based approximation initially, not the
  same mechanism.
