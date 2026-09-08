# Plan: LeJEPA World Model with Gated Injection into the Decoder Stack

**Status:** Proposed — research/pilot stage, not yet scoped for full implementation.
**See also:** [Reasoning Process (Thinking Phase)](reasoning_process_plan.md) proposes and
argues for a phase-conditioned extension of this plan's gate (`gate_reasoning` /
`gate_answer` instead of one global gate) — read that plan's "Interaction with the LeJEPA
World-Model Plan" section before implementing Implementation Phase 3 below if the reasoning
proposal is also in scope.

## Goal

Add a second, self-supervised "world model" signal to ADAI's decoder alongside the existing
encoder-decoder pathway. Concretely:

1. Keep the existing `LLMEncoder` → `CrossAttention` → `LLMDecoder` pathway exactly as it is
   today (fixed, mandatory, paired-supervision conditioning).
2. Add a new **LeJEPA-trained world-model encoder** — a second, independently pretrained
   representation-learning stack — as a side process.
3. Inject the world model's representations into each `DecoderBlock` through a **second,
   gated cross-attention path** that starts as a mathematical no-op and is trained to open only
   as far as it earns its keep.
4. Define a training standard for this new component: how it is pretrained, how it is frozen,
   how its gate is fine-tuned, and how it is evaluated against the un-gated baseline.

This is the follow-up to a design discussion (world model vs. LLM; JEPA-family self-supervision;
gated/zero-init cross-attention as used in RETRO and Flamingo) applied concretely to ADAI's
existing C++ transformer stack. It is a genuinely novel combination — LeJEPA (arXiv:2511.08544)
is a representation-learning objective, not a decoder architecture, and gated side-injection is
documented elsewhere (RETRO, Flamingo, ControlNet) but not in combination with LeJEPA's SIGReg
objective. Treat the architecture below as a proposal to validate at small scale, not an
established result to build directly at production scale.

---

## Background (condensed)

- **JEPA / LeJEPA**: self-supervised representation learning. A context encoder maps views of
  an input to embeddings; a predictor enforces agreement between embeddings of related views
  *in embedding space* (no pixel/token reconstruction, no decoder). LeJEPA specifically adds
  **SIGReg** (Sketched Isotropic Gaussian Regularization), which constrains embeddings toward an
  isotropic Gaussian distribution and removes the need for stop-gradients, EMA teacher networks,
  or architecture-specific heuristics — the paper reports it validated across ~50 architectures.
  Its output is a *feature extractor*, not something that emits tokens.
- **Gated injection**: RETRO and Flamingo both interleave a side signal (retrieved-text
  encodings; image encodings, respectively) into an otherwise-standard decoder stack via
  cross-attention layers wrapped in a `tanh(gate)` residual, gate initialized at zero. Training
  starts as a strict no-op and the model learns to open the gate only where the side signal
  helps. This is the mechanism ADAI will reuse — its own `CrossAttention` class already
  implements exactly the Q-from-decoder / K,V-from-side-encoder shape this needs.

## Why this fits ADAI specifically

ADAI already has almost every piece this needs, which is what makes the proposal tractable
rather than a from-scratch research project:

| Need | Existing ADAI component |
|---|---|
| A bidirectional transformer encoder stack (backbone for the world model) | `LLMEncoder` / `EncoderBlock` — LeJEPA is backbone-agnostic, so the *same* encoder construction can be reused for a second, independently-trained instance |
| Cross-attention from decoder queries to a side encoder's K/V | `CrossAttention` (`src/CrossAttention.{hpp,cpp}`) — already supports `forward_with_cache` for encoder-side K/V that's constant across generation steps, exactly the caching pattern a second, equally-static side input needs |
| A precedent for an auxiliary model that isn't wired into the main forward/backward graph | `RewardModel` (`src/RewardModel.hpp`) — separate training loop, separate save/load, consumed by another component after the fact |
| Optional-component / no-breaking-changes precedent | The original decoder was added to an encoder-only codebase this same way (see [decoder-design.md](../development/architecture/decoder-design.md), "Compatibility with Existing Code") |
| MNS-authoritative architecture + checkpoint versioning | `ModelNameService` / `mns_cli register` — the world model gets its own `ModelRecord`, independently versioned from the chatbot model |

---

## Architecture Overview

```text
                         ┌─────────────────────────────┐
                         │   LLMEncoder (existing,      │
                         │   fixed, mandatory)          │
                         └──────────────┬───────────────┘
                                        │ K, V (always attended)
                                        ▼
Decoder token stream ──▶ DecoderBlock: Norm → Self-Attn → Add
                                        │
                                        ▼
                         Norm → CrossAttention(encoder) → Add   (existing, unchanged)
                                        │
                                        ▼
                         Norm → CrossAttention(world model) → tanh(gate) ⊙ (·) → Add   (NEW)
                                        │
                                        ▼
                         Norm → FeedForward → Add
                                        │
                                        ▼
                                    Output
                                        ▲
                                        │ K, V (gated, starts at 0)
                         ┌──────────────┴───────────────┐
                         │  LeJEPAEncoder (NEW, frozen   │
                         │  after pretraining)           │
                         └───────────────────────────────┘
```

Two side inputs feed the same decoder stack, structurally parallel but functionally distinct:

- **Encoder path** (existing): mandatory, paired-supervision, trained jointly with the decoder
  from the start — this is "what to condition generation on."
- **World-model path** (new): optional, self-supervised, pretrained independently and frozen,
  injected through a gate that starts closed — this is "what the geometry of the training
  distribution looks like," a correction rather than a requirement.

Per [the prior discussion](#background-condensed): SIGReg governs the *shape* of the world
model's embedding space (isotropic, non-collapsed); it has no mechanism to know what any
embedding *means*. The existing encoder supplies task content. The gate is what lets the decoder
decide, empirically, whether the world model's geometric prior is worth listening to at all.

---

## Component Specifications

### 1. `LeJEPAEncoder`

**Purpose:** Self-supervised world-model encoder. Structurally a transformer encoder stack
(reuses `EncoderBlock`, `TokenEmbedding`, `PositionalEncoding` — same composition as
`LLMEncoder`); the difference is entirely in what it's trained on and with what objective.

**Files:** `src/LeJEPAEncoder.hpp`, `src/LeJEPAEncoder.cpp`

```cpp
class LeJEPAEncoder {
   private:
    std::unique_ptr<BPETokenizer> tokenizer;       // shared vocab with LLMEncoder
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::vector<std::unique_ptr<EncoderBlock>> encoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;
    std::unique_ptr<Predictor> predictor;          // embedding-space predictor, see below
    std::unique_ptr<SIGReg> sigreg;                // regularizer, see below

    int vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length;
    bool requires_grad{true};
    float learning_rate{0.001f};
    float sigreg_lambda{1.0f};   // λ weighting SIGReg term against predictor loss

   public:
    LeJEPAEncoder(int vocab_size, int d_model = 512, int num_layers = 6, int num_heads = 8,
                  int d_ff = 2048, int max_seq_length = 512);

    /** Encode one view of input text to contextualized embeddings — same shape contract
     *  as LLMEncoder::encode, so it's a drop-in for anything that only needs embeddings. */
    Matrix encode(const std::string& text);

    /**
     * Self-supervised training step on a single example: constructs two augmented views
     * internally (span masking over the token sequence — no paired target text required,
     * unlike EncoderDecoderModel::train_step), predicts one view's embedding from the
     * other's, and applies SIGReg to the batch of produced embeddings.
     *
     * @return {predictor_loss, sigreg_loss} — logged separately (see Training Standard)
     */
    std::pair<float, float> train_step(const std::string& text);

    void set_requires_grad(bool requires_grad);   // frozen=false after pretraining phase
    void set_learning_rate(float lr);
    void register_parameters_with_optimizer(Optimizer& optimizer);
    void save(const std::string& directory);
    void load(const std::string& directory);
    void print_config() const;

    EncoderBlock* get_encoder_block(int layer);   // mirrors LLMEncoder's diagnostics accessor
};
```

### 2. `Predictor`

**Purpose:** Embedding-space predictor — given the context view's embedding, predict the
target view's embedding. This is *not* a `LanguageModelHead`; there is no vocabulary
projection and no reconstruction. A small feed-forward stack (reuses `FeedForward`) is
sufficient per the LeJEPA paper's own "no architecture-specific tuning" finding.

**Files:** `src/Predictor.hpp`, `src/Predictor.cpp`

```cpp
class Predictor {
   private:
    std::unique_ptr<FeedForward> net;   // d_model -> d_model, predicts in embedding space
   public:
    Predictor(int d_model, int hidden_dim);
    Matrix forward(const Matrix& context_embedding);
    Matrix backward(const Matrix& grad_output);
    void update_weights();
    void zero_grad();
    void register_parameters_with_optimizer(Optimizer& optimizer);
};
```

### 3. `SIGReg`

**Purpose:** Sketched Isotropic Gaussian Regularization — pushes a batch of embeddings toward
an isotropic Gaussian via random 1D projections and a characteristic-function test, per the
LeJEPA paper. Stateless with respect to model weights (it only consumes/produces gradients
through the embeddings passed to it); no learnable parameters of its own.

**Files:** `src/SIGReg.hpp`, `src/SIGReg.cpp`

```cpp
class SIGReg {
   private:
    int num_sketches;   // number of random projection directions
    int d_model;
   public:
    SIGReg(int d_model, int num_sketches = 64);

    /** Compute regularization loss over a batch of embeddings [batch, d_model]. */
    float compute_loss(const Matrix& embeddings);

    /** Gradient of the loss w.r.t. the input embeddings. */
    Matrix backward(const Matrix& embeddings);
};
```

### 4. Gated cross-attention in `DecoderBlock`

**Purpose:** The injection point. Extends the existing `DecoderBlock` — this is the one place
existing code changes rather than purely adds a new file, so it needs the "no breaking changes"
treatment the original decoder addition used: the new path is only active when a world-model
input is actually passed in.

**File:** `src/DecoderBlock.{hpp,cpp}` (extend, don't replace)

```cpp
class DecoderBlock {
    // ... existing members unchanged ...
    std::unique_ptr<CrossAttention> world_model_cross_attention;  // NEW, nullable
    std::unique_ptr<LayerNorm> norm_world;                        // NEW, nullable
    float gate{0.0f};        // NEW — raw gate parameter, tanh(gate) applied at forward time
    float gate_grad{0.0f};   // NEW

   public:
    /**
     * Forward pass, extended with an optional world-model side input.
     *
     * @param world_model_output  Frozen world-model encoder output [wm_seq_len, d_model],
     *   or nullptr to skip the gated path entirely (bit-identical to current behavior —
     *   this is the backward-compatibility guarantee).
     * @param world_model_mask    Optional padding mask, same shape convention as
     *   cross_attn_mask.
     */
    Matrix forward(const Matrix& input, const Matrix& encoder_output,
                   const Matrix& self_attn_mask, const Matrix* cross_attn_mask = nullptr,
                   const Matrix* world_model_output = nullptr,
                   const Matrix* world_model_mask = nullptr);

    // Applied inside forward(), after the existing cross-attention Add & Norm:
    //   if (world_model_output) {
    //       wm_attn = world_model_cross_attention->forward(
    //           norm_world->forward(residual2), *world_model_output, world_model_mask);
    //       residual2 = residual2 + std::tanh(gate) * wm_attn;
    //   }

    float get_gate() const { return gate; }   // for metrics — see Training Standard
};
```

**Sparse injection knob:** per the RETRO precedent (and ADAI's own discussion of periodic vs.
per-layer injection), this need not run on every layer. `LLMDecoder` gains a config-driven
`world_model_inject_every_n_layers` (default: every layer, i.e. `1`, for the initial pilot —
tune down for cost once the gate is shown to open meaningfully) so only every Nth
`DecoderBlock` is constructed with the gated path populated; the rest pass `nullptr` and incur
zero extra cost.

### 5. `EncoderDecoderModel` extension

```cpp
class EncoderDecoderModel {
    // ... existing members unchanged ...
    std::unique_ptr<LeJEPAEncoder> world_model;   // NEW, nullptr = feature disabled entirely
   public:
    /** Attach a pretrained, frozen world model. Passing nullptr disables the feature and
     *  restores exact current behavior (all gates stay unused/uninitialized). */
    void set_world_model(std::unique_ptr<LeJEPAEncoder> wm);
    LeJEPAEncoder* get_world_model() { return world_model.get(); }
};
```

---

## Training Standard

This is the part the user specifically asked to standardize: how these models get trained, in
terms consistent with ADAI's existing config/MNS/checkpoint conventions.

### Phase 0 — LeJEPA pretraining (new, standalone)

- Run via `incremental_trainer` with a new mode flag, e.g. `--objective=lejepa`, so it reuses
  the existing dataset registry / distributed-queue machinery rather than a bespoke script.
- **Data:** raw conversational text, unpaired — no `(input, target)` pairs required, unlike the
  existing `ChatbotTrainer` flow. This means the world model can pretrain on a much larger,
  cheaper-to-source pool than the paired data the encoder-decoder needs.
- **Config:** new keys in `config.trainer.conf`, following the existing "architecturally
  significant keys" table convention rather than a 6th config file (it's trained by the same
  binary, just a different objective):

  ```text
  WORLD_MODEL_ENABLED=false
  WORLD_MODEL_D_MODEL=512
  WORLD_MODEL_NUM_LAYERS=6
  WORLD_MODEL_NUM_HEADS=8
  WORLD_MODEL_D_FF=2048
  WORLD_MODEL_SIGREG_LAMBDA=1.0
  WORLD_MODEL_SIGREG_NUM_SKETCHES=64
  WORLD_MODEL_INJECT_EVERY_N_LAYERS=1
  ```

- **MNS registration:** the world model gets its own `mns_cli register` entry and its own
  `ModelRecord` (own `D_MODEL`/`NUM_HEADS`/etc., immutable after registration — same rule as
  the chatbot model), so its checkpoint compatibility is tracked independently of the
  chatbot's. A world model and a chatbot model are versioned separately and paired explicitly
  (see `set_world_model()` above), not coupled through one MNS record.
- **Checkpointing:** `LeJEPAEncoder::save()`/`load()` under `training_sessions/`, same
  convention as every other component.
- **Metrics:** push `predictor_loss` and `sigreg_loss` separately through the existing
  `TrainingMetricsService` (two new series, not a new service — consistent with how the trainer
  already reports multiple loss components).

### Phase 1 — Gated fine-tuning (attaches to existing training)

- Load the frozen world model into `EncoderDecoderModel::set_world_model()`.
- `LeJEPAEncoder::set_requires_grad(false)` — same toggle `LLMEncoder` already exposes for
  exactly this purpose.
- All `gate` parameters initialize at `0.0f` (⇒ `tanh(gate) == 0`, strictly no-op). This is the
  mandatory first checkpoint of this phase: run the standard training/eval loop with
  `WORLD_MODEL_ENABLED=true` immediately after attaching, and diff outputs against
  `WORLD_MODEL_ENABLED=false` on a fixed validation batch — they must match, verifying the
  addition is genuinely additive before any gate training happens.
- Continue the existing `incremental_trainer train`/`resume` flow unchanged; gradients now also
  flow into `world_model_cross_attention`, `norm_world`, and `gate` per gated `DecoderBlock`.
  Everything else (loss, optimizer, checkpoint cadence, MNS run/session bookkeeping) is
  untouched.
- **New standing metric:** push `mean(tanh(gate))` per gated layer through
  `TrainingMetricsService` every epoch (`DecoderBlock::get_gate()`), so the dashboard shows
  whether/where the world model is actually being used. A gate that stays near zero across
  training is itself a valid (negative) result — it means the world-model signal isn't earning
  its cost, and the pilot should stop there rather than proceeding to Phase 2.

### Phase 2 — Joint fine-tuning (optional, only if Phase 1 gates open meaningfully)

- Once gates cross a chosen threshold (e.g. `|tanh(gate)| > 0.1` on a majority of gated
  layers, config key `WORLD_MODEL_UNFREEZE_GATE_THRESHOLD`), optionally
  `set_requires_grad(true)` on the world model and continue training end-to-end — same
  staged-unfreezing pattern Flamingo uses.

### Evaluation standard

- **A/B via config, not code:** `WORLD_MODEL_ENABLED` toggles the feature off entirely at
  inference (`chatbot_api_server`), so quality comparisons against the existing baseline are a
  config flip, not a rebuild.
- **Regression gate:** the existing `ENABLE_GENERATION_QUALITY_METRICS` BLEU/ROUGE sampling
  (already run during validation) is the acceptance bar — the gated model must not regress
  those scores relative to the `WORLD_MODEL_ENABLED=false` baseline before Phase 2 is allowed
  to proceed.

---

## File Structure

```text
src/
├── LeJEPAEncoder.hpp / .cpp     # new
├── Predictor.hpp / .cpp         # new
├── SIGReg.hpp / .cpp            # new
├── DecoderBlock.hpp / .cpp      # extended (gated cross-attention path)
├── EncoderDecoderModel.hpp / .cpp  # extended (world_model member + accessor)

tests/
├── lejepaencoder_test.cpp       # new
├── predictor_test.cpp           # new
├── sigreg_test.cpp              # new
├── decoderblock_test.cpp        # extended: gate==0 no-op case, gated-path gradient check
```

Registered in `src/CMakeLists.txt` / `tests/CMakeLists.txt` per the standard
`src/Component.{cpp,hpp}` + `tests/component_test.cpp` convention (`CLAUDE.md`, "Code
Conventions").

---

## Implementation Phases

Broken into small, independently reviewable chunks. IDs (`LJ-*`) are referenced from
[docs/proposals/README.md](README.md)'s cross-proposal recommended order — keep them stable if
this section is edited further.

- **LJ-1a — `SIGReg`.** Standalone, no dependency on anything else in this plan. Unit tests:
  loss on a synthetic isotropic-Gaussian batch should be near zero; on a degenerate
  (collapsed/constant) batch it should be large.
- **LJ-1b — `Predictor`.** Standalone (reuses `FeedForward`). Unit test: gradient check on a
  toy embedding pair.
- **LJ-2a — `LeJEPAEncoder` construction.** Wraps the existing `EncoderBlock` stack +
  `TokenEmbedding`/`PositionalEncoding`, mirroring `LLMEncoder`'s own composition. No training
  logic yet — just `encode()` and save/load.
- **LJ-2b — `LeJEPAEncoder::train_step`.** The self-supervised loop: view construction (span
  masking), `Predictor`, `SIGReg`, combined via `sigreg_lambda`. Verify both loss terms trend
  downward on a small synthetic corpus before touching the decoder at all.
- **LJ-3a — Gated `DecoderBlock` extension.** Add the nullable `world_model_cross_attention` /
  `norm_world` / gate path. Critical test is the no-op guarantee
  (`world_model_output == nullptr` ⇒ output identical to current `DecoderBlock::forward`, and
  `gate == 0` with a non-null world-model input also ⇒ identical output).
  **Sequencing note:** if [reasoning_process_plan.md](reasoning_process_plan.md) chunk `RP-2a`
  (phase-boundary tracking in the generation loop) has already landed, build this directly as
  the phase-conditioned `gate_reasoning`/`gate_answer` pair from that plan's "Interaction"
  section instead of a single global gate — see the recommended order in
  [README.md](README.md) for why that avoids rework.
- **LJ-3b — Sparse injection knob.** `world_model_inject_every_n_layers` in `LLMDecoder`; only
  every Nth `DecoderBlock` gets the gated path populated.
- **LJ-4a — `EncoderDecoderModel::set_world_model()`.** Wiring + accessor, no training-loop
  changes yet.
- **LJ-4b — `incremental_trainer --objective=lejepa` mode + config keys.** The new
  `WORLD_MODEL_*` block in `config.trainer.conf`.
- **LJ-4c — MNS registration + checkpointing.** Independent `ModelRecord` for the world model;
  `LeJEPAEncoder::save()`/`load()` under `training_sessions/`.
- **LJ-5 — Pilot run.** Small `d_model`/`num_layers` (e.g. the toy sizes used in
  `EncoderDecoderExample.cpp`), on a subset of existing training data, purely to produce the
  gate-magnitude signal Phase 1's evaluation standard depends on. If `reasoning_process_plan.md`
  chunk `RP-3` (Stage 1 SFT) has also landed by this point, run this jointly with that plan's
  `RP-6` as one combined pilot rather than two separate ones.

## Testing Strategy

- **Unit:** `SIGReg`/`Predictor` numerical checks (above); `DecoderBlock` no-op guarantees
  (above); gradient checking on the gate parameter itself (finite-difference check that
  `d(output)/d(gate)` matches the analytic `tanh` derivative).
- **Integration:** `LeJEPAEncoder::train_step` loss curves trend downward on synthetic data;
  end-to-end `EncoderDecoderModel` forward/backward with a world model attached produces
  finite, non-NaN gradients through the gated path.
- **Regression:** existing `encoderdecoder_test.cpp` / `decoderblock_test.cpp` suites must pass
  unmodified with `world_model_output = nullptr` (the default) — this is the concrete
  expression of "no breaking changes."

## Compatibility

- **No breaking changes.** `world_model_output` defaults to `nullptr` everywhere; a
  `DecoderBlock` or `EncoderDecoderModel` never given a world model behaves exactly as it does
  today, byte-for-byte in the forward pass.
- **Existing encoder untouched.** `LLMEncoder` / the existing mandatory `CrossAttention` path
  are not modified by this proposal at all.
- **Checkpoint isolation.** World-model checkpoints are independent files/MNS records; loading
  an old chatbot checkpoint that predates this feature requires no migration — it simply has no
  world model attached.

## Risks / Open Questions

- **This is a novel combination, not a validated result.** Neither the LeJEPA paper nor the
  RETRO/Flamingo papers describe this exact pairing. Treat the pilot's gate-magnitude readout
  (Phase 1) as the go/no-go signal before any larger investment — a gate that never opens is a
  legitimate, useful outcome, not a failed implementation.
- **"World model" naming.** In the RL/planning literature "world model" usually means a
  dynamics/future-state predictor (Ha & Schmidhuber, Dreamer). What's described here is a
  representation-learning module in the JEPA sense — good geometry, not temporal prediction.
  Worth flagging in any user-facing docs/naming so this doesn't get confused with a planning
  component.
- **Compute cost of a second encoder + second cross-attention per layer.** `WORLD_MODEL_ENABLED`
  and `WORLD_MODEL_INJECT_EVERY_N_LAYERS` are both there specifically so this can be dialed back
  to near-zero marginal cost if the pilot doesn't justify it.
- **GPU backend parity.** The interfaces above sketch CPU-path signatures only; `gpu_forward`/
  `gpu_backward` mirrors for `LeJEPAEncoder` and the gated `DecoderBlock` path (matching the
  existing `#ifdef ADAI_ENABLE_GPU` pattern throughout the codebase) are deferred until the CPU
  pilot (Implementation Phase 5) justifies the investment — do not build the GPU path first.
