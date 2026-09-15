# Plan: LeJEPA World Model + Hippocampal Memory with Gated Injection into the Decoder Stack

**Status:** Proposed — research/pilot stage, not yet scoped for full implementation.
**See also:** [Reasoning Process (Thinking Phase)](reasoning_process_plan.md) proposes and
argues for a phase-conditioned extension of this plan's gate (`gate_reasoning` /
`gate_answer` instead of one global gate) — read that plan's "Interaction with the LeJEPA
World-Model Plan" section before implementing Implementation Phase 3 below if the reasoning
proposal is also in scope. That interaction is now doubly relevant given the hippocampal memory
component added below — see the note in [Component 6](#6-repetition-penalized-gated-cross-attention-hippocampal-memory).

## Goal

Add two self-supervised side signals to ADAI's decoder alongside the existing encoder-decoder
pathway — a slow, consolidated one and a fast, episodic one, mirroring the
**Complementary Learning Systems** framing from neuroscience (McClelland, McNaughton &
O'Reilly 1995; Kumaran, Hassabis & McClelland 2016), where a slow cortical system that builds
structured, generalized representations coexists with a fast hippocampal system that encodes
specific recent experience for rapid recall. Concretely:

1. Keep the existing `LLMEncoder` → `CrossAttention` → `LLMDecoder` pathway exactly as it is
   today (fixed, mandatory, paired-supervision conditioning).
2. Add a new **LeJEPA-trained world-model encoder** — a second, independently pretrained,
   frozen representation-learning stack — as a side process. This is the "slow/cortical" signal:
   consolidated, non-episodic, trained once.
3. Add a new **hippocampal memory model** — a bounded, continuously-updated episodic buffer
   that stores and retrieves recent context, rather than a pretrained encoder. This is the
   "fast/episodic" signal: specific, recent, and — critically — must not let the decoder keep
   recalling the same episode indefinitely.
4. Inject both into each `DecoderBlock` through **independent gated cross-attention paths**
   that start as mathematical no-ops and are trained to open only as far as each earns its keep.
   The hippocampal path's gate additionally carries a **gradually increasing penalty for
   repetition** — attending repeatedly to the same stored episode gets progressively suppressed,
   not just gated on/off once.
5. Define a training standard for both components: how each is pretrained (or, for hippocampal
   memory, why it isn't), how each is frozen or not, how the gates are fine-tuned, and how both
   are evaluated against the un-gated baseline.

This is the follow-up to a design discussion (world model vs. LLM; JEPA-family self-supervision;
gated/zero-init cross-attention as used in RETRO and Flamingo) applied concretely to ADAI's
existing C++ transformer stack. It is a genuinely novel combination — LeJEPA (arXiv:2511.08544)
is a representation-learning objective, not a decoder architecture, and gated side-injection is
documented elsewhere (RETRO, Flamingo, ControlNet) but not in combination with LeJEPA's SIGReg
objective, nor with a hippocampal-style episodic buffer. Treat the architecture below as a
proposal to validate at small scale, not an established result to build directly at production
scale.

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
- **Complementary Learning Systems (CLS)**: the neuroscience precedent for pairing a slow,
  structured system with a fast, episodic one. The hippocampus rapidly encodes specific
  experiences (one-shot, high learning rate) via **pattern separation** — distinct episodes get
  distinct representations, specifically to avoid interference between similar memories — while
  the neocortex slowly consolidates structured, generalized knowledge across many experiences.
  LeJEPA's world model plays the cortical role here (slow, consolidated, frozen after
  pretraining); the new hippocampal memory model below plays the episodic role (fast, updated
  continuously, never "done" training in the SIGReg sense).
- **Coverage mechanism** (See, Liu & Manning 2017, "Get To The Point"): the precedent for the
  repetition penalty specifically. In neural summarization, a *coverage vector* accumulates how
  much attention each source position has already received, and that accumulated value is
  subtracted from future attention scores to the same position — discouraging the decoder from
  repeatedly attending to (and thus repeating) the same source content. This is a penalty on
  *what gets attended to*, not on *which tokens get emitted* — a different mechanism from
  ADAI's existing `TextGenerator::apply_repetition_penalty` (see the table below), and the one
  this proposal adapts for the hippocampal gate.

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
| A source of well-shaped keys for episodic retrieval | `LeJEPAEncoder::encode()` (this same plan, component 1) — SIGReg trains it specifically for isotropic, non-collapsed embeddings, which is exactly the geometry nearest-neighbor retrieval wants. Hippocampal memory reuses it rather than training a third encoder. |
| A precedent — and a specific lesson learned — about repetition penalties | `TextGenerator::apply_repetition_penalty` (`src/TextGenerator.cpp`) already penalizes repeated *tokens* in the output logits. Its own history (see `TD-066` in the code comment) is directly relevant: an earlier version applied the penalty once per *occurrence*, compounding multiplicatively and growing **without bound** over a long generation — logged as a bug and fixed to a bounded, membership-based penalty instead. This plan's hippocampal repetition penalty is a different mechanism (attention-level, not logit-level — see Component 6) but must not repeat that specific mistake; its growth is designed to be bounded by construction, not just informally "gradual." |

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
                         Norm → CrossAttention(hippocampal memory,             (NEW)
                                 repetition-penalized scores) → tanh(gate_h) ⊙ (·) → Add
                                        │
                                        ▼
                         Norm → FeedForward → Add
                                        │
                                        ▼
                                    Output
                                        ▲                        ▲
                                        │ K, V (gated, 0-init)    │ K, V (gated, 0-init;
                         ┌──────────────┴──────────────┐         │  scores penalized by
                         │  LeJEPAEncoder (NEW, frozen  │         │  per-slot coverage)
                         │  after pretraining)          │    ┌────┴──────────────────────┐
                         └───────────────────────────────┘    │ HippocampalMemory (NEW,    │
                                                                │ continuously written/read, │
                                                                │ bounded episodic buffer)   │
                                                                └─────────────────────────────┘
```

Three side inputs feed the same decoder stack, structurally parallel but functionally distinct:

- **Encoder path** (existing): mandatory, paired-supervision, trained jointly with the decoder
  from the start — this is "what to condition generation on."
- **World-model path** (new, cortical-like): optional, self-supervised, pretrained
  independently and frozen, injected through a gate that starts closed — this is "what the
  geometry of the training distribution looks like," a correction rather than a requirement.
- **Hippocampal-memory path** (new, episodic-like): optional, never "pretrained" in the SIGReg
  sense — it fills up during actual use — injected through a gate that also starts closed, but
  whose attention scores are additionally penalized in proportion to how much that specific
  memory slot has already been attended to. This is "what specifically happened recently, but
  don't keep dwelling on the same moment."

Per [the prior discussion](#background-condensed): SIGReg governs the *shape* of the world
model's embedding space (isotropic, non-collapsed); it has no mechanism to know what any
embedding *means*. The existing encoder supplies task content. The world-model gate is what
lets the decoder decide, empirically, whether the world model's geometric prior is worth
listening to at all — and the hippocampal gate additionally decides, moment to moment, whether a
*specific recalled episode* is still worth attending to or has already been drawn on enough.

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

### 5. `HippocampalMemory`

**Purpose:** Fast episodic memory. Structurally *not* a transformer encoder like
`LeJEPAEncoder` — it's a bounded, continuously-updated buffer of `(key, value)` pairs, written
to as generation happens and read from via similarity search. No pretraining phase: it starts
empty and accumulates content during actual use, matching the hippocampus's one-shot encoding
(in contrast to the world model's slow, batch-pretrained consolidation).

**Files:** `src/HippocampalMemory.hpp`, `src/HippocampalMemory.cpp`

```cpp
class HippocampalMemory {
   private:
    struct Slot {
        Matrix key;              // [1, d_model] — from LeJEPAEncoder::encode(), see table above
        Matrix value;             // [1, d_model] — may equal key, or a separately stored payload
        float coverage{0.0f};     // NEW — decayed cumulative attention this slot has received;
                                   // see Component 6. Reset to 0 when a slot is (re)written.
    };
    std::deque<Slot> slots;
    int capacity;                 // ring-buffer bound — oldest slot evicted on overflow
    int d_model;

   public:
    HippocampalMemory(int d_model, int capacity = 512);

    /** Write a new episode. FIFO eviction when at capacity — the simplest possible write
     *  policy (v1); salience-gated writing (only store sufficiently novel episodes, i.e.
     *  pattern separation) is a documented future extension, not required for the pilot. */
    void write(const Matrix& key, const Matrix& value);

    /** Materialize all currently-stored keys/values as K/V matrices for cross-attention,
     *  [num_slots, d_model] each — the shape CrossAttention::forward already expects. */
    std::pair<Matrix, Matrix> read_all() const;

    /** Coverage accessors — see Component 6 for how these are updated and consumed. */
    std::vector<float>& coverage_vector();
    void decay_coverage(float gamma);   // coverage[i] *= gamma, called once per decode step

    int size() const { return static_cast<int>(slots.size()); }
    void clear();   // new conversation / new session boundary

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
};
```

### 6. Repetition-penalized gated cross-attention (hippocampal memory)

**Purpose:** The injection point for `HippocampalMemory`, and the specific mechanism for
"gradually increasing penalty for repetition." Structurally a sibling of Component 4's
world-model gate (same nullable-path, zero-init-gate pattern in `DecoderBlock`), but the
attention scores themselves are modified before softmax — this is what makes the penalty about
*content* (which episode gets recalled) rather than *token identity* (which word gets emitted,
already covered by `TextGenerator::apply_repetition_penalty`). The two penalties catch different
failure modes: a lexical repetition penalty cannot see a model that keeps re-deliberating the
same *idea* in different words each time — exactly the shape of the Qwen3.8 overthinking case
discussed previously (different phrasing each loop — "animation strategies," "color palettes,"
"geometric aesthetics" — but circling the same underlying content). An attention-level coverage
penalty catches that; a token-level one structurally cannot.

**File:** `src/DecoderBlock.{hpp,cpp}` (extend further — same file as Component 4)

```cpp
class DecoderBlock {
    // ... existing + Component 4's world-model members unchanged ...
    std::unique_ptr<CrossAttention> hippocampal_cross_attention;  // NEW, nullable
    std::unique_ptr<LayerNorm> norm_hippocampal;                  // NEW, nullable
    float gate_h{0.0f};        // NEW — separate from the world-model gate
    float gate_h_grad{0.0f};

   public:
    /**
     * @param memory              Pointer to the hippocampal memory to attend over, or nullptr
     *   to skip this path entirely (same no-op guarantee as Component 4).
     * @param repetition_alpha    Penalty growth rate (config: HIPPOCAMPAL_REPETITION_ALPHA).
     * @param repetition_decay    Per-step coverage decay, 0 < gamma <= 1
     *   (config: HIPPOCAMPAL_REPETITION_DECAY). gamma = 1 disables decay (pure running sum);
     *   gamma < 1 makes the penalty track *recent* repetition more than distant repetition.
     */
    Matrix forward(/* ...Component 4's params... */,
                   HippocampalMemory* memory = nullptr,
                   float repetition_alpha = 0.0f, float repetition_decay = 0.95f);

    // Applied inside forward(), after Component 4's world-model Add & Norm:
    //   if (memory && memory->size() > 0) {
    //       auto [K, V] = memory->read_all();
    //       Matrix raw_scores = compute_scores(query, K);   // pre-softmax, pre-CrossAttention
    //       // Bounded, gradual penalty — see rationale in "Why this fits ADAI" above:
    //       // subtractive in score-space (never inverts sign, unlike the old multiplicative
    //       // token-penalty bug), and self-bounding because attn_weight <= 1 per step means
    //       // coverage[i] converges to at most 1/(1-repetition_decay).
    //       for (i in memory->size())
    //           raw_scores[i] -= repetition_alpha * memory->coverage_vector()[i];
    //       hm_attn = hippocampal_cross_attention->forward_with_scores(
    //           norm_hippocampal->forward(residual), raw_scores, V);
    //       residual = residual + std::tanh(gate_h) * hm_attn;
    //       // Coverage update, once per step, decay-then-accumulate:
    //       memory->decay_coverage(repetition_decay);
    //       memory->coverage_vector()[i] += hm_attn.attention_weights[i];  // per slot
    //   }

    float get_gate_h() const { return gate_h; }   // for metrics — see Training Standard
};
```

`CrossAttention::forward` takes raw `query_input`/`kv_input` and computes its own scores
internally, so this needs one new entry point, `forward_with_scores` (or an additive
`score_bias` parameter on the existing `forward`) that accepts a pre-computed additive bias —
a small, targeted extension rather than a new attention class, since every other part of
`CrossAttention` (projections, softmax, backward) is unchanged.

**Auxiliary training signal (optional, matching See et al.'s own coverage loss):** in addition
to the inference-time score penalty above, Phase 1 fine-tuning can add a small
`coverage_loss = sum_i min(attention_weights[i], coverage[i])` term to the training objective —
this gives the decoder a gradient signal to *avoid* needing the penalty in the first place,
rather than relying purely on a decode-time correction. Config-gated
(`HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT`, default `0.0` = off for the initial pilot).

### 7. `EncoderDecoderModel` extension

```cpp
class EncoderDecoderModel {
    // ... existing members unchanged ...
    std::unique_ptr<LeJEPAEncoder> world_model;         // NEW, nullptr = feature disabled entirely
    std::unique_ptr<HippocampalMemory> hippocampal_memory;  // NEW, nullptr = feature disabled entirely
   public:
    /** Attach a pretrained, frozen world model. Passing nullptr disables the feature and
     *  restores exact current behavior (all gates stay unused/uninitialized). */
    void set_world_model(std::unique_ptr<LeJEPAEncoder> wm);
    LeJEPAEncoder* get_world_model() { return world_model.get(); }

    /** Attach an (initially empty) hippocampal memory. Passing nullptr disables the feature. */
    void set_hippocampal_memory(std::unique_ptr<HippocampalMemory> hm);
    HippocampalMemory* get_hippocampal_memory() { return hippocampal_memory.get(); }
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

### Hippocampal memory: no Phase 0, joint training only

`HippocampalMemory` has no self-supervised pretraining stage — there is nothing to pretrain; the
buffer is empty until the model is actually run. Its training standard is narrower than the
world model's:

- **What actually has learnable weights:** `hippocampal_cross_attention`'s Q/K/V/O projections
  and `gate_h` — the buffer's *contents* are just stored activations (keys come from the
  already-trained, frozen `LeJEPAEncoder::encode()`), not something gradient descent touches
  directly.
- **Trained during Phase 1, alongside the world-model gate** — not a separate stage. Same
  no-op-first guarantee: `gate_h` starts at `0.0f`, verified identical-output before any
  training, same as Component 4.
- **Write policy is a runtime/config concern, not a training-time one:** what gets written to
  memory (every turn? every reasoning step, once the reasoning proposal exists? salience-gated?)
  is decided by the calling code (`EncoderDecoderModel`/`ChatbotAPIServer`), not learned. v1
  ships FIFO-always-write; salience-gated writing (pattern separation — only write if
  sufficiently dissimilar from existing slots, by cosine distance in `LeJEPAEncoder` embedding
  space) is a documented extension, not required for the pilot.
- **Config** (same `config.trainer.conf`/`config.chatbot.conf` block convention):

  ```text
  HIPPOCAMPAL_MEMORY_ENABLED=false
  HIPPOCAMPAL_MEMORY_CAPACITY=512
  HIPPOCAMPAL_REPETITION_ALPHA=0.0
  HIPPOCAMPAL_REPETITION_DECAY=0.95
  HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT=0.0
  ```

  `HIPPOCAMPAL_REPETITION_ALPHA=0.0` as the *default* is deliberate: it makes the penalty an
  explicit opt-in magnitude, not just an enabled/disabled flag, so the pilot can sweep it rather
  than committing to one growth rate up front.
- **New standing metrics** (alongside `mean(tanh(gate))` from Component 4): push
  `mean(tanh(gate_h))` and `mean(coverage)` across currently-stored slots through
  `TrainingMetricsService`. A `mean(coverage)` that keeps climbing across a long generation
  without `gate_h` correspondingly shrinking is the concrete symptom to watch for — it would
  mean the penalty isn't actually discouraging repeated recall, only recording it.

### Evaluation standard

- **A/B via config, not code:** `WORLD_MODEL_ENABLED` toggles the feature off entirely at
  inference (`chatbot_api_server`), so quality comparisons against the existing baseline are a
  config flip, not a rebuild.
- **Regression gate:** the existing `ENABLE_GENERATION_QUALITY_METRICS` BLEU/ROUGE sampling
  (already run during validation) is the acceptance bar — the gated model must not regress
  those scores relative to the `WORLD_MODEL_ENABLED=false` baseline before Phase 2 is allowed
  to proceed.
- **Repetition-specific regression check:** distinct-n / self-BLEU (or an equivalent n-gram
  diversity metric) computed over generated validation samples, `HIPPOCAMPAL_MEMORY_ENABLED=true`
  vs. `false`, at a few `HIPPOCAMPAL_REPETITION_ALPHA` settings including `0.0`. The penalty
  earns its complexity only if diversity improves at `alpha > 0` relative to `alpha = 0` with
  the memory otherwise enabled — that isolates the penalty's own effect from the memory
  feature's effect.

---

## File Structure

```text
src/
├── LeJEPAEncoder.hpp / .cpp        # new
├── Predictor.hpp / .cpp            # new
├── SIGReg.hpp / .cpp               # new
├── HippocampalMemory.hpp / .cpp    # new
├── CrossAttention.hpp / .cpp       # extended: forward_with_scores (score-bias entry point)
├── DecoderBlock.hpp / .cpp         # extended (two gated cross-attention paths: world model,
│                                   #   hippocampal memory + repetition penalty)
├── EncoderDecoderModel.hpp / .cpp  # extended (world_model + hippocampal_memory members)

tests/
├── lejepaencoder_test.cpp          # new
├── predictor_test.cpp              # new
├── sigreg_test.cpp                 # new
├── hippocampalmemory_test.cpp      # new — write/evict/read, coverage decay+accumulation
├── decoderblock_test.cpp           # extended: gate==0 no-op cases (both paths), gated-path
│                                   #   gradient checks, repetition-penalty bounding
```

Registered in `src/CMakeLists.txt` / `tests/CMakeLists.txt` per the standard
`src/Component.{cpp,hpp}` + `tests/component_test.cpp` convention (`CLAUDE.md`, "Code
Conventions").

---

## Implementation Phases

Broken into small, independently reviewable chunks. IDs (`LJ-*`) are referenced from
[docs/proposals/README.md](README.md)'s cross-proposal recommended order — keep them stable if
this section is edited further.

- **LJ-1a — `SIGReg`.** [TD-175](../development/guides/TECHNICAL_DEBT.md#td-175-sigreg-sketched-isotropic-gaussian-regularization).
  Standalone, no dependency on anything else in this plan. Unit tests:
  loss on a synthetic isotropic-Gaussian batch should be near zero; on a degenerate
  (collapsed/constant) batch it should be large.
- **LJ-1b — `Predictor`.** [TD-176](../development/guides/TECHNICAL_DEBT.md#td-176-predictor-embedding-space-predictor).
  Standalone (reuses `FeedForward`). Unit test: gradient check on a
  toy embedding pair.
- **LJ-2a — `LeJEPAEncoder` construction.** [TD-177](../development/guides/TECHNICAL_DEBT.md#td-177-lejepaencoder-construction).
  Wraps the existing `EncoderBlock` stack +
  `TokenEmbedding`/`PositionalEncoding`, mirroring `LLMEncoder`'s own composition. No training
  logic yet — just `encode()` and save/load.
- **LJ-2b — `LeJEPAEncoder::train_step`.** [TD-178](../development/guides/TECHNICAL_DEBT.md#td-178-lejepaencodertrain_step-self-supervised-training-loop).
  The self-supervised loop: view construction (span
  masking), `Predictor`, `SIGReg`, combined via `sigreg_lambda`. Verify both loss terms trend
  downward on a small synthetic corpus before touching the decoder at all.
- **LJ-3a — Gated `DecoderBlock` extension.** [TD-180](../development/guides/TECHNICAL_DEBT.md#td-180-gated-decoderblock-extension-world-model--hippocampal-memory-repetition-penalized)
  (filed jointly with `HM-3` — one TD, one change, see that entry). Add the nullable `world_model_cross_attention` /
  `norm_world` / gate path. Critical test is the no-op guarantee
  (`world_model_output == nullptr` ⇒ output identical to current `DecoderBlock::forward`, and
  `gate == 0` with a non-null world-model input also ⇒ identical output).
  **Sequencing note:** if [reasoning_process_plan.md](reasoning_process_plan.md) chunk `RP-2a`
  (phase-boundary tracking in the generation loop) has already landed, build this directly as
  the phase-conditioned `gate_reasoning`/`gate_answer` pair from that plan's "Interaction"
  section instead of a single global gate — see the recommended order in
  [README.md](README.md) for why that avoids rework.
- **LJ-3b — Sparse injection knob.** [TD-181](../development/guides/TECHNICAL_DEBT.md#td-181-sparse-world-model-injection-knob).
  `world_model_inject_every_n_layers` in `LLMDecoder`; only
  every Nth `DecoderBlock` gets the gated path populated.
- **LJ-4a — `EncoderDecoderModel::set_world_model()`.** [TD-182](../development/guides/TECHNICAL_DEBT.md#td-182-encoderdecodermodelset_world_model).
  Wiring + accessor, no training-loop
  changes yet.
- **LJ-4b — `incremental_trainer --objective=lejepa` mode + config keys.** [TD-183](../development/guides/TECHNICAL_DEBT.md#td-183-incremental_trainer---objectivelejepa-mode--world-model-config-keys).
  The new
  `WORLD_MODEL_*` block in `config.trainer.conf`.
- **LJ-4c — MNS registration + checkpointing.** [TD-184](../development/guides/TECHNICAL_DEBT.md#td-184-world-model-mns-registration--checkpointing).
  Independent `ModelRecord` for the world model;
  `LeJEPAEncoder::save()`/`load()` under `training_sessions/`.
- **HM-1 — `HippocampalMemory` buffer.** [TD-179](../development/guides/TECHNICAL_DEBT.md#td-179-hippocampalmemory-buffer).
  Write/evict (FIFO)/`read_all()`, standalone and
  testable without touching the decoder. Depends only on `LJ-2a` (`LeJEPAEncoder::encode()` as
  the key source) — not on `LJ-3a`/`LJ-3b`.
- **HM-2 — `CrossAttention::forward_with_scores`.** [TD-174](../development/guides/TECHNICAL_DEBT.md#td-174-crossattentionforward_with_scores-score-bias-entry-point).
  The score-bias entry point Component 6
  needs; a small, additive extension to the existing class (projections/softmax/backward
  unchanged). Standalone unit test: passing an all-zero bias reproduces `forward()`'s existing
  output exactly.
- **HM-3 — Repetition-penalized gated cross-attention in `DecoderBlock`.** Filed jointly with
  `LJ-3a` as [TD-180](../development/guides/TECHNICAL_DEBT.md#td-180-gated-decoderblock-extension-world-model--hippocampal-memory-repetition-penalized) — see that entry, not a separate TD. The
  `hippocampal_cross_attention`/`norm_hippocampal`/`gate_h` path, coverage accumulation +
  decay, score penalty. **Do this in the same change as `LJ-3a`** if both are in scope — they
  touch the same class, and (per the note in the header of this document) may also need to
  become phase-conditioned per `reasoning_process_plan.md`'s `RP-5`; landing all applicable
  `DecoderBlock` gate work in one pass avoids three separate edits to the same file. Critical
  tests: no-op guarantee (both `memory == nullptr` and `gate_h == 0` ⇒ identical output, matching
  `LJ-3a`'s standard); coverage growth is bounded (`coverage[i] <= 1/(1-repetition_decay)` for
  any input, by construction — verify with a stress test that hammers one slot for many steps).
- **HM-4 — Wiring + config.** [TD-185](../development/guides/TECHNICAL_DEBT.md#td-185-hippocampalmemory-wiring--config--write-policy-call-site).
  `EncoderDecoderModel::set_hippocampal_memory()`, the
  `HIPPOCAMPAL_*` config block, and the write-policy call site (where `EncoderDecoderModel`/
  `ChatbotAPIServer` calls `HippocampalMemory::write()` — v1: once per generated response).
- **LJ-5 / HM-5 — Pilot run.** [TD-186](../development/guides/TECHNICAL_DEBT.md#td-186-lejepa--hippocampal-memory-pilot-run).
  Small `d_model`/`num_layers` (e.g. the toy sizes used in
  `EncoderDecoderExample.cpp`), on a subset of existing training data, to produce both the
  world-model gate-magnitude signal (Phase 1's evaluation standard) and the hippocampal
  `mean(tanh(gate_h))`/`mean(coverage)` readout, swept across a few `HIPPOCAMPAL_REPETITION_ALPHA`
  values. If `reasoning_process_plan.md` chunk `RP-3` (Stage 1 SFT) has also landed by this
  point, run this jointly with that plan's `RP-6` as one combined pilot rather than two or three
  separate ones.

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
- **Hippocampal-specific:** coverage vector never exceeds its analytic bound regardless of how
  long generation runs (property-based/stress test, not just a fixed example); `alpha = 0.0`
  reproduces the un-penalized gated-attention output exactly, isolating "the memory path exists"
  from "the penalty is active."

## Compatibility

- **No breaking changes.** `world_model_output` and the hippocampal `memory` pointer both
  default to `nullptr`/disabled everywhere; a `DecoderBlock` or `EncoderDecoderModel` never
  given either side model behaves exactly as it does today, byte-for-byte in the forward pass.
- **Existing encoder untouched.** `LLMEncoder` / the existing mandatory `CrossAttention` path
  are not modified by this proposal at all.
- **Existing repetition penalty untouched.** `TextGenerator::apply_repetition_penalty` (logit-
  level, token-identity-based) is unchanged and keeps operating exactly as it does today,
  independent of and in addition to the new attention-level penalty — the two are complementary,
  not a replacement.
- **Checkpoint isolation.** World-model checkpoints are independent files/MNS records; loading
  an old chatbot checkpoint that predates this feature requires no migration — it simply has no
  world model or hippocampal memory attached. `HippocampalMemory` itself is not checkpointed as
  part of model weights in the usual sense — it's runtime/session state (`save()`/`load()` exist
  for the narrower case of persisting a long-running session, not for model versioning).

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
- **The repetition penalty is a hypothesis about failure mode, not a guaranteed fix.** It's
  designed to counter *semantic*/episodic repetition (re-attending the same stored moment), on
  the theory that this is a different failure mode than the lexical repetition
  `apply_repetition_penalty` already handles. That theory should be checked against the
  distinct-n/self-BLEU evaluation above before being trusted — it's plausible, grounded in a
  real precedent (coverage mechanisms genuinely do reduce repetition in summarization), but
  ADAI-specific validation hasn't happened yet.
- **Don't let the bound become a false sense of safety.** "Bounded by construction" (coverage
  capped at `1/(1-repetition_decay)`) prevents a TD-066-style runaway value, but a bounded
  penalty can still be miscalibrated — too small to matter, or large enough to make the gate
  effectively unusable once any slot accumulates modest coverage. `HIPPOCAMPAL_REPETITION_ALPHA`
  defaulting to `0.0` and the pilot's alpha sweep (Evaluation standard, above) exist specifically
  so calibration is measured, not assumed.
- **Scope relative to `reasoning_process_plan.md`.** That proposal's own repetition concern
  (Qwen3.8-style reasoning loops) is a natural place to lean on this mechanism harder — e.g. a
  larger `repetition_decay` (longer memory of what's been covered) specifically during a
  `<think>...</think>` span, mirroring that plan's phase-conditioned gate idea. Not designed
  here in detail; flagged as the obvious next amendment once both proposals are further along.
