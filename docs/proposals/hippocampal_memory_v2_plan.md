# Proposal: Hippocampal Memory v2 — Encoding, Retrieval, Forgetting and Consolidation Modeled on the Human Hippocampus

**Status:** Proposed (November 3, 2026). Builds on, and does not replace, `HippocampalMemory`
(HM-1/TD-179, TD-193, TD-194, TD-343, TD-396) and the gated hippocampal cross-attention path in
`DecoderBlock` (TD-180, TD-341, TD-342) from
[lejepa_world_model_gated_injection_plan.md](lejepa_world_model_gated_injection_plan.md).

## 1. Why

The current `HippocampalMemory` takes one idea from the hippocampus literature — a fast,
bounded episodic store beside a slow learner (Complementary Learning Systems) — and implements
the rest with engineering defaults:

| Today (`src/HippocampalMemory.*`, `EncoderDecoderModel::maybe_write_hippocampal_memory()`) | Consequence |
|---|---|
| Every generated response is written; key = value = the response's mean-pooled world-model embedding | No selection: near-duplicate answers fill the buffer; a later *question* is matched against past *answers*, not past situations |
| No text or token IDs are kept, only the pooled vector | Nothing can be reinstated or consolidated into the model's weights — the memory can bias attention but never teach |
| `recall_from_swap()` is LIFO and **has no caller** | The swap file is write-only in practice; an evicted episode never comes back, and the file grows without bound (TD-397) |
| One `HippocampalMemory` per loaded model in `chatbot_api_server` | Every concurrent chat session reads and writes the same memory: one user's episode can steer another user's answer. Not tracked as a TD yet |
| Order is ignored except through coverage decay; no notion of "when" or "which conversation" | No recency, no temporal contiguity, no event structure |
| Keys come from the world model's `LeJEPAEncoder`, and the write path returns early without one | The memory can't run without a world model, and its keys are shaped for world-model pretraining, not for recalling episodes |
| The hippocampal path is allocated by `world_model_inject_every_n_layers` (TD-186) and each episode is one pooled vector | Memory injection can't be placed or tuned on its own, and the decoder sees a single averaged row per episode where the world model gets a token-level sequence |
| No consolidation | Memory never becomes knowledge; the slow model and the fast store never exchange anything |

Human memory research has a model for each of these gaps, and most of them reduce to a few
lines of vector arithmetic on what the class already stores. This proposal sets out the
findings, maps each one to a mechanism, and orders the work.

## 2. What the research says

The first group of sources was checked in this pass. Ebbinghaus, the spacing and testing
effects, and Teyler & DiScenna's indexing theory come from standard references and were not
re-checked.

1. **Two learning systems (CLS).** McClelland, McNaughton & O'Reilly (1995): a hippocampal
   system learns specific episodes fast, and the neocortex slowly extracts structure. Fast
   learning in a shared distributed network causes catastrophic interference, so new episodes
   are held separately and *interleaved* into cortical learning through replay. Kumaran,
   Hassabis & McClelland (2016) extend the hippocampus's role to replay that is *prioritized*,
   not uniform, and to cortex learning fast when new material fits an existing schema. They link
   it explicitly to experience-replay buffers in deep RL.
2. **Pattern separation (dentate gyrus) and pattern completion (CA3).** The dentate gyrus
   recodes inputs sparsely in a larger space, so similar experiences are stored as dissimilar
   traces. CA3's recurrent attractor completes a whole stored trace from a partial cue. CA3
   behaves sigmoidally: small input differences are completed to the same memory, and large
   ones are separated (Knierim & Neunuebel 2015). Good separation at encoding is a precondition
   for accurate completion later (human CA3 imaging review, 2014).
3. **Attention is a one-step Hopfield retrieval.** Ramsauer et al. (2020): the transformer
   attention update is the update rule of a continuous modern Hopfield network, which usually
   retrieves a stored pattern in one step. It has fixed points for single patterns, metastable
   states averaging subsets, and a global average. adai's gated hippocampal cross-attention is
   therefore already a CA3-style completion step. What's missing is the separation step
   upstream of it and control over which patterns it may settle on.
4. **Indexing theory.** Teyler & DiScenna (1986): the hippocampus stores an *index* that points
   to the cortical activity of an episode and reinstates it, not the content itself.
5. **Novelty-gated encoding.** Lisman & Grace (2005): the hippocampus detects input it doesn't
   already store. That novelty signal drives dopamine release back into the hippocampus, which
   strengthens LTP, so the loop decides what enters long-term memory. Later mouse work (2016)
   puts part of that boost in the locus coeruleus rather than the VTA, but the computational
   point holds: encoding strength is gated by novelty and salience.
6. **Synaptic/behavioral tagging.** Frey & Morris: a weak event sets a tag that only becomes
   lasting if a strong event nearby supplies the plasticity resources. Behaviorally, a salient
   experience strengthens memory for weak experiences just before and after it.
7. **Temporal context.** Howard & Kahana (2002), the Temporal Context Model: an internal context
   vector drifts slowly. Items are bound to the context present at encoding and cued by its
   overlap with the current context, which gives recency. Recalling an item *reinstates its
   encoding context*, which makes its temporal neighbours easier to recall next. That is the
   contiguity effect, forward-biased but bidirectional. CMR (Polyn, Norman & Kahana 2009) adds
   semantic and source context.
8. **Event boundaries.** Hippocampal activity rises at event boundaries in continuous experience
   (Ben-Yakov & Henson 2018). Boundary-locked hippocampal–cortical coupling predicts later
   reinstatement in recall (Baldassano et al. 2017), and boundary responses predict story memory
   (2020). Experience is encoded as segmented events, with the hippocampus committing an event
   when it ends.
9. **Prioritized replay.** Mattar & Daw (2018): replay accesses memories in order of utility
   (expected gain × need). Human rest replay favours *weakly learned* material and predicts later
   performance (2018). In rodents, replay share scales with reward.
10. **Schemas accelerate consolidation.** Tse et al. (2007): when new associations fit an existing
    schema, one trial is enough and they become hippocampus-independent within about a day,
    versus the usual weeks.
11. **Reconsolidation.** Nader et al. (2000): retrieving a consolidated memory makes it labile
    again for a window, during which it can be updated or disrupted. Memories not retrieved are
    unaffected.
12. **Forgetting and spacing.** Ebbinghaus: retention falls off roughly exponentially with time
    since learning. Each spaced retrieval flattens the curve (the spacing and testing effects,
    e.g. Roediger & Karpicke 2006).

## 3. Proposed design

Six mechanisms, each tied to a finding above, plus (G): the memory owns its encoder and injects
into the decoder stack on its own schedule, the way the world model does. They are listed in
dependency order, and each one can be turned off on its own with a config key (default off),
the same opt-in pattern the current hippocampal keys use.

```
 user turn ──► (G) memory's own encoder (frozen at serve time) ──► cue key k_q
                                               │
            ┌──────────────────────────────────┼─────────────────────────────┐
            │ (E) cue recall: long-term store ─► top-k into live buffer       │
            │ (C) temporal context c_t drifts; boundary detector              │
            ▼                                                                 │
  live buffer (per session) ──► (G) episode segments [rows, d_model] + mask ──► gated
            ▲                     score_bias += ctx overlap + strength          cross-attn in
            │                     (existing repetition + association terms)     every n-th block
            │                                                                   (CA3 step)
 response ──┤ (A) episode = (cue key, outcome value, text, context, strength)
            │ (B) encoding gate: novelty / salience / reconsolidate / separate
            ▼
  (D) forgetting: retrievability R = s·exp(−Δt/τ); spaced recall grows τ
            │
  (F) offline replay: priority = R × loss gain → JSONL into the chatbot pool
                                    → trainer_service pass → mark consolidated
```

### (A) Episodes as an index with a payload (findings 1, 4)

Extend `Slot` from `{key, value}` to an episode record:

```cpp
struct Episode {
    Matrix key;          // [1, d_model] cue: pooled world-model embedding of the *user turn*
    Matrix value;        // [1, d_model] outcome: pooled embedding of user turn + response
    std::uint64_t id;
    std::string scope;   // see (A.1)
    std::int64_t t_unix; // encoding time
    std::uint32_t event_id;
    Matrix context;      // [1, d_ctx] temporal context at encoding, see (C)
    float strength;      // s, see (B)/(D)
    float tau_hours;     // forgetting time constant, see (D)
    std::uint16_t recalls, replays;
    float consolidation; // 0..1, see (F)
    std::string text;    // optional payload (HIPPOCAMPAL_STORE_TEXT), needed only by (F)
};
```

Separating key from value turns the store from autoassociative (response matched to response)
into a heteroassociative one: *situation → what happened*, which is what the decoder needs when
a similar question comes back. `read_all()` keeps its `[n, d_model]` K/V contract, so
`CrossAttention` and the GPU path don't change. The write moves from
`maybe_write_hippocampal_memory(response)` to a turn-level hook that sees both the prompt and the
response.

**(A.1) Scope — fixes cross-session leakage now.** The live buffer and temporal context become
per conversation session (`ChatbotAPI` already has `shared_ptr` sessions and a per-session turn
mutex, TD-227). The long-term store carries a `scope` tag and recall filters on it.
`HIPPOCAMPAL_SCOPE=session` (default) | `global` (single-tenant deployments that want shared
memory, which is today's behaviour). Because the decoder reads the memory through
`EncoderDecoderModel`, generation has to take the session's memory as a parameter instead of
the model-owned pointer. This is the one structural change in the proposal, and it should land
first, as its own fix, even if nothing else here is accepted.

### (B) Encoding gate: novelty, salience, separation, reconsolidation (findings 2, 5, 6, 11)

At write time, compute `m = max_i cos(k, key_i)` over the live buffer and the long-term index:

| Case | Action | Biology |
|---|---|---|
| `m ≥ τ_same` (0.95) | **Reconsolidate**: no new slot; move the match toward the new episode (`key_i ← norm((1−η)key_i + η k)`, same for value), add strength, refresh time | completion of the same memory; reconsolidation |
| `τ_sep ≤ m < τ_same` (0.75–0.95) | **Separate**: write, but remove part of the projection onto the nearest neighbour: `k' = norm(k − λ (k·k_nn) k_nn)`, λ = 0.3 | dentate-gyrus separation of similar-but-different episodes |
| `m < τ_sep` | Write normally | novel episode |

Initial strength is `s0 = s_base + a·(1 − m) + b·salience`. Salience combines:
- **Surprise:** the world model's own predictor error on the turn (`LeJEPAEncoder`'s `Predictor`
  already exists and is saved since TD-427), the closest analogue to a hippocampal mismatch
  signal.
- **Explicit signals:** an optional `"remember": true` on `/chat/session` and, later, user
  feedback.

If `s0 < s_min` the episode isn't written at all. This is the selection step the current
write-everything path lacks.

**Tagging:** when an episode with salience above `θ_tag` is written, slots in the same event
(C) with low strength get `s += κ_tag·salience`. A salient moment strengthens what led up to it.

A slot recalled in the last `HIPPOCAMPAL_LABILE_TURNS` turns uses a lower `τ_same`, so a
correction ("no, I meant…") updates the recalled memory in place instead of storing a
contradicting duplicate.

Optional later step (an owner decision): a true DG-style sparse code for the *match* decision
only — a fixed, seeded random expansion `d_model → 4·d_model` plus top-k winner-take-all. It
gives sharper novelty estimates than raw cosine, and the stored keys stay dense for attention.

### (C) Temporal context and event boundaries (findings 7, 8)

Each session keeps a context vector `c ∈ R^{d_ctx}` (`d_ctx` = 64), built from a fixed, seeded
random projection `P` of the turn key:

```
c_t = norm(ρ·c_{t−1} + β·P·k_t)        // ρ ≈ 0.9, β ≈ 0.3 (TCM drift)
```

- **Encoding:** each episode stores `context = c_t`.
- **Retrieval bias:** `hippocampal_score_bias()` gains one term,
  `+ γ_ctx · cos(c_now, context_slot)`, beside the existing repetition and association terms.
  That gives recency and temporal contiguity. It goes through the same bounded injection point,
  so the TD-066 lesson about unbounded growth still holds.
- **Retrieved-context reinstatement:** when a slot takes more than `θ_win` of the hippocampal
  attention mass in a committed step (`commit_attention()` already has these weights, TD-343),
  set `c ← norm(c + β_r·context_slot)`. Its temporal neighbours then become easier to reach
  next, which is the contiguity effect, and it largely subsumes the Hebbian association layer
  (TD-194). Keep TD-194 for now and compare the two in the evaluation (§5).
- **Boundaries:** a boundary is a new session, a gap longer than `Δt_gap`, or
  `cos(k_t, k_{t−1}) < τ_boundary` (a topic shift). At a boundary, `β` is raised for that step
  (context jumps), `event_id` increments, and the closing event is **committed**: one summary
  episode (mean of the event's keys and values, strength = max of its members) is written to the
  long-term store. Episodes are then remembered as events as well as turns, with the boost at
  the boundary that the fMRI work shows.

### (D) Forgetting and the spacing effect (finding 12)

Replace "lowest coverage" as the only eviction signal with retrievability:

```
R = s · exp(−Δt / τ)         Δt = now − last access
on recall/replay: s += δ;  if Δt ≥ Δt_min: τ ← τ·(1 + κ_space)   // spaced access lasts longer
evict from the live buffer: lowest  w_R·R + w_c·coverage   (coverage still guards within-turn)
delete from the long-term store: R < R_floor AND (consolidation ≥ 1 OR replays ≥ N_max)
```

The deletion rule bounds the long-term store and closes TD-397. An episode is only forgotten
once it is too weak to matter *and* either learned by the model or given up on.

### (E) Cue-based recall from the long-term store (findings 3, 4)

Replace LIFO `recall_from_swap()` with `recall(cue, k, τ_recall, scope)`:

- The swap file becomes the long-term store. It keeps a separate in-memory index of
  `d_idx = 128` projected keys plus file offsets (about 0.5 KB per episode, so 100,000 episodes
  ≈ 50 MB). Candidates are found by brute-force cosine, which is fine at this scale; switch to
  an ANN index only if measurement says so. They are then re-ranked with full keys read from
  disk.
- Called once per user turn, before generation, with `cue = k_q`. The top-k above `τ_recall`
  (default k = 4) are brought into the live buffer at their stored strength. This is the
  hippocampus completing a past episode from a partial cue and making it available to the
  attention step.
- The store gets a versioned header and records, written via temp + rename like
  `save()` (TD-250), with checked reads using `adai::binio` (TD-423).

### (F) Offline consolidation by prioritized replay (findings 1, 9, 10)

This is the only part that changes model weights, and it is opt-in on two levels:
`HIPPOCAMPAL_STORE_TEXT` and `HIPPOCAMPAL_CONSOLIDATION_ENABLED`, both default false. It turns
user conversations into training data, so whether and how to do this at all is **an owner
decision** (privacy and data retention).

- **Replay selection:** `chatbot_api_server` periodically (aligned with
  `HIPPOCAMPAL_SAVE_INTERVAL_SECONDS`) scores long-term episodes by
  `priority = R × gain × (1 − consolidation)`. `gain` is the model's current per-token loss on
  the episode's (user, response) pair, measured off the request path. Weakly learned episodes
  replay first, per the human replay data and Mattar & Daw's gain term.
- **Interleaving (CLS):** selected episodes are written as standard JSONL pairs to a spool and
  uploaded to the model's own `chatbot` pool on `registry_server`, assigned to the model
  (existing `upload`/`assign`). They are trained by the ordinary `trainer_service` pass
  *alongside* the pool's existing data, never alone, so new episodes are interleaved with old
  knowledge and don't overwrite it. The trainer and MNS stay authoritative for runs and data, as
  the remote-first rule requires. No new training path is introduced.
- **Schema-dependent speed (Tse):** after the pass, `gain` is re-measured when the model reloads.
  An episode whose loss is already low (consistent with what the model knows) gets
  `consolidation = 1` after one replay. A schema-inconsistent one needs several replays across
  passes. Consolidated episodes lose strength faster (D), so the episodic trace fades as the
  knowledge moves into the weights: systems consolidation.

### (G) The memory's own encoder, and injection into the decoder stack like the world model (findings 2, 3, 4)

In the brain, the hippocampus doesn't borrow cortex's representation as is: entorhinal cortex
and the dentate gyrus re-encode cortical input into the hippocampus's own code, and CA1 and the
subiculum project the retrieved trace back into cortex. (G) gives `HippocampalMemory` the same
shape. It owns its encoder (the entorhinal/DG side) and its read-out goes into the decoder
stack through its own gated cross-attention layers (the CA1-to-cortex side), placed and tuned
independently of the world model.

**(G.1) A memory encoder owned by `HippocampalMemory`.**

```cpp
class HippocampalMemory {
    std::unique_ptr<HippocampalEncoder> encoder_;  // owned; frozen while serving
  public:
    EncodedEpisode encode(const std::string& text) const;  // segments [m, d_model] + pooled key
    void write_turn(const std::string& user, const std::string& response);  // encodes itself
    ...
};
```

- `HippocampalEncoder` is a transformer encoder at the chatbot's `d_model`, so its output can be
  attended to with no projection. It is implemented on the existing classes: `LLMEncoder`
  weights plus the `LeJEPAEncoder` training machinery (`Predictor`, SIGReg, `gpu_encode()`), with
  no third copy of encoder code. A fixed, seeded DG expansion head (`d_model → 4·d_model`, top-k
  winner-take-all; the optional step in (B)) sits on top, used only for the match and novelty
  decisions.
- **Where its weights come from, in order:**
  1. its own checkpoint, when one is linked (below);
  2. a frozen copy of the chatbot's own trained encoder (`EncoderDecoderModel::copy_encoder()`,
     the same route RAG uses since TD-247). This is the default bootstrap: it is trained, its
     space is already the one the decoder's cross-attention reads, and it needs no extra
     training step;
  3. the world model's encoder (today's behaviour), only if configured explicitly.
- **Training its own weights** (later, optional; gated on §5 showing (2) is the bottleneck):
  `incremental_trainer --objective=memory_encoder`, a sibling of `--objective=lejepa` with its
  own pass under `trainer_service` (TD-403). The loss is SIGReg (isotropic, non-collapsed
  embeddings, which nearest-neighbour recall wants) plus an InfoNCE term that pulls a user
  turn's pooled key toward its own response and away from other responses in the batch. That is
  the cue → outcome association recall depends on. The data is the ordinary conversation pairs
  of the `chatbot` pool; no new pool is needed.
- **Persistence and identity.** The encoder checkpoint lives with the memory, at
  `<HIPPOCAMPAL_STATE_PATH>.encoder/`, and is separate from the episode state `save()` writes. It
  is registered in MNS as an `encoder`-kind record (TD-196) and linked from the chatbot's
  `connection` as `hippocampal_encoder_name`, resolved the same way a linked world model's
  artifact path is. Unlinked means bootstrap (2). `mns_server` rejects a `d_model` mismatch with
  `409`, the same check `link-world-model` makes.
- **Encoder changes invalidate stored keys.** Every stored episode records the encoder id it was
  encoded with. When the memory loads with a different encoder, episodes with stored text are
  re-encoded in the background, and episodes without text are dropped from recall.
- The write path no longer depends on a world model: `maybe_write_hippocampal_memory()`'s
  `!world_model` early return goes away.

**(G.2) Injection into the decoder stack like the world model.**

The world model hands the decoder a token-level sequence plus a mask (`world_model_output`,
`world_model_mask`), read through zero-initialised gated cross-attention in every n-th
`DecoderBlock`. The memory gets the same contract:

- **Its own placement knob.** `HIPPOCAMPAL_INJECT_EVERY_N_LAYERS` (plus
  `HIPPOCAMPAL_INJECT_OFFSET`) replaces TD-186's reuse of `world_model_inject_every_n_layers` in
  `LLMDecoder`'s constructor (`enable_hippocampal = i % n == offset`). The default `-1` means
  "follow the world-model knob", so every existing checkpoint, whose gated hippocampal layers
  sit exactly where the world-model ones do, keeps loading unchanged. The knob decides which
  blocks have parameters, so it shapes the checkpoint: it goes into the MNS connection standard
  (`hippocampal_inject_every_n_layers`) and is fixed for a checkpoint. Typical placements
  interleave the two paths (world model on even blocks, memory on odd ones) or put the memory
  only in the upper half, nearest the output.
- **A token-level memory sequence, not one row per episode.** Each episode is stored as
  `m ≤ HIPPOCAMPAL_SEGMENTS_PER_EPISODE` (default 8) segment vectors: the encoder's token
  outputs mean-pooled over contiguous spans, plus the pooled key used for indexing. At 768 dims
  that is about 25 KB per episode, against about 200 KB for every token. Before each turn,
  `HippocampalMemory::materialize()` builds the read-out from the episodes the live buffer holds
  after recall (E): a `[Σ m_i, d_model]` matrix and a mask, which is the world model's own
  contract. A learned type embedding (one `d_model` row) and the segment's position inside its
  episode are added, so the decoder can tell episodes apart.
- **Bias and bookkeeping stay per episode.** The coverage, association and context terms from
  (B)–(D) are computed per episode and broadcast over that episode's rows in `score_bias`.
  `commit_attention()` sums an episode's rows' weights before updating coverage. The mask keeps
  padding rows out of the softmax.
- **GPU.** The materialised memory is uploaded once per turn and stays on the device for the
  whole decode, like the world-model output (TD-342). The encoder runs on the device through
  `gpu_encode()`. That is one upload per turn and none per token, which keeps to the GPU
  training rule.

**(G.3) Training the injection path so the gates open.** The hippocampal gates have been
trainable since TD-341, but `ChatbotTrainer` switches memory writes off while training (TD-343),
so the gated path only ever sees an empty or unrelated memory and learns to stay shut.
Memory-augmented training fixes that. For each training sample, the trainer fills a scratch
`HippocampalMemory` with (a) the sample's own earlier turns, when the JSONL carries a
`conversation_id`, and (b) `k` neighbours retrieved from the pass's other samples by the memory
encoder, with unrelated distractors mixed in. The decoder is then trained with that memory
injected. The gate learns to open when the memory holds something useful and to stay shut when
it holds only distractors. This is the recipe retrieval-augmented language models use, and it is
the only way the injection path can learn anything before a serving deployment has memories to
read. The encoder stays frozen here; only the gated cross-attention, its norms and the type
embedding train, on the device like the rest of the pass.

## 4. Configuration

New keys live in `config.chatbot.conf`, and later in the MNS world-model connection standard
(TD-196) alongside the existing `HIPPOCAMPAL_*` keys. Every mechanism is off or neutral by
default:

| Key | Default | Mechanism |
|---|---|---|
| `HIPPOCAMPAL_SCOPE` | `session` | A.1 |
| `HIPPOCAMPAL_ENCODING_GATE` | `false` | B (`τ_same`, `τ_sep`, `λ`, `s_min` as `HIPPOCAMPAL_SAME_THRESHOLD`, …) |
| `HIPPOCAMPAL_CONTEXT_ALPHA` | `0.0` | C (`γ_ctx`; `_DRIFT_RHO`, `_DRIFT_BETA`, `_BOUNDARY_THRESHOLD`) |
| `HIPPOCAMPAL_FORGETTING` | `false` | D (`_TAU_HOURS`, `_SPACING_GAIN`, `_RETRIEVABILITY_FLOOR`) |
| `HIPPOCAMPAL_RECALL_TOP_K` | `0` | E (`_RECALL_THRESHOLD`) |
| `HIPPOCAMPAL_STORE_TEXT`, `HIPPOCAMPAL_CONSOLIDATION_ENABLED` | `false` | F |
| `HIPPOCAMPAL_ENCODER` | `chatbot` | G.1: `chatbot` (copy of the model's encoder), `linked` (MNS `hippocampal_encoder_name`), or `world_model` (today's keys) |
| `HIPPOCAMPAL_INJECT_EVERY_N_LAYERS`, `HIPPOCAMPAL_INJECT_OFFSET` | `-1`, `0` | G.2: `-1` follows `world_model_inject_every_n_layers`; shapes the checkpoint, so MNS connection standard |
| `HIPPOCAMPAL_SEGMENTS_PER_EPISODE` | `8` | G.2 |
| `MEMORY_AUGMENTED_TRAINING`, `MEMORY_AUGMENTED_NEIGHBOURS` | `false`, `4` | G.3, `config.trainer.conf` |

## 5. Evaluation

The gated path is still unproven: TD-341 made the gates trainable only on October 20, 2026,
and no TD-358-era retrained checkpoint exists yet. Each mechanism should therefore show value
on a cheap, synthetic benchmark before it is wired into serving.

- **Recall benchmark** (new test binary, CPU, no trained model needed beyond a small LeJEPA):
  scripted multi-session conversations containing facts introduced once. Measure retrieval
  hit-rate (is the right episode among the top attention mass / recalled set) for (a) today's
  buffer, (b) the memory-owned encoder (G.1) against world-model keys, (c) +B, (d) +C, (e) +E. Include near-duplicate distractor facts, so pattern separation
  can show an effect (or not), and long gaps, so forgetting and recall can.
- **Psychology sanity checks** as unit tests: lag-CRP curve (contiguity) from (C), with
  forward asymmetry after reinstatement; retention vs. spaced vs. massed access curves from (D).
  These test that the mechanism behaves as specified, not that it helps.
- **Leakage test** for A.1: an episode written in session X never gets attention mass in
  session Y under `session` scope.
- **Injection (G.2/G.3):** after memory-augmented training, the per-layer `tanh(gate_h)`
  (already logged every epoch since TD-341) moves away from 0, and loss on samples whose answer
  is only in the injected memory is lower with the memory than with it emptied. A gate that
  stays at 0 means the path learned nothing, whatever the recall numbers say.
- **Consolidation (F):** after one replay pass, per-episode loss falls on replayed episodes
  while loss on a held-out slice of the pool doesn't rise (interference check).

## 6. Implementation phases

| # | Chunk | Depends on | Notes |
|---|---|---|---|
| HM2-0 | Per-session memory scope (A.1) + leakage test | — | Do first; fixes a live correctness/privacy issue independent of the rest; file as a TD |
| HM2-1 | `Episode` record, key/value split, turn-level write hook, versioned state + store format | HM2-0 | `@adai-version` 0.8 → 0.9; old state files load with defaults |
| HM2-2 | Memory-owned encoder (G.1): `HippocampalEncoder`, bootstrap from `copy_encoder()`, encoder id per episode, write path independent of the world model | HM2-1 | Before HM2-4..6: their thresholds are calibrated in this encoder's key space |
| HM2-3 | Own injection knob + segment read-out + mask (G.2), CPU and GPU | HM2-2 | `LLMDecoder`, `DecoderBlock`, MNS connection field; `-1` default keeps every checkpoint loading |
| HM2-4 | Encoding gate (B) incl. reconsolidation window | HM2-2 | Predictor-error salience can follow as HM2-4b |
| HM2-5 | Temporal context + boundaries (C) | HM2-3 | One new `score_bias` term, broadcast per episode; CPU and GPU share `hippocampal_score_bias()` |
| HM2-6 | Long-term store index + cue recall (E), forgetting (D) | HM2-2 | Closes TD-397; retires LIFO `recall_from_swap()` |
| HM2-7 | Memory-augmented training (G.3) | HM2-3 | `ChatbotTrainer`; the gated path's first real chance to learn |
| HM2-8 | Recall benchmark + psychology unit tests + gate-opening check | HM2-4..7 | Go/no-go per mechanism |
| HM2-9 | Memory-encoder objective (`--objective=memory_encoder`) + MNS link | HM2-8 | Only if HM2-8 shows the bootstrap encoder limits recall |
| HM2-10 | Replay/consolidation (F) | HM2-6, owner decision | Uses existing registry upload/assign and `trainer_service`; no new training code path |

HM2-4, HM2-5 and HM2-6 touch different parts of `HippocampalMemory` but share its header and
state format. Land them one at a time, not in parallel. HM2-3 and HM2-7 change checkpoint
contents for models that turn them on, so they belong before the TD-358 retraining if that
retraining is to include the memory path.

## 7. Open questions for the owner

1. **Consolidation of user conversations into weights (F):** allowed at all? With which
   retention period, and with what opt-in at the API level?
2. **Scope:** is `session` the right default, or is there a user/tenant identity on the roadmap
   that a long-term store should key on instead? (`chatbot_api_server` has only server-issued
   session ids, TD-232.)
3. **Association layer (TD-194) vs. context reinstatement (C):** keep both, or retire TD-194 if
   the benchmark shows (C) covers it?
4. **Sparse DG code (B, optional):** worth the extra memory once the cosine-based gate is
   measured?
5. **Memory encoder (G.1):** start from a frozen copy of the chatbot's encoder (proposed), or
   train a dedicated one from the start? A dedicated one adds an MNS record, a training
   objective and a checkpoint to keep in step.
6. **Injection placement (G.2):** which blocks should read memory by default once it has its own
   knob: the same blocks as the world model (today), interleaved with it, or the upper half
   only?

## 8. Sources

- McClelland, McNaughton & O'Reilly (1995), *Psychological Review* — CLS. Kumaran, Hassabis &
  McClelland (2016), "What Learning Systems do Intelligent Agents Need?", *TiCS* 20:512–534 —
  [PDF](https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf)
- Knierim & Neunuebel (2015/2016), pattern separation/completion in DG/CA3 —
  [PMC4792674](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792674);
  human CA3 imaging review (2014) — [PMC3941178](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3941178/)
- Ramsauer et al. (2020), "Hopfield Networks is All You Need" — [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)
- Lisman & Grace (2005), "The Hippocampal-VTA Loop", *Neuron* 46:703–713 —
  [PDF](https://itb.biologie.hu-berlin.de/~kempter/Hippocampus_Journal_Club/Articles/lisman05.pdf);
  locus coeruleus novelty effect (2016) — [Nature](https://link.springer.com/10.1038/nature19325)
- Synaptic/behavioral tagging (Frey & Morris) — [Frontiers review](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2016.00206/pdf);
  tagging in reconsolidation (2023) — [Commun. Biol.](https://link.springer.com/10.1038/s42003-023-05666-5)
- Howard & Kahana (2002), Temporal Context Model —
  [PDF](https://people.bu.edu/marc777/docs/HowardKahana02-JMP.pdf);
  Sederberg, Howard & Kahana (2008) — [PMC2585999](https://pmc.ncbi.nlm.nih.gov/articles/PMC2585999)
- Event boundaries: Ben-Yakov & Henson (2018), Baldassano et al. (2017), reviewed in
  [PMC6788820](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6788820/) and
  [bioRxiv 511782](https://www.biorxiv.org/content/10.1101/511782v1.full-text)
- Mattar & Daw (2018), prioritized memory access —
  [bioRxiv](https://www.biorxiv.org/content/10.1101/225664v4.full.pdf+html);
  human replay prioritizes weakly learned information (2018) — [OUCI](https://ouci.dntb.gov.ua/works/4Y3BBOP9)
- Tse et al. (2007), "Schemas and memory consolidation", *Science* 316:76–82 —
  [PubMed 17412951](https://pubmed.ncbi.nlm.nih.gov/17412951/)
- Nader, Schafe & LeDoux (2000), reconsolidation — reviewed in
  [Learn. Mem. 13:536](https://learnmem.cshlp.org/content/13/5/536.abstract)
- Not re-checked in this pass (standard references): Teyler & DiScenna (1986) indexing theory;
  Ebbinghaus (1885); Roediger & Karpicke (2006) testing effect; Polyn, Norman & Kahana (2009) CMR.
