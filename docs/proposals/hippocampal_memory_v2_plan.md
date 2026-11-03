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

Six mechanisms, each tied to a finding above. They are listed in dependency order, and each one
can be turned off on its own with a config key (default off), the same opt-in pattern the
current hippocampal keys use.

```
 user turn ──► world model (LeJEPA, frozen) ──► cue key k_q
                                               │
            ┌──────────────────────────────────┼─────────────────────────────┐
            │ (E) cue recall: long-term store ─► top-k into live buffer       │
            │ (C) temporal context c_t drifts; boundary detector              │
            ▼                                                                 │
  live buffer (per session) ──► score_bias += ctx overlap + strength ──► gated
            ▲                     (existing repetition + association terms)   cross-attn
            │                                                                 (CA3 step)
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

## 5. Evaluation

The gated path is still unproven: TD-341 made the gates trainable only on October 20, 2026,
and no TD-358-era retrained checkpoint exists yet. Each mechanism should therefore show value
on a cheap, synthetic benchmark before it is wired into serving.

- **Recall benchmark** (new test binary, CPU, no trained model needed beyond a small LeJEPA):
  scripted multi-session conversations containing facts introduced once. Measure retrieval
  hit-rate (is the right episode among the top attention mass / recalled set) for (a) today's
  buffer, (b) +B, (c) +C, (d) +E. Include near-duplicate distractor facts, so pattern separation
  can show an effect (or not), and long gaps, so forgetting and recall can.
- **Psychology sanity checks** as unit tests: lag-CRP curve (contiguity) from (C), with
  forward asymmetry after reinstatement; retention vs. spaced vs. massed access curves from (D).
  These test that the mechanism behaves as specified, not that it helps.
- **Leakage test** for A.1: an episode written in session X never gets attention mass in
  session Y under `session` scope.
- **Consolidation (F):** after one replay pass, per-episode loss falls on replayed episodes
  while loss on a held-out slice of the pool doesn't rise (interference check).

## 6. Implementation phases

| # | Chunk | Depends on | Notes |
|---|---|---|---|
| HM2-0 | Per-session memory scope (A.1) + leakage test | — | Do first; fixes a live correctness/privacy issue independent of the rest; file as a TD |
| HM2-1 | `Episode` record, key/value split, turn-level write hook, versioned state + store format | HM2-0 | `@adai-version` 0.8 → 0.9; old state files load with defaults |
| HM2-2 | Encoding gate (B) incl. reconsolidation window | HM2-1 | Predictor-error salience can follow as HM2-2b |
| HM2-3 | Temporal context + boundaries (C) | HM2-1 | One new `score_bias` term; CPU and GPU share `hippocampal_score_bias()` |
| HM2-4 | Long-term store index + cue recall (E), forgetting (D) | HM2-1 | Closes TD-397; retires LIFO `recall_from_swap()` |
| HM2-5 | Recall benchmark + psychology unit tests | HM2-2..4 | Go/no-go per mechanism |
| HM2-6 | Replay/consolidation (F) | HM2-4, owner decision | Uses existing registry upload/assign and `trainer_service`; no new training code path |

HM2-2, HM2-3 and HM2-4 touch different parts of `HippocampalMemory` but share its header and
state format. Land them one at a time after HM2-1, not in parallel.

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
