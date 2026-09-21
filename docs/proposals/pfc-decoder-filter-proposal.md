# Proposal: Prefrontal-Cortex-Inspired Filtering Layer for LLM Decoding

## 1. Motivation

The human prefrontal cortex (PFC) implements several well-characterized computational
mechanisms for cognitive control — inhibiting inappropriate responses, gating what
enters working memory, allocating control effort adaptively, and resolving conflict
between competing options. These map surprisingly well onto open problems in LLM
decoding: constrained generation, dynamic context weighting, adaptive sampling
strength, and output reranking.

This proposal outlines a modular "PFC filter" architecture for a decoder pipeline,
where each module is inspired by a specific, formalized neuroscience model rather than
a loose metaphor. Each module below can be implemented and tested independently.

## 2. Proposed Modules

### 2.1 Response Inhibition Module
**Biological basis:** Right inferior frontal gyrus / PFC circuits suppressing
prepotent but inappropriate responses.
**Decoding analog:** Selective, context-conditioned suppression — not a uniform
repetition penalty, but a learned or rule-based gate that identifies "prepotent"
(high-probability but inappropriate) tokens and suppresses them specifically.
**Implementation sketch:** A lightweight classifier or logit-bias head trained to
flag high-probability-but-undesirable continuations; applied as a targeted negative
bias rather than global penalty.

### 2.2 Working Memory Gate (PBWM-inspired)
**Biological basis:** O'Reilly & Frank's Prefrontal Cortex–Basal Ganglia Working
Memory (PBWM) model — a basal-ganglia "gate" controls what new information is
written into PFC buffers vs. protected from overwrite.
**Decoding analog:** A gating network that decides, at each step, which parts of the
context window should be weighted heavily (written into "active memory") vs. held
constant vs. ignored.
**Implementation sketch:** An auxiliary gating network (could reuse attention scores
or a small trained head) producing a per-token "write/protect/ignore" signal that
modulates attention weighting before the final logits are computed.

### 2.3 Expected Value of Control (EVC) Module
**Biological basis:** Shenhav, Botvinick & Cohen's EVC framework — control effort is
allocated proportionally to expected payoff vs. cost.
**Decoding analog:** Adaptive filter *strength*. Instead of a fixed filtering
intensity, the system estimates the "stakes" of the current generation step (e.g.,
factual claim, safety-relevant content, high ambiguity) and scales filtering strength
accordingly.
**Implementation sketch:** A cost/benefit estimator (could be a small classifier on
hidden states) outputs a scalar controlling how aggressively other modules intervene.

### 2.4 Conflict Monitoring Module
**Biological basis:** Anterior cingulate cortex–PFC conflict detection — signals for
more control when competing responses have similar value.
**Decoding analog:** Entropy/margin-triggered adjustment — when the top-k logit
distribution is flat (high conflict/ambiguity), trigger stronger filtering,
resampling, or fallback to a safer decoding strategy.
**Implementation sketch:** Compute entropy or top-1/top-2 margin at each step; below
a threshold, escalate to the EVC module or trigger reranking.

### 2.5 Value-Based Reranking Module
**Biological basis:** Ventromedial PFC / orbitofrontal cortex assigning subjective
value to options for choice.
**Decoding analog:** A reward/value head that reranks candidate completions
(beam search or sampled candidates) by learned value rather than raw likelihood.
**Implementation sketch:** Standard reward-model reranking, but explicitly framed and
tuned as the "value assignment" stage distinct from the inhibition and gating stages.

### 2.6 Adaptive Gain Module
**Biological basis:** Aston-Jones & Cohen's locus coeruleus–PFC adaptive gain theory
— tonic vs. phasic norepinephrine modulates exploit (focused) vs. explore (diffuse)
modes.
**Decoding analog:** Dynamic temperature/top-p control, adjusted per-step or
per-segment based on task phase or uncertainty signals (possibly fed by the conflict
monitoring module).
**Implementation sketch:** Temperature as a function of recent entropy trend and
task-phase signal, rather than a static hyperparameter.

### 2.7 Hierarchical Goal Conditioning
**Biological basis:** Badre & D'Esposito's rostral-caudal PFC hierarchy — abstract
goals at higher (rostral) levels constrain concrete actions at lower (caudal) levels.
**Decoding analog:** Nested conditioning where high-level goal/system-level
representations constrain what the token-level filters (2.1–2.6) are permitted to do,
similar to system-prompt-level conditioning constraining turn-level decoding.
**Implementation sketch:** A goal-state vector (derived from system prompt / task
context) passed as conditioning input to each of the other modules, acting as a
top-down constraint.

## 3. Suggested Development Phases

1. **Phase 1 — Instrumentation:** Add entropy/margin logging and hidden-state
   extraction hooks to the decoding loop (supports modules 2.3, 2.4, 2.6).
   Since 2.3's EVC estimator is learned from the start (see 4.3), this phase
   also needs to capture whatever proxy training signal it will bootstrap
   from (Tier 0 entropy/margin as a weak initial label, plus any downstream
   outcome signal — factual error, safety flag, human rating — that can be
   attached after the fact) rather than only logging for human inspection.
2. **Phase 2 — Conflict + Gain:** Implement conflict monitoring and adaptive gain as
   the first end-to-end loop, since they only require step-wise statistics, not new
   trained components.
3. **Phase 3 — Gating:** Implement the PBWM-inspired working memory gate on top of
   existing attention mechanisms.
4. **Phase 4 — Learned components:** Train the inhibition classifier, the EVC
   cost/benefit estimator itself, and the value reranking head — EVC is no
   longer assumed pre-existing/rule-based by this phase (see 4.3); it uses
   whatever proxy signal Phase 1 captured, then controls the other two
   components' intervention strength once trained.
5. **Phase 5 — Hierarchical conditioning:** Wire in goal-state conditioning across
   all modules and evaluate end-to-end.

## 4. Open Questions for the Dev Session

- Should modules 2.1–2.6 operate at the logit level, hidden-state level, or both?
  See 4.1 below for a detailed comparison.
- Is per-step computation (entropy, margin) cheap enough to run at every decode step,
  or should it be sampled/amortized? See 4.2 below — the answer is that most of
  this is expensive enough that it should be, but not uniformly across modules.
- Should the EVC "cost/benefit" estimator be hand-specified (rule-based) initially,
  or learned from the start? Decided: learned from the start, not rule-based —
  see 4.3.
- What's the evaluation harness — task-based benchmarks, human eval, or synthetic
  ambiguity/conflict test sets designed to stress-test modules 2.4 and 2.6
  specifically? Decided: all three, not a single choice — see 4.4.

### 4.1 Logit-Level vs. Hidden-State-Level Intervention

Two fundamentally different points in the decode step's forward pass to
intervene at. "Logit level" means operating on the final vocabulary-sized
score vector produced by the LM head, after the last hidden state has
already been projected out of the model — the same object a repetition
penalty, `logit_bias`, or top-k/top-p filter already operates on. "Hidden
state level" means operating on the model's internal representations
before that projection: layer outputs, the residual stream, or attention
weights/KV-cache entries themselves.

| | Logit level | Hidden-state level |
|---|---|---|
| **Cost per step** | Cheap — O(vocab_size) vector op, no extra forward-pass hooks | Expensive — needs access into the forward pass itself, plus compute/memory for any classifier or gate operating on high-dimensional (`d_model`-sized) internal state |
| **Signal richness** | Limited to what already collapsed into per-token scores — no access to *why* a token scored highly | Rich — attention patterns, discourse/context representations, and "intent" that never survives the projection to vocabulary space |
| **Architecture coupling** | None — portable across any model that exposes logits, including closed/API-served models (this is exactly how OpenAI's `logit_bias` and most existing guided-decoding libraries work) | Tight — tied to this model's specific layer shapes/`d_model`; a change like TD-059's per-head attention fix would silently invalidate any hook built against the prior math |
| **Auditability** | High — "token X suppressed by −2.0 via inhibition rule Y" is directly explainable to an operator, in the same spirit as this project's `ConfirmActionDialog` previews showing the literal effect before it happens | Low — an internal-state nudge's effect on the eventual output is indirect; "why did the model avoid saying X" has no clean one-line explanation |
| **Composability** | Easy — multiple filters are just vector adds/multiplies into the same distribution, order-dependent but mechanically simple to stack and ablate individually | Harder — interventions can interact with the model's own learned dynamics in ways that are difficult to isolate or turn off independently |
| **Reach** | Only the immediate next-token choice — too late to shape multi-token planning or what gets written into attention/context for future steps | Can influence downstream generation too — modifying a hidden state before it's cached shapes subsequent decode steps, not just the current one |
| **Stability risk** | Low — a bounded logit-space nudge keeps the model sampling from token probabilities it was actually trained to produce | Higher — directly edited internal representations can push the model into out-of-distribution internal states it never learned to recover from |
| **Fidelity to the PFC analogy** | Weaker — there is no real neural analog to "a finished vocabulary-sized action list computed once per step"; PFC circuits act on ongoing internal control states, not a final output stage | Stronger — attention/context gating (PBWM) and abstraction-level goal conditioning are inherently about shaping internal representations, not reweighting a finished output |

**Per-module fit, given the above:**

- **2.1 Response inhibition, 2.4 conflict monitoring, 2.6 adaptive gain** —
  all three are naturally logit-level. Their existing closest prior art
  (Section 6.1, 6.4, 6.6) is uniformly implemented at this level (entropy
  computed from the logit distribution, temperature applied to it,
  suppression applied to it), and there's no strong reason to pay the
  hidden-state cost for them.
- **2.2 Working memory gate** — this is the one module where logit-level
  implementation is a real compromise, not just a cost/benefit tradeoff:
  PBWM's actual mechanism is gating *what enters attention*, which has
  already happened by the time logits exist. A logit-level version can only
  approximate this indirectly (e.g., suppressing tokens associated with
  context the gate wants "forgotten"). The closest existing implementation
  found (`flexible_pfc_hippocampal_arxiv_2503.02303.pdf`, Section 6.2)
  operates on retrieved-memory vectors — hidden-state level — supporting
  building 2.2 there despite the added cost/coupling.
- **2.3 EVC, 2.5 value-based reranking** — level-agnostic in principle
  (EVC's cost/benefit estimator just needs *some* difficulty signal; reward
  models already commonly score either full sequences or hidden states).
  Recommend starting these at whichever level their input is cheapest to
  obtain — logit-derived entropy for EVC's initial difficulty signal, and
  full-sequence text for reranking, exactly as existing reward-model
  reranking already does — and revisiting hidden-state access only if the
  simpler signal proves insufficient.
- **2.7 Hierarchical goal conditioning** — inherently hidden-state level:
  a goal-state vector is described as *conditioning input to each of the
  other modules*, which for the logit-level modules above means influencing
  their inputs (still relatively cheap) but for 2.2 means participating in
  the same hidden-state machinery already required there.

Net recommendation: default to logit-level for 2.1/2.4/2.6 (matches
existing prior art, cheapest, most auditable), accept hidden-state-level
for 2.2 as unavoidable given what PBWM gating actually is, and treat
2.3/2.5/2.7 as needing whichever level is already available to them rather
than a new intervention point purpose-built for this proposal.

### 4.2 Per-Step Computation Cost — Sampling and Amortization Strategy

Running every module's own computation at every single decode step is
expensive enough, across a 7-module pipeline, that it should not be the
default — but the right approach is not to amortize uniformly. Some of
this cost is already effectively free; the rest varies by how fast the
underlying signal actually needs to change, and the answer differs sharply
per module.

**What's already free.** Entropy and top-1/top-2 margin (module 2.4's own
trigger signal) are computed directly from the softmax distribution over
logits — a distribution the decode loop has to produce anyway to sample the
next token. Computing entropy/margin from that already-materialized
distribution is a near-zero marginal cost, not a new expensive operation.
This should run every step unconditionally; there's no fidelity/cost
tradeoff to make here at all.

**What's actually expensive, and where amortizing is safe.** The real cost
lives in the *learned auxiliary components* module 2.4's trigger would
invoke — the inhibition classifier (2.1), a gating network operating on
hidden states (2.2), an EVC cost/benefit estimator (2.3), and reward-model
reranking (2.5). These require extra forward passes, extra parameters, or
(for 2.2) hidden-state access, and running them unconditionally at every
step is the cost this question is really about. Whether amortizing one of
these loses meaningful fidelity depends on how fast its underlying signal
actually moves:

- **2.4 conflict monitoring and 2.1 inhibition** — event-triggered by
  nature, not periodic. A prepotent-token error or an entropy spike happens
  at a specific step; amortizing the *trigger check* (skipping some steps'
  entropy computation) would risk missing the exact event these modules
  exist to catch. Fortunately this doesn't need amortizing at all, since
  the trigger is already free (see above) — only the *response* (the
  actual classifier/reranking call) needs to be gated to fire only when
  triggered, which is itself a form of amortization: expensive computation
  happens only on the minority of steps where the cheap signal crosses
  threshold, not on every step regardless.
- **2.6 adaptive gain and 2.3 EVC** — these represent slower-moving "mode"
  signals by design, not per-token decisions. This is literally what the
  adaptive-gain analogy is about: tonic (slow, sustained) vs. phasic (fast,
  transient) modes, not a value that's meaningful to recompute at every
  single token. Holding the current temperature/gain setting and the
  current EVC stakes estimate constant for a window of N tokens (or until
  the conflict-monitoring trigger above fires an override) should not
  meaningfully affect quality — the biological analogy itself argues
  against recomputing these every step.
- **2.2 working memory gate** — more token-sensitive than 2.3/2.6 (which
  context to write/protect can genuinely change from token to token) but
  still coarser than per-token in practice — natural amortization boundary
  is a phrase/clause, not a fixed token count, recomputed on punctuation or
  attention-shift heuristics rather than a timer.
- **2.7 hierarchical goal conditioning** — the slowest-changing signal in
  the whole architecture by construction (a goal-state vector derived from
  the system prompt/task context). This should be computed once per
  generation (or once per major structural boundary, e.g. a new paragraph)
  and otherwise held fixed — recomputing it per-token would contradict its
  own design as a top-down, session-level constraint.

**Recommended tiering**, from cheapest/most-frequent to most-amortized:

| Tier | Cadence | What runs | Modules |
|---|---|---|---|
| 0 | Every step (already free) | Entropy/margin from the existing softmax | 2.4's trigger signal |
| 1 | Event-triggered (only when Tier 0 crosses threshold) | Inhibition classifier, reranking, hidden-state gating query | 2.1, 2.4's escalation response |
| 2 | Periodic — every N tokens, or on a conflict-triggered override | Refresh temperature/gain, EVC stakes estimate, working-memory gate state | 2.6, 2.3, 2.2 |
| 3 | Once per generation / major structural boundary | Compute goal-state vector | 2.7 |

This should be validated empirically rather than taken as final — N for
Tier 2 and the phrase/clause boundary heuristic for 2.2 are both
implementation-phase tuning parameters, and the synthetic ambiguity/conflict
test sets mentioned in the evaluation open question below are the right
place to check whether a given amortization window actually degrades
output quality before committing to it.

### 4.3 EVC Estimator: Learned From the Start

Decided: the EVC cost/benefit estimator is a learned process from the
start, not a hand-specified rule-based stage that gets replaced later.

**Why not rule-based first:** hand-specifying "stakes" heuristics (factual
claim / safety-relevant content / high ambiguity, as sketched in 2.3) is
exactly the kind of brittle rule set that doesn't generalize across domains
or survive model updates — module 2.1 already preferred a learned approach
over a fixed rule for the same reason. It's also more faithful to the
biological analogy, not less: EVC in the brain isn't a lookup table either,
it's calibrated by outcome history via dopaminergic feedback. A rule-based
v1 would just be thrown away once real training signal exists, so starting
there is strictly more total work than bootstrapping the trainable head
immediately on a weak proxy signal.

**Bootstrap/training signal plan**, reflected in the Phase 1/4 updates in
Section 3 above:

- **Initial weak proxy:** Tier 0 entropy/margin (already computed for free,
  per 4.2) is a reasonable starting signal for "this step looks
  ambiguous/high-stakes" even before richer labels exist.
- **Refinement signal:** downstream outcome data — post-hoc factual-error
  flags, safety flags, human ratings, or module 2.5's own reward-model
  scores — refines or replaces the proxy once available. This creates a
  cross-module dependency worth stating explicitly: 2.5's reward model,
  once built, becomes part of 2.3's own training signal, not merely a
  separate downstream consumer of EVC's output. The phase plan should
  account for this coupling rather than treating 2.3 and 2.5 as
  independently developed.
- **Architecture:** deferred to Phase 4 planning rather than fixed now —
  module 2.3's own "implementation sketch" (a small classifier/regressor on
  hidden states) is a reasonable starting point given the Tier 0 signal and
  whatever hidden-state features Phase 1's instrumentation captures.

### 4.4 Evaluation Harness: All Three, Not a Single Choice

Decided: the evaluation harness needs task-based benchmarks, human eval,
*and* synthetic ambiguity/conflict test sets — not a choice among them.
Each covers a failure mode the other two cannot:

- **Task-based benchmarks** confirm the architecture doesn't regress
  baseline capability while adding control — a real risk given the
  stability concerns flagged for hidden-state intervention in 4.1 and the
  amortization/fidelity tradeoffs in 4.2. This is also what makes the
  architecture comparable to the individual point solutions it's meant to
  unify — AdaDec, EDEN, EDT, and the other prior art in Section 6 all report
  benchmark numbers, and without matching evaluation there's no way to know
  whether the unified architecture is competitive with (or worse than) the
  point solutions it supersedes.
- **Human eval** is necessary because 2.1 (inhibition) and 2.5 (value-based
  reranking) are fundamentally about subjective appropriateness/quality
  judgments that automatic benchmarks can't fully capture — matches how the
  reward-model reranking literature itself (Section 6.5) is calibrated
  against human preference data, not automatic metrics, and should be
  reused as a template.
- **Synthetic ambiguity/conflict test sets** are necessary specifically
  because 2.4 and 2.6 trigger on rare, specific conditions (genuine
  ambiguity, high entropy) that occur too infrequently in natural
  benchmark/human-eval data to give statistically meaningful signal on
  whether the trigger logic itself is correct. This is a targeted
  stress-test of trigger behavior, not a capability measure, and needs
  deliberate construction (noted as real engineering scope, not "reuse an
  existing suite"): prompts with genuinely multiple valid continuations
  (2.4 should fire), prompts with one clearly correct continuation (2.4
  should *not* fire — false-positive rate matters as much as true-positive
  here), and prompts spanning a range of "stakes" levels once 2.3 is
  trained enough to be tested (does EVC modulate 2.6's gain appropriately
  across them). This is also, per 4.2's own closing note, the right place
  to validate whether the Tier 2/3 amortization windows actually degrade
  quality before committing to a specific N.

**Relative cost, and when to run each:** human eval is the most expensive
to run repeatedly and should be reserved for milestone checkpoints (e.g.,
end of each Phase in Section 3) rather than every iteration; task-based
benchmarks and the synthetic conflict sets are both automatable and cheap
enough to run continuously during development.

## 5. Key References for Implementation Context

Local copies of every freely-available paper below have been downloaded to
`downloads/arxiv/` (filenames noted per entry) for offline reading.

### 5.1 Neuroscience Foundations

- O'Reilly, R.C. & Frank, M.J. (2006), *Making Working Memory Work: A
  Computational Model of Learning in the Prefrontal Cortex and Basal Ganglia*,
  Neural Computation — PBWM model (working memory gating).
  `oreilly_frank2006_pbwm.pdf`
- Shenhav, A., Botvinick, M.M., & Cohen, J.D. (2013), *The Expected Value of
  Control: An Integrative Theory of Anterior Cingulate Cortex Function*,
  Neuron. `shenhav2013_expected_value_of_control.pdf`
- Aston-Jones, G. & Cohen, J.D. (2005), *An Integrative Theory of Locus
  Coeruleus-Norepinephrine Function: Adaptive Gain and Optimal Performance*,
  Annual Review of Neuroscience — adaptive gain theory (LC-NE system).
  **Not downloaded — Annual Reviews is subscription-only and no
  author-hosted open-access copy was found; access via institutional
  subscription or DOI 10.1146/annurev.neuro.28.061604.135709.**
- Badre, D. & Desrochers, T. (2019), *Hierarchical Cognitive Control and the
  Frontal Lobes* (book chapter, a more current synthesis than the original
  2009 rostral-caudal paper) — rostral-caudal PFC hierarchy.
  `badre_desrochers2019_hierarchical_cognitive_control.pdf`
- Botvinick, M.M. et al. — Conflict monitoring theory (ACC-PFC). No single
  paper downloaded yet; the 2013 Shenhav EVC paper above supersedes/extends
  much of the original conflict-monitoring account and can stand in for it
  during initial reading.
- Hassabis, D., Kumaran, D., Summerfield, C., & Botvinick, M. (2017),
  *Neuroscience-Inspired Artificial Intelligence*, Neuron — general survey
  connecting cognitive-control neuroscience to AI architectures; useful
  framing context for this whole proposal, not module-specific.
  `hassabis2017_neuroscience_inspired_ai.pdf`

### 5.2 Directly Overlapping ML/NLP Work

See Section 6 for how each of these maps onto (and differs from) the modules
in Section 2.

- Nguyen et al. (2023), *A Prefrontal Cortex-inspired Architecture for
  Planning in Large Language Models* (MAP), arXiv:2310.00194 — Microsoft
  Research. `map_arxiv_2310.00194.pdf`
- *AdaDec: Uncertainty-Guided Adaptive Decoding for LLM-based Code
  Generation*, arXiv:2506.08980. `adadec_arxiv_2506.08980.pdf`
- *Entropy-informed Decoding: Adaptive Information-Driven Branching* (EDEN),
  arXiv:2605.09745. `eden_arxiv_2605.09745.pdf`
- *Entropy Adaptive Decoding: Dynamic Model Switching for Efficient
  Inference*, arXiv:2502.06833.
  `entropy_adaptive_decoding_arxiv_2502.06833.pdf`
- *Learning Adaptive LLM Decoding*, arXiv:2603.09065.
  `learning_adaptive_llm_decoding_arxiv_2603.09065.pdf`
- *Look Inward to Explore Outward: Learning Temperature Policy from LLM
  Internal States via Hierarchical RL*, arXiv:2602.13035.
  `look_inward_explore_outward_arxiv_2602.13035.pdf`
- *Flexible Prefrontal Control over Hippocampal Episodic Memory for
  Goal-Directed Generalization*, arXiv:2503.02303 — basal-ganglia-style
  learned gating over memory retrieval, the closest existing analog to
  Section 2.2's working-memory gate.
  `flexible_pfc_hippocampal_arxiv_2503.02303.pdf`
- *Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language
  Model Alignment*, arXiv:2404.01054.
  `regularized_bestofn_arxiv_2404.01054.pdf`

## 6. Related Work and Cross-Over Analysis

This section maps each Section 2 module against existing work found during a
literature pass (September 2026) and states, explicitly, what would be novel
about this proposal's contribution versus what already exists. The overall
finding: **the individual mechanisms in 2.1–2.6 each have closer ML prior art
than this proposal currently cites; the more defensible novelty is the
unifying PFC-organized architecture itself (Section 2.7's hierarchy tying the
others together), not any single filter in isolation.** This section should
be treated as a required addition before circulating the proposal further —
without it, a reviewer familiar with the decoding literature will likely
flag 2.4–2.6 as already-solved problems.

### 6.1 Response Inhibition (Section 2.1)
**Crossover:** Targeted, classifier-gated suppression of specific
high-probability-but-undesirable tokens is a mature technique family (OpenAI
Chat Completions' `logit_bias` parameter; classifier-guided decoding;
RLHF-alignment refusal-suppression research). No paper downloaded
specifically for this module — the crossover is with established production
technique, not a single citable paper.
**Difference:** The proposal's specific framing — a *learned* "prepotency"
classifier distinct from a static banned-token list, explicitly modeled on
right-IFG stopping circuitry rather than a generic content filter — is a
narrower, more principled version of existing logit-bias approaches, not a
new mechanism.

### 6.2 Working Memory Gate (Section 2.2)
**Crossover:** `flexible_pfc_hippocampal_arxiv_2503.02303.pdf` implements
almost exactly this module's sketch: a learned gating policy, explicitly
modeled on basal-ganglia disinhibition, that outputs a scalar controlling
whether a retrieved memory vector is used. It targets episodic-memory
retrieval in an RL/agent setting rather than attention-weight modulation
over a context window, but the gating mechanism itself is a close match.
**Difference:** This proposal's target (attention-weight modulation across a
context window, not memory-retrieval gating) is different enough in
application that the mechanism would need re-deriving, but the paper's gate
design (a small policy network outputting write/protect/ignore-style scalars
conditioned on a "controller state") should be read before designing 2.2's
own gating network from scratch.

### 6.3 Expected Value of Control (Section 2.3)
**Crossover:** No ML work found that invokes "EVC" by name, but the
underlying idea — allocate compute/intervention strength proportional to
estimated stakes or difficulty — is an active subfield:
`learning_adaptive_llm_decoding_arxiv_2603.09065.pdf` trains decoding
policies conditioned on a compute budget; related budget/cost-aware
test-time-compute work (not downloaded — arXiv:2509.09864, arXiv:2503.24377)
covers the same ground without the neuroscience framing.
**Difference:** Explicitly framing the cost/benefit estimator as an EVC
analog (rather than a generic budget controller) is mostly a matter of
framing; the proposal should cite this cluster of work rather than present
adaptive-strength filtering as unaddressed in the literature.

### 6.4 Conflict Monitoring (Section 2.4)
**Crossover:** This module has the closest and most numerous prior art of
any in the proposal. `adadec_arxiv_2506.08980.pdf` (Shannon-entropy-triggered
pause-then-rerank), `eden_arxiv_2605.09745.pdf` (entropy-gated branching
factor — low-entropy steps stay greedy, high-entropy steps get more
expansions), and `entropy_adaptive_decoding_arxiv_2502.06833.pdf`
(uncertainty-triggered model switching) all implement the
"entropy/margin-triggered escalation" design sketched in 2.4, in some cases
including the exact "escalate to stronger intervention or rerank" behavior
proposed here.
**Difference:** None found at the mechanism level for this module as
currently scoped — before implementing 2.4, these three papers should be
read closely, since the proposal risks re-implementing an existing
technique. Any genuine novelty would need to come from how 2.4's output
feeds the other PFC-framed modules (2.3, 2.6), not from the entropy-gating
mechanism itself.

### 6.5 Value-Based Reranking (Section 2.5)
**Crossover:** Reward-model reranking / Best-of-N is the most well-trodden
technique in the entire proposal (PairRM, GRAM, Minimum Bayes Risk decoding,
and known failure modes like reward hacking — see
`regularized_bestofn_arxiv_2404.01054.pdf` for a representative treatment
and mitigation).
**Difference:** None claimed or claimable at the mechanism level. The
proposal's own text already frames this correctly ("standard reward-model
reranking... tuned as the value assignment stage") — this should stay
explicit in any writeup so a reader doesn't mistake 2.5 for a novel
contribution.

### 6.6 Adaptive Gain (Section 2.6)
**Crossover:** Very close existing work: EDT (Entropy-based Dynamic
Temperature Sampling) and AdapT (Adaptive Temperature Sampling for Code
Generation) both raise temperature under high uncertainty and sharpen it
under confidence — precisely the tonic/phasic exploit-explore modulation
this module describes. `look_inward_explore_outward_arxiv_2602.13035.pdf`
goes further, learning a temperature policy from internal states via
hierarchical RL rather than a hand-specified entropy-to-temperature mapping.
**Difference:** As with 2.4, no mechanism-level novelty found — 2.6 as
currently scoped substantially overlaps with EDT/AdapT.
`look_inward_explore_outward_arxiv_2602.13035.pdf`'s learned-policy approach
is the more sophisticated existing baseline to differentiate against, not
the static entropy-threshold papers.

### 6.7 Hierarchical Goal Conditioning (Section 2.7)
**Crossover:** General hierarchical/constrained-generation work exists but
nothing maps cleanly onto the rostral-caudal abstraction gradient
specifically — this is the module with the least direct ML prior art found.
**Difference:** This is the strongest candidate for the proposal's actual
novel contribution: using the rostral-caudal *hierarchy* (not just "a goal
vector conditions decoding," which is common, but a graded abstraction
hierarchy constraining what lower-level modules 2.1–2.6 are permitted to do)
as the top-level organizing structure for the other six modules. Recommend
leading with this framing in any writeup, rather than treating 2.7 as one
module among seven equals.

### 6.8 The Architecture as a Whole
**Crossover:** `map_arxiv_2310.00194.pdf` ("MAP") is an existing paper with
an almost identical name pattern to this proposal and even uses "conflict
monitoring" as one of its five named modules. Its abstract was checked
directly: MAP's five modules (conflict monitoring, state prediction, state
evaluation, task decomposition, orchestration) are each implemented as
**separate LLM calls** communicating results between calls, targeting
**multi-step planning tasks** (Tower of Hanoi, graph traversal, PlanBench,
StrategyQA).
**Difference:** This is a real, defensible, and easy-to-state distinction:
MAP applies PFC-module separation *between* LLM invocations for multi-step
planning; this proposal applies it *within* a single generation pass at the
logit/hidden-state level for general decoding control. A related-work
paragraph citing MAP and stating this distinction explicitly is needed —
without it, a reviewer familiar with the planning-architectures literature
will assume the proposal missed directly relevant prior work.
