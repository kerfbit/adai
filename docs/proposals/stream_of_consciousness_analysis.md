# Analysis: Stream-of-Consciousness Research and the Reasoning Process

**Status:** Analysis (November 9, 2026). This is input to
[reasoning_process_plan.md](reasoning_process_plan.md) and is not a separate feature. It reviews
psychology and neuroscience research on the stream of consciousness: how ongoing human thought is
structured, how it moves, what controls it and how it stops. It then asks which parts carry over to
adai's planned `<think>…</think>` phase. It also covers the parts of the design that border on it:
gated cross-attention in `DecoderBlock` (TD-180), `HippocampalMemory` and its v2 plan, and the PFC
filter proposal.

Each finding below is marked by how strongly it should shape adai: **use** (it changes a design
decision), **measure** (cheap to instrument, then decide), or **don't build on** (the evidence is
weak or the analogy breaks).

## 1. Summary

1. **Count thoughts, not just tokens.** Human thought moves in discrete units. James called them
   "perchings" separated by "flights", and fMRI work finds stable brain states separated by
   transitions (Tseng & Poppenk 2020). The reasoning plan budgets and measures only tokens. A
   reasoning span should be split into thoughts. Its budgets, metrics and overthinking detection
   should then work in thoughts, because that is the unit where the documented overthinking
   failure shows up: repeated "solution rounds" (Chen et al. 2024).
2. **Overthinking matches rumination, not mind-wandering.** Christoff et al. (2016) classify
   thought by its constraints: *deliberate* (goal-directed control) and *automatic*
   (salience-driven capture). Rumination is thought held by strong automatic constraint: it keeps
   returning to the same content. That is the looping, 22,000-token deliberation the reasoning plan
   cites. The fix this suggests is to detect *stickiness* (low movement between thoughts, revisits
   to earlier ones) and respond to it. Lowering diversity overall, or relying on a fixed token cap
   alone, is not the fix it points to.
3. **Stopping is a metacognitive judgment.** Mangan's "feeling of rightness" and Nelson & Narens'
   monitor/control model describe a separate signal that says when thinking has done enough.
   adai can approximate it cheaply from quantities the decoder already computes: the probability
   of `</think>`, and how settled the answer-start distribution is. The token budget stays as the
   backstop.
4. **Most human thinking isn't inner speech.** In Descriptive Experience Sampling, inner speech
   is present in about a quarter of sampled moments, about as often as "unsymbolized thinking".
   Model reasoning text is also not a faithful account of how the model reached its answer
   (Turpin et al. 2023). So a reasoning span should be treated as a working buffer, not as an
   explanation. Keep it out of memory and out of user-facing "why" claims by default.
5. **Mixed evidence for the open gate question.** Self-generated human thought draws on the
   episodic-memory (medial temporal) subsystem of the default network. That is mild support for
   letting the *hippocampal* gate open further during reasoning. It is not support for the
   *world-model* gate, which is what the reasoning plan's postulate is about. Separately, the
   "perceptual decoupling" seen in mind-wandering costs task performance in humans, which argues
   against weakening the encoder path during reasoning.
6. **A conflict found while preparing this:** the reasoning plan puts `<think>`/`</think>` at ids 4
   and 5, but [conversation_turn_tokens_plan.md](conversation_turn_tokens_plan.md) reserves ids 4–6
   for `<system>`/`<user>`/`<assistant>`, and those are already in code (TD-295, November 8, 2026:
   `SpecialTokens.hpp`). Since TD-565 new vocabularies reserve ids 7–19 for future special tokens,
   so the think tokens take the next two available, 7 and 8 (decided November 10, 2026; see SC-0).

## 2. What the research says

Sources marked † were checked in this pass (abstracts, author copies or publisher pages). Sources
without the mark are standard references that weren't re-read; in particular, the text of
James's chapter couldn't be fetched from this environment.

### 2.1 The classical description

**F1. James (1890), "The Stream of Thought" (*Principles of Psychology*, ch. IX).** James
characterizes thought as personal (it belongs to one owner), always changing (no state recurs
exactly), sensibly continuous (even across gaps), about objects other than itself, and selective
(it attends to some parts of its objects and ignores others). The stream has *substantive* parts,
the resting places where a thought is held, and *transitive* parts, the movements between them.
James compares this to a bird's life: an alternation of flights and perchings. Transitive parts are
hard to introspect, because by the time attention reaches one, it has already landed. Around every
definite thought there is a *fringe*: a vague sense of where the thought is going, of what would
satisfy it, and of whether it is on the right track.

**F2. Fringe consciousness and the feeling of rightness.** † Mangan (1993, 2000) develops James's
fringe into a specific claim. The feeling of "rightness" is a summary signal of how well current
conscious content fits its unconscious context. It carries no sensory content, and it drives
monitoring and control. "Feeling of knowing" is the same signal under another name. Reber links it
to processing fluency. May (2004) accepts that rightness tracks coherence but warns that treating
it as a quality-of-processing measure risks a homunculus. Both points matter for a learned
"readiness" signal (§4.3): it has to be calibrated against outcomes, not trusted because it exists.

### 2.2 Structure and rate

**F3. Thoughts are discrete units with measurable boundaries.** † Tseng & Poppenk (2020, *Nature
Communications*) detect transitions between thoughts from fMRI *without decoding their content*.
They find stable periods of brain-network configuration ("meta-states") separated by transitions.
During films, transitions line up with narrative event boundaries. At rest, more frequent
transitions go with higher trait neuroticism ("mental noise"). Their estimate is about 6,200
thoughts per day. The authors also note that self-reports of thought are unreliable, which is why
they used an implicit measure.

**F4. Event segmentation.** (Covered in [hippocampal_memory_v2_plan.md](hippocampal_memory_v2_plan.md)
§2, finding 8.) The hippocampus commits an event to memory at its boundary, and boundary-locked
activity predicts later recall. Thought boundaries (F3) and event boundaries are the same kind of
signal, so they can share one detector.

### 2.3 What moves thought and what holds it

**F5. Spontaneous thought as a family defined by constraints.** † Christoff et al. (2016, *Nat.
Rev. Neurosci.* 17:718) define mind-wandering by how thought *moves*, not by whether it is
off-task. Thought is constrained deliberately (cognitive control holding it to a goal) and
automatically (salient material capturing it: emotion, worry, sensory pull). Creative and
goal-directed thought have high deliberate constraint. Dreaming has little of either.
Mind-wandering sits between them. Rumination and obsessive thought have high *automatic*
constraint: they are stuck, not free. Shifts between these modes correspond to changing
interactions among large-scale networks (default, frontoparietal control, salience).

**F6. "Freely moving" is its own dimension.** † Mills, Raffaelli, Irving, Stan & Christoff (2018,
*Consciousness and Cognition* 58:20) find that freedom of movement correlates only weakly with
being off-task. Treating off-task thought as free movement would misclassify over 40% of thoughts.
For a model, being "on the prompt's topic" and "moving productively" are separate measurements.

**F7. Current concerns.** † Klinger's theory: unfinished goals sensitize a person to related cues,
so spontaneous thought clusters on what is unresolved and measures the gap between the current
state and the goal. Priming concerns increases mind-wandering, though the measured effect is small
(about 3–4%). When goal pursuit is blocked, this turns into rumination.

**F8. Decoupling and executive control.** † Smallwood & Schooler (2015, *Annu. Rev. Psychol.*
66:487) describe mind-wandering as attention decoupling from perception toward internally
generated content. Its content comes from episodic and affective processes, and executive control
regulates it. The competing executive-failure account (McVay & Kane) treats it as a lapse of
control. Either way, decoupling costs performance on the external task.

### 2.4 The networks involved

**F9. Default network subsystems.** † Andrews-Hanna, Saxe & Yarkoni (2014, *NeuroImage*): the
default network is a midline core plus two subsystems. A *medial temporal* subsystem (hippocampal
formation, parahippocampal and retrosplenial cortex) is engaged mainly by episodic retrieval and
simulation. A *dorsal medial* subsystem (dmPFC, TPJ, lateral temporal cortex) is engaged mainly
by mentalizing and reflecting on one's own and others' states. Self-generated thought draws on
both.

**F10. Global workspace.** † Dehaene & Naccache (2001, *Cognition* 79:1); Baars (1988). Many
specialized processors run in parallel and unconsciously. Conscious access is a limited-capacity,
largely serial workspace that one representation at a time "ignites" into and is broadcast from to
every processor. † Goyal, Bengio et al. (2021, ICLR 2022) build this into transformers: specialist
modules compete to write into a small shared workspace, which is then broadcast back to all of
them. They argue that the capacity limit helps specialization and coordination.

### 2.5 Inner speech and its limits

**F11. Inner speech: functions and forms.** † Alderson-Day & Fernyhough (2015, *Psychol. Bull.*
141:931). Inner speech supports working-memory rehearsal, planning, self-regulation and
self-monitoring. Developmentally (Vygotsky) it is internalized private speech. Vygotsky's
tradition distinguishes *expanded* inner speech (close to full sentences) from *condensed* inner
speech (abbreviated, "thinking in pure meaning"). Fernyhough's account, as reported in secondary
sources, is that inner speech is condensed by default and expands under stress or cognitive
challenge. That attribution wasn't checked against the original text.

**F12. Most moments of thought aren't verbal.** † Heavey & Hurlburt (2008), Descriptive Experience
Sampling, 30 participants × 10 random moments. Five phenomena each appear in about a quarter of
moments: inner speech, inner seeing, unsymbolized thinking, feelings and sensory awareness.
Individual differences are large. Some people never report inner speech, others in up to 75% of
samples. The median is about 20%.

### 2.6 Metacognition

**F13. Monitoring and control.** † Nelson & Narens (1990): an object level (the cognition itself)
and a meta level that holds a model of it. Monitoring flows up (how close is this to working?) and
control flows down (keep going, switch strategy, stop). The two flows are logically separate.

### 2.7 Incubation (weak evidence)

**F14. Mind-wandering and incubation.** † Baird et al. (2012, *Psychol. Sci.*) reported that an
undemanding task during an incubation break improved creative performance on previously seen
problems, and attributed this to mind-wandering. A preregistered replication (Murray et al. 2024,
two studies, N = 443) found no evidence for the effect. **Don't build on it.**

### 2.8 Machine reasoning work that touches these findings

- † **Quiet-STaR** (Zelikman et al. 2024): learned start/end-of-thought tokens, a rationale
  generated after every token, mixed with the no-thought prediction, and trained with REINFORCE.
  This is the closest machine analogue of a *continuous* stream that runs alongside ordinary text,
  rather than one block before the answer.
- † **Coconut** (Hao et al. 2024): reasoning in continuous hidden states fed back as input
  embeddings. It can keep several candidate next steps at once (breadth-first) and uses fewer
  thinking steps. It needs a staged curriculum that starts from verbal chain-of-thought. This is
  the machine analogue of condensed or unsymbolized thought (F11, F12).
- † **Overthinking** (Chen et al. 2024, "Do NOT Think That Much for 2+3=?"): reasoning models used
  1,953% more tokens than conventional ones on "2+3". Responses usually contain 2–4 "solution
  rounds" (segments that each reach an explicit answer), and *easier* problems get more rounds.
  Self-training toward fewer redundant rounds reduced tokens. The unit of waste is the round,
  which is a thought-level unit, not a token-level one.
- † **Unfaithful chain-of-thought** (Turpin et al. 2023): biasing cues shift a model's answers,
  and its reasoning rationalizes the biased answer without mentioning the cue.

## 3. Where the analogy holds and where it breaks

| Human stream | adai reasoning span | Holds? |
|---|---|---|
| Serial, one content at a time in the workspace (F10) | Autoregressive: one token at a time, each fed back to every layer | **Yes, structurally.** The emitted token is the broadcast; the gated paths (encoder, world model, memory) are specialists competing to influence it |
| Discrete thoughts with boundaries (F1, F3) | A flat token sequence with no unit above the token | **Missing in adai.** Nothing marks a "thought", so nothing can count, compare or commit one |
| Constraint modes: deliberate, automatic, free (F5) | Encoder cross-attention ≈ deliberate goal constraint; repetition and attractor loops ≈ automatic; sampling temperature ≈ freedom | **Partly.** A useful vocabulary for failure modes, not a mechanism to copy |
| Fringe / rightness: a non-verbal sense of fit (F2) | Output distribution shape, `P(</think>)`, hidden-state readouts | **Yes, as an engineering signal.** Unproven as anything more |
| Mostly non-verbal (F12) | Verbal by construction (tokens) | **Breaks.** A verbal trace is a lossy, possibly unfaithful projection (Turpin). Latent reasoning (Coconut) is the closer analogue |
| Inner speech condensed by default, expanded under difficulty (F11) | Effort levels are fixed budgets chosen by the caller | **Suggestive.** Supports effort that adapts to difficulty |
| Personal, private (F1) | `reasoning_content` stored per message | **Policy, not mechanism.** Supports `preserve_thinking=false` and keeping thinking out of shared memory |
| Default-network memory retrieval during self-generated thought (F9) | Hippocampal gated path | **Mild support** for more memory influence during reasoning (§4.5) |
| Incubation helps (F14) | "Think in the background between turns" | **Don't build on it.** The evidence failed replication |

Nothing here argues that adai should be, or will become, conscious, and nothing in this analysis
depends on that question. The research is used for one purpose: it is a well-studied description
of how a serial, self-feeding generator of internal content organizes, constrains and stops itself,
and adai is about to build one.

## 4. Applications to adai

Each item gives the findings it rests on, what it would mean concretely, where it attaches in the
existing code or plans, and a verdict.

### 4.1 Segment the reasoning span into thoughts (F1, F3, F4; Chen et al.) — **use**

**What:** split a reasoning span into thoughts at boundaries and keep per-thought records:
the start and end positions, and a mean-pooled embedding of the thought.

**Boundary detector, in order of cost:**
1. **Lexical:** sentence or line ends within the span, plus discourse markers that start a new
   attempt ("wait", "alternatively", "let me check"). Cheap and good enough for metrics. It only
   works in the tokenizer's normalized space, and in Unicode mode (TD-564) it works per character.
2. **Representational (a Tseng & Poppenk analogue):** a boundary wherever the cosine distance
   between a rolling window's mean decoder hidden state and the previous window's mean exceeds a
   threshold. This needs no content decoding, just like the fMRI method. It is computed on the
   last decoder layer's output, which generation already produces.
3. **Learned:** a learned boundary token in SFT traces. Deferred until data exists.

**Why it matters:** every other item below operates on thoughts. Chen et al.'s overthinking unit,
the "solution round", is a thought that reaches an answer. Counting rounds directly measures the
failure the reasoning plan is trying to prevent.

**Attaches to:** `TextGenerator::generate_with_reasoning()` (RP-2a), which already tracks the
reasoning phase. The boundaries go into a small `ThoughtTrace` returned beside the text. Nothing
changes in the model.

### 4.2 Detect stickiness, not just length (F5, F6, F7) — **measure, then use**

**What:** with thoughts segmented, compute two per-span dynamics measures, kept separate as
Mills et al. require (F6):
- **Movement:** the mean embedding distance between consecutive thoughts. Near zero means the
  stream isn't moving.
- **Revisit rate:** the fraction of thoughts whose embedding is within a threshold of an *earlier,
  non-adjacent* thought. High means looping back: the rumination signature (high automatic
  constraint, F5).
- **Goal relevance**, as a separate axis: the similarity of each thought to the encoder's
  mean-pooled prompt representation. A thought can move freely and still be on task, or be stuck
  and off task.
- **Solution rounds:** thoughts that contain an answer candidate. Lexically this means an
  `=`/"answer is"/"so the result" marker. Later, a thought after which the readiness signal
  (§4.3) spikes.

**Intervention (opt-in, default off):** when the revisit rate over the last *k* thoughts passes a
threshold, or a thought repeats the answer candidate of an earlier round, the generator
force-closes the span with `</think>` exactly as it does when the budget runs out. The token
budget stays the hard backstop. This turns "this model cannot stop thinking" from a budget
problem into a detected state with a cause.

**Why not reduce repetition globally:** Christoff's framework separates productive deliberate
constraint (staying on the goal) from automatic capture. A global repetition penalty pushes
against both. The detector targets only the stuck pattern. This also fits the PFC proposal's
§2.1, which already argues for *targeted* inhibition over a uniform penalty, and its conflict
monitor (§2.4) could share the per-thought embeddings.

**Attaches to:** the reasoning plan's Evaluation standard (new metrics next to the
span-length percentiles it already proposes) and `TrainingMetricsService` / generation-quality
reporting. The intervention is one more stop condition in RP-2a.

### 4.3 A readiness signal for stopping (F2, F13) — **measure, then use**

**What:** a meta-level monitor (Nelson & Narens) that runs at each thought boundary and estimates
whether thinking has done enough.

**Cheapest version (no new parameters):** at each boundary, read two quantities from the logits
the step already computes:
1. `P(</think>)`, the model's own learned propensity to end the phase. SFT on reasoning traces
   teaches it.
2. The answer-start entropy: append `</think>` to a scratch copy of the KV cache (the cache view
   from TD-388 makes this one extra decode step) and measure the entropy of the first answer
   token. If it stays low and the same across consecutive boundaries, the answer has settled.

Stop when both have been past their thresholds for *m* consecutive thoughts. This is a "feeling of
rightness" in Mangan's sense: a summary of fit, with no content of its own.

**Calibrate, don't trust (May 2004):** log the readiness value next to the final answer's quality
score (BLEU or reward). If readiness doesn't predict quality on the validation set, the signal is
noise and stays off. This is the same "readout before premise" rule the reasoning plan applies to
the gate question.

**Learned version (later):** a small probe head on the decoder's last hidden state, trained to
predict whether the answer would change if thinking continued. Labels come for free from SFT
traces by truncating them at each boundary. This is the PFC proposal's EVC estimator (§2.3)
applied to one decision: whether more thinking is worth its cost.

### 4.4 Effort that adapts to difficulty (F11, F13; Chen et al.) — **use, as a training rule**

Fernyhough's account says inner speech is condensed by default and expands under challenge.
Chen et al. found the reverse in reasoning models: the most rounds on the *easiest* problems. adai
can aim for the human pattern in two places:
- **Inference:** `REASONING_EFFORT=auto`, a fourth option beside the fixed levels. The span starts
  with the `low` budget and is extended to the next level only while the readiness signal (§4.3)
  is low and the stickiness detector (§4.2) is quiet. Thinking expands only when it is both
  unsure and still making progress.
- **Training (RP-4 GRPO):** score completions on answer quality minus a cost per *redundant
  solution round* (rounds after the first one that already reached the final answer), not a cost
  per token. A per-token cost punishes legitimate long derivations. A per-redundant-round cost
  targets exactly what Chen et al. measured. This is EVC: control is paid for only when it buys
  something.

### 4.5 The gate question, revisited (F8, F9, F10) — **measure; adjust the pilot**

The reasoning plan asks whether the world-model gate should open further during `<think>`, and
proposes `gate_reasoning`/`gate_answer` pairs to find out. The research adds three things:

1. **The hippocampal path is the better candidate for the postulate.** Self-generated thought
   draws on the medial temporal subsystem (F9), and Smallwood & Schooler place mind-wandering
   content in episodic processes (F8). So the phase-conditioned pair on the *hippocampal* path
   deserves the same readout as the world-model pair. The LeJEPA plan's chunk 12 already builds
   both pairs, so this is a note for the pilot's report, not new work.
2. **Don't phase-condition the encoder path.** Reducing prompt fidelity during reasoning is the
   machine analogue of perceptual decoupling (F8), which reliably costs external-task performance
   in humans. The encoder cross-attention stays unconditioned and mandatory. That is already the
   reasoning plan's design. This gives a reason to keep it.
3. **Commit memory at thought boundaries, not per token.** TD-343 commits hippocampal memory once
   per generated token. Event and thought boundaries (F3, F4) are where human episodic encoding
   happens, and HM2 (C) already proposes event boundaries for memory. Inside a reasoning span,
   committing at thought boundaries (§4.1) instead of per token cuts the number of writes and
   lines the memory up with semantic units. Outside the span, behavior is unchanged.

### 4.6 Treat the reasoning span as a private working buffer (F1, F11, F12; Turpin) — **use, as policy**

- Most human thought isn't verbal (F12), and a model's verbal trace can rationalize rather than
  report (Turpin). `reasoning_content` is therefore working state, not an explanation. User-facing
  surfaces (the CLI, GUI, `/chat`) should label it as "thinking" if they show it at all, never as
  "why the model answered".
- Keep `preserve_thinking=false` as the default (the reasoning plan already does).
- **Thinking never enters hippocampal memory or replay** (HM2 (F)) unless the owner decides
  otherwise. A rationalization stored as an episode would later be recalled as if it were
  evidence. This adds a third answer to HM2's open question 1 on consolidation: answer text yes or
  no, and thinking text separately, defaulting to no.
- Inner speech's documented function in working-memory rehearsal (F11) is the honest framing of
  what the span buys: more serial compute, and scratch space the model can read back. That is a
  capacity claim, which the pilot can test. It is not a transparency claim.

### 4.7 Condensed (latent) thought (F11, F12; Coconut) — **defer**

Human thinking is mostly condensed or unsymbolized, and Coconut shows that latent thoughts can
explore breadth-first with fewer steps. For adai, defer it for now:
- It needs a curriculum that *starts* from verbal traces, so it comes after RP-3 regardless.
- It gives up the inspectability that §4.1–4.3 depend on for calibration.
- At adai's model sizes, the verbal trace is also the training signal that SFT can supervise.

**Revisit after RP-6**, if the pilot shows reasoning helps but costs too many tokens. A middle step
is to *condense* verbal traces in the SFT data: drop the redundant rounds that §4.2 detects before
training on them.

### 4.8 Background thinking between turns (F7, F14) — **don't build on incubation**

"Let the model think in the background, like incubation" is the tempting application here. The
incubation evidence failed preregistered replication (F14), so this analysis doesn't recommend it.
The part of the research that does hold up, that unresolved concerns get priority (F7), is already
captured by HM2 (F) prioritized replay: replay favors what is weakly learned and unresolved. No
separate mechanism is needed.

## 5. Proposed amendments to the reasoning plan

These are proposals and need the owner's agreement before they are edited into
[reasoning_process_plan.md](reasoning_process_plan.md). Chunk ids `SC-*` are new and would join
the README's recommended order.

| # | Chunk | Depends on | What |
|---|---|---|---|
| SC-0 | Special-token id plan | — | **Resolved by TD-565 (November 9, 2026).** Turn tokens occupy 4–6 (TD-295), and new vocabularies reserve 7–19 for future special tokens with ordinary tokens from 20, so the think tokens take the next available reserved ids, **7 and 8** (owner decision, November 10, 2026), without moving ordinary tokens or changing the format again; later special tokens take the lowest id still unassigned. Build the retraining vocabulary (TD-563) with this layout |
| SC-1 | `ThoughtTrace`: lexical and representational boundary detection inside `generate_with_reasoning()` | RP-2a | Instrumentation only; no change to generated text |
| SC-2 | Thought-dynamics metrics: thoughts per span, movement, revisit rate, goal relevance, solution rounds; pushed with the reasoning plan's span-length metrics | SC-1 | Same pipeline as generation-quality metrics (TD-273) |
| SC-3 | Readiness readout (`P(</think>)` plus answer-start entropy) logged per boundary, with a calibration report against answer quality | SC-1 | Measure only |
| SC-4 | Opt-in stop conditions: stickiness close (§4.2) and readiness close (§4.3); `REASONING_EFFORT=auto` (§4.4) | SC-2, SC-3 and calibration | All off by default; the token budget stays the backstop |
| SC-5 | Hippocampal commits at thought boundaries inside the span (§4.5.3) | SC-1, HM2-5 | Replaces per-token commits within the span only |
| SC-6 | GRPO reward with a redundant-round cost (§4.4) | RP-4, SC-2 | Training-side only |
| — | Latent/condensed thinking (§4.7) | RP-6 | Deferred; revisit on pilot results |

Proposed config keys, all in `config.chatbot.conf` and all off by default:

```text
REASONING_THOUGHT_BOUNDARY=lexical          # lexical | representational
REASONING_THOUGHT_DRIFT_THRESHOLD=0.15      # representational boundary: cosine distance
REASONING_STICKINESS_CLOSE=false
REASONING_STICKINESS_WINDOW=6               # thoughts
REASONING_STICKINESS_REVISIT_THRESHOLD=0.5  # fraction of window that revisits an earlier thought
REASONING_READINESS_CLOSE=false
REASONING_READINESS_THRESHOLD=0.8
REASONING_READINESS_CONSECUTIVE=2           # boundaries
REASONING_EFFORT_DEFAULT=medium             # existing key; gains the value "auto"
```

The threshold values are placeholders to be set from SC-2/SC-3 data, not recommendations.

**Pilot (RP-6 / LJ-5 / HM-5):** add to the report (a) the hippocampal `gate_reasoning` vs
`gate_answer` comparison beside the world-model one (§4.5.1), and (b) thought-dynamics metrics
with reasoning on vs off, so a positive effect can be attributed: fewer solution rounds, less
stickiness, or the same dynamics with better answers.

## 6. Risks and limits

- **Analogy is not mechanism.** Every mapping in §3 is functional. The value of the research is
  that it names failure modes (rumination, decoupling, unfaithful report) and control structures
  (monitor/control, workspace) with decades of measurement behind them. It doesn't show that adai
  works the way a brain does, and the proposals stand or fall on adai's own metrics.
- **Self-report is unreliable in both directions.** Tseng & Poppenk built an implicit measure for
  this reason. A model's reasoning text is a self-report. SC-1's representational detector is the
  implicit counterpart, and the two should be compared, not merged.
- **Thresholds will be model- and data-specific.** Nothing in §5 should default on until SC-2/SC-3
  data exists for the retrained model (after TD-358 and the vocabulary rebuild).
- **Small-model capacity.** Published reasoning results come from models orders of magnitude
  larger than adai's configured architectures. The reasoning plan's own long pole, reasoning-trace
  data, still applies, and none of this analysis shortens it.
- **Readiness can be gamed in training.** If `P(</think>)` feeds a stop rule *and* GRPO rewards
  short spans, the model can learn to close early and guess. Keep the redundant-round cost (SC-6)
  tied to rounds *after* the final answer appears, never to closing early in general.

## 7. Open questions for the owner

1. ~~**SC-0:** which two reserved ids should the think tokens take?~~ **Decided November 10, 2026:**
   the next available in the reserved pool, 7 and 8.
2. Should thinking text ever be shown to users, and if so with what label (§4.6)?
3. Should thinking text be excluded from hippocampal memory and replay permanently, or only by
   default (§4.6)?
4. Is the `auto` effort level wanted at all, or should effort stay caller-chosen until the pilot?
5. Which boundary detector should SC-1 start with: lexical only, or both from the start so they
   can be compared?

## 8. Sources

Checked in this pass (†):

- Christoff, Irving, Fox, Spreng & Andrews-Hanna (2016), "Mind-wandering as spontaneous thought:
  a dynamic framework", *Nat. Rev. Neurosci.* 17:718–731 —
  [PubMed 27654862](https://pubmed.ncbi.nlm.nih.gov/27654862/),
  [Nature](https://www.nature.com/articles/nrn.2016.113)
- Mills, Raffaelli, Irving, Stan & Christoff (2018), "Is an off-task mind a freely-moving mind?",
  *Consciousness and Cognition* 58:20–33 —
  [record](https://experts.umn.edu/en/publications/is-an-off-task-mind-a-freely-moving-mind-examining-the-relationsh/)
- Tseng & Poppenk (2020), "Brain meta-state transitions demarcate thoughts across task contexts
  exposing the mental noise of trait neuroticism", *Nat. Commun.* 11 —
  [PMC7359033](https://pmc.ncbi.nlm.nih.gov/articles/PMC7359033)
- Smallwood & Schooler (2015), "The Science of Mind Wandering: Empirically Navigating the Stream
  of Consciousness", *Annu. Rev. Psychol.* 66:487–518 —
  [author copy](https://eprints.whiterose.ac.uk/103321/)
- Andrews-Hanna, Saxe & Yarkoni (2014), "Contributions of episodic retrieval and mentalizing to
  autobiographical thought", *NeuroImage* — [PMC4001766](https://pmc.ncbi.nlm.nih.gov/articles/PMC4001766)
- Dehaene & Naccache (2001), "Towards a cognitive neuroscience of consciousness", *Cognition*
  79:1–37 — [PDF](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/3702/readings/DehaeneNaccache_WorkspaceModel_Cognition2001.pdf);
  Dehaene et al. (2005), *PLoS Biol.* — [PMC1074751](https://pmc.ncbi.nlm.nih.gov/articles/PMC1074751)
- Goyal, Didolkar, Lamb, … Bengio (2021), "Coordination Among Neural Modules Through a Shared
  Global Workspace", ICLR 2022 — [arXiv:2103.01197](https://arxiv.org/abs/2103.01197)
- Alderson-Day & Fernyhough (2015), "Inner Speech: Development, Cognitive Functions,
  Phenomenology, and Neurobiology", *Psychol. Bull.* 141:931–965 —
  [PMC4538954](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4538954/)
- Heavey & Hurlburt (2008), "The phenomena of inner experience", *Consciousness and Cognition* —
  [PDF](https://www.gwern.net/doc/psychology/inner-voice/2008-heavey.pdf); Hurlburt, "Not everyone
  conducts inner speech" — [Psychology Today](https://www.psychologytoday.com/gb/blog/pristine-inner-experience/201110/not-everyone-conducts-inner-speech)
- Mangan (1993), "Taking Phenomenology Seriously", *Consciousness and Cognition* 2; Mangan (2000),
  "What Feeling Is the 'Feeling of Knowing'?" —
  [CogPrints 7586](https://web-archive.southampton.ac.uk/cogprints.org/7586/index.html);
  May (2004) commentary — [White Rose](https://eprints.whiterose.ac.uk/id/eprint/304/1/mayj1.pdf)
- Nelson & Narens (1990), "Metamemory: A theoretical framework and new findings", *Psychology of
  Learning and Motivation* 26 —
  [PDF](https://sites.socsci.uci.edu/~lnarens/1990/Nelson&Narens_Book_Chapter_1990.pdf)
- Klinger, current concerns theory, reviewed in
  [Frontiers 2013](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00570/pdf)
  and [PMC6156695](https://pmc.ncbi.nlm.nih.gov/articles/PMC6156695)
- Baird et al. (2012), "Inspired by Distraction", *Psychol. Sci.* —
  [PubMed 22941876](https://pubmed.ncbi.nlm.nih.gov/22941876/); failed replication Murray et al.
  (2024), listed in the [FORRT replication atlas](https://forrt.org/flora-replication-atlas/doi/10.1177/0956797612446024/)
- Zelikman et al. (2024), "Quiet-STaR" — [arXiv:2403.09629](https://arxiv.org/abs/2403.09629)
- Hao et al. (2024), "Training Large Language Models to Reason in a Continuous Latent Space"
  (Coconut) — [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- Chen et al. (2024), "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs" —
  [arXiv:2412.21187](https://arxiv.org/abs/2412.21187)
- Turpin, Michael, Perez & Bowman (2023), "Language Models Don't Always Say What They Think" —
  [arXiv:2305.04388](https://arxiv.org/abs/2305.04388)

Not re-checked in this pass (standard references): James (1890), *The Principles of Psychology*,
ch. IX (the text couldn't be fetched from this environment, so §2.1 F1 paraphrases it); Baars
(1988), *A Cognitive Theory of Consciousness*; Vygotsky (1934/1986), *Thought and Language*;
McVay & Kane (2010) executive-failure account.
