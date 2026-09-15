# Proposals

Design proposals and planning documents for ADAI features.

## Active Proposals

| Proposal | Status | Notes |
| --- | --- | --- |
| [Abnormal Samples GUI](abnormal_samples_gui_plan.md) | Partially Implemented | Detection and persistence implemented; GUI not yet built |
| [Transformer Introspection](transformer_introspection_plan.md) | Partially Implemented | Attention weight extraction exists; API endpoint and visualization tool not yet built |
| [Lessons Coverage Expansion](lessons-coverage-expansion.md) | Proposed | 12 new lessons covering architecture design, advanced features, and fine-tuning |
| [LeJEPA World Model + Hippocampal Memory with Gated Injection](lejepa_world_model_gated_injection_plan.md) | Proposed | Research/pilot stage — two side signals injected into the decoder via independent zero-init gated cross-attention paths, alongside (not replacing) the existing encoder: a frozen self-supervised world-model encoder, and a fast episodic memory buffer whose gate carries a bounded, gradually increasing penalty for repeatedly attending to the same stored episode |
| [Reasoning Process (Thinking Phase)](reasoning_process_plan.md) | Proposed | `<think>`/`</think>` reasoning span with effort budgets, GRPO-based RL stage; postulates and proposes a testable phase-conditioned gate extension to the LeJEPA plan above |

## Recommended Order: LeJEPA World Model + Hippocampal Memory + Reasoning Process

These proposals were designed together and share one integration point (gated paths in
`DecoderBlock` — world model, hippocampal memory, and reasoning's phase-conditioning), so their
chunks — `LJ-*`/`HM-*` in
[lejepa_world_model_gated_injection_plan.md](lejepa_world_model_gated_injection_plan.md#implementation-phases),
`RP-*` in [reasoning_process_plan.md](reasoning_process_plan.md#implementation-phases) — are
sequenced together below rather than proposal-by-proposal. Rationale in full is in the reasoning
plan's ["Interaction with the LeJEPA World-Model Plan"](reasoning_process_plan.md#interaction-with-the-lejepa-world-model-plan)
section and the LeJEPA plan's own risk note on scope relative to reasoning; short version:
reasoning's mechanics have no dependency on LeJEPA, LeJEPA's most valuable pilot result depends
on reasoning existing first, and building the `DecoderBlock` gates in the wrong order means
touching that class three separate times instead of once.

| # | Chunk | Proposal | Notes |
|---|---|---|---|
| 1 | `RP-1a` Extend `SpecialTokens.hpp` | Reasoning | No dependencies |
| 2 | `RP-1b` `vocab_builder` support | Reasoning | |
| 3 | `RP-2a` `generate_with_reasoning` core loop | Reasoning | **Do before `LJ-3a`/`HM-3`** — gives the gate work a phase-boundary signal to key off of |
| 4 | `RP-2b` `ConversationContext` extension | Reasoning | Parallel with #3 |
| 5 | `RP-4` `GRPOOptimizer` | Reasoning | Fully standalone; also fixes the dormant `PPOOptimizer`/`ValueFunction` gap on its own merits — safe to slot in anytime, listed here as convenient idle work |
| 6 | `HM-2` `CrossAttention::forward_with_scores` | LeJEPA (hippocampal) | Fully standalone, no dependency on anything else — safe to slot in anytime, similar to `RP-4` |
| — | *(start now, parallel to all of the above)* Reasoning-trace data sourcing/curation | Reasoning | Calendar-bound, not engineering-bound — the actual long pole for `RP-3` |
| 7 | `LJ-1a` `SIGReg` | LeJEPA | No dependencies |
| 8 | `LJ-1b` `Predictor` | LeJEPA | No dependencies |
| 9 | `LJ-2a` `LeJEPAEncoder` construction | LeJEPA | |
| 10 | `LJ-2b` `LeJEPAEncoder::train_step` | LeJEPA | |
| 11 | `HM-1` `HippocampalMemory` buffer | LeJEPA (hippocampal) | Depends on `LJ-2a` (reuses `LeJEPAEncoder::encode()` as the key source), not on `LJ-2b` |
| 12 | `LJ-3a` + `HM-3` Gated `DecoderBlock` extension (world model + hippocampal, one change) | LeJEPA | Build `gate_reasoning`/`gate_answer` phase-conditioning (chunk `RP-2a` is already done) into **both** gated paths while they're being added — **this absorbs `RP-5`**, which is not a separate step. Doing the world-model gate and the hippocampal gate as two later, separate edits to `DecoderBlock` is exactly the rework this ordering exists to avoid. |
| 13 | `LJ-3b` Sparse injection knob | LeJEPA | |
| 14 | `LJ-4a` `EncoderDecoderModel::set_world_model()` | LeJEPA | |
| 15 | `LJ-4b` `incremental_trainer --objective=lejepa` + config keys | LeJEPA | |
| 16 | `LJ-4c` MNS registration + checkpointing | LeJEPA | |
| 17 | `HM-4` `EncoderDecoderModel::set_hippocampal_memory()` + `HIPPOCAMPAL_*` config + write-policy call site | LeJEPA (hippocampal) | Parallel with 14–16; same integration phase |
| 18 | `RP-3` Stage 1 SFT | Reasoning | Gated on the data-sourcing track above landing |
| 19 | `LJ-5`/`HM-5` + `RP-6` Combined pilot | All | One run, not three — the go/no-go moment for the world-model hypothesis, the hippocampal repetition-penalty calibration (alpha sweep), and the reasoning-effect postulate together |
| 20 | Reasoning Stage 2 RL (GRPO fine-tuning, using `RP-4`) | Reasoning | Furthest out, optional, contingent on a positive signal from #19 |

## Archived (Implemented)

The following proposals have been fully implemented and moved to [development/archive/](../development/archive/):

- Adaptive Gradient Clipping
- Advanced Training Metrics
- Dataset Manager Separation
- Dataset Transport (FTP/FTPS)
- HuggingFace SafeTensors Compatibility
- Incremental Trainer Dashboard
- Incremental Trainer Registry Integration
- Length Bucket Sorting
- LLM Operations Tooling
- Model Name Service
- Multi-Instance Metrics Service
- Performance Profiler Upgrades
- Persistent Metrics SQL Storage
- Unicode Tokenizer Upgrade
- Validation Metrics
