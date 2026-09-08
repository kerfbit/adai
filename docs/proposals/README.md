# Proposals

Design proposals and planning documents for ADAI features.

## Active Proposals

| Proposal | Status | Notes |
| --- | --- | --- |
| [Abnormal Samples GUI](abnormal_samples_gui_plan.md) | Partially Implemented | Detection and persistence implemented; GUI not yet built |
| [Transformer Introspection](transformer_introspection_plan.md) | Partially Implemented | Attention weight extraction exists; API endpoint and visualization tool not yet built |
| [Lessons Coverage Expansion](lessons-coverage-expansion.md) | Proposed | 12 new lessons covering architecture design, advanced features, and fine-tuning |
| [LeJEPA World Model with Gated Injection](lejepa_world_model_gated_injection_plan.md) | Proposed | Research/pilot stage — self-supervised world-model encoder injected into the decoder via zero-init gated cross-attention, alongside (not replacing) the existing encoder |
| [Reasoning Process (Thinking Phase)](reasoning_process_plan.md) | Proposed | `<think>`/`</think>` reasoning span with effort budgets, GRPO-based RL stage; postulates and proposes a testable phase-conditioned gate extension to the LeJEPA plan above |

## Recommended Order: LeJEPA World Model + Reasoning Process

These two proposals were designed together and share one integration point (a
phase-conditioned gate in `DecoderBlock`), so their chunks — `LJ-*` in
[lejepa_world_model_gated_injection_plan.md](lejepa_world_model_gated_injection_plan.md#implementation-phases),
`RP-*` in [reasoning_process_plan.md](reasoning_process_plan.md#implementation-phases) — are
sequenced together below rather than proposal-by-proposal. Rationale in full is in the reasoning
plan's ["Interaction with the LeJEPA World-Model Plan"](reasoning_process_plan.md#interaction-with-the-lejepa-world-model-plan)
section; short version: reasoning's mechanics have no dependency on LeJEPA, LeJEPA's most
valuable pilot result depends on reasoning existing first, and building the two proposals'
shared `DecoderBlock` gate in the wrong order means building it twice.

| # | Chunk | Proposal | Notes |
|---|---|---|---|
| 1 | `RP-1a` Extend `SpecialTokens.hpp` | Reasoning | No dependencies |
| 2 | `RP-1b` `vocab_builder` support | Reasoning | |
| 3 | `RP-2a` `generate_with_reasoning` core loop | Reasoning | **Do before `LJ-3a`** — gives the gate work a phase-boundary signal to key off of |
| 4 | `RP-2b` `ConversationContext` extension | Reasoning | Parallel with #3 |
| 5 | `RP-4` `GRPOOptimizer` | Reasoning | Fully standalone; also fixes the dormant `PPOOptimizer`/`ValueFunction` gap on its own merits — safe to slot in anytime, listed here as convenient idle work |
| — | *(start now, parallel to all of the above)* Reasoning-trace data sourcing/curation | Reasoning | Calendar-bound, not engineering-bound — the actual long pole for `RP-3` |
| 6 | `LJ-1a` `SIGReg` | LeJEPA | No dependencies |
| 7 | `LJ-1b` `Predictor` | LeJEPA | No dependencies |
| 8 | `LJ-2a` `LeJEPAEncoder` construction | LeJEPA | |
| 9 | `LJ-2b` `LeJEPAEncoder::train_step` | LeJEPA | |
| 10 | `LJ-3a` Gated `DecoderBlock` extension | LeJEPA | Build as the phase-conditioned `gate_reasoning`/`gate_answer` pair directly (chunk `RP-2a` is already done by this point) — **this absorbs `RP-5`**, which is not a separate step |
| 11 | `LJ-3b` Sparse injection knob | LeJEPA | |
| 12 | `LJ-4a` `EncoderDecoderModel::set_world_model()` | LeJEPA | |
| 13 | `LJ-4b` `incremental_trainer --objective=lejepa` + config keys | LeJEPA | |
| 14 | `LJ-4c` MNS registration + checkpointing | LeJEPA | |
| 15 | `RP-3` Stage 1 SFT | Reasoning | Gated on the data-sourcing track above landing |
| 16 | `LJ-5` + `RP-6` Combined pilot | Both | One run, not two — the go/no-go moment for both the world-model hypothesis and the reasoning-effect postulate |
| 17 | Reasoning Stage 2 RL (GRPO fine-tuning, using `RP-4`) | Reasoning | Furthest out, optional, contingent on a positive signal from #16 |

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
