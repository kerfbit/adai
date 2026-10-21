# `DecoderBlock` — Source File Reference

- **Files:** [`src/DecoderBlock.hpp`](../../../../src/DecoderBlock.hpp), [`src/DecoderBlock.cpp`](../../../../src/DecoderBlock.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_transformer` (used by `LLMDecoder` in `adai_models`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.4.0`, `@adai-reviewed: 2026-09-18`
- **Tests:** [`tests/decoderblock_test.cpp`](../../../../tests/decoderblock_test.cpp) → `decoderblockTests` (47 tests)
- **Origin:** original decoder layer; TD-050 (GPU incremental cache), TD-180 (gated world-model/hippocampal paths), TD-187 (gated-path persistence), TD-194 (hippocampal cross-reference/association)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class
> (linked against the debug libraries) in a scratch program.

---

## 1. What this file is

One **Pre-LN transformer decoder layer**. `LLMDecoder` stacks `num_decoder_layers` of them and
applies a final `LayerNorm` after the last one, because the residual stream is never normalized
inside a block.

```text
x ─┬─ norm1 → masked self-attention ──────────────────────(+)→ r1
   └───────────────────────────────────────────────────────┘
r1 ─┬─ norm2 → cross-attention(K/V = encoder output) ─────(+)→ r2
    └──────────────────────────────────────────────────────┘
r2 ─┬─ [TD-180] norm_world → cross-attn(world model) × tanh(gate) ──────(+)
r2 ─┬─ [TD-180] norm_hippocampal → cross-attn(memory keys, score bias) × tanh(gate_h) ─(+)
r2 ─┬─ norm3 → feed-forward ──────────────────────────────(+)→ output (unnormalized)
```

The two gated paths are **optional per instance** (`enable_world_model`, `enable_hippocampal`,
default false). `LLMDecoder` enables both only on every `world_model_inject_every_n_layers`-th
block (TD-181/TD-186). Each gate starts at 0, so `tanh(gate) = 0` and an allocated path is an exact
no-op until the gate moves.

### Why it matters

Every generated token passes through every block. The gated paths are the mechanism behind the
LeJEPA world-model injection and hippocampal episodic memory features (TD-180–TD-194), whose effect
on the model depends entirely on those gates opening during training (§4).

---

## 2. Construction

`DecoderBlock(d_model, num_heads, d_ff, dropout = 0.1, enable_world_model = false, enable_hippocampal = false)`:

- **Always allocated:** `self_attention` (`MultiHeadAttention`), `cross_attention`
  (`CrossAttention`), `feed_forward`, and `norm1`/`norm2`/`norm3`.
- **Gated paths, when enabled:** each gets its own `CrossAttention` and `LayerNorm`, plus a scalar
  gate.
- **Learning rate:** every sub-layer's `learning_rate` is set from the member (default 0.001).
- **`dropout` is stored, saved and loaded, but never applied.** Dropout isn't implemented anywhere
  in the forward pass, so the "regularization" the parameter promises doesn't happen.

Parameter count for `d_model = 512`, `num_heads = 8`, `d_ff = 2048` without gated paths:

| Component | Parameters |
|---|---|
| Self-attention, cross-attention | 4 · 512² each = 1,048,576 each |
| Feed-forward | 512·2048 + 2048·512 + 2048 + 512 = 2,099,712 |
| `LayerNorm` × 3 | 3,072 |
| **Total** | **4,199,936** (about 16.8 MB float32, plus the same again for gradients) |

Each enabled gated path adds about 1.05 M (cross-attention + norm) and one scalar.

---

## 3. Forward and backward

### `forward(input, encoder_output, self_attn_mask, cross_attn_mask, world_model_output, world_model_mask, memory, repetition_alpha, repetition_decay, cross_reference_alpha, association_decay)`

Pre-LN residual steps as drawn above. It caches every intermediate for `backward()`. The gated
paths run only when their input is supplied **and** the path was allocated:

- **World model:** `r2 += tanh(gate) · CrossAttention(norm_world(r2), world_model_output, world_model_mask)`.
- **Hippocampal memory** (runs only when `memory` is non-empty):
  - Keys from `memory->read_all()` serve as **both K and V**. The values are ignored, which is
    documented as correct when a slot's value equals its key.
  - Each slot gets a score bias of
    `−repetition_alpha · coverage[slot] + cross_reference_alpha · tanh(Σ_j association[slot][j] · coverage[j])`
    (TD-194), and attention runs through `CrossAttention::forward_with_scores()` (TD-174).
  - `r2 += tanh(gate_h) · result`.
  - **Then it mutates the memory:** coverage decays by `repetition_decay` and gains each slot's mean
    attention weight, and every slot pair's association decays by `association_decay` and gains
    their mean co-activation (Hebbian). Both are self-bounded by
    `1 / (1 − decay)`.

The memory is updated on **every** `forward()` call that has `memory`, not once per decode step.
That includes training and validation forwards, and one forward per hypothesis in beam search (the
only generation branch that passes memory, per TD-186), so coverage and associations accumulate
faster than the "once per decode step" design assumes (by inspection).

### `forward_with_cache(…)`

The same three core sub-layers with `MultiHeadAttention`/`CrossAttention::forward_with_cache()`.
It **never runs the gated paths**: it has no world-model or memory parameters (documented). With no
cache or `use_cache = false` it falls back to `forward()`, also without the gated paths. Every
KV-cached generation strategy (greedy, sampling, etc.) therefore ignores an attached world model
and hippocampal memory; only beam search applies them.

### `backward(grad_output, grad_encoder_output&)` / `backward(grad_output)`

This reverses the forward pass:
1. Through the feed-forward and `norm3`.
2. Through the hippocampal path, then the world-model path. That's the reverse of the order they
   were applied, and it's correct: each path's input gradient is taken after the later path's
   contribution has been added.
3. Through cross-attention and `norm2`.
4. Through self-attention and `norm1`.

Gate gradients use `d tanh = 1 − tanh²` and **accumulate** into `gate_grad`/`gate_h_grad`. The
gated paths' K/V gradients (toward the world model and the memory keys) are computed and
discarded: the world model is frozen and memory keys aren't parameters. `grad_encoder_output` is
**overwritten** with this block's cross-attention K/V gradient; `LLMDecoder` sums it across blocks.

---

## 4. Optimization (verified)

| Method | Behaviour |
|---|---|
| `register_parameters_with_optimizer(opt)` | Registers every sub-layer, including the gated paths' cross-attention and norms. **`gate`/`gate_h` are not registered**: they're plain floats and `Optimizer` only takes `Matrix` groups |
| `update_weights()` | Calls each sub-layer's `update_weights()` (each calls `optimizer->step()`, so this over-steps a shared optimizer, TD-297), then applies **plain SGD** to the gates: `gate -= learning_rate · gate_grad` |
| `zero_grad()` | Zeroes every sub-layer and both gate gradients |
| `get_gradient_norm()` | Combines self/cross/FF norms, plus the gated cross-attentions and gate gradients. Excludes `LayerNorm` gradients |
| `set_learning_rate(lr)` | Propagates to every sub-layer and the gate SGD rate |

> **The gates never train in production (verified).** `ChatbotTrainer` registers the model with
> one optimizer and then calls `optimizer->step()` directly (never `update_weights()`), which is
> also how it avoids TD-297. Since the gates aren't in the optimizer, **nothing ever updates them**.
> A block with the world-model path enabled, trained 50 steps the way `ChatbotTrainer` does, ended
> with `gate = 0.000000` (gradients of magnitude up to 34 were computed, then discarded by
> `zero_grad()`). Worse, with `tanh(gate) = 0` the gated cross-attention's own gradient is zero, so
> its weights didn't change either (`W_o` sum Δ = 0.000000). The same 50 steps through
> `DecoderBlock::update_weights()` moved the gate to −1.04 and the weights by −14.1.
>
> So in every `incremental_trainer` training pass, the world-model injection and hippocampal paths
> remain exact no-ops, whatever MNS links or how long training runs. Nothing reads `get_gate()`
> outside tests, so no metric exposes it either, despite the header mentioning a per-layer
> `mean(tanh(gate))` metric. The fix is to make each gate a 1×1 `Matrix` (value + gradient)
> registered with the optimizer.

---

## 5. Persistence

`save(path)` writes to `path`:
- `d_model`, `num_heads`, `d_ff`, `dropout_rate`, `learning_rate`;
- `norm1`/`norm2`/`norm3` gamma and beta as raw floats;
- a presence flag and gate value per gated path (TD-187).

The sub-layers go to sibling files:
- `.self_attn`, `.cross_attn`, `.ff`;
- `.world_model_cross_attn`/`.norm_world` and `.hippocampal_cross_attn`/`.norm_hippocampal` when
  present.

`load(path)` throws on a dimension mismatch or on a gated-path presence mismatch with this instance
(TD-187: fail clearly rather than drop or fabricate trained state). It sets `learning_rate` from the
file **before** those checks. Like `CrossAttention::load()` (TD-298), it never checks the stream
state, so a truncated file leaves garbage `LayerNorm` values or an unset gate rather than throwing
(by inspection). This legacy multi-file format is used by `LLMDecoder::save()`; the main checkpoint
path is SafeTensors (`ModelSerializer`), which has `get_norm*()` accessors but **no accessors for
the gates or gated sub-layers**.

---

## 6. GPU path (`ADAI_ENABLE_GPU`)

`gpu_forward()`, `gpu_forward_with_cache()` (TD-050) and `gpu_backward()` mirror the core Pre-LN
structure only. **No gated paths**, and `gpu_upload_weights()`/`gpu_download_grads()`/
`gpu_zero_grads()` cover only the six core sub-layers. A GPU training pass therefore never trains
the gated sub-layers either; documented as "CPU forward/backward only".

---

## 7. Tests

`decoderblock_test.cpp` has 47 tests:

| Group | Tests |
|---|---|
| Constructor | 3 |
| Forward | 4 |
| Masking | 3 |
| Cross-attention | 2 |
| Backward | 6 |
| Update | 2 |
| Edge cases | 3 |
| Integration | 3 |
| Save/load | 2 |
| Gated path | 8 (no-op guarantees, gate gradient checks) |
| Gated-path save/load | 7 |
| Cross-reference | 4 |

The gated-path tests call `update_weights()` or set gates directly, so none exercise the
optimizer-only update the trainer uses (§4).

---

## 8. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Gates aren't optimizer parameters; `ChatbotTrainer` only calls `optimizer->step()`, so gates stay at 0 and the gated cross-attentions get zero gradient (§4) | World-model injection and hippocampal memory never take effect in trained models | **Verified** | [TD-341](../../guides/TECHNICAL_DEBT.md#td-341-world-model-and-hippocampal-gates-never-train-in-production) |
| Gated paths only in non-cached CPU `forward()` (beam search); KV-cached generation and the GPU path skip them (§3, §6) | Attached world model/memory ignored by greedy and sampling generation and GPU training | By inspection (documented) | [TD-342](../../guides/TECHNICAL_DEBT.md#td-342-kv-cached-generation-and-the-gpu-path-skip-the-gated-paths) |
| Hippocampal memory is mutated on every `forward()` (training, validation, each beam hypothesis), not once per decode step (§3) | Coverage/associations inflate; repetition penalty stronger than designed | By inspection | [TD-343](../../guides/TECHNICAL_DEBT.md#td-343-hippocampal-memory-is-updated-on-every-forward-pass) |
| `dropout` stored and saved but never applied (§2) | No dropout regularization despite the parameter | By inspection | [TD-344](../../guides/TECHNICAL_DEBT.md#td-344-decoderblock-dropout-is-stored-but-never-applied) |
| `load()` unchecked reads; learning rate set before validation; SafeTensors doesn't carry gates or gated sub-layers (§5) | Truncated files load garbage; gated state lost in the main checkpoint format | By inspection | [TD-345](../../guides/TECHNICAL_DEBT.md#td-345-decoderblock-persistence-gaps-for-gated-state) |
| `update_weights()` over-steps a shared optimizer | Already tracked | — | TD-297 |

---

## 9. Merge notes

This file replaces `docs/development/api/transformer/decoder-block.md` (about 1,400 lines). Where
they disagree, this file wins:

- **The gated paths and the GPU path.** The old doc covered only the three core sub-layers. It had
  nothing on TD-180/187/194's gated paths, or on TD-050's GPU path.
- **Constructor signature.** It shows the old four-argument constructor.
- **Parameter count.** It omitted the feed-forward biases (4,197,376 vs 4,199,936).
- **Dropout.** It describes `dropout` as active regularization.

**Kept from the old doc:**
- the Pre-LN rationale;
- the parameter/memory table (corrected);
- the residual/backward structure (now with the gated steps);
- the KV-cache usage notes.

**Dropped:**
- use-case essays (translation, summarization, GPT-style);
- the comparison with `EncoderBlock`;
- generic performance estimates.

---

## Addendum (October 20, 2026): gates are optimizer parameters (TD-341)

`gate`/`gate_h` and their gradients are now 1×1 `Matrix` members (header and source 1.6.0). `register_parameters_with_optimizer()`
registers each as a parameter group when its path is allocated, so an optimizer step moves them. Before, only `update_weights()`'s
plain-SGD step did, and `ChatbotTrainer` never calls it. `update_weights()` keeps that SGD step only when no optimizer is registered.
`get_gate()`/`set_gate()`/`get_gate_grad()` (and the `_h` versions) and the checkpoint format are unchanged.

`ChatbotTrainer` logs per-layer `tanh(gate)`/`tanh(gate_h)` at the end of each epoch for gated models. The GPU path still doesn't run
the gated layers (TD-342).

Tests: `OptimizerOnlyTrainingOpensBothGates` and `UpdateWeightsStillStepsGatesWithoutAnOptimizer` here, and
`encoderdecoder_test.cpp`'s `TrainerStyleOptimizerStepsOpenTheGates`.

