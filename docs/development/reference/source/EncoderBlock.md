# `EncoderBlock` — Source File Reference

- **Files:** [`src/EncoderBlock.hpp`](../../../../src/EncoderBlock.hpp), [`src/EncoderBlock.cpp`](../../../../src/EncoderBlock.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_transformer`; used by `LLMEncoder` (chatbot encoder, RAG encoder) and `LeJEPAEncoder` (world model)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/encoderblock_test.cpp`](../../../../tests/encoderblock_test.cpp) → `encoderblockTests` (34 tests)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class
> (linked against the debug libraries) in a scratch program.

---

## 1. What this file is

One **Pre-LN transformer encoder layer**: self-attention and feed-forward, each preceded by a
`LayerNorm` and wrapped in a residual connection. The residual stream is never normalized inside the
block; `LLMEncoder`/`LeJEPAEncoder` apply a final `LayerNorm` after the last block.

```text
x ─┬─ norm1 → MultiHeadAttention(mask) ─(+)→ r1
   └─────────────────────────────────────┘
r1 ─┬─ norm2 → FeedForward ─────────────(+)→ output (unnormalized)
    └───────────────────────────────────┘
```

It's the decoder block's simpler sibling: no cross-attention, no gated paths
([DecoderBlock.md](DecoderBlock.md)).

### Why it matters

Every encoder in the system is a stack of these:
- the chatbot's encoder, whose output feeds every decoder cross-attention;
- the RAG retrieval encoder ([DocumentStore.md](DocumentStore.md));
- the LeJEPA world model, whose output is injected into the decoder.

---

## 2. API

| Member | Behaviour |
|---|---|
| `EncoderBlock(d_model, num_heads, d_ff, dropout = 0.1)` | Builds `MultiHeadAttention`, `FeedForward` and two `LayerNorm`s and gives them the block's `learning_rate` (0.001). **`dropout` is stored, saved and printed but never applied**, the same as in `DecoderBlock` (TD-344) |
| `forward(input, mask = nullptr)` | `r1 = x + MHA(norm1(x), mask)`, `out = r1 + FF(norm2(r1))`. Caches intermediates |
| `backward(grad_output)` | Reverse of the above, splitting and accumulating through both residuals; returns ∂L/∂input. **Verified correct:** finite differences of the input gradient matched to ≤ 7×10⁻⁴ relative error with and without a padding mask |
| `update_weights()` | **Copies the block's `learning_rate` into all four sub-layers**, then calls each one's `update_weights()`. With a shared optimizer registered, each of those calls `optimizer->step()` (TD-297). There's no `set_learning_rate()`, so the public `learning_rate` member is the only knob, and it overrides anything set on the sub-layers directly |
| `zero_grad()` | All four sub-layers |
| `get_gradient_norm()` | Combines the attention and feed-forward norms in quadrature. **Excludes `LayerNorm` gradients** |
| `clip_gradients(max_norm)` | When the combined norm exceeds `max_norm`, scales the attention and feed-forward gradients by `max_norm / norm`. **`LayerNorm` gradients aren't scaled**, which changes the gradient direction. No caller anywhere; training clips through the optimizer |
| `register_parameters_with_optimizer(opt)` | All four sub-layers |
| `save_weights(file)` / `load_weights(file)` | The header (`d_model`, heads, `d_ff`, dropout) and both norms' gamma/beta go in `file`; attention and feed-forward go in `<file minus extension>_attention.bin`/`_feedforward.bin`. Load throws on a dimension mismatch, but **reads are never checked**, so a truncated file leaves garbage norm parameters (same class as TD-298/TD-345). The saved dropout value is read into a local and discarded. Both print to **`std::cout`** in library code |
| `print_config(name)` | Prints the dimensions and parameter counts (attention 4d², FF 2·d·d_ff + d_ff + d, norms 4d) to stdout. Its `int` arithmetic would overflow for very large dimensions |
| Accessors | `get_d_model()`, `get_num_heads()`, `get_d_ff()`, `get_dropout_rate()`, `get_feed_forward()`, `get_self_attention()` (activation and attention-entropy hooks), `get_norm1()`/`get_norm2()` (SafeTensors) |
| GPU (`ADAI_ENABLE_GPU`) | `gpu_upload_weights()`/`gpu_download_grads()`/`gpu_zero_grads()` over all four sub-layers; `gpu_forward()`/`gpu_backward()` mirror the Pre-LN structure |

**Stale header comments.** `forward()`'s doc lists "self-attention, residual connection and layer
norm, feed-forward, residual connection and layer norm", and `backward()`'s doc starts at "second
layer norm". That's Post-LN order. The class doc itself correctly says Pre-LN. The header also
advertises "optional dropout" and "gradient accumulation and clipping" as features; dropout isn't
applied, and clipping has no caller.

---

## 3. Who calls what

| Caller | Uses |
|---|---|
| `LLMEncoder` | `forward()`/`backward()` per layer; `register_parameters_with_optimizer()`; `save_weights()`/`load_weights()` as `<base>_encoder_block_<i>.bin`; the GPU path |
| `LeJEPAEncoder` | The same, plus `update_weights()` only in its no-optimizer fallback (it calls one top-level `optimizer->step()` otherwise, which avoids TD-297); saves under its world-model directory |
| Hooks (trainer diagnostics) | `get_feed_forward()`/`get_self_attention()` for activation-saturation and attention-entropy hooks |

The chatbot trainer (`ChatbotTrainer`) never calls `update_weights()`; it steps the shared optimizer
once. Note that `EncoderDecoderModel::update_weights()` (the no-optimizer path) doesn't update the
encoder at all ("LLMEncoder doesn't have this method"). That's for `EncoderDecoderModel`'s own
reference.

---

## 4. Tests

`encoderblock_test.cpp` has 34 tests:
- construction;
- forward shapes and masking;
- backward (shapes and gradient flow);
- update/zero;
- gradient norm and clipping;
- save/load round trips;
- the config printout.

**Not covered:**
- truncated-file loading;
- `LayerNorm` gradients in the norm and clipping.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `dropout` stored, saved and printed, never applied (§2) | No dropout regularization in any encoder | By inspection (same as TD-344 for the decoder) | [TD-344](../../guides/TECHNICAL_DEBT.md#td-344-decoderblock-dropout-is-stored-but-never-applied) (amended) |
| `get_gradient_norm()`/`clip_gradients()` ignore `LayerNorm` gradients; `clip_gradients()` unused; header's forward/backward step lists describe Post-LN; "dropout"/"clipping" advertised (§2) | Misreported norms; direction-distorting clip if ever used; misleading docs | By inspection | [TD-356](../../guides/TECHNICAL_DEBT.md#td-356-encoderblock-gradient-norm-and-clipping-ignore-layernorm-header-describes-post-ln) |
| `update_weights()` overwrites sub-layer LRs with the block's; no `set_learning_rate()` (§2) | LR changes must go through the public member; per-sub-layer LRs are silently reset | By inspection | [TD-357](../../guides/TECHNICAL_DEBT.md#td-357-encoderblock-learning-rate-handling-unchecked-loads-and-console-output) |
| `load_weights()` unchecked reads; `std::cout` in `save_weights()`/`load_weights()`/`print_config()` (§2) | Truncated files load garbage; console noise from library code | By inspection | [TD-357](../../guides/TECHNICAL_DEBT.md#td-357-encoderblock-learning-rate-handling-unchecked-loads-and-console-output) |
| `update_weights()` over-steps a shared optimizer | Already tracked | — | TD-297 |

---

## 6. Merge notes

This file replaces `docs/development/api/transformer/encoder-block.md` (about 1,860 lines). Where
they disagree, this file wins:

- **Architecture.** The old doc described a **Post-LN** block,
  `EncoderBlock(x) = LayerNorm(x + FFN(LayerNorm(x + Attention(x))))`, with step-by-step code
  normalizing after each residual. The code is **Pre-LN** (§1), and has been since the Pre-LN
  conversion that added `LLMEncoder`'s final norm.
- **Dropout.** It presented `dropout` as active regularization.

**Kept from the old doc:**
- the sub-component roles;
- the residual-gradient explanation (now in Pre-LN form);
- the save/load file layout;
- the parameter-count formulas.

**Dropped:** complexity essays and usage walkthroughs for a Post-LN design that no longer exists.
The separate [ENCODERBLOCK_TEST_COVERAGE.md](../../testing/coverage/ENCODERBLOCK_TEST_COVERAGE.md)
is kept as test-coverage history.
