# `TokenEmbedding` — Source File Reference

- **Files:** [`src/TokenEmbedding.hpp`](../../../../src/TokenEmbedding.hpp) (271 lines), [`src/TokenEmbedding.cpp`](../../../../src/TokenEmbedding.cpp) (265 lines)
- **Namespace:** global
- **Built into:** `adai_layers`; owned by `LLMEncoder`, `LLMDecoder` and `LeJEPAEncoder` (one each; the encoder's and decoder's are separate, untied matrices)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10` (both files)
- **Tests:** [`tests/tokenembedding_test.cpp`](../../../../tests/tokenembedding_test.cpp) (60 tests, `tokenembeddingTests`; all pass)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real class (and the real `PositionalEncoding`).

---

## 1. What this file is

The learnable `[vocab_size, d_model]` lookup table that turns token IDs into vectors; the first layer of every encoder and decoder.

| Member | Behaviour |
|---|---|
| `TokenEmbedding(vocab, d_model)` | `Matrix::randomize(1/√d_model)` (a **normal** distribution with that standard deviation, from `Matrix`'s shared unseeded RNG, TD-368/TD-435) and a zeroed gradient matrix of the same size. Sizes aren't validated |
| `forward(ids)` | Row lookup into `[len, d_model]`; throws `out_of_range` for an ID outside `[0, vocab)`; caches the IDs (`cached_token_ids`, never read) |
| `backward(ids, grad)` | Checks shapes and IDs, then adds each gradient row into its token's row (repeated tokens accumulate) |
| `set_optimizer(opt)` / `register_parameters()` | Registers the matrix and its gradient with the optimizer (no duplicate check, TD-469) |
| `update_weights()` | With an optimizer, the **shared** `optimizer->step()` (TD-297); otherwise plain SGD over the whole matrix; then `zero_grad()` |
| `zero_grad()`, `get_gradient_norm()` (`float` sum over every entry), `clip_gradients(max)` | Full-matrix loops |
| `get_token_embedding(id)`, `get_embeddings()`, `get_vocab_size()`, `get_embedding_dim()`, `set_embeddings(m)` (SafeTensors path, unchecked shape) | Accessors |
| `save_embeddings()`/`save_weights()`, `load_pretrained()`/`load_weights()` | Binary `vocab_size`, `d_model`, then the floats; loading rejects a dimension mismatch; both print to `std::cout` |
| `initialize_constant(v)`, `reinitialize()`, `print_config()` (stdout) | Utilities |

---

## 2. Problems

### Positional encoding swamps token identity at the model input (verified; owner decision October 17, 2026: must fix)

Each embedding row has norm ≈ 1 (`d_model` entries of standard deviation 1/√d_model; measured 0.97 at `d_model = 768`), while a sinusoidal
positional-encoding row has norm √(d_model/2) (measured 19.6). The encoders and decoder add the two directly, with no √d_model scaling of the embedding
(the original Transformer multiplies embeddings by √d_model for exactly this reason). At initialization, two **different** tokens at the same position
had cosine similarity **0.997**, higher than the **same** token at adjacent positions (0.973): the input carries position far more strongly than
content, and the model must first learn to grow every embedding about 20×, against the global weight decay that also applies to embeddings (TD-469).

This is a training-dynamics and architecture choice rather than a crash; the owner decided on October 17, 2026 that it must be fixed. Changing it
alters every checkpoint's input scale, so it lands with the TD-358 retraining. It may also be relevant to the open Pre-LN residual-growth question (TD-371).

### `load_pretrained()` accepts truncated files (verified)

Reads aren't checked. A file cut to half its size loaded without error, and the rows past the cut kept their random initialization (row 999 unchanged).
The same class as TD-467/TD-369 for the other layers' loaders.

### Smaller items (by inspection)

- **Full-matrix work every step:** `zero_grad()` and the no-optimizer update touch all `vocab × d_model` entries (24.6 M floats at 32k × 768) though only
  a few rows have gradients; `get_gradient_norm()` sums them in `float` (TD-435's drift).
- **No validation** of `vocab_size`/`d_model` (0 gives an infinite scale), or of `set_embeddings()`'s shape.
- **`cached_token_ids` is written and never read**; console output in load/save/print.

---

## 3. Tests

`tokenembedding_test.cpp` (60): construction, forward lookups and bounds, backward accumulation and shape checks, SGD and optimizer updates, gradient norm
and clipping, save/load round trips and mismatch rejection, constant and re-initialization, and accessors.

**Not covered:** truncated files, the scale relationship with positional encoding.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Embedding norm ≈ 1 vs positional norm ≈ 19.6, no √d_model scaling (§2) | Different tokens at one position are 0.997-similar at init; content must grow 20× under weight decay | **Verified** (owner: must fix) | [TD-517](../../guides/TECHNICAL_DEBT.md#td-517-token-embeddings-arent-scaled-by-sqrtd_model-so-position-swamps-content) |
| `load_pretrained()` doesn't check reads (§2) | Truncated checkpoints load with random rows | **Verified** | [TD-518](../../guides/TECHNICAL_DEBT.md#td-518-tokenembedding-unchecked-loading-full-matrix-updates-and-validation-gaps) |
| Full-matrix zeroing/updates; no size/shape validation; unused cache; console output (§2) | Cost and hygiene | By inspection | [TD-518](../../guides/TECHNICAL_DEBT.md#td-518-tokenembedding-unchecked-loading-full-matrix-updates-and-validation-gaps) |
| Shared-optimizer stepping; duplicate registration; unseeded init; float sums | Already tracked | — | TD-297, TD-469, TD-368, TD-435 |

---

## 5. Merge notes

This file replaces `docs/development/api/transformer/token-embedding.md` (1,110 lines). Where they disagree, this file wins: initialization is a normal
distribution with standard deviation 1/√d_model (the old doc said "uniform or normal"), and it isn't Xavier/Glorot in the usual fan-in + fan-out sense. Kept:
the API, the file format and the gradient-accumulation semantics. Dropped: the generic embedding tutorial and examples.
