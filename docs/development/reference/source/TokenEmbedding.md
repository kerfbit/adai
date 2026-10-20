# `TokenEmbedding` — Source File Reference

- **Files:** [`src/TokenEmbedding.hpp`](../../../../src/TokenEmbedding.hpp) (277 lines), [`src/TokenEmbedding.cpp`](../../../../src/TokenEmbedding.cpp) (267 lines)
- **Namespace:** global
- **Built into:** `adai_layers`; owned by `LLMEncoder`, `LLMDecoder` and `LeJEPAEncoder` (one each; the encoder's and decoder's are separate, untied matrices)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.1.0`, `@adai-reviewed: 2026-10-17` (both files)
- **Tests:** [`tests/tokenembedding_test.cpp`](../../../../tests/tokenembedding_test.cpp) (61 tests, `tokenembeddingTests`; all pass)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real class (and the real `PositionalEncoding`).

---

## 1. What this file is

The learnable `[vocab_size, d_model]` lookup table that turns token IDs into vectors; the first layer of every encoder and decoder.

| Member | Behaviour |
|---|---|
| `TokenEmbedding(vocab, d_model)` | `Matrix::randomize(kInitStd)`: a **normal** distribution with standard deviation `kInitStd = 1` (TD-517; it was 1/√d_model), from `Matrix`'s shared unseeded RNG (TD-368/TD-435), and a zeroed gradient matrix of the same size. Sizes aren't validated |
| `forward(ids)` | Row lookup into `[len, d_model]`, deliberately **unscaled** (no √d_model factor; the scale comes from the initialization); throws `out_of_range` for an ID outside `[0, vocab)`; caches the IDs (`cached_token_ids`, never read) |
| `backward(ids, grad)` | Checks shapes and IDs, then adds each gradient row into its token's row (repeated tokens accumulate) |
| `set_optimizer(opt)` / `register_parameters()` | Registers the matrix and its gradient with the optimizer (no duplicate check, TD-469) |
| `update_weights()` | With an optimizer, the **shared** `optimizer->step()` (TD-297); otherwise plain SGD over the whole matrix; then `zero_grad()` |
| `zero_grad()`, `get_gradient_norm()` (`float` sum over every entry), `clip_gradients(max)` | Full-matrix loops |
| `get_token_embedding(id)`, `get_embeddings()`, `get_vocab_size()`, `get_embedding_dim()`, `set_embeddings(m)` (SafeTensors path, unchecked shape) | Accessors |
| `save_embeddings()`/`save_weights()`, `load_pretrained()`/`load_weights()` | Binary `vocab_size`, `d_model`, then the floats; loading rejects a dimension mismatch; both print to `std::cout` |
| `initialize_constant(v)`, `reinitialize()` (N(0, 1), as the constructor), `print_config()` (stdout) | Utilities |

---

## 2. Problems

### Fixed: positional encoding swamped token identity at the model input (TD-517, October 17, 2026)

**Before:** each embedding row had norm ≈ 1, from N(0, 1/d_model) entries (measured 0.97 at `d_model = 768`). A sinusoidal positional-encoding
row has norm √(d_model/2), measured at 19.6. The encoders and decoder add the two unscaled. At initialization, two **different** tokens at the
same position had cosine similarity **0.997**, higher than the **same** token at adjacent positions (0.973). The model would first have to grow
every embedding about 20×, against the global weight decay that also applies to embeddings (TD-469). The owner decided on October 17, 2026
that this must be fixed before the TD-358 retraining.

**Now:** the table is initialized from N(0, 1) (`kInitStd`), and the lookup stays unscaled. This is the T5 convention, rather than the
original Transformer's small init with a √d_model multiply in `forward()`. Rows have norm ≈ √d_model (27.7 at 768), against 19.6 for
positional rows. At initialization, different tokens at one position measure 0.334 cosine, and the same token at adjacent positions 0.991.
`TokenIdentityNotSwampedByPositionAtInit` checks this and fails with the old init.

Why the initialization rather than a forward-pass multiply:
- **Unchanged format:** `forward()`/`backward()` and the saved format are unchanged, so an existing checkpoint loads and behaves exactly as
  before. Nothing is silently rescaled, and no format version is needed. Pre-fix checkpoints need retraining anyway (TD-358).
- **Exports:** SafeTensors/T5 exports keep meaning the raw table, which matches TD-466's decision to make the export T5-faithful.

**Side effects checked (October 19, 2026):**
- `LLMEncoder`, `LLMDecoder` and `LeJEPAEncoder` each own one table, and all three get the new init.
- The output layer has its own weights and isn't tied to the embedding, so initial logits are unaffected.
- Each block's first operation is a Pre-LN `LayerNorm`, so the larger input scale is normalized before attention and the feed-forward layers.

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

`TokenIdentityNotSwampedByPositionAtInit` (added with TD-517) checks the scale relationship with positional encoding at initialization.

**Not covered:** truncated files.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Embedding norm ≈ 1 vs positional norm ≈ 19.6, no √d_model scaling (§2) | Different tokens at one position were 0.997-similar at init | **Fixed October 17, 2026** (N(0, 1) init; owner decision) | [TD-517](../../guides/TECHNICAL_DEBT.md#td-517-token-embeddings-arent-scaled-by-sqrtd_model-so-position-swamps-content) |
| `load_pretrained()` doesn't check reads (§2) | Truncated checkpoints load with random rows | **Verified** | [TD-518](../../guides/TECHNICAL_DEBT.md#td-518-tokenembedding-unchecked-loading-full-matrix-updates-and-validation-gaps) |
| Full-matrix zeroing/updates; no size/shape validation; unused cache; console output (§2) | Cost and hygiene | By inspection | [TD-518](../../guides/TECHNICAL_DEBT.md#td-518-tokenembedding-unchecked-loading-full-matrix-updates-and-validation-gaps) |
| Shared-optimizer stepping; duplicate registration; unseeded init; float sums | Already tracked | — | TD-297, TD-469, TD-368, TD-435 |

---

## 5. Merge notes

This file replaces `docs/development/api/transformer/token-embedding.md` (1,110 lines). Where they disagree, this file wins: initialization is a normal
distribution (the old doc said "uniform or normal"), now with standard deviation 1 (TD-517); the old doc's "Xavier/Glorot" label was never
accurate in the usual fan-in + fan-out sense. Kept:
the API, the file format and the gradient-accumulation semantics. Dropped: the generic embedding tutorial and examples.

---

## Addendum (October 19, 2026): device table (TD-547)

In GPU builds the table also lives on the device:
- `gpu_upload_weights()`, `gpu_download_weights()`, `gpu_zero_grads()`, `gpu_download_grads()` (adds the device gradient into the host one)
  and `gpu_collect_parameters()` (one `GPUParamRef`, layer −1, so `GPUOptimizer` steps it on the device).
- `gpu_forward(ids, pe, pe_rows, pos_offset)` uploads only the token IDs and returns `table[ids[i]] + pe[pos_offset + i]` (no positional row
  past `pe_rows`, as `PositionalEncoding::forward()`). It uploads the table lazily and throws `std::out_of_range` for an ID outside the
  vocabulary.
- `gpu_backward(dout)` scatter-adds `dout` into the device gradient for the IDs of the last `gpu_forward()`, with atomic adds since tokens
  repeat.

GPU training therefore never zeroes, reads or steps the host table (TD-518's full-matrix zeroing remains on the CPU path). Tested by
`gpuOptimizerTests`' `DeviceEmbeddingMatchesTheHostLookupAndBackward` on the oneAPI CPU device.
