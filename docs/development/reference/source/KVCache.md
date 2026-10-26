# `KVCache` — Source File Reference

- **File:** [`src/KVCache.hpp`](../../../../src/KVCache.hpp) (header-only, 226 lines)
- **Namespace:** global (`KVCache`, `DecoderKVCache`)
- **Built into:** header-only; included by `MultiHeadAttention.hpp`, `DecoderBlock.hpp`, `Decoder.hpp`, `EncoderDecoderModel.hpp`; used by `CrossAttention.cpp` and `Decoder.cpp`
- **GPU counterpart:** `GPUKVCache`/`GPUDecoderKVCache` in [MatrixGPU.md](MatrixGPU.md) (pre-allocated)
- **Tests:** no dedicated file; exercised by `inference_optimization_test.cpp` (≈20 uses) and `encoderdecoder_test.cpp`
- **Last traced against the code:** 2026-10-13

> Code-traced reference. Behaviour marked **verified** was confirmed with scratch programs, including
> AddressSanitizer builds.

---

## 1. What this file is

The **CPU key/value cache** for incremental (one-token-at-a-time) decoding. Once a token's
self-attention keys and values have been computed, later decode steps reuse them instead of
reprocessing the whole prefix.

| Type | Contents |
|---|---|
| `KVCache` | `keys`, `values` (`Matrix`, `[length × d_model]`), `current_length` |
| `DecoderKVCache(num_layers)` | One self-attention `KVCache` and one cross-attention `KVCache` per decoder layer |

### Who uses it

`EncoderDecoderModel`'s cached generation (`generate_response()` and the non-beam strategies; beam
search uses no cache) builds a `DecoderKVCache` per request and calls
`LLMDecoder::forward_with_cache()`. For each layer, that passes `get_self_attention_cache(i)` and
`get_cross_attention_cache(i)` down to `DecoderBlock`, which hands them to:
- `MultiHeadAttention::forward_with_cache()`, which appends the new tokens' K/V and attends over the
  whole cache;
- `CrossAttention`'s cached path, which appends the encoder's K/V **once** (when the cache is empty)
  and reuses them.

One `DecoderKVCache` serves exactly **one** generation of one request. The cached keys and values
depend on that request's encoder output and prefix, so a cache must not be shared across hypotheses
(the beam reason above), messages or sessions.

---

## 2. API

### `KVCache`

| Member | Behaviour |
|---|---|
| `is_empty()`, `size()` | `current_length == 0`; `current_length` |
| `clear()` | Resets both matrices to `0 × 0` and the length to 0 (frees the memory) |
| `append(new_keys, new_values)` | Empty cache: **copies** the inputs in, and `current_length = new_keys.rows`. Otherwise: allocates new `[current_length + new_rows × keys.cols]` matrices, copies the old rows and the new rows, and replaces the members |
| `get_keys()`, `get_values()` | Const references to the full cached matrices |

### `DecoderKVCache`

| Member | Behaviour |
|---|---|
| `DecoderKVCache(num_layers)` | `num_layers` empty self- and cross-attention caches. **There's no default constructor** |
| `clear()` / `clear_self_attention()` | Clear everything / only the self-attention caches (keeping the encoder K/V) |
| `get_self_attention_cache(i)`, `get_cross_attention_cache(i)` | `vector::operator[]`, **no bounds check** |
| `is_empty()` | All self-attention caches empty |
| `current_length()` | **Layer 0's** self-attention length only (assumes every layer is in step) |

---

## 3. Problems (verified)

### `append()` has no shape checks

Nothing compares `new_keys` with `new_values`, or the new columns with the cached ones:
- **Wider new rows** (`append(Matrix(2,4), …)` then `append(Matrix(1,8), …)`): the copy writes 8
  columns into rows allocated with 4. ASan: **heap-buffer-overflow WRITE**.
- **Fewer value rows than key rows** (`append(Matrix(3,4), Matrix(1,4))` onto a non-empty cache): the
  loop indexes `new_values` by `new_keys.rows`. ASan: **heap-buffer-overflow READ**.
- **Inconsistent first append** (`keys 3×4`, `values 1×8`): accepted silently (`size() = 3`, values
  `1×8`), corrupting every later step.

Today's callers always pass matching shapes, so these are latent, but the API is public and the
failure is memory corruption rather than an error.

### Every append reallocates and copies the whole cache

There's no pre-allocation, so step *n* copies *n × d_model* floats per matrix, per layer, and a
generation of *n* tokens is O(n² · d_model) in copying alone. Measured, one row per step at d_model
512, for a single cache: 128 steps 15.6 ms, 256 steps 53.8 ms, 512 steps **255 ms** (quadratic: ×4.7
for ×2 steps). With a 24-layer decoder that's about 6 s of pure copying for a 512-token generation,
on top of the attention itself. The GPU cache (`GPUKVCache`) pre-allocates to `max_seq_length`; this
one doesn't, though the same design would work.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `append()` doesn't check key/value or column shapes (§3) | Heap overflow (write or read) or silent corruption on mismatched input | **Verified** (ASan) | [TD-421](../../guides/TECHNICAL_DEBT.md#td-421-kvcacheappend-has-no-shape-checks) |
| Full reallocate-and-copy on every append (§3) | O(n²·d) copying per generation; about 6 s at 24 layers × 512 tokens | **Verified** (timing) | [TD-422](../../guides/TECHNICAL_DEBT.md#td-422-kvcache-copies-the-whole-cache-on-every-append) |
| `get_*_cache()` not bounds-checked; `current_length()` reads only layer 0; no default constructor (§2) | Out-of-range layer index is undefined behaviour; inconsistent layers go unnoticed | By inspection | [TD-421](../../guides/TECHNICAL_DEBT.md#td-421-kvcacheappend-has-no-shape-checks) |
| Generation past `max_seq_length` diverges between cached and full forward | Already tracked | — | TD-347 |

---

## 5. Merge notes

This file replaces `docs/development/reference/kvcache.md` (680 lines). Where they disagree, this file
wins:

- **Cost.** The old doc's table showed a cached step as O(d²) per token, a "50× speedup at token 50".
  A cached step is O(n·d) for the append copy (plus O(n·d) attention over the cache), so the whole
  generation is quadratic (§3). The old doc's "2–3× speedup" figure had no source.
- **"Pattern 2: Chatbot with Conversation History"** kept one `DecoderKVCache` per session in a
  `std::map` and reused it across messages. That's wrong: the cached self-attention K/V depend on the
  encoder output of the message being answered, and the cross-attention cache holds that encoder's K/V,
  so reusing a cache for the next message attends to the previous message's state. It also doesn't
  compile (`DecoderKVCache` has no default constructor for `map::operator[]`).
- **"Pattern 4: Batch Processing"** implied caches enable batching; nothing in the model is batched
  (TD-171).

**Kept:** the API description (§2), the per-layer structure, and the "clear self-attention only, keep
the encoder K/V" pattern.

## Addendum (October 26, 2026): in-place growth and checks (TD-421, TD-422)

`append()` pushes the new rows onto the existing `Matrix` (one vector per row), so no cached row is copied. One cache at d_model 512:
512 steps took 219 ms before and 1.03 ms after. It throws `invalid_argument` on mismatched shapes, and `DecoderKVCache`'s layer
accessors throw `out_of_range`. Tests: `tests/kvcache_test.cpp`.
