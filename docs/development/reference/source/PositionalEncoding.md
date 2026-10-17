# `PositionalEncoding` — Source File Reference

- **Files:** [`src/PositionalEncoding.hpp`](../../../../src/PositionalEncoding.hpp), [`src/PositionalEncoding.cpp`](../../../../src/PositionalEncoding.cpp) (236 lines together)
- **Namespace:** global
- **Built into:** `adai_layers`; owned by `LLMEncoder`, `LLMDecoder` and `LeJEPAEncoder` (one each, sized `max_seq_length × d_model`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10` (both files)
- **Tests:** [`tests/positionalencoding_test.cpp`](../../../../tests/positionalencoding_test.cpp) (29 tests)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real class.

---

## 1. What this file is

The fixed (not learned) sinusoidal positional encoding from "Attention Is All You Need", precomputed once:

```text
PE(pos, i) = sin(pos / 10000^(2·⌊i/2⌋ / d_model))   for even i
           = cos(pos / 10000^(2·⌊i/2⌋ / d_model))   for odd i
```

| Member | Behaviour |
|---|---|
| `PositionalEncoding(max_len, d_model)` | Fills `pos_encoding` `[max_len × d_model]` in `float`. **Verified**: within 3.1×10⁻⁵ of a double-precision reference at 512 × 768 |
| `forward(X)` | Returns `X + PE` for rows `< max_len`; rows past `max_len` get nothing added and a warning goes to `std::cerr` (on every call) |
| `get_encoding()`, `get_position_encoding(pos)` (throws out of range), `get_max_len()`, `get_d_model()` | Accessors |
| `print_config()`, `visualize()` | Console output; no production caller |

There are no parameters and no backward pass: gradients flow straight through to the token embedding.

### Who adds positions where

| Path | Positional encoding |
|---|---|
| Encoders (`LLMEncoder`, `LeJEPAEncoder`), the decoder's full forward and GPU forward | `forward()` (positions `0 … rows−1`) |
| Decoder KV-cached decoding (`forward_with_cache()`) and GPU incremental decoding | **Their own inline copy** of the formula, offset by `current_position` |

The inline copies (`Decoder.cpp`, two of them) were checked against this class: **identical to the bit** (max difference 0 over 512 × 768,
verified). They differ in one way: they keep encoding positions past `max_seq_length`, where `forward()` stops (TD-347, already tracked).

---

## 2. Gaps (by inspection, except where noted)

- **No width check (verified).** `forward()` doesn't compare `X.cols` with `d_model`: a 770-wide input threw `std::out_of_range("Matrix index out of
  bounds")` from the bounds-checked accessor (a confusing error), and a 100-wide input was accepted silently with only its first 100 dimensions encoded.
- **Warning spam.** The over-length warning is printed to `std::cerr` (not `Logger`) on every call, so an over-length input repeats it once per
  forward pass, once per decode step on the uncached paths.
- **Three copies of the formula.** The canonical one here and two inline in `Decoder.cpp`; identical today, but a change to one has to be made to all
  three (the same drift risk TD-216 describes for softmax/GELU). An `encoding_at(pos)` here would let the decoder reuse it.
- **Console output** in `print_config()`/`visualize()`.

Not a defect: positions beyond `max_len` receiving no encoding is tracked as TD-365 (chat input isn't truncated) and TD-347 (cached vs full divergence).

---

## 3. Tests

`positionalencoding_test.cpp` (29 tests): construction for several sizes, value range, position 0, sinusoidal pattern, unique positions, forward-pass
addition and preservation, relative-position and dot-product similarity, wavelength progression, boundaries (single position or dimension,
`max_len`, longer than `max_len`, empty), the Pythagorean identity, odd/even pattern, formula verification, transformer-style usage, accessors.

**Not covered:** input width mismatch.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| No `d_model` width check in `forward()` (§2) | Confusing `out_of_range`, or silently partial encoding | **Verified** | [TD-478](../../guides/TECHNICAL_DEBT.md#td-478-positionalencoding-width-check-warning-spam-and-duplicated-formula) |
| Over-length warning on `std::cerr` every call; formula duplicated in `Decoder.cpp`; console output (§2) | Log spam; drift risk | By inspection | [TD-478](../../guides/TECHNICAL_DEBT.md#td-478-positionalencoding-width-check-warning-spam-and-duplicated-formula) |
| Positions past `max_seq_length`: cached vs full divergence; untruncated chat input | Already tracked | — | TD-347, TD-365 |

---

## 5. Merge notes

This file replaces `docs/development/api/transformer/positional-encoding.md` (1,155 lines). Kept: the formula, the fixed (non-learned) design and the
over-length behaviour. Dropped: generic tutorial material (why sinusoids, wavelength discussion, comparison with learned embeddings, usage examples).
Added: where each path applies positions, the verified parity of the decoder's inline copies, and the width check.
