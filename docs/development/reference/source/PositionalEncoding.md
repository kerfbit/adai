# `PositionalEncoding` — Source File Reference

- **Files:** [`src/PositionalEncoding.hpp`](../../../../src/PositionalEncoding.hpp), [`src/PositionalEncoding.cpp`](../../../../src/PositionalEncoding.cpp) (237 lines together)
- **Namespace:** global
- **Built into:** `adai_layers`; owned by `LLMEncoder`, `LLMDecoder` and `LeJEPAEncoder` (one each, sized `max_seq_length × d_model`)
- **Status tag:** `@adai-status: stable`; `PositionalEncoding.cpp` `@adai-version: 1.1.0`, `.hpp` `1.0.1`; `@adai-reviewed: 2026-11-18` (both files)
- **Tests:** [`tests/positionalencoding_test.cpp`](../../../../tests/positionalencoding_test.cpp) (32 tests)
- **Last traced against the code:** 2026-11-18 (TD-478's fix; first traced 2026-10-17)

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
| `forward(X)` | Returns `X + PE` for rows `< max_len`; rows past `max_len` get nothing added, and a `Logger` warning says so once per process. Throws `std::invalid_argument` when `X.cols != d_model` (TD-478) |
| `get_encoding()`, `get_position_encoding(pos)` (throws out of range), `get_max_len()`, `get_d_model()` | Accessors |
| `print_config()`, `visualize()` | Log through `adai::Logger` at info (`visualize()` one line per position); no production caller (TD-478; they wrote to `std::cout`) |

There are no parameters and no backward pass: gradients flow straight through to the token embedding.

### Who adds positions where

| Path | Positional encoding |
|---|---|
| Encoders (`LLMEncoder`, `LeJEPAEncoder`), the decoder's full forward and GPU forward | `forward()` (positions `0 … rows−1`) |
| Decoder KV-cached decoding (`forward_with_cache()`) and GPU incremental decoding | This class's table (`get_encoding()`), rows `current_position …`; on the GPU, uploaded once (`LLMDecoder::gpu_positional_table()`) |

Until October 2026 both cached paths re-typed the formula inline (checked bit-identical to this class, October 17). The GPU path switched to the
table with TD-547 (October 19) and the CPU path with TD-347 (October 26), which also made them stop at `max_seq_length` as `forward()` does. The
formula now exists only here.

---

## 2. Fixed gaps (TD-478, November 18, 2026)

- **Width check (was verified).** `forward()` didn't compare `X.cols` with `d_model`: a 770-wide input threw `std::out_of_range("Matrix index out of
  bounds")` from the bounds-checked accessor (a confusing error), and a 100-wide input was accepted silently with only its first 100 dimensions
  encoded. It now throws `std::invalid_argument` naming both widths.
- **Warning spam.** The over-length warning went to `std::cerr` on every call, so once per decode step on the uncached paths. It is now a
  `Logger` warning, once per process (a static `std::atomic<bool>`); a full run of `positionalencodingTests` prints it once.
- **Three copies of the formula.** Gone with TD-547 and TD-347 (§1).
- **Console output** in `print_config()`/`visualize()`: now `Logger` at info, as TD-467 did for `MultiHeadAttention`.

Not a defect: positions beyond `max_len` receiving no encoding is tracked as TD-365 (chat input isn't truncated) and TD-347 (cached vs full divergence).

---

## 3. Tests

`positionalencoding_test.cpp` (32 tests): construction for several sizes, value range, position 0, sinusoidal pattern, unique positions, forward-pass
addition and preservation, relative-position and dot-product similarity, wavelength progression, boundaries (single position or dimension,
`max_len`, longer than `max_len`, empty), the Pythagorean identity, odd/even pattern, formula verification, transformer-style usage, accessors.

TD-478 added `InputWidthMustMatchDModel`, `OverLengthWarningIsLoggedAtMostOncePerProcess` (at most once, since an earlier test may have logged it)
and `PrintersLogInsteadOfWritingToStdout`.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| No `d_model` width check; over-length warning on `std::cerr` every call; formula duplicated in `Decoder.cpp`; console output (§2) | — | **Fixed November 18, 2026** | [TD-478](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-478-positionalencoding-width-check-warning-spam-and-duplicated-formula) |
| Positions past `max_seq_length`: cached vs full divergence; untruncated chat input | Already tracked | — | TD-347, TD-365 |

---

## 5. Merge notes

This file replaces `docs/development/api/transformer/positional-encoding.md` (1,155 lines). Kept: the formula, the fixed (non-learned) design and the
over-length behaviour. Dropped: generic tutorial material (why sinusoids, wavelength discussion, comparison with learned embeddings, usage examples).
Added: where each path applies positions, the verified parity of the decoder's inline copies, and the width check.
