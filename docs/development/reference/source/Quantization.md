# `Quantization` — Source File Reference

- **File:** [`src/Quantization.hpp`](../../../../src/Quantization.hpp) (header-only, 496 lines)
- **Namespace:** global (`QuantizationMode`, `CalibrationMethod`, `QuantizationParams`, `Quantizer`, `QuantizedMatrix`, `print_quantization_stats()`)
- **Used by:** **nothing in production**: tests and `examples/Phase5Examples.cpp` only. A quantization tool is listed under TD-014 (deliberately deferred)
- **Status tag:** `@adai-status: beta` ("capped by TD-038 — tested but not wired into any shipped binary"), `@adai-version: 0.7.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/quantization_test.cpp`](../../../../tests/quantization_test.cpp) (6 tests) and 10 in [`tests/phase5_test.cpp`](../../../../tests/phase5_test.cpp)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real header.

---

## 1. What this file is

Per-tensor post-training quantization of `Matrix` data to 8- or 4-bit integers:

| Mode | Range | Formula |
|---|---|---|
| `SYMMETRIC_INT8` / `SYMMETRIC_INT4` | [−127, 127] / [−7, 7] | `scale = max|x| / qmax`, `zero_point = 0` |
| `ASYMMETRIC_INT8` / `ASYMMETRIC_INT4` | [0, 255] / [0, 15] | `scale = (max − min) / (qmax − qmin)`, `zero_point = qmin − round(min / scale)`, **clamped to [qmin, qmax]** |

`Quantizer(mode, calibration = MIN_MAX, percentile = 0.999)`: `calibrate(data)` (min/max, or a percentile range), `quantize_value()`/
`dequantize_value()`, vector and matrix `quantize()`/`dequantize()` (values stored in `std::vector<int8_t>`; `ASYMMETRIC_INT8`'s 128–255 are decoded as
unsigned, the TD-075 fix), `compute_quantization_error()` (MSE). `QuantizedMatrix`: `quantize_from(matrix, quantizer)`, `dequantize(quantizer)`,
`memory_reduction()`, `save()`/`load()` (rows, cols, scale, zero point, qmin, qmax, size, bytes). `print_quantization_stats()` prints to `std::cout`.

---

## 2. Problems (verified)

### Asymmetric modes lose everything when the data doesn't include 0

The zero point is clamped to `[qmin, qmax]` instead of widening the calibration range to include 0 (the standard practice). For data in [5, 10],
`zero_point = −255` was clamped to 0, so every value quantized to 255 and the round trip returned **5.000 for all five inputs** (5, 6, 7.5, 9, 10).
Any tensor whose values are all positive or all negative (post-ReLU activations, for example) is destroyed, not approximated.

### A saved `ASYMMETRIC_INT8` matrix decodes wrongly unless the caller remembers its mode

The mode isn't stored in `QuantizationParams` or in the file, and `dequantize()` uses the `Quantizer` passed to it to decide whether the stored bytes
are signed or unsigned. Reloading an `ASYMMETRIC_INT8` matrix of `0, 5, 15, 20` and dequantizing with a default (symmetric) `Quantizer` gave
**0.000, 5.020, −5.098, −0.078**: exactly the TD-075 corruption that fix was meant to remove. The saved `qmin`/`qmax` already identify the range and could
decide this instead.

### INT4 gives 4×, not 8×

INT4 values are stored one per byte (no packing): `memory_reduction()` reported **3.98×** for `SYMMETRIC_INT4`, while the header and the old API doc
promise 8×.

### Advertised features that don't exist (by inspection)

- **Per-channel quantization** is described in the header ("Different scale/zero-point per output channel"); only per-tensor exists.
- **`CalibrationMethod::MSE`** silently falls back to min/max.

### Smaller items (by inspection)

- `load()` checks no reads, accepts a negative or oversized `size`, and doesn't compare `size` with `rows × cols` (that surfaces later as a dequantize
  error).
- `memory_reduction()` uses `int` byte counts (overflow above ~536M elements).
- `print_quantization_stats()` writes to `std::cout`.

---

## 3. Tests

`quantization_test.cpp` (6 tests): symmetric and asymmetric round trips including the TD-075 above-127 cases, a `QuantizedMatrix` asymmetric round trip,
INT4 round trips. `phase5_test.cpp` (10 tests): defaults, min/max calibration, quantize/dequantize, asymmetric mode, matrices, error, save/load, memory
reduction, INT4, percentile calibration.

**Not covered:** data that excludes 0 in asymmetric modes, reloading with a different mode, the INT4 ratio.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Asymmetric zero point clamped instead of widening the range to include 0 (§2) | All-positive or all-negative tensors collapse to one value | **Verified** | [TD-483](../../guides/TECHNICAL_DEBT.md#td-483-quantization-asymmetric-range-mode-less-files-and-unpacked-int4) |
| Mode not saved; `dequantize()` trusts the caller's `Quantizer` (§2) | A reloaded `ASYMMETRIC_INT8` matrix decodes with TD-075's sign corruption | **Verified** | [TD-483](../../guides/TECHNICAL_DEBT.md#td-483-quantization-asymmetric-range-mode-less-files-and-unpacked-int4) |
| INT4 not packed (3.98×, not 8×); per-channel and MSE calibration advertised but absent (§2) | Misleading claims | **Verified** (ratio); by inspection | [TD-483](../../guides/TECHNICAL_DEBT.md#td-483-quantization-asymmetric-range-mode-less-files-and-unpacked-int4) |
| Unchecked `load()`; `int` byte counts; console output (§2) | Hygiene | By inspection | [TD-483](../../guides/TECHNICAL_DEBT.md#td-483-quantization-asymmetric-range-mode-less-files-and-unpacked-int4) |
| No production use | Context: a quantization tool is deferred | By inspection | TD-014 |

---

## 5. Merge notes

This file replaces `docs/development/api/advanced/quantization.md` (418 lines). Where they disagree, this file wins: it promised "4–8× memory reduction",
an INT4 row at "8×", and an MSE calibration method; INT4 is unpacked (4×) and MSE falls back to min/max. Kept: the modes, formulas and the save format.
Dropped: the size/accuracy tables (not measured in this codebase) and tutorial material.
