# `BinaryIO` — Source File Reference

- **Files:** [`src/BinaryIO.hpp`](../../../../src/BinaryIO.hpp), [`src/BinaryIO.cpp`](../../../../src/BinaryIO.cpp) (`adai_core`)
- **Namespace:** `adai::binio` (`Reader`, `Writer`)
- **Used by:** every checkpoint sidecar written by `EncoderDecoderModel::save_model()` and read by `load_model()`:
  [`EncoderDecoderModel`](EncoderDecoderModel.md) (`.config`), [`LLMEncoder`](LLMEncoder.md), [`Decoder`](Decoder.md),
  [`EncoderBlock`](EncoderBlock.md), [`DecoderBlock`](DecoderBlock.md), [`MultiHeadAttention`](MultiHeadAttention.md),
  [`CrossAttention`](CrossAttention.md), [`FeedForward`](FeedForward.md), [`LayerNorm`](LayerNorm.md),
  [`TokenEmbedding`](TokenEmbedding.md), [`LanguageModelHead`](LanguageModelHead.md)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-10-26`
- **Tests:** [`tests/binaryio_test.cpp`](../../../../tests/binaryio_test.cpp) (5 tests, `binaryioTests`), plus
  `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`
- **Last traced against the code:** 2026-10-26

> Code-traced reference, written with the file (TD-423, October 26, 2026).

---

## What it does

Every checkpoint loader used to read with unchecked `ifstream::read()` calls. A sidecar cut short by a full disk or a crash during save
loaded silently, with NaN or stale weights past the cut, and the server generated garbage instead of refusing to start. Writes weren't
checked either (TD-423).

| Call | Behavior |
|---|---|
| `Reader(path)` | opens the file; throws `std::runtime_error` if it can't |
| `value<T>(what)` | reads one trivially copyable value; a short read throws, naming the file and `what` |
| `matrix(rows, cols, what)` | reads a row-major float matrix (one read per row); throws on a short read or a negative shape |
| `expect_end()` | throws if bytes remain, so a file that doesn't match the model's shapes is rejected |
| `Writer(path)` | opens for writing; throws if it can't |
| `value(v, what)`, `matrix(m, what)` | write; throw as soon as the stream reports a failure |
| `close()` | flushes and closes; throws if any write or the flush failed (buffered failures, e.g. `/dev/full`, surface here) |

## How loaders use it

Each loader reads its whole file into local values and matrices, validates the dimensions, calls `expect_end()`, and only then assigns.
A failed load therefore leaves that layer unchanged. `EncoderDecoderModel::load_model()` loads its files one after another, so a
failure in a later file leaves earlier layers already replaced; callers treat the exception as a failed load.

Not covered: the `.vocab` text file (TD-223), the `.optimizer` training state, `LeJEPAEncoder`, `HippocampalMemory`, `Optimizer` and
the RLHF/LoRA/quantization formats.
