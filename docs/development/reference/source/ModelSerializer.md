# `ModelSerializer` — Source File Reference

- **Files:** [`src/ModelSerializer.hpp`](../../../../src/ModelSerializer.hpp), [`src/ModelSerializer.cpp`](../../../../src/ModelSerializer.cpp) (665 lines together)
- **Namespace:** global (`ModelSerializer`, `ModelSerializer::TensorDescriptor`)
- **Built into:** `adai_models`; uses the vendored `nlohmann/json`
- **Used by:** **nothing in production**: no binary, flag or endpoint calls `export_safetensors()`/`import_safetensors()`; only [`tests/model_serializer_test.cpp`](../../../../tests/model_serializer_test.cpp) (6 tests)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10` (both files)
- **Proposal:** "HuggingFace SafeTensors Compatibility" in [docs/proposals/README.md](../../../proposals/README.md)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linked against the real
> model and serializer.

---

## 1. What this file is

Weight export and import between ADAI's `EncoderDecoderModel` and the
[SafeTensors](https://huggingface.co/docs/safetensors) file format, with tensor names following HuggingFace's
`T5ForConditionalGeneration` convention, plus a `config.json`:

```text
<dir>/model.safetensors   8-byte LE header length, JSON header (dtype/shape/data_offsets + __metadata__), F32 data
<dir>/config.json         T5-style: architectures, model_type "t5", vocab_size, d_model, d_ff, d_kv, num_heads,
                          num_layers, num_decoder_layers, max_length, special token IDs, feed_forward_proj "relu"
```

| Function | Behaviour |
|---|---|
| `write_safetensors(path, tensors, metadata)` | Builds the header with `nlohmann/json`, pads it to 8 bytes, writes header and F32 data; checks the stream |
| `read_safetensors(path)` | Reads every tensor into F32 `TensorDescriptor`s |
| `export_safetensors(model, dir)` | `model.shared.weight` (the **encoder's** embedding), per encoder block Q/K/V/O, norms, FFN `wi`/`wo` weights **and biases**; per decoder block the same plus `EncDecAttention`; both final norms; `lm_head.weight` and `lm_head.bias`. Linear weights are transposed (ADAI right-multiplies) |
| `import_safetensors(model, dir)` | Checks five `config.json` fields against the live model, then `require()`s every tensor above (by element count), transposes back, and sets it. `model.shared.weight` is written to **both** the encoder's and the decoder's embedding. Final norms are optional; unknown keys are ignored |

---

## 2. Problems

### A round trip changes the model (verified)

ADAI's encoder and decoder have **separate, separately trained** token embeddings (`LLMDecoder` owns its own
`TokenEmbedding` and updates it in `backward()`/`update_weights()`). Export writes only the encoder's as
`model.shared.weight`, and import copies it into both. In a probe, the original model's two embeddings differed by up to
0.85; after `export_safetensors()` → `import_safetensors()` into a fresh model, the decoder embedding was the encoder's, and
**the logits for the same input differed by up to 1.05** from the original. `FullModelRoundTrip` passes because it compares
only three weight matrices (encoder Q, decoder Q, LM head), not the embeddings or outputs. The exported decoder embedding
needs its own tensor (e.g. `decoder.embed_tokens.weight`, the name HF uses for untied T5 variants).

Other state isn't exported either: the TD-180 gated world-model/hippocampal layers (TD-345), and the tokenizer vocabulary,
without which the weights can't be used.

### The "HuggingFace T5" format isn't loadable as T5, in either direction (by inspection)

The names and `config.json` claim `T5ForConditionalGeneration`, but the architecture differs from T5 in ways the file
doesn't describe:
- ADAI uses sinusoidal positional encoding; T5 uses learned relative attention bias, which `transformers` would initialize
  **randomly** for an export (`relative_attention_bias` is absent).
- ADAI's LayerNorms have a bias and subtract the mean; T5's `T5LayerNorm` is RMSNorm with weight only.
- ADAI's FFN uses **GELU** with biases; the config says `feed_forward_proj: "relu"`, and T5 linears have no biases.
- ADAI scales attention by 1/√d_k; T5 doesn't. ADAI's LM head is untied and has a bias.

So an export loaded into `transformers` would warn about unexpected keys and compute something different, and importing a
real T5 checkpoint **always fails**: `require()` demands the bias tensors (`layer_norm.bias`, `DenseReluDense.wi.bias`,
`lm_head.bias`, …) that T5 checkpoints don't have. The header's "Unknown tensor keys … are silently ignored (e.g. T5 position
biases)" describes a case that can't be reached. In practice this is an ADAI-specific checkpoint format using T5 names; calling
it HuggingFace interoperability is misleading. (`transformers` isn't installed here, so loading into it wasn't tried.)

**Decision (October 14, 2026):** make it T5-faithful (TD-466). Some differences can be handled in the serializer (fold the
1/√d_k scale into `q.weight`, `tie_word_embeddings: false`, `feed_forward_proj: "gelu"`), but RMSNorm, bias-free linears, relative
position bias and a shared embedding have no T5 representation, so faithfulness needs a T5-style architecture variant of the
model, selected per model (architecture is immutable in MNS), plus verification against `transformers`.

### Smaller items (by inspection)

- **dtype is ignored.** `read_safetensors()` reads every tensor as F32 (`n_bytes / 4` floats); an F16/BF16 file fails later with
  "size mismatch in from_flat" rather than a dtype error.
- **No header validation.** A huge or corrupt header length, or `data_offsets` with `end < start`, become giant allocations
  (`bad_alloc`/`length_error`) instead of a format error; offsets aren't checked against the file size before reading.
- **Partial imports.** Weights are set as they're read, so a missing tensor halfway through leaves the live model half-overwritten.
- **Dead code:** `json_escape()` returns its input unchanged and is never called.
- **`stable` without a production caller.** Nothing in the shipped binaries uses this; the status tag overstates its maturity.

---

## 3. Tests

`model_serializer_test.cpp` (6 tests): protocol round trip, a full-model round trip (three matrices), transpose symmetry,
config mismatch, unknown keys ignored, a missing required tensor.

**Not covered:** the decoder embedding or model outputs after a round trip; non-F32 dtypes; corrupt headers.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Export writes only the encoder's embedding; import copies it to the decoder (§2) | A round trip changes the model's outputs (up to 1.05 in logits) | **Verified** | [TD-465](../../guides/TECHNICAL_DEBT.md#td-465-safetensors-round-trip-replaces-the-decoder-embedding) |
| "T5" names/config for a non-T5 architecture: no relative bias, RMSNorm vs LayerNorm, GELU vs "relu", biases, scaling (§2); decision: make it T5-faithful | Exports aren't usable in `transformers`; real T5 checkpoints can never be imported | By inspection | [TD-466](../../guides/TECHNICAL_DEBT.md#td-466-modelserializer-claims-t5-compatibility-it-doesnt-have-and-skips-input-validation) |
| dtype ignored; no header/offset validation; partial imports; dead `json_escape()`; `stable` with no production caller (§2) | Cryptic errors on real files; half-overwritten models | By inspection | [TD-466](../../guides/TECHNICAL_DEBT.md#td-466-modelserializer-claims-t5-compatibility-it-doesnt-have-and-skips-input-validation) |
| Gated world-model/hippocampal layers not exported | Already tracked | — | TD-345 |
