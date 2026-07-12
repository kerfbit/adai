# HuggingFace SafeTensors Compatibility — Design Proposal

**Status:** Draft  
**Target:** ADAI encoder-decoder transformer  
**Date:** 2026-05-17

---

## 1. Objective

Implement bi-directional model weight interoperability between ADAI's native binary format
and the HuggingFace **SafeTensors** ecosystem. The scope is:

- **Export** — write a standard `.safetensors` file and `config.json` from a loaded ADAI model
- **Import** — read a HuggingFace-style `.safetensors` file (T5/BART-family architecture) into
  a live ADAI model

Neither path should alter ADAI's internal training loop, native save/load paths, or on-disk
binary format. Both paths are pure translation layers on top of the existing component APIs.

---

## 2. Background: Current ADAI Format

### 2.1 File layout

A fully-saved ADAI model produces the following files (where `<root>` is the base path):

```
<root>.config                         # 10 × int32, raw binary, host endian
<root>.vocab                          # BPE vocabulary text file
<root>.encoder                        # LLMEncoder composite binary
<root>.decoder                        # LLMDecoder composite binary
<root>.lm_head                        # LanguageModelHead binary
<root>_decoder_block_N.bin.self_attn  # per-block self-attention (from DecoderBlock::save)
<root>_decoder_block_N.bin.cross_attn # per-block cross-attention
<root>_decoder_block_N.bin.ff         # per-block feed-forward
```

### 2.2 Per-component binary layout

Each component file begins with dimension integers for validation, followed by raw `float32`
values in row-major order. No magic bytes, no version field, no endian annotation.

| Component | Members serialised | Shape (rows × cols) |
|---|---|---|
| `MultiHeadAttention` | d_model, num_heads, W_q, W_k, W_v, W_o | [d_model, d_model] each |
| `CrossAttention` | d_model, num_heads, W_q, W_k, W_v, W_o | [d_model, d_model] each |
| `FeedForward` | d_model, d_ff, W1, W2, b1, b2 | W1:[d_model,d_ff] W2:[d_ff,d_model] |
| `LayerNorm` | dim, eps, gamma, beta | [1, d_model] each |
| `TokenEmbedding` | vocab_size, d_model, table | [vocab_size, d_model] |
| `LanguageModelHead` | d_model, vocab_size, W_output, bias | W:[d_model,vocab_size] bias:[1,vocab_size] |
| `EncoderBlock` | d_model, num_heads, d_ff, dropout, norm1γ, norm1β, norm2γ, norm2β | (inline in block file + sub-files) |
| `DecoderBlock` | d_model, num_heads, d_ff, dropout, lr, norm1γ/β, norm2γ/β, norm3γ/β | (inline in block file + sub-files) |

### 2.3 Key limitations vs. industry formats

- No magic number or format version → unidentifiable by external tools
- Host native endianness → portable only between same-endian machines
- Positional (unnamed) tensors → no introspection possible
- Split across many files → tooling expects one or a handful of files
- Always `float32`, no dtype or quantisation metadata

---

## 3. Target Format: SafeTensors + HuggingFace config.json

### 3.1 SafeTensors binary layout

```
[ 8 bytes: uint64 LE header_size N ]
[ N bytes: UTF-8 JSON header       ]
[ tensor data: packed, row-major   ]
```

The JSON header maps each named tensor to a descriptor:

```json
{
  "__metadata__": { "format": "pt", "adai_version": "1.0" },
  "model.encoder.block.0.layer.0.SelfAttention.q.weight": {
    "dtype":        "F32",
    "shape":        [512, 512],
    "data_offsets": [0, 1048576]
  },
  ...
}
```

All values are **little-endian**. Tensors are laid out contiguously after the header with no
gaps. The format requires all offsets in `data_offsets` to be byte-precise; they are relative
to the start of the tensor data region (i.e., offset 0 = first byte after the header).

Reference: https://huggingface.co/docs/safetensors

### 3.2 HuggingFace config.json

ADAI's architecture maps most naturally onto the T5 (`T5ForConditionalGeneration`) family.
The following config.json fields are needed for any external framework to reconstruct the
model architecture from the weights file:

```json
{
  "architectures":        ["T5ForConditionalGeneration"],
  "model_type":           "t5",
  "adai_native":          true,
  "vocab_size":           <vocab_size>,
  "d_model":              <d_model>,
  "d_ff":                 <d_ff>,
  "d_kv":                 <d_model / num_heads>,
  "num_heads":            <num_heads>,
  "num_layers":           <encoder_layers>,
  "num_decoder_layers":   <decoder_layers>,
  "max_length":           <max_seq_length>,
  "decoder_start_token_id": <bos_token_id>,
  "eos_token_id":         <eos_token_id>,
  "pad_token_id":         <pad_token_id>,
  "feed_forward_proj":    "relu",
  "torch_dtype":          "float32"
}
```

The `adai_native: true` flag allows ADAI's own import path to distinguish its own exports
from third-party T5 checkpoints, which may differ slightly (e.g., relative positional biases
not present in ADAI).

---

## 4. Tensor Naming Convention

### 4.1 Shape transpose rule

ADAI stores weight matrices as **right-multiplied** operands: `output = input @ W`.  
HuggingFace/PyTorch stores them **transposed**: `output = F.linear(input, W) = input @ W.T`.

Therefore, every weight matrix **must be transposed** on both export and import.
Bias vectors and LayerNorm parameters are 1-D and require **no transposition**.

| ADAI member | ADAI shape | HF shape (after transpose) |
|---|---|---|
| `W_q`, `W_k`, `W_v`, `W_o` | [d_model, d_model] | [d_model, d_model] |
| `W1` | [d_model, d_ff] | [d_ff, d_model] |
| `W2` | [d_ff, d_model] | [d_model, d_ff] |
| `W_output` (LMHead) | [d_model, vocab_size] | [vocab_size, d_model] |
| `embedding_table` | [vocab_size, d_model] | [vocab_size, d_model] (no change) |
| `gamma`, `beta` (LayerNorm) | [1, d_model] → flatten | [d_model] |
| `bias` (LMHead) | [1, vocab_size] → flatten | [vocab_size] |
| `b1` (FF) | [1, d_ff] → flatten | [d_ff] |
| `b2` (FF) | [1, d_model] → flatten | [d_model] |

Note: attention weight matrices are square so the transposed shape is identical, but the
values still differ and must be transposed.

### 4.2 Full tensor name map

Using T5-style naming under the `model.` namespace:

**Shared embedding**
```
model.shared.weight                         [vocab_size, d_model]
```

**Encoder — per layer i = 0 … encoder_layers-1**
```
model.encoder.block.{i}.layer.0.SelfAttention.q.weight       [d_model, d_model]
model.encoder.block.{i}.layer.0.SelfAttention.k.weight       [d_model, d_model]
model.encoder.block.{i}.layer.0.SelfAttention.v.weight       [d_model, d_model]
model.encoder.block.{i}.layer.0.SelfAttention.o.weight       [d_model, d_model]
model.encoder.block.{i}.layer.0.layer_norm.weight            [d_model]    (gamma)
model.encoder.block.{i}.layer.0.layer_norm.bias              [d_model]    (beta)
model.encoder.block.{i}.layer.1.DenseReluDense.wi.weight     [d_ff, d_model]
model.encoder.block.{i}.layer.1.DenseReluDense.wi.bias       [d_ff]
model.encoder.block.{i}.layer.1.DenseReluDense.wo.weight     [d_model, d_ff]
model.encoder.block.{i}.layer.1.DenseReluDense.wo.bias       [d_model]
model.encoder.block.{i}.layer.1.layer_norm.weight            [d_model]    (gamma)
model.encoder.block.{i}.layer.1.layer_norm.bias              [d_model]    (beta)
```

**Encoder — final norm** (LLMEncoder applies a final LayerNorm after all blocks; confirm
presence in `LLMEncoder::save_weights` before including this tensor)
```
model.encoder.final_layer_norm.weight                        [d_model]
model.encoder.final_layer_norm.bias                         [d_model]
```

**Decoder — per layer j = 0 … decoder_layers-1**
```
model.decoder.block.{j}.layer.0.SelfAttention.q.weight      [d_model, d_model]
model.decoder.block.{j}.layer.0.SelfAttention.k.weight      [d_model, d_model]
model.decoder.block.{j}.layer.0.SelfAttention.v.weight      [d_model, d_model]
model.decoder.block.{j}.layer.0.SelfAttention.o.weight      [d_model, d_model]
model.decoder.block.{j}.layer.0.layer_norm.weight           [d_model]    (norm1 gamma)
model.decoder.block.{j}.layer.0.layer_norm.bias             [d_model]    (norm1 beta)
model.decoder.block.{j}.layer.1.EncDecAttention.q.weight    [d_model, d_model]
model.decoder.block.{j}.layer.1.EncDecAttention.k.weight    [d_model, d_model]
model.decoder.block.{j}.layer.1.EncDecAttention.v.weight    [d_model, d_model]
model.decoder.block.{j}.layer.1.EncDecAttention.o.weight    [d_model, d_model]
model.decoder.block.{j}.layer.1.layer_norm.weight           [d_model]    (norm2 gamma)
model.decoder.block.{j}.layer.1.layer_norm.bias             [d_model]    (norm2 beta)
model.decoder.block.{j}.layer.2.DenseReluDense.wi.weight    [d_ff, d_model]
model.decoder.block.{j}.layer.2.DenseReluDense.wi.bias      [d_ff]
model.decoder.block.{j}.layer.2.DenseReluDense.wo.weight    [d_model, d_ff]
model.decoder.block.{j}.layer.2.DenseReluDense.wo.bias      [d_model]
model.decoder.block.{j}.layer.2.layer_norm.weight           [d_model]    (norm3 gamma)
model.decoder.block.{j}.layer.2.layer_norm.bias             [d_model]    (norm3 beta)
```

**Decoder — final norm**
```
model.decoder.final_layer_norm.weight                       [d_model]
model.decoder.final_layer_norm.bias                         [d_model]
```

**LM Head**
```
lm_head.weight                                              [vocab_size, d_model]
lm_head.bias                                                [vocab_size]
```

---

## 5. Implementation Plan

### 5.1 New component: `ModelSerializer`

Create `src/ModelSerializer.{cpp,hpp}`. This is the sole new production file. It does not
participate in training or inference; it is a pure I/O translation layer.

```cpp
class ModelSerializer {
public:
    // Export a loaded EncoderDecoderModel to SafeTensors + config.json
    static void export_safetensors(const EncoderDecoderModel& model,
                                   const std::string& output_dir);

    // Import a HuggingFace-style SafeTensors file into a live ADAI model.
    // Expects output_dir to contain model.safetensors + config.json.
    static void import_safetensors(EncoderDecoderModel& model,
                                   const std::string& input_dir);
};
```

`export_safetensors` writes two files:
- `<output_dir>/model.safetensors`
- `<output_dir>/config.json`

`import_safetensors` reads those same two files.

### 5.2 Internal helper: `SafeTensorsIO`

A private `SafeTensorsIO` struct handles the low-level binary protocol:

```cpp
struct TensorDescriptor {
    std::string name;
    std::string dtype;       // "F32"
    std::vector<int64_t> shape;
    std::vector<float> data; // always converted to/from float32
};

struct SafeTensorsIO {
    // Write a set of named tensors to a SafeTensors file (little-endian output)
    static void write(const std::string& path,
                      const std::vector<TensorDescriptor>& tensors,
                      const std::map<std::string, std::string>& metadata);

    // Read all tensors from a SafeTensors file; returns map name → descriptor
    static std::map<std::string, TensorDescriptor>
    read(const std::string& path);
};
```

The writer must:
1. Build the JSON header using the accumulated tensor list and computed byte offsets
2. Write the 8-byte LE header length
3. Write the JSON header (padded with spaces to align tensor data to 8 bytes — required by spec)
4. Write each tensor's raw float data in native float32 converted to little-endian

The reader must:
1. Read the 8-byte LE header length
2. Parse the JSON header (no external JSON library required; a minimal parser suffices
   since the structure is well-defined, or nlohmann/json can be vendored)
3. For each requested tensor name, seek to the data offset and read the float data

### 5.3 Tensor accessor changes required in existing classes

The export path needs read access to weights. Most classes already expose them via
`get_gamma()` / `get_beta()` on `LayerNorm`. The following new getters must be added:

| Class | New getter needed |
|---|---|
| `MultiHeadAttention` | `get_Wq()`, `get_Wk()`, `get_Wv()`, `get_Wo()` returning `const Matrix&` |
| `CrossAttention` | same four |
| `FeedForward` | `get_W1()`, `get_W2()`, `get_b1()`, `get_b2()` |
| `TokenEmbedding` | `get_embeddings()` already exists |
| `LanguageModelHead` | `get_W_output()`, `get_bias()` |
| `LLMEncoder` | `get_final_norm()` if a final LayerNorm exists |
| `LLMDecoder` | `get_final_norm()` if a final LayerNorm exists |

The import path needs setters (or load-from-buffer overloads):

| Class | New setter needed |
|---|---|
| `MultiHeadAttention` | `set_Wq()`, `set_Wk()`, `set_Wv()`, `set_Wo()` accepting `const Matrix&` |
| `CrossAttention` | same four |
| `FeedForward` | `set_W1()`, `set_W2()`, `set_b1()`, `set_b2()` |
| `TokenEmbedding` | `set_embeddings(const Matrix&)` |
| `LanguageModelHead` | `set_W_output()`, `set_bias()` |
| `LayerNorm` | `set_gamma()` / `set_beta()` already exist |

### 5.4 Transpose helper

A free function `Matrix transpose(const Matrix&)` may already exist; if not, add it
to `Matrix.hpp`. Export calls `transpose(W)` on every weight matrix before recording
the tensor. Import calls `transpose` on every weight matrix after reading it, before
passing it to the setter.

### 5.5 Endianness

ADAI targets Linux x86-64 (little-endian). SafeTensors is also little-endian. No byte
swapping is required on this platform. Document this assumption in the implementation with
a static assertion:

```cpp
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
    "SafeTensors I/O assumes little-endian host; add swap logic for big-endian.");
```

### 5.6 JSON serialisation

Rather than vendoring a JSON library just for this component, a minimal hand-written
JSON emitter and a simple key-value JSON parser are sufficient given the predictable
structure of SafeTensors headers. Use `nlohmann/json` (single-header, already common
in HuggingFace tooling) if the team prefers a safer parser; it can be placed in
`external/nlohmann/json.hpp`.

### 5.7 Tokenizer export

HuggingFace expects `tokenizer.json` and `tokenizer_config.json`. ADAI's BPE vocabulary
is in `vocab.txt` (merge rules + token list). A minimal `tokenizer_config.json` should
be written to indicate the tokenizer class; full conversion to HuggingFace tokenizer
format is out of scope for this proposal and can be handled separately.

---

## 6. Export Walkthrough

```
export_safetensors(model, "/path/to/export/")
│
├─ Collect all TensorDescriptors:
│    For each encoder layer i:
│      read norm1.gamma → [d_model], name = "model.encoder.block.{i}.layer.0.layer_norm.weight"
│      read attn.W_q   → transpose → shape [d_model,d_model]
│      ...
│    For each decoder layer j:
│      ... (self_attn, cross_attn, ff, 3 norms)
│    LMHead W_output → transpose → [vocab_size, d_model], name = "lm_head.weight"
│    LMHead bias    → flatten → [vocab_size],             name = "lm_head.bias"
│    embedding_table → no transpose → [vocab_size, d_model], name = "model.shared.weight"
│
├─ SafeTensorsIO::write("/path/to/export/model.safetensors", tensors, metadata)
│    Build JSON header with data_offsets
│    Write 8-byte LE uint64 header length
│    Write JSON header (space-padded to 8-byte alignment)
│    Write tensor data sequentially
│
└─ Write config.json with architecture parameters from model's public getters
```

---

## 7. Import Walkthrough

```
import_safetensors(model, "/path/to/import/")
│
├─ Read config.json → verify d_model, num_heads, d_ff, layers match live model
│    Mismatch → throw descriptive error (same pattern as load_model)
│
├─ SafeTensorsIO::read("/path/to/import/model.safetensors")
│    Returns map<string, TensorDescriptor>
│
├─ For each encoder layer i:
│    Look up "model.encoder.block.{i}.layer.0.SelfAttention.q.weight"
│    Verify shape == [d_model, d_model]
│    transpose → set_Wq(transposed)
│    ...
│
├─ For each decoder layer j: (same pattern)
│
├─ LMHead:
│    look up "lm_head.weight" shape [vocab_size, d_model] → transpose → [d_model, vocab_size]
│    set_W_output(transposed)
│    look up "lm_head.bias"  shape [vocab_size] → set_bias
│
└─ Embedding:
│    look up "model.shared.weight" shape [vocab_size, d_model] → no transpose
│    set_embeddings(matrix)
```

---

## 8. Limitations and Out-of-Scope Items

| Item | Notes |
|---|---|
| Relative position biases (T5) | T5 uses learnable relative position biases; ADAI uses sinusoidal positional encoding. These tensors will be absent in ADAI exports. Importing a real T5 checkpoint will silently skip position bias tensors. |
| Quantised weights (int8/int4) | SafeTensors supports INT8/INT4 dtypes. This proposal is float32 only. Quantisation support is tracked separately in `Quantization.hpp`. |
| `tokenizer.json` | Full HuggingFace tokenizer format conversion is a separate task. |
| Sharded checkpoints (`model-00001-of-00002.safetensors`) | Out of scope; ADAI models are small enough for a single file. |
| Weight sharing (embedding ↔ lm_head) | ADAI does not currently share these weights. If sharing is added, export should write only one tensor and flag it in `__metadata__`. |
| Mixed precision (BF16) | Out of scope for this proposal. |
| External T5 checkpoint import | The naming map aligns with T5, but real T5 checkpoints include relative position bias tensors that ADAI lacks. Import will succeed but silently ignore unknown tensor keys. |

---

## 9. Files To Create / Modify

| Action | Path |
|---|---|
| Create | `src/ModelSerializer.hpp` |
| Create | `src/ModelSerializer.cpp` |
| Create | `tests/model_serializer_test.cpp` |
| Modify | `src/MultiHeadAttention.hpp` — add getters/setters |
| Modify | `src/CrossAttention.hpp` — add getters/setters |
| Modify | `src/FeedForward.hpp` — add getters/setters |
| Modify | `src/LanguageModelHead.hpp` — add getters/setters |
| Modify | `src/TokenEmbedding.hpp` — add set_embeddings |
| Modify | `src/CMakeLists.txt` — add ModelSerializer to library sources |
| Modify | `tests/CMakeLists.txt` — add model_serializer_test |

### 9.1 CMakeLists.txt addition

```cmake
# In src/CMakeLists.txt, add to the adai_lib target sources:
ModelSerializer.cpp
```

---

## 10. Test Plan

`tests/model_serializer_test.cpp` should cover:

1. **Round-trip test** — construct a small ADAI model (d_model=64, 2 enc/dec layers),
   export to a temp directory, import back, verify all weight tensors are numerically
   equal (epsilon = 1e-6 for float32 round-trip)

2. **Header parsing** — write a known SafeTensors file by hand (few tensors), read it
   back, verify shapes and values

3. **Transpose correctness** — verify that `(W.transpose()).transpose() == W` for all
   weight matrices

4. **Config mismatch** — import into a model with wrong d_model, assert exception thrown

5. **Unknown tensor key** — import a SafeTensors file with extra unknown keys, assert
   it succeeds (keys are silently ignored)

6. **Missing required tensor** — import a file missing a required key, assert exception

---

## 11. Dependency Summary

No new external dependencies are strictly required. Options for JSON handling:

| Option | Pros | Cons |
|---|---|---|
| Hand-written minimal JSON | No new files | Fragile for unusual inputs |
| `nlohmann/json` single header | Robust, well-tested | +~1MB header in `external/` |
| `rapidjson` | Fast, small | Less ergonomic API |

Recommendation: vendor `nlohmann/json.hpp` into `external/nlohmann/` (MIT license,
single file, already widely used in HuggingFace tooling).
