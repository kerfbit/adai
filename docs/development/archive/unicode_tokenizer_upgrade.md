# Unicode-Aware Tokenizer Upgrade — Design Proposal

**Status:** Draft  
**Target:** `BPETokenizer` (`src/BPETokenizer.{cpp,hpp}`)  
**Date:** 2026-05-23  
**Related:** [huggingface-safetensors-compatibility.md](huggingface-safetensors-compatibility.md)

---

## 1. Objective

Extend `BPETokenizer` with an **optional Unicode-aware mode** that makes it a drop-in
replacement for the tokenizers shipped alongside HuggingFace model checkpoints. The existing
ASCII-BPE path must remain unchanged and continue to pass all current tests.

Two HuggingFace tokenizer families are in scope:

| Family | Representative models | Key mechanism |
|--------|----------------------|---------------|
| **Byte-level BPE** | GPT-2, RoBERTa, LLaMA, Mistral | All 256 raw bytes mapped to printable Unicode; no `<unk>` possible |
| **SentencePiece (Unigram / BPE)** | T5, mT5, BERT-multilingual | Unicode code-point segments; `▁` (U+2581) as space marker |

---

## 2. Background and Motivation

### 2.1 Current limitations

The existing `BPETokenizer` was designed for English-only training data:

- Pre-tokenisation regex `[A-Za-z]+|[0-9]+|…` matches only ASCII letters and digits.
- `std::tolower` over `unsigned char` is only correct for ASCII; multi-byte UTF-8 sequences
  are not case-folded.
- Multi-byte Unicode codepoints that survive the regex are fed byte-by-byte into `apply_bpe`,
  generating nonsensical subword splits and collapsing to `<unk>` if the byte subsequences
  are absent from the vocabulary.
- The vocabulary file format has no field for tokenizer family, normalization mode, or
  byte-to-unicode mapping table — making round-trip import/export impossible.

### 2.2 Why this matters for HuggingFace weight loading

The [SafeTensors compatibility proposal](huggingface-safetensors-compatibility.md) enables
loading pre-trained weight tensors into ADAI's model layers. Those weights are only useful
if text is tokenized identically to how the original model was trained. A mismatch of even
one token ID produces garbage output at every position.

Concretely:

- A GPT-2 checkpoint encodes `" Hello"` as token `15496` using byte-level BPE with the
  GPT-2 vocabulary (50 257 entries, all encoded via the 256-byte base alphabet).
- The current ADAI tokenizer would encode the same string as a sequence of character-level
  subword tokens from a custom vocabulary — completely different IDs.

The upgrade proposed here closes that gap by allowing `BPETokenizer` to load and execute
a pre-trained HuggingFace vocabulary and merge list directly.

---

## 3. Design

### 3.1 Tokenizer mode enum

```cpp
enum class TokenizerMode {
    ASCII_BPE,        // Current behaviour — unchanged
    BYTE_LEVEL_BPE,   // GPT-2 / RoBERTa / LLaMA style
    SENTENCEPIECE,    // T5 / mT5 / BERT-multilingual style (future)
};
```

`ASCII_BPE` is the default. The other modes are opt-in and do not affect any existing code
path.

### 3.2 Unicode configuration

```cpp
struct UnicodeConfig {
    TokenizerMode mode            = TokenizerMode::ASCII_BPE;
    bool          nfc_normalize   = false;  // apply NFC before tokenising
    bool          lowercase       = false;  // unicode-aware lowercasing (BERT style)
    bool          add_prefix_space = false; // prepend a space before the first word
};
```

### 3.3 Extended constructor

```cpp
explicit BPETokenizer(UnicodeConfig config = {});
```

The existing zero-argument constructor remains as a convenience alias with `ASCII_BPE`
defaults, guaranteeing source-level backwards compatibility.

### 3.4 Byte-level BPE path

GPT-2's byte-level BPE avoids `<unk>` by representing every possible input byte as a
printable Unicode character via a fixed 256-entry bijection (the `bytes_to_unicode()`
mapping standardised by HuggingFace).

The 256 base tokens are assigned as follows (pseudo-code from the reference implementation):

```
bs  = list(range(ord('!'), ord('~')+1))      # ! … ~   (94 entries)
   + list(range(ord('¡'), ord('¬')+1))       # ¡ … ¬   (44 entries)
   + list(range(ord('®'), ord('ÿ')+1))       # ® … ÿ   (50 entries)
cs  = bs[:]
n   = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
# bytes_to_unicode = dict(zip(bs, [chr(c) for c in cs]))
```

This 256-element table is stored as a `std::array<char32_t, 256>` (and its inverse) inside
`BPETokenizer` when the `BYTE_LEVEL_BPE` mode is active.

**Encoding pipeline (BYTE_LEVEL_BPE):**

1. Optionally NFC-normalise the UTF-8 input.
2. Apply the GPT-2 pre-tokenisation regex (Unicode-aware variant):
   `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
3. For each pre-token, encode every byte through `bytes_to_unicode`, yielding a string of
   surrogate-free printable characters.
4. Apply BPE merge rules as today.
5. Map resulting tokens to integer IDs via `vocab`.

**Decoding pipeline (BYTE_LEVEL_BPE):**

1. Concatenate the token strings.
2. Map each character back through `unicode_to_bytes`.
3. Interpret the resulting byte sequence as UTF-8 and return as `std::string`.

### 3.5 Unicode-aware regex

C++17 `<regex>` does not support `\p{L}` / `\p{N}` property escapes. The options are:

| Option | Pros | Cons |
|--------|------|------|
| ICU (`libicu`) | Full Unicode property support | Heavy dependency; adds ~30 MB |
| `re2` | Unicode properties, safe regex | Additional dependency |
| Hand-rolled UTF-8 category check | No dependency | Maintenance burden |
| **Byte-level pass-through** | **No regex change needed** | Only works for BYTE_LEVEL_BPE |

**Recommended approach:** For `BYTE_LEVEL_BPE`, map the entire input to the 256-character
byte alphabet **before** applying the pre-tokenisation regex. This transforms the problem
into ASCII-safe territory — the existing regex already handles multi-byte mapped characters
correctly because the byte encoding uses only printable Latin-1 codepoints. No extra Unicode
library is required.

For `SENTENCEPIECE` mode (future), an optional ICU dependency or bundled lightweight Unicode
table will be introduced under a CMake option (`ADAI_ENABLE_ICU`, default `OFF`).

### 3.6 HuggingFace `tokenizer.json` import/export

HuggingFace fast tokenizers serialise to a single `tokenizer.json` with the schema:

```json
{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": { "<token>": <id>, … },
    "merges": [ "<a> <b>", … ]
  },
  "normalizer": { "type": "NFC" | "Sequence" | null },
  "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": true },
  "decoder": { "type": "ByteLevel" },
  "added_tokens": [ { "id": 0, "content": "<pad>", "special": true }, … ]
}
```

Two new methods are proposed:

```cpp
// Import a HuggingFace fast tokenizer JSON
void load_hf_tokenizer(const std::string& tokenizer_json_path);

// Export current vocabulary + merges as a HuggingFace fast tokenizer JSON
void save_hf_tokenizer(const std::string& output_path) const;
```

`load_hf_tokenizer` will:

1. Parse `tokenizer.json` using a minimal JSON reader (the vendored `cpp-httplib` already
   includes `nlohmann/json.hpp` transitively; otherwise a single-header parser is added
   under `external/`).
2. Set `UnicodeConfig` based on the `pre_tokenizer.type` field.
3. Populate `vocab`, `inverse_vocab`, and `bpe_merges` from the `model` section.
4. Register `added_tokens` as special tokens.

`save_hf_tokenizer` produces a JSON that the `transformers` Python library can load with
`PreTrainedTokenizerFast.from_pretrained(path)`.

### 3.7 Special token remapping

Different model families use different names for the same logical roles:

| Role | ADAI | GPT-2 | LLaMA 2 | T5 |
|------|------|-------|---------|-----|
| Padding | `<pad>` (ID 0) | none | none | `<pad>` (ID 0) |
| Unknown | `<unk>` (ID 1) | none | `<unk>` (ID 0) | `<unk>` (ID 2) |
| Begin | `<bos>` (ID 2) | none | `<s>` (ID 1) | none |
| End | `<eos>` (ID 3) | `<|endoftext|>` (ID 50256) | `</s>` (ID 2) | `</s>` (ID 1) |

`load_hf_tokenizer` reads the `added_tokens` array and constructs a remapping table so that
`encode()` and `decode()` transparently use the loaded IDs. The existing `pad_token_id`,
`unk_token_id`, `bos_token_id`, and `eos_token_id` fields are updated to reflect the
imported values; the public accessor methods (`get_bos_token_id()`, etc.) continue to work
unchanged.

---

## 4. API Surface Changes

### 4.1 New / changed declarations in `BPETokenizer.hpp`

```cpp
// New: tokenizer mode and configuration
enum class TokenizerMode { ASCII_BPE, BYTE_LEVEL_BPE, SENTENCEPIECE };

struct UnicodeConfig {
    TokenizerMode mode             = TokenizerMode::ASCII_BPE;
    bool          nfc_normalize    = false;
    bool          lowercase        = false;
    bool          add_prefix_space = false;
};

class BPETokenizer {
public:
    // Existing constructor preserved as default
    BPETokenizer();

    // New constructor
    explicit BPETokenizer(UnicodeConfig config);

    // New import/export methods
    void load_hf_tokenizer(const std::string& tokenizer_json_path);
    void save_hf_tokenizer(const std::string& output_path) const;

    // New query
    TokenizerMode get_mode() const;

    // All existing public methods unchanged
    …
};
```

### 4.2 No breaking changes

- The existing `BPETokenizer()` constructor, all `encode` / `decode` / `tokenize` signatures,
  and the `load_vocab` / `save_vocab` format are **unchanged**.
- `ASCII_BPE` mode is always the default when no `UnicodeConfig` is supplied.
- The `UnicodeConfig` struct uses designated initializers with defaults, so partial
  initialization is safe.

---

## 5. Implementation Plan

### Phase 1 — Byte-level BPE core (prerequisite for HF weight loading)

| Task | File(s) |
|------|---------|
| Add `TokenizerMode` enum and `UnicodeConfig` struct | `BPETokenizer.hpp` |
| Add `explicit BPETokenizer(UnicodeConfig)` constructor | `BPETokenizer.hpp` / `.cpp` |
| Implement `bytes_to_unicode` / `unicode_to_bytes` table | `BPETokenizer.cpp` (private helpers) |
| Byte-level pre-tokenisation path in `pre_tokenize()` | `BPETokenizer.cpp` |
| Byte-level decode in `decode()` | `BPETokenizer.cpp` |
| Unit tests: round-trip ASCII and multi-byte UTF-8 strings | `tests/tokenizer_test.cpp` |
| Unit test: byte-level tokenization matches GPT-2 reference on `"Hello, world!"` | `tests/tokenizer_test.cpp` |

### Phase 2 — HuggingFace `tokenizer.json` import

| Task | File(s) |
|------|---------|
| Add lightweight JSON dependency (or verify existing availability) | `external/` or CMake |
| Implement `load_hf_tokenizer()` | `BPETokenizer.cpp` |
| Implement `save_hf_tokenizer()` | `BPETokenizer.cpp` |
| Special token remapping on import | `BPETokenizer.cpp` |
| Integration test: load GPT-2 `tokenizer.json`, encode reference sentences, compare IDs | `tests/tokenizer_test.cpp` |
| Update vocab file format spec to record tokenizer mode | `docs/development/api/nlp/tokenizer.md` |

### Phase 3 — SentencePiece mode (optional, future)

Guarded by `ADAI_ENABLE_SENTENCEPIECE` CMake option. Introduces a minimal SentencePiece
decoder for `▁`-prefixed vocabularies used by T5 and multilingual BERT. ICU or a bundled
Unicode category table provides `\p{L}` / `\p{N}` matching.

---

## 6. Testing Strategy

### 6.1 Unit tests (new, in `tests/tokenizer_test.cpp`)

| Test name | What it validates |
|-----------|-------------------|
| `ByteLevelBPE_RoundTripASCII` | Encode then decode ASCII text returns identical string |
| `ByteLevelBPE_RoundTripUnicode` | Encode then decode multi-language UTF-8 text is lossless |
| `ByteLevelBPE_NoUnknownTokens` | Every possible byte (0x00–0xFF) encodes without producing `<unk>` |
| `ByteLevelBPE_GPT2ReferenceVector` | A fixed sentence matches the reference token IDs from the GPT-2 Python tokenizer |
| `LoadHFTokenizer_VocabSize` | After loading a GPT-2 `tokenizer.json`, `get_vocab_size()` returns 50257 |
| `LoadHFTokenizer_SpecialTokenIds` | After loading, special token IDs match the JSON `added_tokens` section |
| `SaveHFTokenizer_RoundTrip` | `save_hf_tokenizer` + `load_hf_tokenizer` produces an identical vocabulary |
| `AsciiModeUnchanged` | Default constructor still produces current tokenization results (regression guard) |

### 6.2 Reference vectors

A small set of (input, expected token IDs) pairs is extracted from the GPT-2 Python tokenizer
and committed as a JSON fixture under `tests/fixtures/gpt2_tokenizer_reference.json`. These
vectors are the ground truth for `ByteLevelBPE_GPT2ReferenceVector`.

---

## 7. Dependencies

| Dependency | Purpose | Approach |
|------------|---------|----------|
| JSON parser | Parse `tokenizer.json` | Verify `nlohmann/json` availability via `cpp-httplib`; otherwise add `external/nlohmann/json.hpp` (single header, MIT) |
| Unicode normalization (Phase 3) | NFC pre-processing | Optional; gated behind `ADAI_ENABLE_ICU` or a bundled minimal NFC table |

No new mandatory runtime dependencies are introduced in Phases 1 or 2.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Byte-level output alters existing model outputs | Low — gated by `TokenizerMode` | Regression test in `AsciiModeUnchanged` |
| JSON parser adds compile-time overhead | Low | Single-header, header-only inclusion |
| GPT-2 reference IDs diverge due to normalization differences | Medium | Capture reference vectors before implementation; assert on exact match |
| `std::regex` performance degrades on long Unicode-mapped strings | Low — strings remain ASCII-range after byte mapping | Profile in benchmark suite if needed |
| SentencePiece mode complexity underestimated | Medium | Strictly phased; Phase 3 is optional and separately scoped |

---

## 9. Acceptance Criteria

- [ ] All existing tokenizer tests continue to pass without modification.
- [ ] `ByteLevelBPE_RoundTripUnicode` passes for UTF-8 inputs in at least: English, French,
      Japanese (CJK), Arabic, and emoji sequences.
- [ ] `ByteLevelBPE_GPT2ReferenceVector` matches the reference JSON exactly.
- [ ] `LoadHFTokenizer_VocabSize` passes against the public GPT-2 `tokenizer.json`.
- [ ] `save_hf_tokenizer` output is accepted by `transformers.PreTrainedTokenizerFast`.
- [ ] No new compiler warnings at `-Wall -Wextra`.
- [ ] Code formatted with `./scripts/format_code.sh` before merge.
