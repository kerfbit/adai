# `SpecialTokens` — Source File Reference

- **File:** [`src/SpecialTokens.hpp`](../../../../src/SpecialTokens.hpp) (header-only, 365 lines)
- **Namespace:** `adai` (`SpecialTokenIDs`, `SpecialTokenStrings`, `SpecialTokenConfig`, free helpers)
- **Used by:** `BPETokenizer.hpp` (default IDs), `TextGenerator` (default `GenerationConfig` IDs; `is_stop_token()` via `SpecialTokenConfig`), `EncoderDecoderModel.cpp` (constructor defaults, before `sync_special_tokens()`), `PipelineInferenceEngine.hpp`, `BatchProcessor.hpp` and `ChatbotAPI.cpp` (fixed PAD/BOS/EOS)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.1`, `@adai-reviewed: 2026-10-18` (doc comment only: the training-target example now drops the leading `BOS`, TD-361)
- **Tests:** [`tests/specialtokens_test.cpp`](../../../../tests/specialtokens_test.cpp) (18 tests, `specialtokensTests`; all pass)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real `BPETokenizer` and `TextGenerator`.

---

## 1. What this file is

The canonical special-token constants and a few helpers.

| Token | ID (`SpecialTokenIDs`) | String (`SpecialTokenStrings`) | Purpose |
|---|---|---|---|
| PAD | 0 | `<pad>` | Padding |
| UNK | 1 | `<unk>` | Out-of-vocabulary |
| BOS | 2 | `<bos>` | Sequence start |
| EOS | 3 | `<eos>` | Sequence end; stops generation |

| Member | Behaviour | Production use |
|---|---|---|
| `SpecialTokenConfig` (default 0/1/2/3, or custom four IDs; getters) | Holds the four IDs | `TextGenerator::is_stop_token()` builds one from its config |
| `SpecialTokenConfig::validate()` | Throws on a negative or duplicate ID | **None** |
| `is_stop_token(id, config)` | EOS or PAD | `TextGenerator` (every generation loop) |
| `is_special_token()`, `is_special_token_string()`, `get_special_token_string()`, `get_special_token_id()`, `create_special_token_set()`, `create_special_token_map()`, `create_inverse_special_token_map()` | Lookups between IDs and strings | **None** (tests only) |
| Doc-comment examples | Encoding/decoding conventions, generation termination, syncing a model's generation config from its tokenizer, the vocabulary file format | — |

The usage examples state the convention the codebase follows: encoder input without special tokens, decoder targets as `[BOS] … [EOS]`,
decoding with special tokens skipped. Literal `"<eos>"` text in user input is tokenized as characters, not as the EOS ID (BPETokenizer.md).

### Where the IDs actually come from

`BPETokenizer` starts with the constants, but `load_vocab()` reads a vocabulary file's own `SPECIAL_TOKENS` section, so a vocabulary can carry any IDs.
`EncoderDecoderModel::sync_special_tokens()` copies the tokenizer's IDs into its generator, and `IntegratedInferenceEngine` asks the tokenizer. Other
components use the constants directly: `PipelineInferenceEngine` (BOS/EOS/PAD in its constructor), `ChatbotAPI`'s batch padding, `BatchProcessor`'s
defaults, and `TextGenerator`'s default config before a sync.

---

## 2. Gaps

### The "standard" isn't enforced (verified)

The header says the IDs are standardized and "must not be changed", but nothing enforces it:

- **Non-standard IDs load.** A vocabulary with PAD=5, UNK=4, BOS=1, EOS=0 loaded, and `encode("ab", true)` gave `1 2 3 0`. The components that use the
  constants directly (above) would then pad with token 0 (EOS here), and `PipelineInferenceEngine` would start with BOS=2 (an ordinary token here) and
  stop on 3, while the tokenizer and synced generator use 1 and 0.
- **Duplicate IDs load.** A vocabulary with BOS = EOS = 3 loaded, and every encoded target became `3 … 3`, so a sequence starts with the stop token.
  `SpecialTokenConfig::validate()` would reject this, but nothing calls it.

`vocab_builder` always writes 0–3, so no shipped vocabulary is affected. The risk is a hand-edited or imported vocabulary. Either reject non-standard IDs
on load (the header's own rule) or make every component take the IDs from the tokenizer.

### Mostly unused API (by inspection)

Of the helpers, only `is_stop_token()` (and `SpecialTokenConfig` as its argument) is used outside tests. `BPETokenizer.cpp` writes the strings as literals
(`"<pad>"` …) rather than using `SpecialTokenStrings`, so the "single source of truth" holds for IDs only.

### Smaller items (by inspection)

- The vocabulary-format example omits the `TOKENIZER_MODE` line and `BPE_MERGES` section that `save_vocab()` writes.
- `is_stop_token()` treats PAD as a stop token; that's intended, but means a model that samples PAD ends generation silently.

---

## 3. Tests

`specialtokens_test.cpp` (18): constants, default and custom configs, `validate()` (accepts defaults, rejects negative and duplicate IDs), every helper
and map, and custom-config behaviour.

**Not covered:** that loaded vocabularies use the standard IDs; the components that hard-code them.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Non-standard or duplicate special-token IDs load; `validate()` is never called; several components use the constants instead of the tokenizer's IDs (§2) | A custom vocabulary silently mis-pads, mis-starts and mis-stops generation | **Verified** (loading); by inspection (consumers) | [TD-505](../../guides/TECHNICAL_DEBT.md#td-505-special-token-ids-arent-enforced-and-most-specialtokens-helpers-are-unused) |
| Most helpers unused; strings duplicated as literals; format example incomplete (§2) | Misleading "single source of truth" | By inspection | [TD-505](../../guides/TECHNICAL_DEBT.md#td-505-special-token-ids-arent-enforced-and-most-specialtokens-helpers-are-unused) |

---

## 5. Merge notes

This file replaces `docs/development/SPECIAL_TOKEN_CONSOLIDATION.md` (191 lines). Where they disagree, this file wins:
- **"Single source of truth" / "ensures all components use the same token IDs".** True for the constants, but vocabularies may override them and
  several components ignore the override (§2); most helpers are unused.
- **Usage locations.** `ParallelDataLoader.hpp`, `Dataset.hpp` and `EfficientBatching.hpp` no longer exist; `SpeculativeDecoding.hpp` and
  `IntegratedInferenceEngine.hpp` don't use this header.
- **"Custom Configurations".** `SpecialTokenConfig` lets the same four tokens take different IDs; it does not support additional special tokens (a
  claim [reasoning_process_plan.md](../../../proposals/reasoning_process_plan.md) relied on).

Kept: the token table, the helper list and the test command (`ctest -R SpecialTokensTests`). Dropped: the migration path and benefits list.
