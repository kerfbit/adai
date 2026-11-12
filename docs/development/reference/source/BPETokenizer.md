# `BPETokenizer` — Source File Reference

- **Files:** [`src/BPETokenizer.hpp`](../../../../src/BPETokenizer.hpp) (338 lines), [`src/BPETokenizer.cpp`](../../../../src/BPETokenizer.cpp) (1,282 lines)
- **Library:** `adai_nlp`
- **Status tags:** `@adai-status: stable`; `BPETokenizer.cpp` `@adai-version: 2.0.2`, `BPETokenizer.hpp` `2.0.1`; `@adai-reviewed: 2026-11-11`
- **Tests:** [`tests/tokenizer_test.cpp`](../../../../tests/tokenizer_test.cpp) → `runTests` (99, ctest `TokenizerTests`); [`tests/tokenizer_error_handling_test.cpp`](../../../../tests/tokenizer_error_handling_test.cpp) → `tokenizerErrorHandlingTests` (36); indirect coverage in `vocabbuilderTests`, `specialtokensTests` and most model and trainer suites
- **Last traced against the code:** 2026-11-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real code (a scratch program linked against the debug
> `adai_nlp`). This document absorbed the old `docs/development/api/nlp/tokenizer.md` in October 2026; what that document got wrong is not
> repeated here.

---

## 1. What this file is

`BPETokenizer` turns text into token ids and back with **byte-pair encoding** (Sennrich et al., 2016) behind a GPT-2-style regex
pre-tokenizer. It is the one tokenizer every model here uses: the encoder-decoder chatbot, the LeJEPA world model, and every trainer and server
that feeds them.

```text
training:  corpus ─► base units (code points or bytes) by frequency ─► pre-tokenize into words
                  ─► repeat: merge the most frequent adjacent pair ─► vocab + ordered merge list
encoding:  text ─► pre-tokenize (regex split; optional lowercasing / whitespace collapsing)
                ─► per word: split into units, replay every merge in order ─► ids (unknown → <unk>)
```

**The vocabulary is part of the model.** Ids index the embedding table and the LM head, so a checkpoint is only meaningful with the exact
vocabulary it was trained with. `EncoderDecoderModel::save()`/`load()` write and read `<checkpoint>.vocab`; servers and every training pass after
the first load the vocabulary from that file. A save-and-load that doesn't reproduce the vocabulary exactly therefore changes what the model
sees between passes (§8).

---

## 2. Where it's used

| Area | Call sites | Uses |
|---|---|---|
| **Building** | `VocabBuilder.cpp` (`vocab_builder`); `IncrementalTrainer.cpp`'s first-pass bootstrap (TD-414); `ChatbotTrainer::build_vocabulary()` | `build_vocab()` + `save_vocab()`; the bootstrap sizes it with `recommend_vocab_size()` unless `VOCAB_BUILD_SIZE` is set, then logs `measure_fertility()`; `vocab_builder` prints `vocab_stats()` and `get_tokens_by_id()` |
| **Loading** | `IncrementalTrainer.cpp` (the model's tokenizer and size probes), `LeJEPATrainingPass.cpp` (the LeJEPA pass's size probe), `ChatbotAPIServer.cpp`, `ChatbotTrainer.cpp`, `LLMEncoder.cpp`, `LeJEPAEncoder.cpp` | `load_vocab()`; `LLMEncoder` and `LeJEPAEncoder` own a private tokenizer |
| **Checkpoints** | `EncoderDecoderModel.cpp` | `save_vocab(path + ".vocab")` / `load_vocab(path + ".vocab")` |
| **Prompts** | `EncoderDecoderModel.cpp`, `ChatbotAPI.cpp`, `ChatbotTrainer.cpp`, `ConversationContext.cpp`, `RAGInference.cpp` | `encode_prompt()` / `encode_conversation()` (turn tokens, TD-295) |
| **Training preprocessing** | `ChatbotTrainer::preprocess_data()` (OpenMP) | `encode()` from many threads on one tokenizer |
| **Generation** | `ChatbotAPI.cpp`, `TextGenerator.cpp`, `RAGInference`, `BatchedInferenceEngine`, the clients | `decode()`, the special-id getters |

`TOKENIZER_MODE` (`unicode` by default, TD-564) chooses the mode of a **new** vocabulary; a loaded one always takes the mode, letter case and
whitespace handling recorded in its file.

---

## 3. Exceptions and settings

| Exception | Base | Thrown by |
|---|---|---|
| `TokenizerInputError` | `std::invalid_argument` | `encode("")`, `decode({})` |
| `TokenizerEncodingError` | `std::runtime_error` | `encode()` / `pre_tokenize()` on invalid UTF-8 |
| `VocabularyFileError` | `std::runtime_error` | `save_vocab()`/`load_vocab()`: empty name, unopenable or unwritable file, malformed lines, bad ids, a newer format, an unknown header line, special ids other than 0–3 |
| `TokenIDError` | `std::out_of_range` | `decode()` on a negative id |

They sit in two standard hierarchies, so `catch (const std::runtime_error&)` misses `TokenizerInputError` and `TokenIDError`.

| Setting | Values (new-tokenizer default first) | Effect |
|---|---|---|
| `TokenizerMode` | `UNICODE` (TD-564), `ASCII` | The atomic unit: a UTF-8 code point, or a byte (a non-ASCII character becomes several byte tokens). Each mode has its own pre-tokenizer regex (the Unicode one counts bytes `0x80`–`0xFF` as letters) |
| `TextNormalization` | `CASED` (TD-560), `LOWERCASE` | `LOWERCASE` lowercases before splitting (in Unicode mode only ASCII bytes) |
| `WhitespaceHandling` | `PRESERVE` (TD-224), `COLLAPSE` | `COLLAPSE` turns each whitespace match into one space; `PRESERVE` keeps text exactly, so line breaks and indentation round-trip |

`set_normalization()`/`set_whitespace()` are meant for before `build_vocab()`; changing them on a loaded vocabulary makes encoding split text
differently from how the vocabulary was learned. Both renew the cache key (§6). `*_name()` give `"cased"`/`"preserve"` etc.

---

## 4. Construction, state and special ids

`BPETokenizer(mode = UNICODE)` registers `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3 and compiles the mode's regex. State: `vocab`
(token → id), `inverse_vocab`, `bpe_merges` (ordered; **order is the algorithm**), `special_tokens`, the four special ids, the three settings,
`has_turn_tokens_`, `first_ordinary_id_`, and `cache_key_`.

**Id layout of a vocabulary built today** (TD-295, TD-565): 0–3 the four specials, 4–6 `<system>`/`<user>`/`<assistant>`, 7–19 placeholder
entries `<reserved_7>`…`<reserved_19>` (all special), ordinary tokens from 20. `first_ordinary_id()` reports 20; a TD-295 vocabulary reports 7,
an older one 4; a vocabulary of 20 or fewer ids keeps the old layout. `encode()` never produces an id below `first_ordinary_id()`, `decode()`
skips them, and `EncoderDecoderModel::mask_unknown_ids()` masks them in generation.

**Turn tokens enter only as ids.** `encode_conversation(turns)` writes each turn's token then its content (`encode(content, false)`), and a
closing `<assistant>`; it requires `has_turn_tokens()`. `encode_prompt(text)` is `<user> text <assistant>` with turn tokens, the plain encoding
without. `encode("<user>")` gives ordinary characters, since the pre-tokenizer splits `<` from letters.

---

## 5. Building: `build_vocab(texts, vocab_size = 10000, frequency_threshold = 1)`

1. A tokenizer that already has a vocabulary is first replaced by a new one keeping only the three settings (TD-223), so a second build
   shares no ids with the first.
2. **Count units** across the texts as `pre_tokenize()` will see them (lowercased for `LOWERCASE`, TD-560).
3. **Reserve ids** 4–19 as above when `vocab_size` is above 20.
4. Under `PRESERVE`, give space, newline and tab ids first: a corpus read line by line never contains a newline (TD-224).
5. **Base units** take ids by frequency, most frequent first, ties by their bytes (TD-225), while they meet `frequency_threshold` and fit
   under `vocab_size`. A unit left out encodes as `<unk>`.
6. **Merges:** `build_bpe_merges(texts, vocab_size − next id)`. It pre-tokenizes the corpus into words and, each round, takes the most
   frequent adjacent pair (`get_most_frequent_pair()`; pairs keyed by the pair itself, ties to the lexicographically smallest, TD-220), appends
   it to `bpe_merges`, gives the merged string the next id, renews the cache key and rewrites every word (`merge_tokens()`, left to right, no
   overlaps). It stops early when no pair is left, which small corpora reach quickly. Each round notes progress for `trainer_service`'s stall
   check (TD-569). Cost: O(merges × corpus tokens), since counts are rebuilt every round.

One corpus gives a byte-identical vocabulary file on any platform (TD-225). Logging is `info` per stage and `debug` per tenth of the merges.

---

## 6. Encoding and decoding

- **`pre_tokenize(text)`:** rejects invalid UTF-8 (`is_valid_utf8()`, RFC 3629: no overlong forms, surrogates, code points past U+10FFFF or lead
  bytes F5–FF, TD-225); lowercases per §3; splits with the mode's regex (contractions, words with an optional leading space, digit runs,
  punctuation runs, whitespace runs); under `COLLAPSE` rewrites each whitespace match to one space (the pattern can match twice between two
  words, so a run isn't always one space).
- **`apply_bpe(word)`:** splits the word into units and replays every merge in order, O(merges × word length). The result is cached per thread,
  keyed by `cache_key_`, a process-unique id renewed by every change to the merges or settings (TD-222), so a reloaded tokenizer, or a new one at
  a reused address, starts with an empty cache. Each thread keeps 4 states' caches, each cleared at 65,536 words.
- **`tokenize()` / `encode(text, add_special_tokens = true)`:** `encode()` validates (empty → `TokenizerInputError`), maps each subword through
  `vocab` (missing, or below `first_ordinary_id()` → `<unk>`), and wraps the result in `<bos>`…`<eos>` when asked. Encoder inputs use
  `add_special_tokens = false`.
- **`decode(ids, skip_special_tokens = true)`:** concatenates each id's token, skipping specials when asked. An empty list throws, a negative id
  throws `TokenIDError`, and ids with no token are left out with one warning per call naming how many and the first (TD-225). Leading spaces are
  part of tokens, so concatenation restores spacing.

`encode()`, `decode()`, `tokenize()`, `pre_tokenize()`, `apply_bpe()`, `encode_conversation()` and `encode_prompt()` are `const` and safe to call
concurrently on one tokenizer; `build_vocab()` and `load_vocab()` must finish before any of them start.

---

## 7. Sizing and diagnostics

- **`recommend_vocab_size(texts, d_model, enc_layers, dec_layers, max_seq_length, mode, normalization)`** (static): a starting point, not a
  fitted value. It combines a dataset base `2000 + 3500·√(words/1000)` (≤ 64,000); a script multiplier (CJK > 30% ×0.55, Arabic/Hebrew > 20%
  ×1.5, Cyrillic > 30% ×1.2, other multibyte > 30% ×1.25 or > 10% ×1.1); a context-length multiplier (≤ 128 ×1.4, ≤ 256 ×1.2, ≥ 1024 ×0.92,
  ≥ 2048 ×0.85); ×1.1 for Unicode mode with more than 5% multibyte; and for `CASED`, 1 + the share of words with an ASCII capital, capped at +30%
  (TD-562). Then it caps at `(3/7)·(12·L_enc + 16·L_dec)·d_model` (embeddings ≤ ~30% of parameters), floors at 2,000 and rounds to 500. No text
  → 2,000.
- **`measure_fertility(texts, sample_limit = 500)`:** tokens per whitespace word over up to `sample_limit` texts (texts that fail to encode are
  skipped). About 1.2–1.8 is healthy; `IncrementalTrainer` warns above 2.5 or below 1.1.
- **`vocab_stats()`** returns mode, settings, size, merge and special counts as text; **`get_tokens_by_id(k)`** the `k` lowest-id tokens (TD-225
  renamed both from `print_vocab_stats()`/`get_top_tokens()`).

---

## 8. The vocabulary file: `save_vocab()` / `load_vocab()`

```text
# BPE Tokenizer Vocabulary v4.0
TOKENIZER_MODE UNICODE|ASCII
NORMALIZATION CASED|LOWERCASE            (v2+, TD-560)
WHITESPACE PRESERVE|COLLAPSE             (v3+, TD-224)
VOCAB_SIZE <n>                           (informational)
RESERVED_SPECIAL_IDS 4|7|20              (v4+, TD-565)
SPECIAL_TOKENS
pad_token_id 0 … eos_token_id 3
VOCAB
<escaped token>\t<id>                    (in id order, TD-225)
BPE_MERGES <count>
<escaped first>\t<escaped second>        (in learned order)
```

Escapes: space `\s`, newline `\n`, tab `\t`, CR `\r`, backslash `\\`. Nothing else is escaped.

**`save_vocab(filename)`** writes `<filename>.tmp`, checks the stream, and renames it into place with `std::filesystem::rename()` (which replaces
an existing file on Windows too). Any failure is a `VocabularyFileError` and removes the temporary file, so a failed save never replaces a good
vocabulary (TD-221). A failed checkpoint save fails the training pass.

**`load_vocab(filename)`** parses into a new tokenizer (`read_vocab_file()`) and takes it only when the whole file loads, so a failure leaves the
tokenizer as it was (TD-223). It starts from the oldest format's meaning (ASCII, `LOWERCASE`, `COLLAPSE`), so a file without those lines reads as
it was written. It refuses a format newer than `kVocabFormatMajor` (4) and an unknown line before the first section (TD-563), an unsupported
`RESERVED_SPECIAL_IDS`, a missing reserved slot, a reserved count above 4 without turn tokens, malformed or negative ids, empty merge halves,
an empty vocabulary, and special ids other than 0–3 or not on their tokens (TD-505). Turn tokens count only when spelled and numbered exactly so.

### Fixed: the reader took header syntax inside the token and merge sections (TD-570, November 11, 2026; was verified)

`read_vocab_file()` applied its header rules to **every** line, whatever section it was in: a line starting with `#` was skipped as a
comment, and one starting with `NORMALIZATION`, `WHITESPACE`, `TOKENIZER_MODE`, `VOCAB_SIZE`, `RESERVED_SPECIAL_IDS` or `BPE_MERGES`, or equal
to `SPECIAL_TOKENS`/`VOCAB`, was read as that header line. `save_vocab()` escapes neither `#` nor letters, so a token or merge starting with one
of these was misread on load:

- **A `#` token disappeared.** Built from text with `## Heading`, `#include` and `#tag`, a vocabulary of 60 tokens loaded back as 58: the `#`
  and `##` tokens and the `#`+`#` merge were skipped. `"## Heading #include #tag"` then encoded to 7 ids instead of 6 and decoded as
  `" Heading #include #tag"`.
- **A keyword token stopped the load.** A cased vocabulary with the token `WHITESPACE` at id 40 made `load_vocab()` throw `Unknown WHITESPACE
  '40'`, so a checkpoint carrying it couldn't be loaded. A token starting `TOKENIZER_MODE` would have switched the mode mid-file, one starting
  `BPE_MERGES` ended the token section early.

Markdown headings, code, hashtags and upper-case identifiers are ordinary training text, and every training pass and server loads the
vocabulary from its file, so `#` tokens never reached a model. **Now** section markers carry no tab and every entry does, so a line with a tab
is always an entry, and the header rules apply only before the first section. Vocabulary files were always written correctly and now load
whole. Tests: `TokensStartingWithAHashSurviveSaveAndLoad`, `TokensStartingWithAHeaderKeywordSurviveSaveAndLoad`,
`HeaderCommentsAndUnknownHeaderLinesAreStillHandled`.

---

## 9. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| The reader applied `#`-comment and header-keyword rules inside the token and merge sections (§8); the constructor's comment called ASCII the default | `#` tokens lost on load; a keyword token failed the load | **Fixed November 11, 2026** (was verified) | [TD-570](../../guides/TECHNICAL_DEBT.md#td-570-bpetokenizerload_vocab-takes-header-syntax-inside-the-token-and-merge-sections) |
| `"|||"` pair keys stopping merge learning | — | **Fixed October 17, 2026** | TD-220 (audit old vocabularies with `scripts/audit_vocab_merges.py`) |
| Silent `save_vocab()` failures | — | **Fixed October 17, 2026** | TD-221 |
| Never-invalidated `apply_bpe()` cache | — | **Fixed October 27, 2026** | TD-222 |
| A failed load gutting the tokenizer; a second build sharing ids | — | **Fixed November 9, 2026** | TD-223 |
| Lossy normalization (case, whitespace) | — | **Fixed** for new vocabularies (TD-560 November 4, TD-224 November 8) | TD-224, TD-560–TD-563 |
| Logging, `const`, misnamed helpers, unordered ids, lenient UTF-8 | — | **Fixed November 10, 2026** | TD-225 |
| Non-standard special ids | — | **Fixed November 9, 2026** | TD-505 |

`stable` holds again with TD-570 fixed.

## 10. History

- **October 2026:** merged in the old `tokenizer.md`; TD-220, TD-221 (October 17), TD-222 (October 27).
- **November 4, 2026:** letter case, format v2 (TD-560–TD-563); Unicode by default (TD-564).
- **November 8, 2026:** whitespace, format v3 (TD-224); turn tokens (TD-295).
- **November 9, 2026:** reserved special ids, format v4 (TD-565); all-or-nothing load and rebuild (TD-223); standard special ids (TD-505).
- **November 10, 2026:** logging and API hygiene (TD-225).
- **November 11, 2026:** progress per merge (TD-569). This document was re-traced against the code, replacing its October sections and folding
  in its addenda; the trace found the §8 reader bug, fixed the same day (TD-570).
