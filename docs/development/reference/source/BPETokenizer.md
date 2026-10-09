# `BPETokenizer` — Source File Reference

- **Files:** [`src/BPETokenizer.hpp`](../../../../src/BPETokenizer.hpp), [`src/BPETokenizer.cpp`](../../../../src/BPETokenizer.cpp)
- **Library:** `adai_nlp`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/tokenizer_test.cpp`](../../../../tests/tokenizer_test.cpp) → `runTests` (53 tests, ctest `TokenizerTests`); [`tests/tokenizer_error_handling_test.cpp`](../../../../tests/tokenizer_error_handling_test.cpp) → `tokenizerErrorHandlingTests` (36, TD-002); indirect coverage in `vocabbuilder_test.cpp` (52) and most model/trainer suites
- **Last traced against the code:** 2026-10-08

> Code-traced reference: every "where it's used" claim points at a real call site, and every
> behaviour flagged as surprising was confirmed by running the real code (a scratch program
> linked against `BPETokenizer.cpp`). It supersedes the older `docs/development/api/nlp/tokenizer.md`
> ("BPETokenizer Class - Context Document", now deleted), whose accurate content has been merged in
> here. See [§12](#12-merge-notes-what-changed-from-the-old-document) for what was corrected or
> dropped.

---

## Contents

1. [What this file is](#1-what-this-file-is)
2. [Where it's used](#2-where-its-used)
3. [Exceptions and `TokenizerMode`](#3-exceptions-and-tokenizermode)
4. [Construction and state](#4-construction-and-state)
5. [Training: `build_vocab()` and the merge learner](#5-training-build_vocab-and-the-merge-learner)
6. [The encode path: `pre_tokenize()` → `apply_bpe()` → `tokenize()` → `encode()`](#6-the-encode-path)
7. [`decode()`](#7-decode)
8. [Persistence: `save_vocab()` / `load_vocab()` and the file format](#8-persistence-save_vocab--load_vocab-and-the-file-format)
9. [Sizing and diagnostics](#9-sizing-and-diagnostics)
10. [Thread safety](#10-thread-safety)
11. [Known gaps and gotchas (summary)](#11-known-gaps-and-gotchas-summary)
12. [Merge notes](#12-merge-notes-what-changed-from-the-old-document)

---

## 1. What this file is

`BPETokenizer` turns text into integer token IDs and back, using **byte-pair encoding** (Sennrich
et al., 2016; the GPT-2-style regex pre-tokenizer). It's the single tokenizer every model in the
project uses: the encoder-decoder chatbot, the LeJEPA world model, and every trainer and server
that feeds them.

**Training** (once per vocabulary):

```text
corpus ─► count atomic units (bytes or code points) ─► base vocab (IDs 4…)
       ─► pre-tokenize into words ─► repeat N times: merge the most frequent adjacent pair
       ─► vocab = 4 specials + base units + N merged strings;  merges = ordered rule list
```

**Encoding** (every request):

```text
text ─► lowercase + regex split into "words" ─► per word: split to units, replay every merge rule in order
     ─► look up each subword's ID (unknown → <unk>) ─► [<bos>] ids… [<eos>]
```

### Why it matters

- **The vocabulary is part of the model.** Token IDs index the embedding table and the LM head, so
  a checkpoint is only meaningful together with the exact vocab file it was trained with. That's
  why `EncoderDecoderModel::save()`/`load()` write and read `<checkpoint>.vocab` alongside the
  weights, and why `IncrementalTrainer` probes a vocab file's size before constructing a model.
- **It defines what the model can see.** Lowercasing and whitespace collapsing (§6) happen here,
  so the model never sees capital letters, newlines or tabs, and can never generate them.
- **It's on every hot path.** Training preprocessing encodes the whole dataset in parallel
  (OpenMP); every chat request is encoded and every response decoded.

---

## 2. Where it's used

| Area | Call sites | Uses |
|---|---|---|
| **Vocabulary creation** | `src/VocabBuilder.cpp` (`vocab_builder` tool, ≈ lines 178–183) | `build_vocab()`, `save_vocab()`, `print_vocab_stats()`, `get_top_tokens()` |
| | `src/IncrementalTrainer.cpp` (≈ lines 215–262, vocab bootstrap) | `recommend_vocab_size()` (when `VOCAB_BUILD_SIZE` is unset), `build_vocab()`, `save_vocab()`, then `measure_fertility()` with warnings above 2.5 / below 1.1 |
| | `src/ChatbotTrainer.cpp` — `build_vocabulary()` (≈ line 78) | `build_vocab(texts, size, 1)` + `save_vocab()` |
| | `src/LLMEncoder.cpp` (≈ 218–223), `src/LeJEPAEncoder.cpp` (≈ 294–299) | Each owns a **private** tokenizer: `load_vocab()` or `build_vocab()` on it |
| **Loading** | `src/ChatbotAPIServer.cpp` (≈ line 315), `src/ChatbotGUI.cpp`, `src/ChatbotTrainer.cpp` (≈ 64), `src/IncrementalTrainer.cpp` — `build_model()` (≈ 122) and vocab-size probes (≈ 166), `src/IncrementalTrainingTool.cpp` (≈ 228) | `load_vocab()` (restores the saved mode) |
| **Checkpoints** | `src/EncoderDecoderModel.cpp` — `save()` (≈ 771) / `load()` (≈ 843) | `save_vocab(path + ".vocab")` / `load_vocab(path + ".vocab")` |
| **Training preprocessing** | `src/ChatbotTrainer.cpp` (≈ lines 533, 581: `#pragma omp parallel for`), `src/RLHFTrainer.cpp` | `encode()` concurrently from many threads on one shared instance |
| **Inference** | `src/ChatbotAPI.cpp` (`generate_response()`, `generate_batch_responses()`), `src/TextGenerator.cpp` (`generate_text()`), `src/RAGInference.hpp`, `src/PipelineInferenceEngine.hpp`, `src/IntegratedInferenceEngine.hpp`, `src/BatchedInferenceEngine.hpp`, `src/ChatbotCLI.hpp`, `src/ChatbotGUI.cpp` | `encode(text, false)` for encoder input; `decode()` of generated IDs; special-token ID getters |

Configuration: `TOKENIZER_MODE = ascii | unicode` (`src/Config.{hpp,cpp}`) → `ServiceConfig::tokenizer_mode`
→ the `BPETokenizer(mode)` constructor in `IncrementalTrainer`; `vocab_builder` has its own
Unicode flag. Loading a vocab file always overrides the constructor's mode with the file's
`TOKENIZER_MODE` line. Per an inline comment in `IncrementalTrainingTool.cpp`, `TOKENIZER_MODE` is
not yet threaded through the LeJEPA path (TD-177).

Several callers carry the same warning comment: `encode("")` **throws** rather than returning an
empty vector, so they check for empty strings first (see `ChatbotTrainer.cpp` ≈ 533 and
`RLHFTrainer.cpp` ≈ 179). `BatchedInferenceEngine` was bitten by this once (see its reference).

`src/DataFetcher.cpp` doesn't use the tokenizer. Its only mention is a TD-006 TODO saying FIM
special tokens (`<|first|>`/`<|middle|>`/`<|last|>`) must be "defined first (see BPETokenizer.cpp)".
No such definition exists yet: the tokenizer knows only the four specials. Tracked in [TD-225](../../guides/TECHNICAL_DEBT.md#td-225-bpetokenizer-logging-and-api-hygiene).

---

## 3. Exceptions and `TokenizerMode`

| Exception | Base | Thrown by |
|---|---|---|
| `TokenizerInputError` | `std::invalid_argument` | `encode("")`, `decode({})` |
| `TokenizerEncodingError` | `std::runtime_error` | `encode()` / `pre_tokenize()` on invalid UTF-8 |
| `VocabularyFileError` | `std::runtime_error` | `load_vocab()`: empty filename, unopenable file, malformed lines, bad/negative IDs, missing specials |
| `TokenIDError` | `std::out_of_range` | `decode()` on a negative ID |

Each prefixes its message ("Tokenizer Input Error: …"). The four live in two different standard
hierarchies, so a `catch (const std::runtime_error&)` does **not** catch `TokenizerInputError` or
`TokenIDError`. That's the mistake `ChatbotTrainer`'s comment describes.

```cpp
enum class TokenizerMode { ASCII, UNICODE };
```

- **`ASCII`** (default) — byte-level. Each byte is one atomic unit, so a multi-byte UTF-8
  character becomes several single-byte tokens, which BPE may later merge back together.
  Lowercasing applies `std::tolower` to every byte.
- **`UNICODE`** — code-point-level. Each UTF-8 code point is one atomic unit. Only ASCII bytes are
  lowercased (lowercasing continuation bytes would corrupt sequences). The pre-tokenizer regex
  treats bytes `0x80`–`0xFF` as letters, so non-ASCII words stay together.

Both modes reject invalid UTF-8 at `encode()`. The ASCII-mode comment ("non-ASCII input passes
through") means *valid* non-ASCII is accepted and split into bytes.

`get_mode()` / `is_unicode_mode()` report the active mode; `IncrementalTrainer` logs it.

---

## 4. Construction and state

```cpp
explicit BPETokenizer(TokenizerMode mode = TokenizerMode::ASCII);
```

Registers the four special tokens from `SpecialTokens.hpp` (`<pad>`=0, `<unk>`=1, `<bos>`=2,
`<eos>`=3) in `vocab`, `inverse_vocab` and `special_tokens`, and compiles the mode's regex once.

| Member | Holds |
|---|---|
| `vocab` | `unordered_map<string, int>`, token → ID |
| `inverse_vocab` | `unordered_map<int, string>`, ID → token |
| `bpe_merges` | ordered `vector<pair<string,string>>`; **order is the algorithm** (earlier merges take precedence) |
| `special_tokens` | set of the 4 special strings, for `decode(skip_special_tokens=true)` |
| `pad/unk/bos/eos_token_id` | defaults from `SpecialTokenIDs`; may be overridden by a vocab file (§8) |
| `token_pattern` | compiled pre-tokenizer regex |

Getters: `get_vocab_size()`, `get_bos/eos/pad/unk_token_id()`. A fresh tokenizer has
`get_vocab_size() == 4`; it can encode (every word becomes `<unk>`) but is useless until trained
or loaded.

**Special tokens can't be injected from text.** `encode("<eos>")` tokenizes the characters
`<`, `eos`, `>` as ordinary text; only `add_special_tokens` produces the real IDs.

---

## 5. Training: `build_vocab()` and the merge learner

```cpp
void build_vocab(const std::vector<std::string>& texts, int vocab_size = 10000,
                 int frequency_threshold = 1);
```

1. **Count atomic units** (bytes or code points) across `texts`.
2. **Base vocabulary:** every unit seen at least `frequency_threshold` times gets the next ID from
   4 upward, until `vocab_size` is reached. Units below the threshold, or beyond the cap, are not
   in the vocab and encode as `<unk>`.
3. **Merges:** `build_bpe_merges(texts, vocab_size − next_id)`, so the final vocab is roughly
   `vocab_size`, unless the corpus runs out of pairs first, which small corpora do quickly.

Progress is printed to `std::cout` throughout (see §11).

**Calling it twice corrupts the vocabulary.** The second call restarts base IDs at 4 and appends
to the existing merges. Running the real code, two `build_vocab()` calls produced 22 tokens with 3
IDs shared by two different tokens. `tokenizer_test`'s `RepeatedBuildVocab` says "Second build
should replace vocabulary" but only checks that the size changed, so it passes. Always use a
fresh instance per build (every production caller does).

**Base IDs depend on hash order.** Base units get IDs in `unordered_map` iteration order, and
`get_most_frequent_pair()` breaks frequency ties by that order too. The same corpus gives the same
vocab with the same standard library, but not necessarily across library implementations or
versions. The saved file is what makes a vocab reproducible, so always ship the file, never
"rebuild from the same data".

### `build_bpe_merges(texts, num_merges)`

Pre-tokenizes the whole corpus into words, splits each into units, then for each of `num_merges`
rounds:

1. `get_most_frequent_pair()` counts every adjacent pair in every word;
2. if none is left, stops early;
3. appends the pair to `bpe_merges`, gives `first+second` the next ID (`vocab.size()`);
4. `merge_tokens()` rewrites every word with that pair merged.

Cost is O(num_merges × corpus tokens) since counts are recomputed from scratch each round. That's
fine at this project's vocab sizes but slow for very large corpora.

### `get_most_frequent_pair(word_tokens)` (static)

Counts pairs in an `unordered_map` keyed by `first + "|||" + second`, returns the max, and splits
the key back at the **first** `"|||"`. Hardened against empty word entries in TD-058.

> **Pipe characters break it.** If `first` itself ends in `|` (e.g. the pair `("|", "|")`, key
> `"|||||"`), splitting at the first `"|||"` returns `first = ""`, which the caller treats as "no
> more pairs". Training then **stops entirely**. Verified: a corpus of `"|| || || ab ab"` lines
> asked for 52 merges and learned **zero**. Markdown tables, `||` in code, and shell pipelines are
> all realistic triggers. Any pair whose split lands inside the separator is affected, not just
> pipes-only ones.

### `merge_tokens(word_tokens, first, second)` (static)

Left-to-right, non-overlapping replacement of `first, second` with `first+second` in every word.

---

## 6. The encode path

### `pre_tokenize(text)`

1. Rejects invalid UTF-8 (`TokenizerEncodingError`); empty input returns `{}`.
2. **Lowercases** (all bytes in ASCII mode; ASCII bytes only in Unicode mode).
3. Splits with the mode's regex:

   ```text
   ASCII:   's|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^ \t\r\nA-Za-z0-9]+|\s+(?!\S)|\s+
   UNICODE: same, with \x80-\xFF added to the letter class and excluded from the punctuation class
   ```

   That gives contractions, words with an optional leading space, digit runs, punctuation runs,
   and whitespace runs.
4. **Collapses every whitespace run to a single space** (`regex_replace("\s+", " ")`).

Steps 2 and 4 make tokenization **lossy**. Verified: `"Hello World\nSecond\tline  here"`
round-trips as `"hello world second line  here"`. Case, newlines and tabs are gone. (The double
space survives as `" "` plus `" here"`.) So no model trained here can learn to produce line
breaks or capitals, and the vocab file's `\n`/`\t` escaping (§8) never fires for real tokens.

For inputs producing ≥ 1,000 matches it also prints progress lines to `std::cout`. That includes
long chat inputs on the server, and interleaved output from parallel preprocessing threads.

### `apply_bpe(word)`

Splits the word into units and replays **every** merge rule in order, so the cost is
O(merges × word length). Results are memoized in a **`thread_local`** cache keyed by
`(tokenizer address, word)`. Natural text is heavily repetitive, so this removed a big
preprocessing bottleneck. The design comment says it's safe because merges are immutable during
parallel encoding. The cache is never invalidated, though:

> **Stale cache (verified).** (1) After `load_vocab()` on an instance that has already encoded
> text, words encode with the **old** merges: a reload to a tiny vocab still returned the old
> 1-token result instead of the correct 4 tokens. (2) If a tokenizer is destroyed and another is
> constructed **at the same address** on the same thread, the new one gets the old one's cached
> results. Entries for destroyed tokenizers are never freed, either. `measure_fertility()` creates
> a temporary tokenizer per call, which adds one more cache entry each time (§9). Production mostly
> avoids this because each `trainer_service` pass is a fresh process and servers load once, but
> nothing in the class prevents it.

### `tokenize(text)` / `encode(text, add_special_tokens = true)`

`tokenize()` is `pre_tokenize()` then `apply_bpe()` per word, concatenated. `encode()` validates
(empty → `TokenizerInputError`, bad UTF-8 → `TokenizerEncodingError`), then maps each subword
through `vocab`, giving `unk_token_id` when it's missing. With `add_special_tokens` it wraps the
result in `bos … eos`.

Encoder inputs are encoded **without** specials (`encode(text, false)`) throughout `ChatbotAPI`.
`TextGenerator` seeds decoding with BOS itself.

---

## 7. `decode()`

```cpp
std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true);
```

Concatenates `inverse_vocab[id]` for each ID. With `skip_special_tokens`, the four special
strings are dropped. Leading spaces are part of tokens (`" world"`), so plain concatenation
restores word spacing.

- Empty `ids` → `TokenizerInputError`; negative ID → `TokenIDError`.
- An ID that isn't in the vocab (≥ vocab size) is **skipped with a `std::cerr` warning**, not an
  exception, so a model emitting out-of-range IDs produces silently shortened text.

`encode()` and `decode()` aren't `const`, though neither mutates anything. That forces the
copy-the-whole-tokenizer workaround in `measure_fertility()`, and stops callers holding a
`const BPETokenizer&` from encoding.

---

## 8. Persistence: `save_vocab()` / `load_vocab()` and the file format

```text
# BPE Tokenizer Vocabulary v1.0
TOKENIZER_MODE ASCII|UNICODE
VOCAB_SIZE <n>                       (informational; ignored on load)
SPECIAL_TOKENS
pad_token_id 0
unk_token_id 1
bos_token_id 2
eos_token_id 3
VOCAB
<escaped token>\t<id>                (unordered_map order, NOT sorted)
...
BPE_MERGES <count>
<escaped first>\t<escaped second>    (in learned order; order matters)
...
```

Escapes: space → `\s`, newline → `\n`, tab → `\t`, CR → `\r`, backslash → `\\`.

### `save_vocab(filename) const`

Writes the format above and logs a summary to `std::cout`.

> **Fails silently.** If the file can't be opened it prints to `std::cerr` and **returns
> normally**. Write errors aren't checked either. Verified with an unwritable path: no exception.
> This matters because `EncoderDecoderModel::save()` calls it for every checkpoint: a full disk or
> bad path yields a checkpoint with weights but **no `.vocab`**, which then fails at `load()`.
> `IncrementalTrainer`'s vocab bootstrap and `vocab_builder` have the same exposure.

### `load_vocab(filename)`

1. Throws `VocabularyFileError` for an empty filename or unopenable file.
2. **Clears** `vocab`, `inverse_vocab`, `bpe_merges` and `special_tokens`, then parses line by
   line. `TOKENIZER_MODE` switches the mode and recompiles the regex. Special-token IDs come from
   `SPECIAL_TOKENS`. Malformed lines, non-numeric/negative/out-of-range IDs, and empty merge
   halves throw `VocabularyFileError`.
3. Requires all four special strings to be present, then re-forces `inverse_vocab[special_id]` to
   the canonical strings.

Gotchas:

- **Not exception-safe (verified).** Because it clears first, a malformed file leaves the
  tokenizer **gutted**: a trained 12-token tokenizer had 1 token after a failed load. A caller that
  catches the exception and keeps going ends up with a broken tokenizer rather than the previous
  one. `chatbot_api_server --pipeline-inference` does exactly that (`ChatbotAPIServer.cpp` ≈ 540):
  it catches a failed `enable_pipeline_inference()`, which reloads the encoder's private
  tokenizer, logs a warning, and keeps serving.
- **Doesn't invalidate the `apply_bpe()` cache** (§6).
- **Special IDs from the file may disagree with `SpecialTokenIDs`.** The loader honours whatever
  IDs the file declares, but other code (`BatchProcessor`, `TextGenerator`, `Dataset`) uses
  `adai::SpecialTokenIDs` constants directly. Every vocab produced here uses 0–3, so they agree in
  practice. A hand-edited or foreign file wouldn't, and nothing checks.
- A file without a `TOKENIZER_MODE` line (pre-Unicode vocabularies) keeps the constructor's mode.

---

## 9. Sizing and diagnostics

### `recommend_vocab_size(texts, d_model, enc_layers, dec_layers, max_seq_length, mode)` (static)

A heuristic used by `IncrementalTrainer` when no explicit size is configured. It combines:

1. **Dataset size:** `2000 + 3500·√(words/1000)`, capped at 64,000 (≈3k vocab at 1k words, ≈14k
   at 100k, ≈44k at 1M);
2. **Script mix** (by Unicode block counts): CJK > 30% → ×0.55; Arabic/Hebrew > 20% → ×1.5;
   Cyrillic > 30% → ×1.2; other multibyte > 30% → ×1.25, > 10% → ×1.1;
3. **Embedding budget cap:** `V ≤ (3/7)·(12·L_enc + 16·L_dec)·d_model`, which keeps the input
   embedding at about 30% of parameters or less;
4. **Context length:** ≤128 → ×1.4; ≤256 → ×1.2; ≥1024 → ×0.92; ≥2048 → ×0.85;
5. **Unicode mode** with > 5% multibyte → ×1.1;

then clamps to ≥ 2000 and rounds to the nearest 500. Empty input returns 2000. The constants are
engineering judgement, not fitted, so treat the result as a starting point and check it with
`measure_fertility()`.

### `measure_fertility(texts, sample_limit = 500) const`

Average BPE tokens per whitespace word over up to `sample_limit` non-empty texts (texts that fail
to encode are skipped). Roughly 1.2–1.8 is healthy, above 2.5 means the vocab is too small, and
below 1.1 suggests it's oversized. `IncrementalTrainer` logs it and warns outside those bands.
Because `encode()` isn't `const`, it **copies the entire tokenizer** into a temporary on every
call. That's expensive for large vocabs, and it adds an `apply_bpe()` cache entry each time.

### `print_vocab_stats() const` / `get_top_tokens(k) const`

Used by `vocab_builder`. `print_vocab_stats()` prints mode, size, merge count and special count
to `std::cout`. **`get_top_tokens(k)` is misnamed:** it returns the `k` tokens with the **lowest
IDs** (always the four specials, then base units in hash order), not the most frequent. The
tokenizer keeps no frequencies at all. Verified: on a corpus that's almost all `z`, it returned
`<pad> <unk> <bos> <eos> a`.

---

## 10. Thread safety

| Operation | Concurrent with itself? | Notes |
|---|---|---|
| `encode()`, `tokenize()`, `decode()`, getters | Yes | Read-only on shared maps and regex; `apply_bpe()`'s cache is per-thread. This is what `ChatbotTrainer`'s OpenMP preprocessing relies on |
| `build_vocab()`, `load_vocab()` | No, and not with any reader | Mutate every table; must finish before any concurrent encoding starts |
| `pre_tokenize()` progress output | — | Threads interleave `\r` progress lines on stdout |

---

## 11. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Verified | Tracked as |
|---|---|---|---|
| `"|||"` pair separator: pairs whose split lands in the separator (e.g. `("|","|")`) stop merge learning (§5) | Pipe-heavy corpora get few or no merges | Yes: 0 of 52 merges learned | [TD-220](../../guides/TECHNICAL_DEBT.md#td-220-bpe-merge-learning-is-corrupted-by-the--pair-key-separator) |
| `apply_bpe()` thread-local cache never invalidated; keyed by address; never freed (§6) | Stale tokenization after `load_vocab()` or address reuse; slow memory growth | Yes, both cases | [TD-222](../../guides/TECHNICAL_DEBT.md#td-222-apply_bpes-thread-local-cache-is-never-invalidated) |
| `save_vocab()` fails silently (§8) | Checkpoints can be written without their `.vocab` | Yes | [TD-221](../../guides/TECHNICAL_DEBT.md#td-221-bpetokenizersave_vocab-fails-silently-so-checkpoints-can-lack-their-vocab) |
| `load_vocab()` not exception-safe (§8) | Failed load leaves a gutted tokenizer | Yes: 12 → 1 tokens | [TD-223](../../guides/TECHNICAL_DEBT.md#td-223-tokenizer-mutators-can-leave-it-gutted-or-with-duplicate-ids) |
| Second `build_vocab()` corrupts IDs; `RepeatedBuildVocab` test doesn't catch it (§5) | Duplicate IDs if an instance is reused | Yes: 3 shared IDs | [TD-223](../../guides/TECHNICAL_DEBT.md#td-223-tokenizer-mutators-can-leave-it-gutted-or-with-duplicate-ids) |
| Lowercasing and whitespace collapsing are lossy (§6) | Models can't see or produce capitals, newlines, tabs | Yes | [TD-224](../../guides/TECHNICAL_DEBT.md#td-224-tokenizer-normalization-is-lossy-no-capitals-newlines-or-tabs) |
| `std::cout`/`std::cerr` throughout library code (§5–9) | Breaks the logging convention; progress spam on long inputs and in parallel preprocessing | By inspection | [TD-225](../../guides/TECHNICAL_DEBT.md#td-225-bpetokenizer-logging-and-api-hygiene) |
| `decode()` skips unknown IDs with only a `cerr` warning (§7) | Silent truncation | By inspection | [TD-225](../../guides/TECHNICAL_DEBT.md#td-225-bpetokenizer-logging-and-api-hygiene) |
| `encode()`/`decode()` not `const` (§7, §9) | Full tokenizer copy in `measure_fertility()`; can't encode via `const&` | By inspection | [TD-225](../../guides/TECHNICAL_DEBT.md#td-225-bpetokenizer-logging-and-api-hygiene) |
| `get_top_tokens()` returns lowest IDs, not most frequent (§9) | Misleading `vocab_builder` output | Yes | [TD-225](../../guides/TECHNICAL_DEBT.md#td-225-bpetokenizer-logging-and-api-hygiene) |
| Base IDs and tie-breaks depend on `unordered_map` order (§5) | Rebuilds aren't portable across standard libraries | By inspection | [TD-225](../../guides/TECHNICAL_DEBT.md#td-225-bpetokenizer-logging-and-api-hygiene) |
| Vocab-file special IDs aren't checked against `SpecialTokenIDs` (§8) | Foreign/hand-edited vocabs could disagree with code using the constants | By inspection | [TD-225](../../guides/TECHNICAL_DEBT.md#td-225-bpetokenizer-logging-and-api-hygiene) |
| `is_valid_utf8()` accepts overlong forms, surrogates and lead bytes up to `0xF7` | Lenient validation | By inspection | [TD-225](../../guides/TECHNICAL_DEBT.md#td-225-bpetokenizer-logging-and-api-hygiene) |

---

## 12. Merge notes: what changed from the old document

This file absorbed `docs/development/api/nlp/tokenizer.md`. Where they disagreed, this file's
code-traced details won.

**Corrected:**
- *No `TokenizerMode`.* The old doc predates Unicode mode entirely. Added §3 and per-mode
  behaviour throughout.
- *Pre-tokenizer regex* was an older pattern (`?[^\s\w]+|\s+`); the real patterns are in §6.
- *File format* lacked the `TOKENIZER_MODE` line and showed the `VOCAB` section sorted by ID; it's
  written in hash-map order (§8).
- *"Thread-safe after vocabulary is built"*: true for concurrent reads, but missed the
  never-invalidated `apply_bpe()` cache and the reload hazards (§6, §10).
- *Example 1's output* claimed a 1,000-token vocab from three short sentences; small corpora run
  out of pairs long before that. Example 2's IDs and Example 3's "pre/process/ing" split were
  illustrative, not real output, so they were dropped.
- *Example 4 / "Get most common tokens"*: `get_top_tokens()` returns lowest IDs (§9).
- *Time-complexity table* listed `encode` as O(T). It's dominated by `apply_bpe()`'s
  O(merges × word length) per *new* word (cached afterwards). The memory estimates were dropped as
  unverified.
- *"Future enhancements: byte-level BPE, caching, token statistics"*: ASCII mode is already
  byte-level, `apply_bpe()` caching exists, and `measure_fertility()`/`recommend_vocab_size()`
  exist.
- *Unit test sample* asserted `decode(encode("test input")) == "test input"`. That only holds
  because the input is already lowercase with single spaces (§6).

**Dropped (wouldn't compile or don't apply):**
- Example 5, "Integration with Neural Network": there's no `NeuralNetwork` class.
- "Batch Encoding" pattern: took `const BPETokenizer&` and called non-`const` `encode()`.
- "Vocabulary Expansion" pattern: there's no API to add tokens.
- Machine-translation and classification use cases: generic.
- "Optimization strategies": described code that isn't there.

**Kept and merged:** the BPE algorithm overview, the special-token table, merge and pair-counting
descriptions (now with the separator caveat), escape sequences, the vocab-size and
frequency-threshold guidance (now tied to `recommend_vocab_size()`/`measure_fertility()`),
"specials on or off" guidance, the lowercase note, merge-order precedence, and references.

**References:** Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword
Units* (2016); Radford et al., *Language Models are Unsupervised Multitask Learners* (2019,
GPT-2 pre-tokenizer).
