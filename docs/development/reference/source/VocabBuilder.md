# `VocabBuilder` — Source File Reference

- **File:** [`src/VocabBuilder.cpp`](../../../../src/VocabBuilder.cpp) (244 lines; `main()` of `vocab_builder`)
- **Binary:** `vocab_builder`, the standalone BPE vocabulary tool. A vocabulary must exist before `incremental_trainer` can start (its first-run bootstrap is unreachable, TD-414)
- **Uses:** [`BPETokenizer`](BPETokenizer.md), the loaders in `VocabBuilderHelpers.hpp`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** none for `main()`; [`tests/vocabbuilder_test.cpp`](../../../../tests/vocabbuilder_test.cpp) (52 tests, `vocabbuilderTests`; all pass) covers the loaders and the tokenizer workflow
- **Operator docs:** [OPERATIONS_MANUAL.md](../../../operations/OPERATIONS_MANUAL.md) §4, [COMMANDS.md](../../../operations/guides/COMMANDS.md)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `vocab_builder` binary.

---

## 1. What this file is

| Flag | Meaning |
|---|---|
| `--input FILE` (repeatable, required) | Input file(s) |
| `--output FILE` (required) | Vocabulary file to write |
| `--vocab-size N` (5000), `--threshold N` (1) | Passed to `BPETokenizer::build_vocab()` (`std::stoi`) |
| `--format plain\|pairs\|json` (default **plain**) | `plain`: every non-empty line is a sample; `pairs`: **JSONL training files** (auto-detected by a leading `{`) or legacy `INPUT:`/`RESPONSE:` lines; `json`: every quoted string in the file |
| `--unicode` | `TokenizerMode::UNICODE` instead of ASCII/byte |
| `--stats` | `print_vocab_stats()` and a "Top 20 tokens" list (actually the lowest IDs, TD-225) |
| `--help` | Usage, exit 0; an unknown option prints usage and exits 1 |

`main()` loads every input with the chosen loader, builds the vocabulary, saves it, optionally prints statistics, then encodes and decodes a fixed
test sentence as a "round-trip test" (a mismatch is reported as "normal for BPE"; the lowercase normalization makes it lossy, TD-224).

---

## 2. Problems

### The default format turns JSON syntax into vocabulary (verified)

Training data is JSONL, but `--format` defaults to `plain`, which treats each JSONL line as raw text, and the help describes `pairs` only as
"INPUT/RESPONSE format". On a small JSONL corpus, `--format plain` produced a vocabulary containing JSON syntax as tokens (`{`, `{"`, `input`,
`response`), 6 of its 99 entries, none of which ever appears in the text the trainer feeds the tokenizer (it parses the JSONL first); `--format pairs`
produced none. On a real corpus the wasted merges grow with the escape sequences and key names. JSONL should be the default (or auto-detected), and the
help should say `pairs` reads JSONL.

### A failed save is reported as success (verified)

With `--output /nonexistent/dir/v.txt`, the tokenizer logged "Could not open file for writing", and the tool still printed "Vocabulary saved
successfully!", "completed successfully" and **exited 0**; no file existed. The cause is `BPETokenizer::save_vocab()` failing silently (TD-221);
scripts and operators relying on the exit code are misled.

### Malformed numbers abort (verified)

`--vocab-size abc` threw an uncaught `std::invalid_argument` from `stoi` (it runs before the `try`): **abort, core dump, exit 134**. `8000x` is accepted
as 8000, and zero or negative sizes aren't rejected (the same class as TD-401/TD-488).

### Smaller items (by inspection)

- **The `json` loader** takes every quoted string in the file, including object keys, and doesn't unescape (`\"` ends the string early; `\n` stays as
  two characters).
- **Operator documentation is out of date:** OPERATIONS_MANUAL.md §4 lists `--output` defaulting to `vocab.txt` (it's required), `--vocab-size`
  defaulting to 10,000 (it's 5,000), and `--min-frequency`, `--num-merges` and `--special-tokens` flags that don't exist (the frequency flag is
  `--threshold`, default 1).
- **The JSONL loader inherits TD-531/TD-532** (`\u` escapes not decoded, keys matched anywhere), so vocabularies built from `json.dumps`-produced data
  learn `u00e9`-style fragments.

### Already tracked

| Item | Tracked as |
|---|---|
| `save_vocab()` fails silently | TD-221 |
| "Top 20 tokens" are the lowest IDs | TD-225 |
| Lossy lowercase/whitespace normalization | TD-224 |
| BPE merge learning corrupted by the `"|||"` separator | TD-220 |
| First-run vocab bootstrap unreachable in `incremental_trainer` | TD-414 |
| JSONL `\u` decoding and key matching | TD-531, TD-532 |

---

## 3. Tests

`vocabbuilder_test.cpp` (52): the three loaders (plain, pairs including JSONL detection, json), multi-source loading, special tokens, and an end-to-end
tokenizer workflow (build, save, load, encode/decode). `main()` itself isn't tested.

**Not covered:** the default format on JSONL input, save failure, argument parsing.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Default `plain` format on JSONL training data; help doesn't say `pairs` reads JSONL (§2) | Vocabulary slots wasted on JSON syntax that never occurs in training text | **Verified** | [TD-534](../../guides/TECHNICAL_DEBT.md#td-534-vocab_builder-defaults-to-plain-text-turning-jsonl-syntax-into-vocabulary) |
| Save failure reported as success, exit 0 (§2) | Scripts proceed without a vocabulary | **Verified** (cause TD-221) | [TD-535](../../guides/TECHNICAL_DEBT.md#td-535-vocab_builder-reports-a-failed-save-as-success-and-aborts-on-malformed-numbers) |
| `stoi` without checks: `abc` aborts, `8000x` accepted, no range check (§2) | Crash or silent misconfiguration | **Verified** | [TD-535](../../guides/TECHNICAL_DEBT.md#td-535-vocab_builder-reports-a-failed-save-as-success-and-aborts-on-malformed-numbers) |
| `json` loader takes keys and doesn't unescape; OPERATIONS_MANUAL.md flags/defaults wrong; inherited JSONL gaps (§2) | Polluted corpora; operators misled | By inspection | [TD-536](../../guides/TECHNICAL_DEBT.md#td-536-vocab_builder-json-loader-and-operator-documentation-drift) |
| Silent save, top-tokens, normalization, merge separator, bootstrap, JSONL parsing | Already tracked | — | TD-221, TD-225, TD-224, TD-220, TD-414, TD-531/532 |
