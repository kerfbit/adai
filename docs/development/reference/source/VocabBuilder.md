# `VocabBuilder` — Source File Reference

- **File:** [`src/VocabBuilder.cpp`](../../../../src/VocabBuilder.cpp) (295 lines; `main()` of `vocab_builder`)
- **Binary:** `vocab_builder`, the standalone BPE vocabulary tool. A vocabulary must exist before `incremental_trainer` can start (its first-run bootstrap is unreachable, TD-414)
- **Uses:** [`BPETokenizer`](BPETokenizer.md), the loaders in `VocabBuilderHelpers.hpp`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.1.0`, `@adai-reviewed: 2026-10-18`
- **Tests:** none for `main()`; [`tests/vocabbuilder_test.cpp`](../../../../tests/vocabbuilder_test.cpp) (59 tests, `vocabbuilderTests`; all pass) covers the loaders and the tokenizer workflow
- **Operator docs:** [OPERATIONS_MANUAL.md](../../../operations/OPERATIONS_MANUAL.md) §4, [COMMANDS.md](../../../operations/guides/COMMANDS.md)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `vocab_builder` binary.

---

## 1. What this file is

| Flag | Meaning |
|---|---|
| `--input FILE` (repeatable, required) | Input file(s) |
| `--output FILE` (required) | Vocabulary file to write |
| `--vocab-size N` (5000), `--threshold N` (1) | Passed to `BPETokenizer::build_vocab()`; since TD-535's fix parsed with `parse_int_in_range()` (whole string, 1–10,000,000 / ≥ 1), and a bad or missing value exits 1 |
| `--format auto\|pairs\|plain\|json` (default **auto** since TD-534's fix; was `plain`) | `auto`: per file, `detect_input_format()` reads the first non-blank line (`{` or `INPUT:`/`RESPONSE:` → `pairs`, `[` → `json`, else `plain`); `plain`: every non-empty line is a sample, with a warning if the file looks like training data; `pairs`: **JSONL training files** (auto-detected by a leading `{`) or legacy `INPUT:`/`RESPONSE:` lines; `json`: every quoted string in the file |
| `--unicode` | `TokenizerMode::UNICODE` instead of ASCII/byte |
| `--stats` | `print_vocab_stats()` and a "Top 20 tokens" list (actually the lowest IDs, TD-225) |
| `--help` | Usage, exit 0; an unknown option prints usage and exits 1 |

`main()` loads every input with the chosen loader, builds the vocabulary, saves it, optionally prints statistics, then encodes and decodes a fixed
test sentence as a "round-trip test" (a mismatch is reported as "normal for BPE"; the lowercase normalization makes it lossy, TD-224).

---

## 2. Problems

### Fixed: the default format turned JSON syntax into vocabulary (TD-534, October 17, 2026)

**Before:** training data is JSONL, but `--format` defaulted to `plain`, which read each JSONL line as raw text, and the help described `pairs`
only as "INPUT/RESPONSE format". On a small JSONL corpus, `--format plain` produced a vocabulary with JSON syntax as tokens (`{`, `{"`,
`input`, `response`): 6 of 99 entries, none of which ever appears in the text the trainer tokenizes.

**Now:** the default is `auto`, which detects each file's format. The help describes `pairs` as JSONL or `INPUT:`/`RESPONSE:` training data,
and an explicit `--format plain` on training data prints a warning. On 300 TinyStories JSONL lines at `--vocab-size 400`:
- `auto` loaded 600 samples, with no `{`, `input` or `response` tokens. The 5 quote tokens left, such as `?"`, come from dialogue in the stories.
- `plain` learned 18 JSON-ish tokens, including escaped `\"` sequences and `":`.

Side effect to know (rechecked October 19, 2026): under `auto`, a **plain-text** corpus whose first non-blank line starts with `{` or `[` is
read as `pairs` or `json`. Pass `--format plain` for such a file; the warning above doesn't fire in that direction. No repository script
calls `vocab_builder` with or without `--format`; `scripts/package_server_bundle.sh` only copies the binary.

### Fixed: a failed save was reported as success (TD-535, via TD-221)

**Before:** with `--output /nonexistent/dir/v.txt`, the tokenizer logged "Could not open file for writing", and the tool still printed
"Vocabulary saved successfully!" and exited 0, with no file written. **Now:** `BPETokenizer::save_vocab()` throws, so the error reaches
`main()`'s catch and the tool exits 1.

### Fixed: malformed numbers aborted (TD-535, October 17, 2026)

**Before:** `--vocab-size abc` threw an uncaught `std::invalid_argument` from `stoi`, which runs before the `try`, so the process aborted
with a core dump (exit 134). `8000x` was accepted as 8000, and zero or negative sizes weren't rejected (the same class as TD-401/TD-488).

**Now:** `parse_int_in_range()` requires the whole string to be a decimal integer in range. It rejects leading spaces, `+`, trailing
characters and overflow. A value-taking flag with no value, or an unknown `--format`, is reported before any work. Checked with the real
binary: `abc`, `8000x`, `0`, `-5`, `--threshold 0`, `--format xml` and a trailing `--vocab-size` each exit 1 with a message.

### Smaller items (by inspection)

- **The `json` loader** takes every quoted string in the file, including object keys, and doesn't unescape (`\"` ends the string early; `\n` stays as
  two characters).
- **Operator documentation was out of date** (fixed October 17, 2026; §4 now matches `--help`): OPERATIONS_MANUAL.md §4 listed `--output` defaulting to `vocab.txt` (it's required), `--vocab-size`
  defaulting to 10,000 (it's 5,000), and `--min-frequency`, `--num-merges` and `--special-tokens` flags that don't exist (the frequency flag is
  `--threshold`, default 1).
- **The JSONL loader inherited TD-531/TD-532** (`\u` escapes not decoded, keys matched anywhere), so vocabularies built from `json.dumps`-produced data
  learned `u00e9`-style fragments. Fixed with TD-531/TD-532 (October 17, 2026), since the loader uses `parse_jsonl_sample()`.

### Already tracked

| Item | Tracked as |
|---|---|
| `save_vocab()` failed silently (fixed) | TD-221 |
| "Top 20 tokens" are the lowest IDs | TD-225 |
| Lossy lowercase/whitespace normalization | TD-224 |
| BPE merge learning corrupted by the `"|||"` separator (fixed) | TD-220 |
| First-run vocab bootstrap unreachable in `incremental_trainer` | TD-414 |
| JSONL `\u` decoding and key matching (fixed) | TD-531, TD-532 |

---

## 3. Tests

`vocabbuilder_test.cpp` (59). The original 52 cover the three loaders (plain, pairs including JSONL detection, json), multi-source loading, special tokens, and an end-to-end
tokenizer workflow (build, save, load, encode/decode). `main()` itself isn't tested.

The 7 added with TD-534/TD-535's fix cover `detect_input_format()` (JSONL, legacy, BOM, JSON array, plain, missing file) and
`parse_int_in_range()` (valid bounds; empty, letters, trailing characters, leading space, `+`, zero, negative, over range, overflow). Save failure
is covered in `tokenizer_test.cpp` (TD-221).

**Not covered:** `main()` itself; it was checked with the real binary instead (§2).

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Default `plain` format on JSONL training data; help doesn't say `pairs` reads JSONL (§2) | Vocabulary slots wasted on JSON syntax that never occurs in training text | **Fixed October 17, 2026** | [TD-534](../../guides/TECHNICAL_DEBT.md#td-534-vocab_builder-defaults-to-plain-text-turning-jsonl-syntax-into-vocabulary) |
| Save failure reported as success, exit 0 (§2) | Scripts proceed without a vocabulary | **Fixed October 17, 2026** (cause TD-221) | [TD-535](../../guides/TECHNICAL_DEBT.md#td-535-vocab_builder-reports-a-failed-save-as-success-and-aborts-on-malformed-numbers) |
| `stoi` without checks: `abc` aborts, `8000x` accepted, no range check (§2) | Crash or silent misconfiguration | **Fixed October 17, 2026** | [TD-535](../../guides/TECHNICAL_DEBT.md#td-535-vocab_builder-reports-a-failed-save-as-success-and-aborts-on-malformed-numbers) |
| `json` loader takes keys and doesn't unescape; OPERATIONS_MANUAL.md flags/defaults wrong; inherited JSONL gaps (§2) | Polluted corpora; operators misled | By inspection | [TD-536](../../guides/TECHNICAL_DEBT.md#td-536-vocab_builder-json-loader-and-operator-documentation-drift) |
| Silent save, top-tokens, normalization, merge separator, bootstrap, JSONL parsing | Already tracked | — | TD-221, TD-225, TD-224, TD-220, TD-414, TD-531/532 |
