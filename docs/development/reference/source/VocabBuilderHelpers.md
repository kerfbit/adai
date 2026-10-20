# `VocabBuilderHelpers` — Source File Reference

- **File:** [`src/VocabBuilderHelpers.hpp`](../../../../src/VocabBuilderHelpers.hpp) (header-only, 178 lines)
- **Namespace:** global (`load_plain_text()`, `load_pairs_format()`, `load_json_format()`; since TD-534/TD-535 also `detect_input_format()` and `parse_int_in_range()`)
- **Used by:** [`vocab_builder`](VocabBuilder.md) only (extracted so the loaders can be unit-tested)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.1.0`, `@adai-reviewed: 2026-10-18`
- **Tests:** in [`tests/vocabbuilder_test.cpp`](../../../../tests/vocabbuilder_test.cpp): 7 `LoadPlainTextTest`, 8 `LoadPairsFormatTest`, 7 `LoadJsonFormatTest`, 5 `DetectInputFormat`, 2 `ParseIntInRange` (of the file's 59; all pass)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real header.

---

## 1. What this file is

The three corpus loaders behind `vocab_builder --format`. Each returns a list of text samples (an unopenable file prints to `std::cerr` and returns
an empty list).

| Loader | Behaviour |
|---|---|
| `load_plain_text(file)` | Every non-empty line, unmodified |
| `load_pairs_format(file)` | Looks at the first non-blank line: if it starts with `{`, reads JSONL with `parse_jsonl_sample()` and adds the input and the response as separate samples (lines not starting with `{` are skipped); otherwise legacy format, adding the rest of each line that starts with `INPUT:` or `RESPONSE:` |
| `load_json_format(file)` | Every quoted string in the whole file (keys included, no unescaping; TD-536) |
| `detect_input_format(file)` | TD-534: `vocab_builder --format auto`'s choice, from the first non-blank line after stripping a UTF-8 BOM: `{` or `INPUT:`/`RESPONSE:` → `"pairs"`, `[` → `"json"`, anything else, or an unreadable/empty file → `"plain"` |
| `parse_int_in_range(text, min, max, out)` | TD-535: whole-string decimal integer in `[min, max]` via `strtol`; rejects empty input, a leading space or `+`, trailing characters, overflow and out-of-range values, leaving `out` unchanged |

---

## 2. Gaps

### The legacy loader reads different text from the trainer (verified)

`ChatbotTrainer::load_conversation_data()` trims each legacy line (leading/trailing spaces and `\r`), accepts indented `INPUT:`/`RESPONSE:` lines and
keeps only complete pairs. `load_pairs_format()` does none of that. On a Windows-style (CRLF) file with one indented pair it returned
`[ hello there\r]`, `[ hi friend\r]` and `[ indented answer\r]`: a leading space and a trailing `\r` on every sample (the tokenizer turns the `\r` into a
trailing space), and the indented pair's input dropped while its response was kept. `load_plain_text()` likewise keeps the `\r` of CRLF files. The
vocabulary is therefore learned from slightly different text than the trainer tokenizes. The impact is small (some merges spent on edge whitespace),
but the two parsers should share one implementation.

### Already tracked

| Item | Tracked as |
|---|---|
| `json` loader takes keys and doesn't unescape | TD-536 |
| JSONL `\u` escapes and key matching (inherited; fixed October 17, 2026 in `parse_jsonl_sample()`) | TD-531, TD-532 |
| Legacy-format response truncation in the registry's own parser | TD-336 |

---

## 3. Tests

The 22 loader tests cover plain (empty lines, missing file, content), pairs (JSONL detection, legacy format, mixed/empty input, missing file), json (strings,
empty array, missing file). Since TD-534/TD-535 (October 17, 2026): `detect_input_format()` (JSONL, legacy, BOM, JSON array, plain, missing
file) and `parse_int_in_range()` (bounds and every rejected form).

**Not covered:** CRLF files, indented legacy lines, parity with the trainer's parser.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Legacy/plain loaders keep leading spaces and `\r`, skip indented `INPUT:` lines and keep unpaired texts, unlike the trainer (§2) | Vocabulary learned from slightly different text than training | **Verified** | [TD-537](../../guides/TECHNICAL_DEBT.md#td-537-vocab_builder-legacy-loader-reads-different-text-from-the-trainer) |
| `json` loader, inherited JSONL gaps, registry legacy parser | Already tracked | — | TD-536, TD-531/532, TD-336 |
