# `TrainingSampleMeta` — Source File Reference

- **File:** [`src/TrainingSampleMeta.hpp`](../../../../src/TrainingSampleMeta.hpp) (header-only, 196 lines)
- **Namespace:** global (`SampleMeta`, `sample_to_jsonl()`, `parse_jsonl_sample()`); helpers in `tsm_detail`
- **Used by:** every JSONL reader and writer for training pairs: `ChatbotTrainer::load_conversation_data()` (all training input), `DatasetRegistry::count_pairs()` and
  segment slicing, `VocabBuilderHelpers::load_pairs_format()` (vocabulary corpora), and `DataFetcher` (writes Gutenberg/HuggingFace output with `sample_to_jsonl()`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/training_sample_meta_test.cpp`](../../../../tests/training_sample_meta_test.cpp) (20 tests, `trainingSampleMetaTests`; all pass)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real header on lines written by Python's `json.dumps`.

---

## 1. What this file is

The JSONL training-pair format and its hand-written reader/writer.

| Member | Behaviour |
|---|---|
| `SampleMeta` | Optional `domain`, `task_type`, `language`, `split`, `quality` (−1 = unset), `weight` (1.0), `token_count` (−1) |
| `sample_to_jsonl(input, response, meta)` | `{"input":…,"response":…}` plus each set metadata field; escapes `"`, `\`, `\n`, `\r`, `\t`; no trailing newline |
| `parse_jsonl_sample(line, input, response, meta)` | Reads each field with `tsm_detail::json_str()`/`json_flt()`/`json_inum()`; returns false only when `input` is empty (an empty `response` is accepted, TD-267) |
| `tsm_detail::json_str(json, key)` | Finds the first `"key"` anywhere in the line, skips to the `:` and whitespace, reads a quoted string, decoding `\n`, `\t`, `\r`; **any other escape is replaced by the character after the backslash** |

Of the metadata, only `quality` is ever written by the trainer (TD-273); nothing reads `weight`, `split`, `domain`, `task_type`, `language` or
`token_count` when training.

---

## 2. Problems

### `\uXXXX` escapes are not decoded, so non-ASCII text from standard JSON tools is corrupted (verified)

JSON writers usually escape non-ASCII characters by default (Python's `json.dumps`, `ensure_ascii=True`; pandas' `to_json`, `force_ascii=True`). For
the two lines `json.dumps` wrote:

| Field in the file | Parsed as |
|---|---|
| `"Café au lait?"` | `Cafu00e9 au lait?` |
| `"Oui — très bon 😀"` | `Oui u2014 tru00e8s bon ud83dude00` |

So any dataset prepared with standard tooling and queued through `dataset_manager add`/upload trains on garbage wherever it has an accented letter, a
curly quote or apostrophe (`don’t` → `donu2019t`), a dash or an emoji, silently, and the same corrupted text feeds vocabulary building and
pair counting. `\b`, `\f` and `\/` are mis-decoded the same way (`\b` → `b`). DataFetcher's own output is raw UTF-8, but its HuggingFace input scanner
has the same gap (TD-314).

### Keys are matched anywhere in the line (verified)

`json_str()` takes the first occurrence of `"input"`, so `{"meta":{"input":"WRONG"},"input":"right",…}` parsed `input` as `WRONG`. Rows from
datasets with nested metadata, or with a field whose value happens to contain a quoted key earlier in the line, read the wrong text.

### The writer produces invalid JSON for control characters (verified)

`sample_to_jsonl()` escapes only `"`, `\`, `\n`, `\r`, `\t`; a `\x01` or `\x08` is written raw, and Python's `json.loads` rejected the line ("Invalid
control character"). Text extracted from books or web pages can contain such characters, so the files the registry produces aren't readable by other
tools.

### Smaller items (by inspection)

- **Unused metadata:** `weight` and `split` look like sampling and split controls but nothing uses them; `quality` is written only to private copies.
- **Numbers** use `std::stof`/`stoi` (locale-dependent; trailing garbage accepted) and are written with default 6-digit precision; a NaN `quality` is
  written as `nan`.
- **A fourth hand-written JSON scanner** in the codebase (TD-453's duplication).

---

## 3. Tests

`training_sample_meta_test.cpp` (20): defaults, output with and without metadata, escaping of quotes/newlines/backslashes, parsing minimal and full
lines, missing input, sentinels, round trips (including escaped characters), and the quality-from-loss helper used by the trainer.

**Not covered:** `\u` escapes, nested keys, control characters, interoperability with a standard JSON parser.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `\uXXXX` (and `\b`, `\f`, `\/`) not decoded (§2) | Non-ASCII text from standard JSON tools trains as `u00e9`-style garbage | **Verified** | [TD-531](../../guides/TECHNICAL_DEBT.md#td-531-jsonl-training-reader-doesnt-decode-uxxxx-corrupting-non-ascii-text) |
| First-occurrence key matching (§2) | Nested or unusual rows read the wrong field | **Verified** | [TD-532](../../guides/TECHNICAL_DEBT.md#td-532-jsonl-training-reader-matches-keys-anywhere-in-the-line) |
| Writer leaves control characters raw (§2) | Registry-produced JSONL rejected by standard parsers | **Verified** | [TD-533](../../guides/TECHNICAL_DEBT.md#td-533-sample_to_jsonl-writes-raw-control-characters-unused-sample-metadata) |
| Unused metadata; locale/precision/NaN number handling; another hand-written scanner (§2) | Misleading fields; minor | By inspection | [TD-533](../../guides/TECHNICAL_DEBT.md#td-533-sample_to_jsonl-writes-raw-control-characters-unused-sample-metadata) |
| HuggingFace scanner `\u` gap; empty responses; private `quality`; duplicated JSON code | Already tracked | — | TD-314, TD-267, TD-273, TD-453 |
