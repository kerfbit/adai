# `DataFetcher` — Source File Reference

- **Files:** [`src/DataFetcher.hpp`](../../../../src/DataFetcher.hpp), [`src/DataFetcher.cpp`](../../../../src/DataFetcher.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `registry_server` and `dataset_manager` (with `ParquetReader`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/DataFetcherTests.cpp`](../../../../tests/DataFetcherTests.cpp) (27 tests, offline); the fetch endpoints' input validation in `dataset_registry_live_test.cpp`
- **Origin:** split out of `IncrementalTrainer.cpp` by TD-028 (dataset manager separation); Phase 11 (server-side fetch), Phase 12/13 (cached rotating slices, Gutenberg cleaning); native Parquet via `ParquetReader`
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real code
> offline. Network calls were blocked with a dead HTTPS proxy, so nothing reached Gutenberg or
> HuggingFace.

---

## 1. What this file is

The training pipeline's **only importer of outside text**. It turns two public sources into
the JSONL `{"input","response",…}` files the dataset registry queues for training:

| Source | Download | Conversion |
|---|---|---|
| Project Gutenberg | One `.txt` per book via `curl` | Strip boilerplate, split into sentences, synthesize QA/"Summarize" pairs from consecutive sentences |
| HuggingFace datasets | `datasets-server` `/parquet` listing, then each Parquet file via `curl`, converted natively by `ParquetReader` | Auto-detect input/output fields, dialog arrays, or split single-text rows in half |

It has no dependency on the model, tokenizer or registry. It returns file paths, and the caller
queues them.

### Why it matters

Everything a model learns from these two sources passes through here, so its conversion choices
*are* the training data's quality (§5). It also runs **inside `registry_server`**, which serves
the Android ops dashboard's Gutenberg/HuggingFace fetch dialogs. Its crash and filesystem
behaviour is therefore the registry daemon's behaviour (§6).

---

## 2. Where it's used

| Caller | Methods | Notes |
|---|---|---|
| `registry_server` `POST /registry/<group>[/<kind>]/fetch/gutenberg` | `ensure_gutenberg_cached()` then static `convert_gutenberg_slice()` | Runs synchronously in the HTTP handler thread. Cache under `<group_dir>/datasets/gutenberg_<id>/`; a per-(book, model) cursor file advances through the book |
| `registry_server` `…/fetch/huggingface` | `ensure_huggingface_cached()` then static `convert_huggingface_slice()` | Same pattern. `dataset_id` and `split` are **allow-listed** to `[A-Za-z0-9._-/]+` first, because this code builds shell commands from them (§6). `model_name` is allow-listed too |
| `dataset_manager gutenberg` / `gutenberg-batch` / `huggingface` (local mode) | `fetch_gutenberg()`, `fetch_gutenberg_batch()`, `fetch_huggingface()` | Writes under `./gutenberg_data` / `./huggingface_data`, then `add_file()`. In remote mode `dataset_manager` calls the registry endpoints instead |

---

## 3. API

### Configuration

`FetcherConfig { gutenberg_output_dir = "gutenberg_data", huggingface_output_dir = "huggingface_data" }`,
relative to the working directory. The class is otherwise stateless; every helper is static.

### Gutenberg

| Method | Behaviour |
|---|---|
| `fetch_gutenberg(book_id, num_pairs = 500)` | Downloads to `<dir>/gutenberg_<id>.txt` (**re-downloaded every call**), converts the first `num_pairs` pairs to `<dir>/gutenberg_<id>_training.jsonl`. Returns the path or `""` |
| `fetch_gutenberg_batch(ids, n)` | Loops `fetch_gutenberg()`; `""` at each failed index |
| `ensure_gutenberg_cached(book_id)` | If `<dir>/gutenberg_<id>/sentences.txt` exists and is non-empty, returns it. Otherwise downloads `book.txt`, cleans it, extracts sentences and writes them one per line |
| `convert_gutenberg_slice(sentences, out, offset, n, &next)` | Builds up to `n` pairs starting at sentence `offset` (an out-of-range offset is clamped), then wraps to sentence 0 once if short. Returns pairs written and the next offset |

**Download** (`download_file`) runs `curl -L -f -s -o "<path>" "<url>"` through `std::system()`,
first for `https://www.gutenberg.org/files/<id>/<id>-0.txt`, then the `<id>.txt` fallback. There's
no timeout, and the exit status is checked plus a non-empty file.

**Cleaning** (`clean_gutenberg_text`), in this order:
1. Cut everything before the line containing `*** START OF` and from `*** END OF` onwards.
2. `strip_toc()`: drop the first "Contents"/"Table of contents" block, up to the first standalone
   chapter marker within 500 lines (if no marker is found, nothing is dropped).
3. `strip_illustration_markers()`: `[Illustration…]`, `[Figure…]`, `[Frontispiece…]`, `[Footnote…]`.
4. `strip_chapter_markers()`: lines that are exactly `CHAPTER|BOOK|PART <roman|digits>[.]`, plus
   the next non-blank line if it's ≤ 60 characters with no terminal `.!?` (assumed to be a title).
5. Collapse runs of spaces/tabs, and 3+ newlines to 2.

**Sentences** (`extract_sentences`): protect decimals (`3.0`), match the regex `[^.!?]+[.!?]+`,
trim, collapse whitespace, and keep 21–499 characters.

**Pairs** (`create_qa_pairs_from_text`), stepping two sentences at a time:
- `(template + s[i], s[i+1])`, where the template is one of six ("What does this mean: ", "Can you
  explain: ", …) picked with unseeded `rand()`;
- `("Summarize: " + s[i] + " " + s[i+1], s[i+2])`.

Metadata is `task_type="qa"`, `domain="literature"`, `language="en"`.

### HuggingFace

| Method | Behaviour |
|---|---|
| `fetch_huggingface(id, n, split, in, out)` | Downloads into `<dir>/<safe_id>_<split>/`, converts the first `n` pairs to `<dir>/<safe_id>_<split>_training.jsonl`. Prints progress with `std::cout`/`std::cerr` (not `Logger`) |
| `ensure_huggingface_cached(id, split)` | Returns `<dir>/<safe_id>_<split>/full_dataset.jsonl` if it exists and is non-empty, otherwise downloads it |
| `convert_huggingface_slice(jsonl, out, in, out_f, offset, n, &next)` | Converts from row `offset`, wrapping to row 0 once if short |

`safe_id` replaces `/` with `_` in the dataset ID. **`split` is used as-is** in the path (§6).

**Download** (`download_hf_full_dataset`):
1. `curl` `https://datasets-server.huggingface.co/parquet?dataset=<id>&config=default&split=<split>`
   into `parquet_info.json`. The `HF_TOKEN` environment variable, if set, is added as an
   `Authorization: Bearer` header **on the curl command line**.
2. Extract each `parquet_files[].url` with a hand-rolled scanner.
3. `curl` each URL to `parquet/part_<i>.parquet`.
4. `ParquetReader::convert_to_jsonl()` writes the parts into `full_dataset.jsonl` (the first part
   truncates, later parts append).

A part that fails to download or convert is **logged and skipped**. The method succeeds if at
least one part converted. Only `config=default` is ever requested.

**Conversion** (`convert_hf_to_training_data`), per JSONL row after `offset`. On the first row it
tries ten field-name pairs (`instruction/output`, `question/answer`, `prompt/completion`, …,
`text/label`) and accepts the first whose two values are both non-empty strings. Then:

| Detected shape | Pair produced |
|---|---|
| Two fields | `(in [+ "\n" + "input" for instruction datasets], out)` |
| Same field twice | Split the text at a sentence end near its middle → `task_type="completion"` |
| No field pair | Look for a dialog array (`dialog`, `turns`, `utterances`, `conversations`) → pairs `(t0,t1)`, `(t2,t3)`, … with `task_type="chat"` |
| None of the above | Use the first `text`/`content`/`document` field of ≥ 40 chars as a split single-text field, and lock that in for later rows |

JSON is read with small string scanners (`hf_extract_string` and friends), not a JSON library.
They find the first `"key"` occurrence anywhere in the row, including inside a nested object,
and keep `\uXXXX` escapes as literal text.

---

## 4. Rotating slices (registry use)

The registry keeps one cursor file per (source, model) and calls the `convert_*_slice()` helpers
with it, so repeated fetches for a model walk through a book or dataset instead of re-serving its
start. Different models get independent cursors over one shared cached download. When a source runs
out mid-request the slice tops up from the start once, so the last slice of a pass repeats the
earliest pairs (verified: a 3-row dataset at offset 2 with `n=3` produced rows 3, 1, 2).

---

## 5. Training-data quality (verified)

The Gutenberg conversion produces **non-sequitur pairs by construction**. Running the real
splitter and pair builder on three Pride-and-Prejudice-style sentences gave:

```text
Q: Can you explain: Darcy walked over to Mrs.
A: Bennet in the drawing room.
Q: Summarize: Darcy walked over to Mrs. Bennet in the drawing room.
A: It was a truth universally acknowledged, said Dr.
```

- **The "answer" to every question is just the next sentence**, and the "summary" of two
  sentences is the third. The model is trained to respond to "Can you explain: X" with whatever
  the author wrote next.
- **Abbreviations break sentences.** `Mr.`, `Mrs.` and `Dr.` end a "sentence", so `Mr.` is dropped
  (too short) and the following text is split mid-clause. Gutenberg's 19th-century novels are dense
  with these.
- The question template comes from unseeded `rand()`, so every run produces the same template
  sequence.
- Everything is tagged `language="en"` regardless of the book.

The HuggingFace path is in better shape, since it uses the dataset's own input/output columns
when they exist, but its fallbacks have the same flavour: a single-text row cut in half becomes
"prompt = first half, response = second half".

---

## 6. Robustness and security

> **The sentence regex crashes the process on long unpunctuated text (verified).** libstdc++'s
> `std::regex` recurses per character, and `[^.!?]+[.!?]+` over a run with no `.`, `!` or `?`
> blew the stack: 20,000 characters was fine, but **100,000 and 400,000 characters died with
> SIGSEGV**. Reachable cases include a book in a script without ASCII sentence punctuation (e.g.
> Chinese, which uses `。`), a long poem, or a large index or table. `fetch/gutenberg` runs this
> inside `registry_server`'s request thread, so **one fetch request can kill the registry
> daemon**, and every later retry of the same book kills it again (nothing is cached on failure).

> **`split` escapes the data directory (verified).** The registry allow-list permits `.` and `/`,
> and `split` is concatenated into the cache path unflattened. `split =
> "x/../../../../ESCAPED"` made `ensure_huggingface_cached()` create `ESCAPED/` **outside** the
> group's data directory. With a real download, `parquet_info.json`, `parquet/` and
> `full_dataset.jsonl` would be written there, and the registry's training-file path is built
> from `split` the same way. The allow-list should reject `..` (or flatten `split` like
> `dataset_id`).

By inspection:

- **Shell commands from strings.** Every download is `std::system("curl … \"<path>\" \"<url>\"")`.
  Inside double quotes, `$(…)` and backticks still expand.
  - The registry allow-lists `dataset_id`/`split`, so its endpoint isn't injectable.
  - `dataset_manager`'s local mode doesn't validate them (the operator's own shell, so not an
    escalation, but a pasted ID with `$(…)` runs it).
  - The Parquet URLs, which come from HuggingFace's API response, are interpolated unvalidated.
  - `HF_TOKEN` appears in the curl command line, visible to other local users via `ps` for the
    download's duration.
  - Calling libcurl directly would remove all four problems. The project already uses it
    optionally, for the FTP dataset transport (`DataTransport.hpp`, linked into `incremental_trainer`
    when found).
- **No timeouts.** `curl` runs without `--max-time`/`--connect-timeout`, synchronously in the
  registry's request thread. A stalled server holds that thread indefinitely.
- **Partial downloads become the permanent cache.** If any Parquet part fails, the rest still
  form `full_dataset.jsonl`, and `ensure_huggingface_cached()` treats any non-empty file as
  complete, so the missing parts are never retried. An interrupted download (a crash mid-append)
  likewise leaves a truncated cache that's served forever.
- **Concurrent fetches of the same dataset race.** Two requests for the same dataset (different
  models, or a retry) download into the same `parquet/part_<i>.parquet` and `full_dataset.jsonl`
  at once. The second request may also see the first's half-written cache as complete.
- **Only `config=default`.** Multi-config datasets (where the parquet listing is per named config)
  fail with "No parquet files found"; there's no way to choose a config.
- **Gutenberg URL scheme.** Only `files/<id>/<id>-0.txt` and `<id>.txt` are tried, not the
  `cache/epub/<id>/pg<id>.txt` path Gutenberg also serves; not checked against the live site here.
- `fetch_gutenberg()` always re-downloads (no cache), unlike `ensure_gutenberg_cached()`.
- `fetch_huggingface()`/`download_hf_full_dataset()` print with `std::cout`/`std::cerr` in library
  code, against the logging convention.

---

## 7. Tests

`DataFetcherTests.cpp` has 27 offline tests:

| Group | Tests |
|---|---|
| `DataFetcherGutenbergCleaningTest` | 10: TOC, chapter and illustration stripping (a `friend` of the class) |
| `GutenbergSliceTest` | 5: offsets, wraparound |
| `HuggingfaceSliceTest` | 8: field detection, offsets, wraparound |
| `DataFetcherTest`, `FetcherConfigTest` | 4 |

**Not covered:**
- downloads (no network in tests);
- long unpunctuated input;
- abbreviations;
- path components in `split`;
- partial or concurrent caches.

---

## 8. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `extract_sentences()` regex segfaults on ~100k+ chars without `.!?` (§6) | One `fetch/gutenberg` request can crash `registry_server`, repeatably | **Verified** | [TD-309](../../guides/TECHNICAL_DEBT.md#td-309-gutenberg-sentence-splitting-can-crash-registry_server) |
| `split` isn't flattened and the allow-list permits `..` (§6) | Directories and files written outside the registry data dir via `fetch/huggingface` | **Verified** | [TD-310](../../guides/TECHNICAL_DEBT.md#td-310-huggingface-split-can-write-outside-the-registry-data-directory) |
| Gutenberg pairs are next-sentence non-sequiturs; abbreviations split sentences; unseeded template choice; fixed `language="en"` (§5) | Low-quality training data from every Gutenberg fetch | **Verified** | [TD-311](../../guides/TECHNICAL_DEBT.md#td-311-gutenberg-training-pairs-are-next-sentence-non-sequiturs) |
| `std::system` curl commands from strings; `HF_TOKEN` on the command line; HF-supplied URLs unvalidated (§6) | Injection where callers don't validate; token visible in `ps` | By inspection | [TD-312](../../guides/TECHNICAL_DEBT.md#td-312-datafetcher-downloads-through-shell-curl-commands) |
| No curl timeouts; synchronous in the registry handler (§6) | A stalled download holds a registry worker forever | By inspection | [TD-312](../../guides/TECHNICAL_DEBT.md#td-312-datafetcher-downloads-through-shell-curl-commands) |
| Partial or interrupted HF downloads cached as complete; concurrent fetches share cache files (§6) | Silently incomplete or corrupt datasets served indefinitely | By inspection | [TD-313](../../guides/TECHNICAL_DEBT.md#td-313-huggingface-cache-accepts-partial-downloads-and-races-concurrent-fetches) |
| `config=default` only; Gutenberg URL patterns limited; `fetch_gutenberg()` never caches; `std::cout` in library code (§6) | Some datasets/books unfetchable; minor | By inspection | [TD-314](../../guides/TECHNICAL_DEBT.md#td-314-datafetcher-source-coverage-and-hygiene-gaps) |

---

## Addendum (October 24, 2026): `split` is flattened (TD-310)

`fetch_huggingface()` and `ensure_huggingface_cached()` name the per-dataset directory with `hf_dataset_dir_name()`:
`<dataset_id>_<split>` with every `/` and `\` turned into `_`, so it is always one path component under
`huggingface_output_dir`. Before, only `dataset_id` was flattened, and a `split` like `x/../../ESCAPED` wrote outside it. Test:
`SplitIsFlattenedIntoTheOutputDirectory`.

---

## Addendum (October 24, 2026): linear text processing (TD-309)

`extract_sentences()`, `strip_illustration_markers()` and `clean_gutenberg_text()`'s whitespace collapses are linear loops instead of
`std::regex`, whose engine recursed per character and segfaulted on about 100,000 characters without a match. That let one
`fetch/gutenberg` request kill `registry_server`. The sentence rules are unchanged; a test checks them against the old regex pipeline.
Tests: `MegabyteInputsWithoutBoundariesReturnNormally`, `SentenceScannerMatchesTheOldRegexOnProse`.
