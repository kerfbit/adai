# `DataFetcher` — Source File Reference

- **Files:** [`src/DataFetcher.hpp`](../../../../src/DataFetcher.hpp), [`src/DataFetcher.cpp`](../../../../src/DataFetcher.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_core`, used by `registry_server` and `dataset_manager` (with `ParquetReader` and `HttpDownload`)
- **Status tag:** `@adai-status: stable`; `DataFetcher.hpp` `@adai-version: 1.5.0`, `DataFetcher.cpp` `@adai-version: 1.8.0`, `@adai-reviewed: 2026-11-19`
- **Tests:** [`tests/DataFetcherTests.cpp`](../../../../tests/DataFetcherTests.cpp) (63 tests; `HuggingFaceCacheTest` and `GutenbergMirrorTest` run against a local httplib server, everything else is offline); the fetch endpoints' input validation in `dataset_registry_live_test.cpp`
- **Origin:** split out of `IncrementalTrainer.cpp` by TD-028 (dataset manager separation); Phase 11 (server-side fetch), Phase 12/13 (cached rotating slices, Gutenberg cleaning); native Parquet via `ParquetReader`
- **Last traced against the code:** 2026-11-19 (TD-579: the Gutenberg mirror fallback in §3, §6, §7 and §8; TD-314 rewrote §1–§8 earlier the same day; the addenda below record how it got there)

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real code.
> Downloads in the tests go to a local server; nothing in the tests reaches Gutenberg or HuggingFace.

---

## 1. What this file is

The training pipeline's **only importer of outside text**. It turns two public sources into
the JSONL `{"input","response",…}` files the dataset registry queues for training:

| Source | Download | Conversion |
|---|---|---|
| Project Gutenberg | One plain-text file per book through `adai::http_download()` (libcurl), from a mirror when the main site is unavailable (TD-579) | Strip boilerplate, split into sentences, make continuation pairs (two sentences in, the next two out; TD-311) |
| HuggingFace datasets | `datasets-server` `/parquet` listing, then each Parquet part, converted natively by `ParquetReader` | Parse each row as JSON; auto-detect input/output fields, dialog arrays, utterance-per-row dialogs, or split single-text rows in half |

It has no dependency on the model, tokenizer or registry. It returns file paths, and the caller
queues them.

### Why it matters

Everything a model learns from these two sources passes through here, so its conversion choices
*are* the training data's quality (§5). It also runs **inside `registry_server`**, which serves
`dataset_manager`'s remote mode and the Android ops dashboard's Gutenberg/HuggingFace fetch dialogs.
Its crash and filesystem behaviour is therefore the registry daemon's behaviour (§6).

---

## 2. Where it's used

| Caller | Methods | Notes |
|---|---|---|
| `registry_server` `POST /registry/<group>[/<kind>]/fetch/gutenberg` | `ensure_gutenberg_cached()` then static `convert_gutenberg_slice()` | Runs synchronously in the HTTP handler thread. Cache under `<group_dir>/datasets/gutenberg_<id>/`; a per-(book, model) cursor file advances through the book |
| `registry_server` `…/fetch/huggingface` | `ensure_huggingface_cached(id, split, config)` then static `convert_huggingface_slice()` | Same pattern. `dataset_id` (`name` or `org/name`), `split` and a non-empty `config` are allow-listed to `[A-Za-z0-9._-]+` components, never `.` or `..`; `model_name` too. A named config has its own cursor (TD-314) |
| `dataset_manager gutenberg` / `gutenberg-batch` / `huggingface [--config C]` (local mode) | `fetch_gutenberg()`, `fetch_gutenberg_batch()`, `fetch_huggingface()` | Writes under `./gutenberg_data` / `./huggingface_data`, then `add_file()`. In remote mode `dataset_manager` calls the registry endpoints instead |

---

## 3. API

### Configuration

`FetcherConfig`:

| Field | Default | Use |
|---|---|---|
| `gutenberg_output_dir` | `gutenberg_data` | Book caches and local training files |
| `huggingface_output_dir` | `huggingface_data` | Dataset caches and local training files |
| `huggingface_api_url` | `https://datasets-server.huggingface.co` | Base of the `/parquet` listing (TD-313) |
| `gutenberg_site_url` | `https://www.gutenberg.org` | Project Gutenberg's main site, tried first (TD-579) |
| `gutenberg_mirrors` | `kDefaultGutenbergMirrors`: `https://gutenberg.pglaf.org`, `https://mirrors.xmission.com/gutenberg`, `https://mirror.csclub.uwaterloo.ca/gutenberg` | Comma-separated mirror base URLs, tried in order when the main site is unavailable; empty = the main site only (TD-579). `ServiceConfig::gutenberg_mirrors` (`GUTENBERG_MIRRORS`) sets it in `registry_server`'s `fetch/gutenberg` and `dataset_manager`'s local `gutenberg`/`gutenberg-batch` |
| `extra_download_hosts`, `allow_plain_http` | empty, `false` | Tests against a local server only |

Paths are relative to the working directory. `fetch_gutenberg()`, `ensure_*_cached()` and
`fetch_huggingface()` read the configuration; the slice converters and text helpers are static.

### Gutenberg

| Method | Behaviour |
|---|---|
| `fetch_gutenberg(book_id, num_pairs = 500)` | Uses the cached `<dir>/gutenberg_<id>/book.txt`, downloading it only when it's missing or empty (TD-314), and converts the first `num_pairs` pairs (`<= kAllPairs`: all) to `<dir>/gutenberg_<id>_training.jsonl`. Returns the path or `""` |
| `fetch_gutenberg_batch(ids, n)` | Loops `fetch_gutenberg()`; `""` at each failed index |
| `ensure_gutenberg_cached(book_id)` | Returns `<dir>/gutenberg_<id>/sentences_v2.txt` if it exists and is non-empty. Otherwise gets `book.txt` the same way (cached or downloaded), cleans it, extracts sentences and writes them one per line |
| `convert_gutenberg_slice(sentences, out, offset, n, &next)` | Builds up to `n` pairs starting at sentence `offset` (an out-of-range offset is clamped), then wraps to sentence 0 once if short. `n <= kAllPairs` (0): every pair to the end of the book, no wrap (TD-575). Returns pairs written and the next offset |

**Download** (`download_gutenberg_book()`) tries `gutenberg_urls(id, gutenberg_site_url)` in order
(TD-314):
1. `<site>/files/<id>/<id>-0.txt` (the UTF-8 edition);
2. `<site>/files/<id>/<id>.txt`;
3. `<site>/cache/epub/<id>/pg<id>.txt` (the generated text every book has).

Each attempt is `adai::http_download()`: `https://` on an allowed host only, timeouts, and the body
renamed into place only after a complete 2xx download (TD-312). Its outcome is one of three
(TD-579):

| Outcome | Meaning | Next |
|---|---|---|
| A Gutenberg text | `looks_like_gutenberg_text()`: not an HTML page (the first non-blank character after any UTF-8 BOM isn't `<`) and `*** START OF` or `***START OF` within the first 256 KiB | Renamed from `book.txt.download` to `book.txt`; done |
| Not there | Any other HTTP error, `404` above all | The source's next file name |
| Unavailable | No answer (connection, TLS, timeout: status 0), `429`, `5xx`, or a 200 that isn't a Gutenberg text | Leave this source |

When the main site is unavailable, each of `gutenberg_mirrors` (`split_gutenberg_mirrors()`: trimmed,
trailing `/` dropped, empty entries skipped) is tried in turn with `gutenberg_mirror_urls(id, base)`:
the digit-path layout, one directory per digit of the ID but the last, then the ID (`0/<id>/` for a
one-digit ID), with `<id>-0.txt` then `<id>.txt` (mirrors don't all carry `cache/epub`). Each URL's own
host is allowed for that request only. When every main-site name answered "not there", the book has
no plain-text file and no mirror is asked. One warning per book names the main site's error when it
falls back; each mirror's failure is logged at debug, and the last one is in the final error. A
cached `book.txt` is therefore never partial and never an error page. **Verified** in
`GutenbergMirrorTest` against a local server; the mirror layout and status codes by HEAD requests to
the real mirrors on November 19, 2026 (`0/5/5.txt`, `1/3/4/1342/1342-0.txt` and
`2/9/8/3/29838/29838.txt` 200 on pglaf; `0/1/1-0.txt` 404).

**Cleaning** (`clean_gutenberg_text`), in this order:
1. Cut everything before the line containing `*** START OF` and from `*** END OF` onwards.
2. `strip_toc()`: drop the first "Contents"/"Table of contents" block, up to the first standalone
   chapter marker within 500 lines (if no marker is found, nothing is dropped).
3. `strip_illustration_markers()`: `[Illustration…]`, `[Figure…]`, `[Frontispiece…]`, `[Footnote…]`.
4. `strip_chapter_markers()`: lines that are exactly `CHAPTER|BOOK|PART <roman|digits>[.]`, plus
   the next non-blank line if it's ≤ 60 characters with no terminal `.!?` (assumed to be a title).
5. Collapse runs of spaces/tabs, and 3+ newlines to 2. Every step is a linear loop (TD-309).

**Sentences** (`extract_sentences`): a linear scanner that ends a sentence at `.`, `!` or `?`, except
after a title (`Mr.`, `Mrs.`, `Dr.`, … 24 in all), a capital initial other than `I`/`A`, a dotted form
(`e.g.`, `U.S.`) or a dot between letters or digits; trims, collapses whitespace, and keeps 21–499
characters (TD-311).

**Pairs** (`create_continuation_pairs`): two consecutive sentences in, up to the next two out, no
overlap, tagged `task_type="completion"`, `domain="literature"`, and `language` from the book's own
`Language:` header (`gutenberg_language()`), empty when mixed, unknown or missing (TD-311).

### HuggingFace

| Method | Behaviour |
|---|---|
| `fetch_huggingface(id, n, split, in, out, config = "")` | Gets the dataset through `ensure_huggingface_cached()` (TD-314: it downloaded it again on every call), converts the first `n` pairs to `<dir>/<folder>_training.jsonl`. Progress and errors go through `Logger` (TD-314) |
| `ensure_huggingface_cached(id, split, config = "")` | Returns the folder's `full_dataset.jsonl` if its `full_dataset.complete` marker exists; otherwise downloads it, one download per folder at a time in the process (TD-313) |
| `convert_huggingface_slice(jsonl, out, in, out_f, offset, n, &next)` | Converts from row `offset`, wrapping to row 0 once if short; `n <= kAllPairs` (0): the rest of the dataset, no wrap (TD-575) |

The dataset folder is `hf_dataset_dir_name()`: `<id>_<split>`, plus `__<config>` when a config is
named (TD-314), with every `/` and `\` turned into `_` (TD-310). An unnamed config keeps the
pre-TD-314 name, so existing caches stay valid.

**Download** (`download_hf_full_dataset`), all inside a private `.download-<random>` folder (TD-313):
1. The `/parquet` listing: `?dataset=<id>&config=<config>&split=<split>` when a config is named,
   else `?dataset=<id>` (the whole dataset's list). Query values are percent-encoded; `HF_TOKEN`, if
   set, is an in-process `Authorization: Bearer` header.
2. `hf_parquet_urls()` parses the reply with nlohmann/json and keeps `parquet_files` entries whose
   `split` matches. With no config named it takes `default` when the dataset has it, else the
   dataset's only config; several others are an error naming them (TD-314: only `config=default`
   was ever requested before, so a dataset without one had "no parquet files"). An entry without a
   `config` or `split` field matches any.
3. Each part is downloaded and converted by `ParquetReader::convert_to_jsonl()`, then deleted.
4. Only when every part converted: the JSONL is renamed over `full_dataset.jsonl` and
   `full_dataset.complete` (dataset, split, config, parts, rows) written atomically. One failed part
   caches nothing.

**Conversion** (`convert_hf_to_training_data`), per JSONL row after `offset`. Each row is parsed
once with nlohmann/json (TD-314); a line that starts with `{` but isn't a JSON object counts as a
row and yields nothing. Only top-level fields are read, and `\uXXXX` escapes are decoded. On the
first row it tries ten field-name pairs (`instruction/output`, `question/answer`,
`prompt/completion`, …, `text/label`) and accepts the first whose two values are both non-empty
strings. Then:

| Detected shape | Pair produced |
|---|---|
| Two fields | `(in [+ "\n" + "input", or "context" when there's no "input", for instruction datasets], out)`: Alpaca's material is `input`, Dolly's `context` (TD-575) |
| Same field twice | Split the text at a sentence end near its middle → `task_type="completion"` |
| No field pair | Look for a dialog array (`dialog`, `turns`, `utterances`, `conversations`) of strings only → pairs `(t0,t1)`, `(t2,t3)`, … with `task_type="chat"`. An array holding anything but strings (e.g. ShareGPT's `{"from","value"}` objects) isn't read as turns |
| One utterance per row (first row has `utterance` and a string or numeric `dialog_id`, no fields given) | TD-575: rows are grouped by `dialog_id` (consecutive rows), each utterance trimmed, and a dialog's turns paired the same way when the id changes or the file ends |
| None of the above | Use the first `text`/`content`/`document` field of ≥ 40 chars as a split single-text field, and lock that in for later rows |

---

## 4. Rotating slices (registry use)

The registry keeps one cursor file per (source, model), and for HuggingFace per (dataset, split,
config, model), and calls the `convert_*_slice()` helpers with it, so repeated fetches for a model
walk through a book or dataset instead of re-serving its start. Different models get independent
cursors over one shared cached download. When a source runs out mid-request the slice tops up from
the start once, so the last slice of a pass repeats the earliest pairs (verified: a 3-row dataset at
offset 2 with `n=3` produced rows 3, 1, 2).

---

## 5. Training-data quality

Gutenberg pairs are continuation text (TD-311): two sentences of a book in, the next two out. Before
TD-311 they were template "questions" answered by the next sentence, with `Mr.`/`Mrs.`/`Dr.` splitting
sentences mid-clause and every book tagged `language="en"`.

The HuggingFace path uses the dataset's own input/output columns when they exist. Its fallbacks are
weaker: a single-text row cut in half becomes "prompt = first half, response = second half". Before
TD-314 its string scanners kept `\uXXXX` escapes as literal text and could read a nested field of the
same name as the row's own. The escape gap mattered little in practice: `ParquetReader` writes
non-ASCII text as raw UTF-8 and escapes only control characters, so only those (rare in text columns)
reached training as `\u00XX`. A row whose strings aren't valid UTF-8 now fails to parse and is skipped;
the scanners passed it through.

---

## 6. Robustness and security

Fixed, and kept here because they explain the current shape (details in the addenda):

- One long unpunctuated text crashed `registry_server` through `std::regex` recursion (TD-309, now
  linear scanners).
- `split` could write outside the registry's data directory (TD-310, now flattened and allow-listed).
- Downloads were `std::system("curl …")` command lines with `HF_TOKEN` visible in `ps`, no timeouts,
  and unvalidated Parquet URLs (TD-312, now `adai::http_download()`).
- Partial or concurrent HuggingFace downloads became the permanent cache (TD-313, now all-or-nothing
  under a per-dataset lock).
- Every Gutenberg fetch failed while `www.gutenberg.org` was down, and whatever downloaded was cached
  as the book, an error page included (TD-579, now mirrors and a content check).

Open:

- `ensure_gutenberg_cached()` takes no lock: two concurrent first fetches of one book both download
  it. Each download is renamed into place whole, so the result is a wasted download, not a corrupt
  cache.
- Every fetch runs synchronously in a `registry_server` request thread, bounded by the download
  timeouts and `REGISTRY_MAX_CONCURRENT_FETCHES` (TD-494).

---

## 7. Tests

`DataFetcherTests.cpp` has 63 tests:

| Group | Tests |
|---|---|
| `DataFetcherGutenbergCleaningTest` | 14: TOC, chapter and illustration stripping, linear scanning of megabyte inputs, the sentence rules against the old regex, abbreviations, the `Language:` header (a `friend` of the class) |
| `GutenbergSliceTest` | 8: offsets, wraparound, `kAllPairs`, continuation pairs, language |
| `HuggingfaceSliceTest` | 13: field detection, offsets, wraparound, `kAllPairs`, an instruction's `input` or `context`, utterance-per-row dialogs, JSON rows (`\uXXXX`, nested fields, a broken row) and string-only dialog arrays (TD-314) |
| `HuggingFaceCacheTest` | 7, against a local server: a failed part caches nothing, an unmarked cache is downloaded again, concurrent requests download once, `fetch_huggingface()` downloads once, and config selection (TD-314: `default` among several, a dataset's only config, several needing one named) |
| `GutenbergMirrorTest` | 9, against a local server playing the main site and two mirrors (TD-579): the main site used when it answers; a `503`, `429`, `502` or an HTML page falls back; a `404` asks no mirror; a mirror's HTML page is rejected for the next mirror; `<id>.txt` after `<id>-0.txt`; `0/<id>/`; an empty mirror list |
| `DataFetcherDownloadTest` | 6: no shell, no plain HTTP, the Gutenberg URL order (TD-314), the mirrors' digit paths, the mirror list's trimming and the text check (TD-579) |
| `DataFetcherTest`, `FetcherConfigTest` | 6, including `split` flattening (TD-310) and `fetch_gutenberg()` reading a cached book (TD-314) |

**Not covered:** a real download from Gutenberg, its mirrors or HuggingFace (the tests never leave the machine). `GUTENBERG_MIRRORS`'s parsing and validation are in `config_test.cpp` (`GutenbergMirrorsKey`).

---

## 8. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) or, once fixed, its archive.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `extract_sentences()` regex segfaults on ~100k+ chars without `.!?` (§6) | One `fetch/gutenberg` request can crash `registry_server`, repeatably (**fixed October 24, 2026**) | **Verified** | [TD-309](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-309-gutenberg-sentence-splitting-can-crash-registry_server) |
| `split` isn't flattened and the allow-list permits `..` (§6) | Directories and files written outside the registry data dir via `fetch/huggingface` (**fixed October 24, 2026**) | **Verified** | [TD-310](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-310-huggingface-split-can-write-outside-the-registry-data-directory) |
| Gutenberg pairs are next-sentence non-sequiturs; abbreviations split sentences; unseeded template choice; fixed `language="en"` (§5) | Low-quality training data from every Gutenberg fetch (**fixed November 7, 2026**) | **Verified** | [TD-311](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-311-gutenberg-training-pairs-are-next-sentence-non-sequiturs) |
| `std::system` curl commands from strings; `HF_TOKEN` on the command line; HF-supplied URLs unvalidated (§6) | Injection where callers don't validate; token visible in `ps` (**fixed in code November 7, 2026**) | By inspection | [TD-312](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-312-datafetcher-downloads-through-shell-curl-commands) |
| No curl timeouts; synchronous in the registry handler (§6) | A stalled download holds a registry worker forever (**fixed in code November 7, 2026**) | By inspection | [TD-312](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-312-datafetcher-downloads-through-shell-curl-commands) |
| Partial or interrupted HF downloads cached as complete; concurrent fetches share cache files (§6) | Silently incomplete or corrupt datasets served indefinitely (**fixed in code November 7, 2026**) | By inspection | [TD-313](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-313-huggingface-cache-accepts-partial-downloads-and-races-concurrent-fetches) |
| Gutenberg books come only from `www.gutenberg.org`; no mirror fallback when it's down; an error page would be cached as the book (§3) | Every Gutenberg fetch fails (`502 fetch_failed`) while the main site is unavailable, as on November 18-19, 2026 (**fixed November 19, 2026**) | **Verified** (main site 503, three mirrors 200; tests) | [TD-579](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-579-gutenberg-downloads-dont-fall-back-to-a-mirror-when-gutenbergorg-is-unavailable) |
| `config=default` only; Gutenberg URL patterns limited; `fetch_gutenberg()`/`fetch_huggingface()` never cache; `std::cout` in library code; string-scanner JSON (§3, §5) | Some datasets/books unfetchable; nested fields could be read as a row's own (**fixed November 19, 2026**) | **Verified** (tests) | [TD-314](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-314-datafetcher-source-coverage-and-hygiene-gaps) |
| `ensure_gutenberg_cached()` has no per-book lock (§6) | Two concurrent first fetches of a book download it twice; the cache stays whole | By inspection | — |

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

## Addendum (November 7, 2026): downloads through libcurl, never a shell (TD-312)

Every download (the book in `download_file()`, the `/parquet` listing and each Parquet part in `download_hf_full_dataset()`) now goes through
`adai::http_download()` (`src/HttpDownload.{hpp,cpp}`), which uses libcurl in process. The `curl`/`std::system` descriptions in §2, §4 and
§6 above are historical. A URL must be `https://` on `gutenberg.org`, `huggingface.co`, `hf.co` or a subdomain of one, checked before any
connection; redirects (download CDNs) are followed over HTTPS only. `HF_TOKEN` is an in-process `Authorization` header that libcurl keeps
from a redirect's host. A transfer has a 30 s connect timeout, a stall limit (below 1 KB/s for 60 s) and a one-hour cap, and the body is
written to `<output>.part` and renamed only after a complete 2xx download. `dataset_id` and `split` are percent-encoded into the
`/parquet` query. A build without libcurl can't download and logs that reason. Tests: `httpDownloadTests` (a local httplib server) and
`DataFetcherDownloadTest`.

## Addendum (November 7, 2026): continuation pairs (TD-311)

At the owner's choice, Gutenberg data is honest continuation text. `create_continuation_pairs()` replaced `create_qa_pairs_from_text()` and
`generate_question_from_sentence()`: a pair is two consecutive sentences in and up to the next two out, with no template wording, tagged
`task_type="completion"` and `domain="literature"`. Pairs don't overlap, and the slice cursor is the first sentence not used, so §3 and §5's
template and `rand()` descriptions are historical. `extract_sentences()` no longer ends a sentence at a title (`Mr.`, `Mrs.`, `Dr.`, `St.`,
… 24 in all), a capital initial other than `I`/`A`, a dotted form (`e.g.`, `U.S.`) or a dot between letters or digits. `language` comes from
the book's own `Language:` header line (`gutenberg_language()`), or stays empty when it is mixed, unknown or missing. The `-0.txt` editions
usually have no header, so it is usually empty. `registry_server`'s per-book cache is now `sentences_v2.txt`, re-split from the saved
`book.txt` when only an old `sentences.txt` exists.

## Addendum (November 7, 2026): all-or-nothing HuggingFace cache (TD-313)

`download_hf_full_dataset()` now downloads the listing and every Parquet part into a private `.download-<random>` folder inside the dataset
folder and converts them there. One failed part or conversion abandons the whole attempt. Otherwise the JSONL is renamed over
`full_dataset.jsonl` and `full_dataset.complete` (dataset, split, parts, rows) is written atomically; the folder is always removed. The
`parquet/` and `parquet_info.json` of §4's layout are gone. `ensure_huggingface_cached()` serves a cache only with that marker, and
holds a per-dataset mutex from the check through the download, so concurrent requests in `registry_server` download once. `FetcherConfig`
has `huggingface_api_url`, plus `extra_download_hosts` and `allow_plain_http` for tests against a local server. §4's description of the
cache check and §6's partial/concurrent cache items are historical.

## Addendum (November 19, 2026): configs, the epub URL, a book cache and JSON rows (TD-314)

- **HuggingFace configs.** `fetch_huggingface()` and `ensure_huggingface_cached()` take a `config`, and so do
  `registry_server`'s `fetch/huggingface` (a `"config"` field), `RemoteTransport`/`DatasetRegistry::remote_fetch_huggingface()`,
  `dataset_manager huggingface --config C` and the Android dashboard's fetch dialog. With none named, the whole `/parquet` listing
  is read and `default` used if the dataset has it, else its only config; several are an error that lists them. Only
  `config=default` was ever requested before, so a dataset without one (`Salesforce/wikitext`, for one) failed with "No parquet
  files found". A named config is cached in `<id>_<split>__<config>/` and gets its own registry cursor.
- **Gutenberg's epub path.** `gutenberg_urls()` replaces `get_gutenberg_url()` and the `-0.txt` → `.txt` fallback inside
  `download_file()`, adding `cache/epub/<id>/pg<id>.txt`. Not checked against the live site: `www.gutenberg.org` answered 503 on
  November 19, 2026 (TD-579).
- **Caches for the local fetches.** `fetch_gutenberg()` downloaded the book to `gutenberg_<id>.txt` on every call; it now shares
  `ensure_gutenberg_book()`'s `gutenberg_<id>/book.txt` with `ensure_gutenberg_cached()`. Old `gutenberg_<id>.txt` files are no
  longer read. `fetch_huggingface()` likewise downloaded the whole dataset every call; it goes through `ensure_huggingface_cached()`.
- **JSON rows.** `hf_extract_string()`, `hf_extract_string_array()`, `hf_extract_object()`, `hf_unescape()` and
  `hf_extract_parquet_urls()` are gone; rows and the listing are parsed with the vendored nlohmann/json. `\uXXXX` escapes (surrogate
  pairs included) are decoded, only top-level fields count, and a dialog array must hold only strings. A row that isn't valid JSON
  (including invalid UTF-8) yields nothing. `ParquetReader`'s own JSONL escapes only control characters, so the escape fix matters
  for those and for a cache written by another tool.
- **Logging.** `fetch_huggingface()` and `download_hf_full_dataset()` printed with `std::cout`/`std::cerr`; they log through `Logger`.
- Tests: `RowsAreParsedAsJson`, `DialogTurnsAreAnArrayOfStrings`, `FetchGutenbergUsesTheCachedBook`, `GutenbergUrlsEndWithTheEpubCache`,
  `WithoutAConfigTheDefaultOneIsUsed`, `ADatasetsOnlyConfigIsUsed`, `SeveralConfigsNeedOneNamed`, `FetchHuggingfaceDownloadsOnce`.

## Addendum (November 19, 2026): Gutenberg mirrors (TD-579)

- `www.gutenberg.org` answered 503 for days (November 18-19, 2026), so `bodacious-badger`'s world-model books (run group `bb01`)
  couldn't be queued, while the official mirrors served them. `download_gutenberg_book()` now falls back to
  `FetcherConfig::gutenberg_mirrors` when the main site is unavailable, using the mirrors' digit-path layout (§3).
- It became a member (it reads the configuration); `download_file()` no longer serves Gutenberg. `gutenberg_urls()` takes the site
  URL, and `FetcherConfig::gutenberg_site_url` lets the tests stand in for the main site.
- A download now reaches `book.txt` only when it looks like a Gutenberg text, so a mirror's or the main site's 200 error page is
  never cached (it was before, for `fetch_gutenberg()` too since TD-314).
- `GUTENBERG_MIRRORS` (`ServiceConfig::gutenberg_mirrors`, `config.registry.conf`, environment) defaults to the same list; every
  entry must be `https://` (`ConfigLoader::validate()`), and empty turns the fallback off. `registry_server` reads it at startup
  (not admin-mutable), and `dataset_manager` passes it in local mode.
- Tests: `GutenbergMirrorTest` (9), `MirrorUrlsUseTheDigitPath`, `MirrorListIsTrimmed`, `OnlyGutenbergTextsAreKept`,
  `ConfigTest.GutenbergMirrorsKey`.
