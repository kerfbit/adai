# `DocumentStore` — Source File Reference

- **Files:** [`src/DocumentStore.hpp`](../../../../src/DocumentStore.hpp) (195 lines), [`src/DocumentStore.cpp`](../../../../src/DocumentStore.cpp) (274)
- **Namespace:** global (no namespace)
- **Built into:** `adai_api` (with `ChatbotAPI`, `RAGInference`), linked by `chatbot_api_server`; also compiled directly into `rag_example`
- **Status tag:** `@adai-status: stable`, `@adai-version: 2.0.0`, `@adai-reviewed: 2026-11-12` (MAJOR: for a document longer than `max_seq_length`, its id, `size()` and removal now work on chunks, §2)
- **Tests:** [`tests/documentstore_test.cpp`](../../../../tests/documentstore_test.cpp) (40 tests); exercised indirectly by `raginference_test.cpp` (42)
- **Last traced against the code:** 2026-11-12

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class with
> a small `LLMEncoder` (`max_seq_length = 16`, tokenizer built from a toy corpus).

---

## 1. What this file is

The **retrieval index for RAG**: an in-memory list of entries, `Document {id, text, embedding,
metadata, parent_id, chunk_index, chunk_count}`, plus an `id → index` hash map. Each entry is
embedded once, when added, by running an `LLMEncoder` over its text and **mean-pooling** the output
rows into one `[1, d_model]` vector. `retrieve(query, k)` embeds the query the same way and returns
the top-k entries by cosine similarity (a brute-force scan).

An entry is a whole document, or one **chunk** of a document longer than the encoder's
`max_seq_length` tokens (TD-349, §3).

### Why it matters

With `RAG_ENABLED`, every chat request retrieves through `RAGInference`, which puts the retrieved
entries' texts into the prompt. The quality of these embeddings decides which knowledge the model is
shown.

---

## 2. API

| Member | Behaviour |
|---|---|
| `DocumentStore(shared_ptr<LLMEncoder>)` | Throws on a null encoder |
| `addDocument(id, text, metadata = {})` | Throws on empty text, or when `id` is already an entry or a chunked document's id, or one of its chunk ids already exists. A document of at most `max_seq_length` tokens is one entry under `id`; a longer one is split into chunks (§3), entries `id#1` … `id#N`. Each entry gets the metadata, `parent_id = id`, its `chunk_index` and `chunk_count`, and an embedding from `encoder->encode(text)` → `getSentenceEmbedding()`. Every entry is embedded before any is stored, so a failure leaves the store unchanged |
| `removeDocument(id)` | Removes every entry whose `parent_id` is `id` (all chunks of a document, or its one entry); failing that, the one entry with that id (a single chunk). Each removal swaps with the last entry, then pops (O(1)), and fixes up the moved entry's index. **Invalidates** or silently retargets earlier `const Document*` (documented). Returns whether anything was removed |
| `retrieve(query, k = 3)` | Empty store → `{}`. `k <= 0` → throws. Otherwise embeds the query, scores every entry, `partial_sort`s the top `min(k, n)` by descending similarity and returns `(score, const Document*)` pairs: a chunked document's hits are its chunks. The pointers are valid until the next mutation |
| `getDocument(id)`, `size()`, `empty()`, `clear()`, `getAllDocumentIds()` | Plain accessors over **entries**: `getDocument("doc")` of a chunked document is `nullptr` (its entries are `doc#1` …), and `size()` counts chunks |
| `chunkRanges(text, window)` (private) | The chunks' `[begin, end)` byte ranges (§3) |
| `getSentenceEmbedding(m)` (private) | Mean over rows; throws if empty |
| `cosineSimilarity(a, b)` (private) | Over row 0; returns 0 for a near-zero norm; throws if the column counts differ |

Every method takes the store's mutex (TD-350), so `retrieve()` encodes and scores a query under it
(`encode()` writes the encoder's layer caches) and documents can be added while other threads
retrieve.

`metadata` is copied to every chunk but nothing reads it: there is no filtering API (TD-351).

---

## 3. Chunking (TD-349)

`LLMEncoder::encode()` embeds at most `max_seq_length` tokens (since TD-365, an over-long input's
**last** ones). Before November 12, 2026 every document was one entry, so a long document was
represented by a single window of it and the rest couldn't be retrieved. **Verified** with
`max_seq_length = 16`: two documents with the same long ending got identical embeddings and tied on
every query, including one quoting the other's opening exactly.

`addDocument()` now compares `encoder->count_tokens(text)` (the untruncated length) with
`max_seq_length`, the *window*, and splits a longer document:

1. The text is cut into **pieces**: a whitespace run and the word after it. The tokenizer's
   pre-tokens (GPT-2 style: a word, number or punctuation run with at most one leading space, or a
   whitespace run) don't cross these boundaries for spaces, tabs and newlines, so a chunk doesn't
   split a token. Each piece's token count is taken on its own.
2. A chunk takes pieces while their counts sum to at most the window (at least one piece). Its real
   count, with the leading whitespace trimmed, is then checked, and pieces dropped from its end until
   it fits. With this tokenizer the piece counts add up exactly, so the check is a safety net.
3. The next chunk starts about a quarter of a window back, sharing at least the previous chunk's last
   piece (so a sentence cut at a boundary appears whole in one chunk) but always starting past the
   previous chunk's start.

A single piece longer than the window (a very long unbroken string) becomes a chunk of its own,
which `encode()` truncates. A document that comes out as one chunk stays one entry under its own id.
Overlapping chunks of one document can both be retrieved, so the prompt may repeat their shared text.

---

## 4. How the server uses it

`ChatbotAPIServer.cpp`, when RAG is enabled with a trained model loaded:
1. Takes a copy of the model's trained encoder and its tokenizer (`EncoderDecoderModel::copy_encoder()`,
   TD-247).
2. Wraps it in a `DocumentStore`.
3. Adds **each top-level `.txt` file** under `RAG_DOCS_PATH` as one document (ID = file stem), which
   the store chunks as needed, and logs `Indexed N documents (M entries)`. There's no recursion, and
   the extension match is exact (`.txt`).

Any exception during indexing disables RAG, via the outer `catch`.

By inspection:
- **Re-embeds everything at every server start.** There's no persistence, and the brute-force scan is
  O(N · d_model) per query, N counting chunks (fine at small scale).
- **NaN embeddings break the sort.** A NaN similarity violates `partial_sort`'s strict weak ordering.
  That's only reachable with NaN weights (e.g. a corrupt load, which TD-423's checked loaders now
  refuse).
- **Special tokens.** TD-351 noted that mean pooling included BOS/EOS rows. Since TD-364,
  `encode()` adds no special tokens by default and this store never asks for them, so that part of
  TD-351 no longer applies.

---

## 5. Tests

`documentstore_test.cpp` has 40 tests:
- construction;
- add, remove and duplicate IDs;
- the swap-remove index fix-up;
- retrieve ordering, `k` bounds and the empty store;
- clear and IDs;
- concurrent retrieval, and adding while retrieving (TD-350);
- chunking (TD-349, `DocumentStoreChunkTest`, with a 16- or 20-token window):
  - a 60-word document becomes overlapping chunks that each fit, start and end it and carry its
    metadata;
  - with words longer than a quarter of the window, consecutive chunks still share their boundary
    word;
  - an opening outside the encoder's window is retrievable;
  - removing a chunked document removes every chunk;
  - its id and chunk ids can't be added again;
  - a document that fits keeps its own id.

Removing the chunking, the overlap, the at-least-one-piece overlap, removal by parent or the parent
duplicate check each fails a test. Dropping the exact-fit check doesn't: with this tokenizer it never
changes a chunk (§3).

`raginference_test.cpp` (42) covers the pipeline around it.

**Not covered:** the server's directory indexing.

---

## 6. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `metadata` unused (no filtering); no persistence (re-embed at startup); NaN scores break the sort (§2, §4) | Dead field; startup cost; undefined ordering on corrupt weights | By inspection | [TD-351](../../guides/TECHNICAL_DEBT.md#td-351-documentstore-embedding-and-housekeeping-gaps) |
| Overlapping chunks of one document can be retrieved together (§3) | The prompt may repeat a quarter window of text | By design | — |

---

## 7. History

- **TD-349 (November 12, 2026):** long documents are chunked to the encoder's window (§3). Before,
  only one window of each file affected retrieval.
- **TD-350 (October 27, 2026):** the store's mutex. `retrieve()` used to run the stateful encoder
  unlocked, safe only because `ChatbotAPI` held `config_mutex_` (TD-230); the old code under TSan
  reported 74 data races and 4 heap-use-after-frees in the two concurrency tests, the fixed code none.
  `LLMEncoder::encode()` logs its token count at debug level instead of printing to stdout.
- **TD-247 (October 26, 2026):** the server's RAG encoder is a copy of the trained model's encoder;
  it used to have random weights.

This file replaced `docs/development/api/memory/DocumentStore.md` (238 lines). That doc listed
"metadata filtering" as an integration pattern (no filtering API exists), called the store safe
without concurrent modification (before TD-350 even `retrieve()` wasn't), and didn't mention
truncation or one-document-per-file indexing.
