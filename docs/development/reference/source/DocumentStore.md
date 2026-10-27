# `DocumentStore` — Source File Reference

- **Files:** [`src/DocumentStore.hpp`](../../../../src/DocumentStore.hpp), [`src/DocumentStore.cpp`](../../../../src/DocumentStore.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_api` (with `ChatbotAPI`, `RAGInference`), linked by `chatbot_api_server`; also compiled directly into `rag_example`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/documentstore_test.cpp`](../../../../tests/documentstore_test.cpp) (32 tests); exercised indirectly by `raginference_test.cpp` (37)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class with
> a small `LLMEncoder` (`max_seq_length = 16`, tokenizer built from a toy corpus).

---

## 1. What this file is

The **retrieval index for RAG**: an in-memory list of `Document {id, text, embedding, metadata}`
plus an `id → index` hash map. Each document is embedded once, when added, by running an
`LLMEncoder` over its text and **mean-pooling** the output rows into one `[1, d_model]` vector.
`retrieve(query, k)` embeds the query the same way and returns the top-k documents by cosine
similarity (a brute-force scan).

### Why it matters

With `RAG_ENABLED`, every chat request goes through `RAGInference::generate()`, which calls
`retrieve()` and puts the retrieved document texts into the prompt. The quality of these
embeddings decides which knowledge the model is shown. Today that quality is limited by an
untrained encoder (TD-247) and by truncation (§3).

---

## 2. API

| Member | Behaviour |
|---|---|
| `DocumentStore(shared_ptr<LLMEncoder>)` | Throws on a null encoder |
| `addDocument(id, text, metadata = {})` | Throws on a duplicate ID or empty text. Embeds with `encoder->encode(text)` → `getSentenceEmbedding()` and appends |
| `removeDocument(id)` | Swap with the last document, then pop (O(1)), and fix up the moved document's index. **Invalidates** or silently retargets earlier `const Document*` (documented) |
| `retrieve(query, k = 3)` | Empty store → `{}`. `k <= 0` → throws. Otherwise embeds the query, scores every document, `partial_sort`s the top `min(k, n)` by descending similarity and returns `(score, const Document*)` pairs. The pointers are valid until the next mutation |
| `getDocument(id)`, `size()`, `empty()`, `clear()`, `getAllDocumentIds()` | Plain accessors |
| `getSentenceEmbedding(m)` (private) | Mean over rows; throws if empty |
| `cosineSimilarity(a, b)` (private) | Over row 0; returns 0 for a near-zero norm; throws if the column counts differ |

`metadata` is stored but nothing reads it: there is no filtering API, despite the old doc's
"metadata filtering" pattern.

---

## 3. How the server uses it (and what that implies)

`ChatbotAPIServer.cpp`, when RAG is enabled:
1. Builds a **fresh `LLMEncoder` with random weights** (TD-247) and loads only the tokenizer vocab.
2. Wraps it in a `DocumentStore`.
3. Adds **each top-level `.txt` file** under `RAG_DOCS_PATH` as **one document** (ID = file stem).
   There's no chunking or recursion, and the extension match is exact (`.txt`).

Any exception during indexing disables RAG entirely, via the outer `catch`.

> **Only a document's first `max_seq_length` tokens count (verified).** `LLMEncoder::encode()`
> truncates its input to `max_seq_length` tokens before embedding. Since every file is one
> document, a long document is represented by its **opening** only. With `max_seq_length = 16`,
> two documents sharing the same first 16 tokens and differing after them got **bit-identical
> embeddings** (|Δ| = 0.000000). They tied on every query, including one matching the tail of one
> of them exactly. At the default `MAX_SEQ_LENGTH = 512`, anything past roughly the first page of
> each file is invisible to retrieval, yet the **whole** file text is what `RAGInference` puts in
> the prompt (subject to its own context limit). Splitting files into overlapping chunks at index
> time would fix both.

By inspection:
- **Mean pooling includes special tokens.** `encode(text, true)` adds them, so every embedding
  shares the same special-token rows, which pulls cosine scores toward each other (the sample
  above scored 0.89 between unrelated texts, with untrained weights).
- **Not thread-safe, and reads aren't read-only.** `retrieve()` calls `encoder->encode()`. That
  mutates the encoder's caches (`requires_grad` defaults to true) and prints
  `Encoding N tokens...` to stdout on every query. Today RAG generation is serialized under
  `ChatbotAPI::config_mutex_` (TD-230), so there's no race. But **fixing TD-230 by releasing that
  lock would expose concurrent `retrieve()` calls to data races inside the encoder**: the store or
  the encoder needs its own lock, or inference mode, first.
- **Re-embeds everything at every server start.** There's no persistence, and the brute-force scan
  is O(N · d_model) per query (fine at small scale).
- **NaN embeddings break the sort.** A NaN similarity violates `partial_sort`'s strict weak ordering.
  That's only reachable with NaN weights (e.g. TD-298-style corrupt loads).

---

## 4. Tests

`documentstore_test.cpp` has 32 tests:
- construction;
- add, remove and duplicate IDs;
- the swap-remove index fix-up;
- retrieve ordering, `k` bounds and the empty store;
- clear and IDs.

`raginference_test.cpp` (37) covers the pipeline around it.

**Not covered:**
- long documents (truncation);
- concurrency;
- the server's directory indexing.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Whole files indexed as single documents, and `encode()` truncates to `max_seq_length` tokens, so only each file's opening affects retrieval (§3) | Content past the first ~512 tokens is unretrievable; same-opening files tie | **Verified** | [TD-349](../../guides/TECHNICAL_DEBT.md#td-349-rag-indexes-whole-files-but-embeds-only-their-first-max_seq_length-tokens) |
| `retrieve()` mutates encoder state and prints to stdout; no locking, currently masked by TD-230's mutex (§3) | Fixing TD-230 naively introduces data races in RAG retrieval | By inspection | [TD-350](../../guides/TECHNICAL_DEBT.md#td-350-documentstoreretrieve-isnt-safe-to-call-concurrently) |
| Mean pooling includes special tokens; `metadata` unused (no filtering); no persistence (re-embed at startup); NaN scores break the sort (§2–3) | Compressed similarity range; dead field; startup cost; undefined ordering on corrupt weights | By inspection | [TD-351](../../guides/TECHNICAL_DEBT.md#td-351-documentstore-embedding-and-housekeeping-gaps) |
| RAG encoder has random weights | Already tracked | — | TD-247 |
| RAG generation holds `config_mutex_` | Already tracked | — | TD-230 |

---

## 6. Merge notes

This file replaces `docs/development/api/memory/DocumentStore.md` (238 lines). Where they disagree,
this file wins:

- **Metadata filtering.** The old doc listed "metadata filtering" as an integration pattern; no
  filtering API exists.
- **Thread safety.** It described thread safety as "no concurrent modification". In fact even
  `retrieve()` isn't safe concurrently, because it runs the stateful encoder.
- **Truncation.** It had no mention of `max_seq_length` truncation or the server's
  one-document-per-file indexing.

**Kept from the old doc:** the data structures, operation semantics, O(N) retrieval
characteristics, and the in-memory/no-ANN limitations.

## Addendum (October 27, 2026): thread safety (TD-350)

Every `DocumentStore` method takes the store's mutex, so `retrieve()` encodes and scores a query under it and `addDocument()` can run
while other threads retrieve. `LLMEncoder::encode()` no longer prints `Encoding N tokens...` to stdout; it logs at debug level. This
let TD-230 release `config_mutex_` during RAG generation. Returned `Document` pointers are still valid only until the next
add/remove/clear. Tests: `ConcurrentRetrieveMatchesSerialResults`, `AddDocumentWhileRetrieving` (old code under TSan: 74 data races,
4 heap-use-after-frees; fixed code: none).
