# `RAGInference` — Source File Reference

- **Files:** [`src/RAGInference.hpp`](../../../../src/RAGInference.hpp), [`src/RAGInference.cpp`](../../../../src/RAGInference.cpp) (366 lines together)
- **Namespace:** global (`RAGInference`, `RAGInference::RAGConfig`)
- **Built into:** `adai_api` (with `ChatbotAPI` and [`DocumentStore`](DocumentStore.md)), linked by `chatbot_api_server`; also used by `rag_example`
- **Used by:** `chatbot_api_server` when `RAG_ENABLED=true` (built in `ChatbotAPIServer.cpp`, handed to `ChatbotAPI::enableRAG()`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0` (hpp) / `1.0.1` (cpp), `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/raginference_test.cpp`](../../../../tests/raginference_test.cpp) (37 tests)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real classes.

---

## 1. What this file is

Retrieval-augmented generation: retrieve the top documents for a query from a [`DocumentStore`](DocumentStore.md), paste them into the prompt
ahead of the query, and generate with the [`EncoderDecoderModel`](EncoderDecoderModel.md). It holds no state of its own beyond a config and two
shared pointers.

### `RAGConfig`

| Field | Default | Server key (`config.chatbot.conf`) |
|---|---|---|
| `num_retrieved_docs` | 3 | `RAG_NUM_DOCS` (3) |
| `retrieval_threshold` | 0.0 (no filter) | `RAG_THRESHOLD` (0.0) |
| `include_scores` | false | — |
| `context_separator` / `context_prefix` / `query_prefix` | `"\n\n"` / `"Context:\n"` / `"\n\nQuestion: "` | — |
| `max_context_length` | 512 ("tokens", see §2) | `RAG_MAX_CONTEXT_LENGTH` (512) |
| `gen_config` (`TextGenerator::GenerationConfig`) | `max_length 100`, `temperature 1.0`, `top_k 0`, `top_p 1.0`, `num_beams 1` | `MAX_GEN_LENGTH`, `TEMPERATURE`, `TOP_P`, `TOP_K` copied at startup; `num_beams` stays 1 |

### Members

| Member | Behaviour |
|---|---|
| Constructor `(model, doc_store, config = {})` | Throws `invalid_argument` on a null model or store |
| `generate(query)` | `generateWithRetrieval()` with a discarded document list |
| `generateWithRetrieval(query, out docs)` | (1) `doc_store->retrieve(query, num_retrieved_docs)`; (2) drop documents scoring below `retrieval_threshold` when it is `> 0`; (3) `formatContext()`; (4) `truncateContext(context, max_context_length)`; (5) `buildAugmentedPrompt()`; (6) `model->generate_response_with_strategy(prompt, max_length, "", temperature, top_k, top_p, num_beams)` (the TD-101 fix: before it, `gen_config` never reached the model) |
| `formatContext(docs)` | Concatenates document texts with `context_separator`, optionally prefixed by `[Doc i (similarity: s)]: `. Empty list → `""` |
| `buildAugmentedPrompt(query, context)` | `context_prefix + context` (only if non-empty) `+ query_prefix + query` |
| `truncateContext(context, max_tokens)` (private, static) | Keeps the first `max_tokens × 4` **characters**, replacing the last 3 with `...`; `max_tokens ≤ 0` → `""`; a 1–3 character budget hard-truncates (an earlier underflow fix, tested) |
| `addDocument`, `removeDocument`, `getDocument`, `getNumDocuments` | Forward to the store. Returned `const Document*` pointers follow `DocumentStore`'s rules: invalidated or silently retargeted by the next mutation |
| `retrieveOnly(query, k = -1)` | Retrieval without generation; `k ≤ 0` uses `num_retrieved_docs` |
| `getConfig()`, `setConfig()` | Unsynchronized; `setConfig()` has no production caller |

### How the server uses it

`ChatbotAPI::generate_response()` checks `rag_engine_` first (under `config_mutex_`, TD-230) and, when RAG is enabled, returns
`rag_engine_->generate(input)` immediately. Everything after that check (speculative decoding, batched/pipeline/integrated engines, the GPU path
and the request's own `GenerationConfig`) is skipped. `input` is the message for `/chat`, and for `/chat/session` the **whole formatted conversation**
(`ConversationContext::format_for_model()`: `"User: …\nAssistant: …\n…"`).

---

## 2. Problems

### The context budget alone fills the model's input (verified arithmetic; by inspection)

`max_context_length` is converted to characters at 4 per token, and only the context is budgeted: the prefixes and the query are added outside it.
The shipped values are `RAG_MAX_CONTEXT_LENGTH=512` and `MAX_SEQ_LENGTH=512`, so even if the estimate were exact, the context alone would use every
position the model has, and the **question would land after position 512**. The encoder input isn't truncated (TD-365), so positions past
`max_seq_length` get no positional encoding: the part of the prompt that says what to answer is the part the model can't place.

The 4-characters estimate is also optimistic for this codebase's tokenizer. A probe that built a vocabulary with this repo's `BPETokenizer` from the
project's own guides and encoded a 2,048-character context of three of those documents got **1,150 tokens (1.8 characters per token)**, and
**1,178 tokens** for the full prompt, 2.3× `max_seq_length`. (That vocabulary was small, 402 entries, so a production vocabulary will compress
better, but no realistic English BPE averages much above 4.) A token-aware budget for the whole prompt, sized from `max_seq_length` minus the query
and the prefixes, is the fix. A character cut can also split a UTF-8 sequence before the `...`.

### `/chat/session` retrieves with the start of the conversation (by inspection)

The query passed to `retrieve()` is the whole formatted conversation. The retrieval encoder (`LLMEncoder::encode()`) keeps only the **first**
`max_seq_length` tokens, so once a conversation is longer than that, retrieval is driven by the oldest turns and the latest question doesn't take
part at all. The same text then also becomes the "question" in the prompt, after the context. Retrieving with the latest user message (and
budgeting the history separately) is the usual approach.

### RAG ignores the server's generation settings (by inspection)

- **Strategy.** It always passes `strategy = ""`, which takes `generate_response_with_strategy()`'s default branch (combined temperature/top-k/top-p
  sampling). `STRATEGY=nucleus` (the shipped value) and `BEAM_WIDTH` have no effect while RAG is on. (Conversely, RAG is one of the few paths where
  `TOP_K` is applied; TD-244.) `num_beams` stays at its default of 1, so TD-261's forced beam search isn't reached from the server; a caller that
  sets `gen_config.num_beams > 1` would get beam search regardless of anything else.
- **Per-request and reloaded values.** `gen_config` is copied once at startup, so `SIGHUP` reloads of `TEMPERATURE`/`TOP_P`/`TOP_K`/`MAX_GEN_LENGTH`
  don't reach it (the reload does update `default_config_`, which RAG bypasses), and neither do any per-request values (TD-254).
- **GPU.** `generate_response_with_strategy()` is CPU-only. On a GPU server, enabling RAG moves every request off the TD-033 GPU decode path onto
  CPU generation.

### Silent or failing configurations (by inspection)

- **`RAG_NUM_DOCS ≤ 0`** isn't validated; `DocumentStore::retrieve()` throws `"k must be positive"` on every request once any document is indexed,
  so every chat request fails. (An empty store returns before the check.)
- **`RAG_THRESHOLD`** can remove every document without any indication: the prompt then has no `Context:` section and nothing is logged, so a
  too-high threshold looks like RAG working but unhelpful.
- **No retrieval logging.** Nothing records which documents were used or their scores; `generateWithRetrieval()`'s output list exists but the server
  calls `generate()`.

### Already tracked elsewhere

| Item | Tracked as |
|---|---|
| Retrieval encoder has random weights (never loaded) | TD-247 |
| Each `.txt` file is one document; only its first `max_seq_length` tokens are embedded | TD-349 |
| `DocumentStore::retrieve()` isn't safe to call concurrently | TD-350 |
| Generation runs while holding `ChatbotAPI::config_mutex_` | TD-230 |
| Encoder input not truncated to `max_seq_length` | TD-365 |

---

## 3. Tests

`raginference_test.cpp` (37 tests): construction (valid, null model, null store, custom config), defaults, add/remove/get documents and counts,
retrieval (default and custom `k`, `k` above the store size, empty store), config get/set and its effect on retrieval, generation (with documents,
empty store, `gen_config` reaching the generator, the retrieval output list, threshold filtering, response type), add/remove cycles, retrieval
consistency and after removal, empty queries, and five `truncateContext()` edge cases (zero/negative budget, never longer than the input, ellipsis,
short input).

**Not covered:** prompt length against `max_seq_length`, session-style multi-turn queries, `num_retrieved_docs ≤ 0`, a threshold that removes
everything.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Context budget = `max_seq_length` at the shipped values; query and prefixes outside the budget; 4-chars-per-token estimate (§2) | The question falls past the model's positional range; prompt ≈ 2.3× `max_seq_length` in the probe | **Verified** (token counts); by inspection | [TD-484](../../guides/TECHNICAL_DEBT.md#td-484-rag-prompt-budget-lets-the-question-fall-past-max_seq_length) |
| `/chat/session` retrieves with the conversation's first `max_seq_length` tokens and uses the whole conversation as the question (§2) | Retrieval ignores the latest question in long conversations | By inspection | [TD-485](../../guides/TECHNICAL_DEBT.md#td-485-rag-session-retrieval-uses-the-start-of-the-conversation) |
| `strategy = ""` always; settings copied once at startup; CPU-only generation (§2) | `STRATEGY`/`BEAM_WIDTH`, reloads and the GPU path are lost while RAG is on | By inspection | [TD-486](../../guides/TECHNICAL_DEBT.md#td-486-rag-path-bypasses-strategy-reloaded-settings-and-gpu-generation) |
| `RAG_NUM_DOCS ≤ 0` fails every request; threshold can silently empty the context; no retrieval logging (§2) | Misconfiguration is hard to see | By inspection | [TD-487](../../guides/TECHNICAL_DEBT.md#td-487-raginference-misconfiguration-and-observability-gaps) |
| Random retrieval encoder, whole-file documents, concurrency, `config_mutex_`, untruncated input | Already tracked | — | TD-247, TD-349, TD-350, TD-230, TD-365 |

---

## 5. Merge notes

This file replaces `docs/development/api/memory/RAGInference.md` (234 lines). Where they disagree, this file wins:
- **"Production-ready engine" / "context truncation prevents excessive prompt size".** The shipped budget alone equals `max_seq_length`, and the
  query is outside it (§2).
- **`truncateContext()` as a `const` member.** It is a private `static` function.
- **"Not thread-safe".** Still true for `RAGInference` itself; in the server, calls are serialized by `ChatbotAPI::config_mutex_` (TD-230) and
  generation by `model_mutex_`.

Kept: the config fields, the members and the retrieve → format → truncate → prompt → generate pipeline. Dropped: the generic integration patterns,
future enhancements, testing recommendations and debugging tips.

## Addendum (October 26, 2026): trained retrieval encoder (TD-247)

`chatbot_api_server` now gives the `DocumentStore` a copy of the trained encoder (`EncoderDecoderModel::copy_encoder()`), so documents
and queries are embedded with learned weights and the model's tokenizer. RAG isn't enabled without a trained model.

## Addendum (October 27, 2026): token-budgeted prompt (TD-484)

`truncateContext()` is replaced by `buildBudgetedPrompt(query, context)`, used by `generateWithRetrieval()` for steps 3–4. It counts
tokens with the model's tokenizer (`countTokens()`), keeps `query_prefix + query` whole, and binary-searches the longest prefix of
the context, cut at a UTF-8 boundary (`utf8Boundary()`) with `"..."`, such that the context is at most `max_context_length` tokens and
the whole prompt at most the model's `max_seq_length`. Documents are in rank order, so the lowest-ranked go first. No room, or a
question that alone reaches `max_seq_length`: the question alone. `chatbot_api_server` caps `RAG_MAX_CONTEXT_LENGTH` to
`max_seq_length`. Tests: `RAGInferencePromptBudgetTest`.
