# `IntegratedInferenceEngine` — Source File Reference (Retired)

> **Retired October 27, 2026 (owner decision on TD-417–TD-420).** `src/IntegratedInferenceEngine.hpp`,
> `tests/integratedinferenceengine_test.cpp`, `benchmarks/IntegratedBenchmark.cpp` (`integrated_benchmark`),
> `ChatbotAPI::enable_integrated_inference()` and `chatbot_api_server --integrated-inference` were removed; the flag now exits 1
> with a message saying so. The engine batched nothing through the model, answered differently from `/chat`, could take the
> server down, and ran the model without its mutex. TD-417–TD-420 are closed in
> [TECHNICAL_DEBT_RESOLVED.md](../../archive/TECHNICAL_DEBT_RESOLVED.md); real batching is planned in
> [real_batching_v2_plan.md](../../../proposals/real_batching_v2_plan.md). Last version:
> `git show 0f7cf0b:src/IntegratedInferenceEngine.hpp`. The sections below describe the code as it was.

- **File:** `src/IntegratedInferenceEngine.hpp` (removed), header-only, 780 lines
- **Namespace:** global (`IntegratedInferenceConfig`, `IntegratedInferenceStats`, `IntegratedRequest`, `IntegratedBatch`, `IntegratedEncoderOutput`, `IntegratedInferenceEngine`, alias `StandardIntegratedEngine`)
- **Built into:** header-only; used by `ChatbotAPI` (`enable_integrated_inference()`), enabled by `chatbot_api_server --integrated-inference`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.2`, `@adai-reviewed: 2026-09-12`
- **Tests:** `tests/integratedinferenceengine_test.cpp` (removed; 50 tests)
- **Related:** [BatchedInferenceEngine.md](BatchedInferenceEngine.md) (its `ThreadSafeQueue`; the TD-211–TD-214 batching findings apply here too), `PipelineInferenceEngine.hpp`, [PARALLEL_OPTIMIZATIONS_QUICK_REFERENCE.md](../../guides/quick-reference/PARALLEL_OPTIMIZATIONS_QUICK_REFERENCE.md)
- **Last traced against the code:** 2026-10-13

> Code-traced reference. Behaviour marked **verified** was confirmed by running the engine against a
> small real `EncoderDecoderModel` in a scratch program.

---

## 1. What this file is

A **three-thread request pipeline** for `/chat` generation:

```text
submit(text, max_length, strategy) ─→ request_queue_ ─→ [batcher thread] groups ≤ max_batch_size or
  batch_timeout_ms, tokenizes each ─→ batch_queue_ ─→ [encoder thread] encode_with_mask() per request
  ─→ encoder_to_decoder_queue_ ─→ [decoder thread] greedy decode per request ─→ promise.set_value()
```

The header's banner promises "all Priority 1–5 optimizations … Target: 50–100× improvement", and
the quick reference lists it as "Production Ready, 5.5×+". In the code, **nothing is batched through the
model**: the encoder thread runs one `encode_with_mask()` per request, and the decoder thread runs one
autoregressive greedy loop per request, recomputing the full decoder each step with no KV cache. The
only real concurrency is that one request's decoding overlaps the next request's encoding. "Parallel
attention" and OpenMP come from the layers themselves, not from this engine. This is the same gap
TD-211 records for `BatchedInferenceEngine`, which real batching (v2 plan, blocked on TD-171) would
address.

---

## 2. API and configuration

| Member | Behaviour |
|---|---|
| `IntegratedInferenceEngine(encoder, decoder, lm_head, tokenizer, config)` | Stores **raw pointers** to the model's components, calls `omp_set_num_threads(num_threads)` (a **process-wide** setting) when configured, and starts the three threads |
| `submit(text, max_length = -1, strategy = "")` | Queues a request (`max_length` -1 → `default_max_length`); returns a `std::future<std::string>`. `request_queue_` is bounded by `max_queue_size`, and `push()` on a full queue **blocks the caller** |
| `shutdown()` / destructor | Shuts the three queues and joins the threads. **Queued and in-flight requests are dropped with their promises unfulfilled**: verified, three requests queued at destruction all threw `std::future_error` (broken promise) |
| `get_stats()` / `reset_stats()` | See §4 |

| Config field | Used? |
|---|---|
| `max_batch_size`, `batch_timeout_ms` | Yes (batch grouping) |
| `encoder_queue_size`, `decoder_queue_size`, `encoder_timeout_ms`, `decoder_timeout_ms`, `max_queue_size` | Yes |
| `num_threads` (with `use_openmp`) | Only to call `omp_set_num_threads()`; `use_openmp = false` doesn't disable anything |
| `default_max_length`, `generation_strategy` | Defaults for `submit()`; **the strategy is never consulted** (§3) |
| `max_tokens_per_batch`, `enable_pipeline`, `parallel_attention`, `enable_stats` | **Never read** |

`prepare_encoder_inputs()` (a padded batch builder) is never called.

---

## 3. Generation behaviour (verified)

`generate_from_encoder_output()` starts from `[BOS]`, runs
`decoder->forward_with_encoder(all generated tokens, encoder_output)` each step, takes the last row
through the LM head, picks the **argmax**, and stops at `EOS` or `max_length`.

- **The strategy is ignored.** `submit(..., "top_k")` returned identical greedy output twice
  (verified). `ChatbotAPI` passes `config.strategy` and its comment admits it's "never consulted"; the
  request's temperature, top-k/p, beam width and repetition settings never reach the engine (TD-214's
  pattern).
- **The encoder input differs from training.** The batcher tokenizes with `tokenizer_->encode(text)`,
  whose default adds `BOS`/`EOS`; `EncoderDecoderModel` trains its encoder without special tokens
  (TD-364's mismatch).
- **It's a different decoding loop from the model's own:** no invalid-vocab masking, no repetition
  handling, no hippocampal/world-model inputs, and `max_length` isn't capped at `max_seq_length`
  (decoder positions past it lose positional encoding, TD-347).

So `--integrated-inference` answers differ from the normal `/chat` path. On a small untrained model the
three test prompts produced different text from `generate_response_with_strategy(..., "greedy", ...)` in
**3 of 3** cases (verified; the text itself is meaningless, but the paths should agree).

---

## 4. Robustness and stats

- **Worker exceptions terminate the process.** The batcher catches tokenizer errors per request, and the
  decoder guards its token-count `encode()`. But `encode_with_mask()`, `forward_with_encoder()` and
  `lm_head_->forward()` run with no `try` in the encoder and decoder threads. Any exception (for example
  a token id outside the model's vocab when the tokenizer and model disagree) escapes the
  `std::thread` and calls `std::terminate()`, taking `chatbot_api_server` down. The other requests'
  promises are never fulfilled either.
- **No model lock.** The engine drives the model's encoder, decoder and head from its own threads
  without `EncoderDecoderModel::model_mutex_`. `/chat/batch` bypasses the dispatcher (TD-234) and calls
  the model under that mutex, so the two can run the **same layer objects concurrently**, which race on
  their per-instance forward caches (the TD-156 class).
- **Stats are partly fiction:** `avg_openmp_speedup`, `avg_attention_speedup`, `cumulative_speedup`,
  `avg_queue_depth` and `requests_dropped` are never set. `avg_batch_size` divides completed requests by
  batches *formed* (including still-in-flight ones), and `pipeline_efficiency` is a formula
  (`(enc + dec) / max(enc, dec)`) rather than a measurement. Unlike `BatchedInferenceEngine`'s
  (TD-213), these aren't exposed by any endpoint.

---

## 5. Tests

`integratedinferenceengine_test.cpp` (50 tests) covers:
- construction, single and concurrent submission;
- batching by size and timeout (including the TD-073 deadline fix);
- empty-input rejection and the TD-162 stats-ordering fix;
- statistics fields and reset, and shutdown idempotence.

**Not covered:**
- agreement with the model's own generation;
- strategy handling;
- exceptions inside the encoder/decoder stages;
- pending requests at shutdown;
- concurrent use alongside other model paths.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| No model batching despite the "27.80×/50–100×/Production Ready" claims; per-request encode and decode, decoding without KV cache (§1) | Throughput claims don't hold; latency per request is worse than the plain path | By inspection | [TD-420](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-420-integratedinferenceengine-claims-dead-config-and-fictional-stats) |
| Generation always greedy, ignores strategy and sampling settings; special tokens added to encoder input; no masking/repetition/memory; `max_length` unbounded by `max_seq_length` (§3) | `--integrated-inference` gives different answers from normal `/chat` | **Verified** (strategy ignored; 3/3 outputs differ) | [TD-417](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-417---integrated-inference-answers-differ-from-normal-chat) |
| Encoder/decoder-stage exceptions escape worker threads → `std::terminate()`; shutdown drops queued requests (broken promises) (§2, §4) | One bad request can crash `chatbot_api_server`; requests in flight at shutdown error out | **Verified** (shutdown); by inspection (terminate) | [TD-418](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-418-integratedinferenceengine-worker-exceptions-terminate-the-server-shutdown-drops-requests) |
| Uses the model's components without `model_mutex_` while `/chat/batch` uses them under it (§4) | Concurrent forward passes race on shared layer caches | By inspection | [TD-419](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-419-integratedinferenceengine-uses-the-model-without-its-mutex) |
| Dead config fields and stats; process-wide `omp_set_num_threads()`; blocking `submit()` when the queue is full; unused `prepare_encoder_inputs()` (§2, §4) | Misleading configuration and statistics | By inspection | [TD-420](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-420-integratedinferenceengine-claims-dead-config-and-fictional-stats) |
| No real batching (TD-171), batched-mode strategy/stats problems | Already tracked for `BatchedInferenceEngine` | — | TD-211, TD-213, TD-214 |
