# `PipelineInferenceEngine` — Source File Reference

- **File:** [`src/PipelineInferenceEngine.hpp`](../../../../src/PipelineInferenceEngine.hpp) (header-only, 551 lines)
- **Namespace:** global (`ThreadSafeQueue<T>`, `PipelineRequest`, `EncoderOutput`, `PipelineConfig`, `PipelineStats`, `PipelineInferenceEngine<…>`, `StandardPipelineEngine`)
- **Used by:** `ChatbotAPI` when `chatbot_api_server` runs with `--pipeline-inference` (`enable_pipeline_inference()`); `IntegratedInferenceEngine.hpp` includes it for `ThreadSafeQueue`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-12`
- **Tests:** [`tests/pipelineinferenceengine_test.cpp`](../../../../tests/pipelineinferenceengine_test.cpp) (61 tests, against mock encoder/decoder/head/tokenizer); `benchmarks/PipelineBenchmark.cpp`
- **Sibling:** [IntegratedInferenceEngine.md](IntegratedInferenceEngine.md) (TD-417–TD-420), which shares most of these patterns
- **Last traced against the code:** 2026-10-16

> Code-traced reference. Behaviour marked **verified** was confirmed with scratch programs: the real engine over mock stages, and the real
> model's encoder/decoder for timing.

---

## 1. What this file is

A two-stage producer/consumer pipeline: one **encoder thread** and one **decoder thread** connected by bounded queues.

```text
submit_batch(texts, max_length) → input_queue_ ──encoder thread──► encoder_->encode(text) per input
                                                                    │
                       encoder_to_decoder_queue_ ◄──────────────────┘
                                │ decoder thread: per input, greedy loop up to max_length:
                                │   decoder_->forward_with_encoder(all tokens so far, enc)   (no KV cache)
                                │   lm_head_->forward(last row) → argmax → stop at EOS
                                └─► tokenizer_->decode() → promise.set_value(results)
```

`ThreadSafeQueue<T>` is a bounded deque with one condition variable (`push()` blocks when full; `try_pop(timeout)`, `pop()`, `shutdown()`).
`submit()` wraps `submit_batch({text})` in a deferred future. `get_stats()` reports counts, average stage times, average latency and
"throughput" (processed / seconds since construction). Stage exceptions are caught and forwarded through the promise.

In `ChatbotAPI`, `generate_response()` calls `pipeline_engine_->submit(input, max_length).get()`. Generation is **always greedy** (strategy,
temperature, top-k/p and repetition settings are not passed in), and the encoder stage uses `LLMEncoder::encode()`, which tokenizes with
the encoder's own tokenizer (loaded from `VOCAB_PATH` by `enable_pipeline_inference()`) and adds `BOS`/`EOS` (TD-364).

---

## 2. Problems

### Shutdown strands queued requests forever (verified)

`shutdown()` clears `running_`, shuts both queues down and joins the workers, which exit as soon as they see `running_ == false`, leaving
any items still queued **inside the queues**. Those items own their `std::promise`s, which live on until the engine itself is destroyed. With
three requests submitted and `shutdown()` called while the first was in the encoder stage, the first future failed with
`Broken promise`, but **the other two were still pending 3 s later**: a caller in `future.get()` blocks indefinitely. (A probe written
without a timeout hung exactly this way.) `submit_batch()` after shutdown fails with `Broken promise`, since `push()` silently drops the item.
In `chatbot_api_server` the engine is destroyed only after the HTTP server stops, so its workers drain the queue first, and a server hang
is unlikely; the contract is still broken for any direct caller.

### It runs the model without `model_mutex_` (by inspection)

`ChatbotAPI` routes generation through the pipeline without taking `model_mutex_`, while `/chat/batch` drives the same model's
`forward()` under that mutex, so a batch request can run concurrently with the decoder thread on the same layers, whose
forward passes cache intermediate state in members (as TD-419 for the integrated engine).

### No real throughput gain (verified timing)

The header promises a "2–3× throughput improvement". The pipeline can only overlap the **encoder** stage with decoding, and decoding
dominates: on a real model (d_model 256, 4+4 layers, debug build), one encoder pass took **144 ms** and 50 greedy decode steps **8,480 ms**,
so the best possible pipelining speedup is **1.017×**. The decode loop re-runs the whole decoder over all tokens generated so far at every
step (no KV cache, O(n²) in output length), and processes a batch's inputs one at a time. The repository's own measurement (1.24× in
PARALLEL_OPTIMIZATIONS_QUICK_REFERENCE.md) comes from `benchmarks/PipelineBenchmark.cpp`, whose stages are `sleep()` calls, not the model.

### Smaller items (by inspection)

- **Different answers from `/chat`**: greedy only, encoder input with `BOS`/`EOS`, no repetition penalty or masking beyond what the decoder
  does (the same class as TD-417).
- **Misleading stats**: `avg_throughput_rps` divides by time since construction, so it decays while idle; `enable_profiling` is never read.
- **One condition variable** for both producers and the consumer with `notify_one()`: a push can wake another blocked pusher instead of the
  consumer, adding up to `try_pop()`'s 50 ms timeout of latency.
- **`std::cerr`** for stage errors; only `std::exception`s are caught (anything else terminates the worker thread's process).

---

## 3. Tests

`pipelineinferenceengine_test.cpp` (61 tests, all against mocks): config and struct defaults, move-only semantics, `ThreadSafeQueue` (push, pop,
timeouts, FIFO order, shutdown unblocking waiters, move-only types), engine lifecycle (construct, shutdown, double shutdown, destructor),
statistics, `submit_batch()`/`submit()` results, multiple sequential batches, special token IDs.

**Not covered:** shutdown with queued requests, the real model, concurrency with other `ChatbotAPI` paths.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Shutdown leaves queued requests' promises alive; post-shutdown submits dropped (§2) | Callers in `future.get()` block forever after an explicit `shutdown()` | **Verified** | [TD-474](../../guides/TECHNICAL_DEBT.md#td-474-pipelineinferenceengine-strands-queued-requests-on-shutdown-and-skips-model_mutex_) |
| Model driven without `model_mutex_` while `/chat/batch` uses it (§2) | Concurrent forward passes race on shared layer state | By inspection | [TD-474](../../guides/TECHNICAL_DEBT.md#td-474-pipelineinferenceengine-strands-queued-requests-on-shutdown-and-skips-model_mutex_) |
| "2–3×" claim vs 1.017× best case; uncached O(n²) greedy decode; serial batch inputs; sleep-based benchmark (§2) | Pipeline mode is no faster (and slower than KV-cached `/chat`) | **Verified** (timing) | [TD-475](../../guides/TECHNICAL_DEBT.md#td-475-pipelineinferenceengine-throughput-claims-greedy-uncached-decoding-and-stats) |
| Greedy-only answers differ from `/chat`; misleading stats; single-cv wakeups; `std::cerr`; non-`std` exceptions (§2) | Inconsistent output; misleading monitoring | By inspection | [TD-475](../../guides/TECHNICAL_DEBT.md#td-475-pipelineinferenceengine-throughput-claims-greedy-uncached-decoding-and-stats) |
| `encode()` adds special tokens; the integrated engine's equivalents | Already tracked | — | TD-364, TD-417–TD-420 |
