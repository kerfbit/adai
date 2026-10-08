# `BatchedInferenceEngine` — Source File Reference

- **File:** [`src/BatchedInferenceEngine.hpp`](../../../../src/BatchedInferenceEngine.hpp) (header-only; no `.cpp`)
- **Library:** none — header-only, compiled into whichever target includes it (`adai_nlp`-level dependencies: `BPETokenizer`, `TextGenerator`, `EfficientBatching`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.3`, `@adai-reviewed: 2026-09-14`
- **Tests:** [`tests/batchedinferenceengine_test.cpp`](../../../../tests/batchedinferenceengine_test.cpp) → binary `batchedinferenceengineTests` (ctest name `BatchedInferenceEngineTests`); integration coverage in `tests/chatbotapi_test.cpp` (`BatchedInference_*`)
- **Benchmark:** [`benchmarks/BatchedInferenceBenchmark.cpp`](../../../../benchmarks/BatchedInferenceBenchmark.cpp) → `batched_inference_benchmark`
- **Last traced against the code:** 2026-10-07

> Code-traced reference: every "where it's used" claim points at a real call site in `src/` at
> the time of writing. Older design write-ups (`docs/development/archive/BATCHED_INFERENCE_SUMMARY.md`,
> `docs/development/guides/quick-reference/PARALLEL_OPTIMIZATIONS_QUICK_REFERENCE.md`) describe the
> original *intent*; where they disagree with this file, this file reflects the code.

---

## Contents

1. [What this file is — and what it actually does](#1-what-this-file-is--and-what-it-actually-does)
2. [Where it's used](#2-where-its-used)
3. [`BatchedInferenceConfig`](#3-batchedinferenceconfig)
4. [`BatchedInferenceStats`](#4-batchedinferencestats)
5. [`InferenceRequest`](#5-inferencerequest)
6. [`BatchedInferenceEngine` — public API](#6-batchedinferenceengine--public-api)
7. [`BatchedInferenceEngine` — private worker internals](#7-batchedinferenceengine--private-worker-internals)
8. [Threading model and request lifecycle](#8-threading-model-and-request-lifecycle)
9. [Testing and benchmarking](#9-testing-and-benchmarking)
10. [Known gaps and gotchas (summary)](#10-known-gaps-and-gotchas-summary)

---

## 1. What this file is — and what it actually does

`BatchedInferenceEngine` is an asynchronous **request queue with a single background worker
thread** for text generation. Callers `submit()` a prompt and immediately get back a
`std::future<std::string>`; the worker thread pulls requests off the queue in groups ("batches"),
runs generation for each, and fulfils each request's promise.

It defines four types:

| Type | Role |
|---|---|
| `BatchedInferenceConfig` | Tuning knobs (batch size, collection timeout, queue capacity, …) |
| `BatchedInferenceStats` | Counters plus derived throughput/latency figures |
| `InferenceRequest` | One queued request: prompt, promise, submit time, per-request generation config and model function |
| `BatchedInferenceEngine` | The queue, the worker thread, and the public `submit()`/stats/shutdown API |

### The most important thing to know: it queues, it does not batch the model

The file's header comment advertises "continuous batching", "process entire batch in single model
forward pass", "pad & batch", "dynamic batch sizing", and a "10–20× throughput improvement". **None
of the model-level batching is implemented.** `process_batch()` (§7.4) loops over the collected
requests and calls `TextGenerator::generate_text()` **once per request, one after another**, on the
worker thread. There is no padding, no combined forward pass, and no length grouping. That isn't a
local omission: the model stack has no batch dimension anywhere (TD-171 in
[TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md)), so a real batched forward pass couldn't be
built here without that work first.

What the engine *does* provide, accurately:

- **Decoupling.** Request submission (HTTP handler threads) is separated from generation (one
  worker thread), with a bounded queue and back-pressure (`submit()` throws when full).
- **Serialization.** All generation runs on one thread, so model calls never overlap.
- **Per-request configuration.** Each request carries its own `GenerationConfig` and, optionally,
  its own model forward function (TD-092 and TD-038 fixes, §5).
- **Grouped draining.** Requests are collected in groups of up to `max_batch_size` or until a
  timeout — which today affects only stats and wake-up cadence, not model efficiency.

So, in this codebase, read "batched" as "queued and serialized". Expect **higher** per-request
latency under concurrency (requests wait their turn), not higher throughput.

### Why it matters

It's the engine behind `chatbot_api_server --batched-inference` — one of the three opt-in
alternative inference modes wired in by TD-038 (the others are `--pipeline-inference` and
`--integrated-inference`). It's also the reference design the more ambitious
`IntegratedInferenceEngine` copied from (it includes this header and cites `process_batch()`'s bug
fixes in its own comments), so fixes and pitfalls here tend to recur there.

---

## 2. Where it's used

| Location | What it does with this file |
|---|---|
| `src/ChatbotAPI.hpp` (includes header; member `std::unique_ptr<BatchedInferenceEngine> batched_engine_`) | Owns at most one engine. Owning via `unique_ptr` ties the worker thread's lifetime to the `ChatbotAPI` object's (RAII shutdown on destruction). `nullptr` = mode disabled |
| `src/ChatbotAPI.cpp` — `ChatbotAPI::enable_batched_inference()` (≈ line 740) | Constructs the engine with a **placeholder** default `model_fn` that throws `std::logic_error` if ever called, and a non-owning `shared_ptr` around `ChatbotAPI`'s own tokenizer (no-op deleter) |
| `src/ChatbotAPI.cpp` — `ChatbotAPI::generate_response()` (≈ lines 831–853) | When `batched_engine_` is set: builds a `TextGenerator::GenerationConfig` from the request, builds a per-request `model_fn` capturing the tokenized input **by value**, calls `submit("", &gen_config, request_model_fn)`, and blocks on `future.get()` |
| `src/ChatbotAPIServer.cpp` (≈ lines 523–531, 663) | `--batched-inference` creates a `BatchedInferenceConfig`, sets `timeout_ms` from `--batch-timeout-ms`, calls `enable_batched_inference()`, logs the mode |
| `src/ChatbotApiServerArgs.{hpp,cpp}` | Parses `--batched-inference` / `--batch-timeout-ms <n>` (default 50, deliberately mirroring `BatchedInferenceConfig::timeout_ms`) |
| `src/IntegratedInferenceEngine.hpp` | Includes this header; uses none of its types directly — references `process_batch()`'s fixes in comments |
| `benchmarks/BatchedInferenceBenchmark.cpp` | Throughput/latency benchmark (§9.2) |

### How `ChatbotAPI` drives it (and why it's shaped that way)

```cpp
auto request_model_fn = [this, input_tokens](const std::vector<int>& decoder_tokens) -> Matrix {
    return model_->forward(input_tokens, decoder_tokens);
};
auto future = batched_engine_->submit("", &gen_config, request_model_fn);
return future.get();
```

Three details are load-bearing:

1. **Empty prompt.** `TextGenerator::generate_text()`'s "prompt" is a *decoder* seed, not encoder
   input. For an encoder-decoder model the actual user text lives in the encoder input, which is
   baked into `request_model_fn`. Passing the user text as `prompt` would make the decoder continue
   from it, which is wrong.
2. **Per-request `model_fn` is mandatory** for this model type (TD-038): there's no single "the
   model" function, because every request has a different encoder input. Hence the throwing
   placeholder default — it turns a missed override into a loud error instead of a silent wrong
   answer.
3. **By-value capture.** The closure runs later, on the worker thread. Capturing `input_tokens` by
   value means no lifetime coordination is needed with the caller's stack frame.

**Mode precedence.** `generate_response()` checks RAG → speculative decoding → batched → pipeline →
integrated → GPU → plain CPU, first match wins. Batched mode is ignored when RAG or a draft model
is enabled, and it overrides the GPU path (TD-033) when both are possible.

**Strategy is not honored.** In batched mode, decoding always goes through
`TextGenerator::generate()`'s combined temperature / top-k / top-p sampling. `ChatbotAPI`'s
`GenerationConfig::strategy` (`"greedy"`, `"beam"`, …) and `beam_width` are not mapped across. (A
`temperature` of exactly 0 still gives greedy argmax.)

---

## 3. `BatchedInferenceConfig`

```cpp
struct BatchedInferenceConfig {
    size_t max_batch_size = 32;
    int timeout_ms = 50;
    int max_tokens_per_batch = 4096;
    PaddingStrategy padding_strategy = PaddingStrategy::LEFT;
    bool use_dynamic_batching = true;
    int max_queue_size = 1000;
    bool enable_request_stats = true;
};
```

| Field | Read by the engine? | Effect |
|---|---|---|
| `max_batch_size` | **Yes** — `collect_batch()`, `collect_batch_no_wait()`, `should_flush_batch()` | Max requests taken from the queue per group |
| `timeout_ms` | **Yes** — `collect_batch()` | How long the worker waits to fill a group, measured from when it *starts* collecting (§7.2). Set from `--batch-timeout-ms` |
| `max_tokens_per_batch` | **Yes, but only via a fixed estimate** — `should_flush_batch()` | Flushes when `batch.size() × 100 ≥ max_tokens_per_batch`. With defaults (4096 / 100 → 41 requests) this never fires before `max_batch_size` (32) does |
| `max_queue_size` | **Yes** — `submit()` | Back-pressure: `submit()` throws `std::runtime_error("Request queue is full")` at capacity |
| `padding_strategy` | **No** | Unused (no padding is performed) | [TD-211](../../guides/TECHNICAL_DEBT.md#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model) |
| `use_dynamic_batching` | **No** | Unused (no length grouping is performed) |
| `enable_request_stats` | **No** | Unused — stats are always collected |

The three unused fields are only exercised by tests that check their default values.

---

## 4. `BatchedInferenceStats`

```cpp
struct BatchedInferenceStats {
    uint64_t total_requests, total_batches, total_tokens_processed;
    uint64_t requests_timeout, requests_batch_full;
    double avg_batch_size, avg_latency_ms, throughput_req_per_sec, throughput_tokens_per_sec;
    void compute_derived_stats(double elapsed_seconds);
};
```

**Raw counters** (updated by the worker under `stats_mutex_`):

| Counter | Incremented in | Meaning |
|---|---|---|
| `total_requests` | `process_batch()`, at the start of each group (`+= batch.size()`) | Requests the worker has *started* on |
| `total_batches` | `process_batch()`, at the start | Groups processed |
| `total_tokens_processed` | `process_batch()`, per non-empty result | Tokens in the generated text, approximated by **re-encoding** the decoded output |
| `requests_timeout` | `collect_batch()` | Despite the name, counts **groups** flushed because the timeout expired with ≥ 1 request collected | [TD-213](../../guides/TECHNICAL_DEBT.md#td-213-batchedinferenceengine-stats-are-misleading-and-unused-in-production) |
| `requests_batch_full` | `collect_batch()` | Counts **groups** flushed by `should_flush_batch()` (size or token estimate) |

Groups drained at shutdown by `collect_batch_no_wait()` count toward neither flush counter.

### `compute_derived_stats(double elapsed_seconds)`

Fills the derived fields from the counters and the wall-clock time since the engine started (or
since the last `reset_stats()`):

- `avg_batch_size = total_requests / total_batches` — guarded against `total_batches == 0`.
- `throughput_req_per_sec = total_requests / elapsed` and
  `throughput_tokens_per_sec = total_tokens_processed / elapsed` — guarded against `elapsed <= 0`.
- `avg_latency_ms = elapsed × 1000 / total_requests`.

> **`avg_latency_ms` is not a latency.** It's the inverse of throughput: wall time divided by
> requests, including all the idle time between requests. `InferenceRequest::submit_time` is
> recorded on every request but never read, so true queue-plus-generation latency is not measured
> anywhere. The unit test (`ComputeDerivedStatsAvgLatency`) encodes the current formula.
>
> It's also **not guarded against `total_requests == 0`**. On a fresh engine `get_stats()`
> produces `avg_latency_ms = +∞` (floating-point division by zero, no crash). Treat it as
> meaningless until at least one request has completed.

Called only from `get_stats()`, on a copy, so the stored counters are never modified by it.

---

## 5. `InferenceRequest`

```cpp
struct InferenceRequest {
    std::string prompt;
    std::promise<std::string> result;
    std::chrono::steady_clock::time_point submit_time;
    TextGenerator::GenerationConfig gen_config;
    TextGenerator::ModelForwardFn model_fn;
    // move-only: copy ctor/assign deleted, move ctor/assign defined noexcept
};
```

The unit of work in the queue. It's **move-only** because it owns a `std::promise`, which can't be
copied. The explicit `noexcept` move operations let `std::queue`/`std::vector` move it cheaply.

| Field | Purpose |
|---|---|
| `prompt` | Decoder seed text passed to `generate_text()` (empty from `ChatbotAPI`, §2) |
| `result` | Promise the worker fulfils with the generated text, or with an exception |
| `submit_time` | Set in `submit()`; **never read** (see §4) |
| `gen_config` | Per-request generation settings. **TD-092 fix:** these used to be captured and then ignored — every request generated with the engine's construction-time config |
| `model_fn` | Per-request model forward function; empty means "use the engine's default". **TD-038 fix:** required in practice for encoder-decoder models (§2) |

**Beam-search contract (from the field's own comment, TD-050 investigation).**
`TextGenerator::generate()` silently switches to `generate_beam_search()` when
`gen_config.num_beams > 1`. Beam search calls `model_fn` with several diverging token sequences, so
a `model_fn` backed by a persistent KV cache would have its cache corrupted. If a caller ever sets
`num_beams > 1`, the `model_fn` **must** be a full-recompute function. Today nothing reaches this:
`ChatbotAPI` never sets `num_beams` (default 1), and its `model_fn` calls `model_->forward()`, which
recomputes from scratch anyway.

---

## 6. `BatchedInferenceEngine` — public API

### 6.1 Constructor

```cpp
BatchedInferenceEngine(TextGenerator::ModelForwardFn model_fn,
                       std::shared_ptr<BPETokenizer> tokenizer,
                       const BatchedInferenceConfig& config = {},
                       const TextGenerator::GenerationConfig& gen_config = {});
```

- Stores the default `model_fn`, the tokenizer, the config, and the default generation config.
- Creates its own `TextGenerator` with that default config and a **fixed seed of 0**. Sampling is
  therefore deterministic for a given sequence of requests, but because one generator (one RNG) is
  shared across all requests, a request's sampled output depends on what was generated before it.
- Sets `running_ = true`, starts the stats clock, and **launches the worker thread** running
  `batch_processing_loop()`. The thread starts in the constructor, so the object must be fully
  usable once construction returns. Member declaration order guarantees everything it touches is
  initialized first.

**Used by:** `ChatbotAPI::enable_batched_inference()`, tests, the benchmark.

### 6.2 Destructor

`~BatchedInferenceEngine()` calls `shutdown()`. This RAII behaviour is why `ChatbotAPI` owns the
engine through `unique_ptr`: destroying the `ChatbotAPI` always stops and joins the worker thread.

### 6.3 `submit()`

```cpp
std::future<std::string> submit(const std::string& prompt,
                                const TextGenerator::GenerationConfig* gen_config = nullptr,
                                TextGenerator::ModelForwardFn model_fn = nullptr);
```

1. Throws `std::runtime_error("Cannot submit request: engine is shutdown")` if `running_` is false.
2. Builds an `InferenceRequest`: copies `*gen_config` if given, otherwise the default config. Moves
   in `model_fn`. Records `submit_time`.
3. Takes the future from the promise.
4. Under `queue_mutex_`: throws `std::runtime_error("Request queue is full")` if
   `queue.size() >= max_queue_size`, otherwise enqueues.
5. Notifies the worker and returns the future.

It's thread-safe and can be called from any number of threads. The future resolves to the
generated text, or rethrows whatever generation threw (§7.4).

**Why it matters:** this is the only way work enters the engine, and the queue-full exception is
the engine's only back-pressure mechanism. `ChatbotAPI::generate_response()` doesn't catch it
specifically: its general `catch (const std::exception&)` rethrows it as
`std::runtime_error("Generation failed: Request queue is full")`. The same wrapping applies to
any exception a request's future rethrows from `get()`.

**Used by:** `ChatbotAPI::generate_response()`, `submit_batch()`, tests, the benchmark.

### 6.4 `submit_batch()`

```cpp
std::vector<std::future<std::string>> submit_batch(const std::vector<std::string>& prompts,
                                                   const TextGenerator::GenerationConfig* gen_config = nullptr);
```

A convenience loop that calls `submit()` once per prompt, all with the same (optional) config and
the engine's default `model_fn`. There's no per-prompt `model_fn` overload, so it's unsuitable for
encoder-decoder use. It isn't atomic: if the queue fills partway through, the earlier prompts are
already queued and the exception propagates.

**Used by:** tests and the benchmark only.

### 6.5 `get_stats()` / `reset_stats()`

- `get_stats()`: under `stats_mutex_`, copies the counters, computes elapsed time since the stats
  clock started, calls `compute_derived_stats()` on the copy, and returns it.
- `reset_stats()`: under `stats_mutex_`, zeroes all counters and restarts the stats clock.

**TD-162 (resolved September 12, 2026, for this engine and `IntegratedInferenceEngine`).**
`process_batch()` now updates the token-count stats
*before* calling `set_value()`. A client that sees its future become ready and immediately calls
`get_stats()` from another thread therefore sees that request's tokens counted. The regression test
is `TotalTokensProcessedVisibleImmediatelyAfterFutureResolves`.

**Used by:** tests (including `chatbotapi_test.cpp`'s `call_get_batched_stats()`, which uses
`total_requests` to prove a request really went through the queue). No production code reads these
stats today: there's no HTTP endpoint or metrics export for them.

### 6.6 `queue_size()` / `is_running()`

- `queue_size()`: current queue length, under `queue_mutex_`. Requests the worker has already taken
  into a group are no longer counted.
- `is_running()`: reads the atomic `running_` flag.

**Used by:** tests only.

### 6.7 `shutdown()`

```cpp
void shutdown();
```

1. `running_.exchange(false)`. If it was already false, return; this makes the call idempotent and
   safe to call from the destructor after an explicit `shutdown()`.
2. Wakes the worker with `notify_one()` and joins it.

After the join, the worker has finished its current group and processed **one final group** of up
to `max_batch_size` queued requests (§7.1). The doc comment's "waits for pending requests to
complete" is only true when no more than `max_batch_size` requests were queued; see the gotchas in
§8.3.

**Used by:** the destructor, tests, the benchmark.

---

## 7. `BatchedInferenceEngine` — private worker internals

### 7.1 `batch_processing_loop()`

The worker thread's body:

```text
while running_:
    batch = collect_batch()
    if batch not empty: process_batch(batch)
final = collect_batch_no_wait()
if final not empty: process_batch(final)
```

A group collected just before shutdown is still processed: `collect_batch()` returns what it
already took, and `process_batch()` doesn't check `running_`.

### 7.2 `collect_batch()`

Collects up to `max_batch_size` requests, holding `queue_mutex_` except while waiting:

- Computes **one deadline** (`now + timeout_ms`) when it starts.
- Loops: `wait_until(deadline, queue non-empty || !running_)`.
  - Woken with work: moves the front request into the group, then calls `should_flush_batch()`. If
    it returns true, increments `requests_batch_full` and returns.
  - Woken by shutdown: returns what it has.
  - Deadline passed: increments `requests_timeout` only if the group is non-empty, then returns.

**Consequences:**
- The timeout runs from when the worker *starts waiting*, not from when the first request arrives.
  A request arriving 45 ms into a 50 ms window waits only about 5 ms more.
- **Idle polling.** With an empty queue, the worker wakes every `timeout_ms` (20 times a second at
  the default) just to return an empty group and loop again. The cost is negligible, but the worker
  is never fully asleep.
- Holding `queue_mutex_` while moving requests briefly blocks `submit()` callers. Each move is O(1).

### 7.3 `collect_batch_no_wait()`

Used only during shutdown. Moves up to `max_batch_size` requests out of the queue without waiting.
It's called **once**, so anything beyond one group is left in the queue (§8.3).

### 7.4 `process_batch()`

The core of the engine:

1. Under `stats_mutex_`: `total_batches++`, `total_requests += batch.size()`.
2. **Generation, sequential, per request (TD-092 / TD-038 fixes):**
   ```cpp
   for (const auto& req : batch) {
       generator_->set_config(req.gen_config);
       const auto& fn = req.model_fn ? req.model_fn : model_fn_;
       results.push_back(generator_->generate_text(fn, *tokenizer_, req.prompt));
   }
   generator_->set_config(default_gen_config_);
   ```
   `TextGenerator` has no per-call config parameter, only `set_config()`, so the config is swapped
   in before each request. That's safe because only the worker thread touches `generator_`. Before
   TD-092 this called `generator_->generate_batch()`, which used one fixed config for every request.
   `generate_batch()` is itself only a sequential loop, so moving away from it lost no parallelism.
3. **Distribution**, for each request in order:
   - Updates the token-count stat inside its own `try/catch`, *before* `set_value()` (TD-162). It
     re-encodes the decoded text, skipping empty results because `encode("")` throws.
   - Calls `set_value(result)`.

   Isolating the stats update fixed a real bug. A request whose generation legitimately produced an
   empty string (e.g. EOS as the first token) used to make `encode()` throw. That exception escaped
   to the outer catch and failed **every other request in the group**. The regression test is
   `EmptyResponseFromOneRequestDoesNotFailOtherRequestsInSameBatch`.
4. **Error path.** If anything in step 2 throws (a model or tokenizer error), the outer
   `catch (const std::exception&)` sets that exception on **every** promise in the group. Each
   `set_exception` is wrapped in its own `try/catch(...)` to tolerate promises already fulfilled.

> **Blast radius.** Because generation runs inside one `try` for the whole group, one request whose
> `model_fn` throws fails **all** requests in its group. That includes requests generated
> successfully *before* it, whose results are discarded. Requests in other groups are unaffected.
> Per-request `try/catch` around `generate_text()` would contain it. That's a possible improvement,
> not current behaviour.

The `else` branch (`i >= results.size()` → "Model failed to generate result") is unreachable today:
the generation loop either produces one result per request or throws.

### 7.5 `should_flush_batch()`

Returns true when `batch.size() >= max_batch_size`, or when `batch.size() × 100 >=
max_tokens_per_batch`. The fixed estimate of 100 tokens per request is called out in the source as a
placeholder ("In production, you'd tokenize and count actual tokens"). With default settings only
the size check can trigger.

---

## 8. Threading model and request lifecycle

### 8.1 Threads and locks

| Thread | Touches |
|---|---|
| Caller threads (e.g. HTTP handlers) | `submit()` → `queue_mutex_`; `get_stats()`/`reset_stats()` → `stats_mutex_`; then typically blocks in `future.get()` |
| One worker thread | `queue_mutex_` (collecting), `stats_mutex_` (counters), `generator_` and `tokenizer_` (unlocked; worker-only by construction) |

Lock order: the worker takes `stats_mutex_` while holding `queue_mutex_` (in `collect_batch()`),
and never the reverse. `get_stats()` takes only `stats_mutex_`, so there's no deadlock cycle (the
`GetStatsDoesNotDeadlock` test covers this).

`tokenizer_` is shared with the rest of `ChatbotAPI`. Concurrent use from the worker and from HTTP
threads (e.g. `generate_response()`'s own `encode()` of the input) relies on `BPETokenizer`'s
encode/decode being safe for concurrent reads.

### 8.2 Lifecycle of one `ChatbotAPI` request in batched mode

```text
HTTP thread                                   Worker thread
-----------                                   -------------
generate_response(input)
  encode(input) -> input_tokens
  build gen_config + request_model_fn
  submit("", &gen_config, fn) ──queue──▶      collect_batch()  (≤ timeout_ms, ≤ max_batch_size)
  future.get()  ……… blocks ………                process_batch():
                                                for each request, in order:
                                                  set_config, generate_text(fn, "", …)
                                                    └─ model_->forward(input_tokens, dec)  (per step)
                                                for each request: stats, set_value(text)
  ◀── returns text ─────────────────────────
```

With N concurrent HTTP requests, the last one waits for all N−1 before it to finish generating.
Total throughput equals the inline path's at best. Since `EncoderDecoderModel::forward()` takes
`model_mutex_` internally (TD-156), inline requests would also serialize on the model, so batched
mode mostly trades lock contention for a queue.

### 8.3 Shutdown edge cases

- **More than `max_batch_size` requests queued at shutdown.** Only one final group is drained. The
  rest stay in the queue and are destroyed with the engine, so their futures throw
  `std::future_error` (`broken_promise`) from `get()`.
- **Submit racing shutdown.** `submit()` checks `running_` *before* taking `queue_mutex_`. A request
  that passes the check, then gets enqueued after the worker's final drain, is never processed and
  also ends in `broken_promise`.
- Neither case arises in normal `chatbot_api_server` operation (the engine lives for the process
  lifetime), but both matter for tests and for any future caller that recreates engines.

---

## 9. Testing and benchmarking

### 9.1 Unit and integration tests

```bash
cmake --build --preset=debug --target batchedinferenceengineTests
./build/debug/tests/batchedinferenceengineTests
./build/debug/tests/chatbotapiTests --gtest_filter="*BatchedInference*"
```

`tests/batchedinferenceengine_test.cpp`:

| Group | Covers |
|---|---|
| `BatchedInferenceConfigTest.*` | Default values of every field (including the three unused ones), custom values |
| `BatchedInferenceStatsTest.*` | `compute_derived_stats()` formulas and its two zero-guards (`elapsed == 0`, `total_batches == 0`) |
| `InferenceRequestTest.*` | Move-only semantics (move works; copy is deleted, checked with type traits) |
| `EngineLifecycleTest.*` | Construct/destruct, idempotent shutdown, `submit`/`submit_batch` throw after shutdown, queue size reflects pending work |
| `EngineFunctionalTest.*` | End-to-end results with an EOS-emitting model, per-request `model_fn` override (TD-038), per-request config incl. `max_length` (TD-092), the empty-response isolation fix, TD-162 stats visibility, queue-full back-pressure, stats non-deadlock |

`QueueFullThrows` is worth reading. Because `collect_batch()` drains the queue almost immediately,
the test parks the worker inside a blocking `model_fn` (released by a `std::promise` gate) so the
queue can actually fill up.

`tests/chatbotapi_test.cpp` (`BatchedInference_*`) checks that the mode is off by default, that
requests really route through the queue (proved by `total_requests` moving), and that concurrent
requests all complete correctly.

**Not tested:** shutdown with more than `max_batch_size` queued requests, the one-throwing-request
blast radius, and `avg_latency_ms` on a fresh engine.

### 9.2 Benchmark

`batched_inference_benchmark` (target in `src/CMakeLists.txt`, linked against `adai_nlp` and
`adai_core`) compares sequential `generate_text()` against the engine, using a **synthetic model
that sleeps** a fixed latency per call. `process_batch()` generates sequentially, so the engine
cannot beat sequential generation on model time. Any reported speedup should be read as an artifact
of the synthetic setup or the measurement, not as evidence of real batching. The "27.80× at
batch=32" figure in `IntegratedInferenceEngine.hpp`'s header and the "10–20×" in this file's header
should be treated the same way. (Not re-run while writing this doc.)

---

## 10. Known gaps and gotchas (summary)

> **Planned (October 8, 2026):** real batching is scheduled as a v2 of this file — see
> [real_batching_v2_plan.md](../../../proposals/real_batching_v2_plan.md). Until then the items
> below describe current (v1.x) behaviour.

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Notes | Tracked as |
|---|---|---|---|
| No real batching: generation is sequential per request (§1, §7.4) | Header comments and "10–20×" claims are misleading; concurrency raises latency, doesn't raise throughput | Blocked on TD-171 (no batch dimension in the model stack) | [TD-211](../../guides/TECHNICAL_DEBT.md#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model) |
| `padding_strategy`, `use_dynamic_batching`, `enable_request_stats` unused (§3) | Setting them does nothing | Only default-value tests reference them | [TD-211](../../guides/TECHNICAL_DEBT.md#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model) |
| `max_tokens_per_batch` uses a fixed 100-tokens-per-request estimate (§7.5) | Never fires with defaults | Placeholder per source comment | [TD-211](../../guides/TECHNICAL_DEBT.md#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model) |
| `avg_latency_ms` is wall-time ÷ requests, and is `+∞` with zero requests (§4) | Misleading metric | `submit_time` is recorded but never used | [TD-213](../../guides/TECHNICAL_DEBT.md#td-213-batchedinferenceengine-stats-are-misleading-and-unused-in-production) |
| `requests_timeout` / `requests_batch_full` count groups, not requests (§4) | Misleading names | | [TD-213](../../guides/TECHNICAL_DEBT.md#td-213-batchedinferenceengine-stats-are-misleading-and-unused-in-production) |
| One throwing request fails its whole group, including already-generated results (§7.4) | Collateral failures | Per-request `try/catch` would contain it | [TD-212](../../guides/TECHNICAL_DEBT.md#td-212-batchedinferenceengine-can-fail-or-abandon-requests-it-already-accepted) |
| Shutdown drains only one final group; submit can race shutdown (§8.3) | `broken_promise` on leftover futures | Not hit by `chatbot_api_server`'s process-lifetime engine | [TD-212](../../guides/TECHNICAL_DEBT.md#td-212-batchedinferenceengine-can-fail-or-abandon-requests-it-already-accepted) |
| Idle worker wakes every `timeout_ms` (§7.2) | Negligible CPU | Single deadline per collection call | [TD-211](../../guides/TECHNICAL_DEBT.md#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model) |
| Shared fixed-seed RNG across requests (§6.1) | A request's sampled output depends on earlier requests | Deterministic only for an identical request sequence | [TD-214](../../guides/TECHNICAL_DEBT.md#td-214-batched-inference-mode-generates-differently-from-the-inline-path) |
| `strategy` / `beam_width` not honored in batched mode (§2) | Always sampling (or greedy at temperature 0) | Documented on `ChatbotAPI::enable_batched_inference()` | [TD-214](../../guides/TECHNICAL_DEBT.md#td-214-batched-inference-mode-generates-differently-from-the-inline-path) |
| Beam search with a cached `model_fn` would corrupt the cache (§5) | Not currently reachable | Contract documented on `InferenceRequest::model_fn` | — (not reachable; documented contract) |
| No production consumer of `get_stats()` (§6.5) | Stats invisible outside tests | Could be exposed alongside `/admin/profile` | [TD-213](../../guides/TECHNICAL_DEBT.md#td-213-batchedinferenceengine-stats-are-misleading-and-unused-in-production) |
