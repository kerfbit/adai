# `PerformanceProfiler` — Source File Reference

- **File:** [`src/PerformanceProfiler.hpp`](../../../../src/PerformanceProfiler.hpp) (header-only, 447 lines)
- **Namespace:** global (`Timer`, `ScopedTimer`, `ProfileStats`, `Profiler`, `Benchmark`, `PROFILE_SCOPE`)
- **Used by:** `ChatbotAPI` only: a `Profiler profiler_` member, `PROFILE_SCOPE(profiler_, "generate_response")` around every response, and `GET /admin/profile`. `Timer`/`ScopedTimer`/`Benchmark` are used only by tests
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.1`, `@adai-reviewed: 2026-09-12`
- **Tests:** `PerformanceProfilerTest` in [`tests/inference_optimization_test.cpp`](../../../../tests/inference_optimization_test.cpp) (8 tests); end-to-end through `ChatbotAPI` in `chatbotapi_test.cpp`
- **Last traced against the code:** 2026-10-15

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real header.

---

## 1. What this file is

Small timing utilities:

| Type | Behaviour |
|---|---|
| `Timer` | `start()`, `stop()` (milliseconds, microsecond resolution), `elapsed()`, `reset()` on `std::chrono::high_resolution_clock` |
| `ScopedTimer(name)` | Prints `[name] N ms` to `std::cout` on destruction |
| `ProfileStats` | Every timing in a `std::vector<double>`, plus count, total, min, max, mean; `compute_median()` and `get_percentile(p)` each copy and sort the vector; `print()` |
| `Profiler` | `start(name)`/`stop(name)` keyed by **(name, thread ID)** so concurrent requests time independently (thread-safety fix); one mutex; `get_stats()`, `print_all()`, `reset()`, static `compare()` |
| `PROFILE_SCOPE(profiler, name)` | `start()` now and `stop()` at scope exit through a real RAII guard (TD-097 fixed: it used to stop on the same line, timing nothing) |
| `Benchmark::run()`/`compare()` | Warm-up plus timed iterations; prints a comparison |

### Production use

`ChatbotAPI::generate_response()` is always wrapped in `PROFILE_SCOPE`, whether or not profiling is enabled; `--profile` only gates whether
`GET /admin/profile` reports the stats (call count, total, mean, median, min, max, p95, p99) or says "disabled".

---

## 2. Problems

### It keeps every request's timing forever, and reading the stats blocks requests (verified)

`ProfileStats` stores every sample, and `generate_response()` records one per chat request for the life of `chatbot_api_server`
whether or not profiling is enabled. After 2,000,000 simulated requests the vector held **2M entries (16 MB)**, growing by 8 bytes per request
with no cap. Reading it costs more as it grows: `get_stats()` copies the vector and sorts it **while holding the profiler's mutex**
(**72 ms** at 2M samples), and every in-flight `generate_response()` needs that mutex to start or stop its timer; `handle_profile()` then
sorts two more copies for p95 and p99 (**107 ms**). A bounded reservoir or fixed histogram would give the same report in constant memory.

### Timings use the wall clock (verified)

`Timer` uses `high_resolution_clock`, which in libstdc++ **is `system_clock`** (`is_steady == false`, checked). An NTP step or manual clock
change during a request produces a wrong, possibly negative, duration that then sits in min/mean/percentiles. `steady_clock` is the right
clock for intervals.

### Smaller items (by inspection)

- **Console output**: `ScopedTimer`, `ProfileStats::print()`, `print_all()`, `compare()` and `Benchmark::compare()` print to `std::cout`
  (`stop()` without `start()` to `std::cerr`); none of these run in production.
- **Percentiles** use the floor index (`p/100 × (n−1)`), slightly low for small samples.
- **`compare()` divides** by `optimized.mean_time` without checking it for zero.

---

## 3. Tests

`PerformanceProfilerTest` (8 tests): timer basics and `elapsed()`, stats, median, percentile, multiple sections, the `PROFILE_SCOPE` fix (times
the whole block), and concurrent same-name timing from different threads. `chatbotapi_test.cpp` exercises `/admin/profile` and concurrent
requests through a real `ChatbotAPI`.

**Not covered:** memory growth, clock steps.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Every timing kept forever (always collected); `get_stats()` sorts a full copy under the mutex (§2) | 8 bytes per request without bound; `/admin/profile` stalls requests (72 ms + 107 ms at 2M samples) | **Verified** | [TD-473](../../guides/TECHNICAL_DEBT.md#td-473-profiler-keeps-every-timing-forever-sorts-under-its-lock-and-uses-the-wall-clock) |
| `high_resolution_clock` is `system_clock` here (§2) | Clock steps produce wrong or negative durations | **Verified** (clock alias) | [TD-473](../../guides/TECHNICAL_DEBT.md#td-473-profiler-keeps-every-timing-forever-sorts-under-its-lock-and-uses-the-wall-clock) |
| Console output; floor-index percentiles; unchecked division in `compare()` (§2) | Hygiene | By inspection | [TD-473](../../guides/TECHNICAL_DEBT.md#td-473-profiler-keeps-every-timing-forever-sorts-under-its-lock-and-uses-the-wall-clock) |

---

## 5. Merge notes

This file replaces `docs/development/reference/performanceprofiler.md` (1,313 lines). Kept: the class APIs and statistics semantics. Dropped:
generic usage tutorials and examples. Added: the production usage (always-on collection gated only for exposure), the thread-safety and
`PROFILE_SCOPE` fixes it predates, and the growth and clock issues.
