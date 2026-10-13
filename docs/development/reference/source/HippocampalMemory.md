# `HippocampalMemory` — Source File Reference

- **Files:** [`src/HippocampalMemory.hpp`](../../../../src/HippocampalMemory.hpp), [`src/HippocampalMemory.cpp`](../../../../src/HippocampalMemory.cpp)
- **Namespace:** global
- **Built into:** `adai_transformer` (included by `DecoderBlock.hpp`); used by `EncoderDecoderModel`, `DecoderBlock`, `chatbot_api_server` and `incremental_trainer`
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.5.0`, `@adai-reviewed: 2026-09-18`
- **Tests:** [`tests/hippocampalmemory_test.cpp`](../../../../tests/hippocampalmemory_test.cpp) → 43 tests
- **Design:** [lejepa_world_model_gated_injection_plan.md](../../../proposals/lejepa_world_model_gated_injection_plan.md) (HM-1, Component 6); TD-179, TD-193 (eviction + swap), TD-194 (associations)
- **Last traced against the code:** 2026-10-13

> Code-traced reference. Behaviour marked **verified** was confirmed by driving the real class
> (linked against the debug libraries) in a scratch program, including a simulation of
> `DecoderBlock`'s per-step coverage bookkeeping.

---

## 1. What this file is

A **bounded episodic memory** of `(key, value)` pairs, each `[1, d_model]`. It's the "hippocampal"
half of the LeJEPA plan's complementary-learning-systems design. It starts empty, gains one entry per
generated response, and is read by `DecoderBlock`'s gated hippocampal cross-attention (TD-180). Each
slot has three kinds of bookkeeping, kept in lockstep in parallel containers:

| Container | Meaning | Maintained by |
|---|---|---|
| `slots_` (`deque<Slot{key, value}>`) | The memories | `write()`, eviction, `recall_from_swap()`, `load()` |
| `coverage_` (`deque<float>`) | How much attention a slot has received (decayed sum); used for the repetition penalty and as the eviction signal | `DecoderBlock` via `coverage_vector()`/`decay_coverage()` |
| `association_` (`deque<deque<float>>`, n × n) | Hebbian "used together" strengths (TD-194) | `DecoderBlock` via `association_matrix()`/`decay_associations()` |

### Why it matters

It's the only state the model accumulates at serving time. Whether the memory fills with useful,
recent content decides whether the hippocampal path can do anything at all. (Separately, TD-341 means
its gate never trains, so in shipped models its output is multiplied by `tanh(0) = 0` today.)

---

## 2. API

| Member | Behaviour |
|---|---|
| `HippocampalMemory(d_model, capacity = 512, swap_filepath = "")` | Rejects non-positive sizes. An empty swap path disables swap |
| `write(key, value)` | Checks both are `[1, d_model]`; at capacity evicts the **least-used** slot first; appends with **coverage 0** and a fresh zero association row and column |
| `read_all()` | **Copies** every slot into new `[n, d_model]` key and value matrices. `DecoderBlock` calls it on every forward pass of every injected layer, so that's O(n·d) per call |
| `coverage_vector()`, `association_matrix()` | Mutable references; the caller does the bookkeeping |
| `decay_coverage(γ)`, `decay_associations(γ)` | Multiply every entry by γ |
| `recall_from_swap()` | Reads the **last** swap record (LIFO), truncates it off the file, and re-inserts it with coverage 0. If the buffer is full, that evicts (and swaps out) another slot. Returns false when swap is off or empty; throws on a `d_model` mismatch |
| `clear()` | Empties all three containers (doesn't touch the swap file) |
| `save(path)` / `load(path)` | `d_model`, `capacity`, count, then `(key, value, coverage)` records. `load()` validates `d_model` and the record count against the file size, evicts least-used slots down to *this* instance's capacity (into swap when enabled; verified: a capacity-4 file into capacity 2 kept the high-coverage slot and the newest), and resets associations to zeros. **`save()` doesn't check its writes** |
| `save_associations(path)` / `load_associations(path)` | n, then the n × n floats. Load requires `n == size()` and validates the size against the file. Save is unchecked |
| `size()`, `get_capacity()`, `get_d_model()`, `swap_enabled()` | Accessors |

### Eviction ("least used", TD-193)

`evict_least_used()` removes the slot with the minimum coverage (ties go to the oldest) and drops its
association row and column. When swap is enabled it first appends the slot to the swap file, which is
an append-only log: a `d_model` header, then `(2·d_model + 1)` floats per record. **Nothing bounds
the swap file**: once the buffer is full, every response appends a record (about 4 KB at
`d_model = 512`) and only `recall_from_swap()` ever shrinks it. Nothing in production calls
`recall_from_swap()`.

> **New memories can be starved out (verified by simulation).** A new slot starts at coverage 0,
> while every older slot carries an accumulated, decayed coverage. Under `DecoderBlock`'s
> bookkeeping (each forward step: `decay_coverage(0.95)`, then add each slot's mean attention
> weight), a slot needs about `1/(1 − γ)` = 20 steps to approach the level of the others. If
> attention over slots is near uniform, the newest slot always has the least coverage and is the
> one evicted at the next write. Simulated with the real class (capacity 8, one write per "response"
> followed by 50 forward steps, 40 responses):
>
> | Attention over slots | Surviving memories (write #) |
> |---|---|
> | Uniform | **0 1 2 3 4 5 6** 39: the first 7 are permanent; each new one survives only until the next write |
> | Random (peaked softmax) | 20 22 26 33 35 37 38 39: normal turnover |
>
> Near-uniform attention is the likely case in shipped models. The hippocampal cross-attention's
> weights get no gradient while its gate is stuck at 0 (TD-341), and keys are mean-pooled
> world-model embeddings of whole responses, which tend to be similar. The policy then degenerates
> into "keep the first `capacity − 1` memories forever". That's the opposite of the FIFO it replaced,
> and of an episodic memory's purpose. Typical fixes are to give new slots a protection window or an
> initial coverage equal to the current mean, or to evict by `coverage / age`.

---

## 3. Who calls what

| Caller | Uses |
|---|---|
| `EncoderDecoderModel::maybe_write_hippocampal_memory()` | After a response is generated, with a world model attached: encodes the response text with the world model, mean-pools it, and calls `write(pooled, pooled)` (key = value). Failures are logged and swallowed |
| `DecoderBlock::forward()` (gated hippocampal path) | `read_all()`, `coverage_vector()`, `association_matrix()` for the score bias; afterwards `decay_coverage()` plus accumulation, and the Hebbian association update. Runs on **every** forward pass, including training and beam hypotheses (TD-343) |
| `chatbot_api_server` | Builds it with `HIPPOCAMPAL_MEMORY_CAPACITY` and a swap file at `<model>.hippocampal.swap`; `load()`/`load_associations()` at startup from `<model>.hippocampal[.assoc]`; saves only on graceful shutdown (TD-250) |
| `incremental_trainer` (`IncrementalTrainer`) | Builds one **without** swap and attaches it for training |

---

## 4. Tests

`hippocampalmemory_test.cpp` (43 tests) covers:
- construction validation, write shape checks, capacity;
- least-used eviction and its tie-break, coverage alignment and decay;
- `read_all()`, `clear()`;
- save/load (round trip, `d_model` mismatch, corrupt counts, capacity change, swap on trim);
- swap (disabled by default, append, LIFO recall, drain, `d_model` mismatch);
- associations (growth, mutation, decay, eviction alignment, recall, reset on load, save/load, mismatch and truncation).

The class itself behaves as documented. **Not covered:** the interaction of the eviction policy with
real coverage dynamics (§2), swap-file growth, and unchecked writes in `save()`/`save_associations()`.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Least-used eviction with new slots starting at coverage 0 starves new memories when attention is near uniform; the buffer freezes on its first `capacity − 1` entries (§2) | The episodic memory stops recording anything new; likely in shipped models (TD-341 keeps the attention untrained) | **Verified** (simulation with the real class) | [TD-396](../../guides/TECHNICAL_DEBT.md#td-396-hippocampal-memory-eviction-starves-new-memories) |
| The swap file grows without bound (one record per response at capacity; nothing calls `recall_from_swap()` in production) (§2) | Unbounded disk use beside the model checkpoint on a long-running server | By inspection | [TD-397](../../guides/TECHNICAL_DEBT.md#td-397-hippocampal-swap-file-grows-without-bound) |
| `read_all()` copies the whole memory on every forward pass of every injected layer; `save()`/`save_associations()` don't check writes; the main and association files are saved separately and can disagree (`load_associations()` then rejects or mismatches) (§2) | Per-step copying; silent partial saves | By inspection | [TD-398](../../guides/TECHNICAL_DEBT.md#td-398-hippocampalmemory-per-step-copies-and-unchecked-saves) |
| Memory mutated on every forward pass; gates never train; saved only on graceful shutdown | Already tracked | — | TD-343, TD-341, TD-250 |
