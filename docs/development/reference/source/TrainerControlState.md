# `TrainerControlState` — Source File Reference

- **File:** [`src/TrainerControlState.hpp`](../../../../src/TrainerControlState.hpp) (header-only, 283 lines)
- **Namespace:** `adai` (`TrainerPhase`, `TrainerLogLevel`, `TrainerLogEntry`, `TrainerControlState`, two `to_string()` overloads)
- **Used by:** one `incremental_trainer --admin-port <N> resume` child pass: created fresh per pass in `IncrementalTrainingTool.cpp`, shared between that
  pass's `IncrementalTrainer` (`set_control_state()`) and its [`TrainerAdminAPI`](TrainerAdminAPI.md). The supervisor (`trainer_service`) has its own,
  separate `TrainerServiceControlState`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/trainer_control_state_test.cpp`](../../../../tests/trainer_control_state_test.cpp) (12 tests, `trainerControlStateTests`; all pass); exercised through `incremental_trainer_control_test.cpp` and `trainer_admin_api_test.cpp`
- **Last traced against the code:** 2026-10-17

> Code-traced reference.

---

## 1. What this file is

The in-process state shared by a training pass (writer of progress, reader of tunables and flags) and the admin HTTP handlers on another thread
(reader of progress, writer of tunables and flags).

| Part | Contents |
|---|---|
| `TrainerPhase` | `Idle`, `LoadingData`, `Tokenizing`, `Training`, `Checkpointing`, `Pausing` (+ `to_string()`) |
| Progress (atomics) | `phase`, `current_epoch`, `total_epochs`, `samples_trained_this_pass`, `last_loss` (0.0), `best_loss` (0.0) |
| Flags | `paused` (also handed to `ChatbotTrainer::set_abort_flag()`, so a pause drains the pass at the next optimizer-step boundary), `checkpoint_requested` (consumed with `exchange(false)`) |
| Bookkeeping | `checkpoints_written`, `last_checkpoint_time_unix` |
| Tunables | `auto_save_enabled` (true), `auto_save_every_samples` (1000), `auto_save_every_minutes` (30), `max_sessions_to_keep` (50); the child seeds them from `config.trainer.conf` before `TrainerAdminAPI::start()` overlays `daemon_config.db` |
| `wake()` / `interruptible_sleep(s)` | Condition variable with a sticky `wake_requested_` flag so a wake before the sleep isn't lost |
| Identity strings | `run_id`, `session_id`, `model_name`, `last_checkpoint_path` behind one mutex (copied, never held across work) |
| `log(level, msg)` / `recent_logs(limit)` | 200-entry ring buffer (monotonic IDs, millisecond timestamps) plus the same message through `adai::Logger` |

The ownership contract in the header (each field written by exactly one side) holds in the code.

---

## 2. Gaps (by inspection)

No defects of its own beyond those already tracked, plus these smaller items:

- **Stale design text.** The class and enum comments still describe the always-on `incremental_trainer serve` loop ("no IPC… for the whole lifetime of
  `serve`", "the supervisory loop reads `paused`"). Since TD-172 the state lives for one child pass, and `trainer_service` keeps its own process-lifetime
  state (`TrainerServiceControlState`) and talks to this one over loopback HTTP.
- **`Pausing` is never set.** No code writes `TrainerPhase::Pausing`; a draining pass reports `training`/`checkpointing` and then `idle`, although
  CLAUDE.md's admin-API table and the enum comment list `pausing` as a reported phase.
- **`wake()`/`interruptible_sleep()` are dead here.** The child never sleeps; `TrainerAdminAPI`'s resume calls `wake()` with no waiter. The supervisor's
  `TrainerServiceControlState` carries its own copy of the same mechanism.
- **The ring buffer isn't "the same messages" as the log file**: `log()` stores Debug entries even when `Logger` filters them out, and `recent_logs()`'s
  `limit` is never used by the API (always 200). Messages are unbounded in length.
- **Phase isn't reset on an exception**: `IncrementalTrainer` sets `Idle` on its normal and pause paths only (the child exits soon after, so this matters
  only for the last status reads).

| Item | Tracked as |
|---|---|
| `last_loss` NaN streamed as invalid JSON; `best_loss` starts at 0 | [TD-520](../../guides/TECHNICAL_DEBT.md#td-520-trainer-admin-status-emits-nan-losses-as-invalid-json) |
| Pause/resume semantics across passes (supervisor-level state) | TD-173 (resolved) |

---

## 3. Tests

`trainer_control_state_test.cpp` (12): defaults, `to_string()` for every phase and level, identity round trips, `paused` usable as an abort pointer,
`checkpoint_requested` consumed once, wake-before-sleep and wake-during-sleep, the sleep timing out, and the log ring (ordering, IDs, eviction, limit).

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Stale `serve` comments; `Pausing` never set; dead wake/sleep in the child; ring vs log-file mismatch; phase not reset on exception (§2) | Misleading docs and status | By inspection | [TD-522](../../guides/TECHNICAL_DEBT.md#td-522-trainercontrolstate-stale-design-text-unused-pausing-phase-and-dead-wake-logic) |
| NaN loss JSON, `best_loss` default | Already tracked | — | TD-520 |
