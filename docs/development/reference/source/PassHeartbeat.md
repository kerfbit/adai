# `PassHeartbeat` and `PassSupervision` — Source File Reference

- **Files:** [`src/PassHeartbeat.hpp`](../../../../src/PassHeartbeat.hpp) (109 lines), [`src/PassHeartbeat.cpp`](../../../../src/PassHeartbeat.cpp) (137), [`src/PassSupervision.hpp`](../../../../src/PassSupervision.hpp) (41), [`src/PassSupervision.cpp`](../../../../src/PassSupervision.cpp) (54)
- **Built into:** `PassHeartbeat.cpp` in `adai_core` (every binary; the training code calls `note_pass_progress()`); `PassSupervision.cpp` only in `trainer_service`
- **Status tags:** all four `beta` 0.1.0 (from `experimental` November 11, 2026: complete for their scope and tested; not yet deployed), reviewed 2026-11-11
- **Tests:** [`tests/pass_heartbeat_test.cpp`](../../../../tests/pass_heartbeat_test.cpp) → `passHeartbeatTests` (16: the counter, the file, the writer, `PassStallWatch`, and `wait_for_pass()` against real child processes, POSIX)
- **Added by:** TD-569 (November 11, 2026)
- **Last traced against the code:** 2026-11-11

---

## 1. Why it exists

`trainer_service` runs each training pass as a child process. TD-566 stops a pass when a sample fails and the GPU turns out to be unusable, but a
pass whose GPU call never returns (seen on ai-machine's Arc B60 after a driver engine reset) fails no sample: it neither progresses nor exits. The
supervisor used to wait for it forever. These files let the supervisor see that a pass has stopped making progress and stop it.

## 2. `PassHeartbeat` (both processes)

- **`note_pass_progress()` / `pass_progress_count()`:** a process-wide relaxed atomic counter. The pass's working code calls `note_pass_progress()`
  as it goes. It is one increment, safe from any thread. The calls:

  | Where | What counts as progress |
  |---|---|
  | `ChatbotTrainer::train_epoch()` / `validate()` | each training / validation sample |
  | `ChatbotTrainer::preprocess_data()` | each pair tokenized (OpenMP workers too) |
  | `ChatbotTrainer::backfill_generation_quality()` | each pair scored (one generation each) |
  | `DatasetRegistry::load_conversation_pairs()` | each line read, both formats |
  | `DataTransport.hpp`'s FTP write callback | each downloaded chunk |
  | `BPETokenizer::build_bpe_merges()` | each merge (a first pass builds the vocabulary) |
  | the LeJEPA pass (`LeJEPATrainingPass.cpp`) | each batch |
  | `IncrementalTrainer::perform_auto_save()` | each checkpoint attempt |

  Progress is noted when work finishes or starts, never from a timer, so a stuck working thread stops the count. The per-epoch generation-quality
  thread doesn't note progress, so it can't hide a stuck training thread.
- **The file:** one line, `"<pid> <count>"`. `write_pass_heartbeat()` writes `<path>.tmp` and renames it over the file, so a reader sees the
  old or the new file, never part of one. `read_pass_heartbeat()` returns nothing for a missing file, an unparsable line, extra fields or a pid ≤ 0.
- **`PassHeartbeatWriter`** (in the pass, `incremental_trainer --heartbeat-file <path>`, which needs `--foreground`): creates the directory, then
  a thread writes the process id and the count at once and again every 2 s while the count changes. The destructor stops the thread and removes
  the file. A write failure is logged once. The pass starts it right after parsing its arguments, so a hang in GPU initialization or the first
  kernel compile is seen too.
- **`PassStallWatch`** (in the supervisor): `start(pid, now)` begins a pass; `observe(heartbeat, now)` takes the latest read and returns true once
  the count has stayed the same for longer than the limit. A heartbeat with another pid (an earlier pass's file) doesn't count, nor does a
  missing file, so a pass that never writes one is stalled from its start. A limit of 0 never stalls. The time is `trainer_service`'s own
  `steady_clock`; the processes never compare clocks.

## 3. `PassSupervision::wait_for_pass()` (supervisor)

Waits for a started `ChildProcess` until one of these happens:

- the pass exits, and its exit code is returned;
- `control.stop_requested` is set (SIGTERM/SIGINT to `trainer_service`): `stop_and_wait()`, `stopped_for_shutdown`;
- (heartbeat path set and limit above 0) no progress for the limit: logs an error naming the pass, the time and the limit, then `stop_and_wait()`
  and `stalled` with `seconds_without_progress`.

`stop_and_wait(graceful_stop_timeout_ms)` sends SIGTERM to that child's pid only, and kills it after the timeout (exit −2). A pass that is only
slow drains on SIGTERM; a stuck one is killed. The child's exit is polled every `poll_interval` (200 ms) and the heartbeat file read every
`heartbeat_interval` (2 s). While waiting, `control.seconds_since_pass_progress` holds the time since the pass's last progress (0 throughout with
the check off); it is −1 once the wait ends. `TrainerServiceMain.cpp` counts a stalled pass in `total_passes_stalled` and backs off (see
[TrainerServiceMain.md](TrainerServiceMain.md)).

## 4. Settings

`TRAINER_PASS_STALL_MINUTES` (`config.trainer.conf`, read only by `trainer_service`): 0–10080, default 30, 0 turns the check off and no
heartbeat file is passed. The file is `<SESSION_DIR>/trainer_service_pass_<pool>.heartbeat`, `<pool>` being the pass's dataset pool
(`chatbot`, `world_model`…, from `--objective`/`--dataset-kind`), in a directory the pass can already write.

## 5. Known gaps

| Item | Impact | Basis |
|---|---|---|
| A long piece of work that notes no progress is taken for a stall | Such a pass would be stopped after the limit and retried | By design; every long loop found notes progress, and 30 minutes is far above the gaps between them |
| `wait_for_pass()` tests are POSIX only | The Windows build compiles them out | As `childProcessTests` |
| One heartbeat file per pool in `SESSION_DIR` | Two `trainer_service`s training the same pool from one `SESSION_DIR` would overwrite each other's file and stop each other's passes as stalled | Not a supported setup (they would also share checkpoints). A chatbot and a LeJEPA service sharing a `SESSION_DIR` are fine: their pools differ. The file had one fixed name until the guide work on November 11, 2026 found the clash |
