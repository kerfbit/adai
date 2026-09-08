# ai-machine oneAPI/SYCL Driver Segfault — Knowledge Base

## TL;DR

`incremental_trainer` running on `ai-machine` (Intel Arc **B60**, Battlemage family, `xe` kernel
driver, oneAPI/SYCL backend) periodically **hangs or segfaults** shortly after the first GPU kernel
of a run gets JIT-compiled. Root cause: **an upstream Intel driver/compiler bug**
(`libigc.so.2`, the closed-source Graphics Compiler) — not a bug in this codebase. Confirmed via a
symbolized crash backtrace plus three corroborating external GitHub issues on the same
Battlemage/`xe` combination. **Not fixable from here.** Response: stopped chasing the driver bug and
built crash-resilience instead (auto-restart + resume from a tokenized-data cache so a crash costs
seconds, not hours) — see [Mitigation](#mitigation-crash-resilience-not-a-fix).

## Symptoms

Two related failure modes, both landing at the same underlying trigger point (first real GPU
compute of a run):

1. **Hang** — `current_sample` stays at 0 indefinitely (observed once for 6.5+ hours) while the
   process burns high CPU. `gdb -p <pid> -batch -ex "thread apply all bt"` shows the main thread
   blocked forever in `MultiHeadAttention::gpu_upload_weights() → upload_sq_matrix() →
   sycl::event::wait → urEventWait → libze_intel_gpu.so` — a **lost completion notification**: the
   Level-Zero driver never signals the event back to userspace, and nothing on either side times
   out, so it never self-recovers.
2. **Segfault** — `incremental_trainer` crashes outright, symbolized (via ASAN+SYCL build +
   `coredumpctl`/`gdb`) to `SIGSEGV` inside `IgcOclTranslationCtx::TranslateImpl` in `libigc.so.2`
   (Intel's Graphics Compiler, JIT-compiling the run's first SYCL kernel), dereferencing a
   near-null pointer (`0x28`).

Both symptoms occur at essentially the same point in the pipeline: right after tokenization
finishes and the first GPU-touching training work begins (`gpu_forward()`'s one-time lazy
`gpu_upload_weights()` init path fires exactly once, on the very first attention block of the very
first sample — see `src/MultiHeadAttention.cpp`).

## Root Cause

**Confirmed 2026-08-10**, after a multi-day investigation: this is upstream Intel Arc
B60/Battlemage `xe` driver + oneAPI/SYCL stack instability, not an application bug.

- Every one of the **6 recorded crash incidents** on this host (Aug 2, Aug 5, two on Aug 8, Aug 9,
  Aug 10) correlates with a `GT0: Engine reset` in `dmesg`/`journalctl -k` within about a minute of
  the crash — regardless of the crash's superficial symptom (`libsycl.so.9`, `libc.so.6`/`memmove`
  during an earlier failed theory, or `libigc.so.2` directly). **Use `sudo journalctl -k --since
  <time>`** — a non-root query misses the real kernel messages entirely and looks empty even when
  the driver actually logged something.
- ASAN (statically linked into a custom `-DENABLE_SYCL=ON -DENABLE_ASAN=ON` build) reported **no
  corruption in the application's own instrumented heap** when the real crash was caught — ruling
  out the earlier heap-corruption theory (below).
- A driver update (`intel-opencl-icd`/`libze-intel-gpu1` etc. bumped via
  `ppa:kobuk-team/intel-graphics`, Intel/Canonical's Battlemage "preview" stack for Ubuntu 24.04)
  made **zero observable difference** — retried and got a byte-for-byte identical segfault
  signature.
- **External corroboration** (WebSearch, same Battlemage/`xe` combination):
  - [intel/intel-graphics-compiler#159](https://github.com/intel/intel-graphics-compiler/issues/159)
  - [vllm-project/vllm#41663](https://github.com/vllm-project/vllm/issues/41663) — same GP-fault +
    `engine_class=bcs` reset combo, on Arc Pro **B70** (same Battlemage family)
  - [darktable-org/darktable#20257](https://github.com/darktable-org/darktable/issues/20257)
  - Also: [ggml-org/llama.cpp#24810](https://github.com/ggml-org/llama.cpp/issues/24810) documents
    that the SYCL backend doesn't cleanly surface GPU device-loss on this stack — it hangs
    indefinitely instead of erroring, matching failure mode #1 above (Vulkan on the same hardware
    correctly returns `VK_ERROR_DEVICE_LOST`).

### Superseded theory (ruled out, kept for context)

An earlier theory pinned the crash on **heap corruption in `EncoderDecoderModel::save_model()`**,
triggered by a real, separate bug that *was* fixed along the way: `last_save_time` was set in
`IncrementalTrainer`'s constructor, *before* dataset download/tokenization — so on a run whose
tokenization takes hours, `should_auto_save()`'s 30-minute default threshold was already satisfied
by the time the first training sample completed, firing `perform_auto_save()` deterministically on
sample 1 of every run. That fix ([IncrementalTrainer.cpp], reset `last_save_time` when the first
sample actually starts training, not at construction) was real and is still in the codebase — but
after deploying it, crashes kept happening (just later, on `auto_save_every_samples`'s real
threshold instead), and the eventual ASAN-caught backtrace pointed at the driver, not the
save-path. The two are unrelated; both fixes/findings are legitimate, just aimed at different bugs.

### Two genuine performance bugs found en route (also fixed, unrelated to the segfault)

Found via live `gdb` thread dumps while diagnosing a 40+-minute tokenization stall during repro
(not itself the crash — this was slow, not hung):

- `BPETokenizer::pre_tokenize()` constructed a fresh `std::regex("\\s+")` on every match instead of
  once — hoisted to a `static const`.
- `BPETokenizer::apply_bpe()` had no memoization — every occurrence of every common word reran the
  full merge scan from scratch. Added a `thread_local` per-instance cache (deliberately not a
  shared cache — `preprocess_data()`'s OpenMP loop calls `apply_bpe()` concurrently on one shared
  tokenizer instance across threads; a naive shared cache would introduce a new race).

Together these took a 5000-row repro's preprocessing from 40+ minutes to ~5 minutes.

## Why this isn't fixable from this codebase

`libigc.so.2` is Intel's closed-source Graphics Compiler; the crash is inside its JIT translation
path, one layer below anything `adai` controls (SYCL kernel submission → Level-Zero → `libigc`/`xe`
kernel driver). A driver-stack bump already didn't help. The only real levers available from here
are: avoid triggering the fault (not currently possible — it's the *first* kernel compile of any
run), or make the training pipeline tolerate the crash cheaply. We chose the latter.

## Mitigation: crash-resilience (not a fix)

**Decision (user, 2026-08-10): pursue crash-resilience instead of continuing to chase the driver
bug.** Implemented and deployed the same day — see memory `project_trainer_crash_resilience` for
full detail. Key pieces:

1. **On-disk tokenized-data cache** (`CACHE_TOKENIZED_DATA=true` in `config.trainer.conf`) — the
   primary fix. Without it, every crash-restart re-tokenizes the whole dataset from scratch (hours
   for a large corpus); with it, a restart skips straight back to training. This was a hard
   requirement from the user before accepting any restart-based plan at all.
2. **`incremental_trainer --foreground resume`** under `systemd` (`scripts/adai-trainer.service`),
   `Restart=always`, `RestartSec=45` — `resume` is safe to call repeatedly (no-op if nothing
   pending) and reloads the best checkpoint.
3. **Dataset-registry acquire-eligibility fix** — a resumed run can reclaim files it had already
   claimed before crashing (matched by `run_id`, MNS-allocated so it's stable across restarts),
   instead of those files looking permanently "stuck" to a dead run.
4. Three real deployment issues found and fixed live on `ai-machine` during rollout: a
   `SystemCallFilter=` hardening line blocking a GPU/Level-Zero syscall (`SIGSYS`/status=31 crash
   loop), a missing `VOCAB_PATH` in the deployed config, and `Restart=on-failure` misreading a
   clean "nothing pending" exit as a failure (switched to `Restart=always`).

**Later evolution (2026-08-10, same day, separate memory `project_trainer_admin_api`):**
`incremental_trainer` gained a `serve` top-level command — a persistent, never-forking process that
internally loops the same `resume`-style logic instead of relying on systemd to restart a whole new
process every idle cycle, plus an optional always-on HTTP admin API (`/admin/status`,
`/admin/pause`, `/admin/checkpoint`, etc., port 8084). This is implemented and tested locally but
**not yet deployed** to `ai-machine` as of this writing — `ai-machine` is still running the
`resume` + `Restart=always` unit, not `serve`.

**What this does NOT do**: fix the underlying driver bug. Crashes will keep happening at roughly
the same rate; the pipeline now just absorbs them cheaply (seconds-to-minutes of lost progress
instead of hours of re-tokenization) instead of needing manual intervention every time.

## Diagnostic Playbook (if this recurs, or recurs on different hardware)

1. **Hang** (current_sample stuck, high CPU, no progress): `gdb -p <pid> -batch -ex "thread apply
   all bt"`. Look for the main thread stuck in `urEventWait`/`libze_intel_gpu.so`. If found, it will
   not self-recover — kill (SIGTERM then SIGKILL; it may not respond to plain TERM while blocked in
   the driver call) and let crash-resilience resume it.
2. **Crash**: `coredumpctl list` to find the PID, then:
   ```bash
   coredumpctl dump <PID> -o core.<PID>
   gdb <binary> core.<PID> -batch -ex "bt full" -ex "thread apply all bt"
   ```
   Note: `coredumpctl gdb <PID> -batch ...` does **not** forward args to gdb on this host's
   systemd-coredump version (parses `-batch` as its own invalid option) — always extract the core
   and invoke `gdb` directly instead.
3. **Correlate with the driver**: `sudo journalctl -k --since "<crash time>"` (must be root — a
   non-privileged query silently misses the real kernel messages and looks clean even when the
   driver actually logged a `GT0: Engine reset`). If a reset appears within roughly a minute of the
   crash/hang, this is very likely the same class of issue.
4. **If a from-scratch repro/ASAN build is needed**: the trainer always forks+daemonizes by default
   (`train`/`retrain`/`resume` — not `serve`), redirecting stdin/stdout/stderr to `/dev/null` via
   `setsid()`+`dup2`, which silently swallows ASAN's default stderr report. Set
   `ASAN_OPTIONS=log_path=<abs-path>:abort_on_error=1:disable_coredump=0:handle_segv=1` before
   invoking — ASAN opens `log_path` as a real file at init, independent of fd 2, so the later
   `dup2` in the child can't eat it. Use `--foreground` (or `serve`) to skip the fork entirely if
   you'd rather not deal with this at all.
5. Use a throwaway, fully-isolated `config.trainer.conf` for any repro
   (`REGISTRY_SERVER_URL`/`NAME_SERVICE_URL`/`METRICS_SERVER_URL` blank, a fake `MODEL_NAME`) — a
   prior repro attempt accidentally pointed at the real production registry/MNS and uploaded test
   data into the live queue.

## Access Constraint

`incremental_trainer`/GPU work runs only on `ai-machine` (192.168.1.24) — not directly reachable
from a Claude session running elsewhere. Diagnostic data collection (`dmesg`/`journalctl`/
`coredumpctl`/`gdb`/ASAN output) has to be run by the machine's operator and copied back; prepare
repro packages/instructions and let them execute and return results, rather than assuming direct
access.

## Related

- Memory: `project_ai_machine_gpu_hang` (full chronological investigation log, closed)
- Memory: `project_trainer_crash_resilience` (the mitigation feature, deployed)
- Memory: `project_trainer_admin_api` (the `serve`/admin-daemon follow-up, implemented, not yet
  deployed)
- `scripts/adai-trainer.service` — the systemd unit encoding the `Restart=always` mitigation
- `CLAUDE.md` → "Training Pipeline" / "Incremental trainer admin API" sections

---

**Status:** Root cause identified and closed as an external/upstream issue (not fixable in this
codebase). Mitigated, not resolved — training tolerates the crash; the crash itself still happens.
**Last updated:** 2026-08-10
