# `Logger` — Source File Reference

- **Files:** [`src/Logger.hpp`](../../../../src/Logger.hpp), [`src/Logger.cpp`](../../../../src/Logger.cpp)
- **Namespace:** `adai`
- **Built into:** `adai_core`; used by essentially every library and binary (`adai::Logger::info/warn/error/debug`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10` (both files)
- **Tests:** [`tests/logger_test.cpp`](../../../../tests/logger_test.cpp) (28 tests)
- **Related:** [LOG_FILE_STANDARDS.md](../../LOG_FILE_STANDARDS.md) (where log files should live)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with scratch programs linked against
> the real `Logger.cpp` and spdlog.

---

## 1. What this file is

A thin static wrapper around one spdlog logger, the project's mandated logging API (CLAUDE.md: "never
`std::cout` in library code"):

| Member | Behaviour |
|---|---|
| `init(level, name = "adai")` | Drops any existing logger with **this** name, creates `spdlog::stderr_color_mt(name)` (stderr, so stdout stays free for TUI output), pattern `[time] [level] msg`, `flush_on(info)` |
| `init(level, FileConfig, name)` | A stderr console sink plus, when `path` is set, a `rotating_file_sink_mt(path, max_size_mb, max_files)`. If the file sink fails, it warns on `std::cerr` and continues console-only. Registers the logger and logs the file settings. `FileConfig::compress` only changes those startup log lines |
| `set_level(string)` / `set_level(Level)` | `DEBUG`/`INFO`/`WARN`(`WARNING`)/`ERROR`, case-insensitive; unknown → `INFO` with a warning. Initializes the logger first if none exists |
| `get()` | Returns the logger, **lazily calling `init(INFO)` if none exists** |
| `debug/info/warn/error(fmt, args...)` | `get()->level(fmt, args...)` (spdlog/fmt formatting) |

Every message at INFO or above flushes the sinks (`flush_on(info)`), so ordering survives a crash, at
the cost of a flush per line.

---

## 2. Problems (verified)

### Concurrent first use throws from a log call

`get()`'s lazy initialization isn't synchronized (`logger_` is a plain static `shared_ptr`). When several
threads log before `init()` has run, more than one of them calls `init()`, and `spdlog::stderr_color_mt()`
throws `spdlog_ex("logger with name 'adai' already exists")` for the losers. Verified: 8 threads calling
`Logger::debug()` in a fresh process, and **3 of 8 calls threw** (in a forked loop, **every one of 200
trials** saw at least one throw). An exception from a logging call is unexpected anywhere and fatal
on a thread with no handler. The unsynchronized `shared_ptr` reads and writes are also a data race, and
`init()` called again while other threads are logging (re-initialization replaces `logger_`) has the
same problem. The binaries call `init()` early from `main()`, which avoids it in practice, but tests and
any code that logs from threads before or during `init()` are exposed.

### Two processes share one rotating log file

`incremental_trainer` (`train`/`retrain`/`resume`) and `trainer_service` fall back to
**`chatbot_server.log`** in the working directory when `LOG_FILE_PATH` is empty, so `trainer_service`
and every child pass it launches write **the same file through separate rotating sinks**. spdlog's
rotating sink isn't multi-process safe: each process tracks its own size and renames the files out from
under the other. Verified with two processes writing 30,000 lines each to a 1 MB × 3 rotating log: they
retained **34,162 lines (2.80 MB)**, while one process writing the same 60,000 lines retained **47,031
(3.85 MB)**. That's about 27% less history than configured, with no corrupted lines. The fallback also
contradicts [LOG_FILE_STANDARDS.md](../../LOG_FILE_STANDARDS.md), which says file logging should be opt-in
(empty path → console only), and the filename suggests the chatbot server's log.

### Other observations (by inspection)

- `incremental_trainer`/`trainer_service` always initialize at `INFO`; only `chatbot_api_server` applies
  `LOG_LEVEL` (at startup and on SIGHUP reload).
- `init(level, name)` drops only the **new** name; re-initializing under a different name leaves the old
  logger registered.
- `FileConfig::compress` does nothing but log three lines.
- `Level::ERROR` is a common macro name on Windows (`wingdi.h` defines `ERROR`), a latent conflict
  for any translation unit including `<windows.h>` without `NOGDI`.
- There are three `TODO`s for JSON output, per-module levels and custom sinks (Future Enhancements in
  TECHNICAL_DEBT.md).

---

## 3. Tests

`logger_test.cpp` (28 tests) covers initialization (console and file), level parsing (including unknown
and case-insensitive), all four log calls, file creation and rotation settings, and re-initialization.

**Not covered:** concurrent first use, or two processes sharing a file.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Unsynchronized lazy `init()` in `get()`: concurrent first use throws `spdlog_ex` from a log call; data race on the static `shared_ptr` (§2) | A logging call can throw or race; re-init during logging is unsafe | **Verified** (3 of 8 threads threw) | [TD-430](../../guides/TECHNICAL_DEBT.md#td-430-logger-lazy-initialization-races-and-can-throw-from-a-log-call) |
| `trainer_service` and its children (and any `incremental_trainer` run) default to the same `chatbot_server.log`, rotated by two processes (§2) | About 27% less log history than configured; violates LOG_FILE_STANDARDS.md's opt-in rule | **Verified** | [TD-431](../../guides/TECHNICAL_DEBT.md#td-431-trainer-processes-share-one-rotating-log-file-and-ignore-log_level) |
| `LOG_LEVEL` ignored by the trainer binaries; `init()` drops only the new name; `compress` is a no-op; `Level::ERROR` vs the Windows macro (§2) | Can't raise or lower the trainer's verbosity; minor leaks | By inspection | [TD-431](../../guides/TECHNICAL_DEBT.md#td-431-trainer-processes-share-one-rotating-log-file-and-ignore-log_level), [TD-430](../../guides/TECHNICAL_DEBT.md#td-430-logger-lazy-initialization-races-and-can-throw-from-a-log-call) |
