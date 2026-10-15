# `ModelNameServiceServer` — Source File Reference

- **File:** [`src/ModelNameServiceServer.cpp`](../../../../src/ModelNameServiceServer.cpp) (171 lines; `mns_server`'s `main()`)
- **Built into:** `mns_server` (with `Config.cpp`, [MnsServerArgs](MnsServerArgs.md) and `adai_mns`'s [ModelNameService](ModelNameService.md))
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-11`
- **Tests:** none for `main()` itself; `mns_server_args_test.cpp` covers its argument layer, `model_name_service_live_test.cpp` the service
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `mns_server`.

---

## 1. What this file is

The daemon wrapper around `ModelNameService`:

```text
parse_mns_server_args()            --help → usage, exit 0; error → usage, exit 1
config = load(discover_config_path(--config, "config.mns.conf"))
data_dir = --data-dir or NAME_SERVICE_DIR
create_directories(data_dir); read <data_dir>/daemon_config.db overrides   (failure → warning)
effective = resolve_mns_server_config(cli, config, overrides)
Logger::init(INFO, "mns_server");  SIGINT/SIGTERM → handler (sets a flag, calls service.stop())
ModelNameService service(data_dir, port); set_registry(url, group); set_admin_enabled(…)
thread: service.start()            (create data_dir, open models.db, load, listen on 0.0.0.0:port)
main: poll the flag every 100 ms → service.stop(); join; exit 1 if listen failed, else 0
```

---

## 2. Problems (verified)

### If `models.db` can't be opened, MNS runs in memory and reports healthy

`ModelNameService::init_db()` logs `cannot open SQLite DB …` and returns with no database; every `persist_*()` then
returns early. Nothing in `start()` or `main()` treats this as fatal. With `<data_dir>/models.db` made unopenable,
the real server **started, reported `{"status":"ok"}` on `/health`, and accepted a registration (201)**; after a restart
the model was gone (`404`, `model_count: 0`). Everything MNS is authoritative for (architecture, run numbering, roles)
silently stops surviving restarts, and nothing visible says so. It should refuse to start (or `/health` should report the
missing persistence).

### An unusable data directory aborts with a core dump

`main()` wraps its own `create_directories()` in `try`/`catch` and only warns, but `ModelNameService::start()` calls it
again on the server thread with no handler. With `--data-dir /proc/mnsx` the process printed `terminate called after
throwing an instance of 'std::filesystem::…::filesystem_error'` and **aborted (exit 134, core dumped)**. It does fail, but
as a crash rather than a clear startup error.

### SIGTERM during startup can hang the process

The signal handler calls `service.stop()` and `main()` calls it again, but if the signal arrives before the server thread
has reached `listen()`, both `stop()` calls happen before there is anything to stop; the thread then enters `listen()` and
`main()` blocks forever in `join()`. In 30 trials sending `SIGTERM` 0–90 ms after launch, **1 was still running 2 s later**
(and had to be `SIGKILL`ed). Under systemd, a stop issued right after a (re)start then waits for `TimeoutStopSec`. Same
class as TD-248 (`chatbot_api_server`). The handler also calls `httplib::Server::stop()` from signal context, which isn't
async-signal-safe (TD-249's class); the main loop already does it.

---

## 3. Smaller items (by inspection)

- **`LOG_LEVEL` is ignored**: the logger is always initialized at `INFO` (as TD-431 for the trainer binaries), and there's no
  file logging option.
- **Stale endpoint lists**: the file comment and `--help` omit `PUT …/progress`, `PUT …/run_group`, `POST …/link-world-model`
  (and the comment omits `/datasets` and `/admin/config`).
- **`SIGHUP` terminates the process** (TD-283).

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Unopenable `models.db` → in-memory mode, `/health` ok, data lost on restart (§2) | MNS silently stops persisting what it's authoritative for | **Verified** | [TD-463](../../guides/TECHNICAL_DEBT.md#td-463-mns_server-runs-in-memory-and-reports-healthy-when-modelsdb-cant-be-opened) |
| `SIGTERM` before `listen()` hangs the process; `stop()` from a signal handler (§2) | Occasional stop hangs until SIGKILL | **Verified** (1/30) | [TD-464](../../guides/TECHNICAL_DEBT.md#td-464-mns_server-startup-crash-early-sigterm-hang-and-fixed-log-level) |
| Unusable data dir aborts with a core dump; `LOG_LEVEL` ignored; stale endpoint lists (§2, §3) | Ugly failure; fixed verbosity; misleading help | **Verified** (abort); by inspection | [TD-464](../../guides/TECHNICAL_DEBT.md#td-464-mns_server-startup-crash-early-sigterm-hang-and-fixed-log-level) |
| `SIGHUP` terminates | Already tracked | — | TD-283 |
