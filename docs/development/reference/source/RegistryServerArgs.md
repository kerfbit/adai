# `RegistryServerArgs` — Source File Reference

- **Files:** [`src/RegistryServerArgs.hpp`](../../../../src/RegistryServerArgs.hpp), [`src/RegistryServerArgs.cpp`](../../../../src/RegistryServerArgs.cpp) (168 lines together)
- **Namespace:** `adai`
- **Built into:** compiled directly into `registry_server` (`add_executable(registry_server RegistryServer.cpp Config.cpp RegistryServerArgs.cpp)`) and the test binary
- **Used by:** `main()` in [`RegistryServer.cpp`](../../../../src/RegistryServer.cpp) only
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-09-11` (both files)
- **Tests:** [`tests/registry_server_args_test.cpp`](../../../../tests/registry_server_args_test.cpp) (11 tests, `registryServerArgsTests`; all pass)
- **Siblings:** [MetricsApiServerArgs.md](MetricsApiServerArgs.md) (TD-438), [MnsServerArgs.md](MnsServerArgs.md) (TD-456), [ChatbotApiServerArgs.md](ChatbotApiServerArgs.md) (TD-240)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `registry_server` binary (`build/debug/bin`) with
> `--config /dev/null` and a scratch data directory.

---

## 1. What this file is

`registry_server`'s command-line parsing and config-precedence resolution, extracted from `RegistryServer.cpp` under TD-035 so they can be unit-tested
without a server. The rest of `main()` (globals, routes, the FTP listener) stays in `RegistryServer.cpp`.

### `parse_registry_server_args(argc, argv)`

A single pass over `argv[1..]`. A flag that takes a value is matched only when a value follows (`i + 1 < argc`); otherwise the flag is ignored.

| Flag | Field | Notes |
|---|---|---|
| `--config PATH` | `config_path` | Passed to `ConfigLoader::discover_config_path(…, "config.registry.conf")` |
| `--port N`, `-p N` | `port` (optional) | `std::stoi` |
| `--data-dir DIR` | `data_dir` (optional) | |
| `--ftp-enabled` | `ftp_enabled` | Starts the embedded [`FtpDataServer`](FtpDataServer.md) |
| `--ftp-port N` (2121), `--ftp-ip IP`, `--ftp-pasv-min N` (50000), `--ftp-pasv-max N` (50099) | plain fields | `std::stoi`; CLI-only |
| `--ftp-secret S`, `--ftps`, `--ftp-cert FILE`, `--ftp-key FILE` | plain fields | CLI-only; the shipped `config.registry.conf`'s `FTPS_ENABLED`/`FTP_DATA_SERVER_SECRET`/port keys are ignored (TD-284) |
| `--ftp-ttl MIN`, `--ftp-max-sessions N` | optional | `std::stoi`; also live-tunable via `PUT /admin/config` |
| `--admin-enabled BOOL` | optional | Lower-cased; `true`/`1`/`yes`/`on` → true, **anything else → false** |
| `--help`, `-h` | `help` | Returns immediately, ignoring later arguments |
| anything else | — | **Silently ignored** (documented in the header as preserved legacy behaviour) |

### `resolve_registry_server_config(args, file_config, admin_overrides)`

| Setting | Precedence (low → high) |
|---|---|
| `port` | `REGISTRY_LISTEN_PORT` (8082) → `--port` |
| `data_dir` | `REGISTRY_DATA_DIR` (`registry_sessions`) → `--data-dir` |
| `ftp_ttl_minutes` | `FTP_TOKEN_TTL_MINUTES` (30) → saved `ftp_token_ttl_minutes` → `--ftp-ttl` |
| `ftp_max_sessions` | `FTP_MAX_SESSIONS_PER_RUN` (4) → saved `ftp_max_sessions_per_run` → `--ftp-max-sessions` |
| `admin_enabled` | `--admin-enabled`, else **true** (no config-file key exists, despite the header's "full file/admin-override/CLI precedence" wording) |

An unparseable saved value is skipped (`try { stoi } catch (...) {}`). `main()` resolves once with the saved overrides from
`<data_dir>/daemon_config.db`, or with none if that store can't be opened (a warning goes to `std::cerr`).

---

## 2. Problems

### Number parsing (verified with the real binary)

| Input | Result |
|---|---|
| `--port abc` | Uncaught `std::invalid_argument` from `stoi`: **abort, core dump, exit 134** |
| `--port 8090x` | Accepted as **8090** (trailing garbage ignored) |
| `--port 99999` | Logged "listening on port 99999" but bound **34463** (the port is truncated to 16 bits) |

The same `stoi` calls handle every numeric flag, so `--ftp-ttl`, `--ftp-max-sessions` and the PASV range have the same failure modes, and none is
range-checked (by inspection): a negative or zero `--ftp-ttl` issues tokens that are already expired, `--ftp-max-sessions 0` makes every FTP login fail
with 421, and `--ftp-pasv-min > --ftp-pasv-max` gives an empty port pool (every `PASV` fails with 425, FtpDataServer.md). The config-file and saved-override
paths accept the same values (TD-287, TD-306).

### A flag missing its value is silently dropped (verified)

Because a value flag is matched only when a value follows, a flag at the end of the command line is ignored without a message. **`registry_server
--admin-enabled` (intending to set it, value forgotten) left `PUT /admin/config` open: HTTP 200**, while `--admin-enabled flase` and `no` both gave 403.
Likewise a trailing `--ftp-secret` leaves the HMAC key random and a trailing `--port` falls back to 8082. (The unknown-value case here fails closed,
unlike `mns_server`'s TD-456, which falls back to `true`.)

### Unknown arguments are ignored (verified)

`--portt 8091` started on **8082** with no message, and a typo such as `--ftp-enable` or `--ftps-enabled` silently leaves FTP or TLS off. Every sibling
daemon reports unknown arguments (or is tracked to: TD-438, TD-448); this is the one parser that documents ignoring them.

### Smaller items (by inspection)

- **`--ftps` (or `--ftp-cert`/`--ftp-key`/`--ftp-secret`) without `--ftp-enabled`** does nothing and says nothing.
- **Header comments are stale:** "these are immutable-at-runtime listener settings, not worth threading through the config file" while the shipped
  `config.registry.conf` carries exactly those keys (TD-284); `admin_enabled` is described as layered under file and admin settings but has no file key.
- **`--help` is honoured only up to its position** and exits 0; bad arguments never print usage.

### Related, in `RegistryServer.cpp`'s `main()` (verified; covered with that file)

When the port is already in use, `svr.listen()` returns false and `registry_server` logs "registry_server stopped" and **exits 0** with no error, so a
supervisor's restart-on-failure policy never fires.

---

## 3. Tests

`registry_server_args_test.cpp` (11 tests): defaults with no arguments, `--help` short-circuit, `-p`, unknown arguments being ignored (asserted as
intended behaviour), every FTP flag applying, `--ftp-ttl`/`--ftp-max-sessions` staying unset when absent, lenient booleans for `--admin-enabled`; and for
the resolver: the file baseline, overrides beating the file, an unparseable override being skipped, and CLI flags beating both.

**Not covered:** malformed or out-of-range numbers, a value flag at the end of the command line.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `stoi` without checks: `abc` aborts with a core dump, `8090x` → 8090, `99999` → 34463; no range checks for TTL, sessions or PASV range (§2) | Crash or a different port than logged; FTP silently unusable with 0/negative values | **Verified** (port); by inspection | [TD-488](../../guides/TECHNICAL_DEBT.md#td-488-registry_server-accepts-malformed-numbers-valueless-flags-and-unknown-arguments) |
| A value flag with no value is ignored: bare `--admin-enabled` leaves the admin API open (§2) | Operator believes a setting applied | **Verified** | [TD-488](../../guides/TECHNICAL_DEBT.md#td-488-registry_server-accepts-malformed-numbers-valueless-flags-and-unknown-arguments) |
| Unknown arguments ignored (`--portt` → 8082) (§2) | Typos go unnoticed | **Verified** | [TD-488](../../guides/TECHNICAL_DEBT.md#td-488-registry_server-accepts-malformed-numbers-valueless-flags-and-unknown-arguments) |
| FTP-only flags without `--ftp-enabled` ignored; stale header comments; `--help` positional (§2) | Confusion | By inspection | [TD-489](../../guides/TECHNICAL_DEBT.md#td-489-registry_server-cli-hygiene-ignored-ftp-flags-stale-comments-and---help) |
| FTP security keys in `config.registry.conf` ignored; config/admin values unchecked | Already tracked | — | TD-284, TD-287, TD-306 |
| Bind failure exits 0 | Belongs to `RegistryServer.cpp` | **Verified** | RegistryServer.md (next) |
