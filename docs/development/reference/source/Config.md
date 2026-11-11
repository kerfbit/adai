# `Config` — Source File Reference

- **Files:** [`src/Config.hpp`](../../../../src/Config.hpp) (1,022 lines), [`src/Config.cpp`](../../../../src/Config.cpp) (2,017 lines)
- **Namespace:** `adai`
- **Built into:** compiled directly into most executables (`chatbot_api_server`, `incremental_trainer`, `trainer_service`, the daemons and tools); it isn't part of a static library
- **Status tags:** `@adai-status: stable`; `Config.hpp` `@adai-version: 1.31.0`, `Config.cpp` `1.31.1`; `@adai-reviewed: 2026-11-11`
- **Tests:** [`tests/config_test.cpp`](../../../../tests/config_test.cpp) → `configTests` (107); `config_startup_log_test.cpp` (the deferred start-up messages); `sighupTests` (the binaries that ignore `SIGHUP`)
- **Operator guide (keys and examples):** [configuration_guide.md](../../configuration_guide.md); the shipped `config.*.conf` files are the other key reference
- **Last traced against the code:** 2026-11-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `ConfigLoader` (a scratch program compiled against
> `Config.cpp`) or the binaries.

---

## 1. What this file is

The one configuration layer for every ADAI binary (CLAUDE.md, "Configuration"):

- **`ServiceConfig`:** one flat struct of every setting for every service (server, architecture, training, generation, metrics, database,
  registry, FTP/FTPS, outliers, BLEU/ROUGE, RAG, MNS, trainer admin, world model, hippocampal memory, auto-save, GPU, tokenizer). Each binary
  reads the fields it needs. `file_values` keeps the file's raw `KEY=VALUE` pairs for reload diffs.
- **`ConfigLoader`:** finds the file, parses it, overlays environment variables, validates, reloads and diffs, and prints the result.
- **Shared helpers** in the header: `GPUStrategy`/`gpu_strategy_from_string()`, `is_dataset_kind()`, the metrics storage backend names, the
  admin-setting range tables for `metrics_api_server`/`registry_server` (`AdminIntSetting`), `parse_http_header_list()`; and the strict parsers
  `parse_strict_int/size/float/bool()` the command-line parsers use.

Every tunable passes through here, so a value mis-read here is wrong everywhere downstream.

---

## 2. Discovery and precedence

```text
discover_config_path(explicit, "config.<service>.conf"):
    explicit --config path (returned even if missing)
  > ./config.<service>.conf  > /etc/adai/config.<service>.conf
  > ./config.conf (legacy)   > /etc/adai/config.conf (legacy; returned even if missing)

load(path):  defaults ─► load_from_file(path) ─► load_from_env()          (load(): the legacy /etc path)
```

Precedence is **command-line flags > environment > file > defaults**; each binary applies its flags after `load()`. A missing file is an
`info` message and the binary runs on defaults plus the environment. Every daemon, `trainer_service` and `incremental_trainer` then run
`validate_or_exit()` (TD-288): any error is logged and the process exits 1. `chatbot_api_server` validates after its flags (TD-241).

---

## 3. The file format (`load_from_file()`)

- One `KEY=VALUE` per line, trimmed; a line starting with `#` is a comment; a line without `=` is warned about and skipped.
- **Values:** a value in matching quotes is taken as written (only a comment may follow the closing quote). Otherwise it ends at an unquoted
  `#` that follows a space or tab (`MODEL_PATH=/models/chat # prod` → `/models/chat`; `SECRET=a#b` keeps its `#`) (TD-289).
- **Numbers** (`parse_int`/`parse_size`/`parse_float`) must be the whole value: `8080x`, `256abc` and `0.5f` are refused; `size_t` keys refuse a
  negative (it used to wrap to 1.8×10¹⁹), floats refuse `nan`/`inf` (TD-287).
- **Booleans** (`parse_bool`, every key, file and environment): `true/1/yes/on/enabled` or `false/0/no/off/disabled`, any case (TD-286).
- **Enumerations** are matched in any case and stored canonically: `TOKENIZER_MODE` (`unicode` default, TD-564), `STRATEGY`, `LOG_LEVEL`
  (`WARNING` → `WARN`), `METRICS_STORAGE_BACKEND`, `DATASET_KIND`, `GPU_STRATEGY`, `LR_SCHEDULE` (TD-289).
- A value that doesn't parse is a warning, and the setting keeps its previous value. An unknown key is a warning (typos show);
  `HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT` is reserved and accepted silently (TD-290). A repeated key: the last wins. `MAX_LENGTH` is an alias for
  `MAX_GEN_LENGTH`.
- **Environment** (`load_from_env()`): every file key has an environment variable, read with the same parsers; an empty variable counts as unset,
  and a bad value warns and is ignored.
- **Messages** at load time go through `Logger::deferred()`: on the console at once, and replayed into the log file when the binary's
  `Logger::init()` opens one, filtered by its level (TD-290).

---

## 4. `validate(config, errors)` and `validate_or_exit()`

`validate()` appends a message for every field out of range and returns whether there were none (TD-288):

- **server:** port, session timeout and limits, `LOG_LEVEL`, log size and count;
- **architecture:** `d_model` 64–8192, `num_heads` 1–64 dividing `d_model`, `d_ff` 64–32768, layers 1–48, `max_seq_length` 16–32768;
- **world model:** the same ranges, `sigreg_lambda ≥ 0`, sketches 1–4096, batch size 2–4096, and `world_model_d_model == d_model` when
  injection is on (TD-186);
- **hippocampal** capacity, alphas and decays; **generation** (`adai::GenerationLimits`, TD-254: length, temperature, `top_p`, `top_k`, beam
  width, strategy);
- **training:** learning rate, epochs, batch size, accumulation steps, weight decay, dropout `[0, 1)`, clipping and adaptive clipping, seed;
- **GPU** fraction `(0, 1]` and device; **every port** (PASV min ≤ max; the trainer's admin port ≠ its child port); every metrics, registry, FTP,
  MNS and auto-save interval, size and count; `TRAINER_PASS_STALL_MINUTES` 0–10080 (TD-569). A field whose 0 means "off" or "no limit" accepts 0.

`validate_or_exit(config, binary)` logs every error and calls `std::exit(1)`. Every shipped `config.*.conf` passes
(`EveryShippedConfigFilePassesValidation`).

---

## 5. `reload()`, `detect_changes()`, `changed_file_keys()`

`reload(config, path, mutex, live_keys = nullptr, cli_overrides = {}, cli_keys = nullptr)` builds a fresh config from the file and the
environment, applies `cli_overrides` (the binary's flags, TD-242), and validates it. An invalid config is logged and the current one kept
(returns false). A valid one is **always** applied under `mutex` (TD-285), and each change is logged: a changed key the caller can't apply live
(not in `live_keys`) as "takes effect only after a restart", one in `cli_keys` as "held by a command-line flag". Only `chatbot_api_server`
reloads (on `SIGHUP`); the other binaries ignore `SIGHUP` and say at start-up that a restart applies changes (TD-283).

`detect_changes(old, new)`: for configs read from a file, every added, changed or removed key, with a value shown as `<hidden>` for a secret
key (`is_secret_key()`: a name containing `SECRET`, `PASSWORD`, `TOKEN`, `HEADERS` or `CREDENTIAL`, and `METRICS_DB_URL`, TD-571); for configs not read from a file, about 25 fields compared directly.
`changed_file_keys()` returns just the keys.

---

## 6. `to_string()` and `print()`

`to_string()` lists every setting by key in 16 sections; a secret key's value (`is_secret_key()`) is `<hidden>`. `print()` logs it line by
line (`chatbot_api_server` at start-up).

---

## 7. Tests

`config_test.cpp` (107): defaults; file parsing per key group; quotes and inline comments; numbers (trailing characters, negatives, non-finite);
booleans in any case and bad ones keeping the default; enum values; the environment overriding the file, with the same checks; discovery order;
`validate()` for every group, and every shipped file passing; reload applying any key, reporting additions/removals with secrets hidden, naming
restart-only and flag-held keys; `METRICS_DB_URL` hidden in the reload diff (TD-571); `to_string()`; the strict parsers; `TRAINER_PASS_STALL_MINUTES`. `config_startup_log_test.cpp`: start-up
messages reaching the log file.

---

## 8. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| The reload diff showed `METRICS_DB_URL` (a PostgreSQL URL that can carry a password), which `to_string()` hid | — | **Fixed November 11, 2026** (`ReloadHidesTheMetricsDatabaseUrl`) | [TD-571](../../guides/TECHNICAL_DEBT.md#td-571-findings-from-the-november-11-2026-trace-of-the-files-td-569-touched) |
| `reload()` formats its timestamp with `std::localtime()`, which isn't thread-safe | Harmless while only one thread reloads | By inspection | Not filed (cosmetic) |
| Reload ignoring untracked keys; `SIGHUP` terminating binaries; FTP keys ignored by `registry_server`; two boolean parsers; partial and negative numbers; `validate()` gaps; inline comments and enum typos; output hygiene | — | **Fixed October 31, 2026** | TD-283–TD-290 |
| Flags dropped by reload; the strict parsers for flags | — | **Fixed November 1, 2026** | TD-242, TD-240, TD-456 |

`stable` stands.

## 9. History

- **October 21–26, 2026:** `WORLD_MODEL_SIGREG_BATCH_SIZE` (TD-503), FTPS verification keys (TD-339), `ALLOW_UNTRAINED_MODEL` (TD-246).
- **October 31, 2026:** `SIGHUP` (TD-283), FTP keys (TD-284), reload applies every change (TD-285), one boolean syntax (TD-286), whole-value numbers
  (TD-287), `validate()` everywhere at start-up (TD-288), inline comments and enum values (TD-289), output through the logger (TD-290).
- **November 1, 2026:** flags re-applied on reload (TD-242), `parse_strict_bool()` and `MNS_ADMIN_ENABLED` (TD-456), `REGISTRY_ADMIN_ENABLED`
  (TD-489), `NAME_SERVICE_HEADERS`/`NAME_SERVICE_CA_FILE` and `parse_http_header_list()` (TD-451).
- **November 4, 2026:** the quality backfill switches (TD-273); `TOKENIZER_MODE` defaults to `unicode` (TD-564).
- **November 11, 2026:** `TRAINER_PASS_STALL_MINUTES` (TD-569); `METRICS_DB_URL` hidden in reload diffs (TD-571). This document was re-traced against the code, replacing its October sections
  and folding in its addenda.
