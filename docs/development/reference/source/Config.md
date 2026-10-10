# `Config` — Source File Reference

- **Files:** [`src/Config.hpp`](../../../../src/Config.hpp), [`src/Config.cpp`](../../../../src/Config.cpp)
- **Namespace:** `adai`
- **Built into:** compiled directly into most executables (`chatbot_api_server`, `incremental_trainer`, `trainer_service`, `chatbot_gui_binary`, the daemons and tools); it isn't part of a static library
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.3.1` (hpp) / `1.3.0` (cpp), `@adai-reviewed: 2026-09-20` / `2026-09-19`
- **Tests:** [`tests/config_test.cpp`](../../../../tests/config_test.cpp) (73 tests)
- **Operator guide (keys and examples):** [configuration_guide.md](../../configuration_guide.md); the shipped `config.*.conf` files are the other key reference
- **Last traced against the code:** 2026-10-09

> Code-traced reference. Every behaviour flagged in §4–§6 was confirmed by running the real
> `ConfigLoader` (a scratch program compiled against `Config.cpp`) unless marked otherwise.

---

## 1. What this file is

The single configuration layer for every ADAI binary (CLAUDE.md, "Configuration"):

- **`ServiceConfig`**: one flat struct of about 110 settings covering every service (server,
  architecture, training, generation, metrics, DB, registry, FTP, outliers, BLEU/ROUGE, RAG, MNS,
  trainer admin, world model, hippocampal memory, auto-save, GPU, tokenizer). Each binary reads only
  the fields it needs.
- **`ConfigLoader`**: finds the right file, parses `KEY=VALUE`, overlays environment variables,
  validates, reloads, and diffs.
- **`GPUStrategy`** plus `gpu_strategy_from_string()` (`background` | `full`).

### Why it matters

Every behaviour an operator can tune passes through here: which port a daemon binds, where
checkpoints go, which model is served, learning rate, gradient clipping, FTPS on or off. A value
that's silently mis-parsed or ignored here becomes the wrong behaviour everywhere downstream, with
nothing in the logs but a line printed before the logger even starts.

---

## 2. Where it's used

`ConfigLoader::discover_config_path()` / `load()` are called by `ChatbotAPIServer.cpp`,
`ChatbotGUI.cpp`, `IncrementalTrainingTool.cpp`, `IncrementalTrainer.cpp`,
`TrainerServiceMain.cpp`, `TrainingMetricsAPIServer.cpp`, `ModelNameServiceServer.cpp`,
`RegistryServer.cpp`, `DatasetManagerTool.cpp`, `MnsCliTool.cpp` and `MnsManagerGUI_main.cpp`.
`IncrementalTrainer::make_incremental_config()` maps `ServiceConfig` into `IncrementalConfig` and
`TrainingConfig` (see [ChatbotTrainer.md](ChatbotTrainer.md)). **`reload()` is called only by
`chatbot_api_server`** (§6).

---

## 3. Discovery and precedence

```text
discover_config_path(explicit, "config.<service>.conf"):
    explicit --config path (returned even if missing)
  > ./config.<service>.conf  > /etc/adai/config.<service>.conf
  > ./config.conf (legacy)   > /etc/adai/config.conf (legacy; returned even if missing)

load(path):   defaults  ─►  load_from_file(path)  ─►  load_from_env()
load():       same, with the legacy /etc/adai/config.conf
```

So precedence is **env > file > defaults**; each binary then applies its own CLI flags on top
(e.g. [ChatbotApiServerArgs.md](ChatbotApiServerArgs.md)). A missing file is just a note on stderr,
and the binary runs on defaults plus env.

The six shipped files are `config.chatbot.conf`, `config.trainer.conf`, `config.metrics.conf`,
`config.mns.conf`, `config.registry.conf` and the legacy `config.conf`. Every key in them is
recognised except `HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT` (in the chatbot and trainer files), which the
trainer file's own comment says is deliberately unread. It still produces an "Unknown
configuration key" warning on every load.

---

## 4. File format and parsing (`load_from_file()`)

- One `KEY=VALUE` per line. Leading and trailing whitespace is trimmed, lines starting with `#`
  are comments, and lines without `=` are warned about and skipped.
- One pair of matching surrounding quotes (`"…"` or `'…'`) is removed from the value.
- **Inline comments are not supported.** `MODEL_PATH=/models/chat # production model` sets the
  path to the whole string, comment included (verified).
- A repeated key: the last occurrence wins. `MAX_LENGTH` is an alias for `MAX_GEN_LENGTH`.
- Unknown keys print a warning (helpful for typos). A value that throws during conversion prints a
  warning, and the field keeps its previous value.

### Numbers

`std::stoi`/`stof`/`stoull`/`stoul` with no checks:

| Input | Result (verified) |
|---|---|
| `PORT=8080x` | 8080 (trailing garbage ignored) |
| `D_MODEL=-1` (size_t field) | **18446744073709551615** (`stoull` wraps negatives) |
| `PORT=abc` | warning, default kept |

There are no range checks at load time; see §5 for what `validate()` covers, and note it doesn't run
at load.

### Booleans: two different parsers

| Accepts | Keys |
|---|---|
| `true`/`1`/`yes`/`on`, **case-insensitive** | `LOG_COMPRESS`, `RAG_ENABLED`, `CACHE_TOKENIZED_DATA`, `TRAINER_ADMIN_ENABLED`, `WORLD_MODEL_ENABLED`, `HIPPOCAMPAL_MEMORY_ENABLED`, `AUTO_SAVE_ENABLED`, `GPU_ENABLED` |
| only lowercase `true`/`1`/`yes` | `GRADIENT_CLIP_ADAPTIVE`, `ENABLE_METRICS_SERVICE`, `METRICS_ENABLE_PERSISTENCE`, `METRICS_ENABLE_PROMETHEUS`, `METRICS_API_ALLOW_CONTROL`, `ENABLE_GENERATION_QUALITY_METRICS`, `FTPS_ENABLED` |

Anything not accepted is silently **false**, with no warning. Verified: `ENABLE_METRICS_SERVICE=On`
and `FTPS_ENABLED=True` both give off, while `GPU_ENABLED=True` gives on. (`FTPS_ENABLED` is moot in
practice, since `registry_server` ignores it entirely; see §5.)

### Enumerations

- `TOKENIZER_MODE`: `unicode` (any case) → Unicode mode; **anything else**, including typos like
  `utf8`, silently means ASCII (verified).
- `GPU_STRATEGY`: `full` / `background`; anything else warns on stderr and falls back to `background`.
- `STRATEGY`, `LOG_LEVEL`, `METRICS_STORAGE_BACKEND`, `DATASET_KIND`: stored as-is (only
  `validate()` checks `STRATEGY` and `LOG_LEVEL`).

### Environment variables (`load_from_env()`)

Every file key has an env counterpart (TD-091 closed the gaps). An empty env var counts as unset.
Typed helpers (`get_env_int`/`size_t`/`float`/`bool`) warn on stderr and ignore unparsable values.
The env boolean parser accepts `true/1/yes/on` and `false/0/no/off` case-insensitively, warning on
anything else. That's stricter than the file parser, except for `FTPS_ENABLED`, which reads
`getenv` directly with the lowercase-only rule.

All load-time messages go to `std::cerr`. That's partly unavoidable, since the logger isn't
initialized yet, but the header's inline `gpu_strategy_from_string()` also writes to `std::cerr`.

---

## 5. `validate(config, errors)`

Checks, appending a message for each failure:

- server: port 1–65535, `session_timeout ≥ 1`, log level in {DEBUG, INFO, WARN, ERROR}, log
  size 1–1024 MB, log files 1–100;
- architecture: `d_model` 64–8192, `num_heads` 1–64 and divides `d_model`, `d_ff` 64–32768,
  layers 1–48, `max_seq_length` 16–32768;
- world model: the same ranges, `sigreg_lambda ≥ 0`, sketches 1–4096, and `world_model_d_model ==
  d_model` whenever injection is on (TD-186);
- hippocampal: capacity 1–65536, `repetition_alpha ≥ 0`, `repetition_decay` in (0, 1];
- generation: `max_gen_length` 1–4096, temperature 0–2, `top_p` 0–1, `top_k` 1–1000, beam width
  1–16, strategy in {greedy, beam, temperature, top_k, nucleus}.

**Not checked** (verified for three of them): `gradient_accumulation_steps` (0 is accepted, and
`ChatbotTrainer` then divides by it), `learning_rate` (−1 accepted), `gpu_memory_fraction` (7.5
accepted), `batch_size`, `num_epochs`, `weight_decay`, every adaptive-clipping field,
`hippocampal_association_decay`/`cross_reference_alpha`, every metrics/registry/FTP/MNS/trainer-admin
port, interval and size (including `FTP_PASV_PORT_MIN ≤ MAX`), and auto-save values.

**The FTP server keys in `config.registry.conf` are ignored.** `registry_server` reads only
`FTP_TOKEN_TTL_MINUTES` and `FTP_MAX_SESSIONS_PER_RUN` from its config file
(`RegistryServerArgs.cpp`). The listener settings are **CLI-only** (`--ftp-port`, `--ftp-pasv-min/max`,
`--ftp-secret`, `--ftps`, `--ftp-cert`, `--ftp-key`), per `RegistryServer.cpp`'s own comment ("CLI-only
listener settings"). Yet `config.registry.conf` ships `FTP_SERVER_PORT`, `FTP_DATA_SERVER_SECRET`
(`change-me-in-production`) and `FTPS_ENABLED` as if they worked, and `ConfigLoader` parses them
without complaint. An operator who sets `FTPS_ENABLED=true` or a real secret in the file gets
**plaintext FTP and random token passwords**, with nothing in the logs. (Nothing in `src/` reads
`ServiceConfig::ftp_data_server_secret` at all.)

**`validate()` runs only inside `reload()`.** No binary validates at startup (TD-241), so every
out-of-range value above, plus `D_MODEL=-1`'s wraparound, reaches the code that uses it.

---

## 6. `reload()` and `detect_changes()`

`reload(config, path, mutex)` builds a fresh config from **file + env** (not CLI; TD-242), validates
it (aborting on errors), then calls `detect_changes(old, new)`. **If that list is empty, it returns
`true` without applying anything.** Otherwise it assigns `config = new_config` under `mutex` and
logs each change.

> **Reload silently ignores most settings (verified).** `detect_changes()` compares only about 25
> fields: model/vocab/draft paths, port, session timeout, log settings, session dir, the six
> architecture fields, and six generation fields. A reload where only other keys changed
> (`METRICS_SERVER_URL`, `AUTO_SAVE_*`, RAG, GPU, MNS, hippocampal…) logs "No configuration changes
> detected", returns **success**, and **keeps the old values**. Verified: changing
> `METRICS_SERVER_URL` and `AUTO_SAVE_EVERY_MINUTES` then reloading left both unchanged, with
> `reload()` returning true. When a tracked field changes too, the whole new config is applied,
> untracked fields included.

> **Only `chatbot_api_server` handles `SIGHUP`.** CLAUDE.md says "Client binaries
> (`chatbot_api_server`, `incremental_trainer`) additionally hot-reload their file via `SIGHUP`."
> But `incremental_trainer` (and `trainer_service`, `metrics_api_server`, `mns_server`,
> `registry_server`) registers only `SIGTERM`/`SIGINT`, so **`SIGHUP` uses the default action and
> terminates them**, killing a training pass mid-epoch for an operator expecting a reload. Of the
> values those binaries would want live, the trainer's auto-save settings are tunable through
> `trainer_service`'s `PUT /admin/config`, not `SIGHUP`.

`detect_changes()` is public and also usable on its own for logging diffs.

---

## 7. `print(config)`

Writes a summary to `std::cout`: server, architecture, generation and RAG settings. It omits most
sections (metrics, registry, FTP, MNS, GPU, training, world model…), but it doesn't print secrets
(`FTP_DATA_SERVER_SECRET`, `METRICS_DB_URL`), which is good. `chatbot_api_server` calls it at
startup.

---

## 8. Tests

`config_test.cpp` (73) covers defaults, file parsing per key group, env overrides, discovery order,
validation rules, reload and change detection, and GPU strategy parsing. Not covered: the boolean
parser split, negative `size_t` wraparound, inline comments, reload with only untracked changes.

---

## 9. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `reload()` returns success but applies nothing when only untracked fields changed (§6) | Settings silently not reloaded | **Verified** | [TD-285](../../guides/TECHNICAL_DEBT.md#td-285-configloaderreload-reports-success-without-applying-untracked-changes) |
| `SIGHUP` terminates every binary except `chatbot_api_server`; CLAUDE.md claims `incremental_trainer` hot-reloads (§6) | Operator reload kills a training pass | By inspection (signal registration) | [TD-283](../../guides/TECHNICAL_DEBT.md#td-283-sighup-terminates-every-binary-except-chatbot_api_server) |
| Two boolean parsers; 7 keys accept only lowercase `true/1/yes`, anything else is silently false (§4) | `ENABLE_METRICS_SERVICE=On` → off, `METRICS_API_ALLOW_CONTROL=True` → off, etc. | **Verified** | [TD-286](../../guides/TECHNICAL_DEBT.md#td-286-configloader-parses-booleans-two-different-ways) |
| Negative values wrap in `size_t` fields; partial numbers accepted (§4) | `D_MODEL=-1` → 1.8×10¹⁹; `PORT=8080x` → 8080 | **Verified** | [TD-287](../../guides/TECHNICAL_DEBT.md#td-287-configloader-accepts-negative-and-partial-numbers) |
| Inline `#` comments become part of the value (§4) | Wrong paths and names | **Verified** | [TD-289](../../guides/TECHNICAL_DEBT.md#td-289-configloader-keeps-inline-comments-and-silently-accepts-enum-typos) |
| `TOKENIZER_MODE` typos silently mean ASCII (§4) | Wrong tokenizer mode for a new vocab | **Verified** | [TD-289](../../guides/TECHNICAL_DEBT.md#td-289-configloader-keeps-inline-comments-and-silently-accepts-enum-typos) |
| `validate()` misses many fields (accumulation 0, LR, GPU fraction, ports…) and never runs at startup (§5) | Crashes or nonsense at use site | **Verified** (validate); TD-241 (startup) | [TD-288](../../guides/TECHNICAL_DEBT.md#td-288-configloadervalidate-misses-fields-that-crash-or-misbehave) |
| `FTPS_ENABLED`, `FTP_DATA_SERVER_SECRET`, `FTP_SERVER_PORT`, PASV range, cert/key in `config.registry.conf` are ignored by `registry_server` (CLI-only) (§5) | Operators setting FTPS or a secret in the file silently get plaintext FTP and random passwords | By inspection (`RegistryServer.cpp`, `RegistryServerArgs.cpp`) | [TD-284](../../guides/TECHNICAL_DEBT.md#td-284-registry_server-ignores-the-ftpftps-security-keys-its-config-file-ships-with) |
| `HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT` in shipped configs but unparsed (§3) | Misleading warning on every load | By inspection | [TD-290](../../guides/TECHNICAL_DEBT.md#td-290-configloader-output-hygiene) |
| stderr/stdout output from library code; `print()` covers a fraction of settings (§4, §7) | Logging convention; incomplete startup summary | By inspection | [TD-290](../../guides/TECHNICAL_DEBT.md#td-290-configloader-output-hygiene) |
