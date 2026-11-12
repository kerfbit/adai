# Training Service (`trainer_service`) Guide

`trainer_service` keeps a model training around the clock. It runs `incremental_trainer` as one training pass at a time: when a pass trains
something it starts the next at once, when nothing is pending it waits 45 s and asks again, and when a pass fails, crashes or hangs it backs
off and tries again with a fresh process. It never touches the GPU itself, so a GPU-driver crash or hang ends only that pass.

For how to queue training data, see the [dataset manager guide](dataset-manager-guide.md); for training settings, the
[training guide](training-guide.md). The code-level reference is [TrainerServiceMain.md](../../development/reference/source/TrainerServiceMain.md).

---

## 1. Before you start

- **Data:** queue it with `dataset_manager` (or the ops dashboard's Registry tab). The service trains whatever its pool holds.
- **Model identity:** with `NAME_SERVICE_URL` set, set `MODEL_NAME` too (or pass `--model`): a service has no terminal for the interactive model
  pick. The architecture then comes from MNS.
- **First run:** nothing to do by hand. With no vocabulary yet, the first pass builds one from its data (sized by `VOCAB_BUILD_SIZE`, or
  automatically) and then the model.
- **Binaries:** `trainer_service` launches the `incremental_trainer` installed beside it, so install both from the same build.

---

## 2. Running it

### Under systemd (the usual way)

```bash
sudo cp scripts/adai-trainer.service /etc/systemd/system/adai-trainer.service
sudo systemctl daemon-reload
sudo systemctl enable --now adai-trainer
journalctl -u adai-trainer -f
```

Adjust the unit's paths, user and `SupplementaryGroups` (the group owning `/dev/dri/renderD128`) first. Keep `SESSION_DIR`, the download
directory and `TRAINER_ADMIN_DIR` in `ReadWritePaths`, and set `LOG_FILE_PATH` (the working directory is read-only under the unit).

### By hand

```bash
trainer_service --config /etc/adai/config.trainer.conf
```

| Option | Meaning |
|---|---|
| `--config <path>` | `config.trainer.conf` to use (default `./config.trainer.conf`, then `/etc/adai/config.trainer.conf`) |
| `--model <name>` | Overrides `MODEL_NAME` |
| `--gpu-strategy background\|full` | Overrides `GPU_STRATEGY` |
| `--objective=chatbot\|lejepa` | What each pass trains: the chatbot (default) or the LeJEPA world model (needs `WORLD_MODEL_ENABLED=true`) |
| `--dataset-kind <kind>` | The pool to train from (`encoder`, `decoder`, `world_model`, `chatbot`); default `world_model` for `lejepa`, else `chatbot` |
| `--help`, `-h` | Usage |

Anything else is an error (exit 1). To train both the chatbot and the world model, run two services, each with its own `--objective` and its
own `TRAINER_ADMIN_PORT`/`TRAINER_CHILD_ADMIN_PORT` (a separate config file); they can share `SESSION_DIR`.

### Stopping it

`systemctl stop adai-trainer`, or SIGTERM. The running pass is asked to stop: it checkpoints, gives its claimed files back to the pool and
exits; after 25 s it is killed. `SIGHUP` is ignored (change the config and restart).

---

## 3. Settings (`config.trainer.conf`)

| Key | Default | Meaning |
|---|---|---|
| `TRAINER_PASS_STALL_MINUTES` | 30 | Stop a pass that has made no progress for this long, and retry after the failure backoff (§5). 0 turns the check off |
| `TRAINER_ADMIN_ENABLED` | false | The admin API (§4) |
| `TRAINER_ADMIN_HOST`, `TRAINER_ADMIN_PORT` | 127.0.0.1, 8084 | Where the admin API listens. It has no authentication: keep it on loopback, or behind an authenticating proxy (the `trainer.kerfbit.dev` Cloudflare tunnel, see [CLOUDFLARE_TUNNEL_RELAY.md](../deployment/CLOUDFLARE_TUNNEL_RELAY.md)) |
| `TRAINER_CHILD_ADMIN_PORT` | 8085 | The loopback port each pass's own admin API uses; must differ from `TRAINER_ADMIN_PORT` |
| `TRAINER_ADMIN_DIR` | `trainer_admin` | Where settings changed through the admin API are kept (`daemon_config.db`) |
| `LOG_FILE_PATH`, `LOG_LEVEL` | console, `INFO` | The service logs to `LOG_FILE_PATH`; each pass logs to the same name with `.pass` added (`trainer.log` → `trainer.pass.log`). Empty: the console (the journal under systemd) |
| `AUTO_SAVE_ENABLED`, `AUTO_SAVE_EVERY_SAMPLES`, `AUTO_SAVE_EVERY_MINUTES`, `MAX_SESSIONS_TO_KEEP` | on, 1000, 30, 50 | Checkpoint cadence and retention; also changeable live through the admin API |

Everything else (`MODEL_NAME`, `NAME_SERVICE_URL`, `REGISTRY_SERVER_URL`, `METRICS_SERVER_URL`, the training hyperparameters, GPU settings)
is read by each pass as `incremental_trainer` reads it. The service validates the file at start-up and exits 1 on an error.

Each pass reports its progress to `<SESSION_DIR>/trainer_service_pass_<pool>.heartbeat` (`<pool>` is `chatbot`, `world_model`…); the file
is removed when the pass ends.

---

## 4. The admin API

With `TRAINER_ADMIN_ENABLED=true`. It stays up the whole time; while a pass runs, most requests are answered by that pass.

```bash
curl -s http://127.0.0.1:8084/health
curl -s http://127.0.0.1:8084/admin/status | python3 -m json.tool
curl -s http://127.0.0.1:8084/admin/logs?limit=50
curl -s -X POST -H 'Content-Type: application/json' http://127.0.0.1:8084/admin/pause
curl -s -X POST -H 'Content-Type: application/json' http://127.0.0.1:8084/admin/resume
curl -s -X POST -H 'Content-Type: application/json' 'http://127.0.0.1:8084/admin/checkpoint?wait_ms=30000'
curl -s http://127.0.0.1:8084/admin/config
curl -s -X PUT -H 'Content-Type: application/json' -d '{"auto_save_every_minutes":15}' http://127.0.0.1:8084/admin/config
```

| Request | What it does |
|---|---|
| `GET /health` | `{"status":"ok"}` while the service runs |
| `GET /admin/status` | The running pass's progress (phase, run and session, epoch, samples, losses, checkpoints) and the service's own figures (below). While a pass is starting, the reply is a 503 that still carries the service's figures |
| `GET /admin/logs[?limit=N]` | The running pass's recent log lines (up to 200) |
| `POST /admin/pause` | Stop starting passes, and drain the running one (checkpoint, give its files back). Lasts until `resume` or a restart |
| `POST /admin/resume` | Start passes again, at once |
| `POST /admin/checkpoint[?wait_ms=N]` | Ask the running pass for a checkpoint now; with `wait_ms` (up to 60,000), wait for it. 409 when no pass is running |
| `GET`/`PUT /admin/config` | The four auto-save/retention settings; a change is kept and used by every later pass |

A write must send `Content-Type: application/json` and no `Origin` header (browser pages are refused).

**The service's own figures in `/admin/status`:**

| Field | Meaning |
|---|---|
| `supervisor_paused` | Paused through the admin API |
| `total_passes_launched`, `total_passes_did_work` | Passes started; passes that trained something |
| `total_passes_failed`, `total_passes_crashed`, `total_passes_stalled` | Passes that exited with an error, were killed by a signal, or were stopped for making no progress |
| `consecutive_failures`, `last_failure_unix` | Failures in a row (they set the backoff) and when the last one was |
| `last_exit_code` | The last pass's exit code (§5) |
| `seconds_since_pass_progress` | How long the running pass has gone without progress; −1 when none is running, 0 with the stall check off |
| `service_uptime_seconds` | Since the service started |

---

## 5. What happens after each pass

| Pass exit | Meaning | The service |
|---|---|---|
| 0 | Trained | Starts the next pass at once (more may be pending) |
| 3 | Nothing pending | Waits 45 s |
| 1 | Failed (registry unreachable, download, checkpoint or training failure, GPU lost) | Counts a failure and backs off |
| 128 + signal (137 SIGKILL, 139 SIGSEGV) | Crashed | Counts a crash and backs off |
| Stopped for no progress (usually 143) | Hung | Counts a stall and backs off |

The backoff is 45 s, doubled for each failure in a row, up to 15 minutes; a pass that trains or finds nothing pending resets it. A pass that
can't even be started (a missing binary) backs off the same way. `/admin/resume` ends any wait at once.

---

## 6. Troubleshooting

| You see | Look at |
|---|---|
| `consecutive_failures` rising, `last_exit_code` 1 | The pass log (`<LOG_FILE_PATH>.pass` or the journal) for the reason: registry or MNS unreachable, a download, a checkpoint that can't be saved |
| `total_passes_crashed` rising, `last_exit_code` 139 | A crash inside the pass, on the Arc B60 usually the GPU driver; see [ONEAPI_SYCL_DRIVER_SEGFAULT.md](troubleshooting/ONEAPI_SYCL_DRIVER_SEGFAULT.md). A pass that finds the GPU lost stops at once and exits 1, and the next pass gets a fresh process |
| `total_passes_stalled` rising | A pass made no progress for `TRAINER_PASS_STALL_MINUTES`, most often a GPU call that never returned after a driver engine reset (`journalctl -k \| grep 'Engine reset'`). The service log names the pass and how long it waited |
| `seconds_since_pass_progress` climbing | The running pass is stuck (or in a very long step); the service stops it at the limit |
| Always idle (`last_exit_code` 3) | Nothing pending for this model in this pool: check the queue (`dataset_manager --kind <pool> status`) and the files' model assignment |
| `/admin/status` answers 503 for a while | A pass is starting; the service's own figures are in the reply |
| `supervisor_paused: true` | Someone paused it; `POST /admin/resume` |
