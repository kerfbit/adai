# `RegistryServer` — Source File Reference

- **File:** [`src/RegistryServer.cpp`](../../../../src/RegistryServer.cpp) (1,939 lines; `main()` and every handler, file-scope globals)
- **Binary:** `registry_server` (port 8082), built from `RegistryServer.cpp`, `Config.cpp` and `RegistryServerArgs.cpp`
- **Clients:** `RemoteTransport` ([DatasetRegistry.md](DatasetRegistry.md)) in `incremental_trainer` and `dataset_manager`; the Android ops dashboard's Registry tab; `mns_server`'s dataset proxy
- **Status tag:** `@adai-status: beta` (TD-035 partly resolved; TD-040 resolved), `@adai-version: 0.10.1`, `@adai-reviewed: 2026-09-20`
- **Tests:** no in-process tests. [`dataset_registry_live_test.cpp`](../../../../tests/dataset_registry_live_test.cpp) (86 tests against a running server; **all pass** against a local instance), [`registry_ftp_confinement_test.cpp`](../../../../tests/registry_ftp_confinement_test.cpp) (4, spawns the binary; pass), `registry_server_args_test.cpp` (11, see [RegistryServerArgs.md](RegistryServerArgs.md))
- **Related:** [FtpDataServer.md](FtpDataServer.md), [DataFetcher.md](DataFetcher.md), [ParquetReader.md](ParquetReader.md), [DaemonConfigStore.md](DaemonConfigStore.md), [DataTransport.md](DataTransport.md)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed against the real `registry_server` binary (`build/debug/bin`), started with
> `--config /dev/null` and a scratch `--data-dir`, using `curl`.

---

## 1. What this file is

The dataset-queue daemon for a training pool: a shared pending-file queue per **group** (and per TD-202 **kind** sub-pool), from which trainers
atomically claim files (`acquire`), train, and commit them (`trained`). Optionally it delivers the bytes through an embedded FTP/FTPS server and fetches
or accepts datasets itself (Gutenberg, HuggingFace, upload).

There is **no authentication** on any endpoint, and it listens on `0.0.0.0`. The design assumes a trusted LAN, with Cloudflare Access in front of the
public `registry.kerfbit.dev` hostname ([CLOUDFLARE_TUNNEL_RELAY.md](../../../operations/deployment/CLOUDFLARE_TUNNEL_RELAY.md)), so any host on the LAN
can claim, reassign, delete or upload.

### State and storage

| State | Where |
|---|---|
| Per group/kind queue and trained registry | `LocalTransport` over `<data_dir>/<group>[/<kind>]/pending_files.txt` and `data_registry.txt` (the standalone format, so files can be inspected by hand). One `GroupState{mutex, transport}` per `group[/kind]`, created on first request (which also creates the directory) |
| Fetched and uploaded data | `<group_dir>/datasets/`, `<group_dir>/uploads/`; fetch cursors in `<group_dir>/fetch_cursors/` |
| Admin overrides | `<data_dir>/daemon_config.db` (`ftp_token_ttl_minutes`, `ftp_max_sessions_per_run`) |
| Session counters | `<data_dir>/session_counters.db`, key `model_name|run_id` |
| FTP settings | File-scope globals set once from the CLI ([RegistryServerArgs.md](RegistryServerArgs.md)); `ftp_token_ttl_min` is atomic and live-tunable |

Every handler reads the whole pending (and/or registry) file, changes it in memory and rewrites it, under the group's mutex. Requests to one group are
serialized; different groups run in parallel on httplib's thread pool (8 threads or one per core).

### Request parsing

Hand-written helpers find `"key":` by substring: `json_string()` (up to the next `"`, no unescaping), `json_int()` (`stoi`, default on failure),
`json_bool()` (literal `true` only), `json_has_key()`, `json_string_array()`, `json_int_array()` (unparseable → 0), and `json_segment_targets()` (flat
objects). They match compact JSON only: **verified**, `{"run_id": "r2", "max_files": 1}` (a space after the colon) returned `400 run_id required`. All
current clients send compact JSON. Output strings are escaped with `json_escape()` (quote, backslash, `\n`, `\r`, `\t` only).

### Endpoints

Every `/registry/<group>/…` route also accepts `/registry/<group>/<encoder|decoder|world_model|chatbot>/…` (TD-202); any other kind 404s. `<group>` is
`[^/]+`, unvalidated (see §2).

| Endpoint | Behaviour |
|---|---|
| `GET …/queue` | Every pending entry: path, `run_id`, `model_name`, source, `added_utc`, size, `num_entries`, checksum, segment range |
| `POST …/acquire {run_id, max_files, model_name}` | Claims up to `max_files` (≤ 0 = all) entries that are unclaimed or already claimed by this `run_id`, and are unassigned or assigned to `model_name`. With FTP active, entries outside `data_dir` are skipped (TD-040). Response: legacy `acquired` + parallel segment arrays, or (FTP) per-file `registry_path`/`ftp_path`/one-time credentials/size/checksum/expiry/segment. **No lease:** a claim stays until `release`, `trained` or a forced delete |
| `POST …/release {run_id, files | segments}` | Clears `run_id` on matching entries; an empty `run_id` releases regardless of owner. `files` matches whole-file entries only |
| `POST …/assign {model_name, paths | segments | count}` | Sets `model_name`: explicit segments; else paths (whole-file); else the first `count` unassigned whole-file entries; else **every** entry. `model_name` must match `[A-Za-z0-9._-]+` |
| `POST …/unassign {model_name, paths | segments, force}` | Clears `model_name`; refuses all-empty; skips claimed entries unless `force` |
| `POST …/delete {paths | segments, force, delete_files}` | Removes from pending (claimed ones only with `force`) and from the trained registry (whole-file only); `delete_files` unlinks the file if it resolves under `<data_dir>/<group>` |
| `POST …/trained {run_id, files, samples, model_id}` | Adds a `DataVersion` per file **not already in the registry** (checksum from the local file, else `REMOTE`), then removes the files from pending where `run_id` matches (or is empty) |
| `GET …/registry`, `…/runs`, `…/history[?model_id=]` | The trained registry; claimed files per run; registry filtered by model |
| `POST …/pending/add {path, segment_start, segment_count}` | Queues any path (warns if outside `data_dir`); returns `added:false` if the same path/segment is already pending |
| `POST …/session/next {model_name, run_id}` | `session-NN`, incrementing per `(model_name, run_id)`; group-agnostic |
| `POST …/fetch/gutenberg {book_id, num_pairs, model_name}`, `…/fetch/huggingface {dataset_id, split, num_pairs, input_field, output_field, model_name}` | Download and cache the source once ([DataFetcher.md](DataFetcher.md)), then write the next `num_pairs` from a per-(source, model) cursor to `<group_dir>/datasets/…_row<offset>_training.jsonl` and queue it. Runs synchronously in the request thread |
| `POST …/upload?filename=` | Writes the raw body to `<group_dir>/uploads/<filename>` (basename only) and queues it |
| `GET`/`PUT /admin/config` | The two FTP settings above (PUT gated by `--admin-enabled`; immutable keys rejected with 400); persisted to `daemon_config.db` |
| `GET /health` | Always `{"status":"ok"}` |

### `main()`

Parses arguments ([RegistryServerArgs.md](RegistryServerArgs.md)), loads `config.registry.conf`, opens the two stores (each failure is a warning on
`std::cerr`), initializes the logger, installs SIGINT/SIGTERM handlers that call `svr.stop()`, starts the FTP server if enabled, registers the routes and
calls `listen("0.0.0.0", port)`.

---

## 2. Problems

### A `..` group escapes `data_dir`, and `delete_files` then unlinks files outside it (verified)

`<group>` is used unvalidated in `data_dir + "/" + group`, and httplib decodes `%2e%2e` and accepts a literal `/../` path segment. So
`/registry/../<action>` operates on `data_dir`'s **parent**:

- `POST /registry/../upload?filename=t1.jsonl` and `/registry/%2e%2e/upload?filename=t2.jsonl` both returned 200 and wrote
  `<data_dir>/../uploads/t1.jsonl` and `t2.jsonl`, plus a `pending_files.txt` in `data_dir`'s parent.
- **Deleting arbitrary files:** with a "precious" file at `<parent>/models/checkpoint.bin`, `pending/add` + `delete` with `delete_files:true` through
  group `g` correctly refused to unlink it (`file_deleted:false`). Through group `..`, the containment root became the parent directory and the same two
  requests **deleted it** (`file_deleted:true`; the file was gone).

With the default relative `REGISTRY_DATA_DIR=registry_sessions`, the parent is the server's working directory, which in a checkout or deployment
directory holds `training_sessions/` checkpoints, configs and binaries. Two unauthenticated requests from any LAN host (or anyone past Cloudflare Access)
can delete any file there that the service user can write. By inspection, group `.` similarly maps to `data_dir` itself, letting `delete` reach `daemon_config.db`,
`session_counters.db` and other groups' files. Groups should be validated against a strict pattern (as model names already are), and every path
built from them should be checked against `data_dir`.

### Concurrent fetches serve the same slice (verified)

The fetch cursor is read and written **outside any lock** (`read_cursor()` → convert → `write_cursor()`), and the output name is derived from the
offset. Eight concurrent `fetch/huggingface` requests for 1,000 pairs each (against a pre-cached 20,000-row dataset) all reported `served_from_row: 0`
and `added: true`, but they wrote **one** file and the cursor advanced only to 1,000: 8,000 pairs were reported and 1,000 queued, and the eight writers
wrote the same file at once. The same applies to Gutenberg. After the cursor wraps around, a fetch rewrites the earlier `_row<offset>` file, which may
still be pending or claimed.

### Upload silently replaces a claimed file (verified)

`upload` truncates and rewrites `<group_dir>/uploads/<filename>` even when that file is pending, claimed by a running trainer, or already trained.
Uploading `d.jsonl`, claiming it as run `r1`, and uploading different content under the same name returned `added: true`, replaced the bytes under the
claim (the trainer then trains, or downloads, data that is not what was queued), and queued nothing new (the path was already pending). The response
doesn't distinguish "added" from "already pending", unlike `pending/add`.

### `trained` doesn't check who claimed the file (verified)

`trained` records a file in the registry regardless of `run_id`, and only the pending removal checks it. With `d.jsonl` claimed by `r1`, a `trained`
call from run `other` returned `trained: 1` and added the registry record (with the replaced file's checksum), while the entry stayed pending and claimed
by `r1`. When `r1` then reported it trained, the response was `trained: 0` (the registry already had the path) although the pending entry was removed.
So a stale or mistaken run can mark another run's file trained, and the real run's samples/`model_id` are never recorded. Because the registry is keyed
by bare path, a file re-added and trained again is never recorded a second time either (with TD-333's segment problem on top).

### Smaller items

- **Bind failure exits 0 (verified).** When the port is in use, `listen()` returns false and the server logs "registry_server stopped" and exits with
  status 0, so a restart-on-failure supervisor never restarts it.
- **SIGTERM during startup is lost (by inspection).** `svr.stop()` before `listen()` has begun does nothing, so a signal in that window leaves the
  server running (the same race as TD-464).
- **Claims never expire (by inspection).** A crashed trainer's files stay claimed until someone releases them by hand (with an empty `run_id`).
- **Long fetches tie up request threads (by inspection).** Downloads run inside the handler; a handful of concurrent fetches can occupy the whole
  thread pool (8 on a small host) and stall `acquire`/`trained`. Upload bodies have no size limit and are held in memory.
- **Compact-only JSON parsing (verified)** and no unescaping of `\"`/`\\` in incoming paths; `json_int_array()` turns bad values into 0.
- **`session/next`** keys on `model_name|run_id` with `model_name` unvalidated, so a `|` in it can collide; `/health` reports ok even when
  `session_counters.db` couldn't be opened.

### Already tracked elsewhere

| Item | Tracked as |
|---|---|
| FTP/FTPS keys in `config.registry.conf` ignored; argument parsing | TD-284, TD-488, TD-489 |
| `SIGHUP` terminates the server | TD-283 |
| Session-counter atomicity, admin PUT partial apply and range checks | TD-304, TD-305, TD-306, TD-308 |
| Gutenberg regex crash, HuggingFace `split` traversal, partial caches, Parquet hangs | TD-309, TD-310, TD-313, TD-470 |
| Segments released/recorded by bare path; sample counts always 0 | TD-333, TD-334 |
| FTP data-connection hijack and protocol gaps | TD-372–TD-377 |

---

## 3. Tests

- `dataset_registry_live_test.cpp` (86 tests, against a running server; skipped when none is reachable): every endpoint's success and validation paths,
  TD-202 kind isolation, TD-205 segments, assign/unassign/delete modes, session numbering, admin config.
- `registry_ftp_confinement_test.cpp` (4 tests): spawns the binary with FTP and checks that out-of-tree entries never get FTP tokens (TD-040).

**Not covered:** group names such as `..` or `.`, concurrent fetches, uploads over claimed files, `trained` from a different run, bind failure.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Unvalidated `<group>`: `..` reaches `data_dir`'s parent; `delete_files` then unlinks files outside the data tree (§2) | Unauthenticated deletion of checkpoints/configs in the working directory; writes outside `data_dir` | **Verified** | [TD-490](../../guides/TECHNICAL_DEBT.md#td-490-registry_server-group-names-escape-data_dir-letting-delete-unlink-outside-it) |
| Fetch cursor read/written without a lock; offset-named outputs (§2) | Concurrent fetches report 8× the pairs actually queued; same file written concurrently; wrap-around overwrites queued files | **Verified** | [TD-491](../../guides/TECHNICAL_DEBT.md#td-491-concurrent-registry-fetches-serve-and-overwrite-the-same-slice) |
| Upload overwrites pending/claimed/trained files and reports `added:true` (§2) | A running trainer's data changes underneath it | **Verified** | [TD-492](../../guides/TECHNICAL_DEBT.md#td-492-registry-upload-silently-replaces-pending-or-claimed-files) |
| `trained` ignores claim ownership; registry dedupes by bare path (§2) | Stale runs mark others' files trained; real results unrecorded | **Verified** | [TD-493](../../guides/TECHNICAL_DEBT.md#td-493-registry-trained-ignores-claim-ownership-and-records-each-path-once) |
| Bind failure exits 0; startup SIGTERM lost; no claim leases; fetches block the thread pool; unbounded uploads; compact-only JSON; `session/next` key and `/health` (§2) | Robustness | **Verified** (exit 0, JSON); by inspection | [TD-494](../../guides/TECHNICAL_DEBT.md#td-494-registry_server-startup-lease-thread-pool-and-parsing-gaps) |
| No authentication | By design (trusted LAN, Cloudflare Access) | — | — |
