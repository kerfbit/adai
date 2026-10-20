# `RegistryTransport` — Source File Reference

- **Files:** [`src/RegistryTransport.hpp`](../../../../src/RegistryTransport.hpp) (651 lines), [`src/RegistryTransport.cpp`](../../../../src/RegistryTransport.cpp) (1,744 lines)
- **Namespace:** global (`DataVersion`, `PendingEntry`, `segment_matches()`, `SegmentTarget`, `PendingFileRange`, `FileToken`, `AcquireResponse`, the three result types, `RegistryTransport`, `LocalTransport`, `RemoteTransport`)
- **Built into:** `adai_core`. `RemoteTransport`'s HTTP code is compiled only with `BUILD_METRICS_API_SERVER` (set alongside httplib); without it every remote method logs an error and fails
- **Used by:** [`DatasetRegistry`](DatasetRegistry.md) (`build_transport()`: `RemoteTransport` when `REGISTRY_SERVER_URL` is set, else `LocalTransport`), hence `incremental_trainer` and `dataset_manager`; **`registry_server` stores every group's state through `LocalTransport`** ([RegistryServer.md](RegistryServer.md))
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.6.0` (hpp) / `1.4.0` (cpp), `@adai-reviewed: 2026-10-19`
- **Tests:** [`tests/registry_transport_test.cpp`](../../../../tests/registry_transport_test.cpp) (10, `registryTransportTests`) and [`tests/RegistryTransportTests.cpp`](../../../../tests/RegistryTransportTests.cpp) (51, `registryTransportPhase9Tests`), all `LocalTransport`; `RemoteTransport` is exercised by `datasetRegistryLiveTests` (87, against a running server). All pass
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linking `libadai_core.a`, using a scratch directory for
> `LocalTransport` and a local `registry_server` for `RemoteTransport`.

---

## 1. What this file is

The storage interface behind the dataset queue, with one local and one HTTP implementation.

### Data types

| Type | Fields / meaning |
|---|---|
| `DataVersion` | A trained-registry record: `data_file`, `checksum`, `num_samples`, `trained`, `model_id`, `added_utc`, `source`, and since TD-333 `segment_start`/`segment_count` (−1/−1 = the whole file); `trained_matches(dv, path, start, count)` is its identity check |
| `PendingEntry` | A queue entry: `path`, `run_id` (claim), `model_name` (assignment), `source`, `added_utc`, `size_bytes`, `num_entries`, `checksum`, `segment_start`/`segment_count` (TD-205; −1/−1 = whole file) |
| `segment_matches(e, path, start, count)` | Identity: same path, and either both whole-file or the exact same range |
| `SegmentTarget`, `PendingFileRange` | `(path, start, count)` addressing one entry / one trainable unit |
| `FileToken`, `AcquireResponse` | An acquired file (path, FTP path and one-time credentials, size, checksum, expiry, range) and the acquire result (`ftp_server_host` empty = read paths directly); `registry_paths()`, `file_ranges()`. `AcquireResponse::failed` (TD-523) is set when the registry couldn't be asked (HTTP error, unreachable, lock failure), so an empty `files` isn't mistaken for "nothing pending" |
| `download_file_name(registry_path)`, `kDownloadChecksumSuffix` | TD-337/338: a trainer's local name for an FTP download (a 16-hex-digit FNV-1a hash of the full path, `_`, the basename) and its checksum sidecar's suffix (`.checksum`); shared by `DataTransport` and `StartupSweep` |
| `RequeueResult` | TD-540: `requeued`/`already_pending`/`not_found` counts, per-path details, `ok` |
| `AssignResult`, `UnassignResult`, `DeleteResult` | Counts plus the exact entries touched; delete has per-target `status` (`deleted`/`skipped_active_run`/`not_found`) and `file_deleted` |

### Interface (`RegistryTransport`)

`load_registry`/`save_registry`, `load_pending`/`save_pending`, `acquire(run_id, max_files, model_name)`, `release(run_id, paths, segments)`,
`commit_trained(run_id, new_entries, trained_paths)`, `add_pending(path, start, count)`, `next_session(model_name, run_id)`,
`assign`/`unassign`/`delete_paths` (modes as in [RegistryServer.md](RegistryServer.md)), and `fetch_gutenberg`/`fetch_huggingface`/`upload_file`.

### `LocalTransport(registry_path, pending_path)`

| Part | Behaviour |
|---|---|
| Registry file | One line per record, **space-separated**: `data_file checksum num_samples trained model_id added_utc source segment_start segment_count` (`-` for an empty string field, including an empty checksum since TD-333, so later columns don't shift); `#` comments skipped; lines without the two segment columns (pre-TD-333) read as whole files |
| Pending file | One line per entry, **tab-separated**, ten positional columns (older, shorter lines load with defaults) |
| `save_*` | `std::ofstream` truncates the file and rewrites it in place; returns `file.good()` |
| Locking | `acquire`, `release`, `commit_trained` (pending half), `add_pending`, `assign`, `unassign`, `delete_paths` take an exclusive `flock` on `<pending>.lock` (`_locking()` on Windows). `load_*`/`save_*` and the registry halves of `commit_trained`/`delete_paths` take none |
| `acquire` | Claims entries that are unclaimed or already this `run_id`'s, and unassigned or this model's; returns paths only |
| `release` | Clears the claim on this `run_id`'s entries matching the whole-file `paths` (whole-file entries only) **and** the exact `segments`, in one call (TD-333; it used to apply only one list) |
| `commit_trained(run_id, new_entries, paths, segments)` | **Appends** `new_entries` to the registry (no dedupe, TD-498), then removes from pending the entries this claim holds: with `segments` (exact mode, TD-333), exactly the whole-file `paths` and those segments; without, every entry of each path (pre-TD-333) |
| `requeue_trained(paths, all, model)` | TD-540: moves each trained record of the listed paths (or all) back to pending as its own unit, the whole file or that segment |
| `add_pending` | Appends a line; returns **true** if the entry was already queued |
| `next_session` | Lazily opens `session_counters.db` next to the pending file (TD-304) |
| `delete_paths` | Removes matching pending entries and trained records: a whole-file target removes every record of the path, a segment target exactly that segment's (TD-333). Unlinks any existing path with `delete_files` (local mode has full filesystem access) except its own two state files |
| Fetch/upload | Unsupported: log a warning, return `""` |

### `RemoteTransport(base_url, run_group, timeout_ms = 5000, dataset_kind = "")`

Parses `http[s]://host[:port]` (default port 8082) and builds `/registry/<run_group>[/<kind>]`. Each call opens a new `httplib::Client` with the
connection and read timeouts set to `timeout_ms` (verified: an unreachable host failed after 2.00 s with a 2 s timeout); fetch and upload allow at least
120 s. Request bodies are built by string concatenation with `json_escape()`; responses are parsed with substring helpers (`json_string()`, `json_int()`,
`json_bool()`, string/int arrays) and by slicing `{…}` objects at the next `}`. `save_registry()`/`save_pending()` are no-ops returning false (the server
owns state; clients use the specific operations). A failed request is logged and returns an empty/false result; nothing is retried.

---

## 2. Problems

### A failed save truncates the queue (verified)

`save_pending()`/`save_registry()` truncate the file and rewrite it in place, with no temporary file and rename. If the write fails part-way (disk full,
quota) or the process dies during it, the file is left truncated. Simulated with `RLIMIT_FSIZE` = 4 KiB: rewriting a 2,001-entry pending file
(74,890 bytes) returned false and left **114 of 2,001 entries**. `registry_server` ignores the return value, so on a full disk every group's queue and
trained registry can be cut short by the next request, silently. Trained files dropped from the registry can then be queued and trained again; pending
files are simply lost.

### Paths with spaces corrupt the trained registry (verified)

The registry format is space-separated and read with `>>`. A record for `/data/my file.jsonl` loaded back as `data_file='/data/my'`,
`checksum='file.jsonl'`, `num_samples=0`, `trained=false`. `registry_server`'s upload accepts filenames with spaces, so any such upload that gets trained
is recorded wrongly, can't be matched by `delete` or the dedupe in `trained`, and shows as untrained. (The tab-separated pending format has the same
problem for paths containing tabs or newlines.)

### `RemoteTransport` mis-parses paths containing `}`, `\` or `"` (verified)

- **`}` (or `{`):** objects are cut at the next `}`, even inside a string. After `add_pending("/data/a}b.jsonl")`, `load_pending()` (`dataset_manager
  status`) returned **2 of 3** entries; the third was invisible. The legacy acquire response (a string array) still returned it, but the FTP acquire
  response is parsed as objects, so with FTP enabled such a file would be claimed by the server and dropped by the client, staying claimed and never
  trained (by inspection).
- **`\` and `"`:** neither side unescapes. `add_pending("C:\data\w.jsonl")` was stored by the server with doubled backslashes, and `load_pending()`/
  `acquire()` returned `C:\\\\data\\\\w.jsonl` (each backslash quadrupled after one round trip), a path that doesn't exist. Windows paths are affected
  directly (Windows builds are supported, TD-160).

### Local and remote implementations disagree (verified, except where noted)

- **`commit_trained`:** `LocalTransport` appends without checking: committing the same file from run `other` (while claimed by `r1`) and then from `r1`
  left **two registry rows** for it. The server deduplicates by path and also ignores the claim (TD-493). Neither checks that the committing run owns the
  claim.
- **`add_pending` duplicate:** `LocalTransport` returns true ("already queued"), `RemoteTransport` false; the header documents false.
- **`assign`/`unassign` results (by inspection):** `RemoteTransport` doesn't parse the response's `segments`, so `result.segments` is always empty while
  `assigned`/`unassigned` count them.
- **Locking (by inspection):** `commit_trained()` and `delete_paths()` read-modify-write the registry without the lock, so two local processes committing
  at once can lose a record; `load_*`/`save_*` are never locked.

### Smaller items (by inspection)

- **`model_id` is never sent:** `RemoteTransport::commit_trained()` sends `run_id`, `files` and `samples` only, and nothing sets `DataVersion::model_id`
  anyway, so the server's `history?model_id=` filter (used by the Android dashboard) never matches.
- **`https://` is accepted and stripped**, then plain HTTP is sent to port 8082 (as TD-444/TD-451 for the other clients).
- **`run_group` and `dataset_kind` aren't URL-encoded or validated** (a `..` group reaches TD-490).
- **Logs report requested, not actual, counts** in `LocalTransport::release()`.
- **`upload_file()` reads the whole file into memory.**

---

## 3. Tests

`registry_transport_test.cpp` (10): `LocalTransport` round trips, missing files, parent directory creation, comments, overwrite, `model_id` and old
formats. `RegistryTransportTests.cpp` (51): acquire (limits, claims, model eligibility), release, commit, legacy formats, fetch stubs,
assign/unassign/delete modes and guards, and TD-205 segment identity and targeting. `RemoteTransport` is covered end to end by
`datasetRegistryLiveTests`.

**Not covered:** failed writes, paths with spaces or special characters, duplicate or foreign-run commits, concurrent local processes.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| In-place truncating saves; a failed write leaves a partial file (§2) | Disk full: 114 of 2,001 queue entries survived; `registry_server` loses queues/registries silently | **Verified** | [TD-495](../../guides/TECHNICAL_DEBT.md#td-495-localtransport-saves-truncate-the-queue-when-a-write-fails) |
| Space-separated registry format (§2) | Uploaded files with spaces are recorded as untrained, unmatchable records | **Verified** | [TD-496](../../guides/TECHNICAL_DEBT.md#td-496-registry-flat-file-formats-break-on-spaces-tabs-and-newlines-in-paths) |
| `RemoteTransport` parsing: `}` hides entries, backslashes multiply, no unescaping (§2) | Invisible queue entries; Windows paths unusable; FTP claims stranded | **Verified** (`}`, `\`); by inspection (FTP) | [TD-497](../../guides/TECHNICAL_DEBT.md#td-497-remotetransport-hides-paths-with--and-multiplies-backslashes) |
| Local/remote divergence: duplicate commits, `add_pending` result, segment results, unlocked registry writes (§2) | Duplicate records; misleading results; lost records under concurrency | **Verified** (commit, add); by inspection | [TD-498](../../guides/TECHNICAL_DEBT.md#td-498-localtransportremotetransport-divergence-dead-model_id-and-client-hygiene) |
| `model_id` never sent or set; `https://` → HTTP; unencoded group; log counts; in-memory upload (§2) | Dead history filter; confusing setup | By inspection | [TD-498](../../guides/TECHNICAL_DEBT.md#td-498-localtransportremotetransport-divergence-dead-model_id-and-client-hygiene) |
| Session counters; server-side claim checks | Already tracked | — | TD-304, TD-493 (claims by bare path, TD-333, and zero sample counts, TD-334, fixed October 19, 2026) |

## Addendum (October 18, 2026): `requeue_trained()`

`RegistryTransport::requeue_trained(paths, all, model_name)` moves trained `DataVersion` entries (the listed paths, or all) back to the pending queue as `PendingEntry`s (whole files, or since TD-333 each trained segment as that segment). Each keeps its source, added time and checksum, and is optionally assigned to `model_name`. A path that is already pending only loses its trained record (`already_pending`).
- `LocalTransport` does it under the pending lock.
- `RemoteTransport` sends `POST /registry/<group>[/<kind>]/requeue` and returns `RequeueResult::ok = false` if the request fails.

Added for the remote-first rule (the trainer takes its data from `registry_server`), so re-queuing happens where the data lives. It's tested in `DatasetRegistryTests` (`Requeue*`, `LocalTransportRequeueTest`) and was checked end to end against a real `registry_server`.

## Addendum (October 19, 2026): segments in trained records (TD-333)

- **Local registry file:** two trailing `segment_start segment_count` columns. A line without them is a whole-file record.
- **`commit_trained(…, trained_segments)`:** in exact mode it removes exactly the trained units. `RemoteTransport` sends aligned `segment_starts`/`segment_counts`.
- **`release()`:** applies paths and segments together.
- **`requeue_trained()`:** restores segment records as segments.
- **`trained_matches()`:** the identity check for a trained record.
