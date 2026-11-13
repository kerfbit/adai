# `RegistryTransport` — Source File Reference

- **Files:** [`src/RegistryTransport.hpp`](../../../../src/RegistryTransport.hpp) (671 lines), [`src/RegistryTransport.cpp`](../../../../src/RegistryTransport.cpp) (1,745)
- **Namespace:** global (`DataVersion`, `PendingEntry`, `segment_matches()`, `trained_matches()`, `SegmentTarget`, `PendingFileRange`, `FileToken`, `AcquireResponse`, `download_file_name()`, `registry_path_problem()`, the result types, `RegistryTransport`, `LocalTransport`, `RemoteTransport`)
- **Built into:** `adai_core`. `RemoteTransport`'s HTTP code is compiled only with `BUILD_METRICS_API_SERVER` (set alongside httplib); without it every remote method logs an error and fails
- **Used by:** [`DatasetRegistry`](DatasetRegistry.md) (`build_transport()`: `RemoteTransport` when `REGISTRY_SERVER_URL` is set, else `LocalTransport`), hence `incremental_trainer` and `dataset_manager`; **`registry_server` stores every group's state through `LocalTransport`** ([RegistryServer.md](RegistryServer.md)); `DataTransport` and `StartupSweep` use `download_file_name()`
- **Status tag:** `@adai-status: stable`, `@adai-version: 2.0.0` (hpp and cpp), `@adai-reviewed: 2026-11-12`
- **Tests:** [`tests/registry_transport_test.cpp`](../../../../tests/registry_transport_test.cpp) → `registryTransportTests` (10) and [`tests/RegistryTransportTests.cpp`](../../../../tests/RegistryTransportTests.cpp) → `registryTransportPhase9Tests` (65), all local; `RemoteTransport` against a real `registry_server` in `registryFtpConfinementTests` (which starts one) and `datasetRegistryLiveTests` (106, skipped unless one is running)
- **Last traced against the code:** 2026-11-12

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linking `libadai_core.a`, using a scratch directory for
> `LocalTransport` and a local `registry_server` for `RemoteTransport`.

---

## 1. Data types

| Type | Fields / meaning |
|---|---|
| `DataVersion` | A trained record: `data_file`, `checksum`, `num_samples`, `trained`, `model_id` (never set, §4), `added_utc`, `source`, and since TD-333 `segment_start`/`segment_count` (−1/−1 = the whole file). `trained_matches(dv, path, start, count)` is its identity check: a whole-file query matches only a whole-file record |
| `PendingEntry` | A queue entry: `path`, `run_id` (the claim, set by `acquire()` in either transport), `model_name` (assignment), `source`, `added_utc`, `size_bytes`, `num_entries` (pairs; −1 = unknown), `checksum`, `segment_start`/`segment_count` (TD-205), `claimed_unix` (TD-494: when the claim was made or renewed) |
| `segment_matches(e, path, start, count)` | Identity: same path, and either both whole-file or the exact same range |
| `SegmentTarget`, `PendingFileRange` | `(path, start, count)` addressing one entry / one trainable unit |
| `FileToken`, `AcquireResponse` | An acquired unit (registry path, FTP path and one-time credentials, size, checksum, expiry, range) and the acquire result: `ftp_server_host` (empty = read paths directly), port, `ftps_enabled`, `files`, `failed` (TD-523: the registry couldn't be asked, so empty `files` doesn't mean "nothing pending") and `claim_lease_seconds` (TD-494); `registry_paths()`, `file_ranges()` |
| `download_file_name(registry_path)`, `kDownloadChecksumSuffix` | TD-337/338: a trainer's local name for an FTP download (a 16-hex-digit FNV-1a hash of the full path, `_`, the basename) and its checksum sidecar's suffix (`.checksum`) |
| `registry_path_problem(path)` | TD-496: why a path can't be queued (empty; a C0, DEL or C1 control character; invalid UTF-8, including overlong forms, surrogates and code points above U+10FFFF), or `""` |
| `AssignResult`, `UnassignResult`, `DeleteResult` | Counts plus the whole-file paths and segments touched; delete has a per-target `status` (`deleted`/`skipped_active_run`/`not_found`), `file_deleted` and range |
| `RequeueResult` | TD-540: `requeued`/`already_pending`/`not_found`, per-path details, `ok` |
| `UploadResult` | TD-552: the stored `path` (empty when nothing was stored; `ok()`), `replaced` (overwrite replaced a queued file) and `refused`, `registry_server`'s 409 reason: `claimed`, `segmented`, or without overwrite `trained`/`already_queued` |

---

## 2. The interface (`RegistryTransport`)

| Method | Contract |
|---|---|
| `load_registry` / `save_registry`, `load_pending` / `save_pending` | Whole lists |
| `acquire(run_id, max_files, model_name)` | Claim up to `max_files` (≤ 0 = all) entries that are unclaimed or already this run's, and unassigned or this model's (an empty `model_name` claims only unassigned ones) |
| `release(run_id, paths, segments)` | Free this run's claims on the whole-file entries of `paths` and exactly the `segments`, together (TD-333) |
| `renew_claims(run_id)` | TD-494: restart the lease on every claim the run holds; the default (local, test doubles) returns 0 |
| `commit_trained(run_id, new_entries, trained_paths, trained_segments)` | Record `new_entries` and remove the trained units from the queue. With `trained_segments` (exact mode, TD-333) only the whole-file entries of `trained_paths` and exactly those segments; without, every entry of each path. An empty `run_id` ignores claims |
| `add_pending(path, start, count)` | Queue one unit |
| `next_session(model, run_id)` | `session-01`, `session-02`, … per (model, run), persisted |
| `assign(model, paths, count, segments)` | Modes in order: `segments` (exactly those); `paths` (whole-file entries); `count > 0` (the first N unassigned whole-file entries); `count == 0` (every entry, segments too); `count < 0` refused (TD-321) |
| `unassign(model, paths, force, segments)` | Explicit targets (whole-file paths, exact segments; a non-empty model also filters), or with none every entry assigned to the model; claimed entries are skipped unless `force` |
| `delete_paths(paths, force, delete_files, segments)` | Remove matching pending entries (claimed ones only with `force`) and trained records (always); `delete_files` also unlinks (§3, §4) |
| `fetch_gutenberg`, `fetch_huggingface`, `upload_file(local_path, overwrite)` | Remote only: the server downloads or receives the file into its own data directory and queues it |
| `requeue_trained(paths, all, model)` | TD-540: move trained records (of `paths`, or all) back to the queue as the units they record |

---

## 3. `LocalTransport(registry_path, pending_path)`

**Files** (TD-496): JSON Lines. The registry file starts `# adai data registry, JSON Lines v2` and the queue
`# adai pending queue, JSON Lines v2`; then one object per entry with every field of `DataVersion`/`PendingEntry` named, so spaces, tabs
and newlines in a path survive. A line that isn't a JSON object is logged and skipped; a field of the wrong type takes its default.
Strings are dumped with invalid UTF-8 replaced by U+FFFD, so an old entry with such a path can't make a save throw. A file **without** the
header is read in the old format (a space-separated registry with `-` for an empty value and optional trailing columns; a tab-separated
queue with up to 11 positional columns) and is rewritten as JSON Lines by the next save, so an upgrade needs no migration step. Older
binaries can't read the new files.

**Saves** (TD-495) build the whole file in memory and write it with `adai::write_file_atomically()` ([AtomicFile.md](AtomicFile.md)):
a failed write leaves the old file whole, logs the reason and returns `false`. Every caller acts on that: `acquire()` claims nothing and
sets `failed`; `assign()`/`unassign()` report nothing changed; `add_pending()` returns false; `requeue_trained()` sets `ok = false`;
`release()`/`commit_trained()`/`delete_paths()` log an error. Where two files change (queue, then registry), a failed second save leaves
the first changed: a unit then ends up both pending and trained, so it trains again rather than being lost.

**Locking:** every operation that changes the queue (`acquire`, `release`, `commit_trained`, `add_pending`, `assign`, `unassign`,
`delete_paths`, `requeue_trained`) holds an exclusive lock on `<pending>.lock` (`flock()`, or `_locking()` on Windows) for its
read-modify-write. `commit_trained()` loads and saves the registry inside that lock; `delete_paths()` and `requeue_trained()` read the
registry before taking it and save it after releasing it (TD-498). `load_*`/`save_*` themselves take no lock.

| Method | Behaviour |
|---|---|
| `acquire` | Under the lock, claims eligible entries in queue order up to the limit, records `claimed_unix` (display only: standalone claims have no lease), saves, and returns tokens with paths and ranges (no FTP). No queue file → nothing |
| `release` | Clears this run's claim on matching entries and logs the number actually released |
| `commit_trained` | TD-493: with a run id, records a new entry only when its unit is pending and claimed by that run (a whole-file record by any claimed entry of the path, a segment by its exact entry), logging how many it skipped; without one, records everything. Appends without removing an earlier record of the same unit (TD-498). Then removes the trained units from the queue as in §2 |
| `add_pending` | Refuses a path `registry_path_problem()` rejects; otherwise appends the unit unless it is already queued, rewriting the queue atomically. Returns **true** for an already-queued unit (TD-498: the header and `RemoteTransport` say false) |
| `next_session` | `SessionCounterStore` in `session_counters.db` beside the queue, opened on first use: one atomic increment per (model, run), so standalone trainers sharing it never get the same number (TD-304, TD-308). If the store can't be opened, `session-01` with a warning |
| `assign` / `unassign` | §2's modes; whole-file targets match only whole-file entries, and bulk modes include segments |
| `delete_paths` | A whole-file target removes its whole-file pending entry and **every** trained record of the path; a segment target exactly its own entry and record (TD-333). `delete_files` unlinks a whole-file target's file if it exists (local mode can reach any path), never for a segment target and never the transport's own two files |
| `requeue_trained` | Each trained record of the targets becomes a pending entry for its unit (keeping source, added time, checksum; assigned to `model` if given), or counts as `already_pending` if that unit is queued; the record is removed either way. Saves the queue first, then the registry |
| `fetch_*`, `upload_file` | Unsupported: a warning and an empty result |

---

## 4. `RemoteTransport(base_url, run_group, timeout_ms = 5000, dataset_kind = "")`

`parse_host_port()` takes `http://` or `https://` off the URL (and any path) and reads `host[:port]`, default port 8082; every request is
then plain HTTP (TD-498). Requests go to `/registry/<run_group>[/<kind>]/…`; the group and kind are put into the URL as given (the server
validates the group, TD-490). Each call opens a new `httplib::Client` with connection and read timeouts of `timeout_ms` (verified: an
unreachable host failed after 2.00 s with a 2 s timeout); the fetches and uploads allow at least 120 s to read (and upload to write).
Nothing is retried.

**JSON** (TD-497): every body is an `nlohmann::json` object dumped with invalid UTF-8 replaced, and every reply is parsed with
`nlohmann::json::parse()` and read by key with type checks (a field of the wrong type takes its default). A reply that isn't a JSON object
is logged and treated as a failure. Paths containing `}`, `{`, `"`, backslashes or non-ASCII text round-trip through every call (they used
to vanish, be cut short or come back with each backslash quadrupled, verified).

| Method | Request | Result |
|---|---|---|
| `load_registry` / `load_pending` | `GET …/registry`, `GET …/queue` | `false` on any failure; entries without a path are skipped |
| `save_registry` / `save_pending` | — | No-ops returning `false`: the server owns its state |
| `acquire` | `POST …/acquire {run_id, max_files, model_name}` | Reads both reply shapes: with FTP, `files` tokens and the FTP host, port and `ftps_enabled`; without, the `acquired` path list with parallel `segment_starts`/`segment_counts`. `claim_lease_seconds` from either. `failed` on an HTTP error or unreadable reply |
| `release` | `POST …/release {run_id, files, segments}` | Logs a failure |
| `renew_claims` | `POST …/renew {run_id}` | `renewed`, or −1 |
| `commit_trained` | `POST …/trained {run_id, files, samples}` (one aligned entry per unit: the new entries with their counts, then trained units already recorded so their pending entries are still removed), plus `segment_starts`/`segment_counts` in exact mode | Logs a failure, and warns when the server lists units it `rejected` (TD-493). `model_id` isn't sent (TD-498) |
| `add_pending` | `POST …/pending/add {path, segment_start, segment_count}` | `true` when added; `false` (a warning) when the server says it was already queued, or on failure |
| `next_session` | `POST …/session/next {model_name, run_id}` | `session_id`; `session-01` on an HTTP error, `""` on an unreadable reply |
| `assign` / `unassign` | `POST …/assign {model_name, paths, count, segments}`, `…/unassign {…, force, segments}` | Counts and `paths`; the reply's `segments` aren't read, so `result.segments` stays empty while the counts include them (TD-498) |
| `delete_paths` | `POST …/delete {paths, force, delete_files, segments}` | Counts and per-target details (status, `file_deleted`, range) |
| `fetch_gutenberg` / `fetch_huggingface` | `POST …/fetch/gutenberg {book_id, num_pairs, model_name}`, `…/fetch/huggingface {dataset_id, num_pairs, split, input_field, output_field, model_name}` | The queued path when the reply says `added`, else `""` |
| `upload_file(local_path, overwrite)` | Reads the whole file into memory (TD-498), then `POST …/upload?filename=<basename>` with the bytes, adding `&overwrite=true` when asked | TD-552: a 409 sets `refused` from the reply's `reason` (`conflict` when it has none) with a warning; another error is logged and returns an empty result. A 200 succeeds when the reply says `added` **or** `replaced`, since a replaced queued file answers `{"added":false,"replaced":true}` |
| `requeue_trained` | `POST …/requeue {paths, all, model_name}` | Counts and details; `ok = false` on failure |

---

## 5. Tests

| Suite | Covers |
|---|---|
| `registry_transport_test.cpp` (10) | `LocalTransport` round trips, missing files, directory creation, comments and blank lines, overwriting, `model_id` |
| `RegistryTransportTests.cpp` (65) | `LocalTransportPhase9Test` (60): acquire (limits, claims, model eligibility), release (a wrong run, a segment), `commit_trained` (recording, the TD-493 claim check, a unit trained again, `model_id`), assign/unassign/delete modes, guards and segment targeting, `next_session()` (per run and model, persisted), failed saves under a file-size limit (TD-495), JSON Lines round trips of paths with spaces, tabs and newlines, old-format files rewritten, unreadable lines skipped, `registry_path_problem()` at `add_pending()`, a negative assign count, and the fetch/upload stubs. `SegmentMatchesTest` (3), `RegistryPathProblemTest` (2) |
| `registryFtpConfinementTests` | `RemoteTransport` against a `registry_server` it starts: paths with JSON syntax through every call (TD-497), acquire's out-of-tree confinement (TD-040), a negative assign count, and `upload_file()` through `dataset_manager add --overwrite` (TD-552) |
| `datasetRegistryLiveTests` (106) | The remote surface against a running `registry_server`, through `DatasetRegistry` and raw HTTP; skipped when none is running |

`requeue_trained()` is tested through `DatasetRegistryTests` ([DatasetRegistry.md](DatasetRegistry.md)).

**Not covered:** releasing paths and segments in one call; concurrent local processes (the unlocked registry half of
`delete_paths()`/`requeue_trained()`).

---

## 6. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `LocalTransport::commit_trained()` appends without dedupe; `add_pending()` returns true for a duplicate where `RemoteTransport` and the header say false | A unit trained twice gets two records; inconsistent results | **Verified** | [TD-498](../../guides/TECHNICAL_DEBT.md#td-498-localtransportremotetransport-divergence-dead-model_id-and-client-hygiene) |
| `delete_paths()`/`requeue_trained()` read and save the registry outside the lock | A concurrent commit can lose a record | By inspection | TD-498 |
| Remote `assign`/`unassign` don't read the reply's `segments` | `result.segments` empty while the counts include them | By inspection | TD-498 |
| `model_id` never set or sent; `https://` sent as plain HTTP; group and kind not URL-encoded; uploads read the whole file into memory | A dead history filter; a misleading URL; memory for large uploads | By inspection | TD-498 |
| `next_session()` answers `""` for an unreadable reply but `session-01` for an HTTP error | A pass can continue with an empty session id | By inspection | — |

---

## 7. History

- **TD-552 (November 12, 2026):** `upload_file(local_path, overwrite)` and `UploadResult`; a replacement is a success. The same day the
  header's comments on `PendingEntry::run_id`, `release()`, `requeue_trained()` and `LocalTransport`'s file formats, which described the
  code before TD-333 and TD-496, were corrected.
- **TD-495–TD-497 (November 4–5, 2026):** atomic saves (a 4 KiB limit used to keep 114 of 2,001 queue entries); JSON Lines files (a
  space-separated registry read `/data/my file.jsonl` back as an untrained `/data/my`); `registry_path_problem()`; nlohmann/json in
  `RemoteTransport` (substring parsing hid paths containing `}` and multiplied backslashes, verified). `load_registry()` reads
  `model_id` since.
- **TD-304, TD-308 (November 3, 2026):** atomic session numbers in their own store.
- **TD-321 (November 5, 2026):** a negative assign count is refused (it meant "all", like 0).
- **TD-493, TD-494 (October 24–25, 2026):** claim-checked local commits (both sides used to record another run's unit) and the remote
  `rejected` warning; claim leases (`claimed_unix`, `claim_lease_seconds`, `renew_claims()`).
- **TD-333 (October 19, 2026):** segments in trained records, exact-mode commits, releases that apply paths and segments together,
  segment-aware requeue and delete.
- **TD-337, TD-338, TD-523 (October 19, 2026):** `download_file_name()`, the checksum sidecar suffix, `AcquireResponse::failed`.
- **TD-540 (October 18, 2026):** `requeue_trained()`.
