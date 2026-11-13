# `DatasetManagerTool` — Source File Reference

- **File:** [`src/DatasetManagerTool.cpp`](../../../../src/DatasetManagerTool.cpp) (1,175 lines; `main()` of the `dataset_manager` binary)
- **Built into:** `dataset_manager` (with `DatasetManagerArgs.cpp` and `Config.cpp`, linking `DataFetcher`, `DatasetRegistry` and `adai_core`; `ModelNameClient` when `BUILD_MNS_SERVER`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.10.0`, `@adai-reviewed: 2026-11-12`
- **Tests:** `registryFtpConfinementTests` runs the real binary against a `registry_server` it starts (11 cases, §5); the parsers are tested by `datasetManagerArgsTests` ([DatasetManagerArgs.md](DatasetManagerArgs.md)) and `DatasetRegistry` by its own suites
- **Operator guide:** [dataset-manager-guide.md](../../../operations/guides/dataset-manager-guide.md) (§6)
- **Last traced against the code:** 2026-11-12

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real
> `dataset_manager` binary against scratch local pools and a local `registry_server`.

---

## 1. What this file is

The **operator CLI for the dataset pool**: queue files and downloads, inspect the queue, assign or
unassign data to models, delete and re-queue entries, migrate legacy entries into per-kind sub-pools
(TD-202), and split files into row-range segments (TD-205). It's one `main()` with an `if/else` chain
over commands. Parsing lives in `DatasetManagerArgs`, storage in `DatasetRegistry` (local flat files,
or HTTP to `registry_server`), and local downloads in `DataFetcher`.

### Why it matters

It and the Android ops dashboard are the two ways operators change **which data trains which
model**. Several commands rewrite the shared pending pool, so its success messages and exit codes are
what operators trust.

---

## 2. Startup

1. **Global options**, only before the command (TD-330): `--config <path>`, `--kind <kind>` (each
   needs a value: `Missing value for …`), and `--help`/`-h`. Any other option there is an error; the
   first argument that isn't an option is the command, and everything after it belongs to the command.
   `help` as the command also prints the usage.
2. `--kind` is checked with `is_valid_dataset_kind()`.
3. **Config:** `discover_config_path(config_path, "config.registry.conf")` (`--config` >
   `./config.registry.conf` > `/etc/adai/config.registry.conf` > the legacy `config.conf` files), then
   `ConfigLoader::load()`. `--kind` overrides `DATASET_KIND` (TD-202).
4. **No command or `--help`** → the usage text (stdout, exit 0).
5. `status`, `list-pending`, `list-trained`, `clear-pending` and `models` refuse arguments.
6. **Run group** (TD-329): for every command but `models`, `resolve_run_group()` settles the group once.
   The model is the one the command names (`assign`, `unassign`, `requeue --model`, via
   `run_group_model()`), otherwise `MODEL_NAME`; a fetch command's `[model]` is only the registry's
   slice-cursor key. With `NAME_SERVICE_URL` and a model, `ModelNameClient::lookup_model()` reads the
   model's record in any state, and its `run_group` wins when it has one. An unregistered model, or one
   with no group, keeps `RUN_GROUP` (or, unset, the `SESSION_DIR` name). An MNS that doesn't answer
   stops the command, exit 1, nothing changed. `assign` also requires the model to be registered and
   prints `Verified model: …`. Against a `registry_server`, the group and where it came from go to
   stderr (`run_group: 'chatbot-v1' (MNS record of model 'm1')`), so listings stay parseable. A local
   pool has no group, so local commands other than `assign` don't contact MNS.
7. Each command builds its own `DatasetRegistry(DatasetRegistry::make_config(svc_config))`:
   `REGISTRY_SERVER_URL` set means `RemoteTransport`, otherwise `LocalTransport` under
   `SESSION_DIR[/<kind>]`.

**Local paths** (TD-328): `add_file()` stores files absolute, and a relative file target of `remove`,
`assign`, `unassign`, `delete`, `requeue`, `migrate` or `segment` is made absolute the same way
(`local_target()`), unless that exact string is already queued or trained (an entry queued relative
before TD-328). Remote paths are the server's and are used as given.

---

## 3. Commands

| Command | Local mode (`LocalTransport`) | Remote mode (`registry_server`) | Exit 1 when |
|---|---|---|---|
| `add [--overwrite] <file>` | `add_file()`: the file must exist; skipped (with a warning) if already trained or queued. `--overwrite` is an error | `remote_upload(file, overwrite)`: uploads the bytes to the server's `uploads/` and prints `Uploaded as …` or `Replaced the queued file …`. A refusal is worded by reason (TD-552): already trained or queued (use `--overwrite`), claimed (wait or release), segmented (delete the segments) | add or upload fails or is refused |
| `gutenberg <id> [n] [model]` | `DataFetcher::fetch_gutenberg()` then `add_file()` | `remote_fetch_gutenberg()` (server-side, per-model rotating slice) | fetch fails |
| `gutenberg-batch <ids> [n] [model]` | `fetch_gutenberg_batch()`, each result `add_file()`d | One `remote_fetch_gutenberg()` per id | none added |
| `huggingface <id> [n] …` | `fetch_huggingface()` then `add_file()` | `remote_fetch_huggingface()` | fetch fails |
| `status` | Kind, pending/trained counts, total samples, `print_registry()`, then each pending entry with its segment range or pair count and model | same | — |
| `list-pending` / `list-trained` | `# Kind: …`, then tab-separated `path`, `segment[a-b]` or `N pairs`, model | same | — |
| `remove <file>` | Checks the file is queued, then `remove_pending()`; a failed save is reported separately | Refused (TD-327, `refuse_local_only_command()`): points at `delete` | not queued, save failed, or remote |
| `clear-pending` | Clears and saves | Refused (TD-327) | save failed, or remote |
| `assign <model> [files] [--count N]` | `assign_model()`, listing the files picked when none were named | same | no pending files, or nothing assigned |
| `unassign <model> [files] [--force]` | `unassign_model()`; reports claimed entries it skipped | same | nothing unassigned **and** nothing skipped |
| `delete <files> [--force] [--delete-files]` | `delete_entries()`, a status per path (and `[file removed]`/`[file kept]` with `--delete-files`) | same | nothing deleted |
| `requeue (--all \| <files>) [--model M]` | TD-540: `requeue_trained()`, a status per path | same, server-side | the request failed, or nothing requeued or already pending |
| `migrate <kind> [files] [--count N]` | Legacy pool → `<kind>` pool, entry by entry (§4) | same | no legacy entries, none matched, or none moved |
| `segment <path> --count N` / `--ranges …` | §4 | §4 | no segment created, or a check or the replacement failed |
| `models` | `ModelNameClient::list_models()`; needs `NAME_SERVICE_URL` (TD-330) | — | unset, query fails, or MNS support not compiled |

**Pair counts** (TD-330): fetch commands report the pairs actually queued (`queued_pairs()`: the queue
entry's `num_entries`, which `registry_server` sets from a fetch's `pairs_written`, or a local count),
naming the requested number only when it wasn't reached.

---

## 4. `migrate` and `segment`

**`migrate`** moves legacy (unkinded) entries into `<kind>`'s pool. Its source is always the legacy
pool; a global `--kind` is ignored with a warning. File targets move every unit of each file (its whole
entry and each segment); `--count N` takes the first N entries; neither moves all. Each entry moves
exactly (TD-326): add it to the destination with its segment range, restore its model by segment
(`assign_model()`), then delete it from the source by segment with `force`. Each step is checked, and a
failure deletes the destination copy again so nothing is left in both pools. It prints
`path [start+count] -> kind` per entry and `Migrated N/M`.

**`segment`** splits a JSONL file's pairs into entries for `--count N` near-equal parts (the first
`pairs % N` get one extra) or the explicit `--ranges`.
1. **The pair count.** Locally, `count_pairs()` on the file. With a registry server (TD-543): a path
   queued there as a whole file uses the server's `num_entries` (the client needn't read it); a local
   file is counted, uploaded as `add` does (`remote_upload()`; a refusal names the reason and says to
   segment the server path instead), and the uploaded path is segmented; anything else exits 1
   untouched. A file with no pairs (legacy `INPUT:`/`RESPONSE:` files) can't be segmented.
2. **Ranges** past the last pair are refused before anything is queued (TD-324).
3. **The whole entry** of the same path is replaced (TD-542), so the file isn't queued and trained both
   ways. A claimed whole entry, or a path that also has trained records (the whole-path delete would
   erase them), is refused before queuing.
4. Each range is queued with `add_pending_path_unchecked()` (the pairs were counted already, maybe by
   the server, so no local existence check). Then the whole entry is deleted with `force=false`; if
   that doesn't remove exactly one entry, the new segments are deleted again and it exits 1. The
   segments take the whole entry's model.

The Android `RegistryRepository.createSegments()` follows the same steps.

---

## 5. Tests

`registryFtpConfinementTests` (`tests/registry_ftp_confinement_test.cpp`) starts a real
`registry_server` on test-owned ports and data directory, and runs the real binary:

| Test | Checks |
|---|---|
| `DatasetManagerRemoveRefusesInRemoteMode` | `remove`/`clear-pending` exit 1 remotely and leave the entry (TD-327) |
| `DatasetManagerUsesTheModelsMnsRunGroupForEveryCommand`, `DatasetManagerRefusesWhenTheNameServiceDoesNotAnswer` | The MNS run group, against a stand-in MNS (TD-329) |
| `DatasetManagerSegmentUploadsALocalFileFirst`, `DatasetManagerSegmentsAServerPathByItsServerPairCount`, `DatasetManagerSegmentRefusesAnUnknownRemotePath` | Remote `segment` (TD-543) |
| `DatasetManagerQueuesLocalFilesAbsolute` | Local paths stored absolute and found again (TD-328) |
| `DatasetManagerGlobalOptionsAndHelp`, `DatasetManagerReportsThePairsActuallyQueued` | TD-330 |
| `DatasetManagerAddOverwriteReplacesQueuedAndTrainedFiles`, `DatasetManagerAddOverwriteNeedsARegistryServer` | `add --overwrite` against queued, trained and claimed files, and locally (TD-552) |

**Not covered:** `gutenberg*`/`huggingface` (they download), `assign`/`unassign`/`delete`/`requeue`
output beyond the MNS case, `migrate`, `status`/`list-*` formatting, and local-mode `segment` (TD-331).

---

## 6. Operator guide

[dataset-manager-guide.md](../../../operations/guides/dataset-manager-guide.md) has sections for `add`,
`gutenberg`, `gutenberg-batch`, `huggingface`, `status`, `list-pending`/`list-trained`, `clear-pending`
(which also explains `remove`), `segment` and `requeue`, plus the global options and the run group.
`assign`, `unassign`, `delete`, `migrate` and `models` have no sections (TD-332); `main()`'s usage text
is their only full description.

---

## 7. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `unassign` exits 0 when every match was skipped as claimed | A no-op looks like success unless the "skipped" line is read | By inspection | [TD-330](../../guides/TECHNICAL_DEBT.md#td-330-dataset_manager-output-exit-code-and-flag-hygiene) (not in its action items) |
| Commands without tests (§5) | Composition bugs go unnoticed (`migrate` and `remove` had them) | By inspection | [TD-331](../../guides/TECHNICAL_DEBT.md#td-331-dataset_manager-has-no-tests-for-its-commands) |
| The guide lacks `assign`, `unassign`, `delete`, `migrate`, `models` (§6) | Pool-changing commands documented only in the usage text | By inspection | [TD-332](../../guides/TECHNICAL_DEBT.md#td-332-operator-guide-covers-only-8-of-15-dataset_manager-commands) |
| `migrate --count N` counts entries, not files | A file's segments may be split between pools | By inspection | — |

---

## 8. History

- **TD-552 (November 12, 2026):** `add --overwrite`; refusals worded by reason; a replacement reported
  as such. Remote `add` couldn't replace a file at all before.
- **TD-327–TD-330, TD-543 (November 7, 2026):** `remove`/`clear-pending` used to print success and exit
  0 against `registry_server` while nothing changed (verified); one run group for every command (only
  `assign` used MNS's, through `resolve_model()`, which MNS refuses while a model is initializing);
  absolute local paths (a relative entry couldn't be found from another working directory, verified);
  actual pair counts and stricter global options (they were stripped from anywhere in argv, and there
  was no `--help`); `segment` against a remote registry (it queued the client's local path, verified).
- **TD-542 (October 19, 2026):** `segment` kept a queued whole entry beside its segments, so every pair
  trained twice.
- **TD-326 (October 19, 2026):** `migrate` copied entries without their segment ranges and ignored its
  delete and assign results. A 30-pair file in 3 segments put the whole file in `chatbot`, left all 3
  segments in the legacy pool and printed "Migrated 1/3"; training both pools trained every pair twice.
- **TD-540 (October 18, 2026):** `requeue`.
- **TD-321–TD-325 (November 2026):** parser fixes; see [DatasetManagerArgs.md](DatasetManagerArgs.md) §4.
