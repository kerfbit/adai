# `DatasetManagerTool` — Source File Reference

- **File:** [`src/DatasetManagerTool.cpp`](../../../../src/DatasetManagerTool.cpp) (`main()` of the `dataset_manager` binary)
- **Built into:** `dataset_manager` (with `DatasetManagerArgs.cpp`, `DataFetcher`, `DatasetRegistry`, `adai_core`; `ModelNameClient` when `BUILD_MNS_SERVER`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.4.0`, `@adai-reviewed: 2026-10-19`
- **Tests:** none for `main()` itself; its parsers are tested by `datasetManagerArgsTests` ([DatasetManagerArgs.md](DatasetManagerArgs.md)) and `DatasetRegistry` by its own suites
- **Operator guide:** [dataset-manager-guide.md](../../../operations/guides/dataset-manager-guide.md) (covers only 8 of the 15 commands, §6)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real
> `dataset_manager` binary against scratch local pools and a local `registry_server`
> (`--port 18082`, scratch data dir).

---

## 1. What this file is

The **operator CLI for the dataset pool**: queue files and downloads, inspect the queue,
assign or unassign data to models, delete entries, migrate legacy entries into per-kind sub-pools
(TD-202), and split files into row-range segments (TD-205). It's one `main()` with an `if/else`
chain over commands. Parsing lives in `DatasetManagerArgs`, storage in `DatasetRegistry` (local
flat files, or HTTP to `registry_server`), and downloads in `DataFetcher`.

### Why it matters

It and the Android ops dashboard are the two ways operators change **which data trains which
model**. Several commands rewrite the shared pending pool, so its success messages and exit codes
are what operators trust (§4).

---

## 2. Startup

1. **Global flags** are scanned across the whole argv: `--config <path>` and `--kind <kind>`. Each
   flag and its value are removed wherever they appear (even after the command), and everything
   else goes to `args`. A trailing flag with no value stays in `args` as an ordinary argument.
   There's no `--help`: `dataset_manager --help` prints "Unknown command".
2. `--kind` is validated with `is_valid_dataset_kind()`.
3. **Config:** `discover_config_path(config_path, "config.registry.conf")`, then
   `ConfigLoader::load()`. `--kind` overrides `svc_config.dataset_kind` (TD-202).
4. **No command** → usage text (stdout, exit 0).
5. Each command builds its own `DatasetRegistry(DatasetRegistry::make_config(svc_config))`:
   `REGISTRY_SERVER_URL` set means `RemoteTransport`, otherwise `LocalTransport` under
   `SESSION_DIR`. `RUN_GROUP`, `MODEL_NAME` and `DATASET_KIND` come from the config file or
   environment.

---

## 3. Commands

| Command | Local mode (`LocalTransport`) | Remote mode (`registry_server`) | Exit 1 when |
|---|---|---|---|
| `add <file>` | `add_file()`: existence check, stores the path **as given** | `remote_upload()`: uploads the bytes into the server's `uploads/` | add/upload fails |
| `gutenberg <id> [n] [model]` | `DataFetcher::fetch_gutenberg()` then `add_file()` | `remote_fetch_gutenberg()` (server-side, per-model rotating slice) | fetch fails |
| `gutenberg-batch <ids> [n] [model]` | `fetch_gutenberg_batch()` | One `remote_fetch_gutenberg()` per ID | none added |
| `huggingface <id> [n] [split] [in] [out] [model]` | `fetch_huggingface()` then `add_file()` | `remote_fetch_huggingface()` | fetch fails |
| `status` | Kind, pending/trained counts, total samples, `print_registry()`, pending list with segments, pair counts and models | same | — |
| `list-pending` / `list-trained` | Tab-separated listing (`path`, `segment[a-b]` or `N pairs`, model) | same | — |
| `remove <file>` | Erases the first entry with that path, then `save_pending_list()` | **Pretends to succeed** (§4) | not found locally |
| `clear-pending` | Clears and saves | Fails: `save_pending` is a no-op on `RemoteTransport` | save fails |
| `assign <model> [files] [--count N]` | Optionally verifies the model in MNS, then `assign_model()` (the group comes from startup, TD-329) | same | nothing assigned |
| `unassign <model> [files] [--force]` | `unassign_model()` | same | nothing unassigned **and** nothing skipped |
| `delete <files> [--force] [--delete-files]` | `delete_entries()`, prints per-path status | same | nothing deleted |
| `migrate <kind> [files] [--count N]` | Legacy pool → `<kind>` pool, entry by entry, each segment exactly: add (with its range), restore the model, delete the source, checking each step and undoing the copy on failure (TD-326). A file target moves every unit of that file | same | nothing migrated |
| `segment <path> --count N` / `--ranges …` | `count_pairs()` on the **local** path, then one `add_pending_path_unchecked(path, start, count)` per range. TD-542: the segments replace an unclaimed whole-file entry of the path and take its model (a claimed entry or trained records refuse first; a failed replacement removes the segments again) | TD-543: a server-queued path is split by its `num_entries`; a local file is uploaded first and the uploaded path segmented; anything else refused | no segment created, or the replacement refused or failed |
| `requeue (--all \| <files>) [--model M]` | TD-540: `requeue_trained()`, server-side against `registry_server` | same | nothing re-queued, or the request failed |
| `models` | `ModelNameClient::list_models()` (defaults to `http://localhost:8083` when `NAME_SERVICE_URL` is unset) | — | query fails / not compiled |

Success messages echo the **requested** pair count ("Book added (500 pairs)"), not what was
actually written.

---

## 4. Behaviour problems

> **Fixed (TD-326, October 19, 2026): `migrate` duplicated segmented data.** It copied each
> legacy entry without its segment range and ignored the results of its delete and assign. On a
> 30-pair file split into 3 segments:
> - the `chatbot` pool got the whole file;
> - the other two segments failed;
> - all 3 segments stayed in the legacy pool;
> - it printed "Migrated 1/3".
>
> Training both pools trained every pair twice. Now each entry moves exactly and every step is
> checked (§3; addendum). `--count` still takes the first N entries regardless of segments.

> **`remove` lies in remote mode (verified).** `DatasetRegistry::remove_pending()` erases the
> entry from its in-memory list and calls `save_pending_list()`, but ignores the result.
> `RemoteTransport::save_pending()` is a no-op that returns `false`. Against a live
> `registry_server`, `remove <path>` printed "✅ Removed from pending queue… Pending files: 0" and
> **exited 0**, while `list-pending` still showed the file. The usage text says `remove` is
> "local mode only", but nothing enforces it. (`clear-pending` does report failure in remote mode,
> though with the misleading "Failed to save pending list".)

> **Local mode stores paths as given (verified).** `add data.jsonl` stored `data.jsonl` verbatim.
> `DataFetcher`'s local output dirs are relative too (`gutenberg_data/`, `huggingface_data/`). A
> trainer whose working directory differs (e.g. a systemd unit) can't find relative entries.

By inspection:

- **`run_group` isn't MNS-authoritative except in `assign`.** Only `assign` resolves the model in
  MNS and adopts its `run_group`. `status`, `list-*`, `unassign`, `delete`, `migrate`, `segment`
  and the fetch commands use the config's `RUN_GROUP`. When MNS assigns a model a different group
  (the [MNS run_group](../../../../CLAUDE.md) standard), `assign` writes into one group while
  every other command looks at another. (**Fixed in code November 7, 2026**, TD-329: see the addendum.)
- **Exit codes and messages.** `unassign` exits 0 with "Unassigned 0" when only skips occurred
  (exit 1 only if both counts are 0). Success lines report requested, not actual, pair counts.
  `models` silently falls back to `localhost:8083`, while `assign` skips verification when MNS is
  unset.
- **Parsing.** `--config`/`--kind` are stripped from anywhere in argv, so they can't be used as
  file names, and a value-less trailing one becomes an argument. There's no `--help`. Parser-level
  issues are TD-321–325.
- **`segment` ranges** aren't checked against the counted pairs (TD-324; **fixed November 7, 2026**: a range past the last pair is refused).

---

## 5. Tests

`main()` has no tests. Each command is a block of `DatasetRegistry` calls, testable only by running
the binary. The parsers ([DatasetManagerArgs.md](DatasetManagerArgs.md)) and `DatasetRegistry`
(`datasetRegistryTests`, `datasetRegistryLiveTests`, `registryTransportTests`) are tested
separately, but command composition isn't. That's exactly where the `migrate` and `remove` bugs
live. A small "CLI against a scratch `SESSION_DIR`" test (like the reproduction above) would cover
it.

---

## 6. Operator guide gaps

[dataset-manager-guide.md](../../../operations/guides/dataset-manager-guide.md) documents `add`,
`gutenberg`, `gutenberg-batch`, `huggingface`, `status`, `list-pending`/`list-trained` and
`clear-pending`. It doesn't cover `remove`, `assign`, `unassign`, `delete`, `migrate`, `segment`,
`models` or the `--kind` flag. Those are the commands that change assignments and pools. The
usage text in `main()` is currently the only complete description.

---

## 7. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `migrate` drops segment ranges and ignores failed source deletes and assigns (§4) | Data landed in both pools, so it was trained twice; misleading "Migrated" counts | **Fixed October 19, 2026** (was verified) | [TD-326](../../guides/TECHNICAL_DEBT.md#td-326-dataset_manager-migrate-duplicates-segmented-data-across-pools) |
| `remove` reports success (exit 0) against `registry_server` while nothing changes (§4) | Operators believe data was removed when it will still train (**fixed in code November 7, 2026**) | **Verified** | [TD-327](../../guides/TECHNICAL_DEBT.md#td-327-dataset_manager-remove-reports-success-in-remote-mode-without-removing-anything) |
| Local `add`/fetches store relative paths (§4) | Trainer with a different working directory can't find queued files (**fixed in code November 7, 2026**) | **Verified** | [TD-328](../../guides/TECHNICAL_DEBT.md#td-328-dataset_manager-stores-relative-paths-in-the-local-pool) |
| Only `assign` adopts the MNS `run_group`; other commands use the config's (§4) | Commands operate on different groups for the same model (**fixed in code November 7, 2026**) | By inspection | [TD-329](../../guides/TECHNICAL_DEBT.md#td-329-dataset_manager-uses-the-mns-run_group-only-in-assign) |
| `unassign` exit status, requested-not-actual pair counts, `models` localhost fallback, no `--help`, global flags stripped anywhere (§2, §4) | Misleading output and exit codes (**fixed in code November 7, 2026**, except `unassign`'s exit status) | By inspection | [TD-330](../../guides/TECHNICAL_DEBT.md#td-330-dataset_manager-output-exit-code-and-flag-hygiene) |
| No tests for command composition (§5) | Bugs above went unnoticed | By inspection | [TD-331](../../guides/TECHNICAL_DEBT.md#td-331-dataset_manager-has-no-tests-for-its-commands) |
| Operator guide covers 8 of 15 commands (§6) | Pool-changing commands undocumented for operators | By inspection | [TD-332](../../guides/TECHNICAL_DEBT.md#td-332-operator-guide-covers-only-8-of-15-dataset_manager-commands) |
| `segment` kept a queued whole-file entry beside its segments (§3) | Every pair trained twice | **Fixed October 19, 2026** (owner decision: replace) | [TD-542](../../guides/TECHNICAL_DEBT.md#td-542-dataset_manager-segment-keeps-a-queued-whole-file-entry-so-its-pairs-train-twice) |
| `segment` against a remote registry queues the client's local path, not the uploaded one (§3) | Segments made off the registry host can't be read by the server (**fixed in code November 7, 2026**) | **Verified** | [TD-543](../../guides/TECHNICAL_DEBT.md#td-543-dataset_manager-segment-queues-a-client-local-path-on-a-remote-registry) |
| Parser issues; unchecked `segment` ranges | Already tracked | — | TD-321–325 |

## Addendum (October 18, 2026): `requeue`

`dataset_manager requeue (--all | <file> ...) [--model <name>]` calls `DatasetRegistry::requeue_trained()`. With `REGISTRY_SERVER_URL` set, that goes to `registry_server`. The command prints a status per file and exits 1 if nothing was re-queued or the request failed. Its arguments are parsed by `parse_requeue_args()`, which rejects unknown flags. The operator guide documents it.

## Addendum (October 19, 2026): `migrate` moves segments exactly (TD-326)

Each legacy entry is moved with its segment range: add, then restore the model by segment, then delete the source by segment. Each step's result is checked, and the destination copy is rolled back on failure. Checked with the real binary: a whole-file entry plus 3 segments moved 4/4 and left the legacy pool empty. That run also found TD-542, where `segment` keeps an existing whole-file entry.

## Addendum (October 19, 2026): `segment` replaces an unclaimed whole-file entry (TD-542)

As the owner decided, the new segments replace a whole-file pending entry of the same path, so the file isn't queued (and trained) both ways. `segment` refuses before queuing anything when that entry is claimed by a run, or when the path has trained records, which the whole-path delete would erase. Otherwise it adds the segments, deletes the whole entry with `force=false`, and assigns the segments to the entry's model. If the delete doesn't remove exactly one entry, it deletes the new segments again and exits 1. The Android `RegistryRepository.createSegments()` follows the same steps.

Checked with the real binaries. Locally, and against a `registry_server` using the server's own path, a whole entry assigned to `m1` became segments assigned to `m1`, with no whole entry left. A whole entry claimed by `run-01` was refused, and nothing was queued. The same run found TD-543: against a remote registry, `segment` queues the client's local path. `add` uploads the file to `<data_dir>/<group>/uploads/` instead, so segments made away from the registry host point at a file the server can't read.

## Addendum (November 7, 2026): `remove`/`clear-pending` refuse in remote mode (TD-327)

Both commands edit the in-memory pending list and save it, which only a local pool can do (`RemoteTransport::save_pending()` is a
no-op). With `REGISTRY_SERVER_URL` set they now print, through `refuse_local_only_command()`, that nothing was changed and that
`dataset_manager delete <file>` is the remote equivalent (it also removes trained records; `--force` for a claimed file), and exit 1.
`remove` used to print "Removed" and exit 0 there while the file stayed queued. In local mode `remove` checks the file is queued, then
reports a failed save separately ("Failed to save pending list; nothing was removed"). `RegistryFtpConfinementTest`
`DatasetManagerRemoveRefusesInRemoteMode` runs the real binary against a spawned `registry_server`.

## Addendum (November 7, 2026): one `run_group` for every command (TD-329)

Before running any command but `models`, `main()` resolves the group once (`resolve_run_group()`). The model is the one the command
names (`assign`, `unassign`, `requeue --model`, via `run_group_model()`), otherwise `MODEL_NAME`; a fetch command's `[model]` is only
the registry's slice-cursor key. `ModelNameClient::lookup_model()` reads the model's full record in any state; `assign` used
`resolve_model()`, whose `/resolve` MNS refuses with 409 while a model is `initializing`, so it failed for every model not yet trained.
The MNS `run_group` wins when the record has one; an unregistered model, or one with no group on record, keeps `RUN_GROUP`. An MNS that
doesn't answer stops the command (exit 1, nothing changed). `assign` still requires a registered model. Against a `registry_server` the
group and its source go to stderr, for example `run_group: 'chatbot-v1' (MNS record of model 'm1')`. Local pools have no group, so local
commands other than `assign` don't contact MNS. `RegistryFtpConfinementTest` `DatasetManagerUsesTheModelsMnsRunGroupForEveryCommand`
and `DatasetManagerRefusesWhenTheNameServiceDoesNotAnswer` run the real binary against a spawned `registry_server` and a stand-in MNS.

## Addendum (November 7, 2026): `segment` against a remote registry (TD-543)

With `REGISTRY_SERVER_URL` set, `segment <path>` splits a path queued on the server as a whole file by the server's pair count for it
(`num_entries`), so the client needn't read it. Otherwise a local file is counted, uploaded as `add` does, and the uploaded
`<data_dir>/<group>/uploads/` path segmented; its segments replace the uploaded whole entry (TD-542). Anything else exits 1 without
queuing. Both flows are the owner's choice. `registry_server` now records `num_entries` as `DatasetRegistry::count_pairs()` (it counted
non-blank lines, too many for a file with a non-pair line, so the last segments would load nothing), and `count_pairs()` no longer keeps the
pairs. Local mode is unchanged. `RegistryFtpConfinementTest` `DatasetManagerSegmentUploadsALocalFileFirst`,
`DatasetManagerSegmentsAServerPathByItsServerPairCount` and `DatasetManagerSegmentRefusesAnUnknownRemotePath` run the real binary.

## Addendum (November 7, 2026): local paths are absolute (TD-328)

In local mode every queued path is absolute and normalized (`DatasetRegistry::add_file()` stores `absolute_local_path()`; `segment`
normalizes its path), so a trainer with another working directory finds the files. A relative file target of `remove`, `assign`,
`unassign`, `delete`, `requeue`, `migrate` or `segment` is made absolute the same way (`local_target()`), unless that exact string is
already queued or trained, which keeps entries queued relative before this fix reachable. Remote paths are untouched.

## Addendum (November 7, 2026): output and global options (TD-330)

Fetch commands report the pairs actually queued (`queued_pairs()`: the queue entry's `num_entries`, which `registry_server` sets from a
fetch's `pairs_written`, or a local count), naming the requested number only when it wasn't reached. `models` needs `NAME_SERVICE_URL`
(no `localhost:8083` fallback). `--help`, `-h` and `help` print the usage. `--config` and `--kind` are read only before the command and
need values; an unknown option there is an error, and `status`, `list-pending`, `list-trained`, `clear-pending` and `models` refuse
arguments. `unassign`'s exit status (0 when every match was skipped as claimed) is unchanged; TD-330's action items didn't cover it.
