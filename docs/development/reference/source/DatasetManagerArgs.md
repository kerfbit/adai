# `DatasetManagerArgs` — Source File Reference

- **Files:** [`src/DatasetManagerArgs.hpp`](../../../../src/DatasetManagerArgs.hpp), [`src/DatasetManagerArgs.cpp`](../../../../src/DatasetManagerArgs.cpp)
- **Namespace:** `adai`
- **Built into:** `dataset_manager` (with `DatasetManagerTool.cpp`) and its test binary
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.3.0`, `@adai-reviewed: 2026-09-20`
- **Tests:** [`tests/dataset_manager_args_test.cpp`](../../../../tests/dataset_manager_args_test.cpp) → `datasetManagerArgsTests` (45 tests)
- **Origin:** TD-035 (parsing pulled out of `DatasetManagerTool.cpp` to make it testable); TD-202 (`migrate`, kinds); TD-205 (`segment`)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by compiling the parsers
> into a scratch program (with UBSan) and running them.

---

## 1. What this file is

The **per-command argument parsers** for the `dataset_manager` CLI. `main()` in
`DatasetManagerTool.cpp` handles the global flags (`--config`, `--kind`, …) and strips the command
name. It then hands the command's own argument slice to the matching `parse_*_args()`, which
returns a small struct with either the parsed fields or `error = true` plus a usage message.

The registry and fetcher calls stay in `main()`, so these parsers are pure functions over
`std::vector<std::string>`, testable with no file or network I/O. `DatasetRegistry`'s behaviour has
its own tests.

### Why it matters

These parsers decide **what `dataset_manager` does to the shared training pool**. `assign`,
`unassign`, `delete`, `migrate` and `segment` change which data gets trained, for which model.
A quietly mis-parsed argument changes the operation rather than failing (§4).

---

## 2. Commands and parsers

All parsers take the command's own arguments, **without** the command name, e.g.
`dataset_manager assign my-model f1 --count 3` → `parse_assign_args({"my-model","f1","--count","3"})`.

| Command | Parser | Syntax | Notes |
|---|---|---|---|
| `add` | `parse_add_args` | `<data_file>` | Extra arguments ignored |
| `gutenberg` | `parse_gutenberg_args` | `<book_id> [num_pairs=500] [model]` | `model` only matters with `REGISTRY_SERVER_URL` (per-model rotating slices) |
| `gutenberg-batch` | `parse_gutenberg_batch_args` | `<id1,id2,…> [num_pairs_each=500] [model]` | Comma-separated IDs; an empty token (`1,,2`) is an error |
| `huggingface` | `parse_huggingface_args` | `<dataset_id> [num_pairs=500] [split=train] [input_field] [output_field] [model]` | All positional; to give `model` you must give every argument before it |
| `remove` | `parse_remove_args` | `<data_file>` | |
| `assign` | `parse_assign_args` | `<model> [files…] [--count N]` | No files and no `--count` → assign **all** pending; `--count` is ignored when files are given (`main()`'s help) |
| `unassign` | `parse_unassign_args` | `<model> [files…] [--force]` | |
| `delete` | `parse_delete_args` | `<files…> [--force] [--delete-files]` | At least one target required |
| `migrate` | `parse_migrate_args` | `<kind> [files…] [--count N]` | `kind` validated by `is_valid_dataset_kind()`; no files and no count → migrate **all** legacy entries |
| `segment` | `parse_segment_args` | `<path> --count N` **or** `--ranges A-B,C-D,…` | Ranges are 0-based inclusive, stored as `{start, B−A+1}`; exactly one mode required; unknown arguments rejected |

`is_valid_dataset_kind(k)` accepts exactly `encoder`, `decoder`, `world_model` and `chatbot`, MNS's
`ModelRecord::kind` vocabulary. `main()` also uses it for the global `--kind` flag.

`make_error<T>(msg)` is the shared helper that builds a `T` with `error = true`.

---

## 3. How `main()` uses the results

Traced in `DatasetManagerTool.cpp`:

- On `error`, `main()` prints `error_message` plus command-specific help and exits 1.
- `assign`/`unassign`/`delete` call `DatasetRegistry::assign_model`/`unassign_model`/`delete_entries`
  with the parsed targets. Both `LocalTransport::assign()` and `registry_server`'s handler treat **no
  targets** with `count <= 0` as `assign_all`.
- `migrate` with no targets copies `count` entries if `count > 0`, **otherwise all** of them.
- `segment` checks the path's pair count locally (`DatasetRegistry::count_pairs()`), then queues one
  pending entry per range via `add_pending_path_unchecked()`. Explicit `--ranges` are **not**
  checked against that count (§4).
- `gutenberg`/`huggingface` pass `book_id`, `dataset_id`, `num_pairs` and the rest straight to the
  registry (which validates `book_id > 0`, the `dataset_id`/`split` allow-list and `model_name`) or,
  in local mode, to `DataFetcher` (which validates nothing; see TD-310, TD-312).

---

## 4. Parsing pitfalls (verified)

Every numeric field uses bare `std::stoi`, and every list command treats any argument it doesn't
recognize as a file target:

| Input | Parsed as | Effect |
|---|---|---|
| `assign m --count -2` (or `0`) | `count = -2`, no targets | **Assigns every pending file to `m`**: `count <= 0` with no targets means "all" |
| `migrate chatbot --count 0` | `count = 0` | **Migrates every legacy pending file** |
| `assign m --count 3abc`, `gutenberg 1342abc 500x` | `3`; `1342`, `500` | Trailing garbage silently dropped |
| `gutenberg -5 -10`, `huggingface x -1`, `gutenberg-batch 1342,0` | Negative/zero IDs and pair counts accepted | Fails later (registry 400, or zero pairs written locally) |
| `assign m --count` (no value), `assign m --count=3` | Target files `--count` / `--count=3` | "No matching pending files" instead of a usage error |
| `delete f.jsonl --delete-file` (typo) | `delete_files = false`, targets `f.jsonl`, `--delete-file` | Deletes the entry but **keeps the file**, with no warning about the unknown flag |
| `unassign m --froce` (typo) | `force = false`, target `--froce` | Silently does nothing useful |
| `segment p --ranges 0-2147483647` | `count = b − a + 1` **overflows** (UBSan: signed integer overflow) → `-2147483648` | Undefined behaviour; a negative-count segment |
| `segment p --ranges 0-9,5-14`; `--ranges` given twice | Overlapping/duplicate ranges accepted | The same pairs train twice |
| `segment p --count -5 --ranges 0-1` | Accepted (the exclusivity check uses `split_count > 0`) | `--count` silently ignored |

The "≤ 0 means all" behaviour is the dangerous one. It's documented only as "omit `--count`", so a
mistyped or negative value quietly turns a "first N" request into "everything".

Other notes:
- `huggingface`'s purely positional optional arguments make `model` awkward to give and easy to
  shift: `huggingface ds 500 my-model` sets `split = "my-model"`.
- `add`/`remove`/`gutenberg` ignore extra arguments rather than rejecting them.
- `dataset_id` isn't validated here. In local mode it reaches `DataFetcher`'s shell `curl` commands
  (TD-312); the registry path validates it server-side.

---

## 5. Tests

`dataset_manager_args_test.cpp` has 45 tests, covering each parser's happy path, missing-argument
errors and non-numeric values:

| Group | Tests |
|---|---|
| `ParseSegmentArgs` | 11 |
| `ParseMigrateArgs` | 6 |
| `ParseAssignArgs` | 5 |
| `ParseGutenbergArgs`, `ParseGutenbergBatchArgs` | 4 each |
| `ParseDeleteArgs`, `ParseUnassignArgs`, `ParseHuggingfaceArgs` | 3 each |
| `ParseAddArgs`, `ParseRemoveArgs`, `IsValidDatasetKind` | 2 each |

**Not covered:**
- negative or zero counts and IDs;
- partial numbers;
- unknown or typo'd flags;
- a flag with no value;
- range overflow and overlap.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `--count` ≤ 0 (including typos like `-2`) means "all" for `assign` and `migrate` (§4) | One mistyped value assigns or migrates the entire pending pool | **Verified** | [TD-321](../../guides/TECHNICAL_DEBT.md#td-321-dataset_manager-treats---count--0-as-all) |
| Unknown flags, typo'd flags and value-less `--count` become file targets (§4) | Operations silently differ from intent (e.g. `--delete-file` keeps files) | **Verified** | [TD-322](../../guides/TECHNICAL_DEBT.md#td-322-dataset_manager-treats-unknown-flags-as-file-targets) |
| `std::stoi` accepts partial numbers; negative/zero IDs and pair counts accepted (§4) | Wrong values pass validation; errors surface late or not at all | **Verified** | [TD-323](../../guides/TECHNICAL_DEBT.md#td-323-dataset_manager-accepts-partial-and-out-of-range-numbers) |
| `--ranges` overflow (UB), overlaps and duplicates accepted, not checked against the file's pair count; `--count` ignored alongside `--ranges` when ≤ 0 (§3–4) | Negative-count segments; pairs queued twice or beyond EOF | **Verified** (overflow, overlap) / by inspection (EOF) | [TD-324](../../guides/TECHNICAL_DEBT.md#td-324-dataset_manager-segment-range-validation-gaps) |
| `huggingface` optional arguments purely positional; extra arguments ignored by several commands (§4) | Easy to shift arguments into the wrong field | By inspection | [TD-325](../../guides/TECHNICAL_DEBT.md#td-325-dataset_manager-huggingface-positional-arguments-shift-silently) |
| `dataset_id` unvalidated before local `DataFetcher` shell commands | Already tracked | — | TD-312 |
