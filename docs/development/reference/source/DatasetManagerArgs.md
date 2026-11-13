# `DatasetManagerArgs` — Source File Reference

- **Files:** [`src/DatasetManagerArgs.hpp`](../../../../src/DatasetManagerArgs.hpp) (134 lines), [`src/DatasetManagerArgs.cpp`](../../../../src/DatasetManagerArgs.cpp) (440)
- **Namespace:** `adai`
- **Built into:** `dataset_manager` (with `DatasetManagerTool.cpp` and `Config.cpp`) and `datasetManagerArgsTests`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0` (hpp and cpp), `@adai-reviewed: 2026-11-12` (promoted from `beta` that day)
- **Tests:** [`tests/dataset_manager_args_test.cpp`](../../../../tests/dataset_manager_args_test.cpp) → `datasetManagerArgsTests` (60 tests)
- **Origin:** TD-035 (parsing pulled out of `DatasetManagerTool.cpp` to make it testable); TD-202 (`migrate`, kinds); TD-205 (`segment`)
- **Last traced against the code:** 2026-11-12

> Code-traced reference. Behaviour marked **verified** was confirmed by compiling the parsers
> into a scratch program (with UBSan) and running them.

---

## 1. What this file is

The **per-command argument parsers** for the `dataset_manager` CLI. `main()` in
`DatasetManagerTool.cpp` reads the global options (`--config`, `--kind`, `--help`) and strips the
command name. It then hands the command's own argument slice to the matching `parse_*_args()`, which
returns a small struct with either the parsed fields or `error = true` plus a message.

The registry and fetcher calls stay in `main()`, so these parsers are pure functions over
`std::vector<std::string>`, testable with no file or network I/O.

### Why it matters

These parsers decide **what `dataset_manager` does to the shared training pool**. `assign`,
`unassign`, `delete`, `migrate` and `segment` change which data gets trained, for which model, so a
mis-parsed argument must fail rather than quietly change the operation. Until November 2026 several
did change it (§4).

---

## 2. Commands and parsers

All parsers take the command's own arguments, **without** the command name, e.g.
`dataset_manager assign my-model f1 --count 3` → `parse_assign_args({"my-model","f1","--count","3"})`.

| Command | Parser | Syntax | Notes |
|---|---|---|---|
| `add` | `parse_add_args` | `[--overwrite] <data_file>` | Exactly one file. `--overwrite` (TD-552) anywhere before `--` |
| `gutenberg` | `parse_gutenberg_args` | `<book_id> [num_pairs=500] [model]` | At most 3 arguments. `model` matters only with `REGISTRY_SERVER_URL` (per-model rotating slices) |
| `gutenberg-batch` | `parse_gutenberg_batch_args` | `<id1,id2,…> [num_pairs_each=500] [model]` | Comma-separated ids, each checked; at most 3 arguments |
| `huggingface` | `parse_huggingface_args` | `<dataset_id> [num_pairs=500] [split=train] [input_field] [output_field] [model]`, or the four optional fields named: `--split`, `--input-field`, `--output-field`, `--model` | TD-325: named fields in any order, alongside or instead of the positional form. A field given both ways, given twice, or a flag with no value is an error; so is a seventh positional |
| `remove` | `parse_remove_args` | `<data_file>` | Exactly one |
| `assign` | `parse_assign_args` | `<model> [files…] [--count N]` | No files and no `--count` → every pending file. `main()` ignores `--count` when files are given. The model can't look like an option |
| `unassign` | `parse_unassign_args` | `<model> [files…] [--force]` | No files → every file assigned to the model |
| `delete` | `parse_delete_args` | `<files…> [--force] [--delete-files]` | At least one target |
| `requeue` | `parse_requeue_args` | `--all` **or** `<files…>`, `[--model <name>]` | TD-540: exactly one of `--all` or files; `--model` needs a value that doesn't look like an option |
| `migrate` | `parse_migrate_args` | `<kind> [files…] [--count N]` | `kind` checked by `is_valid_dataset_kind()`; no files and no `--count` → every legacy entry |
| `segment` | `parse_segment_args` | `<path> --count N` **or** `--ranges A-B,C-D,…` | Ranges are 0-based inclusive, stored as `{start, B−A+1}`; exactly one mode, each at most once; any other argument is an error (§3) |

Every parser that takes files (`add`, `assign`, `unassign`, `delete`, `requeue`, `migrate`, and `huggingface`'s
positionals) refuses an argument that starts with `-` and isn't one of its own flags
(`Unknown option …`, TD-322), and treats `--` as the end of the options, so a file name may start with
`-` after it.

**Numbers** (TD-321, TD-323) go through `parse_strict_int()` (`Config.hpp`), which takes only a
whole number (`3x`, `1342abc` and an empty value are errors). Book ids, pair counts and
`segment --count` must be at least 1 (`parse_positive()`). `assign`'s and `migrate`'s `--count` must
be at least 1 too (`parse_count()`); omitting it is the only way to mean "all", and the error says so.

`is_valid_dataset_kind(k)` accepts exactly `encoder`, `decoder`, `world_model` and `chatbot`, through
`adai::is_dataset_kind()`, the list the trainer uses too (TD-402). `main()` also checks the global
`--kind` with it.

`make_error<T>(msg)` builds a `T` with `error = true`; `unknown_option<T>()` and `missing_value<T>()`
word the two common errors with the command's usage.

---

## 3. `segment`'s ranges

`parse_segment_args()` takes the path first and then:
- `--count N` (at least 1) or `--ranges A-B,…`, each at most once (TD-324: a second `--ranges` used to
  add to the first), and not both;
- each range token needs `0 ≤ A ≤ B`, both whole numbers; its size `B − A + 1` is computed in 64 bits
  and refused above `INT_MAX` (TD-324: `0-2147483647` overflowed to a negative-count segment);
- after sorting by start, overlapping or repeated ranges are refused (they would queue, and train,
  the same pairs twice).

Whether the ranges lie inside the file is checked by `main()` once it knows the pair count
([DatasetManagerTool.md](DatasetManagerTool.md) §3).

---

## 4. History: what used to be mis-parsed (verified)

Before November 2026 every number went through bare `std::stoi` and every list command took any
argument it didn't recognize as a file target. These were confirmed with UBSan, and each is now an
error:

| Input | Was parsed as | Effect | Fixed |
|---|---|---|---|
| `assign m --count -2` (or `0`), `migrate chatbot --count 0` | "all" | Assigned or migrated the **whole pending pool** | TD-321, November 5, 2026 |
| `assign m --count 3abc`, `gutenberg 1342abc 500x` | `3`; `1342`, `500` | Trailing garbage dropped | TD-323, November 7 |
| `gutenberg -5 -10`, `gutenberg-batch 1342,0` | Accepted | Failed later, or wrote zero pairs locally | TD-323 |
| `assign m --count` (no value) | File target `--count` | "No matching pending files" instead of a usage error | TD-322, November 7 |
| `delete f.jsonl --delete-file` (typo) | Extra target | Deleted the entry, **kept the file**, no warning | TD-322 |
| `add --foo` | File name `--foo` | "Data file not found: --foo" | TD-552, November 12 |
| `requeue -a` | File name `-a` | "not found" instead of an unknown-option error | November 12 |
| `segment p --ranges 0-2147483647` | Signed overflow (UB) | A negative-count segment | TD-324, November 7 |
| `segment p --ranges 0-9,5-14`, or `--ranges` twice | Accepted | The same pairs trained twice | TD-324 |
| `huggingface ds 500 my-model` | `split = "my-model"` | A request for a split named after the model | TD-325, November 7 (named fields) |
| `add`/`remove`/`gutenberg`/`gutenberg-batch` with extra arguments | Ignored | Silently did less than asked | TD-325 |

Below the CLI, `LocalTransport::assign()` and `registry_server`'s `assign` refuse a negative count
(0 still means all there, as the Android dashboard sends it); see
[RegistryTransport.md](RegistryTransport.md) and [RegistryServer.md](RegistryServer.md).

---

## 5. Tests

`dataset_manager_args_test.cpp` has 60 tests:

| Group | Tests | Covers |
|---|---|---|
| `ParseSegmentArgs` | 12 | Both modes, mutual exclusion, malformed, backward and negative ranges, unknown arguments; overflow, overlap in any order, repeated flags (`RangeValidationGaps`) |
| `ParseMigrateArgs` | 7 | Kinds, targets, `--count` values |
| `ParseAssignArgs` | 6 | Targets, `--count` 0/negative/partial |
| `ParseHuggingfaceArgs` | 5 | Positional and named fields, conflicts, unknown options |
| `ParseAddArgs` | 4 | One file, `--overwrite` before or after it, an unknown option, `--` |
| `ParseGutenbergArgs`, `ParseGutenbergBatchArgs` | 4 each | Ids and counts |
| `ParseDeleteArgs`, `ParseUnassignArgs` | 3 each | Flags and targets |
| `DatasetManagerArgsOptions` | 3 | TD-322 across commands: unknown options, value-less flags, `--` |
| `ParseRemoveArgs`, `IsValidDatasetKind`, `DatasetManagerArgsNumbers` | 2 each | |
| `DatasetManagerArgsTest` | 2 | `requeue`, and its option rules |
| `DatasetManagerArgsExtras` | 1 | Extra arguments refused |

The test target compiles `Config.cpp` (with `GenerationSettings.cpp` and `Logger.cpp`) for
`parse_strict_int()` and `is_dataset_kind()`.

---

## 6. Known gaps

None of its own. Until November 12, 2026 `requeue` treated only `--…` as an option, so `requeue -a`
took `-a` for a file name, and it had no `--`; it now follows the same rules as every other command
(`RequeueFollowsTheSharedOptionRules`). That was the last inconsistency, and the file moved to `stable`
1.0.0 the same day.

Commands with no tests of their own are tracked in TD-331 ([DatasetManagerTool.md](DatasetManagerTool.md)).
