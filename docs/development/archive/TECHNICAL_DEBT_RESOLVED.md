# Technical Debt - Resolved Items

Resolved items extracted from [TECHNICAL_DEBT.md](../guides/TECHNICAL_DEBT.md).

## Resolved Items

### TD-086: Dataset::get_batch_statistics() Tokenized Raw Dataset Indices Instead of the Requested Split's Samples

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | Dataset (`get_batch_statistics()`) | Use the already-computed split-mapped `data_idx`, not the raw loop counter `i` |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/Dataset.hpp` end to
end (1578 lines). `get_batch_statistics(split_type, tokenizer_fn, batch_size)` samples up to
`batch_size` entries from the requested split's index list to estimate padding efficiency. Its loop
correctly resolved each split-relative position to a real dataset index —
`size_t data_idx = (*indices)[i];` — but then ignored it, tokenizing `data_[i].input` (the raw loop
counter) instead of `data_[data_idx].input`. Since `train_indices_`/`validation_indices_`/
`test_indices_` are shuffled subsets of `[0, data_.size())` (see `split()`), `i` and `data_idx` only
coincide by chance. In practice this meant `get_batch_statistics()` silently computed its stats from
whichever samples happen to occupy raw positions `[0, batch_size)` in the full, unsplit dataset —
almost never the split actually requested, and reading past the split's own sample count entirely once
`batch_size` exceeds it (bounded only by the full dataset's size, not the split's).

The method has no current caller in the codebase (confirmed via `grep -rn` across `src/`/`tests/`) and no
existing test, so this had zero production reachability today, but it's public API with a documented
usage example in its own doc comment, silently wrong for anyone who starts using it. `data_idx` being
computed and then never read is exactly the kind of copy-paste slip this audit has repeatedly found
elsewhere in the codebase (e.g. TD-083's self-comparison bug) — cheap and unambiguous to fix once seen.

Changes Made:
- `src/Dataset.hpp`: `get_batch_statistics()`'s sampling loop now tokenizes `data_[data_idx].input`
  instead of `data_[i].input`.

Verification:
- ✅ Added `BatchStatisticsUsesSplitIndicesNotRawIndices` to `tests/dataset_test.cpp`: builds a dataset
  where each sample's input length directly encodes its raw index (sample *i* is `i+1` characters), so a
  character-counting tokenizer's token count reveals which samples were actually used; computes the
  ground truth by tokenizing the split's real samples directly via `get_split()`, then compares against
  `get_batch_statistics()`'s result for the same split.
- ✅ Before/after regression: reverted only this one-line fix, rebuilt `datasetTests`, confirmed the new
  test fails with the exact predicted symptom (`actual_tokens` 55 — the sum for raw samples 0–9 — instead
  of the correct split-based 110); restored the fix, rebuilt, confirmed pass plus the full 43/43
  `DatasetTests` suite.

Files Changed:

- `src/Dataset.hpp`
- `tests/dataset_test.cpp`

---

### TD-085: RegistryServer's /trained Endpoint Wrote Duplicate Registry Entries for a Path Repeated Within One Request

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | registry_server (`POST /registry/<group>/trained`) | Update the in-loop `existing` dedup set as each new `DataVersion` is pushed, not just once before the loop |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/RegistryServer.cpp`
end to end (1679 lines). `handle_trained()` builds a `std::set<std::string> existing` from the
already-persisted registry once, before iterating the request's `files` array to decide which paths are
new: `if (!existing.count(files[i])) { ...; reg.push_back(std::move(dv)); ++trained; }`. `existing` is
never updated inside that loop, so if the *same request's* `files` array contains the same path twice,
both occurrences pass the `!existing.count()` check and each gets its own `DataVersion` pushed onto
`reg` — writing two entries for one file into the persisted registry (`TrainedDeduplicatesInRegistry`,
the existing coverage, only exercises the cross-call case: the same path committed via two separate
`/trained` calls, which correctly dedupes because the second call's `existing` set is rebuilt from the
by-then-updated registry).

The shipped `RemoteTransport::commit_trained()` client (`src/RegistryTransport.cpp`) doesn't currently
trigger this — it already dedupes when merging `new_entries` against `trained_paths` via a `files_set`
check — but nothing prevents `new_entries` itself, or a direct API caller (`curl`, `dataset_manager`, or
any future client), from listing the same path twice in a single call, and the endpoint's own
`TrainedMultipleFilesInOneCall` test explicitly documents multiple-files-per-call as supported, ordinary
usage. Fixed as a direct, low-risk defensive correction while already reading this exact function,
matching `add_pending_path_locked()`'s existing duplicate-prevention convention just above it in the
same file.

Changes Made:
- `src/RegistryServer.cpp`: `handle_trained()` now inserts `files[i]` into `existing` immediately after
  pushing its new `DataVersion`, so a later duplicate in the same `files` array is correctly rejected by
  the very check already guarding cross-call duplicates.

Verification:
- ✅ Added `TrainedDeduplicatesWithinSingleCall` to `tests/dataset_registry_live_test.cpp` (same live
  `registry_server`-backed harness as the existing `TrainedDeduplicatesInRegistry`), asserting a single
  `/trained` call listing one path twice commits it exactly once (`"trained":1`) and the registry ends up
  with exactly one entry for that path.
- ✅ Before/after regression against a real `registry_server` instance: reverted only the one-line fix,
  rebuilt, confirmed the new test fails with the exact predicted symptom (`"trained":2`, two identical
  `data_file` entries in the registry response); restored the fix, rebuilt, confirmed pass.
- ✅ Full `LiveRegistryTest`/`RemoteTransportTest` suite (82/82) passed against the real `registry_server`
  instance, plus the full 78/78 `ctest` run (which auto-skips this live suite without
  `REGISTRY_SERVER_HOST` set, confirmed unaffected).

Files Changed:

- `src/RegistryServer.cpp`
- `tests/dataset_registry_live_test.cpp`

---

### TD-084: MNS "candidate" State Transition Didn't Verify run_id Ownership, Letting a Superseded Trainer Clobber the Active Run

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | ModelNameService (`PUT /models/{name}/state`, "candidate" transition) | Reject a "candidate" transition from "training" whose `run_id` doesn't match the model's current active `run_id`, mirroring the check `/progress` already had |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/ModelNameService.cpp`
end to end (1697 lines). `handle_state_transition()`'s `"training"` branch is explicit that MNS —
not the client — is authoritative for `run_id` (see CLAUDE.md "MNS/registry-authoritative run and
session numbering"), and `handle_progress_update()` correctly enforces this: it 409s if the caller's
`run_id` doesn't match the model's current `r.run_id`, so a trainer whose run was superseded by a new
`set_training` call (e.g. it looked crashed/hung and a second trainer was started for the same model)
can't push stale progress into the active run's live snapshot. The `"candidate"` branch — reached at the
*end* of a run, exactly the point a long-lived trainer finally reports back — had no such check at all.
It read `run_id` from the request body and only ever used it as `h.run_id = run_id.empty() ? r.run_id :
run_id;` when building the `training_history` entry, never comparing it against `r.run_id`. Any client
supplying any `run_id` (or none) could drive the state to `"candidate"` as long as the model's *current*
state happened to be `"training"` — including a trainer whose run had already been superseded.

Concretely: Trainer A starts `run-01`, pushes progress. It appears to have crashed, so a new
`set_training(new_run=true)` call starts `run-02` for the same model (archiving A's last-known progress
into `training_history` as `incomplete=true`, per the existing crash-recovery path) — Trainer B is now
the active run. Trainer A was not actually dead; it eventually finishes and calls
`PUT /models/{name}/state {"state":"candidate","run_id":"run-01",...}`. Since the model's `cur` state is
still `"training"` (now under `run-02`), the transition was accepted: A's stale artifact and
`training_summary` silently became the record's new state, resetting the live progress snapshot and
clearing `run_id` — completely clobbering Trainer B's still-in-progress run, which then has no way to
`/progress`-push (its `run_id` no longer matches, having been cleared) or legitimately reach
`"candidate"` (the model has already left `"training"`).

Every real caller already supplies a `run_id` on the candidate call — `IncrementalTrainer::run()` always
calls `mns_client_->set_candidate(model_name, current_run_id_, ...)` with the run id captured at
`begin_run()` time (`src/IncrementalTrainer.cpp`), and `mns_cli set-candidate` takes it as a mandatory
positional argument (`src/MnsCliTool.cpp`) — so requiring a match imposes nothing on any legitimate
caller. Deliberately scoped the check to `cur == "training"` only: the `"candidate"` transition is also
valid from `"initializing"` (importing an already-trained model) and `"retired"` (reviving one), and in
both of those `r.run_id` is empty with no active run to protect — an ownership check there would be
meaningless and would break the legitimate import/revival flows, which supply no run in progress to
match against.

While fixing this, discovered eight existing `MNSLiveTest`/`MnsManagerGUILiveTest` tests were
inadvertently relying on the very absence of this check: each drove `"training"` with an arbitrary
client-supplied `run_id` (e.g. `"run-abc"`, `"r1"`) — itself a no-op, since MNS allocates the real
`run_id` server-side — and then reused that same made-up string on the following `"candidate"` call.
Before this fix, the missing ownership check let the mismatched `run_id` through anyway; with it in
place, those calls correctly 409, which caused all eight to fail. Two of them (`ExplicitRetire`,
`DeleteModel_ProductionReturns409`) never asserted on the intermediate candidate call's status and
happened to reach the same final assertion regardless, masking the same gap — left alone since they
aren't wrong once the real bug is fixed, just weaker than they could be.

Changes Made:
- `src/ModelNameService.cpp`: in `handle_state_transition()`'s `new_state == "candidate"` branch, reject
  with 409 (`"run_id does not match the active run; this trainer's run has been superseded"`) when
  `cur == "training"` and the request's `run_id` is empty or doesn't equal `r.run_id`.

Verification:
- ✅ Added `StateTransition_CandidateRejectsSupersededRunId` to
  `tests/model_name_service_live_test.cpp`, reproducing the exact Trainer-A/Trainer-B race: asserts the
  stale candidate call now 409s, the active run's state/`run_id`/progress are untouched, and the real
  active trainer can still legitimately reach `"candidate"` afterward.
- ✅ Before/after regression against a real `mns_server` instance (not just the fixture): reverted only
  the `ModelNameService.cpp` ownership-check hunk, rebuilt `mns_server`/`mnsLiveTests`, confirmed the new
  test fails with the exact predicted symptoms (stale candidate call returns 200, active run's state
  becomes `"candidate"`/`run_id` clears/progress resets to 0, the real trainer's later candidate call
  then 409s because the model already left `"training"`); restored the fix, rebuilt, confirmed pass.
- ✅ Fixed the eight pre-existing tests exposed by the new check (`StateTransition_
  TrainingToCandidate_AttachesArtifact`, `Promote_CandidateToProduction`,
  `Promote_AutoRetiresPreviousProduction`, `ResolveRole_ReturnsArtifact`,
  `ListRoles_ContainsPromotedRole`, `ResolveModel_CandidateReturnsArtifact`,
  `TrainingHistory_StoredAfterStateTransitions`, `TrainingHistory_PersistsAcrossGetModel` in
  `model_name_service_live_test.cpp`; `FullLifecycle_RegisterTrainPromote` and
  `ListRoles_ParsesWithJsonArrayObjects` in `mns_manager_gui_test.cpp`) to capture and use the real
  server-allocated `run_id` from the training response instead of an arbitrary hardcoded string;
  `ListRoles_ParsesWithJsonArrayObjects` was additionally missing the training→candidate→promote
  sequence it needed to populate `/roles` at all.
- ✅ Full `MNSLiveTests` (39/39) and `MnsManagerGuiTests` (43/43) suites pass against a real `mns_server`
  instance launched from a config-isolated directory (to avoid the repo's own `config.mns.conf`
  supplying a `REGISTRY_SERVER_URL` that would otherwise change `/datasets` endpoint behavior).

Files Changed:

- `src/ModelNameService.cpp`
- `tests/model_name_service_live_test.cpp`
- `tests/mns_manager_gui_test.cpp`

---

### TD-083: IncrementalTrainer's Best-Checkpoint Recovery Was Gated on an Unrelated Flag, and Its Symlink Refresh Was a Silent No-Op

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | IncrementalTrainer (checkpoint retention cleanup) | Un-gate the recovery bookkeeping; stop routing the symlink refresh through a self-comparing helper |

Summary:
Found in the same full second-pass re-read of `src/IncrementalTrainer.cpp` that produced TD-082 — two
related bugs in `cleanup_old_sessions()`'s "the checkpoint we just deleted was the recorded best, so
find the next-best from what remains" recovery block:

**(A)** The entire recovery block — resetting `best_validation_loss`/`best_checkpoint_path` and
searching `session_history` for the next-best candidate — was gated behind
`config.enable_checkpoint_symlinks`, a flag whose own name and doc comment ("Create latest/best
symlinks") describe it as controlling *filesystem symlink* bookkeeping only. With symlinks disabled,
the in-memory `best_checkpoint_path`/`best_validation_loss` were left dangling, still referencing the
checkpoint file `remove_model_files()` had just deleted moments earlier in the same function.

**(B)** Even with symlinks enabled (the shipped default — `enable_checkpoint_symlinks` is not exposed
via `ServiceConfig`/`config.trainer.conf` at all, so every real `incremental_trainer` deployment runs
with it `true`), the block refreshed the "best" symlink by calling
`update_best_checkpoint(best_validation_loss, best_checkpoint_path)` — but by that point the caller had
already assigned the same `best_validation_loss`/`best_checkpoint_path` **member variables** to those
exact values via the search loop immediately above. `update_best_checkpoint()`'s own "is this an
improvement" logic compares its `validation_loss` argument against the current
`best_validation_loss` member — which is now the same value — so the comparison is a self-comparison,
always false (barring the vacuous `session_history.size() <= 1` special case, which cannot occur here
since `cleanup_old_sessions()` only ever runs when history exceeds `max_sessions_to_keep`). The "best"
symlink was therefore silently never refreshed, left dangling at the just-deleted file — reachable under
default settings whenever the best-known checkpoint happens to be among the oldest sessions pruned by
retention cleanup (a realistic outcome any time training regresses after an early low-loss run).

Both reproduced directly with standalone programs mirroring the exact member-state and call pattern:
(A) confirmed the search loop and its results are skipped entirely when the flag is false; (B) confirmed
the in-memory `best_checkpoint_path` updates correctly (the caller sets it directly) while the
stand-in-for-the-symlink flag stayed false, i.e. the refresh never fires, when routed through the
self-comparing helper.

Changes Made:

- Moved the `best_validation_loss`/`best_checkpoint_path` reset-and-research block outside the
  `enable_checkpoint_symlinks` check so it always runs when `deleting_best` is true.
- Nested only the symlink-specific work inside `if (config.enable_checkpoint_symlinks)`, and replaced
  the `update_best_checkpoint(...)` call with a direct `create_or_update_symlink(best_checkpoint_path,
  config.best_symlink_name)` (matching what `update_best_checkpoint()` does internally when it does
  decide something is best) — this callsite already knows, unconditionally, that
  `best_checkpoint_path` is the correct new best, so it has no use for `update_best_checkpoint()`'s
  "is this better than the current record" gate in the first place.

Verification:
- ✅ Standalone reproductions confirmed both bugs independently before the fix.
- ✅ Added `CleanupOldSessionsRecoversBestTrackingAndSymlinkWhenBestCheckpointIsPruned` (symlinks
  enabled — covers both the in-memory recovery and the symlink refresh) and
  `CleanupOldSessionsRecoversBestTrackingWhenSymlinksDisabled` (isolates bug A) to
  `tests/incrementaltrainer_test.cpp`, deliberately constructing history where the *oldest* session is
  the *best* one so retention cleanup is guaranteed to delete it. Added a `touch_full_checkpoint()`
  fixture helper and a `friend class IncrementalTrainerTest;` declaration (the private
  `get_best_checkpoint_path()` has no public equivalent) to `IncrementalTrainer.hpp`, plus a
  `best_checkpoint_path_of()` fixture wrapper since friendship does not propagate to gtest's
  `TEST_F`-generated subclasses.
- ✅ Before/after regression: reverted `IncrementalTrainer.cpp` to its pre-fix `HEAD` (which already had
  TD-082's fix, isolating this diff), rebuilt `incrementaltrainerTests`, confirmed both new tests fail
  with the exact predicted symptoms (dangling symlink assertion failure; stale in-memory best path);
  restored the fix, rebuilt, confirmed all 9 relevant tests pass, plus the full `IncrementalTrainerTests`
  suite via `ctest`.

Files Changed:

- `src/IncrementalTrainer.cpp`
- `src/IncrementalTrainer.hpp`
- `tests/incrementaltrainer_test.cpp`

---

### TD-082: IncrementalTrainer's session_history.txt Silently Truncated checkpoint_path at the First Space

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | IncrementalTrainer (session history persistence / checkpoint resume) | Read the rest of the line for checkpoint_path instead of extracting it via `>>` |

Summary:
Found during a full second-pass re-read of `src/IncrementalTrainer.cpp` end to end (2088 lines) as part
of a full independent re-verification of the whole prior audit. `load_session_history()` parsed each
line of `session_history.txt` with
`iss >> session.session_id >> ... >> session.final_validation_loss >> session.checkpoint_path;` —
`checkpoint_path` extracted via the whitespace-delimited `operator>>`. `checkpoint_path` is a filesystem
path built from `get_session_dir()` (`IncrementalConfig::session_dir`, populated from the ordinary,
freely-user-configurable `SESSION_DIR` config key, default `"training_sessions"`), which can legitimately
contain a space (e.g. a human choosing `SESSION_DIR=my training data` or `"training sessions"`, an
entirely unremarkable directory-naming choice). Any such path is silently truncated at the first space
by `>>` — and critically, `iss.fail()` stays **false** in this case (extracting a shorter-than-intended
string via `>>` is not a stream failure), so the existing `if (iss.fail() || session.checkpoint_path.empty())`
guard does not catch it at all. The truncated (nonexistent) path is accepted as a valid checkpoint
reference, silently breaking "resume from best checkpoint" on every subsequent reload — the constructor's
best-checkpoint selection loop, and `resume_last_session()`'s checkpoint lookup, both end up checking
`fs::exists()` against a path that was never real, so training silently proceeds from scratch/whatever
weights happen to already be loaded instead of the intended best checkpoint, with no error surfaced.

Reproduced directly with a standalone program mirroring the exact write (`save_session_history()`) and
read (`load_session_history()`) logic: a `checkpoint_path` of `"training sessions/session_1_checkpoint.bin"`
round-tripped to `"training"` — with `iss.fail()` reporting `0` (false) throughout.

Changes Made:

- `load_session_history()` now extracts only the five numeric fields via `>>`, then reads the remainder
  of the line via `std::getline()` and strips only the field-delimiter whitespace immediately after
  `final_validation_loss`, so `checkpoint_path` (and its optional pipe-encoded
  `|losses:...|vallosses:...` extension, exactly as before) captures the full remainder of the line —
  spaces included — instead of stopping at the first one.

Verification:
- ✅ Standalone reproduction confirmed the bug and the fix, including that `iss.fail()` never actually
  flags the corruption pre-fix.
- ✅ Added `LoadSessionHistoryPreservesSpacesInCheckpointPath` and
  `LoadSessionHistoryPreservesSpacesInCheckpointPathWithExtendedFormat` to
  `tests/incrementaltrainer_test.cpp` (the latter also covering the v2 pipe-encoded per-epoch history
  extension immediately following a space-containing path).
- ✅ Before/after regression: reverted `IncrementalTrainer.cpp` to its pre-fix `HEAD` version, rebuilt
  `incrementaltrainerTests`, confirmed both new tests fail with the exact predicted truncation
  (`".../my"`) and the extended-format test additionally losing all per-epoch history; restored the fix,
  rebuilt, and confirmed all 6 `*LoadSessionHistory*`-filtered tests pass, plus the full
  `IncrementalTrainerTests` suite via `ctest`.
- ✅ Also confirmed via the same read-through that genuinely malformed lines (missing checkpoint_path,
  non-numeric session_id) are still correctly rejected by the unchanged `iss.fail()`/empty-path guard.

Files Changed:

- `src/IncrementalTrainer.cpp`
- `tests/incrementaltrainer_test.cpp`

---

### TD-081: TrainingMetricsAPI's Legacy "0-default" Alias Routes Returned the Wrong HTTP Status Code on Error

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | TrainingMetricsAPI (metrics_api_server legacy compatibility routes) | Add the missing `ApiRequestError` catch to every legacy alias route |

Summary:
Found during a full second-pass re-read of `src/TrainingMetricsAPI.cpp` end to end (2497 lines — the
largest file in the codebase). Every modern, session-key-scoped route (`GET/POST
/api/sessions/{key}/...`) wraps its handler call in `catch (const ApiRequestError& e) { ... res.status =
e.status_code(); }` *before* a generic `catch (const std::exception& e)` fallback, so a handler that
throws `ApiRequestError(404, "Unknown session key: ...")` — the normal outcome of
`resolve_session_service(key, false)` when that key doesn't exist yet — correctly surfaces as HTTP 404.
The pre-TD-018 "legacy" alias routes (`/api/metrics/current`, `/api/metrics/summary`,
`/api/metrics/history`, `/api/metrics/prometheus`, `/api/metrics/csv`, `/api/metrics/abnormal`,
`/api/metrics/generation-quality` GET+POST, `/api/metrics/padding-efficiency`, `/api/session/status`,
`/api/session/epochs`, `/api/session/end`, `/api/epoch/start`, `/api/epoch/end`, `/api/metrics/sample`,
`/api/metrics/validation`, `/api/metrics/best`, `/api/metrics/advanced`, `/api/control/flush`,
`/api/control/clear` — 20 routes in total, all hard-coded to operate on session key `"0-default"`) were
missing that specific catch clause entirely (only `/api/session/start` had it correctly). Since
`ApiRequestError` still derives from `std::exception`, it was still caught — just by the generic clause,
which discards `status_code()` and always returns a hard-coded 500 (GET routes) or 400 (POST routes)
instead. The practical effect: any of these 20 legacy endpoints returns a wrong status code (500/400
instead of 404) whenever `"0-default"` doesn't currently exist — the normal state right after
`metrics_api_server` starts, before any trainer has connected, or after the default session has ended/
been archived — misleading a monitoring client or dashboard into believing the server itself is broken
rather than simply reporting "no session yet."

Confirmed systematically (not just by inspection) with a script cross-referencing every
`server_impl_->server.Get/Post(...)` registration against whether its lambda body contained an
`ApiRequestError` catch: all 20 broken routes identified at once, and the 7 legitimately-excluded routes
(`/api/sessions`, `/api/metrics/compare`, `/api/metrics/aggregate`,
`/api/metrics/prometheus/aggregate`, `/admin/config`, `/api/models`, `/health`) confirmed to never call
`resolve_session_service()` at all, so they correctly need no such catch.

The gap went undetected by the existing test suite because `TrainingMetricsAPIRoutesTest::SetUp()`
always pre-creates `"0-default"` before every test, so no existing test ever exercised a legacy route
against a *missing* default session.

Changes Made:

- Added `catch (const ApiRequestError& e) { ...; res.status = e.status_code(); }` to all 20 affected
  routes, ordered before the existing generic `catch (const std::exception& e)`, matching the exact
  style of the one route (`/api/session/start`) that already had it correctly (including re-calling
  `set_legacy_deprecation_headers()` in the new catch branch, so the `Deprecation`/`Link` headers are
  still present on an error response, consistent with the success path).

Verification:
- ✅ Added a dedicated `TrainingMetricsAPIRoutesTestNoDefaultSession` fixture (deliberately skipping the
  base fixture's `SetUp()`-time `create_or_get_session("0-default")` call) with 3 new tests:
  `LegacyGetAliasReturns404WhenDefaultSessionDoesNotExist`,
  `LegacyPostAliasReturns404WhenDefaultSessionDoesNotExist`,
  `LegacyControlAliasReturns404WhenDefaultSessionDoesNotExist`.
- ✅ Before/after regression: reverted `TrainingMetricsAPI.cpp` to its pre-fix `HEAD` version, rebuilt
  `trainingMetricsApiRoutesTests`, and confirmed all 3 new tests fail with the exact predicted wrong
  status codes (500, 400, 500) instead of 404; restored the fix, rebuilt, and confirmed all 3 pass plus
  the full `TrainingMetricsAPIRoutesTests`/`TrainingMetricsAPILiveTests` suites via `ctest`.

Files Changed:

- `src/TrainingMetricsAPI.cpp`
- `tests/training_metrics_api_routes_test.cpp`

---

### TD-080: RAGInference's truncateContext() Returned a *Longer* String Than Its Input for max_tokens <= 0

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | RAGInference (retrieval-augmented generation context truncation) | Clamp max_chars and handle the too-small-for-an-ellipsis case explicitly |

Summary:
Found while reading `src/RAGInference.hpp`/`.cpp` end to end. `truncateContext()` computes
`max_chars = max_tokens * 4` (a rough chars-per-token approximation) and, when the context exceeds that
budget, returns `context.substr(0, max_chars - 3) + "..."`. When `max_tokens <= 0` (e.g.
`RAGConfig::max_context_length` set to `0` or misconfigured negative), `max_chars - 3` is negative;
`std::string::substr()`'s `count` parameter is `size_t`, so the negative `int` implicitly converts to a
huge unsigned value — which `substr()` then silently clamps back down to *the entire remaining string*.
The function ends up returning the whole, untouched context with `"..."` appended: literally longer than
its input, the exact opposite of "truncate to fit within token limit."

Reproduced directly: a 1000-character context truncated with `max_tokens=0` or `max_tokens=-5` came back
1003 characters long (the full original plus the three-character ellipsis) instead of empty/short.

Changes Made:

- `truncateContext()` now returns `""` immediately when `max_chars <= 0` (no budget for any content).
- Added an explicit `max_chars <= 3` branch (room for content but not for a 3-character ellipsis) that
  hard-truncates without appending "...", instead of falling through to the same negative-subtraction
  hazard.
- Added `friend class RAGInferenceTruncateContextTest;` to `RAGInference.hpp` (mirroring the existing
  `DataFetcherGutenbergCleaningTest` pattern in `DataFetcher.hpp`) so the private static
  `truncateContext()` can be unit-tested directly with hand-written fixture strings, without standing up
  a full model/document-store pipeline just to observe an internal prompt string.

Verification:
- ✅ Standalone reproduction confirmed the bug (1003-char output from a 1000-char input) and the fix
  (empty output for `max_tokens <= 0`, correctly-truncated output otherwise) across a matrix of
  `max_tokens` values including negative, zero, small-positive, and normal.
- ✅ Added 5 tests to `tests/raginference_test.cpp` (`RAGInferenceTruncateContextTest` fixture):
  `ZeroMaxTokensReturnsEmpty`, `NegativeMaxTokensReturnsEmpty`, `ResultNeverLongerThanInput`,
  `NormalTruncationStillAddsEllipsis`, `ContextShorterThanLimitReturnsUnchanged`.
- ✅ Before/after regression: reverted `RAGInference.cpp` to its pre-fix `HEAD` version (keeping the
  fixed header's new friend declaration, since that's test infrastructure rather than part of the bug),
  rebuilt `raginferenceTests`, and confirmed 3 of the 5 new tests fail with the exact predicted
  1003-vs-1000 mismatch; restored the fix, rebuilt, and confirmed all 36 tests in `raginferenceTests`
  pass via `ctest`.

Files Changed:

- `src/RAGInference.hpp`
- `src/RAGInference.cpp`
- `tests/raginference_test.cpp`

---

### TD-079: ConversationContext's Raw-Pointer system_message Caused a Double-Free/Use-After-Free on Copy or Move

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | ConversationContext (chatbot/GUI conversation history) | Store system_message as `std::optional<Message>` instead of a raw owning pointer |

Summary:
Found while reading `src/ConversationContext.hpp`/`.cpp` end to end — the most severe bug found in this
audit pass (a memory-safety defect, not just a logic error). `system_message` was declared as a raw
owning `Message*`, allocated with `new` in `set_system_message()` and freed with `delete` in the
explicitly hand-written destructor (`~ConversationContext() { delete system_message; }`) and in
`clear_all()`. But the class's copy constructor, copy assignment, move constructor, and move assignment
were all declared `= default` — which for a raw-pointer member just copies the pointer *value*. Two
`ConversationContext` instances could therefore end up pointing at the *same* heap-allocated `Message`:
destroying either one frees it, leaving the other with a dangling pointer (a subsequent read through it,
e.g. via `get_system_message()`, is a heap-use-after-free), and destroying the second instance afterward
`delete`s the same block a second time (a double-free). This violates the codebase's own stated
convention (CLAUDE.md: "Ownership: `std::unique_ptr`/`std::shared_ptr`; no raw owning pointers") and is a
classic Rule-of-Five violation: declaring a custom destructor for a raw owning pointer while leaving
copy/move as compiler-generated shallow copies.

Confirmed directly with AddressSanitizer: a minimal reproduction (construct `a`, call
`a.set_system_message(...)`, copy-construct `b` from `*a`, destroy `a`, then read `b`'s system message)
produced `SUMMARY: AddressSanitizer: heap-use-after-free ... in ConversationContext::get_system_message`,
with the ASan trace showing the read hitting memory freed by `a`'s destructor. In current production
code `ConversationContext` is always held via `std::unique_ptr` (`ChatbotAPI`, `ChatbotGUI`) so the
defaulted copy/move are never exercised there — but `create_summarized()` (a public, documented API
member) returns a `ConversationContext` *by value*, which relies on NRVO (an optional, not
standard-mandated, optimization) to avoid triggering the bug in practice, and any future caller that
copies or moves an instance with a system message set (assigns it, stores it in a container, etc.) would
hit it immediately — the existing test suite had no test that directly exercised the copy/move
constructors or assignment operators at all.

Changes Made:

- Changed `system_message` from `Message* system_message{nullptr}` to
  `std::optional<Message> system_message` (added `#include <optional>`). `Message`'s own members
  (two `std::string`s and an `int`) already have correct, safe implicit copy/move, so wrapping it in
  `std::optional` gives `ConversationContext`'s copy/move/destroy all-correct-for-free — no manual
  memory management needed anywhere in the class.
- Changed `~ConversationContext()` from a hand-written `delete system_message;` body to
  `~ConversationContext() = default;` (nothing left to manually release).
- Updated every touch point in `ConversationContext.cpp` (`set_system_message()`, `format_for_model()`,
  `format_with_special_tokens()`, `get_system_message()`, `clear()`, `clear_all()`,
  `truncate_to_limits()`, `save_to_file()`, `get_statistics()`, `create_summarized()`,
  `update_token_count()`) from `!= nullptr`/`new`/`delete` to `.has_value()`/`.emplace()`/`.reset()` —
  `->` member access on the optional needed no changes, since `std::optional` overloads `operator->`.
- Removed a genuinely dead local (`int system_tokens = ...`, already flagged by the compiler as
  `-Wunused-variable`) from `truncate_to_limits()` — computed but never referenced; left the loop's
  actual behavior unchanged since the docstring ("max total tokens in context") is consistent with the
  current behavior of counting the system message's tokens toward the budget, so this reads as vestigial
  dead code rather than an incomplete feature.

Verification:
- ✅ AddressSanitizer reproduction confirmed the bug pre-fix (`heap-use-after-free`) and a clean run
  post-fix with correct, independent copied state.
- ✅ Added four regression tests to `tests/conversationcontext_test.cpp` exercising all four special
  members directly with a system message set: `CopyConstructWithSystemMessageIsIndependent`,
  `CopyAssignWithSystemMessageIsIndependent`, `MoveConstructWithSystemMessagePreservesState`,
  `MoveAssignWithSystemMessagePreservesState` (deliberately not relying on `create_summarized()`, whose
  NRVO-eligible single-return-statement pattern would not reliably exercise the bug).
- ✅ Before/after regression: reverted both `ConversationContext.hpp`/`.cpp` to their pre-fix `HEAD`
  versions, rebuilt the new copy-construction test under ASan, and confirmed it fails with the exact
  predicted `heap-use-after-free` (freed by `clear_all()`'s `delete`, read by `get_system_message()`);
  restored the fix, rebuilt, and confirmed all 62 tests in `conversationcontextTests` pass both normally
  and under a standalone ASan build (`g++ -fsanitize=address`).

Files Changed:

- `src/ConversationContext.hpp`
- `src/ConversationContext.cpp`
- `tests/conversationcontext_test.cpp`

---

### TD-078: GPUManager's Allocation-Tracking Counters Raced Between the Allocating Thread and the SYCL Runtime's Deferred-Free Worker Thread

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | GPUUtils_SYCL (GPUManager, TD-041) | Guard `allocated_bytes_`/`max_memory_bytes_` with a dedicated mutex |

Summary:
Found while reading `src/gpu/sycl/GPUUtils_SYCL.hpp` end to end, made possible for the first time by
an Intel oneAPI SYCL toolchain (`icpx` 2026.1) unexpectedly being present in this environment — this
file's status tag had previously read "unverified (no SYCL toolchain available to build it)".
`GPUManager::allocated_bytes_`/`max_memory_bytes_` are plain (non-atomic) static `size_t` fields read
and written by `reserve_memory()`, `release_memory()`, and several getters — but `release_memory()` is
also called from inside a `sycl::handler::host_task` queued by `GPUMemory::defer_free()` (see that
function's own doc comment on why the free is deferred), which runs on a SYCL-runtime worker thread,
concurrently with whatever application thread calls `reserve_memory()` for the next allocation. This is
an unsynchronized concurrent read-modify-write from two different threads on the same memory — a data
race and undefined behavior regardless of whether any particular run happens to observe a wrong value.

Confirmed as a genuine, real race — not merely a theoretical one — with ThreadSanitizer: a standalone
reproduction exercising `GPUManager::reserve_memory()`/`release_memory()` from 8 concurrent threads (pure
host-side calls, no actual GPU device or SYCL queue involved, since the race is on the static bookkeeping
fields themselves) was flagged immediately: `WARNING: ThreadSanitizer: data race ... on
adai::gpu::GPUManager::allocated_bytes_`, with both conflicting accesses inside `reserve_memory()`
(`GPUUtils_SYCL.hpp:203`). A plain (non-TSan) run of the same reproduction did not reliably show a wrong
final tally at low contention — this is exactly the class of bug that ships silently until it doesn't.

Changes Made:

- Added a dedicated `memory_mutex_` guarding every access to `allocated_bytes_`/`max_memory_bytes_`:
  `reserve_memory()` (via a new private `try_reserve_locked()` helper used both for the fast path and
  the post-`synchronize()` retry), `release_memory()`, `get_used_memory_bytes()`,
  `get_available_memory_bytes()`, `get_memory_limit_bytes()`, `get_device_info()`'s budget line, and the
  two writes inside `initialize()`/`cleanup()`.
- `reserve_memory()`'s call to `synchronize()` (which blocks until any in-flight deferred free's
  `host_task` completes) is deliberately made **without** `memory_mutex_` held — that `host_task` itself
  needs to acquire `memory_mutex_` inside `release_memory()`, so holding the lock across `synchronize()`
  would deadlock against it.
- Added the missing `<mutex>` include, and `<vector>` (used by `enumerate_gpu_devices()`'s return type,
  previously compiled only via transitive inclusion from `<sycl/sycl.hpp>` — same class of portability
  gap as several other files fixed this audit pass).
- Updated the file's status-tag comment: the "unverified (no SYCL toolchain available to build it)"
  clause no longer applies now that a toolchain has been used to actually build it; TD-041's "no
  dedicated test" gap remains open (confirmed: no `GPUUtils_SYCL`-specific test file exists).

Verification:
- ✅ Standalone host-side reproduction under ThreadSanitizer (`icpx -fsycl -fsanitize=thread`) confirmed
  the race pre-fix and its absence post-fix (clean TSan run, same 8-thread/200k-iteration workload).
- ✅ Standalone `icpx -fsycl -Wall -Wextra` compile of the header alone — zero warnings, both before and
  after (the fix introduces no new warnings).
- ✅ Full integration build: configured and built the real `sycl` CMake preset (`ENABLE_SYCL=ON`) for
  the first time in this audit — `cmake --preset=sycl` then `cmake --build --target incremental_trainer`
  — which succeeded end-to-end, compiling and linking this header as part of the actual `adai_gpu`
  library and the full `incremental_trainer` binary.

Files Changed:

- `src/gpu/sycl/GPUUtils_SYCL.hpp`

---

### TD-077: TrainerControlState's wake()/interruptible_sleep() Lost a Wake That Arrived Before the Sleep Started

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | TrainerControlState (incremental_trainer serve admin API) | Add a predicate flag consumed by a predicate-checked `wait_for()` |

Summary:
Found while reading `src/TrainerControlState.hpp` end to end. `wake()` called `wake_cv_.notify_all()`
with no accompanying state change, and `interruptible_sleep()` called the plain (non-predicate)
`wake_cv_.wait_for(lock, duration)` overload. A `condition_variable` has no memory of a notification
that happened before anyone was waiting: if `wake()` fires during the window where the supervisory loop
(`incremental_trainer serve`'s `while (true)` loop in `IncrementalTrainingTool.cpp`) is doing its own
work — e.g. `resume_last_session()` concluding there is nothing pending — rather than already blocked
inside `interruptible_sleep()`, the notification is silently lost, and the *very next*
`interruptible_sleep(45)` call parks for the full 45-second poll interval regardless. This directly
contradicts `POST /admin/resume`'s documented contract (CLAUDE.md: "wake the idle-poll sleep so pending
work is checked immediately") for exactly the timing an operator calling that endpoint cares about.

The existing test suite's own comment on `InterruptibleSleepReturnsEarlyOnWake` even asserted the
mistaken belief that "wait_for still catches that case via its own internal check" — it does not; that
belief is precisely the bug.

Reproduced directly: a standalone program built against the real `TrainerControlState` class called
`wake()`, then (simulating the loop's brief non-sleep work) waited 20ms, then called
`interruptible_sleep(3)` — pre-fix, this returned after the full 3.000s; post-fix, it returned in 0.000s.
A control run with no `wake()` at all confirmed `interruptible_sleep()` still waits out its full requested
duration when nothing wakes it (no regression to the normal idle-poll case).

Changes Made:

- Added a `wake_requested_` bool (guarded by the existing `wake_mutex_`). `wake()` now sets it before
  calling `notify_all()`; `interruptible_sleep()` now uses the predicate-checked
  `wait_for(lock, duration, [this]{ return wake_requested_; })` overload and clears the flag on return.
  This correctly handles both a wake arriving during the wait (notified, predicate now true) and one that
  arrived before the wait began (predicate already true, `wait_for` returns immediately without blocking
  at all) — and continues to absorb genuine spurious OS-level wakeups the same as before, since the
  predicate-checked overload internally loops until the predicate is true or the timeout elapses.

Verification:
- ✅ Standalone reproduction against the real class confirmed both the bug (3.000s) and the fix (0.000s),
  plus the no-wake control case (still waits out the full requested duration).
- ✅ Added `TrainerControlStateTest.InterruptibleSleepDoesNotMissAWakeThatArrivesBeforeItStarts` to
  `tests/trainer_control_state_test.cpp`; corrected the neighboring test's comment that had asserted the
  incorrect belief about `wait_for`'s semantics.
- ✅ Before/after regression: reverted `TrainerControlState.hpp` to its pre-fix `HEAD` version, rebuilt
  `trainerControlStateTests`, confirmed the new test fails with the exact predicted ~5000ms result;
  restored the fix, rebuilt, confirmed all 12 tests in `trainerControlStateTests` pass, plus the
  dependent `TrainerAdminAPITests` suite (1/1) via `ctest`.

Files Changed:

- `src/TrainerControlState.hpp`
- `tests/trainer_control_state_test.cpp`

---

### TD-076: DataTransport's FTP URL Construction Broke on Filenames Containing Spaces or Reserved Characters

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | DataTransport (incremental_trainer FTP dataset fetch) | Percent-encode each `/`-separated ftp_path segment before building the FTP URL |

Summary:
Found while reading `src/DataTransport.hpp` end to end. `fetch()` built the FTP URL by directly
string-concatenating `token.ftp_path` — the registry-computed path of a pending dataset file relative
to `data_dir` — into a `ftp://user:pass@host:port/<ftp_path>` string with no escaping. `ftp_path` comes
straight from real on-disk filenames (`RegistryServer.cpp`'s `file_path.lexically_relative(data_root)`),
and Phase 11 dataset fetches (Gutenberg/HuggingFace downloads, manual uploads) routinely produce
filenames with spaces or punctuation — e.g. a Gutenberg title downloaded as `Pride and Prejudice.txt`.
libcurl does not implicitly percent-encode a URL passed via `CURLOPT_URL`: an unescaped space (or `#`,
`?`, etc.) makes `curl_easy_perform()` fail immediately with `CURLE_URL_MALFORMAT` ("URL using
bad/illegal format or missing URL") — and that code is correctly classified as non-transient by
`dt_detail::is_transient()` (already covered by the pre-existing `UrlMalformatIsNotTransient` test), so
the file is never retried and the fetch hard-fails on a routine filename with no actionable error
pointing at the real cause.

Reproduced directly against libcurl: a raw URL with an embedded space
(`ftp://user:pass@127.0.0.1:2121/some group/training data.jsonl`) returned `CURLE_URL_MALFORMAT` (rc 3);
percent-encoding each path segment (`some%20group/training%20data.jsonl`) changed the result to
`CURLE_COULDNT_CONNECT` (rc 7, since nothing was listening) — proving the URL became well-formed and
curl proceeded past parsing to the connection attempt.

Also found and fixed the same recurring missing-include pattern seen throughout this audit pass:
`std::ofstream` (`WriteCtx::file`, the local `outfile` in `fetch()`) needs `<fstream>`, which was never
included — it compiled only because the file's one production consumer, `IncrementalTrainingTool.cpp`,
happens to include `<fstream>` itself before `DataTransport.hpp`. A standalone compile of the header
alone (`-DBUILD_FTP_TRANSPORT`) reproduced the exact "invalid use of incomplete type 'std::ofstream'"
errors before the fix.

Changes Made:

- Added `dt_detail::url_encode_ftp_path()`: splits the path on `/`, percent-encodes each segment via
  `curl_easy_escape()` (called with a `NULL` handle — confirmed to work on this libcurl version — since
  no `CURL*` exists yet at URL-build time), and rejoins with literal `/` separators so directory
  structure survives encoding.
- `fetch()` now builds the URL via `dt_detail::url_encode_ftp_path(token.ftp_path)` instead of the raw
  field.
- Added the missing `<fstream>` include.

Verification:
- ✅ Standalone C reproduction against libcurl directly confirmed both the bug (`CURLE_URL_MALFORMAT`
  on an unencoded space) and the fix (`CURLE_COULDNT_CONNECT` once encoded — proving the URL parses).
- ✅ Added `UrlEncodeFtpPathTest` (4 cases: alphanumeric passthrough, space encoding, `#`/`?` encoding,
  leading/double-slash preservation) and
  `DataTransportFetchTest.FtpPathWithSpaceDoesNotTriggerUrlMalformat` to
  `tests/DataTransportFtpTests.cpp`.
- ✅ Before/after regression: reverted `src/DataTransport.hpp` to its pre-fix `HEAD` version, rebuilt
  `dataTransportFtpTests`, and confirmed `FtpPathWithSpaceDoesNotTriggerUrlMalformat` fails with the
  exact predicted message (`"...bad/illegal format or missing URL"`); restored the fix, rebuilt, and
  confirmed all 30 tests in `dataTransportFtpTests` and all 13 in `dataTransportTests` pass.
- ✅ Standalone `g++ -std=c++17 -Wall -Wextra` compile of `DataTransport.hpp` both with and without
  `-DBUILD_FTP_TRANSPORT` (the stub path) — zero warnings after the `<fstream>` fix.

Files Changed:

- `src/DataTransport.hpp`
- `tests/DataTransportFtpTests.cpp`

---

### TD-075: Quantization's ASYMMETRIC_INT8 Mode Corrupted Any Value Above the Signed int8_t Range

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Quantization (not shipped — see TD-038) | Reinterpret the stored byte as unsigned before widening back to `int` |

Summary:
Found while reading `src/Quantization.hpp` end to end — the most severe bug found in this file
family. `ASYMMETRIC_INT8` mode's quantized range is documented and used throughout as `[0, 255]` (see
`get_quant_range()`, `calibrate()`, `quantize_value()`'s clamping) — but every value that flows
through the vector-level `quantize()`/`dequantize()` API is stored in a `std::vector<int8_t>`, a
**signed** 8-bit type whose actual range is only `[-128, 127]`. `quantize()`'s explicit
`static_cast<int8_t>` stores the correct bit pattern via well-defined modular wraparound (e.g. 191
becomes -65) — but `dequantize()` passed that stored `int8_t` directly to `dequantize_value(int, ...)`,
where the implicit `int8_t` → `int` conversion **sign-extends** it back to -65, not 191. Every
quantized value above 127 — the entire upper half of `ASYMMETRIC_INT8`'s intended range, and the
*normal* case for realistic data such as post-ReLU activations (all non-negative, naturally using the
full `[0,255]` span) — silently came back corrupted, frequently with a flipped sign.

Reproduced with a standalone program: values `15.0`/`20.0` (unremarkable activation magnitudes) with
calibration data `{0, 5, 10, 15, 20}` quantized correctly to `191`/`255` but dequantized to
`-5.098`/`-0.078` — errors of over 20, in the wrong direction, for exactly the values the mode exists
to represent. No test file existed for this component at all prior to this fix.

Changes Made:

- Added a `decode_stored_value()` helper that reinterprets a stored `int8_t` as `uint8_t` before
  widening to `int` when `mode_ == ASYMMETRIC_INT8` — exactly undoing the storage-time wraparound —
  and used it in `dequantize()`. Other modes (`SYMMETRIC_INT8`'s `[-127,127]`, both INT4 ranges)
  already fit within `int8_t`, so plain sign-extension remains correct for them and is unchanged.
- Added the missing `<iostream>` include (`print_quantization_stats()` uses `std::cout` but the header
  only compiled via transitive inclusion from another header — same class of portability gap as
  TD-074's `MetricsTracker.hpp` fix, given TD-032 already documents a Windows/MinGW target).

Verification:
- ✅ Standalone reproduction confirmed the bug and the fix, including the exact before/after values.
- ✅ Created `tests/quantization_test.cpp` (no test file previously existed for this component) with
  6 tests covering all four quantization modes, including `AsymmetricInt8RoundTripAboveMidpointIsAccurate`,
  `AsymmetricInt8SingleValueQuantizedAbove127DoesNotGoNegative`, and
  `QuantizedMatrixRoundTripAsymmetricInt8`. Confirmed the three `ASYMMETRIC_INT8`-specific tests
  **fail** against the pre-fix code (with the exact reproduction values) and all 6 **pass** against
  the fix — reverted/rebuilt/re-applied to verify both directions. Registered as `QuantizationTests`
  in `tests/CMakeLists.txt`.

Files Changed:

- `src/Quantization.hpp`
- `tests/quantization_test.cpp` (new)
- `tests/CMakeLists.txt`

### TD-074: EfficientBatching's Bucketed Batches Could Silently Exceed max_tokens_per_batch

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Data Pipeline / Batching | Track the true running max length instead of comparing against the first element |

Summary:
Found while reading `src/EfficientBatching.hpp` end to end. `create_bucketed_batches()`'s greedy
batch-forming loop computed each candidate sequence's contribution to the batch's padded size via
`std::max(seq_len, sequences[batch_indices[0]].size())` — comparing only against the **first**
sequence ever added to the batch, not the true running maximum across everything already in it.
Since a bucket groups sequences by a coarse length *range* (e.g. everything up to a boundary), not an
exact length, a batch's true maximum can come from any sequence added to it, not just the first. Once
a longer sequence joined a batch, every later candidate was silently compared against the stale,
smaller first-element length instead of the real current max — letting the loop keep admitting
sequences well past `config.max_tokens_per_batch`, the very budget this function exists to enforce.

Reproduced with a standalone Python simulation of the exact loop: lengths `[50, 100, 30, 30, 30, 30]`
with `max_tokens_per_batch=300` all landed in one batch, whose true padded token count (`600`, once
padded to the real max length of 100) was double the configured limit. The codebase's own existing
test for this function (`BucketedBatches`) already asserted the limit was respected, but used a
`max_tokens_per_batch` generous enough relative to its test sequences' lengths that the boundary
condition was never actually stressed — so the assertion never failed despite the underlying bug.

Changes Made:

- Replaced the first-element comparison with a proper running `batch_max_len` that's updated to
  `std::max(batch_max_len, seq_len)` on every accepted sequence, so each admission decision uses the
  batch's true current maximum. Also removed a `current_tokens` local that was written but never read.

Verification:
- ✅ Standalone Python simulation confirmed the bug and the fix (see Summary).
- ✅ Added `EfficientBatchingTest.BucketedBatchesRespectTokenLimitWithMixedLengthOrder` to
  `tests/datapipeline_test.cpp`, using the exact reproduction lengths. Confirmed it **fails** against
  the pre-fix code (`600 vs 300`) and **passes** against the fix — reverted/rebuilt/re-applied to
  verify both directions.
- ✅ Full `datapipelineTests` (covers `EfficientBatching` and `ParallelDataLoader`): 33/33 pass.

Files Changed:

- `src/EfficientBatching.hpp`
- `tests/datapipeline_test.cpp`

### TD-073: IntegratedInferenceEngine's Batcher Emitted the First Request of Every Window Alone

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Inference / Batching (not shipped — see TD-038) | Reset `batch_deadline` when the first request of a new window arrives |

Summary:
Found while reading `src/IntegratedInferenceEngine.hpp` end to end. `batcher_worker()`'s
`batch_deadline` — meant to give a `batch_timeout_ms` window to accumulate requests into one batch —
was only ever reset to `now() + batch_timeout_ms` *after* a batch was successfully emitted. It was
otherwise initialized once, at thread start, and never touched while `pending_requests` sat empty
during any idle period. The very first request to arrive after such a period found `now()` already
past that stale deadline and was emitted alone immediately as a batch of 1, instead of starting a
fresh accumulation window — defeating the entire point of batching for exactly the request that
should have anchored the next batch. Any request arriving shortly after (well within the configured
timeout) landed in a separate batch instead of being merged with it.

Verified with a standalone Python simulation of `batcher_worker()`'s exact control flow (mirroring
the technique used earlier for TD-062): after a simulated idle period, two requests submitted 5ms
apart — well within a 50ms `batch_timeout_ms` — were emitted as two separate batches of 1 before the
fix, and as a single batch of 2 after it.

This engine is `@adai-status: beta`, capped by [TD-038](../guides/TECHNICAL_DEBT.md#td-038-advanced-features-tested-in-isolation-never-wired-into-a-shipped-binary)
(tested in isolation, never wired into any shipped binary), so the bug currently has no production
impact. It has dedicated tests (`integratedinferenceengine_test.cpp`), but — consistent with TD-038's
description — none of them exercise the actual threaded `submit()` pipeline (all lifecycle tests use
null model pointers and never submit a real request, since doing so requires a fully working
encoder/decoder/LM-head chain that neither this test file nor its `PipelineInferenceEngine`/
`BatchedInferenceEngine` siblings currently construct). A full end-to-end regression test was judged
disproportionate new test infrastructure for a currently-unreachable code path; the standalone
simulation is the appropriate verification level here, matching how earlier timing/loop-logic bugs in
this session (e.g. TD-062) were primarily verified.

Changes Made:

- `batcher_worker()` now resets `batch_deadline` to `now() + batch_timeout_ms` at the moment a request
  is added to a previously-empty `pending_requests`, so every accumulation window is anchored to when
  it actually started rather than to a stale prior deadline.

Verification:
- ✅ Standalone simulation confirmed the bug and the fix (see Summary).
- ✅ Full `integratedinferenceengineTests`: 47/47 pass (unaffected — none of them exercise this timing
  path, confirming the gap this fix addresses).

Files Changed:

- `src/IntegratedInferenceEngine.hpp`

### TD-072: ChatbotCLI's /set Command Crashed the Whole Interactive Session on a Typo

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | CLI / User-Facing | Wrapped the numeric conversions in a try/catch |

Summary:
Found while reading `src/ChatbotCLI.cpp` end to end. `handle_setting()`'s numeric parameters
(`length`/`max_length`, `temperature`/`temp`, `top_p`, `top_k`, `beam_width`) called
`std::stoi`/`std::stof` directly on the user-supplied value with no exception handling. A mistyped
value — `/set length abc`, `/set temp not-a-number` — threw `std::invalid_argument` straight out of
`handle_setting()`. `run()`'s main REPL loop has no try/catch around `handle_command()`, so the
exception propagated all the way up to `ChatbotCLI_main.cpp`'s top-level `catch`, which prints
"Fatal error: ..." and exits — a clean exit, not a raw crash, but one that ends the *entire
interactive chat session* (losing `session_id` and conversation continuity) over a single mistyped
`/set` command. This is a materially worse outcome than the same unguarded-`stoi`-on-CLI-input
pattern seen elsewhere in this codebase's one-shot tools (`IncrementalTrainingTool.cpp`,
`DatasetManagerTool.cpp`), where a bad argument just means re-running the command — here it destroys
state a REPL user would reasonably expect to survive a typo.

Changes Made:

- Wrapped the five numeric `/set` conversions in a single `try`/`catch`; a malformed value now
  prints `"Invalid value '<value>' for parameter '<param>'"` (matching the style of the adjacent
  `/set strategy` validation) and leaves the parameter unchanged, instead of throwing.

Verification:
- ✅ Added `ChatbotCLITest.HandleSettingNonNumericValueDoesNotThrow` to
  `tests/chatbotcli_improved_test.cpp`. Confirmed it **fails** against the pre-fix code
  (`std::invalid_argument` from both `stoi` and `stof`) and **passes** against the fix —
  reverted/rebuilt/re-applied to verify both directions.
- ✅ Full `chatbotcliTests` (83/83) and `chatbotcliImprovedTests` (24/24) pass.
- ✅ `chatbot` binary rebuilds clean.

Files Changed:

- `src/ChatbotCLI.cpp`
- `tests/chatbotcli_improved_test.cpp`

### TD-071: ConversationContext Silently Truncated Multi-Line Messages on Save/Load

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | NLP / Conversation persistence | Escape `\`/`\n`/`\r` so each message is always one line on disk |

Summary:
Found while reading `src/ConversationContext.cpp` end to end. `save_to_file()`/`load_from_file()`
use one line per message (`role|token_count|content`), written and read back with `std::getline()`.
Any message whose content contains an embedded newline — a routine chatbot scenario: a user typing a
multi-line message, or an assistant reply with paragraph breaks — split across multiple lines on
disk. On load, only the *first* line of such a message was parsed (silently truncating its content);
every continuation line had no `|` delimiter, failed the parser's own sanity check, and was silently
dropped via `continue`. This was a genuine, silent, permanent data-loss bug on every save+load
round-trip for any multi-line message, with zero test coverage of the case (all of the file's
existing persistence tests used single-line message content) — and it directly affects
`ChatbotGUI::onSaveConversation()`/`onLoadConversation()`, which call these methods for its real
"Save Conversation"/"Load Conversation" buttons.

Reproduced with a standalone program: saved a context with a 3-line user message and a 2-line system
message, reloaded it, and confirmed both were truncated to only their first line — the rest silently
gone, no error or warning of any kind.

Changes Made:

- Added `escape_for_line()`/`unescape_from_line()` (escaping `\`, `\n`, `\r`) and applied them to
  `content` in both `save_to_file()` (escape on write) and `load_from_file()` (unescape on read), so
  every message is guaranteed to occupy exactly one line on disk regardless of its content.

Verification:
- ✅ Standalone reproduction confirmed the pre-fix truncation and the fix's full round-trip fidelity
  for both a multi-line user message and a multi-line system message.
- ✅ Added `ConversationContextTest.SaveLoadRoundTripPreservesEmbeddedNewlines` to
  `tests/conversationcontext_test.cpp`. Confirmed it **fails** against the pre-fix code (content
  truncated to its first line) and **passes** against the fix — reverted/rebuilt/re-applied to
  verify both directions.
- ✅ Full `conversationcontextTests`: 58/58 pass.
- ✅ Rebuilt `chatbot_gui` (the real consumer of this save/load path) clean.

Files Changed:

- `src/ConversationContext.cpp`
- `tests/conversationcontext_test.cpp`

### TD-070: ModelNameClient's list_models() Corrupted Records on a "}}" Inside a Field Value

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | MNS Client | String-aware brace/quote scanning, matching TD-068's fix |

Summary:
Found while reading `src/ModelNameClient.cpp` end to end, immediately after fixing the same bug
class (TD-068) in `MnsManagerGUI`'s JSON helpers. `list_models()` found each model record's end by
searching for the literal substring `"}}"`  — relying on every record happening to end with two
adjacent closing braces (true only because `ModelNameService::serialize_record()` currently
serializes `tags` last) and, more seriously, breaking on **any** string field anywhere in the record
that happens to contain that same two-character sequence. `run_group` is a free-form, admin-settable
field (`mns_cli register --run-group <value>` / `update --run-group <value>`) with no server-side
format validation, unlike `model_name` (which is regex-constrained and so can never contain `}`) —
and `run_group` is serialized *before* `state`/`updated_utc` in the wire format, so a value containing
`}}` truncated the record before those two fields were reached, extracting them as empty strings.
`json_string_client()` (used by `list_models()` and every other parsing method in this file —
`resolve_model()`, `resolve_role()`, `get_architecture()`, `register_model()`, `set_training()`) had
the same escaped-quote gap as TD-068's `json_value()`: it found a string value's closing quote via a
plain `find('"', ...)` with no awareness of backslash-escapes.

Reproduced two ways: (1) a standalone program using the exact pre-fix logic, and (2) end-to-end
against a real running `mns_server` — registered a model with `run_group: "grp }} weird"`, then
called the real `ModelNameClient::list_models()` over HTTP. Before the fix, that model's `state` and
`updated_utc` came back as empty strings; after the fix, both are correct. `ModelNameClient` is
`@adai-status: stable` and is the MNS client used by every production binary that talks to the name
service (`chatbot_api_server`, `incremental_trainer`, `dataset_manager`, `mns_cli`'s underlying
library), so this had broader blast radius than TD-068's GUI-only scope — the interactive model
picker in `IncrementalTrainingTool.cpp`'s `resolve_model_name()` and `dataset_manager models` (just
fixed as TD-069) both call `list_models()` and would show wrong state/timestamp for any affected
model.

Changes Made:

- Added the same `find_string_end()`-style escape-aware helper used in `MnsJsonHelpers.hpp`
  (duplicated here since this file has its own independent minimal JSON reader) and used it in
  `json_string_client()`, fixing the escaped-quote truncation for every caller in this file.
- `list_models()` now uses proper string-aware brace-depth counting (skipping string content, honoring
  escapes) to find each record's true end, instead of searching for the literal `"}}"` substring.

Verification:
- ✅ Standalone reproduction and a real local `mns_server` end-to-end run both confirmed the bug
  before the fix and correct behavior after.
- ✅ Added `ModelNameClientTest.ListModelsHandlesBraceLikeSequenceInsideFieldValue` to
  `tests/modelnameclient_test.cpp`. Confirmed it **fails** against the pre-fix code and **passes**
  against the fix — reverted/rebuilt/re-applied to verify both directions.
- ✅ Full `modelnameclientTests`: 15/15 pass (including the pre-existing `list_models` test, whose
  fixture data — inherited from the old code's assumptions — still parses correctly under the new
  string-aware logic).

Files Changed:

- `src/ModelNameClient.cpp`
- `tests/modelnameclient_test.cpp`

### TD-069: dataset_manager's `models` Command Never Actually Listed Models

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Tooling / MNS Client | Use the existing `ModelNameClient::list_models()` |

Summary:
Found while reading `src/DatasetManagerTool.cpp` end to end. The `models` command's own help text
promises `"List registered models from name service"`, but the implementation never listed anything:
it parsed `host`/`port` out of the configured MNS URL (apparently in preparation for querying the
`/models` endpoint directly) and then never used either variable, falling back instead to
`client.resolve_model(svc_config.model_name)` — resolving at most the single model named by the
local `MODEL_NAME` config value, or printing "No MODEL_NAME configured" if that was unset. The
justifying comment — `"ModelNameClient doesn't expose list"` — was stale: `ModelNameClient::list_models()`
has existed all along and is already used elsewhere in this exact codebase
(`IncrementalTrainingTool.cpp`'s `resolve_model_name()`), so the workaround was unnecessary from the
start.

Verified end to end against a real running `mns_server` (a throwaway instance on a scratch port/data
dir): registered two models via `POST /models`, then ran `dataset_manager models` with
`NAME_SERVICE_URL` pointing at it. Before the fix, only the single `MODEL_NAME`-configured model (or
nothing, if unset) would ever show up regardless of how many models were actually registered; after
the fix, both registered models are listed with their state and role, matching `MnsManagerGUI`'s and
`IncrementalTrainingTool`'s own model-listing table format.

Changes Made:

- Replaced the dead host/port parsing and single-model `resolve_model()` fallback with
  `client.list_models()`, printed as a table (state, role, model name) matching the format already
  used by `resolve_model_name()` in `IncrementalTrainingTool.cpp` and the Models tab in
  `MnsManagerGUI.cpp`.

Verification:
- ✅ `dataset_manager` rebuilds clean.
- ✅ End-to-end run against a real local `mns_server` with two registered models: both are listed
  correctly (previously only one, or none, would show).

Files Changed:

- `src/DatasetManagerTool.cpp`

### TD-068: MnsManagerGUI's JSON Array Parser Corrupted Records on a Stray Brace in a String Value

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | GUI / MNS Manager | String-aware brace/quote scanning in `MnsJsonHelpers.hpp` |

Summary:
Found while reading `src/MnsManagerGUI.cpp` and its shared `src/MnsJsonHelpers.hpp` end to end.
`json_array_objects()` — used to populate both the "Models" and "Roles" tables — counted `{`/`}`
characters unconditionally to find each object's boundaries, with no awareness of whether it was
inside a JSON string literal. A model's free-form `tags` map (admin-entered `key=value` pairs) or any
other string field containing a literal `{` or `}` character (e.g. a tag value like `"see } section
3"`) desynchronized the depth counter — not just for that one object, but for **every object parsed
after it** in the same array, since the counter never returns to a consistent state. `json_value()`
had the same class of gap: it found the closing quote of a string value via a plain `find('"', ...)`
with no awareness of backslash-escapes, so a value containing an escaped quote (`\"`) was truncated
at the escape.

Reproduced with a standalone program: a 3-model `/models` response where the middle model's `tags`
contained a stray `}` inside a string value. Before the fix, the middle object was truncated early
and the third (real, unrelated) model's row was replaced by a corrupted, empty `{}` fragment — in the
GUI this would silently make a real model disappear from the table and show wrong/truncated data,
with no error or warning of any kind.

Changes Made:

- Added a shared `find_string_end()` helper that scans for a JSON string's closing quote while
  skipping backslash-escaped characters.
- `json_value()` now uses it when extracting a quoted value, fixing the escaped-quote truncation.
- `json_array_objects()` now skips over string content entirely (using the same helper) before
  testing a character for `{`/`}`/`]`, so only structural braces are counted — matching the
  string-awareness `json_pretty()` in the same file already had for indentation.

Verification:
- ✅ Standalone reproduction: a stray `}` inside one object's string value no longer corrupts that
  object or any object after it — confirmed byte-for-byte correct output before vs. after the fix.
- ✅ Added `MnsJsonValue.HandlesEscapedQuoteInStringValue` and
  `MnsJsonArrayObjects.StrayBraceInStringValueDoesNotCorruptParsing` to
  `tests/mns_manager_gui_test.cpp`. Confirmed both **fail** against the pre-fix code and **pass**
  against the fix — reverted/rebuilt/re-applied to verify both directions.
- ✅ Full `mnsManagerGuiTests` (excluding the 8 tests that require a live `mns_server`, which none of
  this environment has running): 35/35 pass.
- ✅ `mns_manager_gui` binary rebuilds clean.

Files Changed:

- `src/MnsJsonHelpers.hpp`
- `tests/mns_manager_gui_test.cpp`

### TD-067: DatasetRegistry's Legacy mark_trained() Overload Left Trained Files in the Pending Queue

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Training / Data Management | Added the same `pending_` cleanup the run_id overload already has |

Summary:
Found while reading `src/DatasetRegistry.cpp` end to end. `DatasetRegistry` has two `mark_trained()`
overloads: the newer Phase 9 `mark_trained(run_id, paths, sample_counts)` (used by every real
production call site — `IncrementalTrainingTool.cpp`, `IncrementalTrainer.cpp`) correctly removes the
now-trained paths from the in-memory `pending_` queue after recording them. The older 2-arg
`mark_trained(paths, sample_counts)` overload — which predates the Phase 9 run-based API and is no
longer called by any production code, only by tests — never got the same cleanup added, leaving a
file simultaneously marked trained *and* still listed as pending. That's a state `add_file()` itself
already refuses to create directly (it declines to re-add a file that `is_trained()` returns true
for), so a caller of this overload for a file already in `pending_` would end up with exactly the
inconsistent state the rest of the class works to prevent. No test exercised the add-then-mark-trained
sequence for this overload, so the asymmetry went unnoticed.

Changes Made:

- `mark_trained(paths, sample_counts)` now removes the newly-trained paths from `pending_`,
  mirroring the run_id overload's existing logic.

Verification:
- ✅ Added `DatasetRegistryTest.MarkTrainedRemovesFileFromPending`: adds a file (making it pending),
  marks it trained, and asserts it's reported trained and no longer pending. Fails without the fix
  (file remains in `pending_files()`), passes with it.
- ✅ Full `datasetRegistryTests` suite: 51/51 pass.

Files Changed:

- `src/DatasetRegistry.cpp`
- `tests/DatasetRegistryTests.cpp`

### TD-066: TextGenerator's Repetition Penalty Compounded Per-Occurrence Instead of Per-Token

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | NLP / Generation quality | Deduplicate `generated_tokens` before applying the penalty |

Summary:
Found while reading `src/TextGenerator.cpp` end to end. `apply_repetition_penalty()` iterated every
element of `generated_tokens` (the full running generated sequence) and divided (or multiplied, for
negative logits) the corresponding logit by `penalty` once per **occurrence** — so a token that had
appeared N times in the sequence so far got penalized by a factor of `penalty^N`, compounding without
bound as generation continued. The standard algorithm this feature is modeled on (CTRL, Keskar et al.
2019 — the paper that introduced repetition penalty as a decoding-time technique) penalizes a token
once if it has appeared **at all**, regardless of how many times; this is also how HuggingFace
`transformers`' `RepetitionPenaltyLogitsProcessor` and llama.cpp's sampler behave (both use a
membership/set semantic, not a count). No unit test isolated `apply_repetition_penalty()` directly —
the existing repetition tests only exercised it indirectly through `generate()`, none of which
distinguished the two semantics.

The practical effect: any token that legitimately needs to recur several times in a longer generation
(common function words, a repeated key noun, punctuation) would be exponentially suppressed the more
it had already appeared, making the model increasingly and disproportionately reluctant to reuse
*any* previously-used token as generation continued — a much stronger and more erratic effect than the
configured `repetition_penalty` value would suggest, worsening the longer the output.

Verified analytically and with a standalone Python check: with `penalty=1.2` and a token appearing
10 times, the buggy formula gives `5.0 / 1.2^10 ≈ 0.808` where the intended once-per-token semantic
gives `5.0 / 1.2 ≈ 4.167` — a difference of 5x, growing exponentially with occurrence count.

Changes Made:

- `apply_repetition_penalty()` now builds a `std::unordered_set<int>` from `generated_tokens` and
  applies the penalty once per distinct token in that set, instead of once per element of the
  original vector.

Verification:
- ✅ Added `TextGeneratorRepetitionTest.PenaltyDoesNotCompoundAcrossRepeats`: a fixed-logit mock model
  that favors one dominant token by a margin that survives exactly one penalty division but not two;
  asserts the dominant token is selected in every generated position under greedy decoding.
- ✅ Confirmed the new test **fails** against the pre-fix code (5 of 9 positions matched the dominant
  token, since the compounding penalty displaced it partway through) and **passes** against the fix
  (9 of 9) — reverted, rebuilt, and re-applied to verify both directions.
- ✅ Full `textgeneratorTests` suite: 36/36 pass.

Files Changed:

- `src/TextGenerator.cpp`
- `tests/textgenerator_test.cpp`

### TD-065: PostgresMetricsDatabase's list_sessions()/get_session() Lost All Data on a Nullable-Column Session Row

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Metrics / Postgres backend | Guarded the three nullable REAL columns with the existing `PQgetisnull()` pattern |

Summary:
Found while reading `src/PostgresMetricsDatabase.cpp` end to end. `list_sessions()` and
`get_session()` both called `std::stof(PQgetvalue(res, i, N))` unconditionally on
`best_validation_loss`, `final_loss`, and `final_validation_loss` — three `REAL` columns declared
nullable in the schema (no `NOT NULL`). `PQgetvalue()` returns `""` for a SQL `NULL`
(indistinguishable from a real empty string without `PQgetisnull()`), and `std::stof("")` throws
`std::invalid_argument`. `query_history()` already had to solve this identical problem for its own
TD-013 migration columns (an `opt_float` lambda checking `PQgetisnull()` first) — the same guard was
never applied to `list_sessions()`/`get_session()`'s three columns.

The concrete trigger is `bootstrap_schema()`'s own documented migration:
`ALTER TABLE sessions ADD COLUMN IF NOT EXISTS final_loss REAL;` /
`... final_validation_loss REAL;` have no `DEFAULT`, so any `sessions` row that existed in a
deployment before this migration ran keeps `final_loss`/`final_validation_loss` as `NULL` until the
next `upsert_session()` call for that specific row — a real, not merely theoretical, state (e.g. an
archived/ended session that's never touched again after the migration).

The failure mode is more severe than losing just the affected row: `execute_with_retry()` wraps the
whole query lambda in a `try`/`catch`, so the `std::stof` exception is caught there and the *entire
operation* is treated as a failed attempt and retried 3 times (with real ~2.1s of backoff) before
giving up — `list_sessions()` returns an **empty vector** (not just missing the bad row — every
other session in the same query result is silently dropped too, since the exception aborts the
result-processing loop) and `get_session()` returns `std::nullopt` even for the exact key requested.

Reproduced against a real local PostgreSQL 16 instance (`initdb`/`pg_ctl` in the scratch dir, no
system service touched): built `adai_core` with `-DENABLE_POSTGRES_METRICS=ON`, inserted one normal
session via `upsert_session()` and one raw-SQL row that omits `final_loss`/`final_validation_loss`
(reproducing exactly what a pre-migration row looks like), then called `list_sessions()`/
`get_session()`. Before the fix: `list_sessions()` returned 0 of the 2 sessions and `get_session()`
returned `nullopt`, with `[PostgresMetricsDB] list_sessions — exception: stof` logged on all 3
retry attempts. After the fix: both sessions are returned correctly and `get_session()` returns the
expected record.

Changes Made:

- Added a shared `pg_opt_float()` static helper (same `PQgetisnull()`-guarded pattern as
  `query_history()`'s local `opt_float` lambda) and applied it to `best_validation_loss` (fallback
  `std::numeric_limits<float>::max()`, matching `SessionRecord`'s own default), `final_loss`, and
  `final_validation_loss` (fallback `0.0f`, matching `SessionRecord`'s defaults) in both
  `list_sessions()` and `get_session()`.
- `best_epoch` (a nullable `INTEGER`) was not touched — `std::atoi("")` on a NULL value already
  returns `0` without throwing, which happens to match `SessionRecord::best_epoch`'s own default, so
  it was silently correct rather than silently wrong.
- No behavior change for any row where these columns are non-NULL (the normal case for every row
  written via `upsert_session()`, which always supplies concrete values).

Verification:
- ✅ Standalone reproduction against a real local Postgres 16 instance, confirmed failing before the
  fix and passing after (see Summary).
- ✅ `adai_core` (built with `-DENABLE_POSTGRES_METRICS=ON`) compiles clean with the fix.
- Not exercised by the existing test suite — `PostgresMetricsDatabase` has zero test coverage,
  tracked separately by [TD-042](../guides/TECHNICAL_DEBT.md#td-042-postgresmetricsdatabase-has-zero-test-coverage)
  (unchanged by this fix; TD-042's own action items — parameterizing `MetricsDatabaseTest.cpp`
  against this backend — would have caught this).

Files Changed:

- `src/PostgresMetricsDatabase.cpp`

### TD-063: ChatbotAPI's JSON Responses Could Be Injected Via an Unescaped session_id/error

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | API / Security | Shared `escape_json_string()` helper applied at all four gaps |

Summary:
Found while reading `src/ChatbotAPI.cpp` end to end. Four separate places in this file wrote a
string field straight into a hand-built JSON response with **no escaping**, while an adjacent field
in the very same function *was* carefully escaped char-by-char — an inconsistent pattern that
turned out to be a real, exploitable gap in all four cases:

1. `handle_chat_session()`'s inline response builder escaped `response` but not `session_id`.
2. `create_json_response()`'s failure branch wrote `error` unescaped (the success branch's
   `response` was escaped).
3. `create_batch_json_response()`'s `session_ids[]` array was unescaped (`responses[]` right next
   to it was escaped).
4. `create_batch_json_response()`'s failure branch wrote `batch_response.error` unescaped.

Both string sources are attacker-reachable: `session_id` can be entirely client-supplied —
`get_or_create_session()` uses a non-empty, not-yet-seen client-supplied `session_id` verbatim as
the new session's key — and `parse_json_string()`'s request-side unescaping correctly turns a
client's `\"` into a literal `"`, so a crafted request can get an actual quote character into the
raw `session_id` string. Every one of the five `/chat*` endpoint handlers' top-level `catch` blocks
passes `e.what()` straight to `create_error_response()`, and `generate_response()`/
`generate_batch_responses()`/`generate_batch_session_responses()` all wrap and rethrow with
`e.what()` embedded, so any exception message that happens to include user-influenced text reaches
the same unescaped path. A crafted value like `evil","injected":true` in either field produces a
syntactically valid but attacker-extended JSON response — e.g.
`..."session_id":"evil","injected":true"}` — letting a client inject sibling fields into (or, with
different content, break the structure of) a response the server never intended to send.

Changes Made:

- Added `ChatbotAPI::escape_json_string()` (public, matches the `\"`/`\\`/`\n`/`\r`/`\t` escaping
  already used correctly elsewhere in the file) as the single escaping implementation.
- Replaced all four unescaped call sites with it, and replaced the two already-correct-but-duplicated
  inline char-by-char loops (`response` in `handle_chat_session`, `responses[i]` in
  `create_batch_json_response`) with calls to the same helper, so there's only one escaping
  implementation left in the file to keep correct.
- Added five regression tests to `tests/chatbotapi_test.cpp`: a direct test of
  `escape_json_string()`'s behavior, and one test per fixed call site asserting that a crafted
  `evil","injected":true`-shaped value can no longer produce an `"injected":true` (or equivalent)
  sibling field in the output.

Verification:

- ✅ `chatbotapiTests` rebuilds clean; all 45 tests pass, including the 5 new regression tests in
  isolation.

Files Changed:

- `src/ChatbotAPI.hpp`
- `src/ChatbotAPI.cpp`
- `tests/chatbotapi_test.cpp`

---

### TD-062: train_epoch()'s Reported Loss/Grad-Norm Drifted Across Epochs

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Training / Reporting | Use the already-correct `update_count` local instead of re-deriving it from `global_step` |

Summary:
Found while reading `src/ChatbotTrainer.cpp` end to end. `train_epoch()` computed its reported
`epoch_loss`/`avg_grad_norm` (and the identical mid-epoch progress-log figures) by dividing
`total_loss`/`total_grad_norm` by a `num_updates` re-derived as
`global_step - epoch * (num_samples / config.gradient_accumulation_steps)`. That formula assumes
every epoch does exactly `num_samples / gradient_accumulation_steps` (**integer-divided**)
optimizer updates — but the loop always force-flushes a final, possibly-partial accumulation
window at the last sample of the epoch (`i == num_samples - 1`) regardless of its size, so whenever
`num_samples` isn't an exact multiple of `gradient_accumulation_steps`, every epoch actually does
one *more* update than the floor-divided formula assumes. The formula never corrected for this, so
the discrepancy compounded by +1 every epoch. Confirmed by simulating the exact loop logic
(100 samples, 32-step accumulation): the true update count is 4 every epoch, but the formula
computed 4, 5, 6, 7, 8, 9 across epochs 0–5 — by epoch 5 more than double the true count, meaning
`epoch_loss` was calculated as `(sum of 4 update-losses) / 9`, understating the reported training
loss by more than half.

**Scope of impact — narrower than a training-correctness bug:** this only affects the *reported*
aggregate figures — `get_final_training_loss()`, `get_training_losses()`, `get_gradient_norms()`,
the "Epoch N complete" log line, mid-epoch progress logs, and (via `IncrementalTrainer::run_training()`)
the `final_loss` field pushed into session history and MNS's `training_history` summaries.
Per-sample metrics pushed to the metrics dashboard (`update_sample_metrics`) use the correctly-scaled
`step_loss_for_cb`, not this aggregate, so those were never affected. **Actual training was never
affected either** — every optimizer step, gradient accumulation, and weight update happened exactly
as intended; only the epoch-level bookkeeping used for display was wrong. Best-checkpoint selection
(`best_validation_loss`) is also unaffected — it's computed independently in `validate()` as a
straightforward `total_loss / num_samples` over the validation set, with no accumulation-window
arithmetic at all.

The original "CRITICAL FIX" comment at this exact line reveals this was already a second attempt: a
prior fix corrected an even more basic bug (dividing by `num_samples` instead of update count) but
introduced this subtler one in the process, and neither one was caught because
`tests/chatbottrainer_test.cpp`'s `GradientAccumulationTest` suite only checks
`batch_size * gradient_accumulation_steps` config arithmetic, never running an actual multi-epoch
training pass with a non-divisible sample count.

Changes Made:

- Replaced both the mid-epoch and epoch-end `num_updates` computations with the already-correct,
  directly-incremented `update_count` local variable (`update_count + 1` mid-epoch, since that
  fires just before `update_count`'s own increment; plain `update_count` at epoch end, after the
  loop has finished incrementing it for every update that occurred) — no epoch-boundary arithmetic
  needed at all, so there's nothing left to drift.
- Added `ChatbotTrainerAbortTest.TrainingLossStaysSaneWithNonDivisibleAccumulationWindow` to
  `tests/chatbottrainer_test.cpp`: 5 training pairs with `gradient_accumulation_steps=3` (2
  updates/epoch: a 3-sample window then a force-flushed 2-sample window) across 4 epochs, asserting
  `get_global_step()` matches the exact expected count and every reported epoch loss is finite and
  positive. Can't assert an exact expected loss value (depends on the model's actual learned
  weights), but exercises the exact non-divisible shape that triggered the drift and would catch a
  regression to a wrong/zero divisor.

Verification:

- ✅ Standalone Python simulation of the exact loop logic confirms both the bug (formula computes
  4, 5, 6, 7, 8, 9 "updates" for epochs 0–5 of a 100-sample/32-step configuration that truly does 4
  every epoch) and that the fix removes the drift entirely (by construction — `update_count` is a
  direct count, not a re-derived estimate).
- ✅ `chatbottrainerTests` rebuilds clean; all 69 tests pass, including the new regression test in
  isolation.

Files Changed:

- `src/ChatbotTrainer.cpp`
- `tests/chatbottrainer_test.cpp`

---

### TD-061: GPU LayerNorm Backward (CUDA and SYCL) Computed Wrong Input Gradients

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | GPU / Core Training Math | One-token fix (`r*r*r` → `r*r`) in both backends, verified analytically and numerically |

Summary:
Found while reading `src/gpu/MatrixGPU.cu` end to end. `layer_norm_bwd_dx_kernel`'s gradient
w.r.t. the LayerNorm input had a genuine math bug affecting **every LayerNorm backward call on the
GPU training path** (multiple per encoder/decoder block, every layer, every step) — this is not a
minor rounding issue: verified by finite-difference check to be off by a large fraction of the
gradient's own magnitude, not a small numerical discrepancy.

Root cause: the kernel accumulates `sum(d_xn * xn)` — `xn` being the *already-normalized*
`(x-mean)*rstd` value, cached from the forward pass — then scales it by `-0.5 * rstd^3` to get the
gradient w.r.t. variance (`d_var`). The CPU implementation
(`LayerNorm::backward()` in `src/LayerNorm.cpp`) computes the mathematically equivalent quantity
using `sum(d_xn * (x - mean))` instead — the *pre-normalization* difference — and that version's
`-0.5 * rstd^3` scaling is correct **for that quantity**. Since `xn = (x-mean) * rstd`, reusing the
already-normalized value needs one *fewer* power of `rstd` to reach the same `d_var`; the GPU
kernel copied the CPU version's `rstd^3` scaling without adjusting for the different quantity it
was scaling, leaving an extra, spurious factor of `rstd` in every input gradient's variance
contribution.

Verified three ways:
1. **Analytical derivation** of the correct closed-form LayerNorm backward (`dx_j = rstd·h_j −
   (rstd/N)·xn_j·S − (rstd/N)·H`, where `h = d_xn`, `S = Σh_i·xn_i`, `H = Σh_i`) from first
   principles via the chain rule through `mean` and `rstd` (both functions of every `x_i` in the
   row) — confirms the correct power of `rstd` is 1, not 2, in the `xn_j·S` term the kernel's
   `d_var` feeds into.
2. **Numerical finite-difference check** (standalone Python/NumPy script implementing the exact
   kernel formula structure): the buggy formula (`r^3` in `d_var`) diverges from central-difference
   numerical gradients by up to ~0.3 in a toy example with gradients of magnitude ~0.02–0.6 — a
   large relative error, not float noise. The corrected formula (`r^2` in `d_var`, everything else
   unchanged) matches finite differences to ~1e-10 across 5 random trials of varying size.
3. **SYCL cross-check**: `src/gpu/sycl/MatrixGPU_SYCL.cpp`'s `matrix_layer_norm_bwd_gpu` has the
   byte-for-byte identical `r * r * r` bug — this CUDA kernel was ported "formula-for-formula" from
   the SYCL version (per this file's own section header comment), so the bug predates this session
   and affects both GPU backends identically. The CPU path (`LayerNorm.cpp`) is unaffected — it
   uses `(x - mean)` directly, which happens to need the correct power of `rstd` already.

No test exercises `matrix_layer_norm_bwd_gpu`/`layer_norm_bwd_dx_kernel` on either backend (grep
confirms zero references in `tests/`), consistent with [TD-041](../guides/TECHNICAL_DEBT.md#td-041-gpuutils-has-no-dedicated-test-on-either-backend)'s
broader finding that GPU-only code has minimal dedicated coverage — why this went undetected.

**Checkpoint impact — narrower than TD-059's:** this bug only affects the *backward* pass (gradient
computation during training), not the forward computation a saved checkpoint's weights encode.
Existing checkpoints trained with `-DENABLE_GPU=ON` or `-DENABLE_SYCL=ON` remain valid and usable —
their forward pass was never wrong — but their training likely converged less well than it should
have, since every LayerNorm's contribution to the gradient was systematically distorted throughout
training. No retraining is *required*, though anyone who trained primarily on GPU may want to
consider a fresh run now that the gradient is correct. CPU-only training was never affected.

Changes Made:

- `src/gpu/MatrixGPU.cu`: changed `d_var`'s scaling from `r * r * r` to `r * r` (one token), with a
  comment explaining the derivation and pointing at `LayerNorm.cpp` for the CPU-side reference
  quantity it must match.
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`: identical one-token fix, identical explanatory comment.
- `src/gpu/MatrixGPU.cu`: retagged `stable` → `beta` (0.9.0) — this specific bug is fixed, but the
  incident exposed that most of this file's kernels (not just `layer_norm_bwd`) have no dedicated
  test and, more fundamentally, can only ever execute on real GPU hardware — which was not available
  to verify any of them during this file's original "stable" rollout or since. `stable` was not an
  earned claim. Action item for whoever next touches this file: add dedicated tests for the
  remaining untested kernels (`matrix_add`/`multiply`/`transpose`, `gelu_backward`,
  `cross_entropy_loss`/`grad`, batch ops) alongside `matrixgpu_td003_test.cpp`'s existing coverage,
  and run the full GPU test suite on real hardware before re-promoting to `stable`.

Verification:

- ✅ Analytical re-derivation of the correct formula from the chain rule (see above).
- ✅ Standalone NumPy finite-difference check, both confirming the bug (pre-fix formula) and the fix
  (post-fix formula, ~1e-10 agreement) across multiple random trials.
- ✅ `adai_gpu` (containing `MatrixGPU.cu`) rebuilds clean under the local CUDA toolchain
  (`nvcc` present; no physical GPU in this environment, so the kernel could not be *run* here — the
  fix is verified mathematically and by a clean compile, not by an on-device numerical test).
- ⚠️ The SYCL build could not be compiled or tested in this environment either (no Intel oneAPI
  `icpx` toolchain available — same limitation noted in TD-041); the fix there is verified only by
  the formula being byte-for-byte identical to the now-fixed-and-compiled CUDA version.

Files Changed:

- `src/gpu/MatrixGPU.cu`
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`

---

### TD-060: EncoderDecoderModel::forward() Segfaulted on an Empty target_tokens

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Core Model / Training | Rewrote the loop bound to a form that can't underflow, plus a regression test |

Summary:
Found while reading `src/EncoderDecoderModel.cpp` end to end, in the course of investigating
TD-050's greedy-decode KV-cache workaround (confirmed that workaround still matches the tracker's
description exactly — nothing new there). `forward()` — a public method explicitly documented
"for custom training loops" — built the teacher-forcing decoder input with
`for (size_t i = 0; i < target_tokens.size() - 1; ++i)`. `size_t` is unsigned, so an empty
`target_tokens` makes `size() - 1` wrap to `SIZE_MAX`, turning the loop into an out-of-bounds read
of `target_tokens[i]` for ever-increasing `i`. Reproduced directly with a standalone
ASan/UBSan build: immediate `SEGV` on the first out-of-bounds `vector::push_back` read. Every
built-in caller (`train_step`/`evaluate`, via `tokenizer->encode(text, true)`) always produces a
non-empty `target_tokens` (at minimum bos+eos), which is why this was never hit through the normal
API surface — but `forward()`/`train_step_tokenized()`/`evaluate_tokenized()` all accept
already-tokenized vectors directly and are documented for exactly that use.

Changes Made:

- Changed the loop condition to `i + 1 < target_tokens.size()`, the same underflow-proof idiom used
  for TD-058, instead of `i < target_tokens.size() - 1`.
- Added `EncoderDecoderModelTest.ForwardEmptyTargetTokensDoesNotCrash` to
  `tests/encoderdecoder_test.cpp`.

Verification:

- ✅ Standalone ASan/UBSan repro confirms the exact SEGV crash mode before the fix.
- ✅ `EncoderDecoderTests` rebuilds clean and passes (this suite alone runs ~9 minutes as part of
  the full project's `ctest` — 77/77 suites, 100% pass, 0 failures, including this fix).

Files Changed:

- `src/EncoderDecoderModel.cpp`
- `tests/encoderdecoder_test.cpp`

---

### TD-058: BPETokenizer::get_most_frequent_pair() Could Underflow Its Loop Bound on an Empty Entry

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | NLP / Tokenizer | Rewrote the loop bound to a form that can't underflow, plus a regression test |

Summary:
Found while reading `src/BPETokenizer.cpp` end to end. `get_most_frequent_pair()` — a **public
static** method, not a private helper — counted adjacent-pair frequencies with
`for (size_t i = 0; i < tokens.size() - 1; i++)` for each inner `tokens` vector. `size_t` is
unsigned, so an empty `tokens` entry makes `tokens.size() - 1` wrap to `SIZE_MAX`, turning the loop
into an out-of-bounds read starting at `tokens[0]` on an empty vector. Every internal caller
(`build_bpe_merges()`) happens to only ever push non-empty entries into `word_tokens`, which is why
this was never hit in practice, but as a public API this method has no way to enforce that
invariant on a caller outside this file.

Changes Made:

- Changed the loop condition to `i + 1 < tokens.size()`, an idiom that is correct (and never
  underflows) for every value of `tokens.size()` including 0, instead of relying on callers to
  never pass an empty entry.
- Added `BPETokenizerTest.GetMostFrequentPairIgnoresEmptyWordEntries` to `tests/tokenizer_test.cpp`,
  calling the public static method directly with an empty entry mixed into `word_tokens` and
  asserting it neither throws/crashes nor lets the empty entry affect the correct result.

Verification:

- ✅ `runTests` (the `tokenizer_test.cpp` binary) rebuilds clean; all 53 tests pass.

Files Changed:

- `src/BPETokenizer.cpp`
- `tests/tokenizer_test.cpp`

---

### TD-057: LazyDataset::get_sample() Threw on a Bare "INPUT:"/"RESPONSE:" Legacy-Format Line

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Data / Dataset | Anchored prefix check + correct substr offset, plus a regression test |

Summary:
Found immediately after TD-056 while continuing the same end-to-end read of `src/Dataset.hpp`.
`LazyDataset::get_sample()`'s legacy-format branch used `line.find("INPUT:") != npos` /
`line.find("RESPONSE:") != npos` only as an existence check, then unconditionally sliced with a
fixed offset — `line.substr(7)` / `line.substr(10)` — assuming the key starts at column 0 and is
immediately followed by `": "`. Two independent problems: (1) a bare `"INPUT:"` or `"RESPONSE:"`
line (no trailing space/content — shorter than the assumed prefix) makes the offset exceed the
line's length, and `std::string::substr` throws `std::out_of_range` when `pos > size()` — confirmed
directly (`"INPUT:".substr(7)` throws `basic_string::substr: __pos (which is 7) > this->size()
(which is 6)`); (2) even for a normal line, `find()` matches the key anywhere in the string but the
substr offset is always relative to column 0, so a key not literally at the start would be sliced
from the wrong position. In practice every file this codebase itself writes
(`Dataset::save_to_file(..., "conversation")`) places the key at column 0, which is why this went
unnoticed, but any hand-edited or externally-supplied legacy-format file with a short/malformed line
could crash `LazyDataset::get_sample()` with an uncaught exception.

Changes Made:

- Replaced `find()` + fixed-offset `substr()` with `line.rfind("INPUT:", 0) == 0` /
  `line.rfind("RESPONSE:", 0) == 0` (anchored prefix checks, same idiom already used by
  `IncrementalTrainer::load_conversation_pairs()`'s legacy-format parser) and `substr()` at the
  prefix's actual length (6 / 9, not 7 / 10), followed by trimming leading whitespace — matches
  behavior for well-formed input while making a short/malformed line return an empty field instead
  of throwing.
- Added `DatasetTest.LazyDatasetHandlesBareLegacyKeys` to `tests/dataset_test.cpp`, asserting
  `get_sample()` doesn't throw on a file containing only bare `"INPUT:"`/`"RESPONSE:"` lines.

Verification:

- ✅ Standalone repro confirms the exact `std::out_of_range` thrown by the old code.
- ✅ `datasetTests` rebuilds clean; all 42 tests pass (41 prior + this one), including in isolation
  (`--gtest_filter=*BareLegacyKeys*`).

Files Changed:

- `src/Dataset.hpp`
- `tests/dataset_test.cpp`

---

### TD-056: Dataset::load_from_file() Called front() on an Empty String for Empty Files

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Data / Dataset | One-line guard + regression test |

Summary:
Found while reading `src/Dataset.hpp` end to end. `load_from_file()`'s format-detection chain reads
the first line into `first_line`, then checks `first_line.front() == '{'` to detect JSONL. For an
existing-but-empty (0-byte) file, `std::getline` leaves `first_line` empty and
`std::string::front()` on an empty string is undefined behavior per the C++ standard. Reproduced
directly: compiled a minimal repro with `-D_GLIBCXX_ASSERTIONS -fsanitize=address,undefined`
(libstdc++ 13) and it aborts with `Assertion '!empty()' failed`; in a normal (non-hardened) build it
returns unspecified data instead of crashing, silently falling through the detection chain rather
than reliably defaulting to conversation format. No existing test in `dataset_test.cpp` loaded a
genuinely empty (as opposed to non-existent) file, so this was never caught.

Changes Made:

- Added `!first_line.empty() &&` to the JSONL-detection branch's condition in `load_from_file()`.
- Added `DatasetTest.LoadEmptyFileDoesNotCrash` to `tests/dataset_test.cpp`, creating a real 0-byte
  file and asserting `load_from_file()` returns `false` (dataset stays empty) instead of crashing.

Verification:

- ✅ Standalone repro confirms the crash mode before the fix (`_GLIBCXX_ASSERTIONS` + ASan/UBSan).
- ✅ `datasetTests` rebuilds clean; all 41 tests pass, including the new regression test in
  isolation (`--gtest_filter=*EmptyFile*`).

Files Changed:

- `src/Dataset.hpp`
- `tests/dataset_test.cpp`

---

### TD-055: get_total_training_time_hours() Truncated Every Session to Whole Hours Before Summing

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | Training / Reporting | One-line fix: sum fractional hours instead of truncated ones |

Summary:
Found while reading `src/IncrementalTrainer.cpp` end to end. `get_total_training_time_hours()` (used
by `print_training_summary()`'s "Total time" line) computed each session's duration with
`std::chrono::duration_cast<std::chrono::hours>(end_time - start_time)` and summed the results.
`duration_cast` to `hours` truncates toward zero, so any single session under 60 minutes — the
common case for incremental/online training passes — contributed exactly 0 to the total, and a
95-minute session contributed 1 rather than ~1.58. The only existing test
(`GetTotalTrainingTimeHoursZeroInitially`) only covers the trivial no-session case, so this never
surfaced. A deployment training in short, frequent increments would see "Total time: 0.00 h" in the
summary log indefinitely regardless of how much wall-clock time had actually elapsed.

Changes Made:

- Changed the per-session duration computation to `std::chrono::duration<double,
  std::ratio<3600>>` (fractional hours) instead of `duration_cast<hours>`, and accumulate in
  `double` before narrowing to the `float` return type once at the end.
- Also removed a stray, contentless `// Metrics API Server Management` section-header comment
  left at the very end of the file (from the original March 2026 metrics-service commit; the
  actual management code lives in `TrainingMetricsAPI.{cpp,hpp}`, already reviewed and found
  complete) — it read like the file had been truncated mid-edit.

Verification:

- ✅ `incrementaltrainerTests` rebuilds clean and the full suite passes.

Files Changed:

- `src/IncrementalTrainer.cpp`

---

### TD-054: ModelNameService's Legacy JSONL-to-SQLite Migration Silently Dropped Every Record

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 8, 2026 | MNS / Persistence | One-line fix: bind the missing 26th column |

Summary:
Found while reading `src/ModelNameService.cpp` end to end. `init_db()` creates the `models` table
with 26 columns (`run_group` added last by a later migration — see the comment above
`add_column_if_missing("models", "run_group", ...)` explaining why it must stay last for
positional-INSERT compatibility). `persist_model()`, the actively-used save path, correctly binds
all 26 columns. `migrate_from_jsonl()` — the one-time import that runs when `models.db` is freshly
created and a legacy `models.jsonl` exists — used an INSERT literal with only 25 `?` placeholders
and never bound `run_group`. Reproduced directly with a standalone SQLite3 script against the
real schema: `sqlite3_prepare_v2` fails with `"table models has 26 columns but 25 values were
supplied"`, and the surrounding `if (... != SQLITE_OK) continue;` silently skips the record with
no logging — so any deployment upgrading from a pre-SQLite (or pre-`run_group`) MNS data directory
would import zero models from its `models.jsonl`, with nothing but a `migrated 0 records` log line
as a symptom (`roles.json` migration is a separate code path and unaffected).

Changes Made:

- Added the missing 26th placeholder to the `INSERT OR REPLACE INTO models VALUES (...)` literal
  in `migrate_from_jsonl()` and bound `r.run_group` (already parsed by `parse_record()`) to it,
  matching `persist_model()`'s statement exactly.
- Added an inline comment at the INSERT explaining the column-count invariant so a future column
  addition doesn't reintroduce the same silent failure in this specific call site.

Verification:

- ✅ Reproduced the failure and the fix against the real 26-column schema with a standalone
  Python/`sqlite3` script (25 placeholders → `OperationalError: table models has 26 columns but
  25 values were supplied`; 26 placeholders → succeeds, `run_group` lands correctly).
- ✅ `adai_mns` (the static library containing `ModelNameService.cpp`) rebuilds clean.

Action item not covered by this fix: `ModelNameService` (the server-side class, as opposed to
`ModelNameClient`) still has no dedicated unit test — only `modelnameclient_test.cpp` (client) and
`mns_manager_gui_test.cpp` (GUI) exercise it, both indirectly over HTTP. A test that actually drives
`migrate_from_jsonl()` against a scratch data dir would have caught this at write time; none exists
today. Filing this gap is out of scope for this fix — flagged for whoever next touches this file,
matching [TD-040](../guides/TECHNICAL_DEBT.md#td-040-ftpdataservers-auth-path-unreviewed-registryserver-untested-in-isolation)'s "no dedicated unit test" framing for a sibling daemon class.

Files Changed:

- `src/ModelNameService.cpp`

---

### TD-029: Fix GCC 13 ICE in raginference_test.cpp

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 7, 2026 (verified, not actively fixed — see note) | Tests / RAGInference | No longer reproduces on GCC 13.3.0 |

Summary:
Filed against an Internal Compiler Error (`cc1plus` SIGSEGV) that reportedly prevented
`raginferenceTests` from compiling on GCC 13. Re-verified during a pass confirming every active
TD item has an appropriate in-code marker: a clean `rm` of the object file followed by a fresh
`cmake --build . --target raginferenceTests` on this machine's GCC 13.3.0 compiled without error,
and all 31 tests in the suite pass. Most likely this was fixed upstream in a GCC 13 point release
between whenever this item was filed (June 7, 2026) and 13.3.0, though the original offending
construct was never identified, so it's possible a different GCC 13.x minor version could still
hit it. No code changes were made — `RAGInference.{cpp,hpp}` were promoted from `beta` to `stable`
since this was their only recorded blocker.

Verification:

- ✅ `raginferenceTests` builds clean on GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) after removing
  the stale object file first.
- ✅ All 31 tests in `raginferenceTests` pass.

If this resurfaces on a different GCC 13.x point release, re-open as a new item rather than
reverting this one — the root cause was never isolated, so there's no fix to "undo."

---

### TD-020: Persistent Metrics Storage via SQL Database

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 7, 2026 (tracker correction — implementation predates this entry) | Training / Metrics / API / Infrastructure | `IMetricsDatabase` abstraction, `SQLiteMetricsDatabase` (WAL mode), optional `PostgresMetricsDatabase`, `MetricsDatabaseFactory`, `MetricsSessionRegistry` DB ownership, four new REST endpoints, config keys, 884-line `MetricsDatabaseTest.cpp` |

Summary:
The proposal (`docs/development/archive/persistent-metrics-sql-storage.md`) was found fully implemented and verified during the per-file production-readiness rollout (see [file-status-standard.md](../guides/file-status-standard.md)) — this entry backfills the tracker, which had continued to list the item as "Active/Planned" after the work was actually done. Nine of the ten action items are complete and verified in code; one narrow sub-item (bundling the SQLite amalgamation specifically for Windows/MinGW cross-compilation, since the current CMake setup only finds a system-installed SQLite3) was never done and has been split off as its own item, [TD-032](../guides/TECHNICAL_DEBT.md#td-032-bundle-sqlite3-amalgamation-for-windows-cross-compilation).

Changes Made:

- ✅ `IMetricsDatabase` interface and `SessionRecord` struct defined in `src/MetricsDatabase.hpp`.
- ✅ `SQLiteMetricsDatabase` implemented with WAL mode (`PRAGMA journal_mode = WAL;`) and prepared statements (`src/SQLiteMetricsDatabase.hpp/.cpp`).
- ✅ Optional `PostgresMetricsDatabase` implemented (`src/PostgresMetricsDatabase.hpp/.cpp`).
- ✅ `MetricsDatabaseFactory::create(...)` implemented — declared in `src/MetricsDatabase.hpp`, defined in `src/SQLiteMetricsDatabase.cpp` (folded into the existing database header rather than a separate `MetricsDatabaseFactory.hpp` file as originally proposed; functionally equivalent).
- ✅ `IMetricsDatabase*` wired into `TrainingMetricsService` via `set_database()`.
- ✅ `MetricsSessionRegistry` owns and initializes the database instance (`src/MetricsSessionRegistry.hpp`), constructing it via `MetricsDatabaseFactory::create()` and injecting it into each session.
- ✅ Four new REST endpoints added to `TrainingMetricsAPI`: `/api/sessions/{key}/metrics/history` (time-range, explicitly commented "TD-020: DB-backed time-range history query"), `/api/metrics/compare` (cross-session), `/api/metrics/aggregate` (status-filtered live session list), `/api/sessions/{key}/metrics/export` (full history export).
- ✅ `METRICS_STORAGE_BACKEND`, `METRICS_DB_PATH`, `METRICS_DB_URL`, `METRICS_DB_POOL_SIZE` all present in `src/Config.hpp/.cpp` and `config.metrics.conf`.
- ✅ `tests/MetricsDatabaseTest.cpp` written (884 lines) covering schema bootstrap, WAL mode, round-trip insert/query, and related paths.
- ⬜ SQLite amalgamation bundling for Windows/MinGW builds — not done; split off as TD-032.

Files Modified:

- `src/MetricsDatabase.hpp` (new)
- `src/SQLiteMetricsDatabase.hpp` / `src/SQLiteMetricsDatabase.cpp` (new)
- `src/PostgresMetricsDatabase.hpp` / `src/PostgresMetricsDatabase.cpp` (new)
- `src/MetricsSessionRegistry.hpp`
- `src/TrainingMetricsService.hpp` / `src/TrainingMetricsService.cpp`
- `src/TrainingMetricsAPI.hpp` / `src/TrainingMetricsAPI.cpp`
- `src/Config.hpp` / `src/Config.cpp`
- `config.metrics.conf`
- `tests/MetricsDatabaseTest.cpp` (new)

Verification:

- ✅ `MetricsDatabaseTest.cpp` present and substantial (884 lines); `src/CMakeLists.txt` links SQLite3 into `adai_core` ("TD-020: Link SQLite3 into adai_core") and registers the SQLite metrics backend
- ✅ Confirmed by direct code inspection during this rollout, not by re-running the historical test suite

---

### TD-031: Fix RegistryServer Logger::init call signature mismatch

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 7, 2026 | RegistryServer / Build | One-line fix: replaced `Logger::init("registry_server")` with `Logger::init(Logger::Level::INFO, "registry_server")` in `src/RegistryServer.cpp:386`; `registry_server` target now builds cleanly |

Summary:
`RegistryServer.cpp` was calling `Logger::init` with a bare string as the first argument, matching a signature that no longer exists after the logger API was updated to require an explicit `Level` as the first parameter. The fix adds `Logger::Level::INFO` (the default used by all other binaries in the project) as the first argument, matching the canonical `Logger::init(Level, const std::string&)` overload.

Changes Made:

- `src/RegistryServer.cpp`: `Logger::init("registry_server")` → `Logger::init(Logger::Level::INFO, "registry_server")` at line 386.

---

### TD-030: GPU Strategy CLI Option for incremental\_trainer

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 7, 2026 | Training / IncrementalTrainer / GPU | Added `--gpu-strategy background\|full` CLI flag and `GPU_STRATEGY` config key; threaded `bool use_low_priority` from `ServiceConfig` through `Matrix::gpu_try_initialize()` to `GPUManager::initialize()`; 4 config-parsing tests added and passing |

Summary:
`incremental_trainer` previously used a hardcoded low-priority CUDA stream regardless of deployment context. This made it polite on shared workstations but underutilised on dedicated training machines. The `background` strategy preserves the original low-priority behaviour; `full` selects the highest-priority CUDA stream so training is never preempted. `GPU_MEMORY_FRACTION` remains an independent knob. The strategy is selectable via `--gpu-strategy` on the command line or `GPU_STRATEGY=` in the config file, with `background` as the default for backward compatibility.

Changes Made:

- `src/Config.hpp`: Added `enum class GPUStrategy : uint8_t { BACKGROUND, FULL }` with `gpu_strategy_from_string()` helper; added `GPUStrategy gpu_strategy = GPUStrategy::BACKGROUND` field to `ServiceConfig`; added `<cstdint>` and `<iostream>` includes.
- `src/Config.cpp`: Added `GPU_STRATEGY` parsing in both `load_from_file()` and `load_from_env()`.
- `src/gpu/GPUUtils.hpp`: Added `bool use_low_priority = true` parameter to `GPUManager::initialize()` (real and CPU-only stub); stream creation now selects `priority_low` or `priority_high` based on the parameter.
- `src/Matrix.hpp` / `src/Matrix.cpp`: Added `bool use_low_priority = true` parameter to `gpu_initialize()` and `gpu_try_initialize()` declarations and definitions; forwarded to `GPUManager::initialize()`.
- `src/IncrementalTrainingTool.cpp`: Added `--gpu-strategy` global flag stripping alongside `--config`; applies CLI override to `svc_config.gpu_strategy`; passes `use_low_priority` to `gpu_try_initialize()`; logs active strategy mode; updated usage text.
- `config.conf` / `config-remote.conf`: Added commented `GPU_STRATEGY` stub after `GPU_MEMORY_FRACTION`.
- `tests/config_test.cpp`: Added `unsetenv("GPU_STRATEGY")` to `clearEnvironmentVariables()`; added 4 tests: `GpuStrategyFileBackground`, `GpuStrategyFileFull`, `GpuStrategyUnknownDefaultsToBackground`, `GpuStrategyEnvVarOverridesFile`; all pass.

---

### TD-028: Separate Dataset Management from IncrementalTrainer

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 7, 2026 | Training / Data Management / IncrementalTrainer | 10-phase refactor introducing `DatasetRegistry`, `DataFetcher`, `RegistryTransport` (Local + Remote), `registry_server` HTTP daemon, and `dataset_manager` binary; 53 tests across 4 suites verified clean |

Summary:
`IncrementalTrainer` previously combined three unrelated concerns: the training loop, data-queue/registry management, and external data fetching (Gutenberg, HuggingFace). The 10-phase refactor separated these into focused components with clean boundaries. `DatasetRegistry` owns the pending queue, trained-file registry, checksums, and the `INPUT:/RESPONSE:` parser; it has zero network dependency and is fully testable with temp files. `DataFetcher` is a stateless class owning all download and conversion logic; each method returns a file path the caller enqueues via `DatasetRegistry::add_file()`. `IncrementalTrainer` was stripped to the training loop, session history, checkpoints, and metrics push, accepting file lists through `train_on_files()` / `retrain_on_files()`. A standalone `dataset_manager` binary links only `DatasetRegistry` and `DataFetcher`, enabling data preparation to run concurrently with a training process. In distributed mode (Phase 9), `DatasetRegistry` is backed by a `registry_server` HTTP daemon that multiple trainer instances query to atomically acquire disjoint subsets of the pending queue, preventing double-training across a pool of machines.

Changes Made:

- ✅ **Phase 1a–1c**: Created `DatasetRegistry`, `DataFetcher`, and a temporary `DatasetManager` backward-compat facade. All Gutenberg/HuggingFace logic and JSON helpers moved to `DataFetcher.cpp`. `#include <regex>` removed from `IncrementalTrainer.cpp`. All three `.cpp` files added to `adai_core`.
- ✅ **Phase 2**: Removed `data_registry_file`, `cache_tokenized_data`, `tokenized_cache_dir` from `IncrementalConfig`; added `DatasetConfig dataset_config_` to `IncrementalTrainer` populated via `DatasetRegistry::make_config(svc)`.
- ✅ **Phase 3**: Added `train_on_files()` and `retrain_on_files()` to `IncrementalTrainer`; deprecated `train_incremental()` and `train_full_retrain()` as delegating shims.
- ✅ **Phase 4**: Removed auto-loading from all three `IncrementalTrainer` constructors; shims now load/save via a local `DatasetRegistry` instance. Six CLI commands updated with explicit load/save calls. All 40 tests pass unchanged.
- ✅ **Phase 5**: `IncrementalTrainingTool` data commands (`add`, `gutenberg`, `gutenberg-batch`, `huggingface`) now construct only `DatasetRegistry`/`DataFetcher` — no model or tokenizer initialised.
- ✅ **Phase 6**: Added `src/DatasetManagerTool.cpp` and `dataset_manager` CMake target. Eight commands (`add`, `gutenberg`, `gutenberg-batch`, `huggingface`, `status`, `list-pending`, `list-trained`, `clear-pending`). No model dependency.
- ✅ **Phase 7**: Removed deprecated shims (`train_incremental()`, `train_on_new_data_only()`, `train_full_retrain()`) and the `DatasetManager` facade entirely. `resume_last_session()` and `IncrementalTrainingTool` `train`/`retrain` commands migrated to `train_on_files()` / `retrain_on_files()` with `DatasetRegistry`.
- ✅ **Phase 8**: Created `src/RegistryTransport.hpp/.cpp` — `DataVersion` (moved from `DatasetRegistry.hpp`), `PendingEntry`, abstract `RegistryTransport` interface, and `LocalTransport` (flat-file I/O). `DatasetRegistry` holds `std::unique_ptr<RegistryTransport> transport_`; transport-injection constructor added for testing. `RegistryTransportTests` (8 tests) added.
- ✅ **Phase 9**: Implemented `RemoteTransport` (cpp-httplib, guarded by `BUILD_METRICS_API_SERVER`) and `registry_server` HTTP daemon (`src/RegistryServer.cpp`, 6 endpoints, port 8082). Added `acquire_pending()`, `release_pending()`, `mark_trained(run_id,…)`, `print_run_assignments()` to `DatasetRegistry`. `LocalTransport` uses advisory `flock()` on a `.lock` sentinel file for per-host atomicity. Pending file format extended to `path\trun_id` (tab-separated; backward compatible). `ServiceConfig`/`DatasetConfig` extended with `REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID`, `REGISTRY_TIMEOUT_MS`. Transport factory auto-selects `LocalTransport` (no URL) or `RemoteTransport`. `RegistryTransportPhase9Tests` (10 tests) added.
- ✅ **Phase 10**: `IncrementalTrainer::resume_last_session()` and `IncrementalTrainingTool` `train`/`retrain` commands now call `acquire_pending(run_id)` instead of `load_pending_list()` + `pending_files()`, and `mark_trained(run_id,…)` instead of the three-call sequence (`mark_trained` + `clear_pending` + `save_pending_list`). `release_pending(run_id, paths)` called on failure to return files to the pool. `run_id` auto-derived from `hostname[:8] + "_" + pid%10000` via `detect_hostname_fragment()` / `detect_pid_mod_10000()` (IncrementalTrainer) or `derive_run_id()` helper (IncrementalTrainingTool).
- ✅ **Tests**: `DatasetRegistryTests` (31 tests across 4 suites: config defaults, `make_config`, `compute_checksum`, `load_conversation_pairs`, full pending-queue and trained-set API, persistence round-trips, and Phase 9 multi-run wrappers) and `DataFetcherTests` (4 offline tests) added to `tests/CMakeLists.txt`. All 53 TD-028 tests pass. Fixed pre-existing bug in `RegistryTransportPhase9Tests` where manually-constructed `DataVersion` objects with empty `checksum` caused the space-delimited registry parser to mis-align fields.

Files Created: `src/DatasetRegistry.hpp`, `src/DatasetRegistry.cpp`, `src/DataFetcher.hpp`, `src/DataFetcher.cpp`, `src/DatasetManagerTool.cpp`, `src/RegistryTransport.hpp`, `src/RegistryTransport.cpp`, `src/RegistryServer.cpp`, `tests/registry_transport_test.cpp`, `tests/RegistryTransportTests.cpp`, `tests/DatasetRegistryTests.cpp`, `tests/DataFetcherTests.cpp`

Files Modified: `src/IncrementalTrainer.hpp`, `src/IncrementalTrainer.cpp`, `src/IncrementalTrainingTool.cpp`, `src/Config.hpp`, `src/Config.cpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`

Files Removed: `src/DatasetManager.hpp`, `src/DatasetManager.cpp` *(Phase 1c backward-compat facade, removed in Phase 7)*

Related Items:

- TD-006: `create_qa_pairs_from_text()` now lives in `DataFetcher.cpp` — natural location for future FIM data generation.
- TD-014: `DataFetcher` is the natural home for future `adai-data-prep` tooling.
- TD-029: Fresh `Config.cpp` compilation in `build-gpu-clang` triggers the same GCC 13 ICE noted during Phase 6; cached object unaffected.

---

### TD-027: Install Script for incremental_trainer Sub-System

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 7, 2026 | Tooling / Deployment / IncrementalTrainer | Created `scripts/install_incremental_trainer.sh` supporting local, remote (SSH+rsync), and coordinator-only install modes; all 12 action items completed |

Summary:
`scripts/install_incremental_trainer.sh` was created to automate deployment of the `incremental_trainer` sub-system (all three binaries: `incremental_trainer`, `dataset_manager`, `registry_server`) to local and remote hosts. The script follows the established pattern from `install_systemd_service.sh` and `install_metrics_service.sh` (color helpers, `set -euo pipefail`, preflight checks, step-by-step logging, confirmation prompt). Local install creates the full directory layout, copies binaries with correct permissions, installs `config.conf` and `vocab.txt`, appends idempotent distributed-registry config stubs, creates the `adai` system user, sets ownership, and verifies both binaries execute correctly. Remote install performs all steps over SSH + rsync. Coordinator mode installs only `registry_server` and writes an `adai-registry.service` systemd unit. `scripts/README.md` was updated with a full usage table. Distributed-registry config stubs (`REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID`, `REGISTRY_TIMEOUT_MS`) were added to both `config.conf` and `config-remote.conf`.

Changes Made:

- ✅ Created `scripts/install_incremental_trainer.sh` with `set -euo pipefail`, color helpers (`info`, `success`, `warn`, `error`), and full argument parsing loop.
- ✅ Implemented all flags: `--install-path`, `--user`, `--group`, `--build-dir`, `--config-src`, `--vocab-src`, `--with-registry-server`, `--coordinator`, `--remote`, `--sync-sessions`, `--ssh-key`, `--help`.
- ✅ Local install: 7-step process covering user creation, directory layout (including `gutenberg_data/`, `huggingface_data/`), binary copy (mode 755), config/vocab copy, registry stub append, ownership, and post-install verification.
- ✅ Remote install: SSH mkdir, rsync binaries + config + vocab, remote chmod, remote registry stub append, optional `--sync-sessions` rsync, and remote post-install verification.
- ✅ Coordinator install: `registry_server` binary only, coordinator `config.conf` stub, `adai-registry.service` systemd unit installed, enabled, and started.
- ✅ Post-install verification runs `incremental_trainer status` (exit-code tolerant) and `dataset_manager --help` both locally and over SSH for remote installs.
- ✅ Updated `scripts/README.md` with full usage examples and options table.
- ✅ Added distributed-registry config stubs to `config.conf` and `config-remote.conf`.

Files Created: `scripts/install_incremental_trainer.sh`

Files Modified: `scripts/README.md`, `config.conf`, `config-remote.conf`

Related Items:

- TD-028 (Resolved): Separated dataset management — defined the three binaries this script deploys.
- TD-008 (Resolved): Daemon Service Implementation — `install_systemd_service.sh` style guide followed.
- TD-025 (Resolved): Background launch capability exposed to operators via the installed trainer.
- TD-018 (Resolved): `install_metrics_service.sh` is a companion script for the metrics API server.

---

### TD-023: Parallel Generation Quality Scoring via Model Snapshot

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 4, 2026 | Training / ChatbotTrainer / Metrics | Added `EncoderDecoderModel::clone()` (save/load via temp files), `std::optional<std::thread> generation_quality_thread_` on `ChatbotTrainer`, `join_generation_quality_thread()` helper, async branch in `compute_generation_quality_metrics()`, join in destructor and `release_model()`, `generation_quality_async_threshold` config key (default 50) wired through `TrainingConfig`, `ServiceConfig`, `Config.cpp`, `IncrementalTrainer::make_incremental_config()`, `config.conf`, and `config-remote.conf`; 10-test suite in `tests/generation_quality_async_test.cpp` |

Summary:
`ChatbotTrainer::compute_generation_quality_metrics()` previously ran synchronously on the training thread, blocking the start of the next epoch for the full `generate_response()` loop. When `generation_quality_sample_size >= generation_quality_async_threshold` (default 50), the function now clones the model weights into a temporary `EncoderDecoderModel` copy via `clone()` (which serialises to a unique temp path under `std::filesystem::temp_directory_path()` and immediately removes the files), launches a `std::thread` that scores against the snapshot, stores the thread in `generation_quality_thread_`, and returns immediately. The training loop can proceed without waiting. Before launching a new thread at the next epoch's validation phase, any previous thread is joined. The destructor and `release_model()` also join the thread, preventing use-after-free. Below the threshold the original synchronous path is unchanged.

Changes Made:

- ✅ Added `generation_quality_async_threshold = 50` to `TrainingConfig` in `src/ChatbotTrainer.hpp`.
- ✅ Added `#include <optional>` and `#include <thread>` to `src/ChatbotTrainer.hpp`; added `std::optional<std::thread> generation_quality_thread_` private member and `join_generation_quality_thread()` private helper declaration.
- ✅ Changed `~ChatbotTrainer() = default` to an explicit destructor in `src/ChatbotTrainer.cpp` that calls `join_generation_quality_thread()`.
- ✅ Implemented `join_generation_quality_thread()` — joins and resets the optional thread.
- ✅ Rewrote `compute_generation_quality_metrics()`: joins prior thread, branches on `sample_size >= config.generation_quality_async_threshold`; async path clones model and launches scoring thread; sync path unchanged.
- ✅ Updated `release_model()` to call `join_generation_quality_thread()` before releasing weights.
- ✅ Implemented `EncoderDecoderModel::clone() const` in `src/EncoderDecoderModel.cpp` using the existing `save_model` / `load_model` pair against a unique temp path; RAII removes all five temp files on success or exception.
- ✅ Added `clone()` declaration to `src/EncoderDecoderModel.hpp` with doc comment.
- ✅ Added `generation_quality_async_threshold = 50` to `ServiceConfig` in `src/Config.hpp`.
- ✅ Added `GENERATION_QUALITY_ASYNC_THRESHOLD` parsing in `src/Config.cpp`.
- ✅ Wired `generation_quality_async_threshold` in `IncrementalTrainer::make_incremental_config()`.
- ✅ Added `GENERATION_QUALITY_ASYNC_THRESHOLD=50` with memory-trade-off comment to `config.conf` and `config-remote.conf`.
- ✅ Created `tests/generation_quality_async_test.cpp` with 10 tests: config defaults, `ServiceConfig` field, `NullMetricsReporter` no-crash, sync path at below-threshold, async path at threshold, 2-epoch thread-join ordering, destructor safety, and sync/async score range validity.
- ✅ Added `generationQualityAsyncTests` target to `tests/CMakeLists.txt`; all 10 tests pass.

Files Modified: `src/ChatbotTrainer.hpp`, `src/ChatbotTrainer.cpp`, `src/EncoderDecoderModel.hpp`, `src/EncoderDecoderModel.cpp`, `src/Config.hpp`, `src/Config.cpp`, `src/IncrementalTrainer.cpp`, `config.conf`, `config-remote.conf`, `tests/CMakeLists.txt`

Files Created: `tests/generation_quality_async_test.cpp`

Related Items: TD-016 (Resolved) — introduced the synchronous `compute_generation_quality_metrics()` path that this item parallelises.

---

### TD-026: Extract GenerationQualityMetrics to Compiled Translation Unit

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 3, 2026 | Metrics / Tooling / Code Quality | Moved all `GenerationQualityEvaluator` method implementations out of the header into a new `src/GenerationQualityMetrics.cpp`; added the `.cpp` to `adai_core`; updated `generationQualityTests` to link `adai_core` |

Summary:
`GenerationQualityMetrics.hpp` was a fully header-only class. All six static method implementations (`evaluate`, `tokenize`, `count_ngrams`, `compute_corpus_bleu`, `compute_corpus_rouge_n`, `compute_corpus_rouge_l`, `lcs_length`) were extracted to `src/GenerationQualityMetrics.cpp`. The header now contains only the `GenerationQualityScore` struct, the `GenerationQualityEvaluator` class declaration with static method signatures, and the three standard-library includes required by the interface (`<map>`, `<string>`, `<vector>`). Implementation-only includes (`<algorithm>`, `<cctype>`, `<cmath>`, `<sstream>`) moved to the `.cpp`. The compiled object is now part of `adai_core`, so any future tooling target (e.g., `adai-eval`) links against it with one line.

Changes Made:

- ✅ Created `src/GenerationQualityMetrics.cpp` with implementations of all seven static methods.
- ✅ Trimmed `src/GenerationQualityMetrics.hpp` to declarations only; removed inline implementations and implementation-only `#include` directives.
- ✅ Added `GenerationQualityMetrics.cpp` to the `adai_core` source list in `src/CMakeLists.txt`.
- ✅ Updated `generationQualityTests` in `tests/CMakeLists.txt` to link `adai_core` (replacing the former header-only setup).
- ✅ `GenerationQualityTests` passes with no regressions.

Files Modified: `src/GenerationQualityMetrics.hpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`

Files Created: `src/GenerationQualityMetrics.cpp`

---

### TD-024: Remove Legacy Standalone ChatbotTrainer Code and Build Target

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 1, 2026 | Training / ChatbotTrainer / Build | Removed nine legacy standalone functions/methods from `ChatbotTrainer` and eliminated no-op `CHATBOT_TRAINER_TEST_BUILD` compile definitions from all CMake targets |

Summary:
`ChatbotTrainer` originally included a full standalone training pipeline (`train(const std::string&)`, `save_checkpoint`, `load_checkpoint`, `finalize_model`, `should_early_stop`, `restore_best_model`, `print_training_summary`, `test_generation`, `print_usage`) and the `TrainingConfig` checkpointing fields that drove them. `IncrementalTrainer` has fully superseded this role; none of the standalone code paths were reachable at runtime. All nine functions, their declarations, the `start_epoch` member variable, four `TrainingConfig` checkpoint fields, and the no-op `CHATBOT_TRAINER_TEST_BUILD` compile definitions were removed, reducing `ChatbotTrainer.cpp` by ~625 lines.

Changes Made:

- ✅ Removed `train(const std::string& output_model_path)` and its body from `src/ChatbotTrainer.cpp`.
- ✅ Removed `print_training_summary(long duration)`, `test_generation()`, and `print_usage()` from `src/ChatbotTrainer.cpp`.
- ✅ Removed `save_checkpoint()`, `load_checkpoint()`, `should_early_stop()`, `restore_best_model()`, `finalize_model()` from `src/ChatbotTrainer.cpp` (~625 lines total removed).
- ✅ Removed `start_epoch` member variable and all references from `src/ChatbotTrainer.cpp`.
- ✅ Removed all corresponding declarations from the public/private sections of `src/ChatbotTrainer.hpp`; removed `save_checkpoints`, `checkpoint_every`, `keep_all_checkpoints`, `resume_from_checkpoint` from `TrainingConfig`.
- ✅ Removed `target_compile_definitions(incremental_trainer PRIVATE CHATBOT_TRAINER_TEST_BUILD)` from `src/CMakeLists.txt`.
- ✅ Removed `CHATBOT_TRAINER_TEST_BUILD` from `chatbottrainerTests`, `incrementaltrainerTests`, and `incrementalTrainerDecouplingTests` in `tests/CMakeLists.txt`.
- ✅ No test cases in `tests/chatbottrainer_test.cpp` referenced the removed code; no test changes needed.
- ✅ `ChatbotTrainerTests` and `IncrementalTrainerDecouplingTests` pass with no regressions.

Files Modified: `src/ChatbotTrainer.cpp`, `src/ChatbotTrainer.hpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`

---

### TD-025: IncrementalTrainer Background Launch with PID Message

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 3, 2026 | Training / IncrementalTrainingTool / CLI | Added `launch_background()` in anonymous namespace; `train`, `retrain`, `resume` branches fork before trainer construction; parent prints structured startup banner and exits 0; child calls `setsid()` and redirects fds; Windows path uses `CreateProcess(DETACHED_PROCESS)` |

Summary:
`incremental_trainer train/retrain/resume` previously blocked the invoking shell for the full duration of training. The fix adds a `launch_background(int argc, char* argv[])` helper (POSIX: `fork()`+`setsid()`; Windows: `CreateProcess(DETACHED_PROCESS)` with `--background-child` sentinel). Each training dispatch branch reads the pending-file count and log path from config before forking, then:

- **Parent** prints a structured startup banner (`[ADAI] Training started in background — PID …`) and returns 0.
- **Child** redirects `stdin`/`stdout`/`stderr` to `/dev/null`, calls `setsid()`, and continues normal training (output flows through `adai::Logger`).

Falls back to foreground execution if `fork()` fails, logging a warning.

Changes Made:

- ✅ Added `launch_background(int argc, char* argv[])` in anonymous namespace at top of `src/IncrementalTrainingTool.cpp` with `#ifndef _WIN32` POSIX path and `#ifdef _WIN32` `CreateProcess` path.
- ✅ Added `--background-child` stripping in the global options parsing loop (Windows sentinel suppression).
- ✅ `train` branch: reads `pending_files.txt` before fork, checks for empty pending list before fork, calls `launch_background()`, prints banner from parent, child continues to `trainer.train_incremental()`.
- ✅ `retrain` branch: counts pending files before fork, calls `launch_background()`, prints banner from parent, child continues to `trainer.train_full_retrain()`.
- ✅ `resume` branch: calls `launch_background()`, prints banner from parent, child continues to `trainer.resume_last_session()`.
- ✅ Created `tests/incremental_trainer_background_test.cpp` covering fork/setsid behavior, PID uniqueness, banner format, log-path inclusion, and Windows `#ifdef` compile-time guard.
- ✅ Added `incrementalTrainerBackgroundTests` target to `tests/CMakeLists.txt`.

Files Modified: `src/IncrementalTrainingTool.cpp`, `tests/CMakeLists.txt`
Files Created: `tests/incremental_trainer_background_test.cpp`

---

### TD-022: Remove Direct Terminal Output from IncrementalTrainer and Dependencies

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| June 2, 2026 | Training / IncrementalTrainer / BPETokenizer / Transformer / Logging | Converted 120+ `std::cout`/`std::cerr` calls to `adai::Logger` across 4 source files; `display_dashboard()` gated behind `isatty()` |

Summary:
All direct terminal output in the `IncrementalTrainer` dependency graph has been replaced with structured `adai::Logger` calls, eliminating the two-channel output problem and enabling full log control via a single sink in daemon mode.

Changes Made:

- ✅ **`src/IncrementalTrainer.cpp`**: Converted `print_training_summary()`, `print_session_history()`, `print_data_registry()`, and `add_huggingface_dataset()` to `Logger::info()`/`Logger::error()`. Gated `display_dashboard()` behind `isatty(STDOUT_FILENO)` / `_isatty(_fileno(stdout))` — non-TTY path emits a compact `Logger::info()` summary; TTY path retains the full ANSI TUI unchanged. Removed the five `COLOR_*` ANSI macros and "Legacy ANSI codes" comment block.
- ✅ **`src/BPETokenizer.cpp`**: Replaced all `build_vocab()` / `build_bpe_merges()` / `pre_tokenize()` phase-banner and `\r`-overwrite progress `std::cout` calls with `Logger::info()`; removed `std::flush`/`\r` usage. Converted `save_vocab()`/`load_vocab()` confirmation `std::cout` to `Logger::info()`; `std::cerr` file-open error to `Logger::error()`; unknown-token and unknown-special-token `std::cerr` warnings to `Logger::warn()`.
- ✅ **`src/LayerNorm.cpp`**: Replaced `#include <iostream>` with `#include "Logger.hpp"`. Converted `print_config()` `std::cout` calls to `adai::Logger::info()`, `save_weights()`/`load_weights()` confirmations to `adai::Logger::info()`, and `set_gamma()`/`set_beta()` dimension-mismatch `std::cerr` errors to `adai::Logger::error()`.
- ✅ **`src/PositionalEncoding.cpp`**: Replaced `#include <iostream>` with `#include "Logger.hpp"` and added `#include <sstream>`. Converted `print_config()` `std::cout` calls to `adai::Logger::info()`, `forward()` sequence-length `std::cerr` warning to `adai::Logger::warn()`, and `visualize()` table rows built via `std::ostringstream` and emitted through `adai::Logger::info()`.

Files Modified: `src/IncrementalTrainer.cpp`, `src/BPETokenizer.cpp`, `src/LayerNorm.cpp`, `src/PositionalEncoding.cpp`

Verification:

- ✅ `adai_core` builds cleanly (`[100%] Built target adai_core`)
- ✅ `tokenizerErrorHandlingTests` builds and links successfully
- ✅ Only pre-existing `incrementaltrainerTests` failures remain (from TD-021: `metrics_push_enabled`/`metrics_config` member references)

---

### TD-021: IncrementalTrainer × Metrics Service Decoupling

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| May 31, 2026 | Training / Metrics / IncrementalTrainer / ChatbotTrainer | `IMetricsReporter` interface, `MetricsPushClient` background push client, `NullMetricsReporter`, bounded priority queue, session label auto-derivation, multi-session registry with sweep thread |

Description:
`IncrementalTrainer` directly instantiated `TrainingMetricsService` — a server-side class owning a ring buffer, file I/O, and push threads — inside the trainer process. All metrics now flow outward via HTTP only through a lightweight `MetricsPushClient`. `MetricsSessionRegistry` gained the TD-018 §4.8 background sweep thread, and `TrainingMetricsService::to_prometheus()` now emits per-session `{session="key"}` labels eliminating metric collisions across concurrent sessions.

Proposal: `docs/development/archive/incremental-trainer-registry-integration.md`

Changes Made:

- ✅ Create `src/IMetricsReporter.hpp` — abstract reporter interface + `AbnormalSample` struct (moved from `TrainingMetricsService.hpp`) + `NullMetricsReporter` no-op implementation.
- ✅ Create `src/MetricsPushClient.hpp` / `src/MetricsPushClient.cpp` — single background push thread, bounded priority queue (`Sample` events are lossy, `Epoch`/`Session` events are never dropped), 3× retry with back-off, 409-conflict retry loop on `start_session()`. `MetricsPushClient.cpp` added to `adai_core` in `src/CMakeLists.txt`.
- ✅ Replace `void set_metrics_service(TrainingMetricsService*)` with `void set_metrics_reporter(IMetricsReporter*)` in `ChatbotTrainer`; updated all ~15 `metrics_service_->` call sites to `metrics_reporter_->`. Removed `adv_cfg = metrics_service_->get_config()` call; outlier thresholds read directly from `TrainingConfig`.
- ✅ Remove `std::unique_ptr<TrainingMetricsService> metrics_service_` from `IncrementalTrainer`; replaced with `std::unique_ptr<IMetricsReporter> metrics_reporter_` + non-owning `MetricsPushClient* push_client_` alias. All three constructors initialize `NullMetricsReporter`. `MetricsPushClient` is constructed per training run in `train_incremental()` and `train_full_retrain()`. Added `get_metrics_session_key()` accessor. `train_incremental()` and `train_full_retrain()` wrap `MetricsPushClient` creation and `start_session()` in a 3-attempt 409-conflict retry loop (suffix progression: `base_key`, `base_key-2`, `base_key-3`).
- ✅ Replace `IncrementalConfig::enable_metrics_service` + `metrics_push_enabled` + `MetricsServiceConfig metrics_config` with flat fields `metrics_server_url`, `metrics_session_label`, `metrics_push_timeout_ms`; empty URL → `NullMetricsReporter`. Removed `#include "TrainingMetricsService.hpp"` from `IncrementalTrainer.hpp`.
- ✅ Move `loss_outlier_z_threshold`, `grad_norm_outlier_threshold`, `max_abnormal_samples` from `MetricsServiceConfig` into `TrainingConfig` (`src/ChatbotTrainer.hpp`). Added `abnormal_sample_count_` member to `ChatbotTrainer`; `flag_abnormal_sample()` calls are now capped at `config.max_abnormal_samples` per training run. Added all three fields to `ServiceConfig` (`src/Config.hpp`) and wired them through `make_incremental_config()`.
- ✅ Implement session label auto-derivation (`"#{id}: {stem} ({host}, {date})"`): `derive_metrics_session_label()` and `build_config_snapshot()` added to the anonymous namespace in `src/IncrementalTrainer.cpp`. Both `train_incremental()` and `train_full_retrain()` now pass `label` (user value or auto-derived) and `snapshot` JSON to `start_session()`. Auto-derivation uses `fs::path(model_path_).stem()`, `detect_hostname_fragment()`, and `strftime`; snapshot captures `d_model`, `heads`, `d_ff`, `enc_layers`, `dec_layers`, `lr`, `batch`, `grad_accum`.
- ✅ Add `METRICS_SESSION_LABEL` to `config.conf` and `config-remote.conf`; remove `METRICS_PUSH_ENABLED` (empty `METRICS_SERVER_URL` is now sufficient to disable push). Added `metrics_session_label` field to `ServiceConfig` (`src/Config.hpp`) and parsed from config file and env var in `src/Config.cpp`. Removed `ServiceConfig::metrics_push_enabled`; `make_incremental_config()` now maps `svc.metrics_server_url` directly.
- ✅ Add background sweep thread to `MetricsSessionRegistry` (TD-018 §4.8): `sweep_thread_`, `stop_sweep_` atomic, `sweep_cv_` + `sweep_mutex_` — constructor gains 4th param `sweep_interval_seconds` (default: 60); thread started when > 0; destructor sets `stop_sweep_ = true`, notifies, joins.
- ✅ Add `label` and `config_snapshot` fields to `MetricsSessionSummary` and `SessionEntry` (`src/MetricsSessionRegistry.hpp`); update `list_sessions()` to populate them.
- ✅ Add `label` and `config_snapshot` params to `TrainingMetricsService::start_session()`; store as member vars (`label_`, `config_snapshot_`); add to `TrainingMetricsSnapshot`; `list_sessions()` reads them from snapshot. Add optional `session_key` param to `to_prometheus()` and `to_prometheus_internal()`; emits `{session="key"}` labels on each metric line when non-empty.
- ✅ Pass `sweep_interval_seconds` (default 60) as 4th argument to `MetricsSessionRegistry` constructor in `src/TrainingMetricsAPIServer.cpp`; added `--sweep-interval-seconds` CLI flag to `ServerConfig`.
- ✅ Add `GET /api/metrics/prometheus/aggregate` endpoint to `TrainingMetricsAPI` that concatenates per-session labelled Prometheus output; `handle_prometheus_metrics()` now passes `session_key` to `to_prometheus()`; `handle_post_session_start()` parses and forwards `label` and `config` fields from POST body.
- ✅ Write `tests/incremental_trainer_registry_test.cpp` (new, 11 tests): `NullMetricsReporter` all-methods-no-crash, `MetricsPushClient` construction/offline/201/409/retry-suffix, `IncrementalConfig` metrics fields, `IncrementalTrainer::get_metrics_session_key()` initially empty.
- ✅ Write `tests/metrics_push_client_test.cpp` (10 tests): queue overflow policy, retry/back-off, 409 no-retry, shutdown drain, `start_session` body, destructor safety.
- ✅ Add 9 tests to `tests/metrics_session_registry_test.cpp`: sweep-thread eviction, `label`/`config_snapshot` propagation, `get_session()` optional semantics.
- ✅ Update `docs/development/TRAINING_METRICS_API.md`: multi-session API catalog, Prometheus aggregate endpoint, `POST /start` body schema, `--sweep-interval-seconds` CLI option.

Files Created: `src/IMetricsReporter.hpp`, `src/MetricsPushClient.hpp`, `src/MetricsPushClient.cpp`, `tests/incremental_trainer_registry_test.cpp`, `tests/metrics_push_client_test.cpp`

Files Modified: `src/IncrementalTrainer.hpp/.cpp`, `src/ChatbotTrainer.hpp/.cpp`, `src/TrainingMetricsService.hpp/.cpp`, `src/TrainingMetricsAPI.hpp/.cpp`, `src/TrainingMetricsAPIServer.cpp`, `src/MetricsSessionRegistry.hpp`, `src/Config.hpp/.cpp`, `src/CMakeLists.txt`, `config.conf`, `config-remote.conf`, `tests/metrics_session_registry_test.cpp`, `tests/CMakeLists.txt`, `docs/development/TRAINING_METRICS_API.md`

---

### TD-019: Stale Metrics Detection and Liveness Accuracy

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| May 31, 2026 | Training / Metrics / API / Config / Dashboard | Staleness threshold config, snapshot stale fields, `get_current_snapshot()` fix, `to_json()` stale fields, API staleness fields, dashboard stale badge, `staleDetectionTests` |

Description:
Metrics endpoints reported `is_training=true` long after ingest stopped because `get_current_snapshot()` overwrote `last_update_time` with the current clock on every read, masking the true last-ingest time. The dashboard used browser-local time for "Last updated", showing a false freshness indicator during trainer/network failures.

Changes Made:

- ✅ **Config** — Added `metrics_staleness_threshold_seconds` (default: 60) to `ServiceConfig` (`src/Config.hpp`), parsed from config file and env var in `src/Config.cpp`. Wired through `MetricsServiceConfig.staleness_threshold_seconds` in `src/IncrementalTrainer.cpp`. Added key to `config.conf` and `config-remote.conf`.
- ✅ **Snapshot fields** — Added `is_stale`, `seconds_since_last_update`, `effective_is_training` to `TrainingMetricsSnapshot` in `src/TrainingMetricsService.hpp`.
- ✅ **Root cause fix** — Removed `snapshot.last_update_time = std::chrono::system_clock::now()` from `get_current_snapshot()` in `src/TrainingMetricsService.cpp`; staleness fields now computed from preserved ingest timestamp. Same logic added to `to_json()` (which runs independently to avoid lock re-entry).
- ✅ **JSON output** — `to_json()` now appends `is_stale`, `seconds_since_last_update`, `effective_is_training` fields to all snapshot JSON responses.
- ✅ **API endpoints** — `handle_session_status()` adds the three stale fields to its JSON response; `handle_health_check()` uses `effective_is_training` for per-session liveness and exposes `any_stale` in `src/TrainingMetricsAPI.cpp`.
- ✅ **Dashboard** — Added `.stale-badge` CSS, stale badge `<span>` in HTML header, updated `updateDashboard()` to use server-provided `seconds_since_last_update`, show "X seconds ago (stale)" label, and use `effective_is_training` for status colour in `dashboard.html`.
- ✅ **Tests** — Created `tests/stale_detection_test.cpp` with 16 tests across 6 suites (`StaleDetectionIngestTimestamp`, `StaleDetectionSecondsAgo`, `StaleDetectionIsStale`, `StaleDetectionEffectiveIsTraining`, `StaleDetectionJson`, `StaleDetectionConfig`). Registered as `staleDetectionTests` CMake target in `tests/CMakeLists.txt`.

Verification:

- ✅ All 16 `staleDetectionTests` pass in `build-gpu-clang`
- ✅ No regressions in `adai_core` build

---

### TD-018: Multi-Instance Training Metrics Service

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| May 31, 2026 | Training / Metrics / API / Config | `MetricsSessionRegistry`, session-scoped routes, trainer session-key wiring, `GlobalMetricsService` proxy, config keys, API docs |

Description:
The metrics stack assumed a single active training session and used shared file paths and flat API routes. Running multiple trainers against one metrics API server could overwrite metrics, cross-contaminate session state, and make dashboards unreliable. All 10 phases of the proposal (`docs/development/archive/multi-instance-metrics-service.md`) are now complete.

Changes Made:

- ✅ **Phase 1** — `TrainingMetricsService` now prepares configured parent directories for all output files, enabling fully caller-supplied per-instance paths. Regression test added in `tests/training_metrics_service_resume_test.cpp`.
- ✅ **Phase 2** — Created `src/MetricsSessionRegistry.hpp`: owns `unordered_map<string, shared_ptr<TrainingMetricsService>>` under a `shared_mutex`, implements `create_or_get_session()`, `get_session()`, `list_sessions()`, and TTL-based `evict_completed_sessions()`. Enforces `max_live_sessions` cap (default 16). Derives per-session file paths (`{key}_metrics.jsonl`, etc.) while preserving legacy paths for `"0-default"`.
- ✅ **Phase 3** — `TrainingMetricsAPI` rewritten to accept `MetricsSessionRegistry*`. All routes moved to `/api/sessions/{key}/...` prefix. Added `GET /api/sessions` (session index) and `GET /api/metrics/aggregate` (live cross-session view). Legacy flat routes preserved as `"0-default"` aliases emitting `Deprecation: true` / `Link:` headers.
- ✅ **Phase 4** — `TrainingMetricsAPIServer` constructs `MetricsSessionRegistry` from `Config` and injects it into `TrainingMetricsAPI`; removed direct construction of a single `TrainingMetricsService`.
- ✅ **Phase 5** — `src/Config.hpp/.cpp`: added `metrics_session_key`, `metrics_max_live_sessions` (16), `metrics_completed_ttl_seconds` (3600), `metrics_sweep_interval_seconds` (60); parsed from both config file and environment variable overrides.
- ✅ **Phase 6** — `src/IncrementalTrainer.cpp`: added `sanitize_session_key()`, `derive_metrics_session_key()` (auto-derives `{id}-{hostname}{pid%10000}` when key is empty), `build_metrics_session_push_base()`, and updated `make_incremental_config()` to set the session-scoped push URL (`{METRICS_SERVER_URL}/api/sessions/{key}`).
- ✅ **Phase 7** — `GlobalMetricsService` updated to proxy through the `"0-default"` slot of a `MetricsSessionRegistry` singleton, preserving the existing `instance().start_session(id, ...)` call-site API.
- ✅ **Phase 8** — Added multi-session tests in `tests/metrics_session_registry_test.cpp` (`ConcurrentSessionCreationIsThreadSafe`, `ConcurrentReadsAndWritesDoNotDeadlock`, `ConcurrentSessionsHaveIsolatedData`) and `tests/training_metrics_api_routes_test.cpp` (`AggregateEndpointCountsAllLiveSessions`, `SessionStartReturnsConflictForActiveSessionKey`, `SessionStartReturns503WhenRegistryIsFull`).
- ✅ **Phase 9** — `config.conf` and `config-remote.conf` updated with all four new keys and comments.
- ✅ **Phase 10** — `docs/development/TRAINING_METRICS_API.md` updated with the full session-scoped route catalog, `GET /api/sessions` and `GET /api/metrics/aggregate` endpoint documentation, backwards-compatibility alias table, and session key configuration reference.

Files Modified:

- `src/MetricsSessionRegistry.hpp` (new)
- `src/TrainingMetricsAPI.hpp` / `src/TrainingMetricsAPI.cpp`
- `src/TrainingMetricsAPIServer.cpp`
- `src/Config.hpp` / `src/Config.cpp`
- `src/IncrementalTrainer.cpp`
- `src/TrainingMetricsService.hpp` / `src/TrainingMetricsService.cpp` — `GlobalMetricsService`
  ended up as a class inside this file rather than its own `GlobalMetricsService.hpp` as
  originally planned; corrected here September 7, 2026 after finding a stale TODO in that class
  still describing this item as pending
- `tests/metrics_session_registry_test.cpp`
- `tests/training_metrics_api_routes_test.cpp`
- `tests/training_metrics_service_resume_test.cpp`
- `config.conf` / `config-remote.conf`
- `docs/development/TRAINING_METRICS_API.md`

Verification:

- ✅ All `metricsSessionRegistryTests` pass (7 tests including 3 new concurrent-session tests)
- ✅ All `trainingMetricsApiRoutesTests` pass (9 tests including 409/503/aggregate coverage)
- ✅ All `incrementaltrainerTests` pass (40 tests) — no regressions
- ✅ Production build (`build-gpu-clang`) and ASAN build (`build-asan`) both clean

---

### TD-003: GPU Memory Management Optimization

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| May 3, 2026 | GPU / Performance | `GPUMatrix` class with persistent device memory, `Matrix::to_gpu()` / `Matrix::from_gpu()` |

Description:
Implemented persistent GPU-resident matrix storage that eliminates per-operation host↔device transfers when chaining multiple GPU operations.  Previously every `multiply_gpu()` / `add_gpu()` etc. call allocated device memory, transferred data, executed the kernel, copied the result back, and freed device memory — incurring PCIe round-trip cost for every single op.  With `GPUMatrix`, data is uploaded once, all intermediate computations remain on-device, and only the final result is downloaded.

Changes Made:

- ✅ Added `#include "GPUUtils.hpp"` to `src/gpu/MatrixGPU.hpp` so the class can use `GPUMemory<float>`, `GPUManager`, and `CUDA_CHECK`.
- ✅ Created `adai::gpu::GPUMatrix` class in `src/gpu/MatrixGPU.hpp` (move-only, RAII via `GPUMemory<float>`):
  - `GPUMatrix(int rows, int cols)` — allocate on-device
  - `upload(const float*, int)` / `download(float*, int)` — blocking host↔device transfers
  - `copy()` — async device-to-device clone
  - `operator*` — cuBLAS SGEMM matrix multiply (stays on device)
  - `operator+` — element-wise add (stays on device)
  - `operator-` — element-wise subtract (stays on device)
  - `scale(float)` — scalar multiply (stays on device)
  - `hadamard(const GPUMatrix&)` — element-wise multiply (stays on device)
  - `transpose()` — shared-memory transpose kernel (stays on device)
  - `apply_activation_inplace(ActivationType)` — in-place activation (stays on device)
  - `sum()` — parallel reduction, returns scalar to host
- ✅ Added `Matrix::to_gpu() const` to `src/Matrix.hpp` / `src/Matrix.cpp` — uploads CPU matrix to a new `GPUMatrix`.
- ✅ Added `static Matrix Matrix::from_gpu(const adai::gpu::GPUMatrix&)` — downloads a `GPUMatrix` back to a CPU `Matrix`.

Files Modified:

- `src/gpu/MatrixGPU.hpp` — `#include "GPUUtils.hpp"` + `GPUMatrix` class
- `src/Matrix.hpp` — `to_gpu()` and `from_gpu()` declarations
- `src/Matrix.cpp` — `to_gpu()` and `from_gpu()` implementations

Usage Example:

```cpp
// Before (TD-003): 3 upload/download round-trips
Matrix C = A.multiply_gpu(B);   // upload A,B → kernel → download C
Matrix D = C.add_gpu(A);        // upload C,A → kernel → download D
Matrix E = D.transpose_gpu();   // upload D   → kernel → download E

// After (TD-003 resolved): 3 uploads, 1 download, 3 on-device ops
auto A_gpu = A.to_gpu();
auto B_gpu = B.to_gpu();
auto C_gpu = A_gpu * B_gpu;           // on-device matmul
auto D_gpu = C_gpu + A_gpu;           // on-device add
auto E_gpu = D_gpu.transpose();       // on-device transpose
Matrix E = Matrix::from_gpu(E_gpu);   // single download
```

Verification:

- ✅ `adai_core` builds clean with `-DENABLE_GPU=ON`
- ✅ CPU-only builds unaffected (entire class inside `#ifdef ADAI_ENABLE_GPU`)
- ✅ All existing `matrixTests` and `matrixSIMDTests` pass — no regressions

---

### TD-017: Adaptive Gradient Clipping

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| April 19, 2026 | Training / ChatbotTrainer / Config / Metrics / Dashboard | EMA-based adaptive clip threshold with spike suppression and warmup |

Description:
Replaced static `GRADIENT_CLIP` scalar (0.5) that was clipping every gradient step in Session 5 (per-step norms 2–23, threshold 4–46× too small). Implemented EMA-based adaptive threshold: `ema ← α×raw_norm + (1−α)×ema`, `effective_clip = clamp(ema × headroom, min, max)` with spike suppression (skip EMA update when `norm > spike_k × ema`) and warmup period.

Changes Made:

- ✅ Added 7 new `ServiceConfig` fields in `src/Config.hpp`: `adaptive_gradient_clip`, `gradient_clip_min`, `gradient_clip_max`, `gradient_clip_ema_decay`, `gradient_clip_headroom`, `gradient_clip_warmup_steps`, `gradient_clip_spike_k`
- ✅ Implemented full config parsing for 7 new keys in `src/Config.cpp` (file key-value reader + env-var override block)
- ✅ Mirrored 7 fields in `ChatbotTrainerConfig` (`src/ChatbotTrainer.hpp`)
- ✅ Mapped new fields in `IncrementalTrainer::make_incremental_config()` (`src/IncrementalTrainer.cpp`)
- ✅ Replaced fixed-clip call in `ChatbotTrainer::train_epoch()` with EMA + clamp adaptive logic; legacy path preserved when `adaptive_gradient_clip=false`
- ✅ Added `current_adaptive_clip_threshold`, `current_adaptive_clip_spikes`, and `epoch_adaptive_clip_thresholds` to `TrainingMetricsSnapshot`; added `update_adaptive_clip_metrics()` and `update_adaptive_clip_epoch()` to `TrainingMetricsService`
- ✅ Extended `to_json()` in `TrainingMetricsService` to emit both new snapshot fields
- ✅ Added `epoch_adaptive_clip_thresholds` array to `TrainingMetricsAPI::handle_epoch_metrics()`
- ✅ Added **Gradient Clipping** panel to `dashboard.html`: dual-line chart (adaptive threshold + raw gradient norm), two metric cards (threshold + spike count)
- ✅ Written unit tests in `tests/adaptive_clipping_test.cpp` (defaults, update, epoch accumulation, JSON emission)

---

### TD-013: Advanced Training Metrics and Outlier Detection

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 14, 2026 | Training / Service | Welford outlier detection, advanced epoch metrics, and activation/entropy/BLEU/padding sub-items |

Description:
Extended the training pipeline with comprehensive observability beyond basic loss and learning rate: throughput ratios, model optimization health signals, generative validation scoring (BLEU/ROUGE), and automated flagging of training samples with abnormally large loss or gradient variations.

Changes Made:

- ✅ Updated `TrainingMetricsSnapshot` to support additional floats (`gradient_variance`, `compute_time_ratio`, `weight_update_ratio`).
- ✅ Added `AbnormalSample` struct with `epoch`, `sample_id`, `input_text`, `target_text`, `loss`, `grad_norm`, `reason`.
- ✅ Added outlier config to `MetricsServiceConfig`: `loss_outlier_z_threshold`, `grad_norm_outlier_threshold`, `max_abnormal_samples`, `abnormal_samples_file`.
- ✅ Implemented `flag_abnormal_sample()`, `get_abnormal_samples()`, `update_advanced_epoch_metrics()` on `TrainingMetricsService`.
- ✅ Implemented `persist_abnormal_samples()` — writes flagged samples to `abnormal_samples.json` on every flag event.
- ✅ Added `Optimizer::get_weight_norm()` to compute the L2 norm of all weight parameters.
- ✅ Instrumented `ChatbotTrainer::train_epoch()` with:
  - Welford online algorithm for per-step gradient norm variance.
  - Per-step loss z-score outlier guard (triggers after ≥10 samples).
  - Gradient-norm absolute outlier guard (`grad_norm_outlier_threshold`).
  - Compute-time ratio tracking (forward+backward nanoseconds / epoch wall nanoseconds).
  - Weight-update ratio computation: `(lr × ||g||₂) /||w||₂` averaged per epoch.
- ✅ Added REST endpoint `GET /api/metrics/abnormal` to `TrainingMetricsAPI`.
- ✅ Activation saturation tracking (implemented April 11, 2026 — see TD-013b).
- ✅ Attention entropy (implemented April 11, 2026 — see TD-013b).
- ✅ BLEU/ROUGE generation quality scores (implemented April 11, 2026 — see TD-016).
- ✅ Batch padding efficiency (implemented April 11, 2026 — see TD-013b).

---

### TD-013b: Batch Padding Efficiency Tracking

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| April 11, 2026 | Training / Service / Metrics | Per-window efficiency computation in `train_epoch()` + `TrainingMetricsService` integration |

Summary:
Implemented batch padding efficiency — the theoretical fraction of non-padding tokens per gradient-accumulation window — as the final deferred item from TD-013. Each gradient-accumulation window's input/target sequence lengths are collected and compared against their padded extent `(max_input_len + max_target_len) × window_size`. The epoch average is reported to `TrainingMetricsService`, exposed via REST, and displayed on the dashboard. When `gradient_accumulation_steps == 1` efficiency is trivially 1.0; with larger accumulation windows the metric reveals sequence-length mismatch within virtual batches.

Changes Made:

- ✅ Added `current_padding_efficiency` (float, default -1 = not computed) and `epoch_padding_efficiencies` (vector) to `TrainingMetricsSnapshot`.
- ✅ Added `update_padding_efficiency(float)` to `TrainingMetricsService` — mutex-safe snapshot update.
- ✅ `TrainingMetricsService::end_epoch()` now pushes `current_padding_efficiency` into `epoch_padding_efficiencies`.
- ✅ `TrainingMetricsService::to_json()` now emits `"current_padding_efficiency"`.
- ✅ Instrumented `ChatbotTrainer::train_epoch()`:
  - Per-window accumulators: `pad_win_actual`, `pad_win_max_input`, `pad_win_max_target`, `pad_win_count` — reset at each accumulation cycle start and collected per sample.
  - At every `should_update` checkpoint: `eff = pad_win_actual / ((max_in + max_tgt) * window_size)`, accumulated into `pad_eff_sum / pad_eff_count`.
  - At epoch end: `avg_padding_efficiency` reported via `metrics_service_->update_padding_efficiency()`.
- ✅ Added REST endpoint `GET /api/metrics/padding-efficiency` to `TrainingMetricsAPI` — returns `current_padding_efficiency` and `epoch_padding_efficiencies` as JSON.
- ✅ `dashboard.html`: added "Padding Eff." metric card; `metricsHistory.paddingEfficiencies` array; `addToHistory()` and `updateDashboard()` wired up (null entries when score is -1 for clean gap rendering).
- ✅ Created `tests/padding_efficiency_test.cpp` — 16 tests across 2 suites (service-level and pure arithmetic). All 16 pass.
- ✅ Added `paddingEfficiencyTests` target to `tests/CMakeLists.txt`.

Files Modified:

- `src/TrainingMetricsService.hpp` — snapshot fields, method declaration
- `src/TrainingMetricsService.cpp` — `update_padding_efficiency()`, `end_epoch()` history push, `to_json()` output
- `src/TrainingMetricsAPI.hpp` — endpoint comment, `handle_padding_efficiency_metrics()` declaration
- `src/TrainingMetricsAPI.cpp` — route + handler implementation
- `src/ChatbotTrainer.cpp` — per-window accumulators + epoch-end push
- `dashboard.html` — "Padding Eff." metric card and JS data flow
- `tests/CMakeLists.txt` — `paddingEfficiencyTests` target

Files Created:

- `tests/padding_efficiency_test.cpp` — 16 new tests

Verification:

- ✅ All 16 `paddingEfficiencyTests` pass
- ✅ `adai_core` builds clean
- ✅ `chatbottrainerTests` builds clean — no regressions

---

### TD-016: BLEU/ROUGE Generation Quality Scoring

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| April 11, 2026 | Training / Service / Metrics | Header-only `GenerationQualityEvaluator` with corpus BLEU-1/2/4 and macro-averaged ROUGE-1/2/L F1 |

Summary:
Implemented end-to-end BLEU and ROUGE generation quality scoring with no external library dependencies. Scoring is opt-in (`enable_generation_quality_metrics = false` by default) and runs on a configurable sub-sample of the validation set during each `validate()` call, calling `model->generate_response()` in eval mode.

Changes Made:

- ✅ Created `src/GenerationQualityMetrics.hpp` — header-only `GenerationQualityEvaluator` providing:
  - `tokenize()`: whitespace split + lowercase + strip leading/trailing punctuation.
  - Corpus BLEU-1, BLEU-2, BLEU-4 with clipped modified precision (Lin & Och 2004 add-1 smoothing) and corpus-level brevity penalty.
  - Macro-averaged ROUGE-1 and ROUGE-2 F1 (precision × recall / (precision + recall) per sentence).
  - Macro-averaged ROUGE-L F1 via rolling 2-row DP LCS (O(m·n) time, O(n) space).
  - `GenerationQualityScore` struct: `bleu1`, `bleu2`, `bleu4`, `rouge1`, `rouge2`, `rougeL` (all default -1.0 = not computed).
- ✅ Extended `TrainingMetricsSnapshot` with `current_bleu4`, `current_rouge1`, `current_rouge2`, `current_rougeL` and per-epoch history vectors `epoch_bleu4`, `epoch_rouge1`, `epoch_rouge2`, `epoch_rougeL`.
- ✅ Added `MetricsServiceConfig::enable_generation_quality` (default `false`) and `generation_quality_sample_size` (default `10`).
- ✅ Implemented `TrainingMetricsService::update_generation_quality_metrics(bleu4, rouge1, rouge2, rougeL)` — mutex-safe update + optional HTTP push to `/api/metrics/generation-quality`.
- ✅ `TrainingMetricsService::end_epoch()` now pushes all four scores into per-epoch history vectors.
- ✅ `TrainingMetricsService::to_json()` now emits `current_bleu4`, `current_rouge1`, `current_rouge2`, `current_rougeL`.
- ✅ Added `TrainingConfig` fields: `enable_generation_quality_metrics` (default `false`), `generation_quality_sample_size` (default `10`), `generation_quality_max_tokens` (default `50`).
- ✅ `ChatbotTrainer::validate()` now calls `compute_generation_quality_metrics()` after updating validation-loss metrics.
- ✅ Implemented `ChatbotTrainer::compute_generation_quality_metrics()` — samples up to `generation_quality_sample_size` pairs from the front of the validation set, calls `model->generate_response()` in eval mode for each, then delegates scoring to `GenerationQualityEvaluator::evaluate()` and pushes results to `metrics_service_`.
- ✅ Added REST endpoint `GET /api/metrics/generation-quality` to `TrainingMetricsAPI` — returns current and per-epoch BLEU/ROUGE history as JSON.
- ✅ `dashboard.html`: added "BLEU-4" and "ROUGE-L" metric cards; `metricsHistory` extended with `bleu4` / `rougeL` arrays; `addToHistory()` and `updateDashboard()` wired up (null entries used when score is -1 so chart gaps are rendered cleanly).
- ✅ Created `tests/generation_quality_test.cpp` — 23 tests across 5 suites (Tokenizer, BLEU, ROUGE, ROUGE-L, Edge). All 23 pass.
- ✅ Added `generationQualityTests` target to `tests/CMakeLists.txt`.

Files Created:

- `src/GenerationQualityMetrics.hpp` — New header-only BLEU/ROUGE library
- `tests/generation_quality_test.cpp` — 23 new tests

Files Modified:

- `src/TrainingMetricsService.hpp` — snapshot fields, config options, method declaration
- `src/TrainingMetricsService.cpp` — `update_generation_quality_metrics()`, `end_epoch()` history push, `to_json()` output
- `src/TrainingMetricsAPI.hpp` — `handle_generation_quality_metrics()` declaration + endpoint comment
- `src/TrainingMetricsAPI.cpp` — route + handler implementation
- `src/ChatbotTrainer.hpp` — `TrainingConfig` fields, `compute_generation_quality_metrics()` method
- `src/ChatbotTrainer.cpp` — `#include "GenerationQualityMetrics.hpp"`, `compute_generation_quality_metrics()` implementation, `validate()` call-site
- `dashboard.html` — BLEU-4 / ROUGE-L metric cards and JS data flow
- `tests/CMakeLists.txt` — `generationQualityTests` target

Verification:

- ✅ All 23 `generationQualityTests` pass
- ✅ `adai_core` (TrainingMetricsService changes) builds clean
- ✅ `chatbottrainerTests` (ChatbotTrainer changes) builds clean

---

### TD-007: Matrix Operations SIMD Acceleration

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| April 11, 2026 | Performance / Matrix Operations | AVX2/FMA + NEON intrinsics and BLAS SGEMM integration across all Matrix operations |

Description:
Added explicit SIMD intrinsics (AVX2/FMA and ARM NEON) and optional BLAS (SGEMM) integration to all performance-critical `Matrix` operations, delivering 2-5x speedup over the previous scalar/OpenMP-only paths for compute-intensive workloads.

Changes Made:

- ✅ Created `src/MatrixSIMD.hpp` — compile-time SIMD capability macros (`ADAI_SIMD_AVX2`, `ADAI_SIMD_FMA`, `ADAI_SIMD_NEON`), runtime `has_avx2()` / `has_fma()` CPUID detection, and `hsum256()` / `hsum128()` horizontal-reduction helpers.
- ✅ Added `ENABLE_BLAS=ON` and `ENABLE_SIMD=ON` CMake options to top-level `CMakeLists.txt`.
- ✅ Updated `src/CMakeLists.txt`:
  - BLAS detection block: `find_package(BLAS)` + `find_path(CBLAS_INCLUDE_DIR cblas.h)` → sets `ADAI_ENABLE_BLAS` and links `${BLAS_LIBRARIES}` when both are present.
  - SIMD flags block: `check_cxx_compiler_flag(-mavx2/-mfma)` → adds target-specific compile options to `adai_core` for non-Release builds (Release already gets them from `-march=native`).
- ✅ `Matrix::operator*` — added BLAS SGEMM path (pack→`cblas_sgemm`→unpack) for matrices ≥ 256 in all dimensions; added AVX2/FMA `ikj`-order inner loop (processes 8 floats per FMA instruction) and NEON `vfmaq_f32` path; OpenMP and scalar paths preserved as fallbacks.
- ✅ `Matrix::operator+` — AVX2 `_mm256_add_ps` / NEON `vaddq_f32` row-wise loop replacing the collapse-2 OpenMP loop; scalar remainder handles non-multiple-of-8 tails.
- ✅ `Matrix::operator-` — AVX2 `_mm256_sub_ps` / NEON `vsubq_f32` row-wise loop.
- ✅ `Matrix::scale()` — AVX2 `_mm256_mul_ps` / NEON `vmulq_f32` row-wise loop.
- ✅ `Matrix::hadamard()` — AVX2 `_mm256_mul_ps` / NEON `vmulq_f32` element-wise loop.
- ✅ `Matrix::apply_gradients()` — AVX2+FMA `_mm256_fmadd_ps(−lr, g, w)` / NEON `vfmsq_n_f32` fused multiply-subtract; falls back to separate mul+sub on AVX2 without FMA.
- ✅ `Matrix::sum()` — AVX2 horizontal-reduction via `hsum256(_mm256)` accumulator / NEON `vaddvq_f32`; correct for any number of columns.
- ✅ All SIMD paths include an `#ifdef ADAI_ENABLE_OPENMP` outer `#pragma omp parallel for` so thread-level and data-level parallelism compose.
- ✅ Removed all TD-007 TODO comments from `src/Matrix.cpp`, `src/CMakeLists.txt`, and `CMakeLists.txt`.
- ✅ Created `tests/matrix_simd_test.cpp` — 91 tests across 11 test suites: parameterised column widths (1–256 including all non-multiples-of-8), shapes, BLAS large-matrix path, CPU feature detection, edge cases, and numerical stability. All 91 tests pass.
- ✅ Added `matrixSIMDTests` target to `tests/CMakeLists.txt`.

Files Modified:

- `src/MatrixSIMD.hpp` — New file
- `src/Matrix.hpp` — `#include "MatrixSIMD.hpp"`
- `src/Matrix.cpp` — BLAS + AVX2/FMA + NEON code paths for all six operations
- `src/CMakeLists.txt` — BLAS detection + SIMD compiler flags; removed TD-007 TODOs
- `CMakeLists.txt` — `ENABLE_BLAS` and `ENABLE_SIMD` options; removed TD-007 TODOs
- `tests/matrix_simd_test.cpp` — New 91-test SIMD test suite
- `tests/CMakeLists.txt` — `matrixSIMDTests` target

Verification:

- ✅ `cmake .. -DENABLE_SIMD=ON -DENABLE_BLAS=ON` reports "SIMD: AVX2 + FMA intrinsics enabled for adai_core"
- ✅ All 91 `matrixSIMDTests` pass (AVX2+FMA active at runtime)
- ✅ All 58 existing `matrixTests` pass — no regressions

---

### TD-015: Validation Metrics Integration

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 14, 2026 | Training / Service | Extended `TrainingMetricsService` with validation perplexity and accuracy tracking |

Description:
Previously, the training pipeline tracked metrics during training passes but lacked granular, systematized metric tracking for the validation phase. Validation metrics are now fully integrated into `TrainingMetricsService` and the dashboard, providing better insights into model generalization and enabling early detection of over-fitting. Proposal: `docs/development/archive/VALIDATION_METRICS_PROPOSAL.md`.

Changes Made:

- ✅ Extended `TrainingMetricsSnapshot` with `current_validation_perplexity`, `current_validation_accuracy`, `epoch_validation_perplexities`, and `epoch_validation_accuracies`.
- ✅ Updated `update_validation_metrics()` signature to accept `validation_loss`, `validation_accuracy` (default `-1.0`), and `validation_perplexity` (default `0` = auto-derived from loss).
- ✅ `end_epoch()` now copies the current validation perplexity and accuracy into the per-epoch history vectors.
- ✅ `to_json()` now emits `current_validation_perplexity` and `current_validation_accuracy` in the API response.
- ✅ `ChatbotTrainer::validate()` now calls `update_validation_metrics(val_loss, -1.0f, val_perplexity)` — passes computed perplexity.
- ✅ `TrainingMetricsAPI::handle_post_validation_metrics()` parses `validation_accuracy` and `validation_perplexity` from the POST body.
- ✅ `handle_epoch_metrics()` returns `epoch_validation_perplexities` and `epoch_validation_accuracies` arrays.
- ✅ `dashboard.html`: added "Val Perplexity" and "Val Accuracy" metric cards; perplexity chart now shows both training and validation perplexity curves with distinct colors; `metricsHistory` and `addToHistory()` extended to track `validationPerplexities`.

---

### TD-012: Increase Test Coverage

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 1, 2026 | Code Quality / Testing | Comprehensive test suite implementation for all components |

Summary:
Achieved ~100% test coverage of all testable code components (excluding GUI and legacy entry points). Implemented dedicated test suites for all major subsystems including Config, Logger, Trainer, and Inference Engines.

Changes Made:

1. ✅ Created test suites for 13+ major components:
   - Config, Logger, IncrementalTrainer, LLMEncoder, DocumentStore
   - RAGInference, BatchProcessor, ParallelDataLoader
   - IntegratedInferenceEngine, PipelineInferenceEngine
   - BatchedInferenceEngine, SpeculativeDecoding, VocabBuilder
2. ✅ Achieved >95% LoC coverage target
3. ✅ Added edge case testing and error handling verification

Files Modified:

- `tests/*` (New test files for all components)
- `tests/CMakeLists.txt`

Coverage Stats:

- Total Testable Lines: ~25,000 LOC
- Tested Lines: ~24,784 LOC
- Coverage: ~99%

---

### TD-011: File Rotation and Management

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 1, 2026 | Logging and Observability | Integrated spdlog rotating file sink |

Summary:
Implemented production-grade log rotation to prevent infinite log growth. Logs are now automatically rotated based on size and count configuration.

Changes Made:

1. ✅ Added rotating file sink to spdlog (rotating_file_sink_mt)
2. ✅ Configured max file size and rotation count
3. ✅ Support compression flag (requires external tool)
4. ✅ Added `LOG_FILE_PATH`, `LOG_MAX_SIZE_MB`, `LOG_MAX_FILES`, `LOG_COMPRESS` configuration options
5. ✅ Dual-sink pattern (console + rotating file)
6. ✅ Thread-safe multi-threaded sink
7. ✅ Validation for log rotation parameters
8. ✅ Integration with configuration hot-reload

Files Modified:

- `src/Logger.cpp`
- `src/Logger.hpp`
- `src/Config.cpp`
- `src/Config.hpp`
- `src/ChatbotAPIServer.cpp`

**Documentation:** `LOG_FILE_ROTATION_COMPLETE.md`

---

### TD-010: Configuration Hot-Reloading

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 1, 2026 | Configuration and Service Management | Implemented SIGHUP handler and thread-safe config updates |

Summary:
Implemented zero-downtime configuration reloading using SIGHUP signals. This allows changing log levels, timeouts, and other parameters without restarting the service.

Changes Made:

1. ✅ Implemented SIGHUP handler to trigger config reload
2. ✅ Added thread-safe configuration updates with mutex protection
3. ✅ Validate new configuration before applying
4. ✅ Log configuration changes with timestamps

Files Modified:

- `src/Config.cpp`
- `src/ChatbotAPIServer.cpp`
- `src/Logger.cpp`

**Documentation:** `CONFIG_HOT_RELOAD_COMPLETE.md`

---

### TD-009: Incremental Trainer Dashboard and Structured Logging

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 2, 2026 | Training / IncrementalTrainer / Observability | Full implementation of Logger integration, per-epoch metric collection, in-place CLI dashboard, v2 session history serialization, and test coverage |

Summary:
Replaced all unstructured `std::cout`/`std::cerr` output in `IncrementalTrainer` with structured `Logger` calls, implemented a real-time in-place CLI dashboard that redraws after every epoch using ANSI cursor movement, and added complete per-epoch metric collection and persistence. Absorbed the scope of TD-004.

Changes Made:

1. ✅ Added `EpochCallback` typedef and `set_epoch_callback()` to `ChatbotTrainer` — fires once per epoch inside `train(int)` without disrupting LR scheduling or data preprocessing
2. ✅ Extended `TrainingSession` struct with four per-epoch vectors: `per_epoch_losses`, `per_epoch_validation_losses`, `per_epoch_learning_rates`, `training_time_per_epoch`
3. ✅ Added `session_start_time_steady_`, `epoch_start_time_steady_`, and `mutable dashboard_lines_drawn_` members to `IncrementalTrainer`
4. ✅ Replaced all `std::cout`/`std::cerr` calls (20+ sites) with `Logger::info`, `Logger::warn`, `Logger::error`, `Logger::debug` across all methods
5. ✅ Overhauled `train_incremental()` to register the epoch callback, collect per-epoch metrics, and invoke `display_dashboard()` after each epoch
6. ✅ Implemented `display_dashboard()` — 10-line in-place redrawing box (ANSI `\033[NA` cursor-up, UTF-8 box-drawing chars) showing:
   - Epoch progress bar with percentage
   - Elapsed time and ETA (`avg_epoch_time × remaining_epochs`)
   - Current loss / val loss with delta arrows (`v` / `^`)
   - Current LR and epoch duration
   - Session best val loss and average epoch time
7. ✅ Implemented `format_duration()` and `progress_bar()` helpers
8. ✅ Rewrote `save_session_history()` with `# VERSION 2` header and `|losses:...|vallosses:...|lrs:...|times:...` pipe-delimited appendage after `checkpoint_path`
9. ✅ Rewrote `load_session_history()` to parse v2 extended format with full backward compatibility for old single-line format
10. ✅ Overhauled `print_training_summary()` with Unicode sparkline bars (▁▂▃▄▅▆▇█) per session, avg epoch time, best val loss, and total training time
11. ✅ Added `spdlog::spdlog` and `Logger.cpp` to `incremental_trainer` and `incrementaltrainerTests` CMake targets
12. ✅ Added 3 new GTest cases (38 total): `LoadSessionHistoryV2ParsesPerEpochVectors`, `SaveLoadSessionHistoryRoundTripWithPerEpochData`, `DisplayDashboardDoesNotCrash` — all pass

Files Modified:

- `src/ChatbotTrainer.hpp` — `EpochCallback` typedef, `epoch_callback_` member, `set_epoch_callback()` declaration
- `src/ChatbotTrainer.cpp` — `set_epoch_callback()` implementation; callback invocation in epoch loop
- `src/IncrementalTrainer.hpp` — Logger include; extended `TrainingSession`; timing/dashboard members; method declarations
- `src/IncrementalTrainer.cpp` — complete overhaul (Logger calls, timing, dashboard, v2 serialization, sparkline summary)
- `src/CMakeLists.txt` — added `Logger.cpp` and `spdlog::spdlog` to `incremental_trainer` target
- `tests/incrementaltrainer_test.cpp` — 3 new TD-009 tests
- `tests/CMakeLists.txt` — added `../src/Logger.cpp` and `spdlog::spdlog` to `incrementaltrainerTests` target

Verification:

- ✅ Full build succeeds (both `incremental_trainer` executable and `incrementaltrainerTests`)
- ✅ All 38 incremental trainer tests pass including the 3 new TD-009 tests
- ✅ v2 session history round-trip verified: write → parse → verify all four per-epoch vectors
- ✅ Dashboard smoke-tested with empty and populated session history

---

### TD-004: Enhanced Metrics Tracking for Training Sessions

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 2, 2026 | Training / IncrementalTrainer | Absorbed and fully implemented as part of TD-009 |

Summary:
All core implementation tasks from TD-004 were completed as part of the TD-009 implementation. The `TrainingSession` struct now carries four per-epoch vectors; `ChatbotTrainer` exposes per-epoch data via `EpochCallback`; session history serialization (`save_session_history` / `load_session_history`) uses a v2 format that persists and reloads these vectors; and `print_training_summary` renders Unicode sparklines for visual trend analysis. The 55 granular TODO comments that were seeded across the codebase have all been removed.

Tasks Completed:

- ✅ Extended `TrainingSession` struct with `per_epoch_losses`, `per_epoch_validation_losses`, `per_epoch_learning_rates`, `training_time_per_epoch`
- ✅ Modified `ChatbotTrainer` to expose per-epoch data via `EpochCallback` mechanism
- ✅ Updated `save_session_history()` and `load_session_history()` with v2 format (backward-compatible)
- ✅ Updated `print_training_summary()` with sparkline visualization of loss/val-loss trends per session
- ✅ Tests added: `LoadSessionHistoryV2ParsesPerEpochVectors`, `SaveLoadSessionHistoryRoundTripWithPerEpochData`
- ✅ All 55 TD-004 TODO comments removed from source files

**See:** [TD-009](#td-009-incremental-trainer-dashboard-and-structured-logging) for full implementation details.

---

### TD-008: Daemon Service Implementation (Steps 1-5)

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| March 1, 2026 | Service Management / Production Deployment | Complete 5-step daemon service transformation |

Summary:
Successfully transformed the ADAI Chatbot API Server from a development application into a production-ready daemon service. Implemented external configuration, graceful shutdown, structured logging, Docker deployment, and systemd integration following a comprehensive 5-step plan.

Changes Made:

#### Step 1: External Configuration ✅

1. Created `Config.hpp` and `Config.cpp` for centralized configuration management
2. Implemented multi-source configuration with priority system:
   - Command-line arguments (highest priority)
   - Environment variables
   - Configuration file
   - Default values (lowest priority)
3. Added support for all parameters: server config, model architecture, text generation
4. Created `config.conf` example configuration file
5. Updated `ChatbotAPIServer.cpp` to use Config system
6. Modified Docker files to support environment variables
7. Documentation: `STEP1_COMPLETE.md`

#### Step 2: Signal Handling ✅

1. Implemented async-signal-safe signal handlers for SIGTERM and SIGINT
2. Created graceful shutdown sequence:
   - Stop HTTP server
   - Complete in-flight requests
   - Save model state (if configured)
   - Clean up resources
   - Exit cleanly
3. Used `std::atomic<bool>` for thread-safe shutdown flag
4. Fixed model parameter order bug (vocab_size, d_model, num_heads)
5. Created test scripts: `test_signal_handling.sh`, `test_sigint.sh`
6. Verified 30-second timeout and clean shutdown
7. Documentation: `STEP2_COMPLETE.md`

#### Step 3: Structured Logging ✅

1. Integrated spdlog v1.12.0 via CMake FetchContent
2. Created `Logger.hpp` and `Logger.cpp` wrapper classes
3. Replaced all `std::cout`/`std::cerr` with structured logging:
   - `Logger::debug()` - detailed debugging info
   - `Logger::info()` - normal operations
   - `Logger::warn()` - warnings
   - `Logger::error()` - errors
4. Implemented timestamped log format: `[YYYY-MM-DD HH:MM:SS.mmm] [level] message`
5. Added configurable log levels (DEBUG, INFO, WARN, ERROR)
6. Configured auto-flush for container environments
7. Documentation: `STEP3_COMPLETE.md`

#### Step 4: Docker Configuration ✅

1. Enhanced `Dockerfile` with comprehensive environment variable documentation
2. Documented all 19 configuration options with inline comments
3. Organized into sections: Server, Model Architecture, Text Generation
4. Updated `docker-compose.yml` with detailed configuration examples
5. Created comprehensive deployment guide: `DOCKER_DEPLOYMENT.md` (500+ lines)
6. Documented configuration priority, volume mounts, health checks
7. Added troubleshooting, security, and monitoring sections
8. Multi-stage build already implemented for minimal runtime image
9. Documentation: `STEP4_COMPLETE.md`

#### Step 5: systemd Service File ✅

1. Created production-ready systemd service file: `scripts/adai.service`
2. Implemented extensive security hardening:
   - Filesystem protection (ProtectSystem=strict)
   - Kernel protection (ProtectKernelLogs, ProtectKernelModules, etc.)
   - Privilege restrictions (NoNewPrivileges, CapabilityBoundingSet=)
   - System call filtering (SystemCallFilter)
   - Network isolation (RestrictAddressFamilies=AF_INET AF_INET6)
3. Configured resource limits:
   - Memory: 4GB max, 3GB soft limit
   - CPU: 50% quota
   - Files: 65536 max, 2GB file size limit
4. Implemented automatic restart: on-failure with rate limiting (5 restarts per 10 minutes)
5. Created automated installation script: `scripts/install_systemd_service.sh`
   - One-command installation
   - Configurable paths, user, group, port
   - 8-step installation with verification
   - Preflight checks and interactive confirmation
6. Created comprehensive deployment guide: `SYSTEMD_DEPLOYMENT.md` (900+ lines)
7. Documentation: `STEP5_COMPLETE.md`, `DAEMON_IMPLEMENTATION_COMPLETE.md`

Files Created:

- Configuration: `src/Config.hpp`, `src/Config.cpp`, `config.conf`, `config.conf.example`
- Logging: `src/Logger.hpp`, `src/Logger.cpp`
- systemd: `scripts/adai.service`, `scripts/install_systemd_service.sh` *(the latter was later
  consolidated into `scripts/install_chatbot_API.sh` — commit `d914070`, "chore: consolidate
  systemd deployment onto install_chatbot_API.sh"; corrected here September 7, 2026)*
- Documentation (now under `docs/development/archive/` after a later doc reorg): `STEP1_COMPLETE.md`,
  `STEP2_COMPLETE.md`, `STEP3_COMPLETE.md`, `STEP4_COMPLETE.md`, `STEP5_COMPLETE.md`,
  `DAEMON_IMPLEMENTATION_COMPLETE.md`; the Docker/systemd deployment guides are now
  `docs/operations/deployment/docker.md` and `docs/operations/deployment/SYSTEMD_DEPLOYMENT.md`
- Testing: `scripts/test_signal_handling.sh`, `scripts/test_sigint.sh` *(the latter was later found
  to be a near-duplicate of the former and tagged `experimental` — see TD-046)*

Files Modified:

- `src/ChatbotAPIServer.cpp` - Added Config, Logger, signal handling
- `CMakeLists.txt` - Added spdlog dependency, Logger.cpp
- `src/CMakeLists.txt` - Linked spdlog library
- `Dockerfile` - Comprehensive environment variable documentation
- `docker-compose.yml` - Enhanced configuration with detailed comments

Verification:

- ✅ Configuration: All sources (CLI, env, file) work with correct priority
- ✅ Signal handling: SIGTERM/SIGINT trigger graceful shutdown
- ✅ Logging: Structured logs with timestamps at all levels
- ✅ Docker: Image builds, container starts, health checks pass
- ✅ systemd: Service file syntax valid, installation script functional

Impact:

- **Production Ready:** Service can now run as managed daemon
- **Observable:** Structured logs, configurable verbosity, timestamps
- **Reliable:** Graceful shutdown, automatic restart, resource limits
- **Secure:** Non-root user, extensive hardening, minimal privileges
- **Portable:** Docker and systemd deployment options
- **Documented:** Comprehensive deployment guides (1400+ lines total)

Related Future Enhancements:
See [Future Improvements](#configuration-and-service-management) section for enhancements building on this foundation:

- Configuration hot-reloading (Config Enhancement #1)
- JSON configuration format (Config Enhancement #2)
- Model state persistence (Config Enhancement #4)
- JSON log output (Logging Enhancement #1)
- Metrics endpoint (Container Enhancement #2)
- Socket activation (Container Enhancement #3)

---

### TD-005: Checkpoint Management and Symbolic Links

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| February 18, 2026 | Training / IncrementalTrainer | Complete symlink management implementation with cross-platform support |

Summary:
Implemented automatic checkpoint symlink management to provide easy access to "latest" and "best" model checkpoints. The system creates and maintains symbolic links (or file copies on Windows) that always point to the most recent checkpoint and the checkpoint with the lowest validation loss.

Changes Made:

1. ✅ Added configuration options for symlink management:
   - `enable_checkpoint_symlinks` - toggle feature on/off (default: true)
   - `latest_symlink_name` - configurable name for latest checkpoint link
   - `best_symlink_name` - configurable name for best checkpoint link

2. ✅ Implemented cross-platform symlink support:
   - Unix/Linux: Uses `std::filesystem::create_symlink()` for true symbolic links
   - Windows: Falls back to file copying for compatibility
   - Platform detection with `is_windows_platform()` helper

3. ✅ Created checkpoint tracking infrastructure:
   - Added `best_validation_loss` and `best_checkpoint_path` private members
   - Automatically initializes best checkpoint from session history on startup
   - Tracks and compares validation loss across all sessions

4. ✅ Implemented symlink helper methods:
   - `update_checkpoint_symlinks()` - Updates "latest" link after each session
   - `update_best_checkpoint()` - Updates "best" link when validation loss improves
   - `create_or_update_symlink()` - Creates/updates symlinks with error handling
   - `remove_symlink_if_exists()` - Safely removes existing links
   - `get_best_checkpoint_path()` - Retrieves path to best checkpoint

5. ✅ Integrated with training workflow:
   - `finalize_session()` now creates/updates symlinks after each session
   - `cleanup_old_sessions()` handles symlink cleanup when deleting checkpoints
   - Recalculates best checkpoint if the best model is deleted during cleanup

6. ✅ Enhanced training summary output:
   - `print_training_summary()` displays symlink paths and targets
   - Shows best checkpoint validation loss for quick reference
   - Handles both symlinks and file copies transparently

Files Modified:

- `include/IncrementalTrainer.hpp` - Added configuration and method declarations
- `src/IncrementalTrainer.hpp` - Added configuration and method declarations (duplicate header)
- `src/IncrementalTrainer.cpp` - Implemented all symlink management logic
  - Added `#include <limits>` for `std::numeric_limits`
  - Updated constructor to initialize best checkpoint tracking
  - Implemented 6 new symlink helper methods (~110 lines)
  - Updated `finalize_session()` to create symlinks
  - Updated `cleanup_old_sessions()` to handle symlink cleanup
  - Updated `print_training_summary()` to display symlink information

Usage Example:

```cpp
IncrementalTrainer trainer("vocab.txt", "model.bin");
trainer.config.enable_checkpoint_symlinks = true;  // Enabled by default
trainer.add_new_data("new_data.txt");
trainer.train_incremental(5);

// Symlinks are automatically created:
// - latest_checkpoint.bin -> training_sessions/session_N_checkpoint.bin
// - best_checkpoint.bin -> training_sessions/session_M_checkpoint.bin (lowest val loss)
```

Benefits Realized:

- Deployment scripts can reference `latest_checkpoint.bin` without parsing history
- Easy access to best model for inference and evaluation
- Automatic cleanup when old checkpoints are removed
- Cross-platform compatibility with Windows fallback
- Configurable for different deployment scenarios

Testing:

- Compiled successfully on Linux (GCC)
- All existing tests pass
- No breaking changes to existing functionality

---

### TD-002: Improve Error Handling in BPE Tokenizer

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| January 28, 2026 | NLP / Tokenization | Comprehensive error handling implementation |

Summary:
Implemented robust error handling and validation for the BPE tokenizer, including custom exception types, UTF-8 validation, input validation, and vocabulary file format validation.

Changes Made:

1. ✅ Created custom exception types:
   - `TokenizerInputError` - for empty/invalid input
   - `TokenizerEncodingError` - for UTF-8 encoding issues
   - `VocabularyFileError` - for malformed vocabulary files
   - `TokenIDError` - for out-of-range token IDs

2. ✅ Implemented UTF-8 validation:
   - `is_valid_utf8()` helper method validates character sequences
   - Detects invalid start bytes, incomplete sequences, and malformed continuation bytes
   - Applied to `encode()` and `pre_tokenize()` methods

3. ✅ Added input validation:
   - `validate_input()` helper checks for empty strings and UTF-8 validity
   - `encode()` validates non-empty input with proper UTF-8
   - `decode()` validates non-empty token ID vector and checks for negative IDs

4. ✅ Enhanced vocabulary file validation:
   - Validates filename is not empty
   - Throws descriptive exceptions for file not found
   - Validates special tokens section format and integer parsing
   - Validates vocabulary entries with tab separators and non-negative IDs
   - Validates BPE merges format and non-empty tokens
   - Ensures loaded vocabulary contains required special tokens

5. ✅ Created comprehensive test suite:
   - 27 tests covering all error conditions
   - Tests for exception types, messages, and inheritance
   - Edge case coverage (empty input, invalid UTF-8, malformed files)
   - All tests passing

Files Modified:

- `src/BPETokenizer.hpp` - Added custom exception types and validation methods
- `src/BPETokenizer.cpp` - Implemented validation throughout encode/decode/load_vocab
- `tests/tokenizer_error_handling_test.cpp` - New comprehensive test suite (27 tests)
- `tests/CMakeLists.txt` - Added tokenizerErrorHandlingTests target

Verification:
All 27 tests pass, validating:

- Empty input detection
- UTF-8 validation for various invalid sequences
- Token ID range checking
- Vocabulary file format validation
- Exception type hierarchy
- Descriptive error messages

---

### TD-001: Complete Optimizer Parameter Exposure

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| January 28, 2026 | Optimizer Integration | Complete parameter registration implementation |

Summary:
Successfully completed parameter exposure for all model components. The optimizer now fully manages all model parameters through the centralized `optimizer->step()` mechanism.

Changes Made:

1. ✅ Verified `LLMEncoder::register_parameters_with_optimizer()` properly exposes all parameters (token embedding, encoder blocks, final norm)
2. ✅ Verified `LLMDecoder::register_parameters_with_optimizer()` properly exposes all parameters (token embedding, decoder blocks, final norm)
3. ✅ Verified `LanguageModelHead::set_optimizer()` properly exposes all parameters (W_output, bias)
4. ✅ Updated `ChatbotTrainer` to use `optimizer->step()` instead of `model->update_weights()`
5. ✅ Removed obsolete TODO comments
6. ✅ Tested training with optimizer integration - works correctly

Files Modified:

- `src/ChatbotTrainer.cpp` - Replaced `model->update_weights()` with `optimizer->step()`, removed outdated comments

Verification:
Training runs successfully with AdamW optimizer using centralized parameter management. All parameter groups are properly registered and updated through `optimizer->step()`.

---

---

