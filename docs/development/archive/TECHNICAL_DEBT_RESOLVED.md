# Technical Debt - Resolved Items

Resolved items extracted from [TECHNICAL_DEBT.md](../guides/TECHNICAL_DEBT.md).

## Resolved Items

### TD-138: test_chatbot_gui.sh Reported "SUCCESS" Even When Every Required Symbol Check Failed

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/test_chatbot_gui.sh` | Track symbol-check failures in `SYMBOL_CHECK_FAILED` and report/exit accordingly instead of an unconditional final "SUCCESS" |

Summary:
Found during this session's third-pass re-read of `scripts/`. The four `nm`-based component checks
(`ChatbotGUI`, `BPETokenizer`, `EncoderDecoderModel`, `ConversationContext` symbols in
`chatbot_gui_binary`) each printed `✅`/`❌` but never recorded the result anywhere — unlike the
existence/permissions/size/ELF checks earlier in the same script, which `exit 1` immediately on
failure. The script always fell through to an unconditional `echo "Build Verification: SUCCESS ✅"`
and `exit 0` regardless of how many symbol checks had actually failed. Reproduced with a copy of the
real script and a fake `nm` that matches nothing: all four checks printed `❌ ... not found`, and the
script still finished with `Build Verification: SUCCESS ✅` and exit code 0 — so a genuinely
broken/stripped/wrong `chatbot_gui_binary` (missing the very classes this test exists to confirm are
linked in) would be reported as a passing build.

Changes Made:
- `scripts/test_chatbot_gui.sh`: added a `SYMBOL_CHECK_FAILED` flag, set to `true` alongside each of
  the four `❌ ... not found` branches. The final section now checks it: prints
  `Build Verification: FAILED ❌` and `exit 1` if any symbol check failed, otherwise the original
  `SUCCESS` path and `exit 0` unchanged.

Verification:
- ✅ Copy of the real script run with a fake `nm` matching zero symbols: **before** the fix, printed
  four `❌` lines but still ended with "SUCCESS ✅", real exit code 0; **after**, ends with
  "FAILED ❌", real exit code 1 (verified via `$?` on the script's own exit, not a piped `tail`'s,
  after the first check mistakenly reported the wrong exit code from the pipeline).
- ✅ Same harness with a fake `nm` matching all four required symbols: still correctly reports
  "SUCCESS ✅" and exit code 0 — the fix doesn't introduce a false failure on a genuinely good build.
- ✅ `bash -n scripts/test_chatbot_gui.sh` — syntax valid.

### TD-137: analyze_code.sh Silently Died After the First File With Warnings, Never Printed a Summary

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/analyze_code.sh` | Replaced `((ISSUES_FOUND++))` with a plain arithmetic assignment; fixed the "files found" count to use `wc -w` instead of `wc -l` |

Summary:
Found during this session's third-pass re-read of `scripts/`. Two bugs, one severe:

1. **Silent early exit under `set -e` — the severe one.** This script has `set -e` at the top. Its
   per-file loop counted files with clang-tidy warnings via `((ISSUES_FOUND++))` — a post-increment
   whose exit status is the *pre*-increment value: `0` the very first time a warning-carrying file
   was found. `set -e` treats any nonzero-exit-status command outside a condition position as fatal,
   so the script terminated immediately at that line — before analyzing any subsequent file and
   before ever printing the "📊 Analysis Summary" section. Reproduced directly against a copy of the
   real (pre-fix) script with a fake `clang-tidy` flagging one of three files: the run stopped dead
   right after printing that one file's warning, exit code 1, files after it in the list (and the
   whole summary) never ran/printed. The exact same `((count++))`-while-starting-at-0 pattern exists
   in the sibling `format_code.sh`, but that script has no `set -e`, so it doesn't trigger this;
   verified by grepping both files for `set -e`.
2. **"Found N files to analyze" undercounted when files were passed as explicit arguments.** When
   invoked as `analyze_code.sh file1.cpp file2.cpp file3.cpp`, `FILES_TO_CHECK="$@"` is one
   space-separated string, and `FILE_COUNT=$(echo "$FILES_TO_CHECK" | wc -l)` counts *newlines*, so
   it always reported `1` file regardless of how many were actually passed (the default,
   `find`-populated path — one path per line — was unaffected). The analysis loop itself
   (`for file in $FILES_TO_CHECK`, unquoted word-splitting) still iterated every file correctly
   regardless, so this was cosmetic-only: a wrong count in the printed banner and final summary.

Changes Made:
- `scripts/analyze_code.sh`: replaced `((ISSUES_FOUND++))` with `ISSUES_FOUND=$((ISSUES_FOUND + 1))`
  — a plain assignment has no increment-dependent exit status, so it's safe under `set -e`.
- Changed `FILE_COUNT=$(echo "$FILES_TO_CHECK" | wc -l)` to `wc -w`, which counts correctly whether
  `FILES_TO_CHECK` is space-separated (explicit args) or newline-separated (the `find` default).

Verification:
- ✅ Standalone bash repro isolating just the `set -e` + `((count++))` pattern: **before**, the loop
  died silently (exit 1) the instant the counter first incremented from 0; **after** (plain
  assignment), the loop ran to completion and printed the final count.
- ✅ Ran a full copy of the real script (`scripts/analyze_code.sh`, unmodified) against a 3-file
  fixture with a fake `clang-tidy` that flags the middle file: **before** the fix, output stopped
  after `src/B.cpp`'s warning, exit code 1, `src/C.cpp` and the summary never appeared; **after**,
  all three files were analyzed, the summary printed "Files analyzed: 3 / Files with issues: 1",
  exit code 0.
- ✅ Same fixture confirmed the file-count fix: "Found 3 files to analyze" (was "Found 1") when
  invoked with three explicit file arguments; the default `find`-based invocation (no arguments)
  against a separate 2-file fixture still correctly reported "Found 2 files to analyze" — unaffected
  by the `wc -w` change, since it was already newline-separated single-path-per-line output.
- ✅ `bash -n scripts/analyze_code.sh` — syntax valid.

### TD-136: test_signal_handling.sh's Graceful-Shutdown Wait Was 2s Despite Printing "5 seconds"

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/test_signal_handling.sh` | Changed `sleep 2` to `sleep 5` to match the printed message and the sibling script's pattern |

Summary:
Found immediately after TD-135, continuing this session's third-pass re-read of `scripts/`. The
SIGTERM graceful-shutdown wait printed `"Waiting for graceful shutdown (5 seconds)..."` but the
actual `sleep` right below it was `sleep 2` — a 3-second shortfall. The sibling script,
`test_sigint.sh` (same structure, SIGINT instead of SIGTERM), prints `"...(3 seconds)..."` and
correctly sleeps `3`, confirming `test_signal_handling.sh`'s `2` was the outlier, not the message.
Consequence: a `chatbot_api_server` that takes 3–5 seconds to shut down gracefully (plausible for a
multi-threaded HTTP server draining connections and joining threads) would still be running when
this script's `kill -0 $SERVER_PID` check ran, triggering the false-positive `"WARNING: Server is
still running after SIGTERM"` branch — which then sends `kill -9` and force-kills a server that was
still in the middle of the very graceful shutdown this script exists to verify, potentially cutting
off in-progress cleanup/log-flush work before it finished on its own.

Changes Made:
- `scripts/test_signal_handling.sh`: changed `sleep 2` to `sleep 5` immediately after the "Waiting
  for graceful shutdown (5 seconds)..." message, so the wait matches what's printed (and what the
  server is actually given to shut down).

Verification:
- ✅ `bash -n scripts/test_signal_handling.sh` — syntax valid.
- ✅ Cross-checked against `test_sigint.sh`'s equivalent wait (`sleep 3` matching its own "...(3
  seconds)..." message) to confirm `5` — not `2` — was the intended value here.

### TD-135: check_ports.sh's `ss` Path Never Actually Reported "No Service Listening"

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/check_ports.sh` | Reset `result` each loop iteration; filter `ss`'s output to `LISTEN` rows before testing emptiness |

Summary:
Found during this session's third full-repository audit pass, restarting on `scripts/`. Two
independent bugs in the same ~10-line loop:

1. **Stale `$result` carryover.** `result` was assigned only inside the `if command -v ss` / `elif
   command -v netstat` branches, with no `else`. On a host with neither tool, `result` kept
   whatever value the *previous* port's iteration left it at — the next port's check would
   silently report the prior port's (possibly "listening") status as its own. Reproduced with a
   standalone harness mimicking the script's exact control flow (fake `command -v` returning true
   for `ss` only on the first of two ports): the second port printed the first port's stale
   `LISTEN ... pid=1234` line instead of "No service listening".
2. **`ss`'s header row made `result` non-empty even with zero matches — the far more reachable
   bug.** `ss -tlnp "sport = :$port"` always prints its column header
   (`State Recv-Q Send-Q Local Address:Port Peer Address:Port Process`) regardless of whether the
   sport filter matched anything; verified directly against a real port with no listener on this
   machine (`ss -tlnp "sport = :19999"` still emitted a 64-byte non-empty header line, exit 0).
   Since `result` was never actually empty whenever `ss` was installed — true on essentially every
   modern Linux host, since `ss` ships with `iproute2` — the "No service listening on port $port"
   message and the entire `lsof` fallback were dead code on every unused port checked with `ss`
   available; the script instead echoed a bare, data-less header line for every port with nothing
   listening. Reproduced by running the (pre-fix) script against this machine's real ports
   8081–8084 (none listening): each printed only the `ss` header, never "No service listening".

Changes Made:
- `scripts/check_ports.sh`: added `result=""` at the top of the loop body (before the
  `ss`/`netstat` branches) so a port with neither tool available always evaluates against its own
  empty state, never the previous port's.
- Changed the `ss` branch to `result=$(ss -tlnp "sport = :$port" 2>/dev/null | grep "LISTEN")`,
  mirroring the `netstat` branch's own grep-filtering pattern, so `result` reflects an actual match
  rather than "ss produced any output at all."

Verification:
- ✅ Standalone bash repro of bug 1 mimicking the script's control flow: **before**, second port
  echoed the first port's stale result; **after** (with `result=""` added), second port correctly
  reported "No service listening".
- ✅ Direct verification of bug 2 against real `ss` output on this machine: confirmed the header row
  is present with zero matching rows (`sport = :19999`, an unused port) and confirmed
  `... | grep "LISTEN"` correctly yields empty output for that same unused port while still
  capturing the real data row for an actually-listening port (8080, a `python3 -m http.server`
  bound for this test).
- ✅ `bash -n scripts/check_ports.sh` — syntax valid.
- ✅ Full before/after run of the real, unmodified/modified script against this machine's actual
  ports 8080 (listening) and 8081–8084 (not listening): **before** the fix, 8081–8084 each printed
  a bare `ss` header line; **after**, each correctly prints "No service listening on port $port",
  while 8080 still correctly reports its `LISTEN ... python3` line and `Process: python3`.

### TD-134: MetricsSessionRegistry's Sweep Thread Raced an Admin Config Write on `completed_ttl_seconds_`

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `src/MetricsSessionRegistry.hpp` | `sweep_loop()` now reads `completed_ttl_seconds_` through the existing mutex-protected `completed_ttl_seconds()` getter instead of the raw member |

Summary:
Found during this session's third full-repository audit pass while re-reading
`MetricsSessionRegistry.hpp` end to end. `completed_ttl_seconds_` (a plain `int`, not atomic) is
guarded by `registry_mutex_` everywhere in the class's public API —
`set_completed_ttl_seconds(int)` writes it under a `std::unique_lock<std::shared_mutex>` on
`registry_mutex_`, and the `completed_ttl_seconds()` getter reads it under a matching
`std::shared_lock`. But the background `sweep_loop()` (a dedicated `std::thread` started whenever
`sweep_interval_seconds > 0`, which is the default in every real deployment — `metrics_api_server`
always constructs its `MetricsSessionRegistry` with a nonzero `sweep_interval_seconds`) read the
member directly and unguarded:
```cpp
lock.unlock();                                    // sweep_mutex_ released
evict_completed_sessions(completed_ttl_seconds_); // <-- raw, unsynchronized read
lock.lock();
```
This read happens once per sweep tick (every `sweep_interval_seconds`, default 60s) with **no lock
of any kind held** — `sweep_mutex_` was just released and `registry_mutex_` was never acquired for
this read. Meanwhile, `TrainingMetricsAPI::handle_admin_put_config()` (`PUT /admin/config`, enabled
by default via `METRICS_API_ALLOW_CONTROL=true`) calls `session_registry_->set_completed_ttl_seconds(...)`
from an arbitrary HTTP-handling thread whenever an operator changes the `completed_ttl_seconds` admin
setting. An unsynchronized concurrent read/write of a non-`atomic` variable from two different threads
is a data race — undefined behavior under the C++ memory model, independent of whether it happens to
"look" safe on any particular platform's `int` representation.

Changes Made:
- `MetricsSessionRegistry::sweep_loop()`: changed `evict_completed_sessions(completed_ttl_seconds_)` to
  `evict_completed_sessions(completed_ttl_seconds())`, routing the read through the same
  `registry_mutex_`-protected getter every other caller already uses. No deadlock risk: the getter's
  `shared_lock` is fully released (function returns) before `evict_completed_sessions()`'s own
  `unique_lock` acquisition begins — they never overlap.

Verification:
- ✅ Standalone regression test added (`tests/metrics_session_registry_test.cpp`,
  `ConcurrentSetCompletedTtlSecondsDuringSweepIsRaceFree`): constructs a `MetricsSessionRegistry` with
  `sweep_interval_seconds=1`, then races a tight-spinning thread calling `set_completed_ttl_seconds()`
  against the registry's own background sweep thread for 2.5 seconds.
- ✅ **Before fix**, built with the `tsan` CMake preset (ThreadSanitizer) and run with ASLR disabled
  (`setarch $(uname -m) -R`, required in this sandboxed environment for TSan's shadow-memory mapping):
  ThreadSanitizer reported `WARNING: ThreadSanitizer: data race ... in MetricsSessionRegistry::sweep_loop()`
  in roughly 1 of every 3–5 runs (inherently timing-dependent, as expected for a real race — the read
  only fires once per second-long sweep tick), confirming the exact read (`sweep_loop()`) and write
  (`set_completed_ttl_seconds()`, via mutex `M0` = `registry_mutex_`) predicted above, both against the
  same stack address.
- ✅ **After fix**, same test, same TSan build, run 15 consecutive times: zero data-race warnings, exit
  code 0 every time (vs. TSan's default `exitcode=66` on a detected race pre-fix).
- ✅ Full `metricsSessionRegistryTests` suite (24 tests, including the new one) passes under the `tsan`
  preset post-fix.

### TD-133: Tizen Dashboard's Settings Save/Cancel Buttons Fired Twice Per Remote OK Press

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `tizen-metrics-app/js` — `app.js` | Removed the redundant `nav.on('ok', ...)` special case for `settings-save-btn`/`settings-cancel-btn` |

Summary:
Found immediately after TD-132, while finishing this session's file-by-file audit of
`tizen-metrics-app/js/`. `TVNav._onKeyDown`'s `KEY.OK` case (`navigation.js`) unconditionally does both
`this._emit('ok', this._focused)` and `this._focused.click()` for whatever element is currently
focused — the `.click()` call exists so that D-pad/remote navigation and an actual mouse/touch click
produce identical behavior for elements that only have a native `click` listener (e.g. the session-list
items). But `settings-save-btn` and `settings-cancel-btn` had special-case branches inside
`app.js`'s `nav.on('ok', ...)` handler (calling `saveSettings()`/`closeSettings()` directly) **and**
their own native `click` listeners wired in `initSettingsInputs()` — so a single OK press on the remote
ran `saveSettings()`/`closeSettings()` twice: once from the `nav.on('ok', ...)` branch, once from the
`.click()`-triggered native listener. For Cancel this was harmless (idempotent UI state reset); for
Save it meant every settings save via the remote fired a full extra `startPolling()`/`openPicker()`
cycle — a wasted duplicate poll of the metrics API, or a wasted duplicate `/api/sessions` fetch,
depending on whether a session was already selected. `card-settings`/`card-session` don't have this
problem — they have no native click listener of their own, so they still need (and keep) their
`nav.on('ok', ...)` branch.

Changes Made:
- `app.js`: removed the `settings-save-btn`/`settings-cancel-btn` branches from `nav.on('ok', ...)`,
  leaving the native `click` listeners (already required for mouse/touch use) as the single path for
  both input methods.

Verification:
- ✅ Manual code-path trace confirmed `card-settings`/`card-session` have no native click listener
  (grepped `index.html` for `onclick`/`addEventListener` — none), so they still require their
  `nav.on('ok', ...)` branch and are unaffected by this change.
- ✅ Standalone Node.js repro (`ok_doublefire_repro.js`) modeling just the control flow (an `emit('ok',
  el)` + `el.click()` pair, mirroring `_onKeyDown`'s OK case exactly) against a mock `settings-save-btn`
  with both a `nav.on('ok', ...)` branch and a native click listener: **before** the fix,
  `saveSettings()` fires twice per simulated OK press; **after**, once. This app has no test framework
  of its own (see TD-049 — `@adai-status: beta` is explicitly capped by that), so this is deliberately a
  standalone, throwaway script rather than a committed test, matching how TD-127/TD-130's coroutine-race
  mechanism was proven in isolation elsewhere in this same audit.
- ✅ `node --check app.js` — syntax valid.

### TD-132: Tizen Dashboard's weight_update_ratio Display Hid a Real Zero Behind "Missing Data"

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `tizen-metrics-app/js` — `app.js` | Fixed the dead trailing ternary so a genuine `0.0` reading displays instead of always falling through to "—" |

Summary:
Found while doing this session's first-ever full read-through of `tizen-metrics-app/js/` (`app.js`,
`navigation.js`, `chart.js` — all in scope per CLAUDE.md's file-status standard, previously only
tag-bumped, never read end to end this session). `applyMetrics()`'s `weight_update_ratio` display used:
```js
var wurStr = (wur != null && !isNaN(wur) && wur !== 0)
    ? (wur < 0.0001 ? wur.toExponential(3) : fmt(wur, 6))
    : (wur === 0 ? '—' : '—');
```
Both arms of the trailing `(wur === 0 ? '—' : '—')` ternary return the identical string — so whether
`wur` was a genuine `0.0` (the server's real initial value before any optimizer step;
`MetricsPushClient.cpp`'s `buf_weight_update_ratio_` defaults to `0.0f`) or truly missing/`NaN`, the
dashboard showed the same "—", indistinguishable from "no data". Every sibling metric in the same
function (e.g. `computeRatioValue`, two lines above) correctly displays a real zero rather than hiding
it — this one field's special-casing of zero was written but never actually took effect, hiding real,
meaningful "no weight update yet" readings from the operator.

Changes Made:
- `app.js`: `wurStr` now checks only `wur != null && !isNaN(wur)` to decide displayed-vs-"—", matching
  the pattern already used by `computeRatioValue`; a genuine `0.0` now formats as `"0.000000"` via
  `fmt(wur, 6)` instead of collapsing to "—". The scientific-notation branch for a real, tiny nonzero
  ratio (`wur !== 0 && wur < 0.0001`) is preserved unchanged.

Verification:
- ✅ Confirmed `weight_update_ratio: 0.0` is a real, server-sent value, not just a theoretical edge
  case: `src/MetricsPushClient.cpp`'s `buf_weight_update_ratio_` is initialized to `0.0f` and pushed
  as-is before the first optimizer step.
- ✅ Standalone Node.js repro (`wur_repro.js`) with the exact buggy ternary copied verbatim alongside
  the fixed version, run against six cases (real zero, tiny nonzero, normal value, undefined, null,
  NaN): **before** the fix, the real-zero case incorrectly printed "—"; **after**, it printed
  `"0.000000"`; all five other cases were identical before and after (no behavior change for the
  already-correct paths). This app has no test framework of its own (TD-049), so this is a standalone
  script, not a committed test — same rationale as TD-133 above.
- ✅ `node --check app.js` — syntax valid.

Also completes this pass's first-ever full read-through of `tizen-metrics-app/js/` — `navigation.js`
and `chart.js` were read in full and found correct (no bugs); `@adai-reviewed` bumped to 2026-09-10 on
all three files, `app.js` version bumped 0.6.0 → 0.7.0.

### TD-131: OpsSettingsDataStore's Comma-Joined Registry Groups Silently Corrupted Names Containing a Comma

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `android/opsdashboard` — `OpsSettingsDataStore.kt` | Replaced `joinToString(",")`/`split(",")` with JSON encoding via kotlinx.serialization |

Summary:
Found while reading through the remaining, previously-unreviewed `opsdashboard` files for this
audit's second pass. `OpsSettingsDataStore` persisted `registryGroups` (a `List<String>`) as a single
DataStore string via `groups.joinToString(",")`, decoded back via `raw.split(",")`. Nothing in the
Settings screen's "Add" flow (`SettingsViewModel.addGroup()`) rejects a comma in a group name — it only
trims and checks for emptiness/duplicates — so a name like `"a,b"` was indistinguishable, once encoded,
from the two separate names `"a"` and `"b"`. The corruption was silent and only surfaced on the next
load of the settings (app restart, or any recomposition that re-collects `OpsSettingsDataStore.settings`):
the group the user actually added would be gone, replaced by two unrelated, wrong ones.

Changes Made:
- `OpsSettingsDataStore.kt`: `encodeGroups`/`decodeGroups` moved to an `internal` companion object (so
  they're directly testable without a real `Context`) and reimplemented using
  `kotlinx.serialization`'s `Json.encodeToString`/`decodeFromString` — already the serialization approach
  used for every DTO in this app, so no character in a group name needs special-casing.
  `decodeGroups` falls back to the old comma-split when JSON parsing fails, so values written by
  installs before this fix (which, by the bug's own precondition, never had an embedded comma) keep
  loading correctly instead of silently resetting to an empty group list.

Verification:
- ✅ Added `OpsSettingsDataStoreTest.kt` (new file, no `Context` needed): a group name containing a comma
  round-trips intact; an old-style plain comma-joined value (as written before this fix) still decodes
  correctly via the fallback; a blank/null raw value decodes to an empty list.
- ✅ Before/after regression: temporarily restored the old comma-join/split bodies in place (keeping the
  companion object so the test still compiled against it), ran the new test — the comma round-trip case
  failed exactly as predicted (`AssertionError`, decoded back as two separate names instead of one).
  Restored the fix, re-ran — all three pass.
- ✅ Full clean `./gradlew` build + `testDebugUnitTest` across all 5 Android modules — 0 failures.

### TD-130: SettingsViewModel (android/app) Raced Navigation the Same Way TD-127 Did in opsdashboard

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `android/app` — `SettingsViewModel.kt`, `SettingsScreen.kt` | Made `save()` a suspend fun with no internal launch, sequenced before `onBack()` in a composable-scoped coroutine — identical fix to TD-127 |

Summary:
Found while reading through the 37 `android/app` files recovered by the TD-129 `.gitignore` fix (this
module had never been reviewed before — it only became visible to git as part of that recovery).
`SettingsScreen`'s Save button did `viewModel.save(); onBack()`, and `SettingsViewModel.save()` was a
plain `fun` that built the settings object and wrote it via
`viewModelScope.launch { settingsDataStore.save(...) }` — a fire-and-forget launch returning before the
write's coroutine actually ran. `onBack()` calls `navController.popBackStack()`, which pops this screen's
`NavBackStackEntry`, clearing its `ViewModelStore` and cancelling `viewModelScope` along with anything
still running in it. If that cancellation landed before the write completed — a real race, since Compose
Navigation's default transition is effectively instant — the save was lost silently: the UI navigates back
as if it saved, but nothing was persisted. This is the exact same mechanism as TD-127
(`opsdashboard`'s `SettingsViewModel`), independently present in this sibling module; the isolated,
empirical proof of the race mechanism recorded under TD-127 (a standalone Kotlin file compiled against the
project's real `kotlinx-coroutines-core` jar, contrasting fire-and-forget-then-cancel losing a write against
sequenced-suspend-then-cancel preserving it) covers this occurrence too — the code shape is identical.

Changes Made:
- `SettingsViewModel.kt`: `save()` → `suspend fun save()`, with the `viewModelScope.launch { ... }` wrapper
  removed — it now directly awaits `settingsDataStore.save(...)`. Also loosened its constructor's
  `settingsDataStore` parameter from the concrete `SettingsDataStore` to the existing `SettingsRepository`
  interface (already the parameter type `ConversationRepository`/`ChatRepository` use), so it can actually
  be constructed with the existing `FakeSettingsRepository` test fake instead of a real `Context`-backed
  DataStore.
- `SettingsScreen.kt`: added `rememberCoroutineScope()`; the Save button's `onClick` now does
  `coroutineScope.launch { viewModel.save(); onBack() }` instead of two unsequenced calls.

Verification:
- ✅ Added `SettingsViewModelTest.kt` (new file): installs a `StandardTestDispatcher` as `Dispatchers.Main`
  (via `kotlinx-coroutines-test`'s `setMain`/`resetMain`, no Robolectric needed) so a fire-and-forget
  `viewModelScope.launch` is provably still *pending* — not yet run — the instant `save()` returns; asserts
  the fake repository already reflects the new host/port immediately after `save()`, with no
  `advanceUntilIdle()` call before the assertion.
- ✅ Before/after regression: temporarily reverted `save()` in place to the pre-fix fire-and-forget shape,
  ran the new test — failed exactly as predicted (`ComparisonFailure`, the fake repository still held its
  construction-time default instead of the newly-entered host). Restored the fix, re-ran — passes.
- ✅ Full `./gradlew :app:testDebugUnitTest` — 11 tests, 0 failures (4 `ChatRepositoryTest` +
  6 `ChatDtoParsingTest` + the 1 new `SettingsViewModelTest`).
- ✅ Full clean `./gradlew` build + `testDebugUnitTest` across all 5 Android modules — 0 failures.

### TD-129: Overly-Broad `.gitignore` Patterns Silently Excluded Real Source Trees From Git Entirely

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `.gitignore`; `android/opsdashboard/.../ui/models/` (5 files); `android/app/src/{main,test,androidTest}/.../chatbot/` (37 files); `tests/fixtures/parquet/` (6 files) | Anchored two bare directory-name patterns to the repo root with a leading `/`, and added a negation exception carving the test fixtures out of a blanket extension rule |

Summary:
Discovered incidentally while bumping `ModelDetailViewModel.kt`'s `@adai-reviewed` date as part of the
Android second-pass audit: `git status` reported the file unmodified despite a real edit. A gitignore
pattern with no leading `/` matches at **any depth** in the tree, not just at the repo root — three
separate rules in `.gitignore`, all written with a root-level C++ build artifact in mind, coincidentally
also matched real, unrelated, long-lived Android/test source directories with the same bare name:

1. `models/` (intended for a root-level `models/` directory of trained C++ checkpoints, referenced by
   `scripts/install_chatbot_API.sh:323`) also matched
   `android/opsdashboard/src/{main,test}/java/com/adai/ops/ui/models/` — the entire "Models" tab package.
   `git ls-files` confirmed `ModelDetailScreen.kt`, `ModelDetailViewModel.kt`, `ModelListScreen.kt`,
   `ModelListViewModel.kt`, and `ModelsRoute.kt` had **never been tracked by git**, since the files were
   first created.
2. `tokenizer`, `encoder`, `chatbot`, `chatbot_trainer` (legacy names from a pre-out-of-source-build
   convention; redundant today since every build lands under `build/<preset>/`, already covered by the
   `build/`/`build-*/` rules) — `chatbot` also matched
   `android/app/src/{main,test,androidTest}/java/com/adai/chatbot/`, the entire `com.adai.chatbot` package.
   `git ls-files android/app/` showed only 11 of 49 files tracked; the missing 37 were **every Kotlin
   source and test file in the whole `android/app` module** — `ChatbotApp.kt`, `MainActivity.kt`, the Room
   DB layer, repositories, DI, network layer, settings, every `ui/` screen/ViewModel, and their unit +
   instrumented tests.
3. `*.parquet` (intended for multi-GB HuggingFace training dumps, per its own "exceed GitHub's 100MB
   limit" comment) also matched `tests/fixtures/parquet/*.parquet` — six tiny (400 bytes–3KB) synthetic
   fixtures generated once by `tests/fixtures/parquet/generate_fixtures.py` and explicitly documented as
   "checked into `tests/fixtures/parquet/`" in `tests/parquet_reader_test.cpp`'s header comment. Only
   `generate_fixtures.py` itself was ever tracked; the six `.parquet` files it produces, which
   `ParquetReaderTest` reads directly off disk at a path relative to `__FILE__`, were not — a fresh clone
   would silently fail every `ParquetReaderTest` case with a missing-file error, with no gitignore-side
   indication why.

A full repo-wide sweep (`git status --porcelain --ignored=matching .`, filtered down to non-build/non-data
noise) after fixing all three found no further collisions: the remaining ignored entries are all
legitimate (build directories, an `android/local.properties` machine-specific SDK path, Room's generated
`android/app/schemas/`, large HuggingFace/Gutenberg training data, a downloaded oneAPI installer, deployed
`adai@<host>` binaries, and `.backup` files).

Changes Made:
- `.gitignore`: `models/` → `/models/`; `tokenizer`/`encoder`/`chatbot`/`chatbot_trainer` →
  `/tokenizer`/`/encoder`/`/chatbot`/`/chatbot_trainer`; added `!tests/fixtures/parquet/*.parquet`
  immediately under the blanket `*.parquet` rule. Each change carries an inline comment explaining what it
  silently excluded and citing this entry.
- `git add -f` (bypassing the not-yet-fixed ignore rules at the time) recovered all 42 previously-invisible
  Kotlin files plus the 6 parquet fixtures as new (`A`) files.
- Read all 42 recovered Kotlin files end-to-end for the first time (they had never been reviewed under
  this audit's standard, unlike files that were merely due for a reviewed-date bump) — see the individual
  per-file `@adai-reviewed`/TD entries for any bugs found during that read.

Verification:
- ✅ `git check-ignore -v` on a representative path from each of the three families confirmed no match
  after the fix (all three previously printed the swallowing rule; all three print nothing after).
- ✅ `git ls-files` counts before/after: `android/opsdashboard/.../ui/models/` 0→5,
  `android/app/` 11→49, `tests/fixtures/parquet/` 1→7 (the script plus all 6 fixtures).
- ✅ `check_file_status.py --strict` unaffected (279 files, 0 problems, both before and after) — it globs
  the filesystem directly rather than going through git, confirming the invisibility was purely a git/CI
  concern, not a gap in the review-standard tooling.
- ✅ Full clean `./gradlew` build + `testDebugUnitTest` across all 5 Android modules re-run after
  recovery to confirm presence-on-disk-only vs. tracked-by-git made no behavioral difference (Gradle was
  always compiling these files from disk regardless of git status).

### TD-128: GroupDetailViewModel's error State Was Computed Incompletely and Never Displayed

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `android/opsdashboard` — `GroupDetailViewModel.kt`, `GroupDetailScreen.kt` | Extended `error` to check all four fetched results; added the missing UI banner |

Summary:
Found while reading `GroupDetailViewModel.kt`/`GroupDetailScreen.kt` end to end. Two compounding gaps:
1. `refresh()` fetches four things every tick (`queue()`, `runs()`, `registry()`, `listModels()`) but its
   `error` computation only ever checked `queueResult`/`runsResult` — `registryResult` (the class's own doc
   comment calls this out as a newer, "Phase 15... previously fetched by nothing in this app" addition) and
   `modelsResult` could fail indefinitely and never surface as an error, even though their stale-data
   fallback (`?: it.registryEntries` / `?: it.models`) was already correctly wired.
2. Independent of (1), `GroupDetailUiState.error` was never actually read anywhere in `GroupDetailScreen.kt`
   at all — confirmed via a full-package grep, the only unrelated `.error` hit was a different data class in
   `GroupListScreen.kt`. Even a `queue()`/`runs()` failure (the two cases the state computation DID already
   handle) produced no visible indication to the user; the screen just silently kept showing stale data.
   `TrainerScreen.kt`'s `StatusSection` already establishes the pattern this screen was missing.

Changes Made:
- `GroupDetailViewModel.kt`: `error` now falls through all four results
  (`queueResult ?: runsResult ?: registryResult ?: modelsResult`).
- `GroupDetailScreen.kt`: added an error banner at the top of `GroupDetailContent`'s `LazyColumn`,
  matching `TrainerScreen`'s "Last status refresh failed: ... (showing last known state)" wording/styling.

Verification:
- ✅ Added `GroupDetailViewModelTest.kt` (new file) with two tests: one injects a `registry()` failure via
  `FakeRegistryApiService` (throwing `IOException`, caught and converted by the existing `safeApiCall`)
  while `queue()`/`runs()` succeed, asserting `state.error` is now populated and the other two fields are
  unaffected; the other confirms no error when every fetch succeeds.
- ✅ Before/after regression: reverted `GroupDetailViewModel.kt` to its pre-fix `HEAD` version in place,
  ran the new test — it failed exactly as predicted (`AssertionError` on the `assertNotNull(state.error)`
  line). Restored the fix, re-ran — both tests pass.
- ✅ Full `./gradlew :opsdashboard:testDebugUnitTest` — 46 tests (44 existing + 2 new), 0 failures, 0
  errors.

### TD-127: SettingsViewModel.save() Raced Navigation-Triggered ViewModel Clearing, Silently Losing Saves

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `android/opsdashboard` — `SettingsViewModel.kt`, `SettingsScreen.kt` | Made `save()` a suspend fun and sequenced it before `onBack()` in a composable-scoped coroutine, instead of a fire-and-forget `viewModelScope.launch` |

Summary:
Found while beginning a full read-through of the Android `opsdashboard` module (the second-pass audit's
first extension beyond `src/`/`scripts/` into `android/` and `tizen-metrics-app/js/`, both in scope per
CLAUDE.md's file-status standard). `SettingsScreen`'s Save button did `viewModel.save(); onBack()` — two
unsequenced calls back to back. `SettingsViewModel.save()` built the settings object and then wrote it via
`viewModelScope.launch { settingsRepository.save(settings) }`: a fire-and-forget launch that returns
immediately, before the DataStore write's coroutine has actually run. `onBack()` calls
`navController.popBackStack()`, which pops this screen's `NavBackStackEntry` — and popping it clears its
`ViewModelStore`, which clears `SettingsViewModel`, which cancels `viewModelScope` and every coroutine still
running in it. If that cancellation happens before the DataStore write's coroutine resumes and completes
(a real race — Compose Navigation's default transition is effectively instant with no custom animation),
the write is cancelled mid-flight and the save is silently lost: the UI has already navigated back as if
it succeeded, with no error surfaced anywhere.

Changes Made:
- `SettingsViewModel.kt`: changed `fun save()` to `suspend fun save()`, removing its internal
  `viewModelScope.launch` — the function now directly awaits `settingsRepository.save(settings)` rather
  than firing it off and returning immediately.
- `SettingsScreen.kt`: the Save button now uses `rememberCoroutineScope()` (a scope tied to the
  composable, not the ViewModel) to launch a coroutine that calls `viewModel.save()` and *then* `onBack()`
  — sequencing them in one coroutine body guarantees `onBack()` (and the ViewModel-clearing cascade it
  triggers) cannot run until the save has actually completed, regardless of how fast that cascade would
  otherwise happen.

Verification:
- ✅ Isolated mechanism proof using the project's actual `kotlinx-coroutines-core` dependency (no Android
  framework needed — full ViewModel-level testing was blocked by `WatchFacePushRepository`'s hard
  dependency on a real `android.content.Context`, which this project has no mocking library or Robolectric
  to construct in a plain JVM test): compiled and ran a standalone repro contrasting the two patterns
  against a fake suspend-based repository.
  - Pre-fix pattern (fire-and-forget launch, then immediately cancel the scope — modeling
    `viewModelScope.launch { save() }` immediately followed by `onCleared()`'s cancellation): the write
    was lost every time (`saved = null`).
  - Post-fix pattern (await the suspend call, then cancel the scope — modeling the fixed sequencing): the
    write always completed first (`saved = "B"`) before the stand-in "onBack" action even ran, confirming
    a sequenced suspend call is fundamentally immune to a scope cancellation that happens after it in the
    same coroutine body, regardless of timing.
- ✅ `./gradlew :opsdashboard:compileDebugKotlin :opsdashboard:testDebugUnitTest` — clean build, all 44
  existing `opsdashboard` unit tests still pass (0 failures, 0 errors); this session also established a
  fresh full-module baseline beforehand (`./gradlew testDebugUnitTest`, all 5 Android modules: 58 tests,
  0 failures).

### TD-126: test_chatbot_gui_comprehensive.sh Had the Same Wrong-Binary Bug as TD-124, More Extensively

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/test_chatbot_gui_comprehensive.sh` | Same fix as TD-124: point size/Qt-linkage/symbol checks at `chatbot_gui_binary` |

Summary:
Found immediately after TD-124, checking this file's simpler sibling for the same bug — and it's present
far more extensively here. Test 3 (size), Test 4 (Qt5/Qt6 linkage, 4 sub-checks), Test 5 (4 component
symbol checks), and Test 6 (3 Qt slot symbol checks) — 12 individual assertions in total — all ran against
the thin `chatbot_gui` launcher instead of `chatbot_gui_binary`, the actual Qt GUI executable that contains
all of that content. Also found a second, unrelated bug while fixing this: the "all tests passed" summary
printed `cd /home/rodney/Repos/adai` — the same single-developer hardcoded-path pattern already fixed
elsewhere this session (TD-116/117/118), here in an `echo`'d instruction rather than an actual `cd`.

Changes Made:
- `scripts/test_chatbot_gui_comprehensive.sh`: introduced a `GUI_BINARY="build/src/chatbot_gui_binary"`
  variable; Test 3's size check and all of Tests 4-6 now target it instead of `chatbot_gui`. Test 3 also
  now requires `chatbot_gui_binary` to exist (in addition to the launcher) before reporting "Executable
  built successfully."
- Changed the printed `cd /home/rodney/Repos/adai` to `cd $(pwd)`, matching wherever the script is
  actually being run from rather than one developer's checkout path.

Verification:
- ✅ Before/after regression against a real, freshly rebuilt tree (same temporary `build/src ->
  debug/src` symlink technique as TD-124; removed after testing, confirmed no stray git changes since
  `build/` is gitignored):
  - Pre-fix: 11 of 28 checks failed — the size check, all 4 Qt-linkage sub-checks, all 4 component-symbol
    checks, and the corresponding slot checks — confirmed via the actual failing-check list.
  - Post-fix: those same 11 checks all pass (27 of 31 passed — the 2 remaining failures, "GUI guide
    documentation not found" and "Quick reference README not found," are genuinely missing files
    unrelated to this bug, confirmed via direct `ls`, and correctly out of scope for this fix).
- ✅ `bash -n` and ShellCheck (`-S warning`, clean — the remaining `-S info` notes are pre-existing,
  unrelated unquoted-numeric-comparison style nits already present throughout this file).

---

### TD-125: ConversationContext's keep_system_message Was a Silent No-Op

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `ConversationContext` (chatbot/GUI conversation history) | Wired `keep_system_message` into `clear()` and `truncate_to_limits()`; left `clear_all()` unconditional |

Summary:
Found while reading `src/ConversationContext.cpp`/`.hpp` end to end. `keep_system_message` is a
constructor parameter and member field (default `true`), stored, round-tripped through
`save_to_file()`/`load_from_file()` (`KEEP_SYSTEM:` line), and propagated in `create_summarized()` —
but nothing ever read it to gate behavior:

- `clear()` unconditionally preserved the system message regardless of the flag.
- `clear_all()` unconditionally dropped the system message regardless of the flag (arguably correct —
  see below).
- `truncate_to_limits()` never evicted the system message at all when trimming to `max_messages`/
  `max_tokens`, regardless of the flag — the eviction loop only ever touches `messages`, and the
  `system_message` optional sits outside it entirely.

So constructing a `ConversationContext` with `keep_system_message=false` had zero observable effect
anywhere. No production caller ever actually passes `false` (`ChatbotAPI::Session` and `ChatbotGUI`
both use the 1–2 arg constructor overloads, which default the flag to `true`), so this was latent
rather than an active behavioral bug for any current caller — but the flag's own doc comment ("Whether
to always keep system message") and the existing `SystemMessageNotTruncated` test (which constructs
with `keep_system_message=true` specifically to explain why the system message survives) both describe
a flag that's supposed to matter, and it didn't.

Design decision: `clear()` and `truncate_to_limits()` now honor the flag — when `false`, `clear()` also
drops the system message, and `truncate_to_limits()` evicts it as a last resort if the token budget is
still exceeded once every regular message is gone. `clear_all()` deliberately stays unconditional: its
own contract ("clear everything including system message") is a distinct, explicit nuclear option, not
a mode governed by the persistent per-instance flag — gating it on `keep_system_message` would silently
break its documented behavior (and the existing `ClearAll` test, which constructs with the default
`true` flag and asserts the system message is gone after `clear_all()`).

Changes Made:
- `src/ConversationContext.cpp`: `clear()` now resets `system_message` when `keep_system_message` is
  false. `truncate_to_limits()` now evicts `system_message` after the existing token-truncation loop
  when `keep_system_message` is false, `total_tokens` is still over `max_tokens`, and a system message
  is set.
- `src/ConversationContext.hpp`: updated the doc comments for the constructor's `keep_system_message`
  parameter, `clear()`, `clear_all()`, and `truncate_to_limits()` to state the actual (now-correct)
  behavior and cross-reference this entry.

Verification:
- ✅ Added 4 tests to `tests/conversationcontext_test.cpp`:
  `TruncationKeepsSystemMessageWhenFlagTrue`, `TruncationEvictsSystemMessageWhenFlagFalse`,
  `ClearDropsSystemMessageWhenFlagFalse`, `ClearAllIgnoresKeepSystemMessageFlag`.
- ✅ Before/after regression: reverted `ConversationContext.{cpp,hpp}` to their pre-fix versions
  (keeping the new tests), rebuilt `conversationcontextTests`, and confirmed exactly the two tests
  tied to the actual behavior change failed (`TruncationEvictsSystemMessageWhenFlagFalse` — system
  message survived with 100 tokens still over the budget of 10; `ClearDropsSystemMessageWhenFlagFalse`
  — system message survived `clear()`), while the flag-true and `clear_all()` tests passed either way
  as expected; restored the fix, rebuilt, and confirmed all 66 tests in `conversationcontextTests` pass.
- ✅ `adai_nlp` and `chatbotcliTests` rebuild cleanly against the changed header/implementation.

Files Changed:
- `src/ConversationContext.hpp`
- `src/ConversationContext.cpp`
- `tests/conversationcontext_test.cpp`

---

### TD-124: test_chatbot_gui.sh Verified the Wrong Binary — Always Failed on a Correct Build

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/test_chatbot_gui.sh` | Point the size/ELF/Qt5/symbol checks at `chatbot_gui_binary`, the real GUI executable, instead of the thin `chatbot_gui` launcher |

Summary:
Found while reading `test_chatbot_gui.sh` end to end, immediately after confirming (for TD-114/116/117/118)
that `chatbot_gui` really does build under `build/src/`. `chatbot_gui` is not the GUI itself — per
`src/CMakeLists.txt`, it's `ChatbotGUI_wrapper.cpp`, a thin `exec()` launcher that just fixes up
snap/library environment variables and then execs the real binary, `chatbot_gui_binary` (built from
`ChatbotGUI_main.cpp` + `ChatbotGUI.cpp`, linked against `adai_models`/`adai_nlp`/`adai_attention`/
`adai_core`). This test script's size check, ELF check, Qt5-dependency listing, and all four `nm` symbol
checks (`ChatbotGUI`, `BPETokenizer`, `EncoderDecoderModel`, `ConversationContext`) ran against the
**wrapper**, which by design contains none of that — confirmed directly: `nm` found 0 matching symbols in
`chatbot_gui` vs. 743 in `chatbot_gui_binary`, and the wrapper has zero Qt5 linkage at all (it doesn't
`#include` anything Qt-related). Worse, the size check (`< 1MB` fails) meant the script `exit 1`'d
immediately at that point on *every* run against a correct build — the wrapper is ~124 KB, the real binary
~17 MB — so this script could never get past its own third check to report anything about the actual
components it exists to verify.

Changes Made:
- `scripts/test_chatbot_gui.sh`: kept the existence/executable-permission checks against the actual
  `chatbot_gui` launcher (that's what a user runs, and checking it exists and is executable is legitimate),
  but added an explicit existence check for `chatbot_gui_binary` and pointed every other check (size, ELF
  validity, Qt5 dependencies, all four symbol checks) at it instead.

Verification:
- ✅ Before/after regression against a real, freshly rebuilt tree (via a temporary `build/src ->
  debug/src` symlink; `build/` is gitignored, symlink removed after testing, confirmed no stray git
  changes):
  - Pre-fix: failed at the size check exactly as predicted — `❌ FAIL: Executable too small (127528
    bytes)`, exit 1, before any of the Qt5/symbol checks ever ran.
  - Post-fix: all checks pass against the real 17 MB `chatbot_gui_binary` — correct size, valid ELF, all
    three Qt5 libraries listed, and all four component symbols found — script reaches its final "Build
    Verification: SUCCESS" banner and exits 0.
- ✅ `bash -n` and ShellCheck (`-S warning`, clean — the two remaining `-S info` notes, unquoted
  `$SIZE`/`$GUI_BINARY` in the size comparison and `numfmt` call, are pre-existing and harmless since
  `$SIZE` is always a plain integer from `stat -c%s`).

### TD-122: verify_gui_parallel.sh's Printed Convenience-Script Path Was Wrong

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/verify_gui_parallel.sh` | Corrected `./run_chatbot_gui.sh` to `./scripts/run_chatbot_gui.sh` |

Summary:
Found while reading `verify_gui_parallel.sh` end to end (it already correctly uses `build/src/chatbot_gui*`
throughout — that target genuinely has no `RUNTIME_OUTPUT_DIRECTORY` override, unlike the `chatbot`/
`chatbot_api_server` bugs fixed as TD-114/116/117/118). Every path in this script is relative to the repo
root, consistent with its own `./build/...` checks — but its closing tip said `./run_chatbot_gui.sh`,
while that script actually lives in `scripts/run_chatbot_gui.sh`. A user following the printed instruction
verbatim from the repo root would get "No such file or directory."

Changes Made:
- `scripts/verify_gui_parallel.sh`: corrected the printed path to `./scripts/run_chatbot_gui.sh`.

Verification:
- ✅ Confirmed `run_chatbot_gui.sh` exists only at `scripts/run_chatbot_gui.sh`, not at the repo root.
- ✅ `bash -n` and a clean ShellCheck pass (`-S warning`).

### TD-121: check_ports.sh's Own Self-Documented Port List Gap (mns_server, Trainer Admin API)

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/check_ports.sh` | Added ports 8083 (mns_server) and 8084 (trainer admin API) to the checked list |

Summary:
This file's own `@adai-status` line already documented the gap being fixed here: `PORTS=(8080 8081
8082)` covers `chatbot_api_server`, `metrics_api_server`, and `registry_server`, but omits `mns_server`
(8083) and `incremental_trainer serve`'s admin API (8084) — both real, current ports per CLAUDE.md's
service/port tables. Since the fix is unambiguous (the two missing ports are well-established elsewhere
in the docs, not a design question) and the change is a trivial, safe addition to a read-only diagnostic
script's port list, fixed it directly rather than leaving the self-acknowledged gap in place.

Changes Made:
- `scripts/check_ports.sh`: `PORTS` now includes 8083 and 8084.
- Removed the now-resolved caveat from the `@adai-status` line.

Verification:
- ✅ Ran the patched script (read-only — it only queries `ss`/`lsof`/`netstat`, no mutation) and confirmed
  it now reports on all five ports, including the two previously-omitted ones.
- ✅ `bash -n` and a clean ShellCheck pass (`-S warning`).

### TD-120: install_oneapi_libs.sh's --help Leaked the Internal Tag Block, Same Bug Class as TD-107

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/install_oneapi_libs.sh` | Replaced the blank-line-bounded `sed` range with the same pattern-based `awk` extraction used to fix TD-107 |

Summary:
Found while reading `install_oneapi_libs.sh` end to end — the second independent occurrence this session
of the TD-107 bug class. `--help` ran `sed -n '2,/^$/s/^# \?//p' "$0"`, intending to print the header
comment block "from line 2 up to its first blank line." But the file's `@adai-status`/`@adai-version`/
`@adai-reviewed` tag block (lines 3-5, right after the shebang) has its own blank line immediately after
it (line 6) — well before the real usage documentation even begins — so the sed range ended there,
printing only the three tag lines. The real usage text (description, `Usage:`, and the full `Options:`
list) never printed at all.

Changes Made:
- `scripts/install_oneapi_libs.sh`: replaced the `sed` one-liner with the same `awk` pattern already used
  to fix TD-107 in `model_service.sh` — skip the shebang, any `@adai-*` tag line, and any other
  non-comment line (blanks, the `set -euo pipefail` line that sits between the tag block and the real
  doc) by pattern, then print every subsequent comment line until the first non-comment line.

Verification:
- ✅ Before/after regression against the real file: pre-fix `--help` printed exactly the 3 tag lines
  (confirmed via `wc -l` = 3, matching the predicted symptom); post-fix prints the full, correct usage
  text (description, `Usage:`, and all five `Options:` entries).
- ✅ `bash -n` and a clean ShellCheck pass (`-S warning`).

### TD-119: fix_markdown_lint.py Corrupted Code-Block Content Containing 2+ Pipe Characters

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/fix_markdown_lint.py` | Gave the MD060/MD036 second pass its own fence-tracking, mirroring the first pass |

Summary:
Found while reading `fix_markdown_lint.py` end to end. The file runs two passes over each markdown file:
the first pass tracks `in_code_block` and explicitly leaves fenced code-block content untouched (per its
own comment, "Inside code blocks: pass through untouched"); the second pass — MD060 table-compacting and
MD036 emphasis-as-heading — has **no fence tracking of its own** and runs unconditionally over every line
the first pass produced, including everything inside code fences. Its trigger condition, "line contains 2+
literal `|` characters," matches extremely common code content: any bash pipeline with two pipes (`cat f |
grep x | wc -l`), or the two adjacent `|` characters in a C/C++ `||` operator. Reproduced directly: a
fenced bash block containing `cat foo.txt | grep bar | wc -l` came out as `cat foo.txt |grep bar| wc -l` —
the tool's table-formatting logic silently stripped the spacing around the pipes in a real, documented
shell command, in direct contradiction of its own stated invariant. Since this script's whole purpose is
bulk in-place rewriting of every `.md` file in the repository, a real (non---check) run risked quietly
degrading code examples throughout the documentation corpus.

While reading the file, also found the docstring's rule list claims MD029 (ordered list renumbering) is
fixed, but no renumbering logic exists anywhere in the code — a "documented but not implemented" gap. Given
the tool's high blast radius (it bulk-rewrites real documentation), a new feature implementation risked
introducing a fresh, untested mutation path rather than fixing a bug with well-understood before/after
behavior; corrected the docstring to accurately describe what the tool does instead.

Changes Made:
- `scripts/fix_markdown_lint.py`: the second pass now tracks its own code-fence state (mirroring the
  first pass's `is_code_fence()`/`in_code_block` logic) and skips both the MD060 and MD036 transformations
  entirely while inside a fence, applying them only to real prose/table content.
- Removed the false MD029 claim from the docstring's rule list, with a note on why it isn't implemented.

Verification:
- ✅ Before/after regression via direct reproduction: a bash code block containing `cat foo.txt | grep
  bar | wc -l` — pre-fix, `fix_markdown()` returned it mangled to `cat foo.txt |grep bar| wc -l`
  (`changed=True`); post-fix, the code block passes through byte-for-byte identical, while a genuine
  markdown table immediately after it in the same input is still correctly compacted
  (`| Col1  |  Col2 |` → `|Col1|Col2|`) — confirming the fence-tracking fix doesn't regress the feature it
  was protecting.
- ✅ Ran the patched tool in its non-mutating `--check` mode against the real repository (155 files
  flagged, consistent with a large existing corpus of minor pre-existing lint issues, no crashes) —
  deliberately did not run the mutating (`--fix`) mode against real files, to avoid a second file-mutation
  incident this session (see the TD-116/117/118 entry's incident note on `run_tests.sh`).
- ✅ `python3 -m py_compile` and `pyflakes` clean.

### TD-118: Six More Scripts Referenced chatbot/chatbot_api_server at the Wrong Build Path

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/model_service.sh`, `verify_cli_parallel.sh`, `test_signal_handling.sh`, `test_sigint.sh`, `test_config_reload.sh`, `test_log_rotation.sh` | Corrected `build/src/` to `build/bin/`; resolved paths from `${BASH_SOURCE[0]}` instead of a hardcoded developer path |

Summary:
Found via a proactive repo-wide grep for `/src/chatbot`, `/src/chatbot_api_server`, etc. across all of
`scripts/*.sh`, prompted by finding the same wrong-directory bug independently in TD-114
(`package_windows.sh`), TD-116 (`manual_test_reload.sh`), and TD-117 (`run_chatbot.sh`). The grep also
turned up many correct references to `build/src/chatbot_gui` — that target genuinely has no
`RUNTIME_OUTPUT_DIRECTORY` override in `src/CMakeLists.txt`, so it really does build under `.../src/`,
unlike `chatbot` and `chatbot_api_server`, which both explicitly set `RUNTIME_OUTPUT_DIRECTORY` to
`${CMAKE_BINARY_DIR}/bin`. Six more files had the wrong one:
- **`model_service.sh`** (already reviewed once this session under TD-107, without catching this) —
  `get_binary()` returned `.../build/src/chatbot_api_server` (and `.../build/release/src/...`) for a tool
  whose entire job is finding and launching that exact binary.
- **`verify_cli_parallel.sh`** — `CHATBOT_BINARY="./build/src/chatbot"`.
- **`test_signal_handling.sh`** and **`test_sigint.sh`** — both also hardcoded
  `/home/rodney/Repos/adai/build/src/chatbot_api_server` (the TD-116-class absolute-path bug) on top of
  the wrong subdirectory.
- **`test_config_reload.sh`** and **`test_log_rotation.sh`** — both `cd /home/rodney/Repos/adai` then
  `./build/src/chatbot_api_server`, the same double bug.

Changes Made:
- All six: `build/src/chatbot*` corrected to `build/bin/chatbot*`.
- The four that also hardcoded `/home/rodney/Repos/adai` (`test_signal_handling.sh`, `test_sigint.sh`,
  `test_config_reload.sh`, `test_log_rotation.sh`) now resolve `REPO_ROOT` from `${BASH_SOURCE[0]}`,
  matching every other script in `scripts/`.

Verification:
- ✅ `src/CMakeLists.txt` confirms both `chatbot` and `chatbot_api_server` set `RUNTIME_OUTPUT_DIRECTORY`
  to `${CMAKE_BINARY_DIR}/bin` — the same authoritative source already used to verify TD-114/116/117 — and
  this session's own build logs show `Linking CXX executable ../bin/chatbot_api_server`.
- ✅ `bash -n` and a clean ShellCheck pass (`-S warning`) on all six patched files (the remaining `-S
  info` notes on `test_log_rotation.sh` — an unused loop counter, `ls` vs `find` style suggestions — are
  pre-existing and unrelated).

### TD-117: run_chatbot.sh Had the Same build/src/ vs build/bin/ Bug, Twice

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/run_chatbot.sh` | Corrected both `CLIENT_BIN` and `SERVER_BIN` from `build/src/` to `build/bin/` |

Summary:
Found immediately after TD-116, while checking `run_chatbot.sh` — the general-purpose "start the server
if needed and launch the CLI" convenience script. `CLIENT_BIN="${BUILD_DIR}/src/chatbot"` and
`SERVER_BIN="${BUILD_DIR}/src/chatbot_api_server"` both pointed at the wrong subdirectory for the same
reason as TD-114/TD-116: neither target has ever built there. Unlike TD-116, this script already resolved
`BUILD_DIR` correctly relative to `${BASH_SOURCE[0]}` — only the `bin` vs `src` subdirectory was wrong.

Changes Made:
- `scripts/run_chatbot.sh`: corrected both `CLIENT_BIN` and `SERVER_BIN` to `${BUILD_DIR}/bin/...`.

Verification:
- ✅ Confirmed against `src/CMakeLists.txt`'s `set_target_properties(chatbot ... RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/bin)` and the equivalent for `chatbot_api_server`, plus this session's own build
  logs showing both binaries actually linked into `.../bin/`.
- ✅ `bash -n` and ShellCheck (`-S warning`, clean — the one remaining `-S info` note, unquoted
  `$SERVER_CMD`, is a pre-existing, intentional word-splitting idiom for the appended CLI flags,
  unrelated to this fix).

### TD-116: manual_test_reload.sh Hardcoded One Developer's Home Directory and the Wrong Build Subdirectory

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/manual_test_reload.sh` | Resolve the repo root from `${BASH_SOURCE[0]}` like every other script; corrected `build/src/` to `build/bin/` |

Summary:
Found while checking the ShellCheck-flagged `cd` calls in `scripts/*.sh`. This manual SIGHUP-reload test
script did `cd /home/rodney/Repos/adai` — a single developer's absolute home-directory path hardcoded into
a file tracked in shared version control — instead of the `SCRIPT_DIR="$(cd "$(dirname
"${BASH_SOURCE[0]}")" && pwd)"` pattern every other script in `scripts/` uses. It happened to work in this
environment purely because the checkout coincidentally lives at that exact path; it would fail immediately
for any other developer, any other checkout location, or CI. Separately, the executable path it then ran —
`./build/src/chatbot_api_server` — was also wrong: `chatbot_api_server`'s CMake target sets
`RUNTIME_OUTPUT_DIRECTORY` to `${CMAKE_BINARY_DIR}/bin` (confirmed directly against `src/CMakeLists.txt`
and against this session's own build logs, which show `Linking CXX executable ../bin/chatbot_api_server`)
— the build has never actually produced a binary at `build/src/chatbot_api_server`.

Changes Made:
- `scripts/manual_test_reload.sh`: resolves `REPO_ROOT` from `${BASH_SOURCE[0]}`, matching every sibling
  script, and `cd`s there with an explicit `|| exit 1` (also addressing the ShellCheck SC2164 note on the
  original bare `cd`). Corrected the binary path to `./build/bin/chatbot_api_server`.

Verification:
- ✅ Verified the corrected path resolution logic directly (prints the real `SCRIPT_DIR`/`REPO_ROOT` and
  the resulting exec path, matching the actual repo layout) and confirmed `chatbot_api_server`'s real
  build output location against both `src/CMakeLists.txt`'s `set_target_properties` call and this
  session's own build log.
- ✅ `bash -n` and a clean ShellCheck pass (`-S warning`).
- A full live run (actually launching the server) is deferred until the `build/debug` rebuild in progress
  completes — see the note below about an incident during this same round of work.

**Incident note:** while testing `run_tests.sh`'s TD-115 fix, a `--coverage` run of the *real* script
against the *real* repository (not a copy) executed its `rm -rf "$BUILD_DIR"` step with
`BUILD_DIR="$PROJECT_ROOT/build"` — deleting the entire `build/` directory, including the fully-built
`build/debug` tree from earlier in this session (verified 78/78 passing). This was a mistake: `run_tests.sh`
should have been exercised in an isolated copy the way every other destructive-flag test in this session
was, not run in place. `build/debug` was immediately reconfigured (`cmake --preset=debug`) and a full
rebuild was kicked off; the tail of this session re-runs the full `ctest` suite once it completes, before
the final checkpoint, to re-confirm the 78/78 baseline.

### TD-115: run_tests.sh Silently Skipped Coverage-Report Generation Whenever a Test Failed

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/run_tests.sh` | Bracket the `ctest` call with `set +e`/`set -e` so a failing test run no longer kills the script before its own `$?` is captured |

Summary:
Found while continuing the sweep for the `set -e` dead-branch bug already fixed under TD-104/111/112/113
in sibling scripts — a new variant. `ctest --output-on-failure[--verbose]` runs as a bare statement (not
an `if`'s own condition), immediately followed by `TEST_RESULT=$?` and, later, an `if [ "$COVERAGE" = true
]` block that generates an lcov coverage report and prints a summary. Under this script's `set -e`, a
failing test run terminates the script right at the bare `ctest` call — `TEST_RESULT=$?` never executes,
and neither does the entire coverage-report section below it. The final exit code happens to still be
correct by coincidence (`set -e`'s own termination propagates the failing command's exit status as the
script's exit status), but the practical, observable bug is that `--coverage` combined with any failing
test silently produced no coverage report at all — exactly the case where seeing what was and wasn't
exercised is most useful.

Changes Made:
- `scripts/run_tests.sh`: wrapped the `ctest` invocation in `set +e` / `set -e`, so a non-zero exit is
  captured into `TEST_RESULT` without killing the script, and execution reaches the coverage-report
  section (and the final `exit $TEST_RESULT`) regardless of whether tests passed.

Verification:
- ✅ Before/after regression against the real script with fake `cmake`/`make`/`ctest`/`lcov` binaries in
  `PATH` (the fake `ctest` exits non-zero to simulate a failing test, `--coverage` requested):
  - Pre-fix: output ended immediately after "fake ctest: 1 test failed" — no "📊 Generating coverage
    report..." line, exactly the predicted symptom.
  - Post-fix: "📊 Generating coverage report..." correctly prints, the `lcov` calls run, and the script
    still exits with the correct code (8, the fake ctest's exit status) via the intended `exit
    $TEST_RESULT` path.
  - Re-verified the plain success path (no sanitizer/coverage flags, tests pass): unaffected, exits 0.
- ✅ `bash -n` and ShellCheck (`-S warning`, clean — the two remaining `-S info` notes, unquoted
  `$CMAKE_OPTS`/`$(nproc)`, are pre-existing, intentional word-splitting idioms unrelated to this fix).

### TD-114: package_windows.sh Looked for Built Executables in the Wrong Directory

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/package_windows.sh` | Corrected `<build_dir>/src/` to `<build_dir>/bin/`, matching CMake's actual `RUNTIME_OUTPUT_DIRECTORY` |

Summary:
Found while reading `package_windows.sh`, the packaging counterpart to the just-fixed `build_windows.sh`.
Its preflight check and all three executable-copy steps hardcode `${BUILD_DIR}/src/chatbot*.exe`, but
`src/CMakeLists.txt` sets `RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin` on the `chatbot` target (and
every other executable target) — every build, Windows cross-compile included, places its output in
`<build_dir>/bin/`, never `<build_dir>/src/`. This meant `package_windows.sh` could never get past its own
first check (`ERROR: chatbot.exe not found!`) against any real `build_windows.sh` output — it has been
completely non-functional, matching the shape of TD-109's "tool that can never succeed" finding.

Changes Made:
- `scripts/package_windows.sh`: corrected all four `${BUILD_DIR}/src/...` references (the preflight check
  plus the three `cp` calls) to `${BUILD_DIR}/bin/...`.
- Left `chatbot_trainer.exe` as-is (still copied best-effort, `|| true` as before) rather than renaming it
  to the actual current `incremental_trainer` target: no `chatbot_trainer` CMake target has existed for a
  long time, but this package's generated `README.txt`/`run_chatbot.bat` document the old
  `--data/--vocab/--output` flag style, not `incremental_trainer`'s real `train`/`retrain`/`resume`/
  `reset`/`serve` subcommands — swapping just the binary name would trade one stale reference for a
  misleading one. Left a comment noting this for a follow-up pass that also rewrites the accompanying
  documentation, rather than partially fixing it here.

Verification:
- ✅ Before/after regression against a realistic dummy build tree (`<build_dir>/bin/chatbot.exe`, etc. —
  matching CMake's real output layout, not a `src/`-based mock): pre-fix reproduced the exact predicted
  `ERROR: chatbot.exe not found!` / exit 1; post-fix found and copied the executables, created the ZIP
  archive, and completed with exit 0 and an accurate package summary.
- ✅ `bash -n` and ShellCheck (`-S warning`, clean; the two `-S info` "read without -r" notes are
  pre-existing and unrelated to this fix — the loop only ever processes plain `.exe`/size-summary output
  with no backslashes).

### TD-113: check_tech_debt.sh Died After Its First Scan and Pointed at a File That No Longer Exists

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/check_tech_debt.sh` | Report scan counts via a global variable instead of the function's own exit status; fixed `DEBT_FILE`'s stale path |

Summary:
Found while sweeping `scripts/*.sh` for the `set -e` dead-branch pattern already fixed under TD-104/
TD-111/TD-112, and turned out to be an even more severe variant. `scan_pattern()` ends with `return
$count` so its caller can read the count back via `$?`; a bash function's return value **is** an exit
status, though, and this script runs under `set -e` — so the very first call, `scan_pattern "TODO" ...`,
returns non-zero the moment even one TODO exists (which is always true for this codebase — CLAUDE.md
itself documents that most `// TD-NNN` tags left inline today are intentional historical footnotes, not
open work), and `set -e` kills the script immediately, right there, before `TODO_COUNT=$?` on the next
line ever runs. In practice this meant the script never got past printing the raw TODO listing: the
FIXME/HACK/XXX scans, the `=== Summary ===` block, the tracked-vs-untracked check, and the "Tracked
Technical Debt" report at the end never executed at all, and the script exited with `count mod 256` — a
meaningless leftover number, not a real status. Separately, `DEBT_FILE="TECHNICAL_DEBT.md"` pointed at
the repo root, but that file was moved to `docs/development/guides/TECHNICAL_DEBT.md` (per this session's
own commits) and never existed there at all in the current repo layout — so even the parts of the script
downstream of the `set -e` bug would have reported "WARNING: TECHNICAL_DEBT.md not found!" and exited,
same TD-109/TD-110 class of "referenced a path that moved elsewhere" bug.

Changes Made:
- `scripts/check_tech_debt.sh`: `scan_pattern()` now sets a global `LAST_COUNT` instead of `return`ing
  the count; each caller reads `LAST_COUNT` immediately after the call instead of `$?`.
- `DEBT_FILE` now points at `docs/development/guides/TECHNICAL_DEBT.md`, its real current location.
- Left the pre-existing "any marker found ⇒ exit 1 with instructions to track it" policy unchanged —
  whether that check should instead verify each marker actually cites a tracked `TD-XXX` (rather than
  treating any marker at all as untracked debt) is a design question, not a bug this pass is positioned
  to resolve; restoring the script to actually *run* and report accurate counts is the scope here.

Verification:
- ✅ Before/after regression against the real script and real repository (not a synthetic mock):
  - Pre-fix: captured output truncated at exactly 42 lines — the TODO listing only — with a meaningless
    exit code (36, i.e. this run's actual TODO count mod 256). No FIXME/HACK/XXX/Summary output at all,
    confirming the predicted symptom.
  - Post-fix: full output (68 lines) including correct `FIXME markers: None found`, `HACK markers: None
    found`, `XXX markers: None found`, an accurate `=== Summary ===` block (`TODO: 36, Total: 36`), and
    the untracked-debt message correctly citing the real path
    (`docs/development/guides/TECHNICAL_DEBT.md`).
- ✅ `bash -n` and ShellCheck (`-S warning`, clean).

### TD-112: build_windows.sh Had the Same set -e/bare-command Dead-Error-Branch Bug, Twice

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/build_windows.sh` | Made both the `cmake` configure step and the `cmake --build` step the `if`'s own condition |

Summary:
Found while proactively sweeping all of `scripts/*.sh` for the same anti-pattern already fixed twice this
session (TD-104 `docker_build.sh`, TD-111 `docker_deploy.sh`): `grep -rln '\[ \$? -eq 0 \]' scripts/*.sh`
turned up a third file. `build_windows.sh` runs under `set -e` and has the identical shape at **two**
separate points — the CMake configure step and the `cmake --build` step each run as a bare multi-line
statement, then check `if [ $? -eq 0 ]` on the next line. In both cases, a real failure (missing
toolchain file, a broken CMakeLists change, a compile error) would trigger `set -e` and kill the script
immediately at the bare command, before either `if` was ever reached — both `else` branches (the red
"✗ CMake configuration failed" / "✗ Build failed" messages plus a controlled `exit 1`) were unreachable
dead code, and a real failure just silently died with cmake's own raw exit code instead.

Changes Made:
- `scripts/build_windows.sh`: both the CMake configure invocation and the `cmake --build` invocation are
  now the direct condition of their `if` statements, same fix pattern as TD-104/TD-111.

Verification:
- ✅ Before/after regression against the real script with a fake `cmake` shim in `PATH`: exercised all
  four paths — configure-fails, build-fails (configure succeeds), and full success.
  - Pre-fix (`git stash` to the original, run, restore fix): a failing fake `cmake` configure call exited
    with its own raw code (5) and printed no "✗ CMake configuration failed" message — the predicted
    symptom.
  - Post-fix: configure failure correctly prints the red failure message and exits `1`; build failure
    (with configure succeeding) correctly prints "✗ Build failed" and exits `1`; full success reaches the
    final "Build Summary" section and exits `0`, unchanged from before.
- ✅ `bash -n` and ShellCheck (`-S warning`, clean — the one remaining `-S info` note at line 108,
  `read exe` without `-r`, is pre-existing, unrelated to this fix, and not a demonstrated bug for the
  plain `.exe` filenames this loop actually processes).

### TD-111: docker_deploy.sh Had the Same set -e/eval Dead-Error-Branch Bug as TD-104

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/docker_deploy.sh` | Use `eval "$RUN_CMD"` directly as the `if` condition, same fix as TD-104 |

Summary:
Found while reading `docker_deploy.sh` end to end — the exact same bug already fixed under TD-104 in the
sibling `docker_build.sh`, independently present here: `start_container()` ran `eval $RUN_CMD` as a bare
statement, then checked `if [ $? -eq 0 ]`. Under this script's `set -e`, a failing `docker run` terminates
the script immediately at the bare `eval`, before the `if` below is ever reached — the `else` branch
(`print_error "Failed to start container"; exit 1`) was unreachable dead code, and a real failure to start
the container just silently killed the script with Docker's own raw exit code.

Changes Made:
- `scripts/docker_deploy.sh`: changed `eval $RUN_CMD` from a bare statement into the `if`'s own condition
  (`if eval "$RUN_CMD"; then ... else ... fi`), identical to the TD-104 fix.

Verification:
- ✅ Before/after regression against the real script (not a synthetic snippet): shadowed `docker` in
  `PATH` with a fake binary whose `run` subcommand exits non-zero.
  - Pre-fix (`git stash` to restore the original in place, run, then restore the fix): exited with the
    fake docker's raw code (42), no `[ERROR] Failed to start container` message ever printed.
  - Post-fix: printed `[ERROR] Failed to start container` and exited the intended code `1`.
  - Re-verified the success path with a fake `docker run` that exits `0`: reaches `[SUCCESS] Container
    started successfully` and the follow-up info lines exactly as before.
- ✅ `bash -n` and a clean ShellCheck pass (`-S warning`) on the patched file.

### TD-110: serve_dashboard.py Served From Its Own Directory Instead of the Repo Root Where dashboard.html Lives

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/serve_dashboard.py` | Serve from the repo root (parent of `scripts/`), not from `scripts/` itself |

Summary:
Found immediately after TD-109, while checking whether `serve_dashboard.py` has its own independent bug
beyond `dashboard.html`'s absence. `DIRECTORY = os.path.dirname(os.path.abspath(__file__))` resolves to
the directory this script itself lives in — `scripts/` — but `dashboard.html` has always lived at the
**repo root**, confirmed by `docs/operations/OPERATIONS_MANUAL.md`'s component table (`| Dashboard |
dashboard.html |`) and by `package_server_bundle.sh`, which reads it from `"${REPO_ROOT}/dashboard.html"`.
So even independent of TD-109 (the file being entirely missing from the repo), this script would still
404 on its own advertised URL (`http://localhost:8082/dashboard.html`, printed by the script itself at
startup) the moment someone places `dashboard.html` back at its documented, correct location — it was
looking one directory too deep the whole time.

Changes Made:
- `scripts/serve_dashboard.py`: `DIRECTORY` now resolves to `dirname(dirname(__file__))` — the repo
  root — instead of `dirname(__file__)`.

Verification:
- ✅ Live end-to-end reproduction: placed a dummy `dashboard.html` at the repo root (its documented,
  correct location) and started the real server.
  - Pre-fix: `GET /dashboard.html` → `404 Not Found` (server logged "File not found").
  - Post-fix: `GET /dashboard.html` → `200 OK` with the dummy file's actual content returned.
- ✅ Also directly confirmed the resolved `DIRECTORY` value: pre-fix `.../scripts` (wrong — that's where
  `dashboard.html` would need to live for the old code to find it, contradicting the documented location);
  post-fix `.../adai` (the real repo root, matching `OPERATIONS_MANUAL.md` and `package_server_bundle.sh`).

### TD-109: package_server_bundle.sh Could Never Succeed — Required Three Files Deleted Elsewhere as Stale

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/package_server_bundle.sh` | Made `config-remote.conf`/`vocab.txt`/`dashboard.html` optional bonus content instead of hard preflight requirements |

Summary:
Found while reading `package_server_bundle.sh` end to end and noticing its preflight check hard-required
`config-remote.conf`, `vocab.txt`, and `dashboard.html` at the repo root — none of which exist in this
repository at all. `git log` traced this to two earlier, unrelated cleanup commits: `1f06218` ("chore: add
opsdashboard Android app, DaemonConfigStore, per-service config files; remove tracked training data
artifacts") deleted `config-remote.conf` and `vocab.txt` as stale/generated data, and `c338e8f` ("chore:
untrack generated dashboard/training-session artifacts") deleted `dashboard.html` under the stated belief
that it's "regenerated per-run" — but no such regeneration mechanism exists anywhere in the codebase, so
it is simply gone from a fresh checkout. Neither cleanup commit updated `package_server_bundle.sh`'s
preflight check to match, so every invocation of the script against the current repo has unconditionally
failed with `Missing file: .../config-remote.conf` (or `vocab.txt`/`dashboard.html`, depending which check
ran first) — the tool has been completely non-functional since `1f06218` landed, with no way to ever
successfully produce a deployment tarball. None of the three files are actually needed for what
`install_server_bundle.sh` (the installer this bundle exists to feed) installs — `mns_server`,
`registry_server`, and `metrics_api_server` — which generates its own `config.conf` from scratch and
never reads a chatbot vocabulary or a static dashboard page at all.

Changes Made:
- `scripts/package_server_bundle.sh`: `config.conf` (which does still exist and is still needed) remains
  a hard requirement; `config-remote.conf`, `vocab.txt`, and `dashboard.html` are now treated the same as
  the file's own already-established "optional bonus" pattern (already used for the `mns_manager_gui`/
  `vocab_builder` binaries just above this code) — included in the tarball if present, skipped with a
  warning otherwise, instead of aborting packaging entirely.
- The generated `README.txt`'s "Training metrics dashboard" section is now appended conditionally, only
  when `dashboard.html` was actually bundled, so the instructions never promise a file that isn't there.

Verification:
- ✅ Before/after regression against the real, current repository state (not a synthetic mock): reverted
  to the pre-fix version in place (`git stash`), ran it — reproduced the exact predicted failure
  (`Missing file: .../config-remote.conf`, exit 1, no tarball produced). Restored the fix, re-ran with the
  same arguments — completed successfully (exit 0), producing a real 3.0M, 17-file tarball containing the
  actual built binaries, `config.conf`, and the install scripts. Extracted the bundled `README.txt` and
  confirmed the "Training metrics dashboard" section is correctly absent (since `dashboard.html` wasn't
  available to bundle in this environment).
- ✅ `bash -n` and a clean ShellCheck pass (`-S warning`) on the patched file.

### TD-108: install_mns_server.sh and mns-cli-guide.md Told Operators to Set a Config Key That Doesn't Exist

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/install_mns_server.sh`, `docs/operations/guides/mns-cli-guide.md` | Corrected `MNS_SERVER_URL` to the real config key, `NAME_SERVICE_URL` |

Summary:
Found while reading `install_mns_server.sh` end to end. Its `print_summary()` tells the operator, after a
successful install: `"Configure clients with: MNS_SERVER_URL=http://<host>:8083"`. But `Config.cpp`
(`ConfigLoader::load()`, the single config parser used by every ADAI binary — see CLAUDE.md
"Configuration") only ever recognizes the literal key `NAME_SERVICE_URL` (both from the config file and
as an environment-variable override); `MNS_SERVER_URL` is not read anywhere in the codebase at all. An
operator who followed this script's own printed instructions verbatim — adding `MNS_SERVER_URL=...` to a
worker's `config.trainer.conf`/`config.chatbot.conf` — would get a config key that's silently ignored:
`name_service_url` stays empty, and per CLAUDE.md's documented MNS-authoritative behavior, the client
silently falls back to client-local run/session numbering with no error, no warning, nothing — exactly the
kind of "accepted but never actually applied" failure mode this session found repeatedly in `src/` (e.g.
TD-092/TD-100/TD-101), just surfacing here as bad documentation rather than a code defect, with the same
practical consequence for the operator. The identical wrong variable name was also independently present
in `docs/operations/guides/mns-cli-guide.md`, describing when `incremental_trainer` calls MNS
automatically.

Changes Made:
- `scripts/install_mns_server.sh`: `print_summary()` now prints `NAME_SERVICE_URL=...`, matching the real
  config key.
- `docs/operations/guides/mns-cli-guide.md`: corrected the same reference.

Verification:
- ✅ Exhaustive repo-wide `grep -rn "MNS_SERVER_URL"` across `src/`, `scripts/`, `docs/`, `tests/`,
  `android/`, `tizen-metrics-app/` before the fix found only these two occurrences — confirming it is
  never read as a real config key anywhere, only ever told to operators. `Config.cpp:341` shows the actual
  (and only) recognized key is a literal `key == "NAME_SERVICE_URL"` string comparison, which is
  unambiguous and needs no dynamic reproduction (a plain string-equality check has no runtime-dependent
  behavior to verify beyond what the source itself guarantees). Re-ran the same grep after the fix: zero
  remaining occurrences of the wrong key anywhere in the tree.

### TD-107: model_service.sh's `help` Command Leaked the Internal File-Status Tag Block

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/model_service.sh` (`cmd_help()`) | Skip the shebang, `@adai-*` tag lines, and leading blank lines by pattern instead of a hardcoded "skip 2 lines" |

Summary:
Found while continuing the `scripts/` read-through (after TD-104/105/106). `cmd_help()` extracts the
script's usage documentation directly from its own top-of-file comment block via `awk`, using `NR < 3
{ next }` to skip past the shebang before printing. That worked back when the shebang was immediately
followed by the doc banner, but every file in this repo (this one included) now carries a 3-line
`@adai-status`/`@adai-version`/`@adai-reviewed` header comment right after the shebang, plus a blank line
— `NR < 3` only skips the shebang and the first tag line, so running `./scripts/model_service.sh help`
(or `-h`, `--help`, or any unrecognized command, which all fall through to the same `cmd_help`) printed
the two remaining internal tag lines and a blank line as if they were the first lines of the actual usage
text, ahead of the real `ADAI Model Service Manager` banner.

Changes Made:
- `scripts/model_service.sh`: rewrote the `awk` extraction to skip lines by *pattern* — the shebang
  (`^#!`), any `@adai-*` tag line, and leading blank lines — rather than a hardcoded line count, so it
  keeps working correctly regardless of how many header lines precede the real doc block (and doesn't
  regress again the next time this file's own tag block changes shape).

Verification:
- ✅ Before/after regression by running the actual `help` command against the real file: pre-fix printed
  the three `@adai-status`/`@adai-version`/`@adai-reviewed`/blank lines before the real banner (confirmed
  against the pre-fix `HEAD` version); post-fix output starts directly at the `====` banner line.
  Also confirmed the two other paths that share `cmd_help()` — `-h` and the "unknown command" error
  fallback — both produce the same corrected output.

### TD-106: setup_postgres() Reported "Schema Applied" Even When the Schema Failed Entirely

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/install_server_bundle.sh` (`setup_postgres()`) | Pass `-v ON_ERROR_STOP=1` to `psql -f`, and fix the `ERROR*` case pattern to match real psql error line format |

Summary:
Found immediately after TD-104/TD-105, continuing the same read-through of `scripts/`. `setup_postgres()`
applies `scripts/setup_postgres.sql` — a single `BEGIN`/…/`COMMIT` transaction creating every metrics/MNS
table — via `psql -d "${PG_DB_NAME}" -f "${SETUP_SQL}" | while IFS= read -r line; do case "${line}" in
...; esac; done`, then unconditionally prints `success "PostgreSQL schema applied"`. Two compounding bugs
made this always look like success even on a total schema failure:
1. **`psql`'s own default behavior**: without `-v ON_ERROR_STOP=1`, `psql -f` prints each statement error
   and keeps going, then still exits `0` at the end (the connection itself succeeded) — even though, since
   the whole file is one transaction, a single bad statement rolls back *everything*, leaving zero tables
   created. `set -euo pipefail` (already at the top of the file) only helps once `psql`'s own exit status is
   actually non-zero — it was always `0` here.
2. **The `case` pattern for detecting errors never matched real output**: real `psql` error lines are
   formatted as `psql:<file>:<line>: ERROR:  <message>`, not a bare `ERROR` prefix, so the `ERROR*)`
   arm silently fell through to the unstyled catch-all `*)` branch — meaning even the per-line diagnostic
   display never actually highlighted a failure in red.
Net effect: a real schema error (e.g. a future edit introduces a syntax error, a permissions issue, an
incompatible PostgreSQL version) would leave the database with **no tables at all**, yet the installer
would print a plain (unhighlighted) error line buried among green `[SUCCESS]` lines, followed immediately
by a confident green `"PostgreSQL schema applied"` banner and continue on to `install_bundle`'s own
"installed!" summary — completely masking a broken deployment.

Changes Made:
- `scripts/install_server_bundle.sh`: added `-v ON_ERROR_STOP=1` to the `psql -f` invocation, so `psql`
  now actually returns a non-zero exit status on the first statement error (matching this codebase's
  established "fail loudly rather than silently corrupt/omit state" philosophy — see `ParquetReader.hpp`'s
  header doc for the same principle applied elsewhere). Combined with the file's existing
  `set -euo pipefail`, the script now aborts immediately after the pipeline finishes displaying all
  diagnostic lines, never reaching the false success banner.
- Fixed the `case` pattern from `ERROR*)` to `*ERROR:*)` so real psql error lines are actually
  detected and printed in red via `error()`, instead of falling through to the plain, unstyled `*)` arm.

Verification:
- ✅ Before/after regression against a **real** PostgreSQL instance (not a mock): initialized a scratch,
  unprivileged PostgreSQL 16 cluster (`initdb`/`pg_ctl`, custom Unix-socket-only data dir, no root needed)
  and ran the exact `psql -f ... | while read ...; case ...; esac; done` pattern as a real standalone
  script file against a deliberately broken schema (a `CREATE TABEL` typo inside a `BEGIN`/`COMMIT` block,
  mirroring the production file's structure).
  - Pre-fix: pipeline exited `0`; the error line printed unstyled (matched only the catch-all `*)` arm);
    script proceeded past `success "PostgreSQL schema applied"` to full completion — exactly the predicted
    false-success symptom, and confirmed via `ROLLBACK` in the psql output that the table was never
    actually created despite an earlier line already having been printed as `[SUCCESS] CREATE TABLE`.
  - Post-fix: `psql` returned exit status `3`; the error line printed correctly styled in red
    (`[ERROR]   psql:...:3: ERROR:  syntax error at or near "TABEL"`); the script aborted immediately
    after displaying it and never reached `success "PostgreSQL schema applied"`.
  - Re-ran the fixed pattern against the real, unmodified `scripts/setup_postgres.sql` against the same
    scratch cluster: all 12 statements applied cleanly, all printed as `[SUCCESS]`, script reached its own
    end normally with exit `0` — the success path is unaffected.
- ✅ `bash -n` and a clean ShellCheck pass (`-S warning`) on the patched file.

### TD-105: install_server_bundle.sh Always Exited 0 Even When a Service Failed to Start

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/install_server_bundle.sh` | Consult the already-computed `all_ok` flag after `print_summary` and `exit 1` when it's false |

Summary:
Found immediately after TD-104, continuing the same read-through of `scripts/`. `install_bundle()`'s
verification step polls each of the three systemd services with `systemctl is-active --quiet` and sets a
local `all_ok=false` if any of them isn't running — but `all_ok` was never actually read again afterward.
`print_summary()` unconditionally printed the cheerful `"ADAI Server Bundle installed!"` banner, and with
no explicit exit code anywhere after that, the function (and the whole script, since `install_bundle` is
its last top-level statement) always returned exit status `0` — success — regardless of whether any
service actually came up. A caller scripting this installer (CI, a provisioning tool, `./install_server_
bundle.sh && echo ok`) would see a clean success exit code even for a partially-broken installation; the
only visible sign of trouble was a `warn` line easy to miss in scrollback, immediately followed by a
"installed!" success banner.

Changes Made:
- `scripts/install_server_bundle.sh`: after `print_summary`, check `all_ok` and, if false, print an
  explicit failure warning and `exit 1`.

Verification:
- ✅ `bash -n` on the patched file and a clean ShellCheck pass (`-S warning`) — no new findings.
- ✅ Before/after regression via an isolated mechanism reproduction (the real function requires root and
  live systemd units, and stubbing that whole environment is exactly the 14-20 hour harness already
  tracked as TD-043 — not repeated here): extracted the same `local all_ok=true` / loop / `print_summary`
  / (no check vs. `exit 1` check) shape into a standalone script with a fake `systemctl is-active`
  that fails for one service.
  - Pre-fix pattern: printed the "installed!" summary and exited `0` despite the failed service —
    the exact predicted symptom.
  - Post-fix pattern: printed the same summary, then the new warning, and exited `1`.
  - Confirmed the all-services-healthy case is unaffected (exits `0`, no extra warning) — the fake
    `systemctl is-active` returning true for every service reaches the same code path with `all_ok`
    still `true`.

### TD-104: docker_build.sh's Error-Handling Branch Was Dead Code Under `set -e`

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `scripts/docker_build.sh` | Use `eval "$BUILD_CMD"` directly as the `if` condition instead of a bare statement followed by `if [ $? -eq 0 ]` |

Summary:
Found while beginning a rigorous read-through of the `scripts/` directory (in scope per CLAUDE.md's
file-status standard) following the completion of the second full-codebase re-read of `src/`. The script
runs under `set -e`, then does:
```bash
eval $BUILD_CMD
if [ $? -eq 0 ]; then
    print_success ...
else
    print_error "Docker build failed"
    exit 1
fi
```
`set -e` exempts a command's exit status only when that command is *itself* the condition of an
`if`/`while`/`until` — a bare statement followed by a separate `if [ $? -eq 0 ]` check does not qualify,
because the bare `eval $BUILD_CMD` is a plain top-level command. So the moment `docker build` actually
fails, `set -e` terminates the whole script right there, before the `if` below is ever reached — the
`else` branch (the friendly `[ERROR] Docker build failed` message and controlled `exit 1`) was unreachable
dead code. In practice, a real build failure just silently killed the script with Docker's own raw exit
code and no diagnostic message from the wrapper at all.

Changes Made:
- `scripts/docker_build.sh`: changed `eval $BUILD_CMD` from a bare statement to the condition of the `if`
  itself (`if eval "$BUILD_CMD"; then ... else ... fi`), which is one of `set -e`'s documented exemptions,
  so the `else` branch now actually executes on failure. Also quoted the `eval` argument.

Verification:
- ✅ Before/after regression via manual reproduction (this script has no automated test — already covered
  by the broader TD-043 "deployment scripts have no automated test" item): shadowed `docker` in `PATH`
  with a fake binary that always exits non-zero, run under both the pre-fix and post-fix script.
  - Pre-fix: script printed only the fake docker's own output and exited with the fake docker's raw exit
    code (17) — no `[ERROR] Docker build failed` message ever printed, exactly the predicted symptom.
  - Post-fix: script printed `[ERROR] Docker build failed` and exited with the intended code `1`.
  - Also re-verified the success path is unaffected: a fake `docker` that exits 0 still reaches
    `print_success` and the "Next steps" output exactly as before.

### TD-103: chatbot_gui Wrapper Located Its Sibling Binary Relative to the Caller's cwd, Not Its Own Install Directory

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `ChatbotGUI_wrapper.cpp` | Resolve own path via `/proc/self/exe` instead of parsing `argv[0]` |

Summary:
Found during the second, independent full-codebase re-audit while reading `ChatbotGUI_wrapper.cpp` end to
end. `main()` located its own directory by taking `argv[0]` and slicing off everything after the last `/`,
then exec'd `<that dir>/chatbot_gui_binary`. This only works when `argv[0]` actually contains a `/` — true
when the wrapper is invoked as `./chatbot_gui` or `build/debug/src/chatbot_gui`, but **not** when it's
invoked via a bare `PATH` lookup (e.g. a `.desktop` launcher's `Exec=chatbot_gui`, or the wrapper copied/
symlinked into a directory on `PATH`), where the shell passes `argv[0]` as literally `"chatbot_gui"` with no
slash at all. In that case `find_last_of("/")` returns `npos`, `exe_dir` silently fell back to `"."`, and the
subsequent `execvp("<cwd>/chatbot_gui_binary", ...)` looked in the *caller's current working directory*
instead of the wrapper's actual install directory — failing with "Failed to execute ... No such file or
directory" for any cwd that doesn't happen to equal the install directory.

Changes Made:
- `src/ChatbotGUI_wrapper.cpp`: extracted a `resolve_exe_dir()` helper that first tries
  `readlink("/proc/self/exe", ...)`, which always resolves to the real path of the running executable
  regardless of how it was invoked; falls back to the old `argv[0]`-parsing behavior only if that fails.

Verification:
- ✅ Before/after regression via manual reproduction (this file has no automated test — pre-existing
  `TD-036` caps its status at "beta" specifically for having no smoke test — so verification is a real,
  reproducible shell repro rather than a gtest case): copied the built `chatbot_gui` wrapper alone into an
  isolated directory on `PATH`, then ran `exec -a chatbot_gui chatbot_gui --help` from an unrelated cwd
  (`/tmp`) so `argv[0]` carries no `/`.
  - Pre-fix: wrapper printed `Error: Failed to execute ./chatbot_gui_binary` / `No such file or directory`
    — it looked in `/tmp` (the caller's cwd), not its own directory.
  - Post-fix: wrapper's error path (with the sibling binary absent) correctly named
    `/tmp/guitest/chatbot_gui_binary` (its own real directory via `/proc/self/exe`); with the sibling
    binary copied alongside it, the exec succeeded and printed the GUI binary's own `--help` usage text.
- ✅ Full rebuild of the `chatbot_gui`/`chatbot_gui_binary` targets succeeds clean under Qt5.

### TD-102: json_pretty() Got Stuck "Inside a String" After a Value Ending in an Escaped Backslash

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `mns_gui::json_pretty()` (`MnsJsonHelpers.hpp`) | Reuse the escape-aware `find_string_end()` scanner (already fixed for the same bug class under TD-068) instead of a naive single-character lookback |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `MnsJsonHelpers.hpp` end to
end, immediately after re-verifying the already-fixed TD-068 bug in `json_array_objects()`. That function's
own comment claims `json_pretty()` "already" skips string content the same (correct, escape-aware) way —
but `json_pretty()` actually used a much weaker check: `if (c == '"' && (i == 0 || s[i-1] != '\\'))` toggles
an `in_string` flag by looking at only the *immediately preceding* character. For a real closing quote that
happens to follow an escaped backslash in the JSON text — e.g. a string value ending in a literal
backslash, such as a Windows-style artifact path `"C:\\models\\"` — that preceding character genuinely is
`\`, so the check misclassified the true closing quote as itself escaped and never toggled back out of
"in string". Every character for the rest of the output (real commas, braces, colons included) was then
appended verbatim with no further indentation or newline insertion, garbling the pretty-printed result from
that point on. `json_array_objects()` had the identical class of bug and was already fixed under TD-068 by
introducing `find_string_end()`, an escape-aware scanner that correctly walks past `\X` pairs instead of
looking at only one preceding character — `json_pretty()` was simply never updated to use it.

Changes Made:
- `src/MnsJsonHelpers.hpp`: `json_pretty()` now calls `find_string_end()` to locate each string literal's
  true closing quote and copies the whole literal (quotes and content) verbatim in one step, instead of
  toggling a flag per-character based on a one-character lookback.

Verification:
- ✅ Added `HandlesStringEndingInEscapedBackslash` to `tests/mns_manager_gui_test.cpp` (next to the
  existing `MnsJsonPretty` tests): pretty-prints `{"path":"C:\\models\\","next":"ok"}` and asserts a
  newline follows the comma after the path value — only possible if the scanner correctly recognized it
  had left the string there.
- ✅ Before/after regression: reverted `MnsJsonHelpers.hpp` to its pre-fix `HEAD`, rebuilt
  `mnsManagerGuiTests`, confirmed the new test fails with the exact predicted symptom (output showed
  `"path": "C:\\models\\","next":"ok"}` — everything after the path value squashed onto one line with no
  further formatting); restored the fix, rebuilt, confirmed pass.
- ✅ Full `mnsManagerGuiTests` non-live suite (36/36; 8 live tests skip without a running `mns_server`,
  unaffected by this change) passes.

### TD-101: RAGInference Silently Dropped Its Own gen_config's temperature/top_k/top_p/num_beams

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `RAGInference::generateWithRetrieval()` | Route through the now-fixed `generate_response_with_strategy()` instead of `generate_response()`, passing all of `gen_config`'s fields |

Summary:
Found immediately after TD-100 while re-reading `RAGInference.cpp` end to end and tracing where
`RAGConfig::gen_config` (declared as a full `TextGenerator::GenerationConfig`, "Generation parameters") was
actually used. `generateWithRetrieval()` called `model->generate_response(augmented_prompt,
config.gen_config.max_length)` — but `generate_response()`'s `max_length` parameter is a documented no-op
(kept only "for interface parity", per the comment on its `gpu_generate_response()` twin), and
`temperature`/`top_k`/`top_p`/`num_beams` were never even passed to it at all — that overload doesn't
accept them. A caller configuring `RAGConfig::gen_config` with a specific temperature, top-k/top-p
threshold, or beam width got none of it honored; RAG generation always ran with whatever
`EncoderDecoderModel`'s internal `generator` already held.

Changes Made:
- `src/RAGInference.cpp`: `generateWithRetrieval()` now calls `generate_response_with_strategy()` (the
  method TD-100 just fixed to actually sync its arguments into `generator`'s config) with an empty
  strategy string — which falls through to the same "combined generate() using config" behavior
  `generate_response()` always used — passing `gen_config`'s `max_length`, `temperature`, `top_k`, `top_p`,
  and `num_beams` through explicitly.

Verification:
- ✅ Added `GenerateSyncsGeneratorConfigFromRAGConfig` to `tests/raginference_test.cpp`: constructs a
  `RAGInference` with a `RAGConfig` carrying distinctive `gen_config` values, calls `generate()`, and
  asserts `model->get_generator()->get_config()` (a public accessor) reflects them — the same
  internal-state verification technique used for TD-100, rather than generated output length/content,
  which would be unreliable against an untrained, randomly-initialized model.
- ✅ Before/after regression: reverted `RAGInference.cpp` to its pre-fix `HEAD`, rebuilt
  `raginferenceTests`, confirmed the new test fails with the exact predicted symptom (`max_length` read
  back as 128 — the model's own `max_seq_length` — instead of the requested 17; `temperature`/`top_k`/
  `top_p` stuck at their defaults instead of 0.42/7/0.55); restored the fix, rebuilt, confirmed pass.
- ✅ Full `raginferenceTests` suite (37/37) passes.

### TD-100: generate_response_with_strategy() Ignored max_length (and Often temperature) for Every Non-Beam Strategy

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `EncoderDecoderModel::generate_response_with_strategy()` | Sync `generator`'s stored config with this call's arguments once, up front, before dispatching to any strategy |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `RAGInference.cpp`, tracing
its call into `EncoderDecoderModel::generate_response()` (already known, per an existing comment on the
`gpu_generate_response()` twin, to silently ignore its own `max_length` parameter — kept only "for
interface parity"), and cross-checking the richer sibling method, `generate_response_with_strategy()`, that
`ChatbotGUI.cpp` calls directly with its own strategy/temperature/top_k/top_p/max_length arguments.
`TextGenerator::generate_greedy()`/`generate_sampling()`/`generate_top_k()`/`generate_nucleus()` each take
at most one of their own filter values as an explicit parameter (e.g. `generate_top_k()`'s own `k`) —
everything else, **including `max_length` itself**, is read from `generator`'s own *stored* `config`, not
from any parameter. Only the `"beam"` branch of `generate_response_with_strategy()` ever pushed this call's
arguments (`max_length`, `num_beams`) into that stored config before generating; every other named strategy
— `"greedy"`, `"sampling"`, `"topk"`, `"nucleus"`, and the unrecognized-strategy fallback — silently
generated using whichever `max_length` `generator` already happened to hold (its constructor default, e.g.
the model's `max_seq_length`, or whatever an unrelated earlier call left behind), never this call's own
argument of the same name. `"topk"` and `"nucleus"` compound this for `temperature` specifically: both call
their own internal `apply_temperature(logits, config.temperature)` (temperature isn't one of their explicit
parameters at all), so a caller's `temperature` argument was silently dropped for those two strategies
too — a caller requesting `generate_response_with_strategy(text, 12, "nucleus", 0.3f, 5, 0.6f)` got
whatever `max_length`/`temperature` `generator` already had, `top_p=0.6f` honored (nucleus's own explicit
parameter), and no error of any kind. Existing tests (`GenerateWith{Sampling,TopK,Nucleus,Greedy}Strategy`)
only asserted `EXPECT_NO_THROW`, never that the requested values actually took effect, so this went
uncaught.

Changes Made:
- `src/EncoderDecoderModel.cpp`: `generate_response_with_strategy()` now reads `generator->get_config()`,
  overwrites `max_length`/`temperature`/`top_k`/`top_p`/`num_beams` with this call's own arguments, and
  writes it back via `generator->set_config()` — once, before the strategy dispatch — so every strategy's
  internal "read from config" fallback sees the values this specific call actually asked for. No
  per-strategy filtering logic changed; only which values feed it.

Verification:
- ✅ Added `GenerateWithStrategySyncsGeneratorConfig` to `tests/encoderdecoder_test.cpp`: calls
  `generate_response_with_strategy()` with `"topk"` (temperature not one of its explicit parameters) and
  distinctive `max_length`/`temperature`/`top_k`/`top_p`/`num_beams` values, then asserts
  `model.get_generator()->get_config()` (a public accessor) actually reflects them — verified directly
  against internal state rather than generated output length/content, which would be unreliable against
  an untrained, randomly-initialized model. A second call with different `"nucleus"` values confirms the
  sync overwrites rather than merely coexisting with a prior call's config.
- ✅ Before/after regression: reverted `EncoderDecoderModel.cpp` to its pre-fix `HEAD`, rebuilt
  `encoderdecoderTests`, confirmed the new test fails with the exact predicted symptom (`max_length` stuck
  at 512 — the model's `max_seq_length` constructor default — instead of the requested 12; `temperature`
  and `top_p` stuck at their 1.0 defaults instead of 0.3/0.6); restored the fix, rebuilt, confirmed pass.
- ✅ Full `encoderdecoderTests` suite (62/62) passes.

### TD-099: conversationcontext_test.cpp and chatbotcli_improved_test.cpp Raced on the Same Hardcoded Filename

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `tests/conversationcontext_test.cpp` (`SaveToFile`) | Use a filename distinctive to this test instead of one shared with another test binary |

Summary:
Found incidentally while verifying this session's second-pass batch: a full `ctest -j$(nproc)` run reported
`ConversationContextTests` failing (`SaveToFile`, `file.good()` false) — running the binary alone passed
100%, pointing at a parallel-execution-only flake rather than a real `ConversationContext` defect. Both
`tests/conversationcontext_test.cpp`'s `SaveToFile` test and `tests/chatbotcli_improved_test.cpp`'s
`ChatbotCLITest` fixture used the exact same hardcoded relative filename, `"test_conversation.txt"`, in
the same working directory (`build/<preset>/`, ctest's default cwd for every test binary). `ChatbotCLITest`
unconditionally calls `std::remove()` on it in `TearDown()` after *every one* of its test cases regardless
of whether that particular test ever wrote it. Under `ctest -j`, both binaries run concurrently, so
`ChatbotCLITest`'s teardown could delete the file out from under `SaveToFile`'s
save-then-reopen-and-verify sequence in the other binary — a classic TOCTOU race between two otherwise
fully independent test binaries sharing unguarded external state (a bare relative filename).

Changes Made:
- `tests/conversationcontext_test.cpp`: renamed `SaveToFile`'s filename to
  `"test_conversationcontext_savetofile.txt"`, distinctive enough that no other test binary in this repo
  could plausibly reuse it.

Verification:
- ✅ Reproduced the race directly and rigorously: reverted just this filename to the pre-fix
  `"test_conversation.txt"`, then ran `chatbotcliImprovedTests` and `conversationcontextTests
  --gtest_filter=*SaveToFile*` concurrently (backgrounded pairs, `wait`ed) for 30 iterations —
  `SaveToFile` failed in 28 of 30 (only 2 passed), matching the exact `file.good()` symptom seen in the
  full ctest run.
- ✅ Restored the fix and re-ran the identical 30-iteration concurrent stress loop: 30/30 passed, 0
  failures — confirms the race is gone, not just less likely.

### TD-098: TokenBatchLoader's Dynamic Batching Silently Misaligned Input/Target Rows and Dropped Sequences

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `TokenBatchLoader::load_batch()` (`ParallelDataLoader.hpp`) | Always build both batches via plain `create_batch()`, dropping the broken `create_dynamic_batches()` path for this already-fixed-size slice |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/BatchProcessor.hpp` end
to end and tracing its `create_dynamic_batches()` free function into every caller — one of which,
`TokenBatchLoader::load_batch()` (the same class fixed for TD-087 earlier in this pass, for an unrelated
dual-queue deadlock), had `config_.use_dynamic_batching` (`true` by default) routed through it with
`max_batch_size == input_sequences.size()` specifically so the whole already-fixed-size slice
(`start_idx..end_idx`, computed just above from a sequential chunk of the shuffled epoch — this loader has
no upstream length-based grouping across the epoch, only this one per-slice call) would land in a single
`TokenBatch`. `create_dynamic_batches()` is designed to split an *unbounded pool* into multiple
length-homogeneous batches, and does so purely from each list's own lengths regardless of `max_batch_size`
whenever the slice's length spread exceeds `length_tolerance` — two real bugs followed: (1)
`input_sequences` and `target_sequences` were sorted **independently** by their own (generally
uncorrelated) lengths before batching, so `input_batch.batch_token_ids[k]` and
`target_batch.batch_token_ids[k]` stopped corresponding to the same original sample the moment their
length orderings diverged — every affected training step would silently pair one sample's input with a
*different* sample's target; (2) when `create_dynamic_batches()` did split a slice into multiple
length-groups, only the first group (`batches[0]`) was kept, silently dropping every sequence in the later
groups from that training step entirely. Padding within a single fixed-membership batch is unaffected by
the sequences' internal order — `create_batch()` always pads every member to the same shared max length
regardless of order — so "dynamic" batching could never have offered any real efficiency benefit at this
call site even before these two bugs. `TokenBatchLoader` still has zero callers anywhere in this codebase
(confirmed via grep, same as at TD-087), so neither defect has ever fired in production, but the existing
TD-087 regression tests (`NextBatchAndTargetBatchStayPaired` etc.) only check that a batch pair came from
the same `load_batch()` call, never that individual rows *within* one already-correctly-paired batch still
correspond to each other — so they did not, and could not, catch this.

Changes Made:
- `src/ParallelDataLoader.hpp`: `TokenBatchLoader::load_batch()` now builds both `input_batch` and
  `target_batch` unconditionally via `create_batch()` — no `create_dynamic_batches()` call, no
  `config_.use_dynamic_batching` branch — guaranteeing every sequence in the slice is kept and that index
  `k` means the same original sample in both batches by construction.

Verification:
- ✅ Added `InputAndTargetRowsWithinABatchStayAligned` to `tests/paralleldataloader_test.cpp`, using the
  existing `TokenBatchLoaderTest` fixture's dataset (input length driven by `i%10`, target length by
  `i%8`, and both encoding the same `i%26` letter in their repeated-character content) — asserts every row
  `k` of every batch has `input_char - 'A' == target_char - 'a'`, which only holds if row `k` in both
  batches still comes from the same original sample.
- ✅ Before/after regression: reverted `ParallelDataLoader.hpp` to its pre-fix `HEAD`, rebuilt
  `paralleldataloaderTests`, confirmed the new test fails with the exact predicted symptom (mismatched
  input/target rows across most of the batch — e.g. `input_char - 'A'` = 19 vs `target_char - 'a'` = 16);
  restored the fix, rebuilt, confirmed pass.
- ✅ Full `paralleldataloaderTests` suite (38/38) passes.

### TD-097: PROFILE_SCOPE Stopped Its Timer Immediately Instead of at Scope Exit

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `PROFILE_SCOPE` macro (`PerformanceProfiler.hpp`) | Replace the immediately-invoked lambda with a real RAII scope-guard whose destructor calls `stop()` |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/PerformanceProfiler.hpp`
end to end (372 lines). `PROFILE_SCOPE(profiler, name)` was meant to be the `Profiler`-based counterpart to
the file's own (correctly-implemented, immediately above it) RAII `ScopedTimer` class — start a timer on
entry, stop it automatically whenever the enclosing scope ends. Instead, its "guard" variable was bound to
the *return value* of an **immediately-invoked** lambda: `auto __profiler_guard_##name = [&](){
profiler.stop(name); return 0; }();` — the trailing `()` calls the lambda, and therefore `stop()`, right
there on that same line, back-to-back with the `start()` call one line above it, before a single
instruction of the "profiled" block that follows had executed. Every measurement taken through this macro
was ~0ms regardless of how much work the block actually did, silently defeating the entire point of a
scoped profiler. Separately, the guard variable's name was built by token-pasting `__profiler_guard_`
directly with the macro parameter (`__profiler_guard_##name`), which only produces a valid identifier when
`name` is itself a bare identifier token — the typical call shape shown in the file's own usage pattern (a
string-literal section name, e.g. `PROFILE_SCOPE(profiler, "encode")`) failed to compile at all, confirmed
by attempting exactly that call shape against the pre-fix macro. `PROFILE_SCOPE` has zero call sites
anywhere in this codebase today, so neither defect had ever been exercised.

Changes Made:
- `src/PerformanceProfiler.hpp`: added a minimal `adai_profiler_detail::ScopeGuard<F>` RAII template (calls
  a stored callable from its destructor) and `make_scope_guard()` factory, then rewrote `PROFILE_SCOPE` to
  construct one of these — via a `__LINE__`-based unique variable name, so a string-literal `name` no
  longer needs to be part of any identifier — with a lambda that calls `stop()`. The destructor now fires
  at actual scope exit instead of on the macro's own line.

Verification:
- ✅ Added `ProfileScopeMacroTimesTheWholeBlockNotJustItsOwnLine` to `tests/inference_optimization_test.cpp`
  (in the existing `PerformanceProfilerTest` fixture): wraps a large busy-loop in `PROFILE_SCOPE(profiler,
  "scoped_section")` and asserts the recorded `total_time` is well above a near-zero floor — a result only
  possible if `stop()` genuinely ran after the loop, not before it.
- ✅ Before/after regression: reverted `PerformanceProfiler.hpp` to its pre-fix `HEAD` and rebuilt
  `inferenceOptimizationTests` — confirmed the exact predicted compile failure (`pasting
  "__profiler_guard_" and ""scoped_section"" does not give a valid preprocessing token`); restored the fix,
  rebuilt, confirmed the new test compiles and passes.
- ✅ Full `inferenceOptimizationTests` suite (20/20, 1 pre-existing `DISABLED_` test unaffected) passes.

### TD-096: SYCL GPUMemory's Deferred Free Could Throw Out of a noexcept Destructor

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `adai::gpu::GPUMemory<T>::defer_free()` (SYCL backend) | Skip the deferred host_task submission when `GPUManager` is no longer initialized, instead of calling `get_queue()` unconditionally |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/gpu/sycl/GPUUtils_SYCL.hpp`
end to end (378 lines). `GPUMemory<T>::defer_free()` — called from `~GPUMemory()` and from the move-assignment
operator's release of the old resource — unconditionally called `GPUManager::get_queue()` to submit the
actual `sycl::free()` as a deferred `host_task`. `get_queue()` throws `std::runtime_error` when
`GPUManager` isn't initialized (e.g. after `GPUManager::cleanup()` — reachable via `Matrix::gpu_cleanup()`
— has already destroyed the queue). `~GPUMemory()` is implicitly `noexcept(true)` (its only members, a raw
pointer and a `size_t`, both have trivial non-throwing destructors), so any exception escaping from inside
it — including one thrown transitively from `defer_free()` — triggers `std::terminate()` immediately, not a
catchable `std::runtime_error`: any `GPUMemory`/`GPUMatrix` object still alive after `GPUManager::cleanup()`
runs would crash the whole process the instant it (or a move-assignment onto it) is destructed. The CUDA
backend's equivalent `GPUMemory::~GPUMemory()` has no analogous hazard — it calls `cudaFree()` directly
without going through `GPUManager` at all — so this was a SYCL-only gap between the two backends that are
otherwise meant to expose the same contract, the same "one backend fixed/hardened, the twin quietly wasn't"
shape this audit keeps finding (see TD-088, TD-090).

Currently unreachable in production: `Matrix::gpu_cleanup()` (the only caller of `GPUManager::cleanup()`
anywhere in this codebase) itself has zero callers — nothing in any shipped binary ever tears the GPU
subsystem down before process exit. But it is a real, sharp trap for the moment graceful GPU shutdown is
implemented (`ChatbotAPIServer.cpp` already carries a "Model State Persistence on Shutdown" TODO in the
same spirit) and any GPU-resident layer state happens to still be alive when it runs.

Changes Made:
- `src/gpu/sycl/GPUUtils_SYCL.hpp`: `defer_free()` now checks `GPUManager::is_available()` first and
  returns immediately if the subsystem is already torn down — there is nothing to submit a `host_task` to
  in that case, and the SYCL runtime reclaims device allocations when the context itself is destroyed.

Verification:
- ⚠️ Inspection-verified only, same caveat as TD-088's SYCL half: this sandboxed environment has no `icpx`
  (`cmake --preset=sycl` fails at the CMake `project()`/compiler-detection step before any compilation is
  attempted), so this fix could not be compiled or exercised. The change is a narrow, purely-defensive
  early return guarded by an existing, already-used predicate (`GPUManager::is_available()`), touching no
  other code path — no observable difference in behavior while `GPUManager` is initialized (the only case
  currently reachable in practice, since nothing calls `cleanup()`).

### TD-095: handle_chat_session() Never Reported a Newly-Created session_id, Breaking All Multi-Turn History

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `ChatbotAPI::handle_chat_session()` | Recover the internally-generated session id via the same reverse pointer-lookup `generate_batch_session_responses()` already uses, before writing the response |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `ChatbotCLI.cpp` end to end
and following its `/chat/session` request/response contract into `ChatbotAPI.cpp`. `ChatbotCLI::generate_response()`
only includes a `session_id` field in its request body once `session_id` is non-empty — the very first
message of every conversation sends none, exactly matching the documented "server allocates one, client
adopts it" pattern. Server-side, `get_or_create_session("")` correctly allocates a fresh internal id
(`create_session_id()`) and creates a new `Session` for it — but `handle_chat_session()` never recovered
that generated id anywhere: it kept using its own local `session_id` variable (parsed once from the
request, at that point still empty) all the way through to the response it writes, so
`{"success":true,"response":"...","session_id":""}` was returned on every single first message.
`ChatbotCLI`, and anything else following the same "start empty, adopt whatever the server returns"
convention, could then never learn the real id to send on the *next* message — so every message request
again arrived with an empty `session_id`, silently creating and immediately abandoning a brand-new session
each time. Multi-turn conversation history — the entire reason `/chat/session` exists, as opposed to the
stateless `/chat` endpoint — never actually accumulated for any client behaving this way.

This is the exact same "might be newly created" case `ChatbotAPI::generate_batch_session_responses()`
already solves correctly, just a few hundred lines away in the same file: it reverse-looks-up
`sessions_` by the `Session*` pointer returned from `get_or_create_session()` to recover the assigned id
whenever the caller passed one empty. `handle_chat_session()` — the *older*, singular-request sibling of
that batch method — never received the same fix, and had no direct unit test coverage to catch it: the
existing TD-063 JSON-escaping tests exercise `handle_chat_session()` only indirectly (it's a private
method, and C++ friendship doesn't propagate to the `TEST_F`-generated fixture subclass), and no live/
integration test for `chatbot_api_server` exists in this repository to have caught it at the HTTP level.

Changes Made:
- `src/ChatbotAPI.cpp`: `handle_chat_session()` now performs the same reverse pointer-lookup into
  `sessions_` (already proven correct in `generate_batch_session_responses()`) immediately after
  `get_or_create_session()`, populating `session_id` before it's used anywhere else in the function —
  including the final JSON response.

Verification:
- ✅ Added a `call_handle_chat_session()` friend-wrapper to the `ChatbotAPITest` fixture in
  `tests/chatbotapi_test.cpp` (matching the existing `call_generate_response()` pattern — `handle_chat_session()`
  is private and friendship doesn't propagate to `TEST_F` subclasses), plus two new tests:
  `HandleChatSession_NewConversationReturnsNonEmptySessionId` (a request with no `session_id` must get a
  non-empty one back) and `HandleChatSession_ReturnedSessionIdContinuesTheSameConversation` (sending that
  id back on a second message must return the identical id, proving it addresses the same live session).
- ✅ Before/after regression: reverted `ChatbotAPI.cpp` to its pre-fix `HEAD`, rebuilt `chatbotapiTests`,
  confirmed both new tests fail with the exact predicted symptom (`session_id` empty both times); restored
  the fix, rebuilt, confirmed pass.
- ✅ Full `chatbotapiTests` suite (48/48) passes.

### TD-094: CheckpointManager's Superseded "Best" Flag Never Cleared On Disk

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `CheckpointManager::save_checkpoint()` | Re-persist each superseded checkpoint's metadata when its `is_best` flag is cleared, not just the in-memory copy |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/CheckpointManager.hpp`
end to end (452 lines). When a new checkpoint becomes the best (lowest validation loss),
`save_checkpoint()` clears `is_best` on every other checkpoint — but only in the in-memory `checkpoints_`
vector; it never called `save_metadata()` again for the checkpoint(s) whose flag just changed, so their
`.meta` file on disk kept saying `is_best=true` forever. This was harmless as long as the process stayed
up (`rotate_checkpoints()` reads the in-memory flag, which was correctly cleared, and by construction only
one entry was ever the true in-memory best). But `load_existing_checkpoints()` — run at the top of every
`CheckpointManager` constructor, i.e. on every resumed training session, a normal and expected path for
this codebase's incremental training model — rebuilds `checkpoints_` straight from these `.meta` files and
trusts each one's `is_best` at face value. Every checkpoint ever marked best in *any* past session before
being superseded would come back as `is_best=true` on the next restart, and `rotate_checkpoints()` never
deletes an `is_best` checkpoint — so stale "best" checkpoints from earlier sessions became permanently
immune to rotation, accumulating on disk indefinitely and silently defeating the class's whole stated
purpose ("Automatic rotation (keep N best checkpoints)", "Automatic cleanup of old checkpoints"). A second,
related consequence: with multiple `is_best=true` entries reloaded, directory-iteration order (not epoch
order) would decide which one `load_existing_checkpoints()`'s tracking loop settled on as
`best_checkpoint_path_`/`best_validation_loss_`, so a genuinely worse checkpoint could even win over the
true best depending on filesystem enumeration order.

Changes Made:
- `src/CheckpointManager.hpp`: `save_checkpoint()`'s existing "mark previous best as not best" loop now
  calls `save_metadata(ckpt)` immediately after clearing each checkpoint's `is_best`, so disk and memory
  never diverge — including self-healing any already-corrupted directory that (pre-fix) accumulated
  multiple stale `is_best=true` files, since every one found still set is cleared and re-persisted here.

Verification:
- ✅ Added `SupersededBestIsClearedOnDisk` to `tests/checkpointmanager_test.cpp`: saves two checkpoints
  where the second supersedes the first, then reads the first's `.meta` file directly off disk and asserts
  it says `is_best=false`.
- ✅ Added `RestartDoesNotResurrectStaleBestFlags`, reproducing the real-world consequence: saves two
  checkpoints (with dummy `.bin` files, matching the existing `LoadExistingCheckpoints` test's pattern) in
  one `CheckpointManager` scope, destroys it (simulating process exit), constructs a fresh
  `CheckpointManager` over the same directory (simulating a resumed session), and asserts exactly one
  reloaded checkpoint is marked best and that it's the correct (lowest validation loss) one.
- ✅ Before/after regression: reverted `CheckpointManager.hpp` to its pre-fix `HEAD`, rebuilt
  `checkpointmanagerTests`, confirmed both new tests fail with the exact predicted symptom (epoch 0's
  on-disk metadata still reading `is_best=true`; the restarted manager reloading `best_count == 2` instead
  of 1); restored the fix, rebuilt, confirmed pass.
- ✅ Full `checkpointmanagerTests` suite (20/20) passes.

### TD-093: LLMEncoder/LLMDecoder's set_learning_rate() Never Reached Their Own Blocks

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `LLMDecoder::set_learning_rate()`, `LLMEncoder::set_learning_rate()` | Propagate the new rate to every sub-component (decoder/encoder blocks, token embedding, final norm) instead of just an unread top-level member |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/Decoder.cpp` end to end
(479 lines) — spotted while checking whether `update_weights(float learning_rate)`'s parameter was actually
used (it isn't; every callee is called with no arguments). `LLMDecoder::set_learning_rate(float lr)` set
only its own `learning_rate` member, which nothing else in the class reads: `token_embedding` and every
`DecoderBlock` kept whichever learning rate they were constructed with (0.001f), forever. This is the exact
opposite of `DecoderBlock::set_learning_rate()` (its own sub-component, one level down), which correctly
propagates to `self_attention`/`cross_attention`/`feed_forward`/`norm1-3` — `LLMDecoder` just never made the
one extra call needed to chain into it. Checking the sibling class turned up the identical defect:
`LLMEncoder::set_learning_rate()` propagated to `token_embedding` and `final_norm` but not to any
`encoder_blocks` entry, despite an inline comment claiming "propagation... happens through their
components" — untrue; `EncoderBlock` has no `set_learning_rate()` to delegate to, only a public
`learning_rate` member that its own `update_weights()` re-syncs to sub-components right before applying
gradients, and nothing ever wrote to that member from the encoder level.

Currently masked in the shipped trainer: `ChatbotTrainer` always constructs an `Optimizer` and calls
`model->register_parameters(*optimizer)` immediately afterward (`ChatbotTrainer.cpp:748`/`:1953`), and once
a component's `optimizer` pointer is set it permanently routes `update_weights()` through `optimizer->step()`
instead of the SGD-fallback branch that reads these `learning_rate` members — so today's training runs are
governed by `optimizer->set_learning_rate()` (called right next to `model->set_learning_rate()` in
`ChatbotTrainer.cpp:828-830`), not by this. But anything that calls `set_learning_rate()` expecting it to
behave like its own sub-component's identically-named method (or a future/alternate caller that trains
without an optimizer) would silently keep training every block at the constructor-default rate with no
error — the same class of silent, hard-to-notice divergence this whole audit keeps finding between two
things that are supposed to mirror each other.

Changes Made:
- `src/Decoder.cpp`: `LLMDecoder::set_learning_rate()` now also sets `token_embedding->learning_rate`,
  calls `block->set_learning_rate(lr)` for every decoder block, and sets `final_norm->learning_rate`.
- `src/LLMEncoder.cpp`: `LLMEncoder::set_learning_rate()` now also sets `block->learning_rate` directly for
  every encoder block (no `set_learning_rate()` exists on `EncoderBlock` to call instead), alongside its
  existing `token_embedding`/`final_norm` propagation.

Verification:
- ✅ Added `SetLearningRatePropagatesToAllSubComponents` to `tests/decoder_test.cpp` and
  `SetLearningRatePropagatesToEncoderBlocks` to `tests/llmencoder_test.cpp`, reading back
  `learning_rate` from every sub-component via existing `get_*()` accessors after calling
  `set_learning_rate()` with a value distinct from the 0.001f constructor default.
- ✅ Before/after regression: reverted both `Decoder.cpp` and `LLMEncoder.cpp` to their pre-fix `HEAD`,
  rebuilt `decoderTests`/`llmencoderTests`, confirmed both new tests fail with the exact predicted symptom
  (every sub-component reading back the 0.001f default instead of the newly-set rate); restored the fix,
  rebuilt, confirmed pass.
- ✅ Full `decoderTests` (48/48) and `llmencoderTests` (37/37) suites pass.

### TD-092: BatchedInferenceEngine Silently Ignored Every Per-Request Generation Config

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | `BatchedInferenceEngine::process_batch()` | Apply each request's own `gen_config` via `generator_->set_config()` before generating it, instead of delegating the whole batch to `generate_batch()` |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/BatchedInferenceEngine.hpp`
end to end (483 lines). `submit()`'s own doc comment advertises a `gen_config` parameter as "Optional
per-request generation config (uses default if not specified)", and `InferenceRequest` faithfully captures
it. But `process_batch()` never read `InferenceRequest::gen_config` at all — it extracted just the prompt
strings and handed them to `generator_->generate_batch(model_fn_, *tokenizer_, prompts)`, which internally
loops calling `generate_text()` using `generator_`'s own fixed member `config` (set once at construction
from `default_gen_config_` and never touched again). Two requests submitted with different
`strategy`/`temperature`/`max_length` always generated identically, silently using whichever config the
engine happened to be constructed with — the entire per-request override feature was a no-op. The existing
`SubmitWithExplicitGenConfig` test only checked that submitting with a `gen_config` didn't crash and
eventually completed; it never checked the config was actually applied, so it passed both before and after
this fix and did not catch the defect.

Changes Made:
- `src/BatchedInferenceEngine.hpp`: `process_batch()` now loops over `batch` directly, calling
  `generator_->set_config(req.gen_config)` immediately before `generator_->generate_text(model_fn_,
  *tokenizer_, req.prompt)` for each request — the same sequential-generation shape `generate_batch()` had
  internally, now actually honoring what each caller asked for. Restores `default_gen_config_` on
  `generator_` after the loop so the engine's own default is what's active between batches.

Verification:
- ✅ Added `SubmitWithExplicitGenConfigAppliesPerRequestMaxLength` to
  `tests/batchedinferenceengine_test.cpp`: builds a small real vocab via `build_vocab()`, uses a model_fn
  that always predicts the same non-stop-token id (so only `max_length` can end generation), submits one
  request with the engine's default `max_length=2` and another with an explicit per-request
  `max_length=15`, and asserts the second produces strictly more decoded output than the first.
- ✅ Before/after regression: reverted only `BatchedInferenceEngine.hpp` to its pre-fix `HEAD`, rebuilt
  `batchedinferenceengineTests`, confirmed the new test fails with the exact predicted symptom (both
  results identical — `"startthe"` vs `"startthe"` — since both silently used the engine's default
  `max_length=2` instead of the override's 15); restored the fix, rebuilt, confirmed pass.
- ✅ Full `batchedinferenceengineTests` suite (52/52) passes.

### TD-091: 19 Config Keys Were Settable via the File but Had No Environment-Variable Override

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 10, 2026 | Config (`ConfigLoader::load_from_env()`) | Add the 19 missing `get_env*()` calls, mirroring their existing `load_from_file()` entries |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/Config.cpp` end to end
(1118 lines, before this fix). CLAUDE.md documents a single, uniform loading precedence for every
configuration key: "env vars → config file → hardcoded defaults." `ConfigLoader::load_from_file()` and
`ConfigLoader::load_from_env()` are two independent, hand-maintained `if (key == "X") / if (auto val =
get_env("X"))` chains that are supposed to mirror each other key-for-key — and 19 keys had drifted out of
sync, present in the file-parsing chain but entirely absent from the environment-variable chain: `ENABLE_
METRICS_SERVICE`, `METRICS_SERVER_URL`, `METRICS_PUSH_TIMEOUT_MS`, `METRICS_HEARTBEAT_INTERVAL_MS`,
`METRICS_ENABLE_PERSISTENCE`, `METRICS_FILE`, `METRICS_SUMMARY_FILE`, `METRICS_PERSIST_EVERY_SAMPLES`,
`METRICS_PERSIST_EVERY_SECONDS`, `METRICS_MAX_RECORDS_IN_MEMORY`, `METRICS_MAX_RECORDS_ON_DISK`,
`METRICS_ENABLE_PROMETHEUS`, `METRICS_PROMETHEUS_FILE`, `METRICS_API_PORT`, `METRICS_API_ALLOW_CONTROL`,
`ENABLE_GENERATION_QUALITY_METRICS`, `GENERATION_QUALITY_SAMPLE_SIZE`, `GENERATION_QUALITY_MAX_TOKENS`,
and `GENERATION_QUALITY_ASYNC_THRESHOLD`. Confirmed via a scripted diff of every `key == "..."` literal in
`load_from_file()` against every `get_env*("...")`/`getenv("...")` literal in `load_from_env()` — this set
was the entire symmetric difference (nothing was missing in the other direction).

Two of these — `METRICS_SERVER_URL` and `METRICS_HEARTBEAT_INTERVAL_MS` — are documented by name in
CLAUDE.md's own "Configuration" table as significant settings, so an operator setting either as an
environment variable (e.g. in a systemd unit's `Environment=` line, or a container's env block) while a
config file also set a value would find the environment variable **silently ignored** — `load()` calls
`load_from_file()` then `load_from_env()` expecting the latter to win, but for these 19 keys
`load_from_env()` never even checked whether the variable was set, so the file's value (or the struct
default, if no file value either) always won regardless. This is exactly the class of defect this whole
audit has repeatedly found in other files — two hand-maintained lists that must mirror each other,
silently drifting apart as keys were added to one but not the other over time.

Changes Made:
- `src/Config.cpp`: added the 19 missing `get_env()`/`get_env_int()`/`get_env_bool()` calls to
  `load_from_env()`, matching each key's type and struct field from its existing `load_from_file()` entry.
  Verified the fix closes the gap completely by re-running the same key-extraction diff script — empty
  result in both directions afterward.

Verification:
- ✅ Added `LoadTrainingMetricsServiceKeysFromEnvironmentVariables` and
  `TrainingMetricsServiceEnvironmentVariablesOverrideFile` to `tests/config_test.cpp`, following the
  existing `LoadMultiInstanceMetricsFromEnvironmentVariables`/`EnvironmentVariablesOverrideFile` patterns.
  The first test deliberately sets every asserted value to differ from `ServiceConfig`'s own struct
  default (`enable_metrics_service` defaults `true`, `metrics_server_url` defaults to
  `"http://localhost:8081"`) — an initial version of this test picked values that happened to match those
  defaults and silently passed against the pre-fix code for 2 of its 3 assertions, since a value equal to
  the untouched default is indistinguishable from a successfully-applied override; only the corrected
  version (differing values) reliably catches the regression.
- ✅ Before/after regression: reverted only `Config.cpp` to its pre-fix `HEAD`, rebuilt `configTests`,
  confirmed both new tests fail with the exact predicted symptom (all three fields read back as their
  struct defaults instead of the env-set values; the override test's `metrics_server_url` read back as the
  file's value rather than the env var's); restored the fix, rebuilt, confirmed pass.
- ✅ Full `configTests` suite (62/62) passes.

Files Changed:

- `src/Config.cpp`
- `tests/config_test.cpp`

---

### TD-090: SQLiteMetricsDatabase Read a NULL best_validation_loss as 0.0 Instead of "No Data"

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | SQLiteMetricsDatabase (`list_sessions()`, `get_session()`) | Check `sqlite3_column_type() == SQLITE_NULL` before reading, defaulting to `std::numeric_limits<float>::max()` |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/SQLiteMetricsDatabase.cpp`
end to end (764 lines) — the default (and only bundled) metrics database backend. `sqlite3_column_double()`
returns `0.0` for a SQL NULL column, indistinguishable from a genuine value of `0.0` — unlike libpq's
`PQgetvalue()`/`PQgetisnull()` split, which `PostgresMetricsDatabase.cpp`'s `pg_opt_float()` already guards
against for exactly this column (TD-065). `best_validation_loss` is a "no data yet" sentinel everywhere
else in the codebase — `SessionRecord`'s own struct default, `TrainingMetricsSnapshot`,
`MetricsSessionSummary`, `ChatbotTrainer`, `CheckpointManager`, and `MetricsTracker` all initialize it to
`std::numeric_limits<float>::max()`, never `0` — yet `list_sessions()`/`get_session()` read it via a bare
`sqlite3_column_double()`, so a row with `best_validation_loss IS NULL` silently came back as `0.0`: a
completed session that never ran validation would report a suspiciously *perfect* validation loss instead
of "no data". `list_sessions()` explicitly supplements the live dashboard with completed/evicted sessions
read from this exact path (`MetricsSessionRegistry::list_sessions()`, `summary.best_validation_loss =
rec.best_validation_loss;`) — the same function whose neighboring `current_loss`/`current_validation_loss`
assignment carries a comment explicitly calling out the analogous risk for a *different* pair of fields
("prevents this from silently reading as 0"), showing the codebase is already alert to this exact failure
mode, just not applied here.

In practice every current C++ write path (`upsert_session()`) always binds a real float for this column
(the struct's own default, if never explicitly set, still binds `FLT_MAX` as a genuine value — never
`NULL`), so a NULL row would only arise from an older code version's INSERT statement, a hand-edited
database, or any other writer that didn't populate the column — the same class of "pre-existing database"
scenario the neighboring, already-existing `MigratesPreExistingDatabaseMissingFinalLossColumns` test
deliberately simulates for a different (missing-entirely) column.

Changes Made:
- `src/SQLiteMetricsDatabase.cpp`: added a `sqlite_opt_float()` helper (checks `sqlite3_column_type() ==
  SQLITE_NULL` before falling back to `sqlite3_column_double()`, mirroring the Postgres backend's
  `pg_opt_float()`) and used it for `best_validation_loss` in both `list_sessions()` and `get_session()`,
  defaulting to `std::numeric_limits<float>::max()`. `final_loss`/`final_validation_loss` were left as
  plain `sqlite3_column_double()` reads — their correct NULL fallback really is `0.0` (matching
  `pg_opt_float(..., 0.0f)` on the Postgres side), so no behavior change was needed there.

Verification:
- ✅ Added `BestValidationLossNullReadsAsSentinelNotZero` to `tests/MetricsDatabaseTest.cpp`, following the
  same "hand-write an old-shaped schema/row, bypassing `SQLiteMetricsDatabase` entirely" pattern as the
  neighboring `MigratesPreExistingDatabaseMissingFinalLossColumns` test: inserts a session row with
  `best_validation_loss` omitted from the column list (NULL), then asserts both `get_session()` and
  `list_sessions()` read it back as `std::numeric_limits<float>::max()`.
- ✅ Before/after regression: reverted only `SQLiteMetricsDatabase.cpp` to its pre-fix `HEAD`, rebuilt
  `metricsDatabaseTests`, confirmed the new test fails with the exact predicted symptom (`0` instead of
  `3.4028235e+38` at both call sites); restored the fix, rebuilt, confirmed pass.
- ✅ Full `metricsDatabaseTests` suite (26/26) passes.

Files Changed:

- `src/SQLiteMetricsDatabase.cpp`
- `tests/MetricsDatabaseTest.cpp`

---

### TD-089: ChatbotAPI's Batch Endpoints Returned Responses Misordered Relative to the Request

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | ChatbotAPI (`generate_batch_responses()`, used by `POST /chat/batch` and `/chat/batch-session`) | Generate responses by iterating the original input list, not the length-sorted batches |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/ChatbotAPI.cpp` end
to end (836 lines) — this is the real, shipped `chatbot_api_server` binary's request handler (confirmed
via `src/ChatbotAPIServer.cpp`, not a dead/unused class). `generate_batch_responses()` called
`create_dynamic_batches()` (`src/BatchProcessor.hpp`) to group the request's messages into padding-
efficient batches, then iterated those batches' contents to build the response list. `create_dynamic_batches()`
sorts sequences by token length internally for padding efficiency (its whole point, and correct for its
original training-data-loading callers, where nothing downstream cares which physical position a sample
came from) and its output `TokenBatch`es carry no memory of each sequence's original index. For a batch
request whose messages have different lengths, this meant `response.responses[i]` was not necessarily the
answer to `inputs[i]` — the API silently returned answers in length-sorted order instead of request order.
This is a correctness bug affecting every real caller of `POST /chat/batch`, and it also broke
`generate_batch_session_responses()` (`/chat/batch-session`): that function maps
`temp_response.responses[i]` back to `actual_session_ids[i]` to append the assistant's reply into the
*correct* session's conversation history, so a misordered `responses` list could append one user's session
with a different, unrelated user's session's actual response.

The bug was invisible to the existing test suite: `GenerateBatchResponses_VariableLengths` already used
inputs of differing lengths, but only asserted `response.responses.size() == 4` — never checked that each
response actually corresponded to its input. Confirmed the "batching" itself provides no real compute
benefit here to give up by fixing this: each item is still generated one row at a time via
`model_->forward(...)` inside the per-batch loop, not through any actual batched Matrix op — so
`create_dynamic_batches()`'s output was being used purely to compute the reported padding-efficiency
statistics, an order-independent aggregate, and had no reason to also dictate response order.

Changes Made:
- `src/ChatbotAPI.cpp`: `generate_batch_responses()` still calls `create_dynamic_batches()` to compute
  `batch_response.stats` (unaffected by internal reordering, since those are aggregate statistics), but
  now generates and appends responses by iterating `input_token_sequences` directly, in the caller's
  original order, instead of iterating the length-sorted `batches`.

Verification:
- ✅ Added `GenerateBatchResponses_PreservesInputOrder` to `tests/chatbotapi_test.cpp`: generates a
  reference response for each input individually (via `generate_response()`, deterministic under
  "greedy"), then asserts `generate_batch_responses()`'s output matches at every index — using inputs
  deliberately *not* in length-sorted order (an already-ascending-length list would "reorder" into the
  same order it started in and wouldn't catch a regression, which is exactly what a first attempt at this
  test — reusing the existing `VariableLengths` test's already-ascending-by-length inputs — silently
  failed to catch, since `create_dynamic_batches()`'s sort produced the same order as the input in that
  case). `generate_response()` is private, so the fixture gained a small `call_generate_response()`
  wrapper (`ChatbotAPI` friends `ChatbotAPITest` specifically, and that friendship doesn't extend to the
  subclasses gtest's `TEST_F` macro generates).
- ✅ Before/after regression: reverted only `ChatbotAPI.cpp` to its pre-fix `HEAD`, rebuilt
  `chatbotapiTests`, confirmed the new test fails with the exact predicted symptom — each
  `response.responses[i]` equal to a *different* index's expected value, a textbook reordering signature
  (e.g. `expected[1]` appeared at `response.responses[0]`); restored the fix, rebuilt, confirmed pass.
- ✅ Full `chatbotapiTests` suite (46/46) passes.

Files Changed:

- `src/ChatbotAPI.cpp`
- `tests/chatbotapi_test.cpp`

---

### TD-088: matrix_sum_gpu() Recurses Forever on an Empty (size == 0) Input — Both GPU Backends

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | GPU (CUDA `MatrixGPU.cu` and SYCL `MatrixGPU_SYCL.cpp`, both `matrix_sum_gpu()`) | Early-return 0.0f for `size <= 0` before computing the group/block count |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/gpu/sycl/
MatrixGPU_SYCL.cpp` end to end (874 lines). `matrix_sum_gpu(data, size)` reduces `size` elements to a
scalar via a two-level reduction: divide into `num_groups = (size + WG_SIZE - 1) / WG_SIZE` work-groups,
reduce each to a partial sum, then — unless `num_groups == 1` — recurse on the `num_groups` partial sums
(`return matrix_sum_gpu(group_sums.get(), num_groups);`). For `size == 0` (an empty tensor — e.g. a
degenerate zero-length batch or sequence reaching `.sum()`/`.mean()`), `num_groups` computes to `0`: not
the `num_groups == 1` base case, and not a value that ever shrinks on the recursive call either —
`matrix_sum_gpu(ptr, 0)` calls itself with the exact same argument forever, unwinding the stack.
`src/gpu/MatrixGPU.cu`'s CUDA implementation has the byte-for-byte identical structure and the identical
bug (`blocks = (size + threads - 1) / threads` → `0` for `size == 0`, same non-terminating recursion) —
this is a pre-existing defect present in both backends equally, not something introduced by one being a
port of the other. `matrix_count_below_threshold_gpu()` (SYCL) / its CUDA counterpart share the exact same
`num_groups`-from-`size` shape and recurse into `matrix_sum_gpu()` for their own final reduction, so they
inherit the fix transitively rather than needing their own guard.

Neither GPU backend can be exercised end-to-end in this environment (no CUDA device present, and no
`icpx`/oneAPI toolchain installed for SYCL — `cmake --preset=sycl` fails at configure with `icpx` not
found), and reachability would require tracing every `.sum()`/`.mean()` GPU call site across the encoder/
decoder/attention/loss code for a genuinely zero-sized tensor, which no specific call site was confirmed
to hit today. The fix is applied as a correctness guard against a real, analytically-confirmed defect
rather than as a response to an observed crash.

Changes Made:
- `src/gpu/sycl/MatrixGPU_SYCL.cpp` and `src/gpu/MatrixGPU.cu`: both `matrix_sum_gpu()` implementations
  now return `0.0f` immediately when `size <= 0`, before computing `num_groups`/`blocks` or allocating the
  group-sums buffer.

Verification:
- ✅ **CUDA side fully compile-verified**: this environment has `nvcc` (CUDA 12.0) even without a physical
  GPU device. Configured `cmake --preset=gpu` and built `adai_core` (which compiles `MatrixGPU.cu`) from
  scratch — compiled clean with the fix in place, confirming the change is syntactically and
  type-correct C++/CUDA. No physical GPU is available to actually execute the kernel, so this is
  compile-verification, not a full before/after runtime regression test.
- ⚠️ **SYCL side is inspection-verified only.** `icpx`/oneAPI is not installed in this environment
  (confirmed: `cmake --preset=sycl` fails at the configure step, before any compilation is attempted), so
  `MatrixGPU_SYCL.cpp`'s copy of the fix could not be compiled or executed here. The change applies the
  identical guard pattern proven to compile correctly on the CUDA side (a plain early-return with no
  SYCL-specific syntax), so confidence is high, but this should be spot-checked by someone with SYCL
  toolchain access before being treated as fully verified.

Files Changed:

- `src/gpu/sycl/MatrixGPU_SYCL.cpp`
- `src/gpu/MatrixGPU.cu`

---

### TD-087: TokenBatchLoader's Two Independent Prefetch Queues Could Deadlock Every Worker Thread (and Mismatch Input/Target Pairs)

| Resolution Date | Component | Resolved By |
|-----------------|-----------|-------------|
| September 9, 2026 | ParallelDataLoader.hpp (`TokenBatchLoader`) | Push/pop input and target batches as one pair through a single queue instead of two independent queues |

Summary:
Found during the second, independent full-codebase re-audit while re-reading `src/ParallelDataLoader.hpp`
end to end (907 lines) — prompted in part by the file's own still-open TD-064 (`paralleldataloaderTests`
hung indefinitely under a full-suite `ctest -j8` run, root cause never found despite extensive TSan/manual
review). `TokenBatchLoader` (a second, distinct class in the same file, alongside the already-investigated
`ParallelDataLoader`) has `config_.load_targets` support that — unlike `ParallelDataLoader`, which has only
one queue — routes input and target batches through **two independent** `ThreadSafeBatchQueue`s. Each
worker thread's loop pushed to `batch_queue_` and then, immediately after, to `target_queue_`, both
bounded to the same capacity (`num_workers * prefetch_factor`).

This has two distinct bugs, both stemming from the same root design flaw (splitting one produced item
across two independently-bounded queues with no coordination between them):

1. **Deadlock.** Any caller that drains `next_batch()` without also draining `next_target_batch()` at a
   matching rate (e.g. it doesn't need targets and simply never calls it, or calls it less often) fills
   `target_queue_` to capacity. Every worker thread then blocks forever inside `target_queue_->push()` —
   which can never unblock, since nothing ever pops from `target_queue_`. Once *every* worker thread is
   stuck there, `batch_queue_` stops being refilled too, so subsequent `next_batch()` calls block forever
   as well: the entire loader wedges, both queues' condition variables parked in `futex_wait` with 0% CPU
   — the exact forensic signature TD-064 recorded ("both of its threads were blocked in `futex_wait_queue`
   with 0% CPU usage"), though `TokenBatchLoader` itself is confirmed unrelated to that specific incident
   (see below).
2. **Mismatched pairs.** Even when a caller *does* call both accessors in lockstep, with `num_workers > 1`
   there is no guarantee that the two queues receive pushes from different worker threads in the same
   relative order — worker A could push its input, then worker B interleaves its own input+target pushes,
   before worker A gets to push its matching target. A caller alternating `next_batch()`/
   `next_target_batch()` could silently receive an input from one loaded batch paired with the target
   from a *different* one.

`TokenBatchLoader` has zero current callers anywhere in the codebase (confirmed via `grep -rn` across
`src/`/`tests/` before this fix — only its own definition referenced it) and, unlike `ParallelDataLoader`,
had no dedicated test coverage at all, so neither bug had ever actually fired; the class is tagged
`@adai-status: experimental`, already capped by TD-052 (fake char-code tokenization) for a separate reason.
This is **not** a root-cause finding for TD-064: `paralleldataloaderTests` (the test binary in the original
incident) never instantiates `TokenBatchLoader` — only `ParallelDataLoader`/`ThreadSafeBatchQueue`, which
use exactly one queue and don't have this specific two-queue interaction at all. TD-064 remains open;
re-reading `ParallelDataLoader`'s own single-queue synchronization here (again) found nothing beyond what
that investigation already documented (predicate-based `condition_variable::wait()` is correctly immune to
the shutdown-flag race one might otherwise suspect, since the predicate itself is checked under the lock
regardless of notification timing).

Changes Made:
- `src/ParallelDataLoader.hpp`: `TokenBatchLoader` now holds one
  `ThreadSafeBatchQueue<std::pair<TokenBatch, TokenBatch>>` instead of two independent queues. The worker
  thread pushes the input/target pair as a single atomic queue operation. `next_batch()` pops the pair,
  returns the input half, and latches the target half into a new `pending_target_` member for the
  immediately-following `next_target_batch()` call to retrieve. `stop()`/`new_epoch()` updated to match
  (one queue to shut down/clear, plus resetting `pending_target_`).

Verification:
- ✅ Standalone reproduction (`std::async` consumer draining only `next_batch()`, watchdog-timed) confirmed
  the deadlock against the pre-fix two-queue design in isolation, mirroring the real class's structure.
- ✅ Added three tests to `tests/paralleldataloader_test.cpp`: `NextBatchDoesNotDeadlockWhenTargetsNeverDrained`
  (drains a full epoch via `next_batch()` alone, asserting via `std::async` + `wait_for(10s)` that it
  completes rather than hanging — matching the timeout-guarded-async pattern already used in
  `batchedinferenceengine_test.cpp` so a regression fails loudly instead of hanging the test binary),
  `NextBatchAndTargetBatchStayPaired` (3 workers, asserts every lockstep-drawn pair is non-empty — the
  structural guarantee from sharing one queue, not a statistical one), and
  `NoTargetsConfiguredReturnsNulloptForTargetBatch` (unchanged-behavior baseline).
- ✅ Before/after regression: reverted only `ParallelDataLoader.hpp` to its pre-fix `HEAD`, rebuilt
  `paralleldataloaderTests`, confirmed `NextBatchDoesNotDeadlockWhenTargetsNeverDrained` hangs (killed by
  an external `timeout`, exactly as predicted); restored the fix, rebuilt, confirmed all 3 new tests pass.
- ✅ Full `paralleldataloaderTests` suite (37/37) run 3 times back-to-back with no flakiness, including all
  pre-existing `ParallelDataLoader`/`ThreadSafeBatchQueue` tests (untouched by this change) still passing.

Files Changed:

- `src/ParallelDataLoader.hpp`
- `tests/paralleldataloader_test.cpp`

---

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

