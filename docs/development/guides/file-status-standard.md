# Per-File Status & Versioning Standard

This standard gives every source file its own **readiness status** and **independent version
number**, so "is this production-ready?" has a single, enforceable answer per file instead of a
prose claim in a summary doc that quietly goes stale.

It complements, and does not replace, [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) — a file's status
is often *capped* by an open TD item; see [Relationship to Technical Debt](#relationship-to-technical-debt).

## Why this exists

`docs/development/reference/chatbot-completeness.md` marks `MultiHeadAttention`'s KV-cache path
"✅ Production-ready," but [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) TD-050 documents that the same
cache produces incorrect results for greedy decoding. Neither doc is wrong for its moment — the
completeness doc just has no mechanism to notice when the ground shifts under it. A tag that lives
in the file itself, is checked by CI, and is cheap to keep current doesn't have that failure mode.

## The tag

Every in-scope file carries a three-line comment block, in the file's native comment syntax,
within the first ~40 lines (after license/`#pragma once`/shebang, before real code):

```cpp
// @adai-status: beta
// @adai-version: 1.2.0
// @adai-reviewed: 2026-09-07
```

```python
# @adai-status: experimental
# @adai-version: 0.3.1
# @adai-reviewed: 2026-08-01
```

Same two-space-then-`#`/`//` form for `.sh`, `.kt`, `.java`, `.js`. (XML and other non-code assets
are out of scope — see [Scope](#scope).)

When a status is held down by a specific known gap, say so on the status line itself:

```cpp
// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md)
```

### `@adai-status`

| Status | Meaning | Callers may assume |
|---|---|---|
| `experimental` | Prototype / proof of concept. | Nothing. Interface and behavior can change or disappear without notice. Default for new files. |
| `beta` | Feature-complete for its intended scope, has test coverage. | It works for the paths its tests cover; known gaps exist (cross-reference the TD item if one exists). |
| `stable` | Production-ready. | Tested, documented, no known correctness gap that would surprise a caller. Safe to build on and deploy. |
| `deprecated` | Superseded. | Don't use in new code. The tag's trailing note must name the replacement. |
| `legacy` | Retained for reference/compatibility only (everything under `legacy/`). | Not maintained, not to be extended. |

### `@adai-version`

Plain per-file SemVer (`MAJOR.MINOR.PATCH`), independent of the project's overall release version
and independent of every other file's. Starts at `0.1.0`.

- **PATCH** — internal fix or refactor, no visible interface/behavior change.
- **MINOR** — backward-compatible addition (new function, new optional parameter, new capability).
- **MAJOR** — breaking change to the file's public interface or behavior.

**`stable` requires `MAJOR ≥ 1`.** Crossing `0.x → 1.0.0` *is* the graduation event and should land
in the same commit that flips `@adai-status` to `stable`. A later MAJOR bump on an already-`stable`
file doesn't auto-demote it, but it does require re-review before the next release — a breaking
change is exactly the kind of thing that can reintroduce the gap `stable` promises isn't there, so
update `@adai-reviewed` deliberately, not reflexively. `deprecated`/`legacy` files keep whatever
version they had when they left active development.

### `@adai-reviewed`

ISO date (`YYYY-MM-DD`) of the last time a human deliberately confirmed the status/version tag
still describes reality — not merely the last time the file was edited. `stable` files with a
`@adai-reviewed` date older than 6 months are flagged by the report generator as needing
re-attestation, because "production-ready" is a claim that rots.

## Scope

Required on:

- `src/**/*.{cpp,hpp,h,cu}`
- `android/**/src/**/*.{kt,java}`
- `tizen-metrics-app/js/*.js`
- Operational scripts in `scripts/*.sh` / `scripts/*.py` (install/deploy/service scripts — not
  one-off analysis throwaways; tag `experimental` if genuinely unsure)

Out of scope (don't add the tag):

- `build*/`, `dist-windows/`, `external/` (vendored third-party code)
- `legacy/` — bulk-tagged `legacy` once (see [Rollout](#rollout)) rather than reviewed file by file
- `tests/**` and Android's `src/test/**` / `src/androidTest/**` — a test file doesn't carry its
  own readiness status; its pass/fail state is an *input* to the status of the file(s) it covers,
  not a subject of this standard itself
- Generated/vendored JSON, XML (including Android manifests/layouts), and `docs/**`

## Tooling

Two scripts live in `scripts/`:

- **`check_file_status.py`** — validates the tag on in-scope files: status is one of the five
  values, version matches SemVer, `stable` implies `MAJOR ≥ 1`, reviewed date parses and isn't in
  the future, `deprecated` names a replacement. Two modes:
  - `--changed [base-ref]` — only files touched relative to `base-ref` (default `origin/main`).
    This is the PR-gating mode.
  - (no flag) — whole repo, for local use and by the report generator.

  Exits nonzero only with `--strict`; without it, it prints a summary and always exits 0. Run it
  in warn-only mode (no `--strict`) in CI until the [rollout](#rollout) finishes, then switch the
  `--changed` check to `--strict` as a required PR check.

- **`gen_status_report.py`** — runs the same scan and (re)writes
  `docs/development/PRODUCTION_READINESS.md`: counts and percentages by status, grouped by
  top-level component (`src`, `android`, `tizen-metrics-app`, `scripts`), the list of
  `experimental`/`beta` files with any TD-### cross-reference, and the list of `stable` files
  overdue for re-attestation. The generated file says so at the top — hand edits to it get
  clobbered on the next run.

```bash
./scripts/check_file_status.py --changed          # what a PR touches, warn-only
./scripts/check_file_status.py --changed --strict # same, fails CI
./scripts/gen_status_report.py                    # regenerate the dashboard
```

## Relationship to Technical Debt

[TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) tracks *why* something isn't done; this standard tracks
*whether a given file is safe to rely on right now*. They cross-reference, they don't merge:

- A file capped below `stable` by a known, tracked gap should say so inline (`capped by TD-050`).
- Resolving a TD item is a natural trigger to re-review and possibly promote the file's status —
  but promotion still requires an explicit `@adai-status`/`@adai-version` edit; closing the TD item
  alone doesn't change the tag.
- Not every `experimental`/`beta` file has an open TD item — plenty of things are simply new and
  unreviewed. Don't manufacture a TD entry just to satisfy this cross-reference.

## Rollout

**Status: complete as of September 7, 2026.** All 279 in-scope files carry a verified tag — no
file was defaulted to `stable` without concrete evidence (dedicated tests, no open TD cap, actual
use by a deployed binary); most of the tree landed at `beta` or `experimental` on exactly that
basis. See [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md) for the current breakdown.

What actually happened, in order:

1. This doc + both scripts landed; `check_file_status.py` ran warn-only in CI.
2. Every in-scope file was tagged by hand, component by component (`src/`, `scripts/`, `android/`,
   `tizen-metrics-app/js/`), cross-checking test coverage, TD-tracker currency (several tags in
   `TECHNICAL_DEBT.md` turned out to be stale — see TD-018/TD-020's resolution history in the
   [archive](../archive/TECHNICAL_DEBT_RESOLVED.md)), and reading the riskiest files directly rather
   than trusting heuristics alone (e.g. `PPOOptimizer.hpp`'s core update loop turned out to be a
   placeholder, not real PPO).
3. The whole-repo check now passes `--strict` with zero problems, so the **PR-gating check is
   live**: `file-status-check` in `.github/workflows/ci.yml` runs
   `check_file_status.py --changed --strict` on every push and pull request — any in-scope file a
   change touches must carry a current, valid tag or CI fails.
4. `docs/development/reference/chatbot-completeness.md` was not retired outright (it has historical
   value as a point-in-time snapshot) but was marked stale in place, pointing readers at
   `PRODUCTION_READINESS.md` as the authoritative source going forward.

Going forward, keeping the tree tagged is now enforced rather than aspirational: a file's tag gets
checked every time someone touches it, by whoever is already in the file and best placed to know
its real status — not re-audited wholesale on some future date.

## Checklist additions

- **CONTRIBUTING.md**: "File status tag (`@adai-status`/`@adai-version`/`@adai-reviewed`) added or
  updated for every new or touched in-scope file."
- **New component** (per [CLAUDE.md](../../../CLAUDE.md)'s existing convention of
  `src/Component.{cpp,hpp}` + `tests/component_test.cpp` + CMake registration): new files start at
  `@adai-status: experimental`, `@adai-version: 0.1.0`.
