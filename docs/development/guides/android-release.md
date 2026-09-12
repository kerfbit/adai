# Android Release & Versioning Standard

This document defines how `:app` and `:opsdashboard` are versioned and released. It exists
because TD-047 found the opposite of a process: both apps have declared `versionName = "0.1.0"`
since they first landed, with no tagging, no release artifacts, and nothing enforcing that a
"release" actually corresponds to what's in the tagged commit.

The [`android-release.yml`](../../../.github/workflows/android-release.yml) workflow enforces the
rules below mechanically — it's not possible to publish a release that violates them. Treat this
doc as the spec and that workflow as the test suite for it.

## Scope

Covers `:app` and `:opsdashboard` only — the two applications TD-047 itself scopes in, and the
only two `android-ci.yml` currently builds as APKs. `:wearsync` is a library module with no
`versionName` of its own; it ships as part of whichever app depends on it. `:wearface` has a
`versionName` but is never installed directly — `:opsdashboard` bundles its built, resigned APK as
a raw asset (`copyWearFaceApk`/`resignWffApk`) and pushes it via Watch Face Push, so its version
isn't independently released. `:wearcomplications` is a real, independent application but isn't
mentioned in TD-047 and has no release process defined here yet — a natural follow-up once it's in
scope.

## Versioning scheme

Each of `:app` and `:opsdashboard` has its own independent version, the same "per-unit SemVer"
philosophy as this repo's [`@adai-version` file-status
standard](file-status-standard.md#adai-version), just applied at the whole-app level instead of
per-file:

- **`versionName`** (`android:versionName`, the human-facing string): plain SemVer
  `MAJOR.MINOR.PATCH`.
  - **PATCH** — bug fix, no user-visible behavior change.
  - **MINOR** — backward-compatible new feature or capability.
  - **MAJOR** — breaking change from the user's or the server's perspective (a data migration,
    dropping support for something, a server API contract change this version now requires).
- **`versionCode`** (`android:versionCode`, the integer Play Store/Android itself use to compare
  builds): a plain integer, **strictly increasing, per app, forever** — never reused, never
  decreased, even across major version bumps or reinstalls. Bump it by 1 on every single release.

`versionName` and `versionCode` are independent of each other and independent of the other app's —
`:app` reaching `1.4.0`/`versionCode 12` says nothing about what `:opsdashboard` is at, the same
way two files' `@adai-version` tags don't track each other.

**Current state:** both apps are still at `versionName = "0.1.0"`, `versionCode = 1` — neither has
had a real release yet. The first one will be the first real exercise of this process.

## Release process

1. In a normal PR, bump `versionName` **and** `versionCode` in the app's `android/<module>/
   build.gradle.kts` (`<module>` is `app` or `opsdashboard`). Nothing else is required to "cut a
   release" — nothing about the app is different from an ordinary commit until you also do step 3.
2. Merge to `main`. [`android-ci.yml`](../../../.github/workflows/android-ci.yml) builds and tests
   it like any other change — confirm it's green before tagging.
3. Tag the merged commit and push the tag:
   ```bash
   git tag app-v1.4.0            # or opsdashboard-v1.4.0
   git push origin app-v1.4.0
   ```
   Pushing the tag is the only trigger — nothing else about tagging a commit does anything by
   itself.
4. [`android-release.yml`](../../../.github/workflows/android-release.yml) runs automatically.
   It **validates the release before building or publishing anything** — see the next section for
   exactly what it checks and why. If a check fails, the workflow fails loudly and no GitHub
   Release is created; fix the mismatch (usually: bump the version you forgot, or re-tag with the
   version you meant) and push a corrected tag.
5. Once green, check the run's own GitHub Release page for the built APKs.

Tags follow **`app-vMAJOR.MINOR.PATCH`** / **`opsdashboard-vMAJOR.MINOR.PATCH`** exactly (matched
by `^(app|opsdashboard)-v[0-9]+\.[0-9]+\.[0-9]+$`) — deliberately not this repo's existing
`v*.*.*` scheme, which [`release.yml`](../../../.github/workflows/release.yml) already owns for
the C++ binaries. Two independently-versioned Android apps in the same repo need their own,
disambiguated tag namespace; a bare `v1.4.0` would be ambiguous between three different release
processes.

## What the release workflow validates, and what each check actually catches

Every check below corresponds to a real, easy mistake in a manual release process — the whole
point of automating this is that none of them can silently ship:

| Check | Catches |
|---|---|
| Tag matches `^(app\|opsdashboard)-v\d+\.\d+\.\d+$` | A typo'd or wrong-shaped tag (`aap-v1.4.0`, `app-1.4.0`, `app-v1.4`) that would otherwise need to fail some other, less obvious way. |
| Tag's version == the module's `versionName` in `build.gradle.kts` at the tagged commit | Tagging `app-v1.4.0` when you actually forgot to bump `versionName` past `0.1.0` in the PR — the single most common release-process mistake, and the exact one that motivated this doc. |
| New `versionCode` is strictly greater than the previous release's (read directly from that previous tag's `build.gradle.kts` via git history — no separate registry to keep in sync) | Forgetting to bump `versionCode`, or copy-pasting an old value — either one breaks Play Store's own "must always increase" rule, and would otherwise only surface as a rejected upload much later. |
| Full `testDebugUnitTest` suite passes, run again at release time | A green `main` at merge time doesn't guarantee the exact tagged commit is still what you think it is (a force-push, a tag on the wrong SHA) — this re-verifies instead of trusting history. |
| `assembleRelease` **and** `assembleDebug` both succeed for the tagged module | The `release` build type can behave differently from `debug` (proguard rules, resource shrinking) even when nothing in the diff looks release-specific. |

Only after all of these pass does the workflow create the GitHub Release and attach the built
APKs.

## Signing status (read before distributing anything)

**Neither app has a real release signing key configured.** `buildTypes { release { ... } }` in
both `build.gradle.kts` files has no `signingConfig`, so `assembleRelease` produces a genuinely
**unsigned** APK (`<module>-release-unsigned.apk` — verified locally with `apksigner verify`,
which correctly reports it as unverifiable). The release workflow attaches it as-is, clearly
labeled, alongside the debug-signed `<module>-debug.apk` (installable directly via
`adb install`, using the well-known public debug key — fine for internal testing, not for handing
to anyone outside the team).

Setting up real signing — generating a production keystore, storing it as encrypted GitHub
secrets, and wiring a `signingConfigs.release` block to it — is a deliberate, one-way decision:
whoever holds that keystore can publish updates to the same Play Store listing (or the same
install identity) forever, and losing it means never being able to update the app under that
identity again. That's not something to generate as a side effect of a CI task; it needs an
explicit decision about who holds the keystore and how it's backed up.

**Decision (September 12, 2026):** real signing is deliberately deferred — both apps are
currently installed via `adb`/side-load on the maintainer's own devices, not distributed to
anyone else or through Play Store, so the debug-signed APK the release workflow already produces
is sufficient. **If that changes** — either app starts going to other people, or Play Store
distribution becomes the real plan — revisit this:

- **Self-managed keystore** (the default answer for direct/GitHub-Releases distribution, no Play
  Store): the maintainer generates a keystore locally (`keytool -genkeypair -v -keystore
  release.jks -keyalg RSA -keysize 2048 -validity 10000 -alias <app>`), backs the resulting
  `.jks` file up somewhere durable and secure themselves (this is the single point of failure —
  losing it is exactly the one-way loss described above), and adds it plus its passwords as
  GitHub encrypted repo secrets themselves (`gh secret set` or the web UI — not something an
  assistant should do on a human's behalf, since it means entering real credentials). A
  `signingConfigs.release` block reading those secrets would then need to be added to both
  `build.gradle.kts` files and `android-release.yml`'s build step.
- **Play App Signing** (only if/when Play Store distribution actually happens): Google holds the
  final signing key and the developer only manages a recoverable upload key — the better answer
  for `:app` specifically if it's ever meant for the Play Store, since losing an upload key is
  fixable through Play Console rather than fatal. Doesn't apply to the current GitHub-Releases
  distribution at all.

## Troubleshooting

- **"Tag doesn't match the required format"** — check the exact pattern above; the app name must
  be exactly `app` or `opsdashboard`, lowercase, followed by `-v` and three dot-separated numbers.
- **"versionName doesn't match the tag"** — bump `versionName` in `build.gradle.kts` to match what
  you meant to tag (or delete the tag and retag with the version that's actually there): `git tag
  -d app-v1.4.0 && git push origin :refs/tags/app-v1.4.0`, then fix and retag.
- **"versionCode did not increase"** — bump `versionCode` by at least 1 past the previous release
  shown in the error.
- **Need to redo a release after it published** — delete the GitHub Release and the tag
  (`gh release delete app-v1.4.0 --yes`, then the `git tag -d`/`push :refs/tags/...` above), fix
  whatever was wrong, and push a fresh tag once the fix is merged. Never force-push an existing
  release tag to a new commit — `versionCode`/`versionName` history for this app depends on every
  past tag still pointing at what it originally did.
