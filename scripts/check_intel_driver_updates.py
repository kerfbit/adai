#!/usr/bin/env python3

# @adai-status: beta        (current — see ONEAPI_SYCL_DRIVER_SEGFAULT.md; capped by TD-045 — see TECHNICAL_DEBT.md)
# @adai-version: 0.6.0
# @adai-reviewed: 2026-09-07

"""
Intel Driver/Compiler Update Monitor

Checks official Intel sources daily for updates relevant to the Arc B60/Battlemage
`xe` + oneAPI/SYCL segfault documented in:
  docs/operations/guides/troubleshooting/ONEAPI_SYCL_DRIVER_SEGFAULT.md

Root cause was pinned to libigc.so.2 (Intel Graphics Compiler) crashing during JIT
kernel translation, alongside a GT0 engine-reset pattern in dmesg correlating with
every recorded crash. This script has no way to know when upstream actually fixes
it, so it polls the sources a human would otherwise have to remember to check by
hand, and only makes noise when one of them changes.

Sources checked (all official Intel channels, plus the Canonical/Intel PPA the
investigation actually used to test a driver bump):
  - intel/intel-graphics-compiler   latest GitHub release
  - intel/compute-runtime           latest GitHub release
  - oneapi-src/level-zero           latest GitHub release
  - dgpu-docs.intel.com             driver install/release notes page (content hash)
  - launchpad kobuk-team PPA        Battlemage preview package page (content hash)

NOTE: intel/intel-graphics-compiler#159, cited in the KB doc's "External
corroboration" section, is NOT tracked here. Checked directly (2026-08-10): it's a
closed 2020/2021 Kaby Lake (kbl) SPIR-V compile segfault against igc v1.0.5353 /
compute-runtime v20.44 -- unrelated hardware, unrelated driver generation, years
before Battlemage existed. It doesn't corroborate this bug and isn't a live
upstream thread worth polling. Worth fixing the KB doc's citation separately; the
other three external corroboration links in that doc (vllm-project/vllm#41663,
darktable-org/darktable#20257, ggml-org/llama.cpp#24810) are not Intel-owned
repos so they're out of scope for an "official Intel sources" monitor anyway.

Usage:
    ./scripts/check_intel_driver_updates.py                 # check + report changes
    ./scripts/check_intel_driver_updates.py --verbose        # also print unchanged sources
    ./scripts/check_intel_driver_updates.py --json           # machine-readable report
    ./scripts/check_intel_driver_updates.py --state-file P   # override state location

Designed for cron: with no changes it prints nothing and exits 0. With changes it
prints a report and exits 2, so a standard crontab entry (no output redirection)
gets you email-on-change for free:

    0 8 * * * /usr/bin/python3 /opt/adai/scripts/check_intel_driver_updates.py \
        >> /var/log/adai/intel_driver_monitor.log 2>&1

A per-source fetch failure (network blip, GitHub rate limit, etc.) is logged as a
warning and does not fail the other sources or corrupt their saved state.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_STATE_FILE = os.path.expanduser("~/.local/state/adai/intel_driver_monitor_state.json")
USER_AGENT = "adai-intel-driver-monitor/1.0 (+https://github.com/; incident: ONEAPI_SYCL_DRIVER_SEGFAULT.md)"
TIMEOUT_SECONDS = 20

SOURCES = [
    {
        "id": "igc_release",
        "label": "intel/intel-graphics-compiler latest release",
        "kind": "github_release",
        "url": "https://api.github.com/repos/intel/intel-graphics-compiler/releases/latest",
        "link": "https://github.com/intel/intel-graphics-compiler/releases",
    },
    {
        "id": "compute_runtime_release",
        "label": "intel/compute-runtime latest release",
        "kind": "github_release",
        "url": "https://api.github.com/repos/intel/compute-runtime/releases/latest",
        "link": "https://github.com/intel/compute-runtime/releases",
    },
    {
        "id": "level_zero_release",
        "label": "oneapi-src/level-zero latest release",
        "kind": "github_release",
        "url": "https://api.github.com/repos/oneapi-src/level-zero/releases/latest",
        "link": "https://github.com/oneapi-src/level-zero/releases",
    },
    {
        "id": "dgpu_docs_install",
        "label": "dgpu-docs.intel.com Intel PPA install/version guide",
        "kind": "page_hash",
        # /driver/installation.html meta-refreshes here; the redirect target is what
        # actually holds the versioned install instructions, so hash it directly.
        "url": "https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html",
        "link": "https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html",
    },
    {
        "id": "kobuk_ppa",
        "label": "kobuk-team/intel-graphics PPA (Battlemage preview packages)",
        "kind": "page_hash",
        "url": "https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics",
        "link": "https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics",
        # Launchpad stamps a per-request "N queries issued in X seconds" line into
        # the footer of every response, which would otherwise make this source
        # "change" on literally every fetch. Strip lines matching this before hashing.
        "volatile_line_pattern": r"queries/external actions issued in",
    },
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return resp.read()


def check_github_release(source):
    data = json.loads(fetch(source["url"]))
    return {
        "tag": data.get("tag_name"),
        "name": data.get("name"),
        "published_at": data.get("published_at"),
        "url": data.get("html_url"),
    }


def check_github_issue(source):
    data = json.loads(fetch(source["url"]))
    return {
        "state": data.get("state"),
        "comments": data.get("comments"),
        "updated_at": data.get("updated_at"),
        "title": data.get("title"),
    }


def check_page_hash(source):
    text = fetch(source["url"]).decode("utf-8", errors="replace")
    pattern = source.get("volatile_line_pattern")
    if pattern:
        text = "\n".join(line for line in text.splitlines() if not re.search(pattern, line))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"sha256": digest, "bytes": len(text)}


CHECKERS = {
    "github_release": check_github_release,
    "github_issue": check_github_issue,
    "page_hash": check_page_hash,
}


def diff_snapshot(old, new):
    """Return a human-readable list of field changes between two snapshots."""
    if old is None:
        return ["first check (no prior state to compare against)"]
    changes = []
    for key in sorted(set(old.keys()) | set(new.keys())):
        old_val, new_val = old.get(key), new.get(key)
        if old_val != new_val:
            changes.append(f"{key}: {old_val!r} -> {new_val!r}")
    return changes


def load_state(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read state file {path} ({e}); starting fresh", file=sys.stderr)
        return {}


def save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def run(state_file, verbose):
    state = load_state(state_file)
    results = []

    for source in SOURCES:
        checker = CHECKERS[source["kind"]]
        entry = {"id": source["id"], "label": source["label"], "link": source["link"]}
        try:
            snapshot = checker(source)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            entry["status"] = "error"
            entry["error"] = str(e)
            results.append(entry)
            continue
        except (json.JSONDecodeError, ValueError) as e:
            entry["status"] = "error"
            entry["error"] = f"unparseable response: {e}"
            results.append(entry)
            continue

        old_snapshot = state.get(source["id"], {}).get("snapshot")
        changes = diff_snapshot(old_snapshot, snapshot)
        entry["status"] = "changed" if changes else "unchanged"
        entry["changes"] = changes
        entry["snapshot"] = snapshot
        results.append(entry)

        state[source["id"]] = {
            "snapshot": snapshot,
            "last_checked": datetime.now(timezone.utc).isoformat(),
        }

    save_state(state_file, state)
    return results


def print_report(results, verbose, as_json):
    if as_json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return any(r["status"] == "changed" for r in results)

    changed = [r for r in results if r["status"] == "changed"]
    errored = [r for r in results if r["status"] == "error"]

    if changed:
        print(f"=== Intel driver/compiler update check: {len(changed)} source(s) changed ===")
        print(f"(re: docs/operations/guides/troubleshooting/ONEAPI_SYCL_DRIVER_SEGFAULT.md)\n")
        for r in changed:
            print(f"[CHANGED] {r['label']}")
            print(f"  {r['link']}")
            for c in r["changes"]:
                print(f"    - {c}")
            print()

    if errored:
        print(f"=== {len(errored)} source(s) failed to fetch (not treated as a change) ===")
        for r in errored:
            print(f"[ERROR] {r['label']}: {r['error']}")
        print()

    if verbose:
        unchanged = [r for r in results if r["status"] == "unchanged"]
        if unchanged:
            print("=== Unchanged ===")
            for r in unchanged:
                print(f"[ok] {r['label']}")

    return bool(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="Path to persisted state JSON")
    parser.add_argument("--verbose", action="store_true", help="Also print unchanged sources")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report instead of text")
    args = parser.parse_args()

    results = run(args.state_file, args.verbose)
    had_changes = print_report(results, args.verbose, args.json)
    had_errors = any(r["status"] == "error" for r in results)

    if had_changes:
        sys.exit(2)
    if had_errors and args.verbose:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
