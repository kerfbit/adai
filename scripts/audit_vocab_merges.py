#!/usr/bin/env python3
# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-10-18
"""Audit saved BPE vocabularies for damage from TD-220.

Before TD-220's fix, BPE merge learning keyed pair counts by `first + "|||" + second`. Pairs
containing '|' were mis-split, which either stopped merge learning early (a short merge list) or
recorded the same wrong rule over and over (duplicate rules). This script reads vocab files written
by BPETokenizer::save_vocab() (`vocab.txt`, every `<checkpoint>.vocab`) and reports, per file:

- the declared and parsed BPE_MERGES counts,
- how many merge rules are duplicates (any duplicate means the file is damaged),
- how many rules involve '|' (the pairs TD-220 mishandled),
- and, with --expected-merges N, whether the merge list is shorter than intended.

A damaged vocabulary needs rebuilding (vocab_builder) and every model trained on it retraining.

Usage:
    ./scripts/audit_vocab_merges.py vocab.txt training_sessions/**/*.vocab
    ./scripts/audit_vocab_merges.py --expected-merges 31744 vocab.txt
Exit status: 0 if no file looks damaged, 1 otherwise, 2 for usage/read errors.
"""

import argparse
import sys


def audit(path, expected_merges):
    """Return (report line, damaged?) for one vocab file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    vocab_size = next((l.split()[1] for l in lines if l.startswith("VOCAB_SIZE ")), "?")
    idx = next((i for i, l in enumerate(lines) if l.startswith("BPE_MERGES ")), None)
    if idx is None:
        return f"{path}: no BPE_MERGES section (not a BPETokenizer vocab?)", True
    declared = int(lines[idx].split()[1])
    merges = [l for l in lines[idx + 1:] if "\t" in l]
    duplicates = len(merges) - len(set(merges))
    pipe_rules = sum(1 for m in merges if "|" in m)
    problems = []
    if duplicates:
        problems.append(f"{duplicates} duplicate rule(s)")
    if len(merges) != declared:
        problems.append(f"declared {declared} but parsed {len(merges)}")
    if expected_merges is not None and len(merges) < expected_merges:
        problems.append(f"short: {len(merges)} < expected {expected_merges}")
    status = "DAMAGED (" + "; ".join(problems) + ")" if problems else "ok"
    line = (f"{path}: vocab={vocab_size} merges={len(merges)} rules_with_pipe={pipe_rules} "
            f"-> {status}")
    return line, bool(problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="vocab files to check")
    parser.add_argument("--expected-merges", type=int, default=None,
                        help="flag files with fewer merge rules than this")
    args = parser.parse_args()

    any_damaged = False
    for path in args.files:
        try:
            line, damaged = audit(path, args.expected_merges)
        except (OSError, ValueError, IndexError) as e:
            print(f"{path}: unreadable ({e})", file=sys.stderr)
            return 2
        print(line)
        any_damaged |= damaged
    return 1 if any_damaged else 0


if __name__ == "__main__":
    sys.exit(main())
