#!/usr/bin/env python3
"""
apply_resolutions.py - Apply review decisions back to source training data files.

For each sample in abnormal_resolutions.json:
  - DISCARDED → remove the INPUT/RESPONSE record from whichever gutenberg training file contains it.
  - EDITED    → patch the INPUT and/or RESPONSE line in-place.
  - KEPT      → no change to the source file.

The training file format (mirrors what ChatbotTrainer::load_conversation_data reads):

    INPUT: <input_text>
    RESPONSE: <target_text>
    <blank line>

Only the text on the INPUT: and RESPONSE: lines is used by the C++ loader;
multi-line spans between those lines are preserved as-is during editing.

Usage (from project root):
    python apply_resolutions.py
    python apply_resolutions.py --resolutions training_sessions/abnormal_resolutions.json \\
                                --data-dir    gutenberg_data/ \\
                                --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingRecord:
    """One INPUT/RESPONSE pair as stored in a training file."""
    input_text:  str
    target_text: str
    # Zero-based line index of the "INPUT:" line in the original file
    input_line:  int
    # Zero-based line index of the "RESPONSE:" line
    response_line: int
    # Zero-based line index of the terminating blank line (or EOF sentinel = len(lines))
    end_line: int


@dataclass
class Resolution:
    """One entry from abnormal_resolutions.json."""
    sample_id:       int
    action:          str   # "kept" | "discarded" | "edited"
    original_input:  str
    original_target: str
    new_input:       Optional[str]
    new_target:      Optional[str]
    notes:           Optional[str]


@dataclass
class ApplyReport:
    """Summary of what was done."""
    matched:         int = 0
    modified:        int = 0
    removed:         int = 0
    skipped_kept:    int = 0
    not_found:       List[int] = field(default_factory=list)
    files_changed:   List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing training files
# ---------------------------------------------------------------------------

def _strip(text: str) -> str:
    """Normalize whitespace the same way the C++ loader does."""
    return text.lstrip(" \t").rstrip(" \t\r\n")


def parse_training_file(path: Path) -> List[TrainingRecord]:
    """
    Parse a training file into TrainingRecord objects, each carrying the
    original line numbers so we can surgically rewrite the file.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=False)
    records: List[TrainingRecord] = []

    cur_input:        Optional[str] = None
    cur_input_line:   int           = -1
    cur_target:       Optional[str] = None
    cur_target_line:  int           = -1

    def _flush(end: int) -> None:
        nonlocal cur_input, cur_input_line, cur_target, cur_target_line
        if cur_input is not None and cur_target is not None:
            records.append(TrainingRecord(
                input_text    = cur_input,
                target_text   = cur_target,
                input_line    = cur_input_line,
                response_line = cur_target_line,
                end_line      = end,
            ))
        cur_input = cur_target = None
        cur_input_line = cur_target_line = -1

    for lineno, raw in enumerate(lines):
        stripped = _strip(raw)
        if not stripped:
            _flush(lineno)
        elif stripped.startswith("INPUT:"):
            cur_input      = _strip(stripped[6:])
            cur_input_line = lineno
        elif stripped.startswith("RESPONSE:"):
            cur_target      = _strip(stripped[9:])
            cur_target_line = lineno

    # Final pair without trailing blank
    _flush(len(lines))

    return records


# ---------------------------------------------------------------------------
# Matching helper
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Collapse internal whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", text).strip()


def find_record(
    records: List[TrainingRecord],
    input_text: str,
    target_text: str,
) -> Optional[TrainingRecord]:
    """
    Find the first record whose input/target matches.
    Tries exact match first, then normalised-whitespace match.
    """
    # Exact
    for r in records:
        if r.input_text == input_text and r.target_text == target_text:
            return r
    # Normalised
    ni, nt = _normalise(input_text), _normalise(target_text)
    for r in records:
        if _normalise(r.input_text) == ni and _normalise(r.target_text) == nt:
            return r
    return None


# ---------------------------------------------------------------------------
# Rewriting a single file
# ---------------------------------------------------------------------------

def _rewrite_file(
    path: Path,
    lines: List[str],
    modifications: List[Tuple[TrainingRecord, Optional[str], Optional[str]]],
    dry_run: bool,
    backup: bool,
) -> int:
    """
    Apply a list of (record, new_input, new_target) modifications to `lines`.
    new_input / new_target being None means "remove the record" (discard).

    Returns the number of records actually changed.
    """
    if not modifications:
        return 0

    # Build a set of actions indexed by INPUT line number
    # action: None → discard, (new_i, new_t) → edit
    actions: Dict[int, Optional[Tuple[Optional[str], Optional[str]]]] = {}
    for rec, new_i, new_t in modifications:
        if new_i is None and new_t is None:
            actions[rec.input_line] = None          # discard
        else:
            actions[rec.input_line] = (new_i, new_t)  # edit

    # Track which line ranges to suppress / modify
    # Build a map: line_no → "skip" | "replace_input:<text>" | "replace_response:<text>"
    skip_lines:    set[int] = set()
    replacements:  Dict[int, str] = {}

    for rec, new_i, new_t in modifications:
        if new_i is None and new_t is None:
            # Discard: suppress the INPUT line, RESPONSE line, and the blank that follows
            skip_lines.add(rec.input_line)
            skip_lines.add(rec.response_line)
            # Also suppress any lines between input_line+1 and end_line (inclusive)
            for ln in range(rec.input_line, rec.end_line + 1):
                skip_lines.add(ln)
        else:
            if new_i is not None:
                replacements[rec.input_line]    = f"INPUT: {new_i}"
            if new_t is not None:
                replacements[rec.response_line] = f"RESPONSE: {new_t}"

    out_lines: List[str] = []
    for lineno, raw in enumerate(lines):
        if lineno in skip_lines:
            continue
        if lineno in replacements:
            out_lines.append(replacements[lineno])
        else:
            out_lines.append(raw)

    # Collapse runs of more than one consecutive blank line (cosmetic)
    final: List[str] = []
    prev_blank = False
    for ln in out_lines:
        is_blank = ln.strip() == ""
        if is_blank and prev_blank:
            continue
        final.append(ln)
        prev_blank = is_blank

    if not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text("\n".join(final) + "\n", encoding="utf-8")

    return len(modifications)


# ---------------------------------------------------------------------------
# Main apply logic
# ---------------------------------------------------------------------------

def apply_resolutions(
    resolutions_path: Path,
    data_dir: Path,
    *,
    dry_run: bool = False,
    backup: bool  = True,
    glob:   str   = "*_training.txt",
) -> ApplyReport:
    """
    Load `resolutions_path`, scan all training files in `data_dir` matching
    `glob`, and apply DISCARD / EDIT actions.

    Args:
        resolutions_path: Path to abnormal_resolutions.json.
        data_dir:         Directory containing the gutenberg training files.
        dry_run:          Report what would change without writing anything.
        backup:           Create .bak copies of modified files (default True).
        glob:             Glob pattern to select training files.

    Returns:
        An :class:`ApplyReport` summarising the changes.
    """
    import json

    if not resolutions_path.exists():
        raise FileNotFoundError(f"Resolutions file not found: {resolutions_path}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Training data directory not found: {data_dir}")

    raw_resolutions: List[dict] = json.loads(resolutions_path.read_text(encoding="utf-8"))

    # Index resolutions by (input_text, target_text) for fast lookup
    # Keep only actions that require file modification
    actionable: List[Resolution] = []
    for entry in raw_resolutions:
        action = entry.get("action", "unreviewed")
        if action not in ("discarded", "edited"):
            continue
        actionable.append(Resolution(
            sample_id      = entry["sample_id"],
            action         = action,
            original_input = entry.get("original_input_text", ""),
            original_target= entry.get("original_target_text", ""),
            new_input      = entry.get("new_input_text"),
            new_target     = entry.get("new_target_text"),
            notes          = entry.get("reviewer_notes"),
        ))

    report = ApplyReport()
    report.skipped_kept = len(raw_resolutions) - len(actionable)

    if not actionable:
        print("No actionable resolutions (discard/edit) found.")
        return report

    # Build lookup key for unresolved resolutions
    unresolved: Dict[Tuple[str, str], Resolution] = {
        (_normalise(r.original_input), _normalise(r.original_target)): r
        for r in actionable
    }

    training_files = sorted(data_dir.glob(glob))
    if not training_files:
        print(f"Warning: no files matching '{glob}' found in {data_dir}")
        return report

    for tf in training_files:
        if not unresolved:
            break  # all resolutions matched

        records = parse_training_file(tf)
        lines   = tf.read_text(encoding="utf-8", errors="replace").splitlines(keepends=False)

        # Find which resolutions match records in this file
        file_mods: List[Tuple[TrainingRecord, Optional[str], Optional[str]]] = []

        for key, res in list(unresolved.items()):
            rec = find_record(records, res.original_input, res.original_target)
            if rec is None:
                continue

            unresolved.pop(key)
            report.matched += 1

            if res.action == "discarded":
                # new_input = None, new_target = None → discard
                file_mods.append((rec, None, None))
                report.removed += 1
            else:
                # edited: carry through only the fields that changed
                new_i = res.new_input   # may be None (original kept)
                new_t = res.new_target  # may be None (original kept)
                file_mods.append((rec, new_i, new_t))
                report.modified += 1

        if file_mods:
            changed = _rewrite_file(tf, lines, file_mods, dry_run=dry_run, backup=backup)
            if changed:
                report.files_changed.append(str(tf))
                action_word = "(DRY RUN) " if dry_run else ""
                print(f"  {action_word}{tf.name}: {len(file_mods)} record(s) modified/removed")

    # Anything left in unresolved was not found in any file
    for res in unresolved.values():
        report.not_found.append(res.sample_id)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="apply-resolutions",
        description="Apply reviewer decisions to source training data files.",
    )
    parser.add_argument(
        "--resolutions",
        default="training_sessions/abnormal_resolutions.json",
        metavar="PATH",
        help="Path to abnormal_resolutions.json (from the review GUI export).",
    )
    parser.add_argument(
        "--data-dir",
        default="gutenberg_data",
        metavar="DIR",
        help="Directory containing *_training.txt files  (default: gutenberg_data/).",
    )
    parser.add_argument(
        "--glob",
        default="*_training.txt",
        metavar="PATTERN",
        help="Glob pattern to select training files  (default: *_training.txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak copies of modified files.",
    )
    args = parser.parse_args(argv)

    resolutions_path = Path(args.resolutions)
    data_dir         = Path(args.data_dir)

    print(f"Resolutions : {resolutions_path}")
    print(f"Data dir    : {data_dir}")
    print(f"Dry run     : {args.dry_run}")
    print(f"Backups     : {not args.no_backup}")
    print()

    report = apply_resolutions(
        resolutions_path = resolutions_path,
        data_dir         = data_dir,
        dry_run          = args.dry_run,
        backup           = not args.no_backup,
        glob             = args.glob,
    )

    print()
    print("── Summary ──────────────────────────────────────")
    print(f"  Resolutions matched in files : {report.matched}")
    print(f"  Records removed  (discard)   : {report.removed}")
    print(f"  Records patched  (edit)       : {report.modified}")
    print(f"  Kept (no file change needed)  : {report.skipped_kept}")
    print(f"  Files changed                 : {len(report.files_changed)}")
    if report.files_changed:
        for f in report.files_changed:
            print(f"    {f}")
    if report.not_found:
        print(f"  NOT FOUND in any file ({len(report.not_found)} samples):")
        for sid in report.not_found:
            print(f"    sample_id={sid}")
    print("─────────────────────────────────────────────────")

    if args.dry_run:
        print("\n[Dry run — no files were modified]")


if __name__ == "__main__":
    main()
