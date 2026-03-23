"""
app.py - Entry point that wires Phase 1 (data) → Phase 2/3/4 (UI).

Usage (from project root):
    python -m tools.abnormal_review.app
    python -m tools.abnormal_review.app --samples training_sessions/abnormal_samples.json
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="abnormal-review",
        description="Human review tool for anomalous training samples.",
    )
    parser.add_argument(
        "--samples",
        default="training_sessions/abnormal_samples.json",
        metavar="PATH",
        help="Path to abnormal_samples.json  (default: training_sessions/abnormal_samples.json)",
    )
    parser.add_argument(
        "--progress",
        default="training_sessions/abnormal_review_progress.json",
        metavar="PATH",
        help="Path to load/save review progress  (default: training_sessions/abnormal_review_progress.json)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh session even if a progress file exists.",
    )
    args = parser.parse_args(argv)

    samples_path  = Path(args.samples)
    progress_path = None if args.no_resume else Path(args.progress)

    if not samples_path.exists():
        print(f"ERROR: samples file not found: {samples_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Late imports so the module can be imported without a display
    from .data_loader import build_review_state
    from .ui.main_window import ReviewMainWindow

    state = build_review_state(
        samples_path=samples_path,
        progress_path=progress_path,
    )

    root = tk.Tk()
    _apply_theme(root)

    ReviewMainWindow(root, state)
    root.mainloop()


def _apply_theme(root: tk.Tk) -> None:
    """Apply a clean visual theme."""
    style = tk.ttk.Style(root)

    # Use a modern built-in theme as base
    available = style.theme_names()
    for preferred in ("clam", "alt", "default"):
        if preferred in available:
            style.theme_use(preferred)
            break

    # Comfortable padding for all buttons
    style.configure("TButton", padding=(8, 4))

    # Slightly larger default font
    root.option_add("*Font", "TkDefaultFont 10")


if __name__ == "__main__":
    main()
