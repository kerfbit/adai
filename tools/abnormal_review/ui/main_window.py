"""
main_window.py - Top-level window: layout orchestration, status bar, menus.
"""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from ..data_loader import (
    DEFAULT_PROGRESS_PATH,
    DEFAULT_RESOLUTIONS_PATH,
    export_cleaned_dataset,
    export_resolutions,
    save_progress,
)
from ..models import ActionType, ReviewState, Sample
from .detail_panel import SampleDetailPanel
from .list_panel import SampleListPanel


class ReviewMainWindow:
    """
    Root window for the Abnormal Samples Review Tool.

    Wires up SampleListPanel (left) ↔ SampleDetailPanel (right) and owns
    the menu bar, status / progress bar, and autosave/export operations.
    """

    AUTOSAVE_INTERVAL_MS = 30_000  # 30 s

    def __init__(self, root: tk.Tk, state: ReviewState) -> None:
        self._root  = root
        self._state = state
        self._current_sample: Optional[Sample] = None
        self._progress_path = DEFAULT_PROGRESS_PATH

        self._root.title("Abnormal Samples Review")
        self._root.geometry("1200x760")
        self._root.minsize(900, 600)

        self._build_menu()
        self._build_header()
        self._build_panels()
        self._build_statusbar()

        self._update_header()
        self._schedule_autosave()

        # Select the first pending sample automatically
        first = self._state.next_pending()
        if first:
            self._select_sample(first)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self._root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Save Progress",        command=self._cmd_save,           accelerator="Ctrl+S")
        file_menu.add_command(label="Save Progress As…",    command=self._cmd_save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Resolutions…",  command=self._cmd_export_resolutions)
        file_menu.add_command(label="Export Cleaned Dataset…", command=self._cmd_export_cleaned)
        file_menu.add_separator()
        file_menu.add_command(label="Quit",                 command=self._on_close,           accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        # Review menu
        review_menu = tk.Menu(menubar, tearoff=False)
        review_menu.add_command(label="Next Pending Sample",         command=lambda: self._navigate_pending(+1), accelerator="Ctrl+→")
        review_menu.add_command(label="Previous Sample",             command=lambda: self._navigate_by_offset(-1))
        review_menu.add_separator()
        review_menu.add_command(label="Jump to Sample ID…",          command=self._cmd_jump_to_id)
        menubar.add_cascade(label="Review", menu=review_menu)

        self._root.config(menu=menubar)

        # Keyboard shortcuts
        self._root.bind_all("<Control-s>", lambda _: self._cmd_save())
        self._root.bind_all("<Control-q>", lambda _: self._on_close())
        self._root.bind_all("<Control-Right>", lambda _: self._navigate_pending(+1))
        self._root.bind_all("<Control-Left>",  lambda _: self._navigate_by_offset(-1))

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Header (progress bar + stats)
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        hdr = ttk.Frame(self._root, padding=(8, 4))
        hdr.pack(side=tk.TOP, fill=tk.X)

        # File path label
        self._file_label = ttk.Label(
            hdr,
            text=f"File: {self._state.source_path}",
            font=("TkDefaultFont", 9),
            foreground="#555555",
        )
        self._file_label.pack(side=tk.LEFT)

        # Stats on the right
        self._stats_label = ttk.Label(hdr, text="", font=("TkDefaultFont", 9, "bold"))
        self._stats_label.pack(side=tk.RIGHT)

        # Progress bar (below)
        prog_frame = ttk.Frame(self._root, padding=(8, 0, 8, 4))
        prog_frame.pack(side=tk.TOP, fill=tk.X)

        self._progress_bar = ttk.Progressbar(
            prog_frame, orient=tk.HORIZONTAL, mode="determinate", length=400
        )
        self._progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._pct_label = ttk.Label(prog_frame, text="0%", width=6)
        self._pct_label.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(self._root, orient=tk.HORIZONTAL).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Panels (PanedWindow)
    # ------------------------------------------------------------------

    def _build_panels(self) -> None:
        paned = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: sample list
        self._list_panel = SampleListPanel(
            paned,
            state=self._state,
            on_select=self._select_sample,
        )
        paned.add(self._list_panel, weight=1)

        # Right: sample detail
        self._detail_panel = SampleDetailPanel(
            paned,
            state=self._state,
            on_decision_made=self._on_decision_made,
            on_navigate=self._navigate_by_offset,
        )
        paned.add(self._detail_panel, weight=3)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_statusbar(self) -> None:
        ttk.Separator(self._root, orient=tk.HORIZONTAL).pack(fill=tk.X, side=tk.BOTTOM)
        bar = ttk.Frame(self._root, padding=(6, 2))
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(bar, textvariable=self._status_var, foreground="#444444").pack(side=tk.LEFT)

        ttk.Button(
            bar, text="Export Resolutions", command=self._cmd_export_resolutions
        ).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(
            bar, text="Export Cleaned Dataset", command=self._cmd_export_cleaned
        ).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(
            bar, text="Save Progress",  command=self._cmd_save
        ).pack(side=tk.RIGHT, padx=(4, 0))

    # ------------------------------------------------------------------
    # Selection & navigation
    # ------------------------------------------------------------------

    def _select_sample(self, sample: Sample) -> None:
        self._current_sample = sample
        self._detail_panel.load_sample(sample)
        self._list_panel.select_sample_id(sample.sample_id)
        self._set_status(
            f"Viewing sample #{sample.sample_id}  "
            f"({sample.display_reason}  loss={sample.loss:.4f}  "
            f"grad_norm={sample.grad_norm:.4f})"
        )

    def _navigate_by_offset(self, direction: int) -> None:
        """Move to the next (+1) or previous (-1) sample in current list order."""
        if self._current_sample is None:
            return
        samples = self._list_panel._sorted_samples
        if not samples:
            return
        ids = [s.sample_id for s in samples]
        try:
            idx = ids.index(self._current_sample.sample_id)
        except ValueError:
            idx = 0
        new_idx = max(0, min(len(ids) - 1, idx + direction))
        self._select_sample(samples[new_idx])

    def _navigate_pending(self, direction: int) -> None:
        """Jump to the next UNREVIEWED sample (wraps around)."""
        nxt = self._state.next_pending(
            after_id=self._current_sample.sample_id if self._current_sample else None
        )
        if nxt:
            self._select_sample(nxt)

    # ------------------------------------------------------------------
    # Decision callback (called by detail panel after an action)
    # ------------------------------------------------------------------

    def _on_decision_made(self, sample_id: int) -> None:
        self._list_panel.refresh(keep_selection=sample_id)
        self._update_header()
        self._detail_panel.refresh_decision()

        action = self._state.get_decision(sample_id).action
        action_labels = {
            ActionType.KEPT:      "kept",
            ActionType.DISCARDED: "discarded",
            ActionType.EDITED:    "saved (edited)",
        }
        label = action_labels.get(action, "updated")
        self._set_status(f"Sample #{sample_id} {label}.  "
                         f"{self._state.pending_count} remaining.")

        # Advance automatically to next pending
        nxt = self._state.next_pending(after_id=sample_id)
        if nxt and nxt.sample_id != sample_id:
            self._root.after(200, lambda: self._select_sample(nxt))

    # ------------------------------------------------------------------
    # Header / progress update
    # ------------------------------------------------------------------

    def _update_header(self) -> None:
        s = self._state
        stats = s.stats()
        self._stats_label.config(
            text=(
                f"Total: {s.total}   "
                f"Reviewed: {s.reviewed_count}   "
                f"Pending: {s.pending_count}   "
                f"| Kept: {stats['kept']}  "
                f"Discarded: {stats['discarded']}  "
                f"Edited: {stats['edited']}"
            )
        )
        pct = (s.reviewed_count / s.total * 100) if s.total else 0
        self._progress_bar["value"] = pct
        self._pct_label.config(text=f"{pct:.1f}%")

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_save(self) -> None:
        path = save_progress(self._state, self._progress_path)
        self._set_status(f"Progress saved → {path}")

    def _cmd_save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Progress As",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="abnormal_review_progress.json",
        )
        if path:
            self._progress_path = path
            self._cmd_save()

    def _cmd_export_resolutions(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Resolutions",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="abnormal_resolutions.json",
        )
        if path:
            out = export_resolutions(self._state, path)
            self._set_status(f"Resolutions exported → {out}  "
                             f"({self._state.reviewed_count} entries)")

    def _cmd_export_cleaned(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Cleaned Dataset",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="abnormal_cleaned.json",
        )
        if path:
            out = export_cleaned_dataset(self._state, path)
            n_kept = self._state.stats()["kept"] + self._state.stats()["edited"]
            self._set_status(f"Cleaned dataset exported → {out}  ({n_kept} samples)")

    def _cmd_jump_to_id(self) -> None:
        dlg = _JumpDialog(self._root)
        self._root.wait_window(dlg)
        if dlg.result is not None:
            try:
                sample = self._state.get_sample(dlg.result)
                self._select_sample(sample)
            except KeyError:
                messagebox.showerror("Not Found", f"Sample ID {dlg.result} not found.")

    # ------------------------------------------------------------------
    # Autosave
    # ------------------------------------------------------------------

    def _schedule_autosave(self) -> None:
        self._root.after(self.AUTOSAVE_INTERVAL_MS, self._autosave)

    def _autosave(self) -> None:
        try:
            save_progress(self._state, self._progress_path)
            self._set_status("Autosaved.")
        except Exception as exc:
            self._set_status(f"Autosave failed: {exc}")
        self._schedule_autosave()

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._state.reviewed_count > 0:
            if messagebox.askyesno(
                "Save Progress",
                "Save progress before quitting?",
            ):
                self._cmd_save()
        self._root.destroy()


# ---------------------------------------------------------------------------
# Simple Jump-to-ID dialog
# ---------------------------------------------------------------------------

class _JumpDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Jump to Sample ID")
        self.resizable(False, False)
        self.grab_set()
        self.result: Optional[int] = None

        ttk.Label(self, text="Enter Sample ID:").pack(padx=12, pady=(12, 4))
        self._var = tk.StringVar()
        entry = ttk.Entry(self, textvariable=self._var, width=20)
        entry.pack(padx=12)
        entry.focus_set()
        entry.bind("<Return>", lambda _: self._ok())

        btn_frame = ttk.Frame(self)
        btn_frame.pack(padx=12, pady=12)
        ttk.Button(btn_frame, text="OK",     command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _ok(self):
        try:
            self.result = int(self._var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid", "Please enter a valid integer.", parent=self)
            return
        self.destroy()
