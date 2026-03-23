"""
list_panel.py - Left panel: filterable / sortable sample list sidebar.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional

from ..models import ActionType, ReviewState, Sample, SortField


# Colour coding per action
_ACTION_COLOURS = {
    ActionType.UNREVIEWED: "#d0d0d0",
    ActionType.KEPT:       "#7ec88a",
    ActionType.DISCARDED:  "#e07070",
    ActionType.EDITED:     "#7aaee0",
}

_ACTION_ICONS = {
    ActionType.UNREVIEWED: "○",
    ActionType.KEPT:       "✓",
    ActionType.DISCARDED:  "✗",
    ActionType.EDITED:     "✎",
}

_SORT_OPTIONS = {
    "Grad Norm (↓)": (SortField.GRAD_NORM, True),
    "Grad Norm (↑)": (SortField.GRAD_NORM, False),
    "Loss (↓)":      (SortField.LOSS,      True),
    "Loss (↑)":      (SortField.LOSS,      False),
    "Sample ID (↑)": (SortField.SAMPLE_ID, False),
    "Epoch (↑)":     (SortField.EPOCH,     False),
}


class SampleListPanel(ttk.Frame):
    """
    Left sidebar that displays all samples in a scrollable listbox with
    filter (by reason / action) and sort controls.

    The caller is notified of selection changes via the `on_select` callback.
    """

    def __init__(
        self,
        parent: tk.Widget,
        state: ReviewState,
        on_select: Callable[[Sample], None],
        **kw,
    ):
        super().__init__(parent, **kw)
        self._state     = state
        self._on_select = on_select
        self._sorted_samples: List[Sample] = []
        self._inhibit_select = False

        self._build_controls()
        self._build_list()
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))

        # --- Reason filter ---
        ttk.Label(ctrl, text="Reason").grid(row=0, column=0, sticky=tk.W)
        self._reason_var = tk.StringVar(value="All")
        self._reason_cb = ttk.Combobox(
            ctrl, textvariable=self._reason_var,
            state="readonly", width=22,
        )
        self._reason_cb.grid(row=0, column=1, sticky=tk.EW, padx=(2, 0))
        self._reason_cb.bind("<<ComboboxSelected>>", self._on_filter_changed)

        # --- Action filter ---
        ttk.Label(ctrl, text="Status").grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
        self._action_var = tk.StringVar(value="All")
        self._action_cb  = ttk.Combobox(
            ctrl, textvariable=self._action_var,
            values=["All"] + [a.value for a in ActionType],
            state="readonly", width=22,
        )
        self._action_cb.grid(row=1, column=1, sticky=tk.EW, padx=(2, 0), pady=(2, 0))
        self._action_cb.bind("<<ComboboxSelected>>", self._on_filter_changed)

        # --- Sort ---
        ttk.Label(ctrl, text="Sort").grid(row=2, column=0, sticky=tk.W, pady=(2, 0))
        self._sort_var = tk.StringVar(value="Grad Norm (↓)")
        self._sort_cb  = ttk.Combobox(
            ctrl, textvariable=self._sort_var,
            values=list(_SORT_OPTIONS.keys()),
            state="readonly", width=22,
        )
        self._sort_cb.grid(row=2, column=1, sticky=tk.EW, padx=(2, 0), pady=(2, 0))
        self._sort_cb.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ctrl.columnconfigure(1, weight=1)

        # --- count label ---
        self._count_label = ttk.Label(self, text="", foreground="#555555")
        self._count_label.pack(side=tk.TOP, fill=tk.X, padx=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(4, 0))

    def _build_list(self) -> None:
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._listbox = tk.Listbox(
            frame,
            selectmode=tk.SINGLE,
            activestyle="none",
            font=("Courier", 10),
            exportselection=False,
            width=30,
        )
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

    # ------------------------------------------------------------------
    # Refresh (rebuilds the listbox from current state + filters)
    # ------------------------------------------------------------------

    def refresh(self, keep_selection: Optional[int] = None) -> None:
        """
        Rebuild the list from state.  If `keep_selection` is a sample_id,
        reselect that row after refresh.
        """
        # Update reason dropdown to match available reasons
        reasons = ["All"] + self._state.distinct_reasons()
        self._reason_cb["values"] = reasons

        # Determine active filters
        reason_filter = self._reason_var.get()
        if reason_filter == "All":
            reason_filter = None

        action_filter_raw = self._action_var.get()
        action_filter: Optional[ActionType] = None
        if action_filter_raw != "All":
            try:
                action_filter = ActionType(action_filter_raw)
            except ValueError:
                pass

        # Sort
        sort_key, sort_desc = _SORT_OPTIONS.get(
            self._sort_var.get(), (SortField.GRAD_NORM, True)
        )
        samples = self._state.sorted_samples(by=sort_key, descending=sort_desc)

        # Filter
        if reason_filter:
            samples = [s for s in samples if s.reason == reason_filter]
        if action_filter is not None:
            samples = [
                s for s in samples
                if self._state.get_decision(s.sample_id).action == action_filter
            ]

        self._sorted_samples = samples

        # Rebuild listbox
        self._inhibit_select = True
        self._listbox.delete(0, tk.END)
        reselect_idx: Optional[int] = None

        for idx, s in enumerate(samples):
            dec    = self._state.get_decision(s.sample_id)
            icon   = _ACTION_ICONS[dec.action]
            colour = _ACTION_COLOURS[dec.action]
            label  = f"{icon} #{s.sample_id:>6}  gn={s.grad_norm:>6.2f}"
            self._listbox.insert(tk.END, label)
            self._listbox.itemconfig(idx, background=colour)
            if keep_selection == s.sample_id:
                reselect_idx = idx

        self._inhibit_select = False

        # Restore selection
        if reselect_idx is not None:
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(reselect_idx)
            self._listbox.see(reselect_idx)

        n = len(samples)
        self._count_label.config(
            text=f"{n} sample{'s' if n != 1 else ''} shown"
        )

    # ------------------------------------------------------------------
    # Selection control (called externally by the main window)
    # ------------------------------------------------------------------

    def select_sample_id(self, sample_id: int) -> None:
        """Highlight the row for `sample_id` in the listbox."""
        for idx, s in enumerate(self._sorted_samples):
            if s.sample_id == sample_id:
                self._inhibit_select = True
                self._listbox.selection_clear(0, tk.END)
                self._listbox.selection_set(idx)
                self._listbox.see(idx)
                self._inhibit_select = False
                return

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_listbox_select(self, _event=None) -> None:
        if self._inhibit_select:
            return
        sel = self._listbox.curselection()
        if not sel:
            return
        sample = self._sorted_samples[sel[0]]
        self._on_select(sample)

    def _on_filter_changed(self, _event=None) -> None:
        # Reapply selection after filter change
        sel = self._listbox.curselection()
        cur_id = None
        if sel:
            cur_id = self._sorted_samples[sel[0]].sample_id
        self.refresh(keep_selection=cur_id)
