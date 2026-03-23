"""
detail_panel.py - Right panel: sample metrics, editable text fields, action buttons.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from ..models import ActionType, ReviewDecision, ReviewState, Sample


_REASON_COLOURS = {
    "grad_norm_outlier":          "#c8702a",
    "grad_norm_and_loss_outlier": "#b83030",
}


class SampleDetailPanel(ttk.Frame):
    """
    Right panel that shows the currently-selected sample's details.

    Callbacks:
        on_decision_made(sample_id)  - called after Keep / Discard / Save Edits
        on_navigate(direction)       - called when Prev / Next is clicked
                                       direction is -1 (prev) or +1 (next)
    """

    def __init__(
        self,
        parent: tk.Widget,
        state: ReviewState,
        on_decision_made: Callable[[int], None],
        on_navigate: Callable[[int], None],
        **kw,
    ):
        super().__init__(parent, **kw)
        self._state            = state
        self._on_decision_made = on_decision_made
        self._on_navigate      = on_navigate
        self._current_sample: Optional[Sample] = None

        self._build()
        self._set_empty_state()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # ---- Metrics bar -------------------------------------------
        metrics = ttk.LabelFrame(self, text="Sample Metrics", padding=6)
        metrics.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        self._metric_vars = {k: tk.StringVar(value="—") for k in
                             ["Sample ID", "Epoch", "Loss", "Grad Norm", "Reason", "Timestamp"]}

        layout = [
            ("Sample ID", 0, 0), ("Epoch",     0, 2),
            ("Loss",      1, 0), ("Grad Norm", 1, 2),
            ("Reason",    2, 0), ("Timestamp", 2, 2),
        ]
        for key, row, col in layout:
            ttk.Label(metrics, text=f"{key}:", font=("TkDefaultFont", 9, "bold")
                      ).grid(row=row, column=col, sticky=tk.W, padx=(0, 4), pady=1)
            lbl = ttk.Label(metrics, textvariable=self._metric_vars[key])
            lbl.grid(row=row, column=col + 1, sticky=tk.W, padx=(0, 16), pady=1)
            if key == "Reason":
                self._reason_label = lbl

        metrics.columnconfigure(1, weight=1)
        metrics.columnconfigure(3, weight=1)

        # ---- Decision badge -----------------------------------------
        self._decision_var   = tk.StringVar(value="")
        self._decision_label = ttk.Label(self, textvariable=self._decision_var,
                                         font=("TkDefaultFont", 9, "italic"),
                                         foreground="#444444")
        self._decision_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(0, 2))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=2)

        # ---- Input text --------------------------------------------
        in_frame = ttk.LabelFrame(self, text="Input Text", padding=4)
        in_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=2)

        self._input_text = tk.Text(
            in_frame, wrap=tk.WORD, height=6,
            font=("TkFixedFont", 10), relief=tk.FLAT,
            borderwidth=1, highlightbackground="#aaaaaa",
            highlightthickness=1, undo=True,
        )
        in_scroll = ttk.Scrollbar(in_frame, orient=tk.VERTICAL, command=self._input_text.yview)
        self._input_text.configure(yscrollcommand=in_scroll.set)
        in_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Target text -------------------------------------------
        tgt_frame = ttk.LabelFrame(self, text="Target Text", padding=4)
        tgt_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=2)

        self._target_text = tk.Text(
            tgt_frame, wrap=tk.WORD, height=4,
            font=("TkFixedFont", 10), relief=tk.FLAT,
            borderwidth=1, highlightbackground="#aaaaaa",
            highlightthickness=1, undo=True,
        )
        tgt_scroll = ttk.Scrollbar(tgt_frame, orient=tk.VERTICAL, command=self._target_text.yview)
        self._target_text.configure(yscrollcommand=tgt_scroll.set)
        tgt_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._target_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Reviewer notes ----------------------------------------
        notes_frame = ttk.LabelFrame(self, text="Reviewer Notes (optional)", padding=4)
        notes_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)

        self._notes_text = tk.Text(
            notes_frame, wrap=tk.WORD, height=2,
            font=("TkDefaultFont", 9),
        )
        self._notes_text.pack(fill=tk.X)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=4)

        # ---- Action buttons ----------------------------------------
        actions = ttk.Frame(self)
        actions.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))

        btn_cfg = dict(padding=(12, 6), width=14)

        self._btn_keep = ttk.Button(
            actions, text="✓  Keep Sample",
            command=self._action_keep, style="Keep.TButton", **btn_cfg,
        )
        self._btn_discard = ttk.Button(
            actions, text="✗  Discard Sample",
            command=self._action_discard, style="Discard.TButton", **btn_cfg,
        )
        self._btn_save_edit = ttk.Button(
            actions, text="✎  Save Edits",
            command=self._action_save_edit, **btn_cfg,
        )
        self._btn_reset = ttk.Button(
            actions, text="↺  Reset",
            command=self._action_reset, width=8,
        )

        self._btn_keep.pack(side=tk.LEFT, padx=(0, 4))
        self._btn_discard.pack(side=tk.LEFT, padx=(0, 4))
        self._btn_save_edit.pack(side=tk.LEFT, padx=(0, 4))
        self._btn_reset.pack(side=tk.LEFT)

        # ---- Navigation bar ----------------------------------------
        nav = ttk.Frame(self)
        nav.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

        self._btn_prev = ttk.Button(
            nav, text="◀  Previous",
            command=lambda: self._on_navigate(-1), width=12,
        )
        self._btn_next = ttk.Button(
            nav, text="Next  ▶",
            command=lambda: self._on_navigate(+1), width=12,
        )
        self._btn_prev.pack(side=tk.LEFT)
        self._btn_next.pack(side=tk.RIGHT)

        # ---- Configure custom styles --------------------------------
        style = ttk.Style(self)
        style.configure("Keep.TButton",    foreground="#1a6e2e")
        style.configure("Discard.TButton", foreground="#8b0000")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_sample(self, sample: Sample) -> None:
        """Populate all fields from `sample` and its current decision."""
        self._current_sample = sample
        dec = self._state.get_decision(sample.sample_id)

        # Metrics
        self._metric_vars["Sample ID"].set(str(sample.sample_id))
        self._metric_vars["Epoch"].set(str(sample.epoch))
        self._metric_vars["Loss"].set(f"{sample.loss:.6f}")
        self._metric_vars["Grad Norm"].set(f"{sample.grad_norm:.6f}")
        self._metric_vars["Reason"].set(sample.display_reason)
        self._metric_vars["Timestamp"].set(sample.timestamp)

        reason_colour = _REASON_COLOURS.get(sample.reason, "#333333")
        self._reason_label.configure(foreground=reason_colour)

        # Text fields — use corrected text if previously edited
        input_val  = dec.new_input_text  if dec.new_input_text  is not None else sample.input_text
        target_val = dec.new_target_text if dec.new_target_text is not None else sample.target_text

        self._set_text(self._input_text,  input_val)
        self._set_text(self._target_text, target_val)
        self._set_text(self._notes_text,  dec.reviewer_notes or "")

        self._update_decision_badge(dec)
        self._update_button_states()

    def refresh_decision(self) -> None:
        """Re-render decision badge / button states after an external change."""
        if self._current_sample:
            dec = self._state.get_decision(self._current_sample.sample_id)
            self._update_decision_badge(dec)
            self._update_button_states()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _action_keep(self) -> None:
        if not self._current_sample:
            return
        notes = self._notes_text.get("1.0", tk.END).strip()
        self._state.get_decision(self._current_sample.sample_id).apply_keep(notes)
        self._update_decision_badge(self._state.get_decision(self._current_sample.sample_id))
        self._update_button_states()
        self._on_decision_made(self._current_sample.sample_id)

    def _action_discard(self) -> None:
        if not self._current_sample:
            return
        notes = self._notes_text.get("1.0", tk.END).strip()
        self._state.get_decision(self._current_sample.sample_id).apply_discard(notes)
        self._update_decision_badge(self._state.get_decision(self._current_sample.sample_id))
        self._update_button_states()
        self._on_decision_made(self._current_sample.sample_id)

    def _action_save_edit(self) -> None:
        if not self._current_sample:
            return
        new_input  = self._get_text(self._input_text)
        new_target = self._get_text(self._target_text)
        notes      = self._notes_text.get("1.0", tk.END).strip()

        # Treat unchanged text as None (no edit)
        orig = self._current_sample
        new_input  = new_input  if new_input  != orig.input_text  else None
        new_target = new_target if new_target != orig.target_text else None

        if new_input is None and new_target is None:
            messagebox.showinfo(
                "No Changes",
                "Neither Input nor Target text was modified.\n"
                "Use 'Keep Sample' if the original is acceptable.",
            )
            return

        self._state.get_decision(orig.sample_id).apply_edit(new_input, new_target, notes)
        self._update_decision_badge(self._state.get_decision(orig.sample_id))
        self._update_button_states()
        self._on_decision_made(orig.sample_id)

    def _action_reset(self) -> None:
        if not self._current_sample:
            return
        if messagebox.askyesno(
            "Reset Decision",
            f"Clear the decision for sample #{self._current_sample.sample_id}?",
        ):
            dec = self._state.get_decision(self._current_sample.sample_id)
            dec.reset()
            self.load_sample(self._current_sample)
            self._on_decision_made(self._current_sample.sample_id)

    # ------------------------------------------------------------------
    # Empty / disabled state
    # ------------------------------------------------------------------

    def _set_empty_state(self) -> None:
        for v in self._metric_vars.values():
            v.set("—")
        self._decision_var.set("")
        self._set_text(self._input_text,  "← Select a sample from the list")
        self._set_text(self._target_text, "")
        self._set_text(self._notes_text,  "")
        self._input_text.config(state=tk.DISABLED)
        self._target_text.config(state=tk.DISABLED)
        self._notes_text.config(state=tk.DISABLED)
        for btn in (self._btn_keep, self._btn_discard,
                    self._btn_save_edit, self._btn_reset,
                    self._btn_prev, self._btn_next):
            btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_decision_badge(self, dec: ReviewDecision) -> None:
        badges = {
            ActionType.UNREVIEWED: "",
            ActionType.KEPT:       "✓ Marked as KEPT",
            ActionType.DISCARDED:  "✗ Marked as DISCARDED",
            ActionType.EDITED:     "✎ Marked as EDITED",
        }
        colours = {
            ActionType.UNREVIEWED: "#444444",
            ActionType.KEPT:       "#1a6e2e",
            ActionType.DISCARDED:  "#8b0000",
            ActionType.EDITED:     "#0050a0",
        }
        self._decision_var.set(badges[dec.action])
        self._decision_label.configure(foreground=colours[dec.action])

    def _update_button_states(self) -> None:
        enabled = self._current_sample is not None
        for btn in (self._btn_keep, self._btn_discard,
                    self._btn_save_edit, self._btn_reset,
                    self._btn_prev, self._btn_next):
            btn.config(state=tk.NORMAL if enabled else tk.DISABLED)
        if enabled:
            self._input_text.config(state=tk.NORMAL)
            self._target_text.config(state=tk.NORMAL)
            self._notes_text.config(state=tk.NORMAL)

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        state = widget.cget("state")
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.edit_reset()  # clear undo history
        widget.config(state=state)

    @staticmethod
    def _get_text(widget: tk.Text) -> str:
        return widget.get("1.0", tk.END).rstrip("\n")
