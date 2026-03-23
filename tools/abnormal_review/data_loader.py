"""
data_loader.py - I/O layer for the Abnormal Samples Review Tool

Responsibilities:
  - Load abnormal_samples.json into a ReviewState.
  - Persist review progress to a JSON checkpoint file.
  - Export a final resolution manifest for downstream dataset cleaning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from .models import ActionType, ReviewDecision, ReviewState, Sample


# ---------------------------------------------------------------------------
# Default paths (relative to the project root)
# ---------------------------------------------------------------------------

DEFAULT_SAMPLES_PATH    = "training_sessions/abnormal_samples.json"
DEFAULT_PROGRESS_PATH   = "training_sessions/abnormal_review_progress.json"
DEFAULT_RESOLUTIONS_PATH = "training_sessions/abnormal_resolutions.json"


# ---------------------------------------------------------------------------
# Loading samples
# ---------------------------------------------------------------------------

def load_samples(path: str | os.PathLike = DEFAULT_SAMPLES_PATH) -> List[Sample]:
    """
    Parse `path` (a JSON array of sample objects) and return a list of
    :class:`Sample` instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain a JSON array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Samples file not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array in {p}; got {type(raw).__name__}"
        )

    return [Sample.from_dict(item) for item in raw]


# ---------------------------------------------------------------------------
# Building (or resuming) a ReviewState
# ---------------------------------------------------------------------------

def build_review_state(
    samples_path: str | os.PathLike = DEFAULT_SAMPLES_PATH,
    progress_path: Optional[str | os.PathLike] = DEFAULT_PROGRESS_PATH,
) -> ReviewState:
    """
    Load samples and, if a progress checkpoint exists at `progress_path`,
    merge the saved decisions back in so the session resumes where it left off.

    Args:
        samples_path:  Path to the original abnormal_samples JSON.
        progress_path: Path to the checkpoint file.  Pass ``None`` to always
                       start a fresh session.

    Returns:
        A :class:`ReviewState` ready to hand to the UI layer.
    """
    samples = load_samples(samples_path)

    # Start fresh
    state = ReviewState(source_path=str(samples_path))
    state.load_samples(samples)

    # Attempt to resume
    if progress_path is not None:
        pp = Path(progress_path)
        if pp.exists():
            saved_state = load_progress(pp)
            # Overlay saved decisions onto the freshly-loaded state
            for sample_id, decision in saved_state.decisions.items():
                if sample_id in state.decisions:
                    state.decisions[sample_id] = decision
            state.session_started_at = saved_state.session_started_at

    return state


# ---------------------------------------------------------------------------
# Saving / loading progress checkpoints
# ---------------------------------------------------------------------------

def save_progress(
    state: ReviewState,
    path: str | os.PathLike = DEFAULT_PROGRESS_PATH,
) -> Path:
    """
    Persist the current review decisions to `path` as a JSON checkpoint.

    The checkpoint only stores decisions (not the full sample text), so it
    is small and safe to overwrite frequently.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(state.to_json())
    return p


def load_progress(
    path: str | os.PathLike = DEFAULT_PROGRESS_PATH,
) -> ReviewState:
    """
    Deserialise a checkpoint file written by :func:`save_progress`.

    Returns a :class:`ReviewState` containing only the decisions (no samples).
    Callers should merge this into a fully-loaded state via
    :func:`build_review_state` rather than using the returned object directly.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Progress file not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    return ReviewState.from_dict(data)


# ---------------------------------------------------------------------------
# Exporting the resolution manifest for downstream use
# ---------------------------------------------------------------------------

def export_resolutions(
    state: ReviewState,
    path: str | os.PathLike = DEFAULT_RESOLUTIONS_PATH,
    *,
    include_unreviewed: bool = False,
) -> Path:
    """
    Write a resolution manifest: one JSON object per reviewed sample.

    Format for each entry:

    .. code-block:: json

        {
            "sample_id": 5679,
            "action": "discard",
            "original_input_text": "Summarize: ...",
            "original_target_text": "A",
            "new_input_text": null,
            "new_target_text": null,
            "reviewer_notes": "Garbage target.",
            "reviewed_at": "2026-03-15T14:00:00+00:00",
            "loss": 2.427368,
            "grad_norm": 23.287205,
            "reason": "grad_norm_outlier",
            "epoch": 10
        }

    This manifest can be read by dataset pre-processing scripts to filter or
    patch samples before the next training run.

    Args:
        state:              The current :class:`ReviewState`.
        path:               Output file path.
        include_unreviewed: When True, UNREVIEWED entries are included too
                            (useful for auditing gaps).

    Returns:
        The resolved :class:`~pathlib.Path` that was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Index samples by id for fast lookup
    sample_index = {s.sample_id: s for s in state.samples}

    records = []
    for decision in state.decisions.values():
        if not include_unreviewed and not decision.is_reviewed:
            continue

        sample: Optional[Sample] = sample_index.get(decision.sample_id)
        record: dict = decision.to_dict()

        # Annotate with original sample fields for traceability
        if sample is not None:
            record["original_input_text"]  = sample.input_text
            record["original_target_text"] = sample.target_text
            record["loss"]      = sample.loss
            record["grad_norm"] = sample.grad_norm
            record["reason"]    = sample.reason
            record["epoch"]     = sample.epoch

        records.append(record)

    # Sort by sample_id for deterministic output
    records.sort(key=lambda r: r["sample_id"])

    with p.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    return p


# ---------------------------------------------------------------------------
# Convenience: export a cleaned dataset (kept + edited samples only)
# ---------------------------------------------------------------------------

def export_cleaned_dataset(
    state: ReviewState,
    path: str | os.PathLike = "training_sessions/abnormal_cleaned.json",
) -> Path:
    """
    Write a JSON array containing only the samples the reviewer chose to keep
    or edit — ready to be appended back into the main training dataset.

    For EDITED samples the corrected text fields are used; for KEPT samples
    the original fields are passed through.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    sample_index = {s.sample_id: s for s in state.samples}
    keep_actions = {ActionType.KEPT, ActionType.EDITED}

    records = []
    for decision in state.decisions.values():
        if decision.action not in keep_actions:
            continue

        sample = sample_index.get(decision.sample_id)
        if sample is None:
            continue

        record = sample.to_dict()

        if decision.action == ActionType.EDITED:
            if decision.new_input_text is not None:
                record["input_text"] = decision.new_input_text
            if decision.new_target_text is not None:
                record["target_text"] = decision.new_target_text
            record["_edited"] = True
        else:
            record["_edited"] = False

        records.append(record)

    records.sort(key=lambda r: r["sample_id"])

    with p.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    return p
