"""
models.py - Data models for the Abnormal Samples Review Tool

Defines the core data structures used to represent training samples,
review decisions, and the overall review session state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """The decision a reviewer can apply to a flagged sample."""
    UNREVIEWED = "unreviewed"
    KEPT       = "kept"      # Valid sample; outlier metrics are a false alarm.
    DISCARDED  = "discarded" # Poor-quality sample; remove from dataset.
    EDITED     = "edited"    # Sample kept but with corrected input/target text.


class SortField(str, Enum):
    """Fields available for sorting the sample list."""
    GRAD_NORM  = "grad_norm"
    LOSS       = "loss"
    SAMPLE_ID  = "sample_id"
    EPOCH      = "epoch"


# ---------------------------------------------------------------------------
# Sample (read from abnormal_samples.json)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """
    Immutable representation of a single abnormal training sample as stored
    in abnormal_samples.json.
    """
    epoch:       int
    sample_id:   int
    loss:        float
    grad_norm:   float
    reason:      str
    input_text:  str
    target_text: str
    timestamp:   str

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            epoch       = int(d["epoch"]),
            sample_id   = int(d["sample_id"]),
            loss        = float(d["loss"]),
            grad_norm   = float(d["grad_norm"]),
            reason      = str(d["reason"]),
            input_text  = str(d["input_text"]),
            target_text = str(d["target_text"]),
            timestamp   = str(d["timestamp"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    # Convenience shortcuts for display
    @property
    def display_reason(self) -> str:
        return self.reason.replace("_", " ").title()


# ---------------------------------------------------------------------------
# ReviewDecision (one per sample; tracks what was decided and when)
# ---------------------------------------------------------------------------

@dataclass
class ReviewDecision:
    """
    Mutable record of the reviewer's decision for a single sample.

    For EDITED decisions, `new_input_text` and/or `new_target_text` carry
    the corrected text; None means the original is kept as-is.
    """
    sample_id:     int
    action:        ActionType = ActionType.UNREVIEWED
    new_input_text:  Optional[str] = None
    new_target_text: Optional[str] = None
    reviewer_notes:  Optional[str] = None
    reviewed_at:     Optional[str] = None  # ISO-8601 UTC string

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def apply_keep(self, notes: str = "") -> None:
        self.action = ActionType.KEPT
        self.new_input_text  = None
        self.new_target_text = None
        self.reviewer_notes  = notes or None
        self._stamp()

    def apply_discard(self, notes: str = "") -> None:
        self.action = ActionType.DISCARDED
        self.new_input_text  = None
        self.new_target_text = None
        self.reviewer_notes  = notes or None
        self._stamp()

    def apply_edit(
        self,
        new_input_text: Optional[str],
        new_target_text: Optional[str],
        notes: str = "",
    ) -> None:
        """Save corrected text.  Pass None for a field to keep the original."""
        self.action          = ActionType.EDITED
        self.new_input_text  = new_input_text
        self.new_target_text = new_target_text
        self.reviewer_notes  = notes or None
        self._stamp()

    def reset(self) -> None:
        """Revert to UNREVIEWED (allows re-review)."""
        self.action          = ActionType.UNREVIEWED
        self.new_input_text  = None
        self.new_target_text = None
        self.reviewer_notes  = None
        self.reviewed_at     = None

    @property
    def is_reviewed(self) -> bool:
        return self.action != ActionType.UNREVIEWED

    def _stamp(self) -> None:
        self.reviewed_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "sample_id":      self.sample_id,
            "action":         self.action.value,
            "new_input_text":  self.new_input_text,
            "new_target_text": self.new_target_text,
            "reviewer_notes":  self.reviewer_notes,
            "reviewed_at":     self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewDecision":
        return cls(
            sample_id      = int(d["sample_id"]),
            action         = ActionType(d.get("action", ActionType.UNREVIEWED)),
            new_input_text  = d.get("new_input_text"),
            new_target_text = d.get("new_target_text"),
            reviewer_notes  = d.get("reviewer_notes"),
            reviewed_at     = d.get("reviewed_at"),
        )


# ---------------------------------------------------------------------------
# ReviewState (the aggregate session object)
# ---------------------------------------------------------------------------

@dataclass
class ReviewState:
    """
    Tracks all reviewer decisions for a single review session.

    Maintains an ordered list of `Sample` objects and a parallel dict of
    `ReviewDecision` objects keyed by `sample_id`.
    """

    samples:   List[Sample]                       = field(default_factory=list)
    decisions: Dict[int, ReviewDecision]          = field(default_factory=dict)
    source_path: str                              = ""
    session_started_at: str                       = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_decisions(self) -> None:
        """Create UNREVIEWED stubs for any sample not yet in decisions."""
        for s in self.samples:
            if s.sample_id not in self.decisions:
                self.decisions[s.sample_id] = ReviewDecision(
                    sample_id=s.sample_id
                )

    def load_samples(self, samples: List[Sample]) -> None:
        self.samples = samples
        self._ensure_decisions()

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_decision(self, sample_id: int) -> ReviewDecision:
        return self.decisions[sample_id]

    def get_sample(self, sample_id: int) -> Sample:
        for s in self.samples:
            if s.sample_id == sample_id:
                return s
        raise KeyError(f"sample_id {sample_id} not found")

    # ------------------------------------------------------------------
    # Filtered / sorted views
    # ------------------------------------------------------------------

    def filter(
        self,
        reason: Optional[str] = None,
        action: Optional[ActionType] = None,
    ) -> List[Sample]:
        """Return a filtered subset of samples."""
        result = list(self.samples)
        if reason:
            result = [s for s in result if s.reason == reason]
        if action is not None:
            result = [
                s for s in result
                if self.decisions[s.sample_id].action == action
            ]
        return result

    def sorted_samples(
        self,
        by: SortField = SortField.GRAD_NORM,
        descending: bool = True,
    ) -> List[Sample]:
        key_fn = {
            SortField.GRAD_NORM: lambda s: s.grad_norm,
            SortField.LOSS:      lambda s: s.loss,
            SortField.SAMPLE_ID: lambda s: s.sample_id,
            SortField.EPOCH:     lambda s: s.epoch,
        }[by]
        return sorted(self.samples, key=key_fn, reverse=descending)

    def pending(self) -> List[Sample]:
        """Samples not yet reviewed."""
        return self.filter(action=ActionType.UNREVIEWED)

    def next_pending(self, after_id: Optional[int] = None) -> Optional[Sample]:
        """
        Return the next UNREVIEWED sample, optionally starting after
        `after_id` in the current natural order.
        """
        ids = [s.sample_id for s in self.pending()]
        if not ids:
            return None
        if after_id is None or after_id not in ids:
            return self.get_sample(ids[0])
        idx = ids.index(after_id)
        next_id = ids[idx + 1] if idx + 1 < len(ids) else ids[0]
        return self.get_sample(next_id)

    # ------------------------------------------------------------------
    # Progress statistics
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def reviewed_count(self) -> int:
        return sum(1 for d in self.decisions.values() if d.is_reviewed)

    @property
    def pending_count(self) -> int:
        return self.total - self.reviewed_count

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {a.value: 0 for a in ActionType}
        for d in self.decisions.values():
            counts[d.action.value] += 1
        counts["total"] = self.total
        return counts

    # ------------------------------------------------------------------
    # Distinct values (for filter dropdowns)
    # ------------------------------------------------------------------

    def distinct_reasons(self) -> List[str]:
        return sorted({s.reason for s in self.samples})

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return self.total

    # ------------------------------------------------------------------
    # Serialisation (progress snapshot — NOT the export format)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "source_path":        self.source_path,
            "session_started_at": self.session_started_at,
            "decisions":          [d.to_dict() for d in self.decisions.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewState":
        state = cls(
            source_path=d.get("source_path", ""),
            session_started_at=d.get("session_started_at", ""),
        )
        for dec_dict in d.get("decisions", []):
            dec = ReviewDecision.from_dict(dec_dict)
            state.decisions[dec.sample_id] = dec
        return state

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
