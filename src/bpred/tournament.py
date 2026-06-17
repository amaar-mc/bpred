"""Tournament (hybrid) branch predictor.

Reference: S. McFarling, "Combining Branch Predictors," WRL Technical Note
TN-36, Digital Equipment Corporation Western Research Laboratory, June 1993.

The design follows the Alpha 21264 tournament predictor: a meta-selector table
of saturating counters chooses between a local and a global sub-predictor.
The chooser is updated only when the two sub-predictors disagree, biasing it
toward whichever was correct.
"""

from __future__ import annotations

from typing import Protocol

from bpred.counter import SaturatingCounter


class BranchPredictor(Protocol):
    """Structural protocol satisfied by BimodalPredictor and GsharePredictor."""

    @property
    def table_size(self) -> int: ...

    def predict(self, *, pc: int) -> bool: ...

    def update(self, *, pc: int, taken: bool) -> None: ...


class TournamentPredictor:
    """Meta-predictor that combines a local and a global sub-predictor.

    The chooser table contains *meta_bits*-bit saturating counters.  A low
    chooser value (< threshold) means "trust the local predictor"; a high
    value (>= threshold) means "trust the global predictor."

    Chooser update rule (classic Alpha 21264):
    - Both correct or both wrong: no update.
    - Only local correct: decrement chooser toward 0 (favor local).
    - Only global correct: increment chooser toward max (favor global).

    Both sub-predictors are always updated with the actual outcome,
    regardless of which one the meta-selector chose.
    """

    _local: BranchPredictor
    _global: BranchPredictor
    _chooser: list[SaturatingCounter]
    _chooser_size: int
    _meta_bits: int
    _threshold: int

    def __init__(
        self,
        *,
        local: BranchPredictor,
        global_: BranchPredictor,
        meta_bits: int,
    ) -> None:
        if meta_bits < 1:
            raise ValueError(f"meta_bits must be >= 1, got {meta_bits}")
        self._local = local
        self._global = global_
        self._meta_bits = meta_bits
        self._threshold = 1 << (meta_bits - 1)
        # Chooser table sized to the larger sub-predictor table.
        self._chooser_size = max(local.table_size, global_.table_size)
        # Start neutral: weakly global (threshold value).
        initial = self._threshold
        self._chooser = [
            SaturatingCounter(bits=meta_bits, initial=initial)
            for _ in range(self._chooser_size)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def table_size(self) -> int:
        """Size of the meta-selector chooser table."""
        return self._chooser_size

    @property
    def meta_bits(self) -> int:
        """Bit-width of each chooser counter."""
        return self._meta_bits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chooser_index(self, *, pc: int) -> int:
        return pc % self._chooser_size

    def _use_global(self, *, pc: int) -> bool:
        """Return True when the chooser favors the global sub-predictor."""
        idx = self._chooser_index(pc=pc)
        return self._chooser[idx].value >= self._threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, *, pc: int) -> bool:
        """Return the prediction chosen by the meta-selector."""
        if self._use_global(pc=pc):
            return self._global.predict(pc=pc)
        return self._local.predict(pc=pc)

    def update(self, *, pc: int, taken: bool) -> None:
        """Update both sub-predictors and the chooser.

        The chooser is updated only when the two sub-predictors disagree.
        """
        local_pred = self._local.predict(pc=pc)
        global_pred = self._global.predict(pc=pc)

        # Update chooser only on disagreement.
        if local_pred != global_pred:
            local_correct = local_pred == taken
            global_correct = global_pred == taken
            idx = self._chooser_index(pc=pc)
            if local_correct and not global_correct:
                self._chooser[idx].decrement()
            elif global_correct and not local_correct:
                self._chooser[idx].increment()

        # Always update both sub-predictors.
        self._local.update(pc=pc, taken=taken)
        self._global.update(pc=pc, taken=taken)

    def __repr__(self) -> str:
        return (
            f"TournamentPredictor(local={self._local!r}, "
            f"global_={self._global!r}, meta_bits={self._meta_bits})"
        )
