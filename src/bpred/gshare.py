"""Gshare branch predictor.

Reference: S. McFarling, "Combining Branch Predictors," WRL Technical Note
TN-36, Digital Equipment Corporation Western Research Laboratory, June 1993.
"""

from __future__ import annotations

from bpred.counter import SaturatingCounter

_COUNTER_BITS = 2


class GsharePredictor:
    """Global-history XOR-indexed branch predictor (gshare).

    A global history register (GHR) of *history_bits* bits is XOR'd with the
    lower *history_bits* bits of the PC to index into a prediction table of
    2-bit saturating counters.

    After each branch the actual outcome is shifted into the GHR (MSB first,
    keeping only the most recent *history_bits* outcomes).
    """

    _table: list[SaturatingCounter]
    _table_size: int
    _history_bits: int
    _history_mask: int
    _ghr: int  # global history register

    def __init__(self, *, history_bits: int, table_size: int) -> None:
        if history_bits < 1:
            raise ValueError(f"history_bits must be >= 1, got {history_bits}")
        if table_size < 1:
            raise ValueError(f"table_size must be >= 1, got {table_size}")
        self._history_bits = history_bits
        self._history_mask = (1 << history_bits) - 1
        self._table_size = table_size
        self._ghr = 0
        initial = 1 << (_COUNTER_BITS - 1)  # weakly taken
        self._table = [
            SaturatingCounter(bits=_COUNTER_BITS, initial=initial)
            for _ in range(table_size)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def table_size(self) -> int:
        """Number of entries in the prediction table."""
        return self._table_size

    @property
    def history_bits(self) -> int:
        """Width of the global history register."""
        return self._history_bits

    @property
    def ghr(self) -> int:
        """Current value of the global history register."""
        return self._ghr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index(self, *, pc: int) -> int:
        """Compute table index: (pc XOR ghr) mod table_size."""
        return (pc ^ self._ghr) % self._table_size

    def _shift_history(self, *, taken: bool) -> None:
        """Shift *taken* into the MSB of the GHR, keeping history_bits."""
        self._ghr = ((self._ghr << 1) | int(taken)) & self._history_mask

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, *, pc: int) -> bool:
        """Return the taken prediction for the branch at *pc*."""
        return self._table[self._index(pc=pc)].predict()

    def update(self, *, pc: int, taken: bool) -> None:
        """Update the counter and GHR for the branch at *pc*."""
        self._table[self._index(pc=pc)].update(taken=taken)
        self._shift_history(taken=taken)

    def __repr__(self) -> str:
        return (
            f"GsharePredictor(history_bits={self._history_bits}, "
            f"table_size={self._table_size})"
        )
