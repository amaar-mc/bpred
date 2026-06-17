"""Bimodal (Smith) branch predictor.

Reference: J. E. Smith, "A study of branch prediction strategies," in
Proceedings of the 8th Annual Symposium on Computer Architecture (ISCA),
pp. 135-148, 1981.
"""

from __future__ import annotations

from bpred.counter import SaturatingCounter


class BimodalPredictor:
    """A table of saturating counters indexed by PC mod table_size.

    Each counter independently tracks the taken/not-taken history for
    branches that alias to the same table entry.  With 2-bit counters
    (counter_bits=2) this is the classic 2-bit predictor from Smith 1981.
    """

    _table: list[SaturatingCounter]
    _table_size: int
    _counter_bits: int

    def __init__(self, *, counter_bits: int, table_size: int) -> None:
        if counter_bits < 1:
            raise ValueError(f"counter_bits must be >= 1, got {counter_bits}")
        if table_size < 1:
            raise ValueError(f"table_size must be >= 1, got {table_size}")
        self._counter_bits = counter_bits
        self._table_size = table_size
        # Initialise all counters to weakly taken (threshold - 1 rounded up)
        # so the predictor starts in a neutral "weakly taken" state.
        initial = 1 << (counter_bits - 1)  # weakly taken
        self._table = [
            SaturatingCounter(bits=counter_bits, initial=initial)
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
    def counter_bits(self) -> int:
        """Bit-width of each saturating counter."""
        return self._counter_bits

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, *, pc: int) -> bool:
        """Return the taken prediction for the branch at *pc*."""
        return self._table[pc % self._table_size].predict()

    def update(self, *, pc: int, taken: bool) -> None:
        """Update the counter for the branch at *pc* with the actual outcome."""
        self._table[pc % self._table_size].update(taken=taken)

    def __repr__(self) -> str:
        return (
            f"BimodalPredictor(counter_bits={self._counter_bits}, "
            f"table_size={self._table_size})"
        )
