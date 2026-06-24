"""Local-history two-level adaptive branch predictor (PAg).

Reference: T.-Y. Yeh and Y. N. Patt, "Two-Level Adaptive Training Branch
Prediction," in Proceedings of the 24th Annual International Symposium on
Microarchitecture (MICRO 24), pp. 51-61, 1991.

This implements the PAg configuration of the Yeh and Patt two-level scheme:

- A *per-address* branch history table (BHT) indexed by PC.  Each BHT entry is
  an N-bit shift register holding the recent taken/not-taken outcomes of that
  specific branch (P = per-address first level).
- A single *global* pattern history table (PHT) of 2-bit saturating counters,
  shared across all branches and indexed by the local history pattern read from
  the BHT (g = global second level).

The "PAg" name decodes as: P (per-address first level) A (adaptive) g (global
second level).  Contrast with the per-address second level "PAp" variant, noted
as future work below.

Indexing
--------
The local history register is an integer in [0, 2^history_bits - 1].  The BHT
is indexed by ``pc % bht_size``.  The PHT is indexed by ``history % pht_size``;
with the natural sizing ``pht_size == 2**history_bits`` this is a direct,
collision-free mapping from every distinct history pattern to its own counter.

PAp (future work)
-----------------
A PAp predictor gives each branch its own private PHT instead of sharing one
global PHT.  That is a strict generalisation (a 2-D PHT indexed by branch and
then by pattern) and is intentionally out of scope here to keep PAg focused.
"""

from __future__ import annotations

from bpred.counter import SaturatingCounter

_COUNTER_BITS = 2


class LocalHistoryPredictor:
    """Two-level adaptive predictor with per-branch local history (PAg).

    Each branch has its own ``history_bits``-wide local history register stored
    in the branch history table (BHT).  A single shared pattern history table
    (PHT) of 2-bit saturating counters is indexed by that local history pattern.

    predict(pc):
        Read the branch's local history, index the PHT with it, and predict
        taken iff the selected counter is in a taken state.

    update(pc, taken):
        Update the indexed PHT counter with the actual outcome, then shift the
        outcome into that branch's local history register (most recent outcome
        in the least-significant bit).
    """

    _bht: list[int]
    _pht: list[SaturatingCounter]
    _history_bits: int
    _bht_size: int
    _pht_size: int
    _history_mask: int

    def __init__(self, *, history_bits: int, bht_size: int, pht_size: int) -> None:
        """Create a LocalHistoryPredictor (PAg).

        Args:
            history_bits: Width N of each per-branch local history register.
                Must be >= 1.
            bht_size:     Number of entries in the branch history table (BHT).
                Must be >= 1.  Indexed by ``pc % bht_size``.
            pht_size:     Number of 2-bit counters in the shared pattern history
                table (PHT).  Must be >= 1.  Indexed by ``history % pht_size``;
                use ``2**history_bits`` for a collision-free mapping.
        """
        if history_bits < 1:
            raise ValueError(f"history_bits must be >= 1, got {history_bits}")
        if bht_size < 1:
            raise ValueError(f"bht_size must be >= 1, got {bht_size}")
        if pht_size < 1:
            raise ValueError(f"pht_size must be >= 1, got {pht_size}")

        self._history_bits = history_bits
        self._bht_size = bht_size
        self._pht_size = pht_size
        self._history_mask = (1 << history_bits) - 1
        # All local histories start empty (all not-taken).
        self._bht = [0] * bht_size
        # All PHT counters start weakly taken (neutral).
        initial = 1 << (_COUNTER_BITS - 1)
        self._pht = [
            SaturatingCounter(bits=_COUNTER_BITS, initial=initial)
            for _ in range(pht_size)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history_bits(self) -> int:
        """Width N of each per-branch local history register."""
        return self._history_bits

    @property
    def bht_size(self) -> int:
        """Number of entries in the branch history table."""
        return self._bht_size

    @property
    def pht_size(self) -> int:
        """Number of counters in the shared pattern history table."""
        return self._pht_size

    @property
    def table_size(self) -> int:
        """Pattern history table size (satisfies the BranchPredictor protocol)."""
        return self._pht_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bht_index(self, *, pc: int) -> int:
        """BHT index: pc mod bht_size."""
        return pc % self._bht_size

    def _local_history(self, *, pc: int) -> int:
        """Return the local history pattern for the branch at *pc*."""
        return self._bht[self._bht_index(pc=pc)]

    def _pht_index(self, *, history: int) -> int:
        """PHT index: history mod pht_size."""
        return history % self._pht_size

    def _shift_history(self, *, pc: int, taken: bool) -> None:
        """Shift *taken* into the LSB of the branch's local history register."""
        idx = self._bht_index(pc=pc)
        self._bht[idx] = ((self._bht[idx] << 1) | int(taken)) & self._history_mask

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, *, pc: int) -> bool:
        """Return the taken prediction for the branch at *pc*."""
        history = self._local_history(pc=pc)
        return self._pht[self._pht_index(history=history)].predict()

    def update(self, *, pc: int, taken: bool) -> None:
        """Update the PHT counter, then shift the outcome into local history."""
        history = self._local_history(pc=pc)
        self._pht[self._pht_index(history=history)].update(taken=taken)
        self._shift_history(pc=pc, taken=taken)

    def __repr__(self) -> str:
        return (
            f"LocalHistoryPredictor(history_bits={self._history_bits}, "
            f"bht_size={self._bht_size}, pht_size={self._pht_size})"
        )
