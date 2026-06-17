"""Perceptron branch predictor.

Reference: D. A. Jimenez and C. Lin, "Dynamic Branch Prediction with
Perceptrons," in Proceedings of the 7th International Symposium on High-
Performance Computer Architecture (HPCA), pp. 197-206, January 2001.

Each perceptron is a vector of integer weights.  The dot product of those
weights with the history vector (bias + H history bits) gives the confidence
and direction of the prediction.  Weight updates use the perceptron learning
rule, gated by a threshold theta to prevent over-training on easy branches.

Training threshold (Jimenez & Lin, Section 3.3):
    theta = floor(1.93 * H + 14)
where H is the history length.  This integer form is the standard used in
hardware and simulation; it derives from the optimal threshold analysis in
the paper.

History encoding:
    taken   -> +1
    not-taken -> -1

The bias input x_0 is always +1.

Weights are stored as plain Python ints and are left unclamped (unbounded).
Hardware implementations clamp to a signed fixed-point range for area
efficiency, but clamping is orthogonal to correctness and is omitted here
to keep the simulator transparent and exact.
"""

from __future__ import annotations

import math


class PerceptronPredictor:
    """Table of perceptrons for branch prediction (Jimenez and Lin 2001).

    Each perceptron is a list of (H + 1) integer weights: w[0] is the bias
    weight (multiplied by x_0 = +1 always) and w[1..H] are multiplied by the
    corresponding bits of the global history register (GHR).

    The table is indexed by pc % table_size.  The GHR is an integer where
    bit position i (0 = most recent) stores the outcome of the i-th most
    recent branch: 1 for taken, 0 for not-taken.  When computing the dot
    product the bits are converted to {+1, -1} on the fly.
    """

    _table: list[list[int]]
    _table_size: int
    _history_length: int
    _theta: int
    _ghr: int         # packed history; bit 0 = most recent outcome
    _history_mask: int

    def __init__(self, *, history_length: int, table_size: int) -> None:
        """Create a PerceptronPredictor.

        Args:
            history_length: Number of history bits H.  Must be >= 1.
            table_size:     Number of perceptrons in the table.  Must be >= 1.
        """
        if history_length < 1:
            raise ValueError(
                f"history_length must be >= 1, got {history_length}"
            )
        if table_size < 1:
            raise ValueError(f"table_size must be >= 1, got {table_size}")

        self._history_length = history_length
        self._table_size = table_size
        # Standard threshold from Jimenez & Lin Section 3.3:
        #   theta = floor(1.93 * H + 14)
        self._theta = math.floor(1.93 * history_length + 14)
        self._history_mask = (1 << history_length) - 1
        self._ghr = 0
        # All weights initialised to 0: bias and history weights start neutral.
        self._table = [
            [0] * (history_length + 1) for _ in range(table_size)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def table_size(self) -> int:
        """Number of entries (perceptrons) in the table."""
        return self._table_size

    @property
    def history_length(self) -> int:
        """Number of global history bits H."""
        return self._history_length

    @property
    def theta(self) -> int:
        """Training threshold: floor(1.93 * H + 14)."""
        return self._theta

    @property
    def ghr(self) -> int:
        """Current value of the global history register (raw packed int)."""
        return self._ghr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index(self, *, pc: int) -> int:
        """Table index: pc mod table_size."""
        return pc % self._table_size

    def _history_vector(self) -> list[int]:
        """Return the current history as a list of +1/-1 values.

        x[0] is the bias (always +1).
        x[i] for i in 1..H corresponds to the (i-1)-th most recent branch,
        where bit (i-1) of _ghr is 1 (taken) or 0 (not-taken).
        """
        x: list[int] = [1]  # x[0] = bias = +1
        for i in range(self._history_length):
            bit = (self._ghr >> i) & 1
            x.append(1 if bit else -1)
        return x

    def _dot(self, *, weights: list[int]) -> int:
        """Compute y = sum(w[i] * x[i]) using current history."""
        x = self._history_vector()
        total = 0
        for w, xi in zip(weights, x):
            total += w * xi
        return total

    def _shift_history(self, *, taken: bool) -> None:
        """Shift the new outcome into the GHR as the most recent bit."""
        self._ghr = ((self._ghr << 1) | int(taken)) & self._history_mask

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, *, pc: int) -> bool:
        """Return the taken prediction for the branch at *pc*.

        Computes y = w dot x; predicts taken if y >= 0.
        """
        idx = self._index(pc=pc)
        y = self._dot(weights=self._table[idx])
        return y >= 0

    def update(self, *, pc: int, taken: bool) -> None:
        """Update the perceptron for *pc* with the actual outcome.

        Training rule (Jimenez & Lin):
          t = +1 if taken, -1 if not-taken.
          Compute y = w dot x.
          If prediction was wrong OR |y| <= theta:
              for each i: w[i] = w[i] + t * x[i]
          Shift taken into the GHR after training.

        The GHR is updated AFTER training so that the same history vector
        used during prediction is also used during training.
        """
        idx = self._index(pc=pc)
        weights = self._table[idx]
        x = self._history_vector()
        y = self._dot(weights=weights)

        predicted_taken = y >= 0
        t = 1 if taken else -1

        # Train when prediction was wrong or confidence is below theta.
        if predicted_taken != taken or abs(y) <= self._theta:
            for i in range(len(weights)):
                weights[i] += t * x[i]

        self._shift_history(taken=taken)

    def __repr__(self) -> str:
        return (
            f"PerceptronPredictor(history_length={self._history_length}, "
            f"table_size={self._table_size})"
        )
