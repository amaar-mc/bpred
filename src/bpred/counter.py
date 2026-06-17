"""Saturating counter used as the base building block for all branch predictors."""

from __future__ import annotations


class SaturatingCounter:
    """An n-bit saturating counter in the range [0, 2^n - 1].

    Prediction is *taken* when the value is >= 2^(n-1).
    Incrementing toward 2^n - 1 models a taken branch; decrementing toward 0
    models a not-taken branch.  At the extremes the counter saturates rather
    than wrapping.

    The n=1 special case produces a 1-bit predictor with states {0, 1}.
    """

    _value: int
    _max: int
    _threshold: int

    def __init__(self, *, bits: int, initial: int) -> None:
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        self._max = (1 << bits) - 1
        self._threshold = 1 << (bits - 1)
        if not (0 <= initial <= self._max):
            raise ValueError(f"initial {initial} out of range [0, {self._max}]")
        self._value = initial

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def value(self) -> int:
        """Current counter value."""
        return self._value

    @property
    def max_value(self) -> int:
        """Maximum value this counter can hold (2^bits - 1)."""
        return self._max

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def predict(self) -> bool:
        """Return True (taken) when value >= threshold."""
        return self._value >= self._threshold

    def increment(self) -> None:
        """Increment toward max_value (saturating)."""
        if self._value < self._max:
            self._value += 1

    def decrement(self) -> None:
        """Decrement toward 0 (saturating)."""
        if self._value > 0:
            self._value -= 1

    def update(self, *, taken: bool) -> None:
        """Increment on taken, decrement on not-taken."""
        if taken:
            self.increment()
        else:
            self.decrement()

    def __repr__(self) -> str:
        return f"SaturatingCounter(value={self._value}, max={self._max})"
