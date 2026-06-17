"""Property-style tests for bpred.

These tests verify algebraic properties that must hold for all inputs,
exercising many combinations without requiring an external library.
"""

from bpred.bimodal import BimodalPredictor
from bpred.counter import SaturatingCounter
from bpred.gshare import GsharePredictor

# ---------------------------------------------------------------------------
# SaturatingCounter properties
# ---------------------------------------------------------------------------


def test_counter_always_in_range() -> None:
    """A 2-bit counter stays in [0, 3] under any sequence of updates."""
    for initial in range(4):
        c = SaturatingCounter(bits=2, initial=initial)
        for taken in [True, False, True, True, False, False, False, True]:
            c.update(taken=taken)
            assert 0 <= c.value <= 3, f"value {c.value} out of range"


def test_counter_saturates_at_max() -> None:
    """Incrementing past max must leave the counter at max."""
    for bits in range(1, 5):
        c = SaturatingCounter(bits=bits, initial=(1 << bits) - 1)
        max_val = c.max_value
        for _ in range(10):
            c.increment()
            assert c.value == max_val


def test_counter_saturates_at_zero() -> None:
    """Decrementing past 0 must leave the counter at 0."""
    for bits in range(1, 5):
        c = SaturatingCounter(bits=bits, initial=0)
        for _ in range(10):
            c.decrement()
            assert c.value == 0


def test_predict_deterministic_given_state() -> None:
    """Same counter state always gives the same prediction.

    For any bits, any initial value, and any sequence of updates, predict()
    must be a pure function of current state -- calling it twice without any
    intervening update returns the same result.
    """
    for bits in range(1, 5):
        for initial in range(1 << bits):
            c = SaturatingCounter(bits=bits, initial=initial)
            first = c.predict()
            second = c.predict()
            assert first == second, (
                f"non-deterministic predict for bits={bits}, initial={initial}"
            )


def test_bimodal_predict_deterministic_given_state() -> None:
    """BimodalPredictor.predict(pc) returns the same result when state unchanged."""
    for table_size in [4, 8, 16]:
        p = BimodalPredictor(counter_bits=2, table_size=table_size)
        for pc in range(table_size * 2):
            first = p.predict(pc=pc)
            second = p.predict(pc=pc)
            assert first == second, (
                f"non-deterministic predict for pc={pc}, table_size={table_size}"
            )


def test_gshare_predict_deterministic_given_state() -> None:
    """GsharePredictor.predict(pc) is deterministic: same state => same result."""
    p = GsharePredictor(history_bits=4, table_size=16)
    for pc in range(32):
        first = p.predict(pc=pc)
        second = p.predict(pc=pc)
        assert first == second, f"non-deterministic predict for pc={pc}"


def test_counter_monotone_increment() -> None:
    """Incrementing a counter never decreases its value."""
    for bits in range(1, 5):
        c = SaturatingCounter(bits=bits, initial=0)
        prev = c.value
        for _ in range((1 << bits) + 2):
            c.increment()
            assert c.value >= prev
            prev = c.value


def test_counter_monotone_decrement() -> None:
    """Decrementing a counter never increases its value."""
    for bits in range(1, 5):
        c = SaturatingCounter(bits=bits, initial=(1 << bits) - 1)
        prev = c.value
        for _ in range((1 << bits) + 2):
            c.decrement()
            assert c.value <= prev
            prev = c.value


def test_prediction_matches_threshold() -> None:
    """predict() must be True iff value >= threshold for all bit widths."""
    for bits in range(1, 5):
        threshold = 1 << (bits - 1)
        for initial in range(1 << bits):
            c = SaturatingCounter(bits=bits, initial=initial)
            expected = initial >= threshold
            assert c.predict() == expected, (
                f"bits={bits}, initial={initial}: expected {expected}, "
                f"got {c.predict()}"
            )
