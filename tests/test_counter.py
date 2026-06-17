"""Tests for SaturatingCounter.

Golden FSM trace for a 2-bit counter (state names: SN=0, WN=1, WT=2, ST=3):

  Start   : state=2 (WT), predict=True
  taken   : state=3 (ST),  predict=True
  taken   : state=3 (ST),  predict=True   [saturates -- stays at 3]
  notaken : state=2 (WT),  predict=True
  notaken : state=1 (WN),  predict=False  [prediction flips here]
  notaken : state=0 (SN),  predict=False  [saturates -- stays at 0]
  taken   : state=1 (WN),  predict=False
"""

from bpred.counter import SaturatingCounter


def _make_2bit(*, initial: int) -> SaturatingCounter:
    return SaturatingCounter(bits=2, initial=initial)


# ---------------------------------------------------------------------------
# Golden FSM values
# ---------------------------------------------------------------------------


def test_2bit_golden_fsm() -> None:
    """Walk the full golden FSM trace documented above."""
    c = _make_2bit(initial=2)  # start at WT

    # Start: state=2, predict=True
    assert c.value == 2
    assert c.predict() is True

    # taken -> state=3 (ST)
    c.update(taken=True)
    assert c.value == 3
    assert c.predict() is True

    # taken again -> state=3 (saturates)
    c.update(taken=True)
    assert c.value == 3
    assert c.predict() is True

    # notaken -> state=2 (WT)
    c.update(taken=False)
    assert c.value == 2
    assert c.predict() is True

    # notaken -> state=1 (WN), prediction flips
    c.update(taken=False)
    assert c.value == 1
    assert c.predict() is False

    # notaken -> state=0 (SN, saturates)
    c.update(taken=False)
    assert c.value == 0
    assert c.predict() is False

    # taken -> state=1 (WN)
    c.update(taken=True)
    assert c.value == 1
    assert c.predict() is False


def test_2bit_requires_two_mispredicts_to_flip() -> None:
    """A 2-bit counter requires two consecutive wrong outcomes to flip prediction.

    Start at ST (state=3, predict=True).
    One not-taken -> WT (still predict=True, no flip yet).
    Second not-taken -> WN (predict=False, flip occurs).
    """
    c = _make_2bit(initial=3)  # ST, predict=True
    assert c.predict() is True

    c.update(taken=False)  # ST -> WT
    assert c.predict() is True  # still True

    c.update(taken=False)  # WT -> WN
    assert c.predict() is False  # now flipped


def test_2bit_saturates_at_max() -> None:
    c = _make_2bit(initial=3)
    c.increment()
    assert c.value == 3


def test_2bit_saturates_at_zero() -> None:
    c = _make_2bit(initial=0)
    c.decrement()
    assert c.value == 0


def test_1bit_counter() -> None:
    c = SaturatingCounter(bits=1, initial=0)
    assert c.max_value == 1
    assert c.predict() is False
    c.update(taken=True)
    assert c.value == 1
    assert c.predict() is True
    c.update(taken=True)  # saturate
    assert c.value == 1


def test_counter_range_never_violated() -> None:
    """Counter must stay in [0, max_value] through many updates."""
    c = SaturatingCounter(bits=3, initial=4)
    for _ in range(20):
        c.update(taken=True)
        assert 0 <= c.value <= c.max_value
    for _ in range(20):
        c.update(taken=False)
        assert 0 <= c.value <= c.max_value
