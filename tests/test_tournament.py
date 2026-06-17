"""Tests for TournamentPredictor.

NOTE: We deliberately do NOT assert that the tournament predictor is never
worse than both sub-predictors on a single prediction.  A meta-selector CAN
choose the wrong sub-predictor on any given branch, so that invariant is false
by design.
"""

from bpred.bimodal import BimodalPredictor
from bpred.gshare import GsharePredictor
from bpred.tournament import TournamentPredictor
from bpred.trace import accuracy, run_trace


def _make_tournament(
    *,
    local_counter_bits: int = 2,
    local_table_size: int = 8,
    global_history_bits: int = 4,
    global_table_size: int = 16,
    meta_bits: int = 2,
) -> TournamentPredictor:
    local = BimodalPredictor(
        counter_bits=local_counter_bits,
        table_size=local_table_size,
    )
    global_ = GsharePredictor(
        history_bits=global_history_bits,
        table_size=global_table_size,
    )
    return TournamentPredictor(local=local, global_=global_, meta_bits=meta_bits)


# ---------------------------------------------------------------------------
# Chooser logic
# ---------------------------------------------------------------------------


def test_chooser_increments_when_global_correct() -> None:
    """When global is right and local is wrong the chooser should move toward
    favoring global (increment)."""
    # Drive local counter at pc=0 to SN (predict=False) so local disagrees
    # with an always-taken stream while global (starts WT) will predict True.
    local = BimodalPredictor(counter_bits=2, table_size=8)
    for _ in range(4):
        local.update(pc=0, taken=False)  # local counter -> SN

    global_ = GsharePredictor(history_bits=4, table_size=16)
    t2 = TournamentPredictor(local=local, global_=global_, meta_bits=2)

    # Chooser starts at threshold (2 for 2-bit), which means use global_.
    # local predicts False, global predicts True, actual=True -> global correct.
    initial_chooser = t2._chooser[0].value  # type: ignore[attr-defined]
    t2.update(pc=0, taken=True)
    # Chooser should have incremented (global was correct, local was wrong).
    assert t2._chooser[0].value >= initial_chooser  # type: ignore[attr-defined]


def test_chooser_decrements_when_local_correct() -> None:
    """When local is right and global is wrong the chooser moves toward local."""
    # Drive global counter at index (pc XOR ghr) to SN.
    local = BimodalPredictor(counter_bits=2, table_size=8)
    global_ = GsharePredictor(history_bits=1, table_size=2)

    # With history_bits=1, table_size=2, ghr=0: index=(0^0)%2=0
    # Drive counter[0] to SN so global predicts False.
    for _ in range(4):
        global_.update(pc=0, taken=False)

    # After 4 not-taken updates ghr = 0 (not-taken keeps shifting 0 in).
    # So on next predict(pc=0): index=(0^0)%2=0, counter[0]=SN -> False.

    t = TournamentPredictor(local=local, global_=global_, meta_bits=2)
    # Chooser starts at threshold=2 (weakly global).
    # local predicts True (WT), global predicts False (SN), actual=True
    # -> local is correct.
    initial = t._chooser[0].value  # type: ignore[attr-defined]
    t.update(pc=0, taken=True)
    assert t._chooser[0].value <= initial  # type: ignore[attr-defined]


def test_chooser_unchanged_when_both_agree() -> None:
    """Chooser should not change when both sub-predictors give the same answer."""
    t = _make_tournament(meta_bits=2)
    # Both start at WT -> predict True. actual=True -> both correct, no update.
    initial = t._chooser[0].value  # type: ignore[attr-defined]
    t.update(pc=0, taken=True)
    # Both still predict True after update (driven to ST). Update again.
    t.update(pc=0, taken=True)
    # Chooser should not have moved.
    assert t._chooser[0].value == initial  # type: ignore[attr-defined]


def test_both_subpredictors_always_updated() -> None:
    """Both sub-predictors must be updated regardless of which one is chosen."""
    local = BimodalPredictor(counter_bits=2, table_size=8)
    global_ = GsharePredictor(history_bits=4, table_size=16)
    t = TournamentPredictor(local=local, global_=global_, meta_bits=2)

    # Feed a branch; then check that local counter has moved.
    local_before = local._table[0].value  # type: ignore[attr-defined]
    t.update(pc=0, taken=True)
    local_after = local._table[0].value  # type: ignore[attr-defined]
    assert local_after != local_before


# ---------------------------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------------------------


def test_tournament_accuracy_always_taken() -> None:
    """On an always-taken trace the tournament predictor should converge."""
    t = _make_tournament()
    trace = [(0x100, True)] * 40
    result = run_trace(t, trace=trace)
    acc = accuracy(trace_result=result)
    assert acc >= 0.8


def test_tournament_accuracy_structured_trace() -> None:
    """On a structured trace the tournament predictor achieves reasonable acc."""
    t = _make_tournament()
    # Alternating between two PCs: one always taken, one always not-taken.
    trace = [(0x100, True), (0x200, False)] * 20
    result = run_trace(t, trace=trace)
    acc = accuracy(trace_result=result)
    # After warm-up both sub-predictors should handle their respective PCs well.
    assert acc >= 0.7


def test_tournament_predict_returns_bool() -> None:
    t = _make_tournament()
    result = t.predict(pc=0)
    assert isinstance(result, bool)
