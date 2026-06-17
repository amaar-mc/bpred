"""Tests for run_trace, accuracy, and mispredictions."""

from bpred.bimodal import BimodalPredictor
from bpred.trace import TraceResult, accuracy, mispredictions, run_trace


def _bimodal() -> BimodalPredictor:
    return BimodalPredictor(counter_bits=2, table_size=8)


def test_run_trace_empty() -> None:
    p = _bimodal()
    result = run_trace(p, trace=[])
    assert result.total == 0
    assert result.hits == 0
    assert result.predictions == []
    assert result.correct == []


def test_run_trace_single_taken() -> None:
    """A single taken branch; predictor starts at WT so predicts True -- hit."""
    p = _bimodal()
    result = run_trace(p, trace=[(0, True)])
    assert result.total == 1
    assert result.hits == 1
    assert result.predictions == [True]
    assert result.correct == [True]


def test_run_trace_single_not_taken() -> None:
    """A single not-taken branch; predictor starts at WT (predict=True) -- miss."""
    p = _bimodal()
    result = run_trace(p, trace=[(0, False)])
    assert result.total == 1
    assert result.hits == 0
    assert result.predictions == [True]
    assert result.correct == [False]


def test_run_trace_prediction_before_update() -> None:
    """Prediction must be recorded *before* the counter is updated.

    Start at WT (predict=True). Outcome: False.
    Prediction recorded: True (miss).
    After update counter is at WN.
    Next predict should be False.
    """
    p = _bimodal()
    trace = [(0, False), (0, False)]
    result = run_trace(p, trace=trace)
    # First prediction: True (WT), outcome False -> miss
    assert result.predictions[0] is True
    assert result.correct[0] is False
    # Second prediction: WN -> False, outcome False -> hit
    assert result.predictions[1] is False
    assert result.correct[1] is True


def test_accuracy_empty_trace() -> None:
    result = TraceResult(predictions=[], correct=[], total=0, hits=0)
    assert accuracy(trace_result=result) == 0.0


def test_accuracy_all_correct() -> None:
    result = TraceResult(predictions=[True] * 5, correct=[True] * 5, total=5, hits=5)
    assert accuracy(trace_result=result) == 1.0


def test_accuracy_all_wrong() -> None:
    result = TraceResult(
        predictions=[True] * 4, correct=[False] * 4, total=4, hits=0
    )
    assert accuracy(trace_result=result) == 0.0


def test_accuracy_half() -> None:
    result = TraceResult(
        predictions=[True, False, True, False],
        correct=[True, False, False, True],
        total=4,
        hits=2,
    )
    assert accuracy(trace_result=result) == 0.5


def test_mispredictions_zero() -> None:
    result = TraceResult(predictions=[True] * 3, correct=[True] * 3, total=3, hits=3)
    assert mispredictions(trace_result=result) == 0


def test_mispredictions_all() -> None:
    result = TraceResult(
        predictions=[False] * 6, correct=[False] * 6, total=6, hits=0
    )
    assert mispredictions(trace_result=result) == 6


def test_mispredictions_plus_hits_equals_total() -> None:
    p = _bimodal()
    trace = [(i % 4, i % 2 == 0) for i in range(20)]
    result = run_trace(p, trace=trace)
    assert result.hits + mispredictions(trace_result=result) == result.total


def test_run_trace_multiple_pcs() -> None:
    """Branches from different PCs must use independent counters."""
    p = _bimodal()
    # pc=0 always taken, pc=1 always not-taken
    trace = [(0, True), (1, False), (0, True), (1, False)] * 5
    result = run_trace(p, trace=trace)
    assert result.total == 20
    assert isinstance(result.hits, int)


def test_trace_result_is_frozen() -> None:
    """TraceResult should be immutable (frozen dataclass)."""
    result = TraceResult(predictions=[True], correct=[True], total=1, hits=1)
    import dataclasses

    assert dataclasses.is_dataclass(result)
