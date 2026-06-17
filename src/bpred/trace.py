"""Trace-driven simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class BranchPredictor(Protocol):
    """Structural protocol for any predictor accepted by run_trace."""

    def predict(self, *, pc: int) -> bool: ...

    def update(self, *, pc: int, taken: bool) -> None: ...


@dataclass(frozen=True)
class TraceResult:
    """Result of running a branch trace through a predictor.

    Attributes:
        predictions: Ordered list of predictions made before each update.
        correct:     Whether each prediction matched the actual outcome.
        total:       Total number of branches in the trace.
        hits:        Number of correctly predicted branches.
    """

    predictions: list[bool]
    correct: list[bool]
    total: int
    hits: int


def run_trace(
    predictor: BranchPredictor,
    *,
    trace: list[tuple[int, bool]],
) -> TraceResult:
    """Feed (pc, taken) pairs through *predictor* and collect statistics.

    For each entry the prediction is recorded *before* the predictor is
    updated, matching real hardware behaviour.

    Args:
        predictor: Any object with ``predict`` and ``update`` methods.
        trace:     Ordered list of (pc, taken) pairs.

    Returns:
        A :class:`TraceResult` with per-branch predictions and aggregate counts.
    """
    predictions: list[bool] = []
    correct: list[bool] = []
    hits = 0

    for pc, taken in trace:
        pred = predictor.predict(pc=pc)
        hit = pred == taken
        predictions.append(pred)
        correct.append(hit)
        if hit:
            hits += 1
        predictor.update(pc=pc, taken=taken)

    return TraceResult(
        predictions=predictions,
        correct=correct,
        total=len(trace),
        hits=hits,
    )


def accuracy(*, trace_result: TraceResult) -> float:
    """Return prediction accuracy as a fraction in [0.0, 1.0].

    Args:
        trace_result: Result from :func:`run_trace`.

    Returns:
        hits / total, or 0.0 for an empty trace.
    """
    if trace_result.total == 0:
        return 0.0
    return trace_result.hits / trace_result.total


def mispredictions(*, trace_result: TraceResult) -> int:
    """Return the number of mispredicted branches.

    Args:
        trace_result: Result from :func:`run_trace`.

    Returns:
        total - hits
    """
    return trace_result.total - trace_result.hits
