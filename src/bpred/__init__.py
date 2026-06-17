"""bpred -- Pure-Python CPU branch predictor simulator.

Provides three classical branch predictors and trace-driven simulation
utilities for computer architecture education.

Predictors
----------
BimodalPredictor
    Smith (1981) table of saturating counters indexed by PC.
GsharePredictor
    McFarling (1993) global-history XOR predictor.
TournamentPredictor
    McFarling (1993) / Alpha 21264-style meta-selecting predictor.

Functions
---------
run_trace(predictor, *, trace)
    Feed a list of (pc, taken) pairs through a predictor.
accuracy(*, trace_result)
    Fraction of correctly predicted branches.
mispredictions(*, trace_result)
    Count of incorrectly predicted branches.
"""

from bpred.bimodal import BimodalPredictor
from bpred.gshare import GsharePredictor
from bpred.tournament import TournamentPredictor
from bpred.trace import TraceResult, accuracy, mispredictions, run_trace

__all__ = [
    "BimodalPredictor",
    "GsharePredictor",
    "TournamentPredictor",
    "TraceResult",
    "accuracy",
    "mispredictions",
    "run_trace",
]
