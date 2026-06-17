# bpred -- Project Instructions

## What this is

Pure-Python simulator of classical CPU branch predictors (bimodal, gshare,
tournament). Zero runtime dependencies. Target: Python 3.10+.

## Source layout

```
src/bpred/
  counter.py    -- SaturatingCounter (shared building block)
  bimodal.py    -- BimodalPredictor
  gshare.py     -- GsharePredictor
  tournament.py -- TournamentPredictor + BranchPredictor protocol
  trace.py      -- run_trace, accuracy, mispredictions, TraceResult
  cli.py        -- argparse CLI entry point
  __init__.py   -- public re-exports
  py.typed      -- PEP 561 marker
tests/
  test_counter.py    -- golden FSM values + property tests for counter
  test_bimodal.py
  test_gshare.py
  test_tournament.py
  test_property.py   -- property-style exhaustive tests
  test_trace.py
```

## Conventions

- No default parameter values anywhere. All parameters are keyword-only.
- Strict mypy passes on all files in src/.
- ruff lint set: E, F, I, UP, ANN.
- No runtime dependencies (stdlib only).
- No em dash characters in any file.
- Commits: `type(scope): description`. No Co-authored-by trailers.

## Running checks

```bash
pytest -q
ruff check .
mypy src
uv build
uv run --with twine twine check dist/*
```

## Predictor invariants

- SaturatingCounter value always stays in [0, max_value].
- predict() is a pure function of current state.
- Gshare GHR is updated AFTER recording the prediction.
- Tournament: chooser updated only when sub-predictors disagree.
- Tournament: both sub-predictors always updated regardless of chooser.

## What NOT to assert in tests

Do NOT assert that tournament is never worse than both sub-predictors on a
single prediction. A meta-selector can choose the wrong sub-predictor.
