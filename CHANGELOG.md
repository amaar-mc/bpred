# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-17

### Added

- `PerceptronPredictor`: Jimenez and Lin (2001) table of integer-weight perceptrons.
  Captures linearly-separable history patterns that bimodal and gshare cannot learn.
  Constructor: `PerceptronPredictor(history_length=H, table_size=N)`.
  Exported from `bpred` top-level package.

## [0.1.0] - 2026-06-17

### Added

- `SaturatingCounter`: n-bit saturating counter building block.
- `BimodalPredictor`: Smith (1981) table of saturating counters indexed by PC.
- `GsharePredictor`: McFarling (1993) global-history XOR predictor.
- `TournamentPredictor`: McFarling (1993) / Alpha 21264-style meta-selecting predictor.
- `run_trace`, `accuracy`, `mispredictions`: trace-driven simulation utilities.
- `bpred` CLI supporting bimodal, gshare, and tournament subcommands.
- Full mypy strict and ruff linting.
- CI on Python 3.10, 3.11, 3.12, 3.13.
