# Project Charter

## Purpose

`bpred` is an open-source, pure-Python simulator of classical CPU branch
predictors intended for use in computer architecture courses and self-directed
study. It provides reference implementations of the bimodal, gshare, and
tournament predictors that students and researchers can read, modify, and
experiment with using real or synthetic branch traces.

## Goals

- Correctness first: every predictor matches the published algorithm exactly.
- Readability: the implementation should be a usable companion to the original
  papers (Smith 1981, McFarling 1993).
- Zero runtime dependencies: install with `pip install bpred` on any Python
  3.10+ environment without pulling additional packages.
- Full type coverage: strict mypy passes on all source files.
- Education-friendly CLI: run a trace and read accuracy results in one command.

## Non-Goals

- Performance: this is a teaching tool, not a production simulator. Pure Python
  is intentional.
- Hardware synthesis or cycle-accurate micro-architectural simulation.
- Support for Python versions below 3.10.

## Scope

The initial release covers three predictors: bimodal, gshare, and tournament.
Future releases may add local-history predictors (PAg, PAp, GAg), TAGE, or
loop predictors, as well as visualisation utilities.

## Relationship to Sibling Projects

`bpred` is developed alongside
[tomasulo](https://github.com/amaar-mc/tomasulo), a pure-Python simulator of
Tomasulo out-of-order execution, and a scoreboarding simulator. Together they
form an open-source library of foundational CPU microarchitecture simulators
for education.

## Governance

Single-maintainer project. Contributions via pull request on GitHub. Issues
are the primary forum for discussion.
