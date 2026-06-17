# Contributing

Thank you for your interest in contributing to `bpred`.

## Setup

```bash
git clone https://github.com/amaar-mc/bpred
cd bpred
pip install -e ".[dev]"
```

## Running checks

```bash
pytest -q          # tests
ruff check .       # linting
mypy src           # type checking
```

All three must pass before opening a pull request.

## Guidelines

- Zero runtime dependencies. All new code must work with the Python standard
  library only.
- Strict typing. All public functions must have complete type annotations, and
  `mypy --strict` must pass.
- No default parameter values on public APIs. All parameters must be explicit
  keyword arguments.
- Test behaviour, not implementation. New predictors must come with tests
  covering correctness on structured traces, not just "it ran without crashing."
- Bug fixes: add a failing test first, then fix the bug.

## Pull requests

- Open an issue first for significant changes.
- Keep commits focused. One logical change per commit.
- Commit message format: `type(scope): description` (e.g.
  `feat(gshare): add history_bits validation`).
- Do not add `Co-authored-by` trailers or automated tool footers.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
