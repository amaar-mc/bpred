"""Command-line interface for bpred."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bpred import (
    BimodalPredictor,
    GsharePredictor,
    TournamentPredictor,
    accuracy,
    mispredictions,
    run_trace,
)
from bpred.tournament import BranchPredictor


def _parse_taken(token: str) -> bool:
    """Convert a taken token to bool.

    Accepts: 1/0, T/F, true/false (case-insensitive).
    """
    normalised = token.strip().lower()
    if normalised in {"1", "t", "true"}:
        return True
    if normalised in {"0", "f", "false"}:
        return False
    raise ValueError(f"Cannot parse taken value: {token!r}")


def _parse_pc(token: str) -> int:
    """Parse a PC value as decimal or hex (0x prefix)."""
    token = token.strip()
    if token.startswith("0x") or token.startswith("0X"):
        return int(token, 16)
    return int(token)


def _load_trace(path: Path) -> list[tuple[int, bool]]:
    """Load a trace file with lines of the form ``<pc> <taken>``.

    Blank lines and lines beginning with ``#`` are ignored.
    """
    trace: list[tuple[int, bool]] = []
    with path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Line {lineno}: expected '<pc> <taken>', got {line!r}"
                )
            pc = _parse_pc(parts[0])
            taken = _parse_taken(parts[1])
            trace.append((pc, taken))
    return trace


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpred",
        description="Simulate classical CPU branch predictors on a branch trace.",
    )
    sub = parser.add_subparsers(dest="predictor", required=True)

    # ------------------------------------------------------------------
    # bimodal
    # ------------------------------------------------------------------
    bimodal = sub.add_parser("bimodal", help="Bimodal (Smith 1981) predictor")
    bimodal.add_argument("--counter-bits", type=int, required=True)
    bimodal.add_argument("--table-size", type=int, required=True)
    bimodal.add_argument("tracefile", type=Path)

    # ------------------------------------------------------------------
    # gshare
    # ------------------------------------------------------------------
    gshare = sub.add_parser("gshare", help="Gshare (McFarling 1993) predictor")
    gshare.add_argument("--history-bits", type=int, required=True)
    gshare.add_argument("--table-size", type=int, required=True)
    gshare.add_argument("tracefile", type=Path)

    # ------------------------------------------------------------------
    # tournament
    # ------------------------------------------------------------------
    tourn = sub.add_parser(
        "tournament", help="Tournament (Alpha 21264-style) predictor"
    )
    tourn.add_argument(
        "--local-predictor",
        choices=["bimodal"],
        required=True,
    )
    tourn.add_argument("--local-counter-bits", type=int, required=True)
    tourn.add_argument("--local-table-size", type=int, required=True)
    tourn.add_argument(
        "--global-predictor",
        choices=["gshare"],
        required=True,
    )
    tourn.add_argument("--global-history-bits", type=int, required=True)
    tourn.add_argument("--global-table-size", type=int, required=True)
    tourn.add_argument("--meta-bits", type=int, required=True)
    tourn.add_argument("tracefile", type=Path)

    return parser


def _build_predictor(args: argparse.Namespace) -> BranchPredictor:
    """Construct the predictor requested by *args*."""
    if args.predictor == "bimodal":
        return BimodalPredictor(
            counter_bits=args.counter_bits,
            table_size=args.table_size,
        )
    if args.predictor == "gshare":
        return GsharePredictor(
            history_bits=args.history_bits,
            table_size=args.table_size,
        )
    if args.predictor == "tournament":
        local: BranchPredictor = BimodalPredictor(
            counter_bits=args.local_counter_bits,
            table_size=args.local_table_size,
        )
        global_: BranchPredictor = GsharePredictor(
            history_bits=args.global_history_bits,
            table_size=args.global_table_size,
        )
        return TournamentPredictor(
            local=local,
            global_=global_,
            meta_bits=args.meta_bits,
        )
    raise ValueError(f"Unknown predictor: {args.predictor}")


def main() -> None:
    """Entry point for the ``bpred`` CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    tracefile: Path = args.tracefile
    if not tracefile.exists():
        print(f"error: trace file not found: {tracefile}", file=sys.stderr)
        sys.exit(1)

    try:
        trace = _load_trace(tracefile)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    predictor = _build_predictor(args)
    result = run_trace(predictor, trace=trace)

    acc = accuracy(trace_result=result)
    misses = mispredictions(trace_result=result)

    print(f"Predictor : {predictor!r}")
    print(f"Branches  : {result.total}")
    print(f"Hits      : {result.hits}")
    print(f"Misses    : {misses}")
    print(f"Accuracy  : {acc:.4%}")
