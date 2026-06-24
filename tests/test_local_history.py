"""Tests for LocalHistoryPredictor (PAg, Yeh and Patt 1991)."""

from bpred.bimodal import BimodalPredictor
from bpred.local_history import LocalHistoryPredictor
from bpred.trace import accuracy, run_trace

# ------------------------------------------------------------------
# Local history register behaviour
# ------------------------------------------------------------------


def test_local_history_starts_empty() -> None:
    """Every BHT entry starts at 0 (no recorded outcomes)."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=8, pht_size=16)
    for pc in range(8):
        assert p._local_history(pc=pc) == 0  # noqa: SLF001


def test_local_history_shifts_lsb_first() -> None:
    """Outcomes shift into the LSB of the per-branch history register."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=8, pht_size=16)
    p.update(pc=0, taken=True)
    assert p._local_history(pc=0) == 0b0001  # noqa: SLF001
    p.update(pc=0, taken=True)
    assert p._local_history(pc=0) == 0b0011  # noqa: SLF001
    p.update(pc=0, taken=False)
    assert p._local_history(pc=0) == 0b0110  # noqa: SLF001
    p.update(pc=0, taken=True)
    assert p._local_history(pc=0) == 0b1101  # noqa: SLF001


def test_local_history_capped_at_history_bits() -> None:
    """A branch's local history never exceeds 2^history_bits - 1."""
    p = LocalHistoryPredictor(history_bits=3, bht_size=4, pht_size=8)
    mask = (1 << 3) - 1
    for _ in range(10):
        p.update(pc=0, taken=True)
        assert p._local_history(pc=0) <= mask  # noqa: SLF001


def test_per_branch_history_is_independent() -> None:
    """Two branches that map to distinct BHT entries keep separate histories."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=8, pht_size=16)
    p.update(pc=0, taken=True)
    p.update(pc=1, taken=False)
    assert p._local_history(pc=0) == 0b0001  # noqa: SLF001
    assert p._local_history(pc=1) == 0b0000  # noqa: SLF001


# ------------------------------------------------------------------
# Periodic pattern learning (the PAg headline capability)
# ------------------------------------------------------------------


def test_alternating_pattern_beats_bimodal() -> None:
    """A strict alternating T,N,T,N,... trace is learned to high accuracy.

    With one bit of local history the PAg predictor distinguishes the two
    phases of the period and reaches ~100% accuracy after warm-up.  A bimodal
    predictor with a single counter is stuck near 50% because it has no
    history and always predicts the majority class.
    """
    n = 400
    alternating_trace = [(0x300, i % 2 == 0) for i in range(n)]

    local = LocalHistoryPredictor(history_bits=1, bht_size=4, pht_size=2)
    bimod = BimodalPredictor(counter_bits=2, table_size=1)

    local_acc = accuracy(trace_result=run_trace(local, trace=alternating_trace))
    bimod_acc = accuracy(trace_result=run_trace(bimod, trace=alternating_trace))

    assert local_acc > 0.95, f"local accuracy too low: {local_acc:.3f}"
    assert bimod_acc <= 0.55, f"bimodal unexpectedly good: {bimod_acc:.3f}"
    assert local_acc > bimod_acc, (
        f"local ({local_acc:.3f}) did not beat bimodal ({bimod_acc:.3f})"
    )


def test_length_k_pattern_captured_when_history_bits_ge_k() -> None:
    """A length-k periodic pattern is learned once history_bits >= k.

    Pattern of period 4: T, T, N, N repeating.  Predicting the next outcome
    from the previous k outcomes requires distinguishing all positions in the
    period, which needs at least k history bits.  With history_bits == 4 the
    PAg predictor reaches near-perfect accuracy after warm-up.
    """
    period = [True, True, False, False]
    k = len(period)
    n = 400
    trace = [(0x400, period[i % k]) for i in range(n)]

    p = LocalHistoryPredictor(history_bits=k, bht_size=4, pht_size=1 << k)
    acc = accuracy(trace_result=run_trace(p, trace=trace))
    assert acc > 0.95, f"period-{k} pattern not learned: {acc:.3f}"


def test_short_history_cannot_capture_long_period() -> None:
    """history_bits below the period leaves the predictor unable to separate
    distinct phases that share the same short suffix.

    Period 4 pattern T,T,N,N with only 1 history bit: the suffixes '...T' and
    '...N' each precede both a same and a different outcome, so a single shared
    counter per pattern cannot reach the near-perfect accuracy of the
    full-history case.
    """
    period = [True, True, False, False]
    n = 400
    trace = [(0x500, period[i % 4]) for i in range(n)]

    short = LocalHistoryPredictor(history_bits=1, bht_size=4, pht_size=2)
    full = LocalHistoryPredictor(history_bits=4, bht_size=4, pht_size=16)

    short_acc = accuracy(trace_result=run_trace(short, trace=trace))
    full_acc = accuracy(trace_result=run_trace(full, trace=trace))

    assert full_acc > 0.95
    assert short_acc < full_acc


# ------------------------------------------------------------------
# Strongly-biased branches
# ------------------------------------------------------------------


def test_always_taken_eventually_accurate() -> None:
    """An always-taken branch reaches high accuracy after warm-up."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=16, pht_size=16)
    trace = [(0x100, True)] * 50
    acc = accuracy(trace_result=run_trace(p, trace=trace))
    assert acc >= 0.9


def test_always_not_taken_eventually_accurate() -> None:
    """An always-not-taken branch reaches high accuracy after warm-up."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=16, pht_size=16)
    trace = [(0x200, False)] * 50
    acc = accuracy(trace_result=run_trace(p, trace=trace))
    assert acc >= 0.9


# ------------------------------------------------------------------
# Counter saturation through the PHT
# ------------------------------------------------------------------


def test_pht_counter_saturates() -> None:
    """Repeated taken outcomes drive the indexed PHT counter to its max."""
    # history_bits=1, so an always-taken branch settles on history pattern 1
    # and repeatedly hits PHT[1], saturating it at max_value (3 for 2-bit).
    p = LocalHistoryPredictor(history_bits=1, bht_size=4, pht_size=2)
    for _ in range(10):
        p.update(pc=0, taken=True)
    assert p._pht[1].value == p._pht[1].max_value  # noqa: SLF001
    assert p._pht[1].value == 3  # noqa: SLF001


def test_pht_starts_weakly_taken() -> None:
    """All PHT counters start in the weakly-taken neutral state (value 2)."""
    p = LocalHistoryPredictor(history_bits=2, bht_size=4, pht_size=4)
    for counter in p._pht:  # noqa: SLF001
        assert counter.value == 2
        assert counter.predict() is True


# ------------------------------------------------------------------
# Table sizing
# ------------------------------------------------------------------


def test_tables_sized_as_documented() -> None:
    """BHT and PHT are sized exactly as requested."""
    p = LocalHistoryPredictor(history_bits=3, bht_size=8, pht_size=8)
    assert len(p._bht) == 8  # noqa: SLF001
    assert len(p._pht) == 8  # noqa: SLF001
    assert p.history_bits == 3
    assert p.bht_size == 8
    assert p.pht_size == 8
    assert p.table_size == 8


# ------------------------------------------------------------------
# Interface conformance
# ------------------------------------------------------------------


def test_predict_returns_bool() -> None:
    """predict() must return bool, matching the BranchPredictor protocol."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=16, pht_size=16)
    assert isinstance(p.predict(pc=0x1000), bool)


def test_update_returns_none() -> None:
    """update() must return None."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=16, pht_size=16)
    assert p.update(pc=0x1000, taken=True) is None


def test_run_trace_compatible() -> None:
    """LocalHistoryPredictor is accepted by run_trace (protocol conformance)."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=16, pht_size=16)
    trace = [(0x100, True), (0x104, False), (0x108, True)]
    result = run_trace(p, trace=trace)
    assert result.total == 3
    assert len(result.predictions) == 3
    assert all(isinstance(pred, bool) for pred in result.predictions)


def test_deterministic() -> None:
    """Same trace produces identical results on two fresh predictors."""
    trace = [(i * 8, i % 3 != 0) for i in range(40)]
    p1 = LocalHistoryPredictor(history_bits=4, bht_size=16, pht_size=16)
    p2 = LocalHistoryPredictor(history_bits=4, bht_size=16, pht_size=16)
    r1 = run_trace(p1, trace=trace)
    r2 = run_trace(p2, trace=trace)
    assert r1.predictions == r2.predictions
    assert r1.correct == r2.correct
    assert r1.hits == r2.hits


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def test_invalid_history_bits_raises() -> None:
    """history_bits < 1 must raise ValueError."""
    try:
        LocalHistoryPredictor(history_bits=0, bht_size=4, pht_size=4)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_bht_size_raises() -> None:
    """bht_size < 1 must raise ValueError."""
    try:
        LocalHistoryPredictor(history_bits=4, bht_size=0, pht_size=4)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_pht_size_raises() -> None:
    """pht_size < 1 must raise ValueError."""
    try:
        LocalHistoryPredictor(history_bits=4, bht_size=4, pht_size=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_repr() -> None:
    """__repr__ includes class name and all three parameters."""
    p = LocalHistoryPredictor(history_bits=4, bht_size=16, pht_size=16)
    r = repr(p)
    assert "LocalHistoryPredictor" in r
    assert "history_bits=4" in r
    assert "bht_size=16" in r
    assert "pht_size=16" in r
