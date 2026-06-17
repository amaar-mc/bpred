# bpred

<p align="center">
  <img src="assets/logo.png" alt="bpred logo" width="160">
</p>

Pure-Python simulator of classical CPU branch predictors for computer architecture education.

Implements three predictors from first principles with zero runtime dependencies:

- **Bimodal** (Smith 1981) -- a table of n-bit saturating counters indexed by PC.
- **Gshare** (McFarling 1993) -- PC XOR global-history register indexes 2-bit counters.
- **Tournament** (McFarling 1993 / Alpha 21264) -- a meta-selector combining local and global sub-predictors.

Part of the same open-source computer architecture education series as [tomasulo](https://github.com/amaar-mc/tomasulo) (out-of-order execution) and scoreboarding.

## Install

```bash
pip install bpred
```

PyPI publication is pending; install from source in the meantime:

```bash
git clone https://github.com/amaar-mc/bpred
cd bpred
pip install -e ".[dev]"
```

## Python API

```python
from bpred import BimodalPredictor, GsharePredictor, TournamentPredictor
from bpred import run_trace, accuracy, mispredictions

# Bimodal: 2-bit counters, 1024-entry table
pred = BimodalPredictor(counter_bits=2, table_size=1024)

# Gshare: 10-bit history, 1024-entry table
pred = GsharePredictor(history_bits=10, table_size=1024)

# Tournament
from bpred import BimodalPredictor, GsharePredictor
local = BimodalPredictor(counter_bits=2, table_size=1024)
global_ = GsharePredictor(history_bits=10, table_size=1024)
pred = TournamentPredictor(local=local, global_=global_, meta_bits=2)

# Feed a trace
trace = [(0x1000, True), (0x1004, False), (0x1008, True)]
result = run_trace(pred, trace=trace)
print(accuracy(trace_result=result))       # e.g. 0.6667
print(mispredictions(trace_result=result)) # e.g. 1
```

## CLI

```
bpred bimodal --counter-bits 2 --table-size 1024 path/to/trace.trace
bpred gshare --history-bits 10 --table-size 1024 path/to/trace.trace
bpred tournament \
  --local-predictor bimodal --local-counter-bits 2 --local-table-size 1024 \
  --global-predictor gshare --global-history-bits 10 --global-table-size 1024 \
  --meta-bits 2 \
  path/to/trace.trace
```

Trace file format -- one branch per line:

```
# pc taken
0x1000 1
0x1004 0
0x1008 T
0x100c false
```

## Accuracy example

Running the bundled sample trace with a gshare predictor:

```
$ bpred gshare --history-bits 4 --table-size 16 examples/sample.trace
Predictor : GsharePredictor(history_bits=4, table_size=16)
Branches  : 20
Hits      : 18
Misses    : 2
Accuracy  : 90.0000%
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
mypy src
```

## License

MIT. See [LICENSE](LICENSE).
