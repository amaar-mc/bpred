# Examples

## sample.trace

A small synthetic branch trace in the `<pc> <taken>` text format accepted by
the `bpred` CLI.

## Running the CLI

**Bimodal predictor:**

```bash
bpred bimodal --counter-bits 2 --table-size 16 examples/sample.trace
```

**Gshare predictor:**

```bash
bpred gshare --history-bits 4 --table-size 16 examples/sample.trace
```

**Tournament predictor:**

```bash
bpred tournament \
  --local-predictor bimodal --local-counter-bits 2 --local-table-size 16 \
  --global-predictor gshare --global-history-bits 4 --global-table-size 16 \
  --meta-bits 2 \
  examples/sample.trace
```

## Trace file format

One branch per line. Blank lines and lines starting with `#` are ignored.

```
# pc  taken
0x1000  1
0x1004  0
0x1008  T
0x100c  false
```

`pc` may be a decimal integer or a hex integer with `0x` prefix.
`taken` may be `1`/`0`, `T`/`F`, or `true`/`false` (case-insensitive).
