# Predictor Architecture

This document describes the three branch predictors implemented in `bpred`,
their algorithmic basis, and the primary academic references.

## References

- J. E. Smith, "A study of branch prediction strategies," in _Proceedings of
  the 8th Annual Symposium on Computer Architecture (ISCA)_, pp. 135-148, 1981.
- S. McFarling, "Combining Branch Predictors," WRL Technical Note TN-36,
  Digital Equipment Corporation Western Research Laboratory, June 1993.

---

## Saturating Counter

A saturating counter is the base component of all three predictors. An n-bit
counter holds a value in [0, 2^n - 1]. It predicts _taken_ when the value is
>= 2^(n-1) and _not-taken_ otherwise. Taken outcomes increment the counter
toward the maximum; not-taken outcomes decrement it toward zero. At either
extreme the counter saturates rather than wrapping.

The 2-bit variant (n=2) is the classic design from Smith (1981). Its four
states are:

| Value | Name            | Predict |
|-------|-----------------|---------|
| 0     | Strongly Not-Taken (SN) | not-taken |
| 1     | Weakly Not-Taken (WN)   | not-taken |
| 2     | Weakly Taken (WT)       | taken |
| 3     | Strongly Taken (ST)     | taken |

Two consecutive mispredictions are required to flip the prediction, providing
hysteresis against noise.

---

## Bimodal Predictor

**Reference:** Smith (1981).

A flat table of `table_size` saturating counters of `counter_bits` bits each.
The table is indexed by `pc mod table_size`. On every branch:

1. Predict using the counter at index `pc mod table_size`.
2. Update that counter with the actual outcome.

Multiple branches whose PCs map to the same entry share a counter (aliasing).
Larger tables reduce aliasing at the cost of hardware area.

---

## Gshare Predictor

**Reference:** McFarling (1993).

Gshare extends the bimodal idea by incorporating global correlation. A global
history register (GHR) of `history_bits` bits records the outcomes of the most
recent branches across the entire program. The table index is:

```
index = (pc XOR ghr) mod table_size
```

Each entry is a 2-bit saturating counter. After every branch the actual
outcome is shifted into the GHR (MSB first), discarding outcomes older than
`history_bits`.

XOR hashing distributes entries more uniformly across the table than either
PC or GHR alone, and exploits cross-branch correlation to improve accuracy
on correlated loops and function-call patterns.

---

## Tournament Predictor

**Reference:** McFarling (1993); Alpha 21264 implementation.

A tournament predictor combines two sub-predictors (a local and a global) using
a meta-selector table. The meta-selector contains saturating counters of
`meta_bits` bits that choose which sub-predictor to trust for each PC.

### Prediction

1. Look up the chooser counter at `pc mod chooser_size`.
2. If counter value < threshold (2^(meta_bits - 1)), use the _local_ predictor.
3. Otherwise use the _global_ predictor.

### Update (Alpha 21264 scheme)

Both sub-predictors are always updated with the actual outcome. The chooser is
updated only when the two sub-predictors _disagree_:

- If only the local predictor was correct: decrement the chooser (bias toward local).
- If only the global predictor was correct: increment the chooser (bias toward global).
- If both or neither was correct: no chooser update.

This approach concentrates meta-selector learning signal on cases where the
choice actually matters.

The chooser table size equals `max(local.table_size, global_.table_size)`.
