# CacheLab v2: pre-registered final-long round 2

Written 2026-11-23T04:43Z, before any simulator run on final-long. The user approved this run: "run, commit only [if] we have superior res[ults]".

## The suite

`suites/final-long.json` (sha256 `1d58620a…a0c4`): 16 traces. No source file is shared with any earlier suite. fl-akamai-1 was dropped when the suite was rebuilt.

The baselines are in `results/engineer/final-long` (protocol v2: S3-FIFO, 3L-Cache cold over the whole trace, and the other published algorithms). Until now final-long has been used only to build the suite and run those baselines.

## The policy

Identical to round 1 (`PREREG.md`), with nothing changed after it:
- v1's frozen 32 (`scripts/vqsim/v1/portfolio.json`) plus 16 screened at the split from `scripts/vqsim/v2/pool.txt` (2,921 designs);
- miniature rate 0.005, floor 200, margin 0, `takeover=0`, `logcap=max(C, 65536)`, `wrapmode=1`;
- the same `vqsim.c`, `vqcore.h` and `vq_portfolio_real.py` (checksums are printed by the launcher, `/mydata/v2-final.sh`);
- spec `32:0.005:200:0:0:16:1:65536:1`.

The same job runs v1 (`32:0.005:200`) for reference. It is v1's first run on final-long.

Preparation: `vq_prepare.py` (dense ids, split and cache sizes from the baselines), then the S3-FIFO cold runs (`vq_run3.py`, `r1` line 0), as for train-long and heldout-long.

## The bar

On final-long, scored on windows 8–39 vs S3-FIFO, v2 passes round 2 if:

1. its mean change is below 3L-Cache's;
2. its median change is below 3L-Cache's;
3. it beats 3L-Cache on at least 11 of the 16 traces (≥ 2/3), each trace counted by its mean over its cells.

Round 1 (heldout-long) passed: 15/22 traces. If round 2 passes too, v2 meets the two-rounds-in-a-row bar, and the work is committed. Otherwise the result is reported as a miss and nothing is committed. The report gives aggregates only, with no trace names.

## Result (added after the run, 2026-11-23T04:48Z; log `logs/final-round2.txt`)

| final-long, 64 cells | mean | median | cells beating 3L-Cache | worse than S3-FIFO | traces beating 3L-Cache |
|---|---|---|---|---|---|
| 3L-Cache | −2.81% | −1.74% | | 20 | |
| v1 | −3.30% | −2.38% | 42 | 3 | 9/16 |
| **v2** | **−3.17%** | **−2.30%** | 42 | 5 | **7/16** |

- **v2 fails round 2.** It meets the mean and median criteria but beats 3L-Cache on only 7 of 16 traces (the bar is 11). Three of its losses are by under 0.1 pt. Nothing is committed, as agreed.
- v2 is worse than v1 here. It gains no trace and loses 2; it is better on 9 traces and worse on 7, by up to 2.85 pt; its mean is 0.13 pt worse.
- The loss is in caches of 2–8% of the objects: v2 −3.97%, v1 −4.44%, 3L-Cache −4.54%.
- v1's selector seldom switches on this suite (median 0 switches); v2 switches a median of 3 times.
- Neither version meets the open-source bar on final-long. v1 also wins only 9/16.

## Diagnosis (post hoc, 2026-11-23; log `logs/final-diagnosis.txt`, ad hoc scripts in `diag/`)

This analysis read final-long's per-cell results, so final-long can no longer serve as a clean test for any version informed by it.

**What failed: the cost of switching the live cache, not the choice of designs.** Each cell's live result was split using the decision trace (`VQSIM_TRACE=1`) and full-trace runs of every design used:

| final-long | live | same choices, no takeover cost | exact leader over the same candidates |
|---|---|---|---|
| v1 | −3.30%, 9/16 traces | −3.67%, 12/16 | −3.72%, 12/16 |
| v2 | −3.17%, 7/16 | −3.80%, 14/16 | −3.89%, 15/16 |

- **Selection worked.** With exact estimates, v2's candidates beat 3L-Cache on 15/16 traces, more than v1's (12/16). The first 20% predicts the rest at least as well as on train: median rank correlation over 300 random pool designs 0.94, against 0.91 on train.
- **Sampling is a small cost:** 0.10 pt, flipping at most one trace.
- **Takeovers and switches cost v2 0.62 pt and v1 0.37 pt** (on train, 0.19 and 0.18). That alone turns 7 of v2's would-be wins into losses. In those traces v2's exact margins over 3L-Cache were 0.08–1.29 pt, and its takeover costs 0.13–3.31 pt.
- **Why it costs more here:**
  - **The selector switches on noise.** v2 switches 3.45 times per cell (v1 1.92), for a sampled lead of a median 0.04% (217 of 221 switches under 1%). It treats switching as free; on train it nearly was, so nothing there showed the problem.
  - **Each switch costs more and for longer.** On train a switch's cost is paid within 1–2 windows. On final-long each v2 switch costs about 3× as much and still costs in the third window after it.
  - **Part of the cost never fades.** Four or more windows after a change, the live cache still trails the design's own run by 0.12 pt of the mean (train: 0.0). That is path dependence: after a takeover, the cache does not converge to the state the design reaches on its own.
- **The miniature caches can't see any of this.** They run cold from the start and never take over another design's cache, so they estimate each design's value without what it costs to get there.
- **An extreme case: 40 of the pool's designs are "static sets".** The small stage's promotion threshold is above the counter cap and ghost hits don't reach main, so the main stage keeps whatever filled it first: the trace's first objects in a cold run, S3-FIFO's contents after a takeover.
  - v2 picked one at the split in two cells of one trace. After the takeover they ran 76 and 23 pt worse than their own cold runs, up to +290% vs S3-FIFO in some windows.
  - An empty start at window 8 converges to the cold behavior; a takeover at window 4, 8 or 12 never does. So it is not an engine bug.
  - Removing the 40 doesn't rescue v2 (−3.19%, 7/16; train unchanged at −4.63%): the selector left those designs within a few windows.

**What it implies.** v2 needs the cost of a switch in its decisions: the miniatures should take over a cache as the live cache does, or the selector should switch only for a lead larger than the switch's measured cost on that workload. The 40 static-set designs should leave the pool on principle. Train-long barely exhibits switching costs, so it can't validate such a fix. That needs a development suite where switching is costly, and a fresh test suite.
