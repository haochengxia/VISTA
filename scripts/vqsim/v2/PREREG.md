# CacheLab v2: pre-registered heldout-long round 1

Written 2026-11-23T02:46Z, before any v2 run on heldout-long. The configuration below was chosen on train-long only (`docs/cachelab-v1.md`, Next item 3, "Pool screening at the split"; log `scripts/vqsim/v1/logs/screen-train.txt`). Nothing is tuned on heldout-long afterwards.

## The policy

- S3-FIFO serves the first 20% of each trace (the split comes from each trace's `meta.json`). The trace is cut into 40 windows; windows 8–39 are scored.
- **Fixed candidates:** v1's frozen 32 designs, `scripts/vqsim/v1/portfolio.json` (sha256 `0e96ab8f…3d7f0`).
- **Screening pool:** `scripts/vqsim/v2/pool.txt`, 2,921 designs (sha256 `0eb708a6…4073ea`). These are the distinct configs of the r1 seeds, the board picks, the rd twins and the gate twins, in that order. Designs equal to a fixed candidate are skipped.
- **Selector:** `vqsim` online mode (`scripts/vqsim/vqsim.c` sha256 `08baf50c…ed68`, `vqcore.h` sha256 `8c5193d0…7da4`), with these settings:
  - miniature rate 0.005, floor 200 objects, margin 0, stage-mapping takeover (`takeover=0`);
  - `screen=16`, `logcap=max(C, 65536)`, `wrapmode=1`.
- **Evaluator:** `scripts/vqsim/study/vq_portfolio_real.py` (sha256 `40d43864…4ab3a`), spec `32:0.005:200:0:0:16:1:65536:1`, `POOL=` the pool file.
- **Memory rule** (chosen by the user): the prefix log holds at most max(C, 65,536) entries of 4-byte ids. The selector's miniature caches are as in v1.

## What is measured

On heldout-long (22 traces, 88 cells), each cell's change in misses vs S3-FIFO over windows 8–39, against 3L-Cache's (cold over the whole trace, from `meta.json`), using the same evaluator as all earlier rounds.

The same run repeats v1 (spec `32:0.005:200`, no screening) as a check. It must reproduce v1's pre-registered result: −3.42% / −1.79%, 12/22 traces.

The report gives aggregates only: mean, median, cells beating 3L-Cache, cells worse than S3-FIFO, traces beating 3L-Cache (a trace's mean over its cells below 3L-Cache's), size classes, and the selector's memory and replay cost. No trace names.

## The bar

v2 passes round 1 if, on heldout-long:

1. its mean change is below 3L-Cache's (3L-Cache: −2.73%);
2. its median change is below 3L-Cache's (−1.24%);
3. it beats 3L-Cache on at least 15 of the 22 traces (≥ 2/3).

The open-source bar needs two such rounds in a row. This is round 1. A miss on any criterion is reported as a miss.

Earlier uses of heldout-long:
- v1 idealized;
- the seeds portfolio;
- the candidate sets;
- the r1+bp portfolio (v1's result);
- the rebuild's reproduction of v1.

## Result (added after the run, 2026-11-23T03:47Z; log `logs/heldout-round1.txt`)

| heldout-long, 88 cells | mean | median | cells beating 3L-Cache | worse than S3-FIFO | traces beating 3L-Cache |
|---|---|---|---|---|---|
| 3L-Cache | −2.73% | −1.24% | | 27 | |
| v1 (check: reproduced exactly) | −3.42% | −1.79% | 47 | 9 | 12/22 |
| **v2** | **−3.58%** | **−1.82%** | **53** | 10 | **15/22** |

- **v2 passes round 1 on all three criteria, with no margin on the third:** 15/22 is exactly the bar.
- Four of the 15 wins are by under 0.1 pt (−0.08, −0.04, −0.02, −0.01 pt on the trace mean), and one loss is by 0.04 pt. A second round could fall either side.
- Against v1 it gains 3 traces and loses none. It is better than v1 on 12 traces and worse on 9.
- The mean gain over v1 is 0.16 pt, against 0.43 pt on train.
- By size: v2 beats 3L-Cache in every class except caches of 8% of the objects or more (−6.88% against −7.63%).
- The selector holds 0.24 C entries in the median cell, plus a prefix log of 0.6 C. The replay costs a median of 3.0 miniature steps per request.
