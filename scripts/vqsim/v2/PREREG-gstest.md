# gs-test: a pre-registered test for the HotOS paper

Written 2026-11-24, before the suite is built. It is the plan's G2 (`docs/hotos/plan.md`): headline numbers on a large
clean suite, and the paper's three claims tested with predictions fixed in advance. It also applies fresh-long's bar to
the same candidate once more.

## The suite (gs-test)

Built on the single server, in the container, with the same settings as fresh-long and gs-dev:

```
build-suite --data-root /traces --datasets cloudphysics,alibabaBlock,tencentBlock --per-dataset 1000 --split heldout
  --full-length 10e6:50e6 --name-prefix gt --seed 20261125 --exclude-suite /mydata/suites/gs-dev.json
  --out /mydata/suites/gs-test.json
```

- **Every eligible trace.** These are whole traces of 10M to 50M requests that gs-dev doesn't use. A census before writing
  this counted 106: cloudphysics 40, alibabaBlock 50 and tencentBlock 16. msr has no unused file in range. With
  `--per-dataset 1000` every eligible file is taken, so the seed only orders the search.
- **Content check.** The first 1M object ids of every trace are hashed. A trace whose hash matches a gs-dev trace, or an
  earlier gs-test trace, is dropped before preparation.
- **Mechanical drop rules, decided from the baselines before any candidate run:**
  - a trace is dropped if its first 20% has too few distinct objects for four distinct cache sizes (`engineer.prepare`);
  - a trace is dropped if S3-FIFO (vqsim or libCacheSim) or 3L-Cache has no misses after the split in one of its cells
    (`vq_learner.py drop`).

  Nothing else is dropped.
- **Cells:** four cache sizes per trace, 1%, 3%, 10% and 30% of the distinct objects in the first 20% (`engineer.prepare`).
- The suite file's sha256 and its composition are added below once it is built, before any candidate run.

### Known leakage limit

The server holds the same public datasets as the lost cluster. Earlier suites drew from these three datasets: train-long
(designs chosen), heldout-long (five looks), final-long (development) and fresh-long (one test). Their source lists were
lost with the cluster, so gs-test can't exclude them.

- At most about 40 of the 106 traces can be in an earlier suite: heldout-long 9, fresh-long 12, and train-long and
  final-long at most their draws from these datasets.
- **Direction of the bias.** Tuning exposure (train-long, and the configuration choices made on heldout-long and
  final-long) can only favor the candidate. The baselines are not tuned, so the baselines and the classic-oracle bound
  are unaffected.
- **Reading.** The candidate's lead on gs-test is an upper estimate. The paper states this.

## Baselines

All baselines come from libCacheSim's Python package (0.3.5, image `hx-cachelab`). Each one runs cold over the whole trace
(times are request indices), and only misses after the split count:

- **S3-FIFO:** vqsim `r1` line 0 is the per-window base. libCacheSim's S3-FIFO is also run and must equal it exactly.
- **3L-Cache:** `ThreeLCache(c, objective="object-miss-ratio")`.
- **The 14 published classic algorithms and Belady**, through `zoo.py`'s runner (`vq_learner.py zoo`): LRU, FIFO,
  CLOCK, SIEVE, ARC, LIRS, 2Q, CLOCK-Pro, MQ, S3-FIFO, W-TinyLFU, LeCaR, Cacheus, LHD.
- **A caveat found in the smoke test.** On one gs-dev trace, libCacheSim's CLOCK-Pro is 458% worse than S3-FIFO, which is
  implausible. It stays among the 14, as on the earlier suites.
  - A broken algorithm can only lower the classic-oracle bound, so it makes P2 slightly easier to meet.
  - The per-algorithm table reports it as measured.

## The candidate: v2 with replay takeover (128)

Exactly fresh-long's candidate, nothing changed:

- **Portfolio:** v1's frozen 32 plus 16 screened at the split from `v2/pool.txt`.
- **Spec:** `32:0.005:200:0:0:16:1:65536:1:0:0:128` in `vq_portfolio_real.py`.
- **Scoring:** windows 8–39 vs S3-FIFO.
- **Checksums (sha256).** The launcher verifies each one in full:

  | File | sha256 |
  |---|---|
  | `vqsim.c` at commit 5c14ae8 (the candidate's simulator) | `355417e737c231ab10d5db00fedf7b4f763e25283904d6a8a33f89e3f0e277dd` |
  | `vqcore.h` | `8c5193d00d8bd4cb656993cf9610116ff0ac9363ba88d9b740e498c485957da4` |
  | `vq_portfolio_real.py` | `832c83f114f0b27121abaae67a09b864651b5ae9ece39fde0ca56250126215de` |
  | `vq_run3.py` | `f807bab8ad5d2dec2c75fa45f2d55663b568f703cdda51300faf9010bd6a6e9b` |
  | `v1/portfolio.json` | `0e96ab8ff3a84c61081dd534f8db5ddfaf72bebca90da37348e89a87fce3d7f0` |
  | `v2/pool.txt` | `0eb708a62a81c5085f6639d88d294b89cff41711cbfff44f2661d168e84073ea` |

- **The takeover grid** uses a separate build of the working-tree `vqsim.c`, which adds only the opt-in `VQSIM_DUMP` hook.
  Every dumping run's output is asserted equal to the candidate's, cell by cell.

## The bar (as for fresh-long)

On gs-test the candidate passes if:

1. its mean change over all cells is below 3L-Cache's;
2. its median change over all cells is below 3L-Cache's;
3. it beats 3L-Cache in more than half of the cells.

## Predictions for the paper's claims

These are fixed now and judged on gs-test as stated:

- **P1, honest evaluation shrinks the wins.** The candidate's lead over 3L-Cache (3L-Cache's mean change minus the
  candidate's) is below 1.0 pt: the upper end of its 95% interval is below 1.0 pt. On train-long, where the designs were
  chosen, it was 2.0 pt.
- **P2, a ceiling for choosing among heuristics.** 3L-Cache beats every classic algorithm on at least 20% of the traces.
  This uses the hindsight-best of the 14 per cell, with no switching cost, as the trace's mean over its cells. The figure
  was 32% on train-long and 44% on final-long.
- **P3, the learner's potential doesn't survive online.**
  - The fixed rule is one-way: v2 runs until the first window w ≥ 9 at which the following holds, then 3L-Cache takes over.
    - The 3L-Cache miniature at rate 0.1, scaled by the window's sampling ratio, beats v2's observed live misses over the
      last 8 windows (from window 8 on).
    - Its estimated gain per window g satisfies g > 0 and g·(40 − w) > 1·C.
  - The takeover is mode A, rebuilt from the last 128·C requests. The first request to a live object that 3L-Cache lacks
    is counted as a miss ("pes").
  - The rule was the best of 24 on gs-dev (lead +0.53 pt pes, v2 +0.43 pt), so it is chosen optimistically.
  - Prediction: its gain over v2 (its lead minus v2's) is less than a third of the gain of the hindsight-best switch window
    (real takeover, pes) over v2.
- **P4, the learner's edge appears late.** 3L-Cache's median change vs S3-FIFO at full scale is lower on windows 8–39 than
  on windows 2–7.

## Also reported (not part of the bar or the predictions)

- Traces beating 3L-Cache; cells worse than S3-FIFO; breakdowns by cache size and by dataset family.
- A bootstrap over traces (10,000 resamples, seed 0) of the lead, with a 95% interval.
- Each classic algorithm vs S3-FIFO; Belady; the classic-oracle bound's lead over 3L-Cache.
- The hindsight per-cell min(v2, 3L-Cache), both with free switching and at the best switch window with the real takeover
  (opt and pes).
- All 24 cost-aware rules of gs-dev, for context only. Only the fixed rule above is judged.
- The takeover transient: misses per cache slot over all cells and windows.
- The miniature's fidelity for 3L-Cache, at the selector's rate and at rates 0.02, 0.05, 0.1 and 0.2.

## Rules

- **One round.** Nothing is tuned on gs-test, before or after. Nothing in this file changes once the suite is built,
  except the "Suite as built" and "Result" sections.
- **Two stages.**
  - Stage A: the suite, the content check, preparation, the baselines and the drop rules.
  - The "Suite as built" section is then added and this file's sha256 is recorded.
  - Stage B: the candidate and all analyses. It prints and checks that sha256 before the candidate runs.
- **Commit rule:** as before, commit only with superior results. Whatever the outcome, the numbers go into the paper as
  measured, with the leakage limit stated.
- **Anonymity:** reports give aggregates and anonymous trace indices only. Dataset families appear only as aggregates.

## Suite as built

Added 2026-11-24, after stage A (log `v2/logs/gstest-A.txt`, 2 h, exit 0) and before stage B. No baseline numbers were looked at.

- **`gs-test.json`, sha256 `28a468f7e2791e770f4d00c9df56f1ac9f5aac143801bdec1de511e0b9107371`:** 106 traces, 2,221.0M
  requests. They are alibabaBlock 50, cloudphysics 40 and tencentBlock 16, as the census counted.
- **Content check:** none of the 106 matches a gs-dev trace or another gs-test trace in its first 1M object ids.
  `gs-test.checked.json` (sha256 `e0107b831803608bc17374cb808dc256652ea1c1d67daa3eff9426c40621dd4c`) keeps all 106.
- **Drop rules:**
  - preparation drops none (every trace has enough distinct objects in its first 20%);
  - the baselines' rule drops none (S3-FIFO and 3L-Cache have misses after the split in every cell).
- **Scored:** 106 traces, 424 cells.
- **Baselines:** the zoo ran 6,360 runs with 0 errors. 3L-Cache and S3-FIFO ran at full scale and as miniatures.

## Result

Stage B (log `v2/logs/gstest-B.txt`, 9 h, exit 0). Its checks:
- the pre-registration's sha256 and every candidate file match;
- the VQSIM_TRACE rerun reproduces the candidate exactly in 424/424 cells;
- the zoo's S3-FIFO equals the pipeline's in 424/424 cells.

**The candidate passes the bar.**

| On 106 traces, 424 cells, vs S3-FIFO (windows 8–39) | v2 + replay 128 | 3L-Cache |
|---|---|---|
| Mean | **−5.11%** | −4.62% |
| Median | **−2.36%** | −2.15% |
| Cells beating 3L-Cache | **226/424** | |
| Traces beating 3L-Cache (cell mean) | 55/106 | |
| Cells worse than S3-FIFO | 46 | 106 |

- **Lead over 3L-Cache: +0.49 pt, 95% interval [+0.11, +0.88]** (10,000 trace resamples, seed 0).
- **By cache size** (1%, 3%, 10%, 30% of the prefix's objects):

  | Cache size | v2 | 3L-Cache | Cells beating 3L-Cache |
  |---|---|---|---|
  | 1% | −2.65% | −2.50% | 53/106 |
  | 3% | −4.09% | −4.04% | 57/106 |
  | 10% | −6.59% | −6.01% | 62/106 |
  | 30% | −7.12% | −5.95% | 54/106 |

- **By family** (v2 vs 3L-Cache):

  | Family | Traces | v2 | 3L-Cache |
  |---|---|---|---|
  | alibabaBlock | 50 | −6.40% | −6.35% |
  | cloudphysics | 40 | −2.77% | −1.67% |
  | tencentBlock | 16 | −6.95% | −6.61% |

**P1 holds.** The interval's upper end, +0.88 pt, is below 1.0 pt.

**P2 holds.** 3L-Cache beats every classic algorithm on 50/106 traces (47%).
- The classic oracle (best of 14 per cell, in hindsight): mean −5.25%, median −2.21%, 221/424 cells beating 3L-Cache,
  56/106 traces.
- Its lead over 3L-Cache is +0.63 pt [+0.21, +1.06].
- The online candidate (−5.11%) comes within 0.14 pt of it.

Per algorithm (mean / median vs S3-FIFO; cells beating 3L-Cache):

| Algorithm | Mean | Median | Cells beating 3L-Cache |
|---|---|---|---|
| LRU | +7.27% | +4.65% | 34 |
| FIFO | +16.06% | +9.48% | 5 |
| CLOCK | +7.36% | +4.79% | 25 |
| SIEVE | +21.28% | +5.36% | 16 |
| ARC | +1.04% | +0.59% | 86 |
| LIRS | +3.06% | +1.24% | 61 |
| 2Q | +0.74% | +0.42% | 76 |
| CLOCK-Pro | +248.93% | +97.43% | 4 (implausible; see the caveat above) |
| MQ | +5.01% | +2.51% | 49 |
| S3-FIFO | 0 | 0 | 106 |
| W-TinyLFU | +14.01% | +3.18% | 51 |
| LeCaR | +4.77% | +3.48% | 42 |
| Cacheus | +4.26% | +3.15% | 49 |
| LHD | −0.82% | +0.77% | 94 |
| Belady | −22.45% | −20.12% | 424 |

**P4 holds.** This was computed early, while the grid ran, by the same code as `describe` (`gst_early.py`); stage B
recomputes it.
- 3L-Cache vs S3-FIFO at full scale: the median is −0.41% on windows 2–7 and −2.15% on windows 8–39 (means −1.53% and
  −4.62%).
- The edge keeps growing through the suffix: −3.69%, −4.74%, −4.95% and −5.05% by quarter.
- The selector-rate miniature misjudges the prefix comparison by a median of +2.73 pt (IQR [+0.43, +8.02]). It makes
  3L-Cache look worse than it is.

**Hindsight potential (context for P3).** min(v2, 3L-Cache) per cell with free switching: −5.98%, lead +1.36 pt
[+1.08, +1.65], 103/106 traces. On gs-dev it was +1.38 pt, 31/32 traces.

Stage B's `describe` reproduced these numbers exactly.
- The miniature's fidelity for 3L-Cache on windows 8–39, as the scaled miniature's misses relative to full scale (median,
  IQR):

  | Rate | Median | IQR |
  |---|---|---|
  | 0.02 | +7.68% | [+1.65, +18.76]% |
  | 0.05 | +4.04% | [+0.18, +11.73]% |
  | 0.1 | +2.84% | [−0.29, +7.05]% |
  | 0.2 | +1.57% | [−0.77, +4.38]% |

  The miniature overstates 3L-Cache's misses at every rate, so it is biased against the learner.
- At the selector's own rate (0.005) the bias is large: the split and sticky rules of `score` never pick 3L-Cache in any
  cell, at any margin.
- At the split, the selector-rate miniature predicts the suffix winner between 3L-Cache and v2 in 222/424 cells, about a
  coin flip.

**P3 holds, more strongly than predicted.**
- **The hindsight-best switch window, with the real takeover:**
  - pes: lead +1.35 pt [+1.06, +1.66], 88/106 traces, a gain of +0.86 pt over v2;
  - opt: lead +1.42 pt.
- **The fixed rule** (rate 0.1, κ 1, H=8, pes) switches in 182 cells:
  - its lead is +0.26 pt [−0.06, +0.60], against v2's +0.49 pt;
  - so its gain over v2 is **−0.23 pt**, against the bound of a third of +0.86 pt (+0.29). Switching to the learner
    online makes the selector worse.
- **Context only (not judged):** the best of the 24 rules on gs-test, chosen in hindsight here, is rate 0.2, κ 1, H=all:
  +0.55 opt and +0.51 pes, i.e. +0.02 to +0.06 pt over v2. Rules at the rates a selector can afford (0.05, 0.1) lose
  0.1–0.4 pt.
- **Takeover transient:** a median 0.14 extra misses per cache slot (IQR [0.00, 1.18]) over all cells and windows.

**Summary.** The candidate passes the bar, and P1–P4 all hold.
- The candidate's lead over 3L-Cache is small but above zero: +0.49 pt [+0.11, +0.88].
- It is close to the classic-oracle bound (+0.63 pt), which itself loses to 3L-Cache on 47% of traces.
- The learner's hindsight potential (+1.35 pt with the real takeover) is not reachable online: the learner's edge
  appears after the decision point, the miniatures misjudge it, and switching to it online costs 0.23 pt.

**Reading.** Read the lead as an upper estimate (see "Known leakage limit"). Taken with the earlier suites, the lead fell
as the tests got cleaner and then settled near 0.5 pt:
- train-long 2.0 pt;
- heldout-long 0.85 pt;
- final-long 0.57 pt;
- fresh-long 0.20 pt (interval through 0);
- gs-dev 0.43 pt;
- gs-test 0.49 pt (interval above 0).
