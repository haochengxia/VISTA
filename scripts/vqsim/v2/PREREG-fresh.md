# CacheLab v2 + replay takeover: pre-registered fresh-long test

Written 2026-11-23, before the suite is built. The user chose this test ("B"): a revised bar on a fresh suite.

## Why a revised bar

On every suite so far the per-trace count was decided by near-ties. A first-principles check also shows the old bar asks
more than choosing among known heuristics can give. Pick the best of all 14 published classic algorithms per cell, in
hindsight and with no switching cost: on final-long that beats 3L-Cache on only 9 of 16 traces (the bar was 11), and on
train-long 17 of 25. This test therefore judges the mean, the median and the share of cells. The trace count is still
reported.

## The suite (fresh-long)

Built by `vista-cachelab build-suite` with the same settings as the earlier long suites:

```
build-suite --data-root /mntData2/data/oracleReuse --datasets msr,cphy,alibabaBlock,tencentBlock,cf,akamai,fiu,ARC
  --per-dataset 4 --split heldout --full-length 10e6:50e6 --name-prefix fr --seed 20261123
  --exclude-suite <every file in suites/: dev-real, dev-mixed, ho-seen, ho-unseen, heldout-real, train-real,
     train-real-2, hl-seen, hl-unseen, heldout-long, train-long, final-long, final-long.full, tl-part>
  --out suites/fresh-long.json
```

- **Whole traces of 10M to 50M requests**, from the same dataset families as heldout-long and final-long. metaStorage
  has no unused files left.
- **No source file is shared with any earlier suite** (the exclusions), and there are no content copies. After building, the
  first 1M requests of every chosen trace are hashed and compared against every earlier suite's traces; an exact match
  is dropped.
- **Mechanical drop rule.** A trace is dropped only if its baselines fail to run, or if S3-FIFO or 3L-Cache has zero
  misses in the scored windows of a cell. This is decided from the baselines, before any candidate runs. Nothing else is
  dropped.
- The suite file's sha256 and its composition (traces per dataset, total requests) are added below once it is built,
  before any simulator run.

## Baselines

`engineer-baselines` (protocol engineer-handover-v2, as for every earlier suite): S3-FIFO, 3L-Cache cold over the whole
trace, the other published algorithms, Belady. One trace per shard across node0–9; the results are gathered in
`results/engineer/fresh-long`.

## The candidate: v2 with replay takeover (128)

Exactly the configuration of the development runs (log `scripts/vqsim/v2/logs/dev-switching.txt`), nothing changed:

- **Portfolio and pool:** v1's frozen 32 (`scripts/vqsim/v1/portfolio.json`) plus 16 screened at the split from
  `scripts/vqsim/v2/pool.txt` (2,921 designs).
- **Selector settings:** miniature rate 0.005, floor 200, margin 0, `takeover=0`, `logcap=max(C, 65536)`, `wrapmode=1`.
- **Replay takeover:** `replay=128`, so a takeover rebuilds the next design's state from the last 128·C requests. Its log
  costs 512 B per cache slot.
- **Spec** `32:0.005:200:0:0:16:1:65536:1:0:0:128` in `vq_portfolio_real.py`.
- **Checksums (sha256):**

  | File | sha256 |
  |---|---|
  | `vqsim.c` | `355417e7…77dd` |
  | `vqcore.h` | `8c5193d0…57da4` |
  | `vq_portfolio_real.py` | `832c83f1…15de` |
  | `vq_prepare.py` | `718ee190…3e37` |
  | `vq_run3.py` | `f807bab8…a9b` |
  | `v1/portfolio.json` | `0e96ab8f…f0` |
  | `v2/pool.txt` | `0eb708a6…73ea` |

  The launcher prints them in full.

Preparation is as for final-long: `vq_prepare.py` (dense ids, split and cache sizes from the baselines), then the
S3-FIFO cold runs (`vq_run3.py`, `r1` line 0).

The same job runs, for reference only (not part of the decision):
- v1 (`32:0.005:200:0:0:0:0:0:0:0:0`);
- v2 without replay (`…:0`);
- v1 with replay 128;
- the best single policy from the LLM knob search, `runs/evolve-2/programs/p024-2.c`, scored with
  `scripts/vqsim/evolve/evolve.py` (polsim, cold).

## The bar

On fresh-long, scored on windows 8–39 vs S3-FIFO, with 3L-Cache cold over the whole trace, the candidate passes if:

1. its mean change over all cells is below 3L-Cache's;
2. its median change over all cells is below 3L-Cache's;
3. it beats 3L-Cache in more than half of the cells.

Also reported, not part of the bar:
- traces beating 3L-Cache (each trace counted by its mean over its cells; the old bar was ≥ 2/3);
- cells worse than S3-FIFO;
- the breakdown by cache size;
- a bootstrap over traces (10,000 resamples, seed 0) of the candidate's mean minus 3L-Cache's, with a 95% interval.

**Rules:**
- **One round.** Nothing is tuned on fresh-long, before or after.
- **Commit rule:** if the candidate passes, the work is committed (the user's condition: commit only with superior
  results); if it fails, the result is reported as a miss and nothing is committed.
- **Anonymity:** the report gives aggregates and anonymous trace indices only.

## Suite as built

Added 2026-11-23, before any simulator run on it.

- **`suites/fresh-long.json`, sha256 `e37e03f705f5ab526c502d01aeedc5a24685c98648c52313dbd1a1365ac86c20`:** 20 traces,
  404.9M requests. That is 4 each from cphy, alibabaBlock, tencentBlock, cf and akamai; msr, fiu and ARC have no unused
  file of 10M to 50M requests.
- **No content copies:** none of the 20 matches the first 1M requests of any of the 143 earlier real-trace sources, or
  of each other.
- **Drop rule applied, from the baselines alone, before any candidate run.** Two traces (both from one dataset) are
  dropped because their baselines failed: their first 20% holds only 1 and 153 distinct objects, too few for four
  distinct cache sizes. 18 traces are scored. The launcher applies the rule mechanically and writes
  `suites/fresh-long.scored.json`.

## Result

Added after the run, 2026-11-23 21:33 node time; log `logs/fresh-test.txt`, launcher `fresh.sh` (all checksums as
pre-registered).

| fresh-long: 18 traces, 72 cells | mean | median | cells beating 3L-Cache | worse than S3-FIFO | traces beating 3L-Cache |
|---|---|---|---|---|---|
| 3L-Cache | −4.27% | −2.21% | | 16 | |
| **v2 + replay 128 (candidate)** | **−4.47%** | **−2.33%** | **39/72** | 4 | 9/18 |
| v2 | −4.28% | −2.31% | 38/72 | 4 | 9/18 |
| v1 + replay 128 | −3.88% | −1.88% | 38/72 | 4 | 8/18 |
| v1 | −3.80% | −1.89% | 38/72 | 2 | 6/18 |
| p024-2 (single policy, LLM knob search) | −3.48% | −1.53% | 36/72 | 8 | 9/18 |

- **The candidate passes the pre-registered bar.** All three criteria hold: mean −4.47% vs −4.27%, median −2.33% vs
  −2.21%, 39 of 72 cells.
- **The margin is thin.**
  - **Bootstrap over traces:** the candidate's mean minus 3L-Cache's is −0.20 pt, with a 95% interval of
    [−1.08, +0.50] pt, ahead in 66.7% of resamples. The difference is not statistically clear.
  - **The old per-trace bar** (at least 2/3 of traces) would not be met: 9/18.
- **Replay takeover helps v2 here:** +0.19 pt of the mean over v2 without it. The trace count and the median are
  about the same.
- **By cache size, the candidate leads 3L-Cache in every bucket:**

  | Cache size | Candidate | 3L-Cache |
  |---|---|---|
  | < 0.5% | −3.12% | −2.86% |
  | 0.5–2% | −2.74% | −2.60% |
  | 2–8% | −5.18% | −5.03% |
  | ≥ 8% | −8.70% | −8.32% |

- **Costs:**
  - **Switching:** a median of 2 switches per cell.
  - **Memory:** the selector's miniatures hold a median of 0.35 C entries. The replay log holds 128 C ids (512 B per
    slot).
  - **CPU:** the prefix replay at the split costs a median of 4.1 miniature steps per request, a one-off.
- **The best single policy from the LLM knob search loses to 3L-Cache on every measure,** as its final-long dev check
  suggested.

Per the commit rule, the work is committed. The claim it supports is limited: on unseen traces, v2 with replay
takeover matches or slightly beats 3L-Cache on mean, median and share of cells. It does not show a clear win.
