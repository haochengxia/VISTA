# CacheLab v1: an online selector over agent-designed FIFO caches

Version 1, frozen 2026-11-17. The policy is `OnlineV1` in `src/vista/benchmarks/cachelab/sandbox/vq.py`. The candidates are in `scripts/vqsim/v1/portfolio.json` (sha256 `0e96ab8f…`). The logs are in `scripts/vqsim/v1/logs/`. The working notes are item 5 onward of "Round 3" in `docs/cachelab-roadmap.md`.

## What it is

A cache that runs FIFO-based eviction designs and switches between them online, starting from S3-FIFO.

- **Design space (VQ, "virtual queues").** An object lives in one of up to 6 FIFO stages. Hits only raise a small counter. At a stage's tail, an object goes up (promotion or reinsertion) if its counter reaches the stage's threshold. Otherwise it goes down to another stage, or leaves into a ghost of recent keys. Per-stage options cover counting, thresholds, counter decay, Delayed FIFO-Reinsertion and Age-Guided Eviction. S3-FIFO, S4-FIFO-like skip regions, 2Q-like layouts and hot/warm mains are all points in this space.
  - `vq.py` (Python, sandbox) and `scripts/vqsim/vqsim.c` (C, 100× faster) agree exactly, which is tested.
- **Designers.** There are two sources of designs:
  - host sweeps: 426 seeds, covering the S3 grid, S4-like designs and random VQ designs;
  - LLM "architects" (GLM-5.3 Flash) sharing a board (`architect.py`). Architects propose families of designs with a hypothesis, read results and findings, and record findings. Over four generations, 17 architects evaluated 156 families and recorded 43 findings.
- **Selector (`OnlineVQ`).**
  - S3-FIFO serves the cache until the takeover; then the live cache runs the candidate with the fewest misses so far, re-chosen at every window start.
  - It estimates each candidate's misses in a miniature simulation on a spatial sample of the keys, from the first request on (SHARDS-style).
  - A switch is a takeover: objects of the old entry stage go to the new design's small stage, all others to its main; counters, hit times and the newest ghost entries are kept.
- **v1 configuration.**
  - 32 candidates, picked greedily on the training cells starting from S3-FIFO, using the idealized online choice; 16 of the 32 are architect designs.
  - Sample rate 0.5%, with at least 200 objects per miniature cache.
  - No switching margin.
  - Decisions at the takeover and every window.

## Protocol

- **Workloads.** Whole real traces of 10M–50M requests, unit object size, 4 cache sizes per trace (0.1%–25% of the trace's objects).
  - train-long: 25 traces, 100 cells.
  - heldout-long: 22 traces, 88 cells. Its traces and datasets are disjoint from training, and it is never used for any choice.
- **Deployment.** S3-FIFO serves the first 20%; the policy takes over its live cache at that point. The score is the change in misses vs S3-FIFO on windows 8–39, the last 80%.
- **Comparison.** 3L-Cache (a learned policy, the strongest published one here) runs on the whole trace from the start, which is an advantage over us.
- **Anonymity.** Agents only ever see anonymous cells.

## Results

### Held-out, pre-registered, each run once

All rows are on heldout-long, 88 cells, last 80%, change vs S3-FIFO.

| | mean | median | beats 3L-Cache (cells) | worse than S3-FIFO | selector memory |
|---|---|---|---|---|---|
| 3L-Cache | −2.73% | −1.24% | – | 27 | – |
| **v1: 32 candidates (seeds + architect picks), 0.5%, floor 200** | **−3.42%** | **−1.79%** | 47/88 | **9** | 0.16·C entries (median) |
| 32 candidates from seeds only, same selector | −3.18% | −1.12% | 48/88 | 11 | 0.16·C |
| all 84 architect-board picks, 1%, floor 1,000 | −3.72% | −1.81% | 53/88 | 7 | over 2× |
| all 426 seeds, 1%, floor 1,000 | −3.31% | −1.29% | 51/88 | 10 | over 2× |
| idealized: all 426, shadow runs, free switching | −4.03% | −1.94% | 61/88 | 2 | – |

v1 by cache size, compared with 3L-Cache:

| cache size (% of objects) | cells | v1 | 3L-Cache |
|---|---|---|---|
| <0.5% | 17 | −2.38% | −1.88% |
| 0.5–2% | 25 | −3.51% | −2.02% |
| 2–8% | 32 | −2.67% | −1.59% |
| ≥8% | 14 | −6.23% | **−7.63%** |

By trace, averaged over the 4 sizes, v1 beats 3L-Cache on 12 of 22 traces.

### Train-long, realistic

| | mean | median | beats 3L-Cache |
|---|---|---|---|
| 3L-Cache | −2.64% | −0.02% | – |
| v1, in-sample | −4.51% | −1.46% | 67/100 |
| v1, 5-fold cross-validation by trace | −4.20% | −1.40% | 72/100 |

### Verdict against the open-source bar

The bar: on heldout-long, beat 3L-Cache on mean and median, on at least two-thirds of the traces, twice.

- v1 beats 3L-Cache on the mean (by 0.7 pt) and the median (by 0.55 pt).
- It is worse than S3-FIFO in a third as many cells as 3L-Cache.
- **It wins only 12/22 traces (needs 15). v1 does not meet the bar.**

## Findings (counterfactuals)

All of these are on train-long unless marked held-out. Logs are in `scripts/vqsim/v1/logs/`.

1. **Choosing is the bottleneck, not the design space.**
   - Per workload, the best of 426 designs in hindsight gets −5.03%, but picking it from the first 20% gets −3.81%.
   - Re-choosing online closes most of the gap: −4.59% idealized on train, −4.03% on held-out.
2. **Online beats static for every candidate set,** by 0.85–1.1 pt (held-out).
3. **No hysteresis.** A margin of 1% or 3% before switching loses (−3.85% and −3.73% vs −3.94%).
4. **Take over late.**
   - Running the selector from window 4 instead of taking over at the 20% point loses 0.8 pt.
   - The early choice is noisy, and S3-FIFO is a safe default meanwhile.
5. **Switching has a real cost of about 0.65 pt.** It is the idealized schedule replayed on a live cache.
   - Of three takeover mappings, the stage mapping is best: −4.51%, against −4.12% by counter and −4.39% all to main.
6. **A miniature simulation needs 100–200 objects.**
   - Without a floor, decisions fall apart (−2.70%).
   - A floor of 1,000 buys nothing over 200.
   - 32 candidates are about as good as all of them.
7. **Longer history hurts.** Ghosts of 2–16× the cache, with ghost hits admitted to main, are worse at every cache size, monotonically. At small caches, the best fixed design goes from −0.16% (0.9×) to +3.07% (16×).
8. **Architect findings, all consistent with the literature.**
   - A ghost hit must re-enter main; re-entering a stage that feeds the ghost loops: +5% to +25%.
   - Counters deeper than 2 bits buy nothing.
   - Age-guided refusal of reinsertion is a clean negative.
   - Delayed reinsertion in main hurts.
   - A two-step probation does not beat a single small queue.
   - Families with catastrophic members poison per-workload choice.
9. **Agent designs transfer to held-out workloads.**
   - As candidates, the 84 designs the board chose (64 by agents) beat the 426 seeds on held-out: −3.72% vs −3.31% mean, −1.81% vs −1.29% median.
   - Within the memory limit, they add 0.24 pt mean and 0.67 pt median to v1.
10. **Boards saturate.**
    - The static objective stopped improving after generation 2 (−4.03% → −4.05% over generation 3).
    - The online objective (generation 4, 32 families) moved its portfolio from −5.08% to −5.11%.
    - More designs from the same space no longer help. A new mechanism is needed.

## Memory

- Designs use 1.0–1.66× S3-FIFO's metadata (vq.py, CPython).
- The selector adds 32 miniature caches of max(0.5%·C, 200) objects.
- Measured like the sandbox (tracemalloc, a synthetic trace), `OnlineV1` vs S3-FIFO:
  - 1.61× at C = 50,000;
  - 3.20× at C = 5,000.
- So v1 fits the 2× limit for caches of about 20,000 objects and more. Below that, the fixed floor (32 × 200 objects) dominates.
- **In C** (`scripts/vqsim/vqonline.h`, bytes allocated vs S3-FIFO in the same engine; `scripts/vqsim/v1/logs/memory-c.txt`), v1 takes 2.45× at C = 5,000, 1.53× at 20,000 and 1.99× from 50,000 up.
  - At small caches the selector's fixed floor dominates (about 0.94 MB for 32 × 200 objects and their ghosts).
  - At large caches the selector is 0.15× and the live cache 1.83×: 4 of the 32 candidates keep a ghost of 1.5–2× the cache (S3-FIFO's is 0.9×), and the engine sizes its tables for the largest.
- **Ghost-capped variant (2026-11-18, train only).** The same greedy restricted to designs whose ghost is at most 0.9× (`GHOST_MAX=0.9` in `vq_portfolio.py`; `scripts/vqsim/v1/portfolio-ghostcap.json`; log `portfolio-ghostcap-cv-train.txt`):
  - 5-fold CV, idealized: −4.32% / −1.41%, against −4.37% / −1.50% uncapped. Capping costs 0.05 pt of mean.
  - 5-fold CV, realistic (live cache, same selector as v1; log `portfolio-ghostcap-cvreal-train.txt`): −3.96% / −1.07%, against v1's −4.20% / −1.40%. Capping costs 0.24 pt of mean and 0.33 of median, mostly at the smallest caches (−2.75% vs −3.38%) and the largest (−4.66% vs −5.14%), so the deep-ghost designs earn their memory there.
  - Memory in C: 1.14× at C = 200,000, 1.17× at 50,000, 1.31× at 20,000, 2.13× at 5,000 (the floor remains).
  - It shares 23 of its 32 designs with v1. It is not frozen and has not been tested on held-out.
- **Open question for the project.** Is the limit purely relative, or relative plus a small constant? With the ghost cap, only the selector's fixed floor breaks a purely relative 2× limit, below about 10,000 objects. Storing miniature-cache entries as fingerprints, or fewer candidates at small caches, would cut that.

## CPU

Each policy's cost relative to S3-FIFO in its own simulator, measured on one train-long trace (15.4M requests), single thread, with `scripts/vqsim/study/vq_cpu.py` (2026-11-17):

| cache size | 3L-Cache / S3-FIFO (libCacheSim, `process_trace`) | v1 / S3-FIFO (vqsim) |
|---|---|---|
| 2,211 | 6.4× | 7.9× |
| 6,632 | 3.4× | 3.3× |
| 22,108 | 3.0× | 2.8× |
| 66,324 | 2.8× | 1.7× |

v1 costs about what 3L-Cache costs. It is cheaper only at large caches; at small ones the floor of 200 objects per miniature simulation makes its 32 sampled simulations as expensive as 3L-Cache's model, or more. So v1 has no clear CPU advantage. In both simulators the overhead is per request, against an S3-FIFO that is 8× faster in vqsim than in libCacheSim.

## Caveats

- **Held-out reuse.** heldout-long has been used for four pre-registered tests: idealized, seeds portfolio, candidate sets, and v1. Each was fixed before it ran, but reporting the best of them is mild selection. A fresh suite should decide the open-source question.
- **Unit object size only.** Size-aware is next.
- **Windows.** The evaluation's windows are the trace length divided by 40. `OnlineV1.PERIOD` (500K requests) stands in for that on a live cache.
- **Protocol asymmetry.** 3L-Cache runs from the start of each trace, while v1 takes over at 20%.

## Reproduce

```
cc -O2 -o vqsim scripts/vqsim/vqsim.c
# Suites: scripts/vqsim/study/vq_prepare.py (dense ids + baselines per trace) on suites/{train,heldout}-long.json
# Full runs of a design list: scripts/vqsim/study/vq_run3.py <suite dir> <designs.jsonl> <tag> <workers>
# Portfolio (greedy, idealized; cv = 5-fold by trace): scripts/vqsim/study/vq_portfolio.py <suite> <tag> 32 r1,bp [cv]
# Realistic selector (live cache, mini-sims): scripts/vqsim/study/vq_portfolio_real.py <suite> <portfolio.json> <tag> <workers> 32:0.005:200
# Architects: vista --profile cachelab --backend claude run --role architect --board B --vq-suite S --vqsim V --objective {static,online} ...
# Boards: scripts/cachelab_vqboard.py {seed,status,final,import-runs,copy-findings}
# Project page: scripts/cachelab_site.py writes site/data.js from the logs; open site/index.html
```

The study scripts are kept as they ran. They expect the node layout (`/mydata/vq`, `/mydata/vqsim3`).

## Next

1. A fresh final suite for the open-source decision.
2. The memory accounting decision.
3. A new mechanism for the large caches where 3L-Cache wins on held-out, and the small caches where it wins on train: reuse-distance-aware admission, not more history.
   - Where 3L-Cache still wins on train (`scripts/vqsim/study/vq_gap.py`, log `gap-train.txt`, v1 realistic 5-fold CV): it leads by more than 0.5 pt in 17 of 100 cells (v1 leads in 60). 13 of the 17 are caches under 2% of the objects, spread over 10 of 25 traces. There the gap is large: 4.2 pt mean, 2.6 median, up to 22 pt.
   - In only 6 of those 17 cells does any other online policy beat v1, and it is mostly W-TinyLFU (largest case: a 0.41% cache, v1 −8.6%, 3L-Cache −31%, W-TinyLFU −29%). Its frequency-sketch admission gate is what the FIFO design space lacks: a TinyLFU-style gate in front of main is the first mechanism to try.
   - v1 captures 21% of OPT's mean gain on train, 3L-Cache 13%.
   - Tried (train only, no LLM): a TinyLFU-style gate in the engine (`gate=`, `gmode=`; `scripts/vqsim/README.md`). A count-min sketch compares objects with the tail of the main stage. It can rescue objects about to leave, filter promotions into main (W-TinyLFU's admission test), or both.
     - Handover runs (`study/vq_gate.py`, log `gate-train.txt`): on S3-FIFO the filter alone is the only mode that does not hurt on average (−0.33% vs S3-FIFO; rescue +0.33%, both −0.05%).
     - With a per-cell hindsight pick, gated designs reach −14.6% where W-TinyLFU beats v1 (W-TinyLFU −15.8%, v1 −9.5%), so the gate does what W-TinyLFU's does. Where 3L-Cache leads, they reach −7.8% against 3L-Cache's −10.1%. W-TinyLFU is poor there (+9.4%), so that lead is not about frequency admission.
     - Online (`study/vq_gate_online.py`, log `gate-online-train.txt`, the frozen v1 portfolio with a gated twin of each design, in-sample in every arm): the mean moves from −4.51% to −4.56% (both twins) or −4.59% (filter twins). Results are mixed per cell (both twins: 8 cells better, 12 worse, one gain of 16 pt), and the traces beaten stay at 18–19 of 25.
     - Not a v2 mechanism on its own. The gate is kept in `vqsim` (off by default), not in v1 or `vqonline.h`.
   - Tried next (2026-11-22, train only, no LLM): a reuse-distance rule in the engine (`rd=`, `rdmode=`; `vqsim` only, compiled in with `VQ_RD`, so v1, `vqonline.h` and the libCacheSim port are unchanged). Each object keeps the gap between its last two requests. A move is judged by the object's predicted wait for its next request, against the residence time of the stage it would enter (a running mean). The rule can refuse promotions or reinsertions, send a ghost hit with a long gap to the small queue instead of main, or rescue an object that does not qualify.
     - Handover runs (`study/vq_rd.py`, log `rd-train.txt`, S3-FIFO and v1's first 8 designs, each with every mode at 0.5, 1 and 2 times the residence time):
       - refusing reinsertions hurts (S3-FIFO +0.4% to +2.6%): a hit in main is worth keeping even when the gap is long;
       - promotions barely change (−0.05% at most), and rescue hurts (+0.2% to +0.6%);
       - **ghost admission helps**: S3-FIFO −0.96% mean, −3.35% at caches of 8% of the objects or more, and −1.26% in the 18 cells where 3L-Cache leads v1 by more than 0.5 pt;
       - it is catastrophic (+60% to +258% at small caches) in designs where ghost hits are main's only way in (a small queue with a D-FR delay, or a pass-through entry stage): main stops being refreshed.
     - Best per cell in hindsight: −4.33% without the rule, −5.15% with it; in the cells where 3L-Cache leads, −6.36% → −8.13% (3L-Cache −9.38%). A design with the rule is the best in 78 of 100 cells.
     - Online, chosen honestly (log `rd-portfolio-train.txt`): portfolios by 5-fold CV by trace from v1's pool (426 seeds + 84 board picks) with and without 1,095 ghost-admission twins (rd = 0.5, 1, 2), then the realistic selector. The v1 pool reproduces v1's numbers exactly (−4.20% / −1.40%, 72/100 cells, 19/25 traces beating 3L-Cache).
       - Choosing all 32 from the larger pool: idealized −4.37% → −5.01%, but realistic only −4.20% → −4.44%, with the median worse (−1.14%) and 17/25 traces. Exact shadows instead of 0.5% samples give −4.59% (18/25), so sampling costs the rd designs about 0.15 pt; taking over and switching the live cache costs both pools 0.3–0.4 pt.
       - Keeping v1's 32 and letting the greedy add 16 (it picks rd designs for 14–16 of them): −4.62% / −1.43%, 72/100 cells, 9 worse than S3-FIFO, 19/25 traces. The same 48 slots without rd give −4.29% / −1.38%, 70/100, 11 worse, 19/25. So the rule adds about −0.33 pt mean and costs nothing else on train; the selector's memory grows from 0.31 to 0.46 C in the median cell.
       - Per trace it cuts v1's worst loss to 3L-Cache from +9.7 to +2.8 pt. About 7 of the 25 traces are within ±0.5 pt of 3L-Cache, where both are close to S3-FIFO, so the count of traces beaten moves by noise there.
   - Where the online selector loses against its idealized version (`study/vq_switchcost.py`, train, exact shadows, the same CV portfolios): the takeover at the split costs 0.13 pt (v1 pool) and 0.23 pt (with rd twins), and the later switches about 0.19 pt; the leader changes 1.6 times per cell after the split. Switching still pays: picking once at the split and never switching gives −3.68% (v1 pool) against the selector's −4.05%, better in 18 cells but worse in 31. A takeover that merges the old stages by the time objects entered them (`takeover=3`, `vqsim` only) changes nothing (−4.20% either way), so the cost is the old design's contents, not their order.
   - Gate and rule together (log `rdgate-portfolio-train.txt`): 1,339 gate twins (filter, rescue + filter, filter + rd ghost admission) in the pool, v1's 32 extended by 16, realistic selector at K=48:
     - with gate twins: −4.73% / −1.40%, 73/100 cells, 11 worse than S3-FIFO, 19/25 traces (the greedy adds about 5 gate and 9 gate + rd designs per fold);
     - with gate and rd twins: −4.56% / −1.43%, 72/100, 10 worse, 19/25; with rd twins alone −4.62% / −1.43%; the same 48 slots without either −4.29% / −1.38%.
     - So the FIFO extensions are worth 0.3–0.45 pt on the mean, and none changes the traces beaten. v1 loses 6 of 25 train traces. One is an exact tie, where nothing moves the miss ratio. Two lose by under 0.15 pt on the trace mean; 3L-Cache's edge there is at the smallest cache (−2.6% vs S3-FIFO). On the other three, 3L-Cache wins at the smallest caches, by up to 31% vs S3-FIFO, and is the best published algorithm there: its learned eviction, which ranks objects by predicted next request, is what the FIFO designs lack. The extensions halve the worst of these losses (+9.7 → +4.2 to +4.8 pt) but close none.
   - **Pool screening at the split** (2026-11-22, train only; log `screen-train.txt`). The idealized selector shows that the pool has a better design than 3L-Cache for most traces the selector loses, and the CV portfolio does not hold it. So at the split the selector also screens a pool: it replays its own sampled prefix through a miniature of each pool design and adds the 16 with the fewest sampled misses as warm candidates (`vqsim` online `pool=`, `screen=`). This is exact: the run matches one where those designs were candidates from the start (tested). v1's CV 32 stay; realistic selector (0.5%, floor 200):

     | selector (train-long, realistic) | 5-fold CV by trace | leave one dataset out |
     |---|---|---|
     | v1 (CV 32) | −4.20% / −1.40%, 19/25 traces | −3.84% / −1.17%, 17/25 |
     | v1 + gate twins (CV 48) | −4.73% / −1.40%, 19/25 | −4.40% / −1.17%, 18/25 |
     | CV 32 + 16 screened from all 2,921 designs, full log | −4.90% / −1.55%, 20/25 | −4.89% / −1.55%, 20/25 |
     | the same, log of at most max(C, 65,536) entries | −4.63% / −1.50%, 20/25 | |
     | CV 32 + 16 from a 64-design coverage pool, same log bound | −4.66% / −1.37%, 19/25 | |
     | 3L-Cache | −2.64% / −0.02% | |

     - Screening fixes dataset shift: leaving a whole dataset out costs v1 0.36 pt and 2 traces, and costs screening nothing. Held-out datasets are where v1 fell short (12/22 traces).
     - The log costs memory. Unbounded, it is the sampled prefix: a median of 1.6 C entries, but up to 2×10⁵ C on traces with a few thousand objects. Bounded to max(C, 65,536) entries, it wraps in 44 cells. A wrapped log is replayed for the CV 32 as well; with `wrapmode=1`, the CV 32 keep their full history and each screened design gets the leader's history, scaled by its replay score. Bounding costs 0.27 pt, all of it in cells whose prefix log is over 5× the bound: 16 of them are on traces with 1.6k–60k objects in total. A bound of max(C, 16,384) costs 0.51 pt. The log holds 4-byte ids (plus a window mark), not cache entries.
     - The replay costs CPU once, at the split. For all 2,921 designs it comes to a median of 5.2 miniature steps per request of the whole trace. The 32 miniatures themselves cost about 0.31 steps per request in the median cell, so the replay is about 17× that. A pool chosen for coverage is as good for 1/27 of that cost (0.19 steps per request). That pool is a greedy facility-location pick, per CV fold, of designs that complement the fold's CV 32 (`study/vq_covpool.py`; idealized, 64 designs reach −5.27% against the full pool's −5.36%). Members 32–255 of a 256-design CV greedy portfolio do worse (−4.56% at the same bound): past K ≈ 96 the greedy's additions are arbitrary (idealized K=96 −5.26%, K=256 −5.26%).
     - How many slots go to fixed and to screened designs barely matters (16 + 32, 32 + 8, 32 + 16, 32 + 32, 48 + 16 all give −4.43% to −4.60% with the 224-design pool).
     - Still lost on train, in every variant: two traces by under 0.1 pt (3L-Cache −2.6% at their smallest cache, the selector about −1.5%), one exact tie, and two where the pool itself falls short of 3L-Cache. On those two, the best design per cell in hindsight averages −11.9% and −16.5% against 3L-Cache's −12.1% and −17.1%. 3,531 gate × rd-ghost cross designs (`xr`) add about 0.2 pt to the best design per cell on one of them, and nothing to the selector: with them in the screened pool it gets −4.65% / −1.46%, 19/25 traces (−4.63% with leave one dataset out), against −4.63% / −1.50%, 20/25 without, at twice the replay cost (11.3 steps per request). Idealized, the selector beats 3L-Cache on 23/25 traces; the gap to realistic is sampling noise, the takeover at the split, and later switches.
     - **Pre-registered heldout-long round 1** (`scripts/vqsim/v2/PREREG.md`, the user's choices: log bound max(C, 65,536), one round). The configuration is v1's frozen 32 + 16 screened from all 2,921 designs. Result: −3.58% / −1.82%, 53/88 cells, 15/22 traces. 3L-Cache: −2.73% / −1.24%. v1, rerun in the same job: −3.42% / −1.79%, 12/22. That passes all three criteria, but the third with no margin: four of the 15 wins are by under 0.1 pt. The open-source bar needs a second round.
     - **Pre-registered final-long round 2** (`scripts/vqsim/v2/PREREG-final.md`; final-long's first use, v1 and v2 in one job). v2: −3.17% / −2.30%, 42/64 cells, **7/16 traces**. v1: −3.30% / −2.38%, 42/64, 9/16. 3L-Cache: −2.81% / −1.74%. **v2 fails the bar**: it beats 3L-Cache on mean and median, but 11/16 traces were needed. It is also worse than v1 there, mostly at caches of 2–8% of the objects (−3.97% against v1's −4.44% and 3L-Cache's −4.54%), where v1 seldom switches and v2 switches about 3 times. So screening's train gain (0.43 pt) shrank to 0.16 pt on heldout-long and reversed on final-long (−0.13 pt). Nothing was committed.
     - **Why it failed** (post-hoc diagnosis; `scripts/vqsim/v2/PREREG-final.md`, log `v2/logs/final-diagnosis.txt`):
       - It's the cost of switching the live cache, not the choice of designs. With exact estimates and free switching, v2's candidates would beat 3L-Cache on 15/16 final-long traces (v1's on 12/16), and the prefix predicts the rest as well as on train.
       - Takeovers and switches cost v2 0.62 pt there (v1 0.37; both about 0.19 on train), which flips 7 traces. v2 switches 3.45 times per cell, on sampled leads of a median 0.04%.
       - On final-long each switch costs about 3× as much as on train and lasts longer. Part of the cost never fades: path dependence after a takeover.
       - The miniatures run cold and never take over a cache, so they can't see this.
       - The pool also holds 40 static-set designs, whose main stage is never refreshed. They caused the two worst cells (+76 and +23 pt after the takeover), but removing them changes little (−3.19%).
       - A fix must price switching into the decisions. Train-long can't validate one, because switching there is nearly free.
     - **Attempts to cut the switching cost** (development, post hoc; log `v2/logs/dev-switching.txt`; final-long used as a development set). None helps; v2 stays at 7/16 final-long traces or fewer in every variant:
       - Trying each switch first on a miniature mirror of the live cache (`trial=H`): train −4.26%, final −3.15%. A one- or two-window replay holds all of a switch's transient and little of its gain, so it refuses good takeovers in large caches. With the split exempt, it kept one train cell in a poor split pick (+47% against −20%).
       - Hysteresis margins: worse on both suites. Switching after the split is how the selector leaves poor split picks.
       - Takeover by counter or all to main: worse than the stage mapping on both suites.
       - Drain (new design starts empty, the old objects only serve hits): far worse (final −0.77%). It loses the warm working set. Removed.
       - Miniatures that take over the mirror's cache at the split, so post-split scores include each design's takeover cost (`reseed=1`): final −3.30%, train −4.57%, no trace gained.
     - **Cheap takeovers by construction** (the user's option 1: keep the bar; same log). Between S3-structured designs (two stages, small → main, main → itself; `study/vq_family.py`) a takeover maps stage to stage. On final-long such takeovers cost 3–5× less than others (v2: +0.08 against +0.38 pt per split takeover, +0.04 against +0.11 per switch). But a selector over the family alone does worse than v1 on both suites (32 candidates: train −4.12%, 17/25; final −3.03%, 8/16), at every candidate count and sampling rate tried (8–32 candidates, 0.5–2%). Its realistic result trails its exact ceiling (−4.71% on train) by 0.6 pt, and denser sampling does not close the gap.
     - **Replay takeover** (`vqsim` online `replay=L`; same log). At a takeover the next design rebuilds its own state by replaying the live cache's last L·C requests, then is reconciled with what is cached. Short logs do worse than the stage mapping; from about 128·C on it does better on final-long without costing train: v2 −3.38%, 11/16 (train −4.66%, 20/25), v1 −3.39%, 10/16. The log costs 512 bytes per cache slot. Since final-long is now a development set, this needs a fresh suite before it counts.
     - **A harness for single policies** (`scripts/vqsim/evolve`): a model writes policies in a C contract (`policy.h`), which run cold over whole traces as 3L-Cache does, with all state in a bounded arena. The driver owns the cache's contents and checks every victim. The S3-FIFO seed reproduces `vqsim` exactly.
     - **The per-trace criterion turns on near-ties.** Across six variants on final-long, 9 traces are won by at least one and 7 by all. Six of the 7 that none wins are within 0.17 pt of 3L-Cache; one is lost by 2.2 pt.
     - **Why the per-trace bar is hard: a bound from the published algorithms.** Pick, for every cell in hindsight and with no switching cost, the best of the 14 published classic algorithms (LRU, FIFO, CLOCK, SIEVE, ARC, LIRS, 2Q, CLOCK-Pro, MQ, S3-FIFO, W-TinyLFU, LeCaR, Cacheus, LHD). That beats 3L-Cache on only 17/25 train-long traces and 9/16 final-long traces. On about 40% of traces 3L-Cache beats every classic algorithm, by 0.1–1.9 pt, while Belady is still 20–36% below S3-FIFO.
     - **Where the learnable signal is** (`study/vq_tailfeat.c`, which records S3-FIFO's tail decisions exactly on all 100 train cells, with each object's next request):
       - **Main queue's reinsertion:** already near the best that the counter, the hits and passes, the time since the last access, the last gap, the request count and the age allow. 64.7% → 65.8% of kept objects are reused within one pass; about +2 pt at 16 passes.
       - **The small queue's promotion (the filter):** large gains, at the same promotion rate. 24% → 39% within one pass and 32% → 56% within 16, using the counter, the time since the last access relative to the pass, and the request count.
       - **This needs a table learned per workload;** one table for all workloads gets about 30%. That is the ingredient fixed heuristics lack.
       - **Learned online** (`evolve/seeds/s3filt.c`), the filter alone gives −0.6% vs S3-FIFO. Its ceiling is about 3% of misses, since the ghost already re-admits the filter's mistakes.
       - **Failed probes:**
         - evicting by predicted next access: +15.9%;
         - a learned main-queue rule: +1.1%;
         - TinyLFU-style admission into S3-FIFO's main: +12% to +39%.
     - **LLM search for single policies** (`evolve/evolve.py`; gpt-6.2-sol, gpt-6.2-astra and gpt-5.5 through an OpenAI-compatible gateway; train-long only, anonymous trace indices).
       - **Fixed policies** (`runs/evolve-1`): at best 13/25.
       - **A knob contract** (`policy_knobs`/`policy_set_knob`): the driver runs one miniature per setting and moves the live cache to the setting missing least. Each setting must take effect on the running cache, without a rebuild.
       - **Results:** this makes mechanisms that win on some workloads safe. The small-share knob alone gives −2.14%, and 100 programs (`runs/evolve-2`) reach 17/25 traces at −3.68%. But on final-long (one dev check) the best gets −2.91%/−1.04%, 10/16, against 3L-Cache's −2.81%/−1.74%.
     - **Pre-registered fresh-long test** (`v2/PREREG-fresh.md`, log `v2/logs/fresh-test.txt`).
       - **Suite:** 20 unused whole traces; 18 scored after the pre-registered drop rule.
       - **Bar:** mean, median and more than half of the cells.
       - **Result:** v2 with replay takeover (128) passes: −4.47% / −2.33%, 39/72 cells, against 3L-Cache's −4.27% / −2.21%.
       - **Caveats:** the bootstrap interval of the mean difference is [−1.08, +0.50] pt, and the trace count is 9/18. So it is a narrow pass, not a clear win.
       - **References:** v2 without replay −4.28%, v1 −3.80%, and the LLM knob search's best single policy −3.48%.
4. A C/libCacheSim port of `OnlineV1`. Started: `scripts/vqsim/vqonline.h` is `OnlineV1` in C, keyed by object id, with memory bounded by the cache and ghost, exact with `vq.py` (tested). On a synthetic 5M-request trace it costs 1.5× its own S3-FIFO at C = 20,000 and 200,000, and 3.2× at C = 2,000, at about 0.02–0.1 µs per request. In libCacheSim (`scripts/vqsim/libcachesim/`, algorithm `onlinev1`, exact with `vqonline`), on the same 5M-request synthetic trace, it takes 1.36× the time of libCacheSim's own S3-FIFO at C = 2,000, 0.47× at 20,000 and 0.39× at 200,000 (libCacheSim's S3-FIFO is built from three sub-caches with their own hash tables). A same-build, same-trace comparison with 3L-Cache is next; libCacheSim needs LightGBM for it.
5. Size-aware variants.
