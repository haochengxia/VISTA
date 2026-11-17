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
- **Open question for the project.** Is the limit purely relative, or relative plus a small constant? Storing ghost and miniature-cache entries as fingerprints would also cut this.

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
```

The study scripts are kept as they ran. They expect the node layout (`/mydata/vq`, `/mydata/vqsim3`).

## Next

1. A fresh final suite for the open-source decision.
2. The memory accounting decision.
3. A new mechanism for the large caches where 3L-Cache wins on held-out, and the small caches where it wins on train: reuse-distance-aware admission, not more history.
4. A C/libCacheSim port of `OnlineV1`.
5. Size-aware variants.
