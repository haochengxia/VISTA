# CacheLab roadmap: what to learn from self-improving agents and from cache papers

Written 2026-11-14 from two literature surveys (sources linked below). It records the
evidence behind the next changes, and the order they are planned in.

## Evidence

### Self-improving agent projects

- **Closest prior work: Vulcan and PolicySmith (UT Austin)** ([2512.25065](https://arxiv.org/abs/2512.25065), [policysmith](https://github.com/ldos-project/policysmith)).
  - Both evolve cache-eviction code on libCacheSim for one workload.
  - Neither keeps experience between runs, and neither shows the agent images.
  - Vulcan searches on the first 1M requests but scores the full trace, prefix included. Our split (observe 20%, score the remaining 80%) is the cleaner protocol.
- **Images help as a progress anchor.** In a GECCO'26 study ([2602.10891](https://arxiv.org/html/2602.10891v1); same model, ~20-minute runs), an LLM curriculum designer did significantly better with numbers plus a *convergence plot* than with numbers alone. Adding behavior/trajectory plots on top gave nothing more.
  - MLES ([repo](https://github.com/QingL2000/MLES)): behavior images gave 96.4 vs 79.3 on Car Racing.
  - Plots beat number strings on time series ([2410.02637](https://arxiv.org/abs/2410.02637)).
  - Screenshots alone did not help VoyagerVision.
  - Lesson: an image has to carry decision-relevant information.
- **Feedback granularity** (Berkeley ADRS, [2510.06189](https://arxiv.org/abs/2510.06189), one scheduling case study):
  - aggregate score only: 7.7% improvement;
  - aggregate plus the worst 5 scenarios: 13.0%;
  - full per-trace detail: 10.2% (the authors blame overfitting).
- **Playbooks: itemize, don't rewrite.**
  - ACE ([2510.04618](https://arxiv.org/abs/2510.04618)) keeps bullets with ids and helpful/harmful counters. Updates are small deltas merged by code, then de-duplicated, over several epochs.
  - Whole-document rewrites caused *context collapse*: 18,282 → 122 tokens in one step, with accuracy below the no-memory baseline.
  - Self-curation without labels reinforces wrong rules (Dynamic Cheatsheet).
  - ExpeL keeps vote counters and contrasts success with failure on the same task.
  - RIZZ promotes rules only on verified outcomes and branches memory by task family.
- **Failure modes of evolutionary coding agents** ([2605.20086](https://arxiv.org/html/2605.20086v1)):
  - about 30% of added lines re-add code deleted earlier;
  - late gains are mostly constant tuning, which a 24-evaluation tuner matches;
  - overfitting to visible tests is common.
- **Integrity: agents game visible checkers.** DGM removed hallucination markers; ADRS saw a faked 90% cost cut.
- **Other mechanisms worth knowing:**
  - per-instance Pareto fronts and merges (GEPA);
  - novelty rejection (ShinkaEvolve);
  - cascade evaluation (AlphaEvolve);
  - an overseer that nudges a looping agent (SICA);
  - pruning weak lineages early (no harness wins everywhere, [2607.18235](https://arxiv.org/abs/2607.18235));
  - crediting a lineage by its descendants (Huxley-Gödel).

### Cache papers: how they diagnose

Most diagnostic figures in the field are comparisons, either between policies or against OPT. Views we lack, by paper:

| Diagnostic | Question | Papers |
|---|---|---|
| Avoidable misses vs Belady | Which misses were mistakes, not inevitable? | LRB, Hawkeye, S3-FIFO Fig. 4 |
| Good-decision ratio (eviction beyond the Belady boundary), victim rank | Are evictions right, per queue? | LRB, LHD Fig. 8, S3-FIFO precision |
| MRC with OPT, equivalent cache size, cliffs | How far from optimal, and at which sizes? | Mattson, LRU-K, LHD, Talus, SHARDS |
| Handover / restart recovery | How much does a switch cost, and how fast does it settle? | 2Q responsiveness, CacheLib restart, SIEVE switch, S4-FIFO Fig. 12 |
| Oracle at one decision point | Which decision has the most headroom? | SIEVE X-Belady, GL-Cache |
| Knob sweep and best knob per window | Is the knob in the right place, and does it drift? | TinyLFU, AdaptSize Fig. 3, S3-FIFO Fig. 10/11, S4-FIFO |
| Demotion speed vs precision | Is S sized right? | S3-FIFO Fig. 10 |
| Cache composition (resident under OPT, dead, hoarded) | Is space wasted on cold objects? | SIEVE, LeCaR, QD-LP Fig. 3 |
| Population view (box plots, worst trace, win/tie/loss) | Is it robust across traces? | S3-FIFO, SIEVE, Cacheus, S4-FIFO |

We already have windowed pairwise miss ratio (view_overview, view_diff), reuse distance and frequency (view_workload), object lifetimes (view_objects), and hit positions (view_positions).

## Plan

### Status (2026-11-15)

- **Done** (tests pass; goes to the node after round 1 of xp2):
  - Phase 1: view_opt; the dashboard with worst windows; the itemized playbook with operations, host credit, op-replaying merges and a knowledge-collapse guard.
  - view_handover and view_knobs from Phase 2. view_knobs is free host-side knob tuning: S3-FIFO with one knob changed at a time, per window, with the best static value and the best value per window (an upper bound for adapting that knob). Its sweep runs as one batch of policies (`RunStore.simulate_many`).
  - Duplicate guard: test_policy says when code is the same program as an earlier experiment (comments, docstrings and formatting ignored); run_experiment says when an experiment's misses match an earlier one's (or S3-FIFO's) in every cell, i.e. the change altered no decision.
  - A fix to windowed views: the last window of any range ending before the trace's end also counted the requests after it.
  - A fix to memory accounting. Import-system memory (~0.25 MB) used to be charged to subclasses of S3-FIFO but not always to S3-FIFO itself, so knob changes looked 2.5× over the bound at small caches.
- **Round 1 of xp2** runs on the earlier code: free-text playbooks, but with the memory fix, view_opt and the dashboard. Its lineages' playbooks reach the analyst as proposals, which it turns into rules.
- **Gateway resilience** (2026-11-15, in both node checkouts; not agent-facing): Claude Code retries a failed request up to 15 times (its maximum) instead of 10. When it still gives up on a 5xx, the harness resumes the same conversation after a pause, up to 6 times per step, before falling back to a fresh session. Before this, 3 of the first 11 round-1 sessions died: one at the startup check, two after hours of fresh restarts that each lost the conversation.
- **Segment timeout** (2026-11-15 05:40): the transport limited one agent process to 300 s × (experiment budget + 1), i.e. 10–20 minutes for 1- and 3-experiment test runs, so they kept restarting without their conversation. CacheLab now defaults to 4 hours (`SEGMENT_TIMEOUT_SECONDS`). The first plain test arm ran under the short limit; it was stopped, kept in runs/xp2/test-aborted-short-timeout, and restarted.
- **Local only, for round 2** (~/VISTA-xp-next stays frozen until both test arms finish, so they run on one version): test_policy measures memory at the smallest, a middle and the largest cache size and says which sizes are over the limit. A plain test run lost its score to a 2.0–2.3× memory use at small caches that the one-size check never showed.
- **Local only, for round 2:** a training run's committed decision is scored verified, like a test run (memory at every size, a checked rerun of the observed part). Before, training feedback was fast simulation only, so a decision over the memory limit was taught as a win. The commit reply now starts with INVALID and the reasons, and the case's outcome records `valid`. The host's counterfactuals of the other experiments stay fast.
- **Served-model guard** (2026-11-15, both node checkouts; not agent-facing): probes to kimi-k3 through the gateway are answered by qwen3.6-35b, and Claude Code only reports the model it asked for. Real sessions were all served kimi-k3 (audit of every assistant turn), but nothing checked it. Now the startup check and every assistant turn must be served by the model the startup check resolved; otherwise the run stops as a crash and the driver reruns it later. A gateway outage (503 overloaded from 12:10 on 2026-11-15) failed the round-1 analyst at startup and plain w09/w10; both are rerun once kimi-k3 serves again.
- **Code versions on the node:** ~/VISTA-xp keeps the round-1 code until round 1 ends (including the rerun of w02, which failed on a gateway timeout). ~/VISTA-xp-next has the current code and runs everything after it: merge, analyst, both test arms, round 2.
- **Larger training suite:** suites/train-real-2.json, 45 traces from 12 datasets with no overlap with the existing suites. Its baselines are computing into results/engineer/train2.

### Long traces from round 2 on (decided 2026-11-15)

A host-only study (S3-FIFO, 3L-Cache and LRU from libCacheSim; no agent) showed that the 1M-request held-out suite is biased against 3L-Cache:

| Setting | 3L-Cache vs S3-FIFO, mean | median | 3L-Cache better |
|---|---|---|---|
| Our protocol, 24 held-out traces, first 1M requests, last 80% scored | +0.94% | +0.10% | 40/96 cells |
| Same, whole trace scored | +1.79% | +0.68% | 36/96 cells |
| Our protocol, the 10 sources with at least 10M requests, first 10M | −1.75% | −1.10% | 24/40 cells, 7/10 traces |
| The same 10 sources at 1M | −0.19% | −0.53% | 6/10 traces |

- Scoring only the last 80% favours 3L-Cache slightly, since it learns during the prefix.
- The bias comes from length: 1M requests is too short for a learned policy to warm up.
- 14 of the 24 sources are shorter than 10M requests in total.
- LRU is worse than S3-FIFO in every setting (+3% to +7%), as published results say.

So round 2 and the open-source test use whole traces of 10M to 50M requests (`build-suite --full-length 10e6:50e6`):

- **suites/heldout-long.json** (prefix `hl`): 22 traces, 481M requests. They come from msr 2, cphy 3, alibabaBlock 3, tencentBlock 3, cf 3, akamai 3, fiu 2, metaStorage 2 and ARC 1, and share nothing with any training or dev suite. fiu-1, metaStorage-1 and ARC-1 are also in heldout-real as 1M prefixes, which were only ever tested.
- **suites/train-long.json** (prefix `tl`): 25 traces, 619M requests. Same datasets, plus metaCDN; none from fiu, metaStorage or ARC, and none from either held-out suite. A content check (hash of the first 1M requests) finds no copies between training and held-out traces.
- **Open-source bar:** on heldout-long, beat 3L-Cache on both mean and median, on at least two-thirds of the traces, in two rounds in a row.
- **The bar in numbers** (host-only, whole traces, last 80% scored, 4 sizes, 2026-11-15). 3L-Cache vs S3-FIFO on heldout-long:

  | Setting | mean | median | 3L-Cache better |
  |---|---|---|---|
  | Unit size, object miss ratio (CacheLab today) | −2.70% | −1.24% | 60/88 cells, 18/22 traces |
  | Real sizes, object miss ratio | −10.51% | −5.07% | 78/88 cells, 20/22 traces |
  | Real sizes, byte miss ratio | −4.86% | −0.00% | 45/88 cells, 10/22 traces |

  Heldout-long baselines (protocol v2, finished 2026-11-16; vs S3-FIFO on the last 80%, 88 cells):

  | Policy | mean | median | better than S3-FIFO |
  |---|---|---|---|
  | Prefix-tuned S3-FIFO (knobs tuned on the observed 20%, deployed at the split) | −2.71% | −1.16% | 75/88 cells, 20/22 traces |
  | 3L-Cache | −2.73% | −1.24% | 60/88 cells, 18/22 traces |
  | Best published per cell (picked in hindsight) | −4.63% | −2.53% | 79/88 cells, 22/22 traces |

  Caveat (found 2026-11-16): the prefix tuning picks GHOST_RATIO 1.5–2.0 in many cells, which uses 2.3–3× S3-FIFO's memory on real traces, over the 2× limit the agent must respect. So the tie below is not memory-fair. Tuning S3-FIFO's four knobs on the observed part already ties 3L-Cache: it beats it on 47 of 88 cells and is more consistent (75 vs 60 cells better than S3-FIFO). So an agent that only tunes knobs well reaches the bar; beating 3L-Cache clearly needs something beyond the knobs, worth about another 1% on average. The headroom view says where that is.

  To clear the bar at unit size, the agent's decisions must average better than −2.70% vs S3-FIFO, with a median better than −1.24%. With real sizes 3L-Cache gains much more (akamai −27% object, −46% byte), so the size-aware rerun after that will be a harder bar.

Leakage fixes for round 2 (local, tests pass):

- **File-only mounts:** the Docker sandbox mounts each trace file and policy file on its own, never its directory. Every materialized trace has a `.json` beside it naming its source, and fast mode runs agent code in that container. Before, `private/` (the full trace's manifest), `traces/cases/` (the analyst's case traces) and the dev traces were mounted whole.
- **Redaction** also hides the lab root as resolved (runs/ is a symlink to /mydata) and the simulator's install path, both of which appear in host-side tracebacks.
- **Training rounds ≥ 2** take workloads no earlier round of the root trained on. Wrapping around to a second pass needs `--allow-repeats` and is listed in `train/round-N/REPEATS.json`.

**Memory gate on decisions (round 2):** 5 of the first 9 scored plain test runs of round 1 were invalid for memory.
- w03 and w06 (both budgets): the agents' own policies were 2.0–2.5× at cache sizes the one-size test_policy never checked.
- w02 (budget 1): the agent submitted the MQ reference unchanged, at 5–7× S3-FIFO's memory, untested.

submit_policy and commit_decision now run test_policy's verified three-size check on the chosen policy (free if that program was already tested). They refuse it when it is over the limit at any size, so the agent can choose again. The prompt says the limit binds references too. In the test lab MQ is 3–5×; LRU, S4-FIFO and FIFO fit.

Container memory now scales with the batch's largest trace (`DockerSandbox.memory_for`). It was 6 GB for all cells of a batch, enough for 1M-request traces but not for full-trace scoring at 50M requests.

Measured on a 45.7M-request trace: one fast S3-FIFO cell takes 646 s under load (71K requests/s) and peaks at 3.17 GB. Two consequences:

- **Hindsight scoring has a host budget.** After a training run, scoring every experiment on the full trace (40 experiments × 4 sizes) would take hours per session. It now stops at 1e9 simulated requests (`run.COUNTERFACTUAL_REQUESTS`; 1M-request workloads never reach it). The order is S3-FIFO and the committed decision first, then the agent's experiments best first by their change on the observed part, then references. Records carry `complete`.
- **Claude Code's tool-call idle limit is raised to 3 h for CacheLab runs.** That limit is 30 minutes for stdio tool servers (Claude Code 2.1.220), and VISTA's tool server sends nothing until a call returns. A verified commit_decision on a 45M-request trace takes about 50 minutes. The new flag is `--claude-tool-idle-timeout` (`CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`); the hard limit is already about 28 h.

Tool timings on a 45.8M-request training workload (9.16M observed; tools/probe on the node; local sandbox, 6 workers, node at load ~45):

- Setup before the agent starts (no credit): lab 297 s, session with its first views 407 s.
- Views: 1–5 s for overview, diff, queues, positions and objects; view_opt 31 s; view_handover 43 s; view_workload 87 s; view_knobs 123 s (the first sweep).
- run_experiment 55 s.
- test_policy 1212 s. It is verified at 3 sizes, and verified mode pays a socket round trip per miss with tracemalloc on throughout. Unloaded and locally: fast 846K requests/s, verified 64K, verified without tracemalloc 164K. So it should take about 2.5 minutes once the baselines stop loading the node. tracemalloc defines the memory limit's measure, so it stays.

**Prefix-tuned S3-FIFO by coordinate descent (protocol engineer-handover-v2).** The v1 baseline tried the full grid: 288 configurations × 4 sizes over the observed 20%. On whole traces that is about 24 hours of 16 cores for train-long alone. v2 descends from the defaults instead: one knob at a time, two passes, each size separately, all values of a knob in one batch. That is about 26 configurations per size, using only the grid's values, so cached grid cells are reused. Evaluations refuse baselines from another protocol. The 1M baselines stay v1 and round 1 runs v1. The three train-long reports made with the grid are kept in results/engineer/train-long-grid-v1.

Per-cell time limits now grow with trace length (`RunStore.cell_seconds`). A fast cell keeps the 1M limit's floor on policy speed. A verified cell gets 1 s more per 2,500 requests, because verified mode is about 10× slower than fast (one socket round trip per miss).

### Phase 1: before the next training run

1. **Belady per request.** One Belady run per (trace, size), keeping per-request outcomes and the forward distance of every request.
   - New view **view_opt**, comparison-first:
     - per window, misses split into unavoidable, avoidable (OPT hits) and lucky;
     - avoidable misses broken down by cause (evicted from S, from M, ghost expired);
     - with compare_to, the avoidable misses B fixed and created relative to A.
   - New view **view_evictions**: good-decision ratio by source queue and window, against the Belady boundary, for two experiments side by side.
2. **Mission-control dashboard**, the attention anchor.
   - One image re-rendered every few experiments:
     - per-cell change vs S3-FIFO for the best experiment so far and the latest one (traces × sizes heatmap);
     - best so far over experiment number vs S3-FIFO and prefix-tuned S3-FIFO;
     - the experiment tree coloured by result;
     - the worst cells;
     - budget used.
   - The numeric arm gets the same content as a table, so the comparison stays fair.
3. **Feedback = aggregate + worst-5 cells/windows** by default, with drill-down on demand.
4. **Itemized library.**
   - Playbook bullets carry an id, a scope (signature features), supporting cases, and helpful/harmful counts.
   - Training sessions and the analyst emit operations (add, edit, upvote, downvote, retire), never whole rewrites. Code applies them and merges libraries.
   - Counts come from the host's hindsight counterfactuals: regret of the decision vs the best candidate on the scored part, credited to the bullets the decision cited.

### Phase 2

- ~~view_handover~~ (done).
- view_mrc: miss ratio curve with OPT and equivalent cache size, from sampled mini-simulations.
- ~~view_headroom~~ (done for round 2; free, no experiment): S3-FIFO replayed on the host with a perfect predictor in charge of one decision. The decisions are promotion from S to M, reinsertion at M's tail, and admission of a ghost hit into M; the view also runs all three together, with OPT for scale. The predictor knows whether an object is requested again within a horizon, a multiple of S3-FIFO's own time to cross M on that cell. Each decision keeps its best of 0.25–4×, and the text lists every multiple. One fixed horizon was misleading: promoting everything reused within one crossing floods M in a small cache, and the 'perfect' promotion added 3–5% misses at the two smallest test sizes. Like experiments, the predictor takes over at the handover point. Without a predictor the replay is the sandbox's S3-FIFO miss for miss (tested). 17 replays per cell run in a spawn process pool, cached on disk. On the 8 test cells 'all three' reaches 32–87% of OPT's gain (more at larger sizes); promotion is the largest single decision in every cell, reinsertion next. On a 45.8M-request training workload (9.16M observed, node at load ~30): 169 s at the smallest size and 129 s at the largest, workers under 2.5 GB, about as long as view_knobs. There 'all three' gains −33.0% against OPT's −39.4% at the smallest size, and matches OPT (−6.2%) at the largest. Promotion (−20.7%) and reinsertion (−16.3%) carry the headroom; ghost admission is worth −3% or less.
  - **Fix (2026-11-16 15:45, synced to ~/VISTA-r2 mid-round):** with a long horizon nearly all of M is 'soon', and the reinsertion oracle's unbounded scan cost O(C) per eviction. A view at C=68,230 ran for 3 h until Claude Code's tool limit, and the agent retried it, which stalled round-2 lineage-2. I stopped that session. The scan now examines at most 64 keys at M's tail and, if all are soon, evicts the one requested latest (headroom VERSION 3). Sessions already running keep the old code; new ones get the fix.
  - **Real-policy baselines in view_headroom:** 3L-Cache and the best of ARC, LIRS, W-TinyLFU and SIEVE, from the libCacheSim zoo on the observed trace only (no leakage). Misses are counted in the view's window, relative to libCacheSim's S3-FIFO in the same run. That S3-FIFO matches ours miss for miss on all 8 test cells. They show how much of the oracle's headroom known algorithms capture. They need a window that runs to the end of the trace (the zoo counts misses after a split), so they have no per-window numbers. The text now also says single-decision changes overlap and don't add up. On the 45.8M-request workload they add about 30 s per cell (200 s and 148 s in all). At the smallest size: all three −33.0%, OPT −39.4%, 3L-Cache −10.1%, ARC −5.4%. At the largest: all three and OPT −6.2%, ARC −2.4%, 3L-Cache −0.4%.
  - **Does the advice vary?** Study on 7 train-long workloads, one per dataset, at the smallest and largest sizes (14 cells, host only, 2026-11-15):
    - The size of the headroom varies a lot by cell: 'all three' ranges from −0.4% (tencentBlock, smallest) to −32.5% (tencentBlock, largest).
    - The ranking varies less: promotion is the largest single decision in 11 of 14 cells and reinsertion in 3. Ghost admission is worth −2% or more in only 3 cells.
    - So the view mainly tells the agent which cells have room, and only sometimes which decision.
    - On the observed part, 3L-Cache is often worse than S3-FIFO (worse in 8 of 14 cells, +38.9% on akamai's largest size); the best other policy is better in 9. That is the length effect again: 3L-Cache needs a long warm-up, and over whole traces it beats S3-FIFO by 2.7% on average. The view's 3L-Cache row therefore understates the real target.
- ~~view_knobs~~ (done).
- ~~Duplicate guard~~ (done).
- ~~Host-side knob tuning for structural candidates~~ (done for round 2): `run_experiment(tune={CONSTANT: [values]})`, up to 3 constants and 12 combinations. Every combination is simulated and the best one after the handover is recorded, for one experiment. Losing variants' logs are deleted. Evidence: prefix-tuned S3-FIFO beat the agents' own decisions in round-1 training (−0.61% vs +0.10% on average), so tuning constants on the observed part generalizes.
- **Correction (2026-11-16 11:50): the tuned reference is now off by default (`--tuned-reference` turns it on).** The round-1 analyst on GLM found that deep ghost queues break the memory limit on real traces. I measured it on tl-cf-4 with test_policy: the tuned reference used 2.6–3.0× S3-FIFO's memory, GHOST_RATIO 2.0 alone 2.6–3.0×, and 1.5 alone 2.3–2.5×. The synthetic test traces showed 1.66×. So the 'strong default' was not deployable; the decision gate would refuse it. Round 2 and the plain heldout-long arm had started with it; both were stopped about 40 minutes in (in xp3/aborted-tuned-memory) and restarted without it. It also qualifies the bar: prefix-tuned S3-FIFO ties 3L-Cache partly by using memory the agent is not allowed. A memory-fair prefix tuning (GHOST_RATIO ≤ 0.9) would be the fair floor; not computed yet.
- **Memory-fair prefix tuning** (started 2026-11-16 13:35 on the node, host only): `scripts/cachelab_memfair.py` reruns the baselines' prefix tuning with GHOST_RATIO capped at S3-FIFO's default 0.9 and writes `results/engineer/heldout-long-memfair` (uncapped values kept as `prefix_tuned_uncapped_*`). This is the fair knob-tuning floor for heldout-long.
- **First round-2 score (xp3, GLM, tl-akamai-4, unseen 80%):** agent −6.78% vs S3-FIFO; 3L-Cache −4.12%, prefix-tuned −7.37% (memory-fair here, ghost ≤ 0.9), 2Q −9.4% (best). Memory 1.5–1.64×. The seen part had promised −9.50%. The policy is S3-FIFO with knobs set per size band (promotion off in the middle band), i.e. close to what prefix tuning finds; its 'enter M only through a ghost hit' is 2Q's admission rule.
- **Prefix-tuned S3-FIFO as a reference (round 2, as first built):** when the engineer has a baselines report, the lab records `s3fifo-tuned`. It is S3-FIFO with the report's prefix-tuned knobs for each cache size, written as ordinary policy code, so the agent can view it, deploy it or build on it. The knobs come from the observed part only; the report's scores on the rest stay hidden. Why: on heldout-long the prefix-tuned baseline already ties 3L-Cache, while round-1 agents' own decisions averaged worse than it (+0.10% vs −0.61% in training). Starting from it, the agent's work goes into mechanisms beyond the knobs. Checks: per size it matches each configuration run alone; the widest configuration (GHOST_RATIO 2.0) uses at most 1.66× S3-FIFO's memory, under the 2× limit; all 47 long-trace reports give a valid policy.
- ~~Retrieval of similar cases by signature~~ (done for round 2): the engineer's case table is ordered by signature distance to the current workload, with the distance shown. Distance is the mean absolute difference of the signature's shares and changes vs S3-FIFO, size by size; lengths are not compared, so 1M-request cases rank against long workloads by shape. Among the 12 round-1 cases the closest pair is 0.06 apart and the median pair 0.22. The analyst still sees every case in case order.
- A second pass (epoch) over the training workloads.

### Round 3: beat 3L-Cache clearly (decided 2026-11-16 after the first round-2 scores)

Evidence (heldout-long, host only, unseen 80%, vs S3-FIFO):

| | mean | median | beats 3L-Cache |
|---|---|---|---|
| 3L-Cache | −2.73% | −1.24% | – |
| pick the published policy that won on the seen 20% (cold start) | −3.43% | −1.28% | 55/88 cells |
| best published per cell, in hindsight | −4.69% | −2.53% | – |
| memory-fair prefix-tuned S3-FIFO (ghost ≤ 0.9) | −2.57% | −0.95% | – |

- **Knob tuning cannot reach the bar** within the memory limit; choosing the right policy family can.
- The round-2 agent (tl-akamai-4) committed after 2 of 40 experiments, both knob grids. It never saw 2Q, the best policy there. And its seen-part promise (−9.5%) shrank to −6.8% on the unseen part.

Changes, local until round 2 and its test arm finish (then a new checkout, ~/VISTA-r3):

1. ~~**Python ports of published policies**~~ (done, local): `arc`, `lirs`, `twoq`, `wtinylfu`, `clock` and `sieve` in the sandbox.
   - Each can take over a live S3-FIFO cache (adopt).
   - They are offered as references, recorded on the observed part like any experiment, and importable.
   - They miss like libCacheSim's originals: CLOCK, SIEVE and ARC exactly; 2Q and LIRS within 0.2%; W-TinyLFU within 4% (its sketch differs).
   - Memory is 0.6–1.4× S3-FIFO on a 5k–40k cache. The sandbox measures it with tracemalloc; LIRS needed a lean non-resident list to fit.
2. ~~**Commit critic**~~ (done, local; session._challenge): refuse the first commit if a reference (published port or tuned S3-FIFO) beats the candidate on the observed part's later windows; ask for ≥ 3 structurally different designs before committing.
3. ~~**Memory margin**~~ (done, part of the critic): warn above 1.8× (a test run measured 1.92× on the observed part and 2.095× on the full trace, invalid).
4. **Online adaptation: evaluated (2026-11-16 21:10, host only, 88 cells).**
   - Setup: 11 libCacheSim policies, 3L-Cache excluded. Misses are counted in 40 windows. A selector picks from earlier windows only, at the picked policy's misses, with switching assumed free.

     | selector | mean | median | beats 3L-Cache |
     |---|---|---|---|
     | static pick from the seen 20% | −3.22% | −0.86% | 53/88 |
     | follow the cumulative leader | −3.95% | −1.07% | 60/88 |
     | best static in hindsight | −4.16% | −1.37% | 61/88 |
     | best per window in hindsight | −5.61% | −2.73% | 66/88 |
     | 3L-Cache | −2.73% | −1.24% | – |

   - **Online helps by correcting a wrong pick, not by tracking drift.** The cumulative leader switches 0 times in the median cell (47 of 88 never switch), and it lands close to the hindsight-best static choice. This matches LAH/S4-FIFO's finding that shifts needing a new prediction are uncommon.
   - **By cache size, the selector beats 3L-Cache's mean at every size:**

     | size | selector | 3L-Cache |
     |---|---|---|
     | smallest | −2.31% | −2.05% |
     | 2nd | −2.86% | −2.43% |
     | 3rd | −2.92% | −1.64% |
     | largest | −7.71% | −4.81% |

   - The median is the weak spot (−1.07% vs −1.24%). Small cells with little headroom decide it.
   - The per-window hindsight (−5.6%) shows large short-term variation that no follower captures.
   - Original note (evaluated above): a policy that keeps choosing among candidates while it runs, using sampled miniature simulations. The diagnostic replays per-window follow-the-leader over the zoo (/tmp/online_select.py, /mydata/onlinesel). Build it only if it beats static picking clearly.

5. **Virtual-queue designs (2026-11-16/17, host only, train-long, 100 cells).**
   - **The design space.** vqsim (scripts/vqsim/vqsim.c) is a C simulator for FIFO designs built from K stages:
     - hits only raise a counter; each stage has its own counting, promotion target/threshold, counter rule on promotion, demotion target and ghost feed;
     - D-FR delay and AGE idle thresholds per stage are options;
     - S3-FIFO and S4-FIFO's skip region are special cases.

     sandbox/vq.py is the same policy in Python, deployable with handover. The two agree miss for miss: cold start, handover, and prefix-only runs.
   - **Sweep r1.** 426 designs: S3-FIFO's 4 knobs on a joint grid (144), 32 S4-FIFO-like skip layouts, and 250 random multi-stage designs. Every ghost is ≤ 0.9, so all are within the memory limit. Each design is picked from windows 2–7 of the seen 20% and scored on the unseen 80% (cold start).

     | | mean | median | beats 3L-Cache |
     |---|---|---|---|
     | 3L-Cache | −2.64% | −0.02% | – |
     | best published per cell, hindsight | −5.12% | −0.70% | 71/100 |
     | pick among the S3 joint grid | −3.50% | −0.72% | 62/100 |
     | + S4-FIFO-like | −3.57% | −0.78% | 64/100 |
     | + random multi-stage (all 426) | −3.81% | −0.90% | 60/100 |
     | S3 grid, hindsight | −4.50% | −1.51% | 72/100 |
     | all 426, hindsight | −5.03% | −1.96% | 74/100 |

   - **More stages help (+0.3 pt mean).** The 426 FIFO designs alone match all published algorithms in hindsight on the mean and beat them on the median. The pick loses 1.2 pt to hindsight, which is the selection noise.
   - **Next: an agent-driven designer** (architect role, architect.py):
     - GLM architects write design families (stages plus value ranges);
     - each family is evaluated in minutes on these cells under the deployment protocol: pick from the seen part, take over S3-FIFO's live cache, score the unseen 80%;
     - all architects share a board (families, contributions to the pool, findings);
     - the final pool is tested once on heldout-long, on a separate board (scripts/cachelab_vqboard.py final).
     - Generation 1 is 5 architects with different directions, on board-g1 seeded with r1.

6. **Choosing well is the bottleneck; online re-choice fixes most of it (2026-11-17 01:40, train-long, r1 cold runs, 40 windows).**
   - **Static choice from the seen part.** Rules matter little: windows 2–7 −3.81% / −0.90%; mean rank over windows −3.97% / −0.89%; a margin before leaving S3-FIFO cuts losses (17 → 11 cells) but lowers the median. Small portfolios learned leaving one workload out are worse than the full pool (k = 8: −3.42% / −0.51%).
   - **Online leader.** In each unseen window, run the design with the fewest misses so far. This assumes shadow runs of every design from the start and free switching.

     | | mean | median | beats 3L-Cache | worse than S3-FIFO |
     |---|---|---|---|---|
     | 3L-Cache | −2.64% | −0.02% | – | – |
     | static pick | −3.81% | −0.90% | 60/100 | 17 |
     | leader, all 426, cumulative | **−4.59%** | **−1.50%** | **73/100** | 6 |
     | leader, S3 grid + S4-like, last 8 windows | −4.20% | −1.50% | 71/100 | 2 |
     | hindsight static | −5.03% | −1.96% | 74/100 | 0 |
     | hindsight per window | −6.60% | −4.08% | 79/100 | 0 |

     The cumulative leader switches 2 times in the median cell.
   - **Pre-registered for heldout-long** (fixed before running; runs once; /mydata/vq/heldout-sweep.sh → online-heldout.txt): (1) static pick, windows 2–7; (2) leader over all 426, cumulative; (3) leader over S3 grid + S4-like, last 8 windows.
   - **Pre-registered heldout-long result (2026-11-17 06:00, 88 cells, run once, unseen 80% vs S3-FIFO).** This is still idealized: shadow runs of every design, free switching.

     | | mean | median | beats 3L-Cache | worse than S3-FIFO |
     |---|---|---|---|---|
     | 3L-Cache | −2.73% | −1.24% | – | – |
     | best published per cell, hindsight | −4.69% | −2.53% | – | – |
     | (1) static pick, windows 2–7 | −2.69% | −1.56% | 55/88 | 12 |
     | **(2) leader over all 426, cumulative** | **−4.03%** | **−1.94%** | **61/88** | **2** |
     | (3) leader over S3 grid + S4-like, last 8 windows | −3.65% | −1.77% | 50/88 | 0 |

     - The online leader transfers from training (−4.59% / −1.50%) to held-out.
     - It beats 3L-Cache by 1.3 points of mean and 0.7 of median, and is worse than S3-FIFO in only 2 of 88 cells.
     - It comes close to the best published algorithm per cell in hindsight (−4.69%), with FIFO designs only.
   - **vqsim online mode (2026-11-17 05:40).**
     - The live cache switches design at window starts, following either a given schedule or sampled miniature simulations of K candidates (rate R; mini cache ≥ M objects).
     - A switch is a takeover between designs.
     - Tests: a one-design schedule equals that design's handover run.
     - Running on train-long (/mydata/vq/vq_online_real.py → online-real.txt): the idealized leader's schedule replayed live (switching cost), and mini-sims at 1% and 0.1% over all 426 designs and over S3 grid + S4-like.
   - **Realistic online, train-long (07:40).** Live cache, real switches, windows 8–39 vs S3-FIFO:

     | | mean | median | beats 3L-Cache |
     |---|---|---|---|
     | idealized leader (each design's cold-run windows) | −4.59% | −1.50% | 73/100 |
     | the same schedule replayed on a live cache | −3.94% | −1.08% | 62/100 |
     | mini-sims, all 426, 1% | −3.97% | −1.08% | 65/100 |
     | mini-sims, all 426, 0.1% | −3.84% | −0.98% | 63/100 |
     | mini-sims, S3 grid + S4-like, 1% | −3.88% | −1.15% | 65/100 |

     - Sampling costs nothing at 1%.
     - Switching costs about 0.65 pt: after a switch the live cache lacks the contents that design would have built itself.
     - Online still beats a static takeover (seed pool static under handover: −3.50% / −0.58%).
     - Next (running): switching hysteresis (margin, horizon), and starting the policy at window 4 instead of taking over at the split (/mydata/vq/vq_switch.py → switch-train.txt).
   - **Board picks.** The 1,802-design pool chooses 83 distinct designs across the 100 cells; 64 of them come from architects and 19 from seeds. These designs plus S3-FIFO (bp_designs.jsonl) are getting full 40-window runs on train-long and heldout-long, as the candidate set for online re-choice.
   - **Making it real:**
     - shadow candidates as miniature simulations (spatial sampling as in SHARDS / Waldspurger et al. ATC'17), about 1–2% of keys each. That costs about 0.01–0.02× cache metadata per candidate, so 16–64 candidates fit the 2× limit;
     - a switch is a takeover between two VQ designs, which vq.py's adopt already does;
     - measure the sampling noise and the switching cost with vqsim before building.
   - **Switching rule (2026-11-17 08:00, train-long, exact shadows replayed live; switch-train.txt).** Windows 8–39 vs S3-FIFO; 3L-Cache −2.64% / −0.02%.

     | | mean | median | beats 3L-Cache | worse than S3-FIFO |
     |---|---|---|---|---|
     | takeover at the split, static | −3.50% | −0.58% | 57/100 | 20 |
     | **takeover at the split, online, margin 0** | **−3.94%** | **−1.08%** | 62/100 | 14 |
     | the same, margin 1% / 3% | −3.85% / −3.73% | −0.84% / −0.72% | 63 / 61 | 15 / 17 |
     | the same, margin 1%, last 8 windows | −3.85% | −1.04% | 64/100 | 16 |
     | from window 4 (like 3L-Cache), static | −2.54% | −0.31% | 55/100 | 27 |
     | from window 4, online, margin 0 | −3.18% | −1.06% | 63/100 | 15 |

     - Hysteresis does not pay: the cost of a switch is smaller than the cost of staying on a worse design.
     - Taking over at the split beats running the policy from window 4: the early choice, made on 2 windows, is noisy, and S3-FIFO is a safe default meanwhile.
   - **Pre-registered realistic heldout test (launched 2026-11-17 07:50; vq_online_final.py → online-final-heldout.txt, then the same on train-long; run once).** The rule is takeover at the split, online, margin 0, 1% mini-sims (mini cache ≥ 1,000). Candidate sets: (A) the 426 seeds (primary); (B) the 84 board picks; (C) A + B. Static variants of A and B for reference.
   - **Memory.** Each mini-sim holds max(r·C, 1,000) objects. 426 candidates at 1% are 4.3× the cache's entries, over the 2× limit. So the deployable selector needs a small candidate set.
   - **Small portfolios (vq_portfolio.py, idealized leader, train-long).** Greedy from {S3-FIFO}, adding the design that lowers the mean most. 5-fold cross-validation by trace (portfolio chosen without the scored traces):

     | K | 2 | 4 | 8 | 16 | 32 | 64 | all 423 |
     |---|---|---|---|---|---|---|---|
     | mean | −0.40% | −3.07% | −3.96% | −4.38% | −4.52% | −4.59% | −4.59% |
     | median | 0.00% | −0.34% | −1.01% | −1.21% | −1.25% | −1.46% | −1.50% |

     16–32 designs keep nearly all of the gain. At 1% each, they cost 0.16–0.32 of the cache's entries for large caches; the 1,000-object floor dominates for small ones.
   - **Realistic, memory-bounded selector (train-long, portfolio-r1.json chosen on train-long, so in-sample; vq_portfolio_real.py → portfolio-real-train.txt, portfolio-mem-train.txt).** Live switching, mini-sims, margin 0. Selector entries = K · max(r·C, floor), median over cells:

     | selector | mean | median | beats 3L-Cache | worse than S3-FIFO | selector entries / C |
     |---|---|---|---|---|---|
     | 3L-Cache | −2.64% | −0.02% | – | – | – |
     | K = 8, 3%, no floor | −2.60% | −0.23% | 59/100 | 17 | 0.24 |
     | K = 16, 1.5%, no floor | −2.70% | −0.50% | 59/100 | 14 | 0.24 |
     | K = 16, 1.5%, floor 100 | −3.98% | −1.03% | 65/100 | 13 | 0.24 |
     | K = 16, 1%, floor 200 | −3.96% | −1.07% | 63/100 | 11 | 0.16 |
     | **K = 32, 0.5%, floor 200** | **−4.14%** | **−1.09%** | **65/100** | 11 | **0.31** |
     | K = 32, 1%, floor 1,000 | −4.12% | −1.12% | 65/100 | 7 | 1.55 |
     | K = 64, 0.5%, floor 500 | −4.19% | −1.10% | 67/100 | 9 | 1.55 |
     | all 426, 1%, floor 1,000 (earlier) | −3.97% | −1.08% | 65/100 | – | ≈ 4+ |

     - A mini-sim needs a floor of 100–200 objects. Below that, tiny miniature caches mislead the choice; the 1,000-object floor buys nothing over 200.
     - The memory-feasible selector is K = 32 at 0.5% with a 200-object floor. It costs 0.31 of the cache's entries in the median cell, at most 0.32·C for caches of 40k objects or more, and a constant 6,400 entries below that.
   - **Pre-registered for heldout-long (fixed 2026-11-17 09:05 before any heldout run of it; run once; vq_portfolio_real.py heldout-long portfolio-r1.json ph1 → portfolio-heldout.txt).** Takeover at the split, online, margin 0, over the first K designs of portfolio-r1.json (greedy on train-long r1 runs):
     - primary: K = 32, rate 0.5%, floor 200;
     - secondary: K = 16, rate 1%, floor 200.
   - **Pre-registered heldout-long result (2026-11-17 08:46, 88 cells, run once; portfolio-heldout.txt).** Windows 8–39 vs S3-FIFO:

     | | mean | median | beats 3L-Cache | worse than S3-FIFO | selector entries / C |
     |---|---|---|---|---|---|
     | 3L-Cache | −2.73% | −1.24% | – | 27 | – |
     | **K = 32, 0.5%, floor 200 (primary)** | **−3.18%** | **−1.12%** | **48/88** | 11 | 0.16 |
     | K = 16, 1%, floor 200 | −3.07% | −0.85% | 50/88 | 10 | 0.16 |
     | (idealized: all 426, shadows, free switching) | −4.03% | −1.94% | 61/88 | 2 | – |

     - **Not a clear win.** It is better than 3L-Cache on the mean by 0.45 pt, with fewer cells worse than S3-FIFO (11 vs 27), but the median is slightly worse and it wins 48 of 88 cells.
     - Training (in-sample portfolio) promised −4.14%. The drop to −3.18% combines three costs: the portfolio fitted to the training traces (cross-validation: about 0.3 pt), switching on a live cache (about 0.65 pt on train), and sampling.
   - **Pre-registered realistic heldout, candidate sets (vq_online_final.py, fixed 07:50, run once; online-final-heldout.txt; 1% mini-sims, floor 1,000, so memory is not bounded).** Windows 8–39 vs S3-FIFO, 88 cells:

     | candidates | mean | median | beats 3L-Cache | worse than S3-FIFO |
     |---|---|---|---|---|
     | 3L-Cache | −2.73% | −1.24% | – | 27 |
     | (A) 426 seeds, online | −3.31% | −1.29% | 51/88 | 10 |
     | (A) static | −2.23% | −1.10% | 51/88 | 13 |
     | **(B) 84 board-g1 picks, online** | **−3.72%** | **−1.81%** | **53/88** | **7** |
     | (B) static | −2.87% | −1.83% | 51/88 | 13 |
     | (C) A + B, online | −3.59% | −1.82% | 52/88 | 8 |

     - **The architects' designs transfer to held-out workloads.** The designs the board-g1 pool chooses on train (64 of 84 by agents) make a better candidate set than the 426 seeds: +0.4 pt mean and +0.5 pt median. They beat 3L-Cache by 1.0 pt mean and 0.6 pt median, and are worse than S3-FIFO in 7 cells vs 27 for 3L-Cache.
     - Online beats static for every set (+0.85 to +1.1 pt mean).
     - Next: B's advantage within the memory limit. The 32-design greedy portfolio from r1 + bp on train is in-sample −5.08% / −1.97% (idealized); in 5-fold CV −4.37% / −1.50%, optimistic because the bp designs were picked on all train traces. 16 of the 32 are board picks.
   - **Realistic, memory-bounded, r1 + bp portfolio (train-long; portfolio-r1bp-real-train.txt, portfolio-r1bp-cvreal-train.txt).** 32 designs at 0.5% with a 200-object floor (0.31·C selector entries, median):
     - in-sample −4.51% / −1.46%, beats 3L-Cache 67/100, worse than S3-FIFO 9;
     - 5-fold CV −4.20% / −1.40%, 72/100.

     The seeds-only portfolio gets −4.14% / −1.09% and −3.96% / −1.05%. Small caches are now about a tie with 3L-Cache (−3.63% vs −3.72%).
   - **Pre-registered for heldout-long (fixed 2026-11-17 ~10:45 before any heldout run of it; run once; portfolio-r1bp.prereg-1045.json; vq_portfolio_real.py heldout-long ... phb → portfolio-r1bp-heldout.txt).** Takeover at the split, online, margin 0, over the first K designs of the r1 + bp greedy portfolio:
     - primary: K = 32, rate 0.5%, floor 200;
     - secondary: K = 16, rate 1%, floor 200.
   - **Pre-registered heldout-long result (2026-11-17 11:00, 88 cells, run once; portfolio-r1bp-heldout.txt).** The first attempt crashed in reporting (heldout meta.json lacks `objects`) before printing anything; the rerun used the cached runs. Windows 8–39 vs S3-FIFO:

     | | mean | median | beats 3L-Cache | worse than S3-FIFO | selector entries / C |
     |---|---|---|---|---|---|
     | 3L-Cache | −2.73% | −1.24% | – | 27 | – |
     | **K = 32, 0.5%, floor 200 (primary)** | **−3.42%** | **−1.79%** | 47/88 | **9** | 0.16 |
     | K = 16, 1%, floor 200 | −3.30% | −1.64% | 48/88 | 10 | 0.16 |
     | seeds-only portfolio, K = 32 (earlier) | −3.18% | −1.12% | 48/88 | 11 | 0.16 |

     By cache size (K = 32 vs 3L-Cache):
     - <0.5% (17 cells): −2.38% vs −1.88%;
     - 0.5–2% (25): −3.51% vs −2.02%;
     - 2–8% (32): −2.67% vs −1.59%;
     - ≥8% (14): −6.23% vs **−7.63%**.

     - Within the memory limit, the architects' designs add +0.24 pt mean and +0.67 pt median over the seeds-only portfolio.
     - The selector beats 3L-Cache by 0.7 pt mean and 0.55 pt median, and is worse than S3-FIFO in 9 cells vs 27.
     - But it wins only 47/88 cells. 3L-Cache is stronger at the largest caches on heldout (the opposite of train), so the advantage is clear on average but not cell by cell.
   - **Where it loses (train-long only, realistic, portfolio chosen by 5-fold CV by trace; portfolio-cv-real-train.txt).** Overall −3.96% / −1.05%, beats 3L-Cache 64/100. By cache size, selector vs 3L-Cache:
     - <0.5% of objects (22 cells): −2.65% vs **−3.72%**;
     - 0.5–2% (31): −3.62% vs −2.90%;
     - 2–8% (34): −4.37% vs −2.49%;
     - ≥8% (13): −5.92% vs −0.57%.

     Small caches are the gap; even the idealized selector loses there (−3.28%). The winners at small caches are 3L-Cache, LHD and LIRS, all of which use reuse history.
   - **Ghost depth is not the answer (host sweep, 30 designs, gd_designs.jsonl; memory ignored).** The designs are S3-shaped, with S size {.05, .1, .2} × S promote {1, 2} × ghost {0.9, 2, 4, 8, 16} C, and ghost hits go to M. The best fixed design per size class (<0.5 / 0.5–2 / 2–8 / ≥8%):
     - G = 0.9: −0.16 / 0.00 / 0.00 / −1.65%;
     - G = 2: +1.25 / +0.67 / +1.04 / +1.14%;
     - G = 16: +3.07 / +4.00 / +2.52 / +1.53%.

     It gets monotonically worse with depth at every size: a ghost hit over a long history admits long-reuse-distance objects into M. Small caches need a better use of reuse distance, not more history.
   - **board-g2: the online objective (2026-11-17 09:40).** It lives in architect.py as `Board(objective="online")`: full 40-window runs, the pool scored as a greedy 32-design portfolio, idealized online choice. It was seeded from existing full runs (426 r1 seeds and 83 board-g1 picks; `cachelab_vqboard.py import-runs`) and has board-g1's findings copied. Pool: −5.08% / −1.97%, beats 3L-Cache 78/100 (in-sample, idealized); the board-g1 picks add −0.30 pt over the seeds.
   - **Generation 4 (09:45; ~/VISTA-r4, /mydata/vq/g4.sh).** Four online architects with 8 evaluations each: small caches, specialists, ordinary cells, cells that lose.
     - **Result (done by 13:00, all four submitted, no errors).** 32 families; the portfolio moved from −5.08% to −5.11% (in-sample, idealized), and 5 families put a design into it. The online board is saturated in this design space too.
   - **Takeover mapping (train-long, the r1 + bp portfolio, K = 32, 0.5%, floor 200; vqsim online `takeover=`; takeover-train.txt).**
     - By stage (v1): −4.51% / −1.46%.
     - By counter (hit since the last move → main): −4.12% / −1.39%.
     - All to main: −4.39% / −1.44%.
     - Without later switches: −4.15%, −3.87% and −4.06% respectively.

     The stage mapping stays.
8. **Version 1 frozen (2026-11-17 13:30): docs/cachelab-v1.md.**
   - `OnlineV1` in sandbox/vq.py.
   - The portfolio, seeds, board picks and logs are in scripts/vqsim/v1/; the study scripts are in scripts/vqsim/study/.
   - Pre-registered heldout: −3.42% / −1.79% vs 3L-Cache −2.73% / −1.24%. It wins 12/22 traces, so it does not meet the open-source bar (15/22).
   - **Deployable selector: `OnlineVQ` in sandbox/vq.py (2026-11-17).**
     - Knobs: CANDIDATES (vqsim config strings), PERIOD (requests per window), RATE, MIN_MINI, MARGIN, FROM_WINDOW, START.
     - Mini-sims are VQ instances on keys with mix64(key) ≤ rate·2⁶⁴. They see the prefix through `observe`. The first choice happens at `adopt`, and later switches use the same takeover as vqsim.
     - Tested: equals vqsim's online mode exactly (misses before and after the split, number of switches) with mini-sims, with hysteresis, and without a handover (start=4). It also runs in the sandbox with no instrumentation errors.
     - Also fixed: VQ.adopt kept too few ghost entries when the new ghost was larger than S3-FIFO's but less than twice as large.
7. **Architect generation 1 (board-g1, 5 GLM architects, started 2026-11-17 00:00).** Its counterfactual findings, all of which match the literature's principles:
   - a ghost hit must re-enter main (re-entering S loops: +5 to +25%);
   - counters deeper than 2 bits buy nothing;
   - AGE as refused reinsertion is a clean negative;
   - D-FR in main hurts (+4.4%), and in S it is neutral to harmful;
   - two-step probation does not beat a single S;
   - families with "catastrophic" arms poison the choice.

   No family has improved the pool yet; the pool's limit is choosing from the seen part (item 6).
   - **Incident, 2026-11-17 00:23–03:30.** A family whose stages demoted into themselves (or into each other) trapped objects. Each eviction then ran to vqsim's guard (4C steps), so 20 vqsim processes hung for 3 h and blocked every architect.
   - **Fix.**
     - architect.py, vqsim.c and vq.py all reject designs with a cycle of moves that need no hit and keep the counter (a demotion, a pass with promote 0, a keep-promotion); every other move lowers the counter, so scans end;
     - vqsim batches get a timeout;
     - a failed simulation returns an error reply, where before it stopped the session.
   - **Relaunch.** The five architects restarted at 03:40 with 8 evaluations each; the board keeps their earlier families. The stuck runs are in runs/arch/g1/stuck-0327.
   - **Generation 1 result (40 families).** The pool went from −3.57% / −0.61% to −3.79% / −0.94%; the median gain is the useful part.
     - The runtime ended each session when its evaluation budget ran out, before submit_families. Now the session enforces the budget itself, and the runtime meter has slack.
   - **Generation 2 (05:30).** Five architects with 10 evaluations each on board-g1, each owning a group of cells: small caches, large caches, cells where the pool loses to S3-FIFO, the cells with the largest 3L-Cache lead, and the median cell.
     - Board at 07:50: 104 families, 1,826 designs. Pool (static, handover): −4.03% / −1.35%, beats 3L-Cache on 66/100, worse than S3-FIFO on 16 (seeds alone: −3.50% / −0.58%).
   - **Generation 3 (07:28).** Three architects with 10 evaluations each:
     - "winners" distills the designs the pool chooses most;
     - "losers" targets cells where the pool still loses;
     - "shift" targets cells where the observed part misleads the choice.

### Experiments (FreeInference, kimi-k3 for every role, at most 8 concurrent sessions)

**Model switch (2026-11-16): xp3 runs on GLM-5.3 Flash.**

- kimi-k3 failed from 2026-11-15 12:10 (503, then 429, at every startup).
- Checked over streaming, as Claude Code calls it: glm-5.3-flash is the real GLM, with tool calls and images ($0.3/$1.0 per M tokens, about 9× cheaper than kimi-k3). deepseek-v4-flash has no image input, so it can't run the visual arm.
- xp2 stays kimi-k3 only; its round-1 tests can finish if kimi-k3 returns.
- xp3 copies xp2's merged round-1 library. It runs the round-1 analyst on GLM, then round 2 on train-long (6 lineages, 1 workload each, budget 40), from ~/VISTA-r2 with run-xp3.sh.
- Model is now a factor: xp3 results are compared within xp3, against the same baselines.

- **xp2 round 1:** 12 training workloads, then merge, analyst, test (experience vs plain) on the 24 held-out workloads.
- **Research-question arms at equal budget:**
  - numbers only;
  - numbers + dashboard;
  - numbers + diagnostic views;
  - both.
- **Hypothesis:** the anchor image carries most of the visual gain.
