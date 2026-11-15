# CacheLab: improving S3-FIFO with a visual research agent

CacheLab turns cache-policy research into a VISTA task. The agent starts from
S3-FIFO, looks at how it behaves on a suite of workloads through linked views,
forms a hypothesis about a mechanism, predicts what a change will do, runs it,
and checks the prediction against the same views. The research question is
whether visual replay and retrievable visual memory let an agent find better
and better-explained improvements than code plus numeric feedback, at the
same experiment budget.

## Roles

| Role | Sees | Decides | Scored on | Flag |
| --- | --- | --- | --- | --- |
| designer | the development traces, in full | one general policy | held-out traces, relative to S3-FIFO | `--role designer` (default) |
| engineer | the first 20% of one workload, as trace `target` | one policy for that workload | the remaining 80%, relative to S3-FIFO, and against S3-FIFO tuned on the prefix and libCacheSim's algorithms | `--role engineer` |

The engineer role follows S4-FIFO's protocol (observe the first 20% of a
trace, then fix the configuration for the rest), with the agent in place of
S4-FIFO's learned control plane. Its decision runs over the whole workload
from a cold cache; only misses after the split count, and every comparison
algorithm is measured the same way. It never learns where the workload comes
from: the prefix and the full trace are both named `target`, the full trace
sits in a directory no experiment mounts, cache sizes derive from the prefix
alone, keys are random ids, and every text the agent reads passes through a
redaction step (a test checks this through every tool).

Deployment is a **handover**, as in production. S3-FIFO serves the prefix;
at the split the decided policy takes over its live cache. The simulator calls
the policy's `adopt(state)` with S3-FIFO's state at that moment
(`policy_api.Handover`: resident objects of S and M from tail to head with
their 2-bit counters, the ghost keys, the knobs) and the policy must place
every resident object in one of its queues (checked; an object it drops is an
error). Before the switch the policy may `observe(key, now, hit)` every request
S3-FIFO served, within the same memory bound. The inherited state is the hard
part, and it is left to the agent: with a large cache it persists long after
the switch (turning a cache over takes about C / miss ratio requests, which can
exceed the whole scored part), and every miss after the switch counts. The
default `adopt` re-inserts objects one by one; S3-FIFO's keeps the queues as
they are and lets new knobs act from then on, so S3-FIFO taking over from
S3-FIFO changes nothing (tested), and knob changes play out through evictions.
The agent's experiments on the prefix hand over the same way, after 20% of it
by default (`run_experiment(handover_at=...)`, 0 for a cold start), and the
queue and diff views mark the handover. Engineer cache sizes are 1%, 3%, 10%
and 30% of the prefix's distinct objects, so the large-cache case is in every
task.

Comparison set (`zoo.py`, all object miss ratio with unit sizes): LRU, FIFO,
CLOCK, SIEVE, ARC, LIRS, 2Q, CLOCK-Pro, MQ, S3-FIFO, W-TinyLFU, LeCaR,
Cacheus, LHD, 3L-Cache, and Belady as the offline bound; plus S3-FIFO with
the knob setting (per cache size, from the 287-point grid) that does best on
the prefix, switched on at the split: tuning by search, the job S4-FIFO's model
does by prediction. The published algorithms run from a cold cache over the
whole trace (libCacheSim cannot take over a cache), so they pay no handover:
an optimistic reference.

## Experience: train on many workloads, tune unseen ones

An engineer can carry experience from earlier workloads, the way a human
engineer does. Training and testing are separate phases:

1. **Training** (`--train --experience LIB`). On each training workload the
   agent does the engineer job, then commits its decision (`commit_decision`
   replaces `submit_policy`). It immediately sees the outcome on the rest of
   that workload: its miss ratio per cache size next to S3-FIFO, S3-FIFO tuned
   on the prefix (with the chosen knobs), every published algorithm and
   Belady (fast simulation; the comparison numbers come from the
   `engineer-baselines` report passed as `--baselines`). It then calls
   `record_experience`: a one-line decision, the lesson (which prefix signals
   predicted the outcome, what it would decide now), and the complete revised
   playbook. Each case is stored with a host-computed **workload signature**
   (per cache size: S3-FIFO's miss ratio, reuse within 0.1C and C, the
   one-hit-wonder share, where S and ghost hits land, LRU and MQ against
   S3-FIFO), so a later engineer can match a new workload against the cases on
   equal terms. Cases never name their workload.
2. **Lineages.** Several training sessions run in parallel, each working
   through its own share of the training workloads in sequence, so each
   library grows case by case. `merge-experience` combines them: all cases,
   renumbered with their origin, and the playbooks as sections of one.
Budgets are deliberately unequal: training is offline, so a training session
gets many experiments (40 by default in the driver); a test engineer decides
from one look at the first 20% (signature and free views) plus one or a few
experiments, the way a deployed control plane would. Training sessions are
told this, so their playbooks map what is observable for free to a decision.

3. **Testing** (`--experience LIB`, no `--train`). The agent gets the playbook
   and the case table in its first message and `read_case` for the details,
   read only, and does the engineer job on a workload none of the training
   sessions saw; no feedback, verified scoring as usual. The control arm is
   the same agent without a library.

After a training run the host also scores every experiment the agent
completed (its rejected candidates and the reference policies too) on the rest
of the workload, and stores these as `counterfactuals` in `result.json`
(signature, policy source, prefix and suffix outcome). The agent never sees
them; they are labels over an open-ended decision space for a learned decision
model later, which a fixed knob grid could not provide.

Every engineer (with or without experience) sees its own workload's signature,
so the experience arm differs from the control only by the library.

A library directory holds `playbook.md`, `playbook-history/` (every revision),
and `cases/case-NNN.json` (signature, decision, reasoning, feedback, outcome,
lesson, experiment log) with `case-NNN.py` (the decided policy).

## Literature: references and worked examples (`--shots`)

With `--shots` the lab also records published policies as experiments the
agent can view, diff, read and import, and the first prompt carries three
worked examples that replay a published analysis on the agent's own data,
with the exact numbers its tools return:

| Paper | Analysis (tool) | Mechanism it led to | Reference |
| --- | --- | --- | --- |
| S3-FIFO (SOSP 2023) | one-hit-wonder share of cache-sized windows (`view_workload`) | quick demotion through a small FIFO | `s3fifo` |
| MQ (USENIX ATC 2001) | reuse distance against the cache, access frequency (`view_workload`) | frequency levels with lifetime decay, history of evicted frequencies | `mq` |
| S4-FIFO (OSDI 2026) | where hits land in S, M and G (`view_positions`) | skip window, ghost threshold, knobs chosen per workload | `s4fifo` |

`mq.py` matches libCacheSim's MQ miss for miss; `s4fifo.py` is S4FIFO.c's
heuristic with auto-tune off (identical to S3-FIFO at default knobs). LRU and
FIFO are also available.

## Arms

| Arm | What the agent gets | Command |
| --- | --- | --- |
| `vista` | every view as an image plus its exact numbers; `inspect`, `read_pixels`, `history`, GUIDE/WORKING | `--arm vista` |
| `numeric` | the same tools and numbers, no images, nothing to inspect | `--arm numeric` |
| parameter search | no agent: random configurations of S3-FIFO's knobs | `search --budget N` |

All arms share one lab, one budget meter (`experiments`), and one scorer.

## The loop

```
overview (where S3-FIFO loses) -> queue replay (what its queues do there)
   -> object timelines (what happened to the objects carrying the misses)
   -> raw records -> hypothesis + prediction -> run_experiment
   -> diff / same views at the same places -> GUIDE.md (supported / refuted)
```

Tools (`src/vista/benchmarks/cachelab/profiles.py`):

- `run_experiment(name, hypothesis, prediction, code)` simulates a complete
  policy module on every dev trace and cache size (one experiment). The
  prediction has two parts: the expected mean change, and the mechanism
  signature expected in the views. The result reports the prediction error.
- `view_overview` shows a capacity × time miss-ratio heatmap per trace, or the
  change against another experiment (blue fewer misses, red more).
- `view_queues` replays S/M/G occupancy, misses by cause (cold, or returning
  after eviction from S or M), evictions (and how many had no hit), transfers,
  ghost hits and reinsertions, plus eviction quality: how soon evicted objects
  return, in cache sizes.
- `view_objects` gives per-object timelines (requests, residency by queue,
  ghost membership, evictions), for given keys or the objects carrying a
  window's misses / regressions.
- `view_diff` compares two experiments request by request.
- `view_workload` characterizes a trace independently of any policy: exact
  LRU stack (reuse) distances against the cache sizes, the access-frequency
  distribution, and the one-hit-wonder share of windows holding C or 0.1C
  distinct objects.
- `view_positions` shows where an experiment's hits land in each queue and
  ghost: objects that entered the queue after the hit object did, in
  multiples of the queue's mean size (in a FIFO, 0 is the head, ~1 the tail).
- `query_events` returns raw records; `experiment_info` lists experiments and
  their code; `test_policy` checks code without spending an experiment;
  `submit_policy` ends the task. Experience runs add `read_case`, and
  training runs replace `submit_policy` with `commit_decision` and
  `record_experience`.

Every view returns the numbers it draws, so a visual finding can be checked
exactly. Images are 1024 px wide; VISTA archives each one under an event
number that `inspect` and `read_pixels` can return to.

## Policy API

`sandbox/policy_api.py`. The simulator owns residency (it decides hits and
enforces capacity); a policy only answers `on_hit`, `on_miss` (admit?),
`evict` and `insert`, and reports internal moves through `ctx` so the views
can replay any variant, not just S3-FIFO. Capacity counts objects. The
reference `sandbox/s3fifo.py` matches libCacheSim's `S3FIFO` miss for miss,
including its cold-start rule (tested against `libcachesim` 0.3.5).

## Integrity

Agent-written code is untrusted, and the agent is optimizing a number, so:

- **Fast mode** (dev experiments) runs the simulator and policy in a
  `python:3.12-slim` container with no network, a read-only root, dropped
  capabilities and only the dev traces mounted. A static check rejects
  imports outside a small allowlist, dunder access, and private attributes
  of other objects.
- **Verified mode** (scoring) keeps the simulator on the host and serves the
  policy from a separate container, one request at a time, so it cannot read
  the trace ahead or touch the records. Dev results are re-measured this way
  and any mismatch with fast mode invalidates the submission (tampering or
  nondeterminism). The two processes talk over a Unix socket; hits are
  batched, and a miss (`on_miss`, `evict` if full, `insert`) is one round
  trip. Verified cells get their own time limit (an hour by default), and a
  "time limit exceeded" result is never cached, since it depends on machine
  load as much as on the policy.
- **Bounded metadata.** Verified runs measure the policy's Python memory;
  more than 2× S3-FIFO's at the same cache size is invalid. Otherwise
  remembering every key ever seen would count as an "improvement".
- **Keys** are random IDs below 2^40: exact in JSON for JavaScript clients,
  and they leak nothing (sequential IDs would reveal "never seen before").
- **Held-out traces** are generated only at final evaluation: new seeds and
  parameters for every dev family, plus two families the agent never sees.

## Workloads

Synthetic families with a known mechanism each (`traces.py`): Zipf with scans
of one-time keys, medium-distance burst reuse, popularity shifts, Zipf mixed
with a loop, plain Zipf, and two interleaved tenants. The agent sees only
neutral names and footprint statistics.

Real traces in libCacheSim's oracleGeneral format (optionally zstd
compressed) are sampled into suite files, which pin each trace's path and
prefix length:

```bash
R=/path/to/oracleReuse   # one directory per dataset
vista-cachelab build-suite --data-root $R --datasets msr,cphy,alibabaBlock,tencentBlock,cf,akamai \
  --per-dataset 1 --split dev --requests 1000000 --max-mb 200 --out suites/dev-real.json
vista-cachelab build-suite --data-root $R --datasets msr,cphy,alibabaBlock,tencentBlock,cf,akamai \
  --per-dataset 3 --split heldout --exclude-suite suites/dev-real.json --out suites/ho-seen.json
vista-cachelab build-suite --data-root $R --datasets fiu,metaStorage,ARC \
  --per-dataset 2 --split heldout --out suites/ho-unseen.json
```

Dev traces get neutral names; held-out traces are named `ho-<dataset>-<k>`,
which only the final report shows. Pass suites with `--dev-suite` and
`--heldout-suite` (combine files with `traces.save_suite`).

## Running

```bash
.venv/bin/python -m pip install -e '.[cachelab]'
docker pull python:3.12-slim

# Simulate a policy file and write its views (people use the same lab)
.venv/bin/vista-cachelab try my_policy.py --view queues:dev-2:4842

# Verified held-out score of a policy file
.venv/bin/vista-cachelab evaluate my_policy.py --out runs/eval.json

# Parameter-search arm
.venv/bin/vista-cachelab search --budget 12 --seed 0 --out runs/search.json

# Agent arms (same backends as the other VISTA benchmarks)
.venv/bin/vista --profile cachelab --backend claude run --arm vista \
  --model opus --effort xhigh --max-experiments 12 --output runs/cachelab-vista
.venv/bin/vista --profile cachelab --backend claude run --arm numeric \
  --model opus --effort xhigh --max-experiments 12 --output runs/cachelab-numeric
```

Reference arms on a large machine (LRU and FIFO, the search arm over several
seeds, and the full parameter grid as an upper bound for knob tuning), then a
per-dataset table of the reports:

```bash
python scripts/cachelab_baselines.py --lab-dir runs/lab --out results/real \
  --dev-suite suites/dev-mixed.json --heldout-suite suites/heldout-real.json \
  --workers 36 --seeds 5 --grid
python scripts/cachelab_summarize.py results/real
```

Batch arms keep only each cell's summary (`prune=True`); full logs take
about 14 MB per 1M-request cell.

Engineer role, one workload (from a suite file, or any oracleGeneral file):

```bash
.venv/bin/vista --profile cachelab --backend codex run --role engineer --shots --arm vista \
  --target-suite suites/heldout-real.json --target ho-msr-1 \
  --model gpt-5.6-sol --effort high --max-experiments 8 \
  --lab-dir runs/eng-lab-1 --output runs/eng-run-1
.venv/bin/vista --profile cachelab --backend codex run --role engineer --arm vista \
  --target-trace /path/to/trace.oracleGeneral.zst --target-requests 1000000 ...

# No agent: S3-FIFO, prefix-tuned S3-FIFO and libCacheSim's algorithms on every target
.venv/bin/vista-cachelab engineer-baselines --target-suite suites/heldout-real.json \
  --sandbox local --workers 4 --lab-dir runs/eng-baselines --out results/engineer --shard 0/8
```

Experience learning, end to end (`scripts/cachelab_experience.py`; run and lab
directories get neutral names, because the agent can see its working
directory, and `ROOT/targets.json` maps them back):

```bash
# training workloads: real traces from the seen datasets, none in any other suite
.venv/bin/vista-cachelab build-suite --data-root /path/to/oracleReuse \
  --datasets msr,cphy,alibabaBlock,tencentBlock,cf,akamai --per-dataset 2 \
  --exclude-suite suites/dev-real.json --exclude-suite suites/heldout-real.json --out suites/train-real.json
.venv/bin/vista-cachelab engineer-baselines --target-suite suites/train-real.json \
  --sandbox local --workers 4 --lab-dir runs/baselines/train --out results/engineer/train --shard 0/4
AGENT="--model gpt-5.6-sol --effort high --binary $CODEX --arm vista --workers 8"
python scripts/cachelab_experience.py train --root runs/xp --suite suites/train-real.json \
  --baselines results/engineer/train --lineages 3 --budget 40 -- $AGENT
python scripts/cachelab_experience.py merge --root runs/xp
python scripts/cachelab_experience.py test --root runs/xp --suite suites/heldout-real.json \
  --baselines results/engineer/test --arms experience:1,plain:1,experience:3,plain:3 --parallel 4 -- $AGENT
python scripts/cachelab_experience.py summarize --root runs/xp
```

Finished runs are reused when a command is repeated, a crashed run is moved
aside and rerun, and a finished run's simulation arrays are deleted (its
summaries stay).

`result.json` records the submission (mechanism, evidence, limitations), every
experiment with its predicted and observed change, tool usage counts, and the
verified dev and held-out evaluation. A run that ends without submitting is
scored `null`; its best dev experiment is evaluated as `fallback_score` for
analysis only.

## What to measure

- held-out score and worst-cell regression, per arm, at equal budget;
- prediction error per experiment, and how it changes over a run;
- experiments that did not improve on the best so far (wasted budget);
- whether the submitted mechanism names the family-specific cause (the
  synthetic families make this checkable by hand or by a judge);
- which views each arm used, and in what order.
