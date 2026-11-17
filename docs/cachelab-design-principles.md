# Cache-design principles for CacheLab agents

Distilled on 2026-11-16 from Juncheng Yang and colleagues' papers:
- the lazy-promotion/quick-demotion paper (HotOS'23);
- S3-FIFO (SOSP'23);
- SIEVE (NSDI'24);
- the lazy-promotion follow-up (VLDB'26);
- GL-Cache (FAST'23);
- the Twitter cache study (OSDI'20);
- "When Fancy Eviction Fails" (2026);
- LAH/S4-FIFO (OSDI'26).

The section numbers point into those papers. Part 1 is meant for the agents' prompts; part 2 is our backlog of host-side diagnostics.

## 1. Principles (agent-facing)

1. **Quick demotion is the main lever.** A cache of size C sees a short window, and in such windows most objects are requested once:
   - 72% (median) in windows of 10% of the objects, against 26% over the whole trace (S3-FIFO §3);
   - so most new objects should leave soon after insertion;
   - measure it: the share of objects that leave S without a hit, and the one-hit share in windows the size of the cache.
2. **Judge a change to S or the ghost by demotion speed and precision.**
   - speed: how fast S evicts relative to LRU;
   - precision: the share of S evictions whose next request is far away (more than about C/miss-ratio requests);
   - at equal speed, the more precise variant has the lower miss ratio (S3-FIFO §6.1).
3. **Promote lazily, at eviction time, not on a hit.** Promoting at eviction is never worse (VLDB'26 §4). An object that keeps its place and is examined when it reaches the tail is the cheap and robust default (FIFO-reinsertion, SIEVE).
4. **Do not reward clustered hits.**
   - About 76% of LRU promotions closely follow a previous request (VLDB'26).
   - Count a hit only if enough time has passed since the last counted one (Delayed FIFO-Reinsertion, below), or only outside a probation region (S4-FIFO's skip region).
5. **Let recency override frequency.** Large counters trap objects that were hot once; 1–2 bits work best. An object idle for longer than about C/miss-ratio requests should not be reinserted (Age-Guided Eviction, below).
6. **The decision is a yes/no question: requested again before it would be evicted?** The horizon is about C/miss-ratio requests. Ranking cached objects finely matters little; with a perfect early-eviction oracle the fallback order barely changes the result (VLDB'26 §4.1).
7. **The ghost is the second chance.** Its horizon decides whether objects whose second request comes after they left S are saved (S3-FIFO §5). A compact ghost (fingerprint + insertion stamp) makes a longer horizon cheap.
8. **Do not reject first-time objects outright** (no Bloom-filter admission). Let a quick second request hit while the object is still in S (S3-FIFO §4).
9. **Adapt slowly, if at all.**
   - Adaptive S sizing overreacts and helped about 2% of traces (S3-FIFO-d, §6.3). ARC improved when its adaptation was capped and slowed (HotOS'23).
   - Choosing the right setting per workload matters more than adapting on every miss (LAH/S4-FIFO).
   - CacheLab's own check: on our long traces, re-choosing the best policy while running helped mainly by correcting a wrong first choice.
10. **Compose simple primitives.**
    - Examples: a probationary FIFO in front of any main policy, a SIEVE hand as the main queue, more FIFO stages ("virtual queues") with their own counting and promotion rules.
    - Prefer these to bespoke per-object logic (HotOS'23 "like LEGO"; SIEVE §6).
11. **Before building a mechanism, run the oracle that isolates it** (view_headroom in CacheLab), and build only if the gap is material ("When Fancy Eviction Fails" §6; GL-Cache).
12. **Tune on one part and check on another.** Within the observed 20%, choose on earlier windows and confirm on later ones. Never tune on what is scored (S3-FIFO footnote on validation).
13. **Report distributions, not just the mean.** Show the worst cells and how many are worse than S3-FIFO:
    - TinyLFU was the closest rival on the mean but worse than FIFO on about 20% of traces (S3-FIFO §5);
    - every state-of-the-art algorithm lost to FIFO on its worst trace (LAH).

## 2. Mechanisms with exact semantics

- **Delayed FIFO-Reinsertion (D-FR, VLDB'26):**
  - on a hit, count it only if `now - last_counted > delay` (delay = 0.05·C insertions; results are insensitive to the value), then set `last_counted = now`;
  - eviction is FIFO-reinsertion with a 1-bit counter;
  - effect: about 60% fewer promotions and a lower miss ratio than FIFO-reinsertion.
- **Age-Guided Eviction (AGE, VLDB'26):**
  - threshold = factor·C/miss-ratio, factor 0.5;
  - at M's tail, while freq > 0: decrement; if `now - last_access >= threshold`, evict; otherwise reinsert and examine the next object;
  - needs an online miss-ratio estimate.
- **Skip region (S4-FIFO):** hits to objects among the most recent κ·|S| insertions into S do not count (κ = 0.25). Equivalent to a pass-through FIFO stage whose hits do not count, placed in front of S.
- **Ghost-to-main threshold (S4-FIFO):** a ghost hit goes to M only after τ ghost hits; otherwise it re-enters S.
- **SIEVE hand as M:** visited objects keep their place and lose the bit; the hand evicts the first unvisited object. Weak on scans without a ghost.
- **S3-FIFO-d (SOSP'23):**
  - two small ghosts (5% each) track evictions from S and from M;
  - when one gets at least 2× the other's hits (over 100), move 0.1% of capacity toward that queue;
  - helped only adversarial traces.
- **Group-level utility (GL-Cache):** objects inserted at similar times form groups with similar reuse; learn per-group utility, evict the worst group merged with its insertion-time neighbours.

## 3. Diagnostics backlog (host side, counterfactual)

1. **One-hit share in windows** of |S|, C and C+|G| distinct objects, on the observed vs the unseen part; plus hits-during-stay at eviction (0, 1, 2+) for S3-FIFO and Belady, split by S and M.
2. **Demotion speed/precision sweep** over |S|/C ∈ {1, 2, 5, 10, 20, 40}%, and promotion precision: the share of S→M moves that get a hit in M.
3. **Cache-time attribution:**
   - a stay costs `T_evict − T_insert`;
   - bucket objects by frequency (1, 2, 3–7, 8+) and report each policy's cache-time share and hits per cache-time;
   - this shows where 3L-Cache's advantage comes from.
4. **Miss taxonomy for S3-FIFO:** compulsory; evicted from S and missed while still in the ghost; returned after the ghost; evicted from M (split by frequency at eviction). Diff the requests missed only by S3-FIFO against those missed only by 3L-Cache.
5. **Belady early eviction with tolerance W:** drop x when its next use is more than W·C/mr away; sweep W ∈ {0.5, 1, 2, 5}. This is the ceiling, and the label, for a "reused before eviction?" classifier.
6. **Offline FIFO-reinsertion pruning:** iteratively skip promotions that turn out useless (no hit before the next eviction), and report misses, promotions and promotion efficiency per round.
7. **Oracle at the candidate, across structures** (view_headroom generalized): M as FIFO-reinsertion, SIEVE or LRU; |S| ∈ {5, 10, 20}%; |G| ∈ {0.5, 1, 2}·|M|.
8. **Drift and transfer:**
   - the cumulative miss difference over time against 3L-Cache;
   - working-set overlap (Jaccard) between the observed and the unseen part;
   - P(reuse within C/mr | hits so far, age) on the observed vs the unseen part, to see whether prefix-tuned thresholds transfer.
