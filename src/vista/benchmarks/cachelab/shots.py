"""Few-shot worked examples: how published designs were derived, replayed on the lab's data.

Each example follows one published analysis through the same tools the agent
has: the question, the analysis (the tool call and the exact numbers it
returns here), the insight, the mechanism it led to, and that mechanism's
measured effect. They demonstrate a process; the numbers come from the
development data (or the engineer's prefix), never from scoring data.
"""

from __future__ import annotations

from .lab import BASELINE, Lab


def _pick_trace(lab: Lab) -> str:
    names = list(lab.dev)
    real = [name for name in names if lab.dev[name].spec.family in ("oracleGeneral", "prefix")]
    pool = real or names
    return max(pool, key=lambda name: lab.dev[name].requests)


def _change(lab: Lab, identity: str, base: str, trace: str) -> str:
    rows = [row for row in lab.compare(lab.experiment(identity), lab.experiment(base))["rows"] if row["trace"] == trace]
    return ", ".join(f"C={row['capacity']}: {row['relative']:+.2%}" for row in rows)


def build_shots(lab: Lab) -> str:
    missing = [name for name in ("lru", "mq", "s4fifo") if name not in lab.references]
    if missing:
        raise ValueError(f"shots need the reference experiments {missing}")
    trace = _pick_trace(lab)
    sizes = lab.capacities[trace]
    capacity = sizes[len(sizes) // 2]
    workload = lab.workload(trace, visual=False).data
    within, one_hit = workload["reuse_within"], workload["one_hit"]
    positions = lab.positions(BASELINE, trace, capacity, visual=False).data["positions"]
    labels = positions["labels"]
    quick = sum(share for label, share in zip(labels, positions["queues"]["S"]["share"], strict=True) if label in ("0-0.01", "0.01-0.05"))
    tail_s = sum(share for label, share in zip(labels, positions["queues"]["S"]["share"], strict=True) if label in ("0.75-1", "1-1.5", "1.5-2", ">=2"))
    tail_g = sum(share for label, share in zip(labels, positions["ghosts"]["G"]["share"], strict=True) if label in ("0.75-1", "1-1.5", "1.5-2", ">=2"))
    frequency = lab.workload(trace, visual=False).text.split("access frequency")[1].split("\n")[0]
    c_key, s_key = f"C({capacity})", f"0.1C({capacity})"
    return "\n".join(
        [
            "# Worked examples: how published designs were derived",
            "",
            "Each example replays a published analysis on your data with your own tools and shows the "
            f"mechanism it led to. Numbers are for trace `{trace}` at C={capacity}. They demonstrate a way "
            "of working, not an answer: the right mechanism here may be different. The reference "
            "policies are recorded experiments (`lru`, `mq`, `s4fifo`) you can view, diff, read "
            "(experiment_info with include_code) and import (`from mq import MQ`, `from s4fifo import "
            "S4FIFO`); viewing them is free.",
            "",
            "## 1. S3-FIFO (SOSP 2023): one-hit wonders, then quick demotion",
            "- Question: how many newly admitted objects are never requested again while they could still be resident?",
            f"- Analysis: view_workload(trace=\"{trace}\") -> in windows holding C={capacity} distinct objects, "
            f"{one_hit[c_key]['one_hit_share']:.0%} of the objects are requested only once; in windows of 0.1C, "
            f"{one_hit[s_key]['one_hit_share']:.0%}.",
            "- Insight: at the scale of the cache most new objects are dead on arrival; holding each for a "
            "whole cache lifetime, as LRU and FIFO do, spends most of the space on them.",
            "- Mechanism: admit new objects to a small FIFO (10% of the cache) and promote only those "
            "requested again there; remember keys evicted from it in a ghost queue, so a quick return goes "
            "straight to the main queue (experiment `s3fifo`).",
            f"- Result here: S3-FIFO against LRU, miss ratio change {_change(lab, BASELINE, 'lru', trace)} "
            "(view_overview experiment=s3fifo compare_to=lru).",
            "",
            "## 2. MQ (USENIX ATC 2001): reuse distance and frequency, then frequency levels with decay",
            "- Question: for a cache whose short-range reuse was absorbed upstream (a storage server behind "
            "client caches), is recency or frequency the better signal?",
            f"- Analysis: view_workload(trace=\"{trace}\") -> {within[s_key]:.0%} of all requests reuse a key "
            f"within 0.1C distinct objects and {within[c_key]:.0%} within C; access frequency (requests per "
            f"object -> share of objects / share of requests){frequency.split(':', 1)[1] if ':' in frequency else frequency}.",
            "- Insight (the paper's three properties): when reuse distances exceed the cache, recency alone "
            "cannot catch them; an object must stay at least as long as typical reuse distances (minimal "
            "lifetime), its frequency should set its priority, and frequency must decay so formerly hot "
            "objects leave.",
            "- Mechanism: eight LRU queues by log2(frequency); an object idle for LIFETIME requests drops one "
            "level; evicted keys keep their frequency in a history (Qout) of 4x the cache (experiment `mq`). "
            "That history alone exceeds this task's memory bound, so a design borrowing it must shrink it.",
            f"- Result here: MQ against S3-FIFO, {_change(lab, 'mq', BASELINE, trace)}.",
            "",
            "## 3. S4-FIFO (OSDI 2026): where hits land, then per-workload knobs",
            "- Question: S3-FIFO's fixed knobs (S = 10%, ghost = 0.9C, promote at 2 hits, ghost hit -> main) "
            "fit some workloads and not others. Which statistics reveal the right setting for a workload?",
            f"- Analysis: view_positions(experiment=\"s3fifo\", trace=\"{trace}\", capacity={capacity}) -> in S, "
            f"{quick:.0%} of hits land within the newest 5% of the queue (quick re-requests) and {tail_s:.0%} in "
            f"its last quarter or beyond; {tail_g:.0%} of ghost hits land in the ghost's last quarter or beyond.",
            "- Insight: quick re-requests look like reuse but often are not, so counting them promotes "
            "noise; hits piling up at S's tail mean S is too short for this workload; ghost hits at the "
            "ghost's tail mean its history is too short; the right knobs therefore differ per workload "
            "and can be read from such statistics early in a trace.",
            "- Mechanism: two more knobs on S3-FIFO, SMALL_SKIP_RATIO (re-requests within the newest "
            "fraction of S do not count toward promotion) and GHOST_TO_MAIN_THRESHOLD (a ghost hit "
            "re-enters S instead of main), and a control plane that measures such statistics over the "
            "first 20% of a trace and sets all five knobs for the rest; in the paper a learned model "
            "maps statistics to knobs. With default knobs `s4fifo` equals S3-FIFO.",
        ]
    )
