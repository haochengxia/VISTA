"""The engineer role: study the first part of one workload, decide, be scored on the rest.

S4-FIFO's control plane observes the first 20% of a trace and a learned model
picks the heuristic's knobs for the remaining 80%. Here the agent plays that
control plane: it sees only the prefix (as the trace `target`), runs views
and experiments on it, and submits one policy.

Deployment is a handover, as in production: S3-FIFO serves the prefix, and at
the split the submitted policy takes over its live cache (policy_api.Handover:
resident objects by queue and order, frequency counters, ghost). Inherited
state matters most for large caches, where it persists long after the switch.
Only the requests after the split are scored, against S3-FIFO (which simply
continues), S3-FIFO with prefix-tuned knobs switched on at the split (the best
grid configuration per cache size), and published algorithms from
libCacheSim, which run from a cold cache over the whole trace (no handover:
an optimistic reference). The agent's own experiments hand over too, after
PREFIX_HANDOVER of the prefix, unless it asks for another point.

Nothing about the trace's origin reaches the agent: both the prefix and the
full trace are named `target`, the full trace lives in a directory no
experiment mounts, and cache sizes derive from the prefix alone (sizing them
from the full trace would reveal how many new objects arrive later).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import zoo
from .engine import BASELINE_SOURCE, HIT, Run, RunError, Sandbox
from .lab import MEMORY_LIMIT_RATIO, SAME, Lab
from .search import DEFAULT, GRID, policy_source
from .traces import (
    Trace,
    TraceSpec,
    capacities,
    materialize,
    split_trace,
)

DEFAULT_SPLIT = 0.2
NAME = "target"
# Cache sizes as shares of the prefix's distinct objects; the largest is where
# a handover is hardest.
ENGINEER_CAPACITY_FRACTIONS = (0.01, 0.03, 0.1, 0.3)
PREFIX_HANDOVER = 0.2  # experiments on the prefix: S3-FIFO serves its first 20%
PROTOCOL = "engineer-handover-v2"  # v2: prefix tuning by coordinate descent (v1: full grid)


@dataclass(frozen=True)
class EngineerTask:
    full: Trace  # private: never mounted for experiments, never named
    prefix: Trace  # what the agent studies, as trace `target`
    split: int
    fraction: float
    capacities: tuple[int, ...]
    secrets: tuple[str, ...]  # identifiers to redact from anything the agent reads
    fractions: tuple[float, ...] = ENGINEER_CAPACITY_FRACTIONS


def prepare(spec: TraceSpec, root: Path, *, fraction: float = DEFAULT_SPLIT, fractions: Sequence[float] = ENGINEER_CAPACITY_FRACTIONS) -> EngineerTask:
    root = root.resolve()
    full = materialize(dataclasses.replace(spec, name=NAME, split="heldout"), root / "private")
    prefix, split = split_trace(full, fraction, root / "traces" / "dev", NAME)
    secrets = [spec.name]
    if spec.source:
        source = Path(spec.source)
        secrets += [str(source), source.name, source.name.split(".")[0], str(source.parent)]
    if split != int(full.requests * fraction):
        raise RuntimeError("the split and the handover point disagree")
    sizes = capacities(prefix.objects, fractions)
    if len(sizes) < len(fractions):
        raise ValueError(f"{spec.name}: the observed part has only {prefix.objects} distinct objects, too few for {len(fractions)} distinct cache sizes")
    return EngineerTask(
        full, prefix, split, fraction, tuple(sizes), tuple(s for s in secrets if len(s) >= 5), tuple(fractions)
    )


def make_lab(task: EngineerTask, root: Path, sandbox: Sandbox, *, references: Sequence[str] = (), workers: int = 1, timeout: float = 900) -> Lab:
    lab = Lab(
        root,
        sandbox,
        dev=(),
        heldout=(),
        dev_traces=(task.prefix,),
        fractions=task.fractions,
        references=references,
        workers=workers,
        timeout=timeout,
        redactions=task.secrets,
        handover=PREFIX_HANDOVER,
    )
    if tuple(lab.capacities[NAME]) != task.capacities:
        raise RuntimeError("the lab and the task disagree on cache sizes")
    return lab


def prune_arrays(root: Path) -> int:
    """Delete the per-request arrays of every simulated cell under a finished
    task's lab (summaries stay, so reports still load); returns bytes freed."""
    freed = 0
    for path in (root / "runs").glob("*/*.bin"):
        freed += path.stat().st_size
        path.unlink()
    for path in (root / "zoo").glob("*/trace.oracleGeneral"):
        freed += path.stat().st_size
        path.unlink()
    return freed


def suffix_miss_ratio(run: Run, split: int) -> float:
    return float((run.outcome[split:] != HIT).mean())


_KNOBS = ("SMALL_RATIO", "GHOST_RATIO", "MOVE_THRESHOLD", "MAX_FREQ")


def tuned_configs(baselines: dict[str, Any] | None, capacities: Sequence[int]) -> dict[int, dict[str, Any]] | None:
    """The prefix-tuned knobs per cache size from a baselines report: chosen on the
    observed part only, so the agent may see them (the report's scores on the rest
    of the workload must stay hidden). None unless every size has a configuration."""
    if not baselines:
        return None
    configs = {row["capacity"]: row.get("prefix_tuned_config") for row in baselines.get("rows", [])}
    if not all(configs.get(capacity) and set(configs[capacity]) == set(_KNOBS) for capacity in capacities):
        return None
    return {capacity: configs[capacity] for capacity in capacities}


def tuned_reference_source(configs: dict[int, dict[str, Any]]) -> str:
    """S3-FIFO with the prefix-tuned knobs of each cache size, as ordinary policy code."""
    rows = "\n".join(f"    {capacity}: ({', '.join(repr(config[name]) for name in _KNOBS)})," for capacity, config in sorted(configs.items()))
    return (
        "from s3fifo import S3FIFO\n\n"
        "# S3-FIFO's knobs chosen separately for each cache size by a search on the observed part\n"
        "# (coordinate descent from the defaults; each candidate took over at the handover point).\n"
        f"# capacity: ({', '.join(_KNOBS)})\n"
        f"KNOBS = {{\n{rows}\n}}\n\n\n"
        "class CachePolicy(S3FIFO):\n"
        "    def __init__(self, capacity, ctx):\n"
        "        knobs = KNOBS.get(capacity)\n"
        "        if knobs is not None:\n"
        f"            self.{', self.'.join(_KNOBS)} = knobs\n"
        "        super().__init__(capacity, ctx)\n"
    )


def prefix_tuned(lab: Lab, task: EngineerTask, *, grid: dict[str, Sequence[Any]] = GRID, passes: int = 2) -> dict[int, dict[str, Any]]:
    """S3-FIFO tuned on the prefix (with the lab's handover, like the agent's
    experiments), per cache size, then switched on at the split of the live
    S3-FIFO cache. Coordinate descent from the defaults: each pass tries every
    value of one knob at a time (all sizes in one batch) and keeps the best,
    about 26 configurations per size instead of the grid's 288, so whole traces
    of tens of millions of requests stay affordable. Run after the agent's
    session."""
    base = lab.experiment("s3fifo")
    reference = {c: base.run(NAME, c).summary["miss_ratio"] for c in task.capacities}
    current = {c: (0.0, dict(DEFAULT)) for c in task.capacities}
    seen: set[tuple[int, tuple[tuple[str, Any], ...]]] = set()
    for _ in range(passes):
        moved = False
        for knob, values in grid.items():
            entries, owners = [], []
            for capacity in task.capacities:
                for value in values:
                    config = {**current[capacity][1], knob: value}
                    key = (capacity, tuple(sorted(config.items())))
                    if config == current[capacity][1] or key in seen:
                        continue
                    seen.add(key)
                    entries.append((policy_source(config), task.prefix, capacity))
                    owners.append((capacity, config))
            for (capacity, config), run in zip(owners, lab.store.simulate_many(entries, handover=lab.handover), strict=True):
                if not isinstance(run, Run):
                    continue
                theirs = reference[capacity]
                change = (run.summary["miss_ratio"] - theirs) / theirs if theirs else 0.0
                run.drop_arrays()  # only the summary is needed
                if change < current[capacity][0]:
                    current[capacity] = (change, config)
                    moved = True
        if not moved:
            break
    finals = lab.store.simulate_many([(policy_source(config), task.full, capacity) for capacity, (_, config) in current.items()], handover=task.fraction)
    chosen = {}
    for (capacity, (change, config)), run in zip(current.items(), finals, strict=True):
        chosen[capacity] = {
            "config": config,
            "prefix_relative_change": change,
            "suffix_miss_ratio": suffix_miss_ratio(run, task.split) if isinstance(run, Run) else None,
        }
    return chosen


def evaluate(
    lab: Lab,
    task: EngineerTask,
    identity: str,
    *,
    zoo_dir: Path,
    workers: int = 1,
    tune: bool = True,
    baselines: dict[str, Any] | None = None,
    verified: bool = True,
    keep_arrays: bool = True,
) -> dict[str, Any]:
    """Score `identity` on the requests after the split.

    `baselines` is an earlier report for the same task (engineer-baselines):
    its prefix-tuned and published results are reused instead of recomputed.
    verified=False is for training feedback only: fast simulation, no memory
    or tampering checks. keep_arrays=False deletes the candidate's full-trace
    logs once scored (nothing views them; a session's hindsight scoring of
    every experiment would otherwise keep tens of GB)."""
    experiment = lab.experiment(identity)
    full_cells = [(task.full, c) for c in task.capacities]
    prefix_cells = [(task.prefix, c) for c in task.capacities]
    # S3-FIFO (and so the baseline) continues unchanged; anything else takes over at the split.
    switch = None if experiment.source == BASELINE_SOURCE else task.fraction
    cand = lab.store.simulate(experiment.source, full_cells, verified=verified, handover=switch)
    base = lab.store.simulate(BASELINE_SOURCE, full_cells, verified=verified)
    cand_prefix = lab.store.simulate(experiment.source, prefix_cells, verified=True, handover=experiment.handover) if verified else {}
    if baselines is not None:
        if [row["capacity"] for row in baselines["rows"]] != list(task.capacities) or baselines.get("split") != task.split:
            raise ValueError("the baselines report belongs to a different task")
        if baselines.get("protocol") != PROTOCOL:
            raise ValueError(f"the baselines report is from protocol {baselines.get('protocol')!r}, not {PROTOCOL}")
        published = {row["capacity"]: {name: {"suffix_miss_ratio": value} for name, value in row["published"].items()} for row in baselines["rows"]}
        tuned = {
            row["capacity"]: {"config": row["prefix_tuned_config"], "suffix_miss_ratio": row["prefix_tuned_s3fifo"]}
            for row in baselines["rows"]
            if row.get("prefix_tuned_s3fifo") is not None
        }
    else:
        published = zoo.suffix_miss_ratios(task.full.keys(), task.split, task.capacities, zoo_dir, workers=workers)
        tuned = prefix_tuned(lab, task) if tune else {}
    rows, failures, memory_violations, mismatches = [], [], [], []
    for capacity in task.capacities:
        c, b = cand[(NAME, capacity)], base[(NAME, capacity)]
        if isinstance(c, RunError) or isinstance(b, RunError):
            failures.append({"capacity": capacity, "error": (c if isinstance(c, RunError) else b).error})
            continue
        ratio = None
        if verified:
            ratio = c.summary["policy_memory_bytes"] / max(1, b.summary["policy_memory_bytes"])
            if ratio > MEMORY_LIMIT_RATIO:
                memory_violations.append({"capacity": capacity, "ratio": round(ratio, 3)})
            fast, checked = experiment.cells.get((NAME, capacity)), cand_prefix[(NAME, capacity)]
            if not (isinstance(fast, Run) and isinstance(checked, Run) and fast.summary["misses"] == checked.summary["misses"]):
                mismatches.append({"capacity": capacity})
        mine, s3 = suffix_miss_ratio(c, task.split), suffix_miss_ratio(b, task.split)
        algorithms = {
            name: result["suffix_miss_ratio"]
            for name, result in published[capacity].items()
            if "suffix_miss_ratio" in result and not (result["suffix_miss_ratio"] >= 1.0 and s3 < 1.0)  # degenerate configs
        }
        online = {name: value for name, value in algorithms.items() if name not in zoo.OFFLINE}
        best_name = min(online, key=online.get) if online else None
        competitors = sorted(online.values())
        rows.append(
            {
                "capacity": capacity,
                "candidate": mine,
                "s3fifo": s3,
                "relative_vs_s3fifo": (mine - s3) / s3 if s3 else 0.0,
                "prefix_tuned_s3fifo": tuned.get(capacity, {}).get("suffix_miss_ratio"),
                "prefix_tuned_config": tuned.get(capacity, {}).get("config"),
                "best_published": best_name,
                "best_published_miss_ratio": online.get(best_name),
                "relative_vs_best_published": (mine - online[best_name]) / online[best_name] if best_name and online[best_name] else None,
                "rank_among_published": 1 + sum(value < mine - SAME * mine for value in competitors),
                "published": algorithms,
                "belady": algorithms.get("Belady"),
                "memory_ratio": None if ratio is None else round(ratio, 3),
            }
        )
    if not keep_arrays and switch is not None:
        for run in cand.values():
            if isinstance(run, Run):
                run.drop_arrays()
    relative = [row["relative_vs_s3fifo"] for row in rows]
    report = {
        "role": "engineer",
        "protocol": PROTOCOL,
        "deployment": "handover at the split from a live S3-FIFO cache; published algorithms from a cold cache",
        "verified": verified,
        "experiment": identity,
        "name": experiment.name,
        "split_fraction": task.fraction,
        "split": task.split,
        "requests": task.full.requests,
        "rows": rows,
        "failures": failures,
        "memory_violations": memory_violations,
        "fast_verified_mismatches": mismatches,
        "mean_relative_vs_s3fifo": float(np.mean(relative)) if rows and not failures else None,
        "mean_relative_vs_best_published": float(np.mean([r["relative_vs_best_published"] for r in rows if r["relative_vs_best_published"] is not None])) if rows else None,
        "mean_relative_prefix_tuned_vs_s3fifo": float(np.mean([(r["prefix_tuned_s3fifo"] - r["s3fifo"]) / r["s3fifo"] for r in rows if r["prefix_tuned_s3fifo"] is not None and r["s3fifo"]])) if tuned else None,
    }
    report["valid"] = not (failures or memory_violations or mismatches)
    report["score"] = report["mean_relative_vs_s3fifo"] if report["valid"] else None
    return report
