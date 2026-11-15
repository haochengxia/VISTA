"""The cache-policy lab: experiments, views, and the final evaluation.

The same Lab backs the VISTA session (the agent), the command line (people)
and the parameter-search baseline, so every arm sees identical numbers.

The objective is the mean relative miss-ratio change against S3-FIFO over
every (trace, cache size) cell; negative is better. Final scores come from
verified simulation on held-out traces the agent never sees, with a bound on
policy metadata so that unbounded history does not count as an improvement.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import analysis, views
from .engine import (
    BASELINE_SOURCE,
    HIT,
    NO_QUEUE,
    REFERENCE_POLICIES,
    SANDBOX_DIR,
    Run,
    RunError,
    RunStore,
    Sandbox,
    handover_index,
)
from .policy_check import check_policy
from .traces import (
    DEFAULT_CAPACITY_FRACTIONS,
    DEV_SUITE,
    HELDOUT_SUITE,
    Trace,
    TraceSpec,
    capacities,
    materialize_suite,
)

BASELINE = "s3fifo"
REFERENCE_NAMES = {
    "s4fifo": "S4-FIFO heuristic, default knobs (OSDI 2026)",
    "mq": "MQ, Multi-Queue (Zhou et al., USENIX ATC 2001)",
    "lru": "LRU",
    "fifo": "FIFO",
}
SAME = 0.001  # |relative change| below 0.1% counts as unchanged
MEMORY_LIMIT_RATIO = 2.0
MAX_EVENT_ROWS = 200
OUTCOME_NAMES = {0: "hit", 1: "miss-cold", 2: "miss", 3: "miss-bypass"}


class LabError(ValueError):
    """A request the agent can fix (unknown id, bad window, ...)."""


@dataclass
class Experiment:
    id: str
    name: str
    source: str
    hypothesis: str
    prediction: dict[str, Any] | None
    cells: dict[tuple[str, int], Run | RunError]
    problems: list[str] = field(default_factory=list)
    handover: float | None = None  # S3-FIFO served this share of each trace before the policy took over

    @property
    def ok(self) -> bool:
        return not self.problems and all(isinstance(cell, Run) for cell in self.cells.values())

    def run(self, trace: str, capacity: int) -> Run:
        cell = self.cells.get((trace, capacity))
        if not isinstance(cell, Run):
            raise LabError(f"experiment {self.id} has no successful run for {trace} at C={capacity}")
        return cell

    def first_error(self) -> tuple[tuple[str, int], RunError] | None:
        for cell, value in self.cells.items():
            if isinstance(value, RunError):
                return cell, value
        return None


@dataclass
class View:
    """What one lab operation returns: exact text, and an image when visual."""

    text: str
    png: bytes | None = None
    data: dict[str, Any] = field(default_factory=dict)


def _prune(cells) -> None:
    for run in cells:
        if isinstance(run, Run):
            for path in run.directory.glob("*.bin"):
                path.unlink(missing_ok=True)


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _series(values: np.ndarray, digits: int = 3) -> str:
    return " ".join("nan" if not np.isfinite(v) else f"{v:.{digits}f}" for v in values)


class Lab:
    def __init__(
        self,
        root: Path,
        sandbox: Sandbox,
        *,
        dev: Sequence[TraceSpec] = DEV_SUITE,
        heldout: Sequence[TraceSpec] = HELDOUT_SUITE,
        fractions: Sequence[float] = DEFAULT_CAPACITY_FRACTIONS,
        buckets: int = 48,
        timeout: float = 900,
        workers: int = 1,
        references: Sequence[str] = (),
        dev_traces: Sequence[Trace] = (),
        redactions: Sequence[str] = (),
        handover: float | None = None,
        sizes: dict[str, Sequence[int]] | None = None,
    ) -> None:
        """`handover`: by default every experiment starts with S3-FIFO serving
        this share of each trace, then the policy takes over the live cache
        (the engineer role's deployment); None runs policies from a cold cache."""
        self.root = root.resolve()
        self.handover = handover
        self.store = RunStore(self.root, sandbox, timeout=timeout)
        self.fractions = tuple(fractions)
        self.workers = workers
        self.dev = {
            trace.spec.name: trace
            for trace in materialize_suite(tuple(dev), self.root / "traces" / "dev", workers=workers)
        }
        # Already materialized traces (the engineer role's prefix).
        self.dev.update({trace.spec.name: trace for trace in dev_traces})
        # Strings that must never reach the agent (paths, source names), longest first.
        self._redactions = sorted({str(self.root), *(r for r in redactions if r)}, key=len, reverse=True)
        self.heldout_specs = tuple(heldout)
        self.buckets = buckets
        # `sizes` fixes a trace's cache sizes (the analyst replays each case at its engineer's sizes).
        self.capacities = {name: list((sizes or {}).get(name) or capacities(trace.objects, self.fractions)) for name, trace in self.dev.items()}
        self.cells = [(trace, capacity) for name, trace in self.dev.items() for capacity in self.capacities[name]]
        self.experiments: dict[str, Experiment] = {}
        self.recorded: set[str] = set()
        self._serial = 0
        self._indexes: dict[str, analysis.TraceIndex] = {}
        self._distances: dict[str, np.ndarray] = {}
        # S3-FIFO taking over from S3-FIFO changes nothing, so the baseline never switches.
        baseline = self._record(BASELINE, "S3-FIFO baseline", BASELINE_SOURCE, "reference policy", None, handover=None)
        if not baseline.ok:
            raise RuntimeError(f"the baseline failed to simulate: {baseline.first_error()}")
        # Published policies, simulated like any experiment and free to view, diff and import.
        self.references = tuple(references)
        for name in self.references:
            if name not in REFERENCE_POLICIES:
                raise ValueError(f"unknown reference policy {name!r}; known: {', '.join(REFERENCE_POLICIES)}")
            reference = self._record(name, REFERENCE_NAMES[name], (SANDBOX_DIR / f"{name}.py").read_text(encoding="utf-8"), "published reference policy", None)
            if not reference.ok:
                raise RuntimeError(f"reference {name} failed to simulate: {reference.first_error()}")

    # -- bookkeeping ----------------------------------------------------------------

    def redact(self, text: str) -> str:
        for secret in self._redactions:
            text = text.replace(secret, "<hidden>")
        return text

    def _next_identity(self) -> str:
        self._serial += 1
        return f"e{self._serial}"

    def is_reference(self, identity: str) -> bool:
        return identity == BASELINE or identity in self.references or identity in self.recorded

    def record(self, identity: str, name: str, source: str, note: str) -> Experiment:
        """A policy recorded by the host (an earlier decision), free to view and diff like a reference."""
        if identity in self.experiments:
            raise ValueError(f"experiment {identity} exists")
        self.recorded.add(identity)
        return self._record(identity, name, source, note, None)

    def _record(self, identity, name, source, hypothesis, prediction, *, handover: float | None | str = "default") -> Experiment:
        handover = self.handover if handover == "default" else handover
        problems = check_policy(source) if identity != BASELINE else []
        cells = {} if problems else self.store.simulate(source, self.cells, handover=handover)
        experiment = Experiment(identity, name, source, hypothesis, prediction, cells, problems, handover)
        self.experiments[identity] = experiment
        return experiment

    def experiment(self, identity: str) -> Experiment:
        found = self.experiments.get(identity)
        if found is None:
            raise LabError(f"unknown experiment {identity!r}; known: {', '.join(self.experiments)}")
        return found

    def trace(self, name: str) -> Trace:
        found = self.dev.get(name)
        if found is None:
            raise LabError(f"unknown trace {name!r}; dev traces: {', '.join(self.dev)}")
        return found

    def capacity(self, trace: str, capacity: int) -> int:
        allowed = self.capacities[self.trace(trace).spec.name]
        if capacity not in allowed:
            raise LabError(f"{trace} is simulated at C in {allowed}, not {capacity}")
        return capacity

    def index(self, trace: str) -> analysis.TraceIndex:
        if trace not in self._indexes:
            self._indexes[trace] = analysis.TraceIndex(self.trace(trace).keys())
        return self._indexes[trace]

    def window(self, trace: str, start: int | None, end: int | None) -> tuple[int, int]:
        n = self.trace(trace).requests
        if start is not None and not 0 <= start < n:
            raise LabError(f"start must be in [0, {n})")
        if end is not None and not 0 < end <= n:
            raise LabError(f"end must be in (0, {n}]")
        first, stop = analysis.window(n, start, end)
        if stop <= first:
            raise LabError("end must be after start")
        return first, stop

    def public_suite(self) -> list[dict[str, Any]]:
        return [{**trace.public(), "capacities": self.capacities[name]} for name, trace in self.dev.items()]

    # -- comparisons -----------------------------------------------------------------

    def compare(self, candidate: Experiment, base: Experiment) -> dict[str, Any]:
        rows = []
        for trace, capacity in self.cells:
            cell = (trace.spec.name, capacity)
            b, c = base.cells.get(cell), candidate.cells.get(cell)
            if not isinstance(b, Run) or not isinstance(c, Run):
                continue
            mb, mc = b.summary["miss_ratio"], c.summary["miss_ratio"]
            row = {
                "trace": cell[0],
                "capacity": capacity,
                "base": mb,
                "candidate": mc,
                "delta": mc - mb,
                "relative": (mc - mb) / mb if mb else 0.0,
            }
            if self.handover:
                # The same window for both: requests after the lab's handover point.
                point = handover_index(trace.requests, self.handover)
                try:
                    ab, ac = float((b.outcome[point:] != HIT).mean()), float((c.outcome[point:] != HIT).mean())
                except FileNotFoundError:  # pruned batch runs keep summaries only
                    ab = ac = None
                if ab is not None:
                    row.update(after_base=ab, after_candidate=ac, after_relative=(ac - ab) / ab if ab else 0.0)
            rows.append(row)
        relative = [row["relative"] for row in rows]
        after = [row["after_relative"] for row in rows if "after_relative" in row]
        return {
            "mean_relative_change_after_handover": float(np.mean(after)) if after and len(after) == len(rows) else None,
            "rows": rows,
            "cells": len(rows),
            "mean_relative_change": float(np.mean(relative)) if rows else math.nan,
            "better": sum(r < -SAME for r in relative),
            "worse": sum(r > SAME for r in relative),
            "unchanged": sum(abs(r) <= SAME for r in relative),
            "worst": max(rows, key=lambda row: row["relative"]) if rows else None,
            "best": min(rows, key=lambda row: row["relative"]) if rows else None,
        }

    def _comparison_text(self, comparison: dict[str, Any], base_id: str, cand_id: str) -> str:
        lines = [
            f"mean relative miss-ratio change of {cand_id} vs {base_id}: {_pct(comparison['mean_relative_change'])} "
            f"over {comparison['cells']} cells ({comparison['better']} better, {comparison['worse']} worse, "
            f"{comparison['unchanged']} unchanged within 0.1%)"
        ]
        after = comparison.get("mean_relative_change_after_handover")
        if after is not None:
            lines.append(
                f"after the handover point ({self.handover:.0%} into each trace; before it every policy matches s3fifo): "
                f"mean relative change {_pct(after)}"
            )
        if comparison["worst"] is not None:
            worst, best = comparison["worst"], comparison["best"]
            lines.append(
                f"best cell {best['trace']} C={best['capacity']} {_pct(best['relative'])}; "
                f"worst cell {worst['trace']} C={worst['capacity']} {_pct(worst['relative'])}"
            )
        lines.append(f"trace    capacity  {base_id:>8}  {cand_id:>8}  delta    relative" + ("  after_handover" if after is not None else ""))
        for row in comparison["rows"]:
            lines.append(
                f"{row['trace']:<8} {row['capacity']:>8}  {row['base']:8.5f}  {row['candidate']:8.5f}  "
                f"{row['delta']:+.5f} {_pct(row['relative'])}" + (f"  {_pct(row['after_relative'])}" if "after_relative" in row else "")
            )
        return "\n".join(lines)

    # -- operations ------------------------------------------------------------------

    def run_experiment(
        self, name: str, code: str, hypothesis: str, prediction: dict[str, Any] | None, *, handover: float | None | str = "default"
    ) -> View:
        """`handover`: the share of each trace S3-FIFO serves before this policy
        takes over (0 or None: cold start); by default the lab's setting."""
        if handover != "default":
            handover = float(handover) if handover else None
            if handover is not None and not 0 < handover <= 0.9:
                raise LabError("handover_at must be 0 (cold start) or a fraction up to 0.9")
        identity = self._next_identity()
        experiment = self._record(identity, name, code, hypothesis, prediction, handover=handover)
        if experiment.problems:
            text = f"experiment {identity} ({name}) was rejected before simulation:\n- " + "\n- ".join(experiment.problems)
            return View(text, data={"experiment": identity, "ok": False})
        failure = experiment.first_error()
        if failure is not None:
            (trace, capacity), error = failure
            failed = sum(isinstance(value, RunError) for value in experiment.cells.values())
            text = (
                f"experiment {identity} ({name}) failed in {failed} of {len(experiment.cells)} cells.\n"
                f"first failure: {trace} C={capacity}: {error.error}\n{error.traceback[-2500:]}"
            )
            return View(text, data={"experiment": identity, "ok": False})
        comparison = self.compare(experiment, self.experiments[BASELINE])
        warnings = self._instrumentation_warnings(experiment)
        parts = [f"experiment {identity} ({name}) completed.", self._handover_text(experiment), self._comparison_text(comparison, BASELINE, identity)]
        if prediction and prediction.get("expected_change_pct") is not None:
            expected = float(prediction["expected_change_pct"])
            observed = comparison["mean_relative_change"] * 100
            parts.append(f"prediction check: expected {expected:+.2f}%, observed {observed:+.2f}% (error {observed - expected:+.2f} points)")
        if warnings:
            parts.append("instrumentation warnings (views of this experiment may be wrong until fixed):\n- " + "\n- ".join(warnings))
        overview = self.overview(identity, BASELINE)
        return View(
            "\n\n".join(parts) + "\n\n" + overview.text,
            overview.png,
            {"experiment": identity, "ok": True, "mean_relative_change": comparison["mean_relative_change"]},
        )

    def _handover_text(self, experiment: Experiment) -> str:
        if not experiment.handover:
            return "start: cold cache (the policy served every request)."
        points = ", ".join(f"{name} at request {handover_index(trace.requests, experiment.handover):,}" for name, trace in self.dev.items())
        return (
            f"start: handover. S3-FIFO served the first {experiment.handover:.0%} of each trace, then this policy took over "
            f"its live cache ({points}); before that point it matches s3fifo exactly."
        )

    def _marks(self, identity: str, trace: str, capacity: int) -> list[int]:
        point = self.experiment(identity).run(trace, capacity).summary.get("handover_at")
        return [point] if point else []

    def dev_experiment(self, name: str, code: str, *, prune: bool = False) -> tuple[Experiment, dict[str, Any] | None]:
        """Run code on the dev suite without rendering views (for batch arms).

        With prune, the candidate's per-request logs are deleted afterwards and
        only each cell's summary is kept, so views of it are no longer possible.
        """
        experiment = self._record(self._next_identity(), name, code, "batch", None)
        comparison = self.compare(experiment, self.experiments[BASELINE]) if experiment.ok else None
        if prune:
            _prune(experiment.cells.values())
        return experiment, comparison

    def _instrumentation_warnings(self, experiment: Experiment) -> list[str]:
        warnings = []
        for (trace, capacity), run in experiment.cells.items():
            summary = run.summary
            if summary["instrumentation_errors"]:
                warnings.append(f"{trace} C={capacity}: {summary['instrumentation_errors']} move reports disagree with residency, e.g. {summary['instrumentation_examples'][:2]}")
            if summary["size_mismatches"]:
                warnings.append(f"{trace} C={capacity}: sizes() disagreed with residency {summary['size_mismatches']} times, e.g. {summary['size_examples'][:1]}")
        return warnings[:6]

    def test_policy(self, code: str) -> View:
        """Static check plus one verified cell with memory measured; no experiment is recorded."""
        problems = check_policy(code)
        if problems:
            return View("rejected before simulation:\n- " + "\n- ".join(problems), data={"ok": False})
        name, trace = next(iter(self.dev.items()))
        capacity = self.capacities[name][1 if len(self.capacities[name]) > 1 else 0]
        cell = (trace, capacity)
        candidate = self.store.simulate(code, [cell], verified=True, handover=self.handover)[(name, capacity)]
        if isinstance(candidate, RunError):
            return View(f"policy failed on {name} C={capacity}: {candidate.error}\n{candidate.traceback[-2500:]}", data={"ok": False})
        base = self.store.simulate(BASELINE_SOURCE, [cell], verified=True)[(name, capacity)]
        ratio = self._memory_ratio(candidate, base)
        # No miss ratio: this check is free, so it must not double as an unmetered experiment.
        text = (
            f"policy ran on {name} C={capacity} (verified mode); policy memory {candidate.summary['policy_memory_bytes']:,} B "
            f"= {ratio:.2f}x s3fifo (limit {MEMORY_LIMIT_RATIO:.1f}x); "
            f"instrumentation errors {candidate.summary['instrumentation_errors']}, size mismatches {candidate.summary['size_mismatches']}."
        )
        return View(text, data={"ok": True, "memory_ratio": ratio})

    @staticmethod
    def _memory_ratio(candidate: Run, base: Run) -> float:
        return candidate.summary["policy_memory_bytes"] / max(1, base.summary["policy_memory_bytes"])

    def overview(self, identity: str, compare_to: str | None = None, *, visual: bool = True) -> View:
        experiment = self.experiment(identity)
        if not experiment.ok:
            raise LabError(f"experiment {identity} did not complete")
        base = self.experiment(compare_to) if compare_to else None
        if base is not None and not base.ok:
            raise LabError(f"experiment {compare_to} did not complete")
        delta = base is not None and base.id != experiment.id
        panels = []
        lines = [
            f"overview of {identity}" + (f" minus {base.id}" if delta else "") + f": {'miss-ratio change' if delta else 'miss ratio'} "
            f"per window, in per-mille, one row per cache size; {self.buckets} equal windows per trace"
        ]
        for name, trace in self.dev.items():
            bounds = analysis.edges(0, trace.requests, self.buckets)
            matrix, row_labels = [], []
            lines.append(f"{name} windows of ~{int(np.diff(bounds).mean())} requests")
            for capacity in self.capacities[name]:
                series = analysis.miss_series(experiment.run(name, capacity), bounds)
                overall = experiment.run(name, capacity).summary["miss_ratio"]
                if delta:
                    reference = base.run(name, capacity)
                    series = series - analysis.miss_series(reference, bounds)
                    change = overall - reference.summary["miss_ratio"]
                    relative = change / reference.summary["miss_ratio"]
                    label = f"{change * 1000:+.1f}‰ ({_pct(relative)})"
                else:
                    label = f"{overall:.4f}"
                matrix.append(series)
                row_labels.append(label)
                lines.append(f"  C={capacity:<6} total {label}: " + " ".join(f"{v * 1000:.0f}" for v in series))
            panels.append(
                {
                    "label": f"{name}  ({trace.requests:,} requests, {trace.objects:,} objects)",
                    "bounds": bounds,
                    "matrix": np.vstack(matrix),
                    "capacities": self.capacities[name],
                    "row_labels": row_labels,
                }
            )
        png = None
        if visual:
            title = f"{identity}: miss-ratio change vs {base.id}" if delta else f"{identity}: miss ratio over time"
            subtitle = f"{experiment.name}. Rows: cache size C (objects). Columns: {self.buckets} time windows. Right: whole-trace value."
            png = views.render_overview(panels, title=title, subtitle=subtitle, delta=delta)
        return View("\n".join(lines), png)

    def queues(self, identity: str, trace: str, capacity: int, start: int | None, end: int | None, buckets: int | None = None, *, visual: bool = True) -> View:
        run = self.experiment(identity).run(trace, self.capacity(trace, capacity))
        first, stop = self.window(trace, start, end)
        index = self.index(trace)
        profile = analysis.queue_profile(run, index, first, stop, buckets or 64)
        quality = analysis.eviction_quality(run, index, first, stop)
        bounds = profile["bounds"]
        window_hits = run.outcome[first:stop] == HIT
        lines = [
            f"queue replay of {identity} on {trace} C={capacity}, requests [{first}, {stop}), {bounds.size - 1} windows of ~{int(np.diff(bounds).mean())} requests"
            + "".join(f"; handover to {identity} at request {mark}" for mark in self._marks(identity, trace, capacity)),
            f"window miss ratio {1 - window_hits.mean():.5f}",
            "window starts: " + " ".join(str(int(b)) for b in bounds[:-1]),
            "miss_ratio: " + _series(profile["miss_ratio"]),
        ]
        for label, values in profile["miss_breakdown"].items():
            lines.append(f"misses[{label}] share of requests: " + _series(values))
        for label, values in profile["occupancy"].items():
            lines.append(f"occupancy[{label}] objects: " + _series(values, 0))
        for group in ("hits", "inserts"):
            for label, values in profile[group].items():
                lines.append(f"{group}[{label}] share of requests: " + _series(values))
        for group in ("evictions", "zero_hit_evictions", "moves", "ghost_hits", "ghost_adds", "ghost_drops"):
            for label, values in profile[group].items():
                lines.append(f"{group}[{label}] per 1k requests: " + _series(values * 1000, 1))
        lines.append("eviction quality in this window (return gap = requests until the evicted key is requested again, in cache sizes C):")
        lines.append(json.dumps(quality, separators=(",", ":")))
        png = None
        if visual:
            png = views.render_queues(
                profile,
                title=f"{identity} on {trace}, C={capacity:,}: queue replay",
                subtitle=f"requests {first:,}–{stop:,}; {self.experiment(identity).name}",
                marks=self._marks(identity, trace, capacity),
            )
        return View("\n".join(lines), png)

    def objects(
        self,
        identity: str,
        trace: str,
        capacity: int,
        start: int | None,
        end: int | None,
        *,
        keys: list[int] | None = None,
        select: str | None = None,
        count: int = 12,
        compare_to: str | None = None,
        visual: bool = True,
    ) -> View:
        run = self.experiment(identity).run(trace, self.capacity(trace, capacity))
        first, stop = self.window(trace, start, end)
        index = self.index(trace)
        other = self.experiment(compare_to).run(trace, capacity) if compare_to else None
        if keys:
            chosen = [int(key) for key in keys][:24]
            how = "requested keys"
        else:
            mode = select or "most_remisses"
            if mode not in analysis.SELECTIONS:
                raise LabError(f"select must be one of {analysis.SELECTIONS}")
            try:
                chosen = analysis.select_keys(run, index, mode, first, stop, max(1, min(count, 24)), other)
            except ValueError as exc:
                raise LabError(str(exc)) from None
            how = f"top {len(chosen)} by {mode}" + (f" vs {compare_to}" if other is not None else "")
        if not chosen:
            raise LabError("no objects match this selection in the window")
        timelines = analysis.object_timeline(run, index, chosen, first, stop)
        lines = [f"object timelines of {identity} on {trace} C={capacity}, requests [{first}, {stop}); {how}"]
        lines.append("requests as t:outcome@queue (H hit, M miss, C cold miss, B bypass; queue at hit, or entered on miss); residency as [t0,t1)queue; ghost as [t0,t1)ghost:how ended; evictions as t:queue(hits while resident)")
        budget = 9000
        for row, line in enumerate(timelines):
            code = {0: "H", 1: "C", 2: "M", 3: "B"}
            requests = " ".join(f"{t}:{code[o]}@{q}" for t, o, q in line["requests"][:60])
            if len(line["requests"]) > 60:
                requests += f" ... (+{len(line['requests']) - 60} more)"
            text = (
                f"#{row} key={line['key']} requests_total={line.get('requests_total', 0)} in_window={len(line['requests'])}\n"
                f"  requests: {requests or '-'}\n"
                f"  residency: {' '.join(f'[{a},{b}){q}' for a, b, q in line['residency']) or '-'}\n"
                f"  ghost: {' '.join(f'[{a},{b}){g}:{how}' for a, b, g, how in line['ghost']) or '-'}\n"
                f"  evictions: {' '.join(f'{t}:{q}({h})' for t, q, h in line['evictions']) or '-'}"
            )
            if other is not None:
                theirs = other.outcome[[t for t, _, _ in line["requests"]]] == HIT if line["requests"] else np.zeros(0, bool)
                mine = np.array([o == 0 for _, o, _ in line["requests"]], dtype=bool)
                text += f"\n  vs {compare_to}: only {identity} hits {int((mine & ~theirs).sum())}, only {compare_to} hits {int((~mine & theirs).sum())}"
            budget -= len(text)
            if budget < 0:
                lines.append(f"(remaining {len(timelines) - row} objects omitted from text; query them by key)")
                break
            lines.append(text)
        png = None
        if visual:
            png = views.render_objects(
                timelines,
                queues=run.queues,
                ghosts=run.ghosts,
                start=first,
                end=stop,
                title=f"{identity} on {trace}, C={capacity:,}: object timelines",
                subtitle=f"requests {first:,}–{stop:,}; {how}",
            )
        return View("\n".join(lines), png)

    def distances(self, trace: str) -> np.ndarray:
        if trace not in self._distances:
            self._distances[trace] = analysis.stack_distances(self.index(trace).dense)
        return self._distances[trace]

    def workload(self, trace: str, *, visual: bool = True) -> View:
        """Policy-independent characterization of one trace (MQ's and S3-FIFO's analyses)."""
        name = self.trace(trace).spec.name
        index = self.index(name)
        sizes = self.capacities[name]
        marks = {}
        for capacity in sizes:
            marks[f"0.1C({capacity})"] = 0.1 * capacity
            marks[f"C({capacity})"] = capacity
        reuse = analysis.reuse_profile(self.distances(name), marks)
        frequency = analysis.frequency_distribution(index)
        windows = {}
        for capacity in sizes:
            windows[f"0.1C({capacity})"] = max(1, int(0.1 * capacity))
            windows[f"C({capacity})"] = capacity
        one_hit = analysis.one_hit_wonders(index.dense, windows)
        lines = [
            f"workload of {name}: {index.n:,} requests, {index.unique.size:,} distinct objects; cache sizes C in {list(sizes)}",
            f"reuse distance (distinct objects since the previous request to the same key): first requests {reuse['first_requests_share']:.4f} of all requests; "
            f"median reuse distance {reuse['median_reuse_distance']}",
            "share of all requests that reuse a key within distance d: "
            + ", ".join(f"d<{label}: {value:.4f}" for label, value in reuse["within_marks"].items()),
            "curve (d, share within d): " + " ".join(f"{int(d)}:{v:.4f}" for d, v in zip(reuse["grid"][::4], reuse["share_within"][::4], strict=True)),
            "access frequency (requests per object -> share of objects / share of requests): "
            + ", ".join(
                f"{label}: {o:.4f}/{r:.4f}"
                for label, o, r in zip(frequency["labels"], frequency["object_share"], frequency["request_share"], strict=True)
            ),
            "one-hit wonders (windows holding N distinct objects -> share requested once in the window): "
            + ", ".join(f"{label} N={v['objects']}: {v['one_hit_share']:.4f}" for label, v in one_hit.items() if v["one_hit_share"] is not None),
        ]
        png = None
        if visual:
            png = views.render_workload(
                {"capacities": sizes, "reuse": reuse, "frequency": frequency, "one_hit": one_hit},
                title=f"{name}: workload characterization",
                subtitle=f"{index.n:,} requests, {index.unique.size:,} distinct objects. Solid rules: cache sizes C; dotted: 0.1C (S3-FIFO's small queue).",
            )
        return View("\n".join(lines), png, {"reuse_within": reuse["within_marks"], "one_hit": one_hit})

    def positions(self, identity: str, trace: str, capacity: int, start: int | None = None, end: int | None = None, *, visual: bool = True) -> View:
        """Where in each queue an experiment's hits land (S4-FIFO's statistics)."""
        experiment = self.experiment(identity)
        run = experiment.run(trace, self.capacity(trace, capacity))
        first, stop = self.window(trace, start, end)
        result = analysis.hit_positions(run, self.index(trace), first, stop)
        lines = [
            f"hit positions of {identity} on {trace} C={capacity}, requests [{first}, {stop}). Position = objects that entered the "
            "queue after this one last did, in multiples of the queue's mean size (FIFO: 0 head, ~1 tail).",
            "bins: " + " ".join(result["labels"]),
        ]
        for group, label in (("queues", "hits"), ("ghosts", "ghost hits")):
            for name, data in result[group].items():
                lines.append(
                    f"{name}: {data['hits']} {label}, mean size {data['mean_size']}, median position {data['median_position']}, "
                    f"share by bin: {' '.join(f'{v:.4f}' for v in data['share'])}"
                )
        png = None
        if visual:
            png = views.render_positions(
                result,
                colors=views.queue_colors(run.queues, run.ghosts),
                title=f"{identity} on {trace}, C={capacity:,}: where hits land in each queue",
                subtitle=f"requests {first:,}–{stop:,}; {experiment.name}",
            )
        return View("\n".join(lines), png, {"positions": result})

    def diff(self, identity: str, base_id: str, trace: str, capacity: int, start: int | None, end: int | None, buckets: int | None = None, *, visual: bool = True) -> View:
        capacity = self.capacity(trace, capacity)
        cand = self.experiment(identity).run(trace, capacity)
        base = self.experiment(base_id).run(trace, capacity)
        first, stop = self.window(trace, start, end)
        result = analysis.diff_profile(base, cand, self.index(trace), first, stop, buckets or 64)
        totals = result["totals"]
        lines = [
            f"diff of {identity} against {base_id} on {trace} C={capacity}, requests [{first}, {stop})"
            + "".join(f"; {who} took over at request {mark}" for who in (identity, base_id) for mark in self._marks(who, trace, capacity)),
            f"misses: {base_id} {totals['base_misses']}, {identity} {totals['cand_misses']} "
            f"({totals['cand_misses'] - totals['base_misses']:+d}); requests only {identity} hits: {totals['improvements']}; "
            f"only {base_id} hits: {totals['regressions']}; first differing request: {totals['first_divergence']}",
            "window starts: " + " ".join(str(int(b)) for b in result["bounds"][:-1]),
            f"miss_ratio[{base_id}]: " + _series(result["base_miss"]),
            f"miss_ratio[{identity}]: " + _series(result["cand_miss"]),
            f"only_{identity}_hits per 1k: " + _series(result["improvements"] * 1000, 1),
            f"only_{base_id}_hits per 1k: " + _series(result["regressions"] * 1000, 1),
            f"objects with most requests that only {base_id} hits: " + json.dumps(result["top_regressed"], separators=(",", ":")),
            f"objects with most requests that only {identity} hits: " + json.dumps(result["top_improved"], separators=(",", ":")),
        ]
        png = None
        if visual:
            png = views.render_diff(
                result,
                base_name=base_id,
                cand_name=identity,
                title=f"{identity} vs {base_id} on {trace}, C={capacity:,}",
                subtitle=f"requests {first:,}–{stop:,}; {self.experiment(identity).name}",
                marks=sorted({*self._marks(identity, trace, capacity), *self._marks(base_id, trace, capacity)}),
            )
        return View("\n".join(lines), png)

    def events(self, identity: str, trace: str, capacity: int, start: int | None, end: int | None, *, key: int | None = None, kinds: list[str] | None = None, limit: int = 100) -> View:
        run = self.experiment(identity).run(trace, self.capacity(trace, capacity))
        first, stop = self.window(trace, start, end)
        index = self.index(trace)
        wanted = set(kinds or ("request", "eviction", "move", "ghost"))
        unknown = wanted - {"request", "eviction", "move", "ghost"}
        if unknown:
            raise LabError(f"unknown kinds {sorted(unknown)}")
        limit = max(1, min(int(limit), MAX_EVENT_ROWS))
        queues, ghosts = run.queues, run.ghosts
        rows: list[tuple[int, int, str]] = []
        if "request" in wanted:
            times = np.arange(first, stop)
            if key is not None:
                times = times[index.keys[first:stop] == np.uint64(key)]
            for t in times[: limit * 2].tolist():
                q = run.queue[t]
                rows.append((t, 0, f"{t} request key={int(index.keys[t])} {OUTCOME_NAMES[int(run.outcome[t])]}" + (f" queue={queues[q]}" if q != NO_QUEUE else "") + (" ghost-hit" if run.ghost_hit[t] else "")))
        if "eviction" in wanted:
            mask = (run.evict_t >= first) & (run.evict_t < stop)
            if key is not None:
                mask &= run.evict_key == np.uint64(key)
            selected = np.flatnonzero(mask)[: limit * 2]
            following = index.next_after(run.evict_key[selected], run.evict_t[selected])
            for i, nxt in zip(selected.tolist(), following.tolist(), strict=True):
                t = int(run.evict_t[i])
                rows.append((t, 1, f"{t} evict key={int(run.evict_key[i])} from={queues[run.evict_q[i]]} inserted={int(run.evict_ins[i])} hits_while_resident={int(run.evict_hits[i])} next_request={'never' if nxt < 0 else nxt}"))
        names = {0: "move", 1: "ghost-add", 2: "ghost-drop", 3: "ghost-hit"}
        if wanted & {"move", "ghost"}:
            mask = (run.ev_t >= first) & (run.ev_t < stop)
            if "move" not in wanted:
                mask &= run.ev_kind != 0
            if "ghost" not in wanted:
                mask &= run.ev_kind == 0
            if key is not None:
                mask &= run.ev_key == np.uint64(key)
            for i in np.flatnonzero(mask)[: limit * 2].tolist():
                kind = int(run.ev_kind[i])
                t = int(run.ev_t[i])
                detail = f"{queues[run.ev_a[i]]}->{queues[run.ev_b[i]]}" if kind == 0 else ghosts[run.ev_a[i]]
                rows.append((t, 2, f"{t} {names[kind]} key={int(run.ev_key[i])} {detail}"))
        rows.sort()
        shown = rows[:limit]
        header = f"raw records of {identity} on {trace} C={capacity}, requests [{first}, {stop})" + (f", key={key}" if key is not None else "") + f": {len(shown)} rows" + (" (truncated; narrow the window)" if len(rows) > limit else "")
        return View("\n".join([header, *(row for _, _, row in shown)]))

    def info(self, identity: str | None = None, include_code: bool = False) -> View:
        if identity is None:
            lines = ["experiments:"]
            base = self.experiments[BASELINE]
            for experiment in self.experiments.values():
                status = "ok" if experiment.ok else "failed"
                change = ""
                if experiment.ok and experiment.id != BASELINE:
                    change = f" mean change {_pct(self.compare(experiment, base)['mean_relative_change'])}"
                start = ""
                if experiment.id != BASELINE and experiment.handover != self.handover:
                    start = f" [handover after {experiment.handover:.0%}]" if experiment.handover else " [cold start]"
                lines.append(f"- {experiment.id} [{status}] {experiment.name}{start}{change}: {experiment.hypothesis[:200]}")
            return View("\n".join(lines))
        experiment = self.experiment(identity)
        payload = {
            "id": experiment.id,
            "name": experiment.name,
            "ok": experiment.ok,
            "hypothesis": experiment.hypothesis,
            "prediction": experiment.prediction,
            "problems": experiment.problems,
            "start": f"handover after {experiment.handover:.0%} of each trace" if experiment.handover else "cold cache",
        }
        text = json.dumps(payload, indent=1)
        if experiment.ok and experiment.id != BASELINE:
            text += "\n" + self._comparison_text(self.compare(experiment, self.experiments[BASELINE]), BASELINE, experiment.id)
        if include_code:
            text += "\n--- code ---\n" + experiment.source
        return View(text)

    # -- final evaluation (host only; never exposed to the agent) -----------------------

    def final_evaluation(self, identity: str, *, prune: bool = False) -> dict[str, Any]:
        experiment = self.experiment(identity)
        heldout = materialize_suite(self.heldout_specs, self.root / "traces" / "heldout", workers=self.workers)
        splits = {"dev": list(self.dev.values()), "heldout": heldout}
        report: dict[str, Any] = {"experiment": identity, "name": experiment.name, "memory_limit_ratio": MEMORY_LIMIT_RATIO}
        for split, traces in splits.items():
            cells = [(trace, capacity) for trace in traces for capacity in capacities(trace.objects, self.fractions)]
            cand = self.store.simulate(experiment.source, cells, verified=True)
            base = self.store.simulate(BASELINE_SOURCE, cells, verified=True)
            rows, failures, memory_violations = [], [], []
            for trace, capacity in cells:
                cell = (trace.spec.name, capacity)
                c, b = cand[cell], base[cell]
                if isinstance(c, RunError) or isinstance(b, RunError):
                    failures.append({"trace": cell[0], "capacity": capacity, "error": (c if isinstance(c, RunError) else b).error})
                    continue
                ratio = self._memory_ratio(c, b)
                if ratio > MEMORY_LIMIT_RATIO:
                    memory_violations.append({"trace": cell[0], "capacity": capacity, "ratio": round(ratio, 3)})
                mb, mc = b.summary["miss_ratio"], c.summary["miss_ratio"]
                row = {"trace": cell[0], "capacity": capacity, "base": mb, "candidate": mc, "relative": (mc - mb) / mb if mb else 0.0, "memory_ratio": round(ratio, 3)}
                if split == "dev":
                    fast = experiment.cells.get(cell)
                    row["fast_matches_verified"] = isinstance(fast, Run) and fast.summary["misses"] == c.summary["misses"]
                rows.append(row)
            if prune and experiment.source != BASELINE_SOURCE:
                _prune(cand.values())
            relative = [row["relative"] for row in rows]
            report[split] = {
                "cells": len(cells),
                "scored": len(rows),
                "mean_relative_change": float(np.mean(relative)) if rows and not failures else None,
                "worst_relative_change": max(relative) if relative else None,
                "better": sum(r < -SAME for r in relative),
                "worse": sum(r > SAME for r in relative),
                "failures": failures,
                "memory_violations": memory_violations,
                "rows": rows,
            }
        report["fast_verified_mismatches"] = [
            {"trace": row["trace"], "capacity": row["capacity"]} for row in report["dev"]["rows"] if not row["fast_matches_verified"]
        ]
        report["valid"] = not (
            report["dev"]["failures"] or report["heldout"]["failures"] or report["dev"]["memory_violations"]
            or report["heldout"]["memory_violations"] or report["fast_verified_mismatches"]
        )
        report["score"] = report["heldout"]["mean_relative_change"] if report["valid"] else None
        return report
