"""The cache-policy lab: experiments, views, and the final evaluation.

The same Lab backs the VISTA session (the agent), the command line (people)
and the parameter-search baseline, so every arm sees identical numbers.

The objective is the mean relative miss-ratio change against S3-FIFO over
every (trace, cache size) cell; negative is better. Final scores come from
verified simulation on held-out traces the agent never sees, with a bound on
policy metadata so that unbounded history does not count as an improvement.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import re
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
TUNED_REFERENCE = "s3fifo-tuned"  # the prefix-tuned baseline, recorded by the host as a reference
REFERENCE_NAMES = {
    "s4fifo": "S4-FIFO heuristic, default knobs (OSDI 2026)",
    "mq": "MQ, Multi-Queue (Zhou et al., USENIX ATC 2001)",
    "lru": "LRU",
    "fifo": "FIFO",
    "arc": "ARC (Megiddo and Modha, FAST 2003)",
    "lirs": "LIRS (Jiang and Zhang, SIGMETRICS 2002)",
    "twoq": "2Q (Johnson and Shasha, VLDB 1994)",
    "wtinylfu": "W-TinyLFU (Einziger et al., ACM TOS 2017; Caffeine)",
    "clock": "CLOCK",
    "sieve": "SIEVE (Zhang et al., NSDI 2024)",
}
SAME = 0.001  # |relative change| below 0.1% counts as unchanged
MEMORY_LIMIT_RATIO = 2.0
MAX_EVENT_ROWS = 200
OUTCOME_NAMES = {0: "hit", 1: "miss-cold", 2: "miss", 3: "miss-bypass"}


MAX_TUNE_VARIANTS = 12
_KNOB = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _trusted_constants() -> set[str]:
    """UPPER_CASE names in the policy modules a candidate may build on."""
    names: set[str] = set()
    for path in SANDBOX_DIR.glob("*.py"):
        if path.stem not in ("simulate", "policy_server"):
            names.update(re.findall(r"\b[A-Z][A-Z0-9_]*\b", path.read_text(encoding="utf-8")))
    return names


def tuned_variants(code: str, tune: dict[str, list[Any]]) -> list[tuple[dict[str, Any], str]]:
    """(assignment, source) for every combination of `tune`, in grid order: the
    module followed by a subclass of CachePolicy that sets the constants."""
    if not isinstance(tune, dict) or not 1 <= len(tune) <= 3:
        raise LabError('tune takes 1 to 3 constants, e.g. {"SMALL_RATIO": [0.05, 0.1, 0.2]}')
    known = _trusted_constants() | set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", code))
    for knob, values in tune.items():
        if not _KNOB.match(knob):
            raise LabError(f"tune: {knob!r} is not an UPPER_CASE class constant")
        if knob not in known:
            raise LabError(f"tune: {knob} appears neither in your code nor in the modules it can build on, so setting it would change nothing")
        if not isinstance(values, list) or not 2 <= len(values) <= 8 or len(set(map(repr, values))) != len(values):
            raise LabError(f"tune: give {knob} 2 to 8 distinct values")
        if not all(isinstance(value, (int, float, bool)) and math.isfinite(value) for value in values):
            raise LabError(f"tune: values of {knob} must be numbers or booleans")
    count = math.prod(len(values) for values in tune.values())
    if count > MAX_TUNE_VARIANTS:
        raise LabError(f"tune: {count} combinations; at most {MAX_TUNE_VARIANTS}")
    variants = []
    for values in itertools.product(*tune.values()):
        assignment = dict(zip(tune, values, strict=True))
        body = "\n".join(f"    {knob} = {value!r}" for knob, value in assignment.items())
        variants.append((assignment, f"{code.rstrip()}\n\n\nclass TunedCachePolicy(CachePolicy):\n{body}\n\n\nCachePolicy = TunedCachePolicy\n"))
    return variants


class LabError(ValueError):
    """A request the agent can fix (unknown id, bad window, ...)."""


REPEAT = 1e-4  # every cell's misses within 0.01% of an earlier experiment's: the same behavior


def program_key(code: str) -> str:
    """The program without comments, docstrings or formatting, so equivalent edits compare equal."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code.strip()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


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
    more: tuple[bytes, ...] = ()  # further images, shown after `png`


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
        # The root also as resolved (a symlinked runs directory shows up resolved in
        # host-side tracebacks) and the simulator's install directory.
        hidden = {str(self.root), str(self.root.resolve()), str(SANDBOX_DIR), str(SANDBOX_DIR.resolve())}
        self._redactions = sorted({*hidden, *(r for r in redactions if r)}, key=len, reverse=True)
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
        self._opt: dict[tuple[str, int], dict[str, Any]] = {}
        self._comparisons: dict[str, dict[str, Any]] = {}
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
        self,
        name: str,
        code: str,
        hypothesis: str,
        prediction: dict[str, Any] | None,
        *,
        handover: float | None | str = "default",
        tune: dict[str, list[Any]] | None = None,
    ) -> View:
        """`handover`: the share of each trace S3-FIFO serves before this policy
        takes over (0 or None: cold start); by default the lab's setting.
        `tune`: class constants of CachePolicy and values to try; every
        combination is simulated and the best one (after the handover) is
        recorded as this one experiment."""
        if handover != "default":
            handover = float(handover) if handover else None
            if handover is not None and not 0 < handover <= 0.9:
                raise LabError("handover_at must be 0 (cold start) or a fraction up to 0.9")
        tuning = None
        if tune:
            variants = tuned_variants(code, tune)  # validates; raises LabError
            if not check_policy(code):
                code, tuning = self._tune(variants, self.handover if handover == "default" else handover)
        identity = self._next_identity()
        repeated = self.same_program(code)
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
        if tuning:
            parts.insert(1, tuning)
        if prediction and prediction.get("expected_change_pct") is not None:
            expected = float(prediction["expected_change_pct"])
            observed = comparison["mean_relative_change"] * 100
            parts.append(f"prediction check: expected {expected:+.2f}%, observed {observed:+.2f}% (error {observed - expected:+.2f} points)")
        same = self.same_behavior(experiment)
        if same is not None:
            other, gap = same
            how = "exactly the same misses" if gap == 0 else f"the same misses within {100 * gap:.3f}%"
            cause = "the same program" if other in repeated else "a different program"
            parts.append(
                f"repeat: {identity} has {how} as {other} in every cell ({cause}). "
                + ("The change did not alter any decision here; " if other == BASELINE else "")
                + "check that the changed code path runs on these traces (knobs may act lazily after a handover) before spending more experiments on it."
            )
        elif repeated:
            parts.append(f"note: {identity} is the same program as {', '.join(repeated)} (comments and formatting ignored).")
        if warnings:
            parts.append("instrumentation warnings (views of this experiment may be wrong until fixed):\n- " + "\n- ".join(warnings))
        overview = self.overview(identity, BASELINE)
        return View(
            "\n\n".join(parts) + "\n\n" + overview.text,
            overview.png,
            {"experiment": identity, "ok": True, "mean_relative_change": comparison["mean_relative_change"], "same_as": same[0] if same else None},
        )

    # -- progress dashboard (the attention anchor) ------------------------------------

    def _comparison(self, identity: str) -> dict[str, Any]:
        if identity not in self._comparisons:
            self._comparisons[identity] = self.compare(self.experiments[identity], self.experiments[BASELINE])
        return self._comparisons[identity]

    def mean_change(self, identity: str) -> float | None:
        """Mean change vs S3-FIFO over the observed cells (after the takeover, if any)."""
        if identity == BASELINE:
            return 0.0
        experiment = self.experiments.get(identity)
        if experiment is None or not experiment.ok:
            return None
        changes = self._cell_changes(self._comparison(identity))
        return float(np.mean(changes)) if changes else None

    def best_reference(self, exclude: str = BASELINE) -> tuple[str, float] | None:
        """The reference policy with the lowest mean change vs S3-FIFO on the observed part."""
        best = None
        for name in (*self.references, *sorted(self.recorded)):
            mean = None if name == exclude else self.mean_change(name)
            if mean is not None and (best is None or mean < best[1]):
                best = (name, mean)
        return best

    def _cell_changes(self, comparison: dict[str, Any]) -> list[float]:
        return [row.get("after_relative", row["relative"]) for row in comparison["rows"]]

    def worst_windows(self, identity: str, count: int = 5, windows: int = 16) -> list[dict[str, Any]]:
        """The (cell, window) pairs where an experiment loses most to S3-FIFO, after its takeover."""
        experiment, base = self.experiments[identity], self.experiments[BASELINE]
        items = []
        for trace, capacity in self.cells:
            name = trace.spec.name
            cand, ref = experiment.cells.get((name, capacity)), base.cells.get((name, capacity))
            if not isinstance(cand, Run) or not isinstance(ref, Run):
                continue
            start = handover_index(trace.requests, experiment.handover) if experiment.handover else 0
            bounds = analysis.edges(start, trace.requests, windows)
            try:
                mine, theirs = analysis.miss_series(cand, bounds), analysis.miss_series(ref, bounds)
            except FileNotFoundError:
                continue
            for i in range(len(bounds) - 1):
                if theirs[i] > 0:
                    items.append(
                        {"trace": name, "capacity": capacity, "start": int(bounds[i]), "end": int(bounds[i + 1]), "relative": float((mine[i] - theirs[i]) / theirs[i]), "s3fifo": float(theirs[i]), "candidate": float(mine[i])}
                    )
        items.sort(key=lambda item: -item["relative"])
        return [item for item in items[:count] if item["relative"] > 0]

    def dashboard(self, latest: str, used: int, budget: int, *, visual: bool = True) -> View:
        """Where the search stands: every experiment against S3-FIFO per cell, the best so far, and
        where the latest experiment loses most. Returned with every experiment."""
        references = [name for name in (*self.references, *sorted(self.recorded)) if self.experiments[name].ok]
        mine = [identity for identity in self.experiments if identity != BASELINE and identity not in references and not self.is_reference(identity) and self.experiments[identity].ok]
        rows = []
        for identity in [*references, *mine]:
            comparison = self._comparison(identity)
            changes = self._cell_changes(comparison)
            rows.append({"id": identity, "name": self.experiments[identity].name, "reference": identity in references, "mean": float(np.mean(changes)) if changes else math.nan, "cells": changes})
        columns = [f"{row['trace']} C={row['capacity']}" for row in self._comparison(BASELINE)["rows"]] if rows else []
        ours = [row for row in rows if not row["reference"]]
        best = min(ours, key=lambda row: row["mean"]) if ours else None
        worst = self.worst_windows(latest) if latest in self.experiments and self.experiments[latest].ok else []
        scope = "after the handover point" if self.handover else "over each trace"
        lines = [f"progress after {latest} ({used} of {budget} experiments used); change vs s3fifo {scope}, per cell:"]
        if best is not None:
            lines.append(f"best so far: {best['id']} ({best['name'][:60]}) mean {_pct(best['mean'])}")
        lines.append("experiment  mean     " + "  ".join(columns))
        shown = [row for row in rows if row["reference"]] + ours[-12:]
        if best is not None and best not in shown:
            shown.append(best)
        for row in shown:
            lines.append(f"{row['id']:<10}  {_pct(row['mean'])}  " + "  ".join(_pct(value) for value in row["cells"]))
        if worst:
            lines.append(f"where {latest} loses most (16 windows per cell, change vs s3fifo in that window):")
            lines += [f"- {item['trace']} C={item['capacity']} requests {item['start']:,}-{item['end']:,}: {_pct(item['relative'])} ({item['s3fifo']:.4f} -> {item['candidate']:.4f})" for item in worst]
        png = None
        if visual and rows:
            png = views.render_dashboard(rows, columns, latest=latest, best=best["id"] if best else None, title=f"Progress after {latest}: {used} of {budget} experiments", subtitle=f"change in miss ratio vs S3-FIFO {scope}; blue = fewer misses")
        return View("\n".join(lines), png, {"best": best, "worst_windows": worst})

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

    def _tune(self, variants: list[tuple[dict[str, Any], str]], handover: float | None) -> tuple[str, str]:
        """Simulate every variant on every cell; return the best source and a summary."""
        base = self.experiments[BASELINE]
        runs = self.store.simulate_many([(source, trace, capacity) for _, source in variants for trace, capacity in self.cells], handover=handover)
        scored = []
        for index, (assignment, source) in enumerate(variants):
            cells = runs[index * len(self.cells) : (index + 1) * len(self.cells)]
            trial = Experiment("tune", "tune", source, "", None, {(trace.spec.name, capacity): run for (trace, capacity), run in zip(self.cells, cells, strict=True)})
            if not trial.ok:
                scored.append((math.inf, index, assignment, trial))
                continue
            comparison = self.compare(trial, base)
            value = comparison["mean_relative_change_after_handover"]
            scored.append((comparison["mean_relative_change"] if value is None else value, index, assignment, trial))
        scored.sort(key=lambda item: (item[0], item[1]))
        best = scored[0]
        for _, _, _, trial in scored[1:]:  # long traces: hundreds of MB per run
            for run in trial.cells.values():
                if isinstance(run, Run):
                    run.drop_arrays()
        shown = ", ".join(
            ("*" if item is best else "") + ",".join(f"{k}={v!r}" for k, v in item[2].items()) + (" failed" if math.isinf(item[0]) else f" {100 * item[0]:+.2f}%")
            for item in scored
        )
        text = (
            f"tuned {len(variants)} combinations of {', '.join(variants[0][0])} on the observed part (mean change vs s3fifo "
            f"after the handover; * = recorded here): {shown}. The choice is fitted to what you see; check later windows."
        )
        return best[3].source, text

    def same_program(self, code: str) -> list[str]:
        """Earlier experiments running this program (comments, docstrings and formatting ignored)."""
        key = program_key(code)
        return [e.id for e in self.experiments.values() if e.id not in self.references and program_key(e.source) == key]

    def same_behavior(self, experiment: Experiment) -> tuple[str, float] | None:
        """The earlier experiment (or reference) whose misses match this one's in every cell, closest first."""
        if not experiment.ok:
            return None
        mine = {cell: run.summary["misses"] for cell, run in experiment.cells.items()}
        found = []
        for other in self.experiments.values():
            if other.id == experiment.id or not other.ok or set(other.cells) != set(mine):
                continue
            gap = max(abs(mine[cell] - run.summary["misses"]) / max(1, run.summary["misses"]) for cell, run in other.cells.items())
            if gap <= REPEAT:
                found.append((gap, other.id))
        return (min(found)[1], min(found)[0]) if found else None

    def test_policy(self, code: str) -> View:
        """Static check plus verified runs at the smallest, a middle and the largest cache
        size with memory measured; no experiment is recorded."""
        problems = check_policy(code)
        if problems:
            return View("rejected before simulation:\n- " + "\n- ".join(problems), data={"ok": False})
        name, trace = next(iter(self.dev.items()))
        sizes = self.capacities[name]
        # Fixed overheads weigh most in small caches, so the bound is checked at both ends.
        chosen = sorted({sizes[0], sizes[len(sizes) // 2], sizes[-1]})
        cells = [(trace, capacity) for capacity in chosen]
        candidates = self.store.simulate(code, cells, verified=True, handover=self.handover)
        for capacity in chosen:
            candidate = candidates[(name, capacity)]
            if isinstance(candidate, RunError):
                return View(f"policy failed on {name} C={capacity}: {candidate.error}\n{candidate.traceback[-2500:]}", data={"ok": False})
        bases = self.store.simulate(BASELINE_SOURCE, cells, verified=True)
        ratios = {capacity: self._memory_ratio(candidates[(name, capacity)], bases[(name, capacity)]) for capacity in chosen}
        over = [capacity for capacity, ratio in ratios.items() if ratio > MEMORY_LIMIT_RATIO]
        runs = [candidates[(name, capacity)] for capacity in chosen]
        # No miss ratio: this check is free, so it must not double as an unmetered experiment.
        text = (
            f"policy ran on {name} (verified mode); policy memory vs s3fifo: "
            + ", ".join(f"C={capacity} {ratio:.2f}x ({candidates[(name, capacity)].summary['policy_memory_bytes']:,} B)" for capacity, ratio in ratios.items())
            + f"; the limit is {MEMORY_LIMIT_RATIO:.1f}x at every cache size. "
            f"instrumentation errors {sum(run.summary['instrumentation_errors'] for run in runs)}, "
            f"size mismatches {sum(run.summary['size_mismatches'] for run in runs)}."
        )
        if over:
            text += (
                f"\nover the memory limit at C={', '.join(str(capacity) for capacity in over)}: a decision with this policy would be "
                "scored invalid. Per-policy and per-object overheads weigh most in small caches."
            )
        ratio = max(ratios.values())
        same = self.same_program(code)
        if same:
            text += (
                f"\nnote: this is the same program as {', '.join(same)} (comments, docstrings and formatting ignored); "
                "its results are already known, so running it again would spend an experiment on nothing new."
            )
        return View(text, data={"ok": True, "memory_ratio": ratio, "memory_ratios": {str(c): round(r, 3) for c, r in ratios.items()}, "same_program": same})

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

    def handover_view(self, identity: str, trace: str, capacity: int, start: int | None = None, end: int | None = None, buckets: int | None = None, *, visual: bool = True) -> View:
        """The deployed takeover against the same policy from a cold cache and S3-FIFO continuing."""
        capacity = self.capacity(trace, capacity)
        experiment = self.experiment(identity)
        if not experiment.handover:
            raise LabError(f"{identity} ran from a cold cache; view_handover needs an experiment that took over at a handover point")
        warm = experiment.run(trace, capacity)
        switch = warm.summary.get("handover_at")
        if not switch:
            raise LabError(f"{identity} has no handover on {trace}")
        cell = (self.dev[trace], capacity)
        cold = self.store.simulate(experiment.source, [cell])[(trace, capacity)]
        if isinstance(cold, RunError):
            raise LabError(f"{identity} failed from a cold cache: {cold.error}")
        base = self.experiment(BASELINE).run(trace, capacity)
        n = self.dev[trace].requests
        first = max(0, switch - (n - switch) // 8) if start is None else start
        first, stop = self.window(trace, first, end)
        if not first <= switch < stop:
            raise LabError(f"the window must contain the handover at request {switch}")
        profile = analysis.handover_profile(base, warm, cold, switch, first, stop, buckets or 64)
        totals = profile["totals"]
        settled = (
            f"90% of the transition cost was paid by request {totals['settled_at']:,} ({totals['settled_at'] - switch:,} requests after the handover)"
            if totals["settled_at"] is not None
            else "the takeover cost nothing beyond the decision (warm did at least as well as cold)"
        )
        lines = [
            f"handover of {identity} on {trace} C={capacity}: S3-FIFO served requests [0, {switch}), {identity} took over its live cache; "
            f"compared with {identity} from a cold cache and S3-FIFO continuing, requests [{first}, {stop})",
            f"after the handover ({totals['requests_after']:,} requests), relative to s3fifo's {totals['base_misses_after']:,} misses: "
            f"deployed (warm) {_pct(totals['deployed_effect'])} = decision (cold) {_pct(totals['decision_effect'])} "
            f"+ transition (warm - cold) {_pct(totals['transition_cost'])}",
            settled,
            f"inherited objects evicted after the handover: {totals['inherited_evictions']:,} ({totals['drained_share_of_cache']:.0%} of the cache), "
            f"{totals['inherited_evicted_unused']:,} of them with no hit while resident",
            "window starts: " + " ".join(str(int(b)) for b in profile["bounds"][:-1]),
            "miss_ratio[s3fifo]: " + _series(profile["base"]),
            "miss_ratio[warm]: " + _series(profile["warm"]),
            "miss_ratio[cold]: " + _series(profile["cold"]),
            "inherited evictions per window: " + " ".join(str(int(v)) for v in profile["inherited_evictions"]),
            "own evictions per window: " + " ".join(str(int(v)) for v in profile["fresh_evictions"]),
        ]
        png = None
        if visual:
            png = views.render_handover(
                profile,
                name=identity,
                title=f"{identity} taking over S3-FIFO on {trace}, C={capacity:,}",
                subtitle=f"deployed {_pct(totals['deployed_effect'])} = decision {_pct(totals['decision_effect'])} + transition {_pct(totals['transition_cost'])} vs s3fifo after the handover",
            )
        return View("\n".join(lines), png, {"handover": totals})

    def opt(self, trace: str, capacity: int) -> dict[str, Any]:
        """Belady (OPT) on one cell, computed once per lab and kept on disk."""
        cache_key = (trace, capacity)
        if cache_key in self._opt:
            return self._opt[cache_key]
        path = self.root / "opt" / f"{self.dev[trace].sha256[:16]}-{capacity}.npz"
        if path.is_file():
            with np.load(path) as data:
                result = {name: data[name] for name in ("hit", "evict_t", "evict_gap")}
        else:
            result = analysis.belady(self.index(trace), capacity)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial.npz")
            np.savez_compressed(partial, **result)
            partial.replace(path)
        result["boundary"] = analysis.belady_boundary(result)
        self._opt[cache_key] = result
        return result

    def opt_view(
        self, identity: str, trace: str, capacity: int, start: int | None, end: int | None, buckets: int | None = None, compare_to: str | None = None, *, visual: bool = True
    ) -> View:
        """An experiment against Belady: avoidable misses (by cause) and good evictions."""
        capacity = self.capacity(trace, capacity)
        experiment = self.experiment(identity)
        run = experiment.run(trace, capacity)
        opt = self.opt(trace, capacity)
        first, stop = self.window(trace, start, end)
        boundary = opt["boundary"]
        profile = analysis.opt_profile(run, opt, self.index(trace), first, stop, buckets or 64, boundary)
        totals = profile["totals"]
        boundary_text = "never (OPT evicted only objects that are not requested again)" if not np.isfinite(boundary) else f"{boundary:,.0f} requests = {boundary / capacity:.2f} C"
        by_queue = ", ".join(f"{name} {count}" for name, count in totals["avoidable_by_queue"].items()) or "none"
        lines = [
            f"OPT (Belady) comparison of {identity} on {trace} C={capacity}, requests [{first}, {stop})"
            + "".join(f"; {identity} took over at request {mark}" for mark in self._marks(identity, trace, capacity)),
            f"misses: {identity} {totals['misses']} ({totals['misses'] / totals['requests']:.4f}), OPT {totals['opt_misses']} "
            f"({totals['opt_misses'] / totals['requests']:.4f}); avoidable (OPT hits, {identity} misses) {totals['avoidable']} = "
            f"{100 * totals['avoidable_share_of_misses']:.1f}% of {identity}'s misses; by the queue the key was last evicted from: {by_queue}",
            f"Belady boundary (OPT rarely evicts an object requested again sooner): {boundary_text}",
            "evictions: " + "; ".join(f"{name} {data['count']} (good share {data['good_share']:.3f})" for name, data in totals["evictions"].items()),
            "window starts: " + " ".join(str(int(b)) for b in profile["bounds"][:-1]),
            f"miss_ratio[{identity}]: " + _series(profile["miss_ratio"]),
            "miss_ratio[OPT]: " + _series(profile["opt_miss_ratio"]),
            "avoidable per 1k: " + _series(profile["avoidable"] * 1000, 1),
        ]
        lines += [f"avoidable per 1k, evicted from {name}: " + _series(series * 1000, 1) for name, series in profile["avoidable_by_queue"].items()]
        lines += [f"good eviction share[{name}]: " + _series(series) for name, series in profile["good_evictions"].items()]
        diff = None
        if compare_to:
            base = self.experiment(compare_to).run(trace, capacity)
            diff = analysis.opt_diff(base, run, opt, first, stop, buckets or 64)
            lines += [
                f"against {compare_to}: of its {diff['totals']['base_avoidable']} avoidable misses {identity} fixed {diff['totals']['fixed']} "
                f"and created {diff['totals']['created']} new ones",
                "fixed per 1k: " + _series(diff["fixed"] * 1000, 1),
                "created per 1k: " + _series(diff["created"] * 1000, 1),
            ]
        png = None
        if visual:
            png = views.render_opt(
                profile,
                name=identity,
                colors=views.queue_colors(run.queues, run.ghosts),
                title=f"{identity} vs OPT on {trace}, C={capacity:,}" + (f" (against {compare_to})" if compare_to else ""),
                subtitle=f"requests {first:,}–{stop:,}; avoidable = {100 * totals['avoidable_share_of_misses']:.1f}% of misses; {experiment.name}",
                diff=diff,
                base_name=compare_to or "",
                marks=self._marks(identity, trace, capacity),
            )
        return View("\n".join(lines), png, {"opt": totals})

    def knob_sweep(self, trace: str, capacity: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
        """Miss masks of S3-FIFO with one knob changed at a time (the values the
        prefix-tuned baseline searches), taking over at the lab's handover point
        like the agent's experiments. Simulated once per cell, kept as packed bits;
        the runs' own logs are dropped (nothing else views them)."""
        from .search import DEFAULT, GRID, policy_source  # search imports this module

        workload = self.trace(trace)
        defaults = {knob: repr(value) for knob, value in DEFAULT.items()}
        wanted = [(knob, value) for knob, values in GRID.items() for value in values if value != DEFAULT[knob]]
        n, switch = workload.requests, handover_index(workload.requests, self.handover)
        path = self.root / "knobs" / f"{workload.sha256[:16]}-{capacity}-{switch}-{self.store.fingerprint[:12]}.npz"
        names = [f"{knob}={value!r}" for knob, value in wanted]
        if path.is_file():
            with np.load(path) as data:
                packed = {name: data[name] for name in names}
        else:
            entries = [(policy_source({knob: value}), workload, capacity) for knob, value in wanted]
            runs = self.store.simulate_many(entries, handover=self.handover)
            failed = [f"{name}: {run.error}" for name, run in zip(names, runs, strict=True) if isinstance(run, RunError)]
            if failed:
                raise LabError("the knob sweep failed: " + "; ".join(failed[:3]))
            packed = {}
            for name, run in zip(names, runs, strict=True):
                packed[name] = np.packbits(run.outcome != HIT)
                run.drop_arrays()
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial.npz")
            np.savez_compressed(partial, **packed)
            partial.replace(path)
        variants: dict[str, dict[str, np.ndarray]] = {}
        for (knob, value), name in zip(wanted, names, strict=True):
            variants.setdefault(knob, {})[repr(value)] = np.unpackbits(packed[name], count=n).astype(bool)
        return variants, defaults

    def knobs_view(self, trace: str, capacity: int, start: int | None = None, end: int | None = None, buckets: int | None = None, *, visual: bool = True) -> View:
        """What each S3-FIFO knob is worth on its own, per window; free (no experiment)."""
        capacity = self.capacity(trace, capacity)
        variants, defaults = self.knob_sweep(trace, capacity)
        n = self.trace(trace).requests
        switch = handover_index(n, self.handover)
        first, stop = self.window(trace, switch if start is None else start, end)
        base = self.experiment(BASELINE).run(trace, capacity)
        profile = analysis.knob_profile(base.outcome != HIT, variants, defaults, first, stop, buckets or 16)
        setting = ", ".join(f"{knob}={value}" for knob, value in defaults.items())
        start_text = f"each variant took over S3-FIFO's live cache at request {switch:,}" if switch else "each variant ran from a cold cache"
        lines = [
            f"knob sweep of S3-FIFO on {trace} C={capacity}: one knob changed at a time from the default ({setting}); {start_text}; "
            f"requests [{first}, {stop}), {len(profile['bounds']) - 1} windows; S3-FIFO misses {profile['base_total']:,}. "
            "Changes are in misses relative to S3-FIFO (negative = fewer). 'best per window' is the value with the fewest misses in each "
            "window; switching to it every window is an upper bound for adapting that knob alone (it ignores the cost of a switch).",
            "window starts: " + " ".join(str(int(b)) for b in profile["bounds"][:-1]),
        ]
        summary = {}
        for knob, data in profile["knobs"].items():
            lines.append(
                f"{knob}: best static value {data['static_best']} ({_pct(data['static_change'])}); "
                f"best value per window {_pct(data['switching_change'])}"
            )
            for label, overall, row in zip(data["values"], data["overall"], data["relative"], strict=True):
                tag = " (default)" if label == data["default"] else ""
                lines.append(f"  {label}{tag}: overall {_pct(float(overall))}; per window % " + " ".join(f"{100 * v:+.1f}" for v in row))
            lines.append("  best per window: " + " ".join(data["best"]))
            summary[knob] = {"static_best": data["static_best"], "static_change": round(data["static_change"], 5), "switching_change": round(data["switching_change"], 5)}
        png = None
        if visual:
            png = views.render_knobs(
                profile,
                title=f"S3-FIFO knobs one at a time on {trace}, C={capacity:,}",
                subtitle=f"requests {first:,}–{stop:,}" + (f" (variants take over at {switch:,})" if switch else "") + "; change in misses vs the default; ● best value per window",
            )
        return View("\n".join(lines), png, {"knobs": summary})

    def headroom(self, trace: str, capacity: int) -> dict[str, Any]:
        """S3-FIFO replayed with a perfect predictor in charge of one decision at a
        time (headroom.py), taking over at the lab's handover point. Computed once
        per cell on the host, kept on disk as packed miss masks."""
        from . import headroom

        workload = self.trace(trace)
        n, switch = workload.requests, handover_index(workload.requests, self.handover)
        path = self.root / "headroom" / f"{workload.sha256[:16]}-{capacity}-{switch}-{self.store.fingerprint[:12]}-v{headroom.VERSION}"
        names = ["s3fifo", *headroom.DECISIONS, headroom.ALL]
        slug = {name: name.replace(" ", "_") for name in names}
        meta_path = path.with_suffix(".json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(path.with_suffix(".npz")) as data:
                masks = {name: np.unpackbits(data[slug[name]], count=n).astype(bool) for name in names}
        else:
            index = self.index(trace)
            workers = getattr(self.store.sandbox, "workers", 1) or 1
            result = headroom.headroom(index.dense, analysis.next_requests(index), capacity, switch, workers=workers)
            masks = result["masks"]
            meta = {
                "crossing": result["crossing"],
                "s_crossing": result["s_crossing"],
                "factors": result["factors"],
                "changes": {decision: {repr(factor): change for factor, change in changes.items()} for decision, changes in result["changes"].items()},
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial.npz")
            np.savez_compressed(partial, **{slug[name]: np.packbits(masks[name]) for name in names})
            partial.replace(path.with_suffix(".npz"))
            meta_path.write_text(json.dumps(meta), encoding="utf-8")  # written last: marks the cell complete
        return {"masks": masks, "switch": switch, **meta}

    HEADROOM_BASELINES = ("3L-Cache", "ARC", "LIRS", "W-TinyLFU", "SIEVE")

    def headroom_baselines(self, trace: str, capacity: int, split: int) -> dict[str, float]:
        """Real policies (libCacheSim, no future knowledge) on the observed trace only:
        change in misses after `split` relative to libCacheSim's S3-FIFO in the same
        run, so both sides share one simulator. Cached per cell."""
        from . import zoo

        algorithms = ("S3-FIFO", *self.HEADROOM_BASELINES)
        workers = getattr(self.store.sandbox, "workers", 1) or 1
        results = zoo.suffix_miss_ratios(self.index(trace).keys, split, [capacity], self.root / "zoo-observed", algorithms=algorithms, workers=min(workers, len(algorithms)))[capacity]
        base = results.get("S3-FIFO", {})
        if "error" in base or not base:
            return {}
        reference = base["misses"] - base["prefix_misses"]
        return {
            name: (result["misses"] - result["prefix_misses"] - reference) / reference
            for name, result in results.items()
            if name != "S3-FIFO" and "error" not in result and reference
        }

    def headroom_view(self, trace: str, capacity: int, start: int | None = None, end: int | None = None, buckets: int | None = None, *, visual: bool = True) -> View:
        """Which of S3-FIFO's decisions is worth predicting better on one cell; free (no experiment)."""
        from . import headroom

        capacity = self.capacity(trace, capacity)
        result = self.headroom(trace, capacity)
        switch, crossing, factors = result["switch"], result["crossing"], result["factors"]
        first, stop = self.window(trace, switch if start is None else start, end)
        opt_miss = ~self.opt(trace, capacity)["hit"]
        profile = headroom.profile(result["masks"], opt_miss, first, stop, buckets or 16)
        rows = profile["rows"]
        # Real policies only count misses after a split, so they need a window that runs to the end.
        baselines = self.headroom_baselines(trace, capacity, first) if stop == self.trace(trace).requests else {}
        real = {}
        if "3L-Cache" in baselines:
            real["3L-Cache"] = baselines["3L-Cache"]
        others = {name: change for name, change in baselines.items() if name != "3L-Cache"}
        if others:
            name = min(others, key=others.get)
            real[f"best other ({name})"] = others[name]
        windows = len(profile["bounds"]) - 1
        for name, change in real.items():
            rows[name] = {"misses": None, "relative": np.full(windows, np.nan), "overall": change, "real": True}
        start_text = f"the predictor takes over S3-FIFO's live cache at request {switch:,}, as experiments do" if switch else "the predictor runs from a cold cache"
        s_text = "" if result["s_crossing"] is None else f" (crossing S takes {result['s_crossing']:,.0f})"
        rules = {
            "promotion": "at the tail of S, move to M exactly the objects requested again within the horizon",
            "reinsertion": "at the tail of M, reinsert exactly the objects requested again within the horizon",
            "ghost admission": "a ghost hit enters M if it is requested again within the horizon, else S",
            headroom.ALL: "all three at once, each at its best horizon",
            "OPT": "Belady, for scale: from a cold cache, every decision perfect",
            **{name: "a real policy without future knowledge (libCacheSim, observed part only, vs libCacheSim's S3-FIFO); whole window only" for name in real},
        }
        lines = [
            f"headroom of S3-FIFO's decisions on {trace} C={capacity}: a perfect predictor replaces one decision, everything else "
            f"stays S3-FIFO; {start_text}; requests [{first}, {stop}), {len(profile['bounds']) - 1} windows; S3-FIFO misses {profile['base_total']:,}.",
            f"The predictor knows whether an object is requested again within a horizon, a multiple of the {crossing:,.0f} requests "
            f"({crossing / capacity:.2f} C) an object takes to cross M in S3-FIFO here{s_text}. Each decision keeps the multiple that "
            "works best for it; all multiples tried are listed (after the handover).",
            "Changes are in misses relative to S3-FIFO (negative = fewer). A decision with a large change is worth a better "
            "predictor (a mechanism that guesses that object's future better); one near 0 is not worth an experiment. "
            "The oracle is a guide, not a bound: a policy can also change the queues themselves. Single decisions overlap, so their "
            "changes do not add up; 'all three' is the joint figure. The real policies at the end show how much of this headroom "
            "known algorithms capture on the same requests"
            + (
                ". They start from a cold cache at the beginning of the observed part, so a learned policy such as 3L-Cache, "
                "which needs a long warm-up, looks weaker here than it is over a whole workload."
                if real
                else " (shown only for a window that runs to the end of the trace)."
            ),
            "window starts: " + " ".join(str(int(b)) for b in profile["bounds"][:-1]),
        ]
        for name, row in rows.items():
            horizon = ""
            if name in factors:
                horizon = f", horizon {factors[name]:g}x crossing = {factors[name] * crossing:,.0f} requests; by multiple: " + ", ".join(
                    f"{float(factor):g}x {_pct(change)}" for factor, change in result["changes"][name].items()
                )
            per_window = "" if row.get("real") else "; per window % " + " ".join(f"{100 * v:+.1f}" for v in row["relative"])
            lines.append(f"{name} ({rules[name]}): overall {_pct(row['overall'])}{horizon}{per_window}")
        png = None
        if visual:
            png = views.render_headroom(
                profile,
                title=f"Headroom of S3-FIFO's decisions on {trace}, C={capacity:,}",
                subtitle=f"requests {first:,}–{stop:,}" + (f" (the predictor takes over at {switch:,})" if switch else "")
                + "; perfect predictor of 'requested again within the horizon?', best horizon per decision",
            )
        summary = {name: round(row["overall"], 5) for name, row in rows.items()}
        return View("\n".join(lines), png, {"headroom": summary, "factors": factors, "crossing": round(crossing, 1)})

    MRC_SCALES = (0.125, 0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0, 8.0)  # cache sizes, in multiples of C
    MRC_ALGORITHMS = ("S3-FIFO", "LRU", "Belady")

    def mrc(self, trace: str, capacity: int) -> dict[str, Any]:
        """Miss-ratio curves around one cell: libCacheSim's S3-FIFO (the same algorithm as
        ours, miss for miss), LRU and Belady at MRC_SCALES x C on the observed trace, from
        a cold cache, counting misses after the handover point (the part experiments
        change). Exact, not sampled; cached per size with the other zoo runs."""
        from . import zoo

        workload, index = self.trace(trace), self.index(trace)
        n, split = workload.requests, handover_index(workload.requests, self.handover)
        objects = int(index.unique.size)
        sizes = sorted({min(objects, max(1, round(scale * capacity))) for scale in self.MRC_SCALES} | {capacity})
        workers = getattr(self.store.sandbox, "workers", 1) or 1
        try:
            results = zoo.suffix_miss_ratios(index.keys, split, sizes, self.root / "zoo-observed", algorithms=self.MRC_ALGORITHMS, workers=min(workers, len(sizes) * len(self.MRC_ALGORITHMS)))
        except ImportError as exc:
            raise LabError("view_mrc needs libCacheSim, which this lab does not have") from exc
        curves = {}
        for algorithm in self.MRC_ALGORITHMS:
            failed = [results[size][algorithm]["error"] for size in sizes if "error" in results[size][algorithm]]
            if failed:
                raise LabError(f"the {algorithm} curve failed: {failed[0]}")
            curves[algorithm] = np.array([results[size][algorithm]["misses"] - results[size][algorithm]["prefix_misses"] for size in sizes], dtype=np.int64)
        return {"sizes": np.array(sizes, dtype=np.int64), "split": split, "scored": n - split, "objects": objects, "curves": curves}

    @staticmethod
    def equivalent_size(sizes: np.ndarray, misses: np.ndarray, target: float) -> float | None:
        """The cache size at which a curve reaches `target` misses, interpolated linearly
        in log size on the curve's non-increasing envelope; None outside the sizes shown."""
        envelope = np.minimum.accumulate(misses.astype(float))
        if target > envelope[0] or target < envelope[-1]:
            return None
        i = int(np.argmax(envelope <= target))
        if i == 0 or envelope[i] == target:
            return float(sizes[i])
        hi, lo = envelope[i - 1], envelope[i]
        share = (hi - target) / (hi - lo)
        return float(np.exp(np.log(sizes[i - 1]) + share * (np.log(sizes[i]) - np.log(sizes[i - 1]))))

    def mrc_view(self, trace: str, capacity: int, experiment: str | None = None, *, visual: bool = True) -> View:
        """How much cache a policy is worth on one cell: S3-FIFO's, LRU's and OPT's
        miss-ratio curves, and the size at which S3-FIFO would match a policy; free."""
        capacity = self.capacity(trace, capacity)
        data = self.mrc(trace, capacity)
        sizes, split, scored, curves = data["sizes"], data["split"], data["scored"], data["curves"]
        at = int(np.flatnonzero(sizes == capacity)[0])
        s3 = curves["S3-FIFO"]
        def worth(misses: float) -> str:
            size = self.equivalent_size(sizes, s3, misses)
            if size is None:
                return f"outside {sizes[0] / capacity:g}-{sizes[-1] / capacity:g} C" if misses > 0 else "no misses"
            return f"{size / capacity:.2f} C ({size:,.0f} objects)"
        points: dict[str, dict[str, Any]] = {}
        for algorithm, label in (("LRU", "LRU"), ("Belady", "OPT")):
            misses = int(curves[algorithm][at])
            points[label] = {"misses": misses, "change": (misses - int(s3[at])) / int(s3[at]) if s3[at] else 0.0, "equivalent": self.equivalent_size(sizes, s3, misses), "worth": worth(misses)}
        if experiment is not None:
            run = self.experiment(experiment).run(trace, capacity)
            misses = int((run.outcome[split:] != HIT).sum())
            points[experiment] = {"misses": misses, "change": (misses - int(s3[at])) / int(s3[at]) if s3[at] else 0.0, "equivalent": self.equivalent_size(sizes, s3, misses), "worth": worth(misses)}
        base = self.experiment(BASELINE).run(trace, capacity)
        own = int((base.outcome[split:] != HIT).sum())
        ratio = {algorithm: values / max(1, scored) for algorithm, values in curves.items()}
        drops = [(float(s3[i] - s3[i + 1]) / s3[i] if s3[i] else 0.0, i) for i in range(len(sizes) - 1)]
        steepest, where = max(drops)
        def scaled(scale: float) -> int | None:
            hit = np.flatnonzero(sizes == max(1, round(scale * capacity)))
            return int(hit[0]) if hit.size else None
        half, double = scaled(0.5), scaled(2.0)
        start_text = f"after request {split:,} (where experiments take over)" if split else "over the whole trace"
        lines = [
            f"miss-ratio curves on {trace} around C={capacity}: S3-FIFO, LRU and OPT (Belady, which knows the future) at "
            f"{len(sizes)} cache sizes from {sizes[0] / capacity:g} C to {sizes[-1] / capacity:g} C, each from a cold cache over the "
            f"observed trace, counting the {scored:,} requests {start_text}; {data['objects']:,} distinct objects. Exact runs "
            "(libCacheSim; its S3-FIFO is the same algorithm as the lab's).",
            "An equivalent size says how much cache a policy is worth: the size at which S3-FIFO would miss as often "
            "(interpolated in log size). A policy worth 1.3 C does what 30% more cache would do for S3-FIFO; OPT's equivalent "
            "size bounds what any policy could be worth here. Where the S3-FIFO curve is flat, extra cache (and a better "
            "policy) buys little; a steep drop between two sizes is a working set just larger than the smaller one.",
            "size (x C) / objects / miss ratio S3-FIFO / LRU / OPT:",
        ]
        for i, size in enumerate(sizes.tolist()):
            lines.append(f"  {size / capacity:6.3g} C {size:>10,}  {ratio['S3-FIFO'][i]:.4f}  {ratio['LRU'][i]:.4f}  {ratio['Belady'][i]:.4f}")
        if double is not None or half is not None:
            parts = []
            if half is not None and s3[at]:
                parts.append(f"half the cache {_pct((int(s3[half]) - int(s3[at])) / int(s3[at]))}")
            if double is not None and s3[at]:
                parts.append(f"twice the cache {_pct((int(s3[double]) - int(s3[at])) / int(s3[at]))}")
            lines.append("S3-FIFO misses vs C: " + ", ".join(parts))
        lines.append(f"steepest S3-FIFO drop: {_pct(-steepest)} from {sizes[where] / capacity:g} C to {sizes[where + 1] / capacity:g} C")
        for label, point in points.items():
            lines.append(f"{label} at C: {point['misses']:,} misses, {_pct(point['change'])} vs S3-FIFO; equivalent S3-FIFO size {point['worth']}")
        if own != int(s3[at]):
            lines.append(f"note: the lab's S3-FIFO misses {own:,} at C after the handover, libCacheSim's {int(s3[at]):,} from a cold cache")
        png = None
        if visual:
            marks = {label: (point["misses"] / max(1, scored), point["equivalent"]) for label, point in points.items() if label not in ("LRU", "OPT")}
            png = views.render_mrc(
                {"sizes": sizes, "capacity": capacity, "ratio": ratio, "marks": marks, "opt_equivalent": points["OPT"]["equivalent"]},
                title=f"Miss-ratio curves on {trace} around C={capacity:,}",
                subtitle=f"misses on requests {split:,}–{split + scored:,} from a cold cache; equivalent size = where S3-FIFO's curve reaches a policy's misses",
            )
        summary = {
            "sizes": sizes.tolist(),
            "miss_ratio": {algorithm: [round(float(v), 6) for v in values] for algorithm, values in ratio.items()},
            "equivalent": {label: (None if point["equivalent"] is None else round(point["equivalent"] / capacity, 4)) for label, point in points.items()},
        }
        return View("\n".join(lines), png, {"mrc": summary})

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
