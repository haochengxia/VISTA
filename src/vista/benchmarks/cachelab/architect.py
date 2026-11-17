"""Architect role: agents design families of virtual-queue FIFO caches, evaluated fast.

A design is a vq.py configuration (FIFO stages with their own counting, promotion and
demotion rules; see sandbox/vq.py). A *family* is a design with value ranges for any of its
fields; its designs are the combinations. Families are evaluated on a suite of long
training workloads ("cells": a trace at a cache size) by vqsim (scripts/vqsim), which
reproduces vq.py exactly at C speed, under the deployment protocol:

- every design of the family runs on the observed 20% of the workload (from a cold cache);
- per cell, the design with the fewest misses on that part (after its first quarter, the
  warm-up) is chosen;
- the chosen design takes over S3-FIFO's live cache at the 20% point, and is scored on the
  remaining 80% against S3-FIFO and against 3L-Cache.

Architects share a board (sqlite): every family evaluated, its hypothesis and outcome, and
findings. The board's pool (every design on it) is itself a family; a new family's
contribution is how much adding it to the pool improves the pool's per-cell choice.
Cells are anonymous (c001...): no trace names or sources reach the agents.

A board with the online objective scores the deployed selector instead (sandbox/vq.py's
OnlineVQ, idealized): S3-FIFO serves the first 20%, then the live cache runs, at each of the
40 windows, the candidate with the fewest misses so far (from window 2) in its shadow run.
Every design gets a full run from a cold cache; the pool's result is that of a greedy
portfolio of at most PORTFOLIO designs (what fits the selector's memory), and a family's own
result is the online choice among its designs.
"""

from __future__ import annotations

import fcntl
import itertools
import json
import os
import sqlite3
import statistics
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vista.core.contracts import (
    Capability,
    Effect,
    ObservationBundle,
    PublicTask,
    SubmissionSpec,
    ToolBinding,
    ToolReply,
    ToolSpec,
)

ONUP = ("reset", "keep", "dec", "half")
MAX_DESIGNS = 48
MAX_STAGES = 6
SEEN_FROM, SEEN_WINDOWS = 2, 8  # choose on windows 2..7 of the observed part (skip the warm-up)
WINDOWS, LEADER_FROM, FIRST_SCORED = 40, 2, 8  # online: 40 windows, misses counted from window 2, scored from window 8 (the split)
PORTFOLIO = 32  # online objective: candidates the deployed selector runs (mini-sims within the memory limit)
OBJECTIVES = ("static", "online")
MEMORY_BUDGET = 1.8  # estimated memory vs S3-FIFO (vq.py on CPython), the limit is 2x
S3_CONFIG = "maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
STAGE_KEYS = ("name", "size", "counts", "promote", "up", "onup", "down", "ghost", "delay", "age")
TOP_KEYS = ("name", "stages", "entry", "overflow", "ghost_entry", "ghost_freq", "ghost_ratio", "max_freq", "adopt_small", "adopt_main")


class FamilyError(ValueError):
    pass


# -- designs and families -------------------------------------------------------------


def memory_estimate(ghost_ratio: float, timed: bool) -> float:
    """vq.py's memory relative to S3-FIFO, fitted on CPython (C=5000 and 20000): residents
    about 0.46, each unit of ghost ratio 0.6, and per-object hit times (delay/age) 0.5."""
    return 0.46 + 0.6 * ghost_ratio + (0.5 if timed else 0.0)


def design_config(design: dict[str, Any]) -> str:
    """A single design (no ranges left) as a vqsim config string; raises FamilyError."""
    stages = design.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
        raise FamilyError(f"stages: 1 to {MAX_STAGES} stages")
    names = [stage.get("name") for stage in stages]
    if not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise FamilyError("every stage needs a distinct name")
    index = {name: i for i, name in enumerate(names)}

    def ref(value, what, allow_none=False):
        if value is None and allow_none:
            return -1
        if value not in index:
            raise FamilyError(f"{what}: {value!r} is not a stage")
        return index[value]

    parts, used, timed = [], 0.0, False
    for i, stage in enumerate(stages):
        unknown = set(stage) - set(STAGE_KEYS)
        if unknown:
            raise FamilyError(f"stage {names[i]}: unknown fields {sorted(unknown)}")
        size = float(stage.get("size", 0.0))
        if i < len(stages) - 1:
            if not 0.0 < size < 1.0:
                raise FamilyError(f"stage {names[i]}: size must be in (0, 1); the last stage takes the rest")
            used += size
        counts = bool(stage.get("counts", True))
        promote = int(stage.get("promote", 1))
        up = ref(stage.get("up", names[i]), f"stage {names[i]} up")
        onup = stage.get("onup", "reset")
        if onup not in ONUP:
            raise FamilyError(f"stage {names[i]}: onup must be one of {ONUP}")
        down = ref(stage.get("down"), f"stage {names[i]} down", allow_none=True)
        ghost = bool(stage.get("ghost", False))
        delay, age = float(stage.get("delay", 0.0)), float(stage.get("age", 0.0))
        if not 0 <= promote <= 15 or not 0 <= delay <= 1 or not 0 <= age <= 8:
            raise FamilyError(f"stage {names[i]}: promote 0..15, delay 0..1, age 0..8")
        if up == i and onup == "keep" and promote > 0:
            raise FamilyError(f"stage {names[i]}: a reinsertion must lower the counter (onup != keep)")
        timed = timed or delay > 0 or age > 0
        parts.append(f"st={size if i < len(stages) - 1 else 0},{int(counts)},{promote},{up},{ONUP.index(onup)},{down},{int(ghost)},{delay},{age}")
    if used > 0.9:
        raise FamilyError("the stages before the last take more than 90% of the cache")
    trap = free_cycle(
        [int(stage.get("promote", 1)) for stage in stages],
        [ref(stage.get("up", names[i]), "up") for i, stage in enumerate(stages)],
        [stage.get("onup", "reset") for stage in stages],
        [ref(stage.get("down"), "down", allow_none=True) for stage in stages],
    )
    if trap:
        raise FamilyError(
            f"objects can circle {' -> '.join(names[i] for i in trap)} without a hit (demotions, passes with promote 0, or "
            "promotions that keep the counter), so nothing would leave those stages; every such path must end in a stage "
            "whose unpromoted objects leave the cache (down: null)"
        )
    max_freq = int(design.get("max_freq", 3))
    ghost_ratio = float(design.get("ghost_ratio", 0.9))
    ghost_freq = int(design.get("ghost_freq", 0))
    if not 1 <= max_freq <= 15 or not 0 <= ghost_freq <= max_freq or not 0 <= ghost_ratio <= 3:
        raise FamilyError("max_freq 1..15, ghost_freq 0..max_freq, ghost_ratio 0..3")
    memory = memory_estimate(ghost_ratio, timed)
    if memory > MEMORY_BUDGET:
        raise FamilyError(f"estimated memory {memory:.2f}x S3-FIFO is over {MEMORY_BUDGET}x (ghost_ratio {ghost_ratio}{', with delay/age' if timed else ''})")
    last = names[-1]
    head = [
        f"maxf={max_freq}",
        f"entry={ref(design.get('entry', names[0]), 'entry')}",
        f"overflow={ref(design.get('overflow', last), 'overflow')}",
        f"ghost={ghost_ratio}",
        f"gentry={ref(design.get('ghost_entry', last), 'ghost_entry')}",
        f"gfreq={ghost_freq}",
        f"asmall={ref(design.get('adopt_small', design.get('entry', names[0])), 'adopt_small')}",
        f"amain={ref(design.get('adopt_main', last), 'adopt_main')}",
    ]
    return " ".join(head + parts)


def free_cycle(promote: list[int], up: list[int], onup: list[str], down: list[int]) -> list[int] | None:
    """A cycle of moves that need no hit and do not lower the counter (a demotion, a pass with
    promote 0, a promotion that keeps the counter), or None. Without one, every eviction scan
    ends: each other move lowers the counter, so an object makes finitely many."""
    edges = {i: set() for i in range(len(promote))}
    for i in edges:
        if down[i] >= 0:
            edges[i].add(down[i])
        if up[i] != i and (promote[i] == 0 or onup[i] == "keep"):
            edges[i].add(up[i])
    state: dict[int, int] = {}  # 1 on the path, 2 done

    def visit(i: int, path: list[int]) -> list[int] | None:
        state[i] = 1
        for j in sorted(edges[i]):
            if state.get(j) == 1:
                return path[path.index(j) :] + [j] if j in path else [i, j]
            if j not in state:
                found = visit(j, path + [j])
                if found:
                    return found
        state[i] = 2
        return None

    for i in edges:
        if i not in state:
            found = visit(i, [i])
            if found:
                return found
    return None


def expand(family: dict[str, Any]) -> list[tuple[str, str]]:
    """The designs of a family as (label, config): every list-valued field (top level or in a
    stage, except stages and names) is a range; the designs are all combinations."""
    if not isinstance(family, dict):
        raise FamilyError("a family is an object")
    unknown = set(family) - set(TOP_KEYS)
    if unknown:
        raise FamilyError(f"unknown fields {sorted(unknown)}")
    stages = family.get("stages")
    if not isinstance(stages, list) or not all(isinstance(stage, dict) for stage in stages):
        raise FamilyError("stages: a list of stage objects")
    axes: list[tuple[tuple, list]] = []
    for key in TOP_KEYS:
        if key not in ("name", "stages") and isinstance(family.get(key), list):
            axes.append(((key,), family[key]))
    for i, stage in enumerate(stages):
        for key in STAGE_KEYS:
            if key != "name" and isinstance(stage.get(key), list):
                axes.append(((i, key), stage[key]))
    if any(not values for _, values in axes):
        raise FamilyError("a range needs at least one value")
    total = 1
    for _, values in axes:
        total *= len(values)
    if total > MAX_DESIGNS:
        raise FamilyError(f"{total} combinations; at most {MAX_DESIGNS} per family (narrow the ranges or split the family)")
    designs, seen = [], set()
    for combo in itertools.product(*(values for _, values in axes)):
        design = {**family, "stages": [dict(stage) for stage in stages]}
        label = []
        for (path, _), value in zip(axes, combo, strict=True):
            if len(path) == 1:
                design[path[0]] = value
                label.append(f"{path[0]}={value}")
            else:
                design["stages"][path[0]][path[1]] = value
                label.append(f"{stages[path[0]].get('name')}.{path[1]}={value}")
        config = design_config(design)
        if config not in seen:
            seen.add(config)
            designs.append((" ".join(label) or "(single design)", config))
    return designs


# -- the suite of cells ---------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    id: str  # anonymous: c001 ...
    directory: Path
    capacity: int
    requests: int
    split: int
    objects: int
    s3fifo: float  # miss ratios on the unseen part
    l3: float
    best: float

    @property
    def size_pct(self) -> float:
        return 100.0 * self.capacity / max(1, self.objects)

    def change(self, ratio: float) -> float:
        return (ratio - self.s3fifo) / self.s3fifo if self.s3fifo else 0.0


def load_cells(suite: Path) -> list[Cell]:
    """A vq suite: one directory per trace with ids.u32 (dense object ids) and meta.json
    (requests, split, capacities, and the unseen-part miss ratios of s3fifo, 3L, best)."""
    cells = []
    for directory in sorted(path for path in suite.iterdir() if (path / "meta.json").is_file()):
        meta = json.loads((directory / "meta.json").read_text())
        objects = meta.get("objects")
        if objects is None:
            import numpy as np

            objects = int(np.memmap(directory / "ids.u32", dtype=np.uint32, mode="r").max()) + 1
            meta["objects"] = objects
            (directory / "meta.json").write_text(json.dumps(meta))
        for capacity in meta["capacities"]:
            key = str(capacity)
            best = meta["best"][key]
            cells.append(
                Cell(
                    f"c{len(cells) + 1:03d}",
                    directory,
                    int(capacity),
                    int(meta["requests"]),
                    int(meta["split"]),
                    int(objects),
                    float(meta["s3fifo"][key]),
                    float(meta["3L"][key]),
                    float(best[1] if isinstance(best, list) else best),
                )
            )
    return cells


# -- the board: cached runs, families, findings ---------------------------------------


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (cell TEXT, mode TEXT, config TEXT, value TEXT, PRIMARY KEY (cell, mode, config));
CREATE TABLE IF NOT EXISTS families (id INTEGER PRIMARY KEY AUTOINCREMENT, author TEXT, name TEXT, spec TEXT,
    designs TEXT, hypothesis TEXT, stats TEXT, created REAL);
CREATE TABLE IF NOT EXISTS findings (id INTEGER PRIMARY KEY AUTOINCREMENT, author TEXT, text TEXT, families TEXT, created REAL);
"""


class Board:
    """Shared by every architect (and the host scripts) through one sqlite file; vqsim runs
    take one of `slots` machine-wide slots (file locks), so concurrent sessions share the CPU."""

    def __init__(self, root: Path, suite: Path, vqsim: Path, *, slots: int = 16, objective: str = "static") -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        self.objective = objective
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "slots").mkdir(exist_ok=True)
        self.slots = slots
        self.vqsim = vqsim
        self.cells = load_cells(suite)
        self.by_id = {cell.id: cell for cell in self.cells}
        self.local = threading.local()
        with self.db() as db:
            db.executescript(SCHEMA)

    def db(self) -> sqlite3.Connection:
        connection = getattr(self.local, "db", None)
        if connection is None:
            connection = sqlite3.connect(self.root / "board.db", timeout=120)
            connection.execute("PRAGMA journal_mode=WAL")
            self.local.db = connection
        return connection

    # -- simulation ---------------------------------------------------------------

    def _slot(self):
        while True:
            for i in range(self.slots):
                handle = open(self.root / "slots" / f"{i}.lock", "w")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return handle
                except BlockingIOError:
                    handle.close()
            time.sleep(0.5)

    def _job(self, cell: Cell, mode: str, config: str) -> str:
        if mode == "prefix":
            return f"{cell.capacity} {cell.split} {SEEN_WINDOWS} {config} until={cell.split}\n"
        if mode == "full":  # cold, the whole workload, per-window misses
            return f"{cell.capacity} {cell.split} {WINDOWS} {config}\n"
        return f"{cell.capacity} {cell.split} 0 {config} handover=1\n"

    def _run_cell(self, cell: Cell, jobs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        handle = self._slot()
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".jobs", dir=self.root, delete=False) as fh:
                fh.write("".join(self._job(cell, mode, config) for mode, config in jobs))
                path = fh.name
            # A backstop: about 1M requests/s per job at worst under load, plus slack.
            timeout = 300 + len(jobs) * cell.requests / 1e6
            try:
                out = subprocess.run(
                    ["nice", "-n", "10", str(self.vqsim), str(cell.directory / "ids.u32"), path], capture_output=True, text=True, check=True, timeout=timeout
                ).stdout
            finally:
                os.unlink(path)
        finally:
            handle.close()
        lines = out.splitlines()
        if len(lines) != len(jobs) or any(line.startswith("error") for line in lines):
            raise RuntimeError(f"vqsim failed on {cell.id}: {lines[:3]}")
        return dict(zip(jobs, lines, strict=True))

    def results(self, wanted: dict[str, set[tuple[str, str]]], workers: int | None = None) -> dict[tuple[str, str, str], list[int]]:
        """{cell id: {(mode, config)}} -> {(cell id, mode, config): numbers}, cached on the board."""
        db, found, missing = self.db(), {}, {}
        modes = {mode for jobs in wanted.values() for mode, _ in jobs}
        cached = {}
        for mode in modes:
            for cell_id, config, value in db.execute("SELECT cell, config, value FROM runs WHERE mode=?", (mode,)):
                cached[cell_id, mode, config] = value
        for cell_id, jobs in wanted.items():
            for mode, config in jobs:
                value = cached.get((cell_id, mode, config))
                if value is not None:
                    found[cell_id, mode, config] = [int(x) for x in value.split()]
                else:
                    missing.setdefault(cell_id, []).append((mode, config))
        if missing:
            # Big cells first; a vqsim process per cell and mode keeps each trace's pass to one
            # set of jobs (handover jobs of one cell reuse S3-FIFO's prefix state).
            batches = [(cell_id, [job for job in jobs if job[0] == mode]) for cell_id, jobs in missing.items() for mode in ("prefix", "handover", "full")]
            batches = sorted((batch for batch in batches if batch[1]), key=lambda batch: -self.by_id[batch[0]].requests * len(batch[1]))
            with ThreadPoolExecutor(workers or self.slots) as pool:
                done = list(pool.map(lambda batch: (batch[0], self._run_cell(self.by_id[batch[0]], batch[1])), batches))
            with db:
                for cell_id, lines in done:
                    for (mode, config), line in lines.items():
                        db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)", (cell_id, mode, config, line))
                        found[cell_id, mode, config] = [int(x) for x in line.split()]
        return found

    # -- selection and scoring ----------------------------------------------------

    def choose(self, configs: list[str]) -> dict[str, str]:
        """Per cell, the config with the fewest misses on the observed part (after the warm-up)."""
        configs = list(dict.fromkeys(configs))
        runs = self.results({cell.id: {("prefix", config) for config in configs} for cell in self.cells})
        chosen = {}
        for cell in self.cells:
            seen = [(sum(runs[cell.id, "prefix", config][2 + SEEN_FROM : 2 + SEEN_WINDOWS]), i) for i, config in enumerate(configs)]
            chosen[cell.id] = configs[min(seen)[1]]
        return chosen

    def observed(self, configs: list[str]) -> dict[str, dict[str, float]]:
        """{config: {cell id: change in misses vs S3-FIFO on the observed part}} (cached runs)."""
        configs = list(dict.fromkeys([S3_CONFIG, *configs]))
        runs = self.results({cell.id: {("prefix", config) for config in configs} for cell in self.cells})
        out = {}
        for config in configs:
            out[config] = {}
            for cell in self.cells:
                base = sum(runs[cell.id, "prefix", S3_CONFIG][2 + SEEN_FROM : 2 + SEEN_WINDOWS])
                mine = sum(runs[cell.id, "prefix", config][2 + SEEN_FROM : 2 + SEEN_WINDOWS])
                out[config][cell.id] = (mine - base) / base if base else 0.0
        return out

    def score(self, chosen: dict[str, str]) -> dict[str, float]:
        """{cell id: change in misses vs S3-FIFO on the unseen part} for a choice per cell, each
        design taking over S3-FIFO's live cache at the split."""
        wanted = {cell_id: {("handover", config), ("handover", S3_CONFIG)} for cell_id, config in chosen.items()}
        runs = self.results(wanted)
        out = {}
        for cell_id, config in chosen.items():
            base = runs[cell_id, "handover", S3_CONFIG][1]
            out[cell_id] = (runs[cell_id, "handover", config][1] - base) / base if base else 0.0
        return out

    # -- the online objective -------------------------------------------------------

    def online(self, candidates: list[str], *, k: int | None = None) -> tuple[list[str], dict[str, float], dict[str, int]]:
        """Idealized online re-choice among `candidates` (shadow runs, free switching): from the split
        on, each window is served by the candidate with the fewest misses in windows 2 .. w-1 (ties:
        the earlier one). With k, the candidates are first narrowed to a greedy portfolio of k that
        starts from S3-FIFO. Returns (the candidates used, {cell id: change vs S3-FIFO on windows
        FIRST_SCORED..39}, {cell id: index in `candidates` of the one that led the most windows})."""
        candidates = list(dict.fromkeys(candidates))
        configs = list(dict.fromkeys([S3_CONFIG, *candidates]))
        runs = self.results({cell.id: {("full", config) for config in configs} for cell in self.cells})
        w = np.array([[runs[cell.id, "full", config][2:] for config in configs] for cell in self.cells], dtype=float)
        scored = w[:, :, FIRST_SCORED:]
        # misses of each design in windows 2 .. w-1, for each scored window w
        so_far = np.cumsum(w[:, :, LEADER_FROM:], axis=2)[:, :, FIRST_SCORED - LEADER_FROM - 1 : WINDOWS - LEADER_FROM - 1]
        base = scored[:, 0].sum(1)
        safe = np.where(base > 0, base, 1.0)
        index = [configs.index(c) for c in candidates]
        if k is not None and k < len(index):
            index = self._greedy(scored, so_far, base, safe, k)
        sub = np.array(index)
        lead = sub[so_far[:, sub].argmin(1)]  # cells x scored windows
        total = np.take_along_axis(scored, lead[:, None, :], 1)[:, 0].sum(1)
        change = np.where(base > 0, (total - base) / safe, 0.0)
        used = [configs[i] for i in index]
        led = {}
        for j, cell in enumerate(self.cells):
            counts = np.bincount(lead[j], minlength=len(configs))
            led[cell.id] = candidates.index(configs[int(counts.argmax())]) if configs[int(counts.argmax())] in candidates else -1
        return used, {cell.id: float(change[j]) for j, cell in enumerate(self.cells)}, led

    @staticmethod
    def _greedy(scored, so_far, base, safe, k: int) -> list[int]:
        """From {S3-FIFO}, add the design that lowers the mean change most, k - 1 times."""
        port = [0]
        lead_so_far, lead_scored = so_far[:, 0].copy(), scored[:, 0].copy()
        while len(port) < k:
            better = so_far < lead_so_far[:, None, :]
            total = np.where(better, scored, lead_scored[:, None, :]).sum(2)
            score = ((total - base[:, None]) / safe[:, None]).mean(0)
            score[port] = np.inf
            i = int(score.argmin())
            if not np.isfinite(score[i]):
                break
            port.append(i)
            lead_so_far = np.where(better[:, i], so_far[:, i], lead_so_far)
            lead_scored = np.where(better[:, i], scored[:, i], lead_scored)
        return port

    def alone(self, configs: list[str]) -> dict[str, float]:
        """{config: mean change vs S3-FIFO on the scored windows, the design alone from a cold start}."""
        configs = list(dict.fromkeys(configs))
        runs = self.results({cell.id: {("full", config) for config in [S3_CONFIG, *configs]} for cell in self.cells})
        out = {}
        for config in configs:
            values = []
            for cell in self.cells:
                base = sum(runs[cell.id, "full", S3_CONFIG][2 + FIRST_SCORED :])
                values.append((sum(runs[cell.id, "full", config][2 + FIRST_SCORED :]) - base) / base if base else 0.0)
            out[config] = statistics.mean(values)
        return out

    def pool_changes(self) -> dict[str, float]:
        """The pool's per-cell result under the board's objective."""
        if self.objective == "online":
            return self.online(self.pool(), k=PORTFOLIO)[1]
        return self.score(self.choose(self.pool()))

    # -- families -----------------------------------------------------------------

    def families(self) -> list[dict[str, Any]]:
        rows = self.db().execute("SELECT id, author, name, spec, designs, hypothesis, stats, created FROM families ORDER BY id").fetchall()
        keys = ("id", "author", "name", "spec", "designs", "hypothesis", "stats", "created")
        out = []
        for row in rows:
            record = dict(zip(keys, row, strict=True))
            for key in ("spec", "designs", "stats"):
                record[key] = json.loads(record[key]) if record[key] else None
            out.append(record)
        return out

    def pool(self, *, exclude: int | None = None, include: int | None = None) -> list[str]:
        """S3-FIFO and every design of the evaluated families (and of `include`, being evaluated)."""
        configs = [S3_CONFIG]
        for family in self.families():
            if family["id"] != exclude and (family["stats"] is not None or family["id"] == include):
                configs.extend(config for _, config in family["designs"])
        return list(dict.fromkeys(configs))

    def add_family(self, author: str, name: str, spec: Any, designs: list[tuple[str, str]], hypothesis: str) -> int:
        with self.db() as db:
            cursor = db.execute(
                "INSERT INTO families (author, name, spec, designs, hypothesis, stats, created) VALUES (?, ?, ?, ?, ?, NULL, ?)",
                (author, name, json.dumps(spec), json.dumps(designs), hypothesis, time.time()),
            )
            return int(cursor.lastrowid)

    def set_stats(self, family_id: int, stats: dict[str, Any]) -> None:
        with self.db() as db:
            db.execute("UPDATE families SET stats=? WHERE id=?", (json.dumps(stats), family_id))

    def add_finding(self, author: str, text: str, families: list[int]) -> int:
        with self.db() as db:
            cursor = db.execute("INSERT INTO findings (author, text, families, created) VALUES (?, ?, ?, ?)", (author, text, json.dumps(families), time.time()))
            return int(cursor.lastrowid)

    def findings(self) -> list[dict[str, Any]]:
        rows = self.db().execute("SELECT id, author, text, families FROM findings ORDER BY id").fetchall()
        return [{"id": row[0], "author": row[1], "text": row[2], "families": json.loads(row[3])} for row in rows]

    # -- evaluation ---------------------------------------------------------------

    def evaluate(self, author: str, family: dict[str, Any], hypothesis: str) -> dict[str, Any]:
        return self.evaluate_designs(author, str(family.get("name") or "unnamed")[:60], family, expand(family), hypothesis)

    def evaluate_designs(self, author: str, name: str, spec: Any, designs: list[tuple[str, str]], hypothesis: str) -> dict[str, Any]:
        family_id = self.add_family(author, name, spec, designs, hypothesis)
        configs = [config for _, config in designs]
        if self.objective == "online":
            stats = self._evaluate_online(family_id, designs)
            self.set_stats(family_id, stats)
            return {"id": family_id, **stats}
        chosen = self.choose(configs)
        changes = self.score(chosen)
        observed = self.observed(configs)
        before = self.choose(self.pool(exclude=family_id))
        after = self.choose(self.pool(include=family_id))
        pool_before, pool_after = self.score(before), self.score(after)
        stats = {
            "designs": len(designs),
            "pick": summarize(self.cells, changes),
            "pool_before": summarize(self.cells, pool_before),
            "pool_after": summarize(self.cells, pool_after),
            "pool_picks": sum(after[cell.id] in set(configs) and after[cell.id] != before[cell.id] for cell in self.cells),
            "per_design": [
                {
                    "label": label,
                    "observed_mean": statistics.mean(observed[config].values()),
                    "picked": sum(chosen[cell.id] == config for cell in self.cells),
                }
                for label, config in designs
            ],
            "cells": {cell.id: changes[cell.id] for cell in self.cells},
            "chosen": {cell.id: configs.index(chosen[cell.id]) for cell in self.cells},
        }
        self.set_stats(family_id, stats)
        return {"id": family_id, **stats}


    def _evaluate_online(self, family_id: int, designs: list[tuple[str, str]]) -> dict[str, Any]:
        configs = [config for _, config in designs]
        _, changes, led = self.online(configs)
        alone = self.alone(configs)
        before, pool_before, _ = self.online(self.pool(exclude=family_id), k=PORTFOLIO)
        after, pool_after, _ = self.online(self.pool(include=family_id), k=PORTFOLIO)
        unique = list(dict.fromkeys(configs))
        return {
            "objective": "online",
            "designs": len(designs),
            "pick": summarize(self.cells, changes),
            "pool_before": summarize(self.cells, pool_before),
            "pool_after": summarize(self.cells, pool_after),
            "pool_picks": sum(config in set(configs) and config not in set(before) for config in after),
            "per_design": [
                {"label": label, "observed_mean": alone[config], "picked": sum(unique[led[cell.id]] == config for cell in self.cells if led[cell.id] >= 0)}
                for label, config in designs
            ],
            "cells": changes,
            "chosen": {cell.id: configs.index(unique[led[cell.id]]) if led[cell.id] >= 0 else -1 for cell in self.cells},
        }


def summarize(cells: list[Cell], changes: dict[str, float]) -> dict[str, Any]:
    values = [changes[cell.id] for cell in cells]
    l3 = [cell.change(cell.l3) for cell in cells]
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": ordered[len(ordered) // 10],
        "p90": ordered[(9 * len(ordered)) // 10],
        "beats_3l": sum(v < t for v, t in zip(values, l3, strict=True)),
        "worse_than_s3fifo": sum(v > 0.0005 for v in values),
        "l3_mean": statistics.mean(l3),
        "l3_median": statistics.median(l3),
        "cells": len(values),
    }


def pct(value: float) -> str:
    return f"{100 * value:+.2f}%"


def summary_line(stats: dict[str, Any]) -> str:
    return (
        f"mean {pct(stats['mean'])}, median {pct(stats['median'])} (p10 {pct(stats['p10'])}, p90 {pct(stats['p90'])}); "
        f"beats 3L-Cache on {stats['beats_3l']}/{stats['cells']} cells; worse than S3-FIFO on {stats['worse_than_s3fifo']}"
    )


SIZE_CLASSES = ((0.0, 0.5, "<0.5%"), (0.5, 2.0, "0.5-2%"), (2.0, 8.0, "2-8%"), (8.0, 1e9, ">=8%"))


def by_size(cells: list[Cell], changes: dict[str, float]) -> list[str]:
    rows = []
    for low, high, label in SIZE_CLASSES:
        group = [cell for cell in cells if low <= cell.size_pct < high]
        if group:
            mine = statistics.mean(changes[cell.id] for cell in group)
            theirs = statistics.mean(cell.change(cell.l3) for cell in group)
            rows.append(f"  cache {label} of the objects ({len(group)} cells): {pct(mine)} vs 3L-Cache {pct(theirs)}")
    return rows


# -- the agent session ----------------------------------------------------------------

OBJECTIVE = {
    "static": (
        "Design families of virtual-queue FIFO caches that, chosen per workload from its observed 20%, "
        "clearly beat S3-FIFO and 3L-Cache on the unseen 80% of long workloads. Finish with submit_families."
    ),
    "online": (
        "Design families of virtual-queue FIFO caches that, as candidates of an online selector that switches "
        "the live cache to the design doing best so far, make it clearly beat S3-FIFO and 3L-Cache on the "
        "unseen 80% of long workloads. Finish with submit_families."
    ),
}


def _schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": required, "properties": properties}


STAGE_SCHEMA = {
    "type": "object",
    "description": "One FIFO stage. Any field except name may be a list of values (a range).",
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 12},
        "size": {"description": "Target size, fraction of the cache (not for the last stage: it takes the rest)."},
        "counts": {"description": "Whether hits in this stage raise the counter (default true)."},
        "promote": {"description": "Counter at the tail needed to go up (0 = always pass on). Default 1."},
        "up": {"description": "Stage an object goes to when promoted (itself = reinsertion). Default: itself."},
        "onup": {"description": "Counter after going up: reset | keep | dec | half. Default reset."},
        "down": {"description": "Stage an unpromoted object goes to; null = it leaves the cache. Default null."},
        "ghost": {"description": "Whether objects leaving from this stage enter the ghost (default false)."},
        "delay": {"description": "D-FR: a hit counts only if more than delay*C insertions passed since the last counted one (0 = off)."},
        "age": {"description": "AGE: at the tail, an object idle for at least age*C insertions is not promoted (0 = off)."},
    },
}
FAMILY_SCHEMA = {
    "type": "object",
    "description": "A design with ranges: any list-valued field is varied; the family is all combinations (at most 48).",
    "required": ["name", "stages"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 60},
        "stages": {"type": "array", "minItems": 1, "maxItems": MAX_STAGES, "items": STAGE_SCHEMA},
        "entry": {"description": "Stage new objects enter (default: the first)."},
        "overflow": {"description": "While the cache is still filling, where new objects go once entry is full (default: the last)."},
        "ghost_entry": {"description": "Stage a ghost hit enters (default: the last)."},
        "ghost_freq": {"description": "Counter a ghost hit starts with (default 0)."},
        "ghost_ratio": {"description": "Ghost capacity, fraction of the cache (default 0.9)."},
        "max_freq": {"description": "Counter cap, 1..15 (default 3)."},
        "adopt_small": {"description": "At the takeover, where S3-FIFO's S objects go (default: entry)."},
        "adopt_main": {"description": "At the takeover, where S3-FIFO's M objects go (default: the last stage)."},
    },
}

SPECS = (
    ToolSpec(
        "evaluate_family",
        (
            "Evaluate a family of designs on every training cell under the deployment protocol (see the system "
            "prompt), scored on the unseen 80%. Returns the family's own result next to 3L-Cache, by cache "
            "size, its contribution to the board's pool, per design its mean result and the cells where it "
            "was chosen, and the cells where the family is furthest behind 3L-Cache. Costs one evaluation; "
            "state the hypothesis it tests."
        ),
        _schema(
            ["family", "hypothesis"],
            {"family": FAMILY_SCHEMA, "hypothesis": {"type": "string", "minLength": 1, "maxLength": 1500}},
        ),
        Capability.ACTION,
        Effect.ENVIRONMENT,
        costs={"experiments": 1},
    ),
    ToolSpec(
        "view_board",
        "The shared board: every family evaluated so far by any architect (author, hypothesis, result, contribution to the pool), the pool's current result, and the findings recorded.",
        _schema([], {"top": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Families to list, best contribution first (default 25)."}}),
        Capability.INSPECTION,
        Effect.NONE,
    ),
    ToolSpec(
        "view_family",
        "One family in full: its spec, hypothesis, every design with its mean result and how often it was chosen, and its per-cell result.",
        _schema(["family"], {"family": {"type": "integer", "minimum": 1}}),
        Capability.INSPECTION,
        Effect.NONE,
    ),
    ToolSpec(
        "view_cells",
        "Every training cell (anonymous): cache size as a share of the trace's objects, S3-FIFO's miss ratio on the unseen part, 3L-Cache's and the best published algorithm's change, and the pool's change (or a given family's).",
        _schema([], {"family": {"type": "integer", "minimum": 1, "description": "Show this family's change too."}}),
        Capability.INSPECTION,
        Effect.NONE,
    ),
    ToolSpec(
        "record_finding",
        "Record a finding on the shared board for every architect (now and in later generations): what a set of families showed about which mechanism or parameter matters, with the numbers.",
        _schema(
            ["text", "families"],
            {"text": {"type": "string", "minLength": 1, "maxLength": 2000}, "families": {"type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": 32}},
        ),
        Capability.ACTION,
        Effect.ENVIRONMENT,
    ),
    ToolSpec(
        "submit_families",
        "End the task: name the families (yours or others') that should form the final pool, and summarize what you learned. The final pool is evaluated once on held-out long workloads.",
        _schema(
            ["families", "summary"],
            {"families": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1, "maxItems": 32}, "summary": {"type": "string", "minLength": 1, "maxLength": 4000}},
        ),
        Capability.SUBMISSION,
        Effect.SUBMISSION,
    ),
)


class ArchitectSession:
    def __init__(self, board: Board, *, author: str, max_evaluations: int, direction: str | None = None) -> None:
        self.board = board
        self.author = author
        self.max_evaluations = max_evaluations
        self.direction = direction
        self.evaluated: list[int] = []
        self.used = 0  # evaluations charged (invalid families are not)
        self.submission_record: dict[str, Any] | None = None
        self.tool_counts: dict[str, int] = {}
        self.task = PublicTask("cachelab/vq-architect", OBJECTIVE[board.objective], {"cells": len(board.cells), "evaluations": max_evaluations, "views": "numbers"}, ObservationBundle())
        self.submission = SubmissionSpec(Effect.SUBMISSION)
        handlers = {
            "evaluate_family": self._evaluate_family,
            "view_board": self._view_board,
            "view_family": self._view_family,
            "view_cells": self._view_cells,
            "record_finding": self._record_finding,
            "submit_families": self._submit_families,
        }
        self.tools = tuple(ToolBinding(spec, self._counted(spec.name, handlers[spec.name])) for spec in SPECS)
        self.initial = initial_prompt(board, max_evaluations, direction)

    def _counted(self, name, handler):
        def call(arguments: Any) -> ToolReply:
            self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
            try:
                return handler(arguments)
            except FamilyError as error:
                return ToolReply(f"Invalid family: {error}", success=False)
            except Exception as error:  # a failed simulation must not end the session
                return ToolReply(f"The evaluation failed ({type(error).__name__}: {str(error)[:300]}); nothing was recorded for it.", success=False)

        return call

    @property
    def submitted(self) -> bool:
        return self.submission_record is not None

    def _evaluate_family(self, arguments: dict[str, Any]) -> ToolReply:
        # The session keeps the evaluation budget (the runtime's meter has slack), so an architect
        # that runs out can still read the board and submit.
        if self.used >= self.max_evaluations:
            return ToolReply(f"No evaluations left ({self.used} of {self.max_evaluations} used). Finish with submit_families.", success=False)
        family = arguments["family"]
        expand(family)  # validate before anything is recorded or counted
        self.used += 1
        stats = self.board.evaluate(self.author, family, arguments["hypothesis"])
        self.evaluated.append(stats["id"])
        left = self.max_evaluations - self.used
        text = family_report(self.board, stats, name=str(family.get("name"))) + f"\nEvaluations left: {left}." + (" Finish with submit_families." if not left else "")
        return ToolReply(text, event_text=f"Evaluated family F{stats['id']}: {summary_line(stats['pick'])}")

    def _view_board(self, arguments: dict[str, Any]) -> ToolReply:
        return ToolReply(board_report(self.board, int(arguments.get("top", 25))))

    def _view_family(self, arguments: dict[str, Any]) -> ToolReply:
        family = next((f for f in self.board.families() if f["id"] == arguments["family"]), None)
        if family is None or family["stats"] is None:
            return ToolReply(f"No evaluated family F{arguments['family']}.", success=False)
        lines = [
            f"F{family['id']} {family['name']} by {family['author']}: {family['hypothesis']}",
            "Spec: " + json.dumps(family["spec"]),
            family_report(self.board, {"id": family["id"], **family["stats"]}, name=family["name"], all_designs=True, all_cells=True),
        ]
        return ToolReply("\n".join(lines))

    def _view_cells(self, arguments: dict[str, Any]) -> ToolReply:
        pool = self.board.pool_changes()
        family = None
        if "family" in arguments:
            family = next((f for f in self.board.families() if f["id"] == arguments["family"] and f["stats"]), None)
            if family is None:
                return ToolReply(f"No evaluated family F{arguments['family']}.", success=False)
        head = "cell  size%   S3-FIFO-mr  3L-Cache  best-published  pool" + (f"  F{family['id']}" if family else "")
        lines = [head]
        for cell in self.board.cells:
            row = f"{cell.id}  {cell.size_pct:6.3f}  {cell.s3fifo:10.4f}  {pct(cell.change(cell.l3)):>8}  {pct(cell.change(cell.best)):>14}  {pct(pool[cell.id]):>7}"
            if family:
                row += f"  {pct(family['stats']['cells'][cell.id]):>7}"
            lines.append(row)
        return ToolReply("\n".join(lines))

    def _record_finding(self, arguments: dict[str, Any]) -> ToolReply:
        finding = self.board.add_finding(self.author, arguments["text"], list(arguments["families"]))
        return ToolReply(f"Recorded finding {finding} on the board.")

    def _submit_families(self, arguments: dict[str, Any]) -> ToolReply:
        known = {family["id"] for family in self.board.families() if family["stats"]}
        unknown = [f for f in arguments["families"] if f not in known]
        if unknown:
            return ToolReply(f"Not evaluated families: {unknown}.", success=False)
        self.submission_record = {"families": list(arguments["families"]), "summary": arguments["summary"]}
        event = f"Submitted families {', '.join(f'F{f}' for f in arguments['families'])}. The task is complete."
        return ToolReply(event, finished=True, event_text=event)

    def submit(self, answer: str) -> ToolReply:
        return ToolReply("Nothing was submitted. Continue with the tools, and finish with submit_families.", success=False)

    def result(self) -> dict[str, Any]:
        return {"submitted": self.submitted, "submission": self.submission_record, "evaluated": self.evaluated, "tool_counts": dict(self.tool_counts), "author": self.author, "direction": self.direction}

    def close(self) -> None:
        pass


def family_report(board: Board, stats: dict[str, Any], *, name: str, all_designs: bool = False, all_cells: bool = False) -> str:
    cells = board.cells
    pick, before, after = stats["pick"], stats["pool_before"], stats["pool_after"]
    online = stats.get("objective") == "online"
    if online:
        lines = [
            f"Family F{stats['id']} '{name}': {stats['designs']} designs on {len(cells)} training cells; online choice among its "
            "designs (each window served by the design with the fewest misses so far), scored on the unseen 80%.",
        ]
    else:
        lines = [
            f"Family F{stats['id']} '{name}': {stats['designs']} designs on {len(cells)} training cells, chosen per cell on the "
            "observed part, scored on the unseen 80% after taking over S3-FIFO's live cache.",
        ]
    lines += [
        f"  this family: {summary_line(pick)}",
        f"  3L-Cache on the same cells: mean {pct(pick['l3_mean'])}, median {pct(pick['l3_median'])}",
        *by_size(cells, stats["cells"]),
        f"Board pool{f' (greedy portfolio of {PORTFOLIO})' if online else ''}: {summary_line(before)}",
        f"  with this family: {summary_line(after)} "
        + (f"({stats['pool_picks']} of its designs enter the portfolio)" if online else f"(its designs now chosen in {stats['pool_picks']} more cells)"),
        (
            "Designs (mean change vs S3-FIFO on the unseen 80%, the design alone from a cold start; cells where it led the most windows):"
            if online
            else "Designs (mean change vs S3-FIFO on the observed part; cells where chosen):"
        ),
    ]
    designs = stats["per_design"]
    ranked = sorted(range(len(designs)), key=lambda i: designs[i]["observed_mean"])
    shown = ranked if all_designs or len(ranked) <= 16 else ranked[:12] + ranked[-4:]
    for i in shown:
        lines.append(f"  d{i + 1:02d} {designs[i]['label']}: {pct(designs[i]['observed_mean'])}; chosen {designs[i]['picked']}")
    if len(shown) < len(ranked):
        lines.append(f"  ... {len(ranked) - len(shown)} middle designs not shown (view_family shows all)")
    gaps = sorted(cells, key=lambda cell: cell.change(cell.l3) - stats["cells"][cell.id])
    lines.append("Cells furthest behind 3L-Cache (cell, cache size % of objects, S3-FIFO miss ratio: family vs 3L-Cache, chosen design):")
    for cell in gaps if all_cells else gaps[:8]:
        chosen = stats["chosen"][cell.id]
        lines.append(
            f"  {cell.id} {cell.size_pct:.3f}% mr {cell.s3fifo:.3f}: {pct(stats['cells'][cell.id])} vs {pct(cell.change(cell.l3))}, "
            + (f"d{chosen + 1:02d}" if chosen >= 0 else "S3-FIFO")
        )
    return "\n".join(lines)


def board_report(board: Board, top: int) -> str:
    families = [family for family in board.families() if family["stats"]]
    if not families:
        return "The board is empty: no family evaluated yet."
    pool = board.pool_changes()
    what = f"Pool now (online over a greedy portfolio of {PORTFOLIO})" if board.objective == "online" else "Pool now"
    lines = [f"Board: {len(families)} families, {len(board.pool())} designs in the pool.", f"{what}: {summary_line(summarize(board.cells, pool))}"]

    def gain(family):
        stats = family["stats"]
        return stats["pool_after"]["mean"] - stats["pool_before"]["mean"]

    lines.append("Families by contribution to the pool when added (mean change of the pool's result), then their own result:")
    for family in sorted(families, key=gain)[:top]:
        stats = family["stats"]
        lines.append(
            f"  F{family['id']} {family['name']} ({family['author']}, {stats['designs']} designs): pool {pct(gain(family))}; "
            f"own {summary_line(stats['pick'])}. Hypothesis: {family['hypothesis'][:300]}"
        )
    findings = board.findings()
    if findings:
        lines.append("Findings:")
        lines.extend(f"  [{f['id']}] {f['author']} (F{', F'.join(map(str, f['families']))}): {f['text']}" for f in findings)
    return "\n".join(lines)


# -- prompts --------------------------------------------------------------------------

PRINCIPLES = Path(__file__).resolve().parents[4] / "docs" / "cachelab-design-principles.md"


def principles_text() -> str:
    """Part 1 of the design principles (agent-facing)."""
    try:
        text = PRINCIPLES.read_text(encoding="utf-8")
    except OSError:
        return ""
    start = text.find("## 1.")
    end = text.find("## 2.")
    return text[start:end].strip() if start >= 0 else ""


SYSTEM = """You are a cache architect. You design families of FIFO-based cache eviction policies and test them on long real workloads, working alongside other architects who share one board.

The design space (vq: "virtual queues"). An object lives in one of K FIFO stages. Hits never move objects; they only raise a small counter (capped at max_freq), and only in stages that count. When the cache is full, the first stage (in order) at or over its target size is examined from its tail. The object there goes up (to any stage; to itself = reinsertion) if its counter reaches the stage's promote threshold, with its counter reset, kept, decremented or halved; otherwise it goes down to another stage, or leaves the cache (into the ghost, a FIFO of recently evicted keys, if the stage feeds it). The scan stays on that stage until an object leaves; after a demotion the stage is picked again. New objects enter `entry`; a miss found in the ghost enters `ghost_entry` with counter ghost_freq. Per stage, delay (Delayed FIFO-Reinsertion) makes clustered hits count once, and age (Age-Guided Eviction) refuses promotion to objects idle for age*C insertions (time is the insertion clock: an object crosses a FIFO cache in about C insertions).

S3-FIFO is the family {"name": "s3fifo", "stages": [{"name": "S", "size": 0.1, "promote": 2, "up": "M", "onup": "reset", "ghost": true}, {"name": "M", "promote": 1, "up": "M", "onup": "dec"}], "ghost_ratio": 0.9, "max_freq": 3}. S4-FIFO's skip region is a stage in front of S with counts false and promote 0 (every object passes on, hits there do not count). A hot stage above main whose unpromoted objects are demoted to main gives a two-level main. Ranges: [0.05, 0.1, 0.2] for a size, [1, 2] for a threshold; a family is all combinations (at most 48), and parameters that only work together should be varied together.

{protocol}

Work like a researcher: state a hypothesis per family, read which designs were chosen and their observed-part means to learn which parameters matter and which interact, look at the cells where you lose, read the board for what others found (do not repeat their families), and record findings others can use. Cells are anonymous; there is nothing to learn about trace identities.

Principles from the cache-design literature (S3-FIFO, SIEVE, lazy promotion, LAH/S4-FIFO):
{principles}

Budget: {evaluations} evaluations. Finish with submit_families: the families that should form the final pool (any architect's), and a summary of what you learned."""


PROTOCOLS = {
    "static": """Protocol (fixed; the same for the final test). Each cell is a long workload (12M-50M requests) at one cache size. A policy sees the first 20% (observed part) while S3-FIFO serves it; each design of a family is simulated on that part, the one with the fewest misses there (after its first quarter) is chosen, and it takes over S3-FIFO's live cache and serves the remaining 80%. A family's score is the per-cell change in misses vs S3-FIFO on those 80%, compared with 3L-Cache (a learned policy, the strongest published one here). Memory: designs must stay within 2x S3-FIFO's metadata; a deep ghost (ghost_ratio) and delay/age cost memory, and families over the estimate are rejected.

Selection over many designs can fit noise on the observed part: a family is only as good as the choice it supports, and the pool's result (all designs on the board) shows whether more designs still help. Aim for a clear advantage over 3L-Cache: a better mean and median, more cells won, few cells worse than S3-FIFO.""",
    "online": """Protocol (fixed; the same for the final test). Each cell is a long workload (10M-50M requests) at one cache size, cut into 40 windows. S3-FIFO serves the first 20% (windows 0-7); from then on the live cache is run by an online selector: it keeps a small portfolio of candidate designs (at most {portfolio}), simulates each of them on a sample of the workload from the start, and at the start of every window switches the live cache to the candidate with the fewest misses so far (counted from window 2). The score is the per-cell change in misses vs S3-FIFO on windows 8-39 (the unseen 80%), compared with 3L-Cache (a learned policy, the strongest published one here). Here the selector is idealized (every design's own full run, free switching); the deployed one pays a small cost per switch. Memory: designs must stay within 2x S3-FIFO's metadata; a deep ghost (ghost_ratio) and delay/age cost memory, and families over the estimate are rejected.

The board's pool is scored as the deployed selector would use it: a greedy portfolio of {portfolio} designs picked from every design on the board (starting from S3-FIFO, adding the design that helps most). So a design is worth adding when it wins windows, phases or workloads that the portfolio's designs lose, even if it is poor elsewhere: the selector only runs it while it leads. Designs that are good everywhere are already on the board; look for what is missing. A family's own result is the same online choice among its designs only. Aim for a clear advantage over 3L-Cache: a better mean and median, more cells won, few cells worse than S3-FIFO.""",
}


def system_prompt(max_evaluations: int, objective: str = "static") -> str:
    protocol = PROTOCOLS[objective].replace("{portfolio}", str(PORTFOLIO))
    return SYSTEM.replace("{protocol}", protocol).replace("{principles}", principles_text()).replace("{evaluations}", str(max_evaluations))


def initial_prompt(board: Board, max_evaluations: int, direction: str | None) -> str:
    cells = board.cells
    sizes = sorted(cell.size_pct for cell in cells)
    lines = [
        f"Training suite: {len(cells)} cells ({len({cell.directory for cell in cells})} workloads x cache sizes from {sizes[0]:.3f}% to {sizes[-1]:.2f}% of a workload's objects).",
        "On these cells (unseen 80%, change vs S3-FIFO): "
        f"3L-Cache mean {pct(statistics.mean(cell.change(cell.l3) for cell in cells))}, median {pct(statistics.median(cell.change(cell.l3) for cell in cells))}; "
        f"the best published algorithm per cell in hindsight mean {pct(statistics.mean(cell.change(cell.best) for cell in cells))}.",
        "",
        board_report(board, 25),
        "",
        f"You have {max_evaluations} evaluations.",
    ]
    if direction:
        lines += ["", f"Your direction (other architects take others): {direction}"]
    return "\n".join(lines)


# -- running one architect ------------------------------------------------------------

PROTOCOL = {"static": "cachelab-vq-architect-v1", "online": "cachelab-vq-architect-online-v1"}


def make_profile(*, backend: str, max_evaluations: int, initial: str, objective: str = "static"):
    from vista.core.profiles import (
        AgentProfile,
        CompactionMode,
        ContextMode,
        InteractionMode,
        PromptBundle,
        SessionPolicy,
    )

    from .profiles import MEMORY_TOOLS

    support = tuple(name for name in MEMORY_TOOLS if backend == "codex" or name != "save_compact_checkpoint")
    return AgentProfile(
        "cachelab-vq-architect/numeric",
        (*(spec.name for spec in SPECS), *support),
        PromptBundle(
            system_prompt(max_evaluations, objective),
            initial,
            step="Continue the design work. Finish with submit_families.",
            checkpoint="Save the compact checkpoint before continuing.",
            recovery=(
                "Continue this same design task from GUIDE, WORKING and the supplied record. Families already "
                "evaluated remain on the board; use view_board and view_family."
            ),
        ),
        SessionPolicy(context=ContextMode.CONTINUOUS, interaction=InteractionMode.TOOLS, compaction=CompactionMode.CHECKPOINT),
    )


def run_architect(
    board: Board,
    *,
    directory: Path,
    backend: str,
    model: str,
    effort: str,
    backend_factory,
    max_evaluations: int,
    author: str,
    direction: str | None = None,
    backend_segments: int | None = None,
    tool_calls: int | None = None,
    transport_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import traceback

    from vista.backends.preflight import source_fingerprint
    from vista.core.artifacts import ArtifactStore
    from vista.core.json import canonical
    from vista.core.runtime import TaskRuntime

    from .profiles import make_experiment
    from .run import research_loop

    root = directory.resolve()
    artifacts = ArtifactStore(root)
    vqsim_digest = __import__("hashlib").sha256(Path(board.vqsim).read_bytes()).hexdigest()[:16]
    artifacts.manifest(
        {
            "benchmark": "CacheLab",
            "protocol": PROTOCOL[board.objective],
            "objective": board.objective,
            "backend": backend,
            "model": model,
            "effort": effort,
            "max_evaluations": max_evaluations,
            "author": author,
            "direction": direction,
            "cells": len(board.cells),
            "board": str(board.root),
            "vqsim": vqsim_digest,
            "transport": transport_provenance or {"validation": "caller-supplied-backend"},
            "source_files": source_fingerprint(),
        }
    )
    session = ArchitectSession(board, author=author, max_evaluations=max_evaluations, direction=direction)
    runtime = backend_session = None
    error = None
    started = time.monotonic()
    try:
        profile = make_profile(backend=backend, max_evaluations=max_evaluations, initial=session.initial, objective=board.objective)
        experiment = make_experiment(
            profile,
            backend=backend,
            model=model,
            effort=effort,
            revision=vqsim_digest,
            suite=[{"cells": len(board.cells)}],
            max_experiments=max_evaluations + 20,  # the session enforces max_evaluations itself
            backend_segments=backend_segments,
            tool_calls=tool_calls,
        )
        runtime = TaskRuntime(session, experiment, artifact_dir=root / "artifacts")
        backend_session = backend_factory(runtime, root / "transport")
        research_loop(session, runtime, backend_session)
        error = runtime.failure if not session.submitted else None
    except Exception:
        error = "run_exception"
        artifacts.record("private_failure", {"traceback": traceback.format_exc()})
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                error = "cleanup_failure"
                artifacts.record("private_failure", {"traceback": traceback.format_exc()})
    result = {
        "benchmark": "CacheLab",
        "protocol": PROTOCOL[board.objective],
        "backend": backend,
        "model": model,
        "effort": effort,
        "error": error,
        **session.result(),
        "usage": runtime.usage if runtime is not None else {},
        "backend_usage": backend_session.usage_reports if backend_session is not None else [],
        "elapsed_seconds": time.monotonic() - started,
    }
    path = root / "result.json"
    path.write_text(canonical(result) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return result
