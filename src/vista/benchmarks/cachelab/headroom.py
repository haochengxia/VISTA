"""Headroom of S3-FIFO's decisions: replay S3-FIFO with a perfect predictor in
charge of one decision, and count the misses that removes.

S3-FIFO makes three predictions about an object's future:

- promotion: at the tail of S, an object hit MOVE_THRESHOLD times moves to M;
- reinsertion: at the tail of M, an object with frequency > 0 goes back to the head;
- ghost admission: a miss whose key is in G enters M instead of S.

The oracle answers the same question with the future: is the object requested
again within a horizon? The horizon is a multiple of the time an object takes
to cross M in S3-FIFO on this cell; which multiple works best depends on the
decision and the cell (promoting everything reused within one crossing floods
M in a small cache), so each decision tries FACTORS and keeps its best.
Everything else stays S3-FIFO. Like the agent's experiments, the oracle takes
over at the handover point, so the result shows what fixing that one decision
could gain on the part an experiment can change. It is a guide, not a bound:
a policy can also change the queues themselves.

This runs the sandbox's S3-FIFO class on the host (trusted code, no agent code)
over the lab's observed trace only.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import NEVER
from .engine import SANDBOX_DIR

DECISIONS = ("promotion", "reinsertion", "ghost admission")
ALL = "all three"
FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)  # horizons tried, in multiples of S3-FIFO's time to cross M
VERSION = 3  # bump when the replay changes (cached results are keyed by it); 3: bounded reinsertion scan
SCAN = 64  # keys the reinsertion oracle examines at M's tail per eviction


def _s3fifo() -> type:
    if str(SANDBOX_DIR) not in sys.path:
        sys.path.append(str(SANDBOX_DIR))  # s3fifo imports policy_api
    from s3fifo import S3FIFO  # type: ignore[import-not-found]

    return S3FIFO


class _Timed(deque):
    """A FIFO that measures how long keys take from its head to its tail."""

    def __init__(self, clock: list[int], since: int) -> None:
        super().__init__()
        self.clock, self.since = clock, since
        self.entered: dict[int, int] = {}
        self.total, self.count = 0, 0

    def appendleft(self, key: int) -> None:
        self.entered[key] = self.clock[0]
        super().appendleft(key)

    def pop(self) -> int:
        key = super().pop()
        entered = self.entered.pop(key)
        if self.clock[0] >= self.since:
            self.total += self.clock[0] - entered
            self.count += 1
        return key

    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


class _Context:
    def move(self, *args) -> None:
        pass

    ghost_add = ghost_drop = ghost_hit = move


def _policy(capacity: int, oracle: dict[str, float], clock: list[int], since: int):
    base = _s3fifo()

    class Oracle(base):
        def __init__(self) -> None:
            super().__init__(capacity, _Context())
            self.small = _Timed(clock, since)
            self.main = _Timed(clock, since)
            self.next_use: dict[int, int] = {}
            self.active = False

        def _soon(self, key: int, decision: str) -> bool:
            following = self.next_use[key]
            return following != NEVER and following - clock[0] <= oracle[decision]

        def insert(self, key: int, now: int) -> str:
            if self.active and self.ghost_hit_pending and "ghost admission" in oracle:
                self.ghost_hit_pending = False
                self.freq[key] = 0
                if self._soon(key, "ghost admission"):
                    self.main.appendleft(key)
                    return "M"
                self.small.appendleft(key)
                return "S"
            return super().insert(key, now)

        def _evict_small(self) -> int | None:
            if not (self.active and "promotion" in oracle):
                return super()._evict_small()
            while self.small:
                key = self.small.pop()
                if self._soon(key, "promotion"):
                    self.freq[key] = 0
                    self.main.appendleft(key)
                    continue
                del self.freq[key]
                self._remember(key)
                return key
            return None

        def _evict_main(self) -> int:
            if not (self.active and "reinsertion" in oracle):
                return super()._evict_main()
            # Keys requested again soon go back to the head, as S3-FIFO's reinsertion does. The scan is
            # bounded: with a long horizon nearly all of M is "soon", and an unbounded scan costs O(C)
            # per eviction (hours on a large cache). If every key examined is soon, the one requested
            # latest is evicted.
            scanned = []
            for _ in range(min(len(self.main), SCAN)):
                key = self.main.pop()
                if not self._soon(key, "reinsertion"):
                    for kept in scanned:
                        self.main.appendleft(kept)
                    del self.freq[key]
                    return key
                scanned.append(key)
            victim = max(scanned, key=self.next_use.__getitem__)
            for kept in scanned:
                if kept != victim:
                    self.main.appendleft(kept)
            del self.freq[victim]
            return victim

    return Oracle()


def replay(keys: np.ndarray, following: np.ndarray, capacity: int, switch: int, oracle: dict[str, float] | None = None) -> dict[str, Any]:
    """S3-FIFO over `keys`; from request `switch` on, the oracle decides each
    decision in `oracle` (decision -> horizon in requests). Returns the per-request
    miss mask and the mean time objects took to cross S and M after `switch`."""
    oracle = oracle or {}
    clock = [0]
    policy = _policy(capacity, oracle, clock, switch)
    resident: set[int] = set()
    miss = np.zeros(len(keys), dtype=bool)
    next_use = policy.next_use
    for now, (key, following_t) in enumerate(zip(keys.tolist(), following.tolist(), strict=True)):
        clock[0] = now
        if now == switch:
            policy.active = bool(oracle)
        next_use[key] = following_t
        if key in resident:
            policy.on_hit(key, now)
            continue
        miss[now] = True
        if policy.on_miss(key, now):
            while len(resident) >= capacity:
                victim = policy.evict(now)
                resident.discard(victim)
                del next_use[victim]  # only resident keys (and the current one) are ever looked up
            policy.insert(key, now)
            resident.add(key)
    return {"miss": miss, "s_crossing": policy.small.mean(), "m_crossing": policy.main.mean()}


def _replay_file(path: str, capacity: int, switch: int, oracle: dict[str, float]) -> np.ndarray:
    with np.load(path, mmap_mode="r") as data:
        keys, following = np.asarray(data["keys"]), np.asarray(data["following"])
    return np.packbits(replay(keys, following, capacity, switch, oracle)["miss"])


def headroom(keys: np.ndarray, following: np.ndarray, capacity: int, switch: int, *, workers: int = 1) -> dict[str, Any]:
    """S3-FIFO's miss mask, and the oracle's for each decision alone (at its best
    horizon) and for all three together; plus every horizon's change, per decision."""
    base = replay(keys, following, capacity, switch)
    crossing = base["m_crossing"] or float(len(keys))  # None: nothing crossed M, the cache never filled
    n = len(keys)
    base_misses = int(base["miss"][switch:].sum())
    sweep = [(decision, factor) for decision in DECISIONS for factor in FACTORS]

    def run_all(oracles: list[dict[str, float]]) -> list[np.ndarray]:
        if workers <= 1 or len(oracles) == 1:
            return [replay(keys, following, capacity, switch, oracle)["miss"] for oracle in oracles]
        try:
            with tempfile.TemporaryDirectory(prefix="headroom-") as scratch:
                path = str(Path(scratch) / "trace.npz")
                np.savez(path, keys=keys, following=following)
                with ProcessPoolExecutor(max_workers=min(workers, len(oracles), os.cpu_count() or 1), mp_context=get_context("spawn")) as pool:
                    packed = list(pool.map(_replay_file, [path] * len(oracles), [capacity] * len(oracles), [switch] * len(oracles), oracles))
        except BrokenProcessPool:  # e.g. an unguarded __main__ under spawn: same result, one at a time
            return [replay(keys, following, capacity, switch, oracle)["miss"] for oracle in oracles]
        return [np.unpackbits(mask, count=n).astype(bool) for mask in packed]

    swept = run_all([{decision: crossing * factor} for decision, factor in sweep])
    changes: dict[str, dict[float, float]] = {decision: {} for decision in DECISIONS}
    best: dict[str, tuple[float, np.ndarray]] = {}
    for (decision, factor), miss in zip(sweep, swept, strict=True):
        misses = int(miss[switch:].sum())
        changes[decision][factor] = (misses - base_misses) / base_misses if base_misses else 0.0
        if decision not in best or misses < int(best[decision][1][switch:].sum()):
            best[decision] = (factor, miss)
    horizons = {decision: crossing * best[decision][0] for decision in DECISIONS}
    together = run_all([horizons])[0]
    masks = {"s3fifo": base["miss"], **{decision: best[decision][1] for decision in DECISIONS}, ALL: together}
    return {"masks": masks, "crossing": float(crossing), "factors": {decision: best[decision][0] for decision in DECISIONS}, "changes": changes, "s_crossing": base["s_crossing"]}


def profile(masks: dict[str, np.ndarray], opt_miss: np.ndarray, start: int, end: int, buckets: int) -> dict[str, Any]:
    """Misses per window of every oracle (and OPT, for scale) relative to S3-FIFO's."""
    from .analysis import bucket_sum, edges

    bounds = edges(start, end, buckets)
    base = bucket_sum(masks["s3fifo"], bounds)
    total = float(base.sum())
    rows: dict[str, dict[str, Any]] = {}
    for name in (*DECISIONS, ALL, "OPT"):
        counts = bucket_sum(opt_miss if name == "OPT" else masks[name], bounds)
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = np.where(base > 0, (counts - base) / np.maximum(base, 1), 0.0)
        rows[name] = {"misses": int(counts.sum()), "relative": relative, "overall": (float(counts.sum()) - total) / total if total else 0.0}
    return {"bounds": bounds, "base_total": int(total), "rows": rows}
