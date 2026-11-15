"""Turn decision logs into the numbers behind every view (host side, numpy).

Everything here is computed from the simulator's own records (outcomes,
evictions, and the moves the policy reported), so a view of a modified policy
means the same thing as a view of the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from .engine import (
    BYPASS,
    COLD,
    GHOST_ADD,
    GHOST_DROP,
    GHOST_HIT,
    HIT,
    MOVE,
    NO_QUEUE,
    REMISS,
    Run,
)

GAP_BINS = (0.25, 0.5, 1.0, 2.0, 4.0)  # return gap after eviction, in cache sizes
GAP_LABELS = ("<0.25C", "0.25-0.5C", "0.5-1C", "1-2C", "2-4C", ">=4C", "never")


@dataclass
class TraceIndex:
    """Per-trace lookups: dense ids and 'next request of key after t'."""

    keys: np.ndarray

    @cached_property
    def n(self) -> int:
        return int(self.keys.size)

    @cached_property
    def _unique(self) -> tuple[np.ndarray, np.ndarray]:
        unique, dense = np.unique(self.keys, return_inverse=True)
        return unique, dense.astype(np.int64)

    @property
    def unique(self) -> np.ndarray:
        return self._unique[0]

    @property
    def dense(self) -> np.ndarray:
        return self._unique[1]

    @cached_property
    def composite(self) -> np.ndarray:
        return np.sort(self.dense * (self.n + 1) + np.arange(self.n, dtype=np.int64))

    def dense_of(self, keys: np.ndarray) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.uint64)
        position = np.searchsorted(self.unique, keys)
        position = np.minimum(position, self.unique.size - 1)
        return np.where(self.unique[position] == keys, position, -1)

    def next_after(self, keys: np.ndarray, times: np.ndarray) -> np.ndarray:
        """First request time of each key strictly after its time, or -1."""
        dense = self.dense_of(keys)
        query = dense * (self.n + 1) + np.asarray(times, dtype=np.int64) + 1
        position = np.searchsorted(self.composite, query)
        found = position < self.composite.size
        value = self.composite[np.minimum(position, self.composite.size - 1)]
        same = found & (value // (self.n + 1) == dense) & (dense >= 0)
        return np.where(same, value % (self.n + 1), -1)

    def request_counts(self) -> np.ndarray:
        return np.bincount(self.dense, minlength=self.unique.size)


def window(n: int, start: int | None, end: int | None) -> tuple[int, int]:
    first = 0 if start is None else max(0, min(int(start), n - 1))
    stop = n if end is None else max(first + 1, min(int(end), n))
    return first, stop


def edges(start: int, end: int, buckets: int) -> np.ndarray:
    count = max(1, min(int(buckets), end - start))
    return np.unique(np.linspace(start, end, count + 1).round().astype(np.int64))


def per_bucket(times: np.ndarray, bounds: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    counts, _ = np.histogram(times, bins=bounds, weights=weights)
    return counts


def bucket_mean(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Mean of a per-request array over each [bounds[i], bounds[i+1])."""
    sums = np.add.reduceat(values.astype(np.float64), bounds[:-1])
    return sums / np.diff(bounds)


def miss_series(run: Run, bounds: np.ndarray) -> np.ndarray:
    return bucket_mean(run.outcome != HIT, bounds)


def remiss_sources(run: Run, index: TraceIndex) -> np.ndarray:
    """For every request, the queue its key was last evicted from (or 255)."""
    source = np.full(index.n, NO_QUEUE, dtype=np.uint8)
    remiss = np.flatnonzero(run.outcome == REMISS)
    if remiss.size == 0 or run.evict_t.size == 0:
        return source
    evicted = index.dense_of(run.evict_key) * (index.n + 1) + run.evict_t.astype(np.int64)
    order = np.argsort(evicted)
    evicted = evicted[order]
    queue = run.evict_q[order]
    query = index.dense[remiss] * (index.n + 1) + remiss
    position = np.searchsorted(evicted, query) - 1
    valid = position >= 0
    same = valid & (evicted[np.maximum(position, 0)] // (index.n + 1) == index.dense[remiss])
    source[remiss[same]] = queue[position[same]]
    return source


def queue_profile(run: Run, index: TraceIndex, start: int, end: int, buckets: int) -> dict[str, Any]:
    bounds = edges(start, end, buckets)
    queues, ghosts = run.queues, run.ghosts
    outcome = run.outcome
    profile: dict[str, Any] = {"bounds": bounds, "queues": queues, "ghosts": ghosts}
    profile["miss_ratio"] = miss_series(run, bounds)
    source = remiss_sources(run, index)
    misses = {"cold": bucket_mean(outcome == COLD, bounds)}
    for position, name in enumerate(queues):
        misses[f"after {name} eviction"] = bucket_mean((outcome == REMISS) & (source == position), bounds)
    if (outcome == BYPASS).any():
        misses["bypassed"] = bucket_mean(outcome == BYPASS, bounds)
    profile["miss_breakdown"] = misses
    hit = outcome == HIT
    profile["hits"] = {name: bucket_mean(hit & (run.queue == i), bounds) for i, name in enumerate(queues)}
    miss = (outcome == COLD) | (outcome == REMISS)
    profile["inserts"] = {name: bucket_mean(miss & (run.queue == i), bounds) for i, name in enumerate(queues)}
    width = np.diff(bounds)
    profile["evictions"] = {
        name: per_bucket(run.evict_t[run.evict_q == i], bounds) / width for i, name in enumerate(queues)
    }
    profile["zero_hit_evictions"] = {
        name: per_bucket(run.evict_t[(run.evict_q == i) & (run.evict_hits == 0)], bounds) / width
        for i, name in enumerate(queues)
    }
    moves = {}
    is_move = run.ev_kind == MOVE
    if is_move.any():
        pairs = run.ev_a[is_move].astype(np.int64) * 16 + run.ev_b[is_move]
        times = run.ev_t[is_move]
        for pair in np.unique(pairs):
            label = f"{queues[pair // 16]}->{queues[pair % 16]}"
            moves[label] = per_bucket(times[pairs == pair], bounds) / width
    profile["moves"] = moves
    for kind, label in ((GHOST_HIT, "ghost_hits"), (GHOST_ADD, "ghost_adds"), (GHOST_DROP, "ghost_drops")):
        selected = run.ev_kind == kind
        profile[label] = {
            name: per_bucket(run.ev_t[selected & (run.ev_a == i)], bounds) / width for i, name in enumerate(ghosts)
        }
    sizes = run.sizes
    sample_t = run.sample_t.astype(np.int64)
    occupancy = {}
    for column, name in enumerate(queues + ghosts):
        inside = (sample_t >= start) & (sample_t < end)
        values = sizes[inside, column].astype(np.float64)
        times = sample_t[inside]
        sums = per_bucket(times, bounds, values)
        counts = per_bucket(times, bounds)
        with np.errstate(invalid="ignore", divide="ignore"):
            occupancy[name] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    profile["occupancy"] = occupancy
    return profile


def eviction_quality(run: Run, index: TraceIndex, start: int, end: int) -> dict[str, Any]:
    capacity = run.summary["capacity"]
    inside = (run.evict_t >= start) & (run.evict_t < end)
    times = run.evict_t[inside].astype(np.int64)
    keys = run.evict_key[inside]
    queue = run.evict_q[inside]
    hits = run.evict_hits[inside]
    residency = times - run.evict_ins[inside].astype(np.int64)
    following = index.next_after(keys, times)
    gap = np.where(following >= 0, (following - times) / capacity, np.inf)
    result = {}
    for position, name in enumerate(run.queues):
        mine = queue == position
        count = int(mine.sum())
        if not count:
            continue
        bins = np.digitize(gap[mine], GAP_BINS)
        bins[np.isinf(gap[mine])] = len(GAP_BINS) + 1
        histogram = np.bincount(bins, minlength=len(GAP_LABELS))
        hit_counts = np.bincount(np.minimum(hits[mine], 3).astype(np.int64), minlength=4)
        result[name] = {
            "evictions": count,
            "return_gap_share": {label: round(float(v) / count, 4) for label, v in zip(GAP_LABELS, histogram, strict=True)},
            "hits_while_resident_share": {
                label: round(float(v) / count, 4) for label, v in zip(("0", "1", "2", "3+"), hit_counts, strict=True)
            },
            "median_residency_requests": int(np.median(residency[mine])),
            "median_residency_in_cache_sizes": round(float(np.median(residency[mine])) / capacity, 3),
        }
    return result


def object_timeline(run: Run, index: TraceIndex, keys: list[int], start: int, end: int) -> list[dict[str, Any]]:
    keys_array = np.asarray(keys, dtype=np.uint64)
    dense = index.dense_of(keys_array)
    queues, ghosts = run.queues, run.ghosts
    timelines = []
    for key, identity in zip(keys_array.tolist(), dense.tolist(), strict=True):
        if identity < 0:
            timelines.append({"key": key, "requests": [], "residency": [], "ghost": [], "evictions": []})
            continue
        times = np.flatnonzero(index.dense == identity)
        changes: list[tuple[int, int, str]] = []  # (time, order, queue or "")
        for t in times.tolist():
            if run.outcome[t] in (COLD, REMISS):
                changes.append((t, 1, queues[run.queue[t]]))
        selected = run.ev_key == key
        ghost_open: int | None = None
        ghost_segments = []
        for t, kind, a, b in zip(run.ev_t[selected].tolist(), run.ev_kind[selected].tolist(), run.ev_a[selected].tolist(), run.ev_b[selected].tolist(), strict=True):
            if kind == MOVE:
                changes.append((t, 2, queues[b]))
            elif kind == GHOST_ADD:
                ghost_open = t
            elif kind in (GHOST_HIT, GHOST_DROP) and ghost_open is not None:
                ghost_segments.append((ghost_open, t, ghosts[a], "hit" if kind == GHOST_HIT else "dropped"))
                ghost_open = None
        if ghost_open is not None:
            ghost_segments.append((ghost_open, index.n, ghosts[0] if ghosts else "", "open"))
        evicted = run.evict_key == key
        evictions = list(zip(run.evict_t[evicted].tolist(), [queues[q] for q in run.evict_q[evicted].tolist()], run.evict_hits[evicted].tolist(), strict=True))
        for t, name, _ in evictions:
            changes.append((t, 0, ""))
        changes.sort()
        segments = []
        current: tuple[int, str] | None = None
        for t, _, name in changes:
            if current is not None:
                segments.append((current[0], t, current[1]))
            current = (t, name) if name else None
        if current is not None:
            segments.append((current[0], index.n, current[1]))
        clip = [(max(a, start), min(b, end), q) for a, b, q in segments if b > start and a < end and b > a]
        requests = [
            (t, int(run.outcome[t]), queues[run.queue[t]] if run.queue[t] != NO_QUEUE else "-")
            for t in times[(times >= start) & (times < end)].tolist()
        ]
        timelines.append(
            {
                "key": key,
                "requests_total": int(times.size),
                "requests": requests,
                "residency": clip,
                "ghost": [(max(a, start), min(b, end), g, how) for a, b, g, how in ghost_segments if b > start and a < end],
                "evictions": [(t, q, h) for t, q, h in evictions if start <= t < end],
            }
        )
    return timelines


SELECTIONS = ("most_remisses", "most_requests", "most_regressed", "most_improved", "evicted_unused_from_main")


def select_keys(
    run: Run, index: TraceIndex, mode: str, start: int, end: int, count: int, other: Run | None = None
) -> list[int]:
    """Pick objects worth a timeline: the ones carrying a window's misses."""
    span = slice(start, end)
    dense = index.dense[span]
    if mode == "most_remisses":
        weight = (run.outcome[span] == REMISS).astype(np.int64)
    elif mode == "most_requests":
        weight = np.ones(dense.size, dtype=np.int64)
    elif mode in ("most_regressed", "most_improved"):
        if other is None:
            raise ValueError(f"{mode} needs compare_to")
        mine, theirs = run.outcome[span] == HIT, other.outcome[span] == HIT
        weight = ((~mine & theirs) if mode == "most_regressed" else (mine & ~theirs)).astype(np.int64)
    elif mode == "evicted_unused_from_main":
        main = len(run.queues) - 1
        chosen = (run.evict_t >= start) & (run.evict_t < end) & (run.evict_q == main) & (run.evict_hits == 0)
        ids = index.dense_of(run.evict_key[chosen])
        totals = np.bincount(ids[ids >= 0], minlength=index.unique.size)
        top = np.argsort(-totals, kind="stable")[:count]
        return [int(index.unique[i]) for i in top if totals[i] > 0]
    else:
        raise ValueError(f"unknown selection {mode!r}")
    totals = np.bincount(dense, weights=weight, minlength=index.unique.size)
    top = np.argsort(-totals, kind="stable")[:count]
    return [int(index.unique[i]) for i in top if totals[i] > 0]


def diff_profile(base: Run, cand: Run, index: TraceIndex, start: int, end: int, buckets: int, top: int = 10) -> dict[str, Any]:
    bounds = edges(start, end, buckets)
    base_hit, cand_hit = base.outcome == HIT, cand.outcome == HIT
    regress = base_hit & ~cand_hit
    improve = ~base_hit & cand_hit
    span = slice(start, end)
    differ = np.flatnonzero(regress[span] | improve[span])
    dense = index.dense[span]
    reg_by_key = np.bincount(dense, weights=regress[span], minlength=index.unique.size)
    imp_by_key = np.bincount(dense, weights=improve[span], minlength=index.unique.size)
    requests = np.bincount(dense, minlength=index.unique.size)

    def leaders(values: np.ndarray, other_values: np.ndarray) -> list[dict[str, int]]:
        order = np.argsort(-values, kind="stable")[:top]
        return [
            {"key": int(index.unique[i]), "count": int(values[i]), "opposite": int(other_values[i]), "requests": int(requests[i])}
            for i in order
            if values[i] > 0
        ]

    return {
        "bounds": bounds,
        "base_miss": miss_series(base, bounds),
        "cand_miss": miss_series(cand, bounds),
        "regressions": bucket_mean(regress, bounds),
        "improvements": bucket_mean(improve, bounds),
        "totals": {
            "requests": end - start,
            "base_misses": int((~base_hit[span]).sum()),
            "cand_misses": int((~cand_hit[span]).sum()),
            "regressions": int(regress[span].sum()),
            "improvements": int(improve[span].sum()),
            "first_divergence": int(start + differ[0]) if differ.size else None,
        },
        "top_regressed": leaders(reg_by_key, imp_by_key),
        "top_improved": leaders(imp_by_key, reg_by_key),
    }


# -- workload analyses from the literature ---------------------------------------
#
# MQ (Zhou et al., 2001) characterized second-level traces by reuse distance and
# access frequency; S3-FIFO (Yang et al., 2023) by the one-hit-wonder ratio of
# windows the size of a cache; S4-FIFO by where in each queue hits land. These
# functions compute the same statistics exactly, from a trace or a run's log.

FREQUENCY_BINS = (1, 2, 3, 5, 9, 17, 33, 65)  # lower bounds: 1, 2, 3-4, 5-8, ..., 65+
POSITION_BINS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def frequency_label(index: int) -> str:
    low = FREQUENCY_BINS[index]
    if index == len(FREQUENCY_BINS) - 1:
        return f"{low}+"
    high = FREQUENCY_BINS[index + 1] - 1
    return str(low) if low == high else f"{low}-{high}"


def position_labels() -> list[str]:
    bounds = (0.0, *POSITION_BINS)
    return [f"{a:g}-{b:g}" for a, b in zip(bounds, bounds[1:])] + [f">={POSITION_BINS[-1]:g}"]


def stack_distances(dense: np.ndarray) -> np.ndarray:
    """LRU stack distance of every request: distinct other keys since the
    previous request to the same key (-1 for a first request). Exact; a
    Fenwick tree over request times, O(n log n)."""
    n = int(dense.size)
    tree = [0] * (n + 1)
    last: dict[int, int] = {}
    out = np.full(n, -1, dtype=np.int64)
    active = 0
    for t, key in enumerate(dense.tolist()):
        previous = last.get(key)
        if previous is not None:
            # active marks after `previous` = active - prefix(previous)
            i, prefix = previous + 1, 0
            while i > 0:
                prefix += tree[i]
                i -= i & -i
            out[t] = active - prefix
            i = previous + 1
            while i <= n:
                tree[i] -= 1
                i += i & -i
            active -= 1
        i = t + 1
        while i <= n:
            tree[i] += 1
            i += i & -i
        active += 1
        last[key] = t
    return out


def frequency_distribution(index: TraceIndex) -> dict[str, Any]:
    """Share of objects and of requests by an object's request count (MQ's Fig. 4)."""
    counts = index.request_counts()
    bins = np.digitize(counts, FREQUENCY_BINS) - 1
    objects = np.bincount(bins, minlength=len(FREQUENCY_BINS))
    requests = np.bincount(bins, weights=counts, minlength=len(FREQUENCY_BINS))
    return {
        "labels": [frequency_label(i) for i in range(len(FREQUENCY_BINS))],
        "object_share": objects / counts.size,
        "request_share": requests / counts.sum(),
    }


def reuse_profile(distances: np.ndarray, marks: dict[str, float]) -> dict[str, Any]:
    """Share of all requests that reuse a key within each marked stack distance
    (MQ's temporal-distance analysis, measured in distinct objects)."""
    n = distances.size
    reuse = distances[distances >= 0]
    sorted_reuse = np.sort(reuse)
    grid = np.unique(np.round(np.geomspace(1, max(2, int(sorted_reuse[-1]) + 1 if reuse.size else 2), 96)).astype(np.int64))
    curve = np.searchsorted(sorted_reuse, grid, side="left") / n
    within = {label: float(np.searchsorted(sorted_reuse, value, side="left")) / n for label, value in marks.items()}
    return {
        "requests": n,
        "first_requests_share": float((distances < 0).mean()),
        "grid": grid,
        "share_within": curve,  # share of requests whose stack distance < grid value
        "within_marks": within,
        "median_reuse_distance": float(np.median(reuse)) if reuse.size else None,
    }


def one_hit_wonders(dense: np.ndarray, sizes: dict[str, int], starts: int = 8) -> dict[str, Any]:
    """For windows of consecutive requests holding N distinct objects, the share
    of those objects requested exactly once (S3-FIFO's one-hit-wonder ratio at
    cache-sized windows), averaged over evenly spaced starting points."""
    n = dense.size
    result = {}
    for label, size in sizes.items():
        ratios = []
        for start in np.linspace(0, n, starts, endpoint=False).astype(np.int64).tolist():
            tail = dense[start:]
            unique, first = np.unique(tail, return_index=True)
            if unique.size < size or size < 1:
                continue
            end = int(np.sort(first)[size - 1]) + 1
            window = np.bincount(tail[:end])
            ratios.append(float((window == 1).sum()) / size)
        result[label] = {"objects": int(size), "windows": len(ratios), "one_hit_share": float(np.mean(ratios)) if ratios else None}
    return result


def hit_positions(run: Run, index: TraceIndex, start: int, end: int) -> dict[str, Any]:
    """Where in each queue hits land (S4-FIFO's hit-position statistics).

    An object's position at a hit is the number of objects that entered the
    same queue (inserted, or moved in, including reinsertions it reports as
    moves) after it last did, divided by the queue's mean occupancy: in a FIFO,
    0 is the head and about 1 the tail. Ghost hits are placed the same way in
    their ghost queue, counting ghost additions.
    """
    queues, ghosts = run.queues, run.ghosts
    n = index.n
    stride = n + 1
    sizes = run.sizes.astype(np.float64)
    labels = position_labels()
    edges_ = np.array(POSITION_BINS)
    result: dict[str, Any] = {"labels": labels, "queues": {}, "ghosts": {}}
    inserted = np.flatnonzero((run.outcome == COLD) | (run.outcome == REMISS))
    is_move = run.ev_kind == MOVE

    def place(entry_t: np.ndarray, entry_dense: np.ndarray, query_t: np.ndarray, query_dense: np.ndarray, mean_size: float):
        order = np.argsort(entry_dense * stride + entry_t, kind="stable")
        composite = (entry_dense * stride + entry_t)[order]
        position = np.searchsorted(composite, query_dense * stride + query_t, side="left") - 1
        found = (position >= 0) & (composite[np.maximum(position, 0)] // stride == query_dense)
        entered = composite[np.maximum(position, 0)] % stride
        times = np.sort(entry_t)
        between = np.searchsorted(times, query_t, side="left") - np.searchsorted(times, entered, side="right")
        depth = between[found] / max(mean_size, 1.0)
        return depth, int((~found).sum())

    def summarize(depth: np.ndarray, unmatched: int, mean_size: float) -> dict[str, Any]:
        bins = np.digitize(depth, edges_)
        counts = np.bincount(bins, minlength=len(labels))
        total = max(1, depth.size)
        return {
            "hits": int(depth.size),
            "mean_size": round(mean_size, 1),
            "share": [round(float(c) / total, 4) for c in counts],
            "median_position": round(float(np.median(depth)), 4) if depth.size else None,
            "unmatched": unmatched,
        }

    window = slice(start, end)
    for q, name in enumerate(queues):
        mine = inserted[run.queue[inserted] == q]
        moved = is_move & (run.ev_b == q)
        entry_t = np.concatenate([mine, run.ev_t[moved].astype(np.int64)])
        entry_dense = np.concatenate([index.dense[mine], index.dense_of(run.ev_key[moved])])
        hits = np.flatnonzero((run.outcome[window] == HIT) & (run.queue[window] == q)) + start
        mean_size = float(sizes[:, q].mean()) if sizes.size else 0.0
        if hits.size == 0 or entry_t.size == 0:
            result["queues"][name] = summarize(np.empty(0), 0, mean_size)
            continue
        depth, unmatched = place(entry_t, entry_dense, hits, index.dense[hits], mean_size)
        result["queues"][name] = summarize(depth, unmatched, mean_size)
    for g, name in enumerate(ghosts):
        column = len(queues) + g
        added = (run.ev_kind == GHOST_ADD) & (run.ev_a == g)
        hit = (run.ev_kind == GHOST_HIT) & (run.ev_a == g) & (run.ev_t >= start) & (run.ev_t < end)
        mean_size = float(sizes[:, column].mean()) if sizes.size else 0.0
        if not hit.any() or not added.any():
            result["ghosts"][name] = summarize(np.empty(0), 0, mean_size)
            continue
        depth, unmatched = place(
            run.ev_t[added].astype(np.int64), index.dense_of(run.ev_key[added]),
            run.ev_t[hit].astype(np.int64), index.dense_of(run.ev_key[hit]), mean_size,
        )
        result["ghosts"][name] = summarize(depth, unmatched, mean_size)
    return result
