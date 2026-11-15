"""Published algorithms from libCacheSim, scored on the part of a trace after a split.

The engineer role decides from the first part of a trace and is scored on the
rest. Every algorithm here runs over the whole trace from a cold cache, like
the agent's policy, and only misses after the split count. Two deterministic
runs give that number at C speed: misses over the whole trace minus misses
over the prefix alone.

Results are cached per (trace content, split, capacity, algorithm) as JSON.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

# Object miss ratio throughout: every object has size 1.
ALGORITHMS: dict[str, Callable[[Any, int], Any]] = {
    "LRU": lambda lcs, c: lcs.LRU(c),
    "FIFO": lambda lcs, c: lcs.FIFO(c),
    "CLOCK": lambda lcs, c: lcs.Clock(c),
    "SIEVE": lambda lcs, c: lcs.Sieve(c),
    "ARC": lambda lcs, c: lcs.ARC(c),
    "LIRS": lambda lcs, c: lcs.LIRS(c),
    "2Q": lambda lcs, c: lcs.TwoQ(c),
    "CLOCK-Pro": lambda lcs, c: lcs.ClockPro(c),
    "MQ": lambda lcs, c: lcs.MQ(c),
    "S3-FIFO": lambda lcs, c: lcs.S3FIFO(c),
    "W-TinyLFU": lambda lcs, c: lcs.WTinyLFU(c),
    "LeCaR": lambda lcs, c: lcs.LeCaR(c),
    "Cacheus": lambda lcs, c: lcs.Cacheus(c),
    "LHD": lambda lcs, c: lcs.LHD(c),
    "3L-Cache": lambda lcs, c: lcs.ThreeLCache(c, objective="object-miss-ratio"),
    "Belady": lambda lcs, c: lcs.Belady(c),
}
OFFLINE = frozenset({"Belady"})  # uses future knowledge: a bound, never a competitor
VERSION = "zoo-v1"


def write_oracle_general(keys: np.ndarray, path: Path) -> None:
    """libCacheSim oracleGeneral records (<u4 time, <u8 id, <u4 size, <i8 next
    access time or -1), sizes 1 and virtual time = request index."""
    n = keys.size
    order = np.lexsort((np.arange(n), keys))
    same = keys[order][1:] == keys[order][:-1]
    following = np.full(n, -1, dtype=np.int64)
    following[order[:-1][same]] = order[1:][same]
    records = np.zeros(n, dtype=np.dtype([("t", "<u4"), ("id", "<u8"), ("size", "<u4"), ("next", "<i8")]))
    records["t"] = np.arange(n, dtype=np.uint32)
    records["id"] = keys
    records["size"] = 1
    records["next"] = following
    assert records.dtype.itemsize == struct.calcsize("<IQIq")
    path.write_bytes(records.tobytes())


def _misses(algorithm: str, trace_file: str, capacity: int, limit: int) -> int:
    import libcachesim as lcs

    reader = lcs.TraceReader(trace_file, lcs.TraceType.ORACLE_GENERAL_TRACE)
    cache = ALGORITHMS[algorithm](lcs, capacity)
    ratio, _ = cache.process_trace(reader, 0, limit)
    return int(round(ratio * limit))


def _job(args: tuple[str, str, int, int, int, str]) -> dict[str, Any]:
    algorithm, trace_file, capacity, split, total, out = args
    try:
        prefix = _misses(algorithm, trace_file, capacity, split)
        whole = _misses(algorithm, trace_file, capacity, total)
        result = {"algorithm": algorithm, "capacity": capacity, "prefix_misses": prefix, "misses": whole, "suffix_miss_ratio": (whole - prefix) / (total - split)}
    except Exception as exc:  # an algorithm may reject a configuration
        result = {"algorithm": algorithm, "capacity": capacity, "error": f"{type(exc).__name__}: {exc}"}
    Path(out).write_text(json.dumps(result), encoding="utf-8")
    return result


def suffix_miss_ratios(
    keys: np.ndarray,
    split: int,
    capacities: Sequence[int],
    directory: Path,
    *,
    algorithms: Sequence[str] = tuple(ALGORITHMS),
    workers: int = 1,
) -> dict[int, dict[str, dict[str, Any]]]:
    """{capacity: {algorithm: result}} for the requests after `split`."""
    digest = hashlib.sha256(keys.tobytes()).hexdigest()[:16]
    root = directory / f"{digest}-{split}"
    root.mkdir(parents=True, exist_ok=True)
    trace_file = root / "trace.oracleGeneral"
    results: dict[int, dict[str, dict[str, Any]]] = {capacity: {} for capacity in capacities}
    pending = []
    for capacity in capacities:
        for algorithm in algorithms:
            out = root / f"{VERSION}-{algorithm}-{capacity}.json"
            if out.is_file():
                results[capacity][algorithm] = json.loads(out.read_text(encoding="utf-8"))
            else:
                pending.append((algorithm, str(trace_file), int(capacity), int(split), int(keys.size), str(out)))
    if pending:
        if not trace_file.is_file():
            write_oracle_general(np.asarray(keys, dtype=np.uint64), trace_file)
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                done = list(pool.map(_job, pending))
        else:
            done = [_job(job) for job in pending]
        for result in done:
            results[result["capacity"]][result["algorithm"]] = result
    return results
