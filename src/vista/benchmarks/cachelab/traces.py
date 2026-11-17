"""Workload traces: seeded synthetic families, oracleGeneral files, and suites.

A trace is a sequence of uint64 object keys; request i happens at virtual time
i. Traces are stored as raw little-endian uint64 files so the stdlib-only
sandbox simulator can read them without numpy.

The synthetic families each stress one known mechanism. The agent sees only
neutral names and footprint statistics, never the family or its parameters,
so it has to find the mechanism from evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

KEY_DTYPE = np.dtype("<u8")
# Keys stay below 2**40 so they survive JSON in JavaScript clients exactly.
KEY_LIMIT = 2**40
ORACLE_GENERAL = np.dtype(
    [("clock", "<u4"), ("obj_id", "<u8"), ("size", "<u4"), ("next", "<i8")]
)


def _zipf_ranks(rng: np.random.Generator, objects: int, alpha: float, n: int) -> np.ndarray:
    weights = 1.0 / np.power(np.arange(1, objects + 1, dtype=np.float64), alpha)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.searchsorted(cdf, rng.random(n), side="right").astype(np.int64)


class _KeySpace:
    """Hands out disjoint pseudo-random key ranges so families never collide."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.issued: set[int] = set()

    def take(self, count: int) -> np.ndarray:
        keys = np.empty(0, dtype=np.uint64)
        while keys.size < count:
            draw = self.rng.integers(1, 2**63, size=(count - keys.size) * 2, dtype=np.uint64)
            draw = np.unique(draw)
            draw = draw[~np.isin(draw, np.fromiter(self.issued, np.uint64, len(self.issued)))]
            self.rng.shuffle(draw)
            keys = np.concatenate([keys, draw[: count - keys.size]])
        self.issued.update(int(key) for key in keys)
        return keys


def zipf(rng, *, requests: int, objects: int, alpha: float) -> np.ndarray:
    keys = _KeySpace(rng).take(objects)
    return keys[_zipf_ranks(rng, objects, alpha, requests)]


def zipf_scan(
    rng, *, requests: int, objects: int, alpha: float, scan_every: int, scan_length: int
) -> np.ndarray:
    """Zipf traffic interrupted by sequential scans of never-reused keys."""
    space = _KeySpace(rng)
    hot = space.take(objects)
    scans = max(0, requests // (scan_every + scan_length))
    scan_keys = space.take(scans * scan_length)
    out = []
    cursor = 0
    background = hot[_zipf_ranks(rng, objects, alpha, requests)]
    position = 0
    while sum(part.size for part in out) < requests:
        out.append(background[position : position + scan_every])
        position += scan_every
        if cursor < scan_keys.size:
            out.append(scan_keys[cursor : cursor + scan_length])
            cursor += scan_length
    return np.concatenate(out)[:requests]


def loop_mix(
    rng, *, requests: int, objects: int, alpha: float, loop_size: int, loop_share: float
) -> np.ndarray:
    """Zipf traffic mixed with a cyclic loop over a fixed set of keys."""
    space = _KeySpace(rng)
    hot = space.take(objects)
    loop = space.take(loop_size)
    from_loop = rng.random(requests) < loop_share
    out = hot[_zipf_ranks(rng, objects, alpha, requests)]
    out[from_loop] = loop[np.arange(int(from_loop.sum())) % loop_size]
    return out


def shift(
    rng, *, requests: int, objects: int, alpha: float, phases: int, overlap: float
) -> np.ndarray:
    """Popularity moves abruptly to a mostly new hot set at each phase."""
    space = _KeySpace(rng)
    keep = int(objects * overlap)
    current = space.take(objects)
    parts = []
    bounds = np.linspace(0, requests, phases + 1).astype(int)
    for phase in range(phases):
        if phase:
            survivors = rng.permutation(current)[:keep]
            current = np.concatenate([survivors, space.take(objects - keep)])
            rng.shuffle(current)
        length = bounds[phase + 1] - bounds[phase]
        parts.append(current[_zipf_ranks(rng, objects, alpha, length)])
    return np.concatenate(parts)


def burst(
    rng,
    *,
    requests: int,
    objects: int,
    alpha: float,
    burst_share: float,
    burst_hits: int,
    burst_gap: int,
) -> np.ndarray:
    """Background Zipf plus new keys reused a few times at a medium distance.

    Each bursty key is requested `burst_hits` times with gaps around
    `burst_gap` requests, then never again: reuse that a small probationary
    queue can miss but the whole cache could hold.
    """
    space = _KeySpace(rng)
    hot = space.take(objects)
    out = hot[_zipf_ranks(rng, objects, alpha, requests)]
    starts = int(requests * burst_share / burst_hits)
    keys = space.take(starts)
    slots = []
    for index in range(starts):
        start = int(rng.integers(0, requests))
        gaps = rng.integers(max(1, burst_gap // 2), burst_gap * 3 // 2 + 1, size=burst_hits - 1)
        times = start + np.concatenate([[0], np.cumsum(gaps)])
        slots.append((times[times < requests], keys[index]))
    for times, key in slots:
        out[times] = key
    return out


def tenants(rng, *, requests: int, a: Mapping[str, Any], b: Mapping[str, Any], share: float) -> np.ndarray:
    """Interleave two independent families at a fixed request share."""
    first = generate(rng, dict(a), requests)
    second = generate(rng, dict(b), requests)
    pick = rng.random(requests) < share
    out = first.copy()
    out[pick] = second[: int(pick.sum())]
    out[~pick] = first[: int((~pick).sum())]
    return out


FAMILIES: dict[str, Callable[..., np.ndarray]] = {
    "zipf": zipf,
    "zipf_scan": zipf_scan,
    "loop_mix": loop_mix,
    "shift": shift,
    "burst": burst,
    "tenants": tenants,
}


def generate(rng: np.random.Generator, spec: Mapping[str, Any], requests: int) -> np.ndarray:
    params = dict(spec)
    family = params.pop("family")
    return FAMILIES[family](rng, requests=requests, **params).astype(KEY_DTYPE)


@dataclass(frozen=True)
class TraceSpec:
    name: str
    split: str  # "dev" or "heldout"
    family: str
    params: Mapping[str, Any] = field(default_factory=dict)
    requests: int = 200_000
    seed: int = 0
    source: str | None = None  # an oracleGeneral file instead of a generator

    def manifest(self) -> dict[str, Any]:
        return {
            "key_format": "random-below-2^40-v1",
            "name": self.name,
            "split": self.split,
            "family": self.family,
            "params": dict(self.params),
            "requests": self.requests,
            "seed": self.seed,
            "source": self.source,
        }


@dataclass(frozen=True)
class Trace:
    spec: TraceSpec
    path: Path
    sha256: str
    requests: int
    objects: int
    one_hit_share: float

    def keys(self) -> np.ndarray:
        return np.fromfile(self.path, dtype=KEY_DTYPE)

    def public(self) -> dict[str, Any]:
        """What the agent may know: identity and footprint, not the generator."""
        return {
            "trace": self.spec.name,
            "requests": self.requests,
            "unique_objects": self.objects,
            "one_hit_object_share": round(self.one_hit_share, 4),
        }


# Default suites. Held-out keeps every dev family with new seeds and
# parameters, and adds two families the agent never sees.
DEV_SUITE = (
    TraceSpec("dev-1", "dev", "zipf_scan", {"objects": 20_000, "alpha": 0.9, "scan_every": 20_000, "scan_length": 6_000}, seed=11),
    TraceSpec("dev-2", "dev", "burst", {"objects": 30_000, "alpha": 0.8, "burst_share": 0.35, "burst_hits": 3, "burst_gap": 1_500}, seed=12),
    TraceSpec("dev-3", "dev", "shift", {"objects": 15_000, "alpha": 1.0, "phases": 4, "overlap": 0.2}, seed=13),
    TraceSpec("dev-4", "dev", "loop_mix", {"objects": 25_000, "alpha": 0.85, "loop_size": 1_800, "loop_share": 0.3}, seed=14),
)
HELDOUT_SUITE = (
    TraceSpec("ho-1", "heldout", "zipf_scan", {"objects": 40_000, "alpha": 0.8, "scan_every": 30_000, "scan_length": 10_000}, seed=101),
    TraceSpec("ho-2", "heldout", "burst", {"objects": 20_000, "alpha": 0.9, "burst_share": 0.25, "burst_hits": 4, "burst_gap": 3_000}, seed=102),
    TraceSpec("ho-3", "heldout", "shift", {"objects": 25_000, "alpha": 0.9, "phases": 6, "overlap": 0.4}, seed=103),
    TraceSpec("ho-4", "heldout", "loop_mix", {"objects": 15_000, "alpha": 1.0, "loop_size": 900, "loop_share": 0.2}, seed=104),
    TraceSpec("ho-5", "heldout", "zipf", {"objects": 50_000, "alpha": 0.7}, seed=105),
    TraceSpec(
        "ho-6",
        "heldout",
        "tenants",
        {
            "a": {"family": "zipf", "objects": 20_000, "alpha": 1.0},
            "b": {"family": "burst", "objects": 10_000, "alpha": 0.6, "burst_share": 0.5, "burst_hits": 2, "burst_gap": 800},
            "share": 0.4,
        },
        seed=106,
    ),
)
SUITES = {"dev": DEV_SUITE, "heldout": HELDOUT_SUITE}

# Cache sizes as fractions of each trace's unique objects.
DEFAULT_CAPACITY_FRACTIONS = (0.005, 0.01, 0.03, 0.1)
MIN_CAPACITY = 32


def capacities(objects: int, fractions: tuple[float, ...] = DEFAULT_CAPACITY_FRACTIONS) -> list[int]:
    return sorted({max(MIN_CAPACITY, int(round(objects * fraction))) for fraction in fractions})


def read_oracle_general(path: Path, limit: int | None = None) -> np.ndarray:
    """Object ids from a libCacheSim oracleGeneral file (optionally .zst).

    With a limit, only that prefix is decompressed. Multi-frame zstd files are
    read across frames, and short reads are retried until the prefix is full.
    """
    want = None if limit is None else limit * ORACLE_GENERAL.itemsize
    with path.open("rb") as stream:
        if path.suffix == ".zst":
            try:
                import zstandard
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Reading .zst traces needs the zstandard package") from exc
            reader = zstandard.ZstdDecompressor().stream_reader(stream, read_across_frames=True)
        else:
            reader = stream
        if want is None:
            data = reader.read()
        else:
            chunks, remaining = [], want
            while remaining > 0:
                chunk = reader.read(min(remaining, 64 << 20))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
    usable = len(data) - len(data) % ORACLE_GENERAL.itemsize
    records = np.frombuffer(data[:usable], dtype=ORACLE_GENERAL)
    return records["obj_id"].astype(KEY_DTYPE)


def remap_keys(keys: np.ndarray, seed: int) -> np.ndarray:
    """Replace every key by a distinct random id below KEY_LIMIT.

    Random rather than first-appearance ids: sequential ids would tell a
    policy for free whether an object was ever seen before.
    """
    unique, inverse = np.unique(keys, return_inverse=True)
    rng = np.random.default_rng([seed, 0x5EED])
    ids = np.empty(0, dtype=np.uint64)
    while ids.size < unique.size:
        draw = rng.integers(1, KEY_LIMIT, size=int(unique.size * 1.2) + 16, dtype=np.uint64)
        ids = np.unique(np.concatenate([ids, draw]))
    rng.shuffle(ids)
    return ids[: unique.size][inverse].astype(KEY_DTYPE)


def _stats(keys: np.ndarray) -> tuple[int, float]:
    _, counts = np.unique(keys, return_counts=True)
    return int(counts.size), float((counts == 1).sum() / max(1, counts.size))


def materialize(spec: TraceSpec, directory: Path) -> Trace:
    """Generate (or convert) a trace once; later calls reuse the stored file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.name}.bin"
    meta_path = directory / f"{spec.name}.json"
    manifest = spec.manifest()
    if path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("spec") == manifest:
            return Trace(spec, path, meta["sha256"], meta["requests"], meta["objects"], meta["one_hit_share"])
    if spec.source is not None:
        keys = read_oracle_general(Path(spec.source), spec.requests or None)
    else:
        keys = generate(np.random.default_rng(spec.seed), {"family": spec.family, **spec.params}, spec.requests)
    keys = np.ascontiguousarray(remap_keys(keys, spec.seed), dtype=KEY_DTYPE)
    data = keys.tobytes()
    path.write_bytes(data)
    objects, one_hit = _stats(keys)
    sha = hashlib.sha256(data).hexdigest()
    meta_path.write_text(
        json.dumps(
            {"spec": manifest, "sha256": sha, "requests": int(keys.size), "objects": objects, "one_hit_share": one_hit},
            indent=2,
        ),
        encoding="utf-8",
    )
    return Trace(spec, path, sha, int(keys.size), objects, one_hit)


def materialize_suite(specs: tuple[TraceSpec, ...], directory: Path, *, workers: int = 1) -> list[Trace]:
    if workers <= 1 or len(specs) <= 1:
        return [materialize(spec, directory) for spec in specs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        return list(pool.map(materialize, specs, [directory] * len(specs)))


# -- real-trace suites -------------------------------------------------------------


def _dataset_files(root: Path, dataset: str) -> list[Path]:
    return sorted(path for path in (root / dataset).rglob("*") if path.is_file() and "oracleGeneral" in path.name)


def count_requests(path: Path, stop_after: int | None = None) -> int:
    """Requests in an oracleGeneral file (optionally .zst). Counting stops once
    it passes `stop_after`, so a huge file costs no more than that prefix."""
    size = ORACLE_GENERAL.itemsize
    if path.suffix != ".zst":
        return path.stat().st_size // size
    import zstandard

    total = 0
    with path.open("rb") as stream:
        reader = zstandard.ZstdDecompressor().stream_reader(stream, read_across_frames=True)
        while chunk := reader.read(64 << 20):
            total += len(chunk)
            if stop_after is not None and total // size > stop_after:
                break
    return total // size


def sample_real_suite(
    root: Path,
    *,
    split: str,
    datasets: list[str],
    per_dataset: int,
    requests: int,
    seed: int,
    exclude: set[str] = frozenset(),
    max_bytes: int | None = None,
    neutral_names: bool = False,
    full_length: tuple[int, int] | None = None,
    name_prefix: str | None = None,
) -> tuple[TraceSpec, ...]:
    """Stratified, reproducible sample of oracleGeneral traces: `per_dataset` files
    from each dataset, each cut to its first `requests` requests.

    With `full_length=(low, high)` traces are used whole instead, and only files
    with low <= requests <= high are chosen (files are tried in a seeded random
    order; a dataset with fewer such files contributes fewer).
    `exclude` holds source paths already used by another split. Dev names are
    neutral when `neutral_names` is set, so the agent does not see the source.
    """
    import random

    specs = []
    prefix = name_prefix or ("dev" if split == "dev" else "ho")
    for dataset in datasets:
        files = [path for path in _dataset_files(root, dataset) if str(path) not in exclude]
        if max_bytes is not None:
            files = [path for path in files if path.stat().st_size <= max_bytes] or files
        if full_length is None:
            chosen = [(path, requests) for path in random.Random(f"{seed}:{split}:{dataset}").sample(files, min(per_dataset, len(files)))]
        else:
            low, high = full_length
            chosen = []
            for path in random.Random(f"{seed}:{split}:{dataset}:full").sample(files, len(files)):
                if len(chosen) == per_dataset:
                    break
                length = count_requests(path, stop_after=high)
                if low <= length <= high:
                    chosen.append((path, length))
        for index, (path, length) in enumerate(sorted(chosen), start=1):
            name = f"{prefix}-r{len(specs) + 1}" if neutral_names else f"{prefix}-{dataset}-{index}"
            specs.append(
                TraceSpec(
                    name,
                    split,
                    "oracleGeneral",
                    {"dataset": dataset, "file": path.name},
                    length,
                    seed=int(hashlib.sha256(str(path).encode()).hexdigest()[:8], 16),
                    source=str(path),
                )
            )
    return tuple(specs)


def save_suite(path: Path, specs: tuple[TraceSpec, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([spec.manifest() for spec in specs], indent=1) + "\n", encoding="utf-8")


def load_suite(path: Path) -> tuple[TraceSpec, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        TraceSpec(row["name"], row["split"], row["family"], row["params"], row["requests"], row["seed"], row["source"])
        for row in rows
    )


def files_suite(paths: list[Path], split: str, requests: int | None = None) -> tuple[TraceSpec, ...]:
    """Real oracleGeneral traces named by position, e.g. for a held-out split."""
    prefix = "dev" if split == "dev" else "ho"
    return tuple(
        TraceSpec(f"{prefix}-file-{index}", split, "oracleGeneral", {"file": path.name}, requests or 0, 0, str(path))
        for index, path in enumerate(paths, start=1)
    )


def split_trace(full: Trace, fraction: float, directory: Path, name: str = "target") -> tuple[Trace, int]:
    """The first `fraction` of an already materialized trace, as a trace of its own.

    Keys are sliced from the full trace's remapped keys, so an object has the
    same id in the prefix and in the full trace. The prefix carries a neutral
    name and no source: it is all the engineer role sees.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    keys = full.keys()
    split = int(keys.size * fraction)
    if split < 1 or split >= keys.size:
        raise ValueError("the split leaves no requests on one side")
    prefix = np.ascontiguousarray(keys[:split])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.bin"
    data = prefix.tobytes()
    path.write_bytes(data)
    objects, one_hit = _stats(prefix)
    spec = TraceSpec(name, "dev", "prefix", {"fraction": fraction}, requests=split, seed=0)
    return Trace(spec, path, hashlib.sha256(data).hexdigest(), split, objects, one_hit), split
