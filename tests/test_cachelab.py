"""CacheLab: simulator fidelity, evaluator integrity, and the VISTA task wiring.

Uses a small trace suite and the local sandbox so it runs in about a minute.
The scripted session test drives the real TaskRuntime tool path (schemas,
evidence archiving, inspect/read_pixels/history, submission) without a model.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

from vista.benchmarks.cachelab import traces
from vista.benchmarks.cachelab.engine import (
    BASELINE_SOURCE,
    SANDBOX_DIR,
    LocalSandbox,
    Run,
)
from vista.benchmarks.cachelab.lab import BASELINE, Lab, LabError
from vista.benchmarks.cachelab.policy_check import check_policy
from vista.benchmarks.cachelab.profiles import make_experiment, make_profile
from vista.benchmarks.cachelab.run import score
from vista.benchmarks.cachelab.session import CacheLabSession
from vista.core.profiles import Phase
from vista.core.runtime import Status, TaskRuntime

DEV = (
    traces.TraceSpec("dev-1", "dev", "burst", {"objects": 4000, "alpha": 0.8, "burst_share": 0.3, "burst_hits": 3, "burst_gap": 400}, requests=30_000, seed=1),
    traces.TraceSpec("dev-2", "dev", "zipf_scan", {"objects": 3000, "alpha": 0.9, "scan_every": 5000, "scan_length": 1500}, requests=30_000, seed=2),
)
HELDOUT = (
    traces.TraceSpec("ho-1", "heldout", "loop_mix", {"objects": 3000, "alpha": 0.9, "loop_size": 200, "loop_share": 0.2}, requests=30_000, seed=3),
)
BIGGER_SMALL = "from s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n    SMALL_RATIO = 0.25\n"
UNBOUNDED_HISTORY = '''from s3fifo import S3FIFO


class CachePolicy(S3FIFO):
    def __init__(self, capacity, ctx):
        super().__init__(capacity, ctx)
        self.seen = {}

    def on_miss(self, key, now):
        self.seen[key] = now
        return super().on_miss(key, now)
'''
UNSEEDED_RANDOM = '''import random
from s3fifo import S3FIFO


class CachePolicy(S3FIFO):
    def on_hit(self, key, now):
        if random.random() < 0.5:
            super().on_hit(key, now)
'''


@pytest.fixture(scope="module")
def lab(tmp_path_factory) -> Lab:
    return Lab(tmp_path_factory.mktemp("lab"), LocalSandbox(workers=4), dev=DEV, heldout=HELDOUT)


def test_baseline_matches_libcachesim(lab: Lab) -> None:
    libcachesim = pytest.importorskip("libcachesim")
    base = lab.experiment(BASELINE)
    for (name, capacity), run in base.cells.items():
        cache = libcachesim.S3FIFO(capacity)
        request = libcachesim.Request()
        request.obj_size = 1
        misses = 0
        for key in lab.dev[name].keys().tolist():
            request.obj_id = key
            misses += not cache.get(request)
        assert run.summary["misses"] == misses, (name, capacity)
        assert run.summary["instrumentation_errors"] == 0
        assert run.summary["size_mismatches"] == 0


def _libcachesim_misses(cache, keys: list[int]) -> int:
    import libcachesim

    request = libcachesim.Request()
    request.obj_size = 1
    misses = 0
    for key in keys:
        request.obj_id = key
        misses += not cache.get(request)
    return misses


def test_mq_matches_libcachesim(lab: Lab) -> None:
    libcachesim = pytest.importorskip("libcachesim")
    source = (SANDBOX_DIR / "mq.py").read_text()
    # A short lifetime so demotion (Adjust) happens often on these small traces.
    source = source.replace("LIFETIME = 10000", "LIFETIME = 300")
    cells = [(trace, capacity) for name, trace in lab.dev.items() for capacity in lab.capacities[name]]
    runs = lab.store.simulate(source, cells)
    for (name, capacity), run in runs.items():
        expected = _libcachesim_misses(libcachesim.MQ(capacity, lifetime=300), lab.dev[name].keys().tolist())
        assert run.summary["misses"] == expected, (name, capacity)
        assert run.summary["instrumentation_errors"] == 0
        assert run.summary["size_mismatches"] == 0


@pytest.mark.parametrize("name, published", [("clock", "CLOCK"), ("sieve", "SIEVE"), ("twoq", "2Q"), ("arc", "ARC"), ("lirs", "LIRS"), ("wtinylfu", "W-TinyLFU")])
def test_published_ports_match_libcachesim(tmp_path, name, published) -> None:
    """The sandbox's Python ports miss like libCacheSim's originals (W-TinyLFU's sketch differs in detail)."""
    libcachesim = pytest.importorskip("libcachesim")
    import importlib

    from vista.benchmarks.cachelab import zoo
    from vista.benchmarks.cachelab.engine import SANDBOX_DIR

    if str(SANDBOX_DIR) not in sys.path:
        sys.path.append(str(SANDBOX_DIR))
    from policy_api import Context

    rng = np.random.default_rng(1)
    keys = (rng.zipf(1.2, size=60_000) % 8_000).astype(np.int64)
    keys[rng.choice(keys.size, size=keys.size // 5, replace=False)] = np.arange(10**6, 10**6 + keys.size // 5)  # one-hit scan
    path = tmp_path / "trace.oracleGeneral"
    zoo.write_oracle_general(keys, path)
    for capacity in (200, 800):  # libCacheSim's W-TinyLFU has no window below 100 objects (1%)
        policy, resident, mine = importlib.import_module(name).CachePolicy(capacity, Context()), set(), 0
        for now, key in enumerate(keys.tolist()):
            if key in resident:
                policy.on_hit(key, now)
                continue
            mine += 1
            if policy.on_miss(key, now):
                while len(resident) >= capacity:
                    resident.remove(policy.evict(now))
                policy.insert(key, now)
                resident.add(key)
            assert sum(policy.sizes()[queue] for queue in policy.QUEUES) == len(resident)
        cache = zoo.ALGORITHMS[published](libcachesim, capacity)
        theirs = sum(not cache.get(request) for request in libcachesim.TraceReader(str(path), libcachesim.TraceType.ORACLE_GENERAL_TRACE))
        assert abs(mine - theirs) / theirs < (0.05 if name == "wtinylfu" else 0.005), (name, capacity, mine, theirs)


def test_published_ports_take_over_a_live_cache(lab: Lab) -> None:
    from vista.benchmarks.cachelab.engine import PUBLISHED_PORTS

    trace = lab.dev["dev-2"]
    cells = [(trace, capacity) for capacity in lab.capacities["dev-2"]]
    for name in PUBLISHED_PORTS:
        result = lab.store.simulate(f"from {name} import CachePolicy\n", cells, verified=True, handover=0.5)
        assert all(isinstance(run, Run) for run in result.values()), (name, result)


def test_published_ports_fit_the_memory_limit_with_room() -> None:
    """Measured like the sandbox (tracemalloc, allocations in the sandbox's files) on a
    cache big enough that per-object state dominates: about 0.6-1.4x S3-FIFO."""
    import importlib
    import tracemalloc

    from vista.benchmarks.cachelab.engine import PUBLISHED_PORTS

    if str(SANDBOX_DIR) not in sys.path:
        sys.path.append(str(SANDBOX_DIR))
    from policy_api import Context

    keys = (np.random.default_rng(2).zipf(1.1, size=120_000) % 60_000).astype(np.int64).tolist()

    def footprint(module: str, capacity: int = 5_000) -> int:
        tracemalloc.start()
        policy, resident = importlib.import_module(module).CachePolicy(capacity, Context()), set()
        for now, key in enumerate(keys):
            if key in resident:
                policy.on_hit(key, now)
            elif policy.on_miss(key, now):
                while len(resident) >= capacity:
                    resident.remove(policy.evict(now))
                policy.insert(key, now)
                resident.add(key)
        snapshot = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(True, f"{SANDBOX_DIR}/*")])
        tracemalloc.stop()
        return sum(stat.size for stat in snapshot.statistics("filename"))

    base = footprint("s3fifo")
    ratios = {name: footprint(name) / base for name in PUBLISHED_PORTS}
    ratios["vq"] = footprint("vq") / base
    assert all(ratio <= 1.6 for ratio in ratios.values()), ratios


VQ_DESIGNS = {  # vqsim config strings: a skip region, a two-level main, D-FR and AGE stages
    "skip": "maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.03,0,0,1,1,-1,0 st=0.07,1,2,2,0,-1,1 st=0,1,1,2,2,-1,0",
    "hot": "maxf=5 entry=0 overflow=2 ghost=0.5 gentry=1 gfreq=1 st=0.1,1,2,2,0,-1,1 st=0.3,1,1,1,3,2,0 st=0,1,2,1,2,-1,0",
    "timed": "maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,1,1,0,-1,1,0.05 st=0,1,1,1,2,-1,0,0.05,1.0",
}


def _vq_source(config: str) -> str:
    return f"from vq import design\n\nCachePolicy = design({config!r})\n"


def test_vq_defaults_are_s3fifo_and_designs_take_over(lab: Lab) -> None:
    cells = [(trace, capacity) for name, trace in lab.dev.items() for capacity in lab.capacities[name]]
    base = lab.experiment(BASELINE).cells
    for handover in (None, 0.3):
        same = lab.store.simulate("from vq import VQ as CachePolicy\n", cells, handover=handover)
        for cell, run in same.items():
            assert np.array_equal(run.outcome, base[cell].outcome), (cell, handover)
    for name, config in VQ_DESIGNS.items():
        for verified in (False, True):
            result = lab.store.simulate(_vq_source(config), cells, handover=0.3, verified=verified)
            for cell, run in result.items():
                assert isinstance(run, Run), (name, run)
                assert run.summary["instrumentation_errors"] == 0 and run.summary["size_mismatches"] == 0, (name, cell)
        assert any(result[cell].summary["misses"] != base[cell].summary["misses"] for cell in result), name
    # Online re-choice among the designs (exact shadows, a short period so it switches).
    online = (
        "from vq import OnlineVQ\n\n\nclass CachePolicy(OnlineVQ):\n"
        f"    CANDIDATES = {('maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0', *VQ_DESIGNS.values())!r}\n"
        "    PERIOD = 2000\n    RATE = 1.0\n    START = 1\n"
    )
    for handover in (None, 0.3):
        result = lab.store.simulate(online, cells, handover=handover)
        for cell, run in result.items():
            assert isinstance(run, Run), (handover, run)
            assert run.summary["instrumentation_errors"] == 0 and run.summary["size_mismatches"] == 0, (handover, cell)


def test_vq_v1_is_the_frozen_portfolio_and_runs() -> None:
    if str(SANDBOX_DIR) not in sys.path:
        sys.path.append(str(SANDBOX_DIR))
    from policy_api import Context
    from vq import S3_CONFIG, V1_CANDIDATES, OnlineV1

    frozen = json.loads((Path(__file__).parents[1] / "scripts/vqsim/v1/portfolio.json").read_text())["portfolio"]
    assert list(V1_CANDIDATES) == frozen and len(frozen) == 32 and frozen[0] == S3_CONFIG
    assert len(OnlineV1.DESIGNS) == 32 and OnlineV1.QUEUES == tuple(f"Q{i}" for i in range(len(OnlineV1.QUEUES)))
    dense = _shifting_trace(24_000, 8, 3)
    policy, resident = type("Short", (OnlineV1,), {"PERIOD": 600, "START": 2})(1_000, Context()), set()
    for now, key in enumerate(dense.tolist()):
        if key in resident:
            policy.on_hit(key, now)
            continue
        policy.on_miss(key, now)
        while len(resident) >= 1_000:
            resident.remove(policy.evict(now))
        policy.insert(key, now)
        resident.add(key)
        assert len(resident) == sum(policy.sizes()[name] for name in policy.QUEUES)
    assert policy.current >= 0 and policy.rate == pytest.approx(0.2)


def test_vqonline_c_matches_online_v1(tmp_path) -> None:
    """scripts/vqsim/vqonline.h (keyed, deployable C) makes OnlineVQ's decisions exactly, with
    object records bounded by the cache and ghost sizes."""
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    binary = tmp_path / "vqonline"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(Path(__file__).parents[1] / "scripts/vqsim/vqonline.c")], check=True)
    if str(SANDBOX_DIR) not in sys.path:
        sys.path.append(str(SANDBOX_DIR))
    from policy_api import Context
    from vq import V1_CANDIDATES, OnlineV1, design

    keys = (_shifting_trace(60_000, 8, 5).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(4)
    (tmp_path / "keys.u64").write_bytes(keys.astype("<u8").tobytes())
    (tmp_path / "candidates").write_text("# v1\n" + "\n".join(V1_CANDIDATES) + "\n")
    max_ghost = max(design(c).GHOST_RATIO for c in V1_CANDIDATES)
    switched = 0
    cases = [
        (500, {"PERIOD": 3000, "RATE": 0.05, "MIN_MINI": 50}),
        (2000, {"PERIOD": 2500, "START": 3, "RATE": 0.02, "MIN_MINI": 100}),
        (6000, {"PERIOD": 4000, "START": 2, "RATE": 1.0}),  # exact shadows
        (1500, {"PERIOD": 2000, "RATE": 0.1, "MIN_MINI": 50, "MARGIN": 0.02}),
    ]
    for capacity, knobs in cases:
        policy, resident, misses = type("Short", (OnlineV1,), knobs)(capacity, Context()), set(), 0
        for now, key in enumerate(keys.tolist()):
            if key in resident:
                policy.on_hit(key, now)
                continue
            misses += 1
            policy.on_miss(key, now)
            while len(resident) >= capacity:
                resident.remove(policy.evict(now))
            policy.insert(key, now)
            resident.add(key)
        args = [f"{name}={getattr(policy, attr)}" for name, attr in [("period", "PERIOD"), ("start", "START"), ("from", "FROM_WINDOW"), ("rate", "RATE"), ("min", "MIN_MINI"), ("margin", "MARGIN")]]
        out = subprocess.run([str(binary), str(tmp_path / "keys.u64"), str(tmp_path / "candidates"), str(capacity), *args], capture_output=True, text=True, check=True).stdout.split()
        assert [int(x) for x in out[:3]] == [misses, policy.switches, policy.current], (capacity, knobs)
        assert int(out[4]) <= capacity * (1 + max_ghost) + 1, out  # records are reused
        switched = max(switched, policy.switches)
    assert switched >= 3


def test_vqonline_v1_header_is_the_frozen_portfolio() -> None:
    import re

    root = Path(__file__).parents[1]
    frozen = json.loads((root / "scripts/vqsim/v1/portfolio.json").read_text())["portfolio"]
    header = (root / "scripts/vqsim/vqonline_v1.h").read_text()
    assert re.findall(r'^\s*"([^"]*)",$', header, re.M) == frozen
    assert f"#define VQO_V1_K {len(frozen)}" in header


def _shifting_trace(n: int, segments: int, seed: int) -> np.ndarray:
    """Dense ids whose popularity and scan share change every n / segments requests."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(segments):
        a, m = [(1.3, 3000), (0.9, 6000), (1.2, 2000), (1.0, 9000)][i % 4]
        size = n // segments
        keys = (rng.zipf(1.0 + a, size=size) % m).astype(np.int64) if a > 1 else rng.integers(0, m, size=size)
        keys = (keys * 7 + (i % 2) * 100_000) % 20_011 + (i % 2) * 50_000
        scan = rng.random(size) < (0.5 if i % 2 else 0.0)
        keys[scan] = 10**6 + i * n + np.arange(scan.sum())
        parts.append(keys)
    return np.unique(np.concatenate(parts), return_inverse=True)[1]


def test_vq_matches_vqsim(tmp_path) -> None:
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    binary = tmp_path / "vqsim"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(Path(__file__).parents[1] / "scripts/vqsim/vqsim.c")], check=True)
    if str(SANDBOX_DIR) not in sys.path:
        sys.path.append(str(SANDBOX_DIR))
    from policy_api import Context
    from vq import design

    rng = np.random.default_rng(4)
    keys = (rng.zipf(1.15, size=40_000) % 8_000).astype(np.int64)
    keys[rng.choice(keys.size, keys.size // 6, replace=False)] = np.arange(10**6, 10**6 + keys.size // 6)
    _, dense = np.unique(keys, return_inverse=True)
    (tmp_path / "ids.u32").write_bytes(dense.astype(np.uint32).tobytes())
    split = keys.size // 5
    cases = [(capacity, config) for capacity in (100, 900) for config in ("maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0", *VQ_DESIGNS.values())]
    (tmp_path / "jobs").write_text("".join(f"{capacity} {split} 0 {config}\n" for capacity, config in cases))
    lines = subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True, check=True).stdout.splitlines()
    for (capacity, config), line in zip(cases, lines, strict=True):
        policy, resident, misses = design(config)(capacity, Context()), set(), [0, 0]
        for now, key in enumerate(dense.tolist()):
            if key in resident:
                policy.on_hit(key, now)
                continue
            misses[now >= split] += 1
            policy.on_miss(key, now)
            while len(resident) >= capacity:
                resident.remove(policy.evict(now))
            policy.insert(key, now)
            resident.add(key)
        assert [int(x) for x in line.split()[:2]] == misses, (capacity, config)
    # Online mode: committing to one design at the split is that design's handover run; switching runs.
    configs = list(VQ_DESIGNS.values())
    block = f"online 900 {split} 40 schedule=s:1\n" + "".join(c + "\n" for c in configs) + "end\n"
    switching = f"online 900 {split} 40 schedule=s:0,20:2,30:1\n" + "".join(c + "\n" for c in configs) + "end\n"
    shadows = f"online 900 {split} 40 rate=0.2 min=50\n" + "".join(c + "\n" for c in configs) + "end\n"
    (tmp_path / "jobs").write_text(f"900 {split} 0 {configs[1]} handover=1\n" + block + switching + shadows)
    ho, one, sw, mini = subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True, check=True).stdout.splitlines()
    assert one.split()[:3] == ho.split()[:2] + ["0"]
    assert sw.split()[2] == "2" and len(sw.split()) == 43 and len(mini.split()) == 43
    # OnlineVQ reproduces the online mode: miniature simulations, hysteresis, start without a handover.
    from policy_api import Handover
    from vq import VQ, OnlineVQ

    dense = _shifting_trace(40_000, 8, 7)
    (tmp_path / "ids.u32").write_bytes(dense.astype(np.uint32).tobytes())
    from vista.benchmarks.cachelab.architect import S3_CONFIG

    candidates = [S3_CONFIG, *VQ_DESIGNS.values(), "maxf=1 ghost=1.5 st=0.3,1,1,1,0,-1,1 st=0,1,1,1,0,-1,0 asmall=1"]
    heads = [("rate=0.2 min=50", 0.2, 0.0, True), ("rate=0.2 min=50 margin=0.02", 0.2, 0.02, True), ("rate=1 start=4", 1.0, 0.0, False)]
    (tmp_path / "jobs").write_text("".join(f"online 900 8000 40 {head}\n" + "".join(c + "\n" for c in candidates) + "end\n" for head, *_ in heads))
    lines = subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True, check=True).stdout.splitlines()
    switched = 0
    for (head, rate, margin, handover), line in zip(heads, lines, strict=True):
        cls = type("Online", (OnlineVQ,), {"CANDIDATES": tuple(candidates), "PERIOD": 1000, "RATE": rate, "MIN_MINI": 50, "MARGIN": margin})
        policy, resident, misses = cls(900, Context()), set(), [0, 0]
        incumbent = VQ(900, Context()) if handover else None
        for now, key in enumerate(dense.tolist()):
            if incumbent is not None and now == 8000:
                tail_first = [[(k, incumbent.state[k] & 15) for k in reversed(queue)] for queue in incumbent.queues]
                assert set(policy.adopt(Handover(900, now, *tail_first, list(incumbent.ghost), {}, not incumbent.has_evicted))) == resident
                incumbent = None
            active, hit = incumbent or policy, key in resident
            if hit:
                active.on_hit(key, now)
            else:
                misses[now >= 8000] += 1
                active.on_miss(key, now)
                while len(resident) >= 900:
                    resident.remove(active.evict(now))
                active.insert(key, now)
                resident.add(key)
            if incumbent is not None:
                policy.observe(key, now, hit)
        assert [int(x) for x in line.split()[:3]] == [*misses, policy.switches], head
        switched = max(switched, policy.switches)
    assert switched >= 3
    trapped = "maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,1,0"  # M demotes into itself: rejected by both
    (tmp_path / "jobs").write_text(f"100 {split} 0 {trapped}\n")
    assert subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True).stdout.startswith("error")
    with pytest.raises(ValueError):
        design(trapped)(100, Context())


def test_s4fifo_defaults_are_s3fifo_and_knobs_act(lab: Lab) -> None:
    cells = [(trace, capacity) for name, trace in lab.dev.items() for capacity in lab.capacities[name]]
    same = lab.store.simulate("from s4fifo import S4FIFO as CachePolicy\n", cells)
    knobs = lab.store.simulate(
        "from s4fifo import S4FIFO\n\n\nclass CachePolicy(S4FIFO):\n    SMALL_SKIP_RATIO = 0.5\n    GHOST_TO_MAIN_THRESHOLD = 1\n", cells
    )
    base = lab.experiment(BASELINE).cells
    for cell, run in same.items():
        assert np.array_equal(run.outcome, base[cell].outcome), cell
        assert knobs[cell].summary["instrumentation_errors"] == 0
    assert any(knobs[cell].summary["misses"] != base[cell].summary["misses"] for cell in knobs)


def test_keys_are_json_safe_and_random(lab: Lab) -> None:
    for trace in lab.dev.values():
        keys = trace.keys()
        assert keys.max() < traces.KEY_LIMIT
        first_seen = keys[np.sort(np.unique(keys, return_index=True)[1])]
        # Random ids: first-appearance order must not be sorted.
        assert not np.all(np.diff(first_seen.astype(np.int64)) > 0)


# Bypasses every other cold miss: exercises on_miss returning False.
SOMETIMES_BYPASS = """from s3fifo import S3FIFO


class CachePolicy(S3FIFO):
    def __init__(self, capacity, ctx):
        super().__init__(capacity, ctx)
        self.flip = False

    def on_miss(self, key, now):
        admitted = super().on_miss(key, now)
        if self.ghost_hit_pending:
            return admitted
        self.flip = not self.flip
        return self.flip
"""


@pytest.mark.parametrize("source", [BIGGER_SMALL, SOMETIMES_BYPASS], ids=["bigger-small", "bypass"])
def test_verified_matches_fast(lab: Lab, source: str) -> None:
    cells = [(trace, lab.capacities[name][1]) for name, trace in lab.dev.items()]
    fast = lab.store.simulate(source, cells)
    verified = lab.store.simulate(source, cells, verified=True)
    for cell, run in fast.items():
        other = verified[cell]
        assert other.summary["mode"] == "verified"
        for field in ("outcome", "queue", "ghost_hit", "ev_t", "ev_key", "ev_kind", "evict_key", "evict_q", "sample_sizes"):
            assert np.array_equal(getattr(run, field), getattr(other, field)), field
        assert other.summary["policy_memory_bytes"] > 0
    if source is SOMETIMES_BYPASS:
        assert any((run.outcome == 3).any() for run in fast.values())


@pytest.mark.parametrize(
    "source, fragment",
    [
        ("import os\nclass CachePolicy: pass\n", "import of 'os'"),
        ("class CachePolicy:\n    def f(self):\n        return open('x')\n", "'open'"),
        ("class CachePolicy:\n    def f(self):\n        return self.ctx._recorder\n", "private attribute"),
        ("class CachePolicy:\n    def f(self):\n        return self.__class__.__bases__\n", "'__class__'"),
        ("class Other: pass\n", "must define CachePolicy"),
    ],
)
def test_policy_check_rejects(source: str, fragment: str) -> None:
    assert any(fragment in problem for problem in check_policy(source))


def test_policy_check_admits_baseline_style() -> None:
    assert check_policy(BIGGER_SMALL) == []
    assert check_policy(BASELINE_SOURCE) == []


def test_unbounded_history_is_flagged(lab: Lab) -> None:
    view = lab.run_experiment("history", UNBOUNDED_HISTORY, "unbounded", None)
    report = lab.final_evaluation(view.data["experiment"])
    assert report["dev"]["memory_violations"]
    assert report["valid"] is False and report["score"] is None


def test_nondeterminism_is_flagged(lab: Lab) -> None:
    view = lab.run_experiment("random", UNSEEDED_RANDOM, "nondeterministic", None)
    report = lab.final_evaluation(view.data["experiment"])
    assert report["fast_verified_mismatches"]
    assert report["valid"] is False


def _call(runtime: TaskRuntime, tool: str, arguments: dict, number: list[int]):
    number[0] += 1
    call_id = f"call-{number[0]}"
    result = runtime.execute(tool, arguments, call_id)
    runtime.acknowledge(call_id)
    return result


def test_scripted_session_end_to_end(lab: Lab, tmp_path: Path) -> None:
    session = CacheLabSession(lab, visual=True, max_experiments=3)
    profile = make_profile(profile="vista", backend="claude", max_experiments=3, initial=session.initial)
    experiment = make_experiment(profile, backend="claude", model="scripted", effort="none", revision="test", suite=lab.public_suite(), max_experiments=3)
    runtime = TaskRuntime(session, experiment, artifact_dir=tmp_path / "artifacts")
    number = [0]
    runtime.begin_turn(runtime.prepare_input(Phase.INITIAL))
    trace = "dev-1"
    capacity = lab.capacities[trace][2]

    queues = _call(runtime, "view_queues", {"experiment": "s3fifo", "trace": trace, "capacity": capacity}, number)
    assert queues.success and len(queues.images) == 1
    label = queues.images[0]["steer_text"]
    event = int(label.split('event="')[1].split('"')[0])

    rejected = _call(runtime, "view_queues", {"experiment": "s3fifo", "trace": trace, "capacity": capacity + 1}, number)
    assert not rejected.success and "simulated at C in" in rejected.text

    zoom = _call(
        runtime,
        "inspect",
        {"question": "Which cause dominates misses?", "views": [{"label": "miss causes", "event": event, "region": {"x": 0, "y": 240, "width": 1024, "height": 200}}]},
        number,
    )
    assert zoom.success and len(zoom.images) == 1
    pixels = _call(
        runtime,
        "read_pixels",
        {"question": "colors", "views": [{"label": "stack", "event": event, "region": {"x": 100, "y": 300, "width": 600, "height": 80}, "rows": 2, "columns": 4}]},
        number,
    )
    assert pixels.success and "palette" in pixels.text

    result = _call(
        runtime,
        "run_experiment",
        {
            "name": "bigger S",
            "hypothesis": "S is too small for medium-distance reuse.",
            "prediction": {"expected_change_pct": -1.0, "mechanism_signature": "fewer misses after S eviction"},
            "code": BIGGER_SMALL,
        },
        number,
    )
    assert result.success and len(result.images) == 2 and "prediction check" in result.text  # overview + progress dashboard
    assert "progress after" in result.text and "best so far:" in result.text
    assert 'kind="current"' in result.images[0]["steer_text"]
    # The lab fixture is shared, so earlier tests already used other ids.
    mine = session.experiment_log[-1]["experiment"]

    for tool, arguments in (
        ("view_diff", {"experiment": mine, "trace": trace, "capacity": capacity}),
        ("view_objects", {"experiment": mine, "trace": trace, "capacity": capacity, "select": "most_regressed", "compare_to": "s3fifo", "count": 6}),
        ("view_overview", {"experiment": mine, "compare_to": "s3fifo"}),
    ):
        reply = _call(runtime, tool, arguments, number)
        assert reply.success and len(reply.images) == 1, (tool, reply.text[:300])
    events = _call(runtime, "query_events", {"experiment": mine, "trace": trace, "capacity": capacity, "start": 1000, "end": 1010}, number)
    assert events.success and "request key=" in events.text
    history = _call(runtime, "history", {"limit": 8}, number)
    assert history.success and json.loads(history.text)["events"]
    assert _call(runtime, "write_guide", {"content": "S evictions return within 1C on dev-1."}, number).success

    submitted = _call(
        runtime,
        "submit_policy",
        {"experiment": mine, "mechanism": "larger S", "evidence": f"event {event}", "limitations": "untested elsewhere"},
        number,
    )
    assert submitted.success and runtime.status == Status.COMPLETED
    runtime.close()
    outcome = session.result()
    assert outcome["submitted"] and outcome["experiments"][0]["observed_change_pct"] is not None
    assert outcome["tool_counts"]["view_queues"] == 2
    report = score(lab, outcome)
    assert report["selected_by"] == "agent" and report["valid"] is True
    assert report["heldout"]["scored"] == report["heldout"]["cells"]


def test_numeric_profile_has_no_images(lab: Lab, tmp_path: Path) -> None:
    session = CacheLabSession(lab, visual=False, max_experiments=2)
    profile = make_profile(profile="numeric", backend="codex", max_experiments=2, initial=session.initial)
    assert "inspect" not in profile.tools and "read_pixels" not in profile.tools
    experiment = make_experiment(profile, backend="codex", model="scripted", effort="none", revision="test", suite=lab.public_suite(), max_experiments=2)
    runtime = TaskRuntime(session, experiment, artifact_dir=tmp_path / "artifacts")
    runtime.begin_turn(runtime.prepare_input(Phase.INITIAL))
    number = [0]
    trace = "dev-2"
    reply = _call(runtime, "view_queues", {"experiment": "s3fifo", "trace": trace, "capacity": lab.capacities[trace][0]}, number)
    assert reply.success and not reply.images and "occupancy[S]" in reply.text
    runtime.close()


def test_test_policy_reveals_no_miss_ratio(lab: Lab) -> None:
    # test_policy is free; reporting performance would make it an unmetered experiment.
    view = lab.test_policy(BIGGER_SMALL)
    assert view.data["ok"] and "memory_ratio" in view.data
    assert "miss" not in view.text.lower()
    for run in lab.experiment("s3fifo").cells.values():
        assert f"{run.summary['miss_ratio']:.5f}" not in view.text


# -- engineer role -------------------------------------------------------------------

SECRET = "zz-secret-dataset-name"


@pytest.fixture(scope="module")
def engineer_setup(tmp_path_factory):
    from vista.benchmarks.cachelab import engineer

    root = tmp_path_factory.mktemp("engineer-lab-with-identifiable-dir")
    source_dir = tmp_path_factory.mktemp(SECRET)
    spec = traces.TraceSpec(SECRET, "heldout", "zipf_scan", {"objects": 3000, "alpha": 0.9, "scan_every": 5000, "scan_length": 1500}, requests=40_000, seed=5)
    # Pretend the workload came from a file whose path names its origin.
    spec = traces.TraceSpec(spec.name, spec.split, spec.family, spec.params, spec.requests, spec.seed)
    task = engineer.prepare(spec, root)
    task = engineer.EngineerTask(task.full, task.prefix, task.split, task.fraction, task.capacities, (*task.secrets, str(source_dir)))
    lab = engineer.make_lab(task, root, LocalSandbox(workers=4), references=("lru", "mq", "s4fifo", "fifo"), workers=4)
    return task, lab


def test_engineer_prefix_is_a_prefix(engineer_setup) -> None:
    task, lab = engineer_setup
    full, prefix = task.full.keys(), task.prefix.keys()
    assert task.split == int(0.2 * full.size) == prefix.size
    assert np.array_equal(full[: task.split], prefix)
    assert lab.dev["target"].public()["trace"] == "target"
    assert tuple(lab.capacities["target"]) == task.capacities


def test_engineer_never_shows_identifiers(engineer_setup) -> None:
    task, lab = engineer_setup
    session = CacheLabSession(lab, visual=True, max_experiments=3, role="engineer", shots=True)
    capacity = task.capacities[1]
    calls = {
        "run_experiment": {"name": "probe", "code": BIGGER_SMALL, "hypothesis": "h", "prediction": {"expected_change_pct": 0.0, "mechanism_signature": "s"}},
        "test_policy": {"code": "import os\nclass CachePolicy: pass\n"},
        "view_overview": {"experiment": "s3fifo", "compare_to": "mq"},
        "view_queues": {"experiment": "s3fifo", "trace": "target", "capacity": capacity},
        "view_objects": {"experiment": "s3fifo", "trace": "target", "capacity": capacity, "select": "most_remisses", "count": 3},
        "view_diff": {"experiment": "mq", "trace": "target", "capacity": capacity},
        "view_workload": {"trace": "target"},
        "view_mrc": {"trace": "target", "capacity": capacity, "experiment": "mq"},
        "view_positions": {"experiment": "s3fifo", "trace": "target", "capacity": capacity},
        "query_events": {"experiment": "s3fifo", "trace": "target", "capacity": capacity, "limit": 20},
        "experiment_info": {"experiment": "mq", "include_code": True},
    }
    bindings = {binding.spec.name: binding for binding in session.tools}
    texts = [session.initial]
    for name, arguments in calls.items():
        texts.append(bindings[name].handler(arguments).text)
    texts.append(bindings["view_queues"].handler({"experiment": "s3fifo", "trace": "nope", "capacity": 1}).text)
    for text in texts:
        assert SECRET not in text
        assert str(lab.root) not in text
        assert "private" not in text
        assert "zipf" not in text  # the generator family is not public either
    assert "Worked examples" in session.initial and "remaining 80%" in system_prompt_for("engineer")


def system_prompt_for(role: str) -> str:
    from vista.benchmarks.cachelab.profiles import system_prompt

    return system_prompt(visual=True, max_experiments=3, role=role)


def test_engineer_scores_only_the_suffix(engineer_setup, tmp_path) -> None:
    from vista.benchmarks.cachelab import engineer
    from vista.benchmarks.cachelab.engine import HIT

    task, lab = engineer_setup
    view = lab.run_experiment("bigger small", BIGGER_SMALL, "h", None)
    report = engineer.evaluate(lab, task, view.data["experiment"], zoo_dir=tmp_path / "zoo", workers=2, tune=False)
    assert report["valid"], report
    for row in report["rows"]:
        run = lab.store.simulate(BIGGER_SMALL, [(task.full, row["capacity"])], handover=task.fraction)[("target", row["capacity"])]
        assert run.summary["handover_at"] == task.split
        assert row["candidate"] == pytest.approx(float((run.outcome[task.split:] != HIT).mean()))
        if "S3-FIFO" in row["published"]:  # libCacheSim's S3-FIFO is the same algorithm
            assert row["published"]["S3-FIFO"] == pytest.approx(row["s3fifo"])
        assert row["belady"] <= min(v for k, v in row["published"].items() if k != "Belady") + 1e-12
    baseline = engineer.evaluate(lab, task, "s3fifo", zoo_dir=tmp_path / "zoo", workers=2, tune=False)
    assert baseline["score"] == 0.0


# -- experience: training sessions write a library, test sessions read it ---------------


def test_training_then_reading_experience(engineer_setup, tmp_path) -> None:
    from vista.benchmarks.cachelab import engineer
    from vista.benchmarks.cachelab.experience import Library, merge
    from vista.benchmarks.cachelab.profiles import protocol

    task, lab = engineer_setup
    zoo_dir = tmp_path / "zoo"
    baselines = engineer.evaluate(lab, task, "s3fifo", zoo_dir=zoo_dir, workers=2, tune=False)

    calls = []

    def feedback(identity: str, verified: bool = False) -> dict:
        calls.append((identity, verified))
        return engineer.evaluate(lab, task, identity, zoo_dir=zoo_dir, baselines=baselines, verified=verified)

    library = Library(tmp_path / "library")
    session = CacheLabSession(lab, visual=False, max_experiments=3, role="engineer", shots=True, library=library, train_feedback=feedback)
    tools = {binding.spec.name: binding.handler for binding in session.tools}
    assert "submit_policy" not in tools and {"commit_decision", "record_experience", "read_case"} <= set(tools)
    assert "Signature of this workload" in session.initial and "this is the first workload" in session.initial
    profile = make_profile(profile="numeric", backend="codex", max_experiments=3, initial=session.initial, role="engineer", train=True, experience=True)
    assert "commit_decision" in profile.tools and "submit_policy" not in profile.tools
    assert "record_experience" in profile.prompts.step
    assert protocol("engineer", True, train=True, experience=True).endswith("/engineer/shots/experience/train")

    assert tools["run_experiment"]({"name": "big S", "code": BIGGER_SMALL, "hypothesis": "h", "prediction": {"expected_change_pct": 0.0, "mechanism_signature": "s"}}).success
    identity = session.experiment_log[-1]["experiment"]
    assert not tools["record_experience"]({"decision": "d", "lesson": "l", "cited": [], "playbook_ops": []}).success
    decision = {"experiment": identity, "mechanism": "bigger S", "evidence": "e", "limitations": "l"}
    challenge = tools["commit_decision"](decision)  # a reference beats it on the observed part: asked once to look again
    assert not challenge.success and "Not final yet" in challenge.text and "does better on the observed part" in challenge.text
    reply = tools["commit_decision"](decision)  # the same decision on a second try is accepted
    assert reply.success and "Outcome on the remaining requests" in reply.text and not reply.finished
    assert not session.submitted
    assert not tools["commit_decision"](decision).success  # only once
    # The decision is scored like a test run: verified (memory, tampering).
    assert calls == [(identity, True)] and session.feedback_report["verified"] and session.feedback_report["valid"]
    assert "INVALID" not in reply.text
    expected = engineer.evaluate(lab, task, identity, zoo_dir=zoo_dir, tune=False, verified=False)
    assert [row["candidate"] for row in session.feedback_report["rows"]] == pytest.approx([row["candidate"] for row in expected["rows"]])
    from vista.benchmarks.cachelab.run import counterfactuals

    labels = {record["experiment"]: record for record in counterfactuals(lab, feedback)}
    assert {"s3fifo", "mq", identity} <= set(labels) and labels["s3fifo"]["mean_relative_vs_s3fifo"] == 0.0
    assert labels[identity]["mean_relative_vs_s3fifo"] == pytest.approx(session.feedback_report["mean_relative_vs_s3fifo"])
    assert labels["mq"]["reference"] and not labels[identity]["reference"] and labels[identity]["source"] == BIGGER_SMALL
    assert all(record["complete"] for record in labels.values())
    # Under a host budget: S3-FIFO and the decision first, references only if it lasts.
    budgeted = counterfactuals(lab, feedback, decision=identity, cost=10, budget=25)
    assert [record["experiment"] for record in budgeted] == ["s3fifo", identity] and not budgeted[0]["complete"]
    assert len(counterfactuals(lab, feedback, decision=identity, cost=10, budget=30)) == 3
    scans = {"op": "add", "scope": "periodic scans in the observed part", "text": "S3-FIFO with S=25%", "cases": ["case-001"]}
    bad = tools["record_experience"]({"decision": "S3-FIFO with S=25%", "lesson": "l", "cited": ["P009"], "playbook_ops": [scans]})
    assert not bad.success and "P009" in bad.text and not library.cases()  # nothing recorded
    bad = tools["record_experience"]({"decision": "S3-FIFO with S=25%", "lesson": "l", "cited": [], "playbook_ops": [{"op": "edit", "id": "P001", "text": "x"}]})
    assert not bad.success and not library.cases()
    reply = tools["record_experience"]({"decision": "S3-FIFO with S=25%", "lesson": "a bigger S helps scans", "cited": [], "playbook_ops": [scans]})
    assert reply.finished and session.submitted and session.submission_record["case"] == "case-001"
    (case, code) = library.case("case-001")
    assert code == BIGGER_SMALL and case["signature"]["cache_sizes"] == list(task.capacities)
    assert case["outcome"]["mean_relative_vs_s3fifo"] == pytest.approx(session.feedback_report["mean_relative_vs_s3fifo"])
    assert case["outcome"]["valid"]
    assert [item["id"] for item in library.items()] == ["P001"] and library.items()[0]["supports"] == ["case-001"]
    assert "[P001] when: periodic scans in the observed part" in library.playbook and "S3-FIFO with S=25%" in library.playbook

    # A test session reads the library and may deploy S3-FIFO or a reference unchanged.
    reader = CacheLabSession(lab, visual=False, max_experiments=3, role="engineer", shots=True, library=library)
    tools = {binding.spec.name: binding.handler for binding in reader.tools}
    assert "submit_policy" in tools and "commit_decision" not in tools and "record_experience" not in tools
    assert "[P001]" in reader.initial and "S3-FIFO with S=25%" in reader.initial and "case-001" in reader.initial
    texts = [reader.initial, tools["read_case"]({"case": "case-001"}).text]
    assert "class CachePolicy(S3FIFO)" in texts[-1]
    missing = tools["read_case"]({"case": "case-009"})
    assert not missing.success and "case-001" in missing.text
    for text in [*texts, session.initial, reply.text]:
        assert SECRET not in text and str(lab.root) not in text and "private" not in text
    # A reference may be deployed unchanged when it fits the memory limit (MQ does not: 3-5x).
    refused = tools["submit_policy"]({"experiment": "mq", "mechanism": "m", "evidence": "e", "limitations": "l"})
    assert not refused.finished and "would be scored invalid" in refused.text
    lru = {"experiment": "lru", "mechanism": "m", "evidence": "e", "limitations": "l"}
    assert tools["submit_policy"](lru).finished or tools["submit_policy"](lru).finished  # at most one challenge

    # A second session's library: its rule cites its own case-001, which becomes case-002 when merged.
    other = Library(tmp_path / "other")
    other.add_case({"signature": case["signature"], "decision": "keep S3-FIFO", "outcome": {}, "cited": []}, "")
    other.apply_ops([{"op": "add", "scope": "no scans", "text": "keep S3-FIFO", "cases": ["case-001"]}], "case-001")
    other.apply_ops([{"op": "edit", "id": "P001", "text": "keep S3-FIFO unless scans appear"}], "case-002")
    old = Library(tmp_path / "old-format")  # a free-text playbook from before itemization
    old.add({"signature": case["signature"], "decision": "MQ", "outcome": {}}, "", "- deploy MQ on small caches")
    merged = Library(merge([library.root, other.root, old.root], tmp_path / "merged"))
    assert [c["case"] for c in merged.cases()] == ["case-001", "case-002", "case-003"]
    rules = {item["id"]: item for item in merged.items()}
    assert rules["P001"]["text"] == "S3-FIFO with S=25%" and rules["P001"]["supports"] == ["case-001"]
    assert rules["P002"]["text"] == "keep S3-FIFO unless scans appear" and rules["P002"]["supports"] == ["case-002"]
    assert [name for name, _ in merged.proposals()] == ["session-3"] and "deploy MQ" in merged.proposals()[0][1]
    assert merged.playbook_is_merged() and "[P002]" in merged.playbook and "deploy MQ" in merged.playbook
    assert merged.case("case-002")[0]["origin"] == "session-2, case-001"


def test_designer_cannot_submit_a_reference(tmp_path) -> None:
    designer_lab = Lab(tmp_path / "lab", LocalSandbox(workers=2), dev=DEV[:1], heldout=HELDOUT, references=("lru",))
    session = CacheLabSession(designer_lab, visual=False, max_experiments=2)
    tools = {binding.spec.name: binding.handler for binding in session.tools}
    for identity in ("s3fifo", "lru"):
        assert not tools["submit_policy"]({"experiment": identity, "mechanism": "m", "evidence": "e", "limitations": "l"}).success


# -- handover: a policy taking over a live S3-FIFO cache ---------------------------------

OBSERVER = """from s3fifo import S3FIFO


class CachePolicy(S3FIFO):
    SMALL_RATIO = 0.3

    def __init__(self, capacity, ctx):
        super().__init__(capacity, ctx)
        self.seen = 0

    def observe(self, key, now, hit):
        assert now == self.seen
        self.seen += 1

    def adopt(self, state):
        assert state.now == self.seen and len(state.resident()) <= self.capacity
        return super().adopt(state)
"""
LOSSY_ADOPT = """from s3fifo import S3FIFO


class CachePolicy(S3FIFO):
    def adopt(self, state):
        placement = super().adopt(state)
        placement.pop(next(iter(placement)))
        return placement
"""


def test_handover(lab: Lab) -> None:
    store, trace = lab.store, lab.dev["dev-2"]
    cells = [(trace, capacity) for capacity in lab.capacities["dev-2"]]
    cold = store.simulate(BASELINE_SOURCE, cells)
    switched = store.simulate(BASELINE_SOURCE, cells, handover=0.3)
    for cell, run in cold.items():
        assert np.array_equal(run.outcome, switched[cell].outcome)  # S3-FIFO taking over S3-FIFO is no change
        assert switched[cell].summary["handover_at"] == int(trace.requests * 0.3)
    lru = (SANDBOX_DIR / "lru.py").read_text()
    for source in (BIGGER_SMALL, OBSERVER, lru):
        fast = store.simulate(source, cells, handover=0.3)
        verified = store.simulate(source, cells, handover=0.3, verified=True)
        for cell, run in fast.items():
            assert run.summary["misses"] == verified[cell].summary["misses"], (source[:40], cell)
            assert run.summary["instrumentation_errors"] == 0 and run.summary["size_mismatches"] == 0
            switch = run.summary["handover_at"]
            assert np.array_equal(run.outcome[:switch], cold[cell].outcome[:switch])  # S3-FIFO served the prefix
    assert fast[cells[0][0].spec.name, cells[0][1]].summary["queues"] == ["S", "M", "Q"]  # LRU after S3-FIFO
    for verified in (False, True):
        error = store.simulate(LOSSY_ADOPT, cells[:1], handover=0.3, verified=verified)[("dev-2", cells[0][1])]
        assert "must place exactly the" in error.error


def test_engineer_experiments_hand_over(engineer_setup) -> None:
    task, lab = engineer_setup
    assert lab.handover == 0.2 and task.fractions[-1] == 0.3
    session = CacheLabSession(lab, visual=False, max_experiments=3, role="engineer")
    tools = {binding.spec.name: binding for binding in session.tools}
    assert "handover_at" in tools["run_experiment"].spec.input_schema["properties"]
    designer = CacheLabSession(Lab(lab.root.parent / "designer-lab", LocalSandbox(workers=2), dev=DEV[:1], heldout=HELDOUT), visual=False, max_experiments=1)
    assert "handover_at" not in {b.spec.name: b for b in designer.tools}["run_experiment"].spec.input_schema["properties"]
    assert "adopt()" in system_prompt_for("engineer") and "adopt()" not in system_prompt_for("designer")
    prediction = {"expected_change_pct": 0.0, "mechanism_signature": "s"}
    default = tools["run_experiment"].handler({"name": "d", "code": BIGGER_SMALL, "hypothesis": "h", "prediction": prediction})
    cold = tools["run_experiment"].handler({"name": "c", "code": BIGGER_SMALL, "hypothesis": "h", "prediction": prediction, "handover_at": 0})
    assert "start: handover" in default.text and "start: cold cache" in cold.text
    first, second = (lab.experiment(record["experiment"]) for record in session.experiment_log)
    assert first.handover == 0.2 and second.handover is None
    capacity = task.capacities[0]  # small enough to evict before the switch
    split = int(task.prefix.requests * 0.2)
    base = lab.experiment(BASELINE).run("target", capacity).outcome
    assert np.array_equal(first.run("target", capacity).outcome[:split], base[:split])
    assert not np.array_equal(second.run("target", capacity).outcome[:split], base[:split])


def test_analyst_reviews_training_cases_in_hindsight(engineer_setup, tmp_path) -> None:
    from vista.benchmarks.cachelab import analyst
    from vista.benchmarks.cachelab.experience import (
        Library,
        absorb,
        compact_counterfactuals,
        signature,
    )

    task, lab = engineer_setup
    lineage = Library(tmp_path / "lineage-1")
    case = lineage.add(
        {"signature": signature(lab), "decision": "S3-FIFO with S=25%", "lesson": "bigger S helps scans", "outcome": {"mean_relative_vs_s3fifo": 0.01}},
        BIGGER_SMALL,
        "- scans: S=25%",
    )
    lineage.set_origin(case, {"target": task.full.spec.manifest(), "split": task.split, "split_fraction": task.fraction, "capacities": list(task.capacities)})
    hindsight = {"experiment": "e1", "name": "big S", "reference": False, "source": BIGGER_SMALL, "prefix_relative_vs_s3fifo": -0.01, "mean_relative_vs_s3fifo": 0.02, "suffix": []}
    lineage.update_case(case, counterfactuals=compact_counterfactuals([hindsight]))

    library = Library(tmp_path / "merged")
    assert absorb(library, lineage, label="session-1") == ["case-001"]
    assert library.origins()["case-001"]["split"] == task.split
    assert [name for name, _ in library.proposals()] == ["session-1"]

    # The analyst replays the whole workload under the case id, with the decision recorded as dec-001.
    alab = analyst.make_lab(library, tmp_path / "analyst-lab", LocalSandbox(workers=4), workers=4)
    assert list(alab.dev) == ["case-001"] and alab.handover == task.fraction
    assert np.array_equal(alab.dev["case-001"].keys(), task.full.keys())
    assert alab.is_reference("dec-001") and alab.experiment("dec-001").source == BIGGER_SMALL
    assert alab.experiment("dec-001").ok

    session = CacheLabSession(alab, visual=False, max_experiments=3, role="analyst", library=library)
    tools = {binding.spec.name: binding.handler for binding in session.tools}
    assert {"update_knowledge", "read_case", "run_experiment"} <= set(tools)
    assert not {"submit_policy", "commit_decision", "record_experience"} & set(tools)
    profile = make_profile(profile="numeric", backend="codex", max_experiments=3, initial=session.initial, role="analyst", experience=True)
    assert "update_knowledge" in profile.tools
    texts = [
        session.initial,
        tools["read_case"]({"case": "case-001"}).text,
        tools["experiment_info"]({"experiment": "dec-001"}).text,
        tools["view_overview"]({"experiment": "dec-001", "compare_to": "s3fifo"}).text,
    ]
    assert "case-001" in session.initial and "dec-001" in session.initial and "big S" in texts[1]
    for text in texts:
        assert SECRET not in text and str(task.full.path) not in text and "ORIGINS" not in text

    assert not tools["update_knowledge"]({"postmortems": [{"case": "case-009", "text": "x"}], "knowledge": "k", "playbook_ops": []}).success
    keep = {"op": "add", "scope": "one-hit share below 0.5", "text": "keep S3-FIFO", "cases": ["case-001"]}
    assert not tools["update_knowledge"]({"postmortems": [], "knowledge": "k", "playbook_ops": [{"op": "retire", "id": "P004", "reason": "r"}]}).success
    reply = tools["update_knowledge"]({"postmortems": [{"case": "case-001", "text": "S=25% lost after the split"}], "knowledge": "- scans pass", "playbook_ops": [keep]})
    assert reply.finished and session.submitted
    assert "[P001] when: one-hit share below 0.5" in library.playbook and library.knowledge == "- scans pass" and not library.proposals()
    assert "postmortem (analyst-1): S=25% lost after the split" in library.case_text("case-001")


def test_view_opt_splits_avoidable_misses(lab: Lab) -> None:
    from vista.benchmarks.cachelab import analysis

    name, capacity = "dev-2", lab.capacities["dev-2"][0]
    view = lab.opt_view(BASELINE, name, capacity, None, None, 32)
    totals = view.data["opt"]
    run = lab.experiment(BASELINE).run(name, capacity)
    opt = lab.opt(name, capacity)
    # OPT never misses more than the policy, and avoidable + unavoidable = all misses.
    assert totals["opt_misses"] <= totals["misses"]
    assert totals["avoidable"] == int(((run.outcome != analysis.HIT) & opt["hit"]).sum())
    assert sum(totals["avoidable_by_queue"].values()) == totals["avoidable"]
    assert view.png is not None and "Belady boundary" in view.text
    # The comparison form: what a bigger S fixed and created relative to S3-FIFO.
    bigger, _ = lab.dev_experiment("big S for opt", BIGGER_SMALL)
    compared = lab.opt_view(bigger.id, name, capacity, None, None, 32, BASELINE)
    assert "fixed" in compared.text and compared.png is not None


def test_memory_bound_charges_policy_state_not_imports(tmp_path) -> None:
    spec = traces.TraceSpec("m", "dev", "zipf_scan", {"objects": 5000, "alpha": 0.8, "scan_every": 5000, "scan_length": 1500}, requests=20_000, seed=3)
    small = Lab(tmp_path / "lab", LocalSandbox(workers=2), dev=[spec], heldout=(), workers=2)
    same = small.test_policy("from s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n    GHOST_RATIO = 0.9\n")
    # An identical subclass costs what S3-FIFO costs; in a tiny cache a few KB of class overhead show.
    ratios = list(same.data["memory_ratios"].values())
    assert 0.9 <= min(ratios) and max(ratios[1:]) <= 1.15 and ratios[0] <= 1.25, same.text
    stash = small.test_policy("from s3fifo import S3FIFO\nBIG = bytearray(4_000_000)\n\n\nclass CachePolicy(S3FIFO):\n    pass\n")
    assert stash.data["memory_ratio"] > 2.0 and "over the memory limit" in stash.text, stash.text  # module-level state still counts


def test_host_credits_cited_rules_from_hindsight(tmp_path) -> None:
    from vista.benchmarks.cachelab.experience import Library

    library = Library(tmp_path / "library")
    library.apply_ops([{"op": "add", "scope": "s", "text": "bigger ghost"}, {"op": "add", "scope": "s", "text": "deploy MQ"}], "seed")

    def case(mean: float, best: float, cited: list[str]) -> str:
        case_id = library.add_case({"signature": {}, "decision": "d", "outcome": {"mean_relative_vs_s3fifo": mean}, "cited": cited}, "")
        hindsight = [{"experiment": "e1", "mean_relative_vs_s3fifo": best}, {"experiment": "s3fifo", "mean_relative_vs_s3fifo": 0.0}]
        library.update_case(case_id, counterfactuals=hindsight)
        return case_id

    good, bad, regret = case(-0.010, -0.012, ["P001"]), case(+0.004, -0.001, ["P002"]), case(-0.002, -0.030, ["P001"])
    assert [library.credit(c) for c in (good, bad, regret)] == ["helpful", "harmful", "harmful"]
    assert library.credit(good) is None  # once per case
    rules = {item["id"]: item for item in library.items()}
    assert (rules["P001"]["helpful"], rules["P001"]["harmful"]) == (1, 1) and (rules["P002"]["helpful"], rules["P002"]["harmful"]) == (0, 1)
    assert "did well 1x, badly 1x" in library.playbook


def test_view_handover_splits_decision_and_transition(engineer_setup) -> None:
    task, lab = engineer_setup
    view = lab.run_experiment("tiny S big ghost", "from s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n    SMALL_RATIO = 0.02\n    GHOST_RATIO = 2.0\n", "h", None)
    identity = view.data["experiment"]
    capacity = task.capacities[-1]
    handover = lab.handover_view(identity, "target", capacity)
    totals = handover.data["handover"]
    assert totals["deployed_effect"] == pytest.approx(totals["decision_effect"] + totals["transition_cost"])
    assert handover.png is not None and "transition" in handover.text and SECRET not in handover.text
    cold = lab.run_experiment("cold", "from s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n    SMALL_RATIO = 0.02\n", "h", None, handover=0)
    with pytest.raises(LabError):
        lab.handover_view(cold.data["experiment"], "target", capacity)


def test_bucket_mean_stops_at_the_window_end() -> None:
    from vista.benchmarks.cachelab import analysis

    values = np.zeros(100)
    values[60:] = 1.0  # all after the window
    assert analysis.bucket_mean(values, analysis.edges(0, 50, 5)).tolist() == [0.0] * 5


def test_view_knobs_sweeps_one_knob_at_a_time(engineer_setup) -> None:
    from vista.benchmarks.cachelab import analysis
    from vista.benchmarks.cachelab.search import policy_source

    task, lab = engineer_setup
    capacity = task.capacities[1]
    view = lab.knobs_view("target", capacity)
    knobs = view.data["knobs"]
    assert set(knobs) == {"SMALL_RATIO", "GHOST_RATIO", "MOVE_THRESHOLD", "MAX_FREQ"}
    for data in knobs.values():
        # The default is a candidate, so the best static value never loses, and the per-window best never loses to it.
        assert data["switching_change"] <= data["static_change"] + 1e-12 <= 1e-12
    assert view.png is not None and "best per window" in view.text and SECRET not in view.text
    # A variant's misses after the handover match simulating it directly.
    from vista.benchmarks.cachelab.engine import handover_index

    trace = lab.trace("target")
    first = handover_index(trace.requests, lab.handover)
    assert first > 0
    run = lab.store.simulate(policy_source({"SMALL_RATIO": 0.02}), [(trace, capacity)], handover=lab.handover)[("target", capacity)]
    base = lab.experiment(BASELINE).run("target", capacity)
    direct = ((run.outcome[first:] != analysis.HIT).sum() - (base.outcome[first:] != analysis.HIT).sum()) / (base.outcome[first:] != analysis.HIT).sum()
    swept = [line for line in view.text.splitlines() if line.strip().startswith("0.02:")][0]
    assert f"overall {direct * 100:+.2f}%" in swept
    # Cached: a second call simulates nothing new and agrees.
    assert lab.knobs_view("target", capacity).text == view.text
    assert lab.knobs_view("target", capacity, buckets=8, visual=False).png is None


def test_view_mrc_prices_a_cell_in_cache_size(engineer_setup) -> None:
    pytest.importorskip("libcachesim")
    from vista.benchmarks.cachelab.lab import Lab

    task, lab = engineer_setup
    capacity = task.capacities[1]
    view = lab.mrc_view("target", capacity, "s3fifo")
    data = view.data["mrc"]
    sizes, ratio = np.array(data["sizes"]), {k: np.array(v) for k, v in data["miss_ratio"].items()}
    assert capacity in sizes and sizes[0] < capacity < sizes[-1]
    assert np.all(ratio["Belady"] <= ratio["S3-FIFO"] + 1e-12) and np.all(ratio["Belady"] <= ratio["LRU"] + 1e-12)
    assert np.all(np.diff(ratio["Belady"]) <= 1e-12)  # OPT is a stack algorithm
    # libCacheSim's S3-FIFO from a cold cache is the lab's S3-FIFO after the handover, so S3-FIFO is worth exactly C.
    assert "note:" not in view.text and data["equivalent"]["s3fifo"] == pytest.approx(1.0)
    assert data["equivalent"]["OPT"] is None or data["equivalent"]["OPT"] > 1.0
    assert view.png is not None and "equivalent S3-FIFO size" in view.text and SECRET not in view.text
    # Interpolation in log size on the non-increasing envelope.
    assert Lab.equivalent_size(np.array([10, 20, 40]), np.array([100, 80, 40]), 60) == pytest.approx(20 * 2**0.5)
    assert Lab.equivalent_size(np.array([10, 20, 40]), np.array([100, 80, 40]), 120) is None
    assert Lab.equivalent_size(np.array([10, 20, 40]), np.array([100, 110, 40]), 100) == pytest.approx(10)
    # Cached: a second call agrees; numbers only without the image.
    again = lab.mrc_view("target", capacity, visual=False)
    assert again.png is None and again.data["mrc"]["miss_ratio"] == data["miss_ratio"]


def test_training_rounds_take_fresh_workloads_then_a_second_pass(tmp_path, monkeypatch) -> None:
    """Rounds >= 2 train each lineage on workloads no earlier round used; once those run
    out, --allow-repeats wraps around (oldest first) and lists the repeats."""
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location("cachelab_experience", Path(__file__).parents[1] / "scripts/cachelab_experience.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    root, suite = tmp_path / "root", tmp_path / "suite.json"
    suite.write_text("[]")
    root.mkdir()
    (root / "targets.json").write_text(json.dumps({"train": {"suite": str(suite.resolve()), "targets": {f"w0{i}": f"t{i}" for i in range(1, 5)}}}))
    trained: list[tuple[int, str]] = []

    def run(out, lab, target, *args, **kwargs):
        out.mkdir(parents=True)
        trained.append((int(out.parent.name.split("-")[1]), out.name))
        return {}

    class Stub:
        def __init__(self, path):
            pass

        def cases(self):
            return []

        def ops_log(self):
            return []

    monkeypatch.setattr(driver, "_run", run)
    monkeypatch.setattr(driver, "_backfill", lambda library, runs: 0)
    monkeypatch.setattr(driver, "Library", Stub)

    def train(round_, per=1, repeats=False, limit=0):
        trained.clear()
        args = argparse.Namespace(root=root, suite=suite, seed=0, lineages=2, round=round_, limit=limit, per_round=per, allow_repeats=repeats, budget=0, backend="claude", baselines=None)
        driver.cmd_train(args, [])
        return sorted(trained)

    assert train(1, limit=1) == [(1, "w01"), (2, "w02")]  # shares: lineage 1 w01 w03, lineage 2 w02 w04
    with pytest.raises(SystemExit, match="merge"):
        train(2)
    (root / "library" / "merged").mkdir(parents=True)
    (root / "library" / "merged" / "playbook.md").write_text("")
    assert train(2) == [(1, "w03"), (2, "w04")]
    assert not (root / "train" / "round-2" / "REPEATS.json").exists()
    with pytest.raises(SystemExit, match="allow-repeats"):
        train(3)
    assert train(3, repeats=True) == [(1, "w01"), (2, "w02")]  # the oldest first
    assert json.loads((root / "train" / "round-3" / "REPEATS.json").read_text()) == [{"lineage": 1, "workload": "w01"}, {"lineage": 2, "workload": "w02"}]
    with pytest.raises(SystemExit, match="per-round"):
        train(4, per=3, repeats=True)


def test_repeats_are_named(engineer_setup) -> None:
    from vista.benchmarks.cachelab.lab import program_key

    task, lab = engineer_setup
    code = 'from s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n    """Smaller S."""\n\n    SMALL_RATIO = 0.03\n'
    same = "# the same idea, reformatted\nfrom s3fifo import S3FIFO\nclass CachePolicy(S3FIFO):\n    SMALL_RATIO = 0.03  # smaller\n"
    assert program_key(code) == program_key(same) != program_key(code.replace("0.03", "0.04"))
    first = lab.run_experiment("small S", code, "h", None)
    assert lab.test_policy(same).data["same_program"] == [first.data["experiment"]]
    again = lab.run_experiment("small S again", same, "h", None)
    assert again.data["same_as"] == first.data["experiment"] and "repeat:" in again.text and "same program" in again.text
    # A change that never alters a decision behaves exactly like S3-FIFO.
    inert = lab.run_experiment("inert", "from s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n    UNUSED = 7\n", "h", None)
    assert inert.data["same_as"] == "s3fifo" and "did not alter any decision" in inert.text
    fresh = lab.run_experiment("bigger ghost", "from s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n    GHOST_RATIO = 2.0\n", "h", None)
    assert fresh.data["same_as"] is None and "repeat:" not in fresh.text


def test_provider_errors_resume_the_same_conversation(tmp_path) -> None:
    """A gateway 5xx ends Claude Code's turn; the harness pauses and resumes that
    conversation (keeping its context) before falling back to a fresh session."""
    from vista.backends.claude import lifecycle
    from vista.backends.claude.runner import ClaudeResult

    def result(segment: int, session: str, *, status: int | None, code: int) -> ClaudeResult:
        path = tmp_path / f"s{segment}"
        return ClaudeResult(segment, code, code, "", session, (), path, path, path, path, path, path, None, api_error_status=status)

    class Controller:
        terminal = False
        step_index = 0
        compact_checkpoint_marker = None
        retry_boundary_pending = False

        def begin_runtime_recovery(self):
            return "recovery"

    class Runner:
        def __init__(self, failures: int) -> None:
            self.calls, self.failures, self.controller = [], failures, Controller()

        def run_task(self, prompt):
            self.calls.append("start")
            return result(0, "s1", status=502, code=1)

        def resume_nonterminal_task(self, session_id, prompt):
            self.calls.append(("resume", session_id))
            if len(self.calls) - 1 <= self.failures:  # the first `failures` resumes fail too
                return result(len(self.calls), session_id, status=502, code=1)
            self.controller.terminal = True
            return result(len(self.calls), session_id, status=None, code=0)

        def run_fresh_runtime_recovery(self, recovery):
            self.calls.append("fresh")
            self.controller.terminal = True
            return result(len(self.calls), "s2", status=None, code=0)

    sleeps: list[float] = []
    runner = Runner(failures=2)
    lifecycle.run_task_with_native_context(runner=runner, controller=runner.controller, prompt="go", results=[], sleep=sleeps.append)
    assert runner.calls == ["start", ("resume", "s1"), ("resume", "s1"), ("resume", "s1")] and len(sleeps) == 3
    # Resuming keeps failing: after the per-step allowance, a fresh session takes over.
    runner = Runner(failures=99)
    lifecycle.run_task_with_native_context(runner=runner, controller=runner.controller, prompt="go", results=[], sleep=sleeps.append)
    assert runner.calls.count(("resume", "s1")) == lifecycle.MAX_PROVIDER_5XX_RESUMES_PER_STEP and runner.calls[-1] == "fresh"


def test_policy_check_blocks_reaching_the_simulator() -> None:
    from vista.benchmarks.cachelab.policy_check import check_policy

    head = "import random\nimport typing\nfrom s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n    def on_request(self, key):\n"
    escapes = [
        "        frame = getattr(getattr(random, '_os').sys, '_getframe')(1)\n",  # the audit's route to the trace
        "        name = '_' + 'os'\n        getattr(random, name)\n",
        "        typing.sys.modules['builtins']\n",
        "        frame = self.f_back\n",
    ]
    for escape in escapes:
        assert check_policy(head + escape), escape
    assert check_policy(head + "        return getattr(self, '_' + '_class__')\n")  # computed names stay out
    # Ordinary dynamic access to the policy's own state stays allowed (S3-FIFO's export does it).
    assert not check_policy(head + "        return getattr(self, 'SMALL_RATIO', 0.1)\n")
    assert not check_policy(head + "        return {name: getattr(self, name) for name in ('SMALL_RATIO', 'MAX_FREQ')}\n")


def test_invalid_decisions_are_named_in_training_feedback() -> None:
    from vista.benchmarks.cachelab.experience import outcome_summary, validity_text

    assert validity_text({"valid": True}) == ""
    report = {
        "valid": False,
        "rows": [],
        "failures": [],
        "memory_violations": [{"capacity": 41, "ratio": 2.31}],
        "fast_verified_mismatches": [{"capacity": 410}],
    }
    text = validity_text(report)
    assert text.startswith("INVALID") and "41 (2.31x)" in text and "C=410" in text and "S3-FIFO unchanged (0%)" in text
    assert outcome_summary(report)["valid"] is False


def test_full_length_suites_pick_whole_traces_in_a_length_range(tmp_path) -> None:
    from vista.benchmarks.cachelab.traces import (
        ORACLE_GENERAL,
        count_requests,
        sample_real_suite,
    )

    def write(path: Path, n: int) -> None:
        records = np.zeros(n, dtype=ORACLE_GENERAL)
        records["obj_id"] = np.arange(n) % 97
        data = records.tobytes()
        if path.suffix == ".zst":
            zstandard = pytest.importorskip("zstandard")
            data = zstandard.ZstdCompressor().compress(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    lengths = {"a.oracleGeneral": 50, "b.oracleGeneral": 400, "c.oracleGeneral.zst": 700, "d.oracleGeneral.zst": 5000}
    for name, n in lengths.items():
        write(tmp_path / "ds" / name, n)
    assert count_requests(tmp_path / "ds" / "c.oracleGeneral.zst") == 700
    assert count_requests(tmp_path / "ds" / "d.oracleGeneral.zst", stop_after=1000) > 1000  # stops early, still over

    specs = sample_real_suite(tmp_path, split="heldout", datasets=["ds"], per_dataset=5, requests=10, seed=1, full_length=(100, 1000), name_prefix="hl")
    assert sorted((spec.params["file"], spec.requests) for spec in specs) == [("b.oracleGeneral", 400), ("c.oracleGeneral.zst", 700)]
    assert [spec.name for spec in specs] == ["hl-ds-1", "hl-ds-2"]
    excluded = sample_real_suite(
        tmp_path, split="heldout", datasets=["ds"], per_dataset=5, requests=10, seed=1, full_length=(100, 1000), exclude={specs[0].source}
    )
    assert [spec.source for spec in excluded] == [specs[1].source]
    # Without full_length, the old behaviour: a fixed prefix of any file.
    prefixes = sample_real_suite(tmp_path, split="heldout", datasets=["ds"], per_dataset=2, requests=10, seed=1)
    assert len(prefixes) == 2 and {spec.requests for spec in prefixes} == {10}


def test_docker_batches_mount_files_alone_and_scale_memory(tmp_path, monkeypatch) -> None:
    import subprocess

    from vista.benchmarks.cachelab import engine

    private = tmp_path / "private"
    private.mkdir()
    (private / "target.json").write_text('{"source": "/data/secret-name.zst"}')
    trace = private / "target.bin"
    with trace.open("wb") as stream:
        stream.truncate(50_000_000 * engine.KEY_BYTES)  # sparse: a 50M-request trace
    small = tmp_path / "traces" / "dev.bin"
    small.parent.mkdir()
    small.write_bytes(b"\0" * 8 * 1000)
    policy = tmp_path / "policies" / "p" / "policy.py"
    policy.parent.mkdir(parents=True)
    policy.write_text("x = 1\n")
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"results": [{"ok": True, "out": "/out/0"}, {"ok": True, "out": "/out/1"}]}), stderr="")

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    sandbox = engine.DockerSandbox(workers=4, memory_gb=6)
    jobs = [{"trace": str(trace), "capacity": c, "out": str(tmp_path / f"out{c}"), "policy": str(policy)} for c in (10, 20)]
    sandbox.run(jobs, work=tmp_path / "work", timeout=60)
    command = seen["command"]
    volumes = [command[i + 1] for i, part in enumerate(command) if part == "-v"]
    mounted = {volume.split(":")[0]: volume.split(":")[1] for volume in volumes}
    assert mounted[str(trace.resolve())].endswith("/target.bin") and mounted[str(policy.resolve())].endswith("/policy.py")
    assert not any(volume.startswith(f"{private.resolve()}:") for volume in volumes)  # never the directory with the manifest
    # Two 50M-request cells at once need more than the 6 GB default; a small batch keeps it.
    memory = int(command[command.index("--memory") + 1].removesuffix("g"))
    assert memory == -(-2 * (engine.CELL_BASE_BYTES + 50_000_000 * engine.CELL_BYTES_PER_REQUEST) // (1 << 30)) > 6
    assert sandbox.memory_for([{"trace": str(small)}]) == 6


def test_redaction_hides_resolved_lab_paths_and_the_simulator_install(engineer_setup, tmp_path) -> None:
    from vista.benchmarks.cachelab import engineer

    _, lab = engineer_setup
    traceback = f'File "{SANDBOX_DIR.resolve()}/simulate.py", line 9; {lab.root.resolve()}/runs/abc; {SECRET}'
    text = lab.redact(traceback)
    assert str(SANDBOX_DIR.resolve()) not in text and str(lab.root.resolve()) not in text and SECRET not in text
    # A lab reached through a symlink (runs/ on another disk) hides the real path too.
    link = tmp_path / "runs-link"
    link.symlink_to(lab.root.parent, target_is_directory=True)
    task = engineer.prepare(traces.TraceSpec("w", "heldout", "zipf_scan", {"objects": 3000, "alpha": 0.6, "scan_every": 2000, "scan_length": 500}, requests=10_000, seed=1), link / lab.root.name / "via-link")
    linked = engineer.make_lab(task, link / lab.root.name / "via-link", LocalSandbox(workers=1))
    assert str(linked.root.resolve()) not in linked.redact(f"{linked.root.resolve()}/runs/x")


def test_claude_tool_calls_may_stay_silent_for_hours(monkeypatch, tmp_path) -> None:
    import argparse
    import types

    import vista.backends.claude as claude
    from vista.backends.claude.runner import ClaudeCodeRunner
    from vista.backends.launch import add_backend_arguments, prepare_backend

    parser = argparse.ArgumentParser()
    add_backend_arguments(parser)
    args = parser.parse_args(["--backend", "claude", "--claude-base-url", "https://gateway.example", "--claude-tool-idle-timeout", "10800"])
    args.task_timeout = 60
    monkeypatch.setenv(args.claude_token_env, "token")
    seen = {}
    monkeypatch.setattr(claude, "create_backend", lambda runtime, **kwargs: seen.update(kwargs))
    factory, _ = prepare_backend(args, parser.error, check=lambda transport, binary: (tmp_path / "claude", {"container_image_id": "image"}))
    factory(None, tmp_path)
    provider = seen["provider"]
    assert provider.mcp_tool_idle_timeout_ms == 10_800_000 and provider.max_retries == 15
    fake = types.SimpleNamespace(
        provider_environment=(), max_output_tokens=None, mcp_tool_timeout_ms=None, max_retries=provider.max_retries,
        mcp_tool_idle_timeout_ms=provider.mcp_tool_idle_timeout_ms, _container_name=lambda session: None,
    )
    arguments = ClaudeCodeRunner._provider_arguments(fake, "s")
    assert "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=10800000" in arguments and "CLAUDE_CODE_MAX_RETRIES=15" in arguments


def test_belady_boundary_is_infinite_when_opt_only_evicts_dead_objects() -> None:
    import warnings

    from vista.benchmarks.cachelab.analysis import NEVER, belady_boundary

    gaps = np.array([NEVER] * 99 + [5], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert belady_boundary({"evict_gap": gaps}) == float("inf")
    assert belady_boundary({"evict_gap": np.array([10, 20, 30, NEVER], dtype=np.int64)}) == pytest.approx(11.5)


def test_a_decision_over_the_memory_limit_is_refused(engineer_setup) -> None:
    _, lab = engineer_setup
    session = CacheLabSession(lab, visual=False, max_experiments=3, role="engineer", shots=True)
    tools = {binding.spec.name: binding.handler for binding in session.tools}
    stash = "from s3fifo import S3FIFO\nBIG = bytearray(4_000_000)\n\n\nclass CachePolicy(S3FIFO):\n    pass\n"
    assert tools["run_experiment"]({"name": "stash", "code": stash, "hypothesis": "h", "prediction": {"expected_change_pct": 0.0, "mechanism_signature": "s"}}).success
    identity = session.experiment_log[-1]["experiment"]
    decision = {"experiment": identity, "mechanism": "m", "evidence": "e", "limitations": "l"}
    refused = tools["submit_policy"](decision)
    assert not refused.success and not refused.finished and "would be scored invalid" in refused.text and not session.submitted
    # S3-FIFO itself is always within the limit.
    assert tools["submit_policy"]({**decision, "experiment": "s3fifo"}).finished


def test_run_experiment_tunes_constants_for_one_experiment(engineer_setup) -> None:
    from vista.benchmarks.cachelab.lab import tuned_variants

    _, lab = engineer_setup
    session = CacheLabSession(lab, visual=False, max_experiments=3, role="engineer", shots=True)
    tools = {binding.spec.name: binding.handler for binding in session.tools}
    request = {"name": "tuned S", "code": BIGGER_SMALL, "hypothesis": "h", "prediction": {"expected_change_pct": 0.0, "mechanism_signature": "s"}}
    reply = tools["run_experiment"]({**request, "tune": {"SMALL_RATIO": [0.05, 0.1, 0.25, 0.4]}})
    assert reply.success and "tuned 4 combinations of SMALL_RATIO" in reply.text and reply.text.count("*") >= 1, reply.text
    experiment = lab.experiment(session.experiment_log[-1]["experiment"])
    assert "class TunedCachePolicy(CachePolicy):" in experiment.source and session.tool_counts["run_experiment"] == 1
    chosen = reply.text.split("*SMALL_RATIO=")[1].split(" ")[0]
    assert f"SMALL_RATIO = {chosen}" in experiment.source
    # Only the recorded variant keeps its per-request logs.
    others = [source for assignment, source in tuned_variants(BIGGER_SMALL, {"SMALL_RATIO": [0.05, 0.1, 0.25, 0.4]}) if repr(assignment["SMALL_RATIO"]) != chosen]
    kept = lab.store.simulate_many([(others[0], trace, capacity) for trace, capacity in lab.cells[:1]], handover=lab.handover)[0]
    assert kept.summary["requests"] > 0
    for bad, message in (
        ({"NOT_A_KNOB": [1, 2]}, "change nothing"),
        ({"SMALL_RATIO": [0.1]}, "2 to 8"),
        ({"SMALL_RATIO": ["big", "small"]}, "numbers"),
        ({"SMALL_RATIO": [0.1, 0.2, 0.3, 0.4], "GHOST_RATIO": [0.5, 0.9, 1.0, 1.5]}, "at most 12"),
        ({"small_ratio": [0.1, 0.2]}, "UPPER_CASE"),
    ):
        refused = tools["run_experiment"]({**request, "tune": bad})
        assert not refused.success and message in refused.text, refused.text
    assert len(session.experiment_log) == 1  # refused requests cost nothing


def test_tune_schema_rejects_malformed_grids_before_charging() -> None:
    import jsonschema

    from vista.benchmarks.cachelab.profiles import TUNE

    validator = jsonschema.Draft202012Validator(TUNE)
    assert validator.is_valid({"SMALL_RATIO": [0.05, 0.1], "USE_GHOST": [True, False]})
    for bad in ({}, {"small_ratio": [0.1, 0.2]}, {"SMALL_RATIO": [0.1]}, {"SMALL_RATIO": [0.1, 0.1]}, {"SMALL_RATIO": ["a", "b"]}, {f"K{i}": [1, 2] for i in range(4)}):
        assert not validator.is_valid(bad), bad


def test_prefix_tuning_descends_one_knob_at_a_time(engineer_setup) -> None:
    import itertools

    from vista.benchmarks.cachelab import engineer
    from vista.benchmarks.cachelab.search import DEFAULT, policy_source

    task, lab = engineer_setup
    grid = {"SMALL_RATIO": (0.05, 0.1, 0.25), "MOVE_THRESHOLD": (1, 2, 3)}
    calls = []
    original = lab.store.simulate_many

    def counting(entries, **kwargs):
        calls.append(len(entries))
        return original(entries, **kwargs)

    lab.store.simulate_many = counting
    try:
        tuned = engineer.prefix_tuned(lab, task, grid=grid)
    finally:
        lab.store.simulate_many = original
    base = lab.experiment("s3fifo")
    for capacity in task.capacities:
        theirs = base.run(engineer.NAME, capacity).summary["miss_ratio"]
        brute = min(
            (lab.store.simulate(policy_source({**DEFAULT, **dict(zip(grid, values, strict=True))}), [(task.prefix, capacity)], handover=lab.handover)[(engineer.NAME, capacity)].summary["miss_ratio"] - theirs) / theirs
            for values in itertools.product(*grid.values())
        )
        row = tuned[capacity]
        assert brute - 1e-12 <= row["prefix_relative_change"] <= 0.0 and row["suffix_miss_ratio"] is not None
        assert all(value == DEFAULT[knob] or value in grid.get(knob, ()) for knob, value in row["config"].items())
    # Prefix runs: fewer than the full grid at every size; the last call is the full-trace run of each size.
    assert sum(calls[:-1]) < len(task.capacities) * 9 and calls[-1] == len(task.capacities)


def test_a_provider_serving_another_model_fails_the_preflight(tmp_path, monkeypatch):
    # FreeInference answered kimi-k3 requests with qwen3.6-35b while kimi-k3 was
    # overloaded; Claude Code still reports the model it asked for.
    from types import SimpleNamespace

    from vista.backends.claude import preflight
    from vista.backends.claude.runner import ServedModelMismatch, foreign_models
    from vista.core.player import ToolInterface

    assert foreign_models(["kimi-k3", "<synthetic>"], "kimi-k3") == []
    assert foreign_models(["kimi-k3", "qwen3.6-35b", "qwen3.6-35b"], "kimi-k3") == ["qwen3.6-35b"]

    served = {"models": ("kimi-k3",)}

    class FakeRunner:
        display_size = (1024, 1024)

        def __init__(self, **kwargs):
            pass

        def run_task(self, prompt):
            return SimpleNamespace(
                returncode=0, init_envelope={"model": "kimi-k3"}, resolved_models=served["models"],
                final_message="READY", stdout_path=tmp_path / "stdout.jsonl", stderr_path=tmp_path / "stderr.txt",
            )  # fmt: skip

    monkeypatch.setattr(preflight, "ClaudeCodeRunner", FakeRunner)
    monkeypatch.setattr(preflight, "validate_instruction_envelope", lambda *args, **kwargs: None)
    tools = ToolInterface(definitions=[], executor=None, action_tools=frozenset(), visual_readers=frozenset(),
                          predict_images=lambda name, arguments: 0, acknowledge=lambda call_id: None)  # fmt: skip

    def run():
        return preflight.run_runtime_preflight(
            visible_dir=tmp_path, config_dir=tmp_path, io_dir=tmp_path, claude_bin=tmp_path / "claude",
            model="kimi-k3", effort="high", timeout=60, tool_interface=tools,
        )  # fmt: skip

    assert run()[1] == "kimi-k3"
    served["models"] = ("kimi-k3", "qwen3.6-35b")
    with pytest.raises(ServedModelMismatch, match="qwen3.6-35b"):
        run()


def test_cases_are_listed_most_similar_workload_first(tmp_path):
    from vista.benchmarks.cachelab.experience import Library, signature_distance

    def sig(once, quick, mr):
        return {"requests": 10, "distinct_objects": 5, "objects_requested_once": once, "requests_to_objects_with_9plus": 0.5,
                "per_size": [{"s3fifo_miss_ratio": mr, "s_hits_quick": quick, "lru_vs_s3fifo": 0.05, "one_hit_share_C_window": None}]}  # fmt: skip

    here = sig(0.4, 0.2, 0.3)
    assert signature_distance(here, here) == 0.0
    # Length is not compared, only shape; features missing on either side are skipped.
    assert signature_distance(here, {**sig(0.4, 0.2, 0.3), "requests": 10**8}) == 0.0
    assert signature_distance(here, sig(0.5, 0.2, 0.3)) == pytest.approx(0.1 / 5)
    assert signature_distance({}, here) is None

    library = Library(tmp_path / "lib")
    for decision, signature in (("far", sig(0.9, 0.9, 0.9)), ("near", sig(0.41, 0.2, 0.3)), ("unsigned", None)):
        library.add_case({"decision": decision, "signature": signature, "outcome": {"mean_relative_vs_s3fifo": -0.01}}, "code")
    rows = library.table(here).splitlines()[1:]
    assert [row.split()[0] for row in rows] == ["case-002", "case-001", "case-003"] and "n/a" in rows[-1]
    assert [row.split()[0] for row in library.table().splitlines()[1:]] == ["case-001", "case-002", "case-003"]
    assert "Most similar workload first" in library.prompt_section(here) and "Most similar" not in library.prompt_section()


def test_the_reinsertion_oracle_scans_a_bounded_part_of_m():
    """With a long horizon nearly every key in M is requested again soon; an unbounded
    scan then costs O(C) per eviction (a view on a large cache once ran for hours)."""
    from vista.benchmarks.cachelab import headroom

    capacity, loop = 400, 520
    keys = np.tile(np.arange(loop, dtype=np.int64), 60)
    following = np.empty_like(keys)
    following[:-loop] = np.arange(loop, len(keys))
    following[-loop:] = analysis_never()
    pops = 0
    original = headroom._Timed.pop

    def counted(self):
        nonlocal pops
        pops += 1
        return original(self)

    headroom._Timed.pop = counted
    try:
        result = headroom.replay(keys, following, capacity, 0, {"reinsertion": 10 * len(keys)})
    finally:
        headroom._Timed.pop = original
    assert pops <= (headroom.SCAN + 2) * int(result["miss"].sum())


def analysis_never():
    from vista.benchmarks.cachelab.analysis import NEVER

    return NEVER


def test_headroom_replays_s3fifo_exactly_and_prices_each_decision(lab):
    from vista.benchmarks.cachelab import analysis, headroom
    from vista.benchmarks.cachelab.engine import HIT

    name = "dev-2"
    capacity = lab.capacities[name][-1]
    index = lab.index(name)
    following = analysis.next_requests(index)
    # Without an oracle the host replay is the sandbox's S3-FIFO, miss for miss.
    base = headroom.replay(index.dense, following, capacity, 0)
    assert (base["miss"] == (lab.experiment(BASELINE).run(name, capacity).outcome != HIT)).all()
    # The oracle takes over at the switch: nothing changes before it.
    switch = 10_000
    oracle = headroom.replay(index.dense, following, capacity, switch, {decision: base["m_crossing"] for decision in headroom.DECISIONS})
    assert (oracle["miss"][:switch] == base["miss"][:switch]).all() and oracle["miss"][switch:].sum() < base["miss"][switch:].sum()

    view = lab.headroom_view(name, capacity)
    changes = view.data["headroom"]
    assert all(changes[decision] < 0 for decision in (*headroom.DECISIONS, headroom.ALL)), changes
    assert changes["OPT"] <= changes[headroom.ALL] and view.png
    assert "by multiple: 0.25x" in view.text and set(view.data["factors"].values()) <= set(headroom.FACTORS)
    # Real policies for scale, from the observed trace only; they need a window that runs to the end.
    assert "3L-Cache" in changes and any(name.startswith("best other (") for name in changes)
    assert "3L-Cache" not in lab.headroom_view(name, capacity, 0, 20_000, visual=False).data["headroom"]
    cached = sorted(path.name for path in (lab.root / "headroom").iterdir())
    assert lab.headroom_view(name, capacity, visual=False).data == view.data
    assert sorted(path.name for path in (lab.root / "headroom").iterdir()) == cached


def test_the_prefix_tuned_knobs_are_offered_as_a_reference(lab):
    from vista.benchmarks.cachelab import engineer
    from vista.benchmarks.cachelab.lab import MEMORY_LIMIT_RATIO, TUNED_REFERENCE
    from vista.benchmarks.cachelab.profiles import initial_prompt
    from vista.benchmarks.cachelab.search import DEFAULT, policy_source

    caps = lab.capacities["dev-1"]
    widest = {"SMALL_RATIO": 0.3, "GHOST_RATIO": 2.0, "MOVE_THRESHOLD": 1, "MAX_FREQ": 5}  # the grid's largest ghost
    configs = {capacity: dict(widest) if i % 2 else dict(DEFAULT) for i, capacity in enumerate(caps)}
    report = {"rows": [{"capacity": c, "prefix_tuned_config": configs[c], "prefix_tuned_s3fifo": 0.5} for c in caps]}
    assert engineer.tuned_configs(report, caps) == configs
    assert engineer.tuned_configs({"rows": report["rows"][:-1]}, caps) is None and engineer.tuned_configs(None, caps) is None

    source = engineer.tuned_reference_source(configs)
    assert not check_policy(source)
    # Each size behaves exactly like its own configuration run alone.
    tuned = lab.store.simulate(source, [(lab.dev["dev-1"], c) for c in caps])
    for c in caps:
        alone = lab.store.simulate(policy_source(configs[c]), [(lab.dev["dev-1"], c)])
        assert tuned[("dev-1", c)].summary["misses"] == alone[("dev-1", c)].summary["misses"], c
    # Within the memory limit on these small synthetic traces. Not on real ones: there GHOST_RATIO 1.5-2.0
    # measured 2.3-3x S3-FIFO's memory (tl-cf-4), which is why the CLI offers this reference only on request.
    check = lab.test_policy(source)
    assert check.data["ok"] and max(check.data["memory_ratios"].values()) <= MEMORY_LIMIT_RATIO, check.text

    text = initial_prompt([], "", role="engineer", references=("lru", TUNED_REFERENCE))
    assert f"`{TUNED_REFERENCE}` is S3-FIFO with its four knobs chosen separately" in text
    assert TUNED_REFERENCE not in initial_prompt([], "", role="engineer", references=("lru",))


def test_vqsim_gate(tmp_path) -> None:
    """The TinyLFU-style gate (vqsim only): each mode acts, a fixed online schedule is the handover run,
    the miniature selector accepts gated candidates, bad gate settings are rejected."""
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    binary = tmp_path / "vqsim"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(Path(__file__).parents[1] / "scripts/vqsim/vqsim.c")], check=True)
    dense = _shifting_trace(40_000, 8, 11)
    (tmp_path / "ids.u32").write_bytes(dense.astype(np.uint32).tobytes())
    s3 = "maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
    three = "maxf=3 st=0.05,0,0,1,1,-1,0 st=0.05,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0"  # entry passes everything to a probation stage
    gated = [f"{s3} gate=1 gmode={m}" for m in (0, 1, 2)] + [f"{three} gate=2 gmode=0"]

    def run(jobs: str) -> subprocess.CompletedProcess:
        (tmp_path / "jobs").write_text(jobs)
        return subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True, check=True)

    out = run("".join(f"200 8000 40 {c} handover=1\n" for c in (s3, *gated, three))).stdout.splitlines()
    after = [int(line.split()[1]) for line in out]
    assert all(a != after[0] for a in after[1:4]) and after[4] != after[5], after  # every mode acts, in a probation design too
    for c in gated:
        ho, one = run(f"200 8000 40 {c} handover=1\nonline 200 8000 40 schedule=s:1\n{s3}\n{c}\nend\n").stdout.splitlines()
        assert one.split()[:3] == ho.split()[:2] + ["0"], c
    mini = run(f"online 200 8000 40 rate=0.3 min=50\n{s3}\n" + "".join(c + "\n" for c in gated) + "end\n").stdout.split()
    assert len(mini) == 43 and int(mini[1]) > 0
    bad = [f"{s3} gate=0", f"{s3} gate=1 gmode=3", f"{s3} gate=2", f"{s3} gate=1 gwin=0"]
    assert all(line.startswith("error") for line in run("".join(f"200 8000 40 {c}\n" for c in bad)).stdout.splitlines())
    assert run(f"online 200 8000 40 rate=0.3 min=50\n{gated[0]}\n{gated[1]} gwin=4\nend\n").stdout.startswith("error bad-config 1")


def test_vqsim_reuse_distance_rule(tmp_path) -> None:
    """The reuse-distance rule (vqsim only, VQ_RD): rd=0 is the design itself, each rdmode bit acts on a trace with
    regular reuse, a fixed online schedule is the handover run, the selector accepts rd candidates, the deployable
    selector (vqonline.h) rejects them, bad settings are rejected."""
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    root = Path(__file__).parents[1] / "scripts/vqsim"
    binary = tmp_path / "vqsim"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(root / "vqsim.c")], check=True)
    rng = np.random.default_rng(5)  # objects requested with a period each (short to long) during a lifetime, plus one-hit noise
    n_obj, horizon = 6000, 120_000
    periods, start, life = rng.integers(5, 1500, n_obj), rng.integers(0, horizon, n_obj), rng.integers(2000, 30_000, n_obj)
    events = sorted((t, k) for k in range(n_obj) for t in range(int(start[k]), int(min(horizon, start[k] + life[k])), int(periods[k])))
    keys = np.array([k for _, k in events])
    keys = np.where(rng.random(keys.size) < 0.3, 10_000 + np.arange(keys.size), keys)
    _, dense = np.unique(keys, return_inverse=True)
    (tmp_path / "ids.u32").write_bytes(dense.astype(np.uint32).tobytes())
    split = dense.size // 5
    s3 = "maxf=3 st=0.1,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0"  # S3-FIFO, promoting on one hit in S

    def run(jobs: str) -> list[str]:
        (tmp_path / "jobs").write_text(jobs)
        return subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True, check=True).stdout.splitlines()

    variants = [f"{s3} rd=0 rdmode=15"] + [f"{s3} rd={x} rdmode={m}" for m in (1, 2, 4, 8) for x in (0.01, 4)]
    out = run("".join(f"1000 {split} 40 {c} handover=1\n" for c in (s3, *variants)))
    after = [int(line.split()[1]) for line in out]
    assert after[1] == after[0], after  # rd=0: off
    for j, m in enumerate((1, 2, 4, 8)):
        assert after[0] != after[2 + 2 * j] or after[0] != after[3 + 2 * j], (m, after)  # each bit acts
    for c in (variants[3], variants[6]):  # reinsertion, ghost admission
        ho, one = run(f"1000 {split} 40 {c} handover=1\nonline 1000 {split} 40 schedule=s:1\n{s3}\n{c}\nend\n")
        assert one.split()[:3] == ho.split()[:2] + ["0"], c
    # takeover=3 merges by stage entry time at switches only: the takeover at the split is unchanged
    three = "maxf=3 st=0.05,0,0,1,1,-1,0 st=0.05,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0"
    plain, merged = (run(f"online 1000 {split} 40 schedule=s:1,20:2{t}\n{s3}\n{three}\n{variants[6]}\nend\n")[0].split() for t in ("", " takeover=3"))
    assert plain[0] == merged[0] and plain[2] == merged[2] == "1"  # same prefix, one switch (at window 20)
    assert plain[3:3 + 20] == merged[3:3 + 20] and plain[3 + 20:] != merged[3 + 20:]  # identical until the switch
    mini = run(f"online 1000 {split} 40 rate=0.3 min=50\n{s3}\n" + "".join(c + "\n" for c in variants[1:]) + "end\n")[0].split()
    assert len(mini) == 43 and int(mini[1]) > 0
    bad = [f"{s3} rd=-1", f"{s3} rd=1 rdmode=16", f"{s3} rdmode=-1"]
    assert all(line.startswith("error") for line in run("".join(f"1000 {split} 40 {c}\n" for c in bad)))
    online = tmp_path / "vqonline"
    subprocess.run(["cc", "-O2", "-o", str(online), str(root / "vqonline.c")], check=True)
    dense.astype(np.uint64).tofile(tmp_path / "keys.u64")
    (tmp_path / "cands").write_text(f"{s3}\n{variants[6]}\n")
    rejected = subprocess.run([str(online), str(tmp_path / "keys.u64"), str(tmp_path / "cands"), "1000"], capture_output=True, text=True)
    assert rejected.returncode != 0 or "error" in (rejected.stdout + rejected.stderr).lower()


def test_vqsim_pool_screening(tmp_path) -> None:
    """Pool screening at the split (vqsim online pool=, screen=): the designs it adds behave exactly as if they had been
    candidates from the start (the replayed prefix is the miniatures' own sample), gated ones included; a pool line
    equal to a candidate is skipped; bad settings are rejected."""
    import os
    import re
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    binary = tmp_path / "vqsim"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(Path(__file__).parents[1] / "scripts/vqsim/vqsim.c")], check=True)
    dense = _shifting_trace(40_000, 8, 11)
    (tmp_path / "ids.u32").write_bytes(dense.astype(np.uint32).tobytes())
    s3 = "maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
    three = "maxf=3 st=0.05,0,0,1,1,-1,0 st=0.05,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0"
    pool = [s3, f"{s3} gate=1 gmode=1", three, f"{three} gate=2 gmode=0", "maxf=1 st=0.02,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0",
            f"{s3} rd=1 rdmode=4", "maxf=2 st=0.3,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0"]
    (tmp_path / "pool").write_text("".join(c + "\n" for c in pool))
    kept_pool = pool[1:]  # s3 is a candidate: skipped

    def run(jobs: str, log: bool = False) -> subprocess.CompletedProcess:
        (tmp_path / "jobs").write_text(jobs)
        env = dict(os.environ, VQSIM_SCREEN_LOG="1") if log else {k: v for k, v in os.environ.items() if k != "VQSIM_SCREEN_LOG"}
        return subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True, check=True, env=env)

    head = "online 200 8000 40 rate=0.3 min=50"
    for m in (1, 3, 6):
        got = run(f"{head} pool={tmp_path / 'pool'} screen={m}\n{s3}\nend\n", log=True)
        chosen = [int(x) for x in re.findall(r"pool line (\d+)", got.stderr)]
        assert len(chosen) == m and len(set(chosen)) == m, got.stderr
        plain = run(f"{head}\n{s3}\n" + "".join(kept_pool[j] + "\n" for j in chosen) + "end\n").stdout
        assert got.stdout == plain and len(plain.split()) == 43, (m, chosen)
    clipped = run(f"{head} pool={tmp_path / 'pool'} screen=50\n{s3}\nend\n", log=True)
    assert len(re.findall(r"pool line", clipped.stderr)) == len(kept_pool)
    # a log bound the prefix never reaches changes nothing; a short one wraps, in either wrap mode
    full = run(f"{head} pool={tmp_path / 'pool'} screen=3\n{s3}\n{three}\nend\n").stdout
    assert run(f"{head} pool={tmp_path / 'pool'} screen=3 logcap=1000000 wrapmode=1\n{s3}\n{three}\nend\n").stdout == full
    for mode in (0, 1):
        short = run(f"{head} pool={tmp_path / 'pool'} screen=3 logcap=400 wrapmode={mode}\n{s3}\n{three}\nend\n").stdout.split()
        assert len(short) == 43 and int(short[0]) == int(full.split()[0])  # the prefix is S3-FIFO's either way
    for bad in (f"{head} screen=2", f"online 200 8000 40 schedule=s:0 pool={tmp_path / 'pool'} screen=2",
                f"{head} start=4 pool={tmp_path / 'pool'} screen=2", f"{head} pool={tmp_path / 'missing'} screen=2"):
        assert run(f"{bad}\n{s3}\nend\n").stdout.startswith("error"), bad


def test_vqsim_trial_before_switch(tmp_path) -> None:
    """trial=H (vqsim online): with rate=1 the mirror is the live cache itself, so each trial's 'stay' count is the live
    cache's misses since the snapshot, and its replay count is what a live cache that really switched there would have
    missed (a fixed schedule with the same history); a switch is made only when the replay wins. Bad settings fail."""
    import os
    import re
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    binary = tmp_path / "vqsim"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(Path(__file__).parents[1] / "scripts/vqsim/vqsim.c")], check=True)
    dense = _shifting_trace(40_000, 8, 11)
    (tmp_path / "ids.u32").write_bytes(dense.astype(np.uint32).tobytes())
    s3 = "maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
    three = "maxf=3 st=0.05,0,0,1,1,-1,0 st=0.05,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0"
    cands = [s3, three, f"{s3} gate=1 gmode=1", "maxf=1 st=0.02,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0",
             "maxf=2 st=0.3,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0", f"{three} gate=2 gmode=0"]
    body = "".join(c + "\n" for c in cands) + "end\n"

    def run(head: str) -> tuple[str, str]:
        (tmp_path / "jobs").write_text(f"{head}\n{body}")
        got = subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True,
                             check=True, env=dict(os.environ, VQSIM_TRACE="1"))
        return got.stdout, got.stderr

    checked = 0
    for h_windows in (1, 2):
        raw, err = run(f"online 200 8000 40 rate=1 min=50 trial={h_windows}")
        out = [int(x) for x in raw.split()]
        assert len(out) == 43
        win = out[3:]
        history: list[tuple[int, int]] = []  # (window, design) of each switch the live cache made
        trials = 0
        for line in err.splitlines():
            if m := re.match(r"trial w=(\d+) live=(-?\d+) leader=(\d+) h=(\d+) stay=(\d+) pass=(\d)", line):
                w, live, leader, h, stay, ok = (int(x) for x in m.groups())
                trials += 1
                w0 = w - h_windows  # the oldest snapshot: a decision point (or before the split) h_windows windows back
                assert stay == sum(win[w0:w]), line
                assert ok == (h < stay), line
                if all(sw < w0 for sw, _ in history):  # the snapshot's own decision made no switch: a schedule can replay it
                    sched = ",".join(f"{sw}:{d}" for sw, d in history) + ("," if history else "") + f"{w0}:{leader}"
                    ref = [int(x) for x in run(f"online 200 8000 40 schedule={sched}")[0].split()]
                    assert sum(ref[3:][w0:w]) == h, (line, sched)
                    checked += 1
                if ok:
                    history.append((w, leader))
            elif m := re.match(r"decide w=(\d+) live=(-?\d+) leader=(\d+)", line):
                w, live, leader = (int(x) for x in m.groups())
                assert live == (history[-1][1] if history else -1), line
        assert trials > 0 and out[2] == max(len(history) - 1, 0)
    assert checked >= 3
    for bad in ("online 200 8000 40 schedule=s:1 trial=1", "online 200 8000 40 rate=1 start=4 trial=1", "online 200 8000 40 trial=-1"):
        assert run(bad)[0].startswith("error bad-online"), bad


def test_vqsim_reseed_at_split() -> None:
    """reseed=1 (vqsim online): at the split every miniature takes over the mirror's (S3-FIFO's) cache. With rate=1 the
    miniatures are exact, so each later leader is the design with the fewest misses over its cold run's windows before
    the split plus a live run's that took over S3-FIFO's cache at the split (schedule=s:i) after it."""
    import os
    import re
    import shutil
    import subprocess
    import tempfile

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    tmp = Path(tempfile.mkdtemp())
    binary = tmp / "vqsim"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(Path(__file__).parents[1] / "scripts/vqsim/vqsim.c")], check=True)
    (tmp / "ids.u32").write_bytes(_shifting_trace(40_000, 8, 11).astype(np.uint32).tobytes())
    s3 = "maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
    three = "maxf=3 st=0.05,0,0,1,1,-1,0 st=0.05,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0"
    cands = [s3, three, f"{s3} gate=1 gmode=1", "maxf=1 st=0.02,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0",
             "maxf=2 st=0.3,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0", "maxf=1 entry=0 overflow=1 st=0.05,1,9,1,0,-1,1 st=0,1,1,1,2,-1,0"]
    body = "".join(c + "\n" for c in cands) + "end\n"

    def run(jobs: str) -> tuple[list[list[int]], str]:
        (tmp / "jobs").write_text(jobs)
        got = subprocess.run([str(binary), str(tmp / "ids.u32"), str(tmp / "jobs")], capture_output=True, text=True, check=True,
                             env=dict(os.environ, VQSIM_TRACE="1"))
        return [[int(x) for x in line.split()] for line in got.stdout.splitlines()], got.stderr

    cold = [r[2:] for r in run("".join(f"200 8000 40 {c}\n" for c in cands))[0]]
    took = [run(f"online 200 8000 40 schedule=s:{i}\n{body}")[0][0][3:] for i in range(len(cands))]
    _, err = run(f"online 200 8000 40 rate=1 min=50 reseed=1\n{body}")
    decisions = [tuple(int(x) for x in m.groups()) for m in re.finditer(r"decide w=(\d+) live=-?\d+ leader=(\d+)", err)]
    assert len(decisions) == 32
    for w, leader in decisions:
        score = [sum(cold[i][2:8]) + sum(took[i][8:w]) for i in range(len(cands))]
        assert leader == score.index(min(score)), (w, leader, score)
    assert len({leader for _, leader in decisions}) > 1  # the scores do move the choice
    plain, _ = run(f"online 200 8000 40 rate=1 min=50\n{body}")
    reseeded, _ = run(f"online 200 8000 40 rate=1 min=50 reseed=1\n{body}")
    p, q = plain[0], reseeded[0]
    assert p[0] == q[0] and p[3:3 + 9] == q[3:3 + 9] and p[3:] != q[3:]  # the same through the split's window, then not
    for bad in ("online 200 8000 40 schedule=s:1 reseed=1", "online 200 8000 40 start=4 reseed=1"):
        (tmp / "jobs").write_text(f"{bad}\n{body}")
        out = subprocess.run([str(binary), str(tmp / "ids.u32"), str(tmp / "jobs")], capture_output=True, text=True, check=True).stdout
        assert out.startswith("error bad-online"), bad


def test_vqsim_replay_takeover() -> None:
    """replay=L (vqsim online): the next design rebuilds its state from the logged requests and is reconciled with what
    is cached. S3-FIFO taking over S3-FIFO from a log of the whole history changes nothing; the live records stay
    consistent after every takeover (VQSIM_CHECK), for multi-stage, gated and static designs, with short and long logs."""
    import os
    import shutil
    import subprocess
    import tempfile

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    tmp = Path(tempfile.mkdtemp())
    binary = tmp / "vqsim"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(Path(__file__).parents[1] / "scripts/vqsim/vqsim.c")], check=True)
    (tmp / "ids.u32").write_bytes(_shifting_trace(40_000, 8, 11).astype(np.uint32).tobytes())
    s3 = "maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
    three = "maxf=3 st=0.05,0,0,1,1,-1,0 st=0.05,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0"
    cands = [s3, three, f"{s3} gate=1 gmode=1", "maxf=1 st=0.02,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0",
             "maxf=2 st=0.3,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0 ghost=0.5", f"{three} gate=2 gmode=0",
             "maxf=1 entry=0 overflow=1 st=0.05,1,9,1,0,-1,1 st=0,1,1,1,2,-1,0"]
    body = "".join(c + "\n" for c in cands) + "end\n"

    def run(jobs: str) -> list[list[int]]:
        (tmp / "jobs").write_text(jobs)
        got = subprocess.run([str(binary), str(tmp / "ids.u32"), str(tmp / "jobs")], capture_output=True, text=True, check=True,
                             env=dict(os.environ, VQSIM_CHECK="1"))
        assert "check failed" not in got.stderr, got.stderr[:2000]
        return [[int(x) for x in line.split()] if line[0].isdigit() else line for line in got.stdout.splitlines()]

    plain = run(f"200 8000 40 {s3}\n")[0]
    assert run(f"online 200 8000 40 schedule=s:0 replay=1000\n{body}")[0][3:] == plain[2:]
    for rp in (0.5, 4, 1000):
        jobs = "".join(f"online 200 8000 40 schedule=s:{i},20:{(i + 2) % 7},30:{(i + 5) % 7} replay={rp}\n{body}" for i in range(7))
        outs = run(jobs + f"online 200 8000 40 rate=0.3 min=50 replay={rp}\n{body}")
        assert all(len(o) == 43 for o in outs) and outs[0][2] == 2
    for bad in ("online 200 8000 40 rate=0.3 min=50 replay=4 trial=1", "online 200 8000 40 rate=0.3 min=50 replay=4 reseed=1",
                "online 200 8000 40 rate=0.3 min=50 replay=-1"):
        assert run(f"{bad}\n{body}")[0].startswith("error bad-online"), bad


def test_evolve_polsim_contract(tmp_path) -> None:
    """The evolve harness: the S3-FIFO seed in the policy.h contract, run by polsim, misses exactly as vqsim's S3-FIFO
    (ghost compaction included); a policy naming an uncached victim is caught; one that uses the heap is rejected."""
    import importlib.util
    import shutil
    import subprocess

    if shutil.which("cc") is None or shutil.which("nm") is None:
        pytest.skip("no C toolchain")
    ev = Path(__file__).parents[1] / "scripts/vqsim/evolve"
    spec = importlib.util.spec_from_file_location("evolve", ev / "evolve.py")
    evolve = importlib.util.module_from_spec(spec); spec.loader.exec_module(evolve)
    binary, err = evolve.build(ev / "seeds/s3fifo.c", tmp_path)
    assert binary and not err, err
    subprocess.run(["cc", "-O2", "-o", str(tmp_path / "vqsim"), str(Path(__file__).parents[1] / "scripts/vqsim/vqsim.c")], check=True)
    rng = np.random.default_rng(7)
    ids = (rng.zipf(2.05, 600_000) % 400_000).astype(np.uint32)
    (tmp_path / "ids.u32").write_bytes(ids.tobytes())
    caps, n = [1, 10, 300, 3000, 30000], len(ids)
    s3 = "maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
    (tmp_path / "jobs").write_text("".join(f"{c} {n // 5} 40 {s3}\n" for c in caps))
    ref = subprocess.run([str(tmp_path / "vqsim"), str(tmp_path / "ids.u32"), str(tmp_path / "jobs")], capture_output=True, text=True, check=True).stdout
    got = subprocess.run([str(binary), str(tmp_path / "ids.u32"), str(n // 5), "40"] + [str(c) for c in caps], capture_output=True, text=True, check=True).stdout
    assert got == ref
    header = '#include "policy.h"\nsize_t policy_bytes(uint64_t c) { (void)c; return 64; }\nvoid policy_init(void *a, uint64_t c) { (void)a; (void)c; }\n'
    header += "void policy_hit(void *a, uint64_t k) { (void)a; (void)k; }\nint policy_admit(void *a, uint64_t k) { (void)a; (void)k; return 1; }\n"
    header += "void policy_insert(void *a, uint64_t k) { (void)a; (void)k; }\n"
    (tmp_path / "cheat.c").write_text(header + "uint64_t policy_victim(void *a) { (void)a; return 12345; }\n")
    (tmp_path / "c").mkdir()
    cheat, err = evolve.build(tmp_path / "cheat.c", tmp_path / "c")
    assert cheat and not err, err
    out = subprocess.run([str(cheat), str(tmp_path / "ids.u32"), str(n // 5), "40", "10"], capture_output=True, text=True, check=True).stdout
    assert out.startswith("error victim")
    (tmp_path / "heap.c").write_text("#include <stdlib.h>\n" + header + "uint64_t policy_victim(void *a) { (void)a; return (uint64_t)(size_t)malloc(8); }\n")
    (tmp_path / "h").mkdir()
    heap, err = evolve.build(tmp_path / "heap.c", tmp_path / "h")
    assert heap is None and "malloc" in err
    # knobs: the share seed held at setting 0 is S3-FIFO; with the harness picking, it reports its setting changes
    (tmp_path / "k").mkdir()
    share, err = evolve.build(ev / "seeds/s3fifo_share.c", tmp_path / "k")
    assert share and not err, err
    assert subprocess.run([str(share), "--knobs"], capture_output=True, text=True, check=True).stdout.strip() == "8"
    assert subprocess.run([str(binary), "--knobs"], capture_output=True, text=True, check=True).stdout.strip() == "1"
    argv = [str(share), str(tmp_path / "ids.u32"), str(n // 5), "40"] + [str(c) for c in caps]
    fixed = subprocess.run(argv, capture_output=True, text=True, check=True, env={**os.environ, "POLSIM_KNOB": "0"})
    assert fixed.stdout == ref and all(l.split()[2:] == ["0", "0"] for l in fixed.stderr.splitlines() if l.startswith("knobs "))
    gated = subprocess.run(argv, capture_output=True, text=True, check=True)
    lines = gated.stdout.splitlines()
    assert len(lines) == len(caps) and not any(l.startswith("error") for l in lines)
    assert [l.split()[1] for l in gated.stderr.splitlines() if l.startswith("knobs ")] == [str(c) for c in caps]


@pytest.fixture(scope="module")
def vq_board(tmp_path_factory):
    import shutil
    import subprocess

    from vista.benchmarks.cachelab.architect import Board

    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    root = tmp_path_factory.mktemp("vq")
    binary = root / "vqsim"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(Path(__file__).parents[1] / "scripts/vqsim/vqsim.c")], check=True)
    suite = root / "suite"
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        keys = (rng.zipf(1.1 + 0.1 * seed, size=30_000) % 6_000).astype(np.int64)
        keys[rng.choice(keys.size, keys.size // 5, replace=False)] = np.arange(10**6, 10**6 + keys.size // 5)
        _, dense = np.unique(keys, return_inverse=True)
        (suite / f"t{seed}").mkdir(parents=True)
        (suite / f"t{seed}" / "ids.u32").write_bytes(dense.astype(np.uint32).tobytes())
        meta = {"requests": keys.size, "split": keys.size // 5, "capacities": [60, 400]}
        meta["s3fifo"], meta["3L"], meta["best"] = {}, {}, {}
        for capacity in meta["capacities"]:
            jobs = root / "jobs"
            jobs.write_text(f"{capacity} {meta['split']} 0 maxf=3 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0 handover=1\n")
            after = int(subprocess.run([str(binary), str(suite / f"t{seed}" / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout.split()[1])
            ratio = after / (keys.size - meta["split"])
            meta["s3fifo"][str(capacity)], meta["3L"][str(capacity)], meta["best"][str(capacity)] = ratio, ratio * 0.98, ["lru", ratio * 0.97]
        (suite / f"t{seed}" / "meta.json").write_text(json.dumps(meta))
    return Board(root / "board", suite, binary, slots=2)


def test_vq_families_expand_and_reject() -> None:
    from vista.benchmarks.cachelab.architect import S3_CONFIG, FamilyError, expand

    s3 = {"name": "s3", "stages": [{"name": "S", "size": 0.1, "promote": 2, "up": "M", "ghost": True}, {"name": "M", "promote": 1, "up": "M", "onup": "dec"}]}
    [(label, config)] = expand(s3)
    assert label == "(single design)" and config.split()[-2:] == ["st=0.1,1,2,1,0,-1,1,0.0,0.0", "st=0,1,1,1,2,-1,0,0.0,0.0"]
    assert set(S3_CONFIG.split()[:6]) <= set(config.split())
    grid = {**s3, "ghost_ratio": [0.5, 0.9], "stages": [{**s3["stages"][0], "size": [0.05, 0.1, 0.2]}, s3["stages"][1]]}
    assert len(expand(grid)) == 6
    for bad in (
        {**s3, "stages": [{"name": "S", "size": 0.1, "up": "X"}, s3["stages"][1]]},  # unknown stage
        {**s3, "stages": [s3["stages"][0], {"name": "M", "promote": 1, "up": "M", "onup": "keep"}]},  # reinsertion keeps
        {**s3, "ghost_ratio": 2.5},  # over the memory estimate
        {**s3, "max_freq": [1, 2, 3, 4, 5, 6, 7], "ghost_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},  # too many designs
        {**s3, "colour": "red"},
        {**s3, "stages": [s3["stages"][0], {"name": "M", "promote": 1, "up": "M", "onup": "dec", "down": "M"}]},  # demotes into itself
        {  # H and M demote into each other: nothing leaves either
            **s3,
            "stages": [s3["stages"][0], {"name": "H", "size": 0.5, "promote": 1, "up": "M", "down": "M"}, {"name": "M", "promote": 1, "up": "M", "onup": "dec", "down": "H"}],
        },
        {**s3, "stages": [{"name": "K", "size": 0.05, "counts": False, "promote": 0, "up": "S"}, {"name": "S", "size": 0.1, "promote": 0, "up": "K"}, s3["stages"][1]]},
    ):
        with pytest.raises(FamilyError):
            expand(bad)


def test_vq_architect_session_evaluates_on_the_shared_board(vq_board) -> None:
    from vista.benchmarks.cachelab.architect import (
        S3_CONFIG,
        ArchitectSession,
        system_prompt,
    )

    board = vq_board
    assert len(board.cells) == 4 and all(cell.id.startswith("c") for cell in board.cells)
    assert board.score({cell.id: S3_CONFIG for cell in board.cells}) == {cell.id: 0.0 for cell in board.cells}
    session = ArchitectSession(board, author="a1", max_evaluations=3, direction="two-level main")
    tools = {binding.spec.name: binding for binding in session.tools}
    assert "t1" not in session.initial and "two-level main" in session.initial and "Principles" in system_prompt(3)
    family = {
        "name": "hot-main",
        "stages": [
            {"name": "S", "size": [0.05, 0.1], "promote": 2, "up": "M", "ghost": True},
            {"name": "H", "size": 0.3, "promote": 1, "up": "H", "onup": "half", "down": "M"},
            {"name": "M", "promote": 1, "up": "H", "onup": "dec"},
        ],
    }
    reply = tools["evaluate_family"].handler({"family": family, "hypothesis": "hot objects stay in H"})
    assert reply.success and "Family F1" in reply.text and "beats 3L-Cache" in reply.text, reply.text
    stats = board.families()[0]["stats"]
    assert stats["designs"] == 2 and sum(d["picked"] for d in stats["per_design"]) == 4
    assert stats["pool_after"]["mean"] <= stats["pool_before"]["mean"] + 1e-12 or stats["pool_picks"] > 0
    bad = tools["evaluate_family"].handler({"family": {"name": "x", "stages": [{"name": "S", "up": "Q"}]}, "hypothesis": "h"})
    assert not bad.success and "Invalid family" in bad.text
    vqsim, board.vqsim = board.vqsim, Path("/bin/false")  # a failed simulation is a failed tool call, not a crash
    failed = tools["evaluate_family"].handler({"family": {**family, "max_freq": 2}, "hypothesis": "h"})
    board.vqsim = vqsim
    assert not failed.success and "evaluation failed" in failed.text
    other = ArchitectSession(board, author="a2", max_evaluations=3)
    assert "F1 hot-main (a1" in other.initial  # the board is shared
    assert tools["record_finding"].handler({"text": "H helps at large caches", "families": [1]}).success
    assert "H helps" in tools["view_board"].handler({}).text
    assert "d02" in tools["view_family"].handler({"family": 1}).text
    assert "c004" in tools["view_cells"].handler({"family": 1}).text
    assert not tools["submit_families"].handler({"families": [9], "summary": "s"}).success
    session.used = session.max_evaluations  # out of evaluations: refused, but the session can still submit
    refused = tools["evaluate_family"].handler({"family": family, "hypothesis": "h"})
    assert not refused.success and "No evaluations left" in refused.text
    done = tools["submit_families"].handler({"families": [1], "summary": "two-level main"})
    assert done.finished and session.result()["submission"]["families"] == [1]
    runs = board.db().execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    tools_again = {binding.spec.name: binding for binding in other.tools}
    tools_again["evaluate_family"].handler({"family": family, "hypothesis": "again"})
    assert board.db().execute("SELECT COUNT(*) FROM runs").fetchone()[0] == runs  # cached: no new simulation


def test_vq_online_board_scores_the_selectors_portfolio(vq_board) -> None:
    from vista.benchmarks.cachelab.architect import (
        S3_CONFIG,
        ArchitectSession,
        Board,
        system_prompt,
    )

    suite = vq_board.cells[0].directory.parent
    board = Board(vq_board.root.parent / "online", suite, vq_board.vqsim, slots=2, objective="online")
    used, changes, led = board.online([S3_CONFIG])
    assert used == [S3_CONFIG] and set(changes.values()) == {0.0} and set(led.values()) == {0}
    designs = [
        "maxf=1 ghost=0.5 st=0.05,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0",
        "maxf=3 st=0.2,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0",
        "maxf=3 entry=0 overflow=2 gentry=2 st=0.03,0,0,1,1,-1,0 st=0.07,1,2,2,0,-1,1 st=0,1,1,2,2,-1,0",
    ]
    # A portfolio of 2 is S3-FIFO plus the design that helps the online choice most.
    pairs = {d: statistics.mean(board.online([S3_CONFIG, d])[1].values()) for d in designs}
    portfolio, two, _ = board.online(designs, k=2)
    assert portfolio == [S3_CONFIG, min(pairs, key=pairs.get)] and statistics.mean(two.values()) == pytest.approx(min(pairs.values()))
    assert board.online(designs, k=10)[0] == designs  # k over the candidates: all of them
    rows = board.db().execute("SELECT value FROM runs WHERE mode='full'").fetchall()
    assert rows and all(len(row[0].split()) == 42 for row in rows)
    session = ArchitectSession(board, author="a1", max_evaluations=2)
    tools = {binding.spec.name: binding for binding in session.tools}
    assert "online selector" in system_prompt(2, "online") and "online selector" not in system_prompt(2)
    family = {"name": "s-size", "stages": [{"name": "S", "size": [0.05, 0.2], "promote": 2, "up": "M", "ghost": True}, {"name": "M", "promote": 1, "up": "M", "onup": "dec"}]}
    reply = tools["evaluate_family"].handler({"family": family, "hypothesis": "S size per phase"})
    assert reply.success and "enter the portfolio" in reply.text and "greedy portfolio" in reply.text, reply.text
    stats = board.families()[0]["stats"]
    assert stats["objective"] == "online" and stats["pool_after"]["mean"] <= stats["pool_before"]["mean"] + 1e-12
    assert sum(d["picked"] for d in stats["per_design"]) + sum(c < 0 for c in stats["chosen"].values()) == 4
    assert "online over a greedy portfolio" in tools["view_board"].handler({}).text and "c004" in tools["view_cells"].handler({"family": 1}).text
