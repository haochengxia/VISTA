"""CacheLab: simulator fidelity, evaluator integrity, and the VISTA task wiring.

Uses a small trace suite and the local sandbox so it runs in about a minute.
The scripted session test drives the real TaskRuntime tool path (schemas,
evidence archiving, inspect/read_pixels/history, submission) without a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vista.benchmarks.cachelab import traces
from vista.benchmarks.cachelab.engine import BASELINE_SOURCE, SANDBOX_DIR, LocalSandbox
from vista.benchmarks.cachelab.lab import BASELINE, Lab
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
    assert result.success and len(result.images) == 1 and "prediction check" in result.text
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

    def feedback(identity: str) -> dict:
        return engineer.evaluate(lab, task, identity, zoo_dir=zoo_dir, baselines=baselines, verified=False)

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
    assert not tools["record_experience"]({"decision": "d", "lesson": "l", "playbook": "p"}).success
    decision = {"experiment": identity, "mechanism": "bigger S", "evidence": "e", "limitations": "l"}
    reply = tools["commit_decision"](decision)
    assert reply.success and "Outcome on the remaining requests" in reply.text and not reply.finished
    assert not session.submitted
    assert not tools["commit_decision"](decision).success  # only once
    expected = engineer.evaluate(lab, task, identity, zoo_dir=zoo_dir, tune=False, verified=False)
    assert [row["candidate"] for row in session.feedback_report["rows"]] == pytest.approx([row["candidate"] for row in expected["rows"]])
    from vista.benchmarks.cachelab.run import counterfactuals

    labels = {record["experiment"]: record for record in counterfactuals(lab, feedback)}
    assert {"s3fifo", "mq", identity} <= set(labels) and labels["s3fifo"]["mean_relative_vs_s3fifo"] == 0.0
    assert labels[identity]["mean_relative_vs_s3fifo"] == pytest.approx(session.feedback_report["mean_relative_vs_s3fifo"])
    assert labels["mq"]["reference"] and not labels[identity]["reference"] and labels[identity]["source"] == BIGGER_SMALL
    reply = tools["record_experience"]({"decision": "S3-FIFO with S=25%", "lesson": "a bigger S helps scans", "playbook": "- scans: S=25%"})
    assert reply.finished and session.submitted and session.submission_record["case"] == "case-001"
    (case, code) = library.case("case-001")
    assert code == BIGGER_SMALL and case["signature"]["cache_sizes"] == list(task.capacities)
    assert case["outcome"]["mean_relative_vs_s3fifo"] == pytest.approx(session.feedback_report["mean_relative_vs_s3fifo"])
    assert library.playbook == "- scans: S=25%"

    # A test session reads the library and may deploy S3-FIFO or a reference unchanged.
    reader = CacheLabSession(lab, visual=False, max_experiments=3, role="engineer", shots=True, library=library)
    tools = {binding.spec.name: binding.handler for binding in reader.tools}
    assert "submit_policy" in tools and "commit_decision" not in tools and "record_experience" not in tools
    assert "- scans: S=25%" in reader.initial and "case-001" in reader.initial
    texts = [reader.initial, tools["read_case"]({"case": "case-001"}).text]
    assert "class CachePolicy(S3FIFO)" in texts[-1]
    missing = tools["read_case"]({"case": "case-009"})
    assert not missing.success and "case-001" in missing.text
    for text in [*texts, session.initial, reply.text]:
        assert SECRET not in text and str(lab.root) not in text and "private" not in text
    assert tools["submit_policy"]({"experiment": "mq", "mechanism": "m", "evidence": "e", "limitations": "l"}).finished

    other = Library(tmp_path / "other")
    other.add({"signature": case["signature"], "decision": "keep S3-FIFO", "outcome": {}}, "", "- default: keep S3-FIFO")
    merged = Library(merge([library.root, other.root], tmp_path / "merged"))
    assert [c["case"] for c in merged.cases()] == ["case-001", "case-002"]
    assert "- scans: S=25%" in merged.playbook and "- default: keep S3-FIFO" in merged.playbook
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
