"""Drive one research run, then score it on the host with held-out traces."""

from __future__ import annotations

import importlib.metadata
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vista.backends.preflight import source_fingerprint
from vista.backends.session import BackendSession
from vista.core.artifacts import ArtifactStore
from vista.core.contracts import ObservationBundle
from vista.core.json import canonical
from vista.core.profiles import Phase
from vista.core.runtime import BudgetExhausted, TaskRuntime

from .engine import simulator_fingerprint
from .lab import BASELINE, Lab
from .profiles import make_experiment, make_profile, protocol
from .session import CacheLabSession


def research_loop(session: CacheLabSession, runtime: TaskRuntime, backend: BackendSession) -> None:
    """Keep the agent working until it submits or a budget ends the task."""
    started = False
    redirect = ""
    try:
        while not session.submitted and not runtime.terminal:
            if started:
                task_input = runtime.prepare_input(Phase.STEP, public_text=redirect, observation=ObservationBundle())
            else:
                task_input = runtime.prepare_input(Phase.INITIAL)
            result = backend.run(task_input)
            started = True
            redirect = ""
            if runtime.terminal or result.returncode != 0 or runtime.failure:
                return
            if not session.submitted:
                reply = runtime.submit(result.final_message)
                redirect = reply.text
    except BudgetExhausted:
        return


def best_dev_experiment(lab: Lab) -> str | None:
    base = lab.experiment(BASELINE)
    scored = [
        (lab.compare(experiment, base)["mean_relative_change"], experiment.id)
        for experiment in lab.experiments.values()
        if experiment.ok and not lab.is_reference(experiment.id)
    ]
    return min(scored)[1] if scored else None


def counterfactuals(lab: Lab, train_feedback: Callable[[str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Every completed experiment of a training run, scored on the rest of the
    workload. Host-side data for learning decisions later; the agent never sees
    it. Fast simulation, so each costs one full-trace run per cache size."""
    base = lab.experiment(BASELINE)
    records = []
    for experiment in lab.experiments.values():
        if not experiment.ok:
            continue
        report = train_feedback(experiment.id)
        records.append(
            {
                "experiment": experiment.id,
                "name": experiment.name,
                "reference": lab.is_reference(experiment.id),
                "source": experiment.source,
                "prefix_relative_vs_s3fifo": lab.compare(experiment, base)["mean_relative_change"],
                "suffix": [
                    {"capacity": row["capacity"], "miss_ratio": row["candidate"], "relative_vs_s3fifo": row["relative_vs_s3fifo"]}
                    for row in report["rows"]
                ],
                "mean_relative_vs_s3fifo": report["mean_relative_vs_s3fifo"],
            }
        )
    return records


def score(lab: Lab, session_result: dict[str, Any], evaluate: Callable[[Lab, str], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Final evaluation of the submission; an unsubmitted run is scored separately.

    `evaluate` replaces the designer's held-out evaluation (the engineer role
    scores the rest of its workload instead)."""
    evaluate = evaluate or (lambda lab_, identity: lab_.final_evaluation(identity))
    submission = session_result.get("submission")
    if submission is not None:
        return {"selected_by": "agent", **evaluate(lab, submission["experiment"])}
    fallback = best_dev_experiment(lab)
    if fallback is None:
        return {"selected_by": None, "valid": False, "score": None}
    report = evaluate(lab, fallback)
    # Not the agent's choice, so it is reported but never counted as the score.
    return {"selected_by": "host-best-dev-fallback", **report, "fallback_score": report["score"], "score": None}


def run_task(
    lab: Lab,
    *,
    directory: Path,
    profile_name: str,
    backend: str,
    model: str,
    effort: str,
    backend_factory: Callable[[TaskRuntime, Path], BackendSession],
    max_experiments: int,
    backend_segments: int | None = None,
    tool_calls: int | None = None,
    transport_provenance: dict[str, Any] | None = None,
    role: str = "designer",
    shots: bool = False,
    split_fraction: float = 0.2,
    evaluate: Callable[[Lab, str], dict[str, Any]] | None = None,
    task_record: dict[str, Any] | None = None,
    library: Any = None,
    train_feedback: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """`library` (experience.Library) gives the agent earlier tuning experience;
    with `train_feedback` the run is a training run: the agent commits a
    decision, sees its outcome on the rest of the workload, and adds a case and
    a revised playbook to the library."""
    root = directory.resolve()
    train = train_feedback is not None
    tag = protocol(role, shots, train=train, experience=library is not None)
    experience_record = None
    if library is not None:
        experience_record = {"cases": len(library.cases()), "playbook_sha256": library.playbook_digest(), "mode": "train" if train else "analyst" if role == "analyst" else "read-only"}
    artifacts = ArtifactStore(root)
    revision = simulator_fingerprint()[:16]
    artifacts.manifest(
        {
            "benchmark": "CacheLab",
            "protocol": tag,
            "benchmark_revision": revision,
            "experience": experience_record,
            "profile": profile_name,
            "backend": backend,
            "model": model,
            "effort": effort,
            "max_experiments": max_experiments,
            "suite": lab.public_suite(),
            "heldout": [spec.manifest() for spec in lab.heldout_specs],
            "sandbox": lab.store.sandbox.provenance(),
            "transport": transport_provenance or {"validation": "caller-supplied-backend"},
            "source_files": source_fingerprint(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "Pillow")},
        }
    )
    session = runtime = backend_session = None
    error = None
    session_result: dict[str, Any] = {}
    started = time.monotonic()
    try:
        session = CacheLabSession(
            lab, visual=profile_name == "vista", max_experiments=max_experiments, role=role, shots=shots, library=library, train_feedback=train_feedback
        )
        profile = make_profile(
            profile=profile_name,
            backend=backend,
            max_experiments=max_experiments,
            initial=session.initial,
            role=role,
            split_fraction=split_fraction,
            train=train,
            experience=library is not None,
        )
        experiment = make_experiment(
            profile,
            backend=backend,
            model=model,
            effort=effort,
            revision=revision,
            suite=lab.public_suite(),
            max_experiments=max_experiments,
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
        elif session is not None:
            session.close()
    if session is not None:
        session_result = session.result()
    if train:
        try:
            session_result["counterfactuals"] = counterfactuals(lab, train_feedback)
        except Exception:
            artifacts.record("private_failure", {"traceback": traceback.format_exc()})
    evaluation: dict[str, Any] = {}
    try:
        # An analyst's product is the library it updated, not a policy to score.
        evaluation = {} if role == "analyst" else score(lab, session_result, evaluate)
    except Exception:
        artifacts.record("private_failure", {"traceback": traceback.format_exc()})
        evaluation = {"error": "evaluation_failure"}
    result = {
        "benchmark": "CacheLab",
        "protocol": tag,
        "profile": profile_name,
        "backend": backend,
        "model": model,
        "effort": effort,
        "max_experiments": max_experiments,
        "role": role,
        "shots": shots,
        "references": list(lab.references),
        "task": task_record or {},
        "experience": experience_record,
        "error": error,
        **session_result,
        "evaluation": evaluation,
        "score": evaluation.get("score"),
        "usage": runtime.usage if runtime is not None else {},
        "backend_usage": backend_session.usage_reports if backend_session is not None else [],
        "elapsed_seconds": time.monotonic() - started,
    }
    path = root / "result.json"
    path.write_text(canonical(result) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return result
