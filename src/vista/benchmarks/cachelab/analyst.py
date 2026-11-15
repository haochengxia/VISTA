"""The analyst role: study training workloads in hindsight and refine the library.

An engineer decides from the first part of a workload and learns its outcome.
An analyst sees more: each training case's whole workload, as a trace named
after the case, with the deployment replayed exactly (S3-FIFO serves the
observed part, the policy takes over at the split; only requests after the
split were scored). Every recorded decision is an experiment (`dec-NNN`) that
can be viewed and compared on every case's workload. The analyst explains why
decisions succeeded or failed, tests candidate rules across all cases, and
writes postmortems, knowledge and a revised playbook into the library.

Only training workloads ever reach an analyst; held-out workloads never do.
Workloads are known by case id: the host maps cases to their sources through
the library's ORIGINS.json, which no tool shows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

from .engine import Sandbox
from .experience import Library
from .lab import Lab
from .traces import TraceSpec, materialize

REFERENCES = ("lru", "mq")


def decision_id(case_id: str) -> str:
    return "dec-" + case_id.split("-")[1]


def make_lab(library: Library, root: Path, sandbox: Sandbox, *, workers: int = 1, references: Sequence[str] = REFERENCES, timeout: float = 900) -> Lab:
    origins = library.origins()
    known = {case["case"] for case in library.cases()}
    cases = sorted(case for case in origins if case in known)
    if not cases:
        raise ValueError(f"{library.root} has no cases with a recorded origin")
    fractions = {origins[case]["split_fraction"] for case in cases}
    if len(fractions) != 1:
        raise ValueError("cases with different split fractions cannot share one analyst lab")
    fraction = fractions.pop()
    traces, sizes, secrets = [], {}, []
    for case in cases:
        origin = origins[case]
        row = origin["target"]
        spec = TraceSpec(row["name"], row["split"], row["family"], row["params"], row["requests"], row["seed"], row["source"])
        trace = materialize(dataclasses.replace(spec, name=case, split="dev"), root.resolve() / "traces" / "cases")
        if int(trace.requests * fraction) != origin["split"]:
            raise RuntimeError(f"{case}: the replayed workload does not match the recorded split")
        traces.append(trace)
        sizes[case] = list(origin["capacities"])
        secrets.append(spec.name)
        secrets.extend(str(value) for value in spec.params.values() if isinstance(value, str))
        if spec.source:
            source = Path(spec.source)
            secrets += [str(source), source.name, source.name.split(".")[0], str(source.parent)]
    lab = Lab(
        root,
        sandbox,
        dev=(),
        heldout=(),
        dev_traces=tuple(traces),
        sizes=sizes,
        references=references,
        workers=workers,
        timeout=timeout,
        redactions=tuple(secret for secret in secrets if len(secret) >= 5),
        handover=fraction,
    )
    for case in cases:
        record, code = library.case(case)
        if code.strip():
            lab.record(decision_id(case), f"decision of {case}", code, record.get("decision", ""))
    return lab


def workloads_text(lab: Lab, library: Library) -> str:
    """The case workloads as the analyst sees them."""
    origins = library.origins()
    lines = ["Training workloads (each named after its case; the engineer saw requests [0, split) and was scored on the rest):"]
    for name, trace in lab.dev.items():
        origin = origins[name]
        lines.append(
            f"- {name}: {trace.requests:,} requests, {trace.objects:,} objects; split at request {origin['split']:,}; "
            f"cache sizes {lab.capacities[name]}; decision experiment `{decision_id(name)}`"
        )
    return "\n".join(lines)
