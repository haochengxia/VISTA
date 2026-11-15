"""Prompts, tool contracts and session policy for the cache-policy research task.

Two profiles share every tool and every number:

- `vista`: views return an image plus the exact numbers; the VISTA tools
  (inspect, read_pixels, history, GUIDE/WORKING) can revisit any earlier view.
- `numeric`: views return only the numbers, and there is nothing to inspect.
  This is the "code + numeric feedback" control arm.

A third arm, parameter search, needs no agent (see search.py).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from vista.core.contracts import Capability, Effect, ToolSpec
from vista.core.profiles import (
    AgentProfile,
    CompactionMode,
    ContextMode,
    Experiment,
    InteractionMode,
    PromptBundle,
    SessionPolicy,
)

from .analysis import SELECTIONS
from .engine import SANDBOX_DIR
from .lab import MEMORY_LIMIT_RATIO

PROTOCOL = "cachelab-s3fifo-improvement-v2"  # v2: workload/position views, references, roles


def protocol(role: str, shots: bool, *, train: bool = False, experience: bool = False) -> str:
    tags = [tag for tag, on in (("shots", shots), ("experience", experience), ("train", train)) if on]
    return "/".join([PROTOCOL, role, *tags])


EVALUATOR = "verified-simulation heldout mean relative miss-ratio change vs S3-FIFO; memory <= 2x S3-FIFO"
PROFILES = ("vista", "numeric")
DEFAULT_EXPERIMENTS = 12

VISUAL_TOOLS = ("inspect", "read_pixels")
MEMORY_TOOLS = ("history", "read_guide", "write_guide", "read_working", "write_working", "save_compact_checkpoint")

_WINDOW = {
    "start": {"type": "integer", "minimum": 0, "description": "First request of the window (default: trace start)."},
    "end": {"type": "integer", "minimum": 1, "description": "End of the window, exclusive (default: trace end)."},
}
_CELL = {
    "experiment": {"type": "string", "minLength": 1, "maxLength": 32, "description": "Experiment id, e.g. s3fifo or e3."},
    "trace": {"type": "string", "minLength": 1, "maxLength": 32},
    "capacity": {"type": "integer", "minimum": 1, "description": "One of the trace's simulated cache sizes."},
}
_BUCKETS = {"buckets": {"type": "integer", "minimum": 8, "maximum": 128, "description": "Time windows (default 64)."}}


def _schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": required, "properties": properties}


def _view_spec(name: str, text: str, properties: dict[str, Any], required: list[str], visual: bool) -> ToolSpec:
    returns = (
        " Returns one image and the exact numbers drawn in it."
        if visual
        else " Returns the exact numbers as text."
    )
    return ToolSpec(
        name,
        text + returns + " Does not use the experiment budget.",
        _schema(required, properties),
        Capability.INSPECTION,
        Effect.NONE,
        max_images=1 if visual else 0,
    )


_SUBMIT_PROPERTIES = {
    "experiment": _CELL["experiment"],
    "mechanism": {"type": "string", "minLength": 1, "maxLength": 4000, "description": "What the change does and why it reduces misses."},
    "evidence": {"type": "string", "minLength": 1, "maxLength": 4000, "description": "The observations that support the mechanism, with experiment ids and view event numbers."},
    "limitations": {"type": "string", "minLength": 1, "maxLength": 3000, "description": "Where the change does not help or hurts, and what remains unexplained."},
}


def experience_specs(*, train: bool, experience: bool) -> tuple[ToolSpec, ...]:
    """Extra tools for experience learning. In training, commit_decision replaces
    submit_policy: it fixes the decision and returns how it did on the rest of
    the workload; record_experience then ends the task."""
    specs = []
    if experience:
        specs.append(
            ToolSpec(
                "read_case",
                "Show one earlier case in full: its workload signature, decision, code, outcome on the rest of its workload, and lesson.",
                _schema(["case"], {"case": {"type": "string", "pattern": "^case-[0-9]{3}$"}}),
                Capability.HISTORY,
                Effect.NONE,
            )
        )
    if train:
        specs.append(
            ToolSpec(
                "commit_decision",
                (
                    "Commit one experiment as your decision for this workload. It is final: it then runs over "
                    "the whole workload, and you get its miss ratio on the remaining requests next to S3-FIFO, "
                    "S3-FIFO with the knobs that did best on the observed part, published algorithms and the "
                    "offline optimum. Then record the case with record_experience."
                ),
                _schema(["experiment", "mechanism", "evidence", "limitations"], _SUBMIT_PROPERTIES),
                Capability.ACTION,
                Effect.ENVIRONMENT,
            )
        )
        specs.append(
            ToolSpec(
                "record_experience",
                (
                    "Record this case for later engineers and revise the playbook, then end the task. Later "
                    "sessions read the playbook and the case table first, and the full case on request."
                ),
                _schema(
                    ["decision", "lesson", "playbook"],
                    {
                        "decision": {"type": "string", "minLength": 1, "maxLength": 300, "description": "One line: what was deployed and with which settings."},
                        "lesson": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 3000,
                            "description": "What the outcome teaches: which prefix signals predicted it, what you would decide now, and for which kinds of workloads.",
                        },
                        "playbook": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 12000,
                            "description": "The complete revised playbook (it replaces the current one): conditions -> decisions, with the cases that support each rule and how strong the evidence is.",
                        },
                    },
                ),
                Capability.SUBMISSION,
                Effect.SUBMISSION,
            )
        )
    return tuple(specs)


HANDOVER_AT = {
    "type": "number",
    "minimum": 0,
    "maximum": 0.9,
    "description": (
        "Share of each trace S3-FIFO serves before this policy takes over its live cache (adopt). "
        "Default 0.2, like the deployment; 0 runs the policy from a cold cache."
    ),
}


UPDATE_KNOWLEDGE = ToolSpec(
    "update_knowledge",
    (
        "Write the results of the analysis into the library and end the task: a postmortem for each case "
        "you studied, the revised knowledge, and the complete revised playbook (both replace the current ones)."
    ),
    _schema(
        ["postmortems", "knowledge", "playbook"],
        {
            "postmortems": {
                "type": "array",
                "maxItems": 64,
                "items": _schema(
                    ["case", "text"],
                    {
                        "case": {"type": "string", "pattern": "^case-[0-9]{3}$"},
                        "text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2500,
                            "description": "Why the decision did as it did: informative and misleading signals, decision vs handover vs change in the workload, the mechanism that carried the effect, with evidence.",
                        },
                    },
                ),
            },
            "knowledge": {
                "type": "string",
                "minLength": 1,
                "maxLength": 16000,
                "description": "Mechanisms and regularities across cases, each with its evidence (cases, experiment ids, numbers) and its strength.",
            },
            "playbook": {
                "type": "string",
                "minLength": 1,
                "maxLength": 12000,
                "description": "The complete revised playbook for engineers who see only the first 20% of a new workload: rules conditioned on what is observable there.",
            },
        },
    ),
    Capability.SUBMISSION,
    Effect.SUBMISSION,
)


def lab_specs(
    visual: bool, *, train: bool = False, experience: bool = False, handover: bool = False, analyst: bool = False
) -> tuple[ToolSpec, ...]:
    core = _lab_specs(visual)
    if handover or analyst:
        core = tuple(_with_handover(spec) if spec.name == "run_experiment" else spec for spec in core)
    if analyst:
        core = tuple(spec for spec in core if spec.name != "submit_policy")
        return core + experience_specs(train=False, experience=True) + (UPDATE_KNOWLEDGE,)
    if train:
        core = tuple(spec for spec in core if spec.name != "submit_policy")
    return core + experience_specs(train=train, experience=experience)


def _with_handover(spec: ToolSpec) -> ToolSpec:
    schema = dict(spec.input_schema)
    schema["properties"] = {**schema["properties"], "handover_at": HANDOVER_AT}
    return dataclasses.replace(spec, input_schema=schema)


def _lab_specs(visual: bool) -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            "run_experiment",
            (
                "Simulate a complete policy module on every development trace and cache size, and "
                "compare it with S3-FIFO. Costs one experiment. The module must define CachePolicy "
                "(see the policy API); it may `from s3fifo import S3FIFO` and override parts. Before "
                "running, state the mechanism you are testing and predict both the aggregate change "
                "and the signature you expect to see in the views. The result gives the per-cell "
                "table, a check of your aggregate prediction"
                + (", and the change-over-time overview image." if visual else ", and the change-over-time overview numbers.")
            ),
            _schema(
                ["name", "hypothesis", "prediction", "code"],
                {
                    "name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "hypothesis": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 3000,
                        "description": "The mechanism you believe limits the current policy, the evidence for it, and why this change addresses it.",
                    },
                    "prediction": _schema(
                        ["expected_change_pct", "mechanism_signature"],
                        {
                            "expected_change_pct": {
                                "type": "number",
                                "minimum": -100,
                                "maximum": 100,
                                "description": "Predicted mean relative miss-ratio change vs s3fifo in percent (negative = fewer misses).",
                            },
                            "mechanism_signature": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 1500,
                                "description": "What should change in the views if the hypothesis is right, and where a regression could appear.",
                            },
                        },
                    ),
                    "code": {"type": "string", "minLength": 1, "maxLength": 65536},
                },
            ),
            Capability.ACTION,
            Effect.ENVIRONMENT,
            max_images=1 if visual else 0,
            costs={"experiments": 1},
        ),
        ToolSpec(
            "test_policy",
            (
                "Check a policy module without recording an experiment: static checks, then one "
                "verified run on a single trace and cache size, reporting errors with a traceback, "
                "instrumentation problems, and the policy's memory relative to s3fifo. It does not "
                "report miss ratios (only experiments do) and does not use the experiment budget."
            ),
            _schema(["code"], {"code": {"type": "string", "minLength": 1, "maxLength": 65536}}),
            Capability.INSPECTION,
            Effect.NONE,
        ),
        _view_spec(
            "view_overview",
            (
                "Miss ratio of an experiment for every development trace, cache size and time window; "
                "with compare_to, the change against that experiment instead."
            ),
            {
                "experiment": _CELL["experiment"],
                "compare_to": {"type": "string", "minLength": 1, "maxLength": 32},
            },
            ["experiment"],
            visual,
        ),
        _view_spec(
            "view_queues",
            (
                "Replay one experiment's queues on one trace and cache size over a window: occupancy, "
                "misses by cause (cold, or returning after eviction from a queue), evictions (and how "
                "many had no hit while resident), transfers and ghost hits, reinsertions; plus "
                "eviction quality (how soon evicted objects are requested again, in cache sizes)."
            ),
            {**_CELL, **_WINDOW, **_BUCKETS},
            ["experiment", "trace", "capacity"],
            visual,
        ),
        _view_spec(
            "view_objects",
            (
                "Timelines of individual objects: each request (hit or miss), residency by queue, "
                "ghost membership and evictions. Give keys, or select the objects carrying a "
                "window's misses (most_remisses, most_requests, evicted_unused_from_main) or, with "
                "compare_to, the objects whose outcome changed (most_regressed, most_improved)."
            ),
            {
                **_CELL,
                **_WINDOW,
                "keys": {"type": "array", "minItems": 1, "maxItems": 24, "items": {"type": "integer", "minimum": 0}},
                "select": {"type": "string", "enum": list(SELECTIONS)},
                "count": {"type": "integer", "minimum": 1, "maximum": 24},
                "compare_to": {"type": "string", "minLength": 1, "maxLength": 32},
            },
            ["experiment", "trace", "capacity"],
            visual,
        ),
        _view_spec(
            "view_diff",
            (
                "Compare two experiments request by request on one trace and cache size: miss ratio "
                "per window, requests only one of them hits, cumulative hits gained, and the objects "
                "with the most changed outcomes."
            ),
            {**_CELL, "baseline": {"type": "string", "minLength": 1, "maxLength": 32, "description": "Default s3fifo."}, **_WINDOW, **_BUCKETS},
            ["experiment", "trace", "capacity"],
            visual,
        ),
        _view_spec(
            "view_workload",
            (
                "Characterize one trace independently of any policy, with the analyses behind MQ and "
                "S3-FIFO: reuse distance (distinct objects between requests to a key) against the "
                "cache sizes, the access-frequency distribution, and the one-hit-wonder share of "
                "windows holding as many distinct objects as the cache (or 0.1x it)."
            ),
            {"trace": _CELL["trace"]},
            ["trace"],
            visual,
        ),
        _view_spec(
            "view_positions",
            (
                "Where an experiment's hits land in each queue and ghost, the statistic S4-FIFO uses "
                "to set its knobs: for each hit, how many objects entered the same queue after this "
                "one did, in multiples of the queue's mean size (in a FIFO, 0 is the head and about 1 "
                "the tail). Hits near 0 are quick re-requests; mass near 1 means the queue is barely "
                "long enough."
            ),
            {**_CELL, **_WINDOW},
            ["experiment", "trace", "capacity"],
            visual,
        ),
        ToolSpec(
            "query_events",
            (
                "Raw records of one experiment on one trace and cache size, in time order: requests "
                "(outcome, queue, ghost hit), evictions (source queue, hits while resident, next "
                "request time), moves and ghost events. Filter by window, key and kind; at most 200 "
                "rows. Does not use the experiment budget."
            ),
            _schema(
                ["experiment", "trace", "capacity"],
                {
                    **_CELL,
                    **_WINDOW,
                    "key": {"type": "integer", "minimum": 0},
                    "kinds": {"type": "array", "minItems": 1, "items": {"type": "string", "enum": ["request", "eviction", "move", "ghost"]}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
            ),
            Capability.INSPECTION,
            Effect.NONE,
        ),
        ToolSpec(
            "experiment_info",
            "List experiments, or show one experiment's hypothesis, prediction, per-cell results and optionally its code.",
            _schema(
                [],
                {
                    "experiment": {"type": "string", "minLength": 1, "maxLength": 32},
                    "include_code": {"type": "boolean"},
                },
            ),
            Capability.HISTORY,
            Effect.NONE,
        ),
        ToolSpec(
            "submit_policy",
            (
                "Submit one completed experiment as your final policy and end the task. It is then "
                "re-run in verified simulation on held-out workloads."
            ),
            _schema(["experiment", "mechanism", "evidence", "limitations"], _SUBMIT_PROPERTIES),
            Capability.SUBMISSION,
            Effect.SUBMISSION,
        ),
    )


def policy_api_text() -> str:
    return (SANDBOX_DIR / "policy_api.py").read_text(encoding="utf-8")


def baseline_text() -> str:
    return (SANDBOX_DIR / "s3fifo.py").read_text(encoding="utf-8")


ROLES = ("designer", "engineer", "analyst")


def _experiments(count: int) -> str:
    return f"{count} experiment" + ("" if count == 1 else "s")


def _engineer_objective(max_experiments: int, split_fraction: float) -> str:
    seen = f"{split_fraction:.0%}"
    rest = f"{1 - split_fraction:.0%}"
    return (
        "You are the cache engineer for one production workload. Decide the eviction policy to "
        "deploy for it, and explain the decision.\n\n"
        f"What you see. Only the first {seen} of the workload's requests, as the trace `target`, "
        "simulated at the cache sizes it will run with. You know nothing else about where it comes "
        "from.\n\n"
        "Deployment. S3-FIFO is serving this workload; it served every request you see. Your decision "
        "takes over S3-FIFO's live cache at the end of what you see: the simulator calls your policy's "
        "adopt() with S3-FIFO's state at that moment (resident objects by queue and order, their "
        "frequency counters, the ghost queue; see Handover in the policy API), and the policy must take "
        "over every resident object. Before the switch it may observe() the requests S3-FIFO served. "
        "The inherited state is part of the problem: it persists after the switch, the longer the "
        "larger the cache, and every miss after the switch counts. Your experiments simulate the same "
        "handover: by default S3-FIFO serves the first 20% of `target` and your policy takes over there "
        "(handover_at moves the point; 0 is a cold start), so before the switch every experiment "
        "matches s3fifo exactly.\n\n"
        f"Scoring. In a verified simulator, only the remaining {rest} of requests are scored: the miss "
        "ratio at each cache size, relative to S3-FIFO continuing unchanged (the mean over cache "
        "sizes; negative is better), and "
        "compared with published algorithms. You may deploy S3-FIFO unchanged (experiment `s3fifo`), "
        "a reference policy, or a policy of your own: whichever you expect to do best on the rest. "
        "The policy may depend on the cache size (passed to "
        "__init__) and may adapt as it runs, but it never sees future requests. The workload "
        "continues past what you see and may drift, so a decision that fits accidents of the "
        "observed prefix is the main risk: check whether a gain holds in the later windows of what "
        "you see (view_overview with compare_to, view_diff with start/end), not only on average. "
        "A policy must be deterministic, and its metadata must "
        "stay bounded by the cache size: its memory is measured, and more than "
        f"{MEMORY_LIMIT_RATIO:g}x S3-FIFO's at the same cache size invalidates the submission. "
        "Capacity counts objects.\n\n"
        f"Budget. {_experiments(max_experiments)} (run_experiment). Views, test_policy, queries and "
        "notes do not use experiments."
    )


def _analyst_objective(max_experiments: int, split_fraction: float) -> str:
    seen = f"{split_fraction:.0%}"
    return (
        "You are the analyst of a cache-engineering team. Engineers tuned S3-FIFO deployments on training "
        f"workloads: each saw only the first {seen} of its workload, decided (keep S3-FIFO, deploy a "
        "reference policy, or deploy its own policy, which took over S3-FIFO's live cache at the split "
        "through adopt()), and was scored on the rest. Their cases and the team's playbook are below. "
        f"Future engineers will decide on new workloads from the first {seen} alone, with one look and at "
        "most a few experiments, guided by the playbook.\n\n"
        "Your job: study the training workloads in hindsight and turn their outcomes into knowledge that "
        "transfers. You see each case's whole workload, as a trace named after the case; requests before "
        "its split are what its engineer saw. Experiments replay the deployment exactly: S3-FIFO serves "
        "the observed part, the policy takes over at the split, and every comparison reports the change "
        "after the handover, which is what was scored (handover_at=0 runs a policy from a cold cache, to "
        "separate the decision from the handover). Each engineer's decision is recorded as experiment "
        "dec-NNN, free to view, diff and compare on every workload; read_case shows a case in full, with "
        "every candidate its engineer tried scored in hindsight.\n\n"
        "For each case, explain why the decision did as it did: which signals in the observed part were "
        "informative and which misled; whether the outcome came from the decision itself, from the "
        "handover, or from the workload changing after the split; which knob or mechanism carried the "
        "effect (isolate it with an experiment when that is the question). Then look across cases for "
        "regularities: which observable signals predict which decision works, and which decisions are "
        "risky. Test a candidate rule on all workloads before writing it down, and state how much evidence "
        "supports it.\n\n"
        "Finish with update_knowledge: a postmortem per case you studied; the knowledge (mechanisms and "
        "regularities, each with its evidence: cases, experiment ids, numbers); and the complete revised "
        f"playbook. The playbook is for engineers who see only the first {seen} of a new workload: "
        "condition every rule on what they can observe there (the signature and views of the observed "
        "part), never on hindsight, and keep it short enough to apply in one look.\n\n"
        f"Budget. {_experiments(max_experiments)} (run_experiment). Views, queries and notes do not use experiments."
    )


def system_prompt(*, visual: bool, max_experiments: int, role: str = "designer", split_fraction: float = 0.2, train: bool = False) -> str:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    objective = (
        "You are a cache-algorithm researcher. Improve S3-FIFO, a cache eviction policy, on a "
        "suite of workloads, and explain why your change works.\n\n"
        "Scoring. Your submitted policy is re-run in a verified simulator on held-out workloads you "
        "never see, at four cache sizes each. The score is the mean relative miss-ratio change "
        "against S3-FIFO over those cells; negative is better. The held-out workloads resemble the "
        "development traces without repeating them, so a change that fits one development trace, "
        "or wins some cells by losing others, is unlikely to score well. A policy must be "
        "deterministic, and its metadata must stay bounded by the cache size: its memory is "
        f"measured, and more than {MEMORY_LIMIT_RATIO:g}x S3-FIFO's at the same cache size "
        "invalidates the submission. Capacity counts objects.\n\n"
        f"Budget. {_experiments(max_experiments)} (run_experiment). Views, test_policy, queries and "
        "notes do not use experiments."
    )
    if role == "engineer":
        objective = _engineer_objective(max_experiments, split_fraction)
    if role == "analyst":
        objective = _analyst_objective(max_experiments, split_fraction)
    if visual:
        observe = (
            "- Start from evidence, not from a list of ideas. Find where S3-FIFO loses: which "
            "traces, cache sizes and time windows (view_overview); what its queues do there "
            "(view_queues); which objects carry the misses and what happened to each of them "
            "(view_objects); and the exact records when a detail matters (query_events). Every "
            "view also returns the numbers it draws.\n"
            "- Characterize the workload itself (view_workload: reuse distance against the cache sizes, "
            "access frequency, one-hit wonders) and see where a policy's hits land in each queue "
            "(view_positions).\n"
            "- Move between scales on purpose: zoom into a window or object to explain a pattern, "
            "then return to the overview to see whether the explanation covers the rest. A striking "
            "pattern is not necessarily the one that costs the most misses.\n"
            "- Every image is kept. Use `inspect` to revisit an earlier view or enlarge a region, "
            "`read_pixels` to measure exact colors, and `history` to find earlier results."
        )
    else:
        observe = (
            "- Start from evidence, not from a list of ideas. Find where S3-FIFO loses: which "
            "traces, cache sizes and time windows (view_overview); what its queues do there "
            "(view_queues); which objects carry the misses and what happened to each of them "
            "(view_objects); and the exact records when a detail matters (query_events). Views "
            "return their numbers as text.\n"
            "- Characterize the workload itself (view_workload: reuse distance against the cache sizes, "
            "access frequency, one-hit wonders) and see where a policy's hits land in each queue "
            "(view_positions).\n"
            "- Move between scales on purpose: zoom into a window or object to explain a pattern, "
            "then return to the overview to see whether the explanation covers the rest. A "
            "striking pattern is not necessarily the one that costs the most misses.\n"
            "- Use `history` to find earlier results."
        )
    method = (
        "# How to work\n\n"
        + observe
        + "\n- Before each experiment, state the mechanism you are testing and predict two things: the "
        "aggregate change, and the signature you expect in the views (what should change, and where "
        "a regression could appear). Afterwards check both, with view_diff and the same views in the "
        "same places, not only the aggregate. If the aggregate moved for a different reason than you "
        "predicted, the change is not yet understood.\n"
        "- Use test_policy to catch errors and memory problems before spending an experiment.\n"
        "- Keep GUIDE.md as a concise, revisable model of how the policy behaves on these workloads "
        "and which hypotheses the evidence supported or refuted. Use WORKING.md for the current "
        "plan and open questions.\n"
        "- Submit with submit_policy when further experiments are more likely to overfit than to "
        "teach you something: name the experiment, the mechanism, the evidence (experiment ids and "
        "view event numbers), and where the change does not help."
    )
    if role == "analyst":
        method = method.split("\n- Submit with submit_policy")[0] + (
            "\n- Finish with update_knowledge when further experiments are unlikely to change the playbook."
        )
    if train:
        method = method.replace("- Submit with submit_policy when", "- Commit your decision with commit_decision when")
        method += (
            "\n\n# Training\n\n"
            "This workload is for training: there is no score to protect, only experience to gain. Decide "
            "exactly as you would for a real deployment, then commit_decision instead of submit_policy. "
            "You will see how the decision did on the rest of the workload. Compare it with what the "
            "observed part suggested, find which signals were predictive and which misled you, and "
            "record_experience: the case, its lesson, and a revised playbook. You have a large budget "
            "here; the engineers who will use the playbook will not. They decide on a new workload from one "
            "look at its first part (the signature and free views) and at most a few experiments, with no "
            "feedback. So the playbook must turn what is observable for free into a decision directly: "
            "which signals to read, which policy to deploy and with which settings (cases' code can be "
            "reused through read_case), and which single experiment to spend, if any, to settle the "
            "remaining doubt. Keep it short, conditional (when X is observed, do Y), and honest about how "
            "much evidence supports each rule."
        )
    return objective + "\n\n" + method


def initial_prompt(
    public_suite: list[dict[str, Any]],
    baseline_table: str,
    *,
    role: str = "designer",
    references: tuple[str, ...] = (),
    shots: str | None = None,
    signature: str | None = None,
    experience: str | None = None,
) -> str:
    heading = (
        "The workload so far (capacity values are the cache sizes it will run with, in objects):"
        if role == "engineer"
        else "Development traces (capacity values are the simulated cache sizes, in objects):"
    )
    lines = [heading]
    for trace in public_suite:
        lines.append(
            f"- {trace['trace']}: {trace['requests']:,} requests, {trace['unique_objects']:,} objects, "
            f"{trace['one_hit_object_share']:.1%} of objects requested once; capacities {trace['capacities']}"
        )
    return "\n".join(
        [
            *lines,
            "",
            "S3-FIFO baseline (experiment id `s3fifo`):",
            baseline_table,
            "",
            "Policy API (policy_api.py):",
            "```python",
            policy_api_text().rstrip(),
            "```",
            "",
            "Baseline policy (s3fifo.py; importable as `from s3fifo import S3FIFO`):",
            "```python",
            baseline_text().rstrip(),
            "```",
            "",
            "Allowed imports: policy_api, s3fifo, s4fifo, mq, lru, fifo, collections, heapq, bisect, "
            "math, random (seed it), itertools, functools, dataclasses, typing, array, statistics.",
            *(
                [
                    "",
                    "Reference policies, already simulated as experiments with these ids (free to view and "
                    "diff; experiment_info with include_code shows their source): "
                    + ", ".join(f"`{name}`" for name in references)
                    + ".",
                ]
                if references
                else []
            ),
            *(["", shots] if shots else []),
            *(["", "# Signature of this workload (host-computed statistics of what you see)", "", signature] if signature else []),
            *(["", experience] if experience else []),
        ]
    )


def analyst_prompt(workloads: str, baseline_table: str, *, references: tuple[str, ...], library: str) -> str:
    return "\n".join(
        [
            workloads,
            "",
            "S3-FIFO baseline (experiment id `s3fifo`), miss ratio over each whole workload:",
            baseline_table,
            "",
            "Policy API (policy_api.py):",
            "```python",
            policy_api_text().rstrip(),
            "```",
            "",
            "Baseline policy (s3fifo.py; importable as `from s3fifo import S3FIFO`):",
            "```python",
            baseline_text().rstrip(),
            "```",
            "",
            "Allowed imports: policy_api, s3fifo, s4fifo, mq, lru, fifo, collections, heapq, bisect, "
            "math, random (seed it), itertools, functools, dataclasses, typing, array, statistics.",
            "",
            "Recorded experiments (free to view, diff and read): the decisions `dec-NNN`, and the reference "
            "policies " + ", ".join(f"`{name}`" for name in references) + ".",
            "",
            library,
        ]
    )


def make_profile(
    *,
    profile: str,
    backend: str,
    max_experiments: int,
    initial: str,
    role: str = "designer",
    split_fraction: float = 0.2,
    train: bool = False,
    experience: bool = False,
) -> AgentProfile:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if backend not in {"codex", "claude"}:
        raise ValueError("Unknown backend")
    visual = profile == "vista"
    lab = tuple(
        spec.name for spec in lab_specs(visual, train=train, experience=experience, handover=role == "engineer", analyst=role == "analyst")
    )
    support = tuple(
        name
        for name in ((*VISUAL_TOOLS, *MEMORY_TOOLS) if visual else MEMORY_TOOLS)
        if backend == "codex" or name != "save_compact_checkpoint"
    )
    return AgentProfile(
        f"cachelab-s3fifo/{profile}",
        (*lab, *support),
        PromptBundle(
            system_prompt(visual=visual, max_experiments=max_experiments, role=role, split_fraction=split_fraction, train=train),
            initial,
            step=(
                "Continue. Finish with commit_decision, then record_experience."
                if train
                else "Continue the analysis. Finish with update_knowledge."
                if role == "analyst"
                else "Continue the research. Finish with submit_policy."
            ),
            checkpoint="Save the compact checkpoint before continuing.",
            recovery=(
                "Continue this same research task from GUIDE, WORKING and the supplied record. "
                "Experiments already run remain available by id; use experiment_info to list them."
            ),
        ),
        SessionPolicy(
            context=ContextMode.CONTINUOUS,
            interaction=InteractionMode.TOOLS,
            compaction=CompactionMode.CHECKPOINT,
        ),
    )


def make_experiment(
    profile: AgentProfile,
    *,
    backend: str,
    model: str,
    effort: str,
    revision: str,
    suite: list[dict[str, Any]],
    max_experiments: int,
    backend_segments: int | None = None,
    tool_calls: int | None = None,
    final_answers: int = 12,
) -> Experiment:
    return Experiment(
        "CacheLab",
        revision,
        PROTOCOL,
        profile,
        backend,
        model,
        effort,
        {
            "views": "image+numbers" if profile.profile_id.endswith("/vista") else "numbers",
            "suite": suite,
            "baseline": "S3-FIFO, libCacheSim-equivalent",
            "capacity_unit": "objects",
        },
        {
            "experiments": max_experiments,
            "final_answers": final_answers,
            **({"backend_segments": backend_segments} if backend_segments is not None else {}),
            **({"tool_calls": tool_calls} if tool_calls is not None else {}),
        },
        EVALUATOR,
    )
