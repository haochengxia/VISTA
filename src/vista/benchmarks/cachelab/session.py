"""One research task as a VISTA TaskSession: lab tools in, evidence out.

Experiments are the environment actions; their overview image becomes the
current observation. View images are historical evidence: VISTA archives each
one with an event number, so `inspect` and `read_pixels` can return to it.
Held-out traces and the final evaluation never pass through this session.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
from collections import Counter
from typing import Any

from PIL import Image

from vista.core.contracts import (
    Effect,
    EvidenceKind,
    Observation,
    ObservationBundle,
    PublicTask,
    SubmissionSpec,
    ToolBinding,
    ToolReply,
    Visual,
)

from .lab import BASELINE, Lab, LabError, View
from .profiles import initial_prompt, lab_specs

VIEW = "chart"
OBJECTIVE = (
    "Improve S3-FIFO's miss ratio on held-out workloads like the development traces, with "
    "bounded metadata, and explain the mechanism. Finish with submit_policy."
)


class CacheLabSession:
    def __init__(
        self,
        lab: Lab,
        *,
        visual: bool,
        max_experiments: int,
        role: str = "designer",
        shots: bool = False,
        library=None,
        train_feedback=None,
    ) -> None:
        self.lab = lab
        self.visual = visual
        self.max_experiments = max_experiments
        self.counter = 0
        self.tool_counts: Counter[str] = Counter()
        self.submission_record: dict[str, Any] | None = None
        self.experiment_log: list[dict[str, Any]] = []
        self.closed = False
        baseline = lab.overview(BASELINE, visual=visual)
        self.role = role
        self.library = library
        self.train_feedback = train_feedback  # experiment id -> engineer report on the rest of the workload
        self.train = train_feedback is not None
        self.decision_record: dict[str, Any] | None = None
        self.feedback_report: dict[str, Any] | None = None
        worked = None
        if shots:
            from .shots import build_shots

            worked = build_shots(lab)
        self.signature = None
        signature_block = None
        if role == "engineer":  # every engineer sees it, with or without experience
            from . import experience

            self.signature = experience.signature(lab)
            signature_block = experience.signature_text(self.signature)
        if role == "analyst":
            from .analyst import workloads_text
            from .profiles import analyst_prompt

            if library is None:
                raise ValueError("an analyst needs a library")
            initial = analyst_prompt(
                workloads_text(lab, library), self._baseline_table(), references=lab.references, library=library.analyst_section()
            )
        else:
            initial = initial_prompt(
                lab.public_suite(),
                self._baseline_table(),
                role=role,
                references=lab.references,
                shots=worked,
                signature=signature_block,
                experience=library.prompt_section() if library is not None else None,
            )
        self.initial = lab.redact(initial)
        observation = self._bundle("baseline-overview", baseline, EvidenceKind.CURRENT)
        self.task = PublicTask(
            "cachelab/s3fifo",
            OBJECTIVE,
            {"suite": lab.public_suite(), "experiments": max_experiments, "views": "image+numbers" if visual else "numbers"},
            observation,
        )
        # A final message without submit_policy is redirected, not accepted.
        self.submission = SubmissionSpec(Effect.SUBMISSION)
        handlers = {
            "run_experiment": self._run_experiment,
            "test_policy": self._test_policy,
            "view_overview": self._view_overview,
            "view_queues": self._view_queues,
            "view_objects": self._view_objects,
            "view_diff": self._view_diff,
            "view_workload": self._view_workload,
            "view_positions": self._view_positions,
            "query_events": self._query_events,
            "experiment_info": self._experiment_info,
            "submit_policy": self._submit_policy,
            "read_case": self._read_case,
            "commit_decision": self._commit_decision,
            "record_experience": self._record_experience,
            "update_knowledge": self._update_knowledge,
        }
        specs = lab_specs(visual, train=self.train, experience=library is not None, handover=role == "engineer", analyst=role == "analyst")
        self.tools = tuple(ToolBinding(spec, self._counted(spec.name, handlers[spec.name])) for spec in specs)

    @property
    def submitted(self) -> bool:
        return self.submission_record is not None

    def _baseline_table(self) -> str:
        rows = ["trace    capacity  miss_ratio"]
        base = self.lab.experiment(BASELINE)
        for (trace, capacity), run in base.cells.items():
            rows.append(f"{trace:<8} {capacity:>8}  {run.summary['miss_ratio']:.5f}")
        return "\n".join(rows)

    def _counted(self, name, handler):
        def call(arguments: Any) -> ToolReply:
            self.tool_counts[name] += 1
            try:
                reply = handler(arguments)
            except LabError as exc:
                reply = ToolReply(str(exc), success=False)
            # Every text the agent receives passes here: no paths or source names.
            redacted = self.lab.redact(reply.text)
            event = None if reply.event_text is None else self.lab.redact(reply.event_text)
            if redacted != reply.text or event != reply.event_text:
                reply = dataclasses.replace(reply, text=redacted, event_text=event)
            return reply

        return call

    def _bundle(self, label: str, view: View, kind: EvidenceKind) -> ObservationBundle:
        if not self.visual or view.png is None:
            return ObservationBundle()
        with Image.open(io.BytesIO(view.png)) as image:
            width, height = image.size
        self.counter += 1
        moment = f"{label}-{self.counter:04d}"
        source = "cachelab-" + hashlib.sha256(view.png).hexdigest()[:24] + f"-{self.counter}"
        return ObservationBundle((Observation(moment, (Visual(VIEW, source, view.png, width, height),), kind),))

    def _reply(self, label: str, view: View, *, current: bool = False, success: bool = True) -> ToolReply:
        kind = EvidenceKind.CURRENT if current else EvidenceKind.HISTORICAL
        return ToolReply(view.text, success=success, observation=self._bundle(label, view, kind))

    # -- handlers ---------------------------------------------------------------------

    def _run_experiment(self, arguments: dict[str, Any]) -> ToolReply:
        handover = arguments["handover_at"] if "handover_at" in arguments else "default"
        view = self.lab.run_experiment(arguments["name"], arguments["code"], arguments["hypothesis"], arguments["prediction"], handover=handover)
        self.experiment_log.append(
            {
                "experiment": view.data.get("experiment"),
                "name": arguments["name"],
                "ok": view.data.get("ok"),
                "expected_change_pct": arguments["prediction"]["expected_change_pct"],
                "observed_change_pct": view.data["mean_relative_change"] * 100 if view.data.get("ok") else None,
                "hypothesis": arguments["hypothesis"],
                "mechanism_signature": arguments["prediction"]["mechanism_signature"],
            }
        )
        used = len(self.experiment_log)
        text = view.text + f"\n\nexperiments used: {used} of {self.max_experiments}"
        return self._reply(f"experiment-{view.data.get('experiment')}", View(text, view.png), current=True, success=bool(view.data.get("ok")))

    def _test_policy(self, arguments: dict[str, Any]) -> ToolReply:
        view = self.lab.test_policy(arguments["code"])
        return ToolReply(view.text, success=bool(view.data.get("ok")))

    def _view_overview(self, arguments: dict[str, Any]) -> ToolReply:
        view = self.lab.overview(arguments["experiment"], arguments.get("compare_to"), visual=self.visual)
        return self._reply("overview", view)

    def _view_queues(self, arguments: dict[str, Any]) -> ToolReply:
        view = self.lab.queues(
            arguments["experiment"], arguments["trace"], arguments["capacity"], arguments.get("start"), arguments.get("end"), arguments.get("buckets"), visual=self.visual
        )
        return self._reply("queues", view)

    def _view_workload(self, arguments: dict[str, Any]) -> ToolReply:
        return self._reply("workload", self.lab.workload(arguments["trace"], visual=self.visual))

    def _view_positions(self, arguments: dict[str, Any]) -> ToolReply:
        view = self.lab.positions(
            arguments["experiment"], arguments["trace"], arguments["capacity"], arguments.get("start"), arguments.get("end"), visual=self.visual
        )
        return self._reply("positions", view)

    def _view_objects(self, arguments: dict[str, Any]) -> ToolReply:
        view = self.lab.objects(
            arguments["experiment"],
            arguments["trace"],
            arguments["capacity"],
            arguments.get("start"),
            arguments.get("end"),
            keys=arguments.get("keys"),
            select=arguments.get("select"),
            count=arguments.get("count", 12),
            compare_to=arguments.get("compare_to"),
            visual=self.visual,
        )
        return self._reply("objects", view)

    def _view_diff(self, arguments: dict[str, Any]) -> ToolReply:
        view = self.lab.diff(
            arguments["experiment"],
            arguments.get("baseline", BASELINE),
            arguments["trace"],
            arguments["capacity"],
            arguments.get("start"),
            arguments.get("end"),
            arguments.get("buckets"),
            visual=self.visual,
        )
        return self._reply("diff", view)

    def _query_events(self, arguments: dict[str, Any]) -> ToolReply:
        view = self.lab.events(
            arguments["experiment"],
            arguments["trace"],
            arguments["capacity"],
            arguments.get("start"),
            arguments.get("end"),
            key=arguments.get("key"),
            kinds=arguments.get("kinds"),
            limit=arguments.get("limit", 100),
        )
        return ToolReply(view.text)

    def _experiment_info(self, arguments: dict[str, Any]) -> ToolReply:
        return ToolReply(self.lab.info(arguments.get("experiment"), bool(arguments.get("include_code"))).text)

    def _decision(self, arguments: dict[str, Any]) -> tuple[Any, str | None]:
        experiment = self.lab.experiment(arguments["experiment"])
        # An engineer may decide to keep S3-FIFO or deploy a reference policy; a designer must change it.
        if self.role != "engineer" and self.lab.is_reference(experiment.id):
            return experiment, "Submit one of your own experiments, not a reference policy."
        if not experiment.ok:
            return experiment, f"Experiment {experiment.id} did not complete; choose a completed experiment."
        return experiment, None

    def _read_case(self, arguments: dict[str, Any]) -> ToolReply:
        try:
            return ToolReply(self.library.case_text(arguments["case"]))
        except KeyError as exc:
            return ToolReply(str(exc.args[0]), success=False)

    def _commit_decision(self, arguments: dict[str, Any]) -> ToolReply:
        if self.decision_record is not None:
            return ToolReply(f"The decision is already committed ({self.decision_record['experiment']}). Record the case with record_experience.", success=False)
        experiment, problem = self._decision(arguments)
        if problem:
            return ToolReply(problem, success=False)
        from . import experience

        self.feedback_report = self.train_feedback(experiment.id)
        self.decision_record = {
            "experiment": experiment.id,
            "mechanism": arguments["mechanism"],
            "evidence": arguments["evidence"],
            "limitations": arguments["limitations"],
        }
        text = experience.feedback_text(self.feedback_report["rows"])
        return ToolReply(
            f"Committed {experiment.id}.\n\n{text}\n\nNow record the case and the revised playbook with record_experience.",
            event_text=f"Committed {experiment.id}.\n\n{text}",
        )

    def _record_experience(self, arguments: dict[str, Any]) -> ToolReply:
        if self.decision_record is None:
            return ToolReply("Commit a decision first (commit_decision).", success=False)
        from . import experience

        experiment = self.lab.experiment(self.decision_record["experiment"])
        record = {
            "signature": self.signature,
            "decision": arguments["decision"],
            "reasoning": self.decision_record["mechanism"],
            "evidence": self.decision_record["evidence"],
            "limitations": self.decision_record["limitations"],
            "feedback": experience.feedback_text(self.feedback_report["rows"]),
            "outcome": experience.outcome_summary(self.feedback_report),
            "lesson": arguments["lesson"],
            "experiments": self.experiment_log,
        }
        case_id = self.library.add(record, experiment.source, arguments["playbook"])
        self.submission_record = {**self.decision_record, "case": case_id, "lesson": arguments["lesson"]}
        event = f"Recorded {case_id} and the revised playbook. The task is complete."
        return ToolReply(event, finished=True, event_text=event)

    def _update_knowledge(self, arguments: dict[str, Any]) -> ToolReply:
        known = {case["case"] for case in self.library.cases()}
        unknown = sorted({item["case"] for item in arguments["postmortems"]} - known)
        if unknown:
            return ToolReply(f"unknown cases {unknown}; known: {', '.join(sorted(known))}", success=False)
        tag = f"analyst-{len(list((self.library.root / 'knowledge-history').glob('*.md'))) + 1}"
        for item in arguments["postmortems"]:
            self.library.add_postmortem(item["case"], item["text"], tag)
        self.library.set_knowledge(arguments["knowledge"], tag)
        self.library.set_playbook(arguments["playbook"], tag)
        self.library.clear_proposals()
        self.submission_record = {
            "analysis": tag,
            "postmortems": [item["case"] for item in arguments["postmortems"]],
            "knowledge_chars": len(arguments["knowledge"]),
            "playbook_chars": len(arguments["playbook"]),
        }
        event = f"Recorded {len(arguments['postmortems'])} postmortems, the knowledge and the playbook ({tag}). The task is complete."
        return ToolReply(event, finished=True, event_text=event)

    def _submit_policy(self, arguments: dict[str, Any]) -> ToolReply:
        experiment, problem = self._decision(arguments)
        if problem:
            return ToolReply(problem, success=False)
        self.submission_record = {
            "experiment": experiment.id,
            "mechanism": arguments["mechanism"],
            "evidence": arguments["evidence"],
            "limitations": arguments["limitations"],
        }
        event = f"Submitted {experiment.id}. The task is complete."
        return ToolReply(event, finished=True, event_text=event)

    def submit(self, answer: str) -> ToolReply:
        finish = "commit_decision, then record_experience" if self.train else "update_knowledge" if self.role == "analyst" else "submit_policy"
        return ToolReply(f"Nothing was submitted. Continue with the lab tools, and finish with {finish}.", success=False)

    def result(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "signature": self.signature,
            "training_feedback": self.feedback_report,
            "submission": self.submission_record,
            "experiments": self.experiment_log,
            "tool_counts": dict(self.tool_counts),
        }

    def close(self) -> None:
        self.closed = True
