"""Tuning experience accumulated across engineer tasks.

Training: an agent does the engineer job on a training workload, commits a
decision, then sees how it did on the rest of that workload (against
S3-FIFO, prefix-tuned S3-FIFO and published algorithms). It records the case
and revises a playbook. Testing: an agent gets the playbook and the casebook,
read only, and does the engineer job on unseen workloads without feedback.

A library is a directory:

    playbook.md            the current playbook (latest revision)
    playbook-history/      every revision, numbered
    knowledge.md           mechanisms found by hindsight analysis, with evidence
    knowledge-history/     every revision
    proposals/             playbook revisions from parallel training sessions,
                           waiting for an analyst to reconcile them
    cases/case-NNN.json    signature, decision, outcome, lesson, postmortems and
                           counterfactuals of one case
    cases/case-NNN.py      the decision's policy code
    ORIGINS.json           host only: which workload each case came from (never
                           shown to an agent; the analyst's lab is built from it)

Refinement loop. Training sessions add cases; an analyst session then studies
the training workloads in hindsight (their whole trace), writes a postmortem
per case, and revises the knowledge and the playbook; the next round of
training starts from the refined library.

Cases never name their workload: they carry a host-computed signature (the
same statistics the analysis views report), so a new workload can be matched
against them on equal terms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import analysis
from .lab import BASELINE, MEMORY_LIMIT_RATIO, Lab

NAME = "target"
MERGED = "merged"
# A decision is credited to the rules it cited when it beat S3-FIFO and was close to
# the best candidate of its session (in hindsight); it counts against them when it
# lost to S3-FIFO or left much on the table.
HELPFUL_GAIN, HELPFUL_REGRET = 0.001, 0.005
HARMFUL_LOSS, HARMFUL_REGRET = 0.001, 0.01


def render_items(items: list[dict[str, Any]], *, retired: bool = False) -> str:
    active = [item for item in items if item.get("status") == "active"]
    if not active and not retired:
        return ""
    active.sort(key=lambda item: (-(item.get("helpful", 0) - item.get("harmful", 0)), item["id"]))
    lines = []
    for item in active + ([item for item in items if item.get("status") == "retired"] if retired else []):
        evidence = []
        if item.get("supports"):
            evidence.append("supported by " + ", ".join(item["supports"]))
        if item.get("contradicts"):
            evidence.append("contradicted by " + ", ".join(item["contradicts"]))
        evidence.append(f"cited by decisions that did well {item.get('helpful', 0)}x, badly {item.get('harmful', 0)}x")
        status = "" if item.get("status") == "active" else f" [RETIRED: {item.get('retired_because', '')}]"
        lines += [f"[{item['id']}]{status} when: {item['scope']}", f"  {item['text']}", f"  evidence: {'; '.join(evidence)}", ""]
    return "\n".join(lines).rstrip()



# -- workload signature -------------------------------------------------------------


def signature(lab: Lab, trace: str = NAME) -> dict[str, Any]:
    """Deterministic statistics of a workload's observed part, per cache size.

    Cache sizes are fixed fractions of the observed distinct objects, so the
    k-th size means the same thing for every workload."""
    sizes = list(lab.capacities[trace])
    index = lab.index(trace)
    counts = index.request_counts()
    workload = lab.workload(trace, visual=False).data
    out: dict[str, Any] = {
        "requests": int(index.n),
        "distinct_objects": int(index.unique.size),
        "objects_requested_once": round(float((counts == 1).mean()), 4),
        "requests_to_objects_with_9plus": round(float(counts[counts >= 9].sum() / counts.sum()), 4),
        "cache_sizes": sizes,
        "per_size": [],
    }
    base = lab.experiment(BASELINE)
    references = {name: lab.experiment(name) for name in ("lru", "mq") if name in lab.experiments}
    for capacity in sizes:
        positions = analysis.hit_positions(base.run(trace, capacity), index, 0, index.n)
        labels = positions["labels"]
        s_share = dict(zip(labels, positions["queues"]["S"]["share"], strict=True))
        g_share = dict(zip(labels, positions["ghosts"]["G"]["share"], strict=True)) if "G" in positions["ghosts"] else {}
        tail = ("0.75-1", "1-1.5", "1.5-2", ">=2")
        base_mr = base.run(trace, capacity).summary["miss_ratio"]
        row = {
            "capacity": capacity,
            "s3fifo_miss_ratio": round(base_mr, 5),
            "reuse_within_0.1C": round(workload["reuse_within"][f"0.1C({capacity})"], 4),
            "reuse_within_C": round(workload["reuse_within"][f"C({capacity})"], 4),
            "one_hit_share_C_window": _round(workload["one_hit"][f"C({capacity})"]["one_hit_share"]),
            "s_hits_quick": round(s_share.get("0-0.01", 0) + s_share.get("0.01-0.05", 0), 4),
            "s_hits_tail": round(sum(s_share.get(label, 0) for label in tail), 4),
            "ghost_hits_tail": round(sum(g_share.get(label, 0) for label in tail), 4),
        }
        for name, experiment in references.items():
            mr = experiment.run(trace, capacity).summary["miss_ratio"]
            row[f"{name}_vs_s3fifo"] = round((mr - base_mr) / base_mr, 4) if base_mr else 0.0
        out["per_size"].append(row)
    return out


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def signature_text(sig: dict[str, Any]) -> str:
    lines = [
        f"{sig['requests']:,} requests observed, {sig['distinct_objects']:,} distinct objects; "
        f"{sig['objects_requested_once']:.1%} of objects requested once; "
        f"{sig['requests_to_objects_with_9plus']:.1%} of requests go to objects requested 9+ times",
        "size  C        s3fifo_mr  reuse<0.1C  reuse<C  one-hit(C)  S_quick  S_tail  G_tail  lru_vs_s3  mq_vs_s3",
    ]
    for k, row in enumerate(sig["per_size"], start=1):
        lines.append(
            f"C{k}    {row['capacity']:<8} {row['s3fifo_miss_ratio']:.4f}     {row['reuse_within_0.1C']:.4f}      "
            f"{row['reuse_within_C']:.4f}   {_fmt(row['one_hit_share_C_window'])}      {row['s_hits_quick']:.4f}   "
            f"{row['s_hits_tail']:.4f}  {row['ghost_hits_tail']:.4f}  {_pct(row.get('lru_vs_s3fifo'))}   {_pct(row.get('mq_vs_s3fifo'))}"
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


def _pct(value: float | None) -> str:
    return "  n/a " if value is None else f"{100 * value:+6.2f}%"


# Signature features compared between workloads: shares in [0, 1] and relative
# changes (clipped to +-1). Absolute counts are left out, so a 1M-request case
# and a 40M-request workload compare on shape, not length.
_GLOBAL_FEATURES = ("objects_requested_once", "requests_to_objects_with_9plus")
_SIZE_FEATURES = (
    "s3fifo_miss_ratio", "reuse_within_0.1C", "reuse_within_C", "one_hit_share_C_window",
    "s_hits_quick", "s_hits_tail", "ghost_hits_tail", "lru_vs_s3fifo", "mq_vs_s3fifo",
)  # fmt: skip


def signature_distance(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """Mean absolute difference of two signatures' features, the k-th cache size
    against the k-th (sizes are the same fractions of the observed objects).
    0 means identical; None if they share no feature."""
    pairs = [(a.get(key), b.get(key)) for key in _GLOBAL_FEATURES]
    for row_a, row_b in zip(a.get("per_size", []), b.get("per_size", []), strict=False):
        pairs += [(row_a.get(key), row_b.get(key)) for key in _SIZE_FEATURES]
    gaps = [abs(max(-1.0, min(1.0, x)) - max(-1.0, min(1.0, y))) for x, y in pairs if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    return sum(gaps) / len(gaps) if gaps else None


# -- feedback on a training workload ----------------------------------------------------


def feedback_text(rows: list[dict[str, Any]]) -> str:
    """The outcome on the unseen rest of a training workload, per cache size."""
    lines = [
        "Outcome on the remaining requests (miss ratio; relative change against S3-FIFO continuing; negative is better). "
        "Your policy and the tuned S3-FIFO took over S3-FIFO's live cache at the split; published algorithms ran "
        "from a cold cache over the whole workload, so they pay no handover:",
        "C        yours    s3fifo   yours_vs_s3  tuned_s3fifo (prefix-best knobs)                        best_published        belady",
    ]
    for row in rows:
        tuned = row.get("prefix_tuned_s3fifo")
        config = row.get("prefix_tuned_config") or {}
        knobs = ",".join(f"{k.lower()}={v}" for k, v in config.items())
        best = row.get("best_published")
        best_mr = row.get("best_published_miss_ratio")
        lines.append(
            f"{row['capacity']:<8} {row['candidate']:.4f}   {row['s3fifo']:.4f}   {100 * row['relative_vs_s3fifo']:+7.2f}%     "
            + (f"{tuned:.4f} ({100 * (tuned - row['s3fifo']) / row['s3fifo']:+.2f}%; {knobs})" if tuned is not None else "n/a")
            + (f"   {best} {best_mr:.4f} ({100 * (best_mr - row['s3fifo']) / row['s3fifo']:+.2f}%)" if best else "")
            + (f"   {row['belady']:.4f}" if row.get("belady") is not None else "")
        )
    names = list(dict.fromkeys(name for row in rows for name in row.get("published", {})))
    if names:
        lines += ["", "All published algorithms on the remaining requests (miss ratio per C; Belady is the offline optimum):"]
        lines.append("algorithm   " + "".join(f"{row['capacity']:>10}" for row in rows))
        for name in names:
            values = [row["published"].get(name) for row in rows]
            lines.append(f"{name:<11} " + "".join("       n/a" if v is None else f"{v:>10.4f}" for v in values))
    return "\n".join(lines)


# -- the library --------------------------------------------------------------------------


class Library:
    def __init__(self, root: Path) -> None:
        self.root = root
        for name in ("cases", "playbook-history", "knowledge-history", "proposals"):
            (root / name).mkdir(parents=True, exist_ok=True)

    @property
    def playbook(self) -> str:
        path = self.root / "playbook.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    @property
    def knowledge(self) -> str:
        path = self.root / "knowledge.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def _revise(self, name: str, text: str, tag: str) -> None:
        history = self.root / f"{name}-history"
        revision = len(list(history.glob("*.md"))) + 1
        (history / f"{revision:03d}-{tag}.md").write_text(text, encoding="utf-8")
        (self.root / f"{name}.md").write_text(text, encoding="utf-8")

    def set_knowledge(self, text: str, tag: str) -> None:
        self._revise("knowledge", text, tag)

    def set_playbook(self, text: str, tag: str) -> None:
        self._revise("playbook", text, tag)

    def proposals(self) -> list[tuple[str, str]]:
        return [(path.stem, path.read_text(encoding="utf-8")) for path in sorted((self.root / "proposals").glob("*.md"))]

    def clear_proposals(self) -> None:
        for path in (self.root / "proposals").glob("*.md"):
            path.unlink()

    # -- itemized playbook ----------------------------------------------------------
    #
    # ACE (Zhang et al., 2025) found that rewriting a whole playbook each step
    # collapses it (18k -> 122 tokens in one step); so the playbook is a list of
    # rules with stable ids that sessions change only through small operations,
    # applied here by code. Each rule says when it applies (in terms observable
    # in the first part of a workload), what to do, which cases support or
    # contradict it (the agents' judgment), and how often it was cited by a
    # decision that did well or badly on the hidden rest (the host's count).

    def items(self) -> list[dict[str, Any]]:
        path = self.root / "playbook.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []

    def _save_items(self, items: list[dict[str, Any]], tag: str) -> None:
        (self.root / "playbook.json").write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
        self.set_playbook(render_items(items), tag)

    def ops_log(self) -> list[dict[str, Any]]:
        path = self.root / "playbook-ops.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []

    def apply_ops(
        self, ops: list[dict[str, Any]], tag: str, *, known_cases: set[str] | None = None, log: bool = True, dry_run: bool = False
    ) -> tuple[list[str], dict[str, str]]:
        """Apply playbook operations; all or nothing. Returns (messages, ids of added rules
        keyed by their position in `ops`, as "+k"). Raises ValueError on an invalid op;
        with dry_run, only checks them."""
        items = self.items()
        by_id = {item["id"]: item for item in items}
        known_cases = known_cases if known_cases is not None else {case["case"] for case in self.cases()}
        next_number = 1 + max((int(item["id"][1:]) for item in items), default=0)
        messages, added = [], {}
        for position, op in enumerate(ops):
            kind = op.get("op")
            cases = list(op.get("cases") or [])
            unknown = [case for case in cases if case not in known_cases]
            if unknown:
                raise ValueError(f"operation {position + 1}: unknown cases {unknown}")
            if kind == "add":
                if not op.get("text") or not op.get("scope"):
                    raise ValueError(f"operation {position + 1}: add needs text and scope")
                rule_id = f"P{next_number:03d}"
                next_number += 1
                item = {"id": rule_id, "scope": op["scope"], "text": op["text"], "supports": cases, "contradicts": [], "helpful": 0, "harmful": 0, "status": "active", "added_by": tag, "edited_by": []}
                items.append(item)
                by_id[rule_id] = item
                added[f"+{position}"] = rule_id
                messages.append(f"added {rule_id}")
                continue
            rule_id = op.get("id")
            if rule_id not in by_id:
                raise ValueError(f"operation {position + 1}: unknown rule {rule_id!r}; known: {', '.join(sorted(by_id)) or 'none'}")
            item = by_id[rule_id]
            if kind == "edit":
                if not op.get("text") and not op.get("scope"):
                    raise ValueError(f"operation {position + 1}: edit needs text or scope")
                item.update({key: op[key] for key in ("text", "scope") if op.get(key)})
                item["edited_by"] = [*item.get("edited_by", []), tag]
                item["status"] = "active"
                messages.append(f"edited {rule_id}")
            elif kind in ("support", "contradict"):
                if not cases:
                    raise ValueError(f"operation {position + 1}: {kind} needs cases")
                field_name = "supports" if kind == "support" else "contradicts"
                item[field_name] = sorted({*item.get(field_name, []), *cases})
                messages.append(f"{kind}ed {rule_id} with {', '.join(cases)}")
            elif kind == "retire":
                if not op.get("reason"):
                    raise ValueError(f"operation {position + 1}: retire needs a reason")
                item["status"] = "retired"
                item["retired_because"] = op["reason"]
                messages.append(f"retired {rule_id}")
            else:
                raise ValueError(f"operation {position + 1}: unknown op {kind!r}")
        if dry_run:
            return messages, added
        self._save_items(items, tag)
        if log:
            with open(self.root / "playbook-ops.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"tag": tag, "ops": ops, "added": added}) + "\n")
        return messages, added

    def credit(self, case_id: str) -> str | None:
        """The host's verified count: a case's decision, scored on the hidden rest
        of its workload against S3-FIFO and against the best candidate of its own
        session, credits the rules it cited. Applied once per case."""
        record, _ = self.case(case_id)
        cited = record.get("cited") or []
        if record.get("credited") or not cited or not record.get("counterfactuals"):
            return None
        mean = (record.get("outcome") or {}).get("mean_relative_vs_s3fifo")
        scored = [item["mean_relative_vs_s3fifo"] for item in record["counterfactuals"] if item.get("mean_relative_vs_s3fifo") is not None]
        if mean is None or not scored:
            return None
        regret = mean - min(scored)
        verdict = "helpful" if mean < -HELPFUL_GAIN and regret <= HELPFUL_REGRET else "harmful" if mean > HARMFUL_LOSS or regret > HARMFUL_REGRET else None
        items = self.items()
        if verdict is not None:
            for item in items:
                if item["id"] in cited:
                    item[verdict] = int(item.get(verdict, 0)) + 1
            self._save_items(items, f"credit-{case_id}")
        self.update_case(case_id, credited=True, credit={"verdict": verdict, "regret": regret})
        return verdict

    # -- host-only data ------------------------------------------------------------

    def origins(self) -> dict[str, Any]:
        path = self.root / "ORIGINS.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def set_origin(self, case_id: str, origin: dict[str, Any]) -> None:
        origins = self.origins()
        origins[case_id] = origin
        (self.root / "ORIGINS.json").write_text(json.dumps(origins, indent=1) + "\n", encoding="utf-8")

    def update_case(self, case_id: str, **fields: Any) -> None:
        path = self.root / "cases" / f"{case_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(fields)
        path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")

    def add_postmortem(self, case_id: str, text: str, tag: str) -> None:
        record, _ = self.case(case_id)
        self.update_case(case_id, postmortems=[*record.get("postmortems", []), {"by": tag, "text": text}])

    def playbook_digest(self) -> str:
        return hashlib.sha256(self.playbook.encode("utf-8")).hexdigest()

    def cases(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((self.root / "cases").glob("case-*.json"))]

    def case(self, case_id: str) -> tuple[dict[str, Any], str]:
        path = self.root / "cases" / f"{case_id}.json"
        if not path.is_file() or "/" in case_id:
            known = ", ".join(case["case"] for case in self.cases())
            raise KeyError(f"unknown case {case_id!r}; known: {known or 'none'}")
        code_path = path.with_suffix(".py")
        return json.loads(path.read_text(encoding="utf-8")), code_path.read_text(encoding="utf-8") if code_path.is_file() else ""

    def add_case(self, record: dict[str, Any], code: str) -> str:
        number = 1 + max((int(case["case"].split("-")[1]) for case in self.cases()), default=0)
        case_id = f"case-{number:03d}"
        record = {"case": case_id, **record}
        (self.root / "cases" / f"{case_id}.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        (self.root / "cases" / f"{case_id}.py").write_text(code, encoding="utf-8")
        return case_id

    def add(self, record: dict[str, Any], code: str, playbook: str) -> str:
        case_id = self.add_case(record, code)
        self.set_playbook(playbook, case_id)
        return case_id

    def playbook_is_merged(self) -> bool:
        """True while the playbook is still the unreconciled sections from a merge."""
        revisions = sorted((self.root / "playbook-history").glob("*.md"))
        return bool(revisions) and revisions[-1].stem.endswith(MERGED)

    def table(self, signature: dict[str, Any] | None = None) -> str:
        """The cases, most similar to `signature` (this workload's) first when given."""
        cases = self.cases()
        if not cases:
            return "(no cases yet)"
        distances = {case["case"]: signature_distance(signature, case["signature"]) if signature and case.get("signature") else None for case in cases}
        if any(distance is not None for distance in distances.values()):
            cases.sort(key=lambda case: (distances[case["case"]] is None, distances[case["case"]] or 0.0, case["case"]))
        lines = ["case      distance  mean_vs_s3fifo  vs_tuned  vs_best_published  reviewed  decision (short)"]
        for case in cases:
            outcome = case.get("outcome", {})
            reviewed = "yes" if case.get("postmortems") else "no"
            distance = distances[case["case"]]
            lines.append(
                f"{case['case']:<9} {'  n/a ' if distance is None else f'{distance:6.3f}'}    {_pct(outcome.get('mean_relative_vs_s3fifo'))}        "
                f"{_pct(outcome.get('mean_relative_vs_tuned'))}  {_pct(outcome.get('mean_relative_vs_best_published'))}            "
                f"{reviewed:<8}  {case.get('decision', '')[:100]}"
            )
        return "\n".join(lines)

    def prompt_section(self, signature: dict[str, Any] | None = None) -> str:
        cases = self.cases()
        if not cases:
            return "\n".join(
                [
                    "# Tuning experience from earlier workloads",
                    "",
                    "None yet: this is the first workload. The playbook you write after seeing the outcome starts it.",
                ]
            )
        parts = [
            "# Tuning experience from earlier workloads",
            "",
            f"Earlier engineer sessions tuned {len(cases)} other workloads, saw how their decisions did on the "
            "rest of each workload, and kept a playbook. Use it as experience, not as rules: check that a "
            "lesson's conditions hold here before applying it. read_case shows a case's full signature, "
            "decision, code, outcome and lesson.",
            "",
            "## Playbook",
            "",
            self.playbook.strip() or "(empty)",
            "",
            *(["## Knowledge (mechanisms found in hindsight analysis)", "", self.knowledge.strip(), ""] if self.knowledge.strip() else []),
            "## Cases",
            "",
            *(
                [
                    "Most similar workload first: distance is the mean absolute difference between a case's signature "
                    "and this workload's (shares and changes vs S3-FIFO, size by size; 0 = identical; "
                    "two different workloads are usually 0.1 to 0.3 apart). "
                    "Similar signatures are a hint, not a guarantee that the same decision works.",
                    "",
                ]
                if signature
                else []
            ),
            self.table(signature),
        ]
        return "\n".join(parts)

    def analyst_section(self) -> str:
        """The library as the analyst sees it: everything, including pending proposals."""
        playbook = render_items(self.items(), retired=True) or "(no rules yet)"
        parts = [
            "# The library",
            "",
            "## Playbook rules (current; ids are stable, change them with playbook_ops)",
            "",
            playbook,
            "",
            "## Knowledge (current)",
            "",
            self.knowledge.strip() or "(empty)",
        ]
        proposals = self.proposals()
        if proposals:
            parts += ["", "## Free-text playbooks from training sessions (turn what holds into rules)"]
            for name, text in proposals:
                parts += ["", f"### {name}", "", text.strip()]
        parts += ["", "## Cases", "", self.table()]
        return "\n".join(parts)

    def case_text(self, case_id: str) -> str:
        record, code = self.case(case_id)
        lines = [
            f"{record['case']}",
            "signature:",
            signature_text(record["signature"]),
            f"decision: {record.get('decision', '')}",
            f"reasoning: {record.get('reasoning', '')}",
            "outcome:",
            record.get("feedback", ""),
            f"lesson: {record.get('lesson', '')}",
            *_counterfactual_lines(record.get("counterfactuals")),
            *(f"postmortem ({item['by']}): {item['text']}" for item in record.get("postmortems", [])),
            "code:",
            "```python",
            code.rstrip(),
            "```",
        ]
        return "\n".join(lines)


def _counterfactual_lines(records: list[dict[str, Any]] | None) -> list[str]:
    """Every candidate of the session, scored on the rest of the workload in hindsight."""
    if not records:
        return []
    every = all(item.get("complete", True) for item in records)
    lines = [
        ("all candidates" if every else "the best candidates on the observed part")
        + " of that session, in hindsight (change vs S3-FIFO on the observed part -> on the rest):"
    ]
    for item in sorted(records, key=lambda r: (r.get("mean_relative_vs_s3fifo") is None, r.get("mean_relative_vs_s3fifo") or 0)):
        lines.append(
            f"  {item['experiment']:<8} {item['name'][:40]:<40} {_pct(item.get('prefix_relative_vs_s3fifo'))} -> {_pct(item.get('mean_relative_vs_s3fifo'))}"
        )
    return lines


def compact_counterfactuals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = ("experiment", "name", "reference", "prefix_relative_vs_s3fifo", "mean_relative_vs_s3fifo", "suffix", "complete")
    return [{key: record.get(key) for key in keep} for record in records]


def absorb(target: Library, source: Library, *, label: str, since: int = 0, since_ops: int = 0) -> list[str]:
    """Copy the cases `source` added after its first `since` into `target`
    (renumbered, with their origins), and replay the playbook operations it made
    after its first `since_ops` (rule and case ids remapped; the host credits the
    copied cases again in `target`). A source without operations (a free-text
    playbook) contributes its playbook as a proposal for the analyst."""
    origins = source.origins()
    added, case_map = [], {}
    for case in source.cases()[since:]:
        record, code = source.case(case["case"])
        old = record.pop("case")
        record["origin"] = f"{label}, {old}"
        record.pop("credited", None)
        record.pop("credit", None)
        new = target.add_case(record, code)
        case_map[old] = new
        if old in origins:
            target.set_origin(new, origins[old])
        added.append(new)
    log = source.ops_log()[since_ops:]
    if log:
        rule_map: dict[str, str] = {}
        skipped = []
        for entry in log:
            ops = []
            for op in entry["ops"]:
                op = dict(op)
                if op.get("cases"):
                    op["cases"] = [case_map.get(case, case) for case in op["cases"]]
                if op.get("id"):
                    op["id"] = rule_map.get(op["id"], op["id"])
                ops.append(op)
            try:
                _, new_ids = target.apply_ops(ops, f"{label}:{entry['tag']}")
            except ValueError as exc:
                skipped.append(f"- {entry['tag']}: {exc}")
                continue
            for key, source_id in (entry.get("added") or {}).items():
                if key in new_ids:
                    rule_map[source_id] = new_ids[key]
        for new in added:
            record, _ = target.case(new)
            if record.get("cited"):
                target.update_case(new, cited=[rule_map.get(rule, rule) for rule in record["cited"]])
        if skipped:
            (target.root / "proposals" / f"{label}-skipped-ops.md").write_text(
                "Playbook operations that could not be replayed:\n" + "\n".join(skipped) + "\n", encoding="utf-8"
            )
    elif source.playbook.strip() and not source.items():
        (target.root / "proposals" / f"{label}.md").write_text(source.playbook, encoding="utf-8")
    for new in added:
        target.credit(new)
    return added


def merge(sources: list[Path], target: Path) -> Path:
    """Combine libraries from parallel training sessions: all cases, renumbered,
    with origins. Until an analyst reconciles them, the playbook is the
    sessions' playbooks as sections (also kept as proposals)."""
    target.mkdir(parents=True, exist_ok=False)
    merged = Library(target)
    knowledge = []
    for lineage, source in enumerate(sources, start=1):
        library = Library(source)
        absorb(merged, library, label=f"session-{lineage}")
        if library.knowledge.strip():
            knowledge.append(f"## From training session {lineage}\n\n{library.knowledge.strip()}")
    sections = [f"## {name.replace('session-', 'From training session ')}\n\n{text.strip()}" for name, text in merged.proposals()]
    if merged.items():
        # Rules replayed from the sessions' operations, unreconciled (duplicates and
        # conflicts across sessions are the analyst's to resolve).
        sections.insert(0, "## Rules from the training sessions (unreconciled)\n\n" + render_items(merged.items()))
    merged.set_playbook("\n\n".join(sections), MERGED)
    if knowledge:
        merged.set_knowledge("\n\n".join(knowledge), MERGED)
    return target


def validity_text(report: dict[str, Any]) -> str:
    """Why a verified decision would not count, or "" when it would."""
    if report.get("valid", True):
        return ""
    reasons = []
    if report.get("failures"):
        failed = report["failures"]
        reasons.append(f"it failed at C={', '.join(str(f['capacity']) for f in failed)} ({failed[0]['error'].strip().splitlines()[-1] if failed[0]['error'].strip() else 'no message'})")
    if report.get("memory_violations"):
        over = ", ".join(f"{v['capacity']} ({v['ratio']:.2f}x)" for v in report["memory_violations"])
        reasons.append(f"its memory was over the limit (S3-FIFO's x {MEMORY_LIMIT_RATIO:g}) at C={over}")
    if report.get("fast_verified_mismatches"):
        sizes = ", ".join(str(m["capacity"]) for m in report["fast_verified_mismatches"])
        reasons.append(f"its checked rerun on the observed requests missed differently from the experiment at C={sizes}")
    return (
        "INVALID: " + "; ".join(reasons) + ". On a test workload this decision would count as S3-FIFO unchanged (0%), "
        "whatever its miss ratios below."
    )


def outcome_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    tuned = [(r["candidate"] - r["prefix_tuned_s3fifo"]) / r["prefix_tuned_s3fifo"] for r in rows if r.get("prefix_tuned_s3fifo")]
    return {
        "valid": report.get("valid", True),
        "mean_relative_vs_s3fifo": report.get("mean_relative_vs_s3fifo"),
        "mean_relative_vs_tuned": float(np.mean(tuned)) if tuned else None,
        "mean_relative_vs_best_published": report.get("mean_relative_vs_best_published"),
    }
