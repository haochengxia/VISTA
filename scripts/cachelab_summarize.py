"""Summarize CacheLab evaluation reports.

    python scripts/cachelab_summarize.py DIR [DIR ...] [--unseen fiu,metaStorage,ARC] [--experiments]

Reads reports written by cachelab_baselines.py and agent runs' result.json
files (searched recursively). --experiments also lists each agent experiment
with its predicted and observed dev change.

Prints one row per evaluated policy: dev and held-out mean relative miss-ratio
change against S3-FIFO, and the held-out mean per dataset (held-out traces are
named ho-<dataset>-<k>). Negative is better.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def pct(value) -> str:
    return "   n/a" if value is None else f"{100 * value:+6.2f}"


def agent_runs(directory: Path):
    for path in sorted(directory.rglob("result.json")):
        data = json.loads(path.read_text())
        if data.get("benchmark") == "CacheLab":
            yield path.parent.name, data


def reports(directory: Path):
    for name, data in agent_runs(directory):
        evaluation = data.get("evaluation") or {}
        if "heldout" in evaluation:
            tag = "" if data.get("submitted") else " (fallback)"
            yield f"agent {data['profile']} {name}{tag}", evaluation
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        if path.name == "references.json":
            for name, report in data.items():
                yield name, report
        elif path.name.startswith("search-seed"):
            config = data.get("selected", {}).get("config", {})
            label = f"search s{data['seed']} " + ",".join(f"{v}" for v in config.values())
            if "evaluation" in data:
                yield label, data["evaluation"]
        elif path.name == "grid.json":
            config = data["best"]["config"]
            yield "grid-best " + ",".join(f"{v}" for v in config.values()), data["evaluation"]
        elif "heldout" in data and "dev" in data:
            yield path.stem, data


def print_experiments(name: str, data: dict) -> None:
    print(f"{data['profile']} {name}: submitted={data.get('submitted')} score={pct(data.get('score'))} error={data.get('error')}")
    print(f"  tools: {json.dumps(data.get('tool_counts', {}), sort_keys=True)}")
    errors = []
    for e in data.get("experiments", []):
        observed, expected = e.get("observed_change_pct"), e.get("expected_change_pct")
        if observed is not None and expected is not None:
            errors.append(abs(observed - expected))
        shown = "  failed" if observed is None else f"{observed:+7.2f}"
        print(f"  {e['experiment'] or '-':>4} {e['name'][:40]:40} predicted {expected:+6.2f}  observed {shown}")
    if errors:
        print(f"  mean |prediction error| {sum(errors) / len(errors):.2f} points over {len(errors)} experiments")
    submission = data.get("submission") or {}
    if submission:
        print(f"  submitted {submission.get('experiment')}: {submission.get('mechanism', '')[:300]}")
    print()


def dataset(trace: str) -> str:
    parts = trace.split("-")
    return parts[1] if len(parts) >= 3 else trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--unseen", default="fiu,metaStorage,ARC")
    parser.add_argument("--experiments", action="store_true")
    args = parser.parse_args()
    unseen = set(args.unseen.split(","))
    rows = [row for directory in args.directories for row in reports(directory)]
    if args.experiments:
        for directory in args.directories:
            for name, data in agent_runs(directory):
                print_experiments(name, data)
    if not rows:
        print("no reports yet")
        return 0
    datasets = sorted({dataset(r["trace"]) for _, report in rows for r in report["heldout"]["rows"]}, key=lambda d: (d in unseen, d))
    header = f"{'policy':34} {'valid':5} {'dev':>6} {'hold':>6} {'seen':>6} {'unseen':>6}  " + " ".join(f"{d[:7]:>7}" for d in datasets)
    print(header)
    print("-" * len(header))
    for label, report in rows:
        groups = defaultdict(list)
        for r in report["heldout"]["rows"]:
            groups[dataset(r["trace"])].append(r["relative"])
        seen_values = [v for d, values in groups.items() if d not in unseen for v in values]
        unseen_values = [v for d, values in groups.items() if d in unseen for v in values]
        mean = lambda values: sum(values) / len(values) if values else None  # noqa: E731
        print(
            f"{label[:34]:34} {str(report['valid']):5} {pct(report['dev']['mean_relative_change'])} "
            f"{pct(report['heldout']['mean_relative_change'])} {pct(mean(seen_values))} {pct(mean(unseen_values))}  "
            + " ".join(f"{pct(mean(groups[d])):>7}" for d in datasets)
        )
    print("\nvalues are mean relative miss-ratio change vs S3-FIFO in %, negative is better")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
