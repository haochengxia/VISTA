"""How much of the engineer's result is the handover, and how much the decision?

For every target of a suite and every cache size, the S3-FIFO knobs tuned on
the prefix (from an engineer-baselines report) are run over the whole trace
twice: switched on at the split of the live S3-FIFO cache (the deployment),
and from a cold cache (as if they had run all along). On the requests after
the split:

    handover cost   = tuned with handover - tuned from cold
    decision effect = tuned from cold - S3-FIFO

both relative to S3-FIFO. A large handover cost at a cache size means the
inherited state, not the choice of knobs, decides the outcome there.

    python scripts/cachelab_handover.py --suite suites/heldout-real.json \\
        --baselines results/engineer/test --lab-dir runs/baselines/test --out results/handover-test.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vista.benchmarks.cachelab import engineer  # noqa: E402
from vista.benchmarks.cachelab.engine import (  # noqa: E402
    BASELINE_SOURCE,
    LocalSandbox,
    Run,
    RunStore,
)
from vista.benchmarks.cachelab.search import policy_source  # noqa: E402
from vista.benchmarks.cachelab.traces import load_suite  # noqa: E402


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--suite", type=Path, required=True)
    cli.add_argument("--baselines", type=Path, required=True)
    cli.add_argument("--lab-dir", type=Path, required=True, help="engineer-baselines --lab-dir (traces are reused)")
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    rows = []
    for spec in load_suite(args.suite):
        report_path = args.baselines / f"{spec.name}.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        root = args.lab_dir / spec.name
        task = engineer.prepare(spec, root, fraction=report["split_fraction"])
        # Its own store: the baselines' cached runs kept summaries only, and suffixes need the arrays.
        store = RunStore((root / "handover-diagnostic").resolve(), LocalSandbox(workers=args.workers))
        for row in report["rows"]:
            config = row.get("prefix_tuned_config")
            if not config:
                continue
            capacity = row["capacity"]
            cell = [(task.full, capacity)]
            source = policy_source(config)
            runs = {
                "s3fifo": store.simulate(BASELINE_SOURCE, cell),
                "handover": store.simulate(source, cell, handover=task.fraction),
                "cold": store.simulate(source, cell),
            }
            runs = {name: result[("target", capacity)] for name, result in runs.items()}
            if not all(isinstance(run, Run) for run in runs.values()):
                print(f"{spec.name} C={capacity}: a run failed", file=sys.stderr)
                continue
            suffix = {name: engineer.suffix_miss_ratio(run, task.split) for name, run in runs.items()}
            base = suffix["s3fifo"]
            rows.append(
                {
                    "target": spec.name,
                    "size": task.capacities.index(capacity),
                    "fraction": task.fractions[task.capacities.index(capacity)],
                    "capacity": capacity,
                    "config": config,
                    "suffix_miss_ratio": suffix,
                    "tuned_handover_vs_s3fifo": (suffix["handover"] - base) / base if base else 0.0,
                    "tuned_cold_vs_s3fifo": (suffix["cold"] - base) / base if base else 0.0,
                    "handover_cost": (suffix["handover"] - suffix["cold"]) / base if base else 0.0,
                }
            )
            print(
                f"{spec.name:<20} C={capacity:<7} handover {100 * rows[-1]['tuned_handover_vs_s3fifo']:+6.2f}%  "
                f"cold {100 * rows[-1]['tuned_cold_vs_s3fifo']:+6.2f}%  handover cost {100 * rows[-1]['handover_cost']:+6.2f}%",
                flush=True,
            )
        engineer.prune_arrays(root / "handover-diagnostic")
    summary = {}
    for fraction in sorted({row["fraction"] for row in rows}):
        group = [row for row in rows if row["fraction"] == fraction]
        summary[str(fraction)] = {
            key: float(np.mean([row[key] for row in group]))
            for key in ("tuned_handover_vs_s3fifo", "tuned_cold_vs_s3fifo", "handover_cost")
        } | {"cells": len(group)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
