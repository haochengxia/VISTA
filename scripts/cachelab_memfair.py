"""Memory-fair prefix-tuned S3-FIFO for an engineer-baselines directory.

The baselines' prefix tuning may pick GHOST_RATIO 1.5 or 2.0. On real traces
those deep ghost queues use 2.3-3x S3-FIFO's memory, over the 2x limit an agent's
policy must meet. This reruns the same tuning (coordinate descent on the observed
prefix, switched on at the split of the live S3-FIFO cache) with GHOST_RATIO
capped at S3-FIFO's default, 0.9, and writes a copy of each report with the
prefix-tuned fields replaced. The uncapped ones stay as prefix_tuned_uncapped_*.

    python scripts/cachelab_memfair.py --suite suites/heldout-long.json \\
        --baselines results/engineer/heldout-long --lab-dir /mydata/memfair/heldout-long \\
        --out results/engineer/heldout-long-memfair --workers 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vista.benchmarks.cachelab import engineer  # noqa: E402
from vista.benchmarks.cachelab.engine import LocalSandbox  # noqa: E402
from vista.benchmarks.cachelab.search import DEFAULT, GRID  # noqa: E402
from vista.benchmarks.cachelab.traces import load_suite  # noqa: E402

MAX_GHOST = DEFAULT["GHOST_RATIO"]
CAPPED = {**GRID, "GHOST_RATIO": tuple(value for value in GRID["GHOST_RATIO"] if value <= MAX_GHOST)}


def capped_report(report: dict, tuned: dict[int, dict]) -> dict:
    """`report` with its prefix-tuned fields replaced by `tuned` (capacity -> prefix_tuned's entry)."""
    out = {**report, "prefix_tuned_grid": {name: list(values) for name, values in CAPPED.items()}}
    rows = []
    for row in report["rows"]:
        entry = tuned.get(row["capacity"], {})
        rows.append(
            {
                **row,
                "prefix_tuned_uncapped_s3fifo": row.get("prefix_tuned_s3fifo"),
                "prefix_tuned_uncapped_config": row.get("prefix_tuned_config"),
                "prefix_tuned_s3fifo": entry.get("suffix_miss_ratio"),
                "prefix_tuned_config": entry.get("config"),
            }
        )
    out["rows"] = rows
    changes = [(r["prefix_tuned_s3fifo"] - r["s3fifo"]) / r["s3fifo"] for r in rows if r["prefix_tuned_s3fifo"] is not None and r["s3fifo"]]
    out["mean_relative_prefix_tuned_vs_s3fifo"] = float(np.mean(changes)) if changes else None
    return out


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--suite", type=Path, required=True)
    cli.add_argument("--baselines", type=Path, required=True, help="engineer-baselines reports, <target>.json")
    cli.add_argument("--lab-dir", type=Path, required=True, help="trace and simulation cache (its own; nothing else is reused)")
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--shard", default="0/1", help="process targets k, k+n, ... given k/n")
    args = cli.parse_args()
    shard, shards = (int(part) for part in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    for position, spec in enumerate(load_suite(args.suite)):
        out, source = args.out / f"{spec.name}.json", args.baselines / f"{spec.name}.json"
        if position % shards != shard or out.is_file() or not source.is_file():
            continue
        started = time.time()
        report = json.loads(source.read_text(encoding="utf-8"))
        root = args.lab_dir / spec.name
        task = engineer.prepare(spec, root, fraction=report["split_fraction"])
        if task.split != report["split"] or list(task.capacities) != [row["capacity"] for row in report["rows"]]:
            raise SystemExit(f"{spec.name}: the report belongs to a different task")
        lab = engineer.make_lab(task, root, LocalSandbox(workers=args.workers), workers=args.workers)
        result = capped_report(report, engineer.prefix_tuned(lab, task, grid=CAPPED))
        out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        engineer.prune_arrays(root)
        for row in result["rows"]:
            s3 = row["s3fifo"]
            fmt = lambda value: "  n/a " if value is None else f"{100 * (value - s3) / s3:+6.2f}%"  # noqa: E731
            print(
                f"{spec.name:<20} C={row['capacity']:<8} capped {fmt(row['prefix_tuned_s3fifo'])} {row['prefix_tuned_config']}"
                f"  uncapped {fmt(row['prefix_tuned_uncapped_s3fifo'])}",
                flush=True,
            )
        print(f"{spec.name}: {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
