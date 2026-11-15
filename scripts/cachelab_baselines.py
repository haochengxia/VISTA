"""Compute-heavy CacheLab reference runs (meant for a large machine).

    python scripts/cachelab_baselines.py --lab-dir DIR --out DIR \
        [--heldout-suite FILE] [--workers N] [--seeds 10] [--budget 12] [--grid]

Writes one JSON file per job into --out:
  references.json   LRU and FIFO against S3-FIFO (verified, dev + held-out)
  search-seed{k}.json  parameter-search arm at the agent's budget, per seed
  grid.json         every grid configuration on dev, the best one scored held-out
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from vista.benchmarks.cachelab.engine import SANDBOX_DIR, DockerSandbox
from vista.benchmarks.cachelab.lab import Lab
from vista.benchmarks.cachelab.search import DEFAULT, GRID, policy_source, search
from vista.benchmarks.cachelab.traces import DEV_SUITE, HELDOUT_SUITE, load_suite


def write(out: Path, name: str, value) -> None:
    (out / name).write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev-suite", type=Path)
    parser.add_argument("--heldout-suite", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--skip-references", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lab = Lab(
        args.lab_dir,
        DockerSandbox(workers=args.workers),
        dev=load_suite(args.dev_suite) if args.dev_suite else DEV_SUITE,
        heldout=load_suite(args.heldout_suite) if args.heldout_suite else HELDOUT_SUITE,
        workers=args.workers,
    )
    print(f"[{time.strftime('%H:%M:%S')}] lab ready: {len(lab.cells)} dev cells, {len(lab.heldout_specs)} held-out traces", flush=True)
    if not args.skip_references:
        references = {}
        for name in ("lru", "fifo"):
            source = (SANDBOX_DIR / f"{name}.py").read_text()
            experiment, _ = lab.dev_experiment(name, source, prune=True)
            references[name] = lab.final_evaluation(experiment.id, prune=True)
        write(args.out, "references.json", references)
    for seed in range(args.seeds):
        write(args.out, f"search-seed{seed}.json", search(lab, budget=args.budget, seed=seed, prune=True))
    if args.grid:
        trials = []
        for values in itertools.product(*GRID.values()):
            config = dict(zip(GRID, values, strict=True))
            if config == DEFAULT:
                continue
            experiment, comparison = lab.dev_experiment("grid", policy_source(config), prune=True)
            trials.append(
                {
                    "experiment": experiment.id,
                    "config": config,
                    "dev_mean_relative_change": comparison["mean_relative_change"] if comparison else None,
                    "dev_rows": comparison["rows"] if comparison else None,
                }
            )
            if len(trials) % 24 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] grid {len(trials)} configs", flush=True)
        best = min((t for t in trials if t["dev_mean_relative_change"] is not None), key=lambda t: t["dev_mean_relative_change"])
        write(args.out, "grid.json", {"trials": trials, "best": best, "evaluation": lab.final_evaluation(best["experiment"], prune=True)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
