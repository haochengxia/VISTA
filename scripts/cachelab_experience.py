"""Experience learning for the engineer role: train on many workloads, then tune unseen ones.

    train      K training lineages in parallel. Each lineage works through its share
               of the training targets one at a time; after every decision the agent
               sees the outcome on the rest of the workload and adds a case and a
               revised playbook to the lineage's library. Round 1 starts from empty
               libraries; with --round N >= 2 every lineage starts from a copy of the
               merged (refined) library and takes the next --per-round targets of its
               share that no lineage of this root has trained on. With --allow-repeats
               it wraps around once those run out (a second pass re-tunes workloads
               with what the library has learned since); repeats are listed in
               train/round-N/REPEATS.json, since the library already holds a case
               for that very workload.
    merge      record each case's origin and hindsight counterfactuals (host only),
               then combine the round's lineage libraries into ROOT/library/merged
               (round 1 creates it; later rounds add their new cases to it). The
               lineages' playbooks become proposals for the analyst.
    analyze    an analyst session studies the training workloads in hindsight and
               rewrites the merged library's postmortems, knowledge and playbook.
    loop       rounds FIRST..LAST of train, merge, analyze: the refinement loop.
    test       every test target, once per arm: `experience` (the merged library,
               read only, no feedback) and `plain` (same agent, no library), each
               with its own experiment budget (arm:budget, e.g. experience:1).
    summarize  training curve per lineage, and test scores against S3-FIFO, S3-FIFO
               tuned on the prefix, and the best published algorithm.

Agent arguments follow `--` and are passed to every run, e.g.

    python scripts/cachelab_experience.py train --root runs/xp --suite suites/train-real.json \\
        --baselines results/engineer/train --lineages 3 --budget 40 -- \\
        --model gpt-5.6-sol --effort high --binary <codex> --arm vista --workers 8

Baselines come from `vista-cachelab engineer-baselines` on the same suite. Run
and lab directories get neutral names (w01, w02, ...), because the agent can see
its working directory; ROOT/targets.json maps them back on the host.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vista.benchmarks.cachelab.engineer import prune_arrays  # noqa: E402
from vista.benchmarks.cachelab.experience import (  # noqa: E402
    Library,
    absorb,
    compact_counterfactuals,
)
from vista.benchmarks.cachelab.traces import load_suite  # noqa: E402

VISTA = REPO / ".venv" / "bin" / "vista"
ARMS = "experience:1,plain:1,experience:3,plain:3"


def _targets(root: Path, phase: str, suite: Path, seed: int) -> dict[str, str]:
    """Neutral id -> target name, fixed on first use."""
    path = root / "targets.json"
    mapping = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if phase not in mapping:
        names = [spec.name for spec in load_suite(suite)]
        random.Random(seed).shuffle(names)
        mapping[phase] = {"suite": str(suite.resolve()), "targets": {f"w{i:02d}": name for i, name in enumerate(names, start=1)}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    elif mapping[phase]["suite"] != str(suite.resolve()):
        raise SystemExit(f"{path} assigns {phase} to {mapping[phase]['suite']}, not {suite}")
    return mapping[phase]["targets"]


MIN_FREE_BYTES = 5 << 30


RUN_BACKOFF_SECONDS = (120, 600)  # retries of a run that crashed before submitting


def _wait_for_disk(path: Path) -> None:
    """A full disk corrupts runs silently (truncated JSON); wait for space instead."""
    warned = False
    while shutil.disk_usage(path).free < MIN_FREE_BYTES:
        if not warned:
            print(f"less than {MIN_FREE_BYTES >> 30} GB free under {path}; waiting before the next run", flush=True)
            warned = True
        time.sleep(300)


def _run(out: Path, lab: Path, target: str, suite: Path, baselines: Path, agent: list[str], *, backend: str, extra: list[str]) -> dict | None:
    """One engineer run; a finished run is reused, a crashed one (no result, or a
    backend failure before anything was submitted) is moved aside and rerun,
    once more after each backoff in RUN_BACKOFF_SECONDS (gateway outages come
    in bursts)."""
    result = out / "task" / "result.json"
    for delay in (0, *RUN_BACKOFF_SECONDS):
        if result.is_file():
            finished = json.loads(result.read_text(encoding="utf-8"))
            if not (finished.get("error") and not finished.get("submission")):
                return finished
        time.sleep(delay)
        _wait_for_disk(out.parent if out.parent.exists() else REPO)
        if out.exists():
            crashed = 1
            while out.with_name(f"{out.name}.crashed-{crashed}").exists():
                crashed += 1
            out.rename(out.with_name(f"{out.name}.crashed-{crashed}"))
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(VISTA), "--profile", "cachelab", "--backend", backend, "run",
            "--role", "engineer", "--shots",
            "--target-suite", str(suite), "--target", target,
            "--baselines", str(baselines),
            "--lab-dir", str(lab), "--output", str(out),
            *extra, *agent,
        ]  # fmt: skip
        with open(out.with_suffix(".log"), "w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=False)
        if result.is_file():
            prune_arrays(lab)  # the run is scored; its views are no longer needed
    if not result.is_file():
        return None
    return json.loads(result.read_text(encoding="utf-8"))


def _round_paths(root: Path, round_: int, k: int) -> tuple[Path, Path, Path]:
    """(library, runs, labs) of lineage k in a round; round 1 keeps its original layout."""
    if round_ == 1:
        return root / "library" / f"lineage-{k}", root / "train" / f"lineage-{k}", root / "labs" / "train" / f"lineage-{k}"
    return (
        root / "library" / f"round-{round_}" / f"lineage-{k}",
        root / "train" / f"round-{round_}" / f"lineage-{k}",
        root / "labs" / "train" / f"round-{round_}" / f"lineage-{k}",
    )


def _backfill(library_dir: Path, runs_dir: Path) -> int:
    """Record the origin (host only) and hindsight counterfactuals of every case these runs added."""
    if not library_dir.is_dir() or not runs_dir.is_dir():
        return 0
    library = Library(library_dir)
    origins, done = library.origins(), 0
    for out in sorted(path for path in runs_dir.iterdir() if path.is_dir() and "." not in path.name):
        result_path = out / "task" / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        case = (result.get("submission") or {}).get("case")
        if not case or case in origins:
            continue
        task = result["task"]
        library.set_origin(case, {key: task[key] for key in ("target", "split", "split_fraction", "capacities")} | {"run": str(out)})
        if result.get("counterfactuals"):
            library.update_case(case, counterfactuals=compact_counterfactuals(result["counterfactuals"]))
        library.credit(case)  # count the cited rules' outcome (host-side hindsight)
        done += 1
    return done


def _trained(root: Path, round_: int) -> set[str]:
    """Workloads any lineage of this root trained on before `round_`."""
    used = set()
    for earlier in range(1, round_):
        runs = _round_paths(root, earlier, 1)[1].parent
        for lineage in runs.glob("lineage-*"):
            used |= {path.name for path in lineage.iterdir() if path.is_dir() and ".crashed-" not in path.name}
    return used


def cmd_train(args, agent: list[str]) -> None:
    targets = _targets(args.root, "train", args.suite, args.seed)
    ids = list(targets)
    shares = [ids[k :: args.lineages] for k in range(args.lineages)]
    merged = args.root / "library" / "merged"
    if args.round == 1:
        lineages = [share[: args.limit or None] for share in shares]
    else:
        if not (merged / "playbook.md").is_file():
            raise SystemExit("round >= 2 starts from ROOT/library/merged; merge (and analyze) first")
        per = args.per_round or 1
        if per > min(len(share) for share in shares if share):
            raise SystemExit("--per-round exceeds a lineage's share of targets")
        used = _trained(args.root, args.round)
        lineages, repeats = [], []
        for k, share in enumerate(shares, start=1):
            fresh = [neutral for neutral in share if neutral not in used]
            if len(fresh) < per and not args.allow_repeats:
                raise SystemExit(
                    f"lineage {k} has {len(fresh)} workloads left that no earlier round trained on; round {args.round} needs {per}. "
                    "Pass --allow-repeats for a second pass (the library already holds cases for those workloads)."
                )
            again = [neutral for neutral in share if neutral in used]  # in share order: oldest first
            chosen = (fresh + again)[:per]
            repeats += [{"lineage": k, "workload": neutral} for neutral in chosen if neutral in used]
            lineages.append(chosen)
        if repeats:
            out = _round_paths(args.root, args.round, 1)[1].parent / "REPEATS.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(repeats, indent=1) + "\n", encoding="utf-8")

    def lineage(k: int) -> None:
        library, runs, labs = _round_paths(args.root, args.round, k)
        if args.round > 1 and not library.exists():
            shutil.copytree(merged, library, ignore=shutil.ignore_patterns("ORIGINS.json", "MERGED-round-*"))
            copy = Library(library)
            (library / "SINCE.json").write_text(json.dumps({"cases": len(copy.cases()), "ops": len(copy.ops_log())}), encoding="utf-8")
        for neutral in lineages[k - 1]:
            extra = ["--train", "--experience", str(library), *(["--max-experiments", str(args.budget)] if args.budget else [])]
            result = _run(runs / neutral, labs / neutral, targets[neutral], args.suite, args.baselines, agent, backend=args.backend, extra=extra)
            _backfill(library, runs)  # later sessions see this case's counterfactuals
            case = (result or {}).get("submission") or {}
            print(f"round {args.round} lineage {k} {neutral}: {case.get('case', 'no case recorded')}", flush=True)

    with ThreadPoolExecutor(max_workers=args.lineages) as pool:
        list(pool.map(lineage, range(1, args.lineages + 1)))


def cmd_merge(args, agent: list[str]) -> None:
    merged = args.root / "library" / "merged"
    rounds = sorted({1, *(int(path.name.split("-")[1]) for path in (args.root / "library").glob("round-*"))})
    round_ = args.round or rounds[-1]
    lineages = []
    k = 1
    while True:
        library, runs, _ = _round_paths(args.root, round_, k)
        if not library.is_dir():
            break
        print(f"round {round_} lineage {k}: {_backfill(library, runs)} cases backfilled", flush=True)
        lineages.append(library)
        k += 1
    if not lineages:
        raise SystemExit(f"no lineage libraries for round {round_}")
    if round_ == 1:
        if merged.exists():
            raise SystemExit(f"{merged} exists; round 1 was merged already")
        command = [str(VISTA.with_name("vista-cachelab")), "merge-experience", "--out", str(merged)]
        for source in lineages:
            command += ["--from", str(source)]
        subprocess.run(command, check=True, cwd=REPO)
        return
    target = Library(merged)
    marker = merged / f"MERGED-round-{round_}"
    if marker.exists():
        raise SystemExit(f"round {round_} was merged already")
    for k, source in enumerate(lineages, start=1):
        since = json.loads((source / "SINCE.json").read_text(encoding="utf-8"))
        added = absorb(target, Library(source), label=f"round-{round_}-session-{k}", since=since["cases"], since_ops=since.get("ops", 0))
        print(f"round {round_} lineage {k}: added {added}", flush=True)
    marker.write_text("", encoding="utf-8")


def cmd_analyze(args, agent: list[str]) -> None:
    library = args.root / "library" / "merged"
    if not (library / "ORIGINS.json").is_file():
        raise SystemExit(f"{library} has no case origins; run merge first")
    done = [path for path in (args.root / "analyst").glob("round-*") if path.is_dir()] if (args.root / "analyst").is_dir() else []
    number = len(done) + 1
    out, lab = args.root / "analyst" / f"round-{number}", args.root / "labs" / "analyst" / f"round-{number}"
    _wait_for_disk(args.root)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(VISTA), "--profile", "cachelab", "--backend", args.backend, "run",
        "--role", "analyst", "--experience", str(library),
        "--lab-dir", str(lab), "--output", str(out),
        *agent,
        *(["--max-experiments", str(args.budget)] if args.budget else []),
        "--workers", str(args.analyst_workers),
    ]  # fmt: skip
    with open(out.with_suffix(".log"), "w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=False)
    result_path = out / "task" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    print(f"analysis {number}: {result.get('submission') or 'no update recorded'}", flush=True)
    prune_arrays(lab)


def cmd_loop(args, agent: list[str]) -> None:
    first, _, last = args.rounds.partition("..")
    for round_ in range(int(first), int(last or first) + 1):
        args.round = round_
        cmd_train(args, agent)
        cmd_merge(args, agent)
        cmd_analyze(args, agent)


def _assert_disjoint(root: Path, suite: Path, phase: str) -> None:
    """A test workload must never have been trained on: compare its source file and
    name with every other phase's suite and with the cases in the libraries."""
    mine = {spec.name: str(Path(spec.source).resolve()) if spec.source else None for spec in load_suite(suite)}
    names, sources = set(mine), {source for source in mine.values() if source}
    mapping = json.loads((root / "targets.json").read_text(encoding="utf-8")) if (root / "targets.json").is_file() else {}
    clashes = []
    for other, entry in mapping.items():
        if other == phase:
            continue
        for spec in load_suite(Path(entry["suite"])):
            source = str(Path(spec.source).resolve()) if spec.source else None
            if spec.name in names or (source and source in sources):
                clashes.append(f"{spec.name} ({other})")
    for origins in (root / "library").glob("**/ORIGINS.json"):
        for case, origin in json.loads(origins.read_text(encoding="utf-8")).items():
            target = origin.get("target") or {}  # the training workload's spec
            source = str(Path(target["source"]).resolve()) if target.get("source") else None
            if target.get("name") in names or (source and source in sources):
                clashes.append(f"{target.get('name')} ({origins.parent.name} {case})")
    if clashes:
        raise SystemExit(f"{suite} overlaps with training: {', '.join(sorted(set(clashes))[:10])}")


def cmd_test(args, agent: list[str]) -> None:
    _assert_disjoint(args.root, args.suite, "test")
    targets = _targets(args.root, "test", args.suite, args.seed)
    library = args.root / "library" / "merged"
    arms = []
    for item in (part.strip() for part in args.arms.split(",") if part.strip()):
        kind, _, budget = item.partition(":")
        if kind not in ("experience", "plain") or not budget.isdigit() or int(budget) < 1:
            raise SystemExit(f"arm {item!r} is not experience:N or plain:N")
        arms.append((kind, int(budget)))
    if any(kind == "experience" for kind, _ in arms) and not (library / "playbook.md").is_file():
        raise SystemExit(f"no merged library at {library}; run merge first")
    ids = list(targets)[: args.limit or None]
    jobs = [(arm, neutral) for neutral in ids for arm in arms]

    def job(item: tuple[tuple[str, int], str]) -> None:
        (kind, budget), neutral = item
        name = f"{kind}-b{budget}"
        extra = (["--experience", str(library)] if kind == "experience" else []) + ["--max-experiments", str(budget)]
        out = args.root / "test" / name / neutral
        result = _run(out, args.root / "labs" / "test" / name / neutral, targets[neutral], args.suite, args.baselines, agent, backend=args.backend, extra=extra)
        print(f"test {name} {neutral}: score={None if result is None else result.get('score')}", flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(job, jobs))


def _pct(value) -> str:
    return "    n/a" if value is None else f"{100 * value:+7.2f}%"


def _against(root: Path, test: dict[str, str], arms: list[str], rival: str) -> list[str]:
    """Each test arm against one libCacheSim algorithm on the remaining requests.

    The rival runs cold over the whole trace; the arm's decision takes over a live
    S3-FIFO cache at the split; only misses after the split count. An invalid
    decision (memory bound, failure) is deployed as S3-FIFO continuing."""
    stats: dict[str, dict[str, list]] = {}

    def add(label: str, pairs: list[tuple[float, float]], invalid: bool = False) -> None:
        entry = stats.setdefault(label, {"means": [], "workloads": [], "cells": [], "invalid": []})
        relative = [(mine - theirs) / theirs for mine, theirs in pairs if theirs]
        if not relative:
            return
        entry["means"].append(sum(relative) / len(relative))
        entry["workloads"].append(sum(relative) / len(relative) < 0)
        entry["cells"] += [value < 0 for value in relative]
        entry["invalid"].append(invalid)

    for neutral in test:
        reference = None
        for arm in arms:
            path = root / "test" / arm / neutral / "task" / "result.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            evaluation = result.get("evaluation") or {}
            rows = [row for row in evaluation.get("rows", []) if row.get("published", {}).get(rival)]
            if not rows:
                continue
            reference = reference or rows
            valid = result.get("score") is not None
            add(arm, [(row["candidate"] if valid else row["s3fifo"], row["published"][rival]) for row in rows], invalid=not valid)
        if reference:
            add("s3fifo", [(row["s3fifo"], row["published"][rival]) for row in reference])
            add("prefix-tuned", [(row["prefix_tuned_s3fifo"], row["published"][rival]) for row in reference if row.get("prefix_tuned_s3fifo") is not None])
    lines = [
        f"## Against {rival} (it runs cold over the whole trace; decisions take over S3-FIFO's live cache at the split; "
        "misses after the split only; negative = fewer misses; an invalid decision counts as S3-FIFO continuing)",
        f"{'arm':<14}{'mean vs ' + rival:>18}{'workloads better':>18}{'sizes better':>14}{'invalid':>9}",
    ]
    for label, entry in stats.items():
        if not entry["means"]:
            continue
        lines.append(
            f"{label:<14}{_pct(sum(entry['means']) / len(entry['means'])):>18}{sum(entry['workloads']):>10}/{len(entry['workloads']):<7}"
            f"{sum(entry['cells']):>7}/{len(entry['cells']):<6}{sum(entry['invalid']):>9}"
        )
    return lines


def cmd_summarize(args, agent: list[str]) -> None:
    mapping = json.loads((args.root / "targets.json").read_text(encoding="utf-8"))
    lines = []
    for lineage in sorted((args.root / "train").glob("lineage-*")):
        lines += [f"## Training {lineage.name}", "run   case      decision_vs_s3fifo  prefix_tuned_vs_s3fifo  best_published_vs_s3fifo  decision"]
        for out in sorted(path for path in lineage.iterdir() if path.is_dir() and "." not in path.name):
            result_path = out / "task" / "result.json"
            if not result_path.is_file():
                lines.append(f"{out.name}   (no result)")
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            feedback = result.get("training_feedback") or {}
            rows = feedback.get("rows", [])
            best = [(r["best_published_miss_ratio"] - r["s3fifo"]) / r["s3fifo"] for r in rows if r.get("best_published_miss_ratio") is not None and r["s3fifo"]]
            submission = result.get("submission") or {}
            lines.append(
                f"{out.name}   {submission.get('case', '-'):<9} {_pct(feedback.get('mean_relative_vs_s3fifo'))}            "
                f"{_pct(feedback.get('mean_relative_prefix_tuned_vs_s3fifo'))}                 {_pct(sum(best) / len(best) if best else None)}                   "
                f"{(submission.get('experiment') or '-')}: {(submission.get('mechanism') or '')[:80]}"
            )
        lines.append("")
    test = mapping.get("test", {}).get("targets", {})
    if test:
        arms = sorted(path.name for path in (args.root / "test").iterdir() if path.is_dir())
        lines += ["## Test (mean relative miss-ratio change vs S3-FIFO on the remaining requests; negative is better)"]
        lines.append("run   " + "".join(f"{arm:>14}" for arm in arms) + "   prefix_tuned  best_published  best_name")
        totals: dict[str, list[float]] = {arm: [] for arm in arms}
        invalid: dict[str, int] = {arm: 0 for arm in arms}
        for neutral, name in test.items():
            cells, reference = [], None
            for arm in arms:
                path = args.root / "test" / arm / neutral / "task" / "result.json"
                result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                score = result.get("score")
                if score is not None:
                    totals[arm].append(score)
                    cells.append(f"{_pct(score):>14}")
                elif result:
                    # Finished but invalid (memory bound, failure): deployed as S3-FIFO continuing.
                    totals[arm].append(0.0)
                    invalid[arm] += 1
                    cells.append(f"{'invalid':>14}")
                else:
                    cells.append(f"{_pct(None):>14}")
                reference = reference or result.get("evaluation")
            tuned = best = best_name = None
            if reference and reference.get("rows"):
                rows = reference["rows"]
                tuned = reference.get("mean_relative_prefix_tuned_vs_s3fifo")
                values = [(r["best_published_miss_ratio"] - r["s3fifo"]) / r["s3fifo"] for r in rows if r.get("best_published_miss_ratio") is not None and r["s3fifo"]]
                best = sum(values) / len(values) if values else None
                best_name = "/".join(dict.fromkeys(str(r.get("best_published")) for r in rows))
            lines.append(f"{neutral}  " + "".join(cells) + f"   {_pct(tuned)}      {_pct(best)}       {best_name or '-'}")
        lines.append(
            "mean  " + "".join(f"{_pct(sum(v) / len(v) if v else None):>14}" for v in totals.values())
            + f"   (n = {', '.join(f'{a}: {len(v)}' for a, v in totals.items())}; invalid runs count as 0: {', '.join(f'{a}: {invalid[a]}' for a in arms)})"
        )
        lines += ["", *_against(args.root, test, arms, "3L-Cache")]
    text = "\n".join(lines)
    print(text)
    (args.root / "summary.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    argv = sys.argv[1:]
    agent: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, agent = argv[:split], argv[split + 1 :]
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("command", choices=("train", "merge", "analyze", "loop", "test", "summarize"))
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--suite", type=Path, help="train/test: the suite of targets")
    cli.add_argument("--baselines", type=Path, help="train/test: engineer-baselines reports for the suite")
    cli.add_argument("--lineages", type=int, default=2, help="train: parallel training sessions")
    cli.add_argument("--arms", default=ARMS, help="test: comma-separated arm:budget, arm in experience|plain")
    cli.add_argument("--budget", type=int, default=0, help="train: experiments per training workload (overrides the agent arguments)")
    cli.add_argument("--parallel", type=int, default=2, help="test: concurrent runs")
    cli.add_argument("--limit", type=int, default=0, help="train: targets per lineage; test: targets (0 = all)")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--round", type=int, default=0, help="train/merge: the round (train default 1; merge default the latest)")
    cli.add_argument("--rounds", default="2..3", help="loop: FIRST..LAST")
    cli.add_argument("--per-round", type=int, default=1, help="train rounds >= 2: targets per lineage per round")
    cli.add_argument("--allow-repeats", action="store_true", help="train rounds >= 2: wrap around to workloads already trained on")
    cli.add_argument("--analyst-workers", type=int, default=16)
    cli.add_argument("--backend", default="codex")
    args = cli.parse_args(argv)
    if args.command == "train" and not args.round:
        args.round = 1
    if args.command in ("train", "test", "loop") and (args.suite is None or args.baselines is None):
        cli.error(f"{args.command} needs --suite and --baselines")
    args.root = args.root.resolve()
    if args.suite is not None:
        args.suite = args.suite.resolve()
    if args.baselines is not None:
        args.baselines = args.baselines.resolve()
    commands = {"train": cmd_train, "merge": cmd_merge, "analyze": cmd_analyze, "loop": cmd_loop, "test": cmd_test, "summarize": cmd_summarize}
    commands[args.command](args, agent)


if __name__ == "__main__":
    main()
