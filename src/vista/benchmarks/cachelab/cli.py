"""CacheLab: improve S3-FIFO with a VISTA agent, by hand, or by parameter search.

    suite        materialize the development traces and print what the agent sees
    build-suite  sample real oracleGeneral traces into a suite file
    try        simulate a policy file on the dev suite and write its views
    evaluate   verified held-out evaluation of a policy file (the agent's score)
    search     parameter-search arm under an experiment budget
    engineer-baselines  S3-FIFO, prefix-tuned S3-FIFO and libCacheSim algorithms on every
               target of --target-suite (the engineer role's comparison, no agent)
    plan, run  agent runs: --arm vista|numeric, --role designer|engineer, --shots
    merge-experience  combine experience libraries (--from DIR, repeated) into --out

Roles. A designer studies the development traces and is scored on held-out
traces. An engineer studies the first --split-fraction of one workload
(--target-suite with --target, or --target-trace) and is scored on the rest.

Experience (engineer role). --experience DIR gives the agent a library of
earlier cases and a playbook. With --train the run is a training run: the
agent commits a decision, sees how it did on the rest of the workload (fast
simulation, using --baselines reports from engineer-baselines), and adds a
case and a revised playbook to DIR. Without --train the library is read only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import DEFAULT_IMAGE, DockerSandbox, LocalSandbox
from .lab import BASELINE, Lab, LabError
from .profiles import DEFAULT_EXPERIMENTS, PROFILES


def parser() -> argparse.ArgumentParser:
    from vista.backends.launch import add_backend_arguments

    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root.add_argument(
        "command", choices=("suite", "build-suite", "try", "evaluate", "search", "engineer-baselines", "merge-experience", "plan", "run")
    )
    root.add_argument("policy", nargs="?", type=Path, help="policy file for try/evaluate")
    root.add_argument("--lab-dir", type=Path, default=Path("runs/cachelab-lab"), help="trace and simulation cache")
    root.add_argument("--sandbox", choices=("docker", "local"), default="docker")
    root.add_argument("--image", default=DEFAULT_IMAGE, help="sandbox image for docker")
    root.add_argument("--workers", type=int)
    root.add_argument(
        "--unsafe-local-sandbox",
        action="store_true",
        help="allow agent runs to execute agent-written code on this host (no isolation)",
    )
    root.add_argument("--out", type=Path, help="try: directory for views; search/evaluate/build-suite: output file")
    root.add_argument("--dev-suite", type=Path, help="suite file for the development split")
    root.add_argument("--heldout-suite", type=Path, help="suite file for the held-out split")
    root.add_argument("--data-root", type=Path, help="build-suite: directory of datasets of oracleGeneral traces")
    root.add_argument("--datasets", help="build-suite: comma-separated dataset directories")
    root.add_argument("--per-dataset", type=int, default=4)
    root.add_argument("--requests", type=int, default=1_000_000, help="build-suite: prefix length per trace")
    root.add_argument("--split", choices=("dev", "heldout"), default="heldout")
    root.add_argument("--exclude-suite", type=Path, action="append", default=[], help="build-suite: suites whose files to skip")
    root.add_argument("--max-mb", type=int, help="build-suite: prefer files at most this size")
    root.add_argument(
        "--view",
        action="append",
        default=[],
        help="try: extra views as kind:trace:capacity[:start:end], kind in queues|objects|diff",
    )
    root.add_argument("--budget", type=int, default=DEFAULT_EXPERIMENTS, help="search: experiments")
    root.add_argument("--seed", type=int, default=0)
    add_backend_arguments(root)
    root.add_argument("--arm", choices=PROFILES, default="vista", help="plan/run: vista (images) or numeric")
    root.add_argument("--model")
    root.add_argument("--effort")
    root.add_argument("--max-experiments", type=int, default=DEFAULT_EXPERIMENTS)
    root.add_argument("--role", choices=("designer", "engineer", "analyst"), default="designer")
    root.add_argument("--shots", action="store_true", help="reference policies plus worked examples from the literature")
    root.add_argument("--target-suite", type=Path, help="engineer: suite file holding the target workload")
    root.add_argument("--target", help="engineer: name of the target in --target-suite")
    root.add_argument("--target-trace", type=Path, help="engineer: an oracleGeneral file as the target workload")
    root.add_argument("--target-requests", type=int, default=1_000_000, help="engineer: requests of --target-trace to use")
    root.add_argument("--split-fraction", type=float, default=0.2, help="engineer: share of the workload the agent sees")
    root.add_argument("--shard", default="0/1", help="engineer-baselines: process targets k, k+n, ... given k/n")
    root.add_argument("--experience", type=Path, help="engineer: experience library directory (written with --train)")
    root.add_argument("--train", action="store_true", help="engineer: training run; feedback after commit_decision, adds a case")
    root.add_argument("--baselines", type=Path, help="engineer: engineer-baselines report, or a directory of them named <target>.json")
    root.add_argument("--from", dest="sources", type=Path, action="append", default=[], help="merge-experience: a library to merge")
    root.add_argument("--output", type=Path, help="run: new directory for the run")
    root.add_argument("--backend-segments", type=int, default=60)
    root.add_argument("--tool-calls", type=int, default=600)
    return root


SHOT_REFERENCES = ("s4fifo", "mq", "lru", "fifo")


def make_sandbox(args: argparse.Namespace):
    if args.sandbox == "docker":
        sandbox = DockerSandbox(image=args.image, workers=args.workers)
        sandbox.image_id  # fail early if the image is missing
        return sandbox
    return LocalSandbox(workers=args.workers)


def make_lab(args: argparse.Namespace, references: tuple[str, ...] = ()) -> Lab:
    from .traces import DEV_SUITE, HELDOUT_SUITE, load_suite

    dev = load_suite(args.dev_suite) if args.dev_suite else DEV_SUITE
    heldout = load_suite(args.heldout_suite) if args.heldout_suite else HELDOUT_SUITE
    return Lab(args.lab_dir, make_sandbox(args), dev=dev, heldout=heldout, workers=args.workers or 1, references=references)


def target_spec(args: argparse.Namespace, cli):
    from .traces import TraceSpec, load_suite

    if args.target_trace is not None:
        if not args.target_trace.is_file():
            cli.error("--target-trace must be an existing oracleGeneral file")
        return TraceSpec(args.target_trace.stem, "heldout", "oracleGeneral", requests=args.target_requests, seed=1, source=str(args.target_trace.resolve()))
    if args.target_suite is None or args.target is None:
        cli.error("the engineer role needs --target-suite with --target, or --target-trace")
    specs = {spec.name: spec for spec in load_suite(args.target_suite)}
    if args.target not in specs:
        cli.error(f"--target must be one of: {', '.join(specs)}")
    return specs[args.target]


def load_baselines(args: argparse.Namespace, spec, cli) -> dict | None:
    if args.baselines is None:
        return None
    path = args.baselines / f"{spec.name}.json" if args.baselines.is_dir() else args.baselines
    if not path.is_file():
        cli.error(f"no baselines report at {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("target", {}).get("name", spec.name) != spec.name:
        cli.error(f"{path} is the baselines report of another target")
    return report


def make_engineer(args: argparse.Namespace, cli, references: tuple[str, ...] = ()):
    """(lab, evaluate, record, train_feedback) for one engineer task under --lab-dir.

    train_feedback is None unless --train: then it scores a decision on the rest
    of the workload with fast simulation, for the agent to learn from."""
    from . import engineer

    spec = target_spec(args, cli)
    task = engineer.prepare(spec, args.lab_dir, fraction=args.split_fraction)
    lab = engineer.make_lab(task, args.lab_dir, make_sandbox(args), references=references, workers=args.workers or 1)
    workers = args.workers or 1
    baselines = load_baselines(args, spec, cli)
    zoo_dir = args.lab_dir / "zoo"

    def verified(lab_: Lab, identity: str) -> dict:
        return engineer.evaluate(lab_, task, identity, zoo_dir=zoo_dir, workers=workers, baselines=baselines)

    feedback: dict[str, dict] = {}

    def fast(identity: str) -> dict:
        # Memoized: the scored arrays are dropped, so a second call would simulate again.
        if identity not in feedback:
            feedback[identity] = engineer.evaluate(
                lab, task, identity, zoo_dir=zoo_dir, workers=workers, baselines=baselines, verified=False, keep_arrays=False
            )
        return feedback[identity]

    record = {"target": spec.manifest(), "split": task.split, "split_fraction": task.fraction, "capacities": list(task.capacities)}
    if args.train:
        # A training run is scored by the feedback it got (cached, so this costs nothing).
        return lab, (lambda lab_, identity: fast(identity)), record, fast
    return lab, verified, record, None


def cmd_merge_experience(args, cli) -> int:
    from .experience import merge

    if not args.sources or args.out is None:
        cli.error("merge-experience needs --from (repeated) and --out")
    missing = [str(path) for path in args.sources if not (path / "cases").is_dir()]
    if missing:
        cli.error(f"not experience libraries: {', '.join(missing)}")
    target = merge(list(args.sources), args.out)
    print(f"merged {len(args.sources)} libraries into {target}")
    return 0


def cmd_engineer_baselines(args, cli) -> int:
    """No agent: every target of a suite against S3-FIFO, prefix tuning and published algorithms."""
    from . import engineer
    from .traces import load_suite

    if args.target_suite is None or args.out is None:
        cli.error("engineer-baselines needs --target-suite and --out")
    args.out.mkdir(parents=True, exist_ok=True)
    shard, shards = (int(part) for part in args.shard.split("/"))
    for position, spec in enumerate(load_suite(args.target_suite)):
        if position % shards != shard:
            continue
        out = args.out / f"{spec.name}.json"
        if out.is_file():
            continue
        root = args.lab_dir / spec.name
        task = engineer.prepare(spec, root, fraction=args.split_fraction)
        lab = engineer.make_lab(task, root, make_sandbox(args), workers=args.workers or 1)
        report = engineer.evaluate(lab, task, BASELINE, zoo_dir=root / "zoo", workers=args.workers or 1)
        report["target"] = spec.manifest()
        _write_json(out, report)
        engineer.prune_arrays(root)  # the report is all later runs need
    return 0


def _write_json(path: Path | None, value) -> None:
    text = json.dumps(value, indent=2, default=str)
    if path is None:
        print(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")


def cmd_try(args, cli) -> int:
    if args.policy is None or not args.policy.is_file():
        cli.error("try needs an existing policy file")
    lab = make_lab(args)
    out = args.out or Path("runs/cachelab-try") / args.policy.stem
    out.mkdir(parents=True, exist_ok=True)
    view = lab.run_experiment(args.policy.stem, args.policy.read_text(encoding="utf-8"), "manual", None)
    print(view.text)
    if not view.data.get("ok"):
        return 1
    identity = view.data["experiment"]
    (out / "overview.png").write_bytes(view.png)
    for request in args.view:
        kind, trace, capacity, *window = request.split(":")
        start, end = (int(window[0]), int(window[1])) if len(window) == 2 else (None, None)
        try:
            if kind == "queues":
                result = lab.queues(identity, trace, int(capacity), start, end)
            elif kind == "objects":
                result = lab.objects(identity, trace, int(capacity), start, end, select="most_regressed", compare_to=BASELINE)
            elif kind == "diff":
                result = lab.diff(identity, BASELINE, trace, int(capacity), start, end)
            else:
                cli.error(f"unknown view kind {kind!r}")
        except LabError as exc:
            cli.error(str(exc))
        name = f"{kind}-{trace}-{capacity}" + (f"-{start}-{end}" if start is not None else "")
        (out / f"{name}.png").write_bytes(result.png)
        (out / f"{name}.txt").write_text(result.text + "\n", encoding="utf-8")
    print(f"views in {out}")
    return 0


def cmd_run(args, cli) -> int:
    from vista.backends.launch import (
        prepare_backend,
        transport_for,
        validate_backend_arguments,
    )

    if not all((args.backend, args.model, args.effort)):
        cli.error("plan/run require --backend, --model and --effort")
    if args.max_experiments < 1:
        cli.error("--max-experiments must be positive")
    if args.sandbox == "local" and args.command == "run" and not args.unsafe_local_sandbox:
        cli.error("agent runs execute agent-written code; use --sandbox docker, or pass --unsafe-local-sandbox")
    validate_backend_arguments(args, cli.error)
    transport = transport_for(args.backend)
    references = SHOT_REFERENCES if args.shots else ()
    if args.role == "designer" and (args.train or args.experience or args.baselines):
        cli.error("--train, --experience and --baselines belong to the engineer and analyst roles")
    if args.role == "analyst" and (args.train or args.baselines or args.experience is None):
        cli.error("the analyst role takes --experience LIB (the library it refines), and no --train or --baselines")
    if args.train and args.experience is None:
        cli.error("--train needs --experience, the library to add the case to")
    if args.train and args.baselines is None:
        cli.error("--train needs --baselines (engineer-baselines reports) for its feedback")
    library = None
    if args.experience is not None:
        from .experience import Library

        if not args.train and not (args.experience / "playbook.md").is_file():
            cli.error(f"{args.experience} holds no playbook; train first or pass --train")
        library = Library(args.experience)
    train_feedback = None
    if args.role == "engineer":
        lab, evaluate, task_record, train_feedback = make_engineer(args, cli, references)
    elif args.role == "analyst":
        from . import analyst

        lab = analyst.make_lab(library, args.lab_dir, make_sandbox(args), workers=args.workers or 1, references=references or analyst.REFERENCES)
        evaluate, task_record = None, {"cases": len(library.cases()), "library": "read-write"}
    else:
        lab, evaluate, task_record = make_lab(args, references), None, {}
    from .profiles import make_experiment, make_profile
    from .session import CacheLabSession

    if args.command == "plan":
        session = CacheLabSession(
            lab,
            visual=args.arm == "vista",
            max_experiments=args.max_experiments,
            role=args.role,
            shots=args.shots,
            library=library,
            train_feedback=train_feedback,
        )
        profile = make_profile(
            profile=args.arm,
            backend=transport,
            max_experiments=args.max_experiments,
            initial=session.initial,
            role=args.role,
            split_fraction=args.split_fraction,
            train=args.train,
            experience=library is not None,
        )
        experiment = make_experiment(
            profile,
            backend=transport,
            model=args.model,
            effort=args.effort,
            revision="plan",
            suite=lab.public_suite(),
            max_experiments=args.max_experiments,
            backend_segments=args.backend_segments,
            tool_calls=args.tool_calls,
        )
        _write_json(
            None,
            {**experiment.manifest(), "initial_prompt": session.initial, "lab_tools": [binding.spec.public_definition() for binding in session.tools]},
        )
        return 0
    if args.output is None:
        cli.error("run requires a new --output directory")
    from .run import run_task

    backend_factory, provenance = prepare_backend(args, cli.error)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = run_task(
        lab,
        directory=args.output / "task",
        profile_name=args.arm,
        backend=transport,
        model=args.model,
        effort=args.effort,
        backend_factory=backend_factory,
        max_experiments=args.max_experiments,
        backend_segments=args.backend_segments,
        tool_calls=args.tool_calls,
        transport_provenance=provenance,
        role=args.role,
        shots=args.shots,
        split_fraction=args.split_fraction,
        evaluate=evaluate,
        task_record=task_record,
        library=library,
        train_feedback=train_feedback,
    )
    summary = {key: result.get(key) for key in ("profile", "submitted", "score", "error", "elapsed_seconds")}
    summary["experiments_run"] = len(result.get("experiments", []))
    print(json.dumps(summary, indent=2))
    return 0 if result.get("submitted") else 1


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "suite":
        lab = make_lab(args)
        print(json.dumps({"dev": lab.public_suite(), "heldout": [spec.manifest() for spec in lab.heldout_specs]}, indent=2))
        print(lab.info(BASELINE).text)
        return 0
    if args.command == "build-suite":
        from .traces import load_suite, sample_real_suite, save_suite

        if args.data_root is None or not args.datasets or args.out is None:
            cli.error("build-suite needs --data-root, --datasets and --out")
        exclude = {spec.source for path in args.exclude_suite for spec in load_suite(path) if spec.source}
        specs = sample_real_suite(
            args.data_root,
            split=args.split,
            datasets=[name.strip() for name in args.datasets.split(",") if name.strip()],
            per_dataset=args.per_dataset,
            requests=args.requests,
            seed=args.seed,
            exclude=exclude,
            max_bytes=None if args.max_mb is None else args.max_mb << 20,
            neutral_names=args.split == "dev",
        )
        save_suite(args.out, specs)
        print(f"wrote {len(specs)} traces to {args.out}")
        return 0
    if args.command == "try":
        return cmd_try(args, cli)
    if args.command == "merge-experience":
        return cmd_merge_experience(args, cli)
    if args.command == "evaluate":
        if args.policy is None or not args.policy.is_file():
            cli.error("evaluate needs an existing policy file")
        lab = make_lab(args)
        view = lab.run_experiment(args.policy.stem, args.policy.read_text(encoding="utf-8"), "manual", None)
        if not view.data.get("ok"):
            print(view.text)
            return 1
        report = lab.final_evaluation(view.data["experiment"])
        _write_json(args.out, report)
        print(f"valid={report['valid']} heldout score={report['score']}", file=sys.stderr)
        return 0
    if args.command == "search":
        from .search import search

        report = search(make_lab(args), budget=args.budget, seed=args.seed)
        _write_json(args.out, report)
        print(f"selected={report.get('selected', {}).get('config')} heldout score={report['score']}", file=sys.stderr)
        return 0
    if args.command == "engineer-baselines":
        return cmd_engineer_baselines(args, cli)
    return cmd_run(args, cli)


if __name__ == "__main__":
    raise SystemExit(main())
