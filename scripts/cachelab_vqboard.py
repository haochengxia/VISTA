"""Host side of the architect workflow (no agent): seed a board, show it, and run the final test.

  seed    put designs (JSON lines {name, family, config}, e.g. from a vqsim sweep) on a board,
          one board family per `family` value, authored "seed"
  status  the board: families, contributions, the pool's result
  final   evaluate a pool of board families on another suite (held-out cells), once, on a
          separate board directory so held-out runs never mix with the training board
          (online objective: the greedy portfolio chosen on the training board, its idealized
          online choice on the held-out cells)
  import-runs    put existing full vqsim runs on a board (online objective) instead of rerunning
          them: per cell, <trace dir>/<tag>-<capacity>.out with one line per design of --designs
  copy-findings  copy another board's findings (as text, marked with that board's name)

Usage:
  cachelab_vqboard.py seed   --board B --suite S --vqsim V --designs designs.jsonl [--families a,b] [--objective online]
  cachelab_vqboard.py status --board B --suite S --vqsim V [--objective online]
  cachelab_vqboard.py final  --board B --suite S --vqsim V --test-board T --test-suite H [--families 3,7] [--seeds] [--objective online]
  cachelab_vqboard.py import-runs   --board B --suite S --vqsim V --designs designs.jsonl --tag r1
  cachelab_vqboard.py copy-findings --board B --suite S --vqsim V --from-board B0
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from vista.benchmarks.cachelab.architect import (
    PORTFOLIO,
    S3_CONFIG,
    Board,
    board_report,
    by_size,
    summarize,
    summary_line,
)


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("command", choices=("seed", "status", "final", "import-runs", "copy-findings"))
    cli.add_argument("--board", type=Path, required=True)
    cli.add_argument("--suite", type=Path, required=True)
    cli.add_argument("--vqsim", type=Path, required=True)
    cli.add_argument("--slots", type=int, default=24)
    cli.add_argument("--designs", type=Path)
    cli.add_argument("--families", help="seed: family values to take; final: board family ids (default: all evaluated)")
    cli.add_argument("--test-board", type=Path)
    cli.add_argument("--test-suite", type=Path)
    cli.add_argument("--seeds", action="store_true", help="final: include the seed families (default: agents' families only, plus S3-FIFO)")
    cli.add_argument("--objective", choices=("static", "online"), default="static")
    cli.add_argument("--tag", help="import-runs: the runs' file tag")
    cli.add_argument("--from-board", type=Path, help="copy-findings: the board to copy from")
    args = cli.parse_args()
    board = Board(args.board, args.suite, args.vqsim, slots=args.slots, objective=args.objective)
    if args.command == "import-runs":
        configs = [json.loads(line)["config"] for line in args.designs.read_text().splitlines()]
        rows, missing = [], 0
        for cell in board.cells:
            path = cell.directory / f"{args.tag}-{cell.capacity}.out"
            lines = path.read_text().splitlines() if path.is_file() else []
            if len(lines) != len(configs) or any(len(line.split()) != 42 for line in lines):
                missing += 1
                continue
            rows.extend((cell.id, "full", config, line) for config, line in zip(configs, lines, strict=True))
        with board.db() as db:
            db.executemany("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)", rows)
        print(f"imported {len(rows)} runs ({len(configs)} designs); {missing} cells without complete runs")
        return 0
    if args.command == "copy-findings":
        source = Board(args.from_board, args.suite, args.vqsim, slots=args.slots)
        for finding in source.findings():
            text = f"[{args.from_board.name}, families {', '.join(f'F{f}' for f in finding['families'])} there] {finding['text']}"
            board.add_finding(finding["author"], text, [])
        print(f"copied {len(source.findings())} findings")
        return 0
    if args.command == "seed":
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for line in args.designs.read_text().splitlines():
            design = json.loads(line)
            groups[design["family"]].append((design["name"], design["config"]))
        wanted = args.families.split(",") if args.families else list(groups)
        for name in wanted:
            stats = board.evaluate_designs("seed", f"seed:{name}", None, groups[name], f"host sweep: {len(groups[name])} designs of family {name}")
            print(f"F{stats['id']} seed:{name} ({stats['designs']} designs): {summary_line(stats['pick'])}", flush=True)
            print(f"  pool after: {summary_line(stats['pool_after'])}", flush=True)
        return 0
    if args.command == "status":
        print(board_report(board, 200))
        return 0
    if args.test_board is None or args.test_suite is None:
        cli.error("final needs --test-board and --test-suite")
    families = [f for f in board.families() if f["stats"] is not None]
    if args.families:
        ids = {int(x) for x in args.families.split(",")}
        families = [f for f in families if f["id"] in ids]
    elif not args.seeds:
        families = [f for f in families if f["author"] != "seed"]
    configs = list(dict.fromkeys([S3_CONFIG, *(config for f in families for _, config in f["designs"])]))
    test = Board(args.test_board, args.test_suite, args.vqsim, slots=args.slots, objective=args.objective)
    if args.objective == "online":
        portfolio, train, _ = board.online(configs, k=PORTFOLIO)
        _, changes, _ = test.online(portfolio)
        stats = summarize(test.cells, changes)
        print(f"Final portfolio: {len(portfolio)} of {len(configs)} designs ({len(families)} families), chosen on {len(board.cells)} training cells")
        print(f"  training: {summary_line(summarize(board.cells, train))}")
        print(f"  held-out (idealized online choice, {len(test.cells)} cells): {summary_line(stats)}")
        print(f"  3L-Cache: mean {100 * stats['l3_mean']:+.2f}%, median {100 * stats['l3_median']:+.2f}%")
        print("\n".join(by_size(test.cells, changes)))
        (args.test_board / "portfolio.json").write_text(json.dumps(portfolio, indent=1))
        return 0
    # The same rule as in training: per cell, the design with the fewest misses on the observed part.
    chosen = test.choose(configs)
    changes = test.score(chosen)
    stats = summarize(test.cells, changes)
    print(f"Final pool: {len(families)} families, {len(configs)} designs; {len(test.cells)} held-out cells")
    print(f"  pool: {summary_line(stats)}")
    print(f"  3L-Cache: mean {100 * stats['l3_mean']:+.2f}%, median {100 * stats['l3_median']:+.2f}%")
    best = statistics.mean(cell.change(cell.best) for cell in test.cells)
    print(f"  best published per cell (hindsight): mean {100 * best:+.2f}%")
    print("\n".join(by_size(test.cells, changes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
