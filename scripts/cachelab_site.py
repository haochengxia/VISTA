"""Build site/data.js for the CacheLab v1 project page from the committed v1 logs.

Every number the page shows comes from scripts/vqsim/v1/logs/*.txt, except a few
that were measured on the node and are recorded in docs/cachelab-v1.md (marked
DOC below). Values are changes in misses vs S3-FIFO, in percent (negative = fewer).

    uv run python scripts/cachelab_site.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "scripts" / "vqsim" / "v1" / "logs"
OUT = ROOT / "site" / "data.js"

ROW = re.compile(
    r"^\s*(?P<label>.+?)\s+mean\s+(?P<mean>[-+]?\d+\.\d+)%\s+median\s+(?P<median>[-+]?\d+\.\d+)%"
    r"\s+beats 3L (?P<beats>\d+)/(?P<cells>\d+)(?:\s+worse(?: than S3-FIFO)? (?P<worse>\d+))?"
)
SIZE = re.compile(
    r"^\s*size (?P<size>\S+)\s+\(\s*(?P<cells>\d+) cells\): 3L\s+(?P<l3>[-+]?\d+\.\d+)%\s+K=32:\s+(?P<v1>[-+]?\d+\.\d+)%"
)
CPU = re.compile(
    r"C=(?P<c>\d+): libCacheSim S3-FIFO (?P<s3a>[\d.]+)s, 3L-Cache (?P<l3>[\d.]+)s \((?P<l3x>[\d.]+)x\)"
    r" \| vqsim S3-FIFO (?P<s3b>[\d.]+)s, v1 (?P<v1>[\d.]+)s \((?P<v1x>[\d.]+)x\)"
)
SIZE_NAMES = {"<0.5%": "<0.5%", "0.5-2%": "0.5–2%", "2-8%": "2–8%", ">=8%": "≥8%"}


def rows(name: str, *, block: int = 0) -> dict[str, dict]:
    """Summary rows of a log, by label. A log may hold several blocks, each starting with its header."""
    found: list[dict[str, dict]] = []
    for line in (LOGS / name).read_text().splitlines():
        if line and not line.startswith(" "):
            found.append({})
            continue
        if m := ROW.match(line):
            row = {
                "mean": float(m["mean"]),
                "median": float(m["median"]),
                "beats": int(m["beats"]),
                "cells": int(m["cells"]),
            }
            if m["worse"] is not None:
                row["worse"] = int(m["worse"])
            found[-1][m["label"].strip()] = row
    return found[block]


def sizes(name: str) -> list[dict]:
    out = []
    for line in (LOGS / name).read_text().splitlines():
        if m := SIZE.match(line):
            out.append(
                {
                    "size": SIZE_NAMES[m["size"]],
                    "cells": int(m["cells"]),
                    "3L": float(m["l3"]),
                    "v1": float(m["v1"]),
                }
            )
    return out


def need(table: dict, label: str) -> dict:
    if label not in table:
        raise SystemExit(f"missing row {label!r}; have {sorted(table)}")
    return table[label]


def build() -> dict:
    v1_ho = rows("portfolio-r1bp-heldout.txt")
    seeds32_ho = rows("portfolio-heldout.txt")
    sets_ho = rows("online-final-heldout.txt")
    ideal_ho = rows("online-heldout.txt")
    k32 = "K=32 rate=0.005 min=200"

    heldout = [
        {"name": "3L-Cache", "role": "rival", **need(v1_ho, "3L-Cache"), "memory": "–"},
        {
            "name": "v1 (32 designs)",
            "role": "ours",
            **need(v1_ho, k32),
            "memory": "0.16·C entries (median)",
        },
        {
            "name": "32 seed designs",
            "role": "variant",
            **need(seeds32_ho, k32),
            "memory": "0.16·C",
        },
        {
            "name": "84 board picks",
            "role": "variant",
            **need(sets_ho, "B online"),
            "memory": "over 2×",
        },
        {
            "name": "all 426 seeds",
            "role": "variant",
            **need(sets_ho, "A online"),
            "memory": "over 2×",
        },
        {
            "name": "idealized bound",
            "role": "ideal",
            **need(ideal_ho, "leader all, cumulative"),
            "memory": "–",
        },
    ]

    v1_train = rows("portfolio-r1bp-real-train.txt")
    v1_cv = rows("portfolio-r1bp-cvreal-train.txt")
    train = [
        {"name": "3L-Cache", **need(v1_train, "3L-Cache")},
        {"name": "v1, in-sample", **need(v1_train, k32)},
        {"name": "v1, 5-fold CV by trace", **need(v1_cv, k32)},
    ]

    cv = rows("portfolio-train.txt", block=0)
    portfolio = [
        {"k": int(label[2:]), "mean": row["mean"], "median": row["median"]}
        for label, row in cv.items()
        if label.startswith("K=")
    ]

    switch = rows("switch-train.txt")
    takeover = [
        {
            "policy": policy,
            "split": need(switch, f"split {key}")["mean"],
            "start": need(switch, f"start {key}")["mean"],
        }
        for policy, key in [
            ("static pick", "static"),
            ("online, no margin", "online m=0%"),
            ("online, 1% margin", "online m=1%"),
            ("online, 3% margin", "online m=3%"),
        ]
    ]

    online_static = [
        {
            "set": "426 seeds",
            "static": need(sets_ho, "A static")["mean"],
            "online": need(sets_ho, "A online")["mean"],
        },
        {
            "set": "84 board picks",
            "static": need(sets_ho, "B static")["mean"],
            "online": need(sets_ho, "B online")["mean"],
        },
    ]

    choose = rows("eval-r1.txt")
    real = rows("online-real.txt")
    choosing = [
        {"name": "pick one design from the first 20%", **need(choose, "all pick")},
        {
            "name": "re-choose online, live cache",
            **need(real, "mini-all r=1% min=1000"),
        },
        {
            "name": "re-choose online, idealized",
            **need(real, "idealized leader (cold)"),
        },
        {
            "name": "best design per workload, hindsight",
            **need(choose, "all hindsight"),
        },
    ]

    cpu = []
    for line in (LOGS / "cpu-compare.txt").read_text().splitlines():
        if m := CPU.search(line):
            cpu.append(
                {"capacity": int(m["c"]), "3L": float(m["l3x"]), "v1": float(m["v1x"])}
            )

    return {
        "heldout": heldout,
        "heldoutCells": heldout[0]["cells"],
        "bySize": {
            "heldout": sizes("portfolio-r1bp-heldout.txt"),
            "train": sizes("portfolio-r1bp-cvreal-train.txt"),
        },
        "train": train,
        "portfolio": portfolio,
        "portfolioAll": need(cv, "all designs")["mean"],
        "train3L": need(cv, "3L-Cache")["mean"],
        "takeover": takeover,
        "onlineStatic": online_static,
        "choosing": choosing,
        "cpu": cpu,
        # DOC: docs/cachelab-v1.md, "Verdict against the open-source bar".
        "tracesWon": {"won": 12, "traces": 22, "bar": 15},
        # DOC: finding 7 (best fixed design per size class; ghost size in multiples of C).
        "ghost": {
            "depths": [0.9, 2, 4, 8, 16],
            "classes": [
                {"size": "<0.5%", "change": [-0.16, 1.25, 2.25, 2.69, 3.07]},
                {"size": "0.5–2%", "change": [0.00, 0.67, 2.01, 3.24, 4.00]},
                {"size": "2–8%", "change": [0.00, 1.04, 1.96, 2.34, 2.52]},
                {"size": "≥8%", "change": [-1.65, 1.14, 1.44, 1.53, 1.53]},
            ],
        },
        # DOC: finding 10 and roadmap item 7 (pool score, static pick under handover, train-long).
        "boards": [
            {"stage": "seeds only", "mean": -3.50},
            {"stage": "gen 1", "mean": -3.79},
            {"stage": "gen 2", "mean": -4.03},
            {"stage": "gen 3", "mean": -4.05},
        ],
        # DOC: "Memory" (tracemalloc, OnlineV1 vs S3-FIFO).
        "memory": [
            {"capacity": 50_000, "ratio": 1.61},
            {"capacity": 5_000, "ratio": 3.20},
        ],
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        "// Generated by scripts/cachelab_site.py from scripts/vqsim/v1/logs. Do not edit.\n"
        f"window.CACHELAB = {json.dumps(data, indent=1, ensure_ascii=False)};\n"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
