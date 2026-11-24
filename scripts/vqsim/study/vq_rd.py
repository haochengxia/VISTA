"""Does a reuse-distance rule (vqsim `rd=`, `rdmode=`) help FIFO designs where 3L-Cache beats v1? Train only.
Per train-long cell, handover runs (S3-FIFO serves the prefix) of S3-FIFO and of v1's first 8 designs, as they
are and with the rule: a move up (promotion, reinsertion), a ghost admission into main, or a rescue is decided
by the object's predicted wait for its next request against rd times the residence time of the stage it would
enter (rdmode bits 1 promotions, 2 reinsertions, 4 ghost admission, 8 rescue).
Scored on windows 8-39 vs S3-FIFO; "3L leads" = cells where 3L-Cache beats v1 by > 0.5 pt, v1 being the frozen
portfolio's online run in sample (TAG-{cap}.out from vq_portfolio_real.py, K=32 rate=0.005 min=200).
  vq_rd.py SUITE_DIR VQSIM WORKERS V1TAG  (needs a vqsim built with VQ_RD; prints aggregates only)"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

root, vqsim, workers, v1tag = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
port = json.load(open(Path(__file__).parents[1] / "v1/portfolio.json"))["portfolio"]
MODES = (("promote", 1), ("reinsert", 2), ("both", 3), ("ghost", 4), ("all3", 7), ("rescue", 8))
XS = (0.5, 1, 2)
DESIGNS = [("S3-FIFO", port[0])] + [(f"S3 {m} x{x:g}", f"{port[0]} rd={x:g} rdmode={b}") for m, b in MODES for x in XS]
for i, c in enumerate(port[1:8], 1):
    DESIGNS.append((f"v1#{i}", c))
    DESIGNS += [(f"v1#{i} {m} x{x:g}", f"{c} rd={x:g} rdmode={b}") for m, b in MODES if b != 1 for x in XS]
TAG = "rd1"

def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text())
    jobs = d / f"{TAG}-{cap}.jobs"; out = d / f"{TAG}-{cap}.out"
    jobs.write_text("".join(f"{cap} {meta['split']} 40 {c} handover=1\n" for _, c in DESIGNS))
    if not (out.is_file() and len(out.read_text().splitlines()) == len(DESIGNS)):
        out.write_text(subprocess.run(["nice", "-n", "10", vqsim, str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout)
    lines = out.read_text().splitlines()
    assert not any(l.startswith("error") for l in lines), (d.name, cap)
    X = np.array([sum(int(x) for x in l.split()[2 + 8:]) for l in lines], float)
    v1 = (sum(int(x) for x in (d / f"{v1tag}-{cap}.out").read_text().splitlines()[0].split()[3 + 8:]) - X[0]) / X[0]
    s3 = meta["s3fifo"][str(cap)]
    objects = meta.get("objects") or int(np.memmap(d / "ids.u32", dtype=np.uint32, mode="r").max()) + 1
    return d.name, 100 * cap / objects, (X - X[0]) / X[0], v1, (meta["3L"][str(cap)] - s3) / s3

cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool:
    got = list(pool.map(cell, cells))
T = [g[0] for g in got]; Z = np.array([g[1] for g in got]); R = np.array([g[2] for g in got]); V1 = np.array([g[3] for g in got]); L3 = np.array([g[4] for g in got])
lead = V1 - L3 > 0.005
names = [n for n, _ in DESIGNS]
bands = [(0, 0.5, "<0.5%"), (0.5, 2, "0.5-2%"), (2, 8, "2-8%"), (8, 1e9, ">=8%")]
print(f"{root.name}: {len(cells)} cells, handover runs, windows 8-39, change vs S3-FIFO; 3L leads v1 in {lead.sum()} cells")
print(f"  {'design':<24}{'all':>8}{'median':>8}" + "".join(f"{b[2]:>9}" for b in bands) + f"{'3L leads':>10}{'better':>8}{'worse':>7}")
base = {n: j for j, n in enumerate(names) if " " not in n or n == "S3-FIFO"}
for j, n in enumerate(names):
    b = base[n.split(" ")[0] if n.startswith("v1#") else "S3-FIFO"]
    d = R[:, j] - R[:, b]
    print(f"  {n:<24}{100*R[:, j].mean():+7.2f}%{100*np.median(R[:, j]):+7.2f}%" + "".join(f"{100*R[(Z >= lo) & (Z < hi), j].mean():+8.2f}%" for lo, hi, _ in bands)
          + f"{100*R[lead, j].mean():+9.2f}%{(d < -0.001).sum():8d}{(d > 0.001).sum():7d}")
rd = [j for j, n in enumerate(names) if " x" in n]; plain = [j for j in range(len(names)) if j not in rd]
bp, bg = R[:, plain].min(1), R.min(1)
for m, _ in MODES:
    bm = R[:, plain + [j for j in rd if f" {m} x" in names[j]]].min(1)
    print(f"  best per cell (hindsight) with {m} only: {100*bm.mean():+.2f}%; in the 3L-leads cells {100*bm[lead].mean():+.2f}%")
print(f"  best per cell (hindsight), without rd {100*bp.mean():+.2f}%, with rd {100*bg.mean():+.2f}%; in the 3L-leads cells {100*bp[lead].mean():+.2f}% -> {100*bg[lead].mean():+.2f}% (v1 {100*V1[lead].mean():+.2f}%, 3L {100*L3[lead].mean():+.2f}%)")
print(f"  an rd design is the per-cell best in {int((R[:, rd].min(1) < R[:, plain].min(1) - 1e-9).sum())} of {len(cells)} cells")
traces = sorted(set(T)); per = lambda v: np.array([np.mean([v[i] for i in range(len(T)) if T[i] == t]) for t in traces])
print(f"  traces where the per-cell best beats 3L-Cache: without rd {(per(bp) < per(L3)).sum()}, with rd {(per(bg) < per(L3)).sum()} of {len(traces)} (v1 in sample {(per(V1) < per(L3)).sum()})")
