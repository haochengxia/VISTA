"""Does a TinyLFU-style admission gate (vqsim `gate=`) help FIFO designs where 3L-Cache beats v1? Train only.
Per train-long cell, handover runs (S3-FIFO serves the prefix) of S3-FIFO with the gate (rescue, filter or both;
small-queue sizes) and of v1's first 8 designs with and without each gate mode in front of their main stage.
A gated variant the engine rejects (its rescue moves would cycle) is left out of that design's row.
Scored on windows 8-39 vs S3-FIFO; "3L leads" = cells where 3L-Cache beats v1 (prbcv) by > 0.5 pt.
  vq_gate.py SUITE_DIR VQSIM WORKERS  (needs a vqsim built with the gate; prints aggregates only)"""
import json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

root, vqsim, workers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
port = json.load(open("/mydata/vq/portfolio-r1bp.prereg-1045.json"))["portfolio"]
S3 = port[0]
def gated(c, **kw):
    k = len(re.findall(r"(?:^| )st=", c))
    main = int(m.group(1)) if (m := re.search(r"(?:^| )amain=(\d+)", c)) else k - 1
    entry = int(m.group(1)) if (m := re.search(r"(?:^| )entry=(\d+)", c)) else 0
    return None if main == entry else c + f" gate={main}" + "".join(f" {a}={b}" for a, b in kw.items())
MODES = (("rescue", 0), ("filter", 1), ("both", 2))
S3_3, S3_1 = S3.replace("st=0.1,", "st=0.03,"), S3.replace("st=0.1,", "st=0.01,")
DESIGNS = [("S3-FIFO", S3)] + [(f"S3 {m}", gated(S3, gmode=g)) for m, g in MODES]
DESIGNS += [("S3 small=0.03 filter", gated(S3_3, gmode=1)), ("S3 small=0.03 both", gated(S3_3, gmode=2)), ("S3 small=0.01 both", gated(S3_1, gmode=2))]
for i, c in enumerate(port[1:8], 1):
    DESIGNS.append((f"v1#{i}", c))
    DESIGNS += [(f"v1#{i} {m}", g) for m, gm in MODES if (g := gated(c, gmode=gm)) is not None]

def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text())
    jobs = d / f"gate2-{cap}.jobs"; out = d / f"gate2-{cap}.out"
    jobs.write_text("".join(f"{cap} {meta['split']} 40 {c} handover=1\n" for _, c in DESIGNS))
    if not (out.is_file() and len(out.read_text().splitlines()) == len(DESIGNS)):
        out.write_text(subprocess.run(["nice", "-n", "10", vqsim, str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout)
    lines = out.read_text().splitlines()
    assert not lines[0].startswith("error"), (d.name, cap)
    X = np.array([np.nan if l.startswith("error") else sum(int(x) for x in l.split()[2 + 8:]) for l in lines], float)
    v1 = (sum(int(x) for x in (d / f"prbcv-{cap}.out").read_text().splitlines()[0].split()[3 + 8:]) - X[0]) / X[0]
    s3 = meta["s3fifo"][str(cap)]
    return 100 * cap / meta["objects"], (X - X[0]) / X[0], v1, (meta["3L"][str(cap)] - s3) / s3

cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool:
    got = list(pool.map(cell, cells))
Z = np.array([g[0] for g in got]); R = np.array([g[1] for g in got]); V1 = np.array([g[2] for g in got]); L3 = np.array([g[3] for g in got])
lead = V1 - L3 > 0.005
names = [n for n, _ in DESIGNS]
bands = [(0, 0.5, "<0.5%"), (0.5, 2, "0.5-2%"), (2, 8, "2-8%"), (8, 1e9, ">=8%")]
print(f"{root.name}: {len(cells)} cells, handover runs, windows 8-39, change vs S3-FIFO; 3L leads v1 in {lead.sum()} cells")
print(f"  {'design':<22}{'all':>8}" + "".join(f"{b[2]:>9}" for b in bands) + f"{'3L leads':>10}")
for j, n in enumerate(names):
    if np.isnan(R[:, j]).all():
        print(f"  {n:<22} rejected by the engine"); continue
    print(f"  {n:<22}{100*np.nanmean(R[:, j]):+7.2f}%" + "".join(f"{100*np.nanmean(R[(Z >= lo) & (Z < hi), j]):+8.2f}%" for lo, hi, _ in bands) + f"{100*np.nanmean(R[lead, j]):+9.2f}%")
gate = [j for j, n in enumerate(names) if n.endswith(tuple(m for m, _ in MODES))]; plain = [j for j in range(len(names)) if j not in gate]
R = np.where(np.isnan(R), np.inf, R)
bp, bg = R[:, plain].min(1), R.min(1)
for m, _ in MODES:
    bm = R[:, plain + [j for j in gate if names[j].endswith(m)]].min(1)
    print(f"  best per cell (hindsight) with {m} gates only: {100*bm.mean():+.2f}%; in the 3L-leads cells {100*bm[lead].mean():+.2f}%")
print(f"  best per cell (hindsight), without gates {100*bp.mean():+.2f}%, with gates {100*bg.mean():+.2f}%; in the 3L-leads cells {100*bp[lead].mean():+.2f}% -> {100*bg[lead].mean():+.2f}% (v1 {100*V1[lead].mean():+.2f}%, 3L {100*L3[lead].mean():+.2f}%)")
print(f"  a gated design is the per-cell best in {int((R[:, gate].min(1) < R[:, plain].min(1) - 1e-9).sum())} of {len(cells)} cells")
