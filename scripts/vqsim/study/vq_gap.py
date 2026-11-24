"""Where does 3L-Cache still beat v1? Train only (no held-out data).
Per train-long cell: v1's realistic, cross-validated result (outputs tagged `prbcv`, as in
portfolio-r1bp-cvreal-train.txt), 3L-Cache, Belady (OPT) and the best other published policy from the
engineer baselines, all as the change in misses vs S3-FIFO on the scored part. Prints aggregates only;
traces are anonymous.
  vq_gap.py /mydata/vq/train-long ~/VISTA-xp/results/engineer/train-long"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

root, reports = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for d in sorted(root.iterdir()):
    if not (d / "meta.json").is_file():
        continue
    meta = json.loads((d / "meta.json").read_text())
    rep = {r["capacity"]: r for r in json.loads((reports / f"{d.name}.json").read_text())["rows"]}
    for cap in meta["capacities"]:
        W0 = [int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]
        base = sum(W0[8:])
        line = (d / f"prbcv-{cap}.out").read_text().splitlines()[0].split()
        v1 = (sum(int(x) for x in line[3 + 8:]) - base) / base
        s3 = meta["s3fifo"][str(cap)]
        r = rep[cap]
        others = {k: v for k, v in r["published"].items() if k != "3L-Cache" and "belady" not in k.lower()}  # online policies only
        best_other = min(others, key=others.get)
        rows.append({
            "trace": d.name, "size": 100 * cap / meta["objects"], "v1": v1,
            "3L": (meta["3L"][str(cap)] - s3) / s3, "opt": (r["belady"] - s3) / s3,
            "other": (others[best_other] - s3) / s3, "other_name": best_other,
        })

v1 = np.array([r["v1"] for r in rows]); l3 = np.array([r["3L"] for r in rows]); opt = np.array([r["opt"] for r in rows])
other = np.array([r["other"] for r in rows]); size = np.array([r["size"] for r in rows])
gap = v1 - l3  # > 0: 3L-Cache is better
lose = gap > 0.005
print(f"train-long: {len(rows)} cells; v1 realistic 5-fold CV vs 3L-Cache, OPT and the best other published policy (change vs S3-FIFO, scored part)")
print(f"  mean: v1 {100*v1.mean():+.2f}%  3L {100*l3.mean():+.2f}%  best other {100*other.mean():+.2f}%  OPT {100*opt.mean():+.2f}%")
print(f"  v1 captures {100*np.mean(v1)/np.mean(opt):.0f}% of OPT's mean gain; 3L-Cache {100*np.mean(l3)/np.mean(opt):.0f}%")
print(f"  3L-Cache better than v1 by > 0.5 pt in {lose.sum()} cells, v1 better by > 0.5 pt in {(gap < -0.005).sum()}")
for lo, hi, label in ((0, 0.5, "<0.5%"), (0.5, 2, "0.5-2%"), (2, 8, "2-8%"), (8, 1e9, ">=8%")):
    m = (size >= lo) & (size < hi)
    print(f"    size {label:<7} {m.sum():3d} cells: 3L ahead in {int((lose & m).sum()):2d}; mean gap v1-3L {100*gap[m].mean():+5.2f} pt; "
          f"OPT {100*opt[m].mean():+6.2f}%, v1 gets {100*v1[m].mean()/opt[m].mean():3.0f}% of it, 3L {100*l3[m].mean()/opt[m].mean():3.0f}%")
print("  in the cells where 3L-Cache is ahead:")
print(f"    gap: mean {100*gap[lose].mean():.2f} pt, median {100*np.median(gap[lose]):.2f} pt, largest {100*gap[lose].max():.2f} pt; "
      f"the best other policy also beats v1 in {int((other[lose] < v1[lose] - 0.005).sum())} of them")
print(f"    best other policy there: {dict(Counter(rows[i]['other_name'] for i in np.flatnonzero(lose)).most_common())}")
traces = sorted({r["trace"] for r in rows})
per = Counter(rows[i]["trace"] for i in np.flatnonzero(lose))
print(f"    spread over traces: {len(per)} of {len(traces)} traces; cells per such trace: {sorted(per.values(), reverse=True)}")
order = np.argsort(-gap)[:10]
print("  the 10 largest 3L-Cache leads (anonymous): size%, v1, 3L, best other, OPT")
for i in order:
    print(f"    {size[i]:6.2f}%  v1 {100*v1[i]:+6.2f}%  3L {100*l3[i]:+6.2f}%  other {100*other[i]:+6.2f}% ({rows[i]['other_name']})  OPT {100*opt[i]:+6.2f}%")
