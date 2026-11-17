import json, statistics as st, sys
from pathlib import Path
import numpy as np
root, designs_file, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
designs = [json.loads(l) for l in designs_file.read_text().splitlines()]
fam = np.array([d["family"] for d in designs])
SEEN_FROM, SPLIT_W = 2, 8   # select on windows 2-7 (observed part after warm-up), score windows 8-39
res = {k: [] for k in ("3L", "best_published", "prefix_tuned(baselines)", "s3grid pick", "s3grid+s4like pick", "all pick", "s3grid hindsight", "all hindsight")}
check = []
for d in sorted(root.iterdir()):
    if not (d / "meta.json").is_file(): continue
    meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        out = d / f"{tag}-{cap}.out"
        if not out.is_file(): continue
        rows = np.array([[int(x) for x in l.split()] for l in out.read_text().splitlines()])
        if len(rows) != len(designs): continue
        after, win = rows[:, 1].astype(float), rows[:, 2:].astype(float)
        n, split = meta["requests"], meta["split"]
        base = after[0]; rel = lambda v: (v - base) / base
        check.append(abs(base / (n - split) - meta["s3fifo"][str(cap)]) / meta["s3fifo"][str(cap)])
        seen = win[:, SEEN_FROM:SPLIT_W].sum(1)
        def pick(mask): idx = np.where(mask)[0]; return rel(after[idx[np.argmin(seen[idx])]])
        s3 = meta["s3fifo"][str(cap)]
        res["3L"].append((meta["3L"][str(cap)] - s3) / s3); res["best_published"].append((meta["best"][str(cap)][1] - s3) / s3)
        res["prefix_tuned(baselines)"].append((meta["tuned"][str(cap)] - s3) / s3)
        res["s3grid pick"].append(pick(np.isin(fam, ["s3fifo", "s3grid"]))); res["s3grid+s4like pick"].append(pick(np.isin(fam, ["s3fifo", "s3grid", "s4like"])))
        res["all pick"].append(pick(np.ones(len(designs), bool)))
        res["s3grid hindsight"].append(rel(after[np.isin(fam, ["s3fifo", "s3grid"])].min())); res["all hindsight"].append(rel(after.min()))
n = len(res["3L"]); print(f"{n} cells; s3fifo check vs baselines: max rel diff {max(check):.2e}")
for k, xs in res.items():
    wins = sum(x < t for x, t in zip(xs, res["3L"]))
    print(f"  {k:<26} mean {100*st.mean(xs):+6.2f}%  median {100*st.median(xs):+6.2f}%  beats 3L {wins}/{n}")
