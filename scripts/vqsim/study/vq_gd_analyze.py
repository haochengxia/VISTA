"""Ghost depth vs cache size (train-long, cold full runs, windows 8-39 vs S3-FIFO; memory ignored)."""
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]); designs = [json.loads(l) for l in open("/mydata/vq/gd_designs.jsonl")]
G = sorted({float(d["config"].split("ghost=")[1].split()[0]) for d in designs})
rows = []
for d in sorted(root.iterdir()):
    if not (d / "meta.json").is_file(): continue
    meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        f = d / f"gd-{cap}.out"
        if not f.is_file() or len(f.read_text().splitlines()) != len(designs): continue
        base = sum(int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2 + 8:])
        ch = [(sum(int(x) for x in l.split()[2 + 8:]) - base) / base for l in f.read_text().splitlines()]
        s3 = meta["s3fifo"][str(cap)]
        rows.append((100 * cap / meta["objects"], (meta["3L"][str(cap)] - s3) / s3, ch))
print(f"{len(rows)} cells")
for lo, hi, label in ((0, 0.5, "<0.5%"), (0.5, 2, "0.5-2%"), (2, 8, "2-8%"), (8, 1e9, ">=8%")):
    g = [r for r in rows if lo <= r[0] < hi]
    if not g: continue
    C = np.array([r[2] for r in g]); L3 = np.array([r[1] for r in g])
    line = f"{label:<7} ({len(g):2d} cells) 3L {100*L3.mean():+6.2f}% |"
    for gd in G:
        idx = [i for i, d in enumerate(designs) if f"ghost={gd:g} " in d["config"] or f"ghost={gd} " in d["config"]]
        fixed = C[:, idx].mean(0).min(); hind = C[:, idx].min(1).mean()
        line += f" G={gd:g}: fixed {100*fixed:+6.2f} hind {100*hind:+6.2f} |"
    print(line)
