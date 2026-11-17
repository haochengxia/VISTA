"""Pre-registered realistic online re-choice (fixed on train-long 2026-11-17 ~09:00), run once per suite:
S3-FIFO serves the prefix; at the split and each later window start the live cache switches to the
candidate with the fewest sampled misses since window 2 (mini-sims, 1% of objects, mini cache >= 1000),
margin 0. Candidate sets: A = 426 seed designs (primary), B = designs the board's pool chooses on
train-long (84 incl. S3-FIFO), C = A + B. 'static' = the first choice only (margin 2: never switch).
Scored on windows 8-39 vs S3-FIFO (r1 design 0), next to 3L-Cache."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
root, workers, tag = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
A = [json.loads(l)["config"] for l in open("/mydata/vq/vq_designs.jsonl")]
B = [json.loads(l)["config"] for l in open("/mydata/vq/bp_designs.jsonl")]
C = list(dict.fromkeys(A + B))
VARIANTS = [("A online", A, 0), ("A static", A, 2), ("B online", B, 0), ("B static", B, 2), ("C online", C, 0)]
def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text()); split = meta["split"]
    out = d / f"{tag}-{cap}.out"
    if not (out.is_file() and len(out.read_text().splitlines()) == len(VARIANTS)):
        jobs = d / f"{tag}-{cap}.jobs"
        jobs.write_text("".join(f"online {cap} {split} 40 rate=0.01 min=1000 margin={m}\n" + "".join(c + "\n" for c in cands) + "end\n" for _, cands, m in VARIANTS))
        out.write_text(subprocess.run(["nice", "-n", "10", "/mydata/vqsim3", str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout)
    W0 = [int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]
    base = sum(W0[8:]); s3 = meta["s3fifo"][str(cap)]
    lines = out.read_text().splitlines()
    return (meta["3L"][str(cap)] - s3) / s3, [(sum(int(x) for x in l.split()[3 + 8:]) - base) / base for l in lines], [int(l.split()[2]) for l in lines]
cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool: got = list(pool.map(cell, cells))
L3 = np.array([g[0] for g in got]); R = np.array([g[1] for g in got]); S = np.array([g[2] for g in got])
def rep(label, x): print(f"  {label:<12} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  beats 3L {int((x < L3).sum())}/{len(x)}  worse than S3-FIFO {int((x > 0.0005).sum())}")
print(f"{root.name}: {len(L3)} cells, windows 8-39 vs S3-FIFO"); rep("3L-Cache", L3)
for k, (name, cands, _) in enumerate(VARIANTS):
    rep(name, R[:, k]); print(f"      {len(cands)} candidates; switches median {np.median(S[:, k]):.0f} mean {S[:, k].mean():.1f}")
