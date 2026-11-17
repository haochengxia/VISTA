"""Switching policies for online re-choice, replayed on a live cache (vqsim schedules).
Decisions use exact shadows (r1 cold windows; 1% mini-sims decide as well). Scored on windows 8-39 vs S3-FIFO.
  split-*  S3-FIFO serves the prefix; the policy takes over at the split (the engineer protocol)
  start-*  the policy runs from window 4 (first choice from windows 2-3), like 3L-Cache from the start
  m=X      switch only if the leader's misses so far are at least X lower than the current design's
  h=H      count misses over the last H windows only"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
root, designs_file, workers, tag = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
designs = [json.loads(l)["config"] for l in designs_file.read_text().splitlines()]
VARIANTS = [("split", 0, None, None), ("split", 0, 0.0, None), ("split", 0, 0.01, None), ("split", 0, 0.03, None), ("split", 0, 0.01, 8),
            ("start", 4, None, None), ("start", 4, 0.0, None), ("start", 4, 0.01, None), ("start", 4, 0.03, None), ("start", 4, 0.01, 8)]
def name(v):
    kind, first, m, h = v
    return f"{kind} " + ("static" if m is None else f"online m={m:.0%}" + (f" h={h}" if h else ""))
def schedule(W, v):
    kind, first, m, h = v
    w0 = 8 if kind == "split" else first
    cur = int(W[:, 2:w0].sum(1).argmin()); out = [("s" if kind == "split" else str(w0), cur)]
    if m is None: return out
    for w in range(w0 + 1, 40):
        lo = 2 if h is None else max(2, w - h)
        cum = W[:, lo:w].sum(1); lead = int(cum.argmin())
        if lead != cur and cum[lead] < (1 - m) * cum[cur]:
            cur = lead; out.append((str(w), cur))
    return out
def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text()); split = meta["split"]
    out = d / f"{tag}-{cap}.out"
    W = np.array([[int(x) for x in l.split()[2:]] for l in (d / f"r1-{cap}.out").read_text().splitlines()], float)
    if not (out.is_file() and len(out.read_text().splitlines()) == len(VARIANTS)):
        blocks = []
        for v in VARIANTS:
            sch = schedule(W, v); used = sorted({i for _, i in sch}); idx = {i: k for k, i in enumerate(used)}
            blocks.append(f"online {cap} {split} 40 schedule=" + ",".join(f"{w}:{idx[i]}" for w, i in sch) + "\n" + "".join(designs[i] + "\n" for i in used) + "end\n")
        jobs = d / f"{tag}-{cap}.jobs"; jobs.write_text("".join(blocks))
        out.write_text(subprocess.run(["nice", "-n", "10", "/mydata/vqsim3", str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout)
    lines = out.read_text().splitlines()
    base = W[0, 8:].sum(); s3 = meta["s3fifo"][str(cap)]
    res = [(sum(int(x) for x in l.split()[3 + 8:]) - base) / base for l in lines]
    sw = [int(l.split()[2]) for l in lines]
    return (meta["3L"][str(cap)] - s3) / s3, res, sw
cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool: got = list(pool.map(cell, cells))
L3 = np.array([g[0] for g in got]); R = np.array([g[1] for g in got]); S = np.array([g[2] for g in got])
def rep(label, x): print(f"  {label:<26} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  beats 3L {int((x < L3).sum())}/{len(x)}  worse {int((x > 0.0005).sum())}")
print(f"{len(L3)} cells, windows 8-39 vs S3-FIFO"); rep("3L-Cache", L3)
for k, v in enumerate(VARIANTS):
    rep(name(v), R[:, k]); print(f"      switches median {np.median(S[:, k]):.0f} mean {S[:, k].mean():.1f}")
