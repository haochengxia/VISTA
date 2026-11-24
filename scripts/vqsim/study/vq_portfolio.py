"""Small candidate sets ("portfolios") for online re-choice, chosen greedily.
Per cell, the per-window misses of every design from full cold runs (r1: 426 seeds; bp: board picks).
Idealized online leader (fewest misses from window 2 on, decided at the split and each later window, margin 0;
ties go to the earlier member), scored on windows 8-39 vs S3-FIFO (r1 design 0). Greedy: start from {S3-FIFO},
add the design that lowers the mean change most.
  vq_portfolio.py SUITE TAG KMAX SOURCES        greedy on all cells, writes /mydata/vq/portfolio-TAG.json
  vq_portfolio.py SUITE TAG KMAX SOURCES cv     5-fold CV by trace: greedy on 4 folds, scored on the 5th
SOURCES: comma-separated r1, bp, or NAME=DESIGNS.jsonl (outputs NAME-<cap>.out from vq_run3.py)
GHOST_MAX=x (env): only designs whose ghost is at most x times the cache (memory-fair: S3-FIFO's is 0.9).
CV_BY=dataset (env, with cv): leave one dataset out instead of 5 folds by trace.
FAMILY=s3 (env): only S3-structured designs (vq_family.py: their takeovers map stage to stage).
START=portfolio-X-cv.json (env, with cv): each fold's greedy extends that fold's portfolio from an earlier CV run
(its designs must be among SOURCES) instead of starting from {S3-FIFO}; START_K (default 32) members of it."""
import json, os, re, sys
from pathlib import Path
import numpy as np
root, tag, kmax, sources = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")
cv = len(sys.argv) > 5 and sys.argv[5] == "cv"
files = {"r1": "/mydata/vq/vq_designs.jsonl", "bp": "/mydata/vq/bp_designs.jsonl"}
files.update({n: f for n, _, f in (s.partition("=") for s in sources) if f})  # NAME=designs.jsonl adds a source
sources = [s.partition("=")[0] for s in sources]
configs, cols = [], {}
for s in sources:
    cols[s] = []
    for line in open(files[s]):
        c = json.loads(line)["config"]
        if c not in configs: configs.append(c)
        cols[s].append(configs.index(c))
cells, Ws, L3, cap_of, obj_of = [], [], [], {}, {}
for d in sorted(root.iterdir()):
    if not (d / "meta.json").is_file(): continue
    meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        W = np.zeros((len(configs), 40)); ok = True
        for s in sources:
            f = d / f"{s}-{cap}.out"
            lines = f.read_text().splitlines() if f.is_file() else []
            if len(lines) != len(cols[s]): ok = False; break
            for i, l in zip(cols[s], lines): W[i] = [int(x) for x in l.split()[2:]]
        if ok:
            cells.append(f"{d.name}/{cap}"); Ws.append(W); cap_of[cells[-1]] = cap; obj_of[d.name] = meta["objects"]
            s3 = meta["s3fifo"][str(cap)]; L3.append((meta["3L"][str(cap)] - s3) / s3)
gmax = float(os.environ.get("GHOST_MAX", "inf"))
banned = np.array([float(m.group(1)) if (m := re.search(r"(?:^| )ghost=([\d.]+)", c)) else 0.9 for c in configs]) > gmax + 1e-9
if os.environ.get("FAMILY") == "s3":
    from vq_family import s3like
    banned |= ~np.array([s3like(c) for c in configs])
W = np.stack(Ws); L3 = np.array(L3)            # cells x designs x 40
X = W[:, :, 8:]                                 # misses in the scored windows 8..39
C = np.cumsum(W[:, :, 2:], axis=2)[:, :, 5:37]  # misses in windows 2..w-1, for w = 8..39
base = X[:, 0].sum(1)
def greedy(mask, kmax, start=(0,)):
    """Portfolio and, after each step, the per-cell result on all cells (None below the start's size)."""
    port = list(start); lead_c = C[:, port[0]].copy(); lead_x = X[:, port[0]].copy()
    for i in port[1:]:  # the start portfolio's leader (ties go to the earlier member)
        b = C[:, i] < lead_c; lead_c = np.where(b, C[:, i], lead_c); lead_x = np.where(b, X[:, i], lead_x)
    results = [None] * (len(port) - 1) + [lead_x.sum(1)]
    for _ in range(kmax - len(port)):
        better = C < lead_c[:, None, :]          # cells x designs x windows: candidate would lead
        tot = np.where(better, X, lead_x[:, None, :]).sum(2)          # cells x designs
        change = (tot - base[:, None]) / base[:, None]
        score = change[mask].mean(0); score[port] = np.inf; score[banned] = np.inf
        i = int(score.argmin()); port.append(i)
        b = better[:, i]; lead_c = np.where(b, C[:, i], lead_c); lead_x = np.where(b, X[:, i], lead_x)
        results.append(lead_x.sum(1))
    return port, [None if r is None else (r - base) / base for r in results]
def rep(label, x): print(f"  {label:<14} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  beats 3L {int((x < L3).sum())}/{len(x)}  worse {int((x > 0.0005).sum())}", flush=True)
def leader_all():
    lead = np.where(banned[None, :, None], np.inf, C).argmin(1)  # cells x windows
    return (np.take_along_axis(X, lead[:, None, :], 1)[:, 0].sum(1) - base) / base
print(f"{root.name}: {len(cells)} cells with all of {sources}; {len(configs) - int(banned.sum())} designs" + (f" (ghost <= {gmax})" if gmax < float("inf") else "") + (" (S3-structured)" if os.environ.get("FAMILY") == "s3" else "") + ((" (leave one dataset out)" if os.environ.get("CV_BY") == "dataset" else " (5-fold CV by trace)") if cv else ""))
rep("3L-Cache", L3); rep("all designs", leader_all())
KS = [k for k in (2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256) if k <= kmax]
if cv:
    traces = sorted({c.split("/")[0] for c in cells}); rng = np.random.default_rng(0); rng.shuffle(traces)
    fold_of = {t: i % 5 for i, t in enumerate(traces)}
    if os.environ.get("CV_BY") == "dataset":  # leave one dataset out: folds are the datasets (trace names are PREFIX-DATASET-K)
        ds = sorted({t.split("-")[1] for t in traces}); fold_of = {t: ds.index(t.split("-")[1]) for t in traces}
    fold = np.array([fold_of[c.split("/")[0]] for c in cells])
    out = {k: np.zeros(len(cells)) for k in KS}
    by_trace = {}
    start = json.load(open(os.environ["START"]))["by_trace"] if os.environ.get("START") else None
    if start:
        start = {t: ps[: int(os.environ.get("START_K", "32"))] for t, ps in start.items()}
        KS = [k for k in KS if k >= len(next(iter(start.values())))]
    for f in sorted(set(fold_of.values())):
        first = start[next(t for t in traces if fold_of[t] == f)] if start else None
        port, res = greedy(fold != f, kmax, [configs.index(c) for c in first] if first else (0,))
        for k in KS: out[k][fold == f] = res[k - 1][fold == f]
        for t in traces:
            if fold_of[t] == f: by_trace[t] = [configs[i] for i in port]
    for k in KS: rep(f"K={k}", out[k])
    json.dump({"suite": root.name, "sources": sources, "by_trace": by_trace}, open(f"/mydata/vq/portfolio-{tag}-cv.json", "w"))
    sizes = np.array([cap_of[c] / obj_of[c.split("/")[0]] * 100 for c in cells])
    for lo, hi, label in ((0, 0.5, "<0.5%"), (0.5, 2, "0.5-2%"), (2, 8, "2-8%"), (8, 1e9, ">=8%")):
        m = (sizes >= lo) & (sizes < hi)
        if m.any(): print(f"    size {label:<7} ({m.sum():3d} cells): K=32 {100*out[32][m].mean():+6.2f}%  3L {100*L3[m].mean():+6.2f}%")
else:
    port, res = greedy(np.ones(len(cells), bool), kmax)
    for k in KS: rep(f"K={k}", res[k - 1])
    json.dump({"suite": root.name, "sources": sources, "portfolio": [configs[i] for i in port],
               "curve": [{"k": k + 1, "mean": float(r.mean()), "median": float(np.median(r))} for k, r in enumerate(res)]},
              open(f"/mydata/vq/portfolio-{tag}.json", "w"), indent=1)
