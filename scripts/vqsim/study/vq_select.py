"""Selection rules over the r1 sweep (cold runs, 40 windows), train-long. Leave-one-workload-out for portfolios."""
import json, sys
from pathlib import Path
import numpy as np
root, designs_file = Path(sys.argv[1]), Path(sys.argv[2])
designs = [json.loads(l) for l in designs_file.read_text().splitlines()]
fam = np.array([d["family"] for d in designs])
cells = []
for d in sorted(root.iterdir()):
    if not (d / "meta.json").is_file(): continue
    meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        rows = np.array([[int(x) for x in l.split()] for l in (d / f"r1-{cap}.out").read_text().splitlines()])
        s3 = meta["s3fifo"][str(cap)]
        cells.append({"trace": d.name, "after": rows[:, 1].astype(float), "win": rows[:, 2:].astype(float), "l3": (meta["3L"][str(cap)] - s3) / s3})
n = len(cells); D = len(designs)
A = np.array([c["after"] for c in cells]); W = np.array([c["win"] for c in cells])  # n x D, n x D x 40
R = (A - A[:, :1]) / A[:, :1]  # suffix change vs S3-FIFO (design 0)
L3 = np.array([c["l3"] for c in cells])
traces = np.array([c["trace"] for c in cells])
def seen(lo, hi): return W[:, :, lo:hi].sum(2)
def report(name, picks):
    x = R[np.arange(n), picks]
    print(f"  {name:<44} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  beats 3L {int((x < L3).sum())}/{n}  worse>0.05% {int((x > 0.0005).sum())}")
def pick(score, mask=None):
    s = score.copy()
    if mask is not None: s[:, ~mask] = np.inf
    return s.argmin(1)
print(f"{n} cells, {D} designs; 3L-Cache mean {100*L3.mean():+.2f}% median {100*np.median(L3):+.2f}%")
print("hindsight all:", f"{100*R.min(1).mean():+.2f}%")
for lo, hi in ((2, 8), (4, 8), (6, 8), (0, 8)):
    report(f"min misses, windows {lo}-{hi-1}", pick(seen(lo, hi)))
S = seen(2, 8); rel = (S - S[:, :1]) / S[:, :1]
for m in (0.0, 0.005, 0.01, 0.02):
    p = pick(S); keep = rel[np.arange(n), p] > -m
    report(f"margin {m:.3f} else S3-FIFO", np.where(keep, 0, p))
# consistency: count windows 2-7 won relative ranks; pick best mean rank
ranks = np.argsort(np.argsort(W[:, :, 2:8], axis=1), axis=1).mean(2)
report("best mean rank over windows 2-7", ranks.argmin(1))
# leave-one-workload-out greedy portfolio
def greedy(train, k):
    P = [0]
    for _ in range(k - 1):
        best, gain = None, None
        for d in range(D):
            if d in P: continue
            Q = P + [d]
            p = np.array(Q)[S[np.ix_(train, Q)].argmin(1)]
            v = R[train, p].mean()
            if gain is None or v < gain: best, gain = d, v
        P.append(best)
    return P
for k in (2, 4, 8, 16):
    picks = np.zeros(n, int); chosen = []
    for t in np.unique(traces):
        test = np.where(traces == t)[0]; train = np.where(traces != t)[0]
        P = greedy(train, k); chosen.append(P)
        picks[test] = np.array(P)[S[np.ix_(test, P)].argmin(1)]
    report(f"LOO greedy portfolio k={k}", picks)
    from collections import Counter
    print("     most used:", [designs[d]["name"] for d, _ in Counter(d for P in chosen for d in P).most_common(6)])
