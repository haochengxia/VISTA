"""Screening pools chosen for coverage (train only, idealized; vq_covpool.py [xr]): per CV fold (5 folds by trace, as vq_portfolio.py), a greedy
facility-location pick of S designs that best complement the fold's CV 32 (per training cell, the best cold result over the
CV 32 leader and the pool so far). Scored on the held-out fold: CV 32 + the 16 pool designs with the fewest misses in the
observed windows 2..7, idealized leader. Writes /mydata/vq/pool-cov<S>[x][lodo]-cv.json (by_trace: that fold's pool).
CV_BY=dataset (env): leave one dataset out, with the CV 32 of portfolio-r1bplodo-cv.json."""
import json, os, sys
from pathlib import Path
import numpy as np
T = Path("/mydata/vq/train-long"); names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
srcs = (("r1", "vq_designs.jsonl"), ("bp", "bp_designs.jsonl"), ("rd", "rd_designs.jsonl"), ("gt", "gt_designs.jsonl"))
if len(sys.argv) > 1 and sys.argv[1] == "xr": srcs += (("xr", "xr_designs.jsonl"),)
allc = [json.loads(l)["config"] for _, f in srcs for l in open(f"/mydata/vq/{f}")]
uniq = {}
for j, c in enumerate(allc): uniq.setdefault(c, j)
keep = np.array(sorted(uniq.values())); configs = [allc[j] for j in keep]; col = {c: i for i, c in enumerate(configs)}
lodo = os.environ.get("CV_BY") == "dataset"
cv = json.load(open(f"/mydata/vq/portfolio-r1bp{'lodo' if lodo else ''}-cv.json"))["by_trace"]
cells, Ws, L3 = [], [], []
for t in names:
    d = T / t; meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        W = np.array([[int(x) for x in l.split()[2:]] for s, _ in srcs for l in (d / f"{s}-{cap}.out").read_text().splitlines()], np.float32)[keep]
        cells.append(t); Ws.append(W); s3 = meta["s3fifo"][str(cap)]; L3.append((meta["3L"][str(cap)] - s3) / s3)
W = np.stack(Ws); L3 = np.array(L3); cells = np.array(cells)
base = W[:, 0, 8:].sum(1)
Xc = (W[:, :, 8:].sum(2) - base[:, None]) / base[:, None]  # cells x designs: cold change on windows 8..39
pre = W[:, :, 2:8].sum(2)
def leader(c, idx):
    P = W[c, idx]; C = np.cumsum(P[:, 2:], axis=1)[:, 5:37]; lead = C.argmin(0); return (P[lead, np.arange(8, 40)].sum() - base[c]) / base[c]
rng = np.random.default_rng(0); tr = sorted(set(names)); rng.shuffle(tr); fold_of = {t: i % 5 for i, t in enumerate(tr)}
if lodo: ds = sorted({t.split("-")[1] for t in tr}); fold_of = {t: ds.index(t.split("-")[1]) for t in tr}
fold = np.array([fold_of[t] for t in cells])
SIZES = (32, 64, 128, 224)
res = {S: np.zeros(len(cells)) for S in SIZES}; full = np.zeros(len(cells)); cvonly = np.zeros(len(cells)); oracle = {S: np.zeros(len(cells)) for S in SIZES}
pools = {S: {} for S in SIZES}
for f in sorted(set(fold_of.values())):
    trn = np.where(fold != f)[0]; tst = np.where(fold == f)[0]
    t0 = next(t for t in names if fold_of[t] == f); c32 = [col[c] for c in cv[t0][:32]]
    cur = np.array([leader(c, c32) for c in trn])
    banned = np.zeros(len(configs), bool); banned[c32] = True; pick = []
    for _ in range(max(SIZES)):
        gain = np.maximum(0, cur[:, None] - Xc[trn]).sum(0); gain[banned] = -1
        j = int(gain.argmax()); pick.append(j); banned[j] = True; cur = np.minimum(cur, Xc[trn, j])
    for c in tst:
        cvonly[c] = leader(c, c32)
        order = [j for j in np.argsort(pre[c], kind="stable") if j not in set(c32)]
        full[c] = leader(c, c32 + order[:16])
        for S in SIZES:
            ps = set(pick[:S]); top = [j for j in np.argsort(pre[c], kind="stable") if j in ps][:16]
            res[S][c] = leader(c, c32 + top); oracle[S][c] = min(Xc[c, c32 + pick[:S]].min(), Xc[c, c32].min())
    for S in SIZES:
        for t in names:
            if fold_of[t] == f: pools[S][t] = [configs[j] for j in pick[:S]]
def rep(lab, x):
    per = lambda v: np.array([v[cells == t].mean() for t in names])
    print(f"  {lab:<34} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  cells beating 3L {int((x < L3).sum()):3d}/100  traces beating 3L {int((per(x) < per(L3)).sum()):2d}/25")
print(f"train-long: {len(cells)} cells, {len(configs)} designs; idealized leader, CV 32 + 16 screened by windows 2..7; " + ("leave one dataset out" if lodo else "5-fold CV"))
rep("3L-Cache", L3); rep("CV 32 alone", cvonly); rep("screened from the full pool", full)
for S in SIZES:
    rep(f"screened from coverage pool {S}", res[S]); rep(f"   (pool {S} oracle, hindsight)", oracle[S])
    json.dump({"by_trace": pools[S]}, open(f"/mydata/vq/pool-cov{S}{'x' if len(srcs) > 4 else ''}{'lodo' if lodo else ''}-cv.json", "w"))
