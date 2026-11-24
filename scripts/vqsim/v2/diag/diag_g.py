"""Per trace (anonymous): which component turns an exact-leader win into a live loss. final-long, v2 and v1."""
import json, re
from pathlib import Path
import numpy as np
src = open("/tmp/diag_b.py").read().split("names = sorted")[0].replace("suite = sys.argv[1]", "suite = 'final'"); g = {}; exec(src, g)
T = g["T"]; names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
rows = []
for k, t in enumerate(names):
    d = T / t; meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        get = g["cold"](d, cap); W0 = np.array([int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]], float); base = W0[8:].sum()
        s3 = meta["s3fifo"][str(cap)]; r = [k, 100 * cap / meta["objects"], (meta["3L"][str(cap)] - s3) / s3]
        for job, err, outl in g["blocks"](d, cap):
            cands, seq = g["parse"](job, err); M = np.array([get(c) for c in cands])
            live = (sum(int(x) for x in outl.split()[3 + 8:]) - base) / base
            cseq = (sum(M[seq[i], 8 + i] for i in range(32)) - base) / base
            lead = np.cumsum(M[:, 2:], axis=1)[:, 5:37].argmin(0); ideal = (M[lead, np.arange(8, 40)].sum() - base) / base
            r += [live, cseq, ideal]
        rows.append(r)
R = np.array(rows); K = R[:, 0].astype(int)
per = lambda j: np.array([R[K == k, j].mean() for k in range(len(names))]) * 100
L3 = per(2); v1l, v1c, v1i, v2l, v2c, v2i = (per(j) for j in range(3, 9))
print("trace  3L     | v1 live  no-takeover exact | v2 live  no-takeover exact | v2 lost by (pt): takeover sampling  (+ = turns against v2)")
for k in np.argsort(v2l - L3)[::-1]:
    tag = "LOSS" if v2l[k] >= L3[k] else "win "
    print(f"t{k:02d} {L3[k]:+7.2f} | {v1l[k]:+7.2f} {v1c[k]:+7.2f} {v1i[k]:+7.2f} | {v2l[k]:+7.2f} {v2c[k]:+7.2f} {v2i[k]:+7.2f} | {tag} {v2l[k]-v2c[k]:+6.2f} {v2c[k]-v2i[k]:+6.2f}   v2 exact margin {v2i[k]-L3[k]:+6.2f}")
w = lambda x: int((x < L3).sum())
print(f"traces beating 3L: v1 live {w(v1l)}, no-takeover {w(v1c)}, exact {w(v1i)} | v2 live {w(v2l)}, no-takeover {w(v2c)}, exact {w(v2i)}")
