"""Takeover cost: split vs later switches, and cache turnover per window (anonymous). usage: diag_c.py final|train"""
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.argv += []; suite = sys.argv[1]
exec(open("/tmp/diag_b.py").read().split("names = sorted")[0].replace("suite = sys.argv[1]", f"suite = {suite!r}"))
VQ = "/mydata/vqsim-diag"
names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
def cell(a):
    k, d, cap = a
    meta = json.loads((d / "meta.json").read_text()); get = cold(d, cap)
    W0 = np.array([int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]], float); base = W0[8:].sum()
    res = dict(k=k, size=100 * cap / meta["objects"], turn=W0[8:].mean() / cap)  # S3-FIFO misses per window / C: cache turnovers per window
    for tag, (job, err, outl) in zip(("v1", "v2"), blocks(d, cap)):
        cands, seq = parse(job, err); d0 = cands[seq[0]]
        jobs = d / f"dgho-{tag}-{cap}.jobs"; jobs.write_text(f"{cap} {meta['split']} 40 {d0} handover=1\n")
        H = np.array([int(x) for x in subprocess.run(["nice", "-n", "10", VQ, str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout.split()[2:]], float)
        M = np.array([get(c) for c in cands])
        live = sum(int(x) for x in outl.split()[3 + 8:]); cseq = sum(M[seq[i], 8 + i] for i in range(32))
        res[f"{tag}_split"] = (H[8:].sum() - get(d0)[8:].sum()) / base  # first pick kept to the end: takeover vs cold
        res[f"{tag}_total"] = (live - cseq) / base
        res[f"{tag}_sw"] = sum(1 for i in range(1, 32) if seq[i] != seq[i - 1])
        res[f"{tag}_keep"] = (H[8:].sum() - base) / base  # never switch after the split
        res[f"{tag}_live"] = (live - base) / base
        # windows until the split takeover's cost is mostly gone: cumulative (handover - cold) reaches 90% of its total
        g = np.cumsum(H[8:] - get(d0)[8:]); res[f"{tag}_decay"] = int(np.argmax(g >= 0.9 * g[-1])) + 1 if g[-1] > 0 else 0
    return res
cells = [(k, T / t, cap) for k, t in enumerate(names) for cap in json.loads((T / t / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(32) as ex: R = list(ex.map(cell, cells))
A = lambda key: np.array([r[key] for r in R], float)
print(f"{suite}: {len(R)} cells; takeover cost = live - same choices without takeover (pt of the mean change)")
for tag in ("v1", "v2"):
    tot, sp, sw = A(tag + "_total"), A(tag + "_split"), A(tag + "_sw")
    later = tot - sp
    print(f"  {tag}: total {100*tot.mean():+.2f} pt = split takeover ~{100*sp.mean():+.2f} + later switches ~{100*later.mean():+.2f}; switches per cell {sw.mean():.2f};"
          f" per switch ~{100*later.sum()/max(sw.sum(),1):+.3f} pt of a cell; keeping the split pick to the end {100*A(tag+'_keep').mean():+.2f}% vs live {100*A(tag+'_live').mean():+.2f}%")
print(f"  S3-FIFO cache turnovers per window (misses per window / C): median {np.median(A('turn')):.2f}")
for lo, hi in ((0, 0.5), (0.5, 2), (2, 8), (8, 1e9)):
    m = (A("size") >= lo) & (A("size") < hi)
    print(f"    size {lo}-{hi:g}% ({int(m.sum())}): turnovers/window median {np.median(A('turn')[m]):.2f}; split cost v1 {100*A('v1_split')[m].mean():+.2f} v2 {100*A('v2_split')[m].mean():+.2f};"
          f" later-switch cost v1 {100*(A('v1_total')-A('v1_split'))[m].mean():+.2f} v2 {100*(A('v2_total')-A('v2_split'))[m].mean():+.2f}; switches v1 {A('v1_sw')[m].mean():.1f} v2 {A('v2_sw')[m].mean():.1f};"
          f" split cost 90% spent after median {np.median(A('v2_decay')[m]):.0f} windows")
