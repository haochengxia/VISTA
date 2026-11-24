"""Takeover cost by change type (post hoc, anonymous aggregates): is a takeover cheap when both designs are S3-structured
(two stages, small -> main, main -> itself, stage-to-stage takeover, small's threshold reachable)? Uses diag_b's data.
Each scored window's cost = live misses - the current design's cold-run misses, charged to the last design change
(the split's takeover from S3-FIFO, or a later switch). usage: diag_fam.py final|train"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(Path(__file__).with_name("diag_b.py")).read()
exec(src[:src.index("names = sorted(")])  # cold(), blocks(), parse(), pool, T
def s3like(c):
    d = {"st": []}
    for tok in c.split():
        k, _, v = tok.partition("=")
        if k == "st": d["st"].append([float(x) for x in v.split(",")])
        else: d[k] = float(v)
    st = d["st"]
    if len(st) != 2: return False
    entry = int(d.get("entry", 0)); maxf = int(d.get("maxf", 3))
    if entry != 0 or int(d.get("asmall", entry)) != 0 or int(d.get("amain", 1)) != 1: return False
    s0, s1 = st
    if int(s0[3]) != 1 or int(s0[5]) != -1 or int(s1[5]) != -1 or int(s1[3]) != 1: return False
    return (s0[2] <= maxf) if int(s0[1]) else (s0[2] == 0)
names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
ncell = 0; agg = {}; cells = {"v1": [], "v2": []}
for t in names:
    d = T / t; meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        ncell += 1; get = cold(d, cap)
        W0 = np.array([int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]], float); base = W0[8:].sum()
        for tag, (job, err, outl) in zip(("v1", "v2"), blocks(d, cap)):
            cands, seq = parse(job, err); M = np.array([get(c) for c in cands]); live = np.array([int(x) for x in outl.split()[3:]], float)
            fam = [s3like(c) for c in cands]
            kind, since, tot = None, 0, 0.0
            for i in range(32):
                w = 8 + i
                if i == 0: kind = "within" if fam[seq[0]] else "other"; since = 0; lab = "split"
                elif seq[i] != seq[i - 1]: kind = "within" if fam[seq[i]] and fam[seq[i - 1]] else "other"; since = 0; lab = "switch"
                c = (live[w] - M[seq[i], w]) / base; tot += c
                key = (tag, lab, kind, min(since, 4)); agg[key] = agg.get(key, 0.0) + c; since += 1
                if since == 1: agg[(tag, lab, kind, "n")] = agg.get((tag, lab, kind, "n"), 0) + 1
            cells[tag].append((all(fam[s] for s in set(seq)), tot, sum(seq[i] != seq[i - 1] for i in range(1, 32))))
print(f"{T.name}: {ncell} cells; takeover cost in pt of the suite mean, by the last change, windows 0..3 and 4+ after it")
for tag in ("v1", "v2"):
    for lab in ("split", "switch"):
        for kind in ("within", "other"):
            n = agg.get((tag, lab, kind, "n"), 0)
            if not n: continue
            parts = [100 * agg.get((tag, lab, kind, s), 0.0) / ncell for s in range(5)]
            print(f"  {tag} {lab:<6} {kind:<6} {n:4d} changes: total {sum(parts):+.3f} pt = " + " ".join(f"{p:+.3f}" for p in parts)
                  + f"; per change {100 * sum(parts) * ncell / n / ncell:+.4f} pt of the mean, {100 * sum(agg.get((tag, lab, kind, s), 0.0) for s in range(5)) / n:+.2f} pt of its cell")
    c = cells[tag]; fa = [x for x in c if x[0]]; ot = [x for x in c if not x[0]]
    print(f"  {tag}: cells using only S3-structured designs {len(fa)} (cost per cell {100*np.mean([x[1] for x in fa]) if fa else float('nan'):+.2f} pt, switches {np.mean([x[2] for x in fa]) if fa else float('nan'):.1f}); others {len(ot)} (cost per cell {100*np.mean([x[1] for x in ot]):+.2f} pt, switches {np.mean([x[2] for x in ot]):.1f})")
