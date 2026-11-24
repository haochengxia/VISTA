"""final-long diagnostics from the decision trace (anonymous aggregates)."""
import json, re
from pathlib import Path
import numpy as np
T = Path("/mydata/vq/final-long"); names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
def chg(line, base): return (sum(int(x) for x in line.split()[3 + 8:]) - base) / base
rows = []
for k, t in enumerate(names):
    d = T / t; meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        W0 = [int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]; base = sum(W0[8:]); s3 = meta["s3fifo"][str(cap)]
        out = (d / f"dg-{cap}.out").read_text().splitlines(); full = (d / f"dgfull-{cap}.out").read_text().splitlines()[0]
        err = (d / f"dg-{cap}.err").read_text().splitlines()
        cut = next(i for i, l in enumerate(err) if l.startswith("screen 0:"))
        def seq(lines):  # live design per window 8..39 (decisions at the split and each window start)
            live, by_w, sw = None, {}, 0
            for l in lines:
                m = re.match(r"decide w=(\d+) live=(-?\d+) leader=(\d+) best=(\d+) live_score=(\d+)", l)
                if not m: continue
                w, cur, lead, best, mine = map(int, m.groups())
                if cur >= 0 and lead != cur: sw += 1
                by_w[w] = lead
            return [by_w.get(w) for w in range(8, 40)], sw
        s1, sw1 = seq(err[:cut]); s2, sw2 = seq(err[cut:])
        # fill windows without a decision line (none expected) with the previous
        for s in (s1, s2):
            for i in range(len(s)):
                if s[i] is None: s[i] = s[i - 1]
        margins = []  # relative lead of the new leader over the live design at v2's switches
        for l in err[cut:]:
            m = re.match(r"decide w=(\d+) live=(-?\d+) leader=(\d+) best=(\d+) live_score=(\d+)", l)
            if m:
                w, cur, lead, best, mine = map(int, m.groups())
                if cur >= 0 and lead != cur and mine > 0: margins.append((mine - best) / mine)
        rows.append(dict(k=k, cap=cap, size=100 * cap / meta["objects"], L3=(meta["3L"][str(cap)] - s3) / s3, v1=chg(out[0], base), v2=chg(out[1], base), full=chg(full, base),
                         sw1=sw1, sw2=sw2, scr_frac=np.mean([x >= 32 for x in s2]), split_scr=s2[0] >= 32, same_split=s1[0] == s2[0], margins=margins,
                         log_wrap=(min(1, max(0.005, 200 / cap)) * meta["split"]) > max(cap, 65536)))
R = rows
A = lambda key: np.array([r[key] for r in R], float)
v1, v2, full, L3, size = A("v1"), A("v2"), A("full"), A("L3"), A("size")
per = lambda x: np.array([x[A("k") == k].mean() for k in range(len(names))])
print(f"final-long {len(R)} cells. mean / median / traces beating 3L:")
for lab, x in (("3L-Cache", L3), ("v1", v1), ("v2 (log bound)", v2), ("v2 unbounded log", full)):
    print(f"  {lab:<18} {100*x.mean():+6.2f}% {100*np.median(x):+6.2f}%  {int((per(x) < per(L3)).sum())}/16")
print(f"switches after the split, per cell: v1 median {np.median(A('sw1')):.0f} mean {A('sw1').mean():.2f}; v2 median {np.median(A('sw2')):.0f} mean {A('sw2').mean():.2f}")
print(f"v2 runs a screened design in {100*A('scr_frac').mean():.0f}% of scored windows; picks a screened design at the split in {int(A('split_scr').sum())}/64 cells; same split pick as v1 in {int(A('same_split').sum())}/64")
d = v2 - v1
for lab, m in (("screened at split", A("split_scr") > 0), ("not screened at split", A("split_scr") == 0)):
    print(f"  {lab:<22} {int(m.sum()):2d} cells: v2 - v1 mean {100*d[m].mean():+.2f} pt; v2 better in {int((d[m] < -1e-4).sum())}, worse in {int((d[m] > 1e-4).sum())}")
for lab, m in (("v2 switches 0-1", A("sw2") <= 1), ("v2 switches 2-3", (A("sw2") >= 2) & (A("sw2") <= 3)), ("v2 switches >=4", A("sw2") >= 4)):
    if m.any(): print(f"  {lab:<22} {int(m.sum()):2d} cells: v2 - v1 mean {100*d[m].mean():+.2f} pt")
for lo, hi in ((0, 0.5), (0.5, 2), (2, 8), (8, 1e9)):
    m = (size >= lo) & (size < hi)
    print(f"  size {lo}-{hi:g}%: {int(m.sum()):2d} cells, v2 - v1 {100*d[m].mean():+.2f} pt; switches v1 {A('sw1')[m].mean():.1f} v2 {A('sw2')[m].mean():.1f}; screened windows {100*A('scr_frac')[m].mean():.0f}%; log wraps in {int(A('log_wrap')[m].sum())}")
mg = np.concatenate([r["margins"] for r in R]) if any(r["margins"] for r in R) else np.array([])
print(f"v2 switches: {mg.size}; leader's sampled lead over the live design at a switch: median {100*np.median(mg):.2f}%, under 1% in {int((mg < 0.01).sum())}, under 0.3% in {int((mg < 0.003).sum())}")
print("total of v2 - v1 over cells, by contribution (pt of the mean): worst 8 cells " + " ".join(f"{100*x/len(R):+.2f}" for x in np.sort(d)[::-1][:8]) + f"; sum {100*d.sum()/len(R):+.2f}")
json.dump(R, open("/tmp/diag_a.json", "w"), default=float)
