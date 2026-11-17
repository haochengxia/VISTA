"""Online re-selection over the r1 sweep: in each unseen window, run the design that leads so far (free switching).
Upper-bound style diagnostic: shadow runs of every candidate from the start."""
import json, sys
from pathlib import Path
import numpy as np
root, designs_file = Path(sys.argv[1]), Path(sys.argv[2])
designs = [json.loads(l) for l in designs_file.read_text().splitlines()]
fam = np.array([d["family"] for d in designs])
W, L3, traces = [], [], []
for d in sorted(root.iterdir()):
    if not (d / "meta.json").is_file(): continue
    meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        rows = np.array([[int(x) for x in l.split()] for l in (d / f"r1-{cap}.out").read_text().splitlines()])
        W.append(rows[:, 2:].astype(float)); s3 = meta["s3fifo"][str(cap)]; L3.append((meta["3L"][str(cap)] - s3) / s3); traces.append(d.name)
W = np.array(W); L3 = np.array(L3); n, D, T = W.shape
base = W[:, 0, 8:].sum(1)
def report(name, misses):
    x = (misses - base) / base
    print(f"  {name:<46} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  beats 3L {int((x < L3).sum())}/{n}  worse {int((x > 0.0005).sum())}")
def run(mask, horizon=None):
    total = np.zeros(n); switches = np.zeros(n)
    prev = None
    for w in range(8, T):
        lo = 2 if horizon is None else max(2, w - horizon)
        s = W[:, :, lo:w].sum(2); s[:, ~mask] = np.inf
        p = s.argmin(1)
        total += W[np.arange(n), p, w]
        if prev is not None: switches += p != prev
        prev = p
    return total, switches
print(f"{n} cells, {D} designs, {T} windows (unseen = 8..{T-1})")
static = W[:, :, 2:8].sum(2).argmin(1); report("static pick (windows 2-7)", W[np.arange(n), static, 8:].sum(1))
report("hindsight static", W[:, :, 8:].sum(2).min(1))
report("hindsight per window", W[:, :, 8:].min(1).sum(1))
for label, mask in (("all", np.ones(D, bool)), ("s3grid", np.isin(fam, ["s3fifo", "s3grid"])), ("s3grid+s4like", np.isin(fam, ["s3fifo", "s3grid", "s4like"]))):
    for h in (None, 8, 4):
        tot, sw = run(mask, h)
        report(f"leader {label}, {'cumulative' if h is None else f'last {h} windows'}", tot)
        print(f"      switches: median {np.median(sw):.0f}, mean {sw.mean():.1f}")
