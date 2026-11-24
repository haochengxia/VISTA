"""The pre-registered fresh-long statistics (PREREG-fresh.md), from vq_portfolio_real.py's per-cell outputs.

usage: fresh_stats.py SUITE_DIR TAG LABEL...   (one label per variant line of TAG-<cap>.out, in order; the first is the candidate)

Per cell, as vq_portfolio_real.py: change in misses vs S3-FIFO (r1 line 0) over windows 8-39; 3L-Cache's from meta.json.
Prints, for 3L-Cache and every variant: mean, median, cells beating 3L-Cache, cells worse than S3-FIFO, traces beating
3L-Cache; the bar's verdict for the candidate; a bootstrap over traces (10,000 resamples, seed 0) of the candidate's
mean minus 3L-Cache's. Trace names are never printed."""
import json, sys
from pathlib import Path
import numpy as np

root, tag, labels = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
rows = []  # (trace index, 3L change, [variant changes])
dirs = sorted(d for d in root.iterdir() if (d / "meta.json").is_file())
for ti, d in enumerate(dirs):
    meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        W0 = [int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]; base = sum(W0[8:])
        s3 = meta["s3fifo"][str(cap)]
        lines = (d / f"{tag}-{cap}.out").read_text().splitlines()
        assert len(lines) == len(labels), (ti, cap, len(lines))
        rows.append((ti, (meta["3L"][str(cap)] - s3) / s3, [(sum(int(x) for x in l.split()[3 + 8:]) - base) / base for l in lines]))
T = np.array([r[0] for r in rows]); L3 = np.array([r[1] for r in rows]); R = np.array([r[2] for r in rows])
traces = np.unique(T)
per = lambda x: np.array([x[T == t].mean() for t in traces])


def line(label, x):
    return (f"  {label:<34} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  cells beating 3L {int((x < L3).sum())}/{len(x)}"
            f"  worse than S3-FIFO {int((x > 0.0005).sum())}  traces beating 3L {int((per(x) < per(L3)).sum())}/{len(traces)}")


print(f"{root.name}: {len(traces)} traces, {len(L3)} cells, windows 8-39 vs S3-FIFO")
print(f"  {'3L-Cache':<34} mean {100*L3.mean():+6.2f}%  median {100*np.median(L3):+6.2f}%")
for j, lab in enumerate(labels): print(line(lab, R[:, j]))
c = R[:, 0]
ok = [c.mean() < L3.mean(), np.median(c) < np.median(L3), (c < L3).sum() > len(c) / 2]
print(f"\nbar for {labels[0]}: (1) mean below 3L-Cache's: {'yes' if ok[0] else 'NO'}; (2) median below 3L-Cache's: {'yes' if ok[1] else 'NO'};"
      f" (3) beats 3L-Cache in more than half of the cells: {'yes' if ok[2] else 'NO'} -> {'PASS' if all(ok) else 'FAIL'}")
rng = np.random.default_rng(0); diffs = []
for _ in range(10000):
    pick = rng.choice(traces, size=len(traces), replace=True)
    m = np.concatenate([np.flatnonzero(T == t) for t in pick])
    diffs.append(c[m].mean() - L3[m].mean())
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"bootstrap over traces (10,000, seed 0): candidate mean - 3L-Cache mean = {100*(c.mean()-L3.mean()):+.2f} pt, 95% interval [{100*lo:+.2f}, {100*hi:+.2f}] pt;"
      f" share of resamples with the candidate ahead: {100*np.mean(np.array(diffs) < 0):.1f}%")
