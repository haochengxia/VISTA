"""Realistic online re-choice over a fixed portfolio (first K designs of portfolio-<p>.json): S3-FIFO serves the prefix;
at the split and each later window the live cache switches to the candidate with the fewest sampled misses since
window 2 (K mini-sims at rate r, mini cache >= MIN objects), margin 0. Scored on windows 8-39 vs S3-FIFO.
Selector memory ~ K * max(r*C, MIN) entries (+ their ghosts), next to the cache's C.
usage: vq_portfolio_real.py SUITE PORTFOLIO_JSON TAG WORKERS K:RATE[:MIN] ..."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
root, pfile, tag, workers = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
spec = json.load(open(pfile))
by_trace = spec.get("by_trace")  # cross-validation: each trace's portfolio was chosen without it
port = spec.get("portfolio")
VARIANTS = []
for spec in sys.argv[5:]:
    parts = spec.split(":"); k, r = int(parts[0]), float(parts[1]); m = int(parts[2]) if len(parts) > 2 else 1000
    margin = float(parts[3]) if len(parts) > 3 else 0.0; tk = int(parts[4]) if len(parts) > 4 else 0
    VARIANTS.append((k, r, m, margin, tk))
def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text()); split = meta["split"]
    out = d / f"{tag}-{cap}.out"
    if not (out.is_file() and len(out.read_text().splitlines()) == len(VARIANTS)):
        jobs = d / f"{tag}-{cap}.jobs"
        mine = by_trace[d.name] if by_trace else port
        jobs.write_text("".join(f"online {cap} {split} 40 rate={r} min={m} margin={margin}" + (f" takeover={tk}" if tk else "") + "\n" + "".join(c + "\n" for c in mine[:k]) + "end\n" for k, r, m, margin, tk in VARIANTS))
        out.write_text(subprocess.run(["nice", "-n", "10", os.environ.get("VQSIM", "/mydata/vqsim3"), str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout)
    W0 = [int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]
    base = sum(W0[8:]); s3 = meta["s3fifo"][str(cap)]
    lines = out.read_text().splitlines()
    mem = [k * max(min(r * cap, cap), min(m, cap)) / cap for k, r, m, *_ in VARIANTS]
    objects = meta.get("objects") or int(np.memmap(d / "ids.u32", dtype=np.uint32, mode="r").max()) + 1
    return 100 * cap / objects, (meta["3L"][str(cap)] - s3) / s3, [(sum(int(x) for x in l.split()[3 + 8:]) - base) / base for l in lines], [int(l.split()[2]) for l in lines], mem
cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool: got = list(pool.map(cell, cells))
Z = np.array([g[0] for g in got]); L3 = np.array([g[1] for g in got]); R = np.array([g[2] for g in got]); S = np.array([g[3] for g in got]); M = np.array([g[4] for g in got])
def rep(label, x): print(f"  {label:<22} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  beats 3L {int((x < L3).sum())}/{len(x)}  worse than S3-FIFO {int((x > 0.0005).sum())}")
print(f"{root.name}: {len(L3)} cells, windows 8-39 vs S3-FIFO; portfolio {pfile}"); rep("3L-Cache", L3)
for j, (k, r, m, margin, tk) in enumerate(VARIANTS):
    rep(f"K={k} r={r:g} min={m} m={margin:g} t={tk}", R[:, j])
    print(f"      switches median {np.median(S[:, j]):.0f}; selector entries / C: median {np.median(M[:, j]):.2f}, max {M[:, j].max():.1f}")
for lo, hi, label in ((0, 0.5, "<0.5%"), (0.5, 2, "0.5-2%"), (2, 8, "2-8%"), (8, 1e9, ">=8%")):
    m = (Z >= lo) & (Z < hi)
    if m.any(): print(f"    size {label:<7} ({m.sum():3d} cells): 3L {100*L3[m].mean():+6.2f}%  " + "  ".join(f"v{j}: {100*R[m, j].mean():+6.2f}%" for j in range(len(VARIANTS))))
