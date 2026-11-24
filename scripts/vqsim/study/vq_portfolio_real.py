"""Realistic online re-choice over a fixed portfolio (first K designs of portfolio-<p>.json): S3-FIFO serves the prefix;
at the split and each later window the live cache switches to the candidate with the fewest sampled misses since
window 2 (K mini-sims at rate r, mini cache >= MIN objects), margin 0. Scored on windows 8-39 vs S3-FIFO.
Selector memory ~ K * max(r*C, MIN) entries (+ their ghosts), next to the cache's C.
usage: vq_portfolio_real.py SUITE PORTFOLIO_JSON TAG WORKERS K:RATE[:MIN[:MARGIN[:TAKEOVER[:SCREEN[:LOGC[:LOGMIN[:WRAP[:TRIAL[:RESEED[:REPLAY]]]]]]]]]]] ...
SCREEN=M: at the split, also screen every design of POOL on the logged sampled prefix and add the best M as candidates
(vqsim pool=, screen=); the selector also holds that log (~ max(r, MIN/C) * split entries), bounded to the last
max(LOGC * C, LOGMIN) entries if LOGC > 0 (vqsim logcap=; WRAP: its wrapmode). POOL (env): a file of configs, one per line, or
cv:PORTFOLIO_CV.json:LO:HI for members LO..HI-1 of each trace's own CV portfolio.
TRIAL=H: try each switch first on a miniature mirror of the live cache, H decision points back (vqsim trial=); the mirror,
its H snapshots and the replay add (H + 2) miniatures to the selector's memory. TRIAL=-H: the same, but the first takeover
is not tried (trialsplit=0). RESEED=1: at the split every miniature takes over the mirror's cache (vqsim reseed=1;
the mirror adds one miniature). REPLAY=L: takeovers rebuild the next design's state from the last L*C requests
(vqsim replay=L; the live cache keeps that log of ids)."""
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
    sc = int(parts[5]) if len(parts) > 5 else 0
    lc = float(parts[6]) if len(parts) > 6 else 0.0; lmin = int(parts[7]) if len(parts) > 7 else 0
    wm = int(parts[8]) if len(parts) > 8 else 0; tr = int(parts[9]) if len(parts) > 9 else 0
    rs = int(parts[10]) if len(parts) > 10 else 0; rp = float(parts[11]) if len(parts) > 11 else 0.0
    VARIANTS.append((k, r, m, margin, tk, sc, lc, lmin, wm, tr, rs, rp))
POOL = os.environ.get("POOL", "")
def pool_of(d, cap):  # one file per cell: the cells of a trace run in parallel
    if not POOL.startswith("cv:"): return POOL
    _, path, lo, hi = POOL.split(":"); f = d / f"pool-{Path(path).stem}-{lo}-{hi}-{cap}.txt"
    f.write_text("".join(c + "\n" for c in json.load(open(path))["by_trace"][d.name][int(lo):int(hi)])); return f
def logcap(cap, lc, lmin): return max(int(lc * cap), lmin) if lc > 0 else 0
def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text()); split = meta["split"]
    out = d / f"{tag}-{cap}.out"
    if not (out.is_file() and len(out.read_text().splitlines()) == len(VARIANTS)):
        jobs = d / f"{tag}-{cap}.jobs"
        mine = by_trace[d.name] if by_trace else port
        pool = pool_of(d, cap) if any(v[5] for v in VARIANTS) else ""
        jobs.write_text("".join(f"online {cap} {split} 40 rate={r} min={m} margin={margin}" + (f" takeover={tk}" if tk else "") + (f" pool={pool} screen={sc}" if sc else "")
                            + (f" logcap={logcap(cap, lc, lmin)}" if sc and lc > 0 else "") + (f" wrapmode={wm}" if sc and lc > 0 and wm else "") + (f" trial={abs(tr)}" if tr else "") + (" trialsplit=0" if tr < 0 else "") + (f" reseed={rs}" if rs else "") + (f" replay={rp:g}" if rp else "")
                            + "\n" + "".join(c + "\n" for c in mine[:k]) + "end\n" for k, r, m, margin, tk, sc, lc, lmin, wm, tr, rs, rp in VARIANTS))
        out.write_text(subprocess.run(["nice", "-n", "10", os.environ.get("VQSIM", "/mydata/vqsim3"), str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout)
    W0 = [int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]
    base = sum(W0[8:]); s3 = meta["s3fifo"][str(cap)]
    lines = out.read_text().splitlines()
    mem = [(k + sc + (abs(tr) + 2 if tr else 1 if rs else 0)) * max(min(r * cap, cap), min(m, cap)) / cap for k, r, m, _, _, sc, _, _, _, tr, rs, _ in VARIANTS]
    log = [min(min(1, max(r, m / cap)) * split, logcap(cap, lc, lmin) or split) / cap if sc else 0 for k, r, m, _, _, sc, lc, lmin, _, _, _, _ in VARIANTS]
    npool = len(Path(pool_of(d, cap)).read_text().splitlines()) if any(v[5] for v in VARIANTS) else 0
    replay = [(npool + k + sc) * g * cap / meta["requests"] for g, (k, *_rest) in zip(log, VARIANTS)]  # miniature steps per request, one-off
    objects = meta.get("objects") or int(np.memmap(d / "ids.u32", dtype=np.uint32, mode="r").max()) + 1
    return 100 * cap / objects, (meta["3L"][str(cap)] - s3) / s3, [(sum(int(x) for x in l.split()[3 + 8:]) - base) / base for l in lines], [int(l.split()[2]) for l in lines], mem, d.name, log, replay
cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool: got = list(pool.map(cell, cells))
Z = np.array([g[0] for g in got]); L3 = np.array([g[1] for g in got]); R = np.array([g[2] for g in got]); S = np.array([g[3] for g in got]); M = np.array([g[4] for g in got])
T = np.array([g[5] for g in got]); traces = sorted(set(T)); LG = np.array([g[6] for g in got]); RP = np.array([g[7] for g in got])
per = lambda x: np.array([x[T == t].mean() for t in traces])  # a trace's mean change over its cells
def rep(label, x): print(f"  {label:<22} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  beats 3L {int((x < L3).sum())}/{len(x)}  worse than S3-FIFO {int((x > 0.0005).sum())}  traces beating 3L {int((per(x) < per(L3)).sum())}/{len(traces)}")
print(f"{root.name}: {len(L3)} cells, windows 8-39 vs S3-FIFO; portfolio {pfile}"); rep("3L-Cache", L3)
for j, (k, r, m, margin, tk, sc, lc, lmin, wm, tr, rs, rp) in enumerate(VARIANTS):
    rep(f"K={k} r={r:g} min={m} m={margin:g} t={tk}" + (f" +{sc}" if sc else "") + (f" log {lc:g}C/{lmin}" + (f" w{wm}" if wm else "") if sc and lc > 0 else "")
        + (f" trial{tr}" if tr else "") + (" reseed" if rs else "") + (f" replay{rp:g}" if rp else ""), R[:, j])
    print(f"      switches median {np.median(S[:, j]):.0f}; selector entries / C: median {np.median(M[:, j]):.2f}, max {M[:, j].max():.1f}"
          + (f"; prefix log / C: median {np.median(LG[:, j]):.1f}, max {LG[:, j].max():.0f}; replay steps / request: median {np.median(RP[:, j]):.2f}, max {RP[:, j].max():.0f}" if sc else ""))
for lo, hi, label in ((0, 0.5, "<0.5%"), (0.5, 2, "0.5-2%"), (2, 8, "2-8%"), (8, 1e9, ">=8%")):
    m = (Z >= lo) & (Z < hi)
    if m.any(): print(f"    size {label:<7} ({m.sum():3d} cells): 3L {100*L3[m].mean():+6.2f}%  " + "  ".join(f"v{j}: {100*R[m, j].mean():+6.2f}%" for j in range(len(VARIANTS))))
