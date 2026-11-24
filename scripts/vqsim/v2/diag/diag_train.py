"""Contrast on train-long: the v2 configuration (scrB1 jobs: CV 32 + 16 screened, log max(C,65536), wrapmode 1) and v1 (prbcv), rerun with the decision trace."""
import json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
T = Path("/mydata/vq/train-long"); VQ = "/mydata/vqsim-diag"; env = dict(os.environ, VQSIM_TRACE="1", VQSIM_SCREEN_LOG="1")
names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
def switches(err):
    sw, mg = 0, []
    for l in err.splitlines():
        m = re.match(r"decide w=(\d+) live=(-?\d+) leader=(\d+) best=(\d+) live_score=(\d+)", l)
        if m:
            w, cur, lead, best, mine = map(int, m.groups())
            if cur >= 0 and lead != cur: sw += 1; mg.append((mine - best) / max(mine, 1))
    return sw, mg
def cell(a):
    d, cap = a
    out = {}
    for tag in ("scrB1", "prbcv"):
        r = subprocess.run(["nice", "-n", "10", VQ, str(d / "ids.u32"), str(d / f"{tag}-{cap}.jobs")], capture_output=True, text=True, check=True, env=env)
        assert r.stdout.splitlines()[0] == (d / f"{tag}-{cap}.out").read_text().splitlines()[0], (d, cap, tag)
        out[tag] = (r.stdout.splitlines()[0], switches(r.stderr))
    meta = json.loads((d / "meta.json").read_text())
    W0 = [int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]; base = sum(W0[8:])
    c = lambda l: (sum(int(x) for x in l.split()[3 + 8:]) - base) / base
    return c(out["prbcv"][0]), c(out["scrB1"][0]), out["prbcv"][1][0], out["scrB1"][1][0], out["scrB1"][1][1], 100 * cap / meta["objects"]
cells = [(d, cap) for d in sorted(T.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(20) as pool: got = list(pool.map(cell, cells))
v1 = np.array([g[0] for g in got]); v2 = np.array([g[1] for g in got]); s1 = np.array([g[2] for g in got]); s2 = np.array([g[3] for g in got]); size = np.array([g[5] for g in got])
mg = np.concatenate([g[4] for g in got]); d = v2 - v1
print(f"train-long {len(got)} cells (reruns identical to the logged runs): v1 {100*v1.mean():+.2f}%, v2 config {100*v2.mean():+.2f}%")
print(f"switches per cell: v1 median {np.median(s1):.0f} mean {s1.mean():.2f}; v2 median {np.median(s2):.0f} mean {s2.mean():.2f}; v2 switch leads: median {100*np.median(mg):.2f}%, under 1% in {int((mg < 0.01).sum())}/{mg.size}")
for lab, m in (("v2 switches 0-1", s2 <= 1), ("v2 switches 2-3", (s2 >= 2) & (s2 <= 3)), ("v2 switches >=4", s2 >= 4)):
    print(f"  {lab:<18} {int(m.sum()):2d} cells: v2 - v1 mean {100*d[m].mean():+.2f} pt")
for lo, hi in ((0, 0.5), (0.5, 2), (2, 8), (8, 1e9)):
    m = (size >= lo) & (size < hi); print(f"  size {lo}-{hi:g}%: {int(m.sum()):2d} cells, v2 - v1 {100*d[m].mean():+.2f} pt; switches v1 {s1[m].mean():.1f} v2 {s2[m].mean():.1f}")
