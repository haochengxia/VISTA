"""Which split picks break under the takeover? (final-long and train, anonymous)"""
import json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
out = {}
for suite in ("final", "train"):
    src = open("/tmp/diag_b.py").read().split("names = sorted")[0].replace("suite = sys.argv[1]", f"suite = {suite!r}")
    g = {}; exec(src, g)
    T, cold, blocks, parse = g["T"], g["cold"], g["blocks"], g["parse"]
    names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
    rows = []
    for k, t in enumerate(names):
        d = T / t; meta = json.loads((d / "meta.json").read_text())
        for cap in meta["capacities"]:
            get = cold(d, cap); W0 = np.array([int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]], float); base = W0[8:].sum()
            for tag, (job, err, outl) in zip(("v1", "v2"), blocks(d, cap)):
                cands, seq = parse(job, err); d0 = cands[seq[0]]
                H = np.array([int(x) for x in (d / f"dgho-{tag}-{cap}.jobs").read_text() and subprocess.run(["/mydata/vqsim-diag", str(d / "ids.u32"), str(d / f"dgho-{tag}-{cap}.jobs")], capture_output=True, text=True).stdout.split()[2:]], float)
                rows.append(dict(suite=suite, k=k, size=100 * cap / meta["objects"], tag=tag, screened=seq[0] >= 32, cfg=d0,
                                 cost=(H[8:].sum() - get(d0)[8:].sum()) / base, coldchg=(get(d0)[8:].sum() - base) / base))
    out[suite] = rows
def feats(c):
    k = c.count(" st="); f = []
    if " rd=" in c: f.append("rd" + re.search(r"rdmode=(\d+)", c).group(1))
    if " gate=" in c: f.append("gate" + re.search(r"gmode=(\d+)", c).group(1))
    f.append(f"K{k}"); g = re.search(r"gentry=(\d+)", c); f.append("gentry" + g.group(1))
    if re.search(r"st=[0-9.]+,\d+,\d+,\d+,\d+,-?\d+,\d+,[1-9]", c): f.append("delay")
    return ",".join(f)
for suite in ("final", "train"):
    R = [r for r in out[suite] if r["tag"] == "v2"]; cost = np.array([r["cost"] for r in R])
    print(f"{suite}: v2 split picks, takeover cost (handover - cold, pt of the cell): mean {100*cost.mean():+.2f}; cells over +2 pt: {int((cost > 0.02).sum())}/{len(R)}; over +10 pt: {int((cost > 0.10).sum())}")
    by = {}
    for r in R: by.setdefault(feats(r["cfg"]), []).append(r["cost"])
    for f, v in sorted(by.items(), key=lambda x: -np.sum(x[1]))[:8]:
        print(f"    {f:<28} picked in {len(v):2d} cells, mean cost {100*np.mean(v):+6.2f} pt, worst {100*np.max(v):+6.2f}")
    R1 = [r for r in out[suite] if r["tag"] == "v1"]; c1 = np.array([r["cost"] for r in R1])
    print(f"  v1 split picks: mean cost {100*c1.mean():+.2f}; over +2 pt: {int((c1 > 0.02).sum())}")
json.dump(out, open("/tmp/diag_d.json", "w"))
