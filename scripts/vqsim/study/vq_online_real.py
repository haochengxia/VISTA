"""Online re-choice made real (train-long, 100 cells). Per cell, vqsim online blocks:
  sched  the idealized leader's schedule (from the r1 cold windows) replayed on a live cache: switching cost
  mini   sampled miniature simulations choose the leader, at several rates / candidate sets
Scores windows 8..39 vs S3-FIFO's (r1 design 0)."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
root, designs_file, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
designs = [json.loads(l) for l in designs_file.read_text().splitlines()]
fam = np.array([d["family"] for d in designs])
small = [i for i in range(len(designs)) if fam[i] in ("s3fifo", "s3grid", "s4like")]
VARIANTS = [("sched-all", None, None), ("mini-all r=1% min=1000", 0.01, 1000), ("mini-all r=0.1% min=200", 0.001, 200), ("mini-s3s4 r=1% min=1000", 0.01, 1000)]
def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text()); split = meta["split"]
    out = d / f"online-{cap}.out"
    if out.is_file() and len(out.read_text().splitlines()) == len(VARIANTS): return d.name, cap, out.read_text().splitlines()
    cold = np.array([[int(x) for x in l.split()[2:]] for l in (d / f"r1-{cap}.out").read_text().splitlines()])
    sched = []
    for w in range(8, 40):
        lead = int(cold[:, 2:w].sum(1).argmin())
        if not sched or sched[-1][1] != lead: sched.append((w, lead))
    blocks = []
    for name, rate, mn in VARIANTS:
        cands = small if "s3s4" in name else list(range(len(designs)))
        opts = "schedule=" + ",".join(f"{w}:{i}" for w, i in sched) if rate is None else f"rate={rate} min={mn}"
        blocks.append(f"online {cap} {split} 40 {opts}\n" + "".join(designs[i]["config"] + "\n" for i in cands) + "end\n")
    jobs = d / f"online-{cap}.jobs"; jobs.write_text("".join(blocks))
    res = subprocess.run(["nice", "-n", "10", "/mydata/vqsim3", str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout
    out.write_text(res); return d.name, cap, res.splitlines()
cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
rows = {v[0]: [] for v in VARIANTS}; l3s = []; ideal = []; static = []; sw = {v[0]: [] for v in VARIANTS}
with ThreadPoolExecutor(workers) as pool:
    for k, (name, cap, lines) in enumerate(pool.map(cell, cells), 1):
        d = root / name; meta = json.loads((d / "meta.json").read_text())
        cold = np.array([[int(x) for x in l.split()[2:]] for l in (d / f"r1-{cap}.out").read_text().splitlines()])
        base = cold[0, 8:].sum(); s3 = meta["s3fifo"][str(cap)]; l3s.append((meta["3L"][str(cap)] - s3) / s3)
        tot = 0
        for w in range(8, 40): tot += cold[int(cold[:, 2:w].sum(1).argmin()), w]
        ideal.append((tot - base) / base); p = cold[:, 2:8].sum(1).argmin(); static.append((cold[p, 8:].sum() - base) / base)
        for (vname, *_), line in zip(VARIANTS, lines):
            x = [int(v) for v in line.split()]; rows[vname].append((sum(x[3 + 8:]) - base) / base); sw[vname].append(x[2])
        print(f"{k}/{len(cells)} {name} {cap}", flush=True)
L3 = np.array(l3s)
def rep(label, xs):
    x = np.array(xs); print(f"  {label:<28} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  beats 3L {int((x < L3).sum())}/{len(x)}  worse {int((x > 0.0005).sum())}")
print(f"{len(L3)} cells (windows 8-39 vs S3-FIFO)"); rep("3L-Cache", L3); rep("static pick (cold)", static); rep("idealized leader (cold)", ideal)
for v in rows: rep(v, rows[v]); print(f"      switches median {np.median(sw[v]):.0f} mean {np.mean(sw[v]):.1f}")
