"""Where does the online selector lose against its idealized version: at the split, or at later switches? Train only.
Per cell, the CV portfolio of its trace (first K designs) and the leader the selector follows with exact shadows
(fewest cold-run misses in windows 2..w-1, at the split and each later window). Its windows 8-39 are scored three
ways, vs S3-FIFO:
  cold       each window as in the leader's cold run (no takeover at all: the idealized selector)
  handover   each window as in the leader's handover run (S3-FIFO's cache taken over at the split; later switches free)
  live       the selector itself with exact shadows (TAG-<cap>.out line J from vq_portfolio_real.py, rate 1)
  vq_switchcost.py SUITE_DIR PORTFOLIO_CV.json K LIVE_TAG LIVE_LINE WORKERS SOURCE=DESIGNS.jsonl ...
(SOURCE: the cold-run outputs SOURCE-<cap>.out from vq_run3.py and their design lists; prints aggregates only)"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

root, pfile, K, live_tag, live_line, workers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
where = {}  # config -> (source, line)
for spec in sys.argv[7:]:
    name, _, path = spec.partition("=")
    for i, line in enumerate(open(path)):
        where.setdefault(json.loads(line)["config"], (name, i))
by_trace = json.load(open(pfile))["by_trace"]
vqsim = os.environ.get("VQSIM", "/mydata/vqsim3")
tag = f"swc{K}-{Path(pfile).stem}"

def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text())
    port = by_trace[d.name][:K]
    out = d / f"{tag}-{cap}.out"
    if not (out.is_file() and len(out.read_text().splitlines()) == len(port)):
        jobs = d / f"{tag}-{cap}.jobs"; jobs.write_text("".join(f"{cap} {meta['split']} 40 {c} handover=1\n" for c in port))
        out.write_text(subprocess.run(["nice", "-n", "10", vqsim, str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout)
    H = np.array([[int(x) for x in l.split()[2:]] for l in out.read_text().splitlines()], float)  # designs x 40
    lines = {}
    W = np.zeros_like(H)
    for j, c in enumerate(port):
        src, i = where[c]
        if src not in lines: lines[src] = (d / f"{src}-{cap}.out").read_text().splitlines()
        W[j] = [int(x) for x in lines[src][i].split()[2:]]
    base = W[0, 8:].sum()  # S3-FIFO (every portfolio starts with it)
    C = np.cumsum(W[:, 2:], axis=1)[:, 5:37]  # misses in windows 2..w-1, for w = 8..39
    lead = C.argmin(0)  # first minimum: ties go to the earlier member
    cold = W[lead, np.arange(8, 40)].sum(); ho = H[lead, np.arange(8, 40)].sum()
    live = sum(int(x) for x in (d / f"{live_tag}-{cap}.out").read_text().splitlines()[live_line].split()[3 + 8:])
    first_cost = (H[lead[0], 8:].sum() - W[lead[0], 8:].sum()) / base  # the first pick, kept to the end: takeover vs cold
    fixed = H[lead[0], 8:].sum()  # pick at the split, never switch
    return (cold - base) / base, (ho - base) / base, (live - base) / base, int((np.diff(lead) != 0).sum()), first_cost, (fixed - base) / base

cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool:
    got = np.array(list(pool.map(cell, cells)))
print(f"{root.name}: {len(cells)} cells; CV portfolio {Path(pfile).name}, K={K}; windows 8-39 vs S3-FIFO; exact shadows")
for j, label in enumerate(("cold (idealized)", "handover (split paid)", "live (split + switches)")):
    print(f"  {label:<26} mean {100*got[:, j].mean():+6.2f}%  median {100*np.median(got[:, j]):+6.2f}%")
print(f"  split cost {100*(got[:, 1] - got[:, 0]).mean():+.2f} pt, switch cost {100*(got[:, 2] - got[:, 1]).mean():+.2f} pt (mean per cell);"
      f" leader changes after the split: median {np.median(got[:, 3]):.0f}, mean {got[:, 3].mean():.1f}; cells with none {int((got[:, 3] == 0).sum())}")
none = got[:, 3] == 0
print(f"  cells with no later switch ({none.sum()}): handover - cold {100*(got[none, 1] - got[none, 0]).mean():+.2f} pt, live - handover {100*(got[none, 2] - got[none, 1]).mean():+.2f} pt")
print(f"  first pick kept to the end, takeover vs cold: mean {100*got[:, 4].mean():+.2f} pt, worst {100*got[:, 4].max():+.2f} pt")
print(f"  pick at the split and never switch: mean {100*got[:, 5].mean():+6.2f}%  median {100*np.median(got[:, 5]):+6.2f}%; better than live in {int((got[:, 5] < got[:, 2] - 1e-4).sum())} cells, worse in {int((got[:, 5] > got[:, 2] + 1e-4).sum())}")
