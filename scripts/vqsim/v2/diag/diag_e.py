import json, re
from pathlib import Path
import numpy as np
D = json.load(open("/tmp/diag_d.json"))["final"]
R = sorted([r for r in D if r["tag"] == "v2"], key=lambda r: -r["cost"])[:3]
src = open("/tmp/diag_b.py").read().split("names = sorted")[0].replace("suite = sys.argv[1]", "suite = 'final'"); g = {}; exec(src, g)
T = g["T"]; names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
for r in R:
    t = names[r["k"]]; d = T / t; meta = json.loads((d / "meta.json").read_text())
    cap = min(meta["capacities"], key=lambda c: abs(100 * c / meta["objects"] - r["size"]))
    get = g["cold"](d, cap); bl = g["blocks"](d, cap)
    W0 = np.array([int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]], float)
    cands, seq = g["parse"](*bl[1][:2]); c1, s1 = g["parse"](*bl[0][:2])
    H = np.array([int(x) for x in __import__("subprocess").run(["/mydata/vqsim-diag", str(d / "ids.u32"), str(d / f"dgho-v2-{cap}.jobs")], capture_output=True, text=True).stdout.split()[2:]], float)
    live2 = np.array([int(x) for x in bl[1][2].split()[3:]], float); live1 = np.array([int(x) for x in bl[0][2].split()[3:]], float)
    C0 = get(r["cfg"])
    print(f"cell t{r['k']:02d} size {r['size']:.2f}% C={cap}: split pick takeover cost {100*r['cost']:+.1f} pt; its cold change {100*r['coldchg']:+.2f}%")
    print(f"  config: {r['cfg']}")
    print(f"  live design index per window 8..39 (>=32 screened): {seq}")
    print(f"  v1's sequence: {s1}")
    rel = lambda x: " ".join(f"{100*(a-b)/b:+.0f}" for a, b in zip(x[8:24], W0[8:24]))
    print(f"  per window vs S3-FIFO cold (%), windows 8..23:\n    pick cold     {rel(C0)}\n    pick handover {rel(H)}\n    v2 live       {rel(live2)}\n    v1 live       {rel(live1)}")
