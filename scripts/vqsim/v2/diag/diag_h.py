"""When is the takeover cost paid? Per window: live misses - the live design's cold-run misses, by windows since the last design change (the split counts)."""
import json, sys
from pathlib import Path
import numpy as np
for suite in ("final", "train"):
    src = open("/tmp/diag_b.py").read().split("names = sorted")[0].replace("suite = sys.argv[1]", f"suite = {suite!r}"); g = {}; exec(src, g)
    T = g["T"]; names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
    acc = {tag: {} for tag in ("v1", "v2")}; tot = {tag: 0.0 for tag in acc}; nb = 0
    for k, t in enumerate(names):
        d = T / t; meta = json.loads((d / "meta.json").read_text())
        for cap in meta["capacities"]:
            get = g["cold"](d, cap); W0 = np.array([int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]], float); base = W0[8:].sum(); nb += 1
            size = 100 * cap / meta["objects"]; cls = 0 if size < 0.5 else 1 if size < 2 else 2 if size < 8 else 3
            for tag, (job, err, outl) in zip(("v1", "v2"), g["blocks"](d, cap)):
                cands, seq = g["parse"](job, err); M = np.array([get(c) for c in cands]); live = np.array([int(x) for x in outl.split()[3:]], float)
                since = 0
                for i in range(32):
                    if i > 0 and seq[i] != seq[i - 1]: since = 0
                    delta = (live[8 + i] - M[seq[i], 8 + i]) / base
                    key = (min(since, 4), "split" if i - since == 0 else "switch")
                    acc[tag].setdefault(key, [0.0, 0]); acc[tag][key][0] += delta; acc[tag][key][1] += 1
                    acc[tag].setdefault(("cls", cls), [0.0, 0]); acc[tag][("cls", cls)][0] += delta
                    tot[tag] += delta; since += 1
    print(f"{suite}: takeover cost by windows since the last design change (pt of the suite mean; windows counted)")
    for tag in ("v1", "v2"):
        parts = []
        for origin in ("split", "switch"):
            for s in range(5):
                v = acc[tag].get((s, origin))
                if v: parts.append(f"{origin}+{s}{'+' if s == 4 else ''}: {100*v[0]/nb:+.2f} ({v[1]})")
        print(f"  {tag} total {100*tot[tag]/nb:+.2f}: " + "; ".join(parts))
        print(f"      by size class <0.5/0.5-2/2-8/>=8%: " + " ".join(f"{100*acc[tag].get(('cls', c), [0])[0]/nb:+.2f}" for c in range(4)))
