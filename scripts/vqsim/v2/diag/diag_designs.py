import json, random, re
from pathlib import Path
T = Path("/mydata/vq/final-long")
pool = open("/users/Haocheng/VISTA/scripts/vqsim/v2/pool.txt").read().splitlines()
v1 = json.load(open("/users/Haocheng/VISTA/scripts/vqsim/v1/portfolio.json"))["portfolio"]
scr = set()
for f in T.glob("*/dg-*.err"):
    scr |= {int(x) for x in re.findall(r"file line (\d+)", f.read_text())}
rnd = random.Random(0).sample(range(len(pool)), 300)
lines = sorted(scr | set(rnd))
designs = list(v1) + [pool[i] for i in lines if pool[i] not in v1]
open("/mydata/vq/fl_diag_designs.jsonl", "w").write("".join(json.dumps({"name": "d", "config": c}) + "\n" for c in designs))
json.dump({"v1": len(v1), "screened_lines": sorted(scr), "random_lines": sorted(rnd)}, open("/mydata/vq/fl_diag_meta.json", "w"))
print(len(designs), "designs;", len(scr), "distinct screened;", len(set(rnd) & scr), "of the random 300 were screened somewhere")
