import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root, designs_file, tag, workers = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
designs = [json.loads(l) for l in designs_file.read_text().splitlines()]
def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text()); out = d / f"{tag}-{cap}.out"
    if out.is_file() and len(out.read_text().splitlines()) == len(designs): return out
    jobs = d / f"{tag}-{cap}.jobs"; jobs.write_text("".join(f"{cap} {meta['split']} 40 {x['config']}\n" for x in designs))
    with open(out, "w") as fh: subprocess.run(["nice", "-n", "10", "/mydata/vqsim3", str(d / "ids.u32"), str(jobs)], stdout=fh, check=True)
    return out
cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool:
    for k, out in enumerate(pool.map(cell, cells), 1): print(f"{k}/{len(cells)} {out.name}", flush=True)
