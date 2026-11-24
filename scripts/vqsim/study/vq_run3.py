"""Cold runs of every design in DESIGNS.jsonl on every cell of a suite: TAG-<cap>.out, one line per design.
  vq_run3.py SUITE_DIR DESIGNS.jsonl TAG WORKERS [CHUNK]
CHUNK (designs per vqsim process, default all) spreads a cell's designs over several processes, so a node with
few cells still uses its cores; the parts are joined in order."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root, designs_file, tag, workers = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
designs = [json.loads(l) for l in designs_file.read_text().splitlines()]
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else len(designs)
parts = [(lo, min(lo + chunk, len(designs))) for lo in range(0, len(designs), chunk)]
def done(out): return out.is_file() and len(out.read_text().splitlines()) == len(designs)
def part(args):
    d, cap, j = args
    lo, hi = parts[j]
    meta = json.loads((d / "meta.json").read_text())
    out = d / f"{tag}-{cap}.out.p{j}"
    if out.is_file() and len(out.read_text().splitlines()) == hi - lo: return
    jobs = d / f"{tag}-{cap}.jobs.p{j}"; jobs.write_text("".join(f"{cap} {meta['split']} 40 {x['config']}\n" for x in designs[lo:hi]))
    with open(out, "w") as fh: subprocess.run(["nice", "-n", "10", "/mydata/vqsim3", str(d / "ids.u32"), str(jobs)], stdout=fh, check=True)
cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
todo = [(d, cap) for d, cap in cells if not done(d / f"{tag}-{cap}.out")]
tasks = sorted(((d, cap, j) for d, cap in todo for j in range(len(parts))), key=lambda t: -(t[0] / "ids.u32").stat().st_size)  # big traces first
with ThreadPoolExecutor(workers) as pool:
    list(pool.map(part, tasks))
for d, cap in todo:
    out = d / f"{tag}-{cap}.out"
    out.write_text("".join((d / f"{tag}-{cap}.out.p{j}").read_text() for j in range(len(parts))))
    assert done(out), out
    for j in range(len(parts)):
        (d / f"{tag}-{cap}.out.p{j}").unlink(); (d / f"{tag}-{cap}.jobs.p{j}").unlink()
print(f"{tag}: {len(cells)} cells, {len(todo)} run now, {len(tasks)} parts", flush=True)
