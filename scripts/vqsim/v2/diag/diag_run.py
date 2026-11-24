"""Diagnostics on final-long (no tuning): rerun the exact round-2 jobs with the decision trace, and v2 with an unbounded log."""
import json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
T = Path("/mydata/vq/final-long"); VQ = "/mydata/vqsim-diag"
env = dict(os.environ, VQSIM_TRACE="1", VQSIM_SCREEN_LOG="1")
def cell(a):
    d, cap = a
    jobs = (d / f"v2fl-{cap}.jobs").read_text()
    r = subprocess.run(["nice", "-n", "10", VQ, str(d / "ids.u32"), str(d / f"v2fl-{cap}.jobs")], capture_output=True, text=True, check=True, env=env)
    (d / f"dg-{cap}.out").write_text(r.stdout); (d / f"dg-{cap}.err").write_text(r.stderr)
    same = r.stdout == (d / f"v2fl-{cap}.out").read_text()
    blocks = jobs.split("end\n"); v2 = blocks[1] + "end\n"
    head, rest = v2.split("\n", 1); head = " ".join(t for t in head.split() if not t.startswith(("logcap=", "wrapmode=")))
    (d / f"dgfull-{cap}.jobs").write_text(head + "\n" + rest)
    r2 = subprocess.run(["nice", "-n", "10", VQ, str(d / "ids.u32"), str(d / f"dgfull-{cap}.jobs")], capture_output=True, text=True, check=True, env=env)
    (d / f"dgfull-{cap}.out").write_text(r2.stdout); (d / f"dgfull-{cap}.err").write_text(r2.stderr)
    return same
cells = [(d, cap) for d in sorted(T.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(32) as pool: same = list(pool.map(cell, cells))
print(f"{len(cells)} cells; instrumented rerun identical to round 2: {sum(same)}/{len(cells)}")
