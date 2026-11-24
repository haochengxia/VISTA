import json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
T = Path("/mydata/vq/train-long"); VQ = "/mydata/vqsim-diag"; env = dict(os.environ, VQSIM_TRACE="1", VQSIM_SCREEN_LOG="1")
def cell(a):
    d, cap = a
    for tag, new in (("prbcv", "dgv1"), ("scrB1", "dgv2")):
        r = subprocess.run(["nice", "-n", "10", VQ, str(d / "ids.u32"), str(d / f"{tag}-{cap}.jobs")], capture_output=True, text=True, check=True, env=env)
        (d / f"{new}-{cap}.out").write_text(r.stdout); (d / f"{new}-{cap}.err").write_text(r.stderr)
cells = [(d, cap) for d in sorted(T.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(16) as pool: list(pool.map(cell, cells))
print("saved")
