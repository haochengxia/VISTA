"""Can the online selector use gated designs? Train only, aggregates only.
Per train-long cell, the v1 online selector (rate 0.005, min 200, margin 0, taking over at the split) over the
frozen v1 portfolio as is, and with a gated twin of each of its designs added (gate in front of the main stage;
mode "both" = rescue + filter, or "filter" = W-TinyLFU's admission test only). Same portfolio in every arm
(chosen on train, so in-sample alike); scored on windows 8-39 vs S3-FIFO's handover run, beside v1's
cross-validated result (prbcv) and 3L-Cache.
  vq_gate_online.py SUITE_DIR VQSIM WORKERS  (needs a vqsim with gates online)"""
import json, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

root, vqsim, workers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
port = json.load(open("/mydata/vq/portfolio-r1bp.prereg-1045.json"))["portfolio"]
def gated(c, mode):
    k = len(re.findall(r"(?:^| )st=", c))
    main = int(m.group(1)) if (m := re.search(r"(?:^| )amain=(\d+)", c)) else k - 1
    entry = int(m.group(1)) if (m := re.search(r"(?:^| )entry=(\d+)", c)) else 0
    return None if main == entry else f"{c} gate={main} gmode={mode}"
def valid(configs):
    """Configs the engine accepts (a rescue that would let moves cycle is rejected)."""
    tmp = Path(tempfile.mkdtemp()); (tmp / "ids.u32").write_bytes(np.arange(64, dtype=np.uint32).tobytes())
    (tmp / "jobs").write_text("".join(f"8 0 0 {c}\n" for c in configs))
    out = subprocess.run([vqsim, str(tmp / "ids.u32"), str(tmp / "jobs")], capture_output=True, text=True, check=True).stdout.splitlines()
    return [c for c, line in zip(configs, out, strict=True) if not line.startswith("error")]
twins = {m: valid([g for c in port if (g := gated(c, gm)) is not None]) for m, gm in (("both", 2), ("filter", 1))}
ARMS = [("v1", port), ("v1 + both twins", port + twins["both"]), ("v1 + filter twins", port + twins["filter"])]
HEAD = "rate=0.005 min=200 margin=0"

def cell(args):
    d, cap = args
    meta = json.loads((d / "meta.json").read_text())
    jobs = d / f"gonl-{cap}.jobs"; out = d / f"gonl-{cap}.out"
    jobs.write_text("".join(f"online {cap} {meta['split']} 40 {HEAD}\n" + "".join(c + "\n" for c in cs) + "end\n" for _, cs in ARMS))
    if not (out.is_file() and len(out.read_text().splitlines()) == len(ARMS)):
        out.write_text(subprocess.run(["nice", "-n", "10", vqsim, str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True).stdout)
    lines = out.read_text().splitlines()
    assert not any(l.startswith("error") for l in lines), lines
    base = sum(int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2 + 8:])
    X = np.array([sum(int(x) for x in l.split()[3 + 8:]) for l in lines], float)
    cv = sum(int(x) for x in (d / f"prbcv-{cap}.out").read_text().splitlines()[0].split()[3 + 8:])
    s3 = meta["s3fifo"][str(cap)]
    return d.name, 100 * cap / meta["objects"], (X - base) / base, (cv - base) / base, (meta["3L"][str(cap)] - s3) / s3, [int(l.split()[2]) for l in lines]

cells = [(d, cap) for d in sorted(root.iterdir()) if (d / "meta.json").is_file() for cap in json.loads((d / "meta.json").read_text())["capacities"]]
with ThreadPoolExecutor(workers) as pool:
    got = list(pool.map(cell, cells))
T = [g[0] for g in got]; Z = np.array([g[1] for g in got]); R = np.array([g[2] for g in got]); CV = np.array([g[3] for g in got]); L3 = np.array([g[4] for g in got])
SW = np.array([g[5] for g in got])
lead = CV - L3 > 0.005
bands = [(0, 0.5, "<0.5%"), (0.5, 2, "0.5-2%"), (2, 8, "2-8%"), (8, 1e9, ">=8%")]
print(f"{root.name}: {len(cells)} cells; online selector, windows 8-39, change vs S3-FIFO; twins: {len(twins['both'])} both, {len(twins['filter'])} filter")
print(f"  {'arm':<20}{'mean':>8}{'median':>8}" + "".join(f"{b[2]:>9}" for b in bands) + f"{'3L leads':>10}{'switches':>10}")
for j, (n, cs) in enumerate(ARMS):
    print(f"  {n + f' ({len(cs)})':<20}{100*R[:, j].mean():+7.2f}%{100*np.median(R[:, j]):+7.2f}%" + "".join(f"{100*R[(Z >= lo) & (Z < hi), j].mean():+8.2f}%" for lo, hi, _ in bands)
          + f"{100*R[lead, j].mean():+9.2f}%{SW[:, j].mean():10.1f}")
print(f"  {'v1 cross-validated':<20}{100*CV.mean():+7.2f}%{100*np.median(CV):+7.2f}%" + "".join(f"{100*CV[(Z >= lo) & (Z < hi)].mean():+8.2f}%" for lo, hi, _ in bands) + f"{100*CV[lead].mean():+9.2f}%")
print(f"  {'3L-Cache':<20}{100*L3.mean():+7.2f}%{100*np.median(L3):+7.2f}%" + "".join(f"{100*L3[(Z >= lo) & (Z < hi)].mean():+8.2f}%" for lo, hi, _ in bands) + f"{100*L3[lead].mean():+9.2f}%")
traces = sorted(set(T))
per = lambda v: np.array([np.mean([v[i] for i in range(len(T)) if T[i] == t]) for t in traces])
l3t = per(L3)
for j, (n, _) in enumerate(ARMS):
    d = R[:, j] - R[:, 0]
    print(f"  {n}: better than v1 in {(d < -0.001).sum()} cells, worse in {(d > 0.001).sum()} (largest gain {100*d.min():+.2f} pt, largest loss {100*d.max():+.2f} pt); "
          f"beats 3L-Cache on {(per(R[:, j]) < l3t).sum()} of {len(traces)} traces")
