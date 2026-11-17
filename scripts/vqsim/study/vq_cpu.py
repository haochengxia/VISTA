"""CPU cost relative to S3-FIFO in the same simulator: 3L-Cache vs S3-FIFO in libCacheSim (process_trace, C),
v1's online selector vs S3-FIFO in vqsim (C). One train-long trace, its 4 cache sizes, single thread."""
import json, subprocess, time
from pathlib import Path
import numpy as np
import libcachesim as lcs
root = Path("/mydata/vq/train-long")
metas = [(d, json.loads((d / "meta.json").read_text())) for d in sorted(root.iterdir()) if (d / "meta.json").is_file()]
d, meta = min((x for x in metas if max(x[1]["capacities"]) > 10_000), key=lambda x: abs(x[1]["requests"] - 15e6))
ids = np.fromfile(d / "ids.u32", dtype=np.uint32)
n = ids.size
trace = Path("/mydata/vq/cpu-trace.oracleGeneral")
rec = np.zeros(n, dtype=np.dtype([("t", "<u4"), ("id", "<u8"), ("size", "<u4"), ("next", "<i8")]))
rec["t"] = np.arange(n, dtype=np.uint32); rec["id"] = ids.astype(np.uint64) + 1; rec["size"] = 1; rec["next"] = -1
trace.write_bytes(rec.tobytes())
port = json.load(open("/mydata/vq/portfolio-r1bp.prereg-1045.json"))["portfolio"]
S3 = port[0]
print(f"one trace: {n/1e6:.1f}M requests, capacities {meta['capacities']}")
for cap in meta["capacities"]:
    t = {}
    for name, make in (("S3-FIFO", lambda: lcs.S3FIFO(cap)), ("3L-Cache", lambda: lcs.ThreeLCache(cap, objective="object-miss-ratio"))):
        cache = make(); t0 = time.process_time()
        cache.process_trace(lcs.TraceReader(str(trace), lcs.TraceType.ORACLE_GENERAL_TRACE))
        t[name] = time.process_time() - t0
    for name, job in (("vqsim S3-FIFO", f"{cap} {meta['split']} 0 {S3}\n"),
                      ("vqsim v1", f"online {cap} {meta['split']} 40 rate=0.005 min=200 margin=0\n" + "".join(c + "\n" for c in port) + "end\n")):
        jf = Path(f"/mydata/vq/cpu-{cap}.jobs"); jf.write_text(job)
        t0 = time.time()
        r = subprocess.run(["/usr/bin/time", "-f", "%U %S", "/mydata/vqsim3", str(d / "ids.u32"), str(jf)], capture_output=True, text=True, check=True)
        u, s = (float(x) for x in r.stderr.strip().splitlines()[-1].split())
        t[name] = u + s
    print(f"  C={cap}: libCacheSim S3-FIFO {t['S3-FIFO']:.1f}s, 3L-Cache {t['3L-Cache']:.1f}s ({t['3L-Cache']/t['S3-FIFO']:.1f}x) | "
          f"vqsim S3-FIFO {t['vqsim S3-FIFO']:.1f}s, v1 {t['vqsim v1']:.1f}s ({t['vqsim v1']/t['vqsim S3-FIFO']:.1f}x)", flush=True)
trace.unlink()
