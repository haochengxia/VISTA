import json, subprocess
from pathlib import Path
import numpy as np
D = json.load(open("/tmp/diag_d.json"))["final"]
R = sorted([r for r in D if r["tag"] == "v2"], key=lambda r: -r["cost"])[:2]
T = Path("/mydata/vq/final-long"); names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
S3 = "maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
for r in R:
    d = T / names[r["k"]]; meta = json.loads((d / "meta.json").read_text()); n = meta["requests"]
    cap = min(meta["capacities"], key=lambda c: abs(100 * c / meta["objects"] - r["size"]))
    ids = np.fromfile(d / "ids.u32", dtype=np.uint32)
    w8 = (8 * n + 39) // 40  # first request of window 8
    suf = Path(f"/tmp/suffix-{r['k']}-{cap}.u32"); ids[w8:].tofile(suf)
    jobs = Path(f"/tmp/sufjobs-{r['k']}-{cap}")
    # empty cache at window 8 (suffix run, 32 windows), and handovers from S3-FIFO at windows 4, 8 (the split) and 12
    jobs.write_text(f"{cap} 0 32 {r['cfg']}\n{cap} 0 32 {S3}\n")
    out = subprocess.run(["/mydata/vqsim-diag", str(suf), str(jobs)], capture_output=True, text=True, check=True).stdout.splitlines()
    E = np.array([int(x) for x in out[0].split()[2:]], float); E3 = np.array([int(x) for x in out[1].split()[2:]], float)
    hj = Path(f"/tmp/hojobs-{r['k']}-{cap}")
    hj.write_text("".join(f"{cap} {(w * n + 39) // 40} 40 {r['cfg']} handover=1\n" for w in (4, 8, 12)) + f"{cap} 0 40 {r['cfg']}\n{cap} 0 40 {S3}\n")
    ho = subprocess.run(["/mydata/vqsim-diag", str(d / "ids.u32"), str(hj)], capture_output=True, text=True, check=True).stdout.splitlines()
    H = [np.array([int(x) for x in l.split()[2:]], float) for l in ho]; C0, S = H[3], H[4]
    rel = lambda x, b: " ".join(f"{100*(a-c)/c:+4.0f}" for a, c in zip(x, b))
    print(f"t{r['k']:02d} size {r['size']:.2f}%: per window 8..27 vs S3-FIFO cold (%); design: {r['cfg'][:60]}...")
    print("  cold from t=0          ", rel(C0[8:28], S[8:28]))
    print("  empty cache at window 8", rel(E[:20], E3[:20]), " (vs S3-FIFO also empty there)")
    for j, w in enumerate((4, 8, 12)):
        print(f"  takeover at window {w:2d}  ", rel(H[j][8:28], S[8:28]))
