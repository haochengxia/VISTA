"""Decomposition of v1 and v2 per cell (anonymous aggregates). usage: diag_b.py final|train"""
import json, re, sys
from pathlib import Path
import numpy as np
suite = sys.argv[1]
pool = open("/users/Haocheng/VISTA/scripts/vqsim/v2/pool.txt").read().splitlines()
rnd = json.load(open("/mydata/vq/fl_diag_meta.json"))["random_lines"]
if suite == "final":
    T = Path("/mydata/vq/final-long")
    rows_of = {json.loads(l)["config"]: i for i, l in enumerate(open("/mydata/vq/fl_diag_designs.jsonl"))}
    def cold(d, cap):
        L = (d / f"dgc-{cap}.out").read_text().splitlines()
        return lambda c: np.array([int(x) for x in L[rows_of[c]].split()[2:]], float)
    def blocks(d, cap):
        jb = (d / f"v2fl-{cap}.jobs").read_text().split("end\n")[:2]; err = (d / f"dg-{cap}.err").read_text()
        cut = err.index("screen 0:"); out = (d / f"dg-{cap}.out").read_text().splitlines()
        return [(jb[0], err[:cut], out[0]), (jb[1], err[cut:], out[1])]
else:
    T = Path("/mydata/vq/train-long")
    where = {}
    for s, f in (("r1", "vq_designs.jsonl"), ("bp", "bp_designs.jsonl"), ("rd", "rd_designs.jsonl"), ("gt", "gt_designs.jsonl")):
        for i, l in enumerate(open(f"/mydata/vq/{f}")): where.setdefault(json.loads(l)["config"], (s, i))
    def cold(d, cap):
        cache = {}
        def get(c):
            s, i = where[c]
            if s not in cache: cache[s] = (d / f"{s}-{cap}.out").read_text().splitlines()
            return np.array([int(x) for x in cache[s][i].split()[2:]], float)
        return get
    def blocks(d, cap):
        return [((d / f"prbcv-{cap}.jobs").read_text().split("end\n")[0], (d / f"dgv1-{cap}.err").read_text(), (d / f"dgv1-{cap}.out").read_text().splitlines()[0]),
                ((d / f"scrB1-{cap}.jobs").read_text().split("end\n")[0], (d / f"dgv2-{cap}.err").read_text(), (d / f"dgv2-{cap}.out").read_text().splitlines()[0])]
def parse(job, err):
    cands = [l.strip() for l in job.strip().splitlines()[1:]]
    cands += [pool[int(m)] for m in re.findall(r"file line (\d+)", err)]
    lead = {}
    for m in re.finditer(r"decide w=(\d+) live=(-?\d+) leader=(\d+)", err): lead[int(m.group(1))] = int(m.group(3))
    seq, cur = [], None
    for w in range(8, 40): cur = lead.get(w, cur); seq.append(cur)
    return cands, seq
names = sorted(d.name for d in T.iterdir() if (d / "meta.json").is_file())
R, rho_all, pre_regret = [], [], []
for k, t in enumerate(names):
    d = T / t; meta = json.loads((d / "meta.json").read_text())
    for cap in meta["capacities"]:
        get = cold(d, cap); W0 = get(pool[0]) if suite == "train" else None
        W0 = np.array([int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]], float); base = W0[8:].sum()
        s3 = meta["s3fifo"][str(cap)]; r = dict(k=k, size=100 * cap / meta["objects"], L3=(meta["3L"][str(cap)] - s3) / s3)
        for tag, (job, err, outl) in zip(("v1", "v2"), blocks(d, cap)):
            cands, seq = parse(job, err); M = np.array([get(c) for c in cands])
            live = (sum(int(x) for x in outl.split()[3 + 8:]) - base) / base
            cseq = (sum(M[seq[i], 8 + i] for i in range(32)) - base) / base
            C = np.cumsum(M[:, 2:], axis=1)[:, 5:37]; lead = C.argmin(0)
            ideal = (M[lead, np.arange(8, 40)].sum() - base) / base
            orac = (M[:, 8:].sum(1).min() - base) / base
            r.update({f"{tag}_live": live, f"{tag}_cseq": cseq, f"{tag}_ideal": ideal, f"{tag}_oracle": orac,
                      f"{tag}_agree": float(np.mean([seq[i] == lead[i] for i in range(32)]))})
            if tag == "v2":
                scr = M[32:]; r["scr_best_future"] = (scr[:, 8:].sum(1).min() - base) / base
                r["v1set_best_future"] = (M[:32, 8:].sum(1).min() - base) / base
        # prefix predictivity over the 300 random pool designs
        P = np.array([get(pool[i]) for i in rnd]); pre = P[:, 2:8].sum(1); fut = P[:, 8:].sum(1)
        rk = lambda x: np.argsort(np.argsort(x)); rho = np.corrcoef(rk(pre), rk(fut))[0, 1]; rho_all.append(rho)
        top = np.argsort(pre)[:16]; r["rand_top16_future"] = (fut[top].min() - base) / base; r["rand_best_future"] = (fut.min() - base) / base
        r["rand_top1_regret"] = (fut[top[0]] - fut.min()) / base; R.append(r)
A = lambda key: np.array([r[key] for r in R], float)
per = lambda x: np.array([x[A("k") == k].mean() for k in range(len(names))])
L3 = A("L3")
print(f"{suite}: {len(R)} cells, windows 8-39 vs S3-FIFO (mean over cells; traces beating 3L-Cache)")
for tag in ("v1", "v2"):
    print(f"  {tag}: live {100*A(tag+'_live').mean():+.2f}% ({int((per(A(tag+'_live')) < per(L3)).sum())}/{len(names)}) | same choices without takeover {100*A(tag+'_cseq').mean():+.2f}%"
          f" | exact leader, same candidates {100*A(tag+'_ideal').mean():+.2f}% ({int((per(A(tag+'_ideal')) < per(L3)).sum())}/{len(names)}) | best candidate in hindsight {100*A(tag+'_oracle').mean():+.2f}%"
          f" | sampled choice = exact leader in {100*A(tag+'_agree').mean():.0f}% of windows")
    print(f"      takeover/switch cost {100*(A(tag+'_live') - A(tag+'_cseq')).mean():+.2f} pt; sampling cost {100*(A(tag+'_cseq') - A(tag+'_ideal')).mean():+.2f} pt; exact leader vs hindsight best {100*(A(tag+'_ideal') - A(tag+'_oracle')).mean():+.2f} pt")
print(f"  v2 - v1: live {100*(A('v2_live') - A('v1_live')).mean():+.2f} pt = takeover {100*((A('v2_live') - A('v2_cseq')) - (A('v1_live') - A('v1_cseq'))).mean():+.2f}"
      f" + sampling {100*((A('v2_cseq') - A('v2_ideal')) - (A('v1_cseq') - A('v1_ideal'))).mean():+.2f} + candidate set (exact leader) {100*(A('v2_ideal') - A('v1_ideal')).mean():+.2f}")
print(f"  hindsight: best of v1's 32 {100*A('v1set_best_future').mean():+.2f}%, best of the 16 screened {100*A('scr_best_future').mean():+.2f}%; screened beat v1's 32 in hindsight in {int((A('scr_best_future') < A('v1set_best_future') - 1e-4).sum())}/{len(R)} cells")
print(f"  prefix predictivity, 300 random pool designs: rank correlation prefix vs future misses median {np.median(rho_all):.2f} (cells below 0.5: {int((np.array(rho_all) < 0.5).sum())});"
      f" best of the exact top-16 by prefix vs best overall {100*(A('rand_top16_future') - A('rand_best_future')).mean():+.2f} pt; exact prefix top-1 regret {100*A('rand_top1_regret').mean():.2f} pt")
for lo, hi in ((0, 0.5), (0.5, 2), (2, 8), (8, 1e9)):
    m = (A("size") >= lo) & (A("size") < hi)
    f = lambda key: 100 * A(key)[m].mean()
    print(f"    size {lo}-{hi:g}% ({int(m.sum())}): v1 live {f('v1_live'):+.2f} ideal {f('v1_ideal'):+.2f} | v2 live {f('v2_live'):+.2f} cseq {f('v2_cseq'):+.2f} ideal {f('v2_ideal'):+.2f} | 3L {f('L3'):+.2f}")
