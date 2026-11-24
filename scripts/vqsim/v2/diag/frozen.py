"""Designs with a stage that receives nothing after warm-up ("frozen": it keeps what filled it first)."""
import json, re, sys
def parse(c):
    kv = dict(t.split("=", 1) for t in c.split() if not t.startswith("st="))
    st = [tuple(float(x) for x in t[3:].split(",")) for t in c.split() if t.startswith("st=")]
    return kv, st
def frozen(c):
    kv, st = parse(c); k = len(st); maxf = int(kv.get("maxf", 3)); entry = int(kv.get("entry", 0))
    ghost = float(kv.get("ghost", 0.9)); gentry = int(kv.get("gentry", -1)); gentry = gentry if gentry >= 0 else k - 1
    gate = int(kv.get("gate", -1)); gmode = int(kv.get("gmode", 0)); rd = float(kv.get("rd", 0)); rdmode = int(kv.get("rdmode", 3))
    edge = {i: set() for i in range(k)}
    for i, s in enumerate(st):
        size, count, promote, up, onup, down = s[0], int(s[1]), int(s[2]), int(s[3]), int(s[4]), int(s[5])
        if promote == 0 or (count and promote <= maxf) or (rd > 0 and rdmode & 8): edge[i].add(up)
        if down >= 0: edge[i].add(down)
        if gate >= 0 and gmode in (0, 2) and i != gate: edge[i].add(gate)
    reach = {entry} | ({gentry} if ghost > 0 else set()); todo = list(reach)
    while todo:
        i = todo.pop()
        for j in edge[i]:
            if j not in reach: reach.add(j); todo.append(j)
    return [i for i in range(k) if i not in reach]
if __name__ == "__main__":
    pool = open("/users/Haocheng/VISTA/scripts/vqsim/v2/pool.txt").read().splitlines()
    v1 = json.load(open("/users/Haocheng/VISTA/scripts/vqsim/v1/portfolio.json"))["portfolio"]
    fz = [i for i, c in enumerate(pool) if frozen(c)]
    print(f"pool: {len(fz)}/{len(pool)} designs have a frozen stage; v1's 32: {sum(1 for c in v1 if frozen(c))}")
    json.dump(fz, open("/tmp/frozen_lines.json", "w"))
    D = json.load(open("/tmp/diag_d.json"))
    for suite in ("final", "train"):
        for tag in ("v1", "v2"):
            R = [r for r in D[suite] if r["tag"] == tag]; f = [r for r in R if frozen(r["cfg"])]
            print(f"  {suite} {tag}: split pick frozen in {len(f)}/{len(R)} cells; their takeover cost {', '.join(f'{100*r['cost']:+.1f}' for r in f) or '-'} pt")
def frozen_share(c):
    kv, st = parse(c); fz = frozen(c)
    sizes = [s[0] for s in st]; sizes[-1] = max(0.0, 1 - sum(sizes[:-1]))
    return sum(sizes[i] for i in fz)
if __name__ == "__main__":
    import numpy as np
    sh = np.array([frozen_share(c) for c in pool])
    print("pool, share of the cache in frozen stages:", {f">={x}": int((sh >= x - 1e-9).sum()) for x in (0.01, 0.1, 0.5, 0.8)})
    print("v1's frozen design: share", [round(frozen_share(c), 3) for c in v1 if frozen(c)], [c for c in v1 if frozen(c)][0][:120])
    for suite in ("final", "train"):
        for tag in ("v1", "v2"):
            R = [r for r in D[suite] if r["tag"] == tag]
            print(f"  {suite} {tag} split picks with frozen share >= 0.5: {sum(1 for r in R if frozen_share(r['cfg']) >= 0.5)}; their costs", [round(100 * r["cost"], 1) for r in R if frozen_share(r["cfg"]) >= 0.5])
    json.dump([i for i, c in enumerate(pool) if frozen_share(c) >= 0.5], open("/tmp/frozen_main_lines.json", "w"))
def stale_share(c):
    """Share of the cache in frozen stages that get filled anyway: at warm-up (the overflow stage) or by a takeover (asmall, amain)."""
    kv, st = parse(c); k = len(st); fz = set(frozen(c))
    filled = {int(kv.get("overflow", k - 1)) if int(kv.get("overflow", -1)) >= 0 else k - 1}
    a_s = int(kv.get("asmall", -1)); a_m = int(kv.get("amain", -1)); filled |= {a_s if a_s >= 0 else int(kv.get("entry", 0)), a_m if a_m >= 0 else k - 1}
    sizes = [s[0] for s in st]; sizes[-1] = max(0.0, 1 - sum(sizes[:-1]))
    return sum(sizes[i] for i in fz & filled)
if __name__ == "__main__":
    ss = np.array([stale_share(c) for c in pool])
    print("pool, share of the cache in frozen stages that do get filled:", {f">={x}": int((ss >= x - 1e-9).sum()) for x in (0.01, 0.1, 0.5, 0.8)}, "; v1's 32:", sum(1 for c in v1 if stale_share(c) > 0))
    for suite in ("final", "train"):
        for tag in ("v1", "v2"):
            R = [r for r in D[suite] if r["tag"] == tag]
            print(f"  {suite} {tag} split picks with stale share > 0: {sum(1 for r in R if stale_share(r['cfg']) > 0)}; costs", [round(100 * r["cost"], 1) for r in R if stale_share(r["cfg"]) > 0])
    json.dump([i for i, c in enumerate(pool) if stale_share(c) > 0], open("/tmp/stale_lines.json", "w"))
