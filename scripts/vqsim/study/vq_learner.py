"""Option (b): can the online selector also hand the live cache to a learned policy (3L-Cache)?

Development on train-long, final-long (a development set) and gs-dev; never on heldout-long or fresh-long. gs-test is a
one-round pre-registered test (scripts/vqsim/v2/PREREG-gstest.md): drop, zoo, bound, test and describe serve it.

  vq_learner.py prepare SUITE.json SUITE WORKERS   a new development suite without the engineer baselines: dense ids,
                                          split (20%) and cache sizes by the engineer's rules (engineer.prepare)
  vq_learner.py mini  SUITE WORKERS       3L-Cache and S3-FIFO per window at full scale, and as miniatures on the
                                          selector's sampled stream (same objects, same size as vqsim's miniatures)
  vq_learner.py rates SUITE R1,R2,.. WORKERS   3L-Cache and S3-FIFO miniatures at other sampling rates (fidelity)
  vq_learner.py finish SUITE              meta.json's suffix miss ratios of S3-FIFO and 3L-Cache from the full runs
                                          (for suites prepared here; the engineer baselines' values are identical)
  vq_learner.py trace SUITE TAG WORKERS [BLOCK]   rerun job block BLOCK (default 2: v2 + replay 128 in rp2T/rp2F) of TAG
                                          with VQSIM_TRACE: the leader's sampled score at every decision; the output
                                          must equal line BLOCK of TAG's output
  vq_learner.py score SUITE TAG           selector rules that may switch the live cache to 3L-Cache, vs v2 and 3L-Cache
  vq_learner.py plan SUITE TAG RATE MARGIN H    a realistic one-way rule: switch to 3L-Cache at the first window where its
                                          miniature at RATE (scaled) beats v2's observed live misses since the split (H=0)
                                          or over the last H windows, by MARGIN
  vq_learner.py grid SUITE TAG WORKERS [L]   the real takeover at every window 9..39 of every cell (mode A below)
  vq_learner.py rules SUITE TAG [L]       cost-aware one-way rules (switch when the estimated gain over the remaining
                                          windows exceeds kappa * C misses), scored with the grid's real takeovers
  vq_learner.py takeover SUITE TAG PLAN WORKERS [L]   the plan's real takeover: v2's live contents at the switch (vqsim
                                          VQSIM_DUMP), 3L-Cache rebuilt from the last L*C requests (L<=0: all) and reconciled
  vq_learner.py drop SUITE                the mechanical drop rule, from the baselines alone (before any candidate run)
  vq_learner.py zoo SUITE WORKERS         the 14 published classic algorithms and Belady per cell (zoo.py's runner)
  vq_learner.py bound SUITE               each algorithm vs S3-FIFO and 3L-Cache; the classic-oracle bound (best of 14 per cell)
  vq_learner.py test SUITE TAG            the pre-registered bar of PREREG-gstest.md, with a 10,000-resample interval
  vq_learner.py describe SUITE TAG R1,..  when 3L-Cache's edge appears, and its miniature's fidelity at each rate

The selector samples object x when mix64(x) <= rate * 2^64, with rate = max(0.005, 200 / C) (at most 1), and its
miniatures hold round(rate * C) objects (vqsim.c). Windows: request t is in window t * 40 // n. Times in the
libCacheSim traces are request indices of the full trace, as for the baselines (zoo.py). Trace names are never printed.
"""
import json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import numpy as np

W = 40; RATE, FLOOR = 0.005, 200
OUT = Path("/mydata/b3l")
ALG = {"3L": lambda lcs, c: lcs.ThreeLCache(c, objective="object-miss-ratio"), "S3": lambda lcs, c: lcs.S3FIFO(c)}


def mix64(x):
    x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def mini_rate(cap):
    r = RATE
    if r * cap < FLOOR: r = FLOOR / cap
    return min(r, 1.0)


def write_og(keys, times, path):
    """oracleGeneral records: time, id (+1, never 0), size 1, time of the next request to the object or -1."""
    n = keys.size; order = np.lexsort((np.arange(n), keys)); same = keys[order][1:] == keys[order][:-1]
    following = np.full(n, -1, dtype=np.int64); following[order[:-1][same]] = order[1:][same]
    rec = np.zeros(n, dtype=np.dtype([("t", "<u4"), ("id", "<u8"), ("size", "<u4"), ("next", "<i8")]))
    rec["t"] = times; rec["id"] = keys.astype(np.uint64) + 1; rec["size"] = 1
    rec["next"] = np.where(following >= 0, times[np.maximum(following, 0)].astype(np.int64), -1)
    tmp = path.with_suffix(".tmp"); tmp.write_bytes(rec.tobytes()); tmp.rename(path)


def run_alg(alg, path, cap, bounds, split=None):
    """Misses per window, and misses at request indices >= split (as the baselines count them)."""
    import libcachesim as lcs
    cache = ALG[alg](lcs, cap); wins = [0] * W; w = 0; nb = bounds[1]; after = 0; split = bounds[-1] if split is None else split
    for i, req in enumerate(lcs.TraceReader(str(path), lcs.TraceType.ORACLE_GENERAL_TRACE)):
        while i >= nb: w += 1; nb = bounds[w + 1]
        if not cache.get(req):
            wins[w] += 1; after += i >= split
    return wins, after


def prepare_trace(a):
    spec, root = a
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from vista.benchmarks.cachelab import engineer
    import shutil
    d = root / spec.name; d.mkdir(parents=True, exist_ok=True)
    if (d / "meta.json").is_file(): return spec.name, "done"
    try:
        task = engineer.prepare(spec, d / "lab", fraction=0.2)
    except ValueError as e:  # too few distinct objects in the observed part: the baselines would fail too
        shutil.rmtree(d, ignore_errors=True); return spec.name, f"dropped ({e.__class__.__name__})"
    _, dense = np.unique(task.full.keys(), return_inverse=True)
    dense.astype(np.uint32).tofile(d / "ids.u32")
    (d / "meta.json").write_text(json.dumps({"requests": int(dense.size), "objects": int(dense.max()) + 1, "split": task.split,
                                             "capacities": list(task.capacities)}))
    shutil.rmtree(d / "lab", ignore_errors=True)
    return spec.name, "ok"


def cmd_prepare(suite_json, suite, workers):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from vista.benchmarks.cachelab.traces import load_suite
    root = Path(f"/mydata/vq/{suite}"); root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex: got = list(ex.map(prepare_trace, [(s, root) for s in load_suite(Path(suite_json))]))
    print(f"  {suite}: {sum(1 for _, s in got if s in ('ok', 'done'))} traces prepared, {sum(1 for _, s in got if s.startswith('dropped'))} dropped"
          " (too few distinct objects in the observed part)")


def cmd_finish(suite):
    for d in traces(suite):
        meta = json.loads((d / "meta.json").read_text()); tdir = OUT / suite / d.name; n, split = meta["requests"], meta["split"]
        for alg, key in (("S3", "s3fifo"), ("3L", "3L")):
            meta[key] = {str(c): json.loads((tdir / f"{alg}-full-{c}.json").read_text())["after"] / (n - split) for c in meta["capacities"]}
        (d / "meta.json").write_text(json.dumps(meta))
    print(f"  {suite}: meta.json has S3-FIFO's and 3L-Cache's suffix miss ratios")


def prep(d):
    """The full trace as oracleGeneral (shared by the trace's cells) and each cell's sampled stream."""
    ids = np.fromfile(d / "ids.u32", dtype=np.uint32); n = ids.size; meta = json.loads((d / "meta.json").read_text())
    tdir = OUT / d.parent.name / d.name; tdir.mkdir(parents=True, exist_ok=True)
    full = tdir / "full.og"
    if not full.is_file(): write_og(ids, np.arange(n, dtype=np.uint32), full)
    bfull = [(w * n + W - 1) // W for w in range(W)] + [n]
    h = mix64(np.arange(int(ids.max()) + 1, dtype=np.uint64))
    for cap in meta["capacities"]:
        r = mini_rate(cap); mcap = max(int(r * cap + 0.5), 1)
        f = tdir / f"mini-{cap}.og"
        pos = np.arange(n) if r >= 1 else np.flatnonzero(h[ids] <= np.uint64(int(r * 18446744073709551615.0)))
        if not f.is_file(): write_og(ids[pos], pos.astype(np.uint32), f)
        (tdir / f"mini-{cap}.json").write_text(json.dumps({"rate": r, "mcap": mcap, "n": int(pos.size),
                                                           "bounds": [int(x) for x in np.searchsorted(pos, bfull[:W])] + [int(pos.size)]}))
    (tdir / "full.json").write_text(json.dumps({"n": n, "bounds": bfull, "split": meta["split"]}))
    return d


def job(a):
    tdir, alg, scale, cap = a
    out = tdir / f"{alg}-{scale}-{cap}.json"
    if out.is_file(): return out
    if scale == "full": info = json.loads((tdir / "full.json").read_text()); path, c = tdir / "full.og", cap
    else: info = json.loads((tdir / f"mini-{cap}.json").read_text()); path, c = tdir / f"mini-{cap}.og", info["mcap"]
    wins, after = run_alg(alg, path, c, info["bounds"], info.get("split") if scale == "full" else None)
    out.write_text(json.dumps({"alg": alg, "scale": scale, "cap": cap, "size": c, "wins": wins, "after": after})); return out


def traces(suite):
    """The suite's traces; with VQ_GROUP=i/n only every n-th from the i-th (the heavy steps run in groups to bound disk use)."""
    ds = sorted(p for p in Path(f"/mydata/vq/{suite}").iterdir() if (p / "meta.json").is_file())
    g = os.environ.get("VQ_GROUP")
    if not g: return ds
    i, n = (int(x) for x in g.split("/")); return ds[i::n]


def rate_prep(a):
    """Sampled streams at miniature rate `rate` (raised to the floor, at most 1) for a trace's cells."""
    d, suite, rate = a
    ids = np.fromfile(d / "ids.u32", dtype=np.uint32); n = ids.size; meta = json.loads((d / "meta.json").read_text())
    tdir = OUT / suite / d.name; bfull = [(w * n + W - 1) // W for w in range(W)] + [n]
    h = mix64(np.arange(int(ids.max()) + 1, dtype=np.uint64))
    for cap in meta["capacities"]:
        r = min(max(rate, FLOOR / cap), 1.0); mcap = max(int(r * cap + 0.5), 1); f = tdir / f"mini-{cap}-r{rate:g}.og"
        pos = np.arange(n) if r >= 1 else np.flatnonzero(h[ids] <= np.uint64(int(r * 18446744073709551615.0)))
        if not f.is_file(): write_og(ids[pos], pos.astype(np.uint32), f)
        (tdir / f"mini-{cap}-r{rate:g}.json").write_text(json.dumps({"rate": r, "mcap": mcap, "n": int(pos.size),
                                                                     "bounds": [int(x) for x in np.searchsorted(pos, bfull[:W])] + [int(pos.size)]}))


def rate_job(a):
    tdir, alg, cap, rate = a
    out = tdir / f"{alg}-mini-{cap}-r{rate:g}.json"
    if out.is_file(): return
    info = json.loads((tdir / f"mini-{cap}-r{rate:g}.json").read_text())
    wins, _ = run_alg(alg, tdir / f"mini-{cap}-r{rate:g}.og", info["mcap"], info["bounds"])
    out.write_text(json.dumps({"alg": alg, "cap": cap, "rate": info["rate"], "size": info["mcap"], "n": info["n"], "wins": wins}))


def cmd_rates(suite, rates, workers):
    """3L-Cache and S3-FIFO miniatures at other sampling rates: how faithful is a larger 3L-Cache miniature?"""
    ds = traces(suite)
    with ProcessPoolExecutor(min(workers, 12)) as ex: list(ex.map(rate_prep, [(d, suite, r) for d in ds for r in rates]))
    jobs = [(OUT / suite / d.name, alg, cap, r) for d in ds for cap in json.loads((d / "meta.json").read_text())["capacities"] for r in rates for alg in ("3L", "S3")]
    jobs.sort(key=lambda j: -json.loads((j[0] / f"mini-{j[2]}-r{j[3]:g}.json").read_text())["n"])
    with ProcessPoolExecutor(workers) as ex: list(ex.map(rate_job, jobs))
    for d in ds:
        for f in (OUT / suite / d.name).glob("mini-*-r*.og"): f.unlink()
    print(f"  {suite}: miniatures at rates {rates}: {len(jobs)} runs")


def cmd_mini(suite, workers):
    ds = traces(suite)
    with ProcessPoolExecutor(min(workers, 12)) as ex: list(ex.map(prep, ds))
    jobs = []
    for i, d in enumerate(ds):
        meta = json.loads((d / "meta.json").read_text()); tdir = OUT / suite / d.name
        for cap in meta["capacities"]:
            jobs += [(tdir, "3L", "full", cap, meta["requests"]), (tdir, "3L", "mini", cap, meta["requests"] * mini_rate(cap)),
                     (tdir, "S3", "mini", cap, meta["requests"] * mini_rate(cap))]
            if i < 2 or "3L" not in meta: jobs.append((tdir, "S3", "full", cap, meta["requests"]))  # a check, or meta's values
    jobs.sort(key=lambda j: -j[4])
    with ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(job, [j[:4] for j in jobs]), 1):
            if k % 50 == 0 or k == len(jobs): print(f"  {k}/{len(jobs)} runs", flush=True)
    for d in ds: (OUT / suite / d.name / "full.og").unlink(missing_ok=True)


def blocks(text):
    out, cur = [], []
    for line in text.splitlines(keepends=True):
        cur.append(line)
        if line.strip() == "end": out.append("".join(cur)); cur = []
    return out


def trace_cell(a):
    d, cap, tag, block = a
    tdir = OUT / d.parent.name / d.name; tdir.mkdir(parents=True, exist_ok=True)
    out, err = tdir / f"v2-{cap}.out", tdir / f"v2-{cap}.err"
    if not out.is_file():
        jobs = tdir / f"v2-{cap}.jobs"; jobs.write_text(blocks((d / f"{tag}-{cap}.jobs").read_text())[block])
        p = subprocess.run(["nice", "-n", "10", os.environ.get("VQSIM", "/mydata/vqsim-fresh"), str(d / "ids.u32"), str(jobs)], capture_output=True,
                           text=True, check=True, env={**os.environ, "VQSIM_TRACE": "1"})
        err.write_text(p.stderr); out.write_text(p.stdout)
    ref = (d / f"{tag}-{cap}.out").read_text().splitlines()[block].split()
    return out.read_text().split() == ref


def cmd_trace(suite, tag, workers, block=2):
    cells = [(d, cap, tag, block) for d in traces(suite) for cap in json.loads((d / "meta.json").read_text())["capacities"]]
    with ThreadPoolExecutor(workers) as ex: same = list(ex.map(trace_cell, cells))
    print(f"  {suite}: {sum(same)}/{len(same)} cells reproduce {tag} line {block} exactly")


def load(suite, tag):
    cells = []
    for ti, d in enumerate(traces(suite)):
        meta = json.loads((d / "meta.json").read_text()); tdir = OUT / suite / d.name
        for cap in meta["capacities"]:
            base = sum(int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:][8:])
            s3 = meta["s3fifo"][str(cap)]; rd = lambda a, s: json.loads((tdir / f"{a}-{s}-{cap}.json").read_text())["wins"]
            v2 = [int(x) for x in (tdir / f"v2-{cap}.out").read_text().split()[3:]]
            dec = {}
            for line in (tdir / f"v2-{cap}.err").read_text().splitlines():
                if line.startswith("decide "):
                    f = dict(kv.split("=") for kv in line.split()[1:]); dec.setdefault(int(f["w"]), int(f["best"]))  # first decision of a window: the split's
            s3full = (tdir / f"S3-full-{cap}.json")
            cells.append(dict(t=ti, cap=cap, base=base, l3=(meta["3L"][str(cap)] - s3) / s3, v2=v2, f3=rd("3L", "full"), m3=rd("3L", "mini"),
                              ms3=rd("S3", "mini"), dec=dec, s3full=json.loads(s3full.read_text())["wins"] if s3full.is_file() else None,
                              r1=[int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]))
    return cells


def stats(label, x, L3, T):
    tr = np.unique(T); per = lambda v: np.array([v[T == t].mean() for t in tr]); dd = per(x) - per(L3)
    rng = np.random.default_rng(0); diffs = []
    for _ in range(4000):
        m = np.concatenate([np.flatnonzero(T == t) for t in rng.choice(tr, len(tr))]); diffs.append(x[m].mean() - L3[m].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {label:<44} mean {100*x.mean():+6.2f}%  median {100*np.median(x):+6.2f}%  cells < 3L {int((x < L3 - 1e-12).sum()):>3}/{len(x)}"
          f"  worse than S3 {int((x > 0.0005).sum()):>2}  traces < 3L >0/>0.25pt {int((dd < 0).sum())}/{int((dd < -0.0025).sum())} of {len(tr)}"
          f"  lead {100*(L3.mean()-x.mean()):+.2f} [{100*-hi:+.2f}, {100*-lo:+.2f}]")


def cmd_score(suite, tag):
    C = load(suite, tag); T = np.array([c["t"] for c in C]); L3 = np.array([c["l3"] for c in C])
    chg = lambda m, c: (m - c["base"]) / c["base"]
    f3 = np.array([chg(sum(c["f3"][8:]), c) for c in C]); v2 = np.array([chg(sum(c["v2"][8:]), c) for c in C])
    ok = [c for c in C if c["s3full"] is not None]
    print(f"{suite}: {len(np.unique(T))} traces, {len(C)} cells")
    print(f"  checks: S3-FIFO in libCacheSim = r1 line 0 per window in {sum(c['s3full'] == c['r1'] for c in ok)}/{len(ok)} cells;"
          f" 3L-Cache here vs the baselines' 3L-Cache: median |diff| {100*np.median(np.abs(f3 - L3)):.2f} pt, max {100*np.abs(f3 - L3).max():.2f} pt")
    stats("3L-Cache (baselines)", L3, L3, T); stats("3L-Cache (this run, cold)", f3, L3, T); stats("v2 + replay 128", v2, L3, T)
    stats("hindsight min(v2, 3L) per cell", np.minimum(v2, f3), L3, T)
    pred = np.array([np.sign(sum(c["m3"][2:8]) - c["dec"][8]) for c in C]); truth = np.sign(f3 - v2)
    agree = (pred == truth) | (truth == 0)
    print(f"  at the split, the 3L-Cache miniature vs v2's leader (sampled, windows 2-7) predicts the suffix winner in {int(agree.sum())}/{len(C)} cells")
    for m in (0.0, 0.01, 0.03, 0.05, 0.10):
        split, sticky = [], []
        for c in C:
            cum = np.concatenate([[0, 0], np.cumsum(c["m3"][2:])])  # cum[w] = 3L miniature misses in windows 2..w-1
            go = sum(c["f3"][8:]) if cum[8] < (1 - m) * c["dec"][8] else sum(c["v2"][8:]); split.append(chg(go, c))
            sw = next((w for w in range(8, W) if w in c["dec"] and cum[w] < (1 - m) * c["dec"][w]), None)
            sticky.append(chg(sum(c["v2"][8:]) if sw is None else sum(c["v2"][8:sw]) + sum(c["f3"][sw:]), c))
        n3 = sum(1 for c in C if np.concatenate([[0, 0], np.cumsum(c["m3"][2:])])[8] < (1 - m) * c["dec"][8])
        stats(f"split: 3L if its miniature leads by {m:.0%} ({n3} cells)", np.array(split), L3, T)
        stats(f"sticky: first window 3L leads by {m:.0%}", np.array(sticky), L3, T)
    print("  (optimistic: the switched-to 3L-Cache is scored as if it had run from the start; no takeover cost)")


def mini_estimate(c, tdir, rate):
    """3L-Cache's full-scale misses per window, estimated by its miniature at `rate` (scaled by the window's sampling ratio)."""
    info = json.loads((tdir / f"mini-{c['cap']}-r{rate:g}.json").read_text()); fb = json.loads((tdir / "full.json").read_text())["bounds"]
    m3 = np.array(json.loads((tdir / f"3L-mini-{c['cap']}-r{rate:g}.json").read_text())["wins"], float)
    return m3 * np.diff(fb) / np.maximum(np.diff(info["bounds"]), 1)


def sticky_window(e3, v2, margin, horizon):
    """First window (from 9) at which 3L-Cache's estimate over the windows since the split (or the last `horizon`) beats v2's
    observed live misses by `margin`; None if never. v2 is live until then, so both sides are observable."""
    for w in range(9, W):
        lo = 8 if horizon == 0 else max(8, w - horizon)
        if e3[lo:w].sum() < (1 - margin) * v2[lo:w].sum(): return w
    return None


def cmd_plan(suite, tag, rate, margin, horizon):
    C = load(suite, tag); ds = traces(suite); plan = {}
    for c in C:
        tdir = OUT / suite / ds[c["t"]].name
        w = sticky_window(mini_estimate(c, tdir, rate), np.array(c["v2"], float), margin, horizon)
        if w is not None: plan[f"{ds[c['t']].name}/{c['cap']}"] = w
    name = f"plan-r{rate:g}-m{margin:g}-h{horizon}"
    (OUT / suite / f"{name}.json").write_text(json.dumps(plan))
    print(f"  {suite}: {name}: {len(plan)}/{len(C)} cells switch to 3L-Cache (window median {np.median(list(plan.values())) if plan else 0:.0f})")


def takeover_cell(a):
    """v2 until window w, then 3L-Cache built from the last L*C requests and reconciled with v2's live contents."""
    d, cap, w, tag, L = a
    import libcachesim as lcs
    tdir = OUT / d.parent.name / d.name; out = tdir / f"take-{cap}-w{w}-L{L:g}.json"
    if out.is_file(): return out
    ids = np.fromfile(d / "ids.u32", dtype=np.uint32); n = ids.size; bounds = [(k * n + W - 1) // W for k in range(W)] + [n]; t = bounds[w]
    dump = tdir / f"live-{cap}-{t}.u32"
    if not dump.is_file():
        jobs = tdir / f"v2-{cap}.jobs"
        p = subprocess.run([os.environ.get("VQSIM", "/mydata/vqsim-dump"), str(d / "ids.u32"), str(jobs)], capture_output=True, text=True, check=True,
                           env={**os.environ, "VQSIM_DUMP": f"{t}:{tdir}/live-{cap}"})
        assert p.stdout.split() == (tdir / f"v2-{cap}.out").read_text().split(), "the dumping run differs from v2"
    live = np.fromfile(dump, dtype=np.uint32); keep = set(live.tolist())
    start = 0 if L <= 0 else max(0, t - int(L * cap))
    path = tdir / f"take-{cap}-{w}.og"; write_og(ids[start:], np.arange(start, n, dtype=np.uint32), path)
    req = lcs.Request(); req.obj_size = 1; req.clock_time = t
    res = {"w": w, "L": L, "start": start, "live": int(live.size)}
    # libCacheSim's 3L-Cache cannot remove objects, so the takeover is measured two ways that bracket a 3L-Cache with removal:
    # A: replay everything; an object cached by the replay but not live (phantom) costs a miss at its first request while
    #    3L-Cache still holds it; a live object 3L-Cache lacks counts its first request as a hit (A_opt) or a miss (A_pes).
    # B: replay only requests to live objects, then insert the live objects 3L-Cache lacks: contents equal the live cache's
    #    exactly, but the model learns from a filtered history.
    def reconcile(cache, mode):
        P, M = set(), set()
        if mode == "A":
            for x in set(np.unique(ids[start:t]).tolist()) | keep:
                req.obj_id = x + 1; has = cache.find(req, False) is not None
                if has and x not in keep: P.add(x)
                elif not has and x in keep: M.add(x)
        else:
            for x in live.tolist():
                req.obj_id = x + 1
                if cache.find(req, False) is None: cache.insert(req); M.add(x)
        res[f"{mode}_phantoms"], res[f"{mode}_missing"] = len(P), len(M)
        return P, M
    for mode in ("A", "B"):
        cache = ALG["3L"](lcs, cap)
        wo, wp = [0] * W, [0] * W; k = w; nb = bounds[w + 1]; P = M = None
        for j, r in enumerate(lcs.TraceReader(str(path), lcs.TraceType.ORACLE_GENERAL_TRACE)):  # one pass: iter() restarts a reader
            i = start + j
            if i < t:
                if mode == "A" or (r.obj_id - 1) in keep: cache.get(r)
                continue
            if P is None: P, M = reconcile(cache, mode)
            while i >= nb: k += 1; nb = bounds[k + 1]
            hit = cache.get(r); x = r.obj_id - 1; miss_o = miss_p = not hit
            if mode == "A" and P and x in P: P.discard(x); miss_o = miss_p = True  # a phantom: really a miss either way
            elif mode == "A" and M and x in M: M.discard(x); miss_o = False if not hit else miss_o; miss_p = True if not hit else miss_p
            wo[k] += miss_o; wp[k] += miss_p
        if mode == "A": res["A_opt"], res["A_pes"] = wo, wp
        else: res["B"] = wo
    path.unlink()
    out.write_text(json.dumps(res))
    return out


def grid_cell(a):
    """Takeover mode A at every window 9..39 of one cell: v2's live contents dumped once per window by one vqsim run, then
    3L-Cache rebuilt from the last L*C requests of the shared full trace and reconciled (phantoms and missing objects)."""
    d, cap, L, ws = a
    import libcachesim as lcs
    tdir = OUT / d.parent.name / d.name; out = tdir / f"grid-{cap}-L{L:g}.json"
    if out.is_file(): return out
    ids = np.fromfile(d / "ids.u32", dtype=np.uint32); n = ids.size; bounds = [(k * n + W - 1) // W for k in range(W)] + [n]
    ts = [bounds[w] for w in ws]
    if not all((tdir / f"live-{cap}-{t}.u32").is_file() for t in ts):
        p = subprocess.run([os.environ.get("VQSIM", "/mydata/vqsim-dump"), str(d / "ids.u32"), str(tdir / f"v2-{cap}.jobs")], capture_output=True,
                           text=True, check=True, env={**os.environ, "VQSIM_DUMP": ",".join(map(str, ts)) + f":{tdir}/live-{cap}"})
        assert p.stdout.split() == (tdir / f"v2-{cap}.out").read_text().split(), "the dumping run differs from v2"
    res = {}
    req = lcs.Request(); req.obj_size = 1
    for w, t in zip(ws, ts):
        live = np.fromfile(tdir / f"live-{cap}-{t}.u32", dtype=np.uint32); keep = set(live.tolist())
        start = 0 if L <= 0 else max(0, t - int(L * cap)); req.clock_time = t
        cache = ALG["3L"](lcs, cap); wo, wp = [0] * W, [0] * W; k = w; nb = bounds[w + 1]; P = M = None
        for i, r in enumerate(lcs.TraceReader(str(tdir / "full.og"), lcs.TraceType.ORACLE_GENERAL_TRACE)):
            if i < start: continue
            if i < t: cache.get(r); continue
            if P is None:
                P, M = set(), set()
                for x in set(np.unique(ids[start:t]).tolist()) | keep:
                    req.obj_id = x + 1; has = cache.find(req, False) is not None
                    if has and x not in keep: P.add(x)
                    elif not has and x in keep: M.add(x)
                np0, nm0 = len(P), len(M)
            while i >= nb: k += 1; nb = bounds[k + 1]
            hit = cache.get(r); x = r.obj_id - 1; miss_o = miss_p = not hit
            if P and x in P: P.discard(x); miss_o = miss_p = True
            elif M and x in M:
                M.discard(x)
                if not hit: miss_o, miss_p = False, True
            wo[k] += miss_o; wp[k] += miss_p
        res[str(w)] = {"start": start, "live": int(live.size), "phantoms": np0, "missing": nm0, "opt": wo, "pes": wp}
        (tdir / f"live-{cap}-{t}.u32").unlink()
    out.write_text(json.dumps(res)); return out


def cmd_grid(suite, tag, workers, L):
    ds = traces(suite); ws = list(range(9, W))
    for d in ds:  # the shared full trace per trace (removed after the mini runs)
        if not (OUT / suite / d.name / "full.og").is_file():
            ids = np.fromfile(d / "ids.u32", dtype=np.uint32); write_og(ids, np.arange(ids.size, dtype=np.uint32), OUT / suite / d.name / "full.og")
    jobs = [(d, cap, L, ws) for d in ds for cap in json.loads((d / "meta.json").read_text())["capacities"]]
    jobs.sort(key=lambda j: -(j[0] / "ids.u32").stat().st_size)
    with ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(grid_cell, jobs), 1):
            if k % 16 == 0 or k == len(jobs): print(f"  {k}/{len(jobs)} cells", flush=True)
    for d in ds: (OUT / suite / d.name / "full.og").unlink(missing_ok=True)


def cmd_rules(suite, tag, L):
    """One-way switch rules scored with the real takeover (grid): v2 until the switch, then 3L-Cache taken over at that window.
    'opt'/'pes' bracket the live objects 3L-Cache lacks at the takeover (first request a hit / a miss)."""
    C = load(suite, tag); ds = traces(suite); T = np.array([c["t"] for c in C]); L3 = np.array([c["l3"] for c in C])
    G = [json.loads((OUT / suite / ds[c["t"]].name / f"grid-{c['cap']}-L{L:g}.json").read_text()) for c in C]
    chg = lambda m, c: (m - c["base"]) / c["base"]
    V = [np.array(c["v2"], float) for c in C]; F = [np.array(c["f3"], float) for c in C]
    print(f"{suite}: one-way switch to 3L-Cache with the real takeover from the last {L:g}*C requests")
    cost = np.array([[(sum(G[i][str(w)]["opt"][w:]) - F[i][w:].sum()) / C[i]["cap"] for w in range(9, W)] for i in range(len(C))])
    print(f"  takeover cost in misses per cache slot (opt), all cells and windows: median {np.median(cost):.2f}, IQR [{np.percentile(cost,25):.2f}, {np.percentile(cost,75):.2f}]")
    stats("v2 + replay 128", np.array([chg(V[i][8:].sum(), c) for i, c in enumerate(C)]), L3, T)
    for kind in ("opt", "pes"):  # hindsight: the best single switch window, or none
        best_w = [chg(min([V[i][8:].sum()] + [V[i][8:w].sum() + sum(G[i][str(w)][kind][w:]) for w in range(9, W)]), c) for i, c in enumerate(C)]
        stats(f"hindsight best switch window (real takeover, {kind})", np.array(best_w), L3, T)
    for rate in (0.05, 0.1, 0.2):
        E = [mini_estimate(c, OUT / suite / ds[c["t"]].name, rate) for c in C]
        for kappa in (0.0, 0.25, 0.5, 1.0):
            for horizon in (0, 8):
                res = {"opt": [], "pes": []}; nsw = 0
                for i, c in enumerate(C):
                    sw = None
                    for w in range(9, W):
                        lo = 8 if horizon == 0 else max(8, w - horizon)
                        gain = (V[i][lo:w].sum() - E[i][lo:w].sum()) / (w - lo)  # estimated misses saved per window by 3L-Cache
                        if gain > 0 and gain * (W - w) > kappa * c["cap"]: sw = w; break
                    nsw += sw is not None
                    for kind in res:
                        res[kind].append(chg(V[i][8:].sum() if sw is None else V[i][8:sw].sum() + sum(G[i][str(sw)][kind][sw:]), c))
                for kind in res: stats(f"rate {rate:g} kappa {kappa:g} H={horizon or 'all'} {kind} ({nsw} switch)", np.array(res[kind]), L3, T)


def cmd_takeover(suite, tag, plan_name, workers, L):
    plan = json.loads((OUT / suite / f"{plan_name}.json").read_text()); root = Path(f"/mydata/vq/{suite}")
    jobs = [(root / k.split("/")[0], int(k.split("/")[1]), w, tag, L) for k, w in plan.items()]
    with ProcessPoolExecutor(workers) as ex: list(ex.map(takeover_cell, jobs))
    C = load(suite, tag); ds = traces(suite); T = np.array([c["t"] for c in C]); L3 = np.array([c["l3"] for c in C])
    chg = lambda m, c: (m - c["base"]) / c["base"]; KINDS = ("A_opt", "A_pes", "B")
    opt, real, extra, info = [], {k: [] for k in KINDS}, {k: [] for k in KINDS}, []
    for c in C:
        key = f"{ds[c['t']].name}/{c['cap']}"; v2 = c["v2"]
        if key not in plan:
            opt.append(chg(sum(v2[8:]), c))
            for k in KINDS: real[k].append(opt[-1])
            continue
        w = plan[key]; tk = json.loads((OUT / suite / ds[c["t"]].name / f"take-{c['cap']}-w{w}-L{L:g}.json").read_text())
        opt.append(chg(sum(v2[8:w]) + sum(c["f3"][w:]), c)); info.append((tk["A_phantoms"] / max(tk["live"], 1), tk["start"] == 0))
        for k in KINDS: real[k].append(chg(sum(v2[8:w]) + sum(tk[k][w:]), c)); extra[k].append((sum(tk[k][w:]) - sum(c["f3"][w:])) / c["base"])
    print(f"{suite}: {plan_name}, takeover from the last {L:g}*C requests ({len(plan)} cells switch;"
          f" replay covers the whole history in {sum(1 for _, s in info if s)}; phantoms after a full replay: median {100*np.median([p for p, _ in info]) if info else 0:.1f}% of C)")
    for k in KINDS:
        if extra[k]: print(f"  takeover cost {k:<5} per switching cell vs 3L-Cache from the start: median {100*np.median(extra[k]):+.2f} pt, mean {100*np.mean(extra[k]):+.2f} pt")
    v2 = np.array([chg(sum(c["v2"][8:]), c) for c in C])
    stats("v2 + replay 128", v2, L3, T); stats("switch to 3L-Cache, no takeover cost", np.array(opt), L3, T)
    for k in KINDS: stats(f"switch to 3L-Cache, takeover {k}", np.array(real[k]), L3, T)


CLASSIC = ("LRU", "FIFO", "CLOCK", "SIEVE", "ARC", "LIRS", "2Q", "CLOCK-Pro", "MQ", "S3-FIFO", "W-TinyLFU", "LeCaR", "Cacheus", "LHD")


def family(d): return d.name.split("-")[1]  # build-suite names held-out traces PREFIX-DATASET-INDEX


def cmd_drop(suite):
    """The mechanical drop rule, from the baselines alone: a trace goes if S3-FIFO (libCacheSim or vqsim) or 3L-Cache has no
    misses after the split in one of its cells. Run after r1 and finish, before any candidate."""
    gone = Path(f"/mydata/vq/{suite}-dropped"); kept = 0
    for d in traces(suite):
        meta = json.loads((d / "meta.json").read_text())
        base = [sum(int(x) for x in (d / f"r1-{c}.out").read_text().splitlines()[0].split()[2:][8:]) for c in meta["capacities"]]
        if min(base) > 0 and min(meta["s3fifo"].values()) > 0 and min(meta["3L"].values()) > 0: kept += 1; continue
        gone.mkdir(exist_ok=True); d.rename(gone / d.name)
    print(f"  {suite}: drop rule (no misses after the split for S3-FIFO or 3L-Cache in a cell): {kept} traces kept,"
          f" {len(list(gone.iterdir())) if gone.is_dir() else 0} dropped")


def cmd_zoo(suite, workers):
    """The 14 published classic algorithms and Belady, cold over the whole trace, misses after the split (zoo.py's runner)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from vista.benchmarks.cachelab import zoo
    jobs = []
    for d in traces(suite):
        meta = json.loads((d / "meta.json").read_text()); tdir = OUT / suite / d.name; tdir.mkdir(parents=True, exist_ok=True)
        if not (tdir / "full.og").is_file():  # shared with the mini runs, which remove it
            ids = np.fromfile(d / "ids.u32", dtype=np.uint32); write_og(ids, np.arange(ids.size, dtype=np.uint32), tdir / "full.og")
        for cap in meta["capacities"]:
            for alg in CLASSIC + ("Belady",):
                out = tdir / f"zoo-{alg}-{cap}.json"
                if not out.is_file(): jobs.append((alg, str(tdir / "full.og"), int(cap), int(meta["split"]), int(meta["requests"]), str(out)))
    jobs.sort(key=lambda j: -j[4])
    with ProcessPoolExecutor(workers) as ex: res = list(ex.map(zoo._job, jobs, chunksize=1))
    print(f"  {suite}: zoo, {len(jobs)} runs, {sum('error' in r for r in res)} errors")


def zoo_cells(suite):
    """Per cell: each algorithm's suffix miss ratio relative to S3-FIFO's (both libCacheSim), as for 3L-Cache in load()."""
    rows = []
    for ti, d in enumerate(traces(suite)):
        meta = json.loads((d / "meta.json").read_text()); tdir = OUT / suite / d.name
        for cap in meta["capacities"]:
            s3 = meta["s3fifo"][str(cap)]; r = {}
            for alg in CLASSIC + ("Belady",):
                z = json.loads((tdir / f"zoo-{alg}-{cap}.json").read_text())
                r[alg] = np.nan if "error" in z else (z["suffix_miss_ratio"] - s3) / s3
            rows.append(dict(t=ti, cap=cap, l3=(meta["3L"][str(cap)] - s3) / s3, z=r))
    return rows


def cmd_bound(suite):
    """The classic-oracle bound: in hindsight and with no switching cost, the best of the 14 classic algorithms per cell."""
    R = zoo_cells(suite); T = np.array([r["t"] for r in R]); L3 = np.array([r["l3"] for r in R])
    print(f"{suite}: {len(np.unique(T))} traces, {len(R)} cells; change in suffix misses vs S3-FIFO; 3L-Cache cold over the whole trace")
    print(f"  check: the zoo's S3-FIFO equals meta's in {sum(abs(r['z']['S3-FIFO']) < 1e-12 for r in R)}/{len(R)} cells")
    for alg in CLASSIC + ("Belady",):
        x = np.array([r["z"][alg] for r in R]); ok = ~np.isnan(x)
        print(f"  {alg:<10} mean {100*np.nanmean(x):+7.2f}%  median {100*np.nanmedian(x):+7.2f}%  cells better than 3L-Cache"
              f" {int((x[ok] < L3[ok] - 1e-12).sum()):>3}/{int(ok.sum())}{'' if ok.all() else f'  ({int((~ok).sum())} errors)'}")
    best = np.array([np.nanmin([r["z"][a] for a in CLASSIC]) for r in R])
    stats("3L-Cache", L3, L3, T); stats("classic oracle (best of 14 per cell)", best, L3, T)
    tr = np.unique(T); d = np.array([best[T == t].mean() - L3[T == t].mean() for t in tr])
    print(f"  3L-Cache beats every classic algorithm (per trace, cell mean) on {int((d >= 0).sum())}/{len(tr)} traces")


def boot(x, y, T, n=10000):
    """Mean of x minus mean of y over cells, resampling traces (n resamples, seed 0): the estimate and a 95% interval."""
    tr = np.unique(T); idx = [np.flatnonzero(T == t) for t in tr]; rng = np.random.default_rng(0); diffs = np.empty(n)
    for k in range(n):
        m = np.concatenate([idx[j] for j in rng.integers(0, len(tr), len(tr))]); diffs[k] = x[m].mean() - y[m].mean()
    return x.mean() - y.mean(), *np.percentile(diffs, [2.5, 97.5])


def cmd_test(suite, tag):
    """The pre-registered bar (PREREG-gstest.md) for v2 + replay 128, and the reported numbers around it."""
    C = load(suite, tag); ds = traces(suite); T = np.array([c["t"] for c in C]); L3 = np.array([c["l3"] for c in C])
    v2 = np.array([(sum(c["v2"][8:]) - c["base"]) / c["base"] for c in C]); tr = np.unique(T)
    print(f"{suite}: {len(tr)} traces, {len(C)} cells; change in misses on windows 8-39 vs S3-FIFO; 3L-Cache cold over the whole trace")
    bar = [(f"mean {100*v2.mean():+.2f}% below 3L-Cache's {100*L3.mean():+.2f}%", v2.mean() < L3.mean()),
           (f"median {100*np.median(v2):+.2f}% below 3L-Cache's {100*np.median(L3):+.2f}%", np.median(v2) < np.median(L3)),
           (f"beats 3L-Cache in {int((v2 < L3 - 1e-12).sum())}/{len(C)} cells, more than half", (v2 < L3 - 1e-12).sum() * 2 > len(C))]
    for text, ok in bar: print(f"  bar: {text}: {'yes' if ok else 'NO'}")
    print(f"  => {'PASS' if all(ok for _, ok in bar) else 'FAIL'}")
    est, lo, hi = boot(L3, v2, T)
    print(f"  lead over 3L-Cache (3L-Cache's mean minus v2's): {100*est:+.2f} pt, 95% interval [{100*lo:+.2f}, {100*hi:+.2f}] (10,000 trace resamples)")
    d = np.array([v2[T == t].mean() - L3[T == t].mean() for t in tr])
    print(f"  traces beating 3L-Cache (cell mean): {int((d < 0).sum())}/{len(tr)}; cells worse than S3-FIFO (> +0.05%):"
          f" v2 {int((v2 > 0.0005).sum())}, 3L-Cache {int((L3 > 0.0005).sum())}")
    rank = np.array([sorted(json.loads((ds[c["t"]] / "meta.json").read_text())["capacities"]).index(c["cap"]) for c in C])
    for k, name in enumerate(("1%", "3%", "10%", "30%")):
        m = rank == k; print(f"  cache {name:>3} of the prefix's objects: v2 {100*v2[m].mean():+.2f}%, 3L-Cache {100*L3[m].mean():+.2f}%, cells < 3L {int((v2[m] < L3[m] - 1e-12).sum())}/{int(m.sum())}")
    fam = np.array([family(ds[c["t"]]) for c in C])
    for f in sorted(set(fam)):
        m = fam == f; print(f"  family {f:<12} ({len(np.unique(T[m]))} traces): v2 {100*v2[m].mean():+.2f}%, 3L-Cache {100*L3[m].mean():+.2f}%")


def cmd_describe(suite, tag, rates):
    """The state problem, descriptively: when 3L-Cache's edge appears, and how faithful its miniature is."""
    C = load(suite, tag); ds = traces(suite); T = np.array([c["t"] for c in C])
    rel = lambda a, b: (a - b) / max(b, 1)
    pre = np.array([rel(sum(c["f3"][2:8]), sum(c["r1"][2:8])) for c in C]); suf = np.array([rel(sum(c["f3"][8:]), sum(c["r1"][8:])) for c in C])
    q = [np.array([rel(sum(c["f3"][a:b]), sum(c["r1"][a:b])) for c in C]) for a, b in ((8, 16), (16, 24), (24, 32), (32, 40))]
    print(f"{suite}: {len(np.unique(T))} traces, {len(C)} cells; 3L-Cache vs S3-FIFO at full scale, both cold from the start")
    print(f"  windows 2-7 (prefix): median {100*np.median(pre):+.2f}%, mean {100*pre.mean():+.2f}%;  windows 8-39: median {100*np.median(suf):+.2f}%,"
          f" mean {100*suf.mean():+.2f}%;  corr(prefix, suffix) {np.corrcoef(pre, suf)[0,1]:.2f}")
    print("  by quarter of the suffix (windows 8-15, 16-23, 24-31, 32-39): mean " + ", ".join(f"{100*x.mean():+.2f}%" for x in q))
    mini = np.array([rel(sum(c["m3"][2:8]), sum(c["ms3"][2:8])) for c in C])
    print(f"  at the selector's rate: miniature 3L-vs-S3 on windows 2-7 minus full scale: median {100*np.median(mini-pre):+.2f} pt,"
          f" IQR [{100*np.percentile(mini-pre,25):+.2f}, {100*np.percentile(mini-pre,75):+.2f}]; corr {np.corrcoef(mini, pre)[0,1]:.2f}")
    for r in rates:
        err = np.array([(mini_estimate(c, OUT / suite / ds[c["t"]].name, r)[8:].sum() - sum(c["f3"][8:])) / max(sum(c["f3"][8:]), 1) for c in C])
        print(f"  rate {r:g}: scaled 3L-Cache miniature vs full scale on windows 8-39: median {100*np.median(err):+.2f}%,"
              f" IQR [{100*np.percentile(err,25):+.2f}, {100*np.percentile(err,75):+.2f}]%")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "drop": cmd_drop(sys.argv[2]); sys.exit()
    if cmd == "zoo": cmd_zoo(sys.argv[2], int(sys.argv[3])); sys.exit()
    if cmd == "bound": cmd_bound(sys.argv[2]); sys.exit()
    if cmd == "test": cmd_test(sys.argv[2], sys.argv[3]); sys.exit()
    if cmd == "describe": cmd_describe(sys.argv[2], sys.argv[3], [float(x) for x in sys.argv[4].split(",")]); sys.exit()
    if cmd == "prepare": cmd_prepare(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif cmd == "mini": cmd_mini(sys.argv[2], int(sys.argv[3]))
    elif cmd == "finish": cmd_finish(sys.argv[2])
    elif cmd == "rates": cmd_rates(sys.argv[2], [float(x) for x in sys.argv[3].split(",")], int(sys.argv[4]))
    elif cmd == "trace": cmd_trace(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 2)
    elif cmd == "score": cmd_score(sys.argv[2], sys.argv[3])
    elif cmd == "plan": cmd_plan(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
    elif cmd == "grid": cmd_grid(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]) if len(sys.argv) > 5 else 128)
    elif cmd == "rules": cmd_rules(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 128)
    elif cmd == "takeover": cmd_takeover(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]) if len(sys.argv) > 6 else 128)
