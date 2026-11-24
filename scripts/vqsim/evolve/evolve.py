"""CacheLab evolve: a model writes single cache policies (policy.h) that run cold over whole traces, as 3L-Cache does;
each is checked, compiled and scored on a suite (train-long only while searching); the best feed the next prompt.

  evolve.py score POLICY.c [SUITE_DIR]                 score one policy (default suite /mydata/vq/train-long)
  evolve.py score-json POLICY.c [WORKERS] [full]        the same, as one JSON line (for remote scoring); full adds
                                                        each knob setting's score when held fixed
  evolve.py run ROOT [--iters N] [--parallel P] [--ideas OLD_ROOT]
                                                        search; state in ROOT (programs, scores, replies); --ideas
                                                        folds mechanisms from an earlier run's programs in as settings

Scoring: per cell, the change in misses vs S3-FIFO over windows 8-39 (the last 80%; S3-FIFO's cold run is line 0 of
r1-<cap>.out), against 3L-Cache's (meta.json). A trace beats 3L-Cache if its mean over its cells is lower. Feedback to
the model is anonymous (trace indices, never names). Model access (OpenAI-compatible chat/completions) comes from the
environment: CACHELAB_LLM_BASE_URL, CACHELAB_LLM_MODEL (a comma-separated list is used in turn), CACHELAB_LLM_API_KEY.
CACHELAB_SCORE_HOST=user@host scores on that host instead (the program is copied to /mydata/evolve/progs there; the
host runs this script from ~/VISTA). The key never leaves the machine that runs `run`."""
import json, os, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SUITE = Path("/mydata/vq/train-long")
ALLOWED = {"memset", "memcpy", "memmove", "memcmp", "sqrt", "sqrtf", "log", "logf", "log2", "log2f", "exp", "expf", "pow", "powf",
           "fabs", "floor", "ceil", "fmin", "fmax", "lround", "__stack_chk_fail", "__memset_chk", "__memcpy_chk"}
MAX_STATIC = 1 << 16


def build(src: Path, work: Path) -> tuple[Path | None, str]:
    """Compile and check a policy: no heap, files or large static data. Returns (polsim binary, error)."""
    obj, binary = work / "policy.o", work / "polsim"
    r = subprocess.run(["cc", "-O2", "-std=gnu11", "-I", str(HERE), "-c", str(src), "-o", str(obj)], capture_output=True, text=True, timeout=120)
    if r.returncode: return None, "compile error:\n" + r.stderr[-3000:]
    undef = set(subprocess.run(["nm", "-u", str(obj)], capture_output=True, text=True).stdout.split()) - {"U"}
    bad = sorted(u for u in undef if u not in ALLOWED)
    if bad: return None, f"uses forbidden symbols: {', '.join(bad)} (no heap, files or I/O; all state in the arena)"
    sz = subprocess.run(["size", str(obj)], capture_output=True, text=True).stdout.splitlines()[-1].split()
    if int(sz[1]) + int(sz[2]) > MAX_STATIC: return None, f"static data {int(sz[1]) + int(sz[2])} bytes > {MAX_STATIC} (keep state in the arena)"
    r = subprocess.run(["cc", "-O2", "-o", str(binary), str(HERE / "polsim.c"), str(obj)], capture_output=True, text=True, timeout=120)
    if r.returncode: return None, "link error:\n" + r.stderr[-3000:]
    return binary, ""


def cells(suite: Path):
    for d in sorted(suite.iterdir()):
        if (d / "meta.json").is_file():
            yield d, json.loads((d / "meta.json").read_text())


def score(src: Path, suite: Path = SUITE, workers: int = 48, slow: float = 20.0, knob: int | None = None, full: bool = False) -> dict:
    """Run a policy on every cell of `suite`; aggregate vs S3-FIFO and 3L-Cache. With knobs, the harness picks the setting
    (polsim's miniatures); knob=k holds setting k fixed; full adds every setting's fixed score."""
    env = dict(os.environ, **({"POLSIM_KNOB": str(knob)} if knob is not None else {}))
    env.pop("CACHELAB_LLM_API_KEY", None)
    with tempfile.TemporaryDirectory() as tmp:
        binary, err = build(src, Path(tmp))
        if not binary: return {"ok": False, "error": err}
        K = int(subprocess.run([str(binary), "--knobs"], capture_output=True, text=True, timeout=30).stdout.strip() or 0)
        if not 1 <= K <= 8: return {"ok": False, "error": "runtime: policy_knobs() must return 1..8"}
        traces = list(cells(suite))

        def one(item):
            d, meta = item
            n = (d / "ids.u32").stat().st_size // 4
            limit = 60 + slow * n * len(meta["capacities"]) / 2e7  # S3-FIFO runs at roughly 2e7 requests/s
            try:
                r = subprocess.run(["nice", "-n", "10", str(binary), str(d / "ids.u32"), str(meta["split"]), "40"] + [str(c) for c in meta["capacities"]],
                                   capture_output=True, text=True, timeout=limit, env=env)
            except subprocess.TimeoutExpired:
                return d.name, None, f"too slow (over {limit:.0f} s on one trace)"
            lines = r.stdout.splitlines()
            if r.returncode or len(lines) != len(meta["capacities"]) or any(l.startswith("error") for l in lines):
                return d.name, None, (r.stderr[-500:] or "") + " ".join(l for l in lines if l.startswith("error")) or f"exit {r.returncode}"
            kn = {int(x.split()[1]): (int(x.split()[2]), int(x.split()[3])) for x in r.stderr.splitlines() if x.startswith("knobs ")}
            out = []
            for cap, l in zip(meta["capacities"], lines):
                W0 = [int(x) for x in (d / f"r1-{cap}.out").read_text().splitlines()[0].split()[2:]]; base = sum(W0[8:])
                s3 = meta["s3fifo"][str(cap)]
                out.append(((sum(int(x) for x in l.split()[2 + 8:]) - base) / base, (meta["3L"][str(cap)] - s3) / s3, 100 * cap / meta["objects"],
                            *kn.get(int(cap), (0, 0))))
            return d.name, out, ""
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex: got = list(ex.map(one, traces))
    fails = [e for _, o, e in got if o is None]
    if fails: return {"ok": False, "error": "runtime: " + fails[0][:1500]}
    per, rows = [], []
    for i, (_, out, _) in enumerate(got):
        x = np.array([o[0] for o in out]); l3 = np.array([o[1] for o in out])
        per.append((x.mean(), l3.mean())); rows += [(i, *o) for o in out]
    P = np.array(per); R = np.array([r[1] for r in rows]); L = np.array([r[2] for r in rows]); Z = np.array([r[3] for r in rows])
    sizes = {lab: (float(R[m].mean()), float(L[m].mean())) for lab, m in
             (("<0.5%", Z < 0.5), ("0.5-2%", (Z >= 0.5) & (Z < 2)), ("2-8%", (Z >= 2) & (Z < 8)), (">=8%", Z >= 8)) if m.any()}
    out = {"ok": True, "mean": float(R.mean()), "median": float(np.median(R)), "l3_mean": float(L.mean()), "l3_median": float(np.median(L)),
           "traces_beating_3l": int((P[:, 0] < P[:, 1]).sum()), "traces": len(P), "cells_beating_3l": int((R < L).sum()), "cells": len(R),
           "worse_than_s3": int((R > 0.0005).sum()), "per_trace": [[float(a), float(b)] for a, b in P], "sizes": sizes, "knobs": K}
    if K > 1 and knob is None:
        out["changes_median"] = float(np.median([r[4] for r in rows]))
        out["final_setting"] = [int(sum(1 for r in rows if r[5] == k)) for k in range(K)]
        if full:
            fixed = [score(src, suite, workers, slow, knob=k) for k in range(K)]
            if all(f["ok"] for f in fixed):
                F = np.array([[a for a, _ in f["per_trace"]] for f in fixed])
                out["fixed"] = [{"mean": f["mean"], "traces_beating_3l": f["traces_beating_3l"], "worse_than_s3": f["worse_than_s3"],
                                 "best_on": int((F.argmin(0) == k).sum())} for k, f in enumerate(fixed)]
                out["oracle_traces"] = int((F.min(0) < P[:, 1]).sum())
    out["seconds"] = time.time() - t0
    return out


def summary(s: dict) -> str:
    if not s["ok"]: return "FAILED: " + s["error"]
    lines = [f"mean {100*s['mean']:+.2f}% median {100*s['median']:+.2f}% vs S3-FIFO (3L-Cache {100*s['l3_mean']:+.2f}% / {100*s['l3_median']:+.2f}%); "
             f"traces beating 3L-Cache {s['traces_beating_3l']}/{s['traces']}; cells {s['cells_beating_3l']}/{s['cells']}; worse than S3-FIFO in {s['worse_than_s3']} cells"]
    lines.append("by cache size (% of objects): " + "; ".join(f"{k} ours {100*a:+.2f}% 3L {100*b:+.2f}%" for k, (a, b) in s["sizes"].items()))
    lines.append("per trace (anonymous index: ours - 3L-Cache, pt): " + " ".join(f"t{i:02d}:{100*(a-b):+.2f}" for i, (a, b) in enumerate(s["per_trace"])))
    if s.get("knobs", 1) > 1 and "final_setting" in s:
        lines.append(f"settings: {s['knobs']}; harness changes per cell (median) {s['changes_median']:g}; cells ending on each setting: "
                     + " ".join(f"{k}:{n}" for k, n in enumerate(s["final_setting"])))
        if "fixed" in s:
            lines.append("each setting held fixed (no harness): " + "; ".join(
                f"#{k} mean {100*f['mean']:+.2f}% beats 3L on {f['traces_beating_3l']} traces, best fixed setting on {f['best_on']}"
                for k, f in enumerate(s["fixed"])) + f"; best fixed setting per trace (hindsight) would beat 3L on {s['oracle_traces']} traces")
    return "\n".join(lines)


def remote_score(src: Path, workers: int, full: bool = True) -> dict:
    host = os.environ["CACHELAB_SCORE_HOST"]; dst = f"/mydata/evolve/progs/{src.name}"
    ssh = ["ssh", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", host]
    subprocess.run(ssh + ["mkdir -p /mydata/evolve/progs"], check=True, capture_output=True, timeout=120)
    subprocess.run(["scp", "-q", str(src), f"{host}:{dst}"], check=True, capture_output=True, timeout=120)
    r = subprocess.run(ssh + [f"cd ~/VISTA && .venv/bin/python scripts/vqsim/evolve/evolve.py score-json {dst} {workers}{' full' if full else ''}"],
                       capture_output=True, text=True, timeout=7200)
    try: return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "scoring host: " + (r.stderr or r.stdout)[-1500:]}


def chat(messages: list[dict], model: str) -> str:
    base, key = os.environ.get("CACHELAB_LLM_BASE_URL"), os.environ.get("CACHELAB_LLM_API_KEY")
    if not (base and key): raise SystemExit("set CACHELAB_LLM_BASE_URL, CACHELAB_LLM_MODEL and CACHELAB_LLM_API_KEY")
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", method="POST",
                                 data=json.dumps({"model": model, "messages": messages}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=1200) as r: return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001 - retry transient gateway errors
            if attempt == 4: raise
            time.sleep(15 * (attempt + 1))
    return ""


SYSTEM = """You design cache eviction policies. Goal: beat 3L-Cache (a learned eviction policy, FAST'25) on as many
traces as possible, by miss count, while keeping its simplicity. Policies run cold over whole real block/CDN/storage
traces; the score is the change in misses vs S3-FIFO over the last 80% of each trace, at four cache sizes per trace
(about 0.1% to 10% of the trace's objects). A trace counts as a win if its mean change beats 3L-Cache's.
Write one complete C file implementing policy.h, including its knob: 2 to 8 settings of one cache structure. The harness
runs a miniature copy of your policy per setting on a sample of the keys and moves the live cache to whichever setting
misses least, so each workload gets its best setting. Setting 0 must behave as S3-FIFO (the reference). The other
settings should each be a different bet (a mechanism, or a mechanism plus parameters) that wins big on some kinds of
workload: a setting that loses on most workloads costs little, because the harness will not pick it there. A setting
change happens on the running cache: it must take effect on the existing contents without a rebuild (keep one shared
structure and metadata; settings change decisions, not layout).
All state in the arena (at most 256 bytes per cache slot + 1 MiB), no malloc, no files, no large static data. Keys are
random-looking 64-bit hashes; use your own hash table in the arena. Be careful: the driver checks every victim is
cached; a crash, a wrong victim or a run over 20x S3-FIFO's time fails.
Reply with a short rationale (a few sentences), then the whole file in one ```c block."""


def prompt(mode: str, best: list[tuple[str, dict]], seed: str, seed_score: dict | None, idea: tuple[str, str] | None = None) -> list[dict]:
    parts = ["policy.h:\n```c\n" + (HERE / "policy.h").read_text() + "```",
             "Reference: S3-FIFO with one knob, the small queue's share (setting 0 is plain S3-FIFO)"
             + (f"; on the training traces:\n{summary(seed_score)}" if seed_score else "") + "\n```c\n" + seed + "```",
             "What earlier work found:\n" + NOTES]
    for i, (code, s) in enumerate(best):
        parts.append(f"Earlier policy #{i + 1}, results on the training traces:\n{summary(s)}\n```c\n{code}```")
    if mode == "fold" and idea:
        parts.append("A mechanism from an earlier search (a fixed policy without knobs). It wins big on some traces and loses on "
                     f"others; per trace vs S3-FIFO (pt): {idea[1]}\n```c\n{idea[0]}```")
        parts.append(f"Write a new policy: {'earlier policy #1' if best else 'the reference'} with this mechanism folded in as one of its "
                     "settings (replace the setting that helps least if there are already 8). Keep one shared structure so a setting change "
                     "needs no rebuild.")
    elif mode == "explore":
        parts.append("Write a new policy whose settings include a substantially different mechanism from those above (not a re-tuning), "
                     "aimed at the traces where the policies above lose to 3L-Cache.")
    else:
        parts.append("Write a new policy that wins on more traces than any above: study where it loses (per trace, per size, which "
                     "settings carry its wins) and fix that: improve or replace the settings that never help, and add bets for the "
                     "traces it loses.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


NOTES = """- No single fixed FIFO-family design beats 3L-Cache on more than 13 of 25 training traces; picking the best design per
  trace (in hindsight) would win 23/25, so the win is in adapting to the workload.
- Switching a running cache between different designs (structures) costs 0.4-0.6 pt on long traces; changing decisions
  within one structure is far cheaper, which is why settings must share one structure.
- An earlier search for single fixed policies found mechanisms that gain 7-15 pt on some traces but lose 4-12 pt on
  others; any one of them beat 3L-Cache on at most 13 traces. As settings, their losses mostly disappear.
- In the reference, every fixed share other than 10% loses to S3-FIFO on average (+0.7% to +8.3%), yet the harness
  picking the share per workload gives -2.14%: settings only need to be right somewhere.
- Helpful mechanisms seen: a ghost hit whose reuse distance is long can go to the small queue instead of main (big
  caches); TinyLFU-style admission into main; more than one main level; frequency-aware eviction in main."""


def run(root: Path, iters: int, parallel: int, ideas: Path | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True); (root / "programs").mkdir(exist_ok=True)
    db = root / "scores.jsonl"
    seed = (HERE / "seeds" / "s3fifo_share.c").read_text()
    models = [m.strip() for m in os.environ.get("CACHELAB_LLM_MODEL", "").split(",") if m.strip()]
    if not models: raise SystemExit("set CACHELAB_LLM_MODEL")
    workers = max(8, 60 // parallel)
    do_score = (lambda src: remote_score(src, workers)) if os.environ.get("CACHELAB_SCORE_HOST") else (lambda src: score(src, workers=workers, full=True))
    pop = [json.loads(l) for l in db.read_text().splitlines()] if db.is_file() else []
    seed_score = next((e["score"] for e in pop if e["id"] == "seed"), None)
    if seed_score is None:
        (root / "programs" / "seed.c").write_text(seed); seed_score = do_score(root / "programs" / "seed.c")
        e = {"id": "seed", "model": "-", "mode": "seed", "score": seed_score}; pop.append(e)
        with db.open("a") as f: f.write(json.dumps(e) + "\n")
        print("seed: " + summary(seed_score).splitlines()[0], flush=True)
    pool = []  # mechanisms from an earlier run: fixed policies that gain >= 3 pt over S3-FIFO on at least one trace
    if ideas and (ideas / "scores.jsonl").is_file():
        for l in (ideas / "scores.jsonl").read_text().splitlines():
            e = json.loads(l); s = e["score"]
            if e["id"] != "seed" and s["ok"] and s.get("knobs", 1) == 1 and min(a for a, _ in s["per_trace"]) < -0.03:
                pool.append(((ideas / "programs" / f"{e['id']}.c").read_text(), " ".join(f"t{i:02d}:{100*a:+.1f}" for i, (a, _) in enumerate(s["per_trace"]))))
    def key(e): s = e["score"]; return (s["traces_beating_3l"], -s["mean"]) if s["ok"] else (-1, 0)
    def attempt(it, i):
        model = models[(it * parallel + i) % len(models)]
        mode = ["refine", "explore", "refine", "fold"][i % 4]
        if mode == "fold" and not pool: mode = "explore"
        ok = [e for e in pop if e["score"]["ok"] and e["score"].get("knobs", 1) > 1]
        best = sorted(ok, key=key, reverse=True)[: (3 if mode == "refine" else 1)]
        idea = pool[(it * parallel + i) % len(pool)] if mode == "fold" else None
        msgs = prompt(mode, [((root / "programs" / f"{e['id']}.c").read_text(), e["score"]) for e in best if e["id"] != "seed"], seed, seed_score, idea)
        pid = f"p{it:03d}-{i}"
        for fix in range(3):  # the first try and up to two repairs
            try: reply = chat(msgs, model)
            except Exception as ex:  # noqa: BLE001
                return {"id": pid, "model": model, "mode": mode, "score": {"ok": False, "error": f"model call: {ex}"}}
            (root / "programs" / f"{pid}.reply{fix}.md").write_text(reply)
            m = re.findall(r"```(?:c|C)\s*\n(.*?)```", reply, re.S)
            if not m: s = {"ok": False, "error": "no ```c block in the reply"}
            else:
                src = root / "programs" / f"{pid}.c"; src.write_text(m[-1]); s = do_score(src)
            if s["ok"] or not any(w in s["error"] for w in ("compile", "link", "forbidden", "static data", "runtime", "no ```c")): break
            msgs = msgs + [{"role": "assistant", "content": reply},
                           {"role": "user", "content": f"That failed: {s['error'][:2500]}\nFix it and reply with the whole corrected file in one ```c block."}]
        return {"id": pid, "model": model, "mode": mode, "fixes": fix, "score": s}
    start = 1 + max([int(e["id"][1:4]) for e in pop if e["id"].startswith("p")] or [-1])
    for it in range(start, start + iters):
        with ThreadPoolExecutor(parallel) as ex: new = list(ex.map(lambda i: attempt(it, i), range(parallel)))
        for e in new:
            pop.append(e)
            with db.open("a") as f: f.write(json.dumps(e) + "\n")
            s = e["score"]; extra = f"; settings {s['knobs']}, hindsight-best setting would win {s.get('oracle_traces', '-')}" if s["ok"] and s.get("knobs", 1) > 1 else ""
            print(f"[{time.strftime('%H:%M:%S')}] {e['id']} {e['model']} {e['mode']}: " + summary(s).splitlines()[0] + extra, flush=True)
        if all(e["score"].get("error", "").startswith("model call") for e in new):
            print("every model call in this round failed (quota or gateway down); stopping", flush=True); return
        b = max((e for e in pop if e["score"]["ok"] and e["id"] != "seed"), key=key, default=None)
        if b: print(f"  best so far {b['id']}: {b['score']['traces_beating_3l']}/{b['score']['traces']} traces, mean {100*b['score']['mean']:+.2f}%", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "score-json":
        print(json.dumps(score(Path(sys.argv[2]), workers=int(sys.argv[3]) if len(sys.argv) > 3 else 48, full="full" in sys.argv[4:])))
    elif len(sys.argv) >= 3 and sys.argv[1] == "score":
        s = score(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else SUITE); print(summary(s)); print(f"({s.get('seconds', 0):.0f} s)")
    elif len(sys.argv) >= 3 and sys.argv[1] == "run":
        a = sys.argv[3:]; it = int(a[a.index("--iters") + 1]) if "--iters" in a else 10; par = int(a[a.index("--parallel") + 1]) if "--parallel" in a else 4
        run(Path(sys.argv[2]), it, par, Path(a[a.index("--ideas") + 1]) if "--ideas" in a else None)
    else:
        print(__doc__)
