"""Dense object ids (uint32) for every trace of a suite, plus split and cache sizes from its baselines."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "VISTA-r2/src"))
from vista.benchmarks.cachelab import engineer
from vista.benchmarks.cachelab.traces import load_suite
suite, baselines, root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for spec in load_suite(suite):
    d = root / spec.name; d.mkdir(parents=True, exist_ok=True)
    if (d / "meta.json").is_file(): continue
    rep = json.loads((baselines / f"{spec.name}.json").read_text())
    task = engineer.prepare(spec, d / "lab", fraction=rep["split_fraction"])
    _, dense = np.unique(task.full.keys(), return_inverse=True)
    dense.astype(np.uint32).tofile(d / "ids.u32")
    rows = rep["rows"]; s3 = {r["capacity"]: r["s3fifo"] for r in rows}
    meta = {"requests": int(dense.size), "objects": int(dense.max()) + 1, "split": rep["split"], "capacities": [r["capacity"] for r in rows],
            "s3fifo": s3, "3L": {r["capacity"]: r["published"]["3L-Cache"] for r in rows},
            "best": {r["capacity"]: [r["best_published"], r["best_published_miss_ratio"]] for r in rows},
            "tuned": {r["capacity"]: r["prefix_tuned_s3fifo"] for r in rows}}
    (d / "meta.json").write_text(json.dumps(meta))
    import shutil; shutil.rmtree(d / "lab", ignore_errors=True)
    print(spec.name, dense.size, flush=True)
