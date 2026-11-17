"""Designs the board's pool chooses per training cell (static rule), most-chosen first, as JSON lines."""
import json, sys
from collections import Counter
from pathlib import Path
from vista.benchmarks.cachelab.architect import Board
board = Board(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), slots=8)
pool = board.pool(); chosen = board.choose(pool)
seeds = {json.loads(l)["config"] for l in Path(sys.argv[4]).read_text().splitlines()}
label = {}
for f in board.families():
    if f["stats"]:
        for lab, cfg in f["designs"]: label.setdefault(cfg, f"F{f['id']} {f['author']} {lab}")
count = Counter(chosen.values())
print(f"pool {len(pool)} designs; {len(count)} distinct chosen; {sum(c in seeds for c in count)} of them seeds", file=sys.stderr)
for cfg, k in count.most_common():
    print(json.dumps({"name": label.get(cfg, "s3fifo"), "family": "seed" if cfg in seeds else "agent", "config": cfg, "chosen": k}))
