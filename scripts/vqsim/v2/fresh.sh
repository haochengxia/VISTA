#!/usr/bin/env bash
# CacheLab v2 + replay takeover (128): the pre-registered fresh-long test (scripts/vqsim/v2/PREREG-fresh.md).
# Runs on node0 from ~/VISTA after the baselines are gathered in results/engineer/fresh-long. Do not edit after launch.
set -euo pipefail
cd ~/VISTA
echo "== $(date +%T) drop rule (baselines failed, or zero scored misses for S3-FIFO or 3L-Cache in a cell)"
.venv/bin/python - <<'EOF'
import json
from pathlib import Path
suite = json.load(open("suites/fresh-long.json")); keep = []
for s in suite:
    f = Path(f"results/engineer/fresh-long/{s['name']}.json")
    if not f.is_file(): print("  dropped one trace: baselines failed"); continue
    rows = json.loads(f.read_text())["rows"]
    if any(not r["s3fifo"] or not r["published"].get("3L-Cache") for r in rows): print("  dropped one trace: zero scored misses"); continue
    keep.append(s)
open("suites/fresh-long.scored.json", "w").write(json.dumps(keep, indent=1) + "\n")
print(f"  scored: {len(keep)} of {len(suite)} traces, {sum(s['requests'] for s in keep)/1e6:.1f}M requests")
EOF
sha256sum suites/fresh-long.json suites/fresh-long.scored.json scripts/vqsim/vqsim.c scripts/vqsim/vqcore.h scripts/vqsim/study/vq_portfolio_real.py \
  scripts/vqsim/study/vq_prepare.py scripts/vqsim/study/vq_run3.py scripts/vqsim/v1/portfolio.json scripts/vqsim/v2/pool.txt \
  scripts/vqsim/v2/PREREG-fresh.md scripts/vqsim/v2/fresh_stats.py scripts/vqsim/v2/fresh.sh \
  scripts/vqsim/evolve/evolve.py scripts/vqsim/evolve/polsim.c scripts/vqsim/evolve/policy.h /mydata/evolve/progs/p024-2.c
echo "== $(date +%T) vq_prepare"
nice -n 5 .venv/bin/python scripts/vqsim/study/vq_prepare.py suites/fresh-long.scored.json results/engineer/fresh-long /mydata/vq/fresh-long | wc -l
echo "== $(date +%T) S3-FIFO runs (r1 line 0)"
.venv/bin/python scripts/vqsim/study/vq_run3.py /mydata/vq/fresh-long /mydata/s3only.jsonl r1 32 | tail -1
gcc -O2 -march=native -o /mydata/vqsim-fresh scripts/vqsim/vqsim.c
export VQSIM=/mydata/vqsim-fresh POOL=$HOME/VISTA/scripts/vqsim/v2/pool.txt
V1=32:0.005:200:0:0:0:0:0:0:0:0; V2=32:0.005:200:0:0:16:1:65536:1:0:0
echo "== $(date +%T) candidate (v2 + replay 128) and references (v1, v2, v1 + replay 128)"
.venv/bin/python scripts/vqsim/study/vq_portfolio_real.py /mydata/vq/fresh-long scripts/vqsim/v1/portfolio.json fr 30 $V2:128 $V1 $V2 $V1:128
echo "== $(date +%T) the pre-registered statistics"
.venv/bin/python scripts/vqsim/v2/fresh_stats.py /mydata/vq/fresh-long fr v2-replay128 v1 v2 v1-replay128
echo "== $(date +%T) reference: the knob search's best single policy (p024-2), polsim cold"
.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts/vqsim/evolve'); import evolve; from pathlib import Path
s = evolve.score(Path('/mydata/evolve/progs/p024-2.c'), Path('/mydata/vq/fresh-long'), workers=40)
print('  p024-2:', evolve.summary(s).splitlines()[0] if s['ok'] else s['error'][:300])"
echo "== $(date +%T) done"
