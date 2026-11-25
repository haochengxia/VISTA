#!/usr/bin/env bash
# Option (b) development study on a new development suite (gs-dev), on a single server in a container.
# The suite: whole traces of 10M-50M requests from the datasets available there; development only (never a test).
# Container mounts: the repo's parent at /work, a data directory at /mydata, the trace datasets at /traces (read only).
set -euo pipefail
exec > /mydata/learner-dev.log 2>&1
cd /work/VISTA; export PYTHONPATH=/work/VISTA/src
S=scripts/vqsim/study/vq_learner.py; SUITE=gs-dev
echo "== $(date +%T) vqsim"
gcc -O2 -march=native -o /mydata/vqsim3 scripts/vqsim/vqsim.c && cp /mydata/vqsim3 /mydata/vqsim-fresh
echo "== $(date +%T) suite"
mkdir -p /mydata/suites
[ -f /mydata/suites/$SUITE.json ] || python -m vista.benchmarks.cachelab.cli build-suite --data-root /traces \
  --datasets msr,cloudphysics,alibabaBlock,tencentBlock --per-dataset 8 --split heldout --full-length 10e6:50e6 \
  --name-prefix gd --seed 20261124 --out /mydata/suites/$SUITE.json
sha256sum /mydata/suites/$SUITE.json
echo "== $(date +%T) prepare (dense ids, split, cache sizes)"
python $S prepare /mydata/suites/$SUITE.json $SUITE 32
echo "== $(date +%T) S3-FIFO in vqsim (r1 line 0)"
python - <<'EOF'
import json; p = json.load(open("scripts/vqsim/v1/portfolio.json"))["portfolio"][0]
open("/mydata/s3only.jsonl", "w").write(json.dumps({"config": p}) + "\n")
EOF
python scripts/vqsim/study/vq_run3.py /mydata/vq/$SUITE /mydata/s3only.jsonl r1 64
echo "== $(date +%T) 3L-Cache and S3-FIFO per window, full scale and miniatures"
python $S mini $SUITE 96
python $S finish $SUITE
echo "== $(date +%T) v2 + replay 128 (v1's frozen 32 + 16 screened from pool.txt), then with VQSIM_TRACE"
VQSIM=/mydata/vqsim-fresh POOL=/work/VISTA/scripts/vqsim/v2/pool.txt python scripts/vqsim/study/vq_portfolio_real.py \
  /mydata/vq/$SUITE scripts/vqsim/v1/portfolio.json gv2 64 32:0.005:200:0:0:16:1:65536:1:0:0:128
python $S trace $SUITE gv2 64 0
echo "== $(date +%T) score"
python $S score $SUITE gv2
echo "== $(date +%T) done"
