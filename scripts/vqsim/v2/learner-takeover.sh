#!/usr/bin/env bash
# Option (b) development study, step 2 (gs-dev only): realistic one-way switch plans and their real takeover cost.
set -euo pipefail
exec > /mydata/learner-takeover.log 2>&1
cd /work/VISTA; S=scripts/vqsim/study/vq_learner.py
gcc -O2 -march=native -o /mydata/vqsim-dump scripts/vqsim/vqsim.c
for spec in "0.1 0 0" "0.1 0.02 0" "0.2 0 0" "0.1 0 8"; do python $S plan gs-dev gv2 $spec; done
export VQSIM=/mydata/vqsim-dump
python $S takeover gs-dev gv2 plan-r0.1-m0-h0 64 128
python $S takeover gs-dev gv2 plan-r0.1-m0-h0 64 0
python $S takeover gs-dev gv2 plan-r0.1-m0.02-h0 64 128
python $S takeover gs-dev gv2 plan-r0.2-m0-h0 64 128
python $S takeover gs-dev gv2 plan-r0.1-m0-h8 64 128
echo "== $(date +%T) done"
