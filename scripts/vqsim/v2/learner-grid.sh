#!/usr/bin/env bash
# Option (b) development study, step 3 (gs-dev only): the real takeover into 3L-Cache at every window, then cost-aware rules.
set -euo pipefail
exec > /mydata/learner-grid.log 2>&1
cd /work/VISTA; S=scripts/vqsim/study/vq_learner.py
gcc -O2 -march=native -o /mydata/vqsim-dump scripts/vqsim/vqsim.c
export VQSIM=/mydata/vqsim-dump
echo "== $(date +%T) grid"
python $S grid gs-dev gv2 90 128
echo "== $(date +%T) rules"
python $S rules gs-dev gv2 128
echo "== $(date +%T) done"
