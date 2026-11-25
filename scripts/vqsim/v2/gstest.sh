#!/usr/bin/env bash
# gs-test, the pre-registered test of PREREG-gstest.md, on a single server in a container.
#   gstest.sh A          the suite, the content check, preparation, the baselines and the drop rules
#   gstest.sh B SHA256   the candidate and every analysis; SHA256 is PREREG-gstest.md's, recorded after stage A
# Container mounts: the repo's parent at /work, a data directory at /mydata, the trace datasets at /traces (read only).
# The candidate's simulator is vqsim.c as of commit 5c14ae8, copied to /work/vqsim-5c14ae8.c.
set -euo pipefail
STAGE=$1
exec > /mydata/gstest-$STAGE.log 2>&1
cd /work/VISTA; export PYTHONPATH=/work/VISTA/src
S=scripts/vqsim/study/vq_learner.py; SUITE=gs-test; G=4; RATES=0.02,0.05,0.1,0.2
SPEC=32:0.005:200:0:0:16:1:65536:1:0:0:128
check() { echo "$2  $1" | sha256sum -c --strict -; }
check_candidate() {
  check /work/vqsim-5c14ae8.c 355417e737c231ab10d5db00fedf7b4f763e25283904d6a8a33f89e3f0e277dd
  check scripts/vqsim/vqcore.h 8c5193d00d8bd4cb656993cf9610116ff0ac9363ba88d9b740e498c485957da4
  check scripts/vqsim/study/vq_portfolio_real.py 832c83f114f0b27121abaae67a09b864651b5ae9ece39fde0ca56250126215de
  check scripts/vqsim/study/vq_run3.py f807bab8ad5d2dec2c75fa45f2d55663b568f703cdda51300faf9010bd6a6e9b
  check scripts/vqsim/v1/portfolio.json 0e96ab8ff3a84c61081dd534f8db5ddfaf72bebca90da37348e89a87fce3d7f0
  check scripts/vqsim/v2/pool.txt 0eb708a62a81c5085f6639d88d294b89cff41711cbfff44f2661d168e84073ea
}

if [ "$STAGE" = A ]; then
  echo "== $(date +%T) simulators: the candidate's (commit 5c14ae8) and the dumping build for the takeover grid"
  check_candidate
  gcc -O2 -march=native -I scripts/vqsim -o /mydata/vqsim3 /work/vqsim-5c14ae8.c && cp /mydata/vqsim3 /mydata/vqsim-fresh
  gcc -O2 -march=native -o /mydata/vqsim-dump scripts/vqsim/vqsim.c
  sha256sum scripts/vqsim/vqsim.c
  echo "== $(date +%T) suite"
  [ -f /mydata/suites/$SUITE.json ] || python -m vista.benchmarks.cachelab.cli build-suite --data-root /traces \
    --datasets cloudphysics,alibabaBlock,tencentBlock --per-dataset 1000 --split heldout --full-length 10e6:50e6 \
    --name-prefix gt --seed 20261125 --exclude-suite /mydata/suites/gs-dev.json --out /mydata/suites/$SUITE.json
  sha256sum /mydata/suites/$SUITE.json
  echo "== $(date +%T) content check (first 1M object ids) against gs-dev and within gs-test"
  python - <<'EOF'
import hashlib, json
from collections import Counter
from multiprocessing import Pool
import numpy as np, zstandard
REC = np.dtype([("t", "<u4"), ("id", "<u8"), ("size", "<u4"), ("next", "<i8")])
def digest(src):
    with open(src, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True); buf = bytearray()
        while len(buf) < 1_000_000 * REC.itemsize and (chunk := r.read(1_000_000 * REC.itemsize - len(buf))): buf += chunk
    return hashlib.sha256(np.frombuffer(bytes(buf[: len(buf) // REC.itemsize * REC.itemsize]), dtype=REC)["id"].tobytes()).hexdigest()
test = json.load(open("/mydata/suites/gs-test.json")); dev = json.load(open("/mydata/suites/gs-dev.json"))
with Pool(16) as p: ht = p.map(digest, [r["source"] for r in test]); hd = set(p.map(digest, [r["source"] for r in dev]))
seen, kept, dev_hit, dup = set(), [], 0, 0
for r, h in zip(test, ht):
    if h in hd: dev_hit += 1
    elif h in seen: dup += 1
    else: kept.append(r)
    seen.add(h)
json.dump(kept, open("/mydata/suites/gs-test.checked.json", "w"), indent=1)
comp = Counter(r["params"]["dataset"] for r in kept)
print(f"  {len(test)} traces: {dev_hit} match a gs-dev trace, {dup} repeat an earlier gs-test trace; {len(kept)} kept,"
      f" {sum(r['requests'] for r in kept)/1e6:.1f}M requests; " + ", ".join(f"{k} {v}" for k, v in sorted(comp.items())))
EOF
  sha256sum /mydata/suites/$SUITE.checked.json
  echo "== $(date +%T) prepare (dense ids, split, cache sizes)"
  python $S prepare /mydata/suites/$SUITE.checked.json $SUITE 32
  echo "== $(date +%T) S3-FIFO in vqsim (r1 line 0)"
  python - <<'EOF'
import json; p = json.load(open("scripts/vqsim/v1/portfolio.json"))["portfolio"][0]
open("/mydata/s3only.jsonl", "w").write(json.dumps({"config": p}) + "\n")
EOF
  python scripts/vqsim/study/vq_run3.py /mydata/vq/$SUITE /mydata/s3only.jsonl r1 64
  echo "== $(date +%T) the zoo, then 3L-Cache and S3-FIFO per window at full scale and as miniatures, in $G groups"
  for g in $(seq 0 $((G - 1))); do
    VQ_GROUP=$g/$G python $S zoo $SUITE 90
    VQ_GROUP=$g/$G python $S mini $SUITE 90
  done
  python $S finish $SUITE
  echo "== $(date +%T) drop rule"
  python $S drop $SUITE
  python - <<'EOF'
import json; from pathlib import Path; from collections import Counter
ds = sorted(p for p in Path("/mydata/vq/gs-test").iterdir() if (p / "meta.json").is_file())
m = [json.loads((d / "meta.json").read_text()) for d in ds]
print(f"  scored: {len(ds)} traces, {sum(len(x['capacities']) for x in m)} cells, {sum(x['requests'] for x in m)/1e6:.1f}M requests; "
      + ", ".join(f"{k} {v}" for k, v in sorted(Counter(d.name.split('-')[1] for d in ds).items())))
EOF
  echo "== $(date +%T) stage A done"
elif [ "$STAGE" = B ]; then
  echo "== $(date +%T) pre-registration and the candidate's files"
  check scripts/vqsim/v2/PREREG-gstest.md "$2"
  check_candidate
  cmp /mydata/vqsim3 /mydata/vqsim-fresh
  echo "== $(date +%T) v2 + replay 128 (v1's frozen 32 + 16 screened from pool.txt), then with VQSIM_TRACE"
  VQSIM=/mydata/vqsim-fresh POOL=/work/VISTA/scripts/vqsim/v2/pool.txt python scripts/vqsim/study/vq_portfolio_real.py \
    /mydata/vq/$SUITE scripts/vqsim/v1/portfolio.json gv2 64 $SPEC
  python $S trace $SUITE gv2 64 0
  echo "== $(date +%T) the bar, P1"
  python $S test $SUITE gv2
  echo "== $(date +%T) the classic-oracle bound, P2"
  python $S bound $SUITE
  echo "== $(date +%T) miniatures at $RATES, and the takeover grid, in $G groups"
  for g in $(seq 0 $((G - 1))); do
    VQ_GROUP=$g/$G python $S rates $SUITE $RATES 90
    VQ_GROUP=$g/$G python $S grid $SUITE gv2 90
  done
  echo "== $(date +%T) hindsight with a learner, and the rules (P3: rate 0.1, kappa 1, H=8, pes)"
  python $S score $SUITE gv2
  python $S rules $SUITE gv2
  echo "== $(date +%T) when 3L-Cache's edge appears, and its miniature's fidelity (P4)"
  python $S describe $SUITE gv2 $RATES
  echo "== $(date +%T) stage B done"
fi
