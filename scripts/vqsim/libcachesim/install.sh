#!/usr/bin/env bash
# Install CacheLab's OnlineV1 into a libCacheSim source checkout (github.com/1a1a11a/libCacheSim):
# copies the engine into libCacheSim/cache/eviction/OnlineV1/ and registers it as "onlinev1".
# Idempotent. Then build libCacheSim as usual and run, e.g.:
#   cachesim <trace> oracleGeneral onlinev1,s3fifo 0.01,0.1 --ignore-obj-size 1
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lcs="${1:?usage: install.sh <libCacheSim checkout>}"
src="$lcs/libCacheSim"
test -f "$src/cache/cacheAlgoRegistry.c" || { echo "not a libCacheSim checkout: $lcs" >&2; exit 1; }
mkdir -p "$src/cache/eviction/OnlineV1"
cp "$here/OnlineV1.c" "$here/../vqcore.h" "$here/../vqonline.h" "$here/../vqonline_v1.h" "$src/cache/eviction/OnlineV1/"
python3 - "$src" <<'PY'
import re, sys
from pathlib import Path
src = Path(sys.argv[1])
def patch(path, anchor, text, after=True):
    p = src / path
    s = p.read_text()
    if text.strip() in s:
        return
    assert s.count(anchor) == 1, (path, anchor)
    p.write_text(s.replace(anchor, anchor + text if after else text + anchor))
patch("cache/CMakeLists.txt", "        eviction/S3FIFO.c\n", "        eviction/OnlineV1/OnlineV1.c\n")
patch("include/libCacheSim/evictionAlgo.h", "cache_t *S3FIFO_init(",
      "cache_t *OnlineV1_init(const common_cache_params_t ccache_params,\n                       const char *cache_specific_params);\n\n", after=False)
s = (src / "cache/cacheAlgoRegistry.c").read_text()
if '"onlinev1"' not in s:
    entries = re.search(r"g_cache_algos\[\] = \{\n(.*?)\n\};", s, re.S)
    assert entries
    lines = entries.group(1).split("\n")
    at = next(i for i, l in enumerate(lines) if re.match(r'\s*\{"(\w[\w-]*)"', l) and re.match(r'\s*\{"(\w[\w-]*)"', l).group(1).lower() > "onlinev1")
    while at > 0 and lines[at - 1].strip().startswith("/*"):  # keep a comment with its entry
        at -= 1
    lines.insert(at, '    {"onlinev1", OnlineV1_init},')
    s = s[: entries.start(1)] + "\n".join(lines) + s[entries.end(1):]
    (src / "cache/cacheAlgoRegistry.c").write_text(s)
print("installed OnlineV1 into", src)
PY
