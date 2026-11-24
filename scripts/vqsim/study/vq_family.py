"""The S3-structured family: designs whose takeover from S3-FIFO, and between each other, maps stage to stage.
Two stages; new objects enter the small stage 0, which promotes into main (stage 1) and otherwise evicts; main reinserts
into itself and otherwise evicts; the takeover sends small to small and main to main (asmall=0, amain=1); small's
promotion threshold is reachable (no static sets). Knobs (sizes, ghost, thresholds, counters, delay, age) and the
add-on rules that keep the structure (gate, rd) are free.
  python vq_family.py < POOL > POOL-s3       keeps the family's lines of a pool file (one config per line)"""
import sys


def s3like(config: str) -> bool:
    d: dict = {"st": []}
    for tok in config.split():
        k, _, v = tok.partition("=")
        if k == "st":
            d["st"].append([float(x) for x in v.split(",")])
        else:
            d[k] = float(v)
    st = d["st"]
    if len(st) != 2:
        return False
    entry, maxf = int(d.get("entry", 0)), int(d.get("maxf", 3))
    if entry != 0 or int(d.get("asmall", entry)) != 0 or int(d.get("amain", 1)) != 1:
        return False
    s0, s1 = st
    if int(s0[3]) != 1 or int(s0[5]) != -1 or int(s1[3]) != 1 or int(s1[5]) != -1:
        return False
    return s0[2] <= maxf if int(s0[1]) else s0[2] == 0


if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() and s3like(line.strip()):
            sys.stdout.write(line)
