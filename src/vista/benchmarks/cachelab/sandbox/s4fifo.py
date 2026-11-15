"""S4-FIFO's heuristic: S3-FIFO with two more knobs (static version).

From "Learning-Augmented Heuristics: Simple, yet Smart, Robust and
Interpretable Cache Eviction" (OSDI 2026), following libCacheSim's S4FIFO.c
with auto-tune off. With the default knobs it is exactly S3-FIFO.

- SMALL_SKIP_RATIO: a re-request to an object inserted into S within the
  most recent SMALL_SKIP_RATIO x len(S) insertions does not count toward
  promotion. This creates a probation region at the head of S that filters
  quick re-requests (for example a scan touching a block twice) that look
  like reuse but are not.
- GHOST_TO_MAIN_THRESHOLD: with a positive value a ghost hit no longer goes
  straight to M; the key stays in G and the object re-enters S like any
  other miss, so it must prove itself again.

The paper's analysis: the right S size, ghost size, promotion threshold and
these two knobs differ across workloads, and cache-level statistics observed
early in a trace reveal which setting fits: where in S, M and G hits land
(hit-position histograms), the one-hit-wonder share, and how many requests
are unique. Its control plane learns that mapping offline and applies a
configuration after observing the first 20% of a trace. That model is not
included here; the knobs and the statistics that drive them are.
"""

from __future__ import annotations

from policy_api import Context, Handover
from s3fifo import S3FIFO


class S4FIFO(S3FIFO):
    SMALL_SKIP_RATIO = 0.0
    GHOST_TO_MAIN_THRESHOLD = 0

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.small_inserts = 0  # insertions into S so far
        self.small_seq: dict[int, int] = {}  # key -> S insertion number

    def adopt(self, state: Handover) -> dict[int, str]:
        placement = super().adopt(state)
        # Inherited S objects keep their order: number them from tail to head.
        self.small_seq = {key: number for number, (key, _) in enumerate(state.small)}
        self.small_inserts = len(state.small)
        return placement

    def on_hit(self, key: int, now: int) -> None:
        seq = self.small_seq.get(key)
        if seq is not None:
            probation = int(self.SMALL_SKIP_RATIO * len(self.small))
            if self.small_inserts - seq < probation:
                return  # a quick re-request inside the probation region
        super().on_hit(key, now)

    def on_miss(self, key: int, now: int) -> bool:
        if self.GHOST_TO_MAIN_THRESHOLD <= 0:
            return super().on_miss(key, now)
        self.ghost_hit_pending = False
        if key in self.ghost:
            self.ctx.ghost_hit(key, "G")  # seen, but stays in G and re-enters S
        return True

    def insert(self, key: int, now: int) -> str:
        queue = super().insert(key, now)
        if queue == "S":
            self.small_seq[key] = self.small_inserts
            self.small_inserts += 1
        return queue

    def _evict_small(self) -> int | None:
        while self.small:
            key = self.small[-1]
            self.small_seq.pop(key, None)
            if self.freq[key] >= self.MOVE_THRESHOLD:
                self.small.pop()
                self.freq[key] = 0
                self.main.appendleft(key)
                self.ctx.move(key, "S", "M")
                continue
            self.small.pop()
            del self.freq[key]
            self._remember(key)
            return key
        return None


CachePolicy = S4FIFO
