"""2Q reference policy (Johnson and Shasha, VLDB 1994), full version, as libCacheSim's TwoQ.

- A1in: FIFO for objects seen once (IN_RATIO of the cache). A hit there does nothing.
- A1out: ghost FIFO of keys evicted from A1in (OUT_RATIO of the cache, keys only).
- Am: LRU for objects requested again after leaving A1in. Only a miss found in
  A1out enters Am: an object must come back after its first stay to count as hot.

Eviction takes A1in's tail while A1in is over its share, else Am's LRU end.
"""

from __future__ import annotations

from collections import OrderedDict

from policy_api import Context, Handover, Policy


class TwoQ(Policy):
    IN_RATIO = 0.25
    OUT_RATIO = 0.5
    QUEUES = ("A1in", "Am")
    GHOSTS = ("A1out",)

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.a1in: OrderedDict[int, None] = OrderedDict()  # tail (oldest) first
        self.am: OrderedDict[int, None] = OrderedDict()  # least recent first
        self.a1out: OrderedDict[int, None] = OrderedDict()  # oldest first
        self.in_capacity = max(1, int(capacity * self.IN_RATIO))
        self.out_capacity = int(capacity * self.OUT_RATIO)
        self.returning = False

    def on_hit(self, key: int, now: int) -> None:
        if key in self.am:
            self.am.move_to_end(key)
            self.ctx.move(key, "Am", "Am")

    def on_miss(self, key: int, now: int) -> bool:
        self.returning = key in self.a1out
        if self.returning:
            del self.a1out[key]
            self.ctx.ghost_hit(key, "A1out")
        return True

    def evict(self, now: int) -> int:
        if self.a1in and (len(self.a1in) >= self.in_capacity or not self.am):
            key, _ = self.a1in.popitem(last=False)
            if self.out_capacity > 0:
                while len(self.a1out) >= self.out_capacity:
                    old, _ = self.a1out.popitem(last=False)
                    self.ctx.ghost_drop(old, "A1out")
                self.a1out[key] = None
                self.ctx.ghost_add(key, "A1out")
            return key
        key, _ = self.am.popitem(last=False)
        return key

    def insert(self, key: int, now: int) -> str:
        if self.returning:
            self.returning = False
            self.am[key] = None
            return "Am"
        self.a1in[key] = None
        return "A1in"

    def adopt(self, state: Handover) -> dict[int, str]:
        # S3-FIFO's small queue is 2Q's A1in, its main queue Am, its ghost A1out.
        placed = {}
        for key, _ in state.small:
            self.a1in[key] = None
            placed[key] = "A1in"
        for key, _ in state.main:
            self.am[key] = None
            placed[key] = "Am"
        for key in state.ghost[-self.out_capacity :] if self.out_capacity > 0 else ():
            self.a1out[key] = None
        return placed

    def sizes(self) -> dict[str, int]:
        return {"A1in": len(self.a1in), "Am": len(self.am), "A1out": len(self.a1out)}


CachePolicy = TwoQ
