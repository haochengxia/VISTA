"""CLOCK reference policy (Corbato 1968), as libCacheSim's Clock with a 1-bit counter.

One FIFO queue with a reference bit per object. A hit sets the bit. At eviction
the tail object with its bit set gets a second chance: the bit is cleared and
the object moves to the head. The first tail object without the bit leaves.
"""

from __future__ import annotations

from collections import OrderedDict

from policy_api import Context, Handover, Policy


class CLOCK(Policy):
    MAX_FREQ = 1  # counter bits: 1 -> a reference bit
    QUEUES = ("Q",)
    GHOSTS = ()

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.queue: OrderedDict[int, int] = OrderedDict()  # key -> counter, tail (oldest) first

    def on_hit(self, key: int, now: int) -> None:
        self.queue[key] = min(self.queue[key] + 1, self.MAX_FREQ)

    def evict(self, now: int) -> int:
        while True:
            key, freq = self.queue.popitem(last=False)
            if freq <= 0:
                return key
            self.queue[key] = freq - 1
            self.ctx.move(key, "Q", "Q")

    def insert(self, key: int, now: int) -> str:
        self.queue[key] = 0
        return "Q"

    def adopt(self, state: Handover) -> dict[int, str]:
        # S3-FIFO's small queue is newer than its main queue only by admission; keep main's
        # order, then small, and carry over whether each object was hit.
        for key, freq in state.main + state.small:
            self.queue[key] = min(freq, self.MAX_FREQ)
        return dict.fromkeys(self.queue, "Q")

    def sizes(self) -> dict[str, int]:
        return {"Q": len(self.queue)}


CachePolicy = CLOCK
