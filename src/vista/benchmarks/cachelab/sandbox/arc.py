"""ARC reference policy (Megiddo and Modha, FAST 2003).

Two resident LRU lists, T1 (seen once recently) and T2 (seen at least twice),
and two ghost lists of evicted keys, B1 (from T1) and B2 (from T2). A miss
found in B1 means T1 was too small: the target size p of T1 grows; one found
in B2 shrinks it. Eviction takes T1's LRU end while T1 is over p, else T2's.
Ghost hits enter T2. Together the four lists hold at most twice the cache.
"""

from __future__ import annotations

from collections import OrderedDict

from policy_api import Context, Handover, Policy


class ARC(Policy):
    QUEUES = ("T1", "T2")
    GHOSTS = ("B1", "B2")

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.t1: OrderedDict[int, None] = OrderedDict()  # least recent first
        self.t2: OrderedDict[int, None] = OrderedDict()
        self.b1: OrderedDict[int, None] = OrderedDict()
        self.b2: OrderedDict[int, None] = OrderedDict()
        self.p = 0.0  # target size of T1
        self.pending = "new"  # what the current miss is: "new", "B1" or "B2"
        self.drop_t1 = False  # case IV.A with T1 full: evict T1's LRU without a ghost

    def on_hit(self, key: int, now: int) -> None:
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
            self.ctx.move(key, "T1", "T2")
        else:
            self.t2.move_to_end(key)
            self.ctx.move(key, "T2", "T2")

    def on_miss(self, key: int, now: int) -> bool:
        c = self.capacity
        if key in self.b1:
            self.p = min(float(c), self.p + max(len(self.b2) / len(self.b1), 1.0))
            del self.b1[key]
            self.ctx.ghost_hit(key, "B1")
            self.pending = "B1"
        elif key in self.b2:
            self.p = max(0.0, self.p - max(len(self.b1) / len(self.b2), 1.0))
            del self.b2[key]
            self.ctx.ghost_hit(key, "B2")
            self.pending = "B2"
        else:
            self.pending = "new"
            if len(self.t1) + len(self.b1) >= c:
                if len(self.t1) < c:
                    if self.b1:
                        old, _ = self.b1.popitem(last=False)
                        self.ctx.ghost_drop(old, "B1")
                else:
                    self.drop_t1 = True
            elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * c and self.b2:
                old, _ = self.b2.popitem(last=False)
                self.ctx.ghost_drop(old, "B2")
        return True

    def evict(self, now: int) -> int:
        if self.drop_t1:
            self.drop_t1 = False
            key, _ = self.t1.popitem(last=False)
            return key
        n1 = len(self.t1)
        if n1 and (n1 > self.p or (self.pending == "B2" and n1 == self.p) or not self.t2):
            key, _ = self.t1.popitem(last=False)
            self.b1[key] = None
            self.ctx.ghost_add(key, "B1")
        else:
            key, _ = self.t2.popitem(last=False)
            self.b2[key] = None
            self.ctx.ghost_add(key, "B2")
        return key

    def insert(self, key: int, now: int) -> str:
        if self.pending in ("B1", "B2"):
            self.pending = "new"
            self.t2[key] = None
            return "T2"
        self.t1[key] = None
        return "T1"

    def adopt(self, state: Handover) -> dict[int, str]:
        # S3-FIFO's small queue holds objects seen once (T1), its main queue proven ones (T2),
        # its ghost recent T1 evictions (B1). T1 starts at its current size.
        placed = {}
        for key, _ in state.small:
            self.t1[key] = None
            placed[key] = "T1"
        for key, _ in state.main:
            self.t2[key] = None
            placed[key] = "T2"
        room = max(0, self.capacity - len(self.t1))
        for key in state.ghost[-room:] if room else ():
            self.b1[key] = None
        self.p = float(len(self.t1))
        return placed

    def sizes(self) -> dict[str, int]:
        return {"T1": len(self.t1), "T2": len(self.t2), "B1": len(self.b1), "B2": len(self.b2)}


CachePolicy = ARC
