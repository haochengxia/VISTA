"""LIRS reference policy (Jiang and Zhang, SIGMETRICS 2002).

Objects are ranked by reuse distance (IRR, the number of distinct objects
between their last two requests) instead of recency:

- LIR objects (low IRR) hold most of the cache (all but HIR_RATIO of it) and
  are never evicted directly.
- Resident HIR objects (high IRR or new) hold the rest, in queue Q; eviction
  takes Q's front.
- Stack S keeps recency order for LIR objects and recently seen HIR objects,
  resident or not. An HIR object requested while still in S has a reuse
  distance smaller than the oldest LIR object's: it becomes LIR, and that
  oldest LIR object becomes HIR. S's bottom is always LIR ("pruning").

Non-resident HIR entries in S are history (ghost "NR"); at most NR_RATIO of
the cache of them are kept, the oldest dropped first, so memory stays bounded.
They are tracked with a deque of keys and lazy deletion (leaner than a second
ordered map, to stay within the 2x memory limit).
"""

from __future__ import annotations

from collections import OrderedDict, deque

from policy_api import Context, Handover, Policy

LIR, HIR, NR = 0, 1, 2  # status of an entry in S


class LIRS(Policy):
    HIR_RATIO = 0.01
    NR_RATIO = 1.0
    QUEUES = ("LIR", "HIR")
    GHOSTS = ("NR",)

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.hir_capacity = max(1, int(capacity * self.HIR_RATIO))
        self.lir_capacity = max(1, capacity - self.hir_capacity)
        self.stack: OrderedDict[int, int] = OrderedDict()  # key -> status, bottom (oldest) first
        self.queue: OrderedDict[int, None] = OrderedDict()  # resident HIR, front (next victim) first
        self.lir_count = 0
        self.nr_order: deque[int] = deque()  # keys as they became NR, oldest first; may hold stale ones
        self.nr_count = 0
        self.nr_capacity = max(1, int(capacity * self.NR_RATIO))

    # -- stack upkeep ---------------------------------------------------------------

    def _prune(self) -> None:
        stack = self.stack
        while stack:
            key = next(iter(stack))
            status = stack[key]
            if status == LIR:
                return
            del stack[key]
            if status == NR:
                self.nr_count -= 1
                self.ctx.ghost_drop(key, "NR")

    def _demote_bottom_lir(self) -> None:
        key, status = self.stack.popitem(last=False)  # the bottom is LIR after pruning
        assert status == LIR
        self.lir_count -= 1
        self.queue[key] = None
        self.ctx.move(key, "LIR", "HIR")
        self._prune()

    def _push(self, key: int, status: int) -> None:
        self.stack.pop(key, None)
        self.stack[key] = status

    # -- hooks ----------------------------------------------------------------------

    def on_hit(self, key: int, now: int) -> None:
        if self.stack.get(key) == LIR:
            bottom = next(iter(self.stack)) == key
            self._push(key, LIR)
            if bottom:
                self._prune()
            return
        # resident HIR
        if key in self.stack:  # reuse distance below the oldest LIR's: becomes LIR
            del self.queue[key]
            self._push(key, LIR)
            self.lir_count += 1
            self.ctx.move(key, "HIR", "LIR")
            self._demote_bottom_lir()
        else:
            self._push(key, HIR)
            self.queue.move_to_end(key)
            self.ctx.move(key, "HIR", "HIR")

    def on_miss(self, key: int, now: int) -> bool:
        if self.stack.get(key) == NR:
            self.ctx.ghost_hit(key, "NR")
        return True

    def evict(self, now: int) -> int:
        if not self.queue:  # only LIR objects resident: the oldest one becomes HIR first
            self._demote_bottom_lir()
        key, _ = self.queue.popitem(last=False)
        if key in self.stack:  # stays in S as history
            self.stack[key] = NR
            self.nr_order.append(key)
            self.nr_count += 1
            self.ctx.ghost_add(key, "NR")
            while self.nr_count > self.nr_capacity:
                old = self.nr_order.popleft()
                if self.stack.get(old) == NR:  # else stale: no longer NR (pruned or requested)
                    del self.stack[old]
                    self.nr_count -= 1
                    self.ctx.ghost_drop(old, "NR")
            if len(self.nr_order) > 2 * self.nr_count + 64:  # drop stale entries now and then
                self.nr_order = deque(k for k in self.nr_order if self.stack.get(k) == NR)
        return key

    def insert(self, key: int, now: int) -> str:
        was_nr = self.stack.get(key) == NR
        if was_nr:
            self.nr_count -= 1  # its nr_order entry goes stale
        if self.lir_count < self.lir_capacity or was_nr:  # warm-up, or back while still in S
            self._push(key, LIR)
            self.lir_count += 1
            if not was_nr or self.lir_count <= self.lir_capacity:
                return "LIR"
            self._demote_bottom_lir()
            return "LIR"
        self._push(key, HIR)
        self.queue[key] = None
        return "HIR"

    def adopt(self, state: Handover) -> dict[int, str]:
        # S3-FIFO's main queue (proven objects) seeds the LIR set, oldest at the bottom of S;
        # its small queue is resident HIR. Anything beyond the LIR share starts as HIR.
        placed = {}
        main = [key for key, _ in state.main]
        keep = main[-self.lir_capacity :]
        for key in keep:  # S from the bottom: LIR objects, oldest first, then the HIR ones
            self.stack[key] = LIR
            self.lir_count += 1
            placed[key] = "LIR"
        for key in main[: len(main) - len(keep)] + [key for key, _ in state.small]:
            self.stack[key] = HIR
            self.queue[key] = None
            placed[key] = "HIR"
        return placed

    def sizes(self) -> dict[str, int]:
        return {"LIR": self.lir_count, "HIR": len(self.queue), "NR": self.nr_count}


CachePolicy = LIRS
