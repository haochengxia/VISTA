"""W-TinyLFU reference policy (Einziger, Friedman and Manes, ACM TOS 2017; Caffeine).

- Window: a small LRU (WINDOW_RATIO of the cache) that every new object enters.
- Main: segmented LRU, probation and protected (PROTECTED_RATIO of main). A hit in
  probation moves the object to protected; protected's overflow goes back to probation.
- Admission: when the window is full, its LRU object (the candidate) competes with
  main's victim (probation's LRU end). The one with the higher estimated frequency
  stays; ties go to the victim (the incumbent).
- Until main is full, the window's overflow moves to probation without a contest.
- Frequency: a count-min sketch (DEPTH rows of WIDTH_RATIO x capacity 8-bit
  counters, conservative update) counts every request; all counters halve every
  RESET_RATIO x main's size in requests, so old popularity fades. These follow
  libCacheSim's WTinyLFU (sample size 32).

Before a handover the sketch counts the requests S3-FIFO served (observe), so it
starts warm.
"""

from __future__ import annotations

from collections import OrderedDict

from policy_api import Context, Handover, Policy

_MASK64 = (1 << 64) - 1
_SEEDS = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93)


class WTinyLFU(Policy):
    WINDOW_RATIO = 0.01
    PROTECTED_RATIO = 0.8
    DEPTH = 4
    WIDTH_RATIO = 4
    RESET_RATIO = 32
    QUEUES = ("W", "P", "R")  # window, probation, protected
    GHOSTS = ()

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.window_capacity = max(1, int(capacity * self.WINDOW_RATIO))
        main = max(1, capacity - self.window_capacity)
        self.protected_capacity = max(1, int(main * self.PROTECTED_RATIO))
        self.window: OrderedDict[int, None] = OrderedDict()  # least recent first
        self.probation: OrderedDict[int, None] = OrderedDict()
        self.protected: OrderedDict[int, None] = OrderedDict()
        self.main_capacity = main
        width = 16
        while width < self.WIDTH_RATIO * capacity:
            width <<= 1
        self.width_bits = width.bit_length() - 1
        self.table = bytearray(self.DEPTH * width)
        self.width = width
        self.additions = 0
        self.reset_at = max(16, self.RESET_RATIO * main)

    # -- frequency sketch -----------------------------------------------------------

    def _slots(self, key: int) -> list[int]:
        shift = 64 - self.width_bits
        return [row * self.width + ((((key ^ seed) * 0xFF51AFD7ED558CCD) & _MASK64) >> shift) for row, seed in enumerate(_SEEDS[: self.DEPTH])]

    def _count(self, key: int) -> None:
        table = self.table
        slots = self._slots(key)
        low = min(table[slot] for slot in slots)
        if low < 255:  # conservative update: only the smallest counters grow
            for slot in slots:
                if table[slot] == low:
                    table[slot] = low + 1
        self.additions += 1
        if self.additions >= self.reset_at:
            self.table = bytearray(value >> 1 for value in table)
            self.additions //= 2

    def frequency(self, key: int) -> int:
        table = self.table
        return min(table[slot] for slot in self._slots(key))

    # -- hooks ----------------------------------------------------------------------

    def observe(self, key: int, now: int, hit: bool) -> None:
        self._count(key)

    def on_hit(self, key: int, now: int) -> None:
        self._count(key)
        if key in self.window:
            self.window.move_to_end(key)
            self.ctx.move(key, "W", "W")
        elif key in self.probation:
            del self.probation[key]
            self.protected[key] = None
            self.ctx.move(key, "P", "R")
            if len(self.protected) > self.protected_capacity:
                demoted, _ = self.protected.popitem(last=False)
                self.probation[demoted] = None
                self.ctx.move(demoted, "R", "P")
        else:
            self.protected.move_to_end(key)
            self.ctx.move(key, "R", "R")

    def on_miss(self, key: int, now: int) -> bool:
        self._count(key)
        return True

    def _main_victim(self) -> tuple[int, OrderedDict]:
        queue = self.probation if self.probation else self.protected
        return next(iter(queue)), queue

    def evict(self, now: int) -> int:
        if not (self.probation or self.protected):  # no main yet (a tiny cache): plain LRU
            key, _ = self.window.popitem(last=False)
            return key
        if len(self.window) < self.window_capacity:  # the new object fits the window: main gives way
            key, queue = self._main_victim()
            del queue[key]
            return key
        candidate = next(iter(self.window))
        victim, queue = self._main_victim()
        if self.frequency(candidate) > self.frequency(victim):
            del self.window[candidate]
            del queue[victim]
            self.probation[candidate] = None
            self.ctx.move(candidate, "W", "P")
            return victim
        del self.window[candidate]
        return candidate

    def insert(self, key: int, now: int) -> str:
        self.window[key] = None
        while len(self.window) > self.window_capacity and len(self.probation) + len(self.protected) < self.main_capacity:
            moved, _ = self.window.popitem(last=False)  # main still filling: no contest
            self.probation[moved] = None
            self.ctx.move(moved, "W", "P")
        return "W"

    def adopt(self, state: Handover) -> dict[int, str]:
        # S3-FIFO's main queue fills protected (its newest objects), the rest is probation;
        # its small queue goes to probation too, newest last. The window starts empty.
        placed = {}
        main = [key for key, _ in state.main]
        cut = max(0, len(main) - self.protected_capacity)
        for key in main[:cut]:
            self.probation[key] = None
            placed[key] = "P"
        for key, _ in state.small:
            self.probation[key] = None
            placed[key] = "P"
        for key in main[cut:]:
            self.protected[key] = None
            placed[key] = "R"
        return placed

    def sizes(self) -> dict[str, int]:
        return {"W": len(self.window), "P": len(self.probation), "R": len(self.protected)}


CachePolicy = WTinyLFU
