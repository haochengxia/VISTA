"""S3-FIFO (Yang et al., SOSP 2023), following libCacheSim's S3FIFO defaults.

Three FIFO queues:

- S, a small probationary queue holding about 10% of the cache. New objects
  enter here unless they were recently evicted from S.
- M, the main queue. Its tail behaves like CLOCK with a 2-bit counter: an
  object with frequency > 0 is reinserted at the head with frequency - 1.
- G, a ghost queue of keys recently evicted from S (no data), about 90% of the
  cache. A miss that hits G enters M directly.

When the cache is full, S is evicted while it holds at least its target size;
otherwise M is evicted. An object at the tail of S that was hit at least
MOVE_THRESHOLD times moves to M instead of leaving, and its frequency resets.

Warm-up (as in libCacheSim): until the first eviction, once S reaches its
target size new objects are inserted into M, so a cold cache fills M first.
"""

from __future__ import annotations

from collections import OrderedDict, deque

from policy_api import Context, Handover, Policy


class S3FIFO(Policy):
    QUEUES = ("S", "M")
    GHOSTS = ("G",)

    SMALL_RATIO = 0.10
    GHOST_RATIO = 0.90
    MOVE_THRESHOLD = 2
    MAX_FREQ = 3

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.small_target = int(capacity * self.SMALL_RATIO)
        self.ghost_capacity = int(capacity * self.GHOST_RATIO)
        self.small: deque[int] = deque()  # head on the left, tail on the right
        self.main: deque[int] = deque()
        self.ghost: OrderedDict[int, None] = OrderedDict()  # oldest first
        self.freq: dict[int, int] = {}
        self.ghost_hit_pending = False
        self.has_evicted = False

    def on_hit(self, key: int, now: int) -> None:
        if self.freq[key] < self.MAX_FREQ:
            self.freq[key] += 1

    def on_miss(self, key: int, now: int) -> bool:
        # Look up history before evictions can push this key out of G.
        self.ghost_hit_pending = key in self.ghost
        if self.ghost_hit_pending:
            del self.ghost[key]
            self.ctx.ghost_hit(key, "G")
        return True

    def insert(self, key: int, now: int) -> str:
        self.freq[key] = 0
        if self.ghost_hit_pending:
            self.ghost_hit_pending = False
            self.main.appendleft(key)
            return "M"
        if not self.has_evicted and len(self.small) >= self.small_target:
            self.main.appendleft(key)
            return "M"
        self.small.appendleft(key)
        return "S"

    def evict(self, now: int) -> int:
        self.has_evicted = True
        if len(self.small) >= self.small_target and self.small:
            victim = self._evict_small()
            if victim is not None:
                return victim
        return self._evict_main()

    def _evict_small(self) -> int | None:
        while self.small:
            key = self.small.pop()
            if self.freq[key] >= self.MOVE_THRESHOLD:
                self.freq[key] = 0
                self.main.appendleft(key)
                self.ctx.move(key, "S", "M")
                continue
            del self.freq[key]
            self._remember(key)
            return key
        return None

    def _evict_main(self) -> int:
        while True:
            key = self.main.pop()
            frequency = self.freq[key]
            if frequency > 0:
                self.freq[key] = frequency - 1
                self.main.appendleft(key)
                self.ctx.move(key, "M", "M")
                continue
            del self.freq[key]
            return key

    def _remember(self, key: int) -> None:
        if self.ghost_capacity <= 0:
            return
        if key in self.ghost:
            return
        while len(self.ghost) >= self.ghost_capacity:
            old, _ = self.ghost.popitem(last=False)
            self.ctx.ghost_drop(old, "G")
        self.ghost[key] = None
        self.ctx.ghost_add(key, "G")

    def sizes(self) -> dict[str, int]:
        return {"S": len(self.small), "M": len(self.main), "G": len(self.ghost)}

    # -- handover: a live cache switching policy ------------------------------------

    def export(self, now: int) -> Handover:
        """This cache's live state, for a policy taking it over at request `now`."""
        knobs = ("SMALL_RATIO", "GHOST_RATIO", "MOVE_THRESHOLD", "MAX_FREQ")
        return Handover(
            self.capacity,
            now,
            [(key, self.freq[key]) for key in reversed(self.small)],
            [(key, self.freq[key]) for key in reversed(self.main)],
            list(self.ghost),
            {name: getattr(self, name) for name in knobs},
            not self.has_evicted,
        )

    def adopt(self, state: Handover) -> dict[int, str]:
        """Keep the queues as they are; this class's knobs act from now on, so
        queue sizes move to the new targets only as evictions happen (switching
        S3-FIFO's knobs on a live cache). Same knobs: identical to no switch."""
        self.small = deque(key for key, _ in reversed(state.small))
        self.main = deque(key for key, _ in reversed(state.main))
        self.freq = {key: min(freq, self.MAX_FREQ) for key, freq in state.resident()}
        ghost = state.ghost
        for key in ghost[: max(0, len(ghost) - self.ghost_capacity)]:
            self.ctx.ghost_drop(key, "G")
        kept = ghost[len(ghost) - self.ghost_capacity :] if self.ghost_capacity > 0 else []
        self.ghost = OrderedDict((key, None) for key in kept)
        self.has_evicted = not state.warm
        placement = {key: "S" for key in self.small}
        placement.update((key, "M") for key in self.main)
        return placement


CachePolicy = S3FIFO
