"""MQ, the Multi-Queue algorithm (Zhou, Philbin and Li, USENIX ATC 2001).

Written for second-level buffer caches (storage servers behind a client
cache), whose traces have little short-range reuse: the client cache already
absorbed it. The paper's analysis of such traces found that (1) reuse
distances are mostly longer than the first-level cache, so a block must stay
at least that long ("minimal lifetime"); (2) a block's access frequency
predicts reuse better than recency ("frequency-based priority"); and (3)
frequency must decay, or blocks hot long ago never leave ("temporal
frequency").

Design, following libCacheSim's MQ.c:

- N_QUEUES LRU queues Q0..Q{N-1}. An object with frequency f lives in
  Q[min(floor(log2 f), N-1)]; eviction takes the LRU end of the lowest
  non-empty queue.
- Every request (hit or insert) advances a virtual clock and runs Adjust:
  the LRU object of each queue k >= 1 whose expire time has passed moves to
  the MRU end of queue k-1, and gets a new expire time. An object not
  accessed for LIFETIME requests drops one level.
- Qout remembers evicted keys with their frequency (QOUT_RATIO x cache
  size, FIFO). A miss found in Qout restores that frequency plus one.

Qout holds 4x the cache in keys, so this reference uses well over twice
S3-FIFO's metadata; a submitted policy must stay within that bound.
"""

from __future__ import annotations

from collections import OrderedDict

from policy_api import Context, Policy


class MQ(Policy):
    N_QUEUES = 8
    LIFETIME = 10000  # requests; the paper suggests the peak reuse distance
    QOUT_RATIO = 4.0
    QUEUES = tuple(f"Q{i}" for i in range(8))
    GHOSTS = ("Qout",)

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.levels = [OrderedDict() for _ in range(self.N_QUEUES)]  # LRU first
        self.level: dict[int, int] = {}
        self.freq: dict[int, int] = {}
        self.expire: dict[int, int] = {}
        self.qout: OrderedDict[int, int] = OrderedDict()  # key -> frequency, oldest first
        self.qout_capacity = int(capacity * self.QOUT_RATIO)
        self.vtime = 0
        self.restored: int | None = None

    def queue_num(self, frequency: int) -> int:
        level = 0
        while frequency > 1 and level < self.N_QUEUES - 1:
            frequency >>= 1
            level += 1
        return level

    def _link(self, key: int, level: int) -> None:
        self.levels[level][key] = None
        self.level[key] = level
        self.expire[key] = self.vtime + self.LIFETIME

    def _adjust(self) -> None:
        self.vtime += 1
        for k in range(1, self.N_QUEUES):
            queue = self.levels[k]
            if queue:
                key = next(iter(queue))
                if self.expire[key] < self.vtime:
                    del queue[key]
                    self._link(key, k - 1)
                    self.ctx.move(key, self.QUEUES[k], self.QUEUES[k - 1])

    def on_hit(self, key: int, now: int) -> None:
        source = self.level[key]
        del self.levels[source][key]
        self.freq[key] += 1
        target = self.queue_num(self.freq[key])
        self._link(key, target)
        if target != source:
            self.ctx.move(key, self.QUEUES[source], self.QUEUES[target])
        self._adjust()

    def on_miss(self, key: int, now: int) -> bool:
        self.restored = self.qout.pop(key, None)
        if self.restored is not None:
            self.ctx.ghost_hit(key, "Qout")
        return True

    def evict(self, now: int) -> int:
        for queue in self.levels:
            if queue:
                key, _ = queue.popitem(last=False)
                break
        frequency = self.freq.pop(key)
        del self.level[key], self.expire[key]
        if self.qout_capacity > 0:
            while len(self.qout) >= self.qout_capacity:
                old, _ = self.qout.popitem(last=False)
                self.ctx.ghost_drop(old, "Qout")
            self.qout[key] = frequency
            self.ctx.ghost_add(key, "Qout")
        return key

    def insert(self, key: int, now: int) -> str:
        frequency = 1 if self.restored is None else self.restored + 1
        self.restored = None
        self.freq[key] = frequency
        level = self.queue_num(frequency)
        self._link(key, level)
        self._adjust()
        return self.QUEUES[level]

    def sizes(self) -> dict[str, int]:
        sizes = {name: len(queue) for name, queue in zip(self.QUEUES, self.levels)}
        sizes["Qout"] = len(self.qout)
        return sizes


CachePolicy = MQ
