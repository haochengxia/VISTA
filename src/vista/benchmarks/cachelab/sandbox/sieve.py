"""SIEVE reference policy (Zhang et al., NSDI 2024).

One FIFO queue, a visited bit per object and a hand. New objects enter at the
head; a hit sets the bit. The hand moves from the tail toward the head: an
object with the bit set loses it and stays in place, the first one without it
is evicted, and the hand stays where it stopped (wrapping to the tail).
"""

from __future__ import annotations

from policy_api import Context, Handover, Policy


class SIEVE(Policy):
    QUEUES = ("Q",)
    GHOSTS = ()

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        self.visited: set[int] = set()  # keys whose bit is set
        self.size = 0
        self.newer: dict[int, int | None] = {}  # toward the head
        self.older: dict[int, int | None] = {}  # toward the tail
        self.head: int | None = None
        self.tail: int | None = None
        self.hand: int | None = None

    def on_hit(self, key: int, now: int) -> None:
        self.visited.add(key)

    def _unlink(self, key: int) -> None:
        newer, older = self.newer.pop(key), self.older.pop(key)
        if newer is None:
            self.head = older
        else:
            self.older[newer] = older
        if older is None:
            self.tail = newer
        else:
            self.newer[older] = newer
        self.visited.discard(key)
        self.size -= 1

    def evict(self, now: int) -> int:
        key = self.hand if self.hand is not None else self.tail
        while key in self.visited:
            self.visited.discard(key)
            key = self.newer[key]
            if key is None:
                key = self.tail
        self.hand = self.newer[key]
        self._unlink(key)
        return key

    def insert(self, key: int, now: int) -> str:
        self.newer[key], self.older[key] = None, self.head
        if self.head is None:
            self.tail = key
        else:
            self.newer[self.head] = key
        self.head = key
        self.size += 1
        return "Q"

    def adopt(self, state: Handover) -> dict[int, str]:
        for key, freq in state.main + state.small:
            self.insert(key, state.now)
            if freq > 0:
                self.visited.add(key)
        return dict.fromkeys(self.newer, "Q")

    def sizes(self) -> dict[str, int]:
        return {"Q": self.size}


CachePolicy = SIEVE
