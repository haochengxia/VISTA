"""FIFO reference policy (one resident queue)."""

from collections import deque

from policy_api import Policy


class FIFO(Policy):
    QUEUES = ("Q",)
    GHOSTS = ()

    def __init__(self, capacity, ctx):
        super().__init__(capacity, ctx)
        self.queue = deque()

    def on_hit(self, key, now):
        pass

    def evict(self, now):
        return self.queue.popleft()

    def insert(self, key, now):
        self.queue.append(key)
        return "Q"

    def sizes(self):
        return {"Q": len(self.queue)}


CachePolicy = FIFO
