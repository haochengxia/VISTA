"""LRU reference policy (one resident queue)."""

from collections import OrderedDict

from policy_api import Policy


class LRU(Policy):
    QUEUES = ("Q",)
    GHOSTS = ()

    def __init__(self, capacity, ctx):
        super().__init__(capacity, ctx)
        self.order = OrderedDict()  # least recent first

    def on_hit(self, key, now):
        self.order.move_to_end(key)

    def evict(self, now):
        key, _ = self.order.popitem(last=False)
        return key

    def insert(self, key, now):
        self.order[key] = None
        return "Q"

    def sizes(self):
        return {"Q": len(self.order)}


CachePolicy = LRU
