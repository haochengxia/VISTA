"""The contract between a cache policy and the CacheLab simulator (stdlib only).

The simulator owns residency. It decides hits, enforces the capacity and
records every outcome, so a policy cannot misreport its own miss ratio. A
policy only decides:

- what to remember about a missed object and whether to admit it (`on_miss`),
- which resident object leaves when the cache is full (`evict`),
- which of its resident queues a newly admitted object enters (`insert`).

Capacity counts objects: every object occupies one slot.

Hook order for one request of `key` at virtual time `now`:

    resident     -> on_hit(key, now)
    not resident -> admitted = on_miss(key, now)
                    if admitted:
                        while the cache is full: victim = evict(now)
                        queue = insert(key, now)

Report internal transitions through `ctx` so the views can replay them:

    ctx.move(key, src, dst)     a resident object changed queue (src == dst
                                for a reinsertion at the head of the same queue)
    ctx.ghost_add(key, ghost)   a non-resident key entered a history queue
    ctx.ghost_drop(key, ghost)  a history entry was discarded without a hit
    ctx.ghost_hit(key, ghost)   a missed key was found in a history queue

Queue names must come from `QUEUES` (resident) and `GHOSTS` (history). The
simulator checks every reported move against the queue it tracked for that
object and reports disagreements as instrumentation errors.

Handover. A run may start with S3-FIFO serving the cache and switch to this
policy at request `now` (the deployment case: a live cache changes policy).
Before the switch the simulator calls `observe(key, now, hit)` for every
request S3-FIFO served (only if the policy overrides it; it must not report
events). At the switch it calls `adopt(state)` once, with S3-FIFO's live
state as a `Handover`; the policy must take over every resident object and
return the queue each one is in now. From then on the usual hooks apply.
"""

from __future__ import annotations


class Context:
    """Reporting hooks; the simulator supplies the implementation."""

    def move(self, key: int, src: str, dst: str) -> None:
        pass

    def ghost_add(self, key: int, ghost: str = "G") -> None:
        pass

    def ghost_drop(self, key: int, ghost: str = "G") -> None:
        pass

    def ghost_hit(self, key: int, ghost: str = "G") -> None:
        pass


class Handover:
    """The live state of the incumbent S3-FIFO when a new policy takes over.

    small, main  resident objects as (key, freq) pairs, from the next eviction
                 candidate (tail) to the newest (head); freq is S3-FIFO's
                 2-bit counter (0..3)
    ghost        keys recently evicted from small, oldest first (not resident)
    config       the incumbent's knobs (SMALL_RATIO, GHOST_RATIO, ...)
    warm         True if the incumbent had not evicted yet (still filling)
    now          the first request the new policy serves
    """

    __slots__ = ("capacity", "now", "small", "main", "ghost", "config", "warm")

    def __init__(self, capacity, now, small, main, ghost, config, warm) -> None:
        self.capacity = capacity
        self.now = now
        self.small = small
        self.main = main
        self.ghost = ghost
        self.config = config
        self.warm = warm

    def resident(self) -> list[tuple[int, int]]:
        """All resident objects: small (tail to head), then main (tail to head)."""
        return self.small + self.main

    def to_json(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_json(cls, data: dict) -> "Handover":
        return cls(
            int(data["capacity"]),
            int(data["now"]),
            [(int(key), int(freq)) for key, freq in data["small"]],
            [(int(key), int(freq)) for key, freq in data["main"]],
            [int(key) for key in data["ghost"]],
            dict(data["config"]),
            bool(data["warm"]),
        )


class Policy:
    """Base class for an eviction policy driven by the CacheLab simulator."""

    QUEUES: tuple[str, ...] = ("Q",)
    GHOSTS: tuple[str, ...] = ()

    def __init__(self, capacity: int, ctx: Context) -> None:
        self.capacity = capacity
        self.ctx = ctx

    def on_hit(self, key: int, now: int) -> None:
        raise NotImplementedError

    def on_miss(self, key: int, now: int) -> bool:
        """Called before any eviction for this request; return False to bypass."""
        return True

    def evict(self, now: int) -> int:
        """Remove one resident object from the policy's state and return its key."""
        raise NotImplementedError

    def insert(self, key: int, now: int) -> str:
        """Place an admitted object; return the name of the queue it entered."""
        raise NotImplementedError

    def sizes(self) -> dict[str, int]:
        """Current occupancy of every queue in QUEUES and GHOSTS."""
        raise NotImplementedError

    def observe(self, key: int, now: int, hit: bool) -> None:
        """Before a handover: one request the incumbent served. Default: ignore."""

    def adopt(self, state: Handover) -> dict[int, str]:
        """At a handover: take over the incumbent's resident objects and return
        {key: queue} for every one of them. Default: insert() them one by one,
        small then main, each from tail to head, so the incumbent's newest
        main objects end up newest here; frequencies and the ghost are dropped."""
        return {key: self.insert(key, state.now) for key, _ in state.resident()}
