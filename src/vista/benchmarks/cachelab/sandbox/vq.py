"""Virtual-queue FIFO policy: a family of lazy-promotion designs built from FIFO stages.

An object lives in one of K FIFO stages (virtual queues). Hits never move objects;
they only raise a small saturating counter, and only in stages that count. When the
cache is full, the first stage (in STAGES order) at or over its target size is
examined from its tail:

- an object whose counter reaches the stage's `promote` threshold goes `up` to a
  stage (itself = reinsertion), its counter reset, kept, decremented or halved;
- otherwise it goes `down` to another stage, or leaves the cache (into the ghost,
  a FIFO of keys, if the stage feeds it).

The scan stays on that stage until an object leaves; after a demotion, or when the
stage runs empty, the stage is picked again. New objects enter ENTRY, ghost hits
GHOST_ENTRY (with counter GHOST_FREQ). Until the first eviction, an object whose
entry stage is full goes to OVERFLOW instead, so a cold cache fills.

Per stage, two options from the VLDB'26 lazy-promotion paper (time is the insertion
clock: an object crosses a FIFO cache in about C insertions):

- delay (Delayed FIFO-Reinsertion): a hit counts only if more than delay*C
  insertions passed since the last counted one, so clustered hits count once;
- age (Age-Guided Eviction): at the tail, an object idle (no hit) for at least
  age*C insertions is not promoted, whatever its counter.

S3-FIFO is two stages (the defaults below). S4-FIFO's skip region is a stage in
front of S whose hits do not count and that passes every object on (promote 0).
Hot/warm main queues, 2Q-like layouts and others are further combinations.

Write a design by subclassing and setting STAGES (and the other knobs); QUEUES is
derived from the stage names. `design(config)` builds a class from a vqsim config
string (scripts/vqsim), whose miss counts this class reproduces exactly.
"""

from __future__ import annotations

from collections import OrderedDict, deque

from policy_api import Context, Handover, Policy

ONUP = {"reset": 0, "keep": 1, "dec": 2, "half": 3}


class Stage:
    """One FIFO stage. size: fraction of the cache (ignored for the last stage, which
    takes the rest); up/down: stage names (down None = leave the cache)."""

    __slots__ = ("name", "size", "counts", "promote", "up", "onup", "down", "ghost", "delay", "age")

    def __init__(self, name, size=0.0, *, counts=True, promote=1, up=None, onup="reset", down=None, ghost=False, delay=0.0, age=0.0):
        if onup not in ONUP:
            raise ValueError(f"onup must be one of {sorted(ONUP)}")
        self.name, self.size, self.counts, self.promote = name, size, counts, promote
        self.up, self.onup, self.down, self.ghost = name if up is None else up, onup, down, ghost
        self.delay, self.age = delay, age


class VQ(Policy):
    STAGES = (
        Stage("S", 0.10, promote=2, up="M", onup="reset", ghost=True),
        Stage("M", promote=1, up="M", onup="dec"),
    )
    QUEUES = ("S", "M")
    GHOSTS = ("G",)
    ENTRY = "S"
    OVERFLOW = None  # default: the last stage
    GHOST_ENTRY = None  # default: the last stage
    GHOST_FREQ = 0
    GHOST_RATIO = 0.90
    MAX_FREQ = 3  # at most 15
    ADOPT_SMALL = None  # at a handover, S3-FIFO's S objects go here (default ENTRY) ...
    ADOPT_MAIN = None  # ... and its M objects here (default: the last stage)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.QUEUES = tuple(stage.name for stage in cls.STAGES)

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        stages = self.STAGES
        names = [stage.name for stage in stages]
        if not stages or len(set(names)) != len(names) or not 1 <= self.MAX_FREQ <= 15:
            raise ValueError("STAGES need distinct names; MAX_FREQ must be 1..15")
        index = {name: i for i, name in enumerate(names)}
        last = len(stages) - 1

        def at(name, default):
            return default if name is None else index[name]

        self.k = len(stages)
        self.names = names
        self.counts = [bool(stage.counts) for stage in stages]
        self.promote = [int(stage.promote) for stage in stages]
        self.up = [index[stage.up] for stage in stages]
        self.onup = [ONUP[stage.onup] for stage in stages]
        self.down = [at(stage.down, -1) for stage in stages]
        self.feeds = [bool(stage.ghost) for stage in stages]
        self.delay = [int(stage.delay * capacity) for stage in stages]
        self.age = [int(stage.age * capacity) for stage in stages]
        for i in range(self.k):
            if self.up[i] == i and self.onup[i] == 1 and self.promote[i] > 0:
                raise ValueError(f"stage {names[i]}: a reinsertion must lower the counter (onup != keep)")
        edges = [set() for _ in stages]  # moves that need no hit and keep the counter
        for i in range(self.k):
            if self.down[i] >= 0:
                edges[i].add(self.down[i])
            if self.up[i] != i and (self.promote[i] == 0 or self.onup[i] == 1):
                edges[i].add(self.up[i])
        for m in range(self.k):  # transitive closure: a cycle would let objects circle without leaving
            for i in range(self.k):
                if m in edges[i]:
                    edges[i] |= edges[m]
        if any(i in edges[i] for i in range(self.k)):
            raise ValueError("objects could circle between stages without a hit; such paths must end in a stage whose objects leave")
        self.entry = index[self.ENTRY]
        self.overflow = at(self.OVERFLOW, last)
        self.ghost_entry = at(self.GHOST_ENTRY, last)
        self.adopt_small = at(self.ADOPT_SMALL, self.entry)
        self.adopt_main = at(self.ADOPT_MAIN, last)
        targets, used = [], 0
        for stage in stages[:-1]:
            targets.append(int(capacity * stage.size))
            used += targets[-1]
        targets.append(max(0, capacity - used))
        self.targets = targets
        self.ghost_capacity = int(capacity * self.GHOST_RATIO)
        self.queues = [deque() for _ in stages]  # head on the left, tail on the right
        self.state: dict[int, int] = {}  # key -> stage * 16 + counter (small ints: no allocation)
        self.timed = any(self.delay) or any(self.age)
        self.stamps: dict[int, int] = {}  # if timed: key -> last counted hit << 32 | last hit (insertion clock)
        self.ghost: OrderedDict[int, None] = OrderedDict()  # oldest first
        self.clock = 0
        self.ghost_hit_pending = False
        self.has_evicted = False

    # -- hooks ----------------------------------------------------------------------

    def on_hit(self, key: int, now: int) -> None:
        value = self.state[key]
        s, freq = value >> 4, value & 15
        if not self.timed:
            if self.counts[s] and freq < self.MAX_FREQ:
                self.state[key] = value + 1
            return
        stamp = self.stamps[key]
        counted, clock = stamp >> 32, self.clock
        if self.counts[s] and freq < self.MAX_FREQ and (not self.delay[s] or clock - counted > self.delay[s]):
            self.state[key] = value + 1
            counted = clock
        self.stamps[key] = counted << 32 | clock

    def on_miss(self, key: int, now: int) -> bool:
        self.ghost_hit_pending = key in self.ghost
        if self.ghost_hit_pending:
            del self.ghost[key]
            self.ctx.ghost_hit(key, "G")
        return True

    def insert(self, key: int, now: int) -> str:
        if self.ghost_hit_pending:
            self.ghost_hit_pending = False
            to, freq = self.ghost_entry, min(self.GHOST_FREQ, self.MAX_FREQ)
        else:
            to, freq = self.entry, 0
            if not self.has_evicted and len(self.queues[to]) >= self.targets[to]:
                to = self.overflow
        self.queues[to].appendleft(key)
        self.state[key] = to << 4 | freq
        self.clock += 1
        if self.timed:
            self.stamps[key] = self.clock << 32 | self.clock
        return self.names[to]

    def _pick(self) -> int:
        queues, targets = self.queues, self.targets
        for i in range(self.k):
            if queues[i] and len(queues[i]) >= targets[i]:
                return i
        for i in range(self.k):
            if queues[i]:
                return i
        raise RuntimeError("evict() on an empty cache")

    def evict(self, now: int) -> int:
        self.has_evicted = True
        queues, state, names = self.queues, self.state, self.names
        s = self._pick()
        guard = 4 * self.capacity + 16  # a design whose moves cycle without hits is cut short
        while True:
            if not queues[s]:  # drained without an eviction: pick again
                s = self._pick()
            key = queues[s].pop()
            value = state[key]
            freq = value & 15
            guard -= 1
            if guard >= 0:
                age = self.age[s]
                idle = age and self.clock - (self.stamps[key] & 0xFFFFFFFF) >= age
                to = self.up[s]
                if freq >= self.promote[s] and not idle and not (self.promote[s] == 0 and to == s):
                    onup = self.onup[s]
                    if onup == 0:
                        freq = 0
                    elif onup == 2:
                        freq = freq - 1 if freq > 0 else 0
                    elif onup == 3:
                        freq >>= 1
                    state[key] = to << 4 | freq
                    queues[to].appendleft(key)
                    self.ctx.move(key, names[s], names[to])
                    continue
                to = self.down[s]
                if to >= 0:
                    state[key] = to << 4 | freq
                    queues[to].appendleft(key)
                    self.ctx.move(key, names[s], names[to])
                    s = self._pick()
                    continue
            del state[key]
            if self.timed:
                del self.stamps[key]
            if self.feeds[s]:
                self._remember(key)
            return key

    def _remember(self, key: int) -> None:
        if self.ghost_capacity <= 0 or key in self.ghost:
            return
        while len(self.ghost) >= self.ghost_capacity:
            old, _ = self.ghost.popitem(last=False)
            self.ctx.ghost_drop(old, "G")
        self.ghost[key] = None
        self.ctx.ghost_add(key, "G")

    def sizes(self) -> dict[str, int]:
        sizes = {name: len(queue) for name, queue in zip(self.names, self.queues)}
        sizes["G"] = len(self.ghost)
        return sizes

    # -- handover -------------------------------------------------------------------

    def adopt(self, state: Handover) -> dict[int, str]:
        """S3-FIFO's S objects go to ADOPT_SMALL, its M objects to ADOPT_MAIN, each in
        their order and with their counters (if that stage counts); its ghost is kept.
        Stages over their target drain as evictions happen."""
        placed = {}
        for objects, to in ((state.small, self.adopt_small), (state.main, self.adopt_main)):
            for key, freq in objects:
                self.queues[to].appendleft(key)
                self.state[key] = to << 4 | (min(freq, self.MAX_FREQ) if self.counts[to] else 0)
                if self.timed:
                    self.stamps[key] = self.clock << 32 | self.clock
                placed[key] = self.names[to]
        ghost = state.ghost
        for key in ghost[: max(0, len(ghost) - self.ghost_capacity)]:
            self.ctx.ghost_drop(key, "G")
        kept = ghost[max(0, len(ghost) - self.ghost_capacity) :] if self.ghost_capacity > 0 else []
        self.ghost = OrderedDict((key, None) for key in kept)
        self.has_evicted = not state.warm
        return placed


def design(config: str, base: type = VQ) -> type:
    """A VQ subclass from a vqsim config string, e.g. S3-FIFO:
    'maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0'
    (st=<size>,<counts>,<promote>,<up>,<onup>,<down>,<feeds_ghost>[,<delay>[,<age>]];
    onup 0 reset 1 keep 2 dec 3 half; down -1 = leave). Stages are named Q0, Q1, ..."""
    knobs, rows = {}, []
    for token in config.split():
        name, _, value = token.partition("=")
        if name == "st":
            rows.append(value.split(","))
        else:
            knobs[name] = value
    onup = {v: k for k, v in ONUP.items()}
    names = [f"Q{i}" for i in range(len(rows))]
    stages = []
    for i, row in enumerate(rows):
        extra = [float(x) for x in row[7:9]] + [0.0, 0.0]
        stages.append(
            Stage(
                names[i],
                float(row[0]),
                counts=row[1] != "0",
                promote=int(row[2]),
                up=names[int(row[3])],
                onup=onup[int(row[4])],
                down=None if int(row[5]) < 0 else names[int(row[5])],
                ghost=row[6] != "0",
                delay=extra[0],
                age=extra[1],
            )
        )

    def stage(key, default):
        return names[int(knobs[key])] if key in knobs else default

    attrs = {
        "STAGES": tuple(stages),
        "ENTRY": stage("entry", names[0]),
        "OVERFLOW": stage("overflow", None),
        "GHOST_ENTRY": stage("gentry", None),
        "GHOST_FREQ": int(knobs.get("gfreq", 0)),
        "GHOST_RATIO": float(knobs.get("ghost", 0.9)),
        "MAX_FREQ": int(knobs.get("maxf", 3)),
        "ADOPT_SMALL": stage("asmall", None),
        "ADOPT_MAIN": stage("amain", None),
    }
    return type("VQDesign", (base,), attrs)


# -- online re-choice among designs ------------------------------------------------------

S3_CONFIG = "maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0"
MASK64 = (1 << 64) - 1


def mix64(x: int) -> int:
    """splitmix64's finalizer (vqsim's): the spatial sample keeps keys whose hash is small."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def _timed(cls: type) -> bool:
    return any(stage.delay or stage.age for stage in cls.STAGES)


class OnlineVQ(Policy):
    """The live cache runs one of CANDIDATES (vqsim config strings) and switches design
    while it runs: at a handover (S3-FIFO served the cache so far) and at the start of
    every later window of PERIOD requests, it takes the candidate with the fewest misses
    since window FROM_WINDOW in miniature simulations. Each candidate runs on a spatial
    sample of the keys (RATE, raised so the miniature cache holds at least MIN_MINI
    objects; RATE >= 1 is an exact shadow) from the first request on, so the selector's
    metadata is about K * max(RATE * C, MIN_MINI) entries next to the cache's C. The live
    cache leaves its design only for a leader at least MARGIN (relative) better.
    Without a handover the cache starts as S3-FIFO and decides from window START on.

    A switch is a takeover: objects of the old entry stage go to the new design's
    ADOPT_SMALL stage, all others to its ADOPT_MAIN stage (tail first), counters capped
    (0 in a stage that does not count), hit times and the newest ghost entries kept.
    With a handover and PERIOD = requests / windows, this reproduces vqsim's online mode
    (`online C split windows rate=R min=M margin=X`; `start=W` without a handover)."""

    CANDIDATES: tuple[str, ...] = (S3_CONFIG,)
    PERIOD = 250_000
    FROM_WINDOW = 2
    START = 4
    RATE = 0.01
    MIN_MINI = 1000
    MARGIN = 0.0
    QUEUES = ("Q0", "Q1")
    GHOSTS = ("G",)
    DESIGNS: tuple[type, ...] = ()  # built from CANDIDATES

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.DESIGNS = tuple(design(config) for config in cls.CANDIDATES)
        cls.QUEUES = tuple(f"Q{i}" for i in range(max(len(d.STAGES) for d in cls.DESIGNS)))

    def __init__(self, capacity: int, ctx: Context) -> None:
        super().__init__(capacity, ctx)
        designs = self.DESIGNS
        if not designs or self.PERIOD < 1:
            raise ValueError("OnlineVQ needs CANDIDATES and PERIOD >= 1")
        # Hit times must survive a switch into a design that uses them.
        self.stamped = any(_timed(d) for d in designs)
        rate = self.RATE
        if rate * capacity < self.MIN_MINI:
            rate = self.MIN_MINI / capacity
        self.rate = min(rate, 1.0)
        self.limit = int(self.rate * 18446744073709551615.0)
        mini_capacity = max(1, int(self.rate * capacity + 0.5))
        null = Context()
        self.minis = [d(mini_capacity, null) for d in designs]
        self.misses = [0] * len(designs)  # sampled misses since FROM_WINDOW
        self.window = -1
        self.current = -1
        self.switches = 0
        self.live = self._fresh(S3FIFO)  # until the first decision

    def _fresh(self, cls: type) -> VQ:
        policy = cls(self.capacity, self.ctx)
        if self.stamped:
            policy.timed = True
        return policy

    # -- the miniature simulations and the choice ---------------------------------------

    def _sample(self, key: int, now: int) -> None:
        if self.rate < 1.0 and mix64(key & MASK64) > self.limit:
            return
        count = now // self.PERIOD >= self.FROM_WINDOW
        for i, mini in enumerate(self.minis):
            if key in mini.state:
                mini.on_hit(key, 0)
                continue
            mini.on_miss(key, 0)
            while len(mini.state) >= mini.capacity:
                mini.evict(0)
            mini.insert(key, 0)
            if count:
                self.misses[i] += 1

    def _leader(self) -> int:
        misses = self.misses
        leader = min(range(len(misses)), key=misses.__getitem__)
        current = self.current
        if current >= 0 and leader != current and misses[leader] >= (1.0 - self.MARGIN) * misses[current]:
            return current
        return leader

    def _tick(self, now: int) -> None:
        window = now // self.PERIOD
        if window == self.window:
            return
        self.window = window
        if self.current >= 0 or window >= self.START:
            leader = self._leader()
            if leader != self.current:
                self._switch(leader)

    def _switch(self, leader: int) -> None:
        old, new = self.live, self._fresh(self.DESIGNS[leader])
        first = self.current < 0  # from S3-FIFO: like a handover (hit times restart)
        for i in range(old.k):
            to = new.adopt_small if i == old.entry else new.adopt_main
            counts, queue, name = new.counts[to], new.queues[to], new.names[to]
            for key in reversed(old.queues[i]):  # tail first
                freq = old.state[key] & 15
                new.state[key] = to << 4 | (min(freq, new.MAX_FREQ) if counts else 0)
                queue.appendleft(key)
                self.ctx.move(key, old.names[i], name)
        if self.stamped:
            new.stamps = {key: 0 for key in old.stamps} if first else old.stamps
        new.clock = 0 if first else old.clock
        new.has_evicted = old.has_evicted
        self._keep_ghost(new, list(old.ghost))
        if not first:
            self.switches += 1
        self.live, self.current = new, leader

    def _keep_ghost(self, new: VQ, ghost: list[int]) -> None:
        drop = max(0, len(ghost) - max(0, new.ghost_capacity))
        for key in ghost[:drop]:
            self.ctx.ghost_drop(key, "G")
        new.ghost = OrderedDict((key, None) for key in ghost[drop:])

    # -- hooks ----------------------------------------------------------------------------

    def observe(self, key: int, now: int, hit: bool) -> None:
        self._sample(key, now)

    def adopt(self, state: Handover) -> dict[int, str]:
        self.window = state.now // self.PERIOD
        leader = self._leader()
        self.live = self._fresh(self.DESIGNS[leader])
        self.current = leader
        return self.live.adopt(state)

    def on_hit(self, key: int, now: int) -> None:
        self._tick(now)
        self.live.on_hit(key, now)
        self._sample(key, now)

    def on_miss(self, key: int, now: int) -> bool:
        self._tick(now)
        admitted = self.live.on_miss(key, now)
        self._sample(key, now)
        return admitted

    def evict(self, now: int) -> int:
        return self.live.evict(now)

    def insert(self, key: int, now: int) -> str:
        return self.live.insert(key, now)

    def sizes(self) -> dict[str, int]:
        sizes = dict.fromkeys(self.QUEUES, 0)
        sizes.update(self.live.sizes())
        return sizes


S3FIFO = design(S3_CONFIG)  # S3-FIFO with stages Q0, Q1
OnlineVQ.DESIGNS = (S3FIFO,)


# -- version 1 ------------------------------------------------------------------------------

# The v1 portfolio (frozen 2026-11-17; scripts/vqsim/v1/portfolio.json): 32 designs picked
# greedily, from S3-FIFO on, for the idealized online choice on the train-long cells, out of 426
# host-swept seeds and the 84 designs the architects' board chose. Order matters (ties go to the
# earlier one). Pre-registered heldout-long result (88 cells, unseen 80%, vs S3-FIFO): mean -3.42%,
# median -1.79%; 3L-Cache -2.73%, -1.24% (docs/cachelab-v1.md).
V1_CANDIDATES = (
    'maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0',  # seed (s3fifo)
    'maxf=7 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 asmall=0 amain=1 st=0.03,1,2,1,0,-1,1,1.0,0.0 st=0,1,1,1,2,-1,0,0.0,0.0',  # architect: F67 median max_freq=7 S.size=0.03 S.delay=1 M.promote=1
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=1 asmall=0 amain=2 st=0.15,0,0,1,0,-1,0,0.0,0.0 st=0.15,1,1,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F79 large ghost_ratio=0.9 skip.size=0.15 S.size=0.15
    'maxf=5 entry=0 overflow=4 ghost=0.5 gentry=1 gfreq=0 st=0.2,1,2,4,0,1,0 st=0.01,0,0,2,1,-1,0 st=0.2,0,0,4,1,-1,0 st=0.6,1,1,3,0,4,0 st=0,1,1,4,0,-1,0',  # seed (vq)
    'maxf=1 entry=0 overflow=2 ghost=2.0 gentry=1 gfreq=0 asmall=0 amain=2 st=0.05,1,2,2,0,-1,1,0.0,0.0 st=0.3,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F80 small ghost_ratio=2 max_freq=1 S.size=0.05 G2.size=0.3
    'maxf=7 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 asmall=0 amain=1 st=0.01,1,2,1,0,-1,1,1.0,0.0 st=0,1,2,1,2,-1,0,0.0,0.0',  # architect: F95 gaps max_freq=7 S.size=0.01 S.delay=1 M.promote=2
    'maxf=1 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 asmall=0 amain=1 st=0.3,1,2,1,0,-1,1,0.0,0.0 st=0,1,1,1,2,-1,0,1.0,0.0',  # architect: F77 gaps max_freq=1 S.size=0.3 M.promote=1 M.delay=1
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.025,0,0,1,1,-1,0 st=0.025,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0',  # seed (s4like)
    'maxf=3 entry=0 overflow=2 ghost=0.5 gentry=2 gfreq=0 st=0.1,0,0,1,1,-1,0 st=0.1,1,2,2,0,-1,1 st=0,1,1,2,2,-1,0',  # seed (s4like)
    'maxf=2 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 st=0.3,1,1,1,0,-1,1 st=0,1,1,1,2,-1,0',  # seed (s3grid)
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=0 asmall=0 amain=2 st=0.05,1,2,1,0,-1,1,0.0,0.0 st=0.1,1,1,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F6 front ghost_entry=P2 P1.size=0.05 P2.size=0.1 P2.promote=1
    'maxf=3 entry=0 overflow=1 ghost=0.5 gentry=0 gfreq=1 asmall=0 amain=1 st=0.05,1,2,1,0,-1,1,0.0,0.0 st=0,1,1,1,2,-1,0,0.0,0.0',  # architect: F5 joint ghost_entry=S ghost_freq=1 ghost_ratio=0.5 S.size=0.05 S.promote=2
    'maxf=3 entry=0 overflow=3 ghost=0.9 gentry=0 gfreq=1 st=0.02,1,2,1,0,-1,1 st=0.02,1,2,3,0,2,0 st=0.05,1,1,3,0,-1,1 st=0,1,1,3,2,-1,0',  # seed (vq)
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0.01,1,3,2,0,-1,1 st=0,1,1,2,2,-1,0',  # seed (vq)
    'maxf=3 entry=0 overflow=2 ghost=0.5 gentry=2 gfreq=0 st=0.025,0,0,1,1,-1,0 st=0.025,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0',  # seed (s4like)
    'maxf=1 entry=0 overflow=2 ghost=1.8 gentry=1 gfreq=0 asmall=0 amain=2 st=0.08,1,2,1,0,-1,1,0.0,0.0 st=0.3,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F108 robust ghost_ratio=1.8 S.size=0.08 S.promote=2 V.size=0.3
    'maxf=2 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 st=0.05,1,3,1,0,-1,1 st=0,1,1,1,2,-1,0',  # seed (s3grid)
    'maxf=3 entry=0 overflow=2 ghost=1.5 gentry=1 gfreq=0 asmall=0 amain=2 st=0.1,1,2,1,0,-1,1,0.0,0.0 st=0.2,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F89 robust ghost_ratio=1.5 max_freq=3 S.promote=2 V.size=0.2
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=1 asmall=0 amain=2 st=0.05,0,0,1,0,-1,0,0.0,0.0 st=0.12,1,1,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,1.0,0.0',  # architect: F52 main S.size=0.12 S.delay=0 M.delay=1
    'maxf=2 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.02,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0',  # seed (s3grid)
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=0 asmall=0 amain=2 st=0.3,1,1,1,0,-1,1,0.0,0.0 st=0.15,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F104 gaps max_freq=3 S.size=0.3 S.promote=1 S.up=V S.delay=0
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=1 gfreq=0 asmall=0 amain=2 st=0.02,1,1,2,0,-1,1,1.0,0.0 st=0.15,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F104 gaps max_freq=3 S.size=0.02 S.promote=1 S.up=M S.delay=1
    'maxf=7 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 asmall=0 amain=2 st=0.05,0,0,1,0,-1,0,0.0,0.0 st=0.03,1,2,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F78 median ghost_ratio=0.9 max_freq=7 S.size=0.03 S.delay=0 M.promote=1
    'maxf=7 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 asmall=0 amain=1 st=0.01,1,2,1,0,-1,1,0.0,0.0 st=0,1,2,1,2,-1,0,0.0,0.0',  # architect: F95 gaps max_freq=7 S.size=0.01 S.delay=0 M.promote=2
    'maxf=7 entry=0 overflow=2 ghost=0.5 gentry=2 gfreq=1 st=0.01,1,2,1,0,1,0 st=0.1,1,1,2,0,-1,1 st=0,1,1,2,2,-1,1',  # seed (vq)
    'maxf=1 entry=0 overflow=2 ghost=0.5 gentry=2 gfreq=0 asmall=0 amain=2 st=0.05,0,0,1,0,-1,0,0.0,0.0 st=0.15,1,1,2,0,-1,1,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F41 timing max_freq=1 S.size=0.15 M.delay=0
    'maxf=2 entry=0 overflow=3 ghost=0.9 gentry=3 gfreq=1 st=0.1,1,2,1,0,-1,1 st=0.05,1,1,3,0,-1,1 st=0.6,1,1,2,0,3,0 st=0,1,1,2,0,-1,0',  # seed (vq)
    'maxf=3 entry=0 overflow=2 ghost=2.0 gentry=1 gfreq=0 asmall=0 amain=2 st=0.3,1,2,1,0,-1,1,0.0,0.0 st=0.3,1,1,2,0,-1,0,0.0,0.0 st=0,1,1,2,2,-1,0,0.0,0.0',  # architect: F101 robust ghost_ratio=2 S.size=0.3 S.promote=2
    'maxf=2 entry=0 overflow=2 ghost=0.9 gentry=0 gfreq=0 st=0.2,1,2,2,1,1,0 st=0.05,0,0,2,1,-1,0 st=0,1,1,2,2,-1,0',  # seed (vq)
    'maxf=5 entry=0 overflow=1 ghost=0.5 gentry=1 gfreq=0 st=0.1,1,3,1,0,-1,1 st=0,1,1,1,2,-1,0',  # seed (s3grid)
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.0125,0,0,1,1,-1,0 st=0.0375,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0',  # seed (s4like)
    'maxf=3 entry=0 overflow=2 ghost=0.9 gentry=2 gfreq=0 st=0.05,0,0,1,1,-1,0 st=0.15,1,1,2,0,-1,1 st=0,1,1,2,2,-1,0',  # seed (s4like)
)


class OnlineV1(OnlineVQ):
    """Version 1: online re-choice among V1_CANDIDATES with 0.5% miniature simulations (each
    caching at least 200 objects), margin 0. The evaluation decided at the split and at each of
    40 windows of a trace (requests / 40, 250K-1.2M requests on train-long); PERIOD is that window
    for a live cache."""

    CANDIDATES = V1_CANDIDATES
    PERIOD = 500_000
    RATE = 0.005
    MIN_MINI = 200
    MARGIN = 0.0

CachePolicy = VQ
