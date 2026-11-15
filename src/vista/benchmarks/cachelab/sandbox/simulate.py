"""Drive cache policies over traces and record every decision (stdlib only).

    python -I simulate.py JOBS.json

JOBS.json holds {"workers": int, "mode": "fast" | "verified", "jobs": [...]}.
A fast job is

    {"policy": path, "trace": path, "capacity": int, "out": directory,
     "sample_every": int, "max_seconds": float, "handover": int}

and runs the policy in this process. With "handover": h > 0, S3-FIFO serves
requests [0, h) and the policy observes them (observe), then takes over the
live cache at request h (adopt); see policy_api.py. A verified job replaces "policy" with
"command", the argv of a policy_server.py process. That process receives one
request at a time and never sees the trace, so it cannot look ahead or touch
this process's records.

A trace is a raw little-endian uint64 array of object keys; request i happens
at virtual time i. Each job writes raw arrays and `summary.json` to its `out`
directory, or `error.json` if the policy failed. This file runs inside the
sandbox next to policy_api.py, so it must not import anything outside the
standard library.
"""

from __future__ import annotations

import array
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from policy_api import Context, Handover, Policy  # noqa: E402

# Request outcomes.
HIT, COLD, REMISS, BYPASS = 0, 1, 2, 3
# Policy-reported events.
MOVE, GHOST_ADD, GHOST_DROP, GHOST_HIT = 0, 1, 2, 3
NO_QUEUE = 255
MAX_ERROR_EXAMPLES = 8
FORMAT_VERSION = 1


def _typed(code: str, size: int) -> array.array:
    values = array.array(code)
    if values.itemsize != size:
        raise RuntimeError(f"array typecode {code!r} is not {size} bytes here")
    return values


class PolicyError(Exception):
    """The policy broke the simulator contract."""


class Recorder(Context):
    def __init__(self, queues: tuple[str, ...], ghosts: tuple[str, ...]) -> None:
        self.queue_index = {name: index for index, name in enumerate(queues)}
        self.ghost_index = {name: index for index, name in enumerate(ghosts)}
        self.now = 0
        self.queue_of: dict[int, int] = {}
        self.counts = [0] * len(queues)
        self.t = _typed("I", 4)
        self.key = _typed("Q", 8)
        self.kind = _typed("B", 1)
        self.a = _typed("B", 1)
        self.b = _typed("B", 1)
        self.errors = 0
        self.examples: list[str] = []
        self.ghost_hit_now = False

    def _error(self, message: str) -> None:
        self.errors += 1
        if len(self.examples) < MAX_ERROR_EXAMPLES:
            self.examples.append(f"t={self.now}: {message}")

    def _event(self, kind: int, key: int, first: int, second: int) -> None:
        self.t.append(self.now)
        self.key.append(key)
        self.kind.append(kind)
        self.a.append(first)
        self.b.append(second)

    def move(self, key: int, src: str, dst: str) -> None:
        source = self.queue_index.get(src)
        target = self.queue_index.get(dst)
        if source is None or target is None:
            raise PolicyError(f"move() names an unknown queue: {src!r} -> {dst!r}")
        tracked = self.queue_of.get(key)
        if tracked is None:
            self._error(f"move({key}, {src!r}, {dst!r}) for a non-resident key")
            return
        if tracked != source:
            self._error(f"move({key}, {src!r}, {dst!r}) but the simulator tracked it elsewhere")
        self.counts[tracked] -= 1
        self.counts[target] += 1
        self.queue_of[key] = target
        self._event(MOVE, key, source, target)

    def _ghost(self, kind: int, key: int, ghost: str) -> None:
        index = self.ghost_index.get(ghost)
        if index is None:
            raise PolicyError(f"unknown ghost queue {ghost!r}")
        self._event(kind, key, index, NO_QUEUE)

    def ghost_add(self, key: int, ghost: str = "G") -> None:
        self._ghost(GHOST_ADD, key, ghost)

    def ghost_drop(self, key: int, ghost: str = "G") -> None:
        self._ghost(GHOST_DROP, key, ghost)

    def ghost_hit(self, key: int, ghost: str = "G") -> None:
        self.ghost_hit_now = True
        self._ghost(GHOST_HIT, key, ghost)


class PolicyContext(Context):
    """The only simulator object a policy sees: the four reporting hooks."""

    __slots__ = ("_recorder",)

    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder

    def move(self, key: int, src: str, dst: str) -> None:
        self._recorder.move(key, src, dst)

    def ghost_add(self, key: int, ghost: str = "G") -> None:
        self._recorder.ghost_add(key, ghost)

    def ghost_drop(self, key: int, ghost: str = "G") -> None:
        self._recorder.ghost_drop(key, ghost)

    def ghost_hit(self, key: int, ghost: str = "G") -> None:
        self._recorder.ghost_hit(key, ghost)


def load_policy(path: str) -> type[Policy]:
    spec = importlib.util.spec_from_file_location("candidate_policy", path)
    if spec is None or spec.loader is None:
        raise PolicyError("cannot load the policy module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_policy"] = module
    spec.loader.exec_module(module)
    cls = getattr(module, "CachePolicy", None)
    if not isinstance(cls, type) or not issubclass(cls, Policy):
        raise PolicyError("the module must define CachePolicy, a Policy subclass")
    check_names(tuple(cls.QUEUES), tuple(cls.GHOSTS))
    return cls


def incumbent_class() -> type[Policy]:
    """A private copy of the reference S3-FIFO for the cache before a handover,
    loaded before any candidate code runs, so a candidate cannot alter it."""
    spec = importlib.util.spec_from_file_location("cachelab_incumbent", HERE / "s3fifo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.S3FIFO


def place(ctx: "Recorder", placement, now: int, queues: tuple[str, ...]) -> None:
    """Apply adopt()'s placement: exactly the resident objects, into the new policy's queues."""
    if not isinstance(placement, dict):
        raise PolicyError("adopt() must return a dict {key: queue name}")
    resident = ctx.queue_of
    if len(placement) != len(resident) or any(key not in placement for key in resident):
        missing = [key for key in resident if key not in placement][:3]
        extra = [key for key in placement if key not in resident][:3]
        raise PolicyError(
            f"adopt() must place exactly the {len(resident)} resident objects "
            f"(got {len(placement)}; missing e.g. {missing}; not resident e.g. {extra})"
        )
    allowed = set(queues)
    for key, name in placement.items():
        if name not in allowed:
            raise PolicyError(f"adopt() placed {key} in {name!r}, which is not in QUEUES")
        index = ctx.queue_index[name]
        old = resident[key]
        if old != index:
            ctx.counts[old] -= 1
            ctx.counts[index] += 1
            resident[key] = index
            ctx._event(MOVE, key, old, index)


def check_names(queues: tuple, ghosts: tuple) -> None:
    names = queues + ghosts
    if not queues or len(queues) > 8 or len(ghosts) > 2:
        raise PolicyError("QUEUES needs 1-8 names and GHOSTS at most 2")
    if len(set(names)) != len(names) or not all(
        isinstance(name, str) and 0 < len(name) <= 8 and name.isalnum() for name in names
    ):
        raise PolicyError("queue names must be distinct alphanumeric strings of 1-8 characters")


def read_trace(path: str) -> array.array:
    keys = _typed("Q", 8)
    with open(path, "rb") as stream:
        keys.frombytes(stream.read())
    if sys.byteorder != "little":
        keys.byteswap()
    return keys


class RemotePolicy:
    """Host-side proxy for a policy running in a policy_server.py process.

    Hits need no answer, so they are buffered until the next request that
    does. Events the policy reports carry their own request time.
    """

    def __init__(self, command: list[str], max_seconds: float) -> None:
        import socket
        import subprocess
        import tempfile
        import threading

        # The policy process connects to a Unix socket in a fresh directory;
        # "{socket_dir}" in the command names that directory.
        self.directory = tempfile.mkdtemp(prefix="cl-")
        address = os.path.join(self.directory, "policy.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(address)
        self.listener.listen(1)
        uses_socket = any("{socket_dir}" in part for part in command)
        argv = [part.replace("{socket_dir}", self.directory) for part in command]
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL if uses_socket else subprocess.PIPE,
            stdout=subprocess.DEVNULL if uses_socket else subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1 << 16,
        )
        self.expired = False
        self.connection = None
        self.timer = threading.Timer(max_seconds, self._expire)
        self.timer.daemon = True
        self.timer.start()
        if uses_socket:
            self.listener.settimeout(1.0)
            waited = 0.0
            while self.connection is None:
                try:
                    self.connection, _ = self.listener.accept()
                except TimeoutError:
                    waited += 1.0
                    if self.process.poll() is not None or waited > 120:
                        raise PolicyError("the policy process did not start") from None
            self.connection.settimeout(None)
            self.writer = self.connection.makefile("w", encoding="utf-8", newline="\n")
            self.reader = self.connection.makefile("r", encoding="utf-8", newline="\n")
        else:
            self.writer, self.reader = self.process.stdin, self.process.stdout
        self.pending: list[str] = []
        self.ctx: Recorder | None = None

    def _expire(self) -> None:
        self.expired = True
        self.process.kill()
        if self.connection is not None:
            try:
                import socket

                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def hello(self, capacity: int) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
        parts = self._call(f"I {capacity}").split(" ")
        if len(parts) != 3:
            raise PolicyError("the policy process broke the protocol")
        queues = tuple(parts[0].split(","))
        ghosts = tuple(name for name in parts[1].split(",") if name)
        return queues, ghosts, parts[2] == "1"

    def bind(self, capacity: int, ctx: "Recorder") -> "RemotePolicy":
        self.ctx = ctx
        return self

    def _call(self, line: str) -> str:
        self.pending.append(line)
        try:
            self.writer.write("\n".join(self.pending) + "\n")
            self.writer.flush()
        except (BrokenPipeError, OSError):
            raise PolicyError(self._exit_reason()) from None
        self.pending.clear()
        while True:
            reply = self.reader.readline()
            if not reply:
                raise PolicyError(self._exit_reason())
            tag, _, rest = reply.rstrip("\n").partition(" ")
            if tag == "R":
                return rest
            if tag == "V":
                self._event(rest)
            elif tag == "X":
                raise PolicyError(rest)
            else:
                raise PolicyError("the policy process broke the protocol")

    def _exit_reason(self) -> str:
        if self.expired:
            return "time limit exceeded"
        return "the policy process exited unexpectedly"

    def _event(self, text: str) -> None:
        parts = text.split(" ")
        ctx = self.ctx
        current = ctx.now
        try:
            ctx.now = int(parts[-1])
            key = int(parts[1])
            if parts[0] == "m":
                ctx.move(key, parts[2], parts[3])
            elif parts[0] == "a":
                ctx.ghost_add(key, parts[2])
            elif parts[0] == "d":
                ctx.ghost_drop(key, parts[2])
            elif parts[0] == "h":
                ctx.ghost_hit(key, parts[2])
            else:
                raise PolicyError("unknown event from the policy process")
        except (IndexError, ValueError):
            raise PolicyError("malformed event from the policy process") from None
        finally:
            ctx.now = current

    def on_hit(self, key: int, now: int) -> None:
        self.pending.append(f"H {key} {now}")
        if len(self.pending) >= 4096:
            self._call("P")

    def on_miss(self, key: int, now: int) -> bool:
        return self._call(f"M {key} {now}") == "1"

    def observe(self, key: int, now: int, hit: bool) -> None:
        self.pending.append(f"O {key} {now} {1 if hit else 0}")
        if len(self.pending) >= 4096:
            self._call("P")

    def adopt(self, state: Handover) -> dict[int, str]:
        reply = self._call("A " + json.dumps(state.to_json(), separators=(",", ":")))
        try:
            data = json.loads(reply)
            return {int(key): str(name) for key, name in data.items()}
        except (ValueError, AttributeError):
            raise PolicyError("adopt() did not return a dict {key: queue name}") from None

    def evict(self, now: int) -> int:
        reply = self._call(f"E {now}")
        try:
            return int(reply)
        except ValueError:
            raise PolicyError("evict() did not return an integer key") from None

    def insert(self, key: int, now: int) -> str:
        return self._call(f"N {key} {now}")

    def miss(self, key: int, now: int, full: bool) -> tuple[bool, int | None, str]:
        """on_miss, then evict() if the cache is full, then insert(): one round trip."""
        reply = self._call(f"W {key} {now} {1 if full else 0}")
        if reply == "0":
            return False, None, ""
        try:
            flag, victim, queue = reply.split(" ")
            return True, (None if victim == "-" else int(victim)), queue
        except ValueError:
            raise PolicyError("the policy process broke the protocol") from None

    def sizes(self) -> dict[str, int]:
        reply = self._call("S")
        sizes = {}
        for item in filter(None, reply.split(",")):
            name, _, value = item.partition("=")
            sizes[name] = int(value)
        return sizes

    def close(self) -> None:
        import shutil

        self.timer.cancel()
        try:
            self.writer.write("Q\n")
            self.writer.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()
        for stream in (self.reader, self.connection, self.listener):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        shutil.rmtree(self.directory, ignore_errors=True)


def simulate(job: dict) -> dict:
    if "command" in job:
        remote = RemotePolicy(list(job["command"]), float(job.get("max_seconds", 600)))
        try:
            incumbent = incumbent_class() if job.get("handover") else None
            queues, ghosts, observes = remote.hello(int(job["capacity"]))
            check_names(queues, ghosts)
            summary = drive(remote.bind, queues, ghosts, job, incumbent=incumbent, observes=observes)
            current, peak = (int(value) for value in remote._call("T").split(","))
            summary["policy_memory_bytes"] = current
            summary["policy_memory_peak_bytes"] = peak
            return summary
        finally:
            remote.close()
    incumbent = incumbent_class() if job.get("handover") else None
    cls = load_policy(job["policy"])

    def build(capacity: int, recorder: Recorder) -> Policy:
        return cls(capacity, PolicyContext(recorder))

    observes = cls.observe is not Policy.observe
    return drive(build, tuple(cls.QUEUES), tuple(cls.GHOSTS), job, incumbent=incumbent, observes=observes)


def drive(factory, queues: tuple[str, ...], ghosts: tuple[str, ...], job: dict, *, incumbent=None, observes: bool = False) -> dict:
    capacity = int(job["capacity"])
    if capacity < 1:
        raise PolicyError("capacity must be positive")
    trace = read_trace(job["trace"])
    n = len(trace)
    sample_every = max(1, int(job.get("sample_every") or max(1, n // 2000)))
    deadline = time.monotonic() + float(job.get("max_seconds", 600))
    switch = int(job.get("handover") or 0)
    if not 0 <= switch < max(1, n):
        raise ValueError(f"handover {switch} is outside the trace")
    policy_queues = queues
    if switch:
        # Views see both: the incumbent's queues before the switch, the policy's after.
        queues = tuple(incumbent.QUEUES) + tuple(name for name in queues if name not in incumbent.QUEUES)
        ghosts = tuple(incumbent.GHOSTS) + tuple(name for name in ghosts if name not in incumbent.GHOSTS)

    ctx = Recorder(queues, ghosts)
    policy = factory(capacity, ctx)
    active = incumbent(capacity, PolicyContext(ctx)) if switch else policy
    observing = bool(switch) and observes
    queue_of = ctx.queue_of
    queue_index = ctx.queue_index
    counts = ctx.counts

    outcome = _typed("B", 1)
    queue_at = _typed("B", 1)
    ghost_hit = _typed("B", 1)
    evict_t, evict_key = _typed("I", 4), _typed("Q", 8)
    evict_q, evict_ins, evict_hits = _typed("B", 1), _typed("I", 4), _typed("I", 4)
    sample_t = _typed("I", 4)
    sample_sizes = _typed("q", 8)
    size_mismatches = 0
    size_examples: list[str] = []
    inserted_at: dict[int, int] = {}
    hits_of: dict[int, int] = {}
    seen: set[int] = set()

    def remove(victim: int, now: int) -> None:
        where = queue_of.pop(victim, None)
        if where is None:
            raise PolicyError(f"evict() returned {victim!r}, which is not resident")
        counts[where] -= 1
        evict_t.append(now)
        evict_key.append(victim)
        evict_q.append(where)
        evict_ins.append(inserted_at.pop(victim))
        evict_hits.append(hits_of.pop(victim))

    # A policy in another process answers a whole miss in one round trip.
    combined = active.miss if isinstance(active, RemotePolicy) else None
    names = queues + ghosts
    for now in range(n):
        key = trace[now]
        ctx.now = now
        if now % 65536 == 0 and time.monotonic() > deadline:
            raise PolicyError(f"time limit exceeded at request {now} of {n}")
        if now == switch and active is not policy:
            place(ctx, policy.adopt(active.export(now)), now, policy_queues)
            active, observing = policy, False
            combined = active.miss if isinstance(active, RemotePolicy) else None
        tracked = queue_of.get(key)
        if tracked is not None:
            outcome.append(HIT)
            queue_at.append(tracked)
            ghost_hit.append(0)
            hits_of[key] += 1
            active.on_hit(key, now)
        else:
            ctx.ghost_hit_now = False
            if combined is None:
                admitted = active.on_miss(key, now)
                if admitted:
                    while len(queue_of) >= capacity:
                        remove(active.evict(now), now)
                    name = active.insert(key, now)
            else:
                admitted, victim, name = combined(key, now, len(queue_of) >= capacity)
                if victim is not None:
                    remove(victim, now)
            if admitted:
                index = queue_index.get(name)
                if index is None:
                    raise PolicyError(f"insert() returned unknown queue {name!r}")
                queue_of[key] = index
                counts[index] += 1
                inserted_at[key] = now
                hits_of[key] = 0
                outcome.append(REMISS if key in seen else COLD)
                queue_at.append(index)
            else:
                outcome.append(BYPASS)
                queue_at.append(NO_QUEUE)
            ghost_hit.append(1 if ctx.ghost_hit_now else 0)
            seen.add(key)
        if observing:
            policy.observe(key, now, tracked is not None)
        if now % sample_every == 0 or now == n - 1:
            sizes = active.sizes()
            sample_t.append(now)
            for name in names:
                sample_sizes.append(int(sizes.get(name, 0)))
            resident = ctx.counts
            reported = [int(sizes.get(name, 0)) for name in queues]
            if reported != resident:
                size_mismatches += 1
                if len(size_examples) < MAX_ERROR_EXAMPLES:
                    size_examples.append(
                        f"t={now}: sizes() reported {reported}, simulator tracked {resident}"
                    )

    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "outcome": outcome,
        "queue": queue_at,
        "ghost_hit": ghost_hit,
        "ev_t": ctx.t,
        "ev_key": ctx.key,
        "ev_kind": ctx.kind,
        "ev_a": ctx.a,
        "ev_b": ctx.b,
        "evict_t": evict_t,
        "evict_key": evict_key,
        "evict_q": evict_q,
        "evict_ins": evict_ins,
        "evict_hits": evict_hits,
        "sample_t": sample_t,
        "sample_sizes": sample_sizes,
    }
    for name, values in arrays.items():
        if sys.byteorder != "little":
            values = array.array(values.typecode, values)
            values.byteswap()
        with open(out / f"{name}.bin", "wb") as stream:
            values.tofile(stream)
    misses = n - outcome.count(HIT)
    summary = {
        "format": FORMAT_VERSION,
        "requests": n,
        "capacity": capacity,
        "queues": list(queues),
        "ghosts": list(ghosts),
        "policy_queues": list(policy_queues),
        "handover_at": switch or None,
        "misses_after_handover": (n - switch) - outcome[switch:].count(HIT) if switch else None,
        "misses": misses,
        "miss_ratio": misses / n if n else 0.0,
        "cold_misses": outcome.count(COLD),
        "bypasses": outcome.count(BYPASS),
        "evictions": len(evict_t),
        "events": len(ctx.t),
        "sample_every": sample_every,
        "instrumentation_errors": ctx.errors,
        "instrumentation_examples": ctx.examples,
        "size_mismatches": size_mismatches,
        "size_examples": size_examples,
        "mode": "verified" if isinstance(policy, RemotePolicy) else "fast",
    }
    return summary


def run_job(job: dict) -> dict:
    started = time.monotonic()
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        summary = simulate(job)
    except BaseException as exc:  # the candidate is untrusted; report everything
        lines = traceback.format_exception(exc)
        error = {
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "traceback": "".join(lines)[-6000:],
        }
        (out / "error.json").write_text(json.dumps(error), encoding="utf-8")
        return {"out": str(out), "ok": False, **error}
    summary["seconds"] = round(time.monotonic() - started, 3)
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return {"out": str(out), "ok": True}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    jobs = spec["jobs"]
    workers = max(1, min(int(spec.get("workers", 1)), len(jobs) or 1, os.cpu_count() or 1))
    if workers == 1:
        results = [run_job(job) for job in jobs]
    else:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.map(run_job, jobs, chunksize=1)
    print(json.dumps({"results": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
