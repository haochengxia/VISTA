"""Serve one policy over stdin/stdout for verified simulation (stdlib only).

    python -I policy_server.py POLICY.py [--measure-memory] [--socket PATH]

With --socket the protocol runs over a Unix socket the driver listens on
(bind-mounted into the container), which avoids the container runtime's
stdio relay on every request; otherwise it uses stdin/stdout.

The trusted driver sends one line per hook and this process answers with
`R <value>`, preceded by `V ...` lines for every event the policy reported.
`X <message>` reports a policy failure. This process never receives the
trace, only the request it is currently answering.

    I <capacity>      -> R <queue,queue,...> <ghost,...> <observes 1|0>
    H <key> <now>     (no reply)
    O <key> <now> <hit>   (no reply)            observe, before a handover
    A <json state>    -> R <json {key: queue}>  adopt, at a handover
    M <key> <now>     -> R 1 | R 0          on_miss
    E <now>           -> R <key>            evict
    N <key> <now>     -> R <queue>          insert
    W <key> <now> <full>  -> R 0 | R 1 <victim|-> <queue>
                      on_miss; if admitted: evict() when full, then insert()
    S                 -> R <name>=<n>,...   sizes
    P                 -> R                  flush buffered hits
    T                 -> R <current>,<peak> traced Python memory in bytes
    Q                                       quit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from policy_api import Context, Handover, Policy  # noqa: E402
from simulate import load_policy  # noqa: E402


class PipeContext(Context):
    def __init__(self, out: list[str]) -> None:
        self.out = out
        self.now = 0

    def move(self, key: int, src: str, dst: str) -> None:
        self.out.append(f"V m {int(key)} {src} {dst} {self.now}")

    def ghost_add(self, key: int, ghost: str = "G") -> None:
        self.out.append(f"V a {int(key)} {ghost} {self.now}")

    def ghost_drop(self, key: int, ghost: str = "G") -> None:
        self.out.append(f"V d {int(key)} {ghost} {self.now}")

    def ghost_hit(self, key: int, ghost: str = "G") -> None:
        self.out.append(f"V h {int(key)} {ghost} {self.now}")


def policy_bytes(tracemalloc) -> int:
    """Live memory allocated by policy code: everything traced except the import
    system (compiled or unmarshalled code, finder caches) and this server's own
    buffers. The import system's share varies by up to ~0.25 MB between runs of
    the same module (bytecode cache, finder state), which would otherwise decide
    the memory bound at small cache sizes."""
    excluded = ("<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>", __file__, str(HERE / "simulate.py"), tracemalloc.__file__)
    snapshot = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(False, name) for name in excluded])
    return sum(stat.size for stat in snapshot.statistics("filename"))


def serve(path: str, measure: bool = False, socket_path: str | None = None) -> int:
    if socket_path is not None:
        import socket

        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(socket_path)
        stdin = connection.makefile("r", encoding="utf-8", newline="\n")
        stdout = connection.makefile("w", encoding="utf-8", newline="\n")
    else:
        stdin, stdout = sys.stdin, sys.stdout
    if measure:
        import tracemalloc

        tracemalloc.start()
    out: list[str] = []
    ctx = PipeContext(out)
    cls = None
    policy = None

    def reply(value: object = "") -> None:
        out.append(f"R {value}")
        stdout.write("\n".join(out) + "\n")
        stdout.flush()
        out.clear()

    try:
        cls = load_policy(path)
        for line in stdin:
            op, _, rest = line.rstrip("\n").partition(" ")
            if op == "H":
                key, now = rest.split(" ")
                ctx.now = int(now)
                policy.on_hit(int(key), ctx.now)
            elif op == "O":
                key, now, hit = rest.split(" ")
                ctx.now = int(now)
                policy.observe(int(key), ctx.now, hit == "1")
            elif op == "A":
                state = Handover.from_json(json.loads(rest))
                ctx.now = state.now
                placement = policy.adopt(state)
                if not isinstance(placement, dict):
                    raise TypeError("adopt() must return a dict {key: queue name}")
                reply(json.dumps({str(int(key)): str(name) for key, name in placement.items()}, separators=(",", ":")))
            elif op == "M":
                key, now = rest.split(" ")
                ctx.now = int(now)
                reply("1" if policy.on_miss(int(key), ctx.now) else "0")
            elif op == "E":
                ctx.now = int(rest)
                reply(int(policy.evict(ctx.now)))
            elif op == "W":
                key, now, full = rest.split(" ")
                ctx.now = int(now)
                if not policy.on_miss(int(key), ctx.now):
                    reply("0")
                    continue
                victim = int(policy.evict(ctx.now)) if full == "1" else "-"
                reply(f"1 {victim} {policy.insert(int(key), ctx.now)}")
            elif op == "N":
                key, now = rest.split(" ")
                ctx.now = int(now)
                reply(str(policy.insert(int(key), ctx.now)))
            elif op == "S":
                sizes = policy.sizes()
                reply(",".join(f"{name}={int(value)}" for name, value in sizes.items()))
            elif op == "P":
                reply()
            elif op == "T":
                if measure:
                    _, peak = tracemalloc.get_traced_memory()
                    reply(f"{policy_bytes(tracemalloc)},{peak}")
                else:
                    reply("-1,-1")
            elif op == "I":
                policy = cls(int(rest), ctx)
                observes = type(policy).observe is not Policy.observe
                reply(",".join(cls.QUEUES) + " " + ",".join(cls.GHOSTS) + " " + ("1" if observes else "0"))
            elif op == "Q":
                return 0
            else:
                raise RuntimeError(f"unknown request {op!r}")
    except BaseException as exc:  # report the policy's failure to the driver
        message = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:2000]
        out.clear()
        stdout.write(f"X {message}\n")
        stdout.flush()
        return 1
    return 0


if __name__ == "__main__":
    options = sys.argv[2:]
    socket_path = options[options.index("--socket") + 1] if "--socket" in options else None
    raise SystemExit(serve(sys.argv[1], "--measure-memory" in options, socket_path))
