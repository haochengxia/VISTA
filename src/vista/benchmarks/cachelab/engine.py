"""Run policies in a sandbox, cache the decision logs, and load them as arrays.

Candidate policies are untrusted code written by the agent. The default
sandbox runs the stdlib-only simulator in a Docker container with no network,
a read-only root, dropped capabilities, and only the policy, the simulator and
the selected traces mounted. Held-out traces are never mounted during the
agent's task. `LocalSandbox` runs the same simulator as a host subprocess; it
is for trusted code (the baseline) and tests only.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .traces import Trace

SANDBOX_SOURCES = ("simulate.py", "policy_api.py", "policy_server.py", "s3fifo.py", "s4fifo.py", "mq.py", "lru.py", "fifo.py")
REFERENCE_POLICIES = ("s4fifo", "mq", "lru", "fifo")  # importable by policies, recorded as experiments
SANDBOX_DIR = Path(str(files(__package__).joinpath("sandbox")))
BASELINE_SOURCE = (SANDBOX_DIR / "s3fifo.py").read_text(encoding="utf-8")
DEFAULT_IMAGE = "python:3.12-slim"
MAX_POLICY_CHARS = 64 * 1024

ARRAY_TYPES = {
    "outcome": np.uint8,
    "queue": np.uint8,
    "ghost_hit": np.uint8,
    "ev_t": np.uint32,
    "ev_key": np.uint64,
    "ev_kind": np.uint8,
    "ev_a": np.uint8,
    "ev_b": np.uint8,
    "evict_t": np.uint32,
    "evict_key": np.uint64,
    "evict_q": np.uint8,
    "evict_ins": np.uint32,
    "evict_hits": np.uint32,
    "sample_t": np.uint32,
    "sample_sizes": np.int64,
}
HIT, COLD, REMISS, BYPASS = 0, 1, 2, 3
MOVE, GHOST_ADD, GHOST_DROP, GHOST_HIT = 0, 1, 2, 3
NO_QUEUE = 255


def simulator_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in SANDBOX_SOURCES:
        digest.update((SANDBOX_DIR / name).read_bytes())
    return digest.hexdigest()


class Sandbox(Protocol):
    name: str

    def provenance(self) -> dict[str, Any]: ...

    def run(self, jobs: list[dict[str, Any]], *, work: Path, timeout: float) -> list[dict[str, Any]]: ...

    def server_command(self, policy: Path) -> list[str]: ...


def _limits(memory_bytes: int, cpu_seconds: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


class LocalSandbox:
    """Host subprocess with resource limits. Not isolation; trusted code only."""

    name = "local"

    def __init__(self, *, workers: int | None = None, memory_gb: int = 4) -> None:
        self.workers = workers or max(1, (os.cpu_count() or 2) // 2)
        self.memory = memory_gb << 30

    def provenance(self) -> dict[str, Any]:
        return {"sandbox": self.name, "python": sys.version.split()[0], "isolation": "none"}

    def server_command(self, policy: Path) -> list[str]:
        return [
            sys.executable, "-I", str(SANDBOX_DIR / "policy_server.py"), str(policy),
            "--measure-memory", "--socket", "{socket_dir}/policy.sock",
        ]

    def run(self, jobs, *, work, timeout):
        work.mkdir(parents=True, exist_ok=True)
        spec = work / "jobs.json"
        spec.write_text(json.dumps({"workers": self.workers, "jobs": jobs}), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-I", str(SANDBOX_DIR / "simulate.py"), str(spec)],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            preexec_fn=_limits(self.memory * self.workers, int(timeout) + 5),
        )
        return _parse(completed, jobs)


class DockerSandbox:
    """One short-lived container per batch: no network, read-only inputs."""

    name = "docker"

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        workers: int | None = None,
        memory_gb: int = 6,
    ) -> None:
        self.image = image
        self.workers = workers or max(1, min(8, (os.cpu_count() or 2) // 2))
        self.memory_gb = memory_gb
        self._image_id: str | None = None

    @property
    def image_id(self) -> str:
        if self._image_id is None:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image, "--format", "{{json .Id}}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self._image_id = json.loads(result.stdout)
        return self._image_id

    def provenance(self) -> dict[str, Any]:
        return {
            "sandbox": self.name,
            "image": self.image,
            "image_id": self.image_id,
            "network": "none",
            "workers": self.workers,
        }

    def _hardening(self, *, cpus: int, memory_gb: int, pids: int) -> list[str]:
        return [
            "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(pids), "--memory", f"{memory_gb}g",
            "--cpus", str(cpus), "--tmpfs", "/tmp:rw,size=64m",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "PYTHONHASHSEED=0",
            "-v", f"{SANDBOX_DIR}:/lab:ro",
        ]

    def server_command(self, policy: Path) -> list[str]:
        return [
            "docker", "run", *self._hardening(cpus=1, memory_gb=2, pids=32),
            "-v", f"{policy.resolve().parent}:/policy:ro",
            "-v", "{socket_dir}:/sock",
            self.image, "python", "-I", "/lab/policy_server.py", f"/policy/{policy.name}",
            "--measure-memory", "--socket", "/sock/policy.sock",
        ]

    def run(self, jobs, *, work, timeout):
        work.mkdir(parents=True, exist_ok=True)
        mounts: dict[Path, str] = {}

        def inside(host: str, kind: str) -> str:
            path = Path(host).resolve()
            directory = path.parent if kind == "file" else path
            if directory not in mounts:
                mounts[directory] = f"/mnt/{len(mounts)}"
            target = mounts[directory]
            return f"{target}/{path.name}" if kind == "file" else target

        translated = []
        outputs = {}
        for job in jobs:
            out = Path(job["out"]).resolve()
            out.mkdir(parents=True, exist_ok=True)
            item = dict(job)
            item["policy"] = inside(job["policy"], "file")
            item["trace"] = inside(job["trace"], "file")
            item["out"] = f"/out/{len(outputs)}"
            outputs[item["out"]] = out
            translated.append(item)
        staging = work / "out"
        staging.mkdir(exist_ok=True)
        for index in range(len(translated)):
            (staging / str(index)).mkdir(exist_ok=True)
        (work / "jobs.json").write_text(
            json.dumps({"workers": self.workers, "jobs": translated}), encoding="utf-8"
        )
        command = [
            "docker", "run", *self._hardening(cpus=self.workers, memory_gb=self.memory_gb, pids=256),
            "-v", f"{work / 'jobs.json'}:/jobs.json:ro",
            "-v", f"{staging}:/out:rw",
        ]
        for host, target in mounts.items():
            command += ["-v", f"{host}:{target}:ro"]
        command += [self.image, "python", "-I", "/lab/simulate.py", "/jobs.json"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        results = _parse(completed, translated)
        for result, item in zip(results, translated, strict=True):
            source = staging / item["out"].removeprefix("/out/")
            target = outputs[item["out"]]
            for path in source.iterdir():
                shutil.move(str(path), target / path.name)
            result["out"] = str(target)
        return results


def _parse(completed: subprocess.CompletedProcess, jobs) -> list[dict[str, Any]]:
    try:
        results = json.loads(completed.stdout.strip().splitlines()[-1])["results"]
    except (IndexError, KeyError, ValueError):
        message = (completed.stderr or completed.stdout or "no output")[-3000:]
        return [
            {"out": job["out"], "ok": False, "error": f"simulator process failed: {message}"}
            for job in jobs
        ]
    return results


@dataclass
class Run:
    """One policy on one trace at one capacity: the full decision log."""

    directory: Path
    summary: dict[str, Any]

    def __getattr__(self, name: str) -> np.ndarray:
        if name not in ARRAY_TYPES:
            raise AttributeError(name)
        values = np.fromfile(self.directory / f"{name}.bin", dtype=np.dtype(ARRAY_TYPES[name]).newbyteorder("<"))
        self.__dict__[name] = values
        return values

    @property
    def queues(self) -> list[str]:
        return list(self.summary["queues"])

    @property
    def ghosts(self) -> list[str]:
        return list(self.summary["ghosts"])

    @cached_property
    def sizes(self) -> np.ndarray:
        columns = len(self.queues) + len(self.ghosts)
        return self.sample_sizes.reshape(-1, columns)

    @property
    def hit(self) -> np.ndarray:
        return self.outcome == HIT

    @property
    def has_arrays(self) -> bool:
        return (self.directory / "outcome.bin").is_file()

    def drop_arrays(self) -> int:
        """Delete the per-request arrays (the summary stays); a later
        simulate() of this cell runs it again. Returns bytes freed."""
        freed = 0
        for path in self.directory.glob("*.bin"):
            freed += path.stat().st_size
            path.unlink()
        for name in ARRAY_TYPES:
            self.__dict__.pop(name, None)
        self.__dict__.pop("sizes", None)
        return freed


class RunStore:
    """Content-addressed cache of simulator outputs under one lab directory."""

    def __init__(self, root: Path, sandbox: Sandbox, *, timeout: float = 900, verified_timeout: float = 3600) -> None:
        self.root = root.resolve()
        self.sandbox = sandbox
        # Per-cell limits. A fast cell runs the policy in-process, so its limit
        # bounds the policy's own cost. A verified cell pays one socket round
        # trip per miss, which on a loaded machine can exceed the fast limit by
        # far, so it gets its own.
        self.timeout = timeout
        self.verified_timeout = verified_timeout
        self.policies = self.root / "policies"
        self.runs = self.root / "runs"
        for path in (self.policies, self.runs):
            path.mkdir(parents=True, exist_ok=True)
        self.fingerprint = simulator_fingerprint()

    def policy_file(self, source: str) -> tuple[str, Path]:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        directory = self.policies / digest[:16]
        directory.mkdir(exist_ok=True)
        path = directory / "policy.py"
        if not path.is_file():
            path.write_text(source, encoding="utf-8")
        return digest, path

    def key(self, policy_digest: str, trace: Trace, capacity: int, verified: bool, switch: int = 0) -> str:
        mode = "verified" if verified else "fast"
        material = f"{self.fingerprint}:{mode}:{policy_digest}:{trace.sha256}:{capacity}"
        if switch:
            material += f":handover={switch}"
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    def simulate(
        self, source: str, cells: list[tuple[Trace, int]], *, verified: bool = False, handover: float | None = None
    ) -> dict[tuple[str, int], Run | RunError]:
        """Return a Run, or a RunError, for every (trace, capacity) cell.

        Fast runs execute the policy inside the simulator process (in the
        sandbox). Verified runs keep the simulator on the host and serve the
        policy from a separate sandboxed process that never sees the trace.
        With `handover` (a fraction), S3-FIFO serves the first
        int(requests * handover) requests of each trace and the policy takes
        over the live cache there.
        """
        if len(source) > MAX_POLICY_CHARS:
            error = RunError(f"policy source exceeds {MAX_POLICY_CHARS} characters")
            return {(trace.spec.name, capacity): error for trace, capacity in cells}
        digest, path = self.policy_file(source)
        results: dict[tuple[str, int], Run | RunError] = {}
        pending = []
        for trace, capacity in cells:
            switch = handover_index(trace.requests, handover)
            directory = self.runs / self.key(digest, trace, capacity, verified, switch)
            cached = load_cell(directory)
            if isinstance(cached, Run) and not cached.has_arrays:
                # Pruned to save disk: run it again rather than hand out a Run without its log.
                shutil.rmtree(directory, ignore_errors=True)
                cached = None
            if cached is not None:
                results[(trace.spec.name, capacity)] = cached
            else:
                pending.append((trace, capacity, directory, switch))
        if not pending:
            return results
        cell_seconds = self.verified_timeout if verified else self.timeout / 2
        # Cells run `workers` at a time; the batch limit covers every round.
        batch_seconds = cell_seconds * -(-len(pending) // max(1, self.sandbox.workers)) + 120
        jobs = []
        for trace, capacity, directory, switch in pending:
            job = {
                "trace": str(trace.path),
                "capacity": capacity,
                "out": str(directory),
                "max_seconds": cell_seconds,
            }
            if switch:
                job["handover"] = switch
            if verified:
                job["command"] = self.sandbox.server_command(path)
            else:
                job["policy"] = str(path)
            jobs.append(job)
        with tempfile.TemporaryDirectory(prefix="cachelab-", dir=self.root) as work:
            try:
                if verified:
                    outcomes = run_driver(jobs, work=Path(work), timeout=batch_seconds, workers=self.sandbox.workers)
                else:
                    outcomes = self.sandbox.run(jobs, work=Path(work), timeout=batch_seconds)
            except subprocess.TimeoutExpired:
                outcomes = [{"ok": False, "error": "the experiment batch timed out"} for _ in jobs]
        for (trace, capacity, directory, _), outcome in zip(pending, outcomes, strict=True):
            cell = (trace.spec.name, capacity)
            cached = load_cell(directory)
            if isinstance(cached, RunError) and "time limit exceeded" in cached.error:
                # Depends on machine load as much as on the policy: report, don't cache.
                shutil.rmtree(directory, ignore_errors=True)
                results[cell] = cached
            elif cached is not None:
                results[cell] = cached
            else:
                # Batch-level failures (timeouts, a crashed sandbox) are not cached.
                shutil.rmtree(directory, ignore_errors=True)
                results[cell] = RunError(outcome.get("error") or "unknown simulator failure")
        return results


def handover_index(requests: int, handover: float | None) -> int:
    """The request at which a policy takes over from S3-FIFO; 0 for a cold start."""
    if not handover:
        return 0
    if not 0 < handover < 1:
        raise ValueError("handover must be a fraction in (0, 1)")
    return int(requests * handover)


@dataclass(frozen=True)
class RunError:
    error: str
    traceback: str = ""


def load_cell(directory: Path) -> Run | RunError | None:
    if (directory / "summary.json").is_file():
        return Run(directory, json.loads((directory / "summary.json").read_text()))
    if (directory / "error.json").is_file():
        details = json.loads((directory / "error.json").read_text())
        return RunError(details.get("error", "policy failed"), details.get("traceback", ""))
    return None


def run_driver(jobs: list[dict[str, Any]], *, work: Path, timeout: float, workers: int) -> list[dict[str, Any]]:
    """Run the trusted simulator on the host for verified jobs."""
    work.mkdir(parents=True, exist_ok=True)
    spec = work / "jobs.json"
    spec.write_text(json.dumps({"workers": workers, "jobs": jobs}), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-I", str(SANDBOX_DIR / "simulate.py"), str(spec)],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    return _parse(completed, jobs)
