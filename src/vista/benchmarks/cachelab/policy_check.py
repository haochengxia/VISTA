"""Static admission check for agent-written policies.

This is a guard rail, not the security boundary. The sandbox isolates the
code, and the final score comes from verified simulation, where the policy
process never sees the trace. The check rejects code that reaches outside the
policy API early, with a message the agent can act on.
"""

from __future__ import annotations

import ast

ALLOWED_IMPORTS = frozenset(
    {
        "policy_api",
        "s3fifo",
        "s4fifo",
        "mq",
        "lru",
        "fifo",
        "arc",
        "lirs",
        "twoq",
        "wtinylfu",
        "clock",
        "sieve",
        "vq",
        "collections",
        "heapq",
        "bisect",
        "math",
        "random",
        "itertools",
        "functools",
        "dataclasses",
        "typing",
        "array",
        "statistics",
        "__future__",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "input",
        "memoryview",
        "help",
        "exit",
        "quit",
    }
)
ALLOWED_DUNDERS = frozenset({"__init__", "__name__", "__slots__", "__future__"})
# Public attributes that lead from an allowed module to the interpreter, the
# operating system or the simulator's stack frames (e.g. typing.sys.modules).
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "sys", "os", "modules", "builtins", "importlib", "subprocess", "socket", "io", "pathlib", "shutil",
        "f_locals", "f_globals", "f_back", "f_code", "f_builtins", "gi_frame", "cr_frame", "ag_frame",
        "tb_frame", "tb_next", "system", "popen", "spawn", "fork", "environ",
    }
)
# Attribute access by name: only with a constant, public name.
DYNAMIC_ATTRIBUTE_CALLS = frozenset({"getattr", "setattr", "delattr", "hasattr"})


def check_policy(source: str) -> list[str]:
    """Return problems; an empty list admits the policy for simulation."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]
    problems = []
    defines_policy = False
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"line {line}: import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"line {line}: import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {line}: {node.id!r} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_ATTRIBUTE_CALLS:
            target = node.args[0] if node.args else None
            name = node.args[1] if len(node.args) > 1 else None
            public = isinstance(name, ast.Constant) and isinstance(name.value, str) and not name.value.startswith("_") and name.value not in FORBIDDEN_ATTRIBUTES
            # The policy's own state may be read through a plain variable (S3-FIFO's export does).
            own = target is not None and not _foreign(target) and isinstance(name, ast.Name)
            if not (public or own):
                problems.append(f"line {line}: {node.func.id}() needs a constant, public attribute name (or self/cls and a variable)")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                problems.append(f"line {line}: attribute {node.attr!r} is not allowed")
            if node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
                problems.append(f"line {line}: attribute {node.attr!r} is not allowed")
            if node.attr.startswith("_") and node.attr[1:2] != "_" and _foreign(node.value):
                problems.append(f"line {line}: private attribute {node.attr!r} of another object")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("__") and node.value.endswith("__"):
                problems.append(f"line {line}: dunder string {node.value!r} is not allowed")
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "CachePolicy" for target in targets):
                defines_policy = True
        elif isinstance(node, ast.ClassDef) and node.name == "CachePolicy":
            defines_policy = True
    if not defines_policy:
        problems.append("the module must define CachePolicy (a class, or `CachePolicy = YourClass`)")
    return problems[:20]


def _foreign(value: ast.expr) -> bool:
    """Private attributes are fine on self/cls, not on other objects (e.g. ctx)."""
    return not (isinstance(value, ast.Name) and value.id in {"self", "cls"})
