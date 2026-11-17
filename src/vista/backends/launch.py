"""Shared command-line selection of a transport.

`--backend` names what the benchmark CLIs launch: `codex` or `claude`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vista.backends.preflight import check_transport

BACKENDS = ("codex", "claude")
TRANSPORTS = {"codex": "codex", "claude": "claude"}


def transport_for(backend: str) -> str:
    try:
        return TRANSPORTS[backend]
    except KeyError:
        raise ValueError(f"Unknown backend {backend!r}") from None


def add_backend_arguments(
    parser: argparse.ArgumentParser, *, backends: tuple[str, ...] = BACKENDS
) -> None:
    # Imported here so naming an argument does not pull in a transport.
    from vista.backends.codex.runner import default_codex_auth_file

    parser.add_argument("--backend", choices=backends)
    parser.add_argument(
        "--auth-file",
        type=Path,
        # All benchmarks default to the file written by `codex login`.
        default=default_codex_auth_file(),
        help="Codex auth.json (default: ~/.codex/auth.json)",
    )
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--task-timeout", type=int)
    if "claude" not in backends:
        return
    parser.add_argument("--claude-token-env", default="CLAUDE_CODE_OAUTH_TOKEN")
    parser.add_argument(
        "--claude-base-url",
        help="claude: serve the model from this Anthropic-compatible endpoint "
        "(Claude Code appends /v1/messages); the credential is read from "
        "--claude-token-env and every model alias is pinned to --model",
    )
    parser.add_argument(
        "--claude-context-window",
        type=int,
        default=200_000,
        help="claude with --claude-base-url: the served model's context window "
        "in tokens (default: 200000)",
    )
    parser.add_argument(
        "--claude-max-retries",
        type=int,
        default=15,
        help="claude with --claude-base-url: Claude Code's retries of one failed "
        "API request before the session ends (default: 15, the most Claude Code "
        "2.1 accepts; its own default is 10)",
    )
    parser.add_argument(
        "--claude-tool-idle-timeout",
        type=int,
        help="claude: seconds a tool call may run without output before Claude Code "
        "ends it (Claude Code's default: 1800 for VISTA's stdio tool server)",
    )
    parser.add_argument("--expected-model-id")
    parser.add_argument(
        "--compact-window",
        type=int,
        help="claude: auto-compaction window in tokens (default: provider or "
        "transport setting)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="claude: CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    )


def validate_backend_arguments(
    args: argparse.Namespace, error: Callable[[str], Any]
) -> None:
    for name in ("compact_window", "max_output_tokens", "claude_context_window", "claude_max_retries", "claude_tool_idle_timeout"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            error(f"--{name.replace('_', '-')} must be positive")


def prepare_backend(
    args: argparse.Namespace,
    error: Callable[[str], Any],
    *,
    check: Callable[[str, Path | None], tuple[Path, dict[str, Any]]] = check_transport,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Validate credentials and the transport, then return (factory, provenance).

    Credential lookups happen before the transport check so a missing token is
    reported without touching Docker. Nothing here runs a model.
    """
    backend = args.backend
    transport = transport_for(backend)
    if backend == "codex":
        if args.auth_file is None or not args.auth_file.is_file():
            error("Codex run requires an existing --auth-file")
        from vista.backends.codex import create_backend

        binary, provenance = check(transport, args.binary)

        def factory(runtime, directory):
            return create_backend(
                runtime,
                directory=directory,
                auth_file=args.auth_file,
                binary=binary,
                ca_bundle=args.ca_bundle,
                task_timeout=args.task_timeout,
                image=provenance["container_image_id"],
            )

        return factory, provenance
    token = os.environ.get(args.claude_token_env)
    if not token:
        error(
            f"Claude run requires the {args.claude_token_env} environment variable"
        )
    from vista.backends.claude import create_backend
    from vista.backends.claude.provider import ANTHROPIC_PROVIDER, ClaudeCodeProvider

    provider, base_url = ANTHROPIC_PROVIDER, getattr(args, "claude_base_url", None)
    if base_url:
        if not base_url.startswith("https://"):
            error("--claude-base-url must be an https:// URL")
        provider = ClaudeCodeProvider(
            name="anthropic-compatible",
            credential_env=args.claude_token_env,
            endpoint_protocol="anthropic-compatible",
            default_model="(set by --model)",
            default_context_window=args.claude_context_window,
            default_base_url=base_url,
            # Compact relative to the served model's window, not Claude's 1M one.
            auto_compact_window=args.claude_context_window,
            # Shared gateways fail in bursts (502/524); riding one out inside the
            # conversation is far cheaper than a fresh session without its context.
            max_retries=args.claude_max_retries,
        )

    binary, provenance = check(transport, args.binary)
    idle = getattr(args, "claude_tool_idle_timeout", None)
    if idle is not None:
        provider = dataclasses.replace(provider, mcp_tool_idle_timeout_ms=idle * 1000)
    if base_url:
        provenance = {**provenance, "provider": provider.name, "provider_base_url": base_url}

    def factory(runtime, directory):
        return create_backend(
            runtime,
            directory=directory,
            oauth_token=token if provider.native else None,
            provider=provider,
            provider_credential=None if provider.native else token,
            provider_base_url=base_url,
            binary=binary,
            ca_bundle=args.ca_bundle,
            expected_model_id=args.expected_model_id,
            task_timeout=args.task_timeout,
            image=provenance["container_image_id"],
            auto_compact_window=args.compact_window,
            max_output_tokens=args.max_output_tokens,
        )

    return factory, provenance
