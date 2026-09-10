"""Standard console launch/progress/completion telemetry for project-owned CLI
harnesses (RQW01).

Decorates existing execution seams -- experiment.bundle.run_managed() being
the first integration point -- rather than inventing a second subprocess
launch abstraction. Emits ONLY to stderr, in quiet (default) or verbose mode,
so machine-readable stdout produced by the harness itself is never touched.

Redaction: command argv is summarized as name + argument COUNT + a short
digest of the full argv, never the raw argument values (which may carry
prompt text, model paths, or credentials passed via CLI flags). Environment
is summarized as key COUNT only. Callers that need to show real values
(build tooling with no secret risk) may pass `show_argv=True` explicitly --
off by default.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

_VERBOSE_ENV = "BC_TELEMETRY_VERBOSE"


def _argv_digest(argv: list[str]) -> str:
    payload = "\x1f".join(argv).encode("utf-8", "surrogateescape")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _emit(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def verbose_enabled() -> bool:
    return os.environ.get(_VERBOSE_ENV, "").strip() not in ("", "0", "false", "False")


@dataclass(frozen=True)
class LaunchSummary:
    command_name: str
    argv_count: int
    argv_digest: str
    env_key_count: int


def summarize_launch(command: list[str], environment: dict[str, str] | None) -> LaunchSummary:
    name = command[0] if command else "<empty>"
    return LaunchSummary(
        command_name=name,
        argv_count=len(command),
        argv_digest=_argv_digest(command),
        env_key_count=len(environment or {}),
    )


@contextmanager
def console_telemetry(
    *, session_id: str, command: list[str], environment: dict[str, str] | None = None,
    show_argv: bool = False,
) -> Iterator[dict[str, object]]:
    """Emit launch/progress/completion telemetry to stderr around one managed
    process launch. Yields a mutable dict the caller may populate with a
    "returncode" and "output_summary" before the block exits; both are
    reported on the completion line. Never raises on the wrapped body's
    exceptions -- always emits a completion line first, then re-raises."""
    summary = summarize_launch(command, environment)
    started_monotonic = time.monotonic()
    verbose = verbose_enabled()

    argv_repr = " ".join(command) if show_argv else f"<{summary.argv_count} args, digest={summary.argv_digest}>"
    _emit(f"[bc-telemetry] launch session={session_id} command={summary.command_name} argv={argv_repr} env_keys={summary.env_key_count}")

    state: dict[str, object] = {"returncode": None, "output_summary": None}
    try:
        if verbose:
            _emit(f"[bc-telemetry] progress session={session_id} state=running")
        yield state
    finally:
        elapsed = time.monotonic() - started_monotonic
        returncode = state.get("returncode")
        output_summary = state.get("output_summary") or ""
        _emit(
            f"[bc-telemetry] completion session={session_id} command={summary.command_name} "
            f"returncode={returncode} elapsed_s={elapsed:.3f} output_summary={output_summary}"
        )
