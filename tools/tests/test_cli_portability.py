"""Offline portability contracts for the canonical BigCherry CLI."""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import __main__ as cli  # noqa: E402


class _Reconfigurable(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.errors_seen: str | None = None

    def reconfigure(self, *, errors: str) -> None:
        self.errors_seen = errors


def test_cli_configures_legacy_console_streams_for_lossless_diagnostics(monkeypatch):
    stdout = _Reconfigurable()
    stderr = _Reconfigurable()
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    cli._configure_output()

    assert stdout.errors_seen == "backslashreplace"
    assert stderr.errors_seen == "backslashreplace"


def test_cli_leaves_capture_streams_untouched(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    # pytest/capture streams commonly omit reconfigure; this must remain safe.
    cli._configure_output()
