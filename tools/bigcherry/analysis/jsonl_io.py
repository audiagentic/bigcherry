"""Shared strict JSONL reader for measurement/observation files.

Both analysis/report.py and analysis/impact.py parse the same
``*.measurements.jsonl`` wire format and need the exact same tolerance
rule: a genuinely TRUNCATED final line (no trailing newline -- the real
signature of a run killed mid-flush) is recoverable; anything else
malformed is not. Shared here after the two independent implementations
drifted (report.py fixed round 1, then round 2 found impact.py's
load_results still had the ORIGINAL bug -- and even report.py's round-1
fix had its own subtler bug, using ``splitlines()`` instead of
``splitlines(keepends=True)``, which made a fully-written but corrupt
final record indistinguishable from a real truncation). One
implementation, one place to get it right (gpt-dev-agent review round 2,
2026-08-31).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class JsonlReadError(ValueError):
    pass


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON-Lines file into its raw rows (every ``kind``), applying
    the torn-vs-corrupt tolerance rule once.

    Tolerates only a genuinely TRUNCATED final line (no trailing newline).
    Any OTHER malformed line -- interior corruption, or a newline-
    terminated but corrupt final record -- raises JsonlReadError instead
    of being silently dropped. A record's silent disappearance is worse
    than a loud failure: this data feeds statistical claims (bootstrap
    CIs, promotion decisions), and "valid A, corrupt B, valid C" reporting
    as just "A, C" with exit 0 gives false confidence.
    """
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    non_blank_indices = [i for i, line in enumerate(raw_lines) if line.strip()]
    last_non_blank = non_blank_indices[-1] if non_blank_indices else -1
    last_line_torn = bool(raw_lines) and not raw_lines[-1].endswith(("\n", "\r"))

    rows: list[dict[str, Any]] = []
    for index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            if index == last_non_blank and last_line_torn:
                print(
                    f"warning: {Path(path).name} line {index + 1} is truncated "
                    "(no trailing newline); ignoring it (final line)",
                    file=sys.stderr,
                )
                break
            raise JsonlReadError(
                f"{Path(path).name} line {index + 1} is malformed JSON and is "
                f"NOT a truncated final line (interior corruption, or a "
                f"fully-written but corrupt final record) -- refusing to "
                f"silently drop it: {exc}"
            ) from exc
        rows.append(row)
    return rows


def read_result_records(path: Path) -> list[dict[str, Any]]:
    """``read_rows`` filtered to ``kind == "result"`` -- the measurements
    JSONL shape both report.py and impact.py's load_results consume."""
    return [row for row in read_rows(path) if row.get("kind") == "result"]
