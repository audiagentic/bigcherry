#!/usr/bin/env python3
"""Summarise slow HIP compiler invocations from Ninja's own timing logs.

This reports per-command elapsed time, not wall time: commands may overlap.
It does not execute compilers or require a GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def hip_durations(root: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    if not root.exists():
        return rows
    for log in sorted(root.rglob(".ninja_log")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) < 5 or not parts[3].endswith(".hip.o"):
                continue
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if start < 0 or end < start:
                continue
            component = log.parent.relative_to(root).as_posix()
            rows.append((end - start, f"{component}/{parts[3]}"))
    return sorted(rows, reverse=True)


def render(rows: list[tuple[int, str]], *, limit: int = 20) -> str:
    lines = ["Radiance HIP compile profile (Ninja per-command elapsed time; overlapping jobs):"]
    if not rows:
        return "\n".join(lines + ["No HIP compilations recorded."])
    total_s = sum(ms for ms, _ in rows) / 1000
    lines.append(f"{len(rows)} HIP commands; sum of elapsed durations (not wall time): {total_s:.1f}s")
    for duration_ms, output in rows[:limit]:
        lines.append(f"  {duration_ms / 1000:8.1f}s  {output}")
    return "\n".join(lines)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("work/builds")
    message = render(hip_durations(root))
    print(message)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n### Radiance HIP compile profile\n\n```text\n")
            f.write(message.replace("```", "'''"))
            f.write("\n```\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
