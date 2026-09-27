#!/usr/bin/env python3
"""Write rebuildable BigCherry job status projections atomically."""
from __future__ import annotations

import argparse
from pathlib import Path

from bigcherry.cli.jobs import build_service
from bigcherry.jobs.status import (
    build_snapshot,
    canonical_json,
    render_markdown,
    render_prometheus,
)
from bigcherry.tuning.journal import atomic_write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="snapshot-bigcherry-jobs")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    service, registry = build_service()
    output = (args.output_root or service.store.root / "status").resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot(service, executors=registry.describe())
    atomic_write(output / "status.json", canonical_json(snapshot))
    atomic_write(
        output / "status.md", render_markdown(snapshot).encode("utf-8")
    )
    atomic_write(
        output / "bigcherry.prom", render_prometheus(snapshot).encode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
