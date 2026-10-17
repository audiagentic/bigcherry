"""CLI handler for the RAM-only engine smoke (MEN15): ``bigcherry --engine <name> smoke-cpu --build <dir>``."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from ..campaign import cpu_smoke


def cmd_smoke_cpu(args: Namespace) -> int:
    try:
        result = cpu_smoke.run_for_engine(
            args.engine, Path(args.build), args.model_root, Path(args.out) if args.out else None,
        )
    except cpu_smoke.CpuSmokeError as exc:
        print(f"smoke-cpu: FAIL ({args.engine}):", file=sys.stderr)
        for failure in exc.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    rss = result.get("peak_rss_bytes")
    print(
        f"smoke-cpu: PASS ({args.engine}) in {result['wall_s']}s; startup {result.get('startup_s')}s; "
        f"peak RSS {'%.0f MiB' % (rss / 1048576) if rss else result.get('peak_rss_note')}"
    )
    return 0
