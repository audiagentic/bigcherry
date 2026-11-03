"""CLI handler for the card-free component tests: ``bigcherry --engine <name> component-tests --build <dir>``."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from ..campaign import runtime_bundle
from ..core import paths


def cmd_component_tests(args: Namespace) -> int:
    build = Path(args.build)
    if not (build / "components").is_dir():
        print(f"component-tests: {build} has no components/ build trees (build the lane first)", file=sys.stderr)
        return 2
    failures = runtime_bundle.run_component_tests(args.engine, source_root=paths.REPO_ROOT, build_dir=build)
    if failures:
        print(f"component-tests: FAIL ({args.engine}):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"component-tests: PASS ({args.engine})")
    return 0
