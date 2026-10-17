"""CLI handler for post-bump hardware acceptance (MEN17): ``bigcherry --engine <name> accept``."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from ..campaign import acceptance
from ..core import engines, paths


def cmd_accept(args: Namespace) -> int:
    """0: every baseline held. 1: a regression, a failed run or a missing measurement. 2: bad declaration or baseline."""
    try:
        layout = engines.load(paths.REPO_ROOT, args.engine)
        passed, results, evidence = acceptance.accept(paths.REPO_ROOT, layout, args.only, Path(args.output))
    except (acceptance.AcceptanceError, engines.EngineError, FileExistsError) as exc:
        print(f"accept: ERROR ({args.engine}): {exc}", file=sys.stderr)
        return 2
    print(acceptance.render(results))
    print(f"accept: {'PASS' if passed else 'FAIL'} ({args.engine}); evidence {evidence}")
    return 0 if passed else 1
