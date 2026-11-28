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
        candidate = None
        state_dir = getattr(args, "candidate_state_dir", None)
        if state_dir is not None:
            from ..release.pin_bump import PinBumpState, PinBumpStop
            try:
                state = PinBumpState.load(Path(state_dir))
            except (OSError, ValueError, KeyError, PinBumpStop) as exc:
                raise acceptance.AcceptanceError(f"invalid candidate state: {exc}") from exc
            if state.lane_build_status != "passed" or not state.lane_build_lane.startswith(layout.name + ":"):
                raise acceptance.AcceptanceError("candidate state does not identify a passed engine lane")
            candidate = acceptance.CandidateRuntime(Path(state.lane_build_dir), state.lane_build_plan_id,
                                                    state.lane_build_binary_sha256, state.to_sha)
        passed, results, evidence = acceptance.accept(paths.REPO_ROOT, layout, args.only, Path(args.output),
                                                      candidate=candidate)
    except (acceptance.AcceptanceError, engines.EngineError, FileExistsError) as exc:
        print(f"accept: ERROR ({args.engine}): {exc}", file=sys.stderr)
        return 2
    print(acceptance.render(results))
    print(f"accept: {'PASS' if passed else 'FAIL'} ({args.engine}); evidence {evidence}")
    return 0 if passed else 1
