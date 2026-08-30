"""HI142/RQ08: real six-cell (Ring/Tree x Simple/LL/LL128) RCCL qualification
campaign driver for the device-3-excluded topology subset.

Diagnostic tooling only -- drives rccl_qualify.run_case (crash-isolated,
one case per process) against a real all_reduce_perf binary. Does not touch
BigCherry's production reduction-provider selection or patch 1225 in any
way. See docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md P2.4-P2.5 and
HI142 for the governing procedure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bigcherry.profiling import rccl_qualify as rq

ALGORITHMS = ("Ring", "Tree")
PROTOCOLS = ("Simple", "LL", "LL128")

# HI18's real recorded production reduction signature (element_count=8192,
# slice_shape=[4096,2,1,1]) -- 8192 * 4 bytes (f32) = 32768 bytes. Used as
# the RQ00-anchor size per gpt's review; 512KiB retained as the RQ07/RQ08
# qualification-standard size since it's what the P1 control/reproducer
# runs already used.
QUALIFICATION_SIZE = 512 * 1024
PRODUCTION_SIZE_ELEMENTS = 8192


def run_six_cell(
    topology: rq.RcclTopology, *, binary: str, visible_devices: tuple[int, ...],
    output_dir: Path, cases_jsonl: Path,
) -> list[rq.RcclCaseResult]:
    results = []
    for algorithm in ALGORITHMS:
        for protocol in PROTOCOLS:
            case = rq.RcclCase(
                topology=topology, element_count=QUALIFICATION_SIZE // 4,
                dtype="float", algorithm=algorithm, protocol=protocol,
            )
            result = rq.run_case(
                case, binary=binary, visible_devices=visible_devices,
                output_dir=output_dir,
            )
            rq.append_result(result, cases_jsonl)
            results.append(result)
            print(
                f"{topology.topology_id:20s} {algorithm:5s} {protocol:7s} "
                f"-> {result.classification:16s} correct={result.correct}"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cases_jsonl = output_dir / "cases.jsonl"

    topologies = [
        (rq.RcclTopology(topology_id="xtx_r9700", device_arches=("gfx1100", "gfx1201")), (0, 2)),
        (rq.RcclTopology(topology_id="xtx1_r9700", device_arches=("gfx1100", "gfx1201")), (1, 2)),
        (rq.RcclTopology(topology_id="xtx_xtx_r9700", device_arches=("gfx1100", "gfx1100", "gfx1201")), (0, 1, 2)),
    ]

    all_pass = True
    for topology, visible_devices in topologies:
        results = run_six_cell(
            topology, binary=args.binary, visible_devices=visible_devices,
            output_dir=output_dir, cases_jsonl=cases_jsonl,
        )
        if any(r.classification != rq.PASS for r in results):
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
