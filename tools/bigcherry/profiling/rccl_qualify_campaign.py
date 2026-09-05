"""HI142/RQ08/GP07: reusable RCCL qualification campaign driver.

Diagnostic tooling only -- drives rccl_qualify.run_case (crash-isolated,
one case per process) against a real all_reduce_perf binary. Does not touch
BigCherry's production reduction-provider selection or patch 1225 in any
way. See docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md P2.4-P2.5,
HI142, and GP07 (docs/planning/completed/gpu-collectives/GP07.md) for the
governing procedure.

GP07 rewrite (2026-09-02, gpt-dev-agent-reviewed): the prior version of
this driver was hardcoded to one topology set ({0,2}/{1,2}/{0,1,2}), one
fixed 512KiB size, no repetitions, and no RCCL-compatibility identity --
GP06's real 3-way RCCL-version comparison (RCCL 1.0.70204 vs 2.30.4,
finding a real regression on the {0,2}/{1,2} class) had to be run through
an uncommitted, ad-hoc variant of this script instead. This rewrite makes
the checked-in driver capable of reproducing that exact matrix: a
configurable topology set (default includes a {0,1} positive control and
a {0,3} device-3 negative control, not just the RCCL-viable pairs),
configurable element counts/algorithms/protocols/repetitions, a required
RCCLCompatibilityRevision, and the runbook's post-fault control-recheck
rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bigcherry.profiling import rccl_qualify as rq
from bigcherry.profiling import rccl_schema as rs

DEFAULT_ALGORITHMS = ("Ring", "Tree")
DEFAULT_PROTOCOLS = ("Simple", "LL", "LL128")

# GP06's real element-count matrix: HI18's dominant real production
# reduction shape (30720 elements, decode-sized) and a real large
# stress shape (2621440 elements, prefill-sized). f32 byte counts:
# 30720*4=122880, 2621440*4=10485760.
DEFAULT_ELEMENT_COUNTS = (30720, 2621440)

DEFAULT_REPETITIONS = 20  # runbook P1.12's fresh-process gate

# GP06's default matrix: {0,1} homogeneous positive control (must always
# pass -- if it doesn't, environment/build/runtime is invalid, stop per
# the runbook's decision tree), {0,2}/{1,2} the HI138-confirmed-viable
# heterogeneous pairs, {0,3} the device-3 negative control (must always
# fail -- HI138's permanent hardware limitation). {0,1,2} is deliberately
# NOT in this default set -- it belongs to GP01's Phase 2 extension once
# the two-rank gate is stable on the pinned compatibility revision, not
# folded into every campaign run.
DEFAULT_TOPOLOGIES: tuple[tuple[rq.RcclTopology, tuple[int, ...]], ...] = (
    (rq.RcclTopology(topology_id="xtx_xtx", device_arches=("gfx1100", "gfx1100")), (0, 1)),
    (rq.RcclTopology(topology_id="xtx0_r9700", device_arches=("gfx1100", "gfx1201")), (0, 2)),
    (rq.RcclTopology(topology_id="xtx1_r9700", device_arches=("gfx1100", "gfx1201")), (1, 2)),
    (rq.RcclTopology(topology_id="xtx0_6900xt", device_arches=("gfx1100", "gfx1030")), (0, 3)),
)

CONTROL_TOPOLOGY = DEFAULT_TOPOLOGIES[0]

# Classifications that trigger the runbook's post-fault safety rule: after
# any of these, re-run the homogeneous control before trusting further
# results (runbook safety invariant 5). DEVICE_LOST additionally stops the
# whole campaign if the control itself cannot be restored -- it is not
# merely "this one case crashed," it is evidence the GPU/driver state may
# be compromised for every subsequent case.
_POST_FAULT_TRIGGERS = frozenset((rq.GPU_FAULT, rq.SIGNAL, rq.TIMEOUT, rq.DEVICE_LOST))


class CampaignAborted(RuntimeError):
    """Raised when the post-fault control recheck itself fails -- the
    runbook requires stopping rather than continuing to trust further
    results once the known-good control can no longer be reproduced."""


def _run_one_case(
    *, topology: rq.RcclTopology, visible_devices: tuple[int, ...],
    element_count: int, algorithm: str, protocol: str,
    binary: str, compatibility: rs.RcclCompatibilityRevision,
    output_dir: Path, cases_jsonl: Path, attempt: int,
) -> rq.RcclCaseResult:
    case = rq.RcclCase(
        topology=topology, element_count=element_count, dtype="float",
        algorithm=algorithm, protocol=protocol,
    )
    result = rq.run_case(
        case, binary=binary, visible_devices=visible_devices,
        output_dir=output_dir, attempt=attempt, compatibility=compatibility,
    )
    rq.append_result(result, cases_jsonl)
    print(
        f"{topology.topology_id:16s} rep{attempt:<3d} {algorithm:5s} {protocol:7s} "
        f"n={element_count:<8d} -> {result.classification:16s} "
        f"correct={result.correct} plan={result.plan_verification}"
    )
    return result


def _recheck_control(
    *, binary: str, compatibility: rs.RcclCompatibilityRevision,
    output_dir: Path, cases_jsonl: Path, attempt: int,
) -> rq.RcclCaseResult:
    """Runbook safety invariant 5: after a hard failure, re-run a known-
    good homogeneous control before trusting subsequent evidence."""
    control_topology, control_devices = CONTROL_TOPOLOGY
    print(f"-- post-fault control recheck (attempt {attempt}) --")
    return _run_one_case(
        topology=control_topology, visible_devices=control_devices,
        element_count=DEFAULT_ELEMENT_COUNTS[0], algorithm=DEFAULT_ALGORITHMS[0],
        protocol=DEFAULT_PROTOCOLS[0], binary=binary, compatibility=compatibility,
        output_dir=output_dir, cases_jsonl=cases_jsonl, attempt=attempt,
    )


def run_matrix(
    *, topologies: tuple[tuple[rq.RcclTopology, tuple[int, ...]], ...],
    element_counts: tuple[int, ...], algorithms: tuple[str, ...],
    protocols: tuple[str, ...], repetitions: int,
    binary: str, compatibility: rs.RcclCompatibilityRevision,
    output_dir: Path,
) -> list[rq.RcclCaseResult]:
    """Run the full topology x element_count x algorithm x protocol x
    repetitions matrix, implementing the runbook's post-fault control-
    recheck rule between cases.

    Raises CampaignAborted if a post-fault control recheck itself fails --
    per the runbook, do not continue trusting further results once the
    known-good control can no longer be reproduced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_jsonl = output_dir / "cases.jsonl"
    results: list[rq.RcclCaseResult] = []
    recheck_counter = 0

    for topology, visible_devices in topologies:
        for element_count in element_counts:
            for algorithm in algorithms:
                for protocol in protocols:
                    for attempt in range(1, repetitions + 1):
                        result = _run_one_case(
                            topology=topology, visible_devices=visible_devices,
                            element_count=element_count, algorithm=algorithm,
                            protocol=protocol, binary=binary,
                            compatibility=compatibility, output_dir=output_dir,
                            cases_jsonl=cases_jsonl, attempt=attempt,
                        )
                        results.append(result)

                        if result.classification in _POST_FAULT_TRIGGERS:
                            recheck_counter += 1
                            control_result = _recheck_control(
                                binary=binary, compatibility=compatibility,
                                output_dir=output_dir, cases_jsonl=cases_jsonl,
                                attempt=1000 + recheck_counter,
                            )
                            results.append(control_result)
                            if control_result.classification != rq.PASS:
                                raise CampaignAborted(
                                    f"post-fault control recheck failed "
                                    f"({control_result.classification}) after "
                                    f"{topology.topology_id}/{algorithm}/{protocol} "
                                    f"n={element_count} attempt={attempt} -- stopping "
                                    "campaign per runbook safety invariant 5; GPU/"
                                    "driver state may be compromised"
                                )
                            if result.classification == rq.DEVICE_LOST:
                                # Runbook: DEVICE_LOST is materially worse than a
                                # per-case SIGNAL/GPU_FAULT even when the control
                                # itself still passes -- stop this campaign run
                                # rather than continue exercising a GPU that just
                                # reported loss.
                                raise CampaignAborted(
                                    f"DEVICE_LOST on {topology.topology_id}/"
                                    f"{algorithm}/{protocol} n={element_count} "
                                    f"attempt={attempt} -- stopping campaign even "
                                    "though the control recheck passed"
                                )

    return results


def summarize(results: list[rq.RcclCaseResult]) -> dict[str, dict[str, int]]:
    """Per-topology classification counts -- the shape a human or GP01/
    GP06 needs to read off "did {0,2} pass 20/20" at a glance."""
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        by_class = summary.setdefault(result.topology_id, {})
        by_class[result.classification] = by_class.get(result.classification, 0) + 1
    return summary


def _parse_compatibility(args: argparse.Namespace) -> rs.RcclCompatibilityRevision:
    if not args.rccl_source_revision and not args.library_build_id:
        raise SystemExit(
            "--rccl-source-revision or --library-build-id is required -- a "
            "bare --rccl-version is not sufficient durable identity (GP06 "
            "found two installs reporting the same version string regress "
            "differently). Record the exact source commit or librccl.so "
            "build-id before running a qualification campaign."
        )
    return rs.RcclCompatibilityRevision(
        rccl_version=args.rccl_version,
        rccl_source_revision=args.rccl_source_revision,
        library_build_id=args.library_build_id,
        rocm_install_label=args.rocm_install_label,
        build_config=args.build_config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="path to all_reduce_perf")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--rccl-version", required=True, help='e.g. "2.28.3"')
    parser.add_argument("--rccl-source-revision", default=None, help="exact git commit/tag")
    parser.add_argument("--library-build-id", default=None, help="SHA256/build-id of librccl.so")
    parser.add_argument("--rocm-install-label", default=None, help='e.g. "vendor/rocm/7.2.4"')
    parser.add_argument("--build-config", default=None)

    parser.add_argument(
        "--element-count", type=int, action="append", dest="element_counts",
        default=None, help="repeatable; default matches GP06's matrix",
    )
    parser.add_argument(
        "--algorithm", action="append", dest="algorithms", default=None,
        help="repeatable; default Ring,Tree",
    )
    parser.add_argument(
        "--protocol", action="append", dest="protocols", default=None,
        help="repeatable; default Simple,LL,LL128",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--topology", action="append", dest="topologies", default=None,
        help=(
            "repeatable, format 'topology_id:arch1,arch2:dev1,dev2' -- "
            "e.g. 'xtx0_r9700:gfx1100,gfx1201:0,2'. Default uses GP06's "
            "matrix ({0,1} control, {0,2}, {1,2}, {0,3} negative control)."
        ),
    )

    args = parser.parse_args(argv)

    compatibility = _parse_compatibility(args)
    element_counts = tuple(args.element_counts) if args.element_counts else DEFAULT_ELEMENT_COUNTS
    algorithms = tuple(args.algorithms) if args.algorithms else DEFAULT_ALGORITHMS
    protocols = tuple(args.protocols) if args.protocols else DEFAULT_PROTOCOLS

    if args.topologies:
        topologies = []
        for spec in args.topologies:
            topology_id, arches, devices = spec.split(":")
            topologies.append((
                rq.RcclTopology(topology_id=topology_id, device_arches=tuple(arches.split(","))),
                tuple(int(d) for d in devices.split(",")),
            ))
        topologies = tuple(topologies)
    else:
        topologies = DEFAULT_TOPOLOGIES

    try:
        results = run_matrix(
            topologies=topologies, element_counts=element_counts,
            algorithms=algorithms, protocols=protocols,
            repetitions=args.repetitions, binary=args.binary,
            compatibility=compatibility, output_dir=Path(args.output_dir),
        )
    except CampaignAborted as exc:
        print(f"CAMPAIGN ABORTED: {exc}", file=sys.stderr)
        return 2

    summary = summarize(results)
    print("\n=== summary ===")
    for topology_id, counts in summary.items():
        print(f"{topology_id:16s} {counts}")

    control_topology_id = CONTROL_TOPOLOGY[0].topology_id
    control_ok = summary.get(control_topology_id, {}).get(rq.PASS, 0) > 0
    if not control_ok:
        print(
            f"FAIL: control topology {control_topology_id!r} never passed -- "
            "environment/build/runtime is invalid per the runbook's decision "
            "tree; do not trust any other result in this run.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

