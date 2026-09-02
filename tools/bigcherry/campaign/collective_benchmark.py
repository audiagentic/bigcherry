"""GP08: real end-to-end performance comparison across AllReduce provider
arms, gated on GP07's qualification evidence.

This is the OPTIMIZE side of the VALIDATE/OPTIMIZE split (GP08,
docs/planning/active/gpu-collectives/GP08.md) -- GP07's rccl_qualify.py/
rccl_qualify_campaign.py answer "is this (RCCL revision, topology) safe to
run at all"; this module answers "given a topology already proven safe,
which provider is actually fastest" using real end-to-end inference
(llama-bench pp/tg, or any command producing metrics this module's shared
--metric regex extraction can parse), never all_reduce_perf microbenchmark
latency.

Reuses tools/bigcherry/campaign/benchmark.py's proven arm-capture,
statistics, and paired-comparison machinery (run_arm_capture,
block_bootstrap_effect, schedule_named_arms) rather than a second,
uncommitted statistics runner -- this is the durable form of the manual
A/B/C shell methodology GP03 used to validate patch 1243/0840's real
hardware numbers.

Design (per GP08's own spec, gpt-dev-agent-reviewed):
* An arm that requires RCCL MUST reference a real GP07 qualification
  artifact (a cases.jsonl produced by rccl_qualify_campaign.py) proving a
  PASS for the exact (compatibility revision, topology) this run targets.
  No qualification artifact, or no matching PASS row -> that arm is
  refused outright, never silently skipped or downgraded to a different
  provider. A benchmark result never confers RCCL admissibility itself --
  that authority belongs solely to GP07's artifacts.
* Arms are described by an --arms-config JSON file (binary + env + extra
  CLI args per arm, since a layer-split arm needs a different -sm flag
  than a tensor-split arm, not just a different env var) -- not a fragile
  colon-delimited CLI mini-DSL.
* N-1 baseline-vs-candidate pairs are interleaved round-robin across
  rounds (not one N!-permutation block per round -- impractical past ~4
  arms; 6 arms would need 720 runs just for one balanced block). Each
  pair still uses the proven alternating-order methodology.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bigcherry.campaign.benchmark import (
    block_bootstrap_effect,
    extract_metrics,
    parse_metric_specs,
    run_arm_capture,
)
from bigcherry.profiling.rccl_schema import RcclCompatibilityRevision


class QualificationRequired(RuntimeError):
    """Raised when an arm requires RCCL but has no matching GP07
    qualification PASS -- this is a hard refusal, never a silent
    downgrade to a different provider or a skip."""


def load_arms_config(path: Path) -> list[dict[str, Any]]:
    """Load the arm list: each entry is
    {"name": str, "binary": str, "env": {str: str}, "extra_args": [str],
     "requires_rccl": bool}. `name` must be unique; `binary` and
     `requires_rccl` are required, `env`/`extra_args` default empty."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: arms config must be a non-empty JSON array")
    arms: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for entry in raw:
        if "name" not in entry or "binary" not in entry:
            raise ValueError(f"{path}: each arm needs at least name and binary: {entry!r}")
        name = str(entry["name"])
        if name in seen_names:
            raise ValueError(f"{path}: duplicate arm name {name!r}")
        seen_names.add(name)
        arms.append({
            "name": name,
            "binary": str(entry["binary"]),
            "env": dict(entry.get("env", {})),
            "extra_args": [str(a) for a in entry.get("extra_args", [])],
            "requires_rccl": bool(entry.get("requires_rccl", False)),
        })
    return arms


def check_qualified(cases_jsonl: Path, *, topology_id: str, revision_id: str) -> bool:
    """Real check against a GP07 qualification artifact: is there at
    least one PASS row for this exact (topology_id, compatibility
    revision)? No fuzzy matching, no "close enough" version string --
    the exact revision_id GP07's RcclCompatibilityRevision computed."""
    if not cases_jsonl.is_file():
        return False
    for line in cases_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("topology_id") == topology_id
            and row.get("compatibility_revision_id") == revision_id
            and row.get("classification") == "pass"
        ):
            return True
    return False


def build_arm_schedule(rounds: int, baseline: str, candidates: list[str], seed: int = 0) -> list[tuple[str, str, str]]:
    """N-1 baseline-vs-candidate pairs, interleaved round-robin across
    rounds, alternating order per round to control for thermal/clock
    drift the same way the existing 2-arm pair_modes() does. Returns a
    flat list of (round_index, first_arm, second_arm) triples in real
    execution order."""
    import random as _random
    rng = _random.Random(seed)
    schedule: list[tuple[int, str, str]] = []
    for round_index in range(rounds):
        order = list(candidates)
        rng.shuffle(order)
        for candidate in order:
            first, second = (baseline, candidate) if round_index % 2 == 0 else (candidate, baseline)
            schedule.append((round_index, first, second))
    return schedule


def pairwise_comparison(
    runs: list[dict[str, Any]], baseline: str, candidate: str, metric: str,
    *, lower_is_better: bool = False, seed: int = 0,
) -> dict[str, Any]:
    """Real statistical comparison of one candidate against the baseline,
    reusing benchmark.py's proven block-bootstrap effect estimator. runs
    must carry a "mode" field equal to the arm name and a "pair" field
    grouping one baseline+candidate execution together."""
    return block_bootstrap_effect(
        runs, candidate, baseline, metric, lower_is_better=lower_is_better, seed=seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigcherry collective-benchmark",
        description=(
            "GP08 OPTIMIZE: real end-to-end performance comparison across "
            "AllReduce provider arms, gated on GP07 qualification evidence."
        ),
    )
    parser.add_argument("--arms-config", required=True, help="JSON arm list (see load_arms_config)")
    parser.add_argument("--baseline", required=True, help="arm name every candidate is compared against")
    parser.add_argument("--output", required=True, help="new artifacts/ directory")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--schedule-seed", type=int, default=0)
    parser.add_argument("--settle-seconds", type=float, default=20.0)
    parser.add_argument("--cwd", default=None)
    parser.add_argument(
        "--metric", action="append", default=[], metavar="NAME=REGEX", required=True,
        help="extract a metric from output; regex must have one numeric capture group; may repeat",
    )
    parser.add_argument("--lower-is-better", action="append", default=[], metavar="NAME")
    parser.add_argument(
        "--qualification-jsonl", default=None,
        help="GP07 rccl_qualify_campaign.py cases.jsonl -- required if any arm has requires_rccl",
    )
    parser.add_argument("--topology-id", default=None, help="topology_id to check qualification against")
    parser.add_argument("--rccl-version", default=None)
    parser.add_argument("--rccl-source-revision", default=None)
    parser.add_argument("--library-build-id", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="shared base command, after --")
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("error: supply the shared base command after --", file=sys.stderr)
        return 2
    if args.rounds < 1 or args.settle_seconds < 0:
        print("error: --rounds must be >= 1 and --settle-seconds must be >= 0", file=sys.stderr)
        return 2

    try:
        arms = load_arms_config(Path(args.arms_config))
        patterns = parse_metric_specs(args.metric)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    arm_by_name = {arm["name"]: arm for arm in arms}
    if args.baseline not in arm_by_name:
        print(f"error: --baseline {args.baseline!r} is not in the arms config", file=sys.stderr)
        return 2
    candidates = [name for name in arm_by_name if name != args.baseline]
    if not candidates:
        print("error: need at least one candidate arm besides --baseline", file=sys.stderr)
        return 2

    # GP08's real refusal gate -- checked BEFORE running anything, not
    # discovered mid-run. An arm requiring RCCL with no matching GP07
    # qualification PASS aborts the whole invocation.
    rccl_arms = [arm for arm in arms if arm["requires_rccl"]]
    if rccl_arms:
        if not args.qualification_jsonl or not args.topology_id or not args.rccl_version:
            print(
                "error: arm(s) "
                + ", ".join(a["name"] for a in rccl_arms)
                + " require RCCL but --qualification-jsonl/--topology-id/--rccl-version "
                "were not all supplied -- refusing to run an RCCL-requiring arm without "
                "checkable qualification evidence",
                file=sys.stderr,
            )
            return 2
        try:
            compatibility = RcclCompatibilityRevision(
                rccl_version=args.rccl_version,
                rccl_source_revision=args.rccl_source_revision,
                library_build_id=args.library_build_id,
            )
            revision_id = compatibility.revision_id
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        qualified = check_qualified(
            Path(args.qualification_jsonl), topology_id=args.topology_id, revision_id=revision_id,
        )
        if not qualified:
            print(
                f"error: no GP07 qualification PASS found for topology_id={args.topology_id!r} "
                f"revision_id={revision_id!r} in {args.qualification_jsonl} -- refusing to run "
                + ", ".join(a["name"] for a in rccl_arms)
                + ". This arm's admissibility must be established by GP07 first; a benchmark "
                "run cannot itself confer RCCL qualification.",
                file=sys.stderr,
            )
            return 1

    output = Path(args.output)
    if output.exists():
        print(f"error: output directory already exists: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)

    cwd = Path(args.cwd) if args.cwd else None
    directions = {name: "lower" for name in args.lower_is_better}
    schedule = build_arm_schedule(args.rounds, args.baseline, candidates, seed=args.schedule_seed)

    runs: list[dict[str, Any]] = []
    pair_counter = 0
    total_executions = len(schedule) * 2
    executed = 0
    for round_index, first_name, second_name in schedule:
        pair_counter += 1
        for side_label, arm_name in (("first", first_name), ("second", second_name)):
            arm = arm_by_name[arm_name]
            import os as _os
            env = dict(_os.environ)
            env.update(arm["env"])
            full_command = [arm["binary"], *command[1:], *arm["extra_args"]]
            run = run_arm_capture(
                command=full_command, cwd=cwd, output=output, pair=pair_counter - 1,
                side=arm_name, env=env, patterns=patterns,
            )
            run["arm"] = arm_name
            run["round"] = round_index
            runs.append(run)
            executed += 1
            if run["returncode"] != 0 or "metric_error" in run:
                details = run.get("metric_error", f"command exit {run['returncode']}")
                (output / "run.json").write_text(
                    json.dumps({"command": command, "arms": arms, "runs": runs, "error": details}, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"failed during round {round_index + 1} arm {arm_name}: {details}", file=sys.stderr)
                return 1
            if executed < total_executions and args.settle_seconds:
                import time as _time
                _time.sleep(args.settle_seconds)

    comparisons: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_runs = [
            {"pair": run["pair"], "mode": run["arm"], "metrics": run.get("metrics", {})}
            for run in runs if run["arm"] in (args.baseline, candidate)
        ]
        comparisons[candidate] = {}
        for metric in sorted(patterns):
            try:
                comparisons[candidate][metric] = pairwise_comparison(
                    candidate_runs, args.baseline, candidate, metric,
                    lower_is_better=directions.get(metric) == "lower", seed=args.schedule_seed,
                )
            except ValueError as exc:
                comparisons[candidate][metric] = {"error": str(exc)}

    summary: dict[str, Any] = {
        "command": command,
        "arms": arms,
        "baseline": args.baseline,
        "rounds": args.rounds,
        "settle_seconds": args.settle_seconds,
        "metrics": sorted(patterns),
        "directions": directions,
        "runs": runs,
        "schedule_seed": args.schedule_seed,
        "comparisons": comparisons,
    }
    if rccl_arms:
        summary["qualification"] = {
            "qualification_jsonl": args.qualification_jsonl,
            "topology_id": args.topology_id,
            "compatibility_revision_id": revision_id,
        }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output / 'run.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
