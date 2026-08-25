"""Offline reporting and prototyping harness for ranking policies (HI50).

Two jobs, kept separate because they answer different questions:

  * **Report mode** (default) summarizes the `ranking_decisions[]` the
    tuner already recorded live for every installed policy -- per-policy
    agreement with the confirmed/provisional outcome, and a disagreement
    list. No recomputation: this is a read of what actually ran.
  * **Prototype mode** (`--policy-module`) loads a policy that is *not*
    compiled into the C++ tuner yet, runs it over the same archived
    finalist data, and reports how it would have compared -- the offline
    simulator HI44 and HI45 both require before touching live hardware.

Input is a raw `*.measurements.jsonl` file today. HI47's experiment bundle
does not yet index/hash that artifact (see HI47.md's implementation status
note), so there is no validated-bundle path to prefer over it yet; once one
exists, this module should default to it and treat a bare JSONL path as the
non-decision-grade diagnostic fallback.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .tuning import ranking as rp


class RankReplayError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RankReplayError(f"malformed measurements line {number}") from exc
        kind = row.get("kind")
        if kind == "header":
            if header is not None:
                raise RankReplayError("duplicate header")
            header = row
        elif kind == "result":
            results.append(row)
        # Other kinds (e.g. none today) are ignored rather than rejected,
        # matching this file family's other readers.
    if header is None:
        raise RankReplayError(f"{path}: no header row")
    return header, results


def _load_prototype_policy(spec: str):
    """Import a --policy-module argument: a registry name
    (ranking_policy.PROTOTYPE_POLICIES), a dotted module path, or a
    filesystem .py path. Validates the module's SPEC before returning it.
    """
    target = rp.PROTOTYPE_POLICIES.get(spec, spec)
    if target.endswith(".py"):
        module_path = Path(target)
        if not module_path.is_file():
            raise RankReplayError(f"{target}: no such policy module file")
        module_spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if module_spec is None or module_spec.loader is None:
            raise RankReplayError(f"{target}: cannot load as a Python module")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise RankReplayError(f"{target}: cannot import policy module") from exc
    if not hasattr(module, "SPEC") or not hasattr(module, "rank"):
        raise RankReplayError(f"{target}: not a valid prototype policy (needs SPEC and rank())")
    rp.validate_policy_spec(module.SPEC)
    return module


def _prototype_decision(module, result: dict[str, Any], config: dict[str, Any]) -> rp.PolicyDecision | None:
    candidates = rp.finalist_metrics(result)
    native = next((c for c in candidates if c.is_native), None)
    if native is None or not candidates:
        return None
    return module.rank(native, candidates, config)


def _actual_outcome(result: dict[str, Any]) -> str:
    """The ranking-stage decision to compare a policy's predicted_winner
    against. `provisional_winner` (HI50+) is the correct, confirmation-
    independent target; a pre-HI50 archive has no such field, so this falls
    back to the final `winner` -- a degraded comparison (it can already
    reflect a confirmation rejection or routing transform), reported as such
    by the caller rather than silently treated as equivalent.
    """
    return result.get("provisional_winner") or result.get("winner", "")


def build_report(
    header: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    prototype_module=None,
) -> dict[str, Any]:
    config = header.get("config", {})
    degraded = not any("provisional_winner" in r for r in results)
    per_policy: dict[str, dict[str, Any]] = {}
    disagreements: list[dict[str, Any]] = []

    def _record(policy_name: str, policy_version: int, dispatch: str,
                predicted: str, actual: str) -> None:
        bucket = per_policy.setdefault(
            policy_name, {"policy_version": policy_version, "agree": 0, "disagree": 0, "total": 0}
        )
        bucket["total"] += 1
        if predicted == actual:
            bucket["agree"] += 1
        else:
            bucket["disagree"] += 1
            disagreements.append({
                "dispatch": dispatch, "policy_name": policy_name,
                "predicted_winner": predicted, "actual": actual,
            })

    for result in results:
        dispatch = result.get("dispatch", "")
        actual = _actual_outcome(result)
        for decision in rp.parse_ranking_decisions(result):
            _record(decision.policy_name, decision.policy_version, dispatch,
                     decision.predicted_winner, actual)
        if prototype_module is not None:
            decision = _prototype_decision(prototype_module, result, config)
            if decision is not None:
                _record(f"{decision.policy_name} (prototype)", decision.policy_version,
                         dispatch, decision.predicted_winner, actual)

    for bucket in per_policy.values():
        bucket["agreement_rate"] = (
            bucket["agree"] / bucket["total"] if bucket["total"] else None
        )

    return {
        "measurements": len(results),
        "degraded_comparison": degraded,
        "policies": per_policy,
        "disagreements": disagreements,
    }


def verify_parity(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Assert the recorded *production* policy's predicted_winner matches
    the result's own provisional_winner on every row where a decision was
    actually attempted (excludes unresolved-canary/native-not-beaten rows,
    where ranking never ran a real challenger through qualification, and
    confirmation-rejected rows, since provisional_winner and the production
    policy's predicted_winner both reflect the pre-confirmation pick and
    should agree there too -- see HI50.md).
    """
    matched = 0
    mismatched: list[dict[str, Any]] = []
    excluded = 0
    for result in results:
        provisional = result.get("provisional_winner")
        if provisional is None:
            excluded += 1
            continue
        decisions = rp.parse_ranking_decisions(result)
        production = next((d for d in decisions if d.is_production), None)
        if production is None:
            excluded += 1
            continue
        if production.predicted_winner == provisional:
            matched += 1
        else:
            mismatched.append({
                "dispatch": result.get("dispatch", ""),
                "policy_name": production.policy_name,
                "predicted_winner": production.predicted_winner,
                "provisional_winner": provisional,
            })
    return {
        "matched": matched, "mismatched": len(mismatched),
        "excluded": excluded, "mismatches": mismatched,
    }


def dispatch_detail(results: list[dict[str, Any]], dispatch: str) -> dict[str, Any]:
    result = next((r for r in results if r.get("dispatch") == dispatch), None)
    if result is None:
        raise RankReplayError(f"{dispatch}: no such dispatch in this measurements file")
    decisions = rp.parse_ranking_decisions(result)
    return {
        "dispatch": dispatch,
        "native": result.get("native", ""),
        "winner": result.get("winner", ""),
        "provisional_winner": result.get("provisional_winner"),
        "promotion_status": result.get("promotion_status"),
        "policies": [
            {
                "policy_name": d.policy_name,
                "policy_version": d.policy_version,
                "is_production": d.is_production,
                "predicted_winner": d.predicted_winner,
                "candidates": [
                    {
                        "name": c.name, "effective_us": c.effective_us,
                        "verdict": c.verdict, "rejection_reason": c.rejection_reason,
                    }
                    for c in d.candidates
                ],
            }
            for d in decisions
        ],
    }


def _print_report(report: dict[str, Any]) -> None:
    if report["degraded_comparison"]:
        print("warning: no provisional_winner field found (pre-HI50 archive); "
              "comparing against final `winner`, which can already reflect a "
              "confirmation rejection or routing transform.", file=sys.stderr)
    print(f"measurements: {report['measurements']}")
    for name, bucket in sorted(report["policies"].items()):
        rate = bucket["agreement_rate"]
        rate_str = f"{rate * 100:.1f}%" if rate is not None else "n/a"
        print(f"  {name} v{bucket['policy_version']}: "
              f"{bucket['agree']}/{bucket['total']} agree ({rate_str})")
    if report["disagreements"]:
        print(f"disagreements: {len(report['disagreements'])}")
        for d in report["disagreements"][:20]:
            print(f"  {d['dispatch']}: {d['policy_name']} predicted "
                  f"{d['predicted_winner']!r}, actual {d['actual']!r}")


def _print_detail(detail: dict[str, Any]) -> None:
    print(f"dispatch {detail['dispatch']}")
    print(f"  native:              {detail['native']}")
    print(f"  winner:              {detail['winner']}")
    print(f"  provisional_winner:  {detail['provisional_winner']}")
    print(f"  promotion_status:    {detail['promotion_status']}")
    for policy in detail["policies"]:
        tag = " (production)" if policy["is_production"] else ""
        print(f"  policy {policy['policy_name']} v{policy['policy_version']}{tag}: "
              f"predicted {policy['predicted_winner']!r}")
        for c in policy["candidates"]:
            reason = f" ({c['rejection_reason']})" if c["rejection_reason"] else ""
            eff = f"{c['effective_us']:.3f}us" if c["effective_us"] is not None else "n/a"
            print(f"    {c['name']:<40} {eff:>12}  {c['verdict']}{reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry rank-replay")
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--dispatch", help="print full per-policy candidate detail for one dispatch")
    parser.add_argument("--verify-parity", action="store_true",
                        help="assert the production policy's predicted_winner matches provisional_winner")
    parser.add_argument("--policy-module",
                        help="registry name, dotted module path, or .py file of a prototype "
                             "policy to replay alongside the recorded ones")
    parser.add_argument("--output", type=Path, help="write the JSON report here too")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a text summary")
    args = parser.parse_args(argv)

    try:
        header, results = _read(args.measurements)
        if args.dispatch:
            payload = dispatch_detail(results, args.dispatch)
            printer = _print_detail
        elif args.verify_parity:
            payload = verify_parity(results)
            printer = None
        else:
            prototype_module = _load_prototype_policy(args.policy_module) if args.policy_module else None
            payload = build_report(header, results, prototype_module=prototype_module)
            printer = _print_report
    except (OSError, RankReplayError, rp.RankingPolicyError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1

    if args.json or printer is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        printer(payload)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.verify_parity and payload.get("mismatched", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
