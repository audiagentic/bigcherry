"""HI34 B1 residency-experiment gate evaluation (H1 -> C -> H2 arms).

Reads three measurements.jsonl artifacts (header + result lines) and reports
the two locked gates:

Gate 1 (prerequisite): hot-replicate repeatability — winner flips between the
two flush=0 arms over signatures completed in both. If the hot envelope is
unstable, the cold arm must not be interpreted.

Gate 2 (hard pass): zero replicated material winner reversals with median
crossover. For each signature whose COLD winner differs from BOTH hot
winners, per hot arm A:

    hot_adv(A)  = (t_A(cold_winner) - t_A(hot_winner)) / t_A(hot_winner)
                  positive: the hot winner is faster on its own context
    cold_adv(A) = (t_cold(hot_winner) - t_cold(cold_winner)) / t_cold(cold_winner)
                  positive: the cold winner is faster on the cold context

A median crossover against arm A requires BOTH to exceed the materiality
threshold. A reversal is a HARD GATE-2 SURVIVOR only if it is replicated
(cold winner differs from BOTH hot winners) AND shows a material median
crossover against BOTH hot arms. Non-replicated flips and one-hot-arm-only
crossovers stay in the diagnostic report but do NOT fail the hard gate.
Sub-threshold flips, ties, and flips without median crossover (e.g. selection
noise) are likewise reported only.

Medians are per-candidate `median_us` over `status == ok` candidates only.
Signatures whose result reason marks an incomplete run (poison/fatal/failed/
run rejected/disabled) in any arm are excluded from that arm's pool — a
failure invalidates only its own arm's observation, per the locked B1
acceptance list.

Arm provenance check (HI65 pre-run requirement): when declared arm specs are
supplied (--arm NAME=MODE[@MB]), each arm's header must match its declaration
before its analysis may be interpreted. EVICT and EVICT_REWARM both carry the
legacy flush_l2=1 wire mirror, so only the resolved pre_sample_mode string can
distinguish them; artifacts that predate that field cannot be attributed and
fail closed. source_revision and manifest_hash must also agree across arms.
On any mismatch evaluate() still returns the full report (for diagnosis) but
header_check.status is 'fail' and main() exits non-zero.

Stdlib only, mirroring tools/compare_tunes.py and tools/verify_slice_a.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INCOMPLETE_REASON_MARKERS = ("poisoned", "fatal", "failed", "run rejected", "disabled")
VALID_PRE_SAMPLE_MODES = ("none", "evict", "evict_rewarm")


class ResidencyGateError(RuntimeError):
    pass


def load_artifact(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Parse one arm artifact into (header, results keyed by signature)."""
    header: dict[str, Any] | None = None
    results: dict[str, dict[str, Any]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResidencyGateError(f"{path}: malformed line {number}") from exc
        kind = row.get("kind")
        if kind == "header":
            if header is not None:
                raise ResidencyGateError(f"{path}: duplicate header")
            header = row
        elif kind == "result":
            signature = row.get("signature")
            if not isinstance(signature, str):
                raise ResidencyGateError(
                    f"{path}: result without signature at line {number}"
                )
            if signature in results:
                raise ResidencyGateError(f"{path}: duplicate signature {signature}")
            results[signature] = row
        else:
            raise ResidencyGateError(
                f"{path}: unknown record kind {kind!r} at line {number}"
            )
    if header is None or not results:
        raise ResidencyGateError(f"{path}: current header and results required")
    return header, results


def parse_arm_spec(value: str) -> tuple[str, dict[str, Any]]:
    """Parse NAME=MODE[@MB] into (arm name, declared provenance spec).

    MODE must be one of VALID_PRE_SAMPLE_MODES; MB is optional and must be a
    positive integer. Fails closed with ValueError on anything else.
    """
    name, sep, rest = value.partition("=")
    if not sep or not name:
        raise ValueError(f"arm spec must be NAME=MODE[@MB], got {value!r}")
    mode, at, mb_text = rest.partition("@")
    if mode not in VALID_PRE_SAMPLE_MODES:
        raise ValueError(
            f"arm mode must be one of {VALID_PRE_SAMPLE_MODES}, got {mode!r}"
        )
    mb: int | None = None
    if at:
        try:
            mb = int(mb_text)
        except ValueError as exc:
            raise ValueError(
                f"arm flush size must be an integer MB, got {mb_text!r}"
            ) from exc
        if mb <= 0:
            raise ValueError("arm flush size must be positive")
    return name, {"pre_sample_mode": mode, "flush_evict_mb": mb}


def check_arm_provenance(
    headers: dict[str, dict[str, Any]], expected_arms: dict[str, dict[str, Any]]
) -> list[str]:
    """Fail-closed provenance check of declared arms against artifact headers.

    Returns the list of mismatches (empty when every declared arm matches).
    An arm whose header lacks pre_sample_mode cannot be attributed to a mode
    and always fails, regardless of its legacy flush_l2 mirror.
    """
    mismatches: list[str] = []
    for arm, spec in expected_arms.items():
        h = headers.get(arm)
        if h is None:
            mismatches.append(f"{arm}: declared arm has no header")
            continue
        actual_mode = h.get("pre_sample_mode")
        want_mode = spec["pre_sample_mode"]
        if actual_mode is None:
            mismatches.append(
                f"{arm}: header lacks pre_sample_mode (artifact predates HI65 "
                "provenance; mode cannot be attributed)"
            )
        elif actual_mode != want_mode:
            mismatches.append(
                f"{arm}: pre_sample_mode {actual_mode!r} != declared {want_mode!r}"
            )
        want_mb = spec.get("flush_evict_mb")
        if want_mb is not None and h.get("flush_evict_mb") != want_mb:
            mismatches.append(
                f"{arm}: flush_evict_mb {h.get('flush_evict_mb')!r} != "
                f"declared {want_mb}"
            )
    for field in ("source_revision", "manifest_hash"):
        values = [h.get(field) for h in headers.values()]
        if any(v is None for v in values):
            mismatches.append(f"{field}: missing in one or more arm headers")
        elif len({str(v) for v in values}) > 1:
            mismatches.append(
                f"{field}: differs across arms: {sorted(str(v) for v in values)}"
            )
    return mismatches


def is_completed(result: dict[str, Any]) -> bool:
    """A result whose reason marks the run incomplete is excluded from gates."""
    reason = str(result.get("reason", ""))
    return not any(marker in reason for marker in INCOMPLETE_REASON_MARKERS)


def _ok_candidate_values(result: dict[str, Any]):
    """Yield (name, median_us) over status == ok candidates with positive medians.

    Fails closed: an ok candidate without a name is malformed artifact data,
    the same class of error compare_tunes.py rejects via CompareError.
    """
    for candidate in result.get("candidates", []):
        if not isinstance(candidate, dict) or candidate.get("status") != "ok":
            continue
        name = candidate.get("name")
        value = candidate.get("median_us")
        if name is None:
            raise ResidencyGateError(
                f"result {result.get('signature')!r}: ok candidate without a name"
            )
        if isinstance(value, (int, float)) and value > 0:
            # value is pre-validated as int/float above, so the conversion below
            # cannot raise; the ast-grep throwing-call matcher does not see that.
            yield str(name), value


def candidate_medians(result: dict[str, Any]) -> dict[str, float]:
    """name -> median_us over status == ok candidates with a positive median."""
    return dict(_ok_candidate_values(result))


@dataclass(frozen=True)
class Gate1Report:
    pairs: int
    flips: list[dict[str, str]]

    @property
    def flip_rate_pct(self) -> float:
        return 100.0 * len(self.flips) / self.pairs if self.pairs else 0.0


def gate1_hot_repeatability(
    hot_a: dict[str, dict[str, Any]],
    hot_b: dict[str, dict[str, Any]],
) -> Gate1Report:
    """Winner flips between the two hot arms over their completed intersection."""
    common = sorted(set(hot_a) & set(hot_b))
    pairs = [s for s in common if is_completed(hot_a[s]) and is_completed(hot_b[s])]
    flips = [
        {
            "signature": s,
            "hot_a": hot_a[s].get("winner", ""),
            "hot_b": hot_b[s].get("winner", ""),
        }
        for s in pairs
        if hot_a[s].get("winner") != hot_b[s].get("winner")
    ]
    return Gate1Report(pairs=len(pairs), flips=flips)


@dataclass(frozen=True)
class CrossoverDetail:
    signature: str
    cold_winner: str
    hot_winners: dict[str, str]  # arm -> winner
    per_arm: list[dict[str, Any]]  # hot_adv/cold_adv/missing info per hot arm
    crosses_any_hot_arm: bool  # diagnostic only; never the hard gate
    crosses_both_hot_arms: bool  # diagnostic only; necessary, not sufficient
    replicated: bool  # cold winner differs from BOTH hot winners

    @property
    def hard_gate_survivor(self) -> bool:
        """The locked hard-gate criterion: replicated AND crossover vs BOTH arms."""
        return self.replicated and self.crosses_both_hot_arms


def _crossover_against(
    arm: str,
    hot_winner: str,
    cold_winner: str,
    hot_medians: dict[str, float],
    cold_medians: dict[str, float],
    material_pct: float,
) -> dict[str, Any]:
    info: dict[str, Any] = {"arm": arm}
    if hot_winner == cold_winner:
        info.update(hot_adv=None, cold_adv=None, crossover=False, reason="same-winner")
        return info
    missing = [
        n
        for n in (hot_winner, cold_winner)
        if n not in hot_medians or n not in cold_medians
    ]
    if missing:
        info.update(
            hot_adv=None,
            cold_adv=None,
            crossover=False,
            reason=f"missing-median:{','.join(missing)}",
        )
        return info
    hot_adv = (hot_medians[cold_winner] - hot_medians[hot_winner]) / hot_medians[
        hot_winner
    ]
    cold_adv = (cold_medians[hot_winner] - cold_medians[cold_winner]) / cold_medians[
        cold_winner
    ]
    info.update(
        hot_adv=hot_adv,
        cold_adv=cold_adv,
        crossover=bool(hot_adv > material_pct and cold_adv > material_pct),
        reason="ok",
    )
    return info


def gate2_material_reversals(
    h1: dict[str, dict[str, Any]],
    cold: dict[str, dict[str, Any]],
    h2: dict[str, dict[str, Any]],
    material_pct: float = 0.05,
) -> list[CrossoverDetail]:
    """All signatures completed in all three arms, evaluated for gate 2.

    Returns one detail per signature whose cold winner differs from at least
    one hot winner (including the non-replicated ones), each annotated with
    replication and per-arm median crossovers. A gate-2 FAIL is: any detail
    with hard_gate_survivor True (replicated AND crossover against BOTH hot
    arms). Non-replicated flips and one-hot-arm-only crossovers are reported
    for diagnosis but do not fail the locked hard gate.
    """
    common = sorted(set(h1) & set(cold) & set(h2))
    details: list[CrossoverDetail] = []
    for signature in common:
        r1, rc, r2 = h1[signature], cold[signature], h2[signature]
        if not (is_completed(r1) and is_completed(rc) and is_completed(r2)):
            continue
        winners = {
            arm: str(r.get("winner", ""))
            for arm, r in (("h1", r1), ("cold", rc), ("h2", r2))
        }
        if winners["cold"] == winners["h1"] and winners["cold"] == winners["h2"]:
            continue
        m1, mc, m2 = candidate_medians(r1), candidate_medians(rc), candidate_medians(r2)
        per_arm = [
            _crossover_against(
                "h1", winners["h1"], winners["cold"], m1, mc, material_pct
            ),
            _crossover_against(
                "h2", winners["h2"], winners["cold"], m2, mc, material_pct
            ),
        ]
        details.append(
            CrossoverDetail(
                signature=signature,
                cold_winner=winners["cold"],
                hot_winners={"h1": winners["h1"], "h2": winners["h2"]},
                per_arm=per_arm,
                crosses_any_hot_arm=any(info["crossover"] for info in per_arm),
                crosses_both_hot_arms=all(info["crossover"] for info in per_arm),
                replicated=(winners["cold"] != winners["h1"])
                and (winners["cold"] != winners["h2"]),
            )
        )
    return details


def evaluate(
    h1_path: Path,
    cold_path: Path,
    h2_path: Path,
    material_pct: float = 0.05,
    expected_arms: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    headers = {}
    arms = {}
    for name, path in (("h1", h1_path), ("cold", cold_path), ("h2", h2_path)):
        header, results = load_artifact(path)
        headers[name] = {
            key: header.get(key)
            for key in (
                "flush_l2",
                "flush_evict_mb",
                "pre_sample_mode",
                "source_revision",
                "manifest_hash",
            )
        }
        arms[name] = results
    if expected_arms is None:
        header_check: dict[str, Any] = {"status": "skipped", "mismatches": []}
    else:
        mismatches = check_arm_provenance(headers, expected_arms)
        header_check = {
            "status": "fail" if mismatches else "ok",
            "mismatches": mismatches,
        }
    gate1 = gate1_hot_repeatability(arms["h1"], arms["h2"])
    details = gate2_material_reversals(
        arms["h1"], arms["cold"], arms["h2"], material_pct
    )
    # The locked hard gate: replicated AND material median crossover against
    # BOTH hot arms. One-hot-arm-only crossovers and non-replicated flips stay
    # in the diagnostic report but never fail the gate.
    survivors = [d for d in details if d.hard_gate_survivor]
    diagnostics = [d for d in details if not d.hard_gate_survivor]
    return {
        "headers": headers,
        "material_pct": material_pct,
        "gate1": {
            "pairs": gate1.pairs,
            "flip_rate_pct": gate1.flip_rate_pct,
            "flips": gate1.flips,
        },
        "gate2": {
            "winner_differences": len(details),
            "replicated": [d.signature for d in details if d.replicated],
            # Hard-gate survivors: the only rows that can fail Gate 2.
            "survivors": [
                {
                    "signature": d.signature,
                    "cold_winner": d.cold_winner,
                    "hot_winners": d.hot_winners,
                    "per_arm": d.per_arm,
                }
                for d in survivors
            ],
            # Reported for diagnosis; do NOT fail the locked hard gate.
            "diagnostics": [
                {
                    "signature": d.signature,
                    "cold_winner": d.cold_winner,
                    "hot_winners": d.hot_winners,
                    "per_arm": d.per_arm,
                    "replicated": d.replicated,
                    "crosses_any_hot_arm": d.crosses_any_hot_arm,
                    "crosses_both_hot_arms": d.crosses_both_hot_arms,
                    "not_a_survivor_because": (
                        "not-replicated"
                        if not d.replicated
                        else "crossover-missing-on-one-hot-arm"
                    ),
                }
                for d in diagnostics
            ],
        },
        "header_check": header_check,
        "verdict": {
            # When the provenance check fails, the gates below are diagnostic
            # only; the run must not be interpreted until headers match.
            "header_check_pass": header_check["status"] != "fail",
            "gate1_pass": gate1.pairs > 0,  # stability judged by caller
            "gate2_hard_pass": not survivors,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h1", type=Path, help="hot arm 1 measurements.jsonl")
    parser.add_argument("cold", type=Path, help="flushed arm measurements.jsonl")
    parser.add_argument("h2", type=Path, help="hot arm 2 measurements.jsonl")
    parser.add_argument("--material-pct", type=float, default=0.05)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=MODE[@MB]",
        help=(
            "declared arm provenance (repeatable), e.g. --arm cold=evict@256; "
            "arms whose headers do not match fail the report (exit 2)"
        ),
    )
    args = parser.parse_args(argv)
    expected: dict[str, dict[str, Any]] = {}
    for value in args.arm:
        try:
            name, spec = parse_arm_spec(value)
        except ValueError as exc:
            parser.error(str(exc))
        if name not in ("h1", "cold", "h2"):
            parser.error(f"arm name must be h1/cold/h2, got {name!r}")
        expected[name] = spec
    report = evaluate(
        args.h1, args.cold, args.h2, args.material_pct, expected_arms=expected or None
    )
    json.dump(report, sys.stdout, indent=2, default=str)
    print()
    # Exit 0: analysis acceptable (check ok or skipped). Exit 2: declared-arm
    # provenance mismatch — the run failed closed and must not be interpreted.
    return 2 if report["header_check"]["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
