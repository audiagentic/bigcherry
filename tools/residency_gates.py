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
threshold; a reversal SURVIVES if it crosses against at least one hot arm.
Sub-threshold flips, ties, and flips without median crossover (e.g. selection
noise) are reported but do not fail the gate.

Medians are per-candidate `median_us` over `status == ok` candidates only.
Signatures whose result reason marks an incomplete run (poison/fatal/failed/
run rejected/disabled) in any arm are excluded from that arm's pool — a
failure invalidates only its own arm's observation, per the locked B1
acceptance list.

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
                raise ResidencyGateError(f"{path}: result without signature at line {number}")
            if signature in results:
                raise ResidencyGateError(f"{path}: duplicate signature {signature}")
            results[signature] = row
        else:
            raise ResidencyGateError(f"{path}: unknown record kind {kind!r} at line {number}")
    if header is None or not results:
        raise ResidencyGateError(f"{path}: current header and results required")
    return header, results


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
                f"result {result.get('signature')!r}: ok candidate without a name")
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
        {"signature": s, "hot_a": hot_a[s].get("winner", ""), "hot_b": hot_b[s].get("winner", "")}
        for s in pairs if hot_a[s].get("winner") != hot_b[s].get("winner")
    ]
    return Gate1Report(pairs=len(pairs), flips=flips)


@dataclass(frozen=True)
class CrossoverDetail:
    signature: str
    cold_winner: str
    hot_winners: dict[str, str]          # arm -> winner
    per_arm: list[dict[str, Any]]        # hot_adv/cold_adv/missing info per hot arm
    crosses_any_hot_arm: bool
    replicated: bool                     # cold winner differs from BOTH hot winners


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
    missing = [n for n in (hot_winner, cold_winner)
               if n not in hot_medians or n not in cold_medians]
    if missing:
        info.update(hot_adv=None, cold_adv=None, crossover=False,
                    reason=f"missing-median:{','.join(missing)}")
        return info
    hot_adv = (hot_medians[cold_winner] - hot_medians[hot_winner]) / hot_medians[hot_winner]
    cold_adv = (cold_medians[hot_winner] - cold_medians[cold_winner]) / cold_medians[cold_winner]
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
    whether it is replicated and whether any hot arm shows a material median
    crossover. A gate-2 FAIL is: any detail with crosses_any_hot_arm True.
    """
    common = sorted(set(h1) & set(cold) & set(h2))
    details: list[CrossoverDetail] = []
    for signature in common:
        r1, rc, r2 = h1[signature], cold[signature], h2[signature]
        if not (is_completed(r1) and is_completed(rc) and is_completed(r2)):
            continue
        winners = {arm: str(r.get("winner", "")) for arm, r in (("h1", r1), ("cold", rc), ("h2", r2))}
        if winners["cold"] == winners["h1"] and winners["cold"] == winners["h2"]:
            continue
        m1, mc, m2 = candidate_medians(r1), candidate_medians(rc), candidate_medians(r2)
        per_arm = [
            _crossover_against("h1", winners["h1"], winners["cold"], m1, mc, material_pct),
            _crossover_against("h2", winners["h2"], winners["cold"], m2, mc, material_pct),
        ]
        details.append(CrossoverDetail(
            signature=signature,
            cold_winner=winners["cold"],
            hot_winners={"h1": winners["h1"], "h2": winners["h2"]},
            per_arm=per_arm,
            crosses_any_hot_arm=any(info["crossover"] for info in per_arm),
            replicated=(winners["cold"] != winners["h1"]) and (winners["cold"] != winners["h2"]),
        ))
    return details


def evaluate(h1_path: Path, cold_path: Path, h2_path: Path,
             material_pct: float = 0.05) -> dict[str, Any]:
    headers = {}
    arms = {}
    for name, path in (("h1", h1_path), ("cold", cold_path), ("h2", h2_path)):
        header, results = load_artifact(path)
        headers[name] = {key: header.get(key) for key in
                         ("flush_l2", "flush_evict_mb", "source_revision", "manifest_hash")}
        arms[name] = results
    gate1 = gate1_hot_repeatability(arms["h1"], arms["h2"])
    details = gate2_material_reversals(arms["h1"], arms["cold"], arms["h2"], material_pct)
    survivors = [d for d in details if d.crosses_any_hot_arm]
    return {
        "headers": headers,
        "material_pct": material_pct,
        "gate1": {"pairs": gate1.pairs, "flip_rate_pct": gate1.flip_rate_pct,
                  "flips": gate1.flips},
        "gate2": {
            "winner_differences": len(details),
            "replicated": [d.signature for d in details if d.replicated],
            "survivors": [
                {
                    "signature": d.signature,
                    "cold_winner": d.cold_winner,
                    "hot_winners": d.hot_winners,
                    "per_arm": d.per_arm,
                }
                for d in survivors
            ],
        },
        "verdict": {
            "gate1_pass": gate1.pairs > 0,          # stability judged by caller
            "gate2_hard_pass": not survivors,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h1", type=Path, help="hot arm 1 measurements.jsonl")
    parser.add_argument("cold", type=Path, help="flushed arm measurements.jsonl")
    parser.add_argument("h2", type=Path, help="hot arm 2 measurements.jsonl")
    parser.add_argument("--material-pct", type=float, default=0.05)
    args = parser.parse_args(argv)
    report = evaluate(args.h1, args.cold, args.h2, args.material_pct)
    json.dump(report, sys.stdout, indent=2, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
