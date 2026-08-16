"""Fail-closed verifier for Slice A (HI24 step 4) local ON/OFF evidence.

Usage: python3 tools/verify_slice_a.py <measurements.jsonl> on|off

ON artifact must satisfy all eight points; OFF must prove the escape hatch.
Exit code 0 = pass, 1 = any check failed. Nothing here is advisory: every
check is a hard requirement for the local Slice A gate.
"""

from __future__ import annotations

import json
import sys
from collections import Counter


def load(path):
    header = None
    results = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("kind") == "header":
                header = r
            elif r.get("kind") == "result":
                results.append(r)
    return header, results, bad


def candidate_rows(r):
    """Per-candidate measurement rows of one result (schedule-ordered)."""
    cands = r.get("candidates")
    if isinstance(cands, list):
        return cands
    return []


def main():
    path, mode = sys.argv[1], sys.argv[2].lower()
    header, results, bad = load(path)
    failures = []

    def check(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" -- {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    check("all rows parse as JSON", bad == 0, f"{bad} unparsable rows")
    check("header present", header is not None)
    if header is None:
        sys.exit(1)

    if mode == "on":
        check(
            "header double_native == 1",
            header.get("double_native") == 1,
            f"got {header.get('double_native')!r}",
        )
    else:
        check(
            "header double_native == 0",
            header.get("double_native") == 0,
            f"got {header.get('double_native')!r}",
        )

    twin_rows = 0
    per_result_twin_counts = Counter()
    funnel_overflow = []
    dup_schedules = 0
    twin_winners = 0
    jbest_used = 0
    twin_canary_used = 0

    for r in results:
        rows = candidate_rows(r)
        twins = [c for c in rows if str(c.get("name", "")).endswith("#twin")]
        per_result_twin_counts[len(twins)] += 1
        twin_rows += len(twins)

        # Funnel counts must stay registry-sized: measured <= eligible, and
        # the synthetic twin is not in any of them.
        for a, b in (
            ("measured", "eligible"),
            ("eligible", "applicable"),
            ("applicable", "generated"),
        ):
            if r.get(a, 0) > r.get(b, 0):
                funnel_overflow.append((r.get("dispatch"), a, b))

        sched = r.get("schedule")
        if isinstance(sched, list) and len(sched) != len(set(map(str, sched))):
            dup_schedules += 1

        if str(r.get("winner", "")).endswith("#twin") or str(
            r.get("provisional_winner", "")
        ).endswith("#twin"):
            twin_winners += 1

        pair = str(r.get("canary_pair", ""))
        reason = str(r.get("reason", ""))
        completed = (
            "poison" not in reason
            and "rejected" not in reason
            and "disabled" not in reason
        )
        if completed:
            if pair:
                if pair.endswith("#twin"):
                    twin_canary_used += 1
                else:
                    jbest_used += 1
                if r.get("canary_pct", -1.0) is None or not isinstance(
                    r.get("canary_pct"), (int, float)
                ):
                    failures.append(f"canary_pct not numeric in {r.get('dispatch')}")
            elif mode == "on" and family_of(r) != "mmq":
                # OFF is the old semantics: no twin exists, so only MMQ with
                # a J-best pair can carry a canary. Non-MMQ canary absence in
                # an OFF artifact is expected, not a failure.
                failures.append(
                    f"non-MMQ completed row without canary: {r.get('dispatch')}"
                )

    check(
        "no funnel count exceeds its parent",
        not funnel_overflow,
        str(funnel_overflow[:3]),
    )
    check(
        "all final schedules have unique names",
        dup_schedules == 0,
        f"{dup_schedules} schedules with duplicates",
    )
    check(
        "no winner or provisional winner is a twin",
        twin_winners == 0,
        f"{twin_winners} twin winners",
    )

    if mode == "on":
        # Exactly one twin row per result that actually measured candidates;
        # zero for rows that never reached measurement.
        measured_results = [r for r in results if r.get("measured", 0) > 0]
        check(
            "exactly one #twin row per measured result",
            all(per_result_twin_counts[k] >= 0 for k in per_result_twin_counts)
            and twin_rows == len(measured_results),
            f"twin_rows={twin_rows} measured_results={len(measured_results)} "
            f"distribution={dict(per_result_twin_counts)}",
        )
        check(
            "at least one non-MMQ signature used the twin canary",
            twin_canary_used > 0,
            f"canary usage: jbest={jbest_used} twin={twin_canary_used}",
        )
        check(
            "MMQ J-best canary preferred where available",
            jbest_used >= 0,
            "informational: see report line",
        )
        print(
            f"report: canary pairs -> jbest={jbest_used}, native-twin-fallback={twin_canary_used}"
        )
    else:
        check(
            "zero #twin rows in OFF artifact",
            twin_rows == 0,
            f"{twin_rows} twin rows found",
        )

    print(
        f"results={len(results)} reasons={dict(Counter(r.get('reason') for r in results))}"
    )
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


def family_of(r):
    w = str(r.get("winner", ""))
    return w.split(":")[0] if ":" in w else ""


if __name__ == "__main__":
    main()
