"""Fail-closed verifier for Slice A (HI24 step 4) ON/OFF tuning artifacts.

Usage: python3 tools/verify_slice_a.py <measurements.jsonl> on|off

ON artifact must satisfy all checks; OFF must prove the escape hatch.
Exit code 0 = pass, 1 = any check failed. Nothing here is advisory: every
check is a hard requirement for the local Slice A gate.

Checks (ON):
- every row parses as JSON; header present and double_native == 1
- funnel counts stay registry-sized (measured <= eligible <= applicable <= generated)
- final schedules contain unique names
- no winner or provisional winner is a twin
- exactly one #twin candidate row per result with measured > 0, zero per result
  with measured == 0 -- checked per result, so 2+0 cannot masquerade as 1+1
- completed non-MMQ rows have an ok twin row and a #twin canary pair
- every emitted canary pair is grounded: it names an actually-measured (ok)
  candidate row of this result. A non-twin MMQ pair must name an ok non-native
  challenger; a #twin pair requires an ok twin row. This is the artifact-level
  form of "J-best preferred where available": the scan only emits the
  challenger's own name when it found a J-best match, so an ungrounded or
  family-wrong pair fails here. (Full preference -- that no eligible measured
  J-best challenger was skipped -- additionally depends on the C++ policy
  function ggml_cuda_mmq_native_j_best, which is not recorded in the artifact;
  porting it would duplicate the very config tables HI24 exists to keep single.
  That direction remains a documented residual risk, verified in C++ review.)
- canary_pct is numeric wherever a canary pair is emitted

Checks (OFF):
- header double_native == 0; zero #twin candidate rows and no #twin canary
  pair anywhere; funnel/schedule/winner semantics as above

"Completed" means the row carried no fatal measurement failure: reasons
containing poison/fatal/failed/run-rejected/disabled markers are excluded.
Note that "native retained (fresh confirmation rejected provisional winner)"
IS completed -- the rejection is of a promotion, not of the measurement.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

_INCOMPLETE_MARKERS = ("poison", "fatal", "failed", "run rejected", "disabled")


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
        return [c for c in cands if isinstance(c, dict)]
    return []


def is_twin_name(name):
    return str(name).endswith("#twin")


def row_completed(r):
    reason = str(r.get("reason", "")).lower()
    return not any(marker in reason for marker in _INCOMPLETE_MARKERS)


def family_of(r, default=""):
    """Family prefix of the row's native name (the signature's family)."""
    native = str(r.get("native", ""))
    if ":" in native:
        return native.split(":")[0]
    return default


def evaluate(header, results, bad_lines, mode):
    """Return (failures, report) for one artifact. Pure function so unit
    tests can feed deliberately malformed synthetic artifacts."""
    failures = []
    jbest_used = 0
    twin_canary_used = 0

    if bad_lines:
        failures.append(f"{bad_lines} unparsable row(s)")
    if header is None:
        failures.append("header missing")
        return failures, {"jbest": 0, "twin_fallback": 0}

    expected_flag = 1 if mode == "on" else 0
    if header.get("double_native") != expected_flag:
        failures.append(
            f"header double_native is {header.get('double_native')!r}, "
            f"expected {expected_flag}"
        )

    funnel_overflow = []
    dup_schedules = 0
    twin_winners = 0
    for r in results:
        # Funnel counts must stay registry-sized: the synthetic twin is not
        # in any of them.
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

        if is_twin_name(r.get("winner")) or is_twin_name(r.get("provisional_winner")):
            twin_winners += 1

    if funnel_overflow:
        failures.append(
            f"funnel count exceeds parent in {len(funnel_overflow)} row(s): "
            f"{funnel_overflow[:3]}"
        )
    if dup_schedules:
        failures.append(f"{dup_schedules} schedule(s) with duplicate names")
    if twin_winners:
        failures.append(f"{twin_winners} winner(s)/provisional winner(s) is a twin")

    for r in results:
        rows = candidate_rows(r)
        twins = [c for c in rows if is_twin_name(c.get("name"))]
        pair = str(r.get("canary_pair", ""))
        dispatch = r.get("dispatch")
        native_name = str(r.get("native", ""))

        if mode == "on":
            # Per-result twin cardinality: totals alone would let a
            # 2-twins + 0-twins pair balance out as 1+1.
            if r.get("measured", 0) > 0:
                if len(twins) != 1:
                    failures.append(
                        f"measured result {dispatch} has {len(twins)} twin row(s), "
                        f"expected exactly 1"
                    )
            elif twins:
                failures.append(
                    f"unmeasured result {dispatch} has {len(twins)} twin row(s), "
                    f"expected 0"
                )

            if not row_completed(r):
                continue

            if pair and not isinstance(r.get("canary_pct"), (int, float)):
                failures.append(f"canary pair without numeric canary_pct in {dispatch}")

            family = family_of(r)
            ok_twins = [c for c in twins if c.get("status") == "ok"]
            if family != "mmq":
                # Non-MMQ: the double-native twin is the only same-kernel
                # pair; a completed row must have measured it and used it.
                if not ok_twins:
                    failures.append(
                        f"non-MMQ completed row {dispatch} has no ok twin measurement"
                    )
                if pair != native_name + "#twin":
                    failures.append(
                        f"non-MMQ completed row {dispatch} canary pair {pair!r} "
                        f"is not the native twin"
                    )
                elif pair:
                    twin_canary_used += 1
            else:
                # MMQ: the J-best scan emits either a measured challenger's
                # own name (it found a J-best match) or the native twin
                # (fallback). Either way the emitted pair must be grounded in
                # this result's candidate rows -- never a fabricated, stale,
                # rejected, or unmeasured name. See module docstring for the
                # documented residual risk on the preference direction.
                ok_challengers = {
                    str(c.get("name"))
                    for c in rows
                    if c.get("status") == "ok"
                    and not is_twin_name(c.get("name"))
                    and c.get("name") != native_name
                }
                expected_twin_pair = native_name + "#twin"
                if pair in ok_challengers:
                    jbest_used += 1
                elif pair == expected_twin_pair:
                    if not ok_twins:
                        failures.append(
                            f"MMQ fallback row {dispatch} emits the twin canary "
                            f"but has no ok twin measurement"
                        )
                    else:
                        twin_canary_used += 1
                elif pair:
                    failures.append(
                        f"MMQ completed row {dispatch} canary pair {pair!r} is not "
                        f"grounded in a measured candidate of this result"
                    )
        else:
            # OFF: no synthetic measurements may exist at all, and the old
            # canary semantics (MMQ J-best pairs only) must be preserved.
            if twins:
                failures.append(
                    f"OFF artifact result {dispatch} has {len(twins)} twin row(s)"
                )
            if is_twin_name(pair):
                failures.append(
                    f"OFF artifact result {dispatch} canary pair {pair!r} is a twin"
                )
            if pair and not isinstance(r.get("canary_pct"), (int, float)):
                failures.append(f"canary pair without numeric canary_pct in {dispatch}")

            # Old semantics: only the MMQ J-best scan could emit a pair, so
            # a completed non-MMQ row with any pair at all is wrong, and an
            # MMQ pair must still name a measured candidate.
            if row_completed(r):
                ok_challengers = {
                    str(c.get("name"))
                    for c in rows
                    if c.get("status") == "ok"
                    and not is_twin_name(c.get("name"))
                    and c.get("name") != native_name
                }
                if family_of(r) != "mmq":
                    if pair:
                        failures.append(
                            f"OFF artifact non-MMQ completed row {dispatch} has "
                            f"canary pair {pair!r}; pre-Slice A non-MMQ rows had none"
                        )
                elif pair and pair not in ok_challengers:
                    failures.append(
                        f"OFF artifact MMQ completed row {dispatch} canary pair "
                        f"{pair!r} is not grounded in a measured candidate"
                    )

    report = {"jbest": jbest_used, "twin_fallback": twin_canary_used}
    return failures, report


def main():
    if len(sys.argv) != 3 or sys.argv[2].lower() not in ("on", "off"):
        print(__doc__)
        sys.exit(2)
    path, mode = sys.argv[1], sys.argv[2].lower()
    header, results, bad = load(path)
    failures, report = evaluate(header, results, bad, mode)

    if mode == "on":
        print(
            f"report: canary pairs -> jbest={report['jbest']}, "
            f"native-twin-fallback={report['twin_fallback']}"
        )
    print(
        f"results={len(results)} reasons={dict(Counter(r.get('reason') for r in results))}"
    )
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
