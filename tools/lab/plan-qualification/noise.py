"""Per-round view of a campaign's paired performance lanes (noise triage).

Usage: python3 tools/lab/plan-qualification/noise.py <run-name> [<run-name> ...]
Prints every paired round's control/subject value and ratio for each lane in
the run's performance artifact, flags rounds whose ratio sits more than 3
median-absolute-deviations from the lane median, and prints each arm's
coefficient of variation -- enough to tell a real effect from one bad round
or an unstable arm.
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import WORK  # noqa: E402


def _value(metrics: object) -> float:
    if isinstance(metrics, (int, float)):
        return float(metrics)
    if isinstance(metrics, dict):
        for v in metrics.values():
            if isinstance(v, (int, float)):
                return float(v)
    raise ValueError(f"no numeric metric in {metrics!r}")


def lane_report(name: str, lane: dict) -> list[str]:
    pairs: dict[int, dict[str, float]] = {}
    order: dict[int, str] = {}
    for row in lane.get("runs") or []:
        pairs.setdefault(row["pair"], {})[row["mode"]] = _value(row["metrics"])
        order.setdefault(row["pair"], row["mode"])
    ratios = {p: v["subject"] / v["control"] for p, v in pairs.items() if "subject" in v and "control" in v}
    if not ratios:
        return [f"  {name}: no paired rounds"]
    med = statistics.median(ratios.values())
    mad = statistics.median(abs(r - med) for r in ratios.values()) or 1e-12
    out = [f"  {name} ({lane.get('metric')}): median ratio {100 * (med - 1):+.3f}%"]
    for p in sorted(ratios):
        v = pairs[p]
        flag = "  <-- outlier" if abs(ratios[p] - med) > 3 * mad else ""
        out.append(f"    pair {p} first={order[p]:7s} control={v['control']:.2f} subject={v['subject']:.2f} "
                   f"ratio={100 * (ratios[p] - 1):+.3f}%{flag}")
    for arm in ("control", "subject"):
        xs = [v[arm] for v in pairs.values() if arm in v]
        if len(xs) > 1:
            out.append(f"    {arm} CV {100 * statistics.stdev(xs) / statistics.fmean(xs):.2f}%")
    return out


def main() -> int:
    for run in sys.argv[1:]:
        files = glob.glob(str(WORK / "runs" / run / "campaign" / "artifacts" / "*perf*.json"))
        if not files:
            print(f"{run}: no performance artifact")
            continue
        doc = json.loads(Path(files[0]).read_text(encoding="utf-8"))
        print(f"== {run}")
        for lane in ("positive", "control"):
            if isinstance(doc.get(lane), dict):
                print("\n".join(lane_report(lane, doc[lane])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
