"""Provisional diagnostics from incomplete RAD50 A/B logs.

This intentionally does not assert final correctness or infer serving gains.
The strict ab_analyze.parse() remains the sole final-evidence parser.
"""
from __future__ import annotations
import math

PATHS = ("mapped-host-pull", "dmabuf-cpu-push")


def provisional_report(text: str) -> str:
    output = [
        "PROVISIONAL RAD50 diagnostics: no correctness, speedup or production claim.",
        "Incomplete observations are retained instead of discarded.",
    ]
    rows: dict[tuple[int, int, str], dict[str, float]] = {}
    contexts = 0
    completed = 0
    for line in text.splitlines():
        if line.startswith("RAD50_AB_CONTEXT "):
            contexts += 1
            output.append("OBSERVED_CONTEXT "+line[len("RAD50_AB_CONTEXT "):])
        elif line.startswith("RAD50_AB_COMPLETE "):
            completed += 1
            output.append("OBSERVED_COMPLETION "+line[len("RAD50_AB_COMPLETE "):])
        elif line.startswith("RAD50_AB "):
            data = {}
            for entry in line.split()[1:]:
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    data[key] = value
            try:
                dev = int(data["device"])
                size = int(data["bytes"])
                style = data["style"]
                path = data["path"]
                p50 = float(data["seq_p50_us"])
                if (dev < 0 or size <= 0 or path not in PATHS or
                        style not in ("direct-scalar", "staged-memcpy", "combine-direct", "combine-staged") or
                        not math.isfinite(p50) or p50 <= 0):
                    raise ValueError("invalid numeric or path value")
            except (KeyError, ValueError) as exc:
                output.append("INCOMPLETE_SAMPLE "+str(exc))
                continue
            output.append(f"OBSERVED_SAMPLE device={dev} bytes={size} "
                          f"style={style} path={path} seq_p50_us={p50:.2f}")
            key=(dev,size,style)
            if path in rows.setdefault(key,{}):
                output.append(f"DUPLICATE_WARNING device={dev} bytes={size} "
                              f"style={style} path={path}")
            rows[key][path]=p50
        elif (line.startswith("RAD50 A/B FAILED") or
              line.startswith("RAD50 A/B WRONG DATA") or
              (line.startswith("RAD50 RESULT test=dmabuf-ab ") or
               line.startswith("RAD50 RESULT test=combine-ab ")) or
              line.startswith("exitCode: ") or
              line.startswith("timedOut: ") or
              line.startswith("truncated: ")):
            output.append("OBSERVED_DIAGNOSTIC "+line)
    for (dev,size,style), pair in sorted(rows.items()):
        if set(pair)==set(PATHS):
            ratio=pair["mapped-host-pull"]/pair["dmabuf-cpu-push"]
            output.append(f"PROVISIONAL_PAIR device={dev} bytes={size} "
                          f"style={style} apparent_ratio={ratio:.3f} "
                          "NOT_VERIFIED=1")
        else:
            missing=set(PATHS)-set(pair)
            output.append(f"UNPAIRED_SAMPLE device={dev} bytes={size} "
                          f"style={style} missing={','.join(sorted(missing))}")
    if contexts>completed:
        output.append(f"INCOMPLETE_WARNING missing_success_marker={contexts-completed}")
    if not rows:
        output.append("INCOMPLETE_WARNING no complete per-path sample rows yet")
    output.append("Use ab_analyze.py without --provisional for verified completed-run comparisons.")
    return "\n".join(output)
