"""VTP19: one-lease balanced A/B/C numerical timing of three TP3 output variants.

Runs original BF16 output projection, three-GPU host sum and BF16 residual
using exactly the same immutable source-verified checkpoint inputs and
independent CPU references under one exclusive agent-lease. A =
VTP16 pageable, B = VTP17 pinned async, C = VTP18 row-reuse pinned.
Order A/B/C/C/B/A per input to partly counter linear warm/clock drift.

A failed GPU oracle aborts the entire run, retaining an incomplete
receipt. This is a microkernel/host-collective A/B experiment, never
a real 27B serving benchmark or native dispatch qualification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp16_gdn_output_gpu as pageable
import xtx27_vtp17_gdn_output_pinned_gpu as pinned
import xtx27_vtp18_gdn_output_rowreuse_gpu as reused
import xtx27_vtp_micro as micro

ROOT = micro.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp19-original-BF16-output-abccba"
SCHEMA = "vtp19-original-bf16-gdn-output-three-physical-abccba-v1"
PURPOSE = "VTP19: balanced A/B/C original BF16 output GEMV/collective control"
VARIANTS = {
    "pageable": pageable.SOURCE,
    "pinned": pinned.SOURCE,
    "rowreuse": reused.SOURCE,
}
ORDER = ("pageable", "pinned", "rowreuse", "rowreuse", "pinned", "pageable")


def inputs(packets: list[Path], oracles: list[Path]) -> list[dict]:
    if not packets and not oracles:
        return []
    if len(packets) != len(oracles) or not 1 <= len(packets) <= 2:
        raise ValueError("provide 1..2 matched original packet/oracle pairs")
    matched = []
    for pkt, oracle in zip(packets, oracles, strict=True):
        source = pageable.validate(pkt, oracle)
        if source["rows"] not in (1, 2, 4, 8):
            raise ValueError("balanced A/B requires original reference rows 1,2,4,8")
        if source["rows"] in (x["rows"] for x in matched):
            raise ValueError("duplicate row shape")
        matched.append(source)
    return sorted(matched, key=lambda x: x["rows"])


def plan(matched: list[dict], repetitions: int) -> list[dict]:
    if type(repetitions) is not int or not 1 <= repetitions <= 30:
        raise ValueError("each arm needs 1..30 repetitions")
    cases = []
    for source in matched:
        for position, variant in enumerate(ORDER, start=1):
            rows = source["rows"]
            cases.append({
                "kind": variant,
                "label": f"m{rows}-arm{position:02d}-{variant}",
                "args": [
                    "--rows", str(rows),
                    "--iterations", str(repetitions),
                    "--pack-dir", source["packet_directory"],
                    "--reference-dir", source["reference_directory"],
                    "--residual", source["residual_file"],
                ],
                "data_definition": (
                    f"VTP19 balanced original BF16 GDN output, M={rows}, "
                    f"A/B/C arm {position}/6, variant={variant}; "
                    "5120x6144 source, 1536/1536/3072 column ownership; "
                    "three physical GPUs, no P2P, ordered F32 host sum; "
                    "independent full BF16 CPU projection+residual oracle; "
                    "not native model serving"
                ),
            })
    return cases


def extract_rows(receipt: dict, directory: Path, expected: list[dict]) -> dict:
    completed = receipt["cases"]
    if len(completed) != len(expected):
        raise ValueError("incomplete scheduled A/B/C case sequence")
    measured = {}
    for actual, planned in zip(completed, expected, strict=True):
        if (actual["label"] != planned["label"] or actual["kind"] != planned["kind"]
                or actual["rc"] != 0 or not actual["oracle_pass"]):
            raise ValueError("invalid partial case or numerical oracle in A/B/C")
        lines = (directory / actual["stdout"]).read_text().splitlines()
        if len(lines) < 2:
            raise ValueError("missing numerical timing and output oracle")
        record = list(csv.DictReader(lines[:2]))
        if len(record) != 1:
            raise ValueError("missing one canonical timing record")
        data = record[0]
        size = int(planned["args"][1])
        if (data.get("oracle") != "PASS" or int(data["rows"]) != size
                or int(data["final_bf16_exact"]) != int(data["total"])
                or int(data["total"]) != size * 5120):
            raise ValueError("original BF16 output reference did not match exactly")
        for key in ("wall_us_p50", "wall_us_p95", "xtx0_kernel_us_p50",
                    "xtx1_kernel_us_p50", "r9700_kernel_us_p50",
                    "max_partial_abs", "max_reduce_abs", "max_final_bf16_abs"):
            value = float(data[key])
            if not 0 <= value < 1e9:
                raise ValueError(f"nonfinite or invalid {key}")
        entry = {
            "variant": actual["kind"], "case": actual["label"],
            "wall_us_p50": float(data["wall_us_p50"]),
            "wall_us_p95": float(data["wall_us_p95"]),
            "xtx0_kernel_us_p50": float(data["xtx0_kernel_us_p50"]),
            "xtx1_kernel_us_p50": float(data["xtx1_kernel_us_p50"]),
            "r9700_kernel_us_p50": float(data["r9700_kernel_us_p50"]),
            "max_partial_abs": float(data["max_partial_abs"]),
            "max_reduce_abs": float(data["max_reduce_abs"]),
            "max_final_bf16_abs": float(data["max_final_bf16_abs"]),
            "bf16_exact": int(data["final_bf16_exact"]),
            "total": int(data["total"]),
            "oracle": "PASS",
        }
        measured.setdefault(str(size), []).append(entry)
    summary = {}
    for row_size, entries in measured.items():
        by_variant = {}
        for variant in VARIANTS:
            arms = [case for case in entries if case["variant"] == variant]
            if len(arms) != 2:
                raise ValueError("expected exactly two balanced arms per variant")
            by_variant[variant] = {
                "arms": [arm["case"] for arm in arms],
                "wall_p50_median_of_arms_us": statistics.median(
                    arm["wall_us_p50"] for arm in arms
                ),
                "wall_p95_median_of_arms_us": statistics.median(
                    arm["wall_us_p95"] for arm in arms
                ),
            }
        baseline = by_variant["pageable"]["wall_p50_median_of_arms_us"]
        for variant in ("pinned", "rowreuse"):
            optimized = by_variant[variant]["wall_p50_median_of_arms_us"]
            by_variant[variant]["indicative_reduction_percent_vs_pageable"] = round(
                (1.0 - optimized / baseline) * 100.0, 2
            )
        by_variant["rowreuse"]["indicative_reduction_percent_vs_pinned"] = round(
            (1.0 - by_variant["rowreuse"]["wall_p50_median_of_arms_us"] /
             by_variant["pinned"]["wall_p50_median_of_arms_us"]) * 100.0, 2
        )
        summary[row_size] = {"arms": entries, "variants": by_variant}
    return summary


def readonly_gpu_telemetry() -> dict:
    """Best-effort read-only evidence of GPU clocks/power/thermal conditions.

    Never adjusts clocks, undervolt, power caps or lease ownership.
    Preserve even missing/failed command as explicit evidence rather
    than assuming stock frequencies during an external power sweep.
    """
    now = datetime.now(UTC).isoformat()
    cmd = ["rocm-smi", "--showclocks", "--showpower", "--showtemp"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                           check=False)
        return {
            "captured_utc": now, "command": cmd,
            "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as err:
        return {
            "captured_utc": now, "command": cmd,
            "returncode": None, "error": repr(err),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "compile", "gpu"), default="plan")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--packet-receipt", action="append", type=Path, default=[])
    parser.add_argument("--reference-receipt", action="append", type=Path, default=[])
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--hipcc")
    a = parser.parse_args(argv)
    if a.phase != "plan" and not a.run:
        parser.error("compile or exclusive-GPU run requires --run")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        parser.error("internal runner refuses to start without owned agent-lease")
    if not 1 <= a.iterations <= 30:
        parser.error("iterations must be in 1..30")
    try:
        matched = inputs(a.packet_receipt, a.reference_receipt)
        if a.phase == "gpu" and not matched:
            raise ValueError("GPU A/B/C needs matched original checkpoint receipts")
        cases = plan(matched, a.iterations)
    except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    digest = micro.source_digest(VARIANTS)
    origin = {
        "schema": SCHEMA,
        "source_digest": digest,
        "input_pairs": matched,
        "schedule": [x["label"] for x in cases],
        "variants": {name: {
            "source": str(src), "sha256": pageable.oracle.sha(src),
        } for name, src in VARIANTS.items()},
        "not_full_model": True,
        "not_native_kernel": True,
        "power_and_temperature_not_enforced": True,
        "numerical_oracle_required_on_every_arm": True,
        "runner_source_sha256": pageable.oracle.sha(Path(__file__)),
        "read_only_clock_power_thermal_telemetry": True,
    }
    if a.phase == "plan":
        print(json.dumps(origin, indent=2))
        return 0
    compiler = micro.compiler_path(a.hipcc)
    if compiler is None:
        parser.error("missing HIP compiler")
    scratch = a.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        parser.error("output must not be inside Git")
    binaries = micro.build_binaries(compiler, scratch, digest, sources=VARIANTS)
    if a.phase == "compile":
        print(json.dumps({
            **origin, "status": "card_free_all_three_compiled",
            "binaries": {k: str(v) for k, v in binaries.items()},
        }, indent=2))
        return 0
    if not a._under_lease:
        forwarded = [
            sys.executable, str(Path(__file__).resolve()), "--phase", "gpu",
            "--run", "--_under-lease", "--iterations", str(a.iterations),
            "--scratch", str(scratch), "--hipcc", str(compiler),
        ]
        for p in a.packet_receipt:
            forwarded.extend(["--packet-receipt", str(p)])
        for p in a.reference_receipt:
            forwarded.extend(["--reference-receipt", str(p)])
        return subprocess.run([
            "agent-lease", "acquire", "gpu", "--ttl", "30m",
            "--purpose", PURPOSE, "--", "env", "VTP04_LEASED=1", *forwarded,
        ], check=False).returncode
    # Require the holder's intended lease before reading environmental
    # evidence; run_leased additionally rejects any resident KFD PIDs.
    # Telemetry is observational and never manipulates the other
    # power-tuning agent's settings.
    before = readonly_gpu_telemetry()
    out = micro.run_leased(
        scratch=scratch, binaries=binaries, cases=cases, digest=digest,
        purpose=PURPOSE,
        receipt_type="VTP19_ORIGINAL_BF16_GDN_OUT_PROJ_PAGEABLE_PINNED_ROWREUSE_ABCCBA",
    )
    after = readonly_gpu_telemetry()
    file = out / "summary.json"
    r = json.loads(file.read_text())
    # Do not convert an incomplete or failed acceptance to PASS.
    analysis = extract_rows(r, out, cases)
    r.update({
        **origin,
        "balanced_abccba": analysis,
        "gpu_environment_observations": {"before": before, "after": after},
        "status": "VTP19_ALL_ARMS_ORACLE_PASS_3GPU_ABCCBA_NOT_MODEL",
        "fairness_limits": [
            "same GPU lease, compiler and original-weight workload",
            "distinct process starts within lease; clocks/power not enforced",
            "different kernel architecture; not real TP2/TP3 serving",
            "per-case kernel timing includes warmups but ignores process startup",
        ],
    })
    file.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"status": r["status"], "receipt": str(file),
                      "summary": {key: {k: v["wall_p50_median_of_arms_us"]
                                     for k, v in value["variants"].items()}
                                  for key, value in analysis.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
