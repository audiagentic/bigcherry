"""VTP12 card-free replay of frozen ABBA GPU receipts with explicit kernel policy.

The v1 pilot selected narrow INT8 for M=64 while VTP05's declared
INT8 dispatch is tiled at M>16. Preserve the raw pilot and original
summary; emit a separate corrected comparison with both the original
VTP05 dispatch and an optimistic best-observed-per-device reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import xtx27_quant_kernel as q
import xtx27_vtp12_compare as compare
import xtx27_vtp12_pilot as pilot

SCHEMA = "vtp12-ffn-abba-replay-v1"


def replay(directory: Path) -> dict:
    doc = json.loads((directory / "campaign.json").read_text())
    if (doc.get("schema") != pilot.SCHEMA or doc.get("challenger") not in
            ("r9700-45", "r9700-50") or doc.get("m") not in (1, 4, 8, 64)
            or doc.get("cases") != pilot.schedule(doc["challenger"], doc["m"])):
        raise ValueError("invalid/drifted immutable VTP12 campaign")
    source = []
    for idx, case in enumerate(doc["cases"]):
        cp = directory / f"case-{idx:02d}.json"
        log = directory / f"case-{idx:02d}.stdout"
        if not cp.is_file() or not log.is_file():
            raise ValueError(f"missing required case/console at index {idx}")
        raw = json.loads(cp.read_text())
        if (raw.get("ordinal") != idx or raw.get("status") !=
                "SYNTHETIC_WARM_ONLY_NUMERICAL_UNCHECKED"
                or any(raw.get(k) != case[k] for k in case)
                or raw.get("returncode") != 0 or
                raw.get("stdout_sha256") != hashlib.sha256(log.read_bytes()).hexdigest()):
            raise ValueError(f"inconsistent case receipt/console at index {idx}")
        gpu = case["gpu"]
        plugin_path = doc["provenance"]["plugins"][gpu]["path"]
        expected = q.command(
            Path(doc["provenance"]["build_dir"]) / "bin" / "r4d_selftest",
            Path(plugin_path), quant="w8a8", geo=case["geometry"],
            m=case["m"], iters=doc["iters"], cold_mib=0,
        )
        if raw.get("argv") != expected:
            raise ValueError(f"command/ABI mismatch at case {idx}")
        timings = compare.parse_perf(log.read_text(), case["geometry"]["N"],
                                     case["geometry"]["K"], case["m"], doc["iters"])
        if raw.get("all_rows_us") != timings:
            raise ValueError(f"kernel timing mismatch at case {idx}")
        frow = "gemm_fp8a8_nt_m16" if case["m"] <= 64 else "gemm_fp8a8_tiled"
        if raw.get("fp8_kernel") != frow or raw.get("fp8_us") != timings[frow]:
            raise ValueError(f"FP8 source row mismatch at case {idx}")
        source.append(raw)
    if len(list(directory.glob("case-*.json"))) != 24:
        raise ValueError("extra or incomplete case set")
    outputs = {}
    for policy in ("vtp05_declared_i8_dispatch", "optimistic_best_observed_i8"):
        corrected = []
        for raw in source:
            m = raw["m"]
            row = (q.selected_row("w8a8", m) if policy == "vtp05_declared_i8_dispatch"
                   else min(("gemm_i8a8_tiled", "gemm_i8a8_nt_m16"),
                            key=lambda key: raw["all_rows_us"][key]))
            rec = dict(raw)
            rec["i8_us"] = raw["all_rows_us"][row]
            rec["i8_kernel"] = row
            corrected.append(rec)
        outputs[policy] = pilot.to_summary(corrected, challenger=doc["challenger"])
    result = {
        "schema": SCHEMA,
        "kind": "CARD_FREE_REPLAY_OF_EXISTING_SYNTHETIC_WARM_RUN_NOT_SERVING",
        "source_campaign": str(directory / "campaign.json"),
        "source_campaign_sha256": q.sha256(directory / "campaign.json"),
        "source_original_summary_sha256": (q.sha256(directory / "summary.json")
                                           if (directory / "summary.json").is_file() else None),
        "cases_validated": len(source),
        "m": doc["m"],
        "challenger": doc["challenger"],
        "original_summary_warning": (
            "V1 M=64 pilot compares narrow INT8 on all three cards; VTP05 "
            "declares tiled INT8 at M>16. Do not use V1 M=64 summary as a "
            "selected-engine baseline." if doc["m"] == 64 else
            "V1 uses the intended narrow INT8 row for M<=16."
        ),
        "policies": outputs,
        "limitations": [
            "No proof runtime selects optimal kernel for each physical GPU",
            "No resident model tensors or attention/GDN layers",
            "No simultaneous three-GPU dispatch, transport or reduction",
            "No activation quantization, held-out quality or speculative decoding",
            "Single nine-iteration selftest per source invocation and two ABBA points per arm",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[3]
    if args.output.resolve().is_relative_to(checkout):
        parser.error("replay report belongs outside checkout")
    data = replay(args.campaign_dir)
    if args.output.exists():
        parser.error("refusing to overwrite existing replay report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.output), "m": data["m"],
        "challenger": data["challenger"],
        "stage_ideal_us": {
            key: {arm: x["stage_estimates"][arm]["ideal_ffn_stage_barriers_us"]
                  for arm in x["stage_estimates"]}
            for key, x in data["policies"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
