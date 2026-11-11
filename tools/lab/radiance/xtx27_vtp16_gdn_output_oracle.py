"""VTP16: original BF16 GDN out-projection, F32 host sum and residual CPU oracle.

Given the byte-exact VTP16 output-column pack, compare a full unsharded
original-BF16 projection against three independent rank-local
projections from original checkpoint weights and original-weight GDN
core outputs. Use an explicit ordered FP32 three-rank reduction, BF16
rounding and original BF16 residual add. No GPU/serving/performance claim.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp16_gdn_output_pack as output_pack

SOURCE = Path(__file__).with_name("xtx27_vtp16_gdn_output_ref.cpp")
ROOT = output_pack.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp16-gdn-output-ref"
SCHEMA = "vtp16-original-bf16-gdn-output-three-rank-cpu-oracle-v1"


def sha(path: Path, offset: int = 0, size: int | None = None) -> str:
    return output_pack.projection.sha(path, offset=offset, count=size)


def validate(package_receipt: Path) -> dict:
    p = json.loads(package_receipt.read_text())
    if (p.get("schema") != output_pack.SCHEMA
            or p.get("status") != "ORIGINAL_BF16_GDN_OUTPUT_COL_SHARDS_AND_RESIDUAL_INPUT_PASS_NO_GPU"
            or p.get("writer_sha256") != sha(Path(output_pack.__file__))):
        raise ValueError("original BF16 output pack is incomplete or writer changed")
    rows = p.get("rows")
    if type(rows) is not int or not 1 <= rows <= 8 or p.get("layer") != 0:
        raise ValueError("unsupported original checkpoint geometry or layer")
    source = p["output_source"]
    if (source.get("shape") != [5120, 6144]
            or source.get("bytes") != 5120 * 6144 * 2
            or sha(Path(source["path"]), source["offset"], source["bytes"])
            != p["original_weights_sha256"]):
        raise ValueError("actual original BF16 checkpoint output projection changed")
    full = Path(p["original_weights_file"])
    if (full.stat().st_size != source["bytes"] or
            sha(full) != p["original_weights_sha256"]):
        raise ValueError("cached original output tensor changed")
    for role, (offset, width) in output_pack.split_columns().items():
        weight = p["rank_weights"][role]
        rank_input = p["rank_inputs"][role]
        if (weight["source_columns"] != [offset, width]
                or weight["shape"] != [5120, width]
                or weight["bytes"] != 5120 * width * 2
                or Path(weight["file"]).stat().st_size != weight["bytes"]
                or sha(Path(weight["file"])) != weight["sha256"]):
            raise ValueError(f"original BF16 output weight shard {role} is corrupted")
        if (rank_input["shape"] != [rows, width]
                or rank_input["bytes"] != rows * width * 2
                or Path(rank_input["file"]).stat().st_size != rank_input["bytes"]
                or sha(Path(rank_input["file"])) != rank_input["sha256"]):
            raise ValueError(f"rank-local BF16 GDN core output {role} is corrupted")
    for key, expected_shape in (("global_input", [rows, 6144]),
                                ("residual", [rows, 5120])):
        item = p[key]
        if (item["shape"] != expected_shape
                or Path(item["file"]).stat().st_size != item["bytes"]
                or sha(Path(item["file"])) != item["sha256"]):
            raise ValueError(f"{key}: mismatch against original GDN output pack")
    if sha(Path(p["weight_input_receipt"])) != p["weight_input_sha256"]:
        raise ValueError("original input checkpoint pack receipt changed")
    if sha(Path(p["core_receipt"])) != p["core_sha256"]:
        raise ValueError("original GDN core input receipt changed")
    return p


def compile_oracle(scratch: Path) -> Path:
    digest = sha(SOURCE)
    build = scratch / "build" / digest[:16]
    binary = build / "vtp16_output_oracle"
    if binary.exists():
        if sha(binary) != (build / "binary.sha256").read_text().strip():
            raise ValueError("cached original output projection CPU oracle binary changed")
        return binary
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", "-O3", "-std=c++17", "-Wall", "-Wextra",
                    str(SOURCE), "-o", str(binary)], check=True)
    (build / "binary.sha256").write_text(sha(binary) + "\n")
    return binary


def run(package_receipt: Path, destination: Path, scratch: Path) -> dict:
    info = validate(package_receipt)
    dest = destination.expanduser().resolve()
    if dest.exists() or dest == ROOT or ROOT in dest.parents:
        raise ValueError("choose fresh output directory outside Git worktree")
    binary = compile_oracle(scratch.resolve())
    dest.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA, "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": sha(Path(__file__)),
        "cpu_oracle_sha256": sha(SOURCE),
        "cpu_binary_sha256": sha(binary),
        "original_packet": str(package_receipt.resolve()),
        "original_packet_sha256": sha(package_receipt),
        "rows": info["rows"], "layer": info["layer"],
        "column_splits": [1536, 1536, 3072],
        "outputs": {},
        "not_gpu": True, "not_engine_serving": True,
    }
    receipt = dest / "summary.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    try:
        command = [
            str(binary), "--m", str(info["rows"]), "--n", "5120",
            "--k", "6144", "--k0", "1536", "--k1", "1536", "--k2", "3072",
            "--package-dir", str(package_receipt.resolve().parent),
            "--residual", info["residual"]["file"], "--output-dir", str(dest),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        if "status=PASS" not in proc.stdout:
            raise ValueError("CPU oracle did not explicitly declare numeric success")
        result["stdout"] = proc.stdout.strip()
        for name, element_size, width in (
            ("unsharded-original.f32", 4, 5120),
            ("ordered-three-rank.f32", 4, 5120),
            ("unsharded-residual.bf16", 2, 5120),
            ("ordered-three-rank-residual.bf16", 2, 5120),
            ("partial-xtx0.f32", 4, 5120),
            ("partial-xtx1.f32", 4, 5120),
            ("partial-r9700.f32", 4, 5120),
        ):
            file = dest / name
            if file.stat().st_size != info["rows"] * width * element_size:
                raise ValueError(f"CPU oracle output {name} has incorrect extent")
            result["outputs"][name] = {
                "file": str(file), "sha256": sha(file), "bytes": file.stat().st_size,
                "shape": [info["rows"], width],
            }
        result["status"] = "ORIGINAL_BF16_GDN_OUTPUT_PARTIAL_F32_REDUCTION_RESIDUAL_CPU_PASS"
    except Exception as e:
        result.update({"status": "FAILED_INCOMPLETE", "error": str(e)})
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-receipt", type=Path, required=True)
    p.add_argument("--phase", choices=("plan", "run"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    a = p.parse_args(argv)
    if a.phase == "run" and not a.run:
        p.error("original BF16 checkpoint numerical processing requires explicit --run")
    try:
        packet = validate(a.packet_receipt)
        if a.phase == "plan":
            print(json.dumps({
                "status": "ORIGINAL_BF16_OUTPUT_COLUMN_PACK_VERIFIED_CARD_FREE",
                "rows": packet["rows"], "layer": packet["layer"],
                "column_splits": [1536, 1536, 3072],
                "cpu_reference_source_sha256": sha(SOURCE),
                "gpu_required": False, "model_serving": False,
            }, indent=2))
            return 0
        dest = (a.output if a.output else a.scratch / datetime.now(UTC).strftime(
            f"layer{packet['layer']}-%Y%m%dT%H%M%S%fZ"
        ))
        record = run(a.packet_receipt, dest, a.scratch)
        print(json.dumps({
            "status": record["status"],
            "receipt": str(dest.expanduser().resolve() / "summary.json"),
            "numerical_result": record["stdout"],
        }, indent=2))
        return 0
    except (OSError, TypeError, KeyError, ValueError, subprocess.CalledProcessError) as e:
        p.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
