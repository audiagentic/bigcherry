"""VTP24: independent original-BF16 FFN CPU oracle and residual handoff.

Reads a source-hashed VTP24 original Qwen3.8 layer-0 FFN shard pack,
which consumes the original GDN core/out projection CPU residual.
Calculates all three original BF16 gate/up, SwiGLU, original down
projection partials, F32 host combine and the resulting complete
BF16 layer0 residual, suitable as layer1 input for future integration.

This is source-integrity and CPU numeric provenance. No GPU, no
native 64-layer model execution or generated-token claim.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp24_ffn_pack as pack

ROOT = pack.ROOT
SOURCE = Path(__file__).with_name("xtx27_vtp24_ffn_ref.cpp")
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp24-original-ffn-cpu"
SCHEMA = "vtp24-qwen38-original-BF16-layer0-ffn-residual-cpu-oracle-v1"


def validate(packet: Path) -> dict:
    data = pack.validate(packet)
    for key, descriptor in data["sources"].items():
        actual = pack.projection.sha(
            Path(descriptor["path"]), descriptor["offset"], descriptor["bytes"]
        )
        if actual != data["original_tensor_sha256"][key]:
            raise ValueError(f"original BF16 FFN checkpoint source changed: {key}")
    return data


def compile_reference(scratch: Path) -> Path:
    digest = pack.projection.sha(SOURCE)
    output = scratch / "build" / digest[:16]
    executable = output / "vtp24_original_bf16_ffn_cpu"
    if executable.exists():
        if pack.projection.sha(executable) != (output / "binary.sha256").read_text().strip():
            raise ValueError("original FFN CPU binary altered")
        return executable
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["g++", "-O3", "-std=c++17", "-Wall", "-Wextra",
         str(SOURCE), "-o", str(executable)],
        check=True,
    )
    (output / "binary.sha256").write_text(pack.projection.sha(executable) + "\n")
    return executable


def run(packet: Path, destination: Path, scratch: Path) -> dict:
    source = validate(packet)
    output = destination.expanduser().resolve()
    if output.exists() or output == ROOT or ROOT in output.parents:
        raise ValueError("original FFN CPU output must be fresh and outside checkout")
    binary = compile_reference(scratch.expanduser().resolve())
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA, "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": pack.projection.sha(Path(__file__)),
        "original_checkpoint_packet": str(packet.resolve()),
        "original_checkpoint_packet_sha256": pack.projection.sha(packet),
        "source_sha256": pack.projection.sha(SOURCE),
        "compiled_binary_sha256": pack.projection.sha(binary),
        "layer": source["layer"], "rows": source["rows"], "outputs": {},
        "upstream_GDN_output_is_CPU_reference": True,
        "not_GPU": True, "not_full_model": True,
    }
    receipt = output / "summary.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    try:
        proc = subprocess.run(
            [str(binary), "--rows", str(source["rows"]),
             "--packet-dir", str(packet.resolve().parent),
             "--out-dir", str(output)],
            capture_output=True, text=True, check=True,
        )
        if "status=ORIGINAL_BF16_QWEN3_8_LAYER0_FULL_FFN_CPU_ORACLE_PASS" not in proc.stdout:
            raise ValueError("original FFN CPU oracle did not announce PASS")
        result["stdout"] = proc.stdout.strip()
        for role, width in zip(pack.ROLES, pack.WIDTHS, strict=True):
            for name, data_width, bytes_per in (
                ("gate", width, 2), ("up", width, 2),
                ("mid", width, 2), ("partial", 5120, 4),
            ):
                ext = "f32" if bytes_per == 4 else "bf16"
                file = output / f"{name}-{role}.{ext}"
                count = source["rows"] * data_width * bytes_per
                if file.stat().st_size != count:
                    raise ValueError(f"bad FFN intermediate extent: {role}/{name}")
                result["outputs"][f"{name}-{role}"] = {
                    "file": str(file), "shape": [source["rows"], data_width],
                    "bytes": count, "sha256": pack.projection.sha(file),
                }
        for name, bytes_per in (
            ("original-ffn-reduced.f32", 4),
            ("original-layer0-post-ffn.bf16", 2),
        ):
            file = output / name
            count = source["rows"] * pack.HIDDEN * bytes_per
            if file.stat().st_size != count:
                raise ValueError(f"bad original FFN complete layer output: {name}")
            result["outputs"][name] = {
                "file": str(file), "shape": [source["rows"], pack.HIDDEN],
                "bytes": count, "sha256": pack.projection.sha(file),
            }
        result["status"] = "ORIGINAL_BF16_LAYER0_FFN_RESIDUAL_CPU_ORACLE_PASS_NOT_GPU"
    except Exception as exc:
        result.update({"status": "FAILED_INCOMPLETE", "error": str(exc)})
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-receipt", type=Path, required=True)
    p.add_argument("--phase", choices=("plan", "cpu"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.phase == "cpu" and not args.run:
        p.error("original FFN source must be explicitly run with --run")
    try:
        info = validate(args.packet_receipt)
        if args.phase == "plan":
            print(json.dumps({
                "schema": SCHEMA, "phase": "card_free",
                "rows": info["rows"], "layer": info["layer"],
                "source_sha256": pack.projection.sha(SOURCE),
                "checkpoint_packet_sha256": pack.projection.sha(args.packet_receipt),
                "full_model": False, "GPU_required": False,
            }, indent=2))
            return 0
        destination = args.output or args.scratch / datetime.now(UTC).strftime(
            f"layer0-m{info['rows']}-%Y%m%dT%H%M%S%fZ"
        )
        record = run(args.packet_receipt, destination, args.scratch)
        print(json.dumps({
            "status": record["status"],
            "receipt": str(destination.resolve() / "summary.json"),
            "oracle": record["stdout"],
        }, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as e:
        p.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
