"""VTP04 isolated HIP microbench driver: dry-run by default, no model required.

--phase plan is card-free. --phase compile --run builds only in scratch.
--phase gpu --run compiles as necessary then acquires the exclusive BigCherry
GPU lease before executing local R9700 and three-physical-GPU probe binaries.

This measures scheduling and physical no-P2P host transport, NOT native Q6/Q8
GEMMs, graph replay, MTP quality, or a production virtual-TP4 runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    "local": Path(__file__).with_name("xtx27_vtp_local.hip"),
    "host_collective": Path(__file__).with_name("xtx27_vtp3_host_collective.hip"),
}
BUILD_FLAGS = ("-O3", "-std=c++17", "--offload-arch=gfx1100", "--offload-arch=gfx1201")
DEFAULT_SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "virtual-tp4"


def csv_positive(text: str, *, upper: int, name: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(,\d+)*", text):
        raise ValueError(f"{name}: expected comma-separated positive integers")
    vals = tuple(int(v) for v in text.split(","))
    if any(v < 1 or v > upper for v in vals) or len(set(vals)) != len(vals):
        raise ValueError(f"{name}: non-unique or out-of-range values (1..{upper})")
    return vals


def compiler_path(configured: str | None) -> Path | None:
    if configured:
        p = Path(configured).expanduser()
        if not p.is_file():
            raise ValueError("HIP compiler path not found")
        return p.resolve()
    root = os.environ.get("ROCM_PATH")
    if root and (Path(root) / "bin" / "hipcc").is_file():
        return (Path(root) / "bin" / "hipcc").resolve()
    found = shutil.which("hipcc")
    return Path(found).resolve() if found else None


def source_digest(sources: dict[str, Path] | None = None) -> str:
    if sources is None:
        sources = SOURCES
    h = hashlib.sha256()
    for name, src in sorted(sources.items()):
        h.update(name.encode())
        h.update(src.read_bytes())
    h.update(" ".join(BUILD_FLAGS).encode())
    return h.hexdigest()


def plan(*, mib: tuple[int, ...], rows: tuple[int, ...], lanes: int, iterations: int,
         reverse: bool = False, physical_tp3: bool = False) -> list[dict]:
    if lanes < 256 or lanes > 262144 or lanes % 256:
        raise ValueError("lanes must be a multiple of 256 in [256,262144]")
    if not 1 <= iterations <= 100:
        raise ValueError("iterations must be in [1,100]")
    result = []
    for v in mib:
        result.append({
            "kind": "local", "args": [
                "--mib-per-shard", str(v), "--lanes", str(lanes),
                "--iterations", str(iterations),
            ],
            "label": f"local-{v}mib-{lanes}lanes",
            "data_definition": "two independent resident UINT32 weight scans; no Q6/Q8 GEMM",
        })
    for v in rows:
        result.append({
            "kind": "host_collective", "args": [
                "--rows", str(v), "--iterations", str(iterations),
            ],
            "label": f"host-f32-{v}rows",
            "data_definition": "physical 3-GPU host-staged F32 collective, width=5120, no P2P",
        })
    if physical_tp3:
        for v in rows:
            result.append({
                "kind": "host_collective",
                "args": ["--rows", str(v), "--iterations", str(iterations), "--single-r9700"],
                "label": f"host-f32-physical-tp3-{v}rows",
                "data_definition": "3 physical GPU F32 contributions, one R9700 rank, no P2P",
            })
    if reverse:
        for case in result:
            case["args"].append("--reverse")
    return result


def build_binaries(hipcc: Path, scratch: Path, digest: str,
                   sources: dict[str, Path] | None = None) -> dict[str, Path]:
    if sources is None:
        sources = SOURCES
    out = scratch / "build" / digest[:16]
    out.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([str(hipcc), "--version"], check=True, text=True, capture_output=True).stdout
    manifest = {
        "source_digest": digest,
        "compiler": str(hipcc),
        "compiler_version": version.strip(),
        "flags": list(BUILD_FLAGS),
        "sources": {name: hashlib.sha256(src.read_bytes()).hexdigest()
                    for name, src in sources.items()},
    }
    receipt_path = out / "build-manifest.json"
    binaries = {kind: out / f"vtp04-{kind}" for kind in sources}
    reusable = False
    if receipt_path.exists() and all(p.is_file() for p in binaries.values()):
        try:
            reusable = json.loads(receipt_path.read_text()) == manifest
        except (OSError, json.JSONDecodeError):
            reusable = False
    if not reusable:
        for kind, src in sources.items():
            binary = binaries[kind]
            temp = out / f".{binary.name}.new"
            result = subprocess.run(
                [str(hipcc), *BUILD_FLAGS, str(src), "-o", str(temp)],
                text=True, capture_output=True, check=False,
            )
            (out / f"{kind}-compile.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                temp.unlink(missing_ok=True)
                raise RuntimeError(f"{kind} HIP compile failed (see {out / (kind + '-compile.log')})")
            temp.replace(binary)
        receipt_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return binaries


def run_leased(
    *, scratch: Path, binaries: dict[str, Path], cases: list[dict], digest: str,
    purpose: str = "VTP04: isolated virtual TP4 local/host microbench",
    receipt_type: str = "VTP04_THREE_PHYSICAL_GPU_MICROBENCH",
) -> Path:
    if os.environ.get("VTP04_LEASED") != "1":
        raise RuntimeError("VTP04 GPU runner requires the exclusive lease wrapper")
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL"):
        if key in os.environ:
            raise RuntimeError(f"refusing inherited GPU mask: {key}")
    lease = subprocess.run(["agent-lease", "status"], check=True, text=True,
                           capture_output=True, timeout=8).stdout
    if f'for "{purpose}"' not in lease:
        raise RuntimeError("VTP04 global GPU lease not owned by this pilot")
    show = subprocess.run(["rocm-smi", "--showpids"], check=True, text=True,
                          capture_output=True, timeout=8).stdout
    if "No KFD PIDs currently running" not in show:
        raise RuntimeError("other KFD processes already resident")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = scratch / "runs" / f"{stamp}-{digest[:10]}"
    out.mkdir(parents=True, exist_ok=False)
    entries: list[dict] = []
    receipt = {
        "status": "incomplete",
        "type": receipt_type,
        "checkout": None,
        "source_digest": digest,
        "not_a_model_benchmark": True,
        "cases": entries,
    }
    try:
        checkout = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, text=True, capture_output=True,
        )
        receipt["checkout"] = checkout.stdout.strip()
        for case in cases:
            command = [str(binaries[case["kind"]]), *case["args"]]
            run = subprocess.run(command, check=False, capture_output=True, text=True)
            (out / f"{case['label']}.stdout").write_text(run.stdout)
            (out / f"{case['label']}.stderr").write_text(run.stderr)
            entries.append({
                "label": case["label"], "kind": case["kind"], "argv": command,
                "rc": run.returncode,
                "oracle_pass": run.returncode == 0 and "PASS" in run.stdout,
                "stdout": f"{case['label']}.stdout",
                "stderr": f"{case['label']}.stderr",
            })
            (out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if run.returncode or not entries[-1]["oracle_pass"]:
                raise RuntimeError(f"VTP04 case failed: {case['label']}; see {out}")
        receipt["status"] = "microbench_finished_oracle_passed_NOT_27B_QUALIFIED"
    finally:
        (out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "compile", "gpu"), default="plan")
    parser.add_argument("--run", action="store_true", help="explicit execution opt-in")
    parser.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--local-mib", default="4,16,64")
    parser.add_argument("--rows", default="1,4,64,512")
    parser.add_argument("--lanes", type=int, default=32768)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--physical-tp3", action="store_true",
                        help="add one-partial-per-physical-GPU exact F32 collective")
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--hipcc", type=str)
    args = parser.parse_args(argv)
    if args.phase != "plan" and not args.run:
        parser.error("compile/GPU phases require --run")
    try:
        mib = csv_positive(args.local_mib, upper=256, name="local-mib")
        rows = csv_positive(args.rows, upper=2048, name="rows")
        cases = plan(mib=mib, rows=rows, lanes=args.lanes, iterations=args.iterations,
                     reverse=args.reverse_order, physical_tp3=args.physical_tp3)
    except ValueError as exc:
        parser.error(str(exc))
    digest = source_digest()
    info = {
        "phase": args.phase,
        "type": "VTP04_NO_MODEL_MICROBENCH",
        "source_digest": digest,
        "cases": cases,
        "limitations": [
            "not native Q6/Q8 GEMM or numerical quality",
            "GPU launch and host-transfer microbenchmark is not end-to-end tokens/sec",
            "three physical GPU collective, not a virtual GPU adapter",
            "no context size or KV-residency validation",
        ],
    }
    if args.phase == "plan":
        print(json.dumps(info, indent=2))
        return 0
    if args._under_lease and os.environ.get("VTP04_LEASED") != "1":
        parser.error("internal leased mode requires agent-lease environment")
    hipcc = compiler_path(args.hipcc)
    if hipcc is None:
        parser.error("no ROCm HIP compiler found")
    scratch = args.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        parser.error("scratch output must be outside the checkout")
    binaries = build_binaries(hipcc, scratch, digest)
    if args.phase == "compile":
        print(json.dumps({"status": "compiled_card_free", "binaries": {k: str(v) for k, v in binaries.items()},
                          "source_digest": digest}, indent=2))
        return 0
    if not args._under_lease:
        forwarded = [sys.executable, str(Path(__file__).resolve()), "--phase", "gpu", "--run",
                     "--_under-lease", "--local-mib", args.local_mib, "--rows", args.rows,
                     "--lanes", str(args.lanes), "--iterations", str(args.iterations),
                     "--scratch", str(scratch), "--hipcc", str(hipcc),
                     *(["--reverse-order"] if args.reverse_order else []),
                     *(["--physical-tp3"] if args.physical_tp3 else [])]
        cmd = ["agent-lease", "acquire", "gpu", "--ttl", "30m",
               "--purpose", "VTP04: isolated virtual TP4 local/host microbench",
               "--", "env", "VTP04_LEASED=1", *forwarded]
        return subprocess.run(cmd, check=False).returncode
    out = run_leased(scratch=scratch, binaries=binaries, cases=cases, digest=digest)
    print(json.dumps({"status": "completed_microbench_not_model_qualification",
                      "receipt": str(out / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
