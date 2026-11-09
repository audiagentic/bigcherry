"""XDP04 — native gfx1100 W4A4 experiment (isolated; never patches serving).

Default --phase plan touches neither GPUs nor build. --phase cpu runs a
deterministic, card-free per-group activation/weight quantization experiment.
--phase compile requires --run and builds one standalone HIP file in scratch.
--phase gpu requires --run and acquires the global GPU lease *around* a standalone
native IU4 WMMA + exact integer-oracle test; failure is never a speed score.

No variant is admitted into the model, no runtime kernel is registered, no
model quality claim is made. GPU kernel timing excludes activation quantization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("xtx27_w4a4_probe.hip")
TILED_SOURCE = Path(__file__).with_name("xtx27_w4a4_tiled.hip")
GROUP = 128
FORMATS = ("normal", "outlier")
DEFAULT_SCRATCH = (
    Path(os.environ.get("BC_WORK_ROOT", str(ROOT.parent))) / "xtx27-prefill" / "w4a4"
)


def pack_nibbles(codes: list[int], *, signed: bool) -> bytes:
    if len(codes) % 2:
        raise ValueError("nibble count must be even")
    lo, hi = (-8, 7) if signed else (0, 15)
    if any(
        isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi
        for v in codes
    ):
        raise ValueError(f"nibble code must be in [{lo},{hi}]")
    return bytes(
        (codes[i] & 15) | ((codes[i + 1] & 15) << 4) for i in range(0, len(codes), 2)
    )


def unpack_nibbles(data: bytes, *, signed: bool) -> list[int]:
    vals = [(byte >> bit) & 15 for byte in data for bit in (0, 4)]
    return [v - 16 if signed and v >= 8 else v for v in vals]


def quant_activation(
    row: list[float], *, bits: int = 4
) -> tuple[list[int], list[float], list[int]]:
    if bits not in (4, 8) or not row or len(row) % GROUP:
        raise ValueError("activation row must span full groups and bits=4 or 8")
    if any(not math.isfinite(v) for v in row):
        raise ValueError("activation has non-finite values")
    max_code = (1 << (bits - 1)) - 1
    codes: list[int] = []
    scales: list[float] = []
    sums: list[int] = []
    for start in range(0, len(row), GROUP):
        block = row[start : start + GROUP]
        scale = max(max(abs(v) for v in block) / max_code, 1e-12)
        q = [
            max(
                -max_code,
                min(
                    max_code,
                    math.floor(v / scale + 0.5)
                    if v >= 0
                    else -math.floor(-v / scale + 0.5),
                ),
            )
            for v in block
        ]
        codes.extend(q)
        scales.append(scale)
        sums.append(sum(q))
    return codes, scales, sums


def quant_weight(row: list[float]) -> tuple[list[int], list[float], list[int]]:
    if not row or len(row) % GROUP or any(not math.isfinite(v) for v in row):
        raise ValueError("weights must be finite and span full 128 columns")
    codes: list[int] = []
    scales: list[float] = []
    zero_points: list[int] = []
    for start in range(0, len(row), GROUP):
        block = row[start : start + GROUP]
        low, high = min(0.0, min(block)), max(0.0, max(block))
        scale = max((high - low) / 15, 1e-12)
        zero = max(0, min(15, math.floor(-low / scale + 0.5)))
        q = [max(0, min(15, math.floor(v / scale + zero + 0.5))) for v in block]
        codes.extend(q)
        scales.append(scale)
        zero_points.append(zero)
    return codes, scales, zero_points


def quant_matmul(
    a: list[list[float]], w: list[list[float]], *, activation_bits: int
) -> list[list[float]]:
    if not a or not w or not a[0] or any(len(row) != len(a[0]) for row in a + w):
        raise ValueError("matrices must have matching, rectangular K")
    aq = [quant_activation(row, bits=activation_bits) for row in a]
    wq = [quant_weight(row) for row in w]
    groups = len(a[0]) // GROUP
    result = []
    for ac, ascales, asums in aq:
        out = []
        for wc, wscales, zeros in wq:
            total = 0.0
            for g in range(groups):
                base = g * GROUP
                dot = sum(ac[i] * wc[i] for i in range(base, base + GROUP))
                # Exact INT32 arithmetic must finish before applying scales.
                corrected = dot - zeros[g] * asums[g]
                total += corrected * ascales[g] * wscales[g]
            out.append(total)
        result.append(out)
    return result


def float_matmul(a: list[list[float]], w: list[list[float]]) -> list[list[float]]:
    return [
        [sum(x * y for x, y in zip(row, weights, strict=True)) for weights in w]
        for row in a
    ]


def accuracy(approx: list[list[float]], ref: list[list[float]]) -> dict:
    x = [v for row in approx for v in row]
    y = [v for row in ref for v in row]
    if not x or len(x) != len(y):
        raise ValueError("different output shapes")
    error2 = sum((v - t) ** 2 for v, t in zip(x, y, strict=True))
    ref2 = sum(v**2 for v in y)
    return {
        "relative_l2": round(math.sqrt(error2 / max(ref2, 1e-30)), 6),
        "max_abs_error": round(max(abs(v - t) for v, t in zip(x, y, strict=True)), 6),
        "top1_row_agreement": round(
            sum(
                max(range(len(row)), key=row.__getitem__)
                == max(range(len(truth)), key=truth.__getitem__)
                for row, truth in zip(approx, ref, strict=True)
            )
            / len(ref),
            6,
        ),
    }


def cpu_probe(*, m: int, n: int, k: int, seed: int, distribution: str) -> dict:
    if not (
        1 <= m <= 256
        and 1 <= n <= 512
        and k >= GROUP
        and k % GROUP == 0
        and k <= 8192
        and m * n * k <= 10_000_000
    ):
        raise ValueError("CPU shape outside bounded 128-group experiment limits")
    if distribution not in FORMATS:
        raise ValueError("unknown distribution")
    rng = random.Random(seed)

    def sample(row: int, i: int, *, activation: bool) -> float:
        value = rng.gauss(0, 0.6 if activation else 0.3)
        if (
            distribution == "outlier"
            and (row * k + i) % (53 if activation else 103) == 0
        ):
            value *= 14
        return value

    a = [[sample(r, i, activation=True) for i in range(k)] for r in range(m)]
    w = [[sample(r, i, activation=False) for i in range(k)] for r in range(n)]
    reference = float_matmul(a, w)
    w4a4 = quant_matmul(a, w, activation_bits=4)
    w4a8 = quant_matmul(a, w, activation_bits=8)
    # Verify pack/unpack of signed and unsigned values on actual quantized rows.
    a_code, _, _ = quant_activation(a[0], bits=4)
    w_code, _, _ = quant_weight(w[0])
    assert unpack_nibbles(pack_nibbles(a_code, signed=True), signed=True) == a_code
    assert unpack_nibbles(pack_nibbles(w_code, signed=False), signed=False) == w_code
    return {
        "schema": "xtx27.w4a4.cpu.v1",
        "status": "SYNTHETIC_NUMERICS_ONLY",
        "distribution": distribution,
        "seed": seed,
        "shape": {"M": m, "N": n, "K": k, "group": GROUP},
        "w4a4_vs_float": accuracy(w4a4, reference),
        "w4a8_vs_float": accuracy(w4a8, reference),
        "limitations": [
            "synthetic matrices, not real model activations or layers",
            "no BF16-to-final-model KL, no positional/context/error propagation",
            "CPU code arithmetic is independent of GPU native WMMA instruction mapping",
        ],
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binaries(output_dir: Path, kernel: str = "oracle") -> tuple[Path, Path]:
    if kernel == "tiled":
        return (
            output_dir / "gfx1100-w4a4-tiled",
            output_dir / "w4a4-tiled-manifest.json",
        )
    if kernel != "oracle":
        raise ValueError("unknown kernel")
    return (output_dir / "gfx1100-w4a4-probe", output_dir / "w4a4-manifest.json")


def source_for(kernel: str) -> Path:
    return TILED_SOURCE if kernel == "tiled" else SOURCE


def command_lines(args: argparse.Namespace) -> dict:
    binary, manifest = binaries(args.output_dir, args.kernel)
    source = source_for(args.kernel)
    hipcc = Path(args.hipcc)
    return {
        "compiler": [
            str(hipcc),
            "-O2",
            "-std=c++17",
            "--offload-arch=gfx1100",
            str(source),
            "-o",
            str(binary),
        ],
        "gpu": [
            "agent-lease",
            "acquire",
            "gpu",
            "--ttl",
            "30m",
            "--purpose",
            "XDP05 tiled gfx1100 W4A4"
            if args.kernel == "tiled"
            else "XDP04 isolated gfx1100 native W4A4 WMMA correctness",
            "--",
            str(binary),
            *(
                ["--shape-27b", str(args.tile_m)]
                if args.shape_27b
                else ["--suite"]
                if args.suite
                else [
                    str(args.m),
                    str(args.n),
                    str(args.k),
                    str(args.iters),
                    str(args.seed),
                    *([str(args.tile_m)] if args.kernel == "tiled" else []),
                ]
            ),
        ],
        "binary": str(binary),
        "manifest": str(manifest),
    }


def check_output_dir(output_dir: Path) -> None:
    if output_dir.resolve().is_relative_to(ROOT) or output_dir.resolve().is_relative_to(
        ROOT.parent.parent / "models"
    ):
        raise ValueError("outputs must stay outside the checkout and model store")


def execute(args: argparse.Namespace) -> dict:
    check_output_dir(args.output_dir)
    commands = command_lines(args)
    result = {
        "schema": "xtx27.w4a4.execution.v1",
        "phase": args.phase,
        "git_root": str(ROOT),
        "source_sha256": sha256_file(source_for(args.kernel)),
        "output_dir": str(args.output_dir),
        "commands": commands,
        "does_not_patch_production": True,
        "native_gfx1100_iu4": True,
        "kernel": args.kernel,
        "tile_m": args.tile_m if args.kernel == "tiled" else None,
        "prepacked_operands": True,
        "gpu_activation_quantization_implemented": False,
        "algorithm": "A4 signed x W4 unsigned, asymmetric W zero correction, group128",
        "limits": "kernel-only timings exclude quantizer, host transfers, TP and serving",
    }
    if args.phase == "plan":
        return result
    if args.phase == "cpu":
        result["numeric"] = cpu_probe(
            m=args.m, n=args.n, k=args.k, seed=args.seed, distribution=args.distribution
        )
        return result
    if not args.run:
        raise ValueError("--run required for compile/gpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary, manifest = binaries(args.output_dir, args.kernel)
    if args.phase == "compile":
        if shutil.which(args.hipcc) is None and not Path(args.hipcc).is_file():
            raise ValueError("HIP compiler not available")
        log = args.output_dir / "build.log"
        with log.open("w") as stream:
            completed = subprocess.run(
                commands["compiler"],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=900,
            )
        if completed.returncode != 0 or not binary.is_file():
            raise RuntimeError(
                f"HIP compile failed (exit {completed.returncode}): {log}"
            )
        record = {
            "schema": "xtx27.w4a4.compile.v1",
            "created_at": datetime.now(UTC).isoformat(),
            "source_sha256": result["source_sha256"],
            "binary_sha256": sha256_file(binary),
            "compiler_command": commands["compiler"],
        }
        manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        result["compile"] = record
        return result
    if not binary.is_file() or not manifest.is_file():
        raise ValueError("compile the standalone probe before GPU test")
    record = json.loads(manifest.read_text())
    if record.get("source_sha256") != result["source_sha256"] or record.get(
        "binary_sha256"
    ) != sha256_file(binary):
        raise ValueError("probe or binary has changed since compile; rebuild first")
    if args.device < 0:
        raise ValueError("device must be a nonnegative inventory index")
    if args.kernel == "tiled":
        logfile = (
            f"gpu-tiled-27b-tile{args.tile_m}.log"
            if args.shape_27b
            else "gpu-tiled-suite.log"
            if args.suite
            else f"gpu-tiled-M{args.m}-N{args.n}-K{args.k}-tile{args.tile_m}-seed{args.seed}.log"
        )
    else:
        logfile = (
            "gpu-suite.log"
            if args.suite
            else f"gpu-M{args.m}-N{args.n}-K{args.k}-seed{args.seed}.log"
        )
    log = args.output_dir / logfile
    environment = os.environ.copy()
    environment["ROCR_VISIBLE_DEVICES"] = str(args.device)
    with log.open("w") as stream:
        completed = subprocess.run(
            commands["gpu"],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1200,
        )
    contents = log.read_text()
    marker = (
        ("XDP05_TILED_SUITE_PASS" if args.suite else "XDP05_TILED_PASS")
        if args.kernel == "tiled"
        else ("XDP04_GPU_SUITE_PASS" if args.suite else "XDP04_GPU_PASS")
    )
    if completed.returncode != 0 or marker not in contents:
        raise RuntimeError(
            f"GPU oracle/lease failed (exit {completed.returncode}): {log}"
        )
    result["gpu"] = {
        "status": (
            "PASS_EXPERIMENTAL_TILED_ORACLE_ONLY"
            if args.kernel == "tiled"
            else "PASS_EXPERIMENTAL_ORACLE_ONLY"
        ),
        "log": str(log),
        "device": args.device,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("plan", "cpu", "compile", "gpu"), default="plan"
    )
    parser.add_argument(
        "--run", action="store_true", help="allow compiler or GPU; default no hardware"
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="GPU oracle cases in one lease (six untiled, 27 tiled variants)",
    )
    parser.add_argument(
        "--kernel",
        choices=("oracle", "tiled"),
        default="oracle",
        help="standalone kernel (oracle default preserves XDP04)",
    )
    parser.add_argument(
        "--shape-27b",
        action="store_true",
        help="tiled M512/N17408/K5120, 512+ sampled CPU oracle cells",
    )
    parser.add_argument(
        "--tile-m",
        choices=(2, 4, 8),
        type=int,
        default=2,
        help="tiled wave geometry: 2x4, 4x2 or 8x1 (M waves x N waves)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ROCR_VISIBLE_DEVICES gfx1100 inventory index",
    )
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--distribution", choices=FORMATS, default="normal")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    try:
        if (
            args.m <= 0
            or args.n <= 0
            or args.k <= 0
            or args.iters < 1
            or args.iters > 10000
        ):
            raise ValueError("positive dimensions and 1..10000 iterations required")
        if args.kernel == "oracle" and args.tile_m != 2:
            raise ValueError("--tile-m only applies to tiled probe")
        if args.suite and args.phase not in ("plan", "gpu"):
            raise ValueError("--suite is for plan/gpu only")
        if args.shape_27b and (
            args.kernel != "tiled" or args.phase not in ("plan", "gpu") or args.suite
        ):
            raise ValueError("--shape-27b requires tiled plan/gpu, not --suite")
        if (
            args.phase == "gpu"
            and not args.suite
            and not args.shape_27b
            and args.kernel == "oracle"
            and (args.m % 16 or args.n % 16 or args.k % GROUP)
        ):
            raise ValueError(
                "GPU probe requires M/N multiples of 16, K multiple of 128"
            )
        report = execute(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
