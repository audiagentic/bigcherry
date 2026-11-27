"""Isolated IQ6R GPTQ SIGSEGV reproducer with gdb backtrace capture.

Reuses a read-only W8A8 RAD, changing only blk.23.attn_v to calibrated
IQ6R-V4. Invoke under an exclusive agent-lease GPU lock. Diagnostic outputs
are not qualified deployable artifacts. Never use this on production paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

TARGET = "blk.23.attn_v.weight"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "xtx2", "dual", "r9700", "cpu"), required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 60 <= args.max_seconds <= 3600:
        parser.error("max-seconds must be between 60 and 3600")
    root = Path(os.environ["BC_WORK_ROOT"]) / "iq6r-v4-qualification"
    source = Path(os.environ["BC_RADIANCE_SOURCE"])
    model = Path(os.environ["BC_MODEL_ROOT"]) / "qwen3.8-27b/Qwen3.8-27B"
    release = json.loads(Path(os.environ["BC_RELEASE_MANIFEST"]).read_text())
    if release.get("tag") != "bc-radiance-1170.26.0" or release.get("source") != "ci":
        raise RuntimeError("not using the selected managed production build")
    build = Path(release["build_dir"])
    if sha256(build / "bin/radiance") != release["binary_sha256"]:
        raise RuntimeError("managed runtime hash mismatch")
    lease = subprocess.run(["agent-lease", "status"], capture_output=True,
                           text=True, check=True).stdout
    if "gpu: held" not in lease or "IQ6R" not in lease:
        raise RuntimeError("exclusive IQ6R GPU lease not held")
    audit = json.loads((root / "gram-sample-audit.json").read_text())
    if audit.get("status") != "SAMPLED_GRAMS_VERIFIED" or audit.get("sources") != 257:
        raise RuntimeError("full independent calibration sample audit missing")
    recipe = root / "crash-focus-blk23.recipe"
    exact = next(
        line for line in (source / "data/recipes/q38-27b-iq6r-v4.recipe").read_text().splitlines()
        if line.startswith(TARGET + " ")
    )
    expected_recipe = ("# diagnostic only: W8A8 baseline plus one calibrated IQ6R projection\n"
                       + exact + "\n"
                       + (source / "data/recipes/q38-27b-w8a8-df2.recipe").read_text())
    if recipe.exists() and recipe.read_text() != expected_recipe:
        raise RuntimeError("diagnostic recipe has drifted from the pinned sources")
    if not recipe.exists():
        recipe.write_text(expected_recipe)
    if (root / "conversion-status-managed.json").is_file():
        failed = json.loads((root / "conversion-status-managed.json").read_text())
        if failed.get("stage") != "failed_no_production_change":
            raise RuntimeError("full conversion failure receipt not finalized")
    output = root / f"crash-focus-{args.mode}-unqualified.rad"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    baseline = Path(os.environ["BC_W8A8_RAD"])
    if not baseline.is_file():
        raise RuntimeError("W8A8 baseline is missing")
    cmd = [
        str(build / "bin/rad-convert"), str(model), "-o", str(output),
        "--home", str(build / "radiance_home"),
        "--kernels", "libr4d,libref" if args.mode == "r9700" else "libr3,libref",
        "--recipe", str(recipe), "--reuse", str(baseline),
        "--set", "quantization_config.quant_method=w8a8",
        "--set", "quantization_config.fmt=fwht128",
        "--report", "1", "-v",
    ]
    env = os.environ.copy()
    env["CALIB"] = str(root / "calib-radgram2")
    if args.mode != "cpu":
        env["HIP_VISIBLE_DEVICES"] = {
            "dual": "0,1", "single": "0", "xtx2": "1", "r9700": "2"
        }[args.mode]
        # --gpus uses ordinals within HIP_VISIBLE_DEVICES, not host IDs.
        cmd += ["--gpus", "0,1" if args.mode == "dual" else "0"]
    gdb = [
        "gdb", "-batch",
        "-ex", "set pagination off",
        "-ex", "set confirm off",
        "-ex", "set print thread-events off",
        "-ex", "run",
        "-ex", "thread apply all bt 12",
        "--args", *cmd,
    ]
    receipt = root / f"crash-focus-{args.mode}-receipt.json"
    log = root / f"crash-focus-{args.mode}-gdb.log"
    start = time.monotonic()
    state = {
        "mode": args.mode, "release": release["tag"],
        "runtime_sha256": release["binary_sha256"],
        "converter_sha256": sha256(build / "bin/rad-convert"),
        "baseline_path": str(baseline), "diagnostic_recipe_sha256": sha256(recipe),
        "status": "running_unqualified_probe", "command": cmd,
        "gdb_log": str(log), "output": str(output),
    }
    receipt.write_text(json.dumps(state, indent=2) + "\n")
    with log.open("w") as logf:
        try:
            result = subprocess.run(gdb, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                    timeout=args.max_seconds, check=False)
            state["gdb_exit_code"] = result.returncode
            log_content = log.read_text(errors="replace")
            if "Program received signal SIGSEGV" in log_content:
                state["status"] = "sigsegv_backtrace_captured"
            elif "exited normally" in log_content:
                state["status"] = "diagnostic_completed_unqualified"
            else:
                state["status"] = "gdb_exited_without_classified_result"
        except subprocess.TimeoutExpired:
            state["status"] = "gdb_timed_out_unqualified"
    state["elapsed_seconds"] = round(time.monotonic() - start, 2)
    receipt.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"status": state["status"], "elapsed": state["elapsed_seconds"],
                      "gdb_exit": state.get("gdb_exit_code"), "log": str(log)}))
    return 0 if state["status"] == "diagnostic_completed_unqualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
