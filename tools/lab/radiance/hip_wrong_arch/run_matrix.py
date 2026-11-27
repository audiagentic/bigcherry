"""Isolated ROCm HIP wrong-architecture A/B. Require exclusive GPU lease.

Invoke through agent-lease. Builds are created separately from their SDK roots;
this runner launches only already-compiled binaries, matching each runtime
through its own embedded RUNPATH and explicit LD_LIBRARY_PATH.
No production binaries, global environment or GPU settings are changed.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ["BC_WORK_ROOT"]) / "iq6r-v4-qualification/hip-wrong-arch"
SDK = {
    "7.2.4": Path("/opt/rocm-7.2.4"),
    "10.1rc2": Path(__file__).resolve().parents[4] / "vendor/rocm/10.1.0rc2",
}
GPUS = {"r9700": "2", "xtx0": "0", "xtx1": "1"}


def main() -> int:
    if "gpu: held" not in subprocess.check_output(
        ["agent-lease", "status"], text=True
    ):
        raise RuntimeError("exclusive GPU lease not held")
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "wrong-arch launch error behavior only, not quantizer compatibility",
        "compiled_arch": "gfx1201",
        "results": [],
    }
    output = ROOT / "matrix-results.json"
    for version, sdk in SDK.items():
        binary = ROOT / f"probe-{version}"
        if not binary.is_file():
            raise FileNotFoundError(binary)
        for name, physical_index in GPUS.items():
            env = os.environ.copy()
            env["HIP_VISIBLE_DEVICES"] = physical_index
            env.pop("ROCR_VISIBLE_DEVICES", None)
            env["LD_LIBRARY_PATH"] = str(sdk / "lib")
            env["LD_DEBUG"] = "libs"
            started = time.monotonic()
            try:
                result = subprocess.run(
                    [str(binary)],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=50,
                    check=False,
                )
                row = {
                    "sdk": version,
                    "device": name,
                    "physical_index": physical_index,
                    "exit_code": result.returncode,
                    "stdout": result.stdout[-4000:],
                    "stderr_tail": result.stderr[-1500:],
                    "runtime_loader_mentions_expected_path": str(
                        sdk / "lib/libamdhip64.so.7"
                    ) in result.stderr,
                    "elapsed_s": round(time.monotonic() - started, 2),
                }
            except subprocess.TimeoutExpired:
                row = {"sdk": version, "device": name, "exit_code": "timeout"}
            report["results"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                version, name, "exit=", row["exit_code"],
                "stdout=", row.get("stdout", "").replace("\n", "; ")[:400],
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
