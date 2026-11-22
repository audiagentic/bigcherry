#!/usr/bin/env python3
"""Timeout-bounded opt-in GPU collective check. Never start this without a GPU lease."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from time import monotonic


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--devices", default="0,1,2",
                   help="physical GPU ordinals; use 0,0,0 with --visible 2 for R9700-only control")
    p.add_argument("--visible", default="0,1,2",
                   help="physical ordinals exposed to ROCm, in order")
    p.add_argument("--row", choices=(
        "ar_oneshot_3rank_exact_host",
        "ar_twoshot_3rank_exact_host",
        "ar_oneshot_3rank_exact",
        "ar_twoshot_3rank_exact",
    ), default="ar_oneshot_3rank_exact_host")
    p.add_argument("--timeout", type=int, default=80)
    args = p.parse_args()
    if args.timeout <= 0 or len(args.devices.split(",")) != 3:
        p.error("positive timeout and exactly three device ordinals required")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = args.visible
    env.pop("HIP_VISIBLE_DEVICES", None)
    cmd = [
        str(args.binary),
        str(args.sdk / "share/radiance/kernels/libr3.so"),
        str(args.sdk / "share/radiance/kernels/libr4d.so"),
        args.row, args.devices,
    ]
    label = f"{args.row}-devices-{args.devices.replace(chr(44), chr(45))}-visible-{args.visible.replace(chr(44), chr(45))}"
    log = args.output / f"{label}.log"
    start = monotonic()
    with log.open("w") as f:
        try:
            rc = subprocess.run(cmd, env=env, stdout=f,
                                stderr=subprocess.STDOUT,
                                timeout=args.timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    lines = log.read_text(errors="replace").splitlines()
    result = {
        "row": args.row,
        "devices": args.devices,
        "visible": args.visible,
        "return_code": rc,
        "duration_seconds": round(monotonic() - start, 2),
        "log": log.name,
        "tail": lines[-70:],
    }
    (args.output / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
