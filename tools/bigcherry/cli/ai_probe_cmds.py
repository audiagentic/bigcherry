"""Independent GPU profile screen; no changes to the power-control commands."""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from ..hardware import ai_probe
from ..hardware.power_receipt import write_json_atomic


def register(sub) -> None:
    p = sub.add_parser("gpu-ai-screen", help="read-only leased AI arithmetic screen of a named GPU power profile")
    p.add_argument("--probe", required=True, type=Path, help="native probe built with the installed ROCm SDK")
    p.add_argument("--profiles", required=True, type=Path, help="canonical power-profiles.yaml")
    p.add_argument("--profile", required=True)
    p.add_argument("--bdf", required=True)
    p.add_argument("--arch", required=True, help="expected architecture; verified by HIP probe")
    p.add_argument("--role", required=True, help="engine or concrete helperN")
    p.add_argument("--stock-confirmed", action="store_true", help="operator has verified factory settings; never writes")
    p.add_argument("--reference", type=Path, help="passing stock screen for candidate")
    p.add_argument("--dtype", nargs="+", choices=ai_probe.DTYPES, default=["fp16"])
    p.add_argument("--n", type=int, default=768)
    p.add_argument("--seconds", type=int, required=True, help="per dtype and load pattern")
    p.add_argument("--pause-ms", nargs="+", type=int, default=[0, 2], help="0 sustained, positive burst gap")
    p.add_argument("--setup-timeout", type=int, default=60, help="extra seconds allowed for SDK setup/oracle")
    p.add_argument("--max-hotspot-c", type=int, required=True, help="explicit safe upper ceiling, bounded by available reported limits")
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=cmd_screen)


def cmd_screen(args: argparse.Namespace) -> int:
    if (not ai_probe.BDF.fullmatch(args.bdf) or not ai_probe.ARCH.fullmatch(args.arch) or
            not 16 <= args.n <= 2048 or not 1 <= args.seconds <= 3600 or
            not 1 <= args.setup_timeout <= 300 or not 1 <= args.max_hotspot_c <= 110 or
            not all(0 <= x <= 1000 for x in args.pause_ms) or
            len(set(args.dtype)) != len(args.dtype) or len(set(args.pause_ms)) != len(args.pause_ms)):
        print("gpu-ai-screen: invalid device, bounds or duplicate test cases", file=sys.stderr)
        return 2
    if args.output.exists() or args.output.is_symlink():
        print("gpu-ai-screen: output already exists; use a new evidence path", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def cancelled(signum, frame):
        raise ai_probe.ScreenError("screen interrupted", "cancelled")

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        result = ai_probe.screen(args)
    except (ai_probe.ScreenError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result = {"kind": "ai-power-screen", "schema_version": 1,
                  "status": getattr(exc, "status", "inconclusive"), "reason": str(exc),
                  "production_qualified": False}
    finally:
        signal.signal(signal.SIGTERM, previous)
    write_json_atomic(args.output, result)
    print(f"AI power screen: {result['status']} ({args.output})")
    return 0 if result["status"] == "pass" else 2
