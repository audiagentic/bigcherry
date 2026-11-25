"""Independent GPU profile screen; no changes to the power-control commands."""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from ..hardware import ai_probe
from ..hardware.power_receipt import write_json_atomic


def register(sub) -> None:
    p = sub.add_parser("gpu-ai-screen", help="leased AI arithmetic and memory screen of a named GPU power profile")
    p.add_argument("--probe", required=True, type=Path, help="native probe built with the installed ROCm SDK")
    p.add_argument("--profiles", required=True, type=Path, help="canonical power-profiles.yaml")
    p.add_argument("--profile", required=True)
    p.add_argument("--bdf", required=True)
    p.add_argument("--arch", required=True, help="expected architecture; verified by HIP probe")
    p.add_argument("--role", required=True, help="engine or concrete helperN")
    p.add_argument("--stock-confirmed", action="store_true", help="operator has verified factory settings; never writes")
    p.add_argument("--observe-current", action="store_true", help="test observed settings without claiming factory reference")
    p.add_argument("--apply-profile", action="store_true",
                   help="explicitly authorize temporary candidate via existing platform power transaction")
    p.add_argument("--confirm-idle", action="store_true",
                   help="confirm no other GPU users or unresolved power journals from any owner")
    p.add_argument("--reference", type=Path, help="passing stock screen for candidate")
    p.add_argument("--dtype", nargs="+", choices=ai_probe.DTYPES, default=["fp16"])
    p.add_argument("--provider", choices=ai_probe.PROVIDERS, default="hipblaslt",
                   help="explicit installed matrix library; no silent fallback")
    p.add_argument("--memory-mib", type=int, help="total allocation for memory dtype; explicit even MiB")
    p.add_argument("--headroom-mib", type=int, help="VRAM to leave free for memory dtype")
    p.add_argument("--warmup-seconds", type=int, default=0, help="additional bounded heating time per case")
    p.add_argument("--min-hotspot-c", type=int, help="required sampled temperature for entire final interval")
    p.add_argument("--max-memory-c", type=int, help="optional explicit memory temperature ceiling")
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
            not 0 <= args.warmup_seconds <= 3600 - args.seconds or
            (args.min_hotspot_c is not None and not 1 <= args.min_hotspot_c < args.max_hotspot_c) or
            (args.warmup_seconds and args.min_hotspot_c is None) or
            (args.max_memory_c is not None and not 1 <= args.max_memory_c <= 110) or
            ("memory" in args.dtype and (args.memory_mib is None or not 2 <= args.memory_mib <= 65536 or
                                        args.memory_mib % 2 or args.headroom_mib is None or
                                        not 256 <= args.headroom_mib <= 65536)) or
            ("memory" not in args.dtype and (args.memory_mib is not None or args.headroom_mib is not None)) or
            ("fp8" in args.dtype and args.provider == "rocblas") or
            not all(0 <= x <= 1000 for x in args.pause_ms) or
            len(set(args.dtype)) != len(args.dtype) or len(set(args.pause_ms)) != len(args.pause_ms)):
        print("gpu-ai-screen: invalid device, bounds or duplicate test cases", file=sys.stderr)
        return 2
    if args.output.exists() or args.output.is_symlink():
        print("gpu-ai-screen: output already exists; use a new evidence path", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    interrupted = False
    def cancel(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        journal_dir = None
        if args.apply_profile:
            # Use the same recovery directory as the existing gpu-power CLI.
            from .power_cmds import _journal_dir
            journal_dir = _journal_dir()
        result = ai_probe.screen(args, cancelled=lambda: interrupted, journal_dir=journal_dir)
    except (ai_probe.ScreenError, OSError, ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
        result = {"kind": "ai-power-screen", "schema_version": 2,
                  "status": getattr(exc, "status", "inconclusive"), "reason": str(exc),
                  "production_qualified": False}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    write_json_atomic(args.output, result)
    print(f"AI power screen: {result['status']} ({args.output})")
    return 0 if result["status"] == "pass" else 2
