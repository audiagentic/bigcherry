"""Independent GPU profile screen; no changes to the power-control commands."""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from ..hardware import ai_probe
from ..core import paths, workspace, run_manifest as rm
from ..core.context import resolve_work_root


def register(sub) -> None:
    p = sub.add_parser("gpu-ai-screen", help="leased AI arithmetic and memory screen of a named GPU power profile")
    p.add_argument("--probe", required=True, type=Path, help="native probe built with the installed ROCm SDK")
    p.add_argument("--profiles", required=True, type=Path, help="canonical power-profiles.yaml")
    p.add_argument("--profile", required=True)
    p.add_argument("--bdf", required=True)
    p.add_argument("--arch", required=True, help="expected architecture; verified by HIP probe")
    p.add_argument("--role", required=True, help="engine or concrete helperN")
    p.add_argument("--stock-confirmed", action="store_true", help="operator has verified factory settings; never writes")
    p.add_argument("--observe-current", action="store_true", help="observe current settings, or explicitly apply a diagnostic candidate against that exact observed reference (not factory qualification)")
    p.add_argument("--apply-profile", action="store_true",
                   help="explicitly authorize temporary candidate via existing platform power transaction")
    p.add_argument("--confirm-idle", action="store_true",
                   help="confirm no other GPU users or unresolved power journals from any owner")
    p.add_argument("--reference", help="managed stock screen run ID (observed reference only in explicit research mode)")
    p.add_argument("--dtype", nargs="+", choices=ai_probe.DTYPES, default=["fp16"])
    p.add_argument("--provider", choices=ai_probe.PROVIDERS, default="hipblaslt",
                   help="explicit installed matrix library; no silent fallback")
    p.add_argument("--memory-mib", type=int, help="total allocation for memory dtype; explicit even MiB")
    p.add_argument("--headroom-mib", type=int, help="VRAM to leave free for memory dtype")
    p.add_argument("--warmup-seconds", type=int, default=0, help="additional bounded heating time per case")
    p.add_argument("--min-hotspot-c", type=int, help="required sampled temperature for entire final interval")
    p.add_argument("--max-memory-c", type=int, help="optional explicit memory temperature ceiling")
    p.add_argument("--batch", type=int, default=1, help="matrix launches per host check; every output checked on-device for batch > 1")
    p.add_argument("--n", type=int, default=768)
    p.add_argument("--seconds", type=int, required=True, help="per dtype and load pattern")
    p.add_argument("--pause-ms", nargs="+", type=int, default=[0, 2], help="0 sustained, positive burst gap")
    p.add_argument("--setup-timeout", type=int, default=60, help="extra seconds allowed for SDK setup/oracle")
    p.add_argument("--max-hotspot-c", type=int, required=True, help="explicit safe upper ceiling, bounded by available reported limits")
    p.add_argument("--work-root", type=Path, help="existing shared managed work root")
    p.set_defaults(func=cmd_screen)


def cmd_screen(args: argparse.Namespace) -> int:
    if (not ai_probe.BDF.fullmatch(args.bdf) or not ai_probe.ARCH.fullmatch(args.arch) or
            not 16 <= args.n <= 2048 or not 1 <= args.seconds <= 3600 or not 1 <= args.batch <= 256 or
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
    # Use the existing WSL01 capture run and capture/v1 harvest recipe.
    # Profile is already reserved for profiler CSV evidence.
    try:
        root = resolve_work_root(args.work_root)
        reference_id = args.reference
        args.reference_run_id = reference_id
        args.reference_digest = None
        refs = []
        inputs = []
        if reference_id:
            ref_dir = workspace.run_dir(root, "capture", reference_id)
            ref_manifest = rm.read_manifest(ref_dir)
            if (ref_manifest["run_id"] != reference_id or ref_manifest["subkind"] != "gpu-ai-screen" or
                    ref_manifest["state"] != "completed" or ref_manifest["execution"]["exit_code"] != 0 or
                    ref_manifest.get("suspect") != "diagnostic_only_not_production_qualification"):
                raise ai_probe.ScreenError("reference must be a completed managed stock screen")
            args.reference = ref_dir / (reference_id + ".power.json")
            if ref_dir.is_symlink() or args.reference.is_symlink():
                raise ai_probe.ScreenError("symlink stock receipt refused")
            args.reference_digest = ai_probe.digest(args.reference)
            ref_role = "observed" if args.observe_current else "stock"
            refs.append({"type": "run", "id": reference_id, "role": ref_role})
            inputs.append({"role": ref_role, "uri": str(args.reference), "sha256": args.reference_digest.removeprefix("sha256:")})
        for role, path in (("probe", args.probe), ("profiles", args.profiles)):
            inputs.append({"role": role, "uri": str(path.resolve()), "sha256": ai_probe.digest(path).removeprefix("sha256:")})
        args.input_digests = {i["role"]: "sha256:" + i["sha256"] for i in inputs}
        run = workspace.create_run(root, "capture", "gpu-ai-screen", plan_item="PWR12", engine="none",
                                   argv=sys.argv[:], purpose="Independent GPU power-profile screen",
                                   inputs=inputs, references=refs, subkind="gpu-ai-screen", repo=paths.REPO_ROOT)
        args.run_id = run.run_id
        args.output = run.path / (run.run_id + ".power.json")
        manifest = rm.update_manifest(run.path, expected_generation=run.manifest()["generation"], state="running")
    except (OSError, ValueError, KeyError, workspace.WorkspaceError, ai_probe.ScreenError) as exc:
        print(f"gpu-ai-screen: {exc}", file=sys.stderr)
        return 2
    interrupted = False
    def cancel(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        journal_dir = None
        if args.apply_profile:
            if root != resolve_work_root():
                raise ai_probe.ScreenError("apply requires the configured work root")
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
    # Keep diagnostic screens out of inference/calibration records; no
    # synthetic throughput or serving manifest identity. Retain negative
    # evidence and let the standard harvest/retention lifecycle manage it.
    rc = 0 if result["status"] == "pass" else 2
    try:
        ai_probe.screen_receipt(run.run_id, result).save(args.output)
        # An unresolved journal/child must stay protected from cleanup.
        hold = result.get("transaction", {}).get("state") == "restore_unverified" or result["status"] == "unreaped"
        rm.update_manifest(run.path, expected_generation=manifest["generation"],
                           state="completed" if rc == 0 else ("cancelled" if result["status"] == "cancelled" else "failed"),
                           patch={"execution": {"exit_code": rc},
                                  "suspect": "diagnostic_only_not_production_qualification",
                                  "retention": {"hold": hold}})
    except (OSError, ValueError) as exc:
        print(f"gpu-ai-screen: evidence finalization failed in {run.path}: {exc}", file=sys.stderr)
        return 2
    print(f"AI power screen: {result['status']} ({args.output})")
    return rc
