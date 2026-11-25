"""PWR12: independent, observe-only AI power-profile screen.

The existing power owner applies/restores settings. This command never writes
sysfs, changes a profile, or grants a production qualification. It consumes the
same named YAML intent, verifies actual controls on one BDF, and exercises a
locally compiled native probe under a verified GPU lease.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import lease, power_probe, power_profiles
from .power_receipt import write_json_atomic

DTYPES = ("fp16", "bf16", "fp32", "fp8", "int8")
BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
ARCH = re.compile(r"^gfx[0-9a-f]+$")


class ScreenError(RuntimeError):
    def __init__(self, reason: str, status: str = "inconclusive"):
        super().__init__(reason)
        self.status = status


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return "sha256:" + hashlib.file_digest(f, "sha256").hexdigest()


def snapshot(bdf: str) -> tuple[dict, Path]:
    card, = power_probe.probe_gpus([bdf])
    if card.driver.value != "amdgpu" or card.hwmon.status != "supported":
        raise ScreenError("selected AMD device or unique hwmon unavailable")
    hw = Path(card.hwmon.source)
    controls = {k: v.current.value if v.current.status == "supported" else None
                for k, v in card.controls.items()}
    cap = controls.get("power_cap_w")
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ScreenError("invalid power-cap readback")
    pci = Path("/sys/bus/pci/devices") / bdf
    identity = {k: power_probe.read_field(pci / k).value
                for k in ("vendor", "device", "subsystem_vendor", "subsystem_device", "unique_id")}
    default_cap = power_probe.read_int(hw / "power1_cap_default").value
    return {"identity": identity, "controls": controls, "default_cap_uw": default_cap}, hw


def verify_settings(snap: dict, settings: dict, stock: dict | None = None) -> None:
    actual = snap["controls"]
    for key, value in settings.items():
        expected = value * 1_000_000 if key == "power_cap_w" else value
        if actual.get(key) != expected:
            raise ScreenError(f"{key}: requested {expected}, observed {actual.get(key)}")
    if stock is not None:
        if snap["identity"] != stock["identity"]:
            raise ScreenError("physical device identity differs from reference")
        if {k: v for k, v in actual.items() if k not in settings} != {
                k: v for k, v in stock["controls"].items() if k not in settings}:
            raise ScreenError("controls outside selected profile differ from stock reference")
    else:
        # Stock is operator-confirmed, never silently applied/reset. The
        # observable offset/cap must agree; clock defaults are not inferred.
        if actual.get("vddgfx_offset_mv") not in (None, 0):
            raise ScreenError("nonzero voltage offset on stock arm")
        cap = actual.get("power_cap_w")
        if type(cap) is not int or cap <= 0 or snap["default_cap_uw"] != cap:
            raise ScreenError("stock cap cannot be verified")


def thermal_sensor(hw: Path, max_c: int) -> tuple[Path, int]:
    matches = [p for p in hw.glob("temp*_label")
               if str(power_probe.read_field(p).value).lower() in ("junction", "hotspot")]
    if len(matches) != 1:
        raise ScreenError("unique junction/hotspot sensor unavailable")
    stem = matches[0].name.removesuffix("_label")
    limit = max_c * 1000
    for suffix in ("_crit", "_emergency"):
        observed = power_probe.read_int(hw / (stem + suffix)).value
        if type(observed) is int and observed > 0:
            limit = min(limit, observed)
    return hw / (stem + "_input"), limit


def sample(sensor: Path, limit: int, samples: list[int]) -> None:
    value = power_probe.read_int(sensor).value
    if type(value) is not int or value <= 0:
        raise ScreenError("hotspot read unavailable")
    samples.append(value)
    if value >= limit:
        raise ScreenError("thermal ceiling reached", "thermal")


def run_case(binary: Path, bdf: str, arch: str, dtype: str, n: int,
             seconds: int, pause: int, setup_timeout: int,
             sensor: Path, limit: int, output_dir: Path) -> dict:
    """Always reap the probe before returning, including signal/thermal/lease failure."""
    samples: list[int] = []
    sample(sensor, limit, samples)
    if not lease.held_by_this_process():
        raise ScreenError("GPU lease not verified")
    with tempfile.TemporaryFile(dir=output_dir) as stdout, tempfile.TemporaryFile(dir=output_dir) as stderr:
        start = time.monotonic()
        proc = subprocess.Popen([str(binary), bdf, arch, dtype, str(n), str(seconds), str(pause)],
                                stdout=stdout, stderr=stderr)
        try:
            while proc.poll() is None:
                if not lease.held_by_this_process():
                    raise ScreenError("GPU lease lost", "lease_lost")
                sample(sensor, limit, samples)
                if time.monotonic() - start > seconds + setup_timeout:
                    raise ScreenError("probe deadline exceeded", "timeout")
                time.sleep(0.25)
            sample(sensor, limit, samples)
            if not lease.held_by_this_process():
                raise ScreenError("GPU lease lost", "lease_lost")
            failure = None
        except ScreenError as exc:
            failure = exc
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        stdout.seek(0)
        stderr.seek(0)
        raw = stdout.read().decode("utf-8", errors="replace")
        diagnostic = stderr.read().decode("utf-8", errors="replace")
    result = {"dtype": dtype, "n": n, "pause_ms": pause, "requested_seconds": seconds,
              "hotspot_min_mc": min(samples), "hotspot_max_mc": max(samples),
              "samples": len(samples), "exit": proc.returncode}
    if failure is not None:
        return result | {"status": failure.status, "diagnostic": str(failure)}
    if proc.returncode != 0:
        return result | {"status": {2: "mismatch", 3: "unavailable"}.get(proc.returncode, "device_error"),
                         "diagnostic": diagnostic[-4000:]}
    try:
        payload = json.loads(raw.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise ScreenError("probe did not emit a valid result") from None
    if (not isinstance(payload, dict) or payload.get("status") != "pass" or
            payload.get("bdf") != bdf or payload.get("arch") != arch or
            payload.get("dtype") != dtype or payload.get("n") != n or
            type(payload.get("iterations")) is not int or payload["iterations"] < 2 or
            type(payload.get("elapsed_s")) not in (int, float) or
            not seconds <= payload["elapsed_s"] < seconds + setup_timeout or
            any(type(payload.get(k)) is not int or payload[k] <= 0
                for k in ("hip", "driver", "hipblaslt")) or
            type(payload.get("workspace")) is not int or payload["workspace"] < 0 or
            not isinstance(payload.get("algorithm"), str) or
            not re.fullmatch("[0-9a-f]{32}", payload["algorithm"])):
        raise ScreenError("probe result identity or coverage mismatch")
    return result | {"status": "pass", "native": payload}


def case_identity(case: dict) -> tuple:
    native = case["native"]
    return tuple(case[k] for k in ("dtype", "n", "pause_ms", "requested_seconds")) + tuple(
        native[k] for k in ("bdf", "arch", "hip", "driver", "hipblaslt", "workspace", "algorithm"))


def screen(args: argparse.Namespace) -> dict:
    if not BDF.fullmatch(args.bdf) or not ARCH.fullmatch(args.arch):
        raise ScreenError("invalid physical device identity")
    profile = power_profiles.get_profile(power_profiles.load_profiles(args.profiles), args.profile)
    settings = profile.settings_for(args.role, args.arch)
    if args.profile == "stock":
        if args.reference or not args.stock_confirmed:
            raise ScreenError("stock arm requires --stock-confirmed and no --reference")
    elif not settings or not args.reference or args.stock_confirmed:
        raise ScreenError("candidate requires matching nonempty profile rules and --reference")
    if not lease.held_by_this_process():
        raise ScreenError("GPU lease not verified")
    binary = args.probe.resolve(strict=True)
    request = {"bdf": args.bdf, "arch": args.arch, "role": args.role,
               "profile": profile.name, "profile_digest": profile.source_digest,
               "settings": settings, "probe_digest": digest(binary), "kernel": os.uname().release,
               "dtypes": args.dtype, "n": args.n, "seconds": args.seconds,
               "pause_ms": args.pause_ms, "max_hotspot_c": args.max_hotspot_c,
               "stock_confirmed": args.stock_confirmed}
    before, hw = snapshot(args.bdf)
    reference_bytes = args.reference.read_bytes() if args.reference else None
    reference = json.loads(reference_bytes) if reference_bytes is not None else None
    if reference_bytes is not None:
        keys = ("bdf", "arch", "role", "probe_digest", "kernel", "dtypes", "n", "seconds", "pause_ms", "max_hotspot_c")
        if (not isinstance(reference, dict) or reference.get("kind") != "ai-power-screen" or reference.get("status") != "pass" or
                reference["request"]["profile"] != "stock" or not reference["request"]["stock_confirmed"] or
                any(reference["request"][k] != request[k] for k in keys)):
            raise ScreenError("incompatible or unpassed stock screen")
        if (reference["before"] != reference["after"] or
                len(reference["cases"]) != len(args.dtype) * len(args.pause_ms) or
                any(c.get("status") != "pass" or c["native"].get("status") != "pass"
                    or c["native"].get("iterations", 0) < 2 for c in reference["cases"])):
            raise ScreenError("stock reference lacks complete passing coverage")
        verify_settings(before, settings, reference["before"])
    else:
        verify_settings(before, settings)
    sensor, limit = thermal_sensor(hw, args.max_hotspot_c)
    if reference is not None and reference["thermal_ceiling_mc"] != limit:
        raise ScreenError("thermal limit differs from stock reference")
    result = {"kind": "ai-power-screen", "schema_version": 1, "request": request,
              "before": before, "reference_digest": ("sha256:" + hashlib.sha256(reference_bytes).hexdigest()
                                   if reference_bytes is not None else None),
              "cases": [], "status": "inconclusive", "thermal_ceiling_mc": limit,
              "production_qualified": False}
    try:
        for dtype in args.dtype:
            for pause in args.pause_ms:
                if snapshot(args.bdf)[0] != before:
                    raise ScreenError("device controls changed between cases")
                case = run_case(binary, args.bdf, args.arch, dtype, args.n, args.seconds,
                                pause, args.setup_timeout, sensor, limit, args.output.parent)
                result["cases"].append(case)
                if case["status"] != "pass":
                    raise ScreenError(f"{dtype}/{pause} ms: {case['status']}", case["status"])
        if reference is not None:
            if [case_identity(c) for c in result["cases"]] != [case_identity(c) for c in reference["cases"]]:
                raise ScreenError("stock and candidate algorithms/coverage differ")
        if digest(binary) != request["probe_digest"]:
            raise ScreenError("native probe changed during screen")
        result["status"] = "pass"
    except ScreenError as exc:
        result.update(status=exc.status, reason=str(exc))
    finally:
        try:
            after, _ = snapshot(args.bdf)
            result["after"] = after
            if before != after:
                result["readback_error"] = "device controls/identity changed during screen"
                if result["status"] == "pass":
                    result.update(status="inconclusive", reason=result["readback_error"])
        except (ScreenError, OSError) as exc:
            result["readback_error"] = f"final readback failed: {exc}"
            if result["status"] == "pass":
                result.update(status="inconclusive", reason=result["readback_error"])
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", required=True, type=Path, help="native probe built with the installed ROCm SDK")
    p.add_argument("--profiles", required=True, type=Path, help="canonical power-profiles.yaml")
    p.add_argument("--profile", required=True)
    p.add_argument("--bdf", required=True)
    p.add_argument("--arch", required=True, help="expected architecture; verified by HIP probe")
    p.add_argument("--role", required=True, help="engine or concrete helperN")
    p.add_argument("--stock-confirmed", action="store_true", help="operator has verified factory settings; never writes")
    p.add_argument("--reference", type=Path, help="passing stock screen for candidate")
    p.add_argument("--dtype", nargs="+", choices=DTYPES, default=["fp16"])
    p.add_argument("--n", type=int, default=768)
    p.add_argument("--seconds", type=int, required=True, help="per dtype and load pattern")
    p.add_argument("--pause-ms", nargs="+", type=int, default=[0, 2], help="0 sustained, positive burst gap")
    p.add_argument("--setup-timeout", type=int, default=60, help="extra seconds allowed for SDK setup/oracle")
    p.add_argument("--max-hotspot-c", type=int, required=True, help="explicit safe upper ceiling, bounded by available reported limits")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if (not BDF.fullmatch(args.bdf) or not ARCH.fullmatch(args.arch) or
            not 16 <= args.n <= 2048 or not 1 <= args.seconds <= 3600 or
            not 1 <= args.setup_timeout <= 300 or not 1 <= args.max_hotspot_c <= 110 or
            not all(0 <= x <= 1000 for x in args.pause_ms) or
            len(set(args.dtype)) != len(args.dtype) or len(set(args.pause_ms)) != len(args.pause_ms)):
        p.error("invalid device, bounds or duplicate test cases")
    if args.output.exists() or args.output.is_symlink():
        p.error("output already exists; use a new evidence path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def cancelled(signum, frame):
        raise ScreenError("screen interrupted", "cancelled")

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        result = screen(args)
    except (ScreenError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result = {"kind": "ai-power-screen", "schema_version": 1,
                  "status": getattr(exc, "status", "inconclusive"), "reason": str(exc),
                  "production_qualified": False}
    finally:
        signal.signal(signal.SIGTERM, previous)
    write_json_atomic(args.output, result)
    print(f"AI power screen: {result['status']} ({args.output})")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
