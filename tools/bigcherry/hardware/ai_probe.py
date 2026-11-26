"""PWR12: independent AI power-profile screen.

Observe by default. An explicitly authorized candidate can use the existing
platform PowerTransaction; the probe itself never writes hardware controls.
This is not production qualification or automatic profile promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from . import gpu_power, lease, power_probe, power_profiles
from .power_transaction import PowerFailed, PowerRefused, PowerTransaction

DTYPES = ("fp16", "bf16", "fp32", "fp8", "int8", "memory")
PROVIDERS = ("hipblaslt", "rocblas")
POLL_SECONDS = 0.25
NATIVE_PROTOCOL = 2
SCREEN_SCHEMA = 3
CORPUS_COUNT = 3
BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
ARCH = re.compile(r"^gfx[0-9a-f]+$")


class ScreenError(RuntimeError):
    def __init__(self, reason: str, status: str = "inconclusive"):
        super().__init__(reason)
        self.status = status


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return "sha256:" + hashlib.file_digest(f, "sha256").hexdigest()


def snapshot(bdf: str, *, sysfs_root: Path = Path("/")) -> tuple[dict, Path]:
    card, = power_probe.probe_gpus([bdf], sysfs_root=sysfs_root)
    if card.driver.value != "amdgpu" or card.hwmon.status != "supported":
        raise ScreenError("selected AMD device or unique hwmon unavailable")
    hw = Path(card.hwmon.source)
    controls = {k: v.current.value if v.current.status == "supported" else None
                for k, v in card.controls.items()}
    cap = controls.get("power_cap_w")
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ScreenError("invalid power-cap readback")
    pci = sysfs_root / "sys/bus/pci/devices" / bdf
    identity = {k: power_probe.read_field(pci / k).value
                for k in ("vendor", "device", "subsystem_vendor", "subsystem_device", "unique_id")}
    if identity["vendor"] != "0x1002" or identity["device"] is None:
        raise ScreenError("physical PCI identity unavailable")
    identity["vbios_version"] = power_probe.read_field(pci / "vbios_version").value
    identity["performance_level"] = power_probe.read_field(pci / "power_dpm_force_performance_level").value
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


def thermal_sensor(hw: Path, max_c: int | None, *, labels=("junction", "hotspot"),
                   required=True) -> tuple[Path | None, int | None]:
    matches = [p for p in hw.glob("temp*_label")
               if str(power_probe.read_field(p).value).lower() in labels]
    if len(matches) != 1:
        if required or matches:
            raise ScreenError(f"unique {labels[0]} sensor unavailable")
        return None, None
    stem = matches[0].name.removesuffix("_label")
    limit = max_c * 1000 if max_c is not None else None
    for suffix in ("_crit", "_emergency"):
        observed = power_probe.read_int(hw / (stem + suffix)).value
        if type(observed) is int and observed > 0:
            limit = min(limit, observed) if limit is not None else observed
    return hw / (stem + "_input"), limit


def sample(sensor: Path, limit: int | None, samples: list[int]) -> None:
    value = power_probe.read_int(sensor).value
    if type(value) is not int or value <= 0:
        raise ScreenError("temperature read unavailable")
    samples.append(value)
    if limit is not None and value >= limit:
        raise ScreenError("thermal ceiling reached", "thermal")


def hot_window(samples: list[tuple[int, int]], start_ns: int, end_ns: int,
               seconds: int, minimum_mc: int) -> dict | None:
    """Prove a full sampled interval inside load, near its end, not cooldown."""
    max_gap = int(4 * POLL_SECONDS * 1e9)
    in_load = [(t, v) for t, v in samples if start_ns <= t <= end_ns]
    if not in_load or end_ns - in_load[-1][0] > max_gap:
        return None
    begin = in_load[-1][0] - seconds * 1_000_000_000
    left = [i for i, (t, _) in enumerate(in_load) if t <= begin]
    if not left:
        return None
    window = in_load[left[-1]:]
    gaps = [b[0]-a[0] for a, b in zip(window, window[1:])]
    if (not gaps or any(v < minimum_mc for _, v in window) or
            any(g <= 0 or g > max_gap for g in gaps)):
        return None
    return {"start_ns": window[0][0], "end_ns": window[-1][0],
            "minimum_mc": min(v for _, v in window), "maximum_mc": max(v for _, v in window),
            "max_gap_ns": max(gaps), "samples": len(window)}


def run_case(binary: Path, bdf: str, arch: str, dtype: str, n: int,
             seconds: int, pause: int, setup_timeout: int,
             sensor: Path, limit: int, output_dir: Path, *,
             provider: str = "hipblaslt", warmup_seconds: int = 0,
             min_hotspot_c: int | None = None, headroom_mib: int = 512,
             memory_sensor: Path | None = None, memory_limit: int | None = None,
             batch: int = 1,
             cancelled=lambda: False) -> dict:
    """Always reap the probe before returning; requested coverage is never a skip."""
    samples: list[int] = []
    timestamps: list[tuple[int, int]] = []
    memory_samples: list[int] = []
    power_samples: list[int] = []
    def observe():
        sample(sensor, limit, samples)
        timestamps.append((time.monotonic_ns(), samples[-1]))
        if memory_sensor is not None:
            sample(memory_sensor, memory_limit, memory_samples)
        watts = power_probe.read_int(sensor.parent / "power1_average").value
        if type(watts) is int and watts >= 0:
            power_samples.append(watts)
    observe()
    if cancelled():
        raise ScreenError("screen interrupted", "cancelled")
    if not lease.held_by_this_process():
        raise ScreenError("GPU lease not verified")
    duration = seconds + warmup_seconds
    launch_ns = time.monotonic_ns()
    with tempfile.TemporaryFile(dir=output_dir) as stdout, tempfile.TemporaryFile(dir=output_dir) as stderr:
        proc = subprocess.Popen([str(binary), bdf, arch, dtype, str(n), str(duration), str(pause),
                                 provider, str(headroom_mib), str(batch)], stdout=stdout, stderr=stderr)
        failure = None
        try:
            while proc.poll() is None:
                if cancelled():
                    raise ScreenError("screen interrupted", "cancelled")
                if not lease.held_by_this_process():
                    raise ScreenError("GPU lease lost", "lease_lost")
                observe()
                if time.monotonic_ns() - launch_ns > (duration + setup_timeout) * 1e9:
                    raise ScreenError("probe deadline exceeded", "timeout")
                time.sleep(POLL_SECONDS)
            observe()
            if cancelled():
                raise ScreenError("screen interrupted", "cancelled")
            if not lease.held_by_this_process():
                raise ScreenError("GPU lease lost", "lease_lost")
        except ScreenError as exc:
            failure = exc
        finally:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            except BaseException as exc:
                raise ScreenError(f"probe exit not verified: {exc}", "unreaped") from exc
        stdout.seek(0)
        stderr.seek(0)
        raw = stdout.read().decode("utf-8", errors="replace")
        diagnostic = stderr.read().decode("utf-8", errors="replace")
    result = {"dtype": dtype, "n": n, "pause_ms": pause, "requested_seconds": seconds,
              "warmup_seconds": warmup_seconds, "batch": batch, "hotspot_min_mc": min(samples),
              "hotspot_max_mc": max(samples), "samples": len(samples),
              "memory_min_mc": min(memory_samples) if memory_samples else None,
              "memory_max_mc": max(memory_samples) if memory_samples else None,
              "mean_sampled_power_uw": sum(power_samples)//len(power_samples) if power_samples else None,
              "power_samples": len(power_samples), "exit": proc.returncode,
              "junction_samples": timestamps}
    if failure is not None:
        return result | {"status": failure.status, "diagnostic": str(failure)}
    if proc.returncode != 0:
        return result | {"status": {2: "mismatch", 3: "unavailable"}.get(proc.returncode, "device_error"),
                         "diagnostic": diagnostic[-4000:]}
    try:
        payload = json.loads(raw.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise ScreenError("probe did not emit a valid result") from None
    validate_native(payload, bdf, arch, dtype, n, provider, duration, setup_timeout,
                    headroom_mib, launch_ns=launch_ns, batch=batch)
    result["native"] = payload
    if min_hotspot_c is not None:
        result["hot_window"] = hot_window(timestamps, payload["start_ns"], payload["end_ns"], seconds, min_hotspot_c*1000)
        if result["hot_window"] is None:
            return result | {"status": "inconclusive", "diagnostic": "hot window not reached/observed"}
    result["hot_window_verified"] = min_hotspot_c is not None
    return result | {"status": "pass"}



def verify_probe_protocol(binary: Path) -> None:
    """Reject a stale native executable before any candidate mutation."""
    try:
        result = subprocess.run([str(binary), "--protocol-version"], capture_output=True,
                                text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScreenError("native probe protocol unavailable") from exc
    if result.returncode or result.stdout.strip() != str(NATIVE_PROTOCOL):
        raise ScreenError("incompatible native probe protocol; rebuild probe")


def validate_native(payload: dict, bdf: str, arch: str, dtype: str, n: int,
                    provider: str, duration: int, setup_timeout: int,
                    headroom_mib: int, *, launch_ns: int | None = None, batch: int = 1) -> None:
    """Reject incomplete or mismatched native coverage in live and reference receipts."""
    valid_algorithm = (isinstance(payload, dict) and
                       isinstance(payload.get("algorithm"), str) and
                       (bool(re.fullmatch("[0-9a-f]{32}", payload["algorithm"])) if provider == "hipblaslt"
                        else payload["algorithm"] == {"rocblas": "rocblas-default-no-atomics",
                                                      "hip-memory": "address-pattern-v1"}.get(provider)))
    if (not valid_algorithm or type(payload.get("protocol_version")) is not int or
            payload["protocol_version"] != NATIVE_PROTOCOL or payload.get("status") != "pass" or
            payload.get("bdf") != bdf or payload.get("arch") != arch or
            payload.get("dtype") != dtype or payload.get("n") != n or
            payload.get("provider") != provider or type(payload.get("batch")) is not int or payload.get("batch") != batch or
            not isinstance(payload.get("library_version"), str) or not payload["library_version"] or
            type(payload.get("iterations")) is not int or payload["iterations"] < 2 or
            type(payload.get("elapsed_s")) not in (int, float) or
            not duration <= payload["elapsed_s"] < duration + setup_timeout or
            any(type(payload.get(k)) is not int or payload[k] <= 0 for k in ("hip", "driver")) or
            type(payload.get("workspace")) is not int or payload["workspace"] < 0 or
            any(type(payload.get(k)) is not int for k in ("start_ns", "end_ns")) or
            not 0 < payload["start_ns"] < payload["end_ns"] or
            (launch_ns is not None and not launch_ns <= payload["start_ns"] < payload["end_ns"] <= time.monotonic_ns()) or
            payload["end_ns"] - payload["start_ns"] < duration * 1e9):
        raise ScreenError("probe result identity or coverage mismatch")
    if dtype != "memory":
        launches = payload["iterations"]
        rounds, remainder = divmod(launches, batch)
        q, r = divmod(launches, CORPUS_COUNT)
        want_launches = [q + int(i < r) for i in range(CORPUS_COUNT)]
        # Rotation touches each corpus once per launch for small batches,
        # or all corpora once per batch for larger batches.
        want_checks = want_launches if batch < CORPUS_COUNT else [rounds] * CORPUS_COUNT
        def exact_counts(key, expected):
            value = payload.get(key)
            return (isinstance(value, list) and all(type(v) is int for v in value)
                    and value == expected)
        def exact_count(key, expected):
            return type(payload.get(key)) is int and payload[key] == expected
        host_checks = sum(want_checks)
        if (remainder or min(want_launches) < 1 or
                not exact_counts("corpus_iterations", want_launches) or
                not exact_counts("corpus_host_checks", want_checks) or
                not exact_count("host_checks", host_checks) or
                not exact_count("host_elements_checked", host_checks * n * n) or
                not exact_count("device_elements_checked", launches * n * n if batch > 1 else 0)):
            raise ScreenError("matrix verification coverage mismatch")
    if dtype == "memory":
        if (payload.get("allocated_bytes") != n * 2**20 or
                type(payload.get("free_before_bytes")) is not int or
                payload["free_before_bytes"] - payload["allocated_bytes"] < headroom_mib * 2**20 or
                type(payload.get("free_after_bytes")) is not int or
                payload["free_after_bytes"] < headroom_mib * 2**20):
            raise ScreenError("memory allocation/headroom coverage mismatch")

def case_identity(case: dict) -> tuple:
    native = case["native"]
    return tuple(case[k] for k in ("dtype", "n", "pause_ms", "requested_seconds", "warmup_seconds", "batch")) + tuple(
        native[k] for k in ("bdf", "arch", "hip", "driver", "provider", "library_version", "workspace", "algorithm", "protocol_version"))


def screen_receipt(run_id: str, result: dict):
    """Use the platform power receipt envelope; screen details are observations."""
    from .power_receipt import Receipt
    receipt = Receipt(run_id, request=result.get("request", {}),
                      transaction=result.get("transaction", {"state": "not_requested", "events": []}))
    for key, phase in (("before", "start"), ("after", "stop")):
        if key in result:
            receipt.observe(phase, result[key])
    receipt.observe("ai_screen", {k: v for k, v in result.items()
                                  if k not in ("request", "transaction", "before", "after")})
    return receipt


def read_screen_receipt(data: bytes, run_id: str) -> dict:
    doc = json.loads(data)
    if (not isinstance(doc, dict) or doc.get("schema_version") != 1 or
            doc.get("run_id") != run_id or doc.get("manifest", {}).get("status") != "unavailable"):
        raise ScreenError("incompatible stock power receipt")
    observations = doc["observations"]
    result = dict(observations["ai_screen"], request=doc["request"], transaction=doc["transaction"],
                  before=observations["start"], after=observations["stop"])
    return result


def screen(args: argparse.Namespace, *, cancelled=lambda: False,
           journal_dir: Path | None = None) -> dict:
    if not BDF.fullmatch(args.bdf) or not ARCH.fullmatch(args.arch):
        raise ScreenError("invalid physical device identity")
    input_digests = getattr(args, "input_digests", {})
    if "profiles" in input_digests and digest(args.profiles) != input_digests["profiles"]:
        raise ScreenError("profile input changed since run creation")
    profile = power_profiles.get_profile(power_profiles.load_profiles(args.profiles), args.profile)
    if "profiles" in input_digests and digest(args.profiles) != input_digests["profiles"]:
        raise ScreenError("profile input changed during read")
    settings = profile.settings_for(args.role, args.arch)
    if args.profile == "stock":
        if args.reference or args.stock_confirmed == args.observe_current:
            raise ScreenError("stock arm requires exactly one of --stock-confirmed or --observe-current and no reference")
    elif not settings or not args.reference or args.stock_confirmed or (args.observe_current and not args.apply_profile):
        raise ScreenError("candidate requires matching nonempty profile rules and --reference")
    if args.apply_profile:
        if args.profile == "stock" or not args.confirm_idle or journal_dir is None:
            raise ScreenError("apply requires candidate, existing transaction journal and --confirm-idle")
        if "sclk_offset_mhz" in settings:
            raise ScreenError("platform transaction does not support sclk_offset_mhz; use observe-only owner path")
    elif args.confirm_idle:
        raise ScreenError("--confirm-idle is only for explicit apply")
    if not lease.held_by_this_process():
        raise ScreenError("GPU lease not verified")
    if cancelled():
        raise ScreenError("screen interrupted", "cancelled")
    binary = args.probe.resolve(strict=True)
    probe_digest = digest(binary)
    if "probe" in input_digests and probe_digest != input_digests["probe"]:
        raise ScreenError("probe changed since run creation")
    verify_probe_protocol(binary)
    if digest(binary) != probe_digest:
        raise ScreenError("probe changed during protocol verification")
    request = {"bdf": args.bdf, "arch": args.arch, "role": args.role,
               "profile": profile.name, "profile_digest": profile.source_digest,
               "settings": settings, "probe_digest": probe_digest, "kernel": os.uname().release,
               "dtypes": args.dtype, "n": args.n, "seconds": args.seconds,
               "pause_ms": args.pause_ms, "max_hotspot_c": args.max_hotspot_c,
               "provider": args.provider, "batch": args.batch, "memory_mib": args.memory_mib, "headroom_mib": args.headroom_mib,
               "setup_timeout": args.setup_timeout,
               "warmup_seconds": args.warmup_seconds, "min_hotspot_c": args.min_hotspot_c,
               "max_memory_c": args.max_memory_c, "observe_current": args.observe_current,
               "stock_confirmed": args.stock_confirmed, "apply_profile": args.apply_profile}
    before, hw = snapshot(args.bdf)
    reference_bytes = args.reference.read_bytes() if args.reference else None
    if reference_bytes is not None:
        if "sha256:" + hashlib.sha256(reference_bytes).hexdigest() != args.reference_digest:
            raise ScreenError("stock power receipt changed since run creation")
        reference = read_screen_receipt(reference_bytes, args.reference_run_id)
    else:
        reference = None
    if reference_bytes is not None:
        keys = ("bdf", "arch", "role", "probe_digest", "kernel", "dtypes", "n", "seconds", "pause_ms", "max_hotspot_c",
                "provider", "batch", "memory_mib", "headroom_mib", "warmup_seconds", "min_hotspot_c", "max_memory_c",
                "setup_timeout")
        if (not isinstance(reference, dict) or reference.get("kind") != "ai-power-screen" or reference.get("schema_version") != SCREEN_SCHEMA or reference.get("status") != "pass" or
                reference["request"]["profile"] != "stock" or
                reference["request"]["stock_confirmed"] is not (not args.observe_current) or
                reference["request"]["observe_current"] is not args.observe_current or
                reference["request"]["apply_profile"] is not False or
                any(reference["request"][k] != request[k] for k in keys)):
            raise ScreenError("incompatible or unpassed stock screen (observed references require explicit research mode)")
        recipe = [(dtype, pause) for dtype in args.dtype for pause in args.pause_ms]
        if (reference["before"] != reference["after"] or len(reference["cases"]) != len(recipe)):
            raise ScreenError("stock reference lacks complete passing coverage")
        for c, (dtype, pause) in zip(reference["cases"], recipe):
            n = args.memory_mib if dtype == "memory" else args.n
            provider = "hip-memory" if dtype == "memory" else args.provider
            batch = 1 if dtype == "memory" else args.batch
            if (c.get("status") != "pass" or
                    (c["dtype"], c["n"], c["pause_ms"], c["requested_seconds"], c["warmup_seconds"], c["batch"]) !=
                    (dtype, n, pause, args.seconds, args.warmup_seconds, batch) or
                    (args.min_hotspot_c is not None and c.get("hot_window_verified") is not True)):
                raise ScreenError("stock reference lacks complete passing coverage")
            validate_native(c["native"], args.bdf, args.arch, dtype, n, provider,
                            args.seconds + args.warmup_seconds, args.setup_timeout,
                            args.headroom_mib or 512, batch=batch)
            if args.min_hotspot_c is not None:
                window = hot_window(c["junction_samples"], c["native"]["start_ns"], c["native"]["end_ns"],
                                    args.seconds, args.min_hotspot_c*1000)
                if window is None or window != c.get("hot_window"):
                    raise ScreenError("reference hot window lacks sampled coverage")
        if args.apply_profile:
            if before != reference["before"]:
                raise ScreenError("current controls/identity differ from " +
                                  ("observed" if args.observe_current else "confirmed stock") + " reference; no writes")
            if not args.observe_current:
                verify_settings(before, {})
        else:
            verify_settings(before, settings, reference["before"])
    elif not args.observe_current:
        verify_settings(before, settings)
    sensor, limit = thermal_sensor(hw, args.max_hotspot_c)
    mem_sensor, mem_limit = thermal_sensor(hw, args.max_memory_c, labels=("mem", "memory"),
                                          required=args.max_memory_c is not None)
    if args.min_hotspot_c is not None and args.min_hotspot_c * 1000 >= limit:
        raise ScreenError("hot minimum is at or above actual ceiling")
    if reference is not None and (reference["thermal_ceiling_mc"] != limit or
                                  reference["memory_ceiling_mc"] != mem_limit):
        raise ScreenError("thermal limit differs from stock reference")
    result = {"kind": "ai-power-screen", "schema_version": SCREEN_SCHEMA, "request": request,
              "before": before, "reference_digest": ("sha256:" + hashlib.sha256(reference_bytes).hexdigest()
                                   if reference_bytes is not None else None),
              "reference_basis": ("observed" if args.observe_current else "factory_confirmed") if reference else None,
              "cases": [], "status": "inconclusive", "thermal_ceiling_mc": limit,
              "memory_ceiling_mc": mem_limit, "production_qualified": False,
              "transaction": {"state": "not_requested"}}
    original = before
    tx = None
    try:
        if args.apply_profile:
            # Reuse the existing platform GPU power transaction, recovery
            # journal and sysfs writer. No second controller or reset command.
            tx = PowerTransaction(args.run_id, journal_dir,
                                  lease_ok=lease.held_by_this_process)
            result["transaction"] = {"state": "new", "run_id": tx.run_id, "journal": str(tx.journal_path),
                                     "original": original}
            # No mutation if current thermal telemetry already refuses.
            sample(sensor, limit, [])
            if mem_sensor is not None:
                sample(mem_sensor, mem_limit, [])
            tx.prepare({args.bdf: gpu_power.Setting(**settings)}, [args.bdf])
            if cancelled() or not lease.held_by_this_process():
                raise ScreenError("interrupted or lease lost before apply")
            if snapshot(args.bdf)[0] != original:
                raise ScreenError("stock state changed during preflight")
            sample(sensor, limit, [])
            if mem_sensor is not None:
                sample(mem_sensor, mem_limit, [])
            tx.apply()
            before, after_hw = snapshot(args.bdf)
            result["before"] = before
            if after_hw != hw:
                raise ScreenError("sensor identity changed during apply")
            verify_settings(before, settings, original)
        for dtype in args.dtype:
            for pause in args.pause_ms:
                if snapshot(args.bdf)[0] != before:
                    raise ScreenError("device controls changed between cases")
                case = run_case(binary, args.bdf, args.arch, dtype,
                                args.memory_mib if dtype == "memory" else args.n, args.seconds,
                                pause, args.setup_timeout, sensor, limit, args.output.parent,
                                provider="hip-memory" if dtype == "memory" else args.provider,
                                warmup_seconds=args.warmup_seconds, min_hotspot_c=args.min_hotspot_c,
                                headroom_mib=args.headroom_mib or 512, memory_sensor=mem_sensor,
                                memory_limit=mem_limit, cancelled=cancelled,
                                batch=1 if dtype == "memory" else args.batch)
                result["cases"].append(case)
                if case["status"] != "pass":
                    raise ScreenError(f"{dtype}/{pause} ms: {case['status']}", case["status"])
        if reference is not None:
            if [case_identity(c) for c in result["cases"]] != [case_identity(c) for c in reference["cases"]]:
                raise ScreenError("stock and candidate algorithms/coverage differ")
        if digest(binary) != request["probe_digest"]:
            raise ScreenError("native probe changed during screen")
        if cancelled() or not lease.held_by_this_process():
            raise ScreenError("interrupted or lease lost after probe", "cancelled")
        result["status"] = "pass"
    except (ScreenError, PowerRefused, PowerFailed, OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status=getattr(exc, "status", "inconclusive"), reason=str(exc))
    finally:
        try:
            after, _ = snapshot(args.bdf)
            result["after"] = after
            if before != after:
                result["readback_error"] = "device controls/identity changed during screen"
                if result["status"] == "pass":
                    result.update(status="inconclusive", reason=result["readback_error"])
        except Exception as exc:
            result["readback_error"] = f"final readback failed: {exc}"
            if result["status"] == "pass":
                result.update(status="inconclusive", reason=result["readback_error"])
        if tx is not None:
            receipt = result["transaction"]
            receipt["state"] = tx.state
            receipt["events"] = list(getattr(tx, "events", []))
            # Any unverifiable child exit or lost lease preserves the journal
            # for existing recovery. Do not restore under a live child.
            if tx.touched:
                try:
                    if result["status"] == "unreaped" or not lease.held_by_this_process():
                        raise ScreenError("restore requires verified child exit and lease")
                    if tx.state not in ("rolled_back", "restored") and not tx.restore(processes_exited=True):
                        raise ScreenError("transaction restore failed")
                    restored, _ = snapshot(args.bdf)
                    receipt["restored_readback"] = restored
                    if restored != original:
                        raise ScreenError("original controls/identity not restored exactly")
                    receipt["state"] = "restored"
                except Exception as exc:
                    receipt.update(state="restore_unverified", reason=str(exc))
                    result["screen_status"] = result["status"]
                    result.update(status="restore_failed", reason=str(exc))
            receipt["events"] = list(getattr(tx, "events", []))
    return result
