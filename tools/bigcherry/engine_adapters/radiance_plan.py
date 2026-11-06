"""A measured serving manifest for one campaign arm (RSC11 B): the harness side of the launcher's plan check.

An arm names a configuration (a layout and its settings) under ``serving``; this module makes the manifest the engine and the
helpers start from. It asks the build's bc-serve for the measured manifest of that exact configuration (plan cache hit) or, on a
miss (bc-serve exit 10), measures the plan first: the helpers come up on their cards, the engine runs to the end of its memory
plan (``--plan-memory``) against the unmeasured manifest and exits, and bc-serve writes the measured manifest from that plan
and stores it. The measuring start uses the GPUs like a cell does, so a campaign that needs one runs under the gpu lease.

This is the same procedure as tools/lab/radiance/lib/plan-check.sh, which retires with the launcher (RSC02).
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from ..core.engines import ServeSpec
from ..tuning.server_runner import _free_port
from .base import SidecarError, SidecarSet, SidecarSpec

CACHE_MISS = 10
_ENGINE_MIB = re.compile(r"--plan-memory: wrote .*; engine start-up took (\d+) MiB of MemAvailable;")
_KEYS = {"layout", "set", "heat", "histogram", "calibration", "live_policy", "experiment", "calibration_run"}


class PlanError(RuntimeError):
    pass


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise PlanError(f"serving.set values are strings, numbers or booleans, got {value!r}")


def _site(path: Path, model: Path, radiance_home: Path, serving: dict[str, Any]) -> None:
    lines = ["[engine]", f"model = {_toml(str(model))}", f"radiance_home = {_toml(str(radiance_home))}"]
    handoff = [(k, serving[k]) for k in ("heat", "histogram") if k in serving]
    if handoff:
        lines += ["[handoff]", *(f"{k} = {_toml(str(v))}" for k, v in handoff)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def helpers_resident_mib(pids: list[int]) -> int:
    """Anonymous memory of every helper plus the shared segment once, MiB rounded up (file pages are reclaimable)."""
    anon = shm = 0
    for pid in pids:
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError as exc:
            raise PlanError(f"helpers resident memory: cannot read /proc/{pid}/status: {exc}") from exc
        a = re.search(r"^RssAnon:\s+(\d+)", status, re.MULTILINE)
        s = re.search(r"^RssShmem:\s+(\d+)", status, re.MULTILINE)
        if not a or not s:
            raise PlanError(f"helpers resident memory: no RssAnon/RssShmem in /proc/{pid}/status")
        anon += int(a.group(1))
        shm = max(shm, int(s.group(1)))
    return (anon + shm + 1023) // 1024


def write_manifest(
    *, name: str, serving: dict[str, Any], binary: Path, model: Path, env: dict[str, str], serve: ServeSpec,
    run_dir: Path, plan_cache: Path, host: str,
    helper_specs: Callable[[Path], list[SidecarSpec]],
) -> Path:
    """The measured manifest for this arm, written under ``run_dir``. ``helper_specs(manifest)`` is the adapter's helper
    processes for a manifest (the measuring start brings them up)."""
    unknown = sorted(set(serving) - _KEYS)
    if unknown or "layout" not in serving:
        raise PlanError(f"{name}: serving takes layout (required), set, heat and histogram; got {sorted(serving)}")
    sets = dict(serving.get("set", {}))
    sets["layout.name"] = serving["layout"]
    sets.setdefault("engine.port", _free_port(host))
    radiance_home = binary.parent.parent / "radiance_home"
    bc_serve = radiance_home / "bin" / "bc-serve"
    if not bc_serve.is_file():
        raise PlanError(f"{name}: no bc-serve in the build ({bc_serve})")
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_cache.mkdir(parents=True, exist_ok=True)
    site = run_dir / "site.toml"
    _site(site, model, radiance_home, serving)
    set_args = [a for key in sorted(sets) for a in ("--set", f"{key}={_toml(sets[key])}")]
    # What else decides the plan but is not a bc-serve setting: the route-heat file's contents and the lab experiment.
    extra = "; ".join(f"{k}-sha256: {_file_sha256(Path(serving[k]))}" for k in ("heat", "experiment") if k in serving) or "none"
    # What the helpers read besides the layout and the settings: the calibration store the live policy is derived from, an explicit
    # lab policy, the lab experiment, and the calibration-run profile.
    objects = []
    for key, flag in (("calibration", "--calibration"), ("live_policy", "--live-policy"), ("experiment", "--experiment")):
        if key in serving:
            objects += [flag, str(serving[key])]
    if serving.get("calibration_run"):
        objects.append("--calibration-run")
    common = ["--site", str(site), *set_args, *objects]

    def run(out: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(bc_serve), "--write-manifest", str(out), *args], capture_output=True, text=True)

    measured = run_dir / "manifest.json"
    cached = ["--plan-cache", str(plan_cache), "--engine-binary", str(binary), "--plan-extra", extra]
    result = run(measured, *cached, *common)
    if result.returncode == 0:
        return measured
    if result.returncode != CACHE_MISS:
        raise PlanError(f"{name}: bc-serve --write-manifest failed ({result.returncode}): {result.stderr.strip()}")

    unmeasured = run_dir / "manifest-measuring.json"
    result = run(unmeasured, "--unmeasured", *common)
    if result.returncode:
        raise PlanError(f"{name}: the unmeasured manifest was not written ({result.returncode}): {result.stderr.strip()}")
    plan = run_dir / "plan-measured.json"
    log = run_dir / "engine-measure.log"
    plan.unlink(missing_ok=True)
    specs = [dataclasses.replace(s, argv=(*s.argv, "--measuring")) for s in helper_specs(unmeasured)]
    helpers = SidecarSet(specs)
    try:
        helpers.start()
        resident = helpers_resident_mib([r.process.pid for r in helpers._running]) if specs else 0
        engine_env = dict(serve.env)
        for var, relative in serve.env_from_binary:
            engine_env[var] = str((binary.parent / relative).resolve())
        engine_env.update(env)
        argv = [str(binary), *(a.replace("{manifest}", str(unmeasured)) for a in serve.manifest_args), "--plan-memory", str(plan)]
        with log.open("w", encoding="utf-8") as sink:
            engine = subprocess.run(argv, env=engine_env, stdout=sink, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except SidecarError as exc:
        raise PlanError(f"{name}: the measuring start's helpers did not come up: {exc}") from exc
    finally:
        records = helpers.stop()
    if engine.returncode or not plan.is_file() or not plan.stat().st_size:
        raise PlanError(f"{name}: the measuring start failed (engine exit {engine.returncode}); see {log}")
    if any(r["killed"] for r in records):
        raise PlanError(f"{name}: the measuring start's helpers did not exit cleanly")
    found = _ENGINE_MIB.findall(log.read_text(encoding="utf-8", errors="replace"))
    if not found:
        raise PlanError(f"{name}: the measuring start did not report the engine's host memory; see {log}")
    result = run(measured, *cached, *common, "--measured-plan", str(plan),
                 "--helpers-resident-mib", str(resident), "--engine-resident-mib", found[-1])
    if result.returncode:
        raise PlanError(f"{name}: the measured manifest was not written ({result.returncode}): {result.stderr.strip()}")
    return measured
