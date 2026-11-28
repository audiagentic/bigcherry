#!/usr/bin/env python3
"""RSC02 S15 leased A/B owner acceptance. Lab-only driver; never a runtime dependency.

The operator explicitly chooses a completed build and inputs. This driver does
not rewrite configs, synthesize manifests, or infer a PASS from an HTTP 200.
Each arm owns a fresh lease and must fully stop before the next lease is taken.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from rsc02_s15_projection import ProjectionError, compare, dry_run, projection

REPO = Path(__file__).resolve().parents[3]
OBSERVED_ENV = (
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "RADIANCE_HOME",
    "ROCM_PATH", "HIP_PATH", "LD_LIBRARY_PATH", "HSA_OVERRIDE_GFX_VERSION",
)
COHORTS = {"dense": ("qwen27b-tp3-q8", "qwen27b-tp3-q8"),
           "helper": ("flashnext-tp2xtx-r9700", "flashnext-tp2xtx-r9700")}
ORDER = (("A", "B"), ("B", "A"), ("A", "B"), ("B", "A"))
CANARY_PROMPT = "The capital of France is"
CANARY_EXPECT = "Paris"
GREEDY_PROMPT = "The capital of Japan is"


class Refusal(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def command(argv: list[str], *, timeout: int = 120, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, **kwargs)


def require_absolute_file(path: str, description: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_file():
        raise Refusal(f"{description} must be an existing absolute regular file: {path}")
    return candidate.resolve()


def require_absolute_dir(path: str, description: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_dir():
        raise Refusal(f"{description} must be an existing absolute directory: {path}")
    return candidate.resolve()


def build_identity(cfg: dict) -> dict:
    build = require_absolute_dir(cfg["build"], "completed build")
    candidates = {
        "engine": build / "bin/radiance",
        "native_owner": Path(cfg["bc_serve"]),
        "helper": build / "radiance_home/helpers/r3_moe_worker",
        "build_metadata": build / "bigcherry-build-metadata-radiance.json",
        "model": Path(cfg["model_file"]),
        "archived_bash_control": Path(cfg["legacy_runner"]),
    }
    # Dense builds may have no helper installed; the helper cohort must have it.
    if cfg["cohort"] == "dense" and not candidates["helper"].is_file():
        candidates.pop("helper")
    return {name: {"path": str(require_absolute_file(str(p), name)), "sha256": digest(p)}
            for name, p in candidates.items()}


def read_process(pid: int, expected_role: str) -> tuple[dict, str]:
    base = Path("/proc") / str(pid)
    try:
        start_before = (base / "stat").read_text().rsplit(") ", 1)[1].split()[19]
        argv = [part.decode("utf-8") for part in (base / "cmdline").read_bytes().split(b"\0") if part]
        values = [(part.split(b"=", 1)) for part in (base / "environ").read_bytes().split(b"\0")
                  if b"=" in part]
        env = {k.decode("utf-8"): v.decode("utf-8") for k, v in values
               if k.decode("utf-8") in OBSERVED_ENV or k.startswith(b"R3_")}
        start_after = (base / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (OSError, UnicodeError, IndexError) as error:
        raise Refusal(f"{expected_role}: cannot inspect owned process {pid}: {error}") from error
    if (start_before != start_after or argv.count("--role") != 1 or
            argv.index("--role") + 1 >= len(argv) or
            argv[argv.index("--role") + 1] != expected_role):
        raise Refusal(f"{expected_role}: PID identity or observed role changed")
    if argv.count("--manifest") != 1 or argv.index("--manifest") + 1 >= len(argv):
        raise Refusal(f"{expected_role}: no unique manifest argument")
    return {"role": expected_role, "argv": argv, "env": env, "pid": pid}, start_after


def capture(processes: dict[str, int]) -> tuple[dict, dict[str, tuple[int, str]]]:
    if not processes or "engine" not in processes or len(processes) != len(set(processes)):
        raise Refusal("missing/duplicate engine or helper role")
    observed, identities = [], {}
    for role, pid in sorted(processes.items()):
        row, stamp = read_process(pid, role)
        observed.append({key: row[key] for key in ("role", "argv", "env")})
        identities[role] = (pid, stamp)
    paths = []
    for item in observed:
        idx = item["argv"].index("--manifest")
        paths.append(item["argv"][idx + 1])
    if len(set(paths)) != 1:
        raise Refusal("roles do not share one manifest")
    path = Path(paths[0])
    # A stored-manifest replay hands every child a pinned descriptor ("/proc/self/fd/N"): it
    # means the child's own open file, so read it through a child's /proc entry. The bytes
    # are verified live here; the recorded hash keeps them bound afterwards.
    pinned = re.fullmatch(r"/proc/self/fd/(\d+)", str(path))
    readable = Path(f"/proc/{processes['engine']}/fd/{pinned.group(1)}") if pinned else path
    before = readable.read_bytes()
    manifest = json.loads(before)
    snapshot = {
        "schema_version": 1,
        "manifest_path": str(path),
        "manifest_file_sha256": hashlib.sha256(before).hexdigest(),
        # Original bytes, not a JSON reserialization. The platform publisher
        # refuses legacy snapshots that cannot prove this exact identity.
        "manifest_raw_base64": base64.b64encode(before).decode("ascii"),
        "manifest": manifest,
        "processes": {"roles": observed},
    }
    if pinned:
        snapshot["manifest_read_path"] = str(readable)
        snapshot["manifest_bytes_verified_live"] = True
    projection(snapshot)  # independent role completeness and byte binding
    if readable.read_bytes() != before:
        raise Refusal("manifest bytes mutated during capture")
    return snapshot, identities


def alive(identity: tuple[int, str]) -> bool:
    pid, stamp = identity
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[19] == stamp
    except (OSError, IndexError):
        return False


def observed_legacy(work_root: Path) -> dict[str, int]:
    pids = work_root / "flashnext/pids"
    rows: dict[str, int] = {}
    for path in pids.glob("*"):
        if path.name == "engine" or (path.name.startswith("helper") and path.name[6:].isdigit()):
            rows[path.name] = int(path.read_text().strip())
    return rows


def observed_native(bc_serve: Path, run_dir: Path) -> tuple[dict[str, int], dict]:
    result = command([str(bc_serve), "status", "--run-dir", str(run_dir), "--json"], timeout=8)
    if result.returncode:
        raise Refusal(f"native status failed: {result.stderr.strip()}")
    status = json.loads(result.stdout)
    if status.get("state") != "running" or not status.get("ok"):
        raise Refusal(f"native owner not healthy: {status.get('state')}")
    rows = {p["role"]: int(p["pid"]) for p in status["processes"] if not p.get("exited", False)}
    return rows, status


def http_request(port: int, endpoint: str, payload: dict | None, timeout: int) -> object:
    url = f"http://127.0.0.1:{port}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"content-type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as result:
        body = result.read(1024 * 1024)
        if result.status != 200:
            raise Refusal(f"HTTP {result.status} at {endpoint}")
        if payload is None:
            return body.decode("utf-8")
        return json.loads(body)


def complete(port: int, prompt: str) -> str:
    response = http_request(port, "/v1/completions", {
        "model": "default", "prompt": prompt, "temperature": 0, "max_tokens": 24,
    }, timeout=120)
    try:
        answer = response["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as error:
        raise Refusal(f"completion JSON lacks choices[0].text: {error}") from error
    if not isinstance(answer, str) or not answer.strip():
        raise Refusal("greedy completion is empty")
    return answer


def power_evidence(run_dir: Path, work_root: Path) -> dict:
    found = {}
    for directory in (run_dir / "power-receipts", work_root / "power-receipts"):
        if directory.is_dir():
            for p in sorted(directory.glob("*.power.json")):
                found[str(p)] = {"sha256": digest(p), "content": json.loads(p.read_text())}
    marker = work_root / "flashnext/gpu-power"
    if marker.is_file():
        found[str(marker)] = {"sha256": digest(marker), "journal_run_id": marker.read_text().strip()}
    return found


def arm(cfg: dict) -> int:
    # Called only inside the agent-lease wrapper; never an independent GPU launcher.
    sys.path.insert(0, str(REPO / "tools"))
    from bigcherry.hardware.lease import held_by_this_process
    if not held_by_this_process():
        raise Refusal("S15 arm refuses to run without an ancestor-held GPU lease")
    cohort, which = cfg["cohort"], cfg["arm"]
    config, layout = COHORTS[cohort]
    work_root, build, bc_serve = Path(cfg["work_root"]), Path(cfg["build"]), Path(cfg["bc_serve"])
    run_dir = work_root / "flashnext"
    out = Path(cfg["out"])
    if any(run_dir.joinpath("pids").glob("*")):
        raise Refusal("legacy PID record remains; refuse reuse until operator checks stop")
    identity_before = build_identity(cfg)
    if cfg["expected_build"] != identity_before:
        raise Refusal("immutable build/model identity changed since preflight")
    record = {"cohort": cohort, "arm": which, "build_identity": identity_before,
              "run_dir": str(run_dir), "state": "admitting"}
    save(out, record)
    owner = None
    status = None
    owned: dict[str, tuple[int, str]] = {}
    stopped = False
    launch_log = out.with_suffix(".log")
    try:
        if which == "A":
            argv = ["bash", cfg["legacy_runner"], "start", "--layout", layout,
                    "--build", str(build), "--no-gpu-power"]
            # Both arms must read the same route-heat bytes: the native arm takes them from the
            # site file, so the control arm is told the same file instead of its own default.
            heat = tomllib.loads(Path(cfg["site"]).read_text(encoding="utf-8")).get("handoff", {}).get("heat")
            if heat:
                argv += ["--heat", heat]
            if cfg.get("host_pool_mib"):
                argv += ["--host-pool-mib", str(cfg["host_pool_mib"])]
            if cfg.get("ngram_placement"):
                argv += ["--ngram-placement", cfg["ngram_placement"]]
            if cfg["power_apply"]:
                argv.remove("--no-gpu-power")
                argv.append("--gpu-power")
            started = command(argv, timeout=840, cwd=REPO,
                              env={**os.environ, "BC_WORK_ROOT": str(work_root),
                                   "BC_MODEL_ROOT": cfg["model_root"],
                                   # the launcher's resource derivation asks bc-serve for the layout
                                   "BIGCHERRY_LAYOUTS": str(REPO / "engines/radiance/layouts")})
            launch_log.write_text(started.stdout + "\n" + started.stderr)
            if started.returncode:
                raise Refusal(f"Bash start failed: {started.returncode}: {started.stderr[-800:]}")
            pids = observed_legacy(work_root)
        else:
            argv = [str(bc_serve), "serve", config, "--configs", cfg["configs"],
                    "--site", cfg["site"], "--model-root", cfg["model_root"],
                    "--run-dir", str(run_dir), "--engine-binary", str(build / "bin/radiance"),
                    "--power-profile", cfg["power_profile"]]
            if cfg["power_apply"]:
                argv.append("--allow-power-apply")
            if cfg.get("host_pool_mib"):
                argv += ["--set", f"memory.host_pool_mib={cfg['host_pool_mib']}"]
            if cfg.get("ngram_placement"):
                argv += ["--set", f'engine.ngram_placement="{cfg["ngram_placement"]}"']
            log = launch_log.open("wb")
            try:
                owner = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, "BC_WORK_ROOT": str(work_root)})
            finally:
                log.close()
            deadline = time.monotonic() + 840
            while time.monotonic() < deadline:
                if owner.poll() is not None:
                    raise Refusal(f"native owner exited early: {owner.returncode}")
                try:
                    pids, status = observed_native(bc_serve, run_dir)
                    break
                except (Refusal, ValueError, subprocess.TimeoutExpired):
                    # The owner answers status only between its start-up phases: a
                    # measuring start can run for minutes, so a slow answer is "not yet".
                    time.sleep(2)
            else:
                raise Refusal("native owner startup timed out before running status")
        snapshot, owned = capture(pids)
        save(out.with_suffix(".snapshot.json"), snapshot)
        port = snapshot["manifest"]["spec"]["processes"]["engine"]["server"]["port"]
        health = http_request(port, "/health", None, timeout=15)
        canary = complete(port, CANARY_PROMPT)
        if CANARY_EXPECT.lower() not in canary.lower():
            raise Refusal("canary was not semantically correct")
        completion = complete(port, GREEDY_PROMPT)
        power_snapshot = command(["rocm-smi"], timeout=30)
        if power_snapshot.returncode:
            raise Refusal("power observation failed before stop")
        record.update(power_observed=power_snapshot.stdout,
                      state="observed", health=health, canary=canary,
                      completion=completion, completion_sha256=hashlib.sha256(completion.encode()).hexdigest(),
                      manifest_sha256=snapshot["manifest_file_sha256"],
                      process_roles=sorted(owned), status=status,
                      power_before_stop=power_evidence(run_dir, work_root))
        save(out, record)
    except Exception as error:
        record.update(state="failed", error=str(error))
        save(out, record)
        raise
    finally:
        # Let the real owner stop *all* children and restore power. Never kill
        # an unrelated/reused PID or claim clean stop on a failed restore.
        stop_argv = (["bash", cfg["legacy_runner"], "stop"] if which == "A"
                     else [str(bc_serve), "stop", "--run-dir", str(run_dir)])
        try:
            stopped_result = command(stop_argv, timeout=180, cwd=REPO,
                                     env={**os.environ, "BC_WORK_ROOT": str(work_root)})
            stopped = stopped_result.returncode == 0
            record["stop"] = {"returncode": stopped_result.returncode,
                              "stdout": stopped_result.stdout, "stderr": stopped_result.stderr}
        except (subprocess.SubprocessError, OSError) as error:
            record["stop"] = {"error": str(error)}
        if owner is not None:
            try:
                owner.wait(timeout=60)
            except subprocess.TimeoutExpired:
                record["stop"]["owner_still_alive"] = True
                stopped = False
        live = {role: pid for role, (pid, stamp) in owned.items() if alive((pid, stamp))}
        try:
            remaining = observed_legacy(work_root) if which == "A" else {}
            if which == "B" and owner is not None and owner.poll() is None:
                remaining["session_owner"] = owner.pid
            if remaining:
                live.update(remaining)
        except (OSError, ValueError) as error:
            live["unverified_pids"] = str(error)
        after_power = command(["rocm-smi"], timeout=30)
        record["power_after_observation"] = {
            "returncode": after_power.returncode, "stdout": after_power.stdout,
            "stderr": after_power.stderr}
        record["power_after_stop"] = power_evidence(run_dir, work_root)
        record["clean_stop"] = stopped and not live and after_power.returncode == 0
        record["remaining_owned_roles"] = live
        record["state"] = "stopped" if record["clean_stop"] and record["state"] == "observed" else "failed"
        save(out, record)
        if not record["clean_stop"]:
            raise Refusal("STOP/POWER REFUSAL: owner or child still live; abort further arms")


def accept(args: argparse.Namespace) -> int:
    if args.confirm != "I-WILL-RUN-ON-LEASED-GPUS":
        raise Refusal("explicit hardware confirmation missing; use dry-run for CI")
    for name in ("build", "configs", "model_root", "work_root"):
        require_absolute_dir(getattr(args, name), name)
    for name in ("bc_serve", "site", "model_file", "legacy_runner"):
        require_absolute_file(getattr(args, name), name)
    if not shutil.which("agent-lease"):
        raise Refusal("agent-lease is required, no unleased fallback")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg.pop("confirm")
    cfg["expected_build"] = build_identity({**cfg, "cohort": "helper"})
    save(out / "inputs.json", cfg)
    for cohort in args.cohorts.split(","):
        if cohort not in COHORTS:
            raise Refusal(f"unknown cohort: {cohort}")
    if not 1 <= args.pairs <= len(ORDER):
        raise Refusal(f"--pairs must be 1..{len(ORDER)}")
    for cohort in args.cohorts.split(","):
        for pair_number, order in enumerate(ORDER[:args.pairs], 1):
            results = []
            for which in order:
                label = f"{cohort}-pair{pair_number}-{which}"
                preflight = out / f"{label}-preflight.txt"
                checks = []
                for argv in (["rocm-smi"], ["free", "-g"]):
                    row = command(argv, timeout=30)
                    checks.append({"command": argv, "returncode": row.returncode,
                                   "stdout": row.stdout, "stderr": row.stderr})
                    if row.returncode:
                        raise Refusal(f"{label}: pre-lease check failed: {argv}")
                preflight.write_text(json.dumps(checks, indent=2))
                cell = {**cfg, "cohort": cohort, "arm": which,
                        "out": str(out / (label + ".json"))}
                config_file = out / (label + ".input.json")
                save(config_file, cell)
                lease = ["agent-lease", "acquire", "gpu", "--ttl", "30m", "--est", "15m",
                         "--purpose", f"RSC02 S15 {label}", "--wait", "--",
                         sys.executable, str(Path(__file__).resolve()), "_arm", str(config_file)]
                completed = command(lease, timeout=1850, cwd=REPO)
                (out / (label + ".lease.log")).write_text(completed.stdout + "\n" + completed.stderr)
                if completed.returncode:
                    raise Refusal(f"{label}: arm failed with {completed.returncode}; next lease REFUSED")
                result = json.loads(Path(cell["out"]).read_text())
                if result["state"] != "stopped" or not result["clean_stop"]:
                    raise Refusal(f"{label}: clean shutdown not verified")
                # The A control is retained as experimental comparison evidence.
                # Only the native B arm has a sealed session-start receipt.
                snapshot_file = Path(cell["out"]).with_suffix(".snapshot.json")
                if which == "B":
                    sys.path.insert(0, str(REPO / "tools"))
                    from bigcherry.bench.radiance_lab_records import publish_rsc02_arm

                    snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
                    record_digest = publish_rsc02_arm(
                        result, snapshot, work_root=Path(cell["work_root"]),
                    )
                    (out / f"{label}.record.sha256").write_text(record_digest + "\n")
                results.append((which, result, snapshot_file))
            snaps = dict((which, json.loads(snap.read_text())) for which, _, snap in results)
            report = compare(snaps["A"], snaps["B"])
            hashes = {which: result["completion_sha256"] for which, result, _ in results}
            report["greedy_sha256"] = hashes
            report["greedy_equal"] = len(set(hashes.values())) == 1
            report["clean_stops"] = all(r["clean_stop"] for _, r, _ in results)
            report["status"] = ("PASS" if report["equal"] and report["greedy_equal"]
                                and report["clean_stops"] else "FAIL")
            save(out / f"{cohort}-pair{pair_number}-comparison.json", report)
            if report["status"] != "PASS":
                raise Refusal(f"{cohort} pair {pair_number}: semantic or greedy mismatch; stop")
    save(out / "summary.json", {"status": "SHORT_AB_PASS", "pairs_per_cohort": args.pairs,
                                 "cohorts": args.cohorts.split(","),
                                 "hardware_full_s15": "NOT_PROVEN: RHP12 and quality/retained-session gates remain"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    rehearsal = sub.add_parser("dry-run", help="CI: real native owner/fake engine S15 rehearsal")
    rehearsal.add_argument("--native-test", type=Path, required=True)
    real = sub.add_parser("accept", help="leased short dense/helper A/B; never run in CI")
    for name in ("build", "bc-serve", "configs", "site", "model-root",
                 "model-file", "work-root", "out"):
        real.add_argument("--" + name, required=True)
    real.add_argument("--legacy-runner", required=True,
                      help="absolute archived Bash control from a pre-S16 checkout")
    real.add_argument("--pairs", type=int, default=1,
                      help="A/B pairs per cohort in ABBA order (1 proves launch equivalence; 4 adds drift control)")
    real.add_argument("--cohorts", default="dense,helper", help="comma list of cohorts to run")
    real.add_argument("--host-pool-mib", type=int, default=0,
                      help="pin the helper cohort's host pool for BOTH arms (0: each arm derives its own)")
    real.add_argument("--ngram-placement", default="disk", choices=("disk", "ram", ""),
                      help="n-gram table placement for BOTH arms: a 91 GiB host cannot hold the 47.7 GiB table beside the pool")
    real.add_argument("--power-profile", default="stock")
    real.add_argument("--power-apply", action="store_true")
    real.add_argument("--confirm", required=True)
    internal = sub.add_parser("_arm", help=argparse.SUPPRESS)
    internal.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "dry-run":
            if [arm for pair in ORDER for arm in pair] != list("ABBAABBA"):
                raise Refusal("ABBA schedule changed")
            return dry_run(args.native_test)
        if args.mode == "_arm":
            cfg = json.loads(args.config.read_text())
            if cfg["arm"] not in ("A", "B") or cfg["cohort"] not in COHORTS:
                raise Refusal("invalid arm/cohort")
            arm(cfg)
            return 0
        return accept(args)
    except (Refusal, ProjectionError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"RSC02 S15 REFUSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
