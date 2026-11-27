"""Versioned on-server workloads and a single append-only, card-free-inspectable result index.

The caller owns server lifecycle and GPU leases. This command NEVER launches a server,
changes power, or acquires a GPU lease. Running a workload sends inference requests.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import re
import statistics
import subprocess
import threading
import time
import tomllib
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ..core import engines, feature_states, host_facts, measurement_producers
from .served_model import served_model
from ..core.context import resolve_work_root
from . import stream_bench

CATALOG = Path(__file__).with_name("workloads.toml")
DEFAULT_RUNS = resolve_work_root() / "results/workloads"
SCHEMA = "bigcherry.workload-result.v1"


class WorkloadError(ValueError):
    pass


def catalog(path: Path = CATALOG) -> dict:
    data = tomllib.loads(path.read_text(encoding="utf-8"))["workload"]
    for key, cfg in data.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]+", key) or int(cfg["version"]) < 1:
            raise WorkloadError(f"invalid workload id/version: {key}")
        if cfg["kind"] not in ("doc", "cold", "stream", "power"):
            raise WorkloadError(f"unknown kind for {key}")
    return data


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def request(url: str, prompt: str, max_tokens: int, timeout: float = 900.0,
            *, ignore_eos: bool = True) -> tuple[float, dict]:
    payload = {"model": served_model(url), "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "ignore_eos": ignore_eos}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        json.dumps(payload).encode(), {"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    usage = result.get("usage") or {}
    if not isinstance(usage.get("prompt_tokens"), int) or not isinstance(usage.get("completion_tokens"), int):
        raise WorkloadError("server response lacks exact usage; refusing to record estimated throughput")
    return elapsed, result


def doc_blob(repo_root: Path) -> tuple[str, str]:
    paths = sorted((repo_root / "docs").rglob("*.md"))
    if not paths:
        raise WorkloadError(f"no docs/*.md files found in {repo_root}")
    blob = "\n\n".join(p.read_text(errors="ignore") for p in paths)
    return blob, hashlib.sha256(blob.encode()).hexdigest()


def doc_slices(blob: str, target: int, first: int, count: int, ratio: float = 3.9):
    """Exact slice/offset rules from runs/policy-onoff-ab/doc_prefill.py."""
    chars = int(target * ratio)
    for k in range(first, first + count):
        offset = (k * 7919 * target) % max(1, len(blob) - chars)
        yield k, f"[doc {target} {k}] " + blob[offset:offset + chars]


def median_values(key: str, samples: list[dict]) -> dict:
    values = [float(s[key]) for s in samples]
    if not values:
        raise WorkloadError("no measurements")
    return {"median": statistics.median(values), "per_rep": samples, "unit": "ms" if key.endswith("_ms") else "tok/s"}


def _run_doc(cfg: dict, url: str, repo_root: Path) -> tuple[dict, dict]:
    blob, corpus_sha = doc_blob(repo_root)
    reps = []
    for k, prompt in doc_slices(blob, cfg["target_tokens"], 1, cfg["reps"], cfg["chars_per_token"]):
        elapsed, result = request(url, prompt, cfg["max_tokens"])
        count = result["usage"]["prompt_tokens"]
        if count <= 0 or elapsed <= 0:
            raise WorkloadError("invalid prefill token count/elapsed")
        reps.append({"slice": k, "prompt_tokens": count, "seconds": round(elapsed, 5),
                     "prefill_tps": round(count / elapsed, 3)})
    return {"prefill_tps": median_values("prefill_tps", reps)}, {
        "corpus_sha256": corpus_sha, "corpus": "checkout docs/**/*.md sorted by path",
        "completed_reps": len(reps),
    }


def _stream_reps(cfg: dict, url: str, *, cold: bool) -> tuple[dict, dict]:
    rounds = []
    count = 1 if cold else cfg["streams"]
    rep_count = cfg["reps"]
    for i in range(rep_count if cold else 1):
        record = stream_bench.run(url, served_model(url), [count], cfg["target_tokens"], cfg["max_tokens"],
                                  1 if cold else rep_count, temperature=0, timeout=900,
                                  seed=cfg["seed_first"] + i if cold else cfg["seed"],
                                  shared_prefix=False, allow_estimate=False, calibrate=True)
        if not record["ok"]:
            raise WorkloadError("stream-bench failed: refusing partial result")
        rounds.extend(record["rounds"])
    if cold:
        reps = [{"rep": i + 1, "ttft_ms": r["aggregate"]["ttft_ms_max"],
                 "prompt_tokens": r["per_stream"][0]["prompt_tokens"],
                 "seed": cfg["seed_first"] + i} for i, r in enumerate(rounds)]
        metrics = {"ttft_ms": median_values("ttft_ms", reps)}
    else:
        reps = [{"rep": i + 1, "decode_aggregate_tps": r["aggregate"]["aggregate_tps"],
                 "decode_engine_tps": (r.get("engine") or {}).get("decode_tps_engine"),
                 "ttft_ms": r["aggregate"]["ttft_ms_max"],
                 "prompt_tokens": [v["prompt_tokens"] for v in r["per_stream"]],
                 "estimated_tokens": r["aggregate"]["estimated_tokens"]}
                for i, r in enumerate(rounds)]
        metrics = {"decode_aggregate_tps": median_values("decode_aggregate_tps", reps),
                   "ttft_ms": median_values("ttft_ms", reps)}
        valid = [r["decode_engine_tps"] for r in reps if r["decode_engine_tps"] is not None]
        if len(valid) == len(reps):
            metrics["decode_engine_tps"] = {"median": statistics.median(valid), "per_rep": valid, "unit": "tok/s"}
    return metrics, {"completed_reps": len(rounds), "streams": count,
                     "throughput_definition": "stream-bench finite-batch makespan; prefill excluded"}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def _manifest_for_server(url: str, work_root: Path) -> Path | None:
    """Use live engine cmdline's exact --manifest, never newest historical manifest."""
    port = urllib.parse.urlparse(url).port
    pid_file = work_root / "flashnext/pids/engine"
    pid = _read_text(pid_file)
    if not pid or not pid.isdecimal():
        return None
    raw = _read_text(Path(f"/proc/{pid}/cmdline"))
    if not raw:
        return None
    args = raw.split("\x00")
    try:
        path = Path(args[args.index("--manifest") + 1])
        manifest = json.loads(path.read_text())
        actual_port = manifest["spec"]["processes"]["engine"]["server"]["port"]
        return path if actual_port == port else None
    except (ValueError, KeyError, IndexError, OSError, json.JSONDecodeError):
        return None


def manifest_info(path: Path | None) -> dict:
    if path is None:
        return {"manifest": None, "build_path": None, "build_commit": None, "layout": None,
                "policy": {"state": "unknown", "calibration_key": None}, "options": {},
                "model": None}
    m = json.loads(path.read_text())
    s = m["spec"]
    e = s["processes"]["engine"]
    h = s.get("handoff", {}).get("placement", {})
    runtime = e.get("runtime", {})
    build = Path(runtime["radiance_home"]).parent if runtime.get("radiance_home") else None
    build_meta = {}
    if build:
        f = build / "bigcherry-build-metadata-radiance.json"
        if f.exists():
            build_meta = json.loads(f.read_text())
    return {
        "manifest": str(path),
        "manifest_digest": m.get("metadata", {}).get("content_digest"),
        "build_path": str(build) if build else None,
        "build_commit": build_meta.get("commit") or build_meta.get("source_commit"),
        "build_id": build_meta.get("build_id"),
        "build_source_slice_id": build_meta.get("source_slice_id"),
        "layout": s.get("intent", {}).get("layout"),
        "model": s.get("intent", {}).get("model"),
        "policy": {"state": "on" if "live_policy" in h else "off",
                   "calibration_key": h.get("live_policy", {}).get("calibration_key")
                   or None},
        "options": {
            "spec_depth": e.get("execution", {}).get("num_speculative_tokens"),
            "chunk": e.get("scheduler", {}).get("max_num_batched_tokens"),
            "host_pool_mib": s.get("resources", {}).get("host", {}).get("pool_mib"),
            "host_kv_mib": s.get("resources", {}).get("host", {}).get("kv_mib"),
            "max_sessions": s.get("intent", {}).get("service", {}).get("max_sessions"),
            "kv_dtype": e.get("execution", {}).get("kv_cache_dtype"),
        },
    }


def _run_power(cfg: dict, url: str) -> tuple[dict, dict]:
    # Passive hwmon readings while sending the same prompts as PWR01.
    duration = cfg["phase_seconds"]
    active_phase = ["idle_loaded"]
    readings: dict[str, list] = {p: [] for p in ("idle_loaded", "decode", "prefill", "idle_after")}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            readings[active_phase[0]].append(host_facts.board_watts())
            stop.wait(1.0)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    metrics = {}
    try:
        for phase in ("idle_loaded", "decode", "prefill", "idle_after"):
            active_phase[0] = phase
            if phase.startswith("idle"):
                time.sleep(duration + cfg["idle_settle_seconds"])
                continue
            start, n, reps = time.perf_counter(), 0, []
            while time.perf_counter() - start < duration:
                if phase == "decode":
                    prompt = "Write a long story about a lighthouse keeper and the sea."
                    max_tokens, token_key = cfg["decode_max_tokens"], "completion_tokens"
                else:
                    prompt = f"Request {uuid.uuid4().int % 32768}-{len(reps)}-{time.time_ns()}. " + (
                        "The quick brown fox jumps over the lazy dog. " * 800) + "Summarise."
                    max_tokens, token_key = cfg["prefill_max_tokens"], "prompt_tokens"
                elapsed, response = request(url, prompt, max_tokens, ignore_eos=False)
                tokens = response["usage"][token_key]
                if tokens < 1:
                    raise WorkloadError(f"{phase}: empty usage token count")
                n += tokens
                reps.append({"tokens": tokens, "seconds": elapsed})
            seconds = time.perf_counter() - start
            metrics[f"{phase}_tps"] = {"median": round(n / seconds, 3), "per_rep": reps,
                                       "unit": "tok/s", "phase_seconds": round(seconds, 3),
                                       "total_tokens": n}
    finally:
        stop.set()
        thread.join(timeout=3)
    watts = {}
    for phase, snapshots in readings.items():
        bdfs = sorted({bdf for s in snapshots for bdf in s})
        watts[phase] = {bdf: statistics.median(s[bdf] for s in snapshots if bdf in s)
                        for bdf in bdfs}
    metrics["board_watts"] = {"median": watts, "per_rep": readings, "unit": "W"}
    return metrics, {"phase_seconds": duration, "power_sensor": "sysfs:power1_average",
                     "cpu_power": "not_sampled"}


def execute(workload_id: str, url: str, repo_root: Path, cfg: dict) -> tuple[dict, dict]:
    kind = cfg["kind"]
    if kind == "doc":
        return _run_doc(cfg, url, repo_root)
    if kind == "cold":
        return _stream_reps(cfg, url, cold=True)
    if kind == "stream":
        return _stream_reps(cfg, url, cold=False)
    if kind == "power":
        return _run_power(cfg, url)
    raise WorkloadError(kind)


def make_row(workload_id: str, cfg: dict, metrics: dict, detail: dict,
             info: dict, *, run_folder: str, source: str = "measured",
             stamp: str | None = None, power: dict | None = None,
             kernel: str | None = None) -> dict:
    return {"schema": SCHEMA, "row_id": uuid.uuid4().hex[:16], "timestamp": stamp or timestamp(),
            "source": source, "workload_id": workload_id, "workload_version": cfg["version"],
            "metrics": metrics, "details": detail, "build_path": info.get("build_path"),
            "build_commit": info.get("build_commit"), "build_id": info.get("build_id"),
            "build_source_slice_id": info.get("build_source_slice_id"),
            "runner_commit": detail.get("runner_commit"),
            "layout": info.get("layout"), "kernel": kernel or platform.release(),
            "model": info.get("model"), "policy": info.get("policy", {"state": "unknown", "calibration_key": None}),
            "power_state": power if power is not None else host_facts.power_state(),
            "options": info.get("options", {}), "manifest": info.get("manifest"),
            "manifest_digest": info.get("manifest_digest"), "run_folder": run_folder}


def _metric_phase(name: str) -> str | None:
    if name.startswith("prefill") or name == "prefill_tps":
        return "prefill"
    if name.startswith("decode"):
        return "decode"
    if name == "ttft_ms":
        return "ttft"
    return None


def publish_record(
    row: dict,
    *,
    manifest_bytes: bytes | None = None,
    work_root: Path | None = None,
    argv: list[str] | None = None,
    environment: dict[str, str] | None = None,
    host_snapshot: dict | None = None,
) -> str:
    """MRC06 new writer; historical row payloads are kept only as source blobs.

    The manifest's bytes must still exist at the time of capture. A historical
    path or an unverified server is not enough to claim a valid build identity.
    """
    manifest_name = row.get("manifest")
    if not manifest_name:
        raise WorkloadError("verified active manifest required for run-record publication")
    manifest_path = Path(manifest_name)
    raw = manifest_bytes if manifest_bytes is not None else manifest_path.read_bytes()
    manifest = json.loads(raw)
    spec = manifest["spec"]
    build = spec["processes"]["engine"]["runtime"]["radiance_home"]
    build_root = Path(build).parent
    metadata_file = build_root / f"bigcherry-build-metadata-{engines.RADIANCE}.json"
    meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    for name in ("build_id", "binary_hash", "source_slice_id", "build_plan_id"):
        if not isinstance(meta.get(name), str) or not meta[name]:
            raise WorkloadError(f"verified build metadata requires {name}")
    if (row.get("build_id"), row.get("layout"), row.get("model")) != (
        meta["build_id"], spec["intent"]["layout"], spec["intent"]["model"]
    ):
        raise WorkloadError("result identity differs from captured strict manifest and build metadata")
    # The engine binary is the runtime artifact whose recorded hash is the build's binary hash (no engine name here).
    artifacts = meta.get("runtime_artifacts")
    named = [n for n, h in artifacts.items() if h == meta.get("binary_hash")] if isinstance(artifacts, dict) else []
    if len(named) != 1:
        raise WorkloadError("build metadata does not name exactly one runtime artifact for its binary hash")
    binary = build_root / "bin" / named[0]
    # The build's metadata is the record-of-build; if the binary is present,
    # enforce that its actual contents still match that metadata.
    if not isinstance(meta["binary_hash"], str) or len(meta["binary_hash"]) != 64:
        raise WorkloadError("build binary SHA-256 missing")
    if binary.is_file() and __import__("hashlib").sha256(binary.read_bytes()).hexdigest() != meta["binary_hash"]:
        raise WorkloadError("build binary no longer matches build metadata")
    # Bind the measured row to the running native owner and its start seal.
    # An independent Python snapshot cannot replace a missing native receipt.
    session_path = manifest_path.parent / "session.json"
    if session_path.is_symlink() or not session_path.is_file():
        raise WorkloadError("native owner session receipt required for measurement")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if Path(session.get("manifest", "")).resolve() != manifest_path.resolve() or (
        session.get("state") not in ("running", "healthy")
    ):
        raise WorkloadError("native owner session does not own the measured manifest")
    try:
        owner_start = host_facts.read_owner_start(manifest_path.parent)
    except (ValueError, TypeError, KeyError) as exc:
        raise WorkloadError(f"native owner start receipt required: {exc}") from exc
    if host_snapshot is not None and host_snapshot != owner_start:
        raise WorkloadError("measured host facts differ from sealed native owner start")
    state_path = manifest_path.parent / feature_states.FILE_NAME
    states = feature_states.read_feature_states(state_path)
    if any(state["engine"] != engines.RADIANCE for state in states):
        raise WorkloadError("feature states and engine identity differ")

    phase_metrics: dict[str, dict[str, float]] = {p: {} for p in ("decode", "prefill", "ttft")}
    metric_specs: dict[str, dict[str, dict[str, Any]]] = {p: {} for p in phase_metrics}
    for name, entry in row["metrics"].items():
        phase = _metric_phase(name)
        if phase is None:
            continue  # board_watts is structured data; retained in exact row blob
        value = entry["median"]
        if type(value) not in (float, int):
            raise WorkloadError(f"non-numeric measured metric {name}")
        phase_metrics[phase][name] = value
        metric_specs[phase][name] = {
            "direction": "lower" if name == "ttft_ms" else "higher",
            "gated": name != "decode_engine_tps",
        }
        # Cold requests use distinct seeds, docs use different slices. Only
        # same-seed stream repeats may enlarge baseline tolerance.
        if row["workload_id"].startswith("decode-") and name in ("decode_aggregate_tps", "ttft_ms"):
            reps = [rep.get(name) for rep in entry.get("per_rep", [])
                    if isinstance(rep, dict) and type(rep.get(name)) in (int, float)]
            if len(reps) >= 2:
                metric_specs[phase][name]["repeat_groups"] = [reps]
    if not any(phase_metrics.values()) and row["workload_id"] != "stream-bench":
        raise WorkloadError("no numeric workload metrics")
    stamp = row["timestamp"]
    config = row.get("details", {}).get("workload_config")
    if not isinstance(config, dict):
        raise WorkloadError("record requires exact workload config")

    facts = owner_start
    return measurement_producers.publish(
        work_root=work_root,
        run_id=row["row_id"],
        recorded_at=stamp,
        identity={
            "engine": engines.RADIANCE, "build_id": meta["build_id"],
            "binary_sha256": meta["binary_hash"],
            "source_revision": row.get("runner_commit") or meta.get("source_commit") or row.get("build_commit") or "",
            "model": spec["intent"]["model"],
            "layout": spec["intent"]["layout"],
        },
        workload={
            "name": row["workload_id"],
            "definition": {**config, "version": row["workload_version"]},
        },
        metrics=phase_metrics, metric_specs=metric_specs,
        effective_config=raw,
        argv=argv or ["bigcherry", "workload", "run", row["workload_id"]],
        environment=environment or {},
        power_settings=row.get("power_state") or {},
        inputs={
            "workload_result": measurement_producers.canonical_blob(row),
            "build_metadata": metadata_file.read_bytes(),
            "native_feature_states": state_path.read_bytes(),
            "native_host_start": (manifest_path.parent / host_facts.START_RECEIPT_FILE).read_bytes(),
            "native_session": session_path.read_bytes(),
        },
        host_facts=facts, feature_states=states,
        validity={"state": "unverified", "reason_codes": ["ATTESTATION_MISSING"]},
    )


def live_info(url: str, work_root: Path, manifest: Path | None) -> dict:
    if manifest is not None:
        expected = json.loads(manifest.read_text())["spec"]["processes"]["engine"]["server"]["port"]
        if expected != urllib.parse.urlparse(url).port:
            raise WorkloadError(f"manifest port {expected} differs from server URL port")
    return manifest_info(manifest if manifest is not None else _manifest_for_server(url, work_root))
