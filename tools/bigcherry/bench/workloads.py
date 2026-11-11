"""Versioned on-server workloads and a single append-only, card-free-inspectable result index.

The caller owns server lifecycle and GPU leases. This command NEVER launches a server,
changes power, or acquires a GPU lease. Running a workload sends inference requests.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
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

from ..core.context import resolve_work_root
from . import stream_bench

CATALOG = Path(__file__).with_name("workloads.toml")
DEFAULT_INDEX = resolve_work_root() / "results/index.jsonl"
DEFAULT_RUNS = resolve_work_root() / "runs/workloads"
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
    payload = {"model": "default", "prompt": prompt, "max_tokens": max_tokens,
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
        record = stream_bench.run(url, "default", [count], cfg["target_tokens"], cfg["max_tokens"],
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


def card_paths(sys_root: Path = Path("/sys")) -> list[tuple[str, Path]]:
    cards = []
    for card in (sys_root / "class/drm").glob("card[0-9]*"):
        device = card / "device"
        if not device.is_dir():
            continue
        # sysfs symlink target ends in the PCI device address.
        bdf = device.resolve().name
        if re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bdf):
            cards.append((bdf.lower(), device))
    return sorted(cards)


def power_state(sys_root: Path = Path("/sys")) -> dict:
    """Read-only per-BDF snapshot; missing controls stay null, never inferred as stock."""
    result = {}
    for bdf, dev in card_paths(sys_root):
        hwmon = list((dev / "hwmon").glob("hwmon*/power1_cap"))
        cap = _read_text(hwmon[0]) if hwmon else None
        od = _read_text(dev / "pp_od_clk_voltage") or ""
        vo = re.search(r"OD_VDDGFX_OFFSET:\s*\n?\s*(-?\d+)\s*m?V", od)
        # OD_SCLK formats differ across driver revisions: never infer a maximum if absent.
        sclk = re.search(r"OD_SCLK:[\s\S]*?(?=OD_[A-Z]+:|$)", od)
        freqs = re.findall(r"\b\d+:\s*(\d+)\s*M?Hz", sclk.group(0)) if sclk else []
        result[bdf] = {
            "power1_cap_uw": int(cap) if cap and cap.isdecimal() else None,
            "vddgfx_offset_mv": int(vo.group(1)) if vo else None,
            "od_sclk_max_mhz": max(map(int, freqs)) if freqs else None,
            "perf_level": _read_text(dev / "power_dpm_force_performance_level"),
        }
    return result


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


def _board_watts(sys_root: Path = Path("/sys")) -> dict:
    readings = {}
    for bdf, dev in card_paths(sys_root):
        power = next(iter((dev / "hwmon").glob("hwmon*/power1_average")), None)
        val = _read_text(power) if power else None
        if val and val.isdecimal():
            readings[bdf] = int(val) / 1e6
    return readings


def _run_power(cfg: dict, url: str) -> tuple[dict, dict]:
    # Passive hwmon readings while sending the same prompts as PWR01.
    duration = cfg["phase_seconds"]
    active_phase = ["idle_loaded"]
    readings: dict[str, list] = {p: [] for p in ("idle_loaded", "decode", "prefill", "idle_after")}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            readings[active_phase[0]].append(_board_watts())
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
            "power_state": power if power is not None else power_state(),
            "options": info.get("options", {}), "manifest": info.get("manifest"),
            "manifest_digest": info.get("manifest_digest"), "run_folder": run_folder}


def append_index(path: Path, row: dict) -> None:
    """One os.write under an advisory lock: no partial/interleaved JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, allow_nan=False, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.write(fd, data) != len(data):
            raise OSError("short index write")
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkloadError(f"{path}:{number}: invalid JSONL: {exc}") from exc
            if obj.get("schema") != SCHEMA:
                raise WorkloadError(f"{path}:{number}: unknown result schema")
            rows.append(obj)
    return rows


def difference(a: dict, b: dict) -> list[str]:
    return [name for name, fn in (
        ("layout", lambda x: x.get("layout")),
        ("build", lambda x: (x.get("build_path"), x.get("build_commit"), x.get("build_id"))),
        ("kernel", lambda x: x.get("kernel")),
        ("model", lambda x: x.get("model")),
        ("policy", lambda x: x.get("policy")),
        ("power", lambda x: x.get("power_state")),
        ("options", lambda x: x.get("options")),
        ("corpus", lambda x: x.get("details", {}).get("corpus_sha256")),
        ("phase", lambda x: (x.get("details", {}).get("phase_seconds"),
                            x.get("details", {}).get("power_sensor"))),
        ("reps", lambda x: x.get("details", {}).get("completed_reps")),
    ) if fn(a) != fn(b)]


def compare(rows: list[dict], workload_id: str, version: int, *,
            layout: str | None = None, since: str | None = None) -> str:
    selected = [r for r in rows if r["workload_id"] == workload_id and r["workload_version"] == version
                and (layout is None or r.get("layout") == layout)
                and (since is None or r["timestamp"][:10] >= since[:10])]
    selected.sort(key=lambda r: r["timestamp"])
    headers = "date                  row              layout              build        policy  metric                    median"
    lines = [f"{workload_id}@{version}: {len(selected)} rows", headers]
    for r in selected:
        metrics = r["metrics"]
        scalar = [k for k in ("prefill_tps", "decode_aggregate_tps", "ttft_ms", "decode_tps", "prefill_tps")
                  if k in metrics]
        name = scalar[0] if scalar else next((k for k in metrics if k != "board_watts"), "board_watts")
        value = metrics[name]["median"]
        if not isinstance(value, (float, int)):
            value = float("nan")
        for metric_name in (("decode_tps", "prefill_tps") if workload_id == "pwr-phases" else (name,)):
            value = metrics[metric_name]["median"]
            lines.append(f"{r['timestamp'][:19]:<21} {r['row_id']:<16} {(r.get('layout') or '?'):<19} "
                         f"{Path(r['build_path']).name[:12] if r.get('build_path') else '?':<12} "
                         f"{r.get('policy', {}).get('state', '?'):<7} {metric_name:<24} {value:.2f}")
    if len(selected) > 1:
        reference = selected[0]
        for r in selected[1:]:
            changes = difference(reference, r)
            if changes:
                lines.append(f"WARNING {r['row_id']} differs from {reference['row_id']}: {', '.join(changes)}")
    if selected and any(r.get("source") == "seeded" for r in selected):
        lines.append("NOTE seeded history may lack per-rep timing, kernel, manifest, or power readback; unknown is not stock.")
    return "\n".join(lines)


def live_info(url: str, work_root: Path, manifest: Path | None) -> dict:
    if manifest is not None:
        expected = json.loads(manifest.read_text())["spec"]["processes"]["engine"]["server"]["port"]
        if expected != urllib.parse.urlparse(url).port:
            raise WorkloadError(f"manifest port {expected} differs from server URL port")
    return manifest_info(manifest if manifest is not None else _manifest_for_server(url, work_root))
