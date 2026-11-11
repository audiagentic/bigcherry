"""Import only identifiable, complete prior benchmark artifacts. No GPU use or leases.

Idempotent by source artifact key. Unknown historical controls remain null/unknown.
The documented Oct 27 '10271712' baseline is corrected by its evidence page to
the preserved, valid pwr01-baseline-10271708 (the earlier 10271703 was cached).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import statistics
from pathlib import Path

from . import workloads as w


def _stamp(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(timespec="seconds")


def _info_from_launch(launch: Path, work_root: Path) -> dict:
    raw = launch.read_text(errors="replace")
    m = re.search(r"engine-(\d{4}-\d{6})\.log", raw)
    candidates = (list((work_root / "flashnext").glob(f"manifest-{m.group(1)}-*.json"))
                  if m else [])
    if len(candidates) == 1:
        return w.manifest_info(candidates[0])
    # Old builds predate exact manifests: retain explicit log statements, never guess options.
    info = w.manifest_info(None)
    layout = re.findall(r"(?:^|\n)(?:layout |Radiance layout )(flashnext-[a-z0-9-]+)", raw)
    build = re.findall(r"(?:^|\n)\s*build\s+([^\s]+)", raw)
    if layout:
        info["layout"] = layout[-1]
    if build:
        info["build_path"] = build[-1]
    info["policy"] = {"state": "unknown", "calibration_key": None}
    return info


def _put(index: Path, existing: set[str], row: dict, key: str) -> bool:
    if key in existing:
        return False
    row["seed_key"] = key
    w.append_index(index, row)
    existing.add(key)
    return True


def _row(workload_id: str, specs: dict, metrics: dict, details: dict, info: dict,
         path: Path, *, kernel: str | None = None, power: dict | None = None, stamp_path: Path | None = None) -> dict:
    details["seeded_from"] = str(path)
    details["historical_metadata"] = "unavailable fields are unknown, not assumed stock"
    return w.make_row(workload_id, specs[workload_id], metrics, details, info,
                      run_folder=str(path), source="seeded", stamp=_stamp(stamp_path or path),
                      power={} if power is None else power, kernel=kernel or "unknown")


def seed_policy(index: Path, existing: set[str], root: Path, specs: dict) -> int:
    folder = root / "runs/policy-onoff-ab"
    if not (folder / "run.sh").exists():
        return 0
    added = 0
    for arm in ("on1", "off1", "off2", "on2"):
        launch = folder / f"{arm}.launch.log"
        if not launch.exists():
            continue
        info = _info_from_launch(launch, root)
        wanted = "on" if arm.startswith("on") else "off"
        if info["policy"]["state"] != wanted:
            # A/B explicitly labels any such arm INVALID; never seed it as valid.
            continue
        # Document prefill, ten 1..10 slices of 44000 nominal tokens.
        doc_files = [folder / f"{arm}.doc{i}.log" for i in range(1, 11)]
        if all(p.is_file() for p in doc_files):
            samples = []
            for i, path in enumerate(doc_files, 1):
                m = re.fullmatch(
                    r"doc slice (\d+): (\d+) prompt tokens in ([\d.]+)s = (\d+) tok/s\s*",
                    path.read_text(),
                )
                if m is None or int(m.group(1)) != i or int(m.group(2)) < 40000:
                    break
                samples.append({"slice": i, "prompt_tokens": int(m.group(2)),
                                "seconds_rounded": float(m.group(3)), "prefill_tps": int(m.group(4))})
            if len(samples) == 10:
                metrics = {"prefill_tps": w.median_values("prefill_tps", samples)}
                details = {"completed_reps": 10, "expected_reps": 10, "corpus_sha256": None,
                           "corpus": "historical main docs/**/*.md; content hash not retained",
                           "metric_precision": "historical printed integer tok/s"}
                row = _row("prefill-doc-44k", specs, metrics, details, info, folder, power={}, stamp_path=launch)
                added += _put(index, existing, row, f"{folder}/{arm}/doc")
        # Cold seeds 501..508: 8 individual stream-bench invocations.
        cold_files = [folder / f"{arm}.cold{i}.json" for i in range(1, 9)]
        if all(p.is_file() for p in cold_files):
            try:
                docs = [json.loads(f.read_text()) for f in cold_files]
                valid = all(d["ok"] and d["config"]["streams"] == [1]
                            and d["config"]["max_tokens"] == 8
                            and d["config"]["prompt_tokens_target"] == 27000
                            and d["config"]["seed"] == 500 + i
                            and len(d["rounds"]) == 1 for i, d in enumerate(docs, 1))
                if valid:
                    samples = [{"rep": i, "seed": 500 + i,
                                "prompt_tokens": d["rounds"][0]["per_stream"][0]["prompt_tokens"],
                                "ttft_ms": d["rounds"][0]["aggregate"]["ttft_ms_max"]}
                               for i, d in enumerate(docs, 1)]
                    metrics = {"ttft_ms": w.median_values("ttft_ms", samples)}
                    row = _row("cold-27k-ttft", specs, metrics,
                               {"completed_reps": 8, "expected_reps": 8}, info, folder, power={}, stamp_path=launch)
                    added += _put(index, existing, row, f"{folder}/{arm}/cold")
            except (KeyError, IndexError, ValueError, json.JSONDecodeError):
                pass
        for stream_count in (1, 6):
            path = folder / f"{arm}.dec{stream_count}.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                cfg = data["config"]
                if not (data["ok"] and cfg["streams"] == [stream_count]
                        and cfg["prompt_tokens_target"] == 900 and cfg["max_tokens"] == 256
                        and cfg["seed"] == 7 and cfg["reps"] == 2 and len(data["rounds"]) == 2
                        and all(not r["aggregate"]["estimated_tokens"] for r in data["rounds"])):
                    continue
                reps = [{"rep": i + 1, "decode_aggregate_tps": r["aggregate"]["aggregate_tps"],
                         "decode_engine_tps": (r.get("engine") or {}).get("decode_tps_engine"),
                         "ttft_ms": r["aggregate"]["ttft_ms_max"]}
                        for i, r in enumerate(data["rounds"])]
                metrics = {"decode_aggregate_tps": w.median_values("decode_aggregate_tps", reps),
                           "ttft_ms": w.median_values("ttft_ms", reps)}
                if all(r["decode_engine_tps"] is not None for r in reps):
                    vals = [r["decode_engine_tps"] for r in reps]
                    metrics["decode_engine_tps"] = {"median": statistics.median(vals),
                                                    "per_rep": vals, "unit": "tok/s"}
                wid = f"decode-{stream_count}stream"
                row = _row(wid, specs, metrics,
                           {"completed_reps": 2, "expected_reps": 2, "streams": stream_count,
                            "throughput_definition": "finite-batch makespan"},
                           info, path, power={}, stamp_path=launch)
                added += _put(index, existing, row, f"{folder}/{arm}/dec{stream_count}")
            except (KeyError, IndexError, ValueError, json.JSONDecodeError):
                continue
    return added


def seed_docpf(index: Path, existing: set[str], root: Path, specs: dict) -> int:
    folder = root / "runs/pwr-stock"
    log, launch = folder / "docpf.out", folder / "docpf.launch.log"
    if not log.exists() or not launch.exists():
        return 0
    samples = []
    for line in log.read_text().splitlines():
        m = re.fullmatch(r"doc slice (\d+): (\d+) prompt tokens in ([\d.]+)s = (\d+) tok/s", line)
        if m is None:
            return 0
        samples.append({"slice": int(m.group(1)), "prompt_tokens": int(m.group(2)),
                        "seconds_rounded": float(m.group(3)), "prefill_tps": int(m.group(4))})
    if not samples or [s["slice"] for s in samples] != list(range(1, len(samples) + 1)):
        return 0
    info = _info_from_launch(launch, root)
    if info["layout"] != "flashnext-4gpu" or info["policy"]["state"] != "off":
        return 0
    details = {"completed_reps": len(samples), "expected_reps": 10,
               "partial_historical_run": len(samples) != 10,
               "corpus_sha256": None, "corpus": "historical main docs/**/*.md; hash not retained",
               "metric_precision": "historical printed integer tok/s"}
    row = _row("prefill-doc-44k", specs,
               {"prefill_tps": w.median_values("prefill_tps", samples)}, details,
               info, folder, power={}, stamp_path=launch)
    return int(_put(index, existing, row, f"{folder}/docpf.out"))


def seed_pwr(index: Path, existing: set[str], root: Path, specs: dict) -> int:
    # Only completed runs with both phase token counters and launch evidence.
    # 10271712 in the old narrative was corrected to the preserved valid 10271708.
    targets = [
        ("pwr01-baseline-10271708", "historical_stock"),
        ("pwr01-baseline-11111036", "stock"),
        ("pwr01-baseline-11111051", "uv125"),
        ("pwr01-baseline-11111104", "stock"),
    ]
    added = 0
    for name, power_label in targets:
        folder = root / "runs" / name
        summary, launch, tokens = (folder / f for f in ("summary.txt", "launch.log", "tokens.txt"))
        if not all(p.exists() for p in (summary, launch, tokens)):
            continue
        info = _info_from_launch(launch, root)
        raw = summary.read_text()
        if name.startswith("pwr01-baseline-1027") and "helpers  XTX 0 1" in launch.read_text():
            info["layout"] = "flashnext-2xtx"
        if info["build_path"] is None:
            b = re.search(r"^build\s+(\S+)", raw, flags=re.MULTILINE)
            if b:
                info["build_path"] = b.group(1)
        c = re.search(r"^commit\s+(\S+)", raw, flags=re.MULTILINE)
        # This is the runner checkout commit, not necessarily the binary's source commit.
        runner_commit = c.group(1) if c else None
        phase_data = {}
        for phase in ("decode", "prefill"):
            m = re.search(rf"^{phase}:\s+(\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s", raw, re.MULTILINE)
            if m is None:
                break
            n, seconds, rate = int(m.group(1)), float(m.group(2)), float(m.group(3))
            phase_data[f"{phase}_tps"] = {
                "median": rate, "per_rep": [{"tokens": n, "seconds": seconds}],
                "unit": "tok/s", "phase_seconds": seconds, "total_tokens": n,
            }
        if len(phase_data) != 2:
            continue
        seconds = 40 if name.startswith("pwr01-baseline-1027") else 60
        phase_data["board_watts"] = {
            "median": {}, "per_rep": [], "unit": "W",
        }
        for line in raw.splitlines():
            if re.match(r"^(gpu[0-9]+|cpu_pkg|total)\s", line):
                parts = line.split()
                if len(parts) == 5:
                    phase_data["board_watts"]["median"][parts[0]] = {
                        phase: float(parts[i + 1])
                        for i, phase in enumerate(("idle_loaded", "decode", "prefill", "idle_after"))
                    }
        if not phase_data["board_watts"]["median"]:
            continue
        # Historical sample protocol used AMD-SMI SOCKET_POWER and CPU RAPL
        # (when available), not today's hwmon sensor.
        details = {"phase_seconds": seconds, "power_sensor": "amd-smi:SOCKET_POWER",
                   "runner_commit": runner_commit, "historical_pwr01": True,
                   "protocol_note": ("old PWR01 script 40s phases; preserved 10271708 is "
                                     "the corrected valid baseline" if seconds == 40
                                     else "PWR01 script 60s phases"),
                   "phase_aggregate_only": True}
        state = {"reported_profile": power_label, "sysfs_readback": None}
        row = _row("pwr-phases", specs, phase_data, details, info, folder, power=state, stamp_path=summary)
        if _put(index, existing, row, str(folder)):
            added += 1
    return added


def seed(index: Path, root: Path, specs: dict | None = None) -> int:
    specs = specs or w.catalog()
    existing = {r["seed_key"] for r in w.read_index(index) if r.get("seed_key")}
    return (seed_policy(index, existing, root, specs)
            + seed_docpf(index, existing, root, specs)
            + seed_pwr(index, existing, root, specs))
