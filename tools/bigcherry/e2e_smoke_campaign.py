"""HI78: fast single-model end-to-end smoke campaign.

record -> inventory -> tune -> dispatch-db -> promote -> export -> replay.

Exercises the whole HI29/HI30/HI31/HI67/HI74 chain against one small local
GGUF model in minutes, not the hours a full HI35/HI36-style production
campaign takes. Built from a scratchpad driver script that found 5 real bugs
(1 production-breaking compile bug, 4 correctness/usability bugs) in under an
hour on its first real run -- see ledger event
chg_20260821_121835_fixed-a-production-breaking-co_3880. Intended to run
before landing any change to the tuner/promotion/replay chain, as a much
cheaper alternative to a full production campaign for catching integration
bugs between the C++ emitter and the Python promotion/replay tooling.

Requires two pre-built llama-server binaries (this script does not build
them -- see docs/reference/BUILD.md for cmake invocations):
  --tune-server    built with GGML_HIP_AUTOTUNE=ON, GGML_HIP_AUTOTUNE_RECORD=ON
  --replay-server  built with GGML_HIP_DISPATCH_REPLAY=ON (GGML_HIP_AUTOTUNE off
                    -- these two options are cmake-mutually-exclusive, so this
                    is necessarily a second build tree, not a runtime switch)

Usage:
    python -m bigcherry.e2e_smoke_campaign \\
        --model G:/models/qwen3.5-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf \\
        --tune-server C:/bcw/bin/llama-server.exe \\
        --replay-server C:/bcw-replay/bin/llama-server.exe \\
        --manifest H:/.../artifacts/<rev>/hip-autotune-manifest.json \\
        --workdir C:/scratch/e2e-smoke

Resumable: each stage's output is checked before rerunning it, so a fixed
bug can be validated by rerunning just the stages downstream of the fix
(delete that stage's output file(s) to force a rerun).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROGRESS_INTERVAL_S = 15
COMPLETIONS = (
    {
        "prompt": "Explain the difference between a mutex and a semaphore in "
                  "three sentences.",
        "n_predict": 128,
        "temperature": 0,
    },
    {
        "prompt": "Write a short poem about the ocean, at least twenty lines "
                  "long, covering waves, tides, storms, and calm mornings.",
        "n_predict": 256,
        "temperature": 0,
    },
)


class CampaignError(RuntimeError):
    pass


@dataclass
class Campaign:
    model: Path
    tune_server: Path
    replay_server: Path
    manifest: Path
    workdir: Path
    port: int = 42301
    host: str = "127.0.0.1"
    n_gpu_layers: int = 99
    ctx_size: int = 4096
    stock_bench: Path | None = None
    tune_bench: Path | None = None
    replay_bench: Path | None = None
    bench_prompt: int = 512
    bench_gen: int = 128
    bench_repetitions: int = 3

    def __post_init__(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.logdir = self.workdir / "logs"
        self.logdir.mkdir(exist_ok=True)
        self.status_path = self.workdir / "status.json"
        self.progress_path = self.workdir / "progress.jsonl"
        if self.stock_bench is not None:
            self.stock_bench = Path(self.stock_bench)
        if self.tune_bench is not None:
            self.tune_bench = Path(self.tune_bench)
        if self.replay_bench is not None:
            self.replay_bench = Path(self.replay_bench)
        self.bench_prompt = int(self.bench_prompt)
        self.bench_gen = int(self.bench_gen)
        self.bench_repetitions = int(self.bench_repetitions)
        if self.bench_prompt <= 0:
            raise ValueError("bench_prompt must be > 0")
        if self.bench_gen <= 0:
            raise ValueError("bench_gen must be > 0")
        if self.bench_repetitions <= 0:
            raise ValueError("bench_repetitions must be > 0")

    # -- status/progress -----------------------------------------------

    def write_status(self, stage: str, state: str, detail: str = "") -> None:
        record = {
            "stage": stage,
            "state": state,
            "detail": detail,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        data: dict = {}
        if self.status_path.exists():
            try:
                data = json.loads(self.status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        data["current"] = record
        data.setdefault("history", []).append(record)
        self.status_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[{record['ts']}] {stage} {state} {detail}", flush=True)

    def sample_progress(self, stage: str, journal_path: Path, t0: float) -> None:
        lines = 0
        if journal_path.exists():
            with journal_path.open(encoding="utf-8", errors="replace") as f:
                lines = sum(1 for _ in f)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "journal_lines": lines,
            "elapsed_s": round(time.time() - t0, 1),
        }
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # -- server lifecycle -------------------------------------------------

    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _wait_healthy(self, proc: subprocess.Popen, timeout_s: int = 180) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            # Fail fast on an early crash (e.g. a missing ROCm DLL on PATH)
            # instead of spinning for the full timeout on a process that
            # already exited.
            if proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(
                    f"{self._base_url()}/health", timeout=2
                ) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(3)
        return False

    def _post_json(self, path: str, payload: dict, timeout_s: int = 120) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url()}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s):
                pass
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise CampaignError(f"POST {path} failed: {exc}") from exc

    def _run_completions(self) -> None:
        for payload in COMPLETIONS:
            self._post_json("/completion", payload)

    def _shutdown(self, proc: subprocess.Popen, timeout_s: int = 90) -> None:
        try:
            self._post_json("/shutdown", {})
        except CampaignError:
            pass
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    def _launch(
        self, server: Path, env_overrides: dict[str, str], log_path: Path
    ) -> subprocess.Popen:
        env = dict(os.environ)
        env.update(env_overrides)
        # LLAMA_SERVER_ENABLE_SHUTDOWN gates the opt-in /shutdown route
        # (patches/0800_server_shutdown_endpoint.py) -- without it a plain
        # process kill on Windows skips backend teardown and silently
        # discards buffered HIP autotune measurements (found the hard way
        # this session: an earlier version of this campaign used a bare
        # kill and every tune run came back empty).
        env["LLAMA_SERVER_ENABLE_SHUTDOWN"] = "1"
        args = [
            str(server), "-m", str(self.model),
            "--port", str(self.port), "--host", self.host,
            "-ngl", str(self.n_gpu_layers), "-fa", "on",
            "--ctx-size", str(self.ctx_size), "--no-webui",
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                args, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            )
        return proc

    def run_server_stage(
        self, stage: str, server: Path, env_overrides: dict[str, str],
        journal_path: Path | None = None,
    ) -> None:
        log_path = self.logdir / f"{stage}-server.log"
        proc = self._launch(server, env_overrides, log_path)
        sampler_stop: threading.Event | None = None
        sampler_thread: threading.Thread | None = None
        try:
            if not self._wait_healthy(proc):
                tail = _tail(log_path)
                exited = proc.poll() is not None
                reason = f"process exited (code {proc.returncode})" if exited \
                    else "health check timed out"
                raise CampaignError(f"{stage}: server did not come up -- {reason}: {tail}")
            if journal_path is not None:
                # Sub-minute progress, on its own timer: hip-autotune-
                # journal.cpp fwrite+fflush+fsync's a record per candidate
                # attempt, so this file's line count is a live, durable
                # signal -- not a stale poll. Runs on a background thread so
                # it keeps sampling every PROGRESS_INTERVAL_S seconds while
                # the (blocking) completion requests below are in flight,
                # the same as the bash prototype's concurrent sampler loop.
                sampler_stop = threading.Event()
                t0 = time.time()

                def _sample_loop() -> None:
                    while not sampler_stop.is_set():
                        self.sample_progress(stage, journal_path, t0)
                        sampler_stop.wait(PROGRESS_INTERVAL_S)

                sampler_thread = threading.Thread(target=_sample_loop, daemon=True)
                sampler_thread.start()
            self._run_completions()
        finally:
            if sampler_stop is not None:
                sampler_stop.set()
            if sampler_thread is not None:
                sampler_thread.join(timeout=PROGRESS_INTERVAL_S + 5)
            self._shutdown(proc)

    # -- stages -------------------------------------------------------

    def s1_record(self) -> Path:
        db = self.workdir / "record.jsonl"
        stage = "S1_record"
        if db.exists() and db.stat().st_size > 0:
            self.write_status(stage, "done", f"{_count_lines(db)} observations (resumed)")
            return db
        self.write_status(stage, "running")
        db.unlink(missing_ok=True)
        self.run_server_stage(
            stage, self.tune_server,
            {"GGML_HIP_DISPATCH_MODE": "record", "GGML_HIP_DISPATCH_DB": str(db)},
        )
        if not db.exists() or db.stat().st_size == 0:
            raise CampaignError(f"{stage}: no record output")
        self.write_status(stage, "done", f"{_count_lines(db)} observations")
        return db

    def s1b_inventory(self, record_db: Path) -> tuple[Path, Path]:
        stage = "S1b_inventory"
        inventory_json = self.workdir / "inventory.json"
        inventory_sqlite = self.workdir / "inventory.sqlite"
        if inventory_json.exists():
            self.write_status(stage, "done", "resumed")
            return inventory_json, inventory_sqlite
        self.write_status(stage, "running")
        _run_bigcherry(
            "inventory", "record", str(record_db),
            "--inventory", str(inventory_json), "--database", str(inventory_sqlite),
        )
        self.write_status(stage, "done")
        return inventory_json, inventory_sqlite

    def s2_tune(self) -> Path:
        tdb = self.workdir / "tune.jsonl"
        measurements = Path(f"{tdb}.measurements.jsonl")
        journal = Path(f"{tdb}.journal.jsonl")
        stage = "S2_tune"
        if measurements.exists() and measurements.stat().st_size > 0:
            self.write_status(stage, "done", f"{_count_lines(measurements)} lines (resumed)")
            return measurements
        self.write_status(stage, "running")
        for p in (tdb, measurements, journal):
            p.unlink(missing_ok=True)
        self.run_server_stage(
            stage, self.tune_server,
            {
                "GGML_HIP_DISPATCH_MODE": "tune",
                "GGML_HIP_DISPATCH_DB": str(tdb),
                "GGML_HIP_TUNE_SCREEN_SAMPLES": "3",
                "GGML_HIP_TUNE_FINAL_SAMPLES": "10",
                # HI64: on Windows/WDDM a single transient HIP timing flake
                # permanently poisons tuning for the rest of the process
                # (fail-closed by design; in-process recovery is out of
                # scope until that item lands). HIP_LAUNCH_BLOCKING=1 is
                # the item's own accepted local-gate workaround -- timings
                # become non-authoritative but the run completes, which is
                # what a process-validation smoke test needs.
                "HIP_LAUNCH_BLOCKING": "1",
            },
            journal_path=journal,
        )
        if not measurements.exists() or measurements.stat().st_size == 0:
            raise CampaignError(f"{stage}: no measurements output")
        self.write_status(stage, "done", f"{_count_lines(measurements)} lines")
        return measurements

    def s2c_dispatch_db(self, measurements: Path) -> Path:
        stage = "S2c_dispatch_db"
        dispatch_db = self.workdir / "dispatch.sqlite"
        self.write_status(stage, "running")
        _run_bigcherry(
            "inventory", "tuning", str(measurements),
            "--database", str(dispatch_db), "--manifest", str(self.manifest),
        )
        self.write_status(stage, "done")
        return dispatch_db

    def s3_promote(self, measurements: Path, dispatch_db: Path) -> Path:
        stage = "S3_promote"
        promoted = self.workdir / "promoted.jsonl"
        self.write_status(stage, "running")
        out = _run_bigcherry(
            "tune-promote", str(measurements),
            "--output", str(promoted), "--dispatch-db", str(dispatch_db),
        )
        self.write_status(stage, "done", out.strip().splitlines()[-1] if out.strip() else "")
        return promoted

    def s4_export(self, promoted: Path) -> Path:
        stage = "S4_export"
        cache = self.workdir / "dispatch.cache"
        self.write_status(stage, "running")
        _run_module(
            "bigcherry.replay_cache", str(promoted),
            "--manifest", str(self.manifest), "--output", str(cache),
        )
        if not cache.exists() or cache.stat().st_size == 0:
            raise CampaignError(f"{stage}: no cache produced")
        self.write_status(stage, "done", f"{cache.stat().st_size} bytes")
        return cache

    def s5_replay(self, cache: Path) -> Path:
        stage = "S5_replay"
        coverage = self.workdir / "coverage.json"
        self.write_status(stage, "running")
        coverage.unlink(missing_ok=True)
        self.run_server_stage(
            stage, self.replay_server,
            {
                "GGML_HIP_DISPATCH_MODE": "replay",
                "GGML_HIP_DISPATCH_CACHE": str(cache),
                "GGML_HIP_DISPATCH_COVERAGE": str(coverage),
            },
        )
        if not coverage.exists():
            raise CampaignError(f"{stage}: no coverage output")
        self.write_status(stage, "done", coverage.read_text(encoding="utf-8"))
        return coverage

    # -- bench (stock vs native vs replay tokens/sec) -----------------------

    @staticmethod
    def _bench_clean_env() -> dict[str, str]:
        """A copy of the process environment with all BigCherry HIP dispatch
        configuration removed, so a parent shell or a previous bench arm
        cannot leak dispatch mode/cache configuration into the next arm."""
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("GGML_HIP_DISPATCH_") or key.startswith("BIGCHERRY_"):
                env.pop(key, None)
        return env

    @staticmethod
    def _bench_parse_json(stdout: str) -> list[dict[str, Any]]:
        text = stdout.strip()
        if not text:
            raise CampaignError("llama-bench produced no JSON output")
        # llama-bench.exe prints a non-JSON diagnostic line ("HIP Library
        # Path: ...") to stdout before the JSON payload on Windows/ROCm --
        # skip any preamble up to the first '[' or '{'.
        start = min(
            (i for i in (text.find("["), text.find("{")) if i != -1),
            default=-1,
        )
        if start > 0:
            text = text[start:]
        try:
            value = json.loads(text)
        except json.JSONDecodeError as array_error:
            # Tolerate a JSONL variant (one object per line) as a fallback.
            rows: list[dict[str, Any]] = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    raise CampaignError(
                        "failed to parse llama-bench JSON output"
                    ) from array_error
                if not isinstance(row, dict):
                    raise CampaignError(
                        f"llama-bench JSONL row {line_number} is not an object"
                    )
                rows.append(row)
            if not rows:
                raise CampaignError(
                    "failed to parse llama-bench JSON output"
                ) from array_error
            return rows

        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict) and isinstance(value.get("results"), list):
            rows = value["results"]
        elif isinstance(value, dict):
            rows = [value]
        else:
            raise CampaignError(
                f"unexpected llama-bench JSON root: {type(value).__name__}"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CampaignError(f"llama-bench JSON row {index} is not an object")
        return rows

    @staticmethod
    def _bench_int(row: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _bench_float(row: dict[str, Any], key: str, *, description: str) -> float:
        value = row.get(key)
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise CampaignError(f"{description} has invalid {key!r}: {value!r}") from exc
        if not math.isfinite(result):
            raise CampaignError(f"{description} has non-finite {key!r}: {value!r}")
        return result

    def _bench_select_metric_row(
        self, rows: list[dict[str, Any]], *, workload: str,
    ) -> dict[str, Any]:
        if workload not in {"pp", "tg"}:
            raise ValueError(f"unknown benchmark workload: {workload!r}")
        exact: list[dict[str, Any]] = []
        compatible: list[dict[str, Any]] = []
        labelled: list[dict[str, Any]] = []
        for row in rows:
            n_prompt = self._bench_int(row, "n_prompt", "n_prompts", "pp")
            n_gen = self._bench_int(row, "n_gen", "n_generation", "tg")
            if workload == "pp":
                if n_prompt == self.bench_prompt and (n_gen is None or n_gen == 0):
                    exact.append(row)
                elif n_prompt is not None and n_prompt > 0 and (n_gen is None or n_gen == 0):
                    compatible.append(row)
            else:
                if n_gen == self.bench_gen and (n_prompt is None or n_prompt == 0):
                    exact.append(row)
                elif n_gen is not None and n_gen > 0 and (n_prompt is None or n_prompt == 0):
                    compatible.append(row)
            test_label = str(
                row.get("test") or row.get("test_name") or row.get("name") or ""
            ).strip().lower()
            if workload == "pp" and test_label.startswith("pp"):
                labelled.append(row)
            elif workload == "tg" and test_label.startswith("tg"):
                labelled.append(row)
        candidates = exact or compatible or labelled
        deduplicated: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for row in candidates:
            if id(row) in seen_ids:
                continue
            seen_ids.add(id(row))
            deduplicated.append(row)
        if len(deduplicated) != 1:
            summary = [
                {"n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"),
                 "test": r.get("test"), "avg_ts": r.get("avg_ts")}
                for r in rows
            ]
            raise CampaignError(
                f"expected exactly one llama-bench {workload} row, "
                f"found {len(deduplicated)}; rows={summary!r}"
            )
        return deduplicated[0]

    def _bench_metrics(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for workload in ("pp", "tg"):
            row = self._bench_select_metric_row(rows, workload=workload)
            description = f"llama-bench {workload} row"
            result[workload] = {
                "avg_ts": self._bench_float(row, "avg_ts", description=description),
                "stddev_ts": self._bench_float(row, "stddev_ts", description=description),
            }
        return result

    def _run_bench_config(
        self, *, name: str, binary: Path, dispatch_mode: str | None,
        dispatch_cache: Path | None = None,
    ) -> dict[str, Any]:
        binary = Path(binary)
        if not binary.is_file():
            raise CampaignError(f"{name} llama-bench binary does not exist: {binary}")
        if dispatch_cache is not None:
            dispatch_cache = Path(dispatch_cache)
            if not dispatch_cache.is_file():
                raise CampaignError(f"{name} dispatch cache does not exist: {dispatch_cache}")
        env = self._bench_clean_env()
        env["LLAMA_SERVER_ENABLE_SHUTDOWN"] = "1"
        if dispatch_mode is not None:
            env["GGML_HIP_DISPATCH_MODE"] = dispatch_mode
        if dispatch_cache is not None:
            env["GGML_HIP_DISPATCH_CACHE"] = str(dispatch_cache.resolve())
        command = [
            str(binary.resolve()), "-m", str(Path(self.model).resolve()),
            "-p", str(self.bench_prompt), "-n", str(self.bench_gen),
            "-r", str(self.bench_repetitions), "-o", "json",
        ]
        log_path = self.logdir / f"bench-{name}.log"
        completed = subprocess.run(
            command, cwd=self.workdir, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        log_path.write_text(
            f"command: {command!r}\n\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise CampaignError(
                f"{name} llama-bench failed with exit code {completed.returncode} "
                f"(see {log_path})"
            )
        rows = self._bench_parse_json(completed.stdout)
        result: dict[str, Any] = {
            "binary": str(binary.resolve()),
            "dispatch_mode": dispatch_mode,
            "metrics": self._bench_metrics(rows),
            "rows": rows,
        }
        if dispatch_cache is not None:
            result["dispatch_cache"] = str(dispatch_cache.resolve())
        return result

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def s6_bench(self, cache: Path) -> Path:
        """Benchmark the model three ways: stock (separate unpatched binary,
        no BigCherry dispatch env at all), native (dispatch on, no tuning
        influence), and replay (the promoted/tuned cache applied). Each arm
        runs as its own subprocess from a freshly cleaned environment so no
        GGML_HIP_DISPATCH_* state leaks between arms."""
        stage = "S6_bench"
        output = self.workdir / "bench.json"
        if output.exists():
            self.write_status(stage, "done", "resumed, already present")
            return output
        self.write_status(stage, "running")
        stock = self._run_bench_config(
            name="stock", binary=self.stock_bench, dispatch_mode=None,
        )
        native = self._run_bench_config(
            name="native", binary=self.tune_bench, dispatch_mode="native",
        )
        replay = self._run_bench_config(
            name="replay", binary=self.replay_bench, dispatch_mode="replay",
            dispatch_cache=cache,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "model": str(Path(self.model).resolve()),
            "params": {
                "n_prompt": self.bench_prompt, "n_gen": self.bench_gen,
                "repetitions": self.bench_repetitions,
            },
            "configs": {"stock": stock, "native": native, "replay": replay},
        }
        self._atomic_write_json(output, payload)
        self.write_status(stage, "done")
        return output

    def s7_report(self, bench_path: Path, measurements: Path) -> Path:
        stage = "S7_report"
        report_path = self.workdir / "report.md"
        self.write_status(stage, "running")
        from .e2e_smoke_report import generate_report
        generate_report(
            self.workdir, bench_path=bench_path, measurements_path=measurements,
            output_path=report_path,
        )
        self.write_status(stage, "done")
        return report_path

    def run(self) -> dict:
        record_db = self.s1_record()
        _inventory_json, _inventory_sqlite = self.s1b_inventory(record_db)
        measurements = self.s2_tune()
        dispatch_db = self.s2c_dispatch_db(measurements)
        promoted = self.s3_promote(measurements, dispatch_db)
        cache = self.s4_export(promoted)
        coverage = self.s5_replay(cache)
        if self.stock_bench and self.tune_bench and self.replay_bench:
            bench_path = self.s6_bench(cache)
            self.s7_report(bench_path, measurements)
        self.write_status("CAMPAIGN", "done", "all stages complete")
        return json.loads(coverage.read_text(encoding="utf-8"))


def _tail(path: Path, n: int = 15) -> str:
    if not path.exists():
        return "(no log)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def _run_bigcherry(*args: str) -> str:
    return _run_module("bigcherry", *args)


def _run_module(module: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise CampaignError(
            f"{module} {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry e2e-smoke-campaign")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tune-server", required=True, type=Path)
    parser.add_argument("--replay-server", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=42301)
    parser.add_argument(
        "--stock-bench", type=Path, default=None,
        help="unpatched/stock llama-bench binary; enables the bench+report "
             "stages when given together with --tune-bench/--replay-bench",
    )
    parser.add_argument("--tune-bench", type=Path, default=None,
                         help="bigcherry llama-bench binary for the native-dispatch bench arm")
    parser.add_argument("--replay-bench", type=Path, default=None,
                         help="bigcherry llama-bench binary for the replay-dispatch bench arm")
    parser.add_argument("--bench-prompt", type=int, default=512)
    parser.add_argument("--bench-gen", type=int, default=128)
    parser.add_argument("--bench-repetitions", type=int, default=3)
    args = parser.parse_args(argv)

    campaign = Campaign(
        model=args.model, tune_server=args.tune_server,
        replay_server=args.replay_server, manifest=args.manifest,
        workdir=args.workdir, port=args.port,
        stock_bench=args.stock_bench, tune_bench=args.tune_bench,
        replay_bench=args.replay_bench, bench_prompt=args.bench_prompt,
        bench_gen=args.bench_gen, bench_repetitions=args.bench_repetitions,
    )
    try:
        coverage = campaign.run()
    except CampaignError as exc:
        campaign.write_status("CAMPAIGN", "failed", str(exc))
        print(f"CAMPAIGN FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(coverage, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
