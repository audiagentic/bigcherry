"""Persistent build diagnostics for synchronous, sequential campaigns.

Descriptor capture is process-wide; never surround parallel worker threads.
Logs do not participate in artifact or build identity.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from ..core.process import output_logs

current: ContextVar[BuildLogs | None] = ContextVar("build_logs", default=None)


class _Sink:
    def __init__(self, path):
        self.file = path.open("ab", buffering=0)
        self.lock = threading.Lock()

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")
        with self.lock:
            self.file.write(data)


class _Tee:
    def __init__(self, stream, sink):
        self.stream, self.sink = stream, sink

    def write(self, text):
        self.sink.write(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def _capture(path):
    """Tee Python and inherited child output without buffering builds."""
    sink = _Sink(path)
    token = output_logs.set((*output_logs.get(), sink))
    streams = []
    try:
        for fd, name in ((1, "stdout"), (2, "stderr")):
            stream = getattr(sys, name)
            stream.flush()
            saved = os.dup(fd)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, fd)
            os.close(write_fd)

            def pump(read_fd=read_fd, saved=saved):
                with os.fdopen(read_fd, "rb", buffering=0) as reader:
                    while data := reader.read(65536):
                        sink.write(data)
                        offset = 0
                        while offset < len(data):
                            offset += os.write(saved, data[offset:])

            thread = threading.Thread(target=pump, daemon=True)
            thread.start()
            try:
                native = stream.fileno() == fd
            except (AttributeError, OSError, ValueError):
                native = False
            if not native:
                setattr(sys, name, _Tee(stream, sink))
            streams.append((fd, name, stream, saved, thread))
        yield
    finally:
        for fd, name, stream, saved, thread in reversed(streams):
            getattr(sys, name).flush()
            setattr(sys, name, stream)
            os.dup2(saved, fd)
            thread.join()
            os.close(saved)
        output_logs.reset(token)
        sink.file.close()


class BuildLogs:
    def __init__(self, work_root: Path, run_id: str | None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8]
        if self.run_id in {".", "..", "unspecified"} or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id):
            raise ValueError("run id must be a safe directory name and cannot be 'unspecified'")
        self.root = work_root / "build-logs" / self.run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.lane = None
        self.started = time.monotonic()
        self.summary = {
            "run_id": self.run_id, "status": "running", "duration_seconds": 0,
            "build_log": str(self.root / "build.log"),
            "summary_log": str(self.root / "summary.json"), "lanes": {},
        }
        self.save()

    def save(self):
        self.summary["duration_seconds"] = time.monotonic() - self.started
        temporary = self.root / "summary.json.tmp"
        temporary.write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.root / "summary.json")

    @contextmanager
    def invocation(self):
        token = current.set(self)
        try:
            with _capture(self.root / "build.log"):
                try:
                    yield
                except BaseException:
                    self.summary["status"] = "failed"
                    for sink in output_logs.get():
                        sink.write(traceback.format_exc())
                    raise
                finally:
                    if self.summary["status"] == "running":
                        self.summary["status"] = "failed"
                    self.save()
            # Copies survive independent cleanup of central invocation logs.
            for lane in self.summary["lanes"].values():
                if lane["status"] == "succeeded" and lane["build_dir"]:
                    shutil.copytree(self.root, Path(lane["build_dir"]) / "build-logs" / self.run_id, dirs_exist_ok=True)
        finally:
            current.reset(token)

    @contextmanager
    def lane_scope(self, lane_id):
        previous = self.lane
        self.lane = lane_id
        # Portable names; digest distinguishes sanitised contract identities.
        directory = re.sub(r"[^A-Za-z0-9_.-]", "_", lane_id) + "-" + uuid.uuid5(uuid.NAMESPACE_OID, lane_id).hex[:8]
        record = self.summary["lanes"].setdefault(lane_id, {
            "status": "running", "duration_seconds": 0, "build_plan_id": None,
            "build_dir": None, "log_dir": str(self.root / directory), "stages": {},
        })
        Path(record["log_dir"]).mkdir(exist_ok=True)
        started = time.monotonic()
        try:
            yield
        except BaseException:
            record["status"] = "failed"
            # Resolution/preflight failures can occur outside scheduled stages.
            with self.stage("lane"):
                raise
        else:
            record["status"] = "succeeded"
        finally:
            record["duration_seconds"] = time.monotonic() - started
            self.save()
            self.lane = previous

    @contextmanager
    def stage(self, name):
        if self.lane is None:
            yield
            return
        lane = self.summary["lanes"][self.lane]
        path = Path(lane["log_dir"]) / f"{name}.log"
        record = lane["stages"][name] = {
            "status": "running", "duration_seconds": 0,
            "log_path": str(path), "first_error": None,
        }
        started = time.monotonic()
        try:
            with _capture(path):
                try:
                    yield
                except BaseException as exc:
                    record["status"] = "failed"
                    record["first_error"] = next((line.strip() for line in str(exc).splitlines() if line.strip()), type(exc).__name__)
                    for sink in output_logs.get():
                        sink.write(traceback.format_exc())
                    raise
                else:
                    record["status"] = "succeeded"
        finally:
            record["duration_seconds"] = time.monotonic() - started
            if record["status"] == "failed":
                with path.open(encoding="utf-8", errors="replace") as log:
                    for line in log:
                        if re.search(r"\berror\b|\bfatal\b", line, re.IGNORECASE):
                            record["first_error"] = line.strip()
                            break
            self.save()

    def set_build(self, build_plan_id, build_dir):
        if self.lane is not None:
            self.summary["lanes"][self.lane].update(build_plan_id=build_plan_id, build_dir=str(build_dir))
            self.save()

    def failure_paths(self, lane_id):
        lane = self.summary["lanes"].get(lane_id, {})
        return [stage["log_path"] for stage in lane.get("stages", {}).values() if stage["status"] == "failed"]


@contextmanager
def stage(name):
    logs = current.get()
    if logs is None:
        yield
    else:
        with logs.stage(name):
            yield