"""RAM-only engine smoke (MEN15): a tiny model on the CPU with no GPU visible, the same for every engine.

The engine's adapter says what to run (``EngineAdapter.cpu_smoke``); this module runs it. It checks the model's
sha256, starts the server through the shared ``ServerRunner`` (so the port, health wait, shutdown and sanitised
environment are the ones every other run uses), waits for health, asks for three short greedy completions twice,
requires the text to be non-empty and identical across the two passes, reads the server's peak resident memory,
shuts the server down gracefully and checks that no GPU was opened. The result is one JSON document; any failure
makes the run fail with every reason listed.

Nothing here names an engine: what differs (arguments, completion route, device-init log lines) is in the spec.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import environment
from ..core.context import ProjectContext
from ..engine_adapters import adapter
from ..engine_adapters.base import GPU_HIDING_ENV, CpuSmokeSpec
from ..tuning.server_runner import ServerError, ServerRunner

MODEL_ROOT_ENV = "BIGCHERRY_MODEL_ROOT"
RESULT_NAME = "cpu-smoke.json"
SCHEMA = 1


class CpuSmokeError(RuntimeError):
    """The smoke could not run or did not pass; ``failures`` lists every reason."""

    def __init__(self, failures: list[str]):
        super().__init__("; ".join(failures))
        self.failures = failures


@dataclass(frozen=True)
class ModelCheck:
    path: str
    expected_sha256: str
    actual_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(spec: CpuSmokeSpec) -> ModelCheck:
    """The model must exist and match the pinned digest. A placeholder digest is refused: nothing is pinned yet."""
    expected = spec.model_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise CpuSmokeError([
            f"{spec.engine}: [smoke.cpu] sha256 is {spec.model_sha256!r}, not a 64-digit hex digest; pin the tiny model "
            f"in engines/{spec.engine}/engine.toml first"
        ])
    if not spec.model.is_file():
        raise CpuSmokeError([f"{spec.engine}: smoke model not found at {spec.model}"])
    actual = sha256_file(spec.model)
    if actual != expected:
        raise CpuSmokeError([
            f"{spec.engine}: smoke model {spec.model} has sha256 {actual}, the pinned digest is {expected}"
        ])
    return ModelCheck(str(spec.model), expected, actual)


def peak_rss_bytes(pid: int | None) -> tuple[int | None, str]:
    """(peak resident set of a process in bytes, note). /proc is read on Linux; elsewhere the value is not measured."""
    if pid is None:
        return None, "no server process"
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return None, f"not measured: no /proc on {sys.platform}"
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024, "VmHWM"
    return None, "not measured: no VmHWM in /proc status"


def _dig(document: Any, path: tuple[str | int, ...]) -> Any:
    for key in path:
        document = document[key]
    return document


def _served_model_id(runner: ServerRunner, models_path: str, timeout_s: int) -> str:
    try:
        with urllib.request.urlopen(f"http://{runner.host}:{runner.port}{models_path}", timeout=timeout_s) as resp:
            return str(json.loads(resp.read().decode("utf-8"))["data"][0]["id"])
    except (urllib.error.URLError, OSError, TimeoutError, KeyError, IndexError, ValueError) as exc:
        raise ServerError(f"GET {models_path} failed: {exc}") from exc


def _complete(runner: ServerRunner, spec: CpuSmokeSpec, prompt: str, model_id: str) -> str:
    api = spec.completion
    body: dict[str, Any] = {api.prompt_field: prompt, api.token_field: spec.n_tokens, **dict(api.fixed)}
    if model_id:
        body["model"] = model_id
    reply = runner.post_json(api.path, body, timeout_s=spec.request_timeout_s)
    try:
        text = _dig(reply, api.text_path)
    except (KeyError, IndexError, TypeError) as exc:
        raise ServerError(f"{api.path} reply has no text at {list(api.text_path)}: {json.dumps(reply)[:200]}") from exc
    if not isinstance(text, str):
        raise ServerError(f"{api.path} text is not a string: {text!r}")
    return text


def gpu_log_hits(spec: CpuSmokeSpec, log_path: Path) -> list[str]:
    """Server-log lines that show a GPU being opened, by the adapter's patterns."""
    if not log_path.is_file():
        return []
    patterns = [re.compile(p) for p in spec.device.forbidden_log_patterns]
    return [
        line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(p.search(line) for p in patterns)
    ]


def _environment_failures(spec: CpuSmokeSpec) -> list[str]:
    return [
        f"{name} must be set to the empty string in the smoke environment (is {spec.env_overrides.get(name)!r})"
        for name in GPU_HIDING_ENV if spec.env_overrides.get(name) != ""
    ]


def run(spec: CpuSmokeSpec, out_dir: Path) -> dict[str, Any]:
    """Run the smoke, write ``cpu-smoke.json`` under ``out_dir`` and return it. Raises CpuSmokeError, after writing the
    document, when anything failed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    result: dict[str, Any] = {
        "schema": SCHEMA, "engine": spec.engine, "ok": False, "failures": [], "binary": str(spec.binary),
        "args": list(spec.args), "env_overrides": dict(spec.env_overrides), "log": str(log_path),
    }
    failures: list[str] = result["failures"]
    started = time.monotonic()
    try:
        failures.extend(_environment_failures(spec))
        if failures:
            return _finish(result, out_dir, started)
        result["model"] = vars(verify_model(spec))
        if not spec.binary.is_file():
            failures.append(f"server binary not found at {spec.binary}")
            return _finish(result, out_dir, started)
        _drive(spec, log_path, result, failures)
    except CpuSmokeError as exc:
        failures.extend(exc.failures)
    except (ServerError, OSError) as exc:
        failures.append(f"server: {exc}")
    return _finish(result, out_dir, started)


def _drive(spec: CpuSmokeSpec, log_path: Path, result: dict[str, Any], failures: list[str]) -> None:
    unset = tuple(sorted(
        name for name in os.environ if name.startswith(spec.env_unset_prefixes) and name not in spec.env_overrides
    )) if spec.env_unset_prefixes else ()
    runner = ServerRunner(
        serve=spec.serve, binary=spec.binary, model=spec.model, port=spec.port, extra_args=spec.args,
        env_overrides=spec.env_overrides, env_unset=unset, log_path=log_path, command_prefix=spec.command_prefix,
    )
    result["port"] = runner.port
    launched = time.monotonic()
    runner.launch()
    try:
        runner.wait_healthy(spec.health_timeout_s)
        result["startup_s"] = round(time.monotonic() - launched, 3)
        model_id = _served_model_id(runner, spec.completion.models_path, 60) if spec.completion.models_path else ""
        passes: list[list[str]] = []
        began = time.monotonic()
        for _ in range(2):
            passes.append([_complete(runner, spec, prompt, model_id) for prompt in spec.prompts])
        result["completion_s"] = round(time.monotonic() - began, 3)
        result["completions"] = [
            {"prompt": prompt, "first": passes[0][i], "second": passes[1][i]} for i, prompt in enumerate(spec.prompts)
        ]
        for i, prompt in enumerate(spec.prompts):
            if not passes[0][i].strip():
                failures.append(f"empty completion for prompt {prompt!r}")
            if passes[0][i] != passes[1][i]:
                failures.append(f"greedy completion differs between the two passes for prompt {prompt!r}")
        rss, note = peak_rss_bytes(runner.pid)
        result["peak_rss_bytes"], result["peak_rss_note"] = rss, note
    finally:
        shutdown = runner.shutdown()
    if shutdown is not None:
        result["shutdown"] = {
            "method": shutdown.method, "requested": shutdown.requested, "forced": shutdown.forced,
            "returncode": shutdown.returncode, "clean": shutdown.clean,
        }
        if not shutdown.clean:
            failures.append(f"server did not shut down cleanly: {result['shutdown']}")
    hits = gpu_log_hits(spec, log_path)
    result["gpu_log_hits"] = hits
    if hits:
        failures.append(f"the server log shows a GPU being opened: {hits[0]!r}")


def _finish(result: dict[str, Any], out_dir: Path, started: float) -> dict[str, Any]:
    result["wall_s"] = round(time.monotonic() - started, 3)
    result["ok"] = not result["failures"]
    (out_dir / RESULT_NAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["ok"]:
        raise CpuSmokeError(list(result["failures"]))
    return result


def resolve_model_root(explicit: str | None) -> Path:
    """--model-root, else BIGCHERRY_MODEL_ROOT, else the default host's model-root."""
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.environ.get(MODEL_ROOT_ENV)
    if from_env:
        return Path(from_env).expanduser()
    try:
        declared = environment.load_default().host(None).model_root
    except environment.EnvironmentError_ as exc:
        raise CpuSmokeError([f"no model root: {exc}"]) from exc
    if not declared:
        raise CpuSmokeError([f"no model root: pass --model-root, set {MODEL_ROOT_ENV}, or declare the host's model-root"])
    return Path(declared).expanduser()


def run_for_engine(engine: str, build: Path, model_root: str | None, out: Path | None) -> dict[str, Any]:
    """Build the engine's spec and run it. Raises CpuSmokeError on any failure (including 'no smoke declared')."""
    context = ProjectContext.resolve(engine=engine)
    spec = adapter(engine).cpu_smoke(context, build, resolve_model_root(model_root))
    if spec is None:
        raise CpuSmokeError([f"{engine} declares no [smoke.cpu] in its engine.toml"])
    return run(spec, out if out is not None else context.work_root / "cpu-smoke" / engine)
