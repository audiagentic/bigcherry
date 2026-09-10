"""CLI adapter for the reusable runtime-placement matrix (RHA06).

The adapter deliberately contains no campaign policy.  A JSON document selects
registered models, host devices and an existing worker command for each cell;
``campaign.runtime_matrix`` owns validation, ordering and progress files.
Workers are invoked without a shell and receive the resolved cell descriptor
through ``BIGCHERRY_RUNTIME_CELL_JSON``.
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..campaign.runtime_matrix import MatrixResolutionError, resolve_matrix, run_matrix
from ..campaign.run_advisories import evaluate_runtime_result, render as render_run_advisories
from ..core import environment


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _models_from_registry(path: Path) -> frozenset[str]:
    import tomllib

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MatrixResolutionError(f"model registry {path}: {exc}") from exc
    models = raw.get("models")
    if not isinstance(models, list):
        raise MatrixResolutionError(f"model registry {path}: models must be an array")
    ids = frozenset(
        entry.get("id")
        for entry in models
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
    )
    if not ids:
        raise MatrixResolutionError(f"model registry {path}: no model ids")
    return ids


def _runtime_profiles_from_recipes(path: Path) -> frozenset[str]:
    import tomllib

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MatrixResolutionError(f"recipe registry {path}: {exc}") from exc
    profiles = raw.get("runtime-profile")
    if not isinstance(profiles, dict) or not profiles:
        raise MatrixResolutionError(f"recipe registry {path}: no runtime profiles")
    return frozenset(name for name in profiles if isinstance(name, str) and name)


def _load_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixResolutionError(f"matrix config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixResolutionError("matrix config must be a JSON object")
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise MatrixResolutionError("matrix config: cells must be an array")
    return value


def _worker(cell, *, root: Path, base_env: Mapping[str, str]):
    server_capture = cell.workload.get("server_capture")
    if server_capture is not None:
        if not isinstance(server_capture, Mapping):
            raise MatrixResolutionError(f"cell {cell.cell_id!r}: server_capture must be an object")
        required = ("model", "bench_configs", "runner_root", "required_metrics")
        if any(key not in server_capture for key in required):
            raise MatrixResolutionError(
                f"cell {cell.cell_id!r}: server_capture requires {', '.join(required)}"
            )
        model = server_capture["model"]
        bench_configs = server_capture["bench_configs"]
        runner_root = server_capture["runner_root"]
        metrics = server_capture["required_metrics"]
        extra_args = server_capture.get("extra_args", [])
        if not all(isinstance(value, str) and value for value in (model, bench_configs, runner_root)):
            raise MatrixResolutionError(f"cell {cell.cell_id!r}: server_capture path/config values must be strings")
        if not isinstance(metrics, list) or not metrics or not all(isinstance(value, str) and value for value in metrics):
            raise MatrixResolutionError(f"cell {cell.cell_id!r}: server_capture required_metrics must be a non-empty string array")
        if not isinstance(extra_args, list) or not all(isinstance(value, str) and value for value in extra_args):
            raise MatrixResolutionError(f"cell {cell.cell_id!r}: server_capture extra_args must be a string array")
        expected = server_capture.get("expected_execution")
        if expected is not None and not isinstance(expected, Mapping):
            raise MatrixResolutionError(f"cell {cell.cell_id!r}: server_capture expected_execution must be an object")
        env_overrides = server_capture.get("environment", {})
        if not isinstance(env_overrides, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in env_overrides.items()
        ):
            raise MatrixResolutionError(f"cell {cell.cell_id!r}: server_capture environment must be a string map")
        from ..campaign.benchmark import run_server_arm_capture
        result = run_server_arm_capture(
            binary=Path(cell.binary), model=Path(model), extra_args=tuple(extra_args),
            output=root, pair=0, side=cell.cell_id, position=0,
            env={**dict(cell.visibility), **dict(env_overrides)}, bench_configs=bench_configs,
            runner_root=Path(runner_root), required_metrics=tuple(metrics),
            repetitions=int(server_capture.get("repetitions", 1)),
            shutdown_method=str(server_capture.get("shutdown_method", "http")),
            expected_execution=dict(expected) if expected is not None else None,
            execution_evidence=str(server_capture.get("execution_evidence", "required")),
        )
        return {"returncode": int(result.get("returncode", 1)), **result}
    command = cell.workload.get("delegate_argv")
    server_bench = cell.workload.get("server_bench")
    if server_bench is not None:
        if not isinstance(server_bench, Mapping):
            raise MatrixResolutionError(f"cell {cell.cell_id!r}: server_bench must be an object")
        from ..campaign.bench_runner import run_bench_runner_server_bench

        required = ("server_url", "bench_configs")
        if any(not isinstance(server_bench.get(key), str) or not server_bench[key].strip() for key in required):
            raise MatrixResolutionError(
                f"cell {cell.cell_id!r}: server_bench requires server_url and bench_configs"
            )
        metrics = run_bench_runner_server_bench(
            server_url=server_bench["server_url"],
            bench_configs=server_bench["bench_configs"],
            repetitions=int(server_bench.get("repetitions", 1)),
            timeout_s=int(server_bench.get("timeout_s", 300)),
            runner_root=Path(server_bench["runner_root"]).resolve()
            if server_bench.get("runner_root") else None,
            model_label=str(server_bench.get("model_label", cell.model_id)),
            env_overrides=dict(cell.visibility),
        )
        return {"returncode": 0, "metrics": metrics, "performance_admitted": False}
    if not isinstance(command, list) or not command or not all(
        isinstance(arg, str) and arg for arg in command
    ):
        raise MatrixResolutionError(
            f"cell {cell.cell_id!r}: workload.delegate_argv must be a non-empty string array"
        )
    child_env = dict(base_env)
    child_env.update(dict(cell.visibility))
    child_env["BIGCHERRY_RUNTIME_CELL_JSON"] = json.dumps(
        cell.document(), sort_keys=True, separators=(",", ":")
    )
    child_env["BIGCHERRY_RUNTIME_CELL_ID"] = cell.cell_id
    cwd = cell.workload.get("delegate_cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd.strip()):
        raise MatrixResolutionError(f"cell {cell.cell_id!r}: delegate_cwd must be a path")
    try:
        completed = subprocess.run(
            command,
            cwd=cwd or None,
            env=child_env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"worker launch failed: {exc}") from exc
    result: dict[str, Any] = {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20_000:],
        "stderr": completed.stderr[-20_000:],
        "performance_admitted": False,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"worker exited with status {completed.returncode}")
    return result


def _quiescence_checker(document: Mapping[str, Any], *, base_env: Mapping[str, str]):
    command = document.get("quiescence_argv")
    if command is None:
        return None
    if not isinstance(command, list) or not command or not all(
        isinstance(arg, str) and arg for arg in command
    ):
        raise MatrixResolutionError(
            "matrix config: quiescence_argv must be a non-empty string array"
        )

    def check() -> bool:
        try:
            completed = subprocess.run(
                command, env=dict(base_env), capture_output=True, check=False,
            )
        except OSError:
            return False
        return completed.returncode == 0

    return check


def cmd_runtime_matrix(args) -> int:
    """Resolve and optionally execute a declarative runtime matrix."""
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    try:
        document = _load_document(config_path)
        env = (
            environment.load(Path(args.environment).resolve())
            if args.environment
            else environment.load_default()
        )
        host = env.host(document.get("host"))
        registry = Path(args.models).resolve() if args.models else Path("config/models.toml").resolve()
        known_models = _models_from_registry(registry)
        recipes = Path(args.recipes).resolve() if args.recipes else Path("config/recipes.toml").resolve()
        known_profiles = _runtime_profiles_from_recipes(recipes)
        for raw in document["cells"]:
            if raw.get("runtime_profile") not in known_profiles:
                raise MatrixResolutionError(
                    f"cell {raw.get('cell_id', '<unknown>')!r}: unknown runtime_profile "
                    f"{raw.get('runtime_profile')!r}"
                )
            topology = raw.get("topology")
            devices = raw.get("devices")
            if topology == "single" and (not isinstance(devices, list) or len(devices) != 1):
                raise MatrixResolutionError(f"cell {raw.get('cell_id', '<unknown>')!r}: single topology requires one device")
            if topology in {"dual", "dual-xtx"} and (not isinstance(devices, list) or len(devices) != 2):
                raise MatrixResolutionError(f"cell {raw.get('cell_id', '<unknown>')!r}: dual topology requires two devices")
        cells = resolve_matrix(document["cells"], host=host, known_models=known_models)
        source_digests = {
            "config": _sha256(config_path),
            "models": _sha256(registry),
            "recipes": _sha256(recipes),
        }
        binary_digests = {
            cell.cell_id: _sha256(Path(cell.binary))
            for cell in cells if Path(cell.binary).is_file()
        }
        resolved = {
            "schema_version": 1,
            "config": str(config_path),
            "host": host.name,
            "model_registry": str(registry),
            "recipe_registry": str(recipes),
            "cells": [cell.document() for cell in cells],
        }
        _atomic_json(output / "resolved-matrix.json", resolved)
        if args.dry_run:
            print(json.dumps(resolved, indent=2, sort_keys=True))
            return 0
        base_env = os.environ.copy()

        def revalidate(cell) -> bool:
            try:
                if any(_sha256(path) != digest for path, digest in (
                    (config_path, source_digests["config"]),
                    (registry, source_digests["models"]),
                    (recipes, source_digests["recipes"]),
                )):
                    return False
                expected = binary_digests.get(cell.cell_id)
                return expected is None or _sha256(Path(cell.binary)) == expected
            except OSError:
                return False

        result = run_matrix(
            cells,
            output=output,
            execute=lambda cell: _worker(cell, root=output, base_env=base_env),
            quiescent=_quiescence_checker(document, base_env=base_env),
            revalidate=revalidate,
        )
        advisory_evaluation = evaluate_runtime_result(result)
        _atomic_json(output / "advisories.json", advisory_evaluation.document())
        advisory_text = render_run_advisories(advisory_evaluation)
        if advisory_text:
            print(advisory_text, file=sys.stderr, end="")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("state") == "completed" else 1
    except (MatrixResolutionError, OSError, ValueError) as exc:
        print(f"runtime-matrix: invalid: {exc}", file=sys.stderr)
        return 2
