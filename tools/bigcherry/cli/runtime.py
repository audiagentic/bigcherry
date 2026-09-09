"""CLI adapter for the reusable runtime-placement matrix (RHA06).

The adapter deliberately contains no campaign policy.  A JSON document selects
registered models, host devices and an existing worker command for each cell;
``campaign.runtime_matrix`` owns validation, ordering and progress files.
Workers are invoked without a shell and receive the resolved cell descriptor
through ``BIGCHERRY_RUNTIME_CELL_JSON``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..campaign.runtime_matrix import MatrixResolutionError, resolve_matrix, run_matrix
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
    command = cell.workload.get("delegate_argv")
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
        cells = resolve_matrix(document["cells"], host=host, known_models=known_models)
        resolved = {
            "schema_version": 1,
            "config": str(config_path),
            "host": host.name,
            "model_registry": str(registry),
            "cells": [cell.document() for cell in cells],
        }
        _atomic_json(output / "resolved-matrix.json", resolved)
        if args.dry_run:
            print(json.dumps(resolved, indent=2, sort_keys=True))
            return 0
        base_env = os.environ.copy()
        result = run_matrix(
            cells,
            output=output,
            execute=lambda cell: _worker(cell, root=output, base_env=base_env),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("state") == "completed" else 1
    except (MatrixResolutionError, OSError, ValueError) as exc:
        print(f"runtime-matrix: invalid: {exc}", file=sys.stderr)
        return 2
