"""Real per-cell smoke worker for run_bump_validation.py's runtime-matrix
delegate_argv: launch the resolved binary against the resolved model and
runtime-profile server-args, wait for /health, send one real completion,
shut down cleanly. Exit 0 only if all of that succeeds.

Invoked by `bigcherry runtime-matrix` as the cell's delegate_argv target;
reads the immutable cell descriptor from BIGCHERRY_RUNTIME_CELL_JSON
(HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES are already set in the inherited
environment -- this script only resolves model path + runtime-profile
server-args and drives ServerRunner).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BC_TOOLS = _REPO_ROOT / "tools"
if str(_BC_TOOLS) not in sys.path:
    sys.path.insert(0, str(_BC_TOOLS))

import tomllib

from bigcherry.tuning.server_runner import ServerError, ServerRunner


def _load_model(model_id: str) -> tuple[Path, bool]:
    with open(_REPO_ROOT / "config" / "models.toml", "rb") as f:
        registry = tomllib.load(f)
    with open(_REPO_ROOT / "config" / "environment.toml", "rb") as f:
        env_cfg = tomllib.load(f)
    model_root = Path(env_cfg["host"][env_cfg["default-host"]]["model-root"])
    for entry in registry["models"]:
        if entry["id"] == model_id:
            return model_root / entry["path"], bool(entry.get("mtp", False))
    raise SystemExit(f"smoke_worker: unknown model_id {model_id!r}")


def _load_server_args(runtime_profile: str) -> list[str]:
    with open(_REPO_ROOT / "config" / "recipes.toml", "rb") as f:
        recipes = tomllib.load(f)
    profile = recipes["runtime-profile"][runtime_profile]
    return list(profile["server-args"])


def main() -> int:
    cell = json.loads(os.environ["BIGCHERRY_RUNTIME_CELL_JSON"])
    model_path, is_mtp = _load_model(cell["model_id"])
    server_args = _load_server_args(cell["runtime_profile"])
    # -ngl 99 (full offload) and a small real context -- this is a launch/
    # complete smoke gate, not a performance benchmark, so keep it light.
    extra_args = ["-ngl", "99", "-c", "4096"]

    print(f"smoke_worker: cell={cell['cell_id']} model={model_path} "
          f"profile={cell['runtime_profile']} devices={cell['devices']}", file=sys.stderr)

    if not model_path.is_file():
        print(f"smoke_worker: model file not found: {model_path}", file=sys.stderr)
        return 1

    runner = ServerRunner(
        binary=Path(cell["binary"]), model=model_path,
        extra_args=tuple(server_args + extra_args),
        shutdown_method="http",
    )
    try:
        with runner:
            runner.wait_healthy(timeout_s=300)
            result = runner.run_completion("The capital of France is", n_predict=16, timeout_s=120)
            content = result.get("content", "")
            if not content.strip():
                print("smoke_worker: completion returned empty content", file=sys.stderr)
                return 1
            print(f"smoke_worker: completion ok, {len(content)} chars generated", file=sys.stderr)
    except ServerError as exc:
        print(f"smoke_worker: server error: {exc}", file=sys.stderr)
        return 1

    if runner.last_shutdown is not None and not runner.last_shutdown.clean:
        print(f"smoke_worker: unclean shutdown: {runner.last_shutdown}", file=sys.stderr)
        return 1

    print(f"smoke_worker: cell={cell['cell_id']} PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
