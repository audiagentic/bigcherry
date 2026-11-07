"""The few facts a campaign arm needs from a radiance serving manifest v1 (RSC11 A3).

The engine and the helpers read and verify the whole manifest themselves (the shared strict reader in
engines/radiance/components/serving/v1). The harness only needs what it must know to run the processes around them: the port
the engine will listen on (to health-check and drive it), the helper cards (to say which card each helper sees) and the shared
memory name (to clear a stale segment before the first helper starts). Anything else missing or malformed is the engine's to
refuse, not the harness's to second-guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ArmManifest:
    port: int
    helper_gpus: tuple[str, ...]
    shm: str


def read(path: Path) -> ArmManifest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read the serving manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"the serving manifest {path} is not JSON: {exc}") from exc
    try:
        spec = document["spec"]
        processes = spec["processes"]
        port = processes["engine"]["server"]["port"]
        helpers = tuple(h["gpu"] for h in processes["helpers"])
        shm = spec.get("handoff", {}).get("transport", {}).get("shm")
    except (KeyError, TypeError) as exc:
        raise ManifestError(f"the serving manifest {path} lacks {exc}") from exc
    if not isinstance(port, int) or isinstance(port, bool):
        raise ManifestError(f"the serving manifest {path}: spec.processes.engine.server.port is not an integer")
    if helpers and not shm:
        raise ManifestError(f"the serving manifest {path} has helpers but no spec.handoff.transport.shm")
    return ArmManifest(port=port, helper_gpus=helpers, shm=shm or "")
