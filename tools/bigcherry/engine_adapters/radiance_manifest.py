"""The few facts a campaign arm needs from a radiance serving manifest v1 (RSC11 A3).

The engine and the helpers read and verify the whole manifest themselves (the shared strict reader in
engines/radiance/components/serving/v1). The harness only needs what it must know to run the processes around them: the port
the engine will listen on (to health-check and drive it), the helper cards (to say which card each helper sees) and the shared
memory name (to clear a stale segment before the first helper starts). Anything else missing or malformed is the engine's to
refuse, not the harness's to second-guess.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .base import QualityLaunch

# Stable Linux UAPI values from linux/fcntl.h. Some Python builds expose
# memfd_create without exposing the corresponding fcntl constants.
_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_IMMUTABLE_SEALS = (
    getattr(fcntl, "F_SEAL_SEAL", 0x0001) | getattr(fcntl, "F_SEAL_SHRINK", 0x0002) |
    getattr(fcntl, "F_SEAL_GROW", 0x0004) | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
)

class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ArmManifest:
    port: int
    helper_gpus: tuple[str, ...]
    shm: str


@contextmanager
def quality_launch(argv: tuple[str, ...]) -> Iterator[QualityLaunch]:
    """Retain the exact native manifest and its working directory through exec."""
    manifest_arg = argv.index("--manifest") + 1
    path = Path(argv[manifest_arg])
    limit = 16 * 1024 * 1024
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC), "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ManifestError(f"{path}: quality manifest must be a bounded regular file")
        content = source.read(limit + 1)
    if len(content) > limit:
        raise ManifestError(f"{path}: quality manifest exceeds the size limit")
    try:
        document = json.loads(content)
    except (UnicodeError, ValueError) as exc:
        raise ManifestError(f"{path}: invalid quality manifest: {exc}") from exc
    if not isinstance(document, dict) or document.get("apiVersion") != "bigcherry/v3":
        yield QualityLaunch(argv)
        return
    try:
        root = Path(document["spec"]["native_config"]["resolution_root"])
    except (KeyError, TypeError) as exc:
        raise ManifestError(f"{path}: v3 quality manifest lacks an execution root") from exc
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ManifestError(f"{path}: v3 quality execution root must be an existing canonical directory")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        cwd = Path(f"/proc/self/fd/{root_fd}")
        if not os.path.samestat(os.fstat(root_fd), root.stat()):
            raise ManifestError(f"{path}: v3 quality execution root changed while opening it")
        manifest_fd = os.memfd_create("bc-quality-manifest", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            with os.fdopen(os.dup(manifest_fd), "wb") as sealed:
                sealed.write(content)
            fcntl.fcntl(manifest_fd, _ADD_SEALS, _IMMUTABLE_SEALS)
            args = list(argv)
            args[0] = str(Path(args[0]).absolute())
            args[manifest_arg] = f"/proc/self/fd/{manifest_fd}"
            for flag in ("--kld-record", "--kld-corpus", "--kld-ref", "--kld-out"):
                if flag in args:
                    i = args.index(flag) + 1
                    args[i] = str(Path(args[i]).absolute())
            # subprocess applies cwd before exec closes descriptors absent from pass_fds.
            yield QualityLaunch(tuple(args), cwd, (manifest_fd,), hashlib.sha256(content).hexdigest())
        finally:
            os.close(manifest_fd)
    finally:
        os.close(root_fd)


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
