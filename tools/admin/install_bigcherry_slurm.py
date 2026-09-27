#!/usr/bin/env python3
"""Install/qualify BigCherry's minimal Slurm v1 host stack.

Dry-run is the default.  ``--apply`` is intentionally root-only and still
requires an already-reviewed accepted BigCherry hardware inventory.  GPU
hardware acceptance remains a separate post-install gate; this installer never
silently enables ``ConstrainDevices=yes``.
"""
from __future__ import annotations

import argparse
import json
import os
import pwd
import grp
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

from admin.render_bigcherry_slurm import install_rendered, render_files  # noqa: E402
from bigcherry.hardware.inventory import InventoryCatalog  # noqa: E402


class InstallError(RuntimeError):
    pass


@dataclass(frozen=True)
class Action:
    kind: str
    argv: tuple[str, ...]
    purpose: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "argv": list(self.argv), "purpose": self.purpose}


def action_plan(*, node_name: str) -> tuple[Action, ...]:
    return (
        Action("command", ("apt-get", "update"), "refresh package metadata"),
        Action(
            "command",
            ("apt-get", "install", "-y", "slurm-wlm", "munge", "jq"),
            "install minimal no-db Slurm stack",
        ),
        Action(
            "command",
            ("systemctl", "enable", "--now", "munge"),
            "start MUNGE authentication",
        ),
        Action("command", ("slurmd", "-G"), "validate generated GRES/config"),
        Action(
            "command",
            ("systemctl", "enable", "slurmctld", "slurmd"),
            "enable scheduler services",
        ),
        Action(
            "command",
            ("systemctl", "restart", "slurmctld"),
            "start/reload controller",
        ),
        Action(
            "command",
            ("systemctl", "restart", "slurmd"),
            "start/reload compute daemon",
        ),
        Action("command", ("scontrol", "ping"), "verify controller health"),
        Action("command", ("sinfo", "-Nel"), "verify node/partition registration"),
        Action(
            "command",
            ("scontrol", "show", "node", node_name),
            "verify rendered node identity",
        ),
    )


def _run(argv: tuple[str, ...], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        argv,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise InstallError(
            f"command failed ({completed.returncode}): {' '.join(argv)}: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return completed


def _mkdir(path: Path, *, mode: int, user: str, group: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, mode)
    shutil.chown(path, user=user, group=group)


def _ensure_munge_key(path: Path = Path("/etc/munge/munge.key")) -> bool:
    if path.is_file() and path.stat().st_size > 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(fd, os.urandom(1024))
        os.fsync(fd)
    finally:
        os.close(fd)
    shutil.chown(path, user="munge", group="munge")
    return True


def _munge_roundtrip() -> None:
    encoded = _run(("munge", "-n")).stdout
    _run(("unmunge",), input_bytes=encoded)


def _validate_preconditions(
    *,
    project_root: Path,
    hardware_root: Path,
    executor_id: str,
    node_name: str,
    cpus: int,
    real_memory_mib: int,
) -> tuple[dict[str, str], str]:
    inventory = InventoryCatalog(hardware_root).load(executor_id)
    if inventory.host_id != node_name:
        raise InstallError(
            f"accepted inventory host_id {inventory.host_id!r} != node {node_name!r}"
        )
    files = render_files(
        project_root=project_root,
        hardware_root=hardware_root,
        executor_id=executor_id,
        node_name=node_name,
        cpus=cpus,
        real_memory_mib=real_memory_mib,
    )
    active = [
        line.strip()
        for line in files["slurm.conf"].splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any(line.startswith("RequeueExit") for line in active):
        raise InstallError("v1 monolithic config must not enable RequeueExit")
    if "ConstrainDevices=no" not in files["cgroup.conf"]:
        raise InstallError(
            "initial Brutus install must use qualified ConstrainDevices=no baseline"
        )
    return files, inventory.material_hash


def apply_install(
    *,
    files: dict[str, str],
    node_name: str,
    command_runner: Callable[[tuple[str, ...]], subprocess.CompletedProcess] | None = None,
) -> dict[str, object]:
    if os.name != "posix" or os.geteuid() != 0:
        raise InstallError("--apply requires root on a POSIX/Linux host")

    runner = command_runner
    def command(argv: tuple[str, ...]) -> None:
        if runner is None:
            _run(argv)
        else:
            completed = runner(argv)
            if completed.returncode != 0:
                raise InstallError(f"mock/runner command failed: {' '.join(argv)}")

    # Package install creates the slurm/munge accounts.  Everything after this
    # point uses explicit ownership/modes and exact generated config.
    command(("apt-get", "update"))
    command(("apt-get", "install", "-y", "slurm-wlm", "munge", "jq"))
    for account in ("slurm", "munge"):
        try:
            pwd.getpwnam(account)
        except KeyError as exc:
            raise InstallError(f"required system account missing after install: {account}") from exc
    for group_name in ("slurm", "munge"):
        try:
            grp.getgrnam(group_name)
        except KeyError as exc:
            raise InstallError(f"required system group missing after install: {group_name}") from exc

    _mkdir(Path("/run/munge"), mode=0o755, user="munge", group="munge")
    _mkdir(Path("/var/log/munge"), mode=0o755, user="munge", group="munge")
    _mkdir(Path("/var/spool/slurmctld"), mode=0o755, user="slurm", group="slurm")
    _mkdir(Path("/var/spool/slurmd"), mode=0o755, user="root", group="root")
    _mkdir(Path("/var/log/slurm"), mode=0o755, user="slurm", group="slurm")
    _mkdir(Path("/etc/slurm"), mode=0o755, user="root", group="root")
    key_created = _ensure_munge_key()

    command(("systemctl", "enable", "--now", "munge"))
    _munge_roundtrip()

    install_rendered(files, Path("/etc/slurm"))
    jobcomp = Path("/var/log/slurm/bigcherry-jobcomp.log")
    jobcomp.touch(exist_ok=True)
    shutil.chown(jobcomp, user="slurm", group="slurm")

    command(("slurmd", "-G"))
    command(("systemctl", "enable", "slurmctld", "slurmd"))
    command(("systemctl", "restart", "slurmctld"))
    command(("systemctl", "restart", "slurmd"))
    command(("scontrol", "ping"))
    command(("sinfo", "-Nel"))
    command(("scontrol", "show", "node", node_name))
    return {"applied": True, "munge_key_created": key_created}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="install_bigcherry_slurm")
    parser.add_argument("--project-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--hardware-root", type=Path, required=True)
    parser.add_argument("--executor-id", default="brutus")
    parser.add_argument("--node-name", default="brutus")
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--real-memory-mib", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        project_root = args.project_root.resolve()
        hardware_root = args.hardware_root.resolve()
        files, accepted_hash = _validate_preconditions(
            project_root=project_root,
            hardware_root=hardware_root,
            executor_id=args.executor_id,
            node_name=args.node_name,
            cpus=args.cpus,
            real_memory_mib=args.real_memory_mib,
        )
        result: dict[str, object] = {
            "schema": "bigcherry.slurm-install-plan.v1",
            "apply": args.apply,
            "node_name": args.node_name,
            "executor_id": args.executor_id,
            "accepted_inventory_hash": accepted_hash,
            "actions": [action.to_dict() for action in action_plan(node_name=args.node_name)],
            "rendered_files": sorted(files),
            "policy": {
                "accounting_db": False,
                "native_monolithic_requeue": False,
                "constrain_devices": False,
            },
        }
        if args.apply:
            result.update(apply_install(files=files, node_name=args.node_name))
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
