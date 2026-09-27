"""Executor registry/configuration loader shared by CLI and future API adapters."""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .executor import Executor
from .local import LocalExecutor
from .remote import RemoteExecutor, SshJsonTransport, SshWorkerConfig
from .slurm import SlurmExecutor, SlurmPolicy


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutorInfo:
    executor_id: str
    kind: str
    host_id: str
    platform_family: str
    default: bool = False


class ExecutorRegistry:
    def __init__(self, executors: Mapping[str, Executor], info: Mapping[str, ExecutorInfo]) -> None:
        self.executors = dict(executors)
        self.info = dict(info)
        if set(self.executors) != set(self.info):
            raise ValueError("executor/info registry keys differ")

    def default_id(self) -> str:
        defaults = [key for key, value in self.info.items() if value.default]
        if len(defaults) != 1:
            raise RegistryError("exactly one executor must be default")
        return defaults[0]

    def describe(self) -> tuple[dict[str, object], ...]:
        return tuple(dataclasses.asdict(self.info[key]) for key in sorted(self.info))


def load_executor_registry(path: Path, *, state_root: Path) -> ExecutorRegistry:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RegistryError(f"cannot read executor registry {path}: {exc}") from exc
    raw = document.get("executor")
    if not isinstance(raw, dict) or not raw:
        raise RegistryError("executor registry requires [executor.<id>] entries")
    executors: dict[str, Executor] = {}
    infos: dict[str, ExecutorInfo] = {}
    for executor_id in sorted(raw):
        value = raw[executor_id]
        if not isinstance(value, dict):
            raise RegistryError(f"executor {executor_id} must be a table")
        kind = str(value.get("kind", ""))
        host_id = str(value.get("host_id", executor_id))
        platform = str(value.get("platform_family", "linux-rocm" if kind == "slurm" else "unknown"))
        if kind == "slurm":
            policy = SlurmPolicy(
                account=None if value.get("account") in (None, "") else str(value["account"]),
                build_partition=str(value.get("build_partition", "bc-build")),
                measure_partition=str(value.get("measure_partition", "bc-measure")),
            )
            executor: Executor = SlurmExecutor(policy=policy)
        elif kind == "local":
            executor = LocalExecutor(state_root / "executors" / executor_id)
        elif kind == "remote":
            ssh_host = value.get("ssh_host")
            if not isinstance(ssh_host, str) or not ssh_host:
                raise RegistryError(f"remote executor {executor_id} requires ssh_host")
            transport = SshJsonTransport(SshWorkerConfig(
                host=ssh_host,
                worker_command=str(value.get("worker_command", "python -m bigcherry.jobs.worker")),
                ssh_binary=str(value.get("ssh_binary", "ssh")),
            ))
            executor = RemoteExecutor(executor_id, transport)
        else:
            raise RegistryError(f"unknown executor kind {kind!r} for {executor_id}")
        executors[executor_id] = executor
        infos[executor_id] = ExecutorInfo(executor_id, kind, host_id, platform, bool(value.get("default", False)))
    registry = ExecutorRegistry(executors, infos)
    registry.default_id()
    return registry
