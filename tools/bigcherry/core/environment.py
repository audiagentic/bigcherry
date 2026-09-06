"""Host environment: the machine-specific facts, loaded once, by role.

Companion to ``config/environment.toml``. The shell side is
``tools/env/bigcherry-env.sh``; this is the same data for Python callers, so
tooling can resolve a model root or a bench port without hardcoding one
person's paths -- which is exactly what the reference docs used to do, in 40
files.

Deliberately NOT part of ``core.config``'s ``Config``. That models the
repository: patch sets, sources, builds, platforms, campaigns, and
``[[trees]]`` (repo CHECKOUTS -- name, path, required, role, expected tooling
revision). None of it describes a HOST: no address, no model root, no
toolchain, no device inventory. The two are orthogonal, one host can carry
several trees, and a tree's path is meaningless without the host it lives on.
Merging them would make every consumer of one depend on the other.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class EnvironmentError_(ValueError):
    """Raised for a malformed or missing environment document."""


@dataclass(frozen=True)
class Device:
    """One GPU, as the runtime addresses it.

    ``index`` is the HIP/ROCR visible-devices ordinal -- the value
    ``--devices`` and ``*_VISIBLE_DEVICES`` take -- not a slot number or a
    PCI id. ``vram_mib`` matters because the SMALLEST participating card
    constrains any cross-architecture comparison: changing model or KV
    quantisation per card changes the work being measured.
    """

    index: int
    arch: str
    model: str
    vram_mib: int


@dataclass(frozen=True)
class Host:
    name: str
    description: str
    hostname: str
    address: str
    home: str
    repo: str
    cache_root: str
    share: str
    model_root: str
    bench_harness: str
    rocm: str
    rocm_shim: str
    production_port: int
    bench_port: int
    devices: tuple[Device, ...]

    def devices_for_arch(self, arch: str) -> tuple[Device, ...]:
        return tuple(d for d in self.devices if d.arch == arch)

    def visible_devices(self, *indices: int) -> str:
        """The value to export as HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES.

        Both must be set for a multi-GPU campaign: the campaign inherits
        ambient visibility rather than restricting it, so that ``-sm tensor``
        topology is preserved, and exposing all four heterogeneous cards makes
        the server fail its AllReduce init and segfault.
        """
        known = {d.index for d in self.devices}
        unknown = [i for i in indices if i not in known]
        if unknown:
            raise EnvironmentError_(
                f"host {self.name!r} has no device(s) {unknown}; known: {sorted(known)}"
            )
        return ",".join(str(i) for i in indices)


@dataclass(frozen=True)
class Environment:
    default_host: str
    hosts: dict[str, Host]
    path: Path

    def host(self, name: str | None = None) -> Host:
        key = name or self.default_host
        found = self.hosts.get(key)
        if found is None:
            raise EnvironmentError_(
                f"{self.path}: unknown host {key!r}; known: {sorted(self.hosts)}"
            )
        return found


def _require(table: dict, key: str, where: str) -> object:
    if key not in table:
        raise EnvironmentError_(f"{where}: missing required key {key!r}")
    return table[key]


def load(path: str | Path) -> Environment:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnvironmentError_(f"{path}: not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise EnvironmentError_(f"{path}: {exc}") from exc

    hosts: dict[str, Host] = {}
    for name, body in (raw.get("host") or {}).items():
        where = f"{path}: host.{name}"
        if not isinstance(body, dict):
            raise EnvironmentError_(f"{where} must be a table")
        devices = []
        for entry in body.get("devices") or ():
            dwhere = f"{where}.devices"
            devices.append(Device(
                index=int(_require(entry, "index", dwhere)),
                arch=str(_require(entry, "arch", dwhere)),
                model=str(entry.get("model", "")),
                vram_mib=int(entry.get("vram-mib", 0)),
            ))
        # Device ordinals are how every caller addresses a GPU, so a duplicate
        # would silently make one of them unreachable.
        seen = [d.index for d in devices]
        if len(seen) != len(set(seen)):
            raise EnvironmentError_(f"{where}: duplicate device index in {seen}")
        hosts[name] = Host(
            name=name,
            description=str(body.get("description", "")).strip(),
            hostname=str(_require(body, "hostname", where)),
            address=str(body.get("address", "")),
            home=str(body.get("home", "")),
            repo=str(body.get("repo", "")),
            cache_root=str(body.get("cache-root", "")),
            share=str(body.get("share", "")),
            model_root=str(body.get("model-root", "")),
            bench_harness=str(body.get("bench-harness", "")),
            rocm=str(body.get("rocm", "")),
            rocm_shim=str(body.get("rocm-shim", "")),
            production_port=int(body.get("production-port", 0)),
            bench_port=int(body.get("bench-port", 0)),
            devices=tuple(sorted(devices, key=lambda d: d.index)),
        )

    if not hosts:
        raise EnvironmentError_(f"{path}: no [host.*] entries")
    default_host = str(raw.get("default-host") or next(iter(hosts)))
    if default_host not in hosts:
        raise EnvironmentError_(
            f"{path}: default-host {default_host!r} is not defined; known: {sorted(hosts)}"
        )
    return Environment(default_host=default_host, hosts=hosts, path=path)


def _repo_root() -> Path:
    """Find the repository by a structural marker, not a parent count.

    tooling-hygiene TR14.FIXED_PARENT_DEPTH: a hardcoded parents[N] silently
    resolves to the wrong directory the moment a module moves, and the failure
    is a confusing missing-file error rather than an obvious one.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "recipes.toml").is_file():
            return candidate
    raise EnvironmentError_(
        f"no repository root above {here}: expected a config/recipes.toml marker"
    )


def default_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root else _repo_root()
    return root / "config" / "environment.toml"


def load_default(repo_root: str | Path | None = None) -> Environment:
    return load(default_path(repo_root))
