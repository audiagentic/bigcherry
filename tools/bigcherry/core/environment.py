"""Host environment: the machine-specific facts, loaded once, by role.

Host-specific values are never committed. The configured file is
``$BIGCHERRY_ENVIRONMENT`` or, by default, the gitignored
``config/environment.local.toml`` (template: ``config/environment.example.toml``);
scalar keys of the default host can be overridden with
``BC_<KEY>`` environment variables. The shell side is
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

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


class EnvironmentError_(ValueError):
    """Raised for a malformed or missing environment document."""


def gpu_visibility_pair(indices: tuple[int, ...]) -> dict[str, str]:
    """Pure two-selector math, no ``Host``/device-inventory validation.

    Shared core of ``Host.gpu_visibility_env`` for a caller that has already
    validated its indices against a real device inventory elsewhere (e.g.
    ``tuning/workflow.py``'s ``_devices_tuple`` parsing) and does not have a
    ``Host`` object at hand. See ``Host.gpu_visibility_env`` for the VA22
    two-selector-trap explanation this implements; do not duplicate that
    logic at a call site -- route through one of these two functions.
    """
    if not indices:
        raise EnvironmentError_("gpu_visibility_pair requires at least one device index")
    if len(set(indices)) != len(indices):
        raise EnvironmentError_(f"duplicate device index in {indices!r}")
    return {
        "ROCR_VISIBLE_DEVICES": ",".join(str(i) for i in indices),
        "HIP_VISIBLE_DEVICES": ",".join(str(pos) for pos in range(len(indices))),
    }


_PCI_LOCATOR_PATTERN = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")


@dataclass(frozen=True)
class Device:
    """One GPU, as the runtime addresses it.

    ``index`` is the HIP/ROCR visible-devices ordinal -- the value
    ``--devices`` and ``*_VISIBLE_DEVICES`` take -- not a slot number or a
    PCI id. ``vram_mib`` matters because the SMALLEST participating card
    constrains any cross-architecture comparison: changing model or KV
    quantisation per card changes the work being measured.

    ``locator`` (PRBE111, 2026-09-13) is the real PCI BDF (bus:device.function,
    canonical lowercase form e.g. "0000:03:00.0", verified via ``rocm-smi
    --showbus`` -- never invented) this physical device occupies. A
    llama-server log reports a device's marketing name and BDF but never its
    gfx ISA, so `bigcherry.experiment.attestation.parse_llama_server_
    attestation()` requires a real, host-configured BDF->arch mapping to
    positively confirm which physical card actually ran a measurement --
    this field is that mapping's source of truth. Optional (``None`` when
    not yet verified for a given index) because populating it with a guessed
    value would let attestation silently confirm the wrong physical device,
    which is worse than the honest fail-closed ARCH_MISMATCH it prevents by
    staying absent.
    """

    index: int
    arch: str
    model: str
    vram_mib: int
    locator: str | None = None


@dataclass(frozen=True)
class Host:
    name: str
    description: str
    hostname: str
    repo_root: str
    #: the single root of BigCherry's working storage on this host; builds/, runs/, worktrees/, toolchains/,
    #: remote-runs/ and the upstream mirror are fixed subfolders of it
    work_root: str
    model_root: str
    bench_harness: str
    #: the default ROCm install (used when a recipe names no SDK version)
    rocm_root: str
    #: named ROCm SDKs for this host, as (version, root) pairs; relative roots are repo-relative
    rocm_sdks: tuple[tuple[str, str], ...]
    production_port: int
    #: port for tests, benchmarks and ad-hoc servers; must never equal production_port
    test_port: int
    #: process names (exact ``comm``) whose presence means the host's GPUs are in use; ``remote run`` refuses to start
    #: while any is running.
    busy_processes: tuple[str, ...]
    #: the host-local environment file a remote run exports as ``BIGCHERRY_ENVIRONMENT`` (a path on the host)
    environment_file: str
    devices: tuple[Device, ...]

    def devices_for_arch(self, arch: str) -> tuple[Device, ...]:
        return tuple(d for d in self.devices if d.arch == arch)

    def gpu_visibility_env(self, *indices: int) -> dict[str, str]:
        """The exact ``ROCR_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` pair
        that makes exactly these physical device ordinals visible, in this
        order, and nothing else.

        VA22, "the two-selector trap" (has recurred more than once in this
        project's history): ``ROCR_VISIBLE_DEVICES`` filters the physical
        device list FIRST; ``HIP_VISIBLE_DEVICES`` then indexes INTO that
        already-filtered list by POSITION, not by physical ordinal. So
        selecting physical devices ``(1, 3)`` needs
        ``ROCR_VISIBLE_DEVICES="1,3"`` (the physical filter) and
        ``HIP_VISIBLE_DEVICES="0,1"`` (position within that filtered list,
        same order) -- setting both to the same physical indices is wrong,
        and for a selection that does not start at physical index 0 (e.g.
        gfx1201 alone at index 2: ``ROCR_VISIBLE_DEVICES="2"``,
        ``HIP_VISIBLE_DEVICES="0"``, never ``"2"``) it silently selects the
        wrong device or nothing at all rather than failing loudly.

        Both variables must be set for a multi-GPU campaign: the campaign
        inherits ambient visibility rather than restricting it, so that
        ``-sm tensor`` topology is preserved, and exposing all four
        heterogeneous cards makes the server fail its AllReduce init and
        segfault.
        """
        known = {d.index for d in self.devices}
        unknown = [i for i in indices if i not in known]
        if unknown:
            raise EnvironmentError_(
                f"host {self.name!r} has no device(s) {unknown}; known: {sorted(known)}"
            )
        return gpu_visibility_pair(indices)


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


_HOST_SCALAR_KEYS = (
    "hostname", "repo-root", "work-root", "model-root",
    "bench-harness", "rocm-root", "production-port", "test-port",
)


def _rocm_sdks(raw: object, where: str, repo_root: str) -> tuple[tuple[str, str], ...]:
    """The ``[host.<name>.rocm-sdks]`` table: version name -> SDK root."""
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise EnvironmentError_(f"{where}: rocm-sdks must be a table of version = root")
    out: list[tuple[str, str]] = []
    for version, root in sorted(raw.items()):
        if not isinstance(root, str) or not root:
            raise EnvironmentError_(f"{where}: rocm-sdks.{version} must be a non-empty path")
        resolved = root if root.startswith("/") else f"{repo_root.rstrip('/')}/{root}"
        out.append((str(version), resolved))
    return tuple(out)


def _apply_env_overrides(raw: dict) -> None:
    """``BC_<KEY>`` overrides a scalar of the default host."""
    hosts = raw.get("host") or {}
    name = raw.get("default-host") or next(iter(hosts), None)
    body = hosts.get(name) if name else None
    if not isinstance(body, dict):
        return
    for key in _HOST_SCALAR_KEYS:
        value = os.environ.get("BC_" + key.upper().replace("-", "_"))
        if value is not None:
            body[key] = value


def load(path: str | Path) -> Environment:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnvironmentError_(
            f"{path}: not found -- host settings are not committed; copy "
            "config/environment.example.toml to config/environment.local.toml "
            "and fill it in, or set BIGCHERRY_ENVIRONMENT"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise EnvironmentError_(f"{path}: {exc}") from exc
    _apply_env_overrides(raw)

    hosts: dict[str, Host] = {}
    for name, body in (raw.get("host") or {}).items():
        where = f"{path}: host.{name}"
        if not isinstance(body, dict):
            raise EnvironmentError_(f"{where} must be a table")
        devices = []
        for entry in body.get("devices") or ():
            dwhere = f"{where}.devices"
            locator_raw = entry.get("locator")
            locator = None
            if locator_raw is not None:
                locator = str(locator_raw).lower()
                if not _PCI_LOCATOR_PATTERN.match(locator):
                    raise EnvironmentError_(
                        f"{dwhere}: locator {locator_raw!r} is not a canonical lowercase "
                        "PCI BDF (expected dddd:dd:dd.d)"
                    )
            devices.append(Device(
                index=int(_require(entry, "index", dwhere)),
                arch=str(_require(entry, "arch", dwhere)),
                model=str(entry.get("model", "")),
                vram_mib=int(entry.get("vram-mib", 0)),
                locator=locator,
            ))
        # Device ordinals are how every caller addresses a GPU, so a duplicate
        # would silently make one of them unreachable.
        seen = [d.index for d in devices]
        if len(seen) != len(set(seen)):
            raise EnvironmentError_(f"{where}: duplicate device index in {seen}")
        # PRBE111: a duplicate locator would mean two configured devices
        # claim the same physical PCI slot -- attestation would then accept
        # either as "the" device at that BDF, silently confusing them.
        seen_locators = [d.locator for d in devices if d.locator is not None]
        if len(seen_locators) != len(set(seen_locators)):
            raise EnvironmentError_(f"{where}: duplicate device locator in {seen_locators}")
        hosts[name] = Host(
            name=name,
            description=str(body.get("description", "")).strip(),
            hostname=str(_require(body, "hostname", where)),
            repo_root=str(body.get("repo-root", "")),
            work_root=str(body.get("work-root", "")),
            model_root=str(body.get("model-root", "")),
            bench_harness=str(body.get("bench-harness", "")),
            rocm_root=str(body.get("rocm-root", "")),
            rocm_sdks=_rocm_sdks(body.get("rocm-sdks"), where, str(body.get("repo-root", ""))),
            production_port=int(body.get("production-port", 0)),
            test_port=int(body.get("test-port", 0)),
            busy_processes=tuple(str(n) for n in body.get("busy-processes", ())),
            environment_file=str(
                body.get("environment-file")
                or (str(body["repo-root"]).rstrip("/") + "/config/environment.local.toml" if body.get("repo-root") else "")
            ),
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
    configured = os.environ.get("BIGCHERRY_ENVIRONMENT")
    if configured:
        return Path(configured)
    root = Path(repo_root) if repo_root else _repo_root()
    return root / "config" / "environment.local.toml"


def example_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root else _repo_root()
    return root / "config" / "environment.example.toml"


def load_default(repo_root: str | Path | None = None) -> Environment:
    return load(default_path(repo_root))
