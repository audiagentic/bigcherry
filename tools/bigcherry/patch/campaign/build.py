"""Generic build infrastructure for the patch-validation campaign: HIP env,
CMake configure/request identity, registry generation, tree builds and
shared campaign primitives (PatchCampaignError, _print, bound artifacts)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from bigcherry.core.paths import REPO_ROOT
from bigcherry.experiment.attestation import ExecutionIdentity


# GPT review (req_8429aa8e0d35496e, 2026-09-11): every server session on the
# shared/legacy path that uses this module's HIP-only selector contract
# (require_device_visibility()/DeviceVisibility, see PNRO17) must explicitly
# unset an inherited ROCR_VISIBLE_DEVICES, not merely avoid setting one --
# ambient env still reaches ServerRunner.launch() (dict(os.environ) +
# env_unset, then env_overrides) otherwise, reproducing the double-filtering
# bug. (RD73's own server sessions live in its patch-local producer and apply
# the same contract there.)
_ROCR_VISIBLE_DEVICES_UNSET: tuple[str, ...] = ("ROCR_VISIBLE_DEVICES",)


def _hip_only(env_overrides: "dict[str, str] | None") -> "dict[str, str] | None":
    """Defense in depth alongside env_unset=_ROCR_VISIBLE_DEVICES_UNSET:
    strip ROCR_VISIBLE_DEVICES from a caller-supplied overrides dict too,
    so a future direct caller passing it through selector_env/env_overrides
    cannot reintroduce the double-filtering hazard via override-ordering
    (ServerRunner.launch() applies env_unset BEFORE env_overrides)."""
    if not env_overrides:
        return None
    return {k: v for k, v in env_overrides.items() if k != "ROCR_VISIBLE_DEVICES"}




LLAMA_CPP_SRC = REPO_ROOT / "vendor" / "llama.cpp"


CMAKE_GENERATOR = "Ninja"


class PatchCampaignError(RuntimeError):
    pass


def _print(msg: str) -> None:
    print(f"[patch-campaign] {msg}", flush=True)


def _write_bound_artifact(run_dir: Path, name: str, payload: object) -> dict[str, str]:
    target = run_dir / "artifacts" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "path": target.relative_to(run_dir).as_posix(),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def _hip_env(hip_path: Path) -> dict[str, str]:
    """Environment for cmake configure/build subprocesses with the chosen
    vendored ROCm toolchain (tools/rocm-env.ps1/.sh) explicit in-process,
    rather than assuming the invoking shell already sourced it -- this tool
    is meant to run unattended/backgrounded, where that assumption doesn't
    hold (hit for real: `find_package(hip)` failed with CMAKE_PREFIX_PATH
    unset when this campaign was launched from a plain shell)."""
    import os

    env = os.environ.copy()
    env["ROCM_PATH"] = str(hip_path)
    env["HIP_PATH"] = str(hip_path)
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(
        p for p in (str(hip_path), env.get("CMAKE_PREFIX_PATH", "")) if p
    )
    env["PATH"] = os.pathsep.join([str(hip_path / "bin"), env.get("PATH", "")])
    return env


def resolve_selected_device_execution_identity(
    *,
    expected_arch: str,
    host_devices=None,
) -> tuple[ExecutionIdentity, dict[str, str]]:
    """PRBE111 shared helper: resolve the real, host-configured
    ExecutionIdentity (architecture + verified PCI locator) and env-override
    dict for the ONE physical device the ambient HIP_VISIBLE_DEVICES
    currently selects.

    Required, fail-closed semantics (never guessed, never partial):
    - HIP_VISIBLE_DEVICES must be explicitly set in the environment.
    - It must select exactly one device (a single numeric index).
    - That index must be a real, configured Device in this host's
      inventory (config/environment.toml).
    - The configured Device must have a real, verified `locator`.
    - The configured Device's `arch` must equal `expected_arch` -- a
      selector pointing at the wrong physical architecture is a caller
      bug, not something to silently accept.

    Any server-based (llama-server) producer needing real execution
    attestation should go through this helper rather than constructing
    its own ExecutionIdentity, so the "never guess a locator" discipline
    stays in exactly one place.
    """
    import os

    from bigcherry.core import environment as bc_environment

    raw = os.environ.get("HIP_VISIBLE_DEVICES")
    if not raw:
        raise PatchCampaignError(
            "resolve_selected_device_execution_identity: HIP_VISIBLE_DEVICES "
            "must be explicitly set (an unset/ambient-default device list "
            "cannot be trusted for a real-hardware measurement)"
        )
    selectors = [part for part in raw.split(",") if part != ""]
    if len(selectors) != 1:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: HIP_VISIBLE_DEVICES={raw!r} "
            "must select exactly one device for a server-based producer"
        )
    try:
        index = int(selectors[0])
    except ValueError as exc:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: HIP_VISIBLE_DEVICES={raw!r} "
            "is not a numeric device index"
        ) from exc

    if host_devices is None:
        host_devices = bc_environment.load_default().host().devices
    matches = [d for d in host_devices if d.index == index]
    if not matches:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: HIP_VISIBLE_DEVICES "
            f"selects index {index}, which is not a configured device in "
            f"config/environment.toml (known indices: {sorted(d.index for d in host_devices)})"
        )
    device = matches[0]
    if device.locator is None:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: device index {index} "
            f"({device.arch}) has no verified locator in config/environment.toml -- "
            "add one (via real `rocm-smi --showbus` output, never invented) before "
            "using it with a server-based attestation producer"
        )
    if device.arch != expected_arch:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: device index {index} is "
            f"configured as {device.arch!r}, but {expected_arch!r} was requested -- "
            "HIP_VISIBLE_DEVICES selects the wrong physical architecture"
        )

    identity = ExecutionIdentity(
        backend="ROCm",
        architectures=(device.arch,),
        locators=(device.locator,),
    )
    selector_env = {"HIP_VISIBLE_DEVICES": str(index)}
    return identity, selector_env


def _requested_cmake_args(
    amdgpu_targets: str, extra_cmake_args: list[str]
) -> list[str]:
    """Identity-relevant CMake intent shared by configure and post-build
    verification -- ONE definition so the two can never silently drift
    apart (capture_completed_build_evidence() checks these same values
    against the resolved CMakeCache.txt)."""
    return [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_HIP=ON",
        f"-DAMDGPU_TARGETS={amdgpu_targets}",
        *extra_cmake_args,
    ]


def _full_requested_cmake_args(
    *,
    hip_path: Path,
    amdgpu_targets: str,
    extra_cmake_args: list[str],
) -> list[str]:
    """Every -D value actually supplied to `cmake` configure, AND what
    capture_completed_build_evidence() is told was requested -- the two
    must describe the same build intent, or the verifier's "requested vs
    resolved cache" check is comparing against an incomplete picture."""
    is_windows = sys.platform == "win32"
    clang = hip_path / "bin" / ("clang.exe" if is_windows else "clang")
    clangxx = hip_path / "bin" / ("clang++.exe" if is_windows else "clang++")

    args = [
        *_requested_cmake_args(amdgpu_targets, extra_cmake_args),
        f"-DCMAKE_C_COMPILER={clang}",
        f"-DCMAKE_CXX_COMPILER={clangxx}",
        f"-DCMAKE_PREFIX_PATH={hip_path}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]

    if is_windows:
        # ROCm's clang driver invokes lld-link, which (like MSVC link.exe)
        # embeds a wall-clock PE timestamp by default -- every relink
        # produces different binary bytes even with byte-identical inputs.
        # /Brepro makes lld-link derive that field from content instead,
        # so a genuine no-op relink is byte-reproducible. Found for real
        # via HI82 item 9: two back-to-back identical campaign runs
        # produced different runtime_bundle_hash values purely from this
        # (diagnosed with GPT, req_cc5af49494fe457a).
        args += [
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,/Brepro",
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,/Brepro",
            "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,/Brepro",
        ]

    return args


_CONFIGURE_REQUEST_SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _configure_request_document(
    *, source: Path, cmake_args: list[str]
) -> dict[str, object]:
    return {
        "schema_version": _CONFIGURE_REQUEST_SCHEMA_VERSION,
        "source": str(source.resolve()),
        "generator": CMAKE_GENERATOR,
        "cmake_args": list(cmake_args),
    }



def _drop_foreign_cmake_cache(build_dir: Path, source: Path) -> None:
    """A build dir keyed by name (not source) outlives a source change: a
    promotion changes the control composition, so the content-addressed
    worktree moves and CMake refuses the old cache ("does not match the
    source"). Drop the cache so the configure below starts clean; objects
    are rebuilt (cheaply, through the shared compiler cache)."""
    import shutil

    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:"):
            home = line.split("=", 1)[1].strip()
            if Path(home).resolve() != Path(source).resolve():
                _print(f"{build_dir.name}: CMake cache belongs to {home}; reconfiguring from {source}")
                cache.unlink()
                shutil.rmtree(build_dir / "CMakeFiles", ignore_errors=True)
                (build_dir / "bigcherry-configure-request.json").unlink(missing_ok=True)
            return

def _configure_request_matches(*, build_dir: Path, expected: dict[str, object]) -> bool:
    """Whether build_dir's existing CMake cache was configured by the exact
    same request as `expected` -- the real fix for the old, too-broad "skip
    configure whenever CMakeCache.txt exists" check (which could silently
    reuse a stale configuration across differently-parameterized
    invocations) without paying for a full reconfigure on every single run
    (which itself can dirty Ninja's dependency graph unnecessarily)."""
    cache = build_dir / "CMakeCache.txt"
    request_path = build_dir / "bigcherry-configure-request.json"
    if not cache.is_file() or not request_path.is_file():
        return False
    try:
        recorded = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == expected


def generate_registry(
    *, source: Path, amdgpu_targets: str, generated_dir: Path
) -> None:
    """Run `bigcherry generate` against an isolated worktree so ggml-hip's
    CMakeLists finds hip-autotune-registry.inc + template-instances/ there.

    Real gap found via HI82 isolation testing: the pre-isolation campaign
    tool never called this itself -- it worked only because the shared
    vendor/llama.cpp tree already had a stale in-tree registry.inc left
    over from earlier manual `bigcherry generate` runs. A fresh isolated
    worktree has no such leftover, so cmake failed with "hip-autotune-
    registry.inc is missing" the first time this ran for real. Writing to
    an explicit --generated-root (not the in-tree default) matches the
    "campaign builds use an out-of-tree generated directory" contract
    already documented in ggml/src/ggml-hip/CMakeLists.txt.

    --force is required: `bigcherry generate`'s "unpatched tree" guard
    checks a ReleaseRecord keyed by git revision (releases/<rev>.json),
    which knows nothing about an isolated worktree's own out-of-band
    patcher.apply_all() run -- patch_source_isolation.materialize_source()
    is itself the real proof the tree is patched (it raises on any failed
    edit), so this bypass is sound, not a shortcut around a real check."""
    import subprocess

    generated_dir.mkdir(parents=True, exist_ok=True)
    _print(f"generating autotune registry into {generated_dir} ...")
    args = [
        sys.executable,
        "-m",
        "bigcherry",
        "--llama-root",
        str(source),
        "generate",
        "--arch",
        amdgpu_targets,
        "--generated-root",
        str(generated_dir),
        "--force",
    ]
    result = subprocess.run(
        args,
        cwd=str(REPO_ROOT / "tools"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise PatchCampaignError(f"bigcherry generate failed:\n{result.stdout}")


def build_tree(
    *,
    name: str,
    extra_cmake_args: list[str],
    hip_path: Path,
    amdgpu_targets: str,
    workdir: Path,
    targets: list[str],
    source: Path,
    generated_proof_callback=None,
) -> Path:
    """cmake configure (if not already configured) + build the given
    targets. Returns the build tree's bin/ directory.

    `source` is an isolated, content-addressed worktree from
    patch_source_isolation.materialize_source() -- never the shared
    vendor/llama.cpp working tree (HI82: that sharing is exactly the
    contamination risk this module was rewritten to close)."""
    import subprocess

    build_dir = workdir / name
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = _hip_env(hip_path)
    cmake_args = _full_requested_cmake_args(
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        extra_cmake_args=extra_cmake_args,
    )
    configure_request = _configure_request_document(
        source=source, cmake_args=cmake_args
    )
    configure_request_path = build_dir / "bigcherry-configure-request.json"

    if generated_proof_callback is not None:
        generated_proof_callback("preconfigure", build_dir)
    _drop_foreign_cmake_cache(build_dir, source)

    # Reconfigure only when the REQUEST actually changed -- the old "skip
    # whenever CMakeCache.txt exists" check could silently reuse a stale
    # configuration across differently-parameterized invocations; the
    # opposite extreme, always reconfiguring, was tried this session and
    # found to needlessly perturb Ninja's dependency graph on every run
    # (compounding with autotune_catalog's now-fixed unconditional compile-
    # input rewrites to make byte-stable resume unreachable in practice).
    if _configure_request_matches(build_dir=build_dir, expected=configure_request):
        _print(f"{name}: configure request unchanged; reusing CMake cache")
    else:
        _print(f"configuring {name} ...")
        args = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build_dir),
            "-G",
            CMAKE_GENERATOR,
            *cmake_args,
        ]
        log_path = log_dir / f"{name}-configure.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                args, stdout=log_file, stderr=subprocess.STDOUT, env=env
            )
        if result.returncode != 0:
            raise PatchCampaignError(f"{name} configure failed (see {log_path})")
        _atomic_write_json(configure_request_path, configure_request)

    if generated_proof_callback is not None:
        generated_proof_callback("postconfigure-precompile", build_dir)

    for target in targets:
        _print(f"building {name} ({target}) ...")
        log_path = log_dir / f"{name}-build-{target}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", target, "-j"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
        if result.returncode != 0:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                    -40:
                ]
            )
            raise PatchCampaignError(f"{name} build ({target}) failed:\n{tail}")
    if generated_proof_callback is not None:
        generated_proof_callback("postcompile", build_dir)
    _print(f"{name}: OK")
    return build_dir / "bin"


def ensure_stock_baseline(
    *,
    hip_path: Path,
    amdgpu_targets: str,
    workdir: Path,
    stock_src: Path,
) -> Path:
    """Build a genuinely unpatched llama.cpp worktree at stock_src (from
    patch_source_isolation.materialize_stock_source(), a real git worktree
    pinned to base_revision with zero patches applied) for a stock
    comparison arm. A patch under test never touches this tree."""
    import subprocess

    build_dir = workdir / "stock"
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _hip_env(hip_path)
    cmake_args = _full_requested_cmake_args(
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        extra_cmake_args=[],
    )
    configure_request = _configure_request_document(
        source=stock_src, cmake_args=cmake_args
    )
    _drop_foreign_cmake_cache(build_dir, stock_src)
    configure_request_path = build_dir / "bigcherry-configure-request.json"

    if _configure_request_matches(build_dir=build_dir, expected=configure_request):
        _print("stock: configure request unchanged; reusing CMake cache")
    else:
        _print("configuring stock baseline ...")
        args = [
            "cmake",
            "-S",
            str(stock_src),
            "-B",
            str(build_dir),
            "-G",
            CMAKE_GENERATOR,
            *cmake_args,
        ]
        log_path = log_dir / "stock-configure.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                args, stdout=log_file, stderr=subprocess.STDOUT, env=env
            )
        if result.returncode != 0:
            raise PatchCampaignError(f"stock configure failed (see {log_path})")
        _atomic_write_json(configure_request_path, configure_request)
    for target in ("llama-bench",):
        log_path = log_dir / f"stock-build-{target}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", target, "-j"],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise PatchCampaignError(f"stock build ({target}) failed (see {log_path})")
    _print("stock: OK")
    return build_dir / "bin"


def _canonical_json(value: object, *, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PatchCampaignError(f"{label} is not canonical JSON data: {exc}") from exc
    return encoded


def _canonical_json_mapping(value: Mapping[str, object], *, label: str) -> str:
    if not isinstance(value, Mapping) or not value:
        raise PatchCampaignError(f"{label} must be a non-empty mapping")
    encoded = _canonical_json(dict(value), label=label)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise PatchCampaignError(f"{label} must encode a JSON object")
    return encoded


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PatchCampaignError(f"cannot hash evidence file {path}: {exc}") from exc
    return digest.hexdigest()
