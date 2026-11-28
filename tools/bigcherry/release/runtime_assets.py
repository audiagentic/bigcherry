"""BU01: deterministic, verified Radiance runtime release bundles.

Release records bind the asset to the pin-bump lane build, source trees and
ROCm version. Never install a release asset before verifying the record's
digest, every archived file's digest and current stock source identity.
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..core import engines

SCHEMA = "bigcherry-radiance-runtime-v1"
RELEASE_LANE = "radiance:stock:linux-radiance"
RECIPES = "engines/radiance/recipes.toml"
SERVING_COMPONENT = "engines/radiance/components/serving"
SERVING_FINGERPRINT = f"{SERVING_COMPONENT}/generated/v3/contract.fingerprint"
SERVING_EDITION = "BC_SERVING_CONTRACT_VERSION"   # a build-cache entry; the only edition is 3
PLAN = re.compile(r"(?m)^(?:pin-bump lane build|release asset lane build): "
                  r"(radiance:stock:linux-radiance) "
                  r"build_plan_id=([0-9a-f]{32}) binary_sha256=([0-9a-f]{64})$")
FILE_LIMIT = 10000
BYTE_LIMIT = 16 * 1024**3
#: The public bc-radiance-1170.0.0 record in releases/v1.17.0.json uses
#: radiance-<plan>.tar.gz; keep accepting that *published* historical archive.
PRODUCT = "bigcherryrad"


def _archive_pattern(*, allow_published_legacy: bool) -> re.Pattern[str]:
    """CI names are product-only; install also recognizes the pinned legacy prefix."""
    prefix = f"(?:{PRODUCT}|radiance)" if allow_published_legacy else PRODUCT
    return re.compile(rf"{prefix}-[0-9a-f]{{32}}\.tar\.gz")


CI_ARCHIVE = _archive_pattern(allow_published_legacy=False)
ARCHIVE_NAME = _archive_pattern(allow_published_legacy=True)
CI_RECEIPT_SUFFIX = ".receipt.json"
# Immutable *policy* source of truth for allowed pre-scope assets, not read
# from mutable release records at runtime. The test asserts exact parity with
# releases/*.json so edits to published records cannot silently expand access.
# Never grandfather a future unscoped bundle by filename alone.
PUBLISHED_LEGACY_ASSETS = {
    "bc-radiance-1170.0.0": {
        "name": "radiance-e5c7cb3dcc920259656edb17ec57ffc0.tar.gz",
        "sha256": "6e74fb9daa33ffe47b8ac181ba2a3563031b90174feea22a8c7fca5a2038686a",
        "manifest_sha256": "02e6dc896389719136662bfb6ff36767bcbc1d1a3a4a8bbba1d1f37e77d5cb92",
    },
}

#: The build's own record (rocm_root, plan identity, toolchain): the launcher's SDK check and the campaign tools read it from the
#: build directory, so an installed build is not usable by them without it. It travels in the bundle like any file, digest included.
BUILD_METADATA = "bigcherry-build-metadata-radiance.json"
# The single run-side payload contract. Each (build-relative glob, required)
# entry names files used by the engine, NOT build products/benchmarks or
# platform tooling such as rad-* diagnostics. bc-serve is engine-coupled and
# must match bin/radiance. Glob groups allow new plugin names to
# ship without hard-coding a model architecture inventory. Required entries
# must match at least one file in every newly packaged build.
RUN_SCOPE = "engine-runtime"
ELF_DEBUG_POLICY = "gnu-objcopy-split-debug-strip-all-v1"
# Deliberately explicit: unknown RSC12 configurations must be reviewed and
# added to the engine payload contract, not silently copied from a checkout.
# Native profiles are selected by exact name: one per named serving config (engines/radiance/configs/NAME.yaml, the
# single authority of that config's engine settings). A checkout's other YAML is local input.
NATIVE_RELEASE_PROFILES = (
    "flashnext-2xtx-combine.yaml",
    "flashnext-2xtx-rdna2-head.yaml",
    "flashnext-2xtx-rdna2-tail-combine.yaml",
    "flashnext-2xtx-rdna2-tail.yaml",
    "flashnext-2xtx.yaml",
    "flashnext-4gpu-bf16i8.yaml",
    "flashnext-4gpu.yaml",
    "flashnext-r9700.yaml",
    "flashnext-tp2xtx-r9700.yaml",
    "flashnext-tp2xtx.yaml",
    "qwen27b-2xtx.yaml",
    "qwen27b-tp3-capacity.yaml",
    "qwen27b-tp3-iq6r.yaml",
    "qwen27b-tp3-long-bf16.yaml",
    "qwen27b-tp3-long-bf16i8.yaml",
    "qwen27b-tp3-long-iq6r-bf16i8.yaml",
    "qwen27b-tp3-long-iq6r.yaml",
    "qwen27b-tp3-long-q4.yaml",
    "qwen27b-tp3-long-q6.yaml",
    "qwen27b-tp3-long.yaml",
    "qwen27b-tp3.yaml",
)
OPTIONAL_NATIVE_PROFILES = NATIVE_RELEASE_PROFILES
NATIVE_PROFILE_MEMBERS = frozenset(f"configs/{name}" for name in OPTIONAL_NATIVE_PROFILES)
NATIVE_PROFILE_LIMIT = 16 * 1024 * 1024
# Approved repo-owned heat seeds. Archive members live at prefix/heat; no
# checkout, lab launcher, or host-specific seed path is shipped.
SHIPPED_HEAT_SEEDS = ("flashnext-mixed.hot-first", "perf51.rank0.hot-first")
# Reviewed digests bind sources outside engine_tree/config_tree to the payload
# policy runtime key. Updating a seed requires changing this record and hence
# forces CI to build a new asset, rather than reusing an earlier release bundle.
SHIPPED_HEAT_DIGESTS = {
    "flashnext-mixed.hot-first": "230398970a6f744723a90f4799e4885a664e06742d8a6f1224ee32f0bfd83b29",
    "perf51.rank0.hot-first": "cbf8e17ad4e0e57470da2c000904d863621d5affcf8876c9b87960b9472ba06e",
}
RUN_MANIFEST = (
    ("bin/radiance", True),
    ("radiance_home/bin/bc-serve", True),
    ("radiance_home/bin/bc-release", True),
    (BUILD_METADATA, True),
    *((f"heat/{name}", True) for name in SHIPPED_HEAT_SEEDS),
    ("radiance_home/helpers/r3_moe_worker", True),
    ("radiance_home/helpers/*.so*", False),
    ("radiance_home/kernels/*.so*", True),
    ("radiance_home/architectures/*.so*", True),
    ("radiance_home/quantizers/*.so*", True),
    ("radiance_home/vectors/*", False),
    ("runtime-prefix/lib/*.so*", False),
    ("runtime-prefix/lib/**/*.so*", False),
    ("runtime-prefix/lib64/*.so*", False),
    ("runtime-prefix/lib64/**/*.so*", False),
)


def selected_native_profiles(names: set[str]) -> tuple[str, ...]:
    return tuple(name for name in OPTIONAL_NATIVE_PROFILES if f"configs/{name}" in names)


def check_native_profile_selection(native_profiles: tuple[str, ...]) -> None:
    if len(set(native_profiles)) != len(native_profiles) or set(native_profiles) - set(OPTIONAL_NATIVE_PROFILES):
        raise BundleError("unapproved or repeated native profile selection")


def payload_contract_key(*, native_profiles: tuple[str, ...] = ()) -> str:
    """Versioned payload schema and selection policy, including optional files."""
    policy = {"schema": SCHEMA, "scope": RUN_SCOPE, "manifest": RUN_MANIFEST,
              "build_metadata": BUILD_METADATA, "build_identity_schema": 2,
              "elf_debug_policy": ELF_DEBUG_POLICY,
              "shipped_heat_digests": SHIPPED_HEAT_DIGESTS}
    if native_profiles:
        check_native_profile_selection(native_profiles)
        policy["native_profiles"] = sorted(native_profiles)
    return hashlib.sha256(json.dumps(policy, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()


class BundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseTarget:
    lane: str
    profiles: tuple[str, ...]
    schema_fingerprint: str
    semantic_fingerprint: str

    def select_profiles(self, requested: tuple[str, ...]) -> tuple[str, ...]:
        check_native_profile_selection(requested)
        if requested and requested != self.profiles:
            raise BundleError("native release requires its declared YAML profiles")
        return self.profiles

    def serving_contract(self, files: dict) -> dict:
        return {"edition": 3, "lane": self.lane, "schema_fingerprint": self.schema_fingerprint,
                "semantic_fingerprint": self.semantic_fingerprint,
                "engine_sha256": files["bin/radiance"]["sha256"],
                "owner_sha256": files["radiance_home/bin/bc-serve"]["sha256"]}


def release_target(root: Path, tag: str | None = None) -> ReleaseTarget:
    """Resolve the release lane from the exact source checkout or immutable tag."""
    def read(path: str) -> str:
        if tag is None:
            return (root / path).read_text(encoding="utf-8")
        result = subprocess.run(["git", "-C", str(root), "show", f"refs/tags/{tag}:{path}"],
                                check=False, capture_output=True, text=True)
        if result.returncode:
            raise BundleError(f"missing {path} at release tag {tag}")
        return result.stdout

    fingerprint = re.fullmatch(
        r"schema (sha256:[0-9a-f]{64})\nsemantics (sha256:[0-9a-f]{64})\n?",
        read(SERVING_FINGERPRINT),
    )
    if fingerprint is None:
        raise BundleError("invalid committed v3 serving contract fingerprints")
    return ReleaseTarget(RELEASE_LANE, NATIVE_RELEASE_PROFILES, *fingerprint.groups())


def verify_serving_build(build: Path, root: Path, target: ReleaseTarget) -> None:
    """Require both halves of the shipped startup protocol to use the same edition."""
    from ..campaign.runtime_bundle import component_build_dir

    owner = component_build_dir(build, "serving", root / SERVING_COMPONENT)
    expected = f"schema {target.schema_fingerprint}\nsemantics {target.semantic_fingerprint}"
    for role, directory, generated in (
        ("engine", build, build / "core/bc-serving-generated/v3"),
        ("owner", owner, owner / "bc-serving-generated/v3"),
    ):
        try:
            cache = (directory / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise BundleError(f"native release has no effective {role} build configuration") from exc
        editions = re.findall(rf"(?m)^{SERVING_EDITION}:[^=\n]+=(.*)$", cache)
        if editions != ["3"]:
            raise BundleError(f"native release requires serving edition 3 in the {role} build")
        try:
            fingerprint = (generated / "contract.fingerprint").read_text(encoding="utf-8").rstrip("\n")
        except OSError as exc:
            raise BundleError(f"native release has no generated {role} serving fingerprint") from exc
        if fingerprint != expected:
            raise BundleError(f"generated {role} serving fingerprint differs from the release recipe")
    built_owner = owner / "bin/bc-serve"
    staged_owner = build / "radiance_home/bin/bc-serve"
    if not built_owner.is_file() or not staged_owner.is_file() or digest(built_owner) != digest(staged_owner):
        raise BundleError("staged bc-serve differs from the verified native owner build")


def require_published_legacy_asset(asset: dict, tag: str | None) -> None:
    published = PUBLISHED_LEGACY_ASSETS.get(tag)
    if published is None or any(asset.get(key) != value for key, value in published.items()):
        raise BundleError("unrecognized legacy unscoped runtime asset")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_tree(root: Path, tree: str) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", f"HEAD:{tree}"],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def sdk_version(sdk: Path) -> str:
    version = sdk / ".info/version"
    if not version.is_file():
        raise BundleError(f"selected ROCm SDK has no version file: {version}")
    return version.read_text().strip()


def stock_identity(root: Path) -> dict[str, str]:
    return {"engine_tree": git_tree(root, "engines/radiance"),
            "config_tree": git_tree(root, "config")}


def source_tree_marker(asset: dict, sdk: Path) -> str:
    """Exact build marker from the verified receipt and selected SDK.

    No mutable installed-tree metadata contributes to the expected value.
    """
    version = sdk_version(sdk)
    if asset.get("rocm_version") != version:
        raise BundleError("source-tree marker SDK differs from release receipt")
    return (f"{asset['engine_tree']} {asset['config_tree']} "
            f"{sdk.resolve()} {version} ")


def select_build(record: dict, work_root: Path, lane: str = RELEASE_LANE) -> tuple[Path, str, str]:
    found = PLAN.findall(record.get("notes", ""))
    if not found:
        raise BundleError("release record has no pin-bump lane build plan ID and binary SHA-256")
    recorded_lane, plan_id, binary_sha = found[-1]
    if recorded_lane != lane:
        raise BundleError(f"release record selects {recorded_lane}, expected {lane}")
    matches = list((work_root / "builds").glob(f"*/{plan_id}"))
    if len(matches) != 1 or not matches[0].is_dir():
        raise BundleError(f"expected one lane build for {plan_id}, found {len(matches)}")
    binary = matches[0] / "bin/radiance"
    if not binary.is_file() or digest(binary) != binary_sha:
        raise BundleError("lane build binary missing or SHA-256 differs from release record")
    return matches[0], plan_id, binary_sha


def built_from(root: Path) -> dict:
    """The exact upstream pin and each owned component's build-against revision.

    Read the declared closure on every pack. A pin move via pin-bump updates
    recipes and component.toml together; a stale or missing declaration refuses
    packaging instead of publishing a mixed engine and plugin generation.
    """
    import tomllib

    engine_file = root / "engines/radiance/engine.toml"
    recipes_file = root / "engines/radiance/recipes.toml"
    try:
        engine = tomllib.loads(engine_file.read_text(encoding="utf-8"))
        recipes = tomllib.loads(recipes_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BundleError(f"cannot read Radiance build identity: {exc}") from exc
    pinned = recipes.get("pinned")
    pinned_version = recipes.get("pinned-version")
    declared = engine.get("components")
    if (not isinstance(pinned, str) or not pinned or
            not isinstance(pinned_version, str) or not pinned_version or
            not isinstance(declared, dict) or not declared):
        raise BundleError("incomplete Radiance pin or component identity declarations")
    components = {}
    for name, relative in sorted(declared.items()):
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]+", name) or
                relative != f"engines/radiance/components/{name}"):
            raise BundleError(f"invalid Radiance component path: {name}")
        try:
            comp = tomllib.loads((root / relative / "component.toml").read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise BundleError(f"missing or invalid build identity for {name}: {exc}") from exc
        against = comp.get("built-against")
        if against != pinned:
            raise BundleError(f"{name} built-against {against!r} differs from Radiance pin {pinned!r}")
        components[name] = against
    return {"project": engines.RADIANCE, "upstream": str(engine.get("upstream", "")),
            "pinned": pinned, "pinned_version": pinned_version,
            "components": components, "note": "BigCherry's patched build, not an upstream release"}


def check_no_interpreter(name: str, path: Path) -> None:
    """Reject scripts and text whose runtime depends on an interpreter.

    The engine payload may contain ELF binaries and opaque binary catalogs,
    but never script entrypoints. Inspect UTF-8 data/metadata, rather than
    scanning arbitrary binary bytes as though they were source code.
    """
    rel = PurePosixPath(name)
    native_profile = name in NATIVE_PROFILE_MEMBERS
    if native_profile and not 0 < path.stat().st_size <= NATIVE_PROFILE_LIMIT:
        raise BundleError(f"native profile must be bounded nonempty YAML: {name}")
    if (any(part == "__pycache__" for part in rel.parts)
            or rel.suffix.lower() in {".py", ".pyc", ".pyo", ".sh", ".bash", ".zsh"}
            or rel.name.startswith(("python", "pypy"))):
        raise BundleError(f"interpreter or shell file in runtime asset: {name}")
    with path.open("rb") as source:
        if source.read(2) == b"#!":
            raise BundleError(f"script shebang in runtime asset: {name}")
        if native_profile or rel.suffix.lower() in {".toml", ".json", ".txt", ".ini", ".conf"}:
            # Scan the complete text file, not just its first bytes. Text is
            # streamed one line at a time; large data files are not decoded.
            source.seek(0)
            for line in source:
                try:
                    text = line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise BundleError(f"non-UTF8 runtime metadata: {name}") from exc
                if native_profile and "\0" in text:
                    raise BundleError(f"NUL in native profile: {name}")
                if re.search(r"(?i)\b(?:python(?:[0-9]+(?:\.[0-9]+)*)?|pypy(?:[0-9]+)?)\b", text):
                    raise BundleError(f"interpreter reference in runtime asset: {name}")


def check_committed_config(root: Path, name: str, content: bytes) -> None:
    """The shipped config must be the committed engine source at HEAD."""
    relative = f"engines/radiance/configs/{name}"
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
        check=False, capture_output=True,
    )
    if result.returncode or result.stdout != content:
        raise BundleError(f"uncommitted or absent tagged RSC12 config: {relative}")


def check_committed_heat(root: Path, name: str, content: bytes) -> None:
    """Ship only the exact checked-in heat source, never mutable lab output."""
    if name not in SHIPPED_HEAT_DIGESTS or hashlib.sha256(content).hexdigest() != SHIPPED_HEAT_DIGESTS[name]:
        raise BundleError(f"unapproved heat seed digest: {name}")
    relative = f"tools/lab/radiance/heat/{name}"
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
        check=False, capture_output=True,
    )
    if result.returncode or result.stdout != content:
        raise BundleError(f"uncommitted or absent tagged heat seed: {relative}")


def check_run_manifest(names: set[str]) -> None:
    """A restored scoped bundle has exactly run-side files and every required kind."""
    for name in names:
        rel = PurePosixPath(name)
        if (any(part == "__pycache__" for part in rel.parts)
                or rel.suffix.lower() in {".py", ".pyc", ".pyo", ".sh", ".bash", ".zsh"}):
            raise BundleError(f"interpreter or shell file in runtime asset: {name}")
        if name not in NATIVE_PROFILE_MEMBERS and not any(
                fnmatch.fnmatchcase(name, pattern) for pattern, _ in RUN_MANIFEST):
            raise BundleError(f"non-runtime file in scoped bundle: {name}")
    for pattern, required in RUN_MANIFEST:
        if required and not any(fnmatch.fnmatchcase(name, pattern) for name in names):
            raise BundleError(f"missing runtime requirement: {pattern}")


def runtime_files(build: Path, root: Path, *, native_profiles: tuple[str, ...] = ()) -> dict[str, Path]:
    """Choose the binary build plus the *committed* engine config closure.

    Configs are source inputs, not generated build products or site overlays.
    The archive-relative paths are independent of the source checkout.
    """
    check_native_profile_selection(native_profiles)
    files: dict[str, Path] = {}
    build_root = build.resolve()
    for pattern, _ in RUN_MANIFEST:
        if pattern.startswith(("configs/", "heat/")):
            continue
        for item in sorted(build.glob(pattern)):
            if not item.is_file():
                continue
            if not item.resolve().is_relative_to(build_root):
                raise BundleError(f"runtime symlink escapes build directory: {item}")
            files[item.relative_to(build).as_posix()] = item
    config_dir = root / "engines/radiance/configs"
    if config_dir.is_symlink() or config_dir.parent.is_symlink():
        raise BundleError("symlink in native config source directory")
    for name in native_profiles:
        source = config_dir / name
        if not source.is_file() or source.is_symlink():
            raise BundleError(f"missing or unsafe RSC12 config: {name}")
        if name in native_profiles:
            if source.stat().st_mode & 0o111:
                raise BundleError(f"native profile must be non-executable: {name}")
            check_no_interpreter(f"configs/{name}", source)
        check_committed_config(root, name, source.read_bytes())
        files[f"configs/{name}"] = source
    heat_dir = root / "tools/lab/radiance/heat"
    if heat_dir.is_symlink() or heat_dir.parent.is_symlink():
        raise BundleError("symlink in heat seed source directory")
    for name in SHIPPED_HEAT_SEEDS:
        source = heat_dir / name
        if not source.is_file() or source.is_symlink():
            raise BundleError(f"missing or unsafe heat seed: {name}")
        check_committed_heat(root, name, source.read_bytes())
        files[f"heat/{name}"] = source
    check_run_manifest(set(files))
    for name, path in files.items():
        check_no_interpreter(name, path)
    if len(files) > FILE_LIMIT:
        raise BundleError("too many files in runtime bundle")
    if sum(path.stat().st_size for path in files.values()) > BYTE_LIMIT:
        raise BundleError("runtime bundle exceeds size limit")
    return files


def split_elf_debug(files: dict[str, Path], scratch: Path
                    ) -> tuple[dict[str, Path], dict[str, Path]]:
    """Split each shipped host ELF without touching the recorded build inputs.

    GNU objcopy operates on the host ELF container, not embedded GPU code
    objects. Separate debug files live outside the runtime archive; every
    checksum of a runtime member is computed from its stripped copy.
    """
    selected = dict(files)
    debug: dict[str, Path] = {}
    objcopy = shutil.which("objcopy")
    for name, source in sorted(files.items()):
        with source.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        if objcopy is None:
            raise BundleError("GNU objcopy required to strip runtime ELF debug data")
        stripped = scratch / "runtime" / name
        symbol_file = scratch / "debug" / (name + ".debug")
        stripped.parent.mkdir(parents=True, exist_ok=True)
        symbol_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, stripped)
        for args in (("--only-keep-debug", str(stripped), str(symbol_file)),
                     ("--strip-all", str(stripped)),
                     (f"--add-gnu-debuglink={symbol_file}", str(stripped))):
            result = subprocess.run([objcopy, *args], check=False, capture_output=True, text=True)
            if result.returncode:
                raise BundleError(f"ELF debug split failed for {name}: {result.stderr.strip()}")
        selected[name] = stripped
        debug[name] = symbol_file
    return selected, debug


def pack(record: dict, root: Path, work_root: Path, sdk: Path, tag: str,
         *, native_profiles: tuple[str, ...] = (), target: ReleaseTarget | None = None) -> dict:
    if not re.fullmatch(r"bc-radiance-[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise BundleError("unsafe Radiance release tag")
    target = target or release_target(root)
    build, plan_id, binary_sha = select_build(record, work_root, target.lane)
    native_profiles = target.select_profiles(native_profiles)
    files = runtime_files(build, root, native_profiles=native_profiles)
    verify_serving_build(build, root, target)
    identity = stock_identity(root)
    version = sdk_version(sdk)
    built_identity = built_from(root)
    out = work_root / "release-assets" / tag
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".elf-debug-", dir=out) as directory:
        files, debug = split_elf_debug(files, Path(directory))
        manifest = {
            "schema": SCHEMA, "product": PRODUCT, "payload_scope": RUN_SCOPE, "built_from": built_identity, "release_tag": tag,
            "plan_id": plan_id, "source_slice_id": build.parent.name,
            "binary_sha256": digest(files["bin/radiance"]),
            "source_binary_sha256": binary_sha,
            "rocm_version": version, **identity,
            "files": {
                name: {"sha256": digest(path), "size": path.stat().st_size,
                       "mode": 0o755 if os.access(path, os.X_OK) else 0o644}
                for name, path in sorted(files.items())
            },
        }
        contract = target.serving_contract(manifest["files"])
        if contract is not None:
            manifest["serving_contract"] = contract
        encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        name = f"{PRODUCT}-{plan_id}.tar.gz"
        dest = out / name
        staged = out / (name + ".pending")
        with staged.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for item_name, content in [("bundle.json", encoded)]:
                        info = tarfile.TarInfo(item_name)
                        info.size, info.mode, info.mtime = len(content), 0o644, 0
                        info.uid = info.gid = 0
                        tar.addfile(info, io.BytesIO(content))
                    for item_name, path in sorted(files.items()):
                        info = tarfile.TarInfo(item_name)
                        info.size, info.mode, info.mtime = path.stat().st_size, manifest["files"][item_name]["mode"], 0
                        info.uid = info.gid = 0
                        with path.open("rb") as f:
                            tar.addfile(info, f)
        sha = digest(staged)
        checksums = name + ".sha256"
        asset = {"schema": SCHEMA, "product": PRODUCT, "payload_scope": RUN_SCOPE,
                 "file_names": sorted(manifest["files"]), "name": name, "sha256": sha,
                 "checksum_name": checksums, "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                 "plan_id": plan_id, "source_slice_id": build.parent.name,
                 "rocm_version": version, "binary_sha256": digest(files["bin/radiance"]),
                 "source_binary_sha256": binary_sha, "built_from": built_identity, **identity}
        if contract is not None:
            asset["serving_contract"] = contract
        old = record.get("runtime_assets", {}).get(tag)
        if old is not None and old != asset:
            staged.unlink()
            raise BundleError("release record asset changed; refusing to replace existing verified digest")
        # Keep optional debug symbols alongside the work root, not inside the
        # release upload staging directory (which must contain exactly three files).
        for item_name, symbol_file in debug.items():
            sidecar = work_root / "release-debug" / tag / (item_name + ".debug")
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(symbol_file, sidecar)
        staged.replace(dest)
        (out / checksums).write_text(f"{sha}  {name}\n")
        return asset


def verify_runtime_tree(target: Path, asset: dict, manifest: dict,
                        *, metadata: tuple[str, ...] = ()) -> None:
    checks = manifest.get("files")
    if not isinstance(checks, dict):
        raise BundleError("runtime manifest has no file map")
    if manifest.get("payload_scope") != asset.get("payload_scope"):
        raise BundleError("runtime manifest/record mismatch: payload_scope")
    contract = manifest.get("serving_contract")
    if contract != asset.get("serving_contract"):
        raise BundleError("runtime manifest/record mismatch: serving_contract")
    if contract is not None and (
        not isinstance(contract, dict)
        or contract.get("engine_sha256") != checks.get("bin/radiance", {}).get("sha256")
        or contract.get("owner_sha256") != checks.get("radiance_home/bin/bc-serve", {}).get("sha256")
    ):
        raise BundleError("runtime serving contract does not bind both archived executables")
    if asset.get("built_from") is not None and manifest.get("built_from") != asset["built_from"]:
        raise BundleError("runtime manifest/record mismatch: built_from")
    if asset.get("payload_scope") == RUN_SCOPE:
        check_run_manifest(set(checks))
        if asset.get("file_names") != sorted(checks):
            raise BundleError("runtime receipt file set differs from manifest")
    for key in ("schema", "plan_id", "source_slice_id", "binary_sha256",
                "source_binary_sha256", "rocm_version", "engine_tree", "config_tree"):
        if manifest.get(key) != asset.get(key):
            raise BundleError(f"runtime manifest/record mismatch: {key}")
    actual: set[str] = set()
    for path in target.rglob("*"):
        if path.is_symlink():
            raise BundleError(f"runtime file symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(target).as_posix())
    if actual != set(checks) | set(metadata):
        raise BundleError("runtime file set differs from manifest")
    for name, receipt in checks.items():
        path = target.joinpath(*PurePosixPath(name).parts)
        if name in NATIVE_PROFILE_MEMBERS and receipt["mode"] != 0o644:
            raise BundleError(f"native profile must be non-executable: {name}")
        if (not path.is_file() or path.stat().st_size != receipt["size"]
                or digest(path) != receipt["sha256"]):
            raise BundleError(f"runtime file SHA-256 mismatch: {name}")
        if stat.S_IMODE(path.stat().st_mode) != receipt["mode"]:
            raise BundleError(f"runtime file mode mismatch: {name}")
        check_no_interpreter(name, path)


def checked_extract(archive: Path, asset: dict, target: Path) -> dict:
    if digest(archive) != asset["sha256"]:
        raise BundleError("runtime release archive SHA-256 does not match release record")
    target.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    manifest = None
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            rel = PurePosixPath(member.name)
            if (rel.is_absolute() or ".." in rel.parts or str(rel) != member.name
                or not member.isfile() or member.name in seen):
                raise BundleError(f"unsafe runtime archive member: {member.name}")
            seen.add(member.name)
            if len(seen) > FILE_LIMIT + 1:
                raise BundleError("too many runtime archive entries")
            total += member.size
            if total > BYTE_LIMIT:
                raise BundleError("runtime release exceeds size limit")
            if member.name == "bundle.json":
                if member.size > 2 * 1024 * 1024:
                    raise BundleError("runtime manifest is too large")
                with tar.extractfile(member) as stream:
                    content = stream.read()
                if hashlib.sha256(content).hexdigest() != asset["manifest_sha256"]:
                    raise BundleError("runtime manifest SHA-256 mismatch")
                try:
                    manifest = json.loads(content)
                except ValueError as exc:
                    raise BundleError("invalid runtime manifest") from exc
                continue
            dest = target.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, dest.open("wb") as output:
                shutil.copyfileobj(stream, output, 1024 * 1024)
            dest.chmod(0o755 if member.mode & 0o111 else 0o644)
    if not isinstance(manifest, dict):
        raise BundleError("runtime archive has no manifest")
    checks = manifest.get("files")
    if not isinstance(checks, dict) or set(checks) != seen - {"bundle.json"}:
        raise BundleError("runtime archive files differ from manifest")
    verify_runtime_tree(target, asset, manifest)
    (target / ".release-bundle.json").write_bytes(content)
    return manifest


def verify_installed_tree(target: Path, asset: dict, sdk: Path) -> None:
    stored = target / ".release-bundle.json"
    if not stored.is_file() or digest(stored) != asset.get("manifest_sha256"):
        raise BundleError("installed runtime manifest SHA-256 mismatch")
    try:
        manifest = json.loads(stored.read_bytes())
    except ValueError as exc:
        raise BundleError("invalid installed runtime manifest") from exc
    verify_runtime_tree(target, asset, manifest,
                        metadata=(".release-bundle.json", ".lab-source-tree"))
    if (target / ".lab-source-tree").read_bytes() != source_tree_marker(asset, sdk).encode():
        raise BundleError("installed source-tree marker differs from verified receipt and SDK")


def restore(archive: Path, checksum: Path, asset: dict, root: Path, work_root: Path,
            sdk: Path, *, tag: str | None = None) -> Path:
    if asset.get("payload_scope") is None:
        require_published_legacy_asset(asset, tag)
    if asset.get("schema") != SCHEMA or asset.get("payload_scope") not in (None, RUN_SCOPE):
        raise BundleError("unsupported runtime asset schema or scope")
    if not ARCHIVE_NAME.fullmatch(asset.get("name", "")):
        raise BundleError("unsafe runtime asset name")
    if asset.get("checksum_name") != asset["name"] + ".sha256":
        raise BundleError("unsafe checksum asset name")
    if asset.get("rocm_version") != sdk_version(sdk) or any(
        asset.get(k) != v for k, v in stock_identity(root).items()
    ):
        raise BundleError("current source trees or ROCm SDK differ from stock release bundle")
    if checksum.read_text().strip() != f"{asset['sha256']}  {asset['name']}":
        raise BundleError("release checksum file differs from release record")
    source = asset["source_slice_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", source):
        raise BundleError("unsafe source slice ID")
    if not re.fullmatch(r"[0-9a-f]{32}", asset["plan_id"]):
        raise BundleError("unsafe plan ID")
    builds = work_root / "builds" / source
    dest = builds / asset["plan_id"]
    if dest.exists():
        raise BundleError(f"build store target already exists; will not replace it: {dest}")
    builds.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="bu01-runtime-", dir=builds))
    try:
        checked_extract(archive, asset, stage)
        if digest(stage / "bin/radiance") != asset["binary_sha256"]:
            raise BundleError("release binary digest differs from recorded lane binary")
        # Historical pre-scope releases have no run-side membership contract.
        if asset.get("payload_scope") is None and not (stage / "radiance_home/helpers/r3_moe_worker").is_file():
            raise BundleError("legacy release missing remote MoE helper binary")
        (stage / ".lab-source-tree").write_text(source_tree_marker(asset, sdk))
        stage.rename(dest)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return dest


def download(root: Path, tag: str, patterns: tuple[str, ...], dest: Path) -> None:
    args = [arg for pattern in patterns for arg in ("-p", pattern)]
    result = subprocess.run(["gh", "release", "download", tag, *args, "-D", str(dest)],
                            cwd=root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise BundleError(f"release asset download failed: {result.stderr.strip()[-250:]}")


def release_asset_names(root: Path, tag: str) -> set[str] | None:
    """Names of the assets attached to a GitHub release, or None when it cannot be read."""
    result = subprocess.run(["gh", "release", "view", tag, "--json", "assets"],
                            cwd=root, capture_output=True, text=True, check=False)
    if result.returncode:
        return None
    return {entry["name"] for entry in json.loads(result.stdout).get("assets", [])}


def _ci_names(archive: str) -> tuple[str, str]:
    return archive + ".sha256", archive[: -len(".tar.gz")] + CI_RECEIPT_SUFFIX


def has_ci_bundle(names: set[str]) -> bool:
    archives = [n for n in names if CI_ARCHIVE.fullmatch(n)]
    return len(archives) == 1 and set(_ci_names(archives[0])) <= names


def tag_identity(root: Path, tag: str) -> dict[str, str]:
    """stock_identity of a tag, from git objects rather than any checkout."""
    def tree(path: str) -> str:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", f"refs/tags/{tag}:{path}"],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise BundleError(f"cannot read {path} tree at tag {tag} (fetch tags first)")
        return result.stdout.strip()
    return {"engine_tree": tree("engines/radiance"), "config_tree": tree("config")}


def require_product(what: str, document: object) -> None:
    """A CI release bundle says it is BigCherry's build; one that does not (or names another product) is refused."""
    product = document.get("product") if isinstance(document, dict) else None
    if product != PRODUCT:
        raise BundleError(f"{what} is not a {PRODUCT} bundle (product {product!r})")


def ci_asset(root: Path, tag: str, stage: Path) -> tuple[dict, Path, Path]:
    """Download a CI-made release's own bundle and return its receipt (the asset), archive and checksum.

    Checks here: the archive matches its .sha256 asset and the receipt's digest; the receipt
    is for this tag and its runtime_key matches its asset fields and the tag's source trees;
    bundle.json names this tag, or the reused_from tag whose source trees are the same. The
    caller checks the ROCm SDK and restores, which checks every file against bundle.json.
    """
    download(root, tag, (PRODUCT + "-*.tar.gz", PRODUCT + "-*.tar.gz.sha256",
                         PRODUCT + "-*" + CI_RECEIPT_SUFFIX), stage)
    archives = [p for p in stage.iterdir() if CI_ARCHIVE.fullmatch(p.name)]
    if len(archives) != 1:
        raise BundleError(f"expected one runtime archive in release {tag}, found {len(archives)}")
    archive = archives[0]
    checksum, receipt_path = (stage / n for n in _ci_names(archive.name))
    for path in (checksum, receipt_path):
        if not path.is_file():
            raise BundleError(f"release {tag} has no {path.name} asset")
    sha, _, name = checksum.read_text().strip().partition("  ")
    if name != archive.name or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise BundleError(f"malformed checksum asset {checksum.name}")
    if digest(archive) != sha:
        raise BundleError("runtime release archive SHA-256 does not match its .sha256 asset")
    try:
        asset = json.loads(receipt_path.read_text())
    except ValueError as exc:
        raise BundleError(f"invalid receipt asset {receipt_path.name}") from exc
    if not isinstance(asset, dict) or asset.get("name") != archive.name or asset.get("sha256") != sha \
            or asset.get("checksum_name") != checksum.name:
        raise BundleError("receipt asset does not describe the release archive")
    require_product("receipt asset", asset)
    if asset.get("release_tag") != tag:
        raise BundleError(f"receipt asset is for {asset.get('release_tag')}, not {tag}")
    key = asset.get("runtime_key")
    if not isinstance(key, dict) or any(key.get(k) != asset.get(k)
                                        for k in ("engine_tree", "config_tree", "rocm_version")):
        raise BundleError("receipt runtime_key differs from its asset fields")
    identity = tag_identity(root, tag)
    if any(key[k] != v for k, v in identity.items()):
        raise BundleError(f"bundle source trees differ from tag {tag}")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.next()
        if member is None or member.name != "bundle.json" or not member.isfile() \
                or member.size > 2 * 1024 * 1024:
            raise BundleError("runtime archive does not start with bundle.json")
        with tar.extractfile(member) as stream:
            manifest = json.loads(stream.read())
    require_product("bundle.json", manifest)
    built_for = manifest.get("release_tag") if isinstance(manifest, dict) else None
    if built_for != tag:
        reused = asset.get("reused_from")
        if reused is None or built_for != reused or tag_identity(root, reused) != identity:
            raise BundleError(f"bundle.json is for {built_for}, not {tag}, and is not a reuse of "
                              "a release with the same source trees")
    return asset, archive, checksum


def fetch(root: Path, work_root: Path, sdk: Path, tag: str) -> Path:
    """Download a release's CI-made bundle, verify it against its published receipt and restore it under the work root."""
    if not tag.startswith("bc-radiance-"):
        raise BundleError(f"not a bc-radiance release tag: {tag!r}")
    stage_dir = work_root / "release-assets" / "downloads"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="bu01-release-", dir=stage_dir))
    try:
        asset, archive, checksum = ci_asset(root, tag, stage)
        return restore(archive, checksum, asset, root, work_root, sdk, tag=tag)
    finally:
        shutil.rmtree(stage)


def latest_tag(root: Path) -> str:
    """The newest published bc-radiance release."""
    result = subprocess.run(["gh", "release", "list", "--limit", "50", "--json", "tagName,publishedAt"],
                            cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise BundleError(f"could not list releases: {result.stderr.strip()[-400:]}")
    tags = [r for r in json.loads(result.stdout) if r.get("tagName", "").startswith("bc-radiance-")]
    if not tags:
        raise BundleError("no bc-radiance release is published")
    return max(tags, key=lambda r: r.get("publishedAt") or "")["tagName"]
