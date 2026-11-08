"""BU01: deterministic, verified Radiance runtime release bundles.

Release records bind the asset to the pin-bump lane build, source trees and
ROCm version. Never install a release asset before verifying the record's
digest, every archived file's digest and current stock source identity.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

SCHEMA = "bigcherry-radiance-runtime-v1"
PLAN = re.compile(r"(?m)^(?:pin-bump lane build|release asset lane build): radiance:stock:linux-radiance build_plan_id=([0-9a-f]{32}) binary_sha256=([0-9a-f]{64})$")
FILE_LIMIT = 10000
BYTE_LIMIT = 16 * 1024**3
#: BigCherry's own build of radiance, patched and extended, is not the upstream release: its assets are named bigcherryrad. Bundles a
#: lab pin-release packed before this name existed are still called radiance-<plan>.tar.gz and still restore.
PRODUCT = "bigcherryrad"
CI_ARCHIVE = re.compile(r"bigcherryrad-[0-9a-f]{32}\.tar\.gz")
ARCHIVE_NAME = re.compile(r"(?:bigcherryrad|radiance)-[0-9a-f]{32}\.tar\.gz")
CI_RECEIPT_SUFFIX = ".receipt.json"
RUN_DIRS = ("bin", "radiance_home", "runtime-prefix/bin", "runtime-prefix/lib", "runtime-prefix/lib64")


class BundleError(RuntimeError):
    pass


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


def select_build(record: dict, work_root: Path) -> tuple[Path, str, str]:
    found = PLAN.findall(record.get("notes", ""))
    if not found:
        raise BundleError("release record has no pin-bump lane build plan ID and binary SHA-256")
    plan_id, binary_sha = found[-1]
    matches = list((work_root / "builds").glob(f"*/{plan_id}"))
    if len(matches) != 1 or not matches[0].is_dir():
        raise BundleError(f"expected one lane build for {plan_id}, found {len(matches)}")
    binary = matches[0] / "bin/radiance"
    if not binary.is_file() or digest(binary) != binary_sha:
        raise BundleError("lane build binary missing or SHA-256 differs from release record")
    return matches[0], plan_id, binary_sha


def built_from(root: Path) -> dict[str, str]:
    """What this build is made from: the upstream radiance release it patches, named so the bundle cannot pass for that release."""
    import tomllib
    try:
        engine = tomllib.loads((root / "engines/radiance/engine.toml").read_text(encoding="utf-8"))
        recipes = tomllib.loads((root / "engines/radiance/recipes.toml").read_text(encoding="utf-8"))
    except OSError:   # a checkout without the engine files (a unit-test fixture) has nothing to say
        engine, recipes = {}, {}
    return {"project": "radiance", "upstream": str(engine.get("upstream", "")),
            "pinned": str(recipes.get("pinned", "")), "note": "BigCherry's patched build, not an upstream release"}


def runtime_files(build: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for dirname in RUN_DIRS:
        base = build / dirname
        if not base.is_dir():
            continue
        for item in sorted(base.rglob("*")):
            if not item.is_file():
                continue
            # Dereference only symlinks pointing to another file INSIDE this build.
            if item.is_symlink() and not item.resolve().is_relative_to(build.resolve()):
                raise BundleError(f"runtime symlink escapes build directory: {item}")
            rel = item.relative_to(build).as_posix()
            files[rel] = item
    for required in ("bin/radiance", "radiance_home/helpers/r3_moe_worker"):
        if required not in files:
            raise BundleError(f"lane build does not contain runtime requirement {required}")
    if not any(k.startswith("radiance_home/kernels/") and k.endswith(".so") for k in files):
        raise BundleError("lane build has no Radiance kernel plugins")
    if len(files) > FILE_LIMIT:
        raise BundleError("too many files in runtime bundle")
    if sum(p.stat().st_size for p in files.values()) > BYTE_LIMIT:
        raise BundleError("runtime bundle exceeds size limit")
    return files


def pack(record: dict, root: Path, work_root: Path, sdk: Path, tag: str) -> dict:
    if not re.fullmatch(r"bc-radiance-[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise BundleError("unsafe Radiance release tag")
    build, plan_id, binary_sha = select_build(record, work_root)
    files = runtime_files(build)
    identity = stock_identity(root)
    version = sdk_version(sdk)
    manifest = {
        "schema": SCHEMA, "product": PRODUCT, "built_from": built_from(root), "release_tag": tag,
        "plan_id": plan_id, "source_slice_id": build.parent.name,
        "binary_sha256": binary_sha,
        "rocm_version": version, **identity,
        "files": {
            name: {"sha256": digest(path), "size": path.stat().st_size,
                   "mode": 0o755 if os.access(path, os.X_OK) else 0o644}
            for name, path in sorted(files.items())
        },
    }
    encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    out = work_root / "release-assets" / tag
    out.mkdir(parents=True, exist_ok=True)
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
    asset = {"schema": SCHEMA, "product": PRODUCT, "name": name, "sha256": sha,
             "checksum_name": checksums, "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
             "plan_id": plan_id, "source_slice_id": build.parent.name,
             "rocm_version": version, "binary_sha256": binary_sha, **identity}
    old = record.get("runtime_assets", {}).get(tag)
    if old is not None and old != asset:
        staged.unlink()
        raise BundleError("release record asset changed; refusing to replace existing verified digest")
    staged.replace(dest)
    (out / checksums).write_text(f"{sha}  {name}\n")
    return asset


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
    for name, receipt in checks.items():
        path = target.joinpath(*PurePosixPath(name).parts)
        if (not path.is_file() or path.stat().st_size != receipt["size"]
            or digest(path) != receipt["sha256"]):
            raise BundleError(f"runtime file SHA-256 mismatch: {name}")
    for key in ("schema", "plan_id", "source_slice_id", "binary_sha256",
                "rocm_version", "engine_tree", "config_tree"):
        if manifest.get(key) != asset.get(key):
            raise BundleError(f"runtime manifest/record mismatch: {key}")
    return manifest


def restore(archive: Path, checksum: Path, asset: dict, root: Path, work_root: Path,
            sdk: Path) -> Path:
    if asset.get("schema") != SCHEMA:
        raise BundleError("unsupported runtime asset schema")
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
        if not (stage / "radiance_home/helpers/r3_moe_worker").is_file():
            raise BundleError("release missing remote MoE helper binary")
        tree = stock_identity(root)
        (stage / ".lab-source-tree").write_text(
            f"{tree['engine_tree']} {tree['config_tree']} {sdk.resolve()} {sdk_version(sdk)} ")
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
    built_for = manifest.get("release_tag") if isinstance(manifest, dict) else None
    if built_for != tag:
        reused = asset.get("reused_from")
        if reused is None or built_for != reused or tag_identity(root, reused) != identity:
            raise BundleError(f"bundle.json is for {built_for}, not {tag}, and is not a reuse of "
                              "a release with the same source trees")
    return asset, archive, checksum


def fetch(record: dict, root: Path, work_root: Path, sdk: Path,
          tag: str | None = None) -> Path:
    assets = record.get("runtime_assets", {})
    if tag is None:
        released = re.findall(r"release (\d+\.\d+\.\d+)", record.get("notes", ""))
        if not released:
            raise BundleError("release record has no completed release to fetch")
        tag = "bc-radiance-" + released[-1]
    asset = assets.get(tag)
    if not isinstance(asset, dict):
        raise BundleError("release record has no prebuilt runtime asset for selected release")
    if not isinstance(tag, str) or not tag.startswith("bc-radiance-"):
        raise BundleError("release tag missing from record")
    # Validate names before passing record-derived patterns to gh download.
    if (asset.get("schema") != SCHEMA
        or not ARCHIVE_NAME.fullmatch(asset.get("name", ""))
        or asset.get("checksum_name") != asset["name"] + ".sha256"
        or not re.fullmatch(r"[0-9a-f]{64}", asset.get("sha256", ""))):
        raise BundleError("unsafe release asset receipt")
    stage_dir = work_root / "release-assets" / "downloads"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="bu01-release-", dir=stage_dir))
    try:
        download(root, tag, (asset["name"], asset["checksum_name"]), stage)
        return restore(stage / asset["name"], stage / asset["checksum_name"],
                       asset, root, work_root, sdk)
    finally:
        shutil.rmtree(stage)


def publish(record: dict, root: Path, work_root: Path, tag: str) -> None:
    asset = record.get("runtime_assets", {}).get(tag)
    if not isinstance(asset, dict):
        raise BundleError("release record lacks verified runtime asset for this release")
    files = work_root / "release-assets" / tag
    archive = files / asset["name"]
    checksums = files / asset["checksum_name"]
    if not archive.is_file() or digest(archive) != asset["sha256"]:
        raise BundleError("locally packaged runtime asset missing or changed after record commit")
    if not checksums.is_file() or checksums.read_text().strip() != f"{asset['sha256']}  {asset['name']}":
        raise BundleError("runtime checksum file missing or changed")
    result = subprocess.run(["gh", "release", "view", tag, "--json", "assets"],
                            cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise BundleError(f"could not inspect release assets: {result.stderr.strip()[-400:]}")
    existing = {entry["name"] for entry in json.loads(result.stdout).get("assets", [])}
    names = {asset["name"], asset["checksum_name"]}
    if existing & names:
        if not names.issubset(existing):
            raise BundleError("release already has only part of the runtime bundle; resolve manually")
        stage = Path(tempfile.mkdtemp(prefix="bu01-verify-upload-", dir=files))
        try:
            result = subprocess.run(["gh", "release", "download", tag,
                                     "-p", asset["name"], "-p", asset["checksum_name"],
                                     "-D", str(stage)],
                                    cwd=root, capture_output=True, text=True)
            if result.returncode or digest(stage / asset["name"]) != asset["sha256"]:
                raise BundleError("existing release asset digest differs from record")
            if (stage / asset["checksum_name"]).read_text() != checksums.read_text():
                raise BundleError("existing release checksum differs from record")
            return
        finally:
            shutil.rmtree(stage)
    result = subprocess.run(["gh", "release", "upload", tag, str(archive), str(checksums)],
                            cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise BundleError(f"gh release upload failed: {result.stderr.strip()[-400:]}")
