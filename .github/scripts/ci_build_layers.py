#!/usr/bin/env python3
"""BU03: trusted, content-keyed Radiance CI engine and component build layers.

Only a successful push to main may *publish* layers. PRs restore them, validate
both the fingerprint and archive digest, and let the production build worker
validate its own source/build/binary metadata before trusting a complete hit.
A missing component layer invalidates cached build metadata, so the real CMake
build runs and rebuilds changed components; it never serves stale plugins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import sys
import tarfile
import tomllib

SCHEMA = "bigcherry-ci-layers-v1"
COMPONENTS = ("libr3", "libr2", "moe-remote", "ngram-helper", "bc-gfx12", "serving")
MAX_FILES = 300_000
MAX_BYTES = 32 * 1024**3


class LayerError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths(root: Path) -> list[str]:
    output = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return sorted(p.decode("utf-8") for p in output.split(b"\0") if p)


def fingerprint_files(root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"bigcherry-ci-layer-v1\0")
    for rel in paths:
        path = root / rel
        if not path.is_file():
            raise LayerError(f"tracked build input missing: {rel}")
        contents = path.read_bytes()
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def input_keys(root: Path) -> dict[str, str]:
    declared = tomllib.loads((root / "engines/radiance/engine.toml").read_text()).get("components", {})
    if not isinstance(declared, dict) or set(declared) != set(COMPONENTS):
        raise LayerError("Radiance component declarations changed; update the layer cache mapping")
    files = tracked_paths(root)
    # Include all code that can affect source materialisation, CMake options,
    # build execution and toolchain, but not independently built components.
    engine_files = [
        p for p in files
        if (
            (p.startswith("engines/radiance/") and not p.startswith("engines/radiance/components/")
             and not p.endswith((".md", ".mdx")))
            or p.startswith(("tools/bigcherry/", ".github/docker/radiance-build/"))
            or p in (
                "config/recipes.toml", "config/environment.example.toml",
                ".github/workflows/radiance-build.yml",
                ".github/workflows/radiance-build-image.yml",
                ".github/scripts/ci_build_layers.py",
            )
        )
    ]
    engine = fingerprint_files(root, engine_files)
    # The Dockerfile tag is a recipe, not a cryptographic image identity.
    # Bind restored binaries to the exact local compiler image when CI runs.
    image_id = os.environ.get("CI_LAYER_IMAGE_ID", "")
    if image_id:
        if not image_id.startswith("sha256:") or len(image_id) != 71:
            raise LayerError("invalid CI compiler image digest")
        engine = hashlib.sha256((engine + ":" + image_id).encode()).hexdigest()
    keys = {"engine": engine}
    # Headers and CMake include files may be consumed across components.
    # A shared interface edit invalidates dependent component layers, while
    # a change confined to one component implementation remains local.
    shared_files = [
        p for p in files if p.startswith("engines/radiance/components/")
        and p.endswith((".h", ".hpp", ".hh", ".hiph", ".cuh", ".inc", ".cmake"))
    ]
    for name in COMPONENTS:
        component_files = sorted(set(shared_files) | {
            p for p in files if p.startswith(f"engines/radiance/components/{name}/")
        })
        if not component_files:
            raise LayerError(f"component {name} has no tracked input files")
        digest = fingerprint_files(root, component_files)
        keys[name] = hashlib.sha256(f"{engine}:{name}:{digest}".encode()).hexdigest()
    return keys


def output_name(name: str) -> str:
    return name.replace("-", "_")


def build_directory(builds: Path) -> Path:
    matches = [
        path for path in builds.glob("*/*")
        if path.is_dir() and (path / "CMakeCache.txt").is_file()
        and (path / "bin/radiance").is_file()
    ]
    if len(matches) != 1:
        raise LayerError(f"expected one completed Radiance build in {builds}, found {len(matches)}")
    return matches[0]


def members_for(build: Path, layer: str) -> list[Path]:
    if layer == "engine":
        return sorted(
            path for path in build.rglob("*")
            if path.is_file() or path.is_symlink()
            if not path.relative_to(build).parts[0] == "components"
        )
    component_dir = build / "components" / layer
    if not component_dir.is_dir():
        raise LayerError(f"component build directory missing: {layer}")
    return sorted(p for p in component_dir.rglob("*") if p.is_file() or p.is_symlink())


def export_layers(root: Path, builds: Path, cache: Path) -> None:
    keys = input_keys(root)
    build = build_directory(builds)
    rel_build = build.relative_to(builds).as_posix()
    for name, key in keys.items():
        target = cache / output_name(name)
        manifest = target / "manifest.json"
        archive = target / "layer.tar.gz"
        if manifest.is_file() and archive.is_file():
            # An exact cache hit was restored; immutable GitHub cache need not
            # (and cannot) be republished under the same key.
            validate_manifest(target, name, key)
            continue
        files = members_for(build, name)
        if not files:
            raise LayerError(f"layer {name} has no build outputs")
        target.mkdir(parents=True, exist_ok=True)
        staged = target / "layer.tar.gz.pending"
        with tarfile.open(staged, "w:gz", compresslevel=1) as tar:
            for path in files:
                rel = f"{rel_build}/{path.relative_to(build).as_posix()}"
                tar.add(path, arcname=rel, recursive=False)
        staged.replace(archive)
        document = {
            "schema": SCHEMA, "name": name, "key": key,
            "build": rel_build, "sha256": sha256(archive),
            "files": len(files), "archive_bytes": archive.stat().st_size,
        }
        manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        print(f"exported {name}: {len(files)} files, {archive.stat().st_size} bytes")


def validate_manifest(folder: Path, name: str, key: str) -> dict:
    doc = json.loads((folder / "manifest.json").read_text())
    if doc.get("schema") != SCHEMA or doc.get("name") != name or doc.get("key") != key:
        raise LayerError(f"{name}: cached layer fingerprint or schema disagrees with source")
    path = PurePosixPath(doc.get("build", ""))
    if len(path.parts) != 2 or any(not p or p in (".", "..") for p in path.parts):
        raise LayerError(f"{name}: invalid cached build identity")
    if not all(len(p) == 32 and all(c in "0123456789abcdef" for c in p) for p in path.parts):
        raise LayerError(f"{name}: invalid content-addressed build directory")
    if sha256(folder / "layer.tar.gz") != doc.get("sha256"):
        raise LayerError(f"{name}: prebuilt archive digest mismatch")
    return doc


def _validate_archive(tar: tarfile.TarFile, rel_build: str, name: str) -> None:
    total, count = 0, 0
    prefix = rel_build + ("/components/" + name + "/" if name != "engine" else "/")
    for member in tar.getmembers():
        count += 1
        total += member.size
        if count > MAX_FILES or total > MAX_BYTES:
            raise LayerError(f"{name}: oversized archive")
        path = PurePosixPath(member.name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != member.name
            or not member.name.startswith(prefix)):
            raise LayerError(f"{name}: unsafe archive member {member.name!r}")
        if not (member.isfile() or member.issym()):
            raise LayerError(f"{name}: unsupported archive member type {member.name!r}")
        if member.issym():
            if posixpath.isabs(member.linkname):
                raise LayerError(f"{name}: absolute symlink in cache")
            dest = posixpath.normpath(posixpath.join(posixpath.dirname(member.name), member.linkname))
            if not dest.startswith(rel_build + "/"):
                raise LayerError(f"{name}: escaping symlink in cache")


def restore_layers(root: Path, builds: Path, cache: Path) -> None:
    keys = input_keys(root)
    first = cache / "engine"
    if not (first / "manifest.json").is_file():
        print("prebuilt engine: cache miss; normal build required")
        return
    base_doc = validate_manifest(first, "engine", keys["engine"])
    selected: list[tuple[str, Path, dict]] = [("engine", first, base_doc)]
    missing: list[str] = []
    for name in COMPONENTS:
        folder = cache / output_name(name)
        if not (folder / "manifest.json").is_file():
            missing.append(name)
            continue
        doc = validate_manifest(folder, name, keys[name])
        if doc["build"] != base_doc["build"]:
            raise LayerError(f"{name}: layer build identity differs from engine")
        selected.append((name, folder, doc))
    for name, folder, doc in selected:
        with tarfile.open(folder / "layer.tar.gz", "r:gz") as tar:
            _validate_archive(tar, doc["build"], name)
            # tar members are verified content-hashed above and restricted to
            # the content-addressed build directory; no arbitrary paths.
            tar.extractall(builds, filter="data")
        print(f"restored prebuilt {name} layer: {doc['files']} files")
    if missing:
        build = builds / base_doc["build"]
        for metadata in build.glob("bigcherry-build-metadata-*.json"):
            metadata.unlink()
        print(f"missing component layer(s): {', '.join(missing)}; forcing real build")
    else:
        print("all engine/component layers restored; build worker validates binary and source identity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("keys", "export", "restore"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--builds", type=Path, default=Path("work/builds"))
    parser.add_argument("--cache", type=Path, default=Path("work/ci-prebuilt"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "keys":
            keys = input_keys(root)
            text = "".join(f"{output_name(name)}={key}\n" for name, key in keys.items())
            if args.output:
                with args.output.open("a") as stream:
                    stream.write(text)
            else:
                print(text, end="")
        elif args.action == "export":
            export_layers(root, args.builds, args.cache)
        else:
            restore_layers(root, args.builds, args.cache)
    except (LayerError, OSError, ValueError, tarfile.TarError) as exc:
        print(f"ci-build-layers: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
