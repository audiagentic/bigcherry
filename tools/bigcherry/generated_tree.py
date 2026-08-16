"""RE14: hash-verified identity for autotune_catalog.emit()'s generated tree.

The files ``emit()`` writes into a build's ``generated_root`` (registry.inc,
build-hash.h, arch.h, the manifest/descriptor JSON, the MMVQ instance
include) were, until now, an unverified side-channel in the campaign build
path: the generate stage wrote them to a run-scoped filesystem location and
the build stage read them back by ``run_id`` convention, with no
``ArtifactRef``/``PipelineService`` boundary checking that the bytes CMake
actually compiles against are the same bytes generate produced. This module
closes that gap: a manifest of every generated file's SHA-256 is built at
generate time, published like any other artifact, and re-verified by the
build stage immediately before invoking cmake.

Two identities are recorded, not one, because they answer different
questions. ``files`` (every generated file's hash) answers "has anything in
this directory changed since generate ran" -- the full integrity check.
``compile_inputs_hash`` (a hash of only the subset of files a compile
actually reads) answers "would two otherwise-identical generation runs
produce a build that could reuse each other's binary" -- and it must
deliberately exclude the raw manifest JSON, which embeds a real-time
``generated_at`` that ``manifest_hash`` itself already excludes for exactly
this reason. Folding the manifest's timestamp into a build-reuse identity
would make two runs with an identical candidate set permanently unable to
reuse each other's binary, for a reason that carries no real difference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class GeneratedTreeError(ValueError):
    pass


SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GeneratedTreeError(f"cannot hash generated file {path}: {exc}") from exc
    return digest.hexdigest()


def _relative_file_set(root: Path) -> dict[str, Path]:
    root = root.resolve()
    if not root.is_dir():
        raise GeneratedTreeError(f"generated root does not exist: {root}")

    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise GeneratedTreeError(f"generated tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise GeneratedTreeError(f"generated tree contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        result[relative] = path
    return result


def _compile_inputs_hash(files: dict[str, str], compile_inputs: tuple[str, ...]) -> str:
    missing = [name for name in compile_inputs if name not in files]
    if missing:
        raise GeneratedTreeError(
            f"declared compile input(s) not found in generated tree: {missing}"
        )
    payload = {name: files[name] for name in compile_inputs}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"bigcherry/generated-compile-inputs/v1\0" + encoded).hexdigest()


def build_manifest(
    generated_root: Path, *, compile_inputs: tuple[Path, ...],
) -> dict[str, Any]:
    """Hash every file under ``generated_root`` and the compile-input subset.

    ``compile_inputs`` are absolute paths (as ``EmitResult.compile_input_paths``
    provides); they are resolved relative to ``generated_root`` here so the
    published manifest is portable (relative paths, not this host's
    absolute layout).
    """
    root = generated_root.resolve()
    files = _relative_file_set(root)
    relative_compile_inputs = tuple(
        sorted(path.resolve().relative_to(root).as_posix() for path in compile_inputs)
    )
    file_hashes = {name: file_sha256(path) for name, path in files.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "files": file_hashes,
        "compile_inputs": list(relative_compile_inputs),
        "compile_inputs_hash": _compile_inputs_hash(file_hashes, relative_compile_inputs),
    }


def verify_tree(generated_root: Path, tree_document: dict[str, Any]) -> None:
    """Re-hash ``generated_root`` and require an EXACT match against
    ``tree_document`` -- not just the compile inputs. A file appearing that
    wasn't there at generate time, a file disappearing, or any file's bytes
    differing are all real integrity failures: the whole point of this
    check is that nothing modifies the generated tree between generate
    publishing its manifest and build actually compiling against it.
    """
    if tree_document.get("schema_version") != SCHEMA_VERSION:
        raise GeneratedTreeError(
            f"generated tree manifest has unknown schema_version "
            f"{tree_document.get('schema_version')!r}"
        )
    recorded = tree_document.get("files")
    if not isinstance(recorded, dict):
        raise GeneratedTreeError("generated tree manifest has no 'files' mapping")

    root = generated_root.resolve()
    actual = _relative_file_set(root)

    missing = sorted(set(recorded) - set(actual))
    extra = sorted(set(actual) - set(recorded))
    if missing or extra:
        raise GeneratedTreeError(
            f"generated tree does not match its manifest: "
            f"missing={missing}, unexpected={extra}"
        )

    mismatched = [
        name for name, expected_hash in recorded.items()
        if file_sha256(actual[name]) != expected_hash
    ]
    if mismatched:
        raise GeneratedTreeError(
            f"generated file(s) modified since the tree manifest was published: {mismatched}"
        )
