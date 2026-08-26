"""HI82: content-addressed source isolation for patch-validation campaigns.

Fixes a real contamination risk found this session: patch_validation_
campaign.py's --build-root reuses ONE mutable vendor/llama.cpp git checkout
across sequential patch applications with no provenance check. Applying
patch A then patch B to the same tree silently composes both patches into
one build rather than testing B in isolation -- this session did exactly
that (1204 RD08 then 1002 unsafe-math, then 1005, then 1205/1206) and only
avoided a real false-attribution result because none of those patches
happened to share touched files. That must not be the ongoing pattern.

Design: GPT (gpt-auto-agent), requests req_51838ef1ea5f4086 (design +
git_worktree_tree/stable_digest/source_identity sketch),
req_2c85f337296e4c30 and req_17ab02fd3af2449a (implementation, both
truncated mid-response by the chat interface -- the identity/key/path
helpers below are GPT's text verbatim; materialize_source/
materialize_stock_source are written directly against GPT's documented
contract from the design review rather than waiting on a third truncated
attempt). Applied per plan item HI82.

Each materialized source lives at <worktree_root>/<source_key>/, a real
`git worktree add` checkout (never the shared vendor/llama.cpp working
tree), with a manifest.json recording the git tree hash the CONTENT was
at when this module last touched it. Reuse is only permitted when the
on-disk tree hash still matches the manifest -- any mutation after the
fact (by this module's own bug, an operator, or another process) is
detected and rejected rather than silently trusted.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import Any

from bigcherry.core import paths
from bigcherry.patch.apply import (
    PATCH_APPLICATION_SEMANTICS_VERSION,
    resolve_contained_target,
)

REPO_ROOT = paths.REPO_ROOT
PATCHES_ROOT = REPO_ROOT / "patches"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
SOURCE_TRANSFORM_SCHEMA_VERSION = 1
SOURCE_TRANSFORM_SEMANTICS_VERSION = 1


class PatchSourceIsolationError(RuntimeError):
    """Raised when a patch-validation source cannot be materialized safely."""


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _code_definition(code: CodeType) -> dict[str, Any]:
    """Return a location-independent, JSON-serializable code definition."""

    def constant(value: Any) -> Any:
        if isinstance(value, CodeType):
            return {"code": _code_definition(value)}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if value is Ellipsis:
            return {"ellipsis": True}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}

    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "constants": [constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _fingerprint_value(value: Any, seen: set[int] | None = None) -> Any:
    """Serialize callable dependencies that can affect transform behavior."""

    seen = set() if seen is None else seen
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, Path):
        return {"path": value.as_posix()}
    if isinstance(value, CodeType):
        return {"code": _code_definition(value)}
    identity = id(value)
    if identity in seen:
        return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
    seen.add(identity)
    try:
        if isinstance(value, tuple):
            return {"tuple": [_fingerprint_value(item, seen) for item in value]}
        if isinstance(value, list):
            return {"list": [_fingerprint_value(item, seen) for item in value]}
        if isinstance(value, dict):
            entries = [
                (_fingerprint_value(key, seen), _fingerprint_value(item, seen))
                for key, item in value.items()
            ]
            return {"dict": sorted(entries, key=lambda entry: repr(entry[0]))}
        if isinstance(value, (set, frozenset)):
            return {
                type(value).__name__: sorted(
                    (_fingerprint_value(item, seen) for item in value), key=repr
                )
            }
        if inspect.ismodule(value):
            return {"module": value.__name__}
        if inspect.isfunction(value) or inspect.ismethod(value):
            function = value.__func__ if inspect.ismethod(value) else value
            return {
                "function": f"{function.__module__}.{function.__qualname__}",
                "code": _code_definition(function.__code__),
            }
        if isinstance(value, type):
            return {"type": f"{value.__module__}.{value.__qualname__}"}
        if hasattr(value, "__dict__"):
            return {
                "object": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _fingerprint_value(vars(value), seen),
            }
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    finally:
        seen.discard(identity)


def _callable_implementation_digest(apply_variant: Callable[[Path], None]) -> str:
    """Digest callable code, its referenced globals, and transform semantics.

    The legacy API accepted a separate digest string. This fingerprint is
    derived from the callable itself instead, including module bytes so edits
    to referenced module-level transform definitions cannot retain a stale
    source identity.
    """

    function = apply_variant
    if not inspect.isfunction(function) and not inspect.ismethod(function):
        function = getattr(function, "__call__", None)
    code = getattr(function, "__code__", None)
    globals_dict = getattr(function, "__globals__", {})
    if code is None or not callable(function):
        raise PatchSourceIsolationError(
            "variant transform must be a Python callable with inspectable code"
        )
    module_name = getattr(function, "__module__", None)
    module_file_digest = None
    module = inspect.getmodule(function)
    if module is not None:
        module_file = getattr(module, "__file__", None)
        if module_file:
            try:
                module_file_digest = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()
            except OSError as exc:
                raise PatchSourceIsolationError(
                    f"cannot read variant implementation module {module_file!r}"
                ) from exc
    referenced_globals = {
        name: _fingerprint_value(globals_dict[name])
        for name in sorted(code.co_names)
        if name in globals_dict
    }
    closure = getattr(function, "__closure__", None)
    closure_values = (
        [_fingerprint_value(cell.cell_contents) for cell in closure]
        if closure is not None
        else []
    )
    payload = {
        "semantics_version": SOURCE_TRANSFORM_SEMANTICS_VERSION,
        "module": module_name,
        "qualname": getattr(function, "__qualname__", None),
        "module_bytes": module_file_digest,
        "code": _code_definition(code),
        "globals": referenced_globals,
        "closure": closure_values,
    }
    return _stable_digest(payload)


@dataclass(frozen=True)
class SourceTransform:
    """Immutable, content-addressed definition of a source variant.

    Structured operations use ``("replace", relative_path, old, new)`` and
    are applied with exactly-one-match semantics. ``from_callable`` is the
    compatibility bridge for existing registered/importable Python transforms;
    its operation contains a derived implementation fingerprint, never a
    caller-provided digest.
    """

    name: str
    schema_version: int
    operations: tuple[Any, ...]
    _apply_variant: Callable[[Path], None] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("source transform name must not be empty")
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise TypeError("source transform schema_version must be an integer")
        if not isinstance(self.operations, tuple):
            raise TypeError("source transform operations must be a tuple")
        try:
            _stable_digest(self.definition)
        except (TypeError, ValueError) as exc:
            raise TypeError("source transform operations must be JSON-serializable") from exc

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "semantics_version": SOURCE_TRANSFORM_SEMANTICS_VERSION,
            "operations": self.operations,
        }

    @property
    def digest(self) -> str:
        return _stable_digest(self.definition)

    @classmethod
    def from_callable(
        cls, name: str, apply_variant: Callable[[Path], None], *, schema_version: int = 1
    ) -> "SourceTransform":
        implementation_digest = _callable_implementation_digest(apply_variant)
        return cls(
            name=name,
            schema_version=schema_version,
            operations=(("callable", implementation_digest),),
            _apply_variant=apply_variant,
        )

    def apply(self, source_dir: Path) -> None:
        if self._apply_variant is not None:
            self._apply_variant(source_dir)
            return
        for operation in self.operations:
            if not isinstance(operation, (tuple, list)) or len(operation) != 4:
                raise PatchSourceIsolationError(
                    f"unsupported source transform operation: {operation!r}"
                )
            kind, relative, old, new = operation
            if kind != "replace" or not all(isinstance(value, str) for value in (relative, old, new)):
                raise PatchSourceIsolationError(
                    f"unsupported source transform operation: {operation!r}"
                )
            try:
                target = resolve_contained_target(source_dir, relative)
            except Exception as exc:
                raise PatchSourceIsolationError(str(exc)) from exc
            text = target.read_text(encoding="utf-8")
            matches = text.count(old)
            if matches != 1:
                raise PatchSourceIsolationError(
                    f"source transform expected exactly one {old!r} match in {relative!r}, found {matches}"
                )
            target.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")


def _patch_registry():
    """The registry module, importable whether this file is loaded as a
    package member or standalone (its existing pattern: REPO_ROOT-derived
    paths + function-level import)."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry.patch import registry as patch_registry # noqa: E402

    return patch_registry


def resolve_base_revision(ref: str, *, repo: Path) -> str:
    """Resolve a git ref (branch/tag/HEAD/full-SHA) to the IMMUTABLE commit SHA
    it points at (peels annotated tags via ``^{commit}``). Fails closed on any
    unresolvable ref. The RESOLVED SHA -- not the requested ref -- is what
    enters the source identity and what ``git worktree add --detach`` receives,
    so a moved ref yields a new identity, never a reused stale worktree.
    (RV80/B2: replaces the implicit ``git_head`` + state-scan baseline.)"""
    return _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repo)


def _overlay_files(overlay_root: Path | None) -> list[tuple[str, str]]:
    """Deterministic ``(relpath, sha256)`` enumeration of every file under the
    source overlay. Shared by :func:`overlay_digest` and
    :func:`_apply_composition` so the EXACT bytes that enter the identity are
    the EXACT bytes applied. POSIX relative paths, sorted. A missing/absent
    overlay (or ``None``) yields ``[]`` (the empty digest)."""
    if overlay_root is None or not overlay_root.is_dir():
        return []
    entries = []
    for source in sorted(overlay_root.rglob("*")):
        if not source.is_file():
            continue
        relpath = source.relative_to(overlay_root).as_posix()
        entries.append((relpath, hashlib.sha256(source.read_bytes()).hexdigest()))
    return sorted(entries)


def overlay_digest(overlay_root: Path | None) -> str:
    """sha256 over the sorted ``(relpath, sha256)`` of every file under the
    source overlay (bigcherry's ``src/`` additions). State-independent and
    deterministic. ``overlay_root=None`` (stock) hashes the empty set."""
    encoded = json.dumps(_overlay_files(overlay_root), separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def composition_digest(composition: Sequence[tuple[str, str]]) -> str:
    """sha256 over the EXACT ORDERED ``[(patch_id, implementation_digest)]``
    list as supplied (which must be the topologically-ordered output of
    ``patchset.resolve_exact``). We do NOT re-sort: a lexicographic sort could
    give two DIFFERENT application orders the same key when a packaged
    ``patch.toml`` changes requires/order with unchanged ``patch.py`` digests.
    The empty composition (stock) hashes the empty list."""
    encoded = json.dumps(
        [list(entry) for entry in composition],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_source_identity_v2(
    *,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    overlay_root: Path | None,
    variant_transform: SourceTransform | None = None,
) -> dict[str, Any]:
    """``bigcherry-patch-source-v2`` (variants: ``-variant-v2``). The payload is
    STATE-INDEPENDENT (RV80/B2): no lifecycle-state scan -- only (a) the
    immutable resolved base SHA, (b) the overlay-bytes digest, (c) the EXPLICIT
    ordered composition, and (d) for variants, the content-addressed transform.
    The requested ref is recorded in the manifest separately and is NOT part of
    the hashed payload."""
    schema = (
        "bigcherry-patch-source-variant-v2"
        if variant_transform is not None
        else "bigcherry-patch-source-v2"
    )
    payload = {
        "schema": schema,
        "resolved_revision": resolved_revision,
        "overlay_digest": overlay_digest(overlay_root),
        "composition": [list(entry) for entry in composition],
        # PA07 (L1.2): a patch's own digest cannot see a change to the
        # SHARED application semantics (anchor matching, noise stripping,
        # guard handling, ...) that also determines its output bytes.
        "patch_application_semantics_version": PATCH_APPLICATION_SEMANTICS_VERSION,
    }
    if variant_transform is not None:
        payload["variant_name"] = variant_transform.name
        # Manifest JSON normalizes tuples to lists; normalize the identity
        # payload the same way so reuse compares the exact persisted shape.
        payload["variant_transform"] = json.loads(
            json.dumps(variant_transform.definition, ensure_ascii=False)
        )
        payload["variant_digest"] = variant_transform.digest
    return {**payload, "source_key": _stable_digest(payload)}


def patch_implementation_digest(patch_name: str) -> str:
    """sha256 of the patch implementation's own bytes (flat OR packaged).

    RS05: resolved through the registry descriptor -- no flat-only
    f"{name}.py" construction left in this module (the old _patch_path() is
    gone; the registry's own discovery is the path-resolution authority and
    enforces the same escape rules).

    Deliberately does NOT also hash tools/bigcherry/patcher.py (the shared
    edit engine) -- that would force a rebuild on any unrelated patcher.py
    docstring tweak. A stricter version could add it if patcher.py's own
    edit semantics start changing in ways that matter; note the tradeoff
    rather than silently picking one side of it.
    """
    registry = _patch_registry().load_registry(PATCHES_ROOT)
    try:
        descriptor = registry.get(patch_name)
    except Exception as exc:
        raise PatchSourceIsolationError(
            f"patch module does not exist: {patch_name!r}"
        ) from exc
    return hashlib.sha256(
        (registry.root / descriptor.implementation_path).read_bytes()
    ).hexdigest()


def _run(argv: list[str], *, cwd: Path, check: bool = True) -> str:
    result = subprocess.run(argv, cwd=cwd, check=check, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise PatchSourceIsolationError(
            f"{' '.join(argv)} (cwd={cwd}) failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_head(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo)


def git_head_tree(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo)


def git_worktree_tree(repo: Path) -> str:
    """The real git tree hash of the current working-tree CONTENT.

    Uses a temporary index (GIT_INDEX_FILE) so this never stages or
    modifies the repo's own index -- `git add -A` here only affects the
    throwaway index, not the operator's real staging area. Captures
    tracked modifications, deletions, and untracked non-ignored additions
    (exactly what a patch's edits + any new files it adds would produce).
    Untracked materialization files are included. Ignored files are rejected
    by the canonical source-identity implementation instead of being silently
    excluded from the tree.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry.source.identity import SourceIdentityError, git_tree_oid  # noqa: E402

    status = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo
    )
    allowed_untracked = {
        line[3:].strip().replace("\\", "/")
        for line in status.splitlines()
        if line.startswith("?? ")
    }
    try:
        return git_tree_oid(repo, allowed_untracked=allowed_untracked)
    except SourceIdentityError as exc:
        raise PatchSourceIsolationError(str(exc)) from exc


def _git_object_format(repo: Path) -> str:
    return _run(["git", "rev-parse", "--show-object-format"], cwd=repo)


def _source_slice_id(*, source_dir: Path, upstream_revision: str, tree_oid: str) -> str:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry.source.identity import source_slice_id  # noqa: E402

    return source_slice_id(
        upstream_revision=upstream_revision,
        tree_oid=tree_oid,
        object_format=_git_object_format(source_dir),
    )


def _manifest_path(source_dir: Path) -> Path:
    # A SIBLING of the worktree, not inside it: writing manifest.json into
    # the worktree itself would make it show up as an untracked file the
    # very next time git_worktree_tree() runs, changing the computed hash
    # out from under the manifest that just recorded it (confirmed by a
    # real reuse-path test failing this way before this fix).
    return source_dir.parent / f"{source_dir.name}.{MANIFEST_NAME}"


def _write_manifest(source_dir: Path, manifest: dict[str, Any]) -> None:
    # PA12: shared with campaign/build.py's canonical-source metadata write
    # rather than each maintaining its own temp-file+fsync+replace copy.
    from bigcherry.source.identity import atomic_write_json
    atomic_write_json(_manifest_path(source_dir), manifest)


def _read_manifest(source_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(source_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_reuse(source_dir: Path, expected_identity: dict[str, str]) -> bool:
    """True if source_dir can be reused as-is for expected_identity.

    Fails CLOSED (raises, does not silently return False and let the
    caller quietly rebuild over a suspicious directory) when a manifest
    exists but disagrees with reality -- that state means something wrote
    into this identity's directory without going through this module.
    """
    manifest = _read_manifest(source_dir)
    if manifest is None:
        # No manifest at all in an existing directory is not safely
        # reusable -- treat as "not materialized yet" so the caller
        # rebuilds from scratch rather than trusting an unmanaged tree.
        return False

    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PatchSourceIsolationError(
            f"unsupported or missing manifest schema at {source_dir}: "
            f"{manifest.get('manifest_schema_version')!r}"
        )

    # Iterate the identity's own keys (not a hardcoded tuple) so this check
    # naturally covers both the stock identity (no framework_baseline_digest
    # key at all) and the patched identity (which has one) without drifting
    # out of sync if either identity shape changes later.
    for key in expected_identity:
        if manifest.get(key) != expected_identity.get(key):
            raise PatchSourceIsolationError(
                f"source provenance mismatch for {key!r} at {source_dir}: "
                f"manifest={manifest.get(key)!r}, expected={expected_identity.get(key)!r}"
            )

    actual_head = git_head(source_dir)
    if actual_head != manifest.get("head"):
        raise PatchSourceIsolationError(
            f"source HEAD changed after materialization at {source_dir}: "
            f"manifest={manifest.get('head')!r}, actual={actual_head!r}"
        )

    actual_tree = git_worktree_tree(source_dir)
    if actual_tree != manifest.get("source_tree_oid"):
        raise PatchSourceIsolationError(
            f"patched source tree was modified after materialization at "
            f"{source_dir}: manifest tree={manifest.get('source_tree_oid')!r}, "
            f"actual tree={actual_tree!r}"
        )

    if manifest.get("patched_tree") != manifest.get("source_tree_oid"):
        raise PatchSourceIsolationError(
            f"manifest tree attestations disagree at {source_dir}: "
            f"patched_tree={manifest.get('patched_tree')!r}, "
            f"source_tree_oid={manifest.get('source_tree_oid')!r}"
        )

    actual_slice_id = _source_slice_id(
        source_dir=source_dir,
        upstream_revision=expected_identity["resolved_revision"],
        tree_oid=actual_tree,
    )
    if manifest.get("source_slice_id") != actual_slice_id:
        raise PatchSourceIsolationError(
            f"source slice provenance mismatch at {source_dir}: "
            f"manifest={manifest.get('source_slice_id')!r}, "
            f"actual={actual_slice_id!r}"
        )

    if manifest.get("materialization_plan_id") != expected_identity.get("source_key"):
        raise PatchSourceIsolationError(
            f"materialization plan provenance mismatch at {source_dir}: "
            f"manifest={manifest.get('materialization_plan_id')!r}, "
            f"expected={expected_identity.get('source_key')!r}"
        )

    return True


def _add_worktree(base_repo: Path, worktree_dir: Path, base_revision: str) -> None:
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    if worktree_dir.exists():
        # A directory with no valid manifest (caught above) or a leftover
        # partial worktree from a killed prior run -- remove the stale
        # directory outright rather than trying to patch up unknown state.
        shutil.rmtree(worktree_dir, ignore_errors=True)
    # prune AFTER the rmtree (and even when the directory is already gone):
    # a registration whose directory no longer exists is "missing but
    # already registered" -- a prune that ran BEFORE the directory was
    # removed leaves it in place, and the re-add below then fails with
    # exit 128 (caught by a real RS05 rebuild test).
    _run(["git", "worktree", "prune"], cwd=base_repo, check=False)
    _run(
        ["git", "worktree", "add", "--detach", str(worktree_dir), base_revision],
        cwd=base_repo,
    )


def _apply_composition(
    source_dir: Path,
    composition: Sequence[tuple[str, str]],
    *,
    overlay_root: Path | None,
    root: Path | None = None,
) -> None:
    """Overlay bigcherry's own source additions (EXACTLY the files whose bytes
    are in the identity's overlay_digest), then apply the EXPLICIT composition
    IN THE GIVEN ORDER via the registry loader (B3 digest re-check applies).

    No lifecycle-state scan anywhere: the composition is exactly what the
    caller resolved (RV80/B6). A patch under test is materialized as it will
    actually SHIP: on top of the explicitly resolved named composition, not
    bare upstream (RD08's anchor-dependency failure is why the composition
    step exists at all).
    """
    registry = _patch_registry().load_registry(root or PATCHES_ROOT)
    from bigcherry import patcher  # noqa: E402

    if overlay_root is not None:
        overlay_entries = _overlay_files(overlay_root)
    else:
        overlay_entries = []
    for relative, file_digest in overlay_entries:
        # overlay_entries is non-empty only when overlay_root is present.
        assert overlay_root is not None
        source = overlay_root / relative
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != file_digest:
            raise PatchSourceIsolationError(
                f"overlay file {relative} changed between identity computation "
                f"and application: {actual} != {file_digest}"
            )
        target.write_bytes(data)

    for patch_id, expected_digest in composition:
        try:
            descriptor = registry.get(patch_id)
        except Exception as exc:
            raise PatchSourceIsolationError(
                f"patch module does not exist: {patch_id!r}"
            ) from exc
        actual_digest = hashlib.sha256(
            (registry.root / descriptor.implementation_path).read_bytes()
        ).hexdigest()
        if actual_digest != expected_digest:
            raise PatchSourceIsolationError(
                f"patch {patch_id} implementation changed between identity "
                f"computation and application: {actual_digest} != {expected_digest}"
            )
        implementation = _patch_registry().load_implementation(
            descriptor, root=registry.root
        )
        results = patcher.apply_all(implementation, source_dir, dry_run=False)
        for result in results:
            if result.failed:
                raise PatchSourceIsolationError(
                    f"patch {patch_id} failed to apply to {result.path} in "
                    f"isolated worktree {source_dir}: "
                    f"{[r.detail for r in result.failed]}"
                )


def _materialize_v2(
    *,
    base_repo: Path,
    worktree_root: Path,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    overlay_root: Path | None,
    requested_revision: str | None = None,
    variant_transform: SourceTransform | None = None,
) -> Path:
    """The single v2 worktree-construction path (RV80/B2).

    `git worktree add --detach` at the RESOLVED SHA (a moved ref is a new
    identity, never a reused stale worktree), overlay + explicit ordered
    composition applied, optional content-addressed variant transform, then a
    state-independent v2 manifest. Reuse is only permitted when manifest AND
    actual on-disk tree hash both match this exact identity.
    """
    from bigcherry.source.identity import plan_lock

    identity = _make_source_identity_v2(
        resolved_revision=resolved_revision,
        composition=composition,
        overlay_root=overlay_root,
        variant_transform=variant_transform,
    )
    source_dir = worktree_root / identity["source_key"]

    # PA12 (L6.2): serialize the whole inspect-cache -> add-worktree ->
    # apply-composition -> write-manifest sequence per source_key, so two
    # processes materializing the same identity concurrently cannot both
    # try to `git worktree add` into the same destination.
    with plan_lock(worktree_root, identity["source_key"]):
        if source_dir.exists() and _verify_reuse(source_dir, identity):
            return source_dir

        _add_worktree(base_repo, source_dir, resolved_revision)
        _apply_composition(
            source_dir,
            composition,
            overlay_root=overlay_root,
            root=PATCHES_ROOT,
        )

        if variant_transform is not None:
            variant_transform.apply(source_dir)

        source_tree_oid = git_worktree_tree(source_dir)
        manifest = {
            **identity,
            "materialization_plan_id": identity["source_key"],
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "requested_revision": requested_revision,
            "head": git_head(source_dir),
            "source_tree_oid": source_tree_oid,
            "source_slice_id": _source_slice_id(
                source_dir=source_dir,
                upstream_revision=resolved_revision,
                tree_oid=source_tree_oid,
            ),
            "patched_tree": source_tree_oid,
        }
        _write_manifest(source_dir, manifest)
        return source_dir


def verify_composition_idempotent(
    *,
    base_repo: Path,
    source: Path,
    worktree_root: Path,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    overlay_root: Path | None = None,
    requested_revision: str | None = None,
) -> bool:
    """Re-run exact materialization and prove it reused the same tree.

    The second invocation must pass the v2 manifest/tree reuse check; equality
    of the resulting path and git tree then proves a no-op reapplication.
    """
    before = git_worktree_tree(source)
    again = materialize_composition(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=composition,
        overlay_root=overlay_root,
        requested_revision=requested_revision,
    )
    return again == source and git_worktree_tree(again) == before


def materialize_composition(
    *,
    base_repo: Path,
    worktree_root: Path,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    overlay_root: Path | None = None,
    requested_revision: str | None = None,
) -> Path:
    """Materialize an isolated worktree for an EXPLICIT composition.

    ``composition`` is the EXACT ORDERED ``[(patch_id, implementation_digest)]``
    list (topologically ordered, e.g. from :func:`resolve_source_composition`)
    -- its entries are re-verified against the live registry before
    application and the whole list (in order) is what enters the v2 source
    identity. An empty composition + ``overlay_root=None`` is the stock
    (pristine upstream) case. ``materialize_source()`` (implicit state-scan
    baseline) is RETIRED (runbook 12 / RV80-B6).
    """
    return _materialize_v2(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=tuple(composition),
        overlay_root=overlay_root,
        requested_revision=requested_revision,
    )


def materialize_source_variant(
    *,
    base_repo: Path,
    worktree_root: Path,
    resolved_revision: str,
    composition: Sequence[tuple[str, str]],
    variant_name: str | None = None,
    overlay_root: Path | None = None,
    requested_revision: str | None = None,
    transform: SourceTransform | None = None,
    # Backward-compatible input only. This value is deliberately not used as
    # identity evidence; the transform digest is always derived below.
    variant_digest: str | None = None,
    apply_variant: Callable[[Path], None] | None = None,
) -> Path:
    """Materialize an isolated worktree for an EXPLICIT composition plus a
    post-composition variant transform ("subject vs control" evidence).

    The two worktrees differ ONLY by the content-addressed transform applied
    on top of the identical composition (e.g. the RD08 package-local correctness producer's
    VDR1-control / VDR2-subject pair). ``transform.digest`` is derived from
    the exact immutable transform definition. The legacy ``variant_digest``
    argument is ignored and retained solely so older callers fail safe while
    migrating.

    RV80/B2: the identity is ``bigcherry-patch-source-variant-v2`` -- the
    same state-independent payload as :func:`materialize_composition` plus
    ``variant_name`` + the derived transform digest. The v1 state-scan identity is
    retired; the composition comes explicitly from the caller (e.g. via
    :func:`resolve_source_composition`), never from a lifecycle-state scan.
    """
    if transform is not None and apply_variant is not None:
        raise TypeError("pass either transform or apply_variant, not both")
    if transform is None:
        if apply_variant is None:
            if variant_name is None:
                raise TypeError("variant_name is required when transform is omitted")
            transform = SourceTransform(
                name=variant_name,
                schema_version=SOURCE_TRANSFORM_SCHEMA_VERSION,
                operations=(),
            )
        else:
            if variant_name is None:
                variant_name = getattr(apply_variant, "__qualname__", "variant")
            transform = SourceTransform.from_callable(variant_name, apply_variant)
    elif variant_name is not None and variant_name != transform.name:
        raise ValueError(
            f"variant_name {variant_name!r} disagrees with transform name {transform.name!r}"
        )

    # A caller may retain an old label forever; only the derived transform can
    # authorize cache reuse.
    del variant_digest
    return _materialize_v2(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=tuple(composition),
        overlay_root=overlay_root,
        requested_revision=requested_revision,
        variant_transform=transform,
    )


def materialize_stock_source(
    *,
    base_repo: Path,
    worktree_root: Path,
    base_revision: str,
) -> Path:
    """A genuinely pristine pinned-upstream worktree for stock bench
    comparison: v2 identity with the EMPTY composition and no overlay. The
    base ref is resolved to its immutable commit SHA first, so a moved ref
    yields a new identity (and the requested ref is recorded in the manifest
    informationally only)."""
    resolved_revision = resolve_base_revision(base_revision, repo=base_repo)
    return _materialize_v2(
        base_repo=base_repo,
        worktree_root=worktree_root,
        resolved_revision=resolved_revision,
        composition=(),
        overlay_root=None,
        requested_revision=base_revision,
    )


def resolve_source_composition(
    source_name: str,
    *,
    focal: str | None = None,
    extra_patches: tuple[str, ...] = (),
    base_ref: str = "HEAD",
    base_repo: Path | None = None,
    recipes: Path | None = None,
    patches_root: Path | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Resolve a source's EXPLICIT named composition for identity purposes.

    ``source_name``'s ``[patch-set.*]`` sets from config/recipes.toml (plus
    an optional single ``focal`` patch and/or an explicit ``extra_patches``
    stack, e.g. the RD08 evidence stack) are resolved through
    ``campaign_resolution.resolve_lane`` + ``patchset.resolve_exact`` -- the
    authoritative exact-composition validator (fail-closed on unknown IDs,
    rejected members, missing requires, internal conflicts) in TRUE
    topological order.

    Returns ``(resolved_base_sha, ordered [(patch_id, impl_digest)])``. This
    is the ONLY sanctioned way the identity/materialization path obtains a
    composition -- RV80/B6 forbids lifecycle-state scans here.
    """
    if base_repo is None:
        raise PatchSourceIsolationError(
            "resolve_source_composition: base_repo is required"
        )
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry.campaign import resolution as campaign_resolution # noqa: E402
    from bigcherry.core import config as campaign_config # noqa: E402
    from bigcherry.core import paths # noqa: E402
    from bigcherry.patch import patchset # noqa: E402

    resolved_revision = resolve_base_revision(base_ref, repo=base_repo)
    cfg = campaign_config.load(recipes or paths.RECIPES)
    patches_root = patches_root or PATCHES_ROOT
    catalog = patchset.catalog(directory=patches_root)
    lane = campaign_resolution.resolve_lane(source_name, cfg, catalog)
    # lane.patch_set.module_ids is the FULL named composition (the merged
    # multi-set result already contains the validated-enhancements members;
    # lane.promoted_enhancements is a declared-config view of them, not an
    # extra). Extras/focal go on top and are deduplicated above.
    ids = list(lane.patch_set.module_ids)
    extras = list(extra_patches)
    if focal is not None:
        if focal in ids:
            raise PatchSourceIsolationError(
                f"focal patch {focal!r} is already in source {source_name!r}'s "
                "named composition"
            )
        extras.append(focal)
    if len(set(extras)) != len(extras):
        raise PatchSourceIsolationError("explicit composition contains duplicates")
    ids = [*ids, *extras]
    resolved = patchset.resolve_exact(tuple(ids), directory=patches_root)
    registry = _patch_registry().load_registry(patches_root)
    composition = tuple(
        (
            module.patch_id,
            hashlib.sha256(
                (
                    registry.root / registry.get(module.patch_id).implementation_path
                ).read_bytes()
            ).hexdigest(),
        )
        for module in resolved.modules
    )
    return resolved_revision, composition


def remove_worktree(base_repo: Path, source_dir: Path) -> None:
    """Explicit teardown -- worktrees are cheap to keep around for reuse, but
    callers doing disk-space cleanup should use this rather than a raw
    rmtree so `git worktree list` in base_repo doesn't accumulate stale
    registrations for a directory that no longer exists."""
    if source_dir.exists():
        _run(
            ["git", "worktree", "remove", "--force", str(source_dir)],
            cwd=base_repo,
            check=False,
        )
    shutil.rmtree(source_dir, ignore_errors=True)
    _manifest_path(source_dir).unlink(missing_ok=True)
    _run(["git", "worktree", "prune"], cwd=base_repo, check=False)
