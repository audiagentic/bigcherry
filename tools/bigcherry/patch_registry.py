"""Patch registry: one normalized, content-identified descriptor for every
patch in ``patches/`` — flat simple patches and packaged patches alike.

patch-system PA02 / RS01 (docs/planning/active/patch-system/
PATCH_REFACTOR_RUNBOOK.md, sections 3-14). Before this module there were two
implicit discovery notions (``patchset.discover_modules()``'s recursive
``*.py`` glob) and one physical representation; after it, the single rule
"patch ID -> resolve once -> :class:`PatchDescriptor`" holds and no
downstream code ever decides whether a patch is flat or packaged.

Discovery (runbook section 4), exactly:

* simple:  ``patches/*.py`` at the ROOT level only, filename not ``_``-prefixed
* packaged: ``patches/**/patch.toml`` at any depth, no relative path
  component may start ``_``
* NEVER: arbitrary nested ``*.py`` (a package's ``patch.py`` and every
  ``validation/*.py`` are only reachable through their package)

The v1 dependency DAG (runbook section 11) is normative: this module imports
only ``paths`` and ``patcher`` (plus stdlib). It never imports ``patchset``,
``check``, or any upper layer. ``patch_validation.py`` (RS07) imports
:const:`VALIDATION_FRAMEWORK_VERSION` from here rather than the other way
around — the constant is pinned in the lower layer so the validation digest
computed during discovery stays cycle-free; ``patch_validation`` re-exports it.

Path fields on :class:`PatchDescriptor` are RELATIVE to the registry root
(the directory passed to :func:`load_registry`; normally
``paths.PATCHES``). That is the canonical, checkout-independent form the
runbook asks for: for the real tree, root-relative IS repo-relative under
``patches/``, and tests can point the registry at a temp root without
changing descriptor contents. Convert to absolute at the I/O boundary only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import paths
from .patcher import FilePatch

# The "validation framework semantic version" (runbook 14.2). Bumping it
# invalidates validation evidence while leaving production source/build
# identity reusable (15). Pinned HERE (lower layer) so the validation digest
# computed at discovery time stays import-cycle-free; patch_validation.py
# imports and re-exports it.
VALIDATION_FRAMEWORK_VERSION = "1"

STATES: tuple[str, ...] = ("validated", "rejected", "untested")
PATCH_KINDS: tuple[str, ...] = ("framework", "upstream-backport", "enhancement")
PATCH_ORIGINS: tuple[str, ...] = ("local", "upstream-commit", "upstream-pr", "external-fork")
PATCH_BACKENDS: tuple[str, ...] = ("hip", "vulkan", "agnostic")

PATCH_TOML_SCHEMA = 1
REPRESENTATION_SIMPLE = "simple"
REPRESENTATION_PACKAGED = "packaged"

# Canonical patch ID: numeric order prefix + ``_`` + identifier characters.
# Mirrors every existing flat module filename (0100_..., 1204_...).
_PATCH_ID_PATTERN = re.compile(r"^(\d{2,})_[0-9A-Za-z_]+$")

_PATCH_TOML_REQUIRED_KEYS = frozenset({"schema", "id", "order", "group", "state"})
_PATCH_TOML_STRING_LIST_KEYS = frozenset({
    "plan-ids", "requires", "conflicts", "requires-options", "forbids-options",
    "subsystems", "hardware", "validation-architectures",
})
_PATCH_TOML_KNOWN_KEYS = _PATCH_TOML_REQUIRED_KEYS | _PATCH_TOML_STRING_LIST_KEYS | frozenset({
    "kind", "origin", "backend", "upstream", "external-source", "experiment-contract",
})


class PatchRegistryError(ValueError):
    """Raised when the patches/ tree violates the registry's fail-closed rules
    (bad schema, duplicate identity, path escape, missing implementation)."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _contains(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_resolve(base: Path, candidate: Path, *, what: str) -> Path:
    """Resolve ``candidate`` (following symlinks) and fail closed if it
    escapes ``base`` (runbook section 4 path safety, section 73 sheet)."""
    resolved = candidate.resolve()
    if not _contains(base, candidate):
        raise PatchRegistryError(f"{what} escapes {base}: {candidate}")
    return resolved


def _is_reserved(part: str) -> bool:
    return part.startswith("_")


# ---------------------------------------------------------------- discovery


def discover_simple_patches(root: Path) -> list[Path]:
    """Simple patches: root-level ``*.py`` only, no ``_`` prefix (runbook 4).

    Deliberately different from the old ``patchset.discover_modules()``
    recursive glob: a nested ``*.py`` is never a simple patch — it belongs to
    a package or does not exist.
    """
    root = root.resolve()
    if not root.is_dir():
        return []
    found: list[Path] = []
    for candidate in sorted(root.glob("*.py")):
        if _is_reserved(candidate.name):
            continue
        _safe_resolve(root, candidate, what="simple patch module")
        found.append(candidate)
    return found


def discover_packaged_patches(root: Path) -> list[Path]:
    """Packaged patches: every ``patch.toml`` at any depth whose relative
    path has no ``_``-prefixed component (runbook 4/6)."""
    root = root.resolve()
    if not root.is_dir():
        return []
    found: list[Path] = []
    for candidate in sorted(root.rglob("patch.toml")):
        relative = candidate.relative_to(root)
        if any(_is_reserved(part) for part in relative.parts):
            continue
        _safe_resolve(root, candidate, what="patch.toml")
        found.append(candidate)
    return found


# ------------------------------------------------------ constant extraction
#
# The metadata readers for legacy flat modules. patchset.py (RS02) delegates
# to these; they are here so the registry owns ALL metadata parsing and the
# old module keeps a thin compatibility façade (runbook 11).


def _constant(path: Path, name: str, pattern: str) -> str | None:
    """Extract a module-level string constant without importing executable code."""
    try:
        content = path.read_text(encoding="utf-8")
        match = re.search(rf'^{name}\s*=\s*["\']({pattern})["\']', content, re.MULTILINE)
        return match.group(1) if match else None
    except OSError:
        return None


def _literal_constant(path: Path, name: str) -> object | None:
    """Read a literal module constant without importing executable code."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return None
    return None


def _constant_strings(path: Path, name: str) -> tuple[str, ...]:
    value = _literal_constant(path, name)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, str) and item for item in value
    ):
        return tuple(value)
    raise PatchRegistryError(f"{path.name}: {name} must be a string or list/tuple of strings")


def module_group(path: Path) -> str:
    """The group a legacy patch module declares (default when absent)."""
    return _constant(path, "GROUP", r"[\w-]+") or "core"


def module_state(path: Path) -> str:
    """The state a legacy patch module declares (default when absent)."""
    return _constant(path, "STATE", r"[\w-]+") or "untested"


def module_upstream(path: Path) -> str | None:
    """The upstream commit SHA a legacy patch backports from, if any."""
    return _constant(path, "UPSTREAM", r"[0-9a-fA-F]{7,40}")


def group_is_explicit(path: Path) -> bool:
    """Whether a legacy module declares GROUP itself (vs. the default)."""
    return _literal_constant(path, "GROUP") is not None


def _module_order(stem: str) -> int:
    match = re.match(r"^(\d+)_", stem)
    return int(match.group(1)) if match else 2**31


# ---------------------------------------------------------------- descriptor


@dataclass(frozen=True)
class PatchDescriptor:
    """Normalized, immutable identity + metadata for one patch, regardless of
    representation (runbook section 9). ``representation`` is
    ``"simple"`` or ``"packaged"``; for simple patches ``package_root``,
    ``metadata_path``, ``validation_path``/``validation_digest`` and the
    descriptive fields that legacy keeps in ``patches/catalog.toml`` are
    ``None``/empty. Path fields are RELATIVE to the registry root (see
    module docstring)."""

    patch_id: str
    order: int

    representation: str
    implementation_path: Path
    package_root: Path | None
    metadata_path: Path | None

    group: str
    state: str

    kind: str | None
    origin: str | None
    backend: str | None

    upstream: str | None
    external_source: str | None
    plan_ids: tuple[str, ...]

    requires: tuple[str, ...]
    conflicts: tuple[str, ...]

    requires_options: tuple[str, ...]
    forbids_options: tuple[str, ...]

    subsystems: tuple[str, ...]
    hardware: tuple[str, ...]
    validation_architectures: tuple[str, ...]

    experiment_contract: str | None

    implementation_digest: str

    validation_path: Path | None
    validation_digest: str | None


# ------------------------------------------------------------------ legacy


def _legacy_descriptor(
    root: Path, path: Path, *, validate_state: bool = True
) -> PatchDescriptor:
    patch_id = path.stem
    if not _PATCH_ID_PATTERN.match(patch_id):
        raise PatchRegistryError(
            f"simple patch filename {patch_id!r} does not match the canonical "
            f"ID pattern <numeric-order>_<name>"
        )
    state = module_state(path)
    # RV80 follow-up (GPT ruling b): the reporting facade (patchset.describe)
    # must be able to SURFACE a malformed lifecycle state (state_valid=False)
    # rather than crash. validate_state=False preserves the raw state; every
    # strict path (load_registry/catalog/resolve/materialization) keeps the
    # default True and stays fail-closed.
    if state not in STATES and validate_state:
        raise PatchRegistryError(f"{patch_id}: invalid STATE={state!r}")
    return PatchDescriptor(
        patch_id=patch_id,
        order=_module_order(patch_id),
        representation=REPRESENTATION_SIMPLE,
        implementation_path=path.relative_to(root),
        package_root=None,
        metadata_path=None,
        group=module_group(path),
        state=state,
        kind=None,
        origin=None,
        backend=None,
        upstream=module_upstream(path),
        external_source=None,
        plan_ids=(),
        requires=_constant_strings(path, "REQUIRES"),
        conflicts=_constant_strings(path, "CONFLICTS"),
        requires_options=(),
        forbids_options=(),
        subsystems=(),
        hardware=(),
        validation_architectures=(),
        experiment_contract=None,
        implementation_digest=_sha256_bytes(path.read_bytes()),
        validation_path=None,
        validation_digest=None,
    )


# ------------------------------------------------------------ patch.toml


def _string_list(record: dict, key: str, *, patch_id: str, label: str) -> tuple[str, ...]:
    value = record.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise PatchRegistryError(
            f"{patch_id}: {label} {key!r} must be a list of non-empty strings"
        )
    if len(set(value)) != len(value):
        raise PatchRegistryError(f"{patch_id}: {label} {key!r} contains duplicates")
    return tuple(value)


def _parse_patch_toml(
    root: Path, toml_path: Path, *, validate_state: bool = True
) -> dict:
    """Parse + schema-validate one ``patch.toml`` (runbook 6/7). Returns the
    validated record. All errors name the file and fail closed. With
    ``validate_state=False`` (reporting facade only) a malformed lifecycle
    ``state`` is preserved rather than rejected; every other schema check is
    unchanged."""
    where = toml_path
    try:
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PatchRegistryError(f"{where}: invalid TOML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PatchRegistryError(f"{where}: top level must be a TOML table")

    unknown = sorted(set(raw) - _PATCH_TOML_KNOWN_KEYS)
    if unknown:
        raise PatchRegistryError(f"{where}: unknown key(s): {', '.join(unknown)}")
    missing = sorted(_PATCH_TOML_REQUIRED_KEYS - set(raw))
    if missing:
        raise PatchRegistryError(f"{where}: missing required key(s): {', '.join(missing)}")

    schema = raw["schema"]
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != PATCH_TOML_SCHEMA:
        raise PatchRegistryError(
            f"{where}: unsupported schema {schema!r} (want {PATCH_TOML_SCHEMA})"
        )

    patch_id = raw["id"]
    if not isinstance(patch_id, str) or not _PATCH_ID_PATTERN.match(patch_id):
        raise PatchRegistryError(
            f"{where}: id {patch_id!r} does not match the canonical ID pattern "
            f"<numeric-order>_<name>"
        )

    # Directory basename == patch ID (runbook 6: mandatory).
    if toml_path.parent.name != patch_id:
        raise PatchRegistryError(
            f"{where}: directory name {toml_path.parent.name!r} must equal id "
            f"{patch_id!r}"
        )

    order = raw["order"]
    if isinstance(order, bool) or not isinstance(order, int):
        raise PatchRegistryError(f"{where}: order must be an integer")
    if order != int(patch_id.split("_", 1)[0]):
        raise PatchRegistryError(
            f"{where}: order {order} does not match id prefix "
            f"{patch_id.split('_', 1)[0]!r}"
        )

    group = raw["group"]
    if not isinstance(group, str) or not group:
        raise PatchRegistryError(f"{where}: group must be a non-empty string")

    state = raw["state"]
    if state not in STATES and validate_state:
        raise PatchRegistryError(f"{where}: state must be one of {STATES}, got {state!r}")

    for key, vocabulary in (
        ("kind", PATCH_KINDS), ("origin", PATCH_ORIGINS), ("backend", PATCH_BACKENDS),
    ):
        value = raw.get(key)
        if value is not None and value not in vocabulary:
            raise PatchRegistryError(
                f"{where}: {key} must be one of {vocabulary}, got {value!r}"
            )

    for key in ("upstream", "external-source", "experiment-contract"):
        value = raw.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise PatchRegistryError(f"{where}: {key} must be a non-empty string when present")
    upstream = raw.get("upstream")
    if upstream is not None and not re.match(r"^[0-9a-fA-F]{7,40}$", upstream):
        raise PatchRegistryError(f"{where}: upstream {upstream!r} is not a commit SHA")

    record = dict(raw)
    for key in _PATCH_TOML_STRING_LIST_KEYS:
        record[key] = _string_list(raw, key, patch_id=patch_id, label="")
    return record


def _contract_hash(contract_id: str, *, contracts_path: Path | None = None) -> str:
    """The canonical Experiment Contract hash (EC01) for ``contract_id`` —
    ``ExperimentContract.contract_hash``: semantic content, schema-normalized
    (key-order independent, defaults applied), computed by the contract
    module that OWNS the schema (runbook 14.2: "linked Experiment Contract
    hash" = semantic content, not file bytes).

    RV80/A1: this used to be an ad-hoc sha256 over the RAW TOML table read
    with tomllib — a second, divergent hashing authority for the same
    contract (raw-table bytes vs the parsed+defaulted schema object), so a
    schema-level normalization (e.g. defaults, tuple ordering) could change
    ``contract_hash`` while the registry's hash stayed put, or vice versa.
    The registry now defers to the canonical hash. A1 amends the §11 DAG pin:
    the registry may import experiment_contract ONE-WAY (it is a leaf config
    module — stdlib + autotune_schema only — and must never import the
    registry).
    """
    file_path = contracts_path or paths.EXPERIMENT_CONTRACTS
    if not file_path.is_file():
        raise PatchRegistryError(
            f"experiment contract {contract_id!r} referenced but no contract file at {file_path}"
        )
    from bigcherry import experiment_contract  # noqa: PLC0415 (A1: one-way leaf import)

    try:
        contracts = experiment_contract.load_contracts(file_path)
    except Exception as exc:  # noqa: BLE001 - schema errors become tree errors
        raise PatchRegistryError(
            f"experiment contracts file {file_path.name} failed to load: {exc}"
        ) from exc
    try:
        return contracts[contract_id].contract_hash
    except KeyError:
        raise PatchRegistryError(
            f"experiment contract {contract_id!r} not found in {file_path.name}"
        ) from None


def _validation_identity(
    root: Path,
    package_dir: Path,
    *,
    contract_id: str | None,
    contract_hash: str | None = None,
) -> tuple[Path | None, str | None]:
    """(validation_path, validation_digest) for a package (runbook 14.2).

    Absent ``validation.toml`` -> (None, None): a packaged patch may ship
    implementation first and add validation later (state promotion is a
    separate concern, runbook 39).
    """
    manifest_toml = package_dir / "validation.toml"
    _safe_resolve(package_dir, manifest_toml, what="validation manifest")
    if not manifest_toml.is_file():
        return None, None

    entries: list[tuple[str, str]] = []
    files: list[Path] = [manifest_toml]
    validation_dir = package_dir / "validation"
    if validation_dir.is_dir():
        for candidate in sorted(validation_dir.rglob("*")):
            if candidate.is_file():
                _safe_resolve(package_dir, candidate, what="validation file")
                files.append(candidate)
    for file_path in sorted(files, key=lambda p: p.relative_to(package_dir).as_posix()):
        relative = file_path.relative_to(package_dir).as_posix()
        entries.append((relative, _sha256_bytes(file_path.read_bytes())))

    contract = None
    if contract_id is not None:
        if contract_hash is None:
            raise PatchRegistryError(
                f"{contract_id}: contract hash must be precomputed by the caller"
            )
        contract = {"id": contract_id, "sha256": contract_hash}
    payload = {
        "schema": "bigcherry-patch-validation-identity-v1",
        "validation_framework_version": VALIDATION_FRAMEWORK_VERSION,
        "files": entries,  # already sorted by relative path
        "experiment_contract": contract,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (
        manifest_toml.relative_to(root),
        _sha256_bytes(encoded.encode("utf-8")),
    )


def _packaged_descriptor(
    root: Path,
    toml_path: Path,
    *,
    contracts_path: Path | None = None,
    validate_state: bool = True,
) -> PatchDescriptor:
    record = _parse_patch_toml(root, toml_path, validate_state=validate_state)
    package_dir = toml_path.parent
    patch_id = record["id"]
    implementation = package_dir / "patch.py"
    if not implementation.is_file():
        raise PatchRegistryError(f"{patch_id}: package is missing patch.py at {implementation}")
    _safe_resolve(package_dir, implementation, what="patch.py")

    contract_id = record.get("experiment-contract")
    # A packaged patch referencing a contract that does not exist is a tree
    # error at DISCOVERY time (fail-closed), regardless of whether this
    # package ships validation yet -- the reference is metadata authority.
    contract_hash = (
        _contract_hash(contract_id, contracts_path=contracts_path)
        if contract_id is not None
        else None
    )
    validation_path, validation_digest = _validation_identity(
        root, package_dir, contract_id=contract_id, contract_hash=contract_hash,
    )
    return PatchDescriptor(
        patch_id=patch_id,
        order=record["order"],
        representation=REPRESENTATION_PACKAGED,
        implementation_path=implementation.relative_to(root),
        package_root=package_dir.relative_to(root),
        metadata_path=toml_path.relative_to(root),
        group=record["group"],
        state=record["state"],
        kind=record.get("kind"),
        origin=record.get("origin"),
        backend=record.get("backend"),
        upstream=record.get("upstream"),
        external_source=record.get("external-source"),
        plan_ids=record["plan-ids"],
        requires=record["requires"],
        conflicts=record["conflicts"],
        requires_options=record["requires-options"],
        forbids_options=record["forbids-options"],
        subsystems=record["subsystems"],
        hardware=record["hardware"],
        validation_architectures=record["validation-architectures"],
        experiment_contract=record.get("experiment-contract"),
        implementation_digest=_sha256_bytes(implementation.read_bytes()),
        validation_path=validation_path,
        validation_digest=validation_digest,
    )


# ----------------------------------------------------------------- registry


@dataclass(frozen=True)
class PatchRegistry:
    """The complete, validated, deterministically ordered set of patches in
    one root (runbook 10). Built by :func:`load_registry` — construct
    descriptors through it so duplicate detection always applies."""

    root: Path
    descriptors: tuple[PatchDescriptor, ...]

    @property
    def by_id(self) -> dict[str, PatchDescriptor]:
        return {d.patch_id: d for d in self.descriptors}

    def get(self, patch_id: str) -> PatchDescriptor:
        known = self.by_id
        if patch_id not in known:
            raise PatchRegistryError(f"unknown patch: {patch_id!r}")
        return known[patch_id]


def load_registry(
    root: Path | None = None,
    *,
    contracts_path: Path | None = None,
) -> PatchRegistry:
    """Discover + validate every patch under ``root`` (default
    ``paths.PATCHES``). Fails closed on: unknown ``patch.toml`` keys, bad
    state, directory/ID mismatch, missing ``patch.py``, non-canonical ID
    pattern, duplicate IDs across ANY representation (runbook 5), and
    path/symlink escape.

    ``contracts_path`` is a test seam for the experiment-contracts file;
    production callers pass nothing and ``paths.EXPERIMENT_CONTRACTS`` is
    used.
    """
    resolved_root = (root or paths.PATCHES).resolve()
    if not resolved_root.is_dir():
        return PatchRegistry(root=resolved_root, descriptors=())

    descriptors: list[PatchDescriptor] = []
    seen: dict[str, str] = {}

    def _add(descriptor: PatchDescriptor) -> None:
        if descriptor.patch_id in seen:
            raise PatchRegistryError(
                f"duplicate patch ID {descriptor.patch_id!r}: "
                f"{seen[descriptor.patch_id]} and "
                f"{descriptor.representation} at {descriptor.implementation_path}"
            )
        seen[descriptor.patch_id] = (
            f"{descriptor.representation} at {descriptor.implementation_path}"
        )
        descriptors.append(descriptor)

    for toml_path in discover_packaged_patches(resolved_root):
        _add(_packaged_descriptor(resolved_root, toml_path, contracts_path=contracts_path))
    for py_path in discover_simple_patches(resolved_root):
        _add(_legacy_descriptor(resolved_root, py_path))

    ordered = tuple(sorted(descriptors, key=lambda d: (d.order, d.patch_id)))
    return PatchRegistry(root=resolved_root, descriptors=ordered)


def describe_all(
    root: Path | None = None, *, contracts_path: Path | None = None
) -> tuple[PatchDescriptor, ...]:
    """Report-only variant of :func:`load_registry` (runbook compatibility
    facade; RV80 follow-up / GPT ruling b).

    Identical discovery and structural validation (ID pattern, path/symlink
    escape, duplicate IDs, TOML schema) EXCEPT it does not fail closed on an
    invalid lifecycle ``state``: the raw state is preserved so the reporting
    facade (:func:`bigcherry.patchset.describe`) can surface it
    (``state_valid=False``) instead of crashing on a typo'd ``STATE``.

    Every strict path -- ``load_registry``, ``catalog``, ``resolve_exact``,
    source materialization -- keeps failing closed; this function is only for
    observational/reporting use and must never feed a resolution or a build.
    """
    resolved_root = (root or paths.PATCHES).resolve()
    if not resolved_root.is_dir():
        return ()
    descriptors: list[PatchDescriptor] = []
    seen: dict[str, str] = {}

    def _add(descriptor: PatchDescriptor) -> None:
        if descriptor.patch_id in seen:
            raise PatchRegistryError(
                f"duplicate patch ID {descriptor.patch_id!r}: "
                f"{seen[descriptor.patch_id]} and "
                f"{descriptor.representation} at {descriptor.implementation_path}"
            )
        seen[descriptor.patch_id] = (
            f"{descriptor.representation} at {descriptor.implementation_path}"
        )
        descriptors.append(descriptor)

    for toml_path in discover_packaged_patches(resolved_root):
        _add(_packaged_descriptor(
            resolved_root, toml_path, contracts_path=contracts_path,
            validate_state=False,
        ))
    for py_path in discover_simple_patches(resolved_root):
        _add(_legacy_descriptor(resolved_root, py_path, validate_state=False))

    return tuple(sorted(descriptors, key=lambda d: (d.order, d.patch_id)))


# ------------------------------------------------------------------- loader


# RV80/B5: frozen v1 rule — a packaged ``patch.py`` may import ONLY the
# Python stdlib and the BigCherry public patch API (``bigcherry.patcher``).
# Validated statically (AST walk) BEFORE execution: an out-of-scope import is
# a load-time tree error with a structured message, not an ImportError (or
# worse, an executed side effect) discovered mid-execution.
_PACKAGED_ALLOWED_MODULE = "bigcherry.patcher"


def _import_disallowed(module_name: str) -> bool:
    """True when a packaged patch.py import of ``module_name`` is outside the
    frozen v1 scope (stdlib + exactly ``bigcherry.patcher``). Deliberately
    strict: ``import bigcherry`` / ``from bigcherry import patcher`` expose
    the whole package surface and are NOT the public patch API — patches must
    spell ``bigcherry.patcher`` explicitly (the form every flat patch uses).
    """
    if module_name == _PACKAGED_ALLOWED_MODULE:
        return False
    return module_name.split(".")[0] not in sys.stdlib_module_names


def _validate_packaged_imports(source: str, descriptor: PatchDescriptor) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PatchRegistryError(
            f"{descriptor.patch_id}: patch.py failed to parse: {exc}"
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _import_disallowed(alias.name):
                    raise PatchRegistryError(
                        f"{descriptor.patch_id}: packaged patch.py may import only "
                        f"the Python stdlib and {_PACKAGED_ALLOWED_MODULE!r}; "
                        f"found {alias.name!r} (line {node.lineno})"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise PatchRegistryError(
                    f"{descriptor.patch_id}: relative imports are not allowed in "
                    f"packaged patch.py (line {node.lineno})"
                )
            if node.module and _import_disallowed(node.module):
                raise PatchRegistryError(
                    f"{descriptor.patch_id}: packaged patch.py may import only "
                    f"the Python stdlib and {_PACKAGED_ALLOWED_MODULE!r}; "
                    f"found {node.module!r} (line {node.lineno})"
                )


def load_implementation(
    descriptor: PatchDescriptor,
    *,
    root: Path | None = None,
) -> list[FilePatch]:
    """Load a patch's production implementation (runbook 12/13), for EITHER
    representation: read current bytes, compile directly, execute in a
    synthetic module, extract ``PATCH``/``PATCHES``.

    Deliberately not ``importlib`` (RV48/RE04 audit fix, inherited from
    ``patchset._load_module``): the standard loader's mtime-keyed pycache can
    serve stale bytecode for a file whose on-disk bytes have changed, and a
    patch's applied effect must never diverge from its content hash.

    RV80/B3: the bytes about to be executed are re-hashed against
    ``descriptor.implementation_digest`` and the load FAILS CLOSED on
    mismatch — a file edited between ``load_registry()`` and this call is a
    tree error, never a silently re-hashed patch. RV80/B5: packaged
    implementations additionally pass static import validation before exec.
    """
    resolved_root = (root or paths.PATCHES).resolve()
    path = resolved_root / descriptor.implementation_path
    if not _contains(resolved_root, path):
        raise PatchRegistryError(
            f"{descriptor.patch_id}: implementation path escapes registry root"
        )
    raw = path.read_bytes()
    actual_digest = _sha256_bytes(raw)
    if actual_digest != descriptor.implementation_digest:
        raise PatchRegistryError(
            f"{descriptor.patch_id}: implementation bytes at "
            f"{descriptor.implementation_path} no longer match the descriptor "
            f"digest (descriptor {descriptor.implementation_digest[:16]}…, "
            f"actual {actual_digest[:16]}…) — the file changed after the registry "
            "was loaded; reload the registry from this tree rather than "
            "executing a patch whose content identity moved"
        )
    source = raw.decode("utf-8")
    if descriptor.representation == REPRESENTATION_PACKAGED:
        _validate_packaged_imports(source, descriptor)
    code = compile(source, str(path), "exec")
    name = f"bigcherry._patches.{descriptor.patch_id}"
    module = ModuleType(name)
    module.__file__ = str(path)
    # Register before exec so a legacy patch module may import its siblings.
    sys.modules[name] = module
    exec(code, module.__dict__)
    found = getattr(module, "PATCHES", None)
    if found is None:
        single = getattr(module, "PATCH", None)
        found = [single] if single is not None else []
    if not found:
        raise PatchRegistryError(
            f"{descriptor.patch_id} defines neither PATCH nor PATCHES"
        )
    return list(found)
