"""Kernel unit ownership and naming (RCS19, docs/design/radiance-kernel-composition.md section 4).

Two checks, both offline and static (no source checkout, no compile):

* **Ownership.** Each patch lists, in ``patch.toml`` ``units``, the kernel translation units whose own source it edits
  (``libr4d/<unit>.hip``); the list must equal what its edits touch. A native rewrite (``components/*/native/<unit>.hip.rw``)
  owns the unit of the same name. A unit edited by more than one layer is listed by :func:`shared_units`; each must be
  recorded, with the reason, in ``unit-sharing.toml`` beside the patches, so a new overlap is a decision and not an accident.
  Region overlap inside a shared unit is gate 1's job (compose_check).
* **Naming.** A type, constant, macro or function a patch edit (in the device sources under ``libr4d/``) or a native
  rewrite introduces ends in ``_<arch>_<layer>``: ``_gfx12_rad0065`` for a patch (its own id), ``_gfx11_r3`` for a libr3
  rewrite. Names that predate the rule are recorded one by one in ``naming-exceptions.toml``; a new one fails.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from ..core import paths

UNIT_FILE = re.compile(r"^libr4d/([A-Za-z0-9_]+)\.hip$")
SHARING_FILE = "unit-sharing.toml"
EXCEPTIONS_FILE = "naming-exceptions.toml"
REWRITE_LAYER = "rewrite:libr3"
REWRITE_SUFFIX = re.compile(r"_gfx11_r3$")


def _patches_root(engine: str) -> Path:
    from ..core.context import ProjectContext
    return ProjectContext.resolve(engine=engine).patches_root


def _implementations(engine: str):
    """(descriptor, [FilePatch]) for every packaged patch."""
    from . import registry
    root = _patches_root(engine)
    reg = registry.load_registry(engine, root)
    for descriptor in reg.descriptors:
        yield descriptor, list(registry.load_implementation(engine, descriptor, root=root))


def patch_units(engine: str) -> dict[str, tuple[str, ...]]:
    """Per patch id, the units its edits touch."""
    out = {}
    for descriptor, patches in _implementations(engine):
        found = sorted({m.group(1) for p in patches if (m := UNIT_FILE.match(p.path))})
        out[descriptor.patch_id] = tuple(found)
    return out


def declared_units(engine: str) -> dict[str, tuple[str, ...]]:
    from . import registry
    root = _patches_root(engine)
    return {d.patch_id: tuple(sorted(d.units)) for d in registry.load_registry(engine, root).descriptors}


def rewrite_files(engine: str) -> list[Path]:
    return sorted((paths.REPO_ROOT / "engines" / engine / "components").glob("*/native/*.rw"))


def rewrite_units(engine: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in rewrite_files(engine):
        if path.name.endswith(".hip.rw"):
            out.setdefault(path.name[: -len(".hip.rw")], []).append(REWRITE_LAYER)
    return out


def layers(engine: str) -> dict[str, list[str]]:
    """Unit -> the layers (patches and the libr3 rewrite) that edit it."""
    out = {unit: list(ls) for unit, ls in rewrite_units(engine).items()}
    for patch_id, units in patch_units(engine).items():
        for unit in units:
            out.setdefault(unit, []).append(f"patch:{patch_id}")
    return {unit: sorted(set(ls)) for unit, ls in sorted(out.items())}


def shared_units(engine: str) -> dict[str, list[str]]:
    return {unit: ls for unit, ls in layers(engine).items() if len(ls) > 1}


def _data(engine: str, name: str) -> dict:
    path = _patches_root(engine) / name
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# ------------------------------------------------------------------------------------------------------- naming
_STRUCT = re.compile(r"\b(?:struct|class|union|enum\s+class|enum)\s+([A-Za-z_]\w*)\s*(?:\{|:|;)")
_USING = re.compile(r"\busing\s+([A-Za-z_]\w*)\s*=")
_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)", re.M)
_ATTR_FUNC = re.compile(r"__(?:global|device|host)__[^;{()]*?\b([A-Za-z_]\w*)\s*\(")
_CONSTEXPR = re.compile(r"\bconstexpr\b([^;{]*);")


def _declarators(statement: str) -> list[str]:
    """Names declared by ``type A = x, B = y`` (depth-0 commas of the statement after ``constexpr``)."""
    names, depth, start = [], 0, 0
    parts = []
    for i, ch in enumerate(statement):
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(statement[start:i])
            start = i + 1
    parts.append(statement[start:])
    for part in parts:
        m = re.search(r"([A-Za-z_]\w*)\s*(?:=|\[|$)", part.strip())
        if m:
            names.append(m.group(1))
    return names


def declared_names(text: str) -> set[str]:
    names = set(_STRUCT.findall(text)) | set(_USING.findall(text)) | set(_DEFINE.findall(text)) | set(_ATTR_FUNC.findall(text))
    for statement in _CONSTEXPR.findall(text):
        names.update(_declarators(statement))
    keywords = {"if", "for", "while", "return", "sizeof", "static_assert"}
    return {n for n in names if n not in keywords}


def introduced_names(added: str, anchor: str) -> set[str]:
    """Names the added text declares that its anchor did not already carry."""
    return declared_names(added) - set(re.findall(r"[A-Za-z_]\w*", anchor))


def _rewrite_blocks(path: Path):
    """(find text, replace text) of each block of a .rw file."""
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(r"^@@ find[^\n]*\n(.*?)^@@ replace\n(.*?)^@@ end", text, re.M | re.S):
        yield m.group(1), m.group(2)


def introduced(engine: str) -> list[tuple[str, str, str]]:
    """(layer, source, name) for every declaration the patches' libr4d edits and the rewrites introduce."""
    out = []
    for descriptor, patches in _implementations(engine):
        for p in patches:
            if not p.path.startswith("libr4d/"):
                continue
            for edit in p.edits:
                for name in sorted(introduced_names(edit.text, edit.anchor)):
                    out.append((f"patch:{descriptor.patch_id}", f"{p.path}:{edit.id}", name))
    for path in rewrite_files(engine):
        for find, replace in _rewrite_blocks(path):
            for name in sorted(introduced_names(replace, find)):
                out.append((REWRITE_LAYER, path.name, name))
    return out


def suffix_ok(layer: str, name: str) -> bool:
    if layer == REWRITE_LAYER:
        return bool(REWRITE_SUFFIX.search(name))
    patch_id = layer.split(":", 1)[1].split("_", 1)[0]
    return bool(re.search(rf"_gfx\d+_{patch_id}$", name))


# ---------------------------------------------------------------------------------------------------- problems
def compare(derived, declared, shared, sharing, introduced_names_, allowed) -> list[str]:
    """The checks over plain data (the fixtures' entry point)."""
    out = []
    for patch_id in sorted(derived):
        if tuple(derived[patch_id]) != tuple(declared.get(patch_id, ())):
            out.append(f"{patch_id}: patch.toml units {list(declared.get(patch_id, ()))} must equal the units its edits touch "
                       f"{list(derived[patch_id])}")
    for unit, ls in shared.items():
        if unit not in sharing:
            out.append(f"unit {unit} is edited by {len(ls)} layers ({', '.join(ls)}): record it, with the reason, in "
                       f"patches/{SHARING_FILE} or move one edit")
    for unit in sharing:
        if unit not in shared:
            out.append(f"{SHARING_FILE}: unit {unit} is no longer shared; remove the entry")
    seen = set()
    for layer, source, name in introduced_names_:
        seen.add((layer, name))
        if not suffix_ok(layer, name) and (layer, name) not in allowed:
            out.append(f"{layer}: {name} (introduced in {source}) lacks the _<arch>_<layer> suffix; rename it or record the "
                       f"exception in patches/{EXCEPTIONS_FILE}")
    for layer, name in sorted(set(allowed) - seen):
        out.append(f"{EXCEPTIONS_FILE}: {layer} {name} is no longer introduced; remove the entry")
    return out


def problems(engine: str) -> list[str]:
    allowed = {(e["layer"], e["name"]) for e in _data(engine, EXCEPTIONS_FILE).get("exception", [])}
    return compare(patch_units(engine), declared_units(engine), shared_units(engine),
                   _data(engine, SHARING_FILE).get("unit", {}), introduced(engine), allowed)
