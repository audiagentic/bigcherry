"""Pure exact patch-set and campaign-lane resolution (BC04 boundary)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from . import config, patchset


class ResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedPatchSet:
    name: str
    module_ids: tuple[str, ...]
    module_hashes: tuple[tuple[str, str], ...]
    classification: str
    required_state: str | None
    patch_set_id: str


@dataclass(frozen=True)
class ResolvedLane:
    name: str
    source_name: str
    patch_set: ResolvedPatchSet
    promoted_enhancements: tuple[str, ...]


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(b"bigcherry/patch-set/v1\0" + encoded, digest_size=16).hexdigest()


def resolve_patch_set(
    name: str,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    extra_patch_ids: tuple[str, ...] = (),
    required_state_override: str | None = None,
    classification: str = "base",
) -> ResolvedPatchSet:
    if name == "all":
        raise ResolutionError("'all' is not a valid production patch-set")
    if name not in cfg.patch_sets:
        raise ResolutionError(f"unknown patch set {name!r}")
    declared = cfg.patch_sets[name]
    ids = tuple(declared.patches) + tuple(extra_patch_ids)
    if len(set(ids)) != len(ids):
        raise ResolutionError("resolved patch set contains duplicate module IDs")
    selected = patchset.resolve_exact(
        ids,
        directory=(catalog[0].path.parent if catalog else None),
        required_state=required_state_override or declared.required_state,
    )
    by_id = {module.patch_id: module for module in catalog}
    if set(by_id) != {module.patch_id for module in patchset.catalog()}:
        raise ResolutionError("catalog argument does not match the physical patch catalog")
    # resolve_exact uses the project catalog; ensure the passed catalog supplies
    # identical content identities before exposing the result.
    for module in selected.modules:
        supplied = by_id.get(module.patch_id)
        if supplied is None or supplied.content_hash != module.content_hash:
            raise ResolutionError(f"catalog identity mismatch for {module.patch_id}")
    module_ids = tuple(module.patch_id for module in selected.modules)
    module_hashes = tuple((module.patch_id, module.content_hash) for module in selected.modules)
    identity = {
        "schema_version": 1,
        "name": name,
        "required_state": required_state_override or declared.required_state,
        "modules": module_hashes,
        "classification": classification,
    }
    return ResolvedPatchSet(
        name=name,
        module_ids=module_ids,
        module_hashes=module_hashes,
        classification=classification,
        required_state=identity["required_state"],
        patch_set_id=_digest(identity),
    )


def resolve_lane(
    source_name: str,
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    experiment: str | None = None,
) -> ResolvedLane:
    if source_name not in cfg.sources:
        raise ResolutionError(f"unknown source {source_name!r}")
    source = cfg.sources[source_name]
    if not source.patch_sets:
        empty = ResolvedPatchSet("empty", (), (), "upstream", None, _digest({"modules": []}))
        return ResolvedLane(source_name, source_name, empty, ())
    if experiment is not None and experiment not in cfg.experiments:
        raise ResolutionError(f"unknown experiment {experiment!r}")
    if len(source.patch_sets) != 1:
        # The shipped enhanced set is empty, but multiple named sets must be
        # resolved as one immutable ordered set rather than label-concatenated.
        first = cfg.patch_sets[source.patch_sets[0]]
        merged = tuple(item for name in source.patch_sets for item in cfg.patch_sets[name].patches)
        temporary = config.PatchSet("__merged__", merged, first.required_state)
        patched = dict(cfg.patch_sets)
        patched[temporary.name] = temporary
        cfg = config.Config(cfg.pinned, patched, cfg.sources, cfg.builds, cfg.platforms, cfg.experiments, cfg.path)
        base_name = temporary.name
    else:
        base_name = source.patch_sets[0]
    resolved = resolve_patch_set(
        base_name, cfg, catalog, classification="experimental" if experiment else "base"
    )
    if experiment:
        extra = cfg.experiments[experiment].patches
        extra_selection = patchset.resolve_exact(
            extra,
            directory=(catalog[0].path.parent if catalog else None),
            required_state=None,
        )
        if set(resolved.module_ids) & {module.patch_id for module in extra_selection.modules}:
            raise ResolutionError("experiment repeats a base patch module")
        modules = tuple(sorted(
            [
                *[next(module for module in catalog if module.patch_id == patch_id)
                  for patch_id in resolved.module_ids],
                *extra_selection.modules,
            ], key=lambda module: (module.order, module.patch_id)
        ))
        module_ids = tuple(module.patch_id for module in modules)
        module_hashes = tuple((module.patch_id, module.content_hash) for module in modules)
        identity = {
            "schema_version": 1,
            "name": resolved.name,
            "required_state": resolved.required_state,
            "modules": module_hashes,
            "classification": "experimental",
        }
        resolved = ResolvedPatchSet(
            name=resolved.name,
            module_ids=module_ids,
            module_hashes=module_hashes,
            classification="experimental",
            required_state=resolved.required_state,
            patch_set_id=_digest(identity),
        )
    return ResolvedLane(
        name=f"{source_name}+{experiment}" if experiment else source_name,
        source_name=source_name,
        patch_set=resolved,
        promoted_enhancements=tuple(
            cfg.patch_sets["validated-enhancements"].patches
        ) if "validated-enhancements" in cfg.patch_sets else (),
    )
