"""PA25: Audit focal evidence/G4 and exact-composition G7 for the seven promoted/touched patches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bigcherry.core import paths
from bigcherry.core import config as core_config
from bigcherry.campaign import resolution as campaign_resolution
from bigcherry.patch import gates as patch_gates
from bigcherry.patch import registry as patch_registry
from bigcherry.patch import patchset
from bigcherry.patch import catalog as patch_catalog
from bigcherry.source.workspace import UpstreamRepository

FOCAL_PATCHES = ["1225", "0840", "1224", "1200", "1001", "1002", "1232"]

# Named experiment contexts: (experiment_name, base_source)
EXPERIMENT_CONTEXTS = [
    ("hi155-hybrid-allreduce-gate", "bigcherry-tuning"),
    ("hi18-reduce-probe", "bigcherry-tuning"),
    ("rd19-only", "bigcherry-tuning"),
    ("hip-internal-allreduce-only", "bigcherry-tuning"),
]

# Focal overlay contexts: (patch_id, base_source)
FOCAL_OVERLAY_CONTEXTS = [
    ("1002", "bigcherry"),
    ("1232", "bigcherry"),
]


def run_audit() -> dict:
    cfg = core_config.load(paths.RECIPES)
    registry = patch_registry.load_registry(paths.PATCHES)
    modules = patchset.catalog(directory=paths.PATCHES)

    source_root = paths.llama_root(None)
    repository = UpstreamRepository(source_root)
    resolved_base_revision = repository.resolve_ref(cfg.pinned)
    target_revision = repository.resolve_ref("HEAD")

    matrix: dict = {
        "schema_version": 1,
        "audit": "PA25",
        "bigcherry_revision": target_revision,
        "llama_cpp_revision": resolved_base_revision,
        "patch_catalog_snapshot": str(paths.PATCH_CATALOG),
        "evidence_root": "None",
        "patches": {},
    }

    # Step 1: Resolve full patch IDs
    patch_id_map: dict[str, str] = {}
    for patch_id in FOCAL_PATCHES:
        full_patch_id = None
        for module in modules:
            if module.patch_id.startswith(patch_id):
                full_patch_id = module.patch_id
                break
        if full_patch_id is None:
            matrix["patches"][patch_id] = {"error": "not found in catalog"}
            continue
        patch_id_map[patch_id] = full_patch_id

    # Step 2: Evaluate G4/G7 for each patch in each context
    for patch_id, full_patch_id in patch_id_map.items():
        descriptor = registry.get(full_patch_id)
        evidence_result = {"state": descriptor.state, "authority": "patch.catalog"}

        g4_g7_results: dict = {}

        # Named experiment contexts
        for context_name, base_source in EXPERIMENT_CONTEXTS:
            if context_name not in cfg.experiments:
                g4_g7_results[context_name] = {
                    "state": "NOT_EVALUATED",
                    "reason": "context not found in recipes",
                }
                continue

            try:
                selection = campaign_resolution.resolve_canonical_selection(
                    base_source,
                    cfg,
                    modules,
                    catalog_directory=paths.PATCHES,
                    experiment=context_name,
                )
                if full_patch_id not in selection.patch_ids:
                    g4_g7_results[context_name] = {
                        "state": "NOT_EVALUATED",
                        "reason": "patch not in selection",
                    }
                    continue

                composition = patchset.resolve_exact(
                    tuple(selection.patch_ids),
                    directory=paths.PATCHES,
                    allow_rejected=False,
                )
                g4_g7_results[context_name] = _evaluate_gates(
                    descriptor,
                    composition,
                    cfg,
                    base_source,
                    context_name,
                    resolved_base_revision,
                    target_revision,
                    source_root,
                    selection,
                    modules,
                )
            except Exception as exc:
                g4_g7_results[context_name] = {"state": "ERROR", "reason": str(exc)}

        # Focal overlay contexts (only for 1002 and 1232)
        for overlay_patch_id, overlay_source in FOCAL_OVERLAY_CONTEXTS:
            context_label = f"focal-overlay-{overlay_patch_id}"
            if patch_id != overlay_patch_id:
                g4_g7_results[context_label] = {
                    "state": "NOT_APPLICABLE",
                    "reason": "focal overlay is for a different patch",
                }
                continue
            try:
                selection = campaign_resolution.resolve_canonical_selection(
                    overlay_source,
                    cfg,
                    modules,
                    catalog_directory=paths.PATCHES,
                    focal_overlay_patch_id=full_patch_id,
                )
                if full_patch_id not in selection.patch_ids:
                    g4_g7_results[context_label] = {
                        "state": "NOT_EVALUATED",
                        "reason": "patch not in focal overlay selection",
                    }
                    continue

                composition = patchset.resolve_exact(
                    tuple(selection.patch_ids),
                    directory=paths.PATCHES,
                    allow_rejected=False,
                )
                g4_g7_results[context_label] = _evaluate_gates(
                    descriptor,
                    composition,
                    cfg,
                    overlay_source,
                    context_label,
                    resolved_base_revision,
                    target_revision,
                    source_root,
                    selection,
                    modules,
                )
            except Exception as exc:
                g4_g7_results[context_label] = {"state": "ERROR", "reason": str(exc)}

        matrix["patches"][full_patch_id] = {
            "evidence": evidence_result,
            "g4_g7": g4_g7_results,
            "blockers": [],
            "artifact_hashes": {},
        }

    return matrix


def _evaluate_gates(
    descriptor,
    composition,
    cfg,
    source_name,
    context_name,
    resolved_base_revision,
    target_revision,
    source_root,
    selection,
    modules,
) -> dict:
    """Evaluate G4 and G7 for a single context. G4 and G7 are independent."""
    context = patch_gates.GateContext(
        descriptor=descriptor,
        composition=composition,
        pinned_ref=cfg.pinned,
        intent=patch_gates.GateIntent.PROMOTE,
        patch_context=patch_catalog.PatchContext(
            backend="agnostic", source=source_name
        ),
        patches_dir=paths.PATCHES,
        resolved_base_revision=resolved_base_revision,
        catalog_path=paths.PATCH_CATALOG,
        evidence_root=None,
        external_sources_path=paths.EXTERNAL_SOURCES,
        validation_baseline_path=paths.VALIDATION_PACKAGE_GRANDFATHER,
        dispositions_dir=paths.DISPOSITIONS,
        source_root=source_root,
        expected_selector=selection.identity,
        allow_legacy_grandfather=True,
        catalog_states={m.patch_id: m.state for m in modules},
        recipe_patch_ids=frozenset(m.patch_id for m in composition.modules),
        target_revision=target_revision,
    )
    results = patch_gates.evaluate_patch_gates(context)
    g4 = next((r for r in results if r.id == patch_gates.GateId.G4), None)
    g7 = next((r for r in results if r.id == patch_gates.GateId.G7), None)
    return {
        "state": "EVALUATED",
        "g4": {
            "status": g4.status.value if g4 else "NOT_RUN",
            "phase": g4.phase if g4 else None,
            "authority": g4.authority if g4 else None,
            "detail": list(g4.detail) if g4 else [],
        },
        "g7": {
            "status": g7.status.value if g7 else "NOT_RUN",
            "phase": g7.phase if g7 else None,
            "authority": g7.authority if g7 else None,
            "detail": list(g7.detail) if g7 else [],
        },
        "selector": selection.identity.to_payload(),
        "composition": [m.patch_id for m in composition.modules],
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
