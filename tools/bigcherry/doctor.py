"""Machine-readable migration assumption audit (BC01)."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from . import patchset, paths, recipes
from .context import ProjectContext


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _resolve_upstream(root: Path, ref: str) -> str | None:
    if not (root / ".git").exists():
        return None
    return _git(root, "rev-parse", f"{ref}^{{commit}}")


def build_report(context: ProjectContext | None = None) -> dict[str, object]:
    context = context or ProjectContext.resolve()
    raw = tomllib.loads(context.config_path.read_text(encoding="utf-8"))
    version = raw.get("version", 1)
    catalog = patchset.catalog(context.patches_root)
    patch_rows = [
        {
            "patch_id": item.patch_id,
            "order": item.order,
            "group": item.group,
            "state": item.state,
            "upstream": item.upstream,
            "content_hash": item.content_hash,
            "requires": list(item.requires),
            "conflicts": list(item.conflicts),
        }
        for item in catalog
    ]
    recipes_report: list[dict[str, object]] = []
    if version == 1:
        legacy = recipes.load_config(context.config_path)
        for name, recipe in sorted(legacy.recipes.items()):
            recipes_report.append(
                {
                    "name": name,
                    "schema": "v1",
                    "ref": recipe.ref,
                    "groups": None if recipe.groups is None else sorted(recipe.groups),
                    "states": None if recipe.states is None else sorted(recipe.states),
                    "source_plan_status": "legacy-selector",
                }
            )
    report: dict[str, object] = {
        "schema_version": 1,
        "project": str(context.project_root),
        "bigcherry_revision": _git(context.project_root, "rev-parse", "HEAD"),
        "bigcherry_dirty": bool(_git(context.project_root, "status", "--porcelain")),
        "config": {
            "path": str(context.config_path),
            "version": version,
            "pinned": raw.get("pinned"),
        },
        "upstream": {
            "legacy_checkout": str(paths.llama_root()),
            "pinned_sha": _resolve_upstream(paths.llama_root(), str(raw.get("pinned", ""))),
            "host_local_object_repo": str(context.upstream_repo),
        },
        "patch_catalog": patch_rows,
        "classification": {
            "framework_candidates": [item.patch_id for item in catalog if item.group == "core" and item.state == "validated"],
            "promoted_enhancements": [],
            "classification_status": "owner-review-required",
            "reason": "validated state/group alone does not establish enhancement promotion",
        },
        "source_plans": recipes_report,
        "known_aliases": {
            "bigcherry-native_vs_bigcherry": {
                "legacy_selector_equal": True,
                "effective_source_identity": "not-yet-materialized",
                "reporting_rule": "do-not-claim-a-source-difference",
            }
        },
        "roots": {
            "work": str(context.work_root),
            "artifacts": str(context.artifacts_root),
            "overlay": str(context.overlay_root),
            "patches": str(context.patches_root),
        },
    }
    return report


def main(*, as_json: bool = False, context: ProjectContext | None = None) -> int:
    report = build_report(context)
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"BigCherry revision: {report['bigcherry_revision']}")
        print(f"Config: v{report['config']['version']} pinned={report['config']['pinned']}")
        print(f"Patch modules: {len(report['patch_catalog'])}")
        print("Promoted enhancement set: none (owner/reviewer classification required)")
        print("bigcherry-native vs bigcherry: legacy selectors equal; no source delta claimed")
    return 0
