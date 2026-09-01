"""Machine-readable migration assumption audit (BC01)."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from typing import cast

from .core import config as campaign_config, paths
from .release import pin_status
from .patch import patchset
from .core.context import ProjectContext


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
    if version != 2:
        raise ValueError(
            f"{context.config_path}: unsupported recipes.toml version {version!r} "
            "-- only version 2 ([source.*]/[patch-set.*]) is supported"
        )
    recipes_report: list[dict[str, object]] = []
    loaded = campaign_config.load(context.config_path)
    for name, source in sorted(loaded.sources.items()):
        recipes_report.append(
            {
                "name": name,
                "schema": "v2",
                "ref": source.ref,
                "overlay": source.overlay,
                "patch_sets": list(source.patch_sets),
                "source_plan_status": "exact-patch-sets",
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
            "pinned_sha": _resolve_upstream(
                paths.llama_root(), str(raw.get("pinned", ""))
            ),
            "host_local_object_repo": str(context.upstream_repo),
        },
        "patch_catalog": patch_rows,
        "classification": {
            "framework_candidates": [
                item.patch_id for item in catalog if item.state == "validated"
            ],
            "validated_core_group": [
                item.patch_id
                for item in catalog
                if item.group == "core" and item.state == "validated"
            ],
            "validated_noncore_group": [
                item.patch_id
                for item in catalog
                if item.group != "core" and item.state == "validated"
            ],
            "promoted_enhancements": [],
            "classification_status": "owner-review-required",
            "reason": "validated state/group alone does not establish enhancement promotion",
        },
        "source_plans": recipes_report,
        "pin_status": _pin_status_section(context),
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


def _pin_status_section(context: ProjectContext) -> dict[str, object]:
    """RE48: the local pin verdict as a diagnostic section.

    Append-only and informational: doctor's exit code is unchanged (0), and
    the gate is `pin-status --strict` / `--complete`, not doctor."""
    repo_paths = pin_status.RepoPaths(
        repo_root=context.project_root,
        llama_root=paths.llama_root(),
        releases_dir=context.project_root / "releases",
        artifacts_dir=context.project_root / "artifacts",
    )
    status = pin_status.local_status(repo_paths)
    return {
        "verdict": status.verdict,
        "pinned_ref": status.pinned_ref,
        "pinned_sha": status.pinned_sha,
        "vendor_head": status.vendor_head,
        "marker_state": status.marker_state,
        "reasons": list(status.reasons),
    }


def main(*, as_json: bool = False, context: ProjectContext | None = None) -> int:
    report = build_report(context)
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        config_report = cast("dict[str, object]", report["config"])
        patch_rows = cast("list[object]", report["patch_catalog"])
        print(f"BigCherry revision: {report['bigcherry_revision']}")
        print(
            f"Config: v{config_report['version']} pinned={config_report['pinned']}"
        )
        print(f"Patch modules: {len(patch_rows)}")
        print("Promoted enhancement set: none (owner/reviewer classification required)")
        print(
            "bigcherry-native vs bigcherry: legacy selectors equal; no source delta claimed"
        )
        pin_status_report = cast("dict[str, object]", report["pin_status"])
        verdict = pin_status_report["verdict"]
        vendor = str(pin_status_report["vendor_head"] or "none")[:12]
        print(
            f"Pin status: {verdict} "
            f"(pin {pin_status_report['pinned_ref']} vs vendor {vendor})"
        )
    return 0
