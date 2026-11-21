"""Deterministic lexical patch-footprint inventory for declared source compositions.

No upstream checkout is modified or required. The patch registry's FilePatch/Edit
definitions are authoritative; regex anchors are counted as authored UTF-8 bytes,
not guessed match lengths or independently proven upstream source locations.
"""

from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from pathlib import Path

from ..core import engines, paths
from ..patch import patchset, registry


def composed_members(recipes: dict, modules: dict, *, engine: str) -> dict[str, tuple[str, ...]]:
    """Validate each declared source's exact patch IDs, without host build config."""
    if recipes.get("version") != 2:
        raise ValueError("footprint requires exact v2 source recipes")
    if recipes.get("engine") != engine:
        raise ValueError(f"recipes declare {recipes.get('engine')!r}, expected {engine!r}")
    if not recipes.get("pinned"):
        raise ValueError("recipes have no pinned upstream revision")
    sets = recipes.get("patch-set", {})
    result = {}
    for source_name, source in sorted(recipes.get("source", {}).items()):
        ids = []
        for set_name in source["patch-sets"]:
            if set_name not in sets:
                raise ValueError(f"source {source_name} references unknown patch-set {set_name}")
            ids.extend(sets[set_name]["patches"])
        # Duplicate membership is valid across patch-sets; only the exact union is composed.
        resolved = patchset.resolve_exact_from_catalog(
            engine, sorted(set(ids)), modules=modules
        )
        result[source_name] = tuple(module.patch_id for module in resolved.modules)
    if not result:
        raise ValueError("recipes declare no source compositions")
    return result


def inventory(
    *,
    engine: str,
    pin: str,
    pin_version: str,
    compositions: dict[str, tuple[str, ...]],
    definitions: dict[str, list],
) -> dict:
    """Compute the footprint from FilePatch objects (pure, testable, no I/O)."""
    membership: dict[str, list[str]] = defaultdict(list)
    for source_name, ids in sorted(compositions.items()):
        for patch_id in ids:
            membership[patch_id].append(source_name)
            if patch_id not in definitions:
                raise ValueError(f"{source_name} references unloaded patch {patch_id}")

    # A later package may patch a file created by an earlier package.
    patch_created_paths = {
        f.path for patch_id in membership for f in definitions[patch_id] if f.create
    }
    per_file: dict[str, dict] = {}
    rows: list[dict] = []
    for patch_id in sorted(membership):
        file_patches = definitions[patch_id]
        touched: set[str] = set()
        upstream: set[str] = set()
        created: set[str] = set()
        created_target_anchors = upstream_anchors = 0
        edits = anchors = anchor_sites = anchor_bytes = inserted_bytes = created_edits = 0
        for file_patch in file_patches:
            if file_patch.path in created and not file_patch.create or (
                file_patch.path in upstream and file_patch.create
            ):
                raise ValueError(f"{patch_id} mixes created and upstream target {file_patch.path}")
            touched.add(file_patch.path)
            if file_patch.create:
                created.add(file_patch.path)
            elif file_patch.path not in patch_created_paths:
                upstream.add(file_patch.path)
            if not file_patch.create and file_patch.path not in patch_created_paths and file_patch.edits:
                file_row = per_file.setdefault(
                    file_patch.path, {"path": file_patch.path, "patch_ids": set(), "anchors": 0,
                                      "anchor_sites": 0, "anchor_bytes": 0, "inserted_bytes": 0}
                )
                file_row["patch_ids"].add(patch_id)
            for edit in file_patch.edits:
                edits += 1
                inserted = len(edit.text.encode("utf-8"))
                inserted_bytes += inserted
                if file_patch.create:
                    created_edits += 1
                else:
                    anchors += 1
                    anchor_sites += edit.expect_matches
                    size = len(edit.anchor.encode("utf-8"))
                    anchor_bytes += size
                    if file_patch.path in patch_created_paths:
                        created_target_anchors += 1
                    else:
                        upstream_anchors += 1
                        file_row["anchors"] += 1
                        file_row["anchor_sites"] += edit.expect_matches
                        file_row["anchor_bytes"] += size
                        file_row["inserted_bytes"] += inserted
        rows.append({
            "patch": patch_id, "sources": membership[patch_id],
            "files_touched": sorted(touched), "upstream_files": sorted(upstream),
            "created_files": sorted(created), "edits": edits, "anchors": anchors,
            "upstream_anchors": upstream_anchors, "created_target_anchors": created_target_anchors,
            "anchor_sites": anchor_sites, "anchor_bytes": anchor_bytes,
            "inserted_bytes": inserted_bytes, "created_edits": created_edits,
        })
    rows.sort(key=lambda row: (-row["anchors"], -row["anchor_bytes"], row["patch"]))
    files = [
        {**row, "patch_ids": sorted(row["patch_ids"]), "patch_count": len(row["patch_ids"])}
        for row in per_file.values()
    ]
    files.sort(key=lambda row: (-row["patch_count"], -row["anchors"], row["path"]))
    return {
        "schema": 1, "engine": engine, "upstream_pin": pin, "upstream_version": pin_version,
        "method": (
            "Lexical FilePatch/Edit definitions for the union of exact declared source "
            "compositions. Anchors count non-create Edit entries; anchor_sites sum "
            "expect_matches (potential matches, not probed). anchor_bytes count raw "
            "regex UTF-8 bytes, inserted_bytes count Edit.text UTF-8 bytes (replacement "
            "body, not net diff). Created-file edits have no upstream anchor; "
            "non-create edits on patch-created files remain anchors, but not upstream anchors. "
            "Does not prove anchor applicability or semantics at the pin."
        ),
        "compositions": {name: list(ids) for name, ids in sorted(compositions.items())},
        "totals": {
            "composed_patches": len(rows), "edits": sum(row["edits"] for row in rows),
            "anchors": sum(row["anchors"] for row in rows),
            "upstream_anchors": sum(row["upstream_anchors"] for row in rows),
            "created_target_anchors": sum(row["created_target_anchors"] for row in rows),
            "anchor_sites": sum(row["anchor_sites"] for row in rows),
            "anchor_bytes": sum(row["anchor_bytes"] for row in rows),
            "inserted_bytes": sum(row["inserted_bytes"] for row in rows),
            "created_edits": sum(row["created_edits"] for row in rows),
            "upstream_files": len(files),
        },
        "patches": rows, "upstream_files": files,
    }


def collect(engine: str, *, repo_root: Path = paths.REPO_ROOT) -> dict:
    engine_cfg = engines.load(repo_root, engine)
    recipes_path = engine_cfg.recipes_path(repo_root)
    recipes = tomllib.loads(recipes_path.read_text(encoding="utf-8"))
    root = engine_cfg.patches_root(repo_root)
    modules = {module.patch_id: module for module in patchset.catalog(engine, root)}
    compositions = composed_members(recipes, modules, engine=engine)
    used = set().union(*(set(ids) for ids in compositions.values()))
    patch_registry = registry.load_registry(engine, root)
    definitions = {
        patch_id: registry.load_implementation(
            engine, patch_registry.get(patch_id), root=root,
            expected_digest=modules[patch_id].content_hash,
        )
        for patch_id in sorted(used)
    }
    return inventory(
        engine=engine, pin=recipes["pinned"],
        pin_version=recipes.get("pinned-version", ""),
        compositions=compositions, definitions=definitions,
    )


def markdown(report: dict) -> str:
    total = report["totals"]
    lines = [
        f"# {report['engine']} patch footprint — {report['upstream_version']}",
        "",
        f"Upstream pin: `{report['upstream_pin']}`.",
        "",
        (f"{total['composed_patches']} composed patches; **{total['anchors']} anchor edits** "
        f"({total['upstream_anchors']} upstream, {total['created_target_anchors']} on patch-created files; "
        f"{total['anchor_sites']} expected match sites); {total['edits']} total edits, "
        f"{total['created_edits']} in created files; {total['upstream_files']} upstream files."),
        "",
        ("Counts are lexical, not a measure of independent semantics, net code savings, "
        "or successful application. Regex anchor bytes and full replacement text bytes "
        "are counted as authored, not as actual upstream match or net diff bytes."),
        "",
        "## Patches (ranked by non-create anchor edits)",
        "",
        "| Patch | Upstream files | Created files | Anchors | Upstream anchors | Edits | Anchor bytes | Inserted bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["patches"]:
        lines.append(
            f"| `{row['patch']}` | {len(row['upstream_files'])} | "
            f"{len(row['created_files'])} | {row['anchors']} | {row['upstream_anchors']} | {row['edits']} | "
            f"{row['anchor_bytes']} | {row['inserted_bytes']} |"
        )
    lines.extend([
        "",
        "## Upstream files (ranked by number of anchoring patches)",
        "",
        "| Upstream file | Patches | Anchors | Anchor bytes | Patch IDs |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for row in report["upstream_files"]:
        ids = ", ".join(f"`{id_}`" for id_ in row["patch_ids"])
        lines.append(
            f"| `{row['path']}` | {row['patch_count']} | {row['anchors']} | "
            f"{row['anchor_bytes']} | {ids} |"
        )
    lines.extend([
        "", "## Composed source memberships", "",
        "| Source | Patch count |", "| --- | ---: |",
    ])
    for name, ids in report["compositions"].items():
        lines.append(f"| `{name}` | {len(ids)} |")
    lines.append("")
    return "\n".join(lines)


def outputs(report: dict) -> tuple[str, str]:
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n", markdown(report)


def publish(report: dict, directory: Path, *, check: bool) -> list[str]:
    """Return stale/missing output paths. In --check, never modify either file."""
    data = outputs(report)
    targets = (directory / "footprint.json", directory / "footprint.md")
    changed = [
        str(target) for target, expected in zip(targets, data)
        if not target.is_file() or target.read_text(encoding="utf-8") != expected
    ]
    if not check:
        directory.mkdir(parents=True, exist_ok=True)
        for target, expected in zip(targets, data):
            target.write_text(expected, encoding="utf-8")
    return changed
