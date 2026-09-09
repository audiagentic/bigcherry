#!/usr/bin/env python3
"""Seed v3 review manifests from the v2 classification map.

This is deliberately a draft generator: continuing predecessors and successor
specs are marked unapproved and require human semantic review before IDs are
allocated or lifecycle state changes are made.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TERMINAL = {"completed", "superseded", "deprecated"}
CAPABILITIES = {"build", "run", "patching", "tuning"}
MIGRATION_ID = "capability-rebaseline-v3-2026-09"


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prefix(namespace: str) -> str:
    parts = namespace.split("-")
    value = parts[0][0] + "".join(part[0] for part in parts[1:] if part)
    return value.upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--v2-map", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    inventory = read_csv(work / "PLAN_INVENTORY.csv")
    v2 = {row["plan_id"]: row for row in read_csv(args.v2_map.resolve())}

    namespace_rows: dict[str, dict[str, str]] = {}
    for row in v2.values():
        namespace = row["target_plan_frontmatter"]
        capability = row["capability"]
        if capability not in CAPABILITIES:
            continue
        namespace_rows.setdefault(
            namespace,
            {
                "capability": capability,
                "plan_namespace": namespace,
                "id_prefix": prefix(namespace),
                "scope": f"{capability} ownership for {namespace.removeprefix(capability + '-')}",
                "approved": "true",
            },
        )
    write_csv(work / "NAMESPACES.csv", ["capability", "plan_namespace", "id_prefix", "scope", "approved"], list(namespace_rows.values()))

    dispositions: list[dict[str, str]] = []
    successors: list[dict[str, str]] = []
    lineage: list[dict[str, str]] = []
    spec_dir = work / "successor-specs"
    spec_dir.mkdir(exist_ok=True)
    for item in inventory:
        source_id = item["source_id"]
        mapped = v2.get(source_id)
        if not mapped:
            dispositions.append({**{k: item.get(k, "") for k in ("source_id", "source_plan", "source_path", "source_state", "source_hash")}, "disposition": "retain-history", "final_state": item["source_state"], "capability": "", "target_namespace": "", "successor_keys": "", "reason": "Terminal historical item outside the active capability move map.", "scope_carry_forward": "", "scope_retired": "", "reviewed_by": "draft-seed", "approved": "true"})
            continue
        if item["source_state"] == "completed":
            dispositions.append({**{k: item.get(k, "") for k in ("source_id", "source_plan", "source_path", "source_state", "source_hash")}, "disposition": "retire-completed", "final_state": "completed", "capability": "", "target_namespace": "", "successor_keys": "", "reason": "Already completed hygiene item; no successor required.", "scope_carry_forward": "", "scope_retired": "Completed before the new planning epoch.", "reviewed_by": "draft-seed", "approved": "true"})
            continue
        namespace = mapped["target_plan_frontmatter"]
        key = f"{namespace}-{source_id.lower()}"
        dispositions.append({**{k: item.get(k, "") for k in ("source_id", "source_plan", "source_path", "source_state", "source_hash")}, "disposition": "successor", "final_state": "superseded", "capability": mapped["capability"], "target_namespace": namespace, "successor_keys": key, "reason": "Continuing work moves into a capability-native planning epoch; semantic review still required.", "scope_carry_forward": item["title"], "scope_retired": "", "reviewed_by": "draft-seed", "approved": "false"})
        successors.append({"successor_key": key, "target_namespace": namespace, "id_prefix": prefix(namespace), "allocated_id": "", "title": item["title"], "work": item["work"], "skill": item["skill"], "priority": item["priority"], "acceptance_boundary": "Capability-native successor for the still-valid scope; refine during semantic review.", "spec_path": f"successor-specs/{key}.md", "approved": "false"})
        lineage.append({"predecessor_id": source_id, "successor_key": key, "relation": "supersedes", "scope_summary": item["title"]})
        (spec_dir / f"{key}.md").write_text(
            f"# {item['title']}\n\nSuccessor key: {key}\nTarget plan: {namespace}\nPredecessor(s): {source_id}\nMigration: {MIGRATION_ID}\n\n## Description\n\nDraft successor seeded from the frozen predecessor. Requires semantic review.\n\n## Inherited evidence and constraints\n\nHistorical evidence remains attributed to {source_id}.\n\n## Acceptance Criteria\n\nDefine an independent acceptance boundary during review.\n\n## Notes\n\nSupersedes: {source_id}\nMigration: {MIGRATION_ID}\n",
            encoding="utf-8",
        )

    write_csv(work / "DISPOSITIONS.csv", ["source_id", "source_plan", "source_path", "source_state", "source_hash", "disposition", "final_state", "capability", "target_namespace", "successor_keys", "reason", "scope_carry_forward", "scope_retired", "reviewed_by", "approved"], dispositions)
    write_csv(work / "SUCCESSORS.csv", ["successor_key", "target_namespace", "id_prefix", "allocated_id", "title", "work", "skill", "priority", "acceptance_boundary", "spec_path", "approved"], successors)
    write_csv(work / "LINEAGE.csv", ["predecessor_id", "successor_key", "relation", "scope_summary"], lineage)

    disposition_by_id = {row["source_id"]: row for row in dispositions}
    inventory_by_id = {row["source_id"]: row for row in inventory}
    path_to_id = {row["source_path"]: row["source_id"] for row in inventory}
    successor_by_id = {
        row["source_id"]: row["successor_keys"]
        for row in dispositions
        if row["disposition"] in {"successor", "split", "merge"}
    }
    ref_path = work / "REFERENCE_DECISIONS.tsv"
    references = read_csv(ref_path, delimiter="\t") if ref_path.exists() else []
    for ref in references:
        target_id = ref["old_ref"] if ref["old_ref"] in inventory_by_id else path_to_id.get(ref["old_ref"], "")
        if not target_id and ref["old_ref"].endswith(".md"):
            basename = Path(ref["old_ref"]).stem
            if re.fullmatch(r"[A-Z]+[0-9]+", basename) and basename in inventory_by_id:
                target_id = basename
        target = disposition_by_id.get(target_id)
        source_id = ref.get("source_id", "") or path_to_id.get(ref.get("source_path", ""), "")
        source = inventory_by_id.get(source_id)
        if not target:
            if source and source["source_state"] in TERMINAL:
                ref["reference_kind"] = "historical_decision"
                ref["decision"] = "preserve"
                ref["rationale"] = "Draft classification: terminal predecessor records a historical supporting path."
                ref["semantic_class"] = "historical_provenance"
                ref["action"] = "preserve_predecessor"
            continue
        if target["disposition"] == "retain-history":
            ref["reference_kind"] = "historical_decision"
            ref["decision"] = "preserve"
            ref["rationale"] = "Draft classification: reference targets a retained historical predecessor."
            ref["semantic_class"] = "historical_provenance"
            ref["action"] = "preserve_predecessor"
            continue
        historical = (source and source["source_state"] in TERMINAL) or target["source_state"] in TERMINAL
        if historical or target["disposition"] == "retire-completed":
            ref["reference_kind"] = "historical_decision"
            ref["decision"] = "preserve"
            ref["rationale"] = "Draft classification: historical/terminal provenance remains attached to the predecessor."
            ref["semantic_class"] = "historical_provenance"
            ref["action"] = "preserve_predecessor"
        else:
            ref["reference_kind"] = "active_scope"
            ref["decision"] = "rewrite"
            ref["target_successor_key_or_id"] = successor_by_id.get(target_id, "")
            ref["rationale"] = "Draft classification: active forward reference follows the successor."
            ref["semantic_class"] = "active_scope"
            ref["action"] = "rewrite_to_successor"
    write_tsv(work / "REFERENCE_DECISIONS.tsv", ["occurrence_id", "source_path", "source_id", "line", "old_ref", "reference_kind", "decision", "target_successor_key_or_id", "rationale", "semantic_class", "action", "context_hash", "context"], references)
    print(f"namespaces={len(namespace_rows)} dispositions={len(dispositions)} successors={len(successors)} lineage={len(lineage)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
