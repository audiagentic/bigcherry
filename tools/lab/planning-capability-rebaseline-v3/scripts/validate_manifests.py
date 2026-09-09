#!/usr/bin/env python3
"""Fail-closed validator for BigCherry capability rebaseline v3 manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

CAPABILITIES = {"build", "run", "patching", "tuning"}
TERMINAL_STATES = {"completed", "superseded", "deprecated"}
DISPOSITIONS = {
    "successor",
    "split",
    "merge",
    "retire-completed",
    "retire-deprecated",
    "retain-history",
}
CONTINUING = {"successor", "split", "merge"}
RELATION_FOR = {"successor": "supersedes", "split": "split_from", "merge": "merged_from"}
REFERENCE_KINDS = {
    "active_dependency",
    "active_followup",
    "active_scope",
    "historical_evidence",
    "historical_review",
    "historical_decision",
    "ambiguous",
}
ACTIVE_REFERENCE_KINDS = {"active_dependency", "active_followup", "active_scope"}
HISTORICAL_REFERENCE_KINDS = {"historical_evidence", "historical_review", "historical_decision"}
REFERENCE_DECISIONS = {"preserve", "rewrite", "remove"}
SEMANTIC_CLASSES = {"active_scope", "historical_provenance", "identity_declaration", "literal_example", "unclassified"}
SEMANTIC_ACTIONS = {"rewrite_to_successor", "preserve_predecessor", "remove", "no_change", "unclassified"}
TRUE_VALUES = {"true", "1", "yes", "y"}
ID_RE = re.compile(r"^[A-Z][A-Z0-9]*\d+$")
LIFECYCLE_PATH_RE = re.compile(r"^docs/planning/(?:active|completed)/[^/]+/[A-Z]+[0-9]+\.md$")


class Problems:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finish(self) -> int:
        for msg in self.warnings:
            print(f"WARN: {msg}")
        for msg in self.errors:
            print(f"ERROR: {msg}")
        print(f"validation_errors={len(self.errors)}")
        print(f"validation_warnings={len(self.warnings)}")
        return 1 if self.errors else 0


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f, delimiter=delimiter)]


def truth(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES


def split_keys(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def find_repo(start: Path) -> Path:
    return Path(run_git(start, "rev-parse", "--show-toplevel").strip()).resolve()


def frozen_identity_set(repo: Path, source_commit: str) -> set[tuple[str, str, str, str]]:
    """Re-enumerate the frozen lifecycle universe independently of manifests."""
    tree = run_git(repo, "ls-tree", "-r", source_commit, "--", "docs/planning/active", "docs/planning/completed").splitlines()
    entries: list[tuple[str, str]] = []
    for line in tree:
        head, path = line.split("\t", 1)
        parts = head.split()
        if len(parts) == 3 and parts[1] == "blob" and LIFECYCLE_PATH_RE.fullmatch(path):
            entries.append((parts[2], path))
    batch = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input="".join(f"{oid}\n" for oid, _ in entries).encode("ascii"),
        text=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if batch.returncode:
        raise RuntimeError(f"git cat-file --batch failed: {batch.stderr.decode('utf-8', 'replace').strip()}")
    blob_by_oid: dict[str, str] = {}
    data = batch.stdout
    pos = 0
    for oid, _ in entries:
        end = data.find(b"\n", pos)
        if end < 0:
            raise RuntimeError("truncated git cat-file header")
        header = data[pos:end].decode("ascii")
        pos = end + 1
        fields = header.split()
        if len(fields) != 3 or fields[1] != "blob":
            raise RuntimeError(f"unexpected git cat-file response: {header}")
        size = int(fields[2])
        blob_by_oid[oid] = data[pos:pos + size].decode("utf-8")
        pos += size + 1
    identities: set[tuple[str, str, str, str]] = set()
    for oid, path in entries:
        # Match run_git(text=True)'s universal-newline normalization used by
        # generate_inventory.py before hashing frozen source content.
        text = blob_by_oid[oid].replace("\r\n", "\n")
        fm: dict[str, str] = {}
        lines = text.splitlines()
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if ":" in line and not line[:1].isspace():
                    key, value = line.split(":", 1)
                    fm[key.strip()] = value.strip().strip("'\"")
        if fm.get("id"):
            identities.add((fm["id"], path, fm.get("state", ""), hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return identities


def resolve_spec(pack: Path, work: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    in_work = work / candidate
    if in_work.exists():
        return in_work
    return pack / candidate


def normalize_path_ref(old_ref: str) -> str:
    return old_ref.replace("\\", "/")


def validate_source_snapshot(repo: Path, source_commit: str, inventory: list[dict[str, str]], p: Problems) -> None:
    try:
        resolved = run_git(repo, "rev-parse", f"{source_commit}^{{commit}}").strip()
    except Exception as exc:
        p.error(str(exc))
        return
    if resolved != source_commit:
        p.error(f"source commit resolves unexpectedly: {resolved}")
        return
    for row in inventory:
        path = row["source_path"]
        try:
            text = run_git(repo, "show", f"{source_commit}:{path}")
        except Exception as exc:
            p.error(f"cannot read frozen source {path}: {exc}")
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != row["source_hash"]:
            p.error(f"source hash mismatch for {row['source_id']} {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--phase", choices=["review", "preapply", "postapply"], required=True)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--skip-source-hash-check", action="store_true")
    ns = ap.parse_args()

    pack = ns.pack.resolve()
    work = ns.work.resolve()
    repo = find_repo(ns.repo.resolve())
    p = Problems()

    lock = json.loads((pack / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    migration_id = lock["migration_id"]
    source_commit = lock["source_commit"]

    pack_manifest_path = pack / "PACK_MANIFEST.json"
    pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    listed_files = {row.get("path", ""): row for row in pack_manifest.get("files", [])}
    actual_files = {
        path.relative_to(pack).as_posix(): path
        for path in pack.rglob("*")
        if path.is_file() and path != pack_manifest_path and "__pycache__" not in path.parts
    }
    if set(listed_files) != set(actual_files):
        p.error(f"PACK_MANIFEST.json file set mismatch (missing={len(set(actual_files) - set(listed_files))}, extra={len(set(listed_files) - set(actual_files))})")
    for rel, path in actual_files.items():
        row = listed_files.get(rel)
        if not row:
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if int(row.get("bytes", -1)) != len(data) or row.get("sha256") != digest:
            p.error(f"PACK_MANIFEST.json hash/size mismatch for {rel}")

    work_lock_path = work / "SOURCE_LOCK.json"
    if work_lock_path.exists():
        work_lock = json.loads(work_lock_path.read_text(encoding="utf-8"))
        if work_lock != lock:
            p.error("working SOURCE_LOCK.json differs from pack source lock")

    inventory = read_rows(work / "PLAN_INVENTORY.csv")
    lifecycle = read_rows(work / "LIFECYCLE_NORMALIZATION.csv")
    dispositions = read_rows(work / "DISPOSITIONS.csv")
    semantic_reviews = read_rows(work / "SEMANTIC_REVIEW.csv")
    namespaces = read_rows(work / "NAMESPACES.csv")
    successors = read_rows(work / "SUCCESSORS.csv")
    lineage = read_rows(work / "LINEAGE.csv")
    dep_rows = read_rows(work / "DEPENDENCY_REMAP.tsv", delimiter="\t")
    dep_candidates = read_rows(work / "DEPENDENCY_CANDIDATES.tsv", delimiter="\t")
    ref_rows = read_rows(work / "REFERENCE_DECISIONS.tsv", delimiter="\t")

    inventory_by_id = {r["source_id"]: r for r in inventory}
    if len(inventory_by_id) != len(inventory):
        p.error("PLAN_INVENTORY.csv contains duplicate source_id values")
    path_to_id = {normalize_path_ref(r["source_path"]): r["source_id"] for r in inventory}
    frozen = frozen_identity_set(repo, source_commit)
    supplied = {
        (r.get("source_id", ""), r.get("source_path", ""), r.get("source_state", ""), r.get("source_hash", ""))
        for r in inventory
    }
    if len(frozen) != 524:
        p.error(f"frozen lifecycle enumeration expected 524 items, found {len(frozen)}")
    frozen_payload = "".join(
        f"{source_id}\t{path}\t{state}\t{digest}\n"
        for source_id, path, state, digest in sorted(frozen)
    )
    frozen_digest = hashlib.sha256(frozen_payload.encode("utf-8")).hexdigest()
    if lock.get("inventory_count") != len(frozen):
        p.error("SOURCE_LOCK.json inventory_count does not match frozen enumeration")
    if lock.get("inventory_sha256") != frozen_digest:
        p.error("SOURCE_LOCK.json inventory_sha256 does not match frozen enumeration")
    if supplied != frozen:
        p.error(f"PLAN_INVENTORY.csv does not exactly match frozen lifecycle universe (missing={len(frozen - supplied)}, extra={len(supplied - frozen)})")

    lifecycle_by_id = {r["source_id"]: r for r in lifecycle}
    if len(lifecycle_by_id) != len(lifecycle):
        p.error("LIFECYCLE_NORMALIZATION.csv contains duplicate source_id values")
    if set(lifecycle_by_id) != set(inventory_by_id):
        p.error("LIFECYCLE_NORMALIZATION.csv must cover exactly the frozen inventory")
    semantic_by_id = {r["source_id"]: r for r in semantic_reviews}
    if len(semantic_by_id) != len(semantic_reviews):
        p.error("SEMANTIC_REVIEW.csv contains duplicate source_id values")
    if set(semantic_by_id) != set(inventory_by_id):
        p.error("SEMANTIC_REVIEW.csv must cover exactly the frozen inventory")
    required_review_fields = [
        "unfinished_work", "acceptance_boundary", "capability", "split_assessment",
        "overlap_assessment", "historical_evidence", "active_dependencies", "reference_notes",
        "reviewed_by",
    ]
    for source_id, row in semantic_by_id.items():
        inv = inventory_by_id.get(source_id)
        if inv:
            for field in ["source_state", "source_path", "source_hash", "title"]:
                if row.get(field, "") != inv.get(field, ""):
                    p.error(f"{source_id}: semantic review {field} differs from frozen inventory")
        for field in required_review_fields:
            if not row.get(field):
                p.error(f"{source_id}: semantic review field {field} is required")
        if ns.phase in {"review", "preapply", "postapply"} and not truth(row.get("approved", "")):
            p.error(f"{source_id}: semantic review is not approved for {ns.phase}")
    for source_id, row in lifecycle_by_id.items():
        if source_id not in inventory_by_id:
            continue
        expected_state = inventory_by_id[source_id].get("source_state", "")
        if row.get("source_state", "") != expected_state:
            p.error(f"{source_id}: lifecycle source_state differs from frozen inventory")
        if row.get("normalized_lifecycle") not in {"continuing", "terminal/no-successor", "adjudicate"}:
            p.error(f"{source_id}: invalid lifecycle normalization {row.get('normalized_lifecycle')!r}")
        if not row.get("reason"):
            p.error(f"{source_id}: lifecycle normalization reason is required")
        if ns.phase in {"review", "preapply", "postapply"}:
            if row.get("normalized_lifecycle") == "adjudicate" or not truth(row.get("approved", "")):
                p.error(f"{source_id}: lifecycle normalization is not approved for {ns.phase}")

    disp_by_id = {r["source_id"]: r for r in dispositions}
    if len(disp_by_id) != len(dispositions):
        p.error("DISPOSITIONS.csv contains duplicate source_id values")
    missing_disp = sorted(set(inventory_by_id) - set(disp_by_id))
    extra_disp = sorted(set(disp_by_id) - set(inventory_by_id))
    if missing_disp:
        p.error(f"missing disposition rows: {missing_disp}")
    if extra_disp:
        p.error(f"disposition rows not in frozen inventory: {extra_disp}")

    if not ns.skip_source_hash_check:
        validate_source_snapshot(repo, source_commit, inventory, p)

    # Freeze copied source metadata in dispositions.
    for source_id, inv in inventory_by_id.items():
        row = disp_by_id.get(source_id)
        if not row:
            continue
        for field in ["source_plan", "source_path", "source_state", "source_hash"]:
            if row.get(field, "") != inv.get(field, ""):
                p.error(f"{source_id}: disposition {field} differs from frozen inventory")

    # Namespace taxonomy must be explicit before successor creation.
    ns_by_name: dict[str, dict[str, str]] = {}
    prefix_counts: Counter[str] = Counter()
    for idx, row in enumerate(namespaces, 2):
        cap = row.get("capability", "")
        name = row.get("plan_namespace", "")
        prefix = row.get("id_prefix", "")
        if cap not in CAPABILITIES:
            p.error(f"NAMESPACES.csv:{idx}: invalid capability {cap!r}")
        if not name:
            p.error(f"NAMESPACES.csv:{idx}: plan_namespace is required")
            continue
        if name in ns_by_name:
            p.error(f"duplicate namespace {name}")
        ns_by_name[name] = row
        if cap in CAPABILITIES and not name.startswith(cap + "-"):
            p.error(f"namespace {name} must start with {cap}-")
        if not prefix or prefix.upper() != prefix or not re.fullmatch(r"[A-Z][A-Z0-9]*", prefix):
            p.error(f"namespace {name}: invalid id_prefix {prefix!r}")
        prefix_counts[prefix] += 1
        if not row.get("scope"):
            p.error(f"namespace {name}: scope is required")
        if not truth(row.get("approved", "")):
            p.error(f"namespace {name}: not approved")
    for prefix, count in prefix_counts.items():
        if prefix and count > 1:
            p.error(f"id_prefix {prefix} is reused by {count} namespaces")

    # Disposition semantics.
    for source_id, row in disp_by_id.items():
        disp = row.get("disposition", "")
        keys = split_keys(row.get("successor_keys", ""))
        if disp not in DISPOSITIONS:
            p.error(f"{source_id}: invalid/unreviewed disposition {disp!r}")
            continue
        if not truth(row.get("approved", "")):
            p.error(f"{source_id}: disposition is not approved")
        if not row.get("reviewed_by", ""):
            p.error(f"{source_id}: disposition reviewed_by is required")
        if not row.get("reason"):
            p.error(f"{source_id}: disposition reason is required")
        final_state = row.get("final_state", "")
        if disp in CONTINUING:
            if final_state != "superseded":
                p.error(f"{source_id}: continuing disposition requires final_state=superseded")
            if row.get("capability") not in CAPABILITIES:
                p.error(f"{source_id}: continuing disposition requires valid capability")
            if row.get("target_namespace") not in ns_by_name:
                p.error(f"{source_id}: unknown target_namespace {row.get('target_namespace')!r}")
            elif ns_by_name[row["target_namespace"]].get("capability") != row.get("capability"):
                p.error(f"{source_id}: capability/target_namespace mismatch")
            if not keys:
                p.error(f"{source_id}: continuing disposition has no successor_keys")
            if disp == "successor" and len(keys) != 1:
                p.error(f"{source_id}: successor disposition requires exactly one successor key")
            if disp == "split" and len(keys) < 2:
                p.error(f"{source_id}: split disposition requires >=2 successor keys")
        else:
            if disp == "retain-history" and inventory_by_id[source_id].get("source_state", "") not in TERMINAL_STATES:
                p.error(f"{source_id}: retain-history is only legal for frozen terminal states")
            if keys:
                p.error(f"{source_id}: retirement/history disposition must not list successor_keys")
            if row.get("capability") or row.get("target_namespace"):
                p.error(f"{source_id}: retirement/history disposition must not assign target capability/namespace")
            expected = {
                "retire-completed": "completed",
                "retire-deprecated": "deprecated",
                "retain-history": inventory_by_id[source_id].get("source_state", ""),
            }[disp]
            if final_state != expected:
                p.error(f"{source_id}: {disp} requires final_state={expected!r}, got {final_state!r}")

    # Lifecycle normalization and disposition are separate review decisions,
    # but they may not contradict one another once a lifecycle is adjudicated.
    for source_id, lifecycle_row in lifecycle_by_id.items():
        normalized = lifecycle_row.get("normalized_lifecycle", "")
        disposition = disp_by_id.get(source_id, {}).get("disposition", "")
        if ns.phase in {"preapply", "postapply"} and normalized == "terminal/no-successor" and disposition in CONTINUING:
            p.error(f"{source_id}: terminal lifecycle cannot retain continuing disposition {disposition}")
        if ns.phase in {"preapply", "postapply"} and normalized == "continuing" and disposition not in CONTINUING:
            p.error(f"{source_id}: continuing lifecycle requires continuing disposition, got {disposition!r}")

    # Successor catalog and specs.
    succ_by_key: dict[str, dict[str, str]] = {}
    allocated_ids: dict[str, str] = {}
    for idx, row in enumerate(successors, 2):
        key = row.get("successor_key", "")
        if not key:
            p.error(f"SUCCESSORS.csv:{idx}: successor_key required")
            continue
        if key in succ_by_key:
            p.error(f"duplicate successor_key {key}")
        succ_by_key[key] = row
        target_ns = row.get("target_namespace", "")
        if target_ns not in ns_by_name:
            p.error(f"successor {key}: unknown target_namespace {target_ns!r}")
        else:
            expected_prefix = ns_by_name[target_ns].get("id_prefix", "")
            if row.get("id_prefix", "") != expected_prefix:
                p.error(f"successor {key}: id_prefix must equal namespace prefix {expected_prefix}")
        for field in ["title", "acceptance_boundary", "spec_path"]:
            if not row.get(field):
                p.error(f"successor {key}: {field} required")
        if not truth(row.get("approved", "")):
            p.error(f"successor {key}: not approved")
        if ns.phase == "review" and row.get("allocated_id", ""):
            p.error(f"successor {key}: allocated_id must remain blank during review")
        spec_value = row.get("spec_path", "")
        if spec_value:
            spec = resolve_spec(pack, work, spec_value)
            if not spec.exists():
                p.error(f"successor {key}: missing spec {spec_value}")
            else:
                text = spec.read_text(encoding="utf-8")
                if migration_id not in text:
                    p.error(f"successor {key}: spec must contain migration id {migration_id}")
                if "Supersedes:" not in text:
                    p.error(f"successor {key}: spec must contain visible 'Supersedes:' lineage")
                if re.search(r"<[^>]+>", text):
                    p.error(f"successor {key}: spec still contains template placeholders")
        if ns.phase in {"preapply", "postapply"}:
            allocated = row.get("allocated_id", "")
            if not allocated:
                p.error(f"successor {key}: allocated_id required for {ns.phase}")
            elif not ID_RE.fullmatch(allocated):
                p.error(f"successor {key}: invalid allocated_id {allocated!r}")
            else:
                prefix = row.get("id_prefix", "")
                if prefix and not allocated.startswith(prefix):
                    p.error(f"successor {key}: allocated_id {allocated} does not start with {prefix}")
                if allocated in inventory_by_id:
                    p.error(f"successor {key}: allocated_id reuses predecessor ID {allocated}")
                if allocated in allocated_ids:
                    p.error(f"allocated_id {allocated} reused by {allocated_ids[allocated]} and {key}")
                allocated_ids[allocated] = key

    # Lineage graph.
    edges_by_pred: dict[str, list[dict[str, str]]] = defaultdict(list)
    preds_by_succ: dict[str, list[dict[str, str]]] = defaultdict(list)
    edge_keys: set[tuple[str, str]] = set()
    for idx, row in enumerate(lineage, 2):
        pred = row.get("predecessor_id", "")
        succ = row.get("successor_key", "")
        relation = row.get("relation", "")
        if pred not in inventory_by_id:
            p.error(f"LINEAGE.csv:{idx}: unknown predecessor {pred!r}")
        if succ not in succ_by_key:
            p.error(f"LINEAGE.csv:{idx}: unknown successor_key {succ!r}")
        if not row.get("scope_summary"):
            p.error(f"LINEAGE.csv:{idx}: scope_summary required")
        pair = (pred, succ)
        if pair in edge_keys:
            p.error(f"duplicate lineage edge {pred}->{succ}")
        edge_keys.add(pair)
        edges_by_pred[pred].append(row)
        preds_by_succ[succ].append(row)
        if pred in disp_by_id and disp_by_id[pred].get("disposition") in CONTINUING:
            expected_relation = RELATION_FOR[disp_by_id[pred]["disposition"]]
            if relation != expected_relation:
                p.error(f"{pred}->{succ}: relation {relation!r}, expected {expected_relation!r}")

    for source_id, row in disp_by_id.items():
        disp = row.get("disposition", "")
        expected_keys = set(split_keys(row.get("successor_keys", "")))
        actual_keys = {e.get("successor_key", "") for e in edges_by_pred.get(source_id, [])}
        if disp in CONTINUING:
            if expected_keys != actual_keys:
                p.error(f"{source_id}: successor_keys {sorted(expected_keys)} != lineage edges {sorted(actual_keys)}")
        elif actual_keys:
            p.error(f"{source_id}: non-continuing disposition has lineage edges")

    for key in succ_by_key:
        if not preds_by_succ.get(key):
            p.error(f"successor {key}: no predecessor lineage")

    for key, edges in preds_by_succ.items():
        merge_preds = [e["predecessor_id"] for e in edges if disp_by_id.get(e["predecessor_id"], {}).get("disposition") == "merge"]
        if merge_preds and len(edges) < 2:
            p.error(f"successor {key}: merge disposition requires >=2 predecessor edges")

    # Reference decisions: require decisions for references to changing predecessors.
    occurrence_ids: set[str] = set()
    expected_occurrences: set[tuple[str, str, str, str]] = set()
    plan_refs_path = work / "PLAN_REFERENCES.tsv"
    if plan_refs_path.exists():
        for ref in read_rows(plan_refs_path, delimiter="\t"):
            expected_occurrences.add((ref.get("occurrence_id", ""), ref.get("source_path", ""), ref.get("line", ""), ref.get("old_ref", "")))
    changing = {sid for sid, row in disp_by_id.items() if row.get("disposition") != "retain-history"}
    for idx, row in enumerate(ref_rows, 2):
        occurrence_id = row.get("occurrence_id", "")
        if not re.fullmatch(r"[0-9a-f]{16}", occurrence_id):
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: occurrence_id must be a 16-digit lowercase hash")
        elif occurrence_id in occurrence_ids:
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: duplicate occurrence_id {occurrence_id}")
        else:
            occurrence_ids.add(occurrence_id)
        source_id = row.get("source_id", "")
        source_path = normalize_path_ref(row.get("source_path", ""))
        if source_id not in inventory_by_id:
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: unknown source_id {source_id!r}")
        elif normalize_path_ref(inventory_by_id[source_id].get("source_path", "")) != source_path:
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: source_id/source_path mismatch for {source_id}")
        occurrence_key = (occurrence_id, row.get("source_path", ""), row.get("line", ""), row.get("old_ref", ""))
        if expected_occurrences and occurrence_key not in expected_occurrences:
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: occurrence is not present in PLAN_REFERENCES.tsv")
        semantic_class = row.get("semantic_class", "")
        action = row.get("action", "")
        if semantic_class not in SEMANTIC_CLASSES:
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: invalid semantic_class {semantic_class!r}")
        if action not in SEMANTIC_ACTIONS:
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: invalid action {action!r}")
        context_hash = row.get("context_hash", "")
        if not re.fullmatch(r"[0-9a-f]{64}", context_hash):
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: context_hash must be sha256")
        elif hashlib.sha256(row.get("context", "").encode("utf-8")).hexdigest() != context_hash:
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: context_hash does not match context")
        old_ref = normalize_path_ref(row.get("old_ref", ""))
        target_id = old_ref if old_ref in inventory_by_id else path_to_id.get(old_ref, "")
        if not target_id or target_id not in changing:
            continue
        if semantic_class == "unclassified" or action == "unclassified":
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: semantic classification is unresolved for {old_ref}")
            continue
        kind = row.get("reference_kind", "")
        decision = row.get("decision", "")
        if kind not in REFERENCE_KINDS or kind == "ambiguous":
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: unresolved reference kind for {old_ref}")
            continue
        if decision not in REFERENCE_DECISIONS:
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: unresolved decision for {old_ref}")
            continue
        if action == "rewrite_to_successor" and decision != "rewrite":
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: rewrite_to_successor must use decision=rewrite")
        if action == "preserve_predecessor" and decision != "preserve":
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: preserve_predecessor must use decision=preserve")
        target_disp = disp_by_id[target_id].get("disposition", "")
        if target_disp in CONTINUING:
            if kind in ACTIVE_REFERENCE_KINDS and decision != "rewrite":
                p.error(f"REFERENCE_DECISIONS.tsv:{idx}: active reference to continuing {target_id} must rewrite")
            if kind in HISTORICAL_REFERENCE_KINDS and decision != "preserve":
                p.error(f"REFERENCE_DECISIONS.tsv:{idx}: historical reference to {target_id} must preserve predecessor")
        if target_disp == "retire-deprecated" and kind in ACTIVE_REFERENCE_KINDS and decision == "preserve":
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: active reference cannot preserve deprecated {target_id}")
        if decision == "rewrite":
            target = row.get("target_successor_key_or_id", "")
            if not target:
                p.error(f"REFERENCE_DECISIONS.tsv:{idx}: rewrite requires target")
            elif target not in succ_by_key and target not in allocated_ids:
                p.error(f"REFERENCE_DECISIONS.tsv:{idx}: unknown rewrite target {target!r}")
        if decision in {"rewrite", "remove"} and not row.get("rationale"):
            p.error(f"REFERENCE_DECISIONS.tsv:{idx}: {decision} requires rationale")

    if expected_occurrences:
        decision_occurrences = {
            (r.get("occurrence_id", ""), r.get("source_path", ""), r.get("line", ""), r.get("old_ref", ""))
            for r in ref_rows
        }
        if decision_occurrences != expected_occurrences:
            p.error(
                f"REFERENCE_DECISIONS.tsv must cover every PLAN_REFERENCES.tsv occurrence "
                f"(missing={len(expected_occurrences - decision_occurrences)}, extra={len(decision_occurrences - expected_occurrences)})"
            )

    # Explicit dependency remaps are stricter than generic references.
    candidate_keys = {(r.get("occurrence_id", ""), r.get("owner_source_id", ""), r.get("old_dependency_id", "")) for r in dep_candidates}
    remap_keys = {(r.get("occurrence_id", ""), r.get("owner_source_id", ""), r.get("old_dependency_id", "")) for r in dep_rows}
    if candidate_keys != remap_keys:
        p.error(f"DEPENDENCY_REMAP.tsv must cover every explicit dependency candidate (missing={len(candidate_keys - remap_keys)}, extra={len(remap_keys - candidate_keys)})")
    for idx, row in enumerate(dep_rows, 2):
        owner = row.get("owner_source_id", "")
        old_dep = row.get("old_dependency_id", "")
        decision = row.get("decision", "")
        target = row.get("target_successor_key_or_id", "")
        if owner and owner not in inventory_by_id:
            p.error(f"DEPENDENCY_REMAP.tsv:{idx}: unknown owner {owner}")
        if old_dep and old_dep not in inventory_by_id:
            p.error(f"DEPENDENCY_REMAP.tsv:{idx}: unknown dependency {old_dep}")
        if decision not in {"rewrite", "preserve", "remove"}:
            p.error(f"DEPENDENCY_REMAP.tsv:{idx}: invalid decision {decision!r}")
        if decision == "rewrite" and target not in succ_by_key and target not in allocated_ids:
            p.error(f"DEPENDENCY_REMAP.tsv:{idx}: unknown rewrite target {target!r}")
        if not row.get("rationale"):
            p.error(f"DEPENDENCY_REMAP.tsv:{idx}: rationale required")

    if ns.phase == "postapply":
        # Lightweight live-tree checks. Full planning/reference validators remain mandatory.
        current: dict[str, tuple[Path, dict[str, str], str]] = {}
        for path in list((repo / "docs/planning/active").rglob("*.md")) + list((repo / "docs/planning/completed").rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            if not text.startswith("---\n") and not text.startswith("---\r\n"):
                continue
            fm: dict[str, str] = {}
            lines = text.splitlines()
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if ":" in line and not line[:1].isspace():
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip().strip("'\"")
            item_id = fm.get("id", "")
            if item_id:
                if item_id in current:
                    p.error(f"postapply: duplicate live plan ID {item_id}")
                current[item_id] = (path, fm, text)
        for source_id, row in disp_by_id.items():
            if source_id not in current:
                p.error(f"postapply: predecessor {source_id} missing; predecessors must not be deleted")
                continue
            _, fm, text = current[source_id]
            expected_state = row.get("final_state", "")
            if fm.get("state") != expected_state:
                p.error(f"postapply: predecessor {source_id} state={fm.get('state')!r}, expected {expected_state!r}")
            if row.get("disposition") in CONTINUING:
                if "Superseded by:" not in text or migration_id not in text:
                    p.error(f"postapply: predecessor {source_id} lacks visible successor lineage")
        for key, row in succ_by_key.items():
            allocated = row.get("allocated_id", "")
            if not allocated or allocated not in current:
                p.error(f"postapply: successor {key}/{allocated} missing")
                continue
            _, fm, text = current[allocated]
            if fm.get("plan") != row.get("target_namespace"):
                p.error(f"postapply: successor {allocated} plan={fm.get('plan')!r}, expected {row.get('target_namespace')!r}")
            if "Supersedes:" not in text or migration_id not in text:
                p.error(f"postapply: successor {allocated} lacks visible predecessor lineage")

    print(f"phase={ns.phase}")
    print(f"source_commit={source_commit}")
    print(f"inventory_items={len(inventory)}")
    print(f"successors={len(successors)}")
    print(f"lineage_edges={len(lineage)}")
    return p.finish()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: validator crashed: {exc}", file=sys.stderr)
        raise SystemExit(2)
