#!/usr/bin/env python3
"""Portable CPU-only fixture check; use --work-root to check a live store."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--work-root", type=Path, help="Optional original work root")
args = ap.parse_args()
gate = json.loads((HERE / "gate-and-lifecycle.json").read_text())
att = json.loads((HERE / "candidate-attestation.json").read_text())
assert gate["source_difference_count"] == len(gate["source_difference_files"]) == 7
assert att["source_sha"] == gate["candidate_source"]
assert att["build_plan_id"] == gate["missing_build"]["build_plan_id"]
assert gate["previous_candidate_strict_v2"]["strict_v2"]
assert gate["previous_candidate_strict_v2"]["state"] == "measured"
assert gate["previous_candidate_strict_v2"]["owner_count"] == 12641
assert gate["missing_build"]["candidate_members_verified"] == 0
assert not gate["missing_build"]["build_directory_exists"]
assert not gate["missing_build"]["artifact_store_directory_exists"]
assert not gate["missing_build"]["build_log_directory_exists"]
assert len(att["members"]) == gate["missing_build"]["candidate_members_total"] == 12
if args.work_root:
    base = args.work_root / "builds" / att["source_slice_id"] / att["build_plan_id"]
    store = args.work_root / "artifacts-store/builds" / att["source_slice_id"] / att["build_plan_id"]
    assert not base.exists() and not store.exists(), "bundle state changed since observation"
    for spec in att["members"].values():
        binary = base / spec["relative_path"]
        assert not binary.exists(), "B member unexpectedly exists"
    assert not (args.work_root / "build-logs/rcs13-run9-b").exists()
print("RCS13_RUN10_CPU_ARTIFACT_RETENTION_STOP_REPRODUCED")
