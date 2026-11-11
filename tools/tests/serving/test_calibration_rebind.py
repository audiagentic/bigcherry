"""Card-free build rebind tests. Synthetic evidence is never a hardware record."""
import json
import subprocess

import pytest
from bigcherry.serving.calibration_rebind import (
    create_rebind,
    select_with_rebind,
)
from bigcherry.serving.calibration_store import CalibrationInvalid, emit_from_run
from bigcherry.serving.measured_plan import PlanMeasurement
from bigcherry.serving.qualified_calibration import select_for_measurement

NOW = 1800000000


def scope(engine="a", helper="a"):
    return {
        "model_digest": "a" * 64,
        "engine_build_digest": engine * 64,
        "helper_build_digest": helper * 64,
        "layout_digest": "a" * 64,
        "inventory_digest": "a" * 64,
        "owner_map_digest": "a" * 64,
        "context_target_tokens": 128,
        "retention_target_tokens": 192,
        "kv_bytes_per_token": 10,
    }


def record(store):
    run = {
        "schema_version": 1,
        "kind": "s8_calibration_measurement",
        "key": "host_ledger_margin.mib",
        "scope": scope(),
        "raw_run": "runs/synthetic/host-room.json",
        "evidence_id": "sha256:" + "d" * 64,
        "method": "synthetic fixture",
        "measured_at_unix_s": NOW - 100,
        "valid_until_unix_s": NOW + 100,
        "source": "cache",
        "selected_value": 9,
        "samples": [{"candidate": 9, "evidence_ref": "synthetic:host"}],
    }
    source = store.parent / "run.json"
    source.write_text(json.dumps(run))
    return emit_from_run(source, store)


def source_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], text=True,
            capture_output=True, check=True,
        ).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "S8 Synthetic")
    git("config", "user.email", "s8@example.invalid")
    (repo / "README.md").write_text("original\n")
    git("add", "README.md")
    git("commit", "-qm", "prior source")
    old = git("rev-parse", "HEAD")
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("documentation only\n")
    git("add", "docs/note.md")
    git("commit", "-qm", "unaffected docs")
    new = git("rev-parse", "HEAD")
    return repo, old, new, git


def target(tmp_path, target_scope=None):
    target_scope = target_scope or scope("b", "c")
    path = tmp_path / "native.json"
    path.write_text(json.dumps({
        "kind": "bc_serve_v2_measured_inputs",
        "schema_version": 1,
        "planning_key": "a" * 64, "provenance": "synthetic engine publication",
        "scope": target_scope,
        "resource_inputs": {
            "owner_map_digest": target_scope["owner_map_digest"],
            "context_target_tokens": target_scope["context_target_tokens"],
            "retention_target_tokens": target_scope["retention_target_tokens"],
            "kv_bytes_per_token": target_scope["kv_bytes_per_token"],
        },
    }))
    return path


def rebound(tmp_path, *, target_scope=None):
    store = tmp_path / "store"
    original = record(store)
    native = target(tmp_path, target_scope)
    repo, old, new, git = source_repo(tmp_path)
    proof = create_rebind(
        original, native, store, repo=repo, old_ref=old, new_ref=new,
        reason="Only unrelated documentation changed; not engine/helper code.",
        now_unix_s=NOW,
    )
    return store, original, native, proof, repo, old, new, git


def test_immutable_explicit_rebind_selected_with_original_run(tmp_path):
    store, source, native, proof, repo, old, new, _git = rebound(tmp_path)
    rows = select_with_rebind(
        store, scope=scope("b", "c"),
        required_keys=("host_ledger_margin.mib",), now_unix_s=NOW,
    )
    assert len(rows) == 1
    assert rows[0]["value"] == 9
    assert rows[0]["scope"] == scope("b", "c")
    assert rows[0]["evidence_id"].endswith("rebinds/" + proof.name)
    assert rows[0]["raw_run"] == "runs/synthetic/host-room.json"
    assert rows[0]["valid_until_unix_s"] == NOW + 100
    assert json.loads(source.read_text())["scope"] == scope()
    entry = json.loads(proof.read_text())
    assert entry["old_scope"]["engine_build_digest"] == "a" * 64
    assert entry["new_scope"]["engine_build_digest"] == "b" * 64
    assert entry["old_commit"] == old and entry["new_commit"] == new
    assert entry["changed_paths"] == ["docs/note.md"]
    assert "why" not in entry  # closed, tool-produced entry
    assert create_rebind(
        source, native, store, repo=repo, old_ref=old, new_ref=new,
        reason="Only unrelated documentation changed; not engine/helper code.",
        now_unix_s=NOW,
    ) == proof


def test_no_record_and_no_rebind_leaves_the_key_out(tmp_path):
    store = tmp_path / "store"
    record(store)
    assert select_with_rebind(store, scope=scope("b", "c"),
                              required_keys=("host_ledger_margin.mib",),
                              now_unix_s=NOW) == ()


def test_rebind_does_not_extend_expired_evidence(tmp_path):
    store, *_ = rebound(tmp_path)
    assert select_with_rebind(store, scope=scope("b", "c"),
                              required_keys=("host_ledger_margin.mib",),
                              now_unix_s=NOW + 101) == ()


def test_rebind_refuses_changed_model_inventory_owner_and_kv(tmp_path):
    for changed in ("model_digest", "inventory_digest", "owner_map_digest",
                    "context_target_tokens", "retention_target_tokens",
                    "kv_bytes_per_token", "layout_digest"):
        case = tmp_path / changed
        case.mkdir()
        store = case / "store"
        original = record(store)
        changed_scope = scope("b", "c")
        changed_scope[changed] = "f" * 64 if "digest" in changed else 2048
        native = target(case, changed_scope)
        repo, old, new, _ = source_repo(case)
        with pytest.raises(CalibrationInvalid, match="scope changed"):
            create_rebind(
                original, native, store, repo=repo, old_ref=old,
                new_ref=new, reason="Only unrelated documentation changed.",
                now_unix_s=NOW,
            )


def test_rebind_accepts_runtime_kernel_and_helper_changes_with_audited_reason(tmp_path):
    store = tmp_path / "store"
    original = record(store)
    native = target(tmp_path)
    repo, old, _new, git = source_repo(tmp_path)
    p = repo / "engines" / "radiance" / "components" / "serving"
    p.mkdir(parents=True)
    (p / "serving.cpp").write_text("changed engine runtime\n")
    (p / "kernels").mkdir()
    (p / "kernels" / "gemv.hip").write_text("changed kernel\n")
    helper = repo / "engines" / "radiance" / "components" / "moe-remote"
    helper.mkdir(parents=True)
    (helper / "r3_moe_worker.cpp").write_text("changed worker runtime\n")
    git("add", ".")
    git("commit", "-qm", "changed engine/kernel/helper source")
    proof = create_rebind(
        original, native, store, repo=repo, old_ref=old,
        new_ref=git("rev-parse", "HEAD"),
        reason="Scope is build-only; native P1 and S5 gate every changed build.",
        now_unix_s=NOW,
    )
    receipt = json.loads(proof.read_text())
    assert receipt["changed_paths"] == sorted((
        "docs/note.md",
        "engines/radiance/components/serving/serving.cpp",
        "engines/radiance/components/serving/kernels/gemv.hip",
        "engines/radiance/components/moe-remote/r3_moe_worker.cpp",
    ))
    assert len(select_with_rebind(
        store, scope=scope("b", "c"),
        required_keys=("host_ledger_margin.mib",), now_unix_s=NOW,
    )) == 1


def test_rebind_allows_empty_source_diff_for_build_metadata_only(tmp_path):
    store = tmp_path / "store"
    original = record(store)
    native = target(tmp_path)
    repo, _old, _new, git = source_repo(tmp_path)
    # Different commit, identical tree (e.g. build flags/SDK change outside Git).
    git("commit", "--allow-empty", "-qm", "rebuild with a changed toolchain")
    proof = create_rebind(
        original, native, store, repo=repo, old_ref=git("rev-parse", "HEAD~1"),
        new_ref=git("rev-parse", "HEAD"),
        reason="Toolchain-only binary change; fresh P1/S5 admission is the gate.",
        now_unix_s=NOW,
    )
    assert json.loads(proof.read_text())["changed_paths"] == []
    assert select_with_rebind(
        store, scope=scope("b", "c"),
        required_keys=("host_ledger_margin.mib",), now_unix_s=NOW,
    )[0]["value"] == 9


def test_rebind_allows_same_source_commit_for_toolchain_rebuild(tmp_path):
    store = tmp_path / "store"
    original = record(store)
    native = target(tmp_path)
    repo, old, _new, _git = source_repo(tmp_path)
    proof = create_rebind(
        original, native, store, repo=repo, old_ref=old, new_ref=old,
        reason="Same source commit; engine/helper binary changed with toolchain.",
        now_unix_s=NOW,
    )
    entry = json.loads(proof.read_text())
    assert entry["old_commit"] == entry["new_commit"] == old
    assert entry["changed_paths"] == []
    assert select_with_rebind(
        store, scope=scope("b", "c"),
        required_keys=("host_ledger_margin.mib",), now_unix_s=NOW,
    )[0]["value"] == 9


def test_tampered_rebind_is_not_used(tmp_path):
    store, _, _, proof, *_ = rebound(tmp_path)
    parsed = json.loads(proof.read_text())
    parsed["reason"] = "Changed without tool"
    proof.write_text(json.dumps(parsed))
    assert select_with_rebind(store, scope=scope("b", "c"),
                              required_keys=("host_ledger_margin.mib",),
                              now_unix_s=NOW) == ()


def test_rebound_margin_record_is_selected(tmp_path):
    store, *_ = rebound(tmp_path)
    measurement = PlanMeasurement(
        "key", "native", None, None, None, {}, (),
        resource_inputs={"host_pool_margin_calibration": {
            "key": "host_ledger_margin.mib", "source": "cache", "evidence": "any", "mib": 9}},
        scope=scope("b", "c"),
    )
    assert len(select_for_measurement(
        measurement, store, helper_count=0,
        histogram_enabled=False, live_policy_requested=False,
        now_unix_s=NOW,
    )) == 1

