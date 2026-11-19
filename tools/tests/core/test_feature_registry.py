"""Platform production feature coverage and llama.cpp adapter gates."""
import pytest
from bigcherry.core import engines, feature_registry, feature_states

AT = "2026-11-20T00:00:00Z"


def state(engine, feature_id, *, mode="active", reason="configured"):
    return dict(engine=engine, feature_id=feature_id, state=mode,
                reason_code=reason, reason_detail="", observed_at=AT)


def test_registry_and_required_snapshots():
    expected = {"live_policy", "mtp", "prefix_cache", "kv_host_tier",
                "helper_combine", "vision_tower_placement", "mm_max_patches"}
    assert set(feature_registry.feature_ids(engines.RADIANCE)) == expected
    assert set(feature_registry.feature_ids(engines.LLAMACPP)) == {"dispatch_cache", "tuning"}
    assert feature_registry.is_production(engines.RADIANCE, "live_policy")
    assert not feature_registry.is_production(engines.RADIANCE, "mtp")
    snapshot = [state(engines.RADIANCE, f) for f in sorted(expected)]
    assert feature_registry.require_coverage(engines.RADIANCE, snapshot) == snapshot
    with pytest.raises(feature_states.FeatureStateError, match="missing"):
        feature_registry.require_coverage(engines.RADIANCE, snapshot[:-1])
    with pytest.raises(feature_states.FeatureStateError, match="undeclared"):
        feature_registry.require_coverage(engines.RADIANCE, snapshot + [state(engines.RADIANCE, "made_up")])
    with pytest.raises(feature_states.FeatureStateError, match="engine"):
        feature_registry.require_coverage(engines.LLAMACPP, snapshot)


def test_llama_adapter_explicit_configuration_and_absence():
    active = feature_registry.llamacpp_states(
        {"GGML_HIP_DISPATCH_MODE": "replay", "GGML_HIP_DISPATCH_CACHE": "cache.bin",
         "GGML_HIP_TUNE_WORKLOAD": "workload.json"}, at=AT)
    assert {s["feature_id"]: s["state"] for s in active} == {
        "dispatch_cache": "active", "tuning": "active"}
    empty = feature_registry.llamacpp_states({}, at=AT)
    assert all(s["state"] == "inactive" and s["reason_code"] == "not_configured" for s in empty)
    feature_states.validate_feature_states(empty)


def test_warnings_and_strict_production_refusal():
    states = [state(engines.RADIANCE, f) for f in feature_registry.feature_ids(engines.RADIANCE)]
    states[0] = state(engines.RADIANCE, "live_policy", mode="inactive", reason="calibration_key_missing")
    warnings = feature_registry.production_warnings(engines.RADIANCE, states)
    assert len(warnings) == 1 and "calibration_key_missing" in warnings[0]
    with pytest.raises(feature_registry.ProductionFeatureError, match="live_policy"):
        feature_registry.production_warnings(engines.RADIANCE, states, strict=True)
    states[0]["reason_code"] = "no_helpers"
    assert feature_registry.production_warnings(engines.RADIANCE, states) == []

def test_platform_status_prints_observed_features_and_strict_gate(tmp_path, monkeypatch, capsys):
    import json
    from argparse import Namespace
    from bigcherry.cli import diagnostics
    states = [state(engines.RADIANCE, f) for f in feature_registry.feature_ids(engines.RADIANCE)]
    states[0] = state(engines.RADIANCE, "live_policy", mode="inactive", reason="calibration_key_missing")
    (tmp_path / feature_states.FILE_NAME).write_text(json.dumps(states))
    monkeypatch.setattr(diagnostics.paths, "llama_root", lambda _: tmp_path)
    monkeypatch.setattr(diagnostics.source_audit, "git_revision", lambda _: ("abc", False))
    monkeypatch.setattr(diagnostics.releases, "all_records", lambda: [])
    args = Namespace(llama_root=None, run_dir=tmp_path, strict_production_features=False)
    assert diagnostics.cmd_status(args) == 0
    out = capsys.readouterr().out
    assert "calibration_key_missing" in out and "live_policy" in out
    args.strict_production_features = True
    assert diagnostics.cmd_status(args) == 2
    capsys.readouterr()
    (tmp_path / feature_states.FILE_NAME).unlink()
    assert diagnostics.cmd_status(args) == 2
