"""Synthetic GPU-free P1 wire/consumer checks; no qualifying hardware evidence."""
import json
from types import SimpleNamespace

import pytest

from bigcherry.engine_adapters.radiance_resources import MIB
from bigcherry.serving.measured_plan import MeasuredPlanRefused, PlanBounds
from bigcherry.serving.p1_inputs import (P1MeasurementSource,
                                          decode_p1_record, parse_p1_json)

_SHA = "a" * 64
_ENGINE = "0000:03:00.0"
_HELPER = "0000:08:00.0"
_LAYOUT = SimpleNamespace(engine_bdfs=(_ENGINE,), helper_bdfs=(_HELPER,))


def sample():
    return {
        "schema_version": 1,
        "kind": "bc_serve_v2_measured_inputs",
        "planning_key": _SHA,
        "provenance": "synthetic engine measuring start",
        "scope": {
            "model_digest": _SHA, "engine_build_digest": _SHA,
            "helper_build_digest": _SHA, "layout_digest": _SHA,
            "inventory_digest": _SHA, "owner_map_digest": _SHA,
            "context_target_tokens": 128,
            "retention_target_tokens": 192, "kv_bytes_per_token": 10,
        },
        "resource_inputs": {
            "host_embedding_bytes": 512, "unowned_expert_bytes": 1024,
            "staging_bytes": 256, "ngram_bytes": 64,
            "checkpoint_bytes": 96, "resident_other_bytes": 2048,
            "kv_bytes_per_token": 10, "context_target_tokens": 128,
            "retention_target_tokens": 192,
            "mem_available_bytes": 100 * MIB,
            "cgroup_headroom_bytes": 90 * MIB,
            "pin_available_bytes": 50 * MIB,
            "owner_map_digest": _SHA,
            "host_pool_margin_calibration": {
                "key": "host_ledger_margin.mib",
                "source": "cache", "evidence": "measured-run.json", "mib": 1,
            },
            "helper_footprints": [{
                "role": "helper0", "bdf": _HELPER,
                "lanes_bytes": 50, "pointer_banks_bytes": 40,
                "slot_map_bytes": 30, "incoming_slot_bytes": 20,
                "total_bytes": 140,
            }],
        },
        "gpu": [{"bdf": _ENGINE, "required_bytes": 2048, "fixed_bytes": 512}],
        "concurrent_sessions": 2,
        "host_reserve_bytes": 4096,
    }


def test_p1_decodes_for_s11_and_preserves_s8_input():
    record = sample()
    plan = parse_p1_json(json.dumps(record), _LAYOUT)
    assert plan.model.kv_bytes_per_token == 10
    assert plan.model.host_staging_bytes == 256
    assert plan.model.unowned_expert_bytes == 1024
    assert plan.worker_footprints_bytes == {_HELPER: 140}
    assert plan.targets.host_pool_margin_bytes == MIB
    assert plan.targets.retained_kv_tokens == 192
    assert plan.gpu[0].bdf == _ENGINE
    assert plan.scope == record["scope"]
    assert plan.resource_inputs == record["resource_inputs"]


@pytest.mark.parametrize("mutation", [
    lambda r: r["resource_inputs"].update(kv_bytes_per_token=0),
    lambda r: r["scope"].update(kv_bytes_per_token=5),
    lambda r: r["resource_inputs"].update(owner_map_digest="b" * 64),
    lambda r: r["resource_inputs"]["helper_footprints"][0].update(total_bytes=139),
    lambda r: r["resource_inputs"]["helper_footprints"][0].update(bdf=_ENGINE),
    lambda r: r["gpu"][0].update(bdf=_HELPER),
    lambda r: r["gpu"][0].update(fixed_bytes=4096),
    lambda r: r["resource_inputs"].update(mem_available_bytes=-1),
    lambda r: r["resource_inputs"].update(pin_available_bytes=True),
    lambda r: r["resource_inputs"].pop("ngram_bytes"),
    lambda r: r.update(unexpected=123),
    lambda r: r["resource_inputs"]["host_pool_margin_calibration"].update(evidence=""),
    lambda r: r["resource_inputs"]["host_pool_margin_calibration"].update(mib=1 << 53),
    lambda r: r.update(concurrent_sessions=1),
])
def test_p1_fails_closed_on_incomplete_or_mixed_native_inputs(mutation):
    r = sample()
    mutation(r)
    with pytest.raises(MeasuredPlanRefused):
        decode_p1_record(r, _LAYOUT)


def test_reject_duplicate_json_members_and_fractional_bytes():
    raw = json.dumps(sample())
    with pytest.raises(MeasuredPlanRefused, match="duplicate"):
        parse_p1_json(raw.replace('"schema_version": 1,',
                                  '"schema_version": 1, "schema_version": 1,'),
                      _LAYOUT)
    with pytest.raises(MeasuredPlanRefused, match="integer"):
        parse_p1_json(raw.replace('"checkpoint_bytes": 96', '"checkpoint_bytes": 9.6'),
                      _LAYOUT)


def test_source_remeasures_with_updated_bounds_and_no_record_fallback():
    calls = []

    def command(layout, bounds):
        assert layout is _LAYOUT
        return ["bc-serve", "--publish-v2-measurement", str(bounds.retained_kv_tokens)]

    def run(argv, **kwargs):
        assert kwargs == {"text": True, "capture_output": True, "check": False}
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps(sample()))

    source = P1MeasurementSource(command, run=run)
    source.measure(_LAYOUT, PlanBounds())
    source.measure(_LAYOUT, PlanBounds(retained_kv_tokens=192))
    assert calls[0] != calls[1]

    def rejected(argv, **kwargs):
        return SimpleNamespace(returncode=4, stdout="", stderr="text only")

    with pytest.raises(MeasuredPlanRefused, match="failed"):
        P1MeasurementSource(command, run=rejected).measure(_LAYOUT, PlanBounds())
