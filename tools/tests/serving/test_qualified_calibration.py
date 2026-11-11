"""Synthetic card-free S13 calibration gate tests; not hardware qualification."""
import json
from types import SimpleNamespace

import pytest
from bigcherry.serving.calibration_store import CalibrationInvalid, emit_from_run
from bigcherry.serving.measured_plan import PlanBounds, PlanMeasurement
from bigcherry.serving.qualified_calibration import (
    QualifiedP1MeasurementSource,
    wanted_s8_records,
    select_for_measurement,
)

NOW = 1800000000
DIGEST = "a" * 64
HELPER = "0000:08:00.0"
LAYOUT = SimpleNamespace(helper_bdfs=(HELPER,), engine_bdfs=("0000:03:00.0",))


def scope():
    return {
        "model_digest": DIGEST, "engine_build_digest": DIGEST,
        "helper_build_digest": DIGEST, "layout_digest": DIGEST,
        "inventory_digest": DIGEST, "owner_map_digest": DIGEST,
        "context_target_tokens": 128, "retention_target_tokens": 192,
        "kv_bytes_per_token": 10,
    }


def measurement():
    return PlanMeasurement(
        DIGEST, "synthetic native publication", None, None, None, {}, (),
        resource_inputs={
            "host_pool_margin_calibration": {
                "key": "host_ledger_margin.mib",
                "source": "cache", "evidence": "evidence-host_ledger_margin.mib",
                "mib": 9,
            }
        },
        scope=scope(),
    )


def add_record(tmp_path, key, *, recorded_scope=None):
    run = {
        "schema_version": 1, "kind": "s8_calibration_measurement",
        "key": key, "scope": recorded_scope or scope(),
        "raw_run": "runs/synthetic/" + key + ".json",
        "evidence_id": "evidence-" + key,
        "method": "synthetic measurement fixture, not qualified hardware",
        "measured_at_unix_s": NOW - 100,
        "valid_until_unix_s": NOW + 100,
        "source": "cache", "selected_value": 9,
        "samples": [
            {"candidate": 9, "evidence_ref": "synthetic:" + key},
        ],
    }
    filename = tmp_path / ("measurement-" + key.replace(".", "-") + ".json")
    filename.write_text(json.dumps(run))
    emit_from_run(filename, tmp_path / "records")


def populate(tmp_path, *, live=False, histogram=False, helpers=1):
    for key in wanted_s8_records(
        helper_count=helpers, live_policy_requested=live,
        histogram_enabled=histogram,
    ):
        add_record(tmp_path, key)


def select(tmp_path, *, live=False, histogram=False, helpers=1, observed=None, now=NOW):
    return select_for_measurement(
        observed or measurement(), tmp_path / "records",
        helper_count=helpers, histogram_enabled=histogram,
        live_policy_requested=live, now_unix_s=now,
    )


def test_engine_only_requires_only_exact_margin(tmp_path):
    add_record(tmp_path, "host_ledger_margin.mib")
    assert len(select(tmp_path, helpers=0)) == 1


def test_selected_helper_requires_all_four_handoff_records(tmp_path):
    populate(tmp_path)
    assert [r["key"] for r in select(tmp_path)] == list(
        wanted_s8_records(helper_count=1, histogram_enabled=False,
                            live_policy_requested=False)
    )


def test_live_policy_requires_period_and_cadence_from_store(tmp_path):
    populate(tmp_path, live=True, histogram=True)
    selected = select(tmp_path, live=True, histogram=True)
    assert selected[-2]["key"] == "handoff.histogram_period_s"
    assert selected[-1]["key"] == "handoff.live_cadence_ns"


@pytest.mark.parametrize("live,hist,helpers", [
    (True, False, 1), (True, True, 0), (False, True, 0),
])
def test_refuse_invalid_policy_and_layout(live, hist, helpers):
    with pytest.raises(CalibrationInvalid):
        wanted_s8_records(
            helper_count=helpers, histogram_enabled=hist,
            live_policy_requested=live,
        )


def test_no_records_still_starts(tmp_path):
    assert select(tmp_path) == ()


def test_only_the_records_present_are_returned(tmp_path):
    add_record(tmp_path, "host_ledger_margin.mib")
    assert [r["key"] for r in select(tmp_path)] == ["host_ledger_margin.mib"]


def test_absent_live_cadence_leaves_the_key_out(tmp_path):
    populate(tmp_path, histogram=True)
    assert "handoff.live_cadence_ns" not in [r["key"] for r in select(tmp_path, live=True, histogram=True)]


def test_expired_record_and_changed_scope_are_skipped(tmp_path):
    populate(tmp_path)
    assert select(tmp_path, now=NOW + 101) == ()
    altered = measurement()
    altered.scope["inventory_digest"] = "b" * 64
    assert select(tmp_path, observed=altered) == ()

def test_remeasure_and_reselect_on_every_bounded_retry(tmp_path):
    populate(tmp_path)
    calls = []

    class Native:
        def measure(self, layout, bounds):
            calls.append(bounds)
            result = measurement()
            if bounds.retained_kv_tokens == 50:
                result.scope["retention_target_tokens"] = 50
            return result

    source = QualifiedP1MeasurementSource(
        Native(), store=tmp_path / "records", helper_count=1,
        histogram_enabled=False, live_policy_requested=False,
        clock=lambda: NOW,
    )
    first = source.measure(LAYOUT, PlanBounds())
    assert len(first.calibration_records) == 5
    retry = source.measure(LAYOUT, PlanBounds(retained_kv_tokens=50))
    assert retry.calibration_records == ()  # the new scope has no records yet; the start goes on without them
    assert len(calls) == 2
