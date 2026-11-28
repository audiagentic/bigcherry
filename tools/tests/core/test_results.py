"""MRC05: engine-neutral, provenance-contained baseline/compare/history contracts."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from bigcherry.cli.main import main
from bigcherry.core import host_facts, results, run_records


def make_record(store, number, *, session="ab-1", decode=40, prefill=1000, ttft=30,
                heat=b"heat-A", policy="active", kernel="7.0", config=None,
                gated=True, repeats=None, layout="tp3", workload="long",
                stamp=None):
    def blob(data):
        return store.put_blob(data)
    config = {"kv": "fp8"} if config is None else config
    doc = {
        "schema_version": 1,
        "run_id": f"launch-{number}",
        "session_id": session,
        "recorded_at": stamp or f"2026-11-{number + 1:02d}T10:00:00Z",
        "identity": {
            "engine": "dummy", "build_id": "b1", "binary_sha256": "a" * 64,
            "source_revision": "abc123", "model": "model", "layout": layout,
        },
        "workload": {"name": workload, "definition": {"prompt": 2048, "output": 256}},
        "metrics": {
            "decode": {"tokens_per_second": decode},
            "prefill": {"tokens_per_second": prefill},
            "ttft": {"milliseconds": ttft},
        },
        "metric_specs": {
            "decode": {"tokens_per_second": {"direction": "higher", "gated": gated,
                        "repeat_groups": repeats or []}},
            "prefill": {"tokens_per_second": {"direction": "higher", "gated": True}},
            "ttft": {"milliseconds": {"direction": "lower", "gated": True}},
        },
        "provenance": {
            "effective_config": blob(json.dumps(config).encode()),
            "argv": blob(b'["serve"]'),
            "environment": blob(b'{"ENV":"clean"}'),
            "power_settings": blob(b'{"cap_watts":230}'),
            "inputs": [
                {"role": "heat", "blob": blob(heat)},
                {"role": "policy", "blob": blob(policy.encode())},
            ],
        },
        "host_facts": host_facts.snapshot_at_start(
            {"kernel_release": {"value": kernel, "readable": True},
             "cards": {"0000:01:00.0": {
                 "bdf": {"value": "0000:01:00.0", "readable": True},
                 "power": {"cap_uw": {"value": 230000000, "readable": True}}
             }}},
            captured_at=stamp or f"2026-11-{number + 1:02d}T10:00:00Z",
        ),
        "feature_states": [{
            "engine": "dummy", "feature_id": "live_policy",
            "state": policy, "reason_code": "enabled" if policy == "active" else "disabled",
            "reason_detail": "", "observed_at": "2026-11-19T00:00:00Z",
        }],
    }
    return {"sha256": store.put_record(doc), "record": doc}


def cohort(store, start=1, values=(40, 44), **kw):
    return [make_record(store, start + i, decode=value, **kw) for i, value in enumerate(values)]


def test_a_record_without_metric_specs_is_refused(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    row = make_record(store, 1)
    assert store.get_record(row["sha256"]) == row["record"]
    bare = deepcopy(row["record"])
    del bare["metric_specs"]
    del bare["session_id"]
    with pytest.raises(run_records.RecordError, match="metric_specs"):
        store.put_record(bare)


def test_single_launch_or_same_run_id_is_refused(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    rows = cohort(store)
    with pytest.raises(results.ResultsError, match="two distinct launches"):
        results.make_baseline(store, rows[:1], "bad")
    duplicate = deepcopy(rows[0]["record"])
    duplicate["recorded_at"] = "2026-11-19T18:00:00Z"
    with pytest.raises(results.ResultsError, match="two distinct launches"):
        results.make_baseline(store, [rows[0],
            {"sha256": store.put_record(duplicate), "record": duplicate}], "bad")
    baseline = results.make_baseline(store, rows, "two")
    with pytest.raises(results.ResultsError, match="two distinct launches"):
        results.compare_records(store, baseline, rows[:1])


def test_spread_repeats_and_one_sided_directions(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    base_rows = [
        make_record(store, 1, decode=100, prefill=1000, ttft=20,
                    repeats=[[98, 102], [92, 108]]),
        make_record(store, 2, decode=110, prefill=1020, ttft=22),
    ]
    baseline = results.make_baseline(store, base_rows, "baseline")
    decode = baseline["metrics"]["decode.tokens_per_second"]
    assert decode["value"] == 105
    assert decode["spread_across_launches"] == pytest.approx(10 / 105)
    assert decode["spread_repeats"] == pytest.approx(16 / 100)
    assert decode["tolerance"] == pytest.approx(.16)
    assert baseline["metrics"]["ttft.milliseconds"]["direction"] == "lower"
    # improvements (including a large TTFT decrease) never trip one-sided gates
    better = cohort(store, 3, values=(130, 132), prefill=1200, ttft=10)
    outcome = results.compare_records(store, baseline, better)
    assert not outcome["failed"]
    assert all(m["verdict"] == "better" for m in outcome["metrics"].values())
    worse = cohort(store, 5, values=(70, 72), prefill=700, ttft=30)
    outcome = results.compare_records(store, baseline, worse)
    assert outcome["failed"]
    assert outcome["metrics"]["decode.tokens_per_second"]["verdict"] == "REGRESSION"
    assert outcome["metrics"]["prefill.tokens_per_second"]["verdict"] == "REGRESSION"
    assert outcome["metrics"]["ttft.milliseconds"]["verdict"] == "REGRESSION"


def test_informational_and_missing_metric(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    baseline = results.make_baseline(store, cohort(store, gated=False), "info")
    now = cohort(store, 3, values=(1, 2), gated=False)
    diff = results.compare_records(store, baseline, now)
    assert diff["metrics"]["decode.tokens_per_second"]["verdict"] == "info"
    assert not diff["failed"]
    missing = []
    for i, row in enumerate(cohort(store, 5, gated=False)):
        doc = deepcopy(row["record"])
        if i == 1:
            del doc["metrics"]["ttft"]["milliseconds"]
            del doc["metric_specs"]["ttft"]["milliseconds"]
        missing.append({"record": doc, "sha256": store.put_record(doc)})
    assert results.compare_records(store, baseline, missing)["metrics"]["ttft.milliseconds"]["verdict"] == "MISSING"


def test_provenance_bundle_embedded_and_integrity_checked_without_store(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    rows = cohort(store)
    baseline = results.make_baseline(store, rows, "portable")
    encoded = json.loads(json.dumps(baseline))
    assert len(encoded["records"]) == 2
    assert encoded["blobs"]
    results.validate_baseline(encoded)
    other = deepcopy(encoded)
    other["blobs"][0]["base64"] = "YWJj"
    with pytest.raises(results.ResultsError, match="blob|sha256"):
        results.validate_baseline(other)
    other = deepcopy(encoded)
    other["records"][0]["record"]["metrics"]["decode"]["tokens_per_second"] = 1
    with pytest.raises(results.ResultsError, match="record|sha256"):
        results.validate_baseline(other)
    # No original run folders, store or source files are needed to inspect baseline.
    assert results.validate_baseline(encoded)["name"] == "portable"


def test_provenance_diff_and_cross_session_guidance(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    baseline = results.make_baseline(store, cohort(store), "base")
    changed = cohort(store, 3, session="ab-2", values=(40, 44),
                     policy="inactive", kernel="7.1", heat=b"heat-B",
                     config={"kv": "bf16"})
    outcome = results.compare_records(store, baseline, changed)
    categories = {d["category"] for d in outcome["provenance_differences"]}
    assert {"config", "heat", "policy", "kernel", "host_facts", "feature_states"} <= categories
    assert any("same-session" in x for x in outcome["warnings"])
    assert not outcome["failed"]  # difference is visible, not silently a metric regression


def test_mixed_workload_direction_and_noncanonical_rows_refused(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    rows = cohort(store)
    other = make_record(store, 3, workload="short")
    with pytest.raises(results.ResultsError, match="workload"):
        results.make_baseline(store, [rows[0], other], "bad")
    changed = deepcopy(rows[1]["record"])
    changed["metric_specs"]["ttft"]["milliseconds"]["direction"] = "higher"
    with pytest.raises(results.ResultsError, match="direction|metric"):
        results.make_baseline(store, [rows[0],
            {"sha256": store.put_record(changed), "record": changed}], "bad")
    forged = deepcopy(rows[0])
    forged["record"]["run_id"] = "forged"
    with pytest.raises(results.ResultsError, match="sha256|digest"):
        results.make_baseline(store, [forged, rows[1]], "bad")


def test_history_metric_is_scoped_and_reports_baseline_and_trailing_drift(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    baseline = results.make_baseline(store, cohort(store, values=(100, 102)), "base")
    cohort(store, 3, values=(101, 100))
    cohort(store, 5, values=(70, 68))
    make_record(store, 7, layout="different", decode=2)
    history = results.history(store, engine="dummy", layout="tp3", workload="long",
                              metric="decode.tokens_per_second", baseline=baseline,
                              trailing_window=2)
    assert len(history["points"]) == 6
    assert history["points"][0]["baseline_drift"] is None  # no 2-launch verdict
    assert history["points"][-1]["baseline_drift"]
    assert history["points"][-1]["trailing_drift"]
    assert not history["points"][3]["trailing_drift"]
    with pytest.raises(results.ResultsError, match="group.name"):
        results.history(store, engine="dummy", layout="tp3", workload="long", metric="tokens_per_second")


def test_cli_baseline_compare_history_and_exit_codes(tmp_path, capsys):
    store = run_records.RunRecordStore(tmp_path)
    cohort(store)
    path = tmp_path / "reference.json"
    common = ["--work-root", str(tmp_path), "--engine", "dummy",
              "--layout", "tp3", "--workload", "long"]
    assert main(["results", "baseline", "--name", "ref", "--to", str(path), *common,
                 "--until", "2026-11-03T23:59:59Z"]) == 0
    assert path.is_file()
    capsys.readouterr()  # compare output must put provenance ahead of verdicts
    cohort(store, 3, values=(20, 21), session="new", policy="inactive")
    assert main(["results", "compare", "--baseline", str(path), *common,
                 "--since", "2026-11-04T00:00:00Z"]) == 1
    output = capsys.readouterr().out
    assert output.index("feature_states") < output.index("decode.tokens_per_second")
    assert "policy" in output and "same-session" in output
    assert main(["results", "history", "--metric", "decode.tokens_per_second",
                 "--baseline", str(path), *common, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["points"][-1]["baseline_drift"] is True


def test_candidate_only_metric_is_reported_and_mixed_sessions_warn(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    baseline = results.make_baseline(store, cohort(store), "base")
    changed = []
    for index, row in enumerate(cohort(store, 3)):
        rec = deepcopy(row["record"])
        rec["session_id"] = "ab-1" if index == 0 else "ab-2"
        rec["metrics"]["prefill"]["new_metric"] = 12 + index
        rec["metric_specs"]["prefill"]["new_metric"] = {
            "direction": "higher", "gated": True,
        }
        changed.append({"sha256": store.put_record(rec), "record": rec})
    report = results.compare_records(store, baseline, changed)
    assert report["metrics"]["prefill.new_metric"]["verdict"] == "NEW"
    assert report["metrics"]["prefill.new_metric"]["baseline"] is None
    assert any("mixed sessions" in warning for warning in report["warnings"])


def test_history_never_counts_the_same_launch_twice(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    first = make_record(store, 1)
    duplicate = deepcopy(first["record"])
    duplicate["recorded_at"] = "2026-11-03T00:00:00Z"
    store.put_record(duplicate)
    with pytest.raises(results.ResultsError, match="duplicate launch"):
        results.history(store, engine="dummy", layout="tp3", workload="long",
                        metric="decode.tokens_per_second")
