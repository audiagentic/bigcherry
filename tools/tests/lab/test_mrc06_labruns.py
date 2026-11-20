"""MRC06 slice 3 card-free producer/ingest/CLI contracts (fixtures first)."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bigcherry.cli.main import main
from bigcherry.core import engines, feature_registry, measurement_producers, run_records

FAKE_HOST = {"kernel_release": {"value": "fixture", "readable": True}}
STATES = [
    {
        "engine": "radiance",
        "feature_id": "live_policy",
        "state": "inactive",
        "reason_code": "not_configured",
        "reason_detail": "",
        "observed_at": "2026-11-20T00:00:00Z",
    }
]
STATES.extend({
    "engine": engines.RADIANCE, "feature_id": name, "state": "inactive",
    "reason_code": "fixture_not_configured", "reason_detail": "",
    "observed_at": "2026-11-20T00:00:00Z",
} for name in feature_registry.feature_ids(engines.RADIANCE) if name != "live_policy")


def make_run(root: Path, *, state="running", events=None):
    root.mkdir(parents=True, exist_ok=True)
    build = root / "build"
    build.mkdir()
    binary = build / "bin/radiance"
    binary.parent.mkdir()
    binary.write_bytes(b"exact engine bytes")
    meta = {
        "binary_hash": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "build_id": "a" * 32,
        "source_slice_id": "b" * 32,
        "build_plan_id": "c" * 32,
    }
    (build / "bigcherry-build-metadata-radiance.json").write_text(json.dumps(meta))
    manifest = root / "manifest-measured.json"
    content = {
        "metadata": {"instance_id": "launch-v1"},
        "spec": {
            "intent": {"layout": "flashnext-4gpu", "model": "Qwen"},
            "plan": {
                "engine_memory": {
                    "model_identity": "m" * 64,
                    "engine_identity": "b" * 64,
                }
            },
            "facts": {},
            "processes": {
                "engine": {
                    "runtime": {"radiance_home": str(build / "radiance_home")},
                    "server": {"port": 18450},
                }
            },
            "power": {"profile": "stock"},
            "handoff": {
                "placement": {
                    "heat_file": str(root / "selected-heat"),
                    "live_policy_state": {
                        "configured": False,
                        "reason_code": "not_configured",
                    },
                }
            },
        },
    }
    manifest.write_text(json.dumps(content))
    (root / "feature-states.v1.json").write_text(json.dumps(STATES))
    (root / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": state,
                "port": 18450,
                "manifest": str(manifest),
                "started_at": "2026-11-20T00:00:00Z",
                "selection": {"calibration_store_key": None},
                "events": [{"kind": k} for k in (events or [])],
                "processes": [
                    {
                        "role": "engine",
                        "argv": ["radiance", "--manifest", str(manifest)],
                        "log": str(root / "engine.log"),
                    }
                ],
                "errors": [],
            }
        )
    )
    (root / "engine.log").write_bytes(b"real engine stdout")
    return manifest, meta


def make_calibration(root: Path):
    from bigcherry.core import calibration_records as cr

    fields = cr.key_declaration("radiance.live_policy")
    scope = {
        "policy_key_version": 2,
        "model": "m1",
        "layout": "flashnext-4gpu",
        "objective": "balanced",
        "power_profile": "stock",
        "engine_cards": [
            {"bdf": "0000:01:00.0", "arch": "gfx1201", "vram_total_mib": 32768}
        ],
        "helper_cards": [],
        "histogram_period_s": 1,
        "helper_runtime": {"combine": True},
        "remote_service": {"max_rows": 1024},
    }
    key = cr.key_of("radiance.live_policy", scope)["sha256"]
    absent = {k: "not observed" for k in fields["quantity_fields"] if k != "txn_p50_ns"}
    lines = [
        "schema_version = 1",
        "[[live_policy_record]]",
        f'plan_cache_key = "{key}"',
        "record_schema_version = 1",
        'calibration_kind = "radiance.live_policy"',
        "calibration_key_fields = " + json.dumps(fields["key_fields"]),
        "calibration_inputs = "
        + json.dumps(json.dumps(scope, sort_keys=True, separators=(",", ":"))),
        'source_plan_key = "' + "c" * 64 + '"',
        'source_engine_build = "' + "b" * 64 + '"',
        'source_revision = "source_slice:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"',
        'run_id = "launch-v1"',
        'host_facts_digest = "' + hashlib.sha256(b"{}").hexdigest() + '"',
        'layout = "flashnext-4gpu"',
        'objective = "balanced"',
        'seeded = "2026-11-20"',
        'evidence = "settled native observation"',
        "txn_p50_ns = 100",
        "absent = {" + ", ".join(f'{k} = "not observed"' for k in absent) + "}",
    ]
    raw = ("\n".join(lines) + "\n").encode()
    snap = root / "calibration-store.v1.toml"
    snap.write_bytes(raw)
    assert cr.read_native_toml(snap)[0]["status"] == "current"
    helper = root / "helper0-calibration.log"
    helper.write_bytes(b"native verified settled helper calibration\n")
    (root / "calibration-heat.v1").write_bytes(b"selected original heat bytes")
    (root / "calibration-prompts-a.v1").write_bytes(b"prompt A\n")
    (root / "calibration-prompts-b.v1").write_bytes(b"prompt B\n")
    return key, snap, helper


def test_calibration_native_evidence_ingests_only_proved_snapshot(tmp_path):
    from bigcherry.bench import radiance_lab_records as lab

    run = tmp_path / "run"
    manifest, _ = make_run(run, state="stopped", events=["calibration_completed"])
    key, snap, helper = make_calibration(run)
    session_path = run / "session.json"
    session = json.loads(session_path.read_text())
    session["selection"]["calibration_store_key"] = key
    session_path.write_text(json.dumps(session))
    evidence = {
        "schema_version": 1,
        "session_id": "launch-v1",
        "calibration_store_key": key,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "calibration_store_sha256": hashlib.sha256(snap.read_bytes()).hexdigest(),
        "helper_log_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "helper_log": helper.name,
        "observed_at": "2026-11-20T02:00:00Z",
        "heat_sha256": hashlib.sha256(
            (run / "calibration-heat.v1").read_bytes()
        ).hexdigest(),
        "prompts_a_sha256": hashlib.sha256(
            (run / "calibration-prompts-a.v1").read_bytes()
        ).hexdigest(),
        "prompts_b_sha256": hashlib.sha256(
            (run / "calibration-prompts-b.v1").read_bytes()
        ).hexdigest(),
    }
    (run / "calibration-evidence.v1.json").write_text(json.dumps(evidence))
    sha = lab.publish_calibration_run(run, work_root=tmp_path, host_snapshot=FAKE_HOST)
    store = run_records.RunRecordStore(tmp_path)
    rec = store.get_record(sha)
    assert rec["workload"]["name"] == "live-policy-calibration"
    assert rec["metrics"]["decode"]["txn_p50_ns"] == 100
    assert rec["identity"]["engine"] == "radiance"
    inputs = {
        entry["role"]: store.get_blob(entry["blob"])
        for entry in rec["provenance"]["inputs"]
    }
    assert inputs["selected_calibration_record"]
    assert inputs["native_calibration_store"] == snap.read_bytes()
    assert inputs["calibration_helper_log"] == helper.read_bytes()
    assert inputs["calibration_heat"] == b"selected original heat bytes"
    assert inputs["calibration_prompts_a"] == b"prompt A\n"
    assert inputs["calibration_prompts_b"] == b"prompt B\n"
    assert rec["host_facts"] == FAKE_HOST
    manifest.unlink()
    snap.unlink()
    assert store.get_record(sha) == rec


def test_calibration_bad_snapshot_or_wrong_key_fails_closed(tmp_path):
    from bigcherry.bench import radiance_lab_records as lab

    run = tmp_path / "run"
    manifest, _ = make_run(run, state="stopped", events=["calibration_completed"])
    key, snap, helper = make_calibration(run)
    session = json.loads((run / "session.json").read_text())
    session["selection"]["calibration_store_key"] = key
    (run / "session.json").write_text(json.dumps(session))
    evidence = {
        "schema_version": 1,
        "session_id": "launch-v1",
        "calibration_store_key": key,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "calibration_store_sha256": hashlib.sha256(snap.read_bytes()).hexdigest(),
        "helper_log_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "helper_log": helper.name,
        "observed_at": "2026-11-20T02:00:00Z",
        "heat_sha256": hashlib.sha256(
            (run / "calibration-heat.v1").read_bytes()
        ).hexdigest(),
        "prompts_a_sha256": hashlib.sha256(
            (run / "calibration-prompts-a.v1").read_bytes()
        ).hexdigest(),
        "prompts_b_sha256": hashlib.sha256(
            (run / "calibration-prompts-b.v1").read_bytes()
        ).hexdigest(),
    }
    path = run / "calibration-evidence.v1.json"
    path.write_text(json.dumps(evidence))
    snap.write_bytes(snap.read_bytes() + b"\n# post-run mutation")
    with pytest.raises(ValueError, match="sha256|digest|mismatch"):
        lab.publish_calibration_run(run, work_root=tmp_path, host_snapshot=FAKE_HOST)
    assert run_records.RunRecordStore(tmp_path).query() == []


def test_probe_records_preserve_exact_rows_after_run_dir_purge(tmp_path):
    from bigcherry.bench import radiance_lab_records as lab

    run = tmp_path / "run"
    make_run(run)
    needle = {
        "rows": [
            {"tokens_target": 48000, "depth": 0.1, "rep": 0, "hit": True, "seconds": 8},
            {
                "tokens_target": 48000,
                "depth": 0.5,
                "rep": 0,
                "hit": False,
                "seconds": 11,
            },
        ],
        "summary": {"48000": {"hits": 1, "of": 2}},
        "chars_per_token": 3.5,
    }
    sha = lab.publish_probe(
        run,
        kind="needle",
        result=needle,
        definition={"tokens": [48000], "depths": [0.1, 0.5], "seed": 7},
        argv=["needle_probe.py", "--tokens", "48000"],
        extra_inputs={"corpus": b"exact original docs"},
        work_root=tmp_path,
        host_snapshot=FAKE_HOST,
    )
    store = run_records.RunRecordStore(tmp_path)
    rec = store.get_record(sha)
    assert rec["metrics"]["prefill"]["needle_hit_rate"] == 0.5
    assert rec["metrics"]["decode"] == {}
    assert rec["workload"]["name"] == "needle-probe"
    assert rec["validity"]["state"] == "unverified"
    image = {
        "format": "bmp",
        "expected": "refused",
        "verdict": "refused",
        "http_status": 400,
        "seconds": 0.4,
        "text": "not supported",
    }
    sha2 = lab.publish_probe(
        run,
        kind="image",
        result=image,
        definition={"format": "bmp", "expected": "refused"},
        argv=["image_probe.py", "--expect", "refused"],
        extra_inputs={"image_payload": b"BMimage fixture"},
        work_root=tmp_path,
        host_snapshot=FAKE_HOST,
    )
    image_rec = store.get_record(sha2)
    assert image_rec["metrics"]["prefill"]["image_expected_match"] == 1
    assert (
        store.get_blob(
            next(
                i["blob"]
                for i in image_rec["provenance"]["inputs"]
                if i["role"] == "image_payload"
            )
        )
        == b"BMimage fixture"
    )
    assert sha != sha2


def test_rsc02_active_runner_records_verified_manifest_and_arm(tmp_path):
    from bigcherry.bench import radiance_lab_records as lab

    run = tmp_path / "run"
    manifest, meta = make_run(run)
    raw = manifest.read_bytes()
    snapshot = {
        "manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_raw_base64": base64.b64encode(raw).decode(),
        "manifest": json.loads(raw),
        "processes": {
            "roles": [
                {
                    "role": "engine",
                    "argv": ["radiance", "--manifest", str(manifest)],
                    "env": {"ROCR_VISIBLE_DEVICES": "0"},
                }
            ]
        },
    }
    session = json.loads((run / "session.json").read_text())
    session["state"] = "stopped"
    (run / "session.json").write_text(json.dumps(session))
    row = {
        "cohort": "dense",
        "arm": "B",
        "state": "stopped",
        "run_dir": str(run),
        "clean_stop": True,
        "canary": "Paris",
        "completion_sha256": "a" * 64,
        "build_identity": {"engine": {"sha256": meta["binary_hash"]}},
        "power_before_stop": {},
    }
    sha = lab.publish_rsc02_arm(
        row,
        snapshot,
        work_root=tmp_path,
        recorded_at="2026-11-20T03:00:00Z",
        host_snapshot=FAKE_HOST,
    )
    store = run_records.RunRecordStore(tmp_path)
    rec = store.get_record(sha)
    assert rec["workload"]["name"] == "rsc02-s15-dense"
    assert rec["metrics"]["decode"]["canary_correct"] == 1.0
    assert store.get_blob(rec["provenance"]["effective_config"]) == raw
    broken = dict(snapshot, manifest_file_sha256="0" * 64)
    with pytest.raises(ValueError, match="manifest.*hash|mismatch"):
        lab.publish_rsc02_arm(row, broken, work_root=tmp_path, host_snapshot=FAKE_HOST)


def test_results_list_today_parity_and_show_by_digest(tmp_path, capsys):
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = measurement_producers.publish(
        work_root=tmp_path,
        run_id="parity-launch",
        recorded_at=stamp,
        identity={
            "engine": "radiance",
            "build_id": "b",
            "binary_sha256": "a" * 64,
            "source_revision": "src",
            "model": "model",
            "layout": "flashnext-4gpu",
        },
        workload={"name": "parity", "definition": {"sequence": "ABBA"}},
        metrics={"decode": {"decode_step_ms": 25.3}, "prefill": {}, "ttft": {}},
        metric_specs={
            "decode": {"decode_step_ms": {"direction": "lower", "gated": True}},
            "prefill": {},
            "ttft": {},
        },
        effective_config=b'{"manifest": true}',
        argv=["bc-serve", "serve"],
        environment={},
        power_settings={},
        inputs={"heat": b"hot-first"},
        host_facts=FAKE_HOST,
        feature_states=STATES,
    )
    assert (
        main(
            [
                "results",
                "list",
                "--work-root",
                str(tmp_path),
                "--today",
                "--engine",
                "radiance",
                "--workload",
                "parity",
                "--json",
            ]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1 and rows[0]["sha256"] == digest
    assert (
        main(["results", "show", digest, "--work-root", str(tmp_path), "--json"]) == 0
    )
    show = json.loads(capsys.readouterr().out)
    assert show["sha256"] == digest
    assert show["record"]["metrics"]["decode"]["decode_step_ms"] == 25.3
    assert {i["role"] for i in show["record"]["provenance"]["inputs"]} == {"heat"}


def test_needle_cli_publishes_instead_of_legacy_results_json(
    tmp_path, monkeypatch, capsys
):
    import importlib.util
    from unittest.mock import patch

    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "needle_probe_mrc06", repo / "tools/lab/radiance/needle_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"
    make_run(run)
    monkeypatch.setattr(module, "corpus", lambda: "original fixture documents")
    monkeypatch.setattr(module, "chars_per_token", lambda *args: 4.0)
    monkeypatch.setattr(
        module,
        "case",
        lambda *args: {
            "tokens_target": args[4],
            "depth": args[5],
            "rep": args[6],
            "code": "a-1234-b",
            "hit": True,
            "answer": "a-1234-b",
            "seconds": 0.1,
            "prompt_tokens": args[4],
            "attempts": 1,
        },
    )
    pointer = tmp_path / "needle.json"
    with patch("bigcherry.core.host_facts.collect_host_facts", return_value=FAKE_HOST):
        assert (
            module.main(
                [
                    "--url",
                    "http://127.0.0.1:18450",
                    "--run-dir",
                    str(run),
                    "--work-root",
                    str(tmp_path),
                    "--tokens",
                    "48000",
                    "--depths",
                    "0.1,0.5",
                    "--repeats",
                    "1",
                    "--out",
                    str(pointer),
                ]
            )
            == 0
        )
    captured = json.loads(pointer.read_text())
    assert list(captured) == [
        "record_sha256"
    ]  # pointer only, not a second metrics store
    rec = run_records.RunRecordStore(tmp_path).get_record(captured["record_sha256"])
    assert rec["metrics"]["prefill"]["needle_hit_rate"] == 1.0
    roles = {i["role"] for i in rec["provenance"]["inputs"]}
    assert "corpus" in roles and "probe_raw_result" in roles


def test_image_cli_captures_exact_sent_pixels_and_refusal(tmp_path, monkeypatch):
    import importlib.util
    from unittest.mock import patch

    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "image_probe_mrc06", repo / "tools/lab/radiance/image_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"
    make_run(run)
    transmitted = []

    def fake_ask(url, model, timeout, kind, image_bytes=None):
        transmitted.append(image_bytes)
        return 400, {"error": {"message": "media decoder disabled"}}, 0.2

    monkeypatch.setattr(module, "ask", fake_ask)
    out = tmp_path / "image.json"
    with patch("bigcherry.core.host_facts.collect_host_facts", return_value=FAKE_HOST):
        assert (
            module.main(
                [
                    "--url",
                    "http://127.0.0.1:18450",
                    "--run-dir",
                    str(run),
                    "--work-root",
                    str(tmp_path),
                    "--expect",
                    "refused",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
    record_digest = json.loads(out.read_text())["record_sha256"]
    store = run_records.RunRecordStore(tmp_path)
    rec = store.get_record(record_digest)
    assert rec["metrics"]["prefill"]["image_expected_match"] == 1.0
    blob = next(
        i["blob"] for i in rec["provenance"]["inputs"] if i["role"] == "image_payload"
    )
    assert transmitted == [store.get_blob(blob)]


def test_calibration_ingest_cli_refuses_no_native_success(tmp_path, capsys):
    run = tmp_path / "run"
    make_run(run, state="stopped", events=[])
    assert (
        main(
            [
                "results",
                "ingest-calibration",
                "--run-dir",
                str(run),
                "--work-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 2
    )
    assert "calibration" in capsys.readouterr().err.lower()
    assert run_records.RunRecordStore(tmp_path).query() == []


def test_probe_wrong_port_refuses_before_network(tmp_path, monkeypatch):
    import importlib.util

    from bigcherry.core.run_records import RecordError

    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "image_probe_port_mrc06", repo / "tools/lab/radiance/image_probe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = tmp_path / "run"
    make_run(run)
    requests = []
    monkeypatch.setattr(mod, "ask", lambda *args, **kwargs: requests.append(1))
    with pytest.raises(RecordError, match="port differs"):
        mod.main(
            [
                "--url",
                "http://127.0.0.1:18451",
                "--run-dir",
                str(run),
                "--work-root",
                str(tmp_path),
            ]
        )
    assert requests == []


def test_noncalibration_lab_requires_measured_sidecar_and_sealed_native_session(tmp_path):
    from bigcherry.bench import radiance_lab_records as lab
    run = tmp_path / "run"
    manifest, _ = make_run(run, state="stopped")
    sidecar = run / "lab-measurement.json"
    source = {
        "schema_version": 1, "run_id": "launch-v1",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "recorded_at": "2026-11-20T03:00:00Z",
        "workload": {"name": "native-lab", "definition": {"profile": "observe"}},
        "metrics": {"decode": {"step_ms": 42}, "prefill": {}, "ttft": {}},
        "metric_specs": {"decode": {"step_ms": {"direction": "lower", "gated": True}}, "prefill": {}, "ttft": {}},
    }
    sidecar.write_text(json.dumps(source))
    sha = lab.publish_native_lab(run, sidecar, work_root=tmp_path, host_snapshot=FAKE_HOST)
    store = run_records.RunRecordStore(tmp_path)
    row = store.get_record(sha)
    assert row["metrics"]["decode"]["step_ms"] == 42
    assert row["validity"]["state"] == "unverified"
    assert any(item["role"] == "native_lab_measurement" for item in row["provenance"]["inputs"])
    altered = dict(source, manifest_sha256="0"*64)
    sidecar.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="manifest"):
        lab.publish_native_lab(run, sidecar, work_root=tmp_path, host_snapshot=FAKE_HOST)
    manifest.unlink()
    assert store.get_record(sha) == row
