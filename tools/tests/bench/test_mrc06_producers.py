"""MRC06 producers: content proofs, offline legacy listing and VA25 validity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bigcherry.bench import workloads
from bigcherry.cli.main import main
from bigcherry.core import engines, feature_registry, host_facts, run_records
from bigcherry.experiment import attestation, bundle


def _manifest(root: Path):
    path = root / "manifest.json"
    build = root / "build"
    build.mkdir()
    binary = build / "bin" / "radiance"
    binary.parent.mkdir()
    binary.write_bytes(b"binary-fixture")
    import hashlib

    meta = {
        "build_id": "b" * 32,
        "binary_hash": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_slice_id": "c" * 32,
        "build_plan_id": "d" * 32,
        "source_commit": "e" * 40,
    }
    meta["runtime_artifacts"] = {binary.name: meta["binary_hash"]}
    (build / "bigcherry-build-metadata-radiance.json").write_text(json.dumps(meta))
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "instance_id": "measured-001",
                    "content_digest": "sha256:" + "f" * 64,
                },
                "spec": {
                    "intent": {"layout": "flashnext-4gpu", "model": "qwen-model"},
                    "processes": {
                        "engine": {
                            "runtime": {"radiance_home": str(build / "radiance_home")},
                            "server": {"port": 9000},
                        }
                    },
                    "handoff": {
                        "placement": {
                            "live_policy_state": {
                                "configured": False,
                                "reason_code": "not_configured",
                            }
                        }
                    },
                },
            }
        )
    )
    states = [
        {
            "engine": "radiance",
            "feature_id": "live_policy",
            "state": "inactive",
            "reason_code": "not_configured",
            "reason_detail": "",
            "observed_at": "2026-11-19T12:00:00Z",
        }
    ]
    states.extend({
        "engine": engines.RADIANCE, "feature_id": name, "state": "inactive",
        "reason_code": "fixture_not_configured", "reason_detail": "",
        "observed_at": "2026-11-19T12:00:00Z",
    } for name in feature_registry.feature_ids(engines.RADIANCE) if name != "live_policy")
    (root / "feature-states.v1.json").write_text(json.dumps(states))
    start = host_facts.snapshot_at_start(
        {"kernel_release": {"value": "test", "readable": True}},
        captured_at="2026-11-19T11:59:59Z", phase="owner_start",
    )
    start_file = root / host_facts.START_RECEIPT_FILE
    start_file.write_text(json.dumps(start))
    (root / "session.json").write_text(json.dumps({
        "schema_version": 1, "state": "running", "manifest": str(path),
        "started_at": "2026-11-19T11:59:58Z",
        "host_facts_start_sha256": hashlib.sha256(start_file.read_bytes()).hexdigest(),
    }))
    return path, meta, binary


def test_workload_producer_pins_manifest_bytes_and_metrics(tmp_path):
    path, meta, _ = _manifest(tmp_path)
    info = workloads.manifest_info(path)
    row = workloads.make_row(
        "prefill-doc-44k",
        {"version": 1, "kind": "doc", "reps": 3},
        {
            "prefill_tps": {
                "median": 3312.0,
                "per_rep": [{"prefill_tps": 3312}, {"prefill_tps": 3330}],
                "unit": "tok/s",
            }
        },
        {"completed_reps": 2, "workload_config": {"kind": "doc", "reps": 3}},
        info,
        run_folder=str(tmp_path / "run"),
        stamp="2026-11-19T12:00:00+00:00",
        power={"gpu": {"cap_uw": 200000000}},
    )
    original = path.read_bytes()
    digest = workloads.publish_record(
        row,
        manifest_bytes=original,
        work_root=tmp_path,
        argv=["bigcherry", "workload", "run", "prefill-doc-44k"],
        environment={"ROCM_PATH": "/opt/fixture"},
        host_snapshot=host_facts.read_owner_start(tmp_path),
    )
    path.unlink()
    store = run_records.RunRecordStore(tmp_path)
    record = store.get_record(digest)
    assert store.get_blob(record["provenance"]["effective_config"]) == original
    assert record["identity"]["binary_sha256"] == meta["binary_hash"]
    assert record["metrics"]["prefill"]["prefill_tps"] == 3312.0
    assert record["workload"]["definition"]["kind"] == "doc"
    assert record["feature_states"][0]["feature_id"] == "live_policy"
    assert not (tmp_path / "results/index.jsonl").exists()


def test_workload_missing_manifest_refused_without_publishing(tmp_path):
    info = workloads.manifest_info(None)
    row = workloads.make_row(
        "cold-27k-ttft",
        {"version": 1},
        {"ttft_ms": {"median": 100.0}},
        {},
        info,
        run_folder=str(tmp_path),
    )
    with pytest.raises((ValueError, OSError), match="manifest|verified"):
        workloads.publish_record(row, work_root=tmp_path)
    assert run_records.RunRecordStore(tmp_path).query() == []


def test_results_list_reads_run_records_only(tmp_path, capsys):
    # A stray pre-record JSONL index is not a results source: nothing lists it, nothing classifies it.
    index = tmp_path / "results" / "index.jsonl"
    index.parent.mkdir()
    index.write_text(json.dumps({"schema": "bigcherry.workload-result.v1", "row_id": "old-1"}) + "\n")
    status = main(["results", "list", "--work-root", str(tmp_path), "--engine", "radiance", "--json"])
    assert status == 0
    assert json.loads(capsys.readouterr().out) == []
    assert not hasattr(workloads, "read_index") and not hasattr(workloads, "append_index")


def test_llama_row_publishes_phase_metrics_and_attestation(tmp_path):
    from bigcherry.bench import llamacpp_records

    binary = tmp_path / "llama-bench"
    binary.write_bytes(b"llama executable")
    row = {
        "timestamp": "20261119_120000",
        "bench_type": "llama-bench",
        "name": "llama-bench-fixture",
        "config_name": "pp512,tg128",
        "model": {"path": "/models/model.gguf", "sha256": "a" * 64},
        "platform": {"name": "gpu2"},
        "device": "ROCm2",
        "build_metadata": {"ref": "abc123"},
        "binary_path": str(binary),
        "toggles_applied": {"n_gpu_layers": 99, "kv_cache_type_k": "q8_0"},
        "command_line": "llama-bench -m /models/model.gguf",
        "results": {"pp512_tps": "3120.5", "tg128_tps": "122.0", "tg128_stddev": "2.2"},
    }
    expected = attestation.ExecutionIdentity("ROCm", ("gfx1100",), ("0000:12:00.0",))
    observed = attestation.ExecutionAttestation(
        "ROCm",
        (attestation.ObservedDevice("gfx1100", "0000:12:00.0"),),
    )
    digest = llamacpp_records.publish_row(
        row,
        work_root=tmp_path,
        expected=expected,
        observed=observed,
        host_snapshot=host_facts.snapshot_at_start(
            {"kernel_release": {"value": "fixture", "readable": True}},
            captured_at="2026-11-19T11:59:59Z"),
    )
    store = run_records.RunRecordStore(tmp_path)
    record = store.get_record(digest)
    assert record["metrics"]["prefill"]["pp512_tps"] == 3120.5
    assert record["metrics"]["decode"]["tg128_tps"] == 122.0
    assert record["validity"]["state"] == "valid"
    proof = next(
        v
        for v in record["provenance"]["inputs"]
        if v["role"] == "execution_attestation"
    )
    assert json.loads(store.get_blob(proof["blob"]))["observed"]["backend"] == "ROCm"


def test_bundle_measurement_marks_missing_attestation_invalid(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    location = tmp_path / "bundle"
    rc = bundle.run_managed(
        location,
        ["python3", "-c", "print('done')"],
        source_revision="a" * 40,
        manifest_hash="b" * 32,
        build_descriptor_hash="c" * 32,
        model=model,
        role="experiment",
    )
    assert rc == 0
    expected = attestation.ExecutionIdentity("ROCm", ("gfx1201",))
    digest = bundle.publish_measurement(
        location,
        metrics={"decode": {"tokens_per_second": 10.0}, "prefill": {}, "ttft": {}},
        metric_specs={"decode": {"tokens_per_second": {"direction": "higher", "gated": False}}, "prefill": {}, "ttft": {}},
        identity={
            "engine": "llama.cpp",
            "build_id": "b1",
            "binary_sha256": "d" * 64,
            "source_revision": "a" * 40,
            "model": "model",
            "layout": "gpu",
        },
        workload={"name": "test", "definition": {"model": "fixture"}},
        expected=expected,
        observed=None,
        work_root=tmp_path,
    )
    store = run_records.RunRecordStore(tmp_path)
    record = store.get_record(digest)
    assert record["validity"]["state"] == "invalid"
    assert "ATTESTATION_MISSING" in record["validity"]["reason_codes"]
    assert any(x["role"] == "experiment_bundle" for x in record["provenance"]["inputs"])
    # Invalid evidence must not seed an accepted comparison.
    from bigcherry.core import results

    with pytest.raises(results.ResultsError, match="invalid|attestation"):
        results.make_baseline(
            store,
            [
                {"sha256": digest, "record": record},
                {"sha256": digest, "record": record},
            ],
            "no",
        )


def test_llama_run_bench_wrapper_replaces_upstream_append(tmp_path):
    from bigcherry.bench import llamacpp_runner

    bench = tmp_path / "bench"
    library = bench / "lib"
    library.mkdir(parents=True)
    (library / "__init__.py").write_text("")
    (library / "config.py").write_text(
        f"RESULTS_FILE = {str(bench / 'results.json')!r}\n"
    )
    (library / "results.py").write_text(
        "def append_result(path, row):\n"
        "    with open(path,'a') as out: out.write('LEGACY APPEND')\n"
    )
    (bench / "results_db.py").write_text(
        "def sync():\n    raise AssertionError('legacy database sync')\n"
    )
    binary = tmp_path / "llama-bench"
    binary.write_bytes(b"actual llama binary")
    row = {
        "bench_type": "llama-bench",
        "results": {"pp512_tps": "2000.0", "tg128_tps": "99.0"},
        "model": {"path": "model.gguf"},
        "build_metadata": {"ref": "abc123"},
        "device": "ROCm0",
        "platform": {"name": "gpu"},
        "binary_path": str(binary),
        "command_line": "llama-bench",
    }
    (bench / "run_bench.py").write_text(
        "from lib.config import RESULTS_FILE\n"
        "from lib.results import append_result\n"
        f"row={row!r}\n"
        "append_result(RESULTS_FILE, row)\n"
        "append_result('family.json', row)\n"
        "from results_db import sync\n"
        "sync()\n"
    )
    (bench / "results.json").write_text('{"results":[]}\n')
    original = (bench / "results.json").read_bytes()
    expected = attestation.ExecutionIdentity("ROCm", ("gfx1100",))
    hashes = llamacpp_runner.run_with_records(
        tmp_path, [], expected=expected, work_root=tmp_path
    )
    assert len(hashes) == 1
    assert (bench / "results.json").read_bytes() == original
    assert not (tmp_path / "family.json").exists()
    record = run_records.RunRecordStore(tmp_path).get_record(hashes[0])
    assert record["validity"]["state"] == "invalid"
    assert "ATTESTATION_MISSING" in record["validity"]["reason_codes"]
    assert record["metrics"]["decode"]["tg128_tps"] == 99.0


def test_bundle_typed_sidecar_cli_and_content_validation(tmp_path, capsys):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"m")
    root = tmp_path / "bundle"
    status = bundle.run_managed(
        root,
        [
            "python3",
            "-c",
            "print('ggml_cuda_init: found 1 ROCm devices (Total VRAM: 24576 MiB):\\n  Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24576 MiB')",
        ],
        source_revision="a" * 40,
        manifest_hash="b" * 32,
        build_descriptor_hash="c" * 32,
        model=model,
        role="measurement",
    )
    assert status == 0
    sidecar = root / "measurement.json"
    doc = {
        "identity": {
            "engine": "llama.cpp",
            "build_id": "bench-1",
            "binary_sha256": "d" * 64,
            "source_revision": "a" * 40,
            "model": "model",
            "layout": "gpu",
        },
        "workload": {"name": "fixture", "definition": {"prompt": 512}},
        "metrics": {"decode": {"tokens_per_second": 12.3}, "prefill": {}, "ttft": {}},
        "metric_specs": {"decode": {"tokens_per_second": {"direction": "higher", "gated": False}},
                         "prefill": {}, "ttft": {}},
        "metric_specs": {
            "decode": {"tokens_per_second": {"direction": "higher", "gated": True}},
            "prefill": {},
            "ttft": {},
        },
        "expected": {"backend": "ROCm", "architectures": ["gfx1100"]},
        "observed": {
            "backend": "ROCm",
            "devices": [{"architecture": "gfx1100", "locator": None}],
        },
    }
    content = json.dumps(doc)
    sidecar.write_text(content)
    status = bundle.main(
        [
            "record",
            str(root),
            "--measurement-file",
            str(sidecar),
            "--work-root",
            str(tmp_path),
        ]
    )
    assert status == 0
    response = json.loads(capsys.readouterr().out)
    record = run_records.RunRecordStore(tmp_path).get_record(response["record_sha256"])
    assert record["validity"]["state"] == "valid"
    assert record["metrics"]["decode"]["tokens_per_second"] == 12.3
    inputs = {i["role"]: i["blob"] for i in record["provenance"]["inputs"]}
    assert (
        run_records.RunRecordStore(tmp_path).get_blob(inputs["measured_results"])
        == content.encode()
    )
    with pytest.raises(bundle.BundleError, match="already"):
        bundle.publish_measurement(
            root,
            metrics=doc["metrics"],
            metric_specs=doc["metric_specs"],
            identity=doc["identity"],
            workload=doc["workload"],
            expected=None,
            observed=None,
            work_root=tmp_path,
        )


def test_bundle_cannot_trust_observed_claim_without_process_evidence(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    root = tmp_path / "bundle"
    assert (
        bundle.run_managed(
            root,
            ["python3", "-c", "print('ordinary measured output')"],
            source_revision="a" * 40,
            manifest_hash="b" * 32,
            build_descriptor_hash="c" * 32,
            model=model,
            role="measurement",
        )
        == 0
    )
    requested = attestation.ExecutionIdentity("ROCm", ("gfx1100",))
    # The sidecar states that ROCm initialized, but the captured child log
    # has no backend initialization. It must remain invalid.
    claim = attestation.ExecutionAttestation(
        "ROCm",
        (attestation.ObservedDevice("gfx1100"),),
    )
    digest = bundle.publish_measurement(
        root,
        work_root=tmp_path,
        expected=requested,
        observed=claim,
        identity={
            "engine": "llama.cpp",
            "build_id": "b1",
            "binary_sha256": "d" * 64,
            "source_revision": "a" * 40,
            "model": "m",
            "layout": "gpu",
        },
        workload={"name": "fixture", "definition": {"prompt": 512}},
        metrics={"decode": {"tokens_per_second": 12.0}, "prefill": {}, "ttft": {}},
        metric_specs={"decode": {"tokens_per_second": {"direction": "higher", "gated": False}}, "prefill": {}, "ttft": {}},
    )
    store = run_records.RunRecordStore(tmp_path)
    record = store.get_record(digest)
    assert record["validity"] == {
        "state": "invalid",
        "reason_codes": ["ATTESTATION_MISSING"],
    }
    with pytest.raises(run_records.RecordError, match="validity"):
        run_records.validate_record(
            {
                **record,
                "validity": {"state": "valid", "reason_codes": ["ATTESTATION_MISSING"]},
            }
        )


def test_release_record_identity_binds_shipped_binary_and_source(tmp_path, monkeypatch):
    import hashlib
    from bigcherry.release import runtime_assets

    path, meta, binary = _manifest(tmp_path)
    root = binary.parent.parent
    shipped = hashlib.sha256(b"stripped bytes").hexdigest()
    binary.write_bytes(b"stripped bytes")
    bundle = {"plan_id": meta["build_plan_id"], "source_slice_id": meta["source_slice_id"],
              "source_binary_sha256": meta["binary_hash"], "binary_sha256": shipped,
              "built_from": {"pinned": "f" * 40}}
    raw = json.dumps(bundle).encode()
    (root / "bundle.json").write_bytes(raw)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"manifest_sha256": hashlib.sha256(raw).hexdigest()}))
    checked = []
    monkeypatch.setattr(runtime_assets, "verify_runtime_tree",
                        lambda *a, **k: checked.append((a, k)))
    actual, inputs = workloads._release_identity(root, meta, receipt)
    assert checked and actual["binary_hash"] == shipped
    assert actual["source_commit"] == "f" * 40
    assert inputs["release_manifest"] == raw
    assert meta["binary_hash"] != shipped
    with pytest.raises(ValueError, match="requires its tagged receipt"):
        workloads._release_identity(root, meta, None)
    (root / "bundle.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="manifest digest"):
        workloads._release_identity(root, meta, receipt)
