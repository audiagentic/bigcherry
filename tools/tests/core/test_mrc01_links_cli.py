"""MRC01 WSL01 typed lifecycle edge and immutable operator contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from bigcherry.cli.main import main
from bigcherry.core import run_manifest, run_records

sys.path.insert(0, str(Path(__file__).parent))
from test_run_records import fixture_record


def _manifest(run_dir: Path, run_id="run-001", engine="engine-a"):
    run_dir.mkdir(parents=True)
    manifest = run_manifest.new_manifest(
        run_id=run_id, kind="replay", purpose="measured run",
        plan_item=None, engine=engine, argv=["run"], owner={"pid": 1},
    )
    run_manifest.write_manifest(run_dir, manifest)
    return manifest


def test_typed_link_round_trip_and_reverse_lookup(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    record = fixture_record(store)
    digest = store.put_record(record)
    directory = tmp_path / "runs" / "replay" / "run-001"
    manifest = _manifest(directory)

    with pytest.raises(run_manifest.ManifestError, match="unlinked"):
        run_manifest.find_run_for_record(tmp_path / "runs", store, digest)
    updated = run_manifest.link_measurement_record(directory, store, digest)
    assert updated["generation"] == manifest["generation"] + 1
    assert updated["references"][-1] == {
        "type": "measurement_record", "id": digest, "role": "measurement"
    }
    assert run_manifest.link_measurement_record(directory, store, digest)["generation"] == updated["generation"]
    assert run_manifest.resolve_measurement_records(directory, store) == [{"sha256": digest, "record": record}]
    found_dir, found_manifest = run_manifest.find_run_for_record(tmp_path / "runs", store, digest)
    assert found_dir == directory
    assert found_manifest["run_id"] == record["run_id"]
    assert store.query(run_id="run-001")[0]["sha256"] == digest


def test_invalid_link_rejected_and_legacy_manifest_remains_readable(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    record = fixture_record(store)
    digest = store.put_record(record)
    directory = tmp_path / "runs" / "replay" / "run-other"
    _manifest(directory, run_id="run-other")
    with pytest.raises(run_manifest.ManifestError, match="run_id"):
        run_manifest.link_measurement_record(directory, store, digest)
    assert not run_manifest.read_manifest(directory)["references"]
    with pytest.raises(run_manifest.ManifestError, match="SHA256"):
        run_manifest.validate_manifest({
            **run_manifest.read_manifest(directory),
            "references": [{"type": "measurement_record", "id": "oops", "role": "measurement"}],
        })
    directory2 = tmp_path / "runs" / "replay" / "run-001"
    _manifest(directory2)
    run_manifest.link_measurement_record(directory2, store, digest)
    store.record_path(digest).write_bytes(b"tampered")
    with pytest.raises(run_records.RecordError):
        run_manifest.resolve_measurement_records(directory2, store)
    with pytest.raises(run_records.RecordError):
        run_manifest.find_run_for_record(tmp_path / "runs", store, digest)


def test_results_put_query_show_and_real_content_recovery_after_delete(tmp_path, capsys):
    seed = run_records.RunRecordStore(tmp_path / "seed")
    rec = fixture_record(seed)
    source = tmp_path / "run-source"
    source.mkdir()
    payloads = []
    for key, ref in [
        ("effective_config", rec["provenance"]["effective_config"]),
        ("argv", rec["provenance"]["argv"]),
        ("environment", rec["provenance"]["environment"]),
        ("power_settings", rec["provenance"]["power_settings"]),
        ("heat", rec["provenance"]["inputs"][0]["blob"]),
    ]:
        f = source / key
        f.write_bytes(seed.get_blob(ref))
        payloads.extend(["--blob", f"{ref['sha256']}={f}"])
    document = source / "record.json"
    document.write_text(json.dumps(rec), encoding="utf-8")
    run_dir = source / "lifecycle"
    _manifest(run_dir)
    args = ["results", "put", "--work-root", str(tmp_path / "output"),
            "--record", str(document), "--run-dir", str(run_dir), *payloads, "--json"]
    assert main(args) == 0
    digest = json.loads(capsys.readouterr().out)["sha256"]
    assert digest == hashlib.sha256(run_records.canonical_record_bytes(rec)).hexdigest()
    assert main(["results", "query", "--work-root", str(tmp_path / "output"),
                 "--run-id", "run-001", "--metric", "prefill.tokens_per_second", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["sha256"] for row in rows] == [digest]
    for f in source.iterdir():
        if f.is_file():
            f.unlink()
    assert main(["results", "show", digest, "--work-root", str(tmp_path / "output"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["record"] == rec
    assert run_manifest.resolve_measurement_records(run_dir, run_records.RunRecordStore(tmp_path / "output"))[0]["record"] == rec


def test_put_does_not_trust_blob_name_or_allow_incomplete_record(tmp_path, capsys):
    store = run_records.RunRecordStore(tmp_path / "seed")
    rec = fixture_record(store)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert main(["results", "put", "--record", str(path), "--work-root", str(tmp_path / "out")]) == 2
    assert "unsafe" in capsys.readouterr().err
    file = tmp_path / "wrong"
    file.write_bytes(b"wrong data")
    ref = rec["provenance"]["effective_config"]["sha256"]
    assert main(["results", "put", "--record", str(path), "--work-root", str(tmp_path / "out"),
                 "--blob", f"{ref}={file}"]) == 2
    assert "sha256" in capsys.readouterr().err
