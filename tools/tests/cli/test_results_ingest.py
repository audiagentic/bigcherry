"""MRC06 typed native and experiment result ingestion commands."""

from pathlib import Path

from bigcherry.__main__ import main


def test_results_ingest_experiment_routes_to_typed_bundle_api(monkeypatch, tmp_path, capsys):
    from bigcherry.experiment import bundle

    seen = []
    def capture(root, measurement, *, work_root=None):
        seen.append((root, measurement, work_root))
        return "e" * 64
    monkeypatch.setattr(bundle, "record_measurement_file", capture)
    folder = tmp_path / "bundle"
    sidecar = folder / "measurement.json"
    rc = main([
        "results", "ingest-experiment", "--bundle", str(folder),
        "--measurement-file", str(sidecar), "--work-root", str(tmp_path),
    ])
    assert rc == 0
    assert seen == [(folder, sidecar, tmp_path)]
    assert "e" * 64 in capsys.readouterr().out


def test_results_ingest_native_lab_routes_to_sealed_owner_adapter(monkeypatch, tmp_path, capsys):
    from bigcherry.bench import radiance_lab_records
    calls = []
    def capture(run_dir, source, *, work_root=None):
        calls.append((run_dir, source, work_root))
        return "d" * 64
    monkeypatch.setattr(radiance_lab_records, "publish_native_lab", capture)
    run, source = tmp_path / "owner", tmp_path / "owner/measure.json"
    assert main(["results", "ingest-native-lab", "--run-dir", str(run),
                 "--measurement-file", str(source), "--work-root", str(tmp_path)]) == 0
    assert calls == [(run, source, tmp_path)]
    assert "d" * 64 in capsys.readouterr().out
