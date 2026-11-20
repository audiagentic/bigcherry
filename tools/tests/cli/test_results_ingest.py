"""MRC06 typed native Radiance measurement ingestion."""

from pathlib import Path

from bigcherry.cli.main import main


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
