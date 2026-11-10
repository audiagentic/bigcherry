"""Card-free registration and refusal tests for the new serve command."""

from __future__ import annotations

import json
from types import SimpleNamespace

from bigcherry.cli import serve
from bigcherry.cli.main import build_parser
from bigcherry.hardware.power_receipt import write_json_atomic


def parse(*words):
    return build_parser().parse_args(["serve", *words])


def test_help_and_explicit_subcommands():
    for words in (
        ("start", "dual", "--run-id", "case1"),
        ("explain", "dual", "--profile", "profile.toml", "--power-profile", "stock"),
        ("status", "case1"),
        ("stop", "case1"),
    ):
        args = parse(*words)
        assert args.func is serve.cmd_serve


def test_explain_reuses_cpp_inventory_resolution(monkeypatch, capsys):
    calls = []
    layout = SimpleNamespace(
        resolved=SimpleNamespace(
            name="test", engine_indices=(0, 1), helpers=[SimpleNamespace(index=2)]
        ),
        engine_bdfs=("0000:01:00.0", "0000:02:00.0"),
        helper_bdfs=("0000:03:00.0",),
    )
    monkeypatch.setattr(
        serve.layout_cmds, "_resolve", lambda a: calls.append(a.name) or layout
    )
    assert serve.cmd_serve(parse("explain", "test")) == 0
    doc = json.loads(capsys.readouterr().out)
    assert calls == ["test"]
    assert doc["engine_bdfs"] == list(layout.engine_bdfs)
    assert doc["helper_bdfs"] == list(layout.helper_bdfs)
    assert doc["plan"] == "not_measured"


def test_live_start_stop_refuse_before_shell_or_gpu(monkeypatch, capsys):
    monkeypatch.setattr(serve.layout_cmds, "_resolve", lambda a: 1 / 0)
    assert serve.cmd_serve(parse("start", "layout", "--run-id", "one")) == 2
    assert "S6 is a card-free supervisor skeleton" in capsys.readouterr().err
    assert serve.cmd_serve(parse("stop", "one")) == 2
    assert "cross-process stop is not activated" in capsys.readouterr().err


def test_status_reports_records_without_claiming_live_state(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(serve, "_record_dir", lambda: tmp_path)
    write_json_atomic(tmp_path / "one.json", {"run_id": "one", "state": "running"})
    assert serve.cmd_serve(parse("status", "one")) == 0
    assert json.loads(capsys.readouterr().out)["live_state"] == "unverified"
    assert serve.cmd_serve(parse("status", "../one")) == 2
    assert "invalid run ID" in capsys.readouterr().err


def test_cli_test_seam_uses_one_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "_record_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(serve, "_journal_dir", lambda: tmp_path / "journals")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    layout = SimpleNamespace(engine_bdfs=("0000:01:00.0",), helper_bdfs=())
    monkeypatch.setattr(serve.layout_cmds, "_resolve", lambda a: layout)
    from bigcherry.serving.supervisor import ProcessSpec, ServingPlan

    class Launcher:
        def __init__(self):
            self.roles = []

        def start(self, spec):
            self.roles.append("start")
            return SimpleNamespace(role=spec.role)

        def ready(self, process, spec):
            self.roles.append("ready")

        def stop(self, process):
            self.roles.append("stop")

        def exited(self, process):
            self.roles.append("exit")
            return True

    class Power:
        journal_path = tmp_path / "journals" / "one.json"

        def stale_journals(self):
            return []

        def prepare(self, *args):
            raise AssertionError("observe-only should not prepare power")

    launcher = Launcher()
    args = parse("start", "layout", "--run-id", "one")
    record = serve.start_with_services(
        args,
        planner=lambda l, rid: ServingPlan(
            manifest, (ProcessSpec("engine", ("fake-engine",)),)
        ),
        launcher=launcher,
        wait=lambda check: check(),
        lease_ok=lambda: True,
        power_factory=lambda rid, ok: Power(),
    )
    assert record["state"] == "stopped"
    assert launcher.roles == ["start", "ready", "stop", "exit"]
