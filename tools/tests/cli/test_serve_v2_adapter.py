"""Inactive S13 CLI bindings; router remains v1-refusing until cutover."""
from types import SimpleNamespace

from bigcherry.cli import serve_v2, serve
from bigcherry.serving.runtime_adapters import LifecycleLimits

LIMITS = LifecycleLimits(readiness_s=2, terminate_s=1, poll_s=.1, stop_wait_s=3)


def test_start_binds_single_inventory_and_s11_services(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(serve, "_record_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(serve, "_journal_dir", lambda: tmp_path / "journals")
    monkeypatch.setattr(serve_v2.layout_cmds, "_resolve", lambda args: ("actual", args.name))
    def adapter(**kwargs):
        got.update(kwargs)
        return {"state": "stopped"}
    monkeypatch.setattr(serve_v2, "start_with_adapters", adapter)
    result = serve_v2.start(
        SimpleNamespace(name="layout", run_id="case"),
        source="real-source", writer="writer", validator="validator",
        limits=LIMITS, lease_ok=lambda: True)
    assert result == {"state": "stopped"}
    assert got["resolve"]("layout") == ("actual", "layout")
    assert got["source"] == "real-source"
    assert got["writer"] == "writer"
    assert got["validator"] == "validator"
    assert got["record_dir"] == tmp_path / "runs"


def test_stop_forwards_only_owner_request(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(serve, "_record_dir", lambda: tmp_path / "runs")
    def request(run_id, record_dir, **kwargs):
        got.update(run_id=run_id, directory=record_dir, **kwargs)
        return {"state": "stopped"}
    monkeypatch.setattr(serve_v2, "request_stop", request)
    result = serve_v2.stop(SimpleNamespace(run_id="case"), limits=LIMITS)
    assert result["state"] == "stopped"
    assert got["run_id"] == "case"
    assert got["directory"] == tmp_path / "runs"
    assert got["limits"] == LIMITS
