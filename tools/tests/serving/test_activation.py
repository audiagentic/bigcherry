"""S13 staged v2 lifecycle and explicit power gates, card-free."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from bigcherry.cli import serve
from bigcherry.cli.main import build_parser
from bigcherry.hardware.gpu_power import Setting
from bigcherry.hardware.power_receipt import write_json_atomic
from bigcherry.serving.activation import (
    QualifiedPowerGrant,
    V2Bindings,
    require_power_authorization,
)
from bigcherry.serving.manifest_power import ManifestPower
from bigcherry.serving.supervisor import ServeRefused

BDF = "0000:01:00.0"
DIGEST = "sha256:" + "a" * 64
LAYOUT = SimpleNamespace(engine_bdfs=(BDF,), helper_bdfs=())


def parse(*words):
    return build_parser().parse_args(["serve", *words])


def policy(*, mode="observe", settings=None, name="stock"):
    requested = {BDF: settings} if settings is not None else {}
    return ManifestPower(
        mode=mode, profile=name, profile_digest=DIGEST,
        manifest_digest=DIGEST,
        devices=({"role": "engine", "bdf": BDF,
                  "settings": {"sclk_max_mhz": settings.sclk_max_mhz}
                  if settings is not None else {}},),
        requested=requested,
    )


def grant(p, layout):
    assert layout is LAYOUT
    return QualifiedPowerGrant(
        profile=p.profile,
        profile_digest=p.profile_digest,
        settings=dict(p.requested),
        evidence={bdf: "PWR08 exact-scope hardware record" for bdf in p.requested},
        engine_build_id="sha256:" + "b" * 64,
        helper_build_id="sha256:" + "c" * 64,
        driver_id="amdgpu pinned driver measurement",
    )


def test_observe_is_write_free_even_when_profile_is_explicit():
    require_power_authorization(
        policy(name="xtx-qualified"), LAYOUT,
        explicit_profile="xtx-qualified", qualification_gate=None,
    )
    with pytest.raises(ServeRefused, match="differs from explicit"):
        require_power_authorization(
            policy(), LAYOUT, explicit_profile="some-other-profile",
            qualification_gate=grant,
        )


def test_no_unqualified_or_implicit_apply():
    p = policy(mode="apply", settings=Setting(sclk_max_mhz=2700), name="xtx-qualified")
    with pytest.raises(ServeRefused, match="explicitly named"):
        require_power_authorization(
            p, LAYOUT, explicit_profile=None, qualification_gate=grant,
        )
    with pytest.raises(ServeRefused, match="no trusted S10 qualification"):
        require_power_authorization(
            p, LAYOUT, explicit_profile="xtx-qualified", qualification_gate=None,
        )
    with pytest.raises(ServeRefused, match="differs"):
        require_power_authorization(
            p, LAYOUT, explicit_profile="other", qualification_gate=grant,
        )
    require_power_authorization(
        p, LAYOUT, explicit_profile="xtx-qualified",
        qualification_gate=grant,
    )


def test_qualified_apply_must_cover_exact_settings_and_device_identities():
    p = policy(mode="apply", settings=Setting(sclk_max_mhz=2700), name="xtx-qualified")
    wrong = QualifiedPowerGrant(
        profile="xtx-qualified", profile_digest=DIGEST, settings={},
        evidence={}, engine_build_id="build", helper_build_id="helper",
        driver_id="driver",
    )
    with pytest.raises(ServeRefused, match="BDF/settings"):
        require_power_authorization(
            p, LAYOUT, explicit_profile="xtx-qualified",
            qualification_gate=lambda *a: wrong,
        )
    with pytest.raises(ServeRefused, match="untyped"):
        require_power_authorization(
            p, LAYOUT, explicit_profile="xtx-qualified",
            qualification_gate=lambda *a: True,
        )
    no_evidence = QualifiedPowerGrant(
        profile="xtx-qualified", profile_digest=DIGEST,
        settings={BDF: Setting(sclk_max_mhz=2700)}, evidence={BDF: ""},
        engine_build_id="", helper_build_id="build", driver_id="driver",
    )
    with pytest.raises(ServeRefused, match="identity absent"):
        require_power_authorization(
            p, LAYOUT, explicit_profile="xtx-qualified",
            qualification_gate=lambda *a: no_evidence,
        )


def test_staged_cli_start_routes_only_to_one_v2_adapter(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serve, "_record_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(serve, "_journal_dir", lambda: tmp_path / "journals")
    binding = V2Bindings(source=object(), writer=object(), validator=object())
    calls = []

    def adapter(**kw):
        calls.append(kw)
        return {"schema_version": 1, "run_id": "example", "state": "stopped"}

    monkeypatch.setattr(serve, "production_v2_bindings", lambda: binding)
    monkeypatch.setattr(serve, "start_with_adapters", adapter)
    args = parse(
        "start", "one", "--run-id", "example",
        "--readiness-s", "15", "--terminate-s", "7",
        "--poll-s", "0.25", "--stop-wait-s", "30",
    )
    assert serve.cmd_serve(args) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "stopped"
    assert len(calls) == 1 and calls[0]["source"] is binding.source
    assert calls[0]["writer"] is binding.writer
    assert calls[0]["validator"] is binding.validator
    assert calls[0]["explicit_power_profile"] is None
    assert calls[0]["limits"].stop_wait_s == 30
    assert calls[0]["record_dir"] == tmp_path / "runs"


def test_staged_cli_requires_operator_lifecycle_and_no_v1_fallback(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(serve, "production_v2_bindings", lambda: V2Bindings(
        source=object(), writer=object(), validator=object()
    ))
    monkeypatch.setattr(
        serve, "start_with_adapters",
        lambda **kw: pytest.fail("incomplete limits must not launch"),
    )
    assert serve.cmd_serve(parse("start", "one", "--run-id", "example")) == 2
    assert "no guessed lifecycle" in capsys.readouterr().err
    with pytest.raises(ServeRefused, match="missing strict native S10"):
        serve.start_v2_with_bindings(
            parse(
                "start", "one", "--run-id", "example",
                "--readiness-s", "15", "--terminate-s", "7",
                "--poll-s", "0.25", "--stop-wait-s", "30",
            ),
            None,
        )


def test_verified_v2_stop_routes_only_to_owner_and_waits(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serve, "_record_dir", lambda: tmp_path)
    manifest = tmp_path / "v2.json"
    manifest.write_text(json.dumps({"apiVersion": "bigcherry/v2"}))
    record = {
        "schema_version": 1, "run_id": "example", "state": "running",
        "manifest": str(manifest), "power_receipt": str(tmp_path / "power.json"),
    }
    write_json_atomic(tmp_path / "example.json", record)
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return {**record, "state": "stopped"}

    monkeypatch.setattr(serve, "request_stop", request)
    assert serve.cmd_serve(parse(
        "stop", "example", "--poll-s", "0.25", "--stop-wait-s", "30"
    )) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "stopped"
    assert calls[0][0] == ("example", tmp_path)
    assert calls[0][1]["limits"].stop_wait_s == 30
    assert not hasattr(calls[0][1]["limits"], "readiness_s")

    # Never send a cross-process stop request to a v1 legacy run.
    manifest.write_text(json.dumps({"apiVersion": "bigcherry/v1"}))
    assert serve.cmd_serve(parse(
        "stop", "example", "--poll-s", "0.25", "--stop-wait-s", "30"
    )) == 2
    assert "not bigcherry/v2" in capsys.readouterr().err
    assert len(calls) == 1


def test_stop_fails_closed_without_explicit_timeout_or_power_receipt(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(serve, "_record_dir", lambda: tmp_path)
    manifest = tmp_path / "v2.json"
    manifest.write_text(json.dumps({"apiVersion": "bigcherry/v2"}))
    write_json_atomic(tmp_path / "example.json", {
        "run_id": "example", "state": "running",
        "manifest": str(manifest), "power_receipt": "power.json"
    })
    assert serve.cmd_serve(parse("stop", "example")) == 2
    assert "requires --poll-s and --stop-wait-s" in capsys.readouterr().err
    write_json_atomic(tmp_path / "example.json", {
        "run_id": "example", "state": "running",
        "manifest": str(manifest)
    })
    assert serve.cmd_serve(parse(
        "stop", "example", "--poll-s", "0.25", "--stop-wait-s", "30"
    )) == 2
    assert "power-receipt identity missing" in capsys.readouterr().err
