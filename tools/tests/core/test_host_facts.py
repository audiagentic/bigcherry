"""MRC02: pure, fixture-driven host facts and run-record integration."""

from __future__ import annotations

import hashlib
from pathlib import Path

from bigcherry.core import host_facts, run_manifest, run_records


def write(root: Path, name: str, contents: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def host_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sys = tmp_path / "sys"
    proc = tmp_path / "proc"
    rocm = tmp_path / "rocm"
    env = tmp_path / "environment.toml"
    write(proc, "sys/kernel/osrelease", "7.0.13-arch1-1\n")
    write(proc, "version", "Linux version 7.0.13 compiled with clang 20\n")
    write(
        proc, "meminfo", "MemTotal:        66000000 kB\nMemAvailable:   40000000 kB\n"
    )
    write(sys, "module/amdgpu/version", "6.15.1\n")
    write(rocm, ".info/version", "7.2.4\n")
    env.write_text("[host]\nname='fixture'\n")
    drm = sys / "class/drm/card0"
    drm.mkdir(parents=True)
    gpu = sys / "devices/pci0000:00/0000:00:01.0/0000:03:00.0"
    hwmon = gpu / "hwmon/hwmon3"
    hwmon.mkdir(parents=True)
    (drm / "device").symlink_to(gpu, target_is_directory=True)
    write(gpu, "gfx_target_version", "11.0.0\n")
    write(gpu, "mem_info_vram_total", "25769803776\n")
    write(gpu, "firmware_version", "vcn: 1.2.3\n")
    write(gpu, "pp_dpm_sclk", "0: 500Mhz\n1: 2440Mhz *\n2: 2600Mhz\n")
    write(gpu, "pp_dpm_mclk", "0: 96MHz\n1: 1250MHz *\n")
    write(
        gpu,
        "pp_od_clk_voltage",
        "OD_SCLK:\n0: 500MHz\n1: 3040MHz\nOD_MCLK:\n0: 97MHz\n1: 1250MHz\nOD_VDDGFX_OFFSET:\n-125mV\n",
    )
    write(gpu, "power_dpm_force_performance_level", "auto\n")
    for filename, value in (
        ("power1_cap", "272000000"),
        ("power1_cap_default", "280000000"),
        ("power1_cap_min", "100000000"),
        ("power1_cap_max", "320000000"),
        ("power1_average", "155000000"),
    ):
        write(hwmon, filename, value + "\n")
    for port, speed, width in (
        (gpu, "16.0 GT/s PCIe", 16),
        (gpu.parent, "8.0 GT/s PCIe", 8),
    ):
        for key, val in (
            ("current_link_speed", speed),
            ("current_link_width", str(width)),
            ("max_link_speed", "16.0 GT/s PCIe"),
            ("max_link_width", "16"),
        ):
            write(port, key, val + "\n")
    return sys, proc, rocm, env


def fake_runner(argv: list[str]) -> str:
    if argv == ["agent-lease", "status"]:
        return 'gpu: held by fixture-worker for "test" since 2026-11-19T00:00:00Z, expires 2026-11-19T01:00:00Z'
    if argv == ["dmidecode", "-t", "memory"]:
        return """Memory Device
    Size: 32 GB
    Configured Memory Speed: 5600 MT/s
Memory Device
    Size: No Module Installed
Memory Device
    Size: 32 GB
    Configured Memory Speed: 5600 MT/s
"""
    raise AssertionError(argv)


def test_facts_all_sources_and_pcie_upstream(tmp_path):
    sys, proc, rocm, env = host_fixture(tmp_path)
    data = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        environment_file=env,
        command=fake_runner,
    )
    assert data["kernel_release"] == {"value": "7.0.13-arch1-1", "readable": True}
    assert "clang 20" in data["kernel_build"]["value"]
    assert host_facts._kernel_build("Linux version 7.0 (maker@buildbox) (gcc 13) #1 SMP") == "Linux version 7.0 (gcc 13) #1 SMP"
    assert host_facts._kernel_build(None) is None
    assert data["rocm_version"]["value"] == "7.2.4"
    assert data["amdgpu_version"]["value"] == "6.15.1"
    assert (
        data["environment_sha256"]["value"]
        == hashlib.sha256(env.read_bytes()).hexdigest()
    )
    assert data["gpu_lease_holder"]["value"] == "fixture-worker"
    assert data["mem_available_kib"]["value"] == 40000000
    assert data["memory"]["populated_dimms"]["value"] == 2
    assert data["memory"]["configured_speed_mt_s"]["value"] == [5600]
    assert data["memory"]["profile"]["readable"] is False
    gpu = data["cards"]["0000:03:00.0"]
    assert gpu["arch"]["value"] == "gfx1100"
    assert gpu["bdf"]["value"] == "0000:03:00.0"
    assert gpu["vram_total_bytes"]["value"] == 25769803776
    assert gpu["firmware_version"]["value"] == "vcn: 1.2.3"
    assert gpu["sclk_current_mhz"]["value"] == 2440
    assert gpu["mclk_current_mhz"]["value"] == 1250
    assert gpu["sclk_od_max_mhz"]["value"] == 3040
    assert gpu["mclk_od_max_mhz"]["value"] == 1250
    assert gpu["power"]["cap_uw"]["value"] == 272000000
    assert gpu["power"]["cap_default_uw"]["value"] == 280000000
    assert gpu["power"]["cap_min_uw"]["value"] == 100000000
    assert gpu["power"]["cap_max_uw"]["value"] == 320000000
    assert gpu["pcie"]["card"]["current_speed"]["value"] == "16.0 GT/s PCIe"
    assert gpu["pcie"]["card"]["current_width"]["value"] == 16
    assert len(gpu["pcie"]["upstream"]) == 1
    assert gpu["pcie"]["upstream"][0]["current_width"]["value"] == 8


def test_zero_cap_is_unreadable_not_zero_and_other_bounds_survive(tmp_path):
    sys, proc, rocm, env = host_fixture(tmp_path)
    write(
        sys,
        "devices/pci0000:00/0000:00:01.0/0000:03:00.0/hwmon/hwmon3/power1_cap",
        "0\n",
    )
    data = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        environment_file=env,
        command=fake_runner,
    )
    cap = data["cards"]["0000:03:00.0"]["power"]
    assert cap["cap_uw"]["value"] is None
    assert cap["cap_uw"]["readable"] is False
    assert cap["cap_uw"]["reason"] == "zero_cap_suspect"
    assert cap["cap_default_uw"]["value"] == 280000000
    assert host_facts.power_state(sys)["0000:03:00.0"]["power1_cap_uw"] is None


def test_missing_unreadable_fact_does_not_infer_value(tmp_path):
    sys, proc, rocm, env = host_fixture(tmp_path)
    dev = sys / "devices/pci0000:00/0000:00:01.0/0000:03:00.0"
    (dev / "pp_dpm_sclk").unlink()
    (dev / "mem_info_vram_total").write_text("invalid\n")
    (rocm / ".info/version").unlink()
    env.unlink()
    data = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        environment_file=env,
        command=lambda argv: None,
    )
    gpu = data["cards"]["0000:03:00.0"]
    for fact in (
        gpu["sclk_current_mhz"],
        gpu["vram_total_bytes"],
        data["rocm_version"],
        data["environment_sha256"],
        data["gpu_lease_holder"],
    ):
        assert fact["readable"] is False and fact["value"] is None


def test_card_paths_power_state_and_board_watts(tmp_path):
    sys, _, _, _ = host_fixture(tmp_path)
    assert [bdf for bdf, _ in host_facts.card_paths(sys)] == ["0000:03:00.0"]
    assert host_facts.power_state(sys)["0000:03:00.0"]["power1_cap_uw"] == 272000000
    assert host_facts.board_watts(sys) == {"0000:03:00.0": 155.0}


def test_record_and_run_manifest_host_facts_round_trip(tmp_path, monkeypatch):
    sys, proc, rocm, env = host_fixture(tmp_path)
    facts = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        environment_file=env,
        command=fake_runner,
    )
    store = run_records.RunRecordStore(tmp_path)
    blob = store.put_blob(b"snapshot")
    original = {
        "schema_version": 1,
        "run_id": "run-001",
        "recorded_at": "2026-11-19T10:00:00Z",
        "identity": {
            "engine": "engine-a",
            "build_id": "build-1",
            "binary_sha256": "a" * 64,
            "source_revision": "abc123",
            "model": "model-a",
            "layout": "layout-a",
        },
        "workload": {"name": "test", "definition": {}},
        "metrics": {"decode": {}, "prefill": {}, "ttft": {}},
        "metric_specs": {"decode": {}, "prefill": {}, "ttft": {}},
        "provenance": {
            "effective_config": blob,
            "argv": blob,
            "environment": blob,
            "power_settings": blob,
            "inputs": [],
        },
        "host_facts": {},
        "feature_states": [],
    }
    start = host_facts.snapshot_at_start(facts, captured_at="2026-11-19T09:59:00Z")
    record = run_records.with_host_facts(original, start)
    assert original["host_facts"] == {}
    digest = store.put_record(record)
    assert store.get_record(digest)["host_facts"] == start
    # The lifecycle manifest uses the same platform-owned collector.
    manifest = run_manifest.new_manifest(
        run_id="run-001",
        kind="ab",
        purpose="benchmark",
        plan_item="MRC02",
        engine="engine-a",
        argv=[],
        owner={},
        host=facts,
    )
    assert run_manifest.validate_manifest(manifest)["host"] == facts


def test_default_environment_path_and_runner_errors(tmp_path, monkeypatch):
    sys, proc, rocm, _ = host_fixture(tmp_path)
    env = tmp_path / "custom.toml"
    env.write_text("setting = true\n")
    monkeypatch.setenv("BIGCHERRY_ENVIRONMENT", str(env))
    data = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        command=lambda argv: (_ for _ in ()).throw(OSError("missing tool")),
    )
    assert (
        data["environment_sha256"]["value"]
        == hashlib.sha256(env.read_bytes()).hexdigest()
    )
    assert data["gpu_lease_holder"]["readable"] is False


def test_od_range_driver_srcversion_and_debugfs_firmware(tmp_path):
    sys, proc, rocm, env = host_fixture(tmp_path)
    dev = sys / "devices/pci0000:00/0000:00:01.0/0000:03:00.0"
    write(
        dev,
        "pp_od_clk_voltage",
        "OD_RANGE:\nSCLK: 500MHz 3100MHz\nMCLK: 97MHz 1275MHz\n",
    )
    (dev / "firmware_version").unlink()
    write(sys, "kernel/debug/dri/0/amdgpu_firmware_info", "VCN firmware: 0x000108\n")
    write(sys, "module/amdgpu/srcversion", "abcdef012345\n")
    data = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        environment_file=env,
        command=fake_runner,
    )
    card = data["cards"]["0000:03:00.0"]
    assert card["sclk_od_max_mhz"]["value"] == 3100
    assert card["mclk_od_max_mhz"]["value"] == 1275
    assert card["firmware_version"]["value"] == "VCN firmware: 0x000108"
    assert data["amdgpu_srcversion"]["value"] == "abcdef012345"


def test_missing_pcie_field_is_explicitly_unreadable(tmp_path):
    sys, proc, rocm, env = host_fixture(tmp_path)
    (sys / "devices/pci0000:00/0000:00:01.0/max_link_width").unlink()
    data = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        environment_file=env,
        command=fake_runner,
    )
    port = data["cards"]["0000:03:00.0"]["pcie"]["upstream"][0]
    assert port["max_width"]["readable"] is False
    assert port["max_width"]["value"] is None
    assert port["current_width"]["value"] == 8


def test_explicit_manifest_collection_and_atomic_record_storage(tmp_path, monkeypatch):
    sys, proc, rocm, env = host_fixture(tmp_path)
    facts = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        environment_file=env,
        command=fake_runner,
    )
    monkeypatch.setattr(host_facts, "collect_host_facts", lambda: facts)
    manifest = run_manifest.new_manifest(
        run_id="mrc02",
        kind="ab",
        purpose="metric",
        plan_item="MRC02",
        engine="generic",
        argv=[],
        owner={},
        collect_host_facts=True,
    )
    assert manifest["host"] == facts
    assert run_manifest.validate_manifest(manifest) == manifest
    # Regular run-manifest construction retains existing explicit-host semantics.
    legacy = run_manifest.new_manifest(
        run_id="other",
        kind="ab",
        purpose="metric",
        plan_item="MRC02",
        engine="generic",
        argv=[],
        owner={},
    )
    assert legacy["host"] == {}
    from unittest.mock import patch

    with patch.object(
        host_facts, "collect_host_facts", side_effect=AssertionError("mutating")
    ):
        try:
            run_manifest.new_manifest(
                run_id="other",
                kind="ab",
                purpose="metric",
                plan_item="MRC02",
                engine="generic",
                argv=[],
                owner={},
                host=facts,
                collect_host_facts=True,
            )
        except run_manifest.ManifestError:
            pass
        else:
            raise AssertionError("ambiguous manifest host inputs must be refused")


def test_lease_idle_is_known_absence_not_missing():
    assert host_facts._lease_holder("gpu: free\n") == {
        "value": None,
        "readable": True,
        "reason": "unheld",
    }


def test_zero_board_average_is_a_real_reading_not_a_missing_cap(tmp_path):
    sys, _, _, _ = host_fixture(tmp_path)
    write(
        sys,
        "devices/pci0000:00/0000:00:01.0/0000:03:00.0/hwmon/hwmon3/power1_average",
        "0\n",
    )
    assert host_facts.board_watts(sys) == {"0000:03:00.0": 0.0}


def test_unknown_pcie_link_speed_is_not_reported_as_a_measurement(tmp_path):
    sys, proc, rocm, env = host_fixture(tmp_path)
    write(
        sys,
        "devices/pci0000:00/0000:00:01.0/0000:03:00.0/current_link_speed",
        "Unknown\n",
    )
    data = host_facts.collect_host_facts(
        sys_root=sys,
        proc_root=proc,
        rocm_root=rocm,
        environment_file=env,
        command=fake_runner,
    )
    speed = data["cards"]["0000:03:00.0"]["pcie"]["card"]["current_speed"]
    assert speed["readable"] is False
    assert speed["value"] is None
    assert speed["reason"] == "malformed"
