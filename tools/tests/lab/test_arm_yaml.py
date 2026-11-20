"""A lab arm is a copy of a native serving YAML with changed values; a typo is an error, never a new key."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/lab/radiance/arm_yaml.py"
BASE = ROOT / "engines/radiance/configs/flashnext-4gpu.yaml"


def run(tmp_path, *sets):
    out = tmp_path / "arm.yaml"
    args = [sys.executable, str(TOOL), "--base", str(BASE), "--out", str(out)]
    for item in sets:
        args += ["--set", item]
    return subprocess.run(args, capture_output=True, text=True), out


def test_changes_are_typed_and_everything_else_is_kept(tmp_path):
    result, out = run(tmp_path, "server.port=18410", "models.flashnext.kv-cache-dtype=bf16_i8",
                      "x-bigcherry.handoff.heat=/run/heat", "x-bigcherry.handoff.histogram-period-s=1",
                      "x-bigcherry.helper.combine=false")
    assert result.returncode == 0, result.stderr
    arm, base = yaml.safe_load(out.read_text()), yaml.safe_load(BASE.read_text())
    assert arm["server"]["port"] == 18410 and arm["models"]["flashnext"]["kv-cache-dtype"] == "bf16_i8"
    assert arm["x-bigcherry"]["handoff"]["heat"] == "/run/heat"
    assert arm["x-bigcherry"]["handoff"]["histogram-period-s"] == 1
    assert arm["x-bigcherry"]["helper"]["combine"] is False
    arm["server"]["port"] = base["server"]["port"]
    arm["models"]["flashnext"]["kv-cache-dtype"] = base["models"]["flashnext"]["kv-cache-dtype"]
    arm["x-bigcherry"]["handoff"] = base["x-bigcherry"]["handoff"]
    arm["x-bigcherry"]["helper"] = base["x-bigcherry"]["helper"]
    assert arm == base


@pytest.mark.parametrize("bad", ["models.flashnext.bogus.x=1", "modelz.flashnext.tp=2", "x-bigcherry.nope.x=1",
                                 "server.port", "=1"])
def test_a_path_that_is_not_in_the_file_is_refused(tmp_path, bad):
    result, out = run(tmp_path, bad)
    assert result.returncode != 0 and not out.exists()
    assert "arm_yaml:" in result.stderr


def test_a_file_without_x_bigcherry_is_not_a_serving_config(tmp_path):
    plain = tmp_path / "plain.yaml"
    plain.write_text("cards: [0]\nmodels: {m: {model: a.rad}}\n")
    result = subprocess.run([sys.executable, str(TOOL), "--base", str(plain), "--out", str(tmp_path / "o.yaml")],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "no x-bigcherry" in result.stderr
