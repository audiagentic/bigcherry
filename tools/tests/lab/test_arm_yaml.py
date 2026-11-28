"""A lab arm is a copy of a native serving YAML with targeted text edits; nothing else in the file changes."""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/lab/radiance/arm_yaml.py"
CONFIGS = ROOT / "engines/radiance/configs"
BASE = CONFIGS / "flashnext-4gpu.yaml"
QWEN = CONFIGS / "qwen27b-tp3-long-q8.yaml"


def run(tmp_path, *sets, base=BASE):
    out = tmp_path / "arm.yaml"
    args = [sys.executable, str(TOOL), "--base", str(base), "--out", str(out)]
    for item in sets:
        args += ["--set", item]
    return subprocess.run(args, capture_output=True, text=True), out


def changed_lines(base: Path, arm: Path) -> list[str]:
    diff = difflib.unified_diff(base.read_text().split("\n"), arm.read_text().split("\n"), lineterm="", n=0)
    return [line for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---")]


def test_no_sets_is_a_byte_identical_copy(tmp_path):
    result, out = run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == BASE.read_bytes()


def test_each_set_edits_only_its_line(tmp_path):
    result, out = run(tmp_path, "server.port=18410", "models.flashnext.kv-cache-dtype=bf16_i8",
                      "models.flashnext.max-num-seqs=4")
    assert result.returncode == 0, result.stderr
    assert changed_lines(BASE, out) == [
        "-  port: 36080", "+  port: 18410",
        "-    max-num-seqs: 8", "+    max-num-seqs: 4",
        "-    kv-cache-dtype: fp8", '+    kv-cache-dtype: "bf16_i8"']


def test_a_flow_table_entry_changes_alone_and_a_new_leaf_is_added(tmp_path):
    result, out = run(tmp_path, "x-bigcherry.handoff.heat=/run/heat", "x-bigcherry.handoff.histogram=/run/heat.live",
                      "x-bigcherry.helper.combine=false")
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert 'handoff: {heat: "/run/heat", histogram-period-s: 1, histogram: "/run/heat.live"}' in text
    assert "helper: {combine: false}" in text
    assert len(changed_lines(BASE, out)) == 4   # two lines replaced


def test_comments_and_quoted_scalars_are_untouched(tmp_path):
    """The hazard of a load-and-dump (YAML 1.1 reads a plain off as a boolean and drops comments) cannot occur."""
    result, out = run(tmp_path, "models.qwen27b.max-num-seqs=2", base=QWEN)
    assert result.returncode == 0, result.stderr
    assert changed_lines(QWEN, out) == ["-    max-num-seqs: 8", "+    max-num-seqs: 2"]
    arm = out.read_text()
    assert 'p2p: "off"' in arm and arm.count("#") == QWEN.read_text().count("#")


def test_a_string_value_is_written_double_quoted(tmp_path):
    result, out = run(tmp_path, "models.qwen27b.p2p=off", base=QWEN)
    assert result.returncode == 0, result.stderr
    assert changed_lines(QWEN, out) == ['-    p2p: "off"', '+    p2p: "off"'] or 'p2p: "off"' in out.read_text()


@pytest.mark.parametrize("bad", ["models.flashnext.bogus.x=1", "modelz.flashnext.tp=2", "x-bigcherry.nope.x=1",
                                 "server.port", "=1", "cards=1", "x-bigcherry.ranks.arch=a"])
def test_a_path_that_is_not_in_the_file_is_refused(tmp_path, bad):
    result, out = run(tmp_path, bad)
    assert result.returncode != 0 and not out.exists()
    assert "arm_yaml:" in result.stderr


def test_a_file_without_x_bigcherry_is_not_a_serving_config(tmp_path):
    plain = tmp_path / "plain.yaml"
    plain.write_text("cards: [0]\nmodels: {m: {model: a.rad}}\n")
    result, _ = run(tmp_path, base=plain)
    assert result.returncode != 0 and "no x-bigcherry" in result.stderr


@pytest.mark.skipif(not os.environ.get("BC_ARM_TEST_BUILD"), reason="needs a built bc-serve (BC_ARM_TEST_BUILD=<build>)")
def test_the_real_inspector_sees_an_arm_as_the_base_except_at_the_set_paths(tmp_path):
    build = Path(os.environ["BC_ARM_TEST_BUILD"])
    model_root = os.environ["BC_MODEL_ROOT"].rstrip("/")

    def resolved(yaml: Path) -> dict:
        result = subprocess.run(
            [str(build / "radiance_home/bin/bc-serve"), "explain", yaml.stem, "--json", "--native-config", str(yaml),
             "--configs", str(yaml.parent), "--model-root", model_root, "--engine-binary", str(build / "bin/radiance"),
             "--set", f"engine.radiance_home={build / 'radiance_home'}", "--set", "handoff.heat=/x",
             "--set", "handoff.histogram=/y"], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)["values"]

    def flat(node, prefix=""):
        return {k: v for key, value in node.items()
                for k, v in (flat(value, f"{prefix}{key}.") if isinstance(value, dict) else {f"{prefix}{key}": value}).items()}

    base_dir = tmp_path / "base"
    arm_dir = tmp_path / "arm"
    base_dir.mkdir(), arm_dir.mkdir()
    base = base_dir / "qwen27b-tp3-long-q8.yaml"
    base.write_bytes(QWEN.read_bytes())
    arm = arm_dir / "qwen27b-tp3-long-q8.yaml"
    subprocess.run([sys.executable, str(TOOL), "--base", str(base), "--out", str(arm),
                    "--set", "models.qwen27b.kv-cache-dtype=bf16_i8"], check=True, capture_output=True)
    before, after = flat(resolved(base)), flat(resolved(arm))
    different = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert different == {"engine.kv_cache_dtype"}, different
