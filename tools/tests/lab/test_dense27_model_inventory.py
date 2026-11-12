"""No-GPU dense27 source checkpoint and format inventory tests."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "tools/lab/dense27"
SPEC = importlib.util.spec_from_file_location("model_inventory", LAB / "model_inventory.py")
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def fixture() -> dict:
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 8,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"]
            + ["linear_attention"] * 3 + ["full_attention"],
            "hidden_size": 5120,
            "num_key_value_heads": 4,
            "num_attention_heads": 24,
            "head_dim": 256,
        },
    }


def test_counts_exact_layers_and_kv_estimate():
    info = mod.inspect_config(fixture(), contexts=[32768], kv_bytes=2)
    assert info["counts"] == {"full_attention": 2, "linear_attention": 6}
    assert info["n_layers"] == 8
    assert info["classical_kv"]["32768"]["logical_full_attention_bytes"] == 256 * 1024 ** 2
    assert info["classical_kv"]["32768"]["per_rank_if_tp2_head_sharded_bytes"] == 128 * 1024 ** 2
    assert info["tp2_even_kv_heads"]
    assert not info["tp3_even_kv_heads"]
    assert info["stage_activation_payload_bytes"]["prefill_2048_rows_bf16"] == 20 * 1024 ** 2


def test_fp8_kv_halves_only_classical_cache():
    fp8 = mod.inspect_config(fixture(), contexts=[32768], kv_bytes=1)
    fp16 = mod.inspect_config(fixture(), contexts=[32768], kv_bytes=2)
    assert fp8["classical_kv"]["32768"]["logical_full_attention_bytes"] * 2 == (
        fp16["classical_kv"]["32768"]["logical_full_attention_bytes"]
    )
    assert "linear attention / GDN recurrent states" in fp8["excluded_from_kv_estimate"]


def test_refuses_unknown_or_inconsistent_architecture():
    data = fixture()
    data["text_config"]["layer_types"][-1] = "unknown_sparse_attention"
    with pytest.raises(ValueError, match="unknown layer types"):
        mod.inspect_config(data, contexts=[8192], kv_bytes=2)
    data = fixture()
    data["text_config"]["num_hidden_layers"] = 64
    with pytest.raises(ValueError, match="num_hidden_layers"):
        mod.inspect_config(data, contexts=[8192], kv_bytes=2)


def test_refuses_bad_head_configuration_and_context():
    data = fixture()
    data["text_config"]["num_key_value_heads"] = 0
    with pytest.raises(ValueError, match="num_key_value_heads"):
        mod.inspect_config(data, contexts=[8192], kv_bytes=2)
    data = fixture()
    with pytest.raises(ValueError, match="contexts"):
        mod.inspect_config(data, contexts=[0], kv_bytes=2)


def test_weight_file_sizes_only_no_file_read(tmp_path):
    a = tmp_path / "control.gguf"
    b = tmp_path / "candidate.gguf"
    a.write_bytes(b"some bytes")
    b.write_bytes(b"other bytes")
    records = mod.inspect_files([f"q8={a}", f"q6={b}"])
    assert [record["file_bytes"] for record in records] == [10, 11]
    assert all("file_gib" in r for r in records)
    with pytest.raises(ValueError, match="duplicate label"):
        mod.inspect_files([f"q8={a}", f"q8={b}"])
    with pytest.raises(ValueError, match="missing weight file"):
        mod.inspect_files([f"unknown={tmp_path / 'missing'}"])


def test_cli_with_fake_config_and_models(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(fixture()), encoding="utf-8")
    model = tmp_path / "control.gguf"
    model.write_bytes(b"fake model")
    result = subprocess.run(
        [sys.executable, str(LAB / "model_inventory.py"), "--config", str(path),
         "--weight-file", f"Q8_0={model}", "--context", "32768"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["n_layers"] == 8
    assert report["weights"][0]["file_bytes"] == len(b"fake model")
    assert "File sizes are not measured VRAM" in report["limitations"][0]
