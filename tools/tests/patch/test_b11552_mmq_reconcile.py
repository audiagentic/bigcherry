"""Static b11552 contracts for reconciled production MMQ patches."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def _load(patch_id: str):
    path = _REPO / "engines" / "llamacpp" / "patches" / patch_id / "patch.py"
    spec = importlib.util.spec_from_file_location("patch_" + patch_id, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_1006_targets_b11552_explicit_q8_dispatch():
    mod = _load("1006_rdna4_mmq_q6k_codegen_fix")
    assert mod._MMQ_DISPATCH_CALL == (
        "            mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream);"
    )
    assert "forced_J" not in mod._MMQ_DISPATCH_CALL
    activation = next(p for p in mod.PATCHES if p.path.endswith("/mmq.cu"))
    marker = next(e for e in activation.edits if e.id == "mmq-q6k-dispatch-marker")
    assert marker.max_span_lines == 2
    assert marker.guard and "1006_rdna4_mmq_q6k_codegen_fix" in marker.guard


def test_1237_uses_explicit_precision_and_b11552_moe_ncols_opt():
    mod = _load("1237_rd30_moe_mmq_compact_grid")
    patch = next(p for p in mod.PATCHES if p.path.endswith("/mmq.cuh"))
    helper = next(e for e in patch.edits if e.id == "rd30-helpers")
    assert "ggml_prec prec_src1>" in helper.anchor
    assert "prec_src1 = GGML_PREC_Q8" not in helper.anchor
    assert helper.max_span_lines == 3

    source = (
        _REPO / "engines" / "llamacpp" / "patches" /
        "1237_rd30_moe_mmq_compact_grid" / "patch.py"
    ).read_text(encoding="utf-8")
    assert "(sig.ne1[2] * sig.n_expert_used + sig.ne0[2] - 1) / sig.ne0[2]" in source
    assert "rd30_effective_J, fallback, cc, GGML_PREC_Q8" in source


def test_1265_keeps_exact_arch_gate_and_span_contract():
    mod = _load("1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2")
    edit = mod.PATCHES[0].edits[0]
    assert edit.max_span_lines == 2
    assert "GGML_CUDA_CC_OFFSET_AMD + 0x1201" in edit.text
    assert "GGML_CUDA_CC_RDNA2" in edit.text
    assert "GGML_CUDA_CC_IS_RDNA4" not in edit.text
