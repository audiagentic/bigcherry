"""HI119: patches/1239 (test_bigcherry_moe_glu_fusion) and 1240
(--moe-glu-file CLI hook) against the real pinned vendor source.

Real-hardware validation (Brutus, 2026-08-25) already confirmed these
compile and run correctly, and that a --moe-glu-file-driven instance
produces the identical recorded dispatch signature (fusion=GATE, op=GLU,
matching HI108's real blocked dispatch shape exactly) as the statically-
registered instance -- see the ledger event and HI119.md for the full
real-hardware trail. This offline suite pins the source-level contract so
that trail can't silently regress."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[3]
_REL = "tests/test-backend-ops.cpp"

_CHAIN_IDS = (
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1236_hi105_deterministic_mul_mat_id_ids",
    "1238_hi119_deterministic_init_mul_mat_id_tensors",
    "1239_hi119_fused_moe_glu_test_case",
    "1240_hi119_moe_glu_file_cli",
)


def _load_patch_module(patch_id: str):
    spec = importlib.util.spec_from_file_location(
        patch_id, ROOT / "patches" / patch_id / "patch.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULES = {pid: _load_patch_module(pid) for pid in _CHAIN_IDS}


def _all_patches():
    patches = []
    for pid in _CHAIN_IDS:
        patches.extend(_MODULES[pid].PATCHES)
    return patches


def _apply_to_copy(tmp_path: Path, patches) -> Path:
    vendor = ROOT / "vendor" / "llama.cpp"
    target = tmp_path / _REL
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor / _REL, target)
    results = apply_all(patches, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]
    return target


def test_1239_requires_1222_and_1238():
    assert _MODULES["1239_hi119_fused_moe_glu_test_case"].REQUIRES == (
        "1222_hi67_deterministic_test_backend_ops_seed",
        "1238_hi119_deterministic_init_mul_mat_id_tensors",
    )


def test_1240_requires_1222_1238_1239():
    assert _MODULES["1240_hi119_moe_glu_file_cli"].REQUIRES == (
        "1222_hi67_deterministic_test_backend_ops_seed",
        "1238_hi119_deterministic_init_mul_mat_id_tensors",
        "1239_hi119_fused_moe_glu_test_case",
    )


def test_full_chain_applies_cleanly(tmp_path):
    _apply_to_copy(tmp_path, _all_patches())


def test_full_chain_is_idempotent(tmp_path):
    target = _apply_to_copy(tmp_path, _all_patches())
    once = target.read_text(encoding="utf-8")
    results = apply_all(_all_patches(), tmp_path)
    assert all(result.ok for result in results)
    twice = target.read_text(encoding="utf-8")
    assert once == twice


def test_new_struct_shares_activation_and_ids_tensor_objects(tmp_path):
    # ggml_cuda_should_fuse_mul_mat compares ffn_up->src[1]/src[2] against
    # ffn_gate->src[1]/src[2] by POINTER (real-hardware confirmed via the
    # HI119 design review) -- gate_mm/up_mm must be built from the exact
    # same `cur`/`ids` C++ variables, not two separately-allocated
    # same-shape tensors.
    target = _apply_to_copy(tmp_path, _all_patches())
    text = target.read_text(encoding="utf-8")
    struct_start = text.index("struct test_bigcherry_moe_glu_fusion : public test_case {")
    struct_end = text.index("\n};", struct_start)
    body = text[struct_start:struct_end]
    assert "ggml_mul_mat_id(ctx, gate_w, cur, ids)" in body
    assert "ggml_mul_mat_id(ctx, up_w, cur, ids)" in body


def test_glu_is_the_terminal_output_not_followed_by_anything(tmp_path):
    # dev-gpt-agent review finding: test_mul_mat_vec_fusion's use_id path
    # adds an unconditional post-GLU MUL, so its GLU is never the terminal
    # output there. This class's own output must be the GLU/swiglu_oai
    # result directly, named and returned with nothing after it.
    target = _apply_to_copy(tmp_path, _all_patches())
    text = target.read_text(encoding="utf-8")
    struct_start = text.index("struct test_bigcherry_moe_glu_fusion : public test_case {")
    struct_end = text.index("\n};", struct_start)
    body = text[struct_start:struct_end]
    fused_glu_pos = body.index('ggml_set_name(out, "fused_glu");')
    return_pos = body.index("return out;", fused_glu_pos)
    # Nothing else constructs a new tensor from `out` between naming it and
    # returning it.
    between = body[fused_glu_pos:return_pos]
    assert "ggml_mul(" not in between
    assert "ggml_add(" not in between


def test_registered_instance_matches_hi108s_real_blocked_dispatch_shape(tmp_path):
    target = _apply_to_copy(tmp_path, _all_patches())
    text = target.read_text(encoding="utf-8")
    # k=2048, n=256, n_mats=256, n_used=8 -- HI108's real routed dispatch
    # (7ef2471585a5aa6fbb49384efe566ac5) exactly.
    assert "GGML_TYPE_Q8_0, glu_op, /*k=*/2048, /*n=*/256, /*m=*/1, /*n_mats=*/256, /*n_used=*/8" in text


def test_moe_glu_file_reader_uses_the_same_constructor_argument_order(tmp_path):
    target = _apply_to_copy(tmp_path, _all_patches())
    text = target.read_text(encoding="utf-8")
    func_start = text.index("static std::vector<std::unique_ptr<test_case>> make_test_cases_from_moe_glu_file(")
    func_end = text.index("\n}", func_start)
    body = text[func_start:func_end]
    assert "iss >> k >> n >> m >> n_mats >> n_used >> broadcast;" in body
    assert "new test_bigcherry_moe_glu_fusion(type, glu_op, k, n, m, n_mats, n_used, broadcast != 0)" in body


def test_moe_glu_file_path_takes_priority_over_test_file_and_eval_switch(tmp_path):
    target = _apply_to_copy(tmp_path, _all_patches())
    text = target.read_text(encoding="utf-8")
    selection_start = text.index("if (moe_glu_file_path != nullptr) {")
    selection_end = text.index("\n    }\n\n    filter_test_cases", selection_start)
    body = text[selection_start:selection_end]
    # moe_glu_file_path is checked FIRST.
    assert body.index("moe_glu_file_path != nullptr") < body.index("test_file_path == nullptr")


def test_call_site_threads_the_new_parameter_through(tmp_path):
    target = _apply_to_copy(tmp_path, _all_patches())
    text = target.read_text(encoding="utf-8")
    assert (
        "test_backend(backend.get(), dev, mode, op_names_filter, params_filter, "
        "output_printer.get(), test_file_path, moe_glu_file_path, parallel_workers);"
    ) in text
