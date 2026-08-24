"""HI105: patches/1236 gives test_generic_op's GGML_OP_MUL_MAT_ID
initializer a deterministic, full-expert-range branch under
BIGCHERRY_TEST_DETERMINISTIC_SEED. Verified against the real pinned
vendor source (not a synthetic fixture) that the fix is idempotent, that
it composes cleanly with patch 1222 (whose helpers it reuses -- REQUIRES
declares that dependency), that the original non-deterministic branch is
preserved byte-for-byte for the default/ADD_ID case, and that the new
branch shuffles the FULL n_expert-wide pool (not just [0, n_expert_used)
the way the pre-patch code did)."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[2]


def _load_patch_module(patch_id: str):
    spec = importlib.util.spec_from_file_location(
        patch_id, ROOT / "patches" / f"{patch_id}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HI105 = _load_patch_module("1236_hi105_deterministic_mul_mat_id_ids")
HI67 = _load_patch_module("1222_hi67_deterministic_test_backend_ops_seed")

_REL = "tests/test-backend-ops.cpp"


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


def test_requires_declares_1222_dependency():
    assert HI105.REQUIRES == ("1222_hi67_deterministic_test_backend_ops_seed",)


def test_patch_applies_cleanly_alone():
    # 1236's own edit does not touch anything 1222 introduces -- it only
    # calls the helpers 1222 declares -- so it must apply standalone too
    # (compilation correctness under REQUIRES is a build-time concern, not
    # an apply-time one).
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _apply_to_copy(Path(tmp), HI105.PATCHES)


def test_patch_applies_cleanly_composed_with_1222(tmp_path):
    target = _apply_to_copy(tmp_path, [*HI67.PATCHES, *HI105.PATCHES])
    text = target.read_text(encoding="utf-8")

    # Both patches' markers present.
    assert "bigcherry (HI67 slice 2a): deterministic tensor init" in text
    assert "bigcherry (HI105): deterministic, full-expert-range routing" in text

    # 1236's new branch is gated to MUL_MAT_ID specifically, reuses 1222's
    # helpers, and shuffles the FULL n_expert pool (not the pre-patch
    # `data[i] = i % n_expert` self-limiting idiom for i in [0, t->ne[0])).
    assert "if (op == GGML_OP_MUL_MAT_ID && bigcherry_deterministic_mode())" in text
    assert "std::vector<int32_t> pool(n_expert);" in text
    assert "pool[i] = (int32_t) i;" in text
    assert "std::shuffle(pool.begin(), pool.end(), det_gen);" in text
    assert "std::vector<int32_t> data(pool.begin(), pool.begin() + t->ne[0]);" in text
    assert 'BIGCHERRY_REF_DIGEST name=%s call_index=%llu digest=%016llx nels=%zu' in text

    # The original non-deterministic branch (default, and ADD_ID always)
    # is preserved byte-for-byte inside the new else, not deleted.
    assert (
        "                    for (int64_t r = 0; r < ggml_nrows(t); r++) {\n"
        "                        std::vector<int32_t> data(t->ne[0]);\n"
        "                        for (int32_t i = 0; i < t->ne[0]; i++) {\n"
        "                            data[i] = i % n_expert;\n"
        "                        }\n"
        "                        std::shuffle(data.begin(), data.end(), rng);\n"
        "                        ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));\n"
    ) in text

    # Guard: the deterministic branch is nested under op == GGML_OP_MUL_MAT_ID
    # specifically -- ADD_ID keeps using the else branch unconditionally,
    # matching this patch's own documented scope restriction.
    mul_mat_id_idx = text.index("if (op == GGML_OP_MUL_MAT_ID && bigcherry_deterministic_mode())")
    branch_start = text.rindex("} else if (op == GGML_OP_MUL_MAT_ID || op == GGML_OP_ADD_ID) {", 0, mul_mat_id_idx)
    assert branch_start < mul_mat_id_idx


def test_patch_is_idempotent(tmp_path):
    target = _apply_to_copy(tmp_path, [*HI67.PATCHES, *HI105.PATCHES])
    once = target.read_text(encoding="utf-8")
    results = apply_all([*HI67.PATCHES, *HI105.PATCHES], tmp_path)
    assert all(result.ok for result in results)
    twice = target.read_text(encoding="utf-8")
    assert once == twice


def test_patch_applies_regardless_of_composition_order(tmp_path):
    # 1222's edit is anchored on init_tensor_uniform (far earlier in the
    # file); 1236's edit is anchored on test_generic_op's MUL_MAT_ID branch
    # (far later). Neither edit's anchor text overlaps the other's output,
    # so apply order must not matter.
    target = _apply_to_copy(tmp_path, [*HI105.PATCHES, *HI67.PATCHES])
    text = target.read_text(encoding="utf-8")
    assert "bigcherry (HI67 slice 2a): deterministic tensor init" in text
    assert "bigcherry (HI105): deterministic, full-expert-range routing" in text
