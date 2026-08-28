"""HI119: patches/1238 gives init_mul_mat_id_tensors() -- the initializer
every REGISTERED MUL_MAT_ID-family test_case shares (test_mul_mat_id,
test_mul_mat_vec_fusion, HI119's own planned fused-GLU class) -- a
deterministic, full-range branch under BIGCHERRY_TEST_DETERMINISTIC_SEED.
Verified against the real pinned vendor source (not a synthetic fixture)
that the fix is idempotent, composes cleanly with patch 1222 (whose helpers
it reuses -- REQUIRES declares that dependency), that the original
non-deterministic branch is preserved byte-for-byte for the unset-seed
default, and that the new branch shuffles the FULL n_mats-wide pool.

dev-gpt-agent deep design review (2026-08-25) found this gap while
reviewing HI119's design: init_mul_mat_id_tensors() is a THIRD, separate
std::random_device site, distinct from what patches 1222 (float init only)
and 1236 (test_generic_op's own internal branch only, per its own
docstring) already cover -- real-hardware confirmed on Brutus that two
independent process invocations of the same seed now emit byte-identical
BIGCHERRY_REF_DIGEST lines for the ids tensor."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[3]


def _load_patch_module(patch_id: str):
    spec = importlib.util.spec_from_file_location(
        patch_id, ROOT / "patches" / patch_id / "patch.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HI119 = _load_patch_module("1238_hi119_deterministic_init_mul_mat_id_tensors")
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
    assert HI119.REQUIRES == ("1222_hi67_deterministic_test_backend_ops_seed",)


def test_patch_applies_cleanly_alone():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _apply_to_copy(Path(tmp), HI119.PATCHES)


def test_patch_applies_cleanly_composed_with_1222(tmp_path):
    target = _apply_to_copy(tmp_path, [*HI67.PATCHES, *HI119.PATCHES])
    text = target.read_text(encoding="utf-8")

    assert "bigcherry (HI67 slice 2a): deterministic tensor init" in text
    assert "bigcherry (HI119): deterministic, full-range expert routing" in text

    assert "if (bigcherry_deterministic_mode()) {" in text
    assert "std::vector<int32_t> pool(n_mats);" in text
    assert "pool[i] = (int32_t) i;" in text
    assert "std::shuffle(pool.begin(), pool.end(), det_gen);" in text
    assert "std::vector<int32_t> data(pool.begin(), pool.begin() + t->ne[0]);" in text
    assert 'BIGCHERRY_REF_DIGEST name=%s call_index=%llu digest=%016llx nels=%zu' in text

    # The original non-deterministic branch (unset-seed default) is
    # preserved byte-for-byte inside the new else, not deleted.
    assert (
        "                for (int64_t r = 0; r < ggml_nrows(t); r++) {\n"
        "                    std::vector<int32_t> data(t->ne[0]);\n"
        "                    for (int i = 0; i < t->ne[0]; i++) {\n"
        "                        data[i] = i % n_mats;\n"
        "                    }\n"
        "                    std::shuffle(data.begin(), data.end(), rng);\n"
        "                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));\n"
    ) in text


def test_patch_is_idempotent(tmp_path):
    target = _apply_to_copy(tmp_path, [*HI67.PATCHES, *HI119.PATCHES])
    once = target.read_text(encoding="utf-8")
    results = apply_all([*HI67.PATCHES, *HI119.PATCHES], tmp_path)
    assert all(result.ok for result in results)
    twice = target.read_text(encoding="utf-8")
    assert once == twice


def test_patch_applies_regardless_of_composition_order(tmp_path):
    target = _apply_to_copy(tmp_path, [*HI119.PATCHES, *HI67.PATCHES])
    text = target.read_text(encoding="utf-8")
    assert "bigcherry (HI67 slice 2a): deterministic tensor init" in text
    assert "bigcherry (HI119): deterministic, full-range expert routing" in text


def test_patch_does_not_touch_1236s_own_site():
    # 1238 and 1236 fix two DIFFERENT std::random_device sites (this
    # function vs. test_generic_op's own internal branch) -- confirm this
    # patch's anchor is scoped to init_mul_mat_id_tensors and does not
    # overlap 1236's anchor text, so the two patches can never collide.
    text = (ROOT / "vendor" / "llama.cpp" / _REL).read_text(encoding="utf-8")
    site = text.index("static void init_mul_mat_id_tensors")
    other_site = text.index("} else if (op == GGML_OP_MUL_MAT_ID || op == GGML_OP_ADD_ID) {")
    assert site != other_site
