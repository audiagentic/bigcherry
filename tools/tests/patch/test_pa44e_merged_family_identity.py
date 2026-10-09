"""PA44-E packaging-equivalence contracts for merged patch families."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all, env_docs  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_RECIPES = _REPO / "config/recipes.toml"
_VENDOR = _REPO / "vendor/llama.cpp"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_Q81 = _load("patch_1307_merged", _REPO / "engines/llamacpp/patches/1307_q81_activation_cache_mmvq/patch.py")
_META = _load("patch_1340_merged", _REPO / "engines/llamacpp/patches/1340_meta_per_device_arena/patch.py")
_P0600 = _load("patch_0600", _REPO / "engines/llamacpp/patches/0600_mmvq_geometry/patch.py")
_P0910 = _load("patch_0910", _REPO / "engines/llamacpp/patches/0910_feature_sets/patch.py")
_P1241 = _load("patch_1241", _REPO / "engines/llamacpp/patches/1241_rd33_mmvq_q8_0_f32_decode/patch.py")
_P1283 = _load("patch_1283", _REPO / "engines/llamacpp/patches/1283_qwen4exp_expert_parallel/patch.py")
_P1303 = _load("patch_1303", _REPO / "engines/llamacpp/patches/1303_attn_kv_tensor_split/patch.py")
_P1326 = _load("patch_1326", _REPO / "engines/llamacpp/patches/1326_sched_async_host_inputs/patch.py")
_P1341 = _load("patch_1341", _REPO / "engines/llamacpp/patches/1341_meta_subset_mirrored/patch.py")

_Q81_RETIRED = (
    "1235_rd09_q81_activation_cache_foundation",
    "1309_rms_norm_mul_q81",
    "1310_act_q81",
    "1311_hc_pre_q81",
    "1312_mul_q81",
)
_META_RETIRED = ("1339_meta_memory_report",)


def _legacy_sequence(module):
    result = []
    for owner, patches, docs in module._PARTS:
        result.extend(patches)
        if docs:
            result.append(env_docs(owner, docs))
    return result


def _shape(patches):
    return tuple(
        (
            patch.path,
            patch.description,
            tuple((edit.id, edit.mode, edit.anchor, edit.guard, edit.text) for edit in patch.edits),
        )
        for patch in patches
    )


def _module_patches(module):
    patches = getattr(module, "PATCHES", None)
    if patches is not None:
        return list(patches)
    patch = getattr(module, "PATCH", None)
    return [patch] if patch is not None else []


def _legacy_part(module, owner):
    for part_owner, patches, docs in module._PARTS:
        if part_owner == owner:
            result = list(patches)
            if docs:
                result.append(env_docs(owner, docs))
            return result
    raise AssertionError(f"missing legacy part {owner}")


def _materialize(root: Path, patches):
    paths = sorted({patch.path for patch in patches})
    for path in paths:
        source = _VENDOR / path
        if source.is_file():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(source, target)
    result = apply_all(list(patches), root)
    assert all(item.ok for item in result), [
        detail.detail for item in result for detail in item.failed
    ]
    return {
        path: (root / path).read_bytes() if (root / path).is_file() else None
        for path in paths
    }


@unittest.skipUnless((_VENDOR / "ggml/src/ggml-cuda/mmvq.cu").exists(), "pinned vendor checkout not present")
class PA44EMergedFamilyIdentity(unittest.TestCase):
    def test_q81_materialized_tree_matches_premerge_order(self):
        common = [
            *_module_patches(_P0600),
            *_module_patches(_P0910),
        ]
        legacy = [
            *common,
            *_legacy_part(_Q81, "1235_rd09_q81_activation_cache_foundation"),
            *_module_patches(_P1241),
            *_legacy_part(_Q81, "1307_q81_activation_cache_mmvq"),
            *_legacy_part(_Q81, "1309_rms_norm_mul_q81"),
            *_legacy_part(_Q81, "1310_act_q81"),
            *_legacy_part(_Q81, "1311_hc_pre_q81"),
            *_legacy_part(_Q81, "1312_mul_q81"),
        ]
        merged = [
            *common,
            *_module_patches(_P1241),
            *_Q81.PATCHES,
        ]
        with tempfile.TemporaryDirectory() as before_td, tempfile.TemporaryDirectory() as after_td:
            before = _materialize(Path(before_td), legacy)
            after = _materialize(Path(after_td), merged)
        self.assertEqual(before, after)

    def test_meta_materialized_tree_matches_premerge_order(self):
        prefix = [
            *_module_patches(_P0910),
            *_module_patches(_P1283),
            *_module_patches(_P1303),
            *_module_patches(_P1326),
        ]
        legacy = [
            *prefix,
            *_legacy_part(_META, "1339_meta_memory_report"),
            *_legacy_part(_META, "1340_meta_per_device_arena"),
            *_module_patches(_P1341),
        ]
        merged = [
            *prefix,
            *_META.PATCHES,
            *_module_patches(_P1341),
        ]
        with tempfile.TemporaryDirectory() as before_td, tempfile.TemporaryDirectory() as after_td:
            before = _materialize(Path(before_td), legacy)
            after = _materialize(Path(after_td), merged)
        self.assertEqual(before, after)

    def test_q81_emits_exact_legacy_patch_sequence(self):
        self.assertEqual(_shape(_Q81.PATCHES), _shape(_legacy_sequence(_Q81)))
        owners = tuple(owner for owner, _, _ in _Q81._PARTS)
        self.assertEqual(
            owners,
            (
                "1235_rd09_q81_activation_cache_foundation",
                "1307_q81_activation_cache_mmvq",
                "1309_rms_norm_mul_q81",
                "1310_act_q81",
                "1311_hc_pre_q81",
                "1312_mul_q81",
            ),
        )

    def test_meta_emits_exact_legacy_patch_sequence(self):
        self.assertEqual(_shape(_META.PATCHES), _shape(_legacy_sequence(_META)))
        self.assertEqual(
            tuple(owner for owner, _, _ in _META._PARTS),
            ("1339_meta_memory_report", "1340_meta_per_device_arena"),
        )

    def test_documented_env_flags_survive_without_auto_injection(self):
        q81_names = tuple(doc.name for doc in _Q81.DOCUMENTED_ENV_DOCS)
        self.assertIn("GGML_HIP_Q8_1_CACHE_MODE", q81_names)
        self.assertIn("BIGCHERRY_RMS_Q81", q81_names)
        self.assertIn("BIGCHERRY_ACT_Q81", q81_names)
        self.assertIn("BIGCHERRY_HC_Q81", q81_names)
        self.assertFalse(hasattr(_Q81, "ENV_DOCS"))

        meta_names = tuple(doc.name for doc in _META.DOCUMENTED_ENV_DOCS)
        self.assertEqual(meta_names, ("BIGCHERRY_META_MEM", "BIGCHERRY_META_PER_DEVICE_ARENA"))
        self.assertFalse(hasattr(_META, "ENV_DOCS"))

    def test_retired_packages_are_not_registered_or_selected(self):
        recipes = _RECIPES.read_text(encoding="utf-8")
        for patch_id in (*_Q81_RETIRED, *_META_RETIRED):
            self.assertNotIn(f'"{patch_id}"', recipes)
            self.assertFalse((_REPO / "patches" / patch_id / "patch.toml").exists())

        self.assertIn('"1307_q81_activation_cache_mmvq"', recipes)
        self.assertIn('"1340_meta_per_device_arena"', recipes)

    def test_survivor_metadata_carries_union_dependencies(self):
        q81 = tomllib.loads(
            (_REPO / "engines/llamacpp/patches/1307_q81_activation_cache_mmvq/patch.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(q81["state"], "validated")
        self.assertEqual(q81["plan-ids"], ["PRBE05", "PRBE06", "QFP13"])
        self.assertEqual(
            q81["requires"],
            ["1241_rd33_mmvq_q8_0_f32_decode", "0910_feature_sets"],
        )

        meta = tomllib.loads(
            (_REPO / "engines/llamacpp/patches/1340_meta_per_device_arena/patch.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["state"], "validated")
        self.assertEqual(meta["plan-ids"], ["MSM01", "MSM02"])
        self.assertEqual(meta["requires"], ["0910_feature_sets"])


if __name__ == "__main__":
    unittest.main()
