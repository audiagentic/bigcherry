"""PA44-B packaging-equivalence tests for the 1344/1345/1347 reference refactors."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import Edit, FilePatch, apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid[:4], _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P1344 = _load("1344_dsv4_hc_grid_index")
P1345 = _load("1345_moe_ids_multiwarp")
P1347 = _load("1347_f32_thin_transposed_mmvf")


def _legacy_1344() -> list[FilePatch]:
    return [
        FilePatch(
            path="ggml/src/ggml-cuda/dsv4-hc.cu",
            language="none",
            edits=(
                Edit(
                    id="legacy-stdlib",
                    anchor=re.escape('#include "dsv4-hc.cuh"\n'),
                    mode="insert_after",
                    text="\n#include <cstdio>   // bigcherry 1344: fprintf\n#include <cstdlib>  // bigcherry 1344: getenv / atoi\n",
                    guard=r"#include <cstdlib>  // bigcherry 1344",
                    expect_matches=1,
                    max_span_lines=2,
                ),
                Edit(
                    id="legacy-kernels",
                    anchor=re.escape(P1344._A_COMB_OP),
                    mode="insert_before",
                    text=P1344._N_KERNELS,
                    guard=r"static __global__ void dsv4_hc_post_grid_f32\(",
                    expect_matches=1,
                    max_span_lines=2,
                ),
                Edit(
                    id="legacy-pre",
                    anchor=re.escape(P1344._A_PRE_LAUNCH),
                    mode="replace",
                    text=P1344._N_PRE_LAUNCH,
                    guard=r"BIGCHERRY_PATCH_HIT patch=1344_dsv4_hc_grid_index path=pre",
                    expect_matches=1,
                    max_span_lines=3,
                ),
                Edit(
                    id="legacy-post",
                    anchor=re.escape(P1344._A_POST_LAUNCH),
                    mode="replace",
                    text=P1344._N_POST_LAUNCH,
                    guard=r"BIGCHERRY_PATCH_HIT patch=1344_dsv4_hc_grid_index path=post",
                    expect_matches=1,
                    max_span_lines=3,
                ),
            ),
        )
    ]


def _legacy_1345() -> list[FilePatch]:
    return [
        FilePatch(
            path="ggml/src/ggml-cuda/mmid.cu",
            language="none",
            edits=(
                Edit(
                    id="legacy-stdlib",
                    anchor=re.escape('#include "mmid.cuh"\n'),
                    mode="insert_after",
                    text="\n#include <cstdio>   // bigcherry 1345: fprintf\n#include <cstdlib>  // bigcherry 1345: getenv / atoi\n",
                    guard=r"#include <cstdlib>  // bigcherry 1345",
                    expect_matches=1,
                    max_span_lines=2,
                ),
                Edit(
                    id="legacy-kernel",
                    anchor=re.escape(P1345._A_LAUNCH_TEMPLATE),
                    mode="insert_before",
                    text=P1345._N_KERNEL,
                    guard=r"static __global__ void bc_mm_ids_helper_mw\(",
                    expect_matches=1,
                    max_span_lines=3,
                ),
                Edit(
                    id="legacy-dispatch",
                    anchor=re.escape(P1345._A_SWITCH),
                    mode="replace",
                    text=P1345._N_SWITCH,
                    guard=r"bigcherry 1345 \(QFP36\): large batches group their rows with several warps per expert",
                    expect_matches=1,
                    max_span_lines=4,
                ),
            ),
        )
    ]


def _legacy_1347() -> list[FilePatch]:
    return [
        FilePatch(
            path="ggml/src/ggml-cuda/ggml-cuda.cu",
            language="none",
            edits=(
                Edit(
                    id="legacy-helpers",
                    anchor=re.escape(P1347._A_FN),
                    mode="insert_before",
                    text=P1347._N_HELPERS,
                    guard=r"static __global__ void bc_f32_thin_transpose\(",
                    expect_matches=1,
                    max_span_lines=2,
                ),
                Edit(
                    id="legacy-dispatch",
                    anchor=re.escape(P1347._A_MMF),
                    mode="replace",
                    text=P1347._N_THIN,
                    guard=r"BIGCHERRY_PATCH_HIT patch=1347_f32_thin_transposed_mmvf",
                    expect_matches=1,
                    max_span_lines=3,
                ),
            ),
        )
    ]


CASES = (
    (
        "1344_dsv4_hc_grid_index",
        P1344,
        "ggml/src/ggml-cuda/dsv4-hc.cu",
        "ggml/src/ggml-cuda/bc-dsv4-hc-grid.cuh",
        '#include "bc-dsv4-hc-grid.cuh"  // bigcherry 1344 implementation\n\n',
        _legacy_1344,
    ),
    (
        "1345_moe_ids_multiwarp",
        P1345,
        "ggml/src/ggml-cuda/mmid.cu",
        "ggml/src/ggml-cuda/bc-moe-ids-multiwarp.cuh",
        '#include "bc-moe-ids-multiwarp.cuh"  // bigcherry 1345 implementation\n\n',
        _legacy_1345,
    ),
    (
        "1347_f32_thin_transposed_mmvf",
        P1347,
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        "ggml/src/ggml-cuda/bc-f32-thin-transposed-mmvf.cuh",
        '#include "bc-f32-thin-transposed-mmvf.cuh"  // bigcherry 1347 implementation\n\n',
        _legacy_1347,
    ),
)


def _copy_target(root: Path, target: str) -> None:
    (root / target).parent.mkdir(parents=True, exist_ok=True)
    copy_pinned(_V / target, root / target)


def _hash_tree(root: Path, *, owned: str | None = None, hook: str | None = None) -> str:
    digest = hashlib.sha256()
    owned_bytes = (root / owned).read_bytes() if owned else b""
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if owned and rel == owned:
            continue
        data = path.read_bytes()
        if hook:
            marker = hook.encode()
            if marker in data:
                self_count = data.count(marker)
                if self_count != 1:
                    raise AssertionError(f"{rel}: expected one owned include, found {self_count}")
                data = data.replace(marker, owned_bytes)
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


@unittest.skipUnless(
    all((_V / target).exists() for _, _, target, _, _, _ in CASES),
    "pinned vendor checkout not present",
)
class PA44BOwnedFileIdentity(unittest.TestCase):
    def test_expanded_source_tree_hash_matches_legacy_layout(self):
        for patch_id, module, target, owned, hook, legacy_factory in CASES:
            with self.subTest(patch_id=patch_id), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                legacy_root = base / "legacy"
                owned_root = base / "owned"
                _copy_target(legacy_root, target)
                _copy_target(owned_root, target)

                legacy_result = apply_all(legacy_factory(), legacy_root)
                self.assertTrue(
                    all(result.ok for result in legacy_result),
                    [edit.detail for result in legacy_result for edit in result.failed],
                )
                owned_result = apply_all(module.PATCHES, owned_root)
                self.assertTrue(
                    all(result.ok for result in owned_result),
                    [edit.detail for result in owned_result for edit in result.failed],
                )

                legacy_hash = _hash_tree(legacy_root)
                expanded_hash = _hash_tree(owned_root, owned=owned, hook=hook)
                self.assertEqual(expanded_hash, legacy_hash)

    def test_validation_state_is_carried_forward(self):
        for patch_id, module, *_ in CASES:
            with self.subTest(patch_id=patch_id):
                metadata = tomllib.loads(
                    (_REPO / "patches" / patch_id / "patch.toml").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["state"], "validated")
                self.assertEqual(module.STATE, "validated")


if __name__ == "__main__":
    unittest.main()
