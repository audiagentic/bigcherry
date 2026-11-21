"""Registry lifetime contract for the managed remote-MoE engine extension.

The patch generator stores the production C++ as a raw source string; check
that unregister leaves no previous engine maps and a second registration
cannot replace a still-live plugin. This is a source contract, not a substitute
for the engine-generation integration test required by RUM06.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / "engines/radiance/patches/rad0002_moe_remote_handoff/patch.py"


def _cpp() -> str:
    tree = ast.parse(PATCH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_BC_MOE_REMOTE_CPP"
            for target in node.targets
        ):
            assert isinstance(node.value, ast.Constant)
            return node.value.value
    raise AssertionError("remote-MoE engine source is missing")


def test_unregister_clears_old_engine_maps():
    source = _cpp()
    unregister = source.split('extern "C" int bc_moe_remote_register', 1)[1].split(
        'if (plugin->struct_size', 1
    )[0]
    assert "g_plugin = BcMoeRemotePlugin{};" in unregister
    assert "g_layer_maps.clear();" in unregister


def test_live_registry_refuses_second_plugin():
    source = _cpp()
    registry = source.split('extern "C" int bc_moe_remote_register', 1)[1].split(
        'g_plugin = *plugin;', 1
    )[0]
    assert "if (g_plugin.struct_size) return -1;" in registry


def test_policy_is_re_read_for_each_bound_generation():
    source = _cpp()
    diagnostic = source.split('extern "C" int bc_moe_remote_shadow', 1)[1].split(
        'extern "C" int bc_moe_remote_diag_build', 1
    )[0]
    # The serving adapter owns the immutable policy for a given generation.
    # The extension must not cache the FIRST generation for the DSO lifetime.
    assert "static const bool" not in diagnostic
    assert "return kDiagBuild && remote_policy().shadow;" in diagnostic
    assert "return kDiagBuild && remote_policy().mask_only;" in diagnostic
    skip = source.split('extern "C" int bc_moe_remote_skip_unit', 1)[1]
    assert "const bool on = remote_policy().skip_owned != 0;" in skip
    assert "static const bool on =" not in skip
