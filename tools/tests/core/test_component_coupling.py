"""Radiance bump-boundary guard: declarations must match owned sources and the target."""
import dataclasses
import re
import tempfile
import unittest
from pathlib import Path

from bigcherry.core import component_coupling, components, engines, paths

ROOT = paths.REPO_ROOT
PATCH_ROOT = ROOT / "engines/radiance/patches"


class ComponentCouplingTests(unittest.TestCase):
    def test_every_radiance_component_coupling_is_declared(self):
        declared = components.load_all(ROOT, engines.RADIANCE)
        self.assertEqual(len(declared), 7)
        for name, component in declared.items():
            with self.subTest(component=name):
                self.assertEqual(
                    component_coupling.problems(component, patch_root=PATCH_ROOT), [])

    def test_undeclared_header_env_and_flag_refuse(self):
        base = components.load_all(ROOT, engines.RADIANCE)["libr3"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bridge.cpp").write_text(
                '#include "rad_abi.h"\n#include "rad_future.h"\n'
                'const char* a = "--future-option";\n'
                'const char* b = "RADIANCE_FUTURE_ENV";\n')
            component = dataclasses.replace(base, root=root)
            result = component_coupling.problems(component)
        self.assertTrue(any("undeclared upstream-headers: rad_future.h" in p for p in result), result)
        self.assertTrue(any("undeclared owned-flags: --future-option" in p for p in result), result)
        self.assertTrue(any("undeclared runtime-env: RADIANCE_FUTURE_ENV" in p for p in result), result)

    def test_deleted_upstream_header_and_abi_change_refuse(self):
        component = components.load_all(ROOT, engines.RADIANCE)["bc-gfx12"]
        result = component_coupling.problems(
            component, upstream_paths={"abi/rad_abi.h"},
            upstream_abi=40)
        self.assertTrue(any("upstream file no longer exists: abi/rad_abi_layout.h" in x for x in result), result)
        declared = component.coupling["abi-version"]
        self.assertTrue(any(f"plugin ABI {declared} differs from upstream 40" in x for x in result), result)

    def test_manifest_option_registry_is_the_single_native_mapping(self):
        serving = components.load_all(ROOT, engines.RADIANCE)["serving"]
        self.assertEqual(serving.coupling["option-registry"], "contract/serving.toml")
        import tomllib
        contract = tomllib.loads((serving.root / serving.coupling["option-registry"]).read_text())
        mapped = {f["radiance_option"] for f in contract["field"] if f.get("radiance_option")}
        self.assertGreater(len(mapped), 100)
        self.assertEqual(len(mapped), sum(bool(f.get("radiance_option")) for f in contract["field"]))
        bridge = (serving.root / "v2/engine_admission.cpp").read_text()
        self.assertIn("kEnvBindings", bridge)
        self.assertIn("engine_admission_from_manifest", bridge)


    def test_coupling_schema_requires_complete_fixed_fields(self):
        import textwrap
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fake"
            root.mkdir()
            base = (
                'schema = 1\nname = "fake"\nengine = "radiance"\n'
                'targets = ["gfx1100"]\nbuilt-against = "v1"\n'
                'outputs = ["x.so"]\n[coupling]\n'
            )
            for fields in (
                'upstream-headers = []\n',
                'upstream-headers = []\nunknown = []\n',
                'upstream-headers = []\nabi-version = true\n',
                'upstream-headers = ["../escape.h"]\nabi-version = 35\n',
            ):
                (root / "component.toml").write_text(base + fields)
                with self.subTest(fields=fields), self.assertRaises(components.ComponentError):
                    components.load(root, "radiance")

    def test_missing_coupling_is_not_silently_accepted_by_gate(self):
        base = components.load_all(ROOT, engines.RADIANCE)["libr3"]
        result = component_coupling.problems(dataclasses.replace(base, coupling={}))
        self.assertTrue(any("missing [coupling]" in message for message in result), result)

    def test_owned_wire_version_drift_refuses_even_if_upstream_abi_matches(self):
        component = components.load_all(ROOT, engines.RADIANCE)["moe-remote"]
        self.assertIn("moe_remote_shm.h::kVersion = 26", component.coupling["wire-constants"])
        modified = dataclasses.replace(
            component,
            coupling={**component.coupling,
                      "wire-constants": ("moe_remote_shm.h::kVersion = 27",)},
        )
        result = component_coupling.problems(modified)
        self.assertTrue(any("wire constant" in message for message in result), result)

    def test_declaration_without_target_abi_is_failure(self):
        component = components.load_all(ROOT, engines.RADIANCE)["bc-gfx12"]
        result = component_coupling.problems(component, upstream_paths=set(component.coupling["upstream-files"]))
        self.assertTrue(any("ABI" in message for message in result), result)


if __name__ == "__main__":
    unittest.main()
