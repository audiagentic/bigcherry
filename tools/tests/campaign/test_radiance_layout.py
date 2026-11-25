"""RAD23: card types (archs.toml) and serving configs (the native YAML files, BIGCHERRY_CONFIGS) as data, resolved against a host's device inventory."""
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bigcherry.core import paths
from bigcherry.engine_adapters import radiance_layout as rl


class RepoDataTests(unittest.TestCase):
    def test_the_shipped_files_load(self):
        archs = rl.load_archs()
        self.assertEqual({"gfx1201", "gfx1100", "gfx1030"} <= set(archs), True)
        self.assertTrue(rl.load_layouts())

    def test_the_versioned_native_representation_schema(self):
        schema = rl._wire_schema(None)
        self.assertEqual(schema["wire_version"], 26)
        self.assertEqual(rl.max_helpers(), 8)
        self.assertIsNone(rl.topology_refusal(1, 3))
        self.assertIsNone(rl.topology_refusal(2, 3))
        self.assertIsNone(rl.topology_refusal(8, 24))
        self.assertEqual(rl.topology_refusal(9, 1), "CombineMask")
        self.assertEqual(rl.topology_refusal(8, 25), "ParticipantMask")
        self.assertEqual(rl.topology_refusal(0, 1), "ZeroTopology")

    def test_archs_name_a_plugin_that_a_component_builds(self):
        # every plugin file in archs.toml is the output of a declared radiance component, or the engine's own libr4d
        built = {"libr4d.so"}
        for toml in (paths.REPO_ROOT / "engines/radiance/components").glob("*/component.toml"):
            built |= set(re.findall(r'kernels/([A-Za-z0-9_.-]+\.so)', toml.read_text(encoding="utf-8")))
        for name, a in rl.load_archs().items():
            self.assertIn(a.plugin, built, name)


class BuildTargetsTests(unittest.TestCase):
    """A card type in archs.toml must be a card the helper is built for: the component must not pin its own list of architectures."""

    def test_the_helper_builds_for_every_card_type_in_archs_toml(self):
        moe = paths.REPO_ROOT / "engines/radiance/components/moe-remote"
        cmake = (moe / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("archs.toml", cmake)
        self.assertIn("HIP_ARCHITECTURES ${R3MOE_TARGETS}", cmake)
        toml = (moe / "component.toml").read_text(encoding="utf-8")
        configure = re.search(r"^configure\s*=\s*\{([^}]*)\}", toml, re.M).group(1)
        self.assertNotIn("R3MOE_TARGETS", configure, "the lane build must not pin the helper's architectures; archs.toml decides")
        # no architecture named as a build target in the helper's CMake (comments aside)
        for line in cmake.splitlines():
            if not line.lstrip().startswith("#"):
                self.assertNotRegex(line, r"HIP_ARCHITECTURES\s+gfx|GPU_TARGETS\s+gfx|R3MOE_TARGETS\s+\"gfx", line)


class EnvLinesTests(unittest.TestCase):
    """The shell assignments of a resolved layout (the cards themselves come from bc-serve's detection, not from here)."""

    def test_engine_and_helper_assignments(self):
        from bigcherry.cli.layout_cmds import _env_lines

        archs = rl.load_archs()
        helper = rl.HelperCard(2, archs["gfx1201"], 32624, 30000, "head", 1)
        r = rl.Resolved("t", (0, 1), "gfx1100", "libr3.so", "m.rad", "exact", (), (helper,),
                        {"R3_MOE_WORKER_COUNT": "1", "R3_MOE_WORKER_ENGINE_RANKS": "2"})
        lines = "\n".join(_env_lines(r))
        for want in ("LAYOUT_ENGINE_DEVICES=0,1", "LAYOUT_ENGINE_RANKS=2", "LAYOUT_TP_WIRE=exact", "LAYOUT_HELPER_COUNT=1"):
            self.assertIn(want, lines)
        self.assertEqual(r.helper_env(0)["ROCR_VISIBLE_DEVICES"], "2")
        self.assertIsNone(rl.topology_refusal(len(r.helpers), len(r.engine_indices)))


class FileRuleTests(unittest.TestCase):
    def roots(self, archs: str, layouts: str):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "engines/radiance").mkdir(parents=True)
        (root / "engines/radiance/archs.toml").write_text(archs, encoding="utf-8")
        (root / "engines/radiance/layouts").mkdir()
        (root / "engines/radiance/layouts/test.toml").write_text(layouts, encoding="utf-8")
        schema = root / "engines/radiance/components/moe-remote"
        schema.mkdir(parents=True)
        (schema / "topology-v26.schema.json").write_text(
            (paths.REPO_ROOT / "engines/radiance/components/moe-remote/topology-v26.schema.json"
            ).read_text(encoding="utf-8"), encoding="utf-8")
        return root

    ARCHS = '''schema = 1
[arch.gfx1100]
plugin = "libr3.so"
formats = ["w4nl64"]
rows-kernel = false
reserve-mib = 1024
roles = ["helper"]
[arch.gfx1201]
plugin = "libr4d.so"
formats = ["w4nl64"]
rows-kernel = false
reserve-mib = 2048
roles = ["engine", "helper"]
[arch.gfx1030]
plugin = "libr2.so"
formats = ["w4nl64"]
rows-kernel = true
reserve-mib = 1024
roles = ["helper"]
'''

    def test_the_mask_is_a_typed_v26_refusal(self):
        self.assertIsNone(rl.topology_refusal(8, 1, self.roots(self.ARCHS, "schema = 1\n")))
        self.assertEqual(rl.topology_refusal(9, 1), "CombineMask")
        self.assertEqual(rl.topology_refusal(1, 32), "ParticipantMask")

    def test_archs_file_rules(self):
        for why, archs in {"plugin must be": self.ARCHS.replace("libr3.so", "../libr3.so"), "formats must be": self.ARCHS.replace('formats = ["w4nl64"]', 'formats = ["fp9"]', 1),
                           "roles must be": self.ARCHS.replace('roles = ["helper"]', 'roles = []', 1), "required": self.ARCHS.replace("reserve-mib = 1024\n", "", 1),
                           "not a gfx": self.ARCHS.replace("[arch.gfx1100]", "[arch.card]")}.items():
            with self.subTest(why=why):
                root = self.roots(archs, "schema = 1\n")
                with self.assertRaisesRegex(rl.LayoutError, why):
                    rl.load_archs(root)


if __name__ == "__main__":
    unittest.main()


class ServingLayoutTests(unittest.TestCase):
    """The radiance adapter resolves a benchmark arm's layout as the launcher does: its model under the model root, its engine
    cards as device selectors (ROCR in rank order, HIP 0..n-1) and the locators they must execute on."""

    def adapter_host(self, model_root):
        from contextlib import ExitStack
        from unittest.mock import patch
        from bigcherry.engine_adapters import radiance_layout
        bdfs = ("0000:03:00.0", "0000:06:00.0", "0000:09:00.0", "0000:17:00.0")
        arches = ("gfx1100", "gfx1100", "gfx1201", "gfx1030")
        mem = (24560, 24560, 32624, 16368)
        facts = {"gpus": [{"bdf": bdf, "hip_ordinal": i, "arch": arches[i], "vram_total_mib": mem[i]}
                          for i, bdf in enumerate(bdfs)]}

        def fake_resolve(binary, args):
            name = next(arg.split("=", 1)[1] for arg in args if arg.startswith("layout.name="))
            two = name == "flashnext-tp2xtx-r9700"
            engine_bdfs = list(bdfs[:2] if two else (bdfs[2],))
            helper_bdfs = [bdfs[2]] if two else []
            arch = radiance_layout.load_archs()["gfx1201"]
            helpers = ([{"bdf": bdfs[2], "hip_ordinal": 2, "arch": "gfx1201",
                         "budget_mib": 30000, "plugin_file": arch.plugin,
                         "formats": list(arch.formats), "rows_kernel": arch.rows_kernel,
                         "placement": "head", "share": 1}] if two else [])
            return {"facts": facts,
                    "values": {"layout": {"name": name, "engine_cards": engine_bdfs,
                                          "helper_cards": helper_bdfs},
                               "engine": {"tp_wire": "exact" if two else None}},
                    "process_helpers": helpers}

        stack = ExitStack()
        stack.enter_context(patch("bigcherry.engine_adapters.radiance._host",
                                  return_value=SimpleNamespace(devices=(), model_root=model_root, work_root="/w")))
        stack.enter_context(patch("bigcherry.engine_adapters.radiance_inventory._run", side_effect=fake_resolve))
        return stack
