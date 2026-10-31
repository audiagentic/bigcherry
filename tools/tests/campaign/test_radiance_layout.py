"""RAD23: card types (archs.toml) and layouts (layouts.toml) as data, resolved against a host's device inventory."""
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bigcherry.core import paths
from bigcherry.core.environment import Device
from bigcherry.engine_adapters import radiance_layout as rl


def host(*devices):
    return SimpleNamespace(devices=tuple(devices))


XTX2 = (Device(0, "gfx1100", "x", 24560), Device(1, "gfx1100", "x", 24560), Device(2, "gfx1201", "r", 32624))
WITH_6900 = XTX2 + (Device(3, "gfx1030", "s", 16368),)


class RepoDataTests(unittest.TestCase):
    def test_the_shipped_files_load_and_every_layout_resolves_on_the_lab_host_shape(self):
        archs = rl.load_archs()
        self.assertEqual({"gfx1201", "gfx1100", "gfx1030"} <= set(archs), True)
        for name in rl.names():
            r = rl.resolve(name, host(*WITH_6900))
            self.assertEqual(r.engine_index, 2)
            self.assertGreaterEqual(len(r.helpers), 2)

    def test_the_protocol_bound_is_read_from_the_header(self):
        header = (paths.REPO_ROOT / "engines/radiance/components/moe-remote/moe_remote_shm.h").read_text(encoding="utf-8")
        self.assertEqual(rl.max_helpers(), int(re.search(r"kMaxHelpers\s*=\s*(\d+)", header).group(1)))

    def test_archs_name_a_plugin_that_a_component_builds(self):
        # every plugin file in archs.toml is the output of a declared radiance component, or the engine's own libr4d
        built = {"libr4d.so"}
        for toml in (paths.REPO_ROOT / "engines/radiance/components").glob("*/component.toml"):
            built |= set(re.findall(r'kernels/([A-Za-z0-9_.-]+\.so)', toml.read_text(encoding="utf-8")))
        for name, a in rl.load_archs().items():
            self.assertIn(a.plugin, built, name)


class ResolveTests(unittest.TestCase):
    def test_two_xtx(self):
        r = rl.resolve("flashnext-2xtx", host(*XTX2))
        self.assertEqual([h.index for h in r.helpers], [0, 1])
        self.assertEqual(r.common_env, {"R3_MOE_WORKER_COUNT": "2", "R3_MOE_WORKER_ENGINE_RANKS": "1", "R3_MOE_WORKER_MIBS": "22528,22528"})
        self.assertEqual(r.helper_env(1)["ROCR_VISIBLE_DEVICES"], "1")
        self.assertEqual(r.helper_env(1)["R3_MOE_WORKER_INDEX"], "1")

    def test_a_third_card_comes_from_the_inventory_not_the_code(self):
        r = rl.resolve("flashnext-2xtx-rdna2-tail", host(*WITH_6900))
        self.assertEqual([(h.index, h.arch.name, h.budget_mib, h.placement) for h in r.helpers],
                         [(0, "gfx1100", 22528, "head"), (1, "gfx1100", 22528, "head"), (3, "gfx1030", 2048, "tail")])
        self.assertEqual(r.common_env["R3_MOE_WORKER_PLACEMENT"], "tail")
        self.assertEqual(r.helper_env(2)["R3_MOE_WORKER_PLUGIN_FILE"], "libr2.so")
        self.assertEqual(r.helper_env(2)["R3_MOE_WORKER_ROWS"], "1")
        # the same layout on a host numbered differently
        moved = (Device(5, "gfx1030", "s", 16368), Device(4, "gfx1100", "x", 24560), Device(7, "gfx1100", "x", 24560), Device(1, "gfx1201", "r", 32624))
        r2 = rl.resolve("flashnext-2xtx-rdna2-tail", host(*sorted(moved, key=lambda d: d.index)))
        self.assertEqual((r2.engine_index, [h.index for h in r2.helpers]), (1, [4, 7, 5]))

    def test_no_card_for_a_group_is_refused_with_the_reason(self):
        with self.assertRaisesRegex(rl.LayoutError, "no gfx1030 card"):
            rl.resolve("flashnext-2xtx-rdna2-tail", host(*XTX2))
        with self.assertRaisesRegex(rl.LayoutError, "no gfx1201 card"):
            rl.resolve("flashnext-2xtx", host(Device(0, "gfx1100", "x", 24560)))

    def test_a_budget_override_by_architecture_and_its_limit(self):
        r = rl.resolve("flashnext-2xtx-rdna2-tail", host(*WITH_6900), budgets={"gfx1030": 4096})
        self.assertEqual(r.common_env["R3_MOE_WORKER_MIBS"], "22528,22528,4096")
        with self.assertRaisesRegex(rl.LayoutError, "exceeds 15344 MiB"):
            rl.resolve("flashnext-2xtx-rdna2-tail", host(*WITH_6900), budgets={"gfx1030": 16000})
        with self.assertRaisesRegex(rl.LayoutError, "no helper group uses"):
            rl.resolve("flashnext-2xtx", host(*XTX2), budgets={"gfx1030": 1024})

    def test_an_unknown_layout_and_an_undescribed_card_type(self):
        with self.assertRaisesRegex(rl.LayoutError, "unknown layout"):
            rl.resolve("nope", host(*XTX2))
        odd = (Device(0, "gfx90a", "mi", 65536), Device(2, "gfx1201", "r", 32624))
        with self.assertRaisesRegex(rl.LayoutError, "no gfx1100 card"):
            rl.resolve("flashnext-2xtx", host(*odd))


class FileRuleTests(unittest.TestCase):
    def roots(self, archs: str, layouts: str):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "engines/radiance").mkdir(parents=True)
        (root / "engines/radiance/archs.toml").write_text(archs, encoding="utf-8")
        (root / "engines/radiance/layouts.toml").write_text(layouts, encoding="utf-8")
        hdr = root / "engines/radiance/components/moe-remote"
        hdr.mkdir(parents=True)
        (hdr / "moe_remote_shm.h").write_text("constexpr uint32_t kMaxHelpers = 2;\n", encoding="utf-8")
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

    def resolve(self, layouts, devices=WITH_6900):
        root = self.roots(self.ARCHS, "schema = 1\n" + layouts)
        return rl.resolve("t", host(*devices), repo_root=root)

    HEAD = '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1100"\nbudget-mib = 1000\n'

    def test_a_valid_minimal_layout(self):
        r = self.resolve(self.HEAD)
        self.assertEqual([h.index for h in r.helpers], [0, 1])

    def test_rules(self):
        bad = {
            "the protocol bound": self.HEAD + '[[layout.t.helper]]\narch = "gfx1030"\nbudget-mib = 100\nplacement = "tail"\n',
            "a tail group must be": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1030"\nbudget-mib = 100\nplacement = "tail"\n',
            "a tail group is one card": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1030"\nbudget-mib = 100\n[[layout.t.helper]]\narch = "gfx1100"\nbudget-mib = 100\nplacement = "tail"\n',
            "already used": self.HEAD + '[[layout.t.helper]]\ndevices = [0]\nbudget-mib = 100\n',
            "may not be used as engine": '[layout.t]\nengine = { arch = "gfx1100" }\n[[layout.t.helper]]\narch = "gfx1030"\nbudget-mib = 100\n',
            "arch or devices": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\nbudget-mib = 100\n',
            "positive integer": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1100"\n',
            "unknown key": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1100"\nbudget-mib = 100\nflavour = "x"\n',
            "no head helper with a share": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1100"\nbudget-mib = 100\nshare = 0\n',
            "device 9 is not": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\ndevices = [9]\nbudget-mib = 100\n',
        }
        for why, text in bad.items():
            with self.subTest(why=why), self.assertRaisesRegex(rl.LayoutError, why if why != "the protocol bound" else "exceed"):
                self.resolve(text)

    def test_shares_are_emitted_only_when_a_head_differs(self):
        r = self.resolve(self.HEAD.replace("budget-mib = 1000\n", "budget-mib = 1000\nshare = 2\n"))
        self.assertEqual(r.common_env["R3_MOE_WORKER_SHARES"], "2,2")
        self.assertNotIn("R3_MOE_WORKER_SHARES", self.resolve(self.HEAD).common_env)

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
