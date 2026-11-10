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
            if name == "qwen27b-2xtx":
                self.assertEqual(r.engine_indices, (0, 1))
                self.assertEqual(len(r.helpers), 0)
                self.assertEqual(r.tp_wire, "wht6")
            elif name == "flashnext-r9700":   # the R9700 alone, no helpers
                self.assertEqual(r.engine_indices, (2,))
                self.assertEqual(len(r.helpers), 0)
            elif name == "flashnext-tp2xtx-r9700":   # RAD49: the XTX pair tensor-parallel, the R9700 its one helper
                self.assertEqual(r.engine_indices, (0, 1))
                self.assertEqual([h.index for h in r.helpers], [2])
                self.assertEqual(r.tp_wire, "exact")
            elif name == "flashnext-tp2xtx":   # RAD49: the XTX pair tensor-parallel, the R9700 and the RX 6900 XT its helpers
                self.assertEqual(r.engine_indices, (0, 1))
                self.assertEqual([h.index for h in r.helpers], [2, 3])
                self.assertEqual(r.tp_wire, "exact")
            else:
                self.assertEqual(r.engine_indices, (2,))
                self.assertGreaterEqual(len(r.helpers), 2)

    def test_the_protocol_bound_is_read_from_the_header(self):
        header = (paths.REPO_ROOT / "engines/radiance/components/moe-remote/moe_remote_shm.h").read_text(encoding="utf-8")
        self.assertEqual(rl.max_helpers(), int(re.search(r"kMaxHelpers\s*=\s*(\d+)", header).group(1)))
        self.assertEqual(rl.max_slots(), int(re.search(r"kMaxSlots\s*=\s*(\d+)", header).group(1)))

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


class ResolveTests(unittest.TestCase):
    def test_dense_two_rank_layout_and_shell_environment(self):
        from bigcherry.cli.layout_cmds import _env_lines
        from bigcherry.core.environment import Device

        r = rl.resolve("qwen27b-2xtx", host(*WITH_6900))
        self.assertEqual(r.engine_indices, (0, 1))
        self.assertEqual(r.engine_arch, "gfx1100")
        self.assertEqual(r.engine_plugin, "libr3.so")
        self.assertEqual(r.model, "radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad")
        self.assertEqual(r.server_args[:2], ("--max-model-len", "8192"))
        self.assertEqual(r.common_env, {})
        self.assertEqual(r.helpers, ())
        lines = "\n".join(_env_lines(r))
        self.assertIn("LAYOUT_ENGINE_DEVICES=0,1", lines)
        self.assertIn("LAYOUT_ENGINE_RANKS=2", lines)
        self.assertIn("LAYOUT_HELPER_COUNT=0", lines)
        self.assertIn("LAYOUT_TP_WIRE=wht6", lines)

        moved = host(Device(6, "gfx1100", "x", 24560), Device(9, "gfx1100", "x", 24560))
        self.assertEqual(rl.resolve("qwen27b-2xtx", moved).engine_indices, (6, 9))
        with self.assertRaisesRegex(rl.LayoutError, "wants 2"):
            rl.resolve("qwen27b-2xtx", host(Device(6, "gfx1100", "x", 24560)))
        with self.assertRaisesRegex(rl.LayoutError, "budget overrides require"):
            rl.resolve("qwen27b-2xtx", host(*XTX2), budgets={"gfx1100": 1000})

    def test_dense_layout_rejects_topology_and_argument_injection(self):
        archs = FileRuleTests.ARCHS.replace('roles = ["helper"]', 'roles = ["engine", "helper"]', 1)
        cases = [
            ('engine = { arch = "gfx1100", count = 2 }', "multi-rank engine requires"),
            ('engine = { devices = [0, 2] }\ntp-wire = "wht6"', "share an architecture"),
            ('engine = { arch = "gfx1100", count = 2 }\ntp-wire = "wht6"\nserver-args = ["--p2p", "on"]', "server-args"),
            ('engine = { arch = "gfx1100", count = 2 }\ntp-wire = "wht6"\nserver-args = ["--tp", "1"]', "server-args"),
            ('engine = { arch = "gfx1100", count = 2 }\n[[layout.t.helper]]\narch = "gfx1030"\nbudget-mib = 100', "requires tp-wire"),
            # the fixture header bounds the request slots at 2: two helpers behind two ranks need 4
            ('engine = { arch = "gfx1100", count = 2 }\ntp-wire = "wht6"\n[[layout.t.helper]]\narch = "gfx1201"\nbudget-mib = 100\n'
             '[[layout.t.helper]]\narch = "gfx1030"\nbudget-mib = 100\nplacement = "tail"', "need 4 request slots; the mailbox protocol has 2"),
        ]
        for recipe, expected in cases:
            with self.subTest(recipe=recipe):
                root = FileRuleTests.roots(self, archs, 'schema = 1\n[layout.t]\nmodel = "m.rad"\n'+recipe+'\n')
                with self.assertRaisesRegex(rl.LayoutError, expected):
                    rl.resolve("t", host(*WITH_6900), repo_root=root)

    def test_a_tensor_parallel_engine_with_its_one_helper(self):
        from bigcherry.cli.layout_cmds import _env_lines

        r = rl.resolve("flashnext-tp2xtx-r9700", host(*WITH_6900))
        self.assertEqual((r.engine_indices, r.engine_arch, r.engine_plugin, r.tp_wire), ((0, 1), "gfx1100", "libr3.so", "exact"))
        self.assertEqual([(h.index, h.arch.name, h.placement) for h in r.helpers], [(2, "gfx1201", "head")])
        # no budget-mib in the layout: the whole card less its archs.toml reserve, from the inventory
        self.assertEqual(r.helpers[0].budget_mib, 32624 - rl.load_archs()["gfx1201"].reserve_mib)
        # a derived budget is bc-serve's (the card limit less the helper's own lanes and tables): no MIBS export, the limit flagged
        self.assertTrue(r.helpers[0].budget_derived)
        self.assertEqual(r.common_env, {"R3_MOE_WORKER_COUNT": "1", "R3_MOE_WORKER_ENGINE_RANKS": "2"})
        self.assertEqual(r.server_args, ("--max-model-len", "65536", "--max-num-seqs", "6", "--host-pool-mib", "4096"))
        lines = "\n".join(_env_lines(r))
        for want in ("LAYOUT_ENGINE_DEVICES=0,1", "LAYOUT_ENGINE_RANKS=2", "LAYOUT_TP_WIRE=exact", "LAYOUT_HELPER_COUNT=1",
                     "LAYOUT_SERVER_ARGS=(--max-model-len 65536 --max-num-seqs 6 --host-pool-mib 4096)"):
            self.assertIn(want, lines)
        self.assertIn("TP ranks: 0,1 (gfx1100), wire exact; helper0: card 2 gfx1201", lines)
        # a budget override still applies to the helper's architecture
        self.assertEqual(rl.resolve("flashnext-tp2xtx-r9700", host(*WITH_6900), budgets={"gfx1201": 20480}).helpers[0].budget_mib, 20480)

    def test_a_tensor_parallel_engine_with_two_helpers(self):
        """RAD49: N helpers behind M ranks; the budgets are derived, the tail and the combine are the layout's."""
        from bigcherry.cli.layout_cmds import _env_lines

        r = rl.resolve("flashnext-tp2xtx", host(*WITH_6900))
        archs = rl.load_archs()
        self.assertEqual((r.engine_indices, r.engine_arch, r.tp_wire), ((0, 1), "gfx1100", "exact"))
        self.assertEqual([(h.index, h.arch.name, h.placement, h.budget_mib) for h in r.helpers],
                         [(2, "gfx1201", "head", 32624 - archs["gfx1201"].reserve_mib),
                          (3, "gfx1030", "tail", 16368 - archs["gfx1030"].reserve_mib)])
        self.assertTrue(all(h.budget_derived for h in r.helpers))
        self.assertEqual(r.common_env, {"R3_MOE_WORKER_COUNT": "2", "R3_MOE_WORKER_ENGINE_RANKS": "2",
                                        "R3_MOE_WORKER_PLACEMENT": "tail", "R3_MOE_WORKER_COMBINE": "1"})
        self.assertEqual(r.server_args, ("--max-model-len", "131072", "--max-num-seqs", "8", "--host-pool-mib", "2048"))
        self.assertEqual(r.helper_env(1)["R3_MOE_WORKER_INDEX"], "1")
        self.assertEqual(r.helper_env(1)["ROCR_VISIBLE_DEVICES"], "3")
        lines = "\n".join(_env_lines(r))
        for want in ("LAYOUT_ENGINE_DEVICES=0,1", "LAYOUT_ENGINE_RANKS=2", "LAYOUT_TP_WIRE=exact", "LAYOUT_HELPER_COUNT=2"):
            self.assertIn(want, lines)
        # the slot bound is the protocol's: helpers x ranks
        self.assertLessEqual(len(r.helpers) * len(r.engine_indices), rl.max_slots())

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
        (hdr / "moe_remote_shm.h").write_text("constexpr uint32_t kMaxHelpers = 2;\nconstexpr uint32_t kMaxSlots = 2;\n", encoding="utf-8")
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
        layouts = layouts.replace("[layout.t]\n", '[layout.t]\nmodel = "m.rad"\n', 1)
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
            "positive integer": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1100"\nbudget-mib = 0\n',
            "unknown key": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1100"\nbudget-mib = 100\nflavour = "x"\n',
            "no head helper with a share": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1100"\nbudget-mib = 100\nshare = 0\n',
            "devices lists a card twice": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\ndevices = [0, 0]\nbudget-mib = 100\n',
            "devices alone": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\ndevices = [0]\ncount = 1\nbudget-mib = 100\n',
            "decided by the cards": self.HEAD + '[layout.t.env]\nR3_MOE_WORKER_MIBS = "1"\n',
            "must be an R3_MOE_WORKER": self.HEAD + '[layout.t.env]\nHOME = "/x"\n',
            "silently off": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\ndevices = [0]\nbudget-mib = 1000\n[layout.t.env]\nR3_MOE_WORKER_COMBINE = "1"\n',
            "device 9 is not": '[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\ndevices = [9]\nbudget-mib = 100\n',
        }
        for why, text in bad.items():
            with self.subTest(why=why), self.assertRaisesRegex(rl.LayoutError, why if why != "the protocol bound" else "exceed"):
                self.resolve(text)

    def test_an_absent_budget_is_the_card_less_its_reserve(self):
        r = self.resolve('[layout.t]\nengine = { arch = "gfx1201" }\n[[layout.t.helper]]\narch = "gfx1100"\n')
        self.assertEqual([h.budget_mib for h in r.helpers], [24560 - 1024, 24560 - 1024])

    def test_layout_env_reaches_the_common_settings_for_two_helpers(self):
        r = self.resolve(self.HEAD + '[layout.t.env]\nR3_MOE_WORKER_COMBINE = "1"\n')
        self.assertEqual(r.common_env["R3_MOE_WORKER_COMBINE"], "1")
        self.assertEqual(len(r.helpers), 2)

    def test_the_combine_bound_is_the_headers(self):
        text = (rl._root(None) / "engines/radiance/components/moe-remote/moe_remote_shm.h").read_text(encoding="utf-8")
        self.assertIn("kCombineHelpers = kMaxHelpers;", text)   # the combine is bounded by the helper bound the resolver checks

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

    def test_a_tp_layout_and_a_one_card_layout(self):
        import os
        from unittest.mock import patch
        from bigcherry.engine_adapters.radiance import RadianceAdapter
        with self.adapter_host("/models"), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BC_MODEL_ROOT", None)
            tp = RadianceAdapter().serving_layout("flashnext-tp2xtx-r9700")
            one = RadianceAdapter().serving_layout("flashnext-r9700")
        want = Path("/models") / rl.load_layouts()["flashnext-tp2xtx-r9700"]["model"]
        self.assertEqual(tp.model, want.resolve())
        self.assertEqual(tp.environment, {"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(tp.expected_execution, {"backend": "ROCm", "architectures": ["gfx1100", "gfx1100"],
                                                 "locators": ["0000:03:00.0", "0000:06:00.0"]})
        self.assertEqual(one.environment, {"ROCR_VISIBLE_DEVICES": "2", "HIP_VISIBLE_DEVICES": "0"})
        self.assertEqual(one.expected_execution["locators"], ["0000:09:00.0"])

    def test_bc_model_root_wins_and_an_unknown_layout_is_refused(self):
        import os
        from unittest.mock import patch
        from bigcherry.engine_adapters.radiance import RadianceAdapter
        with self.adapter_host("/models"), patch.dict(os.environ, {"BC_MODEL_ROOT": "/elsewhere"}):
            self.assertTrue(str(RadianceAdapter().serving_layout("flashnext-r9700").model).startswith("/elsewhere/"))
            with self.assertRaisesRegex(ValueError, "unknown layout"):
                RadianceAdapter().serving_layout("no-such-layout")
