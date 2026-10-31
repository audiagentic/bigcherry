"""RAD01 PR 5: libr3 is a declared radiance component whose runtime settings match what its source reads."""

import re
import unittest

from bigcherry.core import components, config, engines, paths

#: getenv, and the moe-remote helper's validated readers (worker_config.h: u32/flag, peek_u32/peek_flag)
_GETENV = re.compile(r'(?:getenv|\bu32|\bflag|\bchoice|\bpeek_u32|\bpeek_flag)\("((?:R3|BC)_[A-Z0-9_]+)"')
#: settings the campaign harness reads (engine_adapters/radiance.py), not the plugin source
HARNESS_ONLY = {"R3_MOE_WORKER_DEVICES", "R3_MOE_WORKER_LAYOUT"}


class Libr3ComponentTests(unittest.TestCase):
    def setUp(self):
        self.libr3 = components.load_all(paths.REPO_ROOT, engines.RADIANCE)["libr3"]

    def test_declaration(self):
        self.assertEqual(self.libr3.engine, "radiance")
        self.assertEqual(self.libr3.targets, ("gfx1100",))
        recipes = config.load(engines.load(paths.REPO_ROOT, engines.RADIANCE).recipes_path(paths.REPO_ROOT))
        self.assertEqual(self.libr3.built_against, recipes.pinned)

    def test_every_variable_the_source_reads_is_declared_and_nothing_else(self):
        for name, component in components.load_all(paths.REPO_ROOT, engines.RADIANCE).items():
            read = set()
            for path in component.root.rglob("*"):
                if path.is_file() and path.suffix in {".cpp", ".hip", ".h", ".rw"}:
                    read |= set(_GETENV.findall(path.read_text(encoding="utf-8", errors="replace")))
            self.assertEqual(read, set(component.settings) - HARNESS_ONLY, name)

    def test_only_native_attention_is_production(self):
        production = {s.name for s in self.libr3.settings.values() if s.status == "production"}
        self.assertEqual(production, {"R3_ATTN_NATIVE"})

    def test_content_digest_is_stable(self):
        self.assertEqual(self.libr3.content_digest(), self.libr3.content_digest())

    def test_profiles_restate_nothing_and_set_only_production_settings(self):
        self.assertEqual(components.check_environment({"libr3": self.libr3}, {}, production=True), [])
        problems = components.check_environment(
            {"libr3": self.libr3}, {"R3_ATTN_NATIVE": "1", "R3_GDN_NATIVE": "1", "R3_TYPO": "1"}, production=True)
        self.assertIn("R3_ATTN_NATIVE=1 restates libr3's default", problems)
        self.assertIn("R3_GDN_NATIVE is evaluated in libr3; a production profile may not set it", problems)
        self.assertIn("R3_TYPO is not a setting any component declares", problems)

    def test_a_profile_environment_is_checked_against_the_components(self):
        import tempfile
        from pathlib import Path
        text = "\n".join([
            "version = 2",
            'engine = "radiance"',
            'pinned = "p"',
            "[runtime-profile.x]",
            "server-args = []",
            "tune-context = 1",
            "production-context = 1",
            "min-free-vram-bytes-per-device = 0",
            'environment = { R3_ATTN_NATIVE = "1" }',
            "",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(config.ConfigError, "restates libr3's default"):
                config.load(path)
            path.write_text(text.replace('R3_ATTN_NATIVE = "1"', 'R3_AR_WIRE4 = "1"'), encoding="utf-8")
            self.assertEqual(dict(config.load(path).runtime_profiles["x"].environment), {"R3_AR_WIRE4": "1"})

    def test_llamacpp_declares_no_components(self):
        self.assertEqual(components.load_all(paths.REPO_ROOT, engines.LLAMACPP), {})


if __name__ == "__main__":
    unittest.main()


class PromotionTests(unittest.TestCase):
    """MEN11 step f: a promoted setting carries run.json evidence that varies it alone with activation observed."""

    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.json = json
        self.run = {
            "arm_variables": {"off": {"environment": {}, "settings": {"R3_X": "0"}},
                              "on": {"environment": {}, "settings": {"R3_X": "1"}}},
            "setting_activation": {"on": {"R3_X": {"status": "executed"}}},
            "runs": [{"mode": "off", "returncode": 0}, {"mode": "on", "returncode": 0}],
        }

    def component(self, status="validated", default="1", evidence=("e/run.json",)):
        setting = components.Setting("R3_X", default, status, "s", "marker", tuple(evidence))
        from pathlib import Path
        return components.Component("c", "radiance", Path("."), ("t",), "p", ("o",), {"R3_X": setting})

    def write(self, run=None):
        path = self.root / "e" / "run.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(self.json.dumps(self.run if run is None else run), encoding="utf-8")

    def test_good_evidence_passes(self):
        self.write()
        self.assertEqual(components.promotion_problems(self.component(), self.root), ([], []))

    def test_validated_without_evidence_is_a_problem_and_experimental_is_not_judged(self):
        problems, _ = components.promotion_problems(self.component(evidence=()), self.root)
        self.assertEqual(len(problems), 1)
        self.assertEqual(components.promotion_problems(self.component("experimental", evidence=()), self.root), ([], []))

    def test_production_default_on_without_evidence_warns_and_off_default_is_exempt(self):
        problems, warnings = components.promotion_problems(self.component("production", evidence=()), self.root)
        self.assertEqual((problems, len(warnings)), ([], 1))
        self.assertEqual(components.promotion_problems(self.component("production", default="0", evidence=()), self.root), ([], []))

    def test_evidence_must_vary_the_setting_alone_and_show_activation(self):
        for mutate in (
            lambda r: r["setting_activation"]["on"]["R3_X"].update(status="not_executed"),
            lambda r: r["arm_variables"]["off"]["settings"].update(R3_X="1"),
            lambda r: r["arm_variables"]["off"]["settings"].update(R3_Y="2"),
            lambda r: r["arm_variables"]["off"].update(environment={"A": "1"}),
            lambda r: r["runs"][1].update(returncode=1),
            lambda r: r.update(quality={"passed": False}),
        ):
            with self.subTest(mutate=mutate):
                run = self.json.loads(self.json.dumps(self.run))
                mutate(run)
                self.write(run)
                problems, _ = components.promotion_problems(self.component(), self.root)
                self.assertEqual(len(problems), 1, problems)

    def test_missing_evidence_file_is_a_problem(self):
        problems, _ = components.promotion_problems(self.component(), self.root)
        self.assertIn("not a readable run.json", problems[0])

    def test_real_declarations_have_no_promotion_problems(self):
        for component in components.load_all(paths.REPO_ROOT, engines.RADIANCE).values():
            problems, _ = components.promotion_problems(component, paths.REPO_ROOT)
            self.assertEqual(problems, [])
