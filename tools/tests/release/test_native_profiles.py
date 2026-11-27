"""A serving config is its native YAML (the TOML catalogue is gone); the release ships exactly those files."""
from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

import yaml

from bigcherry.release import runtime_assets as assets

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "engines/radiance/configs"
SERVING = ROOT / "engines/radiance/components/serving"
NAMES = sorted(p.stem for p in CONFIGS.glob("*.yaml"))


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def paths(document: dict) -> dict:
    """Every leaf of a config as {yaml path: value}; description, variant-of and differs are provenance, not settings."""
    leaves: dict = {}

    def walk(node, prefix):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else key)
        else:
            leaves[prefix] = node

    walk({k: v for k, v in document.items() if k != "x-bigcherry"}, "")
    walk({k: v for k, v in document["x-bigcherry"].items()
          if k not in ("description", "variant-of", "differs")}, "x-bigcherry")
    return leaves


class NativeProfileTests(unittest.TestCase):
    def test_the_release_ships_exactly_the_config_yaml_files(self):
        self.assertEqual(set(assets.NATIVE_RELEASE_PROFILES), {p.name for p in CONFIGS.glob("*.yaml")})
        self.assertEqual(list(assets.NATIVE_RELEASE_PROFILES), sorted(assets.NATIVE_RELEASE_PROFILES))
        for name in assets.NATIVE_RELEASE_PROFILES:
            self.assertRegex(name, r"^[a-z0-9-]+\.yaml$")

    def test_each_yaml_names_one_model_and_a_card_per_rank(self):
        for name in NAMES:
            with self.subTest(name=name):
                text = (CONFIGS / f"{name}.yaml").read_text(encoding="utf-8")
                tp = int(re.search(r"^    tp: (\d+)$", text, re.M).group(1))
                cards = re.search(r"^cards: \[(.*)\]$", text, re.M).group(1).split(", ")
                self.assertEqual(cards, [str(i) for i in range(tp)])
                self.assertEqual(len(re.findall(r"^    model: ", text, re.M)), 1)

    def test_the_install_side_allowlist_equals_the_packer_list(self):
        cpp = (SERVING / "release_archive.cpp").read_text(encoding="utf-8")
        block = re.search(r"kNativeProfiles = \{(.*?)\};", cpp, re.S).group(1)
        self.assertEqual(re.findall(r'"([^"]+)"', block), [n.removesuffix(".yaml") for n in assets.NATIVE_RELEASE_PROFILES])

    def test_each_config_has_a_closed_x_bigcherry_section(self):
        for name in NAMES:
            with self.subTest(name=name):
                section = load(name)["x-bigcherry"]
                self.assertEqual(section["schema"], 1)
                self.assertEqual(section["power"], {"profile": "efficient"})
                self.assertTrue(section["description"] and section["ranks"])
                self.assertLessEqual(set(section), {"schema", "description", "variant-of", "differs", "ranks", "helpers",
                                                    "service", "memory", "handoff", "helper", "kernels", "power"})
                tp = next(iter(load(name)["models"].values()))["tp"]
                self.assertEqual(sum(r["count"] for r in section["ranks"]), tp)
                self.assertEqual("variant-of" in section, "differs" in section)

    def test_a_variant_differs_from_its_base_exactly_where_it_says(self):
        for name in NAMES:
            section = load(name)["x-bigcherry"]
            if "variant-of" not in section:
                continue
            with self.subTest(name=name):
                mine, base = paths(load(name)), paths(load(section["variant-of"]))
                actual = {k for k in mine.keys() | base.keys() if mine.get(k) != base.get(k) and k != "server.port"}
                self.assertEqual(actual, set(section["differs"]))

    def test_the_resolver_test_presets_are_generated_from_the_yaml_files(self):
        """A preset (test_config_presets.inc) is named after, and generated from, its shipped YAML: it cannot drift."""
        spec = importlib.util.spec_from_file_location("gen_fixture_presets", SERVING / "gen_fixture_presets.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual((SERVING / "test_config_presets.inc").read_text(encoding="utf-8"), module.render(),
                         "run engines/radiance/components/serving/gen_fixture_presets.py --write")


if __name__ == "__main__":
    unittest.main()
