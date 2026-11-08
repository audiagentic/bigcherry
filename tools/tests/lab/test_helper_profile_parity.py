"""The helper section bc-serve writes into a v1 manifest (serving/helper_profile.toml, RSC08 S2a) against what the helpers
are run with today, both ways, failing closed.

Sources diffed: the component settings (component.toml `default`), the launcher's R3_* exports
(tools/lab/radiance/launch-flashnext-best.sh) and `bigcherry layout env` (radiance_layout.resolve against a lab-shaped
inventory, no card needed). Rules:
  * every row names a setting that exists, and its value is what its basis says (component default, unset-means-off,
    the layout's export, a launcher literal);
  * a setting the launcher or the layout exports and that is a helper_common concern is a row, with the same value; one
    that is another section's concern is named in OTHER_SECTIONS with its section, so a new export cannot slip past;
  * a launcher literal is a hand-picked tunable and none may hide in a row whose basis says otherwise.
"""
from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

from bigcherry.core import paths
from bigcherry.core.environment import Device
from bigcherry.engine_adapters import radiance_layout as rl

REPO = paths.REPO_ROOT
SERVING = REPO / "engines" / "radiance" / "components" / "serving"
LAUNCHER = REPO / "tools" / "lab" / "radiance" / "launch-flashnext-best.sh"
LAB_HOST = SimpleNamespace(devices=(Device(0, "gfx1100", "x", 24560), Device(1, "gfx1100", "x", 24560),
                                    Device(2, "gfx1201", "r", 32624), Device(3, "gfx1030", "s", 16368)))

# R3_* the layout exports that are not helper_common leaves, and the section that carries them. (The launcher exports none:
# RSC11/RSC12, the manifest is the one configuration; test_the_launcher_exports_no_setting guards it.)
OTHER_SECTIONS = {
    "R3_MOE_WORKER_COUNT": "processes.helpers (its length)",
    "R3_MOE_WORKER_ENGINE_RANKS": "processes.engine.scheduler.tp",
    "R3_MOE_WORKER_MIBS": "processes.helpers[].vram_budget_mib",
    "R3_MOE_WORKER_PLACEMENT": "handoff.placement.strategy",
    "R3_MOE_WORKER_INDEX": "processes.helpers[].role",
    "R3_MOE_WORKER_PLUGIN_FILE": "processes.helpers[].plugin_file",
    "R3_MOE_WORKER_FORMATS": "processes.helpers[].formats",
}
# a per-helper export that the manifest carries as the helper's own field, not a helper_common leaf: setting -> field
# (R3_MOE_WORKER_ROWS=1 is the architecture requiring the rows kernel: processes.helpers[].rows_kernel; lanes.rows stays false)
PER_HELPER_FIELDS = {"R3_MOE_WORKER_ROWS": "processes.helpers[].rows_kernel"}


def _component_settings() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for toml in (REPO / "engines" / "radiance" / "components").glob("*/component.toml"):
        for name, body in tomllib.loads(toml.read_text(encoding="utf-8")).get("setting", {}).items():
            out[name] = body
    return out


def _rows() -> list[dict]:
    return tomllib.loads((SERVING / "helper_profile.toml").read_text(encoding="utf-8"))["row"]


def _launcher_exports() -> dict[str, str]:
    """The R3_MOE_* assignments in the launcher's own text (comments dropped): name -> its right-hand side as written."""
    out: dict[str, str] = {}
    for line in LAUNCHER.read_text(encoding="utf-8").splitlines():
        code = line.split("#", 1)[0]
        for name, value in re.findall(r"(?<![\^\w])(R3_MOE_[A-Z0-9_]+)=(\"[^\"]*\"|[^\s)\"]+)", code):
            out[name] = value.strip('"')
    return out


def _layout_exports() -> tuple[dict[str, str], list[dict[str, str]]]:
    r = rl.resolve("flashnext-4gpu", LAB_HOST)
    return dict(r.common_env), [{k: v for k, v in r.helper_env(i).items() if k.startswith("R3_")} for i in range(len(r.helpers))]


def _as_value(text: str):
    if text in ("1", "true"):
        return True
    if text in ("0", "false"):
        return False
    return int(text) if re.fullmatch(r"-?\d+", text) else text


class HelperProfileParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _component_settings()
        self.rows = _rows()
        self.launcher = _launcher_exports()
        self.layout_common, self.layout_helpers = _layout_exports()

    def test_the_table_covers_every_helper_common_leaf_once(self) -> None:
        import json
        schema = json.loads((REPO / "docs/design/serving-manifest-v1/serving-manifest.v1.schema.json").read_text(encoding="utf-8"))
        props = schema["$defs"]["helperCommon"]["properties"]
        leaves = {f"{g}/{k}" for g, body in props.items() for k in body.get("properties", {}) if k != "balance_speeds_milli"}
        got = [r["leaf"] for r in self.rows]
        self.assertEqual(sorted(got), sorted(set(got)), "a leaf appears once")
        self.assertEqual(set(got), leaves, "every helper_common leaf (balance_speeds_milli is absent while balance is off) is a row")

    def test_leaves_and_settings_are_one_to_one(self) -> None:
        settings = [r["setting"] for r in self.rows]
        self.assertEqual(sorted(settings), sorted(set(settings)), "a setting feeds one leaf: no two leaves share an R3_* setting")
        for r in self.rows:
            if "from" in r:
                self.assertRegex(r["from"], r"^[a-z_]+\.[a-z_]+$", "a resolver key is group.name")
        froms = [r["from"] for r in self.rows if "from" in r]
        self.assertEqual(sorted(froms), sorted(set(froms)), "a resolver key feeds one leaf")

    def test_the_resolver_reads_layouts_toml_and_keeps_no_copy_of_it(self) -> None:
        # RSC11 A2: serving.cpp embeds engines/radiance/layouts.toml and archs.toml (CMakeLists.txt) and derives helper.combine,
        # the cards, budgets and placement from them; a layout named in its source would be a second copy.
        text = (SERVING / "serving.cpp").read_text(encoding="utf-8")
        layouts = tomllib.loads((REPO / "engines" / "radiance" / "layouts.toml").read_text(encoding="utf-8"))["layout"]
        for name in layouts:
            self.assertNotIn(f'name == "{name}"', text, f"serving.cpp branches on the layout {name}")
        self.assertIn("layouts_seed.inc", text)
        self.assertIn("archs_seed.inc", text)

    def test_every_row_names_a_setting_and_its_basis_holds(self) -> None:
        for r in self.rows:
            with self.subTest(leaf=r["leaf"]):
                self.assertIn(r["setting"], self.settings, "the setting exists in a component.toml")
                default = self.settings[r["setting"]].get("default")
                value = r["value"]
                if r["basis"] == "component_default":
                    self.assertNotEqual(default, "", "a component default is not empty")
                    self.assertEqual(_as_value(default), value)
                elif r["basis"] == "unset_means":
                    self.assertEqual(default, "", "unset-means rows are the ones whose component default is empty")
                    self.assertIn(value, (False, 0, []), "unset is the helper's off, zero or empty")
                elif r["basis"] == "layout":
                    self.assertEqual(_as_value(self.layout_common[r["setting"]]), value, "the layout exports it")
                elif r["basis"] == "resolved":
                    self.assertIn("from", r, "a resolved row names the resolver key the writer reads")
                    self.assertEqual(value, _as_value(default) if default != "" else False,
                                     "its value is only the setting's default (unset is off)")
                elif r["basis"] == "launcher_literal":
                    self.assertEqual(_as_value(self.launcher[r["setting"]]), value, "the launcher sets it")
                else:
                    self.fail("unknown basis " + r["basis"])

    def test_an_export_that_is_a_row_has_the_row_value_and_no_literal_hides_in_a_default(self) -> None:
        by_setting = {r["setting"]: r for r in self.rows}
        exports = {**{k: v for k, v in self.launcher.items() if not v.startswith("$")}, **self.layout_common}
        for name, text in exports.items():
            if name in by_setting:
                with self.subTest(setting=name):
                    row = by_setting[name]
                    if row["basis"] == "resolved":
                        continue   # the layout decides; the writer takes the resolver's value (checked per layout above)
                    self.assertEqual(_as_value(text), row["value"], "the launcher/layout runs the helper with the row's value")
                    self.assertIn(row["basis"], ("layout", "launcher_literal"),
                                  "a value the launcher or layout sets is not recorded as a default")

    def test_every_export_is_a_row_or_in_another_section_or_a_role_patch(self) -> None:
        by_setting = {r["setting"] for r in self.rows}
        seen = set(self.launcher) | set(self.layout_common)
        for env in self.layout_helpers:
            seen |= {k for k in env if k.startswith("R3_MOE_")}
        for name in sorted(seen):
            with self.subTest(setting=name):
                self.assertTrue(name in by_setting or name in OTHER_SECTIONS or name in PER_HELPER_FIELDS,
                                f"{name} is exported but is neither a helper_common row, another section's, nor a helper's own field")

    def test_per_helper_exports_are_the_helpers_own_fields(self) -> None:
        resolved = rl.resolve("flashnext-4gpu", LAB_HOST)
        for i, env in enumerate(self.layout_helpers):
            requires = resolved.helpers[i].arch.rows_kernel
            exported = _as_value(env["R3_MOE_WORKER_ROWS"]) if "R3_MOE_WORKER_ROWS" in env else False
            self.assertEqual(exported, requires, f"helper{i}: the rows kernel is exported exactly when its architecture requires it")

    def test_other_sections_entries_are_not_stale(self) -> None:
        seen = set(self.launcher) | set(self.layout_common)
        for env in self.layout_helpers:
            seen |= set(env)
        self.assertEqual(self.launcher, {}, "the launcher exports no R3_MOE_* setting: the manifest carries them")
        for name in OTHER_SECTIONS:
            self.assertIn(name, seen, f"{name} is no longer exported: drop it from OTHER_SECTIONS")


if __name__ == "__main__":
    unittest.main()
