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

# R3_* the launcher or the layout exports that are not helper_common leaves, and the section that carries them.
OTHER_SECTIONS = {
    "R3_MOE_WORKER": "engine process switch (the engine's own, not a helper setting)",
    "R3_MOE_WORKER_SHM": "handoff.transport.shm",
    "R3_MOE_WORKER_MAX_ROWS": "handoff.transport.max_rows",
    "R3_MOE_WORKER_SMALL_ROWS": "handoff.transport.small_rows",
    "R3_MOE_WORKER_GEMV_LARGE_ROWS": "handoff.transport.gemv_large_rows",
    "R3_MOE_WORKER_MOE_DOWN_TILE": "handoff.kernels.moe_down_tile",
    "R3_MOE_WORKER_SKIP_OWNED": "handoff.transport.skip_owned",
    "R3_MOE_WORKER_HEAT": "handoff.placement.heat_file",
    "R3_MOE_WORKER_HISTOGRAM": "handoff.telemetry.histogram_file",
    "R3_MOE_WORKER_HISTOGRAM_PERIOD_S": "handoff.telemetry.period_s",
    "R3_MOE_WORKER_COUNT": "processes.helpers (its length)",
    "R3_MOE_WORKER_ENGINE_RANKS": "processes.engine.scheduler.tp",
    "R3_MOE_WORKER_MIBS": "processes.helpers[].vram_budget_mib",
    "R3_MOE_WORKER_PLACEMENT": "handoff.placement.strategy",
    "R3_MOE_WORKER_INDEX": "processes.helpers[].role",
    "R3_MOE_WORKER_PLUGIN_FILE": "processes.helpers[].plugin_file",
    "R3_MOE_WORKER_FORMATS": "processes.helpers[].formats",
}
# a per-helper export that is a role patch of a helper_common leaf, by setting: (leaf, value)
ROLE_PATCHES = {"R3_MOE_WORKER_ROWS": ("lanes/rows", True)}


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
        for name, value in re.findall(r"\b(R3_MOE_[A-Z0-9_]+)=(\"[^\"]*\"|[^\s)\"]+)", code):
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
                self.assertTrue(name in by_setting or name in OTHER_SECTIONS or name in ROLE_PATCHES,
                                f"{name} is exported but is neither a helper_common row, another section's, nor a role patch")

    def test_per_helper_exports_are_role_patches_with_the_expected_leaf(self) -> None:
        for i, env in enumerate(self.layout_helpers):
            for name, text in env.items():
                if name in ROLE_PATCHES:
                    leaf, value = ROLE_PATCHES[name]
                    self.assertEqual(_as_value(text), value, f"helper{i}: {name}")
                    self.assertIn(leaf, {r["leaf"] for r in self.rows}, "a role patch targets a helper_common leaf")

    def test_other_sections_entries_are_not_stale(self) -> None:
        seen = set(self.launcher) | set(self.layout_common)
        for env in self.layout_helpers:
            seen |= set(env)
        # R3_MOE_WORKER_HISTOGRAM(_PERIOD_S) are exported only with --record-heat, which the launcher text still mentions
        for name in OTHER_SECTIONS:
            self.assertIn(name, seen, f"{name} is no longer exported: drop it from OTHER_SECTIONS")


if __name__ == "__main__":
    unittest.main()
