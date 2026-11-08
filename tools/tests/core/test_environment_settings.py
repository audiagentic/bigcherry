"""RSC12: no environment settings. The manifest is the one place a serving setting is configured.

A radiance component or patch may still call getenv on an R3_*, RAD_* or BC_* name, but only when the variable is a private
carrier: rad0009 sets it from the manifest before the engine initialises. This test fails on any such read whose name is not
(a) bound to a manifest field by the registry's env_setting, (b) one of the hand-off variables rad0009 sets from spec.handoff
or helper_common (the list below is checked against the patch), (c) an in-process handshake between two of our own
components, or (d) read only by the self-test tool, which is not a serving process.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "engines/radiance/components/serving/contract/serving-v1.toml"
RAD0009 = ROOT / "engines/radiance/patches/rad0009_serving_manifest/patch.py"

#: set by rad0009 from spec.handoff / spec.processes.helper_common (hand-written in the patch, not env_setting)
HANDOFF_BRIDGED = {
    "R3_MOE_WORKER", "R3_MOE_WORKER_SHM", "R3_MOE_WORKER_MAX_ROWS", "R3_MOE_WORKER_SKIP_OWNED",
    "R3_MOE_WORKER_HISTOGRAM", "R3_MOE_WORKER_HISTOGRAM_PERIOD_S", "R3_MOE_WORKER_COMBINE",
}
#: libr3moe tells the engine's step gate its row window once its segment is ready (same process, both ours): not configuration
INTERNAL_HANDSHAKE = {"R3_MOE_REMOTE_MIN_ROWS", "R3_MOE_REMOTE_MAX_ROWS"}
#: the self-test tool toggles these around a check; it is a test harness, never a serving process
SELFTEST_FILE = "r4d_selftest.cpp.rw"

_READ = re.compile(r'(?:getenv|\bu32|\bflag|\bchoice|\bpeek_u32|\bpeek_flag|\benv_on|\brows_env)\("((?:R3|RAD|BC)_[A-Z0-9_]+)"')
_SKIP_PARTS = ("native-shelved", "/tests/", "golden/")


def _tracked(prefix: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", prefix], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\n") if f]


class EnvironmentSettingsGuard(unittest.TestCase):
    def test_every_environment_read_is_a_manifest_carrier(self) -> None:
        bound = set(re.findall(r'env_setting = "([^"]+)"', REGISTRY.read_text(encoding="utf-8")))
        self.assertGreater(len(bound), 14)
        unexplained: dict[str, list[str]] = {}
        for rel in _tracked("engines"):
            if not rel.endswith((".cpp", ".hip", ".h", ".rw", ".py")) or rel.endswith("_test.cpp"):
                continue
            if any(part in rel for part in _SKIP_PARTS) or rel.endswith(SELFTEST_FILE):
                continue
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for name in _READ.findall(text):
                if name in bound or name in HANDOFF_BRIDGED or name in INTERNAL_HANDSHAKE:
                    continue
                unexplained.setdefault(name, []).append(rel)
        self.assertEqual(
            {}, unexplained,
            "an environment variable is read that no manifest field sets: give it an env_setting field in the registry "
            "(or stop reading it); see docs/design/serving-manifest-v1/spec.md amendment 14")

    def test_hand_off_carriers_are_really_set_by_rad0009(self) -> None:
        patch = RAD0009.read_text(encoding="utf-8")
        for name in sorted(HANDOFF_BRIDGED):
            self.assertIn(f'"{name}"', patch, f"{name} is listed as set from the manifest but rad0009 does not set it")

    def test_every_bound_variable_is_declared_once(self) -> None:
        names = re.findall(r'env_setting = "([^"]+)"', REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual(sorted(names), sorted(set(names)), "two registry fields stand in for one variable")


if __name__ == "__main__":
    unittest.main()
