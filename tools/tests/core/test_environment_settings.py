"""RSC11 S1: serving configuration is typed, not a process environment carrier.

Only explicitly non-migrated RAD_*/BC_* settings may still be bridged by the
registry. No R3_* handoff or kernel switch is set by rad0009 or read by the
serving engine. GPU-runtime settings and selftest-only toggles are separate.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "engines/radiance/components/serving/contract/serving-v1.toml"
RAD0009 = ROOT / "engines/radiance/patches/rad0009_serving_manifest/patch.py"

# The handoff is a typed snapshot consumed directly by the engine and plugin.
TYPED_ENGINE_VARS = {
    "R3_MOE_WORKER", "R3_MOE_WORKER_SHM", "R3_MOE_WORKER_MAX_ROWS",
    "R3_MOE_WORKER_SKIP_OWNED", "R3_MOE_WORKER_HISTOGRAM",
    "R3_MOE_WORKER_HISTOGRAM_PERIOD_S", "R3_MOE_WORKER_COMBINE",
    "R3_MOE_REMOTE_MIN_ROWS", "R3_MOE_REMOTE_MAX_ROWS",
    "R3_MOE_NATIVE", "R3_MOE_FP16", "R3_MOE_AHALF",
    "R3_GEMV", "R3_DEC11", "R3_ATTN_NATIVE", "R3_AR_WIRE4",
    "R3_MOE_WORKER_MASK_ONLY", "R3_MOE_WORKER_FUSESTAGE",
}
#: the self-test tool toggles these around a check; it is a test harness, never a serving process
SELFTEST_FILE = "r4d_selftest.cpp.rw"

_READ = re.compile(r'(?:getenv|\bu32|\bflag|\bchoice|\bpeek_u32|\bpeek_flag|\benv_on|\brows_env)\("((?:R3|RAD|BC)_[A-Z0-9_]+)"')
_SKIP_PARTS = ("native-shelved", "/tests/", "golden/")


def _tracked(prefix: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", prefix], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\n") if f]


class EnvironmentSettingsGuard(unittest.TestCase):
    def test_every_environment_read_is_a_manifest_carrier(self) -> None:
        bound = set(re.findall(r'env_setting = "([^"]+)"', REGISTRY.read_text(encoding="utf-8"))) - TYPED_ENGINE_VARS
        self.assertTrue(bound, "the remaining non-migrated manifest carriers must be accounted for")
        self.assertFalse(bound & TYPED_ENGINE_VARS)
        unexplained: dict[str, list[str]] = {}
        for rel in _tracked("engines"):
            if not rel.endswith((".cpp", ".hip", ".h", ".rw", ".py")) or rel.endswith("_test.cpp"):
                continue
            if any(part in rel for part in _SKIP_PARTS) or rel.endswith(SELFTEST_FILE):
                continue
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for name in _READ.findall(text):
                if name in bound:
                    continue
                unexplained.setdefault(name, []).append(rel)
        self.assertEqual(
            {}, unexplained,
            "an environment variable is read that no manifest field sets: give it an env_setting field in the registry "
            "(or stop reading it); see docs/design/serving-manifest-v1/spec.md amendment 14")

    def test_typed_engine_settings_have_no_legacy_environment_bridge(self) -> None:
        patch = RAD0009.read_text(encoding="utf-8")
        self.assertIn("g_engine_settings = settings;", patch)
        self.assertIn("bc_serving_remote_settings", patch)
        self.assertIn("bc_serving_engine_settings", patch)
        for name in ("R3_MOE_WORKER", "R3_MOE_WORKER_SHM", "R3_MOE_WORKER_MAX_ROWS",
                     "R3_MOE_WORKER_SKIP_OWNED", "R3_MOE_WORKER_HISTOGRAM",
                     "R3_MOE_WORKER_HISTOGRAM_PERIOD_S", "R3_MOE_WORKER_COMBINE",
                     "R3_MOE_REMOTE_MIN_ROWS", "R3_MOE_REMOTE_MAX_ROWS"):
            self.assertNotIn(f'setenv("{name}"', patch)
        self.assertNotIn("HANDOFF_BRIDGED", patch)

    def test_every_bound_variable_is_declared_once(self) -> None:
        names = re.findall(r'env_setting = "([^"]+)"', REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual(sorted(names), sorted(set(names)), "two registry fields stand in for one variable")


if __name__ == "__main__":
    unittest.main()
