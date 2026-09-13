"""The host environment must be loadable, complete, and fail closed.

These are contract tests over the REAL config/environment.toml, not a
fixture: the point of the file is that this project's actual host facts are
resolvable from config rather than hardcoded in prose, so a test against a
synthetic document would not check the thing that matters.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from bigcherry.core import environment as env


class EnvironmentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = env.load(env.default_path(ROOT))
        cls.host = cls.env.host()

    def test_default_host_resolves(self):
        self.assertIn(self.env.default_host, self.env.hosts)

    def test_every_path_fact_is_present(self):
        # Each of these replaced literal occurrences in the reference docs.
        # A blank one would silently produce commands like `cd /run_bench.py`,
        # which is worse than the hardcoded path it replaced.
        for field in ("hostname", "home", "repo", "cache_root", "share",
                      "model_root", "bench_harness", "rocm", "rocm_shim"):
            with self.subTest(field=field):
                self.assertTrue(getattr(self.host, field),
                                f"{field} is empty; docs reference it as $BC_*")

    def test_ports_are_distinct_and_set(self):
        # The production port belongs to the inference service. A benchmark
        # that binds it takes production down, and a tuner that assumes it
        # will fail to start -- which has already happened.
        self.assertGreater(self.host.production_port, 0)
        self.assertGreater(self.host.bench_port, 0)
        self.assertNotEqual(self.host.production_port, self.host.bench_port)

    def test_devices_are_declared_with_arch_and_vram(self):
        self.assertTrue(self.host.devices)
        for d in self.host.devices:
            with self.subTest(index=d.index):
                self.assertTrue(d.arch)
                # VRAM is load-bearing: the smallest participating card
                # constrains any cross-architecture comparison.
                self.assertGreater(d.vram_mib, 0)

    def test_device_indices_are_unique_and_sorted(self):
        idx = [d.index for d in self.host.devices]
        self.assertEqual(idx, sorted(idx))
        self.assertEqual(len(idx), len(set(idx)))

    def test_devices_have_real_verified_pci_locators(self):
        # PRBE111: every real Brutus device's PCI BDF was independently
        # verified via `rocm-smi --showbus`/`--showproductname` on the real
        # host (2026-09-13) -- required for llama-server-based attestation
        # (parse_llama_server_attestation()) to positively confirm which
        # physical card ran a measurement, not just its architecture.
        expected = {
            0: "0000:03:00.0", 1: "0000:06:00.0",
            2: "0000:09:00.0", 3: "0000:17:00.0",
        }
        for d in self.host.devices:
            with self.subTest(index=d.index):
                self.assertEqual(d.locator, expected[d.index])

    def test_gpu_visibility_env_rejects_an_unknown_ordinal(self):
        # Fail closed: a typo in a device list must not silently produce a
        # visibility pair that exposes the wrong cards.
        good = self.host.devices[0].index
        self.host.gpu_visibility_env(good)
        with self.assertRaises(env.EnvironmentError_):
            self.host.gpu_visibility_env(good, 99)

    def test_gpu_visibility_env_rejects_empty_and_duplicate_indices(self):
        with self.assertRaises(env.EnvironmentError_):
            self.host.gpu_visibility_env()
        good = self.host.devices[0].index
        with self.assertRaises(env.EnvironmentError_):
            self.host.gpu_visibility_env(good, good)

    def test_gpu_visibility_env_single_device_uses_position_zero(self):
        # VA22, the two-selector trap: for a single device NOT at physical
        # index 0 (e.g. gfx1201 at index 2 on this host), HIP_VISIBLE_DEVICES
        # must be "0" (its position after ROCR filtering), never "2" again --
        # that would select nothing.
        idx = self.host.devices[-1].index
        pair = self.host.gpu_visibility_env(idx)
        self.assertEqual(pair["ROCR_VISIBLE_DEVICES"], str(idx))
        self.assertEqual(pair["HIP_VISIBLE_DEVICES"], "0")

    def test_gpu_visibility_env_multi_device_reindexes_by_position(self):
        if len(self.host.devices) < 2:
            self.skipTest("host has fewer than 2 devices")
        a, b = self.host.devices[0].index, self.host.devices[-1].index
        pair = self.host.gpu_visibility_env(a, b)
        self.assertEqual(pair["ROCR_VISIBLE_DEVICES"], f"{a},{b}")
        self.assertEqual(pair["HIP_VISIBLE_DEVICES"], "0,1")

    def test_unknown_host_raises(self):
        with self.assertRaises(env.EnvironmentError_):
            self.env.host("no-such-host")

    def test_shell_and_python_agree(self):
        # tools/env/bigcherry-env.sh exports the same facts for shell callers.
        # If the two drift, docs using $BC_* and tooling using this module
        # would disagree about the same host.
        script = ROOT / "tools" / "env" / "bigcherry-env.sh"
        self.assertTrue(script.is_file(), "shell companion is missing")
        text = script.read_text(encoding="utf-8")
        for var, value in (("BC_HOST", self.host.hostname),
                           ("BC_MODEL_ROOT", self.host.model_root),
                           ("BC_BENCH_HARNESS", self.host.bench_harness)):
            with self.subTest(var=var):
                self.assertIn(var, text)
        self.assertIn("environment.toml", text,
                      "the shell script must read the same config, not restate it")


class GpuVisibilityPairTests(unittest.TestCase):
    """The pure two-selector math, independent of any Host/device inventory
    -- for a caller (e.g. tuning/workflow.py) that has already validated its
    indices elsewhere and has no Host object at hand."""

    def test_single_nonzero_index_reindexes_to_position_zero(self):
        pair = env.gpu_visibility_pair((2,))
        self.assertEqual(pair, {"ROCR_VISIBLE_DEVICES": "2", "HIP_VISIBLE_DEVICES": "0"})

    def test_noncontiguous_pair_reindexes_by_position(self):
        pair = env.gpu_visibility_pair((1, 3))
        self.assertEqual(pair, {"ROCR_VISIBLE_DEVICES": "1,3", "HIP_VISIBLE_DEVICES": "0,1"})

    def test_order_is_preserved_between_both_selectors(self):
        pair = env.gpu_visibility_pair((3, 1))
        self.assertEqual(pair, {"ROCR_VISIBLE_DEVICES": "3,1", "HIP_VISIBLE_DEVICES": "0,1"})

    def test_empty_indices_rejected(self):
        with self.assertRaises(env.EnvironmentError_):
            env.gpu_visibility_pair(())

    def test_duplicate_indices_rejected(self):
        with self.assertRaises(env.EnvironmentError_):
            env.gpu_visibility_pair((0, 0))

    def test_host_method_and_pure_function_agree(self):
        # Host.gpu_visibility_env validates against real device inventory
        # then delegates to gpu_visibility_pair -- same output either way.
        loaded = env.load(env.default_path(ROOT))
        host = loaded.host()
        idx = host.devices[-1].index
        self.assertEqual(host.gpu_visibility_env(idx), env.gpu_visibility_pair((idx,)))


class DeviceLocatorParsingTests(unittest.TestCase):
    """PRBE111: fail-closed validation for the new locator field, using a
    synthetic document -- the real config is covered by
    EnvironmentContractTests above."""

    def _write(self, tmp_path: Path, devices_toml: str) -> Path:
        doc = (
            'default-host = "h"\n\n'
            "[host.h]\n"
            'hostname = "h"\n\n'
            f"{devices_toml}\n"
        )
        cfg = tmp_path / "environment.toml"
        cfg.write_text(doc, encoding="utf-8")
        return cfg

    def test_locator_is_optional(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = self._write(Path(td), (
                "[[host.h.devices]]\n"
                "index = 0\n"
                'arch = "gfx1100"\n'
            ))
            loaded = env.load(cfg)
            self.assertIsNone(loaded.host("h").devices[0].locator)

    def test_locator_is_lowercased(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = self._write(Path(td), (
                "[[host.h.devices]]\n"
                "index = 0\n"
                'arch = "gfx1100"\n'
                'locator = "0000:03:00.0"\n'
            ))
            loaded = env.load(cfg)
            self.assertEqual(loaded.host("h").devices[0].locator, "0000:03:00.0")

    def test_malformed_locator_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = self._write(Path(td), (
                "[[host.h.devices]]\n"
                "index = 0\n"
                'arch = "gfx1100"\n'
                'locator = "not-a-bdf"\n'
            ))
            with self.assertRaises(env.EnvironmentError_):
                env.load(cfg)

    def test_duplicate_locator_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = self._write(Path(td), (
                "[[host.h.devices]]\n"
                "index = 0\n"
                'arch = "gfx1100"\n'
                'locator = "0000:03:00.0"\n\n'
                "[[host.h.devices]]\n"
                "index = 1\n"
                'arch = "gfx1100"\n'
                'locator = "0000:03:00.0"\n'
            ))
            with self.assertRaises(env.EnvironmentError_):
                env.load(cfg)


if __name__ == "__main__":
    unittest.main()
