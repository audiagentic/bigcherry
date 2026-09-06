"""The host environment must be loadable, complete, and fail closed.

These are contract tests over the REAL config/environment.toml, not a
fixture: the point of the file is that this project's actual host facts are
resolvable from config rather than hardcoded in prose, so a test against a
synthetic document would not check the thing that matters.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from bigcherry.core import environment as env

ROOT = Path(__file__).resolve().parents[3]


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

    def test_visible_devices_rejects_an_unknown_ordinal(self):
        # Fail closed: a typo in a device list must not silently produce a
        # visibility string that exposes the wrong cards.
        good = self.host.devices[0].index
        self.host.visible_devices(good)
        with self.assertRaises(env.EnvironmentError_):
            self.host.visible_devices(good, 99)

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


if __name__ == "__main__":
    unittest.main()
