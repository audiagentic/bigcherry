"""PVPS02 step 3: resolve_benchmark_wiring() -- a patch is generically
benchmarkable iff exactly one of its required capability="performance"
checks declares a recognized benchmark-executor in validation.toml.
Fails closed on zero/ambiguous/unknown wiring, never guesses.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import registry as patch_registry  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402


def _write_patch(
    root: Path, patch_id: str, *, performance_toml: str,
) -> patch_registry.PatchDescriptor:
    order = int(patch_id.split("_", 1)[0])
    package_dir = root / patch_id
    package_dir.mkdir(parents=True)
    (package_dir / "patch.py").write_text("PATCHES = []\n", encoding="utf-8")
    (package_dir / "patch.toml").write_text(
        "schema = 1\n"
        f'id = "{patch_id}"\n'
        f"order = {order}\n"
        'state = "untested"\n'
        'kind = "enhancement"\n'
        'origin = "local"\n'
        'backend = "hip"\n'
        "plan-ids = []\nrequires = []\nconflicts = []\n"
        "requires-options = []\nforbids-options = []\nsubsystems = []\n"
        "hardware = []\nvalidation-architectures = []\nbackends = []\n",
        encoding="utf-8",
    )
    (package_dir / "validation.toml").write_text(
        "schema = 1\n\n"
        "[[check]]\n"
        'id = "apply"\ncapability = "apply"\nvalidator = "apply"\nrequired = true\n\n'
        "[[check]]\n"
        'id = "build"\ncapability = "build"\nvalidator = "build"\nrequired = true\n\n'
        + performance_toml,
        encoding="utf-8",
    )
    registry = patch_registry.load_registry(root)
    return registry.by_id[patch_id]


class ResolveBenchmarkWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_recognized_executor_with_extra_args_resolves(self) -> None:
        descriptor = _write_patch(
            self.root, "9001_example",
            performance_toml=(
                "[[check]]\n"
                'id = "performance"\ncapability = "performance"\nvalidator = "benchmark"\n'
                "required = true\n"
                'benchmark-executor = "paired-llama-bench-v1"\n'
                'benchmark-extra-args = ["-fa", "on"]\n'
            ),
        )
        wiring = vc.resolve_benchmark_wiring(descriptor, root=self.root)
        self.assertEqual(wiring.executor, "paired-llama-bench-v1")
        self.assertEqual(wiring.patch_args, ("-fa", "on"))

    def test_recognized_executor_with_no_extra_args_defaults_empty(self) -> None:
        descriptor = _write_patch(
            self.root, "9002_example",
            performance_toml=(
                "[[check]]\n"
                'id = "performance"\ncapability = "performance"\nvalidator = "autotune-campaign"\n'
                "required = true\n"
                'benchmark-executor = "paired-llama-bench-v1"\n'
            ),
        )
        wiring = vc.resolve_benchmark_wiring(descriptor, root=self.root)
        self.assertEqual(wiring.patch_args, ())

    def test_no_performance_check_at_all_fails_closed(self) -> None:
        descriptor = _write_patch(self.root, "9003_example", performance_toml="")
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_wiring(descriptor, root=self.root)

    def test_performance_check_without_benchmark_executor_fails_closed(self) -> None:
        descriptor = _write_patch(
            self.root, "9004_example",
            performance_toml=(
                "[[check]]\n"
                'id = "performance"\ncapability = "performance"\nvalidator = "benchmark"\n'
                "required = true\n"
            ),
        )
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_wiring(descriptor, root=self.root)

    def test_non_required_performance_check_does_not_count(self) -> None:
        descriptor = _write_patch(
            self.root, "9005_example",
            performance_toml=(
                "[[check]]\n"
                'id = "performance"\ncapability = "performance"\nvalidator = "benchmark"\n'
                "required = false\n"
                'benchmark-executor = "paired-llama-bench-v1"\n'
            ),
        )
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_wiring(descriptor, root=self.root)

    def test_unknown_executor_fails_closed(self) -> None:
        descriptor = _write_patch(
            self.root, "9006_example",
            performance_toml=(
                "[[check]]\n"
                'id = "performance"\ncapability = "performance"\nvalidator = "benchmark"\n'
                "required = true\n"
                'benchmark-executor = "some-future-executor"\n'
            ),
        )
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_wiring(descriptor, root=self.root)

    def test_two_wired_performance_checks_is_ambiguous(self) -> None:
        descriptor = _write_patch(
            self.root, "9007_example",
            performance_toml=(
                "[[check]]\n"
                'id = "performance"\ncapability = "performance"\nvalidator = "benchmark"\n'
                "required = true\n"
                'benchmark-executor = "paired-llama-bench-v1"\n\n'
                "[[check]]\n"
                'id = "performance-2"\ncapability = "performance"\nvalidator = "benchmark"\n'
                "required = true\n"
                'benchmark-executor = "paired-llama-bench-v1"\n'
            ),
        )
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_wiring(descriptor, root=self.root)

    def test_non_list_extra_args_fails_closed(self) -> None:
        descriptor = _write_patch(
            self.root, "9008_example",
            performance_toml=(
                "[[check]]\n"
                'id = "performance"\ncapability = "performance"\nvalidator = "benchmark"\n'
                "required = true\n"
                'benchmark-executor = "paired-llama-bench-v1"\n'
                'benchmark-extra-args = "-fa on"\n'
            ),
        )
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_wiring(descriptor, root=self.root)

    def test_real_rd04_patch_resolves(self) -> None:
        from bigcherry.core import paths

        registry = patch_registry.load_registry(paths.PATCHES)
        descriptor = registry.by_id["1202_rd04_bf16_flash_attn_tile"]
        wiring = vc.resolve_benchmark_wiring(descriptor)
        self.assertEqual(wiring.executor, "paired-llama-bench-v1")
        self.assertEqual(wiring.patch_args, ("-fa", "on", "-ctk", "bf16", "-ctv", "bf16"))

    def test_real_rd08_patch_resolves(self) -> None:
        from bigcherry.core import paths

        registry = patch_registry.load_registry(paths.PATCHES)
        descriptor = registry.by_id["1204_rd08_q6k_mmvq_vdr2"]
        wiring = vc.resolve_benchmark_wiring(descriptor)
        self.assertEqual(wiring.executor, "paired-llama-bench-v1")
        self.assertEqual(wiring.patch_args, ())


if __name__ == "__main__":
    unittest.main()
