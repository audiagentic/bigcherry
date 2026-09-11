"""PVPS02 step 5: resolve_benchmark_model()/resolve_device_pool()/
parse_device_map() -- real model resolution (not just a models.toml
row) and ordered device-pool topology semantics.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


def _write_registry(root: Path, *, topology: str | None = "single", size: int | None = None) -> Path:
    registry_path = root / "models.toml"
    model_path = root / "model.gguf"
    real_size = 1234
    model_path.write_bytes(b"x" * real_size)
    topology_line = f'benchmark-topology = "{topology}"\n' if topology is not None else ""
    size_line = f"size-bytes = {size if size is not None else real_size}\n"
    registry_path.write_text(
        "version = 1\n\n"
        "[[models]]\n"
        'id = "test-model"\n'
        'family = "test"\n'
        'path = "model.gguf"\n'
        f"{size_line}"
        f"{topology_line}",
        encoding="utf-8",
    )
    return registry_path


class ResolveBenchmarkModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_single_topology_resolves(self) -> None:
        registry = _write_registry(self.root, topology="single")
        resolved = vc.resolve_benchmark_model(
            "test-model", model_root=self.root, registry_path=registry,
        )
        self.assertEqual(resolved.topology, "single")
        self.assertEqual(resolved.device_count, 1)
        self.assertEqual(resolved.path, self.root / "model.gguf")

    def test_valid_tensor2_topology_resolves(self) -> None:
        registry = _write_registry(self.root, topology="tensor-2")
        resolved = vc.resolve_benchmark_model(
            "test-model", model_root=self.root, registry_path=registry,
        )
        self.assertEqual(resolved.device_count, 2)

    def test_unknown_model_id_fails_closed(self) -> None:
        registry = _write_registry(self.root)
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_model("nonexistent", model_root=self.root, registry_path=registry)

    def test_missing_file_fails_closed(self) -> None:
        registry = _write_registry(self.root)
        (self.root / "model.gguf").unlink()
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_model("test-model", model_root=self.root, registry_path=registry)

    def test_size_mismatch_fails_closed(self) -> None:
        registry = _write_registry(self.root, size=99999)
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_model("test-model", model_root=self.root, registry_path=registry)

    def test_no_declared_topology_never_defaults_to_single(self) -> None:
        registry = _write_registry(self.root, topology=None)
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_model("test-model", model_root=self.root, registry_path=registry)

    def test_unrecognized_topology_fails_closed(self) -> None:
        registry = _write_registry(self.root, topology="tensor-99")
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_benchmark_model("test-model", model_root=self.root, registry_path=registry)

    def test_real_ministral_entry_in_the_real_registry_is_wired_correctly(self) -> None:
        # Hardware-free: does not require the real gguf file to exist on
        # this machine, only that the real config/models.toml entry
        # parses with the right topology/device-count -- the file-
        # existence/size checks are exercised by the tests above against
        # a synthetic fixture.
        import tomllib

        from bigcherry.core import paths

        raw = tomllib.loads(paths.MODELS.read_text(encoding="utf-8"))
        entries = {e["id"]: e for e in raw["models"] if isinstance(e, dict)}
        self.assertIn("tierM-ministral14b-q4km", entries)
        self.assertEqual(entries["tierM-ministral14b-q4km"]["benchmark-topology"], "single")
        self.assertEqual(entries["tierB-qwen9b-q6k"]["benchmark-topology"], "single")
        self.assertEqual(entries["tierL-qwen27b-q8"]["benchmark-topology"], "tensor-2")


class ParseDeviceMapTests(unittest.TestCase):
    def test_single_entry_parses(self) -> None:
        self.assertEqual(vc.parse_device_map(["gfx1100=0,1"]), {"gfx1100": ("0", "1")})

    def test_multiple_entries_parse(self) -> None:
        self.assertEqual(
            vc.parse_device_map(["gfx1100=0,1", "gfx1030=3"]),
            {"gfx1100": ("0", "1"), "gfx1030": ("3",)},
        )

    def test_order_is_preserved(self) -> None:
        self.assertEqual(vc.parse_device_map(["gfx1100=1,0"]), {"gfx1100": ("1", "0")})

    def test_missing_equals_sign_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.parse_device_map(["gfx1100"])

    def test_blank_architecture_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.parse_device_map(["=0,1"])

    def test_empty_device_list_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.parse_device_map(["gfx1100="])

    def test_blank_device_entry_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.parse_device_map(["gfx1100=0,,1"])

    def test_duplicate_architecture_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.parse_device_map(["gfx1100=0", "gfx1100=1"])


class ResolveDevicePoolTests(unittest.TestCase):
    def test_takes_first_n_ids_in_order(self) -> None:
        pool = vc.resolve_device_pool({"gfx1100": ("1", "0")}, "gfx1100", 1)
        self.assertEqual(pool, ("1",))

    def test_full_pool_for_exact_count(self) -> None:
        pool = vc.resolve_device_pool({"gfx1100": ("0", "1")}, "gfx1100", 2)
        self.assertEqual(pool, ("0", "1"))

    def test_unmapped_architecture_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_device_pool({"gfx1100": ("0",)}, "gfx1201", 1)

    def test_pool_smaller_than_needed_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.resolve_device_pool({"gfx1100": ("0",)}, "gfx1100", 2)


if __name__ == "__main__":
    unittest.main()
