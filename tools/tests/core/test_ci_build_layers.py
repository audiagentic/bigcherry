"""Offline, production-function tests for Radiance prebuilt CI layers."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[3] / ".github/scripts/ci_build_layers.py"
SPEC = importlib.util.spec_from_file_location("ci_build_layers", SCRIPT)
assert SPEC and SPEC.loader
layers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layers)


class CILayersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.builds = self.root / "builds"
        self.cache = self.root / "cache"
        self.build = self.builds / ("a" * 32) / ("b" * 32)
        (self.build / "bin").mkdir(parents=True)
        (self.build / "bin/radiance").write_bytes(b"production engine binary")
        (self.build / "CMakeCache.txt").write_text("configured")
        (self.build / "bigcherry-build-metadata-radiance.json").write_text('{"valid":true}')
        for name in layers.COMPONENTS:
            component = self.build / "components" / name
            component.mkdir(parents=True)
            (component / "plugin.so").write_bytes(name.encode())
        self.keys = {"engine": "c" * 64, **{name: "d" * 64 for name in layers.COMPONENTS}}

    def export(self):
        with patch.object(layers, "input_keys", return_value=self.keys):
            layers.export_layers(self.root, self.builds, self.cache)

    def restore(self, destination):
        with patch.object(layers, "input_keys", return_value=self.keys):
            layers.restore_layers(self.root, destination, self.cache)

    def test_full_layer_hit_preserves_validated_build_metadata(self):
        self.export()
        out = self.root / "restored"
        self.restore(out)
        self.assertEqual((out / ("a" * 32) / ("b" * 32) / "bin/radiance").read_bytes(),
                         b"production engine binary")
        self.assertTrue((out / ("a" * 32) / ("b" * 32) /
                         "bigcherry-build-metadata-radiance.json").is_file())
        for name in layers.COMPONENTS:
            self.assertEqual((out / ("a" * 32) / ("b" * 32) / "components" /
                              name / "plugin.so").read_bytes(), name.encode())

    def test_missing_component_forces_real_build_not_stale_binary_reuse(self):
        self.export()
        (self.cache / "libr3/manifest.json").unlink()
        out = self.root / "partial"
        self.restore(out)
        build = out / ("a" * 32) / ("b" * 32)
        self.assertTrue((build / "bin/radiance").is_file())
        self.assertFalse((build / "components/libr3").exists())
        self.assertTrue((build / "components/libr2/plugin.so").is_file())
        self.assertFalse((build / "bigcherry-build-metadata-radiance.json").exists())

    def test_no_base_cache_is_fallback_without_modifying_builds(self):
        out = self.root / "uncached"
        self.restore(out)
        self.assertFalse(out.exists())

    def test_corrupt_archive_refuses_restore(self):
        self.export()
        with (self.cache / "engine/layer.tar.gz").open("ab") as stream:
            stream.write(b"not the recorded digest")
        with self.assertRaisesRegex(layers.LayerError, "digest mismatch"):
            self.restore(self.root / "corrupt")

    def test_changed_component_key_refuses_misattributed_layer(self):
        self.export()
        self.keys["libr3"] = "e" * 64
        with self.assertRaisesRegex(layers.LayerError, "fingerprint"):
            self.restore(self.root / "mismatched")

    def test_unsafe_archive_path_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".tar") as temp:
            with tarfile.open(temp.name, "w") as tar:
                contents = b"malicious"
                info = tarfile.TarInfo("other/path/evil")
                info.size = len(contents)
                tar.addfile(info, io.BytesIO(contents))
            with tarfile.open(temp.name, "r") as tar:
                with self.assertRaisesRegex(layers.LayerError, "unsafe archive"):
                    layers._validate_archive(tar, "a" * 32 + "/" + "b" * 32, "engine")

    def test_component_declaration_drift_fails_closed(self):
        engine = self.root / "engines/radiance"
        engine.mkdir(parents=True)
        (engine / "engine.toml").write_text('[components]\nlibr3="one"\n')
        with self.assertRaisesRegex(layers.LayerError, "component declarations changed"):
            layers.input_keys(self.root)

    def test_fingerprint_includes_file_content_and_path(self):
        a = self.root / "a"
        b = self.root / "b"
        a.write_text("one")
        b.write_text("one")
        old = layers.fingerprint_files(self.root, ["a", "b"])
        a.write_text("two")
        self.assertNotEqual(layers.fingerprint_files(self.root, ["a", "b"]), old)
        a.write_text("one")
        self.assertNotEqual(layers.fingerprint_files(self.root, ["b"]), old)

    def test_export_container_receives_the_image_digest_the_restore_binds(self):
        # The manifest key written inside the export container is recomputed on the host at restore, with the
        # compiler image digest; an export that does not receive it writes keys no restore can match.
        workflow = (SCRIPT.parents[1] / "workflows/radiance-build.yml").read_text()
        step = workflow.split("ci_build_layers.py export", 1)[0].rsplit("docker run", 1)[1]
        self.assertIn("-e CI_LAYER_IMAGE_ID", step.split("\n", 1)[0])


if __name__ == "__main__":
    unittest.main()
