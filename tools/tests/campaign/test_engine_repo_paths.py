"""Radiance's engine configure passes the shared serving manifest v1 library's .cmake to the engine's CMake (managed
patch rad0011 includes it, failing closed without it): declared in engine.toml [build] repo-paths, resolved to an
absolute file by the build tool, and present in the configure args the radiance lanes use."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bigcherry.campaign import build as campaign_build
from bigcherry.core import config as campaign_config
from bigcherry.core import engines, paths


class EngineRepoPathsTests(unittest.TestCase):
    def test_radiance_declares_the_shared_library_cmake(self) -> None:
        layout = engines.load(paths.REPO_ROOT, "radiance")
        self.assertEqual(dict(layout.repo_paths),
                         {"BC_SERVING_V1_CMAKE": "engines/radiance/components/serving/bc_serving_v1.cmake"})
        options = layout.repo_path_options(paths.REPO_ROOT)
        self.assertTrue(Path(options["BC_SERVING_V1_CMAKE"]).is_absolute())
        self.assertTrue(Path(options["BC_SERVING_V1_CMAKE"]).is_file())

    def test_the_lane_configure_args_include_it(self) -> None:
        layout = engines.load(paths.REPO_ROOT, "radiance")
        build = campaign_config.Build(name="stock", options=(), variant_set=None, needs=frozenset())
        platform = campaign_config.Platform(name="p", targets=("gfx1100",), options=())
        args = campaign_build.cmake_configure_args(
            build, platform, Path("/src"), Path("/build"), targets_option=layout.targets_option,
            rocm_paths=layout.rocm_paths, engine_options=layout.repo_path_options(paths.REPO_ROOT))
        want = (paths.REPO_ROOT / "engines/radiance/components/serving/bc_serving_v1.cmake").resolve().as_posix()
        self.assertIn(f"-DBC_SERVING_V1_CMAKE={want}", args)

    def test_a_declared_file_that_is_missing_is_an_error(self) -> None:
        layout = engines.load(paths.REPO_ROOT, "radiance")
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(engines.EngineError):
                layout.repo_path_options(Path(empty))

    def test_llamacpp_passes_none(self) -> None:
        self.assertEqual(engines.load(paths.REPO_ROOT, "llamacpp").repo_path_options(paths.REPO_ROOT), {})


if __name__ == "__main__":
    unittest.main()
