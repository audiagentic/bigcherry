"""A build restored from a CI release bundle has no CMakeCache.txt and no source slice: the campaign tool identifies it by
what release-install verified (the .lab-source-tree marker) instead of failing on the missing CMake cache."""
import tempfile
import unittest
from pathlib import Path

from bigcherry.campaign import benchmark


class ReleaseBundleIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.build = Path(self.tmp.name)

    def test_a_release_installed_build_is_identified_by_its_marker(self):
        (self.build / ".lab-source-tree").write_text("e" * 40 + " " + "c" * 40 + " /opt/rocm-7.2.4 7.2.4 ")
        self.assertEqual(
            benchmark.release_bundle_identity(self.build),
            {"engine_tree": "e" * 40, "config_tree": "c" * 40, "rocm_sdk": "/opt/rocm-7.2.4", "rocm_version": "7.2.4"})

    def test_a_lane_build_with_a_cmake_cache_is_not_a_release_build(self):
        (self.build / ".lab-source-tree").write_text("e c /opt/rocm 7.2.4")
        (self.build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/src\n")
        self.assertIsNone(benchmark.release_bundle_identity(self.build))

    def test_a_build_with_neither_is_not_a_release_build(self):
        self.assertIsNone(benchmark.release_bundle_identity(self.build))

    def test_a_malformed_marker_is_refused(self):
        (self.build / ".lab-source-tree").write_text("only-two fields")
        with self.assertRaisesRegex(ValueError, "not a release-install marker"):
            benchmark.release_bundle_identity(self.build)


if __name__ == "__main__":
    unittest.main()
