"""No Python tool starts bc-serve's --write-manifest: that writer is the native owner's private handoff, and the launch
resource bridge that called it (bigcherry resources) is gone."""
import unittest

from bigcherry.core import paths


class NoWriterCallerTests(unittest.TestCase):
    def test_no_tool_invokes_the_manifest_writer(self):
        hits = [str(p.relative_to(paths.REPO_ROOT)) for root in ("tools/bigcherry", "tools/lab")
                for p in sorted((paths.REPO_ROOT / root).rglob("*"))
                if p.suffix in {".py", ".sh"} and p.is_file() and "__pycache__" not in p.parts
                and "--write-manifest" in p.read_text(encoding="utf-8", errors="replace")]
        self.assertEqual(hits, [])

    def test_the_launch_bridge_module_is_gone(self):
        self.assertFalse((paths.REPO_ROOT / "tools/bigcherry/engine_adapters/radiance_launch.py").exists())


if __name__ == "__main__":
    unittest.main()
