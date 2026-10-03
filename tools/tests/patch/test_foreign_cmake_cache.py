"""A name-keyed build dir whose CMake cache belongs to another source is reset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bigcherry.patch.campaign import build


class ForeignCmakeCacheTests(unittest.TestCase):
    def _setup(self, root: Path, home: Path) -> Path:
        build_dir = root / "control"
        (build_dir / "CMakeFiles").mkdir(parents=True)
        (build_dir / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={home}\n", encoding="utf-8")
        (build_dir / "bigcherry-configure-request.json").write_text("{}", encoding="utf-8")
        return build_dir

    def test_cache_for_another_source_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / "src-old", root / "src-new"
            old.mkdir()
            new.mkdir()
            build_dir = self._setup(root, old)
            build._drop_foreign_cmake_cache(build_dir, new)
            self.assertFalse((build_dir / "CMakeCache.txt").exists())
            self.assertFalse((build_dir / "CMakeFiles").exists())
            self.assertFalse((build_dir / "bigcherry-configure-request.json").exists())

    def test_cache_for_the_same_source_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            build_dir = self._setup(root, src)
            build._drop_foreign_cmake_cache(build_dir, src)
            self.assertTrue((build_dir / "CMakeCache.txt").exists())
            self.assertTrue((build_dir / "bigcherry-configure-request.json").exists())


if __name__ == "__main__":
    unittest.main()
