"""RCS23: the owned IQR overlay is a required kernel library of every layout that runs a libr3 rank, listed before it."""
import re
import tomllib
import unittest
from pathlib import Path

from bigcherry.core import components, engines, paths

CONFIGS = paths.REPO_ROOT / "engines" / "radiance" / "configs"
OVERLAY, BASES = "libbciqr", ("libr3", "libr4d")


def kernel_lines():
    """(config name, kernels list, serves an IQR container): the IQR variants (iq4r, iq6r) are named in the model path."""
    for path in sorted(CONFIGS.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"(?m)^\s*kernels:\s*(\S+)", text)
        model = re.search(r"(?m)^\s*model:\s*(\S+)", text)
        if m:
            yield path.name, m.group(1).split(","), bool(model and re.search(r"iq[46]r", model.group(1)))


class KernelOrderTests(unittest.TestCase):
    def test_every_layout_with_a_libr3_rank_or_an_iqr_container_lists_the_overlay_first(self):
        seen = 0
        for name, tokens, iqr in kernel_lines():
            if "libr3" in tokens or iqr:
                seen += 1
                self.assertIn(OVERLAY, tokens, f"{name}: libr3 no longer builds the IQR units and libr4d's fused tile lives in the overlay; "
                                               f"list {OVERLAY} first")
                self.assertEqual(tokens.index(OVERLAY), 0, name)
        self.assertGreater(seen, 0)

    def test_no_layout_lists_the_overlay_without_the_base_it_overlays(self):
        for name, tokens, _ in kernel_lines():
            if OVERLAY in tokens:
                self.assertTrue(any(base in tokens for base in BASES), f"{name}: {OVERLAY} without a base library")
                bases = [tokens.index(b) for b in BASES if b in tokens]
                self.assertLess(tokens.index(OVERLAY), min(bases), f"{name}: {OVERLAY} must come before its bases")

    def test_the_overlay_ships_as_a_kernel_library_of_the_release(self):
        declared = components.load_all(paths.REPO_ROOT, engines.RADIANCE)
        self.assertIn("bc-iqr", declared)
        self.assertEqual(list(declared["bc-iqr"].outputs), ["kernels/libbciqr.so"])

    def test_libr3_no_longer_builds_a_unit_the_overlay_owns(self):
        root = paths.REPO_ROOT / "engines" / "radiance" / "components"
        excluded = {line.split()[0] for line in (root / "libr3" / "exclude.txt").read_text().splitlines() if line.startswith("r4d_")}
        origins = tomllib.loads((root / "bc-iqr" / "origins.toml").read_text())
        for unit in origins["unit"]:
            if unit["strategy"] != "whole-unit":
                continue   # an extracted family (a shared header) stays built by libr3 for the units that still use it
            self.assertIn(unit["name"], excluded)
            for rewrite in unit["absorbed-rewrites"]:
                self.assertTrue((paths.REPO_ROOT / rewrite).is_file(), rewrite)
                self.assertFalse((root / "libr3" / "native" / Path(rewrite).name).exists(), "an owned unit's rewrite stays archived")


if __name__ == "__main__":
    unittest.main()
