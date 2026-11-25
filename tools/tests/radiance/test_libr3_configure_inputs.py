"""libr3 regenerates its rewritten copy of libr4d at configure time; every source type the rewriter reads must be a
configure dependency, or an existing build directory keeps a copy made from the old (unpatched or older) text."""
import ast
import re
import unittest
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[3] / "engines/radiance/components/libr3"


def _rewriter_suffixes() -> set[str]:
    tree = ast.parse((COMPONENT / "r3_rewrite.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(isinstance(op, ast.NotIn) for op in node.ops):
            comp = node.comparators[0]
            if isinstance(comp, ast.Tuple) and all(isinstance(e, ast.Constant) and str(e.value).startswith(".") for e in comp.elts):
                return {str(e.value) for e in comp.elts}
    raise AssertionError("the rewriter's source suffix filter was not found")


class Libr3ConfigureInputs(unittest.TestCase):
    def test_every_source_type_the_rewriter_reads_is_a_configure_dependency(self):
        cmake = (COMPONENT / "CMakeLists.txt").read_text(encoding="utf-8")
        match = re.search(r"file\(GLOB r4d_inputs([^)]*)\)", cmake)
        self.assertIsNotNone(match, "libr3 must glob the libr4d sources it rewrites as configure dependencies")
        self.assertIn("CMAKE_CONFIGURE_DEPENDS ${r4d_inputs}", cmake)
        globbed = match.group(1)
        for suffix in sorted(_rewriter_suffixes()):
            self.assertIn("libr4d/*" + suffix, globbed, f"{suffix} sources are rewritten but not a configure dependency")
        self.assertIn("libr4d/CMakeLists.txt", globbed)


if __name__ == "__main__":
    unittest.main()
