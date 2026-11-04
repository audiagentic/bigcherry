"""The resolver (bc-serve serving.cpp) and the v1 manifest registry accept the same values for the enum settings the
manifest carries, so the writer never meets a resolved value the manifest cannot hold (RSC08 S2a)."""
from __future__ import annotations

import re
import tomllib
import unittest

from bigcherry.core import paths

SERVING = paths.REPO_ROOT / "engines" / "radiance" / "components" / "serving"
# resolver key -> v1 registry path
MAP = {
    "service.retention": "/spec/intent/service/retention",
    "service.quality": "/spec/intent/service/quality",
    "engine.prefix_resume": "/spec/processes/engine/execution/prefix_resume",
}


class ResolverV1EnumTests(unittest.TestCase):
    def test_enum_choices_are_the_registry_values(self) -> None:
        cpp = (SERVING / "serving.cpp").read_text(encoding="utf-8")
        registry = {f["path"]: f for f in tomllib.loads((SERVING / "contract" / "serving-v1.toml").read_text(encoding="utf-8"))["field"]}
        for key, path in MAP.items():
            with self.subTest(key=key):
                m = re.search(r'add\("' + re.escape(key) + r'".*?\{((?:"[a-z_]+",?\s*)+)\}\);', cpp, re.S)
                self.assertIsNotNone(m, f"{key} has an enum choice list in serving.cpp")
                self.assertEqual(sorted(re.findall(r'"([a-z_]+)"', m.group(1))), sorted(registry[path]["values"]))


if __name__ == "__main__":
    unittest.main()
