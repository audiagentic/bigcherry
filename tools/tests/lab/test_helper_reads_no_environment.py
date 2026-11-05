"""The helper process reads its whole configuration from the serving manifest: no environment read, no allowlist.

RSC06 (serving manifest v1, spec §11 amendment 10). r3_moe_worker's translation unit is r3_moe_worker.cpp and every local
header it includes (transitively). None may call getenv, and the one environment write is GPU_MAX_HW_QUEUES, which HIP
reads before it initialises. The engine-side files of the same component (moe_remote_hook.cpp, moe_mailbox_*) are not part
of the helper's translation unit and are not scanned.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MOE = ROOT / "engines/radiance/components/moe-remote"
ENTRY = MOE / "r3_moe_worker.cpp"
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
# a call, not a mention in a comment or a log text: strip comments and string literals before looking
GETENV = re.compile(r"\b(?:std::)?getenv\s*\(")
SETENV = re.compile(r"\b(?:std::)?(?:setenv|putenv|unsetenv)\s*\(\s*(\"[^\"]*\")?")
SANCTIONED_WRITES = {'"GPU_MAX_HW_QUEUES"'}


def _code(text: str) -> str:
    """The text with comments and string literals blanked, keeping the first literal of a setenv call checkable."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def _blank_strings(text: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def helper_sources() -> list[Path]:
    seen: dict[Path, None] = {}
    todo = [ENTRY]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen[path] = None
        for name in INCLUDE.findall(path.read_text(encoding="utf-8")):
            target = (path.parent / name).resolve()
            if target.is_file() and MOE.parent in target.parents:
                todo.append(target)
    return list(seen)


def env_findings(text: str) -> list[str]:
    code = _code(text)
    found = []
    if GETENV.search(_blank_strings(code)):
        found.append("getenv")
    for m in SETENV.finditer(code):
        if m.group(1) not in SANCTIONED_WRITES:
            found.append(f"environment write {m.group(1)}")
    return found


class HelperReadsNoEnvironmentTests(unittest.TestCase):
    def test_the_helper_translation_unit_is_what_we_scan(self) -> None:
        names = {p.name for p in helper_sources()}
        # the entry, its settings, and the headers the old environment reads lived in
        self.assertTrue({"r3_moe_worker.cpp", "worker_settings.h", "worker_config.h", "worker_policy_config.h",
                         "worker_shape.h", "worker_owners.h", "worker_harness.h", "moe_remote_gpu.h"} <= names, names)
        self.assertNotIn("moe_remote_hook.cpp", names)   # engine side

    def test_no_helper_source_reads_the_environment(self) -> None:
        for path in helper_sources():
            with self.subTest(path.name):
                self.assertEqual(env_findings(path.read_text(encoding="utf-8")), [], str(path.relative_to(ROOT)))

    def test_the_only_environment_write_is_the_hip_queue_count(self) -> None:
        writes = [m.group(1) for p in helper_sources() for m in SETENV.finditer(_code(p.read_text(encoding="utf-8")))]
        self.assertEqual(writes, ['"GPU_MAX_HW_QUEUES"'])

    def test_the_guard_can_fail(self) -> None:
        self.assertEqual(env_findings('const char* v = std::getenv("R3_MOE_WORKER_X");'), ["getenv"])
        self.assertEqual(env_findings('const char* v = getenv ("X");'), ["getenv"])
        self.assertEqual(env_findings('setenv("R3_MOE_POLICY", "1", 0);'), ['environment write "R3_MOE_POLICY"'])
        self.assertEqual(env_findings('(void)::setenv("GPU_MAX_HW_QUEUES", "5", 0);'), [])
        self.assertEqual(env_findings('// getenv("X") in a comment\nputs("getenv(\\"X\\")");'), [])

    def test_a_helper_without_a_manifest_does_not_start(self) -> None:
        text = ENTRY.read_text(encoding="utf-8")
        self.assertIn('usage: r3_moe_worker --manifest FILE --role helperN', text)
        # exactly one place decides that an unmeasured manifest is acceptable
        self.assertEqual(text.count("PlanUse::measuring"), 1)
        self.assertEqual(text.count('"--measuring"'), 1)


if __name__ == "__main__":
    unittest.main()
