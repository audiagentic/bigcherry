"""HI82 design item 7: post_build_verify()/capture_build_identity() tests.

Named test_hi82_build_identity.py (not test_build_identity.py) because
tools/tests/test_build_identity.py already exists and tests the pre-existing,
unrelated tools/bigcherry/builds.py content-addressed build-reuse system
(BuildPlan/effective_build_id/binary_hash/validate_reuse, RE07/RE14/RE26) --
my first attempt at this file (via Write, not Read-then-Edit) silently
overwrote that file's content; caught and reverted via `git checkout --`
before committing anything, and this module's real overlap with
tools/bigcherry/builds.py is flagged as an open reconciliation question
in the HI82 plan item and a GPT follow-up request rather than resolved
here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.build_identity import (  # noqa: E402
    BuildIdentityError,
    CommandRequirement,
    post_build_verify,
)


class BuildIdentityVerificationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

        self.source = self.root / "source"
        self.build = self.root / "build"

        self.source.mkdir()
        self.build.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _cache(self, *, hip_flags: str = "") -> None:
        (self.build / "CMakeCache.txt").write_text(
            "\n".join((
                "CMAKE_GENERATOR:INTERNAL=Ninja",
                "CMAKE_HIP_ARCHITECTURES:STRING=gfx1100",
                f"CMAKE_HIP_FLAGS:STRING={hip_flags}",
                "",
            )),
            encoding="utf-8",
        )

    def _commands(self, hip_command: str) -> None:
        payload = [
            {
                "directory": str(self.build),
                "file": str(self.source / "kernel.cu"),
                "command": hip_command,
            },
            {
                "directory": str(self.build),
                "file": str(self.source / "host.cpp"),
                "command": (
                    f"clang++ -c {self.source / 'host.cpp'} -o {self.build / 'host.o'}"
                ),
            },
        ]
        (self.build / "compile_commands.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )

    def test_accepts_hip_flags_that_reached_compiler(self):
        flag = "-funsafe-math-optimizations"
        self._cache(hip_flags=flag)
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"{flag} -o {self.build / 'kernel.o'}"
        )

        evidence = post_build_verify(self.build, source_root=self.source, architecture="gfx1100")

        self.assertEqual(evidence.command_source, "compile_commands.json")
        self.assertEqual(evidence.hip_compile_command_count, 1)

        labels = {check.label for check in evidence.checks}
        self.assertIn("hip-architecture", labels)
        self.assertIn("cmake-hip-flags-propagation", labels)

    def test_rejects_cmake_hip_flags_that_did_not_propagate(self):
        self._cache(hip_flags="-funsafe-math-optimizations")
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"-o {self.build / 'kernel.o'}"
        )

        with self.assertRaisesRegex(BuildIdentityError, "did not propagate"):
            post_build_verify(self.build, source_root=self.source, architecture="gfx1100")

    def test_rejects_wrong_real_architecture(self):
        self._cache()
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1201 "
            f"-o {self.build / 'kernel.o'}"
        )

        with self.assertRaisesRegex(BuildIdentityError, "gfx1100"):
            post_build_verify(self.build, source_root=self.source, architecture="gfx1100")

    def test_custom_required_command_token(self):
        self._cache()
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"-DBIGCHERRY_EXAMPLE=1 -o {self.build / 'kernel.o'}"
        )

        evidence = post_build_verify(
            self.build, source_root=self.source, architecture="gfx1100",
            command_requirements=(
                CommandRequirement(
                    label="example patch flag", selector_regex=r"(?i)\.cu\b",
                    required_tokens=("-DBIGCHERRY_EXAMPLE=1",),
                ),
            ),
        )

        self.assertEqual(evidence.checks[-1].label, "example patch flag")
        self.assertEqual(evidence.checks[-1].status, "pass")

    def test_custom_forbidden_command_token(self):
        self._cache()
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"-DBIGCHERRY_BAD_FLAG=1 -o {self.build / 'kernel.o'}"
        )

        with self.assertRaisesRegex(BuildIdentityError, "forbidden_present"):
            post_build_verify(
                self.build, source_root=self.source, architecture="gfx1100",
                command_requirements=(
                    CommandRequirement(
                        label="negative control", selector_regex=r"(?i)\.cu\b",
                        forbidden_tokens=("-DBIGCHERRY_BAD_FLAG=1",),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
