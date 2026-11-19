"""One bounded retry of a `cmake --build` whose compiler died from a signal (a random clang/gcc crash seen at ~1 in 160
identical compiles with plenty of memory free), recorded beside the build and in the work root; never for a real
compile error, a user interrupt or any other command."""

import contextlib
import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.core import process
from bigcherry.release import pin_bump

CRASH = ("FAILED: [code=1] CMakeFiles/libr3.dir/units/r4d_vit_bf16.hip.o \n"
         "clang++: error: unable to execute command: Segmentation fault (core dumped)\n"
         "clang++: error: clang frontend command failed due to signal (use -v to see invocation)\n")
COMPILE_ERROR = "FAILED: [code=1] CMakeFiles/x.dir/a.cpp.o \na.cpp:3:5: error: expected ';' after expression\n"
INTERRUPT = "FAILED: [code=1] CMakeFiles/x.dir/a.cpp.o \nclang++: error: unable to execute command: Interrupt\n"
GCC_CRASH = "FAILED: [code=1] CMakeFiles/s.dir/session.cpp.o \ng++: internal compiler error: Segmentation fault signal terminated program cc1plus\n"


class BuildRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = pin_bump.exec_capable_tempdir(Path(os.environ.get("BC_WORK_ROOT") or tempfile.gettempdir()) / "test-tmp")
        if scratch is not None:
            self.addCleanup(shutil.rmtree, scratch, ignore_errors=True)
        self._tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bin, self.build, self.work = self.root / "bin", self.root / "build", self.root / "work"
        for d in (self.bin, self.build, self.work):
            d.mkdir()
        self.calls = self.root / "calls"
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "BC_WORK_ROOT": str(self.work)}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_cmake(self, *outputs: tuple[str, int]) -> None:
        """A cmake whose n-th call prints outputs[n][0] and exits outputs[n][1] (the last one repeats)."""
        script = self.bin / "cmake"
        lines = ["#!/bin/sh", f'echo x >> "{self.calls}"', f'n=$(wc -l < "{self.calls}")', 'case $n in']
        for i, (text, code) in enumerate(outputs, start=1):
            pat = f"{i})" if i < len(outputs) else "*)"
            lines.append(f"{pat} printf '%s' '{text}'; exit {code};;")
        lines.append("esac")
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)

    def n_calls(self) -> int:
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def run_build(self, *argv: str) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            process.run_checked(["cmake", "--build", str(self.build), *argv], self.root)
        return out.getvalue()

    def receipts(self, where: Path) -> list[dict]:
        return [json.loads(line) for line in where.read_text().splitlines()] if where.exists() else []

    def test_a_compiler_signal_crash_is_retried_once_and_recorded(self) -> None:
        self.fake_cmake((CRASH, 1), ("ok\n", 0))
        out = self.run_build("-j")
        self.assertEqual(2, self.n_calls())
        self.assertIn("Segmentation fault", out)           # the build output still reaches the log unchanged
        self.assertIn("retrying", out)
        mine = self.receipts(self.build / "bigcherry-compile-retries.jsonl")
        host = self.receipts(self.work / "build-compile-retries.jsonl")
        self.assertEqual(mine, host)
        self.assertEqual(["crash", "retry"], [r["event"] for r in mine])
        crash = mine[0]
        self.assertEqual(("SIGSEGV", 1), (crash["signal"], crash["attempt"]))
        self.assertEqual("CMakeFiles/libr3.dir/units/r4d_vit_bf16.hip.o", crash["unit"])
        self.assertEqual(str(self.build), crash["build_dir"])
        self.assertEqual(("ok", 2), (mine[1]["result"], mine[1]["attempt"]))

    def test_a_second_crash_fails_the_build_and_says_so(self) -> None:
        self.fake_cmake((CRASH, 1), (CRASH, 1))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_build()
        self.assertEqual(2, self.n_calls())                # exactly one retry
        mine = self.receipts(self.build / "bigcherry-compile-retries.jsonl")
        self.assertEqual(["crash", "retry"], [r["event"] for r in mine])
        self.assertEqual("failed", mine[1]["result"])
        self.assertTrue(mine[1]["crashed_again"])

    def test_a_real_compile_error_is_not_retried(self) -> None:
        self.fake_cmake((COMPILE_ERROR, 1), ("ok\n", 0))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_build()
        self.assertEqual(1, self.n_calls())
        self.assertFalse((self.build / "bigcherry-compile-retries.jsonl").exists())

    def test_an_interrupt_is_not_retried(self) -> None:
        self.fake_cmake((INTERRUPT, 1), ("ok\n", 0))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_build()
        self.assertEqual(1, self.n_calls())

    def test_a_gcc_internal_compiler_error_signal_is_retried_too(self) -> None:
        self.fake_cmake((GCC_CRASH, 1), ("ok\n", 0))
        self.run_build()
        self.assertEqual(2, self.n_calls())
        crash = self.receipts(self.build / "bigcherry-compile-retries.jsonl")[0]
        self.assertEqual(("SIGSEGV", "CMakeFiles/s.dir/session.cpp.o"), (crash["signal"], crash["unit"]))

    def test_other_commands_never_retry(self) -> None:
        self.fake_cmake((CRASH, 1), ("ok\n", 0))
        with self.assertRaises(subprocess.CalledProcessError):
            process.run_checked(["cmake", "--install", str(self.build)], self.root)
        self.assertEqual(1, self.n_calls())

    def test_a_clean_build_writes_no_receipt(self) -> None:
        self.fake_cmake(("ok\n", 0))
        self.run_build()
        self.assertEqual(1, self.n_calls())
        self.assertFalse((self.build / "bigcherry-compile-retries.jsonl").exists())
        self.assertFalse((self.work / "build-compile-retries.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
