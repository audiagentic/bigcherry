"""Compile real emitted family hooks: production must not reference diagnostics."""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from bigcherry.patch.apply import FilePatch, apply_patch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "coverage_diagnostics_patch", ROOT / "patches/0700_coverage_counters/patch.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FAMILIES = ["MMQ", "MMVQ", "MMVF", "MMF", "BLAS"]


class CoverageDiagnosticsGateTests(unittest.TestCase):
    def transform(self, directory, index, content):
        path = Path(directory) / "hook.cpp"
        path.write_text(content, encoding="utf-8", newline="")
        patch = FilePatch("hook.cpp", (module.PATCHES[index].edits[-1],))
        result = apply_patch(patch, Path(directory))
        return result, path.read_text(encoding="utf-8"), patch

    def test_all_family_hooks_upgrade_once_and_preserve_dispatch(self):
        for i, family in enumerate(FAMILIES):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                original = module._count("GGML_HIP_FAMILY_" + family,
                                         "nullptr" if family != "BLAS" else None)
                result, updated, patch = self.transform(directory, i, original)
                self.assertTrue(result.ok, result.results)
                self.assertTrue(result.changed)
                self.assertEqual(updated.count("ggml_hip_dispatch_family("),
                                 original.count("ggml_hip_dispatch_family("))
                second = apply_patch(patch, Path(directory))
                self.assertTrue(second.ok)
                self.assertFalse(second.changed)

    def test_missing_and_ambiguous_old_hook_fail_without_writing(self):
        original = module._count("GGML_HIP_FAMILY_MMQ", "nullptr")
        for content in ("void unrelated() {}", original + original):
            with tempfile.TemporaryDirectory() as directory:
                result, updated, _ = self.transform(directory, 0, content)
                self.assertFalse(result.ok)
                self.assertEqual(updated, content)

    def test_compiled_production_has_no_diagnostic_dependencies(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as directory:
            functions = []
            for i, family in enumerate(FAMILIES):
                original = module._count("GGML_HIP_FAMILY_" + family,
                                         "nullptr" if family != "BLAS" else None)
                result, updated, _ = self.transform(directory, i, original)
                self.assertTrue(result.ok)
                functions.append(f"void entry{i}() {{\n{updated}\n}}")
            source = """
int ctx, src0, src1, ids, dst;
int dispatch_calls, executed_calls, probe_calls;
bool ggml_hip_dispatch_family(...) { ++dispatch_calls; return false; }
bool ggml_hip_dispatch_is_reentrant();
void ggml_hip_coverage_count_executed(int);
#ifdef GGML_HIP_DISPATCH_DIAGNOSTICS
bool ggml_hip_dispatch_is_reentrant() { ++probe_calls; return false; }
void ggml_hip_coverage_count_executed(int) { ++executed_calls; }
#endif
"""
            source += "\n".join(f"const int GGML_HIP_FAMILY_{f} = {i};"
                                for i, f in enumerate(FAMILIES))
            source += "\n" + "\n".join(functions)
            source += "\nint main() { entry0(); entry1(); entry2(); entry3(); entry4();\n"
            source += """
#ifdef GGML_HIP_DISPATCH_DIAGNOSTICS
return !(dispatch_calls == 4 && executed_calls == 5 && probe_calls == 5);
#else
return !(dispatch_calls == 4 && executed_calls == 0 && probe_calls == 0);
#endif
}
"""
            cpp = Path(directory) / "hook.cpp"
            cpp.write_text(source, encoding="utf-8")
            for diagnostics in (False, True):
                exe = Path(directory) / ("diagnostics.exe" if diagnostics else "production.exe")
                command = [compiler, "-std=c++17", "-O0", "-DGGML_HIP_DISPATCH",
                           str(cpp), "-o", str(exe)]
                if diagnostics:
                    command.append("-DGGML_HIP_DISPATCH_DIAGNOSTICS")
                built = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                ran = subprocess.run([str(exe)], capture_output=True, text=True)
                self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)


if __name__ == "__main__":
    unittest.main()
