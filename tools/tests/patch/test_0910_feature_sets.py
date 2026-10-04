"""Offline tests for 0910_feature_sets: mechanics on pinned ggml.c, plus compile-and-run of the expansion code."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import ENV_DOC_TABLE_END, EnvDoc, apply_all, env_docs  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_REL = "ggml/src/ggml.c"
_VENDOR = _REPO / "vendor/llama.cpp" / _REL
_CC = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")


def _load():
    spec = importlib.util.spec_from_file_location("patch_0910", _REPO / "patches/0910_feature_sets/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch0910Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / _REL).parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, root / _REL)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _REL).read_text(encoding="utf-8")
            self.assertEqual(out.count("bigcherry 0910: named feature sets"), 1)
            self.assertIn("__attribute__((constructor)) static void bc_feature_sets_ctor", out)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(out, (root / _REL).read_text(encoding="utf-8"))


@unittest.skipUnless(_CC, "no C compiler")
class Patch0910Behaviour(unittest.TestCase):
    def _run(self, env_extra):
        code = _P._N.split("#include <signal.h>\n", 1)[1]
        # one documented flag, as a patch's env_docs() row adds it (ACT_Q81 documented, QSA_HOST_REMAP not)
        doc = env_docs("1310_act_q81", (EnvDoc("BIGCHERRY_ACT_Q81", "0|1", "0", 'act "q8" test'),))
        code = code.replace(ENV_DOC_TABLE_END, doc.edits[0].text + ENV_DOC_TABLE_END)
        main = (
            "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n" + code +
            "\nint main(void) {\n"
            '    const char * k[] = {"BIGCHERRY_ACT_Q81", "BIGCHERRY_QSA_HOST_REMAP", "GGML_HIP_Q8_1_CACHE_MODE"};\n'
            '    for (int i = 0; i < 3; i++) { const char * v = getenv(k[i]); printf("%s=%s\\n", k[i], v ? v : "<unset>"); }\n'
            "    return 0;\n}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "t.c"
            exe = Path(td) / ("t.exe" if os.name == "nt" else "t")
            src.write_text(main, encoding="utf-8")
            cc = subprocess.run([_CC, "-std=c11", "-D_GNU_SOURCE", "-D_CRT_SECURE_NO_WARNINGS", "-o", str(exe), str(src)],
                                check=False, capture_output=True, text=True)
            self.assertEqual(cc.returncode, 0, cc.stderr[-2000:])
            env = {k: v for k, v in os.environ.items() if not re.match(r"(BIGCHERRY_|GGML_HIP_)", k)}
            env.update(env_extra)
            p = subprocess.run([str(exe)], env=env, capture_output=True, text=True, check=True)
            return p.stdout, p.stderr

    def test_set_expands(self):
        out, err = self._run({"BIGCHERRY_FEATURES": "flashnext-v6"})
        self.assertIn("BIGCHERRY_ACT_Q81=1", out)
        self.assertIn("BIGCHERRY_QSA_HOST_REMAP=1", out)
        self.assertIn("GGML_HIP_Q8_1_CACHE_MODE=on", out)
        self.assertIn("BIGCHERRY_FEATURES flashnext-v6:", err)
        self.assertIn("BIGCHERRY_QSA_HOST_REMAP=1(not in build)", err)

    def test_explicit_member_wins(self):
        out, err = self._run({"BIGCHERRY_FEATURES": "flashnext-v6", "BIGCHERRY_ACT_Q81": "0"})
        self.assertIn("BIGCHERRY_ACT_Q81=0", out)
        self.assertIn("BIGCHERRY_ACT_Q81=0(explicit)", err)
        self.assertIn("BIGCHERRY_QSA_HOST_REMAP=1", out)

    def test_help_lists_sets_and_docs(self):
        _, err = self._run({"BIGCHERRY_FEATURES": "help"})
        self.assertIn("flashnext-v6 - ", err)
        self.assertIn("BIGCHERRY_ACT_Q81 = 0|1 (default: 0) [1310_act_q81]", err)
        self.assertIn('act "q8" test', err)
        self.assertIn("BIGCHERRY_QSA_HOST_REMAP=1  (no patch in this build documents it)", err)
        self.assertIn("BIGCHERRY_FEATURES = ", err)

    def test_unset_and_unknown(self):
        out, _ = self._run({})
        self.assertIn("BIGCHERRY_ACT_Q81=<unset>", out)
        out, err = self._run({"BIGCHERRY_FEATURES": "nope"})
        self.assertIn("BIGCHERRY_ACT_Q81=<unset>", out)
        self.assertIn("unknown set 'nope'", err)


class EnvDocsLoader(unittest.TestCase):
    """registry.load_implementation turns ENV_DOCS into one 0910 help-table FilePatch, failing closed without 0910."""

    def _registry(self, td, requires):
        from bigcherry.patch import registry
        root = Path(td) / "patches"
        pkg = root / "1999_env_demo"
        pkg.mkdir(parents=True)
        (pkg / "patch.toml").write_text(
            'schema = 1\nid = "1999_env_demo"\norder = 1999\nstate = "untested"\nkind = "enhancement"\n'
            + f"requires = {requires!r}\n".replace("'", '"'), encoding="utf-8")
        (pkg / "patch.py").write_text(
            "from bigcherry.patcher import Edit, EnvDoc, FilePatch\n"
            "PATCHES = [FilePatch(path='a.c', edits=(Edit(id='x', anchor='A', mode='insert_after', text='B',"
            " guard='B', rationale='r'),))]\n"
            "ENV_DOCS = (EnvDoc('BIGCHERRY_DEMO', '0|1', '0', 'demo flag'),)\n", encoding="utf-8")
        reg = registry.load_registry(root)
        return registry, reg.get("1999_env_demo"), root

    def test_env_docs_appended(self):
        with tempfile.TemporaryDirectory() as td:
            registry, desc, root = self._registry(td, ["0910_feature_sets"])
            loaded = registry.load_implementation(desc, root=root)
            self.assertEqual([p.path for p in loaded], ["a.c", "ggml/src/ggml.c"])
            self.assertIn('"BIGCHERRY_DEMO", "1999_env_demo", "0|1", "0", "demo flag"', loaded[1].edits[0].text)

    def test_env_docs_without_0910_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            registry, desc, root = self._registry(td, [])
            with self.assertRaises(registry.PatchRegistryError):
                registry.load_implementation(desc, root=root)


if __name__ == "__main__":
    unittest.main()
