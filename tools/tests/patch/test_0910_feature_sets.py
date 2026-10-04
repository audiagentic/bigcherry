"""Offline tests for 0910_feature_sets: mechanics on the pinned files, plus compile-and-run of the profile loader
against a temporary profile/ folder (QFP23)."""

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
_V = _REPO / "vendor/llama.cpp"
_FILES = ("ggml/src/ggml.c", "ggml/src/ggml-backend-reg.cpp", "tools/server/main.cpp", "CMakeLists.txt",
          "src/llama-model.cpp")
_CC = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")


def _load():
    spec = importlib.util.spec_from_file_location("patch_0910", _REPO / "patches/0910_feature_sets/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()

_PROFILES = {
    "base.ini": "# generic\n[gen]\ndescription = generic\nBIGCHERRY_ACT_Q81 = 1\n\n[other]\ndescription = o\nBIGCHERRY_X = 2\n",
    "model.ini": "[model]\ndescription = model profile\narch = qwen4exp\n@gen\nBIGCHERRY_SCHED_ASYNC_INPUTS = 1\n",
}


@unittest.skipUnless(all((_V / f).exists() for f in _FILES), "pinned vendor checkout not present")
class Patch0910Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in _FILES:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_V / rel, root / rel)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            snap = {f: (root / f).read_text(encoding="utf-8") for f in _FILES}
            c = snap["ggml/src/ggml.c"]
            self.assertEqual(c.count("bigcherry 0910: runtime profiles (BIGCHERRY_FEATURES"), 1)
            self.assertNotIn("exit(0)", c)  # the library never exits
            init = c[c.index("struct ggml_context * ggml_init(struct ggml_init_params params) {"):]
            self.assertTrue(init.split("\n")[1].strip().startswith("(void) ggml_bigcherry_features_init();"))
            reg = snap["ggml/src/ggml-backend-reg.cpp"]
            get_reg = reg[reg.index("static ggml_backend_registry & get_reg() {"):]
            self.assertLess(get_reg.index("ggml_bigcherry_features_init();"), get_reg.index("static ggml_backend_registry reg;"))
            load_all = reg[reg.index("void ggml_backend_load_all_from_path(const char * dir_path) {"):]
            self.assertTrue(load_all.split("\n")[1].strip().startswith("(void) ggml_bigcherry_features_init();"))
            main = snap["tools/server/main.cpp"]
            self.assertLess(main.index("ggml_bigcherry_features_init()"), main.index("return llama_server(argc, argv);"))
            cm = snap["CMakeLists.txt"]
            self.assertLess(cm.index("bigcherry 0910: ship the runtime profile/ folder"), cm.index("add_subdirectory(src)"))
            self.assertIn("install(FILES ${BIGCHERRY_PROFILE_FILES} DESTINATION ${CMAKE_INSTALL_BINDIR}/profile)", cm)
            lm = snap["src/llama-model.cpp"]
            create = lm[lm.index("llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params) {"):]
            self.assertLess(create.index("ggml_bigcherry_features_apply_arch(ml.get_arch_name().c_str());"),
                            create.index("return llama_model_create(arch, params);"))
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(snap, {f: (root / f).read_text(encoding="utf-8") for f in _FILES})


@unittest.skipUnless(_CC, "no C compiler")
class Patch0910Behaviour(unittest.TestCase):
    _exe: Path | None = None
    _td: tempfile.TemporaryDirectory | None = None

    @classmethod
    def setUpClass(cls):
        code = _P._N.split("#include <signal.h>\n", 1)[1]
        doc = env_docs("1310_act_q81", (EnvDoc("BIGCHERRY_ACT_Q81", "0|1", "0", 'act "q8" test'),))
        code = code.replace(ENV_DOC_TABLE_END, doc.edits[0].text + ENV_DOC_TABLE_END)
        main = (
            "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
            "#ifdef _WIN32\n#include <windows.h>\n#endif\n#define GGML_API\n"
            "static void ggml_critical_section_start(void) {}\nstatic void ggml_critical_section_end(void) {}\n" + code +
            "\nint main(void) {\n"
            "    int st = ggml_bigcherry_features_init();\n"
            "    if (ggml_bigcherry_features_init() != st) return 9;  /* idempotent */\n"
            "    if (getenv(\"TEST_ARCH\")) {\n"
            "        printf(\"arch_status=%d\\n\", ggml_bigcherry_features_apply_arch(getenv(\"TEST_ARCH\")));\n"
            "        if (ggml_bigcherry_features_apply_arch(getenv(\"TEST_ARCH\")) != 0) return 8;  /* second model: no-op */\n"
            "    }\n"
            "    printf(\"status=%d\\n\", st);\n"
            '    const char * k[] = {"BIGCHERRY_ACT_Q81", "BIGCHERRY_SCHED_ASYNC_INPUTS", "BIGCHERRY_X"};\n'
            '    for (int i = 0; i < 3; i++) { const char * v = getenv(k[i]); printf("%s=%s\\n", k[i], v ? v : "<unset>"); }\n'
            "    return 0;\n}\n"
        )
        cls._td = tempfile.TemporaryDirectory()
        d = Path(cls._td.name)
        (d / "t.c").write_text(main, encoding="utf-8")
        cls._exe = d / ("t.exe" if os.name == "nt" else "t")
        cc = subprocess.run([_CC, "-std=c11", "-D_GNU_SOURCE", "-D_CRT_SECURE_NO_WARNINGS", "-o", str(cls._exe), str(d / "t.c")],
                            capture_output=True, text=True)
        if cc.returncode != 0:
            raise AssertionError(cc.stderr[-3000:])
        prof = d / "profile"
        prof.mkdir()
        for name, text in _PROFILES.items():
            (prof / name).write_text(text, encoding="utf-8")
        cls.profile_dir = prof

    @classmethod
    def tearDownClass(cls):
        if cls._td is not None:
            cls._td.cleanup()

    def _run(self, env_extra, profiles=None):
        env = {k: v for k, v in os.environ.items() if not re.match(r"(BIGCHERRY_|GGML_HIP_)", k)}
        env["BIGCHERRY_PROFILES"] = str(profiles or self.profile_dir)
        env.update(env_extra)
        p = subprocess.run([str(self._exe)], env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout, p.stderr

    def test_profile_with_include_applies(self):
        out, err = self._run({"BIGCHERRY_FEATURES": "model"})
        self.assertIn("status=0", out)
        self.assertIn("BIGCHERRY_ACT_Q81=1", out)
        self.assertIn("BIGCHERRY_SCHED_ASYNC_INPUTS=1", out)
        self.assertIn("BIGCHERRY_X=<unset>", out)
        self.assertIn("BIGCHERRY_FEATURES model", err)

    def test_explicit_member_wins(self):
        out, err = self._run({"BIGCHERRY_FEATURES": "model", "BIGCHERRY_ACT_Q81": "0"})
        self.assertIn("BIGCHERRY_ACT_Q81=0", out)
        self.assertIn("BIGCHERRY_ACT_Q81=<explicit>", err)

    def test_help_returns_status_and_lists(self):
        out, err = self._run({"BIGCHERRY_FEATURES": "help"})
        self.assertIn("status=1", out)
        self.assertIn("model", err)
        self.assertIn("@gen", err)
        self.assertIn("BIGCHERRY_ACT_Q81 = 0|1 (default: 0) [1310_act_q81]", err)
        self.assertIn('act "q8" test', err)
        self.assertIn("BIGCHERRY_ACT_Q81=<unset>", out)  # help mutates nothing

    def test_errors_apply_nothing(self):
        for req in ("nope", "model,,gen", "model,model", "model," + "x" * 5000):
            out, err = self._run({"BIGCHERRY_FEATURES": req})
            self.assertIn("status=-1", out, req)
            self.assertIn("BIGCHERRY_ACT_Q81=<unset>", out, req)
            self.assertIn("no profile applied", err, req)

    def test_conflict_cycle_and_bad_file(self):
        bad = {
            "conflict": {"a.ini": "[p]\ndescription = d\nBIGCHERRY_ACT_Q81 = 1\n[q]\ndescription = d\nBIGCHERRY_ACT_Q81 = 2\n"
                                  "[r]\ndescription = d\n@p\n@q\n"},
            "cycle": {"a.ini": "[r]\ndescription = d\n@s\n[s]\ndescription = d\n@r\n"},
            "bad line": {"a.ini": "[r]\ndescription = d\nlowercase = 1\n"},
            "dup profile": {"a.ini": "[r]\ndescription = d\n", "b.ini": "[r]\ndescription = d\n"},
        }
        for label, files in bad.items():
            with tempfile.TemporaryDirectory() as td:
                for name, text in files.items():
                    (Path(td) / name).write_text(text, encoding="utf-8")
                out, err = self._run({"BIGCHERRY_FEATURES": "r"}, profiles=Path(td))
                self.assertIn("status=-1", out, label)
                self.assertIn("BIGCHERRY_ACT_Q81=<unset>", out, label)

    def test_single_file_and_missing(self):
        out, _ = self._run({"BIGCHERRY_FEATURES": "gen"}, profiles=self.profile_dir / "base.ini")
        self.assertIn("BIGCHERRY_ACT_Q81=1", out)
        out, err = self._run({"BIGCHERRY_FEATURES": "gen"}, profiles=self.profile_dir / "missing")
        self.assertIn("status=-1", out)
        self.assertIn("not found", err)

    def test_auto_picks_profile_by_arch(self):
        out, err = self._run({"BIGCHERRY_FEATURES": "auto", "TEST_ARCH": "qwen4exp"})
        self.assertIn("status=0", out)
        self.assertIn("arch_status=0", out)
        self.assertIn("BIGCHERRY_ACT_Q81=1", out)  # model -> @gen
        self.assertIn("BIGCHERRY_SCHED_ASYNC_INPUTS=1", out)
        self.assertIn("architecture 'qwen4exp' -> profile 'model'", err)

    def test_auto_without_matching_arch_applies_nothing(self):
        out, err = self._run({"BIGCHERRY_FEATURES": "auto", "TEST_ARCH": "llama"})
        self.assertIn("arch_status=0", out)
        self.assertIn("BIGCHERRY_ACT_Q81=<unset>", out)
        self.assertIn("no profile for architecture 'llama'", err)

    def test_explicit_profile_ignores_arch_hook(self):
        out, _ = self._run({"BIGCHERRY_FEATURES": "gen", "TEST_ARCH": "qwen4exp"})
        self.assertIn("BIGCHERRY_ACT_Q81=1", out)
        self.assertIn("BIGCHERRY_SCHED_ASYNC_INPUTS=<unset>", out)  # auto not requested -> model profile not applied

    def test_unset_does_nothing(self):
        out, _ = self._run({})
        self.assertIn("status=0", out)
        self.assertIn("BIGCHERRY_ACT_Q81=<unset>", out)


if __name__ == "__main__":
    unittest.main()
