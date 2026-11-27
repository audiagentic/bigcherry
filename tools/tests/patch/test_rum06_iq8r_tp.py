"""RUM06 IQ8R tensor-parallel scale geometry, with production patch code executed card-free."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vtp_promotion_support import load_patch_files, patched_pinned_source
from test_vtp14_n1_rank_code_selection import radiance_checkout

PKG19 = "rad0019_vtp14_n2_tp_ownership_fp8"
PKG30 = "rad0030_int8_pipelined_prefill"


def edit(package: str, name: str) -> str:
    return next(e.text for f in load_patch_files(package) for e in f.edits if e.id == name)


def compile_and_run(testcase: unittest.TestCase, source: str) -> subprocess.CompletedProcess[str]:
    cc = shutil.which("g++") or shutil.which("clang++")
    if not cc:
        testcase.skipTest("C++ host compiler unavailable")
    with tempfile.TemporaryDirectory(dir=REPO) as tmp:
        cpp = Path(tmp) / "test.cpp"
        exe = Path(tmp) / "test"
        cpp.write_text(source)
        built = subprocess.run([cc, "-std=c++17", "-O0", str(cpp), "-o", str(exe)],
                               capture_output=True, text=True)
        testcase.assertEqual(built.returncode, 0, built.stderr)
        return subprocess.run([str(exe)], capture_output=True, text=True)


class Iq8rTpTests(unittest.TestCase):
    def test_rmsnorm_plain_uses_probed_code_group(self):
        norm = edit(PKG30, "plain-norm-fed-pair")
        self.assertIn("n / (hi8 ? a.g8 : (int64_t)RAD_SCALE_GROUP)", norm)
        self.assertIn("hi8 ? a.s8 : a.s", norm)
        self.assertNotIn("n / RAD_SCALE_GROUP),", norm)

    def test_fused_mlp_code_scales_follow_pair(self):
        self.assertIn("g.n_embd / w.h.cg()", edit(PKG30, "iq8r-mlp-fold-input-scale"))
        self.assertIn("g.n_ff / w.ffn.cg()", edit(PKG30, "iq8r-mlp-fold-output-scale"))

    def test_real_ownership_rule_with_uneven_rank_shares(self):
        # Compile the exact block injected into the upstream plugin, rather than
        # independently re-implementing the intended ownership check.
        text = edit(PKG19, "n2f-iq8r-group-ownership")
        body = text.split("    /* ---- activation buffers.", 1)[0]
        source = (
            "#include <cstdint>\n#include <cstdio>\n#include <cassert>\n"
            "#define RAD_E_SHAPE 2\n"
            "struct Geom {bool tp_owned; int rank; int64_t n_ff, own_ff_lo;};\n"
            "struct Gdn {int64_t n_head_v, head_v, v_first; "
            "int64_t v_dim() const {return n_head_v*head_v;}};\n"
            "struct Model {bool trunk_i8; int64_t trunk_g8;};\n"
            "int check(const Geom& g, const Gdn& gc, const Model& m) {\n"
            + body + "  return 0;\n}\n"
            "int main() {\n"
            "  auto valid=[](int group,int rank,int64_t ff,int64_t flo,int64_t lv,int64_t vlo){\n"
            "    return check({true,rank,ff,flo},{lv,128,vlo},{true,group});};\n"
            "  assert(valid(256,0,5632,0,15,0)==RAD_E_SHAPE); // 5,5,6: 1920 V columns\n"
            "  assert(valid(256,0,7680,0,21,0)==RAD_E_SHAPE); // 7,7,2: 2688 V columns\n"
            "  assert(valid(256,0,5120,0,12,0)==0);\n"
            "  assert(valid(256,1,4864,5120,12,12)==0);\n"
            "  assert(valid(256,2,7424,9984,24,24)==0); // 4,4,8\n"
            "  assert(valid(256,0,6656,0,18,0)==0);\n"
            "  assert(valid(256,1,6656,6656,18,18)==0);\n"
            "  assert(valid(256,2,4096,13312,12,36)==0); // 6,6,4\n"
            "  assert(valid(128,0,5632,0,15,0)==0); // legacy group only if container declares it\n"
            "  assert(valid(256,0,5504,0,12,0)==RAD_E_SHAPE); // FFN width\n"
            "  assert(valid(256,0,5120,128,12,0)==RAD_E_SHAPE); // FFN start\n"
            "  assert(valid(256,0,5120,0,12,1)==RAD_E_SHAPE); // V start\n"
            "  assert(check({false,0,5504,128},{15,128,0},{true,256})==0); // no ownership\n"
            "  return 0;\n}\n"
        )
        ran = compile_and_run(self, source)
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertIn("V shard 1920 columns", ran.stderr)
        self.assertIn("V shard 2688 columns", ran.stderr)
        self.assertIn("scale group 256", ran.stderr)
        self.assertIn("FFN shard 5504 columns", ran.stderr)

    def test_dense_weight_span_uses_declared_scale_view_shape(self):
        span_patch = edit(PKG19, "n2f-dense-span")
        method = "    int span(" + span_patch.split("    int span(", 1)[1].split(
            "    /* Rows [lo, hi) of the stored block-fp8 weight", 1)[0]
        self.assertIn("w8h ? (rows ? 1 : g8)", method)
        self.assertIn("w8h_s_full", method)
        self.assertIn("mx || w8h", method)
        self.assertIn("linear_w8h_rowf(enc) || linear_w8h_scale2(enc) || iq8r",
                      edit(PKG19, "n2f-dense-scale-view-probe"))
        source = (
            "#include <cstdint>\n#include <cstdio>\n#include <cassert>\n"
            "using RadBuilder=int; using rad_weight=int;\n"
            "constexpr int RAD_SCALE_GROUP=128,RAD_MX_GROUP=32,RAD_SHARD_ROW=0,RAD_SHARD_COL=1,"
            "RAD_E_INVAL=1,RAD_OK=0;\n"
            "int64_t swlo=-1,swhi=-1;\n"
            "int rad_weight_shard_span(RadBuilder*,rad_weight w,int64_t lo,int64_t hi) {\n"
            "  if(w==2){swlo=lo;swhi=hi;}return 0;\n}\n"
            "#define RAD_ARCH_TRY(expr) do {int rc=(expr);if(rc)return rc;}while(0)\n"
            "struct Dense {bool mx=false,w8h=false,w8h_s_full=false;"
            "int64_t g8=128;rad_weight w=1,s=2,ref=0;\n"
            + method + "};\n"
            "int main() {\n"
            "  RadBuilder b=0;Dense d;d.w8h=true;d.g8=256;d.w8h_s_full=true;\n"
            "  assert(d.span(&b,RAD_SHARD_COL,256,768)==0 && swlo==256 && swhi==768);\n"
            "  assert(d.span(&b,RAD_SHARD_COL,128,512)==RAD_E_INVAL);\n"
            "  assert(d.span(&b,RAD_SHARD_ROW,16,32)==0 && swlo==16 && swhi==32);\n"
            "  assert(d.span(&b,RAD_SHARD_ROW,1,15)==RAD_E_INVAL);\n"
            "  d.w8h_s_full=false;d.g8=128;\n"
            "  assert(d.span(&b,RAD_SHARD_COL,256,512)==0 && swlo==2 && swhi==4);\n"
            "  d.w8h=false;\n"
            "  assert(d.span(&b,RAD_SHARD_COL,128,384)==0 && swlo==1 && swhi==3);\n"
            "  return 0;\n}\n"
        )
        ran = compile_and_run(self, source)
        self.assertEqual(ran.returncode, 0, ran.stderr)

    def test_patch_apply_and_idempotence_on_frozen_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned Radiance source; set RADIANCE_SRC")
        with patched_pinned_source(checkout, PKG30) as root:
            norm = (root / "arch/common/rad_linear.h").read_text()
            qwen = (root / "arch/qwen35_fp8/qwen35_fp8.cpp").read_text()
            self.assertIn("n / (hi8 ? a.g8 : (int64_t)RAD_SCALE_GROUP)", norm)
            self.assertIn("BC_RUM06_TP_SCALE", qwen)
            self.assertIn("BC_RUM06_DENSE_SCALE_PROBE", norm)


if __name__ == "__main__":
    unittest.main()
