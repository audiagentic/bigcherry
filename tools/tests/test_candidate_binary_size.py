"""Tests for HI52 part 3's per-candidate device-size extraction.

Fixture-driven on purpose: no GPU, no ROCm toolchain, and no 172 MB library
required. The fixtures are verbatim shapes taken from a real `llvm-readelf
--syms` run against a real gfx1100 code object extracted from
`libggml-hip.so.0.18.0`, so they exercise the actual output format rather than
an idealised one.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import candidate_binary_size as cbs  # noqa: E402


# Verbatim format from a real code object: llvm-readelf --syms prints BOTH
# .dynsym and .symtab, so every kernel appears twice, and the .kd descriptor is
# an OBJECT rather than a FUNC.
REAL_READELF_OUTPUT = """
Symbol table '.dynsym' contains 4 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
     1: 0000000000001b00  3236 FUNC    GLOBAL PROTECTED   7 _ZL7acc_f32PKfS0_Pflllllllll
     2: 0000000000000a00    64 OBJECT  GLOBAL DEFAULT     6 _ZL7acc_f32PKfS0_Pflllllllll.kd
     3: 0000000000004818     1 OBJECT  GLOBAL DEFAULT    10 __hip_cuid_4e906e904329e3f9

Symbol table '.symtab' contains 10 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
     1: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   ABS _ZL7acc_f32PKfS0_Pflllllllll.num_vgpr
     7: 0000000000001b00  3236 FUNC    GLOBAL PROTECTED   7 _ZL7acc_f32PKfS0_Pflllllllll
     8: 0000000000000a00    64 OBJECT  GLOBAL DEFAULT     6 _ZL7acc_f32PKfS0_Pflllllllll.kd
     9: 0000000000004818     1 OBJECT  GLOBAL DEFAULT    10 __hip_cuid_4e906e904329e3f9
"""

# Real mangled names and real sizes observed on gfx1100.
MMQ_MAIN_Q8_J64_FB0 = ("_ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_PKf"
                       "15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_")
MMQ_FIXUP_Q8_J64_FB0 = ("_ZL24mul_mat_q_stream_k_fixupIL9ggml_type8ELi64ELb0EEvPKcPfS2_"
                        "PKi15HIP_vector_typeIjLj3EEiiiii")
MMQ_MAIN_Q2K_J80_FB0 = ("_ZL9mul_mat_qIL9ggml_type10ELi80ELb0EEvPKcPKiS4_S4_PfS5_PKf"
                        "15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_")

# Real mangled names, verified 2026-08-11 against bc-build-vault (ROCm 7.2.4
# clang, gfx1100) by demangling with c++filt -- see HI09b.
MMVQ_Q8_W1_NW1_RPB2_SK0 = (
    "_ZL13mul_mat_vec_qIL9ggml_type8ELi1ELb0ELb0ELi1ELi2EEvPKvS2_PKi"
    "31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj")
MMVQ_MOE_Q8_RPB2 = (
    "_ZL17mul_mat_vec_q_moeIL9ggml_type8ELi2EEvPKvS2_PKiPfj"
    "15HIP_vector_typeIjLj3EEjjjjjjjjj")
MMVF_F32_W3_BS256 = (
    "_ZL13mul_mat_vec_fIffLi3ELi256ELb0ELb0EEvPKT_PKfPKi"
    "31ggml_cuda_mm_fusion_args_devicePfi15HIP_vector_typeIjLj3EEiiiSA_iiiSA_iiii")
MMVF_F16_W2_BS32_ACCF16 = (
    "_ZL13mul_mat_vec_fI6__halfS0_Li2ELi32ELb0ELb0EEvPKT_PKfPKi"
    "31ggml_cuda_mm_fusion_args_devicePfi15HIP_vector_typeIjLj3EEiiiSB_iiiSB_iiii")
MMF_F32_W7_NW4 = "_ZL9mul_mat_fIfLi32ELi7ELi4ELb1EEvPKT_PKfPKiPfiiiiiiiiiiiiiiii"


def _sym_table(entries: list[tuple[int, str, str]]) -> str:
    lines = ["", "Symbol table '.symtab' contains 99 entries:",
             "   Num:    Value          Size Type    Bind   Vis       Ndx Name"]
    for index, (size, kind, name) in enumerate(entries, start=1):
        lines.append(f"  {index:4d}: 0000000000001b00 {size:5d} {kind:7s} "
                     f"GLOBAL PROTECTED   7 {name}")
    return "\n".join(lines) + "\n"


def _manifest(candidates: list[dict]) -> dict:
    return {"source_revision": "abc123", "producer_manifest_hash": "deadbeef",
            "candidates": candidates}


def _mmq_candidate(stable_name: str, type_name: str, j: int, fallback: bool,
                   architectures: list[str]) -> dict:
    return {
        "stable_name": stable_name, "family": "mmq",
        "source_class": "existing_runtime", "architectures": architectures,
        "config": {"type": type_name, "j": j, "fallback": fallback,
                   "nthreads": 256, "occupancy": 2, "i": 128,
                   "sram_layout": type_name, "k_vram": 256, "stream_k": False},
    }


class SymbolParsingTests(unittest.TestCase):
    def test_parses_real_readelf_output_once_per_symbol(self):
        # The .dynsym/.symtab double-listing must collapse to one entry, and the
        # 64-byte .kd OBJECT must not be counted as device code.
        with mock.patch.object(cbs, "_run", return_value=REAL_READELF_OUTPUT):
            sizes = cbs.symbol_sizes(Path("obj"), "llvm-readelf")
        self.assertEqual(sizes, {"_ZL7acc_f32PKfS0_Pflllllllll": 3236})

    def test_zero_size_and_non_func_symbols_excluded(self):
        raw = _sym_table([(0, "FUNC", "_ZL9emptyv"), (64, "OBJECT", "_ZL9thing.kd"),
                          (128, "FUNC", "_ZL9realv")])
        with mock.patch.object(cbs, "_run", return_value=raw):
            sizes = cbs.symbol_sizes(Path("obj"), "llvm-readelf")
        self.assertEqual(sizes, {"_ZL9realv": 128})

    def test_parse_mmq_main_kernel(self):
        parsed = cbs.parse_mmq_symbol(MMQ_MAIN_Q8_J64_FB0)
        self.assertIsNotNone(parsed)
        key, kernel = parsed
        self.assertEqual(key, cbs.MmqKey("q8_0", 64, False))
        self.assertEqual(kernel, "mul_mat_q")

    def test_parse_mmq_fixup_kernel(self):
        key, kernel = cbs.parse_mmq_symbol(MMQ_FIXUP_Q8_J64_FB0)
        self.assertEqual(key, cbs.MmqKey("q8_0", 64, False))
        self.assertEqual(kernel, "mul_mat_q_stream_k_fixup")

    def test_type_code_decoded_from_ggml_enum_not_position(self):
        # ggml_type has gaps (4 and 5 are absent), so 10 is q2_k, not q3_k.
        key, _ = cbs.parse_mmq_symbol(MMQ_MAIN_Q2K_J80_FB0)
        self.assertEqual(key.type_name, "q2_k")

    def test_fallback_flag_decoded(self):
        symbol = MMQ_MAIN_Q8_J64_FB0.replace("ELi64ELb0EE", "ELi64ELb1EE")
        key, _ = cbs.parse_mmq_symbol(symbol)
        self.assertTrue(key.fallback)

    def test_non_mmq_symbol_returns_none(self):
        self.assertIsNone(cbs.parse_mmq_symbol("_ZL7acc_f32PKfS0_Pflllllllll"))
        self.assertIsNone(cbs.parse_mmq_symbol("_ZL18flash_attn_ext_vecILi256E"))

    def test_unknown_ggml_type_code_returns_none(self):
        # A type id this tree has no name for must not silently become a
        # plausible-looking candidate key.
        symbol = MMQ_MAIN_Q8_J64_FB0.replace("ggml_type8E", "ggml_type250E")
        self.assertIsNone(cbs.parse_mmq_symbol(symbol))

    # ---------------------------------------------------------- mmvq (HI09b)

    def test_parse_mmvq_main_kernel(self):
        key, kernel = cbs.parse_mmvq_symbol(MMVQ_Q8_W1_NW1_RPB2_SK0)
        self.assertEqual(key, cbs.MmvqKey("q8_0", 1, 1, 2, False))
        self.assertEqual(kernel, "mul_mat_vec_q")

    def test_mmvq_moe_kernel_is_not_swallowed_by_the_main_parser(self):
        # mul_mat_vec_q is a strict string prefix of mul_mat_vec_q_moe; the
        # length-prefixed mangling (13 vs 17) must keep them apart.
        self.assertIsNone(cbs.parse_mmvq_symbol(MMVQ_MOE_Q8_RPB2))

    def test_mmvq_moe_kernel_parses_as_its_own_shape_and_is_unmapped(self):
        # Not mapped to any candidate (this catalog has no MoE-mode MMVQ
        # candidates), but must be recognised as a distinct kernel rather
        # than silently mis-parsed as the main one.
        self.assertIsNone(cbs.parse_mmvq_symbol(MMVQ_MOE_Q8_RPB2))
        self.assertIsNotNone(cbs._MMVQ_MOE_SYMBOL.match(MMVQ_MOE_Q8_RPB2))

    def test_mmvq_defaulted_explicit_params_resolve(self):
        # nwarps_explicit = rows_per_block_explicit = 0 ("derive as upstream")
        # is mangled explicitly in this build, not omitted -- must resolve,
        # not report missing.
        symbol = MMVQ_Q8_W1_NW1_RPB2_SK0.replace("ELi1ELi2EE", "ELi0ELi0EE")
        key, _ = cbs.parse_mmvq_symbol(symbol)
        self.assertEqual((key.nwarps, key.rows_per_block), (0, 0))

    def test_mmvq_small_k_flag_decoded(self):
        symbol = MMVQ_Q8_W1_NW1_RPB2_SK0.replace("ELb0ELi1ELi2EE", "ELb1ELi1ELi2EE")
        key, _ = cbs.parse_mmvq_symbol(symbol)
        self.assertTrue(key.small_k)

    def test_mmvq_key_of_candidate_round_trips(self):
        candidate = {"config": {"type": "q8_0", "width": 1, "nwarps": 1,
                                "rows_per_block": 2, "small_k": False}}
        key, _ = cbs.parse_mmvq_symbol(MMVQ_Q8_W1_NW1_RPB2_SK0)
        self.assertEqual(cbs.mmvq_key_of_candidate(candidate), key)

    # ---------------------------------------------------------- mmvf (HI09b)

    def test_parse_mmvf_f32_kernel(self):
        key, kernel = cbs.parse_mmvf_symbol(MMVF_F32_W3_BS256)
        self.assertEqual(key, cbs.MmvfKey("f32", 3, 256, "f32"))
        self.assertEqual(kernel, "mul_mat_vec_f")

    def test_parse_mmvf_f16_substitution_accumulator(self):
        # S0_ means type_acc == T verbatim: __half source with __half (f16)
        # accumulator, not the more common f32-accumulator form.
        key, _ = cbs.parse_mmvf_symbol(MMVF_F16_W2_BS32_ACCF16)
        self.assertEqual(key, cbs.MmvfKey("f16", 2, 32, "f16"))

    def test_mmvf_key_of_candidate_round_trips(self):
        candidate = {"config": {"type": "f32", "width": 3, "block_size": 256,
                                "accumulator": "f32"}}
        key, _ = cbs.parse_mmvf_symbol(MMVF_F32_W3_BS256)
        self.assertEqual(cbs.mmvf_key_of_candidate(candidate), key)

    def test_non_mmvf_symbol_returns_none(self):
        self.assertIsNone(cbs.parse_mmvf_symbol(MMQ_MAIN_Q8_J64_FB0))

    # ----------------------------------------------------------- mmf (HI09b)

    def test_parse_mmf_kernel(self):
        # T is the *packed* compute type (float here has no packed form
        # distinct from itself); rows_per_block (32) and has_ids (true) are
        # not part of a candidate's config and are correctly absent from the
        # key.
        key, kernel = cbs.parse_mmf_symbol(MMF_F32_W7_NW4)
        self.assertEqual(key, cbs.MmfKey("f32", 7, 4))
        self.assertEqual(kernel, "mul_mat_f")

    def test_parse_mmf_packed_half_type(self):
        symbol = "_ZL9mul_mat_fI7__half2Li32ELi7ELi4ELb1EEvPKT_PKfPKiPfiiiiiiiiiiiiiiii"
        key, _ = cbs.parse_mmf_symbol(symbol)
        self.assertEqual(key.type_name, "f16")

    def test_parse_mmf_packed_bfloat16_type(self):
        symbol = ("_ZL9mul_mat_fI15__hip_bfloat162Li32ELi7ELi4ELb1EE"
                  "vPKT_PKfPKiPfiiiiiiiiiiiiiiii")
        key, _ = cbs.parse_mmf_symbol(symbol)
        self.assertEqual(key.type_name, "bf16")

    def test_mmf_key_of_candidate_round_trips(self):
        candidate = {"config": {"type": "f32", "width": 7, "nwarps": 4}}
        key, _ = cbs.parse_mmf_symbol(MMF_F32_W7_NW4)
        self.assertEqual(cbs.mmf_key_of_candidate(candidate), key)

    def test_non_mmf_symbol_returns_none(self):
        self.assertIsNone(cbs.parse_mmf_symbol(MMQ_MAIN_Q8_J64_FB0))

    # ------------------------------------------------- cross-family coverage

    def test_resolution_counts_move_per_family(self):
        # The regression this test guards: a broken rule for one family must
        # degrade only that family's resolution, never the others'.
        for symbol, expect_family in (
            (MMQ_MAIN_Q8_J64_FB0, "mmq"),
            (MMVQ_Q8_W1_NW1_RPB2_SK0, "mmvq"),
            (MMVF_F32_W3_BS256, "mmvf"),
            (MMF_F32_W7_NW4, "mmf"),
        ):
            parsed = cbs._parse_mapped_symbol(symbol)
            self.assertIsNotNone(parsed, symbol)
            family, _, _ = parsed
            self.assertEqual(family, expect_family)


class FoldTests(unittest.TestCase):
    def test_same_symbol_same_size_across_objects_is_not_a_warning(self):
        raw = _sym_table([(100, "FUNC", "_ZL9realv")])
        with mock.patch.object(cbs, "_run", return_value=raw):
            merged, warnings = cbs.fold_objects([Path("a"), Path("b")], "llvm-readelf")
        self.assertEqual(merged, {"_ZL9realv": 100})
        self.assertEqual(warnings, [])

    def test_kernel_size_disagreement_warns_and_takes_larger(self):
        outputs = [_sym_table([(100, "FUNC", MMQ_MAIN_Q8_J64_FB0)]),
                   _sym_table([(140, "FUNC", MMQ_MAIN_Q8_J64_FB0)])]
        with mock.patch.object(cbs, "_run", side_effect=outputs):
            merged, warnings = cbs.fold_objects([Path("a"), Path("b")], "llvm-readelf")
        self.assertEqual(merged, {MMQ_MAIN_Q8_J64_FB0: 140})
        self.assertEqual(len(warnings), 1)
        self.assertIn("size varies across 2 code objects", warnings[0])

    def test_shared_helper_disagreement_is_not_warned_about(self):
        # A real library emits `no_device_code` into dozens of TUs at slightly
        # different sizes. That says nothing about any candidate, and warning
        # per pair produced ~70 near-identical lines on the real build.
        helper = "_ZL14no_device_codePKciS0_iS0_"
        outputs = [_sym_table([(12688, "FUNC", helper)]),
                   _sym_table([(12816, "FUNC", helper)])]
        with mock.patch.object(cbs, "_run", side_effect=outputs):
            merged, warnings = cbs.fold_objects([Path("a"), Path("b")], "llvm-readelf")
        self.assertEqual(merged, {helper: 12816})
        self.assertEqual(warnings, [])

    def test_one_warning_per_symbol_not_per_object_pair(self):
        outputs = [_sym_table([(100 + i, "FUNC", MMQ_MAIN_Q8_J64_FB0)])
                   for i in range(6)]
        with mock.patch.object(cbs, "_run", side_effect=outputs):
            _, warnings = cbs.fold_objects([Path(str(i)) for i in range(6)],
                                           "llvm-readelf")
        self.assertEqual(len(warnings), 1)
        self.assertIn("100-105 B", warnings[0])


class ArchAnalysisTests(unittest.TestCase):
    def _analyse(self, entries, candidates, architecture="gfx1100"):
        with mock.patch.object(cbs, "_run", return_value=_sym_table(entries)):
            return cbs.analyse_architecture(
                architecture, [Path("obj")], candidates, "llvm-readelf")

    def test_candidate_size_sums_main_and_fixup_kernels(self):
        result = self._analyse(
            [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0),
             (2048, "FUNC", MMQ_FIXUP_Q8_J64_FB0)],
            [_mmq_candidate("mmq:q8_0:j64:fb0:v1", "q8_0", 64, False, ["gfx1100"])])
        entry = result.candidates["mmq:q8_0:j64:fb0:v1"]
        self.assertEqual(entry["text_bytes"], 32464)
        self.assertEqual(entry["kernels"],
                         {"mul_mat_q": 30416, "mul_mat_q_stream_k_fixup": 2048})

    def test_candidate_for_other_architecture_is_not_analysed(self):
        result = self._analyse(
            [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0)],
            [_mmq_candidate("mmq:q8_0:j64:fb0:v1", "q8_0", 64, False, ["gfx1030"])])
        self.assertEqual(result.candidates, {})
        self.assertEqual(result.unresolved_candidates, [])

    def test_candidate_with_no_symbol_is_unresolved_not_silently_zero(self):
        result = self._analyse(
            [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0)],
            [_mmq_candidate("mmq:q8_0:j999:fb0:v1", "q8_0", 999, False, ["gfx1100"])])
        self.assertEqual(result.unresolved_candidates, ["mmq:q8_0:j999:fb0:v1"])
        self.assertNotIn("mmq:q8_0:j999:fb0:v1", result.candidates)

    def test_symbol_with_no_candidate_is_reported_as_unmapped(self):
        result = self._analyse(
            [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0),
             (40168, "FUNC", MMQ_MAIN_Q2K_J80_FB0)],
            [_mmq_candidate("mmq:q8_0:j64:fb0:v1", "q8_0", 64, False, ["gfx1100"])])
        self.assertEqual(result.unmapped_symbols, [MMQ_MAIN_Q2K_J80_FB0])

    def test_native_wrapper_has_no_instantiation_and_is_not_unresolved(self):
        native = {"stable_name": "mmq:native:v1", "family": "mmq",
                  "source_class": "native_wrapper", "architectures": ["gfx1100"],
                  "config": {"policy": "upstream"}}
        result = self._analyse([(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0)], [native])
        self.assertEqual(result.unresolved_candidates, [])
        self.assertEqual(result.candidates, {})

    def test_unmapped_family_is_skipped_entirely(self):
        mmvq = {"stable_name": "mmvq:q8_0:w4:nw4:rpb2:v1", "family": "mmvq",
                "source_class": "existing_runtime", "architectures": ["gfx1100"],
                "config": {"type": "q8_0", "width": 4, "nwarps": 4}}
        result = self._analyse([(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0)], [mmvq])
        self.assertEqual(result.candidates, {})
        self.assertEqual(result.unresolved_candidates, [])

    def test_mmvq_sums_across_has_fusion_since_it_is_not_in_candidate_config(self):
        # has_fusion is compiled as two distinct symbols but is not part of a
        # candidate's config -- both must be counted as one candidate's
        # footprint, the same way MMQ's main+fixup kernels are summed.
        fused = MMVQ_Q8_W1_NW1_RPB2_SK0.replace("ELi1ELb0ELb0ELi1ELi2EE",
                                                 "ELi1ELb1ELb0ELi1ELi2EE")
        candidate = {"stable_name": "mmvq:q8_0:w1:nw1:rpb2:sk0:v1", "family": "mmvq",
                    "source_class": "new_generated_variant", "architectures": ["gfx1100"],
                    "config": {"type": "q8_0", "width": 1, "nwarps": 1,
                              "rows_per_block": 2, "small_k": False}}
        result = self._analyse(
            [(1000, "FUNC", MMVQ_Q8_W1_NW1_RPB2_SK0), (1500, "FUNC", fused)],
            [candidate])
        entry = result.candidates["mmvq:q8_0:w1:nw1:rpb2:sk0:v1"]
        self.assertEqual(entry["text_bytes"], 2500)
        self.assertEqual(entry["kernels"], {"mul_mat_vec_q": 2500})

    def test_mmq_and_mmvq_keys_never_collide_in_the_shared_dict(self):
        # Both families are keyed in the same by_family_key dict; the (family,
        # key) tuple must keep an MmqKey and an MmvqKey with coincidentally
        # equal-looking fields from resolving to the same bucket.
        mmq_candidate = _mmq_candidate("mmq:q8_0:j64:fb0:v1", "q8_0", 64, False, ["gfx1100"])
        mmvq_candidate = {"stable_name": "mmvq:q8_0:w1:nw1:rpb2:sk0:v1", "family": "mmvq",
                          "source_class": "new_generated_variant", "architectures": ["gfx1100"],
                          "config": {"type": "q8_0", "width": 1, "nwarps": 1,
                                    "rows_per_block": 2, "small_k": False}}
        result = self._analyse(
            [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0),
             (1000, "FUNC", MMVQ_Q8_W1_NW1_RPB2_SK0)],
            [mmq_candidate, mmvq_candidate])
        self.assertEqual(result.candidates["mmq:q8_0:j64:fb0:v1"]["text_bytes"], 30416)
        self.assertEqual(result.candidates["mmvq:q8_0:w1:nw1:rpb2:sk0:v1"]["text_bytes"], 1000)

    def test_symbol_map_shape_matches_resource_report_contract(self):
        result = self._analyse(
            [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0),
             (2048, "FUNC", MMQ_FIXUP_Q8_J64_FB0)],
            [_mmq_candidate("mmq:q8_0:j64:fb0:v1", "q8_0", 64, False, ["gfx1100"])])
        symbol_map = result.symbol_map()
        self.assertEqual(symbol_map[MMQ_MAIN_Q8_J64_FB0], ["mmq:q8_0:j64:fb0:v1"])
        self.assertTrue(all(isinstance(v, list) for v in symbol_map.values()))


class ReportTests(unittest.TestCase):
    def _fake_extract(self, tmp: Path):
        def extract(library, workdir, objdump):
            del library, objdump
            workdir.mkdir(parents=True, exist_ok=True)
            return {"gfx1100": [tmp / "obj.gfx1100"]}
        return extract

    def test_build_report_and_unresolved_summary(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            library = tmp / "libggml-hip.so"
            library.write_bytes(b"not a real library, only hashed")
            manifest = _manifest([
                _mmq_candidate("mmq:q8_0:j64:fb0:v1", "q8_0", 64, False, ["gfx1100"]),
                _mmq_candidate("mmq:q8_0:j99:fb0:v1", "q8_0", 99, False, ["gfx1100"]),
            ])
            entries = [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0)]
            with mock.patch.object(cbs, "extract_code_objects", self._fake_extract(tmp)), \
                 mock.patch.object(cbs, "_run", return_value=_sym_table(entries)):
                report = cbs.build_report(library, manifest, tmp / "work",
                                          objdump="llvm-objdump", readelf="llvm-readelf")

        self.assertEqual(report["artifact_version"], cbs.ARTIFACT_VERSION)
        self.assertEqual(report["mapped_families"], ["mmq", "mmvq", "mmvf", "mmf"])
        self.assertEqual(len(report["library_sha256"]), 64)
        arch = report["architectures"]["gfx1100"]
        self.assertEqual(arch["candidates"]["mmq:q8_0:j64:fb0:v1"]["text_bytes"], 30416)
        problems = cbs.unresolved_summary(report)
        self.assertEqual(len(problems), 1)
        self.assertIn("mmq:q8_0:j99:fb0:v1", problems[0])
        # The report must round-trip: it is written to disk verbatim.
        json.dumps(report)

    def test_main_refuses_to_emit_a_partial_table(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            library = tmp / "libggml-hip.so"
            library.write_bytes(b"stub")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest([
                _mmq_candidate("mmq:q8_0:j99:fb0:v1", "q8_0", 99, False, ["gfx1100"]),
            ])), encoding="utf-8")
            entries = [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0)]
            argv = [str(library), "--manifest", str(manifest_path),
                    "--output", str(tmp / "out.json")]
            with mock.patch.object(cbs, "extract_code_objects", self._fake_extract(tmp)), \
                 mock.patch.object(cbs, "_run", return_value=_sym_table(entries)), \
                 mock.patch.object(cbs, "find_tool", return_value="stub-tool"):
                with self.assertRaises(cbs.BinarySizeError):
                    cbs.main(argv)
                # ...but --allow-unresolved is an explicit, diagnostic opt-out.
                self.assertEqual(cbs.main(argv + ["--allow-unresolved"]), 0)

    def test_main_writes_per_architecture_symbol_maps(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            library = tmp / "libggml-hip.so"
            library.write_bytes(b"stub")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest([
                _mmq_candidate("mmq:q8_0:j64:fb0:v1", "q8_0", 64, False, ["gfx1100"]),
            ])), encoding="utf-8")
            entries = [(30416, "FUNC", MMQ_MAIN_Q8_J64_FB0)]
            with mock.patch.object(cbs, "extract_code_objects", self._fake_extract(tmp)), \
                 mock.patch.object(cbs, "_run", return_value=_sym_table(entries)), \
                 mock.patch.object(cbs, "find_tool", return_value="stub-tool"):
                cbs.main([str(library), "--manifest", str(manifest_path),
                          "--output", str(tmp / "out.json"),
                          "--symbol-map-dir", str(tmp / "maps")])
            written = json.loads((tmp / "maps" / "symbol-map-gfx1100.json")
                                 .read_text(encoding="utf-8"))
        self.assertEqual(written[MMQ_MAIN_Q8_J64_FB0], ["mmq:q8_0:j64:fb0:v1"])


class BundleNameTests(unittest.TestCase):
    def test_recognises_real_offload_bundle_names(self):
        real = "libggml-hip.so.0.18.0.7.hipv4-amdgcn-amd-amdhsa--gfx1201"
        match = cbs._BUNDLE_SUFFIX.search(real)
        self.assertIsNotNone(match)
        self.assertEqual(match.group("arch"), "gfx1201")

    def test_host_bundle_is_not_an_architecture(self):
        host = "libggml-hip.so.0.18.0.7.host-x86_64-unknown-linux-gnu-"
        self.assertIsNone(cbs._BUNDLE_SUFFIX.search(host))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
