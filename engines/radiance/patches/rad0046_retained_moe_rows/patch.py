"""Retain the E4M3 forms consumed by the Flash-Next and 35B helpers."""
import re

from bigcherry.patcher import Edit, FilePatch


def before(edit_id: str, line: str, text: str, rationale: str) -> Edit:
    return Edit(id=edit_id, anchor="^" + re.escape(line), mode="insert_before",
                text=text + "\n", guard="^" + re.escape(text.splitlines()[0]),
                expect_matches=1, rationale=rationale)


PATCHES = (
    FilePatch(language="none", path="libr4d/r4d_rows.cpp", description="Register the retained E4M3 rows.", edits=(
        before("rotated-schema", '    SCHEMA("quant_act_fp8",',
               '''    SCHEMA("had_quant_act_fp8", pQuantActFp8, oQuantActFp8,
           "128-wide Hadamard before per-block E4M3 quantisation, paired with rotated W4 experts."),''',
               "The rotated and plain buffers have different values and distinct op names."),
        before("rotated-row", '{ "quant_act_fp8", "quant_act_fp8", "quant",',
               '''{ "had_quant_act_fp8", "had_quant_act_fp8", "quant",
  "128-wide Hadamard followed by symmetric E4M3 quantisation",
  "n a multiple of 128, group 128", "bf16 input, row-major E4M3 codes and f32 scales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cHadQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_had_quant_act_fp8, nullptr, nullptr, nullptr,
  r4d_shape_quant_act_fp8 },
''', "Use the existing rotation constraint and quantiser operand description."),
        before("grouped-row", '{ "moe_gemm_i8a8", "moe_gemm_q", "moe",',
               '''{ "moe_gemm_fp8a8", "moe_gemm_q", "moe",
  "grouped block-scaled E4M3 expert GEMM over expert-sorted rows",
  "Each class K divisible by 128; N divisible by 64 times parts",
  "E4M3 activations with one f32 scale per row per 128 K values and E4M3 expert weights with bf16 scales per 128x128 tile; bf16 output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMoeGemmQ), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_q, r4d_rad_moe_gemm_q_scratch,
  r4d_rad_layout_moe_fp8, nullptr, r4d_shape_moe_gemm_q },
''', "Restore the exact op, row and plane contract looked up by the two local components."),
    )),
    FilePatch(language="none", path="libr4d/r4d_quant_act_fp8.hip", description="Expose the rotated E4M3 quantiser.", edits=(
        before("rotated-launch", 'extern "C" int r4d_rad_quant_act_fp8(',
               '''extern "C" int r4d_rad_had_quant_act_fp8(const RadArgs* a, RadStream stream) {
    return rad_quant_act_fp8(a, stream, true);
}
''', "The current templated quantiser retains the E4M3 rotation."),
    )),
    FilePatch(language="none", path="libr4d/r4d_moe.hip", description="Expose the original block-scaled GEMM form.", edits=(
        before("grouped-launch", 'extern "C" int r4d_rad_moe_gemm_w4a8(',
               '''extern "C" int r4d_rad_moe_gemm_q(const RadArgs* a, RadStream stream) {
    return moe_gemm_launch<0>(a, stream);
}
''', "W4=0 and I8=false read block-scaled E4M3 expert weights."),
        Edit(
            id="fp8-wide-tile-all-classes",
            anchor=(r"        /\* The 128-column tile where no column tile straddles two parts of the weight\. \*/"
                    r"\n        const bool wide_n = \(N / parts\) % kPrefillCols == 0;"),
            mode="replace",
            text="""        /* The 128-column tile where no column tile straddles two parts of the weight. */
        bool wide_n = (N / parts) % kPrefillCols == 0;
        /* Block-E4M3 carries one scale per 128x128 tile. Each class and each part
         * must start on that column grid before the wide kernel may reuse a scale. */
        if constexpr (W4 == 0 && !I8)
            for (int q = 0; q < ncls; ++q)
                wide_n = wide_n && Nc[q] % parts == 0 && (Nc[q] / parts) % kPrefillCols == 0;""",
            guard=r"Block-E4M3 carries one scale per 128x128 tile",
            expect_matches=1,
            rationale="An unequal expert class can start its part at row 192; a 128-column block would cross the next scale boundary at row 256.",
        ),
    )),
    FilePatch(language="none", path="libr4d/r4d_shapes.cpp", description="Describe grouped E4M3 operands.", edits=(
        before("grouped-shape", 'extern "C" int r4d_shape_moe_gemm_i8(',
               '''extern "C" int r4d_shape_moe_gemm_q(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return moe_gemm_q_shape(p, n_p, operand, out, false);
}
''', "The retained branch describes E4M3 codes and one scale per 128 by 128 elements."),
    )),
    FilePatch(language="none", path="libr4d/r4d_layout.cpp", description="Check each MoE class at its own N and K.", edits=(
        before("grouped-layout", 'extern "C" int r4d_rad_layout_moe_i8(',
               '''extern "C" int r4d_rad_layout_moe_fp8(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* pl, int n, RadLayout* out) {
    MoeView v;
    const int rc = moe_view(p, n_p, operand, &v);
    return rc != RAD_OK ? rc : r4d_rad_layout_fp8_rowmajor(v.p, v.n, v.operand, enc, sel, pl, n, out);
}
''', "Translate the odd class operands through the same view as the int8 form."),
    )),
    FilePatch(language="none", path="libr4d/r4d_fp8_layout.cpp", description="Validate the canonical E4M3 planes.", edits=(
        before("rowmajor-layout", 'extern "C" int r4d_rad_layout_i8_rowmajor(',
               '''extern "C" int r4d_rad_layout_fp8_rowmajor(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    FGeom g;
    const int rc = fgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (operand != FL_B && operand != FL_BSCALE) return RAD_E_UNSUPPORTED;
    if (!fp8_block_enc(enc)) return RAD_E_DTYPE;
    const bool scale = operand == FL_BSCALE;
    if (!r4d_enc::selects(enc, sel, n, scale ? "scale" : "codes") ||
        !r4d_enc::dims(pl[0], scale ? g.sn : g.N, scale ? g.sk : g.K)) return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED; // Verified canonical plane; no relayout.
}
''', "Use upstream geometry and encoding checks for the checkpoint-native plane contract."),
    )),
    FilePatch(language="none", path="libr4d/r4d_plugin.h", description="Declare the retained entry points.", edits=(
        before("entry-declarations", 'int r4d_rad_layout_moe_i8(',
               '''int r4d_rad_moe_gemm_q(const RadArgs*, RadStream);
int r4d_rad_layout_moe_fp8(const RadParam*, int, int, const RadEncoding*, const int*,
                           const RadTensor*, int, RadLayout*);
int r4d_rad_layout_fp8_rowmajor(const RadParam*, int, int, const RadEncoding*, const int*,
                                const RadTensor*, int, RadLayout*);''',
               "The current SDK already declares the rotated launch and grouped shape callback."),
    )),
)
