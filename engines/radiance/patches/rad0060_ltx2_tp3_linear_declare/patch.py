"""Diagnostic LTX C01-C06: reuse native linear, collective, norm and attention operations.

Targets the frozen Radiance 2.2 ABI44 candidate, never production v1.17.
TP1 callers pass no C01Slice and keep their exact original code path.
No real TP3 launch is enabled: the upstream world_size>1 refusal remains.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from bigcherry.patcher import Edit, FilePatch

_SRC = Path(__file__).resolve().parent / "src"
_SHA256 = {
    "bc_ltx_tp3_c01.h": "5bf943238a78644f642c5d00c08d6d14721255a13e22ea9921a664a0de5f6b65",
    "preflight.inc": "4d8d82fea9ddef9fc07d158d05104cc47005a6b5ea7dbcd22d05d1148f730596",
    "bias.inc": "50f1020c9764408aa45884f7024243598cd41712f5e91b38e33d9262811c2392",
    "linear.inc": "4c378fb782fb7bb58185273baa29a1c73f07f7eb86b86038b08236fd482c10f3",
    "attn_state.inc": "8139f6349178f6d5612fb3f4e9d61f7685d1fb4b5bfeaf4f24fd04d46673fdcb",
    "attn_geometry.inc": "a1df6c4946a64da8b3d1e8a459de338685db4d86d1c4847194d200ca209ec7e5",
    "attn_output.inc": "8159c4dfd436138b16db74e1cc25a6f9ae86408e22a3df0588e9b80f7309b965",
    "attn_norm.inc": "2758c87db3c30112f955b3db7fc82ce0d35dd4510cdeec09f32a39a6523fba83",
    "attn_rope.inc": "d4a9fdacfee3c5752b26839bcc2eccd063d70baf5b2145b3be56553018cbc7ff",
    "attn_core.inc": "4a52ed788b8204bede7fd898cd9b17a815d42397aef2df89b4c45a5d45a88835",
    "attn_cross.inc": "6f51439e1ba144957526a4e41942e8bd498f55dda83905dbbf518354ba54297f",
    "global_rms.inc": "18bc49f405708bd19151def2054ba626bdb808ff11924c4d049afbd6ac122bbc",
    "post_reduce.inc": "c5e4ddc2c42bdefbb5c72225ea85270712f3b8625817d61116e40231c5ee6808",
}


def _source(filename: str) -> str:
    data = (_SRC / filename).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != _SHA256[filename]:
        raise RuntimeError(f"LT3 C01 {filename} changed without digest update")
    return data.decode("utf8")


def _edit(label: str, before: str, after: str, marker: str) -> Edit:
    return Edit(
        id=label,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(marker),
        expect_matches=1,
        max_span_lines=max(4, before.count("\n") + 2),
        rationale="LT3 native IQ8R builder-only staging: fail closed if frozen ABI44 source changes",
    )


PATCHES = (
    FilePatch(
        path="arch/ltx2/bc_ltx_tp3_c01.h",
        create=True,
        language="none",
        description="Native C01 LTX2 TP3 IQ8R shard admission geometry",
        edits=(Edit(
            id="ltx-c01-shard-header",
            anchor=r"\A",
            mode="insert_after",
            text=_source("bc_ltx_tp3_c01.h"),
            guard=r"BC_LTX3_C01_HEADER",
            expect_matches=1,
            max_span_lines=1,
            rationale="Create a new LTX-only C01 header; never replace an upstream file",
        ),),
    ),
    FilePatch(
        path="arch/ltx2/ltx2.cpp",
        language="none",
        description="Opt-in LTX Lin::declare sharding, all TP1 call sites unchanged",
        edits=(
            _edit(
                'ltx-c06-attn-state',
                'struct Attn {\n',
                "struct Attn {\n" + _source("attn_state.inc"),
                'BC_LTX3_C06_ATTN_STATE',
            ),
            _edit(
                'ltx-c06-attn-argument',
                'int acc, int group, const Rope& rope, double eps) {\n',
                "int acc, int group, const Rope& rope, double eps,\n"
                "                        const C01Slice* tp = nullptr, rad_buf rms_scratch = 0) {\n",
                'const C01Slice* tp = nullptr',
            ),
            _edit(
                'ltx-c06-attn-geometry',
                '    A.H = H;\n    A.hd = hd;\n    A.inner = H * hd;\n    const int64_t in = A.inner;\n',
                _source("attn_geometry.inc"),
                'BC_LTX3_C06_ATTN_GEOMETRY',
            ),
            _edit(
                'ltx-c06-qkv-shard',
                '                                   { bs("to_q"), bs("to_k"), bs("to_v") }, { in, in, in }, 3 * in,\n                                   qdim, acc, group, xin, bq, rows_q));\n',
                "                                   { bs(\"to_q\"), bs(\"to_k\"), bs(\"to_v\") }, { full, full, full }, 3 * in,\n"
                "                                   qdim, acc, group, xin, bq, rows_q, false, tp)); /* BC_LTX3_C06_QKV */\n",
                'BC_LTX3_C06_QKV',
            ),
            _edit(
                'ltx03-cross-qkv-shard',
                '    else {\n        RAD_ARCH_TRY(A.qkv.declare(b, nm, g, nm.f("%s.q", dn), { w("to_q") }, { bs("to_q") }, {}, in,\n                                   qdim, acc, group, xin, bq, rows_q));\n        RAD_ARCH_TRY(A.kv.declare(b, nm, g, nm.f("%s.kv", dn), { w("to_k"), w("to_v") },\n                                  { bs("to_k"), bs("to_v") }, { in, in }, 2 * in, kvdim, acc, group,\n                                  *kvin, bkv, rows_k));\n    }\n',
                _source('attn_cross.inc'),
                'BC_LTX3_LTX03_CROSS_KV',
            ),
            _edit(
                'ltx-c06-out-reduce',
                '    RAD_ARCH_TRY(A.out.declare(b, nm, g, nm.f("%s.out", dn), { w("to_out.0") }, { bs("to_out.0") }, {},\n                               odim, in, acc, group, ao, y, rows_q));\n',
                _source("attn_output.inc"),
                'BC_LTX3_C06_ATTN_OUTPUT',
            ),
            _edit(
                'ltx-c06-global-norm',
                '    A.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), w("norm_q"), RAD_BF16, { in }, acc, group);\n    A.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), w("norm_k"), RAD_BF16, { in }, acc, group);\n    if (!A.w_qn || !A.w_kn) return RAD_E_INVAL;\n    const rad_buf kb = self ? bq : bkv;\n    A.op_qn = rms_op(b, rows_q, in, eps, A.w_qn, bq, bq);\n    A.op_kn = rms_op(b, rows_k, in, eps, A.w_kn, kb, kb);\n',
                "    A.w_qn = wcopy(b, nm.f(\"%s.q_norm.weight\", dn), w(\"norm_q\"), RAD_BF16, { full }, acc, group);\n"
                "    A.w_kn = wcopy(b, nm.f(\"%s.k_norm.weight\", dn), w(\"norm_k\"), RAD_BF16, { full }, acc, group);\n"
                "    if (!A.w_qn || !A.w_kn) return RAD_E_INVAL;\n"
                "    const rad_buf kb = self ? bq : bkv;\n"
                + _source("attn_norm.inc"),
                'BC_LTX3_C06_ATTN_NORM',
            ),
            _edit(
                'ltx-c06-local-rope',
                '        auto rop = [&](int64_t rows, rad_buf x) {\n            return rw(b, RAD_OP(b, "rope_split",\n                                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_head", H),\n                                           RAD_INT("head_dim", hd), RAD_INT("dim", rope.dim),\n                                           RAD_INT("n_dims", rope.n_dims), RAD_F64("theta", 10000.0),\n                                           RAD_STR("max", rope.max), RAD_STR("dtype", "bf16")),\n                                RAD_NOWEIGHTS),\n                      { x }, { x });\n        };\n',
                _source("attn_rope.inc"),
                'BC_LTX3_C06_ATTN_ROPE',
            ),
            _edit(
                'ltx-c06-gate-heads',
                'RAD_PARAMS(RAD_RANGE("M", 1, rows_q), RAD_INT("n_head", H),\n                                      RAD_INT("head_dim", hd), RAD_STR("dtype", "bf16")),\n',
                "RAD_PARAMS(RAD_RANGE(\"M\", 1, rows_q), RAD_INT(\"n_head\", A.H),\n"
                "                                      RAD_INT(\"head_dim\", hd), RAD_STR(\"dtype\", \"bf16\")), /* BC_LTX3_C06_GATE */\n",
                'BC_LTX3_C06_GATE',
            ),
            _edit(
                'ltx-c06-attn-return',
                '    return A.op_qn && A.op_kn && A.op_attn && A.op_hg && (!self || A.op_v) ? RAD_OK : RAD_E_INVAL;\n',
                "    return (tp ? A.tp_q.reduce && A.tp_k.reduce : A.op_qn && A.op_kn) &&\n"
                "           A.op_attn && A.op_hg && (!self || A.op_v) ? RAD_OK : RAD_E_INVAL; /* BC_LTX3_C06_RETURN */\n",
                'BC_LTX3_C06_RETURN',
            ),
            _edit(
                'ltx-c06-core-norm',
                '    RAD_ISSUE_N(c, A.op_qn, nq, q, RAD_W(A.w_qn), q);\n    RAD_ISSUE_N(c, A.op_kn, nk, k, RAD_W(A.w_kn), k);\n',
                _source("attn_core.inc"),
                'BC_LTX3_C06_ATTN_CORE',
            ),
            _edit(
                'ltx-c06-gate-issue',
                '    RAD_ISSUE_N(c, A.op_hg, nq, brows(bg, nq), brows(ao.a.x, nq));\n',
                "    RAD_ISSUE_N(c, A.op_hg, nq,\n"
                "                A.tp3 ? bcol(bg, A.head0, A.H, nq) : brows(bg, nq),\n"
                "                brows(ao.a.x, nq)); /* BC_LTX3_C06_GATE_ISSUE */\n",
                'BC_LTX3_C06_GATE_ISSUE',
            ),
            _edit(
                "ltx-c06-stg-gate-issue",
                '                RAD_ISSUE_N(c, At.op_hg, n, brows(bg, n), brows(ao.a.x, n));\n',
                "                RAD_ISSUE_N(c, At.op_hg, n,\n"
                "                            At.tp3 ? bcol(bg, At.head0, At.H, n) : brows(bg, n),\n"
                "                            brows(ao.a.x, n)); /* BC_LTX3_C06_STG_GATE */\n",
                "BC_LTX3_C06_STG_GATE",
            ),
            _edit(
                "ltx-c01-deferred-bias-state",
                "    int64_t    N = 0, K = 0;\n",
                "    int64_t    N = 0, K = 0;\n"
                "    bool deferred_bias = false; /* BC_LTX3_C01_DEFERRED_STATE */\n"
                "    bool deferred_has_bias = false;\n"
                "    rad_op post_reduce = 0; /* BC_LTX3_C04_REDUCE_STATE */\n",
                "BC_LTX3_C01_DEFERRED_STATE",
            ),
            _edit(
                "ltx-c01-include",
                "#include <arch/rad_fp8.h>\n",
                "#include <arch/rad_fp8.h>\n"
                "#include \"bc_ltx_tp3_c01.h\" /* BC_LTX3_C01_INCLUDE */\n",
                "BC_LTX3_C01_INCLUDE",
            ),
            _edit(
                "ltx-c01-optional-argument",
                "int group, const Act& in, rad_buf y, int64_t max_rows, bool small = false) {",
                "int group, const Act& in, rad_buf y, int64_t max_rows, bool small = false,\n"
                "                const C01Slice* slice = nullptr) { /* BC_LTX3_C01_ARGUMENT */",
                "BC_LTX3_C01_ARGUMENT",
            ),
            _edit(
                "ltx-c01-preflight",
                '        K = n_in;\n        const char* wn = nm.f("%s.weight", base);\n',
                "        K = n_in;\n"
                + _source("preflight.inc")
                + '        const char* wn = nm.f("%s.weight", base);\n',
                "BC_LTX3_C01_PREFLIGHT",
            ),
            _edit(
                "ltx-c01-bias",
                "            bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_NONE, grp_layer(group));\n"
                "            if (!bias) return RAD_E_INVAL;\n",
                _source("bias.inc"),
                "BC_LTX3_C01_BIAS",
            ),
            _edit(
                "ltx-c01-weight",
                "        RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_NONE, grp_layer(group), wsrc, in.a, y,\n"
                "                               max_rows, parts));\n",
                _source("linear.inc"),
                "BC_LTX3_C01_LINEAR",
            ),
            _edit(
                "ltx-c05-global-rms",
                "struct Attn {\n",
                _source("global_rms.inc") + "struct Attn {\n",
                "BC_LTX3_C05_GLOBAL_RMS",
            ),
            _edit(
                "ltx-c04-finalize",
                "    /* Input rows [ri, ri + rows) to output rows [ro, ro + rows). */\n",
                _source("post_reduce.inc")
                + "    /* Input rows [ri, ri + rows) to output rows [ro, ro + rows). */\n",
                "BC_LTX3_C04_FINALIZE",
            ),
            _edit(
                "ltx-c04-step-reduce",
                "        if (op_b) RAD_ISSUE_N(c, op_b, rows, out, RAD_W(bias), out);\n",
                "        if (deferred_bias && (!post_reduce || (deferred_has_bias && (!op_b || !bias)))) {\n"
                '            rad_step_fail(c, "LTX TP3 deferred projection was not finalized");\n'
                "            return;\n"
                "        }\n"
                "        if (deferred_bias)\n"
                "            RAD_ISSUE_N(c, post_reduce, rows * N, out, RAD_NONE); /* BC_LTX3_C04_ISSUE */\n"
                "        if (op_b) RAD_ISSUE_N(c, op_b, rows, out, RAD_W(bias), out);\n",
                "BC_LTX3_C04_ISSUE",
            ),
        ),
    ),
)
