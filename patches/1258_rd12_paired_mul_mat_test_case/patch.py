"""RD12 correctness support: registered paired plain-MUL_MAT test case.

test-backend-ops --test-file/test_generic_op can represent only one op.
RD12's production selector requires two distinct adjacent MUL_MAT nodes
with the same activation tensor and different weight tensors, so a
single-op test-file case cannot exercise patch 1205.

This diagnostic patch adds the minimal whole-graph case needed by the
RD12 Experiment Contract producer. It is applied identically to control
and subject; only 1205 differs between the two arms.

Both projection outputs are explicit graph roots. This is intentional:
using ADD(k_out, v_out) as a terminal liveness node makes the second
matmul eligible for the pre-existing CUDA MUL_MAT+ADD fusion, which
writes directly to the ADD destination and leaves v_out itself
unmaterialized. Explicitly expanding k_out and returning v_out keeps
the two MUL_MAT nodes live, adjacent, and independently verifiable.
"""

GROUP = "core"
STATE = "untested"

REQUIRES = (
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1223_hi67_machine_readable_correctness_metrics",
)

import re as _re

from bigcherry.patcher import Edit, FilePatch, csource as _csource


_NEW_STRUCT = r'''
// bigcherry (RD12 correctness): two distinct quantized MUL_MAT nodes over
// the exact same activation tensor. This is the production graph shape
// patches/1205_rd12_paired_mmvq_dual_output scans for.
//
// Both projections are explicit graph roots. Do NOT join them with an ADD:
// the pre-existing CUDA MUL_MAT+ADD fusion may consume the second MUL_MAT
// directly into the ADD destination, leaving that intermediate output
// unmaterialized and therefore invalid for fusion_test_nodes() verification.
struct test_bigcherry_rd12_paired_mul_mat : public test_case {
    const ggml_type type;
    const int64_t m;
    const int64_t n;
    const int64_t k;

    ggml_tensor * k_out = nullptr;
    ggml_tensor * v_out = nullptr;

    test_bigcherry_rd12_paired_mul_mat(
            ggml_type type = GGML_TYPE_Q6_K,
            int64_t m = 1024,
            int64_t n = 1,
            int64_t k = 2560)
        : type(type), m(m), n(n), k(k) {}

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT";
    }

    std::string vars() override {
        return "bigcherry_rd12=1,type="
            + std::string(ggml_type_name(type))
            + ",m=" + std::to_string(m)
            + ",n=" + std::to_string(n)
            + ",k=" + std::to_string(k);
    }

    bool run_whole_graph() override {
        return true;
    }

    double max_nmse_err() override {
        return 5e-3;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Distinct weight tensor OBJECTS are mandatory: RD12 explicitly
        // rejects a pair whose src[0] pointers are equal.
        ggml_tensor * k_weight = ggml_new_tensor_2d(ctx, type, k, m);
        ggml_set_name(k_weight, "rd12_k_weight");

        ggml_tensor * v_weight = ggml_new_tensor_2d(ctx, type, k, m);
        ggml_set_name(v_weight, "rd12_v_weight");

        // Both matmuls must share this exact src1 tensor object.
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        ggml_set_name(x, "rd12_x");

        k_out = ggml_mul_mat(ctx, k_weight, x);
        ggml_set_name(k_out, "rd12_k_out");

        v_out = ggml_mul_mat(ctx, v_weight, x);
        ggml_set_name(v_out, "rd12_v_out");

        // test_case::eval() has already created gf before build_graph().
        // Add K as the first independent root here. eval() will expand the
        // returned V root afterwards, producing the required graph-node order:
        //
        //     rd12_k_out
        //     rd12_v_out
        //
        // This keeps both outputs materialized while preserving the exact
        // adjacent-MUL_MAT shape required by RD12's production detector.
        //
        // Only touch gf in MODE_TEST; other modes do not need the paired
        // correctness graph and may not have a usable test graph here.
        if (mode == MODE_TEST) {
            ggml_build_forward_expand(gf, k_out);
        }

        return v_out;
    }

    std::vector<ggml_tensor *> fusion_test_nodes() override {
        return { k_out, v_out };
    }
};
'''

_CLASS_ANCHOR = _re.escape(
    "struct test_mul_mat_vec_fusion : public test_case {"
)

_REGISTRATION_ANCHOR_SOURCE = (
    "    test_cases.emplace_back(new test_dsv4_hc_pre(1, 1));"
)
_REGISTRATION_ANCHOR = _re.escape(
    _csource.strip_noise(_REGISTRATION_ANCHOR_SOURCE, "c")
)

_REGISTRATION_REPLACEMENT = r'''    test_cases.emplace_back(new test_dsv4_hc_pre(1, 1));

    // bigcherry (RD12 correctness): contract-tier Q6_K decode K/V-like
    // projection pair. n=1 deliberately selects the decode/MMVQ path.
    test_cases.emplace_back(new test_bigcherry_rd12_paired_mul_mat(
        GGML_TYPE_Q6_K,
        /*m=*/1024,
        /*n=*/1,
        /*k=*/2560));'''


PATCH = FilePatch(
    path="tests/test-backend-ops.cpp",
    description=(
        "register a whole-graph paired MUL_MAT correctness case that "
        "exercises RD12's real dual-output MMVQ fusion detector"
    ),
    edits=(
        Edit(
            id="rd12-paired-mul-mat-test-case",
            anchor=_CLASS_ANCHOR,
            mode="insert_before",
            rationale=(
                "a single-op --test-file case cannot represent the two "
                "distinct adjacent MUL_MAT nodes required by RD12"
            ),
            text=_NEW_STRUCT,
            guard=(
                r"struct test_bigcherry_rd12_paired_mul_mat "
                r": public test_case"
            ),
        ),
        Edit(
            id="rd12-paired-mul-mat-register",
            anchor=_REGISTRATION_ANCHOR,
            mode="replace",
            rationale=(
                "register one deterministic Q6_K n=1 whole-graph case "
                "for the RD12 correctness producer"
            ),
            text=_REGISTRATION_REPLACEMENT,
            guard=r"new test_bigcherry_rd12_paired_mul_mat\(",
        ),
    ),
)

PATCHES = [PATCH]
