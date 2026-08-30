"""HI119: a new registered test_case, test_bigcherry_moe_glu_fusion, that
builds the REAL fused MUL_MAT_ID(gate) + MUL_MAT_ID(up) + GLU subgraph a
production MoE FFN produces, so BigCherry's correctness-evidence tooling
can generate real evidence for the fused dispatches HI108 found stuck at
rejected_no_correctness_evidence.

Context (full trail in docs/planning/completed/hip-autotune/HI108.md,
HI118.md, and the active HI119.md): HI80/HI105's existing mapper pattern
(signature_to_test_file_line() -> test-backend-ops' --test-file/
test_generic_op escape hatch) can only build ONE op per test case
(make_test_cases_from_file(), tests/test-backend-ops.cpp). Every GLU-op
dispatch signature this tuner records is a FUSED MUL_MAT_ID+GLU epilogue
(confirmed by reading ggml_hip_dispatch_mul_mat/family and the real
ggml_cuda_can_fuse() graph-fusion scanner in ggml-cuda.cu), so no single-op
mapper can represent it. This patch adds a genuine multi-node registered
test_case instead -- the only mechanism that can trigger ggml-cuda's own
existing fusion detector during ordinary backend graph compute.

dev-gpt-agent deep design review (2026-08-25, verified against real source
before being accepted -- see HI119.md's detailed_solution for the full
verification trail):

- test_mul_mat_vec_fusion (this same file) already builds almost exactly
  this shape in its use_id=true path, but adds an unconditional post-GLU
  MUL, so its GLU is never the terminal graph output there. This new class
  is a purpose-built sibling with the GLU (or swiglu_oai) as the terminal,
  stably-named output -- so the correctness metric measures the real fused
  dispatch's own actual output, with nothing after it.
- ggml_cuda_should_fuse_mul_mat/ggml_can_fuse_subgraph (ggml-cuda.cu)
  require gate/up to share the EXACT SAME activation (`cur`) and ids tensor
  OBJECTS (compared by pointer, not merely same-shape separate tensors),
  and each MUL_MAT_ID's result to have exactly ONE consumer (the GLU) --
  this class builds `cur`/`ids` once and passes the same variables to both
  ggml_mul_mat_id() calls, with nothing else referencing gate_mm/up_mm, to
  satisfy both requirements by construction.
- Requires patch 1238 (deterministic init_mul_mat_id_tensors()) for the
  expert-routing determinism this class's initialize_tensors() depends on
  -- without it, native/candidate forced-runs of this class would route to
  different experts despite identical BIGCHERRY_TEST_DETERMINISTIC_SEED
  float inputs.

Scope this slice: the SIMPLE fusion pattern only (gate/up weights, no
bias, no scale) -- confirmed sufficient for both of HI108's real blocked
dispatches via direct inspection of the real Qwen3.6-35B-A3B GGUF (only
.weight tensors exist for ffn_gate_exps/ffn_up_exps, no .bias/.scale
anywhere). The biased/scaled fusion variants are explicitly out of scope
for this patch; a future patch can add them following this same pattern
if a real blocked dispatch is ever found needing them.

Does NOT yet add: the mandatory observed-dispatch-signature-digest
evidence gate the review flagged as required before this class's output
can be trusted as real fused-dispatch evidence (a future HI119 CLI-side
patch/tooling change, not part of this test_case itself -- that gate
belongs in the Python evidence producer that drives this class, reading
BigCherry's own dispatch signature output alongside this class's
BIGCHERRY_REF_DIGEST/CORRECTNESS_METRIC lines)."""

GROUP = "core"
# Verified offline: dry-run apply + idempotence against the real vendored
# checkout (composed with 1222+1236+1238). Compiled cleanly on real Brutus
# hardware (bc-build-hi64hi14) and confirmed the new test case is enumerated
# by test-backend-ops' own case registry and runs without crashing under a
# direct `-p` filter (real GPU, gfx1100/gfx1201/gfx1030). NOT yet promoted to
# "validated" -- that needs a real end-to-end correctness-evidence run
# (BIGCHERRY_TEST_DETERMINISTIC_SEED set, native vs candidate forced
# comparison) against one of HI108's real blocked dispatches, plus the
# observed-signature-digest evidence gate HI119 still needs to add on the
# Python side before this class's output can be trusted as real fused-
# dispatch correctness evidence.
STATE = "untested"

REQUIRES = (
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1238_hi119_deterministic_init_mul_mat_id_tensors",
)

import re as _re

from bigcherry.patcher import Edit, FilePatch, csource as _csource

_NEW_STRUCT = '''
// bigcherry (HI119): real fused MUL_MAT_ID(gate) + MUL_MAT_ID(up) + GLU
// subgraph -- see patches/1239_hi119_fused_moe_glu_test_case/patch.py. The
// EXACT adjacent-node shape ggml_cuda_can_fuse() (ggml-cuda.cu) scans for,
// so running this through the ordinary backend graph-compute path triggers
// the SAME fusion detector production MoE FFN dispatches use, rather than
// faking a fused dispatch signature by hand.
struct test_bigcherry_moe_glu_fusion : public test_case {
    const ggml_type type;
    const ggml_glu_op glu_op;
    const int64_t k;
    const int64_t n;
    const int64_t m;
    const int n_mats;
    const int n_used;
    const bool b; // broadcast: cur's middle dim is 1 (broadcast to every
                  // selected expert) rather than n_used -- ggml_mul_mat_id's
                  // real "ne1[1] broadcastable up to n_expert_used" rule.
                  // HI108's real routed dispatch (7ef2471585a5aa6fbb49384ef
                  // e566ac5) uses the broadcast form (ne1=[k,1,1,1]), the
                  // real up/gate-projection shape -- confirmed against the
                  // plan item's own recorded ne1, not assumed.

    std::string vars() override {
        return VARS_TO_STR8(type, glu_op, k, n, m, n_mats, n_used, b);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "BIGCHERRY_MOE_GLU_FUSION";
    }

    bool run_whole_graph() override { return true; }

    double max_nmse_err() override {
        return 5e-3;
    }

    test_bigcherry_moe_glu_fusion(ggml_type type = GGML_TYPE_F32,
            ggml_glu_op glu_op = GGML_GLU_OP_SWIGLU,
            int64_t k = 256, int64_t n = 32, int64_t m = 4,
            int n_mats = 8, int n_used = 2, bool b = true)
        : type(type), glu_op(glu_op), k(k), n(n), m(m), n_mats(n_mats), n_used(n_used), b(b) {
        GGML_ASSERT(n_used <= n_mats);
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * gate_w = ggml_new_tensor_3d(ctx, type, k, n, n_mats);
        ggml_set_name(gate_w, "gate_w");
        ggml_tensor * up_w = ggml_new_tensor_3d(ctx, type, k, n, n_mats);
        ggml_set_name(up_w, "up_w");

        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, m);
        ggml_set_name(ids, "ids");
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, m, ids->nb[1], 0);
            ggml_set_name(ids, "view_of_ids");
        }

        ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, this->b ? 1 : n_used, m);
        ggml_set_name(cur, "cur");

        // gate/up MUST share the exact same cur/ids tensor OBJECTS (not
        // merely same-shape separate tensors) -- ggml_cuda_should_fuse_
        // mul_mat compares ffn_up->src[1]/src[2] against ffn_gate->src[1]/
        // src[2] by pointer.
        ggml_tensor * gate_mm = ggml_mul_mat_id(ctx, gate_w, cur, ids);
        ggml_set_name(gate_mm, "gate_mm");
        ggml_tensor * up_mm = ggml_mul_mat_id(ctx, up_w, cur, ids);
        ggml_set_name(up_mm, "up_mm");

        // gate_mm/up_mm must each have exactly ONE consumer (this GLU) for
        // ggml_can_fuse_subgraph's elidability check to allow fusion --
        // nothing else may reference them.
        ggml_tensor * out = (glu_op == GGML_GLU_OP_SWIGLU_OAI)
            ? ggml_swiglu_oai(ctx, gate_mm, up_mm, 1.702f, 7.0f)
            : ggml_glu_split(ctx, gate_mm, up_mm, glu_op);
        ggml_set_name(out, "fused_glu");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        init_mul_mat_id_tensors(ctx, n_mats);
    }
};
'''

_REGISTRATION_ANCHOR_SOURCE = '''    test_cases.emplace_back(new test_opt_step_adamw(GGML_TYPE_F32, {10, 5, 4, 3}));
    test_cases.emplace_back(new test_opt_step_sgd(GGML_TYPE_F32, {10, 5, 4, 3}));'''

_REGISTRATION = '''    test_cases.emplace_back(new test_opt_step_adamw(GGML_TYPE_F32, {10, 5, 4, 3}));
    test_cases.emplace_back(new test_opt_step_sgd(GGML_TYPE_F32, {10, 5, 4, 3}));

    // bigcherry (HI119): real fused MUL_MAT_ID(gate)+MUL_MAT_ID(up)+GLU
    // shapes -- see patches/1239_hi119_fused_moe_glu_test_case/patch.py. The
    // Q8_0/k=2048/n=256/n_mats=256/n_used=8/b=true/SWIGLU case matches
    // HI108's real routed blocked dispatch
    // (7ef2471585a5aa6fbb49384efe566ac5, Qwen3.6-35B-A3B) exactly,
    // including the broadcast (ne1[1]==1) shape its real ne1 field records
    // -- an earlier non-broadcast (b=false) instance was checked against
    // real hardware and found to NOT match ne1 exactly, fixed here. F32 is
    // a fast correctness sanity baseline, not a real production shape.
    // Registering fixed instances here is a stopgap for HI119's own step 7
    // (parameterize generically from an arbitrary real dispatch signature,
    // not hard-coded shapes) -- tracked as still open in
    // docs/planning/active/hip-autotune/HI119.md.
    for (ggml_glu_op glu_op : {GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU}) {
        test_cases.emplace_back(new test_bigcherry_moe_glu_fusion(
            GGML_TYPE_Q8_0, glu_op, /*k=*/2048, /*n=*/256, /*m=*/1, /*n_mats=*/256, /*n_used=*/8, /*b=*/true));
        test_cases.emplace_back(new test_bigcherry_moe_glu_fusion(
            GGML_TYPE_F32, glu_op, /*k=*/256, /*n=*/32, /*m=*/2, /*n_mats=*/8, /*n_used=*/2, /*b=*/false));
    }'''

_REGISTRATION_ANCHOR = _re.escape(_csource.strip_noise(_REGISTRATION_ANCHOR_SOURCE, "c"))

_ANCHOR_SOURCE = '''    double max_nmse_err() override {
        return 5e-3;
    }
};

// GGML_OP_SUM'''

_ANCHOR = _re.escape(_csource.strip_noise(_ANCHOR_SOURCE, "c"))

PATCH = FilePatch(
    path="tests/test-backend-ops.cpp",
    description="new registered test_bigcherry_moe_glu_fusion test_case: a real, "
                "terminal-GLU MUL_MAT_ID(gate)+MUL_MAT_ID(up)+GLU subgraph that "
                "triggers ggml-cuda's own graph-fusion detector, for HI119's "
                "fused-dispatch correctness-evidence harness",
    edits=(
        Edit(
            id="hi119-fused-moe-glu-test-case",
            anchor=_ANCHOR,
            mode="replace",
            rationale="insert the new test_case immediately after "
                       "test_mul_mat_vec_fusion's own closing brace, before "
                       "the next op section (GGML_OP_SUM) begins",
            text=r"""    double max_nmse_err() override {
        return 5e-3;
    }
};
""" + _NEW_STRUCT + "\n// GGML_OP_SUM",
            guard=r"struct test_bigcherry_moe_glu_fusion : public test_case",
            max_span_lines=10,
        ),
        Edit(
            id="hi119-fused-moe-glu-registration",
            anchor=_REGISTRATION_ANCHOR,
            mode="replace",
            rationale="register real test_bigcherry_moe_glu_fusion instances in "
                       "make_test_cases() so the new class is actually enumerable/"
                       "runnable via test-backend-ops' own -o/-p filters, not just "
                       "defined",
            text=_REGISTRATION,
            guard=r"bigcherry \(HI119\): real fused MUL_MAT_ID\(gate\)\+MUL_MAT_ID\(up\)\+GLU\n    // shapes",
            max_span_lines=4,
        ),
    ),
)

PATCHES = [PATCH]
