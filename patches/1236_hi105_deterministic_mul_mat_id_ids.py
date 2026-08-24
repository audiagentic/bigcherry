"""HI105: deterministic, full-expert-range routing for test_generic_op's
GGML_OP_MUL_MAT_ID initializer.

HI105 extends HI80's CPU-reference correctness-evidence tooling to
MUL_MAT_ID (MoE-routed) signatures via test-backend-ops' `--test-file` /
`test_generic_op` path, which build_moe_ffn's real dispatch signatures
(RD54's two winners among them) need since they generally don't coincide
with test-backend-ops' fixed registered corpus (the same real-hardware
finding that motivated HI80's test-file path for plain MUL_MAT).

Found (dev-gpt-agent review, 2026-08-24, verified against this exact
vendored source before being accepted) that test_generic_op's own
GGML_OP_MUL_MAT_ID branch has two real gaps that would silently invalidate
HI105's correctness comparison, neither covered by patch 1222 (whose own
scope note says "only init_tensor_uniform()"):

1. Non-determinism: test_generic_op::initialize_tensors() constructs its
   own `std::random_device rd; std::default_random_engine rng(rd());` at
   the top of the function, used by every special-cased I32/I64 branch
   including MUL_MAT_ID's expert-index shuffle. Two independent process
   invocations of the same test case (forced-native, forced-candidate --
   exactly what HI80's evidence producer runs) therefore see DIFFERENT
   random expert routing for identical BIGCHERRY_TEST_DETERMINISTIC_SEED,
   breaking the fundamental HI67 correctness-evidence requirement that N
   and C be compared against the same operation/input.

2. Range confinement: the existing branch builds `data(t->ne[0])` (i.e. an
   n_expert_used-sized vector), fills it `i % n_expert` for i in
   [0, t->ne[0]), then shuffles THAT -- since t->ne[0] (n_expert_used) is
   always <= n_expert, `i % n_expert == i`, so this only ever produces a
   permutation of {0, ..., n_expert_used-1}. It can never select experts
   from the rest of the real expert pool (e.g. 8-of-256), unlike the
   registered test_mul_mat_id class's own init_mul_mat_id_tensors(), which
   shuffles the FULL n_mats-wide index range and views down to n_used.

This patch fixes both, gated to GGML_OP_MUL_MAT_ID specifically (not
ADD_ID, which shares the branch but is out of scope -- untested and
unrelated to HI105's MoE-dispatch correctness contract) and to
BIGCHERRY_TEST_DETERMINISTIC_SEED being set, so:
- unset (the default): every existing caller (upstream's own test suite,
  BigCherry's non-deterministic tune-mode corpus runs, ADD_ID always) is
  byte-for-byte unchanged.
- set: MUL_MAT_ID's ids get a seeded shuffle of the FULL [0, n_expert)
  range, truncated to the first n_expert_used entries per row -- the same
  shape of guarantee patch 1222 gives init_tensor_uniform, reusing that
  patch's helpers (bigcherry_deterministic_mode/_seed/_next_call_index/
  _fnv1a) rather than redeclaring them. REQUIRES enforces 1222 applies
  first so those helpers exist when this edit's code runs (test_generic_op
  is defined far later in the file than init_tensor_uniform, so ordinary
  C++ top-to-bottom visibility already guarantees the helpers are in scope
  by then -- REQUIRES is about apply-time coupling, not runtime scope).

Also emits its own BIGCHERRY_REF_DIGEST line per row (name=<ids tensor
name>, same FNV-1a scheme as patch 1222) so a Python evidence producer can
independently verify the newly-introduced routing input was identical
across the native and candidate processes, not just infer it from the
weight/activation digests.
"""

GROUP = "core"
# Verified offline: dry-run apply + idempotence against the real vendored
# checkout, applied on top of 1222 (both edits apply cleanly, re-applying is
# a no-op). NOT yet compiled or run on real hardware -- promote to
# "validated" once a real build exercises BIGCHERRY_TEST_DETERMINISTIC_SEED
# for a real MUL_MAT_ID test-file case and confirms two independent
# processes emit matching BIGCHERRY_REF_DIGEST lines for the ids tensor.
STATE = "untested"

REQUIRES = ("1222_hi67_deterministic_test_backend_ops_seed",)

import re as _re

from bigcherry.patcher import Edit, FilePatch

_BRANCH = '''                } else if (op == GGML_OP_MUL_MAT_ID || op == GGML_OP_ADD_ID) {
                    const int64_t n_expert = (op == GGML_OP_MUL_MAT_ID) ? sources[0].ne[2] : sources[1].ne[1];
                    if (op == GGML_OP_MUL_MAT_ID && bigcherry_deterministic_mode()) {
                        // bigcherry (HI105): deterministic, full-expert-range routing --
                        // see patches/1236_hi105_deterministic_mul_mat_id_ids.py. Reuses
                        // patch 1222's helpers (this file's init_tensor_uniform, defined
                        // earlier, already declares them).
                        for (int64_t r = 0; r < ggml_nrows(t); r++) {
                            const uint64_t call_index = bigcherry_next_call_index();
                            const uint64_t call_seed = bigcherry_deterministic_seed()
                                ^ (call_index * 0x9E3779B97F4A7C15ull);
                            std::default_random_engine det_gen(static_cast<unsigned>(call_seed));
                            std::vector<int32_t> pool(n_expert);
                            for (int64_t i = 0; i < n_expert; i++) {
                                pool[i] = (int32_t) i;
                            }
                            std::shuffle(pool.begin(), pool.end(), det_gen);
                            std::vector<int32_t> data(pool.begin(), pool.begin() + t->ne[0]);
                            ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                            const uint64_t digest = bigcherry_fnv1a(data.data(), data.size() * sizeof(int32_t));
                            fprintf(stderr, "BIGCHERRY_REF_DIGEST name=%s call_index=%llu digest=%016llx nels=%zu\\n",
                                    ggml_get_name(t), (unsigned long long) call_index,
                                    (unsigned long long) digest, data.size());
                        }
                    } else {
                    for (int64_t r = 0; r < ggml_nrows(t); r++) {
                        std::vector<int32_t> data(t->ne[0]);
                        for (int32_t i = 0; i < t->ne[0]; i++) {
                            data[i] = i % n_expert;
                        }
                        std::shuffle(data.begin(), data.end(), rng);
                        ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                    }
                    }'''

_ORIGINAL_BLOCK = '''                } else if (op == GGML_OP_MUL_MAT_ID || op == GGML_OP_ADD_ID) {
                    const int64_t n_expert = (op == GGML_OP_MUL_MAT_ID) ? sources[0].ne[2] : sources[1].ne[1];
                    for (int64_t r = 0; r < ggml_nrows(t); r++) {
                        std::vector<int32_t> data(t->ne[0]);
                        for (int32_t i = 0; i < t->ne[0]; i++) {
                            data[i] = i % n_expert;
                        }
                        std::shuffle(data.begin(), data.end(), rng);
                        ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                    }'''

_ANCHOR = _re.escape(_ORIGINAL_BLOCK)

PATCH = FilePatch(
    path="tests/test-backend-ops.cpp",
    description="test_generic_op's GGML_OP_MUL_MAT_ID initializer gets a deterministic, "
                "full-expert-range branch under BIGCHERRY_TEST_DETERMINISTIC_SEED, so HI105's "
                "CPU-reference correctness-evidence producer compares native/candidate "
                "against provably identical expert routing (HI105)",
    edits=(
        Edit(
            id="hi105-deterministic-mul-mat-id-ids",
            anchor=_ANCHOR,
            mode="replace",
            rationale="the only std::random_device-driven expert-index generation "
                       "test_generic_op's MUL_MAT_ID path has; ADD_ID (sharing this "
                       "branch) and the non-deterministic default are left untouched",
            text=_BRANCH,
            guard=r"bigcherry \(HI105\): deterministic, full-expert-range routing",
            max_span_lines=20,
        ),
    ),
)

PATCHES = [PATCH]
