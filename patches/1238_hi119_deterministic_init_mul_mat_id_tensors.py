"""HI119: deterministic expert-ID routing for the REGISTERED test_case
classes' shared initializer, init_mul_mat_id_tensors().

HI119 needs to reuse test_mul_mat_vec_fusion's use_id=true path (a
registered C++ test_case, not test-backend-ops' --test-file/test_generic_op
escape hatch) as the template for a new fused MUL_MAT_ID+GLU correctness-
evidence harness -- ggml-cuda's own graph-fusion detector only fires on a
real, adjacent multi-node graph, which test_generic_op's single-op-per-line
model cannot build (see HI108/HI119's own investigation).

Found (dev-gpt-agent deep design review, 2026-08-25, verified against this
exact vendored source before being accepted) that init_mul_mat_id_tensors()
-- the initializer EVERY registered MUL_MAT_ID-family test_case uses
(test_mul_mat_id, test_mul_mat_vec_fusion, and HI119's own planned class)
-- is a THIRD, separate std::random_device site, distinct from the two
patch 1222/1236 already cover:

- patch 1222 only covers init_tensor_uniform() (float/general tensor init).
- patch 1236 only covers test_generic_op's own internal MUL_MAT_ID/ADD_ID
  branch -- its own docstring explicitly scopes itself to that function and
  explicitly calls out init_mul_mat_id_tensors() as a DIFFERENT, untouched
  site ("unlike the registered test_mul_mat_id class's own
  init_mul_mat_id_tensors()").
- init_mul_mat_id_tensors() itself: `std::random_device rd;
  std::default_random_engine rng(rd());` at its own top, used by every
  registered test_case that calls it -- confirmed genuinely non-deterministic
  today, for every existing caller.

Without this fix, two independent process invocations of the SAME
registered MUL_MAT_ID-family test_case (forced-native, forced-candidate --
exactly what a correctness-evidence producer runs) see DIFFERENT random
expert routing even with BIGCHERRY_TEST_DETERMINISTIC_SEED set for the
float side, silently invalidating any correctness comparison built on top
of them -- the same class of gap HI105/1236 fixed for test_generic_op, now
found in the sibling registered-class code path HI119 actually needs.

Reuses patch 1222's helpers (bigcherry_deterministic_mode/_seed/
_next_call_index/_fnv1a) exactly like 1236 does, rather than redeclaring
them. Gated on BIGCHERRY_TEST_DETERMINISTIC_SEED being set, so:
- unset (the default): every existing caller (test_mul_mat_id's own
  upstream test suite runs, non-deterministic tune-mode corpus runs) is
  byte-for-byte unchanged.
- set: the full [0, n_mats) index range is shuffled with a seeded engine
  and truncated to the first t->ne[0] entries per row -- the exact same
  shape of guarantee 1236 gives test_generic_op's own branch, and the exact
  same full-range shuffle-then-truncate semantics this function's own
  ORIGINAL (non-deterministic) code already used, so the deterministic path
  produces routing from the identical distribution, just seeded.

Also emits its own BIGCHERRY_REF_DIGEST line per row (name=<ids tensor
name>, same FNV-1a scheme as patch 1222/1236) so a Python evidence producer
can independently verify the routing input was identical across the native
and candidate processes.
"""

GROUP = "core"
# Verified offline: dry-run apply + idempotence against the real vendored
# checkout, applied on top of 1222 (and independent of / non-conflicting
# with 1236, which edits a different function). NOT yet compiled or run on
# real hardware -- promote to "validated" once a real build exercises
# BIGCHERRY_TEST_DETERMINISTIC_SEED against a registered MUL_MAT_ID-family
# test_case (e.g. test_mul_mat_id or HI119's own new class) and confirms two
# independent processes emit matching BIGCHERRY_REF_DIGEST lines for the ids
# tensor.
STATE = "untested"

REQUIRES = ("1222_hi67_deterministic_test_backend_ops_seed",)

import re as _re

from bigcherry import csource as _csource
from bigcherry.patcher import Edit, FilePatch

_NEW_BODY = '''static void init_mul_mat_id_tensors(ggml_context * ctx, int n_mats) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type == GGML_TYPE_I32) {
            if (ggml_is_view_op(t->op)) { continue; }
            // ids
            if (bigcherry_deterministic_mode()) {
                // bigcherry (HI119): deterministic, full-range expert routing --
                // see patches/1238_hi119_deterministic_init_mul_mat_id_tensors.py.
                // Reuses patch 1222's helpers (this file's init_tensor_uniform,
                // defined earlier, already declares them).
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    const uint64_t call_index = bigcherry_next_call_index();
                    const uint64_t call_seed = bigcherry_deterministic_seed()
                        ^ (call_index * 0x9E3779B97F4A7C15ull);
                    std::default_random_engine det_gen(static_cast<unsigned>(call_seed));
                    std::vector<int32_t> pool(n_mats);
                    for (int i = 0; i < n_mats; i++) {
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
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_mats;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            }
        } else {
            init_tensor_uniform(t);
        }
    }
}'''

_ORIGINAL_BODY = '''static void init_mul_mat_id_tensors(ggml_context * ctx, int n_mats) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type == GGML_TYPE_I32) {
            if (ggml_is_view_op(t->op)) { continue; }
            // ids
            for (int64_t r = 0; r < ggml_nrows(t); r++) {
                std::vector<int32_t> data(t->ne[0]);
                for (int i = 0; i < t->ne[0]; i++) {
                    data[i] = i % n_mats;
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
            }
        } else {
            init_tensor_uniform(t);
        }
    }
}'''

# The real matcher (bigcherry.patch.apply._find_anchor) matches against a
# comment/string-blanked view of the file (csource.strip_noise), not the raw
# text -- otherwise this function's own `// ids` comment would need to be
# reproduced byte-for-byte in the anchor. Building the anchor from the same
# stripped view the matcher actually uses (rather than the raw original body)
# keeps the two in sync automatically instead of hand-blanking the comment.
_ANCHOR = _re.escape(_csource.strip_noise(_ORIGINAL_BODY, "c"))

PATCH = FilePatch(
    path="tests/test-backend-ops.cpp",
    description="init_mul_mat_id_tensors() (shared by every registered MUL_MAT_ID-family "
                "test_case) gets a deterministic, full-range expert-routing branch under "
                "BIGCHERRY_TEST_DETERMINISTIC_SEED, so a correctness-evidence producer built "
                "on a registered test_case (HI119) compares native/candidate against provably "
                "identical expert routing",
    edits=(
        Edit(
            id="hi119-deterministic-init-mul-mat-id-tensors",
            anchor=_ANCHOR,
            mode="replace",
            rationale="the only std::random_device-driven expert-index generation "
                       "init_mul_mat_id_tensors() has -- every registered MUL_MAT_ID-family "
                       "test_case (test_mul_mat_id, test_mul_mat_vec_fusion, HI119's own new "
                       "class) shares this one initializer, so fixing it here covers all of "
                       "them at once rather than duplicating the fix per class",
            text=_NEW_BODY,
            guard=r"bigcherry \(HI119\): deterministic, full-range expert routing",
            max_span_lines=40,
        ),
    ),
)

PATCHES = [PATCH]
