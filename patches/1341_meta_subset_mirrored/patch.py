"""1341: subset-mirrored Meta split state for Qwen4Exp indexer cache.

BIGCHERRY_META_SUBSET_MIRROR=1 seeds cache_idx_(k|v)_l* from BIGCHERRY_ATTN_TS: participating attention devices
hold a complete replica, zero-share devices get a zero-sized simple tensor. active_mask=0 is the legacy/all-devices
state. This first step does not seed KQ/kpool/QSA compute inputs.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "validated"

_A_STATE = r"""        int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
        uint32_t nr[16];
        uint32_t n_segments;
"""
_N_STATE = r"""        int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
        uint32_t nr[16];
        uint32_t n_segments;

        // BigCherry 1341 (MSM03): 0 means legacy/all devices; otherwise bit j says device j owns a full replica.
        uint32_t active_mask;
"""

_A_HELPER_SITE = "static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);\n"
_N_HELPER_SITE = _A_HELPER_SITE + r"""
// BigCherry 1341 (MSM03): subset-mirrored tensors keep their full logical shape but are absent from inactive devices.
static bool ggml_backend_meta_split_device_active(const ggml_backend_meta_split_state & split_state, size_t j) {
    return split_state.active_mask == 0 || (split_state.active_mask & (uint32_t(1) << j)) != 0;
}
"""

_A_MERGE_SITE = r"""    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
"""
_N_MERGE_SITE = r"""    auto merge_active_masks = [](uint32_t a, uint32_t b) -> uint32_t {
        if (a == 0) {
            return b;
        }
        if (b == 0) {
            return a;
        }
        GGML_ASSERT(a == b);
        return a;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
"""

_A_LIGHTNING = r"""    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };
"""
_N_LIGHTNING = r"""    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        uint32_t active_mask = 0;
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            // BigCherry 1341 (MSM03): explicit subset masks must agree; 0 is the legacy/all-devices superset.
            active_mask = merge_active_masks(active_mask, src_ss[i].active_mask);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, active_mask};
    };
"""

_A_PROPAGATE = r"""        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
"""
_N_PROPAGATE = r"""        }

        // BigCherry 1341 (MSM03): a MIRRORED result only needs devices common to its subset-mirrored inputs.
        // active_mask==0 means all devices, so it acts as the identity. Differing explicit subsets fail closed.
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            uint32_t active_mask = split_state.active_mask;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] != nullptr && tensor->src[i] != tensor &&
                        src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                    active_mask = merge_active_masks(active_mask, src_ss[i].active_mask);
                }
            }
            split_state.active_mask = active_mask;
        }

        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
"""

_A_INIT_ASSERT = r"""    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);
"""
_N_INIT_ASSERT = _A_INIT_ASSERT + r"""    GGML_ASSERT(split_state.active_mask == 0 || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
"""

_A_INIT_SLICE = "        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {\n"
_N_INIT_SLICE = r"""        if (split_state.active_mask != 0) {
            // BigCherry 1341 (MSM03): reset ne[0] on every device; the shape array is reused by this loop.
            ne[0] = ggml_backend_meta_split_device_active(split_state, j) ? tensor->ne[0] : 0;
        } else if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
"""

_A_DISABLE = r"""        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
"""
_N_DISABLE = r"""        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.active_mask != 0) {
            GGML_ASSERT(split_state_src.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            for (size_t j = 0; j < n_simple_bufs; j++) {
                if (!ggml_backend_meta_split_device_active(split_state_src, j)) {
                    simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                }
            }
        }
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
"""

_A_MEMSET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
"""
_N_MEMSET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // BigCherry 1341 MSM03: memset subset mirror.
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ggml_backend_meta_split_device_active(split_state, j)) {
                    continue;
                }
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
"""

_A_SET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
"""
_N_SET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // BigCherry 1341 MSM03: synchronous set subset mirror.
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ggml_backend_meta_split_device_active(split_state, j)) {
                    continue;
                }
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
"""

_A_GET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
"""
_N_GET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // BigCherry 1341 MSM03: synchronous get subset mirror.
            // TODO other simple backend may be better
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ggml_backend_meta_split_device_active(split_state, j)) {
                    continue;
                }
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_get(simple_tensor, data, offset, size);
                return;
            }
            GGML_ABORT("subset-mirrored tensor has no active device");
        } break;
"""

_A_ASYNC_SET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
"""
_N_ASYNC_SET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // BigCherry 1341 MSM03: asynchronous set subset mirror.
            for (size_t j = 0; j < n_backends; j++) {
                if (!ggml_backend_meta_split_device_active(split_state, j)) {
                    continue;
                }
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
"""

_A_ASYNC_GET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
"""
_N_ASYNC_GET = r"""        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // BigCherry 1341 MSM03: asynchronous get subset mirror.
            // TODO other simple backend may be better
            for (size_t j = 0; j < n_backends; j++) {
                if (!ggml_backend_meta_split_device_active(split_state, j)) {
                    continue;
                }
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
                return;
            }
            GGML_ABORT("subset-mirrored tensor has no active device");
        } break;
"""

_A_MODEL_SEED = r"""    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
"""
_N_MODEL_SEED = r"""    split_state.axis = tc.axis;

    // BigCherry 1341 (MSM03): first subset-mirror seed is Qwen4Exp's persistent indexer cache only.
    static const bool bigcherry_subset_mirror = getenv("BIGCHERRY_META_SUBSET_MIRROR") != nullptr &&
                                                 atoi(getenv("BIGCHERRY_META_SUBSET_MIRROR")) != 0;
    if (bigcherry_subset_mirror && ud->model->arch == LLM_ARCH_QWEN4EXP &&
            std::regex_match(tensor_name, pattern_idx_cache)) {
        if (!bigcherry_attn_split.enabled) {
            throw std::runtime_error("BIGCHERRY_META_SUBSET_MIRROR requires BIGCHERRY_ATTN_TS for qwen4exp indexer cache");
        }
        uint32_t active_mask = 0;
        for (size_t j = 0; j < ud->n_devices; j++) {
            if (bigcherry_attn_split.split[j] != 0.0f) {
                active_mask |= uint32_t(1) << j;
            }
        }
        if (active_mask == 0) {
            throw std::runtime_error("BIGCHERRY_META_SUBSET_MIRROR: BIGCHERRY_ATTN_TS has no active device");
        }
        const uint32_t all_mask = (uint32_t(1) << ud->n_devices) - 1;
        split_state.active_mask = active_mask == all_mask ? 0 : active_mask;
    }

    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
"""

PATCHES = [
    FilePatch(
        path="ggml/include/ggml-backend.h",
        description="1341: active_mask in Meta split state",
        language="none",
        edits=(
            Edit(id="subset-mirror-state", anchor=_re.escape(_A_STATE), mode="replace", text=_N_STATE,
                 guard=r"BigCherry 1341 \(MSM03\): 0 means legacy/all devices",
                 rationale="Append to ggml_backend_meta_split_state so existing aggregate initializers default to zero.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1341: propagate and honor subset-mirrored Meta split states",
        language="none",
        edits=(
            Edit(id="subset-mirror-active-helper", anchor=_re.escape(_A_HELPER_SITE), mode="replace", text=_N_HELPER_SITE,
                 guard=r"ggml_backend_meta_split_device_active",
                 rationale="Next to the split-state forward declaration used by init and transfer paths.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="subset-mirror-merge-helper", anchor=_re.escape(_A_MERGE_SITE), mode="replace", text=_N_MERGE_SITE,
                 guard=r"auto merge_active_masks =",
                 rationale="Shared propagation rule before generic split-state handling.", expect_matches=1, max_span_lines=2),
            Edit(id="subset-mirror-lightning", anchor=_re.escape(_A_LIGHTNING), mode="replace", text=_N_LIGHTNING,
                 guard=r"explicit subset masks must agree",
                 rationale="LIGHTNING_INDEXER requires compatible full replicas and preserves their subset.", expect_matches=1, max_span_lines=8),
            Edit(id="subset-mirror-propagate", anchor=_re.escape(_A_PROPAGATE), mode="replace", text=_N_PROPAGATE,
                 guard=r"a MIRRORED result only needs devices common to its subset-mirrored inputs",
                 rationale="Common post-op propagation before dimensional split ratios are filled.", expect_matches=1, max_span_lines=4),
            Edit(id="subset-mirror-init-assert", anchor=_re.escape(_A_INIT_ASSERT), mode="replace", text=_N_INIT_ASSERT,
                 guard=r"split_state.active_mask == 0 \|\| split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED",
                 rationale="This first step only supports subset semantics for MIRRORED states.", expect_matches=1, max_span_lines=4),
            Edit(id="subset-mirror-init-zero", anchor=_re.escape(_A_INIT_SLICE), mode="replace", text=_N_INIT_SLICE,
                 guard=r"reset ne\[0\] on every device; the shape array is reused by this loop",
                 rationale="Choose the full or zero first dimension afresh for every subset-mirrored device; ne[] persists across loop iterations.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="subset-mirror-disable-compute", anchor=_re.escape(_A_DISABLE), mode="replace", text=_N_DISABLE,
                 guard=r"if \(split_state_src.active_mask != 0\)",
                 rationale="Any node consuming an absent subset replica must not compute on that device.", expect_matches=1, max_span_lines=7),
            Edit(id="subset-mirror-memset", anchor=_re.escape(_A_MEMSET), mode="replace", text=_N_MEMSET,
                 guard=r"BigCherry 1341 MSM03: memset subset mirror",
                 rationale="Do not write the zero-sized replica during Meta memset.", expect_matches=1, max_span_lines=7),
            Edit(id="subset-mirror-set", anchor=_re.escape(_A_SET), mode="replace", text=_N_SET,
                 guard=r"BigCherry 1341 MSM03: synchronous set subset mirror",
                 rationale="Synchronous Meta set skips inactive replicas.", expect_matches=1, max_span_lines=7),
            Edit(id="subset-mirror-get", anchor=_re.escape(_A_GET), mode="replace", text=_N_GET,
                 guard=r"BigCherry 1341 MSM03: synchronous get subset mirror",
                 rationale="Read a complete active replica instead of assuming device zero.", expect_matches=1, max_span_lines=6),
            Edit(id="subset-mirror-async-set", anchor=_re.escape(_A_ASYNC_SET), mode="replace", text=_N_ASYNC_SET,
                 guard=r"BigCherry 1341 MSM03: asynchronous set subset mirror",
                 rationale="Asynchronous Meta set skips inactive replicas.", expect_matches=1, max_span_lines=7),
            Edit(id="subset-mirror-async-get", anchor=_re.escape(_A_ASYNC_GET), mode="replace", text=_N_ASYNC_GET,
                 guard=r"BigCherry 1341 MSM03: asynchronous get subset mirror",
                 rationale="Asynchronous Meta get selects an active complete replica.", expect_matches=1, max_span_lines=7),
        ),
    ),
    FilePatch(
        path="src/llama-model.cpp",
        description="1341: seed Qwen4Exp cache_idx subset from BIGCHERRY_ATTN_TS",
        language="none",
        edits=(
            Edit(id="subset-mirror-indexer-seed", anchor=_re.escape(_A_MODEL_SEED), mode="replace", text=_N_MODEL_SEED,
                 guard=r"BIGCHERRY_META_SUBSET_MIRROR requires BIGCHERRY_ATTN_TS",
                 rationale="After 1303 selects tensor config/rotation and before dimensional split construction.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_META_SUBSET_MIRROR", "0|1", "0",
           "qwen4exp tensor split: keep cache_idx replicas only on devices with nonzero BIGCHERRY_ATTN_TS share"),
)
