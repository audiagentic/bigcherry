"""1328 (MET05): auxiliary ROCm expert backend outside the tensor-parallel Meta device.

When BIGCHERRY_EXPERT_AUX_DEVICE names a plain GPU that is not a member of the target model's
-sm tensor Meta device, the target context registers a second ordinary backend for it. Routed
Qwen4Exp expert layers may then be placed there with -ot ...=ROCmN. Scheduler crossings between
Meta and that device are forced through the device's pinned host buffer and reject non-MIRRORED
Meta tensors. A marked Qwen4Exp shared-expert merge causes Meta to reduce only the shared partial
branch, then add the full mirrored auxiliary routed-MoE result without another AllReduce.
The feature is off unless BIGCHERRY_EXPERT_AUX_DEVICE is set.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

_H_DECL = """    GGML_API ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);
"""
_H_DECL_NEW = _H_DECL + """
    // BigCherry 1328: helpers for a plain auxiliary backend outside a Meta tensor group.
    GGML_API bool ggml_backend_meta_device_contains(ggml_backend_dev_t meta_dev, ggml_backend_dev_t simple_dev);
    GGML_API enum ggml_backend_meta_split_axis ggml_backend_meta_tensor_split_axis(const struct ggml_tensor * tensor);
    GGML_API void ggml_backend_meta_mark_mirrored_partial_add(struct ggml_tensor * tensor);
    GGML_API bool ggml_backend_meta_is_mirrored_partial_add(const struct ggml_tensor * tensor);
"""

_META_DEVICE_ANCHOR = """ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
"""
_META_DEVICE_HELPER = r'''// BigCherry 1328: membership query used before registering a plain auxiliary backend.
bool ggml_backend_meta_device_contains(ggml_backend_dev_t meta_dev, ggml_backend_dev_t simple_dev) {
    if (!ggml_backend_dev_is_meta(meta_dev) || simple_dev == nullptr) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx =
        (const ggml_backend_meta_device_context *) meta_dev->context;
    return std::find(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(), simple_dev)
        != meta_dev_ctx->simple_devs.end();
}

'''

_META_BIN = """    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
"""
_META_BIN_NEW = """    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // BigCherry 1328: full auxiliary routed-MoE + reduced shared expert.
        if (ggml_backend_meta_is_mirrored_partial_add(tensor)) {
            const bool exact =
                (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) ||
                (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL);
            if (!exact) {
                GGML_ABORT("BigCherry 1328: marked expert merge requires exactly MIRRORED + PARTIAL sources, got %s + %s",
                    ggml_backend_meta_split_axis_name(src_ss[0].axis), ggml_backend_meta_split_axis_name(src_ss[1].axis));
            }
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
"""

_META_END = """ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
"""
_META_END_NEW = _META_END + r'''
// BigCherry 1328: narrow public introspection/marker API for the aux-expert bridge.
enum ggml_backend_meta_split_axis ggml_backend_meta_tensor_split_axis(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
        GGML_ABORT("BigCherry 1328: split-axis query requires a Meta-buffer tensor");
    }
    return ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false).axis;
}

static constexpr int32_t BIGCHERRY_AUX_EXPERT_MERGE_MAGIC = 0x42434158; // "BCAX"

void ggml_backend_meta_mark_mirrored_partial_add(ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->op != GGML_OP_ADD) {
        GGML_ABORT("BigCherry 1328: auxiliary expert merge marker requires GGML_OP_ADD");
    }
    ggml_set_op_params_i32(tensor, 0, BIGCHERRY_AUX_EXPERT_MERGE_MAGIC);
}

bool ggml_backend_meta_is_mirrored_partial_add(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->op == GGML_OP_ADD &&
        ggml_get_op_params_i32(tensor, 0) == BIGCHERRY_AUX_EXPERT_MERGE_MAGIC;
}
'''

_SCHED_STRUCT = """    bool op_offload;

    int debug;
"""
_SCHED_STRUCT_NEW = """    bool op_offload;

    // BigCherry 1328: scheduler-owned pinned host bounce buffer for Meta <-> aux GPU transfers.
    ggml_backend_buffer_t      bc_aux_stage;
    ggml_backend_buffer_type_t bc_aux_stage_buft;
    size_t                     bc_aux_stage_size;

    int debug;
"""

_SCHED_PREALLOC = """    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
"""
_SCHED_PREALLOC_NEW = """    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // BigCherry 1328: keep the exact routed+shared merge on Meta. The PARTIAL shared branch is
    // reduced there; the scheduler-copied auxiliary branch is MIRRORED and must not be reduced.
    if (ggml_backend_meta_is_mirrored_partial_add(tensor)) {
        int meta_backend_id = -1;
        for (int b = 0; b < sched->n_backends; ++b) {
            if (ggml_backend_dev_type(ggml_backend_get_device(sched->backends[b])) == GGML_BACKEND_DEVICE_TYPE_META) {
                if (meta_backend_id != -1) {
                    GGML_ABORT("BigCherry 1328: auxiliary expert merge found multiple Meta scheduler backends");
                }
                meta_backend_id = b;
            }
        }
        if (meta_backend_id == -1) {
            GGML_ABORT("BigCherry 1328: auxiliary expert merge has no Meta scheduler backend");
        }
        SET_CAUSE(tensor, "1.bcax");
        return meta_backend_id;
    }

    // graph input
"""

_COPY_FALLBACK = r'''    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
        ggml_backend_synchronize(input_backend);
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } else {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(input, input_cpy);
    }
'''
_COPY_FALLBACK_NEW = r'''    // BigCherry 1328: no P2P for the auxiliary expert device. Cross only Meta <-> the
    // named ordinary GPU through its pinned host buffer, and only for MIRRORED Meta data.
    const char * bc_aux_name = getenv("BIGCHERRY_EXPERT_AUX_DEVICE");
    auto bc_backend_is_meta = [](ggml_backend_t b) {
        return ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_META;
    };
    auto bc_backend_is_aux = [&](ggml_backend_t b) {
        return bc_aux_name != nullptr && bc_aux_name[0] != '\0' &&
            !bc_backend_is_meta(b) &&
            strcmp(ggml_backend_dev_name(ggml_backend_get_device(b)), bc_aux_name) == 0;
    };
    const bool bc_meta_to_aux = bc_backend_is_meta(input_backend) && bc_backend_is_aux(split_backend);
    const bool bc_aux_to_meta = bc_backend_is_aux(input_backend) && bc_backend_is_meta(split_backend);
    if (bc_meta_to_aux || bc_aux_to_meta) {
        ggml_tensor * meta_tensor = bc_meta_to_aux ? input : input_cpy;
        const ggml_backend_meta_split_axis axis = ggml_backend_meta_tensor_split_axis(meta_tensor);
        if (axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ABORT("BigCherry 1328: Meta <-> %s staging requires MIRRORED tensor %s, got split axis %s",
                bc_aux_name, input->name, ggml_backend_meta_split_axis_name(axis));
        }

        ggml_backend_t aux_backend = bc_meta_to_aux ? split_backend : input_backend;
        ggml_backend_dev_t aux_dev = ggml_backend_get_device(aux_backend);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(aux_dev);
        if (host_buft == nullptr || !ggml_backend_buft_is_host(host_buft)) {
            GGML_ABORT("BigCherry 1328: auxiliary device %s has no pinned host buffer type", bc_aux_name);
        }

        const size_t nbytes = ggml_nbytes(input);
        if (ggml_nbytes(input_cpy) != nbytes) {
            GGML_ABORT("BigCherry 1328: staging layout mismatch for tensor %s", input->name);
        }
        if (sched->bc_aux_stage == nullptr || sched->bc_aux_stage_buft != host_buft ||
                sched->bc_aux_stage_size < nbytes) {
            ggml_backend_synchronize(input_backend);
            ggml_backend_synchronize(split_backend);
            ggml_backend_buffer_free(sched->bc_aux_stage);
            sched->bc_aux_stage = ggml_backend_buft_alloc_buffer(host_buft, nbytes);
            if (sched->bc_aux_stage == nullptr) {
                GGML_ABORT("BigCherry 1328: failed to allocate %zu-byte pinned staging buffer for %s",
                    nbytes, bc_aux_name);
            }
            sched->bc_aux_stage_buft = host_buft;
            sched->bc_aux_stage_size = ggml_backend_buffer_get_size(sched->bc_aux_stage);
        }

        void * stage = ggml_backend_buffer_get_base(sched->bc_aux_stage);
        ggml_backend_synchronize(input_backend);
        ggml_backend_tensor_get(input, stage, 0, nbytes);
        ggml_backend_tensor_set(input_cpy, stage, 0, nbytes);
        ggml_backend_synchronize(split_backend);

        static bool bc_traced = false;
        if (!bc_traced && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1328_aux_rocm_expert_backend aux=%s bytes=%zu\n",
                bc_aux_name, nbytes);
            bc_traced = true;
        }
    } else {
        // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
        // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
        if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
            ggml_backend_synchronize(input_backend);
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(split_backend);
            }
            ggml_backend_tensor_copy(input, input_cpy);
        }
    }
'''

_SCHED_FREE = """    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
"""
_SCHED_FREE_NEW = """    ggml_backend_buffer_free(sched->bc_aux_stage); // BigCherry 1328: pinned aux bounce buffer
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
"""

_CTX_GPU_LOOP = """        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
"""
_CTX_GPU_LOOP_NEW = r'''        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // BigCherry 1328: target-only plain auxiliary expert backend. It is deliberately not appended
        // to model.devices, so it can never become a Meta constituent or enter that device's communicator.
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            const char * bc_aux_name = getenv("BIGCHERRY_EXPERT_AUX_DEVICE");
            if (bc_aux_name != nullptr && bc_aux_name[0] != '\0') {
                if (model.arch != LLM_ARCH_QWEN4EXP) {
                    throw std::runtime_error("BIGCHERRY_EXPERT_AUX_DEVICE is only supported for Qwen4Exp tensor-split target models");
                }
                if (model.devices.size() != 1 ||
                        ggml_backend_dev_type(model.devices[0].dev) != GGML_BACKEND_DEVICE_TYPE_META) {
                    throw std::runtime_error("BIGCHERRY_EXPERT_AUX_DEVICE requires exactly one target Meta device");
                }

                ggml_backend_dev_t bc_aux_dev = nullptr;
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                    if (strcmp(ggml_backend_dev_name(dev), bc_aux_name) == 0) {
                        if (bc_aux_dev != nullptr) {
                            throw std::runtime_error(format("BIGCHERRY_EXPERT_AUX_DEVICE=%s is ambiguous", bc_aux_name));
                        }
                        bc_aux_dev = dev;
                    }
                }
                if (bc_aux_dev == nullptr) {
                    throw std::runtime_error(format("BIGCHERRY_EXPERT_AUX_DEVICE=%s was not found", bc_aux_name));
                }
                const enum ggml_backend_dev_type bc_aux_type = ggml_backend_dev_type(bc_aux_dev);
                if (bc_aux_type != GGML_BACKEND_DEVICE_TYPE_GPU && bc_aux_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    throw std::runtime_error(format("BIGCHERRY_EXPERT_AUX_DEVICE=%s is not a GPU", bc_aux_name));
                }
                if (ggml_backend_meta_device_contains(model.devices[0].dev, bc_aux_dev)) {
                    throw std::runtime_error(format("BIGCHERRY_EXPERT_AUX_DEVICE=%s is already a Meta tensor-group member", bc_aux_name));
                }

                ggml_backend_t bc_aux_backend = ggml_backend_dev_init(bc_aux_dev, nullptr);
                if (bc_aux_backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize auxiliary expert backend %s", bc_aux_name));
                }
                size_t bc_aux_free = 0, bc_aux_total = 0;
                ggml_backend_dev_memory(bc_aux_dev, &bc_aux_free, &bc_aux_total);
                LLAMA_LOG_INFO("%s: BigCherry auxiliary expert backend %s: %zu MiB free / %zu MiB total\n",
                    __func__, bc_aux_name, bc_aux_free/1024/1024, bc_aux_total/1024/1024);
                backends.emplace_back(bc_aux_backend);
            }
        }

        // add ACCEL backends (such as BLAS)
'''

_QWEN_INCLUDE = "#include <cinttypes>\n"
_QWEN_INCLUDE_NEW = _QWEN_INCLUDE + "#include <cstdlib>  // BigCherry 1328: getenv\n#include <cstring>  // BigCherry 1328: strcmp\n"

_QWEN_HEAD = """ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
"""
_QWEN_HEAD_NEW = r'''ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    // BigCherry 1328: a selected layer is all-or-nothing on the ordinary aux device. Mixed routed
    // tensors would combine full and tensor-partial semantics and are rejected before graph reserve.
    bool bc_aux_layer = false;
    if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        const char * bc_aux_name = getenv("BIGCHERRY_EXPERT_AUX_DEVICE");
        if (bc_aux_name != nullptr && bc_aux_name[0] != '\0') {
            auto bc_on_aux = [&](const ggml_tensor * t) {
                if (t == nullptr) {
                    return false;
                }
                ggml_backend_buffer_t buffer = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
                if (buffer == nullptr) {
                    throw std::runtime_error("BigCherry 1328: routed expert tensor is not allocated");
                }
                ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
                return dev != nullptr && strcmp(ggml_backend_dev_name(dev), bc_aux_name) == 0;
            };

            const ggml_tensor * routed[] = {
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_gate_up_exps,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s,
            };
            int n_routed = 0;
            int n_aux = 0;
            for (const ggml_tensor * t : routed) {
                if (t != nullptr) {
                    ++n_routed;
                    n_aux += bc_on_aux(t) ? 1 : 0;
                }
            }
            if (n_aux != 0 && n_aux != n_routed) {
                throw std::runtime_error("BigCherry 1328: partial auxiliary routed-expert placement is unsupported");
            }
            bc_aux_layer = n_aux != 0;

            if (bc_aux_layer) {
                const ggml_tensor * shared[] = {
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_shexp,
                    model.layers[il].ffn_gate_shexp,
                    model.layers[il].ffn_down_shexp,
                    model.layers[il].ffn_gate_inp_shexp,
                };
                if (model.layers[il].ffn_up_shexp == nullptr ||
                        model.layers[il].ffn_gate_shexp == nullptr ||
                        model.layers[il].ffn_down_shexp == nullptr ||
                        model.layers[il].ffn_gate_inp_shexp == nullptr) {
                    throw std::runtime_error("BigCherry 1328: auxiliary experts require the validated Qwen4Exp shared-expert path");
                }
                for (const ggml_tensor * t : shared) {
                    if (t != nullptr && bc_on_aux(t)) {
                        throw std::runtime_error("BigCherry 1328: shared/router tensors must remain on Meta");
                    }
                }
            }
        }
    }

    ggml_tensor * moe_out =
'''

_QWEN_MERGE = """        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
"""
_QWEN_MERGE_NEW = """        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        if (bc_aux_layer) {
            ggml_backend_meta_mark_mirrored_partial_add(cur); // BigCherry 1328: reduce shared branch only
        }
        cb(cur, "ffn_out", il);
"""

PATCHES = [
    FilePatch(
        path="ggml/include/ggml-backend.h",
        description="1328: expose narrow Meta helpers required by the auxiliary expert bridge",
        language="none",
        edits=(
            Edit(id="meta-aux-api", anchor=re.escape(_H_DECL), mode="replace", text=_H_DECL_NEW,
                 guard=r"BigCherry 1328: helpers for a plain auxiliary backend",
                 rationale="Meta device factory declaration is the stable public Meta API seam.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1328: Meta membership, split-state introspection, and exact mirrored+partial merge semantics",
        language="none",
        edits=(
            Edit(id="meta-device-membership", anchor=re.escape(_META_DEVICE_ANCHOR), mode="insert_before",
                 text=_META_DEVICE_HELPER, guard=r"BigCherry 1328: membership query",
                 rationale="Insert beside Meta device construction where the private constituent list is available.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="meta-exact-merge", anchor=re.escape(_META_BIN), mode="replace", text=_META_BIN_NEW,
                 guard=r"BigCherry 1328: full auxiliary routed-MoE \+ reduced shared expert",
                 rationale="Binary-broadcast split propagation owns ADD's Meta reduction semantics.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="meta-aux-helpers", anchor=re.escape(_META_END), mode="replace", text=_META_END_NEW,
                 guard=r"BIGCHERRY_AUX_EXPERT_MERGE_MAGIC",
                 rationale="Public helper implementations sit after the existing Meta backend accessors.",
                 expect_matches=1, max_span_lines=6),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1328: route marked merge to Meta and force Meta/aux crossings through pinned host staging",
        language="none",
        edits=(
            Edit(id="sched-aux-stage-state", anchor=re.escape(_SCHED_STRUCT), mode="replace", text=_SCHED_STRUCT_NEW,
                 guard=r"BigCherry 1328: scheduler-owned pinned host bounce buffer",
                 rationale="Scheduler lifetime owns cross-split copies and therefore the reusable staging allocation.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="sched-meta-merge-backend", anchor=re.escape(_SCHED_PREALLOC), mode="replace", text=_SCHED_PREALLOC_NEW,
                 guard=r"BigCherry 1328: keep the exact routed\+shared merge on Meta",
                 rationale="Pass-1 backend assignment must pin the marked merge before adjacency expansion can select aux.",
                 expect_matches=1, max_span_lines=8),
            Edit(id="sched-pinned-aux-copy", anchor=re.escape(_COPY_FALLBACK), mode="replace", text=_COPY_FALLBACK_NEW,
                 guard=r"BigCherry 1328: no P2P for the auxiliary expert device",
                 rationale="The generic inter-split copy fallback is the unique Meta/ordinary-backend transfer seam and composes after 1326's input fast path.",
                 expect_matches=1, max_span_lines=17),
            Edit(id="sched-free-aux-stage", anchor=re.escape(_SCHED_FREE), mode="replace", text=_SCHED_FREE_NEW,
                 guard=r"BigCherry 1328: pinned aux bounce buffer",
                 rationale="Free scheduler-owned pinned staging before allocator/context teardown.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="src/llama-context.cpp",
        description="1328: register a named plain auxiliary GPU backend in tensor-split target contexts",
        language="none",
        edits=(
            Edit(id="target-aux-backend", anchor=re.escape(_CTX_GPU_LOOP), mode="replace", text=_CTX_GPU_LOOP_NEW,
                 guard=r"BigCherry 1328: target-only plain auxiliary expert backend",
                 rationale="Backends are initialized here before the scheduler list is frozen; model.devices remains the Meta-only communicator membership.",
                 expect_matches=1, max_span_lines=11),
        ),
    ),
    FilePatch(
        path="src/models/qwen4exp.cpp",
        description="1328: validate whole routed-expert layer placement and mark exact shared-expert merge",
        language="none",
        edits=(
            Edit(id="qwen4exp-aux-includes", anchor=re.escape(_QWEN_INCLUDE), mode="replace", text=_QWEN_INCLUDE_NEW,
                 guard=r"BigCherry 1328: strcmp", rationale="Qwen4Exp standard include block; makes the env-gated patch self-contained.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="qwen4exp-aux-layer-validation", anchor=re.escape(_QWEN_HEAD), mode="replace", text=_QWEN_HEAD_NEW,
                 guard=r"BigCherry 1328: a selected layer is all-or-nothing",
                 rationale="Validate placement before constructing the routed MoE graph.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="qwen4exp-mark-merge", anchor=re.escape(_QWEN_MERGE), mode="replace", text=_QWEN_MERGE_NEW,
                 guard=r"BigCherry 1328: reduce shared branch only",
                 rationale="The routed/shared ADD is the exact boundary where full aux output meets Meta partial shared output.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
]
