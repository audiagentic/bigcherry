"""1290: default-off Vulkan communication-provider reference path (PRVP03 phase 0).

Exposes ggml_backend_comm_* through ggml-vulkan's registry so the meta
backend can use a Vulkan-owned AllReduce implementation.

Phase 0:
- BIGCHERRY_VK_ALLREDUCE=host-f32 enables it; unset preserves the stock fallback
- homogeneous Vulkan backend registry only
- contiguous F32 only
- synchronous host reduction
- unsupported cases return false to meta's generic fallback
- BIGCHERRY_PATCH_HIT logs only when the provider actually executes

Drafted with GPT (session ses_62f7004ea2d24bba, req_4f1dfb8d25834e71). Phase 1 replaces the data path
with a mapped-host design (one aligned host region imported into every device via
VK_EXT_external_memory_host, external timeline semaphores, root-device reduce shader).
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_REGISTRY_ANCHOR = """static const struct ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_count,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ NULL,
};
"""

_PROVIDER_CODE = r'''// BigCherry 1290: default-off Vulkan meta communication provider (host-f32 reference path).
struct ggml_backend_vk_comm_context {
    std::vector<ggml_backend_t> backends;
};

static void * ggml_backend_vk_comm_init(ggml_backend_t * backends, size_t n_backends) {
    const char * mode = getenv("BIGCHERRY_VK_ALLREDUCE");
    if (mode == nullptr || strcmp(mode, "host-f32") != 0) {
        return nullptr;
    }
    if (backends == nullptr || n_backends < 2) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[0]));
    if (reg == nullptr) {
        return nullptr;
    }
    // Fail closed if meta ever supplies a heterogeneous backend set.
    for (size_t i = 0; i < n_backends; ++i) {
        if (backends[i] == nullptr || ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[i])) != reg) {
            return nullptr;
        }
    }
    auto * ctx = new ggml_backend_vk_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    return ctx;
}

static void ggml_backend_vk_comm_free(void * comm_ctx) {
    delete static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
}

static bool ggml_backend_vk_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr || tensors == nullptr) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_vk_comm_context *>(comm_ctx_v);
    const size_t n_backends = ctx->backends.size();
    if (n_backends < 2 || tensors[0] == nullptr || tensors[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensors[0])) {
        return false;
    }
    const int64_t n_elements = ggml_nelements(tensors[0]);
    const size_t  nbytes     = ggml_nbytes(tensors[0]);
    if (n_elements <= 0 || nbytes != (size_t) n_elements * sizeof(float)) {
        return false;
    }
    for (size_t i = 0; i < n_backends; ++i) {
        const ggml_tensor * t = tensors[i];
        if (t == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
            ggml_nelements(t) != n_elements || ggml_nbytes(t) != nbytes) {
            return false;
        }
        // Producer work must be visible before host reads.
        ggml_backend_synchronize(ctx->backends[i]);
    }
    std::vector<float> sum((size_t) n_elements);
    std::vector<float> tmp((size_t) n_elements);
    ggml_backend_tensor_get(tensors[0], sum.data(), 0, nbytes);
    for (size_t i = 1; i < n_backends; ++i) {
        ggml_backend_tensor_get(tensors[i], tmp.data(), 0, nbytes);
        // Fixed rank order: deterministic reference reduction.
        for (int64_t j = 0; j < n_elements; ++j) {
            sum[(size_t) j] += tmp[(size_t) j];
        }
    }
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_tensor_set(tensors[i], sum.data(), 0, nbytes);
        ggml_backend_synchronize(ctx->backends[i]);
    }
    if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1290_vulkan_allreduce_host_f32 path=host-f32 bytes=%zu devices=%zu\n",
                      nbytes, n_backends);
    }
    return true;
}

static void * ggml_backend_vk_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) ggml_backend_vk_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) ggml_backend_vk_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *) ggml_backend_vk_comm_allreduce_tensor;
    }
    return nullptr;
}

'''

_REGISTRY_REPLACEMENT = """static const struct ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_count,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ ggml_backend_vk_reg_get_proc_address,
};
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        language="none",
        description="Default-off Vulkan meta communication SPI with a contiguous-F32 host AllReduce reference provider.",
        edits=(
            Edit(
                id="vk-comm-provider",
                anchor=_re.escape(_REGISTRY_ANCHOR),
                text=_PROVIDER_CODE,
                mode="insert_before",
                guard=r"struct ggml_backend_vk_comm_context",
                expect_matches=1,
                rationale="Define the communication SPI immediately before the registry that exposes it.",
            ),
            Edit(
                id="vk-reg-get-proc-address",
                anchor=_re.escape(_REGISTRY_ANCHOR),
                text=_REGISTRY_REPLACEMENT,
                mode="replace",
                guard=_re.escape("/* .get_proc_address = */ ggml_backend_vk_reg_get_proc_address,"),
                expect_matches=1,
                rationale="Expose comm_init/allreduce/free through the registry mechanism consumed by meta.",
            ),
        ),
    ),
]
