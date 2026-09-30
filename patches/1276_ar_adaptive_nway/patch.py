"""1276: compose adaptive AllReduce with the N=3 root path.

Requires 0840_hybrid_allreduce_dispatch and
1244_gp11_internal_allreduce_nway_root. Adaptive keeps the 0860 crossover:
logical tensor bytes below --allreduce-switch-bytes prefer the internal
provider (root3 for N=3), while larger reductions prefer RCCL. This overlay
also makes the root rank selectable once at pipeline init through
BIGCHERRY_AR_ROOT3_ROOT=0|1|2 (default 0).
"""

from __future__ import annotations

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "gpu-collectives"
STATE = "untested"

_ROOT3_FIELDS_OLD = """    uint8_t * host_buf_dev3[3][3] = {};
    int     * arrival_dev3[3]     = {};
};"""
_ROOT3_FIELDS_NEW = """    uint8_t * host_buf_dev3[3][3] = {};
    int     * arrival_dev3[3]     = {};
    int       root3_root          = 0;
};"""

_ENV_U64_ANCHOR = """static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}
"""
_ROOT3_ENV_HELPER = r'''

static int ggml_cuda_ar_root3_root_from_env() {
    const char * value = getenv("BIGCHERRY_AR_ROOT3_ROOT");
    if (value == nullptr || value[0] == '\0' || strcmp(value, "0") == 0) {
        return 0;
    }
    if (strcmp(value, "1") == 0) {
        return 1;
    }
    if (strcmp(value, "2") == 0) {
        return 2;
    }
    GGML_LOG_WARN("%s: BIGCHERRY_AR_ROOT3_ROOT='%s' invalid; using 0\n", __func__, value);
    return 0;
}
'''

_ROOT3_INIT_OLD = """    if (n_devices == 3) {
        // GP11 preliminary: validated raw F32 wire transfer only.
        p->bf16_threshold = 0;
    }"""
_ROOT3_INIT_NEW = """    if (n_devices == 3) {
        // GP11 preliminary: validated raw F32 wire transfer only.
        p->bf16_threshold = 0;
        p->root3_root = ggml_cuda_ar_root3_root_from_env();
    }"""

_ROOT3_INDEX_ANCHOR = """    bool compute[3] = {};
"""
_ROOT3_INDEX_SETUP = """    const int root  = p->root3_root;
    const int leaf0 = (root + 1) % 3;
    const int leaf1 = (root + 2) % 3;
    GGML_ASSERT(root >= 0 && root < 3);

"""

_ROOT3_DATA_OLD = """        float * data0 = data_base[0] + chunk_start;
        float * data1 = data_base[1] + chunk_start;
        float * data2 = data_base[2] + chunk_start;
"""
_ROOT3_DATA_NEW = """        float * root_data  = data_base[root]  + chunk_start;
        float * leaf0_data = data_base[leaf0] + chunk_start;
        float * leaf1_data = data_base[leaf1] + chunk_start;
"""

_ROOT3_ROOT_OLD = """        // -- Root = rank 0. Every mapped pointer below is from consumer 0's
        // alias table. --
        ggml_cuda_set_device(p->devices[0]);
        auto * root_publish = reinterpret_cast<float4 *>(p->host_buf_dev3[0][0] + slot_offset);
        auto * root_leaf0   = reinterpret_cast<const float4 *>(p->host_buf_dev3[0][1] + slot_offset);
        auto * root_leaf1   = reinterpret_cast<const float4 *>(p->host_buf_dev3[0][2] + slot_offset);
        ggml_cuda_ar_kernel3<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[0]>>>(
            reinterpret_cast<const float4 *>(data0), reinterpret_cast<float4 *>(data0),
            root_publish, root_leaf0, root_leaf1, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, 0, slot, 1),
            ggml_cuda_ar_arrival_ptr3(p, 0, slot, 2),
            ggml_cuda_ar_arrival_ptr3(p, 0, slot, 0),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[0][slot].ker, streams[0]));
"""
_ROOT3_ROOT_NEW = """        // Root aliases are resolved for the selected consuming rank.
        ggml_cuda_set_device(p->devices[root]);
        auto * root_publish = reinterpret_cast<float4 *>(p->host_buf_dev3[root][root] + slot_offset);
        auto * root_leaf0   = reinterpret_cast<const float4 *>(p->host_buf_dev3[root][leaf0] + slot_offset);
        auto * root_leaf1   = reinterpret_cast<const float4 *>(p->host_buf_dev3[root][leaf1] + slot_offset);
        ggml_cuda_ar_kernel3<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[root]>>>(
            reinterpret_cast<const float4 *>(root_data), reinterpret_cast<float4 *>(root_data),
            root_publish, root_leaf0, root_leaf1, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, root, slot, leaf0),
            ggml_cuda_ar_arrival_ptr3(p, root, slot, leaf1),
            ggml_cuda_ar_arrival_ptr3(p, root, slot, root),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[root][slot].ker, streams[root]));
"""

_ROOT3_LEAF0_OLD = """        // -- Leaf = rank 1. Every mapped pointer below is from consumer 1's
        // alias table. --
        ggml_cuda_set_device(p->devices[1]);
        auto * leaf1_mine = reinterpret_cast<float4 *>(p->host_buf_dev3[1][1] + slot_offset);
        auto * leaf1_root = reinterpret_cast<const float4 *>(p->host_buf_dev3[1][0] + slot_offset);
        ggml_cuda_ar_kernel3_leaf<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[1]>>>(
            reinterpret_cast<const float4 *>(data1), reinterpret_cast<float4 *>(data1),
            leaf1_mine, leaf1_root, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, 1, slot, 1),
            ggml_cuda_ar_arrival_ptr3(p, 1, slot, 0),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[1][slot].ker, streams[1]));
"""
_ROOT3_LEAF0_NEW = """        ggml_cuda_set_device(p->devices[leaf0]);
        auto * leaf0_mine = reinterpret_cast<float4 *>(p->host_buf_dev3[leaf0][leaf0] + slot_offset);
        auto * leaf0_root = reinterpret_cast<const float4 *>(p->host_buf_dev3[leaf0][root] + slot_offset);
        ggml_cuda_ar_kernel3_leaf<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[leaf0]>>>(
            reinterpret_cast<const float4 *>(leaf0_data), reinterpret_cast<float4 *>(leaf0_data),
            leaf0_mine, leaf0_root, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, leaf0, slot, leaf0),
            ggml_cuda_ar_arrival_ptr3(p, leaf0, slot, root),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[leaf0][slot].ker, streams[leaf0]));
"""

_ROOT3_LEAF1_OLD = """        // -- Leaf = rank 2. Every mapped pointer below is from consumer 2's
        // alias table. --
        ggml_cuda_set_device(p->devices[2]);
        auto * leaf2_mine = reinterpret_cast<float4 *>(p->host_buf_dev3[2][2] + slot_offset);
        auto * leaf2_root = reinterpret_cast<const float4 *>(p->host_buf_dev3[2][0] + slot_offset);
        ggml_cuda_ar_kernel3_leaf<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[2]>>>(
            reinterpret_cast<const float4 *>(data2), reinterpret_cast<float4 *>(data2),
            leaf2_mine, leaf2_root, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, 2, slot, 2),
            ggml_cuda_ar_arrival_ptr3(p, 2, slot, 0),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[2][slot].ker, streams[2]));
"""
_ROOT3_LEAF1_NEW = """        ggml_cuda_set_device(p->devices[leaf1]);
        auto * leaf1_mine = reinterpret_cast<float4 *>(p->host_buf_dev3[leaf1][leaf1] + slot_offset);
        auto * leaf1_root = reinterpret_cast<const float4 *>(p->host_buf_dev3[leaf1][root] + slot_offset);
        ggml_cuda_ar_kernel3_leaf<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[leaf1]>>>(
            reinterpret_cast<const float4 *>(leaf1_data), reinterpret_cast<float4 *>(leaf1_data),
            leaf1_mine, leaf1_root, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, leaf1, slot, leaf1),
            ggml_cuda_ar_arrival_ptr3(p, leaf1, slot, root),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[leaf1][slot].ker, streams[leaf1]));
"""

_HYBRID_WARN_OLD = """        GGML_LOG_WARN("hybrid: internal AllReduce init failed (n_devices != 2?); "
                      "hybrid dispatch will use %s only\\n", have_nccl ? "rccl" : "meta");
"""
_HYBRID_WARN_NEW = """        // PGC10: N=3 internal root3 is valid; init failure is not a device-count diagnostic.
        GGML_LOG_WARN("hybrid: internal AllReduce init failed; "
                      "hybrid dispatch will use %s only\\n", have_nccl ? "rccl" : "meta");
"""

ALLREDUCE = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cu",
    language="none",
    description="Generalize 1244 root3 rank selection while preserving its transport and synchronization protocol.",
    edits=(
        Edit(
            id="adaptive-nway-root-field",
            anchor=_re.escape(_ROOT3_FIELDS_OLD),
            mode="replace",
            text=_ROOT3_FIELDS_NEW,
            guard=r"\broot3_root\s*=\s*0;",
            expect_matches=1,
            rationale="Store the N=3 root rank in the pipeline so the environment is read only at init.",
        ),
        Edit(
            id="adaptive-nway-root-env-parser",
            anchor=_re.escape(_ENV_U64_ANCHOR),
            mode="insert_after",
            text=_ROOT3_ENV_HELPER,
            guard=r"static int ggml_cuda_ar_root3_root_from_env\(\)",
            expect_matches=1,
            rationale="Parse the closed 0|1|2 root-rank domain once, defaulting invalid values to rank 0 with a warning.",
        ),
        Edit(
            id="adaptive-nway-root-init",
            anchor=_re.escape(_ROOT3_INIT_OLD),
            mode="replace",
            text=_ROOT3_INIT_NEW,
            guard=r"p->root3_root = ggml_cuda_ar_root3_root_from_env\(\);",
            expect_matches=1,
            rationale="Read BIGCHERRY_AR_ROOT3_ROOT once when the N=3 internal pipeline is initialized.",
        ),
        Edit(
            id="adaptive-nway-root-indices",
            anchor=_re.escape(_ROOT3_INDEX_ANCHOR),
            mode="insert_after",
            text=_ROOT3_INDEX_SETUP,
            guard=r"const int root\s*=\s*p->root3_root;",
            expect_matches=1,
            rationale="Derive the two leaves from the selected root once per AllReduce call, not per chunk.",
        ),
        Edit(
            id="adaptive-nway-root-data",
            anchor=_re.escape(_ROOT3_DATA_OLD),
            mode="replace",
            text=_ROOT3_DATA_NEW,
            guard=r"float \* root_data\s*=\s*data_base\[root\]",
            expect_matches=1,
            rationale="Address tensor chunks by logical root/leaf rank rather than fixed ranks 0/1/2.",
        ),
        Edit(
            id="adaptive-nway-root-launch",
            anchor=_re.escape(_ROOT3_ROOT_OLD),
            mode="replace",
            text=_ROOT3_ROOT_NEW,
            guard=r"p->host_buf_dev3\[root\]\[root\]",
            expect_matches=1,
            rationale="Launch the existing 1244 root kernel on the selected rank using that consumer's alias table and arrival slots.",
            max_span_lines=24,
        ),
        Edit(
            id="adaptive-nway-leaf0-launch",
            anchor=_re.escape(_ROOT3_LEAF0_OLD),
            mode="replace",
            text=_ROOT3_LEAF0_NEW,
            guard=r"p->host_buf_dev3\[leaf0\]\[leaf0\]",
            expect_matches=1,
            rationale="Generalize the first leaf launch to the rank derived from the selected root.",
            max_span_lines=24,
        ),
        Edit(
            id="adaptive-nway-leaf1-launch",
            anchor=_re.escape(_ROOT3_LEAF1_OLD),
            mode="replace",
            text=_ROOT3_LEAF1_NEW,
            guard=r"p->host_buf_dev3\[leaf1\]\[leaf1\]",
            expect_matches=1,
            rationale="Generalize the second leaf launch to the remaining rank.",
            max_span_lines=24,
        ),
    ),
)

ADAPTIVE = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    language="none",
    description="Remove 0840's obsolete two-device-only diagnostic after composing with 1244.",
    edits=(
        Edit(
            id="adaptive-nway-init-warning",
            anchor=_re.escape(_HYBRID_WARN_OLD),
            mode="replace",
            text=_HYBRID_WARN_NEW,
            guard=r"PGC10: N=3 internal root3 is valid; init failure is not a device-count diagnostic\.",
            expect_matches=1,
            rationale="1244 admits N=3, so an internal-init failure is no longer evidence that the device count is not two.",
            max_span_lines=3,
        ),
    ),
)

PATCHES = [ALLREDUCE, ADAPTIVE]
