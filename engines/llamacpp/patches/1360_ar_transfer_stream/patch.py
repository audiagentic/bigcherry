"""1360 (QFP49): issue RCCL AllReduce on per-device transfer streams.

Default off. With BIGCHERRY_AR_TRANSFER_STREAM=1, the existing NCCL/RCCL
AllReduce is split into begin/end helpers. Pre-work stays on each device's
compute stream, transfer work waits on a recorded input event and runs on a
dedicated non-blocking stream, and the compute stream waits on an output event
before post-work/consumers. The public entry point calls begin/end back to back;
later QFP49 work can interleave independent compute between them.

Transfer streams and two event slots per device are owned by the communication
context and created lazily once. No host thread is added.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "evaluated"

_A_CTX_NCCL = "    std::vector<ncclComm_t>     comms;\n"
_N_CTX_NCCL = r"""    std::vector<ncclComm_t>     comms;

    // BigCherry 1360 (QFP49): one transfer stream per rank and two reusable event slots so a later paired
    // scheduler can have A and B reductions outstanding without creating synchronization objects per call.
    std::vector<cudaStream_t>   bc_ar_transfer_streams;
    std::vector<cudaEvent_t>    bc_ar_transfer_in[2];
    std::vector<cudaEvent_t>    bc_ar_transfer_out[2];
    bool                        bc_ar_transfer_busy[2] = { false, false };
    size_t                      bc_ar_transfer_next_slot = 0;
"""

_A_CTX_DTOR = """    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
"""
_N_CTX_DTOR = r"""    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        // BigCherry 1360: synchronization objects are context-owned, never per-call.
        for (size_t i = 0; i < bc_ar_transfer_streams.size(); ++i) {
            ggml_cuda_set_device(dev_ids[i]);
            for (int slot = 0; slot < 2; ++slot) {
                CUDA_CHECK(cudaEventDestroy(bc_ar_transfer_in[slot][i]));
                CUDA_CHECK(cudaEventDestroy(bc_ar_transfer_out[slot][i]));
            }
            CUDA_CHECK(cudaStreamDestroy(bc_ar_transfer_streams[i]));
        }
"""

_A_NCCL_FN = """static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
"""

_HELPERS = r"""// BigCherry 1360 (QFP49): state retained from begin() until end(). Pool allocations stay alive until
// the compute-stream post-work has been enqueued, matching the lifetime of the stock monolithic path.
struct bc_ar_transfer_state {
    ggml_backend_cuda_comm_context * comm_ctx = nullptr;
    // the state's own copy: the caller's array need not outlive begin() (a paired caller ends a sum later)
    struct ggml_tensor * tensors[GGML_CUDA_MAX_DEVICES] = {};
    int64_t ne = 0;
    size_t n_backends = 0;
    bool small = true;
    int slot = -1;
    cudaStream_t transfer[GGML_CUDA_MAX_DEVICES] = {};
    cudaEvent_t e_in[GGML_CUDA_MAX_DEVICES] = {};
    cudaEvent_t e_out[GGML_CUDA_MAX_DEVICES] = {};
    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
};

static void bc_1360_ar_transfer_resources(ggml_backend_cuda_comm_context * comm_ctx) {
    const size_t n = comm_ctx->backends.size();
    if (comm_ctx->bc_ar_transfer_streams.size() == n) {
        return;
    }

    GGML_ASSERT(comm_ctx->bc_ar_transfer_streams.empty());
    GGML_ASSERT(comm_ctx->bc_ar_transfer_in[0].empty() && comm_ctx->bc_ar_transfer_in[1].empty());
    GGML_ASSERT(comm_ctx->bc_ar_transfer_out[0].empty() && comm_ctx->bc_ar_transfer_out[1].empty());

    comm_ctx->bc_ar_transfer_streams.resize(n);
    for (int slot = 0; slot < 2; ++slot) {
        comm_ctx->bc_ar_transfer_in[slot].resize(n);
        comm_ctx->bc_ar_transfer_out[slot].resize(n);
    }

    for (size_t i = 0; i < n; ++i) {
        ggml_backend_cuda_context * cuda_ctx =
            (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        ggml_cuda_set_device(cuda_ctx->device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&comm_ctx->bc_ar_transfer_streams[i], cudaStreamNonBlocking));
        for (int slot = 0; slot < 2; ++slot) {
            CUDA_CHECK(cudaEventCreateWithFlags(&comm_ctx->bc_ar_transfer_in[slot][i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&comm_ctx->bc_ar_transfer_out[slot][i], cudaEventDisableTiming));
        }
    }
}

static int bc_1360_ar_transfer_acquire_slot(ggml_backend_cuda_comm_context * comm_ctx) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        const int slot = (int) ((comm_ctx->bc_ar_transfer_next_slot + attempt) & 1u);
        if (!comm_ctx->bc_ar_transfer_busy[slot]) {
            comm_ctx->bc_ar_transfer_busy[slot] = true;
            comm_ctx->bc_ar_transfer_next_slot = (size_t) (slot ^ 1);
            return slot;
        }
    }
    GGML_ABORT("BigCherry 1360: more than two AllReduce transfers are outstanding");
}

static bool bc_1360_ar_transfer_begin(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors, bc_ar_transfer_state & state) {
    state.comm_ctx = comm_ctx;
    state.ne = ggml_nelements(tensors[0]);
    if (state.ne == 0) {
        return true;
    }

    state.n_backends = comm_ctx->backends.size();
    for (size_t i = 0; i < state.n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == state.ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
        state.tensors[i] = tensors[i];
    }

    state.small =
        (state.n_backends <= 2 && state.ne < 32768) ||
        (state.n_backends == 3 && state.ne < 131072) ||
        (state.n_backends >= 4 && state.ne < 262144);

    bc_1360_ar_transfer_resources(comm_ctx);
    state.slot = bc_1360_ar_transfer_acquire_slot(comm_ctx);

    to_bf16_cuda_t to_bf16 = nullptr;
    if (!state.small) {
        to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    }

    for (size_t i = 0; i < state.n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx =
            (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        ggml_cuda_set_device(cuda_ctx->device);

        state.transfer[i] = comm_ctx->bc_ar_transfer_streams[i];
        state.e_in[i] = comm_ctx->bc_ar_transfer_in[state.slot][i];
        state.e_out[i] = comm_ctx->bc_ar_transfer_out[state.slot][i];

        if (state.small) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        } else {
            state.tmp[i].pool = &cuda_ctx->pool();
            state.tmp[i].alloc(state.ne);
            if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
                to_bf16(tensors[i]->data, state.tmp[i].get(), state.ne, cuda_ctx->stream());
            } else {
                CUDA_CHECK(cudaMemsetAsync(
                    state.tmp[i].get(), 0, state.ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
            }
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaEventRecord(state.e_in[i], cuda_ctx->stream()));
        CUDA_CHECK(cudaStreamWaitEvent(state.transfer[i], state.e_in[i], 0));
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < state.n_backends; ++i) {
        void * data = state.small ? tensors[i]->data : (void *) state.tmp[i].get();
        NCCL_CHECK(ncclAllReduce(
            data,
            data,
            state.ne,
            state.small ? ncclFloat : ncclBfloat16,
            ncclSum,
            comm_ctx->comms[i],
            state.transfer[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < state.n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx =
            (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        ggml_cuda_set_device(cuda_ctx->device);
        CUDA_CHECK(cudaEventRecord(state.e_out[i], state.transfer[i]));
    }

    return true;
}

static void bc_1360_ar_transfer_end(bc_ar_transfer_state & state) {
    if (state.ne == 0) {
        return;
    }

    to_fp32_cuda_t to_fp32 = nullptr;
    if (!state.small) {
        to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
    }

    for (size_t i = 0; i < state.n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx =
            (ggml_backend_cuda_context *) state.comm_ctx->backends[i]->context;
        ggml_cuda_set_device(cuda_ctx->device);
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), state.e_out[i], 0));
        if (!state.small) {
            to_fp32(state.tmp[i].get(), (float *) state.tensors[i]->data, state.ne, cuda_ctx->stream());
            CUDA_CHECK(cudaGetLastError());
        }
    }

    GGML_ASSERT(state.slot >= 0);
    state.comm_ctx->bc_ar_transfer_busy[state.slot] = false;
    state.slot = -1;
}

"""

_FAST_PATH = r"""    // BigCherry 1360 (QFP49): default off; the stock body below remains byte-for-byte the flag-off path.
    static const bool bc_1360_on = [] {
        const char * env = getenv("BIGCHERRY_AR_TRANSFER_STREAM");
        return env != nullptr && atoi(env) != 0;
    }();
    if (bc_1360_on && ggml_nelements(tensors[0]) != 0) {
        static std::once_flag bc_1360_logged;
        std::call_once(bc_1360_logged, [] {
            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1360_ar_transfer_stream\n");
        });
        bc_ar_transfer_state state;
        if (!bc_1360_ar_transfer_begin(comm_ctx, tensors, state)) {
            return false;
        }
        bc_1360_ar_transfer_end(state);
        return true;
    }

"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1360: run NCCL/RCCL exchange on context-owned transfer streams fenced by compute-stream events",
        language="none",
        edits=(
            Edit(
                id="ar-transfer-context-resources",
                anchor=re.escape(_A_CTX_NCCL),
                mode="replace",
                text=_N_CTX_NCCL,
                guard=r"bc_ar_transfer_streams;",
                rationale="NCCL communicator storage in the communication context.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="ar-transfer-context-free",
                anchor=re.escape(_A_CTX_DTOR),
                mode="replace",
                text=_N_CTX_DTOR,
                guard=r"cudaStreamDestroy\(bc_ar_transfer_streams\[i\]\)",
                rationale="Communication-context destructor before it destroys NCCL communicators.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="ar-transfer-helpers",
                anchor=re.escape(_A_NCCL_FN),
                mode="insert_before",
                text=_HELPERS,
                guard=r"struct bc_ar_transfer_state \{",
                rationale="Immediately before the NCCL AllReduce implementation the helpers split.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="ar-transfer-entry",
                anchor=re.escape(_A_NCCL_FN),
                mode="insert_after",
                text=_FAST_PATH,
                guard=r"BIGCHERRY_PATCH_HIT patch=1360_ar_transfer_stream",
                rationale="At the NCCL entry point so flag-off falls through to the existing body unchanged.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_AR_TRANSFER_STREAM",
        "0|1",
        "0 (off)",
        "run NCCL/RCCL AllReduce exchange on a per-device transfer stream fenced by compute-stream events; "
        "the current entry point calls begin/end back to back",
    ),
)
