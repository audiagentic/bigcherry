"""HI58 - observe actual SPLIT_REDUCE provider and meta handoff."""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch


CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="record provider outcome at the existing allreduce boundary",
    edits=(
        Edit(
            id="reduce-telemetry-include",
            anchor=r'^#include "ggml-cuda/allreduce\.cuh"$',
            rationale="the CUDA communication implementation include block",
            text='\n#include "ggml-cuda/hip-autotune-reduce-telemetry.h"',
            guard=r'hip-autotune-reduce-telemetry\.h',
        ),
        Edit(
            id="reduce-telemetry-context-fields",
            anchor=r'^    try_allreduce_fn            try_allreduce = nullptr;$',
            rationale="retain the already-selected provider name for telemetry",
            text=(
                '\n    try_allreduce_fn            try_allreduce = nullptr;\n'
                '    const char *                provider_name = "unknown";'
            ),
            guard=r'const char \*                provider_name',
        ),
        Edit(
            id="reduce-telemetry-provider-names",
            anchor=r'^static void ggml_backend_cuda_comm_init_none\(ggml_backend_cuda_comm_context \* ret\) \{$',
            rationale="name the existing selected providers without changing selection",
            text=(
                '\nstatic void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {\n'
                '    ret->provider_name = "meta";'
            ),
            guard=r'ret->provider_name = "meta"',
        ),
        Edit(
            id="reduce-telemetry-internal-name",
            anchor=r'^        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;$',
            rationale="label the existing internal provider after successful init",
            text=(
                '\n        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;\n'
                '        ret->provider_name = "internal";'
            ),
            guard=r'ret->provider_name = "internal"',
        ),
        Edit(
            id="reduce-telemetry-nccl-name",
            anchor=r'^        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;$',
            rationale="label the existing RCCL/NCCL provider after successful init",
            text=(
                '\n        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;\n'
                '        ret->provider_name = "rccl";'
            ),
            guard=r'ret->provider_name = "rccl"',
        ),
        Edit(
            id="reduce-telemetry-call",
            anchor=r'^    return comm_ctx->try_allreduce\(comm_ctx, tensors\);$',
            rationale="observe provider success or decline without changing the return value",
            mode="replace",
            text=(
                '    const bool provider_succeeded = comm_ctx->try_allreduce(comm_ctx, tensors);\n'
                '#ifdef GGML_HIP_DISPATCH\n'
                '    ggml_hip_reduce_telemetry_provider(comm_ctx->dev_ids.data(),\n'
                '        comm_ctx->dev_ids.size(), tensors, comm_ctx->provider_name,\n'
                '        provider_succeeded);\n'
                '#endif\n'
                '    return provider_succeeded;'
            ),
            guard=r'ggml_hip_reduce_telemetry_provider\(',
        ),
        Edit(
            id="reduce-telemetry-context-snapshot",
            anchor=r'^\};$\n\n#ifdef GGML_USE_NCCL',
            rationale="expose selected provider and real device ordinals to the opaque meta context",
            mode="replace",
            text=(
                '};\n\n'
                '#ifdef GGML_HIP_DISPATCH\n'
                'bool ggml_hip_reduce_telemetry_context_snapshot(\n'
                '        void * comm_ctx, const int ** devices, size_t * device_count,\n'
                '        const char ** requested_provider) {\n'
                '    if (comm_ctx == nullptr || devices == nullptr || device_count == nullptr ||\n'
                '            requested_provider == nullptr) {\n'
                '        return false;\n'
                '    }\n'
                '    auto * ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx);\n'
                '    *devices = ctx->dev_ids.data();\n'
                '    *device_count = ctx->dev_ids.size();\n'
                '    *requested_provider = ctx->provider_name;\n'
                '    return *device_count >= 2 && *requested_provider != nullptr;\n'
                '}\n'
                '#endif\n\n'
                '#ifdef GGML_USE_NCCL'
            ),
            guard=r'ggml_hip_reduce_telemetry_context_snapshot\(',
        ),
    ),
)


META = FilePatch(
    path="ggml/src/ggml-backend-meta.cpp",
    description="record actual meta butterfly fallback at its execution point",
    edits=(
        Edit(
            id="meta-reduce-telemetry-include",
            anchor=r'^#include "ggml-backend-impl\.h"$',
            rationale="meta backend include block",
            text='\n#include "ggml-cuda/hip-autotune-reduce-telemetry.h"',
            guard=r'hip-autotune-reduce-telemetry\.h',
        ),
        Edit(
            id="meta-reduce-telemetry-node-scope",
            anchor=r'^            bool backend_allreduce_success = false;$',
            rationale="keep the reduction tensors available through fallback completion",
            text=(
                '\n            std::vector<ggml_tensor *> nodes;\n'
                '            bool backend_allreduce_success = false;'
            ),
            guard=r'^            std::vector<ggml_tensor \*> nodes;$',
        ),
        Edit(
            id="meta-reduce-telemetry-node-declaration",
            anchor=r'^                std::vector<ggml_tensor \*> nodes;$',
            rationale="use the shared node vector for provider and fallback telemetry",
            mode="replace",
            text="                // nodes is scoped around the complete provider/fallback path.",
            guard=r'nodes is scoped around the complete provider/fallback path',
        ),
        Edit(
            id="meta-reduce-telemetry-fallback",
            anchor=r'^                const ggml_status status = allreduce_fallback\(i\);$'
                   r'\n^                if \(status != GGML_STATUS_SUCCESS\) \{$',
            rationale="the exact point where the generic fallback has been selected",
            text=(
                '\n                const ggml_status status = allreduce_fallback(i);\n'
                '#ifdef GGML_HIP_DISPATCH\n'
                '                if (backend_ctx->comm_ctx) {\n'
                '                    ggml_hip_reduce_telemetry_fallback_context(\n'
                '                        backend_ctx->comm_ctx, nodes.data(),\n'
                '                        "provider_declined_handoff_meta", 1);\n'
                '                }\n'
                '#endif\n'
                '                if (status != GGML_STATUS_SUCCESS) {'
            ),
            guard=r'ggml_hip_reduce_telemetry_fallback\(',
        ),
    ),
)


PATCHES = [CUDA, META]
