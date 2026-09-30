"""0840: size-adaptive internal/RCCL AllReduce dispatch.

Adaptive initializes RCCL and the stock internal host pipeline together.
`--allreduce-switch-bytes` is owned by 0860: reductions strictly below the
configured byte count prefer host, while reductions at or above it prefer
RCCL. If the preferred provider is unavailable or host rejects a small call,
the other provider is tried before meta fallback.

0840 owns the minimal provider_name field required by adaptive dispatch; the
0830 telemetry package may observe it when present but is not a runtime
dependency. Adaptive never mutates the internal pipeline's wire state. When
composed with 1272_ar_host_compressed_wire, GGML_CUDA_AR_WIRE therefore
selects the exact same host codec/path as the plain internal provider.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="add adaptive provider with CLI-sized host/RCCL per-call dispatch",
    edits=(
        Edit(
            id="hybrid-provider-context-field",
            anchor=r'^    try_allreduce_fn            try_allreduce = nullptr;$',
            rationale="adaptive records the provider selected per call without depending on 0830 telemetry",
            mode="replace",
            text=(
                '    try_allreduce_fn            try_allreduce = nullptr;\n'
                '    const char *                provider_name = "unknown";'
            ),
            guard=r'const char \*                provider_name',
            expect_matches=1,
        ),
        Edit(
            id="hybrid-try-allreduce",
            anchor=(
                r"static bool ggml_backend_cuda_comm_try_allreduce_internal\(\n"
                r"        ggml_backend_cuda_comm_context \* comm_ctx, struct ggml_tensor \*\* tensors\) \{\n"
                r"    return ggml_backend_cuda_comm_allreduce_internal\(comm_ctx, tensors\);\n"
                r"\}"
            ),
            rationale="insert the adaptive dispatcher beside the plain internal wrapper",
            mode="insert_after",
            text=(
                "\n\n"
                "static bool ggml_backend_cuda_comm_try_allreduce_hybrid(\n"
                "        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {\n"
                "    const size_t reduction_bytes = tensors != nullptr && tensors[0] != nullptr\n"
                "        ? ggml_nbytes(tensors[0]) : 0;\n"
                "    const size_t switch_bytes = g_ggml_backend_cuda_comm_config.switch_bytes;\n"
                "    const bool have_internal = comm_ctx->ar_pipeline != nullptr;\n"
                "#ifdef GGML_USE_NCCL\n"
                "    const bool have_rccl = !comm_ctx->comms.empty();\n"
                "#else\n"
                "    const bool have_rccl = false;\n"
                "#endif\n"
                "    const bool prefer_internal = have_internal && (reduction_bytes < switch_bytes || !have_rccl);\n"
                "    if (prefer_internal) {\n"
                "        comm_ctx->provider_name = \"internal\";\n"
                "        if (ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors)) {\n"
                "            return true;\n"
                "        }\n"
                "    }\n"
                "#ifdef GGML_USE_NCCL\n"
                "    if (have_rccl) {\n"
                "        comm_ctx->provider_name = \"rccl\";\n"
                "        return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);\n"
                "    }\n"
                "#endif // GGML_USE_NCCL\n"
                "    if (have_internal && !prefer_internal) {\n"
                "        comm_ctx->provider_name = \"internal\";\n"
                "        return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);\n"
                "    }\n"
                "    return false;\n"
                "}"
            ),
            guard=r"ggml_backend_cuda_comm_try_allreduce_hybrid\(",
            expect_matches=1,
        ),
        Edit(
            id="hybrid-init",
            anchor=r"static void ggml_backend_cuda_comm_init_nccl\(ggml_backend_cuda_comm_context \* ret\) \{",
            rationale="adaptive initialization is independent of the stock greedy provider chain",
            mode="insert_before",
            text=(
                "static void ggml_backend_cuda_comm_init_hybrid(ggml_backend_cuda_comm_context * ret) {\n"
                "    bool have_nccl = false;\n"
                "#ifdef GGML_USE_NCCL\n"
                "    const ggml_cuda_device_info & info = ggml_cuda_info();\n"
                "    if (info.device_count <= info.physical_device_count &&\n"
                "            ggml_backend_cuda_comm_rccl_admission_ok(ret->dev_ids.data(), ret->dev_ids.size())) {\n"
                "        const size_t n = ret->dev_ids.size();\n"
                "        ret->comms.resize(n);\n"
                "        ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());\n"
                "        if (rc == ncclSuccess) {\n"
                "            have_nccl = true;\n"
                "        } else {\n"
                "            ret->comms.clear();\n"
                "            GGML_LOG_WARN(\"hybrid: NCCL init failed (%s); hybrid dispatch will use \"\n"
                "                          \"internal only\\n\", ncclGetErrorString(rc));\n"
                "        }\n"
                "    } else if (info.device_count > info.physical_device_count) {\n"
                "        GGML_LOG_WARN(\"hybrid: NCCL disabled (virtual devices in use); hybrid \"\n"
                "                      \"dispatch will use internal only\\n\");\n"
                "    } else {\n"
                "        GGML_LOG_WARN(\"hybrid: NCCL disabled (RCCL admission check failed); hybrid \"\n"
                "                      \"dispatch will use internal only\\n\");\n"
                "    }\n"
                "#endif // GGML_USE_NCCL\n"
                "    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());\n"
                "    const bool have_internal = ret->ar_pipeline != nullptr;\n"
                "    // Leave the internal pipeline untouched: 1272/GGML_CUDA_AR_WIRE and\n"
                "    // pristine host wire selection must match the plain host provider.\n"
                "    if (!have_internal) {\n"
                "        (void) cudaGetLastError();\n"
                "        GGML_LOG_WARN(\"hybrid: internal AllReduce init failed (n_devices != 2?); \"\n"
                "                      \"hybrid dispatch will use %s only\\n\", have_nccl ? \"rccl\" : \"meta\");\n"
                "    }\n"
                "    if (have_internal || have_nccl) {\n"
                "        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_hybrid;\n"
                "        ret->provider_name = have_internal ? \"internal\" : \"rccl\";\n"
                "        return;\n"
                "    }\n"
                "    ggml_backend_cuda_comm_init_none(ret);\n"
                "}\n\n"
            ),
            guard=r"ggml_backend_cuda_comm_init_hybrid\(ggml_backend_cuda_comm_context \* ret\) \{",
            expect_matches=1,
        ),
    ),
)

CUDA_PROVIDER = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    language="none",
    description="register adaptive on 0860's explicit provider seam",
    edits=(
        Edit(
            id="adaptive-provider-available",
            anchor=r'    if \(p == \"adaptive\" \|\| p == \"p2p\" \|\| p == \"root3\"\) \{\n',
            rationale="0840 provides adaptive, so remove only adaptive from 0860's unavailable list",
            mode="replace",
            text='    if (p == "p2p" || p == "root3") {\n',
            guard=r'    if \(p == \"p2p\" \|\| p == \"root3\"\) \{\n',
            expect_matches=1,
        ),
        Edit(
            id="adaptive-provider-init",
            anchor=(
                r'    \} else if \(provider == \"butterfly\"\) \{\n'
                r'        ggml_backend_cuda_comm_init_none\(ret\);\n'
                r'    \} else \{\n'
            ),
            rationale="select adaptive beside 0860's implemented providers",
            mode="replace",
            text=(
                '    } else if (provider == "butterfly") {\n'
                '        ggml_backend_cuda_comm_init_none(ret);\n'
                '    } else if (provider == "adaptive") {\n'
                '        ggml_backend_cuda_comm_init_hybrid(ret);\n'
                '    } else {\n'
            ),
            guard=r'provider == \"adaptive\"\) \{\n        ggml_backend_cuda_comm_init_hybrid',
            expect_matches=1,
        ),
    ),
)

PATCHES = [CUDA, CUDA_PROVIDER]
